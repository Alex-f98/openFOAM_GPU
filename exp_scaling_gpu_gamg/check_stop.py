#!/usr/bin/env python3
"""Decide si continuar la escalera de mallas tras cada corrida.

Imprime 'CONTINUE' o 'STOP: <razon>'.
Reglas:
  - parar si la ultima corrida no completo los pasos (status != ok)
  - parar si el coste por celda por iteracion empeora > 3x respecto al
    tamano anterior (indicio de thrashing de memoria / paginacion)
  - parar si el uso medio de GPU es ~0% en el tamano mas grande
"""

import csv
import os
import sys

path = sys.argv[1]
if not os.path.exists(path):
    print('STOP: sin resultados (no existe el CSV)')
    sys.exit()
rows = list(csv.DictReader(open(path)))
if not rows:
    print('STOP: sin resultados')
    sys.exit()

last = rows[-1]

if last['status'] != 'ok':
    print(f"STOP: ultima corrida status={last['status']} "
          f"({last['iters_done']} pasos de los pedidos)")
    sys.exit()

if len(rows) >= 2:
    prev = float(rows[-2]['step_per_cell_s'])
    cur = float(last['step_per_cell_s'])
    if prev > 0 and cur > 3.0 * prev:
        print(f'STOP: coste por celda empeoro {cur/prev:.1f}x '
              f'({prev:.3e} -> {cur:.3e} s/celda/paso)')
        sys.exit()

if float(last['gfx_mean_pct']) < 1.0 and float(last['nx']) >= 64:
    print(f"STOP: GPU practicamente inactiva (gfx medio {last['gfx_mean_pct']}%) "
          f"en {last['nx']}^3")
    sys.exit()

print('CONTINUE')
