#!/usr/bin/env python3
"""Analiza el log de icoFoam + el CSV del monitor y escribe una fila de resultados.

Uso:
  analyze.py LOG MON CARD NCELLS WALL RC STEPS NX POOL OUTCSV
"""

import csv
import os
import re
import statistics
import sys

LOG, MON, CARD, NCELLS, WALL, RC, STEPS, NX, POOL, OUTCSV = sys.argv[1:11]
NCELLS = int(NCELLS)
WALL = float(WALL)
RC = int(RC)
STEPS = int(STEPS)
NX = int(NX)

text = open(LOG, errors='replace').read()

iters = len(re.findall(r'^Time = ', text, flags=re.M))
execs = [float(m.group(1)) for m in
         re.finditer(r'^ExecutionTime = ([0-9.eE+-]+) s', text, flags=re.M)]

# Un paso esta COMPLETADO cuando icoFoam imprime su ExecutionTime; la linea
# "Time = " solo indica que el paso ha comenzado. Usar las lineas "Time = "
# sobrestimaba los pasos completados (p.ej. 1/20 cuando el paso 1 ni termino).
completed = len(execs)

deltas = []
for i in range(1, len(execs)):
    deltas.append(execs[i] - execs[i - 1])
if deltas and deltas[0] <= 0:
    deltas = deltas[1:]

step_first = deltas[0] if deltas else float('nan')
rest = deltas[1:] if len(deltas) > 1 else deltas
step_med = statistics.median(rest) if rest else float('nan')
exec_last = execs[-1] if execs else float('nan')
per_cell = step_med / NCELLS if NCELLS else float('nan')

# --- monitor ---
busy, vram = [], []
if os.path.exists(MON):
    with open(MON) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) != 4 or p[1] != CARD:
                continue
            try:
                b = float(p[2])
            except ValueError:
                continue
            busy.append(b)
            try:
                vram.append(float(p[3]))
            except ValueError:
                pass

gfx_mean = statistics.mean(busy) if busy else float('nan')
gfx_max = max(busy) if busy else float('nan')
vram_peak = max(vram) if vram else float('nan')

status = 'ok' if (RC == 0 and completed >= STEPS) else ('timeout' if completed < STEPS else 'fail')

row = [NX, NCELLS, POOL, status, completed, f'{WALL:.2f}', f'{exec_last:.3f}',
       f'{step_first:.4f}', f'{step_med:.4f}', f'{per_cell:.8f}',
       f'{gfx_mean:.1f}', f'{gfx_max:.1f}', f'{vram_peak:.0f}',
       'gfx_runtime' if any('memory pool' in l for l in text.splitlines()) else '']

hdr = ['nx', 'ncells', 'pool', 'status', 'iters_done', 'wall_s', 'exec_last_s',
       'step_first_s', 'step_med_s', 'step_per_cell_s',
       'gfx_mean_pct', 'gfx_max_pct', 'vram_peak_mb', 'notes']

new = not os.path.exists(OUTCSV)
with open(OUTCSV, 'a', newline='') as f:
    w = csv.writer(f)
    if new:
        w.writerow(hdr)
    w.writerow(row)

print('--- RESULT ---')
print(f'nx={NX} ncells={NCELLS} pool={POOL} status={status} steps={completed}/{STEPS} rc={RC}')
print(f'wall={WALL:.1f}s  exec_last={exec_last:.1f}s  first_step={step_first:.2f}s')
print(f'steady step (median)={step_med:.2f}s  per_cell={per_cell:.8f}s')
print(f'gfx mean/max={gfx_mean:.1f}/{gfx_max:.1f}%  vram_peak={vram_peak:.0f}MB')
