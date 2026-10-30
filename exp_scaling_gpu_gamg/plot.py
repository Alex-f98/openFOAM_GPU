#!/usr/bin/env python3
"""Grafica tiempo de solve vs tamano de malla (log-log).

Uso:  plot.py OUT.png [archivo.csv[:etiqueta] ...]

Cada CSV debe tener columnas: nx,ncells,step_med_s,step_per_cell_s,status
Genera dos paneles: tiempo por paso de tiempo y tiempo por celda por paso.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[1]
specs = sys.argv[2:] or ['/openfoamGPU/exp_scaling_gpu_gamg/results_gpu_gamg.csv:GPU']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

for spec in specs:
    path, _, label = spec.partition(':')
    label = label or os.path.basename(path)
    if not os.path.exists(path):
        print('skip (no existe):', path)
        continue
    rows = [r for r in csv.DictReader(open(path)) if r.get('status') == 'ok']
    if not rows:
        print('skip (sin filas ok):', path)
        continue
    x = [int(r['ncells']) for r in rows]
    y1 = [float(r['step_med_s']) for r in rows]
    y2 = [float(r['step_per_cell_s']) for r in rows]
    ax1.plot(x, y1, 'o-', label=label)
    ax2.plot(x, y2, 'o-', label=label)

ax1.set_xlabel('celdas')
ax1.set_ylabel('tiempo por paso de tiempo [s]')
ax1.set_title('Coste por paso de tiempo vs tamano de malla')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, which='both', alpha=0.3)
ax1.legend()

ax2.set_xlabel('celdas')
ax2.set_ylabel('s / celda / paso')
ax2.set_title('Coste normalizado (menor = mejor escalado)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, which='both', alpha=0.3)
ax2.legend()

fig.tight_layout()
fig.savefig(out, dpi=130)
print('escrito:', out)
