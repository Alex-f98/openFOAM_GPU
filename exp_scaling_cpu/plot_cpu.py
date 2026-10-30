#!/usr/bin/env python3
"""Grafico CPU (tiempo vs nucleos) con la referencia GPU superpuesta.

Uso:
  plot_cpu.py OUT.png CPU_CSV [GPU_STEP_S] [GPU_LABEL]

Una linea por tamano de malla. La referencia GPU se dibuja como linea roja
horizontal (mismo caso, 32^3).
"""

import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[1]
cpu_csv = sys.argv[2]
gpu_step = float(sys.argv[3]) if len(sys.argv) > 3 else None
gpu_label = sys.argv[4] if len(sys.argv) > 4 else 'GPU 0 (32³, sin pool)'

rows = [r for r in csv.DictReader(open(cpu_csv)) if r.get('status') == 'ok']
if not rows:
    sys.exit(f'sin filas ok en {cpu_csv}')

by_nx = {}
for r in rows:
    by_nx.setdefault(int(r['nx']), []).append(r)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
colors = {32: 'tab:green', 64: 'tab:blue', 128: 'tab:purple'}

for nx in sorted(by_nx):
    rs = sorted(by_nx[nx], key=lambda r: int(r['nprocs']))
    procs = [int(r['nprocs']) for r in rs]
    step = [float(r['step_med_s']) for r in rs]
    c = colors.get(nx, None)
    ax1.plot(procs, step, 'o-', color=c, label=f'{nx}³ ({nx**3:,} celdas)')
    base = step[procs.index(1)] if 1 in procs else step[0]
    ax2.plot(procs, [base / s for s in step], 'o-', color=c, label=f'{nx}³')

ax1.set_xlabel('núcleos MPI')
ax1.set_ylabel('tiempo por paso de tiempo [s]')
ax1.set_title('CPU: coste por paso vs núcleos')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, which='both', alpha=0.3)

ax2.plot([1, 64], [1, 64], 'k--', alpha=0.4, label='ideal (lineal)')
ax2.set_xlabel('núcleos MPI')
ax2.set_ylabel('speedup vs 1 núcleo')
ax2.set_title('Eficiencia de escalado CPU')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, which='both', alpha=0.3)

if gpu_step:
    ax1.axhline(gpu_step, color='tab:red', lw=2,
                label=f'{gpu_label}\n≥{gpu_step:g} s/paso')
    ax1.annotate(f'GPU 32³: ≥{gpu_step:g} s/paso\n(1 paso en 600 s; con pool\nno completa ni 1 en 1800 s)',
                 xy=(1, gpu_step), xytext=(6, -4), textcoords='offset points',
                 color='tab:red', fontsize=8.5, va='top')

ax1.legend(loc='lower left', fontsize=8)
ax2.legend(loc='upper left', fontsize=8)

fig.tight_layout()
fig.savefig(out, dpi=130)
print('escrito:', out)

for nx in sorted(by_nx):
    rs = sorted(by_nx[nx], key=lambda r: int(r['nprocs']))
    base = float(rs[0]['step_med_s'])
    print(f'--- {nx}³ ---')
    for r in rs:
        s = float(r['step_med_s'])
        print(f"  {int(r['nprocs']):3d} cores: {s:8.4f} s/paso  speedup={base/s:6.2f}x")
    if gpu_step and nx == 32:
        print(f'  GPU / 1 core = {gpu_step/base:,.0f}x mas lento  '
              f'(= {base/gpu_step:.6f} nucleos equivalentes)')
