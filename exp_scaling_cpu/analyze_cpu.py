#!/usr/bin/env python3
"""Analiza un log de icoFoam (serial o MPI) y escribe una fila de resultados CPU.

Uso: analyze_cpu.py LOG NCELLS WALL RC STEPS NX NPROC OUTCSV
"""

import csv
import os
import re
import statistics
import sys

LOG, NCELLS, WALL, RC, STEPS, NX, NPROC, OUTCSV = sys.argv[1:9]
NCELLS = int(NCELLS)
WALL = float(WALL)
RC = int(RC)
STEPS = int(STEPS)
NX = int(NX)
NPROC = int(NPROC)

text = open(LOG, errors='replace').read()

iters = len(re.findall(r'^Time = ', text, flags=re.M))
execs = [float(m.group(1)) for m in
         re.finditer(r'^ExecutionTime = ([0-9.eE+-]+) s', text, flags=re.M)]

# solo la serie del master (por si otros ranks imprimen)
if len(execs) > STEPS * 3:
    execs = execs[:STEPS]

deltas = [execs[i] - execs[i - 1] for i in range(1, len(execs))]
deltas = [d for d in deltas if d > 0]

step_first = deltas[0] if deltas else float('nan')
rest = deltas[1:] if len(deltas) > 1 else deltas
step_med = statistics.median(rest) if rest else float('nan')
exec_last = execs[-1] if execs else float('nan')
per_cell = step_med / NCELLS if NCELLS else float('nan')

status = 'ok' if (RC == 0 and iters >= STEPS) else ('timeout' if iters < STEPS else 'fail')

row = [NX, NCELLS, NPROC, status, iters, f'{WALL:.2f}', f'{exec_last:.3f}',
       f'{step_first:.4f}', f'{step_med:.4f}', f'{per_cell:.9f}']

hdr = ['nx', 'ncells', 'nprocs', 'status', 'iters_done', 'wall_s', 'exec_last_s',
       'step_first_s', 'step_med_s', 'step_per_cell_s']

new = not os.path.exists(OUTCSV)
with open(OUTCSV, 'a', newline='') as f:
    w = csv.writer(f)
    if new:
        w.writerow(hdr)
    w.writerow(row)

print('--- RESULT CPU ---')
print(f'nx={NX} ncells={NCELLS} nprocs={NPROC} status={status} iters={iters}/{STEPS} rc={RC}')
print(f'wall={WALL:.1f}s exec_last={exec_last:.1f}s first_step={step_first:.2f}s')
print(f'steady step (median)={step_med:.3f}s per_cell={per_cell:.3e}s')
