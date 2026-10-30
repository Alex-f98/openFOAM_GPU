#!/usr/bin/env python3
"""Summarise a rocALUTION PoC run: log.TAG + monitor_TAG.csv.

Usage: summarize.py CASE_DIR TAG [NCELLS]
"""
import csv
import os
import re
import statistics as st
import sys

case, tag = sys.argv[1], sys.argv[2]
ncells = int(sys.argv[3]) if len(sys.argv) > 3 else None

log = os.path.join(case, "log." + tag)
mon = os.path.join(case, "monitor_" + tag + ".csv")

txt = open(log, errors="ignore").read()

# --- timings ---------------------------------------------------------------
cum = [float(m) for m in re.findall(r"^ExecutionTime = ([0-9.eE+-]+) s", txt, re.M)]
if len(cum) < 2:
    sys.exit(f"{log}: only {len(cum)} ExecutionTime entries")

step = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
med = st.median(step[1:]) if len(step) > 2 else step[0]
mean = sum(step) / len(step)

nf = re.findall(r"nCells=(\d+) nnz=(\d+)", txt)
nnz = int(nf[-1][1]) if nf else (ncells * 7 if ncells else None)

# CPU-only ExecutionTime (OpenFOAM excludes sub-step sync) vs wall clock
wall = None
w = os.path.join(case, "wall." + tag)
if os.path.exists(w):
    a, b = open(w).read().split()
    wall = float(b) - float(a)

# --- GPU -------------------------------------------------------------------
rows = list(csv.DictReader(open(mon)))
use = [float(r["gpu_use_pct"]) for r in rows if r["gpu_use_pct"] not in ("", "nan")]
vram = [int(r["vram_used_B"]) for r in rows if r["vram_used_B"] not in ("", "nan")]
gfx = [int(r["gfx_activity"]) for r in rows if r["gfx_activity"] not in ("", "nan")]

nz = [u for u in use if u > 0]

print(f"--- {case} [{tag}] ---")
print(f"steps            : {len(step)}")
print(f"step time (s)    : first={step[0]:.3f}  median(2..N)={med:.3f}  mean={mean:.3f}"
      f"  total={cum[-1]:.2f}")
print(f"wall clock (s)   : {wall:.2f}" if wall else "wall clock (s)   : n/a")
if ncells and nnz:
    print(f"cells / nnz      : {ncells} / {nnz}")
    print(f"s/cell/step (med): {med / ncells * 1e6:.3f} us")
if vram:
    base = vram[0]
    print(f"VRAM used        : base={base / 2**20:.0f} MiB  peak={max(vram) / 2**20:.0f} MiB")
if use:
    print(f"GPU use (%)      : median(all)={st.median(use):.0f}"
          f"  median(active)={st.median(nz) if nz else 0:.0f}"
          f"  samples>50%={sum(u > 50 for u in use)}/{len(use)}")
if gfx and len(gfx) > 1:
    print(f"GFX activity     : +{(gfx[-1] - gfx[0]) / 1e9:.1f}e9 counts")

# --- solver iterations -----------------------------------------------------
res = re.findall(
    r"Solving for (\w+), Initial residual = ([0-9.eE+-]+), "
    r"Final residual = ([0-9.eE+-]+), No Iterations (\d+)", txt)
if res:
    from collections import defaultdict
    it = defaultdict(list)
    fin = defaultdict(list)
    for f, r0, rf, n in res:
        it[f].append(int(n))
        fin[f].append(float(rf))
    for f in sorted(it):
        print(f"{f:<4} iters          : median={st.median(it[f]):.1f}"
              f"  max={max(it[f])}  finalRes(median)={st.median(fin[f]):.2e}"
              f"  worst={max(fin[f]):.2e}")
