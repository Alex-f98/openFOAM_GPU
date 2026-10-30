#!/usr/bin/env python3
"""Figures for the rocALUTION PoC (cavity3D, one MI210 vs CPU)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CPU baseline -----------------------------------------------------------
cpu = {}
for r in csv.DictReader(open("/openfoamGPU/exp_scaling_cpu/results_cpu_gamg.csv")):
    if r["status"] != "ok":
        continue
    cpu.setdefault(int(r["nx"]), {})[int(r["nprocs"])] = float(r["step_med_s"])

# --- GPU PoC ----------------------------------------------------------------
gpu = {"gpu-full": {}, "gpu-p-only": {}}
for r in csv.DictReader(open("/openfoamGPU/rocALUTION-solver/results_roc_poc.csv")):
    gpu[r["config"]][int(r["mesh"])] = float(r["step_med_s"])

meshes = [32, 64, 128]
cells = {32: 32768, 64: 262144, 128: 2097152}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

# --- left: scaling ----------------------------------------------------------
for np_, lab, st, col in [(1, "CPU 1 core", "-o", "tab:blue"),
                          (8, "CPU 8 cores", "-^", "tab:cyan"),
                          (32, "CPU 32 cores", "-s", "tab:green"),
                          (64, "CPU 64 cores", "-D", "tab:olive")]:
    xs = [cells[m] for m in meshes if np_ in cpu[m]]
    ys = [cpu[m][np_] for m in meshes if np_ in cpu[m]]
    ax.plot(xs, ys, st, label=lab, color=col, lw=1.6, ms=5)

for key, lab, st, col in [("gpu-p-only", "GPU 0 (p only)", "--o", "tab:orange"),
                          ("gpu-full", "GPU 0 (p + U)", "-o", "tab:red")]:
    xs = [cells[m] for m in meshes]
    ys = [gpu[key][m] for m in meshes]
    ax.plot(xs, ys, st, label=lab, color=col, lw=2.2, ms=7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("cells")
ax.set_ylabel("time per step [s] (median of steps 2-20)")
ax.set_title("cavity3D, Re=100: one MI210 (GPU 0) vs CPU")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
ax.set_xticks([cells[m] for m in meshes])
ax.set_xticklabels([f"{m}$^3$\n{cells[m]:,}" for m in meshes])

# --- right: 128^3 phase breakdown ------------------------------------------
labels = ["OpenFOAM\nassembly\n(CPU)", "CSR values\nfill (CPU)",
          "ReBuildNumeric\nU x3 (CPU host)", "H2D/D2H\ntransfers",
          "GPU solves"]
vals = [3141, 327, 882, 41, 99]   # ms, from log.prod128 rocTiming lines
colors = ["tab:gray", "tab:blue", "tab:purple", "tab:cyan", "tab:red"]

bot = 0
for v, l, c in zip(vals, labels, colors):
    ax2.bar([0], [v], bottom=bot, color=c, edgecolor="white", label=l, width=0.55)
    if v > 200:
        ax2.text(0, bot + v/2, f"{v} ms\n({100*v/sum(vals):.0f}%)",
                 ha="center", va="center", fontsize=9, color="white", weight="bold")
    bot += v

ax2.set_xticks([0])
ax2.set_xticklabels(["128$^3$ (2.1 M cells)\n4.50 s/step"])
ax2.set_ylabel("time per step [ms]")
ax2.set_title("Where the GPU step time goes")
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
ax2.set_ylim(0, 4700)
ax2.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("/openfoamGPU/rocALUTION-solver/poc_scaling.png", dpi=150)
print("wrote poc_scaling.png")
