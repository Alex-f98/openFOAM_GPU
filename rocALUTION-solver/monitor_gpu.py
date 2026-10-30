#!/usr/bin/env python3
"""Sample GPU 0 utilisation / VRAM while a case runs.

Usage: monitor_gpu.py OUT.csv
Stops on SIGTERM/SIGINT. Only card0 (the assigned GPU) is recorded.
"""
import json
import signal
import subprocess
import sys
import time

out = open(sys.argv[1], "w", buffering=1)
out.write("t,gpu_use_pct,vram_used_B,gfx_activity\n")

running = True


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

t0 = time.time()
while running:
    try:
        raw = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL,
        )
        c = json.loads(raw)["card0"]
        out.write(
            f"{time.time() - t0:.3f}"
            f",{c['GPU use (%)']}"
            f",{c['VRAM Total Used Memory (B)']}"
            f",{c['GFX Activity']}\n"
        )
    except Exception as e:  # noqa: BLE001 - monitoring must never kill the run
        out.write(f"{time.time() - t0:.3f},nan,nan,nan\n")
    time.sleep(0.2)
