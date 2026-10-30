#!/usr/bin/env python3
"""Sample GPU0 metrics every 1s. Usage: monitor_gpu.py <outfile> [max_seconds]"""
import json, subprocess, time, sys

out = sys.argv[1]
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0

t0 = time.time()
with open(out, 'w') as f:
    f.write("t,gfx_pct,umc_pct,vram_used_mb,socket_power_w\n")
    while time.time() - t0 < dur:
        t = time.time() - t0
        try:
            r = subprocess.run(["amd-smi", "metric", "--json", "-g", "0"],
                               capture_output=True, text=True, timeout=10)
            d = json.loads(r.stdout)["gpu_data"][0]
            gfx = d["usage"]["gfx_activity"]
            umc = d["usage"]["umc_activity"]
            gfx = gfx["value"] if isinstance(gfx, dict) else "NA"
            umc = umc["value"] if isinstance(umc, dict) else "NA"
            vram = d["mem_usage"]["used_vram"]["value"]
            pw = d["power"]["socket_power"]["value"]
            f.write(f"{t:.2f},{gfx},{umc},{vram},{pw}\n")
        except Exception:
            f.write(f"{t:.2f},ERR,ERR,ERR,ERR\n")
        f.flush()
        time.sleep(1)
