#!/bin/bash
OUT=$1
while true; do
  ts=$(date +%s)
  amd-smi monitor -ptum --json 2>&1 | jq -r --arg ts "$ts" '.[]? | "\($ts),\(.gpu_busy_percent // .gpu_activity.gfx_activity // .["GPU use (%)"] // "NA"),\(.gpu_memory_used // .vram_used // .["GPU memory used (bytes)"] // "NA")"' 2>&1 | head -1 >> $OUT
  sleep 1
done
