#!/usr/bin/env bash
# Run one rocALUTION PoC case with GPU-0 monitoring.
#
#   run_case.sh CASE_DIR TAG
#
# Produces inside CASE_DIR:
#   log.TAG, monitor_TAG.csv, wall.TAG, rc.TAG
set -u

CASE=$1
TAG=${2:-run}

HERE=$(cd "$(dirname "$0")" && pwd)
CASE=$(cd "$CASE" && pwd)

cd "$CASE" || exit 1

rm -f monitor_"$TAG".csv

python3 "$HERE/monitor_gpu.py" monitor_"$TAG".csv &
MON=$!

export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export FOAM_SIGFPE=false
export OMP_NUM_THREADS=8

set +u
source /openfoamGPU/cpu/OpenFOAM-v2606/etc/bashrc
set -u

t0=$(date +%s.%N)
icoFoam > log."$TAG" 2>&1
rc=$?
t1=$(date +%s.%N)

kill "$MON" 2>/dev/null
wait "$MON" 2>/dev/null

echo "$rc" > rc."$TAG"
echo "$t0 $t1" > wall."$TAG"
python3 -c "print(f'rc=$rc wall={float('$t1')-float('$t0'):.2f} s')"
