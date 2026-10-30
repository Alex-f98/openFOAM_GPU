#!/bin/bash
# Usage: run_gpu.sh <casedir> <NX> [max_iters] [timeout_s]
CASEDIR=$1
NX=$2
MAX_ITERS=${3:-40}
TIMEOUT_S=${4:-300}
OUTCSV=/openfoamGPU/exp_scaling_gpu/results_gpu.csv
SIZE=${NX}x${NX}x${NX}

source /openfoamGPU/env_gpu.sh
export FOAM_MEMORY_POOL="host; size=1024; incr=1024"
export FOAM_SIGFPE=false
export HSA_XNACK=1

cd $CASEDIR || exit 1
# Remove only reconstructed time dirs (>0) and processor dirs; NEVER touch 0/
for d in */; do
  dn=${d%/}
  case "$dn" in
    ''|*[!0-9.]*) continue ;;
  esac
  if awk -v v="$dn" 'BEGIN{exit !(v+0>0)}'; then rm -rf "$dn"; fi
done
rm -rf processor* log.* 2>/dev/null
# ensure 0/ exists with U and p
if [ ! -f 0/U ] || [ ! -f 0/p ]; then
  TUTORIAL=/openfoamGPU/OpenFOAM-v2606/tutorials/incompressible/icoFoam/cavity/cavity
  rm -rf 0
  cp -r $TUTORIAL/0 ./0
fi
# blockMesh + checkMesh
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
NCELLS=$(grep -m1 "cells:" log.checkMesh | awk '{print $2}')
echo "NCELLS=$NCELLS"

START=$(date +%s)
timeout ${TIMEOUT_S}s icoFoam > log.icoFoam 2>&1
END=$(date +%s)
WALL=$((END-START))

ITERS=$(grep -c "^Time =" log.icoFoam)

# Per-timestep times from ExecutionTime deltas (steady state metric).
# ExecutionTime increments printed after each completed timestep.
ET_EXEC_TIMES=$(grep "ExecutionTime" log.icoFoam | awk '{print $3}')
# compute deltas
DELTAS=$(echo "$ET_EXEC_TIMES" | awk 'NR>1{printf "%.6f\n", $1-prev} {prev=$1}')
if [ -n "$DELTAS" ]; then
  AVG_STEP=$(echo "$DELTAS" | awk '{s+=$1; n++} END{if(n>0) printf "%.6f", s/n; else print "NA"}')
  SUM_STEP=$(echo "$DELTAS" | awk '{s+=$1} END{printf "%.6f", s}')
  NSTEP=$(echo "$DELTAS" | wc -l)
else
  AVG_STEP="NA"; SUM_STEP="NA"; NSTEP=0
fi

LASTEXEC=$(echo "$ET_EXEC_TIMES" | tail -1)
if [ -z "$LASTEXEC" ]; then LASTEXEC=0; fi
if [ "$AVG_STEP" = "NA" ]; then
  NOTES="nocmplt_step timeout${TIMEOUT_S}s"
else
  NOTES="shared_gpu avg_step($NSTEP steps) timeout${TIMEOUT_S}s"
fi

echo "$SIZE,$NX,$NX,$NX,$NCELLS,$ITERS,$WALL,$AVG_STEP,$SUM_STEP,$LASTEXEC,$NOTES" >> $OUTCSV
echo "---RESULT---"
echo "size=$SIZE ncells=$NCELLS iters=$ITERS wall=$WALL avg_step=$AVG_STEP nsteps=$NSTEP lastexec=$LASTEXEC"
grep -E "^(Time =|ExecutionTime)" log.icoFoam | tail -8
