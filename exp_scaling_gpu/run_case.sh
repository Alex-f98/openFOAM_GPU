#!/bin/bash
# run_case.sh <N> [timeout_s]
# Full pipeline for one mesh size: build, mesh, timed 20-step icoFoam on GPU0,
# GPU monitoring, append row to results_gpu.csv.
N=$1
TIMEOUT_S=${2:-1500}
EXP=/openfoamGPU/exp_scaling_gpu
CASE=$EXP/cavity3d_${N}x${N}x${N}
CSV=$EXP/results_gpu.csv
MON=$EXP/monitor_${N}.log

source /openfoamGPU/env_gpu.sh
# pool disabled - empirically catastrophic with hipstdpar on this build (xnack1 page-migration storm).
unset FOAM_MEMORY_POOL
export FOAM_SIGFPE=false
export HSA_XNACK=1
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0

bash $EXP/make_cavity3d.sh $N > /dev/null

cd $CASE || { echo "cannot cd $CASE"; exit 1; }
blockMesh  > log.blockMesh  2>&1 || { echo "blockMesh failed"; tail -20 log.blockMesh; exit 1; }
checkMesh  > log.checkMesh  2>&1
NCELLS=$(grep -m1 "cells:" log.checkMesh | awk '{print $2}')
echo "=== N=$N ncells=$NCELLS  (GPU0) ==="

python3 $EXP/monitor_gpu.py $MON 3600 &
MPID=$!
sleep 1

T0=$(date +%s.%N)
timeout -k 30 ${TIMEOUT_S} icoFoam > log.icoFoam 2>&1
RC=$?
T1=$(date +%s.%N)

sleep 1
kill $MPID 2>/dev/null
wait $MPID 2>/dev/null

WALL=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.2f", b-a}')
ITERS=$(grep -c "^Time =" log.icoFoam)
SOLVE=$(grep "ExecutionTime" log.icoFoam | tail -1 | awk '{print $3}')
[ -z "$SOLVE" ] && SOLVE=0

if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
  STATUS=timeout
elif [ "$RC" -ne 0 ]; then
  STATUS=fail
elif [ "$ITERS" -eq 20 ]; then
  STATUS=ok
else
  STATUS=fail
fi

SPI=$(awk -v s=$SOLVE -v i=$ITERS 'BEGIN{if(i>0) printf "%.4f", s/i; else print "NA"}')
SPCPI=$(awk -v s=$SOLVE -v i=$ITERS -v c=$NCELLS 'BEGIN{if(i>0&&c>0) printf "%.8f", s/(i*c); else print "NA"}')

# GPU stats from monitor log (skip rows with ERR)
read GMEAN GMAX VPEAK PMEAN <<< $(awk -F, 'NR>1 && $2!="ERR" {g+=$2; n++; if($2>mx)mx=$2; if($4>vp)vp=$4; p+=$5; pn++}
  END{if(n>0) printf "%.1f %.1f %d %.1f", g/n, mx, vp, p/pn; else printf "NA NA NA NA"}' $MON)

NOTES="Re100 U1 nu0.01 20steps dt0.005 GPU0 pool_host1024 rc${RC}"

echo "$N,$NCELLS,$STATUS,$ITERS,$WALL,$SOLVE,$SPI,$SPCPI,$GMEAN,$GMAX,$VPEAK,$NOTES" >> $CSV

echo "---RESULT---"
echo "N=$N ncells=$NCELLS status=$STATUS iters=$ITERS wall=${WALL}s solve=${SOLVE}s s_per_iter=$SPI s_per_cell_iter=$SPCPI"
echo "gpu_gfx mean=${GMEAN}% max=${GMAX}% vram_peak=${VPEAK}MB power_mean=${PMEAN}W"
grep -E "^(Time =|ExecutionTime)" log.icoFoam | tail -6
