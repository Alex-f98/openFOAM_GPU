#!/bin/bash
# isolate_probe.sh  -- run icoFoam 60s under different env configs, report progress
EXP=/openfoamGPU/exp_scaling_gpu
CASE=$EXP/cavity3d_32x32x32
cd $CASE

run_cfg () {
  local name="$1"; shift
  rm -rf [0-9]* 2>/dev/null
  # restore 0 dir
  cp -r /openfoamGPU/OpenFOAM-v2606/tutorials/incompressible/icoFoam/cavity/cavity/0 ./0 2>/dev/null
  python3 - <<'PY'
import re,os
p='0/U'; s=open(p).read(); s=re.sub(r'\n\s*frontAndBack\s*\{[^}]*\}','',s); open(p,'w').write(s)
p='0/p'; s=open(p).read(); s=re.sub(r'\n\s*frontAndBack\s*\{[^}]*\}','',s); open(p,'w').write(s)
PY
  source /openfoamGPU/env_gpu.sh
  export FOAM_SIGFPE=false
  export ROCR_VISIBLE_DEVICES=0
  export HIP_VISIBLE_DEVICES=0
  # apply config overrides passed as env assignments
  for kv in "$@"; do export "$kv"; done
  echo "=== CONFIG: $name  [$*] ==="
  T0=$(date +%s.%N)
  timeout -k 5 60 icoFoam > log.probe 2>&1
  T1=$(date +%s.%N)
  LASTT=$(grep "^Time =" log.probe | tail -1 | awk '{print $3}')
  NSTEP=$(grep -c "^Time =" log.probe)
  LASTEX=$(grep "ExecutionTime" log.probe | tail -1 | awk '{print $3}')
  echo "  steps_started=$NSTEP last_Time=${LASTT:-none} last_Exec=${LASTEX:-none} wall=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.1f",b-a}')s"
  grep -m1 "memory pool" log.probe | sed 's/^/  /'
  cp log.probe log.probe.$name
}

# A: no pool, no xnack
run_cfg "A_nopool_xnack0" "FOAM_MEMORY_POOL=" "HSA_XNACK=0"
# B: pool host, no xnack
run_cfg "B_pool_xnack0" "FOAM_MEMORY_POOL=host; size=1024; incr=1024" "HSA_XNACK=0"
# C: pool host + xnack
run_cfg "C_pool_xnack1" "FOAM_MEMORY_POOL=host; size=1024; incr=1024" "HSA_XNACK=1"
# D: no pool + xnack
run_cfg "D_nopool_xnack1" "FOAM_MEMORY_POOL=" "HSA_XNACK=1"
