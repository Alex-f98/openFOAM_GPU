#!/bin/bash
set -e
source /openfoamGPU/env_gpu.sh
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export FOAM_MEMORY_POOL='host; size=1024; incr=1024'
export FOAM_SIGFPE=false
export HSA_XNACK=1

CASE=$1
LOG=$2
SIZE=$3

cd $CASE
blockMesh > $LOG 2>&1
icoFoam >> $LOG 2>&1
