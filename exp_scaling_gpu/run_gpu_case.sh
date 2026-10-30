#!/bin/bash
set -e
CASE=$1
NX=$2
MAX_ITERS=${3:-60}
TIMEOUT_S=${4:-300}

source /openfoamGPU/env_gpu.sh
export FOAM_MEMORY_POOL=host
export FOAM_MEMORY_POOL_SIZE=1024
export FOAM_MEMORY_POOL_INCR=1024
export FOAM_SIGFPE=false
export HSA_XNACK=1

cd $CASE
# clean
rm -rf [1-9]* processor* 2>&1 | tail -2
# blockMesh
blockMesh > log.blockMesh 2>&1
# checkMesh quick
checkMesh > log.checkMesh 2>&1 | tail -5

# run icoFoam with timeout and timing
/usr/bin/time -v timeout ${TIMEOUT_S}s icoFoam > log.icoFoam 2>&1
echo "---SEPARATOR---"
tail -80 log.icoFoam
