#!/bin/bash
# ---------------------------------------------------------------------------
# Exporta la matriz de presión del caso 128^3 a roc_matrix_p.bin (formato que
# leen gpu_krylov/bench_*.cpp).
#
# Requiere: build CPU + plugin rocALUTION compilado (ver README del traspaso).
#        : librocALUTIONLduSolver.so en $FOAM_USER_LIBBIN y en LD_LIBRARY_PATH
#
# Uso: ./dump128.sh [DESTINO]   (por defecto /tmp/opencode/dump128)
# Duración: ~60 s (1 paso de tiempo; el solve de p con Jacobi es flojo a
# propósito, solo nos interesa la matriz).
# ---------------------------------------------------------------------------
set +u
source /openfoamGPU/exp_scaling_cpu/env_cpu.sh
set -u
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 FOAM_SIGFPE=false

DEST=${1:-/tmp/opencode/dump128}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DEST"
cd "$DEST" || exit 1

if [ ! -f constant/polyMesh/points ]; then
    cp -a /openfoamGPU/exp_scaling_cpu/case_128cube/{system,constant,0} . || exit 1
    blockMesh > log.blockMesh 2>&1 || { echo "blockMesh FALLO"; exit 1; }
fi

# 1 paso de tiempo y sin escribir campos
sed -i "s/^endTime .*/endTime 0.005;/" system/controlDict
sed -i "s/^deltaT .*/deltaT          0.005;/" system/controlDict
sed -i "s/^writeInterval .*/writeInterval   1000;/" system/controlDict
# solver de GPU con volcado; Jacobi para que el paso sea barato
cp "$HERE/fv_dump_roc" system/fvSolution

rm -f roc_matrix_p.bin
icoFoam > log.dump 2>&1
RC=$?
grep -E "rocALUTION build|wrote roc_matrix" log.dump
ls -l roc_matrix_p.bin 2>/dev/null
echo "rc=$RC  -> $DEST/roc_matrix_p.bin"
