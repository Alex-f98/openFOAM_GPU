#!/bin/bash
# Corre un caso cubico NX^3 en CPU con NP procesos MPI (metodo simple).
#
# Uso: run_cpu.sh NX NP [STEPS] [TIMEOUT_S]
#
# Usa exactamente el mismo caso que la campana GPU (mismo gen_case.py,
# misma fvSolution GAMG + twoStageGaussSeidel) para que la comparacion
# sea valida.
set -u

NX=${1:?usage: run_cpu.sh NX NP [STEPS] [TIMEOUT]}
NP=${2:?usage: run_cpu.sh NX NP [STEPS] [TIMEOUT]}
STEPS=${3:-20}
TMO=${4:-1800}

BASE=/openfoamGPU/exp_scaling_cpu
GEN=/openfoamGPU/exp_scaling_gpu_gamg/gen_case.py
CASE="$BASE/case_${NX}cube"
CSV="$BASE/results_cpu_gamg.csv"
mkdir -p "$BASE/logs"

# shellcheck disable=SC1091
source "$BASE/env_cpu.sh"

LOG="$BASE/logs/cpu_${NX}cube_np${NP}.log"

if [ ! -d "$CASE" ]; then
    python3 "$GEN" "$NX" "$STEPS" "$CASE" || exit 1
fi

cd "$CASE" || exit 1

# --- config especifica de CPU ---
# `twoStageGaussSeidel` NO existe en la release oficial v2606 (solo en la rama GPU;
# valid smoothers oficiales: DIC DICGaussSeidel FDIC GaussSeidel
# nonBlockingGaussSeidel symGaussSeidel). Para el baseline CPU se usa
# `symGaussSeidel`, el smoother simetrico estandar de OpenFOAM para GAMG.
if grep -q 'twoStageGaussSeidel' system/fvSolution 2>/dev/null; then
    sed -i 's/twoStageGaussSeidel/symGaussSeidel/' system/fvSolution
    echo "# CPU: smoother twoStageGaussSeidel -> symGaussSeidel (no existe en v2606 oficial)" >> "$LOG"
fi

# limpiar resultados previos (nunca el 0/)
find . -maxdepth 1 -type d -regextype posix-extended \
     -regex '\./[0-9]+(\.[0-9]+)?' ! -name '0' -exec rm -rf {} + 2>/dev/null
rm -rf postProcessing processor* log.* 2>/dev/null || true
: > "$LOG"

if [ ! -f constant/polyMesh/points ]; then
    blockMesh > log.blockMesh 2>&1 || { echo "ERROR: blockMesh fallo"; exit 1; }
fi
checkMesh > log.checkMesh 2>&1
NCELLS=$(awk '/^ *cells:/{print $2; exit}' log.checkMesh)
[ -z "${NCELLS:-}" ] && NCELLS=0
echo "# nCells=$NCELLS case=$CASE np=$NP" >> "$LOG"

if [ "$NP" -gt 1 ]; then
    python3 "$BASE/gen_decomp.py" "$NX" "$NP" > system/decomposeParDict || {
        echo "ERROR: no hay reparto valido para NX=$NX NP=$NP"; exit 3; }
    decomposePar -force > log.decomposePar 2>&1 || {
        echo "ERROR: decomposePar fallo"; tail -20 log.decomposePar; exit 1; }
    RUNCMD="mpirun --allow-run-as-root -np $NP icoFoam -parallel"
else
    RUNCMD="icoFoam"
fi

echo "# START $(date -Is) cmd=$RUNCMD" >> "$LOG"
START=$(date +%s)
# shellcheck disable=SC2086
timeout --signal=INT --kill-after=60 "$TMO" $RUNCMD >> "$LOG" 2>&1
RC=$?
END=$(date +%s)
WALL=$((END - START))
echo "# END $(date -Is) rc=$RC wall=${WALL}s" >> "$LOG"

python3 "$BASE/analyze_cpu.py" "$LOG" "$NCELLS" "$WALL" "$RC" "$STEPS" "$NX" "$NP" "$CSV"
