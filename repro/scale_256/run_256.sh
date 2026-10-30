#!/bin/bash
# ---------------------------------------------------------------------------
# Escalera de malla grande (256^3 = 16,8 M celdas) en CPU, mismo método que
# run_diff.sh: tres variantes para separar ensamblado / p / U.
#
# Uso:  ./run_256.sh [NX] [NP] [STEPS]
#       NX=256 NP=32 STEPS=20 por defecto
#
# Duración: ~12 min (blockMesh + checkMesh + decomposePar + 3 corridas).
# Disco: ~7 GB en el directorio de trabajo. RAM: ~40 GB de pico.
#
# Aviso: la variante p1 (presión con 1 iteración) suele divergir al final
# (NaN) porque el campo de presión no se resuelve; sus tiempos se leen de los
# pasos sanos. gamg y both1 completan los 20 pasos.
# ---------------------------------------------------------------------------
set +u
source /openfoamGPU/exp_scaling_cpu/env_cpu.sh
set -u
export OMP_NUM_THREADS=1

NX=${1:-256}
NP=${2:-32}
STEPS=${3:-20}

HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-/tmp/opencode/repro_${NX}}
C=$BASE/case

mkdir -p "$C"
cd "$C" || exit 1

if [ ! -f constant/polyMesh/points ]; then
    python3 /openfoamGPU/exp_scaling_gpu_gamg/gen_case.py "$NX" "$STEPS" "$C" || exit 1
    blockMesh > log.blockMesh 2>&1 || { echo "blockMesh FALLO"; exit 1; }
fi
checkMesh > log.checkMesh 2>&1
grep -E "^ *cells:" log.checkMesh | head -1

sed -i "s/^endTime .*/endTime         $(python3 -c "print(f'{0.005*$STEPS:g}')");/" system/controlDict
sed -i "s/^writeInterval .*/writeInterval   1000;/" system/controlDict

python3 /openfoamGPU/exp_scaling_cpu/gen_decomp.py "$NX" "$NP" > system/decomposeParDict \
    || { echo "sin reparto para np=$NP"; exit 1; }
rm -rf processor*
decomposePar -force > log.decomposePar 2>&1 || { echo "decompose FALLO"; exit 1; }

: > "$BASE/results.txt"
for V in gamg p1 both1; do
    cp "$HERE/../diff_cpu/fv_$V" system/fvSolution
    for d in processor*; do
        find "$d" -mindepth 1 -maxdepth 1 -type d \
            -regextype posix-extended -regex ".*/[0-9]+(\.[0-9]+)?" \
            ! -name 0 -exec rm -rf {} +
    done
    T0=$(date +%s.%N)
    mpirun --allow-run-as-root --oversubscribe -np "$NP" icoFoam -parallel \
        > "log.np${NP}.${V}" 2>&1
    RC=$?
    T1=$(date +%s.%N)
    EXEC=$(grep -E "^ExecutionTime" "log.np${NP}.${V}" | tail -1 \
           | sed -E 's/.*ExecutionTime = ([0-9.eE+-]+) s.*/\1/')
    echo "np=$NP variant=$V steps=$STEPS rc=$RC exec=$EXEC wall=$(python3 -c "print(f'{$T1-$T0:.2f}')")" \
        | tee -a "$BASE/results.txt"
done

echo
echo "Resultados en $BASE/results.txt  (calcula las fases con ../diff_cpu/split.py)"
python3 "$HERE/../diff_cpu/split.py" "$BASE/results.txt"
