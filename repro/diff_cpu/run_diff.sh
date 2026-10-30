#!/bin/bash
# ---------------------------------------------------------------------------
# Reparto de fases del paso de tiempo en CPU por diferencia de 3 variantes.
#
#   gamg   : p/pFinal con GAMG            -> paso completo
#   p1     : p/pFinal con 1 iteración     -> quita el coste de la presión
#   both1  : p/pFinal y U con 1 iteración -> quita también el de la velocidad
#
#   gamg - p1    = bloque p   (incluye el setup de GAMG de cada solve)
#   p1   - both1 = bloque U
#   both1        = ensamblado + PISO + I/O
#
# Uso:
#   ./run_diff.sh              # 32 y 1 rango (por defecto)
#   ./run_diff.sh 32 64        # los rangos que se quieran
#   BASE=/tmp/mi_dir ./run_diff.sh 32
#
# Salida: $BASE/results.txt y $BASE/work/log.np<NP>.<variante>
# Duración: ~2 min a 32 rangos (20 pasos), ~1 min a 1 rango (8 pasos).
# ---------------------------------------------------------------------------
set +u
source /openfoamGPU/exp_scaling_cpu/env_cpu.sh
set -u
export OMP_NUM_THREADS=1

HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-/tmp/opencode/repro_diff}
SRC=${SRC:-/openfoamGPU/exp_scaling_cpu/case_128cube}
NPS=${*:-"32 1"}

W=$BASE/work
mkdir -p "$W" "$BASE"
cp -a "$SRC/system" "$SRC/constant" "$SRC/0" "$W/" || exit 1
cd "$W" || exit 1

if [ ! -f constant/polyMesh/points ]; then
    blockMesh > log.blockMesh 2>&1 || { echo "blockMesh FALLO"; exit 1; }
fi
checkMesh > log.checkMesh 2>&1
grep -E "^ *cells:" log.checkMesh | head -1

: > "$BASE/results.txt"

clean_times()      # borra los directorios de tiempo, conservando 0/system/constant
{
    find . -mindepth 1 -maxdepth 1 -type d \
        ! -name 0 ! -name system ! -name constant ! -name postProcessing \
        -exec rm -rf {} +
}

for NP in $NPS; do
    if [ "$NP" -gt 1 ]; then
        STEPS=20
        python3 /openfoamGPU/exp_scaling_cpu/gen_decomp.py 128 "$NP" \
            > system/decomposeParDict || { echo "sin reparto para np=$NP"; continue; }
        rm -rf processor*
        decomposePar -force > log.decomposePar 2>&1 || { echo "decompose FALLO"; exit 1; }
        RUN="mpirun --allow-run-as-root --oversubscribe -np $NP icoFoam -parallel"
    else
        STEPS=8
        RUN="icoFoam"
    fi

    ENDT=$(python3 -c "print(f'{0.005*$STEPS:g}')")
    sed -i "s/^endTime .*/endTime         $ENDT;/" system/controlDict
    sed -i "s/^writeInterval .*/writeInterval   1000;/" system/controlDict

    for V in gamg p1 both1; do
        cp "$HERE/fv_$V" system/fvSolution
        rm -rf postProcessing
        if [ "$NP" -gt 1 ]; then
            # NO borrar processor*/constant (ahí está la malla descompuesta).
            for d in processor*; do
                find "$d" -mindepth 1 -maxdepth 1 -type d \
                    -regextype posix-extended -regex ".*/[0-9]+(\.[0-9]+)?" \
                    ! -name 0 -exec rm -rf {} +
            done
        else
            clean_times
        fi

        T0=$(date +%s.%N)
        $RUN > "log.np${NP}.${V}" 2>&1
        RC=$?
        T1=$(date +%s.%N)
        EXEC=$(grep -E "^ExecutionTime" "log.np${NP}.${V}" | tail -1 \
               | sed -E 's/.*ExecutionTime = ([0-9.eE+-]+) s.*/\1/')
        WALL=$(python3 -c "print(f'{$T1-$T0:.2f}')")
        echo "np=$NP variant=$V steps=$STEPS rc=$RC exec=$EXEC wall=$WALL" \
            | tee -a "$BASE/results.txt"
    done
done

echo
echo "Resumen en $BASE/results.txt"
echo "Calcula las fases con: python3 $HERE/../../repro/diff_cpu/split.py $BASE/results.txt"
