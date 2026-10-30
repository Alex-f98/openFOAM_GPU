#!/bin/bash
# Barrido CPU: mide el mismo caso con distinto numero de nucleos.
#
# Uso: run_cpu_ladder.sh [NX] [CORES] [STEPS] [TIMEOUT_S]
#   NX por defecto 32 (la malla mayor que complete la GPU)
#   CORES por defecto "1 2 4 8 16 32"
set -u

NX=${1:-32}
CORES=${2:-"1 2 4 8 16 32"}
STEPS=${3:-20}
TMO=${4:-1800}

BASE=/openfoamGPU/exp_scaling_cpu

echo "### CPU ladder NX=$NX cores=[$CORES] steps=$STEPS"
python3 /openfoamGPU/exp_scaling_gpu_gamg/gen_case.py "$NX" "$STEPS" "$BASE/case_${NX}cube" || exit 1

for NP in $CORES; do
    echo "================ $(date -Is)  np=$NP ================"
    bash "$BASE/run_cpu.sh" "$NX" "$NP" "$STEPS" "$TMO"
    # si un reparto no es posible, seguir con el resto
    [ $? -eq 3 ] && { echo "(saltado np=$NP: sin reparto valido)"; continue; }
done

echo "================ $(date -Is)  FIN ================"
cat "$BASE/results_cpu_gamg.csv" 2>/dev/null
