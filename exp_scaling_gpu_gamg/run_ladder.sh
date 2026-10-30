#!/bin/bash
# Escalera de mallas en GPU. Para solo si una corrida falla o el coste por
# celda empeora >3x (ver check_stop.py).
#
# Uso:  run_ladder.sh [SIZES] [STEPS] [TIMEOUT_S] [POOL]
#   SIZES por defecto: "32 48 64 96 128"
set -u

BASE=/openfoamGPU/exp_scaling_gpu_gamg
SIZES=${1:-"32 48 64 96 128"}
STEPS=${2:-20}
TMO=${3:-1800}
POOL=${4:-'managed; size=1024; incr=1024'}

mkdir -p "$BASE/logs"

for NX in $SIZES; do
    echo "================ $(date -Is)  NX=$NX ================"
    python3 "$BASE/gen_case.py" "$NX" "$STEPS" "$BASE/case_${NX}cube" || {
        echo "STOP: fallo generando el caso $NX"; break; }
    BEFORE=$( [ -f "$BASE/results_gpu_gamg.csv" ] && wc -l < "$BASE/results_gpu_gamg.csv" || echo 0 )
    bash "$BASE/run_case.sh" "$NX" "$STEPS" "$TMO" "$POOL"
    AFTER=$( [ -f "$BASE/results_gpu_gamg.csv" ] && wc -l < "$BASE/results_gpu_gamg.csv" || echo 0 )
    if [ "$AFTER" -le "$BEFORE" ]; then
        echo "STOP: run_case.sh no registro ninguna fila (fallo de ejecucion)"
        break
    fi
    DEC=$(python3 "$BASE/check_stop.py" "$BASE/results_gpu_gamg.csv")
    echo "DECISION: $DEC"
    case "$DEC" in
        STOP*) break ;;
    esac
done

echo "================ $(date -Is)  FIN ================"
cat "$BASE/results_gpu_gamg.csv"
