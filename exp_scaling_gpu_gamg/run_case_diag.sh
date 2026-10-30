#!/bin/bash
# Corre un caso cubico NX^3 en GPU (solo GPU 0) y registra resultados.
#
# Uso: run_case.sh NX [STEPS] [TIMEOUT_S] [POOL]
#   POOL por defecto: 'managed; size=1024; incr=1024'
#
# Deja: logs/gpu_<NX>cube_<pool>.log, .monitor.csv y una fila en results_gpu_gamg.csv
set -u

NX=${1:?usage: run_case.sh NX [STEPS] [TIMEOUT_S] [POOL]}
STEPS=${2:-20}
TMO=${3:-1800}
POOL=${4:-'managed; size=1024; incr=1024'}

BASE=/openfoamGPU/exp_scaling_gpu_gamg
CASE="$BASE/case_${NX}cube"
CSV="$BASE/diagnostics_pool.csv"
LOGD="$BASE/logs"
mkdir -p "$LOGD"

TAG="gpu_${NX}cube_$(echo "$POOL" | cut -d';' -f1 | tr -d ' ')"
LOG="$LOGD/$TAG.log"
MON="$LOGD/$TAG.monitor.csv"

# --- entorno ---
# FIX: el bashrc de OpenFOAM (v2606/etc/bashrc) referencia variables sin definir
# (p.ej. WM_PROJECT_DIR); con 'set -u' el source aborta y el caso nunca se ejecuta.
set +u
source /openfoamGPU/env_gpu.sh
set -u
export ROCR_VISIBLE_DEVICES=0      # SOLO la GPU 0 (la nuestra)
export HIP_VISIBLE_DEVICES=0
export HSA_XNACK=1
export FOAM_SIGFPE=false
export FOAM_MEMORY_POOL="$POOL"

cd "$CASE" || { echo "ERROR: no existe $CASE"; exit 1; }

# --- limpieza: borrar SOLO directorios de tiempo numericos != 0 ---
find . -maxdepth 1 -type d -regextype posix-extended \
     -regex '\./[0-9]+(\.[0-9]+)?' ! -name '0' -exec rm -rf {} + 2>/dev/null
rm -rf postProcessing log.blockMesh log.checkMesh log.icoFoam 2>/dev/null || true
: > "$LOG"
: > "$MON"

# --- malla ---
# FIX: no basta con que exista polyMesh; si el directorio del caso se reutiliza
# con otro NX la malla seria obsoleta (p.ej. restos de un intento anterior) y la
# medicion quedaria mal etiquetada. Regenerar si el numero de celdas no cuadra.
EXPECT=$((NX * NX * NX))
checkMesh > log.checkMesh 2>&1
NCELLS=$(awk '/^ *cells:/{print $2; exit}' log.checkMesh)
[ -z "${NCELLS:-}" ] && NCELLS=0
if [ "$NCELLS" != "$EXPECT" ]; then
    echo "# malla ausente/obsoleta ($NCELLS celdas != $EXPECT): regenerando" >> "$LOG"
    rm -rf constant/polyMesh
    blockMesh > log.blockMesh 2>&1 || { echo "ERROR: blockMesh fallo"; exit 1; }
    checkMesh > log.checkMesh 2>&1
    NCELLS=$(awk '/^ *cells:/{print $2; exit}' log.checkMesh)
    [ -z "${NCELLS:-}" ] && NCELLS=0
fi
# Guardia de validez: la malla debe tener exactamente NX^3 celdas y no puede
# arrastrar los errores del intento anterior (caras duplicadas / parches empty).
if [ "$NCELLS" != "$EXPECT" ]; then
    echo "ERROR: nCells=$NCELLS != $EXPECT en $CASE" | tee -a "$LOG"
    exit 1
fi
if grep -qE "already belongs to some other patch|Trying to specify a boundary face" log.checkMesh log.blockMesh 2>/dev/null; then
    echo "ERROR: malla invalida (caras duplicadas) en $CASE" | tee -a "$LOG"
    exit 1
fi
if grep -qE "^ *frontAndBack|type +empty" log.checkMesh 2>/dev/null; then
    echo "ERROR: la malla contiene parches 'empty' (no es 3D valido)" | tee -a "$LOG"
    exit 1
fi
echo "# nCells=$NCELLS  case=$CASE" >> "$LOG"

# --- identificar tarjeta amdgpu de la GPU 0 (mapeo por BDF, no por orden) ---
# FIX: el codigo original hacia basename(dirname(f)) -> "device" (no el nombre de
# la tarjeta) y el monitor mezclaba las muestras de TODAS las tarjetas, incluida
# la GPU 1 (ajena). Ahora se mapea ROCm GPU 0 -> BDF con amd-smi y se monitoriza
# exclusivamente esa tarjeta.
BDF0=$(amd-smi list 2>/dev/null | awk '/^GPU: 0 *$/{f=1;next} /^GPU: [1-9]/{f=0} f&&/BDF:/{print $2; exit}')
GPU_CARD=""
for d in /sys/class/drm/card[0-9]*/device; do
    [ -e "$d/gpu_busy_percent" ] || continue
    card=$(basename "$(dirname "$d")")
    bdf=$(sed -n 's/^PCI_SLOT_NAME=//p' "$d/uevent" 2>/dev/null)
    echo "# amdgpu card=$card bdf=$bdf" >> "$LOG"
    if [ -n "${BDF0:-}" ] && [ "$bdf" = "$BDF0" ]; then
        GPU_CARD="$card"
    fi
done
if [ -z "$GPU_CARD" ]; then
    # Fallback: primera tarjeta con VRAM (solo si amd-smi no dio el BDF)
    for d in /sys/class/drm/card[0-9]*/device; do
        [ -e "$d/gpu_busy_percent" ] || continue
        t=$(cat "$d/mem_info_vram_total" 2>/dev/null || echo 0)
        if [ "$t" -gt 1073741824 ]; then
            GPU_CARD=$(basename "$(dirname "$d")"); break
        fi
    done
fi
if [ -z "$GPU_CARD" ]; then
    echo "ERROR: no se pudo identificar la tarjeta de la GPU 0" | tee -a "$LOG"
    exit 1
fi
echo "# monitor card=$GPU_CARD (ROCR_VISIBLE_DEVICES=0, BDF=$BDF0) pool=$POOL" >> "$LOG"

# --- monitor en background (solo GPU_CARD) ---
(
  while :; do
    ts=$(date +%s)
    d="/sys/class/drm/$GPU_CARD/device"
    b=$(cat "$d/gpu_busy_percent" 2>/dev/null || echo NA)
    v=$(cat "$d/mem_info_vram_used" 2>/dev/null || echo NA)
    if [ "$v" != "NA" ]; then v=$((v / 1024 / 1024)); fi
    echo "$ts,$GPU_CARD,$b,$v" >> "$MON"
    sleep 1
  done
) &
MONPID=$!

# --- ejecucion ---
echo "# START $(date -Is)" >> "$LOG"
START=$(date +%s)
timeout --signal=INT --kill-after=60 "$TMO" icoFoam >> "$LOG" 2>&1
RC=$?
END=$(date +%s)
WALL=$((END - START))
kill "$MONPID" 2>/dev/null
wait "$MONPID" 2>/dev/null
echo "# END $(date -Is) rc=$RC wall=${WALL}s" >> "$LOG"

python3 "$BASE/analyze.py" "$LOG" "$MON" "$GPU_CARD" "$NCELLS" \
        "$WALL" "$RC" "$STEPS" "$NX" "$POOL" "$CSV"
