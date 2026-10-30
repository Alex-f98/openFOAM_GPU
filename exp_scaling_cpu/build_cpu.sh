#!/bin/bash
# Compila la version CPU oficial (OpenFOAM-v2606.tgz) con Gcc.
# NO ejecutar mientras haya mediciones GPU en curso: satura los 128 nucleos.
set -u

SRC=/openfoamGPU/dist/OpenFOAM-v2606.tgz
ROOT=/openfoamGPU/cpu
DEST=$ROOT/OpenFOAM-v2606

mkdir -p "$ROOT" /openfoamGPU/logs

if [ ! -d "$DEST" ]; then
    echo "extrayendo $SRC -> $ROOT"
    tar xzf "$SRC" -C "$ROOT" || exit 1
fi

cat > "$DEST/etc/prefs.sh" <<'EOF'
export WM_COMPILER=Gcc
export WM_MPLIB=SYSTEMOPENMPI
export WM_LABEL_SIZE=32
export WM_COMPILE_OPTION=Opt
EOF

# El bashrc de OpenFOAM usa variables sin definir -> desactivar nounset al sourcear
set +u
# shellcheck disable=SC1090
source "$DEST/etc/bashrc" WM_COMPILER=Gcc
set -u
echo "WM_PROJECT_DIR=$WM_PROJECT_DIR"
echo "WM_OPTIONS=$WM_OPTIONS"

cd "$DEST" || exit 1
echo "compilando con $(nproc) procesos..."
./Allwmake -j "$(nproc)" -s -q -l > /openfoamGPU/logs/cpu_build.log 2>&1
RC=$?
echo "Allwmake exit=$RC (log: /openfoamGPU/logs/cpu_build.log)"

if [ -x "$FOAM_APPBIN/icoFoam" ]; then
    echo "OK: $FOAM_APPBIN/icoFoam"
    "$FOAM_APPBIN/icoFoam" -help > /dev/null 2>&1 && echo "icoFoam se ejecuta"
else
    echo "FALLO: no se genero icoFoam"
    tail -40 /openfoamGPU/logs/cpu_build.log
    exit 1
fi
