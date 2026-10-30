#!/bin/bash
# ==============================================================================
# Script de compilación de OpenFOAM (rama feature-gpu) para GPUs AMD (gfx90a)
# ==============================================================================

set -e

echo "=== 1. Configurando rutas exactas de AMD ROCm ==="

export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH

# Verificar que amdclang++ responda correctamente
if ! command -v amdclang++ &> /dev/null; then
    echo "[ERROR] No se pudo encontrar 'amdclang++'. Revisa las rutas de ROCm."
    exit 1
fi

echo "Compilador detectado correctamente: $(which amdclang++)"

echo "=== 2. Configurando variables de entorno para OpenFOAM ==="

# Arquitectura objetivo para la tarjeta AMD Instinct MI210
export ROCM_GPU=gfx90a

# Limpiar variables previas de OpenFOAM
unset WM_PROJECT_DIR WM_THIRD_PARTY_DIR

# Cargar entorno de OpenFOAM indicando la regla Amd-gpu
source etc/bashrc WM_COMPILER=Amd-gpu

echo "WM_COMPILER: $WM_COMPILER"
echo "ROCM_GPU: $ROCM_GPU"

echo "=== 3. Lanzando compilación nativa ==="

# Compilación optimizada siguiendo las opciones de la documentación oficial:
# -j (usar todos los cores)
# -s (reducir nivel de salida)
# -q (modo cola / queuing)
# -l (guardar registro en archivo log)
./Allwmake -j -s -q -l
