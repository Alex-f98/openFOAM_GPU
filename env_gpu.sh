# Entorno común para OpenFOAM v2606 con offload GPU (AMD MI210 / gfx90a)
# Uso:  source /openfoamGPU/env_gpu.sh

export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export ROCM_GPU=gfx90a
export HSA_XNACK=1
export WM_COMPILER=Amd-gpu

# El runtime ROCm (libhsa-runtime64/libamdhip64) genera un SIGFPE espurio en su
# ruta de finalizacion (__cxa_finalize) al descargar las librerias. El manejador
# de FPE de OpenFOAM (trapFpe, activo por defecto) lo interpreta como error fatal
# y produce core dump + exit code 136, aunque el programa haya terminado bien.
# Desactivamos el trapping para que las utilidades y solvers terminen limpios.
# NOTA: esto no afecta a los calculos, solo a la deteccion de FPE de la app.
export FOAM_SIGFPE=false

# OpenFOAM v2606 (rama feature-gpu) - clonado en /openfoamGPU/OpenFOAM-v2606
source /openfoamGPU/OpenFOAM-v2606/etc/bashrc WM_COMPILER=Amd-gpu
