# Entorno de la build CPU oficial (Gcc). Uso: source env_cpu.sh
# NOTA: deja nounset desactivado (el bashrc de OpenFOAM usa variables sin definir).
set +u
export WM_COMPILER=Gcc
export WM_MPLIB=SYSTEMOPENMPI
export WM_LABEL_SIZE=32
# asegurarse de que no se hereda nada del entorno GPU
unset HSA_XNACK
unset FOAM_MEMORY_POOL
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source /openfoamGPU/cpu/OpenFOAM-v2606/etc/bashrc WM_COMPILER=Gcc
export FOAM_SIGFPE=false
