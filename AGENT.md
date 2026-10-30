# Guía de Instalación y Compilación de OpenFOAM Nativo en GPU (AMD MI210)

> **AVISO (octubre 2026): documento histórico.** La hipótesis de este texto —que
> `--hipstdpar` descarga también el ensamblado y por eso acelera— **fue medida y es falsa**:
> esa ruta da 451 s/paso a 32³ frente a 0,05 s de la CPU (ver
> [`TRASPASO.md`](TRASPASO.md) §8). Se conserva por el valor del diagnóstico de
> `petsc4Foam`/PETSc, no como receta. Para el estado real y lo que sí funciona, empieza por
> [`TRASPASO.md`](TRASPASO.md) y [`README.md`](README.md).

## 1. Antecedentes y Diagnóstico Técnico

### ¿Por qué NO usar `petsc4Foam` / PETSc en GPU en esta plataforma?
1. **Cuello de Botella CPU/GPU (o-LAS vs LAS):** 
   `petsc4Foam` solo delega a la GPU la resolución de las matrices linealizadas (LAS - ~60% del cómputo). El ensamblado en C++ de las ecuaciones (o-LAS - ~40% del cómputo) continúa ejecutándose en un solo hilo de la CPU, creando una transferencia PCIe continua que frena el rendimiento global.
2. **Fallo de Memoria en Docker (OOM Killer / `Signal 9` / `EFAULT`):** 
   Al intentar paralelizar con MPI (`mpirun -np 2` o `-np 8`) dentro del contenedor Docker para acelerar el ensamblado, la alocación de memoria RAM y de memoria compartida (`/dev/shm`) supera los límites del contenedor, provocando cierres inesperados por `Signal 9 (Killed)` y errores `errno = 14 (EFAULT)`.

### Solución Nativa: OpenFOAM `stdpar` (C++17/20 `std::execution`)
Para evitar el uso de MPI/PETSc y eliminar los cuellos de botella de la CPU, se compila OpenFOAM utilizando paralelismo nativo en C++ (`stdpar`). Esto permite ejecutar el solver (`icoFoam`) en **un solo proceso monohilo (`-g 1`)**, donde la GPU AMD MI210 asume automáticamente tanto el ensamblado (`o-LAS`) como la resolución matemática (`LAS`).

---

## 2. Requisitos del Sistema y Dependencias (Ubuntu 24.04 LTS)

Siguiendo la documentación oficial de requerimientos, instala las herramientas esenciales y librerías del sistema:

```bash
# 1. Herramientas esenciales de compilación
apt-get update && apt-get install -y \
    build-essential autoconf autotools-dev cmake gawk gnuplot \
    flex libfl-dev libreadline-dev zlib1g-dev openmpi-bin libopenmpi-dev \
    mpi-default-bin mpi-default-dev libgmp-dev libmpfr-dev libmpc-dev \
    git bison libncurses-dev

# 2. SDK de AMD ROCm (Proporciona amdclang++ y bibliotecas HIP)
apt-get install -y rocm-hip-sdk
