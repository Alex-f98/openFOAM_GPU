# OpenFOAM v2606 con offload GPU en AMD MI210 — Informe técnico

**Fecha:** octubre 2026
**Plataforma:** Ubuntu 24.04 (contenedor Docker), 2× AMD EPYC 7763, 247 GiB RAM
**GPU:** AMD Instinct MI210 (gfx90a), ROCm 7.2.4 — **solo GPU 0** (la GPU 1 pertenece a otro usuario y no se usa)
**Objetivo:** compilar OpenFOAM v2606 con soporte nativo de GPU y determinar si aporta ventaja real frente a CPU.

---

## 1. Resumen ejecutivo

| Pregunta | Respuesta |
|---|---|
| ¿Se puede compilar OpenFOAM v2606 con offload GPU en AMD? | **Sí.** Build completado: 268 binarios, 128 librerías, Umpire enlazado. |
| ¿El código GPU está en el tarball oficial v2606? | **No.** El soporte GPU vive en la rama `feature-gpu` de `openfoam/gpu/openfoam-ecse`. |
| ¿Se ejecuta realmente en la GPU? | **Sí**, verificado: kernels HIP enlazados y actividad medida en GPU 0 (picos de 99-100 %). |
| ¿Es más rápido que CPU? | **No a la escala alcanzable en esta GPU.** Con `PCG+DIC`: ~451 s/paso a 32³; con `GAMG+twoStageGaussSeidel`: **no completa ni 1 paso en 30 min**. La ganancia publicada en MI300A (×2) aparece a 64M celdas (~100 GB), fuera de los 64 GB de la MI210; a 8M la propia MI300A no gana. Ver la nota de escala en §7. |
| ¿Basta con cambiar el solver a la config recomendada? | **No.** Mejora la estructura del álgebra, pero el cuello de botella es el runtime, no el solver. |
| ¿Escala con el tamaño de malla? | **Al contrario**: 8× celdas (16³→32³) ⇒ ~50× más tiempo por paso. |
| ¿A cuántos núcleos CPU equivale la GPU 0? | **A ~0,0003 núcleos** (1/3.254) en el mejor caso; <1/36.000 con pool. Ver §8.5.1. |
| ¿Hay alguna forma de usar la GPU que sí funcione? | **Sí, con un solver externo** (plugin rocALUTION): `cavity3D` completo en GPU 0, 4,50 s/paso a 128³, validado (ΔU 5e-6). Aun así solo equivale a **1,9 núcleos** (§12). |
| ¿Y repartiendo el trabajo con MPI y usando la GPU solo para el álgebra? | **Correcto pero mucho más lento.** `preconditioner rocALUTION` dentro de `PCG`/`PBiCGStab`: 10× más lento que CPU con 8 ranks y 77× con 16 (128³). El sistema precondicionado pierde convergencia (11→149 iteraciones) y el setup de AMG se paga por rank (§13). |
| ¿Y si el solve entero vive en la GPU (Krylov residente, sin *round trips*, multigrid de jerarquía fija)? | **La GPU sí es una GPU: el bloque de presión (`p`+`pFinal`) a 128³ baja a 80,5 ms, 1,72× menos que 32 núcleos y 43,6× menos que uno.** Por iteración es 5× más rápida, pero necesita 2,9× más iteraciones que GAMG. El paso completo solo mejora **1,20×** porque el ensamblado (31–45 %) sigue en CPU (§14). |

---

## 2. Corrección al plan inicial (`solution_gpt.md`)

El documento `solution_gpt.md` proponía compilar el **tarball oficial** `OpenFOAM-v2606.tgz` con:

```bash
export WM_COMPILER=Clang
export CXX=amdclang++
```

**Esto no habilita GPU.** Verificación realizada sobre el tarball oficial (69.422.947 bytes):

| Comprobación | Resultado |
|---|---|
| `wmake/rules/General/Amd-gpu` | **no existe** |
| `wmake/rules/linux64Amd-gpu` | **no existe** |
| Coincidencias de `gpu` en todo el tar | **0** |
| `FOAM_OFFLOAD`, `hipstdpar`, `stdpar` | **0 apariciones** |
| `WM_COMPILER` aceptados | `Gcc, Clang, Amd, Nvidia, ...` (sin `Amd-gpu`) |

El propio anuncio de v2606 lo indica:

> *"The code is made available as a branch on a development repository here: [source code] ... After further testing and feedback from the Community we intend to integrate the code in the OpenFOAM v2612 release."*

Por tanto, la fuente correcta es:

```
https://gitlab.com/openfoam/gpu/openfoam-ecse.git   rama: feature-gpu
```

Esa rama declara `WM_PROJECT_VERSION=v2606` y sí contiene:

- `wmake/rules/General/Amd-gpu/c++`:
  ```
  CC := amdclang++ -std=c++17 --hipstdpar --offload-arch=$(ROCM_GPU) -DFOAM_OFFLOAD
  ```
- `wmake/rules/linux64Amd-gpu/`
- Integración de Umpire (`MemoryPool.cxx`, `have_umpire`, `etc/config.sh/umpire`)

`compile_gpu.md` era, en lo esencial, correcto (`WM_COMPILER=Amd-gpu`, `ROCM_GPU=gfx90a`).

---

## 3. Entorno y verificación previa

| Elemento | Valor |
|---|---|
| SO | Ubuntu 24.04.4 LTS (contenedor, root) |
| CPU | 2× AMD EPYC 7763 (128 hilos) |
| RAM | 247 GiB |
| GPU 0 | AMD Instinct MI210, `gfx90a`, BDF `0000:03:00.0` |
| GPU 1 | MI210 — **ajena, no usada** |
| ROCm | 7.2.4, `/opt/rocm` |
| Compilador | `amdclang++` (clang 22.0.0git, roc-7.2.4) |
| MPI | Open MPI 4.1.6 (no usado para GPU) |

Prueba de `hipstdpar`:

- `amdclang++ -std=c++17 --hipstdpar --offload-arch=gfx90a` **compila**.
- Ejecución directa sin más → `Memory access fault by GPU node-2`.
- Con `HSA_XNACK=1` → **funciona**.

⇒ `HSA_XNACK=1` es obligatorio en gfx90a para memoria compartida (SVM).

---

## 4. Compilación

### 4.1 Fuente y ThirdParty

```bash
git clone --depth 1 --branch feature-gpu \
  https://gitlab.com/openfoam/gpu/openfoam-ecse.git /openfoamGPU/OpenFOAM-v2606
# commit: 559823f7d1b8ad018c7e42106571edc23b6dcf4e
# ThirdParty: ThirdParty-v2606 (dl.openfoam.com) en /openfoamGPU/ThirdParty-v2606
```

### 4.2 Umpire (memory pool)

Compilado con HIP:

```bash
./makeUMPIRE -DBLT_CXX_STD=c++17 -DENABLE_HIP=ON -DENABLE_CUDA=OFF \
             -DHIP_ROOT_DIR=/opt/rocm -DCMAKE_HIP_ARCHITECTURES=gfx90a
```

Resultado: `platforms/linux64Amd-gpu/umpire-2025.03.0/lib/{libumpire.a,libcamp.a,libfmt.a}`,
detectado por `have_umpire`.

**Bug encontrado y parcheado:** `makeUMPIRE` aceptaba argumentos `-D...` pero **no los
pasaba a cmake**; sin parche se compilaba en C++14 con HIP OFF. Se añadió
`${CMAKE_VARIABLES}` a la línea de cmake.

### 4.3 Configuración

`/openfoamGPU/OpenFOAM-v2606/etc/prefs.sh`:

```sh
export WM_COMPILER=Amd-gpu
export ROCM_GPU=gfx90a
export FOAM_EXTRA_CXXFLAGS=-Wno-overloaded-virtual
```

`/openfoamGPU/env_gpu.sh` (entorno de uso):

```sh
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export ROCM_GPU=gfx90a
export HSA_XNACK=1
export WM_COMPILER=Amd-gpu
source /openfoamGPU/OpenFOAM-v2606/etc/bashrc WM_COMPILER=Amd-gpu
```

### 4.4 Build

```bash
./Allwmake -j -s -q -l     # exit 0
```

- `WM_OPTIONS=linux64Amd-gpuDPInt32Opt`
- 268 binarios, 128 librerías
- `libOpenFOAM.so` (190 MB) con Umpire (`-DFOAM_USE_UMPIRE`, 774 símbolos)

**Excluidos (opcionales):**

| Componente | Motivo |
|---|---|
| CGAL | Incompatibilidad con amdclang++ (`inline asm not supported yet`). Solo afecta a `foamyHexMesh`. |
| KAHIP | CMake no encuentra MPI C++ con `--hipstdpar`. |
| METIS | Fuente ausente en ThirdParty → usar SCOTCH. |
| ADIOS2 | Su test de CMake usa g++/mpicxx, incompatibles con `--hipstdpar`. |

### 4.5 Problemas de ejecución resueltos

| Síntoma | Causa | Solución |
|---|---|---|
| `Memory access fault by GPU node` | SVM sin XNACK | `HSA_XNACK=1` |
| `SIGFPE` / core dump (exit 136) **al terminar** correctamente | `trapFpe` de OpenFOAM captura un SIGFPE espurio del runtime ROCm en `__cxa_finalize` | `FOAM_SIGFPE=false` |
| `inline asm not supported yet` | CGAL/Boost viejos + amdclang++ | `have_cgal` → `no_cgal(); exit` |

---

## 5. Metodología de medición

- **Caso:** cavidad cúbica unitaria, `icoFoam`, Re=100 (`U=1 m/s`, `nu=0.01`).
- **Malla:** NX³ celdas, bloque único, grading uniforme.
- **Pasos:** exactamente 20 pasos de tiempo (`deltaT=0.005`, `endTime=0.1`) en todos los tamaños → mismo trabajo por corrida.
- **Métrica principal:** tiempo por paso en régimen (`mediana` de los pasos 2..20, tomada de `ExecutionTime` del log, que excluye arranque y lectura de malla).
- **Métrica normalizada:** `s / celda / paso` → si se mantiene plana, el escalado es sano.
- **Proceso único, sin MPI** (requisito del build GPU).
- **Solo GPU 0** (`ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`).
- **Monitorización:** `gpu_busy_percent` y `mem_info_vram_used` de sysfs, muestreo cada 1 s.
- **Timeout:** 1800 s por corrida.

### 5.1 Dos errores del caso inicial (corregidos)

El primer intento de "cavity 3D" era inválido por dos motivos:

1. Malla NX×NX×NX con parche `frontAndBack { type empty; }` → `empty` solo es válido en mallas 2D.
2. Cara `(4 5 6 7)` declarada **dos veces** (`fixedWalls` y `frontAndBack`), con el aviso de `blockMesh`:
   *"Trying to specify a boundary face ... which is either an internal face or already belongs to some other patch"*.

Síntomas reales del caso inválido (verificados con `checkMesh`): `2 solution directions (1 1 0)`
y `Failed 2 mesh checks`, además del aviso de `blockMesh` por la cara duplicada.

Matiz importante: `Uy = Uz = 0` en el **primer** paso **no** es síntoma de nada — en t=0
la tapa solo impone componente x, así que esas ecuaciones tienen fuente nula por
construcción. Se comprobó que el caso nuevo es 3D porque `checkMesh` reporta
`3 solution directions (1 1 1)` y porque `Uz` se resuelve en el paso 2
(`Initial residual = 0.729`).

El generador final (`gen_case.py`) usa 6 caras correctas (`movingWall` + 5 en `fixedWalls`) y `checkMesh` válido.

---

## 6. Configuración del solver: por qué importa

El tutorial por defecto de `icoFoam` usa:

```
p { solver PCG; preconditioner DIC; }
```

La documentación oficial de la rama GPU advierte explícitamente:

> *"Do not use explicitly serial linear solver routines such as `DIC`, `DILU`, `GaussSeidel`, or `symGaussSeidel`. Use `GAMG` with `twoStageGaussSeidel` as a smoother instead."*

Motivo: con `par_unseq`, los precondicionadores secuenciales generan muchísimas
operaciones pequeñas → domina el coste por lanzamiento y la paginación SVM.

### 6.1 Evidencia en el código (por qué `DIC` no puede ir en GPU)

En OpenFOAM **no se elige un "solver de GPU"**: el offload es a nivel de bucle
(`std::par_unseq`). Lo que decide el rendimiento es si los bucles internos tienen
dependencias de datos.

| Componente | ¿Offload? | Evidencia |
|---|---|---|
| `PCG` (producto matriz-vector) | **sí** | `lduMatrixOperations.C`, `lduMatrixATmul.C`: `#ifdef FOAM_OFFLOAD` → `matrixOperation(...)` |
| `DIC` (precondicionador) | **no** | `DICPreconditioner.C:116-129`: dos bucles `for (face...)` de sustitución hacia delante/atrás, sin `par_unseq` posible |
| `DILU`, `GaussSeidel`, `symGaussSeidel` | **no** | cero `FOAM_OFFLOAD` en `preconditioners/`, `PCG/`, `smoothSolver/` |
| `GAMG` (agregación, interpolación, escalado) | **sí** | `GAMGSolverAgglomerateMatrix.C`, `GAMGSolverInterpolate.C`, `GAMGSolverScale.C` |
| `twoStageGaussSeidel` | **sí** | `apply_policy(nCells, ...)` + `std::for_each(policy, ...)` sobre direcciones **CSR** (`lowerCSR`, `losortStartAddr`) → actualización independiente por celda |

El detalle clave: la rama GPU **añadió una estructura CSR** (`lowerCSR`) solo para
poder paralelizar el smoother. `GaussSeidel` recorre caras en orden y no admite ese
tratamiento. Por eso `PCG+DIC` en GPU significa *mat-vec en GPU → resolución
triangular completa en un núcleo → mat-vec en GPU*, y el ida y vuelta domina.

Esto explica el resultado de §7 (451 s/paso, GFX 0,1 %). Lo que **no** explica es §8:
con `GAMG+twoStageGaussSeidel` el álgebra ya es paralelizable y aun así la GPU sigue
ociosa. El límite está en el runtime (`hipstdpar` + SVM/XNACK), no en el solver.

Configuración recomendada usada en el experimento:

```
p
{
    solver                  GAMG;
    tolerance               1e-06;
    relTol                  0.01;
    smoother                twoStageGaussSeidel;
    cacheAgglomeration      true;
    nCellsInCoarsestLevel   10;
    agglomerator            faceAreaPair;
    mergeLevels             1;
}
```

---

## 7. Resultados con la configuración por defecto (PCG + DIC)

Medición previa, GPU 0 libre, pool `host`, 20 pasos objetivo, timeout 1500 s:

| Malla | Celdas | Pasos completados | Wall | s/paso (régimen) | s/celda/paso | GFX medio/máx | VRAM |
|---|---|---|---|---|---|---|---|
| 32³ (intento 1) | 32.768 | 0 / 20 | 1500 s (timeout) | — | — | 0,1 % / 94 % | 2410 MB |
| 32³ (intento 2) | 32.768 | **2 / 20** | 1500 s (timeout) | **451,1** | **1,38e-02** | 0,1 % / 94 % | 2410 MB |

Extrapolación a partir de 451 s/paso: ~9.000 s (**2 h 30 min**) para los 20 pasos
en una malla de 32³ (32.768 celdas).

Observaciones:
- `GFX` medio **0,1 %** con picos del **94 %** → trabajo en ráfagas muy cortas.
- El proceso `icoFoam` ~100 % de **un** núcleo de CPU → gran parte del tiempo es orquestación/overhead.
- El CSV de esta campaña (`exp_scaling_gpu/results_gpu.csv`) tiene las columnas
  desalineadas respecto a su cabecera (bug del script antiguo); se conserva como
  evidencia pero **no** se usa para cálculos. La campaña GAMG usa un formato nuevo y validado.
- El script antiguo contaba pasos *iniciados* (`Time = ...`) como completados, así que
  "2 / 20" es una **cota superior** de progreso. La campaña GAMG cuenta pasos *cerrados*
  (líneas `ExecutionTime`).

> **Nota de escala (importante).** Los 451 s/paso son a 32³ = 32.768 celdas, ~250× por
> debajo de las mallas donde se publican los resultados de la rama GPU. En MI300A,
> `cavity` con PCG **no muestra ganancia a 8M celdas** y solo alcanza **×2 a 64M**.
> Es decir: el régimen donde el offload paga (64M) requiere del orden de 100 GB de
> memoria y **no cabe en los 64 GB de la MI210**; y en el régimen alcanzable (≤8M) la
> referencia publicada tampoco gana. Este número caracteriza el **runtime en mallas
> pequeñas** (iteraciones normales, ~3,4 s por iteración), **no** se puede extrapolar
> a 8M: con 133 iteraciones/paso y 3,4 s por iteración, el coste es de orquestación,
> no de cómputo.

> **Nota sobre la referencia CPU:** la comparación informal inicial (cavity 20×20×1 en
> 0,055 s) no está respaldada por ningún log de esta máquina, así que **no se usa** en las
> conclusiones. La línea base CPU válida es el barrido de §8.1, medido con el mismo caso,
> el mismo build y las mismas métricas que la GPU.

---

## 8. Resultados con la configuración recomendada (GAMG + twoStageGaussSeidel)

### 8.1 Validación del caso (por qué esta vez el caso es 3D de verdad)

`checkMesh` a 32³:

```
points: 35937   faces: 101376   internal faces: 95232   cells: 32768
boundary patches: 2
Boundary definition OK.  Number of regions: 1 (OK).
movingWall  1024 faces  ok    fixedWalls  5120 faces  ok
Mesh has 3 geometric (non-empty/wedge) directions (1 1 1)
Mesh has 3 solution (non-empty) directions (1 1 1)
Mesh non-orthogonality Max: 0 average: 0     Mesh OK.
```

Contraste con el caso inválido del intento anterior: `2 solution directions (1 1 0)`
y `Failed 2 mesh checks`.

Matiz metodológico: `Uy = Uz = 0` en el **primer** paso es correcto, no un fallo.
En t=0, `U=(0 0 0)` y la tapa solo impone componente x, así que las ecuaciones de
y/z tienen fuente nula y residual inicial 0 por construcción. La prueba de que es
3D es que `Uz` existe y se resuelve: en el paso 2 da `Initial residual = 0.729402,
4 iteraciones`.

### 8.2 Medición oficial (20 pasos objetivo, GPU 0, timeout 1800 s)

| nx | celdas | pool | estado | pasos cerrados | wall | exec_last | s/paso | s/celda/paso | GFX med/máx | VRAM pico |
|---|---|---|---|---|---|---|---|---|---|---|
| **32** | 32.768 | `managed` | **timeout** | **0 / 20** | 1800 s | — | — | — | 0,1 % / 99 % | 2161 MB |

La escalera se detuvo aquí (`check_stop.py`), por diseño: no se lanzaron
48³, 64³, 96³ ni 128³.

Con `managed` a 32³, `icoFoam` **no cerró ni un paso de tiempo en 30 minutos**:
resolvió `Ux` (4 iters), `Uy`/`Uz` (0 iters), la primera `p` (4 iters, `relTol 0.01`)
y se quedó en el `pFinal` (`tolerance 1e-6`) hasta el SIGINT.

### 8.3 Sondas de diagnóstico (fuera de protocolo, `diagnostics_pool.csv`)

| nx | celdas | pool | estado | pasos cerrados | wall | exec_last | s/paso | s/celda/paso | GFX med/máx | VRAM pico |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 32.768 | `host` | timeout | 0 / 1 | 900 s | — | — | — | 0,4 % / 100 % | 1465 MB |
| 32 | 32.768 | sin pool | timeout | 1 / 4 | 600 s | 162,68 s | 162,7 (paso 1) | 4,96e-03 | 0,2 % / 99 % | 1438 MB |
| 16 | 4.096 | sin pool | **ok** | 4 / 4 | **19 s** | 18,66 s | **3,28** | **8,01e-04** | 5,3 % / 100 % | 1429 MB |

**`managed` vs `host`:** ambos mueren en el mismo punto (`pFinal` a 32³), 0 pasos en
1800 s y 900 s respectivamente. No hay evidencia de que el pool de Umpire ayude; a 16³
`managed` tampoco pasa del `pFinal` en 180 s, mientras que **sin pool** hace 4 pasos en 19 s.
⇒ En esta pila el pool es contraproducente.

**Dato más elocuente de todos:** en la corrida de 16³ (la única que terminó), de
**19 muestras** de monitorización **solo 1** registró GFX > 0 % (media 5,3 %). Es decir,
durante ~18 de los 19 segundos la GPU estuvo **completamente ociosa** mientras `icoFoam`
trabajaba en un núcleo de CPU. Lo mismo con `none` a 32³: 591 de 592 muestras a 0 %.

### 8.4 ¿Escalado sano?

No. Único par comparable (misma configuración, `none`):

| | celdas | s/paso | s/celda/paso |
|---|---|---|---|
| 16³ | 4.096 | 3,28 | 8,01e-04 |
| 32³ | 32.768 | ≥162,7 (solo paso 1) | ≥4,96e-03 |

8× más celdas → **~50×** más tiempo por paso y **6,2×** peor coste por celda.
El crecimiento es superlineal, así que la extrapolación a 48³/64³ no es defendible:
solo los 20 pasos de 32³ ya costarían >1,5 h en el mejor de los casos medidos.

### 8.5 Línea base CPU (release oficial v2606, `Gcc`, MPI)

Mismo caso (mismo `gen_case.py`), misma malla, 20 pasos, misma métrica (mediana de
los pasos 2..20 vía `ExecutionTime`). Build: `linux64GccDPInt32Opt` con gcc 13.3 +
OpenMPI 4.1.6, `decomposePar` método `simple`.

**Única diferencia de configuración:** `twoStageGaussSeidel` **no existe** en la
release oficial (los smoothers válidos son `DIC DICGaussSeidel FDIC GaussSeidel
nonBlockingGaussSeidel symGaussSeidel`), así que en CPU se usa `symGaussSeidel`, el
smoother simétrico estándar de OpenFOAM para GAMG. Es la configuración adecuada para
CPU, igual que `twoStageGaussSeidel` lo es para GPU.

| Malla | Celdas | 1 núcleo | 2 | 4 | 8 | 16 | 32 | 64 | Speedup máx. |
|---|---|---|---|---|---|---|---|---|---|
| 16³ | 4.096 | 0,007 s | — | — | — | — | — | — | — |
| 32³ | 32.768 | **0,050 s** | 0,030 | 0,020 | 0,010 | 0,010 | 0,010 | — | 5,0× (8 núcleos, techo de cuantización) |
| 64³ | 262.144 | **0,570 s** | 0,315 | 0,170 | 0,085 | 0,055 | 0,040 | 0,040 | 14,3× (32 núcleos) |
| 128³ | 2.097.152 | **8,750 s** | 4,240 | 2,215 | 0,960 | 0,540 | 0,330 | 0,260 | 33,7× (64 núcleos, 52 % de eficiencia) |

Nota metodológica: a 32³ el paso dura 0,05 s y `ExecutionTime` solo imprime 2 decimales,
así que de 8 núcleos en adelante el valor se cuantiza a 0,01 s. Por eso la curva se
repitió en 64³ y 128³, donde la resolución es real.

### 8.5.1 ¿Cuántos núcleos de CPU equivale la GPU 0?

| Malla | CPU 1 núcleo | GPU 0 (mejor caso medido) | Factor |
|---|---|---|---|
| 16³ | 0,007 s/paso | 3,28 s/paso (sin pool) | **~470× más lento** |
| 32³ | 0,050 s/paso | ≥162,7 s/paso (sin pool) | **≥3.254× más lento** |
| 32³ | 0,050 s/paso | >1800 s para <1 paso (`managed`) | **>36.000× más lento** |
| 64³ | 0,570 s/paso | no completa ningún paso | — |
| 128³ | 8,750 s/paso | no se intentó (escalera detenida) | — |

**Respuesta: menos de un núcleo — aproximadamente 0,0003 núcleos** (1/3.254) en el
mejor caso, y menos de 1/36.000 con el pool recomendado. No es que la GPU sustituya a
N núcleos: es que en esta pila equivale a una fracción minúscula de **uno solo**.
Incluso en la única malla que la GPU completó (16³), una CPU de un núcleo es ~470× más rápida.


### 8.6 Veredicto de §8

Incluso con la configuración que la documentación oficial marca como la correcta
(`GAMG` + `twoStageGaussSeidel`, sin precondicionadores secuenciales), el resultado
en esta pila es: **32.768 celdas no completan un paso de tiempo en 30 minutos**, con
la GPU ociosa el 99 % del tiempo. No es un problema de elección de solver: es el
coste por llamada de `hipstdpar` + paginación SVM.

---

## 9. Microbenchmarks de `hipstdpar` (contexto)

`std::fill` / `std::transform` con `par_unseq`, 10 M doubles:

| Prueba | hipstdpar | hipstdpar + `--hipstdpar-interpose-alloc` | CPU (TBB) |
|---|---|---|---|
| fill (1.ª vez) | 1,46 s | 0,40 s | 0,003 s |
| fill (2.ª/3.ª vez, mismo buffer) | 0,017 / 0,002 s | 0,24 s | 0,003 s |
| transform | 1,8–7,4 s | 0,23–0,28 s | 0,003 s |
| 100 × fill de 100k | 1,41 s (1.ª) / 0,0012 s (después) | 0,25 s | 0,014 s |
| **first touch 10 M** | **1,99 s** | 0,26 s | 0,005 s |
| **second touch 10 M** | **0,0002 s** | 0,23 s | 0,002 s |

Lectura: el **primer acceso** a un buffer es carísimo (paginación SVM) y el coste
**por llamada** es alto; con `interpose-alloc` se estabiliza pero sigue siendo
más lento que CPU. Un solver CFD genera muchísimas llamadas y campos nuevos →
paga ambos costes continuamente.

Prueba del test oficial `applications/test/std_execution` (2 M elementos):
secuencial 0,0068 s vs paralelo 59,27 s.

---

## 10. Conclusiones (preliminares)

1. **El build es correcto**: la GPU se usa de verdad (kernels HIP + actividad medida en GPU 0).
2. **El código GPU no está en el tarball oficial**: hay que usar la rama `feature-gpu`.
3. **La configuración del solver es necesaria pero no suficiente**: `PCG+DIC` es inviable
   con offload (~451 s/paso a 32³); `GAMG+twoStageGaussSeidel` es la config correcta y
   aun así **no completa 1 paso en 30 min**. El álgebra se arregla; el runtime no.
4. **El overhead de `hipstdpar` con SVM/XNACK en ROCm 7.2.4 es el factor limitante**:
   first-touch de páginas + coste por lanzamiento. Evidencia: GPU ociosa en 18 de 19
   muestras a 16³ mientras la CPU está al 100 % de un núcleo.
5. **El pool de Umpire es contraproducente en esta pila**: `managed` y `host` se atascan
   en `pFinal` a 32³; sin pool se completa 1 paso (162,7 s). A 16³, `managed` tampoco pasa
   del `pFinal` en 180 s frente a 4 pasos en 19 s sin pool.
6. **Las curvas de speedup de la página oficial de v2606 no se reproducen aquí.** Las
   diferencias conocidas: ROCm 7.2.4 (la documentación se publicó sobre una versión
   anterior), gfx90a con `HSA_XNACK`, y Umpire compilado con HIP en modo pool.
7. **Equivalencia medida: la GPU 0 ≈ 0,0003 núcleos de CPU** en 32³ (1/3.254), y
   <1/36.000 con el pool recomendado. La CPU escala de forma sana en cambio:
   33,7× con 64 núcleos en 128³ (52 % de eficiencia).
8. **Hallazgo colateral:** `twoStageGaussSeidel` **no existe** en la release oficial
   v2606; es una adición de la rama GPU. La comparación CPU/GPU no puede usar el
   mismo smoother, sino el mejor de cada plataforma (`symGaussSeidel` vs `twoStageGaussSeidel`).
5. Requisitos operativos no documentados en los apuntes iniciales:
   `HSA_XNACK=1` y `FOAM_SIGFPE=false`.

---

## 11. Recomendaciones

| Prioridad | Acción | Motivo |
|---|---|---|
| 1 | **No usar esta rama en producción** con ROCm 7.2.4 + gfx90a | No completa un paso de 32³ en 30 min |
| 2 | Reintentar en la versión de ROCm de la documentación (serie 6.x) | La brecha puede ser del runtime, no del código |
| 3 | Si se reintenta: **sin pool de Umpire** | `managed`/`host` se atascan en `pFinal`; sin pool al menos avanza |
| 4 | Escalar en CPU con MPI + `GAMG` + `simple`/`scotch` | Es lo que funciona hoy en esta máquina |
| 5 | Reportar upstream con el caso mínimo reproducible | `case_32cube` + logs + monitor son autocontenidos |
| 6 | Revisar v2612 | La integración en mainline está anunciada para esa release |

Caso mínimo para un reporte upstream:

```bash
source /openfoamGPU/env_gpu.sh
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export FOAM_MEMORY_POOL='managed; size=1024; incr=1024'
cd /openfoamGPU/exp_scaling_gpu_gamg/case_32cube
blockMesh && icoFoam        # 20 pasos, Re=100, GAMG+twoStageGaussSeidel
```

Resultado esperado en esta pila: se queda en `pFinal` del paso 1 (>30 min), GPU ociosa.

---

```bash
# 1. entorno
source /openfoamGPU/env_gpu.sh

# 2. caso
python3 /openfoamGPU/exp_scaling_gpu_gamg/gen_case.py 32 20 /tmp/case32
cd /tmp/case32 && blockMesh && checkMesh && icoFoam

# 3. solo GPU 0
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export FOAM_MEMORY_POOL='managed; size=1024; incr=1024'
```

---

## 12. Ruta alternativa medida: solver GPU externo (rocALUTION) — sí funciona

La rama `feature-gpu` descarga el álgebra con `--hipstdpar` y en esta pila es inviable (§7–§8).
Para separar "GPU lenta" de "runtime roto" se construyó una **segunda vía**: un solver lineal
enchufable (`solver rocALUTION;`) que se carga con `libs (...)` en el OpenFOAM oficial y ejecuta
**solo el solve lineal** en la GPU. Detalle completo en **`POC_ROCALUTION_V2606.md`**.

| Malla | Celdas | CPU 1 core | CPU 64 cores | **GPU 0 (p+U)** | GPU / 1 core | GPU / 64 cores |
|---|---|---|---|---|---|---|
| 32³ | 32.768 | 0,050 s | 0,010 s | **0,090 s** | 0,56× | 9× más lenta |
| 64³ | 262.144 | 0,570 s | 0,040 s | **0,470 s** | 1,2× | 12× más lenta |
| 128³ | 2.097.152 | 8,750 s | 0,260 s | **4,500 s** | **1,9×** | **17× más lenta** |

- **Corrección:** ΔU ≤ 5,4e-6 y Δp ≤ 4,3e-5 frente a `GAMG` (32³ y 128³; misma malla, mismos esquemas).
- **VRAM:** 6,7 GB a 128³ (4 jerarquías AMG) de 64 GB → margen para ~10⁸ celdas.
- **Coste:** a 32³ es **5.000× más rápido** que la ruta `hipstdpar` medida (451 s/paso).
- **Por qué no gana:** el 70 % del paso es ensamblado serie y el 27 % preparación/transferencia CSR;
  el cálculo en GPU es el **2 %**. Amdahl pone el techo en ~1,4 núcleos-equivalentes aunque la GPU
  fuera gratis.

**Conclusión:** el hardware y rocALUTION están bien; el problema de la rama `feature-gpu` es el
runtime `hipstdpar` (o su combinación con ROCm 7.2.4/gfx90a), no la GPU. Para producción en este
nodo sigue ganando CPU MPI; la vía GPU solo tiene sentido si se descarga también el ensamblado.

---

## 13. Ruta B medida: MPI + AMG en GPU como preconditionador

El solver de §12 resuelve el bloque **local** de cada rank: en serie es exacto, en MPI sería un
block-Jacobi que no converge al residual global. La variante correcta invierte los papeles:
**OpenFOAM conserva el Krylov global** (reducciones, interfaces, criterio de convergencia) y la GPU
aporta solo `M⁻¹` (un ciclo V de AMG) dentro de `PCG` / `PBiCGStab`. Implementado como
`preconditioner rocALUTION;` (nuevo `rocALUTIONLduPreconditioner`, misma caché CSR/AMG que el solver).

Medido en 128³ (2,1 M celdas, 20 pasos, GPU 0). Tiempos **de pared**:

| Ranks | CPU `GAMG` | GPU precond. | GPU / CPU | Iteraciones `p` |
|---|---|---|---|---|
| 8 | 0,96 s/paso | **9,7 s/paso** (194 s) | **10× más lenta** | 11 → 111 |
| 16 | 0,54 s/paso | **~42 s/paso** (792 s en 19 pasos) | **77× más lenta** | 11 → 149 |
| 32 | 0,33 s/paso | no medido (tendencia monótona) | — | — |
| 64 | 0,26 s/paso | (1 sola GPU; no escala) | — | — |

Tres causas estructurales: (a) *additive Schwarz* sin corrección gruesa pierde convergencia al
crecer el número de subdominios (11 → 149 iteraciones); (b) el setup de AMG se paga **por rank y
por campo** (0,5–3,6 s × 4–5 builds/paso) y la GPU no acelera esa fase; (c) cada rank añade un
contexto HIP sobre la misma GPU, de modo que el coste crece de forma superlineal (2× ranks ⇒ 4,3×
tiempo).

La variante "reunir la matriz y resolver en un único contexto" tampoco puede ganar: con las fases
ya medidas a 128³ (ensamblado 0,20 s en 32 ranks + CSR 0,40 + setup AMG 1,05 + solve 0,15 + gather
≥0,3) el suelo es **≈2,1 s/paso** frente a **0,33 s/paso** de `GAMG` con 32 núcleos; incluso
cacheando la jerarquía (solo `ReBuildNumeric`, 0,21 s) quedaría en ~1,2 s/paso, 3,7× por encima.
El setup de AMG, intrínsecamente secuencial, ya supera él solo el presupuesto completo del paso en CPU.

**Conclusión:** la ruta B es correcta y ejecutable, pero no es más rápida que CPU MPI en ninguna
configuración. El resultado negativo queda medido por tres vías independientes: port completo
`hipstdpar` (§7–§8), solver GPU en serie (§12) y preconditionador GPU en MPI (§13). Detalle
completo, código y reproducción en **`OPCION_B_MPI_GPU.md`**.

---

## 14. Ruta C medida: Krylov residente en GPU con multigrid de jerarquía fija

§13 dejó claro que el problema no era la GPU sino la orquestación: 23 ms por iteración para
bloques de 262 k celdas, con un *round trip* host↔GPU en cada una. Para separar hardware de
integración se exportó la **matriz real de `p`** desde el plugin (`rocDump`, 191 MB) y se midió con
tres bancos HIP independientes, sin OpenFOAM ni MPI. Detalle completo en
**`KRYLOV_RESIDENTE_GPU.md`**.

Malla 128³ = 2.097.152 celdas / 14,58 M nnz. Un único proceso, GPU 0, todo residente en VRAM
(por iteración solo cruzan 3 escalares). Reducción de 2 órdenes de magnitud (≈`relTol 0.01`):

| Preconditioner | Iteraciones | Tiempo | ms/iter | Setup |
|---|---|---|---|---|
| Jacobi (sin setup) | 298 | 168,0 ms | 0,56 | 0 |
| ILU(0) 2 colores | 152 | 261,1 ms | 1,72 | 181 ms |
| **Multigrid agregación 2×2×2** | **14** | **28,3 ms** | 2,02 | **1,0 ms** |

Comparación correcta: OpenFOAM hace **dos** resoluciones de presión por paso (`p` con `relTol 0.01`
y `pFinal` hasta 1e-6), así que la GPU ejecuta la misma cadena en caliente: 12 iteraciones hasta
1e-2 (24,1 ms) + 27 iteraciones hasta 1e-6 (56,4 ms) = **80,5 ms**.

| Plataforma (bloque `p` = `p`+`pFinal`, 128³) | Tiempo | Iteraciones | ms/iter | vs GPU |
|---|---|---|---|---|
| CPU 1 núcleo (`GAMG`) | 3.513 ms | 16,5 | 213 | **43,6× más lenta** |
| CPU 32 rangos (`GAMG`) | 138,5 ms | 13,45 | 10,3 | **1,72× más lenta** |
| CPU 64 rangos (`GAMG`) | 106 ms | — | — | 1,32× más lenta |
| **GPU 0 (Krylov residente + MG)** | **80,5 ms** | 39 | 2,06 | — |

(Método y verificación en **`COMPARACION_PRESION_GPU_CPU.md`**; corrección de un error de
comparación previo, que usaba 28 ms de un solo solve frío contra el bloque completo de CPU.)

Cuatro hechos que fijan la interpretación:

1. **SpMV a 951 GB/s** (0,219 ms, 60 % del pico). El *suelo* por iteración es memoria + latencia:
   0,56 ms con Jacobi, de los cuales 0,22 son SpMV y el resto 3 puntos con sincronización.
2. **El preconditioner es el 90 % de la historia**: mismo hardware, 168 → 28 ms solo por pasar de
   Jacobi a un ciclo V de agregación. El *setup* del multigrid es **1,0 ms** (sumas de Galerkin
   exactas, sin coarsening RS/PMIS ni SpGEMM) frente a 1,05 s **por matriz** del AMG de rocALUTION.
3. **ILU(0) no compensa en GPU**: su *apply* cuesta 1,0 ms (la permutación por colores rompe la
   localidad) y con solo 2 colores degenera en un LU por bloques con Schur diagonal (152 iters).
4. **La jerarquía ocupa ~130 MB** a 128³ frente a 6,7 GB del RS-AMG de §12: el tipo de AMG cambia
   el consumo en un factor ~50, no la malla.

**Qué implica a nivel de paso.** A 128³ / 32 ranks el paso son 0,352 s (medido por diferencia de tres
corridas): ensamblado 0,110 + `p` 0,1385 + `U` 0,103. Si la GPU hace el bloque `p` (80,5 ms): 0,294 s
⇒ **1,20×**; si hace `p` y `U` (U en GPU estimado): ≈1,5×; si el solve fuera gratis: **3,19×**
(techo de Amdahl). Ese
1,20× exige integración de coste nulo: reunir la matriz (≥0,3 s) o reconstruir el CSR (0,40–0,45 s)
ya lo borran por sí solos — que es exactamente por qué la ruta B daba 10–77× *más lenta* aunque el
solve fuese rápido.

**Conclusión:** la GPU no equivale a "dos núcleos", pero tampoco a cientos: con el bucle residente y
un multigrid barato el bloque de presión es **1,72× más rápido que 32 núcleos EPYC** y 43,6× más que
uno, y **cuantos más rangos, menos ventaja** (1,32× a 64 rangos). Por iteración es 5× más rápida; el
multigrid barato necesita 2,9× más iteraciones. Lo que impide que eso llegue al paso es el
**ensamblado en CPU** (31–45 % del paso, y el 70 % en el solver en serie de §12). Superar la unidad a
esta escala exige descargar también el ensamblado, que es precisamente la vía del benchmark publicado
en MI300A (×2, y solo desde 8–64 M celdas).

---

### 14.3 Contraste con el hilo del foro (L40S) y efecto del tamaño

Ver **`RESPUESTA_COMENTARIO_FORO.md`**. Tres resultados nuevos:

1. **FP64 no es el límite** (aunque en una L40S, con 1:64, lo parezca): pasar los coeficientes de la
   matriz a FP32 acelera el bloque `p` solo **1,37×** (80,5 → 58,8 ms), no 64×, y además degrada el
   residual real a 9,4e-6 (por encima de la tolerancia de `pFinal`). El kernel está limitado por
   **bytes movidos**, no por aritmética.
2. **El PCIe sí lo es**: enlace confirmado Gen4 x16 (16 GT/s ×16) con **28 GB/s por sentido** y
   11,7 µs de latencia. Un *round trip* de un vector cuesta **1,2 ms** frente a 2,06 ms de iteración:
   un ida-y-vuelta por iteración duplica el coste. Medido con `gpu_krylov/bench_pcie.cpp`.
3. **El tamaño ayuda despacio**: al pasar de 128³ a 256³ (8× celdas), la ventaja de la GPU en el
   bloque de presión sube de **1,72× a ≈3,5×**, pero la del paso completo solo de **1,20× a ≈1,30×**,
   porque el ensamblado y `U` crecen aún más (19× y 32× respectivamente).

---

## Anexos

- Logs del build: `/openfoamGPU/logs/` (19 archivos)
- Build y uso general: `/openfoamGPU/README_BUILD.md`
- Medida directa CPU vs GPU del bloque de presión: `COMPARACION_PRESION_GPU_CPU.md`
- Contraste con el hilo del foro (L40S), FP32 y PCIe: `RESPUESTA_COMENTARIO_FORO.md`
- **Traspaso para quien continúe: `TRASPASO.md`** (build, ejecución, trampas, pendientes)
- Memoria unificada en la MI210, medida: `MEMORIA_UNIFICADA_MI210.md`
- Scripts reproducibles: `repro/` (ver `repro/README.md`)
- Bancos de medida: `gpu_krylov/bench_mg.cpp` (PCG+MG, `-DUSE_FLOAT_COEFFS`, `WARM=`, `XDUMP=`),
  `gpu_krylov/bench_pcie.cpp` (ancho de banda host↔GPU)
- Experimentos GPU: `/openfoamGPU/exp_scaling_gpu_gamg/`
  - `results_gpu_gamg.csv` — medición oficial (1 fila: 32³ timeout)
  - `diagnostics_pool.csv` — sondas de pool (32³ host, 32³ sin pool, 16³ sin pool)
  - `logs/` — log y monitor de cada corrida
  - `case_32cube/` — caso mínimo reproducible
- Experimentos CPU: `/openfoamGPU/exp_scaling_cpu/`
  - `results_cpu_gamg.csv` — barrido de núcleos (16³/32³/64³/128³)
  - `scaling_cpu_vs_gpu.png` — gráfico tiempo vs núcleos con la referencia GPU
  - `build_cpu.sh`, `run_cpu.sh`, `gen_decomp.py`, `analyze_cpu.py`, `plot_cpu.py`
  - build en `/openfoamGPU/cpu/OpenFOAM-v2606` (`linux64GccDPInt32Opt`)
- Campaña previa con `PCG+DIC` (§7): `/openfoamGPU/exp_scaling_gpu/`
  - `results_gpu.csv` (columnas desalineadas, se conserva como evidencia)
  - `results_gpu.csv.corrupt_*` — corridas descartadas por contención de GPU
- PoC con solver GPU externo (§12): `/openfoamGPU/POC_ROCALUTION_V2606.md`
  - `rocALUTION-solver/` — plugin, casos, scripts de medida, `poc_scaling.png`
  - `rocALUTION-solver/results_roc_poc.csv` — tabla de resultados
  - `OPCION_B_MPI_GPU.md` + `rocALUTION-solver/results_mpi_precond.csv` — ruta MPI + GPU (§13)
  - `rocALUTION-solver/rocALUTIONLduPreconditioner.{H,C}` — preconditionador AMG en GPU
- Krylov residente en GPU (§14): `/openfoamGPU/KRYLOV_RESIDENTE_GPU.md`
  - `gpu_krylov/bench_pcg.cpp` — PCG + Jacobi sobre la matriz real de `p`
  - `gpu_krylov/bench_ilu.cpp` — coloración + permutación + ILU(0) multicolor
  - `gpu_krylov/bench_mg.cpp` — multigrid de agregación (Galerkin exacto) + diagnósticos
  - `gpu_krylov/results_krylov_residente.csv` — tabla de resultados
