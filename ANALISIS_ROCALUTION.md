# Análisis de factibilidad: sustituir AMGx por rocALUTION + rocSPARSE en OpenFOAM v2606

**Fecha:** 2026-10-26
**Máquina:** 1× AMD Instinct MI210 (gfx90a, 64 GB), ROCm 7.2.4, EPYC 7763 (128 hilos)
**Restricción:** GPU 0 únicamente (`ROCR_VISIBLE_DEVICES=0`). Sin PETSc (requisito explícito).
**Alcance:** integrar un solver lineal GPU *externo* en `lduMatrix` de OpenFOAM, en la línea de lo que
hace la integración de AMGx (NVIDIA), pero con librerías AMD.

> **ACTUALIZACIÓN (PoC ejecutada).** Este documento es el análisis previo. La PoC ya está
> implementada y medida: ver **`POC_ROCALUTION_V2606.md`** (solver `rocALUTION`, `cavity3D` completo
> en GPU 0, validado contra CPU). Resumen: 0,090 / 0,470 / **4,500 s/paso** a 32³/64³/128³, VRAM pico
> 6,7 GB a 128³, y **solo el 2 % del paso es cálculo en GPU** (el 98 % es ensamblado y marshalling en
> CPU). Una MI210 equivale a **≈ 1,9 núcleos** de este nodo a 128³ (no sustituye a MPI).
> Las tablas de §4 y las estimaciones de §8/§11 quedan sustituidas por las medidas de la PoC.

---

## 1. Veredicto en una página

| Pregunta | Respuesta |
|---|---|
| ¿Están rocALUTION y rocSPARSE disponibles en esta máquina? | **Sí**: rocALUTION 4.1.0 y rocSPARSE 4.2.0 (ROCm 7.2.4), backend HIP funcionando en MI210. |
| ¿Funciona su AMG en gfx90a? | **Sí**: Ruge-Stüben, 5–7 niveles, converge en 7–10 iteraciones. Resultado correcto (residuo 1e-7). |
| ¿Es rápido el *solve*? | **Sí**: 2,1 M incógnitas en **49,5 ms**; 262 k en 16,7 ms; 32 k en 9,5 ms. |
| ¿Es rápido el *setup* de la jerarquía AMG? | **No**: 574 / 377 / **1.855 ms** para 32³ / 64³ / 128³. **El setup domina 10–40× el solve.** |
| ¿Se puede reutilizar la jerarquía entre pasos de tiempo? | **Sí**: `ReBuildNumeric()` funciona y cuesta **2,8–6 % del `Build()`** (medido a 128³: 48,7 ms vs 1.695 ms). |
| ¿Soporta MPI el paquete instalado? | **No**: `info_rocalution()` dice "MPI is not initialized"; el `.so` no enlaza `libmpi`. Multi-GPU exige compilar de fuente con MPI. |
| ¿Sustituye a AMGx funcionalmente? | **Casi**: cubre AMG + Krylov + precondicionadores; le faltan reutilización de jerarquía, MPI empaquetado y precisión mixta. |
| ¿Merece la pena en esta máquina hoy? | **Sí como PoC acotada**: con `ReBuildNumeric()` el coste por paso baja a ~117 ms a 128³ (vs 1,9 s reconstruyendo). El límite que queda es Amdahl (solo se acelera p), no el setup. |

**Conclusión corta:** la ruta es **técnicamente factible** (existe la API, el hardware responde y los
números del solve son buenos). La reutilización de jerarquía mediante `ReBuildNumeric()` **funciona y
elimina el principal escollo** (ver §4.1), pero el diseño no puede ser "recordar el estado en miembros
del solver" porque OpenFOAM destruye el objeto en cada llamada (ver §5.0). El límite restante es
Amdahl: solo se acelera el solve de `p`, no el ensamblado ni el resto del paso.

**Conclusión medida (PoC, `POC_ROCALUTION_V2606.md`):** el adaptador funciona y `cavity3D` corre
entero en GPU 0 con resultados correctos (ΔU ≈ 5e-6, Δp ≈ 4e-5 frente a GAMG). Pero el reparto real
del paso a 128³ es **70 % ensamblado + 27 % preparación CSR + 2 % GPU**, así que una MI210 rinde como
**1,9 núcleos** de este nodo. Amdahl se cumple: el techo de esta arquitectura (solver-only) es ~1,4
núcleos-equivalentes incluso con coste GPU nulo. Para competir con MPI hay que descargar también el
ensamblado (`hipstdpar`), que hoy no funciona en esta rama.

---

## 2. Qué hay instalado (verificado, no supuesto)

```
ROCm           7.2.4
hipcc          HIP 7.2.53211 / clang 22.0.0
rocALUTION     4.1.0.70204   (/opt/rocm/lib/librocalution.so.1)
rocSPARSE      4.2.0.70204   (/opt/rocm/lib/librocsparse.so.1)
hipSPARSE      4.2.0.70204
```

Capacidades reales de la API (leídas de `/opt/rocm/include`):

- **AMG**: `RugeStuebenAMG`, `SAAMG` (agregación suavizada), `UAAMG` (agregación no suavizada), `PairwiseAMG`.
- **Krylov**: `CG`, `FCG`, `GMRES`, `FGMRES`, `BiCGStab`, `BiCGStab(l)`, `CR`, `IDR`, `QMRCGStab`.
- **Precondicionadores**: ILU multicolor, GS multicolor, Block-Jacobi, FSAI/SPAI, multi-eliminación, saddle-point.
- **Distribuido**: `GlobalMatrix` / `ParallelManager` existen en la API, pero **el paquete no está compilado con MPI**.
- **rocSPARSE**: `csric0`, `csrilu0`, `csrsv`, `csritsv`, `gtsv` (factorizaciones incompletas y sustituciones).

---

## 3. Trampa crítica descubierta (documentar sí o sí)

rocALUTION crea **todos los objetos en host**. Si no se llama a `MoveToAccelerator()`, la librería
reporta `Accelerator backend: HIP` pero **ejecuta en el host con OpenMP**. El resultado es un
rendimiento catastrófico y silencioso:

| Operación (32³, 32.768 incógnitas) | Sin `MoveToAccelerator()` | Con `MoveToAccelerator()` |
|---|---|---|
| SpMV | 24,8 ms | **0,007 ms** |
| dot | 25,4 ms | — |
| axpy | 29,6 ms | — |
| Iteración de CG | 190 ms | **0,10 ms** |

Agravante: el umbral OMP por defecto es 10.000, así que por debajo de ese tamaño usa 1 hilo (parece
rápido) y por encima lanza 128 hilos; en este nodo cada barrera de 128 hilos cuesta ~25 ms:

| `OMP_NUM_THREADS` | SpMV 32³ |
|---|---|
| 1 | 0,143 ms |
| 8 | 0,032 ms |
| 32 | 5,998 ms |
| 128 (defecto) | 25,699 ms |

Cualquier integración debe (a) llamar a `MoveToAccelerator()`, (b) fijar `set_omp_threads_rocalution()`
y (c) **verificar con tiempos**, no con el mensaje de `info_rocalution()`.

---

## 4. Mediciones: rocALUTION AMG en GPU vs baseline CPU ya medido

PoC standalone que resuelve un Laplaciano 7 puntos N³ con Dirichlet (la estructura de la matriz de
presión de `icoFoam`): `poc_rocalution/poc_amg.cpp`, tolerancia relativa 1e-6, norma 2.

| Malla | Incógnitas | nnz | **Setup AMG** | **Solve AMG** | Iteraciones |
|---|---|---|---|---|---|
| 32³ | 32.768 | 223.232 | 574 ms | **9,5 ms** | 7 |
| 64³ | 262.144 | 1.810.432 | 377 ms | **16,7 ms** | 8 |
| 128³ | 2.097.152 | 14.581.760 | 1.855 ms | **49,5 ms** | 10 |

(El setup de 32³ sale mayor que el de 64³ porque el coarsening no es monótono; se reporta tal cual.)

Comparación con el baseline CPU de este mismo proyecto (`results_cpu_gamg.csv`, paso completo de
`icoFoam` = 4 solves lineales, no un solo solve):

| Malla | CPU 1 core | CPU 64 cores | Solve GPU (1 presión) | Setup GPU |
|---|---|---|---|---|
| 32³ | 0,050 s | 0,010 s | 0,0095 s | 0,574 s |
| 64³ | 0,570 s | 0,040 s | 0,0167 s | 0,377 s |
| 128³ | 8,750 s | 0,260 s | 0,0495 s | 1,855 s |

Lectura:

- **El solve GPU es bueno**: a 128³, 49,5 ms para 2,1 M incógnitas es del orden de 30–50× más rápido
  que un solve de presión en 1 core.
- **El setup lo arruina**: reconstruir la jerarquía cuesta 1,86 s a 128³, es decir, ~37× el solve.
  Pagado en cada paso de tiempo, el paso sería **más lento que el CPU de 64 cores** (0,26 s).
- **CG sin precondicionador** (para calibrar): 0,10–0,15 ms/iteración, 64 iteraciones a 32³. Es decir,
  el SpMV y las reducciones en GPU van bien; el problema es exclusivamente el AMG.

### 4.1 Reutilización de jerarquía: `UpdateValuesCSR()` + `ReBuildNumeric()` (medido)

El patrón CSR (row_ptr, col_idx) se congela y solo se actualizan los valores; después se llama a
`ReBuildNumeric()`. Simulación de 4 pasos de tiempo con escalado simétrico `S·A·S` (preserva SPD),
128³, `OMP_NUM_THREADS=8`:

| Precondicionador | `Build()` inicial | `ReBuildNumeric()` | % del Build | Solve | Iteraciones | Residuo |
|---|---|---|---|---|---|---|
| **RugeStueben** | 1.695 ms | **48,7 ms** | 2,9 % | **68,0 ms** | 13 (estable) | 7,0e-07 |
| **SAAMG** | 448 ms | **27,1 ms** | 6,0 % | 81,0 ms | 23 (estable) | 5,8e-07 |
| UAAMG (64³) | 152 ms | 3,8 ms | 2,5 % | 125 ms | 51 | 7,3e-07 |

Coste por paso a 128³ (actualización de valores + rebuild + solve):

| Estrategia | Coste/paso | vs reconstruir |
|---|---|---|
| Reconstruir jerarquía cada paso (RS) | ~1.863 ms | 1× |
| **`ReBuildNumeric()` (RS)** | **~117 ms** | **16× más barato** |
| **`ReBuildNumeric()` (SAAMG)** | **~108 ms** | 17× más barato |

Notas:

- La transferencia de valores (`UpdateValuesCSR`, 117 MB a 128³) **no** está incluida en la columna
  de ReBuildNumeric; se estima en ~5 ms a 23 GB/s (medido).
- Con perturbaciones no acumulativas las iteraciones se mantienen constantes (13 / 23) y el residuo
  es correcto: la jerarquía sigue siendo válida para coeficientes que cambian poco entre pasos.
- Con deriva acumulada (matriz que se aleja de la jerarquía) las iteraciones suben (13 → 23 en 3
  pasos). Recomendación: reconstruir la jerarquía completa cada K pasos o si las iteraciones superan
  un umbral.
- `UAAMG` converge mucho peor (51 iteraciones): descartado para `p`.

---

## 5. Arquitectura de integración en OpenFOAM (diseño concreto)

### 5.0 Restricción de diseño descubierta: el solver se destruye en cada llamada

`fvMatrix::solve()` construye el solver como temporal:

```cpp
// src/finiteVolume/fvMatrices/fvMatrix/fvMatrixSolve.C:219
solverPerf = lduMatrix::solver::New(...)->solve(psiCmpt, sourceCmpt, cmpt);
```

El `autoPtr` se libera al final de la expresión. Por tanto **cualquier estado guardado como miembro de
la clase solver (`isBuilt_`, `A_`, `amg_`) se pierde en cada `solve()`** y la jerarquía se
reconstruiría siempre. El estado persistente debe vivir fuera del objeto solver, con vida ligada a la
malla:

```cpp
// Cache por (malla, campo), propiedad de la malla -> se libera sola
class rocALUTIONCache
:
    public MeshObject<fvMesh, UpdateableMeshObject, rocALUTIONCache>
{
    struct Entry
    {
        const lduAddressing* addressing;   // identifica el patron
        label nCells;
        rocalution::LocalMatrix<double> A;
        rocalution::CG<...> cg;
        rocalution::RugeStuebenAMG<...> amg;
        bool built;
    };
    HashTable<Entry, word> entries_;       // clave: fieldName
    ...
};
```

Ventajas: sobrevive entre pasos, se invalida con cambios de malla (AMR/topología) y no requiere tocar
`fvMatrix`. Es el mismo patrón que usa OpenFOAM para `solutionControl` o los `MeshObject` de
turbulencia.

El punto de entrada es `Foam::lduMatrix::solver` (clase base con `solve(psi, source, cmpt)` y
`solveScalar`). Se añade un solver nuevo, registrado con `addToRunTimeSelectionTable`, sin tocar el
resto del solver de OpenFOAM:

```
src/OpenFOAM/matrices/lduMatrix/solvers/rocALUTION/
├── rocALUTIONSolver.H
├── rocALUTIONSolver.C      // adaptador
├── rocALUTIONCSR.C         // conversión LDU -> CSR persistente
└── Make/files
```

### 5.1 Conversión LDU → CSR

OpenFOAM guarda la matriz por caras: `diag[nCells]`, `upper[nFaces]`, `lower[nFaces]` (vacío si es
simétrica) y la addressing `lduAddr().upperAddr()/lowerAddr()`. El CSR se construye:

- `row_ptr` y `col_idx` dependen **solo de la topología** → se calculan una vez por malla y se quedan
  en device.
- `val` cambia cada paso → solo se reenvía el vector de valores (`nnz·8` bytes).
  A 128³: 14,6 M × 8 B ≈ **117 MB → ~5 ms** a 23 GB/s (medido con `poc_rocalution/hip_latency.cpp`).

Esto evita el error clásico de reconstruir el CSR entero (y reenviar índices) en cada `solve()`.

### 5.2 Semántica que hay que preservar

| Concepto OpenFOAM | Equivalente en rocALUTION |
|---|---|
| `tolerance` / `relTol` sobre residual normalizado | `InitTol()` + iteración externa, porque el criterio de OpenFOAM normaliza por `normFactor` |
| `solverPerformance` (initial/final residual, nIterations) | calcular residuo con `matrix_.Amul()` y mapear `GetIterationCount()` |
| `pRefCell`/`pRefValue` | ya viene plegado en `diag`/`source` antes de llamar al solver; no requiere nada |
| BCs y `processor` patches | ya plegadas en `diag`/`upper`/`source` por `fvMatrix`; en mono-proceso no hay nada extra |
| Simetría (`symMatrix`) | `lduMatrix::solver` se registra aparte para `symMatrix`; en simétrico solo hay `upper` |

### 5.3 Paralelismo

- **Mono-GPU / mono-proceso**: funciona ya con el paquete instalado (cada rango resuelve su matriz local).
- **Multi-GPU**: requiere `GlobalMatrix` + MPI, y el paquete **no trae MPI** → habría que compilar
  rocALUTION de fuente (`-DROCALUTION_MPI=ON`) y añadir los intercambios de halos. Es la parte más
  costosa del proyecto.

---

## 6. AMGx vs rocALUTION + rocSPARSE: qué se gana y qué se pierde

| Capacidad | AMGx (referencia NVIDIA) | rocALUTION 4.1 | Impacto |
|---|---|---|---|
| AMG clásico (Ruge-Stüben) | Sí | Sí | ✅ |
| AMG de agregación | Sí | Sí (`SAAMG`, `UAAMG`) | ✅ |
| Krylov (CG/GMRES/BiCGStab) | Sí | Sí (9 variantes) | ✅ |
| Precondicionadores locales | ILU, Jacobi, etc. | ILU/GS multicolor, FSAI, block-Jacobi | ✅ |
| **Reutilización de jerarquía** | `setup` separado de `solve` | ✅ `ReBuildNumeric()` medido: 49 ms vs 1.695 ms (2,9 %) | ✅ |
| MPI / multi-GPU | Sí | API sí, **paquete no** | ⚠️ |
| Precisión mixta | Sí | Parcial (`mixed_precision.hpp`) | 🟡 |
| Matrices por bloques (saddle-point) | Sí | `BlockMatrix`, `SaddlePoint` | ✅ |
| Determinismo bit a bit | No | No | ➖ |
| Madurez / soporte | Muy alta | Media, nicho HPC | 🟡 |
| Índices | 32/64-bit | API CSR en `int` (32-bit) | 🟡 límite ~2·10⁹ nnz |

**rocSPARSE solo** (sin rocALUTION) es otra opción: da SpMV, `csric0`/`csrilu0` y `csrsv`, suficiente
para escribir un PCG a mano (~300–500 LOC). Pero IC(0)/ILU(0) converge mal para Poisson (iteraciones
tipo O(N)); el AMG es precisamente la razón para usar rocALUTION. **Recomendación: rocALUTION para p,
rocSPARSE solo si se quiere evitar la dependencia.**

---

## 7. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Setup AMG domina el paso | **Resuelto** | — | `UpdateValuesCSR()` + `ReBuildNumeric()` (medido: 2,9 % del Build) |
| Estado persistente entre pasos | **Alta** | Alto | Cache en `MeshObject` (el solver se destruye por llamada) |
| Solo se acelera el solver (Amdahl) | **Alta** | Alto | Medir el reparto solver/ensamblado antes de invertir |
| Fallback silencioso a host | Alta | Muy alto | `MoveToAccelerator()` + test de tiempos en CI |
| Sin MPI en el paquete | Alta | Medio | Compilar de fuente o aceptar 1 GPU |
| Transferencia de 117 MB/paso a 128³ | Media | Medio | Patrón persistente + solo `val` (≈5 ms) |
| Convergencia distinta a GAMG | Media | Medio | Validar contra `icoFoam` con tolerancias equivalentes |
| rocALUTION es producto de nicho | Media | Medio | Abstraer tras la interfaz `lduMatrix::solver` (permite cambiar a otra librería) |
| Índices `int` en CSR | Baja | Bajo | Suficiente hasta ~10⁸ celdas |

---

## 8. Esfuerzo estimado

| Fase | Contenido | Duración |
|---|---|---|
| **0. PoC standalone** | AMG sobre Poisson N³ + verificación de `ReBuildNumeric()` | ✅ **hecho** (1 día) |
| **1. Adaptador mínimo** | `lduMatrix::solver`, CSR persistente, simétrico, mono-proceso, validación vs GAMG | **1–2 semanas** |
| **2. Rendimiento** | Reutilización de jerarquía, solvers U y p, perfilado del reparto de tiempo | **2–4 semanas** |
| **3. Producción** | MPI multi-GPU, todas las BCs, robustez, regresión | **1–2 meses** |

Total realista: **4–8 semanas** para algo usable en una GPU; **2–3 meses** para producción multi-GPU.

### Criterios go / no-go (medibles, antes de invertir más)

1. **Reparto de tiempo** a 128³, 1 core: si el solver lineal es < 40 % del paso → **no-go** (techo < 2×).
2. **Setup reutilizado**: si no se consigue bajar el setup a < 50 ms a 128³ → **no-go para p**.
3. **Amdahl**: si el paso completo con GPU no baja de 0,26 s (el CPU de 64 cores ya medido) → **no-go**.
4. **Corrección**: mismo campo de presión que GAMG dentro de tolerancia en 20 pasos del caso 32³.

---

## 9. Plan de PoC concreto (si se decide seguir)

1. Exportar la matriz real de presión desde `icoFoam` (32³ y 128³) a Matrix Market con un `lduMatrix`
   instrumentado; comparar con el Laplaciano del PoC (coeficientes no uniformes, BCs).
2. Medir el **reparto de tiempo** del paso: `p` vs `U` vs ensamblado (cronómetro en `fvMatrix::solve`).
3. Implementar Fase 1 (adaptador) y validar contra GAMG en el caso 32³ ya existente
   (`exp_scaling_gpu_gamg/case_32cube`).
4. Probar **reutilización de jerarquía**: construir una vez y resolver con matrices perturbadas
   (nuevo paso de tiempo) sin reconstruir; medir degradación de convergencia.
5. Solo si 3 y 4 salen bien: Fase 2 y comparación contra la curva CPU ya medida
   (`exp_scaling_cpu/results_cpu_gamg.csv`).

Ficheros de partida (ya en el repo):
- `poc_rocalution/poc_amg.cpp` — AMG (setup + solve + residuo verificado).
- `poc_rocalution/poc_ops.cpp` — coste por operación (SpMV, dot, axpy, CG).
- `poc_rocalution/hip_latency.cpp` — salud del runtime HIP (launch, memcpy, ancho de banda).
- `poc_rocalution/poc_backend.cpp` — backend efectivo y `info_rocalution()`.

---

## 10. Alternativas descartadas o de segundo orden

| Alternativa | Por qué no (o cuándo sí) |
|---|---|
| **PETSc + petsc4foam** | Excluido por requisito. Además en GPU arrastra hypre/AmgX y no simplifica nada. |
| **rocSPARSE solo (PCG + IC0)** | Menos dependencias, pero convergencia pobre en Poisson; útil solo para `U`. |
| **Seguir con CPU MPI** | Hoy es la línea base más rápida (0,26 s/paso a 128³ con 64 cores) y no tiene riesgo. |
| **Arreglar el offload `hipstdpar` de la rama** | Es el problema original; este análisis lo esquiva, no lo resuelve. |
| **Esperar a que OpenFOAM mainline soporte GPU** | Sin fecha; la rama `feature-gpu` es la única vía oficial y ya la medimos. |

---

## 11. Recomendación

> **Estado tras la PoC (`POC_ROCALUTION_V2606.md`):** los puntos 1–3 ya están decididos con medidas.
> Se implementó el adaptador completo (632 líneas), se validó (ΔU 5e-6, Δp 4e-5) y se midió el
> reparto: **70 % ensamblado, 27 % CSR/marshalling, 2 % GPU**. Resultado: una MI210 ≈ **1,9 núcleos**
> a 128³ y 17× más lenta que 64 cores. **Decisión: no-go para producción; sí como PoC y como línea de
> I+D** mientras el ensamblado no se descargue a GPU.

1. **No** sustituir GAMG por rocALUTION como solución de producción en esta máquina todavía.
2. **Sí** hacer la Fase 1 (2 semanas): el *solve* de 2,1 M incógnitas en 49,5 ms y la reutilización de
   jerarquía a 49 ms/paso demuestran que el álgebra en GPU funciona. El único bloqueo que queda es el
   reparto de tiempo (Amdahl), que se mide en dos días.
3. Si el criterio 1 (reparto) falla, cerrar el tema y quedarse con CPU MPI: es la decisión racional,
   no un fracaso.
4. Documentar la trampa `MoveToAccelerator()` en cualquier guía interna: es la causa más probable de
   "la GPU va lentísima" en futuras integraciones.
5. Documentar también que el solver de OpenFOAM **se destruye en cada llamada**: cualquier caché debe
   vivir en un `MeshObject`, no en miembros del solver.

---

## Anexo A. Reproducir

```bash
cd /openfoamGPU/poc_rocalution
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0

# runtime HIP sano
./hip_latency

# AMG completo (setup + solve + residuo)
OMP_NUM_THREADS=8 ./poc_amg_gpu 128 2

# coste por operacion y CG con iteraciones fijas
OMP_NUM_THREADS=8 ./poc_ops_gpu 32

# backend efectivo
./poc_backend
```

Compilación:

```bash
hipcc -O3 -std=c++17 -I/opt/rocm/include poc_amg.cpp -o poc_amg_gpu \
      -L/opt/rocm/lib -lrocalution -Wl,-rpath,/opt/rocm/lib
```

## Anexo B. Limitaciones de estas mediciones

- La matriz del PoC es un Laplaciano 7 puntos con Dirichlet; la matriz real de `icoFoam` tiene
  coeficientes variables y BCs de pared, así que los tiempos son indicativos (±20 %).
- Se mide **un** solve de presión, no el paso completo (que incluye 2 solves de `U` y 2 de `p`).
- El setup se mide con `OMP_NUM_THREADS=8`; con el valor por defecto (128) es aún peor.
- No se ha probado multi-GPU ni MPI (el paquete no lo soporta).
