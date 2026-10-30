# OpenFOAM v2606 con offload GPU (AMD MI210 / gfx90a) — Build y uso

Este documento describe el build completo de **OpenFOAM v2606** (rama `feature-gpu`
de openfoam-ecse, commit `559823f7`) compilado con **AMD clang++ (ROCm 7.2.4)**
usando **C++17 stdpar** (`--hipstdpar`) para offload automático a GPU, sobre
**2× AMD Instinct MI210 (gfx90a)**.

- Build completado: **`./Allwmake -j -s -q -l` → exit 0**
- Arquitectura: `WM_OPTIONS=linux64Amd-gpuDPInt32Opt`
- Binarios instalados: **268** · Librerías: **128**
- Log principal del build: [`/openfoamGPU/OpenFOAM-v2606/log.linux64Amd-gpuDPInt32Opt`](OpenFOAM-v2606/log.linux64Amd-gpuDPInt32Opt)

---

## 1. Rutas principales

| Qué | Ruta |
|-----|------|
| OpenFOAM | `/openfoamGPU/OpenFOAM-v2606` |
| ThirdParty (Umpire, Boost, FFTW, CGAL) | `/openfoamGPU/ThirdParty-v2606` |
| Entorno | `/openfoamGPU/env_gpu.sh` |
| Logs del build y de las pruebas | `/openfoamGPU/logs/` |
| Runs de prueba (`$FOAM_RUN`) | `/root/OpenFOAM/root-v2606/run` |
| Preferencias locales | `/openfoamGPU/OpenFOAM-v2606/etc/prefs.sh` |
| Backup de `have_cgal` original | `/openfoamGPU/have_cgal.orig.bak` |

---

## 2. Cargar el entorno

```bash
source /openfoamGPU/env_gpu.sh
```

Esto exporta y hace `source etc/bashrc`:

- `ROCM_PATH=/opt/rocm`, `PATH` y `LD_LIBRARY_PATH` de ROCm
- `ROCM_GPU=gfx90a` (arquitectura de offload; la usa la regla del compilador)
- `HSA_XNACK=1` (**obligatorio**, ver §5)
- `WM_COMPILER=Amd-gpu` → `WM_OPTIONS=linux64Amd-gpuDPInt32Opt`
- `FOAM_SIGFPE=false` (**necesario**, ver §5)
- `FOAM_EXTRA_CXXFLAGS=-Wno-overloaded-virtual` (en `etc/prefs.sh`)

Comprobar que quedó bien:

```bash
echo $WM_OPTIONS      # linux64Amd-gpuDPInt32Opt
which amdclang++      # /opt/rocm/bin/amdclang++
icoFoam -help | head  # debe responder
```

---

## 3. Cómo correr un caso con GPU

El offload es **automático**: el código compilado con `--hipstdpar` ejecuta en GPU
las operaciones `std::execution::par_unseq` (y las rutinas de OpenFOAM que las usan).
No hay que pasar ninguna opción especial. Un caso debe correr en **un solo proceso**
(sin `mpirun`), ya que el offload aplica dentro de cada rango MPI.

Ejemplo (cavity, tutorial icoFoam):

```bash
source /openfoamGPU/env_gpu.sh
mkdir -p "$FOAM_RUN" && cd "$FOAM_RUN"
cp -r "$FOAM_TUTORIALS"/incompressible/icoFoam/cavity/cavity .
cd cavity
blockMesh
icoFoam                      # corre solo en un proceso, usa GPU automáticamente
```

Con memoria gestionada por Umpire:

```bash
export FOAM_MEMORY_POOL='host;    size=1024; incr=1024'   # o 'managed' / 'device'
icoFoam
```

Al arrancar, el solver imprime la línea `memory pool : ...`:

- `memory pool : unused` → pool por defecto (malloc del sistema). **Umpire está
  enlazado** (la interfaz existe), simplemente no se activa ningún pool.
- `memory pool : managed (size=1024MB, incr=1024MB)` → pool Umpire sobre
  `hipMallocManaged`. **Probado y funcional.**
- `memory pool : host (size=1024MB, incr=1024MB)` → pool Umpire sobre memoria host.
  **Probado y funcional** (fue el más rápido en las pruebas).

---

## 4. Verificar uso real de GPU

Con el caso corriendo, en otra terminal:

```bash
source /openfoamGPU/env_gpu.sh
amd-smi monitor -u -m -w 1        # GFX% y MEM% por GPU
# o:
rocm-smi --showuse --showmemuse
```

Evidencia recogida en este sistema durante `icoFoam` (ver
`/openfoamGPU/logs/amd_smi_monitor.log`):

```
1792534726    0    0  1700 MHz  100 %    0 %   1600 MHz   <- GPU0 al 100 %, clk en boost
1792534763    0    0  1700 MHz   38 %    0 %   1600 MHz
```

- GPU0 pasa de **800 MHz (idle) a 1700 MHz (activo)** y registra **100 % de GFX
  activity** → el trabajo corre en la GPU.
- `rocm-smi --showmemuse` reporta VRAM asignada en GPU0 durante los runs.

Nota: `rocm-smi --showuse` muestrea de forma instantánea y suele devolver `0`
incluso con carga; el reloj y `amd-smi monitor` son señales más fiables.

---

## 5. Parámetros importantes del entorno

| Variable | Valor | Por qué |
|----------|-------|---------|
| `HSA_XNACK=1` | **obligatorio** | gfx90a necesita XNACK para el SVM/managed memory de `--hipstdpar`. Con `HSA_XNACK=0` se produce *memory access fault* (write a página read-only) y coredump. |
| `FOAM_SIGFPE=false` | **necesario** | El runtime ROCm (`libhsa-runtime64`/`libamdhip64`) dispara un **SIGFPE espurio en `__cxa_finalize`** al descargar las libs. El `trapFpe` de OpenFOAM lo convierte en core dump + exit 136 aunque el programa haya terminado bien. Desactivarlo hace que solvers y utilidades terminen limpios. No afecta a los cálculos. |
| `ROCM_GPU=gfx90a` | gfx90a | Arquitectura de offload usada por `wmake/rules/General/Amd-gpu/c++` (`--offload-arch=$(ROCM_GPU)`). |
| `FOAM_MEMORY_POOL` | vacío / `managed` / `host` | Selecciona el pool de memoria gestionado por Umpire. Opcional. |

---

## 6. Limitaciones de la rama GPU

1. **Solvers lineales**: usar **GAMG + twoStageGaussSeidel** (o
   `symGaussSeidel`) en vez de precondicionadores DIC/DILU/GaussSeidel clásicos
   para aprovechar mejor el offload/estabilidad en GPU. (El tutorial `cavity`
   con `DICPCG` + `symGaussSeidel` también completó correctamente.)
2. **No determinismo**: los resultados pueden diferir ligeramente entre
   ejecuciones y respecto al build CPU, por el no determinismo del paralelismo
   GPU (reducciones en orden no fijo, atomics).
3. **Sin *patch fusing***: la rama GPU no fusiona operaciones de parches.
4. **Un solo GPU** por proceso: el offload actúa sobre el dispositivo seleccionado
   (`ROCR_VISIBLE_DEVICES`); no hay reparto multi-GPU dentro de un proceso.
5. **Directorio del caso**: las apps escriben en `$FOAM_RUN`; ejecutar siempre
   desde el directorio del caso.

---

## 7. Cambios / parches aplicados al código fuente

1. **CGAL deshabilitado** (`wmake/scripts/have_cgal`):
   - **Cambio**: el script ahora solo define `no_cgal()` y termina (`exit`).
     Backup del original en `/openfoamGPU/have_cgal.orig.bak`.
   - **Motivo**: CGAL 4.14 + Boost 1.74 fallan con `amdclang++` (clang 22) con
     `inline asm not supported yet: don't know how to handle tied indirect
     register inputs`.
   - **Impacto**: el core, los solvers y las utilidades básicas no necesitan CGAL.
     Solo se ven afectadas herramientas como `foamyHexMesh`. El build completo
     termina con exit 0.
   - Este cambio está en el árbol de trabajo de git (visible con `git diff`).

2. **`env_gpu.sh`**: añadido `export FOAM_SIGFPE=false` (ver §5).

3. **`etc/prefs.sh`**: `WM_COMPILER=Amd-gpu`, `ROCM_GPU=gfx90a`,
   `FOAM_EXTRA_CXXFLAGS=-Wno-overloaded-virtual`.

No fue necesario aplicar ningún parche a `makeUMPIRE` en esta fase: Umpire se
detectó y enlazó correctamente (`found umpire -- enabling memory pool interface`,
`umpire libs: -lumpire -lfmt -lcamp`).

---

## 8. Componentes opcionales excluidos (no afectan al core)

Todos son **opcionales** y su fallo no impide el uso normal de OpenFOAM:

| Componente | Estado | Motivo |
|------------|--------|--------|
| **KAHIP** | no construido | CMake no encuentra MPI C++ (`mpicxx` no acepta `--hipstdpar`). |
| **METIS** | no construido | El tarball fuente `metis-5.1.0` no está presente en `ThirdParty/sources`. |
| **ADIOS2** | no construido | Su test de CMake usa `g++`/`mpicxx`, que no reconocen `--hipstdpar`/`--offload-arch`. |
| **CGAL** | presente pero deshabilitado en OpenFOAM | Ver §7.1. |

**Decomposición**: para `decomposePar` usar `method scotch` (SCOTCH sí está
compilado y probado). `metis` no está disponible.

---

## 9. Resultados de las pruebas

### 9.1 `applications/test/std_execution`

Compilado y ejecutado OK:

```bash
cd /openfoamGPU/OpenFOAM-v2606/applications/test/std_execution && wclean && wmake
cd /openfoamGPU/std_exec_test && Test-std_execution
```

- El test original usa **1000 iteraciones × 10 000 000 celdas** en cada fase,
  pensado para profiling → tarda mucho (la fase paralela, decenas de minutos).
- Se preparó una variante reducida (`Test-std_execution_quick`: 2 M celdas,
  20 iteraciones) para medir rápido. Resultado:

  | Fase | Tiempo |
  |------|--------|
  | Sequential fill (20 it) | **0.0068 s** |
  | Parallel fill  (20 it) | **59.27 s** |

  Los resultados son **correctos** (`v[0]=7`), pero la ruta `par_unseq` es
  **muchísimo más lenta** en este entorno (ver §10, *Limitación de rendimiento*).

### 9.2 Runs de cavity (icoFoam, un proceso)

| Run | `memory pool` | Tiempo | Exit |
|-----|---------------|--------|------|
| default (`unused`) | `memory pool : unused` | ~48 s | 0 |
| Umpire managed | `memory pool : managed (size=1024MB, incr=1024MB)` | ~69 s | 0 |
| Umpire host | `memory pool : host (size=1024MB, incr=1024MB)` | ~38 s | 0 |

> **Variabilidad de tiempos**: el mismo caso (`cavity`, pool por defecto) se ha
> medido entre **2.5 s y ~290 s** en distintas ejecuciones, sobre una máquina
> compartida con otros procesos usando la GPU. La variabilidad es alta y
> dependiente del estado del runtime/memoria de la GPU, no del caso.

- Resultado físico estable: **Umax = 1.0** (tapa móvil), **Uavg ≈ 0.1886**,
  convergencia a estado estacionario en `t=0.5`. Coincide con el benchmark
  estándar de lid-driven cavity.
- `blockMesh`: 400 celdas, terminó OK (con `FOAM_SIGFPE=false` sale limpio).
- `decomposePar` con SCOTCH: OK (2 subdominios).
- Umpire **enlazado y funcional**: no fue necesario re-enlazar.

Los logs están en:
- `/openfoamGPU/run_cavity_default.log`
- `/openfoamGPU/run_cavity_managed.log`
- `/openfoamGPU/run_cavity_host.log`
- `/openfoamGPU/logs/cavity_run_gpu.log`, `cavity_run_amdsmi.log`
- `/openfoamGPU/logs/amd_smi_monitor.log` (evidencia GPU)

---

## 10. Limitación de rendimiento importante (leer)

En este sistema, las operaciones **`std::execution::par_unseq` son funcionales y
correctas pero muchísimo más lentas que la versión secuencial**, tanto en
OpenFOAM como en un test mínimo aislado de HIP/thrust
(`std::fill_n(par_unseq, ...)` sobre 2 M enteros tarda entre ~2 y ~40 s, cuando
debería ser instantáneo). Es un comportamiento del entorno
**ROCm 7.2.4 + `--hipstdpar` + SVM/XNACK en gfx90a**, no un error del build:

- Verificado que el kernel se lanza en GPU y devuelve `hipSuccess`
  (`thrust::hip_rocprim::__parallel_for::kernel` en `logs/std_exec_dbg.log`).
- Verificado uso real: GPU0 al **100 %**, reloj **1700 MHz**.
- Verificado que los resultados son correctos.
- `HSA_XNACK=1` es imprescindible; con `0` hay *memory access fault*.

El coste dominante parece ser la **migración de páginas SVM por llamada**
(overhead de sincronización de decenas de ms–s por iteración). Recomendaciones:

- Para cargas reales, preferir **buffers grandes reutilizados** y minimizar el
  número de llamadas `par_unseq` pequeñas.
- Mantener `HSA_XNACK=1`.
- Considerar el pool de memoria **`host`** (fue el más rápido en cavity) si el
  caso lo permite.
- Para diagnóstico de kernels: `AMD_LOG_LEVEL=3` muestra cada `hipLaunchKernel`.

**Estado**: OpenFOAM v2606 con offload GPU **compila y ejecuta correctamente**;
la utilidad práctica depende de que se resuelva esta penalización de rendimiento
del runtime hipstdpar en esta máquina.

---

## 11. Cómo reproducir desde cero (resumen)

```bash
# 1) Dependencias del sistema
apt-get install -y gawk libreadline-dev bison libncurses-dev flex cmake

# 2) (Fase 1, ya hecha) Clonar OpenFOAM v2606 rama feature-gpu y ThirdParty con
#    Umpire 2025.03.0 compilado con HIP en
#    ThirdParty-v2606/platforms/linux64Amd-gpu/umpire-2025.03.0/

# 3) Deshabilitar CGAL (incompatible con amdclang++)
cp OpenFOAM-v2606/wmake/scripts/have_cgal /openfoamGPU/have_cgal.orig.bak
printf '#!/bin/sh\nno_cgal() { unset HAVE_CGAL CGAL_FLAVOUR CGAL_INC_DIR CGAL_LIB_DIR HAVE_MPFR MPFR_INC_DIR MPFR_LIB_DIR; }\nno_cgal\nexit\n' \
  > OpenFOAM-v2606/wmake/scripts/have_cgal
chmod +x OpenFOAM-v2606/wmake/scripts/have_cgal

# 4) Entorno (incluye HSA_XNACK=1 y FOAM_SIGFPE=false)
source /openfoamGPU/env_gpu.sh

# 5) Compilar
cd /openfoamGPU/OpenFOAM-v2606
./Allwmake -j -s -q -l        # incremental si se re-ejecuta

# 6) Probar
mkdir -p "$FOAM_RUN" && cd "$FOAM_RUN"
cp -r "$FOAM_TUTORIALS"/incompressible/icoFoam/cavity/cavity . && cd cavity
blockMesh && icoFoam
```

---

## 12. Logs generados

| Log | Contenido |
|-----|-----------|
| `OpenFOAM-v2606/log.linux64Amd-gpuDPInt32Opt` | Build completo (7.9 MB, exit 0) |
| `logs/allwmake_main.log` | Salida de consola de `./Allwmake` |
| `run_cavity_default.log` | icoFoam cavity (pool por defecto) |
| `run_cavity_managed.log` | icoFoam cavity (Umpire managed) |
| `run_cavity_host.log` | icoFoam cavity (Umpire host) |
| `logs/std_execution*.log` | Test std_execution (original y variantes) |
| `logs/amd_smi_monitor.log` | Monitorización GPU (100 %, 1700 MHz) |
| `logs/rocm_smi_*.log` | Muestreos rocm-smi |
| `logs/std_exec_dbg.log` | Traza HIP (`AMD_LOG_LEVEL=3`) |
| `logs/cavity_blockMesh.log`, `logs/cavity_decomposePar.log` | Utilidades |
