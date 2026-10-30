# TRASPASO — Aceleración GPU de OpenFOAM v2606 (`cavity3D`) en AMD MI210

**Para:** quien continúe este trabajo.
**Estado:** cerrado y medido. La GPU **no** acelera esta simulación a la escala
alcanzable en esta máquina; sí se sabe exactamente por qué y cuánto falta para que lo haga.
**Fecha:** octubre 2026.

Este documento es autocontenido: qué se quería, qué se hizo, qué se compiló, cómo se
ejecuta, qué resultados esperar y qué NO repetir. Va de lo operativo a lo técnico. Si
solo tienes 5 minutos, lee §0 y §6.

---

## 0. Resumen en 60 segundos

El objetivo era acelerar `icoFoam` (cavity 3D) con una MI210. Se consiguió **ejecutar
el álgebra en la GPU con precisión FP64 completa y resultados correctos**, y se midió
que aun así no compensa frente a la CPU de esta máquina:

| Números que resumen el trabajo | Valor |
|---|---|
| Mejor resultado absoluto (solve de presión GPU vs **1 núcleo**) | **43,6×** |
| Solve de presión GPU vs **32 rangos** | **1,72×** |
| Paso de tiempo completo vs **32 rangos** (mejor caso, sin mover datos) | **1,20×** |
| Paso completo vs **64 rangos** | **0,93×** (pierde: 0,294 s vs 0,2745 s) |
| Malla a la que haría falta para ganar de verdad (memoria unificada) | 64 M celdas |

**Tres cosas que hay que entender antes de tocar nada:**

1. El paso de `icoFoam` tiene tres bloques: **ensamblado** (31 %), **presión `p`** (39 %) y
   **velocidad `U`** (29 %). Solo `p` está en la GPU. Aunque `p` fuera gratis, el paso solo
   mejoraría 1,65× (ley de Amdahl).
2. La GPU es **5× más rápida por iteración** que 32 núcleos, pero necesita **2,9× más
   iteraciones** (39 vs 13,45) porque su multigrid es más simple que el GAMG de OpenFOAM.
   El producto da el 1,72×.
3. El límite es el **ancho de banda y el movimiento de datos**, no el FP64. Pasando los
   coeficientes a FP32 solo se gana 1,37× (y se rompe la tolerancia). Medido, no supuesto.

**Qué NO repetir** (todo medido, ver §8): la ruta oficial `hipstdpar` de la rama GPU
(451 s/paso), `GAMG+twoStageGaussSeidel` en GPU (no termina 1 paso), y la GPU como
precondicionador MPI (10–77× más lento).

---

## 1. La máquina

| Componente | Detalle |
|---|---|
| CPU | 2× AMD EPYC 7763 = **128 núcleos**, 512 MiB de L3 en total, 247 GB de RAM |
| GPU | 2× AMD Instinct **MI210** (gfx90a), 64 GB de VRAM cada una |
| **GPU usable** | **solo la 0**. La 1 es de otro inquilino: no tocarla |
| Enlace GPU | **PCIe 4.0 ×16** (16 GT/s ×16), **28 GB/s** por sentido medido, 11,7 µs de latencia |
| ROCm | **7.2.4** (`/opt/rocm`, `hipcc` de ROCm) |
| OpenFOAM | **v2606** |
| Build CPU | `/openfoamGPU/cpu/OpenFOAM-v2606` (`linux64GccDPInt32Opt`), del tarball oficial |
| Build GPU | `/openfoamGPU/OpenFOAM-v2606` (rama `feature-gpu` de openfoam-ecse, commit `559823f7`, `linux64Amd-gpuDPInt32Opt`) |
| ThirdParty | `/openfoamGPU/ThirdParty-v2606` (Umpire, Boost, FFTW, CGAL deshabilitado) |

Comprobar la topología y que la GPU 0 está libre:

```bash
rocm-smi --showtopo | head -20      # GPU0 y GPU1, unidas por PCIe
rocm-smi --showuse                  # % de uso (poco fiable, ver §7)
amd-smi monitor -u -m -w 1          # más fiable: GFX% y reloj
```

---

## 2. El problema, en corto

`icoFoam` resuelve flujo incompresible con el algoritmo PISO. Es un cubo con tapa
deslizante (cavity). Cada **paso de tiempo** (0,005 s de tiempo físico) hace:

1. **Ensamblado** de las ecuaciones (`fvMatrix`) y actualización de flujos.
2. **`U`**: velocidad, 3 componentes; `smoothSolver` (Gauss-Seidel) hasta 1e-6.
3. **`p`**: presión, **dos** resoluciones encadenadas — `p` (relTol 0,01) y `pFinal`
   (residual ≤ 1e-6) — normalmente con **GAMG** (multigrid algebraico).

Por eso la unidad de medida de todo este trabajo es **segundos por paso de tiempo**, no
"iteraciones" ni "MFLOPs": las iteraciones de dos solvers distintos no son comparables.

Mallas usadas: cubos de 16³ a 256³ celdas (4 mil a 16,8 M). La de trabajo es **128³ =
2.097.152 celdas**, con una matriz de presión de 14.581.760 no ceros.

---

## 3. Mapa del repositorio

```
/openfoamGPU/
├── README.md                     ← empieza aquí (índice)
├── TRASPASO.md                   ← este documento
├── INFORME_GPU_V2606.md          ← informe técnico completo (la referencia)
├── COMPARACION_PRESION_GPU_CPU.md← la comparación CPU vs GPU definitiva
├── KRYLOV_RESIDENTE_GPU.md       ← el solver que funciona (ruta C)
├── RESPUESTA_COMENTARIO_FORO.md  ← contraste con un hilo del foro (L40S) + FP32 + PCIe
├── OPCION_B_MPI_GPU.md           ← el intento MPI+GPU (fallido, documentado)
├── POC_ROCALUTION_V2606.md       ← PoC del plugin rocALUTION
├── ANALISIS_ROCALUTION.md        ← análisis del AMG de rocALUTION
├── README_BUILD.md               ← build de la rama GPU paso a paso
│
├── env_gpu.sh                    ← entorno de la rama GPU
├── cpu/OpenFOAM-v2606/           ← build oficial CPU (la que se usa para medir)
├── OpenFOAM-v2606/               ← build rama GPU (hipstdpar)
│
├── exp_scaling_cpu/              ← escalera CPU 16³..128³, 1..64 núcleos
│   ├── results_cpu_gamg.csv      ← TABLA BASE de la CPU
│   ├── split_fase_128cube.md     ← primer reparto de fases (64 rangos)
│   ├── gen_case.py               ← genera un caso cavity NX³ limpio
│   ├── gen_decomp.py             ← escribe decomposeParDict válido para NX/NP
│   └── env_cpu.sh, run_cpu*.sh, build_cpu.sh
│
├── exp_scaling_gpu_gamg/         ← primer intento: rama GPU (hipstdpar). Histórico
├── exp_scaling_gpu/              ← aún más antiguo. Histórico
├── std_exec_test/, std_exec_quick/ ← pruebas del runtime stdpar. Histórico
│
├── rocALUTION-solver/            ← PLUGIN que hace funcionar la GPU
│   ├── rocALUTIONLduSolver.{C,H} ← solver (y el volcado rocDump → roc_matrix_p.bin)
│   ├── rocALUTIONLduPreconditioner.{C,H} ← versión "preconditioner" (opción B)
│   ├── results_roc_poc.csv       ← resultados por tamaño (32³/64³/128³)
│   ├── results_mpi_precond.csv   ← resultados de la opción B
│   └── case_*_roc/               ← casos listos con `solver rocALUTION;`
│
├── gpu_krylov/                   ← BANCOS DE MEDIDA en HIP (la ruta que funciona)
│   ├── bench_mg.cpp              ← PCG + multigrid geométrico  ← EL BUENO
│   ├── bench_pcg.cpp             ← PCG + Jacobi / rocsparse
│   ├── bench_ilu.cpp             ← PCG + ILU(0) multicolor
│   ├── bench_pcie.cpp            ← ancho de banda host↔GPU
│   ├── data/roc_matrix_p.bin     ← matriz real de p a 128³ (191 MB, NO versionada:
│   │                               se regenera con repro/dump128.sh, ~1 min)
│   └── results_krylov_residente.csv ← todos los resultados de esta ruta
│
└── repro/                        ← los experimentos, reproducibles (ver repro/README.md)
```

**Si solo vas a mirar tres archivos:** `exp_scaling_cpu/results_cpu_gamg.csv` (CPU),
`gpu_krylov/results_krylov_residente.csv` (GPU) y `COMPARACION_PRESION_GPU_CPU.md` (la
comparación y el método).

**Al clonar el repositorio** (6,6 MB, 547 ficheros) **no vienen**: los builds
(`OpenFOAM-v2606/`, `cpu/`, `ThirdParty-v2606/`, `dist/`), las mallas
(`constant/polyMesh/`), los campos (`0/`, `0.1/`, `processor*/`) ni
`gpu_krylov/data/roc_matrix_p.bin`. Se regeneran con §4 y §5; cada exclusión está
comentada en `.gitignore`.

---

## 4. Compilar

### 4.1 Build CPU oficial (la que se usa para medir)

```bash
cd /openfoamGPU/exp_scaling_cpu
./build_cpu.sh                 # extrae dist/OpenFOAM-v2606.tgz + Allwmake -j
# o, si ya está compilada:
source /openfoamGPU/exp_scaling_cpu/env_cpu.sh
which icoFoam                  # /openfoamGPU/cpu/OpenFOAM-v2606/platforms/linux64GccDPInt32Opt/bin/icoFoam
```

Notas:
- `env_cpu.sh` desactiva `nounset` (el `bashrc` de OpenFOAM usa variables sin definir) y
  **limpia** el entorno GPU (`HSA_XNACK`, `ROCR_VISIBLE_DEVICES`…).
- Si `icoFoam` ya existe, no hace falta recompilar. Tarda ~40 min en frío.

### 4.2 Build de la rama GPU (`feature-gpu`)

```bash
source /openfoamGPU/env_gpu.sh
echo $WM_OPTIONS               # linux64Amd-gpuDPInt32Opt
which amdclang++               # /opt/rocm/bin/amdclang++
cd /openfoamGPU/OpenFOAM-v2606
./Allwmake -j -s -q -l         # ~1 h en frío
```

Requisitos que ya están aplicados en esta copia (detalle en `README_BUILD.md` §7):

- **CGAL deshabilitado** en `wmake/scripts/have_cgal` (si no, el build falla). El original
  está en `/openfoamGPU/have_cgal.orig.bak`.
- `etc/prefs.sh` con `FOAM_EXTRA_CXXFLAGS=-Wno-overloaded-virtual`.
- Reglas de compilador `wmake/rules/General/Amd-gpu/c++` y `wmake/rules/linux64Amd-gpu/`
  (usan `--offload-arch=$ROCM_GPU` con `ROCM_GPU=gfx90a`).

**Aviso:** este build es el que hicimos funcionar, pero su offload (`--hipstdpar`) **no
acelera**: es el que da 451 s/paso (§8). Se conserva porque documenta la ruta oficial y
porque el plugin se apoya en el mismo árbol. Para medir, usa el build CPU + plugin.

Variables obligatorias al ejecutarlo:

| Variable | Valor | Por qué |
|---|---|---|
| `HSA_XNACK=1` | sí | sin esto, el SVM de `--hipstdpar` da *memory access fault* y coredump |
| `FOAM_SIGFPE=false` | sí | el runtime ROCm lanza un SIGFPE espurio al salir → exit 136 |
| `ROCR_VISIBLE_DEVICES=0` y `HIP_VISIBLE_DEVICES=0` | siempre | la GPU 1 es de otro inquilino |
| `FOAM_MEMORY_POOL` | vacío | `managed`/`host` solo para pruebas; el más rápido fue `host` |

### 4.3 Plugin rocALUTION (el que hace que la GPU funcione)

Está compilado **contra la build CPU** (no contra la rama GPU). Para recompilar:

```bash
set +u
source /openfoamGPU/exp_scaling_cpu/env_cpu.sh
set -u
cd /openfoamGPU/rocALUTION-solver
wmake libso
ls -l $FOAM_USER_LIBBIN/librocALUTIONLduSolver.so
# -> /root/OpenFOAM/root-v2606/platforms/linux64GccDPInt32Opt/lib/librocALUTIONLduSolver.so
```

Se carga desde el caso, en `system/controlDict`:

```
libs            ("librocALUTIONLduSolver.so");
```

y se usa en `system/fvSolution`:

```
p
{
    solver              rocALUTION;
    rocPreconditioner   AMG;        // o Jacobi
    rocDevice           0;
    rocDump             false;      // true = escribe roc_matrix_p.bin (una vez)
    tolerance           1e-06;
    relTol              0.01;
    maxIter             100;
}
pFinal { $p; relTol 0; }
```

**Ojo:** el `.so` vive en `$FOAM_USER_LIBBIN`, **fuera** de `/openfoamGPU`. Si alguien
borra `/root/OpenFOAM`, hay que recompilarlo.

### 4.4 Bancos de medida (HIP puro, sin OpenFOAM)

Compilan en segundos y son la vía para tocar el solver:

```bash
cd /openfoamGPU/gpu_krylov
INC="-I/opt/rocm/include -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib"
hipcc -O3 -std=c++17 bench_mg.cpp   -o bench_mg   $INC -lhipblas -lrocsparse
hipcc -O3 -std=c++17 bench_pcg.cpp  -o bench_pcg  $INC -lhipblas -lrocsparse
hipcc -O3 -std=c++17 bench_ilu.cpp  -o bench_ilu  $INC -lhipblas -lrocsparse
hipcc -O3 -std=c++17 bench_mg.cpp -DUSE_FLOAT_COEFFS -o bench_mg_f32 $INC -lhipblas -lrocsparse
hipcc -O3 -std=c++17 bench_pcie.cpp -o bench_pcie $INC -lhipblas
```

### 4.5 Verificar que todo está bien

```bash
source /openfoamGPU/exp_scaling_cpu/env_cpu.sh
icoFoam -help > /dev/null && echo "CPU OK"
ls $FOAM_USER_LIBBIN/librocALUTIONLduSolver.so && echo "plugin OK"
cd /openfoamGPU/gpu_krylov
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
./bench_pcie 16777216 | head -8     # debería dar ~28 GB/s por sentido
```

---

## 5. Ejecutar

Todas las recetas son copiables. `$` = terminal.

### 5.1 Generar un caso limpio

```bash
python3 /openfoamGPU/exp_scaling_gpu_gamg/gen_case.py 128 20 /tmp/micase
cd /tmp/micase && blockMesh
```

- `NX STEPS OUTDIR`: malla NX³, `endTime = STEPS × 0.005`.
- El caso ya trae `GAMG + twoStageGaussSeidel` para `p` (config recomendada).
- Las mallas grandes tardan: 128³ ~10 s, 256³ ~1 min.

> **Importante:** los diccionarios escritos a mano necesitan la cabecera `FoamFile`
> (`version/format/class/object`). Sin ella OpenFOAM no arranca.

### 5.2 CPU: escalera de núcleos

```bash
cd /openfoamGPU/exp_scaling_cpu
./run_cpu_ladder.sh 128 "1 2 4 8 16 32 64" 20
cat results_cpu_gamg.csv
```

Usa 1 hilo por rango (`OMP_NUM_THREADS=1`) y `mpirun --allow-run-as-root`. Tarda ~6 min.

### 5.3 CPU: reparto de fases (el método que da los números finos)

Tres corridas idénticas cambiando **solo** el solver de una ecuación; al restar se aísla
cada fase:

```bash
cd /openfoamGPU/repro/diff_cpu
./run_diff.sh 32 1          # ~3 min
python3 split.py            # tabla de fases + techos de Amdahl
```

| Variante | `p`/`pFinal` | `U` |
|---|---|---|
| `fv_gamg` | GAMG | smoothSolver |
| `fv_p1` | 1 iteración | smoothSolver |
| `fv_both1` | 1 iteración | 1 iteración |

`gamg − p1` = bloque `p`; `p1 − both1` = bloque `U`; `both1` = ensamblado + PISO + I/O.

### 5.4 GPU: el plugin rocALUTION (paso completo en GPU)

```bash
cd /openfoamGPU/rocALUTION-solver
source /openfoamGPU/exp_scaling_cpu/env_cpu.sh
./run_case.sh case_128cube_roc_full run128     # escribe log.run128, monitor_run128.csv, wall.run128
grep -E "rocALUTION build|Solving for" case_128cube_roc_full/log.run128 | head
```

Esperado a 128³: `rocALUTION build: ... time≈400 ms` (por campo y paso) y **~4,5 s/paso**.
Es la única vía que ejecuta `p` **y** `U` en GPU con resultados correctos (ΔU ≈ 5e-6 frente
a CPU), pero es **13,6× más lenta** que 32 rangos.

### 5.5 GPU: el solver residente (la vía buena) — `bench_mg`

```bash
# 1) matriz de p del caso 128³
cd /openfoamGPU/repro && ./dump128.sh /tmp/dump128      # ~1 min -> roc_matrix_p.bin

# 2) PCG + multigrid, cadena de tolerancias como OpenFOAM
cd /tmp/dump128
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
CF=2 SMOOTH=gs NU=2 WARM=1.3e-3,1e-6 \
  /openfoamGPU/gpu_krylov/bench_mg roc_matrix_p.bin 0.13 500 128
```

Esperado (3 repeticiones, ±2 ms):

```
PCG(MG) tol=1e-01 : iters=7   total=13.1 ms   res/||b||=1.283e-01
PCG(MG) WARM tol=1e-03 : iters=12 total=24.0 ms ...
PCG(MG) WARM tol=1e-06 : iters=27 total=56.4 ms res/||b||=4.392e-07
```

`p` (24,0 ms) + `pFinal` (56,4 ms) = **80,5 ms** = el bloque de presión que se compara con
los 138,5 ms de CPU a 32 rangos.

Parámetros de `bench_mg`:

| Qué | Cómo | Valores |
|---|---|---|
| fichero, tol, maxIt, NX | argumentos | `roc_matrix_p.bin 0.13 500 128` |
| factor de agregación | `CF` | 2 (óptimo; 4 probado, peor) |
| barridos por nivel | `NU` | 2 |
| suavizador | `SMOOTH` | `gs` (Gauss-Seidel) o vacío (Jacobi) |
| cadena en caliente | `WARM` | `1.3e-3,1e-6` |
| solo el V-cycle | `MGONLY=1` | — |
| verificar contra CSR | `CHECK=1` | — |
| volcar la solución | `XDUMP=x.bin` | para `repro/check_residual.py` |

### 5.6 GPU: PCIe

```bash
cd /openfoamGPU/gpu_krylov
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
./bench_pcie 16777216     # un vector de p (16,8 MB)
./bench_pcie 175000000    # la matriz completa
```

### 5.7 Malla 256³

```bash
cd /openfoamGPU/repro/scale_256
./run_256.sh 256 32 20     # ~12 min, 7 GB de disco, 40 GB de RAM
```

---

## 6. Qué esperar (resultados de referencia)

### 6.1 Paso de tiempo completo, CPU (mediana de 20 pasos)

| Malla | 1 núcleo | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 64³ | 0,570 s | 0,085 | 0,055 | 0,040 | 0,040 |
| 128³ | 8,750 s | 0,960 | 0,540 | **0,330** | **0,260** |
| 256³ | — | — | — | **6,671 s** | — |

A 32³ la resolución temporal es de 0,01 s y no sirve para comparar: usa 64³ o más.

Estos valores son de la escalera (`results_cpu_gamg.csv`, 20 pasos, mediana). El método
diferencial de §5.3 da 0,352 s a 32 rangos y 0,2745 s a 64: la diferencia (~5 %) es de
método y de carga de la máquina, no del solver. Para comparar con la GPU se usa el
diferencial, que es el que aísla las fases.

### 6.2 Reparto de fases a 128³ (método de §5.3)

| Fase | 1 rango | 32 rangos | 64 rangos |
|---|---|---|---|
| ensamblado + PISO + I/O | 4,765 s (45 %) | 0,1105 s (31 %) | 0,0985 s |
| bloque `p` | 3,513 s (33 %) | **0,1385 s (39 %)** | 0,106 s |
| bloque `U` | 2,398 s (22 %) | 0,1030 s (29 %) | 0,070 s |
| **paso** | **10,675 s** | **0,352 s** | **0,2745 s** |

### 6.3 GPU

| Medida | Valor esperado |
|---|---|
| Bloque `p` residente (128³, FP64) | **80,5 ms** (39 iteraciones, 2,06 ms/iter) |
| Iteración GPU vs 32 rangos | 2,06 vs 10,3 ms → **5×** |
| Iteraciones GPU vs GAMG | 39 vs 13,45 → **2,9× más** |
| Bloque `p` con coeficientes FP32 | 58,8 ms (1,37×) pero residual real 9,4e-6 ❌ |
| Plugin rocALUTION, paso completo | 32³ 0,090 s · 64³ 0,470 s · 128³ **4,500 s** |
| Escalado del kernel 128³→256³ | 6,4–8× para 8× celdas |

Variación aceptable entre repeticiones: **±3 %** en CPU, **±2 ms** en GPU. Si te sale
mucho más, comprueba que no haya otra cosa corriendo (CPU y GPU se estorban).

---

## 7. Trampas conocidas (todas costaron tiempo)

1. **GPU 1 no es nuestra.** Exporta siempre `ROCR_VISIBLE_DEVICES=0` y
   `HIP_VISIBLE_DEVICES=0`. Sin esto, rocALUTION puede elegir la 1.
2. **`set +u` antes de sourcear cualquier `bashrc` de OpenFOAM** (`env_cpu.sh`/`env_gpu.sh`
   ya lo hacen). Con `set -u`, el source aborta a mitad.
3. **`FOAM_SIGFPE=false` siempre.** Si no, los procesos terminan con exit 136 y core dump
   *después* de acabar bien.
4. **`HSA_XNACK=1` solo para la rama GPU.** Con la CPU o el plugin no hace falta; con
   `--hipstdpar` es obligatorio.
5. **`ExecutionTime` de OpenFOAM miente** en GPU y en serie (mide el hilo 0). Usa
   *wall clock* (`date +%s.%N`) o `ClockTime`.
6. **No borres `processor*/constant/`.** `decomposePar` mete la malla ahí; si la limpias,
   las corridas en paralelo fallan. Borra solo los directorios de tiempo.
7. **Cabecera `FoamFile` obligatoria** en cualquier diccionario que escribas a mano.
8. **`mpirun --allow-run-as-root --oversubscribe`** (somos root en el contenedor).
9. **No hay `/usr/bin/time` ni `column`.** Usa `date +%s.%N` y `awk`.
10. **CPU y GPU se estorban.** No midas una mientras corre la otra. `OMP_NUM_THREADS=1`
    por rango en la CPU.
11. **El plugin va con la build CPU**, no con la rama GPU. Mezclarlos no está probado.
12. **El volcado de la matriz a 256³ en serie no terminó en 30 min.** No lo intentes: para
    escalar usa `repro/gen_poisson.py` (Laplaciano 7-puntos sintético del mismo tamaño).
13. **`rocDump` solo vuelca `p`**, una vez por ejecución, al directorio del caso, como
    `roc_matrix_p.bin` (la matriz va negada: diagonal positiva).
14. **La matriz de `p` es singular** (el valor de presión es libre). No te asustes si el
    campo tiene una constante grande: compara quitando la media.
15. **`rocm-smi --showuse` devuelve 0 aunque haya carga.** Usa `amd-smi monitor` o el reloj
    (800 MHz idle → 1700 MHz activo).
16. **Los resultados no son bit-reproducibles** entre GPU y CPU (orden de reducciones).
    Compáralos con tolerancia (ΔU ~1e-5).
17. **No pongas `FOAM_MEMORY_POOL=managed` esperando magia.** En la MI210 la memoria gestionada
    rinde a velocidad de VRAM solo si cabe y se precarga a mano; si no cabe, cae a 1,3–6,6 GB/s
    (y con `HSA_XNACK=1` se cuelga). Medido en `MEMORIA_UNIFICADA_MI210.md`.

---

## 8. Qué se intentó y qué falló (histórico)

| Intento | Qué era | Resultado medido |
|---|---|---|
| `--hipstdpar` (rama `feature-gpu`) | offload automático del bucle PISO | **451 s/paso a 32³** (la CPU hace 0,05 s) |
| `GAMG + twoStageGaussSeidel` en esa rama | la config recomendada por la doc | **no completa 1 paso en 30 min** |
| Plugin rocALUTION, solver completo | `p` y `U` en GPU | funciona; 4,5 s/paso a 128³ (**1,9× vs 1 núcleo, 13,6× más lento que 32 rangos**) |
| Plugin como **preconditioner** en MPI (opción B) | `p` en CPU con AMG en GPU por rango | **10× más lento** con 8 rangos, **77×** con 16; iteraciones 11 → 149 |
| PCG + Jacobi residente | todo en VRAM, precondicionador barato | 1029 iteraciones, 579 ms |
| PCG + ILU(0) multicolor | idem con mejor precondicionador | 524 iteraciones, 815 ms |
| **PCG + MG 2×2×2 (Gauss-Seidel)** | multigrid geométrico | **39 iteraciones, 80,5 ms ← el mejor** |
| Barrido CF/NU (1,2,4 / 1,2) | afinar el multigrid | CF=2, NU=2, GS es el óptimo de los probados |
| Coeficientes en FP32 | probar si el FP64 era el límite | 58,8 ms (1,37×) pero residual 9,4e-6 ❌ |

Conclusión de la tabla: **el problema no es "portar" sino que a 2 M celdas el bloque `p` es
solo el 39 % del paso y la GPU solo gana 1,7× en él.** El registro completo está en
`INFORME_GPU_V2606.md`.

---

## 9. Pendientes, por orden de valor

1. **Medir `U` en GPU** (lo único sin dato). `U` es no simétrica, necesita **BiCGStab +
   ILU0**. Para volcar su matriz hay que tocar 3 líneas en
   `rocALUTION-solver/rocALUTIONLduSolver.C` (~línea 335): cambiar
   `fieldName_ == "p"` por `fieldName_ == "Ux"` y el nombre del fichero. `bench_ilu.cpp` ya
   tiene la factorización; falta el solver. **Sin esta medida, cualquier plan sobre `U` es
   una apuesta.** Estimación: 1–2 h.
2. **Mejorar el multigrid** para cerrar la brecha de iteraciones (39 → ~20): W-cycle,
   agregación suavizada, interpolación de orden mayor. Con 20 iteraciones el bloque `p`
   bajaría a ~45 ms y el paso a ~1,35×.
3. **Ensamblado en el dispositivo**: la única vía para pasar de 2×. Es escribir un
   `icoFoam` residente en HIP (2.000–4.000 líneas, 4–6 meses). El techo es 3,19× (si `p` y
   `U` fueran gratis) y el realista ~2,3×.
4. **Memoria unificada (MI300A) + malla de 64 M celdas**: el único régimen donde se publica
   ×2. Aquí no cabe (166 GB de matriz vs 64 GB) y la MI210 es PCIe. La memoria unificada de
   la MI210 **sí funciona pero no ayuda**: medido en `MEMORIA_UNIFICADA_MI210.md` (815 GB/s
   si cabe y se precarga; 1,3–6,6 GB/s si no cabe).

---

## 10. Índice de documentos

| Documento | Para qué |
|---|---|
| `INFORME_GPU_V2606.md` | Informe completo, 14 secciones, con todo el histórico |
| `COMPARACION_PRESION_GPU_CPU.md` | La comparación CPU vs GPU y el método diferencial |
| `KRYLOV_RESIDENTE_GPU.md` | Diseño y medidas del solver residente |
| `RESPUESTA_COMENTARIO_FORO.md` | FP32, PCIe y comparación con L40S/A100/H100/MI300A |
| `MEMORIA_UNIFICADA_MI210.md` | "¿y no puedes usar memoria unificada?": medido, con los 3 regímenes y las cuentas del caso que no cabe |
| `OPCION_B_MPI_GPU.md` | El intento MPI + GPU (por qué falló) |
| `POC_ROCALUTION_V2606.md` | Cómo se hizo el plugin |
| `ANALISIS_ROCALUTION.md` | Análisis del AMG de rocALUTION |
| `README_BUILD.md` | Build de la rama GPU paso a paso |
| `repro/README.md` | Los scripts reproducibles |

---

## 11. Glosario

| Término | Qué es |
|---|---|
| `icoFoam` | solver de flujo incompresible, laminar, transitorio (PISO) |
| **paso de tiempo** | una iteración externa de `icoFoam` (aquí 0,005 s físicos) |
| `p`, `U` | presión y velocidad (3 componentes) |
| **ensamblado** | construir la matriz y el término fuente de una ecuación |
| **GAMG** | multigrid algebraico de OpenFOAM (precondicionador potente pero caro) |
| **multigrid** | resolver en mallas cada vez más gruesas; pocas iteraciones |
| **precondicionador** | aproximación barata de la inversa que acelera el solver |
| **SpMV** | producto matriz×vector (la operación que domina: es memoria) |
| **CSR** | formato de matriz dispersa (fila, columnas, valores) |
| **FP64 / FP32** | coma flotante de 64 / 32 bits (`double` / `float`) |
| **residual** | cuánto le falta a la solución para cumplir la ecuación (`||b−Ax||`) |
| `relTol` | cuánto debe bajar el residual respecto al inicial |
| **PCIe** | el bus por el que se mueven los datos CPU↔GPU (28 GB/s aquí) |
| **VRAM** | memoria de la GPU (64 GB) |
| **ley de Amdahl** | límite de aceleración al optimizar solo una parte del trabajo |
| **MI300A** | APU de AMD con CPU y GPU compartiendo memoria (sin PCIe) |

---

*Si algo de este documento no cuadra con la realidad, la realidad manda: vuelve a medir
con `repro/` y actualiza este archivo. Los números de §6 son medianas de 2026; no son
sagrados, pero están todos medidos y son reproducibles.*
