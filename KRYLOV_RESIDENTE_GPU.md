# Ruta C — Krylov residente en GPU con preconditionador barato (medido)

**Fecha:** octubre 2026
**GPU:** AMD Instinct MI210 (gfx90a, 64 GB HBM2e, ~1,6 TB/s), ROCm 7.2.4, **solo GPU 0**
**Malla:** `cavity3D` 128³ = 2.097.152 celdas, 14.581.760 nnz (7 puntos), matriz de `p` real
**Objetivo:** responder si la GPU puede ser algo más que "unos pocos núcleos" cuando el álgebra
lineal vive **entera** en el dispositivo.

---

## 1. Por qué esta ruta

Las tres vías anteriores fallaron por motivos distintos y ya medidos:

| Vía | Resultado | Causa raíz |
|---|---|---|
| Port completo `hipstdpar` (§7–§8 del informe) | 451 s/paso a 32³ | runtime/memoria del offload |
| Solver GPU externo en serie (§12) | 4,50 s/paso a 128³ | 70 % ensamblado + 27 % CSR en CPU |
| MPI + AMG como preconditionador (§13) | 9,7–42 s/paso | un *round trip* host↔GPU por iteración + setup AMG |

En §13 el coste por iteración era de **23 ms** para bloques locales de 262 k celdas: eso no es
la GPU, es la orquestación (subir `r`, aplicar, bajar `w`, sincronizar, 8–16 procesos sobre un
mismo dispositivo). La pregunta correcta es: **si el bucle de Krylov no sale nunca del dispositivo
y el preconditioner no paga setup de AMG, ¿cuánto puede realmente esta GPU?**

Para aislarla se exportó la matriz real de `p` desde el plugin (`rocDump`, un solo fichero binario
de 191 MB con `rowPtr`/`colIdx`/`values`/`rhs`) y se midió con tres programas HIP independientes,
sin OpenFOAM ni MPI de por medio. Todo el código está en `/openfoamGPU/gpu_krylov/`.

---

## 2. Qué se implementó

| Fichero | Contenido |
|---|---|
| `bench_pcg.cpp` | SpMV (rocSPARSE) + PCG con **Jacobi** (setup nulo) |
| `bench_ilu.cpp` | coloración del grafo + permutación + **ILU(0)** multicolor (2 colores) |
| `bench_mg.cpp` | **multigrid por agregación** 2×2×2, jerarquía fija, Galerkin exacto |
| `results_krylov_residente.csv` | tabla de resultados |

Detalles que importan:

- **SpMV** con `rocsparse_dcsrmv`: 0,219 ms ⇒ **951 GB/s efectivos** (60 % del pico teórico).
  El mismo SpMV escrito a mano da 230 GB/s: la diferencia es todo.
- **Todo residente.** La matriz, el RHS y los vectores viven en VRAM entre iteraciones. El único
  tráfico host↔GPU por iteración son 3 escalares (`p·Ap`, `r·r`, `r·z`) ≈ 30 µs.
- **ILU(0)**: coloración voraz → 2 colores (el grafo 7-puntos es bipartito) → permutación por color
  → factorización por colores (una *launch* por color). El *apply* son dos barridos triangulares.
- **Multigrid**: los agregados son cubos 2×2×2 del índice de celda (`i + 128j + 16384k`), así que
  el operador grueso sigue siendo de 7 puntos y se calcula con **sumas exactas de Galerkin**
  (`A_c = PᵀAP`, verificado a 1,7e-16). Sin coarsening RS/PMIS y sin SpGEMM: el "setup" es una
  pasada O(nnz) por nivel (**1,0 ms** en total, todos los niveles).
  Suavizado: 2 barridos de Gauss-Seidel por color antes y 2 después, en **orden de colores
  invertido** en el post-suavizado (si no, el ciclo V no es simétrico y CG no converge).
  Nivel más grueso: 2³ = 8 celdas, resuelto con 200 barridos de Jacobi dentro de un solo bloque.

---

## 3. Resultados (128³, 2,1 M celdas, GPU 0)

Reducción del residual de 2 órdenes de magnitud (≈100×, equivalente al `relTol 0.01` de OpenFOAM
partiendo de `p` anterior; ver §5). Tiempos de pared, 3 repeticiones, dispersión <2 %.

| Preconditioner | Iteraciones | Tiempo total | ms/iteración | Setup |
|---|---|---|---|---|
| Jacobi (sin setup) | 298 | 168,0 ms | 0,56 | 0 |
| ILU(0) 2 colores | 152 | 261,1 ms | 1,72 | 176 ms + 4,6 ms factorización |
| **Multigrid 2×2×2** | **14** | **28,3 ms** | 2,02 | **1,0 ms** (todos los niveles) |

El operador que usa el multigrid es exactamente la matriz de OpenFOAM, no una aproximación: se
extraen los 7 coeficientes del CSR (`diag`, caras +x/+y/+z) y se compara `A·x` con el producto
CSR completo ⇒ **error relativo 1,25e-16** (y `diag = Σ|offdiag|`, Laplaciano de Neumann puro).

Convergencia completa (misma tabla, tolerancias más duras):

| tol | Jacobi | ILU(0) | Multigrid |
|---|---|---|---|
| 1e-2 | 298 it / 168 ms | 152 it / 261 ms | **14 it / 28 ms** |
| 1e-4 | 410 it / 230 ms | 209 it / 345 ms | **32 it / 67 ms** |
| 1e-6 | 1029 it / 579 ms | 524 it / 815 ms | **44 it / 94 ms** |
| 1e-8 | 1143 it / 644 ms | 582 it / 908 ms | **52 it / 111 ms** |

Lectura:

1. **El suelo por iteración es 0,56 ms** (Jacobi), y de eso 0,22 ms es SpMV; el resto son 3 puntos
   con sincronización. Es memoria + latencia, no cómputo: 14,6 M nnz en 0,22 ms son 29 GFLOP/s,
   pero 951 GB/s.
2. **Jacobi no basta**: 298 iteraciones. ILU(0) en 2 colores reduce a la mitad, pero su *apply*
   cuesta 1,0 ms (la permutación rompe la localidad y cada barrido es un *gather* disperso), así
   que el total empeora. Con 2 colores ILU(0) degenera en un LU por bloques con Schur diagonal.
3. **El multigrid cambia el orden de magnitud**: 14 iteraciones. Su ciclo V cuesta 1,74 ms porque
   son 5 SpMV equivalentes en el nivel fino (2+2 suavizados + residual); los niveles gruesos, en
   cambio, son gratis.
4. **Memoria**: la jerarquía completa ocupa ~130 MB a 128³ (4 arrays por nivel, 8/7 de factor),
   frente a **6,7 GB** de las 4 jerarquías RS-AMG de rocALUTION (§12). La diferencia es el tipo de
   AMG, no la malla.

---

## 4. Comparación con CPU (bloque de presión completo, misma malla)

Antes de esta tabla hubo un error de comparación, ya corregido: los 28,3 ms de arriba son **una**
resolución fría hasta `relTol 1e-2`. OpenFOAM hace **dos** por paso (`p` y `pFinal`), y la segunda
arranca en caliente pero baja a 1e-6. La comparación correcta es 28,6 ms (`p`) + 56,4 ms (`pFinal`)
= **80,5 ms** para el bloque completo. Detalle y método en **`COMPARACION_PRESION_GPU_CPU.md`**.

| Plataforma (bloque `p` = `p`+`pFinal`, 128³) | Tiempo | Iteraciones | ms/iter | vs GPU |
|---|---|---|---|---|
| CPU 1 núcleo (`GAMG`) | 3.513 ms (medido) | 16,5 | 213 | **43,6× más lenta** |
| CPU 32 rangos (`GAMG`) | 138,5 ms (medido) | 13,45 | 10,3 | **1,72× más lenta** |
| CPU 64 rangos (`GAMG`) | 106 ms (medido) | — | — | 1,32× más lenta |
| **GPU 0, Krylov residente + MG** | **80,5 ms** | 39 | 2,06 | — |

Lo importante de la tabla son las dos columnas centrales: **por iteración la GPU es 5× más rápida,
pero necesita 2,9× más iteraciones**. El multigrid de agregación 2×2×2 converge peor que GAMG. El
producto de ambos efectos es el 1,72×, y por eso **cuantos más rangos, menos sirve la GPU**: a 64
rangos la ventaja cae a 1,32× y a 128 sería como mucho un empate. En este kernel una MI210 equivale
a **~44 núcleos EPYC 7763** (43,6× frente a uno; ~55 si se compara contra 32 rangos, porque la CPU
no escala de forma lineal).

La afirmación "la GPU equivale a dos núcleos" sigue siendo falsa (era un artefacto de la
orquestación, §13), pero la equivalencia real es "unas decenas", no "cientos".

---

## 5. Qué significa a nivel de paso (y por qué no se traduce en 1,72×)

El paso completo a 128³ con 32 ranks cuesta 0,352 s (medido hoy por diferencia de tres corridas):
ensamblado 0,110 s + `p` 0,1385 s + `U` 0,103 s. Sustituyendo:

| Escenario | Paso | Ganancia |
|---|---|---|
| Bloque `p` en GPU (80,5 ms), resto igual | 0,110 + 0,0805 + 0,103 = **0,294 s** | **1,20×** |
| `p` y `U` en GPU (U ≈ 0,04 s, no medido en GPU) | **≈0,23 s** | **≈1,5×** |
| Solve infinitamente rápido (suelo de Amdahl) | 0,110 s | **3,19×** |

Ese 1,20× **exige** además que la integración no pague nada: reunir la matriz en un contexto cuesta
≥0,3 s (§13) y reconstruir el CSR 0,40–0,45 s, cualquiera de los dos solo ya borra la ganancia.
Por eso la ruta B medida daba 10–77× *más lenta* aunque el solve en sí fuera rápido.

Regla que se desprende de las tres campañas: **el ensamblado es el 31–45 % del paso y está en CPU**;
mientras no se descargue, ninguna mejora del solve puede pasar de ~3,2×, y en la práctica se queda
en ~1,2×. Es exactamente el resultado del benchmark publicado en MI300A (×2), que solo aparece a
64 M celdas porque allí el solve pesa relativamente más.

---

## 6. Veredicto

- **La GPU sí es una GPU, pero no tantas.** Con el bucle de Krylov residente y un multigrid de
  jerarquía fija, el bloque de presión a 2,1 M celdas cuesta **80,5 ms**: **1,72× menos que 32
  núcleos** y 43,6× menos que uno. Por iteración la ventaja es 5×, pero el multigrid barato necesita
  2,9× más iteraciones que GAMG.
- **El preconditioner es el 90 % de la historia.** Jacobi → MG es 6× menos tiempo (168 → 28 ms por
  solve frío) con el mismo hardware y sin setup. ILU(0) multicolor no compensa en GPU (su *apply* es
  disperso).
- **El ensamblado es el techo.** A 128³ el paso mejora **1,20×** con el bloque `p` en GPU (1,10× con
  64 rangos), y solo si la integración es gratis. Para superar eso hay que portar el ensamblado (la
  vía del paper, que solo paga a partir de 8 M celdas y con memoria unificada).
- **Requisito de ingeniería, no de hardware:** ningún esquema que haga un *round trip* por iteración
  puede ganar. La integración viable sería: matriz y vectores en VRAM, intercambio solo de halos
  (≈160 KB por iteración a 32 ranks) y reducciones globales de 2 escalares.

---

## 7. Limitaciones (honestas)

- Medido en **un proceso, sin MPI**. El caso MPI requiere intercambio de halos en dispositivo; no se
  implementó porque el techo de Amdahl (§5) ya lo desaconseja a esta escala.
- El multigrid explota que la malla es **estructurada** (agregados 2×2×2 sobre `i + 128j + 16384k`).
  Para mallas no estructuradas haría falta una agregación por grafo (factible, O(nnz), misma idea).
- Agregación **sin suavizar** (P constante): el ciclo V estacionario converge a ~0,71/ciclo, peor que
  un MG maduro (~0,1). Con PCG basta para 14 iteraciones, pero hay margen.
- La exportación de la matriz (0,45 s, una vez) y el *parseo* del fichero (191 MB) **no** están
  incluidos en los tiempos; el solve sí incluye la carga del CSR a VRAM.
- No es bit-reproducible respecto a la CPU (las reducciones cambian el orden de las sumas), igual
  que ya ocurría en §12/§13.

---

## 8. Reproducción

```bash
# 1) exportar la matriz real de p (una vez, dentro del caso 128³ con el plugin)
#    en system/fvSolution:  p { solver rocALUTION; rocPreconditioner Jacobi; rocDump true; }
#    -> roc_matrix_p.bin (n, nnz, rowPtr, colIdx, values, rhs)

# 2) compilar los tres bancos
cd /openfoamGPU/gpu_krylov
for b in bench_pcg bench_ilu bench_mg; do
  hipcc -O3 -std=c++17 $b.cpp -o $b -I/opt/rocm/include -L/opt/rocm/lib \
        -lrocsparse -lhipblas -Wl,-rpath,/opt/rocm/lib
done

# 3) medir (siempre GPU 0)
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
./bench_pcg roc_matrix_p.bin 1e-2 5000
./bench_ilu roc_matrix_p.bin 1e-2 2000
CF=2 SMOOTH=gs NU=2 ./bench_mg roc_matrix_p.bin 1e-2 500 128
```

Salida esperada (última línea de cada uno):

```
PCG(Jacobi) tol=1e-02 : iters=298  total=168.0 ms  per-iter=0.564 ms  SpMV=0.226 ms  res/||b||=9.7e-03
PCG(ILU0)   tol=1e-02 : iters=152  total=261.1 ms  per-iter=1.718 ms  SpMV=0.242 ms  res/||b||=9.7e-03
PCG(MG)     tol=1e-02 : iters=14   total=28.3 ms   per-iter=2.022 ms  res/||b||=8.3e-03
```

Diagnósticos incluidos en `bench_mg`: `MGONLY=n` (ciclos V puros), `CHECK=1` (identidad de
Galerkin), `GSOLVE=n` (Gauss-Seidel solo), `CF` (factor de agregación), `NU` (barridos), `SMOOTH=jac|gs`.
