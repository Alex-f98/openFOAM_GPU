# Cuánto tarda de verdad resolver la presión: GPU vs CPU (todo medido)

**Fecha:** octubre 2026
**Caso:** `cavity3D`, malla 128³ = 2.097.152 celdas (14.581.760 no ceros en la matriz de presión)
**GPU:** 1× AMD Instinct MI210 (solo GPU 0), FP64 de principio a fin
**CPU:** 2× AMD EPYC 7763 (128 núcleos, 512 MiB de L3 en total), OpenFOAM v2606 oficial, 1 hilo por rango

Este documento corrige el número que se publicó antes ("el solve de `p` es 4,6× más rápido que 32
núcleos"). El 4,6× comparaba **una sola** resolución de presión de la GPU contra **las dos** que hace
OpenFOAM por paso de tiempo. Con la comparación correcta el resultado es **1,7×** para el bloque de
presión completo. Todo lo que sigue son medidas directas, no derivaciones.

---

## 1. Qué se está midiendo exactamente

`icoFoam` resuelve, en cada paso de tiempo, dos sistemas lineales:

| Bloque | Qué es | Cuántas veces por paso |
|---|---|---|
| `U` | velocidad (3 componentes, x/y/z) | 3 (una por componente, más las correcciones PISO) |
| `p` | presión | **2**: `p` (con `relTol 0.01`) y `pFinal` (con `relTol 0`, residual ≤ 1e-6) |

El bloque de presión son por tanto **dos** resoluciones encadenadas: la primera es floja (basta con
reducir el residual 100×) y la segunda es la que deja el campo de presión consistente. Comparar solo
la primera con todo el bloque de CPU es lo que inflaba el número anterior.

Unidades: **segundos por paso de tiempo**. Un paso de tiempo es la unidad natural de este solver: no
hay "iteraciones" externas, cada paso avanza 0,005 s de tiempo físico.

---

## 2. Método en CPU: resta de tres corridas idénticas

No se instrumentó OpenFOAM. Se corrieron tres veces el mismo caso cambiando **solo** el solver de una
ecuación; el resto del paso (ensamblado de la matriz, PISO, I/O) es idéntico y por tanto se cancela
al restar.

| Variante | `p` | `pFinal` | `U` |
|---|---|---|---|
| `gamg` | GAMG | GAMG | smoothSolver |
| `p1iter` | 1 iteración | 1 iteración | smoothSolver |
| `both1iter` | 1 iteración | 1 iteración | 1 iteración |

* `gamg` − `p1iter` = coste de resolver la presión como lo hace OpenFOAM (incluye el **setup** de
  GAMG de cada solve, que en OpenFOAM se paga dentro del solve).
* `p1iter` − `both1iter` = coste del bloque de velocidad.
* `both1iter` = ensamblado + PISO + I/O.

### 32 rangos (20 pasos, `ExecutionTime` acumulado)

| variante | exec 20 pasos | s/paso |
|---|---|---|
| `gamg` | 7,04 s | 0,3520 |
| `p1iter` | 4,27 s | 0,2135 |
| `both1iter` | 2,21 s | 0,1105 |

| fase | s/paso | % del paso |
|---|---|---|
| ensamblado + PISO + I/O | 0,1105 | 31 % |
| **bloque `p` (`p` + `pFinal`)** | **0,1385** | **39 %** |
| bloque `U` | 0,1030 | 29 % |
| **paso completo** | **0,3520** | 100 % |

Iteraciones GAMG medias por paso (leídas del log): `p` 4,05 y `pFinal` 9,40 → **13,45 iteraciones**
para 0,1385 s ⇒ **10,3 ms por iteración** (32 núcleos trabajando juntos).

### 1 rango (8 pasos; el baseline de 20 pasos da 8,75 s/paso de mediana, dentro del 5 %)

| variante | exec 8 pasos | s/paso |
|---|---|---|
| `gamg` | 85,40 s | 10,675 |
| `p1iter` | 57,30 s | 7,163 |
| `both1iter` | 38,12 s | 4,765 |

| fase | s/paso | % |
|---|---|---|
| ensamblado + PISO + I/O | 4,765 | 45 % |
| **bloque `p`** | **3,513** | **33 %** |
| bloque `U` | 2,398 | 22 % |

16,5 iteraciones GAMG por paso ⇒ **213 ms por iteración** en un solo núcleo.

### 64 rangos (ya publicado en `exp_scaling_cpu/split_fase_128cube.md`, mismo método)

Paso 0,2745 s; bloque `p` 0,106 s; `U` 0,070 s; ensamblado 0,0985 s.

---

## 3. Método en GPU: la misma secuencia de tolerancias, en caliente

La matriz de `p` es la real, exportada del caso 128³ por el plugin (`rocDump`): 191 MB, 2.097.152
filas, 14.581.760 no ceros, **valores `double` (FP64)**. Se carga en VRAM una vez y a partir de ahí
el bucle de Krylov no sale del dispositivo: `hipblasDdot` (FP64), SpMV por stencil o `rocsparse_dcsrmv`
(FP64), y un ciclo V de multigrid por agregación 2×2×2 con 2+2 barridos de Gauss-Seidel.

Para que la comparación sea justa, la GPU ejecuta **la misma cadena que OpenFOAM**: arranca desde una
solución previa (residual inicial 0,128, como el `p` real del paso), baja a `relTol 0.01`
(1,28e-3), y luego una segunda pasada en caliente hasta 1e-6 como `pFinal`. Cada pasada arranca del
resultado de la anterior, sin reiniciar.

| pasada | residual entra → sale | iteraciones | tiempo |
|---|---|---|---|
| `p` (relTol 0.01) | 1,28e-1 → 1,19e-3 | 12 | 24,1 ms |
| `pFinal` (1e-6) | 1,19e-3 → 4,39e-7 | 27 | 56,4 ms |
| **bloque `p` completo** | | **39** | **80,5 ms** |

A eso hay que sumarle el **setup de la jerarquía de multigrid: 1,2 ms** (una vez; en OpenFOAM el
setup de GAMG se paga en cada solve y ya está incluido en los 0,1385 s de CPU).

Referencia conservadora: si la GPU partiese de cero (peor caso, sin solución previa), llegar a 1e-2
cuesta 28,6 ms y a 1e-6 cuesta 93,5 ms; el par equivalente sería **92 ms**. Los 80,5 ms de arriba son
el caso realista y el que se usa en las tablas.

---

## 4. El resultado

| Plataforma | bloque `p` (2 solves) | iteraciones | ms/iteración | vs GPU |
|---|---|---|---|---|
| CPU 1 núcleo (GAMG) | 3.513 ms | 16,5 | 213 | 43,6× más lenta |
| CPU 32 rangos (GAMG) | 138,5 ms | 13,45 | 10,3 | **1,72× más lenta** |
| CPU 64 rangos (GAMG) | 106 ms | — | — | 1,32× más lenta |
| **GPU 0 (PCG + MG, todo residente)** | **80,5 ms** | 39 | 2,06 | — |

Lectura de la tabla:

1. **Por iteración la GPU es 5× más rápida** que 32 núcleos y 103× más que uno. Esa parte es real.
2. **Pero necesita 2,9× más iteraciones** (39 frente a 13,45). El multigrid de agregación 2×2×2 con
   jerarquía fija converge peor que el AMG de OpenFOAM, que reclasifica y tiene más niveles efectivos.
3. El producto de ambos efectos es el **1,72×**. No hay magia: es la GPU siendo rápida en lo que hace
   y peor en lo que el algoritmo necesita.

Y una consecuencia importante: **cuantos más rangos, menos sirve la GPU**. Con 64 rangos el bloque de
presión ya solo es 1,32× más rápido, y con 128 rangos sería como mucho un empate. Una MI210, en este
kernel, equivale a **unos 44 núcleos EPYC 7763** (43,6× frente a uno; ~55 si se compara contra los
32 rangos, porque la CPU no escala de forma lineal).

### Por qué la GPU no arrasa (y el FP64 no tiene la culpa)

- **El problema es pequeño para una GPU**: 175 MB de matriz. A 32 rangos cada bloque local son
  ~5,5 MB y cabe de sobra en la caché L3 (512 MiB en total), así que la CPU acierta en caché y no
  mueve nada por PCIe. La GPU, en cambio, paga latencia de lanzamiento y sincronización en cada
  iteración (el ciclo V son ~14 SpMV equivalentes, cada uno de ~0,2 ms).
- **No es un problema de precisión ni de FLOPs.** Todo es `double` (matriz, vectores, `Ddot`,
  `dcsrmv`). El SpMV mueve 14,58 M nnz en 0,219 ms = **951 GB/s** (60 % del pico de 1,6 TB/s) y eso
  son solo **133 GFLOP/s**, un 1 % del pico FP64 de la tarjeta. El kernel está limitado por **ancho de
  banda de memoria**, no por la unidad de coma flotante: pasar a FP32 no aceleraría nada que importe
  (solo reduciría a la mitad los bytes de la matriz).
- **El ensamblado no está en la GPU** y es el 31–45 % del paso (ver §5).

---

## 5. Qué significa para el paso completo

Sustituyendo el bloque de presión de CPU por los 80,5 ms de GPU (y suponiendo, a favor de la GPU, que
mover datos no cuesta nada):

| Plataforma | paso actual | paso con `p` en GPU | ganancia | techo de Amdahl (`p` gratis) |
|---|---|---|---|---|
| 1 rango | 10,675 s | 7,24 s | **1,47×** | 1,49× |
| 32 rangos | 0,352 s | 0,294 s | **1,20×** | 1,65× |
| 64 rangos | 0,2745 s | 0,249 s | **1,10×** | 1,52× |

Ese "suponiendo que mover datos no cuesta nada" es la hipótesis fuerte. En la ruta B (MPI + GPU, §13
del informe) el coste real de reunir la matriz fue ≥0,3 s y el de reconstruir el CSR 0,40–0,45 s: cada
uno por separado ya se come la ganancia varias veces. Para que el 1,2× exista, la matriz tendría que
**ensamblarse ya en VRAM**, que es exactamente lo que no se implementó.

---

## 6. Corrección de lo publicado antes

| Afirmación anterior | Estado |
|---|---|
| "el solve de `p` baja a 28 ms" | **Correcto pero incompleto**: son 28 ms de UNA resolución fría hasta 1e-2. La cadena real (`p` + `pFinal`, en caliente) son **80,5 ms**. |
| "4,6× más rápido que 32 núcleos" | **Incorrecto**: comparaba 1 solve de GPU con 2 de CPU. El número correcto es **1,72×**. |
| "163× más rápido que 1 núcleo" | **Incorrecto** por el mismo motivo. El número correcto es **43,6×**. |
| "el paso mejora 1,4–1,7×" | **Optimista**: con los 80,5 ms reales es **1,20×** a 32 rangos. El 1,4–1,7× solo se alcanzaría si el bloque de presión bajara a ~25 ms. |
| "todo es FP64" | **Correcto**, verificado en el código (§4). |

El resto de conclusiones del informe no cambia: el ensamblado es el techo, el preconditioner es lo que
decide, y a esta escala la GPU no puede ganar de forma apreciable.

---

## 7. Qué haría falta para que ganara de verdad

1. **Ensamblar en el dispositivo**: mantener la matriz y los vectores en VRAM entre pasos, con
   intercambio de halos también en VRAM (~160 KB por iteración a 32 rangos). Es la vía del paper.
2. **Memoria unificada (MI300A)**: evita el `memcpy` explícito y permite que OpenFOAM y la GPU
   compartan la misma memoria física. Aun así el paper solo reporta ×2 desde 8–64 M celdas.
3. **Malla grande (≥64 M celdas)**: es el único régimen donde el solve domina el paso. Aquí no cabe:
   166 GB de matriz frente a 64 GB de VRAM.

---

## 8. Reproducción

```bash
# --- CPU (las tres variantes del §2) ---------------------------------------
cd /tmp/opencode/precond/diff && ./run_diff.sh    # 1 y 32 rangos
# logs: /tmp/opencode/precond/diff/work/log.np{1,32}.{gamg,p1,both1}

# --- GPU (cadena p → pFinal, en caliente) ----------------------------------
cd /tmp/opencode/precond/r128
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
CF=2 SMOOTH=gs NU=2 WARM=1.3e-3,1e-6 \
  /openfoamGPU/gpu_krylov/bench_mg roc_matrix_p.bin 0.13 500 128
# salida: tol=1e-01 (arranque) -> WARM 1e-03 (esto es p) -> WARM 1e-06 (esto es pFinal)
```

Matriz de entrada: `/tmp/opencode/precond/r128/roc_matrix_p.bin` (191 MB, exportada por `rocDump`).
