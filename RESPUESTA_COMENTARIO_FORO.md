# El comentario del foro (L40S) aplicado a nuestro caso: qué es cierto y qué hemos medido aquí

**Fecha:** octubre 2026
**Contexto:** hilo de OpenFOAM sobre la descarga GPU de v2606 en una NVIDIA L40S (EPYC 7543 de 32
núcleos). El usuario pregunta si el resultado depende del hardware o de la implementación.
Este documento contrasta cada afirmación con medidas hechas en **esta** máquina (MI210 + 2× EPYC 7763).

---

## 0. Resumen en una tabla

| Afirmación del foro | ¿Cierto? | Qué dicen nuestros números |
|---|---|---|
| "La L40S solo tiene 1,431 TFLOPS FP64 (1:64)" | **Sí, exacto** | Dato oficial de la L40S. La MI210 tiene **11,3 TFLOPS FP64 (1:2)**, 8× más. Pero da igual: no es el límite (§2) |
| "PCIe 4.0 x16 limita" | **Sí** | Confirmado en esta máquina: **16 GT/s ×16**, 28 GB/s por sentido, 11,7 µs de latencia por llamada (§3) |
| "El movimiento de datos no solapado es el cuello" | **Sí, y es el nuestro** | Un *round trip* de un vector cuesta 1,2 ms; una iteración entera 2,06 ms (§3) |
| "MI300A funciona tan bien por la memoria unificada" | **Sí** | Cuantificado en §4: sería la única tarjeta que cambiaría el orden de magnitud |
| "Los casos pequeños/medianos van más rápido en CPU" | **Sí** | Medido: a 128³ la CPU gana el paso completo (1,20× a favor de la GPU *solo* en el solve, y solo si mover datos es gratis); a 256³ son ≈1,30× (§3-bis) |
| "El EPYC 7543 tiene 128 MB de L3" | **No: son 256 MB** | Milan (7543) = 8× 32 MB = 256 MB; los 128 MB son de la generación Rome. El argumento (caché enorme) sigue siendo válido |
| "¿Estoy haciendo algo mal en mi configuración?" | **No** | El régimen es el problema, no la configuración (§5) |

---

## 1. Qué cambia con otra GPU: el modelo que sale de nuestras medidas

Nuestro solve de presión es **memoria-intensivo**, no cálculo-intensivo: el SpMV mueve 175 MB de
matriz en 0,219 ms (≈800 GB/s solo contando la matriz, ~1,4 TB/s contando también el vector), y el
bucle completo sostiene ~1 TB/s. Es decir, **estamos cerca del ancho de banda de la MI210**, no del
pico de FP64 (usamos el 1 % de los 11,3 TFLOPS).

Eso permite predecir el efecto de cambiar de tarjeta con una regla simple: el tiempo escala con
`ancho de banda de pico`, manteniendo la misma eficiencia. Es una **estimación**, no una medida:

| GPU | BW de pico | bloque `p` (est.) | vs 32 núcleos EPYC | paso 32 ranks | ganancia del paso |
|---|---|---|---|---|---|
| **L40S** (foro) | 0,86 TB/s | ≈149 ms | **0,93×** (pierde) | 0,363 s | **0,97×** |
| MI210 (medido aquí) | 1,60 TB/s | **80,5 ms** | 1,72× | 0,294 s | 1,20× |
| A100 80 GB | 2,04 TB/s | ≈63 ms | 2,2× | 0,277 s | 1,27× |
| H100 SXM | 3,35 TB/s | ≈38 ms | 3,6× | 0,252 s | 1,40× |
| MI300A (memoria unificada) | 5,3 TB/s | ≈24 ms | 5,8× | 0,238 s | ≈1,48× |
| *techo: `p` gratis* | — | 0 | ∞ | 0,214 s | **1,65×** |
| *techo: `p` y `U` gratis* | — | 0 | ∞ | 0,110 s | **3,19×** |

Conclusión de la tabla: **el hardware cambia el resultado, pero no lo revoluciona**. Pasar de una
L40S a una H100 mejora el paso de 0,97× a 1,40×; ninguna tarjeta PCIe llega a 2× a esta escala. Y la
L40S, con su FP64 1:64 y sus 864 GB/s, **no puede ganar** en este régimen: su solve sería más lento
que el de 32 núcleos EPYC. Eso explica exactamente la captura del foro sin que el autor haya hecho
nada mal.

---

## 2. ¿Es el FP64 el problema? Prueba directa (medida)

Si el límite fuese la **capacidad de cálculo FP64** (la hipótesis del comentario, razonable para una
L40S con 1:64), pasar los coeficientes de la matriz a FP32 debería acelerar el kernel hasta ~64× en
una L40S y ~2× en una MI210 (por el ratio 1:2).

Lo medimos: misma malla, mismo algoritmo, mismos vectores en FP64, **solo los coeficientes de la
matriz en FP32** (16 B/celda en vez de 32 B/celda).

| Versión | iteraciones | bloque `p` | ms/iter | residual real (A en FP64) |
|---|---|---|---|---|
| Coeficientes FP64 | 39 | **80,5 ms** | 2,06 | 4,4e-7 |
| Coeficientes FP32 | 39 | **58,8 ms** | 1,51 | **9,4e-6** |

Resultados:

1. **Ganancia 1,37×, no 64×.** El kernel no está limitado por FP64: está limitado por los bytes que
   mueve. Al reducir los coeficientes a la mitad, el tráfico baja ~1/3 y el tiempo baja ~1,37×.
   Si el FP64 fuese el cuello, aquí se vería un salto de 2× (MI210 es 1:2) o de 64× en una L40S.
2. **El número de iteraciones no cambia** (39 → 39): la precisión de los coeficientes no afecta a la
   convergencia del multigrid, solo al residual alcanzable.
3. **Pero el residual real se degrada a 9,4e-6**, por encima de la tolerancia de `pFinal` (1e-6). El
   campo de presión difiere un 4,4e-4 en norma (quitando la constante, que es libre en esta
   ecuación). Es decir: la precisión mixta **sería utilizable solo bajando la tolerancia a ~1e-5**, o
   con refinamiento iterativo (que cuesta lo que ahorra).
4. Aun aceptándola, el paso mejoraría de 1,20× a 1,26×. No cambia la conclusión.

---

## 3. ¿Es el PCIe el problema? Medido en esta máquina

Enlace real de las dos MI210 (`/sys/bus/pci/devices/.../current_link_speed`): **16 GT/s, ancho 16** →
PCIe 4.0 x16, como decía el comentario. Medido con `gpu_krylov/bench_pcie.cpp`:

| Transferencia | pageable | pinned (DMA) |
|---|---|---|
| H2D 16,8 MB (un vector de `p` a 128³) | 27,8 GB/s | 26,1 GB/s |
| D2H 16,8 MB | 14,2 GB/s | 27,8 GB/s |
| H2D 175 MB (la matriz completa) | 28,1 GB/s | 28,0 GB/s |
| Latencia por llamada (8 B) | 11,7 µs | — |

Cuentas que importan:

- **Un *round trip* de un vector = 1,2 ms** (0,6 ms de ida + 0,6 ms de vuelta). Nuestra iteración
  completa en la GPU cuesta **2,06 ms**. Es decir: **un solo ida-y-vuelta por iteración duplica el
  coste** y se come la ventaja de 1,72×.
- Los tres vectores del algoritmo (r, p, Ap) suben y bajan: 3,6 ms por iteración, más que la
  iteración entera.
- La matriz entera son 12,5 ms de ida y vuelta: cargarla una vez es asumible (15 % del bloque), pero
  **reconstruirla cada paso** (lo que hace OpenFOAM con la malla descompuesta) es 0,4 s.

Esto es exactamente lo que midió la ruta B (`OPCION_B_MPI_GPU.md`): 10–77× más lenta por hacer un
*round trip* por iteración. Y es lo que dice el comentario: **el movimiento de datos que no se solapa
con el cálculo es un cuello enorme**.

---

## 3-bis. ¿La balanza cambia con el tamaño? Medido a 128³ y 256³

El comentario dice que los casos *pequeños y medianos* van mejor en CPU, dando a entender que con
tamaño la cosa cambia. Lo medimos: mismo caso, misma configuración, 32 rangos, malla 256³
(16,8 M celdas, 8× más que 128³, 117 M de no ceros).

| Fase (32 rangos) | 128³ (2,1 M celdas) | 256³ (16,8 M celdas) | factor por 8× celdas |
|---|---|---|---|
| ensamblado + PISO + I/O | 0,110 s | 1,226 s | 11,1× |
| bloque `p` (GAMG) | 0,1385 s | **2,111 s** | 15,2× |
| bloque `U` | 0,103 s | 3,334 s | 32,4× |
| **paso completo** | **0,352 s** | **6,671 s** | **19,0×** |
| **GPU, bloque `p`** | 80,5 ms (medido) | **≈0,58 s** (0,52–0,64) | **6,4–8×** |

Cómo se obtienen los dos últimos números:

- CPU: medida directa por diferencia de tres corridas (§2 de `COMPARACION_PRESION_GPU_CPU.md`),
  20 pasos completos. Nota de honestidad: la variante `p1iter` (presión con 1 iteración) divergió en
  el paso 18 de 20, así que su tiempo sale de los 17 pasos sanos (4,37–4,88 s/paso, muy estables);
  `gamg` y `both1iter` completaron los 20. El bloque `U` crece 32× porque a 256³ necesita 50
  iteraciones para llegar a 1e-6 (a 128³ bastaban 18–21), es decir, está rozando el límite de su
  solver.
- GPU: a 256³ no se pudo exportar la matriz real (el volcado en serie no terminó en 30 min), así que
  se midió el **escalado** con un Laplaciano 7-puntos sintético del mismo tamaño: 19,8 ms (128³) →
  127,5 ms (256³), es decir **6,4×** para 8× no ceros, con las mismas 10 iteraciones. Aplicando ese
  factor a los 80,5 ms reales: 0,52–0,64 s. Es una estimación **marcada como tal**, pero apoyada en
  una medida del kernel.

Lo que dicen los dos puntos:

1. **La CPU se degrada con el tamaño**: por celda, el paso pasa de 1,68e-7 s a 3,98e-7 s (2,4× peor).
   La GPU, en cambio, escala casi linealmente (6,4–8× para 8× celdas).
2. Por eso la **ventaja de la GPU en el bloque de presión sube de 1,72× a ≈3,5×** al multiplicar por
   8 las celdas. El comentario acierta en la dirección.
3. Pero **la ganancia del paso apenas se mueve: 1,20× → ≈1,30×**, porque el ensamblado y `U` crecen
   *aún más rápido* que la presión. El techo de Amdahl a 256³ es 6,671/1,226 = 5,4×, pero para
   acercarse habría que descargar también el ensamblado y `U`.

Conclusión: el tamaño ayuda, pero no gratis ni rápido. Duplicar la ventaja del solve exigió 8× celdas;
llegar a 2× en el paso completo exigiría un régimen en el que el ensamblado pese mucho menos, que es
justo lo que no ocurre con 32 rangos y una malla estructurada.

---

## 4. ¿Y la MI300A? Cuánto valdría de verdad

La memoria unificada elimina justo el coste de §3: no hay `memcpy`, el ensamblado puede escribir
directamente donde la GPU lee. Nuestra estimación (§1) le da **~24 ms** para el bloque de presión,
4,8× menos que la CPU de 32 núcleos, y un paso de **≈1,48×**.

Eso coincide con el único resultado publicado que gana claramente en MI300A (×2), y también con su
condición: **solo a partir de 8–64 M celdas**. La razón es la misma que aquí: a 2,1 M celdas el
ensamblado y la E/S pesan demasiado, y no caben 64 M celdas en 64 GB (la matriz serían ~166 GB).

> **Nota medida (oct-2026):** "memoria unificada" en la MI210 no es lo de la MI300A. Medimos
> `hipMallocManaged` en esta máquina: 815 GB/s si el buffer cabe en VRAM y se precarga a mano,
> pero **1,3–6,6 GB/s** cuando se deja a los *page faults* (y con `HSA_XNACK=1` se cuelga). En la
> MI300A no hay PCIe que cruzar. Detalle y cuentas del caso de 64 M celdas:
> [`MEMORIA_UNIFICADA_MI210.md`](MEMORIA_UNIFICADA_MI210.md).

---

## 5. ¿Está mal configurado el caso del foro?

Por lo que cuenta (Allrun, `fvSolution` modificado, malla del benchmark, L40S) no hay nada
"equivocado": el solve de presión es correcto y está bien precondicionado. Lo que ocurre es que:

1. una L40S tiene **864 GB/s** (1,9× menos que la MI210) y **1,43 TFLOPS FP64** (8× menos);
2. en ese régimen el ensamblado y el *overhead* dominan el paso (en nuestro caso, 31–45 %);
3. y el caso del benchmark es pequeño/mediano, justo donde la CPU con L3 grande gana.

Lo único que se podría mejorar en su configuración, y no es hardware: usar una **malla mayor** (si la
memoria lo permite) y **no** mover la matriz por PCIe en cada paso. Con 64 M celdas y memoria
unificada es donde el resultado se invierte; con una L40S por PCIe, no.

---

## 6. Nuestro caso es el mejor escenario posible para la GPU, y aun así no gana

Conviene decirlo porque refuerza la conclusión: `cavity3D` es una **caja estructurada de un solo
bloque** con dos tipos de parche; el intercambio de halos a 32 rangos sería de ~160 KB por iteración
(el 1 % de un vector). Es el caso *más favorable* que existe para descargar álgebra a una GPU:

- no hay mallas no estructuradas,
- no hay muchos parches (el problema que menciona el comentario en el caso de la moto),
- la matriz es un Laplaciano de 7 puntos perfectamente regular,
- y el multigrid explota esa regularidad.

Y aun así el resultado es 1,72× en el solve y **1,20× en el paso**. Si en el mejor de los casos no se
gana más, en un caso real con muchos parches y malla no estructurada tampoco.

---

## 7. Reproducción

```bash
# PCIe
cd /openfoamGPU/gpu_krylov
hipcc -O3 -std=c++17 bench_pcie.cpp -o bench_pcie -I/opt/rocm/include -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ./bench_pcie 16777216
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ./bench_pcie 175000000

# FP32 vs FP64 en los coeficientes
hipcc -O3 -std=c++17 bench_mg.cpp -o bench_mg -I/opt/rocm/include -L/opt/rocm/lib -lrocsparse -lhipblas -Wl,-rpath,/opt/rocm/lib
hipcc -O3 -std=c++17 -DUSE_FLOAT_COEFFS bench_mg.cpp -o bench_mg_f32 -I/opt/rocm/include -L/opt/rocm/lib -lrocsparse -lhipblas -Wl,-rpath,/opt/rocm/lib
cd /tmp/opencode/precond/r128
CF=2 SMOOTH=gs NU=2 WARM=1.3e-3,1e-6 XDUMP=x_fp64.bin /openfoamGPU/gpu_krylov/bench_mg     roc_matrix_p.bin 0.13 500 128
CF=2 SMOOTH=gs NU=2 WARM=1.3e-3,1e-6 XDUMP=x_f32.bin  /openfoamGPU/gpu_krylov/bench_mg_f32 roc_matrix_p.bin 0.13 500 128
# residual real de cada solución contra la matriz FP64: ver script en el anexo del informe
```
