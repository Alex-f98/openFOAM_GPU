# ¿Se puede usar memoria unificada en la MI210?

**Respuesta corta:** sí, técnicamente se puede (`hipMallocManaged` funciona), y lo medimos:
**mientras los datos caben en la VRAM rinde igual que la VRAM (815 vs 829 GB/s)**. Pero eso no
es la memoria unificada de la MI300A y **no cambia nada de este problema**: en el régimen donde
la memoria unificada importaría (mallas de decenas de millones de celdas, que no caben en 64 GB)
el rendimiento se hunde a **1,3–6,6 GB/s** (medido), porque los datos van por PCIe.

Este documento responde a la pregunta que va a hacer todo el que llegue: *"¿y no puedes usar
memoria unificada y ya está?"*.

---

## 1. Hay dos cosas que se llaman "memoria unificada"

| | MI210 (esta máquina) | MI300A (la del paper que publica ×2) |
|---|---|---|
| Memoria física | 64 GB HBM en la GPU **+** 247 GB DDR4 en la CPU, **separadas** | 128 GB HBM **compartidos** por CPU y GPU |
| Qué hace la "unificada" | **emula** un espacio único y mueve páginas de 4 KB cuando faltan | no hay nada que mover: es el mismo silicio |
| Velocidad de ese espacio | **28 GB/s** (PCIe 4.0 ×16, medido) | **5,3 TB/s** (HBM) |
| Coste de fallar una página | *page fault* + copia por PCIe | cero |

En la MI210 la memoria unificada **no elimina las copias: las automatiza**. Te ahorra escribir
`hipMemcpy` a mano, no el tiempo de mover los datos. Y el *page fault* suele costar **más** que
una copia explícita, porque migra en trozos pequeños.

La MI210 **sí** tiene el soporte de hardware (gfx90a incluye XNACK), así que funciona. El
problema no es que no se pueda: es que la física del enlace es la misma.

---

## 2. Lo que se midió en esta máquina

Banco: [`repro/bench_managed.cpp`](repro/bench_managed.cpp) — un AXPY (`y = 0.5·x + y`) sobre
4 GB × 2 buffers. Salidas crudas en [`repro/results_managed.txt`](repro/results_managed.txt).

| Modo | 1.ª pasada (migración) | En caliente (mediana) | vs VRAM |
|---|---|---|---|
| `hipMalloc` (VRAM, referencia) | 323 GB/s | **829 GB/s** | 1× |
| `hipMallocManaged` + `hipMemPrefetchAsync` | 1053 GB/s *(ya estaba)* | **815 GB/s** | 1,02× *más lento* |
| `hipMallocManaged` a secas | 0,6 GB/s | **1,3–6,6 GB/s** | **126–638× más lento** |
| `hipMallocManaged` con `HSA_XNACK=1` | **no termina en 120 s** (4 GB) | — | inusable |

Dos detalles que importan:

- **El que rinde es el que se pre-carga a mano.** Sin `hipMemPrefetchAsync`, el hardware va
  migrando páginas de 4 KB bajo demanda y el resultado es catastrófico *aunque los datos
  quepan*. Además no es estable: la misma prueba dio **1,3 GB/s y 6,6 GB/s** en dos corridas
  (5× de diferencia). Un número así no sirve para planificar nada.
- **Precargar cuesta.** Pasar 8 GB con `hipMemPrefetchAsync` tardó **7,4 s** (~1,1 GB/s
  efectivos), muy por debajo de los 28 GB/s que da una copia grande por PCIe: el poblado de
  páginas (2 M páginas de 4 KB) domina. Se paga una vez por buffer y por paso, pero se paga.

---

## 3. Los tres regímenes (y por qué ninguno ayuda aquí)

**A. El working set cabe en VRAM** (hasta ~20 M celdas; a 128³ el estado ocupa 6,7 GB de 64).
→ La memoria unificada **no aporta nada**: en el bucle interior no hay copias que quitar, los
datos ya están en la GPU. Mides 815 en vez de 829 GB/s (un 2 % peor). Cero ganancia.

**B. Cabe, pero nadie precarga** (el caso típico de un port rápido).
→ **1,3–6,6 GB/s**. Es la peor opción de todas, peor que no usar la GPU.

**C. No cabe** (las 64 M celdas del paper, ~166 GB de estado frente a 64 GB de VRAM).
→ Funciona, pero **cada pasada paga PCIe**. Y aquí el PCIe no da ni los 28 GB/s: con el
comportamiento medido en B, el presupuesto realista es el de B, no el de `bench_pcie`.

Cuentas del caso C (estimaciones, con lo medido):

| | Valor | Origen |
|---|---|---|
| Estado de la simulación a 64 M celdas | ~166 GB (2,6 KB/celda) | `OPCION_B_MPI_GPU.md` |
| Exceso sobre VRAM | ~102 GB | 166 − 64 |
| Tráfico PCIe mínimo por pasada | ~204 GB (saca 102 + mete 102) | — |
| Tiempo por pasada a 28 GB/s (caso ideal) | **~7,3 s** | 204 GB ÷ 28 GB/s |
| Tiempo por pasada al 1,3 GB/s medido | **~157 s** | 204 GB ÷ 1,3 GB/s |
| Pasadas que necesita el bloque `p` (13 iteraciones × 3 niveles, orden de magnitud) | ~40 | `KRYLOV_RESIDENTE_GPU.md` |

→ Incluso en el caso **ideal** (PCIe a plena velocidad, sin solapamiento perdido), el bloque de
presión costaría **~5 minutos** por paso. La CPU hace ese mismo paso en ~25 s (extrapolación
lineal desde los 6,67 s medidos a 256³, 16,8 M celdas). La memoria unificada **no arregla el
exceso**: lo reparte por el cable.

---

## 4. Qué haría falta para que la memoria unificada ganara

1. **Que la GPU y la CPU compartan el mismo silicio** (MI300A, MI325X no; es la APU). Ahí no
   hay PCIe que cruzar y el ensamblado puede escribir donde la GPU lee. Es lo que usa el paper
   que reporta ×2.
2. **O que el working set quepa**, que es el caso A: y entonces da igual, porque la GPU ya
   tenía esos datos.

No hay una tercera vía. Con una tarjeta PCIe, "memoria unificada" es una comodidad de
programación, no una fuente de rendimiento.

---

## 5. Cómo reproducirlo

```bash
cd /openfoamGPU/repro
hipcc -O3 -std=c++17 bench_managed.cpp -o bench_managed \
      -I/opt/rocm/include -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib

export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0

MALLOC=device ./bench_managed 4 6                  # referencia VRAM
PREFETCH=1    ./bench_managed 4 6                  # unificada, bien usada
              ./bench_managed 4 6                  # unificada, a secas
HSA_XNACK=1   ./bench_managed 4 2                  # (se cuelga; Ctrl-C)

# caso C (no cabe): 2 x 40 GB > 64 GB de VRAM. NO EJECUTADO.
# Requiere ~90 GB de RAM y ~15-30 min. Se dejó sin correr a propósito.
              ./bench_managed 40 3
```

**Por qué no está ejecutado el caso C:** necesita ~90 GB de RAM en una máquina compartida y,
según las cifras de §3, el resultado ya está acotado entre 7 s y 157 s por pasada. Si alguien
quiere el número exacto, que lo corra en una ventana sin otros usuarios.

---

## 6. Resumen para el que llegue

> La MI210 soporta memoria unificada y funciona, pero **rinde a velocidad de VRAM solo si los
> datos caben y se precargan a mano**. En cuanto el problema no cabe (el único régimen donde
> serviría), cada pasada cruza el PCIe a 1,3–6,6 GB/s medidos. La memoria unificada que hace
> ganar ×2 es la de la MI300A, donde la CPU y la GPU comparten la misma HBM. **No es una opción
> de configuración: es otro hardware.**

Relacionado: [`RESPUESTA_COMENTARIO_FORO.md`](RESPUESTA_COMENTARIO_FORO.md) §4 (extrapolación a
otras tarjetas), [`TRASPASO.md`](TRASPASO.md) §9 (pendientes).
