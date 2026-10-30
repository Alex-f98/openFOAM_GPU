# `repro/` — los experimentos, reproducibles

Todo lo de aquí sirve para volver a obtener los números de
[`COMPARACION_PRESION_GPU_CPU.md`](../COMPARACION_PRESION_GPU_CPU.md) y de
[`RESPUESTA_COMENTARIO_FORO.md`](../RESPUESTA_COMENTARIO_FORO.md).

| Archivo | Qué hace | Duración |
|---|---|---|
| `diff_cpu/run_diff.sh` | Reparto de fases CPU (`ensamblado` / `p` / `U`) por diferencia de 3 variantes. Acepta la lista de rangos: `./run_diff.sh 32 1` | ~3 min |
| `diff_cpu/split.py` | Convierte `results.txt` en la tabla de fases con porcentajes | instantáneo |
| `diff_cpu/fv_gamg`, `fv_p1`, `fv_both1` | Los tres `fvSolution` del método | — |
| `diff_cpu/logs/` | Logs originales de la medición publicada (1 y 32 rangos) | — |
| `diff_cpu/results.txt` | Resultado original de esos logs | — |
| `scale_256/run_256.sh` | Lo mismo a 256³ (16,8 M celdas). Necesita ~7 GB de disco y ~40 GB de RAM | ~12 min |
| `scale_256/log.np32.*` | Logs originales de la corrida 256³ publicada | — |
| `gen_poisson.py` | Genera un Laplaciano 7-puntos N³ en el formato de `roc_matrix_p.bin` (escalado del kernel sin necesidad del caso real) | segundos |
| `check_residual.py` | Recalcula el residual **real** de una solución de `bench_mg` contra la matriz FP64 (con scipy) | segundos |
| `dump128.sh` | Exporta la matriz de `p` del caso 128³ a `roc_matrix_p.bin` (necesita el plugin rocALUTION; tarda ~1 min) | ~1 min |
| `bench_managed.cpp` | Ancho de banda de memoria gestionada (`hipMallocManaged`): con y sin precarga, en VRAM y fuera de ella | ~2 min |
| `results_managed.txt` | Salidas crudas de `bench_managed` (base de [`MEMORIA_UNIFICADA_MI210.md`](../MEMORIA_UNIFICADA_MI210.md)) | — |

## Ejemplos

```bash
# reparto de fases a 32 y 1 rango
cd /openfoamGPU/repro/diff_cpu && ./run_diff.sh 32 1

# malla grande
cd /openfoamGPU/repro/scale_256 && ./run_256.sh 256 32 20

# escalado del kernel de la GPU sin el caso real
python3 /openfoamGPU/repro/gen_poisson.py 256 /tmp/poisson256.bin
cd /tmp && CF=2 SMOOTH=gs NU=2 WARM=1.3e-3,1e-6 \
  /openfoamGPU/gpu_krylov/bench_mg /tmp/poisson256.bin 0.13 500 256

# memoria unificada: ¿qué tan rápido es hipMallocManaged aquí?
cd /openfoamGPU/repro && \
  hipcc -O3 -std=c++17 bench_managed.cpp -o bench_managed \
        -I/opt/rocm/include -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
MALLOC=device ./bench_managed 4 6     # referencia VRAM      (~830 GB/s)
PREFETCH=1    ./bench_managed 4 6     # managed bien usado   (~815 GB/s)
              ./bench_managed 4 6     # managed a secas      (1,3-6,6 GB/s)

# ¿el residual que reporta el banco es el de verdad?
# (la matriz no está en el repo: ./dump128.sh la genera en ~1 min)
cd /tmp/dump128 && XDUMP=x.bin CF=2 SMOOTH=gs NU=2 WARM=1.3e-3,1e-6 \
  /openfoamGPU/gpu_krylov/bench_mg roc_matrix_p.bin 0.13 500 128
python3 /openfoamGPU/repro/check_residual.py roc_matrix_p.bin x.bin
```

Nota: los scripts escriben en `/tmp/opencode/...` por defecto (se puede cambiar con
`BASE=/otra/ruta`). No dejan nada dentro del repositorio salvo los logs que ya están guardados.
