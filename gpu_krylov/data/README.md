# `gpu_krylov/data/` — datos de entrada de los bancos

Aquí va la matriz dispersa que leen `bench_pcg`, `bench_ilu` y `bench_mg`.

**No se versiona** (191 MB, y supera el límite de 100 MB por fichero de GitHub).
Se regenera en ~1 minuto con el plugin rocALUTION:

```bash
cd /openfoamGPU/repro && ./dump128.sh /tmp/dump128
cp /tmp/dump128/roc_matrix_p.bin /openfoamGPU/gpu_krylov/data/
```

Formato (little-endian, todo seguido):

| Campo | Tipo | Tamaño |
|---|---|---|
| `n` (celdas) | int64 | 8 B |
| `nnz` | int64 | 8 B |
| `rowPtr` | int32 × (n+1) | 4·(n+1) B |
| `colIdx` | int32 × nnz | 4·nnz B |
| `values` | float64 × nnz | 8·nnz B |
| `rhs` | float64 × n | 8·n B |

Para el caso 128³: `n = 2.097.152`, `nnz = 14.581.760`.

Si no quieres depender del plugin, `repro/gen_poisson.py` genera un Laplaciano de
7 puntos del mismo tamaño en el mismo formato (sirve para medir el escalado del
kernel, no para reproducir los tiempos exactos del caso real):

```bash
python3 /openfoamGPU/repro/gen_poisson.py 128 /tmp/poisson128.bin
```
