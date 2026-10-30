#!/usr/bin/env python3
"""Verifica el residual REAL de una solución de bench_mg contra la matriz FP64.

El banco mide el residual con su propio operador (el stencil que extrae del
CSR). Si los coeficientes se guardaron en FP32 (-DUSE_FLOAT_COEFFS) o hubo
algún error, ese residual puede mentir. Este script recalcula ||b - A x||/||b||
con la matriz CSR original en FP64 y con scipy.

Uso:
    ./bench_mg roc_matrix_p.bin 0.13 500 128            # (con XDUMP=x.bin)
    check_residual.py gpu_krylov/data/roc_matrix_p.bin x.bin [x2.bin ...]

Requiere numpy y scipy (instalados en esta máquina).
"""
import sys
import numpy as np
from scipy.sparse import csr_matrix

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

mat, xs = sys.argv[1], sys.argv[2:]

with open(mat, "rb") as f:
    head = np.fromfile(f, dtype=np.int64, count=2)
    n, nnz = int(head[0]), int(head[1])
    rp = np.fromfile(f, dtype=np.int32, count=n + 1)
    ci = np.fromfile(f, dtype=np.int32, count=nnz)
    val = np.fromfile(f, dtype=np.float64, count=nnz)
    b = np.fromfile(f, dtype=np.float64, count=n)

A = csr_matrix((val, ci, rp), shape=(n, n))
nb = np.linalg.norm(b)
print(f"matriz {mat}: n={n} nnz={nnz}  ||b||={nb:.3e}")

for path in xs:
    x = np.fromfile(path, dtype=np.float64)
    if x.size != n:
        print(f"  {path}: tamaño {x.size} != {n}, se ignora")
        continue
    r = np.linalg.norm(b - A @ x) / nb
    # la ecuación de presión es singular (constante libre): se compara también
    # el campo quitando su media
    xm = x - x.mean()
    print(f"  {path}: residual real = {r:.3e}   std(x)={x.std():.4f}")
