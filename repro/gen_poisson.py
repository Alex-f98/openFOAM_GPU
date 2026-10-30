#!/usr/bin/env python3
"""Escribe un Laplaciano de Neumann 7-puntos uniforme N^3 en el formato de
roc_matrix_p.bin: int64 n, int64 nnz, int32 rowPtr[n+1], int32 colIdx[nnz],
float64 values[nnz], float64 rhs[n].

Uso: gen_poisson.py N OUT.bin [k]
"""
import sys
import numpy as np

N = int(sys.argv[1])
out = sys.argv[2]
k = float(sys.argv[3]) if len(sys.argv) > 3 else 5.5e-6
n = N**3

idx = np.arange(n, dtype=np.int64).reshape(N, N, N)

R = []
C = []
for ax in range(3):
    a = [slice(None)] * 3
    b = [slice(None)] * 3
    a[ax] = slice(0, N - 1)
    b[ax] = slice(1, N)
    ia = idx[tuple(a)].ravel()
    ib = idx[tuple(b)].ravel()
    R += [ia, ib]
    C += [ib, ia]

R = np.concatenate(R)
C = np.concatenate(C)
V = np.full(R.size, -k)

# diagonal = -(suma de los |offdiag|) de la fila
diag = np.bincount(R, minlength=n).astype(np.float64) * k

alln = np.arange(n, dtype=np.int64)
R = np.concatenate([R, alln])
C = np.concatenate([C, alln])
V = np.concatenate([V, diag])

order = np.lexsort((C, R))
R, C, V = R[order], C[order], V[order]

rp = np.zeros(n + 1, dtype=np.int64)
np.cumsum(np.bincount(R, minlength=n), out=rp[1:])

# rhs consistente: b = A xs con xs suave
# campo aleatorio reproducible (espectro ancho, como un residuo real)
xs = np.random.default_rng(12345).random(n) - 0.5
b = np.bincount(R, weights=V * xs[C], minlength=n)[:n]

with open(out, "wb") as f:
    np.array([n, rp[-1]], dtype=np.int64).tofile(f)
    rp.astype(np.int32).tofile(f)
    C.astype(np.int32).tofile(f)
    V.tofile(f)
    b.tofile(f)

print(f"escrito {out}: n={n} nnz={rp[-1]} ||b||={np.linalg.norm(b):.3e}")
