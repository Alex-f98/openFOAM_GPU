#!/usr/bin/env python3
"""Compare internalField of two OpenFOAM ascii fields.

Usage: cmp_fields.py FIELD_A FIELD_B
Prints max/mean relative difference of the internal field.
"""
import re
import sys
import numpy as np


def read_internal(path):
    txt = open(path, errors='ignore').read()

    m = re.search(r'internalField\s+nonuniform\s+List<(\w+)>', txt)
    if not m:
        raise SystemExit(f'{path}: no nonuniform internalField')

    kind = m.group(1)
    rest = txt[m.end():]

    m2 = re.search(r'^\s*\(', rest, re.M)
    lst = rest[m2.start():]

    depth = 0
    for i, c in enumerate(lst):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                body = lst[1:i]
                break
    else:
        raise SystemExit(f'{path}: unbalanced list')

    if kind == 'vector':
        vals = re.findall(
            r'\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)', body)
        return np.array([[float(a), float(b), float(c)] for a, b, c in vals])
    elif kind == 'scalar':
        vals = re.findall(r'([-+0-9.eE]+)', body)
        return np.array([float(v) for v in vals])
    else:
        raise SystemExit(f'{path}: unsupported type {kind}')


a = read_internal(sys.argv[1])
b = read_internal(sys.argv[2])

if a.shape != b.shape:
    raise SystemExit(f'shape mismatch {a.shape} vs {b.shape}')

d = a - b
scale = np.max(np.abs(b))
rel = np.max(np.abs(d))/max(scale, 1e-300)

print(f'{sys.argv[1]} vs {sys.argv[2]}')
print(f'  cells        : {a.shape[0]}')
print(f'  max |diff|   : {np.max(np.abs(d)):.6e}')
print(f'  max |ref|    : {scale:.6e}')
print(f'  rel max diff : {rel:.6e}')
print(f'  rms diff     : {np.sqrt(np.mean(d**2)):.6e}')
