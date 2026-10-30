#!/usr/bin/env python3
"""Calcula el reparto de fases a partir de results.txt de run_diff.sh.

Uso: split.py [results.txt]

Imprime, por cada número de rangos, los segundos por paso de cada fase y su
porcentaje. Las fórmulas son las de la cabecera de run_diff.sh.
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/repro_diff/results.txt"

runs = {}
for line in open(path, errors="ignore"):
    m = re.search(r"np=(\d+) variant=(\w+) steps=(\d+) rc=(\d+) exec=([0-9.eE+-]+)", line)
    if not m:
        continue
    np_, var, steps, rc, ex = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), float(m.group(5))
    if rc != 0:
        print(f"  (aviso: np={np_} {var} terminó con rc={rc}; revise el log)")
    runs[(np_, var)] = ex / steps

for np_ in sorted({k[0] for k in runs}):
    try:
        g, p1, b1 = runs[(np_, "gamg")], runs[(np_, "p1")], runs[(np_, "both1")]
    except KeyError:
        print(f"np={np_}: faltan variantes")
        continue
    p, u, asm = g - p1, p1 - b1, b1
    print(f"\n=== {np_} rango(s): paso {g:.4f} s ===")
    for name, val in (("ensamblado + PISO + I/O", asm),
                      ("bloque p (p + pFinal)", p),
                      ("bloque U", u)):
        print(f"  {name:26s} {val:8.4f} s  {100*val/g:5.1f} %")
    print(f"  {'paso completo':26s} {g:8.4f} s  100.0 %")
    if p > 0 and asm > 0:
        print(f"  techo si el bloque p fuera gratis: {g/(g-p):.2f}x"
              f" | si p y U fueran gratis: {g/asm:.2f}x")
