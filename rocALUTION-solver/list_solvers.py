#!/usr/bin/env python3
"""Lista los solvers / smoothers / preconditioners que OpenFOAM puede cargar.

OpenFOAM no tiene un comando para listarlos: las tablas de seleccion en tiempo
de ejecucion solo se imprimen cuando la busqueda falla. Este script copia el
caso a un directorio temporal, escribe a proposito un nombre invalido en cada
tabla y captura la lista del FatalIOError.

Uso:
    list_solvers.py CASE

CASE debe ser un caso que arranque (malla en constant/polyMesh y un 0/); si
controlDict carga librerias con `libs (...)`, esas entradas (p.ej. rocALUTION)
apareceran en las listas.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

CASE = os.path.abspath(sys.argv[1])

ENV = dict(os.environ)
ENV.update(
    ROCR_VISIBLE_DEVICES="0",
    HIP_VISIBLE_DEVICES="0",
    FOAM_SIGFPE="false",
    OMP_NUM_THREADS="8",
)

BASH = (
    "set +u; source /openfoamGPU/cpu/OpenFOAM-v2606/etc/bashrc; set -u; "
    "exec icoFoam"
)

# (etiqueta, campo, lineas que se escriben dentro del bloque del solver)
PROBES = [
    ("solver simetrico", "p",
     ["solver                  noSuchSolver;"]),
    ("solver asimetrico", "U",
     ["solver                  noSuchSolver;"]),
    ("precondicionador simetrico", "p",
     ["solver                  PCG;",
      "preconditioner          noSuchPrecon;"]),
    ("precondicionador asimetrico", "U",
     ["solver                  PBiCGStab;",
      "preconditioner          noSuchPrecon;"]),
    ("smoother simetrico", "p",
     ["solver                  smoothSolver;",
      "smoother                noSuchSmoother;"]),
    ("smoother asimetrico", "U",
     ["solver                  smoothSolver;",
      "smoother                noSuchSmoother;"]),
]


def patch(fvSolution, field, lines):
    """Sustituye el contenido del bloque `field { ... }` por `lines`."""
    txt = open(fvSolution).read()
    m = re.search(rf"^    {field}\s*\n    \{{", txt, flags=re.M)
    if not m:
        sys.exit(f"no encuentro el bloque '{field}' en {fvSolution}")

    i = m.end() - 1
    depth = 0
    for j in range(i, len(txt)):
        if txt[j] == "{":
            depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                break
    else:
        sys.exit("bloque sin cerrar")

    body = "".join(f"        {ln}\n" for ln in lines)
    new = f"    {field}\n    {{\n{body}    }}"
    open(fvSolution, "w").write(txt[: m.start()] + new + txt[j + 1:])


def run(case):
    p = subprocess.run(
        ["bash", "-c", BASH],
        cwd=case,
        env=ENV,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=600,
    )
    return p.stdout + p.stderr


lists = {}

for label, field, lines in PROBES:
    tmp = tempfile.mkdtemp(prefix="listsolv.")
    try:
        dst = os.path.join(tmp, "case")
        shutil.copytree(
            CASE, dst,
            ignore=shutil.ignore_patterns(
                "[0-9]*", "processor*", "log.*", "monitor_*.csv",
                "postProcessing", "VTK", "rc.*", "wall.*",
            ),
        )
        shutil.copytree(os.path.join(CASE, "0"), os.path.join(dst, "0"))

        patch(os.path.join(dst, "system/fvSolution"), field, lines)
        out = run(dst)

        m = re.search(r"Valid (.+?) types :\s*\n?\s*\d+\(([^)]*)\)", out)
        if not m:
            m = re.search(r"Valid (.+?) types include:\s*\n?\s*\d+\(([^)]*)\)", out)
        if m:
            lists[label] = (m.group(1), m.group(2).split())
        else:
            lists[label] = ("?", [f"ERROR: {out.strip().splitlines()[-1] if out.strip() else 'sin salida'}"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

print(f"# {CASE}\n")
for label, (desc, items) in lists.items():
    print(f"{label:30s} ({len(items):2d}): {' '.join(sorted(items))}")

allnames = sorted({i for _, items in lists.values() for i in items})
print(f"\n# nombres distintos en total: {len(allnames)}")
print("  " + " ".join(allnames))
