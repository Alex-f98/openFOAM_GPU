#!/usr/bin/env python3
"""Create a rocALUTION GPU-pressure case from an existing cavity case.

Usage: mk_case.py SRC_CASE DST_CASE [STEPS]

The pressure equation is solved on GPU 0 with rocALUTION (CG + Ruge-Stueben
AMG, AMG hierarchy reused through UpdateValuesCSR/ReBuildNumeric). The
velocity equations stay on the CPU with symGaussSeidel.
"""

import os
import re
import shutil
import sys

SRC, DST = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
FULL = (len(sys.argv) > 4 and sys.argv[4] == "full")

if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)

# Drop old results and logs, keep the initial state
for e in os.listdir(DST):
    full = os.path.join(DST, e)
    if re.fullmatch(r"\d+(\.\d+)?", e) and e != "0":
        shutil.rmtree(full) if os.path.isdir(full) else os.remove(full)
    elif e.startswith("log."):
        os.remove(full)

# --- controlDict ---
p = os.path.join(DST, "system/controlDict")
s = open(p).read()
s = re.sub(
    r"^(application\s+\w+;)",
    r'\1\n\nlibs            ("librocALUTIONLduSolver.so");',
    s,
    count=1,
    flags=re.M,
)
s = re.sub(r"^endTime\s+[^;]+;", f"endTime         {0.005 * STEPS:g};", s, flags=re.M)
s = re.sub(r"^deltaT\s+[^;]+;", "deltaT          0.005;", s, flags=re.M)
s = re.sub(r"^writeInterval\s+[^;]+;", f"writeInterval   {STEPS};", s, flags=re.M)
s = re.sub(r"^writeControl\s+[^;]+;", "writeControl    timeStep;", s, flags=re.M)
open(p, "w").write(s)

# --- fvSolution: replace the whole solvers block ---
p = os.path.join(DST, "system/fvSolution")
s = open(p).read()
m = re.search(r"^solvers\s*\n\{", s, flags=re.M)
if not m:
    sys.exit("no solvers block found")

i = m.end() - 1
depth = 0
for j in range(i, len(s)):
    if s[j] == "{":
        depth += 1
    elif s[j] == "}":
        depth -= 1
        if depth == 0:
            break

block = """solvers
{
    p
    {
        solver                  rocALUTION;
        rocPreconditioner       AMG;
        rocDevice               0;
        rocTiming               true;
        tolerance               1e-06;
        relTol                  0.01;
        maxIter                 100;
    }

    pFinal
    {
        $p;
        relTol                  0;
    }

    U
    {
        solver                  rocALUTION;
        rocPreconditioner       AMG;
        rocDevice               0;
        rocTiming               true;
        tolerance               1e-06;
        relTol                  0;
        maxIter                 100;
    }
}""" if FULL else """solvers
{
    p
    {
        solver                  rocALUTION;
        rocPreconditioner       AMG;
        rocDevice               0;
        rocTiming               true;
        tolerance               1e-06;
        relTol                  0.01;
        maxIter                 100;
    }

    pFinal
    {
        $p;
        relTol                  0;
    }

    U
    {
        solver                  smoothSolver;
        smoother                symGaussSeidel;
        tolerance               1e-06;
        relTol                  0;
    }
}"""
s = s[: m.start()] + block + s[j + 1 :]
open(p, "w").write(s)

print(f"created {DST} ({STEPS} steps)")
