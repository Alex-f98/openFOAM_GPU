#!/usr/bin/env python3
"""Escribe un system/decomposeParDict por stdout para NX^3 celdas y NP procesos.

Usa el metodo 'simple' (no requiere SCOTCH/METIS) y elige el reparto (px,py,pz)
mas cubico posible tal que cada factor divida NX.

Sale con codigo 3 si no existe reparto valido (p.ej. NP primo > NX).
"""

import sys

nx = int(sys.argv[1])
np_ = int(sys.argv[2])

best = None
for px in range(1, np_ + 1):
    if np_ % px:
        continue
    for py in range(1, np_ // px + 1):
        if (np_ // px) % py:
            continue
        pz = np_ // (px * py)
        if nx % px or nx % py or nx % pz:
            continue
        score = abs(px - py) + abs(py - pz) + abs(px - pz)
        if best is None or score < best[0]:
            best = (score, (px, py, pz))

if best is None:
    sys.exit(3)

px, py, pz = best[1]

print(f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2606                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains {np_};

method          simple;

simpleCoeffs
{{
    n           ({px} {py} {pz});
    delta       0.001;
}}

// ************************************************************************* //
""")
