#!/usr/bin/env python3
"""Genera un caso 'lid-driven cavity 3D' limpio para icoFoam.

Uso:  gen_case.py NX STEPS OUTDIR

- Malla cubica unitaria NX x NX x NX (una sola bloque, grading uniforme).
- 6 caras: movingWall (y=1, U=(1 0 0)) y fixedWalls (las otras 5, noSlip).
  IMPORTANTE: es un caso realmente 3D, por lo que NO se usa el parche 'empty'
  (el intento anterior usaba 'empty' sobre una malla 3D y ademas duplicaba la
  cara z=1, lo que producia una malla invalida).
- deltaT fijo 0.005, endTime = STEPS*deltaT  -> exactamente STEPS pasos.
- fvSolution con GAMG + twoStageGaussSeidel para p (config recomendada por
  la documentacion oficial de la rama GPU).
"""

import os
import sys

HEADER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2606                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       %(cls)s;
    object      %(obj)s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def hdr(cls, obj):
    return HEADER % {'cls': cls, 'obj': obj}


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


BLOCKMESH = hdr('dictionary', 'blockMeshDict') + """
scale   1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 1)
    (1 0 1)
    (1 1 1)
    (0 1 1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (__NX__ __NX__ __NX__) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    movingWall
    {
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }
    fixedWalls
    {
        type wall;
        faces
        (
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);

mergePatchPairs
(
);

// ************************************************************************* //
"""

CONTROLDICT = hdr('dictionary', 'controlDict') + """
application     icoFoam;

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         __ENDTIME__;

deltaT          __DELTAT__;

writeControl    timeStep;
writeInterval   __STEPS__;

purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
"""

FVSCHEMES = hdr('dictionary', 'fvSchemes') + """
ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

// ************************************************************************* //
"""

FVSOLUTION = hdr('dictionary', 'fvSolution') + """
solvers
{
    p
    {
        solver                  GAMG;
        tolerance               1e-06;
        relTol                  0.01;
        smoother                twoStageGaussSeidel;
        cacheAgglomeration      true;
        nCellsInCoarsestLevel   10;
        agglomerator            faceAreaPair;
        mergeLevels             1;
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
}

PISO
{
    nCorrectors             2;
    nNonOrthogonalCorrectors 0;
    pRefCell                0;
    pRefValue               0;
}

// ************************************************************************* //
"""

TRANSPORT = hdr('dictionary', 'transportProperties') + """
nu              [0 2 -1 0 0 0 0] 0.01;

// ************************************************************************* //
"""

U_FIELD = hdr('volVectorField', 'U') + """
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    movingWall
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }

    fixedWalls
    {
        type            noSlip;
    }
}

// ************************************************************************* //
"""

P_FIELD = hdr('volScalarField', 'p') + """
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }

    fixedWalls
    {
        type            zeroGradient;
    }
}

// ************************************************************************* //
"""


def main():
    nx = int(sys.argv[1])
    steps = int(sys.argv[2])
    outdir = sys.argv[3]

    delta_t = 0.005
    end_time = steps * delta_t

    write(os.path.join(outdir, 'system', 'blockMeshDict'),
          BLOCKMESH.replace('__NX__', str(nx)))
    write(os.path.join(outdir, 'system', 'controlDict'),
          CONTROLDICT.replace('__ENDTIME__', repr(end_time))
                      .replace('__DELTAT__', repr(delta_t))
                      .replace('__STEPS__', str(steps)))
    write(os.path.join(outdir, 'system', 'fvSchemes'), FVSCHEMES)
    write(os.path.join(outdir, 'system', 'fvSolution'), FVSOLUTION)
    write(os.path.join(outdir, 'constant', 'transportProperties'), TRANSPORT)
    write(os.path.join(outdir, '0', 'U'), U_FIELD)
    write(os.path.join(outdir, '0', 'p'), P_FIELD)

    write(os.path.join(outdir, 'Allclean'),
          '#!/bin/sh\ncd "${0%/*}" || exit\n. "$WM_PROJECT_DIR/bin/tools/CleanFunctions"\n'
          'cleanCase\nrm -rf log.* postProcessing\n#------------------------------------------------------------------------------\n')

    print('case generated: %s  (nx=%d, steps=%d, endTime=%g)'
          % (outdir, nx, steps, end_time))


if __name__ == '__main__':
    main()
