#!/bin/bash
# make_cavity3d.sh <N>  -> prints case dir
# Builds a cubic NxNxN all-wall cavity case for icoFoam:
#   Re=100 (U=1, nu=0.01, L=1 -> scale 1.0)
#   exactly 20 timesteps: deltaT=0.005, endTime=0.1
set -e
N=$1
EXPDIR=/openfoamGPU/exp_scaling_gpu
CASEDIR=$EXPDIR/cavity3d_${N}x${N}x${N}
TUTORIAL=/openfoamGPU/OpenFOAM-v2606/tutorials/incompressible/icoFoam/cavity/cavity

rm -rf $CASEDIR
cp -r $TUTORIAL $CASEDIR

python3 - "$CASEDIR" "$N" <<'PYEOF'
import sys, re, os
case, N = sys.argv[1], int(sys.argv[2])

HDR = '''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2606                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       %(cls)s;
    object      %(obj)s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

'''

def write(rel, cls, obj, body):
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(HDR % {'cls': cls, 'obj': obj} + body +
                '\n\n// ************************************************************************* //\n')

# ---------------- blockMeshDict ----------------
bm = f'''scale   1.0;

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
    hex (0 1 2 3 4 5 6 7) ({N} {N} {N}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    movingWall
    {{
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }}
    fixedWalls
    {{
        type wall;
        faces
        (
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
'''
write('system/blockMeshDict', 'dictionary', 'blockMeshDict', bm)

# ---------------- controlDict: exactly 20 steps ----------------
cd = '''application     icoFoam;

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         0.1;

deltaT          0.005;

writeControl    timeStep;
writeInterval   20;
purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;
'''
write('system/controlDict', 'dictionary', 'controlDict', cd)

# ---------------- 0/U and 0/p without frontAndBack ----------------
U = '''dimensions      [0 1 -1 0 0 0 0];

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
'''
write('0/U', 'volVectorField', 'U', U)

p = '''dimensions      [0 2 -2 0 0 0 0];

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
'''
write('0/p', 'volScalarField', 'p', p)
PYEOF

# transportProperties: nu = 0.01 (already tutorial value, ensure it)
sed -i 's/^nu .*/nu              0.01;/' $CASEDIR/constant/transportProperties

echo $CASEDIR
