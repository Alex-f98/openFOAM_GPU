#!/usr/bin/env python3
import sys, re
bmd = sys.argv[1]
with open(bmd) as f:
    s = f.read()
bnd = """boundary
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
);"""
# replace from boundary to end
s2 = re.sub(r'boundary\s*\(.*', bnd + '\n\n\n// ************************************************************************* //\n', s, flags=re.S)
with open(bmd,'w') as f:
    f.write(s2)
