#!/usr/bin/env python3
import os, sys, shutil, subprocess, re

base_tutorial = "/openfoamGPU/OpenFOAM-v2606/tutorials/incompressible/icoFoam/cavity/cavity"
exp_root = "/openfoamGPU/exp_scaling_gpu"

def make_case(nx):
    case_dir = os.path.join(exp_root, f"cavity3d_{nx}")
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    # copy
    shutil.copytree(base_tutorial, case_dir)
    # modify blockMeshDict for 3D cubic
    bmd = os.path.join(case_dir, "system/blockMeshDict")
    with open(bmd) as f:
        s = f.read()
    # set scale
    s = s.replace("scale   0.1;", "scale   0.1;")
    # vertices 3D cube 0,1,0,1,0,h
    verts = """vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 1)
    (1 0 1)
    (1 1 1)
    (0 1 1)
);"""
    s = re.sub(r'vertices\s*\(.*?\);', verts, s, flags=re.S)
    # blocks: (nx nx nx)
    blocks = f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({nx} {nx} {nx}) simpleGrading (1 1 1)\n);"
    s = re.sub(r'blocks\s*\(.*?\);', blocks, s, flags=re.S)
    # boundary - make frontAndBack walls in z? or keep as walls; better make all walls except moving top? original movingWall is face (3 7 6 2) which in old coords; easier to adapt to standard 3D lid driven cavity
    # standard: moving wall at y=1 (top) in some coords? let us look up: original vertices y goes 0->1, movingWall face (3 7 6 2). Vertices in original 2D-like: (3) is (0,1,0), (7) is (0,1,0.1), (6) is (1,1,0.1), (2) is (1,1,0). Yes moving wall at top y=1. frontAndBack are empty (z faces). For 3D full cavity, frontAndBack should be walls too or let us set as walls? Lid driven cavity 3D usually all walls except maybe periodic; but standard 3D cubic cavity has moving lid and others walls. Let us make frontAndBack into walls.
    # replace frontAndBack block
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
    s = re.sub(r'boundary\s*\(.*?\);\s*$', bnd, s, flags=re.S)
    with open(bmd, 'w') as f:
        f.write(s)
    return case_dir

if __name__ == '__main__':
    nx = int(sys.argv[1])
    print(make_case(nx))
