# Reparto ensamblado vs solve en CPU (128^3, 64 rangos, 20 pasos)

Método: tres corridas idénticas salvo el solver. La diferencia entre ellas aísla
el coste de cada solve sin necesidad de instrumentar OpenFOAM.

| variante | p | U | exec (20 pasos) | s/paso |
|---|---|---|---|---|
| `gamg` | GAMG + symGS | smoothSolver + symGS | 5,49 s | 0,2745 |
| `p_1iter` | 1 iteración | smoothSolver + symGS | 3,37 s | 0,1685 |
| `both_1iter` | 1 iteración | 1 iteración | 1,97 s | 0,0985 |

Derivado:

| fase | s/paso | % del paso |
|---|---|---|
| ensamblado + PISO + I/O | 0,0985 | 36 % |
| solve `p` (p + pFinal) | 0,106 | 39 % |
| solve `U` (Ux+Uy+Uz) | 0,070 | 25 % |
| **solve total** | **0,176** | **64 %** |

Comparación con el camino GPU (mismo caso, 1 núcleo + MI210):

| fase | CPU 64 núcleos | GPU (1 núcleo + MI210) |
|---|---|---|
| ensamblado | 0,0985 s | 2,90 s |
| relleno CSR | — | 0,40 s |
| setup AMG (`ReBuildNumeric`) | incluido en el solve | 1,05 s |
| solve | 0,176 s | 0,15 s |
| transferencias | — | 0,05 s |
| **paso completo** | **0,2745 s** | **4,50 s** |

Conclusión: el solve en GPU (0,15 s) está a la par del solve en 64 núcleos
(0,176 s); la diferencia de 16x en el paso completo viene del ensamblado en un
solo núcleo (2,9 s) y del trabajo de host del plugin (1,45 s).

Reproducción: `/tmp/opencode/prof78` (base), `t_ref`, `t_diagp`, `t_diagall`
(copias con `system/fvSolution` modificado; los nombres conservan el histórico
"diag" aunque el solver barato es `smoothSolver` con `maxIter 1`).

Nota: 78 rangos no es descomponible en un cubo 128^3 con `simple` (78 = 2·3·13
no divide 128) y scotch/metis están compilados como stub dummy en esta
instalación, así que se midió con 64 rangos.
