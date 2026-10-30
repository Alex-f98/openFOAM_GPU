# Opción B — MPI + GPU: preconditionador AMG en GPU dentro del Krylov de OpenFOAM

**Fecha:** 2026-10-27
**Nodo:** 1× AMD MI210 (gfx90a, 64 GB), GPU 0 únicamente (`ROCR_VISIBLE_DEVICES=0`)
**OpenFOAM:** v2606 oficial (CPU) + plugin `librocALUTIONLduSolver`
**Objetivo:** comprobar si descargar *solo el álgebra* a la GPU, manteniendo el reparto MPI y
el Krylov global de OpenFOAM, produce un `cavity3D` más rápido que `GAMG` en CPU.

---

## 1. Por qué hacía falta un cambio de arquitectura

El plugin original (`solver rocALUTION;`) resuelve **el bloque local** de cada rank. En serie eso es
la solución exacta del sistema; en paralelo es un **block-Jacobi**, es decir, un método que ignora
el acoplamiento entre subdominios y por tanto **no converge al residuals global** que OpenFOAM
calcula. Usarlo tal cual en MPI daría una respuesta incorrecta (o un `maxIter` agotado).

La arquitectura correcta es la inversa: **OpenFOAM conserva el Krylov global** (reducciones,
intercambio de interfaces, criterio de convergencia) y la GPU aporta **solo el preconditionador**
`M⁻¹ ≈ A_local⁻¹` (un ciclo V de AMG por llamada). Eso es *additive Schwarz*: matemáticamente
válido, con la convergencia degradándose al crecer el número de subdominios.

Implementación: `rocALUTIONLduPreconditioner` (nuevo), registrado en las tablas
`lduMatrix::preconditioner::addsym/asymMatrixConstructorToTable`. Se usa así:

```
p { solver PCG;       preconditioner rocALUTION; tolerance 1e-6; relTol 0.01; }
U { solver PBiCGStab; preconditioner rocALUTION; tolerance 1e-6; relTol 0;    }
```

Coste de código: reutiliza el 90 % del solver (patrón CSR, caché por `mesh`+campo, signo, setup
AMG). La lógica compartida vive en `rocALUTIONCommon.H`; el solver y el preconditionador la incluyen.

---

## 2. Corrección: funciona (serie y paralelo)

| Prueba | Resultado |
|---|---|
| 32³ serie, `PCG + rocALUTION` | converge, 4–6 iteraciones por solve de `p`, `Final residual` ≤ tolerancia |
| 128³, 8 ranks | corre completo (20 pasos), `p` converge en 33–111 iteraciones |
| 128³, 16 ranks | corre (19 pasos completos), `p` converge en 42–149 iteraciones |
| Residuales | `p` 5e-5…8e-7, `U` 2e-7 → mismos criterios que `GAMG` |

No hay cambios en el caso más allá de `fvSolution` y la línea `libs` ya existente.

---

## 3. Rendimiento medido (128³ = 2.097.152 celdas, 20 pasos, GPU 0)

Tiempo **de pared** por paso (20 pasos, `ClockTime`/pasos):

| Ranks | CPU `GAMG` | GPU precond. | GPU / CPU | Iteraciones `p` (CPU → GPU) | Setup AMG por matriz |
|---|---|---|---|---|---|
| 8 | 0,96 s | **9,7 s** (194 s totales) | **10× más lenta** | 11 → 111 | 0,53–0,97 s/rank |
| 16 | 0,54 s | **~42 s** (792 s en 19 pasos) | **77× más lenta** | 11 → 149 | 1,3–3,6 s/rank |
| 32 | 0,33 s | no medido (la tendencia es monótona) | — | — | — |
| 64 | 0,26 s | — | — | — | — |

> Nota metodológica: `ExecutionTime` de OpenFOAM (tiempo de CPU del rank 0) **no** sirve aquí —
> a 8 ranks marca 99 s frente a 194 s de pared, porque el proceso pasa tiempo bloqueado esperando
> a la GPU. Todas las cifras de GPU son de pared (`ClockTime`), que es lo que ve el usuario.

### 3.1 Lectura

1. **El número de iteraciones explota con los ranks** (11 → 111 con 8 ranks, → 149 con 16).
   Es el comportamiento esperado de *additive Schwarz* sin corrección gruesa: la condición del
   sistema precondicionado crece con el número de subdominios. Un `M⁻¹` local no puede propagar
   información entre subdominios; eso lo hace el Krylov global, y por eso necesita muchas más
   iteraciones.
2. **El setup de AMG se paga por rank y por campo.** A 8 ranks cada jerarquía local cuesta
   0,53–0,97 s (1,3–3,6 s con 16 ranks) y hay 4–5 construcciones por paso: ≈4 s/paso solo en
   preparación, sobre 9,7 s/paso totales. La GPU acelera el *apply*, no el *setup* (mayormente CPU:
   coerción en host). La caché funciona también en MPI: 4 builds/paso y no 5, porque `pFinal`
   reutiliza la jerarquía de `p` (mismo `mesh`, mismo `coefSum`).
3. **El coste crece de forma superlineal:** 2× ranks ⇒ 4,3× tiempo/paso. Cada rank añade
   (a) su propio contexto HIP sobre la misma GPU, (b) su setup de AMG, y (c) más iteraciones
   globales. El trabajo total en la GPU crece como `nProcs × iteraciones(nProcs)`.
4. **Contraste con el solver en serie** (§12: 1,9× más rápido que 1 núcleo): aquí se juntan los
   tres efectos anteriores, de ahí el 10×–77× en contra.

---

## 4. Comparación con la referencia publicada

El benchmark de referencia (MI300A) solo gana a partir de **~64 M celdas** (×2 frente a 32 núcleos;
empate a ~8 M). En MI210 el régimen ganador es inalcanzable: 64 M celdas × 2,6 KB/celda ≈ 166 GB
frente a 64 GB de VRAM, y el ensamblado seguiría en CPU (Amdahl).

La opción B **no cambia el orden de magnitud**: sigue siendo un método de Krylov precondicionado
localmente, con el mismo techo de ensamblado (§7) y, además, con degradación de convergencia por
subdominios. No hay ningún punto de la malla donde pueda pasar de 1× a 32 núcleos.

---

## 5. ¿Y la variante "reunir la matriz y resolver en un solo contexto"?

La única arquitectura que evita los N contextos HIP es ensamblar el sistema **global** en un rank
y resolverlo con un único AMG en la GPU (gather + solve global). No hace falta implementarla para
descartarla: sus componentes ya están medidos a 128³ con **un solo contexto** (`log.prod128`).

Desglose medido del paso completo en la GPU con 2,1 M celdas (1 proceso, §12):

| Fase | Coste | ¿Se paraleliza en la variante gather? |
|---|---|---|
| Ensamblado (OpenFOAM) | 2,90 s | sí: 0,20 s con 32 ranks |
| Relleno CSR en host | 0,40 s | sí (repartido) |
| **Setup AMG** | **1,05 s** | **no** (un solo contexto) |
| Transferencias | 0,05 s | no |
| Solve en GPU | 0,15 s | no |

Suelo teórico del paso en la variante gather a 32 ranks:

```
0,20 (ensamblado) + 0,40 (CSR) + 1,05 (setup) + 0,05 + 0,15  = 1,85 s/paso
+ coste del gather de la matriz y el término independiente   ≥ 0,3 s
                                                             ≈ 2,1 s/paso
```

frente a **0,33 s/paso** de `GAMG` con 32 núcleos ⇒ **≥6× más lento**, incluso sin contar el
gather ni la falta de solapamiento. Aunque se cacheara la jerarquía y solo se hiciera
`ReBuildNumeric` (0,21 s), el suelo bajaría a ~1,2 s/paso: **3,7× más lento**. Y cachear más que
eso es incorrecto, porque los coeficientes cambian en cada paso.

**No hay ninguna configuración de la opción B que gane a CPU MPI en esta máquina.** El motivo
último es el mismo de §7: el **setup de AMG** (1,05 s por matriz, 4–5 por paso) es una fase
intrínsecamente serie-secuencial que la GPU no acelera, y ya supera por sí sola el presupuesto
completo del paso en CPU (0,33 s).

---

## 6. Reproducir

```bash
set +u; source /openfoamGPU/cpu/OpenFOAM-v2606/etc/bashrc; set -u
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 FOAM_SIGFPE=false OMP_NUM_THREADS=4

cd /openfoamGPU/rocALUTION-solver && wmake libso      # construye solver + preconditionador

cd <caso>                                             # 128³, system/fvSolution con rocALUTION
cp system/decomposeParDict.32 system/decomposeParDict
decomposePar -force
mpirun --allow-run-as-root -np 32 icoFoam -parallel
```

Ficheros:
- `rocALUTION-solver/rocALUTIONLduPreconditioner.{H,C}` — preconditionador
- `rocALUTION-solver/rocALUTIONCommon.H` — CSR, caché y setup compartidos con el solver
- `rocALUTION-solver/case_*/` — casos y logs
- `rocALUTION-solver/results_mpi_precond.csv` — tabla de medidas de esta ruta
- `rocALUTION-solver/logs_opcion_b/` — logs completos (8 y 16 ranks, comprimidos)

---

## 7. Qué sigue: ruta C (Krylov residente)

El límite de esta ruta está en la orquestación (un *round trip* por iteración), no en la GPU.
Como continuación se midió el mismo solve de `p` con el bucle **entero en el dispositivo** y un
multigrid de jerarquía fija: **80,5 ms para el bloque `p`+`pFinal` a 128³ frente a 138,5 ms de
`GAMG` con 32 ranks** (1,72× más rápido; 43,6× frente a un núcleo). El paso completo, en cambio,
solo mejora **1,20×** porque el ensamblado sigue en CPU. Detalle, método y corrección de una
comparación previa mal planteada en **`COMPARACION_PRESION_GPU_CPU.md`**, en
**`KRYLOV_RESIDENTE_GPU.md`** y en §14 de `INFORME_GPU_V2606.md`.
