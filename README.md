# OpenFOAM v2606 en AMD MI210 — qué se consiguió y por qué no acelera

Este repositorio contiene **todo** el trabajo de aceleración GPU de `icoFoam` (cavity 3D)
en una AMD Instinct MI210: build, plugins, experimentos, bancos de medida y resultados.

> **Empieza por [`TRASPASO.md`](TRASPASO.md)** si vas a continuar el trabajo. Está pensado
> para leerlo en 15 minutos y poder compilar y ejecutar sin ayuda.

**Repositorio:** <https://github.com/Alex-f98/openFOAM_GPU> · el clon pesa 6,6 MB: los builds,
las mallas, los campos y la matriz de 191 MB **no** se versionan (se regeneran, ver
[`TRASPASO.md`](TRASPASO.md) §4 y §5; el detalle está en [`.gitignore`](.gitignore)).

---

## El resultado, en una tabla

| Medida (128³ = 2,1 M celdas) | CPU 32 ranuras | GPU MI210 |
|---|---|---|
| Solve de presión (`p` + `pFinal`) | 138,5 ms | **80,5 ms** → **1,72×** |
| Paso de tiempo completo | 0,352 s | 0,294 s → **1,20×** *(sin contar mover datos)* |
| Paso completo con 64 núcleos de CPU | **0,274 s** | **pierde** (0,93×) |

**Conclusión:** la GPU hace su parte 5× más rápido por iteración, pero necesita 2,9× más
iteraciones y solo cubre el 39 % del paso. A esta escala **no compensa**: 64 núcleos de CPU
sin tocar código ya superan a la GPU. El ×2 que se publica necesita memoria unificada
(MI300A) y ~64 M celdas, que aquí no caben.

Lo que sí queda demostrado y documentado: **por qué** no acelera, con medidas y no con
opiniones (FP64 no es el límite, el ancho de banda sí; PCIe 4.0 ×16 medido; escalado con
el tamaño medido).

---

## Por dónde ir

| Necesito… | Archivo |
|---|---|
| Continuar el trabajo (build, ejecución, trampas) | **[`TRASPASO.md`](TRASPASO.md)** |
| El informe técnico completo | [`INFORME_GPU_V2606.md`](INFORME_GPU_V2606.md) |
| La comparación CPU vs GPU definitiva | [`COMPARACION_PRESION_GPU_CPU.md`](COMPARACION_PRESION_GPU_CPU.md) |
| Cómo funciona el solver GPU que sí funciona | [`KRYLOV_RESIDENTE_GPU.md`](KRYLOV_RESIDENTE_GPU.md) |
| Reproducir los números | [`repro/README.md`](repro/README.md) |
| Compilar OpenFOAM con la rama GPU | [`README_BUILD.md`](README_BUILD.md) |
| "¿Y la memoria unificada?" (la pregunta que hará todo el mundo) | [`MEMORIA_UNIFICADA_MI210.md`](MEMORIA_UNIFICADA_MI210.md) |
| Los intentos fallidos (para no repetirlos) | [`OPCION_B_MPI_GPU.md`](OPCION_B_MPI_GPU.md), [`POC_ROCALUTION_V2606.md`](POC_ROCALUTION_V2606.md) |

---

## Prueba rápida (2 minutos)

```bash
# 1. ¿la GPU responde?
cd /openfoamGPU/gpu_krylov
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
./bench_pcie 16777216            # ~28 GB/s por sentido en PCIe 4.0 x16

# 2. ¿el solver residente da lo esperado?  (80,5 ms de bloque de presión)
#    La matriz de 191 MB no está en el repo: se regenera en ~1 min con el plugin
cd /openfoamGPU/repro && ./dump128.sh /tmp/dump128
cd /tmp/dump128 && CF=2 SMOOTH=gs NU=2 WARM=1.3e-3,1e-6 \
  /openfoamGPU/gpu_krylov/bench_mg roc_matrix_p.bin 0.13 500 128

# 3. ¿la CPU da lo esperado?  (0,33-0,36 s/paso a 32 rangos, 128^3)
cd /openfoamGPU/repro/diff_cpu && ./run_diff.sh 32 && python3 split.py
```

Si los tres dan los números de `TRASPASO.md` §6, el entorno está sano.

---

## Reglas de la casa

1. **Solo la GPU 0** (`ROCR_VISIBLE_DEVICES=0`); la 1 es de otro inquilino.
2. **No medir CPU y GPU a la vez** (se estorban).
3. `FOAM_SIGFPE=false` y `set +u` antes de sourcear OpenFOAM.
4. Un número sin su `log` y su comando **no existe**. Todo resultado va a un `.csv` y un
   `.md` con el método.
5. Las mallas pequeñas (32³) no sirven para cronometrar: 0,01 s de resolución.
