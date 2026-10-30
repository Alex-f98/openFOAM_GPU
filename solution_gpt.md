Para **v2606**, la rama oficial principal ya absorbió el soporte de GPU de la comunidad, por lo que usar la rama experimental antigua `feature-gpu` de los repositorios de desarrollo podría dar conflictos o no ser exactamente la estructura que trae esa versión liberada.

Si vas a compilar la **v2606 oficial**, el procedimiento estándar recomendado en la documentación oficial no requiere usar la regla `WM_COMPILER=Amd-gpu` de ramas experimentales externas, sino compilar directamente sobre la base trunk/v2606 asegurando que las banderas del compilador de ROCm soporten `stdpar`.

Para hacerlo bien y directo con la **v2606 oficial**, el flujo ajustado es el siguiente:

### 1. Descargar el código fuente oficial de la versión v2606

```bash
wget https://dl.openfoam.com/source/v2606/OpenFOAM-v2606.tgz
wget https://dl.openfoam.com/source/v2606/ThirdParty-v2606.tar.gz
tar -xzf OpenFOAM-v2606.tgz
tar -xzf ThirdParty-v2606.tar.gz
cd OpenFOAM-v2606

```

### 2. Configurar las variables de entorno de ROCm y C++ `stdpar`

En lugar de depender de reglas personalizadas viejas, configuramos el entorno para que el compilador de Clang/ROCm gestione el estándar C++ con offload a la GPU `gfx90a`:

```bash
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH

# Forzar compilador de ROCm y arquitectura de tu MI210
export WM_COMPILER=Clang
export CXX=amdclang++
export CC=amdclang
export AMDGPU_TARGETS=gfx90a

# Cargar entorno base
source etc/bashrc

```

### 3. Compilar usando las opciones optimizadas oficiales (`-j -s -q -l`)

```bash
./Allwmake -j -s -q -l

```

De esta forma utilizas la **versión v2606 oficial y estable** que ya incluye de manera nativa los parches de infraestructura de GPU que la comunidad aportó, evitando los problemas de incompatibilidad de ramas temporales anteriores.
