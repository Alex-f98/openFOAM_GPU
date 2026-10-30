// bench_managed.cpp — memoria gestionada (hipMallocManaged) en la MI210.
//
// Mide dos cosas distintas, que es justo donde se confunde la gente:
//
//   1) PASO 1 (migración): el primer recorrido paga el page-fault y la copia
//      RAM -> VRAM de todo el buffer. Baja.
//   2) ESTADO ESTACIONARIO: una vez residente, va a velocidad de HBM. Alto.
//
// Si el buffer NO cabe en la VRAM (p. ej. 80 GB en una tarjeta de 64 GB), el
// estado estacionario desaparece: cada pasada vuelve a paginar por PCIe y el
// ancho de banda se hunde al del enlace. Ese es el número que importa para
// decidir si la memoria unificada sirve para una malla de 64 M celdas.
//
// Compilar:
//   hipcc -O3 -std=c++17 bench_managed.cpp -o bench_managed \
//         -I/opt/rocm/include -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib
//
// Uso:
//   ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 ./bench_managed <GB> [pasadas]
//
//   ./bench_managed 8      # cabe en VRAM  -> mide el mejor caso
//   ./bench_managed 80     # NO cabe       -> mide el caso que importa (lento)
//
// Aviso: con GB > VRAM el programa puede tardar horas. Empieza por 8 y sube.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_CHECK(cmd)                                                         \
    do                                                                         \
    {                                                                          \
        hipError_t e = (cmd);                                                  \
        if (e != hipSuccess)                                                   \
        {                                                                      \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",                    \
                hipGetErrorString(e), __FILE__, __LINE__);                     \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__global__ void initKernel(double* x, double* y, std::size_t n)
{
    std::size_t i = blockIdx.x*(std::size_t)blockDim.x + threadIdx.x;
    std::size_t s = (std::size_t)gridDim.x*blockDim.x;
    for (; i < n; i += s)
    {
        x[i] = 1.0;
        y[i] = 2.0;
    }
}

__global__ void axpyKernel(double* x, double* y, std::size_t n)
{
    std::size_t i = blockIdx.x*(std::size_t)blockDim.x + threadIdx.x;
    std::size_t s = (std::size_t)gridDim.x*blockDim.x;
    for (; i < n; i += s)
    {
        y[i] = 0.5*x[i] + y[i];
    }
}

static double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    const double gb = (argc > 1) ? std::atof(argv[1]) : 8.0;
    const int passes = (argc > 2) ? std::atoi(argv[2]) : 6;
    const std::size_t n = (std::size_t)(gb*(1ULL << 30)/sizeof(double));
    const std::size_t bytes = n*sizeof(double);

    std::size_t freeB = 0, totalB = 0;
    HIP_CHECK(hipMemGetInfo(&freeB, &totalB));
    std::printf
    (
        "GPU: %.1f GB de VRAM (%.1f GB libres) | buffer gestionado: %.1f GB x2\n",
        totalB/1073741824.0, freeB/1073741824.0, gb
    );
    if (2*gb > totalB/1073741824.0)
    {
        std::printf
        (
            "AVISO: 2 x %.1f GB no caben en la VRAM. Esto mide el caso con "
            "pagining por PCIe.\n", gb
        );
    }

    const bool useDevice = (getenv("MALLOC") && std::string(getenv("MALLOC")) == "device");
    const bool prefetch = (getenv("PREFETCH") != nullptr);
    const bool hostTouch = (getenv("HOSTTOUCH") != nullptr);

    double *x = nullptr, *y = nullptr;
    const double t_alloc0 = now_ms();
    if (useDevice)
    {
        HIP_CHECK(hipMalloc(&x, bytes));
        HIP_CHECK(hipMalloc(&y, bytes));
    }
    else
    {
        HIP_CHECK(hipMallocManaged(&x, bytes));
        HIP_CHECK(hipMallocManaged(&y, bytes));
        if (hostTouch)
        {
            // flujo realista: la CPU rellena los datos y despues se pasan a la GPU
            std::memset(x, 0, bytes);
            std::memset(y, 0, bytes);
        }
        if (prefetch)
        {
            HIP_CHECK(hipMemPrefetchAsync(x, bytes, 0));
            HIP_CHECK(hipMemPrefetchAsync(y, bytes, 0));
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
    const double t_alloc = now_ms() - t_alloc0;
    std::printf
    (
        "alloc: %s%s\n",
        useDevice ? "hipMalloc (VRAM, referencia)" : "hipMallocManaged",
        (!useDevice && prefetch) ? " + hipMemPrefetchAsync a GPU0" : ""
    );

    const int block = 256;
    const int grid = 4096;

    // --- 1) primer recorrido: aqui se paga la migracion de paginas ---
    const double t0 = now_ms();
    initKernel<<<grid, block>>>(x, y, n);
    HIP_CHECK(hipDeviceSynchronize());
    const double t_first = now_ms() - t0;

    // --- 2) pasadas en caliente: si cabe, esto es HBM ---
    std::vector<double> t(passes);
    for (int p = 0; p < passes; ++p)
    {
        const double a = now_ms();
        axpyKernel<<<grid, block>>>(x, y, n);
        HIP_CHECK(hipDeviceSynchronize());
        t[p] = now_ms() - a;
    }
    std::vector<double> sorted(t.begin(), t.end());
    std::sort(sorted.begin(), sorted.end());
    const double med = sorted[sorted.size()/2];

    // axpy lee 2 vectores y escribe 1 -> 3 x 8 bytes por elemento
    const double traffic = 3.0*bytes/1073741824.0;   // GB movidos por pasada
    std::printf("alloc                : %8.1f ms\n", t_alloc);
    std::printf("pasada 1 (migracion) : %8.1f ms  -> %6.1f GB/s\n",
        t_first, traffic/(t_first/1000.0));
    std::printf("pasadas en caliente  : %8.1f ms  -> %6.1f GB/s  (mediana de %d)\n",
        med, traffic/(med/1000.0), passes);
    std::printf("rango en caliente    : %.1f - %.1f ms\n", sorted.front(), sorted.back());

    if (med > 0)
    {
        std::printf
        (
            "coste de paginar el buffer entero cada pasada (estimacion): "
            "%.1f ms a 28 GB/s\n", (2*bytes/1073741824.0)/28.0*1000.0
        );
    }

    HIP_CHECK(hipFree(x));
    HIP_CHECK(hipFree(y));
    return 0;
}
