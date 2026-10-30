// Medida del enlace host<->GPU que usa OpenFOAM cuando mueve vectores o matrices.
// Uso: ./bench_pcie [bytes]
//   - pageable: hipMalloc normal (el driver copia via buffer interno)
//   - pinned:   hipHostMalloc (DMA directo)
// Reporta GB/s efectivos y microsegundos por transferencia.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

static double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    const size_t n = (argc > 1 ? strtoull(argv[1], nullptr, 10) : (size_t)16*1024*1024);
    const int reps = 20;

    hipDeviceProp_t pr;
    CHK(hipGetDeviceProperties(&pr, 0));
    printf("GPU: %s  (memoria %.1f GB)\n", pr.name, pr.totalGlobalMem/1e9);
    printf("buffer: %.2f MB   reps=%d\n", n/1e6, reps);

    void* d = nullptr;
    CHK(hipMalloc(&d, n));

    // --- pageable -----------------------------------------------------------
    std::vector<char> h(n);
    for (size_t i = 0; i < n; i += 4096) h[i] = char(i);

    // calentamiento (fija el contexto y las paginas)
    CHK(hipMemcpy(d, h.data(), n, hipMemcpyHostToDevice));
    CHK(hipDeviceSynchronize());

    double t0 = now_ms();
    for (int r = 0; r < reps; ++r) CHK(hipMemcpy(d, h.data(), n, hipMemcpyHostToDevice));
    CHK(hipDeviceSynchronize());
    double tH2D_p = (now_ms() - t0) / reps;

    t0 = now_ms();
    for (int r = 0; r < reps; ++r) CHK(hipMemcpy(h.data(), d, n, hipMemcpyDeviceToHost));
    CHK(hipDeviceSynchronize());
    double tD2H_p = (now_ms() - t0) / reps;

    printf("pageable  H2D  %8.3f ms  %7.2f GB/s\n", tH2D_p, n/1e9/(tH2D_p/1e3));
    printf("pageable  D2H  %8.3f ms  %7.2f GB/s\n", tD2H_p, n/1e9/(tD2H_p/1e3));

    // --- pinned -------------------------------------------------------------
    char* hp = nullptr;
    CHK(hipHostMalloc((void**)&hp, n));
    for (size_t i = 0; i < n; i += 4096) hp[i] = char(i);
    CHK(hipMemcpy(d, hp, n, hipMemcpyHostToDevice));
    CHK(hipDeviceSynchronize());

    t0 = now_ms();
    for (int r = 0; r < reps; ++r) CHK(hipMemcpy(d, hp, n, hipMemcpyHostToDevice));
    CHK(hipDeviceSynchronize());
    double tH2D_pi = (now_ms() - t0) / reps;

    t0 = now_ms();
    for (int r = 0; r < reps; ++r) CHK(hipMemcpy(hp, d, n, hipMemcpyDeviceToHost));
    CHK(hipDeviceSynchronize());
    double tD2H_pi = (now_ms() - t0) / reps;

    printf("pinned    H2D  %8.3f ms  %7.2f GB/s\n", tH2D_pi, n/1e9/(tH2D_pi/1e3));
    printf("pinned    D2H  %8.3f ms  %7.2f GB/s\n", tD2H_pi, n/1e9/(tD2H_pi/1e3));

    // --- latencia (transferencia minima) -----------------------------------
    double one = 0.0;
    const int lreps = 200;
    CHK(hipMemcpy(d, &one, 8, hipMemcpyHostToDevice));
    CHK(hipDeviceSynchronize());
    t0 = now_ms();
    for (int r = 0; r < lreps; ++r) CHK(hipMemcpy(d, &one, 8, hipMemcpyHostToDevice));
    CHK(hipDeviceSynchronize());
    printf("latencia  H2D  8 B: %.1f us por llamada\n", (now_ms() - t0)*1000/lreps);

    // --- coste de un round trip de un vector de 128^3 (16.8 MB) ------------
    const double vec = 2097152.0*8.0;
    printf("\nun vector de p a 128^3 = %.1f MB\n", vec/1e6);
    double bwh = n/1e9/(tH2D_pi/1e3), bwd = n/1e9/(tD2H_pi/1e3);
    printf("  con pinned   : ida %.2f ms + vuelta %.2f ms = %.2f ms\n", vec/1e9/bwh*1e3, vec/1e9/bwd*1e3, vec/1e9/bwh*1e3 + vec/1e9/bwd*1e3);
    printf("  con pageable : ida %.2f ms + vuelta %.2f ms = %.2f ms\n", vec/1e9/(n/1e9/(tH2D_p/1e3))*1e3, vec/1e9/(n/1e9/(tD2H_p/1e3))*1e3,
           vec/1e9/(n/1e9/(tH2D_p/1e3))*1e3 + vec/1e9/(n/1e9/(tD2H_p/1e3))*1e3);

    CHK(hipHostFree(hp));
    CHK(hipFree(d));
    return 0;
}
