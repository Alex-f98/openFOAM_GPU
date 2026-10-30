// Mide latencia/sync de operaciones HIP basicas en esta maquina.
// Compila: hipcc -O3 -std=c++17 hip_latency.cpp -o hip_latency
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>

__global__ void nop(float* p) { if (p) p[0] = 1.0f; }

static double ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main()
{
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::printf("device: %s  clockRate=%.0f MHz  memBusWidth=%d\n",
                prop.name, prop.clockRate / 1000.0, prop.memoryBusWidth);

    const int reps = 20;
    for (long bytes : {4096L, 1L << 20, 16L << 20, 64L << 20})
    {
        std::vector<float> h(bytes / 4, 1.0f);
        float* d = nullptr;
        hipMalloc(&d, bytes);
        hipDeviceSynchronize();

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice);
        hipDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();

        for (int i = 0; i < reps; ++i) hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
        auto t2 = std::chrono::steady_clock::now();

        for (int i = 0; i < reps; ++i) { nop<<<1, 1>>>(d); }
        hipDeviceSynchronize();
        auto t3 = std::chrono::steady_clock::now();

        double bw_in = (bytes / 1e6) / (ms(t0, t1) / reps / 1000.0);
        std::printf("%7.1f MB: H2D %.3f ms (%.0f MB/s)  D2H %.3f ms  launch+sync %.4f ms\n",
                    bytes / 1e6, ms(t0, t1) / reps, bw_in, ms(t1, t2) / reps, ms(t2, t3) / reps);
        hipFree(d);
    }
    return 0;
}
