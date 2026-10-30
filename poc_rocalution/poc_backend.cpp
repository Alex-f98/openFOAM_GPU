// Diagnostico: que backend usa rocALUTION y por que.
#include <rocalution/rocalution.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>

using namespace rocalution;

int main(int argc, char** argv)
{
    const bool force = (argc > 1);
    set_device_rocalution(0);
    int r = init_rocalution();
    std::printf("init_rocalution() -> %d\n", r);

    std::printf("--- info_rocalution() ---\n");
    info_rocalution();

    if (force)
    {
        std::printf("--- disable_accelerator_rocalution(false) ---\n");
        disable_accelerator_rocalution(false);
        info_rocalution();
    }

    // prueba de humo: dot de 1e6 doubles
    const long n = 1000000;
    LocalVector<double> x;
    x.Allocate("x", n);
    x.Ones();
    double t0 = 0.0;
    hipDeviceSynchronize();
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) sum += x.Dot(x);
    hipDeviceSynchronize();
    std::printf("dot(1e6) = %.1f\n", sum);

    stop_rocalution();
    return 0;
}
