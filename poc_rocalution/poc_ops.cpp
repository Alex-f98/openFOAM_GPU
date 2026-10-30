// Aisla el coste por operacion de rocALUTION en esta maquina.
// Compila: hipcc -O3 -std=c++17 -I/opt/rocm/include poc_ops.cpp -o poc_ops \
//          -L/opt/rocm/lib -lrocalution -Wl,-rpath,/opt/rocm/lib
#include <rocalution/rocalution.hpp>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace rocalution;

static double now_ms()
{
    using clk = std::chrono::steady_clock;
    static const auto t0 = clk::now();
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? std::atoi(argv[1]) : 32;
    const long n = 1L * N * N * N;

    int* rp = new int[n + 1];
    int* col = new int[7 * n];
    double* val = new double[7 * n];
    long nnz = 0;
    auto id = [N](int i, int j, int k) { return (i * N + j) * N + k; };

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
            {
                rp[id(i, j, k)] = static_cast<int>(nnz);
                col[nnz] = id(i, j, k); val[nnz++] = 6.0;
                int nb[6][3] = {{i-1,j,k},{i+1,j,k},{i,j-1,k},{i,j+1,k},{i,j,k-1},{i,j,k+1}};
                for (auto& q : nb)
                    if (q[0] >= 0 && q[0] < N && q[1] >= 0 && q[1] < N && q[2] >= 0 && q[2] < N)
                    { col[nnz] = id(q[0], q[1], q[2]); val[nnz++] = -1.0; }
            }
    rp[n] = static_cast<int>(nnz);

    init_rocalution(0);

    LocalMatrix<double> A;
    A.SetDataPtrCSR(&rp, &col, &val, "A", nnz, n, n);

    LocalVector<double> x, y, b;
    x.Allocate("x", n); x.Ones();
    y.Allocate("y", n); y.Zeros();
    b.Allocate("b", n); b.Ones();

    // rocALUTION crea los objetos en HOST: hay que moverlos al acelerador
    A.MoveToAccelerator();
    x.MoveToAccelerator();
    y.MoveToAccelerator();
    b.MoveToAccelerator();

    const int R = 20;

    // --- SpMV ---
    A.Apply(x, &y);
    hipDeviceSynchronize();
    double t = now_ms();
    for (int i = 0; i < R; ++i) A.Apply(x, &y);
    hipDeviceSynchronize();
    double spmv = (now_ms() - t) / R;
    std::printf("N=%d n=%ld nnz=%ld  SpMV: %.3f ms/op\n", N, n, nnz, spmv);

    // --- dot ---
    t = now_ms();
    double s = 0;
    for (int i = 0; i < R; ++i) s += x.Dot(x);
    hipDeviceSynchronize();
    std::printf("  dot:  %.3f ms/op  (suma=%.1f)\n", (now_ms() - t) / R, s);

    // --- axpy (ScaleAdd) ---
    t = now_ms();
    for (int i = 0; i < R; ++i) y.ScaleAdd(1e-9, x);
    hipDeviceSynchronize();
    std::printf("  axpy: %.3f ms/op\n", (now_ms() - t) / R);

    // --- malloc/free churn ---
    t = now_ms();
    for (int i = 0; i < R; ++i) { void* p = nullptr; hipMalloc(&p, n * sizeof(double)); hipFree(p); }
    std::printf("  hipMalloc+Free (%.1f MB): %.3f ms/op\n", n * 8.0 / 1e6, (now_ms() - t) / R);

    // --- CG con numero de iteraciones fijo ---
    CG<LocalMatrix<double>, LocalVector<double>, double> cg;
    cg.SetOperator(A);
    cg.InitTol(0.0, 1e-16, 1e8);
    cg.SetResidualNorm(2);
    cg.Build();

    for (int it : {1, 2, 4, 8, 16, 32, 64})
    {
        if (it > 64) continue;
        cg.InitMaxIter(it);
        y.Zeros();
        double t0 = now_ms();
        cg.Solve(b, &y);
        double dt = now_ms() - t0;
        std::printf("  CG maxiter=%2d -> %.2f ms  (%.2f ms/iter, hecho=%d)\n",
                    it, dt, dt / it, cg.GetIterationCount());
    }

    cg.Clear();
    stop_rocalution();
    return 0;
}
