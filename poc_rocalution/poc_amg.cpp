// PoC: AMG de rocALUTION sobre gfx90a para un Laplaciano 7-puntos N^3
// (misma estructura que la matriz de presion de icoFoam).
//
// Uso: ./poc_amg N [reps]
//   N = celdas por lado (32, 64, 128...)
//
// Compila: hipcc -O3 -std=c++17 -I/opt/rocm/include poc_amg.cpp -o poc_amg \
//                 -L/opt/rocm/lib -lrocalution -Wl,-rpath,/opt/rocm/lib

#include <rocalution/rocalution.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

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
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 2;
    const long n = 1L * N * N * N;

    // --- matriz Laplaciano 7 puntos con Dirichlet (diag=6, off=-1) ---
    int* rp = new int[n + 1];
    int* col = new int[7 * n];
    double* val = new double[7 * n];
    long nnz = 0;

    auto id = [N](int i, int j, int k) -> int { return (i * N + j) * N + k; };

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

    std::printf("N=%d  n=%ld  nnz=%ld (%.2f nnz/fila)\n", N, n, nnz, double(nnz) / n);

    // copias en host para verificar el residuo al final (A tomara propiedad)
    std::vector<int> rp_cpu(rp, rp + n + 1);
    std::vector<int> col_cpu(col, col + nnz);
    std::vector<double> val_cpu(val, val + nnz);

    init_rocalution(0);   // GPU 0

    double t_setup = now_ms();
    LocalMatrix<double> A;
    A.SetDataPtrCSR(&rp, &col, &val, "A", nnz, n, n);   // toma propiedad

    LocalVector<double> x, b;
    x.Allocate("x", n);
    b.Allocate("b", n);
    x.Zeros();
    b.Ones();

    // mover al acelerador (rocALUTION crea en host por defecto)
    A.MoveToAccelerator();
    x.MoveToAccelerator();
    b.MoveToAccelerator();

    using Mat = LocalMatrix<double>;
    using Vec = LocalVector<double>;
    const std::string pc = (argc > 3) ? argv[3] : "amg";

    CG<Mat, Vec, double> cg;
    RugeStuebenAMG<Mat, Vec, double> amg;
    cg.SetOperator(A);
    if (pc == "amg")
        cg.SetPreconditioner(amg);
    cg.InitMaxIter(pc == "amg" ? 300 : 2000);
    cg.InitTol(1e-10, 1e-6, 1e8);
    cg.SetResidualNorm(2);
    cg.Build();
    double build_ms = now_ms() - t_setup;

    for (int r = 0; r < reps; ++r)
    {
        x.Zeros();
        double t0 = now_ms();
        cg.Solve(b, &x);
        double solve_ms = now_ms() - t0;
        std::printf("  rep %d: solve+setup %.2f ms  (AMG build inicial %.1f ms)\n",
                    r, solve_ms, (r == 0 ? build_ms : 0.0));
        if (r == 0)
            std::printf("  iteraciones: %d\n", cg.GetIterationCount());
    }

    // residuo relativo real ||b - Ax||_2 / ||b||_2, calculado en host
    std::vector<double> xh(n);
    x.CopyToData(&xh[0]);           // asegura sincronizacion GPU->host
    double num = 0.0, den = 0.0;
    for (long i = 0; i < n; ++i)
    {
        double Ax = 0.0;
        for (int p = rp_cpu[i]; p < rp_cpu[i + 1]; ++p)
            Ax += val_cpu[p] * xh[col_cpu[p]];
        double d = 1.0 - Ax;         // b = 1
        num += d * d;
        den += 1.0;
    }
    std::printf("  residuo relativo ||b-Ax||/||b|| = %.3e\n", std::sqrt(num / den));
    std::printf("  iteraciones: %d  status: %d\n", cg.GetIterationCount(), cg.GetSolverStatus());

    cg.Clear();
    stop_rocalution();
    return 0;
}
