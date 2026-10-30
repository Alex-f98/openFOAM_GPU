// Verifica si ReBuildNumeric() de rocALUTION reutiliza la jerarquia AMG
// o si por dentro hace un Build() completo.
//
// Simula pasos de tiempo: el patron CSR (row_ptr, col_idx) queda congelado
// y solo cambian los valores (escalado simetrico S*A*S, que preserva SPD).
//
// Uso: ./poc_rebuild N [rs|sa|ua] [nsteps]
//   N      = celdas por lado (32, 64, 128)
//   rs/sa/ua = RugeStueben / SAAMG / UAAMG
//
// Compila:
//   hipcc -O3 -std=c++17 -I/opt/rocm/include poc_rebuild.cpp -o poc_rebuild \
//         -L/opt/rocm/lib -lrocalution -Wl,-rpath,/opt/rocm/lib

#include <rocalution/rocalution.hpp>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace rocalution;

static double now_ms()
{
    using clk = std::chrono::steady_clock;
    static const auto t0 = clk::now();
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? std::atoi(argv[1]) : 64;
    const std::string pc = (argc > 2) ? argv[2] : "rs";
    const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 3;
    const long n = 1L * N * N * N;

    // ---- CSR del Laplaciano 7 puntos ----
    std::vector<int> rp(n + 1), col;
    std::vector<double> val;
    col.reserve(7 * n);
    val.reserve(7 * n);

    auto id = [N](int i, int j, int k) { return (i * N + j) * N + k; };

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
            {
                rp[id(i, j, k)] = static_cast<int>(col.size());
                col.push_back(id(i, j, k)); val.push_back(6.0);
                int nb[6][3] = {{i-1,j,k},{i+1,j,k},{i,j-1,k},{i,j+1,k},{i,j,k-1},{i,j,k+1}};
                for (auto& q : nb)
                    if (q[0] >= 0 && q[0] < N && q[1] >= 0 && q[1] < N && q[2] >= 0 && q[2] < N)
                    { col.push_back(id(q[0], q[1], q[2])); val.push_back(-1.0); }
            }
    rp[n] = static_cast<int>(col.size());
    const long nnz = val.size();

    std::vector<double> b0(n, 1.0);

    // rocALUTION toma propiedad de los punteros -> copias propias
    int* rp_own = new int[n + 1];
    int* col_own = new int[nnz];
    double* val_own = new double[nnz];
    std::memcpy(rp_own, rp.data(), (n + 1) * sizeof(int));
    std::memcpy(col_own, col.data(), nnz * sizeof(int));
    std::memcpy(val_own, val.data(), nnz * sizeof(double));

    std::printf("N=%d  n=%ld  nnz=%ld  precond=%s\n", N, n, nnz, pc.c_str());

    set_device_rocalution(0);
    init_rocalution();

    LocalMatrix<double> A;
    A.SetDataPtrCSR(&rp_own, &col_own, &val_own, "A", nnz, n, n);
    A.MoveToAccelerator();

    LocalVector<double> x, b;
    x.Allocate("x", n); x.Zeros();
    b.Allocate("b", n); b.CopyFromData(b0.data());
    x.MoveToAccelerator();
    b.MoveToAccelerator();

    using Mat = LocalMatrix<double>;
    using Vec = LocalVector<double>;

    CG<Mat, Vec, double> cg;
    RugeStuebenAMG<Mat, Vec, double> rs;
    SAAMG<Mat, Vec, double> sa;
    UAAMG<Mat, Vec, double> ua;

    Solver<Mat, Vec, double>* pre = nullptr;
    if (pc == "sa") pre = &sa;
    else if (pc == "ua") pre = &ua;
    else            pre = &rs;

    cg.SetOperator(A);
    cg.SetPreconditioner(*pre);
    cg.InitMaxIter(500);
    cg.InitTol(1e-10, 1e-6, 1e8);
    cg.SetResidualNorm(2);

    // ---------- Build inicial ----------
    double t = now_ms();
    cg.Build();
    double t_build = now_ms() - t;
    std::printf("  Build()          : %8.1f ms\n", t_build);

    auto solve_and_check = [&](const std::vector<double>& vv, const char* tag)
    {
        x.Zeros();
        double ts = now_ms();
        cg.Solve(b, &x);
        double t_solve = now_ms() - ts;

        // residuo real con la matriz ACTUAL (vv)
        std::vector<double> xh(n);
        x.CopyToData(xh.data());
        double num = 0.0, den = 0.0;
        for (long i = 0; i < n; ++i)
        {
            double Ax = 0.0;
            for (int p = rp[i]; p < rp[i + 1]; ++p)
                Ax += vv[p] * xh[col[p]];
            double d = b0[i] - Ax;
            num += d * d; den += b0[i] * b0[i];
        }
        std::printf("  %-16s: %8.2f ms  iters=%3d  resid=%.3e\n",
                    tag, t_solve, cg.GetIterationCount(), std::sqrt(num / den));
        return t_solve;
    };

    std::vector<double> vcur = val;
    const std::vector<double> val_base = val;  // referencia: perturbacion NO acumulativa
    solve_and_check(vcur, "solve #0");

    // ---------- Pasos siguientes: solo cambian los valores ----------
    for (int s = 1; s <= nsteps; ++s)
    {
        // perturbacion simetrica A' = S A S con S = diag(1 + 0.05*sin(...))
        std::vector<double> sv(n);
        for (long i = 0; i < n; ++i)
            sv[i] = 1.0 + 0.05 * std::sin(0.001 * double(i) + 0.7 * s);

        for (long i = 0; i < n; ++i)
            for (int p = rp[i]; p < rp[i + 1]; ++p)
                val[p] = val_base[p] * sv[i] * sv[col[p]];
        vcur = val;

        A.UpdateValuesCSR(val.data());

        t = now_ms();
        cg.ReBuildNumeric();
        double t_rebuild = now_ms() - t;

        std::printf("  --- paso %d ---\n", s);
        std::printf("  ReBuildNumeric() : %8.1f ms  (%.1f%% del Build)\n",
                    t_rebuild, 100.0 * t_rebuild / t_build);
        solve_and_check(vcur, "solve");
    }

    cg.Clear();
    stop_rocalution();
    return 0;
}
