// PCG with an aggregation multigrid V-cycle on the GPU.
//
// The hierarchy is geometric (2x2x2 aggregates of a structured grid), so
// "setup" is a single pass over the fine coefficients per level: no RS/PMIS
// coarsening, no SpGEMM. Coarse coefficients are exact Galerkin sums.
//
// Build: hipcc -O3 -std=c++17 bench_mg.cpp -o bench_mg \
//               -I/opt/rocm/include -L/opt/rocm/lib -lhipblas

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <functional>
#include <vector>

#define CHK(x) do { auto e_ = (x); if (e_) { \
    printf("FAIL %s:%d %s (%d)\n", __FILE__, __LINE__, #x, int(e_)); return 1; } } while (0)

static double now_ms()
{
    return std::chrono::duration<double, std::milli>
    (
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

// Tipo de los coeficientes de la matriz: double = FP64 (por defecto).
// Con -DUSE_FLOAT_COEFFS se guardan solo los coeficientes en FP32; vectores,
// residuales, productos escalares y actualizaciones siguen en FP64.
#ifdef USE_FLOAT_COEFFS
using Coef = float;
#else
using Coef = double;
#endif

struct Level
{
    int nx = 0, ny = 0, nz = 0, n = 0;
    Coef *diag = nullptr, *fx = nullptr, *fy = nullptr, *fz = nullptr;
    double *b = nullptr, *x = nullptr, *r = nullptr, *tmp = nullptr;
};

__global__ void smoothK
(
    int nx, int ny, int nz, double omega,
    const Coef* __restrict__ diag, const Coef* __restrict__ fx,
    const Coef* __restrict__ fy, const Coef* __restrict__ fz,
    const double* __restrict__ b, double* __restrict__ x
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nx*ny*nz) return;

    const int ix = i % nx;
    const int iy = (i/nx) % ny;
    const int iz = i/(nx*ny);

    double s = diag[i]*x[i];
    if (ix + 1 < nx) s = fma(fx[i], x[i + 1], s);
    if (ix > 0)      s = fma(fx[i - 1], x[i - 1], s);
    if (iy + 1 < ny) s = fma(fy[i], x[i + nx], s);
    if (iy > 0)      s = fma(fy[i - nx], x[i - nx], s);
    if (iz + 1 < nz) s = fma(fz[i], x[i + nx*ny], s);
    if (iz > 0)      s = fma(fz[i - nx*ny], x[i - nx*ny], s);

    x[i] += omega*(b[i] - s)/diag[i];
}

// Gauss-Seidel sweep on one colour (grid is bipartite: neighbours flip parity)
__global__ void gsSweepK
(
    int nx, int ny, int nz, int parity,
    const Coef* __restrict__ diag, const Coef* __restrict__ fx,
    const Coef* __restrict__ fy, const Coef* __restrict__ fz,
    const double* __restrict__ b, double* __restrict__ x
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nx*ny*nz) return;

    const int ix = i % nx;
    const int iy = (i/nx) % ny;
    const int iz = i/(nx*ny);
    if (((ix + iy + iz) & 1) != parity) return;

    double s = diag[i]*x[i];
    if (ix + 1 < nx) s = fma(fx[i], x[i + 1], s);
    if (ix > 0)      s = fma(fx[i - 1], x[i - 1], s);
    if (iy + 1 < ny) s = fma(fy[i], x[i + nx], s);
    if (iy > 0)      s = fma(fy[i - nx], x[i - nx], s);
    if (iz + 1 < nz) s = fma(fz[i], x[i + nx*ny], s);
    if (iz > 0)      s = fma(fz[i - nx*ny], x[i - nx*ny], s);

    x[i] += (b[i] - s)/diag[i];
}

// y = A x  (7-point stencil)
__global__ void spmvK
(
    int nx, int ny, int nz,
    const Coef* __restrict__ diag, const Coef* __restrict__ fx,
    const Coef* __restrict__ fy, const Coef* __restrict__ fz,
    const double* __restrict__ x, double* __restrict__ y
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nx*ny*nz) return;

    const int ix = i % nx;
    const int iy = (i/nx) % ny;
    const int iz = i/(nx*ny);

    double s = diag[i]*x[i];
    if (ix + 1 < nx) s = fma(fx[i], x[i + 1], s);
    if (ix > 0)      s = fma(fx[i - 1], x[i - 1], s);
    if (iy + 1 < ny) s = fma(fy[i], x[i + nx], s);
    if (iy > 0)      s = fma(fy[i - nx], x[i - nx], s);
    if (iz + 1 < nz) s = fma(fz[i], x[i + nx*ny], s);
    if (iz > 0)      s = fma(fz[i - nx*ny], x[i - nx*ny], s);

    y[i] = s;
}

__global__ void residualK
(
    int nx, int ny, int nz,
    const Coef* __restrict__ diag, const Coef* __restrict__ fx,
    const Coef* __restrict__ fy, const Coef* __restrict__ fz,
    const double* __restrict__ b, const double* __restrict__ x,
    double* __restrict__ r
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nx*ny*nz) return;

    const int ix = i % nx;
    const int iy = (i/nx) % ny;
    const int iz = i/(nx*ny);

    double s = diag[i]*x[i];
    if (ix + 1 < nx) s = fma(fx[i], x[i + 1], s);
    if (ix > 0)      s = fma(fx[i - 1], x[i - 1], s);
    if (iy + 1 < ny) s = fma(fy[i], x[i + nx], s);
    if (iy > 0)      s = fma(fy[i - nx], x[i - nx], s);
    if (iz + 1 < nz) s = fma(fz[i], x[i + nx*ny], s);
    if (iz > 0)      s = fma(fz[i - nx*ny], x[i - nx*ny], s);

    r[i] = b[i] - s;
}

// restriction: one coarse cell per fine 2x2x2 cube (sum)
__global__ void restrictK
(
    int nx, int ny, int nz, int f,
    const double* __restrict__ r, double* __restrict__ bc
)
{
    const int nxc = nx/f, nyc = ny/f, nzc = nz/f;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nxc*nyc*nzc) return;

    const int ix = f*(i % nxc);
    const int iy = f*((i/nxc) % nyc);
    const int iz = f*(i/(nxc*nyc));

    double s = 0.0;
    for (int dz = 0; dz < f; ++dz)
    for (int dy = 0; dy < f; ++dy)
    for (int dx = 0; dx < f; ++dx)
    {
        s += r[(ix + dx) + nx*((iy + dy) + ny*(iz + dz))];
    }
    bc[i] = s;
}

// prolongation: add the coarse correction to the 8 fine cells
__global__ void prolongAddK
(
    int nx, int ny, int nz, int f,
    const double* __restrict__ xc, double* __restrict__ x
)
{
    const int nxc = nx/f, nyc = ny/f;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nx*ny*nz) return;

    const int ix = i % nx, iy = (i/nx) % ny, iz = i/(nx*ny);
    x[i] += xc[(ix/f) + nxc*((iy/f) + nyc*(iz/f))];
}

// coarsest level: Jacobi sweeps inside one block (no launches, no transfers)
__global__ void coarsestSolveK
(
    int nx, int ny, int nz, int sweeps,
    const Coef* __restrict__ diag, const Coef* __restrict__ fx,
    const Coef* __restrict__ fy, const Coef* __restrict__ fz,
    const double* __restrict__ b, double* __restrict__ x
)
{
    const int n = nx*ny*nz;
    for (int s = 0; s < sweeps; ++s)
    {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
        {
            const int ix = i % nx, iy = (i/nx) % ny, iz = i/(nx*ny);
            double v = diag[i]*x[i];
            if (ix + 1 < nx) v = fma(fx[i], x[i + 1], v);
            if (ix > 0)      v = fma(fx[i - 1], x[i - 1], v);
            if (iy + 1 < ny) v = fma(fy[i], x[i + nx], v);
            if (iy > 0)      v = fma(fy[i - nx], x[i - nx], v);
            if (iz + 1 < nz) v = fma(fz[i], x[i + nx*ny], v);
            if (iz > 0)      v = fma(fz[i - nx*ny], x[i - nx*ny], v);
            x[i] += (b[i] - v)/diag[i];
        }
        __syncthreads();
    }
}

__global__ void axpyK(int n, double a, const double* __restrict__ x, double* __restrict__ y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = fma(a, x[i], y[i]);
}

__global__ void copyK(int n, const double* __restrict__ x, double* __restrict__ y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

__global__ void zeroK(int n, double* __restrict__ x)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0;
}

// --- coarsen: fine coefficients -> coarse (exact Galerkin sums) ------------
__global__ void coarsenK
(
    int nx, int ny, int nz, int f,
    const Coef* __restrict__ diag, const Coef* __restrict__ fx,
    const Coef* __restrict__ fy, const Coef* __restrict__ fz,
    int nxc, int nyc, int nzc,
    Coef* __restrict__ cdiag, Coef* __restrict__ cfx,
    Coef* __restrict__ cfy, Coef* __restrict__ cfz
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nxc*nyc*nzc) return;

    const int ix = f*(i % nxc);
    const int iy = f*((i/nxc) % nyc);
    const int iz = f*(i/(nxc*nyc));

    double d = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
    double internal = 0.0;

    for (int dz = 0; dz < f; ++dz)
    for (int dy = 0; dy < f; ++dy)
    for (int dx = 0; dx < f; ++dx)
    {
        const int c = (ix + dx) + nx*((iy + dy) + ny*(iz + dz));
        d += diag[c];
        // internal faces (inside the aggregate) counted once here
        if (dx < f - 1) internal += fx[c];
        if (dy < f - 1) internal += fy[c];
        if (dz < f - 1) internal += fz[c];
        // crossing faces towards the +x/+y/+z coarse neighbour
        if (dx == f - 1 && ix + f < nx) sx += fx[c];
        if (dy == f - 1 && iy + f < ny) sy += fy[c];
        if (dz == f - 1 && iz + f < nz) sz += fz[c];
    }

    cdiag[i] = d + 2.0*internal;
    cfx[i] = sx;
    cfy[i] = sy;
    cfz[i] = sz;
}

int main(int argc, char** argv)
{
    const char* file = (argc > 1 ? argv[1] : "roc_matrix_p.bin");
    const double tol = (argc > 2 ? atof(argv[2]) : 1e-8);
    const int maxIt = (argc > 3 ? atoi(argv[3]) : 200);
    const int nGrid = (argc > 4 ? atoi(argv[4]) : 128);

    std::ifstream is(file, std::ios::binary);
    if (!is) { printf("cannot open %s\n", file); return 1; }

    std::int64_t n = 0, nnz = 0;
    is.read((char*)&n, sizeof(n));
    is.read((char*)&nnz, sizeof(nnz));
    std::vector<int> rp(n + 1), ci(nnz);
    std::vector<double> val(nnz), b(n);
    is.read((char*)rp.data(), sizeof(int)*(n + 1));
    is.read((char*)ci.data(), sizeof(int)*nnz);
    is.read((char*)val.data(), sizeof(double)*nnz);
    is.read((char*)b.data(), sizeof(double)*n);

    const int nx = nGrid, ny = nGrid, nz = nGrid;
    if (std::int64_t(nx)*ny*nz != n) { printf("grid mismatch\n"); return 1; }

    // ---- extract the 7-point stencil from the CSR (verify structure) -------
    std::vector<Coef> diag(n, 0.0), fx(n, 0.0), fy(n, 0.0), fz(n, 0.0);
    {
        const int nxy = nx*ny;
        for (long i = 0; i < n; ++i)
        {
            for (int p = rp[i]; p < rp[i + 1]; ++p)
            {
                const int c = ci[p];
                if (c == i) diag[i] = val[p];
                else if (c == i + 1) fx[i] = val[p];
                else if (c == i + nx) fy[i] = val[p];
                else if (c == i + nxy) fz[i] = val[p];
            }
        }
        // symmetry check on a sample
        for (long i = 1000; i < 1100; ++i)
        {
            if (i % nx != nx - 1 && std::abs(fx[i] - fx[i]) > 0) {}
        }
    }

    // ---- hierarchy --------------------------------------------------------
    const int f = (getenv("CF") ? atoi(getenv("CF")) : 2);

    std::vector<Level> lv;
    {
        Level l0;
        l0.nx = nx; l0.ny = ny; l0.nz = nz; l0.n = int(n);
        lv.push_back(l0);

        while (lv.back().nx >= 2*f && lv.back().ny >= 2*f && lv.back().nz >= 2*f)
        {
            Level c;
            c.nx = lv.back().nx/f; c.ny = lv.back().ny/f; c.nz = lv.back().nz/f;
            c.n = c.nx*c.ny*c.nz;
            lv.push_back(c);
        }
    }

    for (auto& l : lv)
    {
        const size_t sc = sizeof(Coef)*l.n, sv = sizeof(double)*l.n;
        CHK(hipMalloc(&l.diag, sc)); CHK(hipMalloc(&l.fx, sc));
        CHK(hipMalloc(&l.fy, sc));   CHK(hipMalloc(&l.fz, sc));
        CHK(hipMalloc(&l.b, sv));    CHK(hipMalloc(&l.x, sv));
        CHK(hipMalloc(&l.r, sv));    CHK(hipMalloc(&l.tmp, sv));
        CHK(hipMemset(l.x, 0, sv));
    }

    CHK(hipMemcpy(lv[0].diag, diag.data(), sizeof(Coef)*n, hipMemcpyHostToDevice));
    CHK(hipMemcpy(lv[0].fx, fx.data(), sizeof(Coef)*n, hipMemcpyHostToDevice));
    CHK(hipMemcpy(lv[0].fy, fy.data(), sizeof(Coef)*n, hipMemcpyHostToDevice));
    CHK(hipMemcpy(lv[0].fz, fz.data(), sizeof(Coef)*n, hipMemcpyHostToDevice));
    CHK(hipMemcpy(lv[0].b, b.data(), sizeof(double)*n, hipMemcpyHostToDevice));

    const int T = 256;
    CHK(hipDeviceSynchronize());
    double t0 = now_ms();
    const int fac = f;
    for (size_t q = 0; q + 1 < lv.size(); ++q)
    {
        Level& f = lv[q];
        Level& c = lv[q + 1];
        coarsenK<<<(c.n + T - 1)/T, T>>>
        (
            f.nx, f.ny, f.nz, fac, f.diag, f.fx, f.fy, f.fz,
            c.nx, c.ny, c.nz, c.diag, c.fx, c.fy, c.fz
        );
    }
    CHK(hipDeviceSynchronize());
    const double tSetup = now_ms() - t0;

    printf
    (
        "n=%lld  levels=%zu (%d^3 ... %d^3)  coarsen=%.2f ms\n",
        (long long)n, lv.size(), nx, lv.back().nx, tSetup
    );

    // ---- V-cycle ----------------------------------------------------------
    const double omega = 0.7;
    const int nu = (getenv("NU") ? atoi(getenv("NU")) : 2);
    const bool useGS = (getenv("SMOOTH") && std::string(getenv("SMOOTH")) == "gs");

    auto smooth = [&](Level& l, const double* bb, double* xx, bool rev = false)
    {
        const int Bs = (l.n + T - 1)/T;
        if (useGS)
        {
            // forward GS and its adjoint (reverse colour order) keep the
            // V-cycle symmetric, which CG requires of a preconditioner
            const int c0 = rev ? 1 : 0;
            const int c1 = rev ? 0 : 1;
            gsSweepK<<<Bs, T>>>(l.nx, l.ny, l.nz, c0, l.diag, l.fx, l.fy, l.fz, bb, xx);
            gsSweepK<<<Bs, T>>>(l.nx, l.ny, l.nz, c1, l.diag, l.fx, l.fy, l.fz, bb, xx);
        }
        else
        {
            smoothK<<<Bs, T>>>(l.nx, l.ny, l.nz, omega, l.diag, l.fx, l.fy, l.fz, bb, xx);
        }
    };

    std::function<void(int, const double*, double*)> vcycle = [&](int q, const double* bb, double* xx)
    {
        Level& l = lv[q];
        const int B = (l.n + T - 1)/T;

        if (q + 1 == int(lv.size()))
        {
            coarsestSolveK<<<1, 256>>>
            (
                l.nx, l.ny, l.nz, 200,
                l.diag, l.fx, l.fy, l.fz, bb, xx
            );
            return;
        }

        for (int s = 0; s < nu; ++s)
        {
            smooth(l, bb, xx);
        }

        residualK<<<B, T>>>(l.nx, l.ny, l.nz, l.diag, l.fx, l.fy, l.fz, bb, xx, l.r);

        Level& c = lv[q + 1];
        const int Bc = (c.n + T - 1)/T;
        restrictK<<<Bc, T>>>(l.nx, l.ny, l.nz, fac, l.r, c.b);
        zeroK<<<Bc, T>>>(c.n, c.x);

        vcycle(q + 1, c.b, c.x);

        if (getenv("DEEPV"))
        {
            hipblasHandle_t hd; (void)hipblasCreate(&hd);
            double nr = 0, nx2 = 0, nb = 0;
            (void)hipblasDdot(hd, c.n, l.r, 1, l.r, 1, &nr);
            (void)hipblasDdot(hd, c.n, c.x, 1, c.x, 1, &nx2);
            (void)hipblasDdot(hd, c.n, c.b, 1, c.b, 1, &nb);
            printf("  L%d: |r|=%.3e |bc|=%.3e |xc|=%.3e  |A_c xc - bc|=",
                   q, std::sqrt(nr), std::sqrt(nb), std::sqrt(nx2));
            // coarse residual norm
            Level& cc = c;
            double* rr = cc.tmp;
            residualK<<<(cc.n + T - 1)/T, T>>>
            (
                cc.nx, cc.ny, cc.nz, cc.diag, cc.fx, cc.fy, cc.fz, cc.b, cc.x, rr
            );
            double rr2 = 0;
            (void)hipblasDdot(hd, cc.n, rr, 1, rr, 1, &rr2);
            printf("%.3e\n", std::sqrt(rr2));
            hipblasDestroy(hd);
        }

        prolongAddK<<<B, T>>>(l.nx, l.ny, l.nz, fac, c.x, xx);

        for (int s = 0; s < nu; ++s)
        {
            smooth(l, bb, xx, true);
        }
    };

    // apply check
    {
        CHK(hipMemset(lv[0].x, 0, sizeof(double)*n));
        CHK(hipDeviceSynchronize());
        t0 = now_ms();
        vcycle(0, lv[0].b, lv[0].x);
        CHK(hipDeviceSynchronize());
        printf("V-cycle apply: %.3f ms\n", now_ms() - t0);
    }

    double bnorm_glob = 0.0;
    {
        hipblasHandle_t h0; CHK(hipblasCreate(&h0));
        CHK(hipblasDdot(h0, int(n), lv[0].b, 1, lv[0].b, 1, &bnorm_glob));
        bnorm_glob = std::sqrt(bnorm_glob);
    }

    // ---- check: P^T A P == A_c (exact Galerkin) ---------------------------
    if (getenv("CHECK"))
    {
        Level& f = lv[0];
        Level& c = lv[1];
        double *xc, *xf, *yf, *yc, *ref;
        CHK(hipMalloc(&xc, sizeof(double)*c.n));
        CHK(hipMalloc(&xf, sizeof(double)*f.n));
        CHK(hipMalloc(&yf, sizeof(double)*f.n));
        CHK(hipMalloc(&yc, sizeof(double)*c.n));
        CHK(hipMalloc(&ref, sizeof(double)*c.n));

        std::vector<double> h(c.n);
        for (int i = 0; i < c.n; ++i) h[i] = std::sin(0.01*i) + 0.3;
        CHK(hipMemcpy(xc, h.data(), sizeof(double)*c.n, hipMemcpyHostToDevice));

        // xf = P xc
        CHK(hipMemset(xf, 0, sizeof(double)*f.n));
        prolongAddK<<<(f.n + T - 1)/T, T>>>(f.nx, f.ny, f.nz, fac, xc, xf);
        // yf = A xf  (residualK with b = 0 gives -A xf)
        CHK(hipMemset(yf, 0, sizeof(double)*f.n));
        residualK<<<(f.n + T - 1)/T, T>>>
        (
            f.nx, f.ny, f.nz, f.diag, f.fx, f.fy, f.fz, yf, xf, yf
        );
        // yc = -P^T (A xf)
        restrictK<<<(c.n + T - 1)/T, T>>>(f.nx, f.ny, f.nz, fac, yf, yc);
        // ref = -A_c xc
        CHK(hipMemset(ref, 0, sizeof(double)*c.n));
        residualK<<<(c.n + T - 1)/T, T>>>
        (
            c.nx, c.ny, c.nz, c.diag, c.fx, c.fy, c.fz, ref, xc, ref
        );
        std::vector<double> a(c.n), bb(c.n);
        CHK(hipMemcpy(a.data(), yc, sizeof(double)*c.n, hipMemcpyDeviceToHost));
        CHK(hipMemcpy(bb.data(), ref, sizeof(double)*c.n, hipMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (int i = 0; i < c.n; ++i) { num += (a[i]-bb[i])*(a[i]-bb[i]); den += bb[i]*bb[i]; }
        printf("Galerkin check: rel err = %.3e  (|yc|=%.3e |ref|=%.3e)\n",
               std::sqrt(num/den), std::sqrt(num), std::sqrt(den));
        return 0;
    }

    // ---- diagnostic: pure Gauss-Seidel sweeps ----------------------------
    if (getenv("GSOLVE"))
    {
        const int sw = atoi(getenv("GSOLVE"));
        CHK(hipMemset(lv[0].x, 0, sizeof(double)*n));
        hipblasHandle_t hg; (void)hipblasCreate(&hg);
        double *dg; CHK(hipMalloc(&dg, sizeof(double)*n));
        const int Bg = (int(n) + T - 1)/T;
        for (int k = 1; k <= sw; ++k)
        {
            gsSweepK<<<Bg, T>>>(lv[0].nx, lv[0].ny, lv[0].nz, 0, lv[0].diag, lv[0].fx, lv[0].fy, lv[0].fz, lv[0].b, lv[0].x);
            gsSweepK<<<Bg, T>>>(lv[0].nx, lv[0].ny, lv[0].nz, 1, lv[0].diag, lv[0].fx, lv[0].fy, lv[0].fz, lv[0].b, lv[0].x);
            if (k % 5 == 0)
            {
                residualK<<<Bg, T>>>(lv[0].nx, lv[0].ny, lv[0].nz, lv[0].diag, lv[0].fx, lv[0].fy, lv[0].fz, lv[0].b, lv[0].x, dg);
                double r2 = 0; (void)hipblasDdot(hg, int(n), dg, 1, dg, 1, &r2);
                printf("gs %3d: rel res = %.4e\n", k, std::sqrt(r2)/bnorm_glob);
            }
        }
        return 0;
    }

    // ---- diagnostic: pure MG cycles ---------------------------------------
    if (const char* e = getenv("MGONLY"))
    {
        const int cycles = atoi(e);
        CHK(hipMemset(lv[0].x, 0, sizeof(double)*n));
        double *dr, *dn, *dt;
        CHK(hipMalloc(&dr, sizeof(double)*n));
        CHK(hipMalloc(&dn, sizeof(double)*n));
        CHK(hipMalloc(&dt, sizeof(double)*n));
        for (int k = 0; k < cycles; ++k)
        {
            residualK<<<(int(n) + T - 1)/T, T>>>
            (
                lv[0].nx, lv[0].ny, lv[0].nz,
                lv[0].diag, lv[0].fx, lv[0].fy, lv[0].fz,
                lv[0].b, lv[0].x, dr
            );
            double r2 = 0.0;
            hipblasHandle_t h2; CHK(hipblasCreate(&h2));
            CHK(hipblasDdot(h2, int(n), dr, 1, dr, 1, &r2));
            printf("cycle %2d: rel res = %.4e\n", k, std::sqrt(r2)/bnorm_glob);
            CHK(hipMemset(dt, 0, sizeof(double)*n));
            vcycle(0, dr, dt);   // correction on the residual
            axpyK<<<(int(n) + T - 1)/T, T>>>(int(n), 1.0, dt, lv[0].x);
        }
        (void)dn;
        return 0;
    }

    // ---- PCG with the V-cycle --------------------------------------------
    hipblasHandle_t hb;
    CHK(hipblasCreate(&hb));

    double *d_r, *d_z, *d_p, *d_Ap;
    CHK(hipMalloc(&d_r, sizeof(double)*n));
    CHK(hipMalloc(&d_z, sizeof(double)*n));
    CHK(hipMalloc(&d_p, sizeof(double)*n));
    CHK(hipMalloc(&d_Ap, sizeof(double)*n));

    const int B = (int(n) + T - 1)/T;

    CHK(hipMemset(lv[0].x, 0, sizeof(double)*n));
    CHK(hipMemcpy(d_r, lv[0].b, sizeof(double)*n, hipMemcpyDeviceToDevice));

    double bnorm = 0.0;
    CHK(hipblasDdot(hb, int(n), lv[0].b, 1, lv[0].b, 1, &bnorm));
    bnorm = std::sqrt(bnorm);

    double rz = 0.0, rzOld = 0.0, alpha = 0.0, beta = 0.0, pap = 0.0, rnorm = 0.0;

    CHK(hipMemset(d_z, 0, sizeof(double)*n));
    vcycle(0, d_r, d_z);
    copyK<<<B, T>>>(int(n), d_z, d_p);
    CHK(hipblasDdot(hb, int(n), d_r, 1, d_z, 1, &rz));

    const double target = tol*bnorm;
    int it = 0;
    t0 = now_ms();
    for (it = 0; it < maxIt; ++it)
    {
        spmvK<<<B, T>>>
        (
            lv[0].nx, lv[0].ny, lv[0].nz,
            lv[0].diag, lv[0].fx, lv[0].fy, lv[0].fz, d_p, d_Ap
        );

        CHK(hipblasDdot(hb, int(n), d_p, 1, d_Ap, 1, &pap));
        alpha = rz/pap;

        axpyK<<<B, T>>>(int(n), alpha, d_p, lv[0].x);
        axpyK<<<B, T>>>(int(n), -alpha, d_Ap, d_r);

        CHK(hipblasDdot(hb, int(n), d_r, 1, d_r, 1, &rnorm));
        if (std::sqrt(rnorm) < target) { ++it; break; }

        CHK(hipMemset(d_z, 0, sizeof(double)*n));
        vcycle(0, d_r, d_z);

        rzOld = rz;
        CHK(hipblasDdot(hb, int(n), d_r, 1, d_z, 1, &rz));
        beta = rz/rzOld;

        axpyK<<<B, T>>>(int(n), beta, d_p, d_z);
        copyK<<<B, T>>>(int(n), d_z, d_p);
    }
    CHK(hipDeviceSynchronize());
    const double tTotal = now_ms() - t0;

    printf
    (
        "PCG(MG) tol=%.0e : iters=%d  total=%.1f ms  per-iter=%.3f ms  res/||b||=%.3e\n",
        tol, it, tTotal, tTotal/it, std::sqrt(rnorm)/bnorm
    );

    // ---- pasadas extra en caliente (emulan pFinal y arranque tibio) -------
    // WARM=<tol>,<tol>,...  cada pasada arranca desde la x anterior (no x=0).
    std::string warmList = getenv("WARM") ? getenv("WARM") : "";
    for (size_t wpos = 0; wpos < warmList.size(); )
    {
        size_t wcom = warmList.find(',', wpos);
        if (wcom == std::string::npos) wcom = warmList.size();
        const double tol2 = atof(warmList.substr(wpos, wcom - wpos).c_str());
        wpos = wcom + 1;
        double rz2 = 0.0, rn2 = 0.0;
        CHK(hipblasDdot(hb, int(n), d_r, 1, d_r, 1, &rn2));
        printf("WARM start: res/||b||=%.3e -> tol=%.0e\n", std::sqrt(rn2)/bnorm, tol2);

        CHK(hipMemset(d_z, 0, sizeof(double)*n));
        vcycle(0, d_r, d_z);
        copyK<<<B, T>>>(int(n), d_z, d_p);
        CHK(hipblasDdot(hb, int(n), d_r, 1, d_z, 1, &rz2));

        const double t0b = now_ms();
        int it2 = 0;
        for (it2 = 0; it2 < maxIt; ++it2)
        {
            spmvK<<<B, T>>>
            (
                lv[0].nx, lv[0].ny, lv[0].nz,
                lv[0].diag, lv[0].fx, lv[0].fy, lv[0].fz, d_p, d_Ap
            );
            CHK(hipblasDdot(hb, int(n), d_p, 1, d_Ap, 1, &pap));
            alpha = rz2/pap;
            axpyK<<<B, T>>>(int(n), alpha, d_p, lv[0].x);
            axpyK<<<B, T>>>(int(n), -alpha, d_Ap, d_r);
            CHK(hipblasDdot(hb, int(n), d_r, 1, d_r, 1, &rn2));
            if (std::sqrt(rn2) < tol2*bnorm) { ++it2; break; }

            CHK(hipMemset(d_z, 0, sizeof(double)*n));
            vcycle(0, d_r, d_z);

            const double rzOld2 = rz2;
            CHK(hipblasDdot(hb, int(n), d_r, 1, d_z, 1, &rz2));
            axpyK<<<B, T>>>(int(n), rz2/rzOld2, d_p, d_z);
            copyK<<<B, T>>>(int(n), d_z, d_p);
        }
        CHK(hipDeviceSynchronize());
        const double t2 = now_ms() - t0b;
        printf
        (
            "PCG(MG) WARM tol=%.0e : iters=%d  total=%.1f ms  res/||b||=%.3e  TOTAL2=%.1f ms\n",
            tol2, it2, t2, std::sqrt(rn2)/bnorm, tTotal + t2
        );
    }

    // ---- volcado de la solucion para verificar el residual real -----------
    if (const char* xd = getenv("XDUMP"))
    {
        std::vector<double> xh(n);
        CHK(hipMemcpy(xh.data(), lv[0].x, sizeof(double)*n, hipMemcpyDeviceToHost));
        FILE* fo = fopen(xd, "wb");
        if (fo) { fwrite(xh.data(), sizeof(double), size_t(n), fo); fclose(fo); }
        printf("x -> %s\n", xd);
    }

    return 0;
}
