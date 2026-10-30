// PCG with multicolour ILU(0) on the GPU: no AMG setup, no host round trips.
//
// Graph colouring -> permutation by colour -> ILU(0) factorisation of the
// permuted matrix (one kernel launch per colour, rows of a colour independent)
// -> forward/backward sweeps likewise. Iteration quality of ILU(0) with the
// cost of a few SpMVs.
//
// Build: hipcc -O3 -std=c++17 bench_ilu.cpp -o bench_ilu \
//               -I/opt/rocm/include -L/opt/rocm/lib -lrocsparse -lhipblas

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocsparse/rocsparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
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

// --- ILU(0) on the permuted matrix ----------------------------------------
// fd: first row of the colour in the colour block; nrows: rows in the colour
__global__ void ilu0Factor
(
    int n, int fd, int nrows,
    const int* __restrict__ rp, const int* __restrict__ ci,
    double* __restrict__ v, const int* __restrict__ dp
)
{
    int i = fd + blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= fd + nrows) return;

    const int e = rp[i + 1];

    for (int p = rp[i]; p < e; ++p)
    {
        const int k = ci[p];
        if (k >= i) break;                      // start of the U part

        const double lik = v[p]/v[dp[k]];
        v[p] = lik;

        // merged update of row i (>k) with row k's U part (>k)
        int q = p + 1;
        int r = dp[k] + 1;
        const int ke = rp[k + 1];

        while (q < e && r < ke)
        {
            const int cq = ci[q], cr = ci[r];
            if (cq < cr) { ++q; }
            else if (cq > cr) { ++r; }
            else { v[q] -= lik*v[r]; ++q; ++r; }
        }
    }
}

__global__ void ilu0Forward
(
    int fd, int nrows,
    const int* __restrict__ rp, const int* __restrict__ ci,
    const double* __restrict__ L, const double* __restrict__ r,
    double* __restrict__ y
)
{
    int i = fd + blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= fd + nrows) return;

    double s = r[i];
    for (int p = rp[i]; p < rp[i + 1]; ++p)
    {
        const int j = ci[p];
        if (j >= i) break;
        s -= L[p]*y[j];
    }
    y[i] = s;
}

__global__ void ilu0Backward
(
    int fd, int nrows,
    const int* __restrict__ rp, const int* __restrict__ ci,
    const double* __restrict__ U, const int* __restrict__ dp,
    const double* __restrict__ y, double* __restrict__ x
)
{
    int i = fd + blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= fd + nrows) return;

    double s = y[i];
    for (int p = dp[i] + 1; p < rp[i + 1]; ++p)
    {
        s -= U[p]*x[ci[p]];
    }
    x[i] = s/U[dp[i]];
}

static inline rocsparse_status spmv
(
    rocsparse_handle sp, rocsparse_mat_descr d, int n, int nnz,
    const double* val, const int* rp, const int* ci,
    const double* x, double* y
)
{
    double one = 1.0, zero = 0.0;
    return rocsparse_dcsrmv
    (
        sp, rocsparse_operation_none, n, n, nnz, &one, d,
        val, rp, ci, nullptr, x, &zero, y
    );
}

int main(int argc, char** argv)
{
    const char* file = (argc > 1 ? argv[1] : "roc_matrix_p.bin");
    const double tol = (argc > 2 ? atof(argv[2]) : 1e-8);
    const int maxIt = (argc > 3 ? atoi(argv[3]) : 5000);

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

    // ---------------- host: colouring + permutation ------------------------
    double tSetup = now_ms();

    std::vector<int> col(n, -1);
    {
        std::vector<int> mark;
        for (long i = 0; i < n; ++i)
        {
            int mx = -1;
            for (int p = rp[i]; p < rp[i + 1]; ++p)
            {
                const int j = ci[p];
                if (j < i && col[j] > mx) mx = col[j];
            }
            if (int(mark.size()) < mx + 2) mark.resize(mx + 2);
            for (int p = rp[i]; p < rp[i + 1]; ++p)
            {
                const int j = ci[p];
                if (j < i) mark[col[j]] = i + 1;
            }
            int c = 0;
            while (c <= mx && mark[c] == i + 1) ++c;
            col[i] = c;
        }
    }

    int ncol = 0;
    for (long i = 0; i < n; ++i) ncol = std::max(ncol, col[i] + 1);

    std::vector<int> cstart(ncol + 1, 0);
    for (long i = 0; i < n; ++i) ++cstart[col[i] + 1];
    for (int c = 0; c < ncol; ++c) cstart[c + 1] += cstart[c];

    std::vector<int> perm(n), iperm(n);
    {
        std::vector<int> next(cstart.begin(), cstart.end() - 1);
        for (long i = 0; i < n; ++i)
        {
            const int ni = next[col[i]]++;
            perm[i] = ni;
            iperm[ni] = int(i);
        }
    }

    // permuted CSR, rows sorted by column
    std::vector<int> prp(n + 1), pci(nnz);
    std::vector<double> pval(nnz);
    prp[0] = 0;
    for (long ni = 0; ni < n; ++ni)
    {
        const int i = iperm[ni];
        prp[ni + 1] = prp[ni] + (rp[i + 1] - rp[i]);
    }
    for (long ni = 0; ni < n; ++ni)
    {
        const int i = iperm[ni];
        const int b0 = prp[ni];
        const int cnt = rp[i + 1] - rp[i];
        for (int q = 0; q < cnt; ++q)
        {
            pci[b0 + q] = perm[ci[rp[i] + q]];
            pval[b0 + q] = val[rp[i] + q];
        }
        // insertion sort by column (<= 7 entries)
        for (int q = 1; q < cnt; ++q)
        {
            const int c = pci[b0 + q];
            const double v = pval[b0 + q];
            int r = q - 1;
            while (r >= 0 && pci[b0 + r] > c)
            {
                pci[b0 + r + 1] = pci[b0 + r];
                pval[b0 + r + 1] = pval[b0 + r];
                --r;
            }
            pci[b0 + r + 1] = c;
            pval[b0 + r + 1] = v;
        }
    }
    tSetup = now_ms() - tSetup;

    printf
    (
        "n=%lld nnz=%lld  colours=%d  setup(colour+perm)=%.0f ms\n",
        (long long)n, (long long)nnz, ncol, tSetup
    );

    // ---------------- device -----------------------------------------------
    int *d_rp, *d_ci, *d_dp;
    double *d_val, *d_lu, *d_b, *d_x, *d_r, *d_z, *d_p, *d_Ap, *d_y;
    CHK(hipMalloc(&d_rp, sizeof(int)*(n + 1)));
    CHK(hipMalloc(&d_ci, sizeof(int)*nnz));
    CHK(hipMalloc(&d_dp, sizeof(int)*n));
    CHK(hipMalloc(&d_val, sizeof(double)*nnz));
    CHK(hipMalloc(&d_lu, sizeof(double)*nnz));
    CHK(hipMalloc(&d_b, sizeof(double)*n));
    CHK(hipMalloc(&d_x, sizeof(double)*n));
    CHK(hipMalloc(&d_r, sizeof(double)*n));
    CHK(hipMalloc(&d_z, sizeof(double)*n));
    CHK(hipMalloc(&d_p, sizeof(double)*n));
    CHK(hipMalloc(&d_Ap, sizeof(double)*n));
    CHK(hipMalloc(&d_y, sizeof(double)*n));

    std::vector<int> dp(n);
    std::vector<double> pb(n);
    for (long i = 0; i < n; ++i)
    {
        int d = -1;
        for (int p = prp[i]; p < prp[i + 1]; ++p) if (pci[p] == i) d = p;
        if (d < 0) { printf("missing diagonal at %ld\n", long(i)); return 1; }
        dp[i] = d;
    }
    for (long i = 0; i < n; ++i) pb[perm[i]] = b[i];

    CHK(hipMemcpy(d_rp, prp.data(), sizeof(int)*(n + 1), hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_ci, pci.data(), sizeof(int)*nnz, hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_dp, dp.data(), sizeof(int)*n, hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_val, pval.data(), sizeof(double)*nnz, hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_lu, pval.data(), sizeof(double)*nnz, hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_b, pb.data(), sizeof(double)*n, hipMemcpyHostToDevice));

    rocsparse_handle sp;
    CHK(rocsparse_create_handle(&sp));
    rocsparse_mat_descr descr;
    CHK(rocsparse_create_mat_descr(&descr));

    const int T = 256;
    const int B = (int(n) + T - 1)/T;

    // ---------------- ILU(0) factorisation ---------------------------------
    CHK(hipDeviceSynchronize());
    double t0 = now_ms();
    for (int c = 0; c < ncol; ++c)
    {
        const int fd = cstart[c];
        const int nr = cstart[c + 1] - fd;
        if (nr > 0) ilu0Factor<<<(nr + T - 1)/T, T>>>(int(n), fd, nr, d_rp, d_ci, d_lu, d_dp);
    }
    CHK(hipDeviceSynchronize());
    const double tFac = now_ms() - t0;

    hipblasHandle_t hb;
    CHK(hipblasCreate(&hb));

    // ---------------- apply check ------------------------------------------
    CHK(hipMemcpy(d_r, d_b, sizeof(double)*n, hipMemcpyDeviceToDevice));
    t0 = now_ms();
    for (int c = 0; c < ncol; ++c)
    {
        const int fd = cstart[c], nr = cstart[c + 1] - fd;
        if (nr > 0) ilu0Forward<<<(nr + T - 1)/T, T>>>(fd, nr, d_rp, d_ci, d_lu, d_r, d_y);
    }
    for (int c = ncol - 1; c >= 0; --c)
    {
        const int fd = cstart[c], nr = cstart[c + 1] - fd;
        if (nr > 0) ilu0Backward<<<(nr + T - 1)/T, T>>>(fd, nr, d_rp, d_ci, d_lu, d_dp, d_y, d_z);
    }
    CHK(hipDeviceSynchronize());
    const double tApply = now_ms() - t0;

    // ---------------- PCG + ILU(0) -----------------------------------------
    CHK(hipMemset(d_x, 0, sizeof(double)*n));
    CHK(hipMemcpy(d_r, d_b, sizeof(double)*n, hipMemcpyDeviceToDevice));

    double bnorm = 0.0;
    CHK(hipblasDdot(hb, int(n), d_b, 1, d_b, 1, &bnorm));
    bnorm = std::sqrt(bnorm);

    double rz = 0.0, rzOld = 0.0, alpha = 0.0, beta = 0.0, pap = 0.0, rnorm = 0.0;

    for (int c = 0; c < ncol; ++c)
    {
        const int fd = cstart[c], nr = cstart[c + 1] - fd;
        if (nr > 0) ilu0Forward<<<(nr + T - 1)/T, T>>>(fd, nr, d_rp, d_ci, d_lu, d_r, d_y);
    }
    for (int c = ncol - 1; c >= 0; --c)
    {
        const int fd = cstart[c], nr = cstart[c + 1] - fd;
        if (nr > 0) ilu0Backward<<<(nr + T - 1)/T, T>>>(fd, nr, d_rp, d_ci, d_lu, d_dp, d_y, d_z);
    }
    copyK<<<B, T>>>(int(n), d_z, d_p);
    CHK(hipblasDdot(hb, int(n), d_r, 1, d_z, 1, &rz));

    const double target = tol*bnorm;
    int it = 0;
    double tSpMV = 0.0;
    hipEvent_t ev0, ev1;
    CHK(hipEventCreate(&ev0));
    CHK(hipEventCreate(&ev1));

    t0 = now_ms();
    for (it = 0; it < maxIt; ++it)
    {
        CHK(hipEventRecord(ev0));
        spmv(sp, descr, int(n), int(nnz), d_val, d_rp, d_ci, d_p, d_Ap);
        CHK(hipEventRecord(ev1));

        CHK(hipblasDdot(hb, int(n), d_p, 1, d_Ap, 1, &pap));
        alpha = rz/pap;

        axpyK<<<B, T>>>(int(n), alpha, d_p, d_x);
        axpyK<<<B, T>>>(int(n), -alpha, d_Ap, d_r);

        CHK(hipblasDdot(hb, int(n), d_r, 1, d_r, 1, &rnorm));
        if (std::sqrt(rnorm) < target) { ++it; break; }

        for (int c = 0; c < ncol; ++c)
        {
            const int fd = cstart[c], nr = cstart[c + 1] - fd;
            if (nr > 0) ilu0Forward<<<(nr + T - 1)/T, T>>>(fd, nr, d_rp, d_ci, d_lu, d_r, d_y);
        }
        for (int c = ncol - 1; c >= 0; --c)
        {
            const int fd = cstart[c], nr = cstart[c + 1] - fd;
            if (nr > 0) ilu0Backward<<<(nr + T - 1)/T, T>>>(fd, nr, d_rp, d_ci, d_lu, d_dp, d_y, d_z);
        }

        rzOld = rz;
        CHK(hipblasDdot(hb, int(n), d_r, 1, d_z, 1, &rz));
        beta = rz/rzOld;

        axpyK<<<B, T>>>(int(n), beta, d_p, d_z);
        copyK<<<B, T>>>(int(n), d_z, d_p);
    }
    CHK(hipDeviceSynchronize());
    const double tTotal = now_ms() - t0;
    {
        float ms = 0;
        CHK(hipEventElapsedTime(&ms, ev0, ev1));
        tSpMV = ms;
    }

    printf
    (
        "PCG(ILU0) tol=%.0e : iters=%d  total=%.1f ms  per-iter=%.3f ms"
        "  SpMV=%.3f ms  factor=%.1f ms  apply=%.3f ms  res/||b||=%.3e\n",
        tol, it, tTotal, tTotal/it, tSpMV, tFac, tApply, std::sqrt(rnorm)/bnorm
    );

    return 0;
}
