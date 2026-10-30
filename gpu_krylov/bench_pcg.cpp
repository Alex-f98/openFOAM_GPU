// GPU-resident Krylov benchmark on a real OpenFOAM pressure matrix (128^3).
//
// Reads the CSR + rhs exported by the rocALUTION plugin (rocDump) and runs
// PCG entirely on the device. Purpose: measure what the GPU can do *without*
// AMG setup and without host round trips per iteration.
//
// Build: hipcc -O3 -std=c++17 bench_pcg.cpp -o bench_pcg \
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
#include <string>
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

// --- simple device kernels -------------------------------------------------

__global__ void jacobiApply
(
    int n, const double* __restrict__ r, const double* __restrict__ d,
    double* __restrict__ z
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) z[i] = r[i]/d[i];
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

// scatter-free CSR SpMV: one workitem per row
__global__ void csrSpMV
(
    int n, const int* __restrict__ rp, const int* __restrict__ ci,
    const double* __restrict__ v, const double* __restrict__ x,
    double* __restrict__ y
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    double s = 0.0;
    for (int p = rp[i], e = rp[i + 1]; p < e; ++p)
    {
        s = fma(v[p], x[ci[p]], s);
    }
    y[i] = s;
}

// dot with block reduction + atomic
__global__ void dotK
(
    int n, const double* __restrict__ a, const double* __restrict__ b,
    double* __restrict__ out
)
{
    __shared__ double sh[256];
    int tid = threadIdx.x;
    double s = 0.0;
    for (int i = blockIdx.x*blockDim.x + tid; i < n; i += blockDim.x*gridDim.x)
    {
        s += a[i]*b[i];
    }
    sh[tid] = s;
    __syncthreads();
    for (int o = blockDim.x/2; o > 0; o >>= 1)
    {
        if (tid < o) sh[tid] += sh[tid + o];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sh[0]);
}

// ROCm 7.x: alpha/beta are pointers, plus a mat_info argument
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

    printf("n=%lld nnz=%lld nnz/row=%.2f\n", (long long)n, (long long)nnz, double(nnz)/n);

    // --- device buffers ---
    int *d_rp, *d_ci;
    double *d_val, *d_b, *d_x, *d_r, *d_z, *d_p, *d_Ap, *d_diag, *d_dot;
    CHK(hipMalloc(&d_rp, sizeof(int)*(n + 1)));
    CHK(hipMalloc(&d_ci, sizeof(int)*nnz));
    CHK(hipMalloc(&d_val, sizeof(double)*nnz));
    CHK(hipMalloc(&d_b, sizeof(double)*n));
    CHK(hipMalloc(&d_x, sizeof(double)*n));
    CHK(hipMalloc(&d_r, sizeof(double)*n));
    CHK(hipMalloc(&d_z, sizeof(double)*n));
    CHK(hipMalloc(&d_p, sizeof(double)*n));
    CHK(hipMalloc(&d_Ap, sizeof(double)*n));
    CHK(hipMalloc(&d_diag, sizeof(double)*n));
    CHK(hipMalloc(&d_dot, sizeof(double)));

    CHK(hipMemcpy(d_rp, rp.data(), sizeof(int)*(n + 1), hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_ci, ci.data(), sizeof(int)*nnz, hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_val, val.data(), sizeof(double)*nnz, hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_b, b.data(), sizeof(double)*n, hipMemcpyHostToDevice));

    // diagonal
    {
        std::vector<double> d(n);
        for (long i = 0; i < n; ++i)
        {
            double s = 1.0;
            for (int p = rp[i]; p < rp[i + 1]; ++p) if (ci[p] == i) s = val[p];
            d[i] = s;
        }
        CHK(hipMemcpy(d_diag, d.data(), sizeof(double)*n, hipMemcpyHostToDevice));
    }

    // --- rocSPARSE handle (optimised SpMV) ---
    rocsparse_handle sp;
    CHK(rocsparse_create_handle(&sp));
    rocsparse_mat_descr descr;
    CHK(rocsparse_create_mat_descr(&descr));

    CHK(spmv(sp, descr, int(n), int(nnz), d_val, d_rp, d_ci, d_p, d_Ap));

    const int T = 256;
    const int B = (int(n) + T - 1)/T;

    // --- SpMV bandwidth ---
    {
        for (int i = 0; i < 5; ++i)
            spmv(sp, descr, int(n), int(nnz), d_val, d_rp, d_ci, d_p, d_Ap);
        CHK(hipDeviceSynchronize());
        const int reps = 100;
        double t0 = now_ms();
        for (int i = 0; i < reps; ++i)
            spmv(sp, descr, int(n), int(nnz), d_val, d_rp, d_ci, d_p, d_Ap);
        CHK(hipDeviceSynchronize());
        double dt = (now_ms() - t0)/reps;
        double bytes = nnz*(8.0 + 4.0) + n*2.0*8.0;
        printf("rocsparse SpMV : %.3f ms  (%.0f GB/s effective)\n", dt, bytes/dt/1e6);

        t0 = now_ms();
        for (int i = 0; i < reps; ++i)
            csrSpMV<<<B, T>>>(int(n), d_rp, d_ci, d_val, d_p, d_Ap);
        CHK(hipDeviceSynchronize());
        dt = (now_ms() - t0)/reps;
        printf("custom   SpMV : %.3f ms  (%.0f GB/s effective)\n", dt, bytes/dt/1e6);
    }

    // --- PCG + Jacobi, device resident (hipBLAS dots) ---
    hipblasHandle_t hb;
    CHK(hipblasCreate(&hb));

    CHK(hipMemset(d_x, 0, sizeof(double)*n));
    CHK(hipMemcpy(d_r, d_b, sizeof(double)*n, hipMemcpyDeviceToDevice));

    double bnorm = 0.0;
    CHK(hipblasDdot(hb, int(n), d_b, 1, d_b, 1, &bnorm));
    bnorm = std::sqrt(bnorm);

    double rz = 0.0, rzOld = 0.0, alpha = 0.0, beta = 0.0, pap = 0.0, rnorm = 0.0;

    // z = M^-1 r ; p = z ; rz = r.z
    jacobiApply<<<B, T>>>(int(n), d_r, d_diag, d_z);
    copyK<<<B, T>>>(int(n), d_z, d_p);
    CHK(hipblasDdot(hb, int(n), d_r, 1, d_z, 1, &rz));

    const double target = tol*bnorm;
    int it = 0;
    double tSpMV = 0.0;
    hipEvent_t ev0, ev1;
    CHK(hipEventCreate(&ev0));
    CHK(hipEventCreate(&ev1));

    double t0 = now_ms();
    for (it = 0; it < maxIt; ++it)
    {
        // SpMV timed with events (async launch is not a measurement)
        CHK(hipEventRecord(ev0));
        spmv(sp, descr, int(n), int(nnz), d_val, d_rp, d_ci, d_p, d_Ap);
        CHK(hipEventRecord(ev1));

        CHK(hipblasDdot(hb, int(n), d_p, 1, d_Ap, 1, &pap));
        alpha = rz/pap;

        axpyK<<<B, T>>>(int(n), alpha, d_p, d_x);
        axpyK<<<B, T>>>(int(n), -alpha, d_Ap, d_r);

        CHK(hipblasDdot(hb, int(n), d_r, 1, d_r, 1, &rnorm));
        if (std::sqrt(rnorm) < target) { ++it; break; }

        jacobiApply<<<B, T>>>(int(n), d_r, d_diag, d_z);
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
        "PCG(Jacobi) tol=%.0e : iters=%d  total=%.1f ms  per-iter=%.3f ms"
        "  SpMV=%.3f ms  res/||b||=%.3e\n",
        tol, it, tTotal, tTotal/it, tSpMV, std::sqrt(rnorm)/bnorm
    );

    hipblasDestroy(hb);
    return 0;
}
