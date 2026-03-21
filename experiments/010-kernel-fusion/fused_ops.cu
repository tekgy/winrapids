// WinRapids Experiment 010: Kernel Fusion Engine
//
// The core insight from Experiment 009: CuPy launches one kernel per operation.
// For `a * b + c`, that's 2 kernels + 1 intermediate buffer (320 MB round-trip
// through VRAM for 10M elements). A fused kernel does it in 1 launch, 0
// intermediates, and uses hardware FMA.
//
// This experiment builds a COMPILE-TIME kernel fusion system in CUDA C++.
// Instead of runtime JIT (which adds latency), we use C++ templates to
// generate fused kernels at compile time. The compiler sees through the
// abstraction and emits optimal PTX.
//
// Architecture:
//   - Expression templates: build an AST at compile time
//   - evaluate() walks the AST per-element inside a single kernel
//   - No intermediate buffers allocated
//   - Hardware FMA used automatically when the compiler sees a*b+c
//
// Comparison targets:
//   1. CuPy separate ops (Experiment 004): 0.531 ms for a*b+c
//   2. Hand-written fused kernel (Experiment 009): 0.192 ms
//   3. This template-fused kernel: should match hand-written
//
// Memory: ~410 MB GPU (5 columns * 80 MB + 10 MB flags). Well under 60 GB.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ============================================================
// Expression Templates — compile-time AST for kernel fusion
// ============================================================

// Tag: a column reference (leaf node)
struct ColumnRef {
    const double* ptr;
    int n;
    __device__ double eval(int idx) const { return ptr[idx]; }
};

// Tag: a scalar constant
struct Scalar {
    double value;
    __device__ double eval(int) const { return value; }
};

// Binary operation nodes
template <typename L, typename R>
struct AddExpr {
    L left;
    R right;
    __device__ double eval(int idx) const {
        return left.eval(idx) + right.eval(idx);
    }
};

template <typename L, typename R>
struct SubExpr {
    L left;
    R right;
    __device__ double eval(int idx) const {
        return left.eval(idx) - right.eval(idx);
    }
};

template <typename L, typename R>
struct MulExpr {
    L left;
    R right;
    __device__ double eval(int idx) const {
        return left.eval(idx) * right.eval(idx);
    }
};

template <typename L, typename R>
struct DivExpr {
    L left;
    R right;
    __device__ double eval(int idx) const {
        return left.eval(idx) / right.eval(idx);
    }
};

// FMA: a*b+c — the compiler will emit hardware FMA for this
template <typename A, typename B, typename C>
struct FmaExpr {
    A a;
    B b;
    C c;
    __device__ double eval(int idx) const {
        return fma(a.eval(idx), b.eval(idx), c.eval(idx));
    }
};

// Comparison nodes (return 0.0 or 1.0 for masking)
template <typename L, typename R>
struct GtExpr {
    L left;
    R right;
    __device__ double eval(int idx) const {
        return left.eval(idx) > right.eval(idx) ? 1.0 : 0.0;
    }
};

template <typename L, typename R>
struct LtExpr {
    L left;
    R right;
    __device__ double eval(int idx) const {
        return left.eval(idx) < right.eval(idx) ? 1.0 : 0.0;
    }
};

// Unary operations
template <typename E>
struct AbsExpr {
    E expr;
    __device__ double eval(int idx) const { return fabs(expr.eval(idx)); }
};

template <typename E>
struct SqrtExpr {
    E expr;
    __device__ double eval(int idx) const { return sqrt(expr.eval(idx)); }
};

// Ternary: where(cond, then_val, else_val)
template <typename Cond, typename Then, typename Else>
struct WhereExpr {
    Cond cond;
    Then then_val;
    Else else_val;
    __device__ double eval(int idx) const {
        return cond.eval(idx) != 0.0 ? then_val.eval(idx) : else_val.eval(idx);
    }
};

// ============================================================
// Operator overloads for building expressions
// ============================================================

template <typename L, typename R>
AddExpr<L, R> operator+(L l, R r) { return {l, r}; }

template <typename L, typename R>
SubExpr<L, R> operator-(L l, R r) { return {l, r}; }

template <typename L, typename R>
MulExpr<L, R> operator*(L l, R r) { return {l, r}; }

template <typename L, typename R>
DivExpr<L, R> operator/(L l, R r) { return {l, r}; }

template <typename A, typename B, typename C>
FmaExpr<A, B, C> fused_fma(A a, B b, C c) { return {a, b, c}; }

template <typename L, typename R>
GtExpr<L, R> gt(L l, R r) { return {l, r}; }

template <typename L, typename R>
LtExpr<L, R> lt(L l, R r) { return {l, r}; }

template <typename E>
AbsExpr<E> abs_expr(E e) { return {e}; }

template <typename E>
SqrtExpr<E> sqrt_expr(E e) { return {e}; }

template <typename Cond, typename Then, typename Else>
WhereExpr<Cond, Then, Else> where_expr(Cond c, Then t, Else e) { return {c, t, e}; }

// ============================================================
// The fusion kernel — evaluates ANY expression tree
// ============================================================

template <typename Expr>
__global__ void fused_eval(Expr expr, double* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expr.eval(idx);
    }
}

// Vectorized version — processes 2 elements per thread
template <typename Expr>
__global__ void fused_eval_vec2(Expr expr, double* __restrict__ output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        output[idx]     = expr.eval(idx);
        output[idx + 1] = expr.eval(idx + 1);
    } else if (idx < n) {
        output[idx] = expr.eval(idx);
    }
}

// ============================================================
// Fused reduction — evaluate expression AND reduce in one kernel
// ============================================================

__device__ double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <typename Expr>
__global__ void fused_reduce_sum(Expr expr, double* __restrict__ output, int n) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double val = 0.0;
    if (i < n) val += expr.eval(i);
    if (i + blockDim.x < n) val += expr.eval(i + blockDim.x);

    val = warp_reduce_sum(val);

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) sdata[warp_id] = val;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        val = (lane < num_warps) ? sdata[lane] : 0.0;
        val = warp_reduce_sum(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

// ============================================================
// Host helpers
// ============================================================

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    float elapsed_ms(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

double host_reduce_sum(const double* d_partial, int nblocks) {
    double* h_partial = (double*)malloc(nblocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, nblocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < nblocks; i++) sum += h_partial[i];
    free(h_partial);
    return sum;
}

// ============================================================
// Benchmarks
// ============================================================

void benchmark_fma_fused(int n, double* d_a, double* d_b, double* d_c) {
    printf("=== Test 1: Fused a*b+c (expression template) ===\n");

    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(double)));

    ColumnRef a_col = {d_a, n};
    ColumnRef b_col = {d_b, n};
    ColumnRef c_col = {d_c, n};

    // Build the expression: a * b + c
    // The compiler sees this as a single expression tree
    auto expr = a_col * b_col + c_col;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CudaTimer timer;

    // Warmup
    fused_eval<<<blocks, threads>>>(expr, d_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    float total_ms = 0;
    int runs = 20;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        fused_eval<<<blocks, threads>>>(expr, d_result, n);
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / runs;
    double bw = 4.0 * n * sizeof(double) / (avg_ms * 1e6);
    printf("  Template fused:  %.4f ms (%.1f GB/s)\n", avg_ms, bw);

    // Compare with explicit FMA
    auto expr_fma = fused_fma(a_col, b_col, c_col);

    total_ms = 0;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        fused_eval<<<blocks, threads>>>(expr_fma, d_result, n);
        total_ms += timer.elapsed_ms();
    }
    float avg_ms_fma = total_ms / runs;
    double bw_fma = 4.0 * n * sizeof(double) / (avg_ms_fma * 1e6);
    printf("  Explicit FMA:    %.4f ms (%.1f GB/s)\n", avg_ms_fma, bw_fma);

    // Vec2 version
    int blocks2 = (n / 2 + threads - 1) / threads;
    total_ms = 0;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        fused_eval_vec2<<<blocks2, threads>>>(expr, d_result, n);
        total_ms += timer.elapsed_ms();
    }
    float avg_ms_v2 = total_ms / runs;
    double bw_v2 = 4.0 * n * sizeof(double) / (avg_ms_v2 * 1e6);
    printf("  Vec2 fused:      %.4f ms (%.1f GB/s)\n", avg_ms_v2, bw_v2);

    // Verify
    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    printf("  First result:    %.6f\n\n", h_result);

    CUDA_CHECK(cudaFree(d_result));
}

void benchmark_complex_expr(int n, double* d_a, double* d_b, double* d_c) {
    printf("=== Test 2: Complex expression (a*b + c*c - a/b) ===\n");
    printf("  CuPy would need: 5 kernel launches + 4 intermediate buffers\n");
    printf("  Fused:           1 kernel launch + 0 intermediate buffers\n\n");

    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(double)));

    ColumnRef a_col = {d_a, n};
    ColumnRef b_col = {d_b, n};
    ColumnRef c_col = {d_c, n};

    // a*b + c*c - a/b: 5 operations, 1 kernel
    auto expr = a_col * b_col + c_col * c_col - a_col / b_col;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CudaTimer timer;

    // Warmup
    fused_eval<<<blocks, threads>>>(expr, d_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    float total_ms = 0;
    int runs = 20;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        fused_eval<<<blocks, threads>>>(expr, d_result, n);
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / runs;
    // 3 reads + 1 write = 4 arrays (no intermediates!)
    double bw = 4.0 * n * sizeof(double) / (avg_ms * 1e6);
    printf("  Fused kernel:    %.4f ms (%.1f GB/s)\n", avg_ms, bw);

    // Verify
    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    printf("  First result:    %.6f\n\n", h_result);

    CUDA_CHECK(cudaFree(d_result));
}

void benchmark_where_expr(int n, double* d_a, double* d_b, double* d_c) {
    printf("=== Test 3: Fused where(a > 0, b * c, -b * c) ===\n");
    printf("  CuPy: comparison + 2 multiplies + negate + where = 5 kernels\n");
    printf("  Fused: 1 kernel, branchless per-element\n\n");

    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(double)));

    ColumnRef a_col = {d_a, n};
    ColumnRef b_col = {d_b, n};
    ColumnRef c_col = {d_c, n};
    Scalar zero = {0.0};
    Scalar neg_one = {-1.0};

    // where(a > 0, b * c, -b * c)
    auto expr = where_expr(
        gt(a_col, zero),      // condition
        b_col * c_col,        // then
        neg_one * b_col * c_col  // else
    );

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CudaTimer timer;

    // Warmup
    fused_eval<<<blocks, threads>>>(expr, d_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    float total_ms = 0;
    int runs = 20;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        fused_eval<<<blocks, threads>>>(expr, d_result, n);
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / runs;
    double bw = 4.0 * n * sizeof(double) / (avg_ms * 1e6);
    printf("  Fused kernel:    %.4f ms (%.1f GB/s)\n", avg_ms, bw);

    // Verify
    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    printf("  First result:    %.6f\n\n", h_result);

    CUDA_CHECK(cudaFree(d_result));
}

void benchmark_fused_reduce(int n, double* d_a, double* d_b, double* d_c) {
    printf("=== Test 4: Fused compute + reduce: sum(a * b + c) ===\n");
    printf("  CuPy: multiply kernel + add kernel + reduce kernel = 3 launches\n");
    printf("  Fused: 1 kernel — compute AND reduce in same launch\n\n");

    ColumnRef a_col = {d_a, n};
    ColumnRef b_col = {d_b, n};
    ColumnRef c_col = {d_c, n};

    auto expr = a_col * b_col + c_col;

    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    int smem = (threads / 32) * sizeof(double);

    double* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));

    CudaTimer timer;

    // Warmup
    fused_reduce_sum<<<blocks, threads, smem>>>(expr, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    float total_ms = 0;
    int runs = 20;
    double result = 0;

    for (int i = 0; i < runs; i++) {
        timer.begin();
        fused_reduce_sum<<<blocks, threads, smem>>>(expr, d_partial, n);
        float ms = timer.elapsed_ms();
        total_ms += ms;
        result = host_reduce_sum(d_partial, blocks);
    }

    float avg_ms = total_ms / runs;
    // Reads 3 columns, no output column (just partial sums)
    double bw = 3.0 * n * sizeof(double) / (avg_ms * 1e6);
    printf("  Fused compute+reduce: %.4f ms (%.1f GB/s)\n", avg_ms, bw);
    printf("  Result:               %.6f\n\n", result);

    CUDA_CHECK(cudaFree(d_partial));
}

void benchmark_chained_ops(int n, double* d_a, double* d_b, double* d_c) {
    printf("=== Test 5: Deep chain — sqrt(abs(a*b + c*c - a)) ===\n");
    printf("  CuPy: mul + mul + add + sub + abs + sqrt = 6 kernels + 5 temps\n");
    printf("  Fused: 1 kernel + 0 temps\n\n");

    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(double)));

    ColumnRef a_col = {d_a, n};
    ColumnRef b_col = {d_b, n};
    ColumnRef c_col = {d_c, n};

    // sqrt(abs(a*b + c*c - a))
    auto expr = sqrt_expr(abs_expr(a_col * b_col + c_col * c_col - a_col));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CudaTimer timer;

    // Warmup
    fused_eval<<<blocks, threads>>>(expr, d_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    float total_ms = 0;
    int runs = 20;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        fused_eval<<<blocks, threads>>>(expr, d_result, n);
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / runs;
    double bw = 4.0 * n * sizeof(double) / (avg_ms * 1e6);
    printf("  Fused kernel:    %.4f ms (%.1f GB/s)\n", avg_ms, bw);

    // Verify
    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    printf("  First result:    %.6f\n\n", h_result);

    CUDA_CHECK(cudaFree(d_result));
}

void benchmark_savings_analysis(int n) {
    printf("=== Memory Savings Analysis ===\n\n");

    size_t col_bytes = n * sizeof(double);
    printf("  Column size: %.1f MB\n\n", col_bytes / 1e6);

    struct OpAnalysis {
        const char* name;
        int cupy_kernels;
        int cupy_temps;
        int fused_kernels;
        int fused_temps;
    };

    OpAnalysis ops[] = {
        {"a * b + c",                        2, 1, 1, 0},
        {"a*b + c*c - a/b",                  5, 4, 1, 0},
        {"where(a>0, b*c, -b*c)",            5, 4, 1, 0},
        {"sum(a * b + c)",                    3, 1, 1, 0},
        {"sqrt(abs(a*b + c*c - a))",          6, 5, 1, 0},
    };

    printf("  %-30s  CuPy Kernels  Temps    Fused Kernels  Temps  VRAM Saved\n", "Expression");
    printf("  %-30s  ------------  -----    -------------  -----  ----------\n", "");

    for (const auto& op : ops) {
        double saved_mb = (double)op.cupy_temps * col_bytes / 1e6;
        printf("  %-30s  %12d  %3d x %.0fMB  %13d  %5d  %.0f MB\n",
               op.name, op.cupy_kernels, op.cupy_temps, col_bytes / 1e6,
               op.fused_kernels, op.fused_temps, saved_mb);
    }
    printf("\n");
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("WinRapids Experiment 010: Kernel Fusion Engine\n");
    printf("=============================================================\n\n");

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));  // Force context init

    int n = 10000000;  // 10M elements, same as Experiments 004/009
    size_t bytes = n * sizeof(double);

    printf("Data size: %d elements (%.1f MB per column)\n", n, bytes / 1e6);
    printf("Total GPU allocation: ~%.1f MB (3 input columns + results)\n\n", 4.0 * bytes / 1e6);

    // Generate data
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    double* h_a = (double*)malloc(bytes);
    double* h_b = (double*)malloc(bytes);
    double* h_c = (double*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = normal(rng);
        h_b[i] = normal(rng);
        h_c[i] = normal(rng);
    }

    // Allocate GPU memory
    double *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice));

    // Run benchmarks
    benchmark_fma_fused(n, d_a, d_b, d_c);
    benchmark_complex_expr(n, d_a, d_b, d_c);
    benchmark_where_expr(n, d_a, d_b, d_c);
    benchmark_fused_reduce(n, d_a, d_b, d_c);
    benchmark_chained_ops(n, d_a, d_b, d_c);
    benchmark_savings_analysis(n);

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("=============================================================\n");
    printf("All GPU memory freed. Experiment 010 complete.\n");
    return 0;
}
