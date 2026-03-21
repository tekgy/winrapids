// WinRapids Experiment 009: Dual-Path GPU DataFrame
//
// Raw CUDA C++ implementations of the operations from Experiment 004.
// Same operations, same data sizes. CUDA events for precise kernel timing.
// Compare against CuPy/GpuFrame to find the abstraction cost.
//
// Operations:
//   1. Sum reduction (10M float64)
//   2. Mean (sum + divide)
//   3. Filtered sum: sum(values) where flag == 1 (10M rows)
//   4. Column arithmetic: a * b + c (10M float64)
//   5. Min/Max reduction
//
// Memory: uses cudaMallocAsync pool, stays well under 60 GB.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
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
// Kernels
// ============================================================

// Warp-level reduction using shuffle
__device__ double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ double warp_reduce_min(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmin(val, other);
    }
    return val;
}

__device__ double warp_reduce_max(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmax(val, other);
    }
    return val;
}

// Block-level sum reduction with warp shuffle
__global__ void reduce_sum_f64(const double* __restrict__ input,
                                double* __restrict__ output, int n) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double val = 0.0;
    if (i < n) val += input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];

    // Warp-level reduction first
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) sdata[warp_id] = val;
    __syncthreads();

    // Final reduction across warps (only first warp)
    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        val = (lane < num_warps) ? sdata[lane] : 0.0;
        val = warp_reduce_sum(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

// Filtered sum: sum(values) where flags == 1
__global__ void filtered_sum_f64(const double* __restrict__ values,
                                  const signed char* __restrict__ flags,
                                  double* __restrict__ output, int n) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double val = 0.0;
    if (i < n && flags[i] == 1) val += values[i];
    if (i + blockDim.x < n && flags[i + blockDim.x] == 1) val += values[i + blockDim.x];

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

// Fused multiply-add: result = a * b + c
__global__ void fma_f64(const double* __restrict__ a,
                         const double* __restrict__ b,
                         const double* __restrict__ c,
                         double* __restrict__ result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fma(a[idx], b[idx], c[idx]);
    }
}

// Vectorized fma using double2 for better memory coalescing
__global__ void fma_f64_vec2(const double2* __restrict__ a,
                              const double2* __restrict__ b,
                              const double2* __restrict__ c,
                              double2* __restrict__ result, int n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        double2 va = a[idx];
        double2 vb = b[idx];
        double2 vc = c[idx];
        result[idx] = make_double2(fma(va.x, vb.x, vc.x),
                                    fma(va.y, vb.y, vc.y));
    }
}

// Min reduction
__global__ void reduce_min_f64(const double* __restrict__ input,
                                double* __restrict__ output, int n) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double val = 1e308;
    if (i < n) val = fmin(val, input[i]);
    if (i + blockDim.x < n) val = fmin(val, input[i + blockDim.x]);

    val = warp_reduce_min(val);

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) sdata[warp_id] = val;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        val = (lane < num_warps) ? sdata[lane] : 1e308;
        val = warp_reduce_min(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

// Max reduction
__global__ void reduce_max_f64(const double* __restrict__ input,
                                double* __restrict__ output, int n) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double val = -1e308;
    if (i < n) val = fmax(val, input[i]);
    if (i + blockDim.x < n) val = fmax(val, input[i + blockDim.x]);

    val = warp_reduce_max(val);

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) sdata[warp_id] = val;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        val = (lane < num_warps) ? sdata[lane] : -1e308;
        val = warp_reduce_max(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

// ============================================================
// Host-side helpers
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

// Complete reduction on host (sum partial results)
double host_reduce_sum(const double* d_partial, int nblocks) {
    double* h_partial = (double*)malloc(nblocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, nblocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < nblocks; i++) sum += h_partial[i];
    free(h_partial);
    return sum;
}

double host_reduce_min(const double* d_partial, int nblocks) {
    double* h_partial = (double*)malloc(nblocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, nblocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double result = h_partial[0];
    for (int i = 1; i < nblocks; i++) result = fmin(result, h_partial[i]);
    free(h_partial);
    return result;
}

double host_reduce_max(const double* d_partial, int nblocks) {
    double* h_partial = (double*)malloc(nblocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, nblocks * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double result = h_partial[0];
    for (int i = 1; i < nblocks; i++) result = fmax(result, h_partial[i]);
    free(h_partial);
    return result;
}

// ============================================================
// Benchmarks
// ============================================================

void benchmark_sum(int n, double* d_data) {
    printf("=== Sum Reduction (n=%d) ===\n", n);

    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    int smem = (threads / 32) * sizeof(double);

    double* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));

    CudaTimer timer;

    // Warmup
    reduce_sum_f64<<<blocks, threads, smem>>>(d_data, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    float total_ms = 0;
    int runs = 20;
    double result = 0;

    for (int i = 0; i < runs; i++) {
        timer.begin();
        reduce_sum_f64<<<blocks, threads, smem>>>(d_data, d_partial, n);
        float ms = timer.elapsed_ms();
        total_ms += ms;
        result = host_reduce_sum(d_partial, blocks);
    }

    float avg_ms = total_ms / runs;
    double bw = (double)n * sizeof(double) / (avg_ms * 1e6);
    printf("  Kernel time: %.4f ms (%.1f GB/s)\n", avg_ms, bw);
    printf("  Result:      %.6f\n\n", result);

    CUDA_CHECK(cudaFree(d_partial));
}

void benchmark_filtered_sum(int n, double* d_values, signed char* d_flags) {
    printf("=== Filtered Sum (n=%d) ===\n", n);

    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    int smem = (threads / 32) * sizeof(double);

    double* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));

    CudaTimer timer;

    // Warmup
    filtered_sum_f64<<<blocks, threads, smem>>>(d_values, d_flags, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    float total_ms = 0;
    int runs = 20;
    double result = 0;

    for (int i = 0; i < runs; i++) {
        timer.begin();
        filtered_sum_f64<<<blocks, threads, smem>>>(d_values, d_flags, d_partial, n);
        float ms = timer.elapsed_ms();
        total_ms += ms;
        result = host_reduce_sum(d_partial, blocks);
    }

    float avg_ms = total_ms / runs;
    double bw = ((double)n * sizeof(double) + (double)n * sizeof(signed char)) / (avg_ms * 1e6);
    printf("  Kernel time: %.4f ms (%.1f GB/s effective)\n", avg_ms, bw);
    printf("  Result:      %.6f\n\n", result);

    CUDA_CHECK(cudaFree(d_partial));
}

void benchmark_fma(int n, double* d_a, double* d_b, double* d_c) {
    printf("=== Column Arithmetic: a*b+c (n=%d) ===\n", n);

    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(double)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CudaTimer timer;

    // Warmup
    fma_f64<<<blocks, threads>>>(d_a, d_b, d_c, d_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Scalar version
    float total_ms = 0;
    int runs = 20;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        fma_f64<<<blocks, threads>>>(d_a, d_b, d_c, d_result, n);
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / runs;
    // 4 arrays * n * 8 bytes (3 read + 1 write)
    double bw = 4.0 * n * sizeof(double) / (avg_ms * 1e6);
    printf("  Scalar FMA:    %.4f ms (%.1f GB/s)\n", avg_ms, bw);

    // Vectorized version (double2)
    int n2 = n / 2;
    int blocks2 = (n2 + threads - 1) / threads;

    total_ms = 0;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        fma_f64_vec2<<<blocks2, threads>>>(
            (double2*)d_a, (double2*)d_b, (double2*)d_c, (double2*)d_result, n2);
        total_ms += timer.elapsed_ms();
    }
    float avg_ms_vec = total_ms / runs;
    double bw_vec = 4.0 * n * sizeof(double) / (avg_ms_vec * 1e6);
    printf("  Vector FMA:    %.4f ms (%.1f GB/s)\n", avg_ms_vec, bw_vec);

    // Verify
    double h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    printf("  First result:  %.6f\n\n", h_result);

    CUDA_CHECK(cudaFree(d_result));
}

void benchmark_minmax(int n, double* d_data) {
    printf("=== Min/Max Reduction (n=%d) ===\n", n);

    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    int smem = (threads / 32) * sizeof(double);

    double* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));

    CudaTimer timer;

    // Min
    reduce_min_f64<<<blocks, threads, smem>>>(d_data, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    int runs = 20;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        reduce_min_f64<<<blocks, threads, smem>>>(d_data, d_partial, n);
        total_ms += timer.elapsed_ms();
    }
    double min_val = host_reduce_min(d_partial, blocks);
    float min_ms = total_ms / runs;

    // Max
    reduce_max_f64<<<blocks, threads, smem>>>(d_data, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    total_ms = 0;
    for (int i = 0; i < runs; i++) {
        timer.begin();
        reduce_max_f64<<<blocks, threads, smem>>>(d_data, d_partial, n);
        total_ms += timer.elapsed_ms();
    }
    double max_val = host_reduce_max(d_partial, blocks);
    float max_ms = total_ms / runs;

    double bw_min = (double)n * sizeof(double) / (min_ms * 1e6);
    double bw_max = (double)n * sizeof(double) / (max_ms * 1e6);

    printf("  Min: %.4f ms (%.1f GB/s) = %.6f\n", min_ms, bw_min, min_val);
    printf("  Max: %.4f ms (%.1f GB/s) = %.6f\n\n", max_ms, bw_max, max_val);

    CUDA_CHECK(cudaFree(d_partial));
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("WinRapids Experiment 009: Dual-Path GPU DataFrame (CUDA C++)\n");
    printf("=============================================================\n\n");

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));  // Force context init

    int n = 10000000;  // 10M elements, same as Experiment 004
    size_t bytes_f64 = n * sizeof(double);
    size_t bytes_i8 = n * sizeof(signed char);

    printf("Data size: %d elements (%.1f MB per f64 column)\n\n", n, bytes_f64 / 1e6);

    // Generate data on host
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_int_distribution<int> coin(0, 1);

    double* h_values = (double*)malloc(bytes_f64);
    double* h_a = (double*)malloc(bytes_f64);
    double* h_b = (double*)malloc(bytes_f64);
    double* h_c = (double*)malloc(bytes_f64);
    signed char* h_flags = (signed char*)malloc(bytes_i8);

    for (int i = 0; i < n; i++) {
        h_values[i] = normal(rng);
        h_a[i] = normal(rng);
        h_b[i] = normal(rng);
        h_c[i] = normal(rng);
        h_flags[i] = (signed char)coin(rng);
    }

    // Allocate device memory (total: ~5 * 80 MB + 10 MB = ~410 MB, well under 60 GB limit)
    double *d_values, *d_a, *d_b, *d_c;
    signed char* d_flags;
    CUDA_CHECK(cudaMalloc(&d_values, bytes_f64));
    CUDA_CHECK(cudaMalloc(&d_a, bytes_f64));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_f64));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_f64));
    CUDA_CHECK(cudaMalloc(&d_flags, bytes_i8));

    // Transfer
    CUDA_CHECK(cudaMemcpy(d_values, h_values, bytes_f64, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_f64, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_f64, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, bytes_f64, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flags, h_flags, bytes_i8, cudaMemcpyHostToDevice));

    // Run benchmarks
    benchmark_sum(n, d_values);
    benchmark_minmax(n, d_values);
    benchmark_filtered_sum(n, d_values, d_flags);
    benchmark_fma(n, d_a, d_b, d_c);

    // Cleanup - ALWAYS free GPU memory
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_flags));
    free(h_values);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_flags);

    printf("=============================================================\n");
    printf("All GPU memory freed. Experiment 009 complete.\n");
    return 0;
}
