// WinRapids Experiment 001: Prove CUDA Works on Windows
//
// Tests: compilation, device query, kernel launch, memory transfer,
//        vector add, parallel reduction, error handling.
// Hardware: RTX PRO 6000 Blackwell, CUDA 13.1, Windows 11

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// ---- Error handling ----

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ---- Kernels ----

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Two-pass parallel reduction: first pass reduces within blocks,
// second pass reduces block results.
__global__ void reduce_sum(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread into shared memory
    float val = 0.0f;
    if (i < n) val += input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ---- Timer utility ----

struct Timer {
    std::chrono::high_resolution_clock::time_point start;

    void begin() { start = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// ---- Device query ----

void print_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("=== CUDA Device Query ===\n");
    printf("CUDA devices found: %d\n\n", device_count);

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability:    %d.%d\n", prop.major, prop.minor);
        printf("  Global memory:         %.1f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  SM count:              %d\n", prop.multiProcessorCount);
        printf("  Max threads/block:     %d\n", prop.maxThreadsPerBlock);
        printf("  Max block dims:        (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dims:         (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Warp size:             %d\n", prop.warpSize);
        printf("  Memory bus width:      %d bits\n", prop.memoryBusWidth);
        printf("  L2 cache size:         %d KB\n", prop.l2CacheSize / 1024);
        printf("  Shared mem/block:      %zu KB\n", prop.sharedMemPerBlock / 1024);
        // Note: memoryClockRate and clockRate removed from cudaDeviceProp in CUDA 13.
        // Use cudaDeviceGetAttribute for clock info instead.
        int mem_clock = 0, gpu_clock = 0;
        cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, i);
        cudaDeviceGetAttribute(&gpu_clock, cudaDevAttrClockRate, i);
        if (mem_clock > 0)
            printf("  Memory clock rate:     %.0f MHz\n", mem_clock / 1000.0);
        if (gpu_clock > 0)
            printf("  Clock rate:            %.0f MHz\n", gpu_clock / 1000.0);
        printf("  Unified addressing:    %s\n", prop.unifiedAddressing ? "yes" : "no");
        printf("  Managed memory:        %s\n", prop.managedMemory ? "yes" : "no");
        printf("  Concurrent kernels:    %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Cooperative launch:    %s\n", prop.cooperativeLaunch ? "yes" : "no");

        // WDDM vs TCC - important for Windows
        printf("  TCC driver:            %s\n", prop.tccDriver ? "yes" : "no (WDDM)");
        printf("  Compute preemption:    %s\n",
               prop.computePreemptionSupported ? "yes" : "no");
        printf("\n");
    }
}

// ---- Vector add test ----

bool test_vector_add(int n) {
    printf("=== Vector Add (n=%d) ===\n", n);
    Timer timer;

    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(n - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy to device
    timer.begin();
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    double h2d_ms = timer.elapsed_ms();
    printf("  H2D transfer:  %.3f ms (%.2f GB/s)\n",
           h2d_ms, (2.0 * bytes) / (h2d_ms * 1e6));

    // Launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Warm up
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    timer.begin();
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    double kernel_ms = timer.elapsed_ms();
    printf("  Kernel time:   %.3f ms (%.2f GB/s effective)\n",
           kernel_ms, (3.0 * bytes) / (kernel_ms * 1e6));

    // Copy back
    timer.begin();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    double d2h_ms = timer.elapsed_ms();
    printf("  D2H transfer:  %.3f ms (%.2f GB/s)\n",
           d2h_ms, bytes / (d2h_ms * 1e6));

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++) {
        float expected = (float)n;
        if (fabsf(h_c[i] - expected) > 1e-5f) {
            printf("  MISMATCH at %d: got %f, expected %f\n", i, h_c[i], expected);
            correct = false;
            break;
        }
    }
    printf("  Result: %s\n\n", correct ? "PASS" : "FAIL");

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return correct;
}

// ---- Reduction test ----

bool test_reduction(int n) {
    printf("=== Parallel Reduction (n=%d) ===\n", n);
    Timer timer;

    size_t bytes = n * sizeof(float);

    // Host data: all ones, so sum = n
    float* h_input = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }

    // Device memory
    float *d_input, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));

    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Warm up
    reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    timer.begin();
    reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    double kernel_ms = timer.elapsed_ms();

    // Finish reduction on host (sum partial results)
    float* h_partial = (float*)malloc(blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float gpu_sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        gpu_sum += h_partial[i];
    }

    float expected = (float)n;
    bool correct = (fabsf(gpu_sum - expected) < 1.0f); // allow small fp error

    printf("  Kernel time:   %.3f ms\n", kernel_ms);
    printf("  GPU sum:       %.0f (expected %.0f)\n", gpu_sum, expected);
    printf("  Bandwidth:     %.2f GB/s\n", bytes / (kernel_ms * 1e6));
    printf("  Result: %s\n\n", correct ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_input);
    free(h_partial);

    return correct;
}

// ---- Managed memory test (important for WDDM) ----

bool test_managed_memory(int n) {
    printf("=== Managed Memory / Unified Addressing (n=%d) ===\n", n);
    Timer timer;

    float *data;
    size_t bytes = n * sizeof(float);

    timer.begin();
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    double alloc_ms = timer.elapsed_ms();
    printf("  Managed alloc: %.3f ms\n", alloc_ms);

    // Initialize on host
    for (int i = 0; i < n; i++) {
        data[i] = 1.0f;
    }

    // Prefetch to device (may not be supported on WDDM - that's interesting to test)
    // CUDA 13 changed cudaMemPrefetchAsync signature: takes cudaMemLocation instead of int device
    timer.begin();
    cudaMemLocation location = {};
    location.type = cudaMemLocationTypeDevice;
    location.id = 0;
    cudaError_t prefetch_err = cudaMemPrefetchAsync(data, bytes, location, 0);
    if (prefetch_err == cudaSuccess) {
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  Prefetch:      %.3f ms (supported)\n", timer.elapsed_ms());
    } else {
        printf("  Prefetch:      NOT SUPPORTED (%s)\n",
               cudaGetErrorString(prefetch_err));
        // Clear the error
        cudaGetLastError();
    }

    // Run reduction on managed memory
    int threads = 256;
    int blocks_needed = (n + threads * 2 - 1) / (threads * 2);
    float *d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks_needed * sizeof(float)));

    timer.begin();
    reduce_sum<<<blocks_needed, threads, threads * sizeof(float)>>>(data, d_partial, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    double kernel_ms = timer.elapsed_ms();

    float* h_partial = (float*)malloc(blocks_needed * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocks_needed * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < blocks_needed; i++) sum += h_partial[i];

    bool correct = (fabsf(sum - (float)n) < 1.0f);
    printf("  Kernel time:   %.3f ms\n", kernel_ms);
    printf("  Sum:           %.0f (expected %d)\n", sum, n);
    printf("  Result: %s\n\n", correct ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(data));
    free(h_partial);

    return correct;
}

// ---- Main ----

int main() {
    printf("WinRapids Experiment 001: CUDA Proof of Life\n");
    printf("=============================================\n\n");

    // Query devices
    print_device_info();

    // CUDA runtime version
    int runtime_version = 0, driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    printf("=== CUDA Versions ===\n");
    printf("  Runtime: %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
    printf("  Driver:  %d.%d\n\n", driver_version / 1000, (driver_version % 1000) / 10);

    // Run tests at increasing scale
    bool all_pass = true;

    // Small: correctness check
    all_pass &= test_vector_add(1024);

    // Medium: 1M elements
    all_pass &= test_vector_add(1 << 20);

    // Large: 64M elements (~256MB per array)
    all_pass &= test_vector_add(1 << 26);

    // Reduction tests
    all_pass &= test_reduction(1 << 20);
    all_pass &= test_reduction(1 << 26);

    // Managed memory test
    all_pass &= test_managed_memory(1 << 20);

    printf("=============================================\n");
    printf("Overall: %s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return all_pass ? 0 : 1;
}
