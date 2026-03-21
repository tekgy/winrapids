// WinRapids Experiment 002: GPU Memory Management on Windows
//
// Questions:
//   1. What's the cudaMalloc overhead on WDDM?
//   2. Do CUDA memory pools work? How much do they help?
//   3. What's the practical VRAM limit under WDDM?
//   4. Pinned vs pageable transfer performance?
//   5. Can we build a simple pool allocator that beats raw cudaMalloc?
//
// Hardware: RTX PRO 6000 Blackwell (95.6 GB), WDDM, CUDA 13.1

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
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

// Simple kernel to verify memory is usable
__global__ void fill_kernel(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    void begin() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed_us() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count();
    }
    double elapsed_ms() { return elapsed_us() / 1000.0; }
};

struct Stats {
    double min_us, max_us, mean_us, median_us, p99_us;
};

Stats compute_stats(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    int n = (int)times.size();
    return {
        times[0],
        times[n - 1],
        sum / n,
        times[n / 2],
        times[(int)(n * 0.99)]
    };
}

void print_stats(const char* label, Stats s) {
    printf("  %-30s min=%8.1f  median=%8.1f  mean=%8.1f  p99=%8.1f  max=%8.1f us\n",
           label, s.min_us, s.median_us, s.mean_us, s.p99_us, s.max_us);
}

// ============================================================
// Test 1: cudaMalloc/cudaFree latency at various sizes
// ============================================================

void test_malloc_latency() {
    printf("\n=== Test 1: cudaMalloc/cudaFree Latency ===\n\n");

    size_t sizes[] = {
        1024,                    // 1 KB
        64 * 1024,               // 64 KB
        1024 * 1024,             // 1 MB
        16 * 1024 * 1024,        // 16 MB
        64 * 1024 * 1024,        // 64 MB
        256 * 1024 * 1024,       // 256 MB
        (size_t)1024 * 1024 * 1024,  // 1 GB
    };
    const char* size_labels[] = {
        "1 KB", "64 KB", "1 MB", "16 MB", "64 MB", "256 MB", "1 GB"
    };

    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int warmup = 3;
    int trials = 20;
    Timer timer;

    for (int si = 0; si < num_sizes; si++) {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            void* p;
            CUDA_CHECK(cudaMalloc(&p, sizes[si]));
            CUDA_CHECK(cudaFree(p));
        }

        std::vector<double> alloc_times, free_times;

        for (int i = 0; i < trials; i++) {
            void* p;
            timer.begin();
            CUDA_CHECK(cudaMalloc(&p, sizes[si]));
            alloc_times.push_back(timer.elapsed_us());

            timer.begin();
            CUDA_CHECK(cudaFree(p));
            free_times.push_back(timer.elapsed_us());
        }

        char label_alloc[64], label_free[64];
        snprintf(label_alloc, sizeof(label_alloc), "cudaMalloc(%s)", size_labels[si]);
        snprintf(label_free, sizeof(label_free), "cudaFree(%s)", size_labels[si]);
        print_stats(label_alloc, compute_stats(alloc_times));
        print_stats(label_free, compute_stats(free_times));
        printf("\n");
    }
}

// ============================================================
// Test 2: CUDA Memory Pools
// ============================================================

void test_memory_pools() {
    printf("\n=== Test 2: CUDA Memory Pools ===\n\n");

    // Check if memory pools are supported
    int pool_supported = 0;
    cudaError_t err = cudaDeviceGetAttribute(&pool_supported,
                                              cudaDevAttrMemoryPoolsSupported, 0);
    if (err != cudaSuccess || !pool_supported) {
        printf("  Memory pools NOT SUPPORTED on this device/driver.\n");
        return;
    }
    printf("  Memory pools: SUPPORTED\n\n");

    // Get the default pool
    cudaMemPool_t default_pool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&default_pool, 0));

    // Try to set pool thresholds
    uint64_t threshold = UINT64_MAX; // keep all freed memory in pool
    CUDA_CHECK(cudaMemPoolSetAttribute(default_pool,
                                        cudaMemPoolAttrReleaseThreshold,
                                        &threshold));

    size_t sizes[] = {
        1024,
        64 * 1024,
        1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    };
    const char* size_labels[] = {
        "1 KB", "64 KB", "1 MB", "16 MB", "64 MB", "256 MB"
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int warmup = 5;
    int trials = 50;
    Timer timer;

    for (int si = 0; si < num_sizes; si++) {
        // Warmup with pool
        for (int i = 0; i < warmup; i++) {
            void* p;
            CUDA_CHECK(cudaMallocAsync(&p, sizes[si], 0));
            CUDA_CHECK(cudaFreeAsync(p, 0));
            CUDA_CHECK(cudaStreamSynchronize(0));
        }

        std::vector<double> alloc_times, free_times;

        for (int i = 0; i < trials; i++) {
            void* p;

            CUDA_CHECK(cudaStreamSynchronize(0));
            timer.begin();
            CUDA_CHECK(cudaMallocAsync(&p, sizes[si], 0));
            CUDA_CHECK(cudaStreamSynchronize(0));
            alloc_times.push_back(timer.elapsed_us());

            timer.begin();
            CUDA_CHECK(cudaFreeAsync(p, 0));
            CUDA_CHECK(cudaStreamSynchronize(0));
            free_times.push_back(timer.elapsed_us());
        }

        char label_alloc[64], label_free[64];
        snprintf(label_alloc, sizeof(label_alloc), "cudaMallocAsync(%s)", size_labels[si]);
        snprintf(label_free, sizeof(label_free), "cudaFreeAsync(%s)", size_labels[si]);
        print_stats(label_alloc, compute_stats(alloc_times));
        print_stats(label_free, compute_stats(free_times));
        printf("\n");
    }

    // Compare: rapid alloc/free cycle (the real use case)
    printf("  --- Rapid alloc/free cycle (1MB, 100 iterations) ---\n");

    // Raw cudaMalloc
    {
        std::vector<double> times;
        for (int i = 0; i < 5; i++) {
            void* p;
            cudaMalloc(&p, 1024*1024);
            cudaFree(p);
        }
        for (int i = 0; i < 100; i++) {
            void* p;
            timer.begin();
            CUDA_CHECK(cudaMalloc(&p, 1024*1024));
            CUDA_CHECK(cudaFree(p));
            times.push_back(timer.elapsed_us());
        }
        print_stats("cudaMalloc+Free cycle", compute_stats(times));
    }

    // Pool-based
    {
        std::vector<double> times;
        for (int i = 0; i < 5; i++) {
            void* p;
            cudaMallocAsync(&p, 1024*1024, 0);
            cudaFreeAsync(p, 0);
            cudaStreamSynchronize(0);
        }
        for (int i = 0; i < 100; i++) {
            void* p;
            timer.begin();
            CUDA_CHECK(cudaMallocAsync(&p, 1024*1024, 0));
            CUDA_CHECK(cudaFreeAsync(p, 0));
            CUDA_CHECK(cudaStreamSynchronize(0));
            times.push_back(timer.elapsed_us());
        }
        print_stats("cudaMallocAsync+Free cycle", compute_stats(times));
    }
    printf("\n");
}

// ============================================================
// Test 3: Pinned vs Pageable Memory Transfer
// ============================================================

void test_pinned_vs_pageable() {
    printf("\n=== Test 3: Pinned vs Pageable Memory Transfer ===\n\n");

    size_t sizes[] = {
        1024 * 1024,             // 1 MB
        16 * 1024 * 1024,        // 16 MB
        64 * 1024 * 1024,        // 64 MB
        256 * 1024 * 1024,       // 256 MB
    };
    const char* size_labels[] = {"1 MB", "16 MB", "64 MB", "256 MB"};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int trials = 10;
    Timer timer;

    for (int si = 0; si < num_sizes; si++) {
        size_t bytes = sizes[si];

        // Device memory
        void* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, bytes));

        // --- Pageable ---
        void* h_pageable = malloc(bytes);
        memset(h_pageable, 0xAB, bytes);

        // Warmup
        CUDA_CHECK(cudaMemcpy(d_buf, h_pageable, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_pageable, d_buf, bytes, cudaMemcpyDeviceToHost));

        std::vector<double> page_h2d, page_d2h;
        for (int i = 0; i < trials; i++) {
            timer.begin();
            CUDA_CHECK(cudaMemcpy(d_buf, h_pageable, bytes, cudaMemcpyHostToDevice));
            page_h2d.push_back(timer.elapsed_us());

            timer.begin();
            CUDA_CHECK(cudaMemcpy(h_pageable, d_buf, bytes, cudaMemcpyDeviceToHost));
            page_d2h.push_back(timer.elapsed_us());
        }

        // --- Pinned ---
        void* h_pinned;
        CUDA_CHECK(cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault));
        memset(h_pinned, 0xAB, bytes);

        // Warmup
        CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost));

        std::vector<double> pin_h2d, pin_d2h;
        for (int i = 0; i < trials; i++) {
            timer.begin();
            CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
            pin_h2d.push_back(timer.elapsed_us());

            timer.begin();
            CUDA_CHECK(cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost));
            pin_d2h.push_back(timer.elapsed_us());
        }

        printf("  --- %s ---\n", size_labels[si]);

        auto page_h2d_s = compute_stats(page_h2d);
        auto page_d2h_s = compute_stats(page_d2h);
        auto pin_h2d_s = compute_stats(pin_h2d);
        auto pin_d2h_s = compute_stats(pin_d2h);

        printf("  Pageable H2D: median=%8.0f us  (%.2f GB/s)\n",
               page_h2d_s.median_us, bytes / (page_h2d_s.median_us * 1e3));
        printf("  Pinned   H2D: median=%8.0f us  (%.2f GB/s)\n",
               pin_h2d_s.median_us, bytes / (pin_h2d_s.median_us * 1e3));
        printf("  Speedup H2D:  %.2fx\n", page_h2d_s.median_us / pin_h2d_s.median_us);
        printf("\n");
        printf("  Pageable D2H: median=%8.0f us  (%.2f GB/s)\n",
               page_d2h_s.median_us, bytes / (page_d2h_s.median_us * 1e3));
        printf("  Pinned   D2H: median=%8.0f us  (%.2f GB/s)\n",
               pin_d2h_s.median_us, bytes / (pin_d2h_s.median_us * 1e3));
        printf("  Speedup D2H:  %.2fx\n", page_d2h_s.median_us / pin_d2h_s.median_us);
        printf("\n");

        free(h_pageable);
        CUDA_CHECK(cudaFreeHost(h_pinned));
        CUDA_CHECK(cudaFree(d_buf));
    }
}

// ============================================================
// Test 4: VRAM capacity probe
// ============================================================

void test_vram_capacity() {
    printf("\n=== Test 4: VRAM Capacity Under WDDM ===\n\n");

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("  Total VRAM:     %.2f GB\n", total_mem / (1024.0 * 1024 * 1024));
    printf("  Free VRAM:      %.2f GB\n", free_mem / (1024.0 * 1024 * 1024));
    printf("  Used (WDDM+OS): %.2f GB\n\n",
           (total_mem - free_mem) / (1024.0 * 1024 * 1024));

    // Try to allocate in 1GB chunks until we fail
    size_t chunk = (size_t)1024 * 1024 * 1024; // 1 GB
    std::vector<void*> allocations;
    size_t total_allocated = 0;

    printf("  Allocating 1 GB chunks...\n");
    while (true) {
        void* p;
        cudaError_t err = cudaMalloc(&p, chunk);
        if (err != cudaSuccess) {
            // Try half
            if (chunk > 64 * 1024 * 1024) {
                chunk /= 2;
                continue;
            }
            break;
        }
        allocations.push_back(p);
        total_allocated += chunk;

        // Verify we can actually use the memory
        int n = (int)(chunk / sizeof(float));
        fill_kernel<<<(n + 255) / 256, 256>>>((float*)p, 1.0f, n);
        cudaError_t kerr = cudaDeviceSynchronize();
        if (kerr != cudaSuccess) {
            printf("  Kernel failed after %.2f GB allocated: %s\n",
                   total_allocated / (1024.0 * 1024 * 1024),
                   cudaGetErrorString(kerr));
            CUDA_CHECK(cudaFree(p));
            allocations.pop_back();
            total_allocated -= chunk;
            break;
        }

        printf("    Allocated: %.2f GB (chunk: %zu MB)\n",
               total_allocated / (1024.0 * 1024 * 1024), chunk / (1024 * 1024));
    }

    // Report final state
    size_t free_after, total_after;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));
    printf("\n  Total allocated:     %.2f GB\n",
           total_allocated / (1024.0 * 1024 * 1024));
    printf("  Remaining free:      %.2f GB\n",
           free_after / (1024.0 * 1024 * 1024));
    printf("  WDDM reserved:      %.2f GB (total - free_initial - allocated)\n",
           (total_mem - free_mem) / (1024.0 * 1024 * 1024));

    // Free everything
    for (void* p : allocations) {
        CUDA_CHECK(cudaFree(p));
    }
    printf("  All chunks freed.\n\n");
}

// ============================================================
// Test 5: Async memcpy with compute overlap
// ============================================================

__global__ void busy_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; i++) {
            val = val * 1.001f + 0.001f;
        }
        data[idx] = val;
    }
}

void test_async_overlap() {
    printf("\n=== Test 5: Async Transfer + Compute Overlap ===\n\n");

    size_t n = 16 * 1024 * 1024; // 16M floats = 64MB
    size_t bytes = n * sizeof(float);

    // Check copy-compute overlap support
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Async engine count:    %d\n", prop.asyncEngineCount);
    printf("  Concurrent kernels:    %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Can overlap copy+compute: %s\n\n",
           (prop.asyncEngineCount > 0 && prop.concurrentKernels) ? "yes" : "no");

    // Create streams
    cudaStream_t compute_stream, copy_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&copy_stream));

    // Pinned host memory (required for async transfers)
    float *h_data;
    CUDA_CHECK(cudaHostAlloc(&h_data, bytes, cudaHostAllocDefault));
    for (size_t i = 0; i < n; i++) h_data[i] = 1.0f;

    // Two device buffers
    float *d_compute, *d_transfer;
    CUDA_CHECK(cudaMalloc(&d_compute, bytes));
    CUDA_CHECK(cudaMalloc(&d_transfer, bytes));

    // Initialize device memory
    CUDA_CHECK(cudaMemcpy(d_compute, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transfer, h_data, bytes, cudaMemcpyHostToDevice));

    Timer timer;
    int blocks = (int)((n + 255) / 256);

    // Sequential: compute then transfer
    timer.begin();
    busy_kernel<<<blocks, 256, 0, compute_stream>>>(d_compute, (int)n, 100);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_transfer, bytes, cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    double sequential_ms = timer.elapsed_ms();
    printf("  Sequential (compute then copy):  %.3f ms\n", sequential_ms);

    // Overlapped: compute and transfer simultaneously
    timer.begin();
    busy_kernel<<<blocks, 256, 0, compute_stream>>>(d_compute, (int)n, 100);
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_transfer, bytes, cudaMemcpyDeviceToHost, copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    double overlapped_ms = timer.elapsed_ms();
    printf("  Overlapped (compute + copy):     %.3f ms\n", overlapped_ms);
    printf("  Overlap benefit:                 %.1f%%\n",
           (1.0 - overlapped_ms / sequential_ms) * 100.0);

    printf("\n");

    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFree(d_compute));
    CUDA_CHECK(cudaFree(d_transfer));
}

// ============================================================
// Test 6: Simple Pool Allocator Prototype
// ============================================================

struct SimplePool {
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<Block> blocks;
    size_t total_allocated = 0;
    size_t pool_hits = 0;
    size_t pool_misses = 0;

    void* alloc(size_t size) {
        // Look for a free block of the right size
        for (auto& b : blocks) {
            if (!b.in_use && b.size == size) {
                b.in_use = true;
                pool_hits++;
                return b.ptr;
            }
        }

        // No match — allocate new
        void* p;
        cudaError_t err = cudaMalloc(&p, size);
        if (err != cudaSuccess) return nullptr;
        blocks.push_back({p, size, true});
        total_allocated += size;
        pool_misses++;
        return p;
    }

    void free(void* ptr) {
        for (auto& b : blocks) {
            if (b.ptr == ptr) {
                b.in_use = false;
                return;
            }
        }
    }

    void destroy() {
        for (auto& b : blocks) {
            cudaFree(b.ptr);
        }
        blocks.clear();
        total_allocated = 0;
    }

    void print_stats() {
        printf("  Pool stats: %zu hits, %zu misses, %.2f MB retained\n",
               pool_hits, pool_misses, total_allocated / (1024.0 * 1024));
    }
};

void test_pool_allocator() {
    printf("\n=== Test 6: Simple Pool Allocator vs Raw cudaMalloc ===\n\n");

    int cycles = 200;
    size_t alloc_size = 4 * 1024 * 1024; // 4 MB — typical DataFrame column
    Timer timer;

    // Raw cudaMalloc
    {
        // Warmup
        for (int i = 0; i < 5; i++) {
            void* p; cudaMalloc(&p, alloc_size); cudaFree(p);
        }

        std::vector<double> times;
        for (int i = 0; i < cycles; i++) {
            void* p;
            timer.begin();
            CUDA_CHECK(cudaMalloc(&p, alloc_size));
            CUDA_CHECK(cudaFree(p));
            times.push_back(timer.elapsed_us());
        }
        print_stats("raw cudaMalloc+Free", compute_stats(times));
    }

    // Simple pool
    {
        SimplePool pool;

        // Warmup (first alloc primes the pool)
        void* p = pool.alloc(alloc_size);
        pool.free(p);

        std::vector<double> times;
        for (int i = 0; i < cycles; i++) {
            timer.begin();
            void* p = pool.alloc(alloc_size);
            pool.free(p);
            times.push_back(timer.elapsed_us());
        }
        print_stats("simple pool alloc+free", compute_stats(times));
        pool.print_stats();
        pool.destroy();
    }

    // CUDA async pool (if supported)
    {
        int pool_supported = 0;
        cudaDeviceGetAttribute(&pool_supported, cudaDevAttrMemoryPoolsSupported, 0);
        if (pool_supported) {
            // Warmup
            for (int i = 0; i < 5; i++) {
                void* p;
                cudaMallocAsync(&p, alloc_size, 0);
                cudaFreeAsync(p, 0);
                cudaStreamSynchronize(0);
            }

            std::vector<double> times;
            for (int i = 0; i < cycles; i++) {
                void* p;
                timer.begin();
                CUDA_CHECK(cudaMallocAsync(&p, alloc_size, 0));
                CUDA_CHECK(cudaFreeAsync(p, 0));
                CUDA_CHECK(cudaStreamSynchronize(0));
                times.push_back(timer.elapsed_us());
            }
            print_stats("cudaMallocAsync+Free", compute_stats(times));
        }
    }

    printf("\n");
}

// ============================================================

int main() {
    printf("WinRapids Experiment 002: GPU Memory Management on Windows\n");
    printf("============================================================\n");

    // Force context initialization
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    test_malloc_latency();
    test_memory_pools();
    test_pinned_vs_pageable();
#ifndef SKIP_CAPACITY_TEST
    test_vram_capacity();
#else
    printf("\n=== Test 4: VRAM Capacity (SKIPPED) ===\n\n");
    {
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        printf("  Total VRAM: %.2f GB, Free: %.2f GB\n\n",
               total_mem / (1024.0*1024*1024), free_mem / (1024.0*1024*1024));
    }
#endif
    test_async_overlap();
    test_pool_allocator();

    printf("============================================================\n");
    printf("Experiment 002 complete.\n");
    return 0;
}
