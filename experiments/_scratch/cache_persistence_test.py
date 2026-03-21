"""Quick test: does CuPy's kernel cache persist across process restarts?

Run this twice in separate Python processes. Compare RawKernel compile time.
If the disk cache works, the second run's compile time should be ~0 ms.
"""
import time
import sys

t0 = time.perf_counter()
import cupy as cp
t_import = time.perf_counter() - t0

# Force CUDA context creation
t1 = time.perf_counter()
cp.cuda.Device(0).use()
_ = cp.zeros(1)
cp.cuda.Device(0).synchronize()
t_ctx = time.perf_counter() - t1

# A unique-ish kernel to test cache behavior
kernel_src = r'''
extern "C" __global__ void cache_test_add(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
'''

t2 = time.perf_counter()
k = cp.RawKernel(kernel_src, 'cache_test_add')
t_compile = time.perf_counter() - t2

# Launch it
n = 1_000_000
a = cp.ones(n, dtype=cp.float64)
b = cp.ones(n, dtype=cp.float64)
c = cp.zeros(n, dtype=cp.float64)

t3 = time.perf_counter()
k((n // 256 + 1,), (256,), (a, b, c, n))
cp.cuda.Device(0).synchronize()
t_launch = time.perf_counter() - t3

total = time.perf_counter() - t0

print(f"CuPy import:         {t_import*1000:>8.1f} ms")
print(f"CUDA context init:   {t_ctx*1000:>8.1f} ms")
print(f"RawKernel compile:   {t_compile*1000:>8.1f} ms")
print(f"First kernel launch: {t_launch*1000:>8.1f} ms")
print(f"Total:               {total*1000:>8.1f} ms")
print(f"\nRun: {sys.argv[1] if len(sys.argv) > 1 else 'unspecified'}")
