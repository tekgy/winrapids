"""
Test: Can GPU read a memory-mapped file directly on Windows?
Step by step, isolating each point of failure.
"""
import numpy as np
import ctypes
import ctypes.wintypes
import os
import sys

n = 100_000  # smaller for safety
data = np.arange(n, dtype=np.float64)
test_path = os.path.abspath("R:/winrapids/experiments/test_mmap.bin")
data.tofile(test_path)
file_size = n * 8
print(f"Step 0: Wrote {file_size/1e6:.1f} MB test file")

kernel32 = ctypes.windll.kernel32

# Step 1: Open file
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
hFile = kernel32.CreateFileW(test_path, GENERIC_READ | GENERIC_WRITE, 1, None, OPEN_EXISTING, 0x80, None)
if hFile == -1:
    print(f"FAIL: CreateFileW error {ctypes.get_last_error()}")
    sys.exit(1)
print(f"Step 1: CreateFileW OK (handle={hFile})")

# Step 2: Create mapping
PAGE_READWRITE = 0x04
hMap = kernel32.CreateFileMappingW(hFile, None, PAGE_READWRITE, 0, 0, None)
if hMap == 0:
    print(f"FAIL: CreateFileMappingW error {ctypes.get_last_error()}")
    sys.exit(1)
print(f"Step 2: CreateFileMappingW OK (handle={hMap})")

# Step 3: Map view
FILE_MAP_ALL_ACCESS = 0xF001F
pBuf = kernel32.MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 0)
if pBuf == 0:
    print(f"FAIL: MapViewOfFile error {ctypes.get_last_error()}")
    sys.exit(1)
print(f"Step 3: MapViewOfFile OK (addr=0x{pBuf:x})")

# Step 4: Read via numpy (zero copy from mapped memory)
mapped = np.ctypeslib.as_array((ctypes.c_double * n).from_address(pBuf), shape=(n,))
assert mapped[0] == 0.0 and mapped[1] == 1.0 and mapped[n-1] == n-1
print(f"Step 4: Numpy read from mmap OK (first={mapped[0]}, last={mapped[n-1]})")

# Step 5: VirtualLock (pin the pages)
ret = kernel32.VirtualLock(ctypes.c_void_p(pBuf), ctypes.c_size_t(file_size))
if ret == 0:
    err = ctypes.get_last_error()
    print(f"Step 5: VirtualLock FAILED (error {err}) — pages may not be pinnable")
    print("  (This is expected if working set quota is too small)")
    # Try anyway with cudaHostRegister
else:
    print(f"Step 5: VirtualLock OK (pages pinned in RAM)")

# Step 6: Try cudaHostRegister
print(f"Step 6: Attempting cudaHostRegister...")
try:
    import cupy as cp

    # Initialize CUDA context first
    _ = cp.zeros(1)
    cp.cuda.Device(0).synchronize()
    print(f"  CUDA context initialized")

    # Try cudaHostRegister via CuPy's low-level API
    # cudaHostRegisterMapped = 2, cudaHostRegisterPortable = 1
    try:
        from cupy.cuda import runtime
        ret = runtime.hostRegister(pBuf, file_size, 2)  # 2 = cudaHostRegisterMapped
        print(f"  cudaHostRegister returned: {ret}")

        # Get device pointer
        dev_ptr = runtime.hostGetDevicePointer(pBuf, 0)
        print(f"  Device pointer: 0x{dev_ptr:x}")

        # Create CuPy array from device pointer
        mem = cp.cuda.UnownedMemory(dev_ptr, file_size, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        gpu_array = cp.ndarray(n, dtype=cp.float64, memptr=memptr)

        # Test: GPU sum of file-mapped memory
        import time
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        result = float(cp.sum(gpu_array))
        cp.cuda.Device(0).synchronize()
        t_gpu = (time.perf_counter() - t0) * 1000

        expected = float(np.sum(data))
        err = abs(result - expected) / abs(expected)

        print(f"")
        print(f"  GPU SUM OF FILE-MAPPED MEMORY:")
        print(f"    Result:   {result:.2f}")
        print(f"    Expected: {expected:.2f}")
        print(f"    Error:    {err:.2e}")
        print(f"    Time:     {t_gpu:.3f} ms")

        if err < 1e-10:
            print(f"")
            print(f"  *** THE GPU READ DIRECTLY FROM THE FILE ***")
            print(f"  *** No cudaMemcpy. No H2D. No allocation. ***")

            # Benchmark comparison
            standard = cp.asarray(data)
            cp.cuda.Device(0).synchronize()

            times_std = []
            times_map = []
            for _ in range(100):
                cp.cuda.Device(0).synchronize()
                t0 = time.perf_counter()
                _ = float(cp.sum(standard))
                cp.cuda.Device(0).synchronize()
                times_std.append((time.perf_counter() - t0) * 1000)

            for _ in range(100):
                cp.cuda.Device(0).synchronize()
                t0 = time.perf_counter()
                _ = float(cp.sum(gpu_array))
                cp.cuda.Device(0).synchronize()
                times_map.append((time.perf_counter() - t0) * 1000)

            t_std = np.median(times_std)
            t_map = np.median(times_map)
            print(f"")
            print(f"  VRAM-resident: {t_std:.3f} ms (p50)")
            print(f"  File-mapped:   {t_map:.3f} ms (p50)")
            print(f"  Ratio:         {t_map/t_std:.1f}x slower (PCIe vs VRAM bandwidth)")

        runtime.hostUnregister(pBuf)

    except Exception as e:
        print(f"  cudaHostRegister path failed: {e}")
        import traceback
        traceback.print_exc()

except ImportError:
    print(f"  CuPy not available")

# Cleanup
kernel32.VirtualUnlock(ctypes.c_void_p(pBuf), ctypes.c_size_t(file_size))
kernel32.UnmapViewOfFile(pBuf)
kernel32.CloseHandle(hMap)
kernel32.CloseHandle(hFile)
os.remove(test_path)
print(f"\nCleanup done.")
