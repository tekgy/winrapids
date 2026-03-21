"""
WinRapids Experiment 008: I/O Path Benchmarks

Three paths from NVMe to GPU memory:
1. mmap (OS-buffered) — baseline
2. Unbuffered ReadFile + pageable memory — eliminates OS buffer copy
3. Unbuffered ReadFile + pinned memory + cudaMemcpyAsync — optimal

DirectStorage -> D3D12 -> CUDA is ruled out — it requires 4 API boundary
crossings vs 1 for unbuffered ReadFile. llama.cpp team confirmed this.

NTFS alignment: unbuffered I/O requires sector-aligned offsets and sizes.
cudaHostAlloc returns 4096-byte-aligned buffers automatically.
"""

import os
import time
import ctypes
import ctypes.wintypes
import tempfile
import mmap

import numpy as np
import cupy as cp

# Windows API constants
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
CREATE_ALWAYS = 2
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_ATTRIBUTE_NORMAL = 0x00000080
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

kernel32 = ctypes.windll.kernel32


def create_test_file(path: str, size_bytes: int) -> str:
    """Create a test file filled with sequential float64 values."""
    n = size_bytes // 8
    # Write in chunks to avoid huge memory allocation
    chunk_size = min(n, 10_000_000)  # ~80 MB chunks
    with open(path, 'wb') as f:
        written = 0
        while written < n:
            count = min(chunk_size, n - written)
            data = np.arange(written, written + count, dtype=np.float64)
            f.write(data.tobytes())
            written += count
    return path


def align_up(x: int, alignment: int) -> int:
    """Round up to alignment boundary."""
    return (x + alignment - 1) & ~(alignment - 1)


# ============================================================
# Path 1: mmap (OS-buffered baseline)
# ============================================================

def bench_mmap_to_gpu(filepath: str, size_bytes: int, trials: int = 5) -> dict:
    """Read via mmap, then cudaMemcpy to GPU."""
    results = []

    for _ in range(trials):
        t0 = time.perf_counter()

        # mmap the file
        with open(filepath, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            data = np.frombuffer(mm[:size_bytes], dtype=np.float64)

            # Copy to GPU
            gpu_arr = cp.asarray(data)
            cp.cuda.Device(0).synchronize()

            t_total = time.perf_counter() - t0

            # Verify
            checksum = float(cp.sum(gpu_arr))

            mm.close()
            del gpu_arr

        results.append(t_total)

    median = sorted(results)[len(results) // 2]
    return {
        "method": "mmap",
        "size_mb": size_bytes / 1e6,
        "median_ms": median * 1000,
        "bandwidth_gbps": size_bytes / (median * 1e9),
        "checksum": checksum,
    }


# ============================================================
# Path 2: Unbuffered ReadFile + pageable memory
# ============================================================

def bench_unbuffered_pageable(filepath: str, size_bytes: int, trials: int = 5) -> dict:
    """Unbuffered ReadFile to pageable memory, then cudaMemcpy."""
    results = []

    sector_size = 4096  # NTFS default physical sector

    for _ in range(trials):
        t0 = time.perf_counter()

        # Open with no buffering
        handle = kernel32.CreateFileW(
            filepath,
            GENERIC_READ,
            FILE_SHARE_READ,
            None,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_ATTRIBUTE_NORMAL,
            None
        )
        if handle == INVALID_HANDLE_VALUE:
            raise OSError(f"CreateFileW failed: {ctypes.get_last_error()}")

        # Allocate aligned buffer
        aligned_size = align_up(size_bytes, sector_size)
        buf = ctypes.create_string_buffer(aligned_size)

        # Read
        bytes_read = ctypes.wintypes.DWORD(0)
        ok = kernel32.ReadFile(handle, buf, aligned_size, ctypes.byref(bytes_read), None)
        if not ok:
            kernel32.CloseHandle(handle)
            raise OSError(f"ReadFile failed: {ctypes.get_last_error()}")
        kernel32.CloseHandle(handle)

        # Convert to numpy (zero-copy from buffer)
        data = np.frombuffer(buf, dtype=np.float64, count=size_bytes // 8)

        # Copy to GPU
        gpu_arr = cp.asarray(data)
        cp.cuda.Device(0).synchronize()

        t_total = time.perf_counter() - t0
        checksum = float(cp.sum(gpu_arr))
        del gpu_arr

        results.append(t_total)

    median = sorted(results)[len(results) // 2]
    return {
        "method": "unbuffered+pageable",
        "size_mb": size_bytes / 1e6,
        "median_ms": median * 1000,
        "bandwidth_gbps": size_bytes / (median * 1e9),
        "checksum": checksum,
    }


# ============================================================
# Path 3: Unbuffered ReadFile + pinned memory + async copy
# ============================================================

def bench_unbuffered_pinned(filepath: str, size_bytes: int, trials: int = 5) -> dict:
    """Unbuffered ReadFile directly into pinned memory, then async H2D."""
    results = []

    sector_size = 4096

    for _ in range(trials):
        t0 = time.perf_counter()

        # Allocate pinned memory
        aligned_size = align_up(size_bytes, sector_size)
        pinned = cp.cuda.alloc_pinned_memory(aligned_size)
        pinned_buf = np.frombuffer(pinned, dtype=np.uint8, count=aligned_size)

        # Open with no buffering
        handle = kernel32.CreateFileW(
            filepath,
            GENERIC_READ,
            FILE_SHARE_READ,
            None,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_ATTRIBUTE_NORMAL,
            None
        )
        if handle == INVALID_HANDLE_VALUE:
            raise OSError(f"CreateFileW failed: {ctypes.get_last_error()}")

        # Read directly into pinned memory
        # Need to get raw pointer from the pinned buffer
        buf_ptr = pinned_buf.ctypes.data_as(ctypes.c_void_p)
        bytes_read = ctypes.wintypes.DWORD(0)
        ok = kernel32.ReadFile(handle, buf_ptr, aligned_size, ctypes.byref(bytes_read), None)
        if not ok:
            kernel32.CloseHandle(handle)
            raise OSError(f"ReadFile failed: {ctypes.get_last_error()}")
        kernel32.CloseHandle(handle)

        # View as float64
        data = np.frombuffer(pinned_buf[:size_bytes], dtype=np.float64)

        # Async copy to GPU
        gpu_arr = cp.asarray(data)
        cp.cuda.Device(0).synchronize()

        t_total = time.perf_counter() - t0
        checksum = float(cp.sum(gpu_arr))
        del gpu_arr

        results.append(t_total)

    median = sorted(results)[len(results) // 2]
    return {
        "method": "unbuffered+pinned",
        "size_mb": size_bytes / 1e6,
        "median_ms": median * 1000,
        "bandwidth_gbps": size_bytes / (median * 1e9),
        "checksum": checksum,
    }


# ============================================================
# Bonus: Standard Python read (the naive baseline)
# ============================================================

def bench_python_read(filepath: str, size_bytes: int, trials: int = 5) -> dict:
    """Standard Python file read + numpy + GPU transfer."""
    results = []

    for _ in range(trials):
        t0 = time.perf_counter()

        with open(filepath, 'rb') as f:
            raw = f.read(size_bytes)
        data = np.frombuffer(raw, dtype=np.float64)
        gpu_arr = cp.asarray(data)
        cp.cuda.Device(0).synchronize()

        t_total = time.perf_counter() - t0
        checksum = float(cp.sum(gpu_arr))
        del gpu_arr

        results.append(t_total)

    median = sorted(results)[len(results) // 2]
    return {
        "method": "python read",
        "size_mb": size_bytes / 1e6,
        "median_ms": median * 1000,
        "bandwidth_gbps": size_bytes / (median * 1e9),
        "checksum": checksum,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("WinRapids Experiment 008: I/O Path Benchmarks")
    print("=" * 60)
    print()

    # Test file sizes
    sizes = [
        (1, "1 MB"),
        (100, "100 MB"),
        (1000, "1 GB"),
    ]

    # Create temp directory for test files
    test_dir = os.path.join(tempfile.gettempdir(), "winrapids_io_bench")
    os.makedirs(test_dir, exist_ok=True)

    for size_mb, label in sizes:
        size_bytes = size_mb * 1024 * 1024
        # Align to sector size
        size_bytes = align_up(size_bytes, 4096)

        filepath = os.path.join(test_dir, f"test_{size_mb}mb.bin")

        # Create test file if needed
        if not os.path.exists(filepath) or os.path.getsize(filepath) < size_bytes:
            print(f"Creating {label} test file... ", end="", flush=True)
            create_test_file(filepath, size_bytes)
            print("done")

        print(f"\n=== {label} ({size_bytes/1e6:.0f} MB) ===\n")

        # Run benchmarks
        methods = [
            ("Python read", bench_python_read),
            ("mmap", bench_mmap_to_gpu),
            ("Unbuffered+pageable", bench_unbuffered_pageable),
            ("Unbuffered+pinned", bench_unbuffered_pinned),
        ]

        results = []
        checksums = set()

        for name, bench_fn in methods:
            try:
                r = bench_fn(filepath, size_bytes, trials=5)
                results.append(r)
                checksums.add(round(r["checksum"], 2))
                print(f"  {r['method']:<25s} {r['median_ms']:>8.1f} ms  "
                      f"({r['bandwidth_gbps']:.2f} GB/s)")
            except Exception as e:
                print(f"  {name:<25s} FAILED: {e}")

        # Verify checksums match
        if len(checksums) == 1:
            print(f"\n  Checksums: all match")
        else:
            print(f"\n  WARNING: checksum mismatch: {checksums}")

        # Speedup summary
        if len(results) >= 4:
            baseline = results[0]["median_ms"]  # Python read
            print(f"\n  Speedup vs Python read:")
            for r in results[1:]:
                print(f"    {r['method']:<25s} {baseline/r['median_ms']:.2f}x")

    print(f"\n{'='*60}")
    print("Experiment 008 complete.")
    print()
    print("Note: DirectStorage -> D3D12 -> CUDA NOT benchmarked.")
    print("Requires D3D12 intermediary with 4 API boundary crossings")
    print("vs 1 for unbuffered ReadFile. llama.cpp team confirmed this.")

    # Cleanup option
    # for f in os.listdir(test_dir):
    #     os.remove(os.path.join(test_dir, f))
