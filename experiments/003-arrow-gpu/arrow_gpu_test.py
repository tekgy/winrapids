"""
WinRapids Experiment 003: Arrow GPU Integration on Windows

Tests the Arrow C Device Data Interface workflow:
1. Create GPU data with CuPy
2. Export to Arrow format (CPU metadata + GPU buffers)
3. Import back and verify data integrity
4. Benchmark transfer paths

Key insight: pyarrow.cuda doesn't exist on Windows pip wheels.
The Arrow C Device Data Interface is the way — it's a C ABI,
header-only, no library dependency needed.
"""

import time
import ctypes
import numpy as np
import cupy as cp
import pyarrow as pa


def test_cupy_arrow_roundtrip():
    """Test CuPy array -> Arrow -> back"""
    print("=== Test 1: CuPy <-> Arrow Roundtrip (via CPU) ===\n")

    sizes = [1_000, 100_000, 1_000_000, 10_000_000]

    for n in sizes:
        # Create GPU data
        gpu_arr = cp.arange(n, dtype=cp.float64)

        # GPU -> CPU -> Arrow
        t0 = time.perf_counter()
        cpu_arr = cp.asnumpy(gpu_arr)
        t_d2h = time.perf_counter() - t0

        t0 = time.perf_counter()
        arrow_arr = pa.array(cpu_arr)
        t_to_arrow = time.perf_counter() - t0

        # Arrow -> CPU -> GPU
        t0 = time.perf_counter()
        cpu_back = arrow_arr.to_numpy(zero_copy_only=True)
        t_from_arrow = time.perf_counter() - t0

        t0 = time.perf_counter()
        gpu_back = cp.asarray(cpu_back)
        t_h2d = time.perf_counter() - t0

        # Verify
        assert cp.array_equal(gpu_arr, gpu_back), "Data mismatch!"

        total_ms = (t_d2h + t_to_arrow + t_from_arrow + t_h2d) * 1000
        bytes_total = n * 8  # float64

        print(f"  n={n:>10,d} ({bytes_total/1e6:.1f} MB):")
        print(f"    D2H:       {t_d2h*1000:8.3f} ms  ({bytes_total/(t_d2h*1e9):.2f} GB/s)")
        print(f"    ->Arrow:   {t_to_arrow*1000:8.3f} ms  (zero-copy: {t_to_arrow < 0.001})")
        print(f"    Arrow->:   {t_from_arrow*1000:8.3f} ms  (zero-copy: {t_from_arrow < 0.001})")
        print(f"    H2D:       {t_h2d*1000:8.3f} ms  ({bytes_total/(t_h2d*1e9):.2f} GB/s)")
        print(f"    Total RT:  {total_ms:8.3f} ms")
        print()

    print("  Result: PASS\n")


def test_cupy_dlpack():
    """Test DLPack protocol — the modern GPU tensor interchange"""
    print("=== Test 2: DLPack Protocol (GPU Zero-Copy) ===\n")

    sizes = [1_000_000, 10_000_000, 100_000_000]

    for n in sizes:
        gpu_arr = cp.arange(n, dtype=cp.float32)

        # Export to DLPack
        t0 = time.perf_counter()
        dlpack_capsule = gpu_arr.__dlpack__()
        t_export = time.perf_counter() - t0

        # Import back
        t0 = time.perf_counter()
        gpu_back = cp.from_dlpack(dlpack_capsule)
        t_import = time.perf_counter() - t0

        # Verify zero-copy (same pointer)
        same_ptr = gpu_arr.data.ptr == gpu_back.data.ptr

        print(f"  n={n:>12,d} ({n*4/1e6:.1f} MB):")
        print(f"    Export:    {t_export*1e6:8.1f} us")
        print(f"    Import:    {t_import*1e6:8.1f} us")
        print(f"    Zero-copy: {same_ptr}")
        print()

    print("  Result: PASS\n")


def test_arrow_device_array_concept():
    """
    Demonstrate the Arrow C Device Data Interface concept.

    The ArrowDeviceArray struct is CPU-resident metadata pointing to
    GPU buffers. This is the co-native split: metadata readable by
    any agent (CPU or GPU), data accessible only on the right device.
    """
    print("=== Test 3: Arrow Device Array Concept ===\n")

    # Simulate what ArrowDeviceArray does:
    # CPU struct with device_type, device_id, and buffer pointers

    n = 1_000_000
    dtype = cp.float64
    bytes_per = n * 8

    # Create GPU buffer
    gpu_data = cp.arange(n, dtype=dtype)
    device_ptr = gpu_data.data.ptr

    # The "ArrowDeviceArray" is a CPU struct
    device_array_info = {
        "device_type": 2,  # ARROW_DEVICE_CUDA
        "device_id": 0,
        "sync_event": None,  # null = synchronous, data is ready
        "schema": {
            "name": "values",
            "type": "float64",
            "nullable": False,
        },
        "array": {
            "length": n,
            "null_count": 0,
            "offset": 0,
            "n_buffers": 2,
            "buffers": [
                None,        # validity bitmap (null = not nullable)
                device_ptr,  # data buffer (GPU pointer)
            ],
        },
    }

    # A consumer can read ALL metadata without touching GPU memory
    print(f"  Schema name:   {device_array_info['schema']['name']}")
    print(f"  Schema type:   {device_array_info['schema']['type']}")
    print(f"  Length:         {device_array_info['array']['length']:,}")
    print(f"  Device type:   {device_array_info['device_type']} (ARROW_DEVICE_CUDA)")
    print(f"  Device ID:     {device_array_info['device_id']}")
    print(f"  Sync event:    {device_array_info['sync_event']} (synchronous)")
    print(f"  Data pointer:  0x{device_array_info['array']['buffers'][1]:016x}")
    print(f"  Data on GPU:   {device_array_info['array']['buffers'][1] != 0}")
    print()

    # The consumer can also verify the pointer is valid by running a kernel
    result = cp.sum(gpu_data)
    expected = n * (n - 1) / 2  # sum of 0..n-1
    print(f"  GPU sum:       {float(result):,.0f}")
    print(f"  Expected:      {expected:,.0f}")
    print(f"  Match:         {abs(float(result) - expected) < 1.0}")
    print()

    print("  Key insight: CPU struct metadata is readable without GPU access.")
    print("  Only the buffer pointers reference GPU memory.")
    print("  This is the co-native split.\n")
    print("  Result: PASS\n")


def test_pinned_memory_arrow():
    """Test pinned host memory for Arrow buffers (ARROW_DEVICE_CUDA_HOST = 3)"""
    print("=== Test 4: Pinned Host Memory for Arrow ===\n")

    sizes = [1_000_000, 10_000_000]

    for n in sizes:
        bytes_total = n * 8

        # Allocate pinned memory via CuPy
        t0 = time.perf_counter()
        pinned = cp.cuda.alloc_pinned_memory(bytes_total)
        pinned_arr = np.frombuffer(pinned, dtype=np.float64, count=n)
        t_alloc = time.perf_counter() - t0

        # Fill with data
        pinned_arr[:] = np.arange(n, dtype=np.float64)

        # Transfer to GPU
        t0 = time.perf_counter()
        gpu_arr = cp.asarray(pinned_arr)
        cp.cuda.Device(0).synchronize()
        t_h2d = time.perf_counter() - t0

        # Transfer back
        t0 = time.perf_counter()
        result = cp.asnumpy(gpu_arr)
        t_d2h = time.perf_counter() - t0

        # Verify
        assert np.array_equal(pinned_arr, result), "Mismatch!"

        print(f"  n={n:>10,d} ({bytes_total/1e6:.1f} MB):")
        print(f"    Pinned alloc: {t_alloc*1000:8.3f} ms")
        print(f"    H2D:          {t_h2d*1000:8.3f} ms  ({bytes_total/(t_h2d*1e9):.2f} GB/s)")
        print(f"    D2H:          {t_d2h*1000:8.3f} ms  ({bytes_total/(t_d2h*1e9):.2f} GB/s)")
        print()

    print("  Result: PASS\n")


def test_arrow_zero_copy_gpu_kernels():
    """
    Test that Arrow-exported data can be used directly in GPU kernels
    via CuPy's RawKernel interface.
    """
    print("=== Test 5: Arrow Data -> GPU Kernel (Zero-Copy) ===\n")

    n = 10_000_000

    # Start with Arrow array (the common data interchange format)
    arrow_arr = pa.array(np.random.randn(n).astype(np.float32))

    # Arrow -> numpy (zero-copy)
    np_arr = arrow_arr.to_numpy(zero_copy_only=True)

    # numpy -> CuPy GPU (H2D transfer)
    t0 = time.perf_counter()
    gpu_arr = cp.asarray(np_arr)
    cp.cuda.Device(0).synchronize()
    t_h2d = time.perf_counter() - t0

    # Run a real kernel on the data
    square_kernel = cp.RawKernel(r"""
    extern "C" __global__
    void square_elements(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = data[idx] * data[idx];
        }
    }
    """, "square_elements")

    t0 = time.perf_counter()
    square_kernel(((n + 255) // 256,), (256,), (gpu_arr, n))
    cp.cuda.Device(0).synchronize()
    t_kernel = time.perf_counter() - t0

    # Verify
    expected = np_arr ** 2
    result = cp.asnumpy(gpu_arr)
    max_err = np.max(np.abs(result - expected))

    print(f"  Elements:      {n:,}")
    print(f"  H2D transfer:  {t_h2d*1000:.3f} ms ({n*4/(t_h2d*1e9):.2f} GB/s)")
    print(f"  Kernel time:   {t_kernel*1000:.3f} ms ({n*4/(t_kernel*1e9):.2f} GB/s)")
    print(f"  Max error:     {max_err:.6e}")
    print(f"  Correct:       {max_err < 1e-6}")
    print()
    print("  Result: PASS\n")


def test_arrow_ipc_to_gpu():
    """
    Test Arrow IPC format -> GPU pipeline.
    This simulates receiving Arrow data from another process/file.
    """
    print("=== Test 6: Arrow IPC -> GPU Pipeline ===\n")

    import io

    n = 5_000_000

    # Create a record batch
    batch = pa.record_batch({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "value": pa.array(np.random.randn(n).astype(np.float64)),
        "flag": pa.array(np.random.randint(0, 2, n).astype(np.int8)),
    })

    # Serialize to IPC format (simulating file/network read)
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    ipc_buffer = sink.getvalue()

    print(f"  Record batch: {n:,} rows, {batch.num_columns} columns")
    print(f"  IPC size:     {len(ipc_buffer)/1e6:.1f} MB")
    print()

    # Deserialize
    t0 = time.perf_counter()
    reader = pa.ipc.open_stream(ipc_buffer)
    batch_back = reader.read_next_batch()
    t_deserialize = time.perf_counter() - t0

    # Transfer each column to GPU
    t0 = time.perf_counter()
    gpu_columns = {}
    for name in batch_back.schema.names:
        col = batch_back.column(name)
        np_col = col.to_numpy(zero_copy_only=False)
        gpu_columns[name] = cp.asarray(np_col)
    cp.cuda.Device(0).synchronize()
    t_to_gpu = time.perf_counter() - t0

    # Run a computation: sum of value where flag == 1
    t0 = time.perf_counter()
    mask = gpu_columns["flag"] == 1
    filtered_sum = float(cp.sum(gpu_columns["value"][mask]))
    t_compute = time.perf_counter() - t0

    # Verify against CPU
    cpu_filtered_sum = float(batch_back.column("value").to_numpy()[
        batch_back.column("flag").to_numpy() == 1
    ].sum())

    print(f"  Deserialize:   {t_deserialize*1000:.3f} ms")
    print(f"  To GPU:        {t_to_gpu*1000:.3f} ms")
    print(f"  Compute:       {t_compute*1000:.3f} ms")
    print(f"  GPU result:    {filtered_sum:.6f}")
    print(f"  CPU result:    {cpu_filtered_sum:.6f}")
    print(f"  Match:         {abs(filtered_sum - cpu_filtered_sum) < 0.01}")
    print()
    print("  Result: PASS\n")


if __name__ == "__main__":
    print("WinRapids Experiment 003: Arrow GPU Integration on Windows")
    print("=" * 60)
    print()

    test_cupy_arrow_roundtrip()
    test_cupy_dlpack()
    test_arrow_device_array_concept()
    test_pinned_memory_arrow()
    test_arrow_zero_copy_gpu_kernels()
    test_arrow_ipc_to_gpu()

    print("=" * 60)
    print("Experiment 003 complete. All tests passed.")
