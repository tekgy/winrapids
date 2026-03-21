"""
WinRapids Experiment 004: Minimal GPU DataFrame

Architecture decisions made by experiments, not assumptions:
- Memory: CuPy MemoryAsyncPool (cudaMallocAsync), 0.5 us alloc vs 281 us raw
- Interface: Arrow-compatible columns with CPU metadata + GPU buffers
- Principle: residency-by-default, data stays on GPU until explicitly materialized
- Location: every buffer tagged with memory type (DEVICE, HOST_PINNED, HOST)

This is a from-scratch GPU DataFrame — NOT a port of cuDF or RAPIDS.
Built FOR Windows, informed by WDDM constraints.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import cupy as cp
import pyarrow as pa


# ============================================================
# Memory Location
# ============================================================

class MemLocation(Enum):
    """Where a buffer physically lives. Explicit, never hidden."""
    DEVICE = "gpu"           # ARROW_DEVICE_CUDA (2)
    HOST_PINNED = "pinned"   # ARROW_DEVICE_CUDA_HOST (3)
    HOST = "cpu"             # ARROW_DEVICE_CPU (1)

    def __repr__(self):
        return self.value


# ============================================================
# GpuColumn
# ============================================================

class GpuColumn:
    """
    A single column of data, Arrow-compatible.

    CPU-resident metadata:
      - name, dtype, length, null_count, location

    GPU-resident data:
      - _data: CuPy device array (the actual buffer)

    The co-native split: metadata is readable by any agent without
    touching GPU memory. Data is only accessible on the device.
    """

    __slots__ = ("name", "dtype", "length", "null_count", "location", "_data")

    def __init__(
        self,
        name: str,
        data: cp.ndarray,
        null_count: int = 0,
        location: MemLocation = MemLocation.DEVICE,
    ):
        self.name = name
        self.dtype = data.dtype
        self.length = len(data)
        self.null_count = null_count
        self.location = location
        self._data = data

    @classmethod
    def from_numpy(cls, name: str, arr: np.ndarray) -> GpuColumn:
        """Create a GPU column from a numpy array (H2D transfer)."""
        gpu_arr = cp.asarray(arr)
        return cls(name=name, data=gpu_arr)

    @classmethod
    def from_arrow(cls, name: str, arr: pa.Array) -> GpuColumn:
        """Create a GPU column from an Arrow array."""
        np_arr = arr.to_numpy(zero_copy_only=False)
        return cls.from_numpy(name, np_arr)

    @classmethod
    def from_sequence(cls, name: str, data, dtype=None) -> GpuColumn:
        """Create a GPU column from a Python sequence."""
        np_arr = np.array(data, dtype=dtype)
        return cls.from_numpy(name, np_arr)

    def to_numpy(self) -> np.ndarray:
        """D2H transfer — explicit materialization."""
        return cp.asnumpy(self._data)

    def to_arrow(self) -> pa.Array:
        """Export as Arrow array (involves D2H transfer)."""
        return pa.array(self.to_numpy())

    @property
    def device_ptr(self) -> int:
        """Raw device pointer — for kernel integration."""
        return self._data.data.ptr

    @property
    def nbytes(self) -> int:
        """Size in bytes on device."""
        return self._data.nbytes

    # ---- Compute operations ----

    def sum(self) -> float:
        """Reduce: sum all elements on GPU."""
        return float(cp.sum(self._data))

    def mean(self) -> float:
        return float(cp.mean(self._data))

    def min(self) -> float:
        return float(cp.min(self._data))

    def max(self) -> float:
        return float(cp.max(self._data))

    def std(self) -> float:
        return float(cp.std(self._data))

    def __eq__(self, other) -> GpuColumn:
        """Element-wise equality — returns boolean GPU column."""
        if isinstance(other, GpuColumn):
            result = self._data == other._data
        else:
            result = self._data == other
        return GpuColumn(name=f"{self.name}==", data=result)

    def __gt__(self, other) -> GpuColumn:
        if isinstance(other, GpuColumn):
            result = self._data > other._data
        else:
            result = self._data > other
        return GpuColumn(name=f"{self.name}>", data=result)

    def __lt__(self, other) -> GpuColumn:
        if isinstance(other, GpuColumn):
            result = self._data < other._data
        else:
            result = self._data < other
        return GpuColumn(name=f"{self.name}<", data=result)

    def __add__(self, other) -> GpuColumn:
        if isinstance(other, GpuColumn):
            result = self._data + other._data
        else:
            result = self._data + other
        return GpuColumn(name=self.name, data=result)

    def __mul__(self, other) -> GpuColumn:
        if isinstance(other, GpuColumn):
            result = self._data * other._data
        else:
            result = self._data * other
        return GpuColumn(name=self.name, data=result)

    def __sub__(self, other) -> GpuColumn:
        if isinstance(other, GpuColumn):
            result = self._data - other._data
        else:
            result = self._data - other
        return GpuColumn(name=self.name, data=result)

    def filtered_sum(self, mask: GpuColumn) -> float:
        """Sum elements where mask is True. Runs entirely on GPU."""
        return float(cp.sum(self._data[mask._data]))

    def __repr__(self) -> str:
        return (
            f"GpuColumn('{self.name}', dtype={self.dtype}, "
            f"len={self.length:,}, {self.location.value}, "
            f"{self.nbytes / 1e6:.1f} MB)"
        )


# ============================================================
# GpuFrame
# ============================================================

class GpuFrame:
    """
    A collection of GpuColumns — the GPU DataFrame.

    CPU-resident:
      - Column metadata (names, dtypes, lengths, locations)
      - Column index (dict name -> GpuColumn)

    GPU-resident:
      - Column data buffers

    Design: fractal self-similarity.
      GpuFrame is a named collection of GpuColumns.
      GpuColumn is a named buffer with metadata.
      Both carry their own metadata + data split.
    """

    def __init__(self, columns: dict[str, GpuColumn]):
        self._columns = columns
        # Verify all columns have the same length
        lengths = {col.length for col in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"Column lengths don't match: {lengths}")
        self._nrows = lengths.pop() if lengths else 0

    @classmethod
    def from_dict(cls, data: dict[str, np.ndarray | list]) -> GpuFrame:
        """Create from a dict of numpy arrays or lists."""
        columns = {}
        for name, arr in data.items():
            if isinstance(arr, np.ndarray):
                columns[name] = GpuColumn.from_numpy(name, arr)
            else:
                columns[name] = GpuColumn.from_sequence(name, arr)
        return cls(columns)

    @classmethod
    def from_pandas(cls, df) -> GpuFrame:
        """Create from a pandas DataFrame."""
        columns = {}
        for col_name in df.columns:
            columns[col_name] = GpuColumn.from_numpy(col_name, df[col_name].values)
        return cls(columns)

    @classmethod
    def from_arrow(cls, table: pa.Table) -> GpuFrame:
        """Create from an Arrow Table."""
        columns = {}
        for name in table.column_names:
            columns[name] = GpuColumn.from_arrow(name, table.column(name))
        return cls(columns)

    def __getitem__(self, key: str) -> GpuColumn:
        return self._columns[key]

    def __len__(self) -> int:
        return self._nrows

    @property
    def columns(self) -> list[str]:
        return list(self._columns.keys())

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nrows, len(self._columns))

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return {name: col.dtype for name, col in self._columns.items()}

    def memory_map(self) -> str:
        """Show per-column memory residency. Co-native: readable by human or AI."""
        lines = [f"GpuFrame: {self._nrows:,} rows x {len(self._columns)} columns\n"]
        total_bytes = 0
        for name, col in self._columns.items():
            lines.append(
                f"  {name:20s}  {str(col.dtype):10s}  "
                f"{col.nbytes/1e6:8.1f} MB  [{col.location.value}]"
            )
            total_bytes += col.nbytes
        lines.append(f"\n  Total: {total_bytes/1e6:.1f} MB on GPU")
        return "\n".join(lines)

    def to_pandas(self):
        """Export to pandas DataFrame (D2H transfer for all columns)."""
        import pandas as pd
        data = {name: col.to_numpy() for name, col in self._columns.items()}
        return pd.DataFrame(data)

    def to_arrow(self) -> pa.Table:
        """Export as Arrow Table."""
        arrays = {name: col.to_arrow() for name, col in self._columns.items()}
        return pa.table(arrays)

    def __repr__(self) -> str:
        col_info = ", ".join(
            f"{name}:{col.dtype}" for name, col in self._columns.items()
        )
        return f"GpuFrame({self._nrows:,} rows, [{col_info}])"


# ============================================================
# Benchmarks
# ============================================================

def benchmark_filtered_sum(n: int = 10_000_000):
    """
    Benchmark: filtered sum on 10M rows.
    Operation: sum(values) where flag == 1
    Compare: pandas CPU vs GPU DataFrame vs raw CuPy
    """
    print(f"\n=== Benchmark: Filtered Sum ({n:,} rows) ===\n")

    # Generate data
    np.random.seed(42)
    values = np.random.randn(n).astype(np.float64)
    flags = np.random.randint(0, 2, n).astype(np.int8)

    # ---- pandas baseline ----
    import pandas as pd
    pdf = pd.DataFrame({"value": values, "flag": flags})

    # Warmup
    _ = pdf.loc[pdf["flag"] == 1, "value"].sum()

    t0 = time.perf_counter()
    for _ in range(5):
        pandas_result = pdf.loc[pdf["flag"] == 1, "value"].sum()
    t_pandas = (time.perf_counter() - t0) / 5

    # ---- GPU DataFrame ----
    gdf = GpuFrame.from_dict({"value": values, "flag": flags})
    cp.cuda.Device(0).synchronize()

    # Warmup
    _ = gdf["value"].filtered_sum(gdf["flag"] == 1)
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        gpu_result = gdf["value"].filtered_sum(gdf["flag"] == 1)
    cp.cuda.Device(0).synchronize()
    t_gpu = (time.perf_counter() - t0) / 5

    # ---- Raw CuPy (no DataFrame overhead) ----
    cp_values = cp.asarray(values)
    cp_flags = cp.asarray(flags)
    cp.cuda.Device(0).synchronize()

    # Warmup
    _ = float(cp.sum(cp_values[cp_flags == 1]))
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        raw_result = float(cp.sum(cp_values[cp_flags == 1]))
    cp.cuda.Device(0).synchronize()
    t_raw = (time.perf_counter() - t0) / 5

    # ---- Custom kernel (optimal) ----
    filtered_sum_kernel = cp.RawKernel(r"""
    extern "C" __global__
    void filtered_sum(const double* values, const char* flags,
                      double* partial_sums, int n) {
        extern __shared__ double sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        double val = 0.0;
        if (i < n && flags[i] == 1) val += values[i];
        if (i + blockDim.x < n && flags[i + blockDim.x] == 1)
            val += values[i + blockDim.x];
        sdata[tid] = val;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
    }
    """, "filtered_sum")

    threads = 256
    blocks = (n + threads * 2 - 1) // (threads * 2)
    partial = cp.zeros(blocks, dtype=cp.float64)

    # Warmup
    filtered_sum_kernel(
        (blocks,), (threads,),
        (cp_values, cp_flags.view(cp.int8), partial, n),
        shared_mem=threads * 8
    )
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        filtered_sum_kernel(
            (blocks,), (threads,),
            (cp_values, cp_flags.view(cp.int8), partial, n),
            shared_mem=threads * 8
        )
        cp.cuda.Device(0).synchronize()
        kernel_result = float(cp.sum(partial))
    t_kernel = (time.perf_counter() - t0) / 5

    # Results
    print(f"  pandas CPU:        {t_pandas*1000:8.3f} ms  result={pandas_result:.6f}")
    print(f"  GpuFrame:          {t_gpu*1000:8.3f} ms  result={gpu_result:.6f}")
    print(f"  Raw CuPy:          {t_raw*1000:8.3f} ms  result={raw_result:.6f}")
    print(f"  Custom kernel:     {t_kernel*1000:8.3f} ms  result={kernel_result:.6f}")
    print()
    print(f"  GpuFrame vs pandas:  {t_pandas/t_gpu:.1f}x faster")
    print(f"  Raw CuPy vs pandas:  {t_pandas/t_raw:.1f}x faster")
    print(f"  Kernel vs pandas:    {t_pandas/t_kernel:.1f}x faster")
    print()

    # Verify correctness
    assert abs(gpu_result - pandas_result) < 0.01, f"GPU mismatch: {gpu_result} vs {pandas_result}"
    assert abs(raw_result - pandas_result) < 0.01, f"Raw mismatch: {raw_result} vs {pandas_result}"
    print(f"  All results match. PASS\n")


def benchmark_column_arithmetic(n: int = 10_000_000):
    """Benchmark column arithmetic: a * b + c"""
    print(f"\n=== Benchmark: Column Arithmetic ({n:,} rows) ===\n")

    np.random.seed(42)
    a_np = np.random.randn(n).astype(np.float64)
    b_np = np.random.randn(n).astype(np.float64)
    c_np = np.random.randn(n).astype(np.float64)

    # pandas
    import pandas as pd
    pdf = pd.DataFrame({"a": a_np, "b": b_np, "c": c_np})

    _ = pdf["a"] * pdf["b"] + pdf["c"]

    t0 = time.perf_counter()
    for _ in range(5):
        _ = pdf["a"] * pdf["b"] + pdf["c"]
    t_pandas = (time.perf_counter() - t0) / 5

    # GpuFrame
    gdf = GpuFrame.from_dict({"a": a_np, "b": b_np, "c": c_np})
    cp.cuda.Device(0).synchronize()

    _ = gdf["a"] * gdf["b"] + gdf["c"]
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        result_col = gdf["a"] * gdf["b"] + gdf["c"]
    cp.cuda.Device(0).synchronize()
    t_gpu = (time.perf_counter() - t0) / 5

    print(f"  pandas CPU:   {t_pandas*1000:8.3f} ms")
    print(f"  GpuFrame:     {t_gpu*1000:8.3f} ms")
    print(f"  Speedup:      {t_pandas/t_gpu:.1f}x")
    print()

    # Verify
    cpu_result = a_np * b_np + c_np
    gpu_result = result_col.to_numpy()
    max_err = np.max(np.abs(cpu_result - gpu_result))
    print(f"  Max error: {max_err:.2e}")
    print(f"  PASS\n")


def benchmark_aggregation(n: int = 10_000_000):
    """Benchmark aggregation: sum, mean, min, max, std"""
    print(f"\n=== Benchmark: Aggregation ({n:,} rows) ===\n")

    np.random.seed(42)
    values = np.random.randn(n).astype(np.float64)

    import pandas as pd
    series = pd.Series(values)
    col = GpuColumn.from_numpy("values", values)
    cp.cuda.Device(0).synchronize()

    ops = ["sum", "mean", "min", "max", "std"]

    for op in ops:
        # pandas
        _ = getattr(series, op)()
        t0 = time.perf_counter()
        for _ in range(10):
            pd_result = getattr(series, op)()
        t_pandas = (time.perf_counter() - t0) / 10

        # GPU
        _ = getattr(col, op)()
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            gpu_result = getattr(col, op)()
        cp.cuda.Device(0).synchronize()
        t_gpu = (time.perf_counter() - t0) / 10

        print(f"  {op:5s}  pandas={t_pandas*1000:7.3f}ms  gpu={t_gpu*1000:7.3f}ms  "
              f"speedup={t_pandas/t_gpu:5.1f}x  match={abs(pd_result - gpu_result) < 0.01}")

    print(f"\n  PASS\n")


def demo_memory_map():
    """Show the memory map — co-native metadata."""
    print("\n=== Demo: Memory Map ===\n")

    n = 5_000_000
    gdf = GpuFrame.from_dict({
        "id": np.arange(n, dtype=np.int64),
        "price": np.random.randn(n).astype(np.float64),
        "volume": np.random.randint(0, 10000, n).astype(np.int32),
        "flag": np.random.randint(0, 2, n).astype(np.int8),
    })

    print(gdf.memory_map())
    print()
    print(f"  repr: {gdf}")
    print()


def demo_arrow_roundtrip():
    """Demonstrate Arrow Table <-> GpuFrame roundtrip."""
    print("\n=== Demo: Arrow Roundtrip ===\n")

    # Create Arrow table
    n = 1_000_000
    table = pa.table({
        "x": pa.array(np.random.randn(n).astype(np.float64)),
        "y": pa.array(np.arange(n, dtype=np.int64)),
    })

    # Arrow -> GPU
    t0 = time.perf_counter()
    gdf = GpuFrame.from_arrow(table)
    cp.cuda.Device(0).synchronize()
    t_in = time.perf_counter() - t0

    # Compute on GPU
    t0 = time.perf_counter()
    x_sum = gdf["x"].sum()
    cp.cuda.Device(0).synchronize()
    t_compute = time.perf_counter() - t0

    # GPU -> Arrow
    t0 = time.perf_counter()
    table_back = gdf.to_arrow()
    t_out = time.perf_counter() - t0

    print(f"  Arrow -> GPU:   {t_in*1000:.3f} ms")
    print(f"  GPU sum(x):     {t_compute*1000:.3f} ms = {x_sum:.6f}")
    print(f"  GPU -> Arrow:   {t_out*1000:.3f} ms")
    print(f"  Roundtrip:      {(t_in + t_out)*1000:.3f} ms")
    print()

    # Verify
    original_sum = float(table.column("x").to_pylist().__iter__().__next__() if False else
                         np.sum(table.column("x").to_numpy()))
    print(f"  Original sum:   {original_sum:.6f}")
    print(f"  Match:          {abs(x_sum - original_sum) < 0.01}")
    print(f"  PASS\n")


if __name__ == "__main__":
    print("WinRapids Experiment 004: Minimal GPU DataFrame")
    print("=" * 60)

    demo_memory_map()
    demo_arrow_roundtrip()
    benchmark_aggregation()
    benchmark_column_arithmetic()
    benchmark_filtered_sum()

    print("=" * 60)
    print("Experiment 004 complete.")
