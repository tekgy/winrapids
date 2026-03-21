"""
WinRapids Experiment 005: Pandas GPU Proxy Pattern

Goal: Make existing pandas code run on GPU transparently, with automatic
CPU fallback for unsupported operations.

Approach: Proxy pattern — wrap pandas DataFrames and Series so that
operations dispatch to GPU when possible, fall back to CPU when not.
No import hooks or monkey-patching (yet). Start with explicit wrapping,
then explore transparent activation.

Design principle: the proxy IS a pandas object to external code (duck typing),
but it CONTAINS a GPU representation for accelerated computation.
"""

from __future__ import annotations

import time
import functools
from typing import Optional

import numpy as np
import cupy as cp
import pandas as pd
import pyarrow as pa


class GpuAcceleratedSeries:
    """
    A pandas Series proxy that accelerates numeric operations on GPU.

    The proxy holds both:
    - _series: the original pandas Series (always valid, source of truth)
    - _gpu: CuPy array (lazily created, invalidated on mutation)

    Operations check if they can run on GPU. If yes, they do.
    If no, they fall back to pandas and log the fallback.
    """

    __slots__ = ("_series", "_gpu", "_gpu_valid", "_name", "_fallback_log")

    def __init__(self, series: pd.Series):
        self._series = series
        self._gpu = None
        self._gpu_valid = False
        self._name = series.name
        self._fallback_log: list[str] = []

    def _ensure_gpu(self) -> cp.ndarray:
        """Lazy H2D transfer — only happens when GPU is actually needed."""
        if not self._gpu_valid:
            if self._series.dtype in (np.float32, np.float64, np.int32, np.int64,
                                       np.int8, np.int16, np.uint8, np.uint16,
                                       np.uint32, np.uint64):
                self._gpu = cp.asarray(self._series.values)
                self._gpu_valid = True
            else:
                return None
        return self._gpu

    def _invalidate_gpu(self):
        """Mark GPU copy as stale after CPU mutation."""
        self._gpu_valid = False
        self._gpu = None

    def _try_gpu(self, op_name: str, gpu_fn, fallback_fn):
        """Try GPU operation, fall back to CPU on failure."""
        gpu = self._ensure_gpu()
        if gpu is not None:
            try:
                return gpu_fn(gpu)
            except Exception:
                pass
        # Fallback
        self._fallback_log.append(op_name)
        return fallback_fn()

    # ---- Aggregations (return scalars) ----

    def sum(self, **kwargs):
        return self._try_gpu(
            "sum",
            lambda g: float(cp.sum(g)),
            lambda: self._series.sum(**kwargs),
        )

    def mean(self, **kwargs):
        return self._try_gpu(
            "mean",
            lambda g: float(cp.mean(g)),
            lambda: self._series.mean(**kwargs),
        )

    def min(self, **kwargs):
        return self._try_gpu(
            "min",
            lambda g: float(cp.min(g)),
            lambda: self._series.min(**kwargs),
        )

    def max(self, **kwargs):
        return self._try_gpu(
            "max",
            lambda g: float(cp.max(g)),
            lambda: self._series.max(**kwargs),
        )

    def std(self, **kwargs):
        return self._try_gpu(
            "std",
            lambda g: float(cp.std(g)),
            lambda: self._series.std(**kwargs),
        )

    def var(self, **kwargs):
        return self._try_gpu(
            "var",
            lambda g: float(cp.var(g)),
            lambda: self._series.var(**kwargs),
        )

    # ---- Arithmetic (return new proxy) ----

    def __add__(self, other):
        if isinstance(other, GpuAcceleratedSeries):
            gpu_self = self._ensure_gpu()
            gpu_other = other._ensure_gpu()
            if gpu_self is not None and gpu_other is not None:
                result = cp.asnumpy(gpu_self + gpu_other)
                return GpuAcceleratedSeries(pd.Series(result, name=self._name))
        return GpuAcceleratedSeries(self._series + _unwrap(other))

    def __mul__(self, other):
        if isinstance(other, GpuAcceleratedSeries):
            gpu_self = self._ensure_gpu()
            gpu_other = other._ensure_gpu()
            if gpu_self is not None and gpu_other is not None:
                result = cp.asnumpy(gpu_self * gpu_other)
                return GpuAcceleratedSeries(pd.Series(result, name=self._name))
        return GpuAcceleratedSeries(self._series * _unwrap(other))

    def __sub__(self, other):
        if isinstance(other, GpuAcceleratedSeries):
            gpu_self = self._ensure_gpu()
            gpu_other = other._ensure_gpu()
            if gpu_self is not None and gpu_other is not None:
                result = cp.asnumpy(gpu_self - gpu_other)
                return GpuAcceleratedSeries(pd.Series(result, name=self._name))
        return GpuAcceleratedSeries(self._series - _unwrap(other))

    def __truediv__(self, other):
        if isinstance(other, GpuAcceleratedSeries):
            gpu_self = self._ensure_gpu()
            gpu_other = other._ensure_gpu()
            if gpu_self is not None and gpu_other is not None:
                result = cp.asnumpy(gpu_self / gpu_other)
                return GpuAcceleratedSeries(pd.Series(result, name=self._name))
        return GpuAcceleratedSeries(self._series / _unwrap(other))

    # ---- Comparison (return boolean proxy) ----

    def __eq__(self, other):
        gpu = self._ensure_gpu()
        if gpu is not None:
            if isinstance(other, GpuAcceleratedSeries):
                other_gpu = other._ensure_gpu()
                if other_gpu is not None:
                    result = cp.asnumpy(gpu == other_gpu)
                    return GpuAcceleratedSeries(pd.Series(result, name=self._name))
            else:
                result = cp.asnumpy(gpu == other)
                return GpuAcceleratedSeries(pd.Series(result, name=self._name))
        return GpuAcceleratedSeries(self._series == _unwrap(other))

    def __gt__(self, other):
        gpu = self._ensure_gpu()
        if gpu is not None:
            result = cp.asnumpy(gpu > (_unwrap_scalar(other)))
            return GpuAcceleratedSeries(pd.Series(result, name=self._name))
        return GpuAcceleratedSeries(self._series > _unwrap(other))

    def __lt__(self, other):
        gpu = self._ensure_gpu()
        if gpu is not None:
            result = cp.asnumpy(gpu < (_unwrap_scalar(other)))
            return GpuAcceleratedSeries(pd.Series(result, name=self._name))
        return GpuAcceleratedSeries(self._series < _unwrap(other))

    def __getitem__(self, key):
        """Indexing — delegate to pandas."""
        if isinstance(key, GpuAcceleratedSeries):
            key = key._series
        result = self._series[key]
        if isinstance(result, pd.Series):
            return GpuAcceleratedSeries(result)
        return result

    # ---- Pandas compatibility ----

    @property
    def values(self):
        return self._series.values

    @property
    def dtype(self):
        return self._series.dtype

    @property
    def name(self):
        return self._name

    @property
    def index(self):
        return self._series.index

    def __len__(self):
        return len(self._series)

    def __repr__(self):
        fallback_info = f" [{len(self._fallback_log)} fallbacks]" if self._fallback_log else ""
        gpu_info = " [GPU]" if self._gpu_valid else " [CPU]"
        return f"GpuAccelerated{gpu_info}{fallback_info}\n{repr(self._series)}"

    def __getattr__(self, name):
        """Fallback: delegate unknown attributes to pandas Series."""
        attr = getattr(self._series, name)
        if callable(attr):
            self._fallback_log.append(name)
            @functools.wraps(attr)
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, pd.Series):
                    return GpuAcceleratedSeries(result)
                elif isinstance(result, pd.DataFrame):
                    return GpuAcceleratedDataFrame(result)
                return result
            return wrapper
        return attr


class GpuAcceleratedDataFrame:
    """
    A pandas DataFrame proxy that accelerates operations on GPU.

    Wraps each numeric column as a GpuAcceleratedSeries.
    Falls back to pandas for string columns, complex indexing, etc.
    """

    __slots__ = ("_df", "_gpu_columns", "_fallback_log")

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._gpu_columns: dict[str, GpuAcceleratedSeries] = {}
        self._fallback_log: list[str] = []

        # Pre-wrap numeric columns
        for col in df.columns:
            if df[col].dtype in (np.float32, np.float64, np.int32, np.int64,
                                  np.int8, np.int16):
                self._gpu_columns[col] = GpuAcceleratedSeries(df[col])

    def __getitem__(self, key):
        if isinstance(key, str) and key in self._gpu_columns:
            return self._gpu_columns[key]
        if isinstance(key, str):
            return self._df[key]
        # Boolean mask
        if isinstance(key, GpuAcceleratedSeries):
            result = self._df[key._series]
            return GpuAcceleratedDataFrame(result)
        result = self._df[key]
        if isinstance(result, pd.DataFrame):
            return GpuAcceleratedDataFrame(result)
        elif isinstance(result, pd.Series):
            return GpuAcceleratedSeries(result)
        return result

    def __setitem__(self, key, value):
        if isinstance(value, GpuAcceleratedSeries):
            self._df[key] = value._series
            self._gpu_columns[key] = value
        else:
            self._df[key] = value
            if key in self._gpu_columns:
                self._gpu_columns[key]._invalidate_gpu()

    @property
    def columns(self):
        return self._df.columns

    @property
    def shape(self):
        return self._df.shape

    @property
    def dtypes(self):
        return self._df.dtypes

    @property
    def loc(self):
        return _LocProxy(self)

    def sum(self, **kwargs):
        """GPU-accelerated sum across columns."""
        result = {}
        for col_name in self._df.columns:
            if col_name in self._gpu_columns:
                result[col_name] = self._gpu_columns[col_name].sum()
            else:
                result[col_name] = self._df[col_name].sum(**kwargs)
        return pd.Series(result)

    def mean(self, **kwargs):
        result = {}
        for col_name in self._df.columns:
            if col_name in self._gpu_columns:
                result[col_name] = self._gpu_columns[col_name].mean()
            else:
                result[col_name] = self._df[col_name].mean(**kwargs)
        return pd.Series(result)

    def describe(self, **kwargs):
        """Fallback to pandas — too complex for a first pass."""
        self._fallback_log.append("describe")
        return self._df.describe(**kwargs)

    def head(self, n=5):
        return self._df.head(n)

    def tail(self, n=5):
        return self._df.tail(n)

    def __len__(self):
        return len(self._df)

    def __repr__(self):
        gpu_cols = len(self._gpu_columns)
        total_cols = len(self._df.columns)
        fallbacks = len(self._fallback_log)
        return (
            f"GpuAcceleratedDataFrame [{gpu_cols}/{total_cols} cols on GPU]"
            f"{f' [{fallbacks} fallbacks]' if fallbacks else ''}\n"
            f"{repr(self._df)}"
        )

    def __getattr__(self, name):
        """Fallback: delegate unknown attributes to pandas DataFrame."""
        attr = getattr(self._df, name)
        if callable(attr):
            self._fallback_log.append(name)
            @functools.wraps(attr)
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    return GpuAcceleratedDataFrame(result)
                elif isinstance(result, pd.Series):
                    return GpuAcceleratedSeries(result)
                return result
            return wrapper
        return attr


class _LocProxy:
    """Proxy for .loc indexer."""
    def __init__(self, gdf: GpuAcceleratedDataFrame):
        self._gdf = gdf

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(row_key, GpuAcceleratedSeries):
                row_key = row_key._series
            result = self._gdf._df.loc[row_key, col_key]
            if isinstance(result, pd.DataFrame):
                return GpuAcceleratedDataFrame(result)
            elif isinstance(result, pd.Series):
                return GpuAcceleratedSeries(result)
            return result
        if isinstance(key, GpuAcceleratedSeries):
            key = key._series
        result = self._gdf._df.loc[key]
        if isinstance(result, pd.DataFrame):
            return GpuAcceleratedDataFrame(result)
        return result


def _unwrap(obj):
    """Unwrap proxy to get underlying pandas object."""
    if isinstance(obj, GpuAcceleratedSeries):
        return obj._series
    if isinstance(obj, GpuAcceleratedDataFrame):
        return obj._df
    return obj


def _unwrap_scalar(obj):
    """Unwrap for comparison — keep scalars as-is."""
    if isinstance(obj, GpuAcceleratedSeries):
        return obj._ensure_gpu()
    return obj


# ============================================================
# Activation function
# ============================================================

def gpu_accelerate(df: pd.DataFrame) -> GpuAcceleratedDataFrame:
    """
    Wrap a pandas DataFrame for GPU acceleration.

    Usage:
        import pandas as pd
        from pandas_proxy import gpu_accelerate

        df = pd.read_csv("big_data.csv")
        gdf = gpu_accelerate(df)

        # Now all operations are GPU-accelerated where possible
        result = gdf["value"].sum()  # Runs on GPU
        grouped = gdf.groupby("category")  # Falls back to pandas
    """
    return GpuAcceleratedDataFrame(df)


# ============================================================
# Tests and Benchmarks
# ============================================================

def test_basic_operations():
    """Test that the proxy produces correct results."""
    print("=== Test 1: Correctness ===\n")

    n = 100_000
    df = pd.DataFrame({
        "a": np.random.randn(n).astype(np.float64),
        "b": np.random.randn(n).astype(np.float64),
        "flag": np.random.randint(0, 2, n).astype(np.int8),
    })

    gdf = gpu_accelerate(df)

    # Aggregations
    assert abs(gdf["a"].sum() - df["a"].sum()) < 0.01
    assert abs(gdf["a"].mean() - df["a"].mean()) < 0.01
    assert abs(gdf["a"].min() - df["a"].min()) < 0.01
    assert abs(gdf["a"].max() - df["a"].max()) < 0.01
    print("  Aggregations: PASS")

    # Arithmetic
    result_gpu = (gdf["a"] * gdf["b"] + gdf["a"]).sum()
    result_cpu = (df["a"] * df["b"] + df["a"]).sum()
    assert abs(result_gpu - result_cpu) < 0.01
    print("  Arithmetic:   PASS")

    # Comparison + filtering
    mask = gdf["flag"] == 1
    filtered_sum = gdf["a"][mask].sum()
    expected = df.loc[df["flag"] == 1, "a"].sum()
    assert abs(filtered_sum - expected) < 0.01
    print("  Filtering:    PASS")

    # DataFrame-level operations
    df_sum = gdf.sum()
    assert abs(df_sum["a"] - df["a"].sum()) < 0.01
    print("  DataFrame sum: PASS")

    # Fallback operations (should work, just slower)
    desc = gdf.describe()
    assert "a" in desc.columns
    print("  describe():   PASS (fallback)")

    print(f"\n  Fallback log: {gdf._fallback_log}")
    print()


def test_transparent_usage():
    """Show that existing pandas code works unchanged."""
    print("=== Test 2: Transparent Usage ===\n")

    # This is real pandas code — the ONLY change is the gpu_accelerate() wrapper
    n = 1_000_000
    df = pd.DataFrame({
        "price": np.random.randn(n).astype(np.float64) * 100 + 1000,
        "volume": np.random.randint(100, 10000, n).astype(np.int64),
        "side": np.random.choice(["buy", "sell"], n),
    })

    # ---- Standard pandas workflow ----
    gdf = gpu_accelerate(df)

    # Numeric operations
    avg_price = gdf["price"].mean()
    total_volume = gdf["volume"].sum()
    price_std = gdf["price"].std()

    # Filtering
    high_volume = gdf[gdf["volume"] > 5000]

    # New column
    gdf["notional"] = gdf["price"] * gdf["volume"]

    print(f"  Avg price:     ${avg_price:,.2f}")
    print(f"  Total volume:  {total_volume:,}")
    print(f"  Price std:     ${price_std:,.2f}")
    print(f"  High vol rows: {len(high_volume):,}")
    print(f"  Notional col:  created")
    print(f"\n  repr:\n{repr(gdf)}")
    print()


def benchmark_proxy_vs_pandas():
    """Benchmark the proxy against raw pandas."""
    print("\n=== Benchmark: Proxy vs Raw Pandas ===\n")

    n = 10_000_000
    df = pd.DataFrame({
        "value": np.random.randn(n).astype(np.float64),
        "flag": np.random.randint(0, 2, n).astype(np.int8),
    })

    gdf = gpu_accelerate(df)

    # Warm up GPU
    _ = gdf["value"].sum()
    cp.cuda.Device(0).synchronize()

    benchmarks = []

    # Sum
    t0 = time.perf_counter()
    for _ in range(10):
        _ = df["value"].sum()
    t_pandas = (time.perf_counter() - t0) / 10

    t0 = time.perf_counter()
    for _ in range(10):
        _ = gdf["value"].sum()
    cp.cuda.Device(0).synchronize()
    t_gpu = (time.perf_counter() - t0) / 10

    benchmarks.append(("sum", t_pandas, t_gpu))

    # Mean
    t0 = time.perf_counter()
    for _ in range(10):
        _ = df["value"].mean()
    t_pandas = (time.perf_counter() - t0) / 10

    t0 = time.perf_counter()
    for _ in range(10):
        _ = gdf["value"].mean()
    cp.cuda.Device(0).synchronize()
    t_gpu = (time.perf_counter() - t0) / 10

    benchmarks.append(("mean", t_pandas, t_gpu))

    # Filtered sum
    t0 = time.perf_counter()
    for _ in range(5):
        _ = df.loc[df["flag"] == 1, "value"].sum()
    t_pandas = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        mask = gdf["flag"] == 1
        _ = gdf["value"][mask].sum()
    cp.cuda.Device(0).synchronize()
    t_gpu = (time.perf_counter() - t0) / 5

    benchmarks.append(("filtered sum", t_pandas, t_gpu))

    # Print results
    for name, t_pd, t_gp in benchmarks:
        speedup = t_pd / t_gp if t_gp > 0 else float("inf")
        print(f"  {name:15s}  pandas={t_pd*1000:8.3f}ms  proxy={t_gp*1000:8.3f}ms  "
              f"speedup={speedup:6.1f}x")

    print()


if __name__ == "__main__":
    print("WinRapids Experiment 005: Pandas GPU Proxy Pattern")
    print("=" * 60)
    print()

    test_basic_operations()
    test_transparent_usage()
    benchmark_proxy_vs_pandas()

    print("=" * 60)
    print("Experiment 005 complete.")
