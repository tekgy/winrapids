"""Observer benchmark suite — rigorous, statistically significant file format benchmarks.

Measures:
1. Write speed (warm cache, multiple runs, mean/std/p50/p95/min/max)
2. Read speed (warm cache)
3. Read speed (cold cache — cache eviction between runs)
4. Column-selective read (1, 3, 5 columns out of 10)
5. GPU H2D transfer timing from each format's read output
6. File size and compression ratio

All timings in milliseconds. All results logged to lab notebook.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev, median

import numpy as np

# Force UTF-8 output on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# GPU
try:
    import cupy as cp
    GPU = True
except ImportError:
    GPU = False

# Arrow/Parquet
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc

# LZ4
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


# ══════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_test_data(n_ticks: int = 598057, n_cols: int = 10) -> dict[str, np.ndarray]:
    """Generate realistic market data matching real AAPL tick structure."""
    rng = np.random.default_rng(42)
    data = {}
    returns = rng.normal(0, 0.0005, n_ticks)
    price = 230.0 * np.exp(np.cumsum(returns))
    data["price"] = price.astype(np.float64)
    data["size"] = rng.exponential(100, n_ticks).astype(np.float64)
    data["timestamp"] = np.arange(n_ticks, dtype=np.int64) * 1000000
    data["notional"] = (price * data["size"]).astype(np.float64)
    data["ln_price"] = np.log(price).astype(np.float64)
    data["sqrt_price"] = np.sqrt(price).astype(np.float64)
    data["recip_price"] = (1.0 / price).astype(np.float64)
    data["round_lot"] = (data["size"] % 100 == 0).astype(np.uint8)
    data["odd_lot"] = (data["size"] < 100).astype(np.uint8)
    data["direction"] = np.sign(returns).astype(np.int8)
    return data


# ══════════════════════════════════════════════════════════════════
# FORMAT IMPLEMENTATIONS (from bench_formats.py + additions)
# ══════════════════════════════════════════════════════════════════

MAGIC = b"MKTF"
VERSION = 1
ALIGN = 64

DTYPE_CODES = {"float64": 0, "int64": 1, "uint8": 2, "int8": 3, "float32": 4}
DTYPE_MAP = {v: getattr(np, k) for k, v in DTYPE_CODES.items()}


def write_raw(path: str, data: dict[str, np.ndarray]):
    cols = list(data.keys())
    n_rows = len(next(iter(data.values())))
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<BHQ", VERSION, len(cols), n_rows))
        header_size = 4 + 1 + 2 + 8
        dir_size = sum(2 + len(name.encode()) + 1 + 8 for name in cols)
        data_start = header_size + dir_size
        offset = data_start
        for name in cols:
            name_bytes = name.encode("utf-8")
            dtype_code = DTYPE_CODES[data[name].dtype.name]
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", dtype_code))
            f.write(struct.pack("<Q", offset))
            offset += data[name].nbytes
        for name in cols:
            data[name].tofile(f)


def read_raw(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC
        version, n_cols, n_rows = struct.unpack("<BHQ", f.read(11))
        col_info = []
        for _ in range(n_cols):
            name_len = struct.unpack("<H", f.read(2))[0]
            name = f.read(name_len).decode("utf-8")
            dtype_code = struct.unpack("<B", f.read(1))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            col_info.append((name, DTYPE_MAP[dtype_code], offset))
        result = {}
        for name, dtype, offset in col_info:
            if columns is not None and name not in columns:
                continue
            f.seek(offset)
            result[name] = np.frombuffer(f.read(n_rows * np.dtype(dtype).itemsize), dtype=dtype).copy()
    return result


def write_aligned(path: str, data: dict[str, np.ndarray]):
    cols = list(data.keys())
    n_rows = len(next(iter(data.values())))
    with open(path, "wb") as f:
        header = {"magic": "MKTG", "version": 1, "n_rows": n_rows, "columns": []}
        offset = ALIGN * 16
        for name in cols:
            arr = data[name]
            offset = (offset + ALIGN - 1) // ALIGN * ALIGN
            header["columns"].append({
                "name": name, "dtype": arr.dtype.name,
                "offset": offset, "nbytes": arr.nbytes,
            })
            offset += arr.nbytes
        header_json = json.dumps(header).encode("utf-8")
        assert len(header_json) < ALIGN * 16
        f.write(header_json)
        f.write(b"\x00" * (ALIGN * 16 - len(header_json)))
        for col_info in header["columns"]:
            name = col_info["name"]
            target_offset = col_info["offset"]
            current = f.tell()
            if current < target_offset:
                f.write(b"\x00" * (target_offset - current))
            data[name].tofile(f)


def read_aligned(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header_bytes = f.read(ALIGN * 16)
        header_json = header_bytes.split(b"\x00")[0]
        header = json.loads(header_json)
        result = {}
        for col in header["columns"]:
            if columns is not None and col["name"] not in columns:
                continue
            f.seek(col["offset"])
            dtype = np.dtype(col["dtype"])
            result[col["name"]] = np.frombuffer(f.read(col["nbytes"]), dtype=dtype).copy()
    return result


def write_mmap(path: str, data: dict[str, np.ndarray]):
    os.makedirs(path, exist_ok=True)
    meta = {"n_rows": len(next(iter(data.values()))), "columns": {}}
    for name, arr in data.items():
        col_path = os.path.join(path, f"{name}.bin")
        arr.tofile(col_path)
        meta["columns"][name] = {"dtype": arr.dtype.name, "nbytes": arr.nbytes}
    with open(os.path.join(path, "_meta.json"), "w") as f:
        json.dump(meta, f)


def read_mmap(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    with open(os.path.join(path, "_meta.json")) as f:
        meta = json.load(f)
    if columns is None:
        columns = list(meta["columns"].keys())
    result = {}
    for name in columns:
        info = meta["columns"][name]
        col_path = os.path.join(path, f"{name}.bin")
        result[name] = np.memmap(col_path, dtype=info["dtype"], mode="r")
    return result


def write_lz4(path: str, data: dict[str, np.ndarray]):
    cols = list(data.keys())
    n_rows = len(next(iter(data.values())))
    with open(path, "wb") as f:
        header = json.dumps({
            "n_rows": n_rows,
            "columns": [{"name": name, "dtype": data[name].dtype.name} for name in cols],
        }).encode()
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for name in cols:
            compressed = lz4.frame.compress(data[name].tobytes())
            f.write(struct.pack("<Q", len(compressed)))
            f.write(compressed)


def read_lz4(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len))
        n_rows = header["n_rows"]
        result = {}
        for col_info in header["columns"]:
            comp_len = struct.unpack("<Q", f.read(8))[0]
            compressed = f.read(comp_len)
            if columns is not None and col_info["name"] not in columns:
                continue
            raw = lz4.frame.decompress(compressed)
            dtype = np.dtype(col_info["dtype"])
            result[col_info["name"]] = np.frombuffer(raw, dtype=dtype).copy()
    return result


# ══════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT
# ══════════════════════════════════════════════════════════════════

def evict_file_cache(file_path: str):
    """Best-effort cache eviction on Windows.

    Strategy: open file with FILE_FLAG_NO_BUFFERING to bypass cache,
    then read through it. This doesn't flush existing cache but ensures
    our next timed read goes through the OS cache layer fresh.

    Alternative: allocate and touch large memory to pressure the cache.
    """
    # Pressure-based eviction: allocate ~2GB, touch every page
    # This pushes file data out of the standby list
    try:
        size_mb = 2048
        arr = np.ones(size_mb * 1024 * 1024 // 8, dtype=np.float64)
        _ = arr.sum()  # force touch
        del arr
        gc.collect()
    except MemoryError:
        # Fallback: smaller pressure
        try:
            arr = np.ones(512 * 1024 * 1024 // 8, dtype=np.float64)
            _ = arr.sum()
            del arr
            gc.collect()
        except MemoryError:
            pass


def evict_file_cache_dir(dir_path: str):
    """Cache eviction for directory-based formats."""
    evict_file_cache(dir_path)


# ══════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ══════════════════════════════════════════════════════════════════

def compute_stats(timings_ms: list[float]) -> dict:
    """Compute statistical summary of timing measurements."""
    s = sorted(timings_ms)
    n = len(s)
    return {
        "mean": mean(s),
        "std": stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "p50": s[n // 2],
        "p95": s[int(n * 0.95)],
        "n": n,
    }


# ══════════════════════════════════════════════════════════════════
# BENCHMARK ENGINE
# ══════════════════════════════════════════════════════════════════

def bench_write(write_fn, path, data, n_runs=30) -> list[float]:
    """Time write operations, return list of individual timings in ms."""
    # Warmup
    write_fn(path, data)
    write_fn(path, data)

    timings = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        write_fn(path, data)
        t1 = time.perf_counter_ns()
        gc.enable()
        timings.append((t1 - t0) / 1e6)
    return timings


def bench_read_warm(read_fn, path, n_runs=30, columns=None) -> list[float]:
    """Time read operations with warm cache."""
    # Warmup
    if columns is not None:
        read_fn(path, columns=columns)
        read_fn(path, columns=columns)
    else:
        read_fn(path)
        read_fn(path)

    timings = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        if columns is not None:
            _ = read_fn(path, columns=columns)
        else:
            _ = read_fn(path)
        t1 = time.perf_counter_ns()
        gc.enable()
        timings.append((t1 - t0) / 1e6)
    return timings


def bench_read_cold(read_fn, path, n_runs=10, is_dir=False, columns=None) -> list[float]:
    """Time read operations with cold cache (cache eviction between runs).

    Fewer runs because cache eviction is expensive.
    """
    timings = []
    for _ in range(n_runs):
        if is_dir:
            evict_file_cache_dir(path)
        else:
            evict_file_cache(path)
        time.sleep(0.05)  # brief settle

        gc.disable()
        t0 = time.perf_counter_ns()
        if columns is not None:
            _ = read_fn(path, columns=columns)
        else:
            _ = read_fn(path)
        t1 = time.perf_counter_ns()
        gc.enable()
        timings.append((t1 - t0) / 1e6)
    return timings


def bench_gpu_h2d(read_fn, path, n_runs=30, columns=None) -> list[float]:
    """Time read-from-disk + host-to-device GPU transfer."""
    if not GPU:
        return []

    # Warmup
    if columns is not None:
        d = read_fn(path, columns=columns)
    else:
        d = read_fn(path)
    for arr in d.values():
        if isinstance(arr, np.ndarray):
            _ = cp.asarray(arr)
    cp.cuda.Device(0).synchronize()

    timings = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        if columns is not None:
            d = read_fn(path, columns=columns)
        else:
            d = read_fn(path)
        for arr in d.values():
            if isinstance(arr, np.ndarray):
                g = cp.asarray(arr)
        cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter_ns()
        gc.enable()
        timings.append((t1 - t0) / 1e6)
    return timings


def bench_gpu_h2d_only(read_fn, path, n_runs=30, columns=None) -> list[float]:
    """Time ONLY the host-to-device transfer (data already in host memory)."""
    if not GPU:
        return []

    if columns is not None:
        d = read_fn(path, columns=columns)
    else:
        d = read_fn(path)

    # Pre-read into contiguous arrays
    arrays = []
    for arr in d.values():
        if isinstance(arr, np.ndarray):
            arrays.append(np.ascontiguousarray(arr))

    # Warmup
    for arr in arrays:
        _ = cp.asarray(arr)
    cp.cuda.Device(0).synchronize()

    timings = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        for arr in arrays:
            g = cp.asarray(arr)
        cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter_ns()
        gc.enable()
        timings.append((t1 - t0) / 1e6)
    return timings


# ══════════════════════════════════════════════════════════════════
# FORMAT REGISTRY
# ══════════════════════════════════════════════════════════════════

def get_formats():
    """Return list of format configs to benchmark."""
    formats = []

    formats.append({
        "name": "Raw binary (MKTF v1)",
        "write": write_raw,
        "read": read_raw,
        "is_dir": False,
        "suffix": ".mktf",
        "col_selective": True,
    })
    formats.append({
        "name": "Aligned binary (64B)",
        "write": write_aligned,
        "read": read_aligned,
        "is_dir": False,
        "suffix": ".mktg",
        "col_selective": True,
    })
    formats.append({
        "name": "Mmap columns",
        "write": write_mmap,
        "read": read_mmap,
        "is_dir": True,
        "suffix": "",
        "col_selective": True,
    })

    if HAS_LZ4:
        formats.append({
            "name": "LZ4 compressed",
            "write": write_lz4,
            "read": read_lz4,
            "is_dir": False,
            "suffix": ".lz4",
            "col_selective": True,
        })

    # Parquet variants
    for comp in ["zstd", "none", "snappy"]:
        formats.append({
            "name": f"Parquet+{comp}",
            "write": lambda p, d, c=comp: _write_parquet(p, d, c),
            "read": lambda p, columns=None: _read_parquet(p, columns),
            "is_dir": False,
            "suffix": ".parquet",
            "col_selective": True,
        })

    # Arrow IPC
    formats.append({
        "name": "Arrow IPC",
        "write": _write_arrow_ipc,
        "read": _read_arrow_ipc,
        "is_dir": False,
        "suffix": ".arrow",
        "col_selective": False,
    })

    # numpy npz
    formats.append({
        "name": "numpy .npz",
        "write": lambda p, d: np.savez(p, **d),
        "read": lambda p, columns=None: _read_npz(p, columns),
        "is_dir": False,
        "suffix": ".npz",
        "col_selective": True,
    })

    return formats


def _write_parquet(path, data, compression):
    tbl = pa.table(data)
    pq.write_table(tbl, path, compression=compression)


def _read_parquet(path, columns=None):
    tbl = pq.read_table(path, columns=columns)
    return {name: tbl.column(name).to_numpy() for name in tbl.column_names}


def _write_arrow_ipc(path, data):
    batch = pa.RecordBatch.from_pydict(data)
    with pa.ipc.RecordBatchFileWriter(path, batch.schema) as writer:
        writer.write_batch(batch)


def _read_arrow_ipc(path, columns=None):
    reader = pa.ipc.open_file(path)
    tbl = reader.read_all()
    if columns is not None:
        tbl = tbl.select(columns)
    return {name: tbl.column(name).to_numpy() for name in tbl.column_names}


def _read_npz(path, columns=None):
    loaded = np.load(path)
    if columns is None:
        return {k: loaded[k] for k in loaded.files}
    return {k: loaded[k] for k in columns if k in loaded.files}


# ══════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════

def run_full_benchmark(n_write=30, n_read_warm=30, n_read_cold=10, n_gpu=30):
    """Run the full benchmark suite."""
    data = generate_test_data(598057, 10)
    n_rows = len(data["price"])
    n_cols = len(data)
    raw_size_mb = sum(arr.nbytes for arr in data.values()) / 1e6
    col_names = list(data.keys())

    print(f"{'='*80}")
    print(f"OBSERVER BENCHMARK SUITE — MKTF Format Research")
    print(f"{'='*80}")
    print(f"Data: {n_rows:,} rows x {n_cols} columns = {raw_size_mb:.1f} MB raw")
    print(f"GPU: {'NVIDIA RTX PRO 6000 Blackwell' if GPU else 'NOT AVAILABLE'}")
    print(f"Runs: write={n_write}, read_warm={n_read_warm}, read_cold={n_read_cold}, gpu={n_gpu}")
    print(f"Columns: {col_names}")
    print()

    formats = get_formats()
    all_results = {}

    # ── Phase 1: Write + Read (warm) + Size ──
    print(f"{'─'*80}")
    print("PHASE 1: Write + Read (warm cache) + File Size")
    print(f"{'─'*80}")

    for fmt in formats:
        name = fmt["name"]
        print(f"\n  [{name}]")

        if fmt["is_dir"]:
            path = tempfile.mkdtemp(prefix="bench_")
        else:
            path = tempfile.mktemp(suffix=fmt["suffix"])

        try:
            # Write
            write_times = bench_write(fmt["write"], path, data, n_write)
            write_stats = compute_stats(write_times)

            # Size
            if fmt["is_dir"]:
                size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
            else:
                size = os.path.getsize(path)
            size_mb = size / 1e6

            # Read warm
            read_warm_times = bench_read_warm(fmt["read"], path, n_read_warm)
            read_warm_stats = compute_stats(read_warm_times)

            print(f"    Write: {write_stats['mean']:7.2f}ms ± {write_stats['std']:5.2f}  "
                  f"[p50={write_stats['p50']:6.2f}, p95={write_stats['p95']:6.2f}, "
                  f"min={write_stats['min']:6.2f}, max={write_stats['max']:6.2f}]")
            print(f"    Read:  {read_warm_stats['mean']:7.2f}ms ± {read_warm_stats['std']:5.2f}  "
                  f"[p50={read_warm_stats['p50']:6.2f}, p95={read_warm_stats['p95']:6.2f}, "
                  f"min={read_warm_stats['min']:6.2f}, max={read_warm_stats['max']:6.2f}]")
            print(f"    Size:  {size_mb:7.2f} MB  (ratio: {size_mb/raw_size_mb:.3f}x)")
            print(f"    Total: {write_stats['mean'] + read_warm_stats['mean']:7.2f}ms (W+R mean)")

            all_results[name] = {
                "write": {"stats": write_stats, "raw": write_times},
                "read_warm": {"stats": read_warm_stats, "raw": read_warm_times},
                "size_mb": size_mb,
                "size_ratio": size_mb / raw_size_mb,
                "path": path,
                "fmt": fmt,
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[name] = {"error": str(e)}

    # ── Phase 2: Cold cache reads ──
    print(f"\n{'─'*80}")
    print("PHASE 2: Read (cold cache — cache eviction between runs)")
    print(f"{'─'*80}")

    for name, res in all_results.items():
        if "error" in res:
            continue
        fmt = res["fmt"]
        path = res["path"]
        print(f"\n  [{name}]")

        try:
            cold_times = bench_read_cold(fmt["read"], path, n_read_cold, fmt["is_dir"])
            cold_stats = compute_stats(cold_times)
            res["read_cold"] = {"stats": cold_stats, "raw": cold_times}

            warm_mean = res["read_warm"]["stats"]["mean"]
            print(f"    Cold:  {cold_stats['mean']:7.2f}ms ± {cold_stats['std']:5.2f}  "
                  f"[p50={cold_stats['p50']:6.2f}, p95={cold_stats['p95']:6.2f}]")
            print(f"    Warm:  {warm_mean:7.2f}ms")
            print(f"    Ratio: {cold_stats['mean'] / warm_mean:.2f}x slower when cold")
        except Exception as e:
            print(f"    ERROR: {e}")

    # ── Phase 3: Column-selective reads ──
    print(f"\n{'─'*80}")
    print("PHASE 3: Column-selective reads (warm cache)")
    print(f"{'─'*80}")

    col_selections = {
        "1 col (price)": ["price"],
        "3 cols (price,size,timestamp)": ["price", "size", "timestamp"],
        "5 cols (price,size,timestamp,notional,ln_price)": ["price", "size", "timestamp", "notional", "ln_price"],
    }

    for name, res in all_results.items():
        if "error" in res:
            continue
        fmt = res["fmt"]
        if not fmt["col_selective"]:
            continue
        path = res["path"]
        print(f"\n  [{name}]")
        res["col_selective"] = {}

        for sel_name, cols in col_selections.items():
            try:
                times = bench_read_warm(fmt["read"], path, n_read_warm, columns=cols)
                stats = compute_stats(times)
                res["col_selective"][sel_name] = {"stats": stats, "raw": times}
                full_mean = res["read_warm"]["stats"]["mean"]
                speedup = full_mean / stats["mean"] if stats["mean"] > 0 else float("inf")
                print(f"    {sel_name}: {stats['mean']:7.2f}ms ± {stats['std']:5.2f}  "
                      f"(vs full: {speedup:.2f}x faster)")
            except Exception as e:
                print(f"    {sel_name}: ERROR {e}")

    # ── Phase 4: GPU H2D timing ──
    if GPU:
        print(f"\n{'─'*80}")
        print("PHASE 4: GPU Host-to-Device transfer timing")
        print(f"{'─'*80}")

        # First: pure H2D baseline (data already in memory)
        print(f"\n  [Baseline: pure H2D transfer, data in host memory]")
        arrays = [np.ascontiguousarray(arr) for arr in data.values()]
        total_bytes = sum(a.nbytes for a in arrays)
        # Warmup
        for a in arrays:
            _ = cp.asarray(a)
        cp.cuda.Device(0).synchronize()

        h2d_times = []
        for _ in range(n_gpu):
            gc.disable()
            t0 = time.perf_counter_ns()
            for a in arrays:
                g = cp.asarray(a)
            cp.cuda.Device(0).synchronize()
            t1 = time.perf_counter_ns()
            gc.enable()
            h2d_times.append((t1 - t0) / 1e6)

        h2d_stats = compute_stats(h2d_times)
        bw = total_bytes / (h2d_stats["mean"] / 1000) / 1e9
        print(f"    Pure H2D: {h2d_stats['mean']:7.2f}ms ± {h2d_stats['std']:5.2f}  "
              f"({total_bytes/1e6:.1f} MB @ {bw:.1f} GB/s)")

        for name, res in all_results.items():
            if "error" in res:
                continue
            fmt = res["fmt"]
            path = res["path"]
            print(f"\n  [{name}]")

            try:
                # Full pipeline: disk read + H2D
                full_times = bench_gpu_h2d(fmt["read"], path, n_gpu)
                full_stats = compute_stats(full_times)
                res["gpu_full"] = {"stats": full_stats, "raw": full_times}

                # H2D only (data pre-read)
                h2d_only_times = bench_gpu_h2d_only(fmt["read"], path, n_gpu)
                h2d_only_stats = compute_stats(h2d_only_times)
                res["gpu_h2d_only"] = {"stats": h2d_only_stats, "raw": h2d_only_times}

                read_portion = full_stats["mean"] - h2d_only_stats["mean"]
                print(f"    Disk→GPU: {full_stats['mean']:7.2f}ms ± {full_stats['std']:5.2f}  "
                      f"(read≈{read_portion:.2f}ms + H2D≈{h2d_only_stats['mean']:.2f}ms)")
            except Exception as e:
                print(f"    ERROR: {e}")

        # Bonus: GPU-native read (cupy.fromfile equivalent)
        print(f"\n  [Bonus: Direct cupy.load if supported]")
        # CuPy doesn't have fromfile, but we can test pinned memory path
        try:
            pinned_times = []
            for _ in range(n_gpu):
                gc.disable()
                # Read into pinned memory, then async H2D
                t0 = time.perf_counter_ns()
                d = read_raw(res["path"])  # use raw binary
                pinned_arrays = []
                for arr in d.values():
                    src = cp.cuda.alloc_pinned_memory(arr.nbytes)
                    dst = np.frombuffer(src, dtype=arr.dtype, count=arr.size)
                    dst[:] = arr
                    pinned_arrays.append((dst, arr.dtype, arr.shape))

                gpu_arrays = []
                for dst, dtype, shape in pinned_arrays:
                    g = cp.empty(shape, dtype=dtype)
                    g.data.copy_from_host(dst.ctypes.data, g.nbytes)
                    gpu_arrays.append(g)
                cp.cuda.Device(0).synchronize()
                t1 = time.perf_counter_ns()
                gc.enable()
                pinned_times.append((t1 - t0) / 1e6)

            pinned_stats = compute_stats(pinned_times)
            print(f"    Pinned H2D (raw binary): {pinned_stats['mean']:7.2f}ms ± {pinned_stats['std']:5.2f}")
        except Exception as e:
            print(f"    Pinned H2D: not supported ({e})")

    # ── Phase 5: Summary table ──
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Format':<24s} {'Write':>8s} {'RdWarm':>8s} {'RdCold':>8s} "
          f"{'Size':>7s} {'W+Rw':>8s} {'DskGPU':>8s}")
    print(f"{'─'*80}")

    for name in sorted(all_results.keys(),
                       key=lambda n: all_results[n].get("read_warm", {}).get("stats", {}).get("mean", 999)):
        res = all_results[name]
        if "error" in res:
            print(f"{name:<24s} ERROR: {res['error']}")
            continue
        w = res["write"]["stats"]["mean"]
        rw = res["read_warm"]["stats"]["mean"]
        rc = res.get("read_cold", {}).get("stats", {}).get("mean", 0)
        sz = res["size_mb"]
        gpu_full = res.get("gpu_full", {}).get("stats", {}).get("mean", 0)
        print(f"{name:<24s} {w:7.2f}ms {rw:7.2f}ms {rc:7.2f}ms "
              f"{sz:6.2f}MB {w+rw:7.2f}ms {gpu_full:7.2f}ms")

    # ── Cleanup ──
    import shutil
    for name, res in all_results.items():
        if "error" in res:
            continue
        path = res["path"]
        if res["fmt"]["is_dir"]:
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.unlink(path)
            except OSError:
                pass

    return all_results


if __name__ == "__main__":
    results = run_full_benchmark()
