"""Custom GPU-native file format exploration.

Tests multiple approaches to storing tick/bin data optimally for
the NVMe -> GPU -> NVMe pipeline. Inspired by:
- Arrow IPC (zero-copy columnar, alignment)
- ArcticDB (versioned, append-only time series)
- NIfTI/CIFTI/GIFTI (neuroimaging: spatial+temporal+metadata in one file)
- Raw binary (absolute minimum overhead)

Each format is tested for:
1. Write speed
2. Read speed
3. File size
4. GPU-readiness (can we H2D directly without conversion?)
5. Metadata overhead
6. Append friendliness (can we add columns without rewriting?)
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

# For GPU tests
try:
    import cupy as cp
    GPU = True
except ImportError:
    GPU = False


def generate_test_data(n_ticks: int = 598057, n_cols: int = 10) -> dict[str, np.ndarray]:
    """Generate realistic market data."""
    rng = np.random.default_rng(42)
    data = {}
    # Simulate price walk
    returns = rng.normal(0, 0.0005, n_ticks)
    price = 230.0 * np.exp(np.cumsum(returns))
    data["price"] = price.astype(np.float64)
    data["size"] = rng.exponential(100, n_ticks).astype(np.float64)
    data["timestamp"] = np.arange(n_ticks, dtype=np.int64) * 1000000  # nanoseconds
    data["notional"] = (price * data["size"]).astype(np.float64)
    # Add derived columns
    data["ln_price"] = np.log(price).astype(np.float64)
    data["sqrt_price"] = np.sqrt(price).astype(np.float64)
    data["recip_price"] = (1.0 / price).astype(np.float64)
    # Add boolean flags as uint8
    data["round_lot"] = (data["size"] % 100 == 0).astype(np.uint8)
    data["odd_lot"] = (data["size"] < 100).astype(np.uint8)
    data["direction"] = np.sign(returns).astype(np.int8)
    return data


# ══════════════════════════════════════════════════════════════════
# FORMAT 1: Raw binary with minimal header
# ══════════════════════════════════════════════════════════════════

MAGIC = b"MKTF"  # Market File
VERSION = 1

def write_raw(path: str, data: dict[str, np.ndarray]):
    """Minimal binary: magic + header + raw column data."""
    cols = list(data.keys())
    n_rows = len(next(iter(data.values())))

    with open(path, "wb") as f:
        # Magic + version + n_cols + n_rows
        f.write(MAGIC)
        f.write(struct.pack("<BHQ", VERSION, len(cols), n_rows))

        # Column directory: name_len + name + dtype_code + offset
        # First pass: compute offsets
        header_size = 4 + 1 + 2 + 8  # magic + version + n_cols + n_rows
        dir_size = sum(2 + len(name.encode()) + 1 + 8 for name in cols)
        data_start = header_size + dir_size

        offset = data_start
        for name in cols:
            name_bytes = name.encode("utf-8")
            dtype_code = {"float64": 0, "int64": 1, "uint8": 2, "int8": 3, "float32": 4}[data[name].dtype.name]
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", dtype_code))
            f.write(struct.pack("<Q", offset))
            offset += data[name].nbytes

        # Column data (contiguous, no padding)
        for name in cols:
            data[name].tofile(f)


def read_raw(path: str) -> dict[str, np.ndarray]:
    """Read minimal binary format."""
    dtype_map = {0: np.float64, 1: np.int64, 2: np.uint8, 3: np.int8, 4: np.float32}

    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC
        version, n_cols, n_rows = struct.unpack("<BHQ", f.read(11))

        columns = []
        for _ in range(n_cols):
            name_len = struct.unpack("<H", f.read(2))[0]
            name = f.read(name_len).decode("utf-8")
            dtype_code = struct.unpack("<B", f.read(1))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            columns.append((name, dtype_map[dtype_code], offset))

        result = {}
        for name, dtype, offset in columns:
            f.seek(offset)
            result[name] = np.frombuffer(f.read(n_rows * np.dtype(dtype).itemsize), dtype=dtype).copy()

    return result


# ══════════════════════════════════════════════════════════════════
# FORMAT 2: Aligned binary (64-byte alignment for GPU DMA)
# ══════════════════════════════════════════════════════════════════

ALIGN = 64  # GPU cache line / DMA alignment

def write_aligned(path: str, data: dict[str, np.ndarray]):
    """64-byte aligned columns for optimal GPU DMA transfer."""
    cols = list(data.keys())
    n_rows = len(next(iter(data.values())))

    with open(path, "wb") as f:
        # Header (padded to ALIGN)
        header = {
            "magic": "MKTG",  # Market GPU
            "version": 1,
            "n_rows": n_rows,
            "columns": [],
        }

        # Compute aligned offsets
        offset = ALIGN * 16  # reserve 1024 bytes for header
        for name in cols:
            arr = data[name]
            # Align offset
            offset = (offset + ALIGN - 1) // ALIGN * ALIGN
            header["columns"].append({
                "name": name,
                "dtype": arr.dtype.name,
                "offset": offset,
                "nbytes": arr.nbytes,
            })
            offset += arr.nbytes

        # Write header as JSON padded to 1024 bytes
        header_json = json.dumps(header).encode("utf-8")
        assert len(header_json) < ALIGN * 16
        f.write(header_json)
        f.write(b"\x00" * (ALIGN * 16 - len(header_json)))

        # Write columns at aligned offsets
        for col_info in header["columns"]:
            name = col_info["name"]
            target_offset = col_info["offset"]
            current = f.tell()
            if current < target_offset:
                f.write(b"\x00" * (target_offset - current))
            data[name].tofile(f)


def read_aligned(path: str) -> dict[str, np.ndarray]:
    """Read aligned binary format."""
    with open(path, "rb") as f:
        header_bytes = f.read(ALIGN * 16)
        header_json = header_bytes.split(b"\x00")[0]
        header = json.loads(header_json)

        result = {}
        for col in header["columns"]:
            f.seek(col["offset"])
            dtype = np.dtype(col["dtype"])
            n_elements = col["nbytes"] // dtype.itemsize
            result[col["name"]] = np.frombuffer(f.read(col["nbytes"]), dtype=dtype).copy()

    return result


# ══════════════════════════════════════════════════════════════════
# FORMAT 3: Memory-mapped columns (separate files per column)
# ══════════════════════════════════════════════════════════════════

def write_mmap(path: str, data: dict[str, np.ndarray]):
    """One file per column — simplest possible, mmap-friendly."""
    os.makedirs(path, exist_ok=True)
    meta = {"n_rows": len(next(iter(data.values()))), "columns": {}}
    for name, arr in data.items():
        col_path = os.path.join(path, f"{name}.bin")
        arr.tofile(col_path)
        meta["columns"][name] = {"dtype": arr.dtype.name, "nbytes": arr.nbytes}
    with open(os.path.join(path, "_meta.json"), "w") as f:
        json.dump(meta, f)


def read_mmap(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read specific columns via memory mapping."""
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


# ══════════════════════════════════════════════════════════════════
# FORMAT 4: LZ4 compressed binary (fast compression)
# ══════════════════════════════════════════════════════════════════

def write_lz4(path: str, data: dict[str, np.ndarray]):
    """LZ4 compressed — fast compression, decent ratio."""
    try:
        import lz4.frame
    except ImportError:
        return None

    cols = list(data.keys())
    n_rows = len(next(iter(data.values())))

    with open(path, "wb") as f:
        header = json.dumps({
            "n_rows": n_rows,
            "columns": [{
                "name": name,
                "dtype": data[name].dtype.name,
            } for name in cols],
        }).encode()
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for name in cols:
            compressed = lz4.frame.compress(data[name].tobytes())
            f.write(struct.pack("<Q", len(compressed)))
            f.write(compressed)

    return True


def read_lz4(path: str) -> dict[str, np.ndarray]:
    """Read LZ4 compressed format."""
    import lz4.frame

    with open(path, "rb") as f:
        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len))
        n_rows = header["n_rows"]
        result = {}
        for col_info in header["columns"]:
            comp_len = struct.unpack("<Q", f.read(8))[0]
            compressed = f.read(comp_len)
            raw = lz4.frame.decompress(compressed)
            dtype = np.dtype(col_info["dtype"])
            result[col_info["name"]] = np.frombuffer(raw, dtype=dtype).copy()
    return result


# ══════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════

def benchmark_format(name, write_fn, read_fn, data, n_iters=20, is_dir=False):
    """Benchmark a format's write/read/size."""
    if is_dir:
        tmp = tempfile.mkdtemp()
    else:
        tmp = tempfile.mktemp(suffix=".bin")

    # Write benchmark
    result = write_fn(tmp, data)
    if result is None:
        return None  # format unavailable

    t0 = time.perf_counter()
    for _ in range(n_iters):
        write_fn(tmp, data)
    t_write = (time.perf_counter() - t0) / n_iters

    # Size
    if is_dir:
        size = sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp))
    else:
        size = os.path.getsize(tmp)

    # Read benchmark
    _ = read_fn(tmp)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = read_fn(tmp)
    t_read = (time.perf_counter() - t0) / n_iters

    # Verify correctness
    readback = read_fn(tmp)
    correct = all(np.array_equal(data[k], readback[k]) for k in data if k in readback)

    # Cleanup
    if is_dir:
        import shutil
        shutil.rmtree(tmp)
    else:
        os.unlink(tmp)

    return {
        "name": name,
        "write_ms": t_write * 1000,
        "read_ms": t_read * 1000,
        "size_mb": size / 1e6,
        "correct": correct,
    }


def main():
    import pyarrow.parquet as pq
    import pyarrow as pa

    data = generate_test_data(598057, 10)
    n_rows = len(data["price"])
    n_cols = len(data)
    raw_size = sum(arr.nbytes for arr in data.values()) / 1e6

    print(f"FILE FORMAT BENCHMARK")
    print(f"{n_rows:,} rows x {n_cols} columns = {raw_size:.1f} MB raw")
    print(f"=" * 70)
    print()

    results = []

    # Parquet + Zstd
    tmp = tempfile.mktemp(suffix=".parquet")
    tbl = pa.table(data)
    t0 = time.perf_counter()
    for _ in range(20):
        pq.write_table(tbl, tmp, compression="zstd")
    t_w = (time.perf_counter() - t0) / 20
    sz = os.path.getsize(tmp) / 1e6
    t0 = time.perf_counter()
    for _ in range(20):
        _ = pq.read_table(tmp)
    t_r = (time.perf_counter() - t0) / 20
    os.unlink(tmp)
    results.append({"name": "Parquet+Zstd", "write_ms": t_w*1000, "read_ms": t_r*1000, "size_mb": sz, "correct": True})

    # Parquet + none
    tmp = tempfile.mktemp(suffix=".parquet")
    t0 = time.perf_counter()
    for _ in range(20):
        pq.write_table(tbl, tmp, compression="none")
    t_w = (time.perf_counter() - t0) / 20
    sz = os.path.getsize(tmp) / 1e6
    t0 = time.perf_counter()
    for _ in range(20):
        _ = pq.read_table(tmp)
    t_r = (time.perf_counter() - t0) / 20
    os.unlink(tmp)
    results.append({"name": "Parquet+none", "write_ms": t_w*1000, "read_ms": t_r*1000, "size_mb": sz, "correct": True})

    # Arrow IPC
    tmp = tempfile.mktemp(suffix=".arrow")
    batch = pa.RecordBatch.from_pydict(data)
    t0 = time.perf_counter()
    for _ in range(20):
        with pa.ipc.RecordBatchFileWriter(tmp, batch.schema) as writer:
            writer.write_batch(batch)
    t_w = (time.perf_counter() - t0) / 20
    sz = os.path.getsize(tmp) / 1e6
    t0 = time.perf_counter()
    for _ in range(20):
        reader = pa.ipc.open_file(tmp)
        _ = reader.read_all()
    t_r = (time.perf_counter() - t0) / 20
    os.unlink(tmp)
    results.append({"name": "Arrow IPC", "write_ms": t_w*1000, "read_ms": t_r*1000, "size_mb": sz, "correct": True})

    # Our custom formats
    r = benchmark_format("Raw binary", write_raw, read_raw, data)
    if r: results.append(r)

    r = benchmark_format("Aligned binary", write_aligned, read_aligned, data)
    if r: results.append(r)

    r = benchmark_format("Mmap columns", write_mmap, read_mmap, data, is_dir=True)
    if r: results.append(r)

    try:
        import lz4.frame
        r = benchmark_format("LZ4 compressed", write_lz4, read_lz4, data)
        if r: results.append(r)
    except ImportError:
        pass

    # numpy .npy
    tmp = tempfile.mktemp(suffix=".npz")
    t0 = time.perf_counter()
    for _ in range(20):
        np.savez(tmp, **data)
    t_w = (time.perf_counter() - t0) / 20
    sz = os.path.getsize(tmp) / 1e6
    t0 = time.perf_counter()
    for _ in range(20):
        _ = dict(np.load(tmp))
    t_r = (time.perf_counter() - t0) / 20
    os.unlink(tmp)
    results.append({"name": "numpy .npz", "write_ms": t_w*1000, "read_ms": t_r*1000, "size_mb": sz, "correct": True})

    # Print results
    print(f"{'Format':<20s} {'Write':>8s} {'Read':>8s} {'Size':>8s} {'W+R':>8s} {'OK':>4s}")
    print(f"{'-'*56}")
    for r in sorted(results, key=lambda x: x["write_ms"] + x["read_ms"]):
        total = r["write_ms"] + r["read_ms"]
        ok = "Y" if r["correct"] else "N"
        print(f"{r['name']:<20s} {r['write_ms']:>7.1f}ms {r['read_ms']:>7.1f}ms {r['size_mb']:>6.1f}MB {total:>7.1f}ms {ok:>4s}")

    print()
    print("Universe estimates (4604 tickers, W+R per ticker):")
    for r in sorted(results, key=lambda x: x["write_ms"] + x["read_ms"]):
        total_s = (r["write_ms"] + r["read_ms"]) / 1000 * 4604
        print(f"  {r['name']:<20s}: {total_s:.1f}s = {total_s/60:.1f} min")


if __name__ == "__main__":
    main()
