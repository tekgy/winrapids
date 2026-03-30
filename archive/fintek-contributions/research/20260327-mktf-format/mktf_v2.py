"""MKTF v2 — Market Tick Format, GPU-native.

Synthesizes findings from the full expedition team:
- Naturalist: 64-byte fixed header + 64-byte column directory entries (NIfTI/ITCH-inspired)
- Observer: 64-byte aligned data wins for GPU pipeline (7.98ms vs 8.77ms raw)
- Observer: mmap is a trap (page faults during H2D = worst path)
- Observer: H2D is only 1.3ms — we're 100% disk-read bound
- Navigator: Absolute timestamps (72K negative deltas = out-of-order ticks)
- Navigator: Correct types = 56% size reduction from parquet strings
- Team lead: Integer-only encoding at 10^-8 scale, file IS GPU memory layout

Design principles:
1. The file IS the GPU memory layout. No deserialization.
2. Every column at a 64-byte aligned offset. Seek to it, read it, H2D it.
3. Fixed-size header and directory. Parseable in a single read.
4. Integer-first: prices as int64 scaled by 10^-8.
5. No compression. NVMe bandwidth > CPU decompression.

Layout:
  [0..64)                          Fixed header (64 bytes)
  [64..64+n_cols*64)               Column directory (64 bytes per entry)
  [aligned(dir_end)..meta_end)     JSON metadata (optional, 64-byte aligned)
  [aligned(meta_end)..)            Column data (each at 64-byte aligned offset)
"""

from __future__ import annotations

import json
import struct
import os
import time
import tempfile
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

MAGIC = b"MKTF"
FORMAT_VERSION = 2
ALIGN = 64

# Dtype codes — matches numpy dtype names
DTYPE_CODES = {
    "float32": 0, "float64": 1, "int32": 2, "int64": 3,
    "uint8": 4, "int8": 5, "uint32": 6, "uint16": 7,
    "uint64": 8, "int16": 9, "bool": 10,
}
DTYPE_FROM_CODE = {v: getattr(np, k) for k, v in DTYPE_CODES.items()}

# Header flags
FLAG_HAS_METADATA = 0x0001


def _align(offset: int) -> int:
    """Round up to next 64-byte boundary."""
    return (offset + ALIGN - 1) & ~(ALIGN - 1)


# ══════════════════════════════════════════════════════════════════
# MKTF v2 WRITER
# ══════════════════════════════════════════════════════════════════

def write_mktf(path: str, columns: dict[str, np.ndarray],
               metadata: dict | None = None) -> int:
    """Write MKTF v2 file.

    Header (64 bytes):
      [0:4]    magic "MKTF"
      [4:6]    version uint16
      [6:8]    flags uint16
      [8:16]   n_rows uint64  (max row count across columns)
      [16:18]  n_cols uint16
      [18:26]  reserved (8 bytes)
      [26:34]  metadata_offset uint64 (0 if none)
      [34:42]  metadata_size uint64 (0 if none)
      [42:64]  reserved (22 bytes)

    Column directory entry (64 bytes each):
      [0:32]   name (32 bytes, null-padded UTF-8)
      [32:33]  dtype_code uint8
      [33:40]  reserved (7 bytes)
      [40:48]  n_elements uint64
      [48:56]  data_offset uint64
      [56:64]  data_nbytes uint64

    Returns total bytes written.
    """
    col_names = list(columns.keys())
    n_cols = len(col_names)
    n_rows = max(len(arr) for arr in columns.values()) if columns else 0

    # Compute layout
    dir_start = ALIGN  # header is exactly 64 bytes
    dir_end = dir_start + n_cols * ALIGN  # each entry is 64 bytes

    # Metadata
    flags = 0
    if metadata:
        flags |= FLAG_HAS_METADATA
        meta_json = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        meta_offset = _align(dir_end)
        meta_size = len(meta_json)
        meta_end = meta_offset + meta_size
    else:
        meta_offset = 0
        meta_size = 0
        meta_end = dir_end

    # Column data offsets
    data_offsets = []
    offset = _align(meta_end)
    for name in col_names:
        data_offsets.append(offset)
        offset = _align(offset + columns[name].nbytes)

    total_size = offset

    # Build header (64 bytes)
    header = bytearray(ALIGN)
    struct.pack_into("<4sHH Q H 8x Q Q 22x", header, 0,
                     MAGIC, FORMAT_VERSION, flags,
                     n_rows, n_cols,
                     meta_offset, meta_size)

    # Build directory
    directory = bytearray(n_cols * ALIGN)
    for i, name in enumerate(col_names):
        arr = columns[name]
        entry_off = i * ALIGN
        name_bytes = name.encode("utf-8")[:32].ljust(32, b"\x00")
        dtype_code = DTYPE_CODES[arr.dtype.name]
        directory[entry_off:entry_off + 32] = name_bytes
        struct.pack_into("<B 7x Q Q Q", directory, entry_off + 32,
                         dtype_code,
                         len(arr), data_offsets[i], arr.nbytes)

    # Write: header + directory + metadata + column data
    with open(path, "wb") as f:
        f.write(header)
        f.write(directory)

        # Pad to metadata offset
        if metadata:
            current = f.tell()
            if current < meta_offset:
                f.write(b"\x00" * (meta_offset - current))
            f.write(meta_json)

        # Column data — tofile for each, with alignment padding
        for i, name in enumerate(col_names):
            current = f.tell()
            target = data_offsets[i]
            if current < target:
                f.write(b"\x00" * (target - current))
            columns[name].tofile(f)

    return total_size


# ══════════════════════════════════════════════════════════════════
# MKTF v2 READER — aligned buffer read (for GPU pipeline)
# ══════════════════════════════════════════════════════════════════

def read_mktf(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read MKTF v2 file. Reads entire file into buffer, slices columns.

    This is the GPU pipeline path: read whole file, then H2D selected columns.
    """
    with open(path, "rb") as f:
        buf = f.read()

    return _parse_mktf(buf, columns)


def read_mktf_selective(path: str, columns: list[str]) -> dict[str, np.ndarray]:
    """Read only selected columns via seek. Faster for partial reads."""
    with open(path, "rb") as f:
        # Read header + directory
        header = f.read(ALIGN)
        magic, version, flags, n_rows, n_cols, meta_offset, meta_size = \
            struct.unpack_from("<4sHH Q H 8x Q Q", header, 0)
        assert magic == MAGIC, f"Not an MKTF file: {magic}"

        # Read directory
        dir_buf = f.read(n_cols * ALIGN)

        # Parse directory, seek to selected columns
        result = {}
        for i in range(n_cols):
            entry_off = i * ALIGN
            name = dir_buf[entry_off:entry_off + 32].rstrip(b"\x00").decode("utf-8")
            if name not in columns:
                continue
            dtype_code, n_elem, data_offset, data_nbytes = \
                struct.unpack_from("<B 7x Q Q Q", dir_buf, entry_off + 32)
            dtype = DTYPE_FROM_CODE[dtype_code]
            f.seek(data_offset)
            result[name] = np.frombuffer(f.read(data_nbytes), dtype=dtype).copy()

    return result


def read_mktf_metadata(path: str) -> dict | None:
    """Read only the metadata from an MKTF file."""
    with open(path, "rb") as f:
        header = f.read(ALIGN)
        _, _, flags, _, _, meta_offset, meta_size = \
            struct.unpack_from("<4sHH Q H 8x Q Q", header, 0)
        if not (flags & FLAG_HAS_METADATA) or meta_size == 0:
            return None
        f.seek(meta_offset)
        return json.loads(f.read(meta_size))


def _parse_mktf(buf: bytes, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Parse MKTF from a buffer."""
    magic, version, flags, n_rows, n_cols, meta_offset, meta_size = \
        struct.unpack_from("<4sHH Q H 8x Q Q", buf, 0)
    assert magic == MAGIC

    result = {}
    for i in range(n_cols):
        entry_off = ALIGN + i * ALIGN
        name = buf[entry_off:entry_off + 32].rstrip(b"\x00").decode("utf-8")
        if columns is not None and name not in columns:
            continue
        dtype_code, n_elem, data_offset, data_nbytes = \
            struct.unpack_from("<B 7x Q Q Q", buf, entry_off + 32)
        dtype = DTYPE_FROM_CODE[dtype_code]
        result[name] = np.frombuffer(buf, dtype=dtype, count=n_elem,
                                     offset=data_offset).copy()

    return result


# ══════════════════════════════════════════════════════════════════
# DATA PREPARATION — Real AAPL with multiple encoding paths
# ══════════════════════════════════════════════════════════════════

AAPL_PATH = Path("W:/fintek/data/fractal/K01/2025-09-02/AAPL/K01P01.TI00TO00.parquet")

COL_MAP = {
    "K01P01.DI01DO01": "price",
    "K01P01.DI01DO02": "size",
    "K01P01.DI01DO03": "timestamp",
    "K01P01.DI01DO04": "ticker",
    "K01P01.DI01DO05": "exchange",
    "K01P01.DI01DO06": "sequence",
    "K01P01.DI01DO07": "conditions",
    "K01P01.DI01DO08": "is_odd_lot",
}

# Condition code -> bit position mapping
CONDITION_BITS = {
    2: 0, 7: 1, 8: 2, 9: 3, 10: 4, 12: 5, 14: 6, 15: 7,
    16: 8, 17: 9, 22: 10, 29: 11, 32: 12, 35: 13, 37: 14, 41: 15, 53: 16,
}

PRICE_SCALE = 10**8  # 10^-8 precision: $230.88 = 23_088_000_000


def load_and_encode(encoding: str = "float32") -> tuple[dict[str, np.ndarray], dict]:
    """Load real AAPL data and encode for MKTF.

    encoding options:
      "float32"  — prices/sizes as float32 (current best)
      "float64"  — prices/sizes as float64 (original parquet)
      "int64"    — prices as int64 * 10^8, sizes as int64 * 10^8
      "int32"    — prices as int32 * 10^4 (stocks only, max $214K)
    """
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(AAPL_PATH))

    raw = {}
    for old, new in COL_MAP.items():
        raw[new] = tbl.column(old).to_numpy()

    n = len(raw["price"])
    cols = {}
    metadata = {
        "ticker": "AAPL",
        "date": "2025-09-02",
        "source": "K01P01.TI00TO00",
        "encoding": encoding,
        "condition_bits": {str(k): v for k, v in CONDITION_BITS.items()},
    }

    # Price + size encoding
    if encoding == "float32":
        cols["price"] = raw["price"].astype(np.float32)
        cols["size"] = raw["size"].astype(np.float32)
        metadata["price_scale"] = 1
    elif encoding == "float64":
        cols["price"] = raw["price"].astype(np.float64)
        cols["size"] = raw["size"].astype(np.float64)
        metadata["price_scale"] = 1
    elif encoding == "int64":
        # Fixed-point: price * 10^8 as int64
        cols["price"] = (raw["price"].astype(np.float64) * PRICE_SCALE).astype(np.int64)
        cols["size"] = (raw["size"].astype(np.float64) * PRICE_SCALE).astype(np.int64)
        metadata["price_scale"] = PRICE_SCALE
        metadata["size_scale"] = PRICE_SCALE
    elif encoding == "int32":
        # Fixed-point: price * 10^4 as int32 (max ~$214K, fine for stocks)
        scale = 10**4
        cols["price"] = (raw["price"].astype(np.float64) * scale).astype(np.int32)
        cols["size"] = (raw["size"].astype(np.float64) * scale).astype(np.int32)
        metadata["price_scale"] = scale
        metadata["size_scale"] = scale

    # Timestamp — absolute int64 (navigator: negative deltas exist)
    cols["timestamp"] = raw["timestamp"].astype(np.int64)

    # Exchange — uint8 (1-21, fits in byte)
    cols["exchange"] = raw["exchange"].astype(np.uint8)

    # Sequence — int32
    cols["sequence"] = np.array(
        [int(s) if s and s != "nan" else 0 for s in raw["sequence"]],
        dtype=np.int32
    )

    # Conditions — uint32 bitmask
    bitmasks = np.zeros(n, dtype=np.uint32)
    for i, s in enumerate(raw["conditions"]):
        if s and isinstance(s, str):
            for code in s.split(","):
                code = code.strip()
                if code:
                    bit = CONDITION_BITS.get(int(code))
                    if bit is not None:
                        bitmasks[i] |= 1 << bit
    cols["conditions"] = bitmasks

    # is_odd_lot — uint8
    cols["is_odd_lot"] = raw["is_odd_lot"].astype(np.uint8)

    return cols, metadata


# ══════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════

def bench_encoding(encoding: str, n_iters: int = 30) -> dict:
    """Benchmark a specific encoding path."""
    cols, metadata = load_and_encode(encoding)
    data_size = sum(arr.nbytes for arr in cols.values())

    tmp = tempfile.mktemp(suffix=".mktf")

    # Warmup
    for _ in range(5):
        write_mktf(tmp, cols, metadata)
        _ = read_mktf(tmp)

    # Write
    t0 = time.perf_counter()
    for _ in range(n_iters):
        write_mktf(tmp, cols, metadata)
    t_write = (time.perf_counter() - t0) / n_iters

    file_size = os.path.getsize(tmp)

    # Full read
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = read_mktf(tmp)
    t_read = (time.perf_counter() - t0) / n_iters

    # Selective read (price + conditions)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = read_mktf_selective(tmp, ["price", "conditions"])
    t_sel = (time.perf_counter() - t0) / n_iters

    # Correctness
    readback = read_mktf(tmp)
    correct = all(np.array_equal(cols[k], readback[k]) for k in cols)

    # Metadata roundtrip
    meta_back = read_mktf_metadata(tmp)
    meta_ok = meta_back is not None and meta_back["ticker"] == "AAPL"

    os.unlink(tmp)

    return {
        "encoding": encoding,
        "data_mb": data_size / 1e6,
        "file_mb": file_size / 1e6,
        "write_ms": t_write * 1000,
        "read_ms": t_read * 1000,
        "sel_ms": t_sel * 1000,
        "correct": correct and meta_ok,
    }


def bench_gpu_ops(n_iters: int = 100):
    """Benchmark GPU operations on float32 vs int64 encoded prices.

    Tests: sum, mean, std, min, max, diff, multiply — the actual ops
    our pipeline runs.
    """
    try:
        import cupy as cp
    except ImportError:
        print("  CuPy not available, skipping GPU benchmarks")
        return None

    cols_f32, _ = load_and_encode("float32")
    cols_i64, _ = load_and_encode("int64")

    price_f32 = cp.asarray(cols_f32["price"])
    price_i64 = cp.asarray(cols_i64["price"])
    size_f32 = cp.asarray(cols_f32["size"])
    size_i64 = cp.asarray(cols_i64["size"])

    cp.cuda.Stream.null.synchronize()

    results = {}

    ops = {
        "sum":  (lambda p, s: cp.sum(p),
                 lambda p, s: cp.sum(p)),
        "mean": (lambda p, s: cp.mean(p),
                 lambda p, s: cp.mean(p.astype(cp.float64)) / PRICE_SCALE),
        "std":  (lambda p, s: cp.std(p),
                 lambda p, s: cp.std(p.astype(cp.float64)) / PRICE_SCALE),
        "min":  (lambda p, s: cp.min(p),
                 lambda p, s: cp.min(p)),
        "max":  (lambda p, s: cp.max(p),
                 lambda p, s: cp.max(p)),
        "diff": (lambda p, s: cp.diff(p),
                 lambda p, s: cp.diff(p)),
        "notional (p*s)": (lambda p, s: p * s,
                           lambda p, s: p * s),  # int64 overflow possible!
    }

    for op_name, (f32_fn, i64_fn) in ops.items():
        # Warmup
        for _ in range(10):
            f32_fn(price_f32, size_f32)
            i64_fn(price_i64, size_i64)
        cp.cuda.Stream.null.synchronize()

        # Float32
        t0 = time.perf_counter()
        for _ in range(n_iters):
            f32_fn(price_f32, size_f32)
        cp.cuda.Stream.null.synchronize()
        t_f32 = (time.perf_counter() - t0) / n_iters

        # Int64
        t0 = time.perf_counter()
        for _ in range(n_iters):
            i64_fn(price_i64, size_i64)
        cp.cuda.Stream.null.synchronize()
        t_i64 = (time.perf_counter() - t0) / n_iters

        results[op_name] = {"f32_us": t_f32 * 1e6, "i64_us": t_i64 * 1e6}

    # H2D benchmark — file read + transfer for each encoding
    print("\n  GPU H2D benchmark:")
    for enc in ["float32", "int64"]:
        cols_enc, meta_enc = load_and_encode(enc)
        tmp = tempfile.mktemp(suffix=".mktf")
        write_mktf(tmp, cols_enc, meta_enc)

        # Warmup
        for _ in range(5):
            data = read_mktf(tmp)
            gpu_arrays = {k: cp.asarray(v) for k, v in data.items()}
            cp.cuda.Stream.null.synchronize()
            del gpu_arrays

        # Timed
        t0 = time.perf_counter()
        for _ in range(30):
            data = read_mktf(tmp)
            gpu_arrays = {k: cp.asarray(v) for k, v in data.items()}
            cp.cuda.Stream.null.synchronize()
            del gpu_arrays
        t_total = (time.perf_counter() - t0) / 30

        sz = os.path.getsize(tmp) / 1e6
        os.unlink(tmp)
        print(f"    {enc:8s}: {t_total*1000:.2f}ms disk->GPU  ({sz:.1f}MB)")

    return results


def main():
    print("=" * 78)
    print("MKTF v2 BENCHMARK — Real AAPL Data (598,057 ticks)")
    print("=" * 78)
    print()

    # ── Encoding comparison ────────────────────────────────────
    print("ENCODING COMPARISON")
    print("-" * 78)
    print(f"{'Encoding':<12s} {'Data':>7s} {'File':>7s} {'Write':>8s} {'Read':>8s} "
          f"{'Sel(2)':>8s} {'W+R':>8s} {'OK':>4s}")
    print("-" * 78)

    results = []
    for enc in ["float32", "float64", "int64", "int32"]:
        r = bench_encoding(enc)
        results.append(r)
        wr = r["write_ms"] + r["read_ms"]
        ok = "Y" if r["correct"] else "N"
        print(f"{r['encoding']:<12s} {r['data_mb']:>6.1f}M {r['file_mb']:>6.1f}M "
              f"{r['write_ms']:>7.1f}ms {r['read_ms']:>7.1f}ms "
              f"{r['sel_ms']:>7.2f}ms {wr:>7.1f}ms {ok:>4s}")

    # ── Universe projection ────────────────────────────────────
    print()
    print("UNIVERSE PROJECTION (4,604 tickers)")
    print("-" * 50)
    for r in results:
        wr = r["write_ms"] + r["read_ms"]
        total_s = wr / 1000 * 4604
        print(f"  {r['encoding']:<12s}: {total_s:>6.1f}s = {total_s/60:>5.1f} min  "
              f"(selective: {r['sel_ms']/1000*4604:.1f}s)")

    # ── Per-column size breakdown ──────────────────────────────
    print()
    print("PER-COLUMN SIZE BREAKDOWN")
    print("-" * 60)
    for enc in ["float32", "int64"]:
        cols, _ = load_and_encode(enc)
        total = sum(arr.nbytes for arr in cols.values())
        print(f"\n  {enc}:")
        for name, arr in cols.items():
            pct = arr.nbytes / total * 100
            print(f"    {name:16s} {arr.dtype.name:8s} {arr.nbytes/1e6:>6.2f}MB ({pct:4.1f}%)")
        print(f"    {'TOTAL':16s} {'':8s} {total/1e6:>6.2f}MB")

    # ── GPU ops benchmark ──────────────────────────────────────
    print()
    print("GPU OPERATIONS: float32 vs int64")
    print("-" * 60)
    gpu_results = bench_gpu_ops()
    if gpu_results:
        print(f"\n  {'Operation':<20s} {'float32':>10s} {'int64':>10s} {'Ratio':>8s}")
        print(f"  {'-'*50}")
        for op, times in gpu_results.items():
            ratio = times["i64_us"] / times["f32_us"]
            winner = "<--" if ratio > 1.1 else ("-->" if ratio < 0.9 else "")
            print(f"  {op:<20s} {times['f32_us']:>8.1f}us {times['i64_us']:>8.1f}us "
                  f"{ratio:>6.2f}x {winner}")

    # ── Summary ────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print()
    best = min(results, key=lambda r: r["write_ms"] + r["read_ms"])
    print(f"  Fastest W+R:      {best['encoding']} at {best['write_ms']+best['read_ms']:.1f}ms")
    smallest = min(results, key=lambda r: r["file_mb"])
    print(f"  Smallest file:    {smallest['encoding']} at {smallest['file_mb']:.1f}MB")
    fastest_sel = min(results, key=lambda r: r["sel_ms"])
    print(f"  Fastest selective: {fastest_sel['encoding']} at {fastest_sel['sel_ms']:.2f}ms")
    print()
    print("  MKTF v2 file layout:")
    print("    [0..64)          Fixed header (magic, version, flags, n_rows, n_cols)")
    print("    [64..64+Nx64)    Column directory (name, dtype, offset, nbytes per col)")
    print("    [aligned..)      JSON metadata (ticker, date, scale factors, condition bits)")
    print("    [aligned..)      Column data (each at 64-byte aligned offset)")


if __name__ == "__main__":
    main()
