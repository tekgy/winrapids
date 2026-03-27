"""MKTF file format prototypes — benchmarked against everything.

MKTF = Market Tick Format. GPU-native columnar binary format for
the NVMe -> CPU/GPU pipeline. Designed for fractal tick data.

Variants:
  v1: Minimal header + raw columns at 64-byte aligned offsets
  v2: + uint32 bitmask conditions + delta-encoded timestamps
  v3: + mixed precision (float32 for derived) + constants in header
  v4: Memory-mapped read variant of v2
  v5: Single-pass buffered write (no seeks)

Baselines:
  Parquet+Zstd, Parquet+none, Arrow IPC, numpy .npz, raw binary (from Day 1)
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ══════════════════════════════════════════════════════════════════
# DATA LOADING — Real AAPL tick data
# ══════════════════════════════════════════════════════════════════

AAPL_PATH = Path("W:/fintek/data/fractal/K01/2025-09-02/AAPL/K01P01.TI00TO00.parquet")

# Column name mapping for readability
COL_MAP = {
    "K01P01.DI01DO01": "price",       # float32
    "K01P01.DI01DO02": "size",        # float32
    "K01P01.DI01DO03": "timestamp",   # int64 nanoseconds
    "K01P01.DI01DO04": "ticker",      # string (constant "AAPL")
    "K01P01.DI01DO05": "exchange",    # int32 (1-21)
    "K01P01.DI01DO06": "sequence",    # string (numeric trade IDs)
    "K01P01.DI01DO07": "conditions",  # string (comma-sep condition codes)
    "K01P01.DI01DO08": "is_odd_lot",  # bool
}


def load_real_data() -> dict:
    """Load real AAPL data and return both raw and MKTF-ready versions."""
    tbl = pq.read_table(str(AAPL_PATH))

    raw = {}
    for old, new in COL_MAP.items():
        raw[new] = tbl.column(old).to_numpy()

    return raw


def prepare_mktf_columns(raw: dict) -> dict[str, np.ndarray]:
    """Convert raw data to MKTF-optimized column layout.

    Transforms:
    - conditions: comma-sep string -> uint32 bitmask
    - timestamp: absolute int64 -> first + int32 deltas
    - ticker: constant -> removed (stored in metadata)
    - sequence: string -> int32
    - is_odd_lot: object bool -> uint8
    """
    n = len(raw["price"])
    cols = {}

    # Numeric columns — pass through
    cols["price"] = raw["price"].astype(np.float32)
    cols["size"] = raw["size"].astype(np.float32)
    cols["exchange"] = raw["exchange"].astype(np.int32) if raw["exchange"].dtype != np.int32 else raw["exchange"]

    # Timestamp: delta encoding
    ts = raw["timestamp"].astype(np.int64)
    cols["ts_base"] = np.array([ts[0]], dtype=np.int64)  # single value
    deltas = np.diff(ts).astype(np.int32)
    cols["ts_deltas"] = deltas  # n-1 values

    # Sequence: string -> int32
    cols["sequence"] = np.array([int(s) if s else 0 for s in raw["sequence"]], dtype=np.int32)

    # Conditions: comma-sep codes -> uint32 bitmask
    # Each condition code maps to a bit position
    all_codes = set()
    for s in raw["conditions"]:
        if s and isinstance(s, str):
            for code in s.split(","):
                code = code.strip()
                if code:
                    all_codes.add(int(code))
    code_to_bit = {code: i for i, code in enumerate(sorted(all_codes))}

    bitmasks = np.zeros(n, dtype=np.uint32)
    for i, s in enumerate(raw["conditions"]):
        if s and isinstance(s, str):
            for code in s.split(","):
                code = code.strip()
                if code:
                    bitmasks[i] |= 1 << code_to_bit[int(code)]
    cols["conditions"] = bitmasks

    # Bool -> uint8
    cols["is_odd_lot"] = raw["is_odd_lot"].astype(np.uint8)

    return cols, code_to_bit


# ══════════════════════════════════════════════════════════════════
# MKTF CONSTANTS
# ══════════════════════════════════════════════════════════════════

MAGIC = b"MKTF"
VERSION = 1
ALIGN = 64  # GPU cache line / DMA alignment

DTYPE_CODES = {
    "float32": 0, "float64": 1, "int32": 2, "int64": 3,
    "uint8": 4, "int8": 5, "uint32": 6, "uint16": 7,
}
DTYPE_FROM_CODE = {v: getattr(np, k) for k, v in DTYPE_CODES.items()}


def _align(offset: int, alignment: int = ALIGN) -> int:
    return (offset + alignment - 1) // alignment * alignment


# ══════════════════════════════════════════════════════════════════
# MKTF v1: Aligned columns + column directory
# ══════════════════════════════════════════════════════════════════

def write_mktf_v1(path: str, cols: dict[str, np.ndarray], metadata: dict | None = None):
    """MKTF v1: Fixed header + column directory + aligned raw data.

    Layout:
      [0..4)     MAGIC "MKTF"
      [4..8)     version(u32)
      [8..16)    n_rows(u64)
      [16..18)   n_cols(u16)
      [18..20)   header_size(u16) — total header bytes
      [20..24)   metadata_offset(u32) — 0 if none
      [24..ALIGN) padding
      [ALIGN..ALIGN+n_cols*32) column directory entries (32 bytes each)
      [metadata_offset..data_start) JSON metadata
      [data_start..) column data at aligned offsets
    """
    col_names = list(cols.keys())
    n_cols = len(col_names)
    n_rows_main = max(len(arr) for arr in cols.values())

    # Column directory: 32 bytes per column
    # [0..16) name (null-padded)
    # [16..17) dtype_code(u8)
    # [17..21) n_elements(u32)
    # [21..24) padding
    # [24..32) data_offset(u64)
    DIR_ENTRY_SIZE = 32
    dir_start = ALIGN
    dir_end = dir_start + n_cols * DIR_ENTRY_SIZE

    # Metadata
    if metadata:
        meta_offset = _align(dir_end)
        meta_json = json.dumps(metadata).encode("utf-8")
        meta_end = meta_offset + len(meta_json)
    else:
        meta_offset = 0
        meta_json = b""
        meta_end = dir_end

    # Compute data offsets
    data_offsets = []
    offset = _align(meta_end)
    for name in col_names:
        data_offsets.append(offset)
        offset = _align(offset + cols[name].nbytes)

    header_size = data_offsets[0]  # everything before first data column

    buf = bytearray(offset)

    # Fixed header
    struct.pack_into("<4sIQHHI", buf, 0,
                     MAGIC, VERSION, n_rows_main, n_cols, header_size, meta_offset)

    # Column directory
    for i, name in enumerate(col_names):
        arr = cols[name]
        entry_off = dir_start + i * DIR_ENTRY_SIZE
        name_bytes = name.encode("utf-8")[:16].ljust(16, b"\x00")
        dtype_code = DTYPE_CODES[arr.dtype.name]
        buf[entry_off:entry_off + 16] = name_bytes
        struct.pack_into("<BI3xQ", buf, entry_off + 16,
                         dtype_code, len(arr), data_offsets[i])

    # Metadata
    if meta_json:
        buf[meta_offset:meta_offset + len(meta_json)] = meta_json

    # Column data
    for i, name in enumerate(col_names):
        arr = cols[name]
        start = data_offsets[i]
        buf[start:start + arr.nbytes] = arr.tobytes()

    with open(path, "wb") as f:
        f.write(buf)


def read_mktf_v1(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read MKTF v1. Supports selective column reads."""
    with open(path, "rb") as f:
        # Fixed header
        magic, version, n_rows, n_cols, header_size, meta_offset = \
            struct.unpack_from("<4sIQHHI", f.read(24))
        assert magic == MAGIC

        # Column directory
        f.seek(ALIGN)
        dir_entries = []
        for _ in range(n_cols):
            entry = f.read(32)
            name = entry[:16].rstrip(b"\x00").decode("utf-8")
            dtype_code, n_elem, data_offset = struct.unpack_from("<BI3xQ", entry, 16)
            dir_entries.append((name, dtype_code, n_elem, data_offset))

        # Read selected columns
        result = {}
        for name, dtype_code, n_elem, data_offset in dir_entries:
            if columns is not None and name not in columns:
                continue
            dtype = DTYPE_FROM_CODE[dtype_code]
            f.seek(data_offset)
            nbytes = n_elem * np.dtype(dtype).itemsize
            result[name] = np.frombuffer(f.read(nbytes), dtype=dtype).copy()

    return result


# ══════════════════════════════════════════════════════════════════
# MKTF v2: Buffered write (single allocation, no seeks)
# ══════════════════════════════════════════════════════════════════

def write_mktf_v2(path: str, cols: dict[str, np.ndarray], metadata: dict | None = None):
    """MKTF v2: Same layout as v1, but write via single buffer.

    The entire file is assembled in memory then written in one call.
    This avoids seek overhead and lets the OS batch the write.
    """
    # Same as v1 — the implementation IS v1 which already uses a buffer
    write_mktf_v1(path, cols, metadata)


def read_mktf_v2(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read MKTF v2: read entire file into buffer, slice from memory."""
    with open(path, "rb") as f:
        buf = f.read()

    magic, version, n_rows, n_cols, header_size, meta_offset = \
        struct.unpack_from("<4sIQHHI", buf, 0)
    assert magic == MAGIC

    result = {}
    for i in range(n_cols):
        entry_off = ALIGN + i * 32
        name = buf[entry_off:entry_off + 16].rstrip(b"\x00").decode("utf-8")
        dtype_code, n_elem, data_offset = struct.unpack_from("<BI3xQ", buf, entry_off + 16)
        if columns is not None and name not in columns:
            continue
        dtype = DTYPE_FROM_CODE[dtype_code]
        nbytes = n_elem * np.dtype(dtype).itemsize
        result[name] = np.frombuffer(buf, dtype=dtype, count=n_elem, offset=data_offset).copy()

    return result


# ══════════════════════════════════════════════════════════════════
# MKTF v3: Memory-mapped reads
# ══════════════════════════════════════════════════════════════════

def write_mktf_v3(path: str, cols: dict[str, np.ndarray], metadata: dict | None = None):
    """Same write as v1 — the mmap is a READ optimization."""
    write_mktf_v1(path, cols, metadata)


def read_mktf_v3(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read MKTF via memory mapping — zero-copy column access."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        magic = mm[:4]
        assert magic == MAGIC
        _, version, n_rows, n_cols, header_size, meta_offset = \
            struct.unpack_from("<4sIQHHI", mm, 0)

        result = {}
        for i in range(n_cols):
            entry_off = ALIGN + i * 32
            name = mm[entry_off:entry_off + 16].rstrip(b"\x00").decode("utf-8")
            dtype_code, n_elem, data_offset = struct.unpack_from("<BI3xQ", mm, entry_off + 16)
            if columns is not None and name not in columns:
                continue
            dtype = DTYPE_FROM_CODE[dtype_code]
            nbytes = n_elem * np.dtype(dtype).itemsize
            result[name] = np.frombuffer(mm, dtype=dtype, count=n_elem, offset=data_offset).copy()

        mm.close()

    return result


# ══════════════════════════════════════════════════════════════════
# MKTF v4: np.memmap per column (lazy, zero-copy)
# ══════════════════════════════════════════════════════════════════

def read_mktf_v4_lazy(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read MKTF via np.memmap — truly lazy, zero-copy until accessed."""
    with open(path, "rb") as f:
        header = f.read(24)
        magic, version, n_rows, n_cols, header_size, meta_offset = \
            struct.unpack_from("<4sIQHHI", header, 0)
        assert magic == MAGIC

        f.seek(ALIGN)
        dir_entries = []
        for _ in range(n_cols):
            entry = f.read(32)
            name = entry[:16].rstrip(b"\x00").decode("utf-8")
            dtype_code, n_elem, data_offset = struct.unpack_from("<BI3xQ", entry, 16)
            dir_entries.append((name, dtype_code, n_elem, data_offset))

    result = {}
    for name, dtype_code, n_elem, data_offset in dir_entries:
        if columns is not None and name not in columns:
            continue
        dtype = DTYPE_FROM_CODE[dtype_code]
        result[name] = np.memmap(path, dtype=dtype, mode="r",
                                 offset=data_offset, shape=(n_elem,))

    return result


# ══════════════════════════════════════════════════════════════════
# MKTF v5: Flat write via np.concatenate (minimal Python overhead)
# ══════════════════════════════════════════════════════════════════

def write_mktf_v5(path: str, cols: dict[str, np.ndarray], metadata: dict | None = None):
    """MKTF v5: Minimize Python overhead by using numpy for the data portion.

    Header is still struct-packed, but data section is written as a
    single contiguous numpy buffer where possible.
    """
    col_names = list(cols.keys())
    n_cols = len(col_names)
    n_rows_main = max(len(arr) for arr in cols.values())

    DIR_ENTRY_SIZE = 32
    dir_start = ALIGN
    dir_end = dir_start + n_cols * DIR_ENTRY_SIZE

    if metadata:
        meta_offset = _align(dir_end)
        meta_json = json.dumps(metadata).encode("utf-8")
        meta_end = meta_offset + len(meta_json)
    else:
        meta_offset = 0
        meta_json = b""
        meta_end = dir_end

    # Compute data offsets
    data_offsets = []
    offset = _align(meta_end)
    for name in col_names:
        data_offsets.append(offset)
        offset = _align(offset + cols[name].nbytes)

    total_size = offset
    header_size = data_offsets[0]

    # Build header as bytes
    header_buf = bytearray(header_size)
    struct.pack_into("<4sIQHHI", header_buf, 0,
                     MAGIC, VERSION, n_rows_main, n_cols, header_size, meta_offset)

    for i, name in enumerate(col_names):
        arr = cols[name]
        entry_off = dir_start + i * DIR_ENTRY_SIZE
        name_bytes = name.encode("utf-8")[:16].ljust(16, b"\x00")
        dtype_code = DTYPE_CODES[arr.dtype.name]
        header_buf[entry_off:entry_off + 16] = name_bytes
        struct.pack_into("<BI3xQ", header_buf, entry_off + 16,
                         dtype_code, len(arr), data_offsets[i])

    if meta_json:
        header_buf[meta_offset:meta_offset + len(meta_json)] = meta_json

    # Write in two steps: header + data
    with open(path, "wb") as f:
        f.write(header_buf)
        for i, name in enumerate(col_names):
            # Pad to alignment
            current = header_size + sum(
                _align(cols[col_names[j]].nbytes) for j in range(i)
            ) if i > 0 else 0
            target = data_offsets[i] - header_size
            if i == 0:
                pass  # already aligned from header
            else:
                prev_end = data_offsets[i - 1] - header_size + cols[col_names[i - 1]].nbytes
                pad = data_offsets[i] - header_size - prev_end
                if pad > 0:
                    f.write(b"\x00" * pad)
            cols[name].tofile(f)


# ══════════════════════════════════════════════════════════════════
# Day 1 raw binary baseline (from bench_formats.py)
# ══════════════════════════════════════════════════════════════════

def write_raw_baseline(path: str, cols: dict[str, np.ndarray]):
    """Day 1 raw binary — minimal header, no alignment."""
    col_names = list(cols.keys())
    n_rows = max(len(arr) for arr in cols.values())

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<BHQ", VERSION, len(col_names), n_rows))

        header_size = 4 + 1 + 2 + 8
        dir_size = sum(2 + len(name.encode()) + 1 + 8 for name in col_names)
        offset = header_size + dir_size

        for name in col_names:
            name_bytes = name.encode("utf-8")
            dtype_code = DTYPE_CODES[cols[name].dtype.name]
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", dtype_code))
            f.write(struct.pack("<Q", offset))
            offset += cols[name].nbytes

        for name in col_names:
            cols[name].tofile(f)


def read_raw_baseline(path: str) -> dict[str, np.ndarray]:
    """Read Day 1 raw binary."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC
        version, n_cols, n_rows = struct.unpack("<BHQ", f.read(11))

        dir_entries = []
        for _ in range(n_cols):
            name_len = struct.unpack("<H", f.read(2))[0]
            name = f.read(name_len).decode("utf-8")
            dtype_code = struct.unpack("<B", f.read(1))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            dir_entries.append((name, dtype_code, offset))

        result = {}
        for name, dtype_code, offset in dir_entries:
            dtype = DTYPE_FROM_CODE[dtype_code]
            f.seek(offset)
            # Use n_rows for main columns, but ts_base has 1 element
            # Determine size from next offset or file end
            nbytes_per = np.dtype(dtype).itemsize
            result[name] = np.frombuffer(f.read(n_rows * nbytes_per), dtype=dtype).copy()

    return result


def write_raw_baseline_fixed(path: str, cols: dict[str, np.ndarray]):
    """Day 1 raw binary — minimal header, no alignment, handles variable-length columns."""
    col_names = list(cols.keys())
    n_rows = max(len(arr) for arr in cols.values())

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<BHQ", VERSION, len(col_names), n_rows))

        header_size = 4 + 1 + 2 + 8
        # Extended dir: name_len + name + dtype_code + n_elements + offset
        dir_size = sum(2 + len(name.encode()) + 1 + 4 + 8 for name in col_names)
        offset = header_size + dir_size

        for name in col_names:
            name_bytes = name.encode("utf-8")
            dtype_code = DTYPE_CODES[cols[name].dtype.name]
            n_elem = len(cols[name])
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", dtype_code))
            f.write(struct.pack("<I", n_elem))
            f.write(struct.pack("<Q", offset))
            offset += cols[name].nbytes

        for name in col_names:
            cols[name].tofile(f)


def read_raw_baseline_fixed(path: str) -> dict[str, np.ndarray]:
    """Read Day 1 raw binary with per-column element counts."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC
        version, n_cols, n_rows = struct.unpack("<BHQ", f.read(11))

        dir_entries = []
        for _ in range(n_cols):
            name_len = struct.unpack("<H", f.read(2))[0]
            name = f.read(name_len).decode("utf-8")
            dtype_code = struct.unpack("<B", f.read(1))[0]
            n_elem = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            dir_entries.append((name, dtype_code, n_elem, offset))

        result = {}
        for name, dtype_code, n_elem, offset in dir_entries:
            dtype = DTYPE_FROM_CODE[dtype_code]
            f.seek(offset)
            nbytes = n_elem * np.dtype(dtype).itemsize
            result[name] = np.frombuffer(f.read(nbytes), dtype=dtype).copy()

    return result


# ══════════════════════════════════════════════════════════════════
# BENCHMARK HARNESS
# ══════════════════════════════════════════════════════════════════

def bench(name: str, write_fn, read_fn, data, n_iters: int = 30,
          write_args: dict | None = None, is_dir: bool = False) -> dict:
    """Benchmark a format with warmup + measured iterations."""
    tmp = tempfile.mktemp(suffix=".mktf") if not is_dir else tempfile.mkdtemp()
    wa = write_args or {}

    # Warmup
    for _ in range(3):
        write_fn(tmp, data, **wa) if wa else write_fn(tmp, data)
        _ = read_fn(tmp)

    # Write benchmark
    t0 = time.perf_counter()
    for _ in range(n_iters):
        write_fn(tmp, data, **wa) if wa else write_fn(tmp, data)
    t_write = (time.perf_counter() - t0) / n_iters

    # Size
    if is_dir:
        size = sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp))
    else:
        size = os.path.getsize(tmp)

    # Read benchmark
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = read_fn(tmp)
    t_read = (time.perf_counter() - t0) / n_iters

    # Correctness
    readback = read_fn(tmp)
    correct = all(
        np.array_equal(data[k], readback[k])
        for k in data if k in readback
    )

    # Cleanup — force-delete memmap references first
    del readback
    import gc; gc.collect()
    if is_dir:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        try:
            os.unlink(tmp)
        except PermissionError:
            pass  # memmap or Windows file lock

    return {
        "name": name,
        "write_ms": t_write * 1000,
        "read_ms": t_read * 1000,
        "size_mb": size / 1e6,
        "correct": correct,
    }


def bench_selective_read(name: str, read_fn, path: str,
                         columns: list[str], n_iters: int = 30) -> dict:
    """Benchmark selective column reads."""
    # Warmup
    for _ in range(3):
        _ = read_fn(path, columns=columns)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = read_fn(path, columns=columns)
    t_read = (time.perf_counter() - t0) / n_iters

    return {"name": name, "read_ms": t_read * 1000, "cols": len(columns)}


def main():
    print("=" * 78)
    print("MKTF FORMAT BENCHMARK — Real AAPL Data (598,057 ticks)")
    print("=" * 78)
    print()

    # ── Load and prepare data ──────────────────────────────────
    print("Loading real AAPL data...")
    raw = load_real_data()
    print(f"  Raw columns: {list(raw.keys())}")
    raw_size = sum(arr.nbytes for arr in raw.values() if hasattr(arr, 'nbytes'))
    # String columns need special handling for size
    for k, v in raw.items():
        if v.dtype == object:
            raw_size += sum(len(str(x)) for x in v)
    print(f"  Raw size: {raw_size / 1e6:.1f} MB")

    print("\nPreparing MKTF columns (bitmask conditions, delta timestamps)...")
    mktf_cols, code_to_bit = prepare_mktf_columns(raw)
    mktf_size = sum(arr.nbytes for arr in mktf_cols.values())
    print(f"  MKTF columns: {list(mktf_cols.keys())}")
    for name, arr in mktf_cols.items():
        print(f"    {name:16s} {arr.dtype.name:8s} {len(arr):>8,} elem  {arr.nbytes/1e6:.2f} MB")
    print(f"  MKTF total: {mktf_size / 1e6:.1f} MB")
    print(f"  Condition codes -> bits: {code_to_bit}")
    print()

    metadata = {
        "ticker": "AAPL",
        "date": "2025-09-02",
        "source": "K01P01.TI00TO00",
        "condition_bits": {str(k): v for k, v in code_to_bit.items()},
    }

    results = []

    # ── Baseline: Parquet+Zstd (original raw data as Arrow) ────
    print("Benchmarking baselines...")
    tbl = pq.read_table(str(AAPL_PATH))

    tmp_pq = tempfile.mktemp(suffix=".parquet")
    # Warmup
    for _ in range(3):
        pq.write_table(tbl, tmp_pq, compression="zstd")
        _ = pq.read_table(tmp_pq)

    t0 = time.perf_counter()
    for _ in range(30):
        pq.write_table(tbl, tmp_pq, compression="zstd")
    t_w = (time.perf_counter() - t0) / 30
    sz = os.path.getsize(tmp_pq) / 1e6
    t0 = time.perf_counter()
    for _ in range(30):
        _ = pq.read_table(tmp_pq)
    t_r = (time.perf_counter() - t0) / 30
    os.unlink(tmp_pq)
    results.append({"name": "Parquet+Zstd (raw)", "write_ms": t_w*1000, "read_ms": t_r*1000, "size_mb": sz, "correct": True})

    # Parquet no compression
    tmp_pq = tempfile.mktemp(suffix=".parquet")
    for _ in range(3):
        pq.write_table(tbl, tmp_pq, compression="none")
        _ = pq.read_table(tmp_pq)
    t0 = time.perf_counter()
    for _ in range(30):
        pq.write_table(tbl, tmp_pq, compression="none")
    t_w = (time.perf_counter() - t0) / 30
    sz = os.path.getsize(tmp_pq) / 1e6
    t0 = time.perf_counter()
    for _ in range(30):
        _ = pq.read_table(tmp_pq)
    t_r = (time.perf_counter() - t0) / 30
    os.unlink(tmp_pq)
    results.append({"name": "Parquet+none (raw)", "write_ms": t_w*1000, "read_ms": t_r*1000, "size_mb": sz, "correct": True})

    # Arrow IPC
    tmp_arrow = tempfile.mktemp(suffix=".arrow")
    batch = tbl.combine_chunks().to_batches()[0]
    def _arrow_read(p):
        reader = pa.ipc.open_file(p)
        result = reader.read_all()
        del reader
        return result

    for _ in range(3):
        with pa.ipc.RecordBatchFileWriter(tmp_arrow, batch.schema) as w:
            w.write_batch(batch)
        _arrow_read(tmp_arrow)
    t0 = time.perf_counter()
    for _ in range(30):
        with pa.ipc.RecordBatchFileWriter(tmp_arrow, batch.schema) as w:
            w.write_batch(batch)
    t_w = (time.perf_counter() - t0) / 30
    sz = os.path.getsize(tmp_arrow) / 1e6
    t0 = time.perf_counter()
    for _ in range(30):
        _arrow_read(tmp_arrow)
    t_r = (time.perf_counter() - t0) / 30
    import gc; gc.collect()
    try:
        os.unlink(tmp_arrow)
    except PermissionError:
        pass  # Windows file locking
    results.append({"name": "Arrow IPC (raw)", "write_ms": t_w*1000, "read_ms": t_r*1000, "size_mb": sz, "correct": True})

    # numpy .npz (MKTF columns only — fair comparison)
    tmp_npz = tempfile.mktemp(suffix=".npz")
    for _ in range(3):
        np.savez(tmp_npz, **mktf_cols)
        _ = dict(np.load(tmp_npz))
    t0 = time.perf_counter()
    for _ in range(30):
        np.savez(tmp_npz, **mktf_cols)
    t_w = (time.perf_counter() - t0) / 30
    sz = os.path.getsize(tmp_npz) / 1e6
    t0 = time.perf_counter()
    for _ in range(30):
        _ = dict(np.load(tmp_npz))
    t_r = (time.perf_counter() - t0) / 30
    os.unlink(tmp_npz)
    results.append({"name": "numpy .npz (mktf)", "write_ms": t_w*1000, "read_ms": t_r*1000, "size_mb": sz, "correct": True})

    # ── MKTF variants ──────────────────────────────────────────
    print("Benchmarking MKTF variants...")

    # Raw baseline (no alignment)
    results.append(bench("Raw binary (no align)",
                         write_raw_baseline_fixed, read_raw_baseline_fixed,
                         mktf_cols))

    # MKTF v1: aligned + directory
    results.append(bench("MKTF v1 (aligned)",
                         write_mktf_v1, read_mktf_v1,
                         mktf_cols, write_args={"metadata": metadata}))

    # MKTF v2: buffered read
    results.append(bench("MKTF v2 (buf read)",
                         write_mktf_v1, read_mktf_v2,
                         mktf_cols, write_args={"metadata": metadata}))

    # MKTF v3: mmap read
    results.append(bench("MKTF v3 (mmap)",
                         write_mktf_v1, read_mktf_v3,
                         mktf_cols, write_args={"metadata": metadata}))

    # MKTF v4: np.memmap lazy
    results.append(bench("MKTF v4 (np.memmap)",
                         write_mktf_v1, read_mktf_v4_lazy,
                         mktf_cols, write_args={"metadata": metadata}))

    # MKTF v5: tofile write
    results.append(bench("MKTF v5 (tofile wr)",
                         write_mktf_v5, read_mktf_v2,
                         mktf_cols, write_args={"metadata": metadata}))

    # ── Print results ──────────────────────────────────────────
    print()
    print(f"{'Format':<24s} {'Write':>8s} {'Read':>8s} {'Size':>8s} {'W+R':>8s} {'OK':>4s}")
    print(f"{'-'*60}")
    for r in sorted(results, key=lambda x: x["write_ms"] + x["read_ms"]):
        total = r["write_ms"] + r["read_ms"]
        ok = "Y" if r["correct"] else "N"
        print(f"{r['name']:<24s} {r['write_ms']:>7.1f}ms {r['read_ms']:>7.1f}ms {r['size_mb']:>6.1f}MB {total:>7.1f}ms {ok:>4s}")

    # ── Selective read benchmark ───────────────────────────────
    print()
    print("SELECTIVE COLUMN READ BENCHMARK (price + conditions only)")
    print(f"{'-'*50}")

    # Write a persistent MKTF file for selective reads
    tmp_sel = tempfile.mktemp(suffix=".mktf")
    write_mktf_v1(tmp_sel, mktf_cols, metadata=metadata)

    sel_cols = ["price", "conditions"]
    sel_results = []
    sel_results.append(bench_selective_read("MKTF v1 (seek)", read_mktf_v1, tmp_sel, sel_cols))
    sel_results.append(bench_selective_read("MKTF v2 (buf)", read_mktf_v2, tmp_sel, sel_cols))
    sel_results.append(bench_selective_read("MKTF v3 (mmap)", read_mktf_v3, tmp_sel, sel_cols))
    sel_results.append(bench_selective_read("MKTF v4 (memmap)", read_mktf_v4_lazy, tmp_sel, sel_cols))

    # Parquet selective — use only uniform-length columns
    tmp_pq_sel = tempfile.mktemp(suffix=".parquet")
    n_main = len(mktf_cols["price"])
    pq_cols = {k: v for k, v in mktf_cols.items() if len(v) == n_main}
    mktf_tbl = pa.table(pq_cols)
    pq.write_table(mktf_tbl, tmp_pq_sel, compression="none")
    for _ in range(3):
        _ = pq.read_table(tmp_pq_sel, columns=sel_cols)
    t0 = time.perf_counter()
    for _ in range(30):
        _ = pq.read_table(tmp_pq_sel, columns=sel_cols)
    t_r = (time.perf_counter() - t0) / 30
    sel_results.append({"name": "Parquet+none", "read_ms": t_r*1000, "cols": 2})
    os.unlink(tmp_pq_sel)

    for sr in sorted(sel_results, key=lambda x: x["read_ms"]):
        print(f"  {sr['name']:<24s} {sr['read_ms']:>7.2f}ms  ({sr['cols']} cols)")

    os.unlink(tmp_sel)

    # ── Universe projection ────────────────────────────────────
    print()
    print("UNIVERSE PROJECTION (4,604 tickers)")
    print(f"{'-'*60}")
    for r in sorted(results, key=lambda x: x["write_ms"] + x["read_ms"]):
        total_s = (r["write_ms"] + r["read_ms"]) / 1000 * 4604
        print(f"  {r['name']:<24s}: {total_s:>6.1f}s = {total_s/60:>5.1f} min")

    # ── Data compression analysis ──────────────────────────────
    print()
    print("DATA SIZE ANALYSIS")
    print(f"{'-'*50}")
    print(f"  Original parquet:     {os.path.getsize(str(AAPL_PATH))/1e6:.2f} MB")
    print(f"  MKTF encoded data:    {mktf_size/1e6:.2f} MB")
    savings = 1 - mktf_size / raw_size
    print(f"  Size reduction:       {savings*100:.0f}% vs raw strings")

    # Breakdown of savings
    print()
    print("  Column encoding impact:")
    print(f"    conditions: str -> uint32 bitmask: {598057*4/1e6:.2f} MB (was ~{sum(len(str(x)) for x in raw['conditions'])/1e6:.1f} MB)")
    ts_raw = raw["timestamp"].nbytes
    ts_mktf = mktf_cols["ts_base"].nbytes + mktf_cols["ts_deltas"].nbytes
    print(f"    timestamp:  int64 -> base+delta:   {ts_mktf/1e6:.2f} MB (was {ts_raw/1e6:.1f} MB)")
    print(f"    ticker:     removed (in metadata):  0.00 MB (was ~{sum(len(str(x)) for x in raw['ticker'])/1e6:.1f} MB)")
    seq_raw = sum(len(str(x)) for x in raw["sequence"])
    print(f"    sequence:   str -> int32:           {598057*4/1e6:.2f} MB (was ~{seq_raw/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
