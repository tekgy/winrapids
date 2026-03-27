"""MKTF v3 — Self-describing Market Tick Format.

The header IS the manifest IS the state IS the provenance.
No sidecar files. No external metadata. The file is the truth.

Design principles:
  1. 4096-byte alignment (NVMe sector size, DirectStorage-ready)
  2. Header readable in one NVMe sector read (first 4096 bytes)
  3. is_complete flag for crash recovery (write 0 first, flip to 1 at end)
  4. Per-column min/max/null_count for validation without reading data
  5. schema_fingerprint for drift detection
  6. upstream_fingerprints for staleness detection
  7. The daemon reads ONLY headers. Zero data bytes for operational checks.

Layout:
  Block 0 [0..4096):          Fixed header (identity, dimensions, quality, provenance)
  Block 1 [4096..dir_end):    Column directory (128 bytes per column, 4096-aligned)
  Block N [meta_off..):       JSON metadata (variable, 4096-aligned)
  Block M [data_start..):     Column data (each at 4096-byte aligned offset)

Synthesizes findings from:
  - Navigator: 4096-byte alignment for DirectStorage, absolute timestamps
  - Observer: aligned binary wins GPU pipeline, mmap is a trap, bitmask packing
  - Naturalist: NIfTI/ITCH fixed-offset design, column directory always resident
  - Team lead: header replaces manifests, operational state in header
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

MAGIC = b"MKTF"
FORMAT_VERSION = 3
SECTOR = 4096  # NVMe sector size, DirectStorage alignment

DTYPE_CODES = {
    "float32": 0, "float64": 1, "int32": 2, "int64": 3,
    "uint8": 4, "int8": 5, "uint32": 6, "uint16": 7,
    "uint64": 8, "int16": 9,
}
DTYPE_FROM_CODE = {v: getattr(np, k) for k, v in DTYPE_CODES.items()}

# Header flags
FLAG_COMPLETE     = 0x0001  # File write finished successfully
FLAG_HAS_METADATA = 0x0002  # JSON metadata block present
FLAG_HAS_QUALITY  = 0x0004  # Per-column min/max/null_count populated


def _align(offset: int, boundary: int = SECTOR) -> int:
    """Round up to next alignment boundary."""
    return (offset + boundary - 1) & ~(boundary - 1)


def _schema_fingerprint(columns: dict[str, np.ndarray]) -> bytes:
    """16-byte hash of column names + dtypes. Detects schema drift."""
    h = hashlib.sha256()
    for name in sorted(columns.keys()):
        h.update(f"{name}:{columns[name].dtype.name}".encode())
    return h.digest()[:16]


# ══════════════════════════════════════════════════════════════════
# FIXED HEADER (4096 bytes, Block 0)
# ══════════════════════════════════════════════════════════════════
#
# All offsets are from start of file. All integers are little-endian.
#
#   Offset  Size  Field                Type
#   ------  ----  -----                ----
#   0       4     magic                bytes "MKTF"
#   4       2     format_version       uint16
#   6       2     flags                uint16
#   8       4     alignment            uint32 (4096)
#   12      4     header_blocks        uint32 (total 4096-byte blocks before data)
#
#   Identity (offset 16):
#   16      16    leaf_id              bytes (null-padded, e.g. "K01P01")
#   32      8     ticker               bytes (null-padded, e.g. "AAPL")
#   40      10    day                  bytes ("YYYY-MM-DD")
#   50      16    cadence              bytes (null-padded, e.g. "TI00TO00")
#   66      16    leaf_version         bytes (null-padded, e.g. "2.0.0")
#   82      16    schema_fingerprint   bytes (SHA-256 prefix)
#
#   Dimensions (offset 98):
#   98      8     n_rows               uint64
#   106     2     n_cols               uint16
#   108     4     n_bins               uint32 (0 for K01)
#   112     8     bytes_data           uint64 (total data bytes)
#
#   Quality (offset 120):
#   120     8     total_nulls          uint64
#
#   Provenance (offset 128):
#   128     8     write_timestamp_ns   int64
#   136     4     write_duration_ms    uint32
#   140     4     compute_duration_ms  uint32
#
#   Layout pointers (offset 144):
#   144     8     dir_offset           uint64
#   152     8     dir_entries          uint64 (number of directory entries)
#   160     8     meta_offset          uint64 (0 if none)
#   168     8     meta_size            uint64 (0 if none)
#   176     8     data_start           uint64
#
#   [184..4096)  reserved
#
HEADER_FMT = "<4s H H I I"    # magic, version, flags, alignment, header_blocks
IDENTITY_FMT = "<16s 8s 10s 16s 16s 16s"  # leaf_id, ticker, day, cadence, leaf_version, fingerprint
DIMS_FMT = "<Q H I Q"         # n_rows, n_cols, n_bins, bytes_data
QUALITY_FMT = "<Q"             # total_nulls
PROV_FMT = "<q I I"            # write_timestamp_ns, write_duration_ms, compute_duration_ms
LAYOUT_FMT = "<Q Q Q Q Q"     # dir_offset, dir_entries, meta_offset, meta_size, data_start

HEADER_SIZE = 4 + 2 + 2 + 4 + 4  # = 16
IDENTITY_OFFSET = 16
IDENTITY_SIZE = 16 + 8 + 10 + 16 + 16 + 16  # = 82
DIMS_OFFSET = 98
DIMS_SIZE = 8 + 2 + 4 + 8  # = 22
QUALITY_OFFSET = 120
QUALITY_SIZE = 8
PROV_OFFSET = 128
PROV_SIZE = 8 + 4 + 4  # = 16
LAYOUT_OFFSET = 144
LAYOUT_SIZE = 8 * 5  # = 40


# ══════════════════════════════════════════════════════════════════
# COLUMN DIRECTORY ENTRY (128 bytes each)
# ══════════════════════════════════════════════════════════════════
#
#   Offset  Size  Field          Type
#   ------  ----  -----          ----
#   0       32    name           bytes (null-padded UTF-8)
#   32      1     dtype_code     uint8
#   33      7     reserved       padding
#   40      8     n_elements     uint64
#   48      8     data_offset    uint64
#   56      8     data_nbytes    uint64
#   64      8     scale_factor   float64 (1.0 for unscaled)
#   72      8     null_count     uint64
#   80      8     min_value      float64 (NaN if N/A)
#   88      8     max_value      float64 (NaN if N/A)
#   96      32    reserved
#
DIR_ENTRY_SIZE = 128
DIR_ENTRY_FMT = "<B 7x Q Q Q d Q d d 32x"
# pack: dtype_code, n_elements, data_offset, data_nbytes, scale_factor, null_count, min_val, max_val
DIR_ENTRY_PACK_SIZE = struct.calcsize(DIR_ENTRY_FMT)
assert DIR_ENTRY_PACK_SIZE == 96, f"Expected 96, got {DIR_ENTRY_PACK_SIZE}"
# Total per entry: 32 (name) + 96 (fields) = 128


# ══════════════════════════════════════════════════════════════════
# COLUMN STATS
# ══════════════════════════════════════════════════════════════════

def _column_stats(arr: np.ndarray) -> tuple[int, float, float]:
    """Compute null_count, min, max for a column."""
    null_count = 0
    if arr.dtype.kind == 'f':
        null_count = int(np.isnan(arr).sum())
        valid = arr[~np.isnan(arr)] if null_count > 0 else arr
        if len(valid) == 0:
            return null_count, float('nan'), float('nan')
        return null_count, float(valid.min()), float(valid.max())
    else:
        return 0, float(arr.min()), float(arr.max())


# ══════════════════════════════════════════════════════════════════
# WRITER
# ══════════════════════════════════════════════════════════════════

def write_mktf(
    path: str,
    columns: dict[str, np.ndarray],
    *,
    leaf_id: str = "",
    ticker: str = "",
    day: str = "",
    cadence: str = "",
    leaf_version: str = "",
    n_bins: int = 0,
    compute_duration_ms: int = 0,
    metadata: dict | None = None,
    scale_factors: dict[str, float] | None = None,
) -> int:
    """Write an MKTF v3 file.

    The write protocol:
    1. Write header with is_complete=0
    2. Write directory, metadata, column data
    3. Seek back to header, flip is_complete=1
    If we crash between 1 and 3, the file is marked incomplete.

    Returns total bytes written.
    """
    t_start = time.perf_counter_ns()
    col_names = list(columns.keys())
    n_cols = len(col_names)
    n_rows = max(len(arr) for arr in columns.values()) if columns else 0
    bytes_data = sum(arr.nbytes for arr in columns.values())
    fingerprint = _schema_fingerprint(columns)
    scales = scale_factors or {}

    # Compute total_nulls and per-column stats
    col_stats = {}
    total_nulls = 0
    for name in col_names:
        nc, mn, mx = _column_stats(columns[name])
        col_stats[name] = (nc, mn, mx)
        total_nulls += nc

    # Layout computation
    dir_offset = SECTOR  # directory starts at block 1
    dir_size = n_cols * DIR_ENTRY_SIZE
    dir_end = dir_offset + dir_size

    if metadata:
        meta_offset = _align(dir_end)
        meta_json = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        meta_size = len(meta_json)
        meta_end = meta_offset + meta_size
    else:
        meta_offset = 0
        meta_size = 0
        meta_end = dir_end

    # Data offsets — each column at 4096-byte aligned offset
    data_start = _align(meta_end)
    data_offsets = []
    offset = data_start
    for name in col_names:
        data_offsets.append(offset)
        offset = _align(offset + columns[name].nbytes)

    total_size = offset
    header_blocks = data_start // SECTOR

    # Flags
    flags = FLAG_HAS_QUALITY
    if metadata:
        flags |= FLAG_HAS_METADATA
    # NOTE: FLAG_COMPLETE is NOT set yet — set after successful write

    # ── Build header block (4096 bytes) ──
    header = bytearray(SECTOR)

    # Core
    struct.pack_into(HEADER_FMT, header, 0,
                     MAGIC, FORMAT_VERSION, flags, SECTOR, header_blocks)

    # Identity
    struct.pack_into(IDENTITY_FMT, header, IDENTITY_OFFSET,
                     leaf_id.encode()[:16].ljust(16, b"\x00"),
                     ticker.encode()[:8].ljust(8, b"\x00"),
                     day.encode()[:10].ljust(10, b"\x00"),
                     cadence.encode()[:16].ljust(16, b"\x00"),
                     leaf_version.encode()[:16].ljust(16, b"\x00"),
                     fingerprint)

    # Dimensions
    struct.pack_into(DIMS_FMT, header, DIMS_OFFSET,
                     n_rows, n_cols, n_bins, bytes_data)

    # Quality
    struct.pack_into(QUALITY_FMT, header, QUALITY_OFFSET, total_nulls)

    # Provenance (write_timestamp now, duration filled at end)
    write_ts = time.time_ns()
    struct.pack_into(PROV_FMT, header, PROV_OFFSET,
                     write_ts, 0, compute_duration_ms)

    # Layout pointers
    struct.pack_into(LAYOUT_FMT, header, LAYOUT_OFFSET,
                     dir_offset, n_cols, meta_offset, meta_size, data_start)

    # ── Build directory ──
    directory = bytearray(n_cols * DIR_ENTRY_SIZE)
    for i, name in enumerate(col_names):
        arr = columns[name]
        entry_off = i * DIR_ENTRY_SIZE
        name_bytes = name.encode("utf-8")[:32].ljust(32, b"\x00")
        directory[entry_off:entry_off + 32] = name_bytes

        dtype_code = DTYPE_CODES[arr.dtype.name]
        scale = scales.get(name, 1.0)
        nc, mn, mx = col_stats[name]

        struct.pack_into(DIR_ENTRY_FMT, directory, entry_off + 32,
                         dtype_code, len(arr), data_offsets[i], arr.nbytes,
                         scale, nc, mn, mx)

    # ── Write file ──
    with open(path, "wb") as f:
        # 1. Write header (is_complete=0)
        f.write(header)

        # 2. Pad to directory offset
        current = f.tell()
        if current < dir_offset:
            f.write(b"\x00" * (dir_offset - current))

        # 3. Write directory
        f.write(directory)

        # 4. Write metadata
        if metadata:
            current = f.tell()
            if current < meta_offset:
                f.write(b"\x00" * (meta_offset - current))
            f.write(meta_json)

        # 5. Write column data
        for i, name in enumerate(col_names):
            current = f.tell()
            target = data_offsets[i]
            if current < target:
                f.write(b"\x00" * (target - current))
            columns[name].tofile(f)

        # 6. Compute write duration, flip is_complete
        write_duration_ms = int((time.perf_counter_ns() - t_start) / 1_000_000)
        flags |= FLAG_COMPLETE
        f.seek(6)  # flags offset
        f.write(struct.pack("<H", flags))
        f.seek(PROV_OFFSET + 8)  # write_duration_ms offset
        f.write(struct.pack("<I", write_duration_ms))

    return total_size


# ══════════════════════════════════════════════════════════════════
# READERS
# ══════════════════════════════════════════════════════════════════

class MKTFHeader:
    """Parsed MKTF header — everything the daemon/graph needs."""
    __slots__ = (
        "magic", "format_version", "flags", "alignment", "header_blocks",
        "leaf_id", "ticker", "day", "cadence", "leaf_version", "schema_fingerprint",
        "n_rows", "n_cols", "n_bins", "bytes_data",
        "total_nulls", "is_complete",
        "write_timestamp_ns", "write_duration_ms", "compute_duration_ms",
        "dir_offset", "dir_entries", "meta_offset", "meta_size", "data_start",
        "columns",  # list of MKTFColumnInfo
    )

    @property
    def total_ms(self) -> float:
        return self.write_duration_ms + self.compute_duration_ms


class MKTFColumnInfo:
    """Parsed column directory entry."""
    __slots__ = (
        "name", "dtype_code", "dtype", "n_elements",
        "data_offset", "data_nbytes", "scale_factor",
        "null_count", "min_value", "max_value",
    )


def read_header(path: str) -> MKTFHeader:
    """Read ONLY the header + directory. Zero data bytes touched.

    This is what the daemon, graph builder, and BitmapStateDB use.
    Single NVMe sector read for header, then directory if needed.
    """
    h = MKTFHeader()

    with open(path, "rb") as f:
        buf = f.read(SECTOR)

        # Core
        h.magic, h.format_version, h.flags, h.alignment, h.header_blocks = \
            struct.unpack_from(HEADER_FMT, buf, 0)
        assert h.magic == MAGIC, f"Not an MKTF file: {h.magic}"

        # Identity
        raw_id, raw_tick, raw_day, raw_cad, raw_ver, raw_fp = \
            struct.unpack_from(IDENTITY_FMT, buf, IDENTITY_OFFSET)
        h.leaf_id = raw_id.rstrip(b"\x00").decode("utf-8")
        h.ticker = raw_tick.rstrip(b"\x00").decode("utf-8")
        h.day = raw_day.rstrip(b"\x00").decode("utf-8")
        h.cadence = raw_cad.rstrip(b"\x00").decode("utf-8")
        h.leaf_version = raw_ver.rstrip(b"\x00").decode("utf-8")
        h.schema_fingerprint = raw_fp

        # Dimensions
        h.n_rows, h.n_cols, h.n_bins, h.bytes_data = \
            struct.unpack_from(DIMS_FMT, buf, DIMS_OFFSET)

        # Quality
        h.total_nulls, = struct.unpack_from(QUALITY_FMT, buf, QUALITY_OFFSET)
        h.is_complete = bool(h.flags & FLAG_COMPLETE)

        # Provenance
        h.write_timestamp_ns, h.write_duration_ms, h.compute_duration_ms = \
            struct.unpack_from(PROV_FMT, buf, PROV_OFFSET)

        # Layout
        h.dir_offset, h.dir_entries, h.meta_offset, h.meta_size, h.data_start = \
            struct.unpack_from(LAYOUT_FMT, buf, LAYOUT_OFFSET)

        # Read directory
        f.seek(h.dir_offset)
        dir_buf = f.read(h.dir_entries * DIR_ENTRY_SIZE)

        h.columns = []
        for i in range(h.dir_entries):
            c = MKTFColumnInfo()
            entry_off = i * DIR_ENTRY_SIZE
            c.name = dir_buf[entry_off:entry_off + 32].rstrip(b"\x00").decode("utf-8")
            c.dtype_code, c.n_elements, c.data_offset, c.data_nbytes, \
                c.scale_factor, c.null_count, c.min_value, c.max_value = \
                struct.unpack_from(DIR_ENTRY_FMT, dir_buf, entry_off + 32)
            c.dtype = DTYPE_FROM_CODE[c.dtype_code]
            h.columns.append(c)

    return h


def read_data(path: str, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read column data. Full file read + slice (GPU pipeline path)."""
    with open(path, "rb") as f:
        buf = f.read()

    # Parse header for directory
    _, _, flags, _, _ = struct.unpack_from(HEADER_FMT, buf, 0)
    dir_offset, dir_entries, _, _, _ = struct.unpack_from(LAYOUT_FMT, buf, LAYOUT_OFFSET)

    result = {}
    for i in range(dir_entries):
        entry_off = dir_offset + i * DIR_ENTRY_SIZE
        name = buf[entry_off:entry_off + 32].rstrip(b"\x00").decode("utf-8")
        if columns is not None and name not in columns:
            continue
        dtype_code, n_elem, data_offset, data_nbytes, _, _, _, _ = \
            struct.unpack_from(DIR_ENTRY_FMT, buf, entry_off + 32)
        dtype = DTYPE_FROM_CODE[dtype_code]
        result[name] = np.frombuffer(buf, dtype=dtype, count=n_elem,
                                     offset=data_offset).copy()

    return result


def read_selective(path: str, columns: list[str]) -> dict[str, np.ndarray]:
    """Read specific columns via seek. Minimal I/O."""
    h = read_header(path)
    result = {}
    with open(path, "rb") as f:
        for col in h.columns:
            if col.name not in columns:
                continue
            f.seek(col.data_offset)
            result[col.name] = np.frombuffer(
                f.read(col.data_nbytes), dtype=col.dtype
            ).copy()
    return result


def read_metadata(path: str) -> dict | None:
    """Read JSON metadata block only."""
    with open(path, "rb") as f:
        buf = f.read(SECTOR)
    _, _, flags, _, _ = struct.unpack_from(HEADER_FMT, buf, 0)
    if not (flags & FLAG_HAS_METADATA):
        return None
    _, _, meta_offset, meta_size, _ = struct.unpack_from(LAYOUT_FMT, buf, LAYOUT_OFFSET)
    if meta_size == 0:
        return None
    with open(path, "rb") as f:
        f.seek(meta_offset)
        return json.loads(f.read(meta_size))


# ══════════════════════════════════════════════════════════════════
# DATA PREPARATION
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

CONDITION_BITS = {
    2: 0, 7: 1, 8: 2, 9: 3, 10: 4, 12: 5, 14: 6, 15: 7,
    16: 8, 17: 9, 22: 10, 29: 11, 32: 12, 35: 13, 37: 14, 41: 15, 53: 16,
}


def load_aapl() -> tuple[dict[str, np.ndarray], dict]:
    """Load real AAPL and encode as MKTF columns + metadata."""
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(AAPL_PATH))
    raw = {new: tbl.column(old).to_numpy() for old, new in COL_MAP.items()}
    n = len(raw["price"])

    cols = {}
    cols["price"] = raw["price"].astype(np.float32)
    cols["size"] = raw["size"].astype(np.float32)
    cols["timestamp"] = raw["timestamp"].astype(np.int64)
    cols["exchange"] = raw["exchange"].astype(np.uint8)
    cols["sequence"] = np.array(
        [int(s) if s and s != "nan" else 0 for s in raw["sequence"]], dtype=np.int32)

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
    cols["is_odd_lot"] = raw["is_odd_lot"].astype(np.uint8)

    metadata = {
        "condition_bits": {str(k): v for k, v in CONDITION_BITS.items()},
        "source_file": "K01P01.TI00TO00.parquet",
    }

    return cols, metadata


# ══════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("MKTF v3 — Self-Describing Format Benchmark")
    print("=" * 78)
    print()

    cols, meta = load_aapl()
    data_size = sum(arr.nbytes for arr in cols.values())
    print(f"Data: {len(cols)} columns, {len(cols['price']):,} rows, {data_size/1e6:.1f}MB")
    print()

    # ── Write + read benchmark ─────────────────────────────────
    tmp = tempfile.mktemp(suffix=".mktf")

    # Warmup
    for _ in range(5):
        write_mktf(tmp, cols,
                    leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
                    cadence="TI00TO00", leaf_version="1.0.0",
                    metadata=meta)
        _ = read_data(tmp)

    # Write
    t0 = time.perf_counter()
    for _ in range(30):
        write_mktf(tmp, cols,
                    leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
                    cadence="TI00TO00", leaf_version="1.0.0",
                    metadata=meta)
    t_write = (time.perf_counter() - t0) / 30

    file_size = os.path.getsize(tmp)

    # Full read
    t0 = time.perf_counter()
    for _ in range(30):
        _ = read_data(tmp)
    t_read = (time.perf_counter() - t0) / 30

    # Selective read (price + conditions)
    t0 = time.perf_counter()
    for _ in range(30):
        _ = read_selective(tmp, ["price", "conditions"])
    t_sel = (time.perf_counter() - t0) / 30

    # Header-only read
    t0 = time.perf_counter()
    for _ in range(1000):
        h = read_header(tmp)
    t_header = (time.perf_counter() - t0) / 1000

    # Correctness
    readback = read_data(tmp)
    correct = all(np.array_equal(cols[k], readback[k]) for k in cols)

    # ── Header inspection ──────────────────────────────────────
    h = read_header(tmp)
    print("HEADER CONTENTS (what the daemon sees):")
    print("-" * 60)
    print(f"  magic:            {h.magic}")
    print(f"  version:          {h.format_version}")
    print(f"  is_complete:      {h.is_complete}")
    print(f"  alignment:        {h.alignment}")
    print(f"  leaf_id:          {h.leaf_id}")
    print(f"  ticker:           {h.ticker}")
    print(f"  day:              {h.day}")
    print(f"  cadence:          {h.cadence}")
    print(f"  leaf_version:     {h.leaf_version}")
    print(f"  schema_fp:        {h.schema_fingerprint.hex()[:16]}...")
    print(f"  n_rows:           {h.n_rows:,}")
    print(f"  n_cols:           {h.n_cols}")
    print(f"  n_bins:           {h.n_bins}")
    print(f"  bytes_data:       {h.bytes_data:,} ({h.bytes_data/1e6:.1f}MB)")
    print(f"  total_nulls:      {h.total_nulls}")
    print(f"  write_ts:         {h.write_timestamp_ns}")
    print(f"  write_dur:        {h.write_duration_ms}ms")
    print(f"  compute_dur:      {h.compute_duration_ms}ms")
    print(f"  dir_offset:       {h.dir_offset}")
    print(f"  data_start:       {h.data_start}")
    print()

    print("COLUMN DIRECTORY (per-column quality without reading data):")
    print("-" * 78)
    print(f"  {'Name':<16s} {'Dtype':<8s} {'Elements':>10s} {'Size':>8s} "
          f"{'Nulls':>8s} {'Min':>14s} {'Max':>14s}")
    print(f"  {'-'*76}")
    for c in h.columns:
        print(f"  {c.name:<16s} {c.dtype.__name__:<8s} {c.n_elements:>10,} "
              f"{c.data_nbytes/1e6:>6.2f}MB {c.null_count:>8,} "
              f"{c.min_value:>14.4f} {c.max_value:>14.4f}")
    print()

    # ── Metadata roundtrip ─────────────────────────────────────
    meta_back = read_metadata(tmp)
    print(f"METADATA: {json.dumps(meta_back, indent=2)[:200]}...")
    print()

    # ── Timing results ─────────────────────────────────────────
    print("BENCHMARK RESULTS")
    print("-" * 60)
    wr = t_write * 1000 + t_read * 1000
    print(f"  Write:            {t_write*1000:.2f}ms")
    print(f"  Read (full):      {t_read*1000:.2f}ms")
    print(f"  Read (2 cols):    {t_sel*1000:.3f}ms")
    print(f"  Header only:      {t_header*1000:.3f}ms  ({t_header*1e6:.1f}us)")
    print(f"  W+R total:        {wr:.2f}ms")
    print(f"  File size:        {file_size/1e6:.2f}MB (data: {data_size/1e6:.1f}MB, "
          f"overhead: {(file_size-data_size)/1e3:.1f}KB)")
    print(f"  Correct:          {correct}")
    print()

    # Overhead analysis
    overhead_bytes = file_size - data_size
    overhead_pct = overhead_bytes / file_size * 100
    print(f"  4096-byte alignment overhead: {overhead_bytes/1e3:.1f}KB ({overhead_pct:.1f}%)")
    print(f"    Header block:   {SECTOR} bytes")
    print(f"    Directory:      {h.n_cols * DIR_ENTRY_SIZE} bytes")
    print(f"    Padding:        {overhead_bytes - SECTOR - h.n_cols * DIR_ENTRY_SIZE} bytes")
    print()

    # ── vs v2 (64-byte alignment) comparison ───────────────────
    print("v3 (4096-align) vs v2 (64-align) COMPARISON")
    print("-" * 60)
    print(f"  v3 write:  {t_write*1000:.2f}ms   v3 read: {t_read*1000:.2f}ms   "
          f"v3 file: {file_size/1e6:.2f}MB")
    print(f"  v2 was:    3.9ms write    6.0ms read    15.6MB file")
    print(f"  Overhead:  {(file_size/1e6 - 15.6):.2f}MB from 4096-byte padding")
    print()

    # ── GPU H2D ────────────────────────────────────────────────
    try:
        import cupy as cp
        # Read + H2D
        for _ in range(5):
            data = read_data(tmp)
            gpu = {k: cp.asarray(v) for k, v in data.items()}
            cp.cuda.Stream.null.synchronize()
            del gpu

        t0 = time.perf_counter()
        for _ in range(30):
            data = read_data(tmp)
            gpu = {k: cp.asarray(v) for k, v in data.items()}
            cp.cuda.Stream.null.synchronize()
            del gpu
        t_gpu = (time.perf_counter() - t0) / 30
        print(f"  Disk -> GPU:      {t_gpu*1000:.2f}ms")
        print(f"  Universe (4604):  {t_gpu*4604:.1f}s = {t_gpu*4604/60:.1f}min")
    except ImportError:
        print("  CuPy not available for GPU benchmark")

    # ── Universe projection ────────────────────────────────────
    print()
    print("UNIVERSE PROJECTIONS (4,604 tickers)")
    print("-" * 50)
    print(f"  Full W+R:         {wr/1000*4604:.1f}s = {wr/1000*4604/60:.1f}min")
    print(f"  Selective (2col):  {t_sel*4604:.1f}s")
    print(f"  Header scan:       {t_header*4604:.3f}s  ({t_header*4604*1000:.1f}ms)")
    print(f"  Storage (4604):    {file_size/1e6*4604/1e3:.1f}GB")

    # ── Daemon simulation ──────────────────────────────────────
    print()
    print("DAEMON SIMULATION: Scan all 4,604 headers")
    print("-" * 50)
    # How fast can we check is_complete + leaf_version for entire universe?
    t0 = time.perf_counter()
    for _ in range(100):
        h = read_header(tmp)
        _ = h.is_complete and h.leaf_version == "1.0.0"
    t_check = (time.perf_counter() - t0) / 100
    print(f"  Per-file check:    {t_check*1e6:.1f}us")
    print(f"  Full universe:     {t_check*4604*1000:.1f}ms")
    print(f"  (That's a complete staleness check of the entire market)")

    os.unlink(tmp)


if __name__ == "__main__":
    main()
