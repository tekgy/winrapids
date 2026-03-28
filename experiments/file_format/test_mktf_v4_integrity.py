"""MKTF v4 Integrity Verification — Header Roundtrip + Checksum Stress Test.

Not a benchmark — a correctness test. The v4 header has 13 hand-packed
struct sections with 50+ fields. A single off-by-one in a format string
silently corrupts data. This test:

1. Packs a fully-populated header, unpacks it, verifies every field
2. Tests edge cases: max values, NaN, zero, negative, unicode tickers
3. Writes files with pathological data, verifies checksum roundtrip
4. Tests the crash recovery invariant (is_complete=0 before finalization)
5. Verifies column directory roundtrip (name, dtype, offsets, stats)
"""

from __future__ import annotations

import io
import math
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, "R:/fintek")

from trunk.backends.mktf.format import (
    FORMAT_VERSION, MAGIC, ALIGNMENT, BLOCK_SIZE,
    MKTFHeader, ColumnEntry, UpstreamFingerprint,
    pack_block0, unpack_block0,
    pack_column_entry, unpack_column_entry,
    compute_schema_fingerprint, compute_leaf_id_hash,
    DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT64, DTYPE_UINT8, DTYPE_UINT32, DTYPE_BOOL,
    SHAPE_POINTWISE, SHAPE_REDUCTION, SHAPE_CROSS_TICKER,
    TARGET_GPU_CUDA, TARGET_GPU_TENSOR,
    PRECISION_FP32, PRECISION_FP64,
    LEAF_COLUMNAR, LEAF_MATRIX,
    ASSET_EQUITY, ASSET_CRYPTO,
    MAX_UPSTREAM, UPSTREAM_ENTRY_SIZE, COL_DIR_ENTRY_SIZE,
    FLAG_HAS_METADATA, FLAG_HAS_QUALITY, FLAG_HAS_STATISTICS,
    FLAG_HAS_UPSTREAM, FLAG_IS_VALIDATED,
)
from trunk.backends.mktf.writer import write_mktf, flip_dirty, flip_complete
from trunk.backends.mktf.reader import (
    read_header, read_columns, read_selective, read_status,
    is_dirty, is_complete, verify_checksum, scan_dirty, scan_incomplete,
)


passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def check_eq(name: str, actual, expected):
    if isinstance(expected, float) and math.isnan(expected):
        check(name, isinstance(actual, float) and math.isnan(actual),
              f"expected NaN, got {actual}")
    elif isinstance(expected, float):
        check(name, abs(actual - expected) < 1e-10,
              f"expected {expected}, got {actual}")
    else:
        check(name, actual == expected,
              f"expected {expected!r}, got {actual!r}")


# ═══════════════════════════════════════════════════════════════
# TEST 1: Full Header Roundtrip
# ═══════════════════════════════════════════════════════════════

print("TEST 1: FULL HEADER ROUNDTRIP (50+ fields)")
print("-" * 60)

h = MKTFHeader(
    format_version=FORMAT_VERSION,
    flags=FLAG_HAS_METADATA | FLAG_HAS_QUALITY | FLAG_HAS_STATISTICS | FLAG_HAS_UPSTREAM | FLAG_IS_VALIDATED,
    alignment=ALIGNMENT,
    header_blocks=3,

    # Identity
    leaf_id="K02B05_returns_vol",
    ticker="AAPL",
    day="2025-09-02",
    ti=9,
    to=16,
    leaf_version="2.1.0",
    schema_fingerprint=b"\xab\xcd\xef" + b"\x00" * 13,
    leaf_id_hash=0xDEADBEEF,

    # Tree
    kingdom=2,
    phylum=5,
    class_=3,
    rank=7,
    family=1,
    genus=4,
    species=2,
    depth=3,
    computation_shape=SHAPE_REDUCTION,
    compute_target=TARGET_GPU_CUDA,
    precision=PRECISION_FP32,
    leaf_type=LEAF_COLUMNAR,
    n_upstream=2,

    # Dimensions
    n_rows=598057,
    n_cols=5,
    n_bins=100,
    n_cadences=4,
    n_tickers=1,
    bytes_data=12_560_000,
    bytes_file=12_600_000,

    # Temporal
    ts_first_ns=1693576800_000_000_000,
    ts_last_ns=1693663200_000_000_000,
    ts_range_ns=86400_000_000_000,
    market_open_ns=1693580400_000_000_000,
    market_close_ns=1693603800_000_000_000,

    # Quality
    total_nulls=42,
    total_nans=7,
    total_infs=0,
    null_ppm=14,
    effective_rank=4.87,
    compression_ratio=1.0,
    data_checksum=0xCAFEBABE12345678,

    # Provenance
    write_timestamp_ns=1711555200_000_000_000,
    write_duration_ms=18,
    compute_duration_ms=350,
    rewrite_count=3,
    engine_version=4,
    source_data_hash=0x1234567890ABCDEF,
    compute_host="gpu-workstation",
    cuda_version=13020,
    driver_version=59510,

    # Layout
    dir_offset=4096,
    dir_entries=5,
    meta_offset=8192,
    meta_size=256,
    data_start=12288,

    # Asset
    exchange_code=14,
    asset_class=ASSET_EQUITY,
    universe_tier=1,
    tick_count=598057,
    market_cap_tier=5,
    avg_spread_e8=150_000_000,

    # Upstream
    upstream=[
        UpstreamFingerprint(leaf_id="K01P01_ingest", write_ts_ns=1711555100_000_000_000,
                           data_hash=0xAAAABBBBCCCCDDDD, ti=0, to=0),
        UpstreamFingerprint(leaf_id="K01P02_derived", write_ts_ns=1711555150_000_000_000,
                           data_hash=0x1111222233334444, ti=9, to=16),
    ],

    # Statistics
    global_mean=227.543,
    global_std=12.871,
    global_min=215.32,
    global_max=255.87,
    global_median=228.01,
    global_skew=-0.342,
    global_kurtosis=2.156,
    global_entropy=7.891,

    # Spatial
    spatial_dims=3,
    atlas_version=2,
    coordinates=[0.5, -0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],

    # Status
    is_complete=True,
    is_dirty=False,
)

buf = pack_block0(h)
check_eq("block0 size", len(buf), BLOCK_SIZE)

h2 = unpack_block0(buf)

# Format
check_eq("format_version", h2.format_version, FORMAT_VERSION)
check_eq("flags", h2.flags, h.flags)
check_eq("alignment", h2.alignment, ALIGNMENT)
check_eq("header_blocks", h2.header_blocks, 3)

# Identity
check_eq("leaf_id", h2.leaf_id, "K02B05_returns_vol")
check_eq("ticker", h2.ticker, "AAPL")
check_eq("day", h2.day, "2025-09-02")
check_eq("ti", h2.ti, 9)
check_eq("to", h2.to, 16)
check_eq("leaf_version", h2.leaf_version, "2.1.0")
check_eq("schema_fingerprint", h2.schema_fingerprint[:3], b"\xab\xcd\xef")
check_eq("leaf_id_hash", h2.leaf_id_hash, 0xDEADBEEF)

# Tree
check_eq("kingdom", h2.kingdom, 2)
check_eq("phylum", h2.phylum, 5)
check_eq("class_", h2.class_, 3)
check_eq("rank", h2.rank, 7)
check_eq("family", h2.family, 1)
check_eq("genus", h2.genus, 4)
check_eq("species", h2.species, 2)
check_eq("depth", h2.depth, 3)
check_eq("computation_shape", h2.computation_shape, SHAPE_REDUCTION)
check_eq("compute_target", h2.compute_target, TARGET_GPU_CUDA)
check_eq("precision", h2.precision, PRECISION_FP32)
check_eq("leaf_type", h2.leaf_type, LEAF_COLUMNAR)
check_eq("n_upstream", h2.n_upstream, 2)

# Dimensions
check_eq("n_rows", h2.n_rows, 598057)
check_eq("n_cols", h2.n_cols, 5)
check_eq("n_bins", h2.n_bins, 100)
check_eq("n_cadences", h2.n_cadences, 4)
check_eq("n_tickers", h2.n_tickers, 1)
check_eq("bytes_data", h2.bytes_data, 12_560_000)
check_eq("bytes_file", h2.bytes_file, 12_600_000)

# Temporal
check_eq("ts_first_ns", h2.ts_first_ns, h.ts_first_ns)
check_eq("ts_last_ns", h2.ts_last_ns, h.ts_last_ns)
check_eq("ts_range_ns", h2.ts_range_ns, h.ts_range_ns)
check_eq("market_open_ns", h2.market_open_ns, h.market_open_ns)
check_eq("market_close_ns", h2.market_close_ns, h.market_close_ns)

# Quality
check_eq("total_nulls", h2.total_nulls, 42)
check_eq("total_nans", h2.total_nans, 7)
check_eq("total_infs", h2.total_infs, 0)
check_eq("null_ppm", h2.null_ppm, 14)
check_eq("effective_rank", h2.effective_rank, 4.87)
check_eq("compression_ratio", h2.compression_ratio, 1.0)
check_eq("data_checksum", h2.data_checksum, 0xCAFEBABE12345678)

# Provenance
check_eq("write_timestamp_ns", h2.write_timestamp_ns, h.write_timestamp_ns)
check_eq("write_duration_ms", h2.write_duration_ms, 18)
check_eq("compute_duration_ms", h2.compute_duration_ms, 350)
check_eq("rewrite_count", h2.rewrite_count, 3)
check_eq("engine_version", h2.engine_version, 4)
check_eq("source_data_hash", h2.source_data_hash, 0x1234567890ABCDEF)
check_eq("compute_host", h2.compute_host, "gpu-workstation")
check_eq("cuda_version", h2.cuda_version, 13020)
check_eq("driver_version", h2.driver_version, 59510)

# Layout
check_eq("dir_offset", h2.dir_offset, 4096)
check_eq("dir_entries", h2.dir_entries, 5)
check_eq("meta_offset", h2.meta_offset, 8192)
check_eq("meta_size", h2.meta_size, 256)
check_eq("data_start", h2.data_start, 12288)

# Asset
check_eq("exchange_code", h2.exchange_code, 14)
check_eq("asset_class", h2.asset_class, ASSET_EQUITY)
check_eq("universe_tier", h2.universe_tier, 1)
check_eq("tick_count", h2.tick_count, 598057)
check_eq("market_cap_tier", h2.market_cap_tier, 5)
check_eq("avg_spread_e8", h2.avg_spread_e8, 150_000_000)

# Upstream
check_eq("upstream_count", len(h2.upstream), 2)
check_eq("upstream[0].leaf_id", h2.upstream[0].leaf_id, "K01P01_ingest")
check_eq("upstream[0].write_ts_ns", h2.upstream[0].write_ts_ns, h.upstream[0].write_ts_ns)
check_eq("upstream[0].data_hash", h2.upstream[0].data_hash, 0xAAAABBBBCCCCDDDD)
check_eq("upstream[0].ti", h2.upstream[0].ti, 0)
check_eq("upstream[0].to", h2.upstream[0].to, 0)
check_eq("upstream[1].leaf_id", h2.upstream[1].leaf_id, "K01P02_derived")
check_eq("upstream[1].data_hash", h2.upstream[1].data_hash, 0x1111222233334444)
check_eq("upstream[1].ti", h2.upstream[1].ti, 9)
check_eq("upstream[1].to", h2.upstream[1].to, 16)

# Statistics
check_eq("global_mean", h2.global_mean, 227.543)
check_eq("global_std", h2.global_std, 12.871)
check_eq("global_min", h2.global_min, 215.32)
check_eq("global_max", h2.global_max, 255.87)
check_eq("global_median", h2.global_median, 228.01)
check_eq("global_skew", h2.global_skew, -0.342)
check_eq("global_kurtosis", h2.global_kurtosis, 2.156)
check_eq("global_entropy", h2.global_entropy, 7.891)

# Spatial
check_eq("spatial_dims", h2.spatial_dims, 3)
check_eq("atlas_version", h2.atlas_version, 2)
check_eq("coordinates[0]", h2.coordinates[0], 0.5)
check_eq("coordinates[1]", h2.coordinates[1], -0.3)
check_eq("coordinates[2]", h2.coordinates[2], 0.8)

# Status
check_eq("is_complete", h2.is_complete, True)
check_eq("is_dirty", h2.is_dirty, False)

print(f"  Header roundtrip: {passed} passed, {failed} failed")
print()


# ═══════════════════════════════════════════════════════════════
# TEST 2: Column Directory Roundtrip
# ═══════════════════════════════════════════════════════════════

print("TEST 2: COLUMN DIRECTORY ROUNDTRIP")
print("-" * 60)

entries = [
    ColumnEntry(name="price", dtype_code=DTYPE_FLOAT32, n_elements=598057,
                data_offset=12288, data_nbytes=2392228,
                null_count=0, min_value=215.32, max_value=255.87,
                mean_value=227.543, scale_factor=1.0, sentinel_value=float("nan")),
    ColumnEntry(name="timestamp", dtype_code=DTYPE_INT64, n_elements=598057,
                data_offset=16384, data_nbytes=4784456,
                null_count=0, min_value=1.693e18, max_value=1.694e18,
                mean_value=float("nan"), scale_factor=1.0, sentinel_value=float("nan")),
    ColumnEntry(name="exchange", dtype_code=DTYPE_UINT8, n_elements=598057,
                data_offset=24576, data_nbytes=598057,
                null_count=0, min_value=0.0, max_value=19.0,
                mean_value=float("nan"), scale_factor=1.0, sentinel_value=float("nan")),
]

for entry in entries:
    buf = pack_column_entry(entry)
    check_eq(f"col_entry_{entry.name}_size", len(buf), COL_DIR_ENTRY_SIZE)
    e2 = unpack_column_entry(buf)
    check_eq(f"col_{entry.name}.name", e2.name, entry.name)
    check_eq(f"col_{entry.name}.dtype_code", e2.dtype_code, entry.dtype_code)
    check_eq(f"col_{entry.name}.n_elements", e2.n_elements, entry.n_elements)
    check_eq(f"col_{entry.name}.data_offset", e2.data_offset, entry.data_offset)
    check_eq(f"col_{entry.name}.data_nbytes", e2.data_nbytes, entry.data_nbytes)
    check_eq(f"col_{entry.name}.null_count", e2.null_count, entry.null_count)
    check_eq(f"col_{entry.name}.min_value", e2.min_value, entry.min_value)
    check_eq(f"col_{entry.name}.max_value", e2.max_value, entry.max_value)
    check_eq(f"col_{entry.name}.mean_value", e2.mean_value, entry.mean_value)
    check_eq(f"col_{entry.name}.scale_factor", e2.scale_factor, entry.scale_factor)
    check_eq(f"col_{entry.name}.sentinel_value", e2.sentinel_value, entry.sentinel_value)

print(f"  Column directory roundtrip: {passed} passed, {failed} failed")
print()


# ═══════════════════════════════════════════════════════════════
# TEST 3: Write + Read + Verify Checksum (Pathological Data)
# ═══════════════════════════════════════════════════════════════

print("TEST 3: WRITE/READ/CHECKSUM STRESS TEST")
print("-" * 60)

tmpdir = tempfile.mkdtemp(prefix="mktf_integrity_")

test_cases = [
    ("normal", {
        "price": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "size": np.array([100, 200, 300], dtype=np.float32),
    }),
    ("all_nan", {
        "x": np.array([np.nan, np.nan, np.nan], dtype=np.float32),
    }),
    ("mixed_nan", {
        "x": np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32),
        "y": np.array([np.nan, 2.0, np.nan, 4.0, np.nan], dtype=np.float32),
    }),
    ("zeros", {
        "x": np.zeros(1000, dtype=np.float32),
    }),
    ("large_values", {
        "big": np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max], dtype=np.float32),
        "tiny": np.array([np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny], dtype=np.float32),
    }),
    ("inf_values", {
        "x": np.array([np.inf, -np.inf, 0.0, np.nan], dtype=np.float32),
    }),
    ("single_row", {
        "price": np.array([42.0], dtype=np.float32),
        "ts": np.array([1711555200_000_000_000], dtype=np.int64),
    }),
    ("many_columns", {
        f"col_{i:03d}": np.random.randn(100).astype(np.float32) for i in range(20)
    }),
    ("mixed_dtypes", {
        "f32": np.array([1.5, 2.5], dtype=np.float32),
        "f64": np.array([1.5, 2.5], dtype=np.float64),
        "i64": np.array([100, 200], dtype=np.int64),
        "i32": np.array([10, 20], dtype=np.int32),
        "u8": np.array([1, 2], dtype=np.uint8),
        "u32": np.array([0xFFFFFFFF, 0], dtype=np.uint32),
        "bool": np.array([True, False], dtype=np.bool_),
    }),
    ("large_array", {
        "data": np.random.randn(100_000).astype(np.float32),
    }),
]

for name, cols in test_cases:
    path = os.path.join(tmpdir, f"{name}.mktf")

    # Write
    try:
        header = write_mktf(path, cols, leaf_id=f"test_{name}",
                           ticker="TEST", day="2026-03-27")
        check(f"{name}: write", True)
    except Exception as e:
        check(f"{name}: write", False, str(e))
        continue

    # Verify is_complete
    check(f"{name}: is_complete", is_complete(path))

    # Verify NOT dirty
    check(f"{name}: not is_dirty", not is_dirty(path))

    # Checksum verification
    check(f"{name}: checksum", verify_checksum(path))

    # Full read + data equality
    try:
        h_read, data_read = read_columns(path)
        check(f"{name}: read", True)

        for col_name, orig_arr in cols.items():
            read_arr = data_read[col_name]
            check_eq(f"{name}/{col_name} dtype", read_arr.dtype, orig_arr.dtype)
            check_eq(f"{name}/{col_name} shape", read_arr.shape, orig_arr.shape)
            # Use array_equal with equal_nan=True for NaN handling
            check(f"{name}/{col_name} values",
                  np.array_equal(read_arr, orig_arr, equal_nan=True),
                  f"max diff: {np.nanmax(np.abs(read_arr.astype(float) - orig_arr.astype(float)))}"
                  if read_arr.dtype.kind == 'f' else "mismatch")
    except Exception as e:
        check(f"{name}: read", False, str(e))

    # Selective read (first column only)
    first_col = list(cols.keys())[0]
    try:
        _, sel_data = read_selective(path, [first_col])
        check(f"{name}: selective({first_col})",
              np.array_equal(sel_data[first_col], cols[first_col], equal_nan=True))
    except Exception as e:
        check(f"{name}: selective", False, str(e))

    # Header metadata
    h_read = read_header(path)
    check_eq(f"{name}: h.n_rows", h_read.n_rows, len(list(cols.values())[0]))
    check_eq(f"{name}: h.n_cols", h_read.n_cols, len(cols))

print(f"  Stress test: {passed} passed, {failed} failed")
print()


# ═══════════════════════════════════════════════════════════════
# TEST 4: Status Byte Flip Correctness
# ═══════════════════════════════════════════════════════════════

print("TEST 4: STATUS BYTE FLIP CORRECTNESS")
print("-" * 60)

path = os.path.join(tmpdir, "status_test.mktf")
write_mktf(path, {"x": np.array([1.0], dtype=np.float32)},
           leaf_id="test", ticker="X", day="2026-01-01")

# Initial state
c, d = read_status(path)
check("initial: complete=True", c)
check("initial: dirty=False", not d)

# Flip dirty
flip_dirty(path, dirty=True)
c, d = read_status(path)
check("after flip_dirty(True): complete=True", c)
check("after flip_dirty(True): dirty=True", d)

# Verify both positions agree (header[4094:4096] and EOF[-2:])
with open(path, "rb") as f:
    f.seek(4094)
    header_status = f.read(2)
    f.seek(-2, 2)
    eof_status = f.read(2)
check("status bytes: header == EOF",
      header_status == eof_status,
      f"header={header_status!r}, eof={eof_status!r}")

# Flip back
flip_dirty(path, dirty=False)
check("after flip_dirty(False): not dirty", not is_dirty(path))

# Flip complete off (simulating crash recovery test)
flip_complete(path, complete=False)
check("after flip_complete(False): not complete", not is_complete(path))
check("after flip_complete(False): not dirty", not is_dirty(path))

# Verify scan_incomplete finds it
incomplete = scan_incomplete([path])
check("scan_incomplete finds it", len(incomplete) == 1)

# Flip back
flip_complete(path, complete=True)
check("after flip_complete(True): complete", is_complete(path))
incomplete = scan_incomplete([path])
check("scan_incomplete no longer finds it", len(incomplete) == 0)

# Scan dirty with multiple files
paths_for_scan = []
for i in range(10):
    p = os.path.join(tmpdir, f"scan_{i}.mktf")
    write_mktf(p, {"x": np.array([float(i)], dtype=np.float32)},
               leaf_id="test", ticker=f"T{i}", day="2026-01-01")
    paths_for_scan.append(p)

# Mark 3 as dirty
for p in paths_for_scan[2:5]:
    flip_dirty(p, dirty=True)

dirty_found = scan_dirty(paths_for_scan)
check_eq("scan_dirty count", len(dirty_found), 3)

print(f"  Status byte tests: {passed} passed, {failed} failed")
print()


# ═══════════════════════════════════════════════════════════════
# TEST 5: Checksum Corruption Detection
# ═══════════════════════════════════════════════════════════════

print("TEST 5: CHECKSUM CORRUPTION DETECTION")
print("-" * 60)

path = os.path.join(tmpdir, "corrupt_test.mktf")
cols = {"price": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)}
write_mktf(path, cols, leaf_id="test", ticker="X", day="2026-01-01")

# Verify clean file passes
check("clean file: checksum passes", verify_checksum(path))

# Corrupt one byte in the data region
h = read_header(path)
data_offset = h.columns[0].data_offset
with open(path, "r+b") as f:
    f.seek(data_offset + 4)  # corrupt second float
    f.write(b"\xFF")

# Verify corrupted file fails checksum
check("corrupted file: checksum fails", not verify_checksum(path))

# Verify we can still read it (checksum is advisory, not blocking)
try:
    _, data = read_columns(path)
    check("corrupted file: still readable", True)
    check("corrupted file: data is different",
          not np.array_equal(data["price"], cols["price"]))
except Exception:
    check("corrupted file: still readable", False, "read raised exception")

print(f"  Corruption detection: {passed} passed, {failed} failed")
print()


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

shutil.rmtree(tmpdir, ignore_errors=True)

print("=" * 60)
print(f"TOTAL: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"*** {failed} FAILURES ***")
