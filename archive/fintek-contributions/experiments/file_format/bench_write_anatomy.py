"""Experiment 14: Micro-Anatomy of a v4 Write.

Experiment 12 dissected reads (97% I/O-bound, Python not the bottleneck).
Now dissect writes. v4 writes are 4x slower than v3 due to crash safety:
  - 2× fsync (NVMe flush-to-media)
  - SHA-256 hash of data region
  - arr.tobytes() copy per column
  - Atomic rename (.tmp → .mktf)

Critical question: what happens for SMALL files?
30-min cadence bins have ~13 rows × 5 cols = ~200 bytes of data.
But the header is always 4096 bytes and fsync is per-file.
If fsync dominates, the cadence grid cost is per-FILE, not per-BYTE.

Questions:
  1. What fraction of write time is fsync vs hashing vs data copy?
  2. How does write cost scale with file size? (Linear? Constant?)
  3. What's the fsync floor for an empty file?
  4. What would a no-fsync writer cost? (for batch recompute where crash
     recovery = full recompute)
  5. Universe projections for cadence grid writes.
"""

from __future__ import annotations

import gc
import hashlib
import io
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.format import (
    ALIGNMENT, BLOCK_SIZE, COL_DIR_ENTRY_SIZE, NUMPY_TO_DTYPE,
    ColumnEntry, MKTFHeader, UpstreamFingerprint,
    _align, _pad, compute_data_hash, compute_leaf_id_hash,
    compute_schema_fingerprint, pack_block0, pack_column_entry,
)
from trunk.backends.mktf.writer import write_mktf as write_v4
from trunk.backends.mktf.reader import read_columns as read_columns_v4

sys.path.insert(0, str(Path("R:/winrapids/research/20260327-mktf-format")))
from mktf_v3 import AAPL_PATH, COL_MAP, CONDITION_BITS


def load_source_columns() -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(AAPL_PATH))
    raw = {new: tbl.column(old).to_numpy() for old, new in COL_MAP.items()}
    n = len(raw["price"])
    cols = {}
    cols["price"] = raw["price"].astype(np.float32)
    cols["size"] = raw["size"].astype(np.float32)
    cols["timestamp"] = raw["timestamp"].astype(np.int64)
    cols["exchange"] = raw["exchange"].astype(np.uint8)
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
    return cols


def make_cadence_cols(source: dict[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
    """Simulate a cadence bin by taking first n_rows from source."""
    return {name: arr[:n_rows].copy() for name, arr in source.items()}


def timed_us(fn, n_runs: int) -> dict:
    gc.disable()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e3)
    gc.enable()
    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.median(arr)),
    }


def timed_ms(fn, n_runs: int) -> dict:
    r = timed_us(fn, n_runs)
    return {k: v / 1000 for k, v in r.items()}


def fmt_us(s: dict) -> str:
    return f"{s['mean']:.0f}us ±{s['std']:.0f}"

def fmt_ms(s: dict) -> str:
    return f"{s['mean']:.2f}ms ±{s['std']:.2f}"


def main():
    N = 20  # runs per benchmark (writes are slow, keep reasonable)

    print("=" * 78)
    print("EXPERIMENT 14: MICRO-ANATOMY OF A v4 WRITE")
    print("=" * 78)
    print(f"Runs per benchmark: {N}")
    print()

    source_cols = load_source_columns()
    n_source = len(source_cols["price"])
    tmpdir = tempfile.mkdtemp(prefix="mktf_wanat_")
    test_path = os.path.join(tmpdir, "test.mktf")
    tmp_path = test_path + ".tmp"

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Decompose full-size write (598K rows, 12.6 MB)
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1: DECOMPOSE FULL-SIZE WRITE (598K rows, 12.6 MB)")
    print("-" * 60)

    # Pre-compute what the writer computes
    col_names = list(source_cols.keys())
    col_arrays = [arr.ravel() for arr in source_cols.values()]

    # 1a: pack_block0 (CPU — build 4096-byte header)
    header = MKTFHeader(leaf_id="K01P01", ticker="AAPL", day="2025-09-02")
    header.n_rows = n_source
    header.n_cols = len(col_names)
    header.leaf_id_hash = compute_leaf_id_hash("K01P01")
    header.schema_fingerprint = compute_schema_fingerprint(
        [(name, NUMPY_TO_DTYPE[arr.dtype]) for name, arr in source_cols.items()])
    stats_pack = timed_us(lambda: pack_block0(header), N)
    block0 = pack_block0(header)
    print(f"  pack_block0 (CPU):          {fmt_us(stats_pack)}")

    # 1b: pack_column_entry × 5 (CPU)
    entries = []
    for name, arr in source_cols.items():
        entries.append(ColumnEntry(
            name=name, dtype_code=NUMPY_TO_DTYPE[arr.dtype],
            n_elements=len(arr), data_offset=0, data_nbytes=arr.nbytes))
    stats_pack_dir = timed_us(lambda: [pack_column_entry(e) for e in entries], N)
    print(f"  pack_column_entry ×5 (CPU): {fmt_us(stats_pack_dir)}")

    # 1c: arr.tobytes() for all columns (memory copy)
    stats_tobytes = timed_us(lambda: [arr.tobytes() for arr in col_arrays], N)
    raw_bytes = [arr.tobytes() for arr in col_arrays]
    total_data = sum(len(b) for b in raw_bytes)
    print(f"  arr.tobytes() ×5 (copy):    {fmt_us(stats_tobytes)}  ({total_data/1e6:.2f} MB)")

    # 1d: SHA-256 hash of all data (CPU)
    def hash_all():
        h = hashlib.sha256()
        for b in raw_bytes:
            h.update(b)
        return struct.unpack("<Q", h.digest()[:8])[0]
    stats_hash = timed_us(hash_all, N)
    print(f"  SHA-256 hash (CPU):         {fmt_us(stats_hash)}")

    # 1e: Raw write (no fsync, no hash — just file I/O)
    def raw_write():
        with open(tmp_path, "wb") as f:
            f.write(block0)
            for e in entries:
                f.write(pack_column_entry(e))
            for b in raw_bytes:
                f.write(b)
            f.write(b"\x01\x00")
    stats_raw_write = timed_us(raw_write, N)
    print(f"  raw write (no fsync):       {fmt_us(stats_raw_write)}")

    # 1f: Single fsync
    def write_and_fsync():
        with open(tmp_path, "wb") as f:
            f.write(block0)
            for b in raw_bytes:
                f.write(b)
            f.flush()
            os.fsync(f.fileno())
    stats_1sync = timed_us(write_and_fsync, N)
    fsync_cost = stats_1sync["mean"] - stats_raw_write["mean"]
    print(f"  write + 1× fsync:           {fmt_us(stats_1sync)}  (fsync ≈ {fsync_cost:.0f}us)")

    # 1g: Double fsync (as in production)
    def write_and_2sync():
        with open(tmp_path, "wb") as f:
            f.write(block0)
            for b in raw_bytes:
                f.write(b)
            f.flush()
            os.fsync(f.fileno())
            # Simulate finalization
            f.seek(0)
            f.write(b"\x01")  # flip a byte
            f.flush()
            os.fsync(f.fileno())
    stats_2sync = timed_us(write_and_2sync, N)
    print(f"  write + 2× fsync:           {fmt_us(stats_2sync)}")

    # 1h: Atomic rename
    def just_rename():
        # Create a dummy tmp file first
        with open(tmp_path, "wb") as f:
            f.write(b"x")
        final = os.path.join(tmpdir, "rename_test.mktf")
        if os.path.exists(final):
            os.unlink(final)
        os.rename(tmp_path, final)
    stats_rename = timed_us(just_rename, N)
    print(f"  atomic rename:              {fmt_us(stats_rename)}")

    # 1i: Full production write (baseline)
    stats_full = timed_us(
        lambda: write_v4(test_path, source_cols, leaf_id="K01P01",
                         ticker="AAPL", day="2025-09-02"),
        N)
    print(f"  ---")
    print(f"  FULL write_mktf():          {fmt_us(stats_full)}")

    # Budget
    total = stats_full["mean"]
    cpu_total = stats_pack["mean"] + stats_pack_dir["mean"] + stats_tobytes["mean"] + stats_hash["mean"]
    io_total = stats_raw_write["mean"]
    fsync_total = stats_2sync["mean"] - stats_raw_write["mean"]
    rename_total = stats_rename["mean"]
    accounted = cpu_total + io_total + fsync_total + rename_total

    print()
    print(f"  TIME BUDGET (% of {total/1000:.1f}ms):")
    print(f"    CPU (pack+tobytes+hash): {cpu_total:.0f}us ({cpu_total/total*100:.0f}%)")
    print(f"    I/O (raw write):         {io_total:.0f}us ({io_total/total*100:.0f}%)")
    print(f"    fsync (2×):              {fsync_total:.0f}us ({fsync_total/total*100:.0f}%)")
    print(f"    rename:                  {rename_total:.0f}us ({rename_total/total*100:.0f}%)")
    print(f"    Unaccounted:             {total - accounted:.0f}us ({(total-accounted)/total*100:.0f}%)")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Write cost vs file size (scaling)
    # ═══════════════════════════════════════════════════════════
    print("PHASE 2: WRITE COST vs FILE SIZE")
    print("-" * 60)

    sizes = [
        ("30min cadence (13 rows)", 13),
        ("5min cadence (78 rows)", 78),
        ("1min cadence (390 rows)", 390),
        ("10s cadence (2340 rows)", 2340),
        ("1s cadence (23400 rows)", 23400),
        ("Full AAPL (598K rows)", n_source),
    ]

    print(f"  {'Size':<28} {'Rows':>8} {'Data':>8} {'File':>8} {'Write':>12} {'ms/KB':>8}")
    print(f"  {'-'*72}")

    size_results = []
    for label, n_rows in sizes:
        cols_sub = make_cadence_cols(source_cols, n_rows)
        data_bytes = sum(arr.nbytes for arr in cols_sub.values())

        p = os.path.join(tmpdir, f"size_{n_rows}.mktf")
        # Warmup
        write_v4(p, cols_sub, leaf_id="K01P01", ticker="AAPL", day="2025-09-02")
        file_bytes = os.path.getsize(p)

        stats = timed_us(
            lambda p=p, c=cols_sub: write_v4(p, c, leaf_id="K01P01",
                                              ticker="AAPL", day="2025-09-02"),
            N)

        ms_per_kb = stats["mean"] / 1000 / (file_bytes / 1024)
        size_results.append((label, n_rows, data_bytes, file_bytes, stats))
        print(f"  {label:<28} {n_rows:>8} {data_bytes/1024:.1f}KB {file_bytes/1024:.1f}KB "
              f"{stats['mean']/1000:>9.2f}ms {ms_per_kb:>6.2f}")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: fsync cost isolation for small files
    # ═══════════════════════════════════════════════════════════
    print("PHASE 3: fsync COST ISOLATION (small files)")
    print("-" * 60)

    # Tiny file: 13 rows, ~200 bytes data
    tiny_cols = make_cadence_cols(source_cols, 13)
    tiny_bytes = [arr.tobytes() for arr in tiny_cols.values()]

    # Tiny: no fsync
    def tiny_no_sync():
        with open(tmp_path, "wb") as f:
            f.write(block0)
            for b in tiny_bytes:
                f.write(b)
    stats_tiny_nosync = timed_us(tiny_no_sync, N * 2)

    # Tiny: 1 fsync
    def tiny_1sync():
        with open(tmp_path, "wb") as f:
            f.write(block0)
            for b in tiny_bytes:
                f.write(b)
            f.flush()
            os.fsync(f.fileno())
    stats_tiny_1sync = timed_us(tiny_1sync, N * 2)

    # Tiny: 2 fsync (production)
    def tiny_2sync():
        with open(tmp_path, "wb") as f:
            f.write(block0)
            for b in tiny_bytes:
                f.write(b)
            f.flush()
            os.fsync(f.fileno())
            f.seek(0)
            f.write(b"\x01")
            f.flush()
            os.fsync(f.fileno())
    stats_tiny_2sync = timed_us(tiny_2sync, N * 2)

    # Tiny: full production
    tiny_path = os.path.join(tmpdir, "tiny.mktf")
    stats_tiny_full = timed_us(
        lambda: write_v4(tiny_path, tiny_cols, leaf_id="K01P01",
                         ticker="AAPL", day="2025-09-02"),
        N * 2)

    tiny_data_size = sum(len(b) for b in tiny_bytes)
    print(f"  Data: {tiny_data_size} bytes (13 rows × 5 cols)")
    print()
    print(f"  no fsync:        {fmt_us(stats_tiny_nosync)}")
    print(f"  1× fsync:        {fmt_us(stats_tiny_1sync)}")
    print(f"  2× fsync:        {fmt_us(stats_tiny_2sync)}")
    print(f"  full production: {fmt_us(stats_tiny_full)}")
    print()

    fsync_1_cost = stats_tiny_1sync["mean"] - stats_tiny_nosync["mean"]
    fsync_2_cost = stats_tiny_2sync["mean"] - stats_tiny_nosync["mean"]
    print(f"  Per-fsync cost (tiny file): {fsync_1_cost:.0f}us = {fsync_1_cost/1000:.2f}ms")
    print(f"  2× fsync cost:              {fsync_2_cost:.0f}us = {fsync_2_cost/1000:.2f}ms")
    print(f"  fsync as % of tiny write:   {fsync_2_cost/stats_tiny_full['mean']*100:.0f}%")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: No-fsync writer (batch recompute mode)
    # ═══════════════════════════════════════════════════════════
    print("PHASE 4: NO-FSYNC WRITER (batch recompute mode)")
    print("-" * 60)
    print("If crash recovery = full recompute, fsync is unnecessary.")
    print()

    # Implement a no-fsync writer
    def write_no_fsync(path_str, columns):
        """Minimal writer: header + dir + data + status. No fsync, no rename."""
        p = Path(path_str)
        col_names = list(columns.keys())
        col_arrays = [arr.ravel() for arr in columns.values()]
        n_elem = len(col_arrays[0])
        n_cols = len(col_names)

        # Layout
        dir_offset = BLOCK_SIZE
        dir_end = dir_offset + n_cols * COL_DIR_ENTRY_SIZE
        data_start = _align(dir_end)
        current = data_start
        col_entries = []
        for name, arr in zip(col_names, col_arrays):
            dtype_code = NUMPY_TO_DTYPE[arr.dtype]
            nbytes = arr.nbytes
            col_entries.append(ColumnEntry(
                name=name, dtype_code=dtype_code,
                n_elements=len(arr), data_offset=current, data_nbytes=nbytes))
            current = _align(current + nbytes)

        # Header
        h = MKTFHeader()
        h.leaf_id = "K01P01"
        h.ticker = "AAPL"
        h.day = "2025-09-02"
        h.n_rows = n_elem
        h.n_cols = n_cols
        h.dir_offset = dir_offset
        h.dir_entries = n_cols
        h.data_start = data_start
        h.is_complete = True
        h.is_dirty = False

        with open(p, "wb") as f:
            f.write(pack_block0(h))
            for e in col_entries:
                f.write(pack_column_entry(e))
            pad = data_start - dir_end
            if pad > 0:
                f.write(b"\x00" * pad)
            for arr in col_arrays:
                f.write(arr.tobytes())
                # No alignment padding between columns for speed
            f.write(b"\x01\x00")

    # Benchmark no-fsync for various sizes
    print(f"  {'Size':<28} {'Production':>12} {'No-fsync':>12} {'Speedup':>10}")
    print(f"  {'-'*62}")

    for label, n_rows in sizes:
        cols_sub = make_cadence_cols(source_cols, n_rows)
        p_prod = os.path.join(tmpdir, f"prod_{n_rows}.mktf")
        p_fast = os.path.join(tmpdir, f"fast_{n_rows}.mktf")

        s_prod = timed_us(
            lambda p=p_prod, c=cols_sub: write_v4(p, c, leaf_id="K01P01",
                                                   ticker="AAPL", day="2025-09-02"),
            N)
        s_fast = timed_us(
            lambda p=p_fast, c=cols_sub: write_no_fsync(p, c),
            N)
        speedup = s_prod["mean"] / s_fast["mean"]
        print(f"  {label:<28} {s_prod['mean']/1000:>9.2f}ms {s_fast['mean']/1000:>9.2f}ms {speedup:>8.1f}x")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Universe projections for cadence grid writes
    # ═══════════════════════════════════════════════════════════
    print("PHASE 5: CADENCE GRID WRITE PROJECTIONS")
    print("-" * 60)

    n_tickers = 4604
    # Cadence grid: 1s, 5s, 10s, 30s, 1min, 5min, 10min, 15min, 20min, 30min
    cadences = [
        ("1s", 23400),
        ("5s", 4680),
        ("10s", 2340),
        ("30s", 780),
        ("1min", 390),
        ("5min", 78),
        ("10min", 39),
        ("15min", 26),
        ("20min", 20),
        ("30min", 13),
    ]

    total_production_ms = 0
    total_nofsync_ms = 0

    print(f"  {'Cadence':<10} {'Rows':>6} {'Prod(ms)':>10} {'No-fsync':>10} {'×4604':>10} {'×4604 nofs':>10}")
    print(f"  {'-'*56}")

    for cad_label, cad_rows in cadences:
        cols_sub = make_cadence_cols(source_cols, min(cad_rows, n_source))
        p_prod = os.path.join(tmpdir, f"cad_prod_{cad_rows}.mktf")
        p_fast = os.path.join(tmpdir, f"cad_fast_{cad_rows}.mktf")

        s_prod = timed_us(
            lambda p=p_prod, c=cols_sub: write_v4(p, c, leaf_id="K01P01",
                                                   ticker="AAPL", day="2025-09-02"),
            max(N // 2, 5))
        s_fast = timed_us(
            lambda p=p_fast, c=cols_sub: write_no_fsync(p, c),
            max(N // 2, 5))

        prod_ms = s_prod["mean"] / 1000
        fast_ms = s_fast["mean"] / 1000
        total_production_ms += prod_ms * n_tickers
        total_nofsync_ms += fast_ms * n_tickers

        print(f"  {cad_label:<10} {cad_rows:>6} {prod_ms:>8.2f}ms {fast_ms:>8.2f}ms "
              f"{prod_ms*n_tickers/1000:>8.1f}s {fast_ms*n_tickers/1000:>8.1f}s")

    print(f"  {'-'*56}")
    print(f"  {'TOTAL':<10} {'':>6} {'':>10} {'':>10} "
          f"{total_production_ms/1000:>8.1f}s {total_nofsync_ms/1000:>8.1f}s")
    print()
    print(f"  Production cadence grid write: {total_production_ms/1000:.0f}s = {total_production_ms/60000:.1f}min")
    print(f"  No-fsync cadence grid write:   {total_nofsync_ms/1000:.0f}s = {total_nofsync_ms/60000:.1f}min")
    print(f"  Savings from no-fsync:         {(1 - total_nofsync_ms/total_production_ms)*100:.0f}%")
    print()

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
