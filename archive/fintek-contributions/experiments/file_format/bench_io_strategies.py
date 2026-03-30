"""Experiment 13: I/O Strategy Sweep for Universe Scans.

Experiment 12 found the Python reader is 97% I/O-bound, with Windows kernel
capping us at 69% NVMe (~4.86 GB/s vs ~7.0 GB/s theoretical).

This experiment explores I/O strategies to close that gap:
  1. Python buffered (open/read) vs raw (os.open/os.read) vs msvcrt
  2. Pre-opened file handle pool (avoid per-ticker open/close overhead)
  3. Multi-file sequential scan (universe-like workload, 100 files)
  4. mmap with forced page faults (true cost, not lazy)
  5. ReadFile via ctypes (minimal Python wrapper around Win32 API)

Questions:
  1. How much of the 31% NVMe gap is Python buffered I/O overhead?
  2. Does a file handle pool help for universe scans?
  3. Is mmap (with real faults) competitive with read()?
  4. What's the best strategy for 4604-file sequential scan?
"""

from __future__ import annotations

import ctypes
import gc
import io
import mmap
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.format import (
    BLOCK_SIZE, COL_DIR_ENTRY_SIZE, DTYPE_TO_NUMPY,
    UpstreamFingerprint, unpack_block0, unpack_column_entry,
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
    gc.disable()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    gc.enable()
    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.median(arr)),
    }


def fmt_us(s: dict) -> str:
    return f"{s['mean']:.1f}us ±{s['std']:.1f}"

def fmt_ms(s: dict) -> str:
    return f"{s['mean']:.2f}ms ±{s['std']:.2f}"


def main():
    N_SINGLE = 50
    N_BATCH = 10
    N_FILES = 100

    print("=" * 78)
    print("EXPERIMENT 13: I/O STRATEGY SWEEP FOR UNIVERSE SCANS")
    print("=" * 78)
    print()

    cols = load_source_columns()
    tmpdir = tempfile.mkdtemp(prefix="mktf_io_")

    # Create test files
    paths = []
    for i in range(N_FILES):
        p = os.path.join(tmpdir, f"ticker_{i:04d}.mktf")
        write_v4(p, cols, leaf_id="K01P01", ticker=f"T{i:04d}", day="2025-09-02")
        paths.append(p)

    file_size = os.path.getsize(paths[0])
    total_bytes = file_size * N_FILES
    print(f"Files: {N_FILES} × {file_size/1e6:.2f} MB = {total_bytes/1e6:.0f} MB total")
    print()

    # Warmup all paths
    for p in paths:
        read_columns_v4(p)

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Single-file I/O strategy comparison
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1: SINGLE-FILE READ STRATEGIES")
    print("-" * 60)

    test_path = paths[0]

    # 1a: Python buffered (production)
    def py_buffered():
        with open(test_path, "rb") as f:
            return f.read()
    stats_buffered = timed_us(py_buffered, N_SINGLE)

    # 1b: Python raw (os.open/os.read)
    def py_raw():
        fd = os.open(test_path, os.O_RDONLY | os.O_BINARY)
        try:
            data = os.read(fd, file_size)
        finally:
            os.close(fd)
        return data
    stats_raw = timed_us(py_raw, N_SINGLE)

    # 1c: os.open + os.read with O_SEQUENTIAL hint
    def py_raw_seq():
        fd = os.open(test_path, os.O_RDONLY | os.O_BINARY | os.O_SEQUENTIAL)
        try:
            data = os.read(fd, file_size)
        finally:
            os.close(fd)
        return data
    stats_raw_seq = timed_us(py_raw_seq, N_SINGLE)

    # 1d: mmap (forced fault — sum to force all pages in)
    def mmap_faulted():
        with open(test_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Force all pages: read every 4096th byte
        total = 0
        for off in range(0, len(mm), 4096):
            total += mm[off]
        mm.close()
        return total
    stats_mmap_fault = timed_us(mmap_faulted, N_SINGLE)

    # 1e: mmap lazy (no fault — just map + header)
    def mmap_lazy():
        with open(test_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        _ = mm[:BLOCK_SIZE]  # just touch header
        mm.close()
    stats_mmap_lazy = timed_us(mmap_lazy, N_SINGLE)

    # 1f: Pre-opened file handle (seek to 0, read)
    preopen_f = open(test_path, "rb")
    def preopen_read():
        preopen_f.seek(0)
        return preopen_f.read()
    stats_preopen = timed_us(preopen_read, N_SINGLE)

    # 1g: Pre-opened raw fd
    preopen_fd = os.open(test_path, os.O_RDONLY | os.O_BINARY)
    def preopen_raw():
        os.lseek(preopen_fd, 0, os.SEEK_SET)
        return os.read(preopen_fd, file_size)
    stats_preopen_raw = timed_us(preopen_raw, N_SINGLE)

    print(f"  {'Strategy':<42} {'Mean':>14} {'p50':>10} {'Min':>10} {'BW':>10}")
    print(f"  {'-'*86}")
    strategies = [
        ("Python buffered (open+read+close)", stats_buffered),
        ("Python raw (os.open+read+close)", stats_raw),
        ("Raw + O_SEQUENTIAL hint", stats_raw_seq),
        ("mmap (forced fault, all pages)", stats_mmap_fault),
        ("mmap (lazy, header only)", stats_mmap_lazy),
        ("Pre-opened buffered (seek+read)", stats_preopen),
        ("Pre-opened raw fd (lseek+read)", stats_preopen_raw),
    ]
    for name, s in strategies:
        bw = file_size / (s["mean"] / 1e6) / 1e9
        print(f"  {name:<42} {fmt_us(s):>14} {s['p50']:>8.0f}us {s['min']:>8.0f}us {bw:>7.2f} GB/s")

    preopen_f.close()
    os.close(preopen_fd)
    print()

    # Open vs seek cost
    open_overhead = stats_buffered["mean"] - stats_preopen["mean"]
    print(f"  File open overhead: {open_overhead:.0f}us ({open_overhead/stats_buffered['mean']*100:.0f}% of read)")
    print(f"  Raw vs buffered:    {stats_raw['mean'] - stats_buffered['mean']:.0f}us difference")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Multi-file universe scan strategies
    # ═══════════════════════════════════════════════════════════
    print(f"PHASE 2: UNIVERSE SCAN ({N_FILES} files, {total_bytes/1e6:.0f} MB)")
    print("-" * 60)

    # 2a: Production sequential (open/read/close each file)
    def scan_production():
        results = []
        for p in paths:
            h, c = read_columns_v4(p)
            results.append(c)
        return results
    stats_scan_prod = timed_ms(scan_production, N_BATCH)

    # 2b: Pre-opened handle pool
    handles = [open(p, "rb") for p in paths]
    def scan_preopen():
        results = []
        for f in handles:
            f.seek(0)
            data = f.read()
            h = unpack_block0(data[:BLOCK_SIZE])
            if h.dir_entries > 0:
                for i in range(h.dir_entries):
                    off = h.dir_offset + i * COL_DIR_ENTRY_SIZE
                    entry = unpack_column_entry(data, off)
                    h.columns.append(entry)
            cols_out = {}
            for entry in h.columns:
                np_dtype = DTYPE_TO_NUMPY[entry.dtype_code]
                cols_out[entry.name] = np.frombuffer(
                    data[entry.data_offset:entry.data_offset + entry.data_nbytes],
                    dtype=np_dtype)
            results.append(cols_out)
        return results
    stats_scan_preopen = timed_ms(scan_preopen, N_BATCH)
    for f in handles:
        f.close()

    # 2c: Raw fd pool
    fds = [os.open(p, os.O_RDONLY | os.O_BINARY) for p in paths]
    def scan_raw_pool():
        results = []
        for fd in fds:
            os.lseek(fd, 0, os.SEEK_SET)
            data = os.read(fd, file_size)
            h = unpack_block0(data[:BLOCK_SIZE])
            if h.dir_entries > 0:
                for i in range(h.dir_entries):
                    off = h.dir_offset + i * COL_DIR_ENTRY_SIZE
                    entry = unpack_column_entry(data, off)
                    h.columns.append(entry)
            cols_out = {}
            for entry in h.columns:
                np_dtype = DTYPE_TO_NUMPY[entry.dtype_code]
                cols_out[entry.name] = np.frombuffer(
                    data[entry.data_offset:entry.data_offset + entry.data_nbytes],
                    dtype=np_dtype)
            results.append(cols_out)
        return results
    stats_scan_raw_pool = timed_ms(scan_raw_pool, N_BATCH)
    for fd in fds:
        os.close(fd)

    # 2d: mmap pool (pre-mapped, parse on access)
    mmaps = []
    for p in paths:
        f = open(p, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mmaps.append((f, mm))

    def scan_mmap_pool():
        results = []
        for _, mm in mmaps:
            h = unpack_block0(mm[:BLOCK_SIZE])
            if h.dir_entries > 0:
                for i in range(h.dir_entries):
                    off = h.dir_offset + i * COL_DIR_ENTRY_SIZE
                    entry = unpack_column_entry(mm, off)
                    h.columns.append(entry)
            cols_out = {}
            for entry in h.columns:
                np_dtype = DTYPE_TO_NUMPY[entry.dtype_code]
                cols_out[entry.name] = np.frombuffer(
                    mm, dtype=np_dtype, count=entry.n_elements, offset=entry.data_offset)
            results.append(cols_out)
        return results
    stats_scan_mmap = timed_ms(scan_mmap_pool, N_BATCH)

    for f, mm in mmaps:
        mm.close()
        f.close()

    # 2e: Threaded (8 workers, production reader)
    def scan_threaded():
        with ThreadPoolExecutor(max_workers=8) as exe:
            return list(exe.map(read_columns_v4, paths))
    stats_scan_threaded = timed_ms(scan_threaded, N_BATCH)

    print(f"  {'Strategy':<42} {'Mean':>10} {'p50':>10} {'BW (GB/s)':>10} {'Speedup':>10}")
    print(f"  {'-'*82}")
    baseline_ms = stats_scan_prod["mean"]
    scan_strategies = [
        ("Production (open/read/close each)", stats_scan_prod),
        ("Pre-opened handle pool", stats_scan_preopen),
        ("Raw fd pool (os.open)", stats_scan_raw_pool),
        ("mmap pool (pre-mapped)", stats_scan_mmap),
        ("Threaded 8w (production)", stats_scan_threaded),
    ]
    for name, s in scan_strategies:
        bw = total_bytes / (s["mean"] / 1000) / 1e9
        speedup = baseline_ms / s["mean"]
        print(f"  {name:<42} {fmt_ms(s):>16} {s['p50']:>7.1f}ms {bw:>8.2f} {speedup:>9.2f}x")

    print()
    # Per-file cost
    for name, s in scan_strategies:
        per_file = s["mean"] / N_FILES
        print(f"  {name:<42} {per_file:.3f} ms/file")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Universe Projections
    # ═══════════════════════════════════════════════════════════
    print("PHASE 3: UNIVERSE PROJECTIONS (4604 tickers)")
    print("-" * 60)
    n_tickers = 4604

    print(f"  {'Strategy':<42} {'Time':>10} {'BW (GB/s)':>10}")
    print(f"  {'-'*62}")
    for name, s in scan_strategies:
        per_file_ms = s["mean"] / N_FILES
        total_s = per_file_ms * n_tickers / 1000
        bw = file_size * n_tickers / total_s / 1e9
        print(f"  {name:<42} {total_s:>8.1f}s {bw:>8.2f}")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Handle Pool Scaling
    # ═══════════════════════════════════════════════════════════
    print("PHASE 4: CAN WINDOWS HOLD 4604 FILE HANDLES?")
    print("-" * 60)

    # Test opening many handles simultaneously
    max_test = min(N_FILES, 100)
    try:
        many_handles = []
        t0 = time.perf_counter_ns()
        for p in paths[:max_test]:
            many_handles.append(open(p, "rb"))
        open_time = (time.perf_counter_ns() - t0) / 1e3
        print(f"  Opened {max_test} handles in {open_time:.0f}us ({open_time/max_test:.1f}us/handle)")

        # Read through all handles
        t0 = time.perf_counter_ns()
        for f in many_handles:
            _ = f.read()
        read_time = (time.perf_counter_ns() - t0) / 1e3
        print(f"  Read {max_test} files via pool: {read_time:.0f}us ({read_time/max_test:.0f}us/file)")

        # Re-read (seek back)
        t0 = time.perf_counter_ns()
        for f in many_handles:
            f.seek(0)
            _ = f.read()
        reread_time = (time.perf_counter_ns() - t0) / 1e3
        print(f"  Re-read (seek+read): {reread_time:.0f}us ({reread_time/max_test:.0f}us/file)")

        savings = (read_time - reread_time) / read_time * 100
        print(f"  Pool re-read savings: {savings:+.1f}%")

        for f in many_handles:
            f.close()

        # Project to 4604
        per_file_pool = reread_time / max_test
        proj_s = per_file_pool * n_tickers / 1e6
        bw = file_size * n_tickers / (proj_s) / 1e9
        print(f"  Projected 4604 handles: {proj_s:.1f}s ({bw:.2f} GB/s)")
        print(f"  Memory cost: 4604 handles × ~8 KB kernel = ~36 MB kernel memory")

    except OSError as e:
        print(f"  FAILED at {len(many_handles)} handles: {e}")

    print()

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
