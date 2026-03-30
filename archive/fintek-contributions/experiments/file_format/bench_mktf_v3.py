"""Observer Experiment 6: MKTF v3 Benchmark — self-describing, 4096-aligned.

Tests:
1. Read/write speed with statistical significance
2. 4096 vs 64-byte alignment overhead (file size + speed)
3. Header-only scan (daemon simulation for 4604 tickers)
4. Full GPU pipeline: source-only (5 cols) + fused recompute
5. Column-selective reads
6. Cold vs warm cache
7. Crash recovery: is_complete flag validation
"""
from __future__ import annotations

import gc
import io
import os
import sys
import tempfile
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "research", "20260327-mktf-format"))

import numpy as np
import cupy as cp

from mktf_v3 import (
    write_mktf, read_data, read_selective, read_header, read_metadata,
    load_aapl, SECTOR, MKTFHeader,
)

# Import v2 for comparison
from mktf_v2 import (
    write_mktf as write_v2, read_mktf as read_v2,
    read_mktf_selective as read_v2_selective,
    load_and_encode,
)

# Reuse the fused kernel from experiment 5
from bench_gpu_precision import (
    FUSED_POINTWISE_FP32, run_fused_fp32, OUTPUT_NAMES,
)


def stats(times):
    s = sorted(times)
    n = len(s)
    m = sum(s) / n
    sd = (sum((x - m) ** 2 for x in s) / max(n - 1, 1)) ** 0.5
    return {"mean": m, "std": sd, "min": s[0], "max": s[-1],
            "p50": s[n // 2], "p95": s[int(n * 0.95)], "n": n}


def fmt(st):
    return f"{st['mean']:7.2f}ms +/- {st['std']:5.2f}  [p50={st['p50']:.2f}, min={st['min']:.2f}]"


def time_fn(fn, n_runs=30, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        gc.enable()
        times.append((t1 - t0) / 1e6)
    return times


def evict_cache():
    try:
        arr = np.ones(2048 * 1024 * 1024 // 8, dtype=np.float64)
        _ = arr.sum()
        del arr
        gc.collect()
    except MemoryError:
        pass


# ══════════════════════════════════════════════════════════════════
# PHASE 1: v3 Basic I/O
# ══════════════════════════════════════════════════════════════════

def phase1():
    print("=" * 80)
    print("PHASE 1: MKTF v3 Basic I/O (Source-Only, 4096-aligned)")
    print("=" * 80)
    print()

    cols, meta = load_aapl()
    n = len(cols["price"])
    data_mb = sum(a.nbytes for a in cols.values()) / 1e6
    print(f"  Data: {n:,} rows, {len(cols)} columns, {data_mb:.2f} MB raw")

    path = tempfile.mktemp(suffix=".mktf")

    # Write
    wt = time_fn(lambda: write_mktf(path, cols,
                  leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
                  cadence="TI00TO00", leaf_version="1.0.0", metadata=meta), 30)
    ws = stats(wt)
    file_mb = os.path.getsize(path) / 1e6

    # Full read
    rt = time_fn(lambda: read_data(path), 30)
    rs = stats(rt)

    # Selective 2-col
    st2 = time_fn(lambda: read_selective(path, ["price", "conditions"]), 30)
    ss2 = stats(st2)

    # Selective 1-col
    st1 = time_fn(lambda: read_selective(path, ["price"]), 30)
    ss1 = stats(st1)

    # Header only (1000 runs for precision)
    ht = time_fn(lambda: read_header(path), 1000, warmup=100)
    hs = stats(ht)

    # Correctness
    rb = read_data(path)
    correct = all(np.array_equal(cols[k], rb[k]) for k in cols)

    # Header inspection
    h = read_header(path)

    overhead_bytes = os.path.getsize(path) - sum(a.nbytes for a in cols.values())

    print(f"  File:   {file_mb:.2f} MB (data: {data_mb:.2f} MB, overhead: {overhead_bytes/1024:.1f} KB)")
    print(f"  Write:  {fmt(ws)}")
    print(f"  Read:   {fmt(rs)}")
    print(f"  Sel 1:  {fmt(ss1)}")
    print(f"  Sel 2:  {fmt(ss2)}")
    print(f"  Header: {hs['mean']*1000:.1f}us +/- {hs['std']*1000:.1f}")
    print(f"  Correct: {'PASS' if correct else 'FAIL'}")
    print(f"  Complete: {h.is_complete}")
    print()

    return path, cols, meta, n, ws, rs, ss1, ss2, hs


# ══════════════════════════════════════════════════════════════════
# PHASE 2: v3 vs v2 Comparison
# ══════════════════════════════════════════════════════════════════

def phase2(v3_path, cols, meta, v3_ws, v3_rs, v3_ss1, v3_ss2, v3_hs):
    print("=" * 80)
    print("PHASE 2: MKTF v3 (4096-align) vs v2 (64-align)")
    print("=" * 80)
    print()

    # Write v2 for comparison
    cols_v2, meta_v2 = load_and_encode("float32")
    v2_path = tempfile.mktemp(suffix=".mktf2")

    wt2 = time_fn(lambda: write_v2(v2_path, cols_v2, meta_v2), 30)
    ws2 = stats(wt2)
    v2_mb = os.path.getsize(v2_path) / 1e6

    rt2 = time_fn(lambda: read_v2(v2_path), 30)
    rs2 = stats(rt2)

    st2_sel = time_fn(lambda: read_v2_selective(v2_path, ["price", "conditions"]), 30)
    ss2_sel = stats(st2_sel)

    v3_mb = os.path.getsize(v3_path) / 1e6

    print(f"  {'Metric':<20s} {'v3 (4096)':>12s} {'v2 (64)':>12s} {'Delta':>10s}")
    print(f"  {'─'*55}")
    print(f"  {'File size':<20s} {v3_mb:>10.2f}MB {v2_mb:>10.2f}MB {(v3_mb-v2_mb):>+8.2f}MB")
    print(f"  {'Write':<20s} {v3_ws['mean']:>10.2f}ms {ws2['mean']:>10.2f}ms {v3_ws['mean']-ws2['mean']:>+8.2f}ms")
    print(f"  {'Read (full)':<20s} {v3_rs['mean']:>10.2f}ms {rs2['mean']:>10.2f}ms {v3_rs['mean']-rs2['mean']:>+8.2f}ms")
    print(f"  {'Sel 2-col':<20s} {v3_ss2['mean']:>10.2f}ms {ss2_sel['mean']:>10.2f}ms {v3_ss2['mean']-ss2_sel['mean']:>+8.2f}ms")
    print(f"  {'Header scan':<20s} {v3_hs['mean']*1000:>8.1f}us {'N/A':>12s} {'N/A':>10s}")
    print()

    os.unlink(v2_path)
    return ws2, rs2


# ══════════════════════════════════════════════════════════════════
# PHASE 3: GPU Pipeline (source-only + fused recompute)
# ══════════════════════════════════════════════════════════════════

def phase3(path, n):
    print("=" * 80)
    print("PHASE 3: GPU Pipeline — Source-Only + Fused Recompute")
    print("=" * 80)
    print()

    # Full pipeline: read -> H2D -> fused kernel -> all 16 arrays on GPU
    def full_pipeline():
        d = read_data(path)
        p_gpu = cp.asarray(d["price"])
        s_gpu = cp.asarray(d["size"])
        ts_gpu = cp.asarray(d["timestamp"])
        ex_gpu = cp.asarray(d["exchange"])
        cond_gpu = cp.asarray(d["conditions"])
        is_odd_gpu = cp.asarray(d["is_odd_lot"])
        derived = run_fused_fp32(p_gpu, s_gpu, ts_gpu, n)
        cp.cuda.Device(0).synchronize()
        return p_gpu, s_gpu, ts_gpu, ex_gpu, cond_gpu, is_odd_gpu, derived

    # Warmup
    full_pipeline()
    full_pipeline()

    ft = []
    for _ in range(30):
        gc.disable()
        t0 = time.perf_counter_ns()
        full_pipeline()
        t1 = time.perf_counter_ns()
        gc.enable()
        ft.append((t1 - t0) / 1e6)

    fs = stats(ft)
    print(f"  Full pipeline (read 7 cols + H2D + fused kernel): {fmt(fs)}")

    # Breakdown
    # Read only
    rt = time_fn(lambda: read_data(path), 30)
    rs = stats(rt)

    # H2D only
    d = read_data(path)
    arrays = [np.ascontiguousarray(v) for v in d.values()]
    for a in arrays:
        cp.asarray(a)
    cp.cuda.Device(0).synchronize()

    ht = []
    for _ in range(30):
        gc.disable()
        t0 = time.perf_counter_ns()
        for a in arrays:
            cp.asarray(a)
        cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter_ns()
        gc.enable()
        ht.append((t1 - t0) / 1e6)
    hs = stats(ht)

    # Fused kernel only
    p_gpu = cp.asarray(d["price"])
    s_gpu = cp.asarray(d["size"])
    ts_gpu = cp.asarray(d["timestamp"])
    cp.cuda.Device(0).synchronize()

    kt = []
    for _ in range(200):
        gc.disable()
        t0 = time.perf_counter_ns()
        run_fused_fp32(p_gpu, s_gpu, ts_gpu, n)
        cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter_ns()
        gc.enable()
        kt.append((t1 - t0) / 1e6)
    ks = stats(kt)

    print()
    print(f"  Breakdown:")
    print(f"    Read 7 cols:      {rs['mean']:7.2f}ms")
    print(f"    H2D transfer:     {hs['mean']:7.2f}ms")
    print(f"    Fused kernel:     {ks['mean']:7.4f}ms  ({ks['mean']*1000:.1f}us)")
    print(f"    Sum:              {rs['mean']+hs['mean']+ks['mean']:7.2f}ms")
    print()
    print(f"  Universe projection (4604 tickers):")
    print(f"    Full pipeline:    {fs['mean'] * 4604 / 1000:.1f}s = {fs['mean'] * 4604 / 60000:.1f} min")
    print()

    return fs


# ══════════════════════════════════════════════════════════════════
# PHASE 4: Cold Cache
# ══════════════════════════════════════════════════════════════════

def phase4(path, warm_rs):
    print("=" * 80)
    print("PHASE 4: Cold vs Warm Cache")
    print("=" * 80)
    print()

    cold_times = []
    for _ in range(10):
        evict_cache()
        time.sleep(0.05)
        gc.disable()
        t0 = time.perf_counter_ns()
        _ = read_data(path)
        t1 = time.perf_counter_ns()
        gc.enable()
        cold_times.append((t1 - t0) / 1e6)

    cs = stats(cold_times)
    ratio = cs["mean"] / warm_rs["mean"]
    print(f"  Warm: {warm_rs['mean']:.2f}ms")
    print(f"  Cold: {cs['mean']:.2f}ms +/- {cs['std']:.2f}")
    print(f"  Ratio: {ratio:.2f}x")
    print()


# ══════════════════════════════════════════════════════════════════
# PHASE 5: Daemon Simulation
# ══════════════════════════════════════════════════════════════════

def phase5(path):
    print("=" * 80)
    print("PHASE 5: Daemon Header Scan Simulation")
    print("=" * 80)
    print()

    # Create 100 files to simulate directory scan
    import shutil
    tmpdir = tempfile.mkdtemp(prefix="daemon_")
    cols, meta = load_aapl()

    print("  Creating 100 MKTF v3 files...")
    paths = []
    for i in range(100):
        p = os.path.join(tmpdir, f"ticker_{i:04d}.mktf")
        write_mktf(p, cols,
                    leaf_id="K01P01", ticker=f"T{i:04d}", day="2025-09-02",
                    cadence="TI00TO00", leaf_version="1.0.0", metadata=meta)
        paths.append(p)

    # Header scan: read all 100 headers
    # Warmup
    for p in paths[:5]:
        read_header(p)

    n_runs = 20
    times = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        for p in paths:
            h = read_header(p)
            _ = h.is_complete and h.leaf_version == "1.0.0" and h.total_nulls == 0
        t1 = time.perf_counter_ns()
        gc.enable()
        times.append((t1 - t0) / 1e6)

    hs = stats(times)
    per_file = hs["mean"] / 100
    print(f"  100 headers scanned: {hs['mean']:.2f}ms ({per_file*1000:.1f}us/file)")
    print(f"  Projected 4604 files: {per_file * 4604:.1f}ms ({per_file * 4604 / 1000:.3f}s)")
    print()

    # What info does the daemon get per header?
    h = read_header(paths[0])
    print(f"  Per header (no data read):")
    print(f"    is_complete: {h.is_complete}")
    print(f"    ticker: {h.ticker}")
    print(f"    day: {h.day}")
    print(f"    schema_fp: {h.schema_fingerprint.hex()[:8]}...")
    print(f"    n_rows: {h.n_rows:,}")
    print(f"    total_nulls: {h.total_nulls}")
    print(f"    write_duration: {h.write_duration_ms}ms")
    print(f"    columns: {len(h.columns)} with min/max/null_count each")
    print()

    # Column quality check without reading data
    print(f"  Column quality (from headers, zero data bytes):")
    for c in h.columns:
        print(f"    {c.name:<16s} min={c.min_value:>14.4f}  max={c.max_value:>14.4f}  "
              f"nulls={c.null_count}")
    print()

    # Crash recovery simulation
    print("  Crash recovery simulation:")
    crash_path = os.path.join(tmpdir, "crash_test.mktf")
    # Write a file but corrupt it by truncating
    write_mktf(crash_path, cols,
                leaf_id="K01P01", ticker="CRASH", day="2025-09-02",
                cadence="TI00TO00", leaf_version="1.0.0", metadata=meta)
    h_good = read_header(crash_path)
    print(f"    Good file: is_complete={h_good.is_complete}")

    # Simulate crash by rewriting with is_complete=0
    import struct
    with open(crash_path, "r+b") as f:
        f.seek(6)
        flags = struct.unpack("<H", f.read(2))[0]
        flags &= ~0x0001  # clear FLAG_COMPLETE
        f.seek(6)
        f.write(struct.pack("<H", flags))

    h_bad = read_header(crash_path)
    print(f"    Crashed file: is_complete={h_bad.is_complete}")
    print(f"    Daemon action: delete and recompute")

    shutil.rmtree(tmpdir)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print()
    print("OBSERVER EXPERIMENT 6: MKTF v3 End-to-End Benchmark")
    print(f"GPU: NVIDIA RTX PRO 6000 Blackwell, {cp.cuda.Device(0).mem_info[1]/1e9:.0f} GB")
    print()

    path, cols, meta, n, v3_ws, v3_rs, v3_ss1, v3_ss2, v3_hs = phase1()
    phase2(path, cols, meta, v3_ws, v3_rs, v3_ss1, v3_ss2, v3_hs)
    gpu_fs = phase3(path, n)
    phase4(path, v3_rs)
    phase5(path)

    os.unlink(path)

    print("=" * 80)
    print("EXPERIMENT 6 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
