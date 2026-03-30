"""Observer benchmark: MKTF v2 end-to-end — disk -> aligned read -> H2D -> GPU ops.

Rigorous statistical benchmarking of pathmaker's MKTF v2 implementation
with real AAPL data across all encoding paths.

Measures:
1. Write/read speed with statistical significance (30 runs)
2. Column-selective reads (1, 2, 4 cols)
3. Cold vs warm cache
4. Full GPU pipeline: disk -> read -> H2D -> GPU compute
5. GPU compute: float32 vs int64 vs int32 throughput
6. End-to-end: parquet source -> MKTF v2 -> GPU (the actual production path)
"""
from __future__ import annotations

import gc
import io
import json
import os
import sys
import tempfile
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# Add research dir to path so we can import mktf_v2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "research", "20260327-mktf-format"))

import numpy as np

try:
    import cupy as cp
    GPU = True
except ImportError:
    GPU = False

import pyarrow.parquet as pq

from mktf_v2 import (
    write_mktf, read_mktf, read_mktf_selective, read_mktf_metadata,
    load_and_encode, PRICE_SCALE,
)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def stats(times: list[float]) -> dict:
    s = sorted(times)
    n = len(s)
    m = sum(s) / n
    sd = (sum((x - m) ** 2 for x in s) / max(n - 1, 1)) ** 0.5
    return {
        "mean": m, "std": sd, "min": s[0], "max": s[-1],
        "p50": s[n // 2], "p95": s[int(n * 0.95)], "n": n,
    }


def fmt_stats(st: dict) -> str:
    return (f"{st['mean']:7.2f}ms +/- {st['std']:5.2f}  "
            f"[p50={st['p50']:6.2f}, min={st['min']:6.2f}, max={st['max']:6.2f}]")


def time_fn(fn, n_runs=30, warmup=3) -> list[float]:
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
        try:
            arr = np.ones(512 * 1024 * 1024 // 8, dtype=np.float64)
            _ = arr.sum()
            del arr
            gc.collect()
        except MemoryError:
            pass


# ══════════════════════════════════════════════════════════════════
# PHASE 1: Encoding path comparison
# ══════════════════════════════════════════════════════════════════

def phase1_encoding_comparison(n_runs=30):
    print("=" * 80)
    print("PHASE 1: MKTF v2 Encoding Path Comparison (Real AAPL Data)")
    print("=" * 80)
    print()

    results = {}

    for enc in ["float32", "float64", "int64", "int32"]:
        print(f"  [{enc}]")
        cols, meta = load_and_encode(enc)
        data_mb = sum(a.nbytes for a in cols.values()) / 1e6

        tmp = tempfile.mktemp(suffix=".mktf")

        # Write
        wt = time_fn(lambda: write_mktf(tmp, cols, meta), n_runs)
        ws = stats(wt)

        file_mb = os.path.getsize(tmp) / 1e6

        # Full read
        rt = time_fn(lambda: read_mktf(tmp), n_runs)
        rs = stats(rt)

        # Selective read: price + conditions (2 cols)
        st2 = time_fn(lambda: read_mktf_selective(tmp, ["price", "conditions"]), n_runs)
        ss2 = stats(st2)

        # Selective read: price only (1 col)
        st1 = time_fn(lambda: read_mktf_selective(tmp, ["price"]), n_runs)
        ss1 = stats(st1)

        # Selective read: price, size, timestamp, conditions (4 cols)
        st4 = time_fn(lambda: read_mktf_selective(tmp, ["price", "size", "timestamp", "conditions"]), n_runs)
        ss4 = stats(st4)

        # Verify correctness
        readback = read_mktf(tmp)
        correct = all(np.array_equal(cols[k], readback[k]) for k in cols)

        readback_sel = read_mktf_selective(tmp, ["price", "conditions"])
        sel_correct = (np.array_equal(cols["price"], readback_sel["price"]) and
                       np.array_equal(cols["conditions"], readback_sel["conditions"]))

        print(f"    Data:      {data_mb:.2f} MB, File: {file_mb:.2f} MB (overhead: {(file_mb/data_mb - 1)*100:.1f}%)")
        print(f"    Write:     {fmt_stats(ws)}")
        print(f"    Read full: {fmt_stats(rs)}")
        print(f"    Read 1col: {fmt_stats(ss1)}")
        print(f"    Read 2col: {fmt_stats(ss2)}")
        print(f"    Read 4col: {fmt_stats(ss4)}")
        print(f"    Correct:   data={'PASS' if correct else 'FAIL'}, selective={'PASS' if sel_correct else 'FAIL'}")
        print()

        results[enc] = {
            "data_mb": data_mb, "file_mb": file_mb,
            "write": ws, "read": rs,
            "sel_1col": ss1, "sel_2col": ss2, "sel_4col": ss4,
            "correct": correct and sel_correct,
            "path": tmp, "cols": cols, "meta": meta,
        }

    return results


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Cold vs warm cache
# ══════════════════════════════════════════════════════════════════

def phase2_cold_warm(enc_results, n_cold=10):
    print("=" * 80)
    print("PHASE 2: Cold vs Warm Cache (MKTF v2)")
    print("=" * 80)
    print()

    for enc, res in enc_results.items():
        path = res["path"]
        warm_mean = res["read"]["mean"]

        cold_times = []
        for _ in range(n_cold):
            evict_cache()
            time.sleep(0.05)
            gc.disable()
            t0 = time.perf_counter_ns()
            _ = read_mktf(path)
            t1 = time.perf_counter_ns()
            gc.enable()
            cold_times.append((t1 - t0) / 1e6)

        cs = stats(cold_times)
        ratio = cs["mean"] / warm_mean
        print(f"  [{enc}]")
        print(f"    Warm: {warm_mean:.2f}ms, Cold: {cs['mean']:.2f}ms +/- {cs['std']:.2f}, Ratio: {ratio:.2f}x")
        res["read_cold"] = cs

    print()


# ══════════════════════════════════════════════════════════════════
# PHASE 3: GPU pipeline
# ══════════════════════════════════════════════════════════════════

def phase3_gpu_pipeline(enc_results, n_runs=30):
    if not GPU:
        print("PHASE 3: SKIPPED (no GPU)")
        return

    print("=" * 80)
    print("PHASE 3: GPU Pipeline — disk -> read -> H2D -> compute")
    print("=" * 80)
    print()

    # Pure H2D baseline
    test_cols, _ = load_and_encode("float32")
    arrays = [np.ascontiguousarray(a) for a in test_cols.values()]
    total_bytes = sum(a.nbytes for a in arrays)
    for a in arrays:
        cp.asarray(a)
    cp.cuda.Device(0).synchronize()

    h2d_times = time_fn(lambda: [cp.asarray(a) for a in arrays] or cp.cuda.Device(0).synchronize(), n_runs)
    # Need proper sync
    h2d_times = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        for a in arrays:
            cp.asarray(a)
        cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter_ns()
        gc.enable()
        h2d_times.append((t1 - t0) / 1e6)

    h2d_st = stats(h2d_times)
    bw = total_bytes / (h2d_st["mean"] / 1000) / 1e9
    print(f"  Pure H2D baseline: {h2d_st['mean']:.2f}ms ({total_bytes/1e6:.1f} MB @ {bw:.1f} GB/s)")
    print()

    for enc in ["float32", "int64", "int32"]:
        res = enc_results[enc]
        path = res["path"]
        print(f"  [{enc}]")

        # Disk -> GPU (full pipeline)
        # Warmup
        d = read_mktf(path)
        for a in d.values():
            cp.asarray(a)
        cp.cuda.Device(0).synchronize()

        full_times = []
        for _ in range(n_runs):
            gc.disable()
            t0 = time.perf_counter_ns()
            d = read_mktf(path)
            for a in d.values():
                cp.asarray(a)
            cp.cuda.Device(0).synchronize()
            t1 = time.perf_counter_ns()
            gc.enable()
            full_times.append((t1 - t0) / 1e6)

        full_st = stats(full_times)
        res["gpu_full"] = full_st

        # H2D only (data pre-read)
        d = read_mktf(path)
        host_arrays = [np.ascontiguousarray(a) for a in d.values()]
        for a in host_arrays:
            cp.asarray(a)
        cp.cuda.Device(0).synchronize()

        h2d_only_times = []
        for _ in range(n_runs):
            gc.disable()
            t0 = time.perf_counter_ns()
            for a in host_arrays:
                cp.asarray(a)
            cp.cuda.Device(0).synchronize()
            t1 = time.perf_counter_ns()
            gc.enable()
            h2d_only_times.append((t1 - t0) / 1e6)

        h2d_only_st = stats(h2d_only_times)
        res["gpu_h2d_only"] = h2d_only_st

        read_portion = full_st["mean"] - h2d_only_st["mean"]
        print(f"    Disk->GPU: {full_st['mean']:.2f}ms  (read~{read_portion:.2f}ms + H2D~{h2d_only_st['mean']:.2f}ms)")

        # Selective disk -> GPU (price + conditions only)
        sel_times = []
        for _ in range(n_runs):
            gc.disable()
            t0 = time.perf_counter_ns()
            d = read_mktf_selective(path, ["price", "conditions"])
            for a in d.values():
                cp.asarray(a)
            cp.cuda.Device(0).synchronize()
            t1 = time.perf_counter_ns()
            gc.enable()
            sel_times.append((t1 - t0) / 1e6)

        sel_st = stats(sel_times)
        print(f"    Sel->GPU:  {sel_st['mean']:.2f}ms  (2 cols: price + conditions)")
        print()

    # ── GPU compute: float32 vs int64 vs int32 ──
    print("  GPU Compute Operations:")
    print("  " + "-" * 70)

    cols_f32, _ = load_and_encode("float32")
    cols_i64, _ = load_and_encode("int64")
    cols_i32, _ = load_and_encode("int32")

    p_f32 = cp.asarray(cols_f32["price"])
    s_f32 = cp.asarray(cols_f32["size"])
    p_i64 = cp.asarray(cols_i64["price"])
    s_i64 = cp.asarray(cols_i64["size"])
    p_i32 = cp.asarray(cols_i32["price"])
    s_i32 = cp.asarray(cols_i32["size"])
    cp.cuda.Device(0).synchronize()

    n_gpu = 100
    ops = {
        "sum":      lambda p, s: cp.sum(p),
        "mean":     lambda p, s: cp.mean(p.astype(cp.float64)),
        "std":      lambda p, s: cp.std(p.astype(cp.float64)),
        "min":      lambda p, s: cp.min(p),
        "max":      lambda p, s: cp.max(p),
        "diff":     lambda p, s: cp.diff(p),
        "p*s":      lambda p, s: p.astype(cp.float64) * s.astype(cp.float64),
        "sort":     lambda p, s: cp.sort(p.copy()),
    }

    print(f"  {'Op':<12s} {'float32':>10s} {'int64':>10s} {'int32':>10s} {'Winner':>10s}")
    print(f"  {'─'*55}")

    for op_name, fn in ops.items():
        timings = {}
        for label, p, s in [("f32", p_f32, s_f32), ("i64", p_i64, s_i64), ("i32", p_i32, s_i32)]:
            # Warmup
            for _ in range(10):
                fn(p, s)
            cp.cuda.Device(0).synchronize()

            t0 = time.perf_counter()
            for _ in range(n_gpu):
                fn(p, s)
            cp.cuda.Device(0).synchronize()
            timings[label] = (time.perf_counter() - t0) / n_gpu * 1e6

        winner = min(timings, key=timings.get)
        print(f"  {op_name:<12s} {timings['f32']:>8.1f}us {timings['i64']:>8.1f}us "
              f"{timings['i32']:>8.1f}us {winner:>10s}")

    print()


# ══════════════════════════════════════════════════════════════════
# PHASE 4: End-to-end vs Parquet
# ══════════════════════════════════════════════════════════════════

def phase4_vs_parquet(enc_results, n_runs=30):
    print("=" * 80)
    print("PHASE 4: End-to-End — MKTF v2 vs Parquet (source format)")
    print("=" * 80)
    print()

    aapl_path = "W:/fintek/data/fractal/K01/2025-09-02/AAPL/K01P01.TI00TO00.parquet"
    parquet_mb = os.path.getsize(aapl_path) / 1e6

    # Parquet read baseline
    pq_times = time_fn(lambda: pq.read_table(aapl_path), n_runs)
    pq_st = stats(pq_times)
    print(f"  Parquet read ({parquet_mb:.1f} MB): {fmt_stats(pq_st)}")

    # Parquet -> numpy conversion
    def pq_to_numpy():
        tbl = pq.read_table(aapl_path)
        return {c: tbl.column(c).to_numpy() for c in tbl.column_names}

    pq_np_times = time_fn(pq_to_numpy, n_runs)
    pq_np_st = stats(pq_np_times)
    print(f"  Parquet->numpy:           {fmt_stats(pq_np_st)}")

    if GPU:
        # Parquet -> GPU
        def pq_to_gpu():
            tbl = pq.read_table(aapl_path)
            arrays = {}
            for c in tbl.column_names:
                a = tbl.column(c).to_numpy()
                if np.issubdtype(a.dtype, np.number) or a.dtype == np.bool_:
                    arrays[c] = cp.asarray(a)
            cp.cuda.Device(0).synchronize()
            return arrays

        # Warmup
        pq_to_gpu()
        pq_to_gpu()

        pq_gpu_times = []
        for _ in range(n_runs):
            gc.disable()
            t0 = time.perf_counter_ns()
            pq_to_gpu()
            t1 = time.perf_counter_ns()
            gc.enable()
            pq_gpu_times.append((t1 - t0) / 1e6)

        pq_gpu_st = stats(pq_gpu_times)
        print(f"  Parquet->GPU:             {fmt_stats(pq_gpu_st)}")

    # MKTF v2 comparison
    print()
    for enc in ["float32", "int64"]:
        res = enc_results[enc]
        gpu_ms = res.get("gpu_full", {}).get("mean", 0)
        read_ms = res["read"]["mean"]
        print(f"  MKTF v2 ({enc:>7s}) read:  {read_ms:.2f}ms  "
              f"(vs parquet {pq_st['mean']:.2f}ms = {pq_st['mean']/read_ms:.1f}x faster)")
        if GPU and gpu_ms > 0:
            pq_gpu_mean = pq_gpu_st["mean"]
            print(f"  MKTF v2 ({enc:>7s}) ->GPU: {gpu_ms:.2f}ms  "
                  f"(vs parquet {pq_gpu_mean:.2f}ms = {pq_gpu_mean/gpu_ms:.1f}x faster)")
    print()

    # Universe projection
    print("  Universe projection (4604 tickers):")
    pq_universe = pq_st["mean"] / 1000 * 4604
    print(f"    Parquet read:  {pq_universe:.1f}s = {pq_universe/60:.1f} min")
    for enc in ["float32", "int64"]:
        res = enc_results[enc]
        mktf_universe = res["read"]["mean"] / 1000 * 4604
        gpu_universe = res.get("gpu_full", {}).get("mean", 0) / 1000 * 4604
        print(f"    MKTF {enc:>7s}: {mktf_universe:.1f}s = {mktf_universe/60:.1f} min "
              f"(GPU: {gpu_universe:.1f}s = {gpu_universe/60:.1f} min)")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print()
    print("MKTF v2 OBSERVER BENCHMARK — Rigorous End-to-End Analysis")
    print(f"GPU: {'NVIDIA RTX PRO 6000 Blackwell' if GPU else 'NOT AVAILABLE'}")
    print(f"Data: Real AAPL ticks (598,057 rows)")
    print()

    enc_results = phase1_encoding_comparison()
    phase2_cold_warm(enc_results)
    phase3_gpu_pipeline(enc_results)
    phase4_vs_parquet(enc_results)

    # Cleanup
    for res in enc_results.values():
        try:
            os.unlink(res["path"])
        except OSError:
            pass

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
