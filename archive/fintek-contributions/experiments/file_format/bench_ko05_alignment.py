"""Experiment 19: KO05 Alignment — 4096 vs 64 bytes.

KO05 files have 25 stat columns. At alignment=4096 (default), each column
gets padded to 4096 bytes = 110 KB minimum. At alignment=64 (GPU cache line),
each column gets padded to 64 bytes = ~3-4 KB total.

Same writer, same reader, same 25 columns. Just a different alignment parameter.
node === node. One column per output, always.
"""

from __future__ import annotations

import gc
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import read_columns, read_header


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


def make_ko05_columns(n_bins: int) -> dict[str, np.ndarray]:
    """25 stat columns: 5 data cols × 5 stats."""
    rng = np.random.default_rng(42)
    stat_names = ["sum", "sum_sq", "min", "max", "count"]
    data_names = ["price", "size", "timestamp", "exchange", "conditions"]
    cols = {}
    for dname in data_names:
        for sname in stat_names:
            cols[f"{dname}_{sname}"] = rng.normal(100, 10, n_bins).astype(np.float32)
    return cols


def main():
    N = 30

    print("=" * 78)
    print("EXPERIMENT 19: KO05 ALIGNMENT (4096 vs 64 bytes)")
    print("=" * 78)
    print(f"Runs per benchmark: {N}")
    print()

    tmpdir = tempfile.mkdtemp(prefix="mktf_align_")

    bin_counts = [1, 13, 78, 390, 780, 2340]
    alignments = [4096, 64]

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Write speed comparison
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1: WRITE SPEED")
    print("-" * 60)

    print(f"  {'Bins':>6} | {'align=4096':>18} | {'align=64':>18} | {'Speedup':>8}")
    print(f"  {'----':>6} | {'----------':>18} | {'--------':>18} | {'-------':>8}")

    for n_bins in bin_counts:
        results = []
        for align in alignments:
            cols = make_ko05_columns(n_bins)
            fpath = Path(tmpdir) / f"a{align}_{n_bins}.mktf"

            # Warm
            for _ in range(3):
                write_mktf(fpath, cols, leaf_id="AAPL.K02.KO05", ticker="AAPL",
                           day="2025-09-02", safe=False, alignment=align)

            t = timed_us(lambda c=cols, p=fpath, a=align: write_mktf(
                p, c, leaf_id="AAPL.K02.KO05", ticker="AAPL",
                day="2025-09-02", safe=False, alignment=a
            ), N)

            file_size = fpath.stat().st_size
            results.append((t, file_size))

        speedup = results[0][0]["mean"] / results[1][0]["mean"]
        print(f"  {n_bins:>6} | {results[0][0]['mean']/1000:>7.2f}ms {results[0][1]/1024:>6.1f}KB "
              f"| {results[1][0]['mean']/1000:>7.2f}ms {results[1][1]/1024:>6.1f}KB "
              f"| {speedup:>6.2f}x")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: File size analysis
    # ═══════════════════════════════════════════════════════════
    print("PHASE 2: FILE SIZE ANALYSIS")
    print("-" * 60)

    for n_bins in [1, 78, 780]:
        cols = make_ko05_columns(n_bins)
        data_bytes = sum(a.nbytes for a in cols.values())

        for align in alignments:
            fpath = Path(tmpdir) / f"size_a{align}_{n_bins}.mktf"
            write_mktf(fpath, cols, leaf_id="X", safe=False, alignment=align)
            fsize = fpath.stat().st_size
            utilization = data_bytes / fsize * 100
            print(f"  {n_bins:>6} bins, align={align:>4}: "
                  f"{fsize:>8} bytes ({fsize/1024:.1f} KB), "
                  f"data={data_bytes} bytes, "
                  f"utilization={utilization:.1f}%")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Write time decomposition
    # ═══════════════════════════════════════════════════════════
    print("PHASE 3: WRITE TIME DECOMPOSITION (78 bins)")
    print("-" * 60)

    # Minimal file (1 column, alignment=64) as floor
    min_cols = {"x": np.zeros(1, dtype=np.float32)}
    for _ in range(3):
        write_mktf(Path(tmpdir) / "min.mktf", min_cols, leaf_id="X", safe=False, alignment=64)
    t_min = timed_us(lambda: write_mktf(
        Path(tmpdir) / "min.mktf", min_cols, leaf_id="X", safe=False, alignment=64
    ), N)

    for align in alignments:
        cols = make_ko05_columns(78)
        fpath = Path(tmpdir) / f"decomp_a{align}.mktf"
        for _ in range(3):
            write_mktf(fpath, cols, leaf_id="X", safe=False, alignment=align)
        t = timed_us(lambda a=align, c=cols: write_mktf(
            Path(tmpdir) / f"decomp_a{a}.mktf", c, leaf_id="X", safe=False, alignment=a
        ), N)

        overhead = t["mean"] - t_min["mean"]
        print(f"  align={align:>4}: {t['mean']/1000:.2f}ms total, "
              f"{t_min['mean']/1000:.2f}ms floor, "
              f"{overhead/1000:.2f}ms column overhead")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Read speed comparison
    # ═══════════════════════════════════════════════════════════
    print("PHASE 4: READ SPEED (78 bins)")
    print("-" * 60)

    for align in alignments:
        cols = make_ko05_columns(78)
        fpath = Path(tmpdir) / f"read_a{align}.mktf"
        write_mktf(fpath, cols, leaf_id="X", safe=False, alignment=align)
        fsize = fpath.stat().st_size

        # Warm
        for _ in range(5):
            read_columns(fpath)

        t = timed_us(lambda p=fpath: read_columns(p), N)
        print(f"  align={align:>4}: {t['mean']:.0f}us ±{t['std']:.0f} ({fsize/1024:.1f} KB)")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Universe projections
    # ═══════════════════════════════════════════════════════════
    print("PHASE 5: UNIVERSE PROJECTIONS")
    print("-" * 60)

    tickers = 4604
    for align in alignments:
        cols = make_ko05_columns(78)
        fpath = Path(tmpdir) / f"proj_a{align}.mktf"
        for _ in range(3):
            write_mktf(fpath, cols, leaf_id="X", safe=False, alignment=align)
        t = timed_us(lambda a=align, c=cols: write_mktf(
            Path(tmpdir) / f"proj_a{a}.mktf", c, leaf_id="X", safe=False, alignment=a
        ), N)
        per_file_ms = t["mean"] / 1000

        for n_cad in [8, 10]:
            total = tickers * n_cad
            total_s = total * per_file_ms / 1000
            print(f"  align={align:>4} × {n_cad} cad: {per_file_ms:.2f}ms/file × {total:,} = {total_s:.1f}s")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 6: Correctness — verify roundtrip
    # ═══════════════════════════════════════════════════════════
    print("PHASE 6: CORRECTNESS")
    print("-" * 60)

    for align in alignments:
        cols = make_ko05_columns(78)
        fpath = Path(tmpdir) / f"correct_a{align}.mktf"
        write_mktf(fpath, cols, leaf_id="X", safe=False, alignment=align)
        header, read_back = read_columns(fpath)

        all_match = True
        for name in cols:
            if not np.array_equal(cols[name], read_back[name]):
                print(f"  MISMATCH: align={align} / {name}")
                all_match = False
        if all_match:
            print(f"  align={align:>4}: all 25 columns roundtrip correctly "
                  f"(header.alignment={header.alignment})")

    print()

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
