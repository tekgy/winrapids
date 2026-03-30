"""Experiment 10: Driver Regression Isolation.

Experiment 7 (R581): 12 workers → 5.62 GB/s, 2.45x speedup
Experiment 8 (R595): 24 workers → 3.61 GB/s, 1.38x speedup

Was it the driver update (R581→R595) or the file count change (100→200)?
This experiment runs the EXACT Experiment 7 config (100 files) on R595
to isolate the variable.

If we get ~5.6 GB/s → it was the file count
If we get ~3.6 GB/s → it was the driver
"""

from __future__ import annotations

import gc
import io
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "20260327-mktf-format"))
from mktf_v3 import write_mktf, read_data, AAPL_PATH, COL_MAP, CONDITION_BITS


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


def main():
    # EXACT Experiment 7 config
    N_FILES = 100
    N_RUNS = 10

    print("=" * 78)
    print("EXPERIMENT 10: DRIVER REGRESSION ISOLATION")
    print("=" * 78)
    print(f"Config: EXACT Experiment 7 (100 files, 10 runs)")
    print(f"Driver: R595 (Experiment 7 was R581, Experiment 8 was R595)")
    print()

    cols = load_source_columns()
    tmpdir = tempfile.mkdtemp(prefix="mktf_iso_")
    paths = []
    for i in range(N_FILES):
        p = os.path.join(tmpdir, f"ticker_{i:04d}.mktf")
        write_mktf(p, cols, leaf_id="K01P01", ticker=f"T{i:04d}", day="2025-09-02",
                    metadata={"mode": "source-only"})
        paths.append(p)

    file_size = os.path.getsize(paths[0])
    total_bytes = file_size * N_FILES
    print(f"Files: {N_FILES}, {file_size/1e6:.2f} MB each, {total_bytes/1e6:.0f} MB total")
    print()

    # Warmup
    for p in paths:
        _ = read_data(p)

    # Sequential baseline
    gc.disable()
    seq_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter_ns()
        for p in paths:
            _ = read_data(p)
        seq_times.append((time.perf_counter_ns() - t0) / 1e6)
    gc.enable()
    seq_mean = np.mean(seq_times)
    seq_bw = total_bytes / (seq_mean / 1000) / 1e9

    print(f"SEQUENTIAL: {seq_mean:.1f}ms ({seq_mean/N_FILES:.2f} ms/file) BW: {seq_bw:.2f} GB/s")
    print()

    # Same worker counts as Experiment 7
    print(f"{'Workers':>8} {'Mean (ms)':>12} {'Std':>8} {'ms/file':>10} "
          f"{'BW (GB/s)':>12} {'Speedup':>10}")
    print("-" * 60)

    for n_w in [1, 2, 4, 8, 12, 16]:
        # Warmup
        with ThreadPoolExecutor(max_workers=n_w) as exe:
            list(exe.map(read_data, paths))

        gc.disable()
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter_ns()
            with ThreadPoolExecutor(max_workers=n_w) as exe:
                list(exe.map(read_data, paths))
            times.append((time.perf_counter_ns() - t0) / 1e6)
        gc.enable()

        mean_t = np.mean(times)
        std_t = np.std(times)
        bw = total_bytes / (mean_t / 1000) / 1e9
        speedup = seq_mean / mean_t
        print(f"{n_w:>8} {mean_t:>12.1f} {std_t:>8.1f} {mean_t/N_FILES:>10.2f} "
              f"{bw:>12.2f} {speedup:>9.2f}x")

    print()
    print("COMPARISON:")
    print(f"  Exp 7 (R581, 100 files): seq 2.29 GB/s, 12w = 5.62 GB/s (2.45x)")
    print(f"  Exp 8 (R595, 200 files): seq 2.62 GB/s, 24w = 3.61 GB/s (1.38x)")
    print(f"  Exp 10 (R595, 100 files): seq {seq_bw:.2f} GB/s  ← same file count as Exp 7")
    print()
    print("If 12w ~ 5.6 GB/s → file count was the variable")
    print("If 12w ~ 3.6 GB/s → driver was the variable")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
