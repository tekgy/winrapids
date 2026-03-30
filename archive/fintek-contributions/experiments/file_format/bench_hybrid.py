"""Experiment 2: Hybrid per-column encoding vs plain aligned binary.

Tests whether per-column compression (delta timestamps, bitmask booleans, LZ4 flags)
provides net benefit over raw aligned binary given the I/O vs CPU tradeoff.
"""
from __future__ import annotations

import gc
import io
import json
import os
import struct
import sys
import tempfile
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import lz4.frame

try:
    import cupy as cp
    GPU = True
except ImportError:
    GPU = False

ALIGN = 64


def generate_test_data(n=598057):
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.0005, n)
    price = 230.0 * np.exp(np.cumsum(returns))
    return {
        "price": price.astype(np.float64),
        "size": rng.exponential(100, n).astype(np.float64),
        "timestamp": np.arange(n, dtype=np.int64) * 1000000,
        "notional": (price * rng.exponential(100, n)).astype(np.float64),
        "ln_price": np.log(price).astype(np.float64),
        "sqrt_price": np.sqrt(price).astype(np.float64),
        "recip_price": (1.0 / price).astype(np.float64),
        "round_lot": (rng.exponential(100, n) % 100 == 0).astype(np.uint8),
        "odd_lot": (rng.exponential(100, n) < 100).astype(np.uint8),
        "direction": np.sign(returns).astype(np.int8),
    }


# ── Aligned binary (baseline) ──

def write_aligned(path, data):
    cols = list(data.keys())
    n_rows = len(next(iter(data.values())))
    with open(path, "wb") as f:
        header = {"magic": "MKTG", "version": 1, "n_rows": n_rows, "columns": []}
        offset = ALIGN * 16
        for name in cols:
            arr = data[name]
            offset = (offset + ALIGN - 1) // ALIGN * ALIGN
            header["columns"].append({
                "name": name, "dtype": arr.dtype.name,
                "offset": offset, "nbytes": arr.nbytes,
            })
            offset += arr.nbytes
        header_json = json.dumps(header).encode("utf-8")
        f.write(header_json)
        f.write(b"\x00" * (ALIGN * 16 - len(header_json)))
        for col_info in header["columns"]:
            target = col_info["offset"]
            cur = f.tell()
            if cur < target:
                f.write(b"\x00" * (target - cur))
            data[col_info["name"]].tofile(f)


def read_aligned(path, columns=None):
    with open(path, "rb") as f:
        header_bytes = f.read(ALIGN * 16)
        header = json.loads(header_bytes.split(b"\x00")[0])
        result = {}
        for col in header["columns"]:
            if columns is not None and col["name"] not in columns:
                continue
            f.seek(col["offset"])
            dtype = np.dtype(col["dtype"])
            result[col["name"]] = np.frombuffer(f.read(col["nbytes"]), dtype=dtype).copy()
    return result


# ── Hybrid (per-column encoding) ──

def write_hybrid(path, data):
    n_rows = len(next(iter(data.values())))
    cols = list(data.keys())
    header_size = 2048

    with open(path, "wb") as f:
        header = {"magic": "MKTF", "version": 2, "n_rows": n_rows, "columns": []}
        offset = header_size
        col_data = []

        for name in cols:
            arr = data[name]
            offset = (offset + ALIGN - 1) // ALIGN * ALIGN

            if arr.dtype == np.float64:
                raw = arr.tobytes()
                encoding = "raw"
            elif name == "timestamp" and arr.dtype == np.int64:
                delta = np.diff(arr, prepend=arr[0])
                raw = lz4.frame.compress(delta.tobytes())
                encoding = "delta_lz4"
            elif arr.dtype in (np.uint8,) and len(np.unique(arr)) <= 2:
                packed = np.packbits(arr.astype(np.uint8))
                raw = lz4.frame.compress(packed.tobytes())
                encoding = "bitmask_lz4"
            elif arr.dtype in (np.uint8, np.int8):
                raw = lz4.frame.compress(arr.tobytes())
                encoding = "lz4"
            else:
                raw = arr.tobytes()
                encoding = "raw"

            header["columns"].append({
                "name": name, "dtype": arr.dtype.name,
                "offset": offset, "nbytes": len(raw),
                "encoding": encoding,
            })
            col_data.append((offset, raw))
            offset += len(raw)

        header_json = json.dumps(header).encode("utf-8")
        assert len(header_json) < header_size
        f.write(header_json)
        f.write(b"\x00" * (header_size - len(header_json)))

        for target, raw in col_data:
            cur = f.tell()
            if cur < target:
                f.write(b"\x00" * (target - cur))
            f.write(raw)


def read_hybrid(path, columns=None):
    with open(path, "rb") as f:
        header_bytes = f.read(2048)
        header = json.loads(header_bytes.split(b"\x00")[0])
        n_rows = header["n_rows"]
        result = {}

        for col in header["columns"]:
            if columns is not None and col["name"] not in columns:
                continue
            f.seek(col["offset"])
            raw = f.read(col["nbytes"])
            enc = col["encoding"]
            dtype = np.dtype(col["dtype"])

            if enc == "raw":
                result[col["name"]] = np.frombuffer(raw, dtype=dtype).copy()
            elif enc == "delta_lz4":
                dec = lz4.frame.decompress(raw)
                delta = np.frombuffer(dec, dtype=dtype)
                result[col["name"]] = np.cumsum(delta)
            elif enc == "bitmask_lz4":
                dec = lz4.frame.decompress(raw)
                packed = np.frombuffer(dec, dtype=np.uint8)
                unpacked = np.unpackbits(packed)[:n_rows]
                result[col["name"]] = unpacked.astype(dtype)
            elif enc == "lz4":
                dec = lz4.frame.decompress(raw)
                result[col["name"]] = np.frombuffer(dec, dtype=dtype).copy()

    return result


def time_fn(fn, n_runs=30):
    """Time a function, return list of ms timings."""
    fn()  # warmup
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


def stats(times):
    s = sorted(times)
    n = len(s)
    m = sum(s) / n
    sd = (sum((x - m) ** 2 for x in s) / (n - 1)) ** 0.5 if n > 1 else 0
    return m, sd, s[n // 2], s[0], s[-1]


def main():
    data = generate_test_data()
    n_runs = 30

    print("=" * 70)
    print("EXPERIMENT 2: Hybrid Per-Column Encoding vs Aligned Binary")
    print("=" * 70)
    print()

    path_h = tempfile.mktemp(suffix=".mktf2")
    path_a = tempfile.mktemp(suffix=".mktg")

    write_hybrid(path_h, data)
    write_aligned(path_a, data)

    sz_h = os.path.getsize(path_h) / 1e6
    sz_a = os.path.getsize(path_a) / 1e6
    print(f"Hybrid size:  {sz_h:.2f} MB")
    print(f"Aligned size: {sz_a:.2f} MB")
    print(f"Savings:      {(1 - sz_h / sz_a) * 100:.1f}%")
    print()

    # Verify correctness
    d_h = read_hybrid(path_h)
    d_a = read_aligned(path_a)
    ok = True
    for k in data:
        if not np.array_equal(data[k], d_h[k]):
            print(f"  MISMATCH hybrid: {k}")
            ok = False
        if not np.array_equal(data[k], d_a[k]):
            print(f"  MISMATCH aligned: {k}")
            ok = False
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    print()

    # ── Full read ──
    for label, read_fn, path in [("Aligned", read_aligned, path_a), ("Hybrid", read_hybrid, path_h)]:
        t = time_fn(lambda: read_fn(path), n_runs)
        m, sd, p50, mn, mx = stats(t)
        print(f"{label:8s} full read:   {m:7.2f}ms +/- {sd:5.2f}  [p50={p50:.2f}, min={mn:.2f}, max={mx:.2f}]")

    print()

    # ── 1-col read (price — raw in both) ──
    for label, read_fn, path in [("Aligned", read_aligned, path_a), ("Hybrid", read_hybrid, path_h)]:
        t = time_fn(lambda: read_fn(path, columns=["price"]), n_runs)
        m, sd, p50, mn, mx = stats(t)
        print(f"{label:8s} 1-col read:  {m:7.2f}ms +/- {sd:5.2f}  [p50={p50:.2f}]")

    print()

    # ── 3-col read ──
    sel3 = ["price", "size", "timestamp"]
    for label, read_fn, path in [("Aligned", read_aligned, path_a), ("Hybrid", read_hybrid, path_h)]:
        t = time_fn(lambda: read_fn(path, columns=sel3), n_runs)
        m, sd, p50, mn, mx = stats(t)
        print(f"{label:8s} 3-col read:  {m:7.2f}ms +/- {sd:5.2f}  [p50={p50:.2f}]")

    print()

    # ── GPU pipeline ──
    if GPU:
        print("GPU Host-to-Device pipeline:")
        for label, read_fn, path in [("Aligned", read_aligned, path_a), ("Hybrid", read_hybrid, path_h)]:
            # warmup
            d = read_fn(path)
            for a in d.values():
                cp.asarray(a)
            cp.cuda.Device(0).synchronize()

            times = []
            for _ in range(n_runs):
                gc.disable()
                t0 = time.perf_counter_ns()
                d = read_fn(path)
                for a in d.values():
                    cp.asarray(a)
                cp.cuda.Device(0).synchronize()
                t1 = time.perf_counter_ns()
                gc.enable()
                times.append((t1 - t0) / 1e6)

            m, sd, p50, mn, mx = stats(times)
            print(f"  {label:8s} disk->GPU:  {m:7.2f}ms +/- {sd:5.2f}  [p50={p50:.2f}, min={mn:.2f}]")
    print()

    # ── Write speed ──
    print("Write speed:")
    for label, write_fn, path in [("Aligned", write_aligned, path_a), ("Hybrid", write_hybrid, path_h)]:
        t = time_fn(lambda: write_fn(path, data), n_runs)
        m, sd, p50, mn, mx = stats(t)
        print(f"  {label:8s} write:      {m:7.2f}ms +/- {sd:5.2f}  [p50={p50:.2f}]")
    print()

    # Cleanup
    os.unlink(path_h)
    os.unlink(path_a)

    print("=" * 70)
    print("VERDICT")
    print("=" * 70)


if __name__ == "__main__":
    main()
