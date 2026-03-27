"""Source-only MKTF: store 5 columns, GPU recomputes everything.

Naturalist insight: GPU recomputes 13 derived features in ~0.1ms.
Reading those derived features from NVMe takes ~12.5ms. Don't store derivables.

Source columns (what MKTF stores):
  price     float32  2.39MB
  size      float32  2.39MB
  timestamp int64    4.78MB
  conditions uint32  2.39MB
  exchange  uint8    0.60MB
  ────────────────────────────
  TOTAL:             12.56MB

Derived columns (GPU recomputes in ~0.1ms):
  ln_price     = log(price)
  sqrt_price   = sqrt(price)
  recip_price  = 1.0 / price
  notional     = price * size
  ln_notional  = log(notional)
  direction    = sign(diff(price))
  is_round_lot = (size % 100 == 0)
  is_odd_lot   = (size < 100)
  ...plus 5 more from K01 spec

This benchmark tests:
1. Source-only MKTF: read 5 cols (12.6MB) + GPU recompute + H2D = ?ms
2. Pre-computed MKTF: read 13 cols (47.8MB) + H2D (no recompute) = ?ms
3. Which is faster end-to-end?

Also benchmarks FP32 vs FP64 kernel execution.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cupy as cp
import numpy as np

# Import our MKTF v3 writer/reader
import sys
sys.path.insert(0, str(Path(__file__).parent))
from mktf_v3 import write_mktf, read_data, read_header, AAPL_PATH, COL_MAP, CONDITION_BITS


def load_source_columns() -> dict[str, np.ndarray]:
    """Load only the 5 source columns from real AAPL data."""
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(AAPL_PATH))
    raw = {new: tbl.column(old).to_numpy() for old, new in COL_MAP.items()}
    n = len(raw["price"])

    cols = {}
    cols["price"] = raw["price"].astype(np.float32)
    cols["size"] = raw["size"].astype(np.float32)
    cols["timestamp"] = raw["timestamp"].astype(np.int64)
    cols["exchange"] = raw["exchange"].astype(np.uint8)

    # Bitmask conditions
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


def load_precomputed_columns() -> dict[str, np.ndarray]:
    """Load source columns + compute all derived columns (13 total)."""
    cols = load_source_columns()
    price = cols["price"].astype(np.float64)
    size = cols["size"].astype(np.float64)

    # Derived columns (what the GPU would compute)
    cols["ln_price"] = np.log(price).astype(np.float32)
    cols["sqrt_price"] = np.sqrt(price).astype(np.float32)
    cols["recip_price"] = (1.0 / price).astype(np.float32)
    cols["notional"] = (price * size).astype(np.float32)
    cols["ln_notional"] = np.log(price * size).astype(np.float32)
    cols["ln_size"] = np.log(size).astype(np.float32)

    # Float32 for derived features (navigator: bf16 kills deltas, f32 is fine)
    cols["is_round_lot"] = ((size % 100) == 0).astype(np.uint8)

    return cols


# ══════════════════════════════════════════════════════════════════
# FP32 GPU RECOMPUTE KERNEL
# ══════════════════════════════════════════════════════════════════

# Single fused kernel: 7 outputs from 2 inputs. One launch, SFU-accelerated.
_FUSED_POINTWISE_F32 = cp.RawKernel(r'''
extern "C" __global__
void fused_pointwise_f32(
    const float* __restrict__ price,
    const float* __restrict__ size,
    float* __restrict__ ln_price,
    float* __restrict__ sqrt_price,
    float* __restrict__ recip_price,
    float* __restrict__ notional,
    float* __restrict__ ln_notional,
    float* __restrict__ ln_size,
    unsigned char* __restrict__ is_round_lot,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float p = price[idx];
    float s = size[idx];

    // SFU-accelerated transcendentals (logf, sqrtf use special function units)
    ln_price[idx] = logf(fmaxf(p, 1e-38f));
    sqrt_price[idx] = sqrtf(fmaxf(p, 0.0f));
    recip_price[idx] = 1.0f / p;

    float not_val = p * s;
    notional[idx] = not_val;
    ln_notional[idx] = logf(fmaxf(not_val, 1e-38f));
    ln_size[idx] = logf(fmaxf(s, 1e-38f));
    is_round_lot[idx] = (((int)s) % 100 == 0) ? 1 : 0;
}
''', 'fused_pointwise_f32')


# Same kernel in FP64 for comparison
_FUSED_POINTWISE_F64 = cp.RawKernel(r'''
extern "C" __global__
void fused_pointwise_f64(
    const double* __restrict__ price,
    const double* __restrict__ size,
    double* __restrict__ ln_price,
    double* __restrict__ sqrt_price,
    double* __restrict__ recip_price,
    double* __restrict__ notional,
    double* __restrict__ ln_notional,
    double* __restrict__ ln_size,
    unsigned char* __restrict__ is_round_lot,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double p = price[idx];
    double s = size[idx];

    ln_price[idx] = log(fmax(p, 1e-308));
    sqrt_price[idx] = sqrt(fmax(p, 0.0));
    recip_price[idx] = 1.0 / p;

    double not_val = p * s;
    notional[idx] = not_val;
    ln_notional[idx] = log(fmax(not_val, 1e-308));
    ln_size[idx] = log(fmax(s, 1e-308));
    is_round_lot[idx] = (((long long)s) % 100 == 0) ? 1 : 0;
}
''', 'fused_pointwise_f64')


def gpu_recompute_f32(price_gpu, size_gpu, n):
    """Run FP32 fused pointwise kernel. Returns dict of GPU arrays."""
    ln_price = cp.empty(n, dtype=cp.float32)
    sqrt_price = cp.empty(n, dtype=cp.float32)
    recip_price = cp.empty(n, dtype=cp.float32)
    notional = cp.empty(n, dtype=cp.float32)
    ln_notional = cp.empty(n, dtype=cp.float32)
    ln_size = cp.empty(n, dtype=cp.float32)
    is_round_lot = cp.empty(n, dtype=cp.uint8)

    threads = 256
    blocks = (n + threads - 1) // threads
    _FUSED_POINTWISE_F32(
        (blocks,), (threads,),
        (price_gpu, size_gpu,
         ln_price, sqrt_price, recip_price,
         notional, ln_notional, ln_size, is_round_lot,
         n)
    )
    return {
        "ln_price": ln_price, "sqrt_price": sqrt_price,
        "recip_price": recip_price, "notional": notional,
        "ln_notional": ln_notional, "ln_size": ln_size,
        "is_round_lot": is_round_lot,
    }


def gpu_recompute_f64(price_gpu, size_gpu, n):
    """Run FP64 fused pointwise kernel."""
    ln_price = cp.empty(n, dtype=cp.float64)
    sqrt_price = cp.empty(n, dtype=cp.float64)
    recip_price = cp.empty(n, dtype=cp.float64)
    notional = cp.empty(n, dtype=cp.float64)
    ln_notional = cp.empty(n, dtype=cp.float64)
    ln_size = cp.empty(n, dtype=cp.float64)
    is_round_lot = cp.empty(n, dtype=cp.uint8)

    threads = 256
    blocks = (n + threads - 1) // threads
    _FUSED_POINTWISE_F64(
        (blocks,), (threads,),
        (price_gpu, size_gpu,
         ln_price, sqrt_price, recip_price,
         notional, ln_notional, ln_size, is_round_lot,
         n)
    )
    return {
        "ln_price": ln_price, "sqrt_price": sqrt_price,
        "recip_price": recip_price, "notional": notional,
        "ln_notional": ln_notional, "ln_size": ln_size,
        "is_round_lot": is_round_lot,
    }


def main():
    print("=" * 78)
    print("SOURCE-ONLY MKTF vs PRE-COMPUTED — Real AAPL (598,057 ticks)")
    print("=" * 78)
    print()

    # ── Prepare data ───────────────────────────────────────────
    print("Loading data...")
    src_cols = load_source_columns()
    pre_cols = load_precomputed_columns()

    src_size = sum(arr.nbytes for arr in src_cols.values())
    pre_size = sum(arr.nbytes for arr in pre_cols.values())
    print(f"  Source-only: {len(src_cols)} cols, {src_size/1e6:.1f}MB")
    print(f"  Pre-computed: {len(pre_cols)} cols, {pre_size/1e6:.1f}MB")
    print(f"  Ratio: {pre_size/src_size:.1f}x larger with pre-computed")
    print()

    # ── Write both MKTF files ──────────────────────────────────
    tmp_src = tempfile.mktemp(suffix=".mktf")
    tmp_pre = tempfile.mktemp(suffix=".mktf")

    write_mktf(tmp_src, src_cols,
               leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
               metadata={"mode": "source-only"})
    write_mktf(tmp_pre, pre_cols,
               leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
               metadata={"mode": "pre-computed"})

    src_file = os.path.getsize(tmp_src)
    pre_file = os.path.getsize(tmp_pre)
    print(f"  Source-only file: {src_file/1e6:.2f}MB")
    print(f"  Pre-computed file: {pre_file/1e6:.2f}MB")
    print()

    # ── Pipeline A: Source-only → read → H2D → GPU recompute ──
    print("PIPELINE A: Source-only MKTF -> GPU recompute (FP32)")
    print("-" * 60)

    n = len(src_cols["price"])

    # Warmup
    for _ in range(5):
        data = read_data(tmp_src)
        p_gpu = cp.asarray(data["price"])
        s_gpu = cp.asarray(data["size"])
        derived = gpu_recompute_f32(p_gpu, s_gpu, n)
        cp.cuda.Stream.null.synchronize()
        del derived, p_gpu, s_gpu, data

    # Timed: full pipeline
    times_a = []
    for _ in range(30):
        t0 = time.perf_counter()
        data = read_data(tmp_src)
        t_read = time.perf_counter()
        p_gpu = cp.asarray(data["price"])
        s_gpu = cp.asarray(data["size"])
        ts_gpu = cp.asarray(data["timestamp"])
        ex_gpu = cp.asarray(data["exchange"])
        cond_gpu = cp.asarray(data["conditions"])
        cp.cuda.Stream.null.synchronize()
        t_h2d = time.perf_counter()
        derived = gpu_recompute_f32(p_gpu, s_gpu, n)
        cp.cuda.Stream.null.synchronize()
        t_compute = time.perf_counter()
        times_a.append({
            "read": (t_read - t0) * 1000,
            "h2d": (t_h2d - t_read) * 1000,
            "compute": (t_compute - t_h2d) * 1000,
            "total": (t_compute - t0) * 1000,
        })
        del derived, p_gpu, s_gpu, ts_gpu, ex_gpu, cond_gpu, data

    avg_a = {k: np.mean([t[k] for t in times_a]) for k in times_a[0]}
    print(f"  Read:     {avg_a['read']:.2f}ms  ({src_file/1e6:.1f}MB)")
    print(f"  H2D:      {avg_a['h2d']:.2f}ms")
    print(f"  Compute:  {avg_a['compute']:.3f}ms  (fused FP32 kernel, 7 outputs)")
    print(f"  TOTAL:    {avg_a['total']:.2f}ms")
    print()

    # ── Pipeline B: Pre-computed → read → H2D (no recompute) ──
    print("PIPELINE B: Pre-computed MKTF -> GPU (just H2D)")
    print("-" * 60)

    # Warmup
    for _ in range(5):
        data = read_data(tmp_pre)
        gpu = {k: cp.asarray(v) for k, v in data.items()}
        cp.cuda.Stream.null.synchronize()
        del gpu, data

    times_b = []
    for _ in range(30):
        t0 = time.perf_counter()
        data = read_data(tmp_pre)
        t_read = time.perf_counter()
        gpu = {k: cp.asarray(v) for k, v in data.items()}
        cp.cuda.Stream.null.synchronize()
        t_h2d = time.perf_counter()
        times_b.append({
            "read": (t_read - t0) * 1000,
            "h2d": (t_h2d - t_read) * 1000,
            "total": (t_h2d - t0) * 1000,
        })
        del gpu, data

    avg_b = {k: np.mean([t[k] for t in times_b]) for k in times_b[0]}
    print(f"  Read:     {avg_b['read']:.2f}ms  ({pre_file/1e6:.1f}MB)")
    print(f"  H2D:      {avg_b['h2d']:.2f}ms")
    print(f"  TOTAL:    {avg_b['total']:.2f}ms")
    print()

    # ── FP32 vs FP64 kernel benchmark ──────────────────────────
    print("FP32 vs FP64 FUSED POINTWISE KERNEL (7 outputs from 2 inputs)")
    print("-" * 60)

    price_f32 = cp.asarray(src_cols["price"])
    size_f32 = cp.asarray(src_cols["size"])
    price_f64 = price_f32.astype(cp.float64)
    size_f64 = size_f32.astype(cp.float64)

    # Warmup
    for _ in range(20):
        gpu_recompute_f32(price_f32, size_f32, n)
        gpu_recompute_f64(price_f64, size_f64, n)
    cp.cuda.Stream.null.synchronize()

    # FP32
    t0 = time.perf_counter()
    for _ in range(200):
        gpu_recompute_f32(price_f32, size_f32, n)
    cp.cuda.Stream.null.synchronize()
    t_f32 = (time.perf_counter() - t0) / 200

    # FP64
    t0 = time.perf_counter()
    for _ in range(200):
        gpu_recompute_f64(price_f64, size_f64, n)
    cp.cuda.Stream.null.synchronize()
    t_f64 = (time.perf_counter() - t0) / 200

    ratio = t_f64 / t_f32
    print(f"  FP32:  {t_f32*1e6:.1f}us  ({t_f32*1000:.3f}ms)")
    print(f"  FP64:  {t_f64*1e6:.1f}us  ({t_f64*1000:.3f}ms)")
    print(f"  Ratio: {ratio:.1f}x faster with FP32")
    print()

    # ── Precision comparison ───────────────────────────────────
    print("PRECISION: FP32 vs FP64 (max relative error)")
    print("-" * 60)
    d32 = gpu_recompute_f32(price_f32, size_f32, n)
    d64 = gpu_recompute_f64(price_f64, size_f64, n)
    cp.cuda.Stream.null.synchronize()

    for key in ["ln_price", "sqrt_price", "recip_price", "notional"]:
        ref = d64[key].get().astype(np.float64)
        test = d32[key].get().astype(np.float64)
        mask = ref != 0
        rel_err = np.max(np.abs((test[mask] - ref[mask]) / ref[mask]))
        print(f"  {key:<16s}: max rel error = {rel_err:.2e}")

    # ── Head-to-head summary ───────────────────────────────────
    print()
    print("=" * 78)
    print("HEAD-TO-HEAD SUMMARY")
    print("=" * 78)
    print()
    print(f"  {'':30s} {'Source-only':>14s} {'Pre-computed':>14s} {'Winner':>10s}")
    print(f"  {'-'*70}")
    print(f"  {'File size':30s} {src_file/1e6:>12.1f}MB {pre_file/1e6:>12.1f}MB "
          f"{'src' if src_file < pre_file else 'pre':>10s}")
    print(f"  {'Disk read':30s} {avg_a['read']:>11.2f}ms {avg_b['read']:>11.2f}ms "
          f"{'src' if avg_a['read'] < avg_b['read'] else 'pre':>10s}")
    print(f"  {'H2D':30s} {avg_a['h2d']:>11.2f}ms {avg_b['h2d']:>11.2f}ms "
          f"{'src' if avg_a['h2d'] < avg_b['h2d'] else 'pre':>10s}")
    print(f"  {'GPU compute':30s} {avg_a['compute']:>11.3f}ms {'0.000':>12s} "
          f"{'pre':>10s}")
    print(f"  {'TOTAL pipeline':30s} {avg_a['total']:>11.2f}ms {avg_b['total']:>11.2f}ms "
          f"{'src' if avg_a['total'] < avg_b['total'] else 'pre':>10s}")
    saved = avg_b['total'] - avg_a['total']
    print(f"  {'Savings':30s} {saved:>11.2f}ms per ticker")
    print()

    # Universe
    print("UNIVERSE (4,604 tickers)")
    print(f"  Source-only:   {avg_a['total']/1000*4604:>6.1f}s = {avg_a['total']/1000*4604/60:.1f}min")
    print(f"  Pre-computed:  {avg_b['total']/1000*4604:>6.1f}s = {avg_b['total']/1000*4604/60:.1f}min")
    print(f"  Storage saved: {(pre_file-src_file)/1e6*4604/1e3:.1f}GB")
    print(f"  FP32 kernel:   {t_f32*4604*1e3:.1f}ms total ({ratio:.0f}x faster than FP64)")

    # Cleanup
    os.unlink(tmp_src)
    os.unlink(tmp_pre)


if __name__ == "__main__":
    main()
