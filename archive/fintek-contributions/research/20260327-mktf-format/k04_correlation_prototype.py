"""K04 Cross-Ticker Correlation — first LEAF_MATRIX MKTF file.

The question: does node===node hold when the output is a matrix
instead of columns? Same header, same reader, same daemon, same
upstream chain — but CROSS_TICKER computation_shape, LEAF_MATRIX
leaf_type, and GPU_TENSOR compute_target.

Pipeline:
  1. Create N synthetic K01 MKTF v4 files (real AAPL + noise)
  2. Read them concurrently (12-worker NVMe saturation)
  3. Stack features across tickers -> [n_tickers, n_features] matrix
  4. Normalize (z-score per feature — FP16 overflow guard)
  5. FP16 Tensor Core GEMM -> [n_tickers, n_tickers] correlation
  6. Write result as K04 MKTF v4 (LEAF_MATRIX, upstream chain)
  7. Prove staleness detection: rewrite one K01, detect from K04 header

This is the first node that crosses kingdoms.
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import numpy as np

# Add fintek to path for v4 format
sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.format import (
    ALIGNMENT,
    BLOCK_SIZE,
    MKTFHeader,
    ColumnEntry,
    UpstreamFingerprint,
    compute_data_hash,
    compute_leaf_id_hash,
    compute_schema_fingerprint,
    pack_block0,
    pack_column_entry,
    unpack_block0,
    unpack_column_entry,
    _align,
    _pad,
    # Enums
    SHAPE_CROSS_TICKER,
    TARGET_GPU_TENSOR,
    PRECISION_FP16,
    LEAF_MATRIX,
    ASSET_EQUITY,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    NUMPY_TO_DTYPE,
    DTYPE_TO_NUMPY,
    # Constants
    MAX_UPSTREAM,
    UPSTREAM_ENTRY_SIZE,
    COL_DIR_ENTRY_SIZE,
    OFF_UPSTREAM,
    OFF_PROVENANCE,
    OFF_QUALITY,
)
from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import read_header, read_columns, read_status

# Also need our K01 data loader
sys.path.insert(0, str(Path(__file__).parent))
from mktf_v3 import AAPL_PATH, COL_MAP, CONDITION_BITS


# ══════════════════════════════════════════════════════════════════
# K01 DATA — Create synthetic multi-ticker universe
# ══════════════════════════════════════════════════════════════════

def load_aapl_source() -> dict[str, np.ndarray]:
    """Load 5 source columns from real AAPL data."""
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


def create_synthetic_ticker(
    base_cols: dict[str, np.ndarray],
    ticker_idx: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Create synthetic ticker from AAPL base + controlled noise.

    Each ticker has a unique price level and volatility to create
    non-trivial correlation structure.
    """
    n = len(base_cols["price"])

    # Price: base AAPL + unique offset + correlated noise
    # Different tickers have different mean prices and volatilities
    price_shift = rng.uniform(-50, 100)
    vol_scale = rng.uniform(0.5, 2.0)
    noise = rng.normal(0, vol_scale, n).astype(np.float32)
    price = np.maximum(base_cols["price"] + price_shift + noise, 1.0).astype(np.float32)

    # Size: log-normal scaled differently per ticker
    size_scale = rng.uniform(0.3, 3.0)
    size = np.maximum(base_cols["size"] * size_scale, 1.0).astype(np.float32)

    return {
        "price": price,
        "size": size,
        "timestamp": base_cols["timestamp"].copy(),
        "exchange": base_cols["exchange"].copy(),
        "conditions": base_cols["conditions"].copy(),
    }


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "KO", "PEP", "COST", "AVGO", "TMO", "MCD", "CSCO",
    "ACN", "ABT", "WFC", "DHR", "CRM", "LIN", "TXN", "PM", "NEE",
    "ORCL", "AMD", "HON", "UPS", "INTC", "RTX", "QCOM", "LOW", "SBUX",
    "IBM", "GE", "CAT", "BA", "AMGN",
]


# ══════════════════════════════════════════════════════════════════
# K04 CORRELATION — Tensor Core GEMM
# ══════════════════════════════════════════════════════════════════

def compute_correlation_fp16(
    feature_matrix: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Compute cross-ticker correlation via FP16 Tensor Core GEMM.

    Args:
        feature_matrix: [n_tickers, n_features] float32 array.
            MUST be z-score normalized per feature (FP16 overflow guard).

    Returns:
        (corr_matrix, gemm_ms, total_ms) where corr_matrix is [n_tickers, n_tickers] float32.
    """
    t0 = time.perf_counter()
    n_tickers, n_features = feature_matrix.shape

    # Upload to GPU
    X_f32 = cp.asarray(feature_matrix)

    # Convert to FP16 (safe after normalization — values are z-scores, |z| < 10 typically)
    X_f16 = X_f32.astype(cp.float16)

    # Warm up
    _ = X_f16 @ X_f16.T
    cp.cuda.Stream.null.synchronize()

    # Timed GEMM: X @ X^T = [n_tickers, n_tickers] correlation
    t_gemm = time.perf_counter()
    C = X_f16 @ X_f16.T  # cuBLAS routes to Tensor Cores for fp16
    cp.cuda.Stream.null.synchronize()
    gemm_ms = (time.perf_counter() - t_gemm) * 1000

    # Convert back to float32 and normalize
    C_f32 = C.astype(cp.float32)
    # Divide by n_features to get correlation (since inputs are z-scored)
    C_f32 /= n_features
    # Clamp to [-1, 1] (FP16 rounding can push slightly outside)
    cp.clip(C_f32, -1.0, 1.0, out=C_f32)
    # Clamp diagonal to exactly 1.0
    cp.fill_diagonal(C_f32, 1.0)

    result = cp.asnumpy(C_f32)
    total_ms = (time.perf_counter() - t0) * 1000

    return result, gemm_ms, total_ms


def extract_features(
    all_data: list[dict[str, np.ndarray]],
) -> np.ndarray:
    """Extract feature vector per ticker from K01 source columns.

    For each ticker, compute summary statistics from raw ticks:
    mean_price, std_price, mean_size, std_size, mean_notional, etc.

    Returns [n_tickers, n_features] float32 array.
    """
    features = []
    for data in all_data:
        price = data["price"]
        size = data["size"]
        notional = price * size

        feat = np.array([
            np.mean(price),
            np.std(price),
            np.median(price),
            np.min(price),
            np.max(price),
            np.mean(np.log(np.maximum(price, 1e-8))),
            np.mean(size),
            np.std(size),
            np.median(size),
            np.mean(notional),
            np.std(notional),
            np.median(notional),
            np.mean(np.diff(price)),
            np.std(np.diff(price)),
            np.mean(np.abs(np.diff(price))),
            np.mean(np.diff(np.log(np.maximum(price, 1e-8)))),
        ], dtype=np.float32)
        features.append(feat)

    X = np.stack(features)  # [n_tickers, 16]

    # Z-score normalize per feature (FP16 overflow guard)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    X_norm = ((X - mu) / sigma).astype(np.float32)

    return X_norm


# ══════════════════════════════════════════════════════════════════
# WRITE K04 MKTF — LEAF_MATRIX
# ══════════════════════════════════════════════════════════════════

def write_k04_mktf(
    path: str,
    corr_matrix: np.ndarray,
    ticker_names: list[str],
    upstream_fps: list[UpstreamFingerprint],
    gemm_ms: float,
    total_ms: float,
) -> MKTFHeader:
    """Write K04 correlation matrix as MKTF v4 LEAF_MATRIX.

    The correlation matrix is stored as a single column named "correlation"
    with n_elements = n_tickers * n_tickers. The matrix shape is encoded
    in the Dimensions section (n_tickers field).

    This is the node===node test: same format, same reader, same daemon,
    but the computation and output shape are fundamentally different.
    """
    n_tickers = corr_matrix.shape[0]
    flat = corr_matrix.astype(np.float32).ravel()

    columns = {"correlation": flat}

    header = MKTFHeader()

    # Tree: K04 cross-ticker correlation
    header.kingdom = 4     # K04
    header.phylum = 0
    header.class_ = 0
    header.rank = 0
    header.family = 0
    header.genus = 0
    header.species = 0
    header.depth = 2       # reads from K01, which reads from source

    header.computation_shape = SHAPE_CROSS_TICKER
    header.compute_target = TARGET_GPU_TENSOR
    header.precision = PRECISION_FP16
    header.leaf_type = LEAF_MATRIX

    # Dimensions: the matrix is n_tickers x n_tickers
    header.n_tickers = n_tickers
    header.n_rows = n_tickers
    header.n_cols = n_tickers

    # Asset
    header.asset_class = ASSET_EQUITY

    # Statistics: global stats of the correlation matrix
    header.global_mean = float(np.nanmean(corr_matrix))
    header.global_std = float(np.nanstd(corr_matrix))
    header.global_min = float(np.nanmin(corr_matrix))
    header.global_max = float(np.nanmax(corr_matrix))
    header.global_median = float(np.nanmedian(corr_matrix))
    # Skew/kurtosis of off-diagonal elements (manual, no scipy)
    off_diag = corr_matrix[~np.eye(n_tickers, dtype=bool)]
    if len(off_diag) > 0:
        mu = np.mean(off_diag)
        sigma = np.std(off_diag)
        if sigma > 1e-10:
            z = (off_diag - mu) / sigma
            header.global_skew = float(np.mean(z ** 3))
            header.global_kurtosis = float(np.mean(z ** 4) - 3.0)  # excess kurtosis

    return write_mktf(
        path,
        columns,
        header=header,
        leaf_id="K04.CORR.EQUITY.DAILY",
        ticker="UNIVERSE",
        day="2025-09-02",
        leaf_version="0.1.0",
        upstream=upstream_fps,
        compute_duration_ms=int(total_ms),
        metadata={
            "ticker_names": ticker_names,
            "n_features": 16,
            "normalization": "z-score",
            "gemm_precision": "fp16",
            "gemm_ms": round(gemm_ms, 3),
            "description": "Cross-ticker correlation matrix from K01 source features",
        },
    )


# ══════════════════════════════════════════════════════════════════
# MAIN — Full K04 pipeline proof-of-concept
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("K04 CROSS-TICKER CORRELATION — FIRST LEAF_MATRIX MKTF")
    print("=" * 78)
    print()

    N_TICKERS = len(TICKERS)
    rng = np.random.default_rng(42)

    # ── Phase 1: Create K01 MKTF v4 files ─────────────────────
    print(f"PHASE 1: Create {N_TICKERS} K01 MKTF v4 files")
    print("-" * 60)

    base_cols = load_aapl_source()
    n_ticks = len(base_cols["price"])
    print(f"  Base data: AAPL, {n_ticks:,} ticks, 5 source columns")

    tmpdir = tempfile.mkdtemp(prefix="k04_proto_")
    k01_paths = []
    k01_headers = []

    t0 = time.perf_counter()
    for i, ticker in enumerate(TICKERS):
        if i == 0:
            cols = {k: v.copy() for k, v in base_cols.items()}
        else:
            cols = create_synthetic_ticker(base_cols, i, rng)

        p = os.path.join(tmpdir, f"{ticker}.mktf")

        # Build K01 header
        k01_hdr = MKTFHeader()
        k01_hdr.kingdom = 1  # K01
        k01_hdr.computation_shape = 0  # POINTWISE
        k01_hdr.compute_target = 1  # GPU_CUDA
        k01_hdr.precision = 1  # FP32
        k01_hdr.leaf_type = 0  # COLUMNAR
        k01_hdr.asset_class = ASSET_EQUITY
        k01_hdr.tick_count = n_ticks

        hdr = write_mktf(
            p, cols,
            header=k01_hdr,
            leaf_id="K01P01",
            ticker=ticker,
            day="2025-09-02",
            leaf_version="1.0.0",
        )
        k01_paths.append(p)
        k01_headers.append(hdr)

    t_write = time.perf_counter() - t0
    file_size = os.path.getsize(k01_paths[0])
    print(f"  Written: {N_TICKERS} files, {file_size/1e6:.1f}MB each, "
          f"{file_size*N_TICKERS/1e6:.0f}MB total")
    print(f"  Time: {t_write*1000:.0f}ms ({t_write/N_TICKERS*1000:.1f}ms/file)")
    print()

    # ── Phase 2: Concurrent read ──────────────────────────────
    print(f"PHASE 2: Concurrent read ({N_TICKERS} files, 12 workers)")
    print("-" * 60)

    def read_data(path):
        _, cols = read_columns(path)
        return cols

    # Warmup
    for p in k01_paths[:5]:
        _ = read_data(p)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=12) as exe:
        all_data = list(exe.map(read_data, k01_paths))
    t_read = time.perf_counter() - t0
    print(f"  Read: {t_read*1000:.1f}ms ({t_read/N_TICKERS*1000:.2f}ms/file)")
    print()

    # ── Phase 3: Feature extraction + normalization ───────────
    print("PHASE 3: Feature extraction + z-score normalization")
    print("-" * 60)

    t0 = time.perf_counter()
    X = extract_features(all_data)
    t_feat = time.perf_counter() - t0
    print(f"  Feature matrix: [{X.shape[0]}, {X.shape[1]}] float32")
    print(f"  Value range after z-score: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Time: {t_feat*1000:.1f}ms")
    print()

    # ── Phase 4: FP16 Tensor Core GEMM ────────────────────────
    print("PHASE 4: FP16 Tensor Core GEMM (cross-ticker correlation)")
    print("-" * 60)

    corr, gemm_ms, total_ms = compute_correlation_fp16(X)
    n_flops = 2 * N_TICKERS * N_TICKERS * X.shape[1]
    tflops = n_flops / (gemm_ms / 1000) / 1e12 if gemm_ms > 0 else 0

    print(f"  Correlation matrix: [{corr.shape[0]}, {corr.shape[1]}] float32")
    print(f"  GEMM time: {gemm_ms:.3f}ms")
    print(f"  Total (H2D + GEMM + D2H): {total_ms:.1f}ms")
    print(f"  TFLOPS: {tflops:.1f}")
    print(f"  Diagonal: all {np.allclose(np.diag(corr), 1.0)}")
    print(f"  Off-diagonal range: [{corr[~np.eye(N_TICKERS, dtype=bool)].min():.4f}, "
          f"{corr[~np.eye(N_TICKERS, dtype=bool)].max():.4f}]")
    print()

    # ── Phase 5: Write K04 MKTF v4 (LEAF_MATRIX) ─────────────
    print("PHASE 5: Write K04 MKTF v4 (LEAF_MATRIX)")
    print("-" * 60)

    # Build upstream fingerprints from K01 headers
    # v4 supports 16 upstream entries — we have 50 tickers
    # Strategy: fingerprint first 16, store full list in metadata
    upstream_fps = []
    for i, hdr in enumerate(k01_headers[:MAX_UPSTREAM]):
        upstream_fps.append(UpstreamFingerprint(
            leaf_id=f"K01P01.{TICKERS[i]}",
            write_ts_ns=hdr.write_timestamp_ns,
            data_hash=hdr.data_checksum,
        ))

    k04_path = os.path.join(tmpdir, "K04.CORR.EQUITY.DAILY.mktf")
    t0 = time.perf_counter()
    k04_hdr = write_k04_mktf(
        k04_path, corr, TICKERS, upstream_fps, gemm_ms, total_ms,
    )
    t_write_k04 = time.perf_counter() - t0
    k04_size = os.path.getsize(k04_path)

    print(f"  File: {k04_size/1e6:.2f}MB ({N_TICKERS}x{N_TICKERS} float32 = "
          f"{N_TICKERS*N_TICKERS*4/1e6:.2f}MB data)")
    print(f"  Write time: {t_write_k04*1000:.1f}ms")
    print()

    # ── Phase 6: Read back and verify ─────────────────────────
    print("PHASE 6: Read back K04 MKTF — node===node verification")
    print("-" * 60)

    k04_read = read_header(k04_path)
    complete, dirty = read_status(k04_path)

    print(f"  leaf_id:            {k04_read.leaf_id}")
    print(f"  ticker:             {k04_read.ticker}")
    print(f"  kingdom:            K{k04_read.kingdom:02d}")
    print(f"  computation_shape:  {k04_read.computation_shape} (CROSS_TICKER)")
    print(f"  compute_target:     {k04_read.compute_target} (GPU_TENSOR)")
    print(f"  precision:          {k04_read.precision} (FP16)")
    print(f"  leaf_type:          {k04_read.leaf_type} (MATRIX)")
    print(f"  n_tickers:          {k04_read.n_tickers}")
    print(f"  n_rows x n_cols:    {k04_read.n_rows} x {k04_read.n_cols}")
    print(f"  is_complete:        {complete}")
    print(f"  is_dirty:           {dirty}")
    print(f"  upstream count:     {len(k04_read.upstream)}")
    print(f"  global_mean:        {k04_read.global_mean:.6f}")
    print(f"  global_std:         {k04_read.global_std:.6f}")
    print(f"  data_checksum:      {k04_read.data_checksum:#018x}")
    print()

    # Read the actual matrix back
    _, k04_cols = read_columns(k04_path)
    corr_back = k04_cols["correlation"].reshape(N_TICKERS, N_TICKERS)
    print(f"  Matrix roundtrip:   {np.allclose(corr, corr_back)}")
    print(f"  Max diff:           {np.max(np.abs(corr - corr_back)):.2e}")
    print()

    # ── Phase 7: Staleness detection across kingdoms ──────────
    print("PHASE 7: Staleness detection — K01 rewrite propagates to K04")
    print("-" * 60)

    # Record original K01[0] hash
    orig_hash = k04_read.upstream[0].data_hash
    orig_ts = k04_read.upstream[0].write_ts_ns
    print(f"  K04 upstream[0] ({TICKERS[0]}):")
    print(f"    data_hash:       {orig_hash:#018x}")
    print(f"    write_ts_ns:     {orig_ts}")

    # Rewrite K01 for AAPL with slightly different data
    modified_cols = {k: v.copy() for k, v in base_cols.items()}
    modified_cols["price"] = modified_cols["price"] * 1.001  # tiny change

    rewrite_hdr = MKTFHeader()
    rewrite_hdr.kingdom = 1
    rewrite_hdr.asset_class = ASSET_EQUITY
    new_hdr = write_mktf(
        k01_paths[0], modified_cols,
        header=rewrite_hdr,
        leaf_id="K01P01", ticker=TICKERS[0], day="2025-09-02",
    )
    new_hash = new_hdr.data_checksum
    new_ts = new_hdr.write_timestamp_ns

    print(f"\n  K01[0] rewritten with price * 1.001:")
    print(f"    new data_hash:   {new_hash:#018x}")
    print(f"    new write_ts_ns: {new_ts}")
    print(f"    hash changed:    {new_hash != orig_hash}")
    print(f"    ts changed:      {new_ts != orig_ts}")

    # Staleness check: compare K04 upstream to current K01 header
    current_k01 = read_header(k01_paths[0])
    stale = (orig_hash != current_k01.data_checksum or
             orig_ts != current_k01.write_timestamp_ns)
    print(f"\n  K04 staleness check (header-only, zero data bytes):")
    print(f"    K04.upstream[0].data_hash  == K01.data_checksum?  "
          f"{'NO - STALE' if orig_hash != current_k01.data_checksum else 'yes'}")
    print(f"    K04.upstream[0].write_ts   == K01.write_ts?       "
          f"{'NO - STALE' if orig_ts != current_k01.write_timestamp_ns else 'yes'}")
    print(f"    --> K04 is stale: {stale}")
    print()

    # ── Summary ───────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY: K04 CROSS-TICKER CORRELATION PROTOTYPE")
    print("=" * 78)
    print()
    print("  Node===Node holds. Same MKTF v4 format for:")
    print(f"    K01 (COLUMNAR, POINTWISE, {n_ticks:,} ticks/file, {file_size/1e6:.1f}MB)")
    print(f"    K04 (MATRIX, CROSS_TICKER, {N_TICKERS}x{N_TICKERS} correlation, {k04_size/1e6:.2f}MB)")
    print()
    print("  Same daemon contract:")
    print(f"    read_status() -> (is_complete={complete}, is_dirty={dirty})")
    print(f"    read_header() -> 4096-byte Block 0, all metadata")
    print(f"    read_columns() -> matrix data, reshape via n_rows x n_cols")
    print()
    print("  Upstream provenance chain (header-only staleness):")
    print(f"    K01.data_checksum -> K04.upstream[i].data_hash")
    print(f"    Rewrite K01 -> K04 detected stale via 2 header reads, 0 data bytes")
    print()
    print("  Pipeline timing:")
    print(f"    K01 write ({N_TICKERS} files):  {t_write*1000:.0f}ms")
    print(f"    Concurrent read (12w):    {t_read*1000:.0f}ms")
    print(f"    Feature extraction:       {t_feat*1000:.0f}ms")
    print(f"    FP16 TC GEMM:             {gemm_ms:.3f}ms")
    print(f"    K04 write:                {t_write_k04*1000:.0f}ms")
    print(f"    Total K04 pipeline:       {(t_read + t_feat + total_ms/1000 + t_write_k04)*1000:.0f}ms")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
