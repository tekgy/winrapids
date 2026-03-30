"""Convert K01P01 parquet files to MKTF format.

Reads existing K01P01 parquet files and writes MKTF equivalents alongside.
Drops string columns (DO04 ticker, DO06 id, DO07 conditions) — MKTF stores
only numeric data. Keeps DO01 (price/f32), DO02 (size/f32), DO03 (ts/i64),
DO05 (exchange/i32), DO08 (is_trf/bool).

Usage:
    python scripts/convert_k01_parquet_to_mktf.py --ticker AAPL
    python scripts/convert_k01_parquet_to_mktf.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "R:/fintek")

from trunk.backends.mktf.writer import write_mktf

DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

# Columns to keep (numeric only — strings dropped)
KEEP_COLUMNS = [
    "K01P01.DI01DO01",  # price (float32)
    "K01P01.DI01DO02",  # size (float32)
    "K01P01.DI01DO03",  # participant_timestamp (int64)
    "K01P01.DI01DO05",  # exchange (int32)
    "K01P01.DI01DO08",  # is_trf (bool)
]


def convert_one(ticker: str) -> dict:
    """Convert one ticker's K01P01 parquet to MKTF.

    Returns dict with stats: {ticker, n_rows, parquet_bytes, mktf_bytes, elapsed_ms}.
    """
    ticker_dir = DATA_ROOT / ticker
    parquet_path = ticker_dir / "K01P01.TI00TO00.parquet"
    mktf_path = ticker_dir / "K01P01.TI00TO00.mktf"

    if not parquet_path.exists():
        return {"ticker": ticker, "error": "parquet not found"}

    t0 = time.perf_counter()

    # Read parquet
    df = pl.read_parquet(parquet_path, columns=KEEP_COLUMNS)
    n_rows = len(df)

    if n_rows == 0:
        return {"ticker": ticker, "error": "empty parquet (0 rows)"}

    # Convert to numpy dict
    columns: dict[str, np.ndarray] = {}
    for col in KEEP_COLUMNS:
        series = df[col]
        if series.dtype == pl.Boolean:
            columns[col] = series.to_numpy().astype(np.uint8)
        else:
            columns[col] = series.to_numpy()

    # Extract day from timestamp
    ts_arr = columns["K01P01.DI01DO03"]
    day_ns = 86_400_000_000_000
    day_start = int((ts_arr[0] // day_ns) * day_ns)
    # Format as date string
    import datetime
    day_str = datetime.datetime.fromtimestamp(
        day_start / 1e9, tz=datetime.timezone.utc
    ).strftime("%Y-%m-%d")

    # Write MKTF (K01 = irreplaceable source data → safe=True)
    header = write_mktf(
        mktf_path,
        columns,
        leaf_id="K01P01.TI00TO00",
        ticker=ticker,
        day=day_str,
        safe=True,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    parquet_bytes = os.path.getsize(parquet_path)
    mktf_bytes = os.path.getsize(mktf_path)

    return {
        "ticker": ticker,
        "n_rows": n_rows,
        "n_cols": len(KEEP_COLUMNS),
        "parquet_bytes": parquet_bytes,
        "mktf_bytes": mktf_bytes,
        "ratio": mktf_bytes / parquet_bytes,
        "elapsed_ms": elapsed_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert K01P01 parquet to MKTF")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str, help="Single ticker to convert")
    group.add_argument("--all", action="store_true", help="Convert all tickers")
    args = parser.parse_args()

    if args.ticker:
        result = convert_one(args.ticker)
        if "error" in result:
            print(f"ERROR: {result['ticker']}: {result['error']}")
            return
        print(f"Converted {result['ticker']}:")
        print(f"  Rows:    {result['n_rows']:,}")
        print(f"  Parquet: {result['parquet_bytes']:,} bytes ({result['parquet_bytes']/1024/1024:.1f} MB)")
        print(f"  MKTF:    {result['mktf_bytes']:,} bytes ({result['mktf_bytes']/1024/1024:.1f} MB)")
        print(f"  Ratio:   {result['ratio']:.2f}x")
        print(f"  Time:    {result['elapsed_ms']:.0f}ms")
        return

    # Batch all tickers
    tickers = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Converting {len(tickers)} tickers...")

    t0 = time.perf_counter()
    total_parquet = 0
    total_mktf = 0
    errors = 0
    converted = 0

    for i, ticker in enumerate(tickers):
        result = convert_one(ticker)
        if "error" in result:
            errors += 1
            continue

        converted += 1
        total_parquet += result["parquet_bytes"]
        total_mktf += result["mktf_bytes"]

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(tickers) - i - 1) / rate
            print(f"  [{i+1}/{len(tickers)}] {ticker:6s}  "
                  f"{result['mktf_bytes']/1024/1024:.1f}MB  "
                  f"{rate:.0f} tickers/s  ETA {eta:.0f}s")

    total_s = time.perf_counter() - t0
    print(f"\n=== BATCH COMPLETE ===")
    print(f"  Tickers:   {converted:,} converted, {errors} errors")
    print(f"  Parquet:   {total_parquet/1024/1024/1024:.2f} GB")
    print(f"  MKTF:      {total_mktf/1024/1024/1024:.2f} GB")
    print(f"  Ratio:     {total_mktf/max(total_parquet,1):.2f}x")
    print(f"  Time:      {total_s:.1f}s ({converted/total_s:.0f} tickers/s)")


if __name__ == "__main__":
    main()
