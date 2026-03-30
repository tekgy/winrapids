"""AMD 15-minute anomaly test: Is it options hedging?

Hypothesis: AMD shows 43.4x at 15min but only 3.4x at 30min because
market makers delta-hedge on a 15-minute cycle. If true, other heavily-
optioned names (NVDA, TSLA, META) should show the same 15min concentration,
while low-options names (BRK.B, JNJ) should not.

Prediction:
  High-options group (AMD, NVDA, TSLA, META): 15min/30min ratio >> 1
  Low-options group (BRK.B, JNJ, KO):        15min/30min ratio ~ 1
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

# Two groups to compare
HIGH_OPTIONS = ["AMD", "NVDA", "TSLA", "META"]
LOW_OPTIONS = ["BRK.B", "JNJ", "KO"]

# Focus on the institutional regime
FOCUS_CADENCES = [
    (1/60,   "1min"),
    (1/120,  "2min"),
    (1/300,  "5min"),
    (1/600,  "10min"),
    (1/900,  "15min"),
    (1/1200, "20min"),
    (1/1800, "30min"),
]


def load_rth_timestamps(ticker: str) -> np.ndarray:
    path = DATA_ROOT / ticker / "K01P01.TI00TO00.parquet"
    tbl = pq.read_table(str(path))
    timestamps_ns = tbl.column("K01P01.DI01DO03").to_numpy().astype(np.int64)
    timestamps_ns = np.sort(timestamps_ns)

    first_ts_s = timestamps_ns[0] / 1e9
    day_start = datetime.datetime.fromtimestamp(first_ts_s, tz=datetime.timezone.utc)
    day_date = day_start.date()
    market_open_utc = datetime.datetime(
        day_date.year, day_date.month, day_date.day,
        13, 30, 0, tzinfo=datetime.timezone.utc
    )
    market_close_utc = datetime.datetime(
        day_date.year, day_date.month, day_date.day,
        20, 0, 0, tzinfo=datetime.timezone.utc
    )
    open_ns = int(market_open_utc.timestamp() * 1e9)
    close_ns = int(market_close_utc.timestamp() * 1e9)

    rth_mask = (timestamps_ns >= open_ns) & (timestamps_ns <= close_ns)
    return timestamps_ns[rth_mask]


def compute_institutional_excess(timestamps_ns: np.ndarray) -> dict:
    iat_ns = np.diff(timestamps_ns)
    iat_us = iat_ns / 1000.0

    iat_nonzero = iat_us[iat_us > 0]
    n_nonzero = len(iat_nonzero)

    if n_nonzero < 1000:
        return {"too_few": True}

    iat_centered = iat_nonzero - iat_nonzero.mean()
    spectrum = np.fft.rfft(iat_centered.astype(np.float32))
    psd = np.abs(spectrum) ** 2 / n_nonzero
    freqs = np.fft.rfftfreq(n_nonzero)

    mean_iat_s = iat_nonzero.mean() / 1e6
    mean_sample_rate = 1.0 / mean_iat_s
    freqs_hz = freqs * mean_sample_rate

    total_power = psd[1:].sum()
    n_freq_bins = len(psd[1:])

    excess = {}
    for freq_target, label in FOCUS_CADENCES:
        if freq_target >= mean_sample_rate / 2:
            continue
        f_lo = freq_target * 0.95
        f_hi = freq_target * 1.05
        mask = (freqs_hz[1:] >= f_lo) & (freqs_hz[1:] <= f_hi)
        if mask.any():
            band_power = psd[1:][mask].sum()
            n_bins_in_band = mask.sum()
            expected = total_power * n_bins_in_band / n_freq_bins
            ex = band_power / expected if expected > 0 else 0
            excess[label] = round(ex, 2)

    return {
        "too_few": False,
        "n_ticks": len(timestamps_ns),
        "n_nonzero": n_nonzero,
        "excess": excess,
    }


def main():
    print("=" * 70)
    print("AMD 15-MINUTE ANOMALY TEST: Options Hedging Hypothesis")
    print("=" * 70)

    all_results = {}

    for group_name, tickers in [("HIGH-OPTIONS", HIGH_OPTIONS),
                                 ("LOW-OPTIONS", LOW_OPTIONS)]:
        print(f"\n{'=' * 70}")
        print(f"  {group_name}: {', '.join(tickers)}")
        print(f"{'=' * 70}")

        for ticker in tickers:
            try:
                ts = load_rth_timestamps(ticker)
                result = compute_institutional_excess(ts)
                all_results[ticker] = result

                if result["too_few"]:
                    print(f"\n  {ticker}: too few ticks")
                    continue

                print(f"\n  {ticker} ({result['n_ticks']:,} RTH ticks):")
                for label, ex in result["excess"].items():
                    bar = "*" * min(int(ex), 50)
                    print(f"    {label:>5s}: {ex:>6.1f}x  {bar}")

                # 15min/30min ratio
                ex15 = result["excess"].get("15min", 0)
                ex30 = result["excess"].get("30min", 0)
                if ex30 > 0:
                    ratio = ex15 / ex30
                    print(f"    15min/30min ratio: {ratio:.1f}x")

            except Exception as e:
                print(f"\n  {ticker}: ERROR {e}")

    # ── Summary comparison ────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("HYPOTHESIS TEST: 15min/30min ratio by options activity")
    print(f"{'=' * 70}")

    print(f"\n{'Ticker':>8s}  {'Group':>12s}  {'Ticks':>8s}  "
          f"{'5min':>6s}  {'10min':>6s}  {'15min':>6s}  {'20min':>6s}  {'30min':>6s}  {'15/30':>6s}")
    print("-" * 85)

    for group_name, tickers in [("HIGH-OPTIONS", HIGH_OPTIONS),
                                 ("LOW-OPTIONS", LOW_OPTIONS)]:
        for ticker in tickers:
            r = all_results.get(ticker, {})
            if r.get("too_few") or "excess" not in r:
                continue

            ex = r["excess"]
            ex15 = ex.get("15min", 0)
            ex30 = ex.get("30min", 0)
            ratio = ex15 / ex30 if ex30 > 0 else float('inf')

            print(f"{ticker:>8s}  {group_name:>12s}  {r['n_ticks']:>8,}  "
                  f"{ex.get('5min', 0):>6.1f}  {ex.get('10min', 0):>6.1f}  "
                  f"{ex15:>6.1f}  {ex.get('20min', 0):>6.1f}  {ex30:>6.1f}  "
                  f"{ratio:>6.1f}")

    # Group averages
    print(f"\n  Group averages:")
    for group_name, tickers in [("HIGH-OPTIONS", HIGH_OPTIONS),
                                 ("LOW-OPTIONS", LOW_OPTIONS)]:
        ratios = []
        ex15s = []
        ex30s = []
        for ticker in tickers:
            r = all_results.get(ticker, {})
            if r.get("too_few") or "excess" not in r:
                continue
            ex15 = r["excess"].get("15min", 0)
            ex30 = r["excess"].get("30min", 0)
            if ex30 > 0:
                ratios.append(ex15 / ex30)
                ex15s.append(ex15)
                ex30s.append(ex30)

        if ratios:
            print(f"    {group_name:>12s}: "
                  f"mean 15min={np.mean(ex15s):.1f}x  "
                  f"mean 30min={np.mean(ex30s):.1f}x  "
                  f"mean ratio={np.mean(ratios):.1f}x  "
                  f"(n={len(ratios)})")

    print(f"\n  Prediction: HIGH-OPTIONS ratio >> 1, LOW-OPTIONS ratio ~ 1")
    print(f"  If confirmed: 15min anomaly is options-hedging signature")
    print(f"  If refuted: 15min anomaly is AMD-specific or noise")


if __name__ == "__main__":
    main()
