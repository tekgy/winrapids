"""Cross-ticker inter-arrival time spectral comparison.

Tests whether the two-regime structure discovered in AAPL
(execution layer 2s-60s, institutional layer 1min-30min)
is universal across tickers with different liquidity profiles.

Ticker selection:
  AAPL  - mega-cap tech, ~530K RTH ticks (baseline)
  MSFT  - mega-cap tech, validation (similar profile?)
  AMD   - large-cap tech, heavily algo-traded
  KO    - large-cap consumer staple, different sector
  CHWY  - mid-cap, lower volume
  AAME  - micro-cap, very low volume (stress test)
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

TICKERS = ["AAPL", "MSFT", "AMD", "KO", "CHWY", "AAME"]

# Full cadence chain from the v3 experiment
ROUND_CHECK = [
    # Sub-second
    (10.0,       "0.1s"),
    (5.0,        "0.2s"),
    (2.0,        "0.5s"),
    # Seconds
    (1.0,        "1s"),
    (1/2,        "2s"),
    (1/3,        "3s"),
    (1/5,        "5s"),
    (1/7,        "7s"),
    (1/10,       "10s"),
    (1/15,       "15s"),
    (1/20,       "20s"),
    (1/30,       "30s"),
    (1/45,       "45s"),
    # Minutes
    (1/60,       "1min"),
    (1/120,      "2min"),
    (1/300,      "5min"),
    (1/600,      "10min"),
    (1/900,      "15min"),
    (1/1800,     "30min"),
]


def load_rth_timestamps(ticker: str) -> np.ndarray:
    """Load and filter timestamps to regular trading hours."""
    path = DATA_ROOT / ticker / "K01P01.TI00TO00.parquet"
    tbl = pq.read_table(str(path))
    timestamps_ns = tbl.column("K01P01.DI01DO03").to_numpy().astype(np.int64)
    timestamps_ns = np.sort(timestamps_ns)

    # Market hours filter: 09:30-16:00 ET = 13:30-20:00 UTC
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


def compute_excess_spectrum(timestamps_ns: np.ndarray) -> dict:
    """Compute spectral excess at round-number frequencies.

    Returns dict with ticker stats and excess values per cadence.
    """
    iat_ns = np.diff(timestamps_ns)
    iat_us = iat_ns / 1000.0

    n_total = len(iat_us)
    n_zero = (iat_ns == 0).sum()
    pct_zero = 100 * n_zero / n_total

    # Remove zero IATs
    iat_nonzero = iat_us[iat_us > 0]
    n_nonzero = len(iat_nonzero)

    if n_nonzero < 1000:
        return {"n_ticks": len(timestamps_ns), "n_nonzero": n_nonzero,
                "pct_zero": pct_zero, "too_few": True, "excess": {}}

    # FFT
    iat_centered = iat_nonzero - iat_nonzero.mean()
    spectrum = np.fft.rfft(iat_centered.astype(np.float32))
    psd = np.abs(spectrum) ** 2 / n_nonzero
    freqs = np.fft.rfftfreq(n_nonzero)

    mean_iat_s = iat_nonzero.mean() / 1e6
    mean_sample_rate = 1.0 / mean_iat_s
    freqs_hz = freqs * mean_sample_rate

    total_power = psd[1:].sum()
    n_freq_bins = len(psd[1:])

    # Compute excess at each round-number frequency
    excess = {}
    for freq_target, label in ROUND_CHECK:
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

    # Spectral slope
    fit_mask = (freqs_hz[1:] >= 0.01) & (freqs_hz[1:] <= 100) & (psd[1:] > 0)
    alpha = None
    r_sq = None
    if fit_mask.sum() > 10:
        log_f = np.log10(freqs_hz[1:][fit_mask])
        log_p = np.log10(psd[1:][fit_mask])
        coeffs = np.polyfit(log_f, log_p, 1)
        alpha = round(-coeffs[0], 3)
        fitted = np.polyval(coeffs, log_f)
        ss_res = ((log_p - fitted) ** 2).sum()
        ss_tot = ((log_p - log_p.mean()) ** 2).sum()
        r_sq = round(1 - ss_res / ss_tot, 4)

    # Peak count
    window = 1001
    half_w = window // 2
    log_psd = np.log10(psd[1:] + 1e-30)
    local_median = np.zeros_like(log_psd)
    # Vectorized sliding median approximation: use stride tricks
    # For speed, sample every 10th point for the median window
    for i in range(len(log_psd)):
        lo = max(0, i - half_w)
        hi = min(len(log_psd), i + half_w + 1)
        local_median[i] = np.median(log_psd[lo:hi])

    peak_mask = (log_psd - local_median) > 1.0
    peak_indices = np.where(peak_mask)[0]
    n_peaks = 0
    if len(peak_indices) > 0:
        clusters = []
        current = [peak_indices[0]]
        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i-1] < 50:
                current.append(peak_indices[i])
            else:
                clusters.append(current)
                current = [peak_indices[i]]
        clusters.append(current)
        n_peaks = len(clusters)

    return {
        "n_ticks": len(timestamps_ns),
        "n_nonzero": n_nonzero,
        "pct_zero": pct_zero,
        "too_few": False,
        "mean_iat_us": round(iat_nonzero.mean(), 1),
        "sample_rate_hz": round(mean_sample_rate, 1),
        "alpha": alpha,
        "r_sq": r_sq,
        "n_peaks": n_peaks,
        "psd_dynamic_range": round(psd[1:].max() / psd[1:].mean(), 0),
        "excess": excess,
    }


def main():
    print("=" * 80)
    print("CROSS-TICKER INTER-ARRIVAL TIME SPECTRAL COMPARISON")
    print("Testing universality of two-regime gradient structure")
    print("=" * 80)

    results = {}
    for ticker in TICKERS:
        print(f"\n{'=' * 80}")
        print(f"  {ticker}")
        print(f"{'=' * 80}")

        try:
            ts = load_rth_timestamps(ticker)
            print(f"  RTH ticks: {len(ts):,}")
            result = compute_excess_spectrum(ts)
            results[ticker] = result

            if result["too_few"]:
                print(f"  ** Too few non-zero IATs ({result['n_nonzero']:,}) for FFT")
                continue

            print(f"  Non-zero IATs: {result['n_nonzero']:,}"
                  f"  ({result['pct_zero']:.1f}% simultaneous)")
            print(f"  Mean IAT: {result['mean_iat_us']:.1f} us"
                  f"  ({result['sample_rate_hz']:.1f} Hz)")
            print(f"  Spectral slope: alpha={result['alpha']}"
                  f"  R^2={result['r_sq']}")
            print(f"  Peaks: {result['n_peaks']}"
                  f"  Dynamic range: {result['psd_dynamic_range']:.0f}x")

            print(f"\n  Excess at round-number periods:")
            for label, ex in result["excess"].items():
                bar = "*" * min(int(ex), 60)
                flag = " ***" if ex > 3.0 else ""
                print(f"    {label:>5s}: {ex:>7.1f}x  {bar}{flag}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[ticker] = {"error": str(e)}

    # ── Comparison table ──────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("COMPARISON TABLE: Excess power at key cadences")
    print(f"{'=' * 80}")

    # Header
    key_cadences = ["0.5s", "1s", "2s", "5s", "10s", "30s",
                    "1min", "5min", "10min", "15min", "30min"]

    header = f"{'Ticker':>6s}  {'Ticks':>8s}  {'MeanIAT':>8s}"
    for c in key_cadences:
        header += f"  {c:>5s}"
    print(f"\n{header}")
    print("-" * len(header))

    for ticker in TICKERS:
        r = results.get(ticker, {})
        if "error" in r or r.get("too_few"):
            n = r.get("n_ticks", 0)
            print(f"{ticker:>6s}  {n:>8,}  {'--':>8s}  (insufficient data)")
            continue

        row = f"{ticker:>6s}  {r['n_ticks']:>8,}  {r['mean_iat_us']:>7.0f}us"
        for c in key_cadences:
            ex = r["excess"].get(c)
            if ex is not None:
                row += f"  {ex:>5.1f}"
            else:
                row += f"  {'--':>5s}"
        print(row)

    # ── Gradient analysis ─────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("GRADIENT ANALYSIS: Is the monotonic increase universal?")
    print(f"{'=' * 80}")

    gradient_cadences = ["1s", "2s", "5s", "10s", "30s", "1min", "5min", "10min", "30min"]

    for ticker in TICKERS:
        r = results.get(ticker, {})
        if "error" in r or r.get("too_few"):
            continue

        excess_vals = []
        for c in gradient_cadences:
            ex = r["excess"].get(c)
            if ex is not None:
                excess_vals.append((c, ex))

        if len(excess_vals) < 3:
            print(f"\n  {ticker}: too few cadences for gradient analysis")
            continue

        # Check monotonicity
        vals = [v for _, v in excess_vals]
        monotonic_breaks = 0
        for i in range(1, len(vals)):
            if vals[i] < vals[i-1]:
                monotonic_breaks += 1

        total_pairs = len(vals) - 1
        monotonic_pct = 100 * (1 - monotonic_breaks / total_pairs)

        # Execution regime (1s-60s) vs institutional regime (1min-30min)
        exec_vals = [v for c, v in excess_vals if c in ["1s", "2s", "5s", "10s", "30s"]]
        inst_vals = [v for c, v in excess_vals if c in ["5min", "10min", "15min", "30min"]]

        exec_mean = np.mean(exec_vals) if exec_vals else 0
        inst_mean = np.mean(inst_vals) if inst_vals else 0
        regime_ratio = inst_mean / exec_mean if exec_mean > 0 else 0

        print(f"\n  {ticker}:")
        print(f"    Gradient monotonicity: {monotonic_pct:.0f}%"
              f" ({monotonic_breaks} breaks in {total_pairs} pairs)")
        print(f"    Execution regime mean (1s-30s): {exec_mean:.1f}x")
        print(f"    Institutional regime mean (5-30min): {inst_mean:.1f}x")
        print(f"    Regime ratio (institutional/execution): {regime_ratio:.1f}x")

    print(f"\n\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")
    print("\n  Is the two-regime structure universal?")
    print("  Check: (1) All tickers show monotonic gradient?")
    print("         (2) All tickers show execution/institutional separation?")
    print("         (3) Gradient steepness varies with liquidity?")


if __name__ == "__main__":
    main()
