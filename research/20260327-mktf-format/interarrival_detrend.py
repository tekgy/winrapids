"""Detrending test: Is the institutional regime real or a daily U-shape artifact?

Scout's concern: the intraday IAT distribution is non-stationary (bursty at
open, slow midday, bursty at close). The FFT mixes these regimes. Longer-period
signals might be riding the broad daily trend rather than reflecting genuine
periodic structure.

Test: compute the spectrum of raw IATs vs detrended IATs (subtract the mean
intraday IAT-vs-time curve). True periodic signals (30min VWAP rhythm, 15min
hedging) survive detrending. Trend-riding artifacts don't.

Also checks for 6.5h session harmonics (hierarchy artifacts).
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

FOCUS_CADENCES = [
    (1/60,   "1min"),
    (1/120,  "2min"),
    (1/300,  "5min"),
    (1/600,  "10min"),
    (1/900,  "15min"),
    (1/1200, "20min"),
    (1/1800, "30min"),
    (1/3600, "1h"),
    (1/5400, "1.5h"),
    (1/7200, "2h"),
    (1/10800, "3h"),
]

# Also check seconds regime for baseline comparison
SECOND_CADENCES = [
    (2.0,    "0.5s"),
    (1.0,    "1s"),
    (1/2,    "2s"),
    (1/5,    "5s"),
    (1/10,   "10s"),
    (1/30,   "30s"),
]

ALL_CADENCES = SECOND_CADENCES + FOCUS_CADENCES


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


def compute_excess(iat_us_nonzero: np.ndarray, cadences: list,
                   label: str = "") -> dict:
    """Compute spectral excess at round-number frequencies."""
    n = len(iat_us_nonzero)
    iat_centered = iat_us_nonzero - iat_us_nonzero.mean()
    spectrum = np.fft.rfft(iat_centered.astype(np.float32))
    psd = np.abs(spectrum) ** 2 / n
    freqs = np.fft.rfftfreq(n)

    mean_iat_s = iat_us_nonzero.mean() / 1e6
    mean_sample_rate = 1.0 / mean_iat_s
    freqs_hz = freqs * mean_sample_rate

    total_power = psd[1:].sum()
    n_freq_bins = len(psd[1:])

    excess = {}
    for freq_target, clabel in cadences:
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
            excess[clabel] = round(ex, 2)

    return excess


def detrend_iats(iat_us: np.ndarray, timestamps_ns: np.ndarray,
                 n_bins: int = 130) -> np.ndarray:
    """Remove the intraday U-shape from inter-arrival times.

    Computes the mean IAT in each time-of-day bin, then subtracts
    the local mean from each IAT. Uses the midpoint timestamp of
    each IAT interval.

    Args:
        iat_us: inter-arrival times in microseconds (non-zero only)
        timestamps_ns: timestamps corresponding to iat_us[i] = t[i+1] - t[i],
                       so we use t[i] as the timestamp for iat_us[i]
        n_bins: number of time-of-day bins (130 = ~3min bins over 6.5h)

    Returns:
        Detrended IATs (IAT - local_mean + global_mean, to preserve scale)
    """
    # Normalize timestamps to [0, 1] within the session
    t_min = timestamps_ns.min()
    t_max = timestamps_ns.max()
    t_norm = (timestamps_ns - t_min) / (t_max - t_min + 1)

    # Assign each IAT to a time-of-day bin
    bin_idx = (t_norm * n_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Compute mean IAT per bin
    bin_sums = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(len(iat_us)):
        b = bin_idx[i]
        bin_sums[b] += iat_us[i]
        bin_counts[b] += 1

    bin_means = np.where(bin_counts > 0,
                         bin_sums / bin_counts,
                         iat_us.mean())

    # Subtract local mean, add global mean (preserve scale)
    global_mean = iat_us.mean()
    local_means = bin_means[bin_idx]
    detrended = iat_us - local_means + global_mean

    return detrended, bin_means, bin_counts


def main():
    print("=" * 70)
    print("DETRENDING TEST: Daily U-Shape vs True Periodic Structure")
    print("=" * 70)

    tickers = ["AAPL", "NVDA", "TSLA"]

    for ticker in tickers:
        print(f"\n{'=' * 70}")
        print(f"  {ticker}")
        print(f"{'=' * 70}")

        ts = load_rth_timestamps(ticker)
        n_ticks = len(ts)
        print(f"  RTH ticks: {n_ticks:,}")

        # Compute IATs
        iat_ns = np.diff(ts)
        iat_us = iat_ns / 1000.0

        # Filter to non-zero
        nonzero_mask = iat_us > 0
        iat_nz = iat_us[nonzero_mask]
        ts_nz = ts[:-1][nonzero_mask]  # timestamps for each IAT
        n_nz = len(iat_nz)
        print(f"  Non-zero IATs: {n_nz:,}")

        # ── The U-shape ──────────────────────────────────────
        detrended, bin_means, bin_counts = detrend_iats(iat_nz, ts_nz)

        # Show the U-shape
        print(f"\n  Intraday IAT U-shape (130 bins, ~3min each):")
        print(f"    Open  (first 10 bins): mean IAT = {bin_means[:10].mean():.0f} us")
        print(f"    Mid   (bins 40-90):    mean IAT = {bin_means[40:90].mean():.0f} us")
        print(f"    Close (last 10 bins):  mean IAT = {bin_means[-10:].mean():.0f} us")
        print(f"    Overall mean:          {iat_nz.mean():.0f} us")
        print(f"    U-shape ratio (mid/open): {bin_means[40:90].mean() / bin_means[:10].mean():.2f}x")

        # ── Raw spectrum ─────────────────────────────────────
        raw_excess = compute_excess(iat_nz, ALL_CADENCES, "raw")

        # ── Detrended spectrum ───────────────────────────────
        det_excess = compute_excess(detrended, ALL_CADENCES, "detrended")

        # ── Comparison ───────────────────────────────────────
        print(f"\n  {'Period':>6s}  {'Raw':>7s}  {'Detrend':>7s}  {'Change':>8s}  {'Verdict':>12s}")
        print(f"  {'-' * 50}")

        for _, label in ALL_CADENCES:
            raw_val = raw_excess.get(label)
            det_val = det_excess.get(label)
            if raw_val is None or det_val is None:
                continue

            if raw_val > 0:
                change_pct = 100 * (det_val - raw_val) / raw_val
            else:
                change_pct = 0

            # Verdict
            if abs(change_pct) < 15:
                verdict = "REAL"
            elif change_pct < -50:
                verdict = "ARTIFACT"
            elif change_pct < -15:
                verdict = "PARTIAL"
            else:
                verdict = "ENHANCED"

            print(f"  {label:>6s}  {raw_val:>7.1f}  {det_val:>7.1f}  "
                  f"{change_pct:>+7.0f}%  {verdict:>12s}")

        # ── Summary stats ────────────────────────────────────
        # Execution regime (1s-30s)
        exec_labels = ["1s", "2s", "5s", "10s", "30s"]
        raw_exec = [raw_excess.get(l, 0) for l in exec_labels if l in raw_excess]
        det_exec = [det_excess.get(l, 0) for l in exec_labels if l in det_excess]

        # Institutional regime (5min-30min)
        inst_labels = ["5min", "10min", "15min", "20min", "30min"]
        raw_inst = [raw_excess.get(l, 0) for l in inst_labels if l in raw_excess]
        det_inst = [det_excess.get(l, 0) for l in inst_labels if l in det_excess]

        # Extended regime (1h-3h)
        ext_labels = ["1h", "1.5h", "2h", "3h"]
        raw_ext = [raw_excess.get(l, 0) for l in ext_labels if l in raw_excess]
        det_ext = [det_excess.get(l, 0) for l in ext_labels if l in det_excess]

        print(f"\n  Regime summary (mean excess):")
        if raw_exec:
            print(f"    Execution (1s-30s):     raw={np.mean(raw_exec):.1f}x  "
                  f"detrend={np.mean(det_exec):.1f}x  "
                  f"change={100*(np.mean(det_exec)/np.mean(raw_exec)-1):+.0f}%")
        if raw_inst:
            print(f"    Institutional (5-30min): raw={np.mean(raw_inst):.1f}x  "
                  f"detrend={np.mean(det_inst):.1f}x  "
                  f"change={100*(np.mean(det_inst)/np.mean(raw_inst)-1):+.0f}%")
        if raw_ext:
            print(f"    Extended (1-3h):         raw={np.mean(raw_ext):.1f}x  "
                  f"detrend={np.mean(det_ext):.1f}x  "
                  f"change={100*(np.mean(det_ext)/np.mean(raw_ext)-1):+.0f}%")

    print(f"\n\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    print("\n  Signals that survive detrending = true periodic structure")
    print("  Signals that collapse = daily U-shape artifact")
    print("  Signals that grow = U-shape was suppressing them")


if __name__ == "__main__":
    main()
