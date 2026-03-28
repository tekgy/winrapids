"""Phase analysis at 5-minute frequency — shared clock or independent algorithms?

The detrended FFT complex coefficients contain phase information.
Extract the phase at the 5min frequency for all 10 tickers.

If phases cluster: shared market-wide clock (VWAP, index rebalancing).
If phases scatter uniformly on [0, 2pi]: independent per-ticker algorithms.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

TICKERS = ["AAPL", "MSFT", "AMD", "NVDA", "TSLA", "META", "KO", "BRK.B", "JNJ", "CHWY"]

# Target frequencies to extract phase at
TARGET_PERIODS = [
    (5,    "5s"),
    (10,   "10s"),
    (30,   "30s"),
    (60,   "1min"),
    (120,  "2min"),
    (300,  "5min"),
    (600,  "10min"),
]


def load_rth_timestamps(ticker):
    path = DATA_ROOT / ticker / "K01P01.TI00TO00.parquet"
    tbl = pq.read_table(str(path))
    ts = tbl.column("K01P01.DI01DO03").to_numpy().astype(np.int64)
    ts = np.sort(ts)
    first_s = ts[0] / 1e9
    d = datetime.datetime.fromtimestamp(first_s, tz=datetime.timezone.utc).date()
    o = int(datetime.datetime(d.year, d.month, d.day, 13, 30, 0,
            tzinfo=datetime.timezone.utc).timestamp() * 1e9)
    c = int(datetime.datetime(d.year, d.month, d.day, 20, 0, 0,
            tzinfo=datetime.timezone.utc).timestamp() * 1e9)
    return ts[(ts >= o) & (ts <= c)]


def detrend(iat_us, ts_ns, n_bins=130):
    t_min, t_max = ts_ns.min(), ts_ns.max()
    t_norm = (ts_ns - t_min) / (t_max - t_min + 1)
    bidx = np.clip((t_norm * n_bins).astype(np.int32), 0, n_bins - 1)
    bsums = np.zeros(n_bins, dtype=np.float64)
    bcnts = np.zeros(n_bins, dtype=np.int64)
    for i in range(len(iat_us)):
        b = bidx[i]
        bsums[b] += iat_us[i]
        bcnts[b] += 1
    bmeans = np.where(bcnts > 0, bsums / bcnts, iat_us.mean())
    return iat_us - bmeans[bidx] + iat_us.mean()


def extract_phase_and_power(iat_us_nz, target_periods_s):
    """Extract phase and excess power at target frequencies from detrended IATs."""
    n = len(iat_us_nz)
    centered = iat_us_nz - iat_us_nz.mean()
    spectrum = np.fft.rfft(centered.astype(np.float32))
    psd = np.abs(spectrum) ** 2 / n
    freqs = np.fft.rfftfreq(n)

    mean_iat_s = iat_us_nz.mean() / 1e6
    mean_sample_rate = 1.0 / mean_iat_s
    freqs_hz = freqs * mean_sample_rate

    total_power = psd[1:].sum()
    n_freq_bins = len(psd[1:])

    results = {}
    for period_s, label in target_periods_s:
        freq_target = 1.0 / period_s
        if freq_target >= mean_sample_rate / 2:
            continue

        # Find the bin closest to the target frequency
        freq_diffs = np.abs(freqs_hz - freq_target)
        best_idx = np.argmin(freq_diffs)

        if best_idx == 0:
            continue

        # Phase from complex coefficient
        phase = np.angle(spectrum[best_idx])  # radians, [-pi, pi]
        power = psd[best_idx]

        # Also compute excess in +-5% band for context
        f_lo = freq_target * 0.95
        f_hi = freq_target * 1.05
        mask = (freqs_hz[1:] >= f_lo) & (freqs_hz[1:] <= f_hi)
        if mask.any():
            band_power = psd[1:][mask].sum()
            n_bins_band = mask.sum()
            expected = total_power * n_bins_band / n_freq_bins
            excess = band_power / expected if expected > 0 else 0
        else:
            excess = 0

        results[label] = {
            "phase_rad": phase,
            "phase_deg": np.degrees(phase),
            "excess": round(excess, 2),
            "freq_hz": freqs_hz[best_idx],
            "freq_err_pct": round(100 * abs(freqs_hz[best_idx] - freq_target) / freq_target, 2),
        }

    return results


def main():
    print("=" * 70)
    print("PHASE ANALYSIS: Shared Clock or Independent Algorithms?")
    print("=" * 70)

    all_phases = {}

    for ticker in TICKERS:
        ts = load_rth_timestamps(ticker)
        if len(ts) < 2000:
            print(f"  {ticker}: too few ticks")
            continue

        iat_ns = np.diff(ts)
        iat_us = iat_ns / 1000.0
        nz = iat_us > 0
        iat_nz = iat_us[nz]
        ts_nz = ts[:-1][nz]

        if len(iat_nz) < 1000:
            continue

        detrended = detrend(iat_nz, ts_nz)
        phases = extract_phase_and_power(detrended, TARGET_PERIODS)
        all_phases[ticker] = phases
        print(f"  {ticker}: {len(ts):,} ticks -> done")

    # ── Phase at 5min across tickers ──────────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE AT 5-MINUTE FREQUENCY (detrended)")
    print(f"{'=' * 70}")

    print(f"\n{'Ticker':>8s}  {'Phase':>8s}  {'Phase':>8s}  {'Excess':>7s}  {'FreqErr':>8s}")
    print(f"{'':>8s}  {'(deg)':>8s}  {'(rad)':>8s}  {'(x)':>7s}  {'(%)':>8s}")
    print("-" * 48)

    phases_5min = []
    for ticker in TICKERS:
        if ticker not in all_phases or "5min" not in all_phases[ticker]:
            continue
        p = all_phases[ticker]["5min"]
        print(f"{ticker:>8s}  {p['phase_deg']:>+8.1f}  {p['phase_rad']:>+8.3f}  "
              f"{p['excess']:>7.1f}  {p['freq_err_pct']:>7.2f}%")
        phases_5min.append(p['phase_rad'])

    if len(phases_5min) >= 3:
        phases_arr = np.array(phases_5min)

        # Circular statistics
        # Mean resultant length R: R=1 means perfect clustering, R=0 means uniform
        cos_sum = np.cos(phases_arr).sum()
        sin_sum = np.sin(phases_arr).sum()
        R = np.sqrt(cos_sum**2 + sin_sum**2) / len(phases_arr)
        mean_phase = np.arctan2(sin_sum, cos_sum)

        print(f"\n  Circular statistics (n={len(phases_arr)}):")
        print(f"    Mean resultant length R = {R:.3f}")
        print(f"    (R=1: perfect clustering, R=0: uniform scatter)")
        print(f"    Mean phase = {np.degrees(mean_phase):+.1f} deg")

        # Rayleigh test for uniformity
        # p ≈ exp(-n*R^2) for large n
        n = len(phases_arr)
        rayleigh_z = n * R**2
        rayleigh_p = np.exp(-rayleigh_z)  # approximate
        print(f"    Rayleigh test: z={rayleigh_z:.2f}, p={rayleigh_p:.4f}")
        if rayleigh_p < 0.05:
            print(f"    -> REJECT uniformity (p < 0.05): phases are CLUSTERED")
            print(f"    -> SHARED CLOCK hypothesis supported")
        else:
            print(f"    -> CANNOT reject uniformity (p >= 0.05): phases are SCATTERED")
            print(f"    -> INDEPENDENT ALGORITHMS hypothesis supported")

    # ── Phase at all target periods ───────────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE CLUSTERING (R) AT ALL TARGET PERIODS")
    print(f"{'=' * 70}")

    print(f"\n{'Period':>8s}  {'R':>6s}  {'MeanPh':>8s}  {'Ray_p':>8s}  {'Verdict':>12s}  {'MeanExc':>8s}")
    print("-" * 60)

    for period_s, label in TARGET_PERIODS:
        phases_at_period = []
        excesses = []
        for ticker in TICKERS:
            if ticker not in all_phases or label not in all_phases[ticker]:
                continue
            phases_at_period.append(all_phases[ticker][label]["phase_rad"])
            excesses.append(all_phases[ticker][label]["excess"])

        if len(phases_at_period) < 3:
            continue

        pa = np.array(phases_at_period)
        cs = np.cos(pa).sum()
        ss = np.sin(pa).sum()
        R = np.sqrt(cs**2 + ss**2) / len(pa)
        mp = np.degrees(np.arctan2(ss, cs))
        n = len(pa)
        rz = n * R**2
        rp = np.exp(-rz)
        mean_exc = np.mean(excesses)

        verdict = "CLUSTERED" if rp < 0.05 else "SCATTERED"
        print(f"{label:>8s}  {R:>6.3f}  {mp:>+8.1f}  {rp:>8.4f}  {verdict:>12s}  {mean_exc:>8.1f}")

    # ── Interpretation ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    print("""
  CLUSTERED phases at a given period -> shared clock / market-wide trigger
    -> K04 on K-SS01 leaves would show correlation everywhere (shared artifact)
    -> The 5min cadence measures a common phenomenon, not ticker-specific info

  SCATTERED phases at a given period -> independent algorithms
    -> K04 on K-SS01 leaves would show sector/strategy clusters
    -> The 5min cadence captures genuine ecosystem diversity

  Mixed (clustered at some periods, scattered at others):
    -> Different phenomena dominate at different timescales
    -> Short periods: synchronized exchange mechanics
    -> Longer periods: independent strategy execution
""")


if __name__ == "__main__":
    main()
