"""Inter-arrival time spectral analysis — K-F01b feasibility experiment.

Tests whether the power spectrum of trade inter-arrival times shows
structure (algorithmic clocks, round-number concentration, 1/f scaling)
or is just Poisson noise.

If the spectrum shows peaks or structure, K-F01b (DFT of inter-arrival
times) is worth building as a gaming/regime detection leaf.

Computes:
  1. Inter-arrival time distribution (basic stats)
  2. Power spectral density via FFT
  3. Poisson noise floor (null hypothesis)
  4. Peak detection (algorithmic clock signatures)
  5. Round-number frequency concentration (gaming signal)
  6. Spectral slope (1/f exponent for regime characterization)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── Load real data ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from mktf_v3 import AAPL_PATH, COL_MAP

import pyarrow.parquet as pq


def main():
    print("=" * 70)
    print("INTER-ARRIVAL TIME SPECTRAL ANALYSIS")
    print("K-F01b Feasibility Experiment")
    print("=" * 70)

    # Load timestamps
    tbl = pq.read_table(str(AAPL_PATH))
    timestamps_ns = tbl.column("K01P01.DI01DO03").to_numpy().astype(np.int64)
    n_ticks = len(timestamps_ns)

    # Sort timestamps (should already be sorted, but be safe)
    timestamps_ns = np.sort(timestamps_ns)

    print(f"\nSource: AAPL {n_ticks:,} ticks (all hours)")
    time_range_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    print(f"Time range: {time_range_s:.0f}s ({time_range_s / 3600:.1f}h)")

    # ── Phase 0: Filter to market hours (09:30-16:00 ET) ───────────
    # Timestamps are nanoseconds since epoch. Find the day, compute
    # 09:30 and 16:00 boundaries.
    import datetime
    first_ts_s = timestamps_ns[0] / 1e9
    day_start = datetime.datetime.fromtimestamp(first_ts_s, tz=datetime.timezone.utc)
    # Assume Eastern Time (UTC-4 for EDT, UTC-5 for EST)
    # 2025-09-02 is in EDT (UTC-4)
    # Market open: 09:30 ET = 13:30 UTC
    # Market close: 16:00 ET = 20:00 UTC
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
    n_rth = rth_mask.sum()
    n_pre_post = n_ticks - n_rth
    timestamps_ns = timestamps_ns[rth_mask]
    n_ticks = len(timestamps_ns)

    time_range_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    print(f"\nFiltered to market hours (09:30-16:00 ET):")
    print(f"  RTH ticks: {n_ticks:,} ({100*n_ticks/(n_ticks+n_pre_post):.1f}%)")
    print(f"  Pre/post-market removed: {n_pre_post:,}")
    print(f"  Time range: {time_range_s:.0f}s ({time_range_s / 3600:.1f}h)")

    # ── Phase 1: Inter-arrival times ────────────────────────────────
    print(f"\n{'-' * 70}")
    print("Phase 1: Inter-arrival time distribution (market hours only)")
    print(f"{'-' * 70}")

    # Compute inter-arrival times in microseconds
    iat_ns = np.diff(timestamps_ns)
    iat_us = iat_ns / 1000.0  # microseconds
    iat_ms = iat_ns / 1_000_000.0  # milliseconds
    iat_s = iat_ns / 1_000_000_000.0  # seconds

    n_iat = len(iat_us)
    print(f"\n  N inter-arrival times: {n_iat:,}")
    print(f"  Mean: {iat_us.mean():.1f} us ({iat_ms.mean():.3f} ms)")
    print(f"  Median: {np.median(iat_us):.1f} us")
    print(f"  Min: {iat_us.min():.1f} us")
    print(f"  Max: {iat_us.max():.1f} us ({iat_s.max():.1f} s)")
    print(f"  Std: {iat_us.std():.1f} us")

    # Distribution shape
    pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(iat_us, pctiles)
    print(f"\n  Percentiles (microseconds):")
    for p, v in zip(pctiles, pct_vals):
        print(f"    {p:>3d}th: {v:>12.1f} us ({v/1000:.3f} ms)")

    # Zero inter-arrival times (simultaneous trades, different timestamps?)
    n_zero = (iat_ns == 0).sum()
    n_sub_us = (iat_ns < 1000).sum()
    print(f"\n  Zero IAT (simultaneous): {n_zero:,} ({100*n_zero/n_iat:.1f}%)")
    print(f"  Sub-microsecond IAT: {n_sub_us:,} ({100*n_sub_us/n_iat:.1f}%)")

    # ── Phase 2: Power Spectral Density ─────────────────────────────
    print(f"\n{'-' * 70}")
    print("Phase 2: Power spectral density (FFT of inter-arrival times)")
    print(f"{'-' * 70}")

    # Remove zero IATs for spectral analysis (they create artifacts)
    iat_nonzero = iat_us[iat_us > 0]
    n_nonzero = len(iat_nonzero)
    print(f"\n  Non-zero IATs for FFT: {n_nonzero:,}")

    # Center the signal (remove mean to eliminate DC component)
    iat_centered = iat_nonzero - iat_nonzero.mean()

    # FFT
    spectrum = np.fft.rfft(iat_centered.astype(np.float32))
    psd = np.abs(spectrum) ** 2 / n_nonzero  # power spectral density
    freqs = np.fft.rfftfreq(n_nonzero)  # in cycles per sample

    # Convert to physical frequency
    # Mean IAT gives the mean sampling rate
    mean_iat_s = iat_nonzero.mean() / 1e6  # seconds
    mean_sample_rate = 1.0 / mean_iat_s  # Hz
    freqs_hz = freqs * mean_sample_rate

    print(f"  Mean sample rate: {mean_sample_rate:.1f} Hz")
    print(f"  Nyquist frequency: {mean_sample_rate/2:.1f} Hz")
    print(f"  Frequency resolution: {freqs_hz[1]:.6f} Hz")
    print(f"  Number of frequency bins: {len(freqs_hz):,}")

    # ── Phase 3: Poisson Noise Floor ────────────────────────────────
    print(f"\n{'-' * 70}")
    print("Phase 3: Poisson noise floor (null hypothesis)")
    print(f"{'-' * 70}")

    # For a Poisson process, inter-arrival times are exponentially distributed
    # The PSD of a Poisson process is flat (white noise) with power = 2*lambda
    # where lambda = mean arrival rate
    mean_rate = 1.0 / mean_iat_s  # arrivals per second
    poisson_noise_level = 2.0 * iat_nonzero.var()  # variance of the process
    # Normalize to match our PSD computation
    poisson_psd = poisson_noise_level / n_nonzero

    print(f"  Mean arrival rate: {mean_rate:.1f} Hz")
    print(f"  Poisson noise level (per-bin PSD): {poisson_psd:.2e}")
    print(f"  Mean PSD: {psd[1:].mean():.2e}")  # skip DC
    print(f"  Max PSD: {psd[1:].max():.2e}")
    print(f"  PSD dynamic range: {psd[1:].max() / psd[1:].mean():.1f}x")

    # ── Phase 4: Peak Detection ─────────────────────────────────────
    print(f"\n{'-' * 70}")
    print("Phase 4: Peak detection (algorithmic clock signatures)")
    print(f"{'-' * 70}")

    # Use logarithmic frequency bins for analysis (equal spacing on log scale)
    # Focus on 0.01 Hz to Nyquist/2
    log_freq_min = -2  # 0.01 Hz
    log_freq_max = np.log10(mean_sample_rate / 2)

    # Find peaks: frequency bins where PSD exceeds 10x the local median
    # Use a sliding window median for local noise floor
    window = 1001  # odd number for symmetric window
    half_w = window // 2

    # Smooth PSD for peak detection (log scale)
    log_psd = np.log10(psd[1:] + 1e-30)  # skip DC, avoid log(0)
    local_median = np.zeros_like(log_psd)
    for i in range(len(log_psd)):
        lo = max(0, i - half_w)
        hi = min(len(log_psd), i + half_w + 1)
        local_median[i] = np.median(log_psd[lo:hi])

    # Peaks: exceed local median by > 1 decade (10x)
    peak_mask = (log_psd - local_median) > 1.0
    peak_indices = np.where(peak_mask)[0]

    if len(peak_indices) > 0:
        # Cluster nearby peaks
        clusters = []
        current_cluster = [peak_indices[0]]
        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i-1] < 50:
                current_cluster.append(peak_indices[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [peak_indices[i]]
        clusters.append(current_cluster)

        print(f"\n  Found {len(clusters)} spectral peaks above 10x local noise floor:")
        for ci, cluster in enumerate(clusters[:20]):  # show top 20
            peak_idx = cluster[np.argmax(log_psd[cluster])]
            peak_freq = freqs_hz[peak_idx + 1]
            peak_power = psd[peak_idx + 1]
            peak_period = 1.0 / peak_freq if peak_freq > 0 else float('inf')
            excess_db = 10 * (log_psd[peak_idx] - local_median[peak_idx])

            # Check if near a round-number frequency
            round_freqs = [0.0333, 0.0667, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
            round_periods = [30, 15, 10, 5, 2, 1, 0.5, 0.2, 0.1]
            near_round = ""
            for rf, rp in zip(round_freqs, round_periods):
                if abs(peak_freq - rf) / rf < 0.05:  # within 5%
                    near_round = f"  << NEAR {rp}s period"
                    break

            print(f"    Peak {ci+1}: f={peak_freq:.4f} Hz"
                  f"  T={peak_period:.3f}s"
                  f"  power={peak_power:.2e}"
                  f"  excess={excess_db:.1f} dB{near_round}")
    else:
        print(f"\n  No peaks found above 10x local noise floor.")
        print(f"  The inter-arrival process appears spectrally flat (Poisson-like).")

    # ── Phase 5: Round-Number Frequency Concentration ───────────────
    print(f"\n{'-' * 70}")
    print("Phase 5: Round-number frequency concentration")
    print(f"{'-' * 70}")

    # Measure total PSD power within ±5% of key frequencies
    # Full cadence chain: sub-second through multi-hour
    # Sorted by period (short to long) = frequency (high to low)
    round_check = [
        # Sub-second
        (10.0,       "0.1s"),
        (5.0,        "0.2s"),
        (2.0,        "0.5s"),
        # Seconds (1-2-5 grid + extras)
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
        (1/90,       "1.5min"),
        (1/120,      "2min"),
        (1/180,      "3min"),
        (1/300,      "5min"),
        (1/420,      "7min"),
        (1/600,      "10min"),
        (1/900,      "15min"),
        (1/1800,     "30min"),
        (1/3600,     "60min"),
    ]

    total_power = psd[1:].sum()  # total PSD power (excluding DC)
    print(f"\n  Total spectral power: {total_power:.2e}")
    print(f"\n  Power near round-number frequencies:")

    for freq_target, label in round_check:
        if freq_target >= mean_sample_rate / 2:
            continue  # skip frequencies above Nyquist
        # Find bins within ±5% of target
        f_lo = freq_target * 0.95
        f_hi = freq_target * 1.05
        mask = (freqs_hz[1:] >= f_lo) & (freqs_hz[1:] <= f_hi)
        if mask.any():
            band_power = psd[1:][mask].sum()
            n_bins_in_band = mask.sum()
            # Expected power if spectrum were flat
            expected = total_power * n_bins_in_band / len(psd[1:])
            excess = band_power / expected if expected > 0 else 0

            flag = " ***" if excess > 3.0 else ""
            print(f"    {label:>5s} ({freq_target:.4f} Hz):"
                  f"  power={band_power:.2e}"
                  f"  expected={expected:.2e}"
                  f"  excess={excess:.1f}x{flag}")

    # ── Phase 6: Spectral Slope (1/f exponent) ──────────────────────
    print(f"\n{'-' * 70}")
    print("Phase 6: Spectral slope (1/f exponent)")
    print(f"{'-' * 70}")

    # Fit log(PSD) vs log(freq) to measure spectral exponent
    # P(f) ~ 1/f^alpha  => log P = -alpha * log f + const
    # Use frequency range 0.01 Hz to 100 Hz (avoiding DC and Nyquist edges)
    fit_mask = (freqs_hz[1:] >= 0.01) & (freqs_hz[1:] <= 100) & (psd[1:] > 0)
    if fit_mask.sum() > 10:
        log_f = np.log10(freqs_hz[1:][fit_mask])
        log_p = np.log10(psd[1:][fit_mask])

        # Robust fit using polyfit
        coeffs = np.polyfit(log_f, log_p, 1)
        alpha = -coeffs[0]
        intercept = coeffs[1]

        # R-squared
        fitted = np.polyval(coeffs, log_f)
        ss_res = ((log_p - fitted) ** 2).sum()
        ss_tot = ((log_p - log_p.mean()) ** 2).sum()
        r_squared = 1 - ss_res / ss_tot

        print(f"\n  Spectral exponent alpha = {alpha:.3f} (P(f) ~ 1/f^alpha)")
        print(f"  R-squared of fit: {r_squared:.4f}")

        if alpha < 0.3:
            print(f"  Interpretation: Near-white noise (alpha < 0.3)")
            print(f"    -> Poisson-like process, minimal temporal correlation")
        elif alpha < 0.8:
            print(f"  Interpretation: Pink noise (0.3 < alpha < 0.8)")
            print(f"    -> Some clustering, moderate correlations")
        elif alpha < 1.5:
            print(f"  Interpretation: 1/f noise (0.8 < alpha < 1.5)")
            print(f"    -> Strong clustering, long-range correlations in arrival times")
            print(f"    -> This is the self-organized criticality signature")
        elif alpha < 2.5:
            print(f"  Interpretation: Brown noise (1.5 < alpha < 2.5)")
            print(f"    -> Very strong clustering, near-random walk in arrival rate")
        else:
            print(f"  Interpretation: Very red noise (alpha > 2.5)")
            print(f"    -> Extreme clustering, possible non-stationarity")
    else:
        print(f"\n  Insufficient frequency bins for slope fit")

    # ── Phase 7: Summary ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY: K-F01b Feasibility")
    print(f"{'=' * 70}")

    has_peaks = len(peak_indices) > 0 if 'peak_indices' in dir() else False
    has_slope = alpha > 0.3 if 'alpha' in dir() else False

    if has_peaks or has_slope:
        print(f"\n  RESULT: Inter-arrival time spectrum shows STRUCTURE.")
        print(f"  K-F01b (DFT of inter-arrival times) is worth building.")
        if has_peaks:
            print(f"  - {len(clusters)} spectral peaks detected (algorithmic clocks?)")
        if has_slope:
            print(f"  - Spectral slope alpha={alpha:.2f} (non-Poisson)")
        print(f"\n  The trading rhythm has fingerprints. The spectrum doesn't lie.")
    else:
        print(f"\n  RESULT: Inter-arrival time spectrum is flat (Poisson noise).")
        print(f"  K-F01b may not carry useful signal for this ticker/date.")
        print(f"  Test with other tickers before abandoning the idea.")


if __name__ == "__main__":
    main()
