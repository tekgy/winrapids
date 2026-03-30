"""Phase stability test: split-half and multi-segment consistency.

The block bootstrap showed +/-170 degree CIs — but block resampling
creates discontinuities that inject phase noise. This may be overly
destructive for phase estimation.

Better test: estimate phase in independent time segments.
If the phase is real, it should be consistent across segments.

Split the trading day into segments, estimate phase in each,
compute within-ticker phase consistency.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

TICKERS = ["AAPL", "MSFT", "AMD", "NVDA", "TSLA", "META", "KO", "BRK.B", "JNJ", "CHWY"]

TARGET_PERIOD_S = 300  # 5 minutes
N_SEGMENTS = 6  # 6.5 hours / 6 = ~65 minutes each (13 five-min cycles per segment)


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
    return ts[(ts >= o) & (ts <= c)], o, c


def detrend(iat_us, ts_ns, n_bins=130):
    t_min, t_max = ts_ns.min(), ts_ns.max()
    if t_max <= t_min:
        return iat_us
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


def extract_phase_at_5min(iat_us_nz):
    """Extract phase at 5min from detrended IATs. Returns (phase, power, excess)."""
    n = len(iat_us_nz)
    if n < 500:
        return None, None, None
    centered = iat_us_nz - iat_us_nz.mean()
    spectrum = np.fft.rfft(centered.astype(np.float32))
    psd = np.abs(spectrum) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    mean_iat_s = iat_us_nz.mean() / 1e6
    mean_sample_rate = 1.0 / mean_iat_s
    freqs_hz = freqs * mean_sample_rate

    freq_target = 1.0 / TARGET_PERIOD_S
    if freq_target >= mean_sample_rate / 2:
        return None, None, None
    freq_diffs = np.abs(freqs_hz - freq_target)
    best_idx = np.argmin(freq_diffs)
    if best_idx == 0:
        return None, None, None

    phase = np.angle(spectrum[best_idx])
    power = psd[best_idx]

    # Excess in +-5% band
    total = psd[1:].sum()
    nfb = len(psd[1:])
    f_lo = freq_target * 0.95
    f_hi = freq_target * 1.05
    mask = (freqs_hz[1:] >= f_lo) & (freqs_hz[1:] <= f_hi)
    if mask.any() and total > 0:
        bp = psd[1:][mask].sum()
        nb = mask.sum()
        exp = total * nb / nfb
        excess = bp / exp if exp > 0 else 0
    else:
        excess = 0

    return phase, power, excess


def angular_distance(a, b):
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def circular_mean_R(phases):
    """Return (mean_phase, R) for an array of phases."""
    cs = np.cos(phases).sum()
    ss = np.sin(phases).sum()
    R = np.sqrt(cs**2 + ss**2) / len(phases)
    mean = np.arctan2(ss, cs)
    return mean, R


def main():
    print("=" * 70)
    print("PHASE STABILITY: Is the 5min phase consistent across time segments?")
    print(f"  {N_SEGMENTS} segments of ~65 minutes each (~13 five-minute cycles)")
    print("=" * 70)

    all_segment_phases = {}
    full_day_phases = {}

    for ticker in TICKERS:
        ts, rth_open, rth_close = load_rth_timestamps(ticker)
        if len(ts) < 2000:
            continue

        # Full day phase (detrended)
        iat_ns = np.diff(ts)
        iat_us = iat_ns / 1000.0
        nz = iat_us > 0
        iat_nz = iat_us[nz]
        ts_nz = ts[:-1][nz]

        if len(iat_nz) < 1000:
            continue

        detrended_full = detrend(iat_nz, ts_nz)
        full_phase, full_power, full_excess = extract_phase_at_5min(detrended_full)
        if full_phase is None:
            continue

        full_day_phases[ticker] = {
            "phase": full_phase,
            "excess": full_excess,
        }

        # Split into time segments
        segment_boundaries = np.linspace(rth_open, rth_close, N_SEGMENTS + 1).astype(np.int64)
        segment_phases = []

        for s in range(N_SEGMENTS):
            seg_start = segment_boundaries[s]
            seg_end = segment_boundaries[s + 1]

            # Get timestamps in this segment
            seg_mask = (ts >= seg_start) & (ts < seg_end)
            seg_ts = ts[seg_mask]
            if len(seg_ts) < 500:
                continue

            seg_iat_ns = np.diff(seg_ts)
            seg_iat_us = seg_iat_ns / 1000.0
            seg_nz = seg_iat_us > 0
            seg_iat_nz = seg_iat_us[seg_nz]
            seg_ts_nz = seg_ts[:-1][seg_nz]

            if len(seg_iat_nz) < 300:
                continue

            # Detrend within segment (local detrending)
            seg_detrended = detrend(seg_iat_nz, seg_ts_nz, n_bins=20)
            sp, _, se = extract_phase_at_5min(seg_detrended)
            if sp is not None:
                segment_phases.append({
                    "phase": sp,
                    "excess": se,
                    "n_trades": len(seg_ts),
                    "segment": s,
                })

        if len(segment_phases) >= 3:
            all_segment_phases[ticker] = segment_phases
            phases_arr = np.array([sp["phase"] for sp in segment_phases])
            seg_mean, seg_R = circular_mean_R(phases_arr)

            print(f"  {ticker}: full={np.degrees(full_phase):+.1f}, "
                  f"{len(segment_phases)} segments, "
                  f"seg_R={seg_R:.3f}, "
                  f"seg_mean={np.degrees(seg_mean):+.1f}")

    if len(all_segment_phases) < 4:
        print("Too few tickers with enough segments")
        return

    # ── Per-ticker segment detail ───────────────────────────────
    print(f"\n{'=' * 70}")
    print("SEGMENT-BY-SEGMENT PHASES (degrees)")
    print(f"{'=' * 70}\n")

    header = f"  {'Ticker':>8s}  {'Full':>7s}"
    for s in range(N_SEGMENTS):
        header += f"  {'S' + str(s):>7s}"
    header += f"  {'R':>6s}  {'Spread':>7s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    ticker_R_values = {}
    for ticker in TICKERS:
        if ticker not in all_segment_phases:
            continue
        sp = all_segment_phases[ticker]
        fp = full_day_phases[ticker]["phase"]

        row = f"  {ticker:>8s}  {np.degrees(fp):>+7.1f}"
        phases_arr = []
        seg_map = {s["segment"]: s for s in sp}
        for s in range(N_SEGMENTS):
            if s in seg_map:
                deg = np.degrees(seg_map[s]["phase"])
                row += f"  {deg:>+7.1f}"
                phases_arr.append(seg_map[s]["phase"])
            else:
                row += f"  {'---':>7s}"

        phases_arr = np.array(phases_arr)
        _, R = circular_mean_R(phases_arr)
        ticker_R_values[ticker] = R

        # Spread: max angular distance between any two segments
        max_dist = 0
        for i in range(len(phases_arr)):
            for j in range(i + 1, len(phases_arr)):
                d = angular_distance(phases_arr[i], phases_arr[j])
                max_dist = max(max_dist, d)

        row += f"  {R:>6.3f}  {np.degrees(max_dist):>7.1f}"
        print(row)

    # ── Excess power in segments ────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EXCESS POWER AT 5MIN IN EACH SEGMENT")
    print(f"{'=' * 70}\n")

    header2 = f"  {'Ticker':>8s}  {'Full':>6s}"
    for s in range(N_SEGMENTS):
        header2 += f"  {'S' + str(s):>6s}"
    print(header2)
    print(f"  {'-' * (len(header2) - 2)}")

    for ticker in TICKERS:
        if ticker not in all_segment_phases:
            continue
        fe = full_day_phases[ticker]["excess"]
        row = f"  {ticker:>8s}  {fe:>6.1f}"
        seg_map = {s["segment"]: s for s in all_segment_phases[ticker]}
        for s in range(N_SEGMENTS):
            if s in seg_map:
                row += f"  {seg_map[s]['excess']:>6.1f}"
            else:
                row += f"  {'---':>6s}"
        print(row)

    # ── Cross-ticker comparison: segment R vs tick count ────────
    print(f"\n{'=' * 70}")
    print("PHASE CONSISTENCY (R) vs TICK COUNT")
    print(f"{'=' * 70}\n")

    print(f"  {'Ticker':>8s}  {'R':>6s}  {'Ticks':>10s}  {'Excess':>7s}  {'Interpretation':>20s}")
    print(f"  {'-' * 56}")

    for ticker in TICKERS:
        if ticker not in ticker_R_values:
            continue
        R = ticker_R_values[ticker]
        fe = full_day_phases[ticker]["excess"]
        ts, _, _ = load_rth_timestamps(ticker)
        n = len(ts)

        if R > 0.8:
            interp = "STABLE phase"
        elif R > 0.5:
            interp = "MODERATE stability"
        elif R > 0.3:
            interp = "WEAK stability"
        else:
            interp = "UNSTABLE (noise)"

        print(f"  {ticker:>8s}  {R:>6.3f}  {n:>10,}  {fe:>7.1f}  {interp:>20s}")

    # ── Pairwise: does AAPL-NVDA coincidence hold per segment? ──
    print(f"\n{'=' * 70}")
    print("AAPL-NVDA DISTANCE PER SEGMENT")
    print(f"{'=' * 70}\n")

    if "AAPL" in all_segment_phases and "NVDA" in all_segment_phases:
        aapl_segs = {s["segment"]: s for s in all_segment_phases["AAPL"]}
        nvda_segs = {s["segment"]: s for s in all_segment_phases["NVDA"]}

        common_segs = set(aapl_segs.keys()) & set(nvda_segs.keys())
        if common_segs:
            print(f"  {'Segment':>8s}  {'AAPL':>8s}  {'NVDA':>8s}  {'Distance':>9s}  {'Seconds':>8s}")
            print(f"  {'-' * 46}")

            distances = []
            for s in sorted(common_segs):
                ap = np.degrees(aapl_segs[s]["phase"])
                np_ = np.degrees(nvda_segs[s]["phase"])
                dist = np.degrees(angular_distance(aapl_segs[s]["phase"], nvda_segs[s]["phase"]))
                secs = dist / 360 * 300
                distances.append(dist)
                print(f"  {'S' + str(s):>8s}  {ap:>+8.1f}  {np_:>+8.1f}  {dist:>9.1f}  {secs:>8.1f}")

            print(f"\n  Mean distance: {np.mean(distances):.1f} deg ({np.mean(distances)/360*300:.0f}s)")
            print(f"  Std distance: {np.std(distances):.1f} deg")
            print(f"  Full-day distance: 0.9 deg")

            if np.mean(distances) < 30:
                print(f"  -> AAPL-NVDA phases TRACK across segments (mean < 30 deg)")
            elif np.mean(distances) < 60:
                print(f"  -> AAPL-NVDA phases partially track (30-60 deg)")
            else:
                print(f"  -> AAPL-NVDA phases do NOT track (mean > 60 deg)")

    # ── Verdict ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}\n")

    R_vals = list(ticker_R_values.values())
    mean_R = np.mean(R_vals)
    max_R = np.max(R_vals)
    min_R = np.min(R_vals)

    print(f"  Mean within-ticker segment R: {mean_R:.3f} (range: {min_R:.3f} to {max_R:.3f})")
    print(f"  R=1: same phase every segment. R=0: random every segment.")

    if mean_R > 0.7:
        print(f"\n  -> Phases are STABLE across segments: three clocks CONFIRMED")
    elif mean_R > 0.4:
        print(f"\n  -> Phases have MODERATE stability: three clocks PLAUSIBLE")
        print(f"  -> Some real phase structure, but noisy at 65-min resolution")
    elif mean_R > 0.2:
        print(f"\n  -> Phases have WEAK stability: three clocks DOUBTFUL")
        print(f"  -> The 5min phase varies substantially within a single day")
    else:
        print(f"\n  -> Phases are UNSTABLE: three clocks NOT SUPPORTED")
        print(f"  -> Phase at 5min is essentially random segment-to-segment")
        print(f"  -> The cluster structure from the full-day FFT may be a single-realization artifact")

    print(f"\n  Block bootstrap showed +/-170 deg CIs (destructive to phase).")
    print(f"  Segment analysis shows R={mean_R:.3f} (constructive test of stability).")
    print(f"  The truth depends on which test you trust more.")


if __name__ == "__main__":
    main()
