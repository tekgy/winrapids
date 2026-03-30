"""Coupling test: does high excess pull phases toward 120 degree spacing?

Tekgy's hypothesis: the 120 degree equilibrium is the Nash attractor for
3 agents sharing a periodic resource. Algorithms couple/uncouple throughout
the day. When excess is high (algorithms active), phases should settle
closer to the maximally even 120 degree configuration. When excess is low,
phases scatter.

Test: for each 65-minute segment, run k=3 circular k-means, measure
the deviation of cluster spacings from 120 degrees, and correlate with
mean spectral excess in that segment.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

TICKERS = ["AAPL", "MSFT", "AMD", "NVDA", "TSLA", "META", "KO", "BRK.B", "JNJ", "CHWY"]

TARGET_PERIOD_S = 300
N_SEGMENTS = 6


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


def extract_phase_and_excess(iat_us_nz):
    """Extract phase and excess at 5min."""
    n = len(iat_us_nz)
    if n < 300:
        return None, None
    centered = iat_us_nz - iat_us_nz.mean()
    spectrum = np.fft.rfft(centered.astype(np.float32))
    psd = np.abs(spectrum) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    mean_iat_s = iat_us_nz.mean() / 1e6
    sr = 1.0 / mean_iat_s
    freqs_hz = freqs * sr

    ft = 1.0 / TARGET_PERIOD_S
    if ft >= sr / 2:
        return None, None
    best_idx = np.argmin(np.abs(freqs_hz - ft))
    if best_idx == 0:
        return None, None

    phase = np.angle(spectrum[best_idx])

    # Excess in +-5% band
    total = psd[1:].sum()
    nfb = len(psd[1:])
    mask = (freqs_hz[1:] >= ft * 0.95) & (freqs_hz[1:] <= ft * 1.05)
    if mask.any() and total > 0:
        bp = psd[1:][mask].sum()
        nb = mask.sum()
        exp = total * nb / nfb
        excess = bp / exp if exp > 0 else 0
    else:
        excess = 0

    return phase, excess


def angular_distance(a, b):
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def circular_kmeans(phases, k, n_restarts=20):
    """Circular k-means, best of n_restarts."""
    n = len(phases)
    best_ss = float('inf')
    best_result = None

    for seed in range(n_restarts):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(-np.pi, np.pi, k)

        for _ in range(100):
            dists = np.array([[angular_distance(p, c) for c in centers] for p in phases])
            labels = np.argmin(dists, axis=1)

            new_centers = np.zeros(k)
            for j in range(k):
                mask = labels == j
                if mask.sum() == 0:
                    new_centers[j] = centers[j]
                    continue
                cs = np.cos(phases[mask]).sum()
                ss = np.sin(phases[mask]).sum()
                new_centers[j] = np.arctan2(ss, cs)

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        within_ss = 0
        cluster_R = np.zeros(k)
        for j in range(k):
            mask = labels == j
            if mask.sum() == 0:
                continue
            cp = phases[mask]
            cs = np.cos(cp).sum()
            ss = np.sin(cp).sum()
            cluster_R[j] = np.sqrt(cs**2 + ss**2) / len(cp)
            within_ss += mask.sum() * (1 - cluster_R[j])

        if within_ss < best_ss:
            best_ss = within_ss
            best_result = (labels, centers, cluster_R, within_ss)

    return best_result


def spacing_deviation(centers_3):
    """How far are three cluster centers from 120 degree equal spacing?

    Returns RMS deviation of the three inter-center spacings from 120 degrees.
    """
    # Sort centers
    sorted_c = np.sort(centers_3 % (2 * np.pi))
    spacings = np.diff(sorted_c)
    spacings = np.append(spacings, 2 * np.pi - sorted_c[-1] + sorted_c[0])
    # Convert to degrees
    spacings_deg = np.degrees(spacings)
    # RMS deviation from 120
    deviations = spacings_deg - 120.0
    rms = np.sqrt(np.mean(deviations**2))
    return rms, spacings_deg


def main():
    print("=" * 70)
    print("COUPLING TEST: Does high excess pull phases toward 120 deg?")
    print(f"  {N_SEGMENTS} segments x {len(TICKERS)} tickers")
    print("=" * 70)

    # ── Load all timestamps ─────────────────────────────────────
    ticker_ts = {}
    rth_bounds = {}
    for ticker in TICKERS:
        ts, o, c = load_rth_timestamps(ticker)
        if len(ts) > 2000:
            ticker_ts[ticker] = ts
            rth_bounds[ticker] = (o, c)

    # Use AAPL bounds for segment boundaries (all same day)
    o, c = rth_bounds["AAPL"]
    segment_boundaries = np.linspace(o, c, N_SEGMENTS + 1).astype(np.int64)

    # ── Extract per-segment phases and excesses ─────────────────
    segment_data = []  # list of dicts per segment

    for s in range(N_SEGMENTS):
        seg_start = segment_boundaries[s]
        seg_end = segment_boundaries[s + 1]

        seg_phases = {}
        seg_excesses = {}

        for ticker in TICKERS:
            if ticker not in ticker_ts:
                continue
            ts = ticker_ts[ticker]
            mask = (ts >= seg_start) & (ts < seg_end)
            seg_ts = ts[mask]
            if len(seg_ts) < 500:
                continue

            iat_ns = np.diff(seg_ts)
            iat_us = iat_ns / 1000.0
            nz = iat_us > 0
            iat_nz = iat_us[nz]
            ts_nz = seg_ts[:-1][nz]
            if len(iat_nz) < 300:
                continue

            detrended = detrend(iat_nz, ts_nz, n_bins=20)
            phase, excess = extract_phase_and_excess(detrended)
            if phase is not None:
                seg_phases[ticker] = phase
                seg_excesses[ticker] = excess

        segment_data.append({
            "phases": seg_phases,
            "excesses": seg_excesses,
            "segment": s,
        })

    # ── Per-segment k=3 clustering and 120-degree deviation ─────
    print(f"\n{'=' * 70}")
    print("PER-SEGMENT: k=3 clustering, 120-degree deviation, mean excess")
    print(f"{'=' * 70}\n")

    print(f"  {'Seg':>4s}  {'nTick':>6s}  {'MeanExc':>8s}  {'MedianExc':>10s}  "
          f"{'120dev':>7s}  {'MeanR':>6s}  {'Spacings':>24s}")
    print(f"  {'-' * 72}")

    results = []

    for sd in segment_data:
        tickers_in_seg = sorted(sd["phases"].keys())
        n_tickers = len(tickers_in_seg)

        if n_tickers < 6:  # need enough for k=3
            continue

        phases = np.array([sd["phases"][t] for t in tickers_in_seg])
        excesses = np.array([sd["excesses"][t] for t in tickers_in_seg])

        mean_exc = np.mean(excesses)
        median_exc = np.median(excesses)

        # k=3 clustering
        labels, centers, cluster_R, within_ss = circular_kmeans(phases, 3)
        mean_R = np.mean(cluster_R)

        # 120-degree deviation
        dev, spacings = spacing_deviation(centers)

        results.append({
            "segment": sd["segment"],
            "mean_excess": mean_exc,
            "median_excess": median_exc,
            "dev_120": dev,
            "mean_R": mean_R,
            "spacings": spacings,
            "n_tickers": n_tickers,
        })

        sp_str = f"[{spacings[0]:.0f}, {spacings[1]:.0f}, {spacings[2]:.0f}]"
        print(f"  S{sd['segment']:>3d}  {n_tickers:>6d}  {mean_exc:>8.1f}  {median_exc:>10.1f}  "
              f"{dev:>7.1f}  {mean_R:>6.3f}  {sp_str:>24s}")

    if len(results) < 4:
        print("\n  Too few segments for correlation")
        return

    # ── Correlation: excess vs 120-degree proximity ─────────────
    print(f"\n{'=' * 70}")
    print("CORRELATION: Does high excess predict proximity to 120 deg?")
    print(f"{'=' * 70}\n")

    mean_excs = np.array([r["mean_excess"] for r in results])
    median_excs = np.array([r["median_excess"] for r in results])
    devs = np.array([r["dev_120"] for r in results])
    mean_Rs = np.array([r["mean_R"] for r in results])

    # Proximity = inverse of deviation (higher = closer to 120)
    # Use negative deviation so positive correlation = high excess -> close to 120

    # Pearson correlations
    if len(mean_excs) >= 3:
        # Excess vs 120-dev (negative = high excess -> low deviation -> good)
        r_exc_dev = np.corrcoef(mean_excs, devs)[0, 1]
        r_med_dev = np.corrcoef(median_excs, devs)[0, 1]
        r_exc_R = np.corrcoef(mean_excs, mean_Rs)[0, 1]

        print(f"  Correlation(mean_excess, 120_deviation):  r = {r_exc_dev:+.3f}")
        print(f"    (Negative = high excess -> closer to 120: SUPPORTS hypothesis)")
        print(f"    (Positive = high excess -> farther from 120: REFUTES hypothesis)")
        print(f"  Correlation(median_excess, 120_deviation): r = {r_med_dev:+.3f}")
        print(f"  Correlation(mean_excess, within_R):         r = {r_exc_R:+.3f}")
        print(f"    (Positive = high excess -> tighter clusters: SUPPORTS hypothesis)")

    # ── Rank correlation (more robust with 6 points) ────────────
    def rank(arr):
        temp = arr.argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(arr), dtype=float)
        return ranks

    def spearman_r(x, y):
        return np.corrcoef(rank(x), rank(y))[0, 1]

    rho_exc_dev = spearman_r(mean_excs, devs)
    rho_exc_R = spearman_r(mean_excs, mean_Rs)

    print(f"\n  Spearman rank correlations (more robust for n={len(results)}):")
    print(f"    rho(mean_excess, 120_deviation) = {rho_exc_dev:+.3f}")
    print(f"    rho(mean_excess, within_R)      = {rho_exc_R:+.3f}")

    # ── Scatter (text-based) ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SCATTER: Each point is one segment")
    print(f"{'=' * 70}\n")

    print(f"  {'Seg':>4s}  {'MeanExc':>8s}  {'120dev':>7s}  {'MeanR':>6s}  {'Direction':>12s}")
    print(f"  {'-' * 42}")

    # Sort by excess
    sorted_results = sorted(results, key=lambda r: r["mean_excess"])
    for r in sorted_results:
        direction = "COUPLED" if r["dev_120"] < np.median(devs) else "UNCOUPLED"
        print(f"  S{r['segment']:>3d}  {r['mean_excess']:>8.1f}  {r['dev_120']:>7.1f}  "
              f"{r['mean_R']:>6.3f}  {direction:>12s}")

    # ── Per-segment cluster membership ──────────────────────────
    print(f"\n{'=' * 70}")
    print("CLUSTER MEMBERSHIP PER SEGMENT (sorted by excess)")
    print(f"{'=' * 70}\n")

    for sd in sorted(segment_data, key=lambda s: np.mean(list(s["excesses"].values())) if s["excesses"] else 0):
        tickers_in = sorted(sd["phases"].keys())
        if len(tickers_in) < 6:
            continue
        phases = np.array([sd["phases"][t] for t in tickers_in])
        excesses = np.array([sd["excesses"][t] for t in tickers_in])
        mean_exc = np.mean(excesses)

        labels, centers, cluster_R, _ = circular_kmeans(phases, 3)

        print(f"  Segment {sd['segment']} (mean excess = {mean_exc:.1f}x):")
        for j in range(3):
            mask = labels == j
            members = [tickers_in[i] for i in range(len(tickers_in)) if labels[i] == j]
            center_deg = np.degrees(centers[j])
            print(f"    C{j}: {center_deg:>+7.1f} deg  R={cluster_R[j]:.3f}  "
                  f"[{', '.join(members)}]")
        print()

    # ── Verdict ─────────────────────────────────────────────────
    print(f"{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}\n")

    if len(results) >= 4:
        if rho_exc_dev < -0.5 and rho_exc_R > 0.5:
            print("  STRONG SUPPORT: High excess -> closer to 120, tighter clusters")
            print("  -> The Nash equilibrium IS the attractor; coupling strength varies")
        elif rho_exc_dev < -0.3 or rho_exc_R > 0.3:
            print("  WEAK SUPPORT: Trend in expected direction but not conclusive")
            print("  -> Suggestive at n=6 segments; needs multi-day data")
        elif abs(rho_exc_dev) < 0.2 and abs(rho_exc_R) < 0.2:
            print("  NO RELATIONSHIP: Excess and spacing are independent")
            print("  -> 120 deg spacing was coincidental in full-day average")
        else:
            print("  MIXED/UNEXPECTED: Check the scatter plot above")

    print(f"\n  NOTE: n={len(results)} segments is very small. Correlations are suggestive,")
    print(f"  not conclusive. Multi-day data (20+ segments) would be needed to")
    print(f"  distinguish real coupling from noise at p < 0.05.")


if __name__ == "__main__":
    main()
