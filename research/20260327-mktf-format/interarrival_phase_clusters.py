"""Phase cluster analysis — is the scatter actually 2-3 tight clusters?

Scout's observation: Rayleigh tests for unimodal clustering.
Three clusters evenly spaced around the circle produce R ≈ 0 even
if each cluster is internally tight. The correct test is circular
k-means (mixture of von Mises distributions).

Test k=1,2,3,4. Report within-cluster concentration and BIC-like criterion.
Also compute the angular distance matrix for visual inspection.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

TICKERS = ["AAPL", "MSFT", "AMD", "NVDA", "TSLA", "META", "KO", "BRK.B", "JNJ", "CHWY"]

TARGET_PERIOD_S = 300  # 5 minutes


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


def extract_phase_at_period(iat_us_nz, period_s):
    """Extract phase at a single target period."""
    n = len(iat_us_nz)
    centered = iat_us_nz - iat_us_nz.mean()
    spectrum = np.fft.rfft(centered.astype(np.float32))
    freqs = np.fft.rfftfreq(n)
    mean_iat_s = iat_us_nz.mean() / 1e6
    mean_sample_rate = 1.0 / mean_iat_s
    freqs_hz = freqs * mean_sample_rate

    freq_target = 1.0 / period_s
    if freq_target >= mean_sample_rate / 2:
        return None, None
    freq_diffs = np.abs(freqs_hz - freq_target)
    best_idx = np.argmin(freq_diffs)
    if best_idx == 0:
        return None, None

    phase = np.angle(spectrum[best_idx])
    power = np.abs(spectrum[best_idx]) ** 2 / n
    return phase, power


def angular_distance(a, b):
    """Shortest angular distance between two angles in radians."""
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def circular_kmeans(phases, k, max_iter=100):
    """Circular k-means clustering.

    Returns: labels, centers, within-cluster R values, total within-cluster SS.
    """
    n = len(phases)
    # Initialize with evenly-spaced centers + small perturbation
    rng = np.random.RandomState(42)
    centers = np.linspace(-np.pi, np.pi, k, endpoint=False) + rng.randn(k) * 0.1

    for _ in range(max_iter):
        # Assign each point to nearest center (angular distance)
        dists = np.array([[angular_distance(p, c) for c in centers] for p in phases])
        labels = np.argmin(dists, axis=1)

        # Update centers using circular mean
        new_centers = np.zeros(k)
        for j in range(k):
            mask = labels == j
            if mask.sum() == 0:
                new_centers[j] = centers[j]
                continue
            cos_sum = np.cos(phases[mask]).sum()
            sin_sum = np.sin(phases[mask]).sum()
            new_centers[j] = np.arctan2(sin_sum, cos_sum)

        # Check convergence
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    # Compute within-cluster R for each cluster
    cluster_R = np.zeros(k)
    cluster_sizes = np.zeros(k, dtype=int)
    total_within_ss = 0.0
    for j in range(k):
        mask = labels == j
        cluster_sizes[j] = mask.sum()
        if mask.sum() == 0:
            continue
        cp = phases[mask]
        cos_sum = np.cos(cp).sum()
        sin_sum = np.sin(cp).sum()
        cluster_R[j] = np.sqrt(cos_sum**2 + sin_sum**2) / len(cp)
        # Within-cluster angular variance = 1 - R
        total_within_ss += mask.sum() * (1 - cluster_R[j])

    return labels, centers, cluster_R, cluster_sizes, total_within_ss


def main():
    print("=" * 70)
    print("PHASE CLUSTER ANALYSIS: Multi-cluster structure at 5min?")
    print("=" * 70)

    # ── Extract phases ──────────────────────────────────────────
    ticker_phases = {}
    ticker_powers = {}

    for ticker in TICKERS:
        ts = load_rth_timestamps(ticker)
        if len(ts) < 2000:
            continue
        iat_ns = np.diff(ts)
        iat_us = iat_ns / 1000.0
        nz = iat_us > 0
        iat_nz = iat_us[nz]
        ts_nz = ts[:-1][nz]
        if len(iat_nz) < 1000:
            continue

        detrended = detrend(iat_nz, ts_nz)
        phase, power = extract_phase_at_period(detrended, TARGET_PERIOD_S)
        if phase is not None:
            ticker_phases[ticker] = phase
            ticker_powers[ticker] = power
            print(f"  {ticker}: phase = {np.degrees(phase):+.1f} deg")

    if len(ticker_phases) < 4:
        print("Too few tickers for cluster analysis")
        return

    tickers_ordered = [t for t in TICKERS if t in ticker_phases]
    phases = np.array([ticker_phases[t] for t in tickers_ordered])
    n = len(phases)

    # ── Angular distance matrix ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ANGULAR DISTANCE MATRIX (degrees)")
    print(f"{'=' * 70}\n")

    header = f"{'':>8s}"
    for t in tickers_ordered:
        header += f"  {t:>6s}"
    print(header)
    print("-" * len(header))

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        row = f"{tickers_ordered[i]:>8s}"
        for j in range(n):
            d = np.degrees(angular_distance(phases[i], phases[j]))
            dist_matrix[i, j] = d
            row += f"  {d:>6.1f}"
        print(row)

    # ── Nearest-neighbor pairs ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("NEAREST-NEIGHBOR PAIRS")
    print(f"{'=' * 70}\n")

    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = 999
        j = np.argmin(dists)
        print(f"  {tickers_ordered[i]:>8s} -> {tickers_ordered[j]:>8s}  "
              f"({dist_matrix[i, j]:.1f} deg = {dist_matrix[i, j] / 360 * 300:.0f}s at 5min)")

    # ── Circular k-means for k=1,2,3,4 ─────────────────────────
    print(f"\n{'=' * 70}")
    print("CIRCULAR K-MEANS (k = 1, 2, 3, 4)")
    print(f"{'=' * 70}")

    # Run multiple random initializations for each k
    best_results = {}
    for k in [1, 2, 3, 4]:
        best_ss = float('inf')
        for seed in range(20):
            rng = np.random.RandomState(seed)
            # Random initialization
            init_centers = rng.uniform(-np.pi, np.pi, k)
            labels_tmp, centers_tmp, R_tmp, sizes_tmp, ss_tmp = circular_kmeans(phases, k)
            if ss_tmp < best_ss:
                best_ss = ss_tmp
                best_results[k] = (labels_tmp, centers_tmp, R_tmp, sizes_tmp, ss_tmp)

    for k in [1, 2, 3, 4]:
        labels, centers, cluster_R, sizes, within_ss = best_results[k]

        # BIC-like: n * log(within_ss/n) + k * log(n)
        # (using circular variance as the "SS")
        if within_ss > 0:
            bic = n * np.log(within_ss / n) + (2 * k) * np.log(n)
        else:
            bic = -np.inf

        print(f"\n  k = {k}:")
        print(f"    Within-cluster angular variance: {within_ss:.3f}")
        print(f"    BIC-like criterion: {bic:.2f}")

        for j in range(k):
            mask = labels == j
            members = [tickers_ordered[i] for i in range(n) if labels[i] == j]
            center_deg = np.degrees(centers[j])
            print(f"    Cluster {j}: center={center_deg:+.1f} deg, R={cluster_R[j]:.3f}, "
                  f"n={sizes[j]}: {', '.join(members)}")

    # ── Scout's lead-lag hypothesis ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("LEAD-LAG: Phase offset as temporal displacement")
    print(f"{'=' * 70}\n")

    # Compute phase offsets relative to AAPL (the reference point)
    if "AAPL" in ticker_phases:
        ref = ticker_phases["AAPL"]
        print(f"  Phase offsets relative to AAPL ({np.degrees(ref):+.1f} deg):\n")
        print(f"  {'Ticker':>8s}  {'Offset':>8s}  {'Lead/Lag':>10s}  {'Seconds':>8s}")
        print(f"  {'-' * 40}")

        for t in tickers_ordered:
            offset_rad = np.arctan2(np.sin(ticker_phases[t] - ref),
                                    np.cos(ticker_phases[t] - ref))
            offset_deg = np.degrees(offset_rad)
            offset_s = offset_deg / 360 * 300  # 5min = 300s
            direction = "LEADS" if offset_s < 0 else "LAGS" if offset_s > 0 else "SYNC"
            print(f"  {t:>8s}  {offset_deg:>+8.1f}  {direction:>10s}  {offset_s:>+8.1f}")

    # ── Interpretation ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    print("""
  The Rayleigh test detects unimodal clustering (one shared clock).
  It fails when multiple tight clusters cancel in the mean vector.

  If k=3 has markedly lower BIC than k=1:
    -> Multi-cluster structure confirmed
    -> Each cluster = a "strategy clock" shared by similar tickers
    -> K04 on (cos(phase), sin(phase)) vectors recovers this structure
    -> Phase lead-lag within clusters = market microstructure finding

  If k=1 ~ k=3 in BIC:
    -> Truly independent — no hidden cluster structure
    -> Each ticker has its own algorithmic phase
    -> K04 on phase would NOT find meaningful clusters

  For K-SS01(PHASE) schema:
    -> Store as (cos(phase), sin(phase)) float32 pair, NOT raw angle
    -> Enables Euclidean K04 correlation (no special circular metrics)
    -> Scout: circular distance = angular distance
""")


if __name__ == "__main__":
    main()
