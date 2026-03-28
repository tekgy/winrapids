"""Bootstrap confidence intervals for phase estimates at 5min.

The three-clocks finding rests on point estimates from single FFT bins.
How robust are those phases? Block bootstrap the IAT sequence (preserving
temporal structure), re-extract phases, and compute 95% CIs.

Key question: is the 0.9 degree AAPL-NVDA coincidence real or noise?
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

TICKERS = ["AAPL", "MSFT", "AMD", "NVDA", "TSLA", "META", "KO", "BRK.B", "JNJ", "CHWY"]

TARGET_PERIOD_S = 300  # 5 minutes
N_BOOTSTRAP = 500
BLOCK_SIZE_TRADES = 5000  # ~10-30 seconds of trades depending on ticker


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


def extract_phase_at_5min(iat_us_nz):
    """Extract phase at 5min from detrended IATs."""
    n = len(iat_us_nz)
    centered = iat_us_nz - iat_us_nz.mean()
    spectrum = np.fft.rfft(centered.astype(np.float32))
    freqs = np.fft.rfftfreq(n)
    mean_iat_s = iat_us_nz.mean() / 1e6
    mean_sample_rate = 1.0 / mean_iat_s
    freqs_hz = freqs * mean_sample_rate

    freq_target = 1.0 / TARGET_PERIOD_S
    if freq_target >= mean_sample_rate / 2:
        return None
    freq_diffs = np.abs(freqs_hz - freq_target)
    best_idx = np.argmin(freq_diffs)
    if best_idx == 0:
        return None
    return np.angle(spectrum[best_idx])


def block_bootstrap(iat_us, block_size, rng):
    """Block bootstrap: resample contiguous blocks to preserve temporal structure."""
    n = len(iat_us)
    n_blocks = (n + block_size - 1) // block_size
    # Create blocks
    blocks = []
    for i in range(0, n, block_size):
        blocks.append(iat_us[i:i + block_size])
    # Resample blocks with replacement
    indices = rng.randint(0, len(blocks), size=n_blocks)
    resampled = np.concatenate([blocks[i] for i in indices])
    return resampled[:n]  # trim to original length


def angular_distance(a, b):
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def main():
    print("=" * 70)
    print("PHASE BOOTSTRAP: How robust are the three clocks?")
    print(f"  {N_BOOTSTRAP} bootstrap replicates, block size = {BLOCK_SIZE_TRADES} trades")
    print("=" * 70)

    rng = np.random.RandomState(42)

    # ── Collect original phases and bootstrap distributions ──────
    ticker_data = {}

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

        # Original phase (detrended)
        detrended = detrend(iat_nz, ts_nz)
        orig_phase = extract_phase_at_5min(detrended)
        if orig_phase is None:
            continue

        # Bootstrap phases
        # Note: we bootstrap the detrended series to measure phase estimation
        # uncertainty, not detrending uncertainty (separate question)
        boot_phases = []
        for b in range(N_BOOTSTRAP):
            boot_iat = block_bootstrap(detrended, BLOCK_SIZE_TRADES, rng)
            bp = extract_phase_at_5min(boot_iat)
            if bp is not None:
                boot_phases.append(bp)

        boot_phases = np.array(boot_phases)

        # Circular statistics on bootstrap distribution
        cos_sum = np.cos(boot_phases).sum()
        sin_sum = np.sin(boot_phases).sum()
        R_boot = np.sqrt(cos_sum**2 + sin_sum**2) / len(boot_phases)
        mean_boot = np.arctan2(sin_sum, cos_sum)

        # Circular 95% CI: sort angular distances from mean, take 97.5th percentile
        dists_from_mean = np.array([angular_distance(bp, mean_boot) for bp in boot_phases])
        ci_95 = np.degrees(np.percentile(dists_from_mean, 95))

        # Bias: difference between original and bootstrap mean
        bias = np.degrees(angular_distance(orig_phase, mean_boot))

        ticker_data[ticker] = {
            "orig_phase": orig_phase,
            "boot_mean": mean_boot,
            "boot_R": R_boot,
            "ci_95": ci_95,
            "bias": bias,
            "boot_phases": boot_phases,
            "n_ticks": len(ts),
        }

        print(f"  {ticker}: orig={np.degrees(orig_phase):+.1f} deg, "
              f"boot_mean={np.degrees(mean_boot):+.1f} deg, "
              f"R={R_boot:.3f}, 95%CI=+/-{ci_95:.1f} deg, bias={bias:.1f} deg")

    if len(ticker_data) < 4:
        print("Too few tickers")
        return

    # ── Phase stability report ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE ESTIMATES WITH 95% CONFIDENCE INTERVALS")
    print(f"{'=' * 70}\n")

    print(f"  {'Ticker':>8s}  {'Phase':>8s}  {'95% CI':>10s}  {'Boot R':>7s}  {'Ticks':>10s}")
    print(f"  {'-' * 50}")

    for ticker in TICKERS:
        if ticker not in ticker_data:
            continue
        d = ticker_data[ticker]
        print(f"  {ticker:>8s}  {np.degrees(d['orig_phase']):>+8.1f}  "
              f"+/-{d['ci_95']:>6.1f}  {d['boot_R']:>7.3f}  {d['n_ticks']:>10,}")

    # ── Key pair distances with error propagation ───────────────
    print(f"\n{'=' * 70}")
    print("KEY PAIR DISTANCES (bootstrap distribution of angular distance)")
    print(f"{'=' * 70}\n")

    pairs = [
        ("AAPL", "NVDA"),   # same cluster, 0.9 deg
        ("KO", "CHWY"),     # same cluster, 4.0 deg
        ("MSFT", "AMD"),    # same cluster, 36.2 deg
        ("AAPL", "KO"),     # cross-cluster, 61.4 deg
        ("AAPL", "MSFT"),   # cross-cluster, 111.2 deg
    ]

    print(f"  {'Pair':>16s}  {'Obs dist':>9s}  {'Boot mean':>10s}  {'Boot 5%':>8s}  "
          f"{'Boot 95%':>9s}  {'p(>obs)':>8s}")
    print(f"  {'-' * 66}")

    for t1, t2 in pairs:
        if t1 not in ticker_data or t2 not in ticker_data:
            continue
        d1 = ticker_data[t1]
        d2 = ticker_data[t2]
        obs_dist = np.degrees(angular_distance(d1["orig_phase"], d2["orig_phase"]))

        # Bootstrap distribution of pairwise distance
        boot_dists = []
        n_boot = min(len(d1["boot_phases"]), len(d2["boot_phases"]))
        for i in range(n_boot):
            bd = angular_distance(d1["boot_phases"][i], d2["boot_phases"][i])
            boot_dists.append(np.degrees(bd))
        boot_dists = np.array(boot_dists)

        boot_mean = np.mean(boot_dists)
        boot_5 = np.percentile(boot_dists, 5)
        boot_95 = np.percentile(boot_dists, 95)
        # p-value: fraction of bootstrap where distance > observed
        # (for same-cluster pairs, we want this to be small)
        p_greater = np.mean(boot_dists > obs_dist)

        label = f"{t1}-{t2}"
        print(f"  {label:>16s}  {obs_dist:>9.1f}  {boot_mean:>10.1f}  {boot_5:>8.1f}  "
              f"{boot_95:>9.1f}  {p_greater:>8.3f}")

    # ── Cluster stability ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CLUSTER STABILITY: How often do k=3 assignments hold?")
    print(f"{'=' * 70}\n")

    # For each bootstrap replicate, assign tickers to the three original clusters
    # based on nearest center, and count how often assignments match original
    original_centers = np.array([
        np.radians(-18.0),   # mega-cap tech
        np.radians(72.4),    # consumer/value
        np.radians(-157.1),  # phase-shifted tech
    ])
    original_labels = {
        "AAPL": 0, "NVDA": 0, "TSLA": 0,
        "KO": 1, "BRK.B": 1, "JNJ": 1, "CHWY": 1,
        "MSFT": 2, "AMD": 2, "META": 2,
    }
    cluster_names = ["Mega-cap tech", "Consumer/value", "Phase-shifted tech"]

    tickers_with_data = [t for t in TICKERS if t in ticker_data]
    n_tickers = len(tickers_with_data)

    # Count how often each ticker stays in its original cluster
    stability_counts = {t: 0 for t in tickers_with_data}
    n_replicates = N_BOOTSTRAP

    for b in range(n_replicates):
        for t in tickers_with_data:
            if b >= len(ticker_data[t]["boot_phases"]):
                continue
            bp = ticker_data[t]["boot_phases"][b]
            # Assign to nearest original center
            dists = [angular_distance(bp, c) for c in original_centers]
            assigned = np.argmin(dists)
            if assigned == original_labels.get(t, -1):
                stability_counts[t] += 1

    print(f"  {'Ticker':>8s}  {'Cluster':>22s}  {'Stability':>10s}")
    print(f"  {'-' * 44}")
    for t in tickers_with_data:
        if t in original_labels:
            cluster = cluster_names[original_labels[t]]
            pct = 100 * stability_counts[t] / n_replicates
            print(f"  {t:>8s}  {cluster:>22s}  {pct:>9.1f}%")

    # ── Overall verdict ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}\n")

    ci_values = [ticker_data[t]["ci_95"] for t in tickers_with_data]
    mean_ci = np.mean(ci_values)
    min_ci = np.min(ci_values)
    max_ci = np.max(ci_values)

    print(f"  Mean 95% CI width: +/-{mean_ci:.1f} deg (range: {min_ci:.1f} to {max_ci:.1f})")

    aapl_nvda_dist = np.degrees(angular_distance(
        ticker_data["AAPL"]["orig_phase"], ticker_data["NVDA"]["orig_phase"]))
    aapl_ci = ticker_data["AAPL"]["ci_95"]
    nvda_ci = ticker_data["NVDA"]["ci_95"]
    combined_ci = np.sqrt(aapl_ci**2 + nvda_ci**2)

    print(f"\n  AAPL-NVDA: {aapl_nvda_dist:.1f} deg observed")
    print(f"    AAPL 95% CI: +/-{aapl_ci:.1f} deg")
    print(f"    NVDA 95% CI: +/-{nvda_ci:.1f} deg")
    print(f"    Combined CI: +/-{combined_ci:.1f} deg")

    if aapl_nvda_dist < combined_ci:
        print(f"    -> Observed distance ({aapl_nvda_dist:.1f}) < combined CI ({combined_ci:.1f})")
        print(f"    -> CANNOT distinguish from same phase (consistent with synchronization)")
    else:
        print(f"    -> Observed distance ({aapl_nvda_dist:.1f}) > combined CI ({combined_ci:.1f})")
        print(f"    -> Phases are distinguishably different")

    stab_values = [100 * stability_counts[t] / n_replicates for t in tickers_with_data if t in original_labels]
    mean_stab = np.mean(stab_values)
    min_stab = np.min(stab_values)
    print(f"\n  Cluster assignment stability: mean={mean_stab:.1f}%, min={min_stab:.1f}%")
    if min_stab > 80:
        print(f"    -> All tickers stable (min > 80%): THREE CLOCKS ROBUST")
    elif min_stab > 50:
        print(f"    -> Most tickers stable but some borderline: THREE CLOCKS LIKELY")
    else:
        print(f"    -> Significant instability: THREE CLOCKS UNCERTAIN at n=10")


if __name__ == "__main__":
    main()
