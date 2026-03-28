"""Cross-ticker detrended comparison — testing scout's prediction.

Prediction: per-ticker institutional "fingerprints" at 15-30min will
collapse under detrending. The differences (AMD 15min vs NVDA 20min vs
TSLA 30min peaks) were U-shape variation, not genuine periodic differences.

Test: run detrended excess for all 7 tickers from the options anomaly test.
Compare raw vs detrended at 5-30min. If fingerprints collapse to similar
values, scout is right.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DATA_ROOT = Path("W:/fintek/data/fractal/K01/2025-09-02")

TICKERS = ["AAPL", "MSFT", "AMD", "NVDA", "TSLA", "META", "KO", "BRK.B", "JNJ", "CHWY"]

CADENCES = [
    (1.0,    "1s"),
    (1/5,    "5s"),
    (1/10,   "10s"),
    (1/30,   "30s"),
    (1/60,   "1min"),
    (1/120,  "2min"),
    (1/300,  "5min"),
    (1/600,  "10min"),
    (1/900,  "15min"),
    (1/1200, "20min"),
    (1/1800, "30min"),
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


def excess_at_cadences(iat_us_nz, cadences):
    n = len(iat_us_nz)
    centered = iat_us_nz - iat_us_nz.mean()
    spec = np.fft.rfft(centered.astype(np.float32))
    psd = np.abs(spec) ** 2 / n
    freqs_hz = np.fft.rfftfreq(n) * (1e6 / iat_us_nz.mean())
    total = psd[1:].sum()
    nfb = len(psd[1:])
    result = {}
    for ft, label in cadences:
        if ft >= 1e6 / (2 * iat_us_nz.mean()):
            continue
        mask = (freqs_hz[1:] >= ft * 0.95) & (freqs_hz[1:] <= ft * 1.05)
        if mask.any():
            bp = psd[1:][mask].sum()
            nb = mask.sum()
            exp = total * nb / nfb
            result[label] = round(bp / exp, 2) if exp > 0 else 0
    return result


def main():
    print("=" * 80)
    print("SCOUT'S PREDICTION TEST: Do institutional fingerprints collapse?")
    print("=" * 80)

    raw_results = {}
    det_results = {}

    for ticker in TICKERS:
        ts = load_rth_timestamps(ticker)
        if len(ts) < 2000:
            print(f"  {ticker}: too few ticks ({len(ts)})")
            continue

        iat_ns = np.diff(ts)
        iat_us = iat_ns / 1000.0
        nz = iat_us > 0
        iat_nz = iat_us[nz]
        ts_nz = ts[:-1][nz]

        if len(iat_nz) < 1000:
            continue

        raw_results[ticker] = excess_at_cadences(iat_nz, CADENCES)

        detrended = detrend(iat_nz, ts_nz)
        det_results[ticker] = excess_at_cadences(detrended, CADENCES)

        print(f"  {ticker}: {len(ts):,} ticks -> done")

    # ── Raw comparison (institutional regime) ─────────────────
    inst_labels = ["5min", "10min", "15min", "20min", "30min"]
    exec_labels = ["1s", "5s", "10s", "30s", "1min"]

    print(f"\n{'=' * 80}")
    print("RAW institutional regime (5-30min)")
    print(f"{'=' * 80}")
    header = f"{'Ticker':>8s}"
    for l in inst_labels:
        header += f"  {l:>6s}"
    header += f"  {'Peak':>6s}  {'PkVal':>6s}"
    print(header)
    print("-" * len(header))

    for ticker in TICKERS:
        if ticker not in raw_results:
            continue
        r = raw_results[ticker]
        row = f"{ticker:>8s}"
        peak_label = ""
        peak_val = 0
        for l in inst_labels:
            v = r.get(l, 0)
            row += f"  {v:>6.1f}"
            if v > peak_val:
                peak_val = v
                peak_label = l
        row += f"  {peak_label:>6s}  {peak_val:>6.1f}"
        print(row)

    # Compute coefficient of variation across tickers at each cadence
    print(f"\n  Cross-ticker CV (std/mean):")
    for l in inst_labels:
        vals = [raw_results[t].get(l, 0) for t in TICKERS if t in raw_results and l in raw_results[t]]
        if vals:
            cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
            print(f"    {l:>6s}: CV = {cv:.2f}  (mean={np.mean(vals):.1f}, std={np.std(vals):.1f})")

    # ── Detrended comparison ──────────────────────────────────
    print(f"\n{'=' * 80}")
    print("DETRENDED institutional regime (5-30min)")
    print(f"{'=' * 80}")
    print(header)
    print("-" * len(header))

    for ticker in TICKERS:
        if ticker not in det_results:
            continue
        r = det_results[ticker]
        row = f"{ticker:>8s}"
        peak_label = ""
        peak_val = 0
        for l in inst_labels:
            v = r.get(l, 0)
            row += f"  {v:>6.1f}"
            if v > peak_val:
                peak_val = v
                peak_label = l
        row += f"  {peak_label:>6s}  {peak_val:>6.1f}"
        print(row)

    print(f"\n  Cross-ticker CV (std/mean):")
    for l in inst_labels:
        vals = [det_results[t].get(l, 0) for t in TICKERS if t in det_results and l in det_results[t]]
        if vals:
            cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
            print(f"    {l:>6s}: CV = {cv:.2f}  (mean={np.mean(vals):.1f}, std={np.std(vals):.1f})")

    # ── Execution regime for comparison ───────────────────────
    print(f"\n{'=' * 80}")
    print("DETRENDED execution regime (1s-1min) — control group")
    print(f"{'=' * 80}")
    header2 = f"{'Ticker':>8s}"
    for l in exec_labels:
        header2 += f"  {l:>6s}"
    print(header2)
    print("-" * len(header2))

    for ticker in TICKERS:
        if ticker not in det_results:
            continue
        r = det_results[ticker]
        row = f"{ticker:>8s}"
        for l in exec_labels:
            v = r.get(l, 0)
            row += f"  {v:>6.1f}"
        print(row)

    print(f"\n  Cross-ticker CV:")
    for l in exec_labels:
        vals = [det_results[t].get(l, 0) for t in TICKERS if t in det_results and l in det_results[t]]
        if vals:
            cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
            print(f"    {l:>6s}: CV = {cv:.2f}  (mean={np.mean(vals):.1f}, std={np.std(vals):.1f})")

    # ── Verdict ───────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("VERDICT: Did fingerprints collapse?")
    print(f"{'=' * 80}")

    # Compare raw vs detrended CV
    print(f"\n  {'Cadence':>8s}  {'Raw CV':>8s}  {'Det CV':>8s}  {'Change':>8s}")
    print(f"  {'-' * 38}")
    for l in inst_labels:
        raw_vals = [raw_results[t].get(l, 0) for t in TICKERS if t in raw_results and l in raw_results[t]]
        det_vals = [det_results[t].get(l, 0) for t in TICKERS if t in det_results and l in det_results[t]]
        if raw_vals and det_vals:
            raw_cv = np.std(raw_vals) / np.mean(raw_vals) if np.mean(raw_vals) > 0 else 0
            det_cv = np.std(det_vals) / np.mean(det_vals) if np.mean(det_vals) > 0 else 0
            change = det_cv - raw_cv
            print(f"  {l:>8s}  {raw_cv:>8.2f}  {det_cv:>8.2f}  {change:>+8.2f}")

    print(f"\n  If detrended CV << raw CV: fingerprints collapsed (scout confirmed)")
    print(f"  If detrended CV ~ raw CV: fingerprints are real (scout refuted)")


if __name__ == "__main__":
    main()
