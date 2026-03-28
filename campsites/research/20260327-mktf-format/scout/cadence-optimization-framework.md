# Cadence Grid Optimization Framework
## Scout Notes — 2026-03-27
## Co-developed with naturalist

---

## The Problem

The current 1-2-5 cadence grid (1s, 2s, 5s, 10s, 20s, 50s, 100s...) is intuition-based.
It should be scientifically calibrated. This document captures the framework for doing that.

---

## Three Objectives (Not One)

The cadence grid serves three distinct goals with different metrics:

**1. Model Training** — maximize total information for ML model input
- Metric: I_total(T) = MI_per_bin(T) × bins_per_day
- Optimized cadence: dense where the product peaks
- Data: from MI experiment with target variable (future returns / volatility / direction)

**2. Gaming Detection** — can we see specific algorithmic signatures?
- Metric: spectral SNR at known algorithm frequencies
- Key cadences: 60s (VWAP/TWAP), ~20s (child order refresh), ~10s (round numbers)
- Data: from K02 KO01b inter-arrival time DFT analysis (done for AAPL)

**3. Cross-ticker Generalization** — informative across the full universe
- Metric: I_total(T) variance across ticker liquidity classes
- Universal cadences: those with high I_total across AAPL AND micro-caps
- Ticker-specific cadences: adaptive per K02 KO05 MI_SCORE leaf
- Data: from K04-class analysis on MI leaves across 50+ tickers

Different objectives → different optimal cadences. Since cadences are separate files,
there's no reason to choose — include all three grids. "Adding cadences is cheap."

---

## The Composite Metric: I_total(T)

Naive optimization maximizes MI_per_bin(T) (spectral excess at each period).
But this ignores sample count — long-period cadences have fewer bins per day.

The correct metric is:

```
I_total(T) = MI_per_bin(T) × (session_duration / T)
```

From AAPL inter-arrival time spectral analysis (market hours only):

| Period | Signal (excess-1) | Bins/day | Signal × Bins |
|--------|-------------------|----------|---------------|
| 60s    | 4.2               | 390      | 1,638         |
| 30s    | 1.9               | 780      | 1,482         |
| 15s    | 1.8               | 1,560    | 2,808         |
| 10s    | 1.4               | 2,340    | 3,276         |
| 5s     | 1.1               | 4,680    | 5,148         |
| **2s** | **1.0**           | **11,700** | **11,700** ← peak |
| 1s     | 0.4               | 23,400   | 9,360         |
| 0.5s   | -0.1              | 46,800   | negative      |
| 0.2s   | -0.3              | 117,000  | negative      |

**For AAPL: I_total peaks at 2s with a broad plateau 2-10s.**

**CAVEAT**: This uses spectral excess as a proxy for MI per bin.
Spectral excess measures WHERE algorithmic timing structure EXISTS.
It does NOT measure whether that structure is PREDICTIVE of returns.
The actual MI experiment (step 2) is required to confirm this.
Spectral analysis is the prior. MI experiment is the update.

---

## The Three-Layer Structure (Updated After Full Cadence Chain)

After extending the spectral analysis to 30min, the spectrum reveals TWO REGIMES:

| Range | Excess | Physical Source |
|-------|--------|-----------------|
| sub-500ms | 0.7-0.9x (suppressed) | Exchange batching artifacts |
| 500ms-60s | 1.4-5.2x | Individual algorithm execution clocks |
| 1min-30min | 5.1-57.7x | Session-level institutional structure |

The gradient doesn't plateau at 60s — it explodes:
- 10min: 22.2x (gamma hedging recalc, institutional execution windows)
- 15min: 33.5x (industry-standard reporting interval, broadest synchronization point)
- 30min: 57.7x (risk limit rechecks, half-hourly vol structure, desk rotation)

**This means the cadence grid serves FOUR distinct functions:**

**Layer 0 — Archival (sub-500ms)**
Spectrally suppressed — exchange batching dominates, not market structure.
BUT: sub-500ms K02 bins serve as "pseudo-raw" data for recomputation if K01 ticks are deleted.
Keep for data preservation, not signal extraction. Irrelevant to I_total optimization.

**Layer 1 — Algorithm clocks (500ms-60s)**
Individual execution synchronization. Moderate excess (1.4-5.2x). High sample count.
I_total optimization lives here — the 2s peak, the 2-10s plateau.
Best for intraday pattern ML training.

**Layer 2 — 1-5 minute genuine periodicity**
⚠️ CORRECTED after detrending test (see below).
1-2 minute signals are REAL and survive detrending (AAPL 5.9x at 1min, NVDA 24.2x at 2min).
5 minute is real for high-liquidity names (NVDA 28.5x detrended — genuine periodic structure).
5 minute for moderate-liquidity names (AAPL 4.1x detrended — partially real).

**Layer 3 — Above 5 minutes: archival + industry convention ONLY**
⚠️ CORRECTED: the "enormous excess" at 10-30min was U-shape artifact, not periodic structure.
Detrending results: AAPL 30min 57.7x → 1.4x (-98%). TSLA 30min 283.2x → 9.5x (-97%).
Long cadences stay in the grid but their justification shifts entirely:
- Industry standard (5min OHLCV is universal)
- Archival/recomputation (if K01 deleted, 5-30min K02 serves as pseudo-raw)
- NOT because of spectral excess — that was U-shape artifact

**Layer 4 — Gaming detection (off-round cadences: 3s, 7s, 11s, etc.)**
Deliberately off round-number marks to detect concentration at boundary edges.
Orthogonal to I_total optimization — serves detection, not training.

**CORRECTION NOTE (added after detrending experiment):**
The "two-regime" narrative (execution layer + institutional layer) was partially wrong.
The raw spectrum showed explosive gradient above 5min. Detrending by subtracting the mean
intraday IAT U-shape (open bursty → midday slow → close bursty) revealed that:
- Everything above ~10min: 90-98% U-shape artifact
- 5min: real for NVDA (28.5x), partially real for AAPL (4.1x), benchmark by ticker
- 1-2 min: genuine periodic structure, survives detrending
- Execution regime (1s-30s): REAL, enhanced by detrending

The "unique institutional fingerprints" at 15-30min across NVDA/TSLA/BRK.B also need
re-examination with detrended data — those cross-ticker comparisons were measuring
U-shape differences, not genuine periodic differences.

---

## Cadence Grid Economics (from pipeline model, Observer Exp 11-14)

"Adding cadences is cheap" is TRUE — but only with batch writes (safe=False).

| Write mode | Cost per additional cadence (full 4604-ticker universe) |
|------------|--------------------------------------------------------|
| Production (fsync) | ~14s per cadence |
| Batch (no-fsync) | ~1.2s per cadence |

Full pipeline model (4604 tickers, 10 cadences):

| Mode | Total time | Bottleneck |
|------|-----------|------------|
| Production (fsync) | ~170s (2.8 min) | Writes 82% |
| Batch (no-fsync) | ~35s | Reads 38% |

**The dual write mode is a prerequisite for adaptive cadence grids.**

With fsync: 10 cadences = 170s, 20 cadences = 340s. Each cadence doubles load linearly.
Without fsync: 10 cadences = 35s, 20 cadences = ~51s. Marginal cost falls sharply.

Scale math: 24.2M derived files/day × 3ms fsync floor = ~20 CPU-hours/day in fsync alone.
With safe=False (is_complete crash recovery): ~35 minutes total. The difference IS the pipeline.

Fixing fsync shifts the bottleneck from writes (82%) to reads (38%) — making async I/O (IOCP)
the next optimization target after the safe=False change lands.

**safe=False is not risky.** K02 files are derived. Crash recovery is `is_complete=0` →
delete → recompute from K01. This mechanism already exists. safe=False trusts it explicitly.

---

## Three-Step Research Design

**Step 1: Spectral Analysis** (DONE — 8 tickers, detrending validated)
- K02 KO01b (IAT DFT, market hours, session_filter params in Block 0)
- Raw spectrum: two-regime gradient (execution 1s-30s, apparent institutional 1min-30min)
- Cross-ticker: two-regime structure confirmed universal across AAPL/MSFT/AMD/KO/CHWY/NVDA/TSLA/BRK.B
- Detrending correction: above ~10min is 90-98% U-shape artifact. 1-2min real. 5min real for high-liquidity.
- Durable findings: 500ms floor, execution gradient 1s-2min confirmed, NVDA 5min=28.5x genuine
- Detrend diagnostic: subtract mean intraday IAT-vs-time (U-shape) before FFT
- Script: research/20260327-mktf-format/interarrival_spectrum.py (v3), interarrival_detrend.py

**Step 2: MI Experiment** (UNBLOCKED as of 2026-03-28 — commit ba66b70)
- N² loop in bin_engine.py:bin_mins() fixed: now calls _fused_stats() → fused CUDA kernel
- extract_progressive_stats() → engine.bin_all_stats() → one kernel launch per (column, cadence)
- 0.2ms compute per ticker, 0.9s for full 4,604 universe — I/O-dominated
- Choose target variable (future 5min return? realized volatility? direction?)
- Compute K02 bins at all candidate cadences for 50+ tickers
- Measure MI(features, target) per cadence per ticker
- Plot MI(T) and I_total(T) = MI(T) × bins/day
- The MI peak IS the optimal training cadence
- The MI plateau boundaries define the useful cadence range

**Step 3: Cross-ticker K04** (BLOCKED — needs K02 + MI leaf infrastructure)
- Store MI landscape per ticker as K02 KO05 MI_SCORE leaf
- Run K04 correlation analysis on MI leaves across full universe
- Universal cadences: high I_total across all liquidity classes
- Adaptive cadences: per K02 KO05 recommendation for individual tickers
- Result: base universal grid + per-ticker adaptive grid

---

## Key Engineering Dependency

Steps 2 and 3 require K02 at production throughput (50+ tickers).

**Current blocker**: `bin_engine.py:bin_mins()` is an N² CPU loop
(one GPU kernel launch per bin, ~1,000 launches for 1,000 bins).
`fused_bin_stats.py` does the same work in ONE fused kernel — but these aren't connected.

**Fix**: wire `fused_bin_stats` into `GPUBinEngine.bin_all_stats()`.
Location: `src/winrapids/bin_engine.py:137-141`.
This unblocks the entire MI experiment + cross-ticker research program.

---

## The K02 KO05 MI_SCORE Leaf Design

When step 2 runs, the MI landscape should be stored as an MKTF leaf:

```
leaf_id:    "K02-MI.KO05"
domain:     sufficient_stats (domain_type=4)
upstream:   K01 (for this ticker+day)

Columns:
  candidate_cadence_ms   float32   [n_candidates rows]
  mi_score               float32
  mi_normalized          float32
  ci_low                 float32
  ci_high                float32

Block 0 transform_params (stat_type=MI_SCORE):
  cadence_range_min_ms, cadence_range_max_ms
  n_candidates, n_samples_for_MI_estimate
  target_variable_id (uint8: 0=return_5min, 1=realized_vol, 2=direction)
```

The daemon treats K02-MI.KO05 like any other leaf:
- upstream = K01 → stale if K01 updates (daily recompute)
- schema_fingerprint covers target_variable_id → changing the target invalidates cached MI leaves
- K04 analysis on K02 KO05 MI leaves reveals universal vs ticker-specific cadences

---

## Predicted Cadence Grid (Pre-MI Experiment)

Based on full spectral analysis (0.1s to 30min), four-layer design:

```
ARCHIVAL LAYER (data preservation, not signal):
  200ms   — pseudo-raw if K01 deleted
  500ms   — pseudo-raw boundary

ALGORITHM CLOCK LAYER (ML training, I_total optimization):
  1s      — near I_total peak
  2s      — I_total peak for liquid names (AAPL-class)
  5s      — high I_total, strong generalization
  10s     — round-number detection + solid I_total
  20s     — child order refresh signature
  30s     — transition; still useful training cadence

GAMING DETECTION (off-round, boundary edge concentration):
  3s      — "just off" 5s boundary
  7s      — "just off" 10s boundary
  11s     — "just off" 15s boundary

INSTITUTIONAL RHYTHM LAYER (regime detection, structure):
  1min    — start of second regime (5.1x excess)
  5min    — industry-standard OHLCV unit (10.1x)
  10min   — gamma hedging, institutional execution windows (22.2x)
  15min   — industry reporting standard, strongest broad sync (33.5x)
  30min   — half-hour rhythm, risk limits, desk rotation (57.7x)
```

The long cadences are NOT low-priority. 30min at 57.7x carries more structural information
per bin than any short cadence. It just serves a different function (regime state, not training).

This is a prior. The MI experiment will confirm or revise the algorithm clock layer.
The institutional rhythm layer is justified by spectral evidence alone — 57.7x excess
doesn't need MI confirmation to be included.

---

## Sources

- Naturalist experiments: `research/20260327-mktf-format/interarrival_spectrum.py`
- K02 KO01b campsite note: `kspace-kingdom.md` (this directory)
- Codebase survey: engineering gap in `src/winrapids/bin_engine.py:137-141`
