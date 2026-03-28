# Market Agency Layers — Scout Synthesis Note
## 2026-03-27
## Unassigned — following the thread from detrending correction

---

## The Provocation

The detrending correction revealed that 90-98% of the raw spectral excess above 10 minutes
was the intraday U-shape (open bursty → midday slow → close bursty) — not genuine periodicity.

The obvious read: "we were wrong about the institutional regime, now corrected."

The more interesting read: **the market is a fundamentally different kind of object
at different time scales.** The FFT was telling us something true and something false
at the same time, and detrending separated them.

---

## Four Layers of Market Agency

The spectral analysis (K02 KO01b, 8 tickers, 0.1s to 30min) — with detrending correction —
reveals four distinct regimes, each with different physics and different appropriate representations:

### Layer A: Exchange Physics (sub-500ms)
**Signal**: Suppressed (0.7-0.9x, below baseline)
**Agent**: The exchange itself
**Physics**: Batching, aggregation, matching-engine clock cycles
**What's periodic**: Nothing useful. Exchange clock is too fast and too regular to
leave MI-informative structure at this scale.
**Right representation**: Raw ticks (K01 archival). No periodic decomposition needed.
**Wrong tool**: FFT (nothing to find), MI optimization (signal suppressed)

### Layer B: Execution Clocks (500ms–2min)
**Signal**: Real periodic excess (1.4-5.2x raw, enhanced after detrending)
**Agent**: Individual algorithms
**Physics**: Child order refresh cycles, round-number timing, VWAP/TWAP ticks
**What's periodic**: Genuinely periodic. Individual algorithms have clocks.
60s = VWAP/TWAP. 20s = child order refresh. 10s = round-number concentration.
**Right representation**: Binned stats (K02) + DFT of inter-arrival times (K02 KO01b)
**I_total peak**: 2s (AAPL-class). The MI experiment lives here.

### Layer C: Strategy Coordination (2–5min)
**Signal**: Genuine, ticker-specific, and the cross-ticker DISCRIMINATIVE PEAK
**Agent**: Strategy-level decision logic — the algorithmic ecosystem specific to each ticker
**Physics**: 5min option gamma hedging recalc, execution window completions, intraday signal refresh
**What's periodic**: Real and ticker-differentiated. Detrended CV at 5min = 0.93, exceeding
every execution-regime cadence (0.31–0.64 at 1s–1min). Genuine regime change at 5min.

**Detrended 5min fingerprint (10 tickers):**
- NVDA 28.5x, TSLA 16.0x, KO 9.6x, MSFT 9.5x, META 4.6x
- AAPL 4.1x, AMD 3.7x, BRK.B 3.6x, JNJ 2.7x, CHWY 1.8x
- 8/10 tickers peak at 5min after detrending

**Not purely liquidity-ordered**: KO (9.6x) and MSFT (9.5x) are similar liquidity but rank
ahead of AAPL (4.1x). Sector and strategy composition matter independently of liquidity.
Options-gamma hypothesis fits NVDA/TSLA well. KO's 9.6x signal is harder to explain with
gamma hedging — consumer staples have different options market structure.

**Right representation**: K02 bins + K02 KO01b + K02 KO05 MI_SCORE leaves
**Cross-ticker**: This is where algorithmic ecosystems diverge. Execution regime variation
reflects liquidity class differences. 5min variation reflects *what algorithms are doing*
— strategy type, sector dynamics, options exposure.
**Phase analysis: RETRACTED — phase not stable at daily scale**
(naturalist, interarrival_phase_bootstrap.py + interarrival_phase_stability.py)

⚠️ CORRECTION (fourth in this thread): The three-clock cluster finding did not survive validation.

**Bootstrap result**: 95% CI per ticker = ±170° (effectively the full circle). Bootstrap R ≈ 0.04.
Cluster assignment stability: 28-40% (barely above 33% random for k=3 clusters).

**Split-day stability**: 6 × 65-minute segments, mean within-ticker R = 0.399 (weak).
AAPL phases by segment: -29°, -51°, -34°, +96°, +71°, +137° — spread: 172°.
NVDA phases by segment: +179°, -128°, +72°, -12°, +67°, +91° — spread: 169°.

**The AAPL-NVDA 0.9° coincidence was a cancellation artifact.** Segment-by-segment
AAPL-NVDA angular distances: [151°, 77°, 106°, 108°, 4°, 46°] — mean 82°. The full-day
vector average happened to cancel to near-zero difference, masking true intra-day drift.

**What this means for Layer C**: The 5min algorithms are real (excess is robust), but they
are NOT wall-clock locked. They oscillate at ~5min period with drifting phase — likely
triggered by order flow surges or events rather than fixed timestamps. The excess itself
is non-stationary within day (AAPL: 0.8x→85.1x across segments, NVDA: 28.5x→88.2x).

**What survives**:
- 5min spectral excess is real and robust (4-28x detrended, across 10 tickers) ✓
- Cross-ticker amplitude CV = 0.93 — tickers differ in HOW MUCH 5min periodicity ✓
- The amplitude fingerprint (NVDA 28.5x, CHWY 1.8x) is the durable cross-ticker feature ✓

**What does not survive**:
- Stable phase at 5min ✗ — drifts within a single day
- Three strategy clocks ✗ — single-realization artifact
- AAPL-NVDA synchronization ✗ — accidental in full-day vector average
- K02 KO05 PHASE as daily leaf ✗ — phase too unstable for a single daily value

**K02 KO05 PHASE design note**: The (cos, sin) encoding is technically correct IF a stable
phase source existed. Daily-granularity phase is not that source. Intraday phase within
a 65-min segment may be more stable (within-segment R = 0.399 is weak but nonzero).
Future investigation: shorter-window phase estimates, conditioned on high-amplitude segments.

### Layer D: Daily Structure (above 5min)
**Signal**: Not periodic — smooth trend
**Agent**: Market session itself, institutional risk schedules, desk rotation
**Physics**: Intraday U-shape (open bursty → midday slow → close bursty) is real,
but it's a smooth daily rhythm — NOT a periodic signal at 15-30min intervals.
The raw FFT decomposed the U-shape into many harmonics and made it look like
giant excess at 10-30min. Detrending revealed: it was all aliases of the same trend.
**What's there**: Real structured non-stationarity. The U-shape is a genuine feature.
**Wrong representation**: FFT (decomposes smooth trends into spurious harmonics)
**Right representation**: Wavelet decomposition (K02 KO04) — can represent both smooth
trends AND localized periodicities in a single framework without the FFT's aliasing problem

---

## The Representation-Layer Match

The detrending correction isn't just a fix — it's revealing that each layer of market agency
wants a different mathematical representation:

| Layer | Agent | Right tool | K-space type | Cross-ticker CV |
|-------|-------|------------|--------------|-----------------|
| Exchange (sub-500ms) | Exchange | Raw ticks | K01 (archival) | — |
| Execution (500ms–2min) | Algorithms | DFT of IAT | K02 KO01b | 0.31–0.64 |
| Strategy (2–5min) | Strategy ecosystems | DFT + MI | K02 KO01b + K02 KO05 MI | **0.93** ← peak |
| Daily structure (5min+) | Session/institution | Wavelet | K02 KO04 | collapsed (U-shape artifact) |

The 5min CV peak (0.93) is the strongest discriminative signal found across all cadences.
This is the scale where tickers are most distinguishable from each other by periodic structure.

The cadence grid covers all four layers — but we've been optimizing for Layer B/C only
(the MI experiment, I_total peak at 2s). The long cadences (5min–30min) stay in the grid
for archival and industry convention, as documented. But their full value might only be
accessible via K02 KO04, not K02 KO01b.

**Implication for kspace-kingdom.md**: The case for K02 KO04 isn't just "completeness."
It's the *only* correct representation for Layer D. DFT cannot see smooth trend structure
without aliasing into spurious harmonics. Wavelets are what Layer D requires.

---

## Why the FFT Lies at Long Periods

The intraday U-shape in inter-arrival times:
- Open (9:30): IAT ≈ very small (bursty trades)
- Midday (12:00): IAT ≈ large (slow)
- Close (15:30): IAT ≈ small again (bursty)

A smooth U on a [0, 23400s] window has a Fourier series dominated by:
- DC component (baseline mean IAT)
- Low-frequency harmonics (period ≈ session length = 23400s, 11700s, 7800s...)
- These land as giant excess in any FFT bin near those periods

A 30min period (1800s) is 13× per session. The U-shape's smooth trend bleeds into
every harmonic that's not resolved away from it. The detrending step works precisely
because it subtracts this smooth baseline before the FFT, so what remains are genuine
above-trend periodic deviations.

The 57.7x raw excess at 30min → 1.4x detrended (for AAPL) is not a small correction.
It's almost entirely artifact. The raw spectrum was measuring the derivative of the U-shape
at 30min intervals, not a 30-minute clock.

**Takeaway for analysis methodology**: Any FFT of non-stationary data will show spurious
periodicity at periods comparable to the trend's own time scale. Detrending is mandatory,
not optional, for periods above ~5× the stationarity window.

---

## The Cross-Ticker Question This Opens

The cross-ticker fingerprint comparisons at 15-30min (NVDA/TSLA/BRK.B) were made with
raw spectrum data. They showed "unique institutional fingerprints" — NVDA peaked at 10min
while TSLA peaked at 20min, etc.

**With detrending**: Those differences were almost certainly measuring differences in
the *shape* of each ticker's daily U-shape — not differences in genuine periodic structure.
NVDA and TSLA might have different open/close burstiness ratios, creating different
U-shape harmonics in the raw FFT.

The re-examination with detrended data (flagged in cadence-optimization-framework.md) is
more important than it sounds. It's not just "verify the fingerprints are still there."
It's testing whether Layer D produces *any* ticker-specific periodic structure, or whether
all cross-ticker differences at 15-30min are purely U-shape variation.

**Hypothesis (before the data)**: After detrending, cross-ticker differences at 15-30min
will largely collapse — they'll all show low excess (~1-3x), and the "unique fingerprints"
will mostly disappear. The real cross-ticker differentiation will remain in Layer B (500ms–2min),
which IS genuinely ticker-specific by liquidity class and algo composition.

---

## A Note on What This Means for the K02 KO04 Design

Naturalist's open question from kspace-kingdom.md #1: "Should K-space files carry the
Nyquist frequency explicitly, or is it derivable from source cadence?"

For K02 KO01b: derivable is fine (Nyquist = 1/(2×cadence)).
For K02 KO04: the question is different. Wavelets don't have a single Nyquist — they have
a *scale range*. The relevant parameters are:
- Finest scale (= cadence × 2, to see sub-cadence structure): auto-derivable
- Coarsest scale (= session duration / 2, to see half-day structure): needs explicit storage

The coarsest scale is important: if we want K02 KO04 to capture the daily U-shape structure,
the coarsest wavelet scale must be ≥ half the session duration (~3 hours). This means
the Block 0 domain descriptor's `scale_max` field in the wavelet transform_params
(currently `[12:16]`) needs to be session-length-aware, not just cadence-aware.

**Suggested addition to kspace-kingdom.md wavelet params**:
```
[12:16]  scale_max     float32   coarsest scale in seconds
                                 Recommended: session_duration_s / 2 = 3 hours
                                 This ensures Layer D (daily trend) is captured
```

---

## Summary: What the Detrending Correction Actually Taught

1. **The market is not uniformly "periodic" across scales.** It's periodic in Layer B/C
   and trending/non-stationary in Layer D. These require different analysis tools.

2. **The FFT is Layer B/C's native language, not Layer D's.** FFT on Layer D data produces
   artifacts, not insight. Detrending fixes this for DFT, but wavelets are the proper native
   language for Layer D.

3. **The cadence grid + K-space kingdom together cover all four layers:**
   - K01: Layer A (raw ticks)
   - K02 + K02 KO01b: Layer B/C (execution + strategy)
   - K02 KO04: Layer D (daily trend structure)

4. **The cross-ticker "institutional fingerprints" are probably U-shape differences,**
   not genuine periodic differences. The real ticker-specific differentiation lives in
   Layer B/C — liquidity class, algo composition, round-number behavior.

5. **The coarsest wavelet scale needs to be session-length-aware** (≥ 3 hours for full-day
   daily trend capture). This is a concrete addition to the K02 KO04 domain descriptor design.

---

## Sources

- Spectral analysis findings: `cadence-optimization-framework.md` (this directory)
- K-space kingdom design: `kspace-kingdom.md` (this directory)
- Detrending methodology: `research/20260327-mktf-format/interarrival_detrend.py`
