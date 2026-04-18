<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# Technical Analysis — Compositional Decomposition
Written: 2026-04-10

## The Key Insight: TA Signals Are Not New Primitives

Every classical TA signal decomposes entirely into tambear primitives that already exist.
They are COMPOSITIONS, not primitives. The list from the campsite note:

| Signal | Decomposition into tambear primitives |
|--------|--------------------------------------|
| MACD | `ema(prices, 2/13)` - `ema(prices, 2/27)` → difference; `ema(diff, 2/10)` → signal line |
| RSI(14) | `log_returns(prices)` → split + / -; rolling_avg of each via `ema`; ratio formula |
| Bollinger(20, 2σ) | `moving_average(prices, 20)` ± `2 * rolling_std(prices, 20)` |
| Donchian(20) | `rolling_max(prices, 20)` - `rolling_min(prices, 20)` |
| Stochastic(14) | `(close - rolling_min(14)) / (rolling_max(14) - rolling_min(14))` |
| Williams %R | Inverse of Stochastic |
| CCI | `(price - moving_average(20)) / (0.015 * mean_abs_deviation(20))` |
| ATR | `max(high-low, |high-prev_close|, |low-prev_close|)` → rolling_avg |

## What's Missing from the Primitive Catalog

To implement these cleanly, we need:
1. `rolling_max(data, window)` — currently absent from signal_processing.rs
2. `rolling_min(data, window)` — currently absent
3. `rolling_std(data, window)` — currently absent (have `moving_average`, not rolling std)
4. `rolling_mean_abs_deviation(data, window)` — for CCI
5. `ema_period(data, period)` — wrapper: alpha = 2/(period+1) (currently `ema` takes alpha)

Actually items 1-4 are ALL variations of the same theme:
```
rolling_aggregate(data, window, op: fn(&[f64]) -> f64) -> Vec<f64>
```
Where op can be max, min, std, MAD, etc.

This is the REAL missing primitive: **`rolling_aggregate`** — a higher-order function
that applies any aggregation to a sliding window. This is a pure Kingdom B primitive.

## Where These Live Architecturally

TA signals are NOT per-bin statistics. They're temporal filter outputs — the output at
each tick depends on the previous N ticks. This means:

- **Kingdom B** (sequential recurrence, can't be parallelized naively)
- **K02 leaves** if applied within a bin (bin → scalar features)
- **K03 cross-cadence** for MACD (inherently compares two cadences)

RSI, Bollinger, Stochastic computed WITHIN a bin from tick data:
These are K02 leaves. The bin's tick prices feed in, summary stats come out.
These are just more elaborate forms of what fintek already does.

## The Fintek Family Question

Where do TA signals live in the fintek taxonomy?
- They don't exist yet
- They'd be a new family: say Family 25 (momentum/trend indicators)
- Each signal is a leaf: RSI leaf, MACD leaf, Bollinger leaf, etc.

But wait — are TA signals actually worth farming? The system philosophy is:
"farm every measurable signal, let V columns carry confidence."

TA signals ARE measurable and carry real information about price structure.
They're widely used by market participants (so other actors react to them).
The market IS a temporospatial system — TA signals describe its state.

So yes: TA leaves should exist in fintek. Family 25 or wherever.

## Implementation Plan

1. Add to `signal_processing.rs`:
   ```rust
   pub fn rolling_max(data: &[f64], window: usize) -> Vec<f64>
   pub fn rolling_min(data: &[f64], window: usize) -> Vec<f64>
   pub fn rolling_std(data: &[f64], window: usize, ddof: usize) -> Vec<f64>
   pub fn rolling_mean_abs_deviation(data: &[f64], window: usize) -> Vec<f64>
   pub fn ema_period(data: &[f64], period: usize) -> Vec<f64>  // alpha = 2/(period+1)
   ```

2. Add to `time_series.rs` or a new `technical.rs`:
   ```rust
   pub fn macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>)
   pub fn rsi(prices: &[f64], period: usize) -> Vec<f64>
   pub fn bollinger_bands(prices: &[f64], window: usize, n_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>)
   pub fn donchian_channels(prices: &[f64], window: usize) -> (Vec<f64>, Vec<f64>)
   pub fn stochastic(close: &[f64], high: &[f64], low: &[f64], k_period: usize) -> Vec<f64>
   pub fn williams_r(close: &[f64], high: &[f64], low: &[f64], period: usize) -> Vec<f64>
   pub fn cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64>
   pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64>
   ```

3. Fintek family 25 bridges: thin wrappers that take bin tick data, apply TA over the
   bin's price sequence, return last value (or last few) as the leaf output.

## The Deeper Question: Are These K02 or K01?

K01 = per-tick (raw tick level). K02 = per-bin (aggregated features of a bin's ticks).

TA signals computed from a bin's tick prices are K02 (the BIN is the observation, 
the output is a few scalars per bin).

But TA signals computed as CONTINUOUS time series across bins would be K01:
"what is the RSI of the price series at this moment?"

The per-bin version (K02) makes sense for classification/feature extraction.
The continuous version (K01) makes sense for trading signal generation.

Fintek appears to be primarily K02. So the TA leaves would compute TA signals
from within-bin price sequences and report the terminal value or a few statistics
of the signal within the bin.

## The Most Interesting Signal: MACD as Kingdom B

MACD = EMA(fast) - EMA(slow). Two competing exponential memory processes.
The signal line = EMA(MACD). Three EMA computations in sequence.

EMA is a first-order linear recurrence: s_t = α*x_t + (1-α)*s_{t-1}
This is Kingdom B (Fock boundary: state carries forward). Cannot be parallelized
naively — but CAN be computed via the parallel prefix scan trick using the
semigroup (a, b) * (c, d) = (a*c, a*d + b) for the affine map structure.

This is the accumulate+gather connection: EMA IS a Kingdom A primitive when
expressed as a parallel prefix over affine maps! The Fock boundary dissolves.

This might be worth a garden note — the observation that all linear IIR filters
have this parallel-scan structure and are therefore genuinely Kingdom A.


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

