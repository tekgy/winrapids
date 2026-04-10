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
