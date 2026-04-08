# Lab Notebook 010: Fintek Leaf Decomposition into Tambear Primitives

**Date**: 2026-04-01
**Authors**: Tekgy + Claude
**Status**: Active
**Depends on**: 001 (accumulate unification), MSR principle

---

## Context & Motivation

Fintek has 121 Rust leaves (206 Python). Each leaf is a separate `execute()` function with its own for-loops, allocations, and output construction. The hypothesis: EVERY leaf decomposes into tambear's `accumulate + gather + fused_expr` primitives, enabling compilation from `spec.toml → .tbs → GPU pipeline`.

The delayed collapse / MSR principle applies: each pipeline stage carries exactly the minimum sufficient representation for its most demanding downstream computation, collapsing only at the DO output.

**Assumption**: K01P01 ingest is done before our system. We take raw tick data (price, size, timestamps) and compute everything from K01P02 onward.

---

## Leaf 1: LogTransform (K01P02C02R01) — TRIVIAL

### Decomposition
```
Input: price[], size[], notional[]
Operation: element-wise ln()
```

| Output | Tambear | Primitive |
|--------|---------|-----------|
| ln(price) | `fused_expr("log(v)")` on price | Element-wise |
| ln(size) | `fused_expr("log(v)")` on size | Element-wise |
| ln(notional) | `fused_expr("log(v)")` on notional | Element-wise |

**Primitives needed**: fused_expr only. ALREADY HAVE IT.
**Kernel count**: 0 separate kernels — all three fuse into whatever consumes them downstream.
**MSR**: 1 f64 per element (identity — no compression).

---

## Leaf 2: DeltaValue (K01P02C03R02F01) — TRIVIAL

### Decomposition
```
Input: 7 channels × 5 lags = 35 outputs
Operation: x[i] - x[i-lag] for lag in {1, 2, 3, 5, 10}
```

| Output | Tambear | Primitive |
|--------|---------|-----------|
| delta_lag_N | `gather(Direct(offset=-N))` → `fused_expr("v - gathered")` | Gather + fused_expr |

**Primitives needed**: gather + fused_expr. ALREADY HAVE BOTH.
**Kernel count**: 1 fused kernel (all 35 outputs share the data loading).
**MSR**: 1 f64 per element per lag (no compression within a lag).

---

## Leaf 3: OHLCV (K02P01C01) — THE FOUNDATION

### Before (current Rust)
200 lines. For-loop over bins. 9 separate vectors allocated. 10 emission vectors. Sequential.

### Decomposition
```
Input: price[], size[], timestamps[], bin_boundaries[]
Grouping: ByKey(bin_id) — all ticks scattered to their bin
```

| Output | Expression (phi) | Operator | Primitive |
|--------|-----------------|----------|-----------|
| open | `price[bin_start]` | — | **gather**(price, bin_starts) — O(n_bins), not O(n_ticks) |
| high | `identity` | **Max** | scatter + Max op (CAS-loop atomicMax, NEW) |
| low | `identity` | **Min** | scatter + Min op (CAS-loop atomicMin, NEW) |
| close | `price[bin_end - 1]` | — | **gather**(price, bin_ends-1) — O(n_bins), not O(n_ticks) |
| volume | `v` (size column) | **Add** | scatter_phi("v") |
| count | `1.0` | **Add** | scatter_phi("1.0") |
| notional | `price * size` | **Add** | scatter_phi("price * size") |
| vwap | `notional / volume` | — | fused_expr on two DOs |
| real_var | `log(p[i+1]/p[i])^2` | **Add** | scatter_phi on log-returns |

### Emissions (10 E columns)
ALL are `fused_expr` on DO columns. Zero additional GPU work — they fuse into the extract step.

```
E01: abs((close - open) / open) > 0.0
E02: abs((close - open) / open) > 0.01
E03: abs((close - open) / open) > 0.02
E04: abs((close - open) / open) > 0.05
E05: volume > 1000
E06: volume > 10000
E07: volume > 100000
E08: count > 100
E09: (high - low) / open > 0.005
E10: real_var > 0.001
```

### What we need that we don't have yet

| Gap | Description | Effort |
|-----|-------------|--------|
| **First op** | `combine(a, b) = a` — keep first element per group | LOW — trivially associative |
| **Last op** | `combine(a, b) = b` — keep last element per group | LOW — trivially associative |
| **Max op for scatter** | `atomicMax` for f64 in scatter | LOW — CAS loop like atomicAdd |
| **Min op for scatter** | `atomicMin` for f64 in scatter | LOW — CAS loop |
| **Multi-column scatter** | Scatter from multiple input columns in one pass | MEDIUM — extend scatter_multi_phi |
| **Sequential log-return** | `log(p[i+1]/p[i])^2` needs adjacent elements | Gather(offset=1) + fused_expr |

### The tambear OHLCV

```tbs
read("@K01P01", columns=["price", "size", "timestamp"])
  .scatter_bins(cadence)
  .accumulate_multi(
      ByKey(bin),
      open     = (price, First),
      high     = (price, Max),
      low      = (price, Min),
      close    = (price, Last),
      volume   = (size, Add),
      count    = ("1.0", Add),
      notional = ("price * size", Add),
  )
  .derive(
      vwap     = "notional / volume",
      real_var = accumulate(ByKey(bin), "log(price[i+1]/price[i])^2", Add),
  )
  .emit(
      price_changed = "abs((close - open) / open) > 0.0",
      gap_1pct      = "abs((close - open) / open) > 0.01",
      gap_2pct      = "abs((close - open) / open) > 0.02",
      gap_5pct      = "abs((close - open) / open) > 0.05",
      vol_1k        = "volume > 1000.0",
      vol_10k       = "volume > 10000.0",
      vol_100k      = "volume > 100000.0",
      count_high    = "count > 100.0",
      range_wide    = "(high - low) / open > 0.005",
      rv_spike      = "real_var > 0.001",
  )
  .write_mktf()
```

**ONE pass through tick data. ONE kernel launch. 9 DOs + 10 Es.** Currently: 200 lines of Rust with sequential for-loops.

### MSR Analysis (delayed collapse)

```
Raw ticks (N ≈ 598K):     MSR = full price/size/timestamp per tick
  ↓ scatter to bins
Per-bin accumulators:      MSR = {first, max, min, last, sum_size, count, sum_notional} = 7 f64 per bin
  ↓ extract
DO columns:                MSR = 9 f64 per bin (7 + vwap + real_var)
  ↓ fused_expr
E columns:                 MSR = 10 f32 per bin (boolean thresholds)
  ↓ write
MKTF:                      Terminal. Collapsed.
```

Collapse ratio: 598K ticks × 3 columns → ~1100 bins × 19 columns. ~1000:1 compression through the pipeline. The MSR at each stage is the minimum needed for the next.

---

## Leaf 4: Returns (K02P01C05)

### Decomposition
```
Input: price[], size[], bin_boundaries[], OHLCV outputs (open, close, high, low)
```

| Output | Depends on | Tambear |
|--------|-----------|---------|
| open_return | open[i], close[i-1] | `fused_expr("(open - prev_close) / prev_close")` where prev_close = `Prefix(Last)` |
| close_return | open[i], close[i] | `fused_expr("(close - open) / open")` — pure DO arithmetic |
| high_low_range | high[i], low[i], open[i] | `fused_expr("(high - low) / open")` — pure DO arithmetic |
| log_return | open[i], close[i] | `fused_expr("log(close / open)")` — pure DO arithmetic |
| abs_return | open[i], close[i] | `fused_expr("abs((close - open) / open)")` — pure DO arithmetic |
| signed_volume | price[], size[] per bin | `accumulate(ByKey(bin), "sign(delta_price) * size", Add)` |

### Key insight
5 of 6 Returns outputs are PURE ARITHMETIC ON OHLCV OUTPUTS. They don't touch raw ticks at all. They're fused_exprs on the DO columns from Leaf 3. Only signed_volume needs to touch raw ticks (it needs per-tick price direction × size).

**Cross-leaf sharing**: Returns INHERITS open, close, high, low from OHLCV via TamSession. Zero recomputation.

**The prev_close dependency** (open_return needs previous bin's close) is a `Prefix(forward)` scan across bins — carry the close value from bin i to bin i+1. One scan pass. Already have the scan engine.

---

## Patterns Emerging

1. **K01P02 pointwise leaves**: pure fused_expr. Trivial. All fuse into consumers.
2. **K02 bin leaves**: accumulate(ByKey(bin), expr, op). Our scatter primitives. ONE pass per leaf.
3. **Cross-bin dependencies**: accumulate(Prefix, carry_state). Our scan. Rare but important (prev_close, cumulative returns).
4. **DO→DO leaves**: pure fused_expr on upstream DOs. Zero tick access. Inherits via TamSession.
5. **E columns**: fused_expr on DOs. Always fuse. Zero additional cost.

---

## Mega-Kernel Architecture (Navigator Analysis, 2026-04-01)

### Column-Partition Model
- 1024 threads per block, each owns ~11 f64 accumulators in registers
- All threads stream ALL ticks in lockstep (broadcast from shared/L1)
- No atomics — each accumulator owned by exactly one thread
- Max/Min are in-register comparisons, not CAS loops
- Register budget: 1024 × 11 × 8 bytes = 90KB / 256KB = 35% ✓

### Two-Pass Architecture
```
Pass 1: ~90 register-accumulation leaves × 31 cadences
        ONE pass through 598K ticks. Column-partition model.
        Covers: OHLCV, counts, distribution stats, returns,
        volatility, entropy, basic indicators.

Pass 2: ~30 scan-based leaves × 31 cadences
        Segmented prefix scans over tick stream.
        Covers: DFA, Hurst R/S, MF-DFA, cumulative features,
        AR models, Kalman, anything needing running state.

K03:    Cross-cadence features on K02 outputs still in L2 (~1.5MB).
        No reload from disk.

MKTF:   The collapse. Everything writes to disk here.
```

Current fintek: 3,751 separate compute units.
Tambear: 2 passes + K03 + write. Same results. ~1000× fewer tick reads.

### K03 L2 Feasibility
K03 reads specific K02 outputs: 5 cadences × 121 leaves × ~100 bins × 3 outputs = ~1.5MB.
Blackwell L2: 128MB. Fits easily. No device-to-host-to-device roundtrip.

### Scan Boundary
~20-30 leaves need O(n_ticks) prefix state that can't fit in registers.
Clean separation: exclude from mega-kernel, run in pass 2.
The scan pass is still massively parallel ACROSS bins — each bin's prefix scan is independent.

---

## Open Questions

1. ~~First/Last ops for scatter~~ → RESOLVED: O/C are gathers from bin_starts/bin_ends. Not scatter.: trivially associative but need GPU atomics that preserve order. If data is pre-sorted by timestamp (it is in fintek), First = min-index, Last = max-index. atomicMin/Max on index → gather value at that index.

2. Multi-column scatter: current scatter_multi_phi scatters ONE value column with multiple expressions. OHLCV needs to scatter MULTIPLE value columns (price AND size) with different ops per column. Extension needed.

3. realized_var needs adjacent-element access within each bin. That's `gather(offset=1)` WITHIN each bin — a segmented gather. Or: precompute log-returns at K01 level (they're already there as K01P02C03) and scatter the squared values.

4. The 31-cadence loop: same leaf runs at 31 different cadences. Same primitives, different bin boundaries. Tam compiles the kernel ONCE, dispatches 31 times with different boundary arrays. 31 launches, not 31 compilations.
