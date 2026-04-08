# Naive Formula Bug Class — Full Codebase Sweep

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: Complete

---

## The Bug Class

The naive formula `Var(x) = E[x²] - E[x]²` (and its OLS cousin `denom = n·Σx² - (Σx)²`) suffers catastrophic cancellation when data has a large constant offset. The subtraction of two nearly-equal large numbers destroys significant digits.

**Threshold**: offset ≈ 1e8 for f64 (loses ~16 - 2*8 = 0 significant digits).

**Fix pattern**: Center by mean before computing sums of squares.

---

## All Instances Found

### HIGH Severity (production code, real data)

| # | File | Line | Formula | Notes |
|---|------|------|---------|-------|
| 1 | `hash_scatter.rs` | 193 | `(sum_sq - sum²/n) / (n-1)` | GPU scatter variance derivation |
| 2 | `intermediates.rs` | — | m2 from naive formula | MomentStats construction |
| 3 | `robust.rs` | 381 | `denom = n*sxx - sx*sx` | LTS ols_subset, SINGULAR at 1e8 |

### MEDIUM Severity

| # | File | Line | Formula | Notes |
|---|------|------|---------|-------|
| 4 | `tambear-py/src/lib.rs` | 101 | `(sum_sqs[g] - sums[g]²/c) / (c-1)` | Python binding, same raw accumulators as #1 |

### LOW Severity

| # | File | Line | Formula | Notes |
|---|------|------|---------|-------|
| 5 | `complexity.rs` | 348 | `denom = n*sxx - sx*sx` | x=0..n-1 (integer), protects denom; numerator can still cancel |
| 6 | `main.rs` | 98 | `(sum_sqs[g] - sums[g]²/c) / (c-1)` | Test reference — validates against equally-buggy formula |

### CORRECT (not a bug)

| File | Line | Notes |
|------|------|-------|
| `complexity.rs` | 254 | `ols_slope` — correctly centers: `dx = x[i] - mx` before computing sums |

---

## Grep Patterns Used

```
sxx - sx * sx
n * sxy - sx * sy
sum_sqs.*sums.*sums
sum_sq.*- sum
```

---

## Structural Observation

The root cause is that the GPU scatter primitive accumulates `{count, sum, sum_sq}` — raw moments. Variance must then be derived as `(sum_sq - sum²/count) / (count-1)`, which IS the naive formula.

The proper fix is structural: accumulate `{count, sum, m2}` where `m2 = Σ(xi - x̄)²` using Welford's online algorithm or the Chan-Golub-LeVeque parallel merge formula. This makes the derivation step `variance = m2 / (count-1)`, which is numerically stable regardless of offset.

The `MomentStats` type already has an `m2` field and uses the CLG merge. The issue is that `hash_scatter.rs` and the Python bindings derive from raw accumulators instead of using `MomentStats`.

---

## Priority

Fix #1-#3 (HIGH) and #4 (MEDIUM). #5-#6 (LOW) can wait.
