# Pitfall: Variance Catastrophic Cancellation (One-Pass Formula)

**Status: CONFIRMED BUG — test `variance_catastrophic_cancellation_exposed` FAILS**

## What broke

The one-pass variance formula `(Σx² - (Σx)²/n) / (n-1)` catastrophically fails
when the mean is large relative to the standard deviation.

**Test data**: 1000 values `1e9, 1e9+1e-6, 1e9+2e-6, ..., 1e9+999e-6`
- True sample variance: ~8.34e-8
- Computed: **-4592.1** (negative! 55 billion times wrong)

**Second test** (`variance_welford_vs_onepass_stress`): 10000 values alternating
`1e8+1` and `1e8-1`. True variance: ~1.0. Computed: **0.0** exactly (total destruction).

## Root cause

The formula computes `Σx²` and `(Σx)²/n` as two large numbers, then subtracts them.
When mean >> std, these numbers agree to many significant digits, and the subtraction
destroys all signal through catastrophic cancellation.

Specifically: with mean = 1e9 and spread = 1e-3:
- `Σx²` ≈ n * (1e9)² = 1e21 (accumulated in 64-bit float)
- `(Σx)²/n` ≈ n * (1e9)² = same

The true difference is ~n * 1e-3 * 1e9 * (something small) ~ 1e3, but the
two terms each have magnitude 1e21. In f64 (53 bits of mantissa = ~15.9 decimal
digits), two 1e21 numbers that agree to 15 digits leave NO significant bits for
the difference.

**The negative result** comes from fp rounding: the accumulated `Σx²` is slightly
LESS than `(Σx)²/n` due to rounding direction during accumulation, producing a
negative "variance."

## The fix: Welford's online algorithm

Welford (1962) maintains running mean and sum of squared deviations:
```
M_1 = x_1;  S_1 = 0
M_k = M_{k-1} + (x_k - M_{k-1}) / k
S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)
variance = S_n / (n - 1)
```

This is numerically stable because `(x_k - M_k)` and `(x_k - M_{k-1})` are
always small deviations from the running mean, never large absolute values.

## Impact

**Any recipe using variance/std_dev/covariance/pearson_r on data with large mean
and small spread returns garbage.** This covers essentially all financial data
(prices are large, movements are small).

## Accumulate decomposition challenge

The current accumulate+gather architecture assumes the accumulate step is a
simple elementwise-then-combine. Welford's algorithm is inherently sequential
(each step depends on the running mean from the previous step) — it's **Kingdom B**.

For the current architecture, the options are:
1. **Two-pass**: Pass 1 computes the mean exactly (sum/count). Pass 2 computes
   Σ(x - mean)² with the mean as a `reference` value. This IS expressible in
   accumulate+gather: a second `AccumulatePass` with `Grouping::All` and the
   reference set to the first-pass mean.
2. **Kahan-compensated one-pass**: Add error compensation terms (Kahan summation).
   Harder to express as accumulate+gather, requires multiple accumulators.
3. **Sort-based**: Sort first, compute from sorted order. Very different architecture.

**Recommended fix**: Two-pass variance. The architecture already supports multiple
passes. The first pass computes mean, the second pass computes sum of squared
deviations. This is correct, parallelizable (each element's contribution to
Σ(xᵢ-μ)² is independent), and expressible in the accumulate+gather framework.

## Files

- **Failing test**: `crates/tambear-primitives/tests/adversarial_baseline.rs`
  - `variance_catastrophic_cancellation_exposed`
  - `variance_welford_vs_onepass_stress`
- **Buggy formula**: `crates/tambear-primitives/src/recipes/mod.rs:183-189`
  (the `variance()` recipe gather expression)
- **Also affected**: `variance_biased()`, `std_dev()`, `covariance()`, `pearson_r()`

## Test that will pass when fixed

Both `variance_catastrophic_cancellation_exposed` and `variance_welford_vs_onepass_stress`
in `adversarial_baseline.rs`.
