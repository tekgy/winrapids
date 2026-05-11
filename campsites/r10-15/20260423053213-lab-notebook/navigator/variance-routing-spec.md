# Variance Algorithm Routing Spec

Written: 2026-04-23
Source: scientist validation (message, post-compaction) + verify_variance_spec_claims.py

## Status

The Rust function `scalar_variance_condition_number` and `primitives/precheck/mod.rs`
do not yet exist in the repo (verified 2026-04-23). The oracle/variance/SPEC.md §4.4
routing pseudocode also does not exist. This note preserves the validated routing
logic so it doesn't have to be re-derived.

## The two failure modes of variance estimation

**Failure mode A — const+signal** (`[1e16]*N + [1e16+10]`):
- condition_number = mean²/variance = (1e16)² / (small) ≈ 1.7e31
- range = 10, |mean| = 1e16, range/|mean| = 1e-15
- range/|mean| << sqrt(eps) ≈ 1.49e-8
- Two-pass gives ~389 trillion ULPs. Chan gives 1 ULP.
- Route to: **Chan**

**Failure mode B — large-mean Gaussian** (`Gaussian(1e8, 1)`):
- condition_number ≈ 9.6e15 > 1/eps ≈ 4.5e15
- range ≈ 6.9, |mean| = 1e8, range/|mean| ≈ 6.9e-8
- range/|mean| > sqrt(eps)
- Two-pass gives ~1.6M ULPs (float64 floor for this case). Chan gives ~29M ULPs.
- Route to: **two-pass**

## Correct routing rule (two-step, order matters)

```
let range_ratio = (x_max - x_min) / mean.abs();
if range_ratio < SQRT_EPS {
    // Failure mode A: const+signal. condition_number is irrelevant —
    // it's enormous but that's the wrong discriminator here.
    // Chan is the only algorithm that survives.
    use Chan;
} else if condition_number >= 1.0 / f64::EPSILON {
    // Failure mode B: high kappa, non-const-signal. Kahan two-pass wins.
    use KahanTwoPass;
} else {
    // Well-conditioned. Chan is fine (it's always correct, just slower).
    use Chan;  // or Welford for streaming
}
```

Constants:
- `SQRT_EPS = f64::EPSILON.sqrt()` ≈ 1.49e-8
- `1.0 / f64::EPSILON` ≈ 4.50e15

## Why range_ratio must come FIRST

Failure mode A has condition_number >> 1/eps, so it would be captured by the
second branch and misrouted to Kahan two-pass — which gives 389T ULPs on that
data. The range_ratio check must precede the condition_number check because:

  range_ratio < SQRT_EPS  ⟹  const+signal structure  ⟹  Chan only

The condition_number threshold is only a useful signal when range_ratio is large
enough that "high kappa" is genuinely caused by variance structure, not by the
signal being tiny relative to the mean.

## What needs to land in the repo

1. `crates/tambear/src/primitives/precheck/mod.rs` —
   `scalar_variance_condition_number(data: &[f64]) -> (f64, f64)`
   returning `(condition_number, range_ratio)` so the caller has both
   without re-scanning. Single-pass over data computing mean, min, max,
   and Kahan-compensated sum-of-squares for mean²/variance.

2. The routing logic above in the caller (variance recipe's method dispatch).

3. A test covering failure mode A: assert that the routing sends
   `[1e16]*N + [1e16+10]` to Chan, not to two-pass.

4. oracle/variance/SPEC.md §4.4 routing pseudocode — currently doesn't exist.

## Empirical verification

verify_variance_spec_claims.py covers §3.1-§3.5 behavioral claims but does NOT
yet cover the const+signal failure mode (mode A). The scientist's ULP numbers
are from their own analysis; the range/|mean| discriminator is mathematically
sound and consistent with the verifier's other results.
