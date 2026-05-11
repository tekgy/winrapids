# Scientist — Phase A Oracle Harness

**Date**: 2026-05-10
**Status**: Phase A oracle infrastructure complete. Waiting for pathmaker's Phase A implementation.

## What shipped

### Corpus generators
- `R:\tambear\oracle\expm1\generate_corpus.py` — 287 entries × 42 categories
- `R:\tambear\oracle\log1p\generate_corpus.py` — 366 entries × 45 categories

### Corpus data (generated)
- `R:\tambear\oracle\expm1\data\generated\canonical_landmarks\corpus.json`
- `R:\tambear\oracle\log1p\data\generated\canonical_landmarks\corpus.json`

### README files
- `R:\tambear\oracle\expm1\README.md` — harness spec + what ships when Phase A lands
- `R:\tambear\oracle\log1p\README.md` — harness spec + near-minus-one regime documented

### Rust harness extensions (in big_float_vs_mpmath.rs)
- `discover_libm_expm1_distribution` — discovery-tier, runs against platform libm
- `libm_expm1_identity_landmarks` — identity landmarks
- `verify_tambear_expm1_zero_ulp` — verification-tier, #[ignore] until Phase A ships
- `discover_libm_log1p_distribution` — discovery-tier, runs against platform libm
- `libm_log1p_identity_landmarks` — identity landmarks
- `verify_tambear_log1p_zero_ulp` — verification-tier, #[ignore] until Phase A ships

## Empirical findings (MSVC libm baseline, 2026-05-10)

### expm1 (f64::exp_m1)
- **74.2% (213/287) at 0 ULP** against mpmath
- **16.4% (47/287) at 1 ULP** — within IEEE tolerance
- **0.3% (1 entry) at 2 ULP** — one point in near_zero_positive category
- **26 very-small entries** pass abs-check (expm1(x) ≈ x regime)
- **0 NaN-policy mismatches, 0 Inf-policy mismatches**
- Max ULP: **2 ULP** at one near_zero_positive point

**Key finding**: MSVC libm expm1 is 74-91% correct, with max 2 ULP. This is
the *platform baseline* — tambear-native expm1 must hit **0 ULP everywhere**.
The single 2-ULP point in near_zero_positive is a real failure mode: it's exactly
the precision-critical ring where expm1 should be getting the answer exactly right.
This is the bug tambear-native expm1 is supposed to fix.

### log1p (f64::ln_1p)
- **80.9% (296/366) at 0 ULP** against mpmath
- **12.0% (44/366) at 1 ULP** — within IEEE tolerance
- **0 entries beyond 1 ULP** — MSVC log1p is max-1-ULP everywhere
- **26 very-small entries** pass abs-check
- **0 NaN-policy mismatches, 0 Inf-policy mismatches**
- Max ULP: **1 ULP** across all categories

**Key finding**: MSVC libm log1p is surprisingly good — max 1 ULP everywhere
including the near_zero regime. Tambear-native log1p needs to match this
AND hit 0 ULP (i.e., be correctly-rounded everywhere the platform libm
produces 1-ULP off answers). The near_zero_positive/negative band currently
has 1-ULP answers; tambear must hit 0 ULP there.

## Near-zero regime characterization

Both expm1 and log1p use `auto_within` abs-check for the `very_small` category
(|x| < 1e-15 where the function value ≈ input). This is the correct harness
decision: at x = 2^-1022, expm1(x) = x to machine precision, and ULP comparison
would spuriously flag tiny absolute errors.

The precision-critical `near_zero_*` category (|x| ∈ [2^-52, 0.5]) uses standard
ULP comparison — the gold is non-trivially away from zero there, so ULP is the
right metric. This is where the value of expm1/log1p over direct exp(x)-1 shows up.

## What Phase A must deliver to pass verification-tier

1. Remove `#[ignore]` from `verify_tambear_expm1_zero_ulp` and fill in the
   `tambear::recipes::elementary::expm1::compute` call.
2. Remove `#[ignore]` from `verify_tambear_log1p_zero_ulp` and fill in the
   `tambear::recipes::elementary::log1p::compute` call.
3. Both tests must pass at **0 ULP** across all non-NaN, non-domain-error entries.
4. Cross-precision check: compute at p=200, round to p=53, verify ≤1 ULP vs libm.

The path for phase A → verification: pathmaker ships the implementation;
scientist updates the `#[ignore]` tests; runs `cargo test --test big_float_vs_mpmath
verify_tambear_expm1_zero_ulp verify_tambear_log1p_zero_ulp -- --ignored`.

## Structural observation: expm1's 2-ULP point

The single 2-ULP failure in MSVC expm1 near_zero_positive is significant.
The near_zero_positive category spans |x| ∈ [2^-52, 0.5] — log-spaced.
The fact that one point in this ring fails at 2 ULP (rather than 0-1 ULP
like the rest) suggests either:
(a) a boundary case in the platform's polynomial evaluation domain, or
(b) a rounding artifact at a specific floating-point magnitude.

This should be investigated when the tambear-native implementation arrives —
if tambear also produces 2 ULP at that point, the polynomial form is wrong
there. If tambear produces 0 ULP, we've found a genuine MSVC bug.
File this in `oracle/expm1/disagreements/` when pathmaker ships.
