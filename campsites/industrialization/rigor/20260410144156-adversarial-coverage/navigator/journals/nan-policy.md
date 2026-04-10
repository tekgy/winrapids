# NaN Policy — A Structural Question

## What the adversarial waves are revealing

Waves 11-15 found 8 NaN-related bugs. The pattern is always the same:
a function that should propagate NaN (invalid input = invalid output) instead
either silently swallows it or produces a wrong answer.

The fixes are correct but ad-hoc. Each fix is a one-off. That doesn't resolve
the underlying policy question.

## The two legitimate NaN contracts

**Contract 1: Pre-filtered (clean data)**
The function documents "input must contain no NaN" as an explicit assumption.
The caller is responsible for filtering. The function can use f64::max/f64::min
freely — they are fast and correct when NaN can't appear.

Example: data_quality.rs operates on `clean.iter()` (already NaN-filtered).
These are correct as-is.

**Contract 2: Propagating (raw data)**
The function accepts raw data that may contain NaN and propagates NaN to the output.
This is what users expect from functions that take user-facing inputs.

Example: `mean`, `variance`, `log_sum_exp`, `MomentStats::merge` — these all
operate on data that might contain NaN and should signal it.

## The structural fix

Add to the codebase a `nan_propagating_min(a: f64, b: f64) -> f64` and
`nan_propagating_max(a: f64, b: f64) -> f64` pair:

```rust
#[inline]
pub fn nan_propagating_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
}

#[inline]  
pub fn nan_propagating_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}
```

These belong in `numerical.rs` or a `utils.rs` alongside the existing math primitives.
They're tiny but the naming makes the intent explicit and searchable.

Then the pattern for folds over raw data:
```rust
// WRONG: swallows NaN
let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

// RIGHT: propagates NaN  
let max = x.iter().cloned().fold(f64::NEG_INFINITY, nan_propagating_max);
```

## What the adversarial agent should target next

Folds over raw (non-filtered) data using f64::max or f64::min:
- clustering.rs:666 — `fold(0.0_f64, f64::max)` on distance scores
- complexity.rs:240-241 — range computation on `cum_dev` (may have NaN if input has NaN)
- descriptive.rs:2055, 2083, 2142, 2145 — `fold(0.0f64, f64::max)` on statistical measures

The filter: only fix the ones where the input array is NOT pre-validated.
Raw user data → fix. Already-clean data → leave.

## Relationship to the Tambear Contract

"Every assumption documented" — the NaN contract is an assumption.
If a function says "input has no NaN", that should be:
1. In the doc comment: `/// Assumes: data contains no NaN.`
2. In debug_assert: `debug_assert!(data.iter().all(|v| !v.is_nan()))`
3. Tested in the adversarial suite

If a function says "propagates NaN", that should be tested with NaN inputs.

This is publication-grade rigor: the assumption is documented, detectable,
and verified. Not implicit, not hoped-for.
