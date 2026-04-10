# Adversarial Rigor Gauntlet — Navigator Session Log

## What happened

The adversarial agent wrote `tests/adversarial_rigor_gauntlet.rs` (67 tests) that caught 8 real bugs. Fixed all 8. All 67 green.

## Bugs found and fixed

### tie_count — assumed sorted input
`tie_count(&[1.0, 2.0, 1.0])` returned 0 ties because the scan compared adjacent elements without sorting first. Fixed: sort + NaN-filter internally before scanning.

### blomqvist_beta — signum(0.0) = 1.0 in Rust
IEEE 754: `0.0f64.signum()` returns 1.0, not 0.0. Median-tied elements (dx = 0.0 or dy = 0.0) were being counted as concordant instead of excluded. Also: NaN inputs silently treated as discordant (-1). Fixed with raw `dx == 0.0` check + early NaN guard.

### hoeffdings_d — NaN silently dropped
NaN observations were excluded from ECDF counts without propagating NaN to the result. Fixed with early NaN guard.

### distance_correlation — overflow for extreme-scale inputs
`(1e300 - (-1e300)).abs()` = Inf. dCor is scale-invariant, so pre-normalize by max(|x|) before computing distances. Fixed.

### log_returns — Inf price passed the guard
`Inf <= 0.0` is false, so infinite prices slipped through. `ln(Inf/100) = Inf`. Fixed by adding `!is_finite()` check.

### adversarial_rigor_gauntlet test itself — wrong expectation
`blomqvist_perfect_positive` used odd n=9 with X=Y, expecting 1.0. With the signum fix, the median element (dx=0) is correctly excluded, giving 8/9. Fixed test to use even n=8.

## Commit

`0e9d42b` — "Rigor gauntlet: 8 bug fixes + 12 information theory primitives + 74 tests"
