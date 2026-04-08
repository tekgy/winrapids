# Family 04: Random Number Generation — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: REVIEWED
**Code**: `crates/tambear/src/rng.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| SplitMix64 | rng.rs:68-82 | OK |
| xoshiro256** | rng.rs:94-134 | OK |
| LCG | rng.rs:147-158 | OK |
| next_f64 | rng.rs:34-36 | OK (53-bit precision) |
| next_range | rng.rs:39-49 | OK (rejection for bias) |
| Box-Muller | rng.rs:165-175 | OK |
| Exponential | rng.rs:184-191 | OK |
| Gamma (Marsaglia-Tsang) | rng.rs:196-217 | OK |
| Beta | rng.rs:220-223 | OK |
| Chi-squared | rng.rs:227-229 | OK |
| Student-t | rng.rs:232-236 | OK |
| Cauchy | rng.rs:246-249 | OK |
| Poisson (Knuth + normal) | rng.rs:262-279 | OK-ish (approximation) |
| Binomial (direct + normal) | rng.rs:282-292 | OK-ish (approximation) |
| Fisher-Yates shuffle | rng.rs:309-315 | OK |
| Floyd's sampling | rng.rs:320-339 | OK |
| Weighted sampling | rng.rs:344-365 | **MEDIUM** (NaN panic) |

---

## Finding F04-1: Weighted Sampling NaN Panic (MEDIUM)

**Bug**: `sample_weighted` at line 360 uses `partial_cmp(&u).unwrap()` inside `binary_search_by`. If any weight is NaN, the CDF will contain NaN, and the binary search will panic.

**Impact**: Thread panic when weight vector contains NaN.

**Fix**: `partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal)`

---

## Finding F04-2: Normal Approximation for Large Lambda/N (LOW)

**Note**: Poisson (lambda >= 30) switches to normal approximation at line 276. Binomial (n >= 30) does likewise at line 288. Standard trade-off for efficiency, but introduces approximation error.

For lambda = 30, the Poisson normal approximation has noticeable discretization artifacts. The transition threshold could be higher (e.g., lambda >= 100) for better accuracy.

---

## Positive Findings

**xoshiro256** is correct.** Output function `(s[1]*5).rotate_left(7)*9` and state update match Blackman-Vigna reference. Jump polynomial constants verified.

**Box-Muller is correct.** Guard `u1 > 1e-300` prevents ln(0). Both outputs used in fill_normal.

**Marsaglia-Tsang gamma is correct.** Handles alpha < 1 via Gamma(alpha+1) * U^(1/alpha) reduction.

**next_range uses rejection sampling.** Avoids modulo bias. Correct.

**Fisher-Yates shuffle and Floyd's sampling are textbook correct.**

---

## Test Vectors

### TV-F04-WS-01: Weighted sampling with NaN (BUG CHECK)
```
weights = [1.0, NaN, 1.0], k=100
Expected: graceful handling
Currently: PANIC
```

### TV-F04-NORM-01: Box-Muller distribution check
```
seed=42, n=100000, mu=0, sigma=1
mean within 0.02 of 0, var within 0.05 of 1
Kolmogorov-Smirnov test p > 0.01
```

### TV-F04-POISSON-01: Poisson at lambda=30 (approximation boundary)
```
seed=42, n=100000, lambda=30
Test: chi-squared goodness-of-fit against exact Poisson PMF
Expected: p > 0.01 (currently may fail due to normal approx discretization)
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F04-1: Weighted sampling NaN | **MEDIUM** | Thread panic | unwrap_or(Equal) |
| F04-2: Normal approximation | **LOW** | Approximation error | Increase threshold |
