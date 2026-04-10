# Special Functions — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: REVIEWED
**Code**: `crates/tambear/src/special_functions.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| erf (A&S 7.1.26) | special_functions.rs:32-42 | OK (1.5e-7 max error) |
| erfc | special_functions.rs:47-59 | OK (no cancellation) |
| log_gamma (Lanczos g=7) | special_functions.rs:69-98 | OK |
| gamma | special_functions.rs:101-103 | OK |
| log_beta | special_functions.rs:106-108 | OK |
| regularized_incomplete_beta (Lentz CF) | special_functions.rs:121-173 | OK |
| regularized_gamma_p (series + CF) | special_functions.rs:182-203 | OK |
| regularized_gamma_q | special_functions.rs:194-203 | OK |
| normal_cdf | special_functions.rs:257-259 | OK |
| normal_sf | special_functions.rs:264-266 | OK (no cancellation) |
| t_cdf | special_functions.rs:273-282 | OK |
| f_cdf | special_functions.rs:288-293 | OK |
| chi2_cdf | special_functions.rs:298-302 | OK |
| chi2_sf | special_functions.rs:305-309 | OK |
| p-value helpers | special_functions.rs:312-329 | OK |

---

## Finding SF-1: erf Precision Limit (LOW)

**Note**: erf uses Abramowitz & Stegun approximation 7.1.26 with max error 1.5×10⁻⁷. This is sufficient for typical hypothesis testing (p-values > 1e-4), but insufficient for:

- 5-sigma tests (p ≈ 5.7e-7): last significant digit may be wrong
- 6-sigma tests (p ≈ 2e-9): completely unreliable
- Quality control (6-sigma methodology)

For higher precision, use the rational approximation from Hart et al. (max error < 1e-15) or Cody's erfc implementation.

**Impact**: p-values in the normal tail beyond ±4.5σ may have relative errors > 1%.

---

## Positive Findings

**ALL special functions are mathematically correct.** This is the strongest module reviewed.

**Lanczos log_gamma is gold standard.** g=7, 9-term Godfrey coefficients. Reflection formula for x < 0.5 handles poles.

**Regularized incomplete beta is robust.** Lentz's continued fraction with symmetry swap for convergence. Even/odd step alternation matches DLMF 8.17.22. Front factor computed in log domain to prevent overflow.

**erfc avoids 1-erf cancellation.** Direct computation for x > 0 (line 48-55) ensures no catastrophic cancellation in the tail. This is correct design.

**normal_sf uses erfc directly.** No cancellation for large x. P-values in the normal tail are computed accurately (subject to erf precision).

**t_cdf via incomplete beta is correct.** The transformation x = ν/(ν+t²) with ½I_x(ν/2, ½) is the standard representation.

---

## Test Vectors

### TV-SF-ERF-01: Known values
```
erf(0) = 0, erf(1) ≈ 0.8427, erf(2) ≈ 0.9953
erf(-x) = -erf(x)
erf(6) ≈ 1.0 (within 1e-15)
```

### TV-SF-LGAMMA-01: Integer factorials
```
log_gamma(1) = 0, log_gamma(5) = ln(24), log_gamma(10) = ln(362880)
log_gamma(0.5) = 0.5 * ln(π)
```

### TV-SF-IBETA-01: Symmetry
```
I_x(a,b) + I_{1-x}(b,a) = 1 for any x,a,b
```

### TV-SF-NORMAL-01: Standard z-scores
```
Φ(0) = 0.5, Φ(1.96) ≈ 0.975, Φ(-1.96) ≈ 0.025
Φ(x) + Φ(-x) = 1
```

### TV-SF-PRECISION-01: erf tail precision (INFORMATIONAL)
```
erf(4.0) = 0.99999998458... (exact 15 digits)
A&S 7.1.26 may only give 7 digits correct
Test: |erf_computed(4.0) - exact| / exact < 1.5e-7
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| SF-1: erf precision limit | **LOW** | p-values wrong beyond 5σ | Higher-order rational approx |
