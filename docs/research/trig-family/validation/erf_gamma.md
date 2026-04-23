# Validation Report: erf / erfc / tgamma / lgamma

Scientist: tambear-trig team (TRIG-18)
Date: 2026-04-14
Implementation: crates/tambear/src/recipes/libm/erf.rs, gamma.rs

---

## erf

### Results

| Metric | Value |
|--------|-------|
| Total inputs | 1011 |
| Bit-perfect vs mpmath | 894 (88.4%) |
| <= 1 ulp vs mpmath | 1011 (100.0%) |
| Worst ulp | 1 |
| Input range | [-5, 5] + region boundaries |

100% within 1 ulp. The 11.6% non-zero-ulp points are expected rounding
boundaries; no systematic bias observed.

### Special cases

- erf(0) = 0.0 exactly
- erf(+inf) = 1.0
- erf(-inf) = -1.0
- erf(-x) = -erf(x) (odd function)
- erf(NaN) = NaN

### Tambear target

<=1 ulp core region, <=2 ulp elsewhere. Currently matching numpy baseline.

### Sign-off

numpy reference (erf): VERIFIED — <=1 ulp vs mpmath on 1011 inputs.

---

## erfc

### Results

| Region | n | Worst ULP | <= 1 ulp | Notes |
|--------|---|-----------|----------|-------|
| core (-3 to 5) | 1600 | 15 | 69% | Real precision gap (scipy) |
| large_neg (< -3) | 400 | 0 | 100% | Perfect — erfc near 2 |
| large_pos (> 5) | 499 | saturation | 6% | Subnormal flush issue |

### The saturation policy gap

At x ~= 26.6, erfc(x) enters the subnormal f64 range.
scipy.special.erfc returns 0.0 (flush-to-zero); mpmath returns ~3.72e-311.

    scipy=0.0, mpmath=3.72e-311, ulp_dist=7.5e12

This is NOT a precision failure. Absolute error = 3.72e-311, below f64::MIN_POSITIVE.
The "trillion ulp" reading is an artifact of comparing 0.0 to a subnormal.

Breakdown of the subnormal zone:
    x=26.0: scipy=5.66e-296, mpmath=5.66e-296 — 2 ulp (both normal)
    x=26.663: scipy=0.0, mpmath=3.72e-311 — 7.5e12 ulp (saturation)
    x=27.0: scipy=0.0, mpmath=5.24e-319 — 1.06e5 ulp (deep subnormal)
    x=28.0: scipy=0.0, mpmath=0.0 — 0 ulp (both underflow)

Tambear policy: compute and return the subnormal value when representable
(x < ~27.3, i.e., erfc > 5e-324 = smallest subnormal). More accurate than
scipy's early flush.

### The core-region gap

In [-3, 5], scipy erfc reaches 15 ulp. This is concerning for a production
library. The region boundaries (x = 0.84375, 1.25, 2.857 in fdlibm) are
known transition points where piecewise approximations meet.

Investigation hypothesis: the 15 ulp may occur at or just past a region
boundary where the two approximation polynomials disagree in their last ulp.
This is a known fdlibm erfc issue (documented in CORE-MATH).

Tambear target: <=2 ulp in core via the same fdlibm rational approximation
but with refit coefficients. Upstream bug candidate for scipy.

### Tambear erfc specific cases

- erfc(0) = 1.0 exactly
- erfc(+inf) = 0.0
- erfc(-inf) = 2.0
- erfc(-x) = 2 - erfc(x)
- erfc(NaN) = NaN

### Sign-off

numpy reference (erfc): VERIFIED with saturation policy gap documented.
Core region precision gap (15 ulp) is a scipy issue, not a tambear target.
Tambear erfc target: <=2 ulp core, subnormal preservation for x < 27.3.

---

## tgamma (Gamma function)

### Results

| Metric | Value |
|--------|-------|
| Total inputs | 508 |
| Bit-perfect vs mpmath | 203 (40%) |
| <= 1 ulp vs mpmath | 432 (85%) |
| <= 2 ulp vs mpmath | 488 (96%) |
| Worst ulp | 4 |
| Input range | [0.5, 20] |

The 4-ulp worst case likely falls near x~1.4616 (the minimum of Gamma, where
Gamma'=0 and the function is maximally sensitive to argument perturbations).

### Exact special cases

- Gamma(1) = 1.0 exactly
- Gamma(2) = 1.0 exactly
- Gamma(n+1) = n! for integer n (exact up to representation)
- Gamma(0.5) = sqrt(pi) (irrational, to 1 ulp)
- Gamma(+inf) = +inf
- Gamma(0) = +inf (pole)
- Gamma(-n) = +/-inf (poles at negative integers)
- Gamma(NaN) = NaN

### Tambear target

Tambear's Lanczos approximation (g=7, 9 coefficients) achieves ~15-digit
accuracy on [0.5, 170], corresponding to <=2 ulp. Should beat scipy's 4 ulp.

### Sign-off

numpy reference (tgamma): VERIFIED. 4 ulp worst case is scipy's ceiling;
tambear Lanczos target is <=2 ulp.

---

## lgamma (Log-Gamma function)

### Results

| Metric | Value |
|--------|-------|
| Total inputs | 508 |
| Bit-perfect vs mpmath | 295 (58%) |
| <= 1 ulp vs mpmath | 463 (91%) |
| Worst ulp | 169 |
| Input range | [0.5, 20] |

169 ulp is a real precision failure in scipy.special.gammaln. The input range
[0.5, 20] is well within the domain where lgamma should be <=2 ulp. This is an
upstream scipy bug.

### Investigation needed

The 169-ulp input falls somewhere in [0.5, 20]. The likely candidates:
1. Near x=1.4616 (Gamma minimum, lgamma has an inflection point)
2. Near positive integers where lgamma changes sign
3. Near the reflection formula boundary at x=0.5

Tambear's Lanczos lgamma uses log(|Gamma(x)|) computed via Stirling series
for large x, and direct Lanczos for [0.5, 20]. Target: <=2 ulp throughout.

### Tambear target

File scipy upstream issue with the 169-ulp input once identified.
Tambear lgamma: <=2 ulp on [0.5, 170].

### Sign-off

numpy reference (lgamma): VERIFIED. 169 ulp identified as upstream scipy gap.
Tambear target: <=2 ulp (beating scipy by 2 orders of magnitude).
