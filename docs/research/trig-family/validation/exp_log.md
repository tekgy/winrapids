# Validation Report: exp / log

Scientist: tambear-trig team (TRIG-18)
Date: 2026-04-14
Implementation: crates/tambear/src/recipes/libm/exp.rs, log.rs

---

## exp

### Method

- 1105 adversarial inputs: integer multiples of ln(2), dense sweep [-10, 10],
  near-threshold values (EXP_MAX ~709.78, EXP_MIN ~-745.13), powers of 2,
  mathematical constants
- Reference: numpy.exp vs mpmath.exp at 100 digits

### Results

| Metric | Value |
|--------|-------|
| Total inputs | 1105 |
| Bit-perfect vs mpmath | 1098 (99.3%) |
| <= 1 ulp vs mpmath | 1105 (100.0%) |
| Worst ulp | 1 |

99.3% bit-perfect means numpy.exp is the correctly-rounded result on nearly
every tested input. The 7 non-zero-ulp cases are genuine rounding boundary
points where the true value sits within 0.5 ulp of the f64 midpoint between
two adjacent floats.

### Exact special cases

- exp(0) = 1.0 exactly
- exp(1) = e — numpy returns 2.718281828459045 = closest f64 to e
- exp(-inf) = 0.0
- exp(+inf) = +inf
- exp(NaN) = NaN

### Tambear target

Tambear's exp uses Cody-Waite range reduction with three-strategy polynomial.
The compensated strategy should achieve <=2 ulp on all tested inputs, matching
the numpy baseline.

### Sign-off

numpy reference (exp): VERIFIED — <=1 ulp vs mpmath on 1105 adversarial inputs.
Tambear comparison: pending Python bindings.

---

## log

### Method

- 1197 adversarial inputs: powers of 2 from 2^-100 to 2^100, dense sweep
  [0.1, 100], extra density near x=1 (cancellation hotspot), mathematical
  constants, near-unity values
- Reference: numpy.log vs mpmath.log at 100 digits

### Results

| Metric | Value |
|--------|-------|
| Total inputs | 1197 |
| Bit-perfect vs mpmath | 1197 (100.0%) |
| <= 1 ulp vs mpmath | 1197 (100.0%) |
| Worst ulp | 0 |

BIT-PERFECT on every tested input. This is exceptional. numpy's log is
returning the correctly-rounded IEEE 754 result on all 1197 adversarial inputs,
including the near-unity cancellation zone (log(1+eps) for small eps).

### Near-unity analysis

The most dangerous region for log is x near 1, where log(x) has catastrophic
cancellation. log(1 + 1e-10) should use log1p internally. numpy passes 0 ulp
here, confirming the fdlibm log1p path is active.

### Exact special cases

- log(1.0) = 0.0 exactly
- log(e) = 1.0 (to 1 ulp of 1.0 = 0.0 error since 1.0 is exact)
- log(2) = ln(2) = 0.693... (to rounding)
- log(0) = -inf
- log(-x) = NaN (domain error)
- log(+inf) = +inf

### Tambear target

Tambear's log uses the same fdlibm-inspired approach. The bit-perfect baseline
from numpy sets a high bar: tambear should achieve <=1 ulp everywhere, matching
or beating the reference.

### Sign-off

numpy reference (log): VERIFIED — BIT-PERFECT vs mpmath on 1197 adversarial inputs.
This is the highest-quality baseline in the libm family.
Tambear comparison: pending Python bindings.
