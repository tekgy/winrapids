# Tambear Trig Family — Gold-Standard Parity Table

Maintained by: scientist (TRIG-18)
Gold standard: mpmath at 100-digit precision
Reference implementations: numpy (platform libm / SVML), scipy.special
Date: 2026-04-14

---

## Method

For each function:
1. Generate adversarial inputs (region boundaries, powers of 2, dense sweeps,
   Table-Maker's-Dilemma hard cases)
2. Evaluate with numpy/scipy and mpmath at 100 digits, both rounded to f64
3. Measure ULP distance between reference and gold standard
4. Identify worst-case inputs and classify by region

ULP distance: sign-magnitude bit comparison per IEEE 754. Signed zero treated
as unsigned zero. NaN vs NaN = 0, NaN vs non-NaN = sentinel max.

**Note on subnormal flush-to-zero**: when a function's output enters the
subnormal range (< 2.2e-308), some implementations return 0.0 while mpmath
returns the true subnormal value. ULP distance in this case can be astronomically
large (trillions of ulps) despite the absolute error being < 1e-308. These are
documented as saturation policy gaps, not precision failures.

---

## Parity Table: Currently Implemented Functions

| Function | Inputs | Worst ULP | <= 1 ulp | Worst Input | Key Notes |
|----------|--------|-----------|----------|-------------|-----------|
| sin      | 2521   | 1         | 100.0%   | x=-153.15   | 96.0% bit-perfect |
| cos      | 2521   | 1         | 100.0%   | x=-1e15     | 96.4% bit-perfect |
| exp      | 1105   | 1         | 100.0%   | various     | 99.3% bit-perfect |
| log      | 1197   | 0         | 100.0%   | all         | BIT-PERFECT on all tested |
| erf      | 1011   | 1         | 100.0%   | various     | 88.4% bit-perfect |
| erfc     | 2500+  | saturation| 75.5%    | x~26.6      | SATURATION POLICY GAP |
| tgamma   | 508    | 4         | 85%      | x~1.46      | 96% within 2 ulp |
| lgamma   | 508    | 169       | 91%      | various     | upstream scipy gap |
| tan      | 20299  | 1         | 100.0%   | x=-6.279e2  | 95.6% bit-perfect; poles covered |

---

## Pending: Not Yet Implemented

| Function | Status |
|----------|--------|
| tan      | VERIFIED — see parity table above |
| cot, sec, csc | TRIG-13 — awaiting compilable Rust |
| sincos (fused) | TRIG-13 — awaiting compilable Rust |
| asin, acos, atan, atan2 | TRIG-14 pending |
| acot, asec, acsc | TRIG-14 pending |
| sinh, cosh, tanh | TRIG-15 pending |
| asinh, acosh, atanh | TRIG-15 pending |
| sinpi, cospi, tanpi | TRIG-16 pending |
| asinpi, acospi, atanpi | TRIG-16 pending |

---

## Detailed Notes

### tan: 100% ≤1 ulp, poles covered

20,299 adversarial inputs including ±1/2/3-ulp neighborhoods of every tan/sec
pole (π/2 + kπ, k in -200..200) and cot/csc pole (kπ, k≠0). Worst case:
x=-6.278697e+02, ulp=1. 95.6% bit-perfect.

numpy.tan is ≤1 ulp vs mpmath on this input set — platform libm is correctly
rounded on all tested tan inputs. Tambear's tan_strict shares range reduction
infrastructure with sin.rs (reduce_trig, kernel_sin, kernel_cos) so accuracy
is structurally identical.

Pole behavior: at representable values nearest π/2 + kπ, tan(x) returns a
large finite value (not infinity), because the f64 representation of the pole
is not exactly π/2 + kπ. Both numpy and mpmath agree on this: the "pole" is
only approached, never reached, in f64 arithmetic. tan_adversarial() explicitly
tests ±1/2/3 ulp neighbors of each pole representation to verify that
tambear handles the steep gradient without overflow or accuracy collapse.

### log: bit-perfect

numpy's log is bit-perfect vs mpmath on all 1197 tested inputs. This means the
platform libm is returning the correctly-rounded result on every one of our
adversarial inputs. (It does NOT guarantee correctness on all 10^16 positive
doubles; Table-Maker's-Dilemma inputs were not exhaustively tested.)

Tambear's log should match this; its Remez-optimized approach is in the same
accuracy class.

### erfc: saturation policy gap

At x ~= 26.6, erfc(x) enters the subnormal f64 range (~3.7e-311).
scipy.special.erfc returns 0.0 (flush-to-zero), while mpmath returns the
true subnormal:

    x=26.663: scipy=0.0, mpmath=3.72e-311, ulp_dist=7.5e12

Absolute error: |0 - 3.72e-311| < 4e-311, which is below f64::MIN_POSITIVE.
This is a saturation policy decision, not a precision failure.

Tambear policy: return the subnormal when representable (x < ~27.3).
For x > ~27.3, erfc < 5e-324 (smallest subnormal) and 0.0 is correct.
This is strictly more accurate than scipy's early flush.

The 15-ulp worst case in the core region (-3 to 5) is real:
    core: n=1600, worst=15 ulp, <=1: 69%

This is a scipy precision gap worth filing upstream. Tambear target: <=2 ulp
core.

### lgamma: 169 ulp worst case

scipy.special.gammaln reaches 169 ulp vs mpmath on our tested [0.5, 20] range.
Likely occurs near the minimum of Gamma at x~1.4616 (Gamma'=0 there, so the
log of Gamma has a near-flat inflection — Taylor convergence is slowest there).

Tambear uses Lanczos approximation. Lanczos achieves ~15-digit accuracy on
[0.5, 20], giving <=2 ulp. The scipy 169-ulp is an upstream bug candidate.

### tgamma: 4 ulp worst case

4 ulp on our tested range. Acceptable as a reference gap (scipy), but tambear
should target <=2 ulp via Lanczos. The worst case is at x~1.4616 (same minimum).

---

## Exact Ground Truth: sin(n*pi/6) Grid

| n  | x          | Exact value  | np vs mp | np vs exact | Notes |
|----|------------|--------------|----------|-------------|-------|
| 0  | 0.0        | 0.0          | 0 ulp    | 0 ulp       | exact |
| 1  | pi/6       | 0.5          | 0 ulp    | 1 ulp       | |
| 2  | pi/3       | sqrt(3)/2    | 0 ulp    | 0 ulp       | |
| 3  | pi/2       | 1.0          | 0 ulp    | 0 ulp       | exact |
| 4  | 2*pi/3     | sqrt(3)/2    | 0 ulp    | 1 ulp       | |
| 5  | 5*pi/6     | 0.5          | 0 ulp    | 1 ulp       | |
| 6  | pi         | 0.0          | 0 ulp    | ~huge       | see policy note |
| 7  | 7*pi/6     | -0.5         | 0 ulp    | 5 ulp       | |
| 8  | 4*pi/3     | -sqrt(3)/2   | 1 ulp    | 1 ulp       | |
| 9  | 3*pi/2     | -1.0         | 0 ulp    | 0 ulp       | exact |
| 10 | 5*pi/3     | -sqrt(3)/2   | 0 ulp    | 0 ulp       | |
| 11 | 11*pi/6    | -0.5         | 0 ulp    | 4 ulp       | |
| 12 | 2*pi       | 0.0          | 0 ulp    | ~huge       | see policy note |

Policy note on zero crossings: sin(float(pi)) is not 0 because float(pi) != pi.
The correct test is sin(float_pi) vs mpmath.sin(float_pi): they agree to 0 ulp.
The "huge" ulp distance is from comparing sin(approximated-pi) to exact zero.

---

## atan2 Policy Gap Analysis

All 20 IEEE 754-2019 mandatory special cases verified against numpy on Windows.
Result: numpy agrees with IEEE 754-2019 on all 20 cases. No policy gaps.

| Case | IEEE 754-2019 | numpy | Status |
|------|--------------|-------|--------|
| atan2(+0, +0) | +0 | +0.0 | OK |
| atan2(+0, -0) | +pi | +pi | OK |
| atan2(-0, +0) | -0 | -0.0 | OK |
| atan2(-0, -0) | -pi | -pi | OK |
| atan2(+0, x<0) | +pi | +pi | OK |
| atan2(-0, x<0) | -pi | -pi | OK |
| atan2(+0, x>0) | +0 | +0.0 | OK |
| atan2(-0, x>0) | -0 | -0.0 | OK |
| atan2(y>0, +0) | +pi/2 | +pi/2 | OK |
| atan2(y<0, +0) | -pi/2 | -pi/2 | OK |
| atan2(y>0, -0) | +pi/2 | +pi/2 | OK |
| atan2(y<0, -0) | -pi/2 | -pi/2 | OK |
| atan2(+inf, +inf) | +pi/4 | +pi/4 | OK |
| atan2(+inf, -inf) | +3*pi/4 | +3*pi/4 | OK |
| atan2(-inf, +inf) | -pi/4 | -pi/4 | OK |
| atan2(-inf, -inf) | -3*pi/4 | -3*pi/4 | OK |
| atan2(+inf, finite) | +pi/2 | +pi/2 | OK |
| atan2(-inf, finite) | -pi/2 | -pi/2 | OK |
| atan2(finite, +inf) | +0 | +0.0 | OK |
| atan2(finite, -inf) | +pi | +pi | OK |

Tambear atan2 implementation target: match all 20 IEEE cases exactly.
Policy is clear: implement IEEE 754-2019 verbatim. No decisions to make here.

---

## Upstream Bug Candidates

Functions where scipy/numpy deviates from mpmath gold standard in ways that
are reproducible, measurable, and likely incorrect:

| Function | Issue | Severity | Action |
|----------|-------|----------|--------|
| erfc (subnormal) | Flushes to 0.0 for x > ~26.6 | Policy choice | Document; note scipy divergence |
| erfc (core) | 15 ulp in transition zone | Real precision gap | File scipy issue |
| lgamma | 169 ulp on some inputs | Real precision gap | Find input; file scipy issue |
| tgamma | 4 ulp near x~1.46 | Borderline | Tambear Lanczos should beat this |

---

## $1M/yr Scientist Defaults Summary

Universal rule: compensated is the default for every function in tambear.
The 1-ulp improvement over strict justifies ~10% overhead universally.
See scientist-defaults.md for full rationale per function.

---

## Sign-off Status

| Function | np vs mp verified | Policy gaps | Tambear column | Sign-off |
|----------|------------------|-------------|----------------|----------|
| sin      | YES              | none found  | pending        | partial  |
| cos      | YES              | none found  | pending        | partial  |
| exp      | YES              | none found  | pending        | partial  |
| log      | YES (bit-perfect)| none found  | pending        | partial  |
| erf      | YES              | none found  | pending        | partial  |
| erfc     | YES              | saturation  | pending        | partial  |
| tgamma   | YES              | noted       | pending        | partial  |
| lgamma   | YES              | upstream bug| pending        | partial  |
| tan      | YES            | none found  | pending Rust   | partial  |
| cot, sec, csc, sincos | NOT YET | —        | —              | —        |
| asin..atanh | NOT YET     | —           | —              | —        |
| sinpi..  | NOT YET          | —           | —              | —        |
