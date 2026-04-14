# Tambear Trig Family — Gold-Standard Parity Table

Maintained by: scientist (TRIG-18)  
Gold standard: mpmath at 100-digit precision  
Reference implementations: numpy (platform libm / SVML), scipy  
Date: 2026-04-13

---

## Method

For each trig function:
1. Generate adversarial inputs: multiples of pi/4, near-zero, large arguments
   (Payne-Hanek domain), Kahan hard cases, dense sweep through [0, 2*pi]
2. Evaluate with numpy (f64, platform libm) and mpmath (100 digits -> rounded to f64)
3. Measure ULP distance between numpy and mpmath
4. Identify worst-case input

The "numpy vs mpmath" column is the parity gap for the *reference* implementations.
When tambear ships Rust bindings, a "tambear vs mpmath" column is added.

ULP distance is computed via sign-magnitude bit manipulation (IEEE 754 bit ordering).
NaN vs non-NaN = sentinel (max uint64). Signed zero treated as unsigned zero.

---

## Parity Table

| Function | Inputs | Worst ULP (np vs mp) | <= 1 ulp | <= 2 ulp | Worst Input | Sign-off |
|----------|--------|---------------------|----------|----------|-------------|----------|
| sin      | 2521   | 1                   | 100.0%   | 100.0%   | x=-153.15... | pending |
| cos      | 2521   | 1                   | 100.0%   | 100.0%   | x=-1e15    | pending |
| tan      | —      | —                   | —        | —        | —           | not yet |
| cot      | —      | —                   | —        | —        | —           | not yet |
| sec      | —      | —                   | —        | —        | —           | not yet |
| csc      | —      | —                   | —        | —        | —           | not yet |
| asin     | —      | —                   | —        | —        | —           | not yet |
| acos     | —      | —                   | —        | —        | —           | not yet |
| atan     | —      | —                   | —        | —        | —           | not yet |
| atan2    | —      | —                   | —        | —        | —           | not yet |
| sinh     | —      | —                   | —        | —        | —           | not yet |
| cosh     | —      | —                   | —        | —        | —           | not yet |
| tanh     | —      | —                   | —        | —        | —           | not yet |
| asinh    | —      | —                   | —        | —        | —           | not yet |
| acosh    | —      | —                   | —        | —        | —           | not yet |
| atanh    | —      | —                   | —        | —        | —           | not yet |
| sinpi    | —      | —                   | —        | —        | —           | not yet |
| cospi    | —      | —                   | —        | —        | —           | not yet |
| tanpi    | —      | —                   | —        | —        | —           | not yet |
| sincos   | —      | —                   | —        | —        | —           | not yet |

---

## Exact Ground Truth: sin(n*pi/6) Grid

These inputs have known closed-form exact values. The "np vs exact" column
measures how far numpy's f64 result is from the true mathematical value.
The "np vs mp" column measures agreement between numpy and mpmath.

| n  | x (radians)    | Exact value     | np vs mp | np vs exact | Notes |
|----|----------------|-----------------|----------|-------------|-------|
| 0  | 0.0000         | 0.0             | 0 ulp    | 0 ulp       | exact |
| 1  | pi/6           | 0.5             | 0 ulp    | 1 ulp       | |
| 2  | pi/3           | sqrt(3)/2       | 0 ulp    | 0 ulp       | |
| 3  | pi/2           | 1.0             | 0 ulp    | 0 ulp       | exact |
| 4  | 2*pi/3         | sqrt(3)/2       | 0 ulp    | 1 ulp       | |
| 5  | 5*pi/6         | 0.5             | 0 ulp    | 1 ulp       | |
| 6  | pi             | 0.0             | 0 ulp    | ~huge       | see note 1 |
| 7  | 7*pi/6         | -0.5            | 0 ulp    | 5 ulp       | |
| 8  | 4*pi/3         | -sqrt(3)/2      | 1 ulp    | 1 ulp       | |
| 9  | 3*pi/2         | -1.0            | 0 ulp    | 0 ulp       | exact |
| 10 | 5*pi/3         | -sqrt(3)/2      | 0 ulp    | 0 ulp       | |
| 11 | 11*pi/6        | -0.5            | 0 ulp    | 4 ulp       | |
| 12 | 2*pi           | 0.0             | 0 ulp    | ~huge       | see note 1 |

**Note 1**: sin(float(pi)) != 0.0 because float(pi) != pi. The f64 representation
of pi is pi + ~1.2e-16, so sin(float_pi) ≈ 1.22e-16 (not zero). This is correct
behavior — the function is not at a zero of the mathematical sine. ULP distance
from 0.0 is enormous but this is not a bug; it's a consequence of argument
representation. The test for zero crossings should be sin(pi_mpmath_rounded_f64)
vs mpmath.sin(pi_mpmath_rounded_f64), not vs exact 0.0.

---

## Exact Ground Truth: cos(n*pi/6) Grid

| n  | x (radians)    | Exact value     | np vs mp | np vs exact | Notes |
|----|----------------|-----------------|----------|-------------|-------|
| 0  | 0.0000         | 1.0             | 0 ulp    | 0 ulp       | exact |
| 1  | pi/6           | sqrt(3)/2       | 0 ulp    | 1 ulp       | |
| 2  | pi/3           | 0.5             | 0 ulp    | 1 ulp       | |
| 3  | pi/2           | 0.0             | 0 ulp    | ~huge       | see note 1 |
| 4  | 2*pi/3         | -0.5            | 0 ulp    | 4 ulp       | |
| 5  | 5*pi/6         | -sqrt(3)/2      | 0 ulp    | 1 ulp       | |
| 6  | pi             | -1.0            | 0 ulp    | 0 ulp       | exact |
| 7  | 7*pi/6         | -sqrt(3)/2      | 0 ulp    | 2 ulp       | |
| 8  | 4*pi/3         | -0.5            | 0 ulp    | 4 ulp       | |
| 9  | 3*pi/2         | 0.0             | 0 ulp    | ~huge       | see note 1 |
| 10 | 5*pi/3         | 0.5             | 0 ulp    | 1 ulp       | |
| 11 | 11*pi/6        | sqrt(3)/2       | 0 ulp    | 2 ulp       | |
| 12 | 2*pi           | 1.0             | 0 ulp    | 0 ulp       | exact |

**Note 1**: Same as sin — cos(float(pi/2)) is not exactly 0.

---

## Large-Argument Analysis (Payne-Hanek domain, |x| > 1.6e6)

| Function | Inputs tested | Worst ULP | Worst input | Notes |
|----------|---------------|-----------|-------------|-------|
| sin      | 8             | 0 ulp     | all zero    | libm and mpmath agree exactly |
| cos      | 8             | 1 ulp     | x=-1e15     | expected — range reduction regime |

cos shows 1 ulp at x=-1e15. This is within budget. Tambear's Payne-Hanek
reduction provides ~120 bits of precision; the 1 ulp seen here is the
rounding boundary from the f64 kernel, not from range reduction failure.

---

## Policy Gaps

These are places where implementations deliberately differ by policy, not error.

| Case | numpy | scipy | mpmath | tambear policy |
|------|-------|-------|--------|----------------|
| sin(+0.0) | +0.0 | +0.0 | 0 | +0.0 (IEEE 754: sign preserved) |
| sin(-0.0) | -0.0 | -0.0 | 0 | -0.0 (IEEE 754: sign preserved) |
| cos(-0.0) | 1.0  | 1.0  | 1 | 1.0 (cos is even) |
| sin(+inf) | NaN  | NaN  | NaN | NaN |
| sin(-inf) | NaN  | NaN  | NaN | NaN |
| sin(NaN)  | NaN  | NaN  | NaN | NaN |
| cos(+inf) | NaN  | NaN  | NaN | NaN |

All implementations agree on the above. No policy gaps found for sin/cos.

---

## $1M/yr Scientist Defaults (TRIG-6)

### sin / cos

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | Worst-case drops from ~2 ulp (strict) to ~1 ulp. ~10% slower. The tradeoff is worth it for a math library. |
| angle_unit      | radians          | Universal convention for numerical work. Degree/turn inputs are a conversion, not a default. |
| range_reduction | auto             | Cody-Waite up to |x| < 2^20*pi/2 (exact for |k| < 2^20), Payne-Hanek beyond. Never force one path. |

**Why not correctly_rounded as default?**  
The correctly_rounded path uses double-double throughout, adding ~3x overhead.
The compensated path already achieves 1 ulp on nearly all inputs. The marginal
gain (0 -> 0 ulp on the hardest cases) doesn't justify the 3x slowdown for
a fintech farm that calls sin() 10^9 times per day.

**Why not strict?**  
The strict single-FMA Horner path has documented worst-case 2 ulps. For
scientific computing where users may accumulate errors across billions of
calls, starting with the 2-ulp path as default is not what a senior
numerical analyst would choose.

---

## Sign-off

sin (initial): pending (Rust bindings needed for tambear column)  
cos (initial): pending (Rust bindings needed for tambear column)  

numpy baseline (Python libm): VERIFIED — ≤ 1 ulp vs mpmath on 2521 adversarial inputs.
