# Libm Special-Values Matrix

*Written by adversarial mathematician, 2026-04-11.*
*This is the acceptance criterion. Every cell must pass before a libm function is "done."*
*Math-researcher: your design doc (campsite 2.5) must address every non-trivial cell in your function's column.*

---

## How to use this table

Each cell contains the mandated return value per IEEE 754-2008 (§9.2) and the
C standard (C99 Annex F / POSIX `<math.h>`). Where standards differ or are silent,
the cell notes the ambiguity and states tambear's chosen policy.

**"mandated"** — IEEE 754 / C standard specifies the exact result.  
**"implementation-defined"** — standard allows any finite result or NaN; we must pick and document.  
**"invalid operation" (sNaN/qNaN)** — standard requires NaN result and optional signal.  
**tambear policy** — our choice where the standard is silent.

### Testing procedure

For every non-trivial cell, two tests are required:

1. A CPU interpreter test: `interpreter.eval(tam_fn, input) == expected`
2. When PTX backend lands: cross-backend agreement test

"Expected" is always an mpmath oracle value computed at 50-digit precision, then
rounded to the nearest f64. For cells where the result is exactly representable
(±inf, NaN, ±0), the expected value is bit-exact.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| +0 | positive zero (+0.0) |
| -0 | negative zero (-0.0) |
| +∞ | positive infinity (f64::INFINITY) |
| -∞ | negative infinity (f64::NEG_INFINITY) |
| NaN | any NaN (result must have `is_nan() == true`) |
| sub | subnormal: any value with `is_subnormal() == true` |
| tiny | smallest positive normal: `f64::MIN_POSITIVE` = 2.2e-308 |
| huge | largest finite: `f64::MAX` = 1.8e308 |
| ε | machine epsilon: `f64::EPSILON` = 2.2e-16 |
| [note] | see footnotes section below |

---

## exp(x) = eˣ

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | 1.0 | exact |
| -0 | 1.0 | exact |
| +∞ | +∞ | exact |
| -∞ | +0 | exact |
| NaN | NaN | IEEE 754 §9.2.1: NaN propagates |
| 1.0 | e = 2.718... | mpmath oracle; 1 ULP tolerance |
| -1.0 | 1/e | mpmath oracle; 1 ULP tolerance |
| 709.78 (≈ ln(MAX)) | finite, close to MAX | mpmath oracle |
| 709.79 | +∞ (overflow) | exact: overflow → inf |
| -708.40 (≈ ln(MIN_POS)) | ≈ MIN_POSITIVE | mpmath oracle; may be subnormal |
| -745.1 | subnormal | [note 1] |
| -746.0 | +0 (underflow) | exact: underflow → +0 |
| sub (positive) | ≈ 1.0 | exp(sub) ≈ 1+sub; full precision test |
| -sub | ≈ 1.0 | exp(-sub) ≈ 1-sub; full precision test |
| huge | +∞ | overflow |
| -huge | +0 | underflow |
| ln(2) ≈ 0.6931 | 2.0 exact | [note 2] critical range-reduction test |
| -ln(2) | 0.5 exact | [note 2] |
| 1e-300 | ≈ 1.0 + 1e-300 | tiny argument; full precision |
| -1e-300 | ≈ 1.0 - 1e-300 | tiny argument; full precision |

---

## ln(x) — natural logarithm

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | -∞ | IEEE 754 §9.2.1: divideByZero signal |
| -0 | -∞ | same as +0 (sign irrelevant for log) |
| -1.0 | NaN | invalid: log of negative [note 3] |
| -huge | NaN | invalid |
| -sub | NaN | invalid |
| +∞ | +∞ | exact |
| NaN | NaN | propagates |
| 1.0 | +0 | exact |
| e | 1.0 | mpmath oracle; 1 ULP |
| 2.0 | ln(2) ≈ 0.6931... | mpmath oracle; 1 ULP; critical constant |
| sub | large negative finite | mpmath oracle [note 4] |
| tiny = MIN_POSITIVE | ≈ -708.4 | mpmath oracle |
| huge | ≈ 709.78 | mpmath oracle |
| 1.0 + ε | ≈ ε | catastrophic cancellation check [note 5] |
| 1.0 - ε/2 | ≈ -ε/2 | same |

---

## sqrt(x) — square root

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | +0 | exact (IEEE 754 mandated) |
| -0 | -0 | exact (sign preserved) |
| +∞ | +∞ | exact |
| NaN | NaN | propagates |
| -1.0 | NaN | invalid (sqrt of negative) |
| -sub | NaN | invalid |
| -∞ | NaN | invalid |
| 1.0 | 1.0 | exact (perfectly representable) |
| 4.0 | 2.0 | exact |
| 2.0 | √2 | **correctly rounded** — IEEE 754 mandates sqrt is always correctly rounded |
| sub | √sub | must not flush subnormal to zero [note 6] |
| tiny | √tiny | mpmath oracle |
| huge | √huge | mpmath oracle; no overflow |
| 0.25 | 0.5 | exact |
| 1e-300 | 1e-150 | exact (no precision loss here) |

**Note**: sqrt is special — IEEE 754 mandates correctly-rounded result. Unlike exp/ln/sin,
sqrt must return the nearest f64 to the true result. No "1 ULP" tolerance — this must be
bit-exact against any correct f64 sqrt implementation.

---

## sin(x)

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | +0 | exact |
| -0 | -0 | exact (sign preserved) |
| +∞ | NaN | invalid (sin of infinity) |
| -∞ | NaN | invalid |
| NaN | NaN | propagates |
| π/6 | 0.5 | mpmath oracle; 1 ULP |
| π/4 | √2/2 | mpmath oracle; 1 ULP |
| π/2 | 1.0 | mpmath oracle; CRITICAL [note 7] |
| π | ≈ 0 (tiny) | mpmath oracle — NOT exactly 0 [note 7] |
| 2π | ≈ 0 (tiny) | same |
| 1e300 | implementation-defined | [note 8] large-argument reduction |
| 1e20 | implementation-defined | large-argument: Payne-Hanek needed |
| sub | ≈ sub | sin(x) ≈ x for tiny x; full precision |
| -sub | ≈ -sub | |
| huge | NaN or implementation-defined | [note 8] |

---

## cos(x)

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | 1.0 | exact |
| -0 | 1.0 | exact (same: cos is even) |
| +∞ | NaN | invalid |
| -∞ | NaN | invalid |
| NaN | NaN | propagates |
| π/3 | 0.5 | mpmath oracle; 1 ULP |
| π/2 | ≈ 0 (tiny) | NOT exactly 0 [note 7] |
| π | -1.0 | mpmath oracle; 1 ULP |
| sub | ≈ 1.0 | cos(sub) ≈ 1 - sub²/2 ≈ 1.0 |
| 1e20 | implementation-defined | large-argument reduction |

---

## tan(x)

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | +0 | exact |
| -0 | -0 | exact |
| +∞ | NaN | invalid |
| -∞ | NaN | invalid |
| NaN | NaN | propagates |
| π/4 | ≈ 1.0 | 1 ULP |
| π/2 | very large finite or ±∞ | [note 9]: tan(π/2) is ±∞ mathematically, but π/2 in f64 is not exactly π/2 |
| sub | ≈ sub | |

---

## pow(x, y)

| Input (x, y) | Mandated result | Notes |
|-------------|----------------|-------|
| (1.0, any) | 1.0 | exact — IEEE 754 mandates |
| (any, 0.0) | 1.0 | exact — IEEE 754 mandates (including pow(NaN, 0) = 1) |
| (any, +0) | 1.0 | exact |
| (-1.0, ±∞) | 1.0 | exact — IEEE 754 |
| (+0, negative) | +∞ | divideByZero signal |
| (-0, negative integer) | -∞ | divideByZero signal (sign depends on parity) |
| (negative, non-integer) | NaN | invalid |
| (+∞, positive) | +∞ | exact |
| (+∞, negative) | +0 | exact |
| (NaN, non-zero) | NaN | propagates |
| (non-one, NaN) | NaN | propagates |
| (2.0, 10.0) | 1024.0 | exact (integer power) |
| (2.0, 0.5) | √2 | = sqrt(2); same correctly-rounded requirement |
| (sub, 0.5) | √sub | subnormal input |
| (huge, 2.0) | +∞ | overflow |

---

## tanh(x)

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | +0 | exact |
| -0 | -0 | exact |
| +∞ | 1.0 | exact |
| -∞ | -1.0 | exact |
| NaN | NaN | propagates |
| very large (≥19.0) | ≈ 1.0 | result indistinguishable from 1.0 in f64 |
| very large negative | ≈ -1.0 | |
| sub | ≈ sub | tanh(x) ≈ x for tiny x |
| 1.0 | tanh(1) ≈ 0.7616 | mpmath oracle; 1 ULP |

---

## sinh(x)

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | +0 | exact |
| -0 | -0 | exact |
| +∞ | +∞ | exact |
| -∞ | -∞ | exact |
| NaN | NaN | propagates |
| huge | +∞ (overflow) | overflow |
| sub | ≈ sub | sinh(x) ≈ x for tiny x |
| 710 | +∞ | overflow (sinh grows as e^x/2) |
| -710 | -∞ | overflow |
| 1.0 | sinh(1) ≈ 1.1752 | mpmath oracle; 1 ULP |

---

## cosh(x)

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | 1.0 | exact |
| -0 | 1.0 | exact (even function) |
| +∞ | +∞ | exact |
| -∞ | +∞ | exact (even: both ±∞ give +∞) |
| NaN | NaN | propagates |
| huge | +∞ | overflow |
| sub | ≈ 1.0 | cosh(x) ≈ 1 + x²/2 ≈ 1 |
| 710 | +∞ | overflow |
| 1.0 | cosh(1) ≈ 1.5431 | mpmath oracle; 1 ULP |

---

## atan(x) — arctangent

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | +0 | exact |
| -0 | -0 | exact |
| +∞ | π/2 | ≈ 1.5707963... — mpmath oracle |
| -∞ | -π/2 | mpmath oracle |
| NaN | NaN | propagates |
| 1.0 | π/4 ≈ 0.7854 | mpmath oracle; 1 ULP |
| -1.0 | -π/4 | mpmath oracle |
| sub | ≈ sub | atan(x) ≈ x for tiny x |
| huge | ≈ π/2 | mpmath oracle |

---

## asin(x) — arcsine

| Input | Mandated result | Notes |
|-------|----------------|-------|
| +0 | +0 | exact |
| -0 | -0 | exact |
| 1.0 | π/2 | mpmath oracle |
| -1.0 | -π/2 | mpmath oracle |
| 1.0 + ε | NaN | domain: |x| > 1 is invalid |
| -1.0 - ε | NaN | domain error |
| +∞ | NaN | outside domain |
| -∞ | NaN | outside domain |
| NaN | NaN | propagates |
| sub | ≈ sub | asin(x) ≈ x for tiny x |
| 0.5 | π/6 ≈ 0.5236 | mpmath oracle; 1 ULP |

---

## acos(x) — arccosine

| Input | Mandated result | Notes |
|-------|----------------|-------|
| 1.0 | +0 | exact |
| -1.0 | π ≈ 3.14159 | mpmath oracle |
| +0 | π/2 | mpmath oracle |
| -0 | π/2 | mpmath oracle |
| 1.0 + ε | NaN | domain error |
| +∞ | NaN | domain error |
| -∞ | NaN | domain error |
| NaN | NaN | propagates |
| 0.5 | π/3 ≈ 1.0472 | mpmath oracle; 1 ULP |

---

## atan2(y, x) — two-argument arctangent

| Input (y, x) | Mandated result | Notes |
|-------------|----------------|-------|
| (+0, +0) | +0 | exact — IEEE 754 |
| (-0, +0) | -0 | exact |
| (+0, -0) | +π | mpmath oracle |
| (-0, -0) | -π | mpmath oracle |
| (+∞, +∞) | π/4 | mpmath oracle |
| (+∞, -∞) | 3π/4 | mpmath oracle |
| (-∞, +∞) | -π/4 | mpmath oracle |
| (-∞, -∞) | -3π/4 | mpmath oracle |
| (NaN, anything) | NaN | propagates |
| (anything, NaN) | NaN | propagates |
| (1.0, +0) | π/2 | mpmath oracle |
| (1.0, -0) | π/2 | mpmath oracle |
| (-1.0, +0) | -π/2 | mpmath oracle |
| (+0, +1) | +0 | exact |
| (-0, +1) | -0 | exact |
| (+0, -1) | +π | mpmath oracle |
| (-0, -1) | -π | mpmath oracle |
| (sub, huge) | ≈ sub/huge | tiny angle; full precision |

---

## Cross-function critical cases

These inputs stress multiple functions simultaneously and must be tested when a libm
pipeline (e.g., `pow(x, y) = exp(y * ln(x))`) is implemented.

| Input | Function | Danger |
|-------|----------|--------|
| exp(-1e-15) | exp | 1 - exp(-ε) loses precision vs 1 - (1 - ε) = ε |
| ln(1 + 1e-15) | ln | same: ln(1+x) ≈ x but computed naively loses precision |
| sin(π - 1e-15) | sin | near reflection symmetry: tiny argument after reduction |
| cos(π/2 - 1e-15) | cos | near the zero: result is tiny |
| pow(1 + 1e-15, 1e15) | pow | limit definition of e; result ≈ e |
| atan2(1e-300, 1) | atan2 | subnormal y with normal x |
| sinh(1e-15) | sinh | = (exp(x) - exp(-x))/2 ≈ x; catastrophic cancellation if computed naively |

---

## Footnotes

**[note 1] Subnormal exp output.** When x is in the range [-745, -708], exp(x) is
subnormal. The implementation must not flush this to zero. Test: `exp(-745.0)` should
return a positive subnormal, not +0.

**[note 2] Critical range-reduction test.** The Cody-Waite reduction for exp uses
ln(2) as a reduction constant. If the implementation uses a slightly wrong ln(2),
the argument `ln(2)` itself will not produce exactly 2.0 — it will produce 2.0 ± 1 ULP.
This is the central test for Cody-Waite coefficient correctness. The ULP distance
from 2.0 should be 0 or 1, never more.

**[note 3] ln of negative.** The C standard mandates NaN and sets `errno = EDOM`.
tambear policy: return NaN, no errno (we don't use global state). The NaN payload
is unspecified; only `is_nan()` is tested.

**[note 4] ln of subnormal.** The result is a large negative finite number, not -∞.
Specifically, `ln(MIN_POSITIVE / 2) ≈ -708.4 - ln(2)`. This requires the implementation
to handle the subnormal exponent correctly; an implementation that extracts the biased
exponent without adjusting for subnormals will return -∞ instead of a finite result.

**[note 5] ln(1 + ε) catastrophic cancellation.** The naive formula `log(1+x)` for x
near 0 computes `1+x` first (losing bits of x to rounding) and then takes the log,
losing more bits in the argument subtraction. The correct implementation uses `log1p`
or its equivalent. tambear's `ln` should be accurate to 1 ULP even for `x = 1e-15`,
which requires argument conditioning — not naive `log(1+x)`. This is a separate
algorithm requirement that must be in the design doc.

**[note 6] sqrt of subnormal.** IEEE 754 mandates correctly-rounded sqrt for all
inputs including subnormals. An implementation that uses a Newton-Raphson iteration
initialized from the biased exponent will get the exponent wrong for subnormals. The
fix is to rescale subnormal inputs: multiply by 2^104 (bringing into normal range),
compute sqrt, divide by 2^52. This technique must be explicitly mentioned in the
design doc.

**[note 7] sin(π) and cos(π/2) are not exactly 0.** The f64 representation of π
is not exactly π. `f64::consts::PI` = 3.141592653589793 (rounded), which differs
from the true π by approximately 1.2e-16. So `sin(f64::consts::PI)` ≠ 0 — it equals
approximately 1.2e-16, the difference propagated through the Taylor expansion near 0.
Similarly, `cos(f64::consts::PI / 2)` is not exactly 0. A test must use the mpmath
oracle with the exact f64 input value, not the "should be 0" intuition.

**[note 8] sin/cos of huge arguments.** For `x > 2^53`, the Cody-Waite range reduction
(which uses a precomputed table of π/2 bits) runs out of table bits. For genuinely huge
arguments (x > 1e20), accurate computation requires a Payne-Hanek extended-precision
reduction. Whether tambear-libm's Phase 1 `sin` supports this is a design decision that
must be documented in the accuracy target (campsite 2.1). If not supported, the implementation
must return NaN or raise an error for arguments exceeding the supported range, rather than
silently returning garbage.

**[note 9] tan(π/2).** The f64 representation of π/2 is slightly below the true π/2
(because f64 π rounds down). So `tan(f64_pi_over_2)` is a very large positive number
(approximately 1.633e16), not ±∞. An mpmath oracle at the exact f64 input value will
confirm this. A test that expects ±∞ is wrong.

---

## ULP tolerance policy

Per `tambear-tam-test-harness::ToleranceSpec`:

| Function | Max ULP error (Phase 1) | Notes |
|----------|------------------------|-------|
| `sqrt` | 0 (bit-exact) | IEEE 754 mandates correctly-rounded sqrt |
| `exp` | 1 ULP | 1-ULP faithful rounding is the accuracy target |
| `ln` | 1 ULP | same |
| `sin` | 1 ULP | same |
| `cos` | 1 ULP | same |
| `tan` | 2 ULP | tan near π/2 requires extra care |
| `atan` | 1 ULP | |
| `asin` | 1 ULP | |
| `acos` | 1 ULP | |
| `atan2` | 1 ULP | |
| `sinh` | 1 ULP | |
| `cosh` | 1 ULP | |
| `tanh` | 1 ULP | |
| `pow` | 2 ULP | computed as exp(y*ln(x)); two rounding points |

**These tolerances are the acceptance criteria** — not starting points to negotiate down
from. If a function cannot meet its tolerance across the entire input domain, the design
must change (better coefficients, more reduction steps, Payne-Hanek for large args).

The ULP bounds here must be referenced when calling `ToleranceSpec::within_ulp(bound)`
in test code. Never guess the bound; cite this table.
