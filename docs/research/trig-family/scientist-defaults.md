# Tambear Trig Family: $1M/yr Scientist Defaults (TRIG-6)

Author: scientist  
Date: 2026-04-13  

The question: for each trig function, what would a deeply-experienced numerical
analyst choose as the default configuration? Not the fdlibm defaults (which
optimize for historical compatibility). Not the "safest" defaults (which would
always be correctly_rounded, which is 3x slower). The defaults a scientist
would want when they open tambear for the first time.

Principles driving the choices:
1. Compensated over strict: the accuracy gap (1 ulp vs 2 ulp worst case) is worth
   ~10% overhead in nearly every use case.
2. Correctly_rounded is for publishing, not for production farms. Default to
   compensated; let users opt into correctly_rounded explicitly.
3. Radians everywhere: this is the convention in scientific computing. Degrees
   and turns are conversions, not defaults.
4. Edge cases on: IEEE 754-2019 special case handling should be on by default.
   Turning it off is an optimization for code that guarantees clean input.
5. sinpi/cospi/tanpi: these exist to get better accuracy near zeros. The whole
   point is better precision. Default to compensated (not strict) because strict
   would throw away the precision gain.

---

## Forward Trig

### sin / cos

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | 1 ulp worst vs 2 ulp; ~10% overhead. Worth it universally. |
| angle_unit      | radians          | Scientific convention. |
| range_reduction | auto             | Let the implementation choose Cody-Waite vs Payne-Hanek. |

### tan

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | Same logic as sin/cos. |
| angle_unit      | radians          | |
| pole_handling   | inf              | Return +inf/-inf at pi/2 + n*pi (IEEE 754). Some users want NaN for "undefined"; expose as option, not default. |

tan has a stronger case for compensated than sin/cos: near-pole arguments
amplify any polynomial error dramatically. A 2-ulp error in the reduced argument
becomes a multi-ulp error in tan near pi/2.

### cot, sec, csc

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | |
| angle_unit      | radians          | |
| pole_handling   | inf              | cot(0)=+inf, sec(pi/2)=+inf, csc(0)=+inf |

cot(x) = cos(x)/sin(x): computed as a ratio. Near zeros of sin(x), the division
amplifies errors. Compensated is non-negotiable here.

sec(x) = 1/cos(x), csc(x) = 1/sin(x): same concern near poles.

### sincos (fused)

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | |
| angle_unit      | radians          | |

The fused pair shares range reduction. This is the natural default for any code
that needs both sin and cos of the same argument — 30-40% faster than calling
both separately.

---

## Pi-scaled

### sinpi(x) = sin(pi*x)

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | See note below. |
| range_reduction | exact            | x mod 2 is exact for |x| < 2^52. Always prefer. |

**Critical note**: sinpi's entire purpose is better accuracy near half-integers.
sinpi(0.5) MUST return exactly 1.0. sinpi(1.0) MUST return exactly 0.0.
sinpi(n) MUST return 0.0 for any integer n.

If we use the "strict" Horner path, we can still guarantee these exact values
by testing x for integer/half-integer before the polynomial. But the compensated
path is still better for the general near-half-integer region.

Default: compensated. Document that correctly_rounded is available for publication.

### cospi(x) = cos(pi*x)

Same analysis as sinpi. cospi(0.5) = 0.0 exactly. cospi(0) = 1.0 exactly.

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | |
| range_reduction | exact            | |

### tanpi(x) = tan(pi*x)

tanpi(0.5) = +inf (pole). tanpi(0) = 0.0. tanpi(0.25) = 1.0.

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | Near-pole amplification makes this non-negotiable. |
| pole_handling   | inf              | |
| range_reduction | exact            | |

---

## Inverse Trig

### asin

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | |
| output_unit     | radians          | |
| near_one_handling | compensated    | asin near x=±1 uses sqrt(1-x^2) which has cancellation. The compensated path uses a stable 2-term formula. |

asin(1.0) = pi/2 exactly (to 1 ulp of float(pi/2)).
asin(-1.0) = -pi/2 exactly.
asin(x) for |x| > 1 = NaN (domain error).

The near-unity issue: asin(x) for x near 1 has the formula
  asin(x) ~= pi/2 - sqrt(2*(1-x))*(1 + ...)
where the sqrt(2*(1-x)) computation is the sensitive part. A 1-ulp error in
(1-x) becomes ~0.5-ulp error in sqrt(1-x) but then 0.5 ulp in the final result
after the pi/2 subtraction with its own rounding. The compensated path handles
this with a careful 2-step evaluation.

### acos

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | |
| output_unit     | radians          | |

acos(1.0) = 0.0 exactly.
acos(-1.0) = pi (to 1 ulp of float(pi)).
acos(x) for |x| > 1 = NaN.

Near x=1: same cancellation as asin. acos(x) = pi/2 - asin(x) — they share
the pathological region.

### atan

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | |
| output_unit     | radians          | |

atan(+inf) = pi/2, atan(-inf) = -pi/2 (not NaN).
atan(0) = 0 exactly.

No near-pole issue — atan is well-conditioned everywhere.

### atan2

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | |
| output_unit     | radians          | |
| ieee754_edges   | on               | The 20+ IEEE 754-2019 special cases for atan2(±0, ±0) etc. MUST be on by default. |

atan2 has more special cases than any other trig function. The correct behavior
per IEEE 754-2019:
- atan2(±0, +0) = ±0
- atan2(±0, -0) = ±pi
- atan2(±0, x<0) = ±pi
- atan2(±0, x>0) = ±0
- atan2(y>0, ±0) = +pi/2
- atan2(y<0, ±0) = -pi/2
- atan2(+inf, +inf) = pi/4
- atan2(+inf, -inf) = 3*pi/4
- atan2(-inf, +inf) = -pi/4
- atan2(-inf, -inf) = -3*pi/4
- atan2(±inf, x finite) = ±pi/2
- atan2(y finite, ±inf) = ±0 or ±pi

There is no reason to ever have ieee754_edges=off as a default. Scientists expect
atan2 to be correct at boundaries.

**Policy gap note**: atan2(0, 0) returns 0 in some C libraries, pi in others.
IEEE 754-2019 specifies atan2(+0, +0) = +0. This is the tambear default.
Document this explicitly so users who hit this edge case understand why.

---

## Hyperbolic

### sinh, cosh, tanh

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | |
| overflow_action | inf              | sinh/cosh overflow at ~710. Return inf (IEEE). |

sinh and cosh share the same intermediate exp(x) computation. This is the
natural shared-pass: compute exp(x) once, then derive sinh=(e^x - e^-x)/2
and cosh=(e^x + e^-x)/2. The shared-pass should be on by default.

tanh(x) = (e^2x - 1)/(e^2x + 1) — numerically stable via expm1 for small x.
Near x=0, use expm1(2x)/(expm1(2x)+2) to avoid catastrophic cancellation.

### asinh, acosh, atanh

| Parameter       | Default          | Rationale |
|-----------------|------------------|-----------|
| precision       | compensated      | |
| near_one_handling | compensated    | atanh near ±1 has cancellation. |

atanh(x) for x near 1: use log1p(2x/(1-x))/2, not log((1+x)/(1-x))/2.
The log1p path avoids cancellation in the numerator.

acosh(x) for x near 1: use sqrt((x-1)*(x+1)) + ... path, not direct cosh^-1.

asinh: well-conditioned away from large x; for large |x|, use log(2*|x|) + sign.

---

## Convergence Check: Defaults Pattern

Looking at the table above:

| Family | precision default | Why not strict | Why not correctly_rounded |
|--------|-------------------|----------------|---------------------------|
| Forward (sin/cos) | compensated | 2 ulp in strict | 3x slower, not needed for farms |
| Forward near-pole (tan/cot/sec/csc) | compensated | near-pole amplification is severe | same |
| Inverse (asin near 1) | compensated | cancellation in sqrt(1-x^2) | same |
| Inverse (atan/atan2) | compensated | well-conditioned but consistent | same |
| Hyperbolic | compensated | expm1 path needs it | same |
| Pi-scaled | compensated | purpose is accuracy; strict undermines it | same |

**The rhyme**: compensated is the universal default across the entire trig family.
The argument is always the same: 10% overhead for 1-ulp improvement.
The only exception would be a function where:
  (a) strict is already provably 1 ulp everywhere, AND
  (b) the compensated path provides no measurable improvement.

No trig function satisfies both (a) and (b). So: compensated everywhere.

**Structural finding**: the trig family forms a closed equivalence class under
the "compensated default" rule. This is not a coincidence — it follows from the
fact that every trig function involves polynomial evaluation over a reduced
domain, and compensated Horner uniformly beats strict Horner on polynomial
domains by 1 ulp at ~10% overhead. The 1 ulp improvement is structural, not
function-specific.

---

## A Note on correctly_rounded

The correctly_rounded path will use double-double precision throughout.
For publication and for correctness proofs, it's the right choice.
For a financial signal farm computing 10^9 trig evaluations per day: no.

Expose it. Document it. Make it easy to use. But don't default to it.

The tambear philosophy (run everything, V columns carry confidence, consumers
decide) applies here: ship all three strategies, document their budgets clearly,
let the user choose. The default is what a scientist would pick blindly for their
first run — compensated.
