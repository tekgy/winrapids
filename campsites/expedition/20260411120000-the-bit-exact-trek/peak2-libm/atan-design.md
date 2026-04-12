# `tam_atan`, `tam_asin`, `tam_acos`, `tam_atan2` — Algorithm Design Document

**Campsites 2.19, 2.20.** Inverse trig functions. Atan is the base primitive; asin, acos, and atan2 compose on top.

**Owner:** math-researcher
**Status:** draft, awaiting navigator + pathmaker review
**Date:** 2026-04-11 (amended 2026-04-12 with stub-vs-composition clarification)

---

## IR stub layout for this function family

Scout (check-ins.md, 2026-04-12) noted that the Phase 1 `.tam` spec declares only five transcendental stubs: `tam_exp`, `tam_ln`, `tam_sin`, `tam_cos`, `tam_pow`. The inverse-trig family needs clarification:

| Function | Mechanism | Stub needed? |
|---|---|---|
| `tam_atan` | **new IR stub** | **YES** — it is a base primitive called by the other three |
| `tam_asin` | pure `.tam` function calling `tam_atan` + `fsqrt.f64` | No |
| `tam_acos` | pure `.tam` function calling `tam_atan` + `fsqrt.f64` | No |
| `tam_atan2` | pure `.tam` function calling `tam_atan` | No |

**Rationale.** The reason a function gets a dedicated stub at the IR level is so that backends *could* inline an optimized implementation if they had one (e.g., dedicated hardware `atan`). In Phase 1 every backend runs our `.tam` libm text, so there's no inlining benefit. But we still want `tam_atan` as a stub so that the translator sees a single named opcode and does not have to untangle mutual recursion between `tam_atan`, `tam_asin`, `tam_acos`, `tam_atan2`. `tam_asin`/`tam_acos`/`tam_atan2` are regular `.tam` functions in the libm text that call `tam_atan`. No cycles, no name resolution drama.

**Net ask of pathmaker:** add **one** new stub declaration (`tam_atan`) to the Phase 1 spec. No new stubs for asin/acos/atan2.

---

## Why `atan` is the base

`atan(x)` is defined for all real `x` and returns a value in `(-π/2, π/2)`. It has no singularities on the real line, a well-behaved Taylor series, and a natural argument-reduction scheme based on the identity
```
atan(1/x) = π/2 - atan(x)   for x > 0
```
which reduces the domain to `[-1, 1]` (or `[0, 1]` by exploiting odd symmetry).

`asin` and `acos` have singularities at `±1`, so we prefer to compose them from `atan` and `sqrt` via identities:
```
asin(x) = atan(x / sqrt(1 - x^2))
acos(x) = π/2 - asin(x)   or   atan2(sqrt(1-x^2), x)
```

`atan2(y, x)` is `atan(y/x)` with quadrant fixup — the "real" atan that knows which half-plane you're in. It's the most-used variant in practice.

Implementing `atan` correctly and then composing is both simpler and more accurate than implementing each independently. (This is the Cody-Waite approach and also the approach in Muller HFPA.)

## `tam_atan(x)` — the algorithm

Given `x : f64`, return `atan(x) : f64` with:
- `max_ulp ≤ 1.0` across 1M random samples, exponent-uniform on `[-fp64_max, fp64_max]`.
- `atan(0) = 0` bit-exact, `atan(-0) = -0` bit-exact.
- `atan(±inf) = ±π/2`.
- `atan(nan) = nan`.

### Range reduction

Reduce `|x|` to `[0, 1]`:
```
if |x| > 1:
    use_reciprocal = True
    x_reduced = 1 / |x|
else:
    use_reciprocal = False
    x_reduced = |x|
```

Further reduce `[0, 1]` to `[0, tan(π/8)] ≈ [0, 0.4142]`:
```
tan_pi_8 = sqrt(2) - 1   ≈ 0.4142135623730950
if x_reduced > tan_pi_8:
    use_shift = True
    # atan(x) = π/4 + atan((x - 1) / (x + 1))
    x_final = (x_reduced - 1) / (x_reduced + 1)
else:
    use_shift = False
    x_final = x_reduced
```

Now `|x_final| ≤ tan(π/8) ≈ 0.4142`, which is a comfortable interval for a polynomial fit.

### Polynomial

Remez-minimax fit `(atan(x) - x) / x^3` on `[-tan(π/8), tan(π/8)]`. Degree 8 in `x²` (effective degree 16 in `x`) is enough for 1 ULP. The polynomial is even in `x` (since `atan` is odd and we divided by `x³`), so only even powers of `x` (which are `x²`, `x⁴`, ...) contribute. Let `Q(u)` be the polynomial in `u = x²`:

```
atan(x) ≈ x + x^3 * Q(x^2)
        = x + x * x^2 * Q(x^2)
```

Horner, no FMA, same as exp/log/sin/cos.

### Reassembly

```
poly = x_final + x_final * x_final^2 * Q(x_final^2)

if use_shift:
    poly = π/4 + poly
if use_reciprocal:
    poly = π/2 - poly
# restore sign from original x
if x < 0:
    poly = -poly
return poly
```

The constants `π/4` and `π/2` are Cody-Waite-split for accuracy in the critical regime (`x` near 1, where subtraction cancellation bites):
```
pi_over_2_hi = ...
pi_over_2_lo = ...
pi_over_4_hi = pi_over_2_hi / 2
pi_over_4_lo = pi_over_2_lo / 2
```

Reassembly then becomes:
```
poly_hi = pi_over_4_hi + poly
poly_lo = pi_over_4_lo
result_hi = pi_over_2_hi - poly_hi
result_lo = pi_over_2_lo - poly_lo
...
```

The exact order of operations matters and is documented in Muller HFPA §11. Pathmaker follows that recipe.

### Special-value handling

```
if isnan(x):        return x
if x == +0.0:       return +0.0
if x == -0.0:       return -0.0
if x == +inf:       return pi_over_2_as_fp64
if x == -inf:       return -pi_over_2_as_fp64
# otherwise: reduce and polynomial path
```

## `tam_asin(x)` via identity

```
asin(x) = atan(x / sqrt(1 - x^2))
```

Domain `[-1, 1]`. At `|x| = 1`, `sqrt(1 - x²) = 0` and the division overflows to `+inf`, then `atan(+inf) = π/2` — correct limit. So the identity naturally handles the endpoint.

However, the division is a source of precision loss: for `|x|` near 1, `1 - x²` loses bits catastrophically. Better identity:
```
asin(x) = atan2(x, sqrt((1 - x) * (1 + x)))
```
The factored `(1 - x) * (1 + x)` avoids the cancellation because no single subtraction is required — `1 - x` and `1 + x` are each computed fresh, and their product has full precision.

For `|x| near 1`, an alternative is a direct polynomial on `asin`, but the factored-sqrt identity is sufficient for 1 ULP.

### Special values

```
if x > 1 or x < -1:  return nan
if x == 0:            return 0
if x == 1:            return pi_over_2_as_fp64
if x == -1:           return -pi_over_2_as_fp64
```

## `tam_acos(x)` via identity

```
acos(x) = π/2 - asin(x)                (direct subtraction — loses precision near x=1)
```
OR, better near `x = 1`:
```
acos(x) = 2 * atan2(sqrt((1 - x) / 2), sqrt((1 + x) / 2))
```

Implementing both and switching based on `|x|` vs threshold is the standard trick. For `|x| < 0.5`, use the simple `π/2 - asin(x)`. For `|x| ≥ 0.5`, use the `atan2(sqrt)` form.

### Special values

```
if x > 1 or x < -1:  return nan
if x == 1:            return 0
if x == -1:           return pi_as_fp64
if x == 0:            return pi_over_2_as_fp64
```

## `tam_atan2(y, x)` — the quadrant-aware atan

```
atan2(y, x) computes atan(y/x) but uses the signs of y and x to return
a result in (-π, π] — the angle in the plane from the positive x-axis
to the point (x, y).
```

Quadrant dispatch:

| `x` | `y` | Result |
|---|---|---|
| `x > 0` | any | `atan(y / x)` |
| `x < 0` | `y ≥ 0` | `atan(y / x) + π` |
| `x < 0` | `y < 0` | `atan(y / x) - π` |
| `x == 0` | `y > 0` | `+π/2` |
| `x == 0` | `y < 0` | `-π/2` |
| `x == 0` | `y == 0` | `+0` (by convention) |

Plus a large table of inf/nan cases:
```
atan2(+0, +0) = +0
atan2(-0, +0) = -0
atan2(+0, -0) = +π
atan2(-0, -0) = -π
atan2(y, +0)   for y > 0 = +π/2
atan2(y, -0)   for y > 0 = +π/2   (yes, `+π/2` not `-π/2` — signed zero doesn't swap sign of the result when x has magnitude 0)
atan2(+0, x)   for x > 0 = +0
atan2(-0, x)   for x > 0 = -0
atan2(+0, x)   for x < 0 = +π
atan2(-0, x)   for x < 0 = -π
atan2(±inf, +inf) = ±π/4
atan2(±inf, -inf) = ±3π/4
atan2(±inf, finite) = ±π/2
atan2(finite, +inf) = ±0 (sign of y)
atan2(finite, -inf) = ±π (sign of y)
atan2(nan, x)   = nan
atan2(y, nan)    = nan
```

This is the biggest special-case table in Phase 1, second only to `pow`. Pathmaker writes a front-end that handles each explicitly. The real-valued path underneath is `atan(y/x)` with the quadrant correction from the table.

### Precision note

For the `atan(y/x)` path, the division `y/x` can introduce a 0.5 ULP error, which composes with `atan`'s 1 ULP to give up to 1.5 ULPs overall. This exceeds our 1 ULP target. The fix: compute `atan(y/x)` without forming the quotient explicitly by instead fitting the 2-argument atan polynomial directly. Or: carry `y/x` as double-double for the critical intermediate.

**Recommendation for Phase 1:** accept a 2-ULP bound for `atan2` (relaxing from 1 ULP) and document this as a per-function deviation. Or: use double-double for the `y/x` intermediate. Navigator should pick.

## Pitfalls

1. **Precision at `|x| ≈ 1` in the range-reduction shift.** `(x - 1)/(x + 1)` has catastrophic cancellation when `x` is very close to `1` but in a subtle way — `x - 1` is small, `x + 1 ≈ 2`, the ratio is small and well-conditioned as long as `x - 1` is computed accurately. Sterbenz applies since `x ≥ tan(π/8) > 0.5`. Safe.
2. **Subnormal near x = 0.** For `|x| < 2^-500`, `atan(x) ≈ x` to full precision. The polynomial `x + x^3 * Q` handles this naturally since the `x³` term is negligible; we just return `x` to 1 ULP.
3. **Odd symmetry of atan.** `atan(-x) = -atan(x)` bit-exact. The polynomial form preserves this if we restore the sign at the end.
4. **`atan2` quadrant table.** Multiple sources conflict on the signed-zero convention. IEEE 754 and the C standard specify it as above; follow that.
5. **`asin(x)` near `|x| = 1`.** The factored `(1-x)(1+x)` form is critical.
6. **`acos(x)` precision near `x = 1`.** The direct `π/2 - asin(x)` form loses bits; the `2 * atan2(sqrt, sqrt)` form is mandatory for `|x| ≥ 0.5`.

## Testing

- `atan`: 1M random samples, exponent-uniform on `[-fp64_max, fp64_max]`. 1 ULP.
- `asin`: 1M random samples, real-uniform on `[-1, 1]`. 1 ULP.
- `acos`: same, 1 ULP.
- `atan2`: 1M random samples, real-uniform on `(y, x) ∈ [-10, 10]²`, plus 2M samples hitting every signed-zero / inf / nan corner of the table. 1-2 ULP (see precision note).
- Identity: `atan(-x) = -atan(x)` bit-exact (for `atan` itself).
- Identity: `atan(x) + atan(1/x) = π/2` for `x > 0`, within 2 ULP (composed).
- Identity: `sin(atan(x)) = x / sqrt(1 + x²)`, within 3 ULP (triple composition).
- Identity: `atan2(sin(θ), cos(θ)) = θ` for `θ ∈ (-π, π]`, within 3 ULP.

## Open questions

1. **Is `atan2` at 2 ULP acceptable, or do we enforce 1 ULP via double-double?** I recommend 2 ULP + document for Phase 1. Navigator decides.
2. **Direct `asin`/`acos` polynomials vs identity compositions?** Identity is simpler and 1 ULP is reachable for `asin`/`acos` via the factored-sqrt form. No reason to implement direct polynomials in Phase 1.

## References

- J.-M. Muller et al., "Handbook of Floating-Point Arithmetic," 2nd ed., 2018, Chapter 11.
- W. J. Cody & W. Waite, "Software Manual for the Elementary Functions," Prentice-Hall, 1980, §§6–7 (atan, asin, acos).
- IEEE 754-2019, §9 (recommended operations, including atan2 corner cases).
