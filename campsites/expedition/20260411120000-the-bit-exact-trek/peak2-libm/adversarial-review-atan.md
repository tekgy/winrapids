# Adversarial Pre-Code Review — `atan-design.md`

**Reviewer:** Adversarial Mathematician
**Date:** 2026-04-12
**Status:** THREE BLOCKING ISSUES. THREE advisory notes. Implementation must not start (Campsites 2.19, 2.20) until resolved.

---

## Summary verdict

The atan/asin/acos/atan2 design is the most polished of the Phase 1 set. The range-reduction cascade is correct, the signed-zero handling is explicitly thought through, and the `(1-x)(1+x)` factored form for asin near ±1 is the right fix. However, three blocking issues: a precision hole in the atan reassembly step (the Cody-Waite π-constant splitting is invoked but not specified), a specific error in the atan2 signed-zero table, and a missing check for `atan2(y, x)` where both arguments are zero.

---

## Blocking issues

### B1 — π/4 and π/2 Cody-Waite split: invoked but not specified

The reassembly section says:

```python
poly_hi = pi_over_4_hi + poly
poly_lo = pi_over_4_lo
result_hi = pi_over_2_hi - poly_hi
result_lo = pi_over_2_lo - poly_lo
...
```

And then: "The exact order of operations matters and is documented in Muller HFPA §11. Pathmaker follows that recipe."

This is insufficient for two reasons:

**First:** "Pathmaker follows that recipe" means the design doc has outsourced its precision specification to a reference that may not be available at implementation time. For exp and log, the design docs (with blocking issues filed) at least attempted to specify the exact op sequence. For atan, the most critical reassembly step — where `π/4` is added and where `π/2` is subtracted — is left entirely to the external reference. This is the same class of issue as log B3 (reassembly deferred to 2.11 hand-off).

**Second:** the reassembly above is WRONG as written. The variables `poly_hi` and `result_hi` are never combined into a final result. The code snippet shows two `hi` values and two `lo` values, but no TwoSum renormalization and no final fp64 result. A pathmaker reading this would not know how to combine them.

For atan near `x = tan(π/8)` (the shift boundary), the reduction is:
- `x_final = (x_reduced - 1) / (x_reduced + 1)` — this is near 0 by construction
- `poly = x_final + x_final^3 * Q(x_final^2)` — near 0
- `poly_hi = π/4_hi + poly` — this is the key step; it adds π/4 ≈ 0.785 to a small number, so there's no catastrophic cancellation here

For atan near `x → ∞` (the reciprocal path):
- `poly = atan(x_final)` for `x_final ≈ 0` (small)
- `poly` is small
- `result = π/2_hi - poly_hi - π/2_lo + poly_lo` — here we subtract a small `poly` from `π/2`. Cancellation occurs.

The Sterbenz-safe condition: `π/2_hi - poly_hi` where `poly_hi ≈ π/4_hi + ε` for some small `ε`. So `π/2_hi - (π/4_hi + ε) = π/4_hi - ε`. No cancellation. But wait: when `poly_hi` is large (approaching `π/2`), then `π/2_hi - poly_hi` can be nearly zero — that IS cancellation. This happens when `atan(x_final)` is near `π/2`, which happens when `x_final` is near 1. But `x_final ≤ tan(π/8) ≈ 0.414`, so `atan(x_final) ≤ π/8`. After the `if use_shift:` correction, `poly_hi ≤ π/4 + π/8 = 3π/8`. Then `π/2 - 3π/8 = π/8` — no catastrophic cancellation.

Actually the cancellation does occur in `atan(x)` for `x` near 1 (which maps to `x_final ≈ 0` after the shift). The result near `x = 1` is `π/4 + atan(x_final)` ≈ `π/4 + 0 = π/4`. No subtraction, so no cancellation. The subtraction only happens in the reciprocal path. And in the reciprocal path for large `x`, `x_final ≈ 1/x ≈ 0`, so `atan(x_final) ≈ 0`, and `π/2 - π/4 - 0 ≈ π/4`. Again no cancellation.

The one case where cancellation bites: `x` slightly above 1, going through the reciprocal+shift path. `x_reduced = 1/x ≈ 1 - ε`, then `x_final = (x_reduced - 1)/(x_reduced + 1) ≈ -ε/2`. Then `poly ≈ -ε/2`, `poly_hi = π/4_hi - ε/2`. Then `result = π/2_hi - (π/4_hi - ε/2) = π/4_hi + ε/2`. The subtraction `π/2_hi - π/4_hi = π/4_hi` is exact (same exponent range, one bit of cancellation at most). Fine.

After analysis: the reassembly is not catastrophically ill-conditioned, but the constants `pi_over_4_hi`, `pi_over_4_lo`, `pi_over_2_hi`, `pi_over_2_lo` are not specified. Without knowing the bit patterns of these constants, an implementer cannot verify that the reassembly achieves 1 ULP.

**Required fix:** Specify the exact hex bit patterns for the Cody-Waite split constants:
```
pi_over_2_hi = 0x3FF921FB54442D18  (1.5707963267948966...)
pi_over_2_lo = 0x3C91A62633145C07  (6.123233995736766e-17)
pi_over_4_hi = 0x3FE921FB54442D18  (0.7853981633974483...)
pi_over_4_lo = 0x3C81A62633145C06  (3.061616997868383e-17)
```
(These are the standard Cody-Waite split values from Kahan; verify against Muller HFPA §11 before committing.)

Also provide the complete final reassembly op sequence, not just intermediate variables.

---

### B2 — `atan2` signed-zero table: one entry conflicts with IEEE 754

The atan2 special-value table includes:

```
atan2(y, -0)   for y > 0 = +π/2   (yes, `+π/2` not `-π/2` — signed zero doesn't swap sign of the result when x has magnitude 0)
```

This is CORRECT per IEEE 754-2019 §9.2.1: `atan2(y, ±0)` for `y > 0` is `+π/2`. The sign of `x = ±0` does not affect the result when `y > 0`. The comment in the design doc is also correct.

BUT: immediately after this, the table has:

```
atan2(+0, x)   for x > 0 = +0
atan2(-0, x)   for x > 0 = -0
atan2(+0, x)   for x < 0 = +π
atan2(-0, x)   for x < 0 = -π
```

IEEE 754-2019 §9.2.1: `atan2(±0, x)` for `x < 0` returns `±π`. The design doc says `atan2(+0, x < 0) = +π` and `atan2(-0, x < 0) = -π`. This is CORRECT.

The WRONG entry: the quadrant dispatch table above the special-value table says:

```
x == 0   y == 0   Result: +0 (by convention)
```

This is wrong. IEEE 754-2019 §9.2.1 specifies ALL four signed-zero combinations for `atan2(±0, ±0)`:
- `atan2(+0, +0) = +0`
- `atan2(-0, +0) = -0`
- `atan2(+0, -0) = +π`
- `atan2(-0, -0) = -π`

The design doc collapses all four into `+0 (by convention)`. This is wrong for three of the four cases. The special-value table below the quadrant table does specify these correctly via the rows `atan2(+0, +0) = +0`, `atan2(-0, +0) = -0`, `atan2(+0, -0) = +π`, `atan2(-0, -0) = -π`. But the quadrant dispatch table contradicts this.

**Required fix:** Remove the row `x == 0, y == 0, Result: +0 (by convention)` from the quadrant dispatch table. Replace with: "For `x == 0` and `y == 0`: see the special-value table — all four signed-zero combinations are distinct cases. The result is determined by the signs of both zeros."

---

### B3 — `atan2(y, x)` when both arguments are ±0: dispatch order matters

Related to B2: the front-end dispatch for `atan2` must check BOTH `x == 0` and `y == 0` together, because the behavior depends on both simultaneously. But the design doc's quadrant dispatch table handles `x == 0` and `y == 0` as separate cases — the `x == 0, y > 0` row, the `x == 0, y < 0` row, and the `x == 0, y == 0` row.

If the dispatch checks `x == 0` first, then for `y == 0` it falls into the `y == 0` sub-case of the `x == 0` branch. If the dispatch checks `y == 0` first (returning `±0` for the `y = +0` case and `±π` for the `y = -0` case based on `x`'s sign), then `x == 0` never has a special case for the both-zero scenario.

The design doc does not specify the dispatch order between:
- `atan2(±0, ±0)` (both zero, sign-sensitive)
- `atan2(y, ±0)` (x is zero, y is nonzero)
- `atan2(±0, x)` (y is zero, x is nonzero)

Without the dispatch order, two implementers will produce different code and one will be wrong.

**Required fix:** Provide a complete ordered dispatch sequence for the `atan2` front-end:

```
1. if isnan(y) or isnan(x): return nan
2. if isinf(y) and isinf(x):
       return ±π/4 or ±3π/4 based on signs
3. if isinf(y): return ±π/2 (sign of y)
4. if isinf(x) and x > 0: return copysign(0.0, y)
5. if isinf(x) and x < 0: return copysign(π, y)
6. # Now y and x are both finite
7. if y == 0 and x == 0: dispatch on sign bits via bitcast
8. if y == 0 and x > 0: return copysign(0.0, y)
9. if y == 0 and x < 0: return copysign(π, y)
10. if x == 0: return copysign(π/2, y)
11. # General case: atan(y/x) ± π based on x sign
```

Note that step 7 handles the both-zero case before the single-zero cases. This order is critical: `y == 0 and x == 0` must be caught BEFORE `y == 0 and x > 0`, because in IEEE 754, `x == 0` evaluates to true even when `x = -0.0`, so step 8 would misfire for `atan2(+0, -0)` without step 7's guard.

---

## Advisory notes

### A1 — `asin(x)` near `|x| = 1`: the two-ULP concern is real

The design doc correctly uses `atan2(x, sqrt((1-x)*(1+x)))` for `asin`. The concern: for `x = 1 - ε` with `ε` small, `(1-x)*(1+x) = ε * (2-ε) ≈ 2ε`. The computation `sqrt(2ε)` introduces 0.5 ULP (IEEE 754 sqrt is correctly-rounded). Then `atan2(x, sqrt(2ε))` for `x ≈ 1` and `sqrt(2ε) ≈ 0`: this is `π/2 - atan(sqrt(2ε)/x)` ≈ `π/2 - sqrt(2ε)`. The subtraction `π/2 - sqrt(2ε)` has cancellation when `sqrt(2ε)` is close to `π/2`, which happens when `ε ≈ π²/8 ≈ 1.23`. So there's no cancellation risk for `x` near 1 specifically.

For `x` exactly equal to `1`, `(1-x)*(1+x) = 0 * 2 = 0`, `sqrt(0) = 0`, `atan2(1, 0) = π/2`. Correct. The special-value check `if x == 1: return pi_over_2_as_fp64` is redundant but harmless.

However: for `x = 1 - 2^{-52}` (the largest fp64 below 1), `1 - x = 2^{-52}` exactly (Sterbenz), `1 + x ≈ 2`, `(1-x)*(1+x) ≈ 2^{-51}`. Then `sqrt(2^{-51}) = 2^{-25.5}` (not exactly representable, rounded to nearest). The subsequent `atan2` call gets an argument `sqrt((1-x)(1+x)) ≈ 2^{-25.5}` with up to 0.5 ULP error. Then `atan(sqrt(·)/1) ≈ sqrt(·)` (since the argument is small), so `asin(x) ≈ π/2 - 2^{-25.5}`. The 0.5 ULP error in `sqrt` propagates directly to the final result. Total error ≤ 1 + 0.5 + 1 (from atan) = 2.5 ULPs.

The design claims 1 ULP for `asin`. This may not hold for inputs very near 1 via this path.

**Advisory:** Add a near-unity special case: for `x` such that `1 - x < 2^{-26}` (i.e., `x > 1 - 2^{-26}`), use a direct polynomial fit to `asin(x)` near 1 (via the identity `asin(x) = π/2 - asin(sqrt((1-x)/2))` and the polynomial for asin near 0). This is the standard trick. Alternatively, document that the 1-ULP bound for `asin` only holds away from `|x| = 1` (say `|x| < 1 - 2^{-26}`), and add a test that measures the actual ULP at `x = 1 - 2^{-52}`.

### A2 — atan2 2 ULP deviation: navigator decision required

The design doc says:

> "For Phase 1: accept a 2-ULP bound for `atan2` (relaxing from 1 ULP) and document this as a per-function deviation. Or: use double-double for the `y/x` intermediate. Navigator should pick."

This is correctly flagged as an open decision requiring navigator input. The adversarial position:

- The 2-ULP bound for `atan2` is reasonable IF the underlying `atan` is 1 ULP and the division `y/x` adds 0.5 ULP. Total: 1.5 ULP, which rounds up to 2 ULP worst case.
- But the `accuracy-target.md` specifies 1 ULP for all Phase 1 functions. Relaxing to 2 ULP for `atan2` requires an explicit navigator exception in `accuracy-target.md`.
- The double-double approach for `y/x` is correct but adds complexity and depends on the double-double infrastructure from `pow`. If `pow`'s double-double is implemented first (as planned), reusing it for `atan2` is straightforward.

**Advisory:** Navigator should record the decision explicitly in `accuracy-target.md` as: "atan2: 2 ULP for Phase 1, pending double-double integration in Phase 2." Do not leave this as an open question in the design doc.

### A3 — TMD corpus for atan family

Known hard cases:
- `atan(1.0)` — true value `π/4 ≈ 0.7853981633974483096...`. The nearest fp64 is `0x3FE921FB54442D18`, which is an exact boundary case.
- `atan(tan(π/4 + 2^{-52}))` — testing the shift boundary.
- `asin(0.5)` — true value `π/6 ≈ 0.5235987755982988730...`. Known TMD candidate.
- `acos(0.5)` — true value `π/3 ≈ 1.0471975511965977461...`. Known TMD candidate.
- `atan2(1.0, 1.0)` — should equal `atan(1.0) = π/4` bit-exact.

Log in `peak2-libm/tmd-corpus/atan-tmd.md`.

---

## Required additions to test battery

1. **All four `atan2(±0, ±0)` cases bit-exact:** `atan2(+0, +0) = +0`, `atan2(-0, +0) = -0`, `atan2(+0, -0) = +π`, `atan2(-0, -0) = -π`.
2. **Cody-Waite constants:** verify `pi_over_2_hi + pi_over_2_lo == pi_over_2_to_200_digits_rounded_to_f64` to within 2 ULPs of 200-digit π/2.
3. **atan signed-zero:** `atan(+0.0).to_bits() == (0.0f64).to_bits()`, `atan(-0.0).to_bits() == (-0.0f64).to_bits()`.
4. **asin near ±1:** measure ULP at `x ∈ {1 - 2^{-52}, 1 - 2^{-26}, 1 - 2^{-10}}`. Document actual max ULP for the near-unity regime.
5. **acos near x = 0:** `acos(0.0) = π/2` to 1 ULP.
6. **Identity: atan(x) + atan(1/x) = π/2** for 1000 positive `x` values, within 2 ULPs.
7. **TMD candidates** listed above.
8. **atan2 with both-inf:** `atan2(+inf, +inf) = π/4`, `atan2(+inf, -inf) = 3π/4`, `atan2(-inf, +inf) = -π/4`, `atan2(-inf, -inf) = -3π/4`, all within 1 ULP.
9. **NaN cases:** `atan(nan) = nan`, `atan2(nan, 1.0) = nan`, `atan2(1.0, nan) = nan` — NaN propagated, not 0.0.

---

## Verdict

**HOLD on Campsites 2.19 and 2.20 implementation until:**

1. B1 resolved: provide exact hex bit patterns for the Cody-Waite split constants `pi_over_2_hi/lo` and `pi_over_4_hi/lo`, and specify the complete final reassembly op sequence for `atan`.
2. B2 resolved: remove the incorrect `atan2(±0, ±0) = +0` row from the quadrant dispatch table. The special-value table's four-row specification is correct; the quadrant table must not contradict it.
3. B3 resolved: provide the complete ordered dispatch sequence for `atan2`'s front-end, with explicit handling of the both-zero case before the single-zero cases.
