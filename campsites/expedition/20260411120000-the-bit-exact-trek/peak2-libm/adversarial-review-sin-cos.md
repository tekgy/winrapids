# Adversarial Pre-Code Review — `sin-cos-design.md`

**Reviewer:** Adversarial Mathematician
**Date:** 2026-04-12
**Status:** TWO BLOCKING ISSUES. THREE advisory notes with one near-blocking.

---

## Summary verdict

The sin/cos design is well-structured. The quadrant dispatch identities are correct, the three-term Cody-Waite recommendation is sound, the polynomial forms are appropriate. Blocking issues are in the special-value handling and the `|x| > 2^20` policy, plus a carried-forward note on ULP tension for tan.

---

## Blocking issues

### B1 — Phase 1 out-of-domain cutoff is inconsistent: `2^20` in body vs `2^30` in dispatch

The "What these functions do" section says:
> "For `|x| > 2^20`: Phase 1 returns `nan`"

The special-value front-end dispatch says:
> "if `|x| > 2^30`: return nan (Phase 1 out-of-domain)"

These conflict. The body defines the domain as `|x| ≤ 2^20`; the special-value section defines it as `|x| ≤ 2^30`. The testing section says "1M in `[-2^30, 2^30]` once the three-term Cody-Waite is in."

The design doc recommends three-term Cody-Waite (which covers `|x| ≤ 2^30`), and the testing section assumes it. But the stated Phase 1 claim (`|x| ≤ 2^20`) is narrower.

**The adversarial concern:** if the three-term Cody-Waite is implemented but the out-of-domain check is at `2^30`, a user with `x = 2^25` gets a number (not nan) and doesn't know whether they're in or out of the claimed domain. Conversely, if the out-of-domain check is at `2^20`, users with `x = 2^25` get nan even though the three-term implementation handles it correctly.

**Required fix:** Pick ONE cutoff and state it consistently everywhere. The recommended choice (implementing three-term and using `2^30` as the cutoff) is fine — but the "What these functions do" section must be updated to match. Or: decide to use two-term (covering `|x| ≤ 2^20`) and update the testing section. The inconsistency must be resolved before implementation.

---

### B2 — `sin(-0.0) = -0.0` bit-exact: sign preservation in the polynomial form

The special-value section states `sin(-0) = -0` bit-exact. The design correctly notes "the polynomial form preserves this automatically because each term is an odd power of r." But the front-end dispatch says:

```
if x == -0.0:     sin: return -0.0
```

This check `x == -0.0` — how is it implemented in `.tam` IR? If using `fcmp_eq.f64(x, 0.0)`, it returns `true` for both `+0.0` and `-0.0`. The front-end check cannot distinguish them.

The design doc says "Front-end dispatch" handles `-0.0` → `-0.0` explicitly. But the `.tam` IR has no dedicated "is this negative zero?" op. To distinguish `+0.0` from `-0.0`, you need:

```
bits = bitcast_f64_to_i64(x)
is_negative = bits < 0          # sign bit is the MSB of i64
is_zero = (x == 0.0)            # IEEE 754: +0 == -0
is_negative_zero = is_negative & is_zero
```

If the pathmaker naively writes `if x == -0.0: return -0.0` expecting IEEE 754 equality to distinguish signed zeros, that's a bug — the check `x == -0.0` is TRUE even for `+0.0`.

**The correct implementation:** The polynomial form `r + r * r² * S(r²)` with `r = -0.0` gives `-0.0 + (-0.0) * 0 * S(0) = -0.0 + (-0.0) * 0 = -0.0 + (-0.0)`. IEEE 754: `-0.0 + (-0.0) = -0.0`. So the polynomial path does produce `-0.0` for input `-0.0`, assuming the range reduction step passes `r = -0.0` through unchanged.

But the range reduction step: `k = round(-0.0 * one_over_piover2) = round(-0.0) = 0`. Then `r_1 = -0.0 - 0 * piover2_hi = -0.0 - 0.0 = -0.0`. IEEE 754: `-0.0 - 0.0 = -0.0`. And `r = r_1 - 0 * piover2_lo = -0.0 - 0.0 = -0.0`. So `r = -0.0` reaches the polynomial, which produces `-0.0`. Correct — the polynomial path handles this right WITHOUT a front-end check.

**But the front-end short-circuit is still used** (presumably for efficiency). If the front-end uses `fcmp_eq.f64(x, 0.0)` for the zero check and then returns `x` directly, it returns whatever sign `x` had. This is also correct IF the front-end returns `x` (not `0.0` or `-0.0` as a literal).

**Required fix:** The front-end dispatch must be specified as:
- If the zero-check fires (for `x = ±0.0`), return `x` directly (preserving whatever bit pattern `x` has), NOT a literal `+0.0` or `-0.0`.
- For `sin`, the returned value is `x` (sign-preserving).
- For `cos`, the returned value is `1.0` literally (cos is even, so cos(+0) = cos(-0) = 1.0).

The cos case is fine either way. The sin case requires "return x, not return 0.0."

---

## Advisory notes (non-blocking)

### A1 — Sign-symmetry adversarial category (from 2.1 sign-off)

Carried forward from my campsite 2.1 sign-off note: the identity `sin(-x) = -sin(x)` must hold **bit-exact** (not just within 1 ULP). The design doc correctly lists this in the testing section. But I want to emphasize: this must be tested for:

- `x` in every quadrant (k = 0, 1, 2, 3)
- `x` at polynomial boundary values
- `x` subnormal
- `x` = NaN (I11: `sin(-NaN) = -NaN`? No — `sin(NaN) = NaN` and the sign of the returned NaN is not specified by IEEE 754. The identity `sin(-x) = -sin(x)` does NOT apply to NaN because NaN sign is not preserved through arithmetic. Document this exception explicitly.)

For NaN: `sin(NaN) = NaN`. IEEE 754 says the sign and payload of a propagated NaN are implementation-defined. So `sin(-NaN)` may or may not equal `-sin(NaN)`. The bit-exact symmetry test should exclude NaN inputs.

**Advisory:** Add to the testing section: "sign symmetry `sin(-x) = -sin(x)` is bit-exact for all finite `x`. NaN inputs are excluded from the symmetry test (NaN sign propagation is implementation-defined)."

### A2 — Quadrant boundary inputs: `x = k * π/4` in fp64

The testing section says "samples at `k * π/4` for k ∈ [-2^20, 2^20]" stress the dispatch. But `π/4` is not exactly representable in fp64. So "k * π/4 in fp64" means `k * fp64(π/4)`, which accumulates error as `k` grows. For large `k` (say `k = 10^6`), `k * fp64(π/4)` differs from the true `k * π/4` by many ULPs.

The adversarial purpose is to test the *dispatch mechanism* (quadrant selection), not the argument accuracy. So what matters is which quadrant the reduced argument lands in. For this test:
- Use fp64 multiples of a well-chosen constant (like `piover4_hi` or `piover2_hi`) to control which quadrant is targeted.
- Don't assume `k * fp64(π/4)` lands in the k-th octant — it may not for large `k`.

**Advisory:** Clarify in the testing section: "The quadrant boundary test uses `k * piover4_hi` (the constant used in the reduction), not `k * math.pi / 4`. This ensures the quadrant boundary is hit exactly, not approximately."

### A3 — `cos(-0.0)` must be `+1.0`, not `+0.0` or `1.0` with any sign ambiguity

The special-value section says `cos(-0) = 1.0`. This is correct. But the front-end dispatch says `return 1.0` — this is the literal `1.0` constant. In the `.tam` IR, `const.f64 1.0` has bit pattern `0x3FF0000000000000` which is positive `+1.0`. No issue. But verify: after range reduction, for `x = -0.0`, does the dispatch select `cos_poly(r)` or return early? If the zero-check fires and returns the literal `1.0`, fine. If the zero-check is not reached (e.g., `|-0.0| > 2^30` is false, so we proceed to range reduction), then `r = -0.0` goes into `cos_poly(-0.0)`. The polynomial is even, so `cos_poly(-0.0) = cos_poly(0.0) = 1.0`. Fine either way.

No blocker, but verify this in tests: `assert cos(-0.0).to_bits() == 1.0f64.to_bits()`.

---

## Tan ULP tension (carried from 2.1 sign-off)

My fourth note from the 2.1 sign-off was: "tan ULP tension." The sin/cos design doc does not address `tam_tan`. Per `accuracy-target.md`, `tam_tan` is deferred. But the sign-off note was specifically: tan has asymptotes at `x = π/2 + k*π`, and near these points, `tan(x)` changes by ~1/sin²(x) per unit change in x — extremely sensitive. For the ULP budget: a 1 ULP error in `x` near a pole produces a potentially unbounded error in `tan(x)`. This makes 1-ULP tan essentially unreachable near its poles via standard Cody-Waite + polynomial.

The design doc correctly defers tan. No action needed now. When tan is designed in Phase 2, the adversarial review must include: "1 ULP claim is unreachable near poles — specify exactly what the bound is for inputs within n ULPs of a pole."

---

## TMD corpus candidates for sin/cos

Known hard cases from the literature:
- `sin(π/4) = cos(π/4) = 1/sqrt(2)`. The true value is `0.707106781186547524...`. The nearest fp64 is a known TMD candidate.
- `sin(1.0)` — the true value is `0.841470984807896506...`. Known to be within 1 ULP of the midpoint between two fp64 values.
- `cos(1.0)` — similar.
- Small argument region: `sin(2^-27)` — the result is approximately `2^-27` with a cubic correction of `2^-81/6`. The cubic correction is far below fp64 precision. The result should be `2^-27` exactly (to 1 ULP, since the correction is below ULP). This is a good test for "small argument returns x" behavior even though we don't have an explicit small-argument branch.

Log in `peak2-libm/tmd-corpus/sin-cos-tmd.md`.

---

## Required additions to test battery

1. **Resolve the 2^20 vs 2^30 cutoff** and test exactly at the boundary: `sin(2^N)` for `N = 20, 30, 31` where N=31 is expected to be `nan` and N=20 or N=30 (whichever is in-domain) is expected to be a number.
2. **Signed-zero sin preservation:** `assert sin(-0.0).to_bits() == (-0.0f64).to_bits()`.
3. **Signed-zero cos:** `assert cos(-0.0).to_bits() == 1.0f64.to_bits()`.
4. **Quadrant coverage:** for each `k ∈ {0, 1, 2, 3}`, verify at least 1000 inputs hitting that quadrant.
5. **Sign symmetry with NaN exception documented.**
6. **Polynomial boundary margin:** at least 100 inputs from `|r| ∈ [π/4 * 0.99, π/4 * 1.01]`.
7. **I11 NaN propagation:** `sin(NaN) = NaN`, `cos(NaN) = NaN` bit-exact (nan propagated, not 0.0).

---

## Verdict

**HOLD on Campsite 2.13 implementation until:**

1. B1 resolved: pick ONE out-of-domain cutoff (`2^20` or `2^30`) and state it consistently in all sections.
2. B2 resolved: specify that the zero front-end dispatch returns `x` (sign-preserving), not a literal `+0.0`.
