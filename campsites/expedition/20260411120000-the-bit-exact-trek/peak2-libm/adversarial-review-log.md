# Adversarial Pre-Code Review — `log-design.md`

**Reviewer:** Adversarial Mathematician
**Date:** 2026-04-12
**Status:** THREE BLOCKING ISSUES. Implementation must not start (Campsite 2.10) until resolved.

---

## Summary verdict

The log design is mathematically sound and the subnormal/special-value front-end is more carefully specified than exp's. However, three blocking issues and four advisories. The cancellation analysis for the reassembly step (§4.4) is incomplete and introduces a subtle correctness risk.

---

## Blocking issues

### B1 — `log(-0.0) = -inf` is stated but the sign is WRONG per IEEE 754

The special-value section states:

```
log(0) = -inf       (both +0 and -0)
```

And later in the test section:
```
assert tam_ln(-0.0) == -inf
```

This is **correct** — IEEE 754-2019 §9.2 specifies that `log(−0)` returns `−∞` (with the divide-by-zero exception). The value is `-inf`, same as `log(+0)`. But the test assertion `tam_ln(-0.0) == -inf` passes for both `+inf == -inf` (which is `false`) and `-inf == -inf` (which is `true`). The issue is: the test verifies the *value* but not the *sign*.

Wait — `-inf == -inf` is true and `+inf == -inf` is false. So the test *does* distinguish the sign. This is actually fine as written.

However: the front-end dispatch in §4.5 says:

```
if x == 0:       return -inf       # both +0 and -0
```

IEEE 754 equality: `+0.0 == 0.0` is true, `-0.0 == 0.0` is true. So this single check catches both. BUT: in the `.tam` IR, `fcmp_eq.f64(x, 0.0)` uses IEEE 754 equality, which treats `+0 == -0`. This is correct.

The real issue: **the check `x == 0` must come AFTER the `x < 0` check**. The front-end sequence in §4.5 is:

```
if x < 0:                     return nan
if x == 0:                    return -inf
```

This order is correct — negative numbers are caught first. But `x == -0.0` satisfies `x < 0` in some interpretations. In IEEE 754: `-0.0 < 0` is **FALSE** because `-0.0 == 0.0`. So `-0.0` would fall through the `x < 0` check and correctly hit `x == 0`. Fine.

**The actual blocking issue:** the front-end dispatch checks `x < 0` before `x == 0`. Since `-0.0 < 0` is false in IEEE 754, `-0.0` reaches the `x == 0` check and returns `-inf`. Correct. But if an implementer uses a different test — say `x <= 0` for "non-positive" — then `-0.0` also hits `x <= 0` and the dispatch order matters. The design doc must explicitly state: "use strict `<` for the negative check, not `<=`, because `-0.0 <= 0` is true but `-0.0` should return `-inf` not `nan`."

**Required fix:** Add explicit note in §4.5: "The negativity check `x < 0` is strict (IEEE 754 `fcmp_lt.f64`). Do NOT use `x <= 0`. The input `-0.0` must return `-inf`, not `nan`, because `fcmp_lt.f64(-0.0, 0.0)` is `false`."

---

### B2 — Subnormal input path: `x * 2^52` loses sign of zero

The subnormal front-end says: multiply `x` by `2^52`, "which turns a subnormal `x` into a normal number." This is correct for positive subnormals. But `x` could be a negative subnormal (e.g., `-2^-1073`), which is in the `x < 0` → `nan` category.

Wait — a negative subnormal satisfies `x < 0`, so it would be caught by the negative check and return `nan`. Correct.

What about **subnormal +0**? `+0` is not subnormal (subnormal requires non-zero mantissa). So `x = 0` (either `+0` or `-0`) is not subnormal.

The real concern: the subnormal check `if x is subnormal` — what exactly is the check in `.tam` IR? In fp64, a number is subnormal iff its biased exponent field is 0 and its mantissa is nonzero. In the `.tam` IR, detecting subnormals requires:

```
bits = bitcast_f64_to_i64(x)
exp_bits = (bits >> 52) & 0x7ff
is_subnormal = (exp_bits == 0) & (bits & 0x000fffffffffffff != 0)
```

This is three ops. The design doc doesn't say how to detect subnormals in `.tam` IR. If the pathmaker naively uses `x < fp64_min_positive` (i.e., `x < 2^-1022`) as the subnormal check, it would also catch negative numbers — which should already have been handled. If the negative check is processed first, this is fine. But it's a dependency that should be explicit.

**Required fix:** Add the exact subnormal detection recipe in `.tam` IR ops to the design doc. "Check subnormal by: extract biased exponent via bitcast+shift+mask, compare to zero. This must come AFTER the `x < 0` and `x == 0` checks."

---

### B3 — Reassembly cancellation analysis in §4.4 is incomplete

The design doc says:

```
# Compute (e * ln2_hi + f) * 1 + (e * ln2_lo + f² * Q(f)) 
t_hi = e_f64 * ln2_hi + f           # may cancel — but f is small enough that we're OK
t_lo = e_f64 * ln2_lo
polyq = f * f * Q(f)                # nonlinear remainder
result = t_hi + (t_lo + polyq)       # add smallest-magnitude terms first
```

And then says "I will flesh this out as part of the Campsite 2.11 hand-off."

This is not adequate for a design doc that is about to drive implementation. The specific cancellation risk is:

- When `e = -1` and `m ≈ 2` (i.e., `x ≈ 1`): `e * ln2_hi ≈ -0.693` and `f = m - 1 ≈ 1`. So `t_hi = -0.693 + 1 = 0.307`. The subtraction loses about 1 bit (the terms are within a factor of ~3, so Sterbenz doesn't apply). This is manageable.
- When `e = -1` and `m ≈ sqrt(2) ≈ 1.414`: `f ≈ 0.414`, `t_hi = -0.693 + 0.414 = -0.279`. The terms are close in magnitude, losing about 1 bit. Still manageable.
- When `e = 0` and `m ≈ 1` (i.e., `x ≈ 1`): `t_hi = 0 * ln2_hi + f = f`. No cancellation. Fine.
- When `e = 1` and `m ≈ 1` (i.e., `x ≈ 2`): `t_hi = ln2_hi + (m-1) = ln2_hi + 1 = 1.693`. No cancellation. Fine.

The dangerous case is `e = -1, m ≈ sqrt(2)`: `t_hi = -0.693 + 0.414 = -0.279`. Both terms are ~0.5 in magnitude; we lose ~1 bit. After adding `t_lo + polyq`, we need the final result to be within 1 ULP of the true `log(x)`. Whether this holds depends on the exact reassembly precision.

**The adversarial concern:** the design doc does not specify the exact precision budget for the reassembly step, nor does it reference a specific Tang (1990) sequence with known error analysis. "I will flesh this out" is not a design specification.

**Required fix:** The §4.4 reassembly must either:
1. Reference the exact Tang 1990 reassembly sequence (page and equation number), with a note that the pathmaker follows it verbatim, OR
2. Provide the explicit fp64 op sequence with the error analysis (each step's contribution to the final ULP budget).

This is a blocker because the reassembly is where log implementations most commonly fail the 1-ULP target.

---

## Advisory notes

### A1 — `log(x)` for x just above 1: the path through `f = m - 1`

For `x = 1.0 + ε` where `ε` is very small (say `ε = 2^-52`), the extraction gives `m = 1.0 + ε`, `e = 0`, `f = m - 1 = ε`. The polynomial computes `log(1 + ε) ≈ ε - ε²/2 + ...`. Since `ε = 2^-52`, the quadratic term `ε²/2 ≈ 2^-105` is far below fp64 precision. The result is just `ε`, exact to 1 ULP.

But the extraction step `f = m - 1` is a subtraction where `m` is very close to 1. Is this subtraction exact? Yes, by Sterbenz's lemma: `m ∈ [1, 1 + 2^-52]`, so `|m - 1| ≤ |m| / 2`. Sterbenz applies. The subtraction `m - 1` is exact.

No issue here, but pathmaker should verify: **the subtraction `f = m - 1` is exact for all `m ∈ [0.5, 2)` after the sqrt(2) shift gives `m ∈ [sqrt(2)/2, sqrt(2)) ≈ [0.707, 1.414)`.**

Sterbenz: for `m ∈ [0.707, 1.414)`, `|m - 1| ≤ 0.414`. And `|m| / 2 ≤ 0.707`. Since `0.414 < 0.707`, Sterbenz holds. The subtraction is exact. Document this explicitly.

### A2 — `log(exp(x)) == x` identity is more demanding than stated

The testing section says: `exp(log(x)) ≈ x` within 2 ULPs. This is the right direction for the round-trip test. The design should also test `log(exp(x)) ≈ x` — which is the *other* direction, and has a different error profile.

For `log(exp(x)) ≈ x`: `exp(x)` has at most 1 ULP error. Then `log(exp(x))` applies `log` to a slightly-wrong argument. The error in `log(y)` from a 1-ULP error in `y` is: `d/dy log(y) * Δy ≈ (1/y) * (1 ULP of y)`. For `y = exp(x) ≈ e^x`, `1/y = e^{-x}`, and `1 ULP of y ≈ y * 2^-52 = e^x * 2^-52`. So the propagated error is `(e^{-x}) * (e^x * 2^-52) = 2^-52`, which is 1 ULP at 1.0. So `log(exp(x))` should be within 2 ULPs of `x` regardless of `x`. Add this identity test.

### A3 — TMD corpus candidates for log

Known hard cases for `log` from the literature:
- `log(3.0/2.0)` — `ln(1.5)`. The true value is `0.405465108108164381978...`. The nearest fp64 is close to the exact midpoint between two adjacent fp64 values (this is a known TMD case for log).
- `log(e)` — should be `1.0` exactly by mathematical definition, but `e` itself is not exactly representable in fp64. The rounded `e` as fp64 gives `log(2.718281828459045...)`, which differs from 1 by a fraction of a ULP.
- `log(1.0 + 2^-52)` — the smallest fp64 above 1. The true result is approximately `2^-52 - 2^-105/2`, which is extremely close to `2^-52`.

Log these in `peak2-libm/tmd-corpus/log-tmd.md`.

### A4 — `log(x)` for `x` with exponent exactly 0 (`m ≈ 1.0`)

When the exponent field of `x` is zero (i.e., `x ∈ [1, 2)`), `e = 0` and `e_f64 = 0.0`. The reassembly then has `t_lo = 0.0 * ln2_lo = 0.0`. The result is `t_hi + 0.0 = t_hi = f + 0.0 * ln2_hi = f`. This collapses to the polynomial result `f + f² * Q(f)`, which is correct. But the `0.0 * ln2_lo` multiply should give exactly `0.0` by IEEE 754 (any finite number times zero is zero). Verify this holds in the `.tam` IR's `fmul` implementation.

---

## Required additions to test battery

1. **The strict-less-than vs less-than-or-equal for negatives:** verify `tam_ln(-0.0) == -inf` (not `nan`).
2. **Subnormal inputs:** at least 10,000 samples from subnormal range, same as accuracy-target.md requirement.
3. **Sterbenz exactness of `f = m - 1`:** verify for `m ∈ {0.707, 0.9, 1.0, 1.1, 1.414}` that `f` is computed exactly.
4. **Tang reassembly sequence:** once math-researcher fleshes out §4.4, verify it matches Tang 1990 eq. (4.7)–(4.11).
5. **Both round-trip directions:** `exp(log(x)) ≈ x` AND `log(exp(x)) ≈ x`, both within 2 ULPs.
6. **TMD candidates:** `log(1.5)`, `log(e)` (as fp64), `log(1 + 2^-52)`.

---

## Verdict

**HOLD on Campsite 2.10 implementation until:**

1. B1 resolved: add explicit note that the negativity check is strict `<`, not `<=`, with justification for `-0.0`.
2. B2 resolved: add exact subnormal detection recipe in `.tam` IR ops.
3. B3 resolved: §4.4 reassembly must be fully specified (Tang 1990 exact reference OR explicit op sequence with error budget), not deferred to 2.11.
