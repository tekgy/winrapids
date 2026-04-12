# Adversarial Pre-Code Review — `exp-design.md`

**Reviewer:** Adversarial Mathematician
**Date:** 2026-04-12
**Status:** BLOCKING ISSUES flagged below. Implementation must not start (Campsite 2.6) until pathmaker and math-researcher have resolved each one.

Navigator assigned: verify `exp(-inf) → +0.0`, `exp(NaN) → NaN`, `exp(-700) → subnormal`, `exp(1000) → +inf` are all explicit. Full review below.

---

## Summary verdict

The design doc is thorough and structurally sound. The Cody-Waite split, Variant B polynomial, and subnormal path are all specified correctly at the conceptual level. However, I found **five blocking issues** and **six advisory notes** that must be addressed before implementation starts.

---

## Blocking issues

### B1 — `exp(-inf)` sign of zero is NOT stated

The special-value section lists:

```
if x == -inf:   return +0.0
```

Good — that's what IEEE 754 specifies. But the *sign* of zero is not flagged as a test requirement anywhere in the testing section. The test plan at §"Testing plan" lists `exp(-inf) = +0` under "Special values" but does not explicitly specify **bit-exact `+0`, not `-0`**.

This matters because:
- Some libms (including some versions of glibc's exp) historically returned `-0` for `exp(-inf)` due to a sign-propagation bug in their underflow path.
- Our underflow path (§3.5) computes `exp(x) = poly_r * 2^n` where `n` is highly negative. If the sign bit of `poly_r` is ever set by a rounding anomaly (it shouldn't be — `exp(r)` for any `r` is positive — but the intermediate `r = r_1 - n * ln2_lo` could transiently have the sign bit perturbed if `n * ln2_lo` is large), the final result could be `-0` rather than `+0`.

**Required fix:** The testing plan must include the explicit assertion:

```
assert tam_exp(-inf).to_bits() == 0.0f64.to_bits()  // +0.0, not -0.0
```

The *bit pattern* of the result must be checked, not just `result == 0.0` (which passes for both `+0` and `-0`).

**Also missing:** `exp(-746)` (just below `x_underflow ≈ -745.133`) — the return must be `+0.0` bit-exact, not `-0.0`. This is the underflow path's signed-zero requirement, not just the `exp(-inf)` special case.

---

### B2 — `exp(-0.0) = 1.0` is mentioned but the *front-end check is misspecified*

The pseudocode at the end reads:

```python
if x == +0.0 or x == -0.0:  return 1.0
```

IEEE 754 equality in fp64: `+0.0 == -0.0` is TRUE. So the check `x == +0.0` already catches both cases. The explicit `or x == -0.0` is redundant — but worse, it gives the reader the impression that `+0.0` and `-0.0` are different checks that might behave differently. In the `.tam` IR, `fcmp_eq.f64(+0.0, 0.0)` returns `true` because IEEE 754 defines them as equal.

**The real issue:** the design doc does not say whether to use `fcmp_eq.f64` or a bitwise check for the `x == 0` dispatch. For NaN inputs, both give the same answer (NaN is not equal to anything). But the *order* of the checks matters. If we test `x == 0` BEFORE testing `isnan(x)`, and if the IR's `fcmp_eq` silently returns false for NaN (which it does), then `isnan` must come FIRST or the NaN case falls through to the range reduction. The pseudocode has `isnan` first, which is correct. But this is not called out as a constraint.

**Required fix:** Add an explicit note in the special-value section: "The `isnan` check MUST precede all other comparisons. Any reordering causes NaN to fall through to range reduction where `round(NaN / ln(2))` is undefined behavior."

---

### B3 — `exp(+709.7827128933840...)` one-ULP boundary is unspecified

The overflow threshold `x_overflow` is documented as "approximately `709.782712893384...`" with the note that the exact fp64 literal is TBD from `§4`. This is correct process, but the boundary behavior is not specified for inputs *at* `x_overflow`.

IEEE 754 defines `exp(x)` for `x` equal to the largest finite argument as a finite result (approximately `1.7976931348623157e+308`). One ULP above that argument, the result rounds to `+inf`. The exact split depends on the fp64 bit pattern of `x_overflow` — which the doc doesn't commit.

**The adversarial concern:** if `x_overflow` is set to the largest `x` for which the mathematical `exp(x) < fp64_max`, but the polynomial + reassembly path returns `+inf` due to rounding, we have a case where `x < x_overflow` but `tam_exp(x) = +inf`. This violates the spec ("if `x < x_overflow`, proceed to range reduction"). The overflow check must be:

```
x_overflow = largest fp64 x such that tam_exp(x) is finite
```

Not:
```
x_overflow = largest fp64 x such that mathematical exp(x) < fp64_max
```

These are different! The implementation's rounding may push the polynomial result over `fp64_max` before the mathematical function does.

**Required fix:** `exp-constants.toml` must include the verified result of `tam_exp(x_overflow)` (must be finite, not `+inf`) AND `tam_exp(nextafter(x_overflow, +inf))` (must be `+inf`). This needs to be run and checked after the polynomial is generated, not assumed from the mpmath boundary computation.

---

### B4 — Cody-Waite exact-input pitfall not addressed

From my campsite 2.1 sign-off notes (flagged previously): the Cody-Waite reduction with `n * ln2_hi` being "exact by Sterbenz" has a silent precondition — Sterbenz's lemma requires `|r_1| ≤ |x| / 2`.

For `x` exactly equal to `n * ln2_hi` for some integer `n` (i.e., `x` is a Cody-Waite-exact input), then `r_1 = x - n * ln2_hi = 0.0` exactly, and `r = 0.0 - n * ln2_lo = -n * ln2_lo`. The exact `exp(-n * ln2_lo) = exp(r)` for the full answer is `exp(n * ln2_hi + r) = exp(x)`, which requires the polynomial to return `exp(-n * ln2_lo)` accurately. For these inputs, the polynomial argument is NOT small — it's `n * ln2_lo` which for large `n` can be up to `1023 * ln2_lo ≈ 1023 * 5.5e-14 ≈ 5.6e-11`. This is within `[-ln(2)/2, ln(2)/2]` because `|ln2_lo|` is small enough, but the polynomial was Remez-fit on the interval `[-ln(2)/2, ln(2)/2]` assuming uniform `r`. The residual polynomial error at inputs `r = n * ln2_lo` for large `n` should be checked specifically.

**The deeper issue:** for `n = 1023`, `r = -1023 * ln2_lo ≈ -5.6e-11`. Is this within the fit interval? `5.6e-11 << ln(2)/2 ≈ 0.35`. Yes. But I don't see any verification that the polynomial error at this *structured* input is still within 1 ULP — the Remez fit guarantees max error over the full interval, not at structured points.

**Required fix:** Include Cody-Waite exact inputs in the adversarial test suite. For `n ∈ {-1074, -1023, ..., 0, ..., 1023}`, compute `x = n * ln2_hi` in fp64, run `tam_exp(x)`, verify against mpmath. This is ~2100 inputs, trivial to enumerate.

---

### B5 — Subnormal boundary double-round hazard not quantified

Section §3.5 describes the subnormal path for `n ∈ [-1074, -1023]` using the two-step split `n = -1022 + (n + 1022)`. The doc says "This forces the subnormal rounding to happen exactly once, in the final multiply." However, the *specific inputs* where this claim is tested are not enumerated.

The adversarial concern: for `n = -1023` (the first subnormal), `poly_r ∈ [0.707, 1.414]`, and `poly_r * 2^-1022` is a normal number (since `poly_r` is normal and `2^-1022` is normal, their product is at the subnormal boundary). The result could be normal or subnormal depending on `poly_r`'s value. If `poly_r > 1`, `poly_r * 2^-1022 > 2^-1022 = fp64_min_normal`, which is normal. If `poly_r < 1`, `poly_r * 2^-1022 < fp64_min_normal`, which is subnormal. The two cases follow different arithmetic paths.

**The two-step split claim**: `intermediate = poly_r * 2^-1022` (normal), then `result = intermediate * 2^(n + 1022)` where `n + 1022 ∈ [-52, -1]`, so `2^(n+1022) ∈ [2^-52, 2^-1]`. The second multiply has one operand in the subnormal-normal transition zone (since `intermediate` is very small) and one operand that's normal. The rounding happens in this second multiply. This is correct.

BUT: the doc does not address what happens when `n = -1074` (the deepest subnormal case). Here `n + 1022 = -52`, so `2^(n+1022) = 2^-52`. The `intermediate = poly_r * 2^-1022` is normal (fine). Then `result = intermediate * 2^-52`. For `poly_r ∈ [0.707, 1.414]`, `intermediate ∈ [0.707 * 2^-1022, 1.414 * 2^-1022]`, and `result = intermediate * 2^-52 ∈ [0.707 * 2^-1074, 1.414 * 2^-1074]`. The minimum subnormal is `2^-1074 ≈ 5e-324`. So `result ∈ [5e-324 * 0.707, 5e-324 * 1.414]`. This is either `2^-1074` (the smallest subnormal) or zero (if the rounding rounds down). This is the true boundary where double-round risks exist.

**Required fix:** Add `exp(-744.440)` and `exp(-745.133)` (boundary values for the deepest subnormal path) to the adversarial test suite with mpmath reference values. Add `exp(x)` for `x` such that the result is exactly `2^-1074` (the smallest subnormal) — this requires finding `x` numerically from mpmath.

---

## Advisory notes (non-blocking but should be addressed)

### A1 — `exp(NaN)` bit pattern not specified

The special-value section says `return x` for NaN, which is correct (quiet NaN propagation). But it does not say whether *signaling NaN* is preserved or quieted. IEEE 754-2019 §6.2 says that a `signalNaN` input to any operation should raise the invalid operation flag and return a quietNaN. Phase 1 doesn't track fp flags, but the question is: does `tam_exp(sNaN)` return a qNaN or the original sNaN bit pattern?

If we `return x` directly and `x` is a sNaN, we've returned a sNaN without quieting it. Most libms return qNaN (they call `x + 0.0` or `x - x` to quiet it). We should document which behavior we choose, even if we defer the sNaN-vs-qNaN question.

**Advisory:** Document the sNaN policy. "Phase 1 preserves NaN bit patterns including signaling NaN" OR "Phase 1 quiets sNaN via `x + 0.0` in the front-end."

### A2 — `exp(x)` for `x` in `[-708.4, -708.3]` — the argument that produces `fp64_min_normal`

The accuracy claim for the "subnormal path" (§3.5) says "Campsite 2.8" tests this. But the design doc doesn't specify what "correct" means at `exp(-708.396...)`. Near this boundary:
- For `x` slightly above `-708.396...`, the result is a normal fp64 (biased exponent = 1).
- For `x` slightly below, the result is the smallest subnormal.
- AT the boundary, `exp(-708.396...)` should equal exactly `fp64_min_normal = 2^-1022 ≈ 2.225e-308`.

The normal-to-subnormal transition is where most libm bugs live. The design doc does not describe how the algorithm behaves at this transition — it handles `n = -1022` (the first subnormal-producing n) without saying what happens when `poly_r * 2^-1022` is just barely above or below `2^-1022`.

**Advisory:** Add explicit test inputs at the normal-subnormal boundary. Specifically: find `x` such that the true mathematical answer is exactly `2^-1022`, verify `tam_exp(x) = 2^-1022` or `tam_exp(x) = 2^-1023 * (1 + 2^-51)` (the two neighboring fp64 values).

### A3 — Horner direction: "highest degree first" is correct but not enforced

§3.3 says "Horner evaluation, inside-out". The pseudocode shows:

```python
p = a_10
p = p * r + a_9
...
p = p * r + a_2
```

This starts with `a_10` (highest degree) and works DOWN. This is correct Horner (Horner's method evaluated from the highest coefficient downward). "Inside-out" is an informal description that could confuse. More precisely: each step is `p := p * r + a_k` where `k` decrements.

**Advisory:** Confirm in the doc that the polynomial degree is 10 (not 8). The body says "degree 10" (with 9 coefficients `a_2 ... a_10`) and separately says "degree ~7" and "degree 8". These numbers are inconsistent. The Remez target should be fixed to one number before implementation starts.

Specifically: "Variant B at degree ~7 for `|r| ≤ ln(2)/2`" is mentioned, then "Polynomial degree target: 10 (one degree of headroom)". Is it 7 (what the math requires), 8 (what §3.3 describes for the reduced polynomial), or 10 (what the pseudocode uses)? The pathmaker must start from an unambiguous degree.

**Blocking-adjacent:** the pathmaker needs a definitive degree before writing any `.tam` code. This is close to a blocker; I'm calling it advisory because the choice of 8 vs 10 is the math-researcher's call.

### A4 — `n_f64` vs `n` in pseudocode

The pseudocode uses:
```python
n_f64 = round_nearest(x * one_over_ln2)
n     = f64_to_i32(n_f64)
r_1   = x - n_f64 * ln2_hi
r     = r_1 - n_f64 * ln2_lo
```

The Cody-Waite subtractions use `n_f64` (the f64 form of the rounded integer), not `n` (the i32). This is correct — you want the exact-representable f64 integer for the subtraction, not a cast back from i32 which would involve a `cvt.s32.f64` and another rounding. But the `ldexp` call uses `n` (the i32):

```python
return ldexp(poly_r, n)
```

This is also correct — `ldexp` takes an integer exponent. But the design doc doesn't say explicitly: "multiply by `n_f64` in the Cody-Waite step, then cast to `n : i32` for the `ldexp` step." A pathmaker who misreads this might use `n` in the subtractions or `n_f64` in the `ldexp`.

**Advisory:** Add an explicit note in the pseudocode: "The subtraction `x - n_f64 * ln2_hi` uses the f64 form of the rounded integer. The `ldexp` call uses the i32 form. Both are computed from the same rounding operation."

### A5 — Polynomial interval margin note

§"Pitfalls" note 3 says the polynomial boundary check should fit a "slightly wider interval" `[-π/4 * 1.01, π/4 * 1.01]`. Wait — that's from sin-cos-design. In exp-design, the Remez polynomial is fit on `[-ln(2)/2, ln(2)/2]`. The design does NOT say to fit with a margin.

The adversarial concern: after `n = round(x * one_over_ln2)`, the reduced argument `r ≈ x - n * ln2`. Due to rounding in `one_over_ln2`, the actual value of `r` after Cody-Waite can slightly exceed `ln(2)/2` for some inputs. Specifically: if `x / ln(2)` is halfway between two integers and the rounding of `one_over_ln2` pushes the computed quotient to the wrong integer, we get `|r| ≈ ln(2)/2` or slightly above.

**Advisory:** Verify (by exhaustive enumeration or by analysis) that the Cody-Waite scheme's `|r|` never exceeds `ln(2)/2 + ε` for any `x ∈ [x_underflow, x_overflow]`. If it can exceed it, add `5%` margin to the Remez fit interval. Otherwise document why no margin is needed.

### A6 — TMD corpus candidates for exp

Table Maker's Dilemma inputs for `exp`: values where `exp(x)` is within `2^-53 / 2` of the midpoint between two adjacent fp64 values (i.e., the true value is just barely above/below the halfway point, so 53 digits of mpmath may not determine the correct rounding direction).

Known hard cases for `exp` from the literature:
- `exp(1.0)` — `e` itself. The true value of `e` is approximately `2.71828182845904523536...`. The nearest fp64 is `2.718281828459045...` which has a rounding error of approximately 0.5 ULP. Whether the implementation rounds up or down is the TMD issue.
- `exp(ln(2)/2)` — exactly at the midpoint of the Cody-Waite interval. The polynomial takes `r = 0` after one Cody-Waite step; `exp(0) = 1` exactly, so `r = ln(2)/2` after no Cody-Waite steps (if `n = 0`). This is a known TMD case in some implementations.
- `exp(-1.0)` — `1/e`. Similar to `exp(1.0)`.

**Advisory (I9′ corpus curation):** Log these in `peak2-libm/tmd-corpus/exp-tmd.md` per Navigator's I9′ assignment. These inputs need MPFR or Arb at 500+ digits to determine the correct rounding direction. They are the future Phase 2 correctness gate.

---

## Required additions to the test battery

The testing plan in §"Testing plan" is good but is missing:

1. **Signed-zero exact bit test:** `assert exp(-inf).to_bits() == 0u64` (not just `== 0.0`).
2. **Underflow signed-zero test:** `assert exp(-746.0).to_bits() == 0u64`.
3. **Cody-Waite exact inputs:** `exp(n * ln2_hi)` for `n ∈ {-1023, -512, 0, 512, 1023}`.
4. **Subnormal boundary pair:** `exp(x)` and `exp(nextafter(x, 0))` where `x` is the smallest argument returning a normal result.
5. **Polynomial degree/interval margin confirmation:** at least 100 samples near `|r| = ln(2)/2` from each side.
6. **TMD candidates:** `exp(1.0)`, `exp(-1.0)`, `exp(0.5)`, `exp(0.693...)`, verified at 500+ digits.

---

## On the ldexp IR op (already confirmed present)

The design doc at the end asks pathmaker to flag whether `ldexp.f64` is in the IR op set. Per interp.rs and ast.rs (read during Peak 1 adversarial sweep): **yes, it is present**. `Op::LdExpF64 { dst, mantissa, exp }` is in the IR and implemented in the interpreter. `F64ToI32Rn` is also present. No IR amendment required for these two ops.

The `BitcastF64ToI64` and `BitcastI64ToF64` ops are also present (needed by log's exponent extraction). The IR op set is complete for all four ops listed in the design doc's "IR additions required" section.

---

## Items carried forward from campsite 2.1 sign-off

My four notes from the 2.1 sign-off:

1. **Cody-Waite exact inputs** — now formalized as B4 above. Test enumerated.
2. **Sign-symmetry category** — exp has no sign symmetry (it's strictly positive), so this note was for trig. Carried forward to sin-cos review.
3. **asin/acos near-±1 clustering** — carried forward to atan review.
4. **tan ULP tension** — carried forward to sin-cos review.

---

## Verdict

**HOLD on Campsite 2.6 implementation until:**

1. B1 resolved: signed-zero test for `exp(-inf)` and `exp(-746)` added to test battery.
2. B2 resolved: `isnan` first ordering called out as a constraint, not just a code pattern.
3. B3 resolved: `x_overflow` defined as "largest x where tam_exp(x) is finite" (not the mathematical boundary), with verification test.
4. B4 resolved: Cody-Waite exact inputs added to the adversarial battery.
5. B5 resolved: `exp(-744.4)` and `exp(-745.1)` boundary cases added to the adversarial battery, with mpmath references.

**Advisory A3 (polynomial degree):** the math-researcher must pick ONE degree before implementation starts. Recommend committing to 10 (the "one degree of headroom" choice) since the pseudocode already uses it.
