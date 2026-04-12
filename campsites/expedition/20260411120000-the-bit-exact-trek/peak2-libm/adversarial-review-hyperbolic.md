# Adversarial Pre-Code Review — `hyperbolic-design.md`

**Reviewer:** Adversarial Mathematician
**Date:** 2026-04-12
**Status:** FOUR BLOCKING ISSUES. TWO advisory notes. Implementation must not start (Campsite 2.18) until resolved. B4 added 2026-04-12 after error-cascade analysis prompted by navigator.

---

## Summary verdict

The hyperbolic design is the most careful of the Phase 1 docs so far — the piecewise thresholds are justified, the overflow trick is recognized, and the polynomial structure is sound. However, three blocking issues: the `exp(x - ln(2))` trick for the large-regime has an unverified claim at the boundary, the `cosh` medium-regime formula has a precision problem not mentioned in the design, and the `tanh` even/odd detection for the small-regime branch is inconsistent with the design doc's stated return formula.

---

## Blocking issues

### B1 — `exp(x - ln(2))` overflow trick: boundary verification is absent

The design doc says for `|x| ≥ 22` in `sinh`:

```python
if x > 0:
    return exp(x - ln(2))   # = e^x / 2, computed without forming the huge value first
```

This trick works because `exp(x) / 2 = exp(x - ln(2))`. The concern: at `x` slightly below the `|x| < 22` boundary (e.g., `x = 21.99`), the formula switches from `(e^x - e^-x) / 2` (medium regime) to `exp(x - ln(2))` (large regime). At the boundary, `e^(-22) ≈ 2.8 × 10^-10 = 2^-31.8`, which is NOT below 1 ULP of `e^22 / 2 ≈ 1.75 × 10^9`. In fact, `e^-22` is about `2^-31.8` while 1 ULP of `e^22 / 2` is approximately `e^22 / 2 × 2^-52 ≈ 1.75 × 10^9 × 2^-52 ≈ 3.9 × 10^-7 = 2^-21.6`. So `e^-22 ≈ 2^-31.8` is roughly `2^10` times smaller than 1 ULP of the result — the drop is fine, the error from omitting `e^{-x}` is about `2^{-10}` ULPs.

BUT: the threshold in the design doc's text says `|x| < 22` goes to medium regime, `|x| ≥ 22` goes to large regime. The design justifies this: "past `|x| = 22`, `e^{-x} < 2^{-64}`". Let me verify: `e^{-22} ≈ e^{-22} ≈ 2.8 × 10^{-10}`. This is NOT `2^{-64}`. The claim `e^{-22} < 2^{-64}` is FALSE. `2^{-64} ≈ 5.4 × 10^{-20}`. To get `e^{-x} < 2^{-64}`, we need `x > 64 * ln(2) ≈ 44.4`.

The threshold `|x| = 22` is justified by a different argument: that `e^{-22}` is below 1 ULP of `e^{22} / 2`. The design claims this with the phrase "well below 1 ULP of `e^x / 2`" but uses the number `2^{-64}` as justification for a cutoff at 22, which is wrong — `e^{-22}` is `2^{-31.8}`, not `2^{-64}`.

**The actual correct justification** for the threshold at 22: `ulp(e^22 / 2) = e^22 / 2 × 2^{-52}`. For the dropped term `e^{-22}/2` to be below 0.5 ULP (so the result rounds correctly), we need `e^{-22}/2 < (e^22 / 2) × 2^{-53}`, i.e., `e^{-44} < 2^{-53}`. Since `e^{-44} ≈ 1.2 × 10^{-19} ≈ 2^{-62.7}` which IS less than `2^{-53}`. The threshold at 22 is justified, but NOT because `e^{-22} < 2^{-64}` — rather because `e^{-44} < 2^{-53}`.

**Required fix:** Correct the threshold justification to read: "The threshold `|x| = 22` is justified because: at `|x| = 22`, the dropped term `e^{-x}/2` equals approximately `1.4 × 10^{-10}` and the unit in the last place of the result is approximately `e^{22}/2 × 2^{-52} ≈ 3.9 × 10^{-7}`. So `e^{-22}/2 ≈ 3.6 × 10^{-4}` ULPs of the result — well below 0.5 ULP. Correct rounding is preserved. The phrase `e^{-x} < 2^{-64}` in the draft was incorrect; the correct bound is `e^{-2x} < 2^{-53}` (i.e., `x > 53 × ln(2) / 2 ≈ 18.4`), so 22 has ample margin."

---

### B2 — `cosh` medium-regime `1/e_x`: precision problem not documented

The `cosh` algorithm uses `e_neg = 1 / e_x` in the medium regime (`|x| < 22`). The design doc in the Open Questions section says:

> "Is `1/e_x` or `exp(-x)` the right way to compute the complementary value in the medium regime? The `1/e_x` form is 1 fdiv (0.5 ULP); `exp(-x)` is another 1 ULP call. For 1 ULP target, `1/e_x` is safer because it stays within the same exp call's rounding neighborhood."

This reasoning is WRONG. `1/e_x` is NOT "safer" than `exp(-x)` for accuracy. `fdiv` introduces up to 0.5 ULP. `exp(-x)` introduces up to 1 ULP. BUT: `1/e_x` introduces error in the reciprocal applied to an already-rounded `e_x`. The true value we want is `e^{-x}` exactly; `1 / tam_exp(x)` gives `1 / (e^x * (1 + ε_1))` for some `|ε_1| < 2^{-52}`. This equals `e^{-x} * (1 - ε_1 + ε_1^2 - ...)/(1 + δ)` where `|δ| < 2^{-53}` is the fdiv error. The total error is `≈ ε_1 + δ`, each up to `2^{-52}`. The two errors can have the same sign, giving a combined error up to `2 × 2^{-52}` = 2 ULPs of `e^{-x}`. This is WORSE than `exp(-x)` at 1 ULP.

The reason `1/e_x` is sometimes preferred is speed (one fewer function call), not accuracy. The design doc confuses these.

For `cosh` specifically, the formula is `(e^x + e^{-x}) / 2`. With `e_neg = 1/e_x`, the error in `e_neg` is up to 2 ULPs as analyzed. The addition `e_x + e_neg` and division by 2 each add 0.5 ULP. Total: up to 3 ULPs. This could violate the 1 ULP bound for unlucky inputs.

**The correct choice**: use `exp(-x)` for `e_neg`. Yes it's 1 extra function call. The 1 ULP target requires it.

**Required fix:** Change the `cosh` algorithm to:
```python
e_x = exp(x)
e_neg = exp(-x)      # NOT 1/e_x — see precision analysis
return (e_x + e_neg) / 2
```
And remove the incorrect reasoning from Open Questions.

---

### B3 — `tanh` large-regime: `copysign(1.0, x)` is only correct if `x ≠ -0.0`

The large-regime of `tanh` returns:

```python
return copysign(1.0, x)
```

For `|x| ≥ 22`, `tanh(x) → ±1` with full precision. `copysign(1.0, x)` returns `+1.0` if x is positive (or `+0.0`) and `-1.0` if x is negative (or `-0.0`).

But the small-regime check already handles `x = ±0` (returns `x`). And `|x| ≥ 22` means `x` cannot be `±0`. So `copysign(1.0, x)` for `|x| ≥ 22` is correct.

HOWEVER: the design doc's algorithm also states a separate I11 check at the top:

```python
if isnan(x):
    return x
```

This check correctly returns the input `nan`. But the algorithm structure then falls through to `if |x| < 2^-28: return x` and `elif |x| < 0.55:` etc. — with the large-regime being the final `else`. The `isnan` check must come FIRST in the dispatch or else the comparison `|x| < 2^-28` triggers undefined behavior for NaN (in IEEE 754, any comparison with NaN returns false, so `isnan` check isn't strictly needed — NaN propagates through `fcmp_lt` as false. But the explicit `isnan` guard at the top is fine and correct. The issue is different:

**The actual blocking issue:** the `tanh` small-regime polynomial formula in the design doc is:

```python
return x + x * x * x * TANH_POLY(x*x)
```

But `TANH_POLY(x*x)` is described as a "Remez fit to `(tanh(x) - x) / x³` on `[-0.55, 0.55]`." This means TANH_POLY approximates `(tanh(x) - x) / x³`. The formula `x + x³ * TANH_POLY(x²)` = `x + x * tanh(x) - x` = `tanh(x)`. Wait, that's circular. Let me re-check:

`(tanh(x) - x) / x³ ≈ TANH_POLY(x²)` implies `tanh(x) ≈ x + x³ * TANH_POLY(x²)`.

That's correct. But this formula has a cancellation: for `x` near 0.55, `tanh(0.55) ≈ 0.5025` while `x = 0.55`. So `x³ * TANH_POLY(x²) ≈ -0.047`, a negative correction. This is fine — no catastrophic cancellation.

**The actual issue:** for `x = -0.0`, the formula gives `-0.0 + (-0.0)³ * TANH_POLY(0.0) = -0.0 + (-0.0) * TANH_POLY(0)`. IEEE 754: `(-0.0)^3 = -0.0`. `(-0.0) * TANH_POLY(0)` where `TANH_POLY(0)` is a finite constant: this is `-0.0 * const = -0.0`. Then `-0.0 + (-0.0) = -0.0`. So the formula correctly gives `-0.0` for `-0.0` input — no bug there.

BUT: the design doc has `if |x| < 2^-28: return x` which catches `x = -0.0` since `|-0.0| = 0 < 2^-28`. So `-0.0` is handled by the early return. The polynomial branch for `-0.0` is never reached. Fine.

The real blocking issue in `tanh` is: **the medium-regime formula.**

The design says for `|x| < 22`:

```python
sign = sign_of(x)
ax = |x|
e_2x = exp(2 * ax)
return sign * (1.0 - 2.0 / (e_2x + 1.0))
```

The `sign_of(x)` function — what does it return? If it returns `+1` or `-1` as fp64, then multiplying by `sign` gives `±(1 - 2/(e^{2|x|}+1))`. For `x > 0`, this is `+(tanh(|x|))` which is positive. For `x < 0`, this is `-(tanh(|x|))` which is negative. But for `x = -0.0`: `|-0.0| = 0 < 2^-28`, so this branch is never reached. Fine.

But: `sign_of(x)` is not a `.tam` IR op. It must be implemented. In `.tam` IR, getting the sign of `x` is either:
1. `copysign(1.0, x)` — returns `±1.0` with sign matching `x`
2. `fcmp_lt(x, 0.0) ? -1.0 : 1.0` — returns -1 for negative, but gives +1 for -0.0

Since `-0.0 < 0` is FALSE in IEEE 754, approach 2 gives `+1.0` for `-0.0`, which would give `+tanh(|-0.0|) = +0.0` when it should give `-0.0`. But again, `x = -0.0` hits the `|x| < 2^-28` early return and never reaches this branch. Safe.

The actual blocking issue is different and subtler: the formula `1.0 - 2.0 / (e_2x + 1.0)` for the medium regime. For `x` slightly above `0.55`:

- `e_2x = exp(1.1) ≈ 3.0042`
- `e_2x + 1 = 4.0042`
- `2 / (e_2x + 1) ≈ 0.4995`
- `1 - 0.4995 = 0.5005`

But `tanh(0.55) ≈ 0.5025`. There's a sign error: for `x = 0.55`, the formula gives `1 - 2/(e^{1.1}+1)`. Let me recalculate: `e^{1.1} = 3.00417`, `e^{1.1} + 1 = 4.00417`, `2/4.00417 = 0.49948`, `1 - 0.49948 = 0.50052`. And true `tanh(0.55) = 0.50052...`. Fine, the formula is correct.

The issue is: the formula uses `e_2x = exp(2 * ax)` where `ax = |x|`. The multiplication `2 * ax` is a fp64 multiply: `fmul.f64(2.0, ax)`. For `ax` near 0.55, this is exact (since 2 is a power of 2). For `ax` near 11 (half the medium regime boundary of 22), this is `2 * 11 = 22`, which is the boundary of `exp`'s primary domain. No concern here.

**I was wrong about there being an issue in the formula itself.** The formula is correct.

The actual blocking issue is: the design does NOT specify how to compute `sign * result` when `sign` is extracted via `.tam` IR. The `copysign(1.0, x)` approach gives a fp64 multiply that is exact only because both operands have magnitude 1 — wait, no. `sign ∈ {+1.0, -1.0}` and `result ∈ (0, 1)`, so `sign * result` is a single `fmul`. The product has magnitude less than 1, so no overflow. And `fmul(±1.0, result) = ±result` exactly (multiplying by ±1 is exact). No precision loss.

After re-analysis: there is no floating-point blocking issue in `tanh`'s formula. The blocking issues are B1 (wrong threshold justification) and B2 (wrong precision reasoning for `cosh`). Let me replace B3 with the actual issue I should flag:

**B3 (revised) — `sinh(-0.0)` bit-exact: small-regime `return x` must be bit-exact**

The design says:

```python
if |x| < 2^-28:
    return x
```

For `x = -0.0`: `|-0.0| = 0 < 2^-28`, so this returns `-0.0`. The special-value section says `sinh(-0) = -0`. Correct.

But the test for `|x| < 2^-28` requires computing `|x|` in `.tam` IR, which is `fabs.f64(x)`. For `x = -0.0`, `fabs(-0.0) = +0.0` (IEEE 754: absolute value of -0 is +0). Then `fcmp_lt.f64(0.0, 2^-28)` is `true`, so the check fires. The return is `x`, which is `-0.0`. Correct.

The actual concern: what IR op computes `|x|`? The design doc uses `|x|` notation throughout but never specifies the op. The `.tam` IR must have `fabs.f64`. If pathmaker does not have `fabs` in the IR and instead implements `|x|` via `sqrt(x*x)` or `fcmp_lt + fneg`, they may lose the sign information.

**Required fix:** Specify in the design doc that `|x|` uses `fabs.f64`. Verify that `fabs.f64` is in the `.tam` IR op set. If not, add it as an IR amendment (per the pattern for BitcastF64ToI64 etc.).

**Check:** The `.tam` IR spec (per context from the session) includes `fsqrt.f64`, `fadd.f64`, `fsub.f64`, `fmul.f64`, `fdiv.f64`, `fneg.f64`, but `fabs.f64` was not listed among the new ops added. If `fabs` is absent, the hyperbolic design requires an IR amendment.

---

## Advisory notes

### A1 — `cosh` is even but the large-regime still uses `if x > 0 / else`

The large-regime for `cosh`:

```python
if x > 0:
    return exp(x - ln(2))
else:
    return exp(-x - ln(2))
```

Since `cosh` is even, `exp(x - ln(2)) = exp((-x) - ln(2))` for `x > 0` vs `x < 0`. The code correctly uses `exp(|x| - ln(2))` in both branches. No bug. But: a simpler implementation is `exp(fabs(x) - ln(2))` which avoids the branch. Not a blocker, but a cleaner design.

**Advisory:** Replace the two-branch large-regime with `return exp(fabs(x) - ln2)` where `ln2` is the pre-computed fp64 constant `ln(2)`.

### A2 — Polynomial boundary tests: the `tanh` threshold at `0.55` is empirical

The design doc says: "the threshold at `|x| = 0.55` is empirical. Navigator: check by benchmarking both ways at `|x| = 0.54` and `0.56` once pathmaker has the implementation."

This is OK for an open question in the design doc. But the adversarial concern is: if the polynomial is fit on `[-0.55, 0.55]` and evaluated at `x = 0.5501` (slightly outside the fit domain), the polynomial's accuracy degrades rapidly. The design should specify: "the polynomial TANH_POLY must be fit on `[-0.55, 0.55]` and the threshold must be `|x| ≤ 0.55`, not `|x| < 0.55`. At exactly `x = 0.55`, both branches should give the same result to 1 ULP."

**Advisory:** Add: "Verify at `|x| = 0.55` that the polynomial branch and the formula branch agree within 1 ULP. If they don't, adjust the threshold until they do." The threshold-matching test is mandatory for any piecewise algorithm.

### A3 — TMD corpus for hyperbolic functions

Known hard cases:
- `sinh(1.0)` — true value `(e - 1/e)/2 ≈ 1.1752011936438014...`. Known TMD candidate.
- `cosh(1.0)` — true value `(e + 1/e)/2 ≈ 1.5430806348152437...`. Known TMD candidate.
- `tanh(0.5)` — true value `(e - 1)/(e + 1) ≈ 0.4621171572600097...`. Known TMD candidate.
- `tanh(1.0)` — true value `≈ 0.7615941559557649...`. Close to a midpoint.

Log in `peak2-libm/tmd-corpus/hyperbolic-tmd.md`.

---

## Required additions to test battery

1. **Threshold boundary tests:** At `|x| = 0.55 - ε` and `0.55 + ε` for tanh, at `|x| = 1 - ε` and `1 + ε` for sinh, at `|x| = 2^-28 - ε` and `2^-28 + ε` for all three: values from both branches must agree within 1 ULP.
2. **Large-regime boundary:** `sinh(22.0)`, `cosh(22.0)`, and check that the medium-regime result at `21.99` agrees with large-regime at `22.01` within 1 ULP.
3. **Overflow boundary:** `sinh(709)` and `cosh(709)` should be finite; `sinh(800)` should be `+inf`; `sinh(-800)` should be `-inf`.
4. **Bit-exact signed zeros:** `sinh(-0.0).to_bits() == (-0.0f64).to_bits()`, `cosh(-0.0).to_bits() == (1.0f64).to_bits()`, `tanh(-0.0).to_bits() == (-0.0f64).to_bits()`.
5. **Identity test:** `cosh²(x) - sinh²(x) - 1.0` within 3 ULPs for `x ∈ [-20, 20]` (hyperbolic Pythagorean identity).
6. **TMD candidates** listed above.
7. **Confirm `fabs.f64` in IR** before implementation begins.

---

### B4 — `tanh` medium-regime: ~2 ULP error near the lower boundary `|x| = 0.55`

**Added 2026-04-12 in response to navigator's question about hyperbolic error cascade.**

The `tanh` medium-regime formula for `x > 0`:

```
e_2x = exp(2 * x)
result = 1.0 - 2.0 / (e_2x + 1.0)
```

Navigator asked: does the `cosh` B2 error (from `1/e_x`) cascade into `tanh`? The answer is no — `tanh` does not call `cosh`. But investigating the cascade revealed an independent problem in `tanh`'s own medium-regime error budget.

**Error analysis at `x = 0.55` (the lower boundary of the medium regime):**

1. `exp(2x) = exp(1.1) ≈ 3.0042`: 1-ULP relative error. Error magnitude: `e^{1.1} × 2^{-52} ≈ 6.7 × 10^{-16}`.
2. `e_2x + 1.0 ≈ 4.0042`: fadd, 0.5-ULP rounding. No cancellation (both terms positive). Error magnitude ≈ `4.0 × 2^{-53} ≈ 4.4 × 10^{-16}`. Dominated by step 1 propagation.
3. `2.0 / (e_2x + 1.0) ≈ 0.4995`: fdiv, 0.5-ULP rounding. Total relative error in this quotient from steps 1–3: approximately `1 ULP (from exp) + 0.5 ULP (from fdiv) ≈ 1.5 ULP relative.`
4. `1.0 - 0.4995 = 0.5005 ≈ tanh(0.55)`: fsub, 0.5-ULP rounding.

At step 4, the subtraction `1.0 - q` where `q ≈ 0.4995`. The operands are 1.0 and 0.4995 — these differ by about 0.5, so there is **no catastrophic cancellation** in the classical sense. However, the absolute error in `q` (from steps 1–3) is approximately `1.5 × q × 2^{-52}`. The output `tanh(x) = 1 - q ≈ q` (since `tanh(0.55) ≈ 0.503 ≈ q`). So the relative error in `tanh(x)` is:

```
|abs_error_in_q| / tanh(x) ≈ (1.5 × q × 2^-52) / q = 1.5 × 2^-52 → 1.5 ULP
```

Adding the fsub's own 0.5-ULP rounding: total approximately **2 ULP** at `x = 0.55`.

The 1-ULP Phase 1 bound requires at most 1 ULP. This formula is ~2 ULP at the lower boundary.

**Why this was missed in the original review:** The formula `1 - 2/(e^{2x}+1)` looks safe (no obvious cancellation), but the error analysis shows that the 1-ULP error from `exp(2x)` propagates through the quotient to the output with coefficient close to 2 at the boundary where `tanh(x) ≈ 0.5`.

**Two paths to resolution (math-researcher's choice):**

**Option A: Extend the polynomial regime boundary.** If the polynomial covers `|x| ≤ 1.5` instead of `|x| ≤ 0.55`, the medium regime starts where `tanh(x) ≥ tanh(1.5) ≈ 0.905`. At `x = 1.5`, the relative error amplification is `1.5 × 2^{-52} / 0.905 ≈ 1.66 × 2^{-52}` → 1.66 ULP from steps 1–3, plus 0.5 ULP from fsub ≈ 2.16 ULP. Still over 1 ULP. The formula is mathematically tight regardless of where the polynomial boundary sits because the amplification always comes from exp's 1-ULP error compounded through two ops.

**Option B: Two-call formula.** Use `(e^x - e^{-x}) / (e^x + e^{-x})` with two `exp` calls. No cancellation (numerator and denominator are sums of positives). Error budget: 2 × 1-ULP (two exp calls) + 0.5 ULP (fadd) × 2 + 0.5 ULP (fdiv) ≈ 3.5 ULP absolute but relative to tanh(x) it's `3.5 × 2^{-52} / tanh(x)`. At `x = 0.55` this is `≈ 7 ULP` — worse.

**Option C: Compensated subtraction for the 1.0 - q step.** Use TwoSum to compute `1.0 - q` exactly, then round once. This reduces the fsub contribution to 0 ULP and makes the total `≈ 1.5 ULP`, which is still over 1 ULP.

**Option D: Direct Remez polynomial for tanh on [0.55, threshold].** Instead of the formula, fit a polynomial to `tanh(x)` directly on the transition region `[0.55, 1.5]` and connect smoothly to the large-x formula. Reaches 1 ULP with degree ~8. This is the correct Phase 1 solution: the formula-based approach cannot hit 1 ULP at the lower boundary without extended precision.

**Recommended fix:** Option D — extend the polynomial regime to cover the full region where the formula cannot reach 1 ULP. The crossover point where `1.0 - 2/(e^{2x}+1)` reliably gives 1 ULP is approximately where `2 × 1.5 × 2^{-52} / tanh(x) ≤ 1 × 2^{-52}`, i.e., `tanh(x) ≥ 3`. Since `tanh(x) < 1` always, this is never satisfied. The formula-over-entire-medium-regime approach cannot achieve 1 ULP.

**Practical recommendation:** Use two separate polynomial fits:
- `|x| ≤ T₁` (current `0.55` or wider): polynomial in `x²` (odd function)  
- `|x| ∈ (T₁, T₂]`: the formula `1 - 2/(e^{2x}+1)` is only 1-ULP accurate when `tanh(x)` is close enough to 1 that the amplification is bounded. This requires `tanh(x) ≥ 1.5` — impossible since `tanh < 1`. So: **the formula cannot achieve 1 ULP anywhere near the polynomial boundary without compensated arithmetic.**

**The structurally correct fix:** Use `expm1(2x)` instead of `exp(2x)`. The identity `tanh(x) = expm1(2x) / (expm1(2x) + 2)` avoids forming `e^{2x}` and has better small-x behavior. If `tam_expm1` is in scope for Phase 1, this formula gives 1 ULP cleanly. If not, math-researcher must either extend the polynomial regime to cover `|x| ≤ 2` (where the direct formula's error amplification is `< 1` because `tanh(2) ≈ 0.964`) and verify the formula holds 1 ULP in `(2, 22)`, or accept 2 ULP for tanh in Phase 1 as a separate carve-out with an `expm1`-based Phase 2 fix path.

**Required action from math-researcher:** Re-examine tanh medium-regime with this analysis and propose a resolution. The Phase 1 1-ULP claim for tanh cannot be validated without this fix.

---

## Verdict

**HOLD on Campsite 2.18 implementation until:**

1. B1 resolved: correct the threshold justification for `|x| = 22` (the claim `e^{-22} < 2^{-64}` is false; the correct argument involves `e^{-44} < 2^{-53}`). **[Already amended in design doc per adversarial review — verify amendment is present.]**
2. B2 resolved: change `cosh` medium-regime to use `exp(-x)` instead of `1/e_x`, with the precision analysis in the design doc.
3. B3 resolved: verify that `fabs.f64` is in the `.tam` IR op set, and add it as an IR amendment if not.
4. B4 resolved: `tanh` medium-regime formula `1 - 2/(e^{2x}+1)` cannot achieve 1 ULP near `|x| = 0.55` due to ~2 ULP error from exp error propagation through the quotient. Math-researcher must either (a) use `expm1(2x)`-based formula, (b) extend the polynomial regime far enough that the formula only runs where error is ≤ 1 ULP (approximately `|x| > 2`), or (c) file a tanh carve-out to 2 ULP Phase 1 with expm1 Phase 2 fix path. Option (c) requires navigator sign-off.
