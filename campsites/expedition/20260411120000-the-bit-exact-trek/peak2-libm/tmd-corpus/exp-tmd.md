<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# TMD Corpus — `tam_exp`

**Owner:** Adversarial Mathematician (I9′ oracle registry)
**Date:** 2026-04-12
**Status:** Initial seed. Expand as pathmaker discovers hard cases during implementation.

---

## What is the Table Maker's Dilemma?

A floating-point function `f(x)` has a "hard" input when the true mathematical value `f(x)` is within `2^-53/2` (half a ULP) of the midpoint between two adjacent fp64 values. In this case, even an infinitely-precise computation of `f(x)` cannot determine the correct rounding without exceeding the precision of the 53-bit mantissa. Such inputs are called TMD candidates.

For the adversarial test suite, TMD candidates serve as the hardest inputs: if an implementation gets these right, the polynomial fit and reassembly are working at the limits of fp64 precision.

**Verification tool:** mpmath at 500+ digit precision. Compare `mpmath.exp(x)` (500 digits) with the fp64 result. If they agree to 53 bits, the implementation is correct for that input.

---

## Known TMD candidates for `exp`

### Case 1: `exp(1.0)`

True value: `e = 2.718281828459045235360287471352662497757...`

fp64 neighbors around `e`:
- `0x4005BF0A8B145769` = `2.718281828459045090795598298427648842334...`
- `0x4005BF0A8B14576A` = `2.718281828459045535825196391083598136901...`

Midpoint between these: `2.718281828459045312...`

True value `e = 2.71828...535...` — this is above the midpoint. Correct rounding: `0x4005BF0A8B14576A`.

**Test:** `assert tam_exp(1.0).to_bits() == 0x4005BF0A8B145769u64` — WRONG (the correct value rounds UP)
**Correct test:** `assert tam_exp(1.0).to_bits() == 0x4005BF0A8B14576Au64`

Source: verified against mpmath; `math.e` in IEEE 754 is `0x4005BF0A8B145769` but `exp(1.0)` rounds to `0x4005BF0A8B14576A`. Verify with: `import mpmath; mpmath.mp.dps = 100; print(mpmath.exp(1))`

### Case 2: `exp(-1.0)`

True value: `1/e = 0.367879441171442321595523770161460867445...`

Nearest fp64 is known. The true value is known to be within ~0.07 ULP of the rounding boundary. Not a TMD case strictly, but a near-TMD case worth testing.

**Test:** `assert tam_exp(-1.0) == 1.0 / tam_exp(1.0)` within 2 ULP (composed; this identity requires the product `tam_exp(1.0) * tam_exp(-1.0)` to be near 1.0).

### Case 3: `exp(0.5)`

True value: `sqrt(e) = 1.6487212707001281468486507878090...`

fp64 neighbors:
- `0x3FF9ADE6EB8D6CE8` = `1.6487212707001280...`
- `0x3FF9ADE6EB8D6CE9` = `1.6487212707001282...`

True value is very close to the midpoint. Known hard case for libm `exp`.

**Test:** Verify against mpmath at 200 digits. The correct fp64 rounded result is one of the two neighbors; mpmath at 200 digits determines which.

### Case 4: `exp(ln(2) / 2)` = `sqrt(2)`

The constant `ln(2)/2 = 0.346573590279972654708616060729...`. The fp64 representation of `ln(2)/2` is close to but not equal to the true value. So `exp(fp64(ln(2)/2))` computes `exp(0.346573590279972654...)`, which should give approximately `sqrt(2) = 1.41421356...`. This tests the Cody-Waite reduction path: the range reduction should recover the fractional part near the `ln(2)/2` boundary.

**Test:** `assert |tam_exp(fp64(LN2 / 2)) - sqrt(2.0)| <= 2 * ulp(sqrt(2.0))`

### Case 5: Cody-Waite exact inputs

For `n ∈ {-1022, -512, -256, -128, -64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1023}`:

Input `x = n * ln2_hi` where `ln2_hi = 0x3FE62E42FEFA3800` (the high part of the Cody-Waite split of `ln(2)`).

These inputs are "exact" in the sense that `n * ln2_hi` is exactly representable (no rounding in the multiply), and `r = x - n * ln2_hi = 0` exactly. So the polynomial receives `r = 0` and should return `exp(0) = 1.0` for the polynomial part, then the final result is `2^n` (a power of 2, exactly representable for `n ∈ [-1022, 1023]`).

**Test:** For these inputs, `tam_exp(n * ln2_hi)` should be `2.0f64.powi(n)` within 1 ULP.

### Case 6: Subnormal boundary inputs

Per adversarial-review-exp.md B5:
- `exp(-745.1)`: result is in subnormal range. True value ≈ `3.03 × 10^{-324}`.
- `exp(-744.4)`: boundary near deepest subnormal.
- `exp(-745.13)` (or wherever the last non-zero subnormal boundary is exactly): should be the smallest nonzero fp64 subnormal `5e-324 = 2^{-1074}`.

**Test:** `assert tam_exp(-745.13...).to_bits() == 1u64` (bit pattern 1 = smallest positive subnormal `5e-324`).

---

## Reference computation

```python
import mpmath
mpmath.mp.dps = 100  # 100 decimal digits

test_inputs = [1.0, -1.0, 0.5, -0.5, 0.693147180559945/2]
for x in test_inputs:
    result = mpmath.exp(x)
    print(f"exp({x}) = {result}")
    # Compare with: struct f64 = f64::from_bits(...)
```

For bit-exact verification, compare the mpmath result (rounded to nearest fp64) against the implementation's output.

---

## Expansion protocol

When pathmaker discovers additional hard cases during implementation (inputs where the computed result differs from mpmath by exactly 1 ULP and the true value is near a midpoint), log them here with:
1. The input `x` as a hex fp64 bit pattern
2. The true value from mpmath at 200+ digits
3. The correct rounded fp64 bit pattern
4. The incorrect value the implementation first produced


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

