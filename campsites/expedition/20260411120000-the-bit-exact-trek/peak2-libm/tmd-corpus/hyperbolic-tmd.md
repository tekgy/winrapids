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

# TMD Corpus — `tam_sinh`, `tam_cosh`, `tam_tanh`

**Owner:** Adversarial Mathematician (I9′ oracle registry)
**Date:** 2026-04-12
**Status:** Initial seed.

---

## Known TMD candidates for hyperbolic functions

### Case 1: `sinh(1.0)`

True value: `(e - 1/e) / 2 = 1.17520119364380145688238185059560081515...`

fp64 neighbors:
- `0x3FF2CD9FC44EB982` = `1.17520119364380134...`
- `0x3FF2CD9FC44EB983` = `1.17520119364380157...`

Midpoint: `1.17520119364380146...`. True value `...145...` is extremely close to the midpoint at `...146...`. This is a near-TMD case — within 0.1 ULP of the rounding boundary. High probability of implementation-dependent rounding.

**Test:** Verify `tam_sinh(1.0)` against `mpmath.sinh(1)` at 200 digits. The correct fp64 rounding must be verified, not assumed.

### Case 2: `cosh(1.0)`

True value: `(e + 1/e) / 2 = 1.54308063481524377847790562075984737637...`

fp64 representation: the 15th–17th significant digits of the true value determine the rounding. Verify with mpmath.

**Test:** `assert tam_cosh(1.0)` matches `mpmath.cosh(1)` at 200 digits, rounded to nearest fp64.

### Case 3: `tanh(0.5)`

True value: `tanh(0.5) = (e^{0.5} - e^{-0.5}) / (e^{0.5} + e^{-0.5}) = (e - 1)/(e + 1)` (no, that's tanh(0.5) via e)

Let me compute: `e^{0.5} = sqrt(e) ≈ 1.64872127...`. So `tanh(0.5) = (1.64872 - 0.60653) / (1.64872 + 0.60653) = 1.04219 / 2.25525 = 0.46212...`

True value: `tanh(0.5) = 0.462117157260009758502130974425756959370...`

This input falls in the medium regime (`|x| ≥ 0.55`? No, `0.5 < 0.55`). So it uses the polynomial. The polynomial fit should be evaluated.

Verify with mpmath: `mpmath.tanh(0.5)` at 200 digits.

### Case 4: `tanh(1.0)`

True value: `tanh(1.0) = 0.761594155955764888119458282604793760420...`

This is near the midpoint between fp64 values in the `[0.75, 1.0)` range. Known near-TMD case.

fp64 neighbors near `0.7615941559557649`:
- `0x3FE85EFAB514F394`
- `0x3FE85EFAB514F395`

True value `...888...` vs fp64 boundary — verify with mpmath.

**Test:** `assert tam_tanh(1.0).to_bits()` matches correct rounding of mpmath result.

### Case 5: Piecewise boundary precision tests

The piecewise boundaries for `tanh` (`|x| = 0.55`) and `sinh` (`|x| = 1.0`) are adversarially important because both branches must agree to 1 ULP at the boundary.

**Test at boundary for `tanh`:**

```
x_below = 0.55 - f64::EPSILON * 0.55  # fp64 just below 0.55
x_at    = 0.55f64                      # fp64 representation of 0.55
x_above = 0.55 + f64::EPSILON * 0.55  # fp64 just above 0.55
```

`tam_tanh(x_below)` uses polynomial; `tam_tanh(x_above)` uses formula. Both should give results within 1 ULP of `mpmath.tanh(0.55)`.

**Test at boundary for `sinh`:**

```
x_below = 1.0 - f64::EPSILON
x_at    = 1.0f64
x_above = 1.0 + f64::EPSILON
```

`tam_sinh(x_below)` uses polynomial; `tam_sinh(x_above)` uses formula. Both within 1 ULP of `mpmath.sinh(1.0)`.

### Case 6: `cosh(-0.0)` and `sinh(-0.0)` signed zeros

- `cosh(-0.0)` must return `1.0` bit-exact (= `0x3FF0000000000000`)
- `sinh(-0.0)` must return `-0.0` bit-exact (= `0x8000000000000000`)

These are not TMD cases (the correct values are exact), but they are adversarial bit-exact tests.

### Case 7: Large argument near overflow boundary

`sinh(709.0)` — the input `709.0` is slightly below the overflow boundary. The result is a large finite number. `cosh(709.0)` similarly.

True values: `mpmath.sinh(709)` and `mpmath.cosh(709)` at 100 digits determine the correct fp64 rounding.

`sinh(710.0)` should overflow to `+inf`. `sinh(709.9)` should be finite. The exact crossover point determines where the large-regime formula `exp(x - ln(2))` also overflows.

`exp(709.0 - ln(2)) = exp(709.0 - 0.693...) = exp(708.306...) ≈ 2.8 × 10^{307}` — finite, just under fp64 max (`≈ 1.8 × 10^{308}`).

`exp(710.0 - 0.693...) = exp(709.306...) ≈ 7.5 × 10^{307}` — still finite.

`exp(711.0 - 0.693...) = exp(710.306...)` — this overflows to `+inf`.

**Test:** Verify the exact overflow threshold for `sinh` using `exp(x - ln(2))`: the overflow occurs when `x - ln(2) > ln(fp64_max) ≈ 709.78`. So `sinh(710.472...)` is the approximate overflow point. Test `sinh(710)` is finite and `sinh(712)` is `+inf`.

---

## Reference computation

```python
import mpmath
mpmath.mp.dps = 100

test_inputs = [1.0, 0.5, 1.0, 0.55, 710.0, 1e-10]
for x in test_inputs:
    print(f"sinh({x}) = {mpmath.sinh(mpmath.mpf(str(x)))}")
    print(f"cosh({x}) = {mpmath.cosh(mpmath.mpf(str(x)))}")
    print(f"tanh({x}) = {mpmath.tanh(mpmath.mpf(str(x)))}")
```

---

## Expansion protocol

When pathmaker discovers additional hard cases during implementation, log them here.


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

