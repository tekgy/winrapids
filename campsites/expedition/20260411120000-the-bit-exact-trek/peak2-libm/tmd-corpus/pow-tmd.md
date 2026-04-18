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

# TMD Corpus — `tam_pow`

**Owner:** Adversarial Mathematician (I9′ oracle registry)
**Date:** 2026-04-12
**Status:** Initial seed.

---

## Known TMD candidates for `pow`

### Case 1: `pow(2.0, 0.5)` = `sqrt(2)`

True value: `sqrt(2) = 1.41421356237309504880168872420969807856967...`

fp64 neighbors:
- `0x3FF6A09E667F3BCC` = `1.41421356237309492...`
- `0x3FF6A09E667F3BCD` = `1.41421356237309514...`

Midpoint: `1.41421356237309503...`

True value `...504...` vs midpoint `...503...`: the true value is ABOVE the midpoint, so correct rounding rounds UP to `0x3FF6A09E667F3BCD`.

**Test:** `assert tam_pow(2.0, 0.5).to_bits() == 0x3FF6A09E667F3BCDu64`

Note: `fsqrt.f64(2.0)` is required to give the same answer by IEEE 754 (correctly-rounded sqrt). So `tam_pow(2.0, 0.5)` should equal `fsqrt(2.0)` IF the implementation routes `b = 0.5` to `fsqrt`. If it goes through the `exp(0.5 * log(2))` path, it may give a 1-ULP error due to composition. This is the tension raised in adversarial-review-pow.md A1.

### Case 2: `pow(3.0, 1.0/3.0)` = `∛3`

True value: `3^{1/3} = 1.44224957030740838232163831078010958839186...`

fp64 neighbors:
- `0x3FF7148B0F6B5E8B`
- `0x3FF7148B0F6B5E8C`

Midpoint location relative to the true value determines correct rounding. Verify with mpmath.

Note: `1.0/3.0` is not exactly representable in fp64. The actual input to `tam_pow` is `pow(3.0, fp64(1/3))` where `fp64(1/3) ≈ 0.3333333333333333148296...`. So `tam_pow(3.0, fp64(1/3))` computes `3^{0.3333...}` not exactly `∛3`.

**Test:** Verify `tam_pow(3.0, 1.0/3.0)` against `mpmath.power(3, mpmath.mpf('0.3333333333333333148...'))` at 200 digits.

### Case 3: `pow(fp64(e), 1.0)` = `fp64(e)` bit-exact

The exponent `1.0` is exactly representable. `pow(x, 1.0)` should return `x` bit-exact (since `exp(1.0 * log(x)) = x` by definition, and the double-double infrastructure should be precise enough to hit the correct rounding for `b = 1.0`).

True value: `pow(fp64(e), 1.0)` should equal `fp64(e)` = `0x4005BF0A8B145769`.

**Test:** `assert tam_pow(std::f64::consts::E, 1.0).to_bits() == std::f64::consts::E.to_bits()`

Note: The identity `pow(x, 1.0) == x` must hold bit-exact for ALL finite `x > 0`. This is a special case of the general identity `pow(x, 1) = x`. The double-double path must be accurate enough to round to exactly `x` in this case.

### Case 4: `pow(2.0, -1.0)` = `0.5`

`pow(2.0, -1.0) = 2^{-1} = 0.5` exactly. Since `0.5` is exactly representable in fp64, this must be bit-exact.

**Test:** `assert tam_pow(2.0, -1.0) == 0.5f64`

The integer-exponent path was removed in Phase 1 (design doc decision). So this goes through `exp(-1.0 * log(2.0)) = exp(-ln(2)) = 1/2`. Since `ln(2)` is accurate to 1 ULP and `exp` is accurate to 1 ULP, the composed error is at most 2 ULPs. But `0.5` is so "clean" that the implementation should hit it exactly.

### Case 5: `pow(10.0, 3.0)` = `1000.0`

For large integer exponents, the double-double path must give exact integer results for "nice" bases. `10^3 = 1000` is exactly representable.

**Test:** `assert tam_pow(10.0, 3.0) == 1000.0f64`

### Case 6: `pow(tiny, large_b)` — underflow to subnormal

For `a = 2^{-100}` and `b = 10.8`:
- `log(a) = -100 * ln(2) ≈ -69.3`
- `b * log(a) ≈ -748.4`
- `exp(-748.4)` is in the subnormal range (below `2^{-1022}`)

This tests that `pow`'s `exp_dd` call correctly handles subnormal results.

**Test:** `assert tam_pow(2.0f64.powi(-100), 10.8)` is a subnormal (non-zero) and matches mpmath at 200 digits.

---

## Special-case table verification

All 25+ entries from the design doc's special-case table must have corresponding bit-exact tests. The most likely-to-be-wrong entries (per adversarial-review-pow.md B1):

| Input | Expected | Bit pattern |
|-------|----------|-------------|
| `pow(-0.0, 3.0)` | `-0.0` | `0x8000000000000000` |
| `pow(-0.0, 2.0)` | `+0.0` | `0x0000000000000000` |
| `pow(-0.0, -3.0)` | `-inf` | `0xFFF0000000000000` |
| `pow(-inf, 3.0)` | `-inf` | `0xFFF0000000000000` |
| `pow(-inf, 2.0)` | `+inf` | `0x7FF0000000000000` |
| `pow(-inf, -3.0)` | `-0.0` | `0x8000000000000000` |
| `pow(nan, 0.0)` | `1.0` | `0x3FF0000000000000` |
| `pow(1.0, nan)` | `1.0` | `0x3FF0000000000000` |
| `pow(nan, 1.0)` | `nan` | (NaN bit pattern varies but != 1.0) |

---

## Reference computation

```python
import mpmath
mpmath.mp.dps = 100

test_cases = [
    (2.0, 0.5),
    (3.0, 1.0/3.0),
    (2.718281828459045, 1.0),
    (2.0, -1.0),
    (10.0, 3.0),
]
for a, b in test_cases:
    result = mpmath.power(mpmath.mpf(str(a)), mpmath.mpf(str(b)))
    print(f"pow({a}, {b}) = {result}")
```

---

## Expansion protocol

When pathmaker discovers additional hard cases, log them here.


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

