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

# `tam_tan` — Algorithm Design Document

**Owner:** math-researcher
**Status:** draft, awaiting navigator + pathmaker review
**Date:** 2026-04-12

**Upstream dependency:** `sin-cos-design.md`, `accuracy-target.md`.
**Downstream:** low-frequency consumer — most statistical code doesn't use tan directly. Kept in Phase 1 because adversarial's `special-values-matrix.md` includes tan at 2 ULP.

---

## Accuracy target

**Phase 1 bound: `max_ulp ≤ 2.0`** on 1M random samples, exponent-uniform over `|x| ≤ 2^30` (same bound as sin/cos via three-term Cody-Waite). Per adversarial's matrix; team-lead implicitly approved via sign-off.

**Pole-exclusion clause** (navigator ruling 2026-04-12, via adversarial B2): The 2-ULP bound applies on `|x| ≤ 2^30` **excluding** inputs where `|cos(x_f64)| < 2^-26`. In the pole-exclusion zone, the oracle runner flags the input (sign + finiteness check only) and does not count it against the ULP bar. The exclusion threshold `2^-26` is approximately 1 ULP of `cos(x)` near `cos(x) ≈ 0` — inside this neighborhood, a 1-ULP error in the input produces an unbounded error in `tan` by the chain rule. Excluding this zone makes the 2-ULP claim meaningful and measurable.

Why 2 ULP not 1 ULP: tan has **poles at `x = (k + 1/2)·π`** where the function diverges to `±inf`. Near these poles the function's output is catastrophically sensitive to the input's precision — a 1-ULP error in `x` translates to an unbounded error in `tan(x)`. No Phase 1 implementation can reach 1 ULP uniformly over the domain; 2 ULP is the honest bar given that we compute `tan = sin/cos` with a near-pole guard.

## IR stub layout

**No new stub.** `tam_tan` is a regular `.tam` function that calls `tam_sin` and `tam_cos` (both existing stubs). Same pattern as sinh/cosh/tanh — composition, no new opcode.

| Function | Mechanism | Stub needed? |
|---|---|---|
| `tam_tan` | pure `.tam` function calling `tam_sin` + `tam_cos` + `fdiv` | No |

## Algorithm

```
tam_tan(x) = tam_sin(x) / tam_cos(x)
```

That's it. Both `tam_sin` and `tam_cos` have 1-ULP accuracy per their own designs; their quotient has ~2 ULP accuracy by composition (1 ULP each + 0.5 ULP from fdiv = ~2 ULP worst case, usually better in practice).

### Near-pole behavior

Near `x = k·π + π/2`, `cos(x) → 0` and `tan(x) → ±inf`. The `fdiv.f64` in the implementation produces `sin/cos → ±inf` naturally when cos is zero (as an fp64 operation); we don't need special handling at the exact pole. At points *near* the pole where cos is tiny but nonzero, the naive `sin/cos` is still correct by IEEE 754 — the error in tan is bounded by the relative error in the quotient, not by any near-pole pathology in our algorithm.

**What the adversarial matrix note [9] warns about** is that `tan(f64::consts::PI / 2)` is NOT `±inf` because `f64`'s representation of π/2 is slightly below the true value. The result is a very large finite number (~1.633e16). Our implementation gives this automatically — no special case needed — because we feed the exact f64 input to sin and cos and divide. mpmath's oracle at the same exact f64 input agrees.

### Special-value handling

The same front-end dispatch pattern as sin/cos. The NaN/inf front-end checks:

```
if isnan(x): return x         ; I11 preservation
if isinf(x): return nan       ; tan(±inf) undefined per IEEE 754
; ELIDE explicit zero check — sign-of-zero falls out of the composition:
;   tam_sin(±0) = ±0, tam_cos(±0) = 1.0, fdiv(±0, 1.0) = ±0 per IEEE 754.
;   DO NOT use "if x == +0: return +0" / "if x == -0: return -0" — because
;   fcmp_eq(+0.0, -0.0) = true per IEEE 754 §5.10, so the first branch
;   would catch -0 and return +0 (wrong). The composition path is correct
;   without the explicit branch. (Adversarial B1 resolution, 2026-04-12.)
s = tam_sin(x)
c = tam_cos(x)
return s / c
```

### Out-of-domain inputs

For `|x| > 2^30`, the reduction is out of spec (Phase 1 cap, same as sin/cos). Return `nan` front-end.

## Pitfalls

1. **Naive `tan = sin/cos` at the pole is fine.** IEEE 754's `fdiv` of a finite nonzero by `+0` produces `+inf` correctly. Don't add a special-case branch that "helpfully" returns `+inf` before the division — it would break the sign of `tan(π/2 - ε)` near the pole.

2. **The `x = k·π` case.** `tan(k·π) = 0` mathematically, but `f64::consts::PI * k` is not exactly `k·π` for any nonzero `k`, so `sin(k·π_f64)` is tiny but nonzero and `cos(k·π_f64)` is ≈ ±1. The quotient is tiny, not zero. This is correct per the mpmath oracle at the exact f64 input.

3. **Symmetry.** `tan(-x) = -tan(x)` bit-exact. The `sin(-x) / cos(-x) = -sin(x) / cos(x)` composition preserves this automatically because sin preserves the negative sign and cos doesn't.

4. **No polynomial of its own.** Don't fit a separate Remez polynomial for tan in Phase 1. The composition path is simpler, bit-deterministic across backends (same ops, same order), and hits the 2 ULP target. Phase 2 could use a dedicated polynomial on a reduced interval to drop to 1 ULP, but that's Phase 2.

## Testing

Per adversarial's special-values matrix `tan` column:

| Input | Expected |
|---|---|
| `+0` | `+0` exact |
| `-0` | `-0` exact |
| `+inf` | `nan` (sin/cos undefined at infinity) |
| `-inf` | `nan` |
| `nan` | `nan` |
| `π/4` | `≈ 1.0` within 2 ULP |
| `f64::consts::PI / 2` | `~1.633e16` (NOT inf — mpmath oracle) |
| `±subnormal` | `≈ ±subnormal` — tan(x) ≈ x for tiny x |

Plus 1M exponent-uniform samples on `|x| ≤ 2^30`, plus the sign-symmetry identity `tan(-x) == -tan(x)` bit-exact.

## Open questions

1. **Is the composition ordering `s = sin(x); c = cos(x); return s/c` bit-deterministic across backends?** Yes, because:
   - Both sin and cos are deterministic pure functions of `x`.
   - The fdiv is a single IEEE 754 op.
   - So the same input `x` produces the same intermediate `s`, `c`, and the same quotient on every backend.
   - This holds AS LONG AS sin and cos themselves are cross-backend deterministic, which is the whole point of Peak 2.

2. **Could Phase 1 `tan` cheat by computing `sin` and `cos` together via a shared polynomial evaluation?** Maybe. The shared `r = x - k·π/2` range reduction is valuable. But as a Phase 1 implementation, the two-function composition is simpler and pathmaker can optimize later. **Defer.**

## References

- W. J. Cody & W. Waite, "Software Manual for the Elementary Functions," Prentice-Hall, 1980, §§3–4 (tan derivation as sin/cos).
- J.-M. Muller et al., "Handbook of Floating-Point Arithmetic," 2nd ed., 2018, Chapter 11.
- Adversarial's `special-values-matrix.md` `tan(x)` column and note [9] on `tan(π/2_f64) ≠ ±inf`.


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

