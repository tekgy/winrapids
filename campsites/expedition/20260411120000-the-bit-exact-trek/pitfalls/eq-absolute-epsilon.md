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

# Pitfall: Expr::Eq Uses Absolute Epsilon 1e-15

**Status: FIXED in tbs::eval and codegen/cuda.rs (scout, 2026-04-11). Pitfall doc retained as specification for fcmp_eq.f64 IR semantics.**
**Original blast radius was zero — no recipe used Expr::Eq. Fix landed proactively.**

## What the code does

`tbs/mod.rs:278`:
```rust
Expr::Eq(a, b) => {
    let diff = (eval(a,...) - eval(b,...)).abs();
    if diff < 1e-15 { 1.0 } else { 0.0 }
}
```

`codegen/cuda.rs:99`:
```c
(fabs((A) - (B)) < 1e-15 ? 1.0 : 0.0)
```

Both use the same absolute epsilon. They agree — both are wrong in the same way.

## Where it fails

**Case 1: values near zero.** If A and B are both on the order of 1e-20, then two
values that differ by 1e-20 (an enormous relative difference — 100% error) compare
as equal because `|diff| = 1e-20 < 1e-15`.

```
A = 1e-20, B = 2e-20
diff = 1e-20 < 1e-15 → Eq returns 1.0 (equal!)
true relative error: 100%
```

**Case 2: values with large magnitude.** Two f64 values that differ by exactly 1 ULP
near the magnitude 1e-15 may have `|diff| = 1e-15` exactly — which is NOT less than
1e-15, so they compare as NOT equal even though they're as close as IEEE 754 allows.

More precisely: near 1.0, 1 ULP ≈ 2.2e-16 (half of epsilon). Two values that differ
by 4 ULPs have `|diff| ≈ 8.9e-16 < 1e-15`, so they compare as equal even though
they differ by 4 ULPs.

**Case 3: NaN.** `Eq(NaN, NaN)` should return 0.0 (NaN is not equal to anything).
Current implementation: `diff = (NaN - NaN).abs() = NaN.abs() = NaN`. Then
`NaN < 1e-15` is false → returns 0.0. Accidentally correct, but for the wrong reason.
If the implementation ever changes (e.g., short-circuit on NaN), this coincidence breaks.

## Why current blast radius is zero

`grep -r "Expr::Eq" crates/tambear-primitives/src/recipes/` finds no matches.
No current recipe uses equality comparison in its accumulate or gather expressions.
The `adversarial_tbs_expr.rs` failing test documents the bug but cannot be triggered
by any current recipe running through the GPU.

## The fix (as landed)

**Scientist correction (2026-04-11):** The initial proposed fix of `va.to_bits() == vb.to_bits()` was wrong.
`+0.0` and `-0.0` have different bit patterns (`0x0000...` vs `0x8000...`) but are mathematically equal.
`to_bits()` would make them compare unequal, introducing a new bug while fixing the old one.

The correct fix — IEEE value equality with explicit NaN propagation:

```rust
Expr::Eq(a, b) => {
    let va = eval(a, ...);
    let vb = eval(b, ...);
    if va.is_nan() || vb.is_nan() { f64::NAN } else if va == vb { 1.0 } else { 0.0 }
}
```

- `+0.0 == -0.0` → 1.0 (IEEE 754: they compare equal)
- `NaN == anything` → NaN (propagates, doesn't silently return 0.0)
- `1.0 == 1.0` → 1.0
- `1.0 == 1.0 + ulp` → 0.0

**Also fixed:** `Gt` and `Lt` had the same missing NaN guard (scientist catch). Same fix applied.

The IR spec must document which semantics `fcmp_eq.f64` uses. Choices:
1. **Bit-exact**: `+0 != -0`, NaN != NaN. Use for "are these the exact same bits?"
2. **IEEE value equality**: `+0 == -0`, NaN != NaN. Use for "are these the same value?" (this is what landed)
3. **Unordered equality**: NaN == NaN. Unusual — avoid unless there's a specific need.

**Recommendation**: tambear policy is IEEE value equality (+0 == -0) with NaN propagation
(Eq(NaN, anything) → NaN). This matches the fix in tbs::eval and the analogous behavior
in fmin/fmax/Gt/Lt. Consistent NaN propagation across ALL comparison ops.

If "fuzzy" equality is needed, it should be a separate expression:
`ApproxEq(a, b, tol)` where `tol` is explicit and caller-supplied, not hardcoded.

## Action required — pathmaker (campsite 1.9)

When designing the `fcmp_eq.f64` op in the `.tam` IR spec:

1. Choose the semantics explicitly (bit-exact vs IEEE value equality)
2. Document in the IR spec what `+0 == -0` returns
3. Document NaN behavior (should propagate to NaN, not silently 0.0)
4. Ensure the CPU interpreter and PTX assembler implement the same semantics

The current `tbs::eval` Eq bug should be fixed at the same time the IR spec defines
`fcmp_eq.f64` — the two are the same decision made in the same pass.

## Related

- Failing test: `adversarial_tbs_expr.rs::eq_with_values_differing_at_last_bit_is_one`
- Same bug in GPU codegen: `codegen/cuda.rs:99`
- Both must be fixed together — fixing only CPU will cause CPU/GPU disagreement on Eq


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

