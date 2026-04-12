# Pitfall: Expr::Eq Uses Absolute Epsilon 1e-15

**Status: KNOWN BUG — no recipe currently uses Expr::Eq, so blast radius is zero today.**
**Flag to pathmaker when they reach campsite 1.9 (type-checker) and the .tam IR fcmp_eq op.**

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

## The fix — when it's needed

The correct semantics for `fcmp_eq.f64` in the `.tam` IR are bit-exact equality:

```rust
Expr::Eq(a, b) => {
    let va = eval(a, ...);
    let vb = eval(b, ...);
    if va.to_bits() == vb.to_bits() { 1.0 } else { 0.0 }
}
```

And correspondingly in `codegen/cuda.rs`:
```c
(__double_as_longlong(A) == __double_as_longlong(B) ? 1.0 : 0.0)
```

Note: `+0.0` and `-0.0` have different bit patterns but are mathematically equal.
The IR spec must document which semantics `fcmp_eq.f64` uses. Choices:
1. **Bit-exact**: `+0 != -0`, NaN != NaN. Use for "are these the exact same bits?"
2. **IEEE value equality**: `+0 == -0`, NaN != NaN. Use for "are these the same value?"
3. **Unordered equality**: NaN == NaN. Unusual — avoid unless there's a specific need.

**Recommendation**: tambear policy should be IEEE value equality (+0 == -0) with NaN
propagation (Eq(NaN, anything) → NaN, not 0.0). This aligns with how the other boolean
expressions (Gt, Lt) handle NaN: they return 0.0, which means "the comparison is false."
Consistently, Eq(NaN, x) should return 0.0 (the comparison cannot be established).

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
