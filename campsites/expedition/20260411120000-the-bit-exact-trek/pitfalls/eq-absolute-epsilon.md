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
