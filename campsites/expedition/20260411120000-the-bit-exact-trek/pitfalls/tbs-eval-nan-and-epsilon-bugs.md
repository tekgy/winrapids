# Pitfall: TBS Expr Evaluation — 4 Bugs in tbs::eval

**Status: CONFIRMED BUGS — tests `eq_with_values_differing_at_last_bit_is_one`,
`sign_of_nan_is_zero_not_nan`, `tbs_max_nan_x_is_nan`, `tbs_min_nan_x_is_nan` FAIL**

## Bug 1 — Eq uses hardcoded 1e-15 epsilon (wrong for bit-exact semantics)

In `crates/tambear-primitives/src/tbs/mod.rs`:
```rust
Expr::Eq(a, b) => {
    let diff = (...a - ...b).abs();
    if diff < 1e-15 { 1.0 } else { 0.0 }
}
```

**The problem**: `1.0` and `1.0 + f64::EPSILON` differ by `2.22e-16`, which is
less than `1e-15`. So `Eq(1.0, 1.0 + epsilon)` returns `1.0` (equal). They are
NOT equal — they differ by one ULP. For the bit-exact trek, "equal" means
"same bits," not "close enough."

**Why this matters for I5/I10**: If two backends compute a value that differs by
one ULP (which is possible under different instruction sequences), an `Eq`
expression will return 1 on one backend and 0 on the other. That's a cross-backend
disagreement in a boolean result — arguably worse than a numeric disagreement.

**Fix**: Use exact floating-point equality:
```rust
Expr::Eq(a, b) => if eval(a,...) == eval(b,...) { 1.0 } else { 0.0 }
```

The 1e-15 epsilon was probably added for "fuzzy equality in user expressions."
But that belongs in a `FuzzyEq` op or a `using(tolerance=...)` parameter, not
hardcoded into the base `Eq` op. The base op must be exact.

---

## Bug 2 — Sign(NaN) returns 0.0 instead of NaN

```rust
Expr::Sign(a) => {
    let v = eval(a,...);
    if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
}
```

`NaN > 0.0` is false; `NaN < 0.0` is false; so Sign(NaN) falls through to `0.0`.
NaN is silently converted to the number 0 with no indication that the input was
undefined.

**Impact**: Any expression that uses `Sign` on a NaN-contaminated column will
produce `0.0` — a valid-looking but wrong result that masks the NaN.

**Fix**:
```rust
Expr::Sign(a) => {
    let v = eval(a,...);
    if v.is_nan() { v }  // propagate NaN
    else if v > 0.0 { 1.0 }
    else if v < 0.0 { -1.0 }
    else { 0.0 }
}
```

---

## Bug 3 — Tbs Min/Max NaN propagation is argument-order-dependent

```rust
Expr::Min(a, b) => {
    let va = eval(a,...);
    let vb = eval(b,...);
    if va <= vb { va } else { vb }
}
```

- `Min(NaN, 5.0)`: `NaN <= 5.0` is false → returns `vb = 5.0`. **NaN silently dropped.**
- `Min(5.0, NaN)`: `5.0 <= NaN` is false → returns `vb = NaN`. **NaN accidentally preserved.**

The behavior depends on which argument position the NaN is in. This is
**order-dependent NaN behavior** — a class of bug that is almost impossible to
debug because results change when you reorder operands.

The same asymmetry exists for Max:
- `Max(NaN, 5.0)`: `NaN >= 5.0` is false → returns `vb = 5.0`. NaN dropped.
- `Max(5.0, NaN)`: `5.0 >= NaN` is false → returns `vb = NaN`. NaN preserved.

**Fix**: propagate NaN from either argument:
```rust
Expr::Min(a, b) => {
    let va = eval(a,...);
    let vb = eval(b,...);
    if va.is_nan() || vb.is_nan() { f64::NAN }
    else if va <= vb { va } else { vb }
}
```

Or equivalently, use Rust's built-in `f64::min` which has *different* NaN
semantics (it propagates NaN from the second argument, not the first), so even
that can't be used as-is. Must be explicit.

---

## Summary of all NaN silent-failure sites identified so far

| Location | Bug | Test |
|---|---|---|
| `accumulates/mod.rs:146` | `Op::Min`: NaN silently skipped | `min_with_nan_is_nan` |
| `accumulates/mod.rs:147` | `Op::Max`: NaN silently skipped | `max_with_nan_is_nan` |
| `tbs/mod.rs` `Expr::Min` | NaN dropped when in first arg | `tbs_min_nan_x_is_nan` |
| `tbs/mod.rs` `Expr::Max` | NaN dropped when in first arg | `tbs_max_nan_x_is_nan` |
| `tbs/mod.rs` `Expr::Sign` | NaN → 0 | `sign_of_nan_is_zero_not_nan` |
| `tbs/mod.rs` `Expr::Eq` | 1-ULP differences treated as equal | `eq_with_values_differing_at_last_bit_is_one` |

**Pattern**: every comparison-based operation (`<`, `>`, `<=`, `>=`, `==`)
silently mishandles NaN because IEEE 754 defines all comparisons with NaN as
false. The fix in each case is an explicit `.is_nan()` check before the comparison.

This pattern will recur in the PTX assembler (Peak 3) and SPIR-V assembler
(Peak 7): their comparison instructions have the same IEEE 754 NaN semantics.
The `.tam` IR needs explicit NaN-propagation semantics documented so each backend
lowers them consistently.

## Files

- **Failing tests**: `crates/tambear-primitives/tests/adversarial_tbs_expr.rs`
  - `eq_with_values_differing_at_last_bit_is_one`
  - `sign_of_nan_is_zero_not_nan`
  - `tbs_min_nan_x_is_nan`
  - `tbs_max_nan_x_is_nan`
- **Buggy code**: `crates/tambear-primitives/src/tbs/mod.rs`
  (Eq, Sign, Min, Max match arms)
