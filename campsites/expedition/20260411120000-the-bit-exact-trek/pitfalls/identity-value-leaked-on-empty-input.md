# Pitfall: Accumulator Identity Value Leaks on Empty Input

**Status: CONFIRMED BUG — tests `min_empty_is_not_silently_inf`,
`max_empty_is_not_silently_neg_inf`, `linf_norm_all_nan_is_nan` FAIL**

## What broke

- `min_all([])` returns **+Inf** (should be NaN — undefined for empty set)
- `max_all([])` returns **-Inf** (should be NaN — undefined for empty set)
- `linf_norm([NaN, NaN, NaN])` returns **-Inf** (should be NaN)

## Root cause

In `execute_pass_cpu`, accumulators are initialized to the operation's identity:
```rust
let mut accs: Vec<f64> = pass.slots.iter().map(|_| match pass.op {
    Op::Add => 0.0,
    Op::Max => f64::NEG_INFINITY,
    Op::Min => f64::INFINITY,
    Op::Mul => 1.0,
    ...
}).collect();
```

When the data loop runs zero iterations (empty input) or all NaN iterations
(NaN-poisoned — see `nan-silent-ignored-by-min-max.md`), the accumulator
is never updated. The initialization value escapes as the result.

For `Add`, the identity `0.0` is a defensible answer for empty sum.
For `Mul`, the identity `1.0` is a defensible answer for empty product.
For `Min` and `Max`, the identities are +Inf and -Inf — **not meaningful results**
for an empty set. They look like extreme outliers to any downstream consumer.

### The linf_norm case

`linf_norm` uses `Op::Max` on `|x|`. For all-NaN input:
- Each element: `abs(NaN)` = NaN (the TBS `Abs` expression preserves NaN)
- NaN comparison: `NaN > NEG_INFINITY` is false (IEEE 754)
- The accumulator stays at NEG_INFINITY — the identity leaks out

This is the same NaN-ignoring bug from `nan-silent-ignored-by-min-max.md`,
but manifesting differently: instead of returning the min/max of non-NaN values,
it returns -Inf because ALL values are NaN.

## Why this is dangerous

`+Inf` returned as the min of a dataset is a **plausible-looking pathological value**
that could be treated as "this column has an extreme minimum." It's not flagged as
an error. A downstream consumer would have to know to check for Inf separately.

`-Inf` returned as the max norm of a dataset with NaN values similarly looks like
a real result to a consumer that doesn't inspect it carefully.

## The fix

Post-loop: if the data was empty (or all-NaN for min/max), return NaN.

```rust
// After the accumulation loop:
let n_processed = data.len(); // or count of non-NaN elements
for (j, (_, name)) in pass.slots.iter().enumerate() {
    match pass.op {
        Op::Min if n_processed == 0 => accs[j] = f64::NAN,
        Op::Max if n_processed == 0 => accs[j] = f64::NAN,
        _ => {}
    }
}
```

A cleaner approach: track whether any element was actually accumulated:
```rust
let mut any_valid = false;
for &x in data.iter() {
    let val = eval(expr, x, ...);
    if !val.is_nan() {
        // accumulate normally
        any_valid = true;
    }
}
if !any_valid {
    // set all accumulators to NaN for Min/Max
}
```

## Decision needed: what is the contract?

Should we use **propagate-NaN** (any NaN in → NaN out) or **skip-NaN** semantics?

Most statistical software uses skip-NaN by default (e.g. R's `mean(c(1, NA, 3), na.rm=TRUE)`).
But **skip-NaN is silent** — it changes the effective n. The tambear philosophy is
to be explicit. The right contract is:
- By default: propagate-NaN (contaminated data → NaN result)
- Via `using(na.rm=true)`: skip-NaN with explicit acknowledgment

For now, the failing tests assert propagate-NaN semantics. That is the correct
default for bit-exact, no-surprises arithmetic.

## Files

- **Failing tests**: `crates/tambear-primitives/tests/adversarial_baseline.rs`
  - `min_empty_is_not_silently_inf`
  - `max_empty_is_not_silently_neg_inf`
  - `linf_norm_all_nan_is_nan`
- **Buggy code**: `crates/tambear-primitives/src/accumulates/mod.rs:129-136`
  (initialization) and `146-148` (comparison without NaN check)
