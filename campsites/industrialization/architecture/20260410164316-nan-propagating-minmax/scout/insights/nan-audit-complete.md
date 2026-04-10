# NaN-Eating Audit — Complete Scout Scan
Written: 2026-04-10

## Status of Previously Cited Instances

The expedition state cited three instances as "remaining":
- `clustering.rs:666` — ALREADY FIXED: uses `crate::numerical::nan_max`
- `complexity.rs:240-241` — ALREADY FIXED: uses `crate::numerical::nan_max` and `nan_min`

Both were already fixed before this audit. The expedition state was stale on these two.

## Grep Methodology

Pattern: `fold(f64::NEG_INFINITY, f64::max)`, `fold(f64::INFINITY, f64::min)`,
`fold(0.0_f64, f64::max)`, `fold(0.0_f64, f64::min)`.

Total matches: 89 across the codebase. After excluding tests, experiments (experiment0-2),
utilities (bigfloat, format, equipartition), and cases where data is pre-cleaned before
the fold — the true production NaN-eating bugs are:

## TRUE BUGS (genuine NaN-eating in production paths)

### Bug 1: `neural.rs:242` — `softmax` eats NaN inputs

```rust
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);  // BUG
    ...
}
```

**Why it's a bug**: Logit vectors may contain NaN if upstream computation (linear layer,
loss computation) produced NaN. The softmax would silently return a non-NaN distribution
when it should propagate the NaN signal.

**Also**: `log_softmax` at line 251 has the same bug.

**Fix**: Add NaN guard before fold, OR switch to `nan_max`:
```rust
let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
```

### Bug 2: `irt.rs:275` — log-sum-exp in EAP estimation eats NaN log-weights

```rust
let max_lw = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);  // BUG
```

**Why it's a bug**: `log_weights` = log_lik + log_prior. If a theta value produces
degenerate log-likelihood (log(0) → -inf is handled, but NaN from other sources is not).
The EAP estimate would return a finite value without signaling that something went wrong.

**Fix**:
```rust
let max_lw = log_weights.iter().copied().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
```

### Bug 3: `signal_processing.rs:2232-2235` — ICA max stats eat NaN

```rust
let max_negentropy = negentropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let kurt_max = kurtoses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let kurt_min = kurtoses.iter().cloned().fold(f64::INFINITY, f64::min);
```

**Why it's a bug**: ICA negentropy and kurtosis computed from separated components may
contain NaN if a component is degenerate (e.g., all-constant). These max/min stats
would silently return non-NaN values while hiding the degenerate component.

**Fix**: Switch all three to `crate::numerical::nan_max` / `nan_min`.

### Bug 4: `numerical.rs:1172-1173` — ARMA residual amplitude eats NaN

```rust
let amp = late_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - late_x.iter().cloned().fold(f64::INFINITY, f64::min);
```

**Why**: ARMA residuals may contain NaN if model fitting diverged. The amplitude
measure would return a finite-looking number hiding the divergence.

**Fix**: `nan_max` / `nan_min`.

## SAFE (looks like bug, isn't)

The following match the pattern but are safe because data is pre-cleaned:

| File | Lines | Why Safe |
|------|-------|---------|
| `data_quality.rs:108-109` | min/max range | `clean` = `filter(!is_nan && is_finite)` on line 104 |
| `data_quality.rs:196` | max_diff | `diffs` computed from `clean` values |
| `data_quality_catalog.rs:280-292` | min/max range | `clean` filtered on line 279 |
| `complexity.rs:1533-1534` | mfdfa width | `valid_h` filtered with `is_finite()` on line 1531 |
| `numerical.rs:664` | log_sum_exp | Explicit NaN guard on line 662 |
| `nonparametric.rs:1528-1529` | KDE bounds | `clean` filtered beforehand |
| `graph.rs:778` | max_deg | degrees = sum of adjacency values (0/1, no NaN source) |
| `interpolation.rs:1463,1467` | interpolation bounds | computed from polynomial evaluations, controlled domain |
| `hypothesis.rs:2818-2819` | test helper | inside `#[cfg(test)]` helper function only |
| `linear_algebra.rs:1575` | spectral norm via SVD | sigma values from SVD are always finite |
| `irt.rs:275` | ... wait | SEE Bug 2 above — not safe |

## Priority Fix List for Pathmaker

In priority order (impact × likelihood of triggering):

1. **`neural.rs:242,251`** — softmax and log_softmax. Used in neural network inference;
   NaN logits are a common failure mode during training. HIGH priority.

2. **`signal_processing.rs:2232-2235`** — ICA stats. ICA on financial data is commonly
   used and degenerate components are frequent. HIGH priority.

3. **`irt.rs:275`** — EAP estimation. Triggered when theta grid produces degenerate
   log-likelihood. MEDIUM priority (specialized use case).

4. **`numerical.rs:1172-1173`** — ARMA amplitude. ARMA fitting can diverge on
   non-stationary series. MEDIUM priority.

## The Pattern for Wave 16 Tests (Adversarial)

For each bug above, the test pattern is:
```rust
#[test]
fn softmax_nan_input_propagates() {
    let x = vec![1.0, f64::NAN, 2.0];
    let result = softmax(&x);
    assert!(result.iter().any(|v| v.is_nan()),
        "softmax should propagate NaN, got {:?}", result);
}
```

Currently: this test FAILS (softmax silently eats the NaN and returns a valid-looking
distribution). After fix: test passes (NaN propagates through).

Same pattern for log_softmax, the ICA stats, irt EAP, and ARMA amplitude.
