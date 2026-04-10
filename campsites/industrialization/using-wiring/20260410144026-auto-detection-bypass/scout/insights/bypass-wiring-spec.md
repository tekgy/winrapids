# Auto-Detection Bypass — Wiring Specification
Written: 2026-04-10

## What This Is

The tbs_executor's auto-detection layer (`correlation`, `t_test_2`, `anova`, `regression`,
`cluster_analyze`, `ts_analyze`, `vol_analyze`, `dim_reduce`) makes method-selection decisions
using hardcoded thresholds. The `using_bag` is threaded through `execute()` — it's already
there, just not consulted for these decisions. This is the complete spec for wiring all 26
violations.

## The Fix Pattern — Memorize This Once

```rust
// BEFORE (hardcoded):
let x_norm = px > 0.05;

// AFTER (using() wired):
let normality_alpha = using_bag.get_f64("normality_alpha").unwrap_or(0.05);
let x_norm = px > normality_alpha;
```

For usize values (there's no `get_usize` on UsingBag — use get_f64 + cast):
```rust
// BEFORE:
if col.len() < 5000 { shapiro_wilk(...) } else { dagostino_pearson(...) }

// AFTER:
let n_threshold = using_bag.get_f64("normality_test_n_threshold")
    .map(|v| v as usize).unwrap_or(5000);
if col.len() < n_threshold { shapiro_wilk(...) } else { dagostino_pearson(...) }
```

## Priority 1 — Auto-Detection Decision Gates (Fix These First)

These silently select which statistical test runs. A user who prefers α=0.01 has zero
recourse today. These are the most user-impactful violations.

### V-04: normality_alpha and variance_alpha (tbs_executor.rs)

**Lines to fix**: 484, 485, 603, 604, 614, 758, 764, 789, 1148, 1152

At the TOP of each auto-detection function (before any decision branches), add:
```rust
let normality_alpha = using_bag.get_f64("normality_alpha").unwrap_or(0.05);
let variance_alpha = using_bag.get_f64("variance_alpha").unwrap_or(0.05);
```

Then replace every `> 0.05` / `< 0.05` that is part of a normality or variance decision
with `> normality_alpha` / `< normality_alpha` and `> variance_alpha` / `< variance_alpha`.

Note: Do NOT replace `0.05` in test assertions (hypothesis.rs, nonparametric.rs) — those
are hardcoded CORRECT (they're testing the output, not making a decision). The audit's
false-positives section lists exactly which lines to skip.

### V-03: normality_test_n_threshold (tbs_executor.rs)

**Lines to fix**: 452, 593, 749, 1142

All four occurrences are identical: `if col.len() < 5000 {...}`. Fix:
```rust
let n_thresh = using_bag.get_f64("normality_test_n_threshold")
    .map(|v| v as usize).unwrap_or(5000);
if col.len() < n_thresh { shapiro_wilk(...) } else { dagostino_pearson(...) }
```

### V-05: vif_threshold (tbs_executor.rs line 1139)

```rust
// Before:
let multicollinear = max_vif > 10.0;

// After:
let vif_threshold = using_bag.get_f64("vif_threshold").unwrap_or(10.0);
let multicollinear = max_vif > vif_threshold;
```

### V-06: kmo_threshold and bartlett_alpha (tbs_executor.rs lines 1233, 1249)

```rust
let kmo_threshold = using_bag.get_f64("kmo_threshold").unwrap_or(0.5);
let bartlett_alpha = using_bag.get_f64("bartlett_alpha").unwrap_or(0.05);
// ...
let pca_viable = kmo >= kmo_threshold && bartlett_p < bartlett_alpha;
if kmo < kmo_threshold { ...
```

### V-07: hopkins_threshold (tbs_executor.rs line 1009)

```rust
let hopkins_threshold = using_bag.get_f64("hopkins_threshold").unwrap_or(0.5);
let has_structure = hopkins > hopkins_threshold;
```

### V-09: arch_alpha (tbs_executor.rs lines 1450, 1901)

Two locations — `time_series_analyze` and `vol_analyze`:
```rust
let arch_alpha = using_bag.get_f64("arch_alpha").unwrap_or(0.05);
let has_arch = arch_p < arch_alpha;
```

## Priority 2 — Auto-Path Parameter Passthrough (PARTIAL violations)

These are where the NAMED STEP correctly reads from `usize_arg`/`f64_arg`, but the
auto-path (the code path when the user doesn't name the step) hardcodes the same values.

### V-11: ewma_lambda in vol_analyze auto-path (tbs_executor.rs line 1926)

```rust
let ewma_lambda = using_bag.get_f64("ewma_lambda").unwrap_or(0.94);
let values = crate::volatility::ewma_variance(&col, ewma_lambda);
```

### V-12: max_lag formula in time_series_analyze auto-path (tbs_executor.rs line 1442)

```rust
let max_lag = using_bag.get_f64("max_lag")
    .map(|v| v as usize)
    .unwrap_or_else(|| (col.len() / 4).min(20).max(1));
```

### V-13: garch_max_iter in vol_analyze auto-path (tbs_executor.rs line 1905)

```rust
let garch_max_iter = using_bag.get_f64("garch_max_iter")
    .map(|v| v as usize).unwrap_or(200);
let garch = crate::volatility::garch11_fit(&col, garch_max_iter);
```

### V-10: ARCH lag cap and fraction denominator (tbs_executor.rs lines 1447, 1898)

```rust
let arch_lags_cap = using_bag.get_f64("arch_lags_cap")
    .map(|v| v as usize).unwrap_or(5);
let arch_lags_denom = using_bag.get_f64("arch_lags_fraction_denom")
    .map(|v| v as usize).unwrap_or(10);
let arch_lags = arch_lags_cap.min(col.len() / arch_lags_denom).max(1);
```

### V-08: scale_ratio_warn_threshold (tbs_executor.rs line 999)

```rust
let scale_ratio_thresh = using_bag.get_f64("scale_ratio_warn_threshold").unwrap_or(10.0);
if scale_ratio > scale_ratio_thresh { lints.push(...) }
```

### P-01: influence_threshold for Cook's D (tbs_executor.rs line 1155)

The primitive `cooks_distance_with_threshold` already takes `Option<f64>`. The executor
should pass the using_bag value instead of always calling the no-threshold variant:
```rust
let influence_threshold = using_bag.get_f64("influence_threshold");
let influence = crate::hypothesis::cooks_distance_with_threshold(
    x_with_intercept, residuals, influence_threshold
);
```

## Priority 3 — Primitive-Level Violations (Non-Executor)

These are in tambear primitives directly, not the executor layer. Different fix discipline —
add the parameter to the function signature and thread it through.

### V-01: garch_is_valid in data_quality.rs (line 388)

```rust
// Add min_vol_clustering_acf parameter:
pub fn garch_is_valid(returns: &[f64], min_vol_clustering_acf: Option<f64>) -> bool {
    let threshold = min_vol_clustering_acf.unwrap_or(0.05);
    has_vol_clustering(returns, threshold) && ...
}
```
Callers in tbs_executor: pass `using_bag.get_f64("min_vol_clustering_acf")`.

### V-02: DataQualitySummary::from_slice (data_quality.rs lines 623, 627, 628)

Add a parameterized constructor:
```rust
pub fn from_slice_with_params(
    x: &[f64],
    jump_k: f64,
    vol_clustering_acf: f64,
    trend_r2_threshold: f64,
) -> Self { ... }

pub fn from_slice(x: &[f64]) -> Self {
    Self::from_slice_with_params(x, 3.0, 0.1, 0.5)
}
```
tbs_executor callers: use `from_slice_with_params` with values from `using_bag`.

### V-19: pinv SVD threshold (linear_algebra.rs line 650)

```rust
pub fn pinv(a: &[Vec<f64>], rcond: Option<f64>) -> Vec<Vec<f64>> {
    let tol = rcond.unwrap_or(1e-12);
    // ...
    if svd_res.sigma[i] > tol { ... }
}
```
Old callers: pass `None` for backward compat (same behavior).

### V-16/V-17/V-18: IRT clamp bounds (irt.rs)

Lower priority — add `IrtFitParams` struct or optional parameters to `fit_2pl`.
Not urgent unless psychometrics research is active.

### V-15: scoring.rs adaptive threshold (scoring.rs lines 42, 54)

```rust
let empty_fallback = empty_fallback_threshold.unwrap_or(0.5);
let floor = min_adaptive_threshold.unwrap_or(0.3);
```
These are in `threshold_adaptive` — add the two parameters to its signature.

## Implementation Order for the Pathmaker

**Step 1** (one hour): V-03 + V-04 + V-05 + V-06 + V-07 + V-09 — all in tbs_executor.rs.
These are pure search-and-replace once you add the 4-5 let-bindings at function tops.
One function at a time: `correlation`, `t_test_2`, `anova`, `regression`, `cluster_analyze`,
`dim_reduce`, `time_series_analyze`, `vol_analyze`.

**Step 2** (30 min): V-10 + V-11 + V-12 + V-13 + V-08 + P-01 — auto-path parameter
passthrough. Same file, same pattern.

**Step 3** (30 min): V-19 — add `rcond: Option<f64>` to `pinv`, update all callers with `None`.

**Step 4** (45 min): V-01 + V-02 — data_quality.rs parameterization + executor wiring.

**Step 5** (when IRT is in scope): V-16 + V-17 + V-18.

## What NOT to Touch

The audit identified these as false positives — leave them alone:
- All `0.05` in `#[test]` blocks (test assertions, not decisions)
- `1.4826` MAD consistency factor (mathematical constant)
- `0.9`, `1.34`, `1.06` in bandwidth rules (define specific estimators)
- `0.99` near-IGARCH boundary (definitional, not tunable)
- `tbs_executor.rs:f64_arg("huber_k", ..., 1.345)` — already exposed via TBS syntax

## The Deeper Issue: using_bag Scope

One subtlety: `using_bag` in tbs_executor is currently populated at the STEP level via
`using` chain elements on individual steps. The auto-detection paths (`correlation`,
`cluster_analyze`, etc.) fire MULTIPLE internal steps. The question for the pathmaker:

Should `normality_alpha` in `using()` apply globally to the session, or only to the
current step?

Current architecture: `using_bag.drain()` is called between steps (clears after each).
If a user writes `correlation(x,y).using(normality_alpha=0.01)`, the using() applies to
the `correlation` step — and the auto-detection logic INSIDE that step reads from the
same `using_bag` instance BEFORE drain. This is correct behavior.

The pathmaker doesn't need to change the drain architecture — just read from `using_bag`
inside the auto-detection function, which already has access to the bag as it was set for
that step.
