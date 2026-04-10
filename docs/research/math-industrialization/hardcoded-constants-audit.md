# Hardcoded Constants Audit — tambear/src/

**Date**: 2026-04-10  
**Scope**: All `.rs` files in `crates/tambear/src/`  
**Method**: Exhaustive grep + targeted reading  
**Status**: Read-only audit. No source files modified.

---

## How to Read This Audit

Each finding has:
- **File + line** (line numbers are approximate — verify before patching)
- **Value** and what it represents
- **Proposed parameter name** with `Option<f64>` type
- **Suggested default** (matching the current hardcoded value)
- **Status** — whether it is already parameterized

Findings are grouped by severity:

- **VIOLATION**: Truly hardcoded — no knob exists at any layer
- **PARTIAL**: Parameter exists at one layer but not another (e.g., the primitive takes it but the auto-detection layer hardcodes it)
- **COSMETIC**: Hardcoded in a way that is defensible (e.g., mathematical constant, not a tuning knob; or a test-only assertion)

---

## VIOLATIONS — Truly Hardcoded, No Knob

### V-01: `garch_is_valid` — vol clustering threshold 0.05

**File**: `data_quality.rs:388`
```rust
has_vol_clustering(returns, 0.05)
```
**What it is**: ACF(r²) threshold for declaring volatility clustering present. The raw `has_vol_clustering` takes a `threshold: f64` parameter, but `garch_is_valid` calls it with a hardcoded 0.05.  
**Proposed name**: `min_vol_clustering_acf: Option<f64>`  
**Default**: `0.05`  
**Status**: VIOLATION. `has_vol_clustering` is correctly parameterized; `garch_is_valid` wraps it without exposing the knob.

---

### V-02: `DataQualitySummary::from_slice` — three hardcoded thresholds

**File**: `data_quality.rs:623,627,628`
```rust
jump_ratio: jump_ratio_proxy(x, 3.0),
has_vol_clustering: has_vol_clustering(x, 0.1),
has_trend: has_trend(x, 0.5),
```
**What they are**:
- `3.0` — sigma-equivalent for jump detection (k-sigma rule)
- `0.1` — ACF threshold for volatility clustering
- `0.5` — R² threshold for trend detection

`from_slice` uses canonical values and cannot be overridden. There is no `from_slice_with_params` variant.  
**Proposed fix**: `from_slice_with_params(x, jump_k, vol_acf_threshold, trend_r2_threshold)` with `from_slice` delegating to it with the documented defaults.  
**Status**: VIOLATION. All three values embedded in the summary constructor.

---

### V-03: `tbs_executor` normality decision boundary — 5000

**File**: `tbs_executor.rs:452,593,749,1142`
```rust
if col.len() < 5000 { shapiro_wilk(...) } else { dagostino_pearson(...) }
```
**What it is**: The sample-size cutoff at which the executor switches from Shapiro-Wilk to D'Agostino-Pearson. This appears four times identically.  
**Proposed name**: `normality_test_n_threshold: Option<usize>`  
**Default**: `5000`  
**Status**: VIOLATION. Same hardcoded constant in four places in the auto-detection layer.

---

### V-04: `tbs_executor` — normality alpha 0.05 in four decision trees

**File**: `tbs_executor.rs:484,485,603,604,614,758,764,789,1148,1152`

The auto-detection layer (`correlation`, `t_test_2`, `anova`, `regression`) all hardcode the normality decision at `p > 0.05` and the variance homogeneity decision at `p > 0.05`. None of these are reachable via `using()`.

```rust
let x_norm = px > 0.05;
let y_norm = py > 0.05;
let equal_var = levene_p > 0.05;
let all_normal = norm_results.iter().all(|(p, _)| *p > 0.05);
let resid_normal = resid_norm_p > 0.05;
let heteroscedastic = bp.p_value < 0.05;
```

**Proposed name**: `normality_alpha: Option<f64>`, `variance_alpha: Option<f64>`  
**Default**: `0.05` for both  
**Status**: VIOLATION. These alpha values percolate into method-selection decisions in the auto-detection pipeline, but the `using()` bag is not queried for them.

---

### V-05: `tbs_executor` — VIF threshold 10.0 for multicollinearity

**File**: `tbs_executor.rs:1139`
```rust
let multicollinear = max_vif > 10.0;
```
**What it is**: VIF threshold above which Ridge is recommended. The statsmodels/R convention is VIF > 10 for "severe", VIF > 5 for "moderate". This is a researcher-adjustable knob.  
**Proposed name**: `vif_threshold: Option<f64>`  
**Default**: `10.0`  
**Status**: VIOLATION. Not reachable via `using()`.

---

### V-06: `tbs_executor` — KMO threshold 0.5 for PCA viability

**File**: `tbs_executor.rs:1233,1249`
```rust
let pca_viable = kmo >= 0.5 && bartlett_p < 0.05;
if kmo < 0.5 { ...
```
**What it is**: Kaiser-Meyer-Olkin minimum adequate sampling threshold. The 0.5 value ("miserable but acceptable") is a convention, not a law. Researchers may want to warn at 0.6 ("mediocre") or allow 0.4 in exploratory work.  
**Proposed name**: `kmo_threshold: Option<f64>`, plus `bartlett_alpha: Option<f64>` for the 0.05  
**Default**: `0.5`, `0.05`  
**Status**: VIOLATION.

---

### V-07: `tbs_executor` — Hopkins threshold 0.5 for cluster structure

**File**: `tbs_executor.rs:1009`
```rust
let has_structure = hopkins > 0.5; // Hopkins > 0.5 suggests clusters
```
**What it is**: Hopkins statistic boundary for declaring the data has cluster structure. Convention is 0.5, but this is tunable.  
**Proposed name**: `hopkins_threshold: Option<f64>`  
**Default**: `0.5`  
**Status**: VIOLATION.

---

### V-08: `tbs_executor` — scale ratio 10.0 for normalization lint

**File**: `tbs_executor.rs:999`
```rust
if scale_ratio > 10.0 { lints.push(...) }
```
**What it is**: Column range ratio above which L302 lint fires ("clustering may be dominated by large-scale features"). Tunable — some domains expect wide dynamic range.  
**Proposed name**: `scale_ratio_warn_threshold: Option<f64>`  
**Default**: `10.0`  
**Status**: VIOLATION.

---

### V-09: `tbs_executor` — ARCH-LM alpha 0.05 in two auto-detect paths

**File**: `tbs_executor.rs:1450,1901`
```rust
let has_arch = arch_p < 0.05;
```
Both `time_series_analyze` and `vol_analyze` use this. Neither reads from `using()`.  
**Proposed name**: `arch_alpha: Option<f64>`  
**Default**: `0.05`  
**Status**: VIOLATION.

---

### V-10: `tbs_executor` — ARCH max_lags formula `5.min(n/10)`

**File**: `tbs_executor.rs:1447,1898`
```rust
let arch_lags = 5.min(col.len() / 10).max(1);
```
Both the `time_series_analyze` and `vol_analyze` paths embed the max-lag formula for ARCH-LM. The `5` (max) and `10` (denominator) are tunable.  
**Proposed names**: `arch_max_lags: Option<usize>` (override the formula entirely), or expose `arch_lags_cap: Option<usize>` and `arch_lags_fraction_denom: Option<usize>`  
**Default**: cap=5, denom=10  
**Status**: VIOLATION.

---

### V-11: `tbs_executor` — EWMA lambda 0.94 in `vol_analyze` auto-path

**File**: `tbs_executor.rs:1926`
```rust
let values = crate::volatility::ewma_variance(&col, 0.94);
```
The explicit `ewma_var` step (line 1868) correctly passes `lambda` as a `f64_arg`. But the auto-path in `vol_analyze` hardcodes it.  
**Proposed name**: `ewma_lambda: Option<f64>`  
**Default**: `0.94` (RiskMetrics daily)  
**Status**: PARTIAL. The primitive and the named step are parameterized; the auto-detect path is not.

---

### V-12: `tbs_executor` — max_lag formula `n/4` capped at 20 in auto-path

**File**: `tbs_executor.rs:1442`
```rust
let max_lag = (col.len() / 4).min(20).max(1);
```
The named `acf`/`pacf` steps take `max_lag` as an explicit argument (lines 1369, 1377). The `time_series_analyze` auto-path hardcodes `n/4` and 20.  
**Proposed name**: `max_lag: Option<usize>`  
**Default**: `min(n/4, 20)`  
**Status**: PARTIAL.

---

### V-13: `tbs_executor` — GARCH max_iter 200 in auto-path

**File**: `tbs_executor.rs:1905`
```rust
let garch = crate::volatility::garch11_fit(&col, 200);
```
The named `garch` step (line 1323) uses `usize_arg(step, "max_iter", 1, 200)`. The `vol_analyze` auto-path hardcodes 200.  
**Proposed name**: `garch_max_iter: Option<usize>`  
**Default**: `200`  
**Status**: PARTIAL.

---

### V-14: `superposition.rs` — alpha 0.05 in `sweep_two_sample_tests`

**File**: `superposition.rs:244`
```rust
pub fn sweep_two_sample_tests(x: &[f64], y: &[f64]) -> Superposition {
    sweep_two_sample_tests_alpha(x, y, 0.05)
}
```
**Status**: PARTIAL. `sweep_two_sample_tests_alpha` correctly takes an alpha parameter. `sweep_two_sample_tests` (the convenience wrapper) hardcodes 0.05 — documented in the doc comment. This is the cleanest case in the codebase: two functions, explicit separation. No change needed unless `using()` integration is desired.

---

### V-15: `scoring.rs` — `threshold_adaptive` fallback 0.5 and floor 0.3

**File**: `scoring.rs:42,54`
```rust
if drops.is_empty() { return 0.5; }
...
(median + k * var.sqrt()).max(0.3)
```
**What they are**: Default threshold when no data is available (0.5) and the minimum threshold floor (0.3) in the adaptive changepoint scorer.  
**Proposed names**: `empty_fallback_threshold: Option<f64>`, `min_adaptive_threshold: Option<f64>`  
**Default**: `0.5`, `0.3`  
**Status**: VIOLATION. Neither is reachable via `using()`.

---

### V-16: `irt.rs` — Newton-Raphson inner loop count hardcoded at 10

**File**: `irt.rs:69,101`
```rust
for _ in 0..10 {  // person Newton-Raphson inner steps
for _ in 0..10 {  // item Newton-Raphson inner steps
```
**What it is**: The number of Newton-Raphson inner steps inside `fit_2pl`. The outer iteration count is already a parameter (`max_iter`), but the inner loop is not.  
**Proposed name**: `nr_inner_iter: Option<usize>`  
**Default**: `10`  
**Status**: VIOLATION.

---

### V-17: `irt.rs` — discrimination and difficulty clamp bounds

**File**: `irt.rs:85,118,119,164`
```rust
theta = theta.clamp(-6.0, 6.0);   // ability bounds
a = a.clamp(0.1, 5.0);            // discrimination bounds
b = b.clamp(-5.0, 5.0);           // difficulty bounds
```
**What they are**: IRT parameter feasibility constraints. For most operational testing these are fine, but specialized CAT systems or adaptive testing platforms may need wider/narrower ranges.  
**Proposed names**: `theta_min: Option<f64>`, `theta_max: Option<f64>`, `disc_min: Option<f64>`, `disc_max: Option<f64>`, `diff_min: Option<f64>`, `diff_max: Option<f64>`  
**Default**: `-6.0 / 6.0`, `0.1 / 5.0`, `-5.0 / 5.0`  
**Status**: VIOLATION.

---

### V-18: `irt.rs` — initial difficulty clamp 0.01/0.99

**File**: `irt.rs:60`
```rust
difficulty: -logit(p_correct.clamp(0.01, 0.99)),
```
**What it is**: Clamp on the empirical proportion correct before taking the logit. Prevents ±∞ starting values for perfect items. This 0.01/0.99 is the boundary — sometimes users set it to 0.001/0.999 for large-n datasets.  
**Proposed name**: `p_correct_eps: Option<f64>`  
**Default**: `0.01`  
**Status**: VIOLATION.

---

### V-19: `linear_algebra.rs` — `pinv` SVD threshold 1e-12

**File**: `linear_algebra.rs:650`
```rust
if svd_res.sigma[i] > 1e-12 {
    sigma_inv[i] = 1.0 / svd_res.sigma[i];
}
```
**What it is**: Numerical threshold below which singular values are treated as zero in the pseudoinverse. This 1e-12 is a reasonable numerical default but researchers working with ill-conditioned systems may want to set a relative threshold or a different absolute cutoff.  
**Proposed name**: `rcond: Option<f64>`  
**Default**: `1e-12` (absolute) — alternatively `max_sigma * machine_eps` (relative)  
**Status**: VIOLATION. The `rank(a, tol)` function correctly takes a tolerance parameter, but `pinv` does not expose one.

---

### V-20: `data_quality.rs:514` — ACF lag cap formula `min(n/4, 20)`

**File**: `data_quality.rs:514`
```rust
let max_lag = (n / 4).min(20).max(2);
```
Used in `garch_is_valid` via `has_vol_clustering`, and in the GARCH-validity check path.  
**Status**: PARTIAL. The formula is documented in a comment. Not exposed as a parameter.

---

### V-21: `time_series.rs:1545` — ADF trim fraction 0.15

**File**: `time_series.rs:1545`
```rust
let t = trim.unwrap_or(0.15);
```
**Status**: CORRECTLY PARAMETERIZED. `trim` is an `Option<f64>` parameter. Default is documented. No violation.

---

## PARTIAL — Parameterized at Primitive Level, Hardcoded at Layer Above

### P-01: `hypothesis.rs` — `cooks_distance` uses `None` in the primary entry point

**File**: `hypothesis.rs:1036`
```rust
pub fn cooks_distance(...) -> InfluenceResult {
    cooks_distance_with_threshold(x_with_intercept, residuals, None)
}
```
The `with_threshold` variant correctly takes `influence_threshold: Option<f64>`. But the `tbs_executor` always calls `cooks_distance` (the no-threshold variant), meaning the 4/n default is fixed in the pipeline.  
**Status**: PARTIAL. The primitive is correctly split. The executor should accept `influence_threshold` from `using()`.

---

### P-02: `hypothesis.rs` — confidence level 0.95 hardcoded in TestResult constructors

**File**: `hypothesis.rs:199,241,290,667,705`
```rust
ci_level: 0.95,
```
The `ci_level` field in `TestResult` carries the confidence level. But multiple internal constructors hardcode 0.95. The `nonparametric.rs:353` CI function correctly uses `alpha.unwrap_or(0.05)`.  
**Proposed fix**: These should derive `ci_level` from the actual alpha used, not hardcode 0.95.  
**Status**: PARTIAL.

---

### P-03: `nonparametric.rs:353` — alpha for Hodges-Lehmann CI

**File**: `nonparametric.rs:353`
```rust
let alpha = alpha.unwrap_or(0.05);
```
**Status**: CORRECTLY PARAMETERIZED via `Option<f64>`. No violation.

---

### P-04: `hypothesis.rs:899,2014` — alpha in Mann-Whitney and Breusch-Pagan

**File**: `hypothesis.rs:899`, `hypothesis.rs:2014`
```rust
let alpha = alpha.unwrap_or(0.05);
```
**Status**: CORRECTLY PARAMETERIZED via `Option<f64>`. No violation.

---

## MATHEMATICAL CONSTANTS — Not Violations

These look like hardcoded numbers but are mathematical constants defining the specific estimator. Changing them would produce a different estimator, not a tuned version of the same one. They are **not** violations.

| Value | Location | What it is |
|-------|----------|------------|
| `1.345` | `robust.rs:25,40,108,584,593,616,623` | Huber k for 95% efficiency at the normal — **parameter, fully exposed** via function arg |
| `4.685` | `robust.rs:26,49,116,252,257,268,602` | Bisquare k for 95% efficiency at the normal — **fully exposed** |
| `1.4826` | `data_quality_catalog.rs:197,276`, `robust.rs:149,150,257` | MAD consistency factor — defines the scaled MAD estimator |
| `0.9` in Silverman | `nonparametric.rs:1168` | Coefficient in Silverman's rule `h = 0.9 · min(σ, IQR/1.34) · n^{-1/5}` — defines the specific rule |
| `1.34` in Silverman | `nonparametric.rs:1166` | IQR divisor in Silverman's rule |
| `1.06` in Scott | `nonparametric.rs:1176` | Coefficient in Scott's rule |
| `0.99` in volatility near-IGARCH | `volatility.rs:115,380,520` | `α+β > 0.99` stationarity boundary — these are definitional flags |

**Note on Huber/bisquare**: The function signatures for `huber_m_estimate` and `bisquare_m_estimate` already take `k` as an explicit parameter. The 1.345 and 4.685 only appear as documented defaults in tests and the TBS executor's `f64_arg` defaults. This is correct practice.

---

## SUMMARY TABLE

| ID | File | Line | Value | Represents | Param Name | Status |
|----|------|------|-------|------------|------------|--------|
| V-01 | `data_quality.rs` | 388 | `0.05` | vol clustering ACF threshold in `garch_is_valid` | `min_vol_clustering_acf` | VIOLATION |
| V-02a | `data_quality.rs` | 623 | `3.0` | jump ratio sigma multiplier in `from_slice` | `jump_k` | VIOLATION |
| V-02b | `data_quality.rs` | 627 | `0.1` | vol clustering ACF in `from_slice` | `vol_clustering_acf` | VIOLATION |
| V-02c | `data_quality.rs` | 628 | `0.5` | trend R² threshold in `from_slice` | `trend_r2_threshold` | VIOLATION |
| V-03 | `tbs_executor.rs` | 452,593,749,1142 | `5000` | SW→D'A-P normality test crossover | `normality_test_n_threshold` | VIOLATION |
| V-04 | `tbs_executor.rs` | 484,603,614,758,764,789,1148,1152 | `0.05` | normality/homoscedasticity decision alpha | `normality_alpha`, `variance_alpha` | VIOLATION |
| V-05 | `tbs_executor.rs` | 1139 | `10.0` | VIF multicollinearity threshold | `vif_threshold` | VIOLATION |
| V-06a | `tbs_executor.rs` | 1233 | `0.5` | KMO adequacy threshold | `kmo_threshold` | VIOLATION |
| V-06b | `tbs_executor.rs` | 1233 | `0.05` | Bartlett test alpha for PCA viability | `bartlett_alpha` | VIOLATION |
| V-07 | `tbs_executor.rs` | 1009 | `0.5` | Hopkins cluster structure threshold | `hopkins_threshold` | VIOLATION |
| V-08 | `tbs_executor.rs` | 999 | `10.0` | scale ratio lint threshold | `scale_ratio_warn_threshold` | VIOLATION |
| V-09 | `tbs_executor.rs` | 1450,1901 | `0.05` | ARCH-LM alpha in auto-detect paths | `arch_alpha` | VIOLATION |
| V-10 | `tbs_executor.rs` | 1447,1898 | `5`, `10` | ARCH lag cap and fraction denominator | `arch_max_lags` | VIOLATION |
| V-11 | `tbs_executor.rs` | 1926 | `0.94` | EWMA lambda in `vol_analyze` auto-path | `ewma_lambda` | PARTIAL |
| V-12 | `tbs_executor.rs` | 1442 | `n/4`, `20` | ACF max-lag formula in `time_series_analyze` | `max_lag` | PARTIAL |
| V-13 | `tbs_executor.rs` | 1905 | `200` | GARCH max_iter in `vol_analyze` | `garch_max_iter` | PARTIAL |
| V-14 | `superposition.rs` | 244 | `0.05` | sweep alpha in convenience wrapper | — | PARTIAL (clean) |
| V-15a | `scoring.rs` | 42 | `0.5` | empty-data fallback threshold | `empty_fallback_threshold` | VIOLATION |
| V-15b | `scoring.rs` | 54 | `0.3` | adaptive threshold floor | `min_adaptive_threshold` | VIOLATION |
| V-16 | `irt.rs` | 69,101 | `10` | Newton-Raphson inner steps in `fit_2pl` | `nr_inner_iter` | VIOLATION |
| V-17a | `irt.rs` | 85,164 | `-6.0/6.0` | ability theta clamp bounds | `theta_min`, `theta_max` | VIOLATION |
| V-17b | `irt.rs` | 118 | `0.1/5.0` | discrimination clamp bounds | `disc_min`, `disc_max` | VIOLATION |
| V-17c | `irt.rs` | 119 | `-5.0/5.0` | difficulty clamp bounds | `diff_min`, `diff_max` | VIOLATION |
| V-18 | `irt.rs` | 60 | `0.01/0.99` | p_correct clamp before logit | `p_correct_eps` | VIOLATION |
| V-19 | `linear_algebra.rs` | 650 | `1e-12` | pseudoinverse SVD threshold in `pinv` | `rcond` | VIOLATION |
| V-20 | `data_quality.rs` | 514 | `n/4`, `20` | ACF lag cap in GARCH validity path | `max_lag` | PARTIAL |
| P-01 | `tbs_executor.rs` | 1155 | `4/n` | Cook's D threshold in executor | `influence_threshold` | PARTIAL |
| P-02 | `hypothesis.rs` | 199,241,290,667,705 | `0.95` | ci_level hardcoded in constructors | — | PARTIAL |

---

## Priority Recommendations

**High priority** (affect auto-detection decisions silently, user has no recourse):

1. **V-04** — The normality and variance-homogeneity alpha values inside the `tbs_executor` auto-detection trees. A researcher who prefers α=0.01 for their normality check has no way to express this. These are the most impactful violations because they silently determine which statistical test gets run.

2. **V-05** — VIF threshold. A commonly debated parameter (some use 5, some 10, some 7.5). Easy fix: read `using_bag.get_f64("vif_threshold").unwrap_or(10.0)`.

3. **V-06** — KMO threshold and Bartlett alpha. PCA viability gating is a methodological judgment; 0.5 is a floor, not a law.

4. **V-09** — ARCH-LM alpha. Auto-detection of ARCH effects drives GARCH model selection; this alpha should flow from `using()`.

**Medium priority** (partial parameterizations — the primitive is right but the auto-path isn't):

5. **V-11, V-12, V-13** — EWMA lambda, ACF max_lag, GARCH max_iter in auto-paths. These are already correct in the named steps. Auto-paths should read from `using()` with the same defaults.

6. **P-01** — Cook's D threshold. The primitive is correctly split into `cooks_distance` / `cooks_distance_with_threshold`. The executor should pass the `using()` value.

**Lower priority** (uncommon use cases or defensive bounds):

7. **V-16/V-17/V-18** — IRT inner loop counts and clamp bounds. Relevant only for CAT/adaptive testing research.

8. **V-19** — `pinv` SVD threshold. Typically set relative to the largest singular value (MATLAB uses `max(m,n)*eps*max(svd)`). Worth a `rcond: Option<f64>` parameter.

9. **V-01/V-02** — `DataQualitySummary::from_slice` canonical thresholds. These are documented as "canonical" — the fix is adding a parameterized variant, not removing the defaults.

---

## False Positives (Not Violations)

The following `0.05` occurrences are **not violations** — they are test assertions checking that a computed p-value is above or below a threshold, or they are documented mathematical constants:

- `hypothesis.rs:2473,2678,2810,2900,2905,2962,2972` — test assertions in `#[test]` blocks
- `nonparametric.rs:2256,2324,2342,2367,2388,2396,2448,2551,2560,2650,2683,2694,2737,2860,2870,2908` — test assertions
- `multivariate.rs:1047,1071,1115` — test assertions
- `panel.rs:762,778,884` — test assertions
- `stochastic.rs:657,666` — interest rate parameter `r=0.05` in Black-Scholes test data (correct)
- `signal_processing.rs:2546` — EMD sift_threshold default, passed explicitly to `emd()` — **correctly parameterized**
- `series_accel.rs:843,851` — `0.95` as a ratio threshold for convergence type detection (mathematical characterization, not a user knob)
- `volatility.rs:30,115,380,422,520` — `0.99` as the near-IGARCH boundary: this defines what "near-integrated" means mathematically
- `special_functions.rs:1196,1198,1200,1202` — test assertions verifying distribution quantiles

The `1.4826` MAD consistency factor also appears in several places — this is a mathematical constant (1/qnorm(0.75) under the normal), not a tunable parameter.
