# Family 35: Causal Inference — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (all methods reduce to regression/matching from existing families)

---

## Core Insight: No New Primitives

Every causal inference method in this family is a composition of existing primitives:
- 2SLS = two F10 OLS calls
- DiD = F12 panel FE + interaction
- RDD = F10 WLS on filtered subset
- PSM = F10 logistic + F01 KNN matching
- IPW = F10 logistic for propensity + weighted regression
- DoublyRobust = IPW + outcome model

~90 lines Phase 1. The math is in the IDENTIFICATION, not the computation.

---

## 1. Potential Outcomes Framework (Rubin Causal Model)

### Notation
- Y_i(1): potential outcome if treated
- Y_i(0): potential outcome if not treated
- T_i ∈ {0, 1}: treatment indicator
- Observed: Y_i = T_i·Y_i(1) + (1-T_i)·Y_i(0)

### Estimands
- **ATE** = E[Y(1) - Y(0)] (Average Treatment Effect)
- **ATT** = E[Y(1) - Y(0) | T=1] (Average Treatment Effect on Treated)
- **CATE** = E[Y(1) - Y(0) | X=x] (Conditional ATE)
- **LATE** = E[Y(1) - Y(0) | Compliers] (Local ATE, for IV)

### Fundamental Problem
Cannot observe both Y(1) and Y(0) for same unit. All methods make ASSUMPTIONS to bridge this gap.

---

## 2. Propensity Score Methods

### Propensity Score
```
e(x) = P(T=1 | X=x)
```
Estimated by logistic regression: e(x) = logistic(x'β). This is F10.

### 2a. Propensity Score Matching (PSM)
For each treated unit, find nearest control(s) by |e_i - e_j|.

**Implementation**: F10 logistic → propensity scores → F01 distance (1D) → nearest neighbor matching.

### Matching Methods
| Method | Description |
|--------|------------|
| 1:1 nearest | Each treated matched to closest control |
| 1:k nearest | k nearest controls |
| Caliper | Match only if |e_i - e_j| < δ |
| Mahalanobis | Match on X directly using Mahalanobis distance |
| Exact | Exact match on categorical covariates |
| CEM (Coarsened Exact) | Coarsen continuous X, then exact match |

### 2b. Inverse Probability Weighting (IPW)
```
ATE_IPW = (1/n) Σ [T_i·Y_i/e(X_i) - (1-T_i)·Y_i/(1-e(X_i))]
```

### CRITICAL: Extreme propensity scores (e near 0 or 1) cause huge weights
- **Trimming**: Exclude units with e < 0.05 or e > 0.95
- **Stabilized weights**: w_i = T_i/e(X_i) · P(T=1) + (1-T_i)/(1-e(X_i)) · P(T=0)

### 2c. Doubly Robust (AIPW)
```
ATE_DR = (1/n) Σ [μ̂₁(X_i) - μ̂₀(X_i) + T_i(Y_i - μ̂₁(X_i))/e(X_i) - (1-T_i)(Y_i - μ̂₀(X_i))/(1-e(X_i))]
```
where μ̂_t(x) = E[Y|X=x, T=t] (outcome model).

**Key property**: Consistent if EITHER propensity score model OR outcome model is correct (not both needed).

---

## 3. Instrumental Variables (IV)

### Already covered in F12 (Panel Data) assumption document.

Key additions for causal context:
- **LATE interpretation** (Imbens & Angrist 1994): IV estimates treatment effect for compliers only
- **Monotonicity assumption**: instrument affects treatment in one direction only
- **Exclusion restriction**: instrument affects outcome ONLY through treatment

---

## 4. Regression Discontinuity Design (RDD)

### Sharp RDD
Treatment deterministic at cutoff c:
```
T_i = I(X_i ≥ c)
```
```
τ_RDD = lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x]
```

### Implementation: Local polynomial regression (F10 WLS) on each side of cutoff
```
Left:  min_β Σ_{X_i<c} K((X_i-c)/h) · (Y_i - β₀ - β₁(X_i-c))²
Right: min_β Σ_{X_i≥c} K((X_i-c)/h) · (Y_i - β₀ - β₁(X_i-c))²
τ̂ = β̂₀_right - β̂₀_left
```

### Bandwidth Selection
- **MSE-optimal** (Imbens-Kalyanaraman): h_opt that minimizes MSE of τ̂
- **Coverage-error-rate optimal** (Calonico-Cattaneo-Titiunik): for confidence intervals

### Fuzzy RDD
Treatment probability jumps at cutoff (not deterministic):
```
τ_fuzzy = [lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x]] / [lim_{x↓c} E[T|X=x] - lim_{x↑c} E[T|X=x]]
```
This is Wald estimator = IV with instrument = I(X ≥ c).

### Diagnostics
- McCrary density test: no manipulation of running variable at cutoff
- Covariate balance at cutoff: pre-treatment covariates should be smooth at c
- Sensitivity to bandwidth choice: plot τ̂(h) for range of h

---

## 5. Difference-in-Differences (DiD)

### Already covered in F12. Additional causal details:

### Staggered Treatment Timing
- **Callaway-Sant'Anna (2021)**: Group-time ATT estimates, aggregated with proper weights
- **Sun-Abraham (2021)**: Interaction-weighted estimator
- **de Chaisemartin-D'Haultfoeuille (2020)**: Allows heterogeneous effects

### CRITICAL: Never use naive TWFE with staggered treatment. The Goodman-Bacon decomposition shows negative weights contaminate the estimate.

---

## 6. Synthetic Control Method

### Idea
Construct a synthetic counterfactual for the treated unit using a weighted combination of control units:
```
Ŷ_1(0)_t = Σ_{j=2}^{J+1} w_j · Y_{jt}    for post-treatment t
```
where weights w solve:
```
min_w Σ_{t∈pre} (Y_{1t} - Σ w_j Y_{jt})²
s.t.  w_j ≥ 0,  Σ w_j = 1
```

### Implementation: Constrained optimization (F05 projected gradient with simplex constraint).

### Inference: Permutation test — apply method to each control unit as "placebo treated" → distribution of effects under null.

---

## 7. Mediation Analysis

### Baron-Kenny Framework
```
Total effect:   Y = cT + ε₁
Path a:         M = aT + ε₂
Path b+c':      Y = c'T + bM + ε₃
```
Indirect effect = a·b = c - c'. Direct effect = c'.

### Sobel Test
```
z = a·b / √(b²·SE_a² + a²·SE_b²)
```

### Modern: Bootstrap the indirect effect (no normality assumption needed).

### CRITICAL: Baron-Kenny requires NO unmeasured confounders of M→Y relationship. This is a very strong assumption often violated.

---

## 8. Sensitivity Analysis

### Rosenbaum Bounds
For matched studies: how large must the hidden bias Γ be to change the conclusion?
```
1/Γ ≤ P(T=1|X)/P(T=0|X) / [P(T=1|X')/P(T=0|X')] ≤ Γ
```
Report the Γ at which p-value crosses 0.05.

### E-value (VanderWeele & Ding 2017)
Minimum strength of unmeasured confounding that could explain away the observed effect:
```
E-value = RR + √(RR·(RR-1))
```
where RR = observed risk ratio.

---

## Sharing Surface

### Reuse from Other Families
- **F10 (Regression)**: Logistic for propensity scores, WLS for RDD, outcome models
- **F01 (Distance)**: Nearest-neighbor matching (1D and multivariate)
- **F12 (Panel)**: DiD = panel FE + interaction, 2SLS = IV
- **F05 (Optimization)**: Constrained optimization for synthetic control
- **F07 (Hypothesis)**: Permutation tests for synthetic control inference
- **F08 (Nonparametric)**: Bootstrap for mediation, permutation for RDD

### Structural Rhymes
- **PSM = nearest-neighbor in propensity space**: same as F01 KNN + F20 matching
- **IPW = weighted accumulate**: same as F06 weighted statistics
- **RDD = local regression**: same as F10 kernel-weighted OLS
- **Doubly robust = insurance against model misspecification**: same philosophy as F09 robust statistics
- **Synthetic control = constrained weighted average**: simplex-projected regression

---

## Implementation Priority

**Phase 1** — Core methods (~90 lines, as noted in task):
1. Propensity score estimation (F10 logistic)
2. PSM (1:1, 1:k nearest, caliper)
3. IPW (with stabilized weights + trimming)
4. 2SLS (F12 call)
5. DiD (F12 call)
6. RDD (sharp, F10 WLS on each side)

**Phase 2** — Advanced (~100 lines):
7. Doubly robust (AIPW)
8. Fuzzy RDD
9. Synthetic control (constrained optimization)
10. Callaway-Sant'Anna (staggered DiD)

**Phase 3** — Diagnostics + extensions (~80 lines):
11. Balance diagnostics (standardized mean differences)
12. Overlap assessment (propensity score distribution)
13. McCrary density test (RDD)
14. Rosenbaum bounds + E-value
15. Mediation analysis (Baron-Kenny + bootstrap)

---

## Composability Contract

```toml
[family_35]
name = "Causal Inference"
kingdom = "A (all reduce to regression/matching compositions)"

[family_35.shared_primitives]
propensity_score = "F10 logistic regression"
matching = "F01 distance + nearest-neighbor"
ipw = "Weighted accumulate with propensity weights"
did = "F12 panel FE + interaction"
rdd = "F10 local WLS at cutoff"

[family_35.reuses]
f10_regression = "Logistic, WLS, outcome models"
f01_distance = "Matching (propensity and Mahalanobis)"
f12_panel = "DiD, 2SLS/IV"
f05_optimization = "Synthetic control (constrained)"
f07_hypothesis = "Permutation tests"
f08_nonparametric = "Bootstrap, permutation"

[family_35.provides]
ate = "Average treatment effect (IPW, DR, matching)"
att = "Average treatment effect on treated"
rdd_effect = "Regression discontinuity effect"
did_effect = "Difference-in-differences effect"
synthetic = "Synthetic control counterfactual"
```
