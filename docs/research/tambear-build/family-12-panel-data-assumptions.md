# Family 12: Panel Data & Econometrics — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (FE/RE = GramMatrix after centering) + B (first-difference = Affine lag scan)

---

## Core Insight: Panel FE = Demeaned OLS = F06 + F10 Reuse

Fixed Effects panel regression is OLS on group-mean-centered data. The centering uses F06 per-group MomentStats. The regression uses F10 GramMatrix. ~120 new lines for Phase 1.

---

## 1. Panel Data Structure

Data: y_{it} for individual i = 1..N, time t = 1..T.

### Balanced vs Unbalanced
- **Balanced**: every unit observed at every time period. Matrix form: N × T.
- **Unbalanced**: some (i,t) missing. Requires careful indexing.

### Model
```
y_{it} = x'_{it}β + α_i + λ_t + ε_{it}
```
where α_i = individual effect, λ_t = time effect.

---

## 2. Fixed Effects (FE) Estimator — Within Estimator

### Demean (within transformation)
```
ỹ_{it} = y_{it} - ȳ_i    (subtract individual mean)
x̃_{it} = x_{it} - x̄_i
```

### Then OLS on demeaned data
```
β̂_FE = (X̃'X̃)⁻¹X̃'ỹ
```

### As GramMatrix
The demeaning is `RefCenteredStats` from F06 (per-group centering). Then X̃'X̃ and X̃'ỹ are GramMatrix subblocks from F10.

**Implementation**: F06 per-group means → subtract → F10 OLS on centered data. Nearly zero new code.

### Standard Errors
- **Clustered SE** (robust to heteroskedasticity + within-unit correlation):
```
V̂_cluster = (X̃'X̃)⁻¹ (Σ_i X̃'_i û_i û'_i X̃_i) (X̃'X̃)⁻¹
```
where û_i = vector of residuals for unit i.

**CRITICAL**: Always use clustered SE with panel FE. Non-clustered SE are wrong (understate uncertainty).

### Degrees of Freedom
df = NT - N - K (lose N df from N individual effects being absorbed). Adjust F-statistics accordingly.

### Edge Cases
- Time-invariant regressors (e.g., gender): eliminated by demeaning. Cannot estimate in FE.
- N large, T small: incidental parameters problem for nonlinear models (not for linear FE).
- Singleton groups (N_i = 1): demeaned data is zero → exclude.

---

## 3. Random Effects (RE) Estimator — GLS

### Model
```
y_{it} = x'_{it}β + α_i + ε_{it}
```
where α_i ~ N(0, σ²_α), ε_{it} ~ N(0, σ²_ε).

### GLS Estimator
```
β̂_RE = (X'Ω⁻¹X)⁻¹X'Ω⁻¹y
```
where Ω = block-diagonal with Ω_i = σ²_ε·I_T + σ²_α·ιι' (ι = ones vector).

### Quasi-Demeaning
Equivalent to OLS on quasi-demeaned data:
```
y_{it} - θ·ȳ_i    where θ = 1 - √(σ²_ε / (T·σ²_α + σ²_ε))
```
θ = 0: pooled OLS. θ = 1: fixed effects. RE = weighted average of between and within.

### THIS IS LME (F11)
RE panel model is exactly the random intercept LME:
- X = regressors
- Z = unit indicator (one column per individual)
- G = σ²_α · I_N
- R = σ²_ε · I_{NT}

**Implementation**: Call F11 Henderson solve. Zero new code.

### Variance Component Estimation
- **Swamy-Arora**: σ̂²_ε from FE residuals, σ̂²_α from between-estimator residuals minus σ̂²_ε/T
- **REML**: Via F11 REML machinery

---

## 4. First-Difference (FD) Estimator

### Transform
```
Δy_{it} = y_{it} - y_{i,t-1} = Δx'_{it}β + Δε_{it}
```

### As Affine Lag Scan (Kingdom B)
First-differencing = lag operation within each unit: Δy_{it} = y_{it} - L·y_{it}. This is a scan with operator state = previous value.

### Then OLS on differenced data.

### When FD = FE
For T = 2: FD and FE give identical estimates. For T > 2: differ. FE is more efficient under i.i.d. errors; FD is more efficient under random walk errors.

---

## 5. Two-Way Fixed Effects

### Model
```
y_{it} = x'_{it}β + α_i + λ_t + ε_{it}
```

### Implementation: Double-demean
```
ỹ_{it} = y_{it} - ȳ_i - ȳ_t + ȳ    (subtract individual mean, time mean, add back grand mean)
```
Two passes of RefCenteredStats (F06): once by individual, once by time.

### CRITICAL: The Goodman-Bacon Decomposition
For staggered treatment timing (DiD), two-way FE is a weighted average of 2×2 DiD estimates from all (treated/not-yet-treated, timing) pairs. Some weights can be NEGATIVE → estimator can be WRONG (sign reversal). This is the "bad controls" problem in modern DiD.

**Recommendation**: Use Callaway-Sant'Anna or Sun-Abraham instead of naive TWFE for staggered DiD.

---

## 6. Hausman Test

### Hypothesis
H₀: RE is consistent (α_i uncorrelated with x_{it})
H₁: RE is inconsistent → use FE

### Test Statistic
```
H = (β̂_FE - β̂_RE)'[V̂_FE - V̂_RE]⁻¹(β̂_FE - β̂_RE) ~ χ²(K)
```

### Implementation: F07 chi-square test. Needs both FE and RE estimates.

### CRITICAL: If V̂_FE - V̂_RE is not positive definite (can happen with robust SE), the test is undefined. Use Mundlak (1978) approach instead: RE model with group means as additional regressors, test significance of group mean coefficients.

---

## 7. Dynamic Panel Models

### Model
```
y_{it} = ρ·y_{i,t-1} + x'_{it}β + α_i + ε_{it}
```

### Nickell Bias
FE estimator of ρ is biased when T is small: E[ρ̂_FE - ρ] ≈ -(1+ρ)/(T-1). For T = 10, bias can be ~20%.

### Arellano-Bond (GMM)
Use lagged levels as instruments for first-differenced equation:
```
Δy_{it} = ρ·Δy_{i,t-1} + Δx'_{it}β + Δε_{it}
```
Instruments: y_{i,t-2}, y_{i,t-3}, ... (valid under no serial correlation in ε).

### Arellano-Bover/Blundell-Bond (System GMM)
Add equations in levels with lagged differences as instruments. More efficient than difference GMM, especially when ρ is close to 1.

### Two-Step GMM
```
β̂ = (X'ZWZ'X)⁻¹X'ZWZ'y
```
where Z = instrument matrix, W = optimal weight matrix (inverse of moment covariance).

**Windmeijer (2005) correction**: Small-sample correction for two-step GMM standard errors. Without it, SE are severely downward biased.

---

## 8. Instrumental Variables (IV) / 2SLS

### Model
Endogenous: y = Xβ + ε where E[x'ε] ≠ 0.
Instruments Z: E[z'ε] = 0 and E[z'x] ≠ 0 (relevance).

### 2SLS
Stage 1: X̂ = Z(Z'Z)⁻¹Z'X (project X onto Z)
Stage 2: β̂ = (X̂'X)⁻¹X̂'y

Equivalently:
```
β̂_2SLS = (X'P_Z X)⁻¹ X'P_Z y    where P_Z = Z(Z'Z)⁻¹Z'
```

### Diagnostics
- **Weak instruments**: First-stage F < 10 → weak IV bias (rule of thumb, Stock & Yogo 2005)
- **Overidentification**: Sargan/Hansen J-test when #instruments > #endogenous
- **Durbin-Wu-Hausman**: Test endogeneity — add first-stage residuals to structural equation

### Implementation: Two GramMatrix solves (one for first stage, one for second). F10 machinery.

---

## 9. Difference-in-Differences (DiD)

### Basic 2×2 DiD
```
ATT = (Ȳ_{treat,post} - Ȳ_{treat,pre}) - (Ȳ_{control,post} - Ȳ_{control,pre})
```
Equivalent regression: y_{it} = β₀ + β₁·Treat_i + β₂·Post_t + β₃·(Treat_i × Post_t) + ε_{it}. β₃ = ATT.

### Parallel Trends Assumption
E[Y(0)_{post} - Y(0)_{pre} | Treat=1] = E[Y(0)_{post} - Y(0)_{pre} | Treat=0]

**CRITICAL**: This is UNTESTABLE for the treatment period. Can only check pre-treatment trends.

### Event Study / Dynamic DiD
```
y_{it} = α_i + λ_t + Σ_{k≠-1} β_k · D_{it}^k + ε_{it}
```
where D_{it}^k = 1 if unit i is k periods from treatment. β_k for k < 0 should be ~0 (pre-trend test).

---

## 10. Additional Econometric Methods

### 10a. Heckman Selection Model
```
Selection:  z*_i = w'_iγ + u_i    (observe z_i = 1 if z*_i > 0)
Outcome:    y_i = x'_iβ + ε_i     (observe only when z_i = 1)
```
Corr(u, ε) = ρ ≠ 0 → selection bias.

Two-step: Probit on selection → compute IMR λ(w'_iγ̂) → include IMR in outcome regression.

### 10b. Quantile Regression (Panel)
min_β Σ ρ_τ(y_{it} - x'_{it}β - α_i) where ρ_τ(u) = u(τ - I(u<0)).

### 10c. Correlated Random Effects (Mundlak 1978)
```
y_{it} = x'_{it}β + x̄'_i·γ + α̃_i + ε_{it}
```
Add group means as regressors to RE model. If γ ≠ 0 → reject RE, coefficients on x_{it} match FE.

---

## Sharing Surface

### Reuse from Other Families
- **F06 (Descriptive)**: Per-group means (RefCenteredStats for FE demeaning)
- **F10 (Regression)**: OLS on demeaned data, 2SLS, regression diagnostics
- **F11 (Mixed Effects)**: RE panel = random intercept LME
- **F07 (Hypothesis Testing)**: Hausman test (χ²), Sargan/Hansen J-test
- **F05 (Optimization)**: GMM criterion function optimization
- **F17 (Time Series)**: First-difference = lag operator, AR dynamics

### Consumers of F12
- **Fintek**: Panel regressions across tickers (cross-sectional + time series)
- **F35 (Causal)**: DiD is a causal inference method; IV is instrument-based causal identification

### Structural Rhymes
- **FE = demeaned OLS**: same as F10 regression with indicator variables absorbed
- **RE = LME random intercept**: exactly F11 Henderson equations
- **FD = Affine lag scan**: same as F17 differencing
- **Hausman = F07 chi-square**: same test infrastructure
- **2SLS = two-stage GramMatrix solve**: same as F10 used twice
- **DiD = interaction regression**: same as F10 with structured design matrix

---

## Implementation Priority

**Phase 1** — Core panel models (~120 lines, as noted in task):
1. Fixed Effects (FE) via RefCenteredStats + OLS
2. Random Effects (RE) via F11 LME call
3. First-Difference (FD) via Affine lag scan + OLS
4. Two-Way FE (double-demeaning)
5. Hausman test (F07 χ² call)
6. Clustered standard errors

**Phase 2** — IV + GMM (~150 lines):
7. 2SLS (two GramMatrix solves)
8. Arellano-Bond (difference GMM)
9. Blundell-Bond (system GMM)
10. Weak instrument diagnostics (first-stage F)
11. Sargan/Hansen overidentification test

**Phase 3** — DiD + causal (~100 lines):
12. Basic 2×2 DiD
13. Event study / dynamic DiD
14. Callaway-Sant'Anna (heterogeneous treatment timing)
15. Parallel trends pre-test

**Phase 4** — Extensions (~100 lines):
16. Heckman selection correction
17. Mundlak (correlated random effects)
18. Panel quantile regression
19. Dynamic FE with Nickell bias correction

---

## Composability Contract

```toml
[family_12]
name = "Panel Data & Econometrics"
kingdom = "A (FE/RE demeaned OLS) + B (first-difference lag scan)"

[family_12.shared_primitives]
fe_demean = "RefCenteredStats (F06) per unit → demeaned OLS (F10)"
re_gls = "LME (F11) with random intercept"
first_diff = "Affine lag scan within units"
twfe = "Double-demeaning (two RefCenteredStats passes)"
iv_2sls = "Two-stage GramMatrix solve"

[family_12.reuses]
f06_descriptive = "Per-group means for demeaning"
f10_regression = "OLS, 2SLS, regression diagnostics"
f11_mixed = "RE = random intercept LME"
f07_hypothesis = "Hausman, Sargan, Hansen tests"
f05_optimizer = "GMM criterion optimization"
f17_time_series = "Lag operators, differencing"

[family_12.provides]
fe_estimates = "Fixed effects β̂ + clustered SE"
re_estimates = "Random effects β̂ + variance components"
hausman = "FE vs RE test"
did = "Difference-in-differences ATT"
iv = "Instrumental variables estimates"
gmm = "Arellano-Bond / Blundell-Bond"

[family_12.consumers]
fintek = "Cross-sectional panel regressions"
f35_causal = "DiD, IV for causal inference"
```
