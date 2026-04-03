# Family 13: Survival Analysis — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — A (Kaplan-Meier, Nelson-Aalen after sort), C (Cox PH partial likelihood optimization)

---

## Core Insight: Censoring Is the Distinguishing Feature

Survival analysis is regression/statistics with INCOMPLETE observations. Some subjects haven't experienced the event yet (right-censored). The entire family exists because standard methods (mean, regression) fail when observations are censored — they systematically underestimate durations.

Every method in this family handles censoring through a common mechanism:
1. Sort by time (transform)
2. At each event time, compute a risk-set-based quantity (who was still "alive"?)
3. Accumulate these quantities (prefix product for KM, prefix sum for Nelson-Aalen, weighted scatter for Cox)

**Structural rhyme**: Kaplan-Meier survival = prefix product of conditional survival probabilities. Same `accumulate(Prefix(forward), ...)` as F17 time series, different operator (multiply instead of Affine).

---

## 1. Survival Function Basics

### Definitions
- **T**: random variable = time to event
- **S(t) = P(T > t)**: survival function (probability of surviving past time t)
- **f(t)**: density function
- **h(t) = f(t)/S(t)**: hazard function (instantaneous rate of event)
- **H(t) = -log(S(t)) = ∫₀ᵗ h(u)du**: cumulative hazard
- **Relationship**: S(t) = exp(-H(t))

### Censoring Types

| Type | What's observed | Example |
|------|----------------|---------|
| Right censoring | T_i > C_i (event not yet occurred) | Patient alive at end of study |
| Left censoring | T_i < C_i (event already occurred) | Disease onset before first test |
| Interval censoring | L_i < T_i < R_i | Event between two visits |

**Most common**: right censoring. All methods below handle right censoring. Interval censoring requires modified likelihood.

### Data Format
Each observation: (time_i, event_i) where event_i = 1 if observed, 0 if censored.

---

## 2. Kaplan-Meier Estimator

### Formula
At each distinct event time t_j:
```
Ŝ(t) = Π_{t_j ≤ t} (1 - d_j/n_j)
```
where d_j = number of events at t_j, n_j = number at risk just before t_j.

### GPU Decomposition
1. **Sort** by time (transform)
2. **Risk set counts**: n_j = n - (cumulative events + censored before t_j). This is `n - accumulate(Prefix(forward), (d_j + c_j), Add)`.
3. **Survival increments**: 1 - d_j/n_j (elementwise, parallel)
4. **Cumulative product**: `accumulate(Prefix(forward), (1 - d_j/n_j), Multiply)`

The prefix product is the key operation. Can also be computed as exp of prefix sum of log terms:
```
Ŝ(t) = exp(accumulate(Prefix(forward), log(1 - d_j/n_j), Add))
```
This converts prefix product to prefix sum — numerically stabler for small survival decrements.

### Greenwood's Variance
```
Var(Ŝ(t)) = Ŝ(t)² · Σ_{t_j ≤ t} d_j / (n_j · (n_j - d_j))
```

**Confidence intervals**: log-log transform recommended (keeps CI in [0,1]):
```
CI: Ŝ(t)^exp(±z_{α/2} · √(Var(log(-log(Ŝ(t))))))
```

### Median Survival
Smallest t where Ŝ(t) ≤ 0.5. If Ŝ never reaches 0.5: median undefined (report this).

---

## 3. Nelson-Aalen Estimator

### Cumulative Hazard
```
Ĥ(t) = Σ_{t_j ≤ t} d_j/n_j
```

**GPU**: `accumulate(Prefix(forward), d_j/n_j, Add)` — prefix sum.

### Relationship to Kaplan-Meier
```
Ŝ_NA(t) = exp(-Ĥ(t))
```

For large n_j: KM and NA are nearly identical. For small n_j: KM is preferred (guaranteed ≤ 1).

---

## 4. Log-Rank Test

### Comparing Two Survival Curves
At each event time t_j:
```
Expected events in group 1: E_{1j} = n_{1j} · d_j / n_j
```

Test statistic:
```
χ² = (Σ(O_{1j} - E_{1j}))² / Σ Var_j
```

where Var_j = n_{1j}·n_{2j}·d_j·(n_j-d_j) / (n_j²·(n_j-1)).

Under H₀: χ² ~ χ²(1).

### Stratified Log-Rank
When confounders exist: compute O-E within strata, sum across.

### GPU decomposition
- Sort by time
- Risk set per group: parallel prefix sums
- O-E per event time: elementwise (parallel)
- Sum: reduce

---

## 5. Cox Proportional Hazards

### Model
```
h(t|X) = h₀(t) · exp(X'β)
```

Hazard is a product of baseline hazard h₀(t) (unspecified) and relative risk exp(X'β).

### Partial Likelihood
```
L(β) = Π_{i: event_i=1} [exp(X_i'β) / Σ_{j∈R(t_i)} exp(X_j'β)]
```

where R(t_i) = risk set at time t_i (all j with time_j ≥ time_i).

### Log-Partial Likelihood
```
ℓ(β) = Σ_{i: event_i=1} [X_i'β - log(Σ_{j∈R(t_i)} exp(X_j'β))]
```

### Score and Hessian
```
U(β) = Σ_{i: event} [X_i - X̄_risk(t_i)]

H(β) = -Σ_{i: event} [Var_risk(X, t_i)]
```

where X̄_risk and Var_risk are weighted mean/variance of covariates over risk set with weights exp(X'β).

### Newton-Raphson Estimation
```
β_{k+1} = β_k - H(β_k)⁻¹ · U(β_k)
```

Kingdom C: iterative. Each iteration needs risk-set computations (Kingdom A after sort).

### CRITICAL: Risk Set Computation
The denominator Σ_{j∈R(t_i)} exp(X_j'β) changes at each event time. **Efficient**: compute as a SUFFIX sum (accumulate in reverse):

```
risk_sum(t_i) = accumulate(Prefix(reverse), exp(X_j'β), Add)
```

Sort by time descending → prefix sum → each position has sum of exp(Xβ) for all later times.

### Breslow Approximation for Ties
When multiple events at same time:
```
ℓ_Breslow = Σ_{i: event} [X_i'β - d_k · log(Σ_{j∈R(t_k)} exp(X_j'β))]
```

### Efron Approximation (better for ties)
```
ℓ_Efron = Σ_k Σ_{l=1}^{d_k} [X_{kl}'β - log(Σ_{j∈R(t_k)} exp(X_j'β) - (l-1)/d_k · Σ_{j∈D(t_k)} exp(X_j'β))]
```

### Proportional Hazards Assumption
Test: Schoenfeld residuals should have zero slope over time. Test each covariate:
```
cor(Schoenfeld_residual_j, rank(time)) ≈ 0
```

If violated: use time-varying coefficients or stratified Cox.

### GPU decomposition
- Sort by time: transform
- exp(Xβ): elementwise parallel
- Risk set sums: `accumulate(Prefix(reverse), exp_xb, Add)` — suffix sum
- Score/Hessian: weighted scatter over risk sets
- Newton step: F02 linear solve

---

## 6. Parametric Survival Models

### Common Distributions

| Distribution | S(t) | h(t) | Parameters |
|-------------|------|------|-----------|
| Exponential | exp(-λt) | λ (constant) | λ > 0 |
| Weibull | exp(-(t/λ)^k) | (k/λ)(t/λ)^(k-1) | λ > 0, k > 0 |
| Log-normal | 1-Φ((log(t)-μ)/σ) | f(t)/S(t) | μ, σ > 0 |
| Log-logistic | 1/(1+(t/α)^β) | (β/α)(t/α)^(β-1)/(1+(t/α)^β) | α > 0, β > 0 |
| Gompertz | exp(-(b/a)(exp(at)-1)) | b·exp(at) | a, b > 0 |
| Generalized Gamma | complex | complex | μ, σ, Q |

### Accelerated Failure Time (AFT) Model
```
log(T) = X'β + σ·W
```
where W has a specified distribution (extreme value → Weibull, normal → log-normal, logistic → log-logistic).

### MLE
```
ℓ(θ) = Σ_{i: event} log(f(t_i|θ)) + Σ_{i: censored} log(S(t_i|θ))
```

Optimization: F05 (Newton or L-BFGS). Each evaluation is parallel across observations.

---

## 7. Competing Risks

### Problem
Multiple event types (e.g., death from cancer vs death from other causes). Standard KM treats competing events as censoring — THIS IS WRONG (biased).

### Cumulative Incidence Function (CIF)
```
CIF_k(t) = Σ_{t_j ≤ t} Ŝ(t_j⁻) · (d_{kj}/n_j)
```

### Fine-Gray Model (subdistribution hazard)
```
h_k(t|X) = h_{k0}(t) · exp(X'β_k)
```

Modification: subjects who experience competing event REMAIN in risk set (with decreasing weights). This is a weighted Cox model.

### GPU decomposition
- Same as Cox with modified risk set weights
- Multiple CIFs computed in parallel (one per event type)

---

## 8. Concordance Index (C-statistic)

### Definition
```
C = P(risk_i > risk_j | T_i < T_j)
```

Probability that the model correctly orders two subjects. C = 0.5 is random, C = 1.0 is perfect.

### Computation
For all concordant pairs (i,j) where T_i < T_j and both uncensored (or i uncensored):
```
C = (concordant pairs) / (concordant + discordant pairs)
```

O(n²) naive. O(n log n) with merge-sort-based counting.

### GPU: parallel pairwise comparison (Tiled accumulate over subject pairs).

---

## 9. Numerical Stability

### Kaplan-Meier
- **Tiny survival decrements**: log-transform for prefix sum instead of direct product. log(1 - d/n) ≈ -d/n for small d/n.
- **Risk set = 0**: can happen after heavy censoring. S(t) is undefined beyond last observation. Return NaN or carry forward last estimate.

### Cox PH
- **exp(Xβ) overflow**: center covariates (subtract mean). This keeps Xβ near 0.
- **Risk set sum underflow**: compute in log-space: log(Σ exp(Xβ)) via log-sum-exp trick.
- **Singular Hessian**: multicollinearity in covariates. Use ridge penalty or remove covariates.
- **Large β estimates**: indicates separation (a covariate perfectly predicts event). Firth's penalized likelihood.

### General
- **Tied event times**: use Efron approximation (not Breslow) for accuracy.
- **Log-likelihood in log-space**: always sum log terms, never multiply probabilities.

---

## 10. Edge Cases

| Algorithm | Edge Case | Expected |
|-----------|----------|----------|
| KM | All observations censored | S(t) = 1 everywhere (no events). Valid but uninformative. |
| KM | All events at same time | Single step function. S drops to (n-d)/n. |
| KM | No censoring | KM = empirical CDF complement. |
| KM | n_j = 0 (risk set empty) | S undefined beyond this point. |
| Log-rank | One group has no events | Test statistic = 0. No evidence of difference. |
| Cox | Perfect prediction (separation) | β → ±∞. Use Firth penalization. |
| Cox | Time-varying covariates | Extended Cox with counting process format. |
| Cox | n_events < 10 per covariate | Overfitting. Reduce covariates. |
| Parametric | Non-positive times | Log-normal/log-logistic require T > 0. Error. |
| C-index | No concordant pairs | Undefined. Return NaN. |

---

## Sharing Surface

### Reuses from Other Families
- **F02 (Linear Algebra)**: Hessian solve for Cox Newton-Raphson
- **F05 (Optimization)**: MLE for parametric models, Cox estimation
- **F06 (Descriptive)**: risk set counts, prefix sums
- **F07 (Hypothesis)**: χ² distribution for log-rank, Wald tests
- **F08 (Nonparametric)**: bootstrap for survival CIs, permutation for log-rank
- **F10 (Regression)**: Cox is essentially weighted regression per IRLS template
- **F11 (Mixed Effects)**: frailty models = Cox with random effects

### Provides to Other Families
- **F34 (Bayesian)**: Bayesian survival models (prior on baseline hazard)
- **F12 (Panel)**: duration models, event history analysis
- **F18 (Volatility)**: hazard-based models for financial event timing

### Structural Rhymes
- **KM = prefix product of survival probabilities**: same as F06 cumulative product
- **Cox risk set = suffix sum of exp(Xβ)**: `accumulate(Prefix(reverse), exp_xb, Add)`
- **Cox IRLS = weighted least squares iteration**: same template as F10 logistic, F15 IRT
- **AFT = log-linear regression with censored observations**: F10 regression with modified likelihood

---

## Implementation Priority

**Phase 1** — Core nonparametric (~100 lines):
1. Kaplan-Meier estimator (with Greenwood variance, log-log CI)
2. Nelson-Aalen cumulative hazard
3. Log-rank test (unstratified and stratified)
4. Median survival, restricted mean survival time

**Phase 2** — Cox PH (~150 lines):
5. Cox PH estimation (Newton-Raphson, Efron ties)
6. Risk set computation via suffix sum
7. Breslow baseline hazard estimator
8. Schoenfeld residuals + PH test
9. Concordance index

**Phase 3** — Parametric + extensions (~150 lines):
10. Parametric survival (Weibull, log-normal, log-logistic, exponential)
11. AFT models
12. Competing risks (CIF, Fine-Gray)
13. Firth penalized Cox (for separation)

**Phase 4** — Advanced (~100 lines):
14. Time-varying covariates (counting process format)
15. Frailty models (shared, nested — wraps F11)
16. Cox-Snell residuals, martingale residuals, deviance residuals

---

## Composability Contract

```toml
[family_13]
name = "Survival Analysis"
kingdom = "Mixed — A (KM prefix product, risk sets), C (Cox PH Newton-Raphson)"

[family_13.shared_primitives]
kaplan_meier = "Prefix product of conditional survival"
cox_ph = "Partial likelihood maximization with suffix-sum risk sets"
nelson_aalen = "Prefix sum of hazard increments"
log_rank = "Risk-set-weighted comparison of survival curves"

[family_13.reuses]
f02_linear_algebra = "Hessian solve for Cox Newton step"
f05_optimization = "MLE for parametric survival"
f06_descriptive = "Risk set counts, cumulative sums"
f07_hypothesis = "Chi-square for log-rank, Wald tests"
f10_regression = "Cox as weighted regression (IRLS template)"
f11_mixed_effects = "Frailty models (random effects in survival)"

[family_13.provides]
survival_function = "KM or parametric S(t) with CI"
hazard_ratios = "Cox PH exp(β) with CI"
cumulative_incidence = "CIF for competing risks"
concordance = "C-statistic for model discrimination"

[family_13.consumers]
f34_bayesian = "Bayesian survival models"
f12_panel = "Duration models in panel data"
f18_volatility = "Event timing models in finance"
```
