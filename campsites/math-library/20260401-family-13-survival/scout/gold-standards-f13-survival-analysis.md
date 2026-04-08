# F13 Survival Analysis — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 13 (Survival Analysis).
Key architectural question: does Cox PH partial likelihood use IRLS?
Answer confirmed below: YES — Cox PH score equations are solved via Newton-Raphson
with the Hessian as the weight matrix = weighted scatter accumulate.
This makes Cox PH the **5th domain** for the IRLS master template.

---

## The IRLS Master Template — 5th Domain Confirmed

From the F10 supplement, the unified template:
```
Iteration:
  weights wᵢ = f(μᵢ, yᵢ)          // weight formula differs by algorithm
  β̂_new = solve(X'WX, X'Wz)        // weighted GramMatrix solve
```

Cox PH fits as:
```
wᵢ = second-order contribution from partial likelihood Hessian
z  = score vector element divided by weight
β_new = β + H^{-1} · score          // Newton step
```

Specifically:
- Score: `U(β) = Σᵢ∈D [xᵢ - Σⱼ∈R(tᵢ) xⱼ·exp(xⱼβ) / Σⱼ∈R(tᵢ) exp(xⱼβ)]`
- Hessian: `H(β) = -Σᵢ∈D [Σⱼ∈R(tᵢ) xⱼxⱼ'·exp(xⱼβ)/W_i - (Σⱼ∈R(tᵢ) xⱼ·exp(xⱼβ)/W_i)²]`

This IS a weighted outer product accumulate over the risk set R(tᵢ).
Each failure time contributes: `scatter_multi_phi_weighted(risk_set, w=exp(xβ))`.

**Tambear implication**: Cox PH = `scatter_multi_phi_weighted` on risk sets,
same primitive as GLM/Robust/EM. Different weight computation, same accumulate structure.

---

## Gold Standard Libraries

### R: survival package (definitive oracle)

```r
library(survival)

# Kaplan-Meier:
km_fit <- survfit(Surv(time, status) ~ group, data=df)
summary(km_fit)
km_fit$time           # unique event times
km_fit$surv           # S(t) estimate
km_fit$n.event        # number of events at each time
km_fit$n.risk         # number at risk at each time
km_fit$std.err        # Greenwood SE

# Plot:
plot(km_fit, conf.int=TRUE, col=c("blue","red"))

# Log-rank test:
survdiff(Surv(time, status) ~ group, data=df)
# Returns chi-square statistic and p-value

# Peto-Peto log-rank (weighted):
survdiff(Surv(time, status) ~ group, rho=1, data=df)  # rho=1 = Peto-Peto
```

```r
# Cox PH:
cox_fit <- coxph(Surv(time, status) ~ x1 + x2 + strata(grp), data=df)
summary(cox_fit)

cox_fit$coef              # β̂ (log hazard ratios)
exp(cox_fit$coef)         # HR = exp(β̂) (hazard ratios)
cox_fit$var               # variance-covariance matrix of β̂
cox_fit$loglik            # [null, final] log partial likelihood
cox_fit$score             # score test statistic (= log-rank at β=0)
cox_fit$wald.test         # Wald test statistic
cox_fit$concordance       # concordance (C-index) = AUC analog
cox_fit$linear.predictors # Xβ̂ for each observation

# Cox with time-varying covariates:
coxph(Surv(tstart, tstop, status) ~ x1 + x2, data=df_tv)

# Schoenfeld residuals (proportional hazards test):
cox_zph <- cox.zph(cox_fit)
cox_zph  # table with chi-square per covariate
plot(cox_zph)  # plot scaled Schoenfeld residuals vs time
```

```r
# Parametric survival:
library(survival)
survreg(Surv(time, status) ~ x1 + x2, data=df, dist="weibull")
survreg(Surv(time, status) ~ x1 + x2, data=df, dist="exponential")
survreg(Surv(time, status) ~ x1 + x2, data=df, dist="lognormal")
survreg(Surv(time, status) ~ x1 + x2, data=df, dist="loglogistic")
```

### Python: lifelines (primary oracle)

```python
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullFitter

# Kaplan-Meier:
kmf = KaplanMeierFitter()
kmf.fit(durations=df['time'], event_observed=df['status'])
kmf.survival_function_   # DataFrame: index=time, col=KM_estimate
kmf.median_survival_time_
kmf.confidence_interval_  # confidence band
kmf.cumulative_hazard_    # H(t) = -log(S(t))

# Grouped KM:
from lifelines.statistics import logrank_test, multivariate_logrank_test
results = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
results.p_value
results.test_statistic  # chi-square

# Cox PH:
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='status')
cph.print_summary()
cph.params_              # β̂ (log HR)
cph.hazard_ratios_       # exp(β̂)
cph.summary              # full table with CIs and p-values
cph.concordance_index_   # C-index
cph.log_likelihood_      # partial log-likelihood

# Predictions:
cph.predict_survival_function(X_new)  # S(t | x) for new data
cph.predict_cumulative_hazard(X_new)
cph.predict_median(X_new)             # median survival time
cph.predict_partial_hazard(X_new)     # exp(xβ̂) = relative risk

# Schoenfeld residuals / PH test:
cph.check_assumptions(df, p_value_threshold=0.05)
```

```python
# scikit-survival (sklearn-compatible interface):
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

# Convert target to structured array:
y = np.array([(bool(status), time) for status, time in zip(df.status, df.time)],
             dtype=[('status', bool), ('time', float)])

cox = CoxPHSurvivalAnalysis(alpha=0)   # alpha=0 = unregularized
cox.fit(X, y)
cox.coef_                              # β̂
cox.score(X, y)                        # concordance index

# KM:
times, surv_prob = kaplan_meier_estimator(y['status'], y['time'])
```

---

## Kaplan-Meier: What It Computes

```
S(t) = Π_{t_i ≤ t} (1 - d_i / n_i)

where:
  d_i = number of events at time t_i
  n_i = number at risk just before t_i (includes censored, excludes already-failed)
```

This is a **product over event times**, not a sum. The ECDF update rule.

**Tambear decomposition**:
1. Sort by time — SortedPermutation (Kingdom B infrastructure)
2. For each unique event time: gather d_i and n_i (event count and risk set size)
3. Product: `accumulate(Prefix, KaplanMeierStep, Multiply)` — running product
4. KaplanMeierStep(d, n) = (n - d) / n

This uses Prefix accumulate with multiply (not add). The combine op is multiplicative —
`S(t₁ ∪ t₂) = S(t₁) · S(t₂|t₁)`. Associative in the same sense as the Affine scan.

**Greenwood formula** (variance of S(t)):
```
Var(S(t)) = S(t)² · Σ_{t_i ≤ t} d_i / (n_i · (n_i - d_i))
```
= S(t)² · `accumulate(Prefix, GreenwoodTerm, Add)` — second running accumulate over the same sorted event times.

**Key**: the `n_i` (risk set size) is NOT trivially computed from the sorted data.
n_i = total observations - (events before t_i) - (censored before t_i).
Requires careful handling of ties between event time and censoring time.
**Tie rule**: censored observations at time t are counted in the risk set at t
(they might have survived past t — we just don't know).

---

## Log-Rank Test: What It Computes

```
χ² = (Σ_t O_t - E_t)² / Var

where for each event time t:
  O_t = observed events in group 1 at time t
  E_t = expected events in group 1 = (d_t · n_{1t}) / n_t
  (hypergeometric distribution under H₀)
```

**Tambear decomposition**: this is a `scatter(ByTime, ...)` → sum statistics per event time.
Same pattern as contingency table accumulation (F07 chi-square).

The log-rank test IS a chi-square test on a 2×(number of event times) contingency table,
where each column is one event time's 2×2 table of (events/censored) × (group1/group2).

---

## Cox PH: Full Mechanics

### The Partial Likelihood

Cox avoids specifying the baseline hazard h₀(t) by using only the information
at each failure time: which observation failed given the risk set.

```
L(β) = Π_{i: δᵢ=1} [exp(xᵢβ) / Σⱼ∈R(tᵢ) exp(xⱼβ)]

log L(β) = Σ_{i: δᵢ=1} [xᵢβ - log Σⱼ∈R(tᵢ) exp(xⱼβ)]
```

**Tied events** (multiple failures at same time): Breslow approximation (default) or Efron:
- Breslow: treat as if sequential, approximate denominator = full risk set product
- Efron: more accurate, intermediate denominator between Breslow and exact

### Score Vector

```
U_k(β) = ∂ log L / ∂β_k
       = Σ_{i: δᵢ=1} [x_{ik} - ē_k(tᵢ, β)]

where ē_k(t, β) = Σⱼ∈R(t) x_{jk}·exp(xⱼβ) / Σⱼ∈R(t) exp(xⱼβ)
                = weighted mean of feature k in risk set R(t) with weights exp(xⱼβ)
```

**Tambear key**: `ē_k(t, β)` = `scatter(ByGroup{risk_set_at_t}, WeightedMean{exp(xβ)}, feature_k)`.
This is `scatter_multi_phi_weighted` on the risk set for event time t.

### Hessian (Information Matrix)

```
H_kl(β) = -∂²log L / ∂β_k ∂β_l
         = Σ_{i: δᵢ=1} [ē_{kl}(tᵢ) - ē_k(tᵢ)·ē_l(tᵢ)]

where ē_{kl}(t) = Σⱼ∈R(t) x_{jk}·x_{jl}·exp(xⱼβ) / W(t)
     (weighted second moment = GramMatrix of risk set features weighted by exp(xβ))
```

**This IS a GramMatrix accumulate** on the risk set, with weights = exp(xⱼβ).
Same as F09's `Weighted GramMatrix`. Same as F16's EM M-step.

### Newton-Raphson Step

```
β_new = β + H(β)^{-1} · U(β)
```

= same Cholesky solve as OLS/Ridge/IRLS.

### Risk Set Computation

The risk set R(tᵢ) = {j : tⱼ ≥ tᵢ} — all subjects still under observation at time tᵢ.
For n subjects sorted by event time, R(tᵢ) is a suffix of the sorted array.
This is the Grouping::Suffix pattern — or equivalently, a reverse prefix scan.

**Trap**: risk sets are NESTED and shrinking. For ordered events t₁ < t₂ < ... < tₖ:
R(t₁) ⊃ R(t₂) ⊃ ... ⊃ R(tₖ). The weighted sums can be computed by scanning once
in reverse time order, subtracting each exiting observation. O(n) total work.

This is the **backward scan** pattern — `accumulate(Prefix, ..., Add)` run in reverse.

---

## Baseline Hazard: Breslow Estimator

After fitting β̂, the cumulative baseline hazard is:

```
Ĥ₀(t) = Σ_{i: tᵢ≤t, δᵢ=1} 1 / Σⱼ∈R(tᵢ) exp(xⱼβ̂)
```

= `accumulate(Prefix, BreslowStep, Add)` over sorted event times.
Each step contributes d_i / W(t_i) where W(t_i) = Σⱼ∈R(tᵢ) exp(xⱼβ̂).

---

## Parametric Survival: AFT Models

Accelerated Failure Time (AFT) models specify the distribution of log(T):

```
log(T) = xβ + σε
where ε ~ distribution (Weibull, log-normal, log-logistic)
```

These are standard parametric regression on log(T) — map directly to F10 regression
once the (possibly censored) likelihood is formed.

**Log-likelihood with censoring**:
```
ℓ(β, σ) = Σ_{i: δᵢ=1} log f((log tᵢ - xᵢβ)/σ)/σ
          + Σ_{i: δᵢ=0} log S((log tᵢ - xᵢβ)/σ)
```

This is a weighted MLE with two types of observations.
For Weibull/Exponential: closed form score equations.
For log-normal: Gaussian score equations with censored residuals.

**Tambear**: AFT = weighted GramMatrix solve with censored observations having modified z
(contribution from survival function, not density). IRLS framework applies.

```python
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

waf = WeibullAFTFitter()
waf.fit(df, duration_col='time', event_col='status')
waf.params_              # β̂ for each covariate
waf.print_summary()
```

---

## Competing Risks

When multiple causes of failure exist, standard KM gives biased estimates.

```r
library(cmprsk)
# Fine & Gray regression (competing risks):
fg <- crr(df$time, df$status, df[, covariates])
# status = 0 (censored), 1 (event of interest), 2 (competing event)

# Cumulative incidence function:
cif <- cuminc(df$time, df$status, df$group)
```

```python
from lifelines import AalenJohansenFitter
ajf = AalenJohansenFitter()
ajf.fit(df['time'], df['status'], event_of_interest=1)
ajf.cumulative_density_  # CIF: P(T ≤ t, cause = 1)
```

Fine & Gray = Cox regression on a pseudo-likelihood with modified risk set
(competing events remain in risk set but with downweighted contribution).
= same `scatter_multi_phi_weighted` with modified weights for competing events.

---

## Validation: WHAS500 Dataset (Standard Test Case)

```python
from lifelines.datasets import load_whas500

df = load_whas500()
# 500 patients, columns: lenfol (follow-up time), fstat (vital status), bmi, age, ...

from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(df, duration_col='lenfol', event_col='fstat')
cph.print_summary()
# Concordance index ~0.73 for age + bmi model (documented in lifelines docs)
```

```python
# Rossi recidivism dataset (also commonly used):
from lifelines.datasets import load_rossi
rossi = load_rossi()
# week (follow-up), arrest (event), fin (financial aid - main covariate)
cph.fit(rossi, duration_col='week', event_col='arrest')
# fin HR ≈ 0.62 (financial aid reduces recidivism by ~38%)
```

---

## Implementation Traps

### 1. Ties in Event Times

```python
# lifelines default: Efron approximation
CoxPHFitter(baseline_estimation_method="breslow")  # explicit Breslow
CoxPHFitter()  # default is Efron

# R survival default: Efron
coxph(Surv(t, s) ~ x, ties="efron")   # default
coxph(Surv(t, s) ~ x, ties="breslow")
coxph(Surv(t, s) ~ x, ties="exact")   # very slow, discrete time
```

**Trap**: lifelines and R may give slightly different results at tied event times
if Breslow vs Efron handling differs. Always specify explicitly for validation.

### 2. Censoring vs Event at Same Time

Standard convention: if censoring occurs at same time as event, the censored
observation IS in the risk set at that time (they could have had the event).

Some implementations get this backwards. Test with:
```python
# Create tied data:
df_test = pd.DataFrame({'time': [5, 5, 10], 'status': [1, 0, 1]})
# At t=5: 1 event, 1 censoring; at t=10: 1 event
# Risk set at t=5 should include ALL 3 (n_risk=3, d=1)
# Risk set at t=10 should include just the t=10 observation (n_risk=1, d=1)
```

### 3. Concordance Index (C-Index)

```python
from lifelines.utils import concordance_index
c = concordance_index(df['time'], -cph.predict_partial_hazard(df), df['status'])
# NOTE: negative of partial hazard because higher hazard = shorter survival
# Same as Harrell's C
```

```r
survival::concordance(cox_fit)$concordance  # 0.5 = random, 1.0 = perfect
```

C-index = P(higher risk → shorter survival | comparable pair). AUC analog for survival.

### 4. Time-Varying Covariates

Standard Cox cannot handle time-varying covariates directly.
The `Surv(tstart, tstop, event)` formulation splits each observation into intervals.

```r
# Must reshape to counting process format:
# (id, tstart, tstop, status, x)
# One row per interval per subject
coxph(Surv(tstart, tstop, status) ~ x + cluster(id), data=df_tv)
```

---

## Tambear Decomposition Summary

| Algorithm | Primitives | Kingdom |
|-----------|-----------|---------|
| Kaplan-Meier S(t) | Sort + Prefix(Multiply) | B |
| Greenwood variance | Sort + Prefix(Add) | B |
| Log-rank test | scatter(ByTime, Count/Expected) | A |
| Cox NR score | scatter_weighted(ByRiskSet) over event times | A+C |
| Cox NR Hessian | scatter_weighted GramMatrix over risk sets | A+C |
| Cox Breslow Ĥ₀ | Prefix(Add) over sorted event times | B |
| AFT Weibull | weighted GramMatrix (IRLS) | C (via F10) |
| Competing risks | scatter_weighted with modified weights | A+C |

**Core insight**: Cox PH requires computing weighted sums over RISK SETS (nested suffixes
of the sorted time array), not over the full data. This is a "shrinking scan" pattern:
start from the largest risk set (all subjects) and remove subjects as time advances.
Equivalent to a reverse Prefix scan. All weighted sums reduce to `scatter_multi_phi_weighted`
on the risk set, confirming Cox PH as the 5th IRLS domain.

**New primitive gap**: risk set management = sorted suffix indexing. Need:
- Observations sorted by event time (SortedPermutation)
- For each event time t_i: efficiently identify R(t_i) = subjects with t_j >= t_i
- This is O(n) via backward scan — existing Prefix(reverse) if Prefix can run backward

Alternatively: precompute risk set boundaries after sort (CPU, O(n log n)) and
pass boundaries to GPU for the weighted GramMatrix accumulate.

---

## IRLS Master Template — Full Confirmation

| Family | Weight formula | Z vector |
|--------|---------------|---------|
| OLS | wᵢ = 1 | z = y |
| Logistic | wᵢ = μᵢ(1-μᵢ) | z = ηᵢ + (yᵢ-μᵢ)/wᵢ |
| Poisson | wᵢ = μᵢ | z = ηᵢ + (yᵢ-μᵢ)/μᵢ |
| Huber robust | wᵢ = ψ(εᵢ/σ)/(εᵢ/σ) | z = y |
| GMM EM | wᵢₖ = rᵢₖ (responsibilities) | z = x |
| **Cox PH** | wᵢ ~ H_ii (Hessian diagonal) | z = β + H⁻¹·U |

Cox PH uses Newton-Raphson (= IRLS with exact Hessian) rather than IRLS with
approximate working weights, but the inner step is still: weighted GramMatrix solve.
`scatter_multi_phi_weighted` covers all 6 families with different weight functions.
