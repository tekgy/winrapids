# F12 Panel Data & Econometrics — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 12 (Panel Data & Econometrics).
This is the Stata replacement family — the algorithms here are what econometricians reach for
daily. The navigator's insight (F12 creation doc) is right: Fixed Effects is nearly free
after F06 (MomentStats(ByKey) = group means) + F10 (GramMatrix + Cholesky). The rest
of the family builds on that foundation.

Key architectural insight upfront: every panel estimator is **demean + OLS** with a
different demeaning strategy. The demean step = tambear accumulate (group means extraction).
The OLS step = GramMatrix solve (already established). F12 adds no new primitives — it
adds new grouping patterns and a new way to compose existing infrastructure.

---

## Index

1. Fixed Effects (FE) — within estimator
2. Random Effects (RE) — GLS with block diagonal Omega
3. First Differences (FD) — difference out time-invariant effects
4. 2SLS / Instrumental Variables (IV)
5. Difference-in-Differences (DiD) — 2×2 and staggered
6. GMM — Arellano-Bond dynamic panel GMM
7. Regression Discontinuity (RD) — local linear at cutoff
8. Synthetic Control Method

---

## Critical Package Landscape

### R Panel Packages

```r
# Core: plm — all panel estimators (FE, RE, FD, IV panel, GMM)
install.packages("plm")

# IV/2SLS (non-panel): AER — ivreg()
install.packages("AER")

# DiD (staggered): did — Callaway-Sant'Anna (2021)
install.packages("did")

# DiD (event study): staggered — Sun-Abraham, Gardner 2-stage
install.packages("staggered")  # by Kyle Butts
install.packages("fixest")     # feols() — extremely fast FE + IV

# RD: rdrobust — the standard for RD analysis
install.packages("rdrobust")

# Synthetic Control: Synth — Abadie, Diamond, Hainmueller
install.packages("Synth")

# Clustered SEs (sandwich estimator):
install.packages("sandwich")
install.packages("lmtest")
```

### Python Panel Packages

```python
# Core: linearmodels — mirrors plm for Python
# pip install linearmodels
from linearmodels import PanelOLS, PooledOLS, RandomEffects, BetweenOLS
from linearmodels import IV2SLS, IVGMM, IVLIML
from linearmodels.panel import compare  # compare multiple specs

# DiD / Causal:
# pip install doubleml
import doubleml

# DiD (staggered): pyfixest mirrors R fixest
# pip install pyfixest
import pyfixest as pf

# RD: rdd / rdrobust-python
# pip install rdrobust
from rdrobust import rdrobust

# Synth: pysynth or SynthControl
# pip install pysynth
```

---

## Section 1: Fixed Effects (Within Estimator)

### What FE Does

Panel data: N entities (firms, countries, people) observed T times each.
FE removes entity-specific unobserved heterogeneity by demeaning:

```
y_it - ȳ_i = (x_it - x̄_i)'β + (ε_it - ε̄_i)

where ȳ_i = (1/T) Σ_t y_it  (entity mean over time)
      x̄_i = (1/T) Σ_t x_it  (entity mean vector over time)
```

After demeaning (the "within" transformation), OLS on the demeaned data = FE estimator.

**This IS tambear MomentStats(ByKey{entity_id}) + subtract + GramMatrix.**

### Tambear Decomposition

```
Step 1: group_means_y = accumulate(ByKey{entity_id}, phi="v", Add) / count(ByKey{entity_id})
        group_means_X = accumulate(ByKey{entity_id}, phi="v_k", Add) / count(ByKey{entity_id})
        [This is MomentStats(order=1, ByKey{entity_id}) — already established]

Step 2: y_demeaned_it = y_it - group_means_y[entity_i]
        X_demeaned_it = x_it - group_means_X[entity_i]
        [This is a gather(ByKey{entity_id}) → subtract — already established]

Step 3: β̂_FE = OLS on (y_demeaned, X_demeaned)
        = GramMatrix(X_demeaned) + Cholesky solve
        [F10 GramMatrix path]

Step 4: SE adjustment: degrees of freedom = n - N - k
        (subtract N entity-specific means from residual df)
```

Two-way FE (entity + time): demean by entity AND time. Iterative demeaning (Gauss-Seidel)
until convergence. R's `fixest` uses this. Equivalent to including entity and time dummies.

### R: plm

```r
library(plm)

# Create panel data frame (entity = "id", time = "year"):
pdata <- pdata.frame(df, index=c("id", "year"))

# One-way FE (entity effects):
fe_model <- plm(y ~ x1 + x2,
                data   = pdata,
                model  = "within",
                effect = "individual")   # entity FE

# Two-way FE (entity + time):
fe2_model <- plm(y ~ x1 + x2,
                 data   = pdata,
                 model  = "within",
                 effect = "twoways")    # entity + time FE

# Summary:
summary(fe_model)
# Note: plm does NOT report entity FE intercepts — they're partialled out.
# It reports within-transformation R² (not comparable to OLS R²).

# Extract FE intercepts (entity-specific fixed effects):
fixef(fe_model)           # vector of entity fixed effects α̂_i
fixef(fe_model, type="dmean")   # demeaned (sum to zero)
fixef(fe_model, type="dfirst")  # first entity = reference

# Residuals and fitted values:
residuals(fe_model)
fitted(fe_model)

# F-test for joint significance of FE (tests H₀: all α_i equal):
pFtest(fe_model, ols_model)    # plm F-test: FE vs pooled OLS
```

**Trap: plm's R² is within-R², NOT overall R².**
The within-R² measures variation explained AFTER removing entity means.
If you compare plm's R² to sklearn's R², they will disagree — they measure different things.

**Trap: plm requires balanced or declared unbalanced panels.**
For unbalanced panels (different T_i per entity), check the `pbalancedness()` diagnostic:
```r
pdim(pdata)           # check panel dimensions (N, T, balanced?)
is.pbalanced(pdata)   # TRUE/FALSE
```

### R: fixest (faster, especially for two-way FE)

```r
library(fixest)

# Two-way FE (highly optimized for large N, T):
fe2_fast <- feols(y ~ x1 + x2 | id + year,
                  data = df,
                  cluster = ~id)   # clustered SE by entity
summary(fe2_fast)

# feols syntax: formula | fixed_effects (after the pipe)
# Multiple FEs: | id + year + industry
# Interaction FEs: | id^year

# Compare to OLS without FE:
ols_base <- feols(y ~ x1 + x2, data=df)

# Robust SE options:
feols(y ~ x1 + x2 | id, data=df, se="hetero")       # Heteroscedastic-robust
feols(y ~ x1 + x2 | id, data=df, cluster=~id)        # Cluster-robust (by entity)
feols(y ~ x1 + x2 | id, data=df, cluster=~id+year)   # Two-way clustered SE

# etable() for regression tables:
etable(ols_base, fe2_fast, tex=FALSE)
```

**fixest is 10-100x faster than plm for large panels** (Demsar-style alternating projections
algorithm for two-way FE). Use fixest as the primary oracle for large-N benchmarks.

### Python: linearmodels.PanelOLS

```python
import pandas as pd
import numpy as np
from linearmodels import PanelOLS

# Panel data must have a MultiIndex: (entity, time)
df = df.set_index(['id', 'year'])

# One-way entity FE:
fe_model = PanelOLS(
    dependent  = df['y'],
    exog       = df[['x1', 'x2']],
    entity_effects = True,    # entity FE
    time_effects   = False,
)
fe_result = fe_model.fit(cov_type='clustered', cluster_entity=True)
print(fe_result.summary)

# Two-way FE:
fe2_model = PanelOLS(
    dependent      = df['y'],
    exog           = df[['x1', 'x2']],
    entity_effects = True,
    time_effects   = True,
)
fe2_result = fe2_model.fit()

# Key result attributes:
fe_result.params          # β̂
fe_result.std_errors      # SE(β̂)
fe_result.tstats          # t-statistics
fe_result.pvalues         # p-values
fe_result.rsquared        # within R²
fe_result.rsquared_between  # between R² (across entity means)
fe_result.rsquared_overall  # overall R² (pooled data)
fe_result.estimated_effects  # α̂_i (entity fixed effects)
```

### Clustered Standard Errors

This is the #1 trap in panel econometrics. Heteroscedasticity-robust SEs (HC0/HC1) are
WRONG for panel data — panel residuals are correlated within entity. Use cluster-robust SEs.

**The Sandwich Estimator (clustered):**
```
Var(β̂_FE) = (X̃'X̃)^{-1} · [Σ_i X̃_i'ε̃_iε̃_i'X̃_i] · (X̃'X̃)^{-1}

where X̃ = demeaned X, ε̃ = within-transformation residuals
      i = entity index (clusters)
      X̃_i = rows of X̃ belonging to entity i (T_i × k matrix)
      ε̃_i = residuals for entity i (T_i × 1 vector)
```

This is NOT the Huber-White sandwich — Huber-White treats each observation as a cluster.
Clustered SEs treat each entity as a cluster. They are the SAME formula but clustered
differently.

```r
# R: using sandwich + lmtest (for non-plm models):
library(sandwich)
library(lmtest)

# Cluster-robust SE for lm model (non-panel context):
lm_fit <- lm(y ~ x1 + x2 + factor(id), data=df)
coeftest(lm_fit, vcov=vcovCL(lm_fit, cluster=~id))

# For plm FE models (vcovHC is heteroscedastic only):
coeftest(fe_model, vcov=vcovHC(fe_model, type="HC1"))      # heteroscedastic
coeftest(fe_model, vcov=vcovCL(fe_model, cluster=~id))     # clustered by entity
coeftest(fe_model, vcov=vcovDC(fe_model))                  # double-clustered (id + time)

# Note: vcovHC(plm) is confusingly named — plm's vcovHC already does
# cluster-robust adjustment for panel (it's not plain HC1).
# vcovCL from sandwich gives explicit entity clustering.
```

```python
# linearmodels clustered SE options:
fe_result = fe_model.fit(cov_type='clustered', cluster_entity=True)    # cluster by entity
fe_result = fe_model.fit(cov_type='clustered', cluster_time=True)      # cluster by time
fe_result = fe_model.fit(cov_type='clustered',
                          cluster_entity=True, cluster_time=True)       # two-way cluster
fe_result = fe_model.fit(cov_type='robust')                             # HC robust (NOT for panels)
```

---

## Section 2: Random Effects (RE)

### What RE Does

RE assumes entity effects α_i are RANDOM (uncorrelated with x_it). Under this assumption,
RE is more efficient than FE (uses between-entity variation too). If the assumption is
violated, RE is inconsistent — use Hausman test to decide.

The RE estimator is GLS with block-diagonal Ω:
```
Var(u_it) = σ²_α + σ²_ε    (same entity, same time)
Var(u_it, u_is) = σ²_α      (same entity, different time)
Var(u_it, u_js) = 0          (different entity)

Ω = block_diag(Ω_1, ..., Ω_N)
Ω_i = σ²_ε · I_{T_i} + σ²_α · 1_{T_i}1'_{T_i}  (T_i × T_i block per entity)
```

The RE estimator is the quasi-demeaned (partial demeaning) estimator:
```
y_it - θ_i·ȳ_i = (x_it - θ_i·x̄_i)'β + ...

where θ_i = 1 - sqrt(σ²_ε / (T_i·σ²_α + σ²_ε))
```

Note: θ ∈ (0, 1). When θ=0, RE = pooled OLS (ignore entity structure). When θ=1, RE = FE.

### Tambear Decomposition

```
Step 1: Estimate variance components (σ²_α, σ²_ε) from OLS residuals
        σ²_ε = RSS_within / (n·T - N - k)    [from FE residuals]
        σ²_α = RSS_between / (N - k - 1) - σ²_ε/T  [from between-entity OLS]

Step 2: Compute θ per entity: θ_i = 1 - sqrt(σ²_ε / (T_i·σ²_α + σ²_ε))

Step 3: Quasi-demean: y*_it = y_it - θ·ȳ_i
                              x*_it = x_it - θ·x̄_i
        [Same MomentStats(ByKey) as FE, different θ factor]

Step 4: OLS on (y*, X*) = GramMatrix + Cholesky
```

RE is a weighted blend of FE (within) and between-entity OLS — it needs BOTH demeaning
paths. The variance component estimation is itself an OLS solve (the "between" estimator).

### R: plm (RE)

```r
library(plm)

# Random Effects:
re_model <- plm(y ~ x1 + x2,
                data   = pdata,
                model  = "random",
                random.method = "swar")   # Swamy-Arora (default, most common)

# Other variance component methods:
re_amemiya <- plm(y ~ x1 + x2, data=pdata, model="random", random.method="amemiya")
re_walhus   <- plm(y ~ x1 + x2, data=pdata, model="random", random.method="walhus")
re_nerlove  <- plm(y ~ x1 + x2, data=pdata, model="random", random.method="nerlove")

# ercomp() extracts variance components:
ercomp(re_model)
# $sigma2: named vector with "idios" (σ²_ε) and "id" (σ²_α)
# $theta: the θ parameter(s)

summary(re_model)
```

**Trap: random.method matters.** Different methods give different σ²_α estimates, different θ,
and thus different β̂_RE. Swamy-Arora (swar) is the standard; use it for gold standard
comparisons. Match the method when comparing R vs Python.

### Hausman Test (FE vs RE)

```r
# Hausman test: H₀ = RE is consistent (entity effects uncorrelated with x)
# If rejected (p < 0.05): use FE. If not rejected: RE is more efficient.
phtest(fe_model, re_model)
# Returns chi-square statistic and p-value

# The test statistic:
# H = (β̂_FE - β̂_RE)' [Var(β̂_FE) - Var(β̂_RE)]^{-1} (β̂_FE - β̂_RE)
# Under H₀: H ~ χ²(k)  where k = number of time-varying regressors
```

**Tambear decomposition of Hausman test:**
```
d = β̂_FE - β̂_RE           [vector subtraction after solving both systems]
V = Var(β̂_FE) - Var(β̂_RE)  [matrix difference — both from GramMatrix inverses]
H = d' V^{-1} d             [quadratic form = GramMatrix inverse matvec]
p = chi2.sf(H, df=k)        [chi-square tail probability]
```

This is F07 territory (chi-square test), using the same variance matrices already
computed from GramMatrix solves.

```python
from linearmodels import RandomEffects
re_model = RandomEffects(df['y'], df[['x1','x2']]).fit()

# Hausman test (compare FE to RE):
from linearmodels.panel.utility import generate_cov
# linearmodels does not have a dedicated Hausman test function;
# compute manually:
import numpy as np
b_fe = fe_result.params
b_re = re_result.params
V_fe = fe_result.cov
V_re = re_result.cov
# Use only time-varying regressors (not constant columns):
diff = b_fe - b_re
V_diff = V_fe - V_re  # Hausman assumes efficiency of RE under H₀
H_stat = float(diff @ np.linalg.solve(V_diff, diff))
from scipy.stats import chi2
p_value = chi2.sf(H_stat, df=len(diff))
print(f"Hausman H={H_stat:.4f}, p={p_value:.4f}")
```

---

## Section 3: First Differences (FD)

### What FD Does

FD removes entity-specific effects by taking first differences across time:
```
Δy_it = y_it - y_{i,t-1}
Δx_it = x_it - x_{i,t-1}

OLS on: Δy_it = Δx_it'β + Δε_it
```

FD is equivalent to FE when T=2. For T>2, FE is more efficient IF errors are i.i.d.
If errors follow a random walk (unit root), FD is more efficient.

### Tambear Decomposition

```
FD = lag_gather + subtract + OLS

lag_gather:
  For each entity i, observation (i, t):
    y_lag[i,t] = y[i, t-1]   // Segmented lag within each entity's time series
    x_lag[i,t] = x[i, t-1]

  This is gather(Segmented{entity_id, time_offset=-1}) — NOT currently implemented.
  Closest available: Windowed(w=2) applied within each entity segment.

subtract:
  Δy = y - y_lag   // elementwise subtract after gather
  ΔX = X - X_lag

OLS:
  β̂_FD = GramMatrix(ΔX) + Cholesky solve
```

FD requires **Segmented(Prefix{entity})** grouping = the `todo!()` grouping in current
tambear. This is the one genuinely new grouping FD needs beyond what FE provides.

### R: plm (FD)

```r
# First Differences:
fd_model <- plm(y ~ x1 + x2,
                data  = pdata,
                model = "fd")

summary(fd_model)
# Note: the intercept in FD represents a time trend (Δy when Δx=0).
# Suppress with -1 if no trend assumed.

fd_no_trend <- plm(y ~ x1 + x2 - 1,
                   data  = pdata,
                   model = "fd")
```

**Trap: FD loses the first time period.** If your panel has T=5, the FD dataset has T-1=4
observations per entity. The intercept counts as a parameter, so df = N(T-1) - k - 1
(with intercept) or N(T-1) - k (without).

```python
# First Differences in linearmodels: use FirstDifferenceOLS
from linearmodels import FirstDifferenceOLS

fd_model = FirstDifferenceOLS(df['y'], df[['x1','x2']]).fit()
fd_result = fd_model.fit()
fd_result.params
fd_result.std_errors
```

---

## Section 4: 2SLS / Instrumental Variables (IV)

### What IV/2SLS Does

When x is endogenous (correlated with ε), OLS is biased. IV uses instruments z that are:
1. Correlated with x (relevance: Cov(z, x) ≠ 0)
2. Uncorrelated with ε (exclusion: Cov(z, ε) = 0)

2SLS is the most common IV estimator:
```
Stage 1: x̂_it = Z_it'π̂   (regress endogenous x on instruments Z)
         π̂ = (Z'Z)^{-1} Z'x    [GramMatrix solve 1]

Stage 2: y_it = x̂_it'β̂_2SLS + ε_it   (regress y on predicted x)
         β̂_2SLS = (X̂'X̂)^{-1} X̂'y    [GramMatrix solve 2, using X̂ from stage 1]
```

**Tambear decomposition: 2SLS = two sequential GramMatrix solves.**

The IV estimator in matrix form (directly, without two stages):
```
β̂_IV = (Z'X)^{-1} Z'y     [just-identified case, one instrument per endogenous var]

β̂_2SLS = (X̂'X)^{-1} X̂'y  [over-identified: X̂ = Z(Z'Z)^{-1}Z' X = projection]
```

Both are GramMatrix operations. The two-stage formulation is just a convenient computation
strategy — the final estimator is the same.

### R: AER::ivreg (non-panel IV)

```r
library(AER)

# IV regression: y ~ exogenous_regressors + endogenous | instruments
# Formula: y ~ x1 + x_endog | x1 + z1 + z2
# (everything left of | = regressors, right of | = valid instruments)
# Exogenous regressors (x1) appear on BOTH sides — they instrument themselves.

iv_model <- ivreg(y ~ x1 + x_endog | x1 + z1 + z2,
                  data = df)

summary(iv_model)
summary(iv_model, diagnostics=TRUE)
# diagnostics=TRUE adds:
#   - Wu-Hausman test (endogeneity: H₀ = OLS is consistent)
#   - Sargan/Basmann test (overidentification: H₀ = instruments are valid)
#   - Weak instruments test (F-test for first stage: rule of thumb F > 10)

# Extract:
coef(iv_model)         # β̂_2SLS
vcov(iv_model)         # variance-covariance matrix
residuals(iv_model)    # structural residuals
```

### R: plm::plm with IV (panel IV)

```r
# Panel IV with entity FE (Hausman-Taylor, Anderson-Hsiao):
fe_iv_model <- plm(y ~ x1 + x_endog | x1 + z1 + z2,
                   data   = pdata,
                   model  = "within",
                   inst.method = "bvk")   # default
# bvk = Balestra-Vagia-Krishnakumar instruments (standard panel IV)
# Note: plm IV applies FE AFTER 2SLS-style instrumentation.
```

### R: fixest::feols with IV (preferred for large panels)

```r
library(fixest)

# IV with FE:
# Formula: y ~ exogenous | FE | endogenous ~ instruments
iv_fe <- feols(y ~ x1 | id + year | x_endog ~ z1 + z2,
               data = df,
               cluster = ~id)

summary(iv_fe)
# feols IV: partial F-stat for first stage, Sargan test available
fitstat(iv_fe, ~ ivf + ivwald + sargan)  # instrument diagnostics
```

### Python: linearmodels.IV2SLS

```python
from linearmodels import IV2SLS

# IV2SLS(y, exogenous, endogenous, instruments)
iv_model = IV2SLS(
    dependent   = df['y'],
    exog        = df[['const', 'x1']],    # must add constant manually
    endog       = df[['x_endog']],
    instruments = df[['z1', 'z2']],
)
iv_result = iv_model.fit(cov_type='robust')

iv_result.params           # β̂_2SLS (endogenous + exogenous)
iv_result.std_errors       # SE
iv_result.pvalues          # p-values
iv_result.first_stage      # first-stage regression results
iv_result.wu_hausman       # Wu-Hausman endogeneity test
iv_result.wooldridge_score # Wooldridge score test for endogeneity
iv_result.sargan           # Sargan over-identification test
iv_result.basmann          # Basmann test
```

### Weak Instrument Test

```r
# Rule of thumb: first-stage F > 10 (Staiger-Stock 1997)
# For multiple instruments: use Stock-Yogo critical values

# In ivreg:
summary(iv_model, diagnostics=TRUE)$diagnostics
# Row "Weak instruments": F-statistic for first stage

# In fixest:
fitstat(iv_fe, ~ ivf)   # partial F for each endogenous variable
```

The F > 10 rule is conservative. Stock-Yogo (2005) provide critical values for specific
bias tolerances (e.g., "F > 16.38 for at most 10% relative bias, 2 instruments").

**Key trap: weak instruments cause 2SLS to have heavy-tailed distributions and large bias.**
The Nagar (1959) approximation: bias of 2SLS ≈ (1/F) × (OLS bias). So F=10 → 2SLS still
has ~10% of the OLS endogeneity bias.

---

## Section 5: Difference-in-Differences (DiD)

### Standard 2×2 DiD

Treatment vs control, before vs after. The classic DiD estimator:

```
DiD = (ȳ_treat_post - ȳ_treat_pre) - (ȳ_ctrl_post - ȳ_ctrl_pre)

Equivalent regression:
y_it = α + β·treat_i + γ·post_t + δ·(treat_i × post_t) + ε_it

δ̂ = DiD estimate (OLS coefficient on interaction term)
```

**Tambear decomposition**: DiD is F07 two-way ANOVA structure.
```
group means = MomentStats(ByKey{(treat, post)})  // 4 cells
DiD = means[1,1] - means[1,0] - means[0,1] + means[0,0]  // contrast
```
Or equivalently: OLS with treatment, post, and interaction dummy — standard GramMatrix solve.

```r
# Manual 2x2 DiD:
df$treat_x_post <- df$treat * df$post
lm_did <- lm(y ~ treat + post + treat_x_post, data=df)
summary(lm_did)
coef(lm_did)["treat_x_post"]  # DiD estimate = δ̂

# With entity FE (recommended for panel data):
fe_did <- feols(y ~ treat_x_post | id + year, data=df, cluster=~id)
# Here 'treat' and 'post' are absorbed by id and year FEs respectively
# Only the interaction term is identified in two-way FE
```

```python
import statsmodels.formula.api as smf
result = smf.ols('y ~ treat + post + treat:post', data=df).fit()
result.params['treat:post']   # DiD estimate
```

### Parallel Trends Assumption

DiD is only valid if treatment and control groups would have trended the same way in the
absence of treatment. **This is untestable with only two periods** but can be checked
visually or via pre-trends test in event study designs.

```r
# Pre-trends test: regress on leads and lags of treatment
# If pre-period coefficients ≈ 0: parallel trends is plausible
feols(y ~ i(time_to_treat, ref=-1) | id + year, data=df, cluster=~id)
# i() creates event-study dummies centered on -1 (period before treatment)
# Pre-period: time_to_treat ∈ {-3, -2} should have β ≈ 0
```

### Staggered DiD: The Major Trap

**With staggered treatment timing (different entities treated at different times), the
simple 2×2 DiD regression is BIASED.** This is the major empirical finding from
Callaway-Sant'Anna (2021), Goodman-Bacon (2021), Sun-Abraham (2021).

The problem: twoway FE DiD decomposes into a weighted average of 2×2 comparisons,
but some weights are NEGATIVE — it can show a negative aggregate even when every single
2×2 comparison shows a positive effect.

**Never use `lm(y ~ treat_x_post + factor(id) + factor(t))` with staggered timing.**

### Callaway-Sant'Anna (2021): The Gold Standard for Staggered DiD

```r
library(did)

# att_gt() computes group-time average treatment effects (ATT(g,t))
# Group = cohort = period when entity first treated
# Time = calendar period
cs_did <- att_gt(
    yname   = "y",           # outcome variable
    tname   = "year",        # time variable
    idname  = "id",          # entity identifier
    gname   = "first_treat", # period of first treatment (0 = never treated)
    data    = df,
    control_group = "nevertreated",  # or "notyettreated"
    est_method = "reg",              # regression adjustment; or "dr", "ipw"
    bstrap  = TRUE,          # bootstrap SEs (recommended)
    biters  = 1000,          # bootstrap iterations
)
summary(cs_did)

# Aggregate to overall ATT:
agg_att <- aggte(cs_did, type="simple")   # simple average of ATT(g,t)
agg_att_dynamic <- aggte(cs_did, type="dynamic")  # event-study aggregation
summary(agg_att)

# Plot event-study:
ggdid(agg_att_dynamic)  # pre-trends test + post-treatment dynamics
```

```python
# Callaway-Sant'Anna in Python — no direct 1:1 port
# Options:
# 1) pyfixest (implements Callaway-Sant'Anna):
import pyfixest as pf
fit = pf.did2s(df, yname="y", first_stage="~ 0 | id + year",
               second_stage="~ treat", treatment="treat",
               cluster="id")  # Gardner (2021) 2-stage estimator
# 2) Manual implementation using linearmodels IV for the DR estimator
```

### Sun-Abraham (2021): Alternative for Staggered DiD

```r
library(fixest)

# Sun-Abraham via feols: interact treatment indicators with cohort (first treatment period)
sa_did <- feols(y ~ sunab(first_treat, year) | id + year,
                data    = df,
                cluster = ~id)
iplot(sa_did)   # event-study plot, pre-trends + dynamics

# Heterogeneity-robust aggregate effect:
aggregate(sa_did, agg="ATT")
```

**Note**: Sun-Abraham and Callaway-Sant'Anna give numerically identical estimates when
using the same control group. Sun-Abraham is faster (absorbed into feols); CS gives
more flexible aggregation.

---

## Section 6: GMM — Arellano-Bond Dynamic Panel

### What Arellano-Bond Does

Dynamic panel: lagged dependent variable as regressor. OLS is biased (Nickell 1981 bias):
```
y_it = α·y_{i,t-1} + x_it'β + α_i + ε_it
```

In FE estimation, y_{i,t-1} is correlated with the demeaned error — FE is also biased.
Arellano-Bond (1991) solves this via GMM:

```
First-difference to remove FE:
Δy_it = α·Δy_{i,t-1} + Δx_it'β + Δε_it

Instruments for Δy_{i,t-1}:
  y_{i,t-2} is uncorrelated with Δε_it if ε is serially uncorrelated
  y_{i,t-3}, y_{i,t-4}, ... are additional valid instruments
  This gives T(T-1)/2 instruments — "instrument proliferation" problem
```

The moment conditions: E[y_{i,t-s} · Δε_it] = 0 for s ≥ 2.

GMM weighting matrix: two-step GMM uses estimated variance of residuals as weight.

### Tambear Decomposition

```
Arellano-Bond = FD + GMM with lagged levels as instruments

Step 1: First difference (Segmented lag gather + subtract)
Step 2: Construct instrument matrix Z (lagged levels stacked as block-diagonal)
        Z is sparse — dense construction wastes memory
Step 3: One-step GMM: β̂ = (X̃'Z(Z'Z)^{-1}Z'X̃)^{-1} X̃'Z(Z'Z)^{-1}Z'ỹ
        where X̃ = [Δy_{t-1}, ΔX], ỹ = Δy
        This is a chain of GramMatrix operations: Z'Z, Z'X̃, Z'ỹ
Step 4: Two-step: use residuals from step 3 to update weight matrix W
        β̂_2step = (X̃'ZW Z'X̃)^{-1} X̃'ZW Z'ỹ
        W = Σ_i (Z_i' ε̂_i ε̂_i' Z_i)^{-1}  [sandwich of moment conditions]
Step 5: Sargan/Hansen test (over-identification)
        Autocorrelation tests (AR(1) should exist, AR(2) should not)
```

### R: plm::pgmm

```r
library(plm)

# Arellano-Bond (AB1 = one-step, AB2 = two-step):
ab_model <- pgmm(
    dynformula(y ~ x1 + x2, lag.form=list(y=1:2, x1=0:1)),
    # dynformula: y with lags 1-2 as regressors; x1 with lags 0-1
    data       = pdata,
    index      = c("id", "year"),
    effect     = "twoways",
    model      = "twosteps",    # one-step: "onestep"
    transformation = "d",       # "d" = first differences (Arellano-Bond)
                                # "ld" = levels+differences (Blundell-Bond, system GMM)
    lag.gmm    = list(y=c(2, 99))  # use lags 2-99 of y as instruments
)
summary(ab_model)
# Key outputs:
# - Coefficient on lag(y): the autoregressive parameter α̂
# - Sargan test: overidentification (H₀ = instruments valid)
#   High p-value = instruments not rejected (want p > 0.10)
# - AR(1): should reject (H₀ = no first-order autocorrelation)
#   AR(2): should NOT reject (H₀ = no second-order autocorrelation)
#   If AR(2) is rejected: first lags are NOT valid instruments
```

**Trap: instrument count.** With many time periods, the number of instruments grows as
O(T²). Rule of thumb: instruments < entities (I < N). With too many instruments, the
Sargan test is oversized (rejects too easily). Collapse instruments with `lag.gmm=list(y=2:4)`
instead of `list(y=2:99)`.

```r
# Blundell-Bond system GMM (adds level equations with lagged differences as instruments):
bb_model <- pgmm(
    dynformula(y ~ x1 + lag(y, 1), lag.form=list(x1=0:1)),
    data       = pdata,
    index      = c("id", "year"),
    model      = "twosteps",
    transformation = "ld"     # "ld" = level+difference = system GMM
)
# System GMM is preferred when the autoregressive parameter is close to 1
# (near-unit-root panels: asset prices, large aggregate-level data)
```

```python
# Dynamic panel GMM in linearmodels:
from linearmodels.panel import DynamicPanel  # note: experimental in older versions
# Alternative: use plm in R as oracle; Python support for AB is limited.

# For production Python GMM, the most reliable is the rpy2 bridge to R's plm:
import rpy2.robjects as ro
# ... or replicate via moment conditions manually
```

---

## Section 7: Regression Discontinuity (RD)

### What RD Does

Treatment D_i is a deterministic function of a "running variable" X_i:
```
D_i = 1 if X_i ≥ c  (sharp RD)
      P(D_i=1|X_i) has a jump at c  (fuzzy RD)
```

The RD estimate = local average treatment effect at the cutoff c:
```
τ_RD = lim_{x↓c} E[y|X=x] - lim_{x↑c} E[y|X=x]
```

Estimated via local linear regression (LLR) in a bandwidth h around c:
```
Left: minimize Σ_{c-h ≤ X_i < c} (y_i - α_L - β_L(X_i - c))² · K((X_i-c)/h)
Right: minimize Σ_{c ≤ X_i ≤ c+h} (y_i - α_R - β_R(X_i - c))² · K((X_i-c)/h)
τ_RD = α_R - α_L
```

K = kernel weight function (triangular kernel is MSE-optimal for LLR).

### Tambear Decomposition

```
RD = Masked{X_i ∈ (c-h, c+h)} + WLS with kernel weights

Step 1: Select observations in bandwidth: Masked{abs(X_i - c) < h}
Step 2: Assign kernel weights: w_i = K((X_i-c)/h)
        Triangular: w_i = (1 - |X_i-c|/h)  [linear decay to boundary]
Step 3: WLS with design matrix [1, (X-c), D, D·(X-c)] (4-column):
        β̂ = weighted GramMatrix solve
        τ̂_RD = β̂[3]  (coefficient on treatment indicator D)

Bandwidth selection (rdrobust default: MSE-optimal):
  h* = C · n^{-1/5}  (rate-optimal)
  C estimated from data via pilot regression
  This is an optimization problem, NOT a one-pass accumulate
```

WLS = weighted GramMatrix = already established via F10. The only new piece is bandwidth
selection (CCT optimal bandwidth, from Calonico-Cattaneo-Titiunik 2014).

### R: rdrobust

```r
library(rdrobust)

# Basic RD estimate:
rdd <- rdrobust(y=df$outcome, x=df$running_var, c=0)
# c = cutoff (default 0; center your running variable if needed)
summary(rdd)
# Key outputs:
# - τ̂ (coefficient): conventional, bias-corrected (bc), robust (rbc)
# - SE: conventional, robust
# - CI: conventional 95% CI, robust CI (use robust for inference)
# - Bandwidth: left (h_l), right (h_r), pilot bandwidth (b)
# - Effective N: observations within bandwidth (left, right)

# Extract:
rdd$coef    # c(conventional, bias-corrected, robust)
rdd$se      # standard errors
rdd$pv      # p-values
rdd$ci      # confidence intervals (lower, upper)
rdd$bws     # bandwidths used
rdd$N_h     # effective sample sizes

# Bandwidth selection only (without estimation):
rdbwselect(y=df$outcome, x=df$running_var, c=0, bwselect="mserd")
# bwselect options: "mserd" (MSE, same bw left/right), "msetwo" (MSE, different bw)
#                   "cerrd" (CER coverage error rate), "certwo"

# Local linear with manual bandwidth:
rdrobust(y=df$outcome, x=df$running_var, c=0, h=5.0)  # fixed bandwidth = 5

# Fuzzy RD (local IV at cutoff):
rdrobust(y=df$outcome, x=df$running_var, c=0, fuzzy=df$treatment)

# RD plot:
rdplot(y=df$outcome, x=df$running_var, c=0)  # binned scatter plot with fit
```

**Trap: DO NOT use polynomial RD.** High-order polynomial global RD (as in old literature)
is sensitive to polynomial degree and sample composition outside the bandwidth.
Gelman-Imbens (2019) showed this causes severe bias. **Local linear is the current standard.**

**Trap: Conventional CI is too narrow.** The `rdd$ci[, "Robust"]` column is the
Calonico-Cattaneo-Titiunik robust CI that accounts for bias from the bandwidth selection.
Always report the robust CI. The conventional CI is anti-conservative.

```python
from rdrobust import rdrobust, rdbwselect, rdplot

rdd = rdrobust(y=df['outcome'].values, x=df['running_var'].values, c=0)
print(rdd)
rdd.coef    # [conventional, bias-corrected, robust]
rdd.se      # standard errors
rdd.ci      # confidence intervals (lower, upper rows)
rdd.bws     # bandwidths
```

---

## Section 8: Synthetic Control Method

### What Synthetic Control Does

For cases where there is no clean control group: use a convex combination of control units
to construct a synthetic counterfactual for the treated unit.

```
y_1t = Σ_{j=2}^{J+1} w_j · y_jt   (pre-treatment fit)

Find weights w = (w_2, ..., w_{J+1}):
  Minimize ||X_1 - X_0 W||_V  (pre-treatment covariate balance)
  subject to: w_j ≥ 0, Σ_j w_j = 1

where X_0 = [pre-treatment outcomes, predictors for controls] (J × k matrix)
      X_1 = [pre-treatment outcomes, predictors for treated unit] (k-vector)
      V = diagonal weight matrix (how important is each predictor)
```

The ATT post-treatment:
```
τ̂_t = y_1t - Σ_{j=2}^{J+1} ŵ_j · y_jt   (for t in post-treatment period)
```

V is estimated via nested optimization (outer = V, inner = w given V).
Inference via permutation ("in-space" placebo tests — apply to all control units).

### Tambear Decomposition

```
Synthetic Control = constrained quadratic optimization

Step 1: Construct X_0 (J × k predictor matrix for controls)
        Rows = control units; columns = pre-period outcomes + covariates
        → accumulate(ByKey{unit_id}, phi=...) for each predictor

Step 2: Inner optimization: given V, find w
        min_{w ≥ 0, Σw=1} ||X_1 - X_0'w||²_V
        = quadratic program with simplex constraint
        → F05 (optimization): projected gradient or NNLS on simplex

Step 3: Outer optimization: find V to minimize pre-treatment fit
        → F05 (Nelder-Mead or gradient over V diagonal)

Step 4: Post-treatment gap: y_1t - X_0'w
        → gather(fixed weights from w) + subtract

Step 5: Placebo inference: repeat steps 1-4 for each control unit
        → embarrassingly parallel over control units
```

Synthetic Control needs no new tambear primitives — it's F05 optimization + GramMatrix
covariate construction + gather. The constraint is the new piece (simplex projection).

### R: Synth

```r
library(Synth)

# Synth requires specific data format:
dataprep_out <- dataprep(
    foo          = df,
    predictors   = c("gdp", "population", "trade_openness"),
    predictors.op = "mean",     # how to aggregate predictors over pre-period
    special.predictors = list(  # specific year values as predictors:
        list("gdp", 1985, "mean"),
        list("gdp", 1990, "mean"),
    ),
    dependent    = "gdp_growth",
    unit.variable = "country_id",
    time.variable = "year",
    treatment.identifier = 7,        # treated unit's id
    controls.identifier  = c(2, 3, 5, 8, 10),  # control units
    time.predictors.prior = 1970:1990,  # pre-treatment period
    time.optimize.ssr    = 1970:1990,   # period to minimize pre-fit SSR
    time.plot            = 1970:2000,   # full plot range
)

# Estimate:
synth_out <- synth(data.prep.obj=dataprep_out,
                   method="BFGS",    # outer optimization method
                   optimxmethod="BFGS")

# Tables and plots:
synth.tables <- synth.tab(dataprep.res=dataprep_out, synth.res=synth_out)
print(synth.tables$tab.pred)   # predictor balance
print(synth.tables$tab.w)      # weights per control unit
path.plot(synth.res=synth_out, dataprep.res=dataprep_out)  # outcome paths

# Extract weights:
synth_out$solution.w   # optimal control unit weights
synth_out$solution.v   # optimal predictor weights (V matrix diagonal)
```

```python
# Python: pysynth (mirrors R Synth)
from pysynth import Synth as PySynth

# Or use the newer SynthControl package:
# pip install SynthControl
from SynthControl import SynthControlMultiperiod

# Minimal interface (treated = one unit, controls = rest):
sc = PySynth()
sc.fit(
    treatment_unit = 7,
    control_units  = [2, 3, 5, 8, 10],
    outcome_var    = 'gdp_growth',
    predictor_vars = ['gdp', 'population'],
    time_var       = 'year',
    unit_var       = 'country_id',
    pre_periods    = list(range(1970, 1991)),
    post_periods   = list(range(1991, 2001)),
    data           = df,
)
sc.weights_  # control unit weights
sc.treatment_effect_  # ATT for each post-treatment period
```

**Placebo inference (in-space):**

```r
# Apply synthetic control to each control unit (as if each were treated):
store <- matrix(NA, nrow=n_periods, ncol=n_controls+1)
store[, 1] <- gaps.plot(synth_out, dataprep_out)$gap  # treated unit gaps

for (j in control_units) {
    # Re-run synth treating unit j as the "treated" unit
    dp_j <- dataprep(... treatment.identifier=j, controls.identifier=all_except_j ...)
    s_j  <- synth(data.prep.obj=dp_j)
    store[, j_index] <- gaps_j
}

# Gap ratio: pre-RMSPE / post-RMSPE for each unit
pre_fit  <- sqrt(colMeans(store[pre_periods,  ]^2, na.rm=TRUE))
post_gap <- sqrt(colMeans(store[post_periods, ]^2, na.rm=TRUE))

# Rank treated unit's gap ratio among all units:
p_value <- mean(post_gap / pre_fit >= (post_gap / pre_fit)[1])
# If treated unit has largest ratio: p = 1/J (minimum achievable)
```

---

## Tambear Decomposition Summary Table

| Algorithm | Grouping | Expression | Op | Notes |
|-----------|----------|------------|-----|-------|
| FE (within) | ByKey{entity_id} | phi = v | Mean | Group mean = MomentStats(ByKey, order=1) |
| FE (within) | All | Tiled(p+1,p+1) | Add | GramMatrix on demeaned data |
| RE | ByKey{entity_id} | phi = v | Mean | Same as FE, then θ-scale |
| RE variance comps | ByKey{entity_id} | phi = (v-mean)² | Sum | Between/within variance |
| FD | Segmented{entity, lag=1} | v - lag(v) | — | Requires Segmented (todo!) |
| 2SLS Stage 1 | Tiled(p+1,p+1) | DotProduct | Add | GramMatrix on [X, Z] |
| 2SLS Stage 2 | Tiled(p+1,p+1) | DotProduct | Add | GramMatrix on [X_hat, y] |
| DiD (2×2) | ByKey{(treat,post)} | phi = v | Mean | 4-cell means, contrast |
| DiD (regression) | Tiled(p+1,p+1) | DotProduct | Add | GramMatrix on [1, treat, post, interact] |
| RD (LLR) | Masked{abs(X-c)<h} | DotProduct | Add | Weighted GramMatrix in bandwidth |
| Synth control | ByKey{unit_id} | phi = v_k | Mean | Predictor construction; then F05 QP |
| Arellano-Bond | Segmented + Tiled | mixed | mixed | FD + instrument block-diagonal GramMatrix |

**New primitive required**: `Segmented(Prefix{entity})` for First Differences and Arellano-Bond.
Everything else is FE-style demeaning (ByKey + MomentStats) + GramMatrix solves.

---

## Sharing Opportunities

| Family | What F12 Consumes | Notes |
|--------|-------------------|-------|
| F06 | MomentStats(ByKey) = group means for entity demeaning | FE is literally F06 + F10 |
| F10 | GramMatrix solve = all regression stages | Every F12 estimator uses this |
| F07 | Chi-square test = Hausman test, Sargan test | Same distribution, same extraction |
| F05 | Optimization = bandwidth selection (RD), synth weights (SC) | F05 is the inner solver |
| F11 | RE variance components ≈ LME variance components | RE is a special case of LME with block diagonal Omega |

F12 does not need to implement ANY new math — it only needs to compose existing pieces.

---

## Validation Targets

### Fixed Effects — verify tambear demean + OLS against plm

```r
# Generate panel data:
set.seed(42)
N <- 50    # entities
T <- 10    # time periods per entity
n <- N * T

df <- data.frame(
    id   = rep(1:N, each=T),
    time = rep(1:T, times=N),
    x1   = rnorm(n),
    x2   = rnorm(n)
)
# Entity FE (correlated with x for endogeneity to matter):
alpha_i <- rnorm(N, mean=0, sd=2)
df$alpha  <- alpha_i[df$id]
df$y <- 1.5 * df$x1 - 0.8 * df$x2 + df$alpha + rnorm(n, sd=0.5)

library(plm)
pdata <- pdata.frame(df, index=c("id", "time"))
fe_fit <- plm(y ~ x1 + x2, data=pdata, model="within")
re_fit <- plm(y ~ x1 + x2, data=pdata, model="random")

cat("FE coefficients:\n")
print(coef(fe_fit))    # should be ≈ (1.5, -0.8) — unbiased
cat("\nRE coefficients:\n")
print(coef(re_fit))    # should be biased toward OLS (entity effect is correlated)
cat("\nHausman test:\n")
print(phtest(fe_fit, re_fit))  # should reject (p < 0.05)
```

**Expected FE coefficients**: β1 ≈ 1.5, β2 ≈ -0.8 (close to true, unbiased despite correlated FE)
**Expected RE coefficients**: different from FE (biased toward pooled OLS)
**Expected Hausman p-value**: < 0.05 (FE preferred)

```python
import numpy as np
import pandas as pd
from linearmodels import PanelOLS

np.random.seed(42)
N, T = 50, 10
ids   = np.repeat(np.arange(N), T)
times = np.tile(np.arange(T), N)
x1 = np.random.randn(N*T)
x2 = np.random.randn(N*T)
alpha = np.random.randn(N)[ids] * 2
y = 1.5*x1 - 0.8*x2 + alpha + 0.5*np.random.randn(N*T)

df = pd.DataFrame({'id': ids, 'time': times, 'y': y, 'x1': x1, 'x2': x2})
df = df.set_index(['id', 'time'])

fe_result = PanelOLS(df['y'], df[['x1','x2']], entity_effects=True).fit(
    cov_type='clustered', cluster_entity=True)
print(fe_result.params)   # should match R plm to < 1e-8
print(fe_result.std_errors)
```

### 2SLS — verify against ivreg

```r
set.seed(42)
n <- 500
z1 <- rnorm(n)                     # instrument (excluded)
z2 <- rnorm(n)                     # instrument (excluded)
x_exog <- rnorm(n)                 # exogenous regressor
u <- rnorm(n)                      # error
x_endog <- 0.7*z1 + 0.5*z2 + 0.5*u + rnorm(n, sd=0.5)  # endogenous, correlated with u
y <- 2.0*x_endog + 1.0*x_exog + u

df_iv <- data.frame(y=y, x_endog=x_endog, x_exog=x_exog, z1=z1, z2=z2)

library(AER)
iv_fit <- ivreg(y ~ x_endog + x_exog | z1 + z2 + x_exog, data=df_iv)
summary(iv_fit, diagnostics=TRUE)
# Expected: x_endog ≈ 2.0, x_exog ≈ 1.0
# OLS will be biased (x_endog is correlated with error)
ols_fit <- lm(y ~ x_endog + x_exog, data=df_iv)
cat("OLS x_endog:", coef(ols_fit)["x_endog"], "\n")  # biased, will be > 2.0
cat("IV  x_endog:", coef(iv_fit)["x_endog"], "\n")   # should be ≈ 2.0
```

### DiD — verify 2×2 against manual computation

```r
set.seed(42)
n_units <- 200
df_did <- data.frame(
    id      = rep(1:n_units, each=2),
    post    = rep(c(0, 1), n_units),
    treat   = rep(sample(c(0,1), n_units, replace=TRUE), each=2)
)
# True DiD effect = 3.0
df_did$y <- with(df_did, 1 + 2*treat + 1.5*post + 3.0*(treat*post) + rnorm(nrow(df_did)))

# Regression DiD:
lm_did <- lm(y ~ treat + post + treat:post, data=df_did)
cat("DiD estimate:", coef(lm_did)["treat:post"], "\n")  # should be ≈ 3.0

# Manual cell means check:
cell_means <- aggregate(y ~ treat + post, data=df_did, FUN=mean)
print(cell_means)
did_manual <- (cell_means[cell_means$treat==1 & cell_means$post==1, "y"] -
               cell_means[cell_means$treat==1 & cell_means$post==0, "y"]) -
              (cell_means[cell_means$treat==0 & cell_means$post==1, "y"] -
               cell_means[cell_means$treat==0 & cell_means$post==0, "y"])
cat("DiD manual:", did_manual, "\n")
# Regression and manual should match exactly (same formula, same data)
```

### RD — verify against rdrobust

```r
set.seed(42)
n <- 1000
x_run <- rnorm(n, mean=0, sd=2)    # running variable
# True RD effect = 2.0 at cutoff = 0
y_rd <- 1.0 + 0.5*x_run + 2.0*(x_run >= 0) + rnorm(n, sd=1)

library(rdrobust)
rdd_fit <- rdrobust(y=y_rd, x=x_run, c=0)
summary(rdd_fit)
# Expected: τ̂ ≈ 2.0, robust CI covers 2.0
cat("RD estimate (robust):", rdd_fit$coef["Robust"], "\n")
cat("95% CI:", rdd_fit$ci["Robust", ], "\n")
```

---

## Package Version Notes

As of 2026:
- `plm` 2.6+ handles unbalanced panels and instruments correctly
- `fixest` 0.12+ has Sun-Abraham via `sunab()` and stable `feols` IV
- `rdrobust` 2.1+ uses CCT bandwidth (older versions used IK bandwidth — different numerics)
- `did` 2.1+ (Callaway-Sant'Anna) requires `gname=0` for never-treated units
- `AER` 1.2-10+ has ivreg with corrected heteroscedasticity diagnostics
- `linearmodels` 4.29+ has stable PanelOLS with clustered SE
- `pyfixest` 0.18+ has did2s Gardner estimator for staggered DiD

**R is the gold standard oracle for all of F12.** Python implementations (linearmodels,
pyfixest) are secondary verification. When they disagree with R: R is right.
Exception: rdrobust Python implementation directly calls same C code as R — results identical.

---

## Appendix A: Exact Variance Component Formulas (RE Estimators)

These are the formulas plm implements internally. Source: Baltagi (2013) "Econometric
Analysis of Panel Data," 5th ed., Chapter 2; and plm source code (ercomp.R).

### Setup

One-way error components model (balanced panel, N entities, T periods, n = NT):
```
y_it = x_it'β + α_i + ε_it

α_i ~ iid(0, σ²_α)   (individual effects — the "random" part)
ε_it ~ iid(0, σ²_ε)  (idiosyncratic errors)

Composite error: u_it = α_i + ε_it
Var(u_it) = σ²_α + σ²_ε    (same entity/time)
Cov(u_it, u_is) = σ²_α     (same entity, different time; s ≠ t)
Cov(u_it, u_js) = 0         (different entities)
```

All three estimators below recover (σ̂²_ε, σ̂²_α) using quadratic forms of residuals from
preliminary regressions. The difference is which preliminary regression is used.

### The Within (FE) and Between Transformations

```
Within-transformation residuals ê_it^W:
  Run FE (within) regression on demeaned data
  ê_it^W = (y_it - ȳ_i) - (x_it - x̄_i)'β̂_FE

Between-transformation residuals ê_i^B:
  Collapse to entity means: ȳ_i, x̄_i
  Run OLS on entity-mean data: ȳ_i = x̄_i'β̂_B + error_i
  ê_i^B = ȳ_i - x̄_i'β̂_B

Pooled OLS residuals ê_it^P:
  Run pooled OLS on raw data: y_it = x_it'β̂_OLS + error
  ê_it^P = y_it - x_it'β̂_OLS
```

### Estimator 1: Swamy-Arora (swar) — default in plm

Uses within regression for σ²_ε and between regression for σ²_α:

```
σ̂²_ε = (Σ_i Σ_t ê_it^W²) / (n - N - K)
       = RSS_within / (N(T-1) - K)

         ← denominator: total obs - entity dummies - regressors
            (plm dfcor=2 default: subtract K estimated parameters)

σ̂²_α = [(Σ_i ê_i^B²) / (N - K - 1)] - σ̂²_ε / T
       = [RSS_between / (N - K - 1)] - σ̂²_ε / T

         ← between regression has N observations, K+1 parameters
            the σ̂²_ε/T correction removes idiosyncratic contamination
```

Reference: Swamy and Arora (1972, Econometrica 40:261-75), Baltagi & Chang (1994, J. Econometrics 62:67-89) for unbalanced version.

Unbalanced panel: T is replaced by T* = (1/N)(n - Σ_i T_i²/n) = harmonic-mean-like correction.

### Estimator 2: Wallace-Hussain (walhus)

Uses pooled OLS residuals for BOTH variance components:

```
σ̂²_ε = (Σ_i Σ_t ê_it^P² - (1/T) Σ_i (Σ_t ê_it^P)²) / (n - N - K)
       = [RSS_pooled - T·RSS_between_of_pooled] / (n - N - K)

σ̂²_α = [(1/T) Σ_i (Σ_t ê_it^P)² / (N - K - 1)] - σ̂²_ε / T
```

The OLS residuals replace the FE/between residuals. Less efficient than Swamy-Arora
(uses biased preliminary OLS residuals) but simpler — no within regression needed first.

Reference: Wallace and Hussain (1969, Econometrica 37:55-70).

### Estimator 3: Amemiya (amemiya)

Uses within regression residuals for BOTH variance components:

```
σ̂²_ε = (Σ_i Σ_t ê_it^W²) / (n - N - K)
       = RSS_within / (N(T-1) - K)
       [Same as Swamy-Arora for σ̂²_ε]

σ̂²_α = [(Σ_i (Σ_t ê_it^W)²/T) / (N - K)] - σ̂²_ε / T
       [Uses sum of within residuals per entity, squared and averaged]
```

The second formula uses within residuals collapsed to entity means (not a separate
between regression). Degrees of freedom: N - K (not N - K - 1 as in Swamy-Arora).

Reference: Amemiya (1971, J. Econometrics 1:61-86).

### The θ Transformation Parameter

Given σ̂²_ε and σ̂²_α, the quasi-demeaning factor θ is:

```
θ = 1 - sqrt(σ̂²_ε / (T·σ̂²_α + σ̂²_ε))

For unbalanced panels: θ_i = 1 - sqrt(σ̂²_ε / (T_i·σ̂²_α + σ̂²_ε))   [per-entity θ]
```

The RE quasi-demeaned variables:
```
y*_it = y_it - θ·ȳ_i
x*_it = x_it - θ·x̄_i
```

Then β̂_RE = OLS(y*, X*) = GramMatrix(X*) solve on quasi-demeaned data.

Interpretation:
- θ = 0  →  RE = pooled OLS (no entity structure)
- θ = 1  →  RE = FE (full within demeaning)
- 0 < θ < 1  →  RE blends between and within variation

### Check values (balanced, N=50, T=10, K=2)

Using the validation dataset from Section 1 (seed=42, true β=(1.5, -0.8)):

Expected ercomp() output from plm:
```r
ercomp(re_fit)
# $sigma2: idios ≈ 0.25, id ≈ 3.9-4.0  (σ²_ε ≈ 0.25, σ²_α ≈ 4.0)
# $theta: ≈ 0.86   (because T·σ²_α >> σ²_ε → θ near 1 → RE close to FE)
```

---

## Appendix B: Breusch-Pagan LM Test for Random Effects

Tests H₀: σ²_α = 0 (no random effects; pooled OLS is correct) vs H₁: σ²_α > 0.

### Formula (Breusch & Pagan 1980, balanced panel)

```
LM = [nT / (2(T-1))] · [(Σ_i (Σ_t ê_it)²) / (Σ_i Σ_t ê_it²) - 1]²

where ê_it = pooled OLS residuals (y_it - x_it'β̂_OLS)
      n = number of entities
      T = number of time periods (balanced)

Under H₀: LM ~ χ²(1)
```

Equivalently:
```
A = Σ_i (Σ_t ê_it)²     ← sum of squared entity-sum-of-residuals
B = Σ_i Σ_t ê_it²       ← total sum of squared residuals (= RSS_OLS)

LM = [nT / (2(T-1))] · (A/B - 1)²
```

The ratio A/B is 1 under H₀ (entity sums of residuals average to zero when there are
no entity effects). With σ²_α > 0, entity sums are correlated and A/B > 1.

### Unbalanced Panel: Baltagi-Li (1990) Modification

```
LM_BL = [Σ_i T_i²/(2(T_i - 1)) · (A_i/B - 1)²]  — per-entity weighted version

where A_i = (Σ_t ê_it)²     (entity i squared sum)
      B   = Σ_i Σ_t ê_it²   (total RSS)

Reported by Stata's xttest0 (after xtreg, re).
Under H₀: LM_BL ~ 0.5·δ₀ + 0.5·χ²(1)  [mixture, one-sided]
```

### R Implementation

```r
# plmtest(): one-sided LM test for random effects
plmtest(pooling_model, type="bp", effect="individual")
# type="bp"    = Breusch-Pagan (default)
# type="honda" = Honda (1985) — square root of BP, normal test
# type="kw"    = King-Wu (1997) — robust to heteroscedasticity
# effect="individual", "time", "twoways"

# Or: construct manually:
pool_fit <- plm(y ~ x1 + x2, data=pdata, model="pooling")
e <- residuals(pool_fit)
N <- length(unique(pdata$id))
T <- length(unique(pdata$time))
e_mat <- matrix(e, nrow=T, ncol=N)     # T rows, N columns
A <- sum(colSums(e_mat)^2)             # Σ_i (Σ_t ê_it)²
B <- sum(e^2)                          # total RSS
LM_stat <- (N*T / (2*(T-1))) * (A/B - 1)^2
p_value  <- pchisq(LM_stat, df=1, lower.tail=FALSE)
```

### Python Implementation

```python
import numpy as np
from scipy.stats import chi2
from linearmodels import PooledOLS

pool_result = PooledOLS(df['y'], df[['x1','x2']]).fit()
e = pool_result.resids.values.reshape(N, T)  # N × T matrix
A = np.sum(np.sum(e, axis=1)**2)   # Σ_i (Σ_t ê_it)²
B = np.sum(e**2)                   # total RSS
LM_stat = (N*T / (2*(T-1))) * (A/B - 1)**2
p_value  = chi2.sf(LM_stat, df=1)
```

---

## Appendix C: Hausman Test — Exact Formula and Numerical Notes

### Formula

```
d = β̂_FE - β̂_RE       (k-vector of coefficient differences)
                         [only time-varying regressors; drop time-constant cols]

V = Var(β̂_FE) - Var(β̂_RE)   (k × k matrix difference)
    = (X̃'X̃)^{-1} - (X*'X*)^{-1}  [approximate; exact under H₀ assumptions]

H = d' V^{-1} d    →  χ²(k)  under H₀
```

Where k = number of time-varying regressors (not counting entity-constant regressors,
because FE cannot identify them — they're absorbed by entity dummies).

### Numerical Trap: Non-Positive-Definite V

Under finite samples, V = Var(FE) - Var(RE) may not be positive definite even though
it should be asymptotically. When V is not PD:

1. plm's `phtest()` uses the Moore-Penrose pseudoinverse V† (via `MASS::ginv`)
2. The degrees of freedom = rank(V†), not k
3. Stata does the same (documented in `hausman` help)

**Implication for Tambear**: use pinv (pseudo-inverse) not regular solve for the Hausman
quadratic form. The chi-square df = rank(V).

### Robust Hausman (Mundlak approach)

When heteroscedasticity or autocorrelation is present, V = Var(FE) - Var(RE) is
unreliable. Use the Mundlak augmented regression instead (Appendix D).

---

## Appendix D: Mundlak Correction (Correlated Random Effects)

### Model

```
Standard RE: y_it = x_it'β + α_i + ε_it,  α_i ~ iid(0, σ²_α)

Mundlak decomposition: α_i = x̄_i'θ + ν_i
where x̄_i = (1/T_i) Σ_t x_it   (entity-mean of each time-varying covariate)
      ν_i ~ iid(0, σ²_ν), Cov(x_it, ν_i) = 0

Substituting:
  y_it = x_it'β + x̄_i'θ + ν_i + ε_it

This is a random effects model on augmented regressors [x_it, x̄_i].
```

### Implementation

```r
# Step 1: Compute entity means of each time-varying covariate
pdata$x1_mean <- ave(pdata$x1, pdata$id, FUN=mean)
pdata$x2_mean <- ave(pdata$x2, pdata$id, FUN=mean)

# Step 2: Run RE on augmented model (original + entity means)
mundlak_fit <- plm(y ~ x1 + x2 + x1_mean + x2_mean,
                   data=pdata, model="random")
summary(mundlak_fit)

# Step 3: Test H₀: θ = 0 (RE is consistent)
#   Joint significance of x1_mean, x2_mean coefficients
library(car)
linearHypothesis(mundlak_fit, c("x1_mean = 0", "x2_mean = 0"),
                 vcov.=vcovHC(mundlak_fit))
# Chi-square test, df=2 (number of means added)
# If rejected: θ ≠ 0 → entity effects correlated with x → use FE
```

```python
import pandas as pd
# Entity means:
df['x1_mean'] = df.groupby('id')['x1'].transform('mean')
df['x2_mean'] = df.groupby('id')['x2'].transform('mean')

from linearmodels import RandomEffects
mundlak_result = RandomEffects(df['y'], df[['x1','x2','x1_mean','x2_mean']]).fit()

# Test θ = 0: chi-square on x1_mean, x2_mean coefficients
from scipy.stats import chi2
b_means = mundlak_result.params[['x1_mean','x2_mean']]
V_means  = mundlak_result.cov.loc[['x1_mean','x2_mean'], ['x1_mean','x2_mean']]
chi2_stat = float(b_means @ np.linalg.solve(V_means, b_means))
p_value   = chi2.sf(chi2_stat, df=2)
print(f"Mundlak test: χ²={chi2_stat:.4f}, p={p_value:.4f}")
```

### Key Properties

1. β̂ from the Mundlak RE regression **equals β̂_FE exactly** (for balanced panels).
   The Mundlak RE with entity means is numerically identical to the FE estimator.

2. θ̂ captures the bias in pooled OLS or naive RE: how much of x̄_i is driving α_i.

3. The test for θ = 0 is equivalent to the Hausman test, but ROBUST to heteroscedasticity
   and clustering when sandwich SEs are used (unlike classical Hausman which assumes
   spherical errors under H₀).

4. This is the recommended alternative to the Hausman test when:
   - Errors are heteroscedastic or clustered
   - The matrix Var(FE) - Var(RE) is not positive definite
   - Standard errors are panel-robust (clustered)

---

## Appendix E: Wooldridge Test for Serial Correlation

Tests H₀: no serial correlation in idiosyncratic errors ε_it (AR(1) coefficient = 0).

### Algorithm (Wooldridge 2002, Section 10.5)

```
Step 1: Run first-difference regression on FD data:
        Δy_it = Δx_it'β̂_FD + Δê_it
        Retrieve FD residuals: Δê_it = Δy_it - Δx_it'β̂_FD

Step 2: Regress FD residuals on their one-period lag:
        Δê_it = ρ · Δê_{i,t-1} + η_it     [AR(1) in FD residuals]
        with cluster-robust SE (by entity i)

Step 3: Under H₀ (ε_it are serially uncorrelated):
        Cov(Δê_it, Δê_{i,t-1}) = -0.5 · Var(Δê_it)
        → H₀: ρ = -0.5  [NOT ρ = 0!]

Step 4: F-test for H₀: ρ = -0.5 with cluster-robust variance
        F = [(ρ̂ - (-0.5)) / SE_cluster(ρ̂)]²   ~ F(1, N-1)
        → equivalently: t-statistic for H₀: ρ = -0.5
```

The key insight: if ε_it ~ iid, then Δε_it = ε_it - ε_{i,t-1} has Cov(Δε_it, Δε_{i,t-1})
= -σ²_ε, while Var(Δε_it) = 2σ²_ε, so Corr = -0.5. Under serial correlation, Corr ≠ -0.5.

### R Implementation

```r
# plm: pwfdtest() (Wooldridge first-difference test)
fd_model <- plm(y ~ x1 + x2, data=pdata, model="fd")
pwfdtest(fd_model)
# Reports: F-statistic and p-value for H₀: ρ_fd = -0.5
# Implemented with cluster-robust SE (Arellano 1987 sandwich)

# Alternative: pwartest() (Wooldridge RE-based autocorrelation test)
# Uses RE model residuals instead of FD residuals
re_model  <- plm(y ~ x1 + x2, data=pdata, model="random")
pwartest(re_model)

# Manual implementation:
fd_resid <- residuals(fd_model)
# Note: fd_model loses T=1 observations; residuals indexed by (i, t), t >= 2
# Lag the fd residuals within each entity and test ρ = -0.5
```

```python
# No direct equivalent in linearmodels; compute manually:
import numpy as np
import pandas as pd
from linearmodels import FirstDifferenceOLS

fd_result = FirstDifferenceOLS(df['y'], df[['x1','x2']]).fit()
fd_resid  = fd_result.resids  # indexed by (entity, time), first period dropped

# Create lagged FD residuals within entity:
df_resid = fd_resid.reset_index()
df_resid.columns = ['id','time','fd_resid']
df_resid['fd_resid_lag'] = df_resid.groupby('id')['fd_resid'].shift(1)
df_resid = df_resid.dropna()

# Regress fd_resid on fd_resid_lag (H₀: coefficient = -0.5):
import statsmodels.api as sm
X = sm.add_constant(df_resid['fd_resid_lag'])
y_r = df_resid['fd_resid']
ols = sm.OLS(y_r, X).fit(cov_type='cluster', cov_kwds={'groups': df_resid['id']})

rho_hat = ols.params['fd_resid_lag']
se_rho  = ols.bse['fd_resid_lag']
t_stat  = (rho_hat - (-0.5)) / se_rho  # test ρ = -0.5, NOT ρ = 0
F_stat  = t_stat**2
from scipy.stats import f as f_dist
p_value = f_dist.sf(F_stat, dfn=1, dfd=len(df_resid['id'].unique())-1)
print(f"Wooldridge test: F={F_stat:.4f}, p={p_value:.4f}")
# Reject H₀ → serial correlation present → use cluster-robust SE or GLS
```
