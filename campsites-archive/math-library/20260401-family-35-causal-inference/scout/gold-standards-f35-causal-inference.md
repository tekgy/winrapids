# F35 Causal Inference — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 35 (Causal Inference).
This family answers the hardest question in data science: not "what predicts Y?" but
"what *causes* Y to change?" The key structural insight upfront: nearly every causal
estimator is a regression variant with one of three modifications — reweighting observations,
residualizing to remove confounders, or matching to equate covariate distributions.

**All three modifications decompose cleanly onto tambear primitives:**
- Reweighting = `accumulate(All, WeightedMean{w}, Add)` — F10 GramMatrix with obs-level weights
- Residualizing = two sequential GramMatrix solves (already F12 DML pattern)
- Matching = DistancePairs + row argmin (F01/F20)

The navigator's creation note is exactly right: F35 is causal IV/2SLS, DiD, and RD with
matching and reweighting wrappers. Every estimator here builds on infrastructure already
established in F10, F12, and F01. F35 adds no new primitives — it adds new *compositions*
and new *validity conditions* on when those compositions identify causal effects.

**Financial relevance**: market microstructure research routinely asks causal questions —
does a regime change (news event, policy, tick size change) causally alter liquidity?
Do large trades *cause* price impact or merely predict it? DML is particularly valuable
for separating signal effects from confounded correlations in factor research.

---

## Index

1. Potential Outcomes Framework (ATE, ATT, ATU)
2. Propensity Score Methods (IPW, Matching, AIPW)
3. Double/Debiased Machine Learning (DML)
4. Causal Forests & Heterogeneous Treatment Effects (CATE)
5. Synthetic Control Method
6. Mediation Analysis (direct/indirect effects)
7. Directed Acyclic Graphs (DAGs, d-separation)
8. Regression Discontinuity — cross-reference F12
9. Difference-in-Differences — cross-reference F12
10. Tambear Decomposition Summary

---

## Critical Package Landscape

### R Causal Packages

```r
# Potential outcomes / IPW:
install.packages("causalweight")   # Bodory & Huber — IPW, AIPW, DML
install.packages("WeightIt")       # Greifer — universal weighting interface
install.packages("cobalt")         # Greifer — covariate balance tables/plots

# Matching:
install.packages("MatchIt")        # Ho, Imai, King, Stuart — the standard
install.packages("Matching")       # Sekhon — genetic matching, exact SE
install.packages("optmatch")       # optimal full matching (network flow)

# Double ML:
install.packages("DoubleML")       # Bach et al. — R implementation of doubleml
install.packages("hdm")            # Belloni — LASSO-based causal inference

# Causal forests:
install.packages("grf")            # Athey/Wager/Tibshirani — generalized random forests

# IV/2SLS (see F12, but listed here for completeness):
install.packages("AER")            # ivreg() — standard IV
install.packages("fixest")         # feols() with instruments

# Mediation:
install.packages("mediation")      # Imai, Keele, Tingley — the standard
install.packages("lavaan")         # structural equation mediation

# DiD (staggered, see F12 for detail):
install.packages("did")            # Callaway-Sant'Anna
install.packages("staggered")

# RD (see F12 for detail):
install.packages("rdrobust")

# Synthetic control (see Section 5):
install.packages("Synth")          # Abadie, Diamond, Hainmueller — the original
install.packages("SCtools")        # inference for synthetic control
install.packages("tidysynth")      # tidy interface to Synth

# DAGs:
install.packages("dagitty")        # Textor et al. — d-separation, adjustment sets
install.packages("ggdag")          # ggplot2 interface to dagitty
```

### Python Causal Packages

```python
# Core causal ML:
# pip install econml
from econml.dml import LinearDML, CausalForestDML, NonParamDML
from econml.iv.dml import DMLIV
from econml.causal_forest import CausalForest
from econml.metalearners import TLearner, SLearner, XLearner

# Double ML (separate, academic-standard):
# pip install doubleml
from doubleml import DoubleMLPLR, DoubleMLIRM, DoubleMLPLIV, DoubleMLIIVM

# Causal ML / IPW:
# pip install causalml
from causalml.inference.meta import LRSRegressor, XGBTRegressor
from causalml.propensity import ElasticNetPropensityModel
from causalml.match import NearestNeighborMatch

# Matching:
# pip install pymatch   (MatchIt-style for Python)
# pip install causalinference

# Synthetic control:
# pip install pysynth
import pysynth

# DAGs:
# pip install causalnex   (Bayesian network + DAG structure learning)
import causalnex
# OR use dagitty via rpy2 bridge

# Potential outcomes framework data generators:
# pip install causaldm
from causaldm.learners import IPW, AIPW, DML
```

---

## Section 1: Potential Outcomes Framework

### The Rubin Causal Model

For binary treatment D ∈ {0, 1} and outcome Y:
- Y(1) = potential outcome under treatment
- Y(0) = potential outcome under control
- Observed: Y = D·Y(1) + (1-D)·Y(0)
- Fundamental problem: only ONE potential outcome observed per unit

**Estimands:**
```
ATE  = E[Y(1) - Y(0)]                    # Average Treatment Effect (all units)
ATT  = E[Y(1) - Y(0) | D=1]              # ATE on the Treated
ATU  = E[Y(1) - Y(0) | D=0]              # ATE on the Untreated (control)
CATE = E[Y(1) - Y(0) | X=x]             # Conditional ATE at covariates x
```

**Identification assumptions:**
1. SUTVA (Stable Unit Treatment Value): no interference between units
2. Unconfoundedness / ignorability: Y(0), Y(1) ⊥ D | X (no unmeasured confounders)
3. Overlap / positivity: 0 < P(D=1|X) < 1 for all X (everyone has positive treatment prob)

**Financial context**: ATE = average price impact of a large trade event.
ATT = impact specifically on trades that *were* large (conditioning on treatment).

### R: causaldm / baseline comparison

```r
# Simplest estimator under unconfoundedness: regression adjustment
library(lm)

# Step 1: fit outcome model separately for treated and control
m1 <- lm(Y ~ X1 + X2 + X3, data=subset(df, D==1))
m0 <- lm(Y ~ X1 + X2 + X3, data=subset(df, D==0))

# Step 2: predict both potential outcomes for all units
df$y1hat <- predict(m1, newdata=df)
df$y0hat <- predict(m0, newdata=df)

# Step 3: estimate ATE, ATT, ATU
ATE <- mean(df$y1hat - df$y0hat)
ATT <- mean(df$y1hat[df$D==1] - df$y0hat[df$D==1])
ATU <- mean(df$y1hat[df$D==0] - df$y0hat[df$D==0])

# This is the "T-learner" / regression adjustment estimator.
# Its validity depends entirely on the outcome model being correctly specified.
```

### Python: econml metalearners

```python
from econml.metalearners import TLearner, SLearner, XLearner
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# T-learner: separate models for treated/control
tlearner = TLearner(models=GradientBoostingRegressor())
tlearner.fit(Y, T, X=X)
cate_t = tlearner.effect(X)          # CATE for each unit

# S-learner: single model with treatment as feature
slearner = SLearner(overall_model=GradientBoostingRegressor())
slearner.fit(Y, T, X=X)
cate_s = slearner.effect(X)

# X-learner: better for imbalanced treatment (rare events)
xlearner = XLearner(models=GradientBoostingRegressor(),
                    propensity_model=LogisticRegression())
xlearner.fit(Y, T, X=X)
cate_x = xlearner.effect(X)

# ATE from CATE:
ATE_t = np.mean(cate_t)
ATT_t = np.mean(cate_t[T == 1])
ATU_t = np.mean(cate_t[T == 0])
```

**Validation targets (synthetic data):**
```python
# Generate: Y = 2*D + 3*X + noise, so true ATE=2
np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
ps = 1 / (1 + np.exp(-X[:, 0]))          # propensity depends on X[:,0]
D = np.random.binomial(1, ps)
Y = 2*D + 3*X[:, 0] + np.random.randn(n)

# Expected: ATE ≈ 2.0 (within ~0.1 for n=1000)
# T-learner with linear models should recover ATE=2.0 exactly (model is correctly specified)
# T-learner with tree models: ≈ 2.0 ± 0.15
```

**Trap: S-learner can under-estimate CATE when treatment effect is small relative to
main effects.** The single model may put treatment variable at low importance and
shrink toward zero CATE. Use T-learner or X-learner when treatment effect heterogeneity
is the primary question.

---

## Section 2: Propensity Score Methods

### Propensity Score

The propensity score e(X) = P(D=1 | X) is a balancing score: conditional on e(X),
the treatment D is independent of X. This reduces the confounding problem from
"condition on all X" to "condition on a scalar."

**Key theorem (Rosenbaum & Rubin 1983)**: If unconfoundedness holds given X,
it also holds given e(X). The propensity score is sufficient.

### 2.1 Propensity Score Estimation

```r
# R: logistic regression (baseline)
ps_model <- glm(D ~ X1 + X2 + X3 + X1:X2, data=df, family=binomial())
df$pscore <- fitted(ps_model)

# Diagnostics — propensity score overlap:
hist(df$pscore[df$D==1], col=rgb(1,0,0,0.5), main="Propensity Score Overlap")
hist(df$pscore[df$D==0], col=rgb(0,0,1,0.5), add=TRUE)
# Overlap region should be substantial. If not: SUTVA violation or wrong model.

# Trim extreme propensity scores (Section 2.2 trap):
df <- df[df$pscore >= 0.05 & df$pscore <= 0.95, ]
```

```python
# Python: logistic regression propensity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Logistic (simple, interpretable):
lr = LogisticRegression(max_iter=500)
lr.fit(X, D)
pscore_lr = lr.predict_proba(X)[:, 1]

# Gradient boosted (flexible, non-parametric):
gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(X, D)
pscore_gb = gb.predict_proba(X)[:, 1]

# Tambear decomposition:
# Propensity score = logistic regression = F10/GLM path
# accumulate(All, GramMatrix{phi=sigmoid_features}, Add) + Newton-Raphson
```

### 2.2 Inverse Probability Weighting (IPW)

IPW reweights each observation by the inverse of its probability of receiving
its actual treatment. This creates a "pseudo-population" where treatment is
independent of X.

```
ATE weights:    w_i = D_i / e(X_i) + (1-D_i) / (1 - e(X_i))

ATT weights:    w_i = 1{D_i=1}  +  1{D_i=0} * e(X_i) / (1 - e(X_i))
                [treated get weight 1; control weighted up to match treated distribution]
```

**ATE estimator:**
```
ATE_IPW = (1/n) Σ_i [D_i * Y_i / e(X_i)]  -  (1/n) Σ_i [(1-D_i) * Y_i / (1 - e(X_i))]
```

This is exactly: `accumulate(Treated, WeightedMean{w=1/e(X)}, Add)` minus
`accumulate(Control, WeightedMean{w=1/(1-e(X))}, Add)`.

```r
# R: IPW using WeightIt
library(WeightIt)
library(cobalt)

# Estimate weights for ATE:
w_out <- weightit(D ~ X1 + X2 + X3,
                  data   = df,
                  method = "ps",       # propensity score IPW
                  estimand = "ATE")    # or "ATT", "ATU"

# Check balance — the KEY diagnostic for IPW:
bal.tab(w_out, stats=c("m","v"), thresholds=c(m=0.1))
# Standardized mean differences should be < 0.1 after weighting

# Visualize balance:
love.plot(w_out, abs=TRUE, thresholds=c(m=0.1))

# Estimate ATE with survey-weighted regression:
library(survey)
design <- svydesign(ids=~1, weights=~weights(w_out), data=df)
model  <- svyglm(Y ~ D, design=design)
summary(model)  # coefficient on D = ATE_IPW
```

```python
# Python: IPW from scratch (transparent)
import numpy as np
from sklearn.linear_model import LogisticRegression

def ipw_ate(Y, D, X, trim=(0.05, 0.95)):
    """IPW ATE estimator with propensity trimming."""
    lr = LogisticRegression(max_iter=500)
    lr.fit(X, D)
    ps = lr.predict_proba(X)[:, 1]

    # Trim extreme propensity scores:
    mask = (ps >= trim[0]) & (ps <= trim[1])
    Y, D, ps = Y[mask], D[mask], ps[mask]

    # Hajek (normalized) IPW — more stable than Horvitz-Thompson:
    w1 = D / ps
    w0 = (1 - D) / (1 - ps)
    ate = np.sum(w1 * Y) / np.sum(w1) - np.sum(w0 * Y) / np.sum(w0)
    return ate

# Tambear decomposition:
# ate = accumulate(All, WeightedMean{w=D/ps}, Add)
#     - accumulate(All, WeightedMean{w=(1-D)/(1-ps)}, Add)
# Both terms = F10 GramMatrix with observation-level weights
```

**Validation targets:**
```python
# True ATE = 2.0 in the synthetic data above
ate_ipw = ipw_ate(Y, D, X)
# Expected: ATE_IPW ≈ 2.0 ± 0.15 for n=1000
# IPW and regression adjustment should agree when both models correct
```

**CRITICAL TRAP: Extreme Propensity Scores**
When some e(X_i) ≈ 0 or e(X_i) ≈ 1, the IPW weights blow up.
A single observation with e(X) = 0.001 gets weight 1000 — dominating the estimate.

```r
# Stabilized IPW weights (SIPWs) — always prefer over raw IPW:
# w_i = e(X_i) / e_mean for treated; (1-e(X_i)) / (1-e_mean) for control
# Stabilized weights are bounded, not unbounded.
w_out_stable <- weightit(D ~ X1 + X2,
                         data     = df,
                         method   = "ps",
                         estimand = "ATE",
                         stabilize = TRUE)

# Also: trim at 1st/99th percentile of weight distribution:
# Any weight > 10 * mean weight is suspicious.
summary(w_out)  # shows weight summary — look for max/mean ratio
```

### 2.3 Propensity Score Matching

Matching creates a control group that "looks like" the treated group by finding
close-propensity-score neighbors in the control pool.

```r
# R: MatchIt — the standard
library(MatchIt)

# 1-to-1 nearest neighbor matching on propensity score:
m_out <- matchit(D ~ X1 + X2 + X3,
                 data   = df,
                 method = "nearest",    # nearest neighbor
                 ratio  = 1,            # 1:1 matching
                 replace = FALSE)       # WITHOUT replacement (default — see trap)

# Balance check after matching:
summary(m_out)           # standardized mean differences before/after
plot(summary(m_out))     # love plot

# Extract matched dataset:
m_data <- match.data(m_out)

# Estimate ATT on matched data:
ate_match <- lm(Y ~ D + X1 + X2 + X3, data=m_data,
                weights=weights)    # MatchIt provides weights
summary(ate_match)
```

**CRITICAL TRAP: MatchIt default is nearest-neighbor WITHOUT replacement.**
Without replacement = greedy matching that depends on row order. The first treated
unit gets its ideal match; later units may get bad matches.

```r
# Preferred: matching WITH replacement or optimal matching:
m_with_replace <- matchit(D ~ X1 + X2 + X3,
                          data    = df,
                          method  = "nearest",
                          replace = TRUE)   # with replacement — better matches

# Even better: optimal full matching (minimizes total distance globally):
m_optimal <- matchit(D ~ X1 + X2 + X3,
                     data   = df,
                     method = "full")   # full optimal matching

# Or: genetic matching (Sekhon) — iteratively optimizes balance:
library(Matching)
m_gen <- matchit(D ~ X1 + X2 + X3,
                 data   = df,
                 method = "genetic")
```

**Tambear decomposition for matching:**
```
Step 1: e(X) estimation = logistic regression = F10/GLM
Step 2: DistancePairs(treated_pscore, control_pscore) = F01
        distance_matrix[i,j] = |e(X_treated_i) - e(X_control_j)|
Step 3: row argmin = F20 (reduction over DistancePairs)
        match[i] = argmin_j distance_matrix[i,j]
Step 4: ATT = mean(Y_treated - Y_matched_control)
             = accumulate(ByMatch, Subtract + Mean, Add)

For Mahalanobis matching:
Step 2: distance_matrix[i,j] = sqrt((X_i - X_j)' Σ^{-1} (X_i - X_j))
        = GramMatrix kernel evaluation = F10
```

### 2.4 Doubly Robust Estimator (AIPW)

The Augmented IPW (AIPW) estimator combines regression adjustment with IPW.
It is **doubly robust**: consistent if EITHER the outcome model OR the propensity model
is correctly specified (but not necessarily both).

**AIPW formula:**
```
ATE_AIPW = (1/n) Σ_i [m1(X_i) - m0(X_i)
              + D_i(Y_i - m1(X_i)) / e(X_i)
              - (1-D_i)(Y_i - m0(X_i)) / (1 - e(X_i))]

where m1(X) = E[Y | D=1, X], m0(X) = E[Y | D=0, X]  (outcome models)
      e(X) = P(D=1 | X)                                (propensity model)
```

The first term is regression adjustment. The second/third terms are "augmentation" —
they correct for misspecification in the outcome model using IPW residuals.

```r
# R: AIPW via causalweight package
library(causalweight)

# AIPW ATE:
result <- treatweight(y=df$Y, d=df$D, x=cbind(df$X1, df$X2, df$X3),
                      trim=0.05,      # trim propensity scores < 0.05 or > 0.95
                      ATET=FALSE,     # ATE (use TRUE for ATT)
                      boot=100)       # bootstrap standard errors
result$effect    # ATE estimate
result$se        # standard error
result$pval      # p-value
```

```python
# Python: AIPW from scratch
def aipw_ate(Y, D, X, trim=(0.05, 0.95), n_folds=5):
    """
    Doubly robust AIPW with cross-fitting.
    Cross-fitting uses different folds for model fitting vs. plug-in
    to avoid over-fitting bias in the augmentation term.
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import KFold
    import numpy as np

    n = len(Y)
    influence = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        # Propensity model:
        ps_m = LogisticRegression(max_iter=500).fit(X_tr, D_tr)
        ps = ps_m.predict_proba(X_te)[:, 1]
        ps = np.clip(ps, trim[0], trim[1])

        # Outcome models:
        m1_m = LinearRegression().fit(X_tr[D_tr==1], Y_tr[D_tr==1])
        m0_m = LinearRegression().fit(X_tr[D_tr==0], Y_tr[D_tr==0])
        m1 = m1_m.predict(X_te)
        m0 = m0_m.predict(X_te)

        # AIPW influence function:
        influence[test_idx] = (m1 - m0
                               + D_te * (Y_te - m1) / ps
                               - (1 - D_te) * (Y_te - m0) / (1 - ps))

    ate = np.mean(influence)
    se  = np.std(influence) / np.sqrt(n)
    return ate, se

# Tambear decomposition:
# AIPW = F10 (two outcome regressions + propensity) + weighted correction term
# The correction = accumulate(All, WeightedResidual{w=D/ps}, Add)
#               - accumulate(All, WeightedResidual{w=(1-D)/(1-ps)}, Add)
```

**Validation targets:**
```python
# True ATE = 2.0 in synthetic data
ate_aipw, se_aipw = aipw_ate(Y, D, X)
# Expected: ATE_AIPW ≈ 2.0 ± 0.10 (tighter than pure IPW)
# 95% CI should cover 2.0: (ate_aipw - 1.96*se_aipw, ate_aipw + 1.96*se_aipw)

# Double robustness test:
# 1. Misspecify propensity (wrong features): AIPW still consistent IF outcome model correct
# 2. Misspecify outcome model (wrong features): AIPW still consistent IF propensity correct
# 3. Misspecify both: AIPW is biased — no free lunch
```

**AIPW semiparametric efficiency note**: AIPW achieves the semiparametric efficiency bound —
no regular estimator can have lower asymptotic variance under the nonparametric model.
This is why AIPW is the default recommendation when you don't know which model is correct.

---

## Section 3: Double/Debiased Machine Learning (DML)

### What DML Does

DML (Chernozhukov et al. 2018) removes the confounding effect of controls W on both
the treatment D and the outcome Y, then estimates the causal effect from the residuals.

**Partially Linear Model:**
```
Y = θ·D + g(W) + ε,   E[ε | D, W] = 0
D = m(W) + v,          E[v | W] = 0

where θ = causal effect of D on Y
      g(W) = nonparametric confounding function
      m(W) = nonparametric propensity function
```

**DML estimator (two residualization steps + IV moment):**
```
Step 1: Ỹ = Y - Ê[Y|W]   (residualize Y on controls)
Step 2: D̃ = D - Ê[D|W]   (residualize D on controls)
Step 3: θ̂ = (D̃'Ỹ) / (D̃'D̃)   (IV-style moment condition)
```

Step 3 is a single-variable IV regression of Ỹ on D̃. With continuous Y and binary D,
this is also the FWL theorem (Frisch-Waugh-Lovell): regress Ỹ on D̃.

**This IS three GramMatrix operations:**
- Step 1: GramMatrix solve for E[Y|W] (F10)
- Step 2: GramMatrix solve for E[D|W] (F10)
- Step 3: IV moment = D̃'Ỹ / D̃'D̃ (scalar — trivial accumulate)

The ML flexibility comes from using flexible models (LASSO, trees, neural nets) for
steps 1 and 2, not from any new primitives.

### Cross-Fitting (CRITICAL — Not Optional)

DML requires **cross-fitting** (also called sample splitting or cross-validation).
Fitting and evaluating the nuisance functions (E[Y|W], E[D|W]) on the same sample
produces in-sample overfitting bias that does NOT vanish asymptotically.

```
5-fold cross-fitting protocol:
1. Split data into 5 folds K_1, ..., K_5
2. For each fold k:
   a. Train nuisance models on all folds EXCEPT k (K\{k})
   b. Predict residuals Ỹ, D̃ on fold k only
3. Concatenate all out-of-fold residuals
4. Estimate θ from the concatenated residuals
```

**CRITICAL TRAP: Using in-sample residuals produces SEVERELY biased DML estimates.**
This is not a small-sample issue — the bias is O(1) and does not shrink with n.
Every DML implementation that skips cross-fitting is wrong, full stop.

### R: DoubleML

```r
library(DoubleML)
library(mlr3)
library(mlr3learners)

# Define learners for nuisance functions:
ml_y <- lrn("regr.ranger",   num.trees=100, min.node.size=5)  # for E[Y|W]
ml_d <- lrn("regr.ranger",   num.trees=100, min.node.size=5)  # for E[D|W]
# (binary D: use classif.ranger for propensity, but regr works too)

# Create DoubleML data object:
# dml_data: Y=outcome, D=treatment, X=covariates (controls)
dml_data <- DoubleMLData$new(df,
                              y_col  = "Y",
                              d_cols = "D",
                              x_cols = c("W1", "W2", "W3"))

# Partially Linear Regression (PLR) — the canonical DML:
dml_plr <- DoubleMLPLR$new(dml_data,
                             ml_l = ml_y,      # learner for E[Y|W]
                             ml_m = ml_d,      # learner for E[D|W]
                             n_folds = 5,      # k-fold cross-fitting
                             n_rep   = 1)      # repetitions of cross-fitting

dml_plr$fit()
dml_plr$summary()
# $coef: θ̂ (causal effect)
# $se: standard error (semiparametrically efficient)
# $t_stat, $pval: hypothesis test θ = 0

# Interactive Regression Model (IRM) — for binary treatment + CATE:
ml_y2 <- lrn("regr.ranger",   num.trees=100)
ml_d2 <- lrn("classif.ranger", num.trees=100)   # binary treatment

dml_irm <- DoubleMLIRM$new(dml_data,
                            ml_g = ml_y2,    # outcome model
                            ml_m = ml_d2,    # propensity model
                            n_folds = 5)
dml_irm$fit()
dml_irm$summary()    # ATE estimate (doubly robust)
```

### Python: doubleml

```python
from doubleml import DoubleMLPLR, DoubleMLIRM
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegressionCV
import pandas as pd
import numpy as np
import doubleml as dml

# Create DoubleML data object:
dml_data = dml.DoubleMLData.from_arrays(
    x=X,      # controls (n x k matrix)
    y=Y,      # outcome (n,)
    d=D       # treatment (n,)
)

# PLR with random forest nuisance:
ml_l = RandomForestRegressor(n_estimators=100, min_samples_leaf=5)  # E[Y|W]
ml_m = RandomForestRegressor(n_estimators=100, min_samples_leaf=5)  # E[D|W]

dml_plr = DoubleMLPLR(dml_data,
                       ml_l = ml_l,
                       ml_m = ml_m,
                       n_folds = 5)
dml_plr.fit()
print(dml_plr.summary)
# coef: θ̂; std err; t; P>|t|; [0.025, 0.975]
```

```python
# Python: econml LinearDML (more flexible)
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

est = LinearDML(
    model_y = GradientBoostingRegressor(),     # E[Y|W]
    model_t = GradientBoostingClassifier(),    # E[D|W]  (binary treatment)
    cv      = 5,                               # cross-fitting folds
    random_state = 42
)
est.fit(Y, D, X=X_het, W=W_conf)
# X_het: features for CATE heterogeneity
# W_conf: confounders to residualize out

# ATE:
print(est.ate_)
print(est.ate_stderr_)

# CATE at new points:
cate = est.effect(X_het_test)
cate_lb, cate_ub = est.effect_interval(X_het_test, alpha=0.05)
```

**Validation targets:**
```python
# True θ = 2.0 in synthetic PLR:
# Y = 2*D + sin(W1) + W2^2 + noise
# D = W1 + 0.5*W2 + noise
np.random.seed(42)
n = 2000
W = np.random.randn(n, 3)
D = W[:, 0] + 0.5*W[:, 1] + 0.3*np.random.randn(n)
Y = 2*D + np.sin(W[:, 0]) + W[:, 1]**2 + np.random.randn(n)

# Expected: θ̂ ≈ 2.0 ± 0.05 for n=2000 with random forest nuisance
# DML is root-n consistent and asymptotically normal despite nonparametric nuisance
```

---

## Section 4: Causal Forests and CATE Estimation

### What Causal Forests Do

Causal forests (Wager & Athey 2018) estimate the *heterogeneous treatment effect*
CATE(x) = E[Y(1) - Y(0) | X=x] at each point x in covariate space.

**Key difference from predictive forests**: causal forest splits maximize heterogeneity
in the *treatment effect* (CATE), not in the outcome Y. A split is good if the two
resulting leaves have very different average treatment effects.

**Algorithm sketch:**
```
For each tree b in B:
1. Draw subsample (WITHOUT replacement, half the data)
2. Grow tree by recursive splitting:
   - At each node, split on feature j at value c to maximize:
     τ̂(left) - τ̂(right)  (difference in estimated CATE across children)
   - τ̂ estimated by local 2SLS in each candidate child
3. Final CATE(x) = average over trees of leaf-level treatment effects
```

### R: grf (Generalized Random Forests)

```r
library(grf)

# Standard causal forest:
cf <- causal_forest(
    X    = as.matrix(df[, c("X1","X2","X3")]),  # covariates
    Y    = df$Y,                                  # outcome
    W    = df$D,                                  # binary treatment
    num.trees   = 2000,
    min.node.size = 5,
    sample.fraction = 0.5,    # subsampling for honesty
    honesty = TRUE,           # honest estimation (required for valid CI)
    seed = 42
)

# Estimate CATE for each observation:
cate_hat   <- predict(cf)$predictions        # point estimates
cate_ci    <- predict(cf, estimate.variance=TRUE)
# cate_ci$variance.estimates: variance of CATE estimates

# 95% CI for CATE:
se <- sqrt(cate_ci$variance.estimates)
cate_lower <- cate_hat - 1.96 * se
cate_upper <- cate_hat + 1.96 * se

# ATE:
ate_result <- average_treatment_effect(cf)
# ate_result["estimate"]: ATE
# ate_result["std.err"]:  SE (augmented IPW under the hood)

# ATT:
att_result <- average_treatment_effect(cf, target.sample="treated")

# Variable importance (CATE heterogeneity, not Y prediction):
vi <- variable_importance(cf)
# Each entry: how much this variable contributes to CATE heterogeneity
# NOT the same as feature importance in a predictive forest
```

```r
# Test for CATE heterogeneity — is CATE constant or heterogeneous?
test_calibration(cf)
# Outputs:
#   mean.forest.prediction: should ≈ ATE if calibrated
#   differential.forest.prediction: should > 0 if heterogeneous CATE
# p-value on differential.forest.prediction tests H₀: CATE is constant

# Best linear projection of CATE onto covariates:
blp <- best_linear_projection(cf, A=as.matrix(df[, c("X1","X2","X3")]))
# Shows which X covariates explain CATE heterogeneity most linearly
```

### Python: econml CausalForest

```python
from econml.causal_forest import CausalForest
import numpy as np

cf = CausalForest(
    n_estimators   = 2000,
    min_samples_leaf = 5,
    max_features   = "auto",
    random_state   = 42
)
cf.fit(Y, D, X=X)

# CATE estimates:
cate = cf.predict(X_test)

# Confidence intervals:
cate_lb, cate_ub = cf.predict_interval(X_test, alpha=0.05)

# ATE:
print(cf.ate_)
```

```python
# econml CausalForestDML — DML + causal forest combined
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

cfdml = CausalForestDML(
    model_y    = GradientBoostingRegressor(),
    model_t    = GradientBoostingClassifier(),
    n_estimators = 1000,
    cv         = 5,
    random_state = 42
)
cfdml.fit(Y, D, X=X_het, W=W_conf)
cate = cfdml.effect(X_het_test)
```

**Validation targets:**
```r
# Synthetic: CATE depends on X1
# Y(1) - Y(0) = 1 + 2*X1   (heterogeneous effect)
set.seed(42)
n <- 2000
X <- matrix(rnorm(n*3), n, 3)
D <- rbinom(n, 1, plogis(X[,1]))
Y <- (1 + 2*X[,1]) * D + X[,2] + rnorm(n)

cf <- causal_forest(X, Y, D, num.trees=2000, honesty=TRUE)
cate_hat <- predict(cf)$predictions

# Expected:
# cor(cate_hat, 1 + 2*X[,1]) > 0.8  (forest recovers linear CATE)
# ATE ≈ E[1 + 2*X1] = 1 + 2*0 = 1.0
cor(cate_hat, 1 + 2*X[,1])     # should be > 0.8
average_treatment_effect(cf)    # should be ≈ 1.0 ± 0.1
```

**CRITICAL TRAP: Variable importance in causal forests ≠ predictive forests.**
In a predictive forest, high importance = the variable predicts Y well.
In a causal forest, high importance = the variable explains WHERE treatment effects differ.
A variable can have zero predictive importance but high causal importance, and vice versa.
Do NOT use causal forest variable importance to select control variables.

---

## Section 5: Synthetic Control

### What Synthetic Control Does

When you have one treated unit (a country, market, firm) and want to estimate the
counterfactual "what would have happened without treatment?", synthetic control
constructs a weighted average of control units that best matches pre-treatment outcomes.

**Setup:**
```
J+1 units: unit 1 = treated, units 2..J+1 = donors (never treated)
T0 pre-treatment periods, T1 post-treatment periods

Synthetic control: ŷ_1t(0) = Σ_{j=2}^{J+1} w_j * y_jt   for t > T0

where w = (w_2, ..., w_{J+1}) minimizes:
    ||X_1 - X_0 w||²_V   subject to: w_j ≥ 0, Σ w_j = 1

X_1: pre-treatment characteristics of treated unit
X_0: pre-treatment characteristics of donor units (columns)
V: diagonal matrix of predictor importance (also optimized)
```

**Treatment effect:** τ_t = y_1t - ŷ_1t(0) for t > T0

**This is a constrained QP (F05 optimization) — not a regression.**
The constraint (weights sum to 1, non-negative) means it is NOT solved by GramMatrix.
It requires a quadratic programming solver.

### R: Synth

```r
library(Synth)

# Data must be in "long" format with dataprep():
dataprep.out <- dataprep(
    foo            = df,              # data frame
    predictors     = c("gdp","pop","trade"),  # matching variables
    predictors.op  = "mean",         # how to summarize over pre-treatment period
    time.predictors.prior = 1980:1990,  # pre-treatment period for predictors
    special.predictors = list(
        list("outcome", 1985, "mean"),    # match on lagged outcome
        list("outcome", 1988, "mean")
    ),
    dependent      = "outcome",       # outcome variable
    unit.variable  = "unit_id",       # unit identifier
    time.variable  = "year",          # time identifier
    treatment.identifier  = 1,        # treated unit's ID
    controls.identifier   = 2:20,     # donor unit IDs
    time.optimize.ssr     = 1980:1990, # optimize fit over pre-treatment
    time.plot             = 1980:2000  # full plot range
)

synth.out <- synth(dataprep.out)
# synth.out$solution.w: optimal weights w_j
# synth.out$loss.w:     pre-treatment fit (lower = better synthetic control)

# Tables and plots:
synth.tables <- synth.tab(synth.out, dataprep.out)
synth.tables$tab.v   # predictor importance V matrix
synth.tables$tab.w   # donor weights

# Plot actual vs. synthetic:
path.plot(synth.out, dataprep.out,
          Ylab="Outcome", Xlab="Year",
          Ylim=c(0, 100), Legend=c("Treated", "Synthetic"))

# Treatment effect (post-treatment gap):
gaps.plot(synth.out, dataprep.out)
```

### Python: pysynth

```python
import pysynth as ps
import pandas as pd

# df: panel data, columns = [unit, time, outcome, cov1, cov2, ...]
sc = ps.Synth()
sc.fit(
    df,
    unit_col     = "unit",
    time_col     = "time",
    outcome_col  = "outcome",
    treated_unit = "unit_A",
    pre_periods  = range(1980, 1991),
    post_periods = range(1991, 2001)
)

print(sc.weights_)             # donor weights (J,) summing to 1
print(sc.pre_rmspe_)           # pre-treatment RMSPE (fit quality)
treatment_effect = sc.gaps_    # time series of (actual - synthetic) post-treatment
```

**Inference via permutation (placebo tests):**
```r
# Standard Synth provides no p-values. Inference via "placebo tests":
# Apply synthetic control to EACH donor unit (as if it were treated).
# If actual effect >> distribution of placebo effects: significant.

# In-time placebos: use only pre-treatment period; test if gap = 0 when it should
# In-space placebos: use each donor as "treated" — see if actual effect is an outlier

# Exclude units with poor pre-treatment fit (pre-RMSPE > 2x treated unit):
# These make poor placebos because they couldn't be well-matched anyway.
```

**Validation target:**
```python
# Pre-treatment RMSPE should be small (< 5% of outcome scale)
# Post-treatment gaps should show a clear break at treatment time
# Placebo test: treated unit's post/pre RMSPE ratio should be >> donor average
```

**Tambear decomposition:**
```
Step 1: Build X_1 (treated predictor vector) and X_0 (donor predictor matrix)
        = accumulate(ByUnit + ByTime_PrePeriod, Mean{predictors}, Add)
        = F06/F12 group means

Step 2: Solve constrained QP: min ||X_1 - X_0 w||² s.t. w≥0, 1'w=1
        = F05 optimization (quadratic objective + linear constraints)
        Uses active set, interior point, or projected gradient

Step 3: Counterfactual = X_0_post @ w
        = gather(ByDonor) + weighted sum
        = accumulate(All, WeightedSum{w=donor_weights}, Add)
```

---

## Section 6: Mediation Analysis

### What Mediation Does

Mediation analysis decomposes the total effect of D on Y into:
- **Direct effect (DE)**: D → Y (not through M)
- **Indirect effect (IE)**: D → M → Y (through mediator M)
- **Total effect (TE)**: TE = DE + IE

```
D ──────────────────────────────────► Y   (direct)
D ──────► M ──────────────────────── ► Y  (mediated through M)
```

**Baron-Kenny sequential regression (classical approach):**
```
Equation 1: Y = a₀ + a₁·D + ε₁              (total effect: a₁ = TE)
Equation 2: M = b₀ + b₁·D + ε₂              (D → M: b₁)
Equation 3: Y = c₀ + c₁·D + c₂·M + ε₃       (direct effect: c₁ = DE)

Indirect effect (IE) = b₁ × c₂  (product of coefficients)
DE = c₁,  TE = a₁,  IE = TE - DE = a₁ - c₁  (should ≈ b₁×c₂)
```

**All three equations are F10 GramMatrix solves.**

### R: mediation package

```r
library(mediation)

# Step 1: mediator model (D predicts M):
med.model <- lm(M ~ D + X1 + X2, data=df)

# Step 2: outcome model (D + M predict Y):
out.model <- lm(Y ~ D + M + X1 + X2, data=df)

# Step 3: mediation analysis with bootstrap CI:
med.out <- mediate(med.model, out.model,
                   treat   = "D",       # treatment variable name
                   mediator = "M",      # mediator variable name
                   boot = TRUE,         # bootstrap (preferred over delta method)
                   sims = 1000,         # number of bootstrap samples
                   boot.ci.type = "perc")   # percentile CI (or "bca" for BCa)

summary(med.out)
# ACME:  Average Causal Mediation Effect (indirect effect = IE)
# ADE:   Average Direct Effect (direct effect = DE)
# Total: Total Effect (TE = IE + DE)
# Prop:  Proportion mediated (IE / TE)
# All with 95% CI

# Plot:
plot(med.out)
```

**Sensitivity analysis** — mediation requires "no unmeasured mediator-outcome confounders":
```r
sens.out <- medsens(med.out, rho.by=0.1, effect.type="indirect")
summary(sens.out)
# rho = correlation between residuals of mediator and outcome models
# rho=0: assumption holds; sensitivity shows how much rho can be before IE → 0
plot(sens.out)
```

### Python: from scratch (product of coefficients)

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def mediation_analysis(Y, D, M, X=None, n_bootstrap=1000, alpha=0.05):
    """
    Baron-Kenny mediation with bootstrap CI.
    Y: outcome, D: treatment, M: mediator, X: confounders
    """
    def _fit(y, features):
        lr = LinearRegression().fit(features, y)
        return lr.coef_

    def _stack(D, M, X):
        parts = [D.reshape(-1,1), M.reshape(-1,1)]
        if X is not None:
            parts.append(X)
        return np.hstack(parts)

    def _stack_d(D, X):
        parts = [D.reshape(-1,1)]
        if X is not None:
            parts.append(X)
        return np.hstack(parts)

    n = len(Y)

    # Point estimates:
    coef_med = _fit(M, _stack_d(D, X))      # D → M: coef_med[0] = b1
    coef_out = _fit(Y, _stack(D, M, X))     # D+M → Y: coef_out[0]=c1 (DE), coef_out[1]=c2

    b1 = coef_med[0]   # D → M
    c1 = coef_out[0]   # direct effect
    c2 = coef_out[1]   # M → Y (controlling D)
    ie = b1 * c2       # indirect effect (product of coefficients)
    te = c1 + ie       # total effect

    # Bootstrap CI for indirect effect:
    ie_boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        Y_b, D_b, M_b = Y[idx], D[idx], M[idx]
        X_b = X[idx] if X is not None else None

        cm_b = _fit(M_b, _stack_d(D_b, X_b))
        co_b = _fit(Y_b, _stack(D_b, M_b, X_b))
        ie_boot[i] = cm_b[0] * co_b[1]

    ci_lo = np.percentile(ie_boot, 100 * alpha / 2)
    ci_hi = np.percentile(ie_boot, 100 * (1 - alpha / 2))

    return {
        "direct_effect":   c1,
        "indirect_effect": ie,
        "total_effect":    te,
        "prop_mediated":   ie / te if te != 0 else np.nan,
        "ie_ci_95":        (ci_lo, ci_hi)
    }
```

**Validation targets:**
```python
# Synthetic: Y = 0.5*D + 0.8*M + noise, M = 0.6*D + noise
# DE = 0.5, b1=0.6, c2=0.8 → IE = 0.6*0.8 = 0.48, TE = 0.5+0.48 = 0.98
np.random.seed(42)
n = 500
D = np.random.binomial(1, 0.5, n)
M = 0.6 * D + np.random.randn(n)
Y = 0.5 * D + 0.8 * M + np.random.randn(n)

result = mediation_analysis(Y, D, M)
# Expected: DE ≈ 0.5, IE ≈ 0.48, TE ≈ 0.98, prop_mediated ≈ 0.49
# Bootstrap CI for IE should exclude 0 (IE is real)
```

**Tambear decomposition:**
```
Step 1: GramMatrix solve for [b0, b1]: M ~ D + X   (F10)
Step 2: GramMatrix solve for [c0, c1, c2]: Y ~ D + M + X   (F10)
Step 3: IE = b1 * c2  (scalar multiply — trivial)
Step 4: Bootstrap = accumulate over bootstrap resamples
        = F07 bootstrap CI pattern
```

---

## Section 7: Directed Acyclic Graphs (DAGs)

### What DAGs Provide

DAGs encode causal assumptions as graph structure. The key tools:
- **d-separation**: reads off conditional independence from graph structure
- **Adjustment sets**: valid sets of variables to condition on for identification
- **Backdoor criterion**: identifies which confounders to block
- **Front-door criterion**: identifies when mediation enables identification even with unmeasured confounding

**DAG ≠ estimator.** A DAG specifies what to condition on; a regression (F10) or IPW
performs the estimation. DAG analysis is a design-time tool.

### R: dagitty

```r
library(dagitty)
library(ggdag)

# Define DAG in dagitty syntax:
# -> = arrow (causal direction)
# <-> = bidirected edge (unmeasured common cause)
dag <- dagitty('dag {
    D [exposure]
    Y [outcome]
    X1 -> D
    X1 -> Y
    D -> Y
    D -> M -> Y
    U -> D; U -> Y  [unobserved]
}')

# Check if D -> Y is identified:
isIdentified(dag, exposure="D", outcome="Y")

# Find valid adjustment sets (minimal conditioning sets for backdoor):
adjustmentSets(dag, exposure="D", outcome="Y", effect="total")
# Returns: list of minimal sets to condition on

# Conditional independence implications:
impliedConditionalIndependencies(dag)
# These are testable implications — can validate against data

# d-separation queries:
dseparated(dag, X="D", Y="Y", Z=c("X1"))    # is D ⊥ Y | X1?
dconnected(dag, X="D", Y="Y", Z=c())        # is D connected to Y?

# Instrumental variable identification:
instrumentalVariables(dag, exposure="D", outcome="Y")
# Returns: variables that qualify as IVs given DAG assumptions

# Plot:
library(ggdag)
ggdag(dag) + theme_dag()
ggdag_adjustment_set(dag, exposure="D", outcome="Y") + theme_dag()
```

### Python: causalnex

```python
from causalnex.structure import StructureModel

# Build DAG manually:
sm = StructureModel()
sm.add_edges_from([
    ("D", "Y"),
    ("X1", "D"),
    ("X1", "Y"),
    ("D", "M"),
    ("M", "Y")
])

# Or use dagitty via string (rpy2 bridge or causality package):
# pip install causality
# pip install pgmpy  (Bayesian network structure)
```

**Practical workflow — causal graph before estimation:**
```
1. Draw the DAG (use dagitty web UI or R code)
2. Identify minimal adjustment sets
3. Check testable implications against data
4. If implications hold: proceed with estimation using identified sets
5. If implications fail: DAG is misspecified — revise

Example:
  DAG says: X1 ⊥ Y | D  (X1 d-separated from Y given D)
  Test: cor(X1, Y | D) should ≈ 0
  If not: omitted variable or wrong causal structure
```

---

## Section 8: Cross-Reference — Regression Discontinuity

RD design is fully documented in **F12 gold standards** (panel econometrics).
The core structure:
- Local linear regression at a cutoff (F10: GramMatrix with kernel-weighted obs)
- Bandwidth selection: `rdrobust` package implements Imbens-Kalyanaraman (IK) bandwidth
- Running variable = X, cutoff = c, treatment = 1{X ≥ c}

```r
library(rdrobust)
rdd_out <- rdrobust(y=df$Y, x=df$running_var, c=0)
summary(rdd_out)   # LATE at cutoff with optimal bandwidth
rdplot(y=df$Y, x=df$running_var, c=0)  # visual check
```

**Key tambear link**: RD is F10 (GramMatrix with kernel/distance weights near cutoff).
The bandwidth creates an implicit "ByRange" grouping — only obs near the cutoff enter
the local regression. This is a Distance-filtered accumulate.

---

## Section 9: Cross-Reference — Difference-in-Differences

DiD is fully documented in **F12 gold standards** (panel econometrics).
Two-by-two DiD = FE with a treatment × post_period interaction.
Staggered DiD = Callaway-Sant'Anna or Gardner 2-stage.

The causal inference content specific to DiD:
- **Parallel trends assumption**: the counterfactual trend for treated units equals
  the observed trend for control units (in the absence of treatment)
- **Violation check**: event study plot — pre-treatment coefficients should be ≈ 0
- **Heterogeneous treatment timing**: use Callaway-Sant'Anna `did` package, NOT TWFE

---

## Section 10: Tambear Decomposition Summary

```
Causal estimator          | Tambear primitives              | Family
--------------------------|---------------------------------|--------
Regression adjustment     | GramMatrix + Cholesky           | F10
IPW (ATE weights)         | accumulate(All, WgtdMean{1/ps}) | F10 weighted
Propensity score est.     | GramMatrix + Newton (logistic)  | F10/GLM
NN matching               | DistancePairs + row argmin      | F01 + F20
AIPW (doubly robust)      | F10 × 2 + F10 weighted          | F10 × 3
DML (two residualizations)| GramMatrix × 2 + IV moment      | F10 × 2 + F12
Causal forest             | F10 local moments per leaf      | F10 × forests
Synthetic control         | group means + constrained QP    | F06 + F05
Mediation (BK)            | GramMatrix × 2 + scalar product | F10 × 2
RD (local linear)         | F10 with bandwidth kernel       | F10 (F12 ref)
DiD (TWFE)               | F12 FE with interaction term    | F12
DAG adjustment set        | design tool → F10 with chosen Z | F10 (chosen Z)
```

**Core architectural insight**: F35 adds no new primitives to tambear.
Every causal estimator is:
1. Some combination of F10 (GramMatrix/regression) — possibly weighted
2. Applied with a specific conditioning strategy (group means, residuals, matching weights)
3. Combined with F05 (constrained QP) for synthetic control

The causal *logic* lives in:
- Which observations get which weights (IPW, matching)
- Which variables to condition on (DAG-identified adjustment sets)
- Which nuisance functions to residualize out first (DML)
- Which optimization objective to use (synthetic control, causal forest splitting)

None of these require new GPU kernels. They require composing existing primitives
with the right structure. This is the sharing surface: causal inference as F10
with structured observation weights and conditioning sets.

---

## Key Traps Summary

```
Trap                               | Package affected       | Fix
-----------------------------------|------------------------|----------------------------------
MatchIt default = no replacement   | MatchIt (R)            | Use replace=TRUE or method="full"
Extreme propensity scores (ps≈0,1) | All IPW methods        | Trim at 5th/95th pctile or use stabilized weights
DML in-sample residuals            | All DML implementations| Always cross-fit (n_folds=5)
Causal forest var importance ≠ pred| grf, CausalForest      | Use for CATE heterogeneity only
AIPW needs ≥1 correct model        | AIPW/causalweight      | Fit both; double robustness is not magic
Parallel trends untested           | DiD (all packages)     | Always plot event study pre-trends
Synth: no built-in inference       | Synth (R)              | Permutation/placebo tests required
Baron-Kenny: assumes no D-M conf.  | mediation (R)          | Sensitivity analysis (medsens)
DAG: structure is an assumption    | dagitty                | Test implied independencies
IV exclusion restriction           | AER, linearmodels      | Cannot be tested — must argue it
```

---

## Financial Signal Farm Applications

### Market Microstructure Causal Questions

```
Question                               | Estimator    | Treatment D         | Outcome Y
---------------------------------------|--------------|---------------------|------------------
Does tick size change affect spreads?  | RD           | tick_size_change     | bid_ask_spread
Does a news event cause price impact?  | DiD / SCM    | news_event           | return_vol
Do dark pool trades cause lit slippage?| IV (venue)   | dark_pool_indicator  | price_impact
Does trade size cause permanent impact?| DML          | log_trade_size       | log_price_move
Heterogeneous impact across stocks?    | Causal Forest| large_trade_dummy    | 1min_return
Does order flow cause vol clustering?  | Mediation    | D=order_flow         | M=vol_regime → Y=spread
```

**DML is particularly valuable for market microstructure** because:
1. Trade size, direction, time-of-day are all confounded
2. We want the *causal* price impact after removing predictable components
3. Cross-fitting handles the "alpha decay" problem — we're not fitting to the same
   observations we're computing the effect for

**Synthetic control for policy changes**: when a market structure change (MIFID II,
Reg NMS, exchange fee change) affects one exchange and not others, synthetic control
builds a "what would have happened" counterfactual from unaffected exchanges.

---

## Environment Setup

```bash
# R:
install.packages(c("MatchIt", "WeightIt", "cobalt", "mediation",
                   "DoubleML", "grf", "Synth", "dagitty", "ggdag",
                   "causalweight", "AER", "mlr3", "mlr3learners"))

# Python:
pip install econml doubleml causalml pysynth causalnex
pip install scikit-learn numpy pandas scipy
```

```python
# Quick environment check:
import econml; print(econml.__version__)      # >= 0.14
import doubleml; print(doubleml.__version__)  # >= 0.5
from econml.dml import LinearDML             # should import clean
from econml.causal_forest import CausalForest  # should import clean
```
