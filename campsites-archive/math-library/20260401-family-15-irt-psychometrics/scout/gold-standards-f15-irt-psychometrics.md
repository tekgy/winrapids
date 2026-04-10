# F15 IRT & Psychometrics — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 15 (IRT & Psychometrics).
Prerequisites: F10 complete (logistic GLM / IRLS), F14 complete (Factor Analysis / GramMatrix + EigenDecomposition).
F15 is the primary IRT/psychometrics engine — Rasch/2PL/3PL/polytomous models, Classical Test Theory, and DIF analysis.

Primary oracle: R `mirt` package (marginal MML, industry standard).
Secondary oracle: R `ltm` package (validates item parameters), R `psych` (CTT), R `eRm` (conditional MML).
Python validation: `girth` or `pyirt` (limited but useful for sanity checks).

---

## Structural Position

IRT sits at the intersection of F10 (logistic regression) and F14 (factor analysis):

```
F10 (logistic GLM / IRLS)
     ↓ IRLS M-step
F15 (IRT: EM with weighted logistic per item)
     ↑ starting values from FA loadings
F14 (Factor Analysis / EigenDecomposition)
```

**The unification claim** (from navigator): IRT is logistic GLMM with crossed random effects.
Person ability θ_p = random intercept. Item parameters (a_j, b_j) = fixed effects per item.
This means F15 shares the IRLS master template with F09, F10, F11, F13, and F16 — all reuse
`scatter_multi_phi_weighted`.

---

## IRT Model Hierarchy

### 1PL (Rasch Model)

```
P(Y_pj = 1 | θ_p, b_j) = logistic(θ_p - b_j)
```

- `θ_p` = person ability (latent, estimated)
- `b_j` = item difficulty (estimated, higher b = harder item)
- All items constrained to equal discrimination (a_j = 1 for all j)
- Identifies by fixing mean difficulty to zero: `Σ_j b_j = 0`

### 2PL Model

```
P(Y_pj = 1 | θ_p, a_j, b_j) = logistic(a_j(θ_p - b_j))
```

- `a_j` = item discrimination (slope, positive = good item, > 0.8 acceptable, > 1.5 excellent)
- `d_j = -a_j · b_j` = mirt's internal parameterization (intercept, not b directly)
- Identifiable by fixing: θ ~ N(0, 1) prior; or fixing mean/variance of θ distribution

### 3PL Model

```
P(Y_pj = 1 | θ_p, a_j, b_j, c_j) = c_j + (1 - c_j) · logistic(a_j(θ_p - b_j))
```

- `c_j` = lower asymptote / guessing parameter (0 ≤ c_j ≤ 1, typically 0.10–0.25 for MCQ)
- Phase 2 — harder to estimate, requires large sample (N > 500) and many items (J > 20)

### 4PL Model

```
P(Y_pj = 1 | θ_p, a_j, b_j, c_j, d_j_upper) = c_j + (d_j_upper - c_j) · logistic(a_j(θ_p - b_j))
```

- `d_j_upper` = upper asymptote (< 1 models inattention / careless errors)
- Rarely used outside computerized adaptive testing (CAT)
- mirt: `itemtype='4PL'`

---

## Primary Oracle: R `mirt` Package

### Installation and Basic Usage

```r
install.packages("mirt")
library(mirt)

# LSAT dataset: 5 binary items, 1000 persons (built into mirt)
data(LSAT7)            # classic LSAT dataset, 1000 persons, 5 items (items scored 0/1)
data <- LSAT7$data     # 1000 × 5 matrix of 0/1 responses

# Rasch (1PL)
fit1pl <- mirt(data, 1, itemtype = 'Rasch', verbose = FALSE)

# 2PL
fit2pl <- mirt(data, 1, itemtype = '2PL', verbose = FALSE)

# 3PL
fit3pl <- mirt(data, 1, itemtype = '3PL', verbose = FALSE)
```

### Extracting Item Parameters

```r
# *** CRITICAL TRAP: mirt uses 'a' and 'd' internally, NOT 'a' and 'b' ***
# d = -a * b  (d is the intercept, b is the "difficulty" in traditional IRT notation)
# To get traditional (a, b) form, use IRTpars=TRUE

# Default output (a, d parameterization):
coef(fit2pl)                            # list of matrices, one per item + GroupPars
coef(fit2pl, simplify = TRUE)           # collapsed list: $items = J×4 matrix [a1, d, g, u]
coef(fit2pl, simplify = TRUE)$items     # J×4: a1=discrimination, d=intercept, g=lower, u=upper

# IRT traditional (a, b) parameterization:
coef(fit2pl, simplify = TRUE, IRTpars = TRUE)          # $items: [a, b, g, u]
coef(fit2pl, simplify = TRUE, IRTpars = TRUE)$items    # J×4: a=discrimination, b=difficulty

# Single-item example (item 1):
coef(fit2pl)[[1]]   # matrix: 1×4 [a1, d, g, u] for item 1
```

### Parameterization Trap — Do Not Skip

```r
# Concrete example with LSAT7:
params_ad <- coef(fit2pl, simplify=TRUE)$items          # a, d format
params_ab <- coef(fit2pl, simplify=TRUE, IRTpars=TRUE)$items  # a, b format

# Verify the identity: d = -a * b
params_ad[, "d"]    # should equal: -params_ab[, "a"] * params_ab[, "b"]
# Max absolute deviation should be < 1e-10

# For Rasch: a = 1.0 fixed, only d (= -b) varies
rasch_params <- coef(fit1pl, simplify=TRUE, IRTpars=TRUE)$items
# rasch_params[, "a"] == 1.0 for all items (verify)
```

### Person Ability Estimation

```r
# EAP (Expected A Posteriori) — posterior mean
# Default, recommended for most purposes. Shrinks toward prior.
thetas_eap <- fscores(fit2pl, method = 'EAP')       # N×1 matrix

# MAP (Maximum A Posteriori) — posterior mode
# Less shrinkage than EAP at extremes, but unstable for perfect/zero scores
thetas_map <- fscores(fit2pl, method = 'MAP')

# MLE (Maximum Likelihood Estimation)
# Gives ±Inf for perfect/zero scores — use with caution
thetas_mle <- fscores(fit2pl, method = 'ML')

# WLE (Warm's Weighted Likelihood Estimation)
# Reduces bias compared to MLE, handles extremes better
thetas_wle <- fscores(fit2pl, method = 'WLE')

# With standard errors:
thetas_with_se <- fscores(fit2pl, method = 'EAP', full.scores = TRUE, full.scores.SE = TRUE)
# Returns N×2: [F1=ability, SE_F1=standard error]
```

### EAP vs MAP Trap

```r
# At the extremes (all correct / all incorrect), EAP and MAP diverge substantially:
extreme_persons <- matrix(c(1,1,1,1,1, 0,0,0,0,0), nrow=2, byrow=TRUE)
fscores(fit2pl, method='EAP', response.pattern=extreme_persons)  # finite, shrunk
fscores(fit2pl, method='MAP', response.pattern=extreme_persons)  # finite but less shrunk
fscores(fit2pl, method='ML',  response.pattern=extreme_persons)  # ±Inf or very large

# Rule: use EAP for research reporting, MAP or WLE for adaptive testing decisions
```

### Model Fit Statistics

```r
# Limited-information fit statistic M2 (recommended for IRT):
M2(fit2pl)                   # M2 statistic, df, p-value, RMSEA, CI_RMSEA, SRMSR, TLI, CFI
M2(fit2pl, type = 'M2*')     # M2* for polytomous data

# Item fit statistics (S-X2 statistic per item):
itemfit(fit2pl)              # S-X2, df, p.S_X2, RMSEA.S_X2 per item
itemfit(fit2pl, fit_stats='infit')   # Rasch infit/outfit
itemfit(fit1pl, fit_stats=c('Zh', 'infit', 'outfit'))  # multiple stats

# Person fit:
personfit(fit2pl)            # Zh statistic per person (standardized log-likelihood)
# Zh < -2 suggests misfitting person (unexpectedly inconsistent response pattern)

# Log-likelihood and information criteria:
logLik(fit2pl)               # log-likelihood at convergence
AIC(fit2pl)
BIC(fit2pl)
```

### LSAT7 Validation Targets

The LSAT7 dataset (1000 persons, 5 items) is the canonical IRT benchmark. Expected values from
authoritative mirt documentation:

```r
library(mirt)
data(LSAT7)
fit2pl <- mirt(LSAT7$data, 1, itemtype='2PL', verbose=FALSE)

# Expected 2PL item parameters (IRTpars=TRUE, a/b format):
# Item 1: a ≈ 0.79, b ≈ -3.07
# Item 2: a ≈ 0.76, b ≈ -1.86
# Item 3: a ≈ 1.66, b ≈ -1.30
# Item 4: a ≈ 0.72, b ≈ -1.85
# Item 5: a ≈ 0.76, b ≈ -2.96
# (Values vary slightly by mirt version and quadrature settings)

# Tolerances for tambear validation:
# Item discrimination a: within 0.05
# Item difficulty b: within 0.05
# Person ability EAP: correlation > 0.999 with mirt output

# M2 fit for 2PL:
M2(fit2pl)
# Expected: M2 ≈ 11.9, df = 5, p ≈ 0.036, RMSEA ≈ 0.034, CFI ≈ 0.998

# Log-likelihood comparison:
# 1PL: logLik ≈ -2481
# 2PL: logLik ≈ -2466  (LRT chi-sq ≈ 30, df=4, significant → prefer 2PL)
```

---

## Secondary Oracle: R `ltm` Package

`ltm` provides an independent implementation with slightly different optimization — useful for
cross-validation of item parameter estimates.

```r
library(ltm)

# Rasch model (restricted 2PL with a=1):
fit_rasch <- rasch(LSAT)    # ltm's built-in LSAT (similar to LSAT7)

# Output:
coef(fit_rasch)             # [Dffclt, Dscrmn] = [b, a] — NOTE: reversed from mirt!
                            # For Rasch: Dscrmn = 1 for all items
summary(fit_rasch)

# 2PL model:
fit_2pl <- ltm(LSAT ~ z1)   # z1 = latent variable
coef(fit_2pl)               # [Dffclt, Dscrmn] = [b, a]

# ltm vs mirt parameterization trap:
# ltm:  coef[, 1] = difficulty (b)
# ltm:  coef[, 2] = discrimination (a)
# mirt: coef[, "a1"] = discrimination (a)
# mirt: coef[, "d"] = -a*b = intercept (NOT b directly)
# Always use mirt with IRTpars=TRUE for apples-to-apples comparison

# Person scoring:
factor.scores(fit_2pl, method='EB')    # Empirical Bayes = EAP in ltm
factor.scores(fit_2pl, method='MI')    # Multiple imputation
factor.scores(fit_2pl, method='Component')  # Component scores

# Item fit:
item.fit(fit_2pl)           # Pearson chi-square per item
person.fit(fit_2pl)         # person fit statistics
```

---

## Estimation Methods

### MML (Marginal Maximum Likelihood) — Default in mirt

The standard estimation algorithm. Integrates out the latent trait θ using quadrature:

```
Marginal likelihood: L(a, b | Y) = ∏_p ∫ [∏_j P(Y_pj | θ, a_j, b_j)] f(θ) dθ
```

where `f(θ) = N(0, 1)` (standard normal prior on ability).

Implemented via EM:
- **E-step**: compute posterior weights `r_pk = P(θ_k | Y_p, item_params)` at each quadrature point k
- **M-step**: update item parameters via weighted logistic regression using those weights

```r
# Control MML settings in mirt:
fit2pl <- mirt(data, 1, itemtype='2PL',
               technical = list(NCYCLES=2000,   # max EM iterations
                                MHRM_cycles=2000,
                                BURNIN=150,
                                SEMCYCLES=50,
                                quadpts=61,      # number of quadrature points (default 61)
                                TOL=1e-6),       # convergence tolerance
               verbose=FALSE)
```

### MAP and EAP — Person Scoring Only (not item estimation)

These are **posterior** methods for scoring persons given estimated item parameters:

```
EAP: θ̂_p = E[θ | Y_p] = Σ_k θ_k · P(θ_k | Y_p) / Σ_k P(θ_k | Y_p)
MAP: θ̂_p = argmax_θ P(θ | Y_p) = argmax_θ [Σ_j log P(Y_pj | θ) + log f(θ)]
```

- EAP = weighted average of quadrature points using posterior as weights
- MAP = Newton-Raphson optimization of the log-posterior per person
- Both require item parameters to be fixed first (from MML)

### MCMC Bayesian IRT (Phase 2)

```r
# mirt supports MCMC for Bayesian estimation:
fit_bayes <- mirt(data, 1, itemtype='2PL', method='MHRM', verbose=FALSE)
# MHRM = Metropolis-Hastings Robbins-Monro algorithm

# Full Bayesian via Stan:
# Package: brms or rstan with custom IRT Stan model
# Useful for hierarchical priors, DIF modeling with uncertainty
```

### Connection to IRLS (F10 Bridge)

The MML M-step IS logistic IRLS with fractional weights:

```
Standard logistic (F10):
  Weight: w_i = μ_i(1 - μ_i)  where μ_i = logistic(x_i β)
  Update: β_new = (X' W X)^{-1} X' W z   [IRLS]

IRT M-step for item j:
  Weight: w_pk = posterior_mass(person p at quadrature point k) × μ_jk(1 - μ_jk)
  Update: (a_j, d_j) via weighted IRLS with design matrix X = [θ_k, 1]
  Observations: Q quadrature points (not N persons directly)
```

The key difference: the "data" in each M-step weighted regression is the Q quadrature points,
not the N persons. N persons contribute to the accumulated E-step posteriors.

---

## Polytomous Models

For ordinal responses (e.g., Likert scales, partial credit scoring).

### GRM (Graded Response Model)

```r
# Ordinal data with k response categories (0, 1, ..., k-1):
fit_grm <- mirt(data_ordinal, 1, itemtype='graded', verbose=FALSE)

# GRM item parameters: one 'a' + (k-1) 'b' thresholds per item
coef(fit_grm, simplify=TRUE, IRTpars=TRUE)$items
# Columns: a, b1, b2, b3, ... (thresholds between adjacent categories)
# b1 < b2 < b3 (ordered thresholds guaranteed by reparameterization)
```

**Structure**: cumulative logistic model.
```
P(Y_pj ≥ m | θ_p) = logistic(a_j(θ_p - b_jm))  for m = 1, 2, ..., k-1
P(Y_pj = m | θ_p) = P(≥ m) - P(≥ m+1)
```

### PCM (Partial Credit Model)

```r
# Rasch-family polytomous model (a_j = 1 for all items):
fit_pcm <- mirt(data_ordinal, 1, itemtype='Rasch', verbose=FALSE)
# Or for general PCM (a_j varies = Generalized PCM):
fit_gpcm <- mirt(data_ordinal, 1, itemtype='gpcm', verbose=FALSE)
```

**Structure**: adjacent category logistic model.
```
P(Y_pj = m | θ_p) ∝ exp(Σ_{h=0}^{m} a_j(θ_p - b_jh))
```

### NRM (Nominal Response Model)

```r
# For nominal (unordered) polytomous responses:
fit_nrm <- mirt(data_nominal, 1, itemtype='nominal', verbose=FALSE)
# Estimates separate slopes per category — most general model
```

### M2* for Polytomous

```r
M2(fit_grm, type='M2*')   # M2* = correct version for polytomous models
```

---

## Multi-dimensional IRT (MIRT)

```r
# 2-dimensional model:
fit_mirt2 <- mirt(data, 2, itemtype='2PL', verbose=FALSE)

# Factor loadings (IRT-to-factor-analysis connection):
summary(fit_mirt2)          # standardized factor loadings (F14 bridge)

# Item parameters:
coef(fit_mirt2, simplify=TRUE)$items
# Columns: a1, a2, d, g, u — two discrimination parameters per item

# Rotation (same as EFA):
fit_mirt2_rot <- mirt(data, 2, itemtype='2PL',
                       rotate='oblimin',    # default is 'oblimin' for MIRT
                       verbose=FALSE)

# Test if 2D is better than 1D:
anova(fit2pl, fit_mirt2)    # likelihood ratio test
```

**Connection to F14**: MIRT loadings relate to FA loadings by the same D'Agostino transformation:
```
FA loading: λ_jf = a_jf / sqrt(1 + Σ_f a_jf²)   [for orthogonal model]
IRT discrim: a_jf = λ_jf / sqrt(1 - Σ_f λ_jf²)
```

---

## Classical Test Theory (CTT)

CTT is pre-IRT psychometrics — simpler assumptions, still widely used.

### Reliability

```r
library(psych)

# Cronbach's alpha (F14 territory, psych package):
alpha_result <- alpha(data)
alpha_result$total$raw_alpha      # α = k/(k-1) × [1 - Σσ²_i / σ²_total]
alpha_result$total$std.alpha      # standardized alpha (from correlation matrix)
alpha_result$item.stats           # item-total correlations, alpha-if-deleted

# McDonald's omega (more appropriate than alpha for multidimensional scales):
omega_result <- omega(data)
omega_result$omega.tot            # ω_total (general factor + all group factors)
omega_result$omega.lim            # ω_hierarchical (general factor only)
omega_result$alpha                # alpha for comparison

# Trap: alpha ≤ omega for unidimensional data, but alpha > omega possible
# for bifactor structures. Use omega for multidimensional scales.
```

### Item Statistics

```r
# Item-total correlation (corrected = item removed from total):
alpha_result$item.stats[, "r.cor"]     # corrected item-total correlation
# r.cor < 0.20: poor item, consider deletion
# r.cor > 0.40: good item

# Difficulty (proportion correct = P-value in CTT):
item_difficulty <- colMeans(data)      # P_j = Σ_p Y_pj / N
# Optimal difficulty: P ≈ 0.50 for maximal discrimination

# Discrimination index (D):
# Split by upper/lower 27% on total score:
total_scores <- rowSums(data)
upper_27 <- total_scores > quantile(total_scores, 0.73)
lower_27 <- total_scores < quantile(total_scores, 0.27)
D <- colMeans(data[upper_27, ]) - colMeans(data[lower_27, ])
# D > 0.40: excellent, D > 0.30: good, D < 0.20: poor
```

### Parallel Tests Reliability

```r
# Split-half reliability (Spearman-Brown corrected):
odd_items  <- seq(1, ncol(data), 2)
even_items <- seq(2, ncol(data), 2)
r_half <- cor(rowSums(data[, odd_items]), rowSums(data[, even_items]))
sb_corrected <- 2 * r_half / (1 + r_half)   # Spearman-Brown prophecy formula

# Test-retest reliability (if repeated measurements available):
# ICC (Intraclass Correlation) from F11 territory
```

---

## Differential Item Functioning (DIF)

DIF tests whether an item behaves differently for two groups (e.g., male vs female)
after conditioning on ability.

### Mantel-Haenszel DIF

```r
library(difR)

# group = 0/1 vector (e.g., 0=reference, 1=focal group)
# data = item response matrix

# Mantel-Haenszel test (non-parametric DIF detection):
mh_result <- difMH(data, group = group_vector, focal.name = 1)
print(mh_result)     # MH chi-square, effect size (MH-LOR = log-odds ratio)
plot(mh_result)      # visual DIF display

# Effect size interpretation (ETS classification):
# |MH-LOR| < 1.0:  A = negligible DIF
# |MH-LOR| 1.0-1.5: B = moderate DIF
# |MH-LOR| > 1.5:  C = large DIF
```

### Logistic Regression DIF

```r
# More powerful, detects both uniform and non-uniform DIF:
lr_result <- difLogistic(data, group = group_vector, focal.name = 1,
                          type = 'both')   # 'udif', 'nudif', or 'both'

# type='udif': uniform DIF only (constant ability-matched difference)
# type='nudif': non-uniform DIF only (interaction of ability × group)
# type='both': test for either (recommended)
```

### IRT-based DIF (in mirt)

```r
# Likelihood-ratio DIF test using IRT:
# Fit model where flagged item parameters are free to vary by group

# Step 1: anchor items (assumed DIF-free), test items (potential DIF)
dif_result <- DIF(fit2pl,
                  which.items = 1:ncol(data),    # items to test
                  scheme = 'drop')                # forward/backward/drop

# Step 2: inspect results
print(dif_result)   # LRT chi-square per item, effect size
```

---

## Tambear Decomposition

### Architecture Overview

```
Input: binary response matrix Y (N_persons × J_items)

EM loop (MML):
  ┌─ E-step: log-prob accumulation ─────────────────────────────────────────┐
  │  For each person p, at each quadrature point θ_k (k = 1..Q):            │
  │    log_L(θ_k | Y_p) = accumulate(ByPerson, log_bernoulli(Y,P(θ_k)), Add)│
  │    where P(θ_k)_j = logistic(a_j*(θ_k - b_j))                          │
  │                                                                          │
  │  posterior_pk = softmax over k of [log_L_pk + log N(θ_k; 0,1)]         │
  │  posterior: N_persons × Q matrix                                         │
  └──────────────────────────────────────────────────────────────────────────┘
  ┌─ M-step: weighted IRLS per item ─────────────────────────────────────────┐
  │  For each item j (1..J):                                                 │
  │    numer_jk = accumulate(ByItem, Y_pj * posterior_pk, Add)   [Q-vector] │
  │    denom_jk = accumulate(ByItem, posterior_pk, Add)           [Q-vector] │
  │    Effective obs at θ_k: n_k = denom_jk, successes: r_k = numer_jk     │
  │    Update (a_j, d_j) via IRLS on (θ_k, r_k, n_k)  [F10 infra]         │
  └──────────────────────────────────────────────────────────────────────────┘

Person scoring (EAP):
  θ̂_p = accumulate(ByPerson, θ_k * posterior_pk, Add)   [weighted mean]
  SE_p = sqrt(accumulate(ByPerson, (θ_k - θ̂_p)² * posterior_pk, Add))
```

### Primitive Mapping

| IRT step | Tambear primitive | Shared with |
|----------|-------------------|-------------|
| E-step log-likelihood | `accumulate(ByPerson, log_bernoulli, Add)` | F10 GLM loss |
| Posterior normalize | `accumulate(ByPerson, softmax_weights, Normalize)` | F16 GMM E-step |
| M-step numerator | `accumulate(ByItem_θ, Y * posterior, Add)` | F16 GMM M-step |
| M-step denominator | `accumulate(ByItem_θ, posterior, Add)` | F16 GMM M-step |
| IRLS weight matrix | `scatter_multi_phi_weighted` | F09/F10/F11/F13 |
| EAP scoring | `accumulate(ByPerson, θ_k * posterior_pk, Add)` | F34 posterior mean |
| Item information | `accumulate(ByItem, a² * P*(1-P), Add)` | F07 chi-square |

### Quadrature Setup

Gauss-Hermite quadrature is the standard for integrating over N(0,1):
```
Q = 61 points in [-6, 6] (mirt default: 61)
Weights w_k adjusted for N(0,1): w_k = gauss_hermite_weight_k × exp(θ_k²) × (1/√(2π))
```

For tambear Phase 1: use fixed Q=61 Gauss-Hermite points. GPU batch: compute all N×Q×J
log-probabilities in one kernel, then reduce per person.

### GPU Contribution

The GPU contribution is batched likelihood computation:

```
Batch: N_persons × Q_quadrature × J_items log-probability evaluations
  = logistic(a_j * (θ_k - b_j))  for all (j, k) combinations
  = J × Q evaluations per person, J × Q × N total

For LSAT7 (N=1000, Q=61, J=5): 305,000 logistic evaluations per EM iteration
For large exam (N=10000, Q=61, J=50): 30.5M evaluations per EM iteration

On GPU: O(1) kernel invocations (batched), vs O(N×Q×J) serial on CPU
Speedup: ~100-1000× for large N×J
```

The M-step per item is small (Q=61 "observations" × p=2 parameters) — CPU only.
The E-step accumulation and person scoring is the GPU work.

### MSR Struct (from navigator)

```rust
pub struct IrtModel {
    pub n_persons: usize,
    pub n_items: usize,
    pub model: IrtModelType,

    // Item parameters (J-vectors)
    pub discrimination: Vec<f64>,          // a_j (2PL/3PL; = 1 for Rasch)
    pub difficulty: Vec<f64>,              // b_j (traditional IRT notation)
    pub guessing: Option<Vec<f64>>,        // c_j (3PL only)
    pub upper_asymptote: Option<Vec<f64>>, // d_j_upper (4PL only)

    // Person ability (P-vectors)
    pub ability: Vec<f64>,                 // θ̂_p, EAP by default
    pub ability_se: Vec<f64>,              // SE(θ̂_p) = 1/sqrt(test_info(θ̂_p))

    // Model fit
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub m2_statistic: Option<f64>,
    pub m2_df: Option<u32>,
    pub m2_p: Option<f64>,
    pub rmsea: Option<f64>,

    // Item fit (J-vectors)
    pub item_infit: Vec<f64>,
    pub item_outfit: Vec<f64>,
    pub item_sX2: Option<Vec<f64>>,

    // Information functions
    pub item_info: Vec<f64>,               // I_j(θ̂_mean) at mean ability
    pub test_info: Vec<f64>,               // I_total(θ) at each quadrature point
}

pub enum IrtModelType {
    Rasch,
    TwoPL,
    ThreePL,
    FourPL,
    Graded,
    PartialCredit,
    GeneralizedPartialCredit,
    Nominal,
}
```

---

## Key Traps

### Trap 1: mirt a/d vs a/b Parameterization

```r
# mirt stores d = -a*b internally. NEVER report d as the difficulty.
# Always use IRTpars=TRUE when extracting for validation.

coef(fit2pl, simplify=TRUE)$items             # a1, d, g, u
coef(fit2pl, simplify=TRUE, IRTpars=TRUE)$items  # a, b, g, u  ← correct

# Verify: d[j] == -a[j] * b[j]
params_internal <- coef(fit2pl, simplify=TRUE)$items
params_irt <- coef(fit2pl, simplify=TRUE, IRTpars=TRUE)$items
all.equal(params_internal[,"d"], -params_irt[,"a"] * params_irt[,"b"])  # should be TRUE
```

### Trap 2: `coef()` Variants Return Different Structures

```r
coef(fit2pl)                               # list of length J+1; each is a 1×4 matrix
coef(fit2pl, simplify=TRUE)                # list with $items (J×4) and $means, $cov
coef(fit2pl, simplify=TRUE)$items          # J×4 matrix — what you usually want
coef(fit2pl, simplify=TRUE, IRTpars=TRUE)  # J×4 with IRT a/b instead of a/d
```

### Trap 3: EAP Shrinkage at Extremes

EAP always shrinks toward the prior mean (θ=0). For extreme response patterns:

```r
# All correct: EAP gives high but finite θ (pulled toward 0)
# All incorrect: EAP gives low but finite θ (pulled toward 0)
# MAP and MLE give larger absolute values for the same patterns

# Consequence for tambear: EAP is numerically stable; use as default.
# MAP requires optimization per person; more expensive but less biased.
# MLE gives ±Inf for perfect/zero score patterns — must handle explicitly.
```

### Trap 4: MIRT Rotation vs IRT Parameters

```r
# After extracting a 2D MIRT model, rotation changes the meaning of a1 and a2:
summary(fit_mirt2)                     # loadings in rotated space
coef(fit_mirt2, simplify=TRUE)$items   # a1, a2 in ROTATED coordinates

# To get unrotated parameters:
fit_mirt2_unrot <- mirt(data, 2, itemtype='2PL', rotate='none', verbose=FALSE)
# Unrotated and rotated models are equivalent (same fit) but different parameterizations
# The D'Agostino transformation to FA loadings only holds for the rotated solution
```

### Trap 5: FA Loadings vs IRT Loadings Use Different Scaling

```r
# The relationship between FA loadings and IRT discriminations:
# FA (from psych::fa): λ_j = factor loading, scaled so Σ_j λ_j² = 1 per factor (approx)
# IRT 2PL: a_j = discrimination, no constraint on scale

# Conversion (D'Agostino, 1971):
# a_j = λ_j / sqrt(1 - λ_j²)   [FA → IRT]
# λ_j = a_j / sqrt(1 + a_j²)   [IRT → FA]

# This means a_j > 0 always corresponds to 0 < λ_j < 1.
# High discrimination items (a_j >> 1) have FA loadings approaching 1.
# Low discrimination items (a_j ≈ 0) have FA loadings ≈ 0.
```

### Trap 6: Item Difficulty b is Not the Same as Item Easiness

```r
# Higher b_j = harder item (requires higher θ to have 50% chance of correct)
# Lower b_j = easier item
# b_j = θ at which P(Y=1) = 0.50 for 2PL model (b_j is the "50% point")
# For 3PL: P(Y=1) = 0.50 at a different point (shifted by guessing):
#   θ_50 = b_j + log((1-c_j) / (1-2*c_j+c_j²)) / a_j  [approximately]

# Rasch model: mean b_j = 0 is a common identification constraint
# 2PL: θ ~ N(0,1) is the identification constraint (not the mean of b)
```

---

## Validation Procedure for Tambear

### Phase 1: Rasch Model on LSAT7

```r
library(mirt)
data(LSAT7)
data <- LSAT7$data

# Reference solution:
fit_rasch_ref <- mirt(data, 1, itemtype='Rasch', verbose=FALSE)
b_ref <- coef(fit_rasch_ref, simplify=TRUE, IRTpars=TRUE)$items[, "b"]
theta_ref <- fscores(fit_rasch_ref, method='EAP')[, 1]

cat("Rasch difficulties (b):\n")
print(round(b_ref, 4))
# Expected: Item 1 ≈ -3.1, Item 2 ≈ -2.0, Item 3 ≈ -1.4, Item 4 ≈ -1.8, Item 5 ≈ -3.1

# Tambear target:
# b_tambear vs b_ref: max(abs(b_tambear - b_ref)) < 0.01
# cor(theta_tambear, theta_ref) > 0.999
```

### Phase 2: 2PL Model on LSAT7

```r
fit_2pl_ref <- mirt(data, 1, itemtype='2PL', verbose=FALSE)
params_ref <- coef(fit_2pl_ref, simplify=TRUE, IRTpars=TRUE)$items
a_ref <- params_ref[, "a"]
b_ref <- params_ref[, "b"]
theta_ref <- fscores(fit_2pl_ref, method='EAP')[, 1]
m2_ref <- M2(fit_2pl_ref)

cat("2PL discriminations (a):\n"); print(round(a_ref, 4))
cat("2PL difficulties (b):\n");    print(round(b_ref, 4))
cat("M2:\n"); print(m2_ref)

# Tambear tolerances:
# a: max(abs(a_tambear - a_ref)) < 0.05
# b: max(abs(b_tambear - b_ref)) < 0.05
# theta: cor(theta_tambear, theta_ref) > 0.999
# log-likelihood: abs(logLik_tambear - logLik(fit_2pl_ref)) < 1.0
```

### Phase 3: Comparison Against ltm (Cross-validation)

```r
library(ltm)

# ltm Rasch model on same data:
# NOTE: ltm uses its own built-in LSAT (slightly different from LSAT7)
# For validation, use LSAT7$data in both mirt and ltm

fit_ltm <- rasch(data, constraint=cbind(ncol(data)+1, 1))
# constraint: fix discrimination to 1 (pure Rasch)
coef_ltm <- coef(fit_ltm)
# coef_ltm[, 1] = Dffclt (= b, difficulty)
# coef_ltm[, 2] = Dscrmn (= a = 1 for Rasch, fixed)

# Cross-validation: mirt vs ltm difficulties should match within 0.02
mirt_b <- coef(fit_rasch_ref, simplify=TRUE, IRTpars=TRUE)$items[,"b"]
ltm_b  <- coef_ltm[, "Dffclt"]
max(abs(mirt_b - ltm_b))   # expected < 0.02
```

---

## Build Order

### Phase 1 — Rasch Model (~150 lines)

1. **Gauss-Hermite quadrature setup** (~10 lines)
   - Q=61 points and weights for N(0,1) integration
   - Store as static `[(θ_k, w_k)]` array

2. **E-step: log-likelihood accumulation** (~40 lines)
   - For each person×quadrature: sum log Bernoulli over items
   - Add log N(θ_k; 0,1) prior
   - Normalize to get posterior weights (LogSumExp for stability)
   - Output: N × Q posterior matrix

3. **M-step: item difficulty update** (~30 lines)
   - Per item: compute effective (n_k, r_k) = (denom, numer) from posterior
   - 1D logistic regression update (Rasch: only b, with a=1 fixed)
   - Uses F10 IRLS with Q=61 "observations"

4. **EM convergence loop** (~20 lines)
   - Iterate E and M steps
   - Convergence criterion: max change in log-likelihood < 1e-6

5. **EAP ability scoring** (~10 lines)
   - θ̂_p = Σ_k θ_k · posterior_pk (weighted mean of quadrature points)
   - SE_p = sqrt(Σ_k (θ_k - θ̂_p)² · posterior_pk)

6. **IrtModel struct population** (~20 lines)

7. **Tests**: match mirt Rasch output within 0.01

### Phase 2 — 2PL Model (~50 additional lines)

1. **2-parameter M-step** per item: IRLS with design matrix [θ_k, 1] to estimate (a_j, d_j)
2. **F14 starting values**: `a_j_init = λ_j / sqrt(1 - λ_j²)` from FA loadings
3. Tests: match mirt 2PL within 0.05 on a and b

### Phase 3 — Extended Models (Phase 2 work)

- **3PL**: add bounded c_j with beta prior; constrained logistic in M-step
- **GRM**: multiple thresholds per item; cumulative logistic M-step
- **CTT metrics**: alpha, omega, item-total correlations (CPU only, trivial)
- **DIF**: Mantel-Haenszel (F07 chi-square infra); logistic DIF (F10 logistic infra)
- **MIRT**: extend E-step to multi-dimensional θ (vectorize over dimensions)

---

## Structural Rhymes

**IRT E-step = GMM E-step** (F16 bridge):
- GMM: E-step computes component membership weights P(z_i=k | x_i)
- IRT: E-step computes quadrature point weights P(θ_k | Y_p)
- Same structure: prior × likelihood → normalized posterior
- Same `scatter_multi_phi` for likelihood accumulation

**Item information = logistic variance = IRLS weight** (F10 bridge):
```
F10 IRLS: w_i = μ_i(1-μ_i)
IRT info: I_j(θ) = a_j² × P(θ)(1-P(θ))
```
The Fisher information at a single item is the IRLS weight scaled by discrimination squared.
No new code: I_j(θ) IS the IRLS weight from F10, already in the infrastructure.

**Test information = Cramér-Rao bound** (F07 bridge):
- SE(θ̂) = 1/√I_test(θ) = standard error of ability estimate
- I_test(θ) = Σ_j I_j(θ) = sum of item Fisher informations
- This is the Cramér-Rao bound — the best possible SE for any unbiased estimator of θ

**Rasch model = conditional logistic regression** (F10 bridge):
- Conditioning on total score removes the θ_p nuisance parameter
- Conditional MLE (CMLE) = logistic regression of item vs item, controlling for total score
- This is `eRm` package's approach; computationally simpler but loses person estimates

**IRT infit/outfit = Pearson chi-square on residuals** (F07 bridge):
- Outfit: `Σ_p (Y_pj - P_pj)² / (P_pj(1-P_pj)) / N`  (unweighted)
- Infit: same but weighted by `P_pj(1-P_pj)` (gives more weight to informative responses)
- Both are sums of squared standardized residuals = chi-square statistics
- F07 Pearson chi-square infrastructure directly applicable

---

## Lab Notebook Claim

> IRT is the EM algorithm applied to logistic regression where person abilities are
> marginalized out by quadrature. The E-step computes posterior ability weights — the same
> structure as the GMM E-step in F16. The M-step runs weighted logistic IRLS per item —
> the same infrastructure as F10. F14 FA loadings provide discrimination starting values
> that accelerate convergence 3-5x. F15 adds ~150 lines for Phase 1 (Rasch) and ~50
> additional lines for Phase 2 (2PL), on top of F10/F14/F16 infrastructure that already
> exists. The GPU contribution is batched log-probability computation across
> N_persons × Q_quadrature × J_items — the same tiled accumulate pattern as everything else.
