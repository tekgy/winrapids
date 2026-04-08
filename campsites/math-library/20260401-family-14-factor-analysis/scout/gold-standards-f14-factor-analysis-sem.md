# F14 Factor Analysis & SEM — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 14 (Factor Analysis & SEM).
Prerequisites: F10 GramMatrix (covariance/correlation matrix), F22 EigenDecomposition.
F14 is the Mplus replacement — the most complex consumer of the GramMatrix path.

---

## Two Distinct Sub-families

### Exploratory Factor Analysis (EFA)
- Extracts latent factors from correlations
- Rotation-free or with oblique/orthogonal rotation
- Gold standard: R `psych` package, Python `factor_analyzer`

### Structural Equation Modeling (SEM)
- Specifies latent variable structure a priori
- Tests fit of hypothesized model against covariance matrix
- Gold standard: R `lavaan` package (academic standard), Python `semopy`

---

## The GramMatrix Connection

Both EFA and CFA/SEM work on the **sample covariance or correlation matrix** S.
S is exactly the GramMatrix from F10 (centered, normalized by n-1 or n).

```
S = X_c' X_c / (n-1)      [sample covariance]
R = D^{-1/2} S D^{-1/2}   [correlation matrix, D = diag of variances]
```

These are already computed if F10's GramMatrix is in TamSession.
F14 starts from S or R — it does NOT re-scan the data.

---

## Exploratory Factor Analysis (EFA)

### The Core: Factor Model

```
x = Λ · f + ε
where:
  Λ = p×k loading matrix (what we want)
  f = k×1 latent factors (unobserved)
  ε = p×1 unique factors (measurement error)

Implied covariance: Σ = Λ Φ Λ' + Ψ
where Φ = factor covariance, Ψ = diag(unique variances)
```

EFA estimates Λ and Ψ such that the implied Σ matches the sample S.

### Extraction Methods

```r
library(psych)

# Principal Axis Factoring (PAF — most common in social science):
fa(R, nfactors=3, fm="pa", rotate="oblimin")

# Maximum Likelihood (ML — for confirmatory or model fit tests):
fa(R, nfactors=3, fm="ml", rotate="oblimin")

# Minimum Residual (minres — default in psych):
fa(R, nfactors=3, fm="minres", rotate="oblimin")  # minimizes sum of squared residuals

# Principal Components (technically not FA — extracts from full variance):
principal(R, nfactors=3, rotate="varimax")

# From raw data:
fa(X_data, nfactors=3, fm="ml", rotate="oblimin")
```

### Output

```r
fit <- fa(R, nfactors=3, fm="ml", rotate="oblimin")
fit$loadings         # Λ matrix: p×k factor loadings
fit$communalities    # h² = Σ_k Λ_{jk}² (variance explained per variable)
fit$uniquenesses     # u² = 1 - h² (unique variance per variable)
fit$factors          # k factor names
fit$Phi              # factor correlation matrix (if oblique rotation)
fit$RMSEA            # root mean square error of approximation (model fit)
fit$TLI              # Tucker-Lewis index
fit$BIC              # Bayesian Information Criterion
```

### Rotation Methods

```r
# Orthogonal (factors uncorrelated):
fa(R, nfactors=3, rotate="varimax")   # maximize variance of squared loadings per factor
fa(R, nfactors=3, rotate="quartimax") # maximize variable-level variance

# Oblique (factors may be correlated — more common in practice):
fa(R, nfactors=3, rotate="oblimin")   # minimize correlation between factors
fa(R, nfactors=3, rotate="promax")    # oblique, starts from varimax
fa(R, nfactors=3, rotate="none")      # no rotation
```

**Trap**: oblique rotation gives loadings AND a factor correlation matrix Φ.
The "pattern matrix" (loadings) ≠ "structure matrix" (pattern × Φ).
Python's factor_analyzer gives pattern matrix; R's psych gives both.

### Python: factor_analyzer

```python
from factor_analyzer import FactorAnalyzer

fa = FactorAnalyzer(n_factors=3, method='ml', rotation='oblimin')
fa.fit(X)

fa.loadings_           # Λ matrix (p × k)
fa.get_communalities() # h² per variable
fa.get_eigenvalues()   # eigenvalues of correlation matrix (for scree plot)
fa.get_factor_variance()  # (SS loadings, proportional var, cumulative var) per factor

# Determine number of factors:
fa_none = FactorAnalyzer(n_factors=X.shape[1], rotation=None)
fa_none.fit(X)
ev, v = fa_none.get_eigenvalues()
# Kaiser criterion: keep factors with eigenvalue > 1
n_factors = (ev > 1).sum()
```

### How Many Factors? Methods

```r
# 1. Kaiser criterion: eigenvalues > 1 (often overestimates)
library(psych)
VSS.scree(R)   # scree plot

# 2. Parallel analysis (gold standard):
fa.parallel(R, fm="ml")
# Compares observed eigenvalues against random data eigenvalues

# 3. Very Simple Structure (VSS):
VSS(R, n=8, fm="ml")  # VSS criterion per number of factors

# 4. MAP (Minimum Average Partial):
MAP(R)
```

**Best practice**: parallel analysis is the current gold standard for number of factors.

---

## Confirmatory Factor Analysis (CFA) with lavaan

CFA specifies the factor structure a priori and tests fit.

```r
library(lavaan)

# CFA model specification:
model <- '
  # Measurement model (CFA)
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
'

fit <- cfa(model, data=HolzingerSwineford1939)
summary(fit, fit.measures=TRUE, standardized=TRUE)

# Key output:
parameterEstimates(fit)           # loadings, variances, covariances
lavaan::fitMeasures(fit, c("rmsea", "cfi", "tli", "srmr"))  # fit indices
lavaan::modificationIndices(fit)  # suggested model modifications
lavaan::lavResiduals(fit)         # residuals between S and Σ_implied

# Model fit indices:
# RMSEA < 0.06: excellent, < 0.08: acceptable, > 0.10: poor
# CFI > 0.95: excellent, > 0.90: acceptable
# SRMR < 0.08: acceptable
```

### Python: semopy

```python
from semopy import Model

model_text = '''
visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9
'''

model = Model(model_text)
model.fit(data)

model.inspect()          # parameter estimates
model.calc_stats()       # fit indices
model.sem_res            # access internal optimization results
```

---

## Full SEM with lavaan

SEM adds structural paths between latent variables (regression among factors).

```r
model_sem <- '
  # Measurement model
  ind60 =~ x1 + x2 + x3
  dem60 =~ y1 + y2 + y3 + y4
  dem65 =~ y5 + y6 + y7 + y8

  # Structural model (regression between latents)
  dem60 ~ ind60
  dem65 ~ ind60 + dem60

  # Covariances
  y1 ~~ y5
  y2 ~~ y4
'

fit <- sem(model_sem, data=PoliticalDemocracy)
```

**SEM = CFA (measurement model) + path analysis (structural model)**

---

## Fit Indices: What to Match

The gold standard fit indices (lavaan as oracle):

| Index | Formula | Good | Poor |
|-------|---------|------|------|
| RMSEA | sqrt((χ²/df - 1)/(n-1)) | < 0.06 | > 0.10 |
| CFI | 1 - (χ²_model - df_model) / (χ²_null - df_null) | > 0.95 | < 0.90 |
| TLI | ((χ²_null/df_null) - (χ²_model/df_model)) / (χ²_null/df_null - 1) | > 0.95 | < 0.90 |
| SRMR | sqrt(Σ(s_ij - σ̂_ij)²/(p(p+1)/2)) | < 0.08 | > 0.10 |

All indices derive from: sample covariance S, model-implied covariance Σ̂, and degrees of freedom.
χ² = (n-1) · discrepancy(S, Σ̂) where discrepancy is a matrix distance measure.

---

## Tambear Decomposition

```
Input: TamSession → GramMatrix → S (sample covariance)

EFA pipeline:
1. S from GramMatrix (already computed if F10 exists in session)
2. R = standardize S to correlation matrix (CPU, p×p)
3. Extract: eigendecomposition of R → initial factor loadings (F22 primitive)
4. Rotation: Varimax = iterative algorithm on p×k loadings matrix (CPU)
5. Communalities: Σ_k Λ_{jk}² (CPU, trivial)

CFA pipeline:
1. S from GramMatrix (same as above)
2. Model specification → generate Σ̂(θ) as function of parameters
3. Minimize discrepancy(S, Σ̂(θ)) via optimization (F05 optimizer)
4. Gradient: ∂discrepancy/∂θ involves Σ̂(θ)^{-1} · S (matrix algebra, CPU)

The GPU work: ONE tiled accumulate to compute S (= GramMatrix of centered X).
Everything else is CPU-side p×p linear algebra.
```

**Key insight**: for p < 1000 (features), F14 is almost entirely CPU-side matrix algebra on a p×p matrix. The GPU contribution is in computing the n→p compression (GramMatrix). Once that's done, SEM/CFA iterates on the p×p problem.

---

## Validation: Holzinger-Swineford Dataset (Standard FA Test Case)

```r
library(lavaan)
library(psych)

# Classic dataset:
data(HolzingerSwineford1939)  # 9 cognitive tests, 301 students

# 3-factor CFA (canonical model):
model <- '
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
'
fit <- cfa(model, data=HolzingerSwineford1939)
fitMeasures(fit, c("rmsea", "cfi", "srmr"))
# RMSEA ≈ 0.092, CFI ≈ 0.931, SRMR ≈ 0.065 (from lavaan documentation)
```

```python
# Python equivalent using factor_analyzer:
import pandas as pd
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser

# model specification...
```

**The Holzinger-Swineford fit indices** are well-documented in the lavaan documentation and SEM textbooks. These can serve as tambear validation targets.

---

## Estimation Methods and Oracle Matching

| Method | Gold standard | Objective minimized |
|--------|--------------|-------------------|
| ML | `lavaan(estimator="ML")` | log(|Σ̂|) + tr(SΣ̂^{-1}) - log(|S|) - p |
| GLS | `lavaan(estimator="GLS")` | tr((S - Σ̂)²) normalized |
| ULS | `lavaan(estimator="ULS")` | sum of squared residuals |
| WLS | `lavaan(estimator="WLS")` | weighted residuals |
| DWLS | `lavaan(estimator="DWLS")` | diagonal WLS (for ordinal data) |

For tambear Phase 1: ML estimation only. GLS is simpler (no log det) but less common.
