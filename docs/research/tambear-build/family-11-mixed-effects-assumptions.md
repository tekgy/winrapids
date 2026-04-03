# Family 11: Mixed Effects & Multilevel Models — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (Henderson equations = GramMatrix([X|Z]) + regularization) + C (variance component estimation = iterative EM/Newton)

---

## Core Insight: LME = Self-Tuning Ridge Regression

Henderson's mixed model equations are EXACTLY the normal equations for Ridge regression on the augmented design matrix [X|Z], where the regularization parameter λ = σ²/σ_u² is learned from the data. The GramMatrix machinery from F10 does all the heavy lifting.

---

## 1. Linear Mixed Effects Model (LME)

### Model
```
y = Xβ + Zu + ε
```
where:
- X (n × p): fixed effects design matrix
- Z (n × q): random effects design matrix
- β (p × 1): fixed effects coefficients
- u (q × 1): random effects, u ~ N(0, G)
- ε (n × 1): residuals, ε ~ N(0, R)

Typically R = σ²I_n (independent errors) and G = block-diagonal with variance components.

### Marginal Model
```
y ~ N(Xβ, V)    where V = ZGZ' + R
```

### Henderson's Mixed Model Equations
The joint estimator (β̂, û) satisfies:
```
[X'R⁻¹X    X'R⁻¹Z  ] [β̂]   [X'R⁻¹y]
[Z'R⁻¹X    Z'R⁻¹Z+G⁻¹] [û ] = [Z'R⁻¹y]
```

For R = σ²I:
```
[X'X      X'Z    ] [β̂]   [X'y]
[Z'X    Z'Z+σ²G⁻¹] [û ] = [Z'y]
```

### THIS IS GramMatrix([X|Z]) + Diagonal Regularization

Let W = [X | Z] (n × (p+q) augmented design matrix). Then:
```
W'W = [X'X   X'Z]
      [Z'X   Z'Z]
```
Henderson's equations = W'W with σ²G⁻¹ added to the (2,2) block.

**Implementation**: tiled_accumulate of [X|Z] gives W'W. Add regularization to bottom-right block. Cholesky solve. Done.

### BLUE and BLUP
- β̂ = BLUE (Best Linear Unbiased Estimator) of β
- û = BLUP (Best Linear Unbiased Predictor) of u
- BLUP shrinks random effects toward zero (like Ridge). Shrinkage = σ²/(σ² + σ_u²·n_g) per group.

### Why "Self-Tuning Ridge"
Ridge regression: (X'X + λI)β̂ = X'y. Here: [W'W + Λ][β̂; û] = W'y where Λ = diag(0_p, σ²G⁻¹).
The variance components σ², G determine the regularization strength. REML estimates them from the data → the ridge penalty tunes itself.

---

## 2. Variance Component Estimation

### 2a. Maximum Likelihood (ML)
Maximize:
```
log L(β, σ², G) = -n/2 · log(2π) - ½ log|V| - ½(y-Xβ)'V⁻¹(y-Xβ)
```
where V = ZGZ' + R.

**Problem**: ML estimates of variance components are biased downward (same issue as dividing by n instead of n-1 for sample variance).

### 2b. REML (Restricted/Residual Maximum Likelihood)
Maximize the likelihood of a set of error contrasts (linear combinations of y that don't depend on β):
```
log L_R(σ², G) = -½(n-p)log(2π) - ½log|V| - ½log|X'V⁻¹X| - ½y'Py
```
where P = V⁻¹ - V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹.

**REML is the standard.** Produces unbiased variance estimates. Equivalent to integrating out β (marginalizing over fixed effects).

### 2c. EM Algorithm for Variance Components

E-step: Compute conditional expectations of u given y:
```
û = G·Z'·V⁻¹(y - Xβ̂)
Var(u|y) = G - G·Z'·V⁻¹·Z·G
```

M-step: Update variance components:
```
σ²_new = (1/n)||y - Xβ̂ - Zû||² + σ²·tr(I - σ⁻²Z·Var(u|y)·Z')/n
G_new = ûû' + Var(u|y)    (for single random effect component)
```

**Properties**: Monotone convergence (log-likelihood never decreases). Very slow convergence near boundary (σ² → 0). Simple but reliable.

### 2d. Newton-Raphson / Fisher Scoring for REML
Score vector:
```
∂L_R/∂θ_i = -½ tr(P·∂V/∂θ_i) + ½ y'P(∂V/∂θ_i)Py
```

Expected Fisher information:
```
I(θ)_{ij} = ½ tr(P·∂V/∂θ_i · P · ∂V/∂θ_j)
```

Update: θ_{new} = θ_{old} + I(θ)⁻¹ · score(θ).

**Faster convergence than EM** (quadratic near optimum) but can diverge or produce negative variance estimates.

### 2e. Practical Strategy
1. Start with EM (5-10 iterations) to get into the right neighborhood
2. Switch to Newton/Fisher scoring for fast convergence
3. If Newton produces negative variance → clamp to 0 (boundary estimate)

### Kingdom: Variance estimation = C (iterative). Each iteration's linear solve = A (GramMatrix + Cholesky).

---

## 3. Random Effects Structures

### 3a. Random Intercept
```
y_{ij} = x'_{ij}β + u_j + ε_{ij},    u_j ~ N(0, σ²_u)
```
Z = indicator matrix (columns = groups). G = σ²_u · I_J where J = number of groups.

### 3b. Random Intercept + Slope
```
y_{ij} = (β₀ + u_{0j}) + (β₁ + u_{1j})x_{ij} + ε_{ij}
[u_{0j}, u_{1j}]' ~ N(0, G_2)    where G_2 = [σ²₀  σ₀₁; σ₀₁  σ²₁]
```

### 3c. Covariance Structures for G

| Structure | Parameters | Form |
|-----------|-----------|------|
| Diagonal | q | σ²_k for each random effect |
| Unstructured | q(q+1)/2 | Full positive-definite matrix |
| Compound Symmetry | 2 | σ² + ρσ² for all pairs |
| AR(1) | 2 | σ²ρ^{\|i-j\|} |
| Toeplitz | q | Different parameter for each lag |

### 3d. Crossed vs Nested Random Effects
**Nested**: students within schools. Z is block-diagonal.
**Crossed**: items × subjects (psychometrics). Z is NOT block-diagonal. Sparse.

**CRITICAL**: Crossed random effects make Z'Z dense. Use sparse matrix operations or the Cholesky parameterization.

---

## 4. Cholesky Parameterization of G

Parameterize G = LL' where L is lower-triangular with positive diagonal. This:
1. Guarantees G is positive-definite by construction
2. Avoids constrained optimization
3. Is how lme4 (standard R package) works internally

The variance components become: θ = vec(L) / σ (relative Cholesky factor).

### Profiled Likelihood
Fix θ, solve Henderson's equations for β̂ and û, then:
```
σ²(θ) = ||y - Xβ̂ - Zû||² / n
```
Optimize only over θ (lower-dimensional). This is "profiling out" β and σ².

---

## 5. GLMM (Generalized Linear Mixed Models)

### Model
```
g(E[y|u]) = Xβ + Zu
```
where g = link function. y|u follows exponential family (Bernoulli, Poisson, etc.).

### Estimation Methods

**PQL (Penalized Quasi-Likelihood)**:
Linearize: working response η̃ = g(ŷ) + (y - ŷ)/g'(ŷ). Then fit LME with working response and working weights.
- Fast but biased for binary data with small clusters
- NOT recommended for variance component inference

**Laplace Approximation**:
```
∫ f(y|u)f(u) du ≈ f(y|û)f(û) · (2π)^{q/2} |H|^{-1/2}
```
where H = -∂²log[f(y|u)f(u)]/∂u² evaluated at mode û.
- Better than PQL, standard in lme4
- One Newton solve per evaluation

**Adaptive Gauss-Hermite Quadrature (AGHQ)**:
```
∫ f(y|u)f(u) du ≈ Σ_k w_k · f(y|u_k)f(u_k) · |Σ̂|^{1/2}
```
where u_k are quadrature points centered at mode, adapted by Σ̂.
- Most accurate, but O(K^q) where K = number of quadrature points per dimension
- Only practical for q ≤ 3-5 random effect dimensions

---

## 6. Multilevel / Hierarchical Models

### Two-Level Model
Level 1: y_{ij} = β_{0j} + β_{1j}x_{ij} + ε_{ij}
Level 2: β_{0j} = γ₀₀ + γ₀₁w_j + u_{0j}
         β_{1j} = γ₁₀ + γ₁₁w_j + u_{1j}

Combined: y_{ij} = γ₀₀ + γ₁₀x_{ij} + γ₀₁w_j + γ₁₁w_jx_{ij} + u_{0j} + u_{1j}x_{ij} + ε_{ij}

This IS an LME with:
- Fixed: X = [1, x, w, w·x]
- Random: Z = [1, x] (per group)

### ICC (Intraclass Correlation Coefficient)
```
ICC = σ²_u / (σ²_u + σ²_ε)
```
Proportion of variance attributable to grouping. ICC = 0 → no group effect → OLS is fine.

### Centering
- **Grand-mean centering**: x - x̄ (overall mean). Fixed effects = average relationship.
- **Group-mean centering**: x - x̄_j (group mean). Separates within- and between-group effects.

**CRITICAL**: Choice of centering changes interpretation of random intercept variance. Group-mean centering decomposes within/between effects cleanly; grand-mean centering conflates them.

### Design Effect
```
DEFF = 1 + (n̄_cluster - 1) · ICC
```
Effective sample size = n / DEFF. Ignoring clustering inflates significance.

---

## 7. Model Diagnostics

### Likelihood Ratio Test for Random Effects
Compare nested models: -2(log L_reduced - log L_full) ~ χ²(df).

**CRITICAL**: When testing σ²_u = 0, the parameter is on the boundary. The LRT statistic follows a mixture: ½χ²(0) + ½χ²(1), NOT χ²(1). The p-value from standard χ² is conservative by factor 2. Correct: p = ½ · P(χ²(1) > LRT).

### Information Criteria
- AIC = -2·log L + 2k
- BIC = -2·log L + k·log(n)

**CRITICAL**: Use REML log-likelihood ONLY when comparing models with same fixed effects. Use ML log-likelihood when comparing models with different fixed effects.

### Residual Types
- **Marginal**: y - Xβ̂ (ignoring random effects)
- **Conditional**: y - Xβ̂ - Zû (including random effects)
- **Pearson**: conditional residuals scaled by √Var
- For GLMM: deviance residuals

---

## 8. Computational Considerations

### Sparse Z Matrix
For nested/hierarchical designs, Z is extremely sparse (block structure). Exploit:
- Sparse Cholesky (e.g., CHOLMOD)
- The fill-in pattern of Z'Z + G⁻¹ is predictable from the nesting structure
- For crossed designs: Z'Z can be dense → different strategy needed

### Profiling
Profile out β: given θ (variance parameters), β̂(θ) has closed form (Henderson solve). Optimize only over θ, which is typically low-dimensional (2-10 parameters).

### From GramMatrix
The key matrices are:
```
X'X, X'Z, Z'Z, X'y, Z'y
```
These are ALL subblocks of the GramMatrix of [X | Z | y]:
```
GramMatrix([X|Z|y]) = [X'X   X'Z   X'y]
                       [Z'X   Z'Z   Z'y]
                       [y'X   y'Z   y'y]
```

**One tiled_accumulate of the augmented matrix gives EVERYTHING.** Same sharing pattern as F33 (Multivariate Analysis) and F10 (Regression).

---

## Edge Cases

### Singular G (Boundary Estimates)
When σ²_u → 0 for some random effect component:
- G becomes singular → G⁻¹ undefined
- Solution: use Cholesky parameterization (L → 0 is fine)
- Boundary warnings in lme4: "singular fit" — the model is saying the random effect variance is zero

### Convergence Failures
- Too many variance parameters relative to data → flat likelihood surface
- Near-zero variance components → slow convergence (EM especially)
- Negative definite Hessian at convergence → not at a maximum

### Small Cluster Sizes
- BLUP shrinkage is extreme for small clusters (pulled toward grand mean)
- GLMM with binary data and small clusters: PQL is biased, use Laplace or AGHQ

### Separation in GLMM
Perfect prediction within some groups → infinite random effect. Regularize or use penalized estimation.

---

## Sharing Surface

### Reuse from Other Families
- **F10 (Regression)**: GramMatrix (X'X), Cholesky solve — the core of Henderson's equations
- **F06 (Descriptive)**: Per-group MomentStats for computing group means, ICCs
- **F05 (Optimization)**: Newton-Raphson/L-BFGS for REML optimization of variance components
- **F07 (Hypothesis Testing)**: Likelihood ratio tests, Wald tests
- **F33 (Multivariate Analysis)**: Multivariate LME shares the same GramMatrix subblock extraction

### Consumers of F11
- **F12 (Panel Data)**: Random effects panel model = LME applied to panel data
- **F13 (Survival)**: Frailty models = survival with random effects
- **F15 (IRT)**: IRT models = GLMM with logit link and item/person random effects
- **F16 (Mixture)**: Growth mixture models = LME within latent classes
- **F30 (Spatial)**: Spatial random effects = LME with structured G (spatial covariance)

### Structural Rhymes
- **LME = self-tuning Ridge**: Henderson equations = Ridge normal equations with learned λ
- **BLUP = shrinkage estimator**: same as James-Stein, Empirical Bayes
- **REML = integrating out β**: same philosophy as F34 Bayesian marginalization
- **ICC = signal-to-noise ratio**: same concept as reliability in F15 IRT
- **EM for variance components = F16 EM for mixture models**: same E-step/M-step structure

---

## Implementation Priority

**Phase 1** — Core LME (~90 lines, as noted in task):
1. Henderson's equations solver (GramMatrix([X|Z]) + regularization + Cholesky)
2. BLUE (β̂) and BLUP (û) extraction
3. EM for variance components (σ², σ²_u for random intercept)
4. ICC computation

**Phase 2** — REML + richer structures (~120 lines):
5. REML via profiled likelihood + L-BFGS (from F05)
6. Random intercept + slope (2×2 G matrix)
7. Cholesky parameterization of G
8. Likelihood ratio tests for random effects (with boundary correction)
9. AIC/BIC

**Phase 3** — GLMM (~100 lines):
10. PQL (iterative weighted LME)
11. Laplace approximation
12. AGHQ (for low-dimensional random effects)

**Phase 4** — Diagnostics + extensions (~80 lines):
13. Residual types (marginal, conditional, Pearson)
14. Three-level models
15. Crossed random effects (sparse Z handling)
16. Influence diagnostics

---

## Composability Contract

```toml
[family_11]
name = "Mixed Effects & Multilevel Models"
kingdom = "A (Henderson solve per iteration) + C (variance component estimation loop)"

[family_11.shared_primitives]
henderson_solve = "GramMatrix([X|Z]) + diagonal regularization + Cholesky"
variance_em = "EM algorithm for variance components"
reml = "Profiled REML likelihood + L-BFGS optimization"
blup = "Best Linear Unbiased Prediction (shrinkage estimator)"

[family_11.reuses]
f10_gram_matrix = "X'X, X'Z, Z'Z, X'y, Z'y — all from one tiled_accumulate"
f10_cholesky = "Henderson system solve"
f05_optimizer = "L-BFGS for REML optimization of variance components"
f06_group_stats = "Per-group MomentStats for ICC, group means"
f07_hypothesis = "LRT for random effects, Wald tests for fixed effects"

[family_11.provides]
lme = "Linear mixed effects model fitting"
glmm = "Generalized linear mixed model (Laplace/AGHQ)"
blup = "Shrinkage predictions for random effects"
icc = "Intraclass correlation coefficient"
variance_components = "Estimated σ², G matrix"

[family_11.consumers]
f12_panel = "Random effects panel model"
f13_survival = "Frailty models"
f15_irt = "IRT as GLMM"
f16_mixture = "Growth mixture models"
f30_spatial = "Spatial random effects"

[family_11.session_intermediates]
henderson_system = "HendersonSystem(model_id) — augmented GramMatrix + regularization"
variance_components = "VarComp(model_id) — current σ², G estimates"
blup_values = "BLUP(model_id) — random effect predictions"
profiled_likelihood = "ProfiledREML(model_id, θ) — cached for optimization"
```
