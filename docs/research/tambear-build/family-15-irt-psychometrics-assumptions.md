# Family 15: IRT & Psychometrics — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: C (EM = iterative) using A (GramMatrix-like accumulates in M-step) and B (logistic IRLS)

---

## Core Insight: IRT = Logistic GLMM with Crossed Random Effects

An IRT model is a GLMM (F11) where:
- Observations: binary responses Y_{ij} (person i, item j)
- Fixed effects: item parameters (difficulty, discrimination)
- Random effects: person abilities θ_i
- Link: logit

The EM algorithm: E-step = posterior of θ given responses, M-step = weighted logistic regression per item. Factor Analysis loadings (F14) can initialize.

---

## 1. Rasch Model (1PL)

### Model
```
P(Y_{ij} = 1 | θ_i) = logistic(θ_i - b_j)
                      = 1 / (1 + exp(-(θ_i - b_j)))
```
where θ_i = person ability, b_j = item difficulty.

### Key Properties
- **Specific objectivity**: item difficulty ordering independent of person sample
- **Sufficient statistics**: raw score Σ_j Y_{ij} is sufficient for θ_i
- **Rasch = conditional logistic regression**: can estimate b_j by conditioning on raw score

### Estimation: Conditional Maximum Likelihood (CML)
Condition on raw scores (sufficient statistics for θ):
```
L_C(b | data) = Π_r P(response pattern | raw score = r, b)
```
Eliminates θ from likelihood. Then estimate b by MLE of L_C.

### Joint MLE (JML)
Estimate θ and b simultaneously. BIASED — Neyman-Scott problem (incidental parameters).
Bias: O(1/J) for b̂, O(1/I) for θ̂.

### Marginal MLE (MMLE) — Standard
Integrate out θ:
```
L_M(b) = Π_i ∫ Π_j P(Y_{ij}|θ)^{Y_{ij}} (1-P(Y_{ij}|θ))^{1-Y_{ij}} · f(θ) dθ
```
where f(θ) = N(0,1) typically.

Integral computed via Gauss-Hermite quadrature with Q points:
```
≈ Π_i Σ_{q=1}^{Q} w_q · Π_j P(Y_{ij}|X_q)^{Y_{ij}} (1-P(Y_{ij}|X_q))^{1-Y_{ij}}
```
where X_q = quadrature points, w_q = weights.

---

## 2. Two-Parameter Model (2PL)

### Model
```
P(Y_{ij} = 1 | θ_i) = logistic(a_j(θ_i - b_j))
```
where a_j = discrimination (item slope), b_j = difficulty.

### Relation to Factor Analysis
Under normal ogive (probit instead of logit):
```
P(Y_{ij} = 1 | θ_i) = Φ(a_j·θ_i - a_j·b_j)
```
This is a single-factor model where:
- Factor loading = a_j
- Threshold = a_j·b_j
- Factor = θ_i

**F14 Factor Analysis loadings can initialize 2PL parameters.**

### Identification
Need constraints:
- Fix mean and variance of θ (e.g., θ ~ N(0,1))
- OR fix one item's parameters

---

## 3. Three-Parameter Model (3PL)

### Model
```
P(Y_{ij} = 1 | θ_i) = c_j + (1-c_j) · logistic(a_j(θ_i - b_j))
```
where c_j = guessing parameter (pseudo-chance, lower asymptote).

### CRITICAL: c_j is notoriously hard to estimate
- Requires very large N (> 1000 per item)
- Often fixed: c_j = 1/k for k-choice items
- Bayesian estimation with Beta(5, 17) prior on c_j is common (mean ≈ 0.2 for 5-choice)

### Edge Cases
- c_j > 0.5: model says guessing helps more than knowledge → something is wrong
- c_j = 0: reduces to 2PL
- a_j = 0: item has no discrimination → useless item

---

## 4. Graded Response Model (GRM — Samejima 1969)

### For polytomous items (Likert scales, 0 to K_j categories)

```
P*(Y_{ij} ≥ k | θ_i) = logistic(a_j(θ_i - b_{jk}))    for k = 1, ..., K_j
```
```
P(Y_{ij} = k | θ_i) = P*(Y_{ij} ≥ k | θ_i) - P*(Y_{ij} ≥ k+1 | θ_i)
```
with P*(≥ 0) = 1 and P*(≥ K_j+1) = 0.

### Constraint: b_{j1} < b_{j2} < ... < b_{jK_j} (ordered thresholds)

---

## 5. Partial Credit Model (PCM — Masters 1982)

### Model (generalization of Rasch to polytomous)
```
P(Y_{ij} = k | θ_i) = exp(Σ_{s=0}^{k} (θ_i - δ_{js})) / Σ_{m=0}^{K_j} exp(Σ_{s=0}^{m} (θ_i - δ_{js}))
```
where δ_{js} = step difficulty for category s of item j.

### GPCM (Generalized PCM)
Add discrimination: replace θ_i with a_j·θ_i.

---

## 6. EM Algorithm for IRT (Bock & Aitkin 1981)

### E-step
Compute posterior of θ for each person using current item parameters:
```
p(θ = X_q | Y_i, â, b̂) ∝ w_q · Π_j P(Y_{ij} | X_q, â_j, b̂_j)
```
This gives expected "counts" at each quadrature point.

### M-step
For 2PL: weighted logistic regression for each item, with quadrature-expanded data:
```
For item j: maximize Σ_q Σ_i r_{iq} · [Y_{ij}·log P(Y_{ij}|X_q) + (1-Y_{ij})·log(1-P(Y_{ij}|X_q))]
```
where r_{iq} = posterior weight of person i at quadrature point q.

**This IS IRLS (Iteratively Reweighted Least Squares) — F10 weighted logistic regression.**

### Convergence: typically 20-100 EM iterations. Accelerate with:
- Ramsay's acceleration
- Louis's method for SE (from EM, not requiring full Hessian)
- Supplemented EM (SEM) for standard errors

---

## 7. Person Ability Estimation

### MLE (Maximum Likelihood)
```
θ̂_i = argmax Σ_j [Y_{ij}·log P_j(θ) + (1-Y_{ij})·log(1-P_j(θ))]
```
Solve via Newton-Raphson. SE = 1/√I(θ̂) where I = Fisher information.

### CRITICAL: MLE is undefined for perfect scores (all correct or all incorrect). θ̂ → ±∞.

### EAP (Expected A Posteriori)
```
θ̂_EAP = E[θ|Y_i] = Σ_q X_q · p(θ=X_q|Y_i) / Σ_q p(θ=X_q|Y_i)
```
Always finite (shrunk toward prior mean). Recommended default.

### MAP (Maximum A Posteriori)
```
θ̂_MAP = argmax [log L(Y_i|θ) + log f(θ)]
```
With N(0,1) prior: adds -θ²/2 penalty. Also always finite.

---

## 8. Test Information and Reliability

### Item Information Function
```
I_j(θ) = [P'_j(θ)]² / [P_j(θ)(1-P_j(θ))]
```

For 2PL:
```
I_j(θ) = a²_j · P_j(θ) · (1-P_j(θ))
```
Maximum at θ = b_j: I_max = a²_j/4.

### Test Information = Σ_j I_j(θ)
SE(θ̂) = 1/√(Σ_j I_j(θ))

### Reliability
```
ρ = 1 - 1/I(θ)·Var(θ)    (at a specific θ)
```
IRT reliability is θ-dependent (unlike Cronbach's α which is a single number).

### Cronbach's Alpha
```
α = (J/(J-1)) · (1 - Σ σ²_j / σ²_total)
```

**CRITICAL**: α is a LOWER BOUND on reliability, and only equals reliability under tau-equivalence (equal factor loadings). Often misinterpreted.

---

## 9. Differential Item Functioning (DIF)

### Logistic Regression Method (Swaminathan & Rogers)
```
logit(P(Y=1)) = β₀ + β₁·θ + β₂·Group + β₃·θ·Group
```
- β₂ ≠ 0: uniform DIF (different difficulty)
- β₃ ≠ 0: non-uniform DIF (different discrimination)

### Mantel-Haenszel (MH) Method
Stratify by raw score, compute odds ratio:
```
α_MH = Σ_k (A_k·D_k/T_k) / Σ_k (B_k·C_k/T_k)
```
where A,B,C,D are 2×2 table entries within stratum k.

Convert to delta metric: Δ_MH = -2.35 · ln(α_MH).
|Δ_MH| > 1.0: moderate DIF. |Δ_MH| > 1.5: large DIF.

---

## 10. Computerized Adaptive Testing (CAT)

### Algorithm
1. Start: θ̂₀ = 0 (or prior mean)
2. Select next item: maximize information at current θ̂ (or other criterion)
3. Administer item, update θ̂ (EAP or MLE)
4. Stop when: SE(θ̂) < threshold OR max items reached

### Item Selection Criteria
- **Maximum Information**: select j = argmax I_j(θ̂)
- **KL Information**: accounts for uncertainty in θ̂
- **Content balancing**: ensure items cover required content domains

### Exposure Control
Shadow test: select from a constrained subset to prevent over-exposure of "best" items.

---

## Sharing Surface

### Reuse from Other Families
- **F11 (Mixed Effects)**: IRT = GLMM with crossed random effects
- **F14 (Factor Analysis)**: FA loadings initialize 2PL discriminations
- **F10 (Regression)**: M-step = weighted logistic regression (IRLS)
- **F05 (Optimization)**: Newton-Raphson for person parameter estimation
- **F07 (Hypothesis)**: DIF tests (χ², LRT for nested IRT models)
- **F06 (Descriptive)**: Item statistics (p-values, point-biserial correlations)

### Structural Rhymes
- **IRT = factor analysis on binary data**: same latent structure, different measurement model
- **EM for IRT = EM for mixture models (F16)**: same E-step/M-step structure
- **EAP = Bayesian posterior mean (F34)**: same shrinkage principle
- **Test information = Fisher information (F10)**: same concept at item/test level
- **CAT = sequential optimal design**: same as Bayesian adaptive design

---

## Implementation Priority

**Phase 1** — Rasch + 2PL (~130 lines, as noted in task):
1. Rasch (1PL) via CML + MMLE
2. 2PL via EM (Bock-Aitkin: E-step quadrature, M-step IRLS)
3. Person ability estimation (MLE, EAP, MAP)
4. Item/test information functions
5. Gauss-Hermite quadrature points and weights

**Phase 2** — Extended models (~100 lines):
6. 3PL (with Bayesian prior on guessing)
7. GRM (Graded Response Model)
8. PCM/GPCM (Partial Credit)
9. Cronbach's alpha + other reliability indices

**Phase 3** — DIF + CAT (~80 lines):
10. DIF detection (logistic regression + Mantel-Haenszel)
11. CAT engine (item selection + ability update + stopping rules)
12. Item exposure control

---

## Composability Contract

```toml
[family_15]
name = "IRT & Psychometrics"
kingdom = "C (EM iterations) + A (M-step accumulates)"

[family_15.shared_primitives]
irt_likelihood = "Item response function (1PL/2PL/3PL/GRM/PCM)"
em_irt = "EM algorithm: E-step=quadrature posterior, M-step=weighted IRLS"
ability_estimation = "EAP/MAP/MLE for person parameters"
information = "Item and test information functions"

[family_15.reuses]
f11_glmm = "IRT as logistic GLMM with crossed random effects"
f14_factor = "FA loadings initialize discrimination parameters"
f10_regression = "Weighted logistic regression in M-step"
f05_optimizer = "Newton-Raphson for MLE ability estimation"
f07_hypothesis = "LRT for model comparison, DIF tests"
f06_descriptive = "Item p-values, point-biserial correlations"

[family_15.provides]
item_params = "Calibrated item parameters (a, b, c)"
person_abilities = "θ̂ estimates with SE"
information_functions = "Item and test information"
dif = "DIF detection results"
cat = "Adaptive testing engine"

[family_15.session_intermediates]
item_params = "ItemParams(test_id) — calibrated a, b, c per item"
ability_estimates = "Abilities(test_id) — θ̂, SE per person"
posterior_weights = "PosteriorWeights(test_id) — E-step quadrature weights"
```
