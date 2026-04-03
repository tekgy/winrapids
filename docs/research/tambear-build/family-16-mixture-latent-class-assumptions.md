# Family 16: Mixture & Latent Class Models — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: C (EM iterations) wrapping A (E-step posteriors + M-step weighted scatter)

---

## Core Insight: EM Is the IRLS Template — Again

Every model in this family follows the same EM loop:
1. **E-step**: compute posterior responsibilities (soft assignments). Embarrassingly parallel per observation. Kingdom A.
2. **M-step**: update parameters using weighted sufficient statistics. This is `scatter_multi_phi_weighted` — the SAME operation as logistic regression (F10), mixed effects (F11), IRT (F15), and Cox PH (F13).

The IRLS master template covers this entire family. The only difference is the WEIGHT FUNCTION:
- **GMM**: weights = posterior probabilities from Gaussian mixture likelihood
- **LCA**: weights = posterior probabilities from multinomial/Bernoulli likelihood
- **HMM**: weights = forward-backward (α·β) probabilities
- **Factor mixture**: weights = class posteriors within FA model

**Structural rhyme**: GMM M-step = weighted MomentStats per cluster. Same as F06 weighted describe(), same as F20 KMeans centroid update, same as F11 LME variance component estimation.

---

## 1. Gaussian Mixture Models (GMM)

### Model
```
p(x) = Σ_{k=1}^K π_k · N(x | μ_k, Σ_k)
```

where π_k = mixing proportions (Σπ_k = 1), μ_k = means, Σ_k = covariance matrices.

### EM Algorithm

**E-step** (responsibilities):
```
γ_{ik} = π_k · N(x_i | μ_k, Σ_k) / Σ_{j=1}^K π_j · N(x_i | μ_j, Σ_j)
```

Each responsibility is independent → embarrassingly parallel across observations.

**M-step** (parameter updates):
```
N_k = Σ_i γ_{ik}
π_k = N_k / N
μ_k = (1/N_k) · Σ_i γ_{ik} · x_i
Σ_k = (1/N_k) · Σ_i γ_{ik} · (x_i - μ_k)(x_i - μ_k)'
```

This IS `scatter_multi_phi_weighted(ByKey(cluster), [x, x·x'], γ, Add)` — weighted scatter.

### Log-Likelihood
```
ℓ = Σ_i log(Σ_k π_k · N(x_i | μ_k, Σ_k))
```

Monitor for convergence: |ℓ_{t+1} - ℓ_t| < ε (typically 1e-6).

### Covariance Parameterization

| Type | Parameters per cluster | Total parameters |
|------|----------------------|-----------------|
| Full | p(p+1)/2 | K·p(p+1)/2 |
| Diagonal | p | K·p |
| Spherical | 1 | K |
| Tied (shared) | p(p+1)/2 | p(p+1)/2 |

**Rule**: n >> total parameters. For full covariance with p=10, K=3: need n >> 165. For p=100: need n >> 15,150. Use diagonal or tied when n is limited.

### CRITICAL: Singularity in EM
If a component collapses to a single point, Σ_k → 0, likelihood → ∞. This is a degenerate MLE.

**Fixes**:
1. Regularization: Σ_k = Σ̂_k + λI (add small diagonal). sklearn uses reg_covar=1e-6.
2. Restart: if det(Σ_k) < ε, reinitialize that component
3. Bayesian: MAP with Wishart prior on Σ_k (prevents collapse)

### Initialization

| Method | Description | Quality |
|--------|------------|---------|
| KMeans | Run KMeans, use cluster assignments | Good, deterministic |
| KMeans++ | KMeans++ initialization | Better spread |
| Random | Random responsibilities | Poor, many restarts needed |
| Random from data | μ_k = random data points | Acceptable |

**Decision**: KMeans initialization (1 run), then run EM from multiple starts (5-10), keep best log-likelihood.

### GPU decomposition
- E-step: `accumulate(Contiguous, gaussian_pdf(x_i, μ_k, Σ_k), Identity)` per cluster → normalize (parallel per obs)
- M-step: `scatter_multi_phi_weighted(ByKey(k), stats_expr, γ, Add)` — weighted sufficient statistics per cluster
- Log-likelihood: `accumulate(All, log_sum_exp(per_cluster_log_probs), Add)` — reduce

---

## 2. Latent Class Analysis (LCA)

### Model
Categorical (manifest) variables, categorical latent variable:
```
P(X₁=x₁, ..., X_p=x_p) = Σ_{k=1}^K π_k · Π_{j=1}^p P(X_j=x_j | class=k)
```

### EM for LCA

**E-step**:
```
γ_{ik} = π_k · Π_j P(x_{ij} | class=k) / Σ_l π_l · Π_j P(x_{ij} | class=l)
```

**M-step**:
```
π_k = N_k / N
P(X_j=c | class=k) = Σ_{i: x_ij=c} γ_{ik} / N_k
```

M-step is just weighted counts per class per variable per category. `scatter_add_weighted`.

### Latent Profile Analysis (LPA)
Continuous indicators, categorical latent variable. Same as GMM but often with constrained (diagonal) covariance.

### GPU decomposition
- E-step: log-space for products of many probabilities (avoid underflow)
- M-step: weighted histogram per class per variable (scatter)

---

## 3. Hidden Markov Models (HMM)

### Model
- States: S = {1, ..., K} (discrete, hidden)
- Transitions: A[i,j] = P(s_{t+1}=j | s_t=i)
- Emissions: B[k] = P(x_t | s_t=k) (Gaussian, multinomial, etc.)
- Initial: π[k] = P(s_1=k)

### Forward Algorithm (α)
```
α_t(k) = P(x_1, ..., x_t, s_t=k)
α_1(k) = π_k · B_k(x_1)
α_t(k) = [Σ_j α_{t-1}(j) · A[j,k]] · B_k(x_t)
```

**This IS an Affine scan.** State = vector α_t. Transition = matrix multiply by A then pointwise multiply by B_k(x_t).

Kingdom B: `accumulate(Prefix(forward), emission_weighted_transition, MatMul)`.

### Backward Algorithm (β)
```
β_T(k) = 1
β_t(k) = Σ_j A[k,j] · B_j(x_{t+1}) · β_{t+1}(j)
```

Reverse scan: `accumulate(Prefix(reverse), reverse_transition, MatMul)`.

### Baum-Welch (EM for HMM)

**E-step**: forward-backward to compute γ_t(k) = P(s_t=k | X) and ξ_t(j,k) = P(s_t=j, s_{t+1}=k | X).

```
γ_t(k) = α_t(k) · β_t(k) / P(X)
ξ_t(j,k) = α_t(j) · A[j,k] · B_k(x_{t+1}) · β_{t+1}(k) / P(X)
```

**M-step**:
```
π̂_k = γ_1(k)
Â[j,k] = Σ_t ξ_t(j,k) / Σ_t γ_t(j)
B̂_k: update emission parameters using γ_t(k) as weights
```

### Viterbi (MAP state sequence)
```
δ_t(k) = max_{s_1,...,s_{t-1}} P(s_1,...,s_{t-1}, s_t=k, x_1,...,x_t)
ψ_t(k) = argmax_j [δ_{t-1}(j) · A[j,k]]
```

Forward scan with Max instead of Sum. Then backtrack via ψ.

### CRITICAL: Numerical Underflow
α values decay exponentially with T. After ~100 steps, underflow in f64.

**Fix**: scaling. At each t, scale α_t by c_t = 1/Σ_k α_t(k). Log-likelihood = -Σ_t log(c_t). Same for β.

### GPU decomposition
- Forward/backward: sequential per sequence (scan), parallel across SEQUENCES
- Emission probabilities: parallel per observation per state
- M-step: weighted scatter (same as GMM)
- **Main GPU win**: many independent sequences in parallel (batch over sequences)

---

## 4. Growth Mixture Models

### Model
Combines:
- Latent classes (F16 LCA/LPA)
- Growth trajectories within classes (F11 LME)
- Different growth parameters per class

```
y_{it} = Λ_t · η_i + ε_{it}     (measurement model per time)
η_i | class=k ~ N(μ_ηk, Σ_ηk)   (growth factors vary by class)
```

### EM
- E-step: compute class posteriors AND growth factor posteriors within class
- M-step: update class-specific growth parameters using F11 LME machinery

### Implementation: compose F16 EM loop around F11 LME per class. No new primitives.

---

## 5. Model Selection (Number of Classes)

### Information Criteria
```
AIC = -2ℓ + 2k
BIC = -2ℓ + k·log(n)
```

BIC preferred for mixture models (penalizes complexity more, consistent estimator of K).

### Classification Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Entropy | E = 1 - Σ_i Σ_k (-γ_{ik}·log(γ_{ik})) / (n·log(K)) | 1 = perfect separation |
| BLRT | Bootstrap LR test: -2(ℓ_K - ℓ_{K+1}) | p-value for K vs K+1 |
| VLMR-LRT | Vuong-Lo-Mendell-Rubin | Approximate LR test |
| ICL | BIC - 2·Σ_i Σ_k γ_{ik}·log(γ_{ik}) | BIC + entropy correction |

### CRITICAL: Always run K=1,2,...,K_max and compare. No method reliably selects K in one shot.

### Practical Guide
1. BIC minimum → candidate K
2. Entropy > 0.8 → classes are separable
3. All classes have reasonable size (>5% of n)
4. Classes are substantively interpretable
5. BLRT significant for K vs K-1

---

## 6. Numerical Stability

### Log-Space Computation
All mixture model computations must be in log-space:
```
log(Σ_k exp(log_π_k + log_f_k)) = log_sum_exp(log_π + log_f)
```

The log-sum-exp trick: `log_sum_exp(a) = max(a) + log(Σ exp(a_i - max(a)))`.

### Degeneracy Prevention
- Regularize covariance: Σ_k + λI
- Minimum component weight: if π_k < 1/(10n), merge or restart
- Maximum iterations + convergence check

### EM Convergence
- EM guarantees ℓ_{t+1} ≥ ℓ_t (monotonic increase). If violated: numerical error.
- EM converges to local maximum (not global). Multiple restarts essential.
- Convergence can be slow near saddle points. Consider EM acceleration (Aitken, SQUAREM).

---

## 7. Edge Cases

| Algorithm | Edge Case | Expected |
|-----------|----------|----------|
| GMM | K=1 | Just compute mean + covariance. No EM needed. |
| GMM | Component collapses | Regularize. Restart if det(Σ) < ε. |
| GMM | n < K·p | Too few observations per class. Reduce K or use diagonal. |
| LCA | All indicators have 1 category | Degenerate. Error. |
| LCA | P(x|class) = 0 | Add Laplace smoothing (add 0.5 to counts). |
| HMM | Transition to absorbing state | Row of A has one 1.0. Valid but check intent. |
| HMM | T < K states | Can't estimate. Too few observations. |
| HMM | Long sequence (T > 1000) | Scaling essential. Monitor for numerical drift. |
| All EM | Non-convergence at max iterations | Return best iterate + warning. |
| All | K_max too large | Empty classes. Automatically prune. |

---

## Sharing Surface

### Reuses from Other Families
- **F06 (Descriptive)**: weighted MomentStats for GMM M-step (scatter_multi_phi_weighted)
- **F01 (Distance)**: Mahalanobis distance for GMM E-step
- **F02 (Linear Algebra)**: Cholesky for multivariate Gaussian, determinant for log-likelihood
- **F05 (Optimization)**: direct optimization alternative to EM
- **F11 (Mixed Effects)**: LME within growth mixture models
- **F20 (Clustering)**: KMeans for GMM initialization, cluster validation metrics
- **F25 (Information Theory)**: entropy for classification quality

### Provides to Other Families
- **F14 (Factor Analysis)**: factor mixture models (FA within latent classes)
- **F20 (Clustering)**: GMM as soft clustering, model-based clustering
- **F34 (Bayesian)**: Dirichlet process mixture (infinite mixture)
- **F15 (IRT)**: mixture IRT (class-specific item parameters)
- **F17 (Time Series)**: HMM for regime-switching time series

### Structural Rhymes
- **GMM E-step = softmax over cluster log-likelihoods**: same as F23 attention softmax
- **GMM M-step = weighted scatter = IRLS M-step**: same as F10/F11/F13/F15
- **HMM forward = matrix-valued Affine scan**: F17 Kalman filter generalized to discrete states
- **EM monotonicity = MM (majorization-minimization)**: same convergence guarantee

---

## Implementation Priority

**Phase 1** — GMM core (~120 lines):
1. Gaussian mixture EM (full, diagonal, spherical, tied covariance)
2. KMeans initialization
3. Multiple restarts with best log-likelihood selection
4. BIC/AIC for model selection

**Phase 2** — Latent class (~100 lines):
5. LCA (categorical indicators)
6. LPA (continuous indicators, diagonal covariance)
7. Entropy, ICL for classification quality
8. BLRT (bootstrap likelihood ratio test)

**Phase 3** — HMM (~150 lines):
9. Forward-backward (scaled)
10. Baum-Welch EM
11. Viterbi decoding
12. HMM with Gaussian/multinomial emissions

**Phase 4** — Extensions (~100 lines):
13. Growth mixture models (F16 × F11 composition)
14. Factor mixture models (F16 × F14 composition)
15. EM acceleration (SQUAREM)
16. Bayesian mixture (Dirichlet process, wraps F34)

---

## Composability Contract

```toml
[family_16]
name = "Mixture & Latent Class Models"
kingdom = "C (EM iterations) wrapping A (E-step posteriors + M-step weighted scatter)"

[family_16.shared_primitives]
em_loop = "Iterate E-step → M-step until convergence"
e_step = "Posterior responsibilities (parallel per observation)"
m_step = "Weighted sufficient statistics (scatter_multi_phi_weighted)"
forward_backward = "HMM α/β via matrix-valued Affine scan"
viterbi = "MAP state sequence via matrix-valued Max scan"

[family_16.reuses]
f06_descriptive = "Weighted MomentStats for GMM M-step"
f01_distance = "Mahalanobis distance for E-step"
f02_linear_algebra = "Cholesky, determinant for Gaussian log-likelihood"
f05_optimization = "Direct optimization, EM acceleration"
f11_mixed_effects = "LME within growth mixture"
f20_clustering = "KMeans initialization, validation metrics"

[family_16.provides]
class_posteriors = "Soft assignment probabilities (γ_ik)"
class_parameters = "Per-class distribution parameters"
model_selection = "BIC, AIC, entropy, BLRT for K selection"
hmm_decoded = "Viterbi state sequences, filtered state probabilities"

[family_16.consumers]
f14_factor_analysis = "Factor mixture models"
f20_clustering = "Model-based clustering"
f34_bayesian = "Dirichlet process mixtures"
f15_irt = "Mixture IRT"
f17_time_series = "Regime-switching models"
```
