# F16 Mixture Models — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 16 (Mixture Models / GMM / EM).
Primary gold standards: R `mclust` package and Python `sklearn.mixture.GaussianMixture`.

---

## R: mclust Package — The Canonical GMM Reference

`mclust` is the gold standard for GMM in academic statistics. It implements 14 parameterized
covariance models and uses BIC for automatic model selection.

### Model Name Codes (key for matching)

The covariance structure is encoded in 3-letter codes: **Volume · Shape · Orientation**

| Code | Volume | Shape | Orientation | Description |
|------|--------|-------|-------------|-------------|
| `EII` | Equal | Isotropic | Identity | Spherical, same σ² per cluster |
| `VII` | Variable | Isotropic | Identity | Spherical, different σ²_k per cluster |
| `EEI` | Equal | Equal | Axis-aligned | Diagonal, same variances |
| `VEI` | Variable | Equal | Axis-aligned | Diagonal, scale differs |
| `EVI` | Equal | Variable | Axis-aligned | Diagonal, shape differs |
| `VVI` | Variable | Variable | Axis-aligned | Diagonal, fully variable (**Phase 1 target**) |
| `EEE` | Equal | Equal | Ellipsoidal | Full covariance, shared across clusters |
| `VVV` | Variable | Variable | Ellipsoidal | Full covariance, per-cluster (**most general**) |

**Tambear Phase 1 target**: `VVI` (diagonal, fully variable) — matches `scatter_multi_phi_weighted`
per-dimension structure. `EII` and `VII` are special cases (spherical).

### Basic Usage

```r
library(mclust)

# Fixed K, specific model:
fit <- Mclust(X, G=3, modelNames="VVI")   # diagonal, variable
fit$parameters$mean                       # μ_k, shape (d, K)
fit$parameters$variance$sigma             # Σ_k, shape (d, d, K) — diagonal for VVI
fit$parameters$pro                        # π_k, length K (mixing weights)
fit$z                                     # soft assignments r_nk, shape (N, K)
fit$classification                        # hard labels = argmax r_nk

# Auto-select K via BIC:
fit <- Mclust(X)                          # tries G=1:9 and all model codes
summary(fit$BIC)                          # BIC matrix (G × model)
plot(fit$BIC)                             # BIC curve

# Log-likelihood and convergence:
fit$loglik                                # final log-likelihood
fit$df                                    # number of parameters
fit$BIC                                   # BIC = 2*loglik - df*log(N)
```

### Accessing E-step Output (responsibilities)

```r
fit$z          # shape (N, K) — the r_nk matrix. Row sums = 1.
fit$z[1,]      # responsibilities for point 1 across all clusters
```

This is `SoftAssignment` in tambear's intermediate types.

### mclust vs. Naive EM Convergence Criterion

mclust uses: `|log L(t) - log L(t-1)| / (1 + |log L(t)|) < tol` (relative change).
Default tol = 1e-5 in `mclust`. Tambear's navigator uses `Δlog(L) < 1e-6` (absolute change).
For validation, match log-likelihood final value, not iteration count.

---

## Python: sklearn.mixture.GaussianMixture

```python
from sklearn.mixture import GaussianMixture

# Basic usage:
gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',     # 'full', 'tied', 'diag', 'spherical'
    tol=1e-3,                   # convergence threshold on log-likelihood
    max_iter=100,
    n_init=1,                   # number of random restarts
    init_params='kmeans',       # 'kmeans', 'k-means++', 'random', 'random_from_data'
    random_state=42,
)
gmm.fit(X)

# Parameters:
gmm.means_            # μ_k, shape (K, d)
gmm.covariances_      # Σ_k: shape depends on covariance_type
gmm.precisions_       # Σ_k^{-1} (stored for efficiency)
gmm.weights_          # π_k, shape (K,)
gmm.converged_        # bool — did it converge?
gmm.n_iter_           # iterations taken
gmm.lower_bound_      # final log-likelihood lower bound (per sample)

# Inference:
probs = gmm.predict_proba(X)   # soft assignments r_nk, shape (N, K)
labels = gmm.predict(X)        # hard labels = argmax r_nk
log_prob = gmm.score(X)        # mean log-likelihood per sample
```

### covariance_type Mapping to mclust Codes

| sklearn | mclust equivalent | Description |
|---------|------------------|-------------|
| `'spherical'` | `'EII'` or `'VII'` | Single σ² per cluster (VII = variable) |
| `'diag'` | `'VVI'` | Diagonal Σ, variable per cluster — **Phase 1** |
| `'tied'` | `'EEE'` | Full Σ, same across all clusters |
| `'full'` | `'VVV'` | Full Σ, per cluster — most general |

**Covariances_ shape by type:**
- `'spherical'`: shape `(K,)` — one float per cluster
- `'diag'`: shape `(K, d)` — d values per cluster
- `'tied'`: shape `(d, d)` — one matrix
- `'full'`: shape `(K, d, d)` — K matrices

---

## Initialization Traps

### Trap 1: Random restarts are essential for GMM

GMM has local optima. Default `n_init=1` in sklearn can give poor solutions.
For validation tests: use `random_state=42` and `n_init=10` to get reproducible best solution.

```python
gmm = GaussianMixture(n_components=3, n_init=10, random_state=42)
```

### Trap 2: sklearn's `lower_bound_` is per-sample, mclust's `loglik` is total

```python
# sklearn lower_bound_ is mean log-likelihood per sample:
log_lik_total = gmm.lower_bound_ * N  # to compare with mclust$loglik
```

### Trap 3: K-means initialization vs. random (default differs)

sklearn default `init_params='kmeans'` — initialize means via K-means centroids.
mclust default: hierarchical agglomeration for initialization (different algorithm).
These can produce different starting points and converge to different local optima.

For parity tests: use `init_params='random_from_data'` in sklearn and match the seed,
or compare final log-likelihoods rather than parameter values (which may differ for
equivalent solutions with permuted cluster labels).

### Trap 4: Label switching — cluster 0 in mclust ≠ cluster 0 in sklearn

GMM cluster labels are arbitrary permutations. Two correctly fitted models may produce
identical soft assignments with different cluster indices.

For validation: compare the SET of (μ_k, σ_k, π_k) tuples after sorting by μ_k[0],
not the raw arrays.

---

## Covariance Structure Computation (for tambear validation)

### Diagonal case (VVI / sklearn 'diag'):

For tambear Phase 1, the M-step computes:
```
μ_kd = Σ_n r_nk * x_nd / N_k          (weighted mean per dimension)
σ²_kd = Σ_n r_nk * (x_nd - μ_kd)² / N_k  (weighted variance per dimension)
```

To verify against sklearn:
```python
gmm = GaussianMixture(n_components=K, covariance_type='diag', n_init=10, random_state=42)
gmm.fit(X)
# gmm.covariances_[k, d] = σ²_kd  ← this is what tambear's M-step should produce
```

### Spherical case (EII/VII / sklearn 'spherical'):

```python
gmm = GaussianMixture(n_components=K, covariance_type='spherical', n_init=10, random_state=42)
# gmm.covariances_[k] = σ²_k (one value per cluster, same across dimensions)
```

Tambear spherical: compute `σ²_kd` per-dimension, then average across d.

---

## E-step Gaussian Log-Density Computation

The critical numerical piece: evaluating `log N(x_n | μ_k, σ²_k)` stably.

For diagonal Σ:
```
log N(x | μ, σ²) = -d/2 · log(2π) - 1/2 · Σ_d log(σ²_kd) - 1/2 · Σ_d (x_d - μ_kd)² / σ²_kd
```

The last term is a weighted L2 distance: `Σ_d (x_d - μ_kd)² / σ²_kd` = Mahalanobis distance
with diagonal covariance = **elementwise-scaled squared distance**.

For spherical Σ (single σ²_k):
```
log N(x | μ, σ²_k) = -d/2 · log(2π) - d/2 · log(σ²_k) - ‖x - μ_k‖² / (2σ²_k)
```

This is exactly proportional to the RBF kernel: `exp(-γ_k · ‖x - μ_k‖²)`.
E-step for spherical GMM = applying per-cluster RBF kernels to the distance matrix.

**Numerical stability**: always compute in log space. `log_r_nk = log_pi_k + log_N(x_n | params_k)`,
then normalize via log-sum-exp (the `LogSumExpOp` the navigator identified).

---

## Validation Test Data (Simple Separable GMM)

```python
import numpy as np
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(42)
cluster1 = rng.normal([0.0, 0.0], 0.5, size=(100, 2))
cluster2 = rng.normal([5.0, 0.0], 0.5, size=(100, 2))
cluster3 = rng.normal([2.5, 4.0], 0.5, size=(100, 2))
X = np.vstack([cluster1, cluster2, cluster3])

gmm = GaussianMixture(n_components=3, covariance_type='diag',
                      n_init=10, random_state=42)
gmm.fit(X)
print("means:", gmm.means_)
print("variances:", gmm.covariances_)
print("weights:", gmm.weights_)
print("log_lik:", gmm.lower_bound_ * 300)
print("converged:", gmm.converged_)
```

Well-separated clusters (within 0.5 std) should produce near-perfect assignment.
For tambear validation: soft assignment matrix should have max entries > 0.999 for each point.

---

## BIC for Model Selection

```r
# R mclust BIC (higher = better in mclust, opposite of usual convention):
bic_value <- fit$BIC   # mclust uses: 2*loglik - df*log(N) (positive-good)

# Python sklearn BIC (lower = better, standard convention):
bic_value = gmm.bic(X)  # = -2*loglik + df*log(N) (positive-good, lower better)
```

**Sign convention differs!** mclust maximizes BIC; sklearn minimizes BIC. They are negatives of each other.

---

## Termination Criteria

| Gold Standard | Criterion | Default |
|--------------|-----------|---------|
| R mclust | relative change in log-likelihood | 1e-5 |
| sklearn GMM | change in lower bound | 1e-3 |
| tambear (proposed) | absolute change in log-likelihood | 1e-6 |

For maximum tolerance match: use `tol=1e-8` in sklearn for validation, compare converged log-likelihoods.
