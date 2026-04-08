# F30 Sharing Surface: Spatial Statistics as Weighted GramMatrix

Created: 2026-04-01T06:30:36-05:00
By: navigator

Prerequisites: F10 complete (GLS + Cholesky), F01 complete (DistancePairs).

---

## Core Insight: Spatial Statistics = Regression with Structured Covariance

Spatial statistics applies when data has geographic coordinates and nearby observations
are correlated. The canonical issue: standard OLS assumes independence, but spatial
data violates this. The fix: GLS (F10 Phase 2) with a spatially structured covariance matrix.

**Spatial statistics = F10 GLS where Σ is determined by geographic distance.**

---

## Spatial Weights Matrix (W)

The spatial weights matrix W encodes which locations are "neighbors":

**Contiguity-based** (rook/queen for areal data):
- Rook: W[i,j] = 1 if areas i and j share an edge
- Queen: W[i,j] = 1 if areas i and j share an edge OR corner

**Distance-based** (for point data):
- k-nearest neighbors: W[i,j] = 1 if j ∈ KNN(i) → F01 KNN
- Distance threshold: W[i,j] = 1 if d(i,j) < h → F01 DistancePairs threshold
- Inverse distance: W[i,j] = 1/d(i,j)^α → F01 DistancePairs + element-wise power

**W is standardized** row-wise: `W_std[i,j] = W[i,j] / Σ_j W[i,j]`.
W_std is the row-normalized adjacency matrix — same structure as PageRank's matrix (F29).

---

## Moran's I (Spatial Autocorrelation)

```
I = (N / Σ_{i,j} w_ij) · (Σ_{i,j} w_ij (x_i - x̄)(x_j - x̄)) / (Σ_i (x_i - x̄)²)
  = (N / W_sum) · (z' W z) / (z' z)
```

where z = x - x̄ = centered observations (RefCenteredStats).

**Tambear decomposition**:
1. `z = x - mean(x)` — scatter_phi("v - r") = RefCenteredStats (F06)
2. `z' W z = Σ_{i,j} w_ij z_i z_j` = weighted sum of spatial cross-products = GramMatrix with W as weight matrix
   = `scatter_phi("w_ij * z_i * z_j", ByAll)` — weighted accumulate
3. `z' z` = F06 sum of squares
4. I = scalar ratio

**Key insight**: `z' W z` is a weighted bilinear form — a GramMatrix where weights come from
the spatial weights matrix W. Same structure as the kernel GramMatrix from F22/F21.

**New infrastructure**: the spatial weights W is a sparse N×N matrix. For small N (< 10K),
dense is fine. For areal data (states, counties), typically N < 5000 — dense is always fine.

**Moran's I distribution under H₀**: I ~ N(E[I], Var[I]) approximately.
E[I] = -1/(N-1), Var[I] = function of W's trace and eigenvalues.
This is F07 z-test territory.

### Geary's C (related)

```
C = ((N-1) Σ_{i,j} w_ij (x_i - x_j)²) / (2 W_sum Σ_i (x_i - x̄)²)
```

`(x_i - x_j)²` = L2 distance squared = DistancePairs extraction.
Numerator = `scatter_phi("w_ij * (x_i - x_j)^2", ByAll)` — weighted L2² sum.

**This IS DistancePairs weighted by W.** Free from F01.

---

## Spatial Autoregressive Models

### SAR (Simultaneous Autoregressive)

```
y = ρ W y + X β + ε,    ε ~ N(0, σ² I)
```

Solution: `(I - ρW) y = X β + ε` → `y = (I - ρW)^{-1} X β + (I - ρW)^{-1} ε`

This is GLS where `Σ = σ² [(I-ρW)'(I-ρW)]^{-1}`.

**Estimation**: profile likelihood over ρ (one-parameter optimization) + GLS for β.
At each ρ: `Σ^{-1} = [(I-ρW)'(I-ρW)]/σ²` — Cholesky of this structure.

**Tambear path**:
1. Outer loop over ρ (F05 optimization or grid search)
2. At each ρ: form `(I - ρW)` — sparse matrix
3. `Σ^{-1/2} y` via forward substitution = F10 triangular solve
4. GLS = F10 GLS on transformed data

### SLM (Spatial Lag Model) = SAR for Econometrics

Same model, different name in spatial econometrics. IV/2SLS estimation common:
instrument for Wy with W²X, W³X. Reduces to F35 2SLS.

### SEM (Spatial Error Model)

```
y = X β + u,    u = λ W u + ε
```

Error structure: `Σ = σ² [(I-λW)'(I-λW)]^{-1}`.
Same estimation pattern as SAR but on residuals.

### Geographically Weighted Regression (GWR)

```
y_i = β₀(u_i, v_i) + Σ_k β_k(u_i, v_i) x_ik + ε_i
```

Local coefficient estimates vary continuously with geographic location.

At each location i: WLS with weights w_j = K(d(i,j)/h) (kernel-smoothed distance weights).
N separate WLS regressions (one per location) = N F10 WLS calls.
The weights come from DistancePairs + kernel smoothing.

**Tambear path**: N × F10 WLS on local observations. For N = 50K locations: 50K Cholesky solves
on small matrices (local observations per location). GPU: parallelize across locations.

---

## Kriging (Geostatistical Interpolation)

**What it does**: predict value at new location x₀ given observed values at {x₁,...,xN}.

```
Ŷ(x₀) = Σ_i w_i Y(x_i)    with optimal weights w_i
```

Optimal weights = from GLS: `w = Σ(x₀)' Σ^{-1}` where:
- `Σ[i,j] = C(d(x_i, x_j))` = covariance function of distance
- `Σ(x₀)[i] = C(d(x₀, x_i))` = covariance vector of prediction point to observations

**Tambear path**:
1. Fit variogram: `C(d) = σ²(1 - γ(d))` from DistancePairs + binned variance estimates
2. Build covariance matrix: `Σ[i,j] = C(d(i,j))` — function of DistancePairs
3. Cholesky(Σ) — F10 infrastructure
4. Predict: `Ŷ(x₀) = c₀' Σ^{-1} Y` — triangular solve with one new right-hand side

**Ordinary kriging = GLS with structured covariance = F10 Phase 2.**
The variogram fitting (step 1) is the only new code: binned average of DistancePairs-indexed variances.

---

## MSR Types F30 Produces

```rust
pub struct SpatialModel {
    pub n_obs: usize,
    pub model_type: SpatialModelType,
    pub coordinates: Arc<Vec<f64>>,   // (N, 2) or (N, 3) geographic coords

    /// Spatial dependence parameter:
    pub rho: Option<f64>,    // SAR: autoregressive parameter
    pub lambda: Option<f64>, // SEM: spatial error parameter

    /// Coefficients (from GLS):
    pub beta: Vec<f64>,
    pub beta_se: Vec<f64>,

    /// Spatial diagnostics:
    pub moran_i_residuals: f64,   // should be near 0 if model captures spatial structure
    pub lr_test_spatial: Option<f64>,  // LR test for ρ≠0 or λ≠0
}

pub struct SpatialAutocorrelation {
    pub moran_i: f64,
    pub moran_z: f64,
    pub moran_p: f64,
    pub geary_c: f64,
    pub local_moran: Vec<f64>,  // LISA — local indicator, shape (N,)
}
```

---

## Build Order

**Phase 1 (Moran's I + Geary's C)**:
1. Spatial weights W from DistancePairs: threshold or KNN-based (~20 lines)
2. Row-normalize W (~5 lines)
3. Moran's I: RefCenteredStats + weighted bilinear form (~30 lines)
4. Geary's C: DistancePairs L2² + weighted sum (~20 lines)
5. Local Moran (LISA): per-observation z_i · (W z)_i — point-level Moran
6. Tests: `spdep::moran.test()` in R, `libpysal.weights + esda.Moran` in Python

**Phase 2 (SAR + SEM)**:
1. Profile likelihood over ρ: F05 optimization + F10 GLS at each ρ (~60 lines)
2. Sparse (I - ρW) and Cholesky (~30 lines)
3. Tests: `spdep::lagsarlm()`, `spdep::errorsarlm()` in R

**Phase 3 (Kriging, GWR)**:
- Variogram fitting: binned DistancePairs average (~40 lines)
- Kriging prediction: covariance matrix + F10 Cholesky solve
- GWR: N × F10 WLS calls (GPU-parallelizable)

**Gold standards**:
- R `spdep` package: `moran.test()`, `lagsarlm()`, `errorsarlm()`
- R `gstat` package: variogram fitting, kriging
- Python `libpysal` + `esda`: spatial diagnostics
- Python `geopandas` + `pointpats`: GWR

---

## The Lab Notebook Claim

> Spatial statistics is F10 GLS where the covariance matrix Σ is determined by geographic distance. Moran's I = weighted bilinear form on RefCenteredStats (GramMatrix with spatial weights W). Kriging = GLS with variogram-derived covariance (F10 Cholesky). SAR = GLS on spatially filtered data (outer loop over ρ via F05). GWR = N × F10 WLS calls parallelized by location. F30 adds ~75 lines for Phase 1 (spatial autocorrelation diagnostics). The spatial domain's novelty is the spatial weights construction (W from DistancePairs) — everything downstream is existing regression infrastructure.
