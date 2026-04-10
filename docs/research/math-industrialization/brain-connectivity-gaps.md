# Brain Connectivity Primitives — Gap Analysis

Generated: 2026-04-10  
Scope: EEG, MEG, fMRI, ECoG, animal electrophysiology — but the math is universal  
(oscillator coupling, spectral analysis, directed information, graph topology)

---

## What We Have (Confirmed in `crates/tambear/src/`)

Before gap analysis, the confirmed tambear inventory relevant to this domain:

| Primitive | File | Notes |
|-----------|------|-------|
| `fft`, `rfft`, `ifft`, `irfft` | `signal_processing.rs` | Full complex FFT from first principles |
| `hilbert` | `signal_processing.rs` | Analytic signal via FFT + half-spectrum zero |
| `envelope` | `signal_processing.rs` | |`hilbert(x)` |
| `instantaneous_frequency` | `signal_processing.rs` | Phase derivative |
| `periodogram`, `welch` | `signal_processing.rs` | PSD estimation |
| `cross_spectral` | `spectral.rs` | Returns `CrossSpectralResult { freqs, magnitude, phase, coherence }` — MSC included |
| `multitaper_psd` | `spectral.rs` | Multi-taper PSD |
| `band_power`, `relative_band_power` | `spectral.rs` | Frequency-band integration |
| `transfer_entropy` | `information_theory.rs` | Binned histogram estimator |
| `shannon_entropy`, `kl_divergence` | `information_theory.rs` | Full information-theory catalog |
| `ar_fit`, `ar_burg_fit` | `time_series.rs` | Univariate AR (Yule-Walker + Burg) |
| `levinson_durbin` | `time_series.rs` | Core Levinson recursion |
| `arma_fit`, `arima_fit` | `time_series.rs` | Univariate ARMA/ARIMA |
| `modularity`, `label_propagation` | `graph.rs` | Community detection |
| `clustering_coefficient` | `graph.rs` | Mean local clustering |
| `pagerank` | `graph.rs` | Fixed-point centrality |
| `degree_centrality`, `closeness_centrality` | `graph.rs` | Graph centrality |
| `kruskal`, `prim` (MST) | `graph.rs` | Minimum spanning tree |
| `floyd_warshall`, `dijkstra` | `graph.rs` | All-pairs + single-source shortest paths |
| `mutual_information`, `normalized_mutual_information` | `information_theory.rs` | Full MI catalog |
| `morlet_cwt` | `signal_processing.rs` | Continuous wavelet transform |
| `fast_ica` | `signal_processing.rs` | Independent component analysis |
| `covariance_matrix`, `cca` | `multivariate.rs` | Multivariate statistics |
| `delay_embed` | `time_series.rs` | Phase-space reconstruction |

---

## Gaps — Organized by Category

### 1. Phase Connectivity (Functional, Undirected) — MISSING

All of these compose from `hilbert` + existing arithmetic. They're missing as named primitives.

#### 1a. Phase-Locking Value (PLV)
```
PLV = |mean_t( exp(i * (phi_x(t) - phi_y(t))) )|
```
- **Tambear primitives it uses**: `hilbert` (→ instantaneous phase via `atan2(imag, real)`), complex mean, complex modulus
- **Gap**: `phase_locking_value(x, y)` — the named primitive doesn't exist. The sub-operations exist, but the composition is not surfaced.
- **Domain universality**: used in seismology (fault coupling), mechanics (rotor synchrony), econometrics (business cycle phase synchrony), climate science (monsoon teleconnections). Completely general oscillator math.

#### 1b. Phase-Lag Index (PLI)
```
PLI = |mean_t( sign(Im(S_xy(f))) )|
```
- **Tambear primitives it uses**: `cross_spectral` (already returns phase per bin), `sign`, scalar mean
- **Gap**: `phase_lag_index(x, y, fs, seg_len, overlap)` — one-liner on top of `cross_spectral`, not yet named.
- **Domain universality**: the asymmetric imaginary part trick eliminates zero-lag correlations from shared reference — useful in any sensor network where common-mode artifacts inflate connectivity estimates.

#### 1c. Weighted PLI (wPLI)
```
wPLI = |mean(Im(S_xy))| / mean(|Im(S_xy)|)
```
- **Tambear primitives it uses**: `cross_spectral` imaginary component, weighted mean
- **Gap**: `weighted_pli(x, y, fs, seg_len, overlap)` — adds weighting to PLI for bias reduction.

#### 1d. Imaginary Coherence
```
ImCoh(f) = Im(S_xy(f)) / sqrt(S_xx(f) * S_yy(f))
```
- **Tambear primitives it uses**: `cross_spectral` (we return magnitude and phase but not the signed imaginary component directly)
- **Gap**: `imaginary_coherence(x, y, fs, seg_len, overlap)` — `cross_spectral` computes MSC (magnitude-squared) but discards imaginary sign. Need to expose `Im(S_xy) / sqrt(Sxx * Syy)`.

#### 1e. Envelope Correlation
```
r_env = pearson_r( |hilbert(x)|, |hilbert(y)| )
```
- **Tambear primitives it uses**: `hilbert` → `envelope`, `pearson_r` (in `nonparametric.rs`)
- **Gap**: trivial composition — `envelope_correlation(x, y)` is a 3-liner. Missing as a named primitive.
- **Domain universality**: amplitude-amplitude coupling across any band-limited oscillators. Used in climate teleconnection, acoustics, RF.

#### 1f. Amplitude-Amplitude Coupling (AAC) / Band-Limited Power Envelope Correlation
```
AAC = pearson_r( bandpower_x(t), bandpower_y(t) )
```
where `bandpower_x(t)` is sliding-window power in a frequency band.
- **Tambear primitives it uses**: `fir_bandpass` + `envelope` + `rolling_variance_prefix` (as sliding power) + `pearson_r`
- **Gap**: `amplitude_amplitude_coupling(x, y, f_low, f_high, fs, window)` — the pieces exist, the composition doesn't.

---

### 2. Effective Connectivity (Directed) — PARTIALLY MISSING

#### 2a. Multivariate AR (MVAR) Model — MISSING
```
X(t) = sum_{k=1}^{p} A_k * X(t-k) + noise
```
where `X(t)` is a vector, `A_k` are `d x d` coefficient matrices.
- **We have**: univariate `ar_fit` (Yule-Walker + Burg), `levinson_durbin`
- **Gap**: `mvar_fit(data: &Mat, p: usize) -> MvarResult` — multivariate extension. Uses multivariate Yule-Walker (Whittle's algorithm or OLS stacking).
- **Why this is the keystone**: PDC and DTF (below) both require the MVAR `A(f)` matrix. No MVAR = no PDC, no DTF.
- **Domain universality**: MVAR is just multivariate linear prediction. VAR models in econometrics, system identification in control engineering, geophysics signal separation — all the same math.

#### 2b. Partial Directed Coherence (PDC) — MISSING
```
PDC_ij(f) = A_ij(f) / sqrt( sum_k |A_kj(f)|^2 )
```
where `A(f) = I - sum_k A_k * exp(-i*2*pi*f*k)` is the spectral representation of the MVAR model.
- **Tambear primitives it uses**: `mvar_fit` (missing), `fft` of coefficient matrices, matrix column normalization
- **Gap**: `partial_directed_coherence(mvar: &MvarResult, n_freqs: usize) -> Vec<Mat>`
- **Domain universality**: frequency-domain causal influence. Used in neuroscience, econometrics (Geweke decomposition), control systems.

#### 2c. Directed Transfer Function (DTF) — MISSING
```
DTF_ij(f) = H_ij(f) / sqrt( sum_k |H_ik(f)|^2 )
```
where `H(f) = A(f)^{-1}` is the transfer function matrix.
- **Tambear primitives it uses**: `mvar_fit` (missing), matrix inversion at each frequency bin (we have `lu_solve` in `linear_algebra.rs`)
- **Gap**: `directed_transfer_function(mvar: &MvarResult, n_freqs: usize) -> Vec<Mat>`

#### 2d. Granger Causality (Bivariate) — MISSING (despite data_quality.rs mentioning it)
```
GC(X->Y) = ln( var(AR_Y_restricted) / var(AR_Y_full) )
```
where restricted model uses only Y's past, full model uses X and Y's past.
- **Tambear primitives it uses**: `ar_fit` (univariate for restricted), a bivariate version for the full model, `arma_css_residuals`, `variance_ratio_test`
- **Gap**: `granger_causality(x: &[f64], y: &[f64], p: usize) -> GrangerResult` returning GC statistic + F-test p-value
- **Domain universality**: identical math to Geweke's measure of linear feedback. Used in econometrics (Granger 1969 = economics Nobel prize), ecology, climate, EEG. Transfer entropy is the nonlinear generalization — we have that. We're missing the linear case.

#### 2e. Spectral Granger Causality (Geweke decomposition) — MISSING
```
f_Y|X(lambda) = ln( Syy(lambda) / (Syy(lambda) - |H_yx(lambda)|^2 * Sigma_xx) )
```
- **Tambear primitives it uses**: `mvar_fit` (missing), `cross_spectral`
- **Gap**: depends on MVAR

#### 2f. Dynamic Causal Modeling (DCM) — PARTIALLY ADDRESSED
- DCM is a bilinear state-space model `dx/dt = (A + u*B)*x + C*u` with Bayesian model inversion.
- We have: `kalman.rs` (state-space), `bayesian.rs` (variational inference, MCMC)
- **Gap**: The specific DCM generative model + variational Laplace inversion is not assembled. This is the most domain-specific item on this list — DCM's bilinear structure is primarily a neuroscience/physiology tool, not a general physics primitive. The components (state-space + variational Bayes) are general; the wiring is domain-specific.
- **Assessment**: lower priority; components exist, assembly is domain-specific.

---

### 3. Phase-Amplitude Coupling (PAC) — MISSING

PAC measures whether the phase of a low-frequency oscillation modulates the amplitude of a high-frequency oscillation. Used universally in any coupled nonlinear oscillator.

#### 3a. Modulation Index (MI / Tort et al.)
```
MI = (log(N) - H(A|phi)) / log(N)
```
where amplitude `A` is distributed into N phase bins, H is Shannon entropy of that distribution.
- **Tambear primitives it uses**: `hilbert` → phase + amplitude, `histogram` (we have), `shannon_entropy` (we have), `kl_divergence` (uniform reference)
- **Gap**: `modulation_index(phase_signal: &[f64], amp_signal: &[f64], n_bins: usize) -> f64` — all sub-primitives exist, composition missing.
- **Domain universality**: tremor analysis (motor cortex theta-gamma coupling is same math as motor tremor beta-gamma), seismology, RF signal analysis.

#### 3b. Mean Vector Length (MVL / Canolty et al.)
```
MVL = |mean_t( A(t) * exp(i * phi(t)) )|
```
- **Tambear primitives it uses**: `hilbert` → phase + amplitude, complex weighted mean, `c_abs`
- **Gap**: `mean_vector_length(phase_signal: &[f64], amp_signal: &[f64]) -> f64` — trivial composition.

#### 3c. GLM-PAC
```
A(t) ~ beta_0 + beta_1 * cos(phi(t)) + beta_2 * sin(phi(t))
```
- **Tambear primitives it uses**: `hilbert`, `ols` (we have in linear_algebra), F-test (in hypothesis.rs)
- **Gap**: `glm_pac(phase_signal: &[f64], amp_signal: &[f64]) -> PacGlmResult` — OLS with circular basis functions.

#### 3d. Phase-Phase Coupling (n:m Locking)
```
PLV_{n:m} = |mean( exp(i * (n*phi_x - m*phi_y)) )|
```
- **Tambear primitives it uses**: `hilbert` → phase, complex exponential, mean
- **Gap**: `phase_phase_coupling(x: &[f64], y: &[f64], n: i32, m: i32) -> f64` — generalized PLV at harmonic ratios.

---

### 4. Graph-Theoretic Connectivity — MOSTLY PRESENT, GAPS AT EDGES

#### What we have:
- `clustering_coefficient` ✓
- `modularity` + `label_propagation` ✓
- `pagerank` ✓
- `degree_centrality`, `closeness_centrality` ✓
- `kruskal`, `prim` MST ✓
- `floyd_warshall` all-pairs shortest paths ✓

#### Gaps:

**Betweenness Centrality** — MISSING
```
BC(v) = sum_{s!=v!=t} sigma_st(v) / sigma_st
```
- **Tambear primitives it uses**: `floyd_warshall` or repeated `dijkstra`, path counting
- **Gap**: `betweenness_centrality(g: &Graph) -> Vec<f64>` — O(V^3) with Floyd-Warshall, O(VE) with Brandes algorithm.
- **Domain universality**: universal graph primitive. Used in social networks, logistics, infrastructure vulnerability.

**Eigenvector Centrality** — MISSING
```
x_i = (1/lambda) * sum_j A_ij * x_j
```
- **Tambear primitives it uses**: power iteration (we have in `linear_algebra.rs` as `power_iteration` or similar)
- **Gap**: `eigenvector_centrality(adj: &[f64], n: usize) -> Vec<f64>` — power iteration on adjacency matrix.

**Small-World Coefficient (sigma / omega)** — MISSING
```
sigma = (C/C_rand) / (L/L_rand)
```
- **Tambear primitives it uses**: `clustering_coefficient`, `diameter`, random graph generators (missing)
- **Gap**: `small_world_sigma(g: &Graph, n_rand: usize) -> f64` — requires random Erdos-Renyi graph generation for null model.

**Rich Club Coefficient** — MISSING
```
phi(k) = E_{>k} / (N_{>k} * (N_{>k} - 1))
```
- **Tambear primitives it uses**: degree sequence, subgraph edge count
- **Gap**: `rich_club_coefficient(g: &Graph, k: usize) -> f64`

**Louvain Community Detection** — MISSING (we have label propagation but not Louvain)
- **Gap**: `louvain(g: &Graph) -> Vec<usize>` — greedy modularity optimization. Label propagation is not equivalent.

**Network-Based Statistic (NBS)** — MISSING
```
permutation testing on connected subnetworks of connectivity matrices
```
- **Tambear primitives it uses**: permutation tests (we have in `hypothesis.rs`), connected components (we have)
- **Gap**: `network_based_statistic(group1_mats: &[Mat], group2_mats: &[Mat], threshold: f64, n_perm: usize) -> NbsResult`
- **Domain universality**: any problem requiring permutation inference on a graph metric. Used in structural engineering, protein interaction networks, epidemiological contact networks.

---

### 5. Bicoherence / Bispectrum — MISSING

The bispectrum detects quadratic phase coupling: whether two frequencies `f1`, `f2` phase-lock to produce `f1+f2`.

```
B(f1, f2) = E[ X(f1) * X(f2) * X*(f1+f2) ]
bicoherence = |B(f1,f2)|^2 / (E[|X(f1)*X(f2)|^2] * E[|X(f1+f2)|^2])
```

- **Tambear primitives it uses**: `rfft`, complex multiplication, ensemble averaging over segments
- **Gap**: `bispectrum(data: &[f64], fs: f64, seg_len: usize) -> BispectrumResult` and `bicoherence(data: &[f64], ...)` 
- **Domain universality**: wave-wave interaction in plasma physics, nonlinear acoustics, ocean waves, RF harmonic distortion. Completely general second-order nonlinearity detector.

---

### 6. Source Localization Math — DOMAIN-SPECIFIC COMPONENTS

These are the most neuroscience-specific items. The math is general linear inverse problems but the physical setup (electromagnetic forward model) is domain-specific.

#### Lead Field / Gain Matrix (Forward Model)
- Requires boundary element method (BEM) or finite element for electromagnetic head model
- **We have**: BEM/FEM is listed in CLAUDE.md scope under civil/electrical engineering — but not yet implemented
- **Assessment**: the forward model construction is physics-domain (Maxwell's equations on geometry). The linear algebra once the lead field matrix is computed is fully general.

#### Minimum Norm Estimate (MNE)
```
J = L^T (L*L^T + lambda*C_noise)^{-1} * M
```
where L is the lead field matrix, M is the sensor measurement.
- **Tambear primitives it uses**: matrix multiply, `cholesky_solve` or regularized least squares
- **Gap**: `minimum_norm_estimate(L: &Mat, M: &Mat, lambda: f64) -> Mat` — generic regularized left-inverse. Not brain-specific.
- **Domain universality**: any linear inverse problem with underdetermined system + regularization. Geophysical tomography, medical CT reconstruction, astronomical deconvolution.

#### Beamformer (LCMV)
```
W_j = (C^{-1} * L_j) / (L_j^T * C^{-1} * L_j)
```
where C is the sensor covariance matrix and L_j is the lead field for source j.
- **Tambear primitives it uses**: `covariance_matrix`, `cholesky_solve`, matrix-vector multiply
- **Gap**: `lcmv_beamformer(L: &Mat, C: &Mat) -> Mat` — weighted spatial filter. Used universally in radar/sonar beamforming, radio astronomy, seismic array processing.

#### sLORETA
```
J_sLORETA = (L^T * C^{-1} * L)^{-1/2} * J_MNE_standardized
```
- **Tambear primitives it uses**: MNE + matrix square root (we have in `linear_algebra.rs`?)
- **Gap**: depends on MNE being implemented first.

---

## Universality Classification

| Measure | Neuroscience label | General math domain |
|---------|-------------------|---------------------|
| PLV | Phase-locking value | Coupled oscillator phase synchrony |
| PLI, wPLI | Phase-lag index | Imaginary cross-spectrum asymmetry |
| Imaginary coherence | ImCoh | Signed spectral correlation |
| Envelope correlation | AAC | Amplitude-amplitude coupling |
| Modulation index | PAC | Phase-amplitude coupling (entropy variant) |
| MVL | PAC | Phase-amplitude coupling (vector strength) |
| Phase-phase coupling | n:m locking | Generalized phase synchrony at harmonics |
| MVAR fit | MVAR | Multivariate autoregression = VAR (econometrics) |
| PDC | Partial directed coherence | Frequency-domain Granger causality |
| DTF | Directed transfer function | Transfer matrix spectral decomposition |
| Granger causality | GC | Linear predictive causality (Granger 1969) |
| Bispectrum / bicoherence | — | Second-order spectral nonlinearity |
| Betweenness centrality | — | Universal graph centrality |
| Eigenvector centrality | — | Power-iteration centrality |
| Louvain | — | Greedy modularity community detection |
| NBS | Network-based statistic | Permutation testing on subgraphs |
| Rich club | — | High-degree node density |
| LCMV beamformer | — | Spatial filter / constrained optimization |
| MNE | Minimum norm estimate | Regularized linear inverse problem |

**Everything in this list is general mathematics.** The labels are neuroscience conventions. The math is domain-free.

---

## Priority Ordering for Implementation

### Tier 1 — High impact, simple compositions (all sub-primitives exist)

1. `phase_locking_value(x, y, fs)` — `hilbert` + complex mean + modulus
2. `phase_lag_index(x, y, fs, seg_len, overlap)` — `cross_spectral` + sign + mean
3. `weighted_pli(x, y, fs, seg_len, overlap)` — weighted PLI variant
4. `imaginary_coherence(x, y, fs, seg_len, overlap)` — modify `cross_spectral` to expose Im(S_xy)/sqrt(Sxx*Syy)
5. `envelope_correlation(x, y)` — `envelope` + `pearson_r`
6. `mean_vector_length(phase, amp)` — `hilbert` + complex weighted mean
7. `modulation_index(phase, amp, n_bins)` — `histogram` + `shannon_entropy` + `kl_divergence`
8. `phase_phase_coupling(x, y, n, m)` — generalized PLV
9. `glm_pac(phase, amp)` — OLS with sin/cos phase regressors
10. `betweenness_centrality(g)` — Brandes algorithm on existing graph
11. `eigenvector_centrality(adj, n)` — power iteration
12. `rich_club_coefficient(g, k)` — simple degree-subgraph computation

### Tier 2 — Require new infrastructure (MVAR)

13. `mvar_fit(data: &Mat, p: usize) -> MvarResult` — KEYSTONE; unlocks PDC + DTF + spectral Granger
14. `granger_causality(x, y, p)` — bivariate AR model comparison
15. `partial_directed_coherence(mvar, n_freqs)` — depends on `mvar_fit`
16. `directed_transfer_function(mvar, n_freqs)` — depends on `mvar_fit`
17. `spectral_granger(mvar, n_freqs)` — Geweke decomposition

### Tier 3 — Require additional infrastructure

18. `bispectrum(data, fs, seg_len)` — new 3-frequency accumulation
19. `bicoherence(data, fs, seg_len)` — normalized bispectrum
20. `louvain(g)` — greedy modularity optimization (label propagation is not equivalent)
21. `small_world_sigma(g, n_rand)` — requires Erdos-Renyi random graph generator
22. `network_based_statistic(...)` — permutation testing on subgraph connectivity
23. `lcmv_beamformer(L, C)` — regularized spatial filter
24. `minimum_norm_estimate(L, M, lambda)` — regularized linear inverse

---

## Critical Architectural Note — `cross_spectral` Modification

The existing `cross_spectral` in `spectral.rs` computes MSC (magnitude-squared coherence) and discards the imaginary sign. To unlock PLI, wPLI, and imaginary coherence, `CrossSpectralResult` needs to be extended:

```rust
pub struct CrossSpectralResult {
    pub freqs: Vec<f64>,
    pub magnitude: Vec<f64>,
    pub phase: Vec<f64>,        // already present: atan2(Im, Re)
    pub coherence: Vec<f64>,    // already present: MSC
    // ADD:
    pub imaginary_part: Vec<f64>,  // Im(S_xy) / sqrt(Sxx * Syy) — signed imaginary coherence
    pub cross_spectrum: Vec<Complex>,  // full complex S_xy per bin — needed for PLI, wPLI
}
```

This is a backward-compatible extension (new fields). All existing callers continue to work.

---

## MVAR: The Single Highest-Leverage Gap

Everything in the effective connectivity category (Granger, PDC, DTF, spectral Granger) depends on `mvar_fit`. We have univariate `ar_fit` (Yule-Walker + Burg). The multivariate extension uses either:

1. **OLS stacking**: reshape VAR into a single regression system `Y = Z * B + E` where `Z` is the lagged regressor block matrix. We have `ols` and `ridge` in `multivariate.rs`. MVAR via OLS is literally 10 lines.
2. **Multivariate Yule-Walker** (Whittle 1963): generalizes `levinson_durbin` to matrix recursion. We have the scalar `levinson_durbin`.

Recommended approach: OLS stacking first (simple, correct, reuses existing primitives), then Whittle for the sharing story (intermediate `mvar_gram_matrix` shared across all MVAR consumers via TamSession).

---

## Summary

**What we have**: FFT, Hilbert transform, PSD, MSC coherence, transfer entropy, univariate AR, graph algorithms, clustering coefficient, modularity, information theory catalog.

**Tier 1 gaps** (trivial compositions, implement today): PLV, PLI, wPLI, imaginary coherence, envelope correlation, MVL, modulation index, phase-phase coupling, GLM-PAC, betweenness centrality, eigenvector centrality, rich club.

**Tier 2 gaps** (require MVAR keystone): Granger causality, PDC, DTF, spectral Granger. Implement `mvar_fit` via OLS stacking first — it unlocks 4 primitives immediately.

**Tier 3 gaps** (new algorithmic territory): bispectrum, Louvain, NBS, beamformer, MNE.

**Domain-specific boundary**: DCM (bilinear state-space + variational Laplace) and the electromagnetic forward model for source localization are the only genuinely domain-specific items. Everything else is general coupled-oscillator and graph mathematics applicable across seismology, econometrics, engineering, climate science, and acoustics.
