# Neuroimaging & Electrophysiology Math Gap Analysis

**Date**: 2026-04-10  
**Scope**: Mathematical primitives underlying neuroimaging/EEG/MEG tools — stripped of domain wrappers.  
**Method**: Grep of tambear/src/ cross-referenced with web research on FreeSurfer, ANTs, SPM, FSL, MNE-Python, MRtrix, Camino, DiPy, EEGLAB, FieldTrip, Brainstorm, Kilosort, DESeq2, MPlus.

---

## Section 1 — Primitives We Already Have (domain wrapper only needed)

These are general math already implemented in tambear. A neuroimaging domain wrapper is thin (call existing primitives, apply domain-specific parameter naming). No new math needed.

### 1.1 Linear Algebra
**PCA** (`dim_reduction.rs::pca`) — used by: DTI orientation analysis, spike sorting feature extraction, EEG artifact decomposition, gene expression matrices.  
**SVD** (`linear_algebra.rs::svd`) — used by: CCA, ICA pre-whitening, pseudoinverse in inverse solutions.  
**QR** (`linear_algebra.rs::qr`) — used by: GLM design matrix orthogonalization, MNE inverse.  
**Cholesky** (`linear_algebra.rs::cholesky`) — used by: noise covariance in MNE/LCMV, Kalman filtering in state-space.  
**Conjugate Gradient / GMRES** (`linear_algebra.rs::conjugate_gradient`, `gmres`) — used by: large BEM forward solves, high-resolution head models.  
**Eigendecomposition** (`linear_algebra.rs::sym_eigen`) — used by: Laplace-Beltrami operator (FreeSurfer spherical mapping), graph Laplacian spectral embedding, CCA.  
**Matrix exponential / log** (`linear_algebra.rs::matrix_exp`, `matrix_log`) — used by: diffeomorphic registration (log-domain Demons, geodesic SyN), Lie group updates.  
**Tridiagonal scan** (`linear_algebra.rs::solve_tridiagonal_scan`) — used by: spline fitting on cortical surfaces.  
**Gram-Schmidt** (`linear_algebra.rs::gram_schmidt_modified`) — used by: multi-taper orthogonal tapers (DPSS), beamformer null-space.

### 1.2 Interpolation & Approximation
**RBF interpolation** (`interpolation.rs::rbf_interpolate`) — used by: EEG channel interpolation (spherical splines are a special case: thin-plate spline on sphere), kriging for spatial transcriptomics.  
**B-spline** (`interpolation.rs::bspline_eval`) — used by: ANTs B-spline regularized diffeomorphic registration.  
**Cubic spline / PCHIP** (`interpolation.rs::natural_cubic_spline`, `pchip`) — used by: HRF convolution baseline, resampling LFP signals.  
**Gaussian Process regression** (`interpolation.rs::gp_regression`) — used by: spatial transcriptomics kriging, covariance structure estimation.

### 1.3 Signal Processing
**FFT / IFFT** (`signal_processing.rs::fft`, `ifft`, `rfft`) — universal.  
**Morlet CWT** (`signal_processing.rs::morlet_cwt`) — MNE-Python time-frequency analysis.  
**Multitaper PSD** (`spectral.rs::multitaper_psd`) — AFNI 3D spectral analysis, FieldTrip multitaper.  
**Welch PSD** (`signal_processing.rs::welch`) — fMRI resting-state power, EEG band power.  
**Cross-spectral density** (`spectral.rs::cross_spectral`) — coherence, imaginary coherence numerator.  
**Hilbert transform** (`signal_processing.rs::hilbert`) — analytic signal for instantaneous phase (PLV, PAC, envelope correlation).  
**Savitzky-Golay filter** (`signal_processing.rs::savgol_filter`) — LFP smoothing, CSD preprocessing.  
**Haar DWT / DB4** (`signal_processing.rs::haar_dwt`, `db4_dwt`) — wavelet decomposition for EEG artifact removal.  
**FIR filters** (`signal_processing.rs::fir_lowpass`, `fir_bandpass`) — EEG band extraction (delta/theta/alpha/beta/gamma).  
**ICA / FastICA** (`signal_processing.rs::fast_ica`) — EEGLAB/MNE artifact removal (ocular, cardiac).  
**EMD** (`signal_processing.rs::emd`) — empirical mode decomposition for EEG nonlinear analysis.  
**Autocorrelation / cross-correlation** (`signal_processing.rs::autocorrelation`, `cross_correlate`) — functional connectivity, HRF estimation.  
**Lomb-Scargle** (`spectral.rs::lomb_scargle`) — irregular-sampled spike trains, circadian rhythm.  
**Spectral band power** (`spectral.rs::band_power`, `relative_band_power`) — delta/theta/alpha/beta/gamma band extraction.  
**Spectral peaks** (`spectral.rs::spectral_peaks`) — dominant frequency detection.  
**Wigner-Ville** (`signal_processing.rs::wvd_features`) — high-resolution time-frequency for EEG.

### 1.4 Statistics & Hypothesis Testing
**GLM fit** (`hypothesis.rs::glm_fit`) — SPM voxel-wise GLM, fNIRS channel-wise GLM.  
**Permutation test** (`nonparametric.rs::permutation_test_mean_diff`) — FieldTrip cluster-based permutation, FSL randomise.  
**Bootstrap** (`nonparametric.rs::bootstrap_percentile`) — confidence intervals on connectivity measures.  
**KS test two-sample** (`nonparametric.rs::ks_test_two_sample`) — GSEA enrichment score base.  
**Multiple comparisons** (`hypothesis.rs::bonferroni`, `holm`, `benjamini_hochberg`) — voxel-wise FDR correction.  
**LME / mixed effects** (`mixed_effects.rs::lme_random_intercept`) — longitudinal neuroimaging, repeated-measures fMRI.  
**Logistic regression** (`hypothesis.rs::logistic_regression`) — brain-behavior classification.  
**WLS** (`multivariate.rs::wls`) — weighted GLM for heteroscedastic imaging data.

### 1.5 Multivariate
**LDA** (`multivariate.rs::lda`) — EEG BCI (CSP is a variant of LDA in sensor-space).  
**CCA** (`multivariate.rs::cca`) — cross-modal neuroimaging.  
**MANOVA** (`multivariate.rs::manova`) — multivariate voxel-space tests.  
**Mahalanobis distance** (`multivariate.rs::mahalanobis_distances`) — noise normalization in MNE dSPM.  
**Ridge regression** (`multivariate.rs::ridge`) — regularized inverse in MNE.  
**Covariance matrix** (`multivariate.rs::covariance_matrix`) — noise covariance estimation for MNE/LCMV.

### 1.6 Information Theory
**Transfer entropy** (`information_theory.rs::transfer_entropy`) — Granger-like directed connectivity.  
**Mutual information** (`information_theory.rs::mutual_information`, `normalized_mutual_information`) — ICA independence criterion.  
**KL divergence** (`information_theory.rs::kl_divergence`) — Tort PAC modulation index.  
**Shannon entropy** (`information_theory.rs::shannon_entropy`) — spectral entropy, spike train entropy.  
**Wasserstein 1D** (`information_theory.rs::wasserstein_1d`) — shape distance between ODFs, persistence diagrams.

### 1.7 Graph Theory
**Betweenness centrality** — NOTE: `graph.rs` has `closeness_centrality` and `degree_centrality` but NOT betweenness. **Gap — see Section 2.**  
**Clustering coefficient** (`graph.rs::clustering_coefficient`) — small-worldness numerator.  
**Modularity** (`graph.rs::modularity`) — brain community detection.  
**Shortest paths** (`graph.rs::floyd_warshall`, `dijkstra`) — global efficiency = harmonic mean of path lengths.  
**MST** (`graph.rs::kruskal`, `prim`) — minimum spanning tree for connectivity.  
**Graph Laplacian** (`graph.rs::graph_laplacian`) — spectral graph embedding.  
**PageRank** (`graph.rs::pagerank`) — hub detection in connectome.

### 1.8 Topological Data Analysis
**Persistent homology H0/H1** (`tda.rs::rips_h0`, `rips_h1`) — TDA on brain networks.  
**Bottleneck / Wasserstein distance** (`tda.rs::bottleneck_distance`, `wasserstein_distance`) — comparing persistence diagrams.

### 1.9 Dimensionality Reduction
**t-SNE** (`dim_reduction.rs::tsne`) — single-cell gene expression visualization.  
**MDS** (`dim_reduction.rs::classical_mds`) — structural similarity analysis.  
**NMF** (`dim_reduction.rs::nmf`) — functional connectivity factorization.  
**KDE** (`nonparametric.rs::kde`) — spike density function, neural rate maps.

### 1.10 Survival / Time-to-Event
**Kaplan-Meier, Cox** (`survival.rs`) — already present; used in clinical neuroimaging studies.

---

## Section 2 — General Math Primitives Missing from Tambear

These are general-purpose math that tambear DOES NOT yet have, discovered via the neuroimaging tools above. Each is useful beyond neuroscience.

### 2.1 Graph Theory Gaps

**`betweenness_centrality(g: &Graph) -> Vec<f64>`**  
Math: for each node v, fraction of all shortest paths (σ(s,t)) passing through v: BC(v) = Σ_{s≠v≠t} σ(s,t|v)/σ(s,t). Brandes algorithm: O(V·E).  
Tools: brain connectome analysis (Brain Connectivity Toolbox), NBS.  
Composes from: `floyd_warshall` or iterative BFS/Dijkstra + accumulate over paths.

**`global_efficiency(g: &Graph) -> f64`**  
Math: E_glob = (1/(N(N-1))) Σ_{i≠j} 1/d(i,j). Harmonic mean of inverse path lengths.  
Tools: brain network efficiency (Latora-Marchiori measure), connectomics.  
Composes from: `floyd_warshall` + harmonic mean.

**`local_efficiency(g: &Graph) -> Vec<f64>`**  
Math: E_loc(i) = E_glob of the subgraph of i's neighbors.  
Tools: local integration measure in brain networks.  
Composes from: `betweenness_centrality` + subgraph extraction.

**`rich_club_coefficient(g: &Graph, k_thresholds: &[usize]) -> Vec<f64>`**  
Math: φ(k) = 2E_{>k} / (N_{>k} · (N_{>k}-1)). Ratio of edges among high-degree nodes vs maximum possible.  
Tools: brain hub organization (Van Den Heuvel 2011), connectomics.  
Composes from: degree sequence + induced subgraph edge count.

**`participation_coefficient(g: &Graph, community: &[usize]) -> Vec<f64>`**  
Math: PC(i) = 1 - Σ_m (k_{im}/k_i)^2 where k_{im} = edges from i to module m. Measures how distributed a node's connections are across modules.  
Tools: connector hub detection in brain networks, FieldTrip.  
Composes from: `modularity` + per-node module edge counts.

**`network_based_statistic(g_obs: &Graph, g_null: &[Graph], threshold: f64) -> Vec<usize>`**  
Math: suprathreshold edge subgraph → connected components → permutation-test the largest component size. Zalesky 2010.  
Tools: FieldTrip NBS, mass-univariate edge testing without overcorrection.  
Composes from: `connected_components` + `permutation_test`.

**`small_world_sigma(g: &Graph, n_random: usize) -> f64`**  
Math: σ = (C/C_rand) / (L/L_rand). Ratio of clustering coefficient to path length, normalized by random graph baselines.  
Tools: brain small-world characterization.  
Composes from: `clustering_coefficient`, `diameter`, random graph generation.

### 2.2 Spherical Harmonics

**`spherical_harmonic_real(l: usize, m: i32, theta: f64, phi: f64) -> f64`**  
Math: Y_l^m(θ,φ) = N_lm · P_l^|m|(cos θ) · {cos(mφ), sin(|m|φ)} for real basis. Associated Legendre polynomials + normalization.  
Tools: FreeSurfer cortical surface parameterization, MRtrix ODF representation (SH basis), EEG/MEG dipole source expansion, EEGLAB spherical spline interpolation.  
Composes from: `associated_legendre_polynomial` (missing), factorial/normalization constants.

**`associated_legendre_polynomial(l: usize, m: usize, x: f64) -> f64`**  
Math: P_l^m(x) via recurrence. Used in SH basis.  
Composes from: basic polynomial recursion.

**`sh_coefficients_from_samples(theta: &[f64], phi: &[f64], values: &[f64], l_max: usize) -> Vec<f64>`**  
Math: Least-squares fit in SH basis. Often regularized (Tikhonov). Used for ODF representation in diffusion MRI.  
Composes from: `spherical_harmonic_real` + `lstsq` or `ridge`.

**`sh_reconstruct(coeffs: &[f64], theta: &[f64], phi: &[f64]) -> Vec<f64>`**  
Math: Σ_{l,m} c_lm · Y_l^m(θ,φ). Inverse of SH fit.  
Composes from: `spherical_harmonic_real`.

### 2.3 Boundary Element Method (BEM)

**`bem_matrix_collocation(triangles: &[Triangle], conductivities: &[f64]) -> Mat`**  
Math: Assembles the BEM system matrix via constant-basis or linear-basis Galerkin/collocation weighting of the single-layer and double-layer potential operators. Each matrix element = ∫_Γ G(r,r') dS' integrated over pairs of triangles. O(N²) assembly, O(N³) solve. Used in EEG/MEG forward modeling.  
Tools: MNE-Python forward model, FieldTrip BEM.  
Composes from: triangle-to-triangle solid angle integrals, `cholesky` or `lu` for solve.  
Difficulty: HIGH. Requires singular integral regularization for self-terms, quadrature on curved panels.  
Note: genuinely domain-specific in parameterization but the math (Galerkin BEM on surface meshes) appears in acoustics, electromagnetics, and elasticity — not neuroscience-only.

**`fast_multipole_bem(triangles: &[Triangle], charges: &[f64], order: usize) -> Vec<f64>`**  
Math: FMM acceleration of BEM matrix-vector products from O(N²) to O(N log N). Tree decomposition of source/target panels, multipole expansion, local expansion, L2L/M2M/M2L translation operators.  
Tools: 2024 fast BEM for high-resolution head models (Makarov et al.).  
Composes from: tree spatial indexing + multipole coefficient accumulation.  
Difficulty: VERY HIGH.

### 2.4 Phase Synchrony Measures

**`phase_locking_value(phase_x: &[f64], phase_y: &[f64]) -> f64`**  
Math: PLV = |1/N Σ_t exp(i(φ_x(t) - φ_y(t)))| = |mean of complex phase differences|.  
Tools: MNE-Python `spectral_connectivity`, EEGLAB, FieldTrip.  
Composes from: `hilbert` (get phase) → complex mean → magnitude.

**`phase_lag_index(phase_x: &[f64], phase_y: &[f64]) -> f64`**  
Math: PLI = |E[sign(Im(C_xy(f)))]| where C_xy is the cross-spectrum. Insensitive to zero-lag (volume conduction) interactions.  
Tools: MNE, FieldTrip. Vinck 2011.  
Composes from: `cross_spectral` → imaginary part → sign → mean.

**`weighted_phase_lag_index(cross_spectra: &[Complex]) -> f64`**  
Math: wPLI = |E[Im(C_xy)]| / E[|Im(C_xy)|]. Weights phase differences by imaginary cross-spectral magnitude.  
Tools: MNE-Python, FieldTrip. Vinck 2011.  
Composes from: imaginary cross-spectrum → weighted mean.

**`debiased_squared_wpli(cross_spectra: &[Complex]) -> f64`**  
Math: dwPLI² = (N · wPLI² - 1) / (N - 1). Corrects for sample-size inflation.  
Tools: MNE-Python.  
Composes from: `weighted_phase_lag_index`.

**`imaginary_coherence(cross_spectra: &[Complex]) -> f64`**  
Math: ImC = Im(C_xy) / sqrt(|C_xx| · |C_yy|). Normalized imaginary part of coherency.  
Tools: MNE, FieldTrip. Nolte 2004.  
Composes from: `cross_spectral` → normalize imaginary part.

**`envelope_correlation(x: &[f64], y: &[f64]) -> f64`**  
Math: Pearson r between |analytic(x)| and |analytic(y)|. Often with orthogonalization: regress out zero-lag component first.  
Tools: MNE-Python source connectivity, MEG resting-state.  
Composes from: `hilbert` → `envelope` → `pearson_r`.

### 2.5 Cross-Frequency Coupling (PAC)

**`pac_mean_vector_length(phase_low: &[f64], amp_high: &[f64]) -> f64`**  
Math: MVL = |1/N Σ_t A(t) · exp(i·φ(t))|. Canolty 2006. Mean of amplitude-weighted phase vector.  
Tools: MNE-Python, Brainstorm, FieldTrip.  
Composes from: `hilbert` (both bands) → complex inner product.

**`pac_modulation_index(phase_low: &[f64], amp_high: &[f64], n_bins: usize) -> f64`**  
Math: MI = KL(A_distribution ‖ Uniform) / log(n_bins). Distribute high-frequency amplitudes into phase bins, compute KL from uniform. Tort 2010.  
Tools: Brainstorm, MNE, EEGLAB winPACT.  
Composes from: `hilbert` → phase binning → `kl_divergence`.

**`pac_glm(phase_low: &[f64], amp_high: &[f64]) -> f64`**  
Math: Regress A(t) ~ sin(φ(t)) + cos(φ(t)) + intercept; R² or F-statistic is the PAC measure. Cohen 2008.  
Composes from: `glm_fit`.

**`phase_amplitude_coupling_surrogate(phase_low: &[f64], amp_high: &[f64], n_surrogates: usize) -> f64`**  
Math: Normalize MI by comparing to distribution of MI under random temporal shifts of phase or amplitude time series.  
Composes from: `pac_modulation_index` + random circular shifts.

### 2.6 Beamformer Inverse Solutions

**`lcmv_beamformer(fwd: &Mat, cov: &Mat, reg: f64) -> Mat`**  
Math: LCMV spatial filter W = (C⁻¹L) / (L'C⁻¹L) where L is the leadfield column (forward solution), C is the data covariance. Output power at location r: P(r) = W'CW. Regularization: C_reg = C + reg·trace(C)/n · I.  
Tools: MNE-Python, FieldTrip, Brainstorm.  
Composes from: `inv` (with Tikhonov regularization) + matrix products.

**`dics_beamformer(fwd: &Mat, csd: &Mat, reg: f64) -> Mat`**  
Math: Like LCMV but in frequency domain. Uses cross-spectral density matrix instead of time-domain covariance. Gross 2001.  
Composes from: `cross_spectral` → `lcmv_beamformer`.

**`mne_inverse(fwd: &Mat, noise_cov: &Mat, source_cov: &Mat, snr: f64) -> Mat`**  
Math: W = ΓL'(LΓL' + λ²C_n)⁻¹ where Γ = source covariance, C_n = noise covariance, λ² = 1/SNR². Standard minimum norm estimate (MNE, Lin 2006).  
Composes from: `cholesky_solve` + matrix products.

**`dspm_normalize(mne_weights: &Mat, fwd: &Mat, noise_cov: &Mat) -> Mat`**  
Math: dSPM = W_MNE / sqrt(W_MNE · C_n · W_MNE'). Noise-normalized MNE. Mahalanobis normalization over the sensor noise.  
Composes from: `mne_inverse` + `mahalanobis_distances`.

**`sloreta_normalize(mne_weights: &Mat, fwd: &Mat, noise_cov: &Mat) -> Mat`**  
Math: sLORETA = W_MNE / sqrt(diag(W_MNE · C_data · W_MNE')). Resolution-normalized MNE (Pascual-Marqui 2002). Standardized by estimated source covariance.  
Composes from: `mne_inverse` + diagonal of spatial filter covariance.

### 2.7 Diffeomorphic Registration Math

**`image_jacobian(deform_field: &[Vec<f64>], voxel_size: &[f64]) -> Vec<f64>`**  
Math: J(x) = det(∂φ/∂x). Jacobian determinant of a deformation field. Computed via finite differences on displacement field. Used to measure local volume change (ANTs).  
Composes from: numerical Jacobian (finite differences) + `det`.

**`velocity_to_diffeomorphism(v: &[Vec<f64>], n_steps: usize) -> Vec<Vec<f64>>`**  
Math: Integration of stationary velocity field via scaling-and-squaring: φ = exp(v). φ^(1/2^n) = v/2^n (small displacement), then compose n times. Arsigny 2006.  
Composes from: field composition + `matrix_exp` (pointwise).

**`image_cross_correlation_gradient(fixed: &[f64], moving: &[f64], sigma: f64) -> Vec<f64>`**  
Math: -∂/∂φ CC(I,J∘φ) = -∂CC/∂J · ∇(J∘φ). Closed-form gradient of local cross-correlation similarity metric. Used in SyN registration.  
Composes from: Gaussian blur, image gradient, pointwise products.

**`laplace_beltrami_eigenvalues(vertices: &[[f64; 3]], faces: &[[usize; 3]], k: usize) -> (Vec<f64>, Vec<Vec<f64>>)`**  
Math: Cotangent-weight discrete Laplace-Beltrami operator on triangle mesh. L_ij = -½(cot α_ij + cot β_ij) for adjacent vertices. Eigendecompose. Used for surface registration, shape analysis, cortical parameterization.  
Composes from: cotangent weight assembly (sparse matrix) + `sym_eigen` (or iterative solver for k smallest).  
Difficulty: HIGH (sparse matrix eigendecomposition).

### 2.8 Marching Cubes / Surface Extraction

**`marching_cubes(volume: &[f64], dims: [usize;3], iso_value: f64) -> (Vec<[f64;3]>, Vec<[usize;3]>)`**  
Math: For each 2×2×2 voxel cube: look up 256-case table based on which corners exceed iso_value; generate triangle vertices by linear interpolation along edges. Lorensen & Cline 1987.  
Tools: FreeSurfer white/pial surface extraction, 3D Slicer.  
Difficulty: MEDIUM. Pure geometry + table lookup.  
Note: used in any isosurface extraction (medical, scientific visualization, implicit surfaces). Not neuroscience-specific.

**`mesh_smooth_laplacian(vertices: &[[f64;3]], faces: &[[usize;3]], n_iter: usize, lambda: f64) -> Vec<[f64;3]>`**  
Math: v_i ← v_i + λ · (Σ_j∈N(i) w_ij v_j - v_i). Iterative averaging of vertex positions with neighbors. Used for cortical surface smoothing post-marching-cubes.  
Composes from: mesh adjacency + weighted vertex average.

**`mean_curvature(vertices: &[[f64;3]], faces: &[[usize;3]]) -> Vec<f64>`**  
Math: H(v) = (1/2A) Σ_j∈N(v) (cot α_j + cot β_j)(v - v_j). Cotangent-weight mean curvature. Used as cortical shape feature.  
Composes from: cotangent weight computation + mesh area.

**`gaussian_curvature(vertices: &[[f64;3]], faces: &[[usize;3]]) -> Vec<f64>`**  
Math: K(v) = (2π - Σ_f θ_f) / A(v). Angle defect / vertex area. Used in Gauss-Bonnet and topology analysis.  
Composes from: angle computation at vertex in each incident triangle.

### 2.9 Current Source Density

**`csd_1d(lfp_depth: &[f64], spacing: f64) -> Vec<f64>`**  
Math: CSD(z) = -∂²V/∂z². Second spatial derivative via finite differences. Applied to laminar LFP recordings.  
Composes from: second-order finite difference stencil.

**`csd_surface_laplacian(signal: &[f64], positions: &[[f64;2]], lambda: f64) -> Vec<f64>`**  
Math: Spherical spline surface Laplacian for scalp EEG. Fits a spherical spline to electrode potentials, takes analytic second derivative. Perrin 1989.  
Composes from: `sh_coefficients_from_samples` + analytic second derivative of SH.

### 2.10 Spatial Autocorrelation (Grid Cells / Place Fields)

**`rate_map_2d(spikes: &[(f64,f64)], positions: &[(f64,f64)], sigma: f64, bins: usize) -> Vec<f64>`**  
Math: Gaussian kernel density estimate of spike rate on 2D position grid. KDE applied to occupancy-normalized spike map.  
Composes from: `kde` generalized to 2D (or bivariate KDE).

**`spatial_autocorrelation_2d(rate_map: &[f64], bins: usize) -> Vec<f64>`**  
Math: r(dx,dy) = (ΣΣ (f(x,y)-f̄)(f(x+dx,y+dy)-f̄)) / (N·σ²). 2D autocorrelation of firing rate map. Used for grid cell analysis.  
Composes from: `fft2d` + pointwise magnitude squared → `ifft2d` (autocorrelation theorem).

**`grid_score(autocorr: &[f64], bins: usize) -> f64`**  
Math: Rotate autocorrelogram at 60°, 120°, 180°, 240°, 300°. Grid score = min(r(60°), r(120°)) - max(r(180°)). Sargolini 2006.  
Composes from: `spatial_autocorrelation_2d` + image rotation + pearson_r on masked annulus.

### 2.11 Negative Binomial GLM (Transcriptomics)

**`neg_binomial_glm_fit(X: &Mat, y: &[u64], size_factors: &[f64]) -> NbGlmResult`**  
Math: y ~ NegBin(μ, α) with log link μ_i = s_i · exp(X_i β). IRLS (iteratively reweighted least squares) with working response z = η + W⁻¹(y-μ), weight W = μ²/(μ + αμ²). DESeq2/edgeR style.  
Tools: DESeq2, edgeR. Used for brain atlas gene expression (Allen Brain Atlas), single-cell transcriptomics.  
Composes from: `neg_binomial_pmf` + IRLS (composable from `wls`).

**`shrinkage_dispersion_estimate(dispersions: &[f64], means: &[f64]) -> Vec<f64>`**  
Math: Fit trend line dispersion ~ f(mean) in log space. Then compute MAP estimate: α_MAP = argmax P(y|α) · P(α|trend). Empirical Bayes shrinkage toward fitted trend. DESeq2.  
Composes from: `gp_regression` or simple nonlinear fit + log-normal prior.

**`wald_test_nb(beta: f64, se_beta: f64) -> f64`**  
Math: z = β/SE(β), p = 2·Φ(-|z|). Already composable from `normal_sf`.  
**Already have:** `special_functions.rs::normal_sf`. Just needs a wrapper.

### 2.12 Structural Equation Modeling

**`sem_fit_lisrel(Lambda: &Mat, Theta: &Mat, Psi: &Mat, Beta: &Mat, S: &Mat, estimator: SemEstimator) -> SemResult`**  
Math: LISREL model: Σ(θ) = Λ(I-B)⁻¹Ψ(I-B)⁻ᵀΛᵀ + Θ. ML estimator: minimize F_ML = log|Σ| + tr(SΣ⁻¹) - log|S| - p. Gradient via chain rule through matrix inverse. MPlus.  
Tools: MPlus, R lavaan, structural neuroimaging path models.  
Composes from: `inv`, `mat_mul`, `log_det`, `sym_eigen`.

**`wlsmv_polychoric_weight_matrix(S: &Mat, polychoric: &Mat, threshold: &[f64]) -> Mat`**  
Math: Diagonal weight matrix from asymptotic variances of polychoric correlations. Used for WLSMV estimation with ordinal indicators.  
Composes from: polychoric correlation (missing — see below) + asymptotic variance formula.

**`polychoric_correlation(x: &[i32], y: &[i32], n_cats_x: usize, n_cats_y: usize) -> f64`**  
Math: MLE of Pearson r of underlying bivariate normal given observed ordinal categories and thresholds. Two-step: (1) estimate thresholds from marginals via normal quantile, (2) maximize bivariate normal likelihood over r.  
Tools: MPlus, R psych package. Used for ordinal SEM, psychometrics, IRT linking.  
Composes from: `normal_quantile`, `regularized_incomplete_beta` (for bivariate normal integral).  
**NOTE**: related to `tetrachoric` (already in `nonparametric.rs`) which is the 2×2 special case. Polychoric is the k×l generalization.

### 2.13 GSEA Running-Sum Statistic

**`gsea_enrichment_score(ranked_stats: &[f64], gene_set_mask: &[bool], p: f64) -> f64`**  
Math: Walk ranked list; when gene in set: add |stat|^p / sum_in_set; else subtract 1/(N-|set|). ES = max deviation from zero. Weighted KS statistic (Subramanian 2005).  
Tools: DESeq2 → GSEA, fgsea (fast GSEA via permutation). Used in spatial transcriptomics brain atlas.  
Composes from: prefix scan (accumulate) + max operation.

**`gsea_permutation_pvalue(es_obs: f64, es_null: &[f64]) -> f64`**  
Math: p = |{es_null ≥ es_obs}| / N_perm for positive ES. Standard permutation p-value.  
Composes from: count + ratio.

### 2.14 Random Field Theory (RFT)

**`euler_characteristic_density(t: f64, df: f64, d: usize) -> f64`**  
Math: EC density ρ_d(t) for a t-field in d-dimensional smooth domain. Uses Worsley (1996) formula involving chi-squared tail probabilities and Hermite polynomials. Expected EC ≈ search_volume × EC_density gives voxel-corrected p-value.  
Tools: SPM cluster correction, FSL GRF.  
Composes from: `chi2_sf`, `normal_sf`, `gamma` + Hermite polynomial evaluation.  
Difficulty: MEDIUM (complex formula but well-specified).

**`resels_from_fwhm(fwhm: &[f64], voxel_dims: &[f64], volume: usize) -> f64`**  
Math: RESEL = resolution element. V_resels = volume / prod(FWHM_i / voxel_size_i). Used to express search volume in smoothness-adjusted units.  
Composes from: simple arithmetic.

**`smoothness_from_residuals(residuals: &[Vec<f64>], mask: &[bool]) -> Vec<f64>`**  
Math: Estimate smoothness from spatial covariance of residual images: FWHM = sqrt(-8 ln 2 / (2 ln(r(1)))) where r(1) is correlation at lag 1 in each direction. Forman 1995.  
Composes from: spatial autocorrelation at lag 1 in each dimension.

---

## Section 3 — Domain-Specific Kernels (Neuroscience-Specific But Implementable as Math)

These have narrow domain specificity but are still computable math that belongs in the tambear catalog.

### 3.1 Hemodynamic Response Function (HRF)

**`hrf_canonical(t: &[f64]) -> Vec<f64>`**  
Math: Double-gamma: h(t) = (t/d1)^a1 · exp(-(t-d1)/b1) - c·(t/d2)^a2 · exp(-(t-d2)/b2). Default Glover/SPM parameters. Convolved with stimulus to model BOLD response.  
Composes from: `gamma_pdf` (already have).

**`hrf_convolution(stimulus: &[f64], hrf: &[f64]) -> Vec<f64>`**  
Math: BOLD ~ stimulus * hrf (convolution). Already have `convolve`. Just a wrapper with downsampling.

**`hrf_basis_set(t: &[f64], n_derivatives: usize) -> Vec<Vec<f64>>`**  
Math: Canonical HRF + temporal derivative + dispersion derivative. Used for flexible HRF modeling in SPM GLM.  
Composes from: `hrf_canonical` + numerical derivative.

### 3.2 Modified Beer-Lambert Law (fNIRS)

**`mbll_concentration(delta_od: &[f64], wavelengths: &[f64], dpf: f64, extinction: &Mat) -> Vec<f64>`**  
Math: ΔC = ε⁻¹ · ΔOD / (DPF · L) where ε is the extinction coefficient matrix (HbO/HbR at each wavelength), DPF = differential pathlength factor. Linear system solve.  
Composes from: `inv` or `lstsq`.

### 3.3 Diffusion Tensor Estimation

**`dti_tensor(dwi_signals: &[f64], b_vectors: &[[f64;3]], b_values: &[f64], b0: f64) -> [f64;6]`**  
Math: log(S/S₀) = -b · (g' D g) for each gradient direction g. D is 3×3 symmetric positive definite tensor (6 free parameters). Linear fit in log space via OLS: b_matrix · d_vec = log_signals.  
Composes from: `lstsq` or `qr_solve`.

**`dti_metrics(tensor: &[f64;6]) -> DtiMetrics`**  
Math: Eigendecompose 3×3 symmetric tensor → λ1, λ2, λ3. FA = sqrt(3/2 · ((λ1-λ̄)²+(λ2-λ̄)²+(λ3-λ̄)²)/(λ1²+λ2²+λ3²)). MD = (λ1+λ2+λ3)/3. AD = λ1. RD = (λ2+λ3)/2.  
Composes from: `sym_eigen` on 3×3.

**`constrained_spherical_deconvolution(dwi: &[f64], b_vecs: &[[f64;3]], b_vals: &[f64], response_fn: &[f64], l_max: usize) -> Vec<f64>`**  
Math: y = X·f where X is the SH-based forward matrix from the single-fiber response function. NNLS (non-negative least squares) with SH regularization. Tournier 2007.  
Composes from: `sh_coefficients_from_samples` + `lstsq` with non-negativity constraint.

### 3.4 Tractography

**`streamline_euler(odf: &[Vec<f64>], seed: [f64;3], step_size: f64, max_steps: usize) -> Vec<[f64;3]>`**  
Math: Euler integration along peak of ODF field. At each point: find peak direction in ODF → step.  
Composes from: ODF peak finding + 3D Euler integration.

**`iFOD2_probabilistic_step(fod: &[f64], prev_dir: [f64;3], kappa: f64) -> [f64;3]`**  
Math: Sample candidate steps from Watson distribution (concentration κ) around current direction. Accept/reject by FOD amplitude at sampled direction. Tournier 2010.  
Composes from: Watson distribution sampler + FOD interpolation.

### 3.5 EEG/MEG Source Localization Components

**`leadfield_dipole(r_source: [f64;3], r_sensor: [f64;3]) -> [f64;3]`**  
Math: Analytical free-space dipole forward model (simplified): B = μ₀/(4π) · (d × (r-r')) / |r-r'|³. Full head model uses BEM.  
Composes from: cross-product + norm.

**`dipole_fitting(fwd: &Mat, data: &[f64]) -> ([f64;3], [f64;3])`**  
Math: Nonlinear least squares over dipole location. Fixed orientation: linear solve at each candidate location, nonlinear search over 3D space.  
Composes from: `lstsq` + `optimization`.

### 3.6 Sharp Wave Ripple Detection (SWR)

**`detect_ripples(lfp: &[f64], fs: f64, ripple_band: (f64,f64), threshold_sd: f64) -> Vec<(usize,usize)>`**  
Math: (1) bandpass filter to ripple band (typically 80-180 Hz), (2) envelope, (3) threshold crossing at k·std, (4) merge overlapping events.  
Composes from: `fir_bandpass` + `envelope` + threshold + merge intervals.

### 3.7 Directed Transfer Function / Partial Directed Coherence

**`directed_transfer_function(ar_coeffs: &[Mat], noise_cov: &Mat, n_freqs: usize) -> Vec<Mat>`**  
Math: DTF(f) = |H(f)|² normalized by row. H(f) = A(f)⁻¹ where A(f) = I - Σ_k A_k exp(-2πifk). Kaminski 1991. Used in FieldTrip directed connectivity.  
Composes from: `ar_fit` (MVAR) + matrix inverse at each frequency.

**`partial_directed_coherence(ar_coeffs: &[Mat], noise_cov: &Mat, n_freqs: usize) -> Vec<Mat>`**  
Math: PDC(f) = A_ij(f) / sqrt(Σ_k |A_kj(f)|²). Column-normalized spectral matrix of AR coefficients. Baccalá 2001.  
Composes from: MVAR AR coefficients + FFT + column normalization.

---

## Summary Table

| Category | Already in Tambear | Missing — General Math | Missing — Domain-Specific |
|---|---|---|---|
| Linear algebra | Full (LU, QR, SVD, Cholesky, CG, GMRES, matrix exp) | — | — |
| Graph theory | Degree, clustering, modularity, PageRank, MST, paths | Betweenness, global/local efficiency, rich club, participation coefficient, NBS, small-world σ | — |
| Spherical harmonics | — | Y_l^m, associated Legendre, SH fit/reconstruct | — |
| Signal processing | FFT, CWT, Hilbert, filters, ICA, multitaper, Welch | — | CSD surface Laplacian (requires SH) |
| Phase synchrony | Autocorrelation, cross-spectra | PLV, PLI, wPLI, debiased wPLI², imaginary coherence, envelope correlation | — |
| Cross-frequency coupling | KL divergence, Hilbert | PAC MVL, PAC modulation index (Tort), PAC GLM, surrogate normalization | — |
| Beamformers / inverse | Ridge, Cholesky, covariance | LCMV, DICS, MNE inverse, dSPM normalize, sLORETA | — |
| Diffeomorphic registration | Matrix log/exp | Jacobian determinant of deformation field, velocity→diffeomorphism (scaling+squaring), image cross-correlation gradient | — |
| Surface geometry | — | Marching cubes, Laplace-Beltrami eigenvalues, mean/Gaussian curvature, mesh smoothing | Cortical surface–specific defaults |
| Current source density | — | CSD 1D (second spatial derivative), CSD surface Laplacian | — |
| NB GLM (transcriptomics) | Neg binomial PMF/CDF, WLS | NB-IRLS GLM fit, shrinkage dispersion (empirical Bayes), polychoric correlation | GSEA running-sum statistic |
| Random field theory | Normal/chi2 CDFs, gamma | Euler characteristic density, resel computation, smoothness from residuals | — |
| SEM | Factor analysis, LME | LISREL Σ(θ) model, WLSMV weight matrix | Polychoric generalization |
| Spatial analysis | Kriging, KDE, spatial autocorrelation | 2D KDE (bivariate), 2D spatial autocorrelation (via fft2d), grid score | Rate maps (place/grid cells) |
| HRF | Gamma PDF, convolution | — | Canonical HRF, HRF basis set |
| fNIRS | lstsq | — | Modified Beer-Lambert Law |
| DTI | lstsq, sym_eigen | — | Tensor estimation, FA/MD/AD/RD metrics, CSD |
| Tractography | — | — | Streamline Euler, iFOD2 probabilistic |
| Directed connectivity | AR fit | MVAR extension of AR fit, DTF, PDC | — |
| SWR detection | FIR filter, envelope | — | Ripple detection pipeline |

---

## Priority Order for Implementation

**Highest leverage** (used across many tools, general math, no neuroscience-specific knowledge needed):

1. **Betweenness centrality** — missing from graph.rs despite having all path primitives
2. **Global/local efficiency, rich club, participation coefficient, small-world sigma** — graph.rs gaps, computable from existing paths
3. **Spherical harmonics** (Y_l^m + associated Legendre + SH fit/reconstruct) — unlocks FreeSurfer surface analysis, MRtrix ODF, EEG interpolation
4. **PLV, PLI, wPLI, imaginary coherence, envelope correlation** — compose from existing Hilbert + cross_spectral
5. **PAC measures** (MVL, MI, GLM variant) — compose from existing Hilbert + KL divergence
6. **LCMV beamformer + MNE inverse + dSPM + sLORETA** — compose from existing Cholesky + ridge
7. **Marching cubes** — pure geometry, general isosurface extraction
8. **Laplace-Beltrami eigenvalues** — requires sparse Laplacian + eigensolver
9. **NB GLM fit (IRLS)** — composes from existing WLS + neg_binomial_pmf
10. **Polychoric correlation** — generalizes tetrachoric (already have)
11. **GSEA running-sum statistic** — pure accumulate operation
12. **RFT Euler characteristic density** — pure formula composition from existing CDF functions
13. **DTF / PDC** — composes from MVAR AR fit extension + FFT

**Lower priority** (domain-specific wrappers or very high difficulty):
- BEM matrix assembly / FMM (VERY HIGH difficulty, narrow usage)
- Diffeomorphic registration math (requires image data structures not yet in tambear)
- Tractography (requires 3D ODF field data structure)
- HRF, fNIRS MBLL (thin wrappers on existing math)
