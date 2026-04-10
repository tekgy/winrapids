# Spatial Transforms, Multiple Comparison Corrections, and N-D Array Operations — Gap Analysis

**Date**: 2026-04-10  
**Scope**: Rigid/affine/deformable transforms, multiple comparison corrections, N-D generalizations  
**Key insight**: Most of these are the SAME primitive at different dimensionalities. The question is whether tambear has the N-D generalization or only the 1D special case.

---

## Current State in tambear/src/

Quick inventory from grep:

| File | Relevant content |
|------|-----------------|
| `signal_processing.rs` | FFT 1D, FFT 2D (separable row/col), convolve (1D), Morlet CWT (1D), Haar/DB4 DWT (1D) |
| `hypothesis.rs` | bonferroni, holm, benjamini_hochberg — NO hochberg, storey, BY, RFT, cluster-perm, TFCE |
| `linear_algebra.rs` | SVD (one-sided Jacobi), LU, QR, Cholesky, eigendecomp (symmetric Jacobi) — NO polar decomposition, NO Procrustes |
| `interpolation.rs` | Lagrange, Newton, cubic spline, RBF, B-spline (1D), GP regression (1D) — NO N-D versions |
| `spatial.rs` | Geostatistics, Kriging, Moran's I — entirely 2D; this is the spatial STATISTICS module, not spatial TRANSFORMS |
| `manifold.rs` | Poincaré/sphere/geodesic distance operators, ManifoldMixture — geometry without transform arithmetic |
| `multivariate.rs` | covariance, LDA, CCA, Mahalanobis, ridge/lasso — no geometric transforms |

**Nothing in tambear implements**: rotation matrices, quaternions, SE(3), affine transform matrices, deformable fields, or any geometric transform composition.

---

## Part 1: Rigid Body Transforms (6 DOF)

### What's needed

**Rotation representations** — three equivalent representations, each with different composition properties:

| Representation | Storage | Composition | Interpolation |
|---------------|---------|-------------|---------------|
| Rotation matrix R ∈ SO(3) | 3×3 (9 floats) | matrix multiply | matrix logarithm |
| Euler angles (φ,θ,ψ) | 3 floats | trig compose (gimbal lock danger) | lerp (path-dependent) |
| Axis-angle (n̂, θ) | 4 floats | Rodrigues formula | lerp θ, slerp n̂ |
| Quaternion q ∈ S³ | 4 floats | Hamilton product | SLERP |

**Rodrigues' rotation formula**: R = I + sin(θ)K + (1-cos(θ))K² where K is the skew-symmetric matrix of axis n̂.

**SE(3) group**: 4×4 homogeneous transform T = [R|t; 0|1]. Composition = matrix multiply. Inverse: T⁻¹ = [Rᵀ| -Rᵀt; 0|1].

**SLERP** (spherical linear interpolation): q(t) = q₁(q₁⁻¹q₂)ᵗ = q₁·sin((1-t)Ω)/sin(Ω) + q₂·sin(tΩ)/sin(Ω) where Ω = arccos(q₁·q₂).

**Matrix logarithm for SE(3)**: needed for geodesic interpolation in Lie algebra se(3). Log map: log(R) = θ/(2sinθ)(R - Rᵀ) when θ ≠ 0.

**Procrustes alignment**: Given point sets {xᵢ} and {yᵢ}, find R,t (and optionally scale s) minimizing Σ‖yᵢ - sRxᵢ - t‖². Solution: center both clouds → SVD of cross-covariance matrix Hxy = XᵀY → R = VUᵀ (with det correction for reflections) → t = ȳ - Rsx̄.

**Pivot calibration**: Given N poses Tᵢ each mapping tool tip to tracker, find tip position p_tool such that Tᵢ p_tool = p_pivot (constant). Assembles 3N×6 linear system, solves via SVD.

### Tambear status: NONE

None of these primitives exist. The math involved (rotation matrices, quaternions, SE(3)) is entirely absent from tambear.

### Accumulate+gather decomposition

- **Rodrigues**: pure arithmetic on a 3×3 kernel — Kingdom A (embarrassingly parallel per-point)
- **Quaternion Hamilton product**: same structure as complex multiply — Kingdom A
- **SLERP**: element-wise per parameter t — Kingdom A
- **Procrustes**: center (mean reduce) → cross-covariance (accumulate DotProduct) → SVD → det-flip check. The SVD is already in linear_algebra.rs. Procrustes = thin wrapper around existing primitives.
- **SE(3) composition**: 4×4 matrix multiply — already expressible via mat_mul if we add the 4×4 homogeneous struct
- **Pivot calibration**: assembles overdetermined linear system → SVD solve → already have pseudoinverse

### Priority primitives to implement

1. `quaternion_mul(q1, q2) -> Quaternion` — Hamilton product
2. `quaternion_to_rotation_matrix(q) -> Mat3x3` — standard formula
3. `rotation_matrix_to_quaternion(r: &Mat) -> Quaternion` — via trace+eigenvector
4. `euler_to_rotation_matrix(roll, pitch, yaw, convention) -> Mat3x3`
5. `rodrigues(axis: &[f64; 3], angle: f64) -> Mat` — skew → Rodrigues formula
6. `slerp(q1, q2, t) -> Quaternion`
7. `se3_compose(t1: &Mat, t2: &Mat) -> Mat` — 4×4 homogeneous multiply
8. `se3_inverse(t: &Mat) -> Mat` — Rᵀ block structure
9. `se3_log(t: &Mat) -> [f64; 6]` — matrix log in se(3) Lie algebra
10. `procrustes(source: &Mat, target: &Mat, allow_scale: bool) -> ProcrustesResult` — SVD-based
11. `pivot_calibration(poses: &[Mat]) -> [f64; 3]` — overdetermined SVD solve

---

## Part 2: Affine Transforms (12 DOF)

### What's needed

**4×4 affine matrix**: encodes rotation + translation + anisotropic scaling + shearing in homogeneous coordinates. 12 DOF (vs 6 for rigid).

**Polar decomposition**: A = R·S where R ∈ SO(n) is orthogonal and S is symmetric positive semi-definite (the "stretch" factor). Computed via SVD: A = UΣVᵀ → R = UVᵀ, S = VΣVᵀ.

**Affine registration cost functions** (for image-to-image alignment):
- **Sum of squared differences (SSD)**: Σᵢ(I₁(xᵢ) - I₂(T(xᵢ)))² — simple, assumes same intensity scale
- **Normalized cross-correlation (NCC)**: (I₁·I₂)/(‖I₁‖‖I₂‖) — intensity scale invariant
- **Mutual information (MI)**: H(I₁) + H(I₂) - H(I₁,I₂) — modality invariant; already have entropy primitives in information_theory.rs
- **Normalized MI (NMI)**: (H(I₁) + H(I₂)) / H(I₁,I₂)

### Tambear status

- **Polar decomposition**: NOT implemented. SVD exists (one-sided Jacobi in linear_algebra.rs) so R = UVᵀ, S = VΣVᵀ is straightforward to add — thin wrapper around existing svd().
- **Affine matrix struct**: NOT implemented. Could be a thin newtype over Mat with transform composition helpers.
- **SSD, NCC**: NOT implemented as named primitives (trivial arithmetic but should be named for discoverability).
- **MI for images**: The MI primitive exists in information_theory.rs — needs a wrapper that discretizes image intensities into joint histogram bins.

### Priority primitives to implement

1. `polar_decompose(a: &Mat) -> (Mat, Mat)` — (R, S) from SVD — 5-line wrapper around existing svd()
2. `affine_compose(a1: &Mat, a2: &Mat) -> Mat` — 4×4 multiply (same as se3_compose but general 12-DOF)
3. `affine_inverse(a: &Mat) -> Option<Mat>` — via LU (already have lu_solve)
4. `ssd(image1: &[f64], image2: &[f64]) -> f64` — sum of squared differences
5. `ncc(image1: &[f64], image2: &[f64]) -> f64` — normalized cross-correlation
6. `nmi_images(image1: &[f64], image2: &[f64], n_bins: usize) -> f64` — wrapper around joint entropy

---

## Part 3: Deformable / Non-Rigid Transforms

### What's needed

**B-spline free-form deformation (FFD)**: A grid of control points {pᵢⱼₖ} with B-spline basis functions. The displacement at any point x = Σᵢⱼₖ Bᵢ(u)Bⱼ(v)Bₖ(w)·pᵢⱼₖ where (u,v,w) are local grid coordinates. Widely used in medical image registration (Rueckert 1999).

**Thin-plate spline (TPS)**: Interpolates N landmark displacements with minimum bending energy. Solution: f(x) = aᵀx + Σᵢ wᵢ φ(‖x - xᵢ‖) where φ(r) = r² log r (in 2D) or r (in 3D). Setting up the linear system: K·w = y where Kᵢⱼ = φ(‖xᵢ - xⱼ‖).

**Diffeomorphic transforms**: Guarantee invertibility and topology preservation. Three variants:
- **Scaling and squaring**: Compute φ = exp(v) by recursively composing φ ← φ∘φ (N times). Approximates flow of stationary velocity field v. O(N·grid) where N ≈ 7-10.
- **Stationary velocity field (SVF)**: φ = exp(v) once computed via scaling-and-squaring; v parameterizes the transform.
- **LDDMM** (Large Deformation Diffeomorphic Metric Mapping): Time-varying velocity field v(t), integrate ODE ∂φ/∂t = v(t,φ(t)), minimize ‖v‖²_V + data term. Requires geodesic shooting or gradient descent on path energy.

**Jacobian determinant**: For deformation field u(x), the local volume change is det(I + ∂u/∂x). Jacobian < 0 = folding (non-diffeomorphic). Computed from finite-difference gradients of displacement field.

**Displacement field composition**: φ₁∘φ₂(x) = φ₁(φ₂(x)) — requires interpolating φ₁ at off-grid locations.

**Displacement field inversion**: No closed form; iterative: φ⁻¹ ← φ⁻¹ + (x - φ(φ⁻¹(x))) until convergence.

### Tambear status: NONE

None of these primitives exist. The mathematical building blocks are largely present (B-spline basis in interpolation.rs, linear system solve in linear_algebra.rs, finite differences are trivial) but no deformable transform primitives are implemented.

### Accumulate+gather decomposition

- **TPS setup**: Kᵢⱼ computation = accumulate(all pairs, φ(‖xᵢ - xⱼ‖), Store) — Kingdom A pairwise
- **B-spline FFD evaluation**: for each point, gather control points from neighborhood (fixed-radius gather) → apply B-spline weights — Kingdom A
- **Jacobian determinant**: finite-difference gradient at each point → 3×3 determinant — Kingdom A per-point
- **Displacement composition**: for each x, gather φ₂(x) → interpolate φ₁ at that location — Kingdom A with irregular gather
- **Displacement inversion**: iterative convergence — Kingdom C (fixed-point iteration)
- **LDDMM**: ODE integration — Kingdom B (sequential recurrence per time step) with spatial Kingdom A

### Priority primitives to implement

1. `bspline_basis_nd(knots_per_dim: &[Vec<f64>], x: &[f64], p: usize) -> f64` — N-D B-spline basis evaluation
2. `bspline_ffd_eval(control_grid: &[f64], grid_shape: &[usize], x: &[f64], p: usize) -> Vec<f64>` — FFD displacement at point
3. `tps_fit(landmarks_source: &Mat, landmarks_target: &Mat) -> TpsTransform` — thin-plate spline fit
4. `tps_eval(tps: &TpsTransform, x: &[f64]) -> Vec<f64>` — evaluate TPS at new point
5. `jacobian_determinant(disp_field: &[f64], shape: &[usize], spacing: &[f64]) -> Vec<f64>` — per-voxel Jacobian det
6. `displacement_compose(phi1: &[f64], phi2: &[f64], shape: &[usize]) -> Vec<f64>` — field composition
7. `scaling_and_squaring(v: &[f64], shape: &[usize], n_steps: usize) -> Vec<f64>` — exp(v) via repeated composition

---

## Part 4: Multiple Comparison Corrections

### Current state (hypothesis.rs)

| Method | Status |
|--------|--------|
| Bonferroni | HAVE — `bonferroni(p_values)` |
| Holm-Bonferroni (step-down) | HAVE — `holm(p_values)` |
| Benjamini-Hochberg (FDR) | HAVE — `benjamini_hochberg(p_values)` |
| Hochberg step-up | MISSING |
| Benjamini-Yekutieli (dependent tests) | MISSING |
| Storey q-value (π₀ estimation) | MISSING |
| Random field theory (RFT) | MISSING |
| Cluster-based permutation testing | MISSING |
| TFCE | MISSING |
| Network-based statistic (NBS) | MISSING |
| Local FDR | MISSING |

### Missing primitives — mathematical detail

**Hochberg step-up**: Like Holm but more powerful under independence. Sort p values ascending p₍₁₎ ≤ ... ≤ p₍ₘ₎. Starting from largest: if p₍ₖ₎ ≤ α/(m-k+1), reject all H₍₁₎,...,H₍ₖ₎. Implementation: scan from largest rank downward, stop at first rejection, reject everything below. Dominates Holm under independence.

**Benjamini-Yekutieli (BY)**: BH with correction factor cₘ = Σᵢ₌₁ᵐ 1/i (harmonic number). Adjusted p = p·m·cₘ/rank. Controls FDR under arbitrary dependence. Simple: multiply BH adjusted p-values by cₘ.

**Storey q-value**: Estimates π₀ (proportion of true null hypotheses) from the p-value distribution, then q = π₀·p·m/rank. The π₀ estimator: at a tuning parameter λ, π̂₀(λ) = #{p > λ} / (m(1-λ)). Bootstrap or spline over λ values to find the stable estimate. q-values are then FDR-adjusted with π̂₀ < 1, giving more power than BH.

**Random Field Theory (RFT)**: For spatially smooth test statistic images (e.g., t-maps, z-maps). The key quantity is the expected Euler characteristic E[χ(Aᵤ)] of the excursion set Aᵤ = {x: Z(x) > u}:

For a 3D Gaussian field: E[χ(Aᵤ)] = R₃·(4 ln 2)^(3/2)·(2π)⁻² · (u² - 1)·φ(u)  + R₂·...  + R₁·...  + R₀·Φ̄(u)

where Rᵢ are the Minkowski functionals (Lipschitz-Killing curvatures) of the search volume, and φ, Φ̄ are the standard normal PDF and survival function. The Rᵢ depend on the FWHM smoothness. For t-fields, chi²-fields, F-fields, the EC formulae differ but are all in the Worsley 1996 family.

**Resels**: Resolution elements = volume / (FWHM)^D. The smoothness FWHM is estimated from the gradient covariance of residuals.

**Cluster-based permutation testing**:
1. Compute observed test statistic map (e.g., t-values per voxel/sensor)
2. Apply primary threshold u (e.g., uncorrected p < 0.05) → binarize → find connected clusters → compute cluster-level statistic (e.g., sum of t-values)
3. Permute condition labels B times → repeat steps 1-2 → build null distribution of max-cluster statistic
4. Threshold observed cluster statistics at (1-α) percentile of null distribution

Requires: connected components in N-D (see Part 5), permutation loop.

**TFCE** (Threshold-Free Cluster Enhancement): For each voxel h and threshold t ∈ [0, max]: TFCE(h) = ∫ e(h,t)^E · t^H dt where e(h,t) is the extent of the cluster containing h at threshold t, E and H are parameters (default: E=0.5, H=2). Integrates cluster evidence over all thresholds simultaneously — no arbitrary cluster-forming threshold.

**Network-Based Statistic (NBS)**: Same idea as cluster-based but for graph edges. Apply edge-level threshold → find connected components of the thresholded graph → permute → build null distribution of max-component size.

**Local FDR**: Efron's local FDR = proportion of nulls at each observed z-value: lfdr(z) = π₀·f₀(z)/f(z) where f₀ is the null density and f is the empirical density (estimated via Poisson regression on z-histogram). Requires density estimation.

### Priority primitives to implement

1. `hochberg(p_values: &[f64]) -> Vec<f64>` — step-up, dominates Holm under independence
2. `benjamini_yekutieli(p_values: &[f64]) -> Vec<f64>` — BH * harmonic_number(m), arbitrary dependence
3. `storey_pi0(p_values: &[f64], lambda: Option<&[f64]>) -> f64` — proportion null estimator
4. `storey_qvalues(p_values: &[f64]) -> Vec<f64>` — FDR-adjusted with π₀ < 1
5. `rft_euler_characteristic_gaussian(u: f64, resels: &[f64]) -> f64` — expected EC for 3D Gaussian field
6. `rft_threshold(alpha: f64, resels: &[f64], field_type: RftFieldType, df: Option<f64>) -> f64` — corrected threshold
7. `cluster_permutation_test(stat_map: &[f64], shape: &[usize], labels: &[usize], n_perms: usize, threshold: f64) -> ClusterResult` — permutation cluster test
8. `tfce(stat_map: &[f64], shape: &[usize], e: f64, h: f64, connectivity: usize) -> Vec<f64>` — TFCE enhancement
9. `nbs(edge_stats: &[f64], adjacency: &[(usize,usize)], n_nodes: usize, threshold: f64, labels: &[usize], n_perms: usize) -> NbsResult` — network-based statistic
10. `local_fdr(z_scores: &[f64], n_bins: usize) -> Vec<f64>` — Efron local FDR

---

## Part 5: N-Dimensional Array Operations

### The key structural insight

**The 1D/2D special case is NOT the primitive.** The N-D version IS the primitive. Every 1D operation we have is a degenerate case of the N-D version at ndim=1. The architecture should be:

```
convolve_nd(data: &[f64], shape: &[usize], kernel: &[f64], kernel_shape: &[usize]) -> Vec<f64>
```

...with `convolve(a, b)` being a convenience alias for ndim=1. This way every downstream consumer (neuroimaging, image processing, climate data) gets the full generality for free.

### Current state

| Operation | 1D | 2D | 3D | N-D |
|-----------|----|----|-----|-----|
| FFT | HAVE (signal_processing.rs) | HAVE (fft2d, separable) | MISSING | MISSING |
| Convolution | HAVE (convolve) | MISSING | MISSING | MISSING |
| Gaussian smoothing | MISSING (closest: moving_average) | MISSING | MISSING | MISSING |
| Morphological (erosion/dilation) | MISSING | MISSING | MISSING | MISSING |
| Connected components | MISSING | MISSING | MISSING | MISSING |
| Distance transform | MISSING | MISSING | MISSING | MISSING |
| Interpolation / resampling | HAVE (1D spline, RBF) | MISSING | MISSING | MISSING |
| Gradient (finite diff) | MISSING | MISSING | MISSING | MISSING |
| Laplacian | MISSING | MISSING | MISSING | MISSING |
| Hessian matrix | MISSING | MISSING | MISSING | MISSING |
| Radon transform | MISSING | MISSING | N/A | MISSING |
| N-D FFT | HAVE 1D, HAVE 2D | — | MISSING | MISSING |

### Mathematical details for missing primitives

**N-D convolution**: (f*g)(x) = Σ_y f(y)·g(x-y) over the N-D index lattice. Separable kernels (e.g., Gaussian) decompose into 1D convolutions along each axis. For non-separable kernels: either direct O(n^D · k^D) or FFT-based O(n^D log n^D).

**Gaussian smoothing**: Gaussian kernel G(x; σ) = (2πσ²)^(-D/2) exp(-‖x‖²/(2σ²)) is separable: G(x₁,...,xD) = ∏ᵢ G(xᵢ; σᵢ). Smoothing = convolve_nd with separable Gaussian kernel along each axis independently. FWHM = 2√(2 ln 2) σ ≈ 2.355σ.

**Morphological operations**:
- Erosion: (f ⊖ B)(x) = min_{b∈B} f(x+b)
- Dilation: (f ⊕ B)(x) = max_{b∈B} f(x-b)
- Opening = erosion then dilation; closing = dilation then erosion
- For binary images: erosion = AND over structuring element; dilation = OR
- Accumulate pattern: min/max reduce over neighborhood (Kingdom A with local gather)

**Connected components (N-D)**: Union-find on the N-D grid. Connectivity defines the neighborhood: in 3D, 6-connectivity (face-sharing), 18-connectivity (face+edge), 26-connectivity (face+edge+corner). General N-D: 2D hypercube faces give 2D neighbors per axis, increasingly larger neighborhoods at higher connectivity. Algorithm: two-pass (label forward, resolve backward) or union-find with path compression.

**Distance transform**: For binary mask M, DT(x) = min_{y: M(y)=0} ‖x - y‖. Exact Euclidean distance transform in O(n^D) via the separable algorithm (Meijster/Felzenszwalb): decompose into 1D parabola lower envelope passes along each dimension.

**N-D interpolation / resampling**: Given data on grid G₁, resample onto grid G₂. For each output point x ∈ G₂: find enclosing grid cell in G₁, interpolate (linear: N-linear, i.e., trilinear in 3D; cubic: N-cubic; B-spline: recursive prefilter then N-D B-spline evaluation). The N-linear interpolation is the tensor product of 1D linear interpolations.

**Gradient (finite difference)**: ∂f/∂xᵢ at each grid point via centered differences: (f(x + hᵢ) - f(x - hᵢ)) / (2hᵢ). Result is D arrays of same shape as input. Accumulate pattern: for each dimension, gather two neighbors, compute difference.

**Laplacian**: Σᵢ ∂²f/∂xᵢ² = Σᵢ (f(x+hᵢ) - 2f(x) + f(x-hᵢ)) / hᵢ². Separable: sum of second differences along each axis. Scalar field → scalar field.

**Hessian matrix**: H(f)(x) = [∂²f/∂xᵢ∂xⱼ] at each grid point. Returns D×D matrix at each voxel. Cross terms: (∂²f/∂xᵢ∂xⱼ)(x) ≈ (f(x+eᵢ+eⱼ) - f(x+eᵢ-eⱼ) - f(x-eᵢ+eⱼ) + f(x-eᵢ-eⱼ)) / (4hᵢhⱼ). Result shape: [n₁,...,nD, D, D].

**N-D FFT**: Generalization of fft2d — apply 1D FFT along each axis sequentially. fft_nd(data, shape) = apply fft(data[:,..., :, :]) along axis D, then axis D-1, ..., axis 0. The existing fft2d already shows this pattern; generalizing to N axes is mechanical.

**Radon transform (2D)**: R(θ, s) = ∫ f(x,y) δ(x cosθ + y sinθ - s) dx dy — integrates f along lines at angle θ and offset s. Discretized: for each (θ,s) pair, sum f along the corresponding line through the image. Inverse Radon (filtered back-projection) = 1D FFT per projection → ramp filter → back-project → sum.

### Priority primitives to implement

**First tier — enables cluster-perm and RFT above:**

1. `convolve_nd(data: &[f64], shape: &[usize], kernel: &[f64], k_shape: &[usize]) -> Vec<f64>` — N-D direct convolution
2. `convolve_nd_separable(data: &[f64], shape: &[usize], kernels_1d: &[Vec<f64>]) -> Vec<f64>` — separable kernel (product of 1D)
3. `gaussian_smooth_nd(data: &[f64], shape: &[usize], sigma: &[f64]) -> Vec<f64>` — separable Gaussian smoothing
4. `connected_components_nd(mask: &[bool], shape: &[usize], connectivity: usize) -> (Vec<i32>, usize)` — union-find, returns labels + count
5. `gradient_nd(data: &[f64], shape: &[usize], spacing: &[f64]) -> Vec<Vec<f64>>` — centered finite differences per axis
6. `fft_nd(data: &[Complex], shape: &[usize]) -> Vec<Complex>` — general N-D FFT

**Second tier — fuller N-D algebra:**

7. `laplacian_nd(data: &[f64], shape: &[usize], spacing: &[f64]) -> Vec<f64>` — sum of second differences
8. `hessian_nd(data: &[f64], shape: &[usize], spacing: &[f64]) -> Vec<f64>` — D×D Hessian at each voxel (flat D²·n output)
9. `distance_transform_nd(mask: &[bool], shape: &[usize], spacing: &[f64]) -> Vec<f64>` — Euclidean DT via parabola envelope
10. `interpolate_nd(data: &[f64], source_shape: &[usize], target_coords: &[f64]) -> Vec<f64>` — N-linear / N-cubic
11. `resample_nd(data: &[f64], source_shape: &[usize], target_shape: &[usize], order: usize) -> Vec<f64>` — regrid
12. `morphology_erode_nd(data: &[bool], shape: &[usize], struct_elem: &[bool], se_shape: &[usize]) -> Vec<bool>`
13. `morphology_dilate_nd(data: &[bool], shape: &[usize], struct_elem: &[bool], se_shape: &[usize]) -> Vec<bool>`
14. `radon_transform(image: &[f64], rows: usize, cols: usize, angles: &[f64]) -> Vec<f64>` — 2D sinogram
15. `radon_backproject(sinogram: &[f64], n_angles: usize, n_offsets: usize, output_size: usize) -> Vec<f64>` — filtered back-projection

---

## Part 6: Masking and Confound Handling

### What's needed

These are all "apply same 1D operation to each element independently" patterns — the core primitive is the 1D operation; the multi-element version is a parallel map.

| Operation | Math | Status |
|-----------|------|--------|
| Confound regression | Per-voxel OLS, keep residuals: r = y - X(XᵀX)⁻¹Xᵀy | Partially — have OLS, but no "apply to each column of data matrix" wrapper |
| Scrubbing/censoring | Boolean mask on time axis, remove marked time points | MISSING — trivial but needs named primitive |
| Bandpass per element | Apply FIR/IIR filter to each signal independently | Have filter design (fir_lowpass etc) + fir_filter — MISSING parallel apply |
| Global signal regression | Compute mean across all elements per time point, regress out | MISSING |
| CompCor | PCA on noise ROI signals → top K components → regress out | Have PCA (dim_reduction.rs), have OLS — MISSING the pipeline |
| Adaptive smoothing | Variable-kernel Gaussian based on local SNR | MISSING |
| Despiking | Detect outlier time points per signal via robust stats | Have robust stats — MISSING the per-element detection loop wrapper |

### Key insight

Most of these reduce to: `apply_per_element(data_matrix: &[f64], shape: &[usize], time_axis: usize, fn: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<f64>` — a parallelizable map over spatial positions. The actual math (filter, regression, PCA) is already implemented. What's missing is the infrastructure to apply it element-wise over a large N-D array.

### Priority primitives

1. `confound_regress(data: &[f64], n_timepoints: usize, confounds: &Mat) -> Vec<f64>` — per-signal OLS residuals in parallel
2. `scrub_timepoints(data: &[f64], n_timepoints: usize, keep_mask: &[bool]) -> Vec<f64>` — censor time points
3. `bandpass_apply_nd(data: &[f64], shape: &[usize], time_axis: usize, coeffs: &[f64]) -> Vec<f64>` — apply same FIR filter per element
4. `global_signal_regression(data: &[f64], n_timepoints: usize) -> Vec<f64>` — compute GSR, regress out
5. `compcor(data_noise_roi: &[f64], n_timepoints: usize, k: usize) -> Mat` — PCA on noise region → components

---

## Summary Table

| Category | Have | Missing (Priority) |
|----------|------|-------------------|
| Rigid transforms | — | quaternion, rotation matrix, SE(3), SLERP, Procrustes |
| Affine transforms | SVD (buildable polar decomp) | polar_decompose, affine ops, SSD/NCC |
| Deformable transforms | B-spline 1D | TPS, FFD, diffeomorphic (scaling-and-squaring), Jacobian det |
| Multiple comparisons | Bonferroni, Holm, BH | Hochberg, BY, Storey q-value, RFT, cluster-perm, TFCE, NBS, local FDR |
| N-D array ops | FFT 1D+2D, convolve 1D | convolve_nd, gaussian_smooth_nd, connected_components_nd, gradient_nd, fft_nd, distance_transform_nd, interpolate_nd, morphology |
| Confound handling | OLS, PCA, FIR filter | confound_regress wrapper, scrubbing, bandpass_apply_nd, global signal regression, CompCor |

**Most impactful quick wins** (primitives where the math is trivial, leveraging existing pieces):
1. `hochberg` + `benjamini_yekutieli` + `storey_qvalues` — 3 functions, ~40 lines each, complete the multiple-comparisons catalog
2. `polar_decompose` — 5-line SVD wrapper, unlocks affine decomposition
3. `gradient_nd` — finite differences, enables all downstream gradient-based ops
4. `connected_components_nd` — union-find, required for cluster-based permutation tests
5. `fft_nd` — mechanical generalization of existing fft2d pattern
6. `gaussian_smooth_nd` — separable product of existing 1D primitives
7. `procrustes` — SVD + centering, uses existing linear algebra

**Structurally harder** (require genuinely new math or iterative algorithms):
- RFT Euler characteristic EC formulas (needs Worsley 1996 implementation)
- TFCE (requires sorted threshold sweep + cluster extent tracking)
- LDDMM (ODE integration + variational optimization)
- Storey π₀ (needs bootstrap/spline over λ)
- Quaternion/SE(3) apparatus (new struct types, not just new functions)
