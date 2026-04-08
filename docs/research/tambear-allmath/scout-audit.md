# Tambear All-Math Scout Audit

**Date**: 2026-04-06  
**Scout**: Claude (scout role, tambear-allmath expedition)  
**Campsite**: tambear-allmath/audit  

---

## Executive Summary

tambear is a substantial mathematical library. It is **not close to "all math in all fields."** It has excellent, deep coverage of *applied statistics, signal processing, ML/AI, and project-specific research math* (Collatz, equipartition, series acceleration). But entire major fields are absent or thin: number theory, combinatorics, classical geometry, PDEs, orthogonal polynomials, stochastic processes, control theory, coding theory, and special function breadth.

**Total test count**: ~2,918 `#[test]` functions across all files.  
**Test quality**: Predominantly **math truth tests**, not snapshots. The gold standard suite compares directly against scipy/numpy/sklearn pre-computed values. Adversarial tests assert mathematical properties (e.g., "equilateral triangle produces exactly one H1 cycle with persistence 0"). Inline tests verify known analytical results (e.g., `rk4(dy/dt = y) → y(1) ≈ e`).

---

## What's Implemented

### Group 1: Numerical Foundations

| Module | Field | Key Functions | Inline Tests |
|--------|-------|--------------|-------------|
| `special_functions` | Special functions (statistical) | erf/erfc, log_gamma, digamma, trigamma, regularized beta/gamma, normal/t/F/chi2 CDFs | 24 |
| `numerical` | Numerical analysis | bisection, Newton, secant, Brent root-finding; central/Richardson derivatives; simpson/gauss-legendre/adaptive_simpson/trapezoid quadrature; euler/RK4/RK45/RK4-system ODEs; fixed_point | 21 |
| `linear_algebra` | Linear algebra | mat ops (mul/add/sub/scale/vec), dot, norm, outer; LU, Cholesky, QR, SVD, pinv, sym_eigen, power_iteration, cond, rank, solve, lstsq | 25+ |
| `optimization` | Optimization | backtracking line search, golden section, gradient descent, Adam, AdaGrad, RMSProp, L-BFGS, Nelder-Mead, coordinate descent, projected gradient | 15 |
| `interpolation` | Interpolation | Lagrange, Newton divided diff, Neville, lerp, nearest; natural/clamped/monotone-Hermite/Akima/PCHIP cubic spline; Chebyshev; polyfit; RBF; barycentric rational; B-spline; GP regression; Padé | 30+ |
| `series_accel` | Series acceleration | partial_sums, Cesàro, Aitken Δ², Wynn epsilon (+ streaming), Richardson, Euler transform, Abel sum, Euler-Maclaurin zeta, convergence detection | 48 |
| `bigint` | Arbitrary precision integers | U256 (stack), BigInt (FFT-multiply, full arithmetic) | 20+ |
| `bigfloat` | Arbitrary precision floats | BigFloat, BigComplex, Riemann ζ (Borwein), Hardy Z, find_zeta_zero, Euler factors | 15+ |
| `number_theory` | Number theory | sieve, segmented_sieve, is_prime (Miller-Rabin det.), next_prime, prime_count; gcd, lcm, extended_gcd, mod_inverse, mod_pow, mul_mod; crt; euler_totient, mobius, factorize, factorize_complete, pollard_rho; legendre/jacobi, sqrt_mod (Tonelli-Shanks), primitive_root, discrete_log (BSGS); continued_fraction, convergents, best_rational; partition_count; sum_of_two_squares, pell_fundamental; rsa_keygen/encrypt/decrypt; dh_public_key/shared_secret | 60+ |

### Group 2: Probability & Statistics

| Module | Field | Key Functions | Inline Tests |
|--------|-------|--------------|-------------|
| `descriptive` | Descriptive statistics | moments (mean/variance/skew/kurtosis), quantile/median/quartiles/IQR, geometric/harmonic/trimmed/winsorized mean, MAD, Gini, Bowley/Pearson skewness, bowley_skewness | — |
| `rng` | Random number generation | SplitMix64, Xoshiro256, LCG; Box-Muller normal; exponential, gamma, beta, chi2, t, F, Cauchy, lognormal; Bernoulli, Poisson, binomial, geometric; shuffle, sample_without_replacement, sample_weighted | 20 |
| `hypothesis` | Statistical hypothesis tests | one/two-sample/Welch/paired t; one-way ANOVA; chi2 goodness-of-fit/independence; proportion z-tests; effect sizes (Cohen's d, Glass's Δ, Hedges' g, point-biserial r); odds ratio; Bonferroni/Holm/BH correction | 20+ |
| `nonparametric` | Nonparametric statistics | rank, Spearman, Kendall τ; Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis; KS tests (1-sample, 2-sample); bootstrap; permutation test; KDE (+ FFT-KDE, Silverman BW); runs test; sign test; level_spacing_r_stat | 34 |
| `robust` | Robust statistics | Huber/bisquare/Hampel M-estimates; Qn/Sn/τ scale; LTS (simple); MCD 2D; medcouple | 22 |
| `information_theory` | Information theory | Shannon/Rényi/Tsallis entropy; KL/JS divergence; cross entropy; mutual information, NMI, adjusted MI, variation of information, conditional entropy; histogram-based MI | 15+ |
| `bayesian` | Bayesian methods | Metropolis-Hastings MCMC; Bayesian linear regression; ESS; R-hat | 15+ |
| `multivariate` | Multivariate statistics | Hotelling T² (one/two-sample); MANOVA; LDA; CCA; Mardia normality test | 10+ |
| `copa` | Covariance accumulator | COPA (centered outer product, streaming, parallel-mergeable); copa_pca | 5+ |

### Group 3: Time Series & Finance

| Module | Field | Key Functions | Inline Tests |
|--------|-------|--------------|-------------|
| `time_series` | Time series | AR model fit/predict; differencing; simple exponential smoothing; Holt linear; ADF unit root test; ACF/PACF | 8 |
| `stochastic` | Stochastic processes | Brownian motion; Brownian bridge; GBM; Black-Scholes (price+delta, 🔷 partial Greeks); OU process; Poisson (homogeneous+non); Markov chains (DTMC+CTMC); birth-death; M/M/1, Erlang-C; random walk; Itô/Stratonovich integrals | 40+ |
| `volatility` | Financial volatility | GARCH(1,1) fit/forecast; EWMA variance; realized variance; bipower variation; BNS jump test; Roll spread; Kyle lambda; Amihud illiquidity; annualize_vol | 9 |
| `panel` | Panel econometrics | Fixed effects, random effects, first differences, two-way FE; Hausman test; Breusch-Pagan RE test; 2SLS; DiD | 15 |
| `mixed_effects` | Mixed effects models | LME random intercept; ICC; design effect | 5+ |
| `survival` | Survival analysis | Kaplan-Meier estimator (+ median); log-rank test; Cox PH model | 6 |
| `irt` | Item Response Theory | Rasch/2PL/3PL probability; 2PL EM fitting; MLE/EAP ability estimation; item/test information; SEM; Mantel-Haenszel DIF | 10+ |

### Group 4: Signal Processing & Spectral

| Module | Field | Key Functions | Inline Tests |
|--------|-------|--------------|-------------|
| `signal_processing` | Signal processing | FFT/IFFT/RFFT/FFT2D; windows (Hann/Hamming/Blackman/Bartlett/Kaiser/flat-top); periodogram, Welch, STFT, spectrogram; convolve, cross-correlate, autocorrelation; DCT-II/III; FIR lowpass/highpass/bandpass; Biquad/Butterworth IIR; moving average, EMA, Savitzky-Golay; Hilbert/envelope/instantaneous freq; cepstrum; Haar DWT/IDWT + wavedec/waverec; Db4 DWT/IDWT; Goertzel; zero-crossing rate; median filter | 33 |
| `spectral` | Advanced spectral | Lomb-Scargle; cross-spectral density; spectral entropy; band power; spectral peaks; multitaper PSD (DPSS) | 10 |
| `spectral_gap` | Spectral gap research | Arnoldi eigenvalues; spectral gap for arithmetic/modular/lazy-walk/persistence maps | 26 |

### Group 5: Machine Learning & AI

| Module | Field | Key Functions | Inline Tests |
|--------|-------|--------------|-------------|
| `clustering` | Clustering | DBSCAN, HDBSCAN-like, agglomerative | 10+ |
| `kmeans` | K-means | k-means++ initialization, Lloyd's algorithm | 10+ |
| `knn` | K-nearest neighbors | knn_from_distance, knn_session | 5+ |
| `neural` | Neural network primitives | ReLU/LeakyReLU/ELU/SELU/GELU/Swish/Mish/Sigmoid/Tanh activations (+ backward); Softmax/log-softmax; conv1d/2d + transpose; max/avg pooling; BatchNorm/LayerNorm/RMSNorm/GroupNorm/InstanceNorm; dropout; linear/bilinear; embedding; positional encoding; RoPE; scaled dot-product attention; multi-head attention; MSE/BCE/cross-entropy/Huber/cosine/hinge/focal losses; clip_grad_norm; label_smooth; temperature_scale; top-k/top-p | 56 |
| `dim_reduction` | Dimensionality reduction | PCA, classical MDS, t-SNE, NMF | 10+ |
| `factor_analysis` | Factor analysis | PAF, varimax rotation, Cronbach's α, McDonald's ω, scree elbow, Kaiser criterion | 10+ |
| `manifold` | Manifold learning | Manifold, ManifoldMixture, ManifoldDistanceOp; Euclidean/Hyperbolic/Spherical/Mixture manifolds | 15+ |
| `mixture` | Mixture models | GMM (EM algorithm), BIC/AIC model selection | 10+ |
| `superposition` | Superposition/collapse | Superposition, SuperpositionView | 11 |
| `train` | Model training | Linear regression fit; logistic regression fit; Cholesky solve | 20+ |

### Group 6: Geometry, Graphs, Topology

| Module | Field | Key Functions | Inline Tests |
|--------|-------|--------------|-------------|
| `graph` | Graph algorithms | BFS, DFS, topological sort, connected components; Dijkstra, Bellman-Ford, Floyd-Warshall; Kruskal/Prim MST; degree/closeness centrality, PageRank; label propagation; modularity; max flow; diameter, density, clustering coefficient | 20+ |
| `spatial` | Spatial statistics | Euclidean/Haversine distance; empirical variogram; spherical/exponential/gaussian variogram models; ordinary Kriging; Moran's I, Geary's C; Ripley's K/L; Clark-Evans R | 16 |
| `tda` | Topological data analysis | Vietoris-Rips H0, H1; bottleneck distance; Wasserstein distance; persistence statistics/entropy; Betti curve | 10 |

### Group 7: Multivariate & Causal

| Module | Field | Key Functions | Inline Tests |
|--------|-------|--------------|-------------|
| `multivariate` | Multivariate statistics | Hotelling T² (1/2 sample); MANOVA; LDA; CCA; Mardia normality test | 11 |
| `causal` | Causal inference | Propensity scores, PSM, IPW; DiD; RDD (sharp); E-value; doubly robust ATE | 8+ |

### Group 8: Complexity & Dynamical Systems

| Module | Field | Key Functions | Inline Tests |
|--------|-------|--------------|-------------|
| `complexity` | Dynamical complexity | Sample entropy, ApEn, permutation entropy; Hurst RS, DFA, Higuchi FD; Lempel-Ziv; correlation dimension; largest Lyapunov exponent; full Lyapunov spectrum | 20+ |
| `fold_irreversibility` | Collatz fold math | fold_irreversibility_theorem; Nyquist margin; generalized symmetric step; temporal coverage; family analysis | 30+ |
| `layer_bijection` | Layer bijection proofs | compute_layers, is_layer_bijective, verify_all_layers, inverse_of_3_mod_2j, prove_layer_injectivity, collatz_permutation, is_transitive | 20+ |
| `extremal_orbit` | Extremal orbit analysis | extremal, padic_fixed_point_mod, build_affine_table, verify_extremal_dominance, verify_mihailescu_for_extremals | 15+ |
| `multi_adic` | Multi-adic analysis | p-adic valuation/distance/norm/digits; multi-adic trajectory; batch profiles; distance matrices | 21 |
| `collatz_parallel` | Collatz parallel computation | verify_drops, verify_profiled, batch_verify (BigInt variants) | 10+ |

### Group 9: Project-Specific Math

| Module | Field | Key Functions | Inline Tests |
|--------|-------|--------------|-------------|
| `copa` | COPA framework | copa_from_data, copa_pca | 10+ |
| `equipartition` | Thermodynamic math | free_energy, euler_factor, fugacity; fold_target, solve_fold; nucleation hierarchy; verify_fold_surface; phase_sweep | 15+ |
| `proof` | Formal proof framework | Property declarations (associativity, commutativity, homomorphism, scatter correctness); proof contexts; Collatz four pillars | 43 |
| `accumulate` | Core primitive | AccumulateEngine with 8 grouping patterns × multiple expr/op combinations; softmax | 20+ |

---

## Test Quality Assessment

### Tier 1: True Gold Standard (scipy/numpy/sklearn/faer parity)
**File**: `tests/gold_standard_parity.rs` — **489 tests** (502KB file; initial audit read only ~100 lines and underreported scope)  
These tests compare tambear output against pre-computed values from scipy/sklearn/numpy/faer with abs+rel tolerance (not snapshot). Correct coverage:

| Family | Tests cover |
|--------|-------------|
| Accumulate primitives | softmax, dot product, L2 distance |
| Linear models | linear regression, logistic regression |
| Clustering | DBSCAN |
| Descriptive stats (f06) | 20+ tests vs numpy oracle |
| Hypothesis testing (f07) | one-sample/two-sample/Welch t-tests, ANOVA, chi², effect sizes, corrections |
| Nonparametric/rank (f08) | Spearman, Kendall, Mann-Whitney, Wilcoxon, KS, sign test |
| Information theory (f25) | Shannon, Rényi, Tsallis, KL, JS, cross-entropy |
| Neural primitives (f23) | activations, loss functions, attention, batch/layer norm, conv1d, pooling |
| Manifold geometry (f28) | Poincaré, spherical geodesic, mixture |
| KNN (f35), Graph (f29) | brute-force KNN, BFS/Dijkstra/PageRank |
| Optimization (f05) | gradient descent, Adam |
| SVD vs faer | diagonal, ill-conditioned, rank-deficient, rectangular |

**Correction**: Earlier recommendation to "add oracle tests for hypothesis/SVD/KDE" was wrong — those already have oracle coverage. **Genuine remaining gaps**: survival (KM/Cox PH), ARMA/GARCH, KDE, physics module.

| Family | Module | Coverage |
|--------|--------|----------|
| f06 Descriptive stats | `descriptive.rs` | mean, variance, std, skewness, kurtosis, median, quartiles, gini, geo/harmonic mean, MAD, trimmed mean |
| f07 Hypothesis tests | `hypothesis.rs` | one-sample t, two-sample t, Welch, paired t, ANOVA, chi², proportions z, effect sizes, corrections |
| f08 Nonparametric | `nonparametric.rs` | Spearman, Kendall, Mann-Whitney, Wilcoxon, KS, sign test |
| f23 Neural | `neural.rs` | activations, losses, attention, batch/layer norm, conv1d, pooling |
| f25 Info theory | `information_theory.rs` | Shannon, Rényi, Tsallis, KL, JS, cross-entropy |
| f28 Manifold | `multivariate.rs` | Poincaré, spherical geodesic, mixture |
| f29 Graph | `graph.rs` | Dijkstra, Bellman-Ford, Floyd-Warshall, MST, PageRank, connected components |
| f35 KNN | `knn.rs` | brute-force KNN |
| f05 Optimization | `optimization.rs` | Nelder-Mead, L-BFGS-B, golden section, box-constrained |
| SVD | `linear_algebra.rs` | vs faer oracle: diagonal, ill-conditioned, rank-deficient, rectangular |
| Accumulate | `accumulate.rs` | softmax, dot product, L2 distance, linear/logistic regression, DBSCAN |

**Genuine gaps** (no gold standard coverage): survival (KM/Cox PH/log-rank), ARMA/GARCH, KDE, physics (42 in-module only), Bayesian MCMC (limited).

### Tier 2: Mathematical Truth Tests
**Files**: `tests/adversarial_boundary*.rs` (10 files, ~550 tests total), `tests/adversarial_disputed.rs` (113 tests), `tests/svd_adversarial.rs` (29 tests)  
These assert mathematical properties from first principles:
- "equilateral triangle: exactly one H1 cycle with persistence 0"
- "RK4 should beat Euler at same step count"
- "t-SNE gradient is Gauss-Seidel not Jacobi (proved by asymmetry)"
- Boundary conditions: NaN inputs, degenerate cases, empty inputs

**Inline tests** in numerical modules also assert known analytical results:
- `rk4(dy/dt = y, y(0)=1) → y(1) ≈ e` (within 1e-8)
- `rk4_system(harmonic oscillator) → y₁(π) ≈ cos(π) = -1`
- `fixed_point(Babylonian method) → √2` (within 1e-12)

### Tier 3: Scale Ladder Tests
**Files**: `tests/scale_ladder*.rs` (5 files, ~17 tests)  
Verify correct behavior across magnitudes. Thin but present.

### Concern: Snapshot Risk Modules
A few modules have **very low test counts relative to function count**:
- `volatility.rs`: 11 functions, 9 tests — some GARCH tests may be snapshot-like
- `time_series.rs`: 8 functions, 8 tests — may be minimal
- `bayesian.rs`: MCMC tests are inherently probabilistic (hard to gold-standard)
- `causal.rs`: doubly-robust ATE — complex, few adversarial tests

**Flag**: The adversarial suite doesn't cover `volatility`, `causal`, `panel`, or `irt` in any depth. These modules deserve adversarial tests with known analytical solutions.

---

## The Biggest Gaps

These are major mathematical fields with **zero or near-zero coverage** in tambear.

### Gap 1: Elementary Number Theory — ZERO COVERAGE

The entire field of classical number theory is absent:

| Missing | Description |
|---------|-------------|
| Sieve of Eratosthenes / Atkin | Prime generation |
| Miller-Rabin primality test | Probabilistic primality |
| Pollard's rho / p-1 | Integer factorization |
| GCD / LCM (integer) | Extended Euclidean algorithm |
| Modular arithmetic | modpow, modular inverse, CRT |
| Euler's totient φ(n) | Multiplicative function |
| Möbius function μ(n) | Sieve + Möbius inversion |
| Legendre/Jacobi symbol | Quadratic reciprocity |
| Primitive roots | Discrete logarithm |
| Quadratic residues | Tonelli-Shanks square root mod p |

**Exception**: The multi_adic / bigint / collatz_* modules contain p-adic valuation and arbitrary precision arithmetic, but these are research-specific, not the general number theory toolkit.

### Gap 2: Special Functions Breadth — THIN

Current: erf, gamma, beta, digamma, trigamma, CDFs.  
**Missing**:

| Missing | Description |
|---------|-------------|
| Bessel functions | J₀, J₁, Jₙ, Y₀, Y₁, I₀, I₁, K₀, K₁ — fundamental in physics/engineering |
| Airy functions | Ai(x), Bi(x) |
| Elliptic integrals | K(k), E(k), Π(n,k) — Legendre complete and incomplete |
| Elliptic functions | Jacobi sn, cn, dn |
| Hypergeometric functions | ₂F₁, ₁F₁, ₀F₁ (confluent, Kummer) |
| Riemann zeta (f64 fast) | ζ(s) for real s — BigFloat version exists, no fast f64 |
| Hurwitz zeta | ζ(s,q) |
| Lambert W function | W₀, W₋₁ branches |
| Fresnel integrals | S(x), C(x) |
| Exponential integrals | Ei(x), E₁(x), En(x) |
| Sine/cosine integrals | Si(x), Ci(x) |
| Struve functions | Hᵥ, Lᵥ |
| Parabolic cylinder functions | D_ν |
| Polygamma beyond trigamma | ψ^(n) for n ≥ 2 |
| Clausen function | Cl₂(x) |
| Lerch transcendent | Φ(z,s,a) |

### Gap 3: Combinatorics — ZERO COVERAGE

Nothing here. This is a fundamental field:

| Missing | Description |
|---------|-------------|
| Factorial, double factorial | n!, n!! |
| Binomial coefficients | C(n,k), log-binomial for overflow |
| Multinomial coefficients | |
| Stirling numbers 1st/2nd kind | Permutation / partition structures |
| Bell numbers | Partition of set |
| Catalan numbers | Many combinatorial interpretations |
| Integer partitions | p(n) generating function |
| Set partitions (gen.) | |
| Derangements | Number of permutations with no fixed point |
| Euler numbers/Bernoulli numbers | Generating functions, zeta values |
| Partition function (integer) | Hardy-Ramanujan formula |
| Generating functions (formal) | Polynomial power series over combinatorial objects |

### Gap 4: Stochastic Processes — THIN

Current: MCMC (Metropolis-Hastings), GARCH, EWMA.  
**Missing**:

| Missing | Description |
|---------|-------------|
| Geometric Brownian Motion | dS = μS dt + σS dW — market fundamental |
| Ornstein-Uhlenbeck process | Mean-reverting SDE |
| Brownian motion simulation | Discrete-time Wiener process |
| Jump diffusion (Merton/Kou) | GBM + Poisson jumps |
| Heston stochastic vol | SDE with correlated vol process |
| Variance Gamma process | |
| Markov chains | Transition matrix, steady state, absorption times |
| Markov decision processes | Value iteration, policy iteration |
| HMC / NUTS | Hamiltonian Monte Carlo for Bayesian inference |
| Gibbs sampling | |
| Particle filter | Sequential Monte Carlo |

### Gap 5: Financial Math — INCOMPLETE

Current: GARCH, realized vol, market microstructure.  
**Missing**:

| Missing | Description |
|---------|-------------|
| Black-Scholes pricing | Call/put pricing, closed form |
| Option Greeks | Δ, Γ, Θ, V, ρ |
| Binomial option pricing | CRR tree |
| Monte Carlo option pricing | Path-dependent options |
| Value at Risk (VaR) | Historical, parametric, Monte Carlo |
| Expected Shortfall (CVaR) | |
| Portfolio optimization | Markowitz mean-variance frontier |
| Nelson-Siegel yield curve | Svensson extension |
| Duration, convexity | Fixed income math |
| Interest rate models | Vasicek, CIR, Hull-White |
| Credit risk | Merton structural model, CDS pricing |

### Gap 6: PDEs — ZERO COVERAGE

Not a single PDE solver exists:

| Missing | Description |
|---------|-------------|
| Finite difference — 1D | Heat equation, wave equation |
| Finite difference — 2D | Laplace/Poisson equation |
| Finite element — 1D | Galerkin method |
| Method of characteristics | First-order PDE |
| Crank-Nicolson | Implicit ODE/PDE scheme |
| Implicit Euler | Stiff systems |
| Boundary value problems | Shooting method, BVP solver |

**Note**: ODE solvers exist (euler, RK4, RK45, RK4_system) but no PDE machinery.

### Gap 7: Orthogonal Polynomials — MISSING

Current: Chebyshev nodes/coefficients/evaluation.  
**Missing**:

| Missing | Description |
|---------|-------------|
| Legendre polynomials | Pₙ(x), recursion, roots/weights |
| Hermite polynomials | Hₙ(x) (physicist/probabilist) |
| Laguerre polynomials | Lₙ(x), associated Lᵅₙ(x) |
| Jacobi polynomials | Pₙ^(α,β)(x) — generalizes Legendre/Chebyshev |
| Gegenbauer polynomials | Ultraspherical |
| Associated Legendre | Pₙᵐ(x) — spherical harmonics basis |
| Spherical harmonics | Yₗᵐ(θ,φ) |

### Gap 8: Control Theory — ZERO COVERAGE

| Missing | Description |
|---------|-------------|
| Kalman filter | Optimal linear estimator |
| Extended Kalman filter | Nonlinear extension |
| Unscented Kalman filter | Sigma-point extension |
| PID controller | Proportional-integral-derivative |
| LQR controller | Linear quadratic regulator |
| State space representation | A, B, C, D matrices |
| Observability/controllability | Rank of Gramians |
| Lyapunov stability | P matrix test (not same as Lyapunov exponent) |
| Transfer functions | Z-transform, poles/zeros |
| Digital filter design | Bilinear transform, analog-to-digital |

### Gap 9: ML Algorithms — Several Missing

Current: neural layers, clustering, KNN, KMeans, GMM, PCA, t-SNE, NMF, manifold, superposition.  
**Missing**:

| Missing | Description |
|---------|-------------|
| Decision tree | CART, information gain splitting |
| Random forest | Ensemble of decision trees |
| Gradient boosting | GBM, XGBoost-style |
| Support vector machine (SVM) | SMO algorithm, kernel trick |
| Naive Bayes | Gaussian, multinomial, Bernoulli |
| Hidden Markov Model (HMM) | Viterbi, forward-backward, Baum-Welch |
| EM algorithm (general) | E-step / M-step framework |
| ISOMAP | Geodesic distance + MDS |
| UMAP | Uniform manifold approximation |
| Spectral clustering | Graph Laplacian-based |
| OPTICS | Density-based ordering |
| Autoencoder | Encoder-decoder neural network |
| Sequence models | RNN, LSTM, GRU cell math |
| CTC loss | Connectionist temporal classification |
| Attention variants | FlashAttention, sparse attention |

### Gap 10: Wavelets Beyond Haar/Db4 — THIN

Current: Haar, Db4.  
**Missing**:

| Missing | Description |
|---------|-------------|
| Daubechies family | db6, db8, db20, etc. |
| Symlets | Symmlets sym4, sym8 |
| Coiflets | Coiflet1-5 |
| Biorthogonal wavelets | bior1.1, bior2.2, etc. |
| Continuous wavelet transform | Morlet, Mexican hat, Paul |
| Wavelet packets | Full binary tree of WP coefficients |
| 2D wavelet transform | Image decomposition |
| Empirical mode decomposition | Hilbert-Huang transform |
| Synchrosqueezing | Reassignment-based time-frequency |

### Gap 11: Regression / GLM — THIN

Current: linear regression, logistic regression (train module), basic hypothesis tests.  
**Missing**:

| Missing | Description |
|---------|-------------|
| Ridge regression (L2) | Tikhonov regularization |
| LASSO (L1) | Coordinate descent on L1 |
| Elastic Net | L1 + L2 combined |
| Polynomial regression | Degree > 1 design matrix |
| Poisson regression | GLM with log link |
| Negative binomial regression | Overdispersed count data |
| Gamma regression | Positive continuous data |
| Quantile regression | Pinball loss minimization |
| LOESS / LOWESS | Locally weighted scatterplot smoothing |
| GAM (Generalized Additive Models) | Spline-based nonparametric regression |

### Gap 12: Transform Variants — MISSING

Current: FFT, DCT-II/III, Haar/Db4 DWT.  
**Missing**:

| Missing | Description |
|---------|-------------|
| Number-theoretic transform (NTT) | FFT over Z_p — fundamental for polynomial multiplication |
| Walsh-Hadamard transform (WHT) | Fast ±1 transforms |
| Discrete Hartley transform (DHT) | Real-valued alternative to FFT |
| Chirp-Z transform | FFT at arbitrary frequencies |
| Fractional Fourier transform | FrFT — interpolation between time and frequency |
| Radon transform | Projections in 2D (tomography) |
| Abel transform | Cylindrical symmetry inversion |

---

## Flags and Concerns

### Flag 1: Thin Test Coverage in Finance/Causal Modules
`volatility.rs` (11 functions, 9 tests), `causal.rs` (6 functions, ~0 adversarial tests). These need adversarial test suites with known analytical solutions. For GARCH(1,1), there are known maximum-likelihood conditions that can be verified analytically.

### Flag 2: GARCH Convergence Not Verified
`garch11_fit` uses gradient-based optimization. There is no adversarial test verifying that it recovers known parameters from simulated data. This is a silent-wrong-answer risk.

### Flag 3: IRT EAP/MLE Not Gold-Standard Verified
`ability_eap` and `ability_mle` should be checked against established IRT software (flexMIRT, mirt). Currently only has inline tests.

### Flag 4: KMeans++ Initialization Not Tested Separately
The k-means module has cluster tests but it's unclear whether kmeans++ initialization quality is independently verified. Poor initialization silently degrades clustering.

### Flag 5: rips_h1 Single-Point Bug (Known, Flagged in Adversarial Test)
`adversarial_boundary.rs:53` documents: "BUG: rips_h0 returns empty diagram for n=1. The n<2 early return skips the survivor pair. Should emit (birth=0, death=∞)." This is a **known failing test** — the test asserts what should be true but the code doesn't do it. Status: unfixed.

### Flag 6: t-SNE Is Gauss-Seidel (Known Issue)
`adversarial_disputed.rs` confirms the t-SNE gradient loop is Gauss-Seidel, not Jacobi (later points see earlier points' updated positions). This is technically incorrect for the standard t-SNE algorithm but is documented as a known behavior.

### Flag 7: SVD Accuracy for Ill-Conditioned Matrices
`svd_adversarial.rs` has 29 tests specifically for SVD edge cases. Check if these pass or if there are known failures.

---

## Summary by Priority

### High Priority Gaps (fundamental, widely needed)
1. ~~**Number theory primitives**~~ — **NOT A GAP** (correction 2026-04-06): `number_theory.rs` has GCD/LCM, ExtGCD, CRT, Miller-Rabin, Pollard's ρ, Euler totient, Möbius, primitive roots, BSGS discrete log, Tonelli-Shanks, continued fractions, partition_count, RSA arithmetic, DH; `bigfloat.rs` has Bernoulli. Genuine remaining gaps: Sieve of Atkin, Lucas-Lehmer, quadratic sieve, Kronecker symbol, Pohlig-Hellman, GF(p^n)/GF(2), polynomial factorization over Z_p.
2. **Special functions breadth** — Bessel, Airy, elliptic integrals: needed for physics/engineering math
3. **Combinatorics** — factorials, binomial, Stirling, Catalan: appears in dozens of algorithms
4. **Orthogonal polynomials** — Legendre, Hermite, Laguerre: Gaussian quadrature, spectral methods
5. ~~**Stochastic processes**~~ — **NOT A GAP** (correction 2026-04-06): `stochastic.rs` has Brownian motion, GBM, Black-Scholes, OU, Poisson (homogeneous+non), Markov chains (DTMC+CTMC), birth-death, M/M/1, Erlang-C, random walk, Itô/Stratonovich. Genuine gaps: CIR, Heston, Lévy, Hawkes, Variance Gamma, Euler-Maruyama SDE solver framework, full Black-Scholes Greeks (gamma/vega/theta/rho missing).

### Medium Priority Gaps
6. **PDEs** — finite difference heat/wave/Laplace: needed for physical modeling
7. **Financial math** — Black-Scholes, Greeks, VaR: directly relevant to WinRapids mission
8. **Control theory** — Kalman filter: used in signal processing and financial state estimation
9. **Ridge/LASSO/ElasticNet** — regularized regression: standard ML toolkit
10. **Advanced ML** — SVM, HMM, EM, decision trees: major algorithm families

### Lower Priority (specialized)
11. Coding theory
12. Categorical algebra
13. Differential geometry (beyond manifold module)
14. Measure theory

---

## Module File Count

```
Total .rs files in crates/tambear/src/: ~90
Active math modules: ~50
Support/infrastructure modules: ~20
Research/experiment modules: ~10
Legacy/unused: ~10
```

## Test Distribution

| Test Category | Count | Quality |
|--------------|-------|---------|
| Gold standard (scipy/numpy/faer parity) | 489 | Excellent — compares to trusted oracle; broader than initially reported |
| Adversarial boundary | ~550 | Excellent — mathematical property assertions |
| Adversarial disputed | 113 | Excellent — settles mathematical disputes |
| Adversarial TBS | 49 | Good — TBS-specific math |
| SVD adversarial | 29 | Good — numerical edge cases |
| Scale ladder | ~17 | Good — magnitude coverage |
| Inline math truth | ~1,300 | Good — analytical result verification |
| **Total** | **~2,918** | |

---

## Appendix A: Deep Dives — 5 Specific Modules

### A1: `proof.rs` — What Kind of Proofs?

**Field**: Formal verification / algebraic proof framework  
**Size**: 3,002 lines, 43 inline tests

This is a Lean4-inspired symbolic proof system embedded in Rust. It is NOT a general theorem prover — it's a specialized system for proving that tambear's GPU primitives have the algebraic structure needed for correctness.

**Sorts**: Nat, Real, Bool, Vec(n, sort), Mat(m,n,sort), Named(string), Arrow(A→B), Product(A×B)  
**Terms**: variables, literals, BinApp/UnApp, lambda, accumulate {grouping, expr, op, data}, Hole  
**BinOps**: Add, Mul, Max, Min, Dot, Sub, Div, Compose  
**UnOps**: Neg, Sq, Sqrt, Log, Exp, Abs, One  
**GroupingTag**: All, ByKey, Prefix, Segmented, Tiled, Windowed(n), Masked

**What it proves**: Algebraic structure of tambear primitives:
- Associativity, commutativity, identity existence (monoid axioms)
- Homomorphism: `scatter_phi(x) op scatter_phi(y) = scatter_phi(x op y)`
- Merge correctness for CopaState
- Collatz four pillars theorem (computational proof)
- Rank-n accumulator properties

**Proof methods**:
- `ByStructure(struct, StructuralFact)` — if you declared a structure, its properties hold by declaration
- `ByCompute(ComputeMethod)` — run the computation and check the result
- `ByInduction(base, step)` — inductive proof
- `ByLemma(name)` — appeal to a previously proven theorem
- `Assumed` — placeholder for truths we haven't proven yet

**Test quality**: 43 tests, all math truth. Tests check: correct proof verification (associativity of Add), correct rejection (Sub is NOT associative), compilation of verified proofs to AccumulateEngine calls.

**What's missing**: No general quantifier reasoning, no proof of GPU kernel correctness, no bridge to the CUDA/wgpu backend.

---

### A2: `series_accel.rs` — What Series?

**Field**: Series acceleration / summability theory  
**Size**: 2,229 lines, 48 inline tests

The module frames series acceleration in the accumulate+gather language. The key insight: every accelerator is a post-processing pass on the prefix scan of partial sums.

**Methods implemented**:
- `partial_sums` — prefix scan (= accumulate(terms, Prefix, Value, Add))
- `cesaro_sum` — arithmetic mean of partial sums (Cesàro summability)
- `aitken_delta2` — Aitken's Δ² process (Shanks e₁ = Aitken for first-order)
- `wynn_epsilon` — full Wynn ε-algorithm tableau with early stopping
- Streaming Wynn epsilon — incremental O(depth) per term
- `richardson_extrapolate` — Richardson extrapolation (given f(h), extrapolate to h→0)
- `euler_transform` — Euler binomial transform for alternating series
- `abel_sum` — Abel summability via exponential series
- `richardson_partial_sums` — Richardson applied to partial sums
- `euler_maclaurin_zeta` — Euler-Maclaurin formula for ζ(s)
- `detect_convergence` — heuristic: alternating / monotone / oscillating / geometric / none
- `accelerate` — auto-selector (picks best method given convergence detection)

**Kingdom mapping** (explicitly documented in module):
- Cesàro/Aitken/Euler: Kingdom A (commutative, Windowed or ByKey on partial sums)
- Wynn epsilon: Kingdom BC (non-commutative inner + iterative outer) — documented as "first clean inhabitant of the (ρ=1, σ=1) cell"
- Richardson: Kingdom A (weighted combine across resolutions = K03 cross-cadence)

**What's missing**: No Padé-based extrapolation via Wynn (it's in interpolation.rs as a separate concern), no Levin-u transform, no Neville-Aitken table for ODEs.

---

### A3: `fold_irreversibility.rs` — What's This Doing?

**Field**: Research mathematics — Collatz conjecture structural analysis  
**Size**: 3,119 lines (largest in codebase)

This is the primary research file for the WinRapids Collatz investigation. It is NOT general-purpose math — it implements the "four-pillar architecture" for the Collatz conjecture proof attempt.

**Core concept**: The "fold" is the first moment of contraction in a Collatz trajectory. An odd number n has τ = trailing_ones(n) consecutive expansion steps, then a mandatory contraction. The module investigates whether this fold is *irreversible* — meaning post-contraction trajectories never re-attain their initial τ.

**What's implemented**:
- `trailing_ones`, `trailing_zeros`, `temperature` — bit structure utilities
- `collatz_odd_step`, `collatz_odd_step_checked` — single Collatz step on odd numbers
- `trace_fold`, `fold_sweep` — trajectory analysis with fold detection
- `fold_extremal_analysis` — extremal orbits by temperature k
- `carry_analysis`, `trajectory_max_tau` — carry mechanism analysis
- `verify_ceiling`, `verify_ceiling_extremals` — ceiling theorem verification
- `carry_mechanism_analysis`, `ratio_contraction_extremals`, `ratio_contraction_exhaustive`
- `tau_sequence` — sequence of temperatures along trajectory
- `carry_chain_statistics` — empirical mean/std of contraction per temperature class
- BigInt variants: `collatz_odd_step_big`, `trace_fold_big`, `ratio_contraction_big`
- `fold_irreversibility_theorem` — builds a Theorem via proof.rs, takes sweep result as witness
- `generalized_symmetric_step`, `nyquist_margin`, `trace_generalized`, `family_analysis`
- `empirical_vd`, `temporal_coverage_extremal` — statistical analysis of step distributions

**Kingdom classification** (documented in module):
- Bit manipulation: K00 (scalar, no accumulate)
- Trajectory tracing: Kingdom B (sequential prefix scan)
- Statistical analysis: Kingdom A (batch statistics over trajectories)

**Relationship to rest of codebase**: Uses `proof.rs` to frame results as Theorems with `ComputeMethod::Sweep` witnesses. The `fold_irreversibility_theorem` function takes a `FoldSweepResult` and packages it as a proof object.

---

### A4: `superposition.rs` — The Math Underneath

**Field**: Parameter superposition / automatic hyperparameter search  
**Size**: 5 pub fns, 11 inline tests

This module is the implementation of "run everything" — every sweepable parameter runs simultaneously. It is **NOT a quantum superposition or linear algebra superposition** — it is an engineering pattern for hyperparameter search without user selection.

**The math**: No new math. The Superposition struct wraps TbsStepOutput for every configuration in the swept range. The `agreement` field is operation-specific — for clustering it's the Rand index between configurations (already in `pipeline.rs`). The `modal_value` is the most-represented cluster count across the sweep.

**Sweep ranges generated**:
- Clustering: k = 2..min(20, √n), always including requested k
- Dim reduction: n_components = 1..min(d, 10)
- Time series: geometric window progression [3,5,7,10,15,20,30,50,75,100,...]
- KDE: bandwidth multipliers [0.5, 0.75, 1.0, 1.5, 2.0] × Silverman

**What's interesting**: The `is_informative()` check (agreement < 0.99) is the diagnostic signal. High sensitivity → parameter matters, user should think about it. Low sensitivity → any parameter works, collapse is stable.

**What's missing**: No formal analysis of when superposition agreement implies stability. No connection to stability theory or PAC learning bounds.

---

### A5: `tam.rs` — What Convergence?

**Field**: Composition depth superposition / kingdom detection  
**Size**: ~400 lines, 15 inline tests

`tam()` runs a single binary operation at ALL composition depths simultaneously:
- Depth 0 (`once`): single-pass fold — `fold(data, op, init)`
- Depth 1 (`scan`): prefix scan — sequential left-fold at each position
- Depth ∞ (`converge`): iterate until fixed point — `apply until |x_n - x_{n-1}| < tol`

**Convergence criterion**: Fixed-point iteration on the `converge` branch. After `once` produces a starting point, iterate `op(x, new_data_element)` until `distance(x_n, x_{n-1}) < tol` or `max_iter` reached.

**`EmergentDepth` classification**: The diagnostic emerges from comparing outputs:
- `Chaotic`: `once` succeeds, `converge` diverges — Kingdom A emerges (example: sum)
- `FixedPoint`: `once` disagrees with `converge`, `converge` converges — Kingdom C emerges (example: Heron's method for √2)
- `Monotone`: `scan` is monotone, `converge` agrees with `scan` limit
- `Oscillating`: scan oscillates; converge may stabilize
- `Undetermined`: insufficient data

**The TamValue trait**: Requires `is_finite()`, `distance(other) -> f64`, and optional `partial_cmp_value()`. Implemented for `f64`, `f32`, `Vec<f64>`.

**Tests**: Verify that sum converges `Chaotic`, Heron's method converges to `√2` within 1e-10, and emergent depth classification is correct for known cases.

**What this IS**: A meta-diagnostic for determining whether an operation has fixed-point structure. Useful for market signal stability analysis — does this signal stabilize or churn?

---

## Appendix B: The Bug Inventory — "Green Suite, Real Bugs"

*Synthesized from initial scout audit + observer correction + scout-2 full classification (2026-04-06)*

### The eprintln! Anti-Pattern and Its Lifecycle

The adversarial tests document bugs using `eprintln!("CONFIRMED BUG: ...")` with assertions that always pass:

```rust
let result = std::panic::catch_unwind(|| some_function(bad_input));
match result {
    Ok(val) => {
        eprintln!("CONFIRMED BUG: function returns {} for degenerate case", val);
        assert!(val.is_finite() || !val.is_finite()); // tautology — always passes
    },
    Err(_) => eprintln!("CONFIRMED BUG: function panics"),
}
```

`cargo test` is green. Bugs are logged to stderr only if `--nocapture` is used. The lifecycle: bug found → eprintln written → bug fixed in production → test upgraded to `assert!` → "CONFIRMED BUG" comment never updated to say "FIXED." This means some eprintln annotations are **stale** (bug already fixed) and others are **active** (bug remains).

### File Status (verified by grep)

| File | eprintln! | Status |
|------|-----------|--------|
| `adversarial_boundary.rs` | **0** | All fixed — 81 `assert!`s verify fixes |
| `adversarial_boundary2.rs` | 5 | Active |
| `adversarial_boundary5.rs` | 12 | **Active frontier** — multivariate/causal/knn |
| `adversarial_boundary6.rs` | 10 | Active |
| `adversarial_boundary7.rs` | 26 | Active (mix of CONFIRMED BUG + NOTE) |
| `adversarial_boundary8.rs` | 29 | Active |
| `adversarial_boundary9.rs` | 20 | Active |
| `adversarial_boundary10.rs` | 37 | **Active frontier** — 31 `NOTE:` + 6 `CONFIRMED BUG` |
| `adversarial_disputed.rs` | ~20 | **STALE** — documents old bugs now fixed; tests misleading |

**Total**: ~139 eprintln lines across boundary2-10, **54 unique bug descriptions** (scout-2 full classification).

### CRITICAL: Infinite Loops (production hang, never return)

| Bug | Module | Trigger | Notes |
|-----|--------|---------|-------|
| `kaplan_meier` infinite loop on NaN | `survival.rs` | NaN in times array | Regression: panic was fixed via `total_cmp`, but NaN detection may leave while-loop running |
| `log_rank_test` infinite loop on NaN | `survival.rs` | NaN in times | Same regression pattern |
| `max_flow` infinite loop | `graph.rs` | source == sink | Augmenting path always exists |
| `sample_geometric(p=0)` infinite loop | `rng.rs` | p=0 | Bernoulli(0) never succeeds; spins forever |
| `ising1d_exact` infinite recursion | `physics.rs` | degenerate input | Surfaced by navigator day-one; exact trigger undocumented — needs investigation |

### HIGH: Panics on Valid Mathematical Inputs

| Bug | Module | Trigger |
|-----|--------|---------|
| `ANOVA` panics on empty groups | `hypothesis.rs` | Empty slice |
| `correlation_matrix` panics on constant data | `multivariate.rs` | 0/0 in stddev |
| `DID` panics with no post-treatment observations | `causal.rs` | Empty post slice |
| `GP regression` panics with `noise_var=0` | `interpolation.rs` | Singular kernel matrix |
| `Hotelling T²` panics with n=1 | `multivariate.rs` | Singular covariance |
| `MCMC` panics when burnin > n_samples | `bayesian.rs` | Index out of bounds |
| `MCMC` panics when `log_target` returns -Inf | `bayesian.rs` | exp(-Inf)=0, division |
| `MCMC` panics with `proposal_sd=0` | `bayesian.rs` | Normal(0,0) undefined |
| `Bayesian regression` panics on underdetermined system | `bayesian.rs` | n < d |
| `conv1d` panics on stride=0 | `neural.rs` | Division by zero |
| `mcd_2d` panics on collinear data | `robust.rs` | Singular scatter matrix |
| `nn_distances` panics on single point | `spatial.rs` | n < 2 |
| `propensity_scores` panics on perfect separation | `causal.rs` | logistic divergence |
| `knn_from_distance` panics with k=0 | `knn.rs` | Division / empty result |

### MEDIUM: Silent NaN/Inf (wrong finite answer, no error)

| Bug | Module | Symptom |
|-----|--------|---------|
| `batch_norm` with eps=0 on constant data | `neural.rs` | NaN/Inf |
| `BCE loss` with predicted exactly 0 or 1 | `neural.rs` | log(0) = -Inf |
| `chi2_goodness_of_fit` with zero expected count | `hypothesis.rs` | NaN |
| `Clark-Evans R` with area=0 | `spatial.rs` | NaN (div by zero) |
| `cosine_similarity_loss` for zero vectors | `neural.rs` | NaN |
| `cox_ph` with perfect separation | `survival.rs` | NaN (exp overflow in log-likelihood) |
| `global_avg_pool2d` with 0 spatial dims | `neural.rs` | NaN |
| `GP` with `length_scale=0` | `interpolation.rs` | NaN/Inf |
| `GP` with `noise=0` fails to interpolate at training points | `interpolation.rs` | Numerical precision |
| **LME σ² M-step biased** | `mixed_effects.rs:146-151` | EM trace correction uses `sigma2` where `ng·sigma2_u` needed — σ² biased toward zero; worst when σ²_u << σ² |
| `KDE` with `bandwidth=0` | `nonparametric.rs` | NaN/Inf |
| `KNN` selects NaN-distance neighbor over finite neighbor | `knn.rs` | Wrong selection |
| `medcouple` for 2 data points | `robust.rs` | NaN |
| `Moran's I` for constant values | `spatial.rs` | 0/0 = NaN |
| `Ripley's K` with area=0 | `spatial.rs` | Non-finite |
| `RoPE` with base=0 | `neural.rs` | NaN/Inf |
| `sample_exponential(lambda=0)` | `rng.rs` | Returns value (should guard) |
| `sample_gamma(alpha=0)` | `rng.rs` | NaN |
| `silverman_bandwidth` for constant data | `nonparametric.rs` | Returns 0 or NaN → KDE blows up downstream |
| `temperature_scale` with T=0 | `neural.rs` | Division by zero → Inf |

### LOW: Wrong Mathematical Answer (finite but incorrect)

| Bug | Module | Correct behavior |
|-----|--------|-----------------|
| `Dijkstra` with negative weights | `graph.rs` | Should error or use Bellman-Ford |
| ~~`erf(0)` returns non-zero~~ | ~~`special_functions.rs`~~ | **STALE** — guard at line 34 confirmed |
| `Lagrange` with duplicate x | `interpolation.rs` | Should error on degenerate nodes |
| `Rényi entropy at α=1 ≠ Shannon entropy` | `information_theory.rs` | Limit must be taken as α→1 |
| `R-hat` for identical chains | `bayesian.rs` | Should return ~1.0, not NaN |
| `R-hat` for single chain | `bayesian.rs` | Between-chain var undefined, should error |
| `Richardson extrapolation` with ratio=1 | `series_accel.rs` | ratio^p - 1 = 0, should guard |
| `Tsallis entropy at q=1 ≠ Shannon entropy` | `information_theory.rs` | Limit must be taken as q→1 |
| `breusch_pagan_re` with t=1 | `panel.rs` | Division by t-1=0 |
| `mat_mul` dimension mismatch | `linear_algebra.rs` | Silent wrong answer (no check) |
| `Aitken Δ²` for constant sequence | `series_accel.rs` | 0/0 → NaN, should return early |

### Stale Disputed Tests (adversarial_disputed.rs)

These tests document bugs in an **older version** of the code that have since been fixed. The tests still pass but are now **misleading** — they assert on non-panic/non-degenerate output rather than positively verifying the fix.

| "Bug" | Current status |
|-------|---------------|
| t-SNE uses Gauss-Seidel gradient | **FIXED** — line 293 in `dim_reduction.rs` has Jacobi comment + grad_buf pattern |
| t-SNE missing early exaggeration | **FIXED** — line 296: `let exag = if iter < 250 { 4.0 } else { 1.0 };` |
| kaplan_meier panic on NaN | **FIXED** via `total_cmp` — but introduced infinite-loop regression (see CRITICAL above) |

Recommendation: rewrite these three tests to positively assert the corrected behavior.

### Recommended Fix Priority

1. **Fix the 4 infinite-loop bugs** — these hang production; require code fixes, not test fixes
2. **Rewrite 3 stale disputed tests** — replace misleading tests with positive assertions of fixed behavior
3. **Convert active CONFIRMED BUG tests** — for unfixed bugs, upgrade from silent `eprintln!` to `#[should_panic]` or proper `Err(...)` assertions; makes failures actionable
4. **Add regression tests for fixed bugs** — `boundary.rs` has 81 asserts that do this right; extend the pattern to boundary2-10 as bugs get fixed
5. **Classify NOTE: items in boundary10** — for each panic-on-degenerate-input: either add a guard (return `Err`) or add `// INTENTIONAL: panics when X` in production source

Note on `NOTE:` items in boundary10: tambear's NaN-free invariant pushes these edge cases to the ingestion boundary (`.lock()`). Some panics-on-degenerate-input are *correct* under that invariant — the fix is upstream (boundary rejection), not inside the function.

---

## Appendix C: Gold Standard Cross-Reference

The math-library campsites (campsites/math-library/) contain gold standards for ~18 families. Cross-referencing against tambear's implementations reveals additional gaps not covered in the main gap analysis.

### Family 08: Nonparametric — What Gold Standards Specify, What's Missing

The gold standard explicitly documents these as "blocked on SortedPermutation" (meaning they need rank infrastructure that exists in tambear but weren't wired up):

| Missing from tambear | What it is | Gold standard oracle |
|---------------------|------------|---------------------|
| Jarque-Bera test | Normality test from moments (skew + kurtosis) | scipy.stats.jarque_bera |
| D'Agostino-Pearson K² | More powerful normality test | scipy.stats.normaltest |
| Shapiro-Francia (large n) | Royston (1992) approx to Shapiro-Wilk | scipy.stats.shapiro |
| Anderson-Darling | More powerful than KS for tails | scipy.stats.anderson |
| Shapiro-Wilk (full) | Gold standard normality test | scipy.stats.shapiro |

**Note**: These are NOT infrastructure-blocked — MomentStats(order=4) exists. Jarque-Bera is zero new GPU compute, pure extraction from existing MomentStats.

### Family 10: Regression — What Gold Standards Specify, What's Missing

Gold standard documents the entire regression family. Tambear has linear + logistic. Missing:

| Missing | Oracle |
|---------|--------|
| Ridge regression (L2) | sklearn.linear_model.Ridge |
| LASSO | sklearn.linear_model.Lasso (coordinate descent) |
| Elastic Net | sklearn.linear_model.ElasticNet |
| Cook's distance / leverage | statsmodels OLS influence |
| Studentized residuals | statsmodels OLS influence |
| VIF (variance inflation factor) | statsmodels variance_inflation_factor |
| Durbin-Watson statistic | statsmodels durbin_watson |
| Poisson GLM | statsmodels.formula.api.glm (family=Poisson) |
| Negative binomial GLM | statsmodels.formula.api.glm (family=NegativeBinomial) |
| Quantile regression | statsmodels.regression.quantile_regression |

### Family 18: Volatility — Kingdom-Mapped Gap Analysis

The volatility gold standard has the most detailed tambear integration mapping. What's implemented vs. specified:

| Algorithm | Status | Notes |
|-----------|--------|-------|
| GARCH(1,1) | Implemented | Bug: perfect separation causes NaN LL |
| EGARCH | Missing | Needs Custom(log-variance) op |
| GJR-GARCH | Missing | Threshold asymmetry |
| FIGARCH | Missing | Fractional differencing |
| HAR-RV | Missing | Multi-scale RV regression |
| Black-Scholes IV | Missing | Root-finding via Newton on BS formula |
| VIX-style | Missing | Model-free, strip integration |
| TSRV | Missing | Two-scale realized variance |
| Stochastic Volatility (MCMC) | Missing | Needs HMC/NUTS |

### Key Cross-Reference Finding: Jarque-Bera is Zero Work

From the gold standard: "Zero new GPU compute. Pure extraction from MomentStats(order=4)."

tambear has `MomentStats` with order=4 support. The Jarque-Bera test is:
```
JB = n/6 · [g₁² + (g₂_excess)²/4]
p = chi2_survival(JB, df=2)  // chi2_sf already exists in special_functions.rs
```

This is extractable from existing `moments_ungrouped()` output with ~10 lines of code. No new accumulate primitives needed. Yet it's not implemented.

D'Agostino-Pearson K² is slightly more complex (requires the normal transform for skewness) but the oracle is clear: scipy.stats.normaltest.

---

## Appendix D: Pathmaker Handoff Queue

*Synthesized from scout audit + observer test audit + math-researcher verification + naturalist challenges. Ordered by effort and urgency.*

### Tier 0 — Fix First (bugs in production, no new math needed)

| Task | File | Effort | Notes |
|------|------|--------|-------|
| LME σ² M-step | `mixed_effects.rs:146-151` | ~5 lines | Replace trace_correction with `Σ_g ng·tau2_g`; formula in `math-verification.md` Issue 4 |
| kaplan_meier + log_rank infinite loop on NaN | `survival.rs` | ~2 guards | NaN check before while-loop; `total_cmp` fix introduced regression |
| max_flow infinite loop when source==sink | `graph.rs` | ~1 guard | Early return if source == sink |
| sample_geometric(p=0) infinite loop | `rng.rs` | ~1 guard | Guard p > 0 at entry |
| rips_h0 empty for n=1 | `tda.rs` | ~3 lines | n<2 early return skips lone survivor; fix: add survivor to persistence diagram |
| tbs_lint.rs kingdom taxonomy drift | `tbs_lint.rs` | refactor | Challenge 26: GARCH misclassified Kingdom C; manual lookup will always drift; needs derived classification |

### Tier 1 — Small (naming/docstring/≤35 lines)

| Task | Effort | Notes |
|------|--------|-------|
| **Challenge 27**: `Op::canonical_structure() -> Option<Structure>` | ~35 lines | Closes proof↔accumulate gap; every accumulate call auto-generates correctness certificate; navigator routing directly |
| **Challenge 24**: prime thermodynamics naming | docstring + 1 test + 5 lines | `prime_fold_surface(n)` calling `nucleation_hierarchy` over first n primes; test verifies partial sum vs `bigfloat::zeta(2.0)` |
| Jarque-Bera test | ~10 lines | Zero new GPU compute; `MomentStats(order=4)` exists; `JB = n/6·(g₁² + g₂²/4)`, then `chi2_sf(JB, 2)` |
| D'Agostino-Pearson K² | ~20 lines | Normal transform for skewness + chi2_sf; oracle: `scipy.stats.normaltest` |

### Tier 2 — Medium (Tier 1 implementations, specs ready)

| Task | Spec location | Notes |
|------|--------------|-------|
| OLS diagnostics (residuals, Cook's D, leverage, VIF, AIC/BIC) | `regression-specs.md` | lstsq exists; add hat matrix + diagnostics around it |
| Ridge regression | `regression-specs.md` | Closed form: (XᵀX + λI)⁻¹Xᵀy; SVD path for GCV |
| Lasso | `regression-specs.md` | Coordinate descent + soft-threshold S(z,λ) |
| Logistic regression | `regression-specs.md` | IRLS; stability clipping specified |
| Poisson regression | `regression-specs.md` | IRLS; same engine as logistic |
| **Challenge 29**: Kalman filter | `garden/entry-006` | Transcription job; hard math solved; linear discrete KF + RTS smoother. **Oracle test required before merge** — run TWO cases on same local-level model (F=1,H=1,Q=1,R=4,prior N(0,10)):<br>• n=4: obs=[0,0,0,10] — assert `smoothed[1] > filtered[1]` (2-level Blelloch tree)<br>• n=8: obs=[0,0,0,10,0,0,0,0] — assert `smoothed[1] > filtered[1]` (3-level tree; exercises step-3 combine(e01,e23) — first position where RTS backward J_t sign error manifests per garden/entry-006)<br>Reference: statsmodels `UnobservedComponents` or filterpy `rts_smoother` for exact values. |
| Distribution objects (Exponential, Gamma, Beta, Poisson, Binomial...) | `taxonomy.md §2.2` | Sampling already exists in rng.rs; need PDF/CDF/PPF/MLE per distribution |

### Tier 3 — Large (new primitives, highest impact)

**Core finding (naturalist, 2026-04-06)**: Every sequential algorithm of the form `state_t = A_t · state_{t-1} + b_t` parallelizes via the affine composition semiring (Blelloch scan, O(log n)). The `Op` enum extension (challenge 32) is the entire implementation queue. See `~/.claude/garden/20260406-the-affine-composition-semiring.md`.

**Complete Op enum target list (implementation order):**

| Op | Algorithms unlocked | Notes |
|----|---------------------|-------|
| `Op::WelfordMerge` | Online mean/variance, grouped moments | Already in descriptive.rs — formalize as Op |
| `Op::AffineCompose` (b=0) | GARCH σ² recurrence, EWMA, AR(1), Adam m/v (no bias) | Scalar: state = A·prev |
| `Op::AffineCompose` (b≠0) | Holt's trend, full GARCH with ω intercept, ARMA innovations | Full affine: state = A·prev + b |
| `Op::LogSumExpMerge` | Softmax (stable), Cox partial likelihood, forward HMM | log(exp(a) + exp(b)) in monoid form |
| `Op::OuterProductAdd` | Gram matrices (BLR, OLS, CCA, LDA, MANOVA, Hotelling) | Σ xᵢxᵢᵀ — confirmed in bayesian.rs:121-128 |
| `Op::MatMulPrefix(3)` | Thomas algorithm (cubic splines), IIR biquad Direct Form II | 3×3 homogeneous coord matrix; covers `natural_cubic_spline`, `clamped_cubic_spline`, `biquad_cascade` |
| `Op::SarkkaMerge` | Full Kalman filter + RTS smoother | Requires correction term from `garden/entry-006`; see challenge 29 warning on RTS sign |

**Additional findings:**
- Particle filter (SMC) = Kingdom A (not B): `{accumulate(All, propagate), accumulate(All, reweight), gather(resample)}` — embarrassingly parallel; currently missing from `bayesian.rs` entirely
- `SpatialWeights` in `spatial.rs` claims CSR format in docstring but uses `Vec<Vec<(usize,f64)>>` — documentation-before-code gap (real; spatial.rs:222)
- Savitzky-Golay: listed in signal_processing.rs header AND implemented at line 732 — NOT a gap (naturalist initially misread)

### Test Rewrite Queue (observer Phase 2)

**Methodology correction**: counting `assert_eq` and `> 0.0` patterns misclassifies tests that use `close(val, expected, tol, label)` — those are MATH tests. Reading actual assertion content is the only reliable classifier.

**Observer verification result** (direct code reading, 2026-04-06):
- `tda.rs` — 7/10 MATH: merge distances exact (deaths at 1.0, 2.0), persistence entropy = ln(2), Betti numbers at computed thresholds, bottleneck/Wasserstein = 0 for identical diagrams. **h1_triangle** fixed (now verifies 2 H₀ merges at r=1, 0 persistent H₁).
- `causal.rs` — MATH quality: DiD verifies ATT=3.0 exactly, IPW verifies ATE = mean difference (identity), RDD verifies 3.0 jump, E-value uses exact formula.
- `panel.rs` — MATH quality: all tests do known-coefficient recovery (FE→β=3.0, FD→β=4.0, Hausman correct behavior).
- `time_series.rs` — MATH quality: `close()` calls for exact finite-difference values, SES of constant = constant, ACF(0) = 1.0 exactly.
- `volatility.rs` — Confirmed weak. **Now fixed**: `ewma_constant_returns` checks r²=0.0001 (mathematically derivable); `garch_fits` checks α within 0.15 of 0.1 and β within 0.15 of 0.85.

**Actual remaining test rewrite needs** (3 fixed, 1 remaining module-level, 3 stale):
1. `volatility.rs` — ✓ **complete, all MATH quality**: ~~ewma_constant~~ fixed; ~~garch_fits~~ fixed; ~~roll_spread~~ fixed (pure bid-ask bounce DGP, S=0.02, `close(result, 0.02, 5e-4)`); ~~kyle_lambda~~ fixed (exact linear DGP ΔP=0.005·Q no noise, `close(result, 0.005, 1e-8)`). Amihud and bipower remain as sign checks — no closed-form oracle for arbitrary return sequences.
2. `adversarial_disputed.rs` — 3 stale tests: rewrite to positively assert corrected behavior (t-SNE Jacobi two-phase update, kaplan_meier NaN guard, early exaggeration factor)
3. Kalman smoother oracle test — needed before challenge 29 merges (see Tier 2 queue above)

---

*Updated: 2026-04-06. Incorporates navigator deep-dive requests, observer bug count cross-check, math-researcher verification (Issue 4 LME bug), naturalist challenges 24-28; session 2: three previously undocumented modules discovered — number_theory.rs (comprehensive number theory: §7.1 was entirely wrong), stochastic.rs (BM/GBM/Black-Scholes/OU/Poisson/Markov chains: §3.7 was entirely wrong), multivariate.rs (Hotelling T²/MANOVA/LDA/CCA/Mardia); spatial.rs CSR docstring fixed.*
