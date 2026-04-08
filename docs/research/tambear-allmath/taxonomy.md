# Tambear Math Taxonomy
**Author**: math-researcher  
**Date**: 2026-04-06  
**Status**: Living document — add as discovered

This is the exhaustive roadmap of ALL math that tambear must implement.
Each section: ✅ = implemented, 🔲 = gap, 🔷 = partial.
Accumulate+gather decompositions noted where non-obvious.

---

## FORMAT CONVENTION
Each algorithm entry:
```
**Name** — brief description
- Decomposition: accumulate(grouping, expr, op) + gather(addressing)
- Sufficient stats: what minimal representation exists
- Parameters: ALL valid tunings across any field
- Assumptions: data requirements
- Expert default: standard practice
```

---

# PART 1 — NUMERICAL FOUNDATIONS

## 1.1 Arbitrary Precision Arithmetic
- ✅ BigInt — arbitrary precision integers (`bigint.rs`)
- ✅ BigFloat — arbitrary precision floating point (`bigfloat.rs`)
- 🔲 Rational arithmetic — exact fractions (p/q form, GCD normalization)
- 🔲 Interval arithmetic — guaranteed bounds, IEEE 1788 compliance
- 🔲 Dual numbers — forward-mode automatic differentiation, ε² = 0

## 1.2 Root Finding
- ✅ **Bisection** — bracketed root finding (`numerical.rs`)
  - Decomposition: sequential iteration (no accumulate), gather(last bracket)
  - Parameters: tolerance, max_iter, bracket [a,b]
  - Assumptions: sign change in [a,b], continuous function
  - Default: tol=1e-10, max_iter=1000

- ✅ **Newton-Raphson** — f(x)/f'(x) iteration (`numerical.rs::newton`)
  - Decomposition: sequential, gather(current estimate)
  - Parameters: x0, tol, max_iter
  - Assumptions: differentiable, nonzero derivative near root
  - Default: tol=1e-10, max_iter=100

- ✅ **Secant method** — finite difference Newton (no derivative) (`numerical.rs::secant`)
  - Parameters: x0, x1, tol, max_iter

- ✅ **Brent's method** — bisection + secant + inverse quadratic interpolation (`numerical.rs::brent`)
  - Best general-purpose 1D root finder. Default algorithm.

- ✅ **Fixed point** — iterate g(x) until convergence (`numerical.rs::fixed_point`)
  - Bonus: not in taxonomy v1. Heron's method for √2 works here.

- 🔲 **Broyden's method** — quasi-Newton for nonlinear SYSTEMS
  - Sufficient stats: approximate Jacobian B
  - Parameters: x0 (vector), tol, max_iter

- 🔲 **Halley's method** — cubic convergence, needs f''
  - Parameters: x0, tol, max_iter

## 1.3 Numerical Integration (Quadrature)
- ✅ **Trapezoidal rule** — O(h²) (`numerical.rs::trapezoid`)
  - Decomposition: accumulate(consecutive_pairs, trapezoid_area, sum)

- ✅ **Simpson's rule** — O(h⁴), pairwise: (f(a) + 4f(m) + f(b)) · h/6 (`numerical.rs::simpson`)
  - Decomposition: accumulate(triplets, simpson_contrib, sum)

- ✅ **Adaptive Simpson** — recursive halving until error < tol (`numerical.rs::adaptive_simpson`)
  - Parameters: tol, max_depth

- 🔲 **Romberg integration** — Richardson extrapolation on trapezoid rule
  - Decomposition: gather(trapezoid_table) → Richardson accelerate

- 🔷 **Gauss-Legendre** — 5-point only (`numerical.rs::gauss_legendre_5`); full n-point family missing
  - Parameters: n_points (2,3,4,5,7,10,20,...)
  - Sufficient stats: Legendre roots + weights (precomputed)

- 🔲 **Gauss-Hermite** — for ∫ f(x)exp(-x²)dx (quantum mechanics, probability)
  - Parameters: n_points
  - Sufficient stats: Hermite roots + weights

- 🔲 **Gauss-Laguerre** — for ∫₀^∞ f(x)exp(-x)dx
  - Parameters: n_points

- 🔲 **Clenshaw-Curtis** — cosine basis quadrature, good for smooth functions
  - Parameters: n_points, uses DCT for weights

- 🔲 **Monte Carlo integration** — for high-dimensional integrals
  - Decomposition: accumulate(samples, f_eval, sum) / n
  - Parameters: n_samples, rng_seed

- 🔲 **Quasi-Monte Carlo** — low-discrepancy sequences (Sobol, Halton)
  - Parameters: n_samples, sequence_type

## 1.4 Numerical Differentiation
- 🔲 **Forward difference** — (f(x+h) - f(x)) / h
- ✅ **Central difference** — (f(x+h) - f(x-h)) / (2h) (`numerical.rs::derivative_central`, `derivative2_central`)
- ✅ **Richardson extrapolation for derivatives** — O(h^2n) accuracy (`numerical.rs::derivative_richardson`)
- 🔲 **Complex step differentiation** — Im[f(x+ih)] / h, machine precision accurate

## 1.5 ODE Solvers
- ✅ **Euler method** — y_{n+1} = y_n + h·f(t_n, y_n) (`numerical.rs::euler`)
  - Decomposition: accumulate(time_steps, euler_step, state_update)
  
- ✅ **Runge-Kutta 4** — classic 4-stage method, O(h⁴) (`numerical.rs::rk4`, `rk4_system`)
  - Parameters: h (step size), t_span, y0
  - Default algorithm for non-stiff ODEs

- ✅ **Dormand-Prince (RK45)** — adaptive step, embedded pair (`numerical.rs::rk45`)
  - Parameters: rtol, atol, h_init, h_max
  - Industry default for non-stiff adaptive ODE

- 🔲 **Adams-Bashforth** — explicit multistep (order 1-5)
- 🔲 **Adams-Moulton** — implicit multistep predictor-corrector
- 🔲 **BDF (Backward Differentiation Formulas)** — for stiff systems (LSODE/VODE style)
  - Parameters: order (1-6), rtol, atol
  
- 🔲 **Symplectic Euler** — preserves symplectic structure (physics engines)
- 🔲 **Störmer-Verlet** — second-order, time-reversible, symplectic
  - Parameters: h (step size)
  - Used for: molecular dynamics, celestial mechanics

- 🔲 **Forest-Ruth** — 4th-order symplectic
- 🔲 **Leapfrog** — half-step velocity update, standard MD

## 1.6 PDE Solvers
- 🔲 **Finite difference — heat equation** — explicit/implicit/Crank-Nicolson
  - Decomposition: accumulate(spatial_neighbors, stencil_op, update)
  
- 🔲 **Finite difference — wave equation**
- 🔲 **Finite difference — Laplace/Poisson** — iterative (Jacobi, Gauss-Seidel, SOR)
- 🔲 **Finite difference — advection** — upwind schemes, Lax-Wendroff
- 🔲 **Spectral methods for PDEs** — expand in Fourier/Chebyshev basis
- 🔲 **Finite element basics** — 1D, 2D triangular elements, Galerkin

## 1.7 Linear Systems (Advanced)
Already have: LU, Cholesky, QR, SVD, pinv, sym_eigen, power_iteration

- 🔲 **GMRES** — iterative solver for non-symmetric sparse systems
- 🔲 **Conjugate Gradient (CG)** — iterative for SPD systems
  - Decomposition: accumulate(CG_direction, mat_vec_apply, update)
  - Parameters: tol, max_iter, preconditioner

- 🔲 **BiCGSTAB** — stabilized BiCG for non-symmetric
- 🔲 **Preconditioners** — diagonal (Jacobi), incomplete LU (ILU)
- 🔲 **Schur decomposition** — A = QSQ*, triangular form
- 🔲 **Generalized eigenvalues** — Ax = λBx (LAPACK's dsygv analog)
- 🔲 **Matrix exponential** — expm(A) via Padé approximation + scaling/squaring
  - Used for: ODEs, Lie groups, Markov chains
  - Sufficient stats: eigendecomposition if symmetric

- 🔲 **Matrix square root** — A^(1/2), Schur method
- 🔲 **Matrix logarithm** — logm(A), inverse of expm
- 🔲 **Hessenberg form** — reduction to near-triangular, preprocessing for QR iteration
- 🔲 **Band matrix solvers** — banded Cholesky/LU for FEM
- 🔲 **Sparse matrix formats** — CSR, CSC, COO; sparse BLAS operations

---

# PART 2 — PROBABILITY & STATISTICS

## 2.1 Descriptive Statistics
Already have: moments, quantiles, geo/harmonic/trimmed/winsorized mean, IQR, CV, SEM, skewness, kurtosis, Bowley

- 🔲 **L-moments** — linear combinations of order statistics, robust to outliers
  - L1=mean, L2=half of expected range, L-skewness, L-kurtosis
  - Sufficient stats: order statistics

- 🔲 **Probability weighted moments (PWM)**
- 🔲 **Weighted descriptive stats** — all moments with sample weights
  - Decomposition: accumulate(samples, weighted_moment, sum) / sum(weights)

- 🔲 **Robust location: median, MAD, Harrell-Davis quantile**
  - Decomposition: sort → gather(specific indices) or smooth gather

## 2.2 Probability Distributions — Full Library
Already sampling some (normal, exp, gamma, beta, chi2, t, F, Cauchy, lognormal, Bernoulli)

Each distribution needs: PDF, CDF, SF, PPF (quantile function), log-PDF, mean, variance, entropy, KL-divergence, MLE fit, MOM fit.

**Continuous:**
- ✅ Normal (Gaussian) — via special_functions
- ✅ Student-t — via special_functions
- ✅ F-distribution — via special_functions
- ✅ Chi-squared — via special_functions
- 🔲 **Uniform** (continuous) — trivial but needed
- 🔷 **Exponential** — sampling only (`rng.rs::sample_exponential`); PDF/CDF/PPF missing
- 🔷 **Gamma** — sampling only (`rng.rs::sample_gamma`); PDF/CDF/PPF missing
  - Sufficient stats for MLE: n, Σx, Σln(x)
  
- 🔷 **Beta** — sampling only (`rng.rs::sample_beta`); PDF/CDF/PPF missing
  - Sufficient stats for MLE: n, Σln(x), Σln(1-x)

- 🔲 **Weibull** — shape k, scale λ; reliability, survival
  - Sufficient stats for MLE: n, Σxᵏ, Σln(x)

- 🔲 **Pareto** — power-law tail; extreme values, income distribution
- 🔷 **Log-normal** — sampling only (`rng.rs::sample_lognormal`); PDF/CDF/PPF missing

- 🔲 **Triangular** — simple bounded support
- 🔲 **Logistic** — heavier tails than normal; S-shaped CDF
- 🔲 **Laplace (double exponential)** — L1 analog of normal; robust noise
- 🔷 **Cauchy** — sampling only (`rng.rs::sample_cauchy`); PDF/CDF/PPF missing; limit of ratio of normals
- 🔲 **Gumbel (type I extreme value)** — maximum of exponentials
- 🔲 **Fréchet (type II extreme value)** — maximum of Pareto
- 🔲 **Weibull as GEV type III (reversed)** — minimum distribution
- 🔲 **GEV (Generalized Extreme Value)** — unifies Gumbel/Fréchet/Weibull
  - Parameters: μ (location), σ (scale), ξ (shape)
  - ξ=0 → Gumbel, ξ>0 → Fréchet, ξ<0 → Weibull tail

- 🔲 **GPD (Generalized Pareto Distribution)** — excess over threshold
  - Parameters: μ, σ, ξ; POT method for EVT

- 🔲 **Von Mises** — circular distribution; direction data
  - Parameters: μ (mean angle), κ (concentration)

- 🔲 **Stable distributions (α-stable)** — Lévy, Cauchy, Normal as special cases
  - Parameters: α (stability), β (skewness), γ (scale), δ (location)
  - No closed-form PDF except α=1 (Cauchy), α=2 (Normal), α=1/2,β=1 (Lévy)

- 🔲 **Rayleigh** — special Weibull (k=2); radio signal amplitudes
- 🔲 **Maxwell-Boltzmann** — speed distribution in gas
- 🔲 **Inverse Gaussian (Wald)** — first passage time of Brownian motion
- 🔲 **Generalized Gamma** — Stacy distribution; unifies gamma, Weibull, log-normal

**Discrete:**
- 🔷 **Bernoulli** — sampling only (`rng.rs::sample_bernoulli`); PMF/CDF object missing
- 🔷 **Binomial** — sampling only (`rng.rs::sample_binomial`); PMF/CDF/PPF object missing
  - Sufficient stats: (n, k) where k = sum of successes
  - PMF: C(n,k) pᵏ (1-p)^(n-k)

- 🔷 **Poisson** — sampling only (`rng.rs::sample_poisson`); PMF/CDF/PPF object missing
  - Sufficient stats for MLE: n, Σx (since λ̂ = x̄)
  - PMF: exp(-λ)λᵏ/k!

- 🔲 **Negative Binomial** — failures before r successes; overdispersion model
  - Parameters: r (successes), p; or mean μ + overdispersion φ

- 🔷 **Geometric** — sampling only (`rng.rs::sample_geometric`); PMF/CDF/PPF object missing
- 🔲 **Hypergeometric** — sampling without replacement
- 🔲 **Multinomial** — generalization of binomial to k categories
- 🔲 **Zipf / Zeta** — power law, frequency of words/cities
- 🔲 **Conway-Maxwell-Poisson** — flexible count, handles under/over-dispersion

**Multivariate:**
- 🔲 **Multivariate Normal** — Σ (covariance), μ (mean); Mahalanobis distance, PDF, sampling
  - Sufficient stats: n, Σx, xxᵀ (second moment matrix)
  - Decomposition: accumulate(observations, outer_product, sum_matrix)

- 🔲 **Multivariate-t** — heavier tails, ν degrees of freedom
- 🔲 **Dirichlet** — multivariate beta; Bayesian prior for categorical
- 🔲 **Wishart** — distribution over positive definite matrices
- 🔲 **Inverse-Wishart** — conjugate prior for Σ in MVN

## 2.3 Statistical Testing — Extended
Already have: t-tests, ANOVA, chi², proportions, nonparametric suite

- 🔲 **Levene's test** — equality of variances (robust)
- 🔲 **Bartlett's test** — equality of variances (assumes normality)
- 🔲 **Brown-Forsythe test** — robust Levene
- 🔲 **F-test for variances** — Fisher's exact variance ratio
- 🔲 **Shapiro-Wilk** — normality test (small n)
- 🔲 **Anderson-Darling** — normality / general distribution test
- 🔲 **Lilliefors** — KS test with estimated parameters (normality)
- 🔲 **Jarque-Bera** — normality via skewness+kurtosis
- 🔲 **D'Agostino-Pearson** — normality omnibus
- 🔲 **Granger causality** — temporal precedence test (VAR-based)
- 🔲 **KPSS test** — stationarity (null = stationary, complement to ADF)
- 🔲 **PP test (Phillips-Perron)** — unit root with nonparametric correction
- 🔲 **Ljung-Box** — residual autocorrelation
- 🔲 **Durbin-Watson** — serial correlation in regression residuals
- 🔲 **Breusch-Godfrey** — serial correlation, general order
- 🔲 **White's test** — heteroskedasticity
- 🔲 **Breusch-Pagan (regression)** — heteroskedasticity
- 🔲 **RESET test (Ramsey)** — functional form misspecification
- 🔲 **Hausman test** — already in panel but needs general formulation
- 🔲 **Post-hoc tests**: Tukey HSD, Scheffé, Dunnett, Games-Howell
- 🔷 **Multiple comparisons**: ✅ Bonferroni/Holm/BH (`hypothesis.rs`); 🔲 BY, Storey q-value missing
- 🔲 **Cochran's Q** — repeated measures k-group binary
- 🔲 **Friedman test** — nonparametric repeated-measures ANOVA
- 🔲 **Page's test** — ordered alternatives nonparametric

## 2.4 Regression — Full Library
- 🔷 **OLS (complete)** — `linear_algebra.rs::lstsq` (QR-based coefficients only); residuals, Cook's D, leverage, VIF, BIC/AIC, F-test, per-coeff t-tests all missing
  - Decomposition: accumulate(rows, xᵀx_contribution, sum_matrix) + gather(Cholesky solve)
  - Sufficient stats: n, XᵀX (p×p), Xᵀy (p×1), Σy², for all standard statistics

- 🔲 **GLS** — generalized least squares, heteroskedastic/correlated errors
- 🔲 **WLS** — weighted least squares
- 🔲 **Ridge regression** — L2 penalty: (XᵀX + λI)⁻¹Xᵀy
  - Decomposition: same as OLS with diagonal shift; closed form
  - Parameters: λ (shrinkage), standardize (bool)

- 🔲 **Lasso** — L1 penalty, sparsity-inducing; coordinate descent or ISTA/FISTA
  - Parameters: λ, path (full regularization path vs single)
  - No sufficient stats — path algorithm needed

- 🔲 **Elastic net** — α·L1 + (1-α)·L2
  - Parameters: λ, α (mixing ratio)

- 🔲 **Logistic regression (binary)** — Newton-Raphson or IRLS
  - Sufficient stats: same as OLS under IRLS steps
  - Parameters: C (regularization), max_iter, solver

- 🔲 **Logistic regression (multinomial/softmax)**
- 🔲 **Probit regression** — Φ⁻¹(p) = Xβ; ordinal version (proportional odds)
- 🔲 **Poisson regression** — log(μ) = Xβ, IRLS
  - Sufficient stats: n, Σxᵢ, Σxᵢyᵢ per group
  
- 🔲 **Negative binomial regression** — overdispersed count data
- 🔲 **Gamma regression** — positive continuous, log link
- 🔲 **Quantile regression** — minimize Σρ_τ(yᵢ - xᵢᵀβ); interior point or simplex
  - Parameters: τ (quantile), solver

- 🔲 **Tobit model** — censored regression; maximum likelihood
- 🔲 **Heckman selection model** — two-stage correction for sample selection
- 🔲 **GAM (Generalized Additive Model)** — sum of smooth functions, backfitting
- 🔲 **LOWESS / LOESS** — local polynomial regression smoothing
  - Parameters: span (bandwidth), degree (1 or 2), iterations (robustness)
  - Decomposition: for each x*, accumulate(nearby_points, weighted_poly_fit, solve)

- 🔲 **Polynomial regression** — orthogonal or Vandermonde basis
- 🔲 **Spline regression** — B-splines, natural splines, smoothing splines
  - Parameters: knots or df, lambda (smoothing)

## 2.5 Model Selection & Information Criteria
- 🔲 **AIC** — -2·log(L) + 2k
- 🔲 **BIC** — -2·log(L) + k·log(n)
- 🔲 **AICc** — AIC corrected for small samples
- 🔲 **DIC** — Deviance Information Criterion (Bayesian)
- 🔲 **WAIC** — Widely Applicable IC (Bayesian)
- 🔲 **Cross-validation** — k-fold, leave-one-out (LOO), stratified
  - Decomposition: accumulate(folds, train_eval, loss_sum) / k
- 🔲 **R², adjusted R²** — coefficient of determination
- 🔲 **Likelihood ratio test** — χ² = -2·Δlog(L)
- 🔲 **Wald test** — (β̂ - β₀)ᵀ Var(β̂)⁻¹ (β̂ - β₀)
- 🔲 **Score test (Lagrange Multiplier)** — gradient of log-L at null

## 2.6 Copulas
- 🔲 **Gaussian copula** — Φₙ(Φ⁻¹(u₁), ..., Φ⁻¹(uₙ); Σ)
- 🔲 **Student-t copula** — heavier tail dependence
- 🔲 **Clayton copula** — lower tail dependence; θ ∈ (-1,∞)\{0}
- 🔲 **Gumbel copula** — upper tail dependence; θ ≥ 1
- 🔲 **Frank copula** — symmetric tail dependence; θ ∈ ℝ\{0}
- 🔲 **Joe copula** — strong upper tail dependence
- 🔲 **Vine copulas (C-vine, D-vine, R-vine)** — pair copula constructions
- 🔲 **Empirical copula** — rank-based, nonparametric

## 2.7 Extreme Value Theory
- 🔲 **Block maxima method** — fit GEV to annual/period maxima
  - Sufficient stats for GEV MLE: sorted block maxima
- 🔲 **POT (Peaks Over Threshold)** — fit GPD to exceedances
  - Parameters: threshold u (or fraction exceedances), GPD (σ, ξ)
- 🔲 **Return level estimation** — x_T = quantile at return period T
- 🔲 **Mean residual life plot** — diagnostic for threshold choice
- 🔲 **Hill estimator** — tail index for Pareto-like tails
- 🔲 **Pickands estimator** — GPD shape from order statistics

## 2.8 Bayesian Inference — Extended
Already have: MCMC Metropolis-Hastings, Bayesian linear regression, ESS, R-hat

- 🔲 **Gibbs sampler** — coordinate-wise conditional sampling
  - Decomposition: accumulate(coordinates, conditional_draw, state_update)
  
- 🔲 **HMC (Hamiltonian Monte Carlo)** — leapfrog proposals, gradient-based
  - Parameters: L (leapfrog steps), ε (step size), mass matrix M
  - Requires: gradient of log-posterior

- 🔲 **NUTS (No-U-Turn Sampler)** — adaptive HMC; industry standard (Stan)
  - Parameters: δ (target acceptance), adapt_delta, max_depth

- 🔲 **Slice sampling** — univariate; bracket and shrink
- 🔲 **SMC (Sequential Monte Carlo)** — particle filters, population MCMC
  - Decomposition: accumulate(particles, likelihood_weight, resample_gather)

- 🔲 **Variational inference (mean field ELBO)** — optimize lower bound
- 🔲 **Laplace approximation** — Gaussian around mode
- 🔲 **Empirical Bayes** — marginal likelihood maximization
- 🔲 **Conjugate update rules** — Normal-Normal, Gamma-Poisson, Beta-Binomial, Dir-Categorical
- 🔲 **BMA (Bayesian Model Averaging)**
- 🔲 **Bayes factors** — marginal likelihood ratio

## 2.8b Multivariate Statistics
Already have (`multivariate.rs`): Hotelling T² (one/two-sample), MANOVA, LDA, CCA, Mardia multivariate normality.

- ✅ **Hotelling T² (one-sample)** — `hotelling_one_sample` in `multivariate.rs`
- ✅ **Hotelling T² (two-sample)** — `hotelling_two_sample` in `multivariate.rs`
- ✅ **MANOVA** — `manova` in `multivariate.rs`
- ✅ **LDA (Linear Discriminant Analysis)** — `lda` in `multivariate.rs`
- ✅ **CCA (Canonical Correlation Analysis)** — `cca` in `multivariate.rs`
- ✅ **Mardia multivariate normality test** — `mardia_normality` in `multivariate.rs`
- 🔲 **Pillai/Wilks/Lawley-Hotelling/Roy traces** — MANOVA test statistics
- 🔲 **Box's M test** — equality of covariance matrices
- 🔲 **Mahalanobis distance outlier test**
- 🔲 **Profile analysis** — repeated measures MANOVA variant
- 🔲 **Partial correlations / semi-partial correlations**
- ✅ **COPA (Centered Outer Product Accumulate)** — `copa.rs`: one-pass streaming covariance; `copa_from_data`, `copa_pca`

## 2.9 Resampling
Already have: bootstrap_percentile, permutation_test

- 🔲 **Bootstrap CI methods** — basic, percentile, BCa (bias-corrected accelerated), studentized
  - Decomposition: accumulate(bootstrap_samples, statistic, collect) → CI_extract
  - Parameters: n_bootstrap, ci_level, method

- 🔲 **Jackknife** — leave-one-out bias/variance estimation
  - Decomposition: accumulate(data_minus_i, statistic, collect)
  - Sufficient stats: depends on statistic; often recomputable from n, sum, sum_sq

- 🔲 **Block bootstrap** — for time series (preserves autocorrelation)
  - Parameters: block_length (or optimal via Politis-Romano)

- 🔲 **Circular block bootstrap** — wraps-around for stationarity
- 🔲 **Parametric bootstrap** — simulate from fitted model

---

# PART 3 — TIME SERIES

## 3.1 Classical
Already have: AR, differencing, SES, Holt linear, ADF, ACF, PACF

- 🔲 **MA(q) fit** — moving average model; invertibility condition
- 🔲 **ARMA(p,q)** — autoregressive moving average; Yule-Walker, innovations algorithm
  - Sufficient stats: ACVF at lags 0..max(p,q)

- 🔲 **ARIMA(p,d,q)** — differencing + ARMA; Box-Jenkins methodology
  - Parameters: (p,d,q), include_constant, method (MLE, CSS)

- 🔲 **SARIMA(p,d,q)(P,D,Q)_s** — seasonal ARIMA
- 🔲 **Holt-Winters (triple exponential smoothing)** — trend + seasonality
  - Parameters: α (level), β (trend), γ (seasonal), seasonal_type (add/mul)

- 🔲 **STL decomposition** — Season-Trend using LOESS; robust
  - Parameters: s_window, t_window, l_window, n_inner, n_outer

- 🔲 **X-13ARIMA-SEATS style seasonal adjustment**

## 3.2 State Space Models
- 🔲 **Kalman filter** — linear Gaussian; prediction + update
  - Decomposition: accumulate(observations, kalman_update, state_posterior)
  - Parameters: F (transition), H (observation), Q (process noise), R (obs noise), x0, P0
  - Sufficient stats: (x_t, P_t) posterior mean and covariance

- 🔲 **Kalman smoother (RTS)** — backward pass, smoothed states
- 🔲 **EM for state space** — estimate F, H, Q, R from data
- 🔲 **Structural time series** — local level, local linear trend, seasonal components
- 🔲 **EKF (Extended Kalman Filter)** — nonlinear systems, Jacobian linearization
- 🔲 **UKF (Unscented Kalman Filter)** — sigma-point based, better nonlinear approx
- 🔲 **Particle filter** — sequential Monte Carlo for nonlinear/non-Gaussian
  - Parameters: n_particles, resampling_threshold

## 3.3 Multivariate Time Series
- 🔲 **VAR(p)** — vector autoregression; OLS equation-by-equation
  - Decomposition: accumulate(rows_and_lags, xtx_contribution, sum_matrix)
  - Sufficient stats: block XᵀX and Xᵀy matrices

- 🔲 **VECM** — vector error correction; cointegrated VAR
- 🔲 **Granger causality test** — F-test on VAR restrictions (already noted)
- 🔲 **Cointegration (Engle-Granger)** — OLS residuals + ADF test
- 🔲 **Johansen cointegration** — trace/max eigenvalue tests; VECM rank
- 🔲 **SVAR** — structural VAR; identification via Cholesky/long-run restrictions

## 3.4 GARCH Family
Already have: GARCH(1,1), EWMA

- 🔲 **GARCH(p,q)** — general lag orders
- 🔲 **EGARCH** — exponential GARCH; asymmetric leverage
- 🔲 **GJR-GARCH (TGARCH)** — threshold asymmetry
- 🔲 **FIGARCH** — fractionally integrated GARCH; long memory
- 🔲 **GARCH-M** — GARCH in mean; risk premium
- 🔲 **DCC-GARCH** — dynamic conditional correlation; multivariate
- 🔲 **Realized GARCH** — intraday realized measures
- 🔲 **HAR-RV** — Heterogeneous AR for realized variance; daily+weekly+monthly

## 3.5 Spectral Time Series
Already have: Lomb-Scargle, cross-spectral, multitaper, spectral entropy, band power

- 🔲 **Coherence** — |Cxy(f)|² / (Pxx(f)·Pyy(f)); frequency-domain correlation
- 🔲 **Phase spectrum** — arg(Cxy(f))
- 🔲 **Granger spectral causality** — Geweke decomposition
- 🔲 **EMD (Empirical Mode Decomposition)** — data-adaptive basis, no a priori
  - Produces IMFs (Intrinsic Mode Functions); HHT follows
- 🔲 **EEMD/CEEMDAN** — noise-assisted EMD variants
- 🔲 **SSA (Singular Spectrum Analysis)** — trajectory matrix + SVD → components

## 3.6 Change Point Detection
- 🔲 **CUSUM** — cumulative sum; sequential detection
  - Decomposition: accumulate(residuals, cusum_update, max_track)
  
- 🔲 **Bayesian change point (BOCPD)** — online; hazard function
- 🔲 **PELT** — pruned exact linear time; penalized likelihood
- 🔲 **Binary segmentation** — greedy; O(n log n)
- 🔲 **Wild binary segmentation** — random intervals variant
- 🔲 **e-divisive** — energy statistic change point

## 3.7 Stochastic Processes
Already have (`stochastic.rs`): extensive — Brownian motion, GBM, Black-Scholes, OU, Poisson (homogeneous+non), Markov chains (DTMC+CTMC), birth-death, queues, random walks, Itô/Stratonovich integrals.

- ✅ **Brownian motion (Wiener process)** — `brownian_motion`, `brownian_bridge`, `quadratic_variation` in `stochastic.rs`
- ✅ **Geometric Brownian Motion (GBM)** — `geometric_brownian_motion`, `gbm_expected`, `gbm_variance`
  - Parameters: μ (drift), σ (volatility), S0, T, n_steps
  - Decomposition: accumulate(time_steps, gbm_step, path)

- ✅ **Ornstein-Uhlenbeck process** — `ornstein_uhlenbeck`, `ou_stationary_variance`, `ou_autocorrelation`
  - Parameters: θ (mean reversion), μ (long-run mean), σ
  - Sufficient stats for MLE: n, Σx_{t}, Σx_{t-1}, Σx_t x_{t-1}, Σx_t²

- 🔲 **Cox-Ingersoll-Ross (CIR)** — positive, mean-reverting; interest rates
- 🔲 **Heston model** — stochastic volatility; correlated BMs
- 🔲 **Variance Gamma process** — Brownian subordinated to Gamma time
- 🔲 **CGMY / Tempered stable** — generalized exponential tilted stable
- 🔲 **Lévy processes** — general; infinite divisibility
- 🔲 **Hawkes process** — self-exciting point process; λ(t) = μ + Σ φ(t-tᵢ)
  - Sufficient stats: branching ratio, baseline intensity
  - Decomposition: accumulate(past_events, hawkes_kernel, intensity)
  
- ✅ **Poisson process** — `poisson_process`, `nonhomogeneous_poisson`, `poisson_count`, `poisson_expected_count`
- ✅ **Markov chains (DTMC)** — `markov_n_step`, `stationary_distribution`, `mean_first_passage_time`, `is_ergodic`, `mixing_time`
- ✅ **CTMC** — `ctmc_transition_matrix`, `ctmc_stationary`, `ctmc_holding_time`
- ✅ **Birth-death process** — `birth_death_stationary`; M/M/1 queue `mm1_queue`; Erlang-C `erlang_c`
- ✅ **Random walk** — `simple_random_walk`, `first_passage_time_cdf`, `return_probability_1d`, `rw_expected_maximum`
- ✅ **Itô/Stratonovich integrals** — `ito_integral`, `stratonovich_integral`, `ito_lemma_verification`
- 🔲 **Renewal process** — interarrival distribution; Wald's identity
- 🔲 **SDE solvers (Euler-Maruyama, Milstein)** — general framework; currently only GBM and OU have custom paths
- 🔷 **Black-Scholes** — `black_scholes` in `stochastic.rs`; returns (price, delta); missing gamma/vega/theta/rho

---

# PART 4 — SPATIAL STATISTICS

Already have: variogram models, ordinary kriging, Moran's I, Geary's C, Ripley's K/L, Clark-Evans, NN distances

- 🔲 **Universal kriging** — drift included in mean model
- 🔲 **Simple kriging** — known mean; Bayesian interpretation
- 🔲 **Co-kriging** — multivariate; cross-variogram
- 🔲 **Sequential simulation** — sequential Gaussian simulation (SGS)
- 🔲 **Variogram fitting** — WLS to empirical variogram; anisotropy
- 🔲 **Local Moran's I** — LISA statistics; spatial clusters/outliers
- 🔲 **Getis-Ord G*** — local hotspot statistic
- 🔲 **Spatial scan statistic (Kulldorff)** — disease cluster detection
- 🔲 **GWR (Geographically Weighted Regression)** — spatially varying coefficients
  - Parameters: bandwidth, kernel (gaussian/bisquare), adaptive (bool)
  - Decomposition: for each location, accumulate(nearby, weighted_OLS, solve)

- 🔲 **Spatial regression** — spatial lag (SAR), spatial error (SEM) models
- 🔲 **Areal data models** — CAR, SAR priors for lattice data
- 🔲 **Spatial point process models** — Cox process, log-Gaussian Cox
- 🔲 **Quadrat analysis** — variance-to-mean ratio for regularity/clustering
- 🔲 **F-function** — nearest neighbor CDF; complement of G-function
- 🔲 **Pair correlation function (PCF)** — g(r) = K'(r) / (2πr)

---

# PART 5 — SIGNAL PROCESSING

## 5.1 Already Implemented
FFT/IFFT/RFFT, FFT2D, windows (6 types), periodogram, Welch, STFT, spectrogram, convolution, cross-correlation, autocorrelation, DCT2/3, FIR filters (lowpass/highpass/bandpass), IIR Biquad/Butterworth, moving average, EMA, Savitzky-Golay, Hilbert, envelope, instantaneous frequency, cepstrum, Haar DWT/IDWT/wavedec

## 5.2 Wavelet Families (Full Filter Bank)
- 🔲 **Daubechies db2-db20** — orthogonal, compact support
- 🔲 **Symlets sy2-sy20** — near-symmetric Daubechies
- 🔲 **Coiflets coif1-coif5** — wavelets AND scaling functions with vanishing moments
- 🔲 **Biorthogonal bior families** — linear phase, non-orthogonal
- 🔲 **Meyer wavelet** — infinitely differentiable, frequency domain
- 🔲 **Morlet (complex)** — Gabor modulated Gaussian; complex CWT
- 🔲 **Mexican hat (Ricker)** — 2nd derivative of Gaussian; CWT
- 🔲 **CWT (Continuous Wavelet Transform)** — with arbitrary scales and mother wavelet
- 🔲 **WPD (Wavelet Packet Decomposition)** — full subband tree
- 🔲 **SWT (Stationary Wavelet Transform)** — undecimated DWT

## 5.3 Advanced Spectral
- 🔲 **MUSIC algorithm** — multiple signal classification; super-resolution frequency
  - Parameters: signal_subspace_dim, n_freqs
  - Decomposition: accumulate(covariance_outer, sum_matrix) → SVD → gather(null_space)

- 🔲 **ESPRIT** — rotational invariance for frequency estimation
- 🔲 **Capon beamformer** — minimum variance distortionless response
- 🔲 **WOSA (Welch's Overlapped Segment Averaging)** — already have Welch
- 🔲 **Thomson multitaper** — already have; ensure dpss tapers correct
- 🔲 **S-transform** — time-frequency, phase reference; STFT with Gaussian width ∝ 1/f

## 5.4 Compressed Sensing
- 🔲 **Basis pursuit (L1 minimization)** — min ‖x‖₁ s.t. Ax = b
  - Solved via LP or ADMM
  
- 🔲 **OMP (Orthogonal Matching Pursuit)** — greedy sparse recovery
  - Decomposition: accumulate(iterations, max_correlation_gather, residual_update)
  - Parameters: sparsity k or stopping criterion

- 🔲 **LASSO for signals** — min ½‖Ax-b‖² + λ‖x‖₁; proximal gradient
- 🔲 **ADMM** — alternating direction method of multipliers; general splitting
  - Parameters: ρ (augmented Lagrangian), max_iter, tol

- 🔲 **CoSaMP** — compressive sampling matching pursuit
- 🔲 **Subspace pursuit** — similar to CoSaMP, cleaner convergence

## 5.5 Audio/Music
- 🔲 **MFCC** — mel-frequency cepstral coefficients; speech/music feature
  - Parameters: n_mfcc, n_fft, hop_length, n_mels, fmin, fmax
  - Decomposition: STFT → mel_filterbank_gather → log → DCT → gather(n_mfcc)

- 🔲 **Mel spectrogram** — STFT → mel filterbank projection
- 🔲 **Chromagram** — pitch class energy; 12 bins
- 🔲 **Constant-Q Transform (CQT)** — geometrically spaced frequency bins
- ✅ **Zero crossing rate** — `signal_processing.rs::zero_crossing_rate`
- 🔲 **Spectral rolloff, flatness, centroid** — timbral features

---

# PART 6 — MACHINE LEARNING MATH

## 6.1 Kernels (All Types)
- 🔲 **RBF (Gaussian)** — exp(-‖x-y‖²/(2σ²)); universal kernel
- 🔲 **Laplacian** — exp(-‖x-y‖/σ)
- 🔲 **Polynomial** — (xᵀy + c)^d
- 🔲 **Sigmoid** — tanh(α xᵀy + c)
- 🔲 **Matérn** — ν parameter family; ν=1/2 = Laplacian, ν=3/2, ν=5/2, ν→∞ = RBF
  - Parameters: ν, length_scale
  
- 🔲 **Periodic** — exp(-2sin²(π‖x-y‖/p)/ℓ²)
  - Parameters: period p, length_scale ℓ

- 🔲 **Rational Quadratic** — mixture of RBF at different length scales; α parameter
- 🔲 **Spectral mixture kernel** — sum of Gaussian spectral components
- 🔲 **Kernel composition** — sum, product, scaling
- 🔲 **String kernels** — for sequences

## 6.2 Gaussian Processes
- 🔲 **GP regression (exact)** — posterior = N(μ*, Σ*); O(n³) training
  - Decomposition: accumulate(data_pairs, K_row, cov_matrix) → Cholesky solve → gather(test_predictions)
  - Sufficient stats: (K + σ²I)⁻¹ y (alpha vector) + Cholesky factor L

- 🔲 **GP hyperparameter optimization** — marginal likelihood maximization via gradient
- 🔲 **Sparse GP** — Nyström/inducing points, O(nm²) where m << n
  - Methods: FITC, VFE, SVGP

- 🔲 **GP classification** — Laplace/EP approximation
- 🔲 **Multi-output GP** — intrinsic co-regionalization model

## 6.3 Support Vector Machines
- 🔲 **SVM primal (hard margin)** — min ½‖w‖² s.t. yᵢ(wᵀxᵢ+b) ≥ 1
- 🔲 **SVM soft margin** — C-SVM; Lagrangian dual
  - Parameters: C (slack), kernel, ε (SVR tube width)
  
- 🔲 **SMO algorithm** — sequential minimal optimization; standard SVM solver
  - Decomposition: accumulate(pairs, smo_update, alpha_coeffs)

- 🔲 **SVR** — support vector regression; ε-insensitive loss
- 🔲 **One-class SVM** — novelty detection
- 🔲 **SVDD** — support vector data description (sphere around data)

## 6.4 Decision Trees & Ensembles
- 🔲 **CART** — classification and regression trees; Gini/entropy/MSE split criteria
  - Decomposition: for each candidate split: accumulate(sorted_feature, gini_left+right, min_impurity)

- 🔲 **Random forests** — bagged CART + feature subsampling
  - Parameters: n_estimators, max_features, max_depth, min_samples_split

- 🔲 **Gradient boosting (GBDT)** — additive tree model; Newton step
  - Parameters: learning_rate, n_estimators, max_depth, subsample, colsample

- 🔲 **AdaBoost** — exponential loss; reweighted data
- 🔲 **Bagging** — bootstrap aggregation; general ensemble
- 🔲 **Isolation Forest** — anomaly detection via random partitioning
- 🔲 **Extra Trees** — extremely randomized; random splits

## 6.5 Neural Network Math
Already have: activations, some forward/backward

- ✅ **Batch normalization** — μ, σ from batch; γ, β learned (`neural.rs::batch_norm`)
  - Sufficient stats: batch mean, batch variance
  - Decomposition: accumulate(batch, mean_var, normalize) → gather(scale+shift)

- ✅ **Layer normalization** — across feature dim; no batch dep (`neural.rs::layer_norm`)
- ✅ **Group normalization** — groups of channels (`neural.rs::group_norm`)
- ✅ **Instance normalization** — per sample per channel (`neural.rs::instance_norm`)
- ✅ **Dropout** — Bernoulli mask; effective ensemble of 2^n networks (`neural.rs::dropout`)
- ✅ **Attention mechanism** — Q(KᵀV)/√d_k; softmax weights (`neural.rs::scaled_dot_product_attention`)
  - Decomposition: accumulate(keys, qk_dot, sum) → softmax_gather → accumulate(values, attn_weight, weighted_sum)
  - Parameters: d_k, heads, dropout

- ✅ **Multi-head attention** — parallel attention with projection (`neural.rs::multi_head_attention`)
- ✅ **Positional encoding** — sin/cos and RoPE (`neural.rs::positional_encoding`, `rope`)
- 🔲 **LSTM gates** — i, f, g, o gates; cell state; backprop
- 🔲 **GRU** — reset/update gates; simpler than LSTM
- 🔲 **Backpropagation through time (BPTT)** — gradient through sequence
- ✅ **Softmax** — exp(x) / Σexp(x); log-sum-exp trick for stability (`neural.rs::softmax`)
- ✅ **Cross-entropy loss** — -Σ yᵢ log(p̂ᵢ) (`neural.rs::cross_entropy_loss`)
- ✅ **Hinge loss** — max(0, 1 - y·ŷ); SVM connection (`neural.rs::hinge_loss`)
- ✅ **Focal loss** — down-weight easy examples; object detection (`neural.rs::focal_loss`)
- 🔲 **Contrastive loss** — metric learning; InfoNCE
- ✅ **Convolutional layer math** — `neural.rs::conv1d`, `conv2d`, `conv2d_transpose`
- 🔲 **Depthwise separable convolution** — factored convolution
- 🔲 **Weight initialization** — Glorot/Xavier, He, orthogonal
  - Parameters: gain, mode (fan_in/fan_out/fan_avg)

## 6.6 Optimization (ML-specific)
Already have: SGD (`gradient_descent`), Adam, AdaGrad, RMSprop, L-BFGS (`optimization.rs`)

- 🔲 **SGD with momentum** — v = γv + α∇, θ = θ - v
- 🔲 **Nesterov momentum** — lookahead gradient
- 🔲 **AdamW** — Adam + decoupled weight decay
  - Parameters: lr, β₁, β₂, ε, weight_decay

- 🔲 **LAMB** — layer-wise adaptive moments for large batch
- 🔲 **Lion** — sign-based momentum optimizer
- 🔲 **Learning rate schedulers** — cosine annealing, OneCycle, warmup, step decay
- ✅ **Gradient clipping** — by norm and by value (`neural.rs::clip_grad_norm`, `clip_grad_value`)
- 🔲 **Proximal gradient** — for L1 + smooth; ISTA/FISTA
  - FISTA: accumulate(iterations, gradient_step + prox_gather, momentum_update)

- 🔲 **Frank-Wolfe (conditional gradient)** — projection-free; over simplex/nuclear norm ball
- 🔲 **Mirror descent** — generalized gradient; Bregman divergence

## 6.7 Dimension Reduction (Additional)
Already have: PCA, MDS, t-SNE, NMF, LDA

- 🔲 **UMAP** — uniform manifold approximation; faster, better structure than t-SNE
  - Parameters: n_neighbors, min_dist, metric, n_components

- 🔲 **ISOMAP** — geodesic distances + MDS
  - Decomposition: accumulate(knn_graph, floyd_warshall, dist_matrix) → MDS

- 🔲 **Locally Linear Embedding (LLE)** — local linear reconstruction
- 🔲 **Laplacian Eigenmaps** — spectral embedding via graph Laplacian
  - Decomposition: accumulate(neighbor_pairs, similarity_weight, sum) → SVD

- 🔲 **Kernel PCA** — nonlinear PCA via kernel trick
- 🔲 **Autoencoders** — bottleneck network; reconstruction loss
- 🔲 **Variational Autoencoder (VAE)** — KL(q(z|x) || p(z)) + reconstruction
- 🔲 **Diffusion maps** — random walk on data manifold; diffusion distance
- 🔲 **SRP (Sparse Random Projections)** — Johnson-Lindenstrauss approx
- 🔲 **FastICA** — independent component analysis; kurtosis/negentropy contrast
  - Decomposition: accumulate(data, g_prime_expectations, update_W) → orthogonalize

## 6.8 Clustering (Additional)
Already have: DBSCAN (`clustering.rs`), discovery with GPU

- 🔷 **k-means** — Lloyd's algorithm, `kmeans.rs` exists but uses deterministic evenly-spaced init (NOT k-means++); convergence by label stability
  - Decomposition: accumulate(points, nearest_centroid_gather, cluster_sum+count) → centroid_update
  - Parameters: k, init (k-means++, random), max_iter, tol
  - Gap: k-means++ initialization missing; tests use `println!` not `assert!`

- 🔲 **k-medoids (PAM)** — actual data points as centers; robust to outliers
- ✅ **Gaussian Mixture Models (GMM)** — EM algorithm; soft assignments (`mixture.rs`)
  - Sufficient stats: Σ(γᵢ), Σ(γᵢxᵢ), Σ(γᵢxᵢxᵢᵀ) per component
  - Decomposition: E-step: accumulate(points, responsibility, sum_per_cluster); M-step: gather(params)

- 🔲 **HDBSCAN** — hierarchical DBSCAN; condensed tree; robust k selection
- 🔲 **Spectral clustering** — graph Laplacian eigenvectors + k-means
  - Decomposition: accumulate(similarity_pairs, laplacian, sum_matrix) → SVD → k-means

- 🔲 **Agglomerative hierarchical** — linkage (single, complete, average, Ward)
  - Ward: accumulate(merge_pairs, ward_distance, priority_queue_gather)

- 🔲 **Mean shift** — kernel density gradient ascent; automatic k
- 🔲 **OPTICS** — ordering points to identify clustering structure
- 🔲 **Affinity propagation** — message passing; no k needed
- 🔲 **LVQ (Learning Vector Quantization)** — competitive learning

## 6.9 Reinforcement Learning Math
- 🔲 **Value function** — V(s) = E[Σγᵗr_t]
- 🔲 **Bellman equation** — V(s) = R(s) + γ Σ P(s'|s,a) V(s')
- 🔲 **Value iteration** — dynamic programming
  - Decomposition: accumulate(next_states, bellman_update, max_gather)

- 🔲 **Policy iteration** — evaluate then improve
- 🔲 **Q-learning** — off-policy TD; Q(s,a) ← Q + α(r + γ maxQ' - Q)
- 🔲 **TD(λ)** — eligibility traces; unifies TD(0) and MC
- 🔲 **REINFORCE** — policy gradient; G_t ∇log π(a|s)
- 🔲 **Actor-critic** — value baseline for REINFORCE
- 🔲 **PPO** — clipped surrogate objective
- 🔲 **Bandit algorithms** — UCB, Thompson sampling, ε-greedy, EXP3

## 6.10 Graph Neural Network Math
- 🔲 **Graph Laplacian** — L = D - A; normalized Laplacian
  - Decomposition: accumulate(edge_list, degree+adj, combine)

- 🔲 **Spectral GCN** — Chebyshev polynomial approximation of convolution
- 🔲 **Message passing** — aggregate(neighbors) → update(self + message)
  - Decomposition: accumulate(neighbor_messages, sum/mean/max, aggregate) → gather(self_state)

- 🔲 **Node2Vec** — random walk + skip-gram embeddings
- 🔲 **DeepWalk** — uniform random walk embeddings
- 🔲 **GraphSAGE** — inductive, sample-and-aggregate

---

# PART 7 — PURE MATHEMATICS

## 7.1 Number Theory
Already have (`number_theory.rs`): extensive — see below. Also `bigint.rs` has BigInt/U256 with GCD; `bigfloat.rs` has Bernoulli.

- ✅ **GCD/LCM** — `gcd`, `lcm` in `number_theory.rs` (u64); also `BigInt::gcd`, `U256::gcd` in `bigint.rs`
- ✅ **Extended Euclidean** — `extended_gcd` + `mod_inverse` in `number_theory.rs`
  - Decomposition: accumulate(division_steps, bezout_update, carry)

- ✅ **Modular exponentiation** — `mod_pow` in `number_theory.rs`; `mul_mod` for overflow-safe u64
- ✅ **CRT (Chinese Remainder Theorem)** — `crt` in `number_theory.rs`
- ✅ **Euler's totient φ(n)** — `euler_totient` + `sieve_totients` in `number_theory.rs`
- ✅ **Möbius function μ(n)** — `mobius` in `number_theory.rs`
- ✅ **Sieve of Eratosthenes** — `sieve` + `segmented_sieve` in `number_theory.rs`
  - Decomposition: accumulate(odd_candidates, mark_composites, boolean_gather)

- 🔲 **Sieve of Atkin** — faster for large N
- ✅ **Trial division** — inside `factorize` in `number_theory.rs`
- ✅ **Pollard's rho** — `pollard_rho` + `factorize_complete` in `number_theory.rs`
- ✅ **Miller-Rabin primality** — `is_prime` in `number_theory.rs` (deterministic for n < 3.3e24)
- 🔲 **Lucas-Lehmer primality** — for Mersenne numbers
- 🔲 **AKS primality** — polynomial-time deterministic (mostly of theoretical interest)
- 🔲 **Quadratic sieve** — sub-exponential factorization
- ✅ **Legendre/Jacobi symbols** — `legendre`, `jacobi` in `number_theory.rs`; Kronecker 🔲
- ✅ **Quadratic residues (Tonelli-Shanks)** — `sqrt_mod` in `number_theory.rs`
- ✅ **Primitive roots** — `primitive_root` in `number_theory.rs`
- ✅ **Discrete logarithm (BSGS)** — `discrete_log` in `number_theory.rs`; Pohlig-Hellman 🔲
- ✅ **Continued fractions** — `continued_fraction`, `convergents`, `best_rational`, `cf_period`
- 🔲 **Farey sequences and Stern-Brocot tree**
- ✅ **Partition function p(n)** — `partition_count` in `number_theory.rs` (Euler recurrence)
- ✅ **Bernoulli numbers** — `BigFloat::bernoulli` in `bigfloat.rs`
- ✅ **p-adic numbers** — `multi_adic.rs`: valuation, distance, norm, digits
- ✅ **Divisor functions** — `num_divisors`, `sum_divisors`, `divisors`, `factorize`
- ✅ **Sum of two squares** — `sum_of_two_squares`; Pell's equation `pell_fundamental`
- ✅ **Euler product / zeta** — `euler_product_approx`, `basel_sum_exact`

## 7.2 Abstract Algebra
- 🔲 **Polynomial arithmetic** — addition, multiplication, GCD (Euclidean), evaluation
  - Decomposition: accumulate(coefficient_pairs, poly_multiply_term, collect_terms)

- 🔲 **Polynomial factorization** — over Q, Z_p; Berlekamp, Cantor-Zassenhaus
- 🔲 **Finite fields GF(p^n)** — irreducible polynomial, field operations
- 🔲 **GF(2) arithmetic** — XOR-based; used in coding theory, cryptography
- 🔲 **Ring operations** — Z_n with general n; quotient rings
- 🔲 **Group representations** — character table, orthogonality relations
- 🔲 **Symmetric group S_n** — permutation operations, cycle notation, transpositions
- 🔲 **Lattice theory** — meet, join, Hasse diagram
- Already have: algebraic structures in proof.rs (semigroup, monoid, group, lattice)

## 7.3 Linear Algebra (Advanced)
Already have extensive coverage; gaps:

- 🔲 **Matrix functions** — expm, sqrtm, logm (noted above)
- 🔲 **Tensor operations** — outer products, mode-n products, tensor contractions
  - Decomposition: accumulate(index_tuples, product_contribution, sum_by_output_index)

- 🔲 **Tucker decomposition** — core tensor + factor matrices; truncated via HOSVD
- 🔲 **CP/PARAFAC decomposition** — sum of rank-1 tensors; ALS algorithm
  - Decomposition: ALS: accumulate(mode_unfold, factor_khatri_rao, solve) per mode

- 🔲 **Tensor train (TT/MPS)** — matrix product state; quantum-inspired
- 🔲 **Kronecker product / vec / unvec** — vectorization tricks
- 🔲 **Khatri-Rao product** — column-wise Kronecker; used in tensor decomp
- 🔲 **Randomized SVD** — count sketch + SVD; O(mn log k) instead of O(mn min(m,n))
  - Parameters: rank k, n_oversampling, n_power_iter

- 🔲 **Block matrix operations** — Schur complement, block inversion lemma
- 🔲 **Sylvester equation** — AX + XB = C; Bartels-Stewart algorithm
- 🔲 **Lyapunov equation** — AX + XAᵀ = Q; continuous Lyapunov
- 🔲 **Riccati equation** — CARE: AᵀX + XA - XBR⁻¹BᵀX + Q = 0; control theory

## 7.4 Analysis & Measure Theory
- 🔲 **Fourier series** — coefficients, convergence, Gibbs phenomenon, Parseval
  - Decomposition: accumulate(signal_samples, exp_basis, sum) / N

- 🔲 **Fourier transform properties** — shift, scale, convolution theorem, Parseval
- 🔲 **Laplace transform** — analytic; for ODE/control theory
- 🔲 **Z-transform** — discrete-time; system analysis
- 🔲 **Mellin transform** — multiplicative convolution; distribution of products
- 🔲 **Wavelet transform theory** — admissibility condition, CWT inversion formula
- 🔲 **Functional analysis basics** — norms, inner products, completeness, Hilbert spaces
- 🔲 **Operator theory** — bounded operators, spectrum, resolvent
- 🔲 **Sobolev spaces** — Hᵏ norms; relevant for PDE numerical analysis

## 7.5 Topology & Geometry
Already have: TDA (Rips complex, persistence diagrams)

- 🔲 **Čech complex** — dual to Rips; nerve theorem
- 🔲 **Alpha complex** — intersection of balls; Delaunay triangulation based
- 🔲 **Persistent homology (general)** — boundary matrices, reduction algorithm
- 🔲 **Cubical homology** — for image data
- 🔲 **Mapper algorithm** — topological data analysis via cover+clustering
- 🔲 **Čech cohomology** — cohomological persistence
- 🔲 **Discrete Morse theory** — gradient vector fields on simplicial complexes
- 🔲 **Euler characteristic** — χ = V - E + F; topological invariant
- 🔲 **Riemannian geometry** — metric tensor, geodesics, curvature, parallel transport
  - Parameters: manifold type, metric
  - Already have manifold types in manifold.rs; extend

- 🔲 **Lie groups** — matrix groups (SO(n), SU(n), SE(n)); exponential map, log map
- 🔲 **Convex geometry** — convex hull, halfspace intersection
- 🔲 **Voronoi diagrams** — Fortune's algorithm O(n log n)
- 🔲 **Delaunay triangulation** — dual of Voronoi
- 🔲 **Computational geometry** — point-in-polygon, line intersection, convex hull 2D

## 7.6 Graph Theory (Advanced)
Already have: BFS, DFS, toposort, components, Dijkstra, Bellman-Ford, Floyd-Warshall, Kruskal, Prim, centralities, PageRank, modularity, max flow, diameter, density, clustering coefficient

- 🔲 **A* search** — heuristic-guided shortest path
- 🔲 **Bidirectional Dijkstra** — faster shortest path for single pair
- 🔲 **Johnson's algorithm** — all-pairs reweighted for sparse graphs
- 🔲 **Strongly connected components** — Tarjan's, Kosaraju's
- 🔲 **Bridges and articulation points** — Tarjan's bridge-finding
- 🔲 **Bipartite matching (Hopcroft-Karp)** — maximum matching O(√V · E)
- 🔲 **Hungarian algorithm** — minimum weight bipartite matching O(n³)
- 🔲 **Network flow (Dinic's)** — O(V²E) max flow
- 🔲 **Push-relabel** — O(V²√E) max flow
- 🔲 **Min-cost max-flow** — MCMF; successive shortest paths
- 🔲 **Graph coloring** — greedy, chromatic polynomial (small graphs)
- 🔲 **Graph isomorphism** — Weisfeiler-Leman test; VF2 algorithm
- 🔲 **Planarity testing** — Kuratowski/LR-planarity O(n)
- 🔲 **Tree decomposition** — treewidth; dynamic programming on tree
- 🔲 **Betweenness centrality** — Brandes algorithm O(VE)
- 🔲 **Eigenvector centrality** — power iteration (already have)
- 🔲 **HITS (Hubs and Authorities)** — Kleinberg's algorithm
- 🔲 **Lovász theta** — SDP relaxation of clique/chromatic number
- 🔲 **Spectral graph theory** — Cheeger inequality, expander mixing lemma
  - Already have spectral_gap.rs; ensure Cheeger bound computation

---

# PART 8 — PHYSICS ENGINES

## 8.1 Classical Mechanics
- 🔲 **Lagrangian mechanics** — generalized coordinates, Euler-Lagrange equations
- 🔲 **Hamiltonian mechanics** — phase space, Hamilton's equations
- 🔲 **Poisson brackets** — {f,g} = Σ(∂f/∂qᵢ ∂g/∂pᵢ - ∂f/∂pᵢ ∂g/∂qᵢ)
- 🔲 **Action-angle variables** — integrable systems
- 🔷 **Rigid body** — `physics.rs::euler_rotation`, `rotational_kinetic_energy` (basic rotation); inertia tensor/Euler angles/quaternion missing
  - Decomposition: accumulate(mass_elements, r×(r×), moment_matrix)
  
- ✅ **N-body gravity** — direct O(n²) (`physics.rs::nbody_gravity`)
  - Decomposition (direct): accumulate(particle_pairs, force_contribution, sum)
  - Note: Barnes-Hut O(n log n) missing

- 🔲 **Projectile motion** — analytic under gravity + drag
- ✅ **Orbital mechanics** — `physics.rs::kepler_orbit`, `vis_viva`, `KeplerOrbit`; double pendulum also (`double_pendulum_rk4`), SHO (`sho_exact`, `sho_energy`, `dho_underdamped`)

## 8.2 Statistical Mechanics
- ✅ **Partition function** — `physics.rs::partition_function` (canonical ensemble, Z = Σ exp(-βEᵢ))
  - Decomposition: accumulate(states, exp_neg_beta_E, sum)

- ✅ **Thermodynamic averages** — `physics.rs::mean_energy`, `heat_capacity_canonical`, `boltzmann_probabilities`, `gibbs_entropy`, `helmholtz_free_energy`
  - Decomposition: accumulate(states, A·exp_weight, sum) / Z

- ✅ **Ising model** — exact 1D (`physics.rs::ising1d_exact`) + 2D Metropolis MC (`ising2d_metropolis`)
  - Decomposition: accumulate(spin_pairs, coupling_energy, sum)
  - Parameters: J (coupling), h (field), β (inverse temperature), lattice
  - Note: Wolff cluster / Wang-Landau missing

- ✅ **Monte Carlo Metropolis** — `physics.rs::ising2d_metropolis` (physics version with spin energy)
- 🔲 **Wolff cluster algorithm** — faster MC for Ising near criticality
- 🔲 **Wang-Landau** — density of states estimation; flat histogram
- 🔲 **Mean field theory** — self-consistent equations; variational free energy
- 🔲 **Transfer matrix method** — 1D chains exact solution
- 🔲 **Molecular dynamics (MD)** — Verlet/leapfrog + thermostat/barostat
  - Decomposition: accumulate(atom_pairs, LJ+Coulomb, force_sum) → integrate

- 🔲 **Lennard-Jones potential** — V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
- 🔲 **Ewald summation** — long-range Coulomb in periodic box
- 🔲 **Fast multipole method (FMM)** — O(n) long-range interactions
- 🔷 **Quantum oscillator + Bose-Einstein** — `physics.rs::qho_energy`, `bose_einstein_occupation`, `planck_spectral_energy`, `wien_displacement`, `arrhenius`, `equilibrium_constant`

## 8.3 Quantum Mechanics
- ✅ **Schrödinger equation** — 1D numerical (`physics.rs::schrodinger1d`, `sym_tridiag_eigvals`)
- ✅ **Finite difference for QM** — particle in box (`particle_in_box_energy`, `particle_in_box_wf`), QHO (`qho_energy`)
- 🔷 **Harmonic oscillator** — energy levels `qho_energy`; ladder operators/wavefunctions missing
- 🔲 **Angular momentum** — L² eigenstates; spherical harmonics Y_lm
- 🔲 **Spin operators** — Pauli matrices σₓ, σᵧ, σᵤ; commutation relations
- ✅ **Density matrix** — `physics.rs::density_matrix_trace`, `density_matrix_purity`, `von_neumann_entropy_diagonal`
- ✅ **State + time evolution** — `physics.rs::Amplitude`, `normalize_state`, `time_evolve_state`, `expectation_value`, `uncertainty`, `heisenberg_uncertainty_product`
- ✅ **Hydrogen atom** — `physics.rs::hydrogen_energy_ev`, `hydrogen_wavelength`
- ✅ **Tunneling** — `physics.rs::tunneling_transmission`
- 🔲 **Perturbation theory** — first/second order energy corrections
- 🔲 **Variational principle** — E₀ ≤ ⟨ψ|Ĥ|ψ⟩/⟨ψ|ψ⟩
- 🔲 **Tight binding model** — lattice Hamiltonian; band structure
- 🔲 **Quantum circuit simulation** — state vector, unitary gates, measurement

## 8.4 Fluid Dynamics
- 🔷 **Incompressible Navier-Stokes** — vorticity formulation (`physics.rs::vorticity_step`, `poisson_sor`); full 3D NS missing
- ✅ **Vorticity-stream function** — `physics.rs::vorticity_step`, `poisson_sor`
- 🔲 **Lid-driven cavity** — standard CFD benchmark; finite difference
- ✅ **Dimensionless numbers** — `physics.rs::reynolds_number`, `mach_number`, `prandtl_number`, `nusselt_dittus_boelter`
- ✅ **Flow equations** — `physics.rs::bernoulli_velocity`, `poiseuille_flow_rate`, `poiseuille_velocity_profile`, 1D Euler (`euler1d_lax_friedrichs`, `cfl_timestep`)
- 🔲 **Lattice Boltzmann method (LBM)** — mesoscopic; BGK collision operator
  - Decomposition: accumulate(lattice_sites, streaming_gather, collision_update)

- 🔲 **SPH (Smoothed Particle Hydrodynamics)** — meshfree; kernel-based
  - Decomposition: accumulate(neighbor_particles, kernel_contribution, sum)

## 8.5 Thermodynamics
- ✅ **Equations of state** — ideal gas (`physics.rs::ideal_gas_pressure`, `ideal_gas_temperature`, `ideal_gas_internal_energy`, `ideal_gas_entropy_change`), van der Waals (`vdw_pressure`, `vdw_critical`)
- 🔷 **Thermodynamic potentials** — Helmholtz F (`physics.rs::helmholtz_free_energy`); H, G, Legendre transforms missing
- 🔲 **Maxwell relations** — partial derivative identities
- ✅ **Heat capacity** — `physics.rs::heat_capacity_canonical` (canonical); Debye/Einstein models missing
- ✅ **Heat transfer** — `physics.rs::heat_flux_fourier`, `newton_cooling`, `stefan_boltzmann`, `carnot_efficiency`, `otto_efficiency`
- 🔲 **Phase transitions** — Clausius-Clapeyron; order parameter
- 🔲 **Entropy production** — second law; Onsager reciprocal relations

## 8.6 Special Relativity
- ✅ **Lorentz transformations** — `physics.rs::lorentz_factor`, `time_dilation`, `length_contraction`
- 🔲 **4-vectors** — contravariant/covariant; inner product with η metric
- ✅ **Energy-momentum relation** — `physics.rs::relativistic_kinetic_energy`, `relativistic_momentum`, `mass_energy`, `relativistic_velocity_addition`
- ✅ **Aberration and Doppler** — `physics.rs::relativistic_doppler`

---

# PART 9 — CRYPTOGRAPHY & CODING THEORY

## 9.1 Cryptographic Primitives
Already have (`number_theory.rs`): RSA and DH; all modular arithmetic primitives for crypto.

- 🔲 **Montgomery multiplication** — modular mult without division; for large integers
- 🔲 **Barrett reduction** — fast modular reduction
- ✅ **RSA** — `rsa_keygen`, `rsa_encrypt`, `rsa_decrypt` in `number_theory.rs` (u64 toy size; no PKCS#1)
- ✅ **Diffie-Hellman** — `dh_public_key`, `dh_shared_secret` in `number_theory.rs`
- 🔲 **ElGamal** — discrete log based encryption
- 🔲 **Elliptic curve arithmetic** — point addition, doubling, scalar multiplication
  - Decomposition: accumulate(scalar_bits, point_double/add_gather, result_point)
  - Fields: Weierstrass (secp256k1), Montgomery (Curve25519), Edwards (Ed25519)

- 🔲 **ECDH** — Elliptic Curve Diffie-Hellman
- 🔲 **ECDSA** — signature with EC
- 🔲 **EdDSA** — deterministic, Edwards curve signatures
- 🔲 **Pairing-based cryptography** — Weil/Tate/ate pairings; BLS signatures

## 9.2 Hash Functions
- 🔲 **SHA-256** — Merkle-Damgård construction; 256-bit
  - Decomposition: accumulate(blocks, sha256_compression, state_update)

- 🔲 **SHA-512** — 64-bit word version
- 🔲 **SHA-3/Keccak** — sponge construction; different security model
- 🔲 **BLAKE2/BLAKE3** — fast, secure; parallel-friendly
- 🔲 **MD5** — legacy; broken but computationally important
- 🔲 **HMAC** — keyed hash; secret-prefix + HMAC construction

## 9.3 Coding Theory
- 🔲 **Hamming codes** — (7,4), (15,11), general; single error correction
- 🔲 **Reed-Solomon codes** — polynomial evaluation codes; BCH generalization
  - Over GF(q); systematic encoding; Berlekamp-Massey decoder
  
- 🔲 **BCH codes** — Bose-Chaudhuri-Hocquenghem; multiple error correction
- 🔲 **LDPC codes** — belief propagation decoding; capacity approaching
  - Decomposition: BP: accumulate(check_nodes, message_passing, belief_update)

- 🔲 **Turbo codes** — parallel concatenated convolutional; SISO decoding
- 🔲 **Polar codes** — channel polarization; Arıkan; successive cancellation decoding
- 🔲 **Fountain codes (LT/Raptor)** — rateless; erasure channels
- 🔲 **Convolutional codes** — shift register; Viterbi decoding
  - Decomposition (Viterbi): accumulate(symbols, trellis_step, path_update)

- 🔲 **Tanner graph** — bipartite representation of LDPC

## 9.4 Lattice Cryptography
- 🔲 **LWE (Learning With Errors)** — post-quantum foundation
- 🔲 **RLWE** — ring variant; more efficient
- 🔲 **SVP/CVP** — shortest/closest vector; NP-hard; LLL lattice reduction
- 🔲 **LLL algorithm** — lattice basis reduction; 1982 Lenstra-Lenstra-Lovász

---

# PART 10 — DOMAIN-SPECIFIC MATH

## 10.1 Neuroimaging (fMRI / EEG / MEG)
- 🔲 **HRF (Haemodynamic Response Function)** — double-gamma; convolution with design matrix
  - Parameters: peak_delay, undershoot_delay, peak_dispersion, undershoot_dispersion, ratio

- 🔲 **fMRI GLM** — general linear model; F-contrast, t-contrast
  - Decomposition: accumulate(timepoints, design_xᵀx, sum) → solve → t-stat_gather

- 🔲 **ICA for fMRI** — spatial ICA; identify noise and signal components
- 🔲 **EEG frequency bands** — delta/theta/alpha/beta/gamma power
- 🔲 **EEG source localization (LORETA)** — minimum current estimate; smooth
- 🔲 **MNE (Minimum Norm Estimate)** — distributed source model
- 🔲 **Beamforming** — spatial filter; LCMV/DICS
- 🔲 **Phase-amplitude coupling (PAC)** — modulation index; MI-based
- 🔲 **Coherence (neural)** — already in time series; domain-specific thresholds
- 🔲 **PLV (Phase Locking Value)** — synchrony measure; |(1/N)Σexp(iΔφ)|
  - Decomposition: accumulate(time_pairs, phase_diff_complex, sum) → magnitude

## 10.2 Genomics & Bioinformatics
- 🔲 **Hardy-Weinberg equilibrium** — p² + 2pq + q² = 1; chi-square test
- 🔲 **Linkage disequilibrium** — D, D', r² between SNPs
- 🔲 **GWAS statistics** — logistic regression per SNP; genomic inflation factor λ
- 🔲 **Population stratification PCA** — remove confounding in GWAS
- 🔲 **FST (fixation index)** — between-population variance / total variance
- 🔲 **Sequence alignment score** — Smith-Waterman, Needleman-Wunsch
  - Decomposition: accumulate(DP_cells, score_update, traceback_gather)

- 🔲 **Edit distance (Levenshtein)** — DP; O(mn) time
- 🔲 **Phylogenetic trees** — UPGMA, neighbor-joining; maximum parsimony
- 🔲 **Differential expression** — negative binomial GLM; DESeq2/edgeR models
  - Sufficient stats: gene×sample count matrix + library size factors

## 10.3 Seismology & Geophysics
- 🔲 **Travel-time tomography** — Radon transform inversion; SIRT
- 🔲 **Moment tensor** — double-couple components; fault plane solution
- 🔲 **Seismic amplitude spectrum** — source spectrum model (ω-² falloff)
- 🔲 **HVSR (Horizontal/Vertical Spectral Ratio)** — soil amplification
- 🔲 **Coda Q** — scattering attenuation from coda decay
- 🔲 **Cross-correlation beamforming** — array processing for slowness-azimuth

## 10.4 Astrophysics
- 🔲 **Hubble parameter** — H(z) integration; comoving distance
- 🔲 **Luminosity distance** — D_L = (1+z) D_c; cosmological
- 🔲 **Stellar structure** — Lane-Emden equation; polytropes
- 🔲 **N-body integration** — already noted in 8.1
- 🔲 **Matched filtering** — gravitational wave detection; SNR = ∫(h̃(f)s̃*(f)/Sₙ(f))df
  - Decomposition: accumulate(frequency_bins, template_cross_power/noise_gather, sum)

- 🔲 **Period-folding** — epoch folding; chi-square test; optical/X-ray timing
- 🔲 **Power spectral density normalization** — for noise characterization

## 10.5 Climate & Atmospheric Science
- 🔲 **EOF/PCA for climate** — spatial patterns; already have PCA; need weighting by cos(lat)
- 🔲 **Teleconnection indices** — PDO, AMO, ENSO; correlation with spatial field
- 🔲 **Spectral analysis of seasonal cycle** — harmonic decomposition
- 🔲 **Running trend** — Mann-Kendall test on detrended anomalies
- 🔲 **Extreme precipitation statistics** — Gumbel/GEV fit to annual maxima
- 🔲 **Multi-model ensemble statistics** — mean, spread, reliability

## 10.6 Finance (Extended)
Already have: volatility models, Kyle lambda, Amihud, Roll spread

- 🔲 **Black-Scholes formula** — C = S·N(d₁) - K·exp(-rT)·N(d₂)
  - Parameters: S (spot), K (strike), T (time), r (rate), σ (vol)
  - Greeks: delta, gamma, theta, vega, rho, vanna, charm, volga

- 🔲 **Binomial option pricing tree** — CRR model; American exercise
- 🔲 **Monte Carlo option pricing** — GBM paths; payoff expectation
  - Decomposition: accumulate(paths, payoff_compute, sum) / n

- 🔲 **Implied volatility** — Newton-Raphson on BS formula
- 🔲 **Local volatility** — Dupire equation; σ_local(K,T)
- 🔲 **VaR/CVaR (Expected Shortfall)** — parametric, historical, Monte Carlo
  - Historical VaR: sort returns, gather(α-quantile)
  - CVaR: accumulate(tail_returns, sum) / (n·α)

- 🔲 **Markowitz portfolio** — efficient frontier; QP: min w'Σw s.t. w'μ ≥ r, Σwᵢ=1
- 🔲 **Black-Litterman** — Bayesian mixing of prior + investor views
- 🔲 **Factor models** — Fama-French 3/5-factor; CAPM regression
  - Sufficient stats: same as OLS; factor loadings

- 🔲 **Duration and convexity** — bond sensitivity to yield changes
- 🔲 **Term structure models** — Vasicek, Hull-White, CIR for short rate

---

# PART 11 — CONTROL THEORY & ENGINEERING MATH

## 11.1 Control Theory
- 🔲 **Transfer functions** — H(s) = Y(s)/U(s); poles, zeros, gain
- 🔲 **State space** — ẋ = Ax + Bu, y = Cx + Du; controllability/observability
  - Decomposition: accumulate(gramians, discrete_lyapunov, sum_matrix)

- 🔲 **Pole placement** — Ackermann's formula
- 🔲 **LQR (Linear Quadratic Regulator)** — optimal gain K from Riccati
- 🔲 **Kalman filter** — already noted above; connect to control here
- 🔲 **Frequency response** — Bode magnitude/phase; H(jω)
- 🔲 **Nyquist criterion** — stability from Nyquist plot; winding number
- 🔲 **Routh-Hurwitz** — stability without computing poles
- 🔲 **Lyapunov stability** — V(x) > 0, V̇ < 0 → stable equilibrium
- 🔲 **PID control** — proportional-integral-derivative; Ziegler-Nichols tuning
  - Decomposition: accumulate(error_samples, pid_update, output_sum)

## 11.2 Finite Elements (1D & 2D)
- 🔲 **1D Galerkin FEM** — hat functions, stiffness matrix assembly
  - Decomposition: accumulate(elements, local_stiffness, global_assemble_gather)

- 🔲 **2D triangular FEM** — shape functions, area coordinates
- 🔲 **Isoparametric elements** — reference element + Jacobian mapping
- 🔲 **Assembly** — element → global; scatter-add pattern
  - Decomposition: accumulate(elements, local_matrix, scatter_add_gather) — directly accumulate+gather

## 11.3 Information & Coding (Engineering)
Already have: information theory module covers basic measures. Gaps:

- 🔲 **Huffman coding** — optimal prefix-free code; entropy approaching
  - Decomposition: accumulate(symbols, probability_sort, tree_build_gather)

- 🔲 **Arithmetic coding** — fractional bits; range encoding
- 🔲 **LZ77/LZ78/LZW** — lossless compression; dictionary-based
- 🔲 **BWT (Burrows-Wheeler Transform)** — for compression preprocessing
- 🔲 **Channel capacity (Shannon)** — C = B log₂(1 + SNR)
- 🔲 **Rate-distortion** — D(R) tradeoff; blahut-Arimoto algorithm

---

# PART 12 — FORMAL METHODS & LOGIC

## 12.1 Formal Verification
Already have: proof.rs with algebraic structures

- 🔲 **SAT solving** — DPLL, CDCL; unit propagation, learned clauses
- 🔲 **SMT solving** — SAT + theory lemmas (arithmetic, equality)
- 🔲 **Model checking** — LTL/CTL formulae; BDD-based symbolic MC
- 🔲 **Type theory** — System F, dependent types, propositions as types

## 12.2 Category Theory
- 🔲 **Functors and natural transformations** — Hom functors, adjunctions
- 🔲 **Monads** — unit, join, bind; Kleisli composition
- 🔲 **Limits and colimits** — products, coproducts, pullbacks, pushouts
- 🔲 **Sheaf theory** — local-to-global; applications in ML (sheaf attention)
- 🔲 **Topoi** — generalized spaces; elementary topos

---

# GAP SUMMARY (Priority Ordered)

## Tier 1 — Critical for Practical Use (implement first)
1. Full distribution library (Exponential, Gamma, Beta, Weibull, Binomial, Poisson, NB, Multinomial) — distribution objects with PDF/CDF/PPF/MLE
2. OLS complete with all diagnostics
3. Logistic/Poisson/NB regression
4. Ridge, Lasso, Elastic Net
5. AIC/BIC/AICc/cross-validation
6. Multiple comparison corrections: Bonferroni/Holm/BH already ✅; need BY, Storey q-value
7. Normality tests (Shapiro-Wilk, Anderson-Darling, Jarque-Bera)
8. Kalman filter + smoother
9. ARMA/ARIMA/Holt-Winters
10. GBM simulation + option pricing

## Tier 2 — High Value, Active Research Fields
11. Gaussian processes (exact + sparse)
12. MCMC complete (Gibbs, HMC, NUTS)
13. Copulas (Gaussian, t, Clayton, Gumbel)
14. VAR/VECM + Granger causality
15. Random forests + gradient boosting
16. K-means (🔷 exists but needs k-means++ init), GMM (✅ in mixture.rs)
17. UMAP, Isomap
18. Wavelet families (Daubechies etc.)
19. SVM (full with SMO)
20. Change point detection (PELT, BOCPD)

## Tier 3 — Physics & Pure Math
21. Numerical integration: trapezoidal/Simpson/adaptive already ✅; need Gauss-Legendre full family (have 5-pt only), Romberg, Gauss-Hermite/Laguerre, Monte Carlo
22. ODE solvers: Euler/RK4/RK45 already ✅; need symplectic (Störmer-Verlet, leapfrog), stiff solvers (BDF), multistep methods
23. Classical mechanics: N-body/orbital already ✅; need Lagrangian/Hamiltonian formalism, symplectic integrators, Barnes-Hut
24. Statistical mechanics: partition_function/Ising/MC already ✅; need Wolff, Wang-Landau, MD, LJ potential
25. GF(2) arithmetic + polynomial factorization
26. Elliptic curve arithmetic
27. Error correction codes (RS, BCH, LDPC)

## Tier 4 — Domain-Specific, Long Tail
28. Neuroimaging GLM, beamforming
29. Finance: Black-Scholes, Greeks, implied vol
30. Climate: EOF with lat weighting
31. Quantum: density matrix, spin operators
32. Fluid: lattice Boltzmann

---

# ACCUMULATE+GATHER PATTERNS (Master List)

The fundamental insight: every computation is `accumulate(grouping, expr, op) + gather(addressing)`.

## Core Patterns Seen So Far

| Pattern | Grouping | Expr | Op | Gather |
|---------|----------|------|----|--------|
| Mean | all points | x | sum | / n |
| Variance | all points | (x-μ)² | sum | / (n-ddof) |
| Grouped moments | by group_id | x, x² | sum | / n_g |
| Covariance (COPA) | all pairs | outer(xᵢ,xᵢ) | sum | / (n-1) |
| OLS | all rows | xᵢᵀxᵢ, xᵢyᵢ | sum_matrix | solve(XᵀX, Xᵀy) |
| k-means E-step | by nearest_centroid | sum(x), count | parallel | / count |
| GMM E-step | by mixture_component | γ_ik * x | sum | normalize responsibilities |
| FFT | by freq bin | x[n]·exp(−2πi·kn/N) | sum | - |
| Kalman update | sequential | H^T R^{-1} H, H^T R^{-1} z | sum | solve |
| GP regression | all pairs | k(xᵢ,xⱼ) | matrix_fill | Cholesky_solve |
| Attention | by query | exp(q·kⱼ/√d) | sum | weighted_sum(v) |
| PageRank | by in-edge | α·PR(j)/out_j | sum | + (1-α)/n |
| Hawkes intensity | past events | φ(t-tᵢ) | sum | + μ |
| Viterbi | by state | max_{prev}(V_{t-1}·a·b) | max | argmax_trace |
| BP (LDPC) | by check_node | Σ tanh(m/2) | product | 2 arctanh |
| N-body | pairwise | F_ij = G·m·m/r² | sum | - |
| Persistent homology | by simplex | boundary_matrix | reduce | column_kill |
| STL | local_window | polynomial_weights | WLS | evaluate |

---

# VERSION HISTORY
- 2026-04-06: Initial taxonomy by math-researcher (this document)
- 2026-04-06: Scout pass (session 1) — corrected 🔲→✅/🔷 for confirmed implementations in sections 1.2-1.5, 2.2-2.4, 6.5-6.6, 6.8, 5.5, 8.1-8.6; corrected Tier 3 gap summary
- 2026-04-06: Scout pass (session 2) — major corrections: §7.1 Number Theory (number_theory.rs comprehensive: GCD/LCM, ExtGCD, CRT, Miller-Rabin, Pollard ρ, totient, Möbius, primitive roots, BSGS, Tonelli-Shanks, CF, partition_count, RSA, DH); §3.7 Stochastic Processes (stochastic.rs: BM, GBM, Black-Scholes 🔷, OU, Poisson, Markov DTMC+CTMC, birth-death, queues, random walk, Itô); added §2.8b Multivariate Statistics (multivariate.rs: Hotelling T², MANOVA, LDA, CCA, Mardia); §9.1 RSA+DH ✅; spatial.rs CSR docstring fixed
