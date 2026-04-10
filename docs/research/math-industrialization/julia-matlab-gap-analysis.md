# Julia / MATLAB Gap Analysis — Tambear Missing Primitives

> Research date: 2026-04-10  
> Focus: What Julia/MATLAB have that scipy/R already cover is EXCLUDED.  
> This is strictly the delta: things unique to Julia or MATLAB ecosystems that tambear does not yet own.

---

## Method

Surveyed Julia packages: DifferentialEquations.jl (ODE/SDE/DDE/DAE), HypothesisTests.jl, Distributions.jl, StatsBase.jl, MultivariateStats.jl, Clustering.jl, Optim.jl, SpecialFunctions.jl, ApproxFun.jl, NonlinearSolve.jl, ProbNumDiffEq.jl, GaussianProcesses.jl, StateSpaceModels.jl.

Surveyed MATLAB toolboxes: Financial Toolbox, Financial Instruments Toolbox, Econometrics Toolbox, Signal Processing Toolbox, Optimization Toolbox, Statistics and Machine Learning Toolbox.

Cross-referenced against tambear's current modules: `special_functions.rs`, `optimization.rs`, `hypothesis.rs`, `time_series.rs`, `numerical.rs`, `series_accel.rs`, `state_space.rs`, `signal_processing.rs`.

---

## Section 1 — ODE/SDE/DDE/DAE Solvers (Julia DifferentialEquations.jl)

**This is the biggest gap.** scipy has `solve_ivp` with ~8 methods. Julia's SciML ecosystem has 100+. The gaps below are all methods WITHOUT scipy equivalents.

### 1.1 High-Order Explicit Runge-Kutta (missing entirely)

| Method | Order | Unique property |
|--------|-------|-----------------|
| Tsit5 | 5/4 | Tsitouras optimized coefficients; outperforms Dormand-Prince on most smooth problems |
| Vern6 / Vern7 / Vern8 / Vern9 | 6th–9th | Verner's "most efficient" methods; lazy interpolants for dense output |
| Feagin10 / Feagin12 / Feagin14 | 10th–14th | Extremely high order; only sensible option for ultra-tight tolerances |
| PFRK87 | 8/7 | Phase-fitted, zero dissipation — ideal for oscillatory problems |
| Anas5 | 5 | Explicitly designed for periodic/oscillatory problems |
| FRK65 | 6/5 | Zero numerical dissipation |
| TanYam7 | 7 | 7th-order explicit |
| OwrenZen3/4/5 | 3–5 | Optimized interpolation (dense output quality) |

scipy has only: RK45 (≈DP5), RK23 (≈BS3), DOP853 (8th order). All others above are genuine gaps.

### 1.2 Exponential Integrators (missing entirely in tambear)

These are for stiff problems where the linear part can be exactly exponentiated. Far faster than SDIRK for certain problem classes (reaction-diffusion, nonlinear Schrödinger).

- `LawsonEuler` — 1st order exponential Euler
- `NorsettEuler` — 1st order, integrating factor
- `ETDRK2`, `ETDRK3`, `ETDRK4` — 2nd/3rd/4th order Exponential Time Differencing Runge-Kutta
- `Exprb32`, `Exprb43` — adaptive exponential Rosenbrock (3rd/4th order)
- `Exp4` — 4th order Krylov exponential
- `EPIRK4s3A`, `EPIRK4s3B` — 4th order EPIRK
- `EPIRK5P1`, `EPIRK5P2`, `EXPRB53s3` — 5th order EPIRK

**Accumulate+gather view**: φ-functions (φ₀, φ₁, φ₂...) are the key primitive. φ₁(c·A)·b is the essential kernel. Matrix-vector products via Krylov (Arnoldi iteration) for large A. This is Kingdom B (sequential Krylov steps).

### 1.3 Strong-Stability Preserving (SSP) Runge-Kutta (missing)

Critical for hyperbolic PDEs (conservation laws, advection). scipy has none.

- `SSPRK22` (2-stage, order 2, SSP coefficient 1)
- `SSPRK33` (3-stage, order 3, classic Shu-Osher)
- `SSPRK53` (5-stage, order 3, optimal SSP coefficient)
- `SSPRK63`, `SSPRK73`, `SSPRK83` (6/7/8-stage, order 3)
- `SSPRK54` (5-stage, order 4)
- `SSPRK104` (10-stage, order 4, highest SSP coefficient for order 4)
- `SSPSDIRK2` (implicit SSP, 2nd order)

**Primitive needed**: `ssp_rk_step(f, u, dt, stages, weights, nodes)` + SSP coefficient computation.

### 1.4 Low-Storage Runge-Kutta (missing)

Optimize for memory: 2N or 3N storage instead of standard k-stage storage. Essential for very large PDE systems.

- `CarpenterKennedy2N54` — 4th order, 2N storage, 5 stages
- `ORK256` — 6th order, 2N, wave propagation optimized
- `NDBLSRK124/134/144` — advection-DG optimized
- `RDPK3Sp35/49/510` — dispersion-relation preserving for DG
- `CKLLSRK54_3C`, `CKLLSRK65_4M` — compressible Navier-Stokes optimized
- `Parsani-Ketcheson-Deconinck` variants — spectral element methods

**Key primitive**: 2N Williamson update: `u += a*k; k = b*k + c*f(u,t)`.

### 1.5 Symplectic / Geometric Integrators (partial gap)

tambear has some physics in `physics.rs` but these dedicated integrators are missing:

- `ImplicitMidpoint` (symplectic, preserves quadratic invariants)
- `IRKGL16` (16th-order Gauss-Legendre FIRK, symplectic)
- `Leapfrog / Verlet` — standard molecular dynamics
- `Ruth3`, `Ruth4` — 3rd/4th order symplectic
- Yoshida-Suzuki composition methods (6th/8th order symplectic from 4th order base)
- `KahanLi8` (8th order symplectic)
- `McAte5`, `Blanes3Pd` — optimized symplectic coefficients

**Key property**: these preserve the symplectic 2-form ∑dqᵢ∧dpᵢ. Phase space volume is invariant. Essential for Hamiltonian system long-time integration.

### 1.6 SDE Solvers Beyond Euler-Maruyama (major gap)

tambear's `stochastic.rs` likely only has EM. Julia has 30+ SDE algorithms:

**Nonstiff (strong order 1.0):**
- `RKMil` — Milstein for scalar/diagonal noise (strong order 1.0 vs EM's 0.5)
- `RKMilCommute` — Milstein for commutative noise
- `RKMilGeneral` — Milstein for non-commutative noise (iterated Stratonovich integrals)
- `WangLi3SMil_A/B/C/D/E/F` — 6 variants, strong order 1.0, 3-stage

**Strong order 1.5 (additive noise):**
- `SRA`, `SRA1`, `SRA2`, `SRA3` — strong order 1.5 for additive noise
- `SOSRA`, `SOSRA2` — stability-optimized SRA variants

**Strong order 1.5 (diagonal noise):**
- `SRI`, `SRIW1`, `SRIW2` — strong order 1.5 for diagonal noise
- `SOSRI`, `SOSRI2` — stability-optimized SRI variants

**High weak order (for expected values, not path accuracy):**
- `DRI1`, `RI1`–`RI6`, `RDI1WM`–`RDI4WM` — weak order 2.0
- `PL1WM`, `PL1WMA` — weak order 2.0
- `NON`, `SIEA`, `SIEB`, `SMEA`, `SMEB` — weak order 2.0

**Stiff (implicit):**
- `ImplicitEM` — drift-implicit Euler-Maruyama
- `ImplicitEulerHeun` — drift-implicit for Stratonovich
- `ImplicitRKMil` — implicit Milstein
- `ISSEM`, `ISSEulerHeun` — split-step implicit
- `SKenCarp` — strong 1.5/weak 2.0, additive noise, adaptive (the best stiff SDE solver)

**S-ROCK (stiff, stabilized explicit):**
- `SROCK1` (weak order 1), `SROCK2` (strong 1.0/weak 2.0)
- `SROCKEM` — for systems with large spectral radius

**Derivative-based:**
- `PCEuler` — requires ggprime (∂g/∂x); enables order 1.5 without Lévy area

**Key primitive gap**: Lévy area computation (J_{i,j} iterated stochastic integrals) for strong-order > 0.5 with non-commutative noise. This is the genuinely hard part — approximating L^{ij} without simulating the full Brownian bridge.

### 1.7 DDE Solvers (missing entirely)

tambear has no delay differential equation support. Julia's `MethodOfSteps` wraps any ODE solver:

- Constant delay: `MethodOfSteps(Tsit5())` — tracks discontinuity propagation
- State-dependent delay: requires adaptive stepping with event detection
- Neutral DDEs (delays in derivative terms): `MethodOfSteps(Rosenbrock23())`

**Key structural challenge**: history function h(t, p) must be interpolated for t < t₀. The Hermite interpolant from the ODE integrator serves this role. Discontinuities in the delay propagate through the solution at multiples of the lag — must be tracked as breakpoints.

### 1.8 DAE Solvers (gap)

tambear has none. MATLAB's `ode15i` and Julia's `IDA` fill this:

- `DABDF2` — 2nd order adaptive BDF for fully-implicit F(du, u, p, t) = 0
- `DFBDF` — variable-order BDF (like ode15i/DASSL)
- `RadauIIA5` for DAEs (index-1 systems)
- `Rodas4`, `Rodas5P` via mass matrix formulation
- External: `IDA` (Sundials), `DASSL`, `DASKR`

**Key primitive**: consistent initialization (finding du₀ from F(du₀, u₀, p, t₀)=0). Requires specialized Newton iteration on the implicit system.

### 1.9 Probabilistic Numerical ODE Solvers (missing — genuinely novel)

Julia's `ProbNumDiffEq.jl` (no scipy/MATLAB equivalent):

- `EK0` — Extended Kalman filter ODE solver, 0th order linearization
- `EK1` — Extended Kalman filter ODE solver, 1st order linearization

Output is a **posterior distribution** over solution trajectories, not just a point estimate. Uncertainty quantification for numerical integration is native.

**Accumulate+gather view**: GP prior is an IWP (integrated Wiener process). EKF update at each step: predict via Kronecker-structured state transition matrix, correct via likelihood. This is Kingdom B (sequential) but with a well-defined sufficient statistic (Gaussian belief state).

---

## Section 2 — Optimization Gaps (Julia Optim.jl + MATLAB Optimization Toolbox)

tambear has: gradient descent, Adam, AdaGrad, RMSProp, L-BFGS, Nelder-Mead, golden section, projected gradient, penalty method.

### 2.1 Riemannian / Manifold Optimization (missing)

Optim.jl supports optimization on:
- Sphere manifold (normalization constraint): `x/‖x‖`
- Stiefel manifold (orthogonality constraint): `Xᵀ X = I`
- Grassmann manifold: subspace optimization
- Symmetric positive definite matrices (SPD manifold)

Operations needed: retraction (project tangent step back to manifold), parallel transport (move gradient across manifold), geodesic distance.

**Why this matters for tambear**: PCA, ICA, dictionary learning, Riemannian kernel methods all optimize on manifolds. The current penalty-based approach loses manifold structure.

### 2.2 Krylov Trust-Region (missing)

- `KrylovTrustRegion` — trust region with Steihaug-CG for the subproblem. Scales to large problems (matrix-free, only needs Hv products). Better than full Newton for large n.

### 2.3 MILP / Integer Programming (major gap)

MATLAB's Optimization Toolbox has `intlinprog` (branch-and-cut MILP). No tambear equivalent. Julia has `HiGHS.jl`, `Cbc.jl` via `JuMP.jl`. Primitives needed:
- Branch-and-bound with LP relaxation at each node
- Cutting planes (Gomory cuts, mixed-integer Gomory)
- Presolve: bound tightening, coefficient strengthening

### 2.4 SOCP / SDP (missing)

- Second-order cone programming: min cᵀx s.t. ‖Aᵢx + bᵢ‖ ≤ cᵢᵀx + dᵢ
- Semidefinite programming: min cᵀx s.t. F₀ + Σ xᵢFᵢ ⪰ 0

Both are interior-point methods but with specialized barrier functions. Neither exists in tambear.

### 2.5 NGMRES / OACCEL (acceleration, missing)

Nonlinear GMRES (NGMRES) and Anderson acceleration — converge fixed-point iterations by building a Krylov-like subspace of iterates and solving a small least-squares system. Useful for: self-consistent field iterations, EM algorithm acceleration, optimizer warm-starting.

---

## Section 3 — Probability Distributions (Distributions.jl gaps)

tambear has standard distributions via special functions. The following are in Distributions.jl but NOT in tambear or scipy.stats:

### Continuous distributions not in scipy.stats:
- `Chernoff` — Chernoff distribution (supremum of Brownian motion minus parabola)
- `JohnsonSU` — Johnson S_U family (unbounded, heavy-tailed transformation of normal)
- `Kolmogorov` — limiting distribution of KS statistic
- `KSDist` — exact finite-n KS distribution (not approximation)
- `KSOneSided` — one-sided KS distribution
- `NormalCanon` — canonical (information) parameterization: N(η, Λ) where η=Λμ
- `NormalInverseGaussian` — 4-parameter; subclass of GH distribution; heavy tails + skewness
- `PGeneralizedGaussian` — generalized Gaussian/exponential power; p=1 is Laplace, p=2 is Normal
- `Semicircle` — Wigner semicircle; emerges in random matrix theory
- `SkewedExponentialPower` — 5-parameter; unifies skewness + tail weight
- `StudentizedRange` — distribution of (max-min)/S for normal samples; used in Tukey HSD

### Discrete distributions not in scipy.stats:
- `BernoulliLogit` — Bernoulli parameterized by log-odds directly (avoids sigmoid)
- `PoissonBinomial` — sum of independent Bernoullis with different p_i (via DFT algorithm)
- `Soliton` — ideal soliton distribution (used in fountain codes / LT codes)

### Matrix-variate (entirely missing from tambear):
- `Wishart` — distribution over positive definite matrices; p×p; df > p-1
- `InverseWishart` — conjugate prior for multivariate normal covariance
- `MatrixNormal` — M×N matrix with row and column covariance structure
- `MatrixTDist` — matrix t-distribution
- `MatrixBeta` — matrix-variate beta

**Why Wishart matters**: Bayesian multivariate analysis, random matrix theory, covariance estimation with uncertainty — all require Wishart sampling and likelihood.

---

## Section 4 — Weighted Statistics (StatsBase.jl gaps)

Julia's StatsBase.jl has first-class weighted variants of nearly everything. tambear's weighting support is partial. Missing weighted primitives:

- `mean(x, weights)`, `var(x, weights)`, `std(x, weights)` — all weight types
- `AnalyticWeights` — for analytical approximations (denominator n, not n-1, not Σw)
- `FrequencyWeights` — integer replication counts (denominator Σw - 1)
- `ProbabilityWeights` — inverse-sampling-probability (denominator n - 1)
- `UnitWeights` — all-ones sentinel (allows generic code paths)
- Weighted covariance matrices with each of the 4 weight types, each producing different bias corrections
- `wquantile` — weighted quantile (multiple algorithms: type 1-9 all weighted)
- `wmean_and_var` — joint computation (avoids second pass)
- Weighted `entropy`, weighted `kurtosis`, weighted `skewness`
- Efraimidis-Spirakis A-Res algorithm for weighted sampling without replacement (O(n log n))
- Weighted Kaplan-Meier (left-truncation weights)

---

## Section 5 — Hypothesis Tests (HypothesisTests.jl gaps)

tambear's `hypothesis.rs` is substantial. True gaps:

- `Clark-West` test — predictive accuracy comparison for nested models (returns CW statistic + p-value)
- `Diebold-Mariano` test — equal predictive accuracy of two forecasts (not the same as any t-test)
- `White` test — heteroskedasticity test (regression of squared residuals on all cross-products)
- `Wald-Wolfowitz` runs test — tests independence via run lengths (not Ljung-Box)
- `Hotelling T²` multivariate location test (single sample and two-sample)
- Equality of covariance matrices (Box's M test)
- `Fligner-Killeen` test — nonparametric equality of variances (robust to non-normality)
- `Breusch-Godfrey` test — serial correlation in regression residuals (general lag test)
- `Power divergence` family — unifies chi-square, log-likelihood ratio, Freeman-Tukey, etc. via parameter λ

---

## Section 6 — Signal Processing (MATLAB Signal Processing Toolbox gaps)

tambear has `signal_processing.rs`. MATLAB-unique items:

### 6.1 Subspace spectral estimation (missing)

- `MUSIC` (Multiple Signal Classification) — eigendecompose covariance into signal+noise subspaces; peak-find in noise-subspace rejection = DOA / frequency estimation. Resolves frequencies closer than Rayleigh limit.
- `ESPRIT` (Estimation of Signal Parameters via Rotational Invariance) — related to MUSIC; uses invariance structure of signal subspace; yields frequencies directly (no search)
- `Root-MUSIC` — root the MUSIC polynomial instead of peak-searching; exact for polynomial systems
- `MinVariance` / Capon — minimum variance distortionless response (MVDR) beamformer as spectral estimator

**Key primitive**: eigendecompose Rxx, partition into signal subspace (top K eigenvectors) and noise subspace. MUSIC pseudo-spectrum = 1/‖aᴴ(ω)E_n‖².

### 6.2 Parametric spectral estimation (partial gap)

- `Burg method` — AR parameter estimation via Levinson-Durbin on Burg reflection coefficients; preserves stationarity; more stable than OLS AR fitting
- `Covariance method` (forward only Burg variant)
- `Modified covariance method` (forward-backward Burg)
- `Yule-Walker` — AR via autocorrelation matrix; less biased than OLS for short sequences

### 6.3 Cepstrum analysis (missing)

- `cceps` — complex cepstrum: IFFT(log(FFT(x))); used for speech, echo removal, deconvolution
- `icceps` — inverse complex cepstrum (reconstruct from cepstrum + phase unwrapping)
- `rceps` — real cepstrum: IFFT(log|FFT(x)|); minimum-phase analysis
- `Liftering` — windowing in the cepstral domain to separate fast (excitation) from slow (vocal tract)

**Why useful for finance**: cepstrum separates multiplicative components in log-domain; market volatility signature isolation is an application.

### 6.4 Chirp Z-Transform (missing)

The CZT evaluates the Z-transform on a spiral contour in the z-plane, not just the unit circle. Enables:
- FFT of prime lengths (no power-of-2 constraint)
- Zoomed spectral analysis of a sub-band with arbitrary resolution
- Evaluation of z-transform at arbitrary points for pole-zero analysis

Algorithm: Bluestein's algorithm — reduces to 3 FFTs + pointwise multiplication.

### 6.5 Synchrosqueezing / Reassignment (missing)

- `Synchrosqueezing transform` — sharpens CWT/STFT by reassigning energy to instantaneous frequency; resolves multicomponent signals with crossing frequencies
- `Reassignment method` — time-frequency energy reassignment using local group delay and instantaneous frequency
- `Concentration of frequency and time (ConceFT)` — multi-taper version of synchrosqueezing

### 6.6 Correlation / Matched Filter (gap)

- `xcorr` with 'coeff', 'unbiased', 'biased' normalization modes — MATLAB distinguishes all four
- `finddelay` — finds delay between signals via cross-correlation peak
- `gccphat` — GCC-PHAT (generalized cross-correlation with phase transform): time-delay estimation in reverberant environments

---

## Section 7 — Financial Primitives (MATLAB Financial + Financial Instruments Toolbox)

tambear has `volatility.rs`, `time_series.rs`, `causal.rs`. MATLAB-unique:

### 7.1 Fixed-Income / Yield Curve (missing)

- `bndprice` — bond price from yield-to-maturity (handles day-count conventions: actual/actual, 30/360, actual/360, actual/365)
- `bndyield` — yield-to-maturity from price (Newton iteration on bond pricing equation)
- `bndconvy` — convexity of bond
- `bnddury` — modified and Macaulay duration
- `zbtprice` / `zbtyield` — zero curve bootstrapping from coupon bond price/yield data
- `zero2disc`, `zero2fwd`, `zero2pyld` — curve conversions: zero → discount → forward → par yield
- Yield curve interpolation: linear on discount factors, log-linear, cubic spline (Steeley, McCulloch)

### 7.2 Option Pricing Beyond Black-Scholes (partial gap)

- Heston model (stochastic volatility): analytical formula via characteristic function + Fourier inversion
- SABR model: Hagan's approximate formula for implied vol
- CEV model: Constant Elasticity of Variance — analytical for European options
- Binomial tree (CRR, LR, Tian variants) for American options
- Trinomial tree for barrier/lookback options
- Finite difference for American options: Crank-Nicolson + PSOR (projected SOR)
- Jump-diffusion (Merton, Kou): characteristic function methods

**Key primitive gap**: `characteristic_function(model, ω)` — returns ψ(ω) for fast Fourier option pricing (Carr-Madan, Lewis). This single primitive unlocks Heston, VG, NIG, CGMY, all affine models.

### 7.3 Interest Rate Models (missing)

- Vasicek SDE simulation and bond pricing: dr = κ(θ-r)dt + σdW; P(t,T) has closed form
- CIR model: dr = κ(θ-r)dt + σ√r dW; non-central chi-squared distribution for r(T)
- Hull-White (one-factor): extension of Vasicek; time-varying θ(t) fitted to yield curve
- Ho-Lee: dr = θ(t)dt + σdW; simplest no-arbitrage model; trinomial tree calibration
- Black-Derman-Toy: lognormal rates on binomial tree; calibrated to term structure + vol structure
- HJM framework: forward rate SDE df(t,T) = α(t,T)dt + σ(t,T)dW; no-arbitrage drift condition

### 7.4 Yield Curve Fitting Models (missing)

- Nelson-Siegel: r(T) = β₀ + β₁·(1-e^{-τT})/(τT) + β₂·[(1-e^{-τT})/(τT) - e^{-τT}]
- Nelson-Siegel-Svensson (extended): adds second hump term
- Smith-Wilson (used by EIOPA for insurance): extrapolates beyond observed maturities to ultimate forward rate
- Cubic spline fit to zero rates with smoothness penalty

### 7.5 Credit Derivatives (missing)

- Merton structural model: equity = call on firm assets; default when A < D; implied PD from market prices
- Hazard rate / intensity model: Λ(t) from CDS spreads; survival probability S(t) = exp(-∫Λ dt)
- CDS pricing: annuity leg vs protection leg; par spread from hazard rate
- CDS option: payer/receiver swaption on CDS; Black model on par spread

---

## Section 8 — Econometrics (MATLAB Econometrics Toolbox gaps)

### 8.1 Cointegration (partial gap)

tambear's time_series.rs likely has Engle-Granger. Missing:
- `jcitest` — Johansen cointegration test: trace statistic and max eigenvalue statistic; correct asymptotic critical values from MacKinnon-Haug-Michelis tables
- `vecm` — Vector Error-Correction Model: VAR in differences + cointegrating relations; estimation (OLS, MLE); impulse response; forecast error variance decomposition

### 8.2 Bayesian VAR (missing)

- Minnesota prior (Litterman): shrinks VAR coefficients toward random walk; handles curse of dimensionality for large VARs
- Sims-Zha prior: Normal-Wishart conjugate; analytical posterior; fast sampling
- Stochastic volatility VAR: time-varying error covariance
- Identification: sign restrictions, zero restrictions, external instruments (Proxy-SVAR)

### 8.3 Time-Varying Parameter Models (missing)

- TVP-VAR: Kalman filter + smoother on VAR with random-walk drifting coefficients
- Markov-switching VAR (MS-VAR): Hamilton filter for regime probabilities; EM estimation
- Threshold VAR (TVAR): different VAR in each regime defined by observable threshold

---

## Section 9 — Linear Algebra (MATLAB unique functions)

### 9.1 Generalized SVD (missing)

MATLAB's `gsvd(A, B)` returns the generalized SVD: A = U·C·Xᴴ, B = V·S·Xᴴ where Cᴴ·C + Sᴴ·S = I. Used in: generalized least squares, CCA, regularized regression, Fisher LDA, weighted PCA.

scipy doesn't have a user-facing GSVD (only wraps LAPACK dggsvd3 at low level). This is a genuine gap.

### 9.2 Ordered Schur Decomposition (missing)

- `ordschur(U, T, select)` — reorder Schur decomposition so selected eigenvalues appear first. Used in: stable/unstable manifold computation, spectral projectors, DARE/CARE solution refinement.

### 9.3 Structured Matrix Factorizations (missing)

- `ldl(A)` — LDL^T factorization for symmetric (not necessarily PD) matrices; pivoting for indefinite
- `rq(A)` — RQ factorization (reverse QR); used in Hessenberg reduction
- `hess(A)` — upper Hessenberg form; prerequisite for QR iteration; MATLAB exposes this explicitly
- `balance(A)` — Parlett-Reinsch diagonal similarity scaling before eigenvalue computation; reduces condition number of eigenvectors

---

## Section 10 — Function Approximation (Julia ApproxFun.jl)

### 10.1 Chebyshev operator spectral methods (missing)

ApproxFun represents functions and differential operators in Chebyshev basis and solves BVPs spectrally (ultra-high accuracy, spectral convergence). No scipy/MATLAB equivalent at this level:

- Automatic degree determination: represents f to machine precision with minimal coefficients
- Operator arithmetic: L·u = f solved directly in coefficient space via recurrence
- Multi-interval domains: piecewise Chebyshev with continuity conditions at breakpoints
- Evaluation of (non-linear) ODE BVPs via spectral linearization + Newton

**Key primitive**: `cheb_coefficients(f, n)`, `cheb_evaluate(c, x)`, Chebyshev differentiation matrix `D_cheb`, Clenshaw-Curtis quadrature weights.

tambear has `series_accel.rs` but likely not the full Chebyshev operator infrastructure.

### 10.2 Rational Approximation (missing)

- `Remez algorithm` — minimax rational approximation: finds p(x)/q(x) minimizing ‖f(x) - p(x)/q(x)‖∞
- `Padé approximant` — rational Taylor matching: p[m,n](x) = P_m(x)/Q_n(x) matching first m+n+1 Taylor terms
- `AAA algorithm` (Nakatsukasa-Sète-Trefethen 2018) — adaptive Antoulas-Anderson rational approximation; near-minimax in O(N²) iterations; handles poles, branch cuts, essential singularities

The AAA algorithm is genuinely new (2018) and has no scipy/MATLAB equivalent of comparable quality. It is used for: surrogate modeling, fast approximation of special functions, rational quadrature nodes.

---

## Section 11 — Nonlinear Solvers (Julia NonlinearSolve.jl)

tambear has basic Newton solvers. Missing beyond standard Newton/bisection:

- `Klement` — Jacobian-free quasi-Newton for nonlinear systems; updates approximation without finite differences
- `PseudoTransient` — solve F(u)=0 by time-marching du/dt = F(u) to steady state; robust for stiff-near-singular Jacobians
- `SparseDiff` — automatic Jacobian via sparse coloring (matrix-free groups of finite differences that don't interfere)
- `TrustRegion` method for nonlinear systems (Moré-Sorensen)
- `LevenbergMarquardt` for least-squares nonlinear systems
- Anderson acceleration as a nonlinear solver wrapper

---

## Section 12 — Probabilistic Numerical Methods (Novel Julia — No scipy/MATLAB equivalent)

These are research-frontier methods with no established scipy or MATLAB implementations:

### 12.1 Probabilistic ODE Solvers

`ProbNumDiffEq.jl` EK0/EK1: output is a `GaussMarkovProcess` posterior, not a point trajectory. Enables:
- Calibrated uncertainty in simulation results
- Sequential Monte Carlo integration over uncertain ODE solutions
- Model comparison using marginal likelihoods of ODE trajectories

### 12.2 Bayesian Quadrature (BQ)

Estimate ∫f(x)p(x)dx with uncertainty by placing GP prior on f. The posterior over the integral is a Gaussian. Optimal nodes are not uniform/Gauss — they are determined by minimizing posterior variance.

Key: `bq_weights(kernel, nodes)` — compute BQ weights from kernel matrix. For SE kernel: equivalent to kernel quadrature / quasi-Monte Carlo but with error bars.

### 12.3 Iterated Extended Kalman Smoother (IEKS) for Parameter Estimation

When fitting ODE models to data, wrap ODE solution in EKF/IEKS. Parameters and states are jointly estimated. Avoids the need for explicit adjoint / sensitivity computation.

---

## Priority Summary

| Priority | Gap | Why urgent |
|----------|-----|-----------|
| P0 | Milstein SDE (RKMil, SRI, SRA families) | Strong order 1.0/1.5 vs EM's 0.5 — fundamental correctness gap |
| P0 | MUSIC / ESPRIT subspace spectral methods | No scipy equivalent; essential for high-res frequency estimation |
| P0 | Wishart / InverseWishart distributions | Bayesian multivariate analysis requires this |
| P0 | Characteristic function option pricing (Heston/affine) | One primitive unlocks all affine models |
| P1 | SSP Runge-Kutta family | Hyperbolic PDE / conservation law support |
| P1 | Exponential integrators (ETDRK family) | Stiff reaction-diffusion; order of magnitude faster than SDIRK |
| P1 | Johansen cointegration (jcitest) | Standard econometrics; currently absent |
| P1 | Generalized SVD | CCA, regularized regression, Fisher LDA |
| P1 | Burg / MUSIC / Yule-Walker AR estimation | Parametric spectral estimation family |
| P1 | PoissonBinomial distribution | DFT algorithm; no scipy equivalent |
| P1 | Cepstrum (complex + real) | Standard DSP primitive; MATLAB-specific |
| P2 | Riemannian manifold optimization | PCA, dictionary learning on manifolds |
| P2 | Chirp Z-Transform | Prime-length FFT, zoomed spectra |
| P2 | AAA rational approximation | Near-minimax rational approx; 2018 algorithm |
| P2 | Yield curve models (Nelson-Siegel, Hull-White) | Fixed income primitives |
| P2 | DDE support (MethodOfSteps pattern) | Delay systems; no current support |
| P3 | Probabilistic ODE solvers (EK0/EK1) | Research frontier; calibrated uncertainty |
| P3 | Bayesian quadrature | GP-based integration with error bars |
| P3 | TVP-VAR, MS-VAR | Time-varying econometric models |

---

## Primitive Extraction — What Each Gap Decomposes Into

### Milstein → primitives needed
- `levy_area_approx(dW_i, dW_j, dt)` — approximate J_{ij} iterated integral
- `milstein_correction(g, dgdx, dW, dt)` — the ½ g·(∂g/∂x)·(dW²-dt) correction term

### MUSIC → primitives needed
- `covariance_matrix(X)` — already exists (TamSession shared)
- `eigendecompose(R)` — already exists
- `signal_noise_partition(eigenvalues, K)` — determine K signal eigenvectors
- `music_pseudospectrum(noise_eigenvecs, steering_matrix)` — 1/‖Eₙᴴa(ω)‖²

### Characteristic function pricing → primitives needed
- `cf_heston(u, S0, K, T, r, q, kappa, theta, sigma, rho, v0)` → ℂ
- `carr_madan_fft(cf, K_grid, S0, r, T, alpha)` → option price array
- `lewis_formula(cf, K, S0, r, T)` → single option price via contour integral

### Wishart → primitives needed
- `bartlett_decomposition_sample(df, scale_chol)` — Bartlett's method for Wishart sampling
- `wishart_logpdf(X, df, scale)` — Wishart log density
- `inverse_wishart_sample(df, scale_inv)` — via Wishart inversion

---

*Sources consulted: docs.sciml.ai/DiffEqDocs, juliastats.org, julianlsolvers.github.io/Optim.jl, juliamath.github.io/SpecialFunctions.jl, juliaapproximation.github.io/ApproxFun.jl, mathworks.com/help/finance, mathworks.com/help/fininst, mathworks.com/help/econ, mathworks.com/help/signal, arxiv.org/abs/2403.16341 (NonlinearSolve.jl paper)*
