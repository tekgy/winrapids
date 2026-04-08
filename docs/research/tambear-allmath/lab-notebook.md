# Lab Notebook: tambear-allmath Scientific Audit

**Date**: 2026-04-06
**Author**: observer (independent scientific documentarian)
**Branch**: main
**Status**: Active — ongoing expedition
**Mission**: ALL MATH IN ALL FIELDS, from scratch, no dependencies, GPU-native, accumulate+gather decomposed

---

## Purpose of This Notebook

I am embedded as an independent scientific observer, not part of the creative team. My job:

1. **Verify claims** — what IS true, not what we hope is true
2. **Document the mathematical state** — which implementations are correct, which are approximate, which are wrong
3. **Track test quality** — "all tests pass" is not correctness; tests must assert mathematical truth
4. **Identify known bugs** — catalog confirmed issues, not silence them
5. **Record discoveries in real time** — not after the fact

The principle: *if a test passes but the answer is wrong, the test is wrong.*

---

## Session 1: Initial Audit — 2026-04-06

### Before (written before exploring)

**Hypothesis**: A large body of math is implemented (~520+ functions per team briefing), but test quality will be uneven — gold standard tests against scipy/numpy will be rigorous, while in-source unit tests may be code snapshots rather than truth assertions.

**Question this session answers**: What IS the current state? Where are the verified gems, and where are the known holes?

---

### Finding 1: Scale of the codebase

**Measured (not estimated)**:

| Metric | Count |
|--------|-------|
| Source files in crates/tambear/src/ | 85 |
| Total source lines | ~68,239 |
| Test files in crates/tambear/tests/ | 18 |
| Total `#[test]` markers (src + tests) | 2,861 |
| Known confirmed bugs in test comments | 77 |

**Module sizes (top 10 by line count)**:

| Module | Lines | Notes |
|--------|-------|-------|
| fold_irreversibility.rs | 3,119 | Novel — Collatz/thermodynamic |
| proof.rs | 3,002 | Formal proofs |
| bigfloat.rs | 2,581 | Arbitrary precision |
| series_accel.rs | 2,229 | Series acceleration (Aitken, Wynn, Richardson) |
| neural.rs | 1,904 | Neural net math |
| descriptive.rs | 1,837 | Descriptive statistics |
| tbs_executor.rs | 1,757 | TBS language runtime |
| manifold.rs | 1,629 | Geometric manifolds |
| interpolation.rs | 1,612 | Interpolation methods |
| signal_processing.rs | 1,441 | Signal processing |

---

### Finding 2: Test quality classification

**Test file breakdown**:

| File | Tests | Quality Assessment |
|------|-------|--------------------|
| gold_standard_parity.rs | 826 | **GOLD** — verified against scipy/numpy/sklearn |
| adversarial_boundary*.rs (×10) | ~422 | **BOUNDARY** — edge cases, mathematical correctness focus |
| adversarial_disputed.rs | 113 | **DISPUTED** — documenting known algorithmic deviations |
| adversarial_tbs.rs | 49 | **TBS** — language runtime tests |
| scale_ladder.rs | 13 | **SCALE** — mostly structural |
| scale_ladder_dbscan_knn.rs | 1 | structural |
| scale_ladder_descriptive.rs | 1 | structural |
| scale_ladder_kde.rs | 1 | structural |
| scale_ladder_kde_fft.rs | 1 | structural |
| svd_adversarial.rs | 29 | **ADVERSARIAL** — SVD edge cases |

**In-source test quality sample**:

| Module | Tests | Quality |
|--------|-------|---------|
| accumulate.rs | 25 | Structural/smoke |
| neural.rs | 56 | Mostly structural |
| series_accel.rs | 48 | Mix: some math truth, some smoke |
| special_functions.rs | 24 | Mix |
| graph.rs | 23 | Mix |

**Critical observation**: The gold_standard_parity.rs file is 12,932 lines and tests against actual scipy/numpy expected values. This is the highest-quality test in the suite. ONLY softmax is explicitly marked "VERIFIED ✓". The rest have gold standard references but no explicit sign-off.

---

### Finding 3: Confirmed bugs catalog

77 confirmed bugs documented in test comments. Key clusters:

**Survival analysis (adversarial_boundary2.rs)**:
- `kaplan_meier` — CONFIRMED infinite loop on NaN times (uses `partial_cmp().unwrap()`)
- `cox_ph` — CONFIRMED infinite loop on NaN times
- `log_rank_test` — CONFIRMED infinite loop on NaN times  
- `cox_ph` with perfect separation — CONFIRMED NaN log-likelihood (exp overflow)

**TDA (adversarial_boundary.rs)**:
- `rips_h0` with n=1 — CONFIRMED empty diagram returned; should produce (birth=0, death=∞)
- `kde_fft` with identical data — CONFIRMED returns empty

**Nonparametric (adversarial_boundary.rs)**:
- `kruskal_wallis` with empty group — CONFIRMED 0/0 = NaN

**Mixed effects (adversarial_boundary.rs)**:
- `icc_oneway` with k=1 — CONFIRMED division by (k-1)=0, masked by NaN>0.0 branch
- `icc_oneway` with n=k — CONFIRMED ms_within = 0/0

**KNN (adversarial_boundary5.rs)**:
- `knn` with k=0 — CONFIRMED subtract overflow at knn.rs:111
- KNN NaN distance — CONFIRMED NaN enters as nearest neighbor
- `knn_from_distance` with k=0 — CONFIRMED panic

**Multivariate (adversarial_boundary5.rs)**:
- `correlation_matrix` on constant data — CONFIRMED NaN (0/0)
- Hotelling T² with n=1 — CONFIRMED singular covariance issue
- Propensity scores with perfect separation — CONFIRMED infinite IPW weights

**Volatility (adversarial_boundary3.rs)**:
- EWMA sigma2[0] not floored — CONFIRMED
- AR near-unit-root → sigma2 negative in Levinson-Durbin — CONFIRMED
- ADF with too many lags → negative MSE → NaN statistic — CONFIRMED

**KDE (adversarial_boundary10.rs)**:
- Silverman bandwidth on constant data — CONFIRMED returns 0 (causing div-by-zero in KDE)
- KDE with bandwidth=0 — CONFIRMED NaN/Inf output

**RNG (adversarial_boundary10.rs)**:
- `sample_exponential(lambda=0)` — CONFIRMED no guard
- `sample_gamma(alpha=0)` — CONFIRMED NaN
- `sample_geometric(p=0)` — CONFIRMED infinite loop

**Important observation**: These bugs are DOCUMENTED in tests — they pass because the tests use `eprintln!` to report the bug and then don't fail hard on it (or assert a relaxed property). This means **the test suite passes despite these bugs being present**. The bugs are cataloged but not fixed.

---

### Finding 4: Mathematical correctness survey (by domain)

**Special Functions (special_functions.rs, 20 pub fns)**:
- erf, erfc, log_gamma, gamma, log_beta, digamma, trigamma
- regularized_incomplete_beta, regularized_gamma_p/q
- normal_cdf, normal_sf, t_cdf, f_cdf, chi2_cdf, chi2_sf
- normal_two_tail_p, t_two_tail_p, f_right_tail_p, chi2_right_tail_p
- Status: Gold standard tests cover these; need to verify implementations

**Linear Algebra (linear_algebra.rs, 41+ pub fns)**:
- Full: Mat struct, LU, Cholesky, QR, SVD, eigenvalues, pseudoinverse
- High-quality SVD adversarial tests exist (svd_adversarial.rs, 29 tests)
- Gold standard compares against numpy for dot product, L2 distance

**Descriptive Statistics (descriptive.rs, Family F06)**:
- Mean, variance, std, skewness, kurtosis, CV, trimmed mean, geometric/harmonic mean, MAD, Gini
- Gold standard tests against scipy/numpy verified at tolerance 1e-9 to 1e-14
- Cancellation canary test for centered implementation exists ✓

**Hypothesis Testing (Family F07)**:
- t-test (one-sample, two-sample, Welch, paired), ANOVA, chi-square (GoF + independence)
- Proportion tests, effect sizes (Cohen's d, η², ω²), multiple comparison corrections
- Sharing chain: F07 consumes F06's MomentStats — tested

**Non-parametric (F08 + nonparametric.rs)**:
- Ranking, Spearman, Kendall tau, Mann-Whitney U, Wilcoxon, Kruskal-Wallis, KS test, Sign test
- Gold standard against scipy.stats
- Known bug: Kruskal-Wallis with empty group → NaN

**Information Theory (F25)**:
- Shannon, Rényi, Tsallis entropy
- KL divergence, JS divergence, cross-entropy
- Mutual information, NMI, variation of information, conditional entropy, AMI
- Gold standard against scipy.stats.entropy, sklearn.metrics

**Series Acceleration (series_accel.rs)**:
- Aitken Δ², Wynn epsilon, Richardson extrapolation, Euler transform, Abel sum
- Euler-Maclaurin zeta, detect_convergence, generic accelerate()
- 48 inline tests — quality unclear (smoke vs truth)

**Neural (neural.rs)**:
- Activations: relu, leaky_relu, elu, selu, gelu, swish, mish, sigmoid, tanh, softplus, softsign, hard_sigmoid, hard_swish, softmax, log_softmax
- Conv1d, conv2d, max_pool, batch_norm, layer_norm
- 56 inline tests; softmax has gold standard

---

### Finding 5: Notable algorithmic deviations (from adversarial_disputed.rs)

**t-SNE**:
1. Gradient update is Gauss-Seidel (updates in-place) not Jacobi (freeze snapshot) — documented, not fixed
2. Missing early exaggeration (standard: multiply P by 4.0 for first ~250 iterations) — documented, not fixed
3. These are real departures from the canonical algorithm; may affect results on non-trivial data

---

### Step 1 Summary — Initial landscape

**Hypothesis assessment**: Confirmed. Test quality is highly bimodal:
- gold_standard_parity.rs = rigorous, verified against external references
- In-source tests = mix of smoke and structural; mathematical correctness unclear without manual inspection
- adversarial tests = mathematically meaningful, but many bugs are "documented but not fixed"

**Key metric**: 77 confirmed bugs, all passing tests (because tests report bugs with eprintln and relax assertions). This means **`cargo test` passes while bugs are present**. The green test suite is not correctness.

**Surprise**: gold_standard_parity.rs is 12,932 lines and 826 tests — the expedition has done serious ground-truth work. Only softmax has a formal "VERIFIED ✓" mark. This is the critical gap: most gold standard tests have expected values from scipy but no one has explicitly signed off that the impl is correct (only that it matches the computed gold standard).

**Next**: Deeper audit of gold_standard_parity.rs to classify which sections are likely correct vs likely suspicious. Focus on numerical precision, edge cases in statistical distributions.

---

---

### Step 2: Test run results — 2026-04-06

**Before**: Expected all tests to pass (green suite is the claim), but knowing 77 documented bugs exist.

**Result (partial, first run completed)**:

```
Main library unit tests:     1,346 passed; 0 failed; 5 ignored  (175.92s)
adversarial_boundary*.rs:      422 passed; 0 failed  (all 10 files)
adversarial_disputed.rs:       113 passed; 0 failed   (31.74s)
```

**Discussion**: Confirmed — zero test failures despite 77 documented bugs. The test suite is genuinely green AND genuinely buggy simultaneously. This is the critical finding.

The 5 ignored tests are significant — what are they? Need to investigate. Potentially: tests that panic on NaN input (infinite loops would cause timeout, so `#[ignore]` is the workaround).

The 175-second runtime for the main library suite is expected given the MCMC and iterative algorithms.

---

---

### Finding 8: Mathematical correctness spot-checks

**Manually verified correct**:

| Algorithm | Method | Verdict |
|-----------|--------|---------|
| erf(1.0) | Known value 0.8427007929 ± 1.5e-7 | ✓ Correct, uses A&S 7.1.26 |
| erfc(x) + erf(x) = 2 | Algebraic identity | ✓ Verified in test |
| digamma(1) = -γ | Euler-Mascheroni constant | ✓ Correct at FINE_TOL |
| digamma recurrence ψ(x+1) = ψ(x) + 1/x | Identity | ✓ Verified for 6 x values |
| trigamma(1) = π²/6 | Known value (Basel) | ✓ Correct |
| I_x(a,b) + I_{1-x}(b,a) = 1 | Symmetry identity | ✓ Verified at 1e-10 |
| t(df=1) is Cauchy, P(T≤1) = 0.75 | Known distribution | ✓ Correct |
| Normal CDF Φ(x) + Φ(-x) = 1 | Symmetry | ✓ Verified |
| Holm correction, p=[0.01,0.04,0.03,0.5] | Manual computation | ✓ Implementation correct |
| Bonferroni: p*m, capped | Definition | ✓ Trivially correct |
| BH FDR ≤ Bonferroni (always) | Mathematical property | ✓ Verified |
| Wynn ε on Leibniz series, 20 terms | π/4 at 1e-10 | ✓ Correct |
| Richardson extrapolation | 100x improvement on ∫x² dx | ✓ Correct |
| Euler transform on Leibniz | 100x improvement | ✓ Correct |
| Aitken Δ² acceleration | 10x improvement | ✓ Correct |
| Odds ratio (10,5,5,10) = 4.0 | OR = ad/bc | ✓ Exact at 1e-14 |

**109 sections in gold_standard_parity.rs** reference scipy/numpy/sklearn expected values. All 826 tests pass. The test oracles were pre-computed and the code agrees with them.

**Key caveat**: Not all oracles have been independently cross-checked by the observer. The confidence chain is: scipy → oracle script → hardcoded expected values → tambear. If scipy has a bug or the oracle script was wrong, all downstream tests inherit the error. For standard statistical functions (t-test, chi-square, Mann-Whitney), scipy is the gold standard. This chain is trustworthy.

---

---

### Step 3: Complete test suite results (final verification)

All background test runs confirmed. Full breakdown:

| Test suite | Passed | Failed | Ignored | Runtime |
|-----------|--------|--------|---------|---------|
| lib unit tests | 1,346 | 0 | 5 (CUDA GPU) | 175s |
| adversarial_boundary*.rs (×10) | 422 | 0 | 0 | fast |
| adversarial_disputed.rs | 113 | 0 | 0 | 35s |
| adversarial_tbs.rs | 49 | 0 | 0 | 0.2s |
| gold_standard_parity.rs | 826 | 0 | 0 | 37s |
| scale_ladder.rs | 0 | 0 | 13 (benchmarks) | 0s |
| scale_ladder_descriptive.rs | 1 | 0 | 0 | 30s |
| scale_ladder_kde.rs | 1 | 0 | 0 | 19s |
| scale_ladder_kde_fft.rs | 1 | 0 | 0 | **196s** |
| scale_ladder_dbscan_knn.rs | 1 | 0 | 0 | 10s |
| svd_adversarial.rs | 22 | 0 | 7 (benchmarks) | fast |
| **TOTAL** | **~2,818** | **0** | **~25** | — |

**CONFIRMED**: Zero failures across the entire suite.

**Performance flag**: `scale_ladder_kde_fft.rs` takes 196 seconds for a single test. This is extremely slow and suggests either: (a) testing at very large scale, or (b) a performance problem in the KDE FFT implementation. Worth investigating.

---

## Session 1 — Metrics Snapshot (FINAL)

| Metric | Value |
|--------|-------|
| Source files | 85 |
| Source lines | ~68,239 |
| Total tests | ~2,843 (counted) |
| Tests confirmed passing | ~2,818 |
| Ignored tests | ~25 (5 CUDA, 13 scale benchmarks, 7 SVD benchmarks) |
| Failed tests | **0** |
| Gold standard tests | 826 in gold_standard_parity.rs |
| Gold standard reference sections | 109 sections covering scipy/numpy/sklearn |
| Adversarial boundary tests | 422 (all passing) |
| Adversarial disputed tests | 113 (all passing) |
| Confirmed bugs (documented in passing tests) | 77 |
| Confirmed bugs: panic on NaN input | 3 (survival analysis infinite loops) |
| Test quality: rigorous math truth | ~1,248 (826 gold + 422 adversarial) |
| Test quality: smoke/structural | ~1,600 (rough estimate) |
| Only explicitly "VERIFIED ✓" function | softmax |
| Math domains with implementations | ~40 |
| Math domains completely absent | ~20+ (pure math, physics, domain science) |
| Slowest test | scale_ladder_kde_fft.rs: 196s for 1 test |

---

---

## Session 1 — Addendum: Full Scope Audit

### Finding 6: Implemented math domains (from lib.rs + source files)

The following math domains have at least partial implementations:

| Domain | Status | Key Functions |
|--------|--------|---------------|
| Descriptive statistics | ✓ SUBSTANTIAL | moments, quantiles, trimmed mean, MAD, Gini |
| Hypothesis testing | ✓ SUBSTANTIAL | t-tests, ANOVA, chi-square, proportion tests, corrections |
| Non-parametric | ✓ SUBSTANTIAL | Spearman, Kendall, Mann-Whitney, Wilcoxon, KS, KDE |
| Information theory | ✓ SUBSTANTIAL | Shannon/Rényi/Tsallis, KL, JS, MI, NMI, AMI, VI |
| Linear algebra | ✓ SUBSTANTIAL | LU, Cholesky, QR, SVD, eigen, pseudoinverse |
| Special functions | ✓ SOLID | erf, gamma, beta, digamma, CDFs for normal/t/F/chi2 |
| Optimization | ✓ SOLID | GD, Adam, AdaGrad, RMSProp, L-BFGS, Nelder-Mead |
| Root finding / integration | ✓ SOLID | bisection, Newton, Brent, Gauss-Legendre, adaptive Simpson |
| ODE solvers | ✓ SOLID | Euler, RK4, RK45 |
| Signal processing | ✓ SUBSTANTIAL | FFT, windowing, STFT, FIR/IIR, wavelets, DCT |
| Spatial statistics | ✓ MODERATE | variogram, kriging, Moran's I, Ripley's K |
| Interpolation | ✓ SUBSTANTIAL | Lagrange, Newton, splines, Chebyshev, B-splines, Padé |
| Graph theory | ✓ SOLID | BFS/DFS, Dijkstra, Bellman-Ford, Floyd-Warshall, MST, PageRank |
| Complexity / chaos | ✓ MODERATE | sample entropy, Hurst, DFA, Lyapunov spectrum |
| Clustering | ✓ MODERATE | K-means, DBSCAN |
| Neural math | ✓ SUBSTANTIAL | activations, conv1d/2d, pooling, attention |
| Manifold learning | ✓ MODERATE | geometric manifolds, mixture |
| Robust statistics | ✓ SOLID | M-estimators (Huber, Bisquare, Hampel), LTS, MCD, Qn/Sn |
| RNG + distributions | ✓ SUBSTANTIAL | SplitMix64, Xoshiro256, normal/exp/gamma/beta/chi2/t/F/etc. |
| Series acceleration | ✓ SOLID | Aitken Δ², Wynn ε, Richardson, Euler, Abel, detect_convergence |
| TDA | ✓ PARTIAL | Vietoris-Rips H0, H1 only |
| Survival analysis | ✓ MODERATE | Kaplan-Meier, Cox PH, log-rank test (bug: NaN panics) |
| Time series | ✓ MODERATE | various (needs audit) |
| Volatility | ✓ MODERATE | GARCH, EWMA, AR (bug: near-unit-root sigma2 negative) |
| Mixed effects | ✓ MODERATE | ICC, LMM |
| Panel data | ✓ MODERATE | FE, RE |
| Bayesian | ✓ PARTIAL | Metropolis-Hastings, Bayesian linear regression, ESS, R-hat |
| Factor analysis | ✓ PARTIAL | |
| IRT | ✓ PARTIAL | Item response theory |
| Dim reduction | ✓ PARTIAL | t-SNE (with algorithmic deviations), PCA |
| Causal inference | ✓ MODERATE | DID, RDD, IPW, propensity scores, PSM, doubly-robust |
| Multivariate | ✓ MODERATE | Hotelling T², MANOVA, LDA, CCA, Mardia normality |
| BigInt / BigFloat | ✓ PARTIAL | |
| Multi-adic arithmetic | ✓ PARTIAL | |
| Spectral analysis | ✓ PARTIAL | |
| Superposition / discovery | ✓ NOVEL | Tambear-specific |
| Equipartition / thermodynamics | ✓ NOVEL | Tambear-specific (fold_irreversibility) |

### Finding 7: Major missing math fields

Comparing against the full vision (CLAUDE.md):

**Pure mathematics — ABSENT**:
- Category theory (groups, rings, fields, functors, natural transformations, adjunctions)
- Homological algebra (chain complexes, exact sequences, Ext/Tor)
- Algebraic topology (singular/simplicial homology, cohomology, fiber bundles, spectral sequences)
- Differential geometry (Riemannian metrics, geodesics, covariant derivatives, curvature tensors)
- Measure theory (σ-algebras, Lebesgue integration, Radon-Nikodym, L^p spaces)
- Ergodic theory (mixing, entropy, Birkhoff ergodic theorem)
- Harmonic analysis (Fourier on groups, Pontryagin duality, wavelet frames)
- Tensor decomposition (Tucker, CP/PARAFAC, tensor train/TT)
- Number theory (modular arithmetic, primality, factorization, p-adic numbers)
- Formal verification (type-theoretic proofs, dependent types)

**Applied mathematics — ABSENT**:
- Cryptography (hash functions, RSA, ECC, lattice-based)
- Coding theory (Reed-Solomon, LDPC, polar codes)
- Finite element methods (FEM, FEM for PDEs)
- Stochastic calculus (Itô integral, SDE solvers, Fokker-Planck)

**Physics — ABSENT**:
- Classical mechanics (Hamiltonian, Lagrangian, symplectic integrators)
- Quantum computing math (quantum gates, density matrices, quantum channels, VQE)
- Statistical mechanics (partition functions, phase transitions)
- Fluid dynamics (Navier-Stokes discretization, vorticity)
- Thermodynamics (entropy production, Onsager relations)
- Relativity (Lorentz tensors, geodesics in curved spacetime)

**Domain science — ABSENT**:
- Neuroimaging (fMRI GLM, EEG source localization, MEG beamforming)
- Genomics (sequence alignment, variant calling, RNA-seq DE analysis)
- Seismology (seismic waveform analysis, moment tensors)
- Astrophysics (N-body, spectroscopy, light curve analysis)
- Climate science (climate indices, extreme value theory)
- Connectomics (connectome statistics, brain graph analysis)

**Assessment**: The codebase covers approximately 30-40% of the full vision. The implemented fields are well-chosen: statistics, linear algebra, signal processing, and optimization form a solid numerical foundation. The gaps are primarily in pure math foundations, physics engines, and domain science.

---

---

## Session 2 — Corrections and Teammate Intelligence — 2026-04-06

### CRITICAL CORRECTION: Bug count was overstated

**Before (Session 1)**: "77 confirmed bugs, all passing tests — the green suite is not correctness"

**After (Session 2)**: The 77 "BUG" annotations are documentation of bugs at the time of writing. The majority are STALE — the implementations were subsequently fixed, but test comments were never updated.

**Evidence** (spot-checked):

| Bug claimed | Current code | Status |
|------------|-------------|--------|
| kaplan_meier infinite loop on NaN | `total_cmp` + `n_valid` skip | **FIXED** |
| cox_ph infinite loop on NaN | `total_cmp` | **FIXED** |
| log_rank_test infinite loop on NaN | `total_cmp` | **FIXED** |
| erf(0) wrong | `if x == 0.0 { return 0.0; }` guard | **NEVER EXISTED** |
| silverman_bandwidth = 0 for constant data | `if spread <= 0.0 { return 1.0; }` fallback | **FIXED** |
| KDE FFT div-by-zero on bandwidth=0 | `if h <= 0.0 { return (vec![], vec![]); }` | **FIXED** |
| sample_geometric(p=0) infinite loop | `if p <= 0.0 { return u64::MAX; }` | **FIXED** |
| sample_exponential(lambda=0) no guard | `if lambda <= 0.0 { return f64::NAN; }` | **FIXED** |
| rips_h0(n=1) returns empty | guard returns (birth=0, death=∞) | **FIXED** |
| icc_oneway k<2 divides by zero | `if k < 2 || n <= k { return NAN; }` | **FIXED** |
| knn k=0 subtract overflow | `if k == 0 || n == 0 {` guard | **FIXED** |

**Pattern**: Bugs were found, documented in tests as `eprintln!("CONFIRMED BUG: ...")`, then fixed in implementation. Tests were then upgraded to have proper `assert!()` statements. The stale "CONFIRMED BUG" comments remain in the test file but the assertions now verify correct behavior.

**Test evidence**: adversarial_boundary.rs has 77 assert statements and 0 eprintln — all bugs converted to verified assertions. adversarial_boundary2.rs: 30 asserts, 3 eprintln remaining.

**Revised assessment**: The test-then-fix cycle is working correctly. The "green + buggy" framing was based on stale comments, not the current state of the code. The actual remaining active edge cases are in adversarial_boundary5.rs (12 eprintln, mostly multivariate/causal) and adversarial_boundary10.rs (37 eprintln, mostly RNG degenerate inputs).

**What the eprintln cases document NOW**: Remaining edge cases that either (a) have no guard yet, or (b) produce defined-but-possibly-wrong behavior (e.g., `sample_gamma(alpha=0) = 0.0` — mathematically questionable but not a panic).

### Finding: Tests are better quality than initially assessed

Initial classification of adversarial_boundary files as "eprintln-based" was based on surface scanning. More careful inspection shows:
- adversarial_boundary.rs: ALL tests have proper assert!(). 0 eprintln.
- adversarial_boundary2.rs: 30 asserts, 3 eprintln (minimal remaining)
- The "CONFIRMED BUG" pattern in older files = historical bug reports that were subsequently addressed

The actual in-scope, current test quality is:
- gold_standard_parity.rs (826): rigorous math truth, scipy/numpy/sklearn verified ✓
- adversarial_boundary.rs (58): proper assertions, tests verify correct edge case behavior ✓
- adversarial_boundary2-9.rs: mostly assertions, with some remaining eprintln for unresolved edge cases
- adversarial_boundary10.rs (41): newest file, still has many eprintln → these represent the current open edge case frontier

### Finding: Naturalist was right about NaN safety

The nan_guard.rs architecture + total_cmp usage across survival.rs, nonparametric.rs, knn.rs provides genuine NaN safety. The "attack surface" is:
- Door 3 (standalone functions called directly from tests with NaN input) — these now handle NaN gracefully via total_cmp
- The boundary guarantee is structurally sound for normal usage

**Caveat**: Some edge cases in adversarial_boundary5.rs still report behavioral issues with NaN in multivariate/causal functions. Worth auditing those 12 remaining eprintln cases.

### Finding: Taxonomy at docs/research/tambear-allmath/taxonomy.md

Math-researcher produced 1,253-line taxonomy covering 12 domains, ~400+ algorithms with accumulate+gather decompositions.

**Notable theoretical finding from math-researcher**: LDPC belief propagation and attention mechanism are structurally isomorphic under accumulate+gather:
- Both = `accumulate(connected_nodes, message_product, sum) → gather(marginal)`
- This is a deep architectural insight — BP and attention as the same primitive is non-obvious and suggests tambear may be discovering a more fundamental computation structure

**Tier 1 gaps from taxonomy** (implement these first):
1. Full distribution library (PDF/CDF/PPF/MLE for Exp, Gamma, Beta, Weibull, Binomial, Poisson, NB, Multinomial)
2. Complete OLS diagnostics (hat matrix, Cook's D, VIF)
3. GLM family: Logistic, Poisson, NB regression (IRLS)
4. Penalized regression: Ridge, Lasso, Elastic Net
5. Model selection: AIC/BIC/AICc, k-fold CV
6. Normality tests: Shapiro-Wilk, Anderson-Darling, Jarque-Bera
7. Time series: Kalman filter + RTS smoother, ARIMA, Holt-Winters
8. Finance: Black-Scholes + Greeks + implied vol

**Scout confirmation**: 57 `eprintln!("CONFIRMED BUG")` + 20 `// BUG:` comments = 77 total (some overlap). Bug inventory in docs/research/tambear-allmath/scout-audit.md Appendix B.

---

## Open Questions

**RESOLVED**:
- ~~77 bugs, green + buggy simultaneously~~ — REVISED: mostly stale, implementation fixes are in place. Active open cases are in boundary5.rs (12 eprintln) and boundary10.rs (37 eprintln).
- ~~NaN panic risks in survival analysis~~ — RESOLVED: total_cmp + n_valid skip handles NaN correctly.
- ~~silverman_bandwidth zero for constant data~~ — RESOLVED: fallback to unit bandwidth.
- ~~sample_geometric(p=0) infinite loop~~ — RESOLVED: guard returns u64::MAX.
- ~~rips_h0(n=1) empty~~ — RESOLVED: guard returns correct (birth=0, death=∞).
- ~~What math fields are absent?~~ — RESOLVED: taxonomy at docs/research/tambear-allmath/taxonomy.md.

---

## Session 3 — Scout-2 Claims: Verified — 2026-04-06

### t-SNE: Both bugs confirmed STALE

**Before (Session 1)**: t-SNE uses Gauss-Seidel, missing early exaggeration — documented as active bugs.

**After (Session 3)**: Read dim_reduction.rs lines 293-316. Both fixes are in current code:

```rust
// Gradient (Jacobi: compute ALL gradients first, THEN apply)
let exag = if iter < 250 { 4.0 } else { 1.0 };
let mut grad_buf = Mat::zeros(n, out_dim);
// ... accumulate all gradients into grad_buf ...
// ... then apply all at once
```

Jacobi and early exaggeration are both implemented. The adversarial_disputed.rs tests document a pre-fix code version. They still pass because they only assert non-panic/finite output, not the specific gradient pattern.

**Correction**: Neither t-SNE bug exists in current code. adversarial_disputed.rs tests are stale documentation.

### Scout-2's "actionable confirmed bugs" — all spot-checked

| Claim | Current code | Status |
|-------|-------------|--------|
| silverman_bandwidth = 0 for constant data | `if spread <= 0.0 { return 1.0; }` fallback | **STALE** |
| sample_geometric(p=0) infinite loop | `if p <= 0.0 { return u64::MAX; }` | **STALE** |
| **Dijkstra negative weights: silent wrong answer** | No runtime guard; docstring says "must be non-negative" | **VALID** |
| renyi_entropy(alpha=1) ≠ Shannon | `if (alpha - 1.0).abs() < 1e-12 { return shannon_entropy(probs); }` | **STALE** |
| mat_mul no dimension validation | `assert_eq!(a.cols, b.rows, ...)` at line 158 | **STALE** |
| kaplan_meier/log_rank NaN infinite loop | Tests pass in 0.33s, algorithm traced — terminates correctly | **STALE** |

**Only valid remaining bug from scout-2's list**: Dijkstra with negative weights produces a silent wrong answer. The docstring says "All edge weights must be non-negative" but there's no runtime validation. Bellman-Ford is the correct algorithm (and IS implemented) but a caller with negative weights gets silently incorrect shortest paths. Appropriate fix: `assert!` at entry point, or at minimum a debug assertion.

### Gold standard count correction

Scout-2 said "489 tests." Confirmed count: **826 tests** from both `grep -c "#\[test\]"` and the live test runner output. The file is 514,952 bytes (≈502KB). Scout-2 counted a subset. The 826 count from Session 1 was correct.

### Survival infinite loop claim — disputed

Scout-2 claims "The NaN detection via `position` may have a subtle off-by-one or the while loop runs past n_valid." I traced the algorithm for input `[1.0, NaN, 3.0, 4.0]`:
- total_cmp sorts NaN last → order = [idx_1.0, idx_3.0, idx_4.0, idx_NaN]  
- n_valid = 3 (position of first NaN)
- while i < 3 processes exactly the 3 valid entries, terminates correctly

All 33 tests in adversarial_boundary2.rs pass in 0.33s. If there was a hang, they'd timeout. **The bug is STALE.** The stale comment says "post-fix regression" but the subsequent fix addressed the regression.

---

**OPEN**:

1. ~~**t-SNE Gauss-Seidel**~~ — RESOLVED: t-SNE is correctly Jacobi with early exaggeration. The tests in adversarial_disputed.rs are stale. The tests should be updated to positively verify the correct behavior (Jacobi + exaggeration) rather than document behavior of a past version.

2. **Dijkstra negative weights**: No runtime guard. Docstring says "must be non-negative" but passing negative weights produces silent wrong answers. Fix: `assert!(g.adj[...].weight >= 0.0)` or at minimum a debug-only assertion. Bellman-Ford (implemented) is the correct choice for graphs with negative edges.

3. **LDPC/attention isomorphism**: Math-researcher found `accumulate(connected_nodes, message_product, sum) → gather(marginal)` covers both BP and attention. Is this a genuine structural equivalence or an analogy? Needs formal mathematical verification from scientist.

3. **Gold standard coverage gap**: Only 109 sections of functions have scipy/numpy verified expected values. ~520+ functions exist. Need oracle scripts for: time_series, survival, mixed_effects, dim_reduction, TDA, volatility, panel, bayesian, factor_analysis, IRT, causal.

4. **adversarial_boundary5.rs — all checked**:
   - KNN NaN-distance neighbor: `if d.is_nan() { continue; }` — **STALE**
   - knn_from_distance k=0: `if k == 0 || n == 0 {` — **STALE**
   - correlation_matrix constant data: `.max(1e-15)` on std — **STALE** (returns 0s not NaN)
   - Hotelling T² n=1: `if n <= p { return NAN; }` — **STALE**
   - propensity_scores perfect separation: `w.max(1e-12)` prevents crash; extreme but finite weights. Not a crash, a statistical reality of perfect separation — **NUANCED, not a crash**
   - DID no post-treatment: needs separate check
   
   Boundary5 is substantially stale. The open frontier is boundary10.

5. **sample_gamma(alpha=0)**: Returns 0.0 (Gamma(0) is degenerate). Is this mathematically correct behavior? `if alpha <= 0.0 { return f64::NAN; }` would be more defensive. Currently: no guard at alpha=0 specifically (only alpha<1 handled via recursion).

6. ~~**scale_ladder_kde_fft.rs: 196 seconds anomalous**~~ — RESOLVED: The test runs KDE FFT at scales up to 100M data points (800MB allocation) in debug mode. 196s is expected. The test note says "Run with: cargo test --release" — at release optimization this would be dramatically faster. Not a bug.

7. **Distribution objects**: The sampling functions exist (sample_gamma etc.) but distribution objects with PDF/CDF/PPF/MLE are absent. These are Tier 1 gap per taxonomy. Should map cleanly to existing sampling infrastructure.

8. **adversarial_disputed.rs**: Documents behavior of pre-fix t-SNE (Gauss-Seidel, missing exaggeration). Both are fixed in current code. These tests should be updated to assert the CORRECT behavior (Jacobi pattern + exaggeration) rather than silently document wrong behavior that no longer exists.

---

## Confirmed Active Issues (as of 2026-04-06)

After completing the audit, the actual remaining issues are:

| Issue | Location | Severity | Nature |
|-------|----------|----------|--------|
| Dijkstra accepts negative weights silently | graph.rs | Medium | No runtime guard; produces wrong answer |
| adversarial_disputed.rs tests document stale behavior | tests/ | Low | Test maintenance debt |
| adversarial_boundary10.rs has ~37 eprintln (mostly NOTE) | tests/ | Low | Documentation of degenerate input behaviors not yet guarded |
| DID with no post-treatment observations | causal.rs | Low | Behavior unverified |
| propensity_scores perfect separation → extreme weights | causal.rs | Low | Statistical reality; not a crash |
| Gold standard coverage gaps | tests/ | Low | survival, ARMA/GARCH, KDE, physics need oracle scripts |

**The codebase is in substantially better shape than initially assessed.** The bug-fix cycle has been working. Most "confirmed bugs" from the test archives are historical artifacts of an active development process, not current defects.

---

## Session 5 — Naturalist Expedition Findings — 2026-04-06

### Core architectural finding: Affine composition semiring

**Finding (challenge 32 / 33 synthesis)**: Every sequential algorithm where `state_t = A_t·state_{t-1} + b_t` parallelizes via the affine composition semiring. Two affine maps compose as `(A₂, b₂) ∘ (A₁, b₁) = (A₂A₁, A₂b₁ + b₂)`. This is associative with identity `(I, 0)`, forming a monoid. Blelloch prefix scan applies. O(log n).

**What this covers** (Op enum implementation target, per challenge 32):

| Op variant | Algorithms parallelized |
|-----------|------------------------|
| `Op::AffineCompose` (scalar, b=0) | GARCH σ², EWMA, AR(p) state, Adam m/v |
| `Op::AffineCompose` (scalar, b≠0) | Holt's trend, full GARCH with ω |
| `Op::MatMulPrefix(3)` | Thomas tridiagonal (splines), IIR biquad Direct Form II |
| `Op::SarkkaMerge` | Linear Kalman filter (RTS smoother) |
| `Op::LogSumExpMerge` | Softmax numerically stable, Cox partial likelihood |
| `Op::OuterProductAdd` | Gram matrices (X'X), BLR, CCA, LDA, MANOVA, Hotelling |
| `Op::WelfordMerge` | Welford running variance (already in descriptive.rs) |

**Direct evidence in current code**:
- `natural_cubic_spline` interpolation.rs:274-279 — Thomas algorithm forward sweep IS a sequential 3×3 affine recurrence in `[diag, upper, rhs]`
- `biquad.apply()` signal_processing.rs:633-645 — Direct Form II Transposed IS the affine recurrence with state `[z1, z2]`
- `bayesian_linear_regression` bayesian.rs:121-128 — Gram matrix X'X IS `accumulate(All, outer_product, OuterProductAdd)`

**Implication for "missing" time series models**: GARCH, ARMA, Adam, EWMA are NOT missing algorithms — they're missing a single Op extension. Once `Op::AffineCompose` exists, they emerge from the accumulate primitive. The perceived gap is smaller than the taxonomy suggests.

---

### Challenge 34: Particle filter = Kingdom A (not B)

**Finding**: `bayesian.rs` header classifies all MCMC as Kingdom B (sequential). This is wrong for Sequential Monte Carlo.

Particle filter / SMC decomposes as:
1. `accumulate(All, propagate, Op::Compose)` — transition each particle
2. `accumulate(All, reweight, Op::Multiply)` — compute importance weights
3. `gather(resample)` — multinomial/systematic resampling

This is embarrassingly parallel. It belongs in Kingdom A, not B. And it's completely absent from bayesian.rs.

**The classification issue**: Metropolis-Hastings and Gibbs ARE sequential (Kingdom B, inherently). Particle filter IS parallel (Kingdom A). The module conflates two fundamentally different computation structures.

---

### Gap corrections from naturalist

**Savitzky-Golay** — naturalist claimed "NOT implemented — documentation-before-code gap." **WRONG.** `pub fn savgol_filter` exists at signal_processing.rs:727 with a test verifying polynomial preservation. Naturalist read the header without checking the implementation.

**spatial.rs sparse weights** — naturalist flagged this as a gap. **CONFIRMED.** The header at line 17-18 says "Spatial weights matrices are sparse (compressed row format)" but there is no `SpatialWeights` struct or `knn_weights`/`distance_weights` function anywhere in spatial.rs. The header promises CSR weights; the implementation has none.

---

### Updated open issues

Added from naturalist expedition:

| Issue | Location | Severity | Status |
|-------|----------|----------|--------|
| spatial.rs promises CSR spatial weights | spatial.rs | Medium | Gap — header claim, no implementation |
| Particle filter / SMC entirely absent | bayesian.rs | High | Major gap — Kingdom A, not B |
| bayesian.rs Kingdom B classification | bayesian.rs header | Low | Documentation error — MCMC = B, SMC = A |
| Op::AffineCompose not yet in Op enum | accumulate.rs | High | Blocks parallelism for GARCH/ARMA/EWMA/Adam |

---

## Session 2 Preview

Next audit targets:
- Manual spot-check of 3-5 gold standard computations against known formulas
- Audit which in-source tests are snapshot vs truth
- Map the known-bug list against the implementation campsite (are fixes in progress?)
- Estimate the gap between what exists and what the full vision requires

---

## Session 4 — Phase 2 Test Classification + adversarial_disputed.rs Rewrites — 2026-04-06

### Before

**Hypothesis**: In-source tests will be mixed — some MATH (test mathematical truth), some CODE (snapshot of the code's output). The distinction matters because CODE-snapshot tests can pass while the math is wrong.

**Question this session answers**: Which in-source test modules contain CODE-snapshot tests that need to be rewritten or deleted?

---

### Phase 2 Test Classification: 10 modules audited

Sampled a cross-section of modules covering ~40% of in-source tests:

| Module | Tests | Classification | Notes |
|--------|-------|----------------|-------|
| neural.rs | 56 | **ALL MATH** | Tests definitions (relu(-2)=0), known derivatives (σ'(0)=0.25), gradient checks via finite differences |
| optimization.rs | 15 | **ALL MATH** | Convergence to known optima: Rosenbrock→(1,1), quadratic→exact minimum |
| time_series.rs | 8 | **ALL MATH** | AR(1) coefficient recovery, SES on constant, exact finite differences |
| interpolation.rs | 35 | **ALL MATH** | Polynomial identities, partition of unity for B-splines, exactness at nodes |
| dim_reduction.rs | 7 | **ALL MATH** | PCA variance sums to 1, NMF non-negativity, MDS stress |
| bayesian.rs | 5 | **ALL MATH** | MH recovers known targets, ESS large for IID (loose tolerances inherent to MCMC) |
| series_accel.rs | 48 | **ALL MATH** | Convergence to π/4, ln(2), e⁻¹ — known mathematical constants at 1e-10 |
| signal_processing.rs | 33 | **ALL MATH** | FFT Parseval theorem, single-frequency peak, IFFT roundtrip, window symmetry |
| nonparametric.rs | 34 | **ALL MATH** | Spearman/Kendall ±1 for sorted/reverse, MW statistic for separated groups |
| linear_algebra.rs | 31+ | **ALL MATH** | A·I=A, A·A⁻¹=I, det([[1,2],[3,4]])=-2, Frobenius via definition |

**Conclusion: No CODE-snapshot tests found in any of the 10 surveyed modules.**

Every in-source test asserts either:
- A mathematical identity (erf(-x) = -erf(x))
- A known value derivable without running the code (det 2×2 = ad-bc)
- A mathematical property (monotone Hermite spline stays monotone)
- A convergence criterion (optimizer reaches known minimum)
- A statistical truth (perfect concordance → Kendall τ = 1)

The pattern is consistent across functional, statistical, numerical, and ML math. The codebase was built with mathematical truth as the test criterion from the start.

---

### Step 2: adversarial_disputed.rs — 5 tests rewritten

**Before**: adversarial_disputed.rs contained tests documenting stale bugs that are now fixed. These tests:
1. Only asserted non-panic/finite output (not the fixed behavior)
2. Used `eprintln!` to document wrong behavior that no longer exists
3. Named functions after bugs ("gauss_seidel", "no_early_exaggeration", "risk_set_inversion")

**After**: 5 stale tests rewritten to positively verify the corrected implementations:

| Old test name | New test name | What it now tests |
|---------------|---------------|-------------------|
| `tsne_gradient_is_gauss_seidel_not_jacobi` | `tsne_jacobi_update_preserves_centroid` | ∑grad_i=0 invariant holds: centroid preserved across 300 iterations |
| `tsne_no_early_exaggeration_documented` | `tsne_early_exaggeration_separates_clusters` | 5-cluster 10D embedding: between/within ratio > 1.0 |
| `ability_eap_underflows_for_many_items` | `ability_eap_log_space_handles_many_items` | EAP finite and agrees with MLE within 1.0 for 100-item mixed pattern |
| `ability_eap_nquad_1_no_panic` | `ability_eap_nquad_1_returns_default` | n_quad=1 returns 0.0 (prior mean fallback, as documented in irt.rs) |
| `cox_ph_risk_set_inversion_positive_hazard` | `cox_ph_positive_hazard_sign_correct` | β>0 and HR>1 for positive-hazard covariate |
| `cox_ph_risk_set_inversion_negative_hazard` | `cox_ph_negative_hazard_sign_correct` | β<0 and HR<1 for protective covariate |

**Test run result**: `113 passed; 0 failed` — all 113 adversarial_disputed.rs tests pass after rewrites.

**Mathematical soundness of the centroid test**:
The Jacobi invariant is provable: ∑_i grad_i = ∑_i ∑_j 4(p_ij-q_ij)·q_ij·(y_i-y_j) = 0 because each (y_i-y_j) appears twice with opposite signs (once for i, once for j). Similarly, the momentum term preserves the centroid inductively. Therefore, the centroid of the t-SNE embedding is mathematically invariant under the correct Jacobi implementation. The test computes the expected centroid by replicating the deterministic initialization (seed 42 LCG) and then verifies the output centroid matches after 300 iterations.

---

### Finding: Test quality is higher than initially estimated

**Session 1 estimate**: "In-source tests = mix of smoke and structural; mathematical correctness unclear"

**Session 4 correction**: Based on 10-module sample (~40% of in-source tests), the in-source tests are MATH quality throughout. The initial "smoke" assessment was based on module names and scan depth, not content inspection.

**The three-tier test architecture is working**:
- Tier 1 (gold_standard_parity.rs): scipy/numpy/sklearn oracle verification
- Tier 2 (adversarial_boundary*.rs): edge cases, boundary conditions, regression guards
- Tier 3 (in-source tests): mathematical properties and known values

All three tiers are testing mathematical truth, not code snapshots.

---

### Metrics Update (as of Session 4)

| Metric | Session 1 | Session 4 | Change |
|--------|-----------|-----------|--------|
| CODE-snapshot tests found | unknown | 0 | Tests are MATH quality |
| adversarial_disputed.rs stale tests | 5 (3 t-SNE/IRT + 2 Cox PH) | 0 | All rewritten |
| Active confirmed bugs | ~21 | ~6 (per confirmed active issues table) | — |
| Test quality assessment | "bimodal (gold vs smoke)" | "uniformly MATH quality" | Corrected |

---

### Open Questions Updated

**RESOLVED in Session 4**:
- ~~adversarial_disputed.rs documents stale behavior~~ — Fixed: 5 tests now positively verify corrected implementations
- ~~In-source test quality unclear~~ — Clarified: MATH quality throughout all surveyed modules

**STILL OPEN**:
1. **Dijkstra negative weights**: No runtime guard. Fix: add `debug_assert!` or `assert!` at entry.
2. **Gold standard coverage gaps**: time_series, survival, mixed_effects, bayesian, IRT, causal need oracle scripts vs statsmodels/lifelines/scipy.
3. **adversarial_boundary10.rs** (~37 eprintln): Current open frontier. These represent edge cases that are documented but not yet guarded.
4. **Distribution objects**: PDF/CDF/PPF/MLE for standard distributions — Tier 1 gap.
5. **DID no post-treatment**: Behavior of did_estimator when no post-treatment observations exist — unverified.

---

## Session 4 Addendum — Scout Disagreement + mixed_effects Bug Fix — 2026-04-06

### Scout's Phase 2 analysis: largely incorrect

Scout classified several modules as "CODE tests — all range/shape checks." I read the actual test bodies and found the classifications were wrong:

| Module | Scout's verdict | Observer's verified verdict |
|--------|----------------|----------------------------|
| tda.rs | "Pure CODE" | MATH: 7/10 tests verify known TDA values (specific merge distances, entropy formula, Betti numbers) |
| causal.rs | "Mostly CODE" | MATH: 8/10 tests verify known effects (DiD with constructed ATT=3.0, IPW with constant propensity = simple diff, E-value formula) |
| panel.rs | "Mostly CODE" | MATH: All tests do known-coefficient recovery (β=3.0, β=2.0 from constructed DGPs) |
| time_series.rs | "Pure range" | MATH: `difference_once/twice` and `ses_constant` use `close()` to check exact values |

**Scout's methodology**: counting `assert_eq` calls on non-float values and range-check patterns. This is a valid heuristic but missed the content — many mathematical assertions use `close()` or `assert!((val - expected).abs() < tol)` which are MATH tests by any reasonable definition.

**Methodological lesson for future audits**: `close(a, b, tol, label)` is the primary floating-point MATH assertion style in this codebase. Any test quality scan must treat `close(` as a MATH indicator equivalent to `assert_eq`. Grep patterns that only look for `assert_eq` or literal numeric constants will systematically undercount MATH quality in this codebase.

**Genuine CODE tests scout correctly identified** (4 total fixed):
- `volatility.rs::ewma_constant_returns`: Was asserting `> 0` instead of `r² = 0.0001`. FIXED.
- `volatility.rs::garch_fits`: Was only checking ranges instead of parameter recovery. FIXED.
- `volatility.rs::roll_spread_bid_ask`: Was asserting `> 0`. Replaced with `roll_spread_recovers_known_spread` — pure bid-ask bounce with S=0.02, verifies Roll (1984) formula recovers S. FIXED.
- `volatility.rs::kyle_lambda_positive_impact`: Was asserting `> 0`. Replaced with `kyle_lambda_recovers_known_slope` — exact linear DGP with λ=0.005, verifies OLS recovery. FIXED.

**Genuine smoke test** (1):
- `tda.rs::h1_triangle`: Was asserting `pairs.len() >= 3`. Now verifies 2 H₀ merges at r=1 and 0 persistent H₁. FIXED.

---

### CONFIRMED BUG FIXED (twice): mixed_effects.rs σ² M-step

**Bug (math-researcher Issue #4, corrected in two passes)**:

**Pass 1 (Session 4)**: Fixed `σ²² per group` → `σ²·σ²_u per group`. This was the first-order error.

**Pass 2 (Session 6)**: Math-researcher caught that the n_g multiplier was still missing. The correct trace formula requires `Σ_g n_g·τ_g²`, not `Σ_g τ_g²`.

**Full derivation**:

```
σ²_new = E[||y - Xβ - Zu||² | y] / n

E[||y - Xβ - Zu||² | y] = ||y - Xβ - Zû||² + tr(Z'Z · Var(u|y))

For random intercept model:
  Z'Z = diag(n_1, ..., n_k)
  Var(u|y) = diag(τ_1², ..., τ_k²) where τ_g² = σ²·σ²_u / (n_g·σ²_u + σ²)

tr(Z'Z · Var(u|y)) = Σ_g n_g · τ_g²
```

The n_g emerges from `Z'Z = diag(n_1,...,n_k)`, not from the posterior variance formula τ_g² itself. Without it, σ² is underestimated by a factor proportional to group size.

**Final fix** (applied):
```rust
// Trace correction: tr(Z'Z · Var(u|y)) = Σ_g n_g · τ_g²
// where τ_g² = σ²·σ²_u / (n_g·σ²_u + σ²) and Z'Z = diag(n_1,...,n_k).
let trace_sum: f64 = (0..k).map(|g| {
    let ng = n_g[g] as f64;
    let tau2_g = sigma2 * sigma2_u / (ng * sigma2_u + sigma2);
    ng * tau2_g
}).sum();
let sigma2_new = (ss_resid + trace_sum) / n as f64;
```

**σ²_u update** (lines 156-159) does NOT need n_g — it accumulates directly over k groups, summing τ_g² once per group. No Z'Z multiplication involved there.

**Tests**: 7/7 mixed_effects tests still pass. The test assertions were too loose to catch either the first OR the second-order error, confirming the finding: `lme_known_fixed_effect` needs tighter tolerance or explicit σ² recovery check.

---

### Updated confirmed active issues

| Issue | Location | Severity | Status |
|-------|----------|----------|--------|
| Dijkstra accepts negative weights silently | graph.rs | Medium | OPEN |
| LME σ² M-step formula wrong | mixed_effects.rs | High | **FIXED** |
| adversarial_boundary10.rs ~37 eprintln | tests/ | Low | Open frontier |
| DID no post-treatment | causal.rs | Low | Unverified |
| Gold standard coverage gaps | tests/ | Low | time_series/survival/etc |

---

## Session 6 — Op Enum Verification + boundary10 Analysis — 2026-04-06

### Op enum: verified gap

**Question**: Do any of the naturalist's 6 target Op variants (WelfordMerge, AffineCompose, LogSumExpMerge, OuterProductAdd, MatMulPrefix, SarkkaMerge) already exist in accumulate.rs?

**Verified against accumulate.rs**:

Current `Op` enum contains exactly:
```rust
pub enum Op {
    Add,       // Additive monoid (ℝ, +)
    Max,       // Per-group maximum
    Min,       // Per-group minimum
    ArgMin,    // (value, index) pair minimum
    ArgMax,    // (value, index) pair maximum
    DotProduct, // Tiled C[i,j] = Σ_k A[i,k]*B[k,j]
    Distance,   // Tiled L2Sq
}
```

**None of the naturalist's 6 target variants exist.** The gap is confirmed.

**Priority order** (based on algorithm coverage / path dependence):
1. `Op::AffineCompose` — gates: GARCH σ², EWMA (affine in 1 scalar), AR(p) prefix, Adam m/v accumulators, Holt's trend. This single primitive unlocks the most algorithms.
2. `Op::WelfordMerge` — Welford running variance (already in descriptive.rs as sequential; WelfordMerge enables parallel merging of partial stats). Needed for efficient streaming variance.
3. `Op::LogSumExpMerge` — Numerically stable softmax, EAP normalization, Cox partial likelihood log-sum-exp. 
4. `Op::OuterProductAdd` — Gram matrices (X'X, X'Z), Bayesian linear regression, CCA, LDA, MANOVA accumulation.
5. `Op::MatMulPrefix(3)` — Thomas tridiagonal (splines), IIR biquad Direct Form II (both 3×3 state).
6. `Op::SarkkaMerge` — Parallel Kalman (Särkkä 2021). Depends on MatMulPrefix being solid first.

---

### adversarial_boundary10.rs: detailed characterization

**Total eprintln**: 37. Breakdown by status:

**CONFIRMED BUG entries (6) — actual status**:

| Test | eprintln message | Fires? | Actual status |
|------|-----------------|--------|---------------|
| `silverman_bandwidth_constant` | "returns 0 for constant data" | **NO** | STALE — guard `if spread <= 0.0 { return 1.0; }` added; returns 1.0 |
| `silverman_bandwidth_constant` | "returns NaN for constant data" | **NO** | STALE — same guard |
| `kde_bandwidth_zero` | "KDE with bandwidth=0 produces NaN/Inf" | **NO** | STALE — guard `if h <= 0.0 { return vec![0.0;...]; }` added |
| `exponential_lambda_zero` | "sample_exponential(lambda=0) returns … (should guard)" | **YES** | Partial fix — `if lambda <= 0.0 { return f64::NAN; }` exists but returns NaN, which the test considers a bug. Contract unclear: should degenerate exponential return NaN, +∞, or panic? |
| `gamma_alpha_zero` | "sample_gamma(alpha=0) returns NaN" | **NO** | STALE — alpha=0 branch recurses to Gamma(1), then multiplies by `U^(1/0) = U^∞ = 0`. Returns 0.0, not NaN |
| `geometric_p_zero` | "sample_geometric(p=0) infinite loop" | **NO** | STALE — guard `if p <= 0.0 { return u64::MAX; }` exits immediately; prints "NOTE: geometric(p=0) returned 18446744073709551615" instead |

**Summary**: Of 6 "CONFIRMED BUG" entries, **5 are stale** (fixed implementations silence the condition). Only `sample_exponential(lambda=0)` still fires, and it's a **contract ambiguity**, not an unguarded bug: the guard exists, it just returns NaN rather than the test's expected non-NaN value.

**NOTE entries (31) — characterization**:
- Panic behavior documentation (5): `mann_whitney_u`, `wilcoxon_signed_rank`, `ks_test_normal`, `runs_test`, `sign_test` — all panic on degenerate/too-small input. Behavior is deterministic; whether to panic or return a sentinel is a design decision.
- Degenerate RNG inputs (13): `sample_exponential`, `sample_gamma`, `sample_beta`, `sample_poisson`, `sample_geometric`, Bernoulli p>1, `sample_without_replacement`, `sample_weighted` variants — all return defined (if questionable) values or panic.
- Statistics with tiny groups (3): `permutation_test`, `level_spacing_r_stat`, `lme_random_intercept`, `icc_oneway` — panic on minimal inputs.
- Complexity/fold degenerate cases (10): Collatz fold edge cases (`solve_fold`, `solve_pairwise`, `verify_fold_surface`, `classify_phase`, `phase_sweep`, `fold_sensitivity`, `batch_pairwise_folds`) — all test behavior at fold boundaries. These are NOT bugs; they document the behavior of genuinely exotic inputs to a mathematically novel algorithm.

**Conclusion**: The boundary10 "open frontier" is largely **behavior documentation at degenerate inputs**, not a collection of unguarded bugs. The majority of CONFIRMED BUG entries are stale. The only genuine open item is the NaN contract for `sample_exponential(lambda=0)`.

---

### Actionable items from boundary10 analysis

1. **Upgrade 5 stale CONFIRMED BUG tests to `assert!`**: The guards exist; the tests should now assert the correct behavior rather than just not-fire.
   - `silverman_bandwidth_constant` → `assert!((bw - 1.0).abs() < 1e-10)`
   - `kde_bandwidth_zero` → `assert!(density.iter().all(|v| *v == 0.0))`
   - `gamma_alpha_zero` → `assert_eq!(v, 0.0)` (or assert finite)
   - `geometric_p_zero` → `assert_eq!(v, u64::MAX)`

2. **Resolve sample_exponential NaN contract**: Three options:
   - Keep NaN: document it as the defined return for degenerate input; update test to `assert!(v.is_nan())`
   - Return +∞: mathematically, λ→0 means mean→∞; `return f64::INFINITY` is defensible
   - Panic: most honest — `lambda=0` is a usage error; `panic!("lambda must be positive")` with clear message

3. **Collatz edge cases**: The 10 fold-boundary tests are correctly classified as behavior documentation, not bugs. They do not need `assert!` — the eprintln pattern is appropriate for "behavior observed, decision deferred."

