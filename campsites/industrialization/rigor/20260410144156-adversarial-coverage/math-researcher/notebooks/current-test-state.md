# Current Test State — 2026-04-10

## Library Tests: ALL PASS

```
cargo test --lib → 2191 passed, 0 failed, 5 ignored
Duration: ~170s
```

The library is clean. All 2191 inline tests pass.

## Integration Tests: NOT YET VERIFIED

The `tests/` directory has 45 test files including:
- 16 adversarial_* files (boundary, CI, correlations, density, disputed, etc.)
- 4 gold_standard_* files (parity, phase8, posthoc, prereq, shapiro)
- 5 scale_ladder_* files
- 4 workup_* files (kendall_tau, pearson_r, inversion_count, incomplete_beta)
- 1 svd_adversarial.rs

The compilation errors I initially observed (rank/sigmoid name collisions,
mat_approx_eq signature) may be in these integration test files. They would
need to be compiled separately: `cargo test --test <name>`.

## Earlier Compilation Error Analysis

The errors were:
1. `rank` name collision — E0252: likely a test file importing both
   `nonparametric::rank` and `linear_algebra::rank` (matrix rank)
2. `sigmoid` name collision — E0252: likely importing both
   `linear_algebra::sigmoid` and `special_functions::logistic`
3. `mat_approx_eq` takes 3 args now but tests call with 4 — the 4th arg
   was a message string that was removed from the helper function

These are in the integration tests, not the library. The library itself
compiles and passes all 2191 tests.

## Test Distribution by Module (approximate from grep)

| Module | Approx # tests | Notes |
|---|---|---|
| descriptive | ~80 | moments, quantile, forecast |
| nonparametric | ~200+ | rank, correlation, KS, SW, bootstrap, KDE |
| hypothesis | ~150+ | t-tests, ANOVA, chi2, effect sizes, power |
| linear_algebra | ~150+ | factorizations, solvers, regression |
| special_functions | ~200+ | distributions, orthogonal polynomials, Bessel |
| information_theory | ~100+ | entropy, MI, divergences |
| complexity | ~100+ | SampEn, DFA, Hurst, Lyapunov, RQA, MFDFA |
| time_series | ~200+ | AR, ARMA, ARIMA, ADF, KPSS, spectral, STL |
| volatility | ~100+ | GARCH variants, realized vol, microstructure |
| clustering | ~80+ | hierarchical, validation, gap |
| optimization | ~50+ | GD, Adam, L-BFGS, Nelder-Mead |
| numerical | ~100+ | root finding, integration, ODE |
| series_accel | ~80+ | Aitken, Wynn, Richardson, Euler |
| multivariate | ~80+ | covariance, Hotelling, MANOVA, LDA, CCA |
| robust | ~50+ | M-estimators, scale estimators, LTS, MCD |
| other | ~300+ | train, IRT, kalman, survival, graph, etc. |
