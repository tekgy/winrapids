<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

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


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

