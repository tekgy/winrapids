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

# Riemann Zero Statistical Portrait — Research Plan

## What Already Exists

`bigfloat.rs` already has:
- `zeta_complex(s)` — complex ζ(s) for s ≠ 1
- `hardy_z(t, prec)` — Z(t) function (real-valued on critical line)
- `find_zeta_zero(t_lo, t_hi, prec, tol)` — bisection zero finder
- `montgomery_odlyzko_r_statistic` test — finds ~30 zeros, computes r-stat
- `level_spacing_r_stat` — the SAME primitive used for market eigenvalue spacings

The structural rhyme is already verified: zeta zero spacings → GUE statistics ← 
market correlation eigenvalue spacings. Same primitive, different domains.

## The Full Statistical Portrait

Apply EVERY tambear complexity/information/spectral primitive to the zeta zero sequence,
treating it as a "time series" of heights on the critical line.

### Phase 1: Basic Statistics of Zero Spacings

Using normalized spacings δ_n = (t_{n+1} - t_n) × log(t_n)/(2π):

| Primitive | Expected result | What it tells us |
|---|---|---|
| `moments_ungrouped(spacings)` | mean≈1, variance≈0.42 | GUE predicts σ²≈0.42 |
| `permutation_entropy(spacings)` | Near max (close to random) | Level repulsion pattern |
| `sample_entropy(spacings)` | High (complex dynamics) | Predictability of gaps |
| `dfa(spacings)` | α ≈ 0.5 (uncorrelated) | Long-range correlations? |
| `hurst_rs(spacings)` | H ≈ 0.5 | Same question, different estimator |
| `acf(spacings, 50)` | Near zero for all lags | GUE predicts no autocorrelation |
| `spectral_flatness(psd)` | Near 1 (white spectrum) | Frequency structure |

### Phase 2: Distribution of Spacings

| Primitive | Expected result | Reference |
|---|---|---|
| `histogram_auto(spacings)` | Wigner surmise shape | p(s)=(π/2)s·exp(-πs²/4) |
| `ks_test_normal(spacings)` | REJECT (not Gaussian) | Spacings are Wigner, not normal |
| `shapiro_wilk(spacings)` | REJECT | Same |
| `anderson_darling(spacings)` | REJECT | Same |
| `kl_divergence(empirical, wigner)` | Small | Closeness to Wigner surmise |
| `wasserstein_1d(empirical, wigner_samples)` | Small | Same, metric version |

### Phase 3: Higher-Order Statistics

| Primitive | What it measures | Known result? |
|---|---|---|
| `mfdfa(spacings)` | Multifractal spectrum | Unknown — novel |
| `transfer_entropy(spacings, shifted)` | Directed information flow | Should be zero (stationarity) |
| `rqa(spacings)` | Recurrence structure | Unknown — novel |
| `correlation_dimension(spacings)` | Fractal dimension of attractor | Unknown — novel |
| `largest_lyapunov(spacings)` | Chaos indicator | Unknown — novel |

### Phase 4: Number-Theoretic Primitives

| Analysis | Primitive composition | What it shows |
|---|---|---|
| Pair correlation | histogram of t_j - t_i (all pairs, normalized) | Montgomery's conjecture: 1 - (sin πu/πu)² |
| k-point correlation | higher-order contingency tables | GUE k-point functions |
| Nearest-neighbor spacing distribution | sort(gaps), ecdf | Wigner surmise vs actual |
| Number variance | variance of N(t, t+L) as L varies | GUE predicts Σ²(L) ~ (2/π²)(log(2πL) + 1 + γ - π²/8) |

## What Makes This Publishable

1. **No one has applied the full complexity toolkit to zeta zeros.** Individual statistics
   (spacing distribution, pair correlation, number variance) have been computed by analytic
   number theorists. But the full battery — MFDFA, permutation entropy, sample entropy,
   RQA, DFA, Lyapunov exponents — has not been applied systematically.

2. **The structural rhyme is testable.** If zeta zero spacings and market eigenvalue spacings
   produce the SAME r-statistic (which the test already shows), do they also produce the
   same MFDFA spectrum? The same permutation entropy? The same RQA? If yes, the structural
   equivalence runs deeper than the r-statistic. If no, the specific differences tell us
   WHERE the analogy breaks.

3. **Tambear can do it.** Every primitive needed exists. The zeta zero finder exists.
   The portrait is a composition of existing primitives applied to a novel dataset.

## Computational Requirements

- 1000+ zeros: extend scanning range to t ~ 1000 (need Hardy Z function at higher t)
- Higher precision: current prec=64 sufficient for r-statistic, may need prec=128+ for
  higher-order correlations
- Runtime: ~30 zeros takes seconds; 1000 zeros at prec=128 might take minutes

## Next Steps

1. Extend `montgomery_odlyzko_r_statistic` to compute 1000+ zeros
2. Apply the Phase 1 primitives (moments, PE, SampEn, DFA, ACF)
3. Compare to GUE theoretical predictions
4. If results are clean, apply Phase 2-4
5. If the structural rhyme with market eigenvalues holds across all measures,
   write it up as a cross-domain universality result


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

