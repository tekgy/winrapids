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

# Universality Class Experiment: GOE vs GUE vs Poisson

*Scout, 2026-04-10 (from math-researcher's message during hold)*

## The discriminating experiment

**Question**: Do market eigenvalue spacings follow GOE (expected for real correlation matrices)
or GUE (zeta zeros' universality class)?

**Why this matters**: Finding GUE statistics in market eigenvalues would imply hidden
complex structure — effective Fourier/phase structure in returns that makes the correlation
matrix behave like a complex Hermitian ensemble rather than a real symmetric one.

## The Wigner surmise CDFs (the primitives needed)

**GOE** (β=1, real symmetric / Wishart ensemble):
- Density: p(s) = (π/2)·s·exp(-πs²/4)
- CDF: P(s) = 1 - exp(-πs²/4)
- Repulsion: LINEAR — p(s) ~ s as s → 0

**GUE** (β=2, complex Hermitian / Montgomery's conjecture for zeta zeros):
- Density: p(s) = (32/π²)·s²·exp(-4s²/π)
- CDF: P(s) = 1 - (1 + 8s²/π)·exp(-4s²/π)   ← CLOSED FORM (math-researcher confirmed)
- Repulsion: QUADRATIC — p(s) ~ s² as s → 0
- Note: exact GUE (Fredholm determinant of sine kernel) differs from surmise by ~1% in bulk,
  ~2% in tails. For n~1000 zeros/eigenvalues, sampling uncertainty (~3%) exceeds this — use
  surmise. Exact GUE only needed for n > 10^6 (Odlyzko-scale experiments).

**Poisson** (no level repulsion / random, uncorrelated):
- Density: p(s) = exp(-s)
- CDF: P(s) = 1 - exp(-s)
- Repulsion: NONE — p(s) → 1 as s → 0

## Why r-statistic can't discriminate GOE from GUE

The r-statistic (ratio of consecutive spacings) has means:
- Poisson: 0.386
- GOE: 0.530
- GUE: 0.536

GOE and GUE means differ by only 0.006 — indistinguishable with any realistic sample.
The KS test against the Wigner surmise CDF IS discriminating: the s vs s² small-spacing
behavior is measurable from the spacing histogram shape.

## The theoretical expectations

**Market eigenvalues** (sample correlation matrix C = X^T X / n, X real):
- Wishart ensemble → GOE universality class
- DEFAULT EXPECTATION: GOE
- Finding GUE → implies hidden complex structure in returns

**Riemann zeta zeros** (nontrivial zeros on critical line):
- Montgomery's pair correlation conjecture (1973) → GUE
- Odlyzko's numerical verification confirms GUE to high precision
- Symmetry is unitary, not orthogonal → GUE, not GOE

## The experiment steps

1. Compute spacing distribution of bulk market eigenvalues (after Marchenko-Pastur unfolding)
2. KS test against GOE Wigner surmise CDF (expected)
3. KS test against GUE Wigner surmise CDF (zeta zeros' class)
4. KS test against Poisson (null: no level repulsion)
5. Compute r-statistic for qualitative check (though can't separate GOE/GUE)
6. Compare to zeta zeros spacing distribution under same three tests

**Publishable either way**:
- Market = GOE, zeta = GUE: structural rhyme holds at r-stat level but breaks at
  spacing distribution level. Different universality classes despite similar appearance.
- Market = GUE: genuinely surprising. Implies effective complex structure. New finding.
- Market = Poisson: no level repulsion in market eigenvalues. Also interesting.

## The missing primitive

`ks_test_custom(empirical: &[f64], theoretical_cdf: impl Fn(f64) -> f64) -> KsResult`

We have `ks_test_normal`. The gap is an arbitrary reference CDF parameter.

The Wigner surmise CDFs as first-class primitives:
```rust
pub fn wigner_surmise_goe_cdf(s: f64) -> f64  // 1 - exp(-π·s²/4)
pub fn wigner_surmise_gue_cdf(s: f64) -> f64  // 1 - (1 + 8s²/π)·exp(-4s²/π)  [closed form]
pub fn poisson_spacing_cdf(s: f64) -> f64      // 1 - exp(-s)
```

These are standalone primitives that compose with `ks_test_custom` to produce
the three-reference-distribution framing.

## Connection to the classification bijection

This experiment tests whether the universality class of a system can be read
from its eigenvalue spacing distribution. If market eigenvalues are GOE and zeta
zeros are GUE, the r-statistic (which can't separate them) is a coarser invariant
than the spacing distribution (which can).

This is the same structure as the Kingdom classification: the r-statistic is like
the Kingdom label (coarse), while the spacing distribution is like the (grouping, op,
semiring) triple (fine). Universality class is the fine invariant.


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

