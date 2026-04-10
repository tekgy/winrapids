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
- CDF: requires numerical integration (not closed form)
- Repulsion: QUADRATIC — p(s) ~ s² as s → 0

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
pub fn wigner_surmise_gue_cdf(s: f64) -> f64  // numerical integration of (32/π²)·s²·exp(-4s²/π)
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
