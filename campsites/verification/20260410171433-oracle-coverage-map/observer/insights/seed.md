# Oracle Coverage Map — Seed Document

*2026-04-10 — observer*

---

## What This Is

A systematic inventory of the rigor tier for every primitive in tambear.
The sharing phyla map revealed WHERE sharing happens. The oracle coverage map
reveals WHAT KIND of verification each primitive has.

The map does not tell you if a primitive is correct. It tells you whether
a failure would be detected, and by what mechanism.

---

## The Four Tiers (Aristotle's formulation)

| Tier | Color | Definition | What it catches |
|------|-------|-----------|-----------------|
| Theorem-oracle | Green | Verified against mpmath/SymPy at 50+ digits | Wrong series coefficients, wrong normalization, wrong algorithm class |
| Theorem-algebraic | Green | Asserts mathematical necessity (not just reasonable output) | Wrong formula, wrong limiting behavior, violated invariant |
| Parity-oracle | Yellow | Compared against scipy/R/MATLAB/numpy (double-precision peer) | Wrong implementation relative to reference, large errors |
| Contract | Orange | Boundary conditions: NaN propagation, empty input, overflow | Missing guards, wrong IEEE 754 handling |
| Behavior | Red | Output is finite/reasonable for normal inputs | Crashes, gross errors |
| None | Black | No external tests for this primitive | Unknown |

A primitive can have multiple tiers simultaneously. The MAP entry records the
highest tier it currently satisfies, plus which tiers it has.

---

## Current State (measured 2026-04-10)

### Green — Theorem-oracle (mpmath at 50dp)

These have workup files with explicit mpmath citations:

| Primitive | File | Tests | Oracle | Bug history? |
|-----------|------|-------|--------|-------------|
| `erfc` | `workup_erfc.rs` | 21 | mpmath 50dp | YES — 2 boundary regressions found and fixed by oracle |
| `kendall_tau` | `workup_kendall_tau.rs` | 12 | mpmath + scipy | No |
| `pearson_r` | `workup_pearson_r.rs` | 18 | mpmath | No |
| `incomplete_beta` | `workup_incomplete_beta.rs` | 20 | mpmath | No |
| `inversion_count` | `workup_inversion_count.rs` | 18 | mpmath | No |
| `pinv` | `workup_pinv.rs` | 23 | mpmath | No (workup prompted rcond fix) |
| `normal_cdf` | `workup_normal_cdf.rs` | ? | scipy/statsmodels ULP scan | No |

**Also green (theorem-algebraic, no oracle file but asserted mathematically):**
- `matrix_exp`: `exp(t·I) = e^t·I` (theorem-test, caught Padé [6/6] bug)
- `kendall_tau`: `τ = 1` for monotone sequence (mathematical necessity)
- FFT: Parseval's theorem `Σ|x|² = (1/N)Σ|X|²` (in adversarial_wave13)
- `garch11_filter`: affine semigroup associativity (in volatility.rs, documented)

**Total green primitives: ~10** (out of ~400 pub fns, ~2.5%)

### Yellow — Parity-oracle (scipy/numpy peer comparison)

`gold_standard_parity.rs` — **471 test functions** comparing tambear against
pre-computed scipy/sklearn/numpy oracle values. This is the largest single
concentration of oracle verification in the codebase.

Also: `gold_standard_shapiro.rs` (14 tests), `gold_standard_prereq.rs`,
`gold_standard_posthoc.rs`, `gold_standard_phase8.rs`.

The gold_standard tests compare against scipy at double precision — sufficient
to catch wrong formulas, wrong normalizations, off-by-one errors, wrong degrees
of freedom. NOT sufficient to catch errors below scipy's own precision (i.e.,
bugs that scipy also has, or precision issues in the 10th+ decimal place).

**Estimated yellow primitives: ~80-120** (those covered by gold_standard_parity.rs,
which touches accumulate, descriptive, hypothesis, nonparametric families)

### Orange — Contract only (NaN/boundary, no oracle)

The adversarial wave tests (waves 1-19, ~300+ external tests) assert:
- NaN in → NaN out
- Empty input → NaN or defined fallback
- Edge cases don't produce Inf/crash

These are important but not oracle tests. They don't verify the math is right —
they verify the guards are right. A primitive can pass all contract tests and
still return a wrong answer for normal inputs.

**Estimated orange primitives: ~150-200**

### Red — Behavior only

Internal `#[cfg(test)]` tests that check:
- Output is finite for normal inputs
- Round-trip properties (not against external oracle)
- Self-consistency (not against external reference)

**Estimated red: ~50-100 primitives with only internal behavior tests**

### Black — No external tests

New primitives from waves 6-23 that have internal tests but no workup file
and are not covered by gold_standard_parity.

**Estimated black: ~50+ primitives** (exact count requires systematic scan)

---

## The Rigor Gap

The gap between theorem-oracle (green, ~10 primitives) and the full catalog
(~400 pub fns) is the rigor gap. Aristotle's formulation: this is WHERE
mathematical bugs can hide undetected.

**What the Padé finding proves:** `matrix_exp` had wrong Padé [6/6] coefficients.
The internal behavior tests (does it converge? is the result finite?) passed.
The gold_standard tests might have passed (scipy uses Padé too, so a mild
coefficient error might produce agreement at double precision). The theorem-test
`exp(0.5·I) = e^0.5·I` caught it because it's a mathematical necessity —
if the coefficients are wrong, this exact equality fails, regardless of what
scipy does.

**The map's purpose:** identify which primitives could have a Padé-class error
(wrong formula structure, wrong coefficient, wrong normalization) that would
pass all current tests but fail against a theorem-oracle.

---

## Proposed Next Steps

1. **Systematic scan**: for each pub fn in the catalog, assign a tier based on
   which external test files cover it. This produces the actual map, not estimates.

2. **Tier-upgrade targets**: primitives currently in orange/red that have
   closed-form theorem-tests available (e.g., `spearman_r` has `ρ = 1` for
   monotone as an algebraic theorem — a one-line test that upgrades it to green).

3. **mpmath workup queue**: primitives where the formula is complex enough that
   oracle parity at 50dp is the only way to catch coefficient errors.
   Candidates: `renyi_divergence`, `wasserstein_1d`, `energy_distance`, 
   `chi2_divergence`, `bhattacharyya_distance`.

4. **Dual-oracle principle**: for primitives where two independent oracles agree
   (mpmath + scipy both cited), the confidence is higher than either alone.
   The `kendall_tau` workup already does this. Promote the pattern.

---

## Connection to the Sharing Phyla Map

The sharing phyla map says: WHICH primitives share intermediates and when.
The oracle coverage map says: WHICH primitives have theorem-verified output.

Together they answer: when a TamSession sharing reuse happens (a downstream
primitive reads a cached intermediate), is the shared intermediate itself
verified? A shared intermediate that is only contract-tested propagates
orange-tier confidence into all downstream consumers.

This is the verification analog of the holographic error correction: multiple
independent theorem-tests of the same intermediate would catch corruption
that a single test misses.

---

## Status Note

This campsite was seeded by the observer (2026-04-10) based on measurements
from the current test corpus. The estimates (80-120 yellow, 150-200 orange,
etc.) need to be replaced with actual counts from a systematic scan.

The systematic scan is the work this campsite tracks.
