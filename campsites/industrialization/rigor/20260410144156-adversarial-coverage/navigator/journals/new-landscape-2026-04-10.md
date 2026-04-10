# New Landscape — 2026-04-10 (Navigator Logbook)

## Team-lead seeded 16 new campsites from the day's discoveries.

Full count: 52 campsites now. Significant expansion. Here's the navigator's
reading of what each means for the expedition.

---

## Architecture Campsites — Actionable Now

### op-identity-method (HIGH PRIORITY)
Math-researcher has the full design spec at:
`industrialization/architecture/20260410164316-op-identity-method/math-researcher/notebooks/identity-degenerate-spec.md`

The spec is complete. Every Op needs two methods: `identity()` (monoid element,
for padding) and `degenerate()` (failure signal, for empty/invalid input).
7 bugs already found empirically from conflating these. One-time implementation
cost, permanent correctness benefit.

**Who:** pathmaker (it's a scan engine contract change).
**What:** Add `identity()` + `degenerate()` to Op enum, update all scan engines
to use `op.identity()` for padding, update empty-input returns to use `op.degenerate()`.
**Dependency:** None. This is self-contained.

### nan-propagating-minmax (IN PROGRESS)
Scout's audit is complete. The systemic pattern is documented.
We've been fixing these one wave at a time (waves 11-17).
The audit campsite has the full remaining list.

**Navigator recommendation:** Adversarial agent should pull the remaining
locations from the audit and write wave 18 targeting them all at once, rather
than one per wave. The pattern is fully understood; the remaining fixes are
mechanical.

### pinv-relative-rcond (DONE)
Fixed in ff13e63. Campsite can be closed.

### ema-is-kingdom-a (SPECCED)
Scout wrote the spec. When `ema_period` gets implemented: use affine map
parallel prefix, not a sequential loop. The Fock boundary dissolves.

### using-two-scopes (DESIGN RESOLVED)
The open question from Aristotle is answered. Math-researcher + naturalist +
aristotle converged:
- **Consumed** at Level 0-1 (primitives): each primitive grabs its own params
- **Persistent** at Level 2-3 (pipelines): analysis-level consistency
- **Step-specific**: `step.with(n_bins=50)` for one-step overrides
- **Pipeline-level**: `using(alpha=0.01)` for all hypothesis tests in that analysis

This resolves the design blocker for task #2. Pathmaker can proceed.

### graph-grouping (SPECCED)
9th grouping pattern — Laplacian algebra, GNN workload. No notes yet beyond
the creation file. Math-researcher should develop.

---

## Theory Campsites — Publishable Claims

These four are paper-grade discoveries. Each is a seed only — no development yet.

### classification-bijection
Claim: algorithm classes (Kingdom A/B/C/D) = sharing clusters (which algorithms
share which intermediates). One structure, not two. If true, the kingdom taxonomy
IS the sharing taxonomy — a single coherent theory.

**Who should develop:** Aristotle. The bijection proof or counterexample.

### k7-scheduling-bound
Claim: CovMatrix between 7 columns is the clique K_7. K_7 determines the
scheduling bound. Memory constraint and error-correction distance are aspects
of the same combinatorial structure.

**Who should develop:** Math-researcher. The scheduling bound derivation.

### holographic-error-correction (TESTABLE NOW)
Claim: `.discover()` implements holographic error correction in the epistemic sense.
`view_agreement` drops when intermediates are corrupted. Code distance = number
of views. Running everything IS the correctness property.

**The testable experiment:**
1. Take a dataset and run `discover_correlation(col_x=0, col_y=1)`
2. Artificially corrupt the Pearson view's intermediate (inject bit-flip into covariance)
3. Measure: does `view_agreement` drop? Does the corrupted view stand out?
4. If yes: the claim is confirmed experimentally.

This is the most immediately testable of the four theory claims.
**Who should develop:** Scientist. Design the corruption experiment. Run it.

### five-atomic-groupings
Claim: 5 atomic grouping patterns generate 15 composite patterns; 4 gaps are
predicted by the theory and not yet found in practice. The theory predicts
which composite groupings CAN exist.

**Who should develop:** Aristotle + math-researcher. Gap-filling and proof.

---

## Ideas/Research Campsites — Future Landscape

### gray-scott-garch-rhyme
(F,k) space in Gray-Scott ↔ (α+β, leverage) space in GJR-GARCH. Both have
bifurcation boundaries — one produces Turing patterns, the other volatility
clustering. The structural rhyme suggests a common mathematical object.

**Navigator note:** This is beautiful but speculative. Develop carefully.
The rhyme is real. Whether it's the SAME mathematical object is the question.

### riemann-zero-statistical-portrait
The prime distribution looks like eigenvalue statistics of random matrices
(GUE). Riemann zeros follow the same distribution. A "statistical portrait"
using tambear primitives is achievable now. This would be a publishable
computation — not a proof of RH, but a numerical confirmation of the
Montgomery-Odlyzko law at precision no prior computation has reached.

**Who should develop:** Math-researcher. Use tambear's existing spectral
and statistical primitives. The computation itself may already be possible.

---

## Cross-Pollination Notes

The `using-two-scopes` decision unblocks task #2. Pathmaker should read that
campsite before starting the wiring work.

The `op-identity-method` spec is the architectural fix that prevents the entire
class of bugs adversarial waves 13-17 have been finding. Implementing it makes
future waves shorter (the scan engine itself would reject wrong padding).

The holographic error correction claim is TESTABLE and the test requires only
existing primitives + the `discover()` infrastructure that's already wired.
This is the clearest path to a publishable result from today's theory work.

---

## What I'd Direct Each Agent

**Pathmaker:** Read `using-two-scopes`. Then implement `op-identity-method`.
Then proceed with task #2 (using() wiring) using the resolved design.

**Adversarial:** Write wave 18 targeting all remaining NaN-eating instances
from the nan-audit campsite. One comprehensive wave instead of one per session.

**Scientist:** Design and run the holographic error-correction experiment.
It's testable now. This is the clearest path from insight to publication.

**Math-researcher:** Develop the classification-bijection theory. If the
kingdom taxonomy IS the sharing taxonomy, that's a fundamental architectural
insight that should be in a paper.

**Aristotle:** First-principles deconstruction of the five-atomic-groupings
claim. Do 5 atoms really generate exactly 15 composites? Are the 4 gaps
structurally forced or coincidental?

**Naturalist/Scout:** The Riemann zero portrait is an open field. Start mapping
which tambear primitives would be needed and whether the computation is feasible.
