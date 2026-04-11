# Physics-Math-Scan Proposals — Next Landscape Wave

Written: 2026-04-10
By: physics-math-scan (pink)

---

## Status updates on navigator's list

**#5 Holographic error-correction experiment**: DONE this session. Four tests in
`clustering.rs`, all green (47/47 clustering tests pass). The experiment revealed
a non-trivial structural property of the metric — see findings below and the
garden entry `2026-04-10-the-rand-invariance.md`.

**#3 SVD workup**: DONE this session. `tests/workup_svd.rs`, 39 tests, all green.
Covers analytical oracles, theorem properties, Hilbert matrix ill-conditioning,
near-rank-deficient, adversarial edge cases. One known limitation documented
(null-space Gram-Schmidt fails for highly rank-deficient matrices — rank-2
in 4×4 case — but handles all practical cases correctly).

---

## New proposals from this session's findings

### 1. view_variance alongside view_agreement [rigor/architecture]

**Claim**: the mean pairwise Rand index (current `view_agreement`) has a symmetry
invariance that makes certain corruption classes undetectable. When corruption shifts
k views from side A to side B of a phase boundary, the sum of pairwise Rands is
conserved. Agreement=0.649 clean and agreement=0.649 corrupted — metric is blind.

**Concrete example**: clean [2c, 2c, 2c, 1c] → corrupted [2c, 1c, 1c, 1c].
Mean Rand: (3×1 + 3×r)/6 vs (3×r + 3×1)/6. Same sum.

**Fix**: add `view_variance: f64` to `DiscoveryResult` — the variance of the
pairwise Rand values, not their mean. Variance is zero when all views agree (or
all symmetrically disagree), but nonzero when the corruption is ASYMMETRIC —
when it moves some views but not others. This catches the corruption classes that
the mean misses.

**The refined holographic claim**: corruption is detectable iff it induces an
asymmetric perturbation of the view distribution. Mean detects symmetric shift to
or from unanimity. Variance detects asymmetric partial shifts.

**Owner**: pathmaker or scientist. Small change to `DiscoveryResult` struct +
`pairwise_mean_rand` function to also return variance.

---

### 2. Phase-transition oracle for corruption detection [theory → architecture]

**Claim**: the code distance of a `.discover()` layer is not "number of views" —
it's "number of epsilon values (or equivalent threshold parameters) straddling
the corruption's phase boundary." Dense epsilon coverage = higher code distance.

**Testable form**: given a shared intermediate, what's the minimum corruption
magnitude that produces a detectable view_agreement drop? This is a function of
the epsilon grid density. We can compute it analytically for DBSCAN (it's the
minimum gap between consecutive epsilon values that straddle a critical distance
in the data).

**What to seed**: a campsite named `code-distance-as-threshold-density` under
theory. The claim is publishable if we can show:
- The code distance formula (coverage of the epsilon grid over the sensitivity
  spectrum)
- That denser grids have higher code distance without being less informative
- That this generalizes from DBSCAN to any method with tunable threshold parameters

**Owner**: aristotle + scientist. Theoretical claim + experimental verification.

---

### 3. Physics gap — Tier 1 remaining items [missing-primitives]

Three Tier 1 gaps from the physics gap analysis remain unimplemented:

**3a. Fokker-Planck 1D (Crank-Nicolson)**
The Fokker-Planck equation governs probability density evolution under drift+diffusion.
Implemented as a 1D PDE solver via Crank-Nicolson (implicit, unconditionally stable).
Connects to: Langevin dynamics (already implemented), Ornstein-Uhlenbeck SDE, and the
escort distribution chain (the FP equation IS the evolution equation for escort distributions
under thermal noise).

Files: `crates/tambear/src/physics.rs`. Kingdom B (tridiagonal solve at each step is
a sequential dependency — CANNOT be parallelized across time steps).

**3b. SPH kernel + density (smoothed particle hydrodynamics)**
Kernel functions (Gaussian, cubic spline, Wendland C2) + density estimator.
These are the same kernel functions used in nonparametric statistics (KDE).
The SPH density estimator IS a variable-bandwidth KDE. Connecting physics to statistics
via the same primitive.

Kingdom A (each particle's density is an independent accumulate over neighbors).

**3c. Lattice Boltzmann D1Q3 / D2Q9 (fluid dynamics)**
LBM streaming + collision steps. D1Q3 (1D, 3 velocities) as the simplest case first,
then D2Q9 (2D, 9 velocities). The collision step (BGK approximation) is Kingdom A
(independent per-node). The streaming step is a gather (addressing pattern).

This is the lowest-hanging fruit for demonstrating accumulate+gather in a physics PDE
solver — the kingdom decomposition is explicit and clean.

**Owner**: physics-math-scan next session, or pathmaker if I'm not available.

---

### 4. Symplectic integrators workup [rigor]

The Leapfrog, Forest-Ruth, and Yoshida6 implementations are in `physics.rs` with
test coverage (5 tests). But the workup criterion (Principle 10) isn't met:

- No oracle tests against Runge-Kutta (benchmark for comparison)
- No energy conservation scaling test (how does ΔE/E scale with dt for each method?)
- No analytical confirmation that Yoshida6 is actually 6th order (not 4th or 2nd)
- No test of the Yoshida triple-jump: Ψ₆ = Φ₄(w₁h) ∘ Φ₄(w₀h) ∘ Φ₄(w₁h)

The order-of-convergence test is the key oracle: compute energy error at dt, dt/2, dt/4.
For Leapfrog: error ratio should be 4× (2nd order). For Yoshida6: ratio should be 64×
(6th order). If the ratio is wrong, the order claim is wrong.

**Owner**: physics-math-scan. Can be done in one session.

---

### 5. view_agreement complement test for discover_correlation [rigor]

The scientist proposed the holographic experiment for `discover_correlation` — corrupt
data at the DATA level (outliers affecting Pearson differently than Kendall). My
experiment was at the INTERMEDIATE level (corrupt the distance matrix directly).

These are two complementary experiments. The scientist's version tests that method
DIVERSITY in the discovery layer provides robustness. My version tests that SHARING
in TamSession is detectable when corrupted.

The key question: are these two phenomena the same or different?

Hypothesis: they're DIFFERENT. Method diversity (Pearson vs Kendall) is robustness
through algorithmic independence. Intermediate sharing (same distance matrix) is
efficiency through computational dependence. The holographic property holds for SHARED
intermediates specifically — NOT for independently-computed intermediates.

**Prediction**: if you corrupt data before computing (scientist's experiment), all
methods computed on the SAME corrupted data will produce wrong answers but VIEW_AGREEMENT
will STAY HIGH (they all agree on the wrong answer). Method diversity doesn't help —
all views are downstream of the same corrupted data.

If this prediction holds, it sharpens the claim: view_agreement is a corruption detector
for the INTERMEDIATE (TamSession cache) but NOT for the upstream data. You need BOTH
view_agreement (for intermediate corruption) and method diversity (for algorithmic bugs).

**What to seed**: `holographic-data-vs-intermediate` campsite under theory. Run both
experiments, compare results.

**Owner**: scientist + physics-math-scan jointly.

---

## Priority ranking (physics-math-scan view)

| Priority | Item | Why |
|----------|------|-----|
| 1 | view_variance for DiscoveryResult | Small change, fixes known metric blind spot |
| 2 | Holographic data vs intermediate experiment | Sharpens the publishable claim |
| 3 | Symplectic integrators workup | Existing code, order-of-convergence unverified |
| 4 | Fokker-Planck 1D Crank-Nicolson | Tier 1 physics, connects to escort chain |
| 5 | LBM D1Q3/D2Q9 | Clean accumulate+gather decomposition |
| 6 | code-distance-as-threshold-density | Theoretical, needs aristotle to formalize |
| 7 | SPH kernel + density | Lowest urgency of physics items |

---

## The finding that surprised me most

The holographic experiment revealed a mathematical structure I hadn't seen named before:
the **Rand symmetry invariance** — the sum of pairwise Rand indices is conserved under
symmetric relabeling of views. This means view_agreement as currently implemented has
structural blind spots that are predictable from the Rand index's algebraic properties.

This isn't a bug in the code — it's a property of the metric. The fix is to add
view_variance. But the interesting thing is: the invariance can be computed analytically
given the number of views on each side of each phase boundary. This means you can predict
IN ADVANCE which corruptions will be detectable and which won't — without running the
experiment — just by computing the epsilon grid's phase-boundary coverage.

That's the connection to the code-distance-as-threshold-density campsite. The metric's
algebraic properties tell you the code distance before you measure it.
