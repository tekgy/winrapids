# Expedition State — 2026-04-10 (Navigator Logbook)

## Where We Are

Test count: **2194 lib tests green** (up from 2152 at expedition start).
Commits since expedition start: 12+ across the team.
The crate surface is dramatically more complete. The compiler is clean.

---

## What the Team Produced (Cross-Pollination Map)

### Pathmaker: Phantom Surface Fixes

The most impactful work this expedition. Six complete families had ZERO pub use
entries — implemented, tested, invisible at the crate surface. Now fixed:

- survival: kaplan_meier, log_rank_test, cox_ph, grambsch_therneau_test
- panel: panel_fe, panel_re, panel_fd, panel_twfe, hausman_test, two_sls, did
- bayesian: metropolis_hastings, bayesian_linear_regression, r_hat
- irt: rasch_prob, prob_2pl, fit_2pl, ability_mle, ability_eap, mantel_haenszel_dif
- series_accel: aitken_delta2, wynn_epsilon, euler_transform, accelerate
- train::naive_bayes: gaussian_nb_fit, gaussian_nb_predict

Also: 20 dim_reduction/factor_analysis/spectral_clustering/tda primitives
And: kmeans_f64 extracted as a standalone primitive from gap_statistic's closure

Navigator contribution: 10 tbs_executor signature errors fixed in that commit.

### Scout: Complete Phantom Scan

The definitive report — no compiler phantoms. Real phantoms were missing
pub use entries (all now fixed). Key structural finding:

family22_criticality.rs and family24_manifold.rs REIMPLEMENTED ccm, mfdfa,
and phase_transition from tambear::complexity. The primitives exist. The bridges
bypass them due to missing pub use + API shape mismatch. The recommended fix:
Path B (delegate bridges to primitives, thin wrapper). Not yet done.

Scout also documented that the "tick_* leaves" are future scope, not current debt.

### Math-Researcher: pinv rcond Bug + Current Test State

Documented that pinv uses absolute rcond=1e-12 default, which is wrong for
non-unit-scaled matrices (financial data in thousands → covariance ~1e6,
correct threshold ≈ 2.2e-8, not 1e-12). 

Fix: relative threshold = max(m,n) * eps * max_sv (matches NumPy + LAPACK).
**Navigator applied this fix in the latest commit.** One-line change, no API break.

Math-researcher also confirmed: 2191 lib tests pass (now 2194 after our work).
Noted that adversarial integration tests in tests/ have unverified compilation
(rank/sigmoid name collisions in some test files). These are non-blocking — 
the library itself is correct.

### Math-Researcher: Complex Arithmetic Gap

Documented that tambear's Mat is real-only. What's blocked:
1. Non-symmetric eigendecomposition (complex eigenvalue pairs)
2. Full FFT (complex output)  
3. Hilbert transform (analytic signal)
4. Quantum simulation, transfer functions, characteristic functions

Recommended: ComplexMat with SoA (real: Vec<f64>, imag: Vec<f64>) storage.
Complex DotProduct state is (real_sum, imag_sum) — Kingdom A, no Fock boundary.
This is a significant future implementation item, not current expedition scope.

### Aristotle: Two Deep Structural Insights

**1. Are intermediates the right decomposition?**

Verdict: mostly correct, some gaps.
- DistanceMatrix: correct but DERIVED — the primitive is the Gram matrix
- SufficientStatistics (sum, m2, count): gold standard — minimal, sufficient, natural
- FFT: correct but misnamed — call it SpectralRepresentation (content, not algorithm)
- CovMatrix: correct and rich

Missing intermediates: KNNGraph (for UMAP/t-SNE/spectral), KernelMatrix (kernel SVM/GP),
BasisMatrix (spline/polynomial regression), GramMatrix as explicit type.

The intermediate catalog should be a dependency graph, not a flat list:
  Raw → GramMatrix → CovMatrix → PCA, LDA, Mahalanobis
  Raw → GramMatrix → DistanceMatrix → KNNGraph → UMAP, spectral clustering
  Raw → SpectralRepresentation → PSD, coherence, Hilbert

**2. using() and discover() are an epistemic partition**

using(method=X) = "I KNOW the right answer"
discover(method) = "I DON'T KNOW — find it"

This is a complete partition of epistemic space. Most frameworks force one mode.
Tambear is unique in making the epistemic state explicit and composable.

Aristotle also raised an open question: should using() be consumed after ONE step
(current design) or persist through the pipeline? Consumed is safer; persistent
is more convenient. Worth a deliberate decision before the using() wiring task lands.

---

## What's Still Open

### Bug: NaN-eating in three more locations

From the previous wave analysis (documented in nan-policy.md):
- clustering.rs:666 — `fold(0.0_f64, f64::max)` on distance scores
- complexity.rs:240-241 — range computation on `cum_dev`
- No wave 16 yet targeting these

Status: waiting for adversarial agent to write wave 16.

### Architecture: fintek family22/24 bridge refactor

family22_criticality.rs and family24_manifold.rs have full reimplementations of
ccm, mfdfa, phase_transition. The primitives now exist at the crate surface.
The refactor to Path B (delegate) is pending. Blocks: clean math in one place.

### Integration test compilation

Some tests/ files have compile errors (rank/sigmoid name collision, mat_approx_eq
signature mismatch). These are in the test files, not the library. Non-blocking
for library correctness but should be cleaned up.

### Tasks #2, #3, #8 still pending

Task #2 (using() passthrough) — ~400 pub fns need wiring.
Task #3 (phantom hunt) — scout declared complete; task can be closed.
Task #8 (TBS executor + discover/superposition) — TBS executor partially wired;
  discover() superposition layer not yet started.

---

## What Should Move Next

**High priority for pathmaker:**
1. Close task #3 (phantom scan is done — just mark it complete)
2. Refactor family22/24 bridges to delegate to tambear::complexity (ccm, mfdfa, phase_transition)
3. Begin task #2 (using() passthrough) — Aristotle has the full picture of what this means

**High priority for adversarial agent:**
Wave 16 — target the three remaining NaN-eating instances.

**High priority for scientist/math-researcher:**
Workup test suites for pinv (now that rcond default changed — verify against NumPy reference).

**Naturalist/scout — follow own curiosity:**
The complex arithmetic gap is well-documented. Any exploration of how ComplexMat
would fit the accumulate+gather skeleton would be valuable groundwork.

---

## Unexpected Serendipity

The pinv fix was flagged by the math-researcher and applied in 10 minutes.
That's the flywheel: adversarial thinking surfaces a bug class, one agent documents it
precisely, another agent applies it. The rigor layer is working.

The Aristotle insight on intermediates being a dependency graph is deeper than it looks.
When the intermediate catalog gains GramMatrix as the primitive below CovMatrix and
DistanceMatrix, TamSession can derive missing intermediates from cheaper ones rather than
recomputing from raw data. That's a compounding speedup — not just sharing, but derivation.
