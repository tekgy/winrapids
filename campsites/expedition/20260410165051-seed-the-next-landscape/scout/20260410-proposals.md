# Scout's Next-Landscape Proposals

Written: 2026-04-10

## Priority order (scout's read): (3) → (5) → (4) → (2) → (1)
affine-prefix-scan is the foundation everything else builds on.

---

**1. `industrialization/architecture/tropical-semiring`**
Op::TropicalMinPlus as a new Op variant. PELT, Viterbi, all-pairs-shortest-paths
are Kingdom A in tropical semiring but labeled B — we lack the Op.
Once it exists, these admit GPU-parallel schedules via tropical matmul.
GPU shortest-path via tropical matmul already exists in literature.
Role: math-researcher (verify) + pathmaker (implement)

**2. `industrialization/architecture/kingdom-classification-audit`**
Systematic scan of all Kingdom B labels in codebase. GARCH filter, exponential
smoothing, PELT confirmed mislabeled. Audit produces corrections list +
reclassification spec. Mislabeled ops miss GPU parallelization.
Role: scout

**3. `industrialization/architecture/affine-prefix-scan`** ← FOUNDATION
`affine_prefix_scan(a: &[f64], b: &[f64], s0: f64) -> Vec<f64>` in signal_processing.rs.
EMA, GARCH filter, EWMA variance, all first-order IIR filters delegate to this.
GPU path = scan kernel replacing sequential loop.
Spec complete in EMA campsite (`campsites/industrialization/architecture/20260410164422-ema-is-kingdom-a/scout/insights/ema-kingdom-a-spec.md`).
Role: pathmaker

**4. `industrialization/sharing/gram-matrix-tier1`**
GramMatrix as Tier 1 intermediate in TamSession with derivation rules to
DistanceMatrix and CovarianceMatrix. Unlocks materializing-view pattern.
Spec: `~/.claude/garden/2026-04-10-intermediate-dependency-dag.md`
Requires: IntermediateTag::GramMatrix variant + derive_from() method.
Role: pathmaker

**5. `industrialization/architecture/garch-filter-extraction`** ← HIGH LEVERAGE
Extract `garch11_filter(returns, omega, alpha, beta) -> Vec<f64>` from garch11_fit
as standalone public primitive with correct Kingdom A classification.
MLE outer loop (Kingdom C) calls the filter (Kingdom A).
GPU benefit: 200 optimization iterations each call scan kernel vs sequential loop.
For tick-level data (millions of obs), this is the dominant cost.
Role: pathmaker (straightforward extraction, math done)
