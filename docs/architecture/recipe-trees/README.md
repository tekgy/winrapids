# Recipe trees — the catalog-as-tree pattern

**Status**: Practice template introduced 2026-05-08, anchored on naturalist's
`~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md`.

## What this is

A directory of per-family catalog trees. Each file maps a literature-named
recipe family (means, distances, correlations, kernels, sketches, tail
estimators, etc.) onto its underlying kernel(s) + parameter axes.

The pattern is the naturalist's: *names are parameter assignments on a
graph; the graph is the actual catalog.* When you implement by name, you
build N implementations and discover later they share a kernel; when you
implement by graph, you build the kernel and let names attach to parameter
assignments.

## Why this directory exists separately from `adding-a-recipe.md`

`adding-a-recipe.md` is the operational template — *here is the file
layout for a single recipe.* This directory is the **structural** template
— *here is the catalog topology a recipe family lives inside.*

The two compose. When pathmaker adds a new recipe, the question
`adding-a-recipe.md` doesn't yet ask is: *is this an independent recipe
or a parameter assignment on an existing kernel?* The catalog tree for
the relevant family answers it. If a tree exists for the family, the
new recipe is mapped onto an existing branch + parameter assignment, or
extends the tree by surfacing a new axis. If no tree exists, the answer
defaults to "independent recipe" and the family is unmapped.

## How to use a tree

Three audiences, three uses:

**Catalog browser (user)**: walk the tree from root to the leaf they want.
Adjacent leaves are adjacent in parameter space — when one method doesn't
fit, the obvious next method to try is structurally next-to-it, not
alphabetically next-to-it. This is `discover()` from a different angle:
the relationship IS the answer.

**Recipe author (pathmaker)**: before implementing a new recipe, find
its position in the relevant tree. If the position is on an existing
kernel branch, the implementation is a parameter assignment + a recipe
wrapper (~20 lines), not a new kernel. If the position is a new axis,
extend the tree first; implement second.

**Reviewer (math-researcher / aristotle / observer)**: walk the tree
looking for unmapped literature variants (gaps the team hasn't built yet)
and unparameterized kernel-overlaps (places where two named recipes
should collapse into one kernel + two recipe wrappers).

## How to add a tree for a new family

1. **Identify the kernel(s).** Most families have one kernel; some have
   several. Three structurally distinct kernel topologies have surfaced
   so far, all valid:

   - **Overlapping kernels** (means is the canonical example) — multiple
     kernels reach the same literature-named leaves via different
     parameterizations. `geometric_mean` is reachable from
     `GeneralizedMean(p=0)` AND `TransformedMean(in=log, out=exp)`.
     This happens when the named transformations are reversible across
     kernels — different parameter spaces with shared fixed-points.

   - **Disjoint kernels** (sketches is the canonical example) — multiple
     kernels with no overlapping leaves. DDSketch is in
     `CompressedHistogram` only; KLL is in `RandomizedCompactor` only;
     GK is in `RankTuple` only. This happens when the underlying state
     shapes (bucket-grid vs randomized-multiset vs rank-tuple-list)
     don't share vocabulary — there's no parameterization in one kernel
     that produces what another kernel produces.

   - **Clustered kernels with embedding bridges** (distances is the
     canonical example) — kernels cluster into regions, with overlap
     *within* a cluster and disjointness *between* clusters; bridges
     across cluster boundaries exist as *embeddings*, not unifications.
     Distances resolves into a "geometric" cluster (MinkowskiNorm +
     InnerProductDistance + parts of Divergence) and a "combinatorial"
     cluster (SetDistance + SequenceEdit + GraphDistance). Bridges
     between them are parameter-dependent compositions of two kernels:
     spectral lift (graph → eigenvector coords → Lp), MinHash
     (set → sketch → Hamming), k-shingles (sequence → set → Jaccard).
     The bridge is a *recipe* that composes two kernels, not a new
     kernel that unifies them. When you spot this shape, preserve the
     cluster boundaries honestly — don't force fake unification just
     because some leaves can be embedded into another cluster's space.

   None of the three topologies is "right" — they're family-determined
   by the underlying mathematical structure. When you draft a new tree,
   check all three possibilities; don't force overlap when the family
   is genuinely disjoint, don't miss overlap when it's there, and
   don't collapse bridge-connected clusters into one unified kernel
   when the bridge is lossy and parameter-dependent.
2. **Identify the parameter axes** for each kernel. Each axis is a
   `using()` knob in the eventual recipe API.
3. **Map every literature-named variant** to a kernel + parameter
   assignment. Multiple-paths-to-same-leaf is structural information to
   surface, not a bug.
4. **Surface the gaps** — parameter combinations no literature has
   named but tambear should expose anyway, per anti-YAGNI.
5. **Surface the overlaps** — where two kernels produce the same
   answer for some leaves and different answers for others. These are
   the design questions that decide which kernel becomes "primary" in
   the implementation.
6. **Identify the accumulate+gather decomposition** for each kernel.
   The cleanest decomposition typically wins as the implementation
   target; the other kernels become recipe wrappers if the named-leaves
   set is wide enough to deserve the syntax.

7. **Identify composition patterns over the kernels.** Some families
   expose recipe-level structure that *isn't* itself a kernel — it's a
   parameter axis that composes existing kernels into derived recipes.
   Correlations surfaced two: a *multivariate axis* (CCA, partial-
   correlation matrix, multiple R²) and a *copula axis* (Spearman-from-
   copula, tail-dependence, Schweizer-Wolff σ). Each composition
   pattern wraps one or more kernels with additional structure
   (matrix-valued output, copula transformation) without introducing a
   new kernel. Surface them in the tree as a separate section after
   the per-kernel sections — they're real catalog structure, just at
   a different abstraction level than the kernels themselves.

## Structural patterns observed across trees

As the catalog grows, structural patterns surface that aren't visible
from any single tree. Name them here as they're observed; they
inform how future trees get drafted.

- **Synonym collapses** — single recipes hidden behind multiple
  literature names. The tree pattern makes them visible.
  Confirmed instances: `quartic_kernel ≡ biweight_kernel` (kernels);
  `rbf_kernel ≡ gaussian_pd_kernel ≡ squared_exponential` (kernels);
  `matern_1/2 ≡ laplace ≡ exponential_pd_kernel` (kernels —
  Matérn is the parent form; Laplace and RBF are fixed-ν children);
  `Wasserstein_p(empirical_1d) ≡ L_p(sort(x) - sort(y))` (distances).
  When you find one, document it explicitly — three names with one
  recipe is structural information worth preserving.

- **Sharing-graph hubs** — some families connect to most other trees
  via TamSession-shareable intermediates; some are leaves. Correlations
  is the richest hub observed so far (consumes means, distances,
  kernels, sketches; produces inputs to copulas, regression, linear-
  algebra). Hub families warrant earlier implementation because their
  shareable intermediates pay rent across many trees; leaf families
  can defer to consumer demand.

- **Cross-tree shared intermediates** — the same intermediate appears
  as the canonical shared state for multiple trees. Examples observed:
  pairwise-L2-distance matrix is universal across distances and
  StationaryPD kernels; rank-transformed values are shared by
  correlations (Spearman) and tail-estimators (order statistics);
  the 3-field moment trio is shared by means, distances, correlations.
  When you draft a new tree, check whether existing trees already
  own a recipe that should be your shared intermediate; consume via
  TamSession rather than re-implementing.

## Trees in this directory

- `means.md` — the centrality/central-tendency family (~30 named
  literature variants across 5 kernels). First pilot. Overlapping
  topology.
- `sketches.md` — quantile sketches family (DDSketch, KLL, GK,
  t-digest, ...) across 3 kernels. Second pilot. Disjoint topology.
- `kernels.md` — smoothing kernels + positive-definite kernels (~40
  literature names across 7 sub-kernels under 2 disjoint top-level
  kernel-of-kernels). Disjoint topology with shared functional-form
  vocabulary.
- `distances.md` — dissimilarity / metric / divergence family (~50+
  literature names across 6 kernels). Clustered topology with
  embedding bridges between geometric and combinatorial clusters.
- `correlations.md` — correlation family (~20+ literature names
  across 4 kernels). Mixed topology; MomentCorrelation is the
  catalog-collapse star. Surfaced composition patterns (multivariate,
  copula) as a new structural axis.

Future: `tail-estimators.md`, `dispersions.md`, `entropies.md`,
`divergences.md`, `information-criteria.md`, `clustering.md`,
`regression.md`, `factorizations.md`, `time-series.md`...

The catalog grows organically — one family at a time, ratified by
math-researcher, used by pathmaker on next-recipe-add to that family.
No team-wide deadline. The trees accumulate.
