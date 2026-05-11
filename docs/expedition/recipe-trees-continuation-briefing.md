# Recipe trees continuation — briefing

**Status**: Ready to pick up. Briefed by main-thread Claude + Tekgy at the close of the 2026-05-08/09 session.

**One-line summary**: Continue the catalog-as-tree pattern for the next several recipe families. Two pilots shipped (`means.md` + `sketches.md`); the pattern + README live at `R:\winrapids\docs\architecture\recipe-trees\`. Next candidates: distances, correlations, kernels.

**Lightweight, exploratory**: each tree is one doc (~200-350 lines), produced by reading the literature and applying the catalog-as-tree pattern from the README. Per-family, math-researcher's literature work pairs naturally with scout's structural-surveying.

---

## Why now

Naturalist's holonomic essays surfaced the pattern (`~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` — names are parameter assignments on a graph; the graph is the actual catalog). The means + sketches pilots proved the pattern works for two structurally-different family topologies (means = overlapping kernels; sketches = disjoint kernels). The pattern doc + README updates capture both topologies.

Continuing the pattern across more families:
- Builds the catalog topology that the recipe-tier implementation will sit inside
- Surfaces gaps (parameter combinations the literature hasn't named but tambear should expose)
- Acts as design substrate for Sweep 35+ implementation (each tree is the "what should this family compose to?" answer before the code lands)

---

## Required pre-flight reading (in order)

1. **`R:\winrapids\docs\architecture\recipe-trees\README.md`** — the pattern. Six steps to draft a tree. Both topologies (overlapping + disjoint) named with their structural conditions.

2. **`R:\winrapids\docs\architecture\recipe-trees\means.md`** — the overlapping-kernels pilot. Five kernels with shared literature-named leaves. The mean family resolved.

3. **`R:\winrapids\docs\architecture\recipe-trees\sketches.md`** — the disjoint-kernels pilot. Three kernels with no shared leaves. Sub-agent surprise: the topology distinction emerged from the second pilot.

4. **Naturalist's garden essays** — the source pattern (`2026-05-08-the-name-is-a-parameter.md`).

5. **`R:\winrapids\docs\architecture\holonomic-architecture.md`** — for placement in the holonomic taxonomy (kernel states are content-addressed at recipe tier; the trees themselves are catalog-tier reference docs).

---

## Candidate families (no specific order — pick what calls)

### Distances (`recipe-trees/distances.md`)

- **Literature**: Lp norms (Euclidean, Manhattan, Chebyshev, Minkowski-p), Mahalanobis, cosine, Hamming, Jaccard, Wasserstein/EMD, MMD, divergences (KL, JS, Bregman), edit/string distances, graph distances, ...
- **Suspected topology**: mixed. Probably 4-5 kernels: parametric-Lp (Minkowski(p) covers many), inner-product-based (cosine + dot family), set-based (Jaccard, Sørensen-Dice), probability-distribution-based (KL, Wasserstein), edit-based. Some overlap (Euclidean = L2 = inner-product), some disjoint (string-edit ≠ Lp).
- **Cross-tree connection**: Distances connect to means (Fréchet mean = arg min of squared distance) and to clustering recipes.

### Correlations (`recipe-trees/correlations.md`)

- **Literature**: Pearson, Spearman, Kendall (τ-a, τ-b, τ-c), point-biserial, phi, polychoric, tetrachoric, distance correlation, MIC, Hoeffding's D, Schweizer-Wolff, Blomqvist β, copula-based, partial, semi-partial, canonical (CCA).
- **Suspected topology**: 3-4 kernels along axes (rank-vs-raw, scale-invariance, tail-sensitivity). Pearson + Spearman likely share a "moment-based correlation" kernel parameterized by rank-transform. Kendall is its own kernel (concordance-based). Distance correlation + MIC live in a different mathematical family entirely.
- **Cross-tree connection**: Correlations consume means (`pearson_r = covariance / (sx * sy)`); ranks live in their own family.

### Kernels (`recipe-trees/kernels.md`)

- **Literature**: Gaussian, Epanechnikov, tricube, biweight, cosine, Silverman, triangular; positive-definite kernels for SVM/GP (RBF, Matérn, polynomial, sigmoid, Bessel-J); wavelet kernels.
- **Suspected topology**: likely 2 disjoint kernel-of-kernels: (a) smoothing kernels (KDE family) parameterized by (shape, bandwidth, support); (b) positive-definite kernels (Mercer kernels) parameterized by (form, bandwidth, hyperparameters).
- **Cross-tree connection**: KDE kernels show up in `kernel_smoothed_mean` (already in `means.md`); RBF appears in regression/SVM recipes; tail-estimator families use kernel-weighted variants.

### Tail estimators (`recipe-trees/tail-estimators.md`)

- **Literature**: Hill, Pickands, MEF/MEFE, Generalized Pareto Distribution MLE, Method of Probability Weighted Moments (PWM), de Haan & Resnick, weighted Hill (Aban-Meerschaert).
- **Suspected topology**: probably one kernel (tail-quantile estimation from order statistics) parameterized by (threshold-choice, weighting-fn, asymptotic-form).
- **Cross-tree connection**: All tail estimators consume order statistics; some compose with sketches (DDSketch already gives you the relative-error quantiles); ties to extreme-value-theory recipes.

### Dispersions (`recipe-trees/dispersions.md`)

- **Literature**: variance/std, MAD, IQR, range, quartile-deviation, robust-MAD (Rousseeuw), Sn/Qn (Croux-Rousseeuw), Gini coefficient, entropy as dispersion.
- **Suspected topology**: 2-3 kernels (moment-based, rank-based, distribution-based).

### Information-theoretic (`recipe-trees/information.md`)

- **Literature**: every entropy variant (Shannon, Rényi, Tsallis, differential, conditional, joint, cross, relative), every divergence (KL, JS, Bregman, f-divergences, Wasserstein, MMD, Hellinger, TV, energy), mutual information variants (NMI, AMI, corrected), transfer entropy, directed information, channel capacity.
- **Suspected topology**: 2 kernels — entropy-family (single-distribution) and divergence-family (distribution-pair). Mutual information is a parameter assignment on divergence (KL between joint and product-of-marginals).

---

## What "done" looks like per tree

Each tree should follow the means.md / sketches.md template:

1. **Status header** — drafter, anchor on naturalist's pattern, pre-flight reading
2. **TL;DR table** of kernels — name, parameter axes, named-leaves it covers, what it's disjoint from
3. **Per-kernel sections** — formula, parameter axes, literature-named leaves table, gaps (literature hasn't named but reachable), accumulate+gather decomposition, sharing opportunities via TamSession
4. **Cross-kernel structural map** — visual representation of overlap vs disjoint
5. **Open questions for math-researcher walk-through** — at least 4-6 concrete questions
6. **Implementation roadmap** — not a commitment, just an ordering hint for when the family is implemented
7. **Threads downstream** — which other trees connect to this one

---

## Suggested team composition

This is exploratory work. Three options:

**Option 1 — Sub-agent per tree** (parallelizable). Each tree is well-bounded (~30 named literature entries × 4-5 kernels). Spawn a `general-purpose` sub-agent per tree with the means.md / sketches.md / README as templates. Main-thread reviews + commits.

**Option 2 — Scout + math-researcher pair** (if JBD team spawned for Sweep 35). Scout does the structural survey; math-researcher verifies the literature anchoring. Naturalist optionally reads the result for cross-tree patterns.

**Option 3 — Solo + sub-agents**. Same as Option 1 but no team spawn; just main-thread + sub-agents.

**Recommended**: Option 1 or Option 3. Trees are well-bounded; sub-agents do them efficiently; no need for full JBD coordination overhead.

---

## Tasks queued

Each tree is one task. Ordering doesn't matter much — pick what calls. Trees can land independently.

1. Distances tree
2. Correlations tree
3. Kernels tree
4. Tail-estimators tree
5. Dispersions tree
6. Information-theoretic tree
7. (Future) Topology trees: clustering, dimensionality-reduction, time-series-changepoint, etc.

After 2-3 more trees land, evaluate: is the catalog topology stable enough to drive the recipe-tier implementation roadmap? If yes, the catalog meta-pass is effectively complete and we shift to implementation.

---

## Risks / open questions

- **The disjoint-kernels topology** (from sketches) may be more common than the overlapping-kernels topology (from means). The README names both, but if disjoint dominates, the implementation strategy is different (no kernel-sharing to optimize; instead, per-kernel implementations + cross-kernel sharing via TamSession at the *result* level).
- **Cross-tree gaps**: some recipes will naturally live at the intersection of two trees (Fréchet mean uses distances; pearson_r consumes means; tail estimators consume sketches). The trees can name these cross-references but the canonical recipe location is single — pick one tree per recipe and link from the other.
- **Implementation-vs-design tension**: the trees are design substrate, but they're also forward-looking. If the implementation surfaces a different topology than the tree predicted, the tree gets updated. The trees aren't ratified DECs.

---

## Substrate trail

- `R:\winrapids\docs\architecture\recipe-trees\README.md` — pattern
- `R:\winrapids\docs\architecture\recipe-trees\means.md` — overlapping pilot
- `R:\winrapids\docs\architecture\recipe-trees\sketches.md` — disjoint pilot
- `~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` — naturalist's pattern source
- `R:\winrapids\docs\architecture\holonomic-architecture.md` — placement in tier taxonomy
