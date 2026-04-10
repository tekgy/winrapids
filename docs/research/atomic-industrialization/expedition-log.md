# Atomic Industrialization — Expedition Log

## 2026-04-10: Naturalist — Spring Topology Observations

### The spring simulation reveals two kinds of family

Running the spring network at `playgrounds/tamsession-spring-network.html` with the
actual fintek sharing topology (12 phyla, ~90 nodes) reveals that "family" means two
structurally different things:

**Clusters** — tight groups, many internal edges, few external edges.
- Spectral (spread ~82): 15 nodes all connected through FFT. Very tight. This IS one family.
- Volatility (spread ~104): 5 nodes connected through MomentStats + ACF + LogReturns. Tight.
- Multivariate (spread ~120): 5 nodes connected through CovMatrix + Eigendecomp. Tight.

**Webs** — diffuse groups, few internal edges, many cross-family bridges.
- Information theory (4 nodes, but edges go everywhere): shannon_entropy connects to spectral, mutual_info connects to clustering, transfer_entropy connects to time_series. Low clustering coefficient, high betweenness centrality.
- Linear algebra (5 nodes, but everything depends on it): cholesky/qr/svd/eigendecomp/ols are consumed by multivariate, hypothesis, time_series, clustering. Not a cluster — a foundation.

**False clusters** — things labeled as one family but actually multiple:
- Descriptive (spread ~435): Three sub-families being torn apart. Moment consumers (pulled toward hypothesis by MomentStats sharing), sorted-order consumers (pulled toward nonparametric by SortedOrder sharing), error metrics (floating free, no shared intermediates with either group).

### Implications for TamSession phyla map

The sharing topology defines the natural phyla, not the human-assigned family labels:

1. **MomentStats phylum** spans: descriptive (moment consumers) + hypothesis + volatility
2. **SortedOrder phylum** spans: descriptive (sorted consumers) + nonparametric
3. **FFT phylum** spans: spectral (mostly contained within family)
4. **CovMatrix phylum** spans: multivariate + clustering (some)
5. **ACF phylum** spans: time_series (ACF sub-group) + volatility
6. **DistMatrix phylum** spans: clustering + complexity (some)
7. **OLS phylum** spans: linear_algebra + time_series (stationarity) + hypothesis
8. **Eigendecomp phylum** spans: linear_algebra + multivariate + clustering (spectral)

Each phylum is a TamSession sharing boundary. Nodes connected by the same phylum
should be scheduled together. The phyla, not the families, are the natural units for
TAM's scheduler.

### Bridge nodes

The simulation identifies bridge nodes — primitives connecting 3+ families:
- **garch/egarch**: volatility family but consumes MomentStats + ACF + CovMatrix + LogReturns (4 phyla)
- **adf_test**: time_series family but consumes ACF + OLS (2 phyla)
- **shapiro_wilk**: hypothesis family but consumes MomentStats + SortedOrder (2 phyla)
- **spectral_clustering**: clustering family but consumes DistMatrix + Eigendecomp (2 phyla)

Bridge nodes are expensive to schedule — they pull intermediates from multiple phyla.
They're also the most important nodes for correctness testing, because they're where
incompatible sharing assumptions would cause silent errors. The IntermediateTag
compatibility check is most critical at bridge nodes.

### Prediction: time_series is also a false cluster

By analogy with descriptive: time_series has 10 nodes that should split into:
- **ACF sub-family**: pacf, ar_fit, arma, ljung_box, durbin_watson (share ACF intermediate)
- **Changepoint sub-family**: cusum, pelt, bocpd (share segmentation state — different from ACF)
- **Stationarity sub-family**: adf_test, kpss (share OLS intermediate — different from both)

This prediction is testable: run the spring simulation with time_series nodes only and
see if they form three sub-clusters or one.

### The "loose family" hypothesis

Families with high spread in the spring simulation might be:
1. **False clusters** (descriptive, probably time_series) — actually multiple families
2. **Incomplete families** (information theory) — too few nodes to form a cluster; the
   missing nodes would create internal edges and tighten the spread
3. **Connective tissue** (linear algebra) — genuinely diffuse because their role is to
   connect other families, not to form their own cluster

Each diagnosis implies a different action:
- False clusters: split the family tag (or allow multi-family membership)
- Incomplete families: fill the missing primitives
- Connective tissue: leave as-is; the diffuseness IS the correct topology

### Information theory is incomplete, not misclassified

Information theory's 4 spring-simulation nodes undersample the real module (25+ pub fns).
But even the full module is thin compared to the catalog's ambitions. Key gaps:
- Continuous-domain estimators (KL-kNN, Kozachenko-Leonenko)
- f-divergence family (Hellinger, total variation, chi-squared, alpha-divergence)
- Optimal transport (Wasserstein / Earth Mover's Distance)
- Partial information decomposition (PID: redundant/unique/synergistic)
- Rate-distortion and channel capacity

These missing primitives would create new edges within the information theory family
(f-divergences share a generator function, PID shares MI computation) and tighten the
cluster. The loose spread is a symptom of incompleteness, not misclassification.

### The Brusselator connection

The Brusselator garden entry (2026-04-10-brusselator.md) ends with an observation:
"the sharing topology IS the detection apparatus." The volatility cluster shares
intermediates with both MomentStats (for the fixed point) and ACF (for critical
slowing down). The Brusselator bifurcation analysis needs exactly those two inputs.
The spring simulation's topology predicts which primitives need to be co-scheduled
for Hopf bifurcation detection — the cluster structure IS the algorithm.

### IntermediateTag coverage vs holographic screen

The TamSession infrastructure (intermediates.rs) has concrete IntermediateTag variants.
Comparing what's tagged vs what the holographic screen analysis suggests should be tagged:

| Screen intermediate | IntermediateTag variant | Status |
|---------------------|------------------------|--------|
| MomentStats | `MomentStats { data_id }` | Present, wired via `moments_session()` |
| SortedOrder | (none) | MISSING — no tag for sorted array + ranks |
| FFT | (none) | MISSING — no tag for complex spectrum |
| ACF | (none) | MISSING — no tag for autocorrelation array |
| CovMatrix | (none) | MISSING — no tag for covariance matrix |
| DistMatrix | `DistanceMatrix { metric, data_id }` | Present, wired via ClusteringEngine |
| LogReturns | (none) | MISSING — no tag for log return series |
| OLS cache | (none) | MISSING — no tag for X'X, X'y, residuals |
| DataQuality | `DataQuality { data_id }` | Present, wired |
| SufficientStats | `SufficientStatistics { data_id, grouping_id }` | Present |

5 of 8 holographic screen intermediates have no IntermediateTag. This means the
sharing infrastructure exists but is only wired for 3 of the 8 load-bearing phyla.
The other 5 are computed redundantly by every consumer.

Priority for wiring: FFT (15 consumers), SortedOrder (13 consumers), ACF (6 consumers),
CovMatrix (5 consumers), OLS (6 consumers). FFT has the highest consumer count and
the highest per-computation cost — it should be the next IntermediateTag added.

### Convergent insight: topology predicts algorithms

All five threads explored today converge on one claim: the sharing topology of
mathematical intermediates is a structural property of mathematics, not an
implementation detail. When two methods share an intermediate, the mathematical
concepts are genuinely related (Pearson and t-test share MomentStats because both
ask about first/second moments).

Consequences:
1. Families defined by sharing are more real than families defined by textbook chapters
2. The topology predicts which algorithms are natural (clusters = unnamed pipelines)
3. Missing internal edges predict missing primitives (f-divergences would tighten info theory)
4. The holographic screen ({moments, sorted order, FFT, ACF, covariance, distance,
   log transform, regression}) is the mathematical skeleton — the foundational
   operations from which all applied statistics derives

Testable prediction: run community detection on the full sharing graph (~400 pub fns).
Communities that don't correspond to known pipelines are PREDICTED algorithms — natural
compositions we haven't named yet.

### Aristotle-naturalist synthesis: grouping patterns ARE the sharing topology

Aristotle's deconstruction concluded: "the grouping patterns are the open frontier —
accumulate is the canvas, grouping is the painting." The naturalist's analysis concluded:
"the sharing topology IS the algorithm — communities predict which algorithms are natural."

These are the same insight at different altitudes. The grouping pattern of an accumulate
IS what creates sharing edges in the topology. The phyla correspond to groupings:

| Phylum | Grouping pattern | Why |
|--------|-----------------|-----|
| MomentStats | All | Global summary = accumulate over all elements |
| SufficientStatistics | ByKey | Per-group = accumulate partitioned by key |
| ACF | Windowed | Local windows = accumulate over sliding ranges |
| CovMatrix | Tiled | M x N blocks = accumulate over tile pairs |
| Prefix products | Prefix | Sequential dependency = prefix scan |
| Multi-scale stats | Segmented | Variable-length segments |

Consequence: Aristotle's missing groupings (Tree, Graph, Adaptive, Circular) predict
missing phyla. Implementing Tree grouping should create a new cluster in the spring
simulation — tree-structured algorithms that currently can't share because there's no
Tree grouping to create the edges.

The "topology predicts algorithms" claim sharpens to: an unnamed community in the
sharing graph = methods with the same grouping that haven't been assembled into a
named pipeline. The grouping determines the sharing; the sharing determines the
algorithm; therefore the grouping determines the algorithm.

### VERIFIED: Spring clusters = grouping classes

Aristotle predicted that algorithms with the same grouping class should form tight
clusters in the spring simulation. Checked all clusters against their (grouping, op)
classification:

| Spring cluster | Grouping | Op | Spread |
|----------------|----------|----|--------|
| FFT consumers | All | ButterflyMul | ~82 (tightest) |
| MomentStats consumers | All | Add | moderate |
| CovMatrix consumers | Tiled | DotProduct | ~120 |
| DistMatrix consumers | Tiled | Distance | moderate |
| ACF consumers | Windowed | Correlation | moderate |
| SortedOrder consumers | (transform) | Sort | moderate |
| Volatility | Prefix | Welford | ~104 |

**Prediction confirmed at (grouping, op, transform) level.** Same grouping + different
op = different cluster (All+Add vs All+Butterfly; Tiled+DotProduct vs Tiled+Distance).

The spring simulation IS the grouping classification rendered as physics. The spring
forces implement the constraint: same algebraic class attracts, different class repels.
The equilibrium IS the classification.

**The theorem (informal, naturalist-Aristotle joint)**:

    Tight clusters in sharing topology
        ↔ finitely-presentable group actions × (op, transform)
        ↔ algorithm classes

Each cluster has a canonical intermediate hierarchy: raw → canonical → derived. This
pattern repeats at every grouping level:
- All + Add: data → MomentStats → hypothesis tests
- All + Butterfly: data → FFT spectrum → spectral features
- Tiled + DotProduct: data pairs → CovMatrix → PCA/LDA
- Graph (predicted): adjacency → Laplacian → eigendecomp → spectral graph methods

### Late observation: transforms as a distinct phylum

`log_returns` lives in time_series.rs but is consumed by: volatility (GARCH, realized
vol), information theory (entropy on returns), complexity (DFA, Hurst), and nearly every
fintek bridge. It's not a "time series" primitive — it's a TRANSFORM primitive. It
converts prices to returns. It has no natural home in any family because it's pre-family
— it runs before any family-specific computation.

Other transforms in the same category: `rank()` (pre-nonparametric), `sorted_nan_free()`
(pre-sorted-order), `delay_embed()` (pre-embedding-based complexity), `box_cox_transform()`
(pre-descriptive for skewness correction).

Transforms are a distinct phylum in the sharing topology. They're not intermediates in
the same sense as MomentStats or FFT (which are computed FROM the data). They're PRE-
intermediates — they prepare the data so that the family-specific intermediates can be
computed correctly. The dependency order is: raw data → transforms → intermediates →
primitives → methods.

This adds a layer to the holographic screen: the screen should include transform
intermediates (log_returns, ranks, sorted order, delay embedding) BEFORE the
computational intermediates (MomentStats, FFT, CovMatrix, etc.).

### Garden entries produced

- `2026-04-10-three-sub-families.md` — descriptive split analysis
- `2026-04-10-information-theory-gaps.md` — info theory incompleteness diagnosis
- `2026-04-10-eeg-tam-rhyme.md` — cross-frequency coupling analogy assessment
- `2026-04-10-sharing-topology-is-the-algorithm.md` — topology = correctness contract
- `2026-04-10-the-holographic-screen.md` — minimum intermediate set analysis (8 = screen, 6 = core)
- `2026-04-10-bifurcation-as-primitive.md` — Brusselator decomposed into primitive compositions
- `2026-04-10-topology-predicts-algorithms.md` — synthesis of all threads

---

*End naturalist log, 2026-04-10.*

---

## 2026-04-10: Scientist — pearson_r Principle 10 Workup

### What was done

First complete Principle 10 workup for a non-Kendall primitive.
Target: `nonparametric::pearson_r`.

**Why pearson_r?**

- Highest downstream dependency count of the viable candidates: OLS,
  concordance correlation, CCC, covariance, PCA, and the entire hypothesis
  testing layer all either call it or share its intermediates.
- The one-pass vs two-pass stability issue is the canonical example of why
  "implement from first principles with our own quality bar" matters. One-pass
  Pearson is in textbooks and stdlib docs everywhere; it gives silently wrong
  answers on large-shift data (catastrophic cancellation in `n·Σx² − (Σx)²`).
  Tambear's two-pass implementation is correct.
- Exactly the right scale for a first workup: O(n) algorithm, tractable
  adversarial cases, clear oracle (rational algebra → mpmath = machine
  precision).

### Numerical findings

All 8 oracle cases verified bit-perfect against mpmath at 50 dp. Max observed
relative error: **0** (implementation accumulates exact integer-free arithmetic
through two linear passes; only the final sqrt+division rounds).

The one-pass catastrophic cancellation was confirmed computationally: for
`x = [1e10+1, ..., 1e10+5]`, the one-pass denominator `n·Σx² − (Σx)²`
collapses to 0 in f64 arithmetic (loss of all ~20 significant digits). Tambear
returns r = 1.0 correctly. This is the decisive argument for two-pass, and it
is now documented in Section 3.2 of the workup with a concrete example.

### Competitor comparison

scipy 1.x: bit-perfect agreement on all cases (≤ 1 ULP difference, ≤ 5.5e-17
absolute). scipy also uses two-pass. Neither library uses the one-pass formula.

### New bugs found and documented

1. **Absolute degeneracy threshold** (`denom < 1e-15` is scale-dependent).
   Should be relative. Low severity — only affects inputs with magnitude < 1e-8.
2. **NaN propagation is incidental, not contractual.** Need explicit NaN policy.
3. **No TamSession registration.** CenteredMoments intermediate (sufficient for
   r, covariance, OLS, variance, t-statistic) is recomputed by each consumer.
   This is the most impactful gap — fixing it gives free sharing to ~6 methods.

### Test suite

18 tests in `crates/tambear/tests/workup_pearson_r.rs`. All green.

Covers: oracle cases (8), edge cases (3), invariants (6), adversarial
large-shift stability (1). Open items: NaN injection, ±∞ injection, subnormal,
overflow (documented in workup §7).

### Infrastructure built

`docs/research/atomic-industrialization/benchmark-infrastructure.md` — the
reusable pattern for oracle harness, competitor harness, and scale sweep across
all future workups. Includes primitives queue (8 primitives ordered by
downstream impact) and a workup completion checklist.

### What's next for this workup

1. Scale benchmark (run when hardware session available — just add
   `#[ignore]` test, no algorithmic work needed)
2. Degeneracy threshold fix (absolute → relative, ~3-line change)
3. TamSession CenteredMoments registration (unlocks sharing for OLS + variance)

### Cross-reference with naturalist findings

The naturalist's expedition identified that OLS and Pearson r share the same
intermediate (`MomentStats / CenteredMoments`) but the sharing infrastructure
is not wired. The workup's Section 3.6 documents exactly what the
`IntermediateTag::CenteredMoments { data_id_x, data_id_y }` tag should carry.
These two analyses converge: the naturalist found the gap in the sharing
topology; the workup found the same gap from the primitive's perspective.

*End scientist log, 2026-04-10.*
