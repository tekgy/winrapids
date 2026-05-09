# Recipe tree — sketches (streaming quantile estimators)

**Status**: Second pilot of the catalog-as-tree pattern (`recipe-trees/README.md`).

**Drafted**: 2026-05-08 by sub-agent under team-lead delegation. Awaiting math-researcher walk-through for kernel taxonomy ratification (especially: is KLL's compactor genuinely a separate kernel, or a `CompressionPolicy` variant?); pathmaker walk-through for accumulate+gather decomposition validation.

**Anchor**: Naturalist's `~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` lines 122–138 — the worked example that introduced this tree's headline kernel `CompressedHistogram<B, C, I>`. Tekgy + main-thread Claude conversation in `important-conversation.md` lines 1–110 supplies the rank-error / value-error split that drove the 2026-05-08 vocabulary lock from KLL to DDSketch as default.

---

## TL;DR — three kernels, ~12 literature names

The streaming-quantile family resolves to **three kernels** distinguished by what they compress:

| Kernel | Compresses | Parameter axes | Named leaves it covers | Disjoint from |
|---|---|---|---|---|
| **CompressedHistogram** `H<B, C, I, M>` | the *value axis* (into buckets) | bucket-assignment `B`, compression `C`, interpolation `I`, mergeability `M` | DDSketch, linear histogram, equal-count histogram, log histogram, HDR histogram, t-digest (degenerate B), q-digest | rank-error compactor sketches |
| **RandomizedCompactor** `K<k, R>` | the *sample axis* (via probabilistic eviction) | capacity `k`, rng strategy `R`, level scheduling | KLL, MRL (Munro-Paterson-Random), variants of "compactor" trees | bucket-based and rank-tuple sketches |
| **RankTuple** `G<ε, T>` | the *rank-error envelope* (per stored sample) | error budget `ε`, tuple-pruning `T` | Greenwald-Khanna, GK-merge variants | bucket-based and compactor sketches |

The three kernels carve up the design space along the question *"what gets discarded when the sketch can no longer hold everything?"* CompressedHistogram discards value resolution within a bucket; RandomizedCompactor discards individual samples randomly; RankTuple discards samples whose rank-bound is dominated by neighbors. The error guarantees (relative-value, rank, rank) follow from this choice, not the other way around.

**Reservoir sampling, Munro-Paterson without randomization, q-digest, count-min-style "frequency" sketches**: deliberately scoped out for now (see open question #6). They're streaming-quantile-adjacent but non-overlapping with the three kernels above.

---

## Kernel 1 — CompressedHistogram `H<B, C, I, M>`

The naturalist's headline kernel. A bucket-assignment function `B: ℝ → BucketId` plus a compression policy `C` plus an interpolation policy `I` plus a mergeability tag `M`. Every literature sketch that *compresses values into a bucket structure* (DDSketch, t-digest, every histogram variant) is a parameter assignment on this kernel.

### Parameter axes

```rust
pub struct CompressedHistogram<B, C, I, M> {
    pub bucket: B,         // BucketAssignment
    pub compress: C,       // CompressionPolicy
    pub interp: I,         // InterpolationPolicy
    pub merge: M,          // MergeMode (tag, not behavior)
}

pub enum BucketAssignment {
    Log { gamma: f64 },                    // bucket(x) = ⌈log_γ(|x|)⌉ · sign(x)
    Linear { width: f64 },                 // bucket(x) = ⌊x / w⌋
    Singleton,                             // one bucket per distinct sample
    HdrHybrid { precision_bits: u8 },      // log-of-linear (Gil Tene)
    Custom(Box<dyn Fn(f64) -> i64>),       // any monotone surjection ℝ → ℤ
}

pub enum CompressionPolicy {
    None,                                  // unbounded — keep every bucket
    KBound { k: usize, eviction: Eviction },     // cap at k buckets, merge neighbors when full
    EqualCount { k: usize },               // merge until each bucket holds ~n/k samples
    DeltaScale { delta: f64 },             // t-digest's δ — bucket capacity scales by rank
}

pub enum Eviction {
    MergeSmallestNeighbor,
    MergeLowestCount,
    DropTail,                              // for skewed-data heuristics
}

pub enum InterpolationPolicy {
    Midpoint,                              // return bucket midpoint for any q
    Linear,                                // linear interp by within-bucket count
    Exact,                                 // require Singleton bucket; return exact stored value
}

pub enum MergeMode {
    BitExact,                              // associative, permutation-invariant
    RoundingEquivalent,                    // associative up to floating-point rounding
    OrderDependent,                        // merge result depends on merge order (KLL-class)
}
```

### Literature-named leaves

| Name | bucket | compress | interp | merge |
|---|---|---|---|---|
| `ddsketch_quantile` | Log{γ} | None | Midpoint | BitExact |
| `ddsketch_collapsing(k)` | Log{γ} | KBound{k, MergeSmallestNeighbor} | Midpoint | RoundingEquivalent |
| `linear_histogram_quantile` | Linear{w} | None | Linear | BitExact |
| `equal_count_histogram` | Linear{w} (init) | EqualCount{k} | Linear | OrderDependent |
| `log_histogram_quantile` | Log{γ} | None | Linear | BitExact |
| `hdr_histogram_quantile` | HdrHybrid{p} | None | Midpoint | BitExact |
| `tdigest_quantile` | Singleton (init) | DeltaScale{δ} | Linear | RoundingEquivalent |
| `q_digest_quantile` | Custom (binary tree) | KBound{k, MergeLowestCount} | Midpoint | RoundingEquivalent |

### Gaps the literature has not named (anti-YAGNI candidates)

These parameter combinations are reachable but unnamed:
- `bit_exact_collapsing_log_histogram` — Log{γ} + KBound{k, MergeSmallestNeighbor} + a deterministic eviction rule that preserves bit-exact merge (DDSketch's collapsing variant claims this; whether literature actually awards bit-exact is contested)
- `equal_count_log_histogram` — Log{γ} + EqualCount{k} (combines DDSketch's tail behavior with t-digest's centroid density)
- `tdigest_with_log_init` — Log{γ} initial seed + DeltaScale{δ} compression (recovers a wider relative-error band than vanilla t-digest)
- `singleton_no_compress` — exact quantile via sorted dictionary (degenerate; useful as a reference oracle for ε=0)
- `hdr_with_delta_scale` — HdrHybrid + DeltaScale (rare-but-defensible for latency distributions with heavy-tailed jitter)

Per anti-YAGNI: these are reachable through the kernel without per-name implementation. Recipe wrappers materialize when literature names them; the unnamed combinations remain reachable via direct kernel calls.

### Accumulate + gather decomposition

For non-compressing variants (`compress: None`):
```
CompressedHistogram<B, None, I, _>(stream):
  let bucket_counts = accumulate(
    stream,
    grouping: GroupBy(B(x)),               // bucket assignment IS the grouping
    expr: 1,
    op: Add,
  )
  let cdf = accumulate(
    bucket_counts.iter_sorted(),
    grouping: Prefix,
    expr: bucket_counts(i),
    op: Add,
  )
  return |q| interpolate(cdf, q · n, I)
```

For KBound compression, the accumulate carries a heap-side-state for "smallest neighbor pair" — Kingdom A in steady state, Kingdom B at compaction events. The kernel honestly declares the boundary; TAM schedules compaction.

For DeltaScale (t-digest), the accumulate uses `grouping: GroupBy(B(x))` with `B = nearest_centroid` — but the centroid set itself evolves with the stream, making `B` data-dependent. This is **Kingdom C** (iterative centroid placement); the centroid update is a fixed-point iteration per insert.

### Sharing opportunities via TamSession

- **Bucket-counts intermediate** is shareable across all consumers asking for *any* quantile of the same `(stream, B, C)` triple. `price_percentiles` (9 quantiles) and `cvar_with` (1 quantile) on the same return series share one histogram build. Tag: `(stream_fingerprint, bucket_fn_id, compress_policy_id)`.
- **CDF prefix-sum** is shareable across multi-quantile queries on the same histogram. Tag: `(histogram_id, ordering)`.
- **Interpolation policy is NOT a sharing dimension** — it's applied at query time over a shared CDF. Two consumers with different `I` reuse the same CDF and apply their own interpolation.

The sharing contract enforces compatibility per Tambear Contract item 3: a downstream consumer asking for "a histogram of these returns at γ=1.005" must NOT reuse a cached histogram built at γ=1.01 even though the *shape* matches. The `bucket_fn_id` in the tag includes γ.

---

## Kernel 2 — RandomizedCompactor `K<k, R>`

The KLL family. An array of "compactors" (sample buffers), each at a different level. When a compactor fills, it sorts, randomly evicts every other element, and promotes the survivors to the next level. The sketch is a tree of compactors with capacity decaying geometrically by level.

This is **structurally distinct from CompressedHistogram**: there are no buckets. The compressed object is a multi-level *sample* (a randomly-thinned version of the original stream), not a multi-bucket histogram of it.

### Parameter axes

```rust
pub struct RandomizedCompactor<R> {
    pub k: usize,                      // capacity per compactor
    pub rng: R,                        // RngStrategy
    pub level_schedule: LevelSchedule, // how compactor sizes decay across levels
    pub sort_at: SortPoint,            // when each compactor sorts internally
}

pub enum RngStrategy {
    DeterministicSeeded { seed: u64 },     // reproducible
    PerStreamRandom,                       // re-randomized per stream
    AntithesisCoupled,                     // evict-pair selected to minimize variance
}

pub enum LevelSchedule {
    GeometricHalving,                  // KLL canonical: |C_h| = k · c^h, c ∈ (0.5, 1)
    Constant,                          // Munro-Paterson: |C_h| = k for all h
    AdaptiveByLoad,                    // grow lower levels as data skews
}

pub enum SortPoint {
    OnCompaction,                      // sort just before evicting
    OnInsert,                          // maintain sorted invariant always
    Lazy,                              // sort only when queried
}
```

### Literature-named leaves

| Name | k | rng | level_schedule | sort_at |
|---|---|---|---|---|
| `kll_quantile` | k | DeterministicSeeded | GeometricHalving (c ≈ 2/3) | OnCompaction |
| `mrl_quantile` (Munro-Paterson-Random) | k | DeterministicSeeded | Constant | OnCompaction |
| `kll_antithesis` | k | AntithesisCoupled | GeometricHalving | OnCompaction |
| `compactor_lazy` | k | DeterministicSeeded | GeometricHalving | Lazy |

### Why this kernel is separate from `CompressedHistogram`

A KLL compactor doesn't produce a function `bucket: ℝ → BucketId`. It produces a *thinned multiset* — a random sample where each surviving element carries a weight `2^h` (its level). The quantile query is:
```
KLL.quantile(q):
  collect all surviving samples weighted by 2^h
  sort them
  return the value at cumulative weight q · n
```
There is no "bucket structure" to compose with `bucket: Log{γ}` or `bucket: Linear{w}`. The compactor IS the compression. Trying to express KLL as `CompressedHistogram<Singleton, Compactor, Linear, OrderDependent>` requires inventing a `Compactor` compression-policy variant that doesn't compose with the rest of `CompressionPolicy` — at which point you've smuggled a separate kernel into a `using()` slot. The honest move is two kernels.

This is the core open question for math-researcher (#1 below): *is there a unified kernel one level up that covers both?* Possibly yes — "stream → ordered multiset" is a common abstraction — but the parameter axes don't share enough vocabulary for the unification to pay rent today.

### Accumulate + gather decomposition

```
RandomizedCompactor<k, rng>(stream):
  let compactor_levels = accumulate(
    stream,
    grouping: TreeFock,                 // hierarchical compactor levels
    expr: insert_with_compaction(x, k, rng),
    op: CompactorMerge,
  )
  return |q| weighted_quantile(compactor_levels, q)
```

`TreeFock` is the Fock-boundary grouping introduced for hierarchical state — TAM handles cross-level promotion. **Kingdom A** for inserts, **Kingdom B** at compaction events (each compaction is a small sequential scan).

### Sharing opportunities via TamSession

Limited. The compactor state is permutation-*sensitive* — two consumers building "a KLL sketch on these returns" with different insertion orders produce different state. Sharing only fires when the upstream consumer guarantees the same insertion order *and* the same seeded RNG. In practice: only same-call consumers share. Cross-call sharing in the KLL kernel is rare; this is a structural reason to prefer `CompressedHistogram` when sharing across recipes is desirable.

---

## Kernel 3 — RankTuple `G<ε, T>`

The Greenwald-Khanna sketch. State is a sorted list of triples `(value, g_i, Δ_i)` where `g_i` is the rank gap to the previous tuple and `Δ_i` is the maximum rank-error any subsequent merge could introduce. Tuples are pruned when their rank-error budget is dominated by neighbors, capping memory at `O(log(εn) / ε)`.

This is **structurally distinct from both other kernels**: there are no buckets *and* no random eviction. Every retained sample is exact; what's compressed is the *envelope of rank-error tolerances* around each retained sample.

### Parameter axes

```rust
pub struct RankTuple<T> {
    pub epsilon: f64,             // rank-error budget
    pub prune: T,                 // PruneStrategy
    pub merge_mode: GkMerge,
}

pub enum PruneStrategy {
    GreenwaldKhanna,              // canonical: prune (g+Δ) ≤ 2εn
    Aggressive { factor: f64 },   // tighter prune at memory cost
    Conservative { slack: f64 },  // looser prune for adversarial streams
}

pub enum GkMerge {
    Cormode,                      // "GK-merge" — adjusts Δ on each merged tuple
    BiasedQuantile,               // Cormode-Korn — bias toward tail accuracy
}
```

### Literature-named leaves

| Name | epsilon | prune | merge_mode |
|---|---|---|---|
| `gk_quantile` | ε | GreenwaldKhanna | Cormode |
| `gk_biased_tail` | ε | GreenwaldKhanna | BiasedQuantile |
| `gk_aggressive` | ε | Aggressive{f} | Cormode |

### Why this kernel is separate

The state isn't a histogram and isn't a random sample. It's a sorted dictionary of `(value, rank-bound)` pairs. The pruning rule depends on a *running comparison of three tuples at a time* — Kingdom B intrinsically. Trying to expose this as a `CompressionPolicy` variant on `CompressedHistogram<Singleton, …>` would force the histogram's bucket-structure abstractions over a sketch that has no buckets.

### Accumulate + gather decomposition

```
RankTuple<ε>(stream):
  let tuples = accumulate(
    stream,
    grouping: SortedInsertWithPrune(ε),    // intrinsically sequential
    expr: GkTuple(x),
    op: GkInsert,
  )
  return |q| binary_search_for_rank(tuples, q · n)
```

**Kingdom B** throughout. The sequential dependency is fundamental: the prune decision for tuple `i+1` depends on tuple `i`'s post-insert state.

### Sharing opportunities via TamSession

The tuple list, once built, is **shareable across multi-quantile queries** on the same `(stream, ε, prune, merge_mode)` tuple — same as `CompressedHistogram`'s CDF. Tag: `(stream_fingerprint, gk_params_id)`. Cross-recipe sharing fires whenever two recipes ask for any GK quantile on the same stream at compatible ε.

---

## Cross-kernel structural map

```
                        streaming quantile family
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
CompressedHistogram      RandomizedCompactor          RankTuple
  H<B, C, I, M>              K<k, R>                    G<ε, T>
        |                         |                         |
   +----+----+----+              +-+--+                  +--+--+
   |    |    |    |              |    |                  |     |
 ddsketch  linear  hdr  tdigest  kll   mrl              gk   gk_biased
   |        hist   hist   |       |     |
   |          |     |     |     antithesis
   |  log_hist     equal_count
   |               hist
   |
   q_digest

                                                [unnamed gaps]
                                                bit-exact-collapsing log
                                                equal-count log
                                                tdigest with log seed
                                                hdr + delta-scale
                                                ...
```

Disjoint kernels, no overlapping leaves. This is structurally different from the means tree, where named leaves are reachable from multiple kernels — here, the kernels carve disjoint subsets because the underlying data structures don't share state shape.

---

## Open questions for math-researcher walk-through

1. **Is there a unified kernel above the three?** The three kernels all answer `quantile(q): ℝ → ℝ` from a stream, but their state shapes differ structurally (bucket-grid vs randomized-multiset vs rank-tuple-list). Is "stream-to-orderable-summary" worth elevating to a top-level kernel with the three current kernels as `using(strategy: …)` variants? My read: not yet. The parameter axes don't share enough vocabulary for the unification to pay rent. But the rhyme is real — `RankMean` from the means tree calls *any* of these and can't tell which is underneath. That client-perspective unification may itself be the structural answer.

2. **Is `tdigest_quantile` correctly classified?** I placed it under `CompressedHistogram` with `bucket: Singleton (init), compress: DeltaScale{δ}, interp: Linear, merge: RoundingEquivalent`. But t-digest's "centroids" are *adaptive* — the bucket boundaries move as the stream evolves. This is `bucket: Singleton` only at init; in steady state, the bucket-assignment is data-dependent and Kingdom C. Is it more honestly a fourth kernel (`AdaptiveCentroidHistogram`)? Or does the `compress: DeltaScale` tag carry enough information that consumers can route around the kingdom shift?

3. **Where does HDR Histogram (Gil Tene) fit?** I classified it as `bucket: HdrHybrid` — log-of-linear bucket assignment. This is structurally a Custom variant of `BucketAssignment`, but it's named in the literature, so it earns its own enum entry. Is there a more general `bucket: PiecewiseAffine` that covers HDR + future hybrids in one axis?

4. **Greenwald-Khanna vs RankTuple naming**: I named the kernel `RankTuple` to avoid the GK name, since it's the kernel-not-the-implementation. But "GK" is the canonical name in the literature and recipes already say "gk_quantile." Is there a better kernel name that doesn't bury the GK heritage? Candidates: `GkFamily`, `RankBoundedSample`, `EpsilonRankSketch`.

5. **q-digest as `CompressedHistogram` vs separate kernel**: I placed `q_digest_quantile` under `CompressedHistogram` with `bucket: Custom (binary tree)`. But q-digest's bucket-tree is structurally a binary tree of nested intervals, not a flat bucket grid. The flat grid assumption shows up in the `CompressedHistogram` accumulate decomposition (`grouping: GroupBy(B(x))` returns one bucket per `x`). For q-digest, a value lives in O(log V) nested buckets simultaneously, not one. Is this a fourth kernel, or does `CompressedHistogram` need a `bucket_arity: Flat | Tree` axis?

6. **Scope question — adjacent families deferred**: I deliberately scoped out reservoir sampling, count-min sketches, and Munro-Paterson without randomization. They're streaming-quantile-adjacent but their state shapes don't match the three kernels above. Should they live in a sibling tree (`recipe-trees/streaming-summaries.md`)? Or is "all streaming summaries" actually one family with a wider kernel taxonomy? Surface for naturalist + math-researcher.

7. **The `MergeMode` axis as derived vs free**: I made `merge: M` a parameter axis on `CompressedHistogram`, but the merge mode is *derived* from `(B, C, I)` choices in most cases — DDSketch is BitExact because Log{γ} + None + Midpoint composes that way; t-digest is RoundingEquivalent because Singleton + DeltaScale + Linear admits floating-point reordering. Is `M` an independent axis, or is it a *type-level witness* that should be checked at construction (F13 antibody) rather than chosen by the user? My intuition: type-level witness, not free parameter. But that has API implications worth ratifying.

---

## Implementation roadmap

This doc is the catalog tree. **Implementation is downstream and not blocked by this doc.** The current state in `R:\tambear\crates\tambear\src\primitives\specialist\quantile_sketch.rs` is a `SketchAlgorithm` enum with four flat variants (`Kll`, `Gk`, `Tdigest`, `DdSketch`) — no kernel decomposition yet. Three recipes (`cvar`, `hill_estimator_streaming`, `price_percentiles`) match-arm on the variants directly.

If/when the team decides to refactor:

1. **Build `CompressedHistogram<B, C, I, M>` first** — covers DDSketch (locked default) and the histogram variants. Most-used path; cleanest accumulate+gather decomposition.
2. **Migrate `DdSketch` to a recipe wrapper** — `bucket: Log{γ}, compress: None, interp: Midpoint, merge: BitExact`. ~20 lines + sibling pointers.
3. **Build `RandomizedCompactor<k, R>`** for KLL + MRL. Kingdom A/B-honest implementation; sharing is limited so the sharing infrastructure can be lighter.
4. **Build `RankTuple<ε, T>`** for GK + GK-biased. Kingdom B throughout.
5. **Migrate `cvar`, `hill_estimator_streaming`, `price_percentiles`** to dispatch on `(kernel_id, params)` rather than `SketchAlgorithm` flat variants. The three-kernel taxonomy lets these recipes accept *any* sketch with a `quantile(q)` interface uniformly.
6. **t-digest classification** is gated on math-researcher walk-through (open question #2). Either ships as a `CompressedHistogram` variant with adaptive bucket-assignment annotations or earns its own kernel `AdaptiveCentroidHistogram`.

The migration is opt-in. The current flat-enum implementation is correct; the kernel decomposition is a refactor for clarity and `using()` knob coverage, not a bug fix.

---

## What the tree teaches

Three things visible from this artifact that are NOT visible from the per-name recipe list:

1. **The three kernels carve the design space along *what gets discarded***, not along names. DDSketch and t-digest aren't different "algorithms" — they're different `(B, C, I)` assignments on one kernel. KLL and GK aren't different "variants of compactor sketches" — they're different kernels because their state shapes don't share. The kernel boundary is structural, not historical.

2. **Sharing fires at the kernel level, not the recipe level.** When `cvar_5%`, `cvar_1%`, `var_5%`, and `price_percentiles_sip` all run on the same return stream with the locked DDSketch default, they share *one* `CompressedHistogram` build via TamSession. The fan-out is automatic from the kernel decomposition; the flat-enum implementation can't do this without bespoke per-recipe caching.

3. **The unnamed-gap list is a research surface.** `bit_exact_collapsing_log_histogram`, `equal_count_log_histogram`, `tdigest_with_log_init`, `hdr_with_delta_scale` — these are structurally reachable but unnamed in the literature. Some may be uninteresting (degenerate); some may be genuinely useful and just unnamed. The tree exposes them; tambear can ship them via direct kernel calls; if any one of them turns out to be load-bearing for SIP or fintek, it earns a recipe wrapper and a name.

The naturalist's claim was *the recipe catalog becomes a map, not a list*. For sketches, the map is a flat triptych — three disjoint kernels — rather than the means tree's overlapping pentagon. Both topologies are valid tree shapes; the family decides which it is.

---

## Threads downstream of this tree

- **Means tree** (`recipe-trees/means.md`): `RankMean<Q, S>` is the means-side surface that calls into the sketches family. The means tree's `RankMean.composition with sketches` section delegates to *any* of the three kernels here. The two trees connect at this seam — when a user says `median(x).using(streaming=true)`, the means tree's `RankMean` recipe wraps a sketches-tree kernel call. Sharing flows through TamSession across the tree boundary.

- **Tail estimators tree** (`recipe-trees/tail-estimators.md`, TBD): Hill, Pickands, DEdH, GPD-fit. All build on streaming-quantile thresholds. The threshold-finding step is a sketches-tree call; the tail-fitting step lives in the tail-estimators tree. The two compose via TamSession sharing of the tail-region sample.

- **Histograms / descriptive stats tree** (TBD): the `CompressedHistogram` kernel without compression (`compress: None`) IS a histogram. The descriptive-stats tree (mode, KDE, count-based central tendency from `recipe-trees/modes.md` deferred from the means tree) shares this kernel. Sharing fires across trees: a user asking for both "the histogram of these prices" and "the 9 SIP percentiles" pays for one bucket-counts intermediate.

- **Distributed-merge tree** (TBD): the `merge: BitExact | RoundingEquivalent | OrderDependent` axis is a sketches-tree concern but propagates to every recipe that does cross-shard aggregation. SIP's Merkle anchoring, distributed signal farms, and any cross-machine reduction depend on the merge tag. A future tree may surface this axis as its own taxonomy.

These are not assignments — they're invitations. The tree pattern propagates naturally as someone touches a family.
