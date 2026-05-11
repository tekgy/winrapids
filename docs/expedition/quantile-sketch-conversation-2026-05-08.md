(KLL/GK/t-digest/DDSketch), 4 Observations structs, vocab lock — 2805 tests green. Next: pick a thread
  (tambear-sip standalone binary, parallel CPU surface, or something else). (disable recaps in /config)

❯ based on how you described ddsketch, i'm not clear on the pros and cons compared to KLL as our default

● Fair question — and the honest answer is the choice is more interesting than I made it sound when I locked KLL
  as default. Let me lay it out properly.

  The fundamental difference: what kind of error you bound

  This is the crux:

  - KLL bounds RANK error. ε=0.01 means "the returned value is at a true rank within ±1% of the requested
  quantile." So quantile(0.5) returns the value sitting somewhere between rank 0.49 and 0.51. The value itself
  could be far from the true median if the distribution is steep there.
  - DDSketch bounds RELATIVE-VALUE error. ε=0.01 means "the returned value is within ±1% of the true value at that
   quantile." So quantile(0.5) returns a value that's within 1% of the actual median. The rank could be off by
  more than 1%, but the number you get is close to right.

  These optimize for different questions, and for most real consumer use cases, relative-value error is what
  people actually want. "What's the 99th percentile latency?" — you want milliseconds within 1%, not rank within
  1%. Same for "what's the 99th percentile price?" — you want dollars within 1%, not rank position within 1%.

  Comparison table

  ┌─────────────────────────────┬─────────────────────────────────┬───────────────────────────────────────────┐
  │          Property           │               KLL               │                 DDSketch                  │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ Error type                  │ Rank-based                      │ Value-relative                            │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ ε=0.01 means...             │ Returned value's true rank in   │ Returned value within ±1% of true value   │
  │                             │ [0.49, 0.51]                    │                                           │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ Permutation invariance      │ No (compaction order matters)   │ Yes (state is a function of multiset      │
  │                             │                                 │ only)                                     │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ Cross-shard merge           │ "Close enough" — diverges with  │ Bit-exact associative — identical         │
  │                             │ merge order                     │ regardless of merge order                 │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ Sort needed                 │ Yes (per-compactor at           │ Never (BTreeMap sorted by construction)   │
  │                             │ compaction time)                │                                           │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │                             │ O(k · log(n/k)) —               │ O(log_γ(value_range)) — depends on data   │
  │ Memory per ε                │ minimum-memory at given         │ range                                     │
  │                             │ rank-accuracy                   │                                           │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ Wide value ranges           │ Memory unaffected by range      │ Memory grows with log(range) — ~3000      │
  │                             │                                 │ buckets for ±10^17 at ε=0.01              │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ Tight value ranges          │ Same memory as wide             │ Less memory — fewer buckets needed        │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ Multi-orders-of-magnitude   │ Treats them all the same        │ Naturally adapts — log bucketing gives    │
  │ data                        │                                 │ constant relative resolution              │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ Mixed sign + zero           │ Treats as continuous            │ Native handling via signed bucket index + │
  │                             │                                 │  zero bucket                              │
  ├─────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────┤
  │ Determinism strength        │ Run-to-run on same input        │ All of: run-to-run,                       │
  │                             │                                 │ permutation-invariant, bit-exact-merge    │
  └─────────────────────────────┴─────────────────────────────────┴───────────────────────────────────────────┘

  The benchmark output told me something I should re-evaluate

  The merge_fidelity_comparison test output I showed you earlier — KLL diverged ~5% on lognormal q=0.5/0.9 between
   full and merged paths. For SIP's Merkle chain, that's a non-starter at scale. A SIP file's hash includes its
  quantile-derived header fields. If two valid execution orderings produce different hash, the chain has multiple
  valid roots — that's a fork.

  DDSketch is bit-exact across any merge order. KLL is "close enough" but not bit-exact. For SIP specifically,
  that difference is load-bearing.

  Honest re-evaluation: should DDSketch be the default?

  For tambear's use cases, I think DDSketch is probably the better locked default, not KLL. The reasoning:

  1. Most consumers want relative-value error, not rank error. Price percentiles, latency percentiles, return
  percentiles — all are read as values, not ranks.
  2. SIP's Merkle anchoring needs bit-exact distributed merge. Only DDSketch provides this.
  3. "Kill the sort" was a foundational principle. KLL has internal sorts at compaction time; DDSketch has none
  anywhere.
  4. Permutation invariance means the sketch state itself is verifiable independently — same multiset always
  produces the same buckets. This is auditable in a way the others aren't.

  Where KLL would still win:
  - Recipes that genuinely care about rank accuracy (e.g., "the median of this discrete-valued distribution")
  - Tightly-bounded data where DDSketch's bucket count is wasteful relative to KLL's compactor capacity
  - Workloads where memory is genuinely the bottleneck

  But those are minority cases for what tambear is for.

  What I'd recommend

  Switch the locked default to DDSketch, leave KLL/GK/t-digest as using(sketch: ...) overrides for cases where
  their specific properties matter. The vocabulary doc would need a one-line update; the four sketch-dependent
  recipes already accept all four algorithms via SketchAlgorithm.

  Or keep KLL as default and add a clear note in vocabulary.md that DDSketch is the right choice when bit-exact
  distributed merge matters (e.g., for SIP's Merkle chain) — making the choice explicit per use case rather than
  "default fits all."

  Your call. The infrastructure is symmetric — switching the default is a one-line change in
  SketchAlgorithm::DEFAULT and a corresponding update in vocabulary.md. Or we can keep the locked default and
  document the per-use-case override more prominently.

  Honestly tracking this — when I locked KLL as default in vocabulary.md earlier today, I was making a
  conservative engineering choice based on KLL's theoretical optimality. Now that I've built DDSketch and seen its
   actual properties, I think I picked wrong for tambear's specific use cases. Worth correcting.

✻ Crunched for 1m 2s

❯ what about other ways to arrive at median that are NOT naive sort based? quickselect or median of medians
  (which we shoudl probably also have a deterministic linear time option) or something? somethign that we
  discover WORKS but only because we're accumulate+gather? we have found that in our work in this project before
  that 2 seemingly unrelated algorithms are in fact 2 ways to get at the same exact thing, keepin a running
  sample or reservoir, remedian stat, what would the error be on eta learning rate param and signum function, or
  etc. -- somethign like this idea but tuned for our accumulators ```Here is a crazy approach that you might try.
   This is a classical problem in streaming algorithms. The rules are

  You have limited memory, say O(log n) where n is the number of items you want
  You can look at each item once and make a decision then and there what to do with it, if you store it, it costs
   memory, if you throw it away it is gone forever.
  The idea for the finding a median is simple. Sample O(1 / a^2 * log(1 / p)) * log(n) elements from the list at
  random, you can do this via reservoir sampling (see a previous question). Now simply return the median from
  your sampled elements, using a classical method.

  The guarantee is that the index of the item returned will be (1 +/- a) / 2 with probability at least 1-p. So
  there is a probability p of failing, you can choose it by sampling more elements. And it wont return the median
   or guarantee that the value of the item returned is anywhere close to the median, just that when you sort the
  list the item returned will be close to the half of the list.

  This algorithm uses O(log n) additional space and runs in Linear time.``` accumualte with a histogram or
  something, idk. or this idea ```2

  Find Min and Max of the list containing N items through linear search and name them as HighValue and LowValue
  Let MedianIndex = (N+1)/2

  1st Order Binary Search:

  Repeat the following 4 steps until LowValue < HighValue.

  Get MedianValue approximately = ( HighValue + LowValue ) / 2

  Get NumberOfItemsWhichAreLessThanorEqualToMedianValue = K

  is K = MedianIndex, then return MedianValue

  is K > MedianIndex ? then HighValue = MedianValue Else LowValue = MedianValue

  It will be faster without consuming memory

  2nd Order Binary Search:

  LowIndex=1 HighIndex=N

  Repeat Following 5 Steps until (LowIndex < HighIndex)

  Get Approximate DistrbutionPerUnit=(HighValue-LowValue)/(HighIndex-LowIndex)

  Get Approximate MedianValue = LowValue + (MedianIndex-LowIndex) * DistributionPerUnit

  Get NumberOfItemsWhichAreLessThanorEqualToMedianValue = K

  is (K=MedianIndex) ? return MedianValue

  is (K > MedianIndex) ? then HighIndex=K and HighValue=MedianValue Else LowIndex=K and LowValue=MedianValue

  It will be faster than 1st order without consuming memory

  We can also think of fitting HighValue, LowValue and MedianValue with HighIndex, LowIndex and MedianIndex to a
  Parabola, and can get ThirdOrder Binary Search which will be faster than 2nd order without consuming memory and
   so on...``` how bad is the estimate from class interval method? geometric median iterative weiszfeld, minheap
  and maxheapa combination built during accumulate but on the side, or some version of
  https://stdcxx.apache.org/doc/stdlibref/nth-element.html for accumulate or gather or somethign, ```mport heapq

  def find_median(arr):
  min_heap = [] # Min-heap for the larger half
  max_heap = [] # Max-heap for the smaller half

  for num in arr:
  heapq.heappush(max_heap, -heapq.heappushpop(min_heap, num))
  if len(max_heap) > len(min_heap):
  heapq.heappush(min_heap, -heapq.heappop(max_heap))

  if len(min_heap) > len(max_heap):
  return min_heap[0]
  else:
  return (min_heap[0] - max_heap[0]) / 2

  arr = [12, 3, 5, 7, 4, 19, 26]
  print(find_median(arr)) # Output: 7
  Copy
  Explanation

  Min-Heap and Max-Heap: Use two heaps to keep track of the smaller and larger halves of the array. The min-heap
  stores the larger half, and the max-heap (inverted to act as a max-heap) stores the smaller half.

  Balancing Heaps: For each element in the array, push it into the min-heap and then immediately pop the smallest
   element from the min-heap and push it into the max-heap. If the max-heap has more elements than the min-heap,
  move an element from the max-heap to the min-heap.

  Finding Median: If the min-heap has more elements than the max-heap, the median is the root of the min-heap.
  Otherwise, it is the average of the roots of both heaps.

  Considerations

  Time Complexity: The time complexity of this approach is O(n log n) due to heap operations.

  Space Complexity: The space complexity is O(n) as we store all elements in heaps.

  This method ensures that you can find the median efficiently without sorting the entire array.``` etc?

● This is the right question and it has a beautiful answer. The quick version: once you have parallel
  accumulate+gather, several of these "streaming median tricks" stop being tricks and become natural primitives —
  and a few of them give us things sketches genuinely can't. Let me walk through them through the
  accumulate+gather lens.

  The deep insight first

  Most of those streaming-median algorithms were invented under the constraint "one pass over the data, bounded
  memory, no random access." That constraint comes from era of disk-based or single-machine streaming. We don't
  have that constraint. We have parallel accumulate over the full dataset, multiple passes if we want, and
  arbitrary scatter/gather addressing.

  When you remove the streaming constraint, the cost-benefit landscape inverts. Some algorithms that were
  brilliant under streaming become unnecessary; others become more powerful than they were originally.

  Algorithm-by-algorithm

  Binary search on value (the "no memory" 1st-order method you cited)

  loop:
    midpoint = (low + high) / 2
    k = count(x <= midpoint)
    if k == target_rank: return midpoint
    if k > target_rank: high = midpoint else low = midpoint

  Through accumulate+gather: Each iteration is literally one accumulate(data, Grouping::All, Op::Add,
  Expr::Custom("x <= guess")). That's it. No sketch, no centroid, no compactor, no sort. One scatter-count per
  iteration.

  Convergence: O(log(value_range / ε_value)) iterations. For SIP-scale data with ε = 1 ULP precision in dollars
  over a 1e10 range: ~50 iterations. At 100K events/bucket × 50 iterations = 5M ops per quantile. Trivial.

  Properties:
  - Exact in the limit (value-converges to true median of values; rank-converges to true rank with one pass to
  count ties at the converged value)
  - Deterministic and bit-exact (no randomness, no order-dependence)
  - O(1) memory (just three f64s: low, high, midpoint)
  - Permutation-invariant by construction
  - Mergeable trivially: each shard's binary search is independent; combine by re-running on full data — OR keep
  running counts and converge globally
  - Composable with using(): trivially exposes precision_bits as a knob

  This is the cleanest median primitive imaginable for accumulate+gather. It's what KLL/DDSketch/sort were all
  trying to approximate. We can just do it directly.

  The only thing we sacrifice vs DDSketch is the bounded cost — a sketch ingests once and answers infinitely many
  quantile queries; binary search on value runs the loop fresh per query. For SIP where we ask 9 percentiles per
  bucket, that's 9× the loop count. Still trivial.

  Median of medians (deterministic linear selection)

  Through accumulate+gather: Each level is accumulate(data, Grouping::ByKey{chunk_id_of_5},
  Op::MedianOf5Elements). Median-of-5 is a fixed sequence of 6 comparisons — a tiny custom op. Then partition
  around the result via another accumulate. Recurse on the chosen partition.

  Properties:
  - O(n) deterministic worst-case (vs quickselect's O(n²) worst-case)
  - Each level is parallel; depth is O(log n)
  - Bit-exact deterministic (no randomness)
  - Returns the exact k-th order statistic, not approximate

  Where it wins over binary-search-on-value: when you want a guaranteed element of the input as the answer (median
   of an ordered set, not a value that might fall between two elements). For continuous data either approach
  works; for discrete or repeated-value data, median-of-medians gives you a true datum.

  Quickselect / nth_element

  Same shape as median-of-medians but with simpler pivot choice. With random pivot: average O(n) but O(n²) worst
  case. With median-of-medians as the pivot: deterministic O(n) (which IS just median-of-medians).

  Through accumulate+gather: Same partition-via-scatter structure. The ONLY downside vs median-of-medians is the
  worst-case bound; otherwise structurally the same.

  Two-heap (min-heap + max-heap)

  Maintains the running median as items arrive. Genuinely sequential — each insert depends on prior heap state.

  Through accumulate+gather: Doesn't fit naturally as a streaming primitive. Could be expressed as Kingdom C(A)
  with inner heap operations parallel within an insert, but the outer per-item-loop is sequential.

  When it's still useful: when the consumer needs the running median at every step (online queries during
  ingestion). Like Welford's algorithm gives running mean at every step. The two-heap version is the
  running-median analog.

  So WelfordObservations got built earlier as the streaming-mean tracker; the two-heap analog would be
  RunningMedianObservations or similar. Different shape from sketches because it's online/incremental, not
  finalize-time.

  Reservoir sampling + classical median

  Reservoir gives a uniform random sample; classical median sorts the sample.

  Through accumulate+gather: Reservoir sampling is sequential (each item's accept probability depends on stream
  position). Could parallelize via "weighted reservoir per shard, merge at end." Then sort the small reservoir and
   take the median.

  When it's useful: when you want a quick approximate median with confidence bounds and you have non-deterministic
   acceptable. Per locked vocabulary, randomness is opt-in (using(deterministic: false) or similar). For
  deterministic SIP work, this isn't the right path.

  Weiszfeld iteration

  Iterative approximation of the geometric median (L1-median in d dimensions). Each iteration is a Kingdom A
  weighted mean. Outer loop is C-convergent.

  For univariate median: overkill — binary search on value is simpler and faster.

  For multivariate L1-median (e.g., geometric center of a 2D point cloud): this is the right algorithm. Becomes a
  recipe for that specific use case.

  Histogram / class interval

  Equal-width or equal-count bins, find crossing bin, interpolate within.

  Through accumulate+gather: Two-pass:
  1. accumulate(data, ByKey{bin(x)}, Add) → bucket counts
  2. Prefix scan over counts → find crossing bucket → interpolate

  This is literally what DDSketch is — the bin function bin(x) = ⌈log_γ(x)⌉ with its native sign-handling. Linear
  bucketing is a different DDSketch variant. Equal-count bucketing is essentially what t-digest does with its
  k-bound.

  So histogram-based methods aren't a separate option — they're a generalization of which bin function the sketch
  uses. DDSketch is histogram with logarithmic bins; we could trivially add a LinearBinSketch variant as a sibling
   for cases where linear bins are more natural (uniform-distribution data, or scientific instruments with linear
  ADCs).

  The equivalences you're sensing

  You're right that several of these algorithms are the same thing under transformation. Mapping them out:

  ┌───────────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │         Algorithm         │                   Equivalent under accumulate+gather                   │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Histogram + interpolation │ DDSketch with linear bins                                              │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ t-digest                  │ DDSketch with adaptive non-uniform bins                                │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Reservoir sampling + sort │ Approximate KLL with k=1 compactor                                     │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Welford running mean      │ Reservoir sampling with k=1 (just keep the running average)            │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Two-heap streaming median │ Online version of partition-around-pivot                               │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Quickselect               │ Median-of-medians with random pivot (worst-case O(n²) instead of O(n)) │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Median-of-medians         │ Recursive accumulate over fixed-size chunks                            │
  ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Binary search on value    │ Iterated count-leq accumulate, no scaffolding                          │
  └───────────────────────────┴────────────────────────────────────────────────────────────────────────┘

  The DEEPEST equivalence: a sketch is "how do I avoid running through the data multiple times?" Once we can run
  multiple passes cheaply (because accumulate is parallel and cheap), some sketches lose their reason for
  existing. Binary search on value is the parallel-friendly answer to "find a quantile" that needs no sketch at
  all.

  What I'd build

  In priority order:

  1. quantile_binary_search(data, q, value_precision) recipe — pure accumulate, deterministic, exact-to-precision,
   bit-exact, permutation-invariant, no sketch needed. The cleanest possible median/quantile primitive. ~30 lines.

  This replaces sketches for the SIP use case. Sketches still earn their place when (a) you want to ingest once
  and answer many queries cheaply (precomputed sketch state), or (b) you genuinely need bounded memory for
  unbounded streams. For SIP's hourly buckets, neither applies — we can re-run the binary search per query.

  2. median_of_medians(data, target_rank) recipe — deterministic O(n) selection that returns an exact element of
  the input. The k-th order statistic primitive. ~50 lines.

  3. RunningMedianObservations struct — two-heap-style sibling of WelfordObservations for streaming queries during
   ingestion. For consumers that need median at every step. ~80 lines.

  4. Add LinearBinSketch variant — DDSketch with linear bucketing, for uniform-distribution data where log
  bucketing wastes resolution. ~150 lines.

  5. weiszfeld_geometric_median(points_d, tol) recipe — for multivariate L1-median in 2D/3D point clouds.
  Specialized but not a SIP need; build when it's actually called for.

  My recommendation

  Start with #1. Binary search on value is, to my reading, the correct way to compute a quantile in our model.
  Sketches are great when you want to amortize ingestion across many queries; binary search is better when you
  want exact and don't mind multi-pass.

  For SIP specifically:
  - Header percentiles (F11..F19) computed via binary search → exact to ULP, bit-identical regardless of
  distribution, regardless of merge order
  - CVaR (mean of values below quantile q) also via binary search → exact threshold + Kulisch mean of
  below-threshold subset
  - Hill estimator → either binary search for the threshold, or order-statistic version via median-of-medians

  This might mean we don't need sketches as the locked default at all — sketches become the using(approximate:
  true) opt-in for when amortization matters more than exactness.

  The bigger insight: by faithfully expressing things in accumulate+gather, we discover that "the streaming-median
   problem" is largely an artifact of streaming constraints we don't have. The right primitive for our model is
  binary search on value plus, when needed, deterministic order-statistic selection.

  Want me to build quantile_binary_search and verify the property claims (bit-exact merge, permutation-invariance,
   exact-to-ULP)? And then we can re-evaluate which sketches we actually need vs which are vestigial.

✻ Cooked for 2m 4s

※ recap: Building tambear's math layer for SIP; just finished comparing 4 quantile sketches and proposed building
  `quantile_binary_search` instead since accumulate+gather makes sketch-free exact medians cheap. Next: get your
  go on building it and re-evaluating whether sketches stay locked default. (disable recaps in /config)

❯ and for what we were doing with KLL and replaced with DDSketch as default, was that ONLY quantiles? somehow I
  was thinking those applied to more than that

● Good catch — I was being narrow. Sketches do more than quantile queries; the user's instinct that we'd want them
   for more was right.

  What we actually used sketches for

  In the three SIP recipes I built (cvar, price_percentiles, hill_estimator_streaming), the sketch IS used purely
  for quantile estimation — find the value at q, then the recipe does its own follow-up pass for the mean /
  log-mean. So in those recipes specifically, binary-search-on-value would replace the sketch cleanly.

  What sketches give us BEYOND quantile estimation

  This is where the picture gets richer, and it's especially true for DDSketch and t-digest:

  Things a sketch can answer that binary-search-on-value can't (cheaply)

  A sketch is a compressed empirical distribution — once built, it answers many questions about the data without
  re-passing. Binary search answers ONE question per pass.

  For DDSketch specifically (and to a lesser extent t-digest), the bucket structure exposes:

  - count_in_range(a, b) — how many observations fell in [a, b]? Sum the bucket counts that intersect the range.
  O(buckets) lookup, no extra pass.
  - cdf(x) — P(X ≤ x). Cumulative count up to bucket containing x. O(buckets).
  - mode_bucket() — which value range was most frequent? Argmax over bucket counts. O(buckets).
  - approximate_distribution() — emit the bucket centers + counts as a histogram. Compressed view of the shape.
  - tail_mean_below(q) / tail_mean_above(q) — already what CVaR computes, but available cheaply once the sketch is
   built.
  - trimmed_mean(q_low, q_high) — mean over the middle portion. Walk buckets between two quantile thresholds;
  weight by bucket centers.
  - distribution_distance(other_sketch) — KS-statistic, Wasserstein, KL divergence between two empirical
  distributions can be computed from their sketches (especially DDSketch where buckets align deterministically
  across sketches).

  Things that benefit from sketch's amortization

  Built once, queried often:
  - 9 SIP percentiles — sketch built once, 9 queries cheap. Binary search would be 9 separate full-data passes.
  - CVaR + 9 percentiles + Hill in one recipe call — all from one sketch ingestion.
  - Multi-quantile QQ plots — compare two sketches at many quantiles at once.

  Things that benefit from sketch's mergeability across shards

  This is where DDSketch is exceptional:
  - Per-bucket-by-time-window distribution tracking — every 100ms bucket has its own sketch; merge them across the
   hour for the hour-level distribution; merge hours across the day for daily; merge days across the month. The
  sketch is the right unit of "compressed distribution we ship between aggregation levels."
  - Cross-shard distribution comparison — shard A and shard B each build a sketch; ship both to a coordinator;
  coordinator computes distribution-divergence directly from sketches without re-streaming raw data.
  - Stream-then-query — a long-running ingestion populates a sketch incrementally; queries against the sketch are
  cheap independent of how many items were ingested.

  Things sketches do that we hadn't yet built recipes for

  These are likely future SIP-adjacent or general tambear needs:
  - Distribution change-point detection — run two sketches over consecutive windows; flag when their distance
  exceeds threshold.
  - Anomaly detection — cdf(x) < ε or cdf(x) > 1 - ε flags x as a tail observation.
  - Density estimation — bucket counts ARE a histogram density estimate.
  - Stratified sampling weights — quantile-defined strata for subsequent processing.

  Reframing the binary-search vs sketch trade-off

  Now I see the real shape:

  Binary-search-on-value is best when:
  - You want ONE specific quantile exactly
  - You can afford log(1/ε) passes
  - You don't need to ship the result around or query it repeatedly
  - Use case: "give me the median of this bucket, right now"

  Sketches are best when:
  - You want MANY queries against the same distribution
  - The distribution itself is a useful object (mergeable, shippable, queryable)
  - You need to compare distributions or compute derived statistics beyond simple quantiles
  - Use case: "build a compressed view of this distribution that future code can ask many questions of"

  For SIP's column-graph fields specifically — which is "compute these 9 percentiles + CVaR + Hill from each
  bucket's data, then never touch that bucket again" — both work. Binary search has cleaner determinism
  guarantees; sketches give cheaper-per-query if we want lots of queries from the same data.

  For SIP's downstream architecture — Merkle anchoring + distributed merge + cross-bucket / cross-hour aggregation
   — sketches earn their place. The sketch IS the compressed distribution that ships between aggregation levels.
  Binary search has nothing to ship.

  Recommendation update

  Build both. They're not redundant.

  - quantile_binary_search — the exact-single-query primitive (planned next)
  - KLL/GK/t-digest/DDSketch — the compressed-distribution primitives we already have
  - Add DDSketch-derived methods (count_in_range, cdf, tail_mean, mode_bucket, distribution_distance) — these
  unlock the distribution-as-object use cases beyond quantile. ~50-100 lines of new methods on DdSketch plus
  tests.

  For SIP recipes specifically, recipes can pick:
  - cvar(returns, tail_pct).using(strategy: "exact_binary_search") — bit-exact, single-quantile-per-call
  - cvar(returns, tail_pct).using(strategy: "ddsketch") — amortized, mergeable, shippable

  Both are useful. Both should exist. The user was right to remember sketches mattered for more than just quantile
   queries — that's the distribution-as-object dimension I underweighted.

  So the actual menu is:
  1. Build quantile_binary_search (exact quantile primitive)
  2. Add the distribution-query methods to DDSketch (count_in_range, cdf, tail_mean, mode_bucket,
  distribution_distance)
  3. Decide per-recipe which strategy is the recipe's default and which are using() overrides

  Want me to do (1) and (2) together? They're complementary; both extend our quantile/distribution toolkit in
  non-overlapping ways.

✻ Cogitated for 1m 7s

❯ and is the binary search version in accumulate+gather deterministic? same regardless of what hardware run on?

● Yes, and it's stronger than even DDSketch's determinism guarantee. Here's why.

  The hot loop

  loop:
    midpoint = (low + high) / 2.0
    k = accumulate(data, Grouping::All, Op::Add, Expr::Custom("x <= midpoint"))  // count
    if k == target_rank: return midpoint
    if k > target_rank: high = midpoint else: low = midpoint

  Per iteration:
  - One scalar arithmetic (midpoint computation) — IEEE 754, bit-exact on any hardware ✓
  - One accumulate over the data — counting how many x <= midpoint
  - One integer comparison + assignment

  The key insight: the reduction is integer-valued

  The accumulate step sums a bunch of 0s and 1s (the mask x <= midpoint). The result is mathematically an integer
  in [0, n].

  Sum of 0/1 values is exact in f64 for any reasonable n. f64 holds integers exactly up to 2^53 ≈ 9 quadrillion.
  SIP's tick volume isn't getting near that.

  This means the accumulate is bit-exact regardless of strategy:
  - Naive atomicAdd on GPU? Same exact integer count, regardless of which thread won which add
  - Kulisch on CPU? Same answer (Kulisch is exact, but here naive is also exact)
  - Tree-merge across shards? Same answer (integer addition is associative)
  - Any thread count, any backend, any merge order — same exact integer

  So we don't even need Kulisch for this specific case. The accumulate's integer-valued nature gives us
  bit-exactness for free, no compensated arithmetic needed.

  Iteration-count determinism

  The convergence path is deterministic given the data: same midpoint → same count → same bracket update → same
  next midpoint. Two runs on the same data converge in the same number of iterations to the same value, on any
  hardware.

  Comparison to the sketches' determinism

  ┌─────────────────────────────────┬──────────────────────────────┬───────────────┬─────────────────────────┐
  │            Property             │             KLL              │   DDSketch    │ Binary-search on value  │
  ├─────────────────────────────────┼──────────────────────────────┼───────────────┼─────────────────────────┤
  │ Run-to-run on same hardware     │ ✓                            │ ✓             │ ✓                       │
  ├─────────────────────────────────┼──────────────────────────────┼───────────────┼─────────────────────────┤
  │ Across thread counts            │ ~ (depends on Kulisch)       │ ✓             │ ✓                       │
  ├─────────────────────────────────┼──────────────────────────────┼───────────────┼─────────────────────────┤
  │ Across backends (CPU/GPU/door)  │ ~ (depends on Kulisch impls  │ ✓             │ ✓ even without Kulisch  │
  │                                 │ all matching)                │               │                         │
  ├─────────────────────────────────┼──────────────────────────────┼───────────────┼─────────────────────────┤
  │ Permutation-invariant           │ ✗                            │ ✓             │ ✓                       │
  ├─────────────────────────────────┼──────────────────────────────┼───────────────┼─────────────────────────┤
  │ Bit-exact mergeable across      │ ✗                            │ ✓             │ ✓                       │
  │ shards                          │                              │               │                         │
  ├─────────────────────────────────┼──────────────────────────────┼───────────────┼─────────────────────────┤
  │ Doesn't need Kulisch for det.   │ ✗                            │ ✗ (uses       │ ✓ (integer-valued       │
  │ guarantee                       │                              │ BTreeMap)     │ accumulate)             │
  └─────────────────────────────────┴──────────────────────────────┴───────────────┴─────────────────────────┘

  The last row is what makes this special. Even our deterministic-Add infrastructure is unnecessary for
  binary-search-on-value's determinism. The mask-then-count pattern is determinism-bulletproof at the IEEE-754
  level alone.

  Edge cases I want to flag

  - NaN inputs: NaN <= midpoint is always false in IEEE 754. So NaN values count as 0 — they get effectively
  skipped. This matches our locked NaN-skip policy automatically, no extra logic needed.
  - ±Inf inputs: -Inf <= midpoint is true, +Inf <= midpoint is false. So Inf values affect the count if not
  pre-filtered. The recipe should apply the same is_finite() gate at the expression level (x.is_finite() && x <=
  midpoint) — keeps it on the locked Inf-skip path.
  - Ties at the converged midpoint: if many values equal midpoint, the count could include all of them.
  Convergence still works; it just takes a few more iterations to tighten the bracket. The returned value is still
   bit-exact deterministic.

  Kingdom classification

  C(A) — outer convergence loop is sequential (depth = O(log(1/ε))), inner accumulate is Kingdom A and fully
  parallel.

  For SIP at 100K events/bucket with 50 iterations to converge to ULP precision: 5M ops total. On modern CPU
  that's milliseconds. On GPU we'd dispatch 50 small accumulates — still trivially fast. The convergence depth is
  the bottleneck, not the data passes.

  So the answer is yes, definitively

  Binary search on value gives us:
  - Bit-exact across any hardware (without needing Kulisch)
  - Permutation-invariant (the count is multiset-determined)
  - Mergeable trivially (counts add)
  - Exact to ULP precision in bounded iteration count
  - Compatible with locked NaN/Inf skip semantics for free

  For SIP's Merkle anchoring story: this is the cleanest possible quantile primitive. The output value is provably
   bit-identical regardless of execution topology, because the computation reduces to "count of integer-valued
  mask" which is IEEE-exact under any accumulation strategy.

  Want me to build it now and replace the sketch dispatch in cvar / price_percentiles / hill_estimator_streaming
  with this as the new default, keeping sketches as using(strategy: ...) overrides for the amortization /
  distribution-as-object cases?

✻ Baked for 57s

❯ so then for this table:       Algorithm         │                   Equivalent under accumulate+gather
            │
    ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
    │ Histogram + interpolation │ DDSketch with linear bins                                              │
    ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
    │ t-digest                  │ DDSketch with adaptive non-uniform bins                                │
    ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
    │ Reservoir sampling + sort │ Approximate KLL with k=1 compactor                                     │
    ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
    │ Welford running mean      │ Reservoir sampling with k=1 (just keep the running average)            │
    ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
    │ Two-heap streaming median │ Online version of partition-around-pivot                               │
    ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
    │ Quickselect               │ Median-of-medians with random pivot (worst-case O(n²) instead of O(n)) │
    ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
    │ Median-of-medians         │ Recursive accumulate over fixed-size chunks                            │
    ├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
    │ Binary search on value    │ Iterated count-leq accumulate, no scaffolding -- do we need to implement
  everythign in the left column as MATH when it can be implemented as a specific composition of things to do what
   we say in teh right column under accumulate+gather? or can teh left column be a recipe that lets the user
  specify what they want in the language THEY want it, but WE know that under the hood that means t-digest is
  doing ddsketch with adaptive non-uniform bins? etc

● You've put your finger on a really important architectural choice. Let me revise that table because I was being
  a bit sloppy — some of those "equivalences" are exact and some are spiritual cousins. The right design depends
  on which.

  Three categories, not one

  ┌─────────────────────────────────┬──────────────────────────────────────────────────────────────────────────┐
  │      Kind of relationship       │                        Right architectural answer                        │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Same kernel, different          │ One implementation, multiple recipe wrappers exposing it under each      │
  │ parameter                       │ literature name                                                          │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Same shape, distinct math       │ Separate implementations with cross-reference docs; ideally a shared     │
  │                                 │ trait/interface                                                          │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Same name as our work, by       │ Just our work, no separate implementation                                │
  │ tautology                       │                                                                          │
  └─────────────────────────────────┴──────────────────────────────────────────────────────────────────────────┘

  Let me re-examine each row honestly:

  ┌────────────────────────────┬─────────────────────────────────────────────────────────────┬───────────────┐
  │            Row             │                           Reality                           │   Category    │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
  │ Histogram + interpolation  │ Exactly true. Same kernel: bucket-count scatter +           │ Same kernel,  │
  │ = DDSketch with linear     │ crossing-bucket interpolation. Difference is purely the bin │ parameter     │
  │ bins                       │  function.                                                  │               │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
  │ t-digest = DDSketch with   │ Spiritual cousin. Both are compressed histograms. But       │ Distinct      │
  │ adaptive bins              │ t-digest's centroid merging under a k-bound is genuinely    │ algorithms    │
  │                            │ different math from DDSketch's fixed log buckets.           │               │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
  │ Reservoir + sort = KLL     │ Distant cousin. KLL has hierarchical levels (weight         │ Distinct      │
  │ with k=1                   │ 2^level); reservoir is flat. They share "stochastic         │ algorithms    │
  │                            │ subsampling" but the math diverges.                         │               │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
  │ Welford = Reservoir with   │ I was wrong. Welford is the exact running mean. Reservoir-1 │ Not           │
  │ k=1                        │  keeps one random sample. Different things. Apologies.      │ equivalent    │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
  │ Two-heap = Online          │ Spiritual cousin. Both find the boundary. Heap is O(log n)  │ Distinct      │
  │ partition                  │ per insert; partition is O(n) per pivot. Different          │ algorithms    │
  │                            │ complexity profiles.                                        │               │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
  │ Quickselect =              │ Exactly true. Same kernel: partition-and-recurse.           │ Same kernel,  │
  │ Median-of-medians with     │ Difference is purely pivot selection.                       │ parameter     │
  │ random pivot               │                                                             │               │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
  │ Median-of-medians =        │                                                             │               │
  │ Recursive accumulate over  │ Tautology. That's the implementation, not an alternative.   │ Just our work │
  │ chunks                     │                                                             │               │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
  │ Binary search on value =   │ Tautology. Same.                                            │ Just our work │
  │ Iterated count-leq         │                                                             │               │
  └────────────────────────────┴─────────────────────────────────────────────────────────────┴───────────────┘

  So the architectural answer for tambear breaks down as:

  For "same kernel, different parameter" cases

  Yes, exactly your design. One implementation, multiple recipe wrappers exposing it under whatever literature
  name the user reaches for. Recipes are flat composition by name — the whole point of the recipe tier per locked
  vocabulary is that users can find functionality by the name they know.

  // One kernel:
  fn binned_histogram_quantile(data, q, bin_fn) -> f64 { ... }

  // Three named recipe wrappers, all dispatching to it:
  recipes/ddsketch_quantile.rs       → calls with log bin_fn
  recipes/linear_histogram_quantile.rs → calls with linear bin_fn
  recipes/equal_count_histogram_quantile.rs → calls with equal-count bin_fn

  // Same pattern for partition-recurse kernel:
  fn partition_select(data, k, pivot_strategy) -> f64 { ... }

  recipes/quickselect.rs           → calls with random_pivot
  recipes/median_of_medians.rs     → calls with median_of_5_pivot
  recipes/introselect.rs           → calls with hybrid_pivot (quickselect → median-of-medians fallback)

  A user reaching for quickselect gets it. A user reaching for median_of_medians gets it. Both find what they
  expect; under the hood it's one well-tested kernel with a parameter choice.

  This is exactly aligned with locked vocabulary: recipes are flat, named compositions. They can wrap shared
  kernels. Discoverability via literature name is the goal.

  For "spiritual cousin, distinct math" cases

  Separate implementations, but exposed via a common trait so they're swappable. Already have this with the
  QuantileSketch trait — KLL/GK/t-digest/DDSketch all implement it; users pick via using(sketch: ...). We could
  extend it to cover quantile-via-binary-search, quantile-via-partition-select, etc. Each is its own
  implementation; the trait makes them substitutable for the same query.

  For "tautology" cases

  These ARE the implementation. We just give them their literature name as a recipe. Binary-search-on-value gets a
   recipe called quantile_binary_search (or maybe quantile_exact since "binary search" isn't really the
  user-facing name for the property they care about). Median-of-medians gets a recipe called median_of_medians.

  What this looks like for the catalog

  We'd end up with something like:

  Kernels (Tier 1 / Tier 4 implementation, depending on shape):
  - binned_histogram_quantile_kernel (parameterized by bin function)
  - partition_select_kernel (parameterized by pivot strategy)
  - quantile_via_binary_search_kernel (no parameters; this IS the algorithm)
  - kll_sketch, gk_sketch, tdigest_sketch, ddsketch (genuinely distinct sketch primitives)

  Recipes (Tier 4, named compositions):
  - ddsketch_quantile → binned kernel + log bins
  - linear_histogram_quantile → binned kernel + linear bins
  - equal_count_histogram_quantile → binned kernel + equal-count bins
  - quickselect → partition kernel + random pivot
  - median_of_medians → partition kernel + median-of-5 pivot
  - nth_element → quickselect (alias)
  - quantile_exact → binary-search kernel
  - quantile_kll, quantile_gk, quantile_tdigest → wrap the four sketch types

  A user reaches for the recipe named after the algorithm they know. They get it. Under the hood, it's the right
  shared kernel doing the work.

  The discoverability dimension

  This is what makes your design better than just "have one quantile recipe with a strategy parameter." Users
  searching the codebase for quickselect find it as a recipe — they don't have to know that under the hood it's
  partition_select(strategy: random_pivot). The literature name IS in the code, surfaced by grep or the IDE recipe
   browser.

  The recipe file's docstring explains the relationship: "This is quickselect (Hoare 1961). Implementation note:
  composes the partition-select kernel with random pivot strategy. For deterministic O(n) worst-case, use
  median_of_medians (same kernel, different pivot)."

  Future agents and users find what they expect by name AND see the relationship to the underlying shared kernel.

  Short answer

  Yes — recipes that share a kernel are EXACTLY the right pattern. Build the kernel once, wrap it under each name
  people would reach for. Add a doc paragraph that names the shared kernel and links to siblings.

  For "spiritual cousin" cases (different math), keep separate implementations but unify under traits where the
  substitution makes sense. Users still pick by name, but the implementations are genuinely different.

  This is what the recipe tier was designed for, and your instinct just made me realize we should build it that
  way for every shared-kernel case, not just sketches.

  Want me to start implementing this — beginning with the binned-histogram kernel + 2-3 recipe wrappers +
  binary-search exact quantile + partition-select kernel + 2 pivot-strategy recipe wrappers?

✻ Sautéed for 1m 16s

※ recap: Building tambear's quantile/select stack so users can call algorithms by literature name (quickselect,
  median_of_medians, ddsketch_quantile, etc.) backed by shared kernels. Next: implement the binned-histogram
  kernel, partition-select kernel, exact-quantile binary search, and the recipe wrappers for each.

❯          Row             │                           Reality                           │   Category    │
    ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
    │ Histogram + interpolation  │ Exactly true. Same kernel: bucket-count scatter +           │ Same kernel,  │
    │ = DDSketch with linear     │ crossing-bucket interpolation. Difference is purely the bin │ parameter     │
    │ bins                       │  function.                                                  │               │
    ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
    │ t-digest = DDSketch with   │ Spiritual cousin. Both are compressed histograms. But       │ Distinct      │
    │ adaptive bins              │ t-digest's centroid merging under a k-bound is genuinely    │ algorithms    │
    │                            │ different math from DDSketch's fixed log buckets.           │               │
    ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
    │ Reservoir + sort = KLL     │ Distant cousin. KLL has hierarchical levels (weight         │ Distinct      │
    │ with k=1                   │ 2^level); reservoir is flat. They share "stochastic         │ algorithms    │
    │                            │ subsampling" but the math diverges.                         │               │
    ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
    │ Welford = Reservoir with   │ I was wrong. Welford is the exact running mean. Reservoir-1 │ Not           │
    │ k=1                        │  keeps one random sample. Different things. Apologies.      │ equivalent    │
    ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
    │ Two-heap = Online          │ Spiritual cousin. Both find the boundary. Heap is O(log n)  │ Distinct      │
    │ partition                  │ per insert; partition is O(n) per pivot. Different          │ algorithms    │
    │                            │ complexity profiles.                                        │               │
    ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
    │ Quickselect =              │ Exactly true. Same kernel: partition-and-recurse.           │ Same kernel,  │
    │ Median-of-medians with     │ Difference is purely pivot selection.                       │ parameter     │
    │ random pivot               │                                                             │               │
    ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
    │ Median-of-medians =        │                                                             │               │
    │ Recursive accumulate over  │ Tautology. That's the implementation, not an alternative.   │ Just our work │
    │ chunks                     │                                                             │               │
    ├────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────┤
    │ Binary search on value =   │ Tautology. Same.                                            │ Just our work │
    │ Iterated count-leq         │                                                             │               │
    └────────────────────────────┴─────────────────────────────────────────────────────────────┴───────────────┘


    so a few questions.
    if t-digest does centroid merging under a k-bound and ddsketch does fixed log buckets, but otherwise they're
  both compressed histograms,
    can they still *BE* parameters or custom compilations of the same underlying primitive? ddsketch as a recipe
  not a primitive. the fixed log bucket is a prim, the centroid thign is a prim, the histogram (plus various
  options) gives us ANY histogram including allowing different kinds of compression
    then t-digest is also a recipe built on the primitives?

    can stochastic subsampling be a primitive, and hierarchical leveling is a primitive or something, and
  reservoirsort and KLL are recipes that compose all the decomposable pieces of shared math?

    is the only difference between 2heap and online partition the complexity profile? are they producing
  equivalent output from 2 different methods? or they ahve other differences tha tmatter?



● Three good questions. Let me take them in order with the right level of nuance.

  Q1: t-digest and DDSketch as recipes on a shared "compressed histogram" primitive

  Yes — and this is actually the cleaner architecture. Let me articulate the unifying primitive.

  A "compressed histogram" primitive has three parameter slots:

  1. Bucket assignment: how does an incoming value map to a bucket?
    - Log: bucket_id(x) = ⌈log_γ(|x|)⌉ (signed)
    - Linear: bucket_id(x) = ⌊x / width⌋
    - Singleton-then-merge: bucket_id(x) = unique (each insert is its own centroid)
    - User-defined: any f64 → BucketKey function
  2. Compression policy: when buckets accumulate, how do they consolidate?
    - None: never compress (DDSketch's policy — buckets are fixed once defined)
    - k-bound merge: walk adjacent buckets in value order, merge under a bound function (t-digest's policy with
  k(q) = (δ/(2π))·arcsin(2q−1))
    - Max-bucket-count: enforce a max count, merge nearest pair when exceeded
    - Equal-count rebalance: maintain ~equal counts per bucket
    - User-defined
  3. Interpolation policy: at quantile query time, how do we estimate the answer between buckets?
    - Midpoint anchor (DDSketch)
    - Linear between adjacent centroid means (t-digest)
    - Step function (no interpolation)

  With this primitive, the recipes write themselves:

  // Same kernel, three named recipes:
  recipes/ddsketch_quantile.rs:
      composes Histogram(bucket=Log{γ}, compress=None, interp=Midpoint)

  recipes/tdigest_quantile.rs:
      composes Histogram(bucket=Singleton, compress=KBound{δ}, interp=LinearBetweenMeans)

  recipes/linear_histogram_quantile.rs:
      composes Histogram(bucket=Linear{width}, compress=None, interp=Midpoint)

  recipes/equal_count_histogram_quantile.rs:
      composes Histogram(bucket=Linear{width}, compress=EqualCountRebalance, interp=Midpoint)

  The ONE primitive at Tier 1 is CompressedHistogram<B, C, I> parameterized by those three policies. All four
  recipes are thin wrappers. Other future recipes (variable-width log buckets, exponential compression, etc.) drop
   in trivially.

  This is genuinely better than what we have today. It collapses ~600 lines of three separate sketch
  implementations into one ~250-line generic primitive plus 4 thin recipes (~20 lines each).

  KLL and GK don't fit this shape — they're not compressed histograms. KLL is hierarchical compactors with
  stochastic compaction. GK is sorted-tuple lists with bound-based merge. They genuinely need their own
  primitives.

  Q2: Stochastic subsampling and hierarchical leveling as primitives

  Yes-ish, but with care about not over-abstracting. Honest breakdown:

  Stochastic subsampling has a real shared shape:
  - Random-replacement reservoir (uniform sample of the stream)
  - KLL's coin-flip-during-compaction (which half of a level promotes)
  - Other algorithms: weighted reservoir, A-Res, A-Chao, priority-sampling

  The shared abstraction: "given a stream of items and a deterministic PRNG, decide which to keep." That's a
  primitive. We could call it DeterministicReservoir or similar. Reservoir-sampling-and-sort is one recipe over
  it; KLL's coin-flip-compaction is not directly one — KLL's compaction has a hierarchical structure that differs.

  Hierarchical leveling shows up in:
  - KLL compactor levels (each level represents 2^level items)
  - Skip lists (each level is a skip pointer)
  - LSM trees (each level is bigger and rarer-access)
  - Wavelet trees, segment trees

  These are all "data partitioned across levels with promotion rules." But the OPERATIONS on each are distinct:
  KLL queries quantiles, skip lists query ordered keys, LSM trees query KV pairs.

  Honest answer:

  - DeterministicReservoir primitive — yes, real shared shape, worth pulling out. Used by reservoir sampling,
  weighted sampling, priority sampling, and (with the right shape) anywhere we need bounded-memory stochastic
  compression.
  - "Hierarchical leveling" as a shared primitive — premature. We have ONE current consumer (KLL). Until we add
  skip lists / LSM / wavelet trees, this is a structural pattern not a primitive. Build the primitive when the
  second consumer needs it, not before.

  So my recommendation: extract DeterministicReservoir now (we'll want it for many sampling-based recipes); leave
  KLL's hierarchical compactor as KLL-internal until a second consumer appears for it.

  The principle here: anti-YAGNI says build siblings we can imagine. It does NOT say abstract patterns to capture
  single consumers. Reservoir is justified because we can name 5+ recipes that'll use it. Hierarchical leveling
  has 1 consumer in tambear today; abstracting it is fishing for a use case.

  Q3: Two-heap vs online partition

  Same output on the same fixed input. DIFFERENT query patterns. The choice depends on what kind of question
  you're answering.

  ┌─────────────────────────┬────────────────────────────────────────┬────────────────────────────────────────┐
  │        Property         │                Two-heap                │            Online partition            │
  ├─────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
  │ Output on fixed input   │ Same value                             │ Same value                             │
  ├─────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
  │ Insertion-order         │ No (each insert mutates heap state)    │ Yes (final partition only depends on   │
  │ independence            │                                        │ multiset)                              │
  ├─────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
  │ Memory                  │ O(n) — all values stored across the    │ O(1) extra — works in place over the   │
  │                         │ two heaps                              │ existing data                          │
  ├─────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
  │ Per-insert cost         │ O(log n) heap balance                  │ N/A (not insert-driven)                │
  ├─────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
  │ Per-query cost          │ O(1) — just look at heap tops          │ O(n) — re-runs partition each query    │
  ├─────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
  │ Streaming online        │ Yes — query the median of the current  │ No — needs the full batch in hand      │
  │ support                 │ prefix at any time                     │                                        │
  ├─────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
  │ Mergeability            │ Hard (combining two pairs of heaps     │ Trivial — re-run on combined data      │
  │                         │ non-trivially)                         │                                        │
  ├─────────────────────────┼────────────────────────────────────────┼────────────────────────────────────────┤
  │ Fits accumulate+gather  │ Awkwardly (per-insert sequential heap  │ Naturally (partition IS                │
  │                         │ ops)                                   │ scatter-by-comparison)                 │
  └─────────────────────────┴────────────────────────────────────────┴────────────────────────────────────────┘

  So they're answering different variants of the same question:

  - Two-heap answers: "What's the median of the stream prefix observed so far?" — running query, updates with
  every insert, available cheaply at any moment.
  - Partition answers: "What's the median of this fixed batch?" — single answer over a known dataset.

  For SIP per-bucket-finalize: partition (or binary-search-on-value) is natural — we have the bucket's tick batch,
   we want one median, then we move on.

  For SIP-style real-time monitoring: two-heap is natural — running median visible at every tick arrival without
  recomputation.

  The two-heap streaming median is actually closer in role to WelfordObservations (which gives running mean at
  every step) than to the batch quantile primitives. It's an online observable that maintains a running median as
  a side state during ingestion. Conceptually it's a RunningMedianObservations sibling of Welford.

  So for the tambear catalog:

  - Batch median / quantile → recipe(s) over quantile_binary_search kernel, or partition_select kernel, or
  compressed_histogram kernel — pick by query pattern (single-exact vs multi-quantile vs
  amortized-distribution-as-object).
  - Online median during ingestion → RunningMedianObservations sibling of Welford.

  These aren't competing — they're different tools for different question patterns.

  Concrete proposal

  Given all three answers, here's what I think the catalog should look like:

  New primitives:
  1. CompressedHistogram<B, C, I> — replaces the ~600 lines of separate DDSketch/t-digest implementations with one
   generic primitive
  2. PartitionSelect kernel — partition-and-recurse for k-th order statistic
  3. QuantileBinarySearch kernel — iterated count-leq for exact quantile
  4. DeterministicReservoir — bounded-memory sampling primitive
  5. RunningMedianObservations — sibling of Welford for online median tracking

  Recipes (named compositions over these primitives):
  - Quantile/histogram family: ddsketch_quantile, tdigest_quantile, linear_histogram_quantile,
  equal_count_histogram_quantile, quantile_exact (binary search), quantile_kll (still its own primitive)
  - Selection family: quickselect, median_of_medians, nth_element
  - Sampling family: reservoir_sample, weighted_reservoir
  - Streaming family: running_median (using RunningMedianObservations)

  Refactor needed: the existing DDSketch and t-digest implementations would collapse into the CompressedHistogram
  primitive plus their two recipe wrappers. KLL and GK stay as their own primitives (they don't fit the histogram
  shape). DDSketch becomes a recipe, not a primitive.

  This is a meaningful refactor — 3 commits or so — but ends up with a cleaner architecture and ships several new
  primitives the catalog will reach for repeatedly. Plus it's a worked example of "literature-name recipes that
  share kernels."

  Want me to start with the CompressedHistogram primitive + the 4 recipe wrappers? That gives the immediate
  reduction in code-line duplication and demonstrates the shared-kernel pattern, then we can layer the other
  pieces (binary-search, partition-select, reservoir, running-median) on the same model.

✻ Worked for 1m 53s

❯ i'm going to push back a little on the wait for 2nd consumer. we're implementing THE WORLD everything that
  exists in math will get built. not maybe. and some insanely large % of it before launch. msut be useful for at
  least two? (idk we havne't decided this before just spitalling) or more entire fields of study (mathematics +
  genomics, or stats+bayesstats + theoretical physics) or something before it's worth launchign the product. they
   can be the smallest fields, but we need to be able to launch fully killing the comptuational software for an
  entire field tog et the funding and institutional pressure to force mass adoption by a field.   no-tech-debt +
  insane-extreme-anti-YAGNI or replace YAGNI with YAWNI YOU ALWAYS WILL NEED IT or something more appropriate.
  we're thinking about it now. we have the context and udnerstanding of what and why now. we will NEVER have this
   moment again.      for the hierarchical leveling/operations on each are dinstinct etc. KLL queries quantiles,
  skip lists query ordered keys, LMS trees query KV pairs. it sounds liek they do sahr eap rimitive but run on
  different DATA entirely. if we need quantiles, generat ethem in an earlier accumulate step, same for if we need
   ordered keys or kv pairs or something. we push them into the earliest accumulate where we CAN get themf rom
  the input data and keep themf or when we need them later in the pipeline. KLL is a recipe that gets the
  quantiles with wahtever method(s) with wahtever parameters etc that are in the primitive(s) used for that, and
  then it does somethign else, and then something else, each a primitive that can be pushed into whichever
  accumulate+gather pass is the next earliest that can use the previous passes data to dot he next step. we do
  repeated accumulate+gather passes, but never allt he pases for a single recipe only. many whole pipelines that
  look like 20-30 SETS of passes may be able to compress into a few if we dot he righ tthings in the righ torder
  to the right columns. that's part of the shareability and decomposition anchors for tambear.  plus, we may also
   be able to have primitives that SHARE an accumulator but use a different gather step based on parameters,
  changing whre the system will push the computations in to different accumulate+gather steps. the system will
  optimize that during IR/compile based on the entire pipeline all at once.