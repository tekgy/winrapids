# Adversarial attacks on Sweep 27 Pre-Check Library

**Author:** adversarial · **Date:** 2026-04-22
**Target:** Sweep 27 planned pre-check library (27B — Plain-Rust pre-check library)

These are SILENT failure modes — cases where a pre-check returns
confident evidence that is factually wrong. Wrong evidence is worse
than no evidence: it infects Shape.tags, `has_known_non_finite`,
codegen specialization, using()-annotations, and auto-fix insertion
with false certainty.

Aristotle's Q-rec-2 (phase-1-8-tam-cannot-tell-the-future.md) named
the invariant: "profile says YES only after evidence." These attacks
probe whether each pre-check can actually satisfy that invariant.

---

## ATTACK 27-1 — scan_nonfinite: sampling misses tail NaN

**Verdict: Critical gap if sampling is used.**

### The break

Sweep 27 README says DataProfile is computed at "attach time" and
mentions "small-sample probes" (first 1000 rows, or a random sample).
If `scan_nonfinite` scans a SAMPLE rather than the full column, it
can return `has_known_non_finite = false` when NaN exists in row 1001.

Specific adversarial dataset: a financial time series where the LAST
row (the most recent tick) has NaN because the closing auction data
hasn't arrived yet. The profiler scans the first 1000 rows and
declares clean. Codegen elides the validity branch. The last row
propagates NaN silently.

### Why this is the worst case

A profile claim of `has_known_non_finite = false` causes the
JIT codegen to EMIT A KERNEL WITHOUT THE NaN GUARD BRANCH. When that
kernel encounters the NaN it was promised doesn't exist, the output
is wrong in an unpredictable way (depends on how the arithmetic
propagates). No error. No warning. The `has_known_non_finite = false`
tag in Shape was a lie.

### Required discipline

`scan_nonfinite` MUST scan the full column to make a definite false
claim. The ONLY path to `has_known_non_finite = false` with
safety is exhaustive scan. Partial scan can only produce
`has_known_non_finite = unknown` (conservative).

The README says "cached by dataset hash" — but the hash must be over
the FULL dataset, not over the sample. If the profiler hashes only
the sample and stores `false`, a subsequent dispatch on the full
dataset (same hash? different?) hits the gap.

**Fix required in 27A/27B:** `scan_nonfinite` behavior must be:
- `scan_nonfinite(full_column)` → definite result (safe to emit `known_false`)
- `scan_nonfinite_sample(col, n)` → returns `Unknown` (never sets `known_false`)

The Shape's `has_known_non_finite` must only be set to `false` by the
full-scan path. Sample path at most sets it to `true` (if NaN found)
or leaves it at `Unknown`.

---

## ATTACK 27-2 — Jarque-Bera on samples: false normality for bimodal

**Verdict: Structural flaw. JB test on samples fails for bimodal distributions.**

### The break

Jarque-Bera tests whether skewness and kurtosis match a Gaussian.
For a symmetric bimodal distribution (equal-weight mixture of two
Gaussians, means ±μ, same variance):

- Skewness = 0 (symmetric → S=0)
- Kurtosis depends on μ: for small μ (peaks close together),
  excess kurtosis → 0; the mixture approaches a single Gaussian.

JB statistic = n/6 * (S² + (E-K)²/4) ≈ 0 when both are small.

**JB accepts H0 (Gaussian) for a symmetric bimodal distribution.**

A symmetric bimodal return distribution (common in assets with
regime-switching: bull/bear, open/close gaps) will pass JB. TAM
then declares "distribution: Gaussian" in the DataProfile and
recommends Pearson correlation at step 12 (the README example).
The correct recommendation for heavy-tailed or multimodal data is
Spearman or Kendall.

### The silent wrong recommendation chain

1. JB passes → `DistributionShape = Gaussian`
2. `Gaussian` tag → pipeline-aware check for step 12 (correlation) picks Pearson
3. Pearson on bimodal data is biased — the regression-to-the-mean
   structure differs between modes; Pearson aggregates them wrong
4. Using-annotation says "Pearson recommended, bivariate normality
   confirmed" — this is a false citation of evidence

### Required discipline

27B must document which distribution families JB is BLIND to:
- Symmetric bimodal (skewness = 0, kurtosis can be low)
- Symmetric heavy-tailed (skewness = 0, but kurtosis > 3 WILL fire JB correctly)

The fix: supplement JB with Anderson-Darling (sensitive to both tails
and body) AND a multimodality test (Silverman's bandwidth test, or
diptest for unimodality). JB alone is insufficient for the
"distribution is Gaussian" claim in a using()-annotation.

**Or:** weaken the claim — never emit "bivariate normality confirmed,"
only emit "JB did not reject normality (p>α)" with the explicit caveat
that JB is insensitive to symmetric bimodal alternatives. The
evidence-grounded prose in the README is then honest: "JB test
(p=0.42) did not reject normality; note: JB is blind to symmetric
bimodal alternatives — consider Anderson-Darling for stronger evidence."

---

## ATTACK 27-3 — condition_number_estimate on subsamples: wrong estimate for structured matrices

**Verdict: Gap. Subsampling a matrix destroys its structural conditioning.**

### The break

The README mentions "condition number estimate" via a small-sample
probe ("first 1000 rows"). For an n×p matrix where n >> 1000 and
p is moderate, the full matrix may be near-singular because a few
DISTANT rows are nearly collinear with each other — but those rows
don't appear in the first 1000.

Specific adversarial case: a p=10 feature matrix where:
- Rows 1-1000: well-conditioned (condition number ~ 10)
- Rows 9000-10000: one additional "cluster" that is nearly
  collinear with an earlier cluster (adds the near-singular subspace)

Condition number estimate from the first 1000 rows: ~10 (fine).
True condition number of the full matrix: ~1e15 (near-singular).

TAM declares "condition number normal, proceed with default solver"
in the using()-annotation. The full dispatch hits the near-singular
matrix. Default solver (LU factorization) produces numerically garbage
output. No error thrown (LU succeeds; it just produces wrong output
when condition >> 1e12).

### Required discipline

The condition_number_estimate pre-check for Step 14 (matrix inversion)
in the README example CANNOT be run on a subsample. The condition
number is a global property of the full matrix.

For very large matrices (n > memory capacity), the correct approach
is either:
- Randomized SVD on a larger sample (not "first 1000 rows" but a
  random sketch that preserves the global singular value structure)
- Report `condition_unknown` (conservative) and decline to advise

Running on first 1000 rows and reporting a condition number for the
full matrix is actively dangerous — it will produce false confidence
for exactly the adversarial case (structured near-singularity where
the near-collinear directions only emerge across the full dataset).

**Fix required:** condition_number_estimate must either:
1. Run on the FULL matrix (expensive but correct)
2. Use a randomized sketch that samples uniformly across ALL rows
   (not first-N rows) and documents that it is an estimate with
   explicit error bounds
3. Report `Unknown` for matrices too large to estimate safely

The "small-sample probe" design (27D) cannot be used for condition
number estimation.

---

## ATTACK 27-4 — infer_dtype on "numeric strings": locale-dependent parsing

**Verdict: Platform-sensitive silent failure.**

### The break

The README example:
> "Column Y is a string column whose values look numeric ('1234', '5678', ...)."

`infer_dtype` must parse strings to determine if they are numeric.
The adversarial case: strings using non-ASCII decimal separators
common in non-US locales:

- German format: "1.234,56" (1234.56 in European notation: period=thousands, comma=decimal)
- French format: "1 234,56" (space=thousands, comma=decimal)

These strings LOOK non-numeric to a US-locale parser (Rust's
`str::parse::<f64>()` will fail on "1.234,56"). The dtype inferrer
declares `String` type and does NOT auto-insert the cast step.

The actual data is numeric. The user receives no auto-fix. The
pipeline proceeds with string semantics. The mistake is invisible
because the inferrer correctly said "this is not parseable as f64"
— but it failed to detect the locale variant.

Worse case: the user's test dataset (rows 1-100) uses US notation;
their production dataset (rows 1-10000) uses European notation for
25% of rows because it came from a European feed. The inferrer on
the test dataset declares numeric; on production declares string.
Different behavior on different data shapes. Profile diverges from
reality.

### Required discipline

`infer_dtype` must be documented as locale-sensitive. The behavior
for non-US numeric strings must be explicit: either:
1. Declare attempt to parse ALL common locale formats (complex, fragile)
2. Declare US-locale only and document that non-US numeric strings
   are returned as `InferredDtype::String` (not auto-cast)
3. Return `InferredDtype::AmbiguousNumericString` when the format
   is non-standard, triggering a using()-annotation that asks the
   user to specify the locale

The warning in case (2) must NOT say "auto-inserted a cast step" —
it must say "string values that did not parse as f64 in US locale."
The README's example wording ("'1234', '5678'") avoids this issue but
27B must not accidentally create a locale-dependent parser that
silently misclassifies European numeric strings as non-numeric.

---

## ATTACK 27-5 — sparsity_fraction: zero vs missing value conflation

**Verdict: Semantic ambiguity that produces wrong method recommendations.**

### The break

`sparsity_fraction` returns the fraction of zero values. The README:
> "Column X is 95% zeros. Step N's default method assumes dense input."

The adversarial case: the user's data has 95% MISSING VALUES (NaN)
that were filled with 0 by a prior imputation step. The data is not
sparse in the "structural zeros" sense — it is imputed.

`sparsity_fraction` sees 95% zeros and recommends the sparse variant
of Step N. The sparse variant (e.g., sparse matrix-vector product)
treats the zeros as structural and compresses them. The imputed zeros
are NOT structural — they carry uncertainty. A correlation computed
with 95% of values imputed as 0 is biased toward 0 (the imputed
values dominate).

The using()-annotation says "sparse variant: expected 8x speedup"
but the CORRECT annotation should be "95% of values were imputed as
0; this data may need to be handled with missingness-aware methods
before sparsity optimization."

### Required discipline

27B must distinguish between:
- `structural_zeros`: zeros that were explicitly encoded as zero
  in the original data (genuinely sparse)
- `imputed_zeros`: zeros that came from NaN-imputation
- `unknown_zeros`: zeros with no provenance information

The `scan_nonfinite` check (27-1) should run FIRST. If NaN count
was > 0 and the user's pipeline included any imputation step upstream,
`sparsity_fraction` should annotate its result as "potentially
imputed zeros" and decline to recommend the sparse variant without
user confirmation.

Without this distinction, `sparsity_fraction` + auto-fix to sparse
variant is a correctness hazard for imputed data.

---

## ATTACK 27-6 — sample_stats skewness/kurtosis: catastrophic on heavy tails (Cauchy)

**Verdict: The sample estimates are meaningless for Cauchy-like distributions.**

### The break

`sample_stats` computes skewness and kurtosis on a sample. For
Cauchy-distributed data (heavy-tailed; undefined mean, undefined
variance), the sample mean and sample variance are NOT converging
estimators — they diverge as n increases.

On a sample of 1000 rows from a Cauchy distribution:
- Sample mean: some value (depends on which extreme values were sampled)
- Sample variance: some large value (but NOT the true "variance" — undefined)
- Sample kurtosis: may be enormous (dominated by the most extreme sampled values)

JB test on these: JB statistic → ∞ (kurtosis term dominates) → correctly REJECTS normality.

But: what if the sample happened to miss the extreme tails? Cauchy
has heavy enough tails that even n=1000 samples routinely miss the
most extreme quantiles. In that case, the sample kurtosis is moderate
and JB might NOT reject. The using()-annotation says "distribution
consistent with normality" for Cauchy data.

This is a double failure:
1. `sample_stats` computes kurtosis from a sample that has no
   well-defined population kurtosis → the estimate is noise
2. JB test based on that estimate may fail to reject normality
3. Downstream method selection picks Pearson (normality assumed)
4. Pearson on Cauchy data: the sample correlation is NOT a
   consistent estimator of anything meaningful (the joint moments
   don't exist)

### Required discipline

27B must include a "Cauchy detection" step that precedes JB:
- Run a tail-index estimator (Hill estimator on the top-k order
  statistics, or the Pickands-Balkema-de Haan estimator)
- If tail index < 2 (infinite variance): declare `DistributionShape::InfiniteVariance`
  and SKIP JB (it is not applicable)
- Emit warning: "Column X may have infinite variance (tail index
  estimate: α ≈ 1.2 < 2). JB test is not applicable. Pearson
  correlation, OLS, and other second-moment-based methods are
  inappropriate for this data."

Hill estimator is cheap (O(n log n) sort + O(k) estimation) and
robust. It should be a mandatory pre-step before JB in 27B.

---

## ATTACK 27-7 — cardinality check: hash collision in unique-count estimation

**Verdict: Likely safe if exact, but risky if approximate (HyperLogLog).**

### The break

If `cardinality` uses HyperLogLog or similar approximate
unique-count estimator (natural choice for large datasets), it has
a built-in relative error of ~1-2%.

Adversarial case: Step 20 requests k=10 clusters on n=12 points
(the README's infeasibility example). The exact count is 12.
HyperLogLog with 1% relative error could return 12 ± 0.12 → any
value in {11, 12, 13} with high probability.

If HLL returns 11, TAM says "k=10 clusters on n=11: marginal,
proceed with warning." The kernel then receives n=12 → fails mid-
dispatch (or produces degenerate clusters with empty groups).

More dangerous: k=10 on n=100 where 88 of the 100 points are
duplicates. True unique count = 12. HLL estimates ~12 (accurate).
TAM correctly warns. But: Step 20's "infeasibility" check uses n
(total rows) not n_unique. If it uses n_unique and HLL is accurate,
fine. If it uses n (100) → "k=10 on n=100: feasible" → but the
algorithm receives effectively 12 unique points.

### Required discipline

The k-vs-n feasibility check (27C) must distinguish:
- n_total: total rows
- n_unique: unique points (relevant for clustering feasibility)

Both must be available. The feasibility check uses `min(n_total, n_unique)`.
If `cardinality` is approximate, it must document the error bound
and the feasibility check must add margin: "k <= n_unique_estimate - safety_margin."

For exact feasibility (the README example is trying to catch obvious
infeasibility), exact counting is preferable. HyperLogLog is a
last resort for very large n_unique.

---

## ATTACK 27-8 — pipeline-step-aware checks: check runs on wrong column after pipeline reorder

**Verdict: State-coupling gap. Pre-check result may become stale after pipeline edit.**

### The break

The README says checks run "as the user adds a step" and results are
cached by `(pipeline_ir_hash, dataset_hash)`.

User flow:
1. User attaches dataset.
2. User adds Step 12 (correlation on columns 0 and 1). Pipeline-aware
   check runs: JB on columns 0 and 1 → normality confirmed → Pearson recommended.
3. User REORDERS the pipeline: inserts a new Step 11.5 that TRANSFORMS
   column 0 (e.g., log-transform). The pipeline_ir_hash changes.
4. The log-transform changes column 0's distribution. The old JB
   check result (on the original column 0) is now STALE.
5. Pipeline cache invalidates (new ir_hash). New check runs on the
   TRANSFORMED column 0 → JB on log(X) for log-normal X → correct result.

This case is fine IF the cache invalidates correctly on pipeline_ir_hash change.

**The silent failure:** what if the user adds Step 12 FIRST (on
an intermediate representation) and the check runs against the
inferred shape of the intermediate — but the inference of that shape
assumed some default behavior that the user later overrides via `using()`?

Example: Step 11 outputs a "normalized" column under the default
`using(normalize=true)`. JB checks the normalized column → normal
distribution confirmed. User changes to `using(normalize=false)`.
Pipeline_ir_hash changes IF the `using()` override flows into the IR
hash. But if `using()` overrides are stored separately from the IR
and the IR hash only covers the topology, the cache key may not
include this particular using() change.

### Required discipline

The cache key `(pipeline_ir_hash, dataset_hash)` must include ALL
`using()` annotations that affect the shape of intermediate outputs.
If `using(normalize=false)` changes the distribution of step 11's
output, it must change the IR hash (or be explicitly part of the
pre-check cache key).

This means the pre-check cache key is effectively
`(full_pipeline_hash_including_using_annotations, dataset_hash)`.
Partial-IR hashing that ignores using() overrides will produce stale
pre-check results that look valid (cache hit) but apply to a different
pipeline configuration.

---

## Convergence: what these 8 attacks share

All 8 attacks trace to one of three root causes:

1. **Sampling that claims full-data coverage** (Attacks 1, 3):
   `has_known_non_finite = false` from a partial scan, or
   condition number from first-N rows. The invariant "say YES only
   after exhaustive evidence" is violated by partial scans.

2. **Statistical test assumptions violated by the data being tested**
   (Attacks 2, 6): JB is blind to symmetric bimodal distributions;
   sample kurtosis is meaningless for Cauchy. The pre-check must
   check its own preconditions before running.

3. **Semantic ambiguity in the feature being measured** (Attacks 4,
   5, 7, 8): zeros (sparse vs imputed), numeric strings (locale
   variants), unique counts (exact vs approximate), cache keys
   (topology-only vs including using()).

The common fix pattern: **every pre-check must document what it
CANNOT safely claim** alongside what it can. The DataProfile
struct (27A) must carry confidence tags alongside values:
`NonFinite::KnownAbsent` vs `NonFinite::Unknown`. Never
collapse `Unknown` to `KnownAbsent` based on partial evidence.

---

## Severity table

| Attack | Pre-check | Silent failure mode | Severity |
|--------|-----------|---------------------|----------|
| 27-1 | scan_nonfinite | Sample misses tail NaN → false `known_false` → wrong kernel emitted | **Critical** |
| 27-2 | jarque_bera | Symmetric bimodal passes JB → false normality claim → wrong method pick | **High** |
| 27-3 | condition_number | First-N rows miss structured near-singularity → false safe claim → LU garbage | **High** |
| 27-4 | infer_dtype | Non-US locale numeric strings → false non-numeric claim → missing auto-cast | **Medium** |
| 27-5 | sparsity_fraction | Imputed zeros classified as structural → sparse method on imputed data → biased result | **High** |
| 27-6 | sample_stats | Cauchy tail missed in sample → kurtosis noise → JB falsely accepts normality | **High** |
| 27-7 | cardinality | Approximate count errors → wrong feasibility decision | **Medium** |
| 27-8 | pipeline cache | using() changes not in IR hash → stale pre-check for changed pipeline config | **Medium** |

**One critical. Four high. Three medium. Zero ignored.**

The critical finding (27-1) is the direct path from "profile says
`has_known_non_finite = false`" to "kernel without NaN guard runs
on data with NaN." This closes Q-rec-2 from aristotle's
phase-1-8-tam-cannot-tell-the-future.md: the invariant CAN be
violated by partial scanning, and the fix is exhaustive scanning
or conservative `Unknown` reporting from sample-based checks.
