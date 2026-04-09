# Workup: `nonparametric::kendall_tau`

**Family**: nonparametric (rank-based correlation)
**Status**: draft — numerical verification complete, benchmarks pending
**Author**: scientist
**Last updated**: 2026-04-09
**Module**: `crates/tambear/src/nonparametric.rs`
**Function signature**: `pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64`

---

## 1. Mathematical definition

### 1.1 The quantity computed

Kendall's rank correlation coefficient τ-b (tau-b), a measure of ordinal
association between two variables that handles tied ranks. Range [−1, 1].

### 1.2 Canonical definition

Given paired observations (xᵢ, yᵢ) for i = 1,…,n, each unordered pair of
indices {i, j} (with i < j) is classified by the sign of (xᵢ − xⱼ) and
(yᵢ − yⱼ):

- **Concordant (C)**: both differences have the same (nonzero) sign.
- **Discordant (D)**: the differences have opposite nonzero signs.
- **Tied in x only (Tₓ)**: xᵢ = xⱼ but yᵢ ≠ yⱼ.
- **Tied in y only (Tᵧ)**: yᵢ = yⱼ but xᵢ ≠ xⱼ.
- **Joint tie**: xᵢ = xⱼ and yᵢ = yⱼ. Joint ties are excluded from all
  counts, per Kendall (1945).

The tau-b statistic is

```
        C − D
τ_b = ────────────────────
      √(C+D+Tₓ)(C+D+Tᵧ)
```

When there are no ties, Tₓ = Tᵧ = 0 and τ_b reduces to tau-a:
τ_a = (C − D) / (n(n−1)/2).

### 1.3 Assumptions

- **Required**: inputs `x` and `y` are finite-length slices of `f64` values
  with equal length. Joint ties (xᵢ = xⱼ AND yᵢ = yⱼ) are handled.
- **Not assumed**: marginal distributions, normality, monotonic relation,
  sortedness.
- **Runtime-checked**: length mismatch panics (debug-assert via
  `assert_eq!`). `n < 2` returns NaN. `denom == 0` (all ties in one
  dimension) returns NaN.
- **Not checked at runtime**: NaN values in `x` or `y`. A NaN propagates
  through the pairwise comparisons; the result is undefined. Callers
  should filter NaNs before invocation.

### 1.4 Kingdom declaration

**Kingdom A** — commutative. The pairwise comparison is symmetric in `i`
and `j`, and the global counts (C, D, Tₓ, Tᵧ) are commutative sums. Every
pair contributes independently to the counters. Trivially parallelizable
across the n(n−1)/2 pairs.

### 1.5 Accumulate+gather decomposition

```
input:  x, y : [f64; n]
output: τ_b : f64

step 1: pairwise indexing
    pairs = { (i, j) : 0 ≤ i < j < n }                 (Kingdom A)

step 2: per-pair classification
    class(i, j) = match (sign(xᵢ − xⱼ), sign(yᵢ − yⱼ))
        (nonzero, nonzero) if same sign   → concordant
        (nonzero, nonzero) if opposite    → discordant
        (zero, nonzero)                   → tied_x
        (nonzero, zero)                   → tied_y
        (zero, zero)                      → joint_tie (excluded)

step 3: accumulate into 4 counters
    (C, D, Tₓ, Tᵧ) = accumulate(pairs, All, class, SumByLabel)

step 4: scalar gather
    τ_b = (C − D) / sqrt((C + D + Tₓ) · (C + D + Tᵧ))
```

The current implementation is a naive O(n²) double loop. An O(n log n)
merge-sort variant (Knight 1966) is possible and is noted as a future
optimization in Section 10.

---

## 2. References

- [1] Kendall, M. G. (1938). *A New Measure of Rank Correlation*.
  Biometrika 30(1/2):81–93.
- [2] Kendall, M. G. (1945). *The Treatment of Ties in Ranking Problems*.
  Biometrika 33(3):239–251. Defines τ-b with the tie-split denominator.
- [3] Knight, W. R. (1966). *A Computer Method for Calculating Kendall's
  Tau with Ungrouped Data*. JASA 61(314):436–439. The O(n log n)
  merge-sort algorithm.
- [4] Kendall, M. G. & Gibbons, J. D. (1990). *Rank Correlation Methods*
  (5th ed.). Oxford University Press. Chapter 3.

---

## 3. Implementation notes

### 3.1 Algorithm chosen

Naive pairwise enumeration. O(n²) time, O(1) extra space, direct
translation of Definition 1.2. Chosen for auditability and bit-perfect
match with the definition; the merge-sort optimization is deferred
until n > 10⁴ benchmarks demand it.

### 3.2 Numerical stability

The only arithmetic is `dx * dy` for sign detection and a final
`sqrt((C+D+Tₓ)(C+D+Tᵧ))`. Overflow in `dx * dy`: extreme inputs where
`|dx * dy| > f64::MAX / 2` would lose sign, but since we only read the
sign, we could replace with `dx.signum() * dy.signum()`. Currently
unmitigated — see Section 10.

Counters are `i64`, which supports n ≤ 2⁽³¹˙⁵⁾ ≈ 3 billion before
n(n−1)/2 overflows. Realistic use cases are far below this.

### 3.3 Parameters

| Parameter | Type | Valid range | Default | Reference |
|-----------|------|-------------|---------|-----------|
| `x` | `&[f64]` | any finite | (none) | Definition 1.2 |
| `y` | `&[f64]` | any finite | (none) | Definition 1.2 |

No tuning parameters. The function is pure τ-b; variants like τ-a, τ-c,
Goodman-Kruskal γ are separate primitives (some not yet implemented).

### 3.4 Input validation

- `n = 0` or `n = 1` → `NaN` (insufficient data).
- `x.len() != y.len()` → panics via `assert_eq!`. This is caller contract
  violation, not degenerate data; a panic is the correct response.
- All-constant `x` or all-constant `y` → `NaN` (`denom == 0`).
- NaN in input → undefined. **See Section 10 for mitigation plan.**

### 3.5 Shareable intermediates

None currently. A future merge-sort implementation would produce the
sorted permutation of `x` (and the induced permutation of `y`) as
intermediates usable by `spearman` and any rank-based method. Tag would
be `IntermediateTag::SortedPermutation { data_id }`.

The Tₓ and Tᵧ counts are themselves potentially useful to downstream
users (they equal the "correction for ties" terms used in variance
estimators of τ under H₀). A `KendallCounters` struct returning
`(C, D, Tx, Ty)` alongside τ is a candidate enhancement.

---

## 4. Unit tests

Inline tests in `crates/tambear/src/nonparametric.rs` and the workup
parity suite `crates/tambear/tests/workup_kendall_tau.rs`.

Checklist:

- [x] Known-answer: monotone increasing → τ = 1
- [x] Known-answer: monotone decreasing → τ = −1
- [x] Empty input → NaN
- [x] n=1 → NaN
- [x] Constant x (degenerate denom) → NaN
- [x] NaN handling — behavior documented as undefined, test asserts no
  panic on filtered inputs
- [ ] ±∞ handling — not yet tested
- [ ] Subnormal handling — not yet tested
- [x] Shift invariance in x
- [x] Scale invariance in x (positive scale)
- [x] Sign flip under negation of x
- [x] Symmetry τ(x, y) = τ(y, x)
- [x] Range bounds [−1, 1] across 50 random cases

---

## 5. Oracle tests — against extended precision

Oracle script (inline, see Section 6 below for the mpmath implementation):
`docs/research/workups/nonparametric/kendall_tau-oracle.py` (one-off).

### 5.1 Method

A first-principles Kendall τ-b implementation using `mpmath.mpf` at 50
decimal digits. All four counters (C, D, Tₓ, Tᵧ) are accumulated exactly
in integer arithmetic; only the final `sqrt(…)` divides at 50-digit
precision. The result is then compared against scipy and against tambear.

### 5.2 Test cases

| Case | n | Description | mpmath truth | tambear | rel err |
|------|---|-------------|--------------|---------|---------|
| 1 | 5 | monotone increasing | +1.000000000000000 | +1.000000000000000 | 0 |
| 2 | 5 | monotone decreasing | −1.000000000000000 | −1.000000000000000 | 0 |
| 3 | 5 | light ties | +0.9486832980505138 | +0.9486832980505138 | 0 |
| 4 | 6 | heavy ties | +0.41666666666666663 | +0.41666666666666663 | 0 |
| 5 | 20 | numpy.randn seed 42 | −0.042105263157894736 | −0.042105263157894736 | 0 |
| 6 | 4 | constant x (degenerate) | NaN | NaN | — |

### 5.3 Maximum observed relative error

**0** (bit-perfect with mpmath across all tested cases). Since the
implementation accumulates integer counts and only divides once, the only
sources of error are the final `sqrt` and the `(C−D)/denom` division,
both of which are correctly rounded in IEEE 754.

---

## 6. Cross-library comparison

### 6.1 Competitors tested

| Library | Version | Function | Agrees with mpmath? | Agrees with tambear? |
|---------|---------|----------|---------------------|----------------------|
| scipy | 1.17.1 | `scipy.stats.kendalltau` | **yes** (err = 0 across cases 1–6) | **yes** (err = 0) |
| mpmath | 1.3.0 | first-principles (reference) | ground truth | **yes** |

### 6.2 Discrepancies found

None. All cases agree bit-perfectly with both scipy and mpmath.

### 6.3 scipy history note

scipy's `kendalltau` has had several historical changes in how it reports
p-values and how it handles the `variant` argument (`'b'` vs `'c'`).
These affect p-value computation and the alternative τ-c variant, not the
τ-b point estimate itself. Our `kendall_tau` computes only τ-b and does
not return a p-value, so we are not affected by those scipy version
differences.

If we add a p-value computation in the future, we will need to re-check
against scipy 1.17.x (exact permutation) vs older versions
(asymptotic normal approximation). The defaults differ between versions.

### 6.4 Verdict

**Bit-perfect τ-b** against scipy 1.17.1 across all tested inputs.
Matches the mpmath first-principles computation at 50 dp to machine
precision (rel err = 0).

Libraries not yet tested: R `cor(..., method="kendall")`, statsmodels,
MATLAB `corr(…,'Type','Kendall')`, Julia `StatsBase.corkendall`. These
are queued for future expansion of Section 6.1 but are not blocking for
workup sign-off as long as at least two independent implementations
(scipy + mpmath) agree.

---

## 7. Adversarial inputs

Status: partial.

- [x] Small-n edge cases (n=0, n=1, n=2) → return sentinel or compute
  correctly
- [x] Constant-input degeneracy → NaN
- [x] Heavy-tie inputs (>50% ties) → reasonable output, no panic
- [ ] Catastrophic-cancellation input — not applicable (no subtraction of
  nearly-equal floats; only sign comparisons)
- [ ] Large n (10⁶) memory / time — not yet benchmarked. At 10⁶ the O(n²)
  algorithm needs 10¹² operations, ~1000 seconds at 1 Gop/s. **See
  Section 10.**
- [ ] NaN injection — currently undefined behavior. **See Section 10.**
- [ ] ±∞ injection — currently undefined behavior.
- [ ] Subnormal injection — currently undefined behavior.
- [ ] Overflow in `dx * dy` — currently unmitigated.

---

## 8. Invariants and proofs

Enforced in `crates/tambear/tests/workup_kendall_tau.rs`:

1. **Symmetry**: τ(x, y) = τ(y, x). ✓
2. **Shift invariance in x**: τ(x + c, y) = τ(x, y) for any c. ✓
3. **Scale invariance in x under positive scaling**: τ(αx, y) = τ(x, y)
   for α > 0. ✓
4. **Sign flip under negation**: τ(−x, y) = −τ(x, y). ✓
5. **Range bound**: τ ∈ [−1, 1] for all inputs. ✓ (checked over 50
   random cases with Xoshiro256 seed 12345)

Not yet enforced as tests:

- **Triangle-ish inequality**: no clean one exists for τ; skip.
- **Monotone transformation invariance**: τ(f(x), g(y)) = τ(x, y) for
  monotone increasing f, g. This is the defining property of rank
  correlations; we test it as case-level for sort-preserving transforms.

---

## 9. Benchmarks and scale ladder

Status: **not yet run**. The current O(n²) implementation will not scale
past ~10⁴ in reasonable time. The scale ladder is:

| n | Est. CPU time | Feasible? |
|---|---|---|
| 10¹ | < 1 µs | yes |
| 10² | ~10 µs | yes |
| 10³ | ~1 ms | yes |
| 10⁴ | ~100 ms | yes |
| 10⁵ | ~10 s | yes, but slow |
| 10⁶ | ~1000 s | infeasible |
| 10⁸ | months | infeasible |

Action: before signing this workup off, implement Knight's O(n log n)
merge-sort algorithm. Until then, the primitive is capped at n ≈ 10⁵ for
interactive use.

---

## 10. Known bugs / limitations / open questions

- **TASK: NaN handling is undefined.** An NaN in `x` or `y` propagates
  through `dx * dy` and the sign classification silently rounds to
  "neither concordant nor discordant", so counts are under-reported
  instead of the result being NaN. Mitigation: either filter NaN pairs
  at entry, or detect any NaN and return NaN. Decision: return NaN when
  any pair contains NaN, matching scipy's `nan_policy='propagate'`
  default. **Severity: medium, spec bug.**

- **TASK: Overflow in `dx * dy` sign check.** For `|dx|, |dy|` near
  `f64::MAX / 2`, the product may overflow to infinity, which preserves
  the sign, or to NaN if one side is ±∞. Replace `dx * dy > 0` with
  `dx.signum() == dy.signum() && dx != 0.0 && dy != 0.0`. **Severity:
  low, only affects pathological inputs.**

- **TASK #XX (to be created): Implement Knight's O(n log n) algorithm.**
  The current O(n²) naive loop limits the primitive to n ≈ 10⁵ for
  interactive use. Knight (1966) gives an O(n log n) algorithm via two
  merge-sorts that counts inversions exactly. This would move tambear
  to parity with scipy's performance (scipy uses the same algorithm)
  and enable the full scale ladder in Section 9.

- **Open question**: should we return `KendallResult { tau, c, d, tx,
  ty, joint_ties }` rather than a bare f64? The counters are useful for
  variance estimation and p-value computation. Current answer: expose
  the bare value for simplicity; add a `kendall_tau_full` variant if/when
  downstream methods need the counters.

- **Open question**: τ-a, τ-c (Stuart-Kendall), and Goodman-Kruskal γ are
  related measures. Should they share a sorting step and co-compute? Yes,
  but that's a refactor once the merge-sort implementation lands.

---

## 11. Sign-off

- [x] Section 1-3 written by scientist
- [x] Unit tests in Section 4 passing (12/12 in workup_kendall_tau.rs)
- [x] Oracle tests in Section 5 at target precision (bit-perfect)
- [x] Cross-library comparison in Section 6 complete for scipy + mpmath
- [ ] Adversarial suite in Section 7 passing (NaN, ±∞, subnormal missing)
- [ ] Benchmarks in Section 9 complete through n = 10⁶ minimum (blocked
  on Knight's O(n log n) algorithm)
- [x] Known bugs in Section 10 documented with severity and mitigation
- [ ] Reviewed by scientist (this doc)
- [ ] Reviewed by adversarial
- [ ] Reviewed by math-researcher

**Overall status**: draft. Ready for math-researcher review of Sections 1–3,
adversarial review of Section 7 (to hit the missing NaN/∞/subnormal cases),
and a naturalist decision on whether the O(n log n) rewrite should happen
now or be deferred behind an open task.

---

## Appendix A: reproduction artifacts

- Oracle script: inline Python in the bash transcript of this session;
  to be extracted to `research/gold_standard/kendall_tau_oracle.py` on
  next workup sweep.
- Parity test file: `crates/tambear/tests/workup_kendall_tau.rs`
- Benchmark harness: not yet written (blocked on O(n log n) rewrite).

## Appendix B: version history

| Date | Author | Change |
|------|--------|--------|
| 2026-04-09 | scientist | Initial draft; bit-perfect against mpmath + scipy; open items for NaN handling, overflow, O(n log n) rewrite. |
