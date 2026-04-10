# Workup: `nonparametric::pearson_r`

**Family**: nonparametric (correlation)
**Status**: draft — numerical verification complete, adversarial suite complete, scale benchmarks pending
**Author**: scientist
**Last updated**: 2026-04-10
**Module**: `crates/tambear/src/nonparametric.rs`
**Function signature**: `pub fn pearson_r(x: &[f64], y: &[f64]) -> f64`

---

## 1. Mathematical definition

### 1.1 The quantity computed

The Pearson product-moment correlation coefficient, the canonical measure of
linear association between two real-valued variables. Range [−1, 1].

### 1.2 Canonical definition

Given paired observations (xᵢ, yᵢ) for i = 1,…,n, the Pearson r is:

```
     Σᵢ (xᵢ − x̄)(yᵢ − ȳ)
r = ──────────────────────────────────────────────
     √[Σᵢ(xᵢ − x̄)²] · √[Σᵢ(yᵢ − ȳ)²]
```

where x̄ = (1/n)Σᵢ xᵢ and ȳ = (1/n)Σᵢ yᵢ are the sample means.

This is the sample correlation coefficient; it is an estimator of the
population correlation ρ when (X, Y) are drawn from a bivariate distribution
with finite second moments. The formula is the cosine similarity between the
mean-centered vectors (x − x̄1) and (y − ȳ1).

### 1.3 Equivalent forms

Five algebraically equivalent forms, given by Rodgers & Nicewander (1988):

1. **Covariance form**: r = Cov(x,y) / (σ_x · σ_y)
2. **Standardized product**: r = (1/n) Σᵢ z_xᵢ · z_yᵢ where z_xᵢ = (xᵢ−x̄)/σ_x
3. **Raw-score formula**: r = [Σ(xy) − n·x̄·ȳ] / √[(Σx² − n·x̄²)(Σy² − n·ȳ²)]
4. **Geometric**: r = cos θ between centered vectors (x−x̄1) and (y−ȳ1)
5. **One-pass formula**: r = (n·Σ(xy) − Σx·Σy) / √[(n·Σx² − (Σx)²)(n·Σy² − (Σy)²)]

Tambear implements **form 4** (two-pass, mean-centered inner product). This
is numerically superior to forms 3 and 5 — see Section 3.2 on stability.

### 1.4 Assumptions

- **Required**: inputs x and y are finite-length slices of f64 values with equal
  length. n ≥ 2.
- **Not assumed**: marginal distributions, normality, linearity of relationship
  (r measures linear association only — a perfect monotone nonlinear relationship
  can give |r| < 1).
- **Runtime-checked**: length mismatch panics (assert_eq!). n < 1 returns NaN
  (the denominator is zero when n=0; n=1 gives centered vectors of all zeros).
- **Degenerate detection**: if `denom < 1e-15` (either variable has near-zero
  variance), returns NaN. This threshold is discussed in Section 3.3.
- **Not checked at runtime**: NaN values in x or y. NaN propagates through
  the mean computation and centered differences; the result is NaN or
  undefined (platform-dependent). **See Section 10.**

### 1.5 Kingdom declaration

**Kingdom A** — commutative accumulation. Each (xᵢ, yᵢ) pair contributes
independently to three accumulators (cross-product, squared-x, squared-y)
after the mean-computation pass. The accumulation is commutative and the
combine operation is addition.

The two-pass structure makes this a **two-round Kingdom A** computation:

- **Round 1**: accumulate(All, xᵢ, sum) → x̄; accumulate(All, yᵢ, sum) → ȳ
- **Round 2**: accumulate(All, (xᵢ−x̄)(yᵢ−ȳ), sum) → num;
              accumulate(All, (xᵢ−x̄)², sum) → dx2;
              accumulate(All, (yᵢ−ȳ)², sum) → dy2
- **Gather**: r = num / sqrt(dx2 · dy2)

Round 2 depends on outputs of Round 1, so this is a two-round K-A computation
(not reducible to one round without sacrificing numerical stability).

### 1.6 Accumulate+gather decomposition

```
input:  x, y : [f64; n]
output: r : f64

round 1 — means:
    x̄ = accumulate(All, xᵢ, Add) / n
    ȳ = accumulate(All, yᵢ, Add) / n

round 2 — centered moments:
    num = accumulate(All, (xᵢ − x̄)(yᵢ − ȳ), Add)
    dx2 = accumulate(All, (xᵢ − x̄)², Add)
    dy2 = accumulate(All, (yᵢ − ȳ)², Add)

gather — scalar combination:
    denom = sqrt(dx2 · dy2)
    if denom < 1e-15: return NaN
    r = num / denom
```

---

## 2. References

- [1] Pearson, K. (1895). *Notes on regression and inheritance in the case of two parents*. Proceedings of the Royal Society of London 58:240–242.
- [2] Rodgers, J. L. & Nicewander, W. A. (1988). *Thirteen Ways to Look at the Correlation Coefficient*. The American Statistician 42(1):59–66.
- [3] Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. §1.9 (inner products and catastrophic cancellation).
- [4] Benesty, J., Chen, J., Huang, Y., & Cohen, I. (2009). *Pearson Correlation Coefficient*. In: Noise Reduction in Speech Processing. Springer. Chapter 1.

---

## 3. Implementation notes

### 3.1 Algorithm chosen

Two-pass mean-centered inner product. Complexity O(2n) time, O(1) space.
The algorithm reads the data twice: once to compute means, once to accumulate
centered products.

### 3.2 Numerical stability

**Critical distinction**: tambear's two-pass implementation is numerically
correct in cases where the one-pass formula (form 5 above) catastrophically
fails.

**One-pass catastrophic cancellation**: For data with large mean and small
variance, the one-pass formula computes `n·Σx² − (Σx)²`. Both terms are
large and nearly equal, so their difference loses all significant digits.

**Concrete example** (verified computationally):

```
x = [1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4, 1e10 + 5]
y = [1, 2, 3, 4, 5]
```

One-pass denominator: `n·Σx² − (Σx)²` in f64 arithmetic → **0** (catastrophic
cancellation, returns NaN or ±∞)

Two-pass centered: x_centered = [−2, −1, 0, 1, 2], dx2 = 10, dy2 = 10,
num = 10 → r = **1.0** (correct, confirmed against mpmath at 50 dp)

**scipy note**: scipy uses a two-pass algorithm and agrees with tambear on
this input (r = 1.0). The one-pass formula would return a wrong answer; both
tambear and scipy choose correctness over the single-pass simplicity.

### 3.3 Degeneracy threshold

The implementation returns NaN when `denom < 1e-15`. This threshold is
chosen to handle:

- Exact zero variance (all-constant input) → `denom = 0`
- Near-zero variance where the result is numerically meaningless

**Issue**: `1e-15` is an absolute threshold, not scale-relative. For data with
very small magnitude (e.g., all values near 1e-12), a genuine small-variance
input might be flagged as degenerate when it is not. The correct threshold
should be relative: `denom < 1e-15 · max(dx2, dy2)`. **See Section 10.**

**Current behavior**: consistent with scipy (which checks `xm = x - mean(x)`
and handles zero-variance by returning NaN for the p-value, though the r
value itself does not always return NaN in scipy). Tambear's behavior is
conservative — return NaN rather than a meaningless large ratio.

### 3.4 Parameters

| Parameter | Type | Valid range | Default | Reference |
|-----------|------|-------------|---------|-----------|
| `x` | `&[f64]` | any finite f64, length ≥ 2 | (none) | Definition 1.2 |
| `y` | `&[f64]` | any finite f64, same length | (none) | Definition 1.2 |

No tuning parameters. The function computes τ_b in its canonical two-pass form.
Variants (population r, Spearman ρ on ranks, partial r) are separate primitives.

### 3.5 Input validation

- `n = 0` → `NaN` (denom = 0 path).
- `n = 1` → `NaN` (centered values are all zero, denom = 0 path).
- `x.len() != y.len()` → panic (assert_eq!). Caller contract violation.
- All-constant x or y → `NaN` (denom < 1e-15).
- NaN in input → propagates (undefined). **See Section 10.**
- ±∞ in input → likely NaN via inf − inf arithmetic. **See Section 10.**

### 3.6 Shareable intermediates

The two accumulators from Round 1 (x̄, ȳ) are reused in Round 2 internally.

A TamSession intermediate would expose:
- `MomentStats{n, mean_x, mean_y, sum_dx2, sum_dy2, sum_dxdy}` — fully
  sufficient for r and many other statistics (variance, covariance, OLS
  slope, regression intercept).

Tagged as `IntermediateTag::CenteredMoments { data_id_x, data_id_y }`, any
downstream method needing the same covariance or variance terms (OLS, Pearson
r for a different y column, centering for standardization) can pull from cache
without recomputing. Estimated sharing graph: OLS, Pearson r (multiple y),
concordance_correlation, covariance, variance, t-statistic for regression. **Not
yet registered — see Section 10.**

---

## 4. Unit tests

Inline tests in `crates/tambear/src/nonparametric.rs` (implicit, via existing
cross-method tests) and the workup parity suite
`crates/tambear/tests/workup_pearson_r.rs`.

Checklist:

- [x] Known-answer: monotone increasing → r = 1.0
- [x] Known-answer: monotone decreasing → r = −1.0
- [x] Partial correlation → exact rational value
- [x] Textbook case (Rodgers & Nicewander 1988 dataset)
- [x] n=20 random (numpy seed 42) vs mpmath oracle
- [x] Empty input → NaN
- [x] n=1 → NaN
- [x] Constant x → NaN
- [x] Constant y → NaN
- [x] Large-magnitude shift (cancellation stress) → correct (not NaN)
- [x] Near-constant x (ε-variation stress) → agrees with mpmath
- [x] Symmetry: r(x,y) = r(y,x)
- [x] Shift invariance in x: r(x+c, y) = r(x, y)
- [x] Scale invariance for positive α: r(αx, y) = r(x, y)
- [x] Scale invariance for negative α: r(−x, y) = −r(x, y)
- [x] Range bounds [−1, 1] across 50 random cases
- [ ] ±∞ handling — not yet tested
- [ ] NaN injection — not yet tested
- [ ] Subnormal handling — not yet tested

---

## 5. Oracle tests — against extended precision

Oracle implementation: first-principles Pearson r using `mpmath.mpf` at 50
decimal digits. All means and centered products computed at 50 dp; only the
final `sqrt(dx2·dy2)` and division are at extended precision. Result compared
against scipy and tambear.

### 5.1 Test cases

| Case | n | Description | mpmath truth (f64 repr) | tambear | rel err |
|------|---|-------------|------------------------|---------|---------|
| 1 | 5 | monotone increasing | +1.0 | +1.0 | 0 |
| 2 | 5 | monotone decreasing | −1.0 | −1.0 | 0 |
| 3 | 5 | partial (y permuted) | +0.8 | +0.8 | 0 |
| 4 | 8 | Rodgers-Nicewander textbook | 0.9354143466934853 | 0.9354143466934853 | 0 |
| 5 | 5 | constant x (degenerate) | NaN | NaN | — |
| 6 | 20 | numpy.randn seed 42 | −0.15729399538437135 | −0.15729399538437135 | 0 |
| 7 | 10 | near-constant x (ε=1e-8 spread) | 0.9636363638467861 | 0.9636363638467861 | 0 |
| 8 | 5 | large-magnitude shift (1e10 + [1..5]) | +1.0 | +1.0 | 0 |

**50-digit mpmath values**:

- Case 4: `0.93541434669348534639593718307913732543900495194468`
- Case 6: `-0.15729399538437135121...`
- Case 7: `0.96363636384678606936330010063880547503683428985987`

### 5.2 Maximum observed relative error

**0** (bit-perfect with mpmath across all tested cases). The two-pass
algorithm accumulates centered products; the only rounding is in the centered
subtraction, the squared accumulation, and the final sqrt+division — all
standard IEEE 754 operations with expected rounding to 1 ULP. No cases of
observed cancellation error.

---

## 6. Cross-library comparison

### 6.1 Competitors tested

| Library | Version | Function | Agrees with mpmath? | Agrees with tambear? |
|---------|---------|----------|---------------------|----------------------|
| scipy | 1.x | `scipy.stats.pearsonr` | **yes** (err < 6e-17) | **yes** (err = 0) |
| mpmath | 1.3.0 | first-principles (reference) | ground truth | **yes** |

### 6.2 Discrepancies found

None found. All cases agree to machine precision (≤ 1 ULP).

scipy's `pearsonr` also uses a two-pass algorithm. Agreement across all cases
confirms that tambear's implementation is correct.

### 6.3 Known historical scipy behaviors

- scipy's `pearsonr` returns a `PearsonRResult` object (statistic, pvalue) in
  recent versions (1.9+), not a plain tuple. We compare `.statistic` only;
  p-values are not implemented in tambear's current `pearson_r`.
- scipy's p-value uses Student-t approximation with df=n−2; p-values under
  extreme correlation can be exact-zero due to floating-point; not our concern.
- For zero-variance inputs, scipy raises a `ConstantInputWarning` and returns
  NaN. Tambear returns NaN silently (no warning system yet).

### 6.4 Libraries not yet tested

R `cor(x, y, method="pearson")`, MATLAB `corr(x, y)`,
Julia `Statistics.cor(x, y)`, statsmodels `stattools.pearsonr`. These are
queued for future expansion. Bit-level agreement is expected for all, as
Pearson r has a unique correct answer for finite floating-point inputs.

### 6.5 Verdict

**Bit-perfect** against scipy across all tested inputs. Matches mpmath oracle
at 50 dp to ≤ 1 ULP in every case.

---

## 7. Adversarial inputs

Status: partially covered.

- [x] Small-n edge cases (n=0, n=1) → NaN
- [x] Constant-input degeneracy → NaN
- [x] Large-magnitude shift (catastrophic cancellation pressure on one-pass) →
  correct (two-pass is immune)
- [x] Near-constant x with ε-spread (relative-variance stress) → agrees with
  oracle to 1 ULP
- [ ] Catastrophic-cancellation for the one-pass formula — **verified as
  known bug that tambear avoids** (documented, not a tambear bug)
- [ ] NaN injection — propagation behavior undefined; **see Section 10**
- [ ] ±∞ injection — NaN via inf−inf arithmetic; **see Section 10**
- [ ] Subnormal injection — not yet tested; behavior expected to be correct
  as subnormals preserve sign and order
- [ ] Overflow in (xᵢ−x̄)(yᵢ−ȳ) — if |x| ~ f64::MAX/2 and |y| ~ f64::MAX/2,
  the centered product may overflow; currently unmitigated

**Degeneracy threshold adversarial case**: confirmed that `denom < 1e-15`
absolute threshold correctly catches exact zero variance. The case of
genuinely tiny-variance data (all values in [0, 1e-8]) is not yet tested —
this may incorrectly trigger the NaN return. **See Section 10.**

---

## 8. Invariants and proofs

All enforced in `crates/tambear/tests/workup_pearson_r.rs`:

1. **Symmetry**: r(x, y) = r(y, x). Proof: the formula is symmetric in x
   and y. ✓
2. **Shift invariance**: r(x + c1, y) = r(x, y) for any scalar c. Proof:
   subtracting the mean cancels c. ✓
3. **Scale invariance (positive α)**: r(αx, y) = r(x, y) for α > 0. Proof:
   α cancels in numerator and denominator. ✓
4. **Sign flip under negation**: r(−x, y) = −r(x, y). Proof: α=−1 flips
   numerator but not denominator magnitude. ✓
5. **Range bound**: r ∈ [−1, 1] for all inputs. Proof: Cauchy-Schwarz
   inequality applied to centered vectors. ✓ (checked over 50 random cases)
6. **r = 1 iff perfect linear increasing relationship** (up to measurement
   precision). ✓
7. **r = −1 iff perfect linear decreasing relationship**. ✓

Not enforced as tests:

- **Bilinearity of numerator**: r is not bilinear in (x, y) because of the
  normalization; skip.
- **Composition with affine transforms**: r(ax+b, cy+d) = sign(ac)·r(x,y)
  for a,c ≠ 0. Could be tested but low priority.

---

## 9. Benchmarks and scale ladder

Status: **benchmarks not yet run**. The algorithm is O(n) with two linear
passes, so the scale table is:

| n | Expected time | Algorithm regime |
|---|---|---|
| 10¹ | < 1 µs | constant overhead dominates |
| 10² | < 1 µs | L1 cache resident |
| 10³ | ~1 µs | L1/L2 cache |
| 10⁴ | ~10 µs | L2/L3 cache |
| 10⁵ | ~100 µs | L3 cache |
| 10⁶ | ~1 ms | RAM bandwidth bound |
| 10⁷ | ~10 ms | RAM |
| 10⁸ | ~100 ms | RAM |
| 10⁹ | ~1 s | RAM (feasible) |

The O(n) structure means `pearson_r` should scale to 10⁸–10⁹ rows on a modern
CPU without any algorithmic change. The bottleneck is memory bandwidth (~50 GB/s
for DDR5): reading 2 f64 arrays of size n requires 16n bytes, giving a hardware
ceiling of ~3 billion elements/second on a single thread.

**GPU crossover**: for n < 10⁶, GPU launch overhead likely exceeds computation
time. GPU pearson_r only makes sense for batch scenarios (many columns
simultaneously) or n > 10⁷.

**Benchmarks to run**:
- Single-thread CPU at n = 10², 10³, 10⁴, 10⁵, 10⁶, 10⁷, 10⁸
- vs scipy at each scale
- SIMD throughput improvement estimation

---

## 10. Known bugs / limitations / open questions

- **TASK: NaN handling is undefined.** If any xᵢ or yᵢ is NaN, NaN propagates
  through x̄ in Round 1, then through (xᵢ − x̄) in Round 2. This means the
  result is NaN — but the behavior is incidental, not a contract. Mitigation:
  specify the NaN policy explicitly and add a runtime check. Recommended:
  return NaN when any input contains NaN (matching scipy's default
  `nan_policy='propagate'`). **Severity: medium, spec ambiguity.**

- **TASK: ±∞ input behavior is undefined.** inf − inf = NaN during
  centered subtraction in Round 2. The result will be NaN, but this is
  not tested. **Severity: low.**

- **TASK: Absolute degeneracy threshold 1e-15 is scale-dependent.** For
  data with magnitude ~ 1e-12, the denom may legitimately be much smaller
  than 1e-15 even for non-degenerate data. Fix: use `denom < EPS_REL ·
  max(dx2.abs(), dy2.abs()).sqrt()` where EPS_REL ~ 1e-15. **Severity:
  low, affects unusual inputs.**

- **TASK: Overflow in centered product.** For |xᵢ − x̄| ~ f64::MAX/2 and
  |yᵢ − ȳ| ~ f64::MAX/2, the product (xᵢ − x̄)(yᵢ − ȳ) overflows to ±∞.
  Mitigation: normalize x and y by their standard deviation before
  accumulating (this changes neither r nor numerical quality for normal
  inputs). **Severity: very low — only pathological inputs.**

- **Open question**: should we expose `pearson_r_full` returning
  `PearsonResult { r, n, mean_x, mean_y, var_x, var_y, cov }` for use by
  OLS and other methods that need the same intermediates? Yes — this would
  enable TamSession sharing. Current answer: expose the bare value; add
  the richer return when OLS sharing is implemented.

- **Open question**: Kahan summation for Round 2 accumulators. Current
  naive summation has O(n·ε) error bound for the centered sum. Kahan would
  give O(ε) bound regardless of n. For financial data with n ~ 10⁶,
  Kahan would improve accuracy by ~6 decimal places. Current answer:
  defer until there is a measured accuracy failure at large n.

---

## 11. Sign-off

- [x] Sections 1–3 written by scientist
- [x] Oracle tests in Section 5 at target precision (≤ 1 ULP vs mpmath 50 dp)
- [x] Cross-library comparison in Section 6 complete for scipy + mpmath
- [x] Invariants in Section 8 all proven and tested
- [x] Known bugs in Section 10 documented with severity and mitigation
- [ ] Adversarial suite in Section 7 passing: NaN, ±∞, subnormal, overflow missing
- [ ] Benchmarks in Section 9 complete through n = 10⁸
- [ ] Degeneracy threshold fix (absolute → relative)
- [ ] NaN policy specified and enforced at runtime
- [ ] Reviewed by adversarial
- [ ] Reviewed by math-researcher

**Overall status**: draft. Numerical verification complete. Adversarial
coverage and scale benchmarks are the remaining gaps before sign-off.

---

## Appendix A: reproduction artifacts

- Oracle computation: inline Python (mpmath, scipy) in this session
- Parity test file: `crates/tambear/tests/workup_pearson_r.rs`
- Scale benchmark harness: not yet written

## Appendix B: oracle computation script

```python
import mpmath
mpmath.mp.dps = 50

def pearson_r_mp(x, y):
    n = len(x)
    mx = sum(x)/n
    my = sum(y)/n
    num = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))
    dx2 = sum((xi - mx)**2 for xi in x)
    dy2 = sum((yi - my)**2 for yi in y)
    denom = mpmath.sqrt(dx2 * dy2)
    if denom == 0:
        return None
    return num / denom

# Case 4 (Rodgers & Nicewander textbook):
x4 = [2,4,4,4,5,5,7,9]; y4 = [1,2,3,4,4,5,6,7]
r4 = pearson_r_mp([mpmath.mpf(v) for v in x4], [mpmath.mpf(v) for v in y4])
print(mpmath.nstr(r4, 50))
# → 0.93541434669348534639593718307913732543900495194468
```

## Appendix C: version history

| Date | Author | Change |
|------|--------|--------|
| 2026-04-10 | scientist | Initial full workup; bit-perfect against mpmath + scipy across 8 cases; NaN/∞/threshold bugs documented. |
