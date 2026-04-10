# Scipy Gap Scan — Next Landscape Proposals

*2026-04-10, end of industrialization expedition*

---

## What I did this session

Mapped tambear's coverage against scipy's complete public API across 9 submodules
(~893 functions). Result at `docs/research/math-industrialization/scipy-gap-analysis.md`.

Headline: ~172 HAVE, ~42 PARTIAL, ~679 MISSING. Tambear has ~19% of scipy by function
count. The flat list obscures which 8 functions unlock 40% of the rest.

Also found a real bug while checking boundary handling: `log_gamma` returns +∞ for ALL
x ≤ 0, including non-integer negatives where the answer is finite. Reported to adversarial
for wave 19.

---

## Proposals for Next Landscape

### 1. `scipy-gap-dependency-topology` — HIGH

**What:** The scipy gap analysis is currently a flat list of 679 missing functions.
That's the wrong representation for deciding what to implement next. What's needed
is the dependency graph: which missing primitives are upstream of which others?

**The structure that emerged from this session:**
- **Tier 0 (unlocks everything):** `erfinv`, `betaincinv`, `gammaincinv`, `hyp2f1`
- **Tier 1 (unlocks distributions):** `erfinv` → ~40 normal-family quantiles; `betaincinv` → t/F/chi2/beta quantiles at full precision; `hyp2f1` → non-central distributions, elliptic integrals, associated Legendre polynomials
- **Tier 2 (unlocks geometry/physics):** elliptic integrals (ellipk, ellipe, Carlson forms) → Riemann surface computations, pendulum, geodesics; `lambertw` → tree algorithms, Lambert-Beer law, delay equations
- **Tier 3 (leaves):** ~80 continuous distributions, boolean distances, Lp distances — mechanical catalog, no new algorithmic ideas

**Work:** Read scipy's implementation of each missing upstream function, trace its
dependencies into tambear, produce a DAG. Probably 2-3 hours of research.

**Why now:** Before implementing anything from the gap list, the topology tells us
where to start. Implementing in wrong order = building leaves before roots.

---

### 2. `log-gamma-negative-domain` — HIGH (bug fix, not research)

**What:** `log_gamma(x)` returns +∞ for ALL x ≤ 0. Should only return +∞ at integer
non-positives (poles). Non-integer negatives (x = -0.5, -1.5, -2.5, ...) have finite
values via the reflection formula: `ln|Γ(x)| = ln(π) - ln|sin(πx)| - log_gamma(1-x)`.

**Current code:**
```rust
pub fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 { return f64::INFINITY; }  // catches ALL x ≤ 0
    if x < 0.5 {  // reflection formula — never reached for x < 0
        ...
    }
```

**Fix:**
```rust
pub fn log_gamma(x: f64) -> f64 {
    // Poles at non-positive integers only
    if x <= 0.0 {
        if x == x.floor() { return f64::INFINITY; }  // integer pole
        // Non-integer negative: use reflection formula
        let lsin = (std::f64::consts::PI * x).sin().abs().ln();
        return std::f64::consts::PI.ln() - lsin - log_gamma(1.0 - x);
    }
    ...
```

**Oracle:** `log_gamma(-0.5) = ln(2√π) ≈ 1.7232658...`
`log_gamma(-1.5) = ln(4√π/3) ≈ 0.8600472...`

**Why matters:** `log_gamma` is called by `log_beta`, which is called by
`regularized_incomplete_beta`, which is called by every t/F/chi2 test. Any distribution
that computes log-likelihoods with negative intermediate parameter values gets +∞.
This is a root-level bug in the special functions tree.

**Adversarial already notified.** This could be wave 19 target or a standalone fix.

---

### 3. `normal-quantile-precision-audit` — MODERATE

**What:** `normal_quantile` uses Acklam's (2003) rational approximation, documented as
"accurate to ~1.15e-9." At extreme tails (p = 1e-15, 1 - 1e-15), this may have several
ULP of error. Every test statistic, CI, and power calculation that uses the normal
quantile in the tails is affected.

**The audit:** Compare `tambear::normal_quantile(p)` against a Newton-on-erf reference
(or mpmath's arbitrary-precision ndtri) at p values across the full range including
extreme tails. Measure ULP error. If ≤ 2 ULP everywhere: the approximation is fine and
we can move on. If > 2 ULP in tails: we have a precision bug that's been silently
affecting every extreme-quantile computation.

**Connection to adversarial:** This is the catastrophic-cancellation / special-function-
precision question applied to the most-used quantile function. Either we confirm the
approximation is good enough, or we find the bug.

**Who:** Scientist (it's an experiment — measure, don't assume) or adversarial
(it fits the "assert a mathematical truth, see if the code violates it" pattern).

---

### 4. `scipy-gap-quick-wins-wave` — MODERATE

**What:** 20-30 missing functions from the gap list that are trivial implementations
(< 30 lines each, no new algorithms, compose entirely from existing primitives):

- `zscore` / `zmap` — (x - mean) / std, trivial
- `sem` — std / sqrt(n), trivial  
- `mode` — argmax of frequency count, trivial
- `gstd` — exp(std(log(x))), trivial composition
- `trim1` / `trimboth` — sort + slice, trivial
- `sigmaclip` — iterative outlier removal, < 20 lines
- `circmean` / `circvar` / `circstd` — trig moments, moderate (20-30 lines each)
- `cumfreq` / `relfreq` — cumsum variants, trivial
- `percentileofscore` — binary search on sorted data, trivial
- `ranksums` — presentation alias for mann_whitney, trivial
- All boolean distances (jaccard, dice, hamming, yule, rogerstanimoto, russellrao, sokalsneath) — set formulas, 5-10 lines each
- All Lp distances (cityblock, chebyshev, minkowski, seuclidean, sqeuclidean, braycurtis, canberra) — distance formulas, 5-10 lines each

**Why:** These are the easiest 10% of the gap to close and they're used constantly.
`zscore`, `sem`, `mode` appear in almost every analysis pipeline. The distance functions
are needed for KNN, clustering validation, and any spatial computation.

**Estimate:** One pathmaker session, ~150-200 lines total, closes ~30 gaps.

---

## Cross-Pollination Notes

The `log-gamma-negative-domain` bug connects to adversarial's `adversarial-special-function-poles`
campsite proposal. The specific test cases I sent adversarial are ready to become wave 19.

The `normal-quantile-precision-audit` connects to observer's `oracle-coverage-map` proposal.
The normal quantile is exactly the kind of "we haven't checked the tails" gap the observer
is mapping.

The `scipy-gap-dependency-topology` is the structural complement to the flat gap analysis.
The flat list is the bulk data; the topology is the holographic screen — the minimum
representation that encodes what to build first.

---

*The scipy gap analysis found the territory. The dependency topology is the map.*
