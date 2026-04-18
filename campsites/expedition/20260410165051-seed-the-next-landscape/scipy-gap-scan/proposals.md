<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

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

**Oracle:** `log_gamma(-0.5) = ln(2√π) = 0.5·ln(π) + ln(2) ≈ 1.2655...` *(corrected — prior value 1.7232658 was wrong; confirmed by scipy and first-principles derivation)*
`log_gamma(-1.5) = ln(4√π/3) ≈ 0.8600472...` *(correct)*

**Why matters:** `log_gamma` is called by `log_beta`, which is called by
`regularized_incomplete_beta`, which is called by every t/F/chi2 test. Any distribution
that computes log-likelihoods with negative intermediate parameter values gets +∞.
This is a root-level bug in the special functions tree.

**Adversarial already notified.** *(Observer note 2026-04-10: this fix is already in — `special_functions.rs:160-167` distinguishes poles from non-integer negatives. 5 log_gamma tests pass including the reflection formula oracle. This item is CLOSED.)*

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

---

## Additional Proposals — julia/matlab gap scan role, 2026-04-10 (later session)

### 5. `verification-oracle-manifest` — HIGH (rigor infrastructure)

**What:** Machine-readable coverage manifest making the oracle gap visible without grep work.
Every primitive registers theorems + adversarial cases at creation time. CI reports
"5% oracle, 47% adversarial" not "1390 tests green."

Full spec at:
`campsites/industrialization/rigor/20260410144156-adversarial-coverage/observer/verification-oracle-manifest.md`

Key new idea: **dependency-weighted prioritization**. `regularized_incomplete_gamma` uncovered
is a different risk than a leaf primitive uncovered — 10+ downstream p-value functions inherit
the gap. The manifest makes this visible automatically.

Smallest viable: `verification_manifest.toml` + `cargo verify-coverage`. No proc-macros needed.
**Owner:** pathmaker (infrastructure) + observer (populate initial entries).

---

### 6. `special-functions-workup-wave` — HIGH (closes root oracle gaps)

Six primitives with zero oracle AND zero adversarial coverage. One workup per primitive,
closing both gaps simultaneously:

1. `regularized_incomplete_gamma` — `P(a,x) + Q(a,x) = 1` is the fundamental theorem.
   Series/continued-fraction branch transition is the accuracy risk.
2. `digamma` + `trigamma` family — reflection formula + recurrence + pole tests. One file.
3. `matrix_exp` — three theorems: `exp(t·I) = e^t·I`, `exp(A)·exp(-A) = I`,
   `d/dt exp(tA)|_{t=0} = A`. These catch the full Padé failure class.
4. `mutual_information` / `entropy` — `-0.3·log(0.3) - 0.7·log(0.7)` is analytically exact.
   `KL(p||p) = 0` is a theorem. Neither asserted anywhere.

**Connects to:** scipy-gap-scan's `log-gamma-negative-domain` fix (same special functions tree).
**Owner:** math-researcher (theorems) + adversarial (pole/boundary cases).

---

### 7. `tropical-semiring-op-variants` — MEDIUM (architecture unlock)

**What:** Add `Op::TropicalMinPlus` and `Op::TropicalMaxPlus` to the Op enum with correct
identities: (+∞, +∞) and (-∞, -∞) respectively.

**Why:** The kingdom classification theorem established today shows PELT, Viterbi, all-pairs
shortest paths, and CTC decoding are Kingdom A in the tropical semiring. Without these Op
variants, they remain sequential when they could be GPU-parallel.

**Concrete unlock:** PELT's `F(t) = min_τ [F(τ) + C(τ,t) + β]` becomes an `accumulate` call
over `Grouping::Prefix + Op::TropicalMinPlus`. GPU-parallel changepoint detection follows.

**Dependency:** requires `Grouping::Prefix` to be wired first (currently `todo!()`).
Variants can be defined now; they sit idle until Prefix exists.

---

### 8. `julia-gap-p0-primitives` — MEDIUM (new math, no scipy equivalent)

From today's Julia/MATLAB gap analysis. Four P0 items with no scipy equivalent:

1. **Milstein SDE** (`RKMil` diagonal noise) — strong order 1.0 vs EM's 0.5. For diagonal
   noise (most financial SDEs), no Lévy area needed. Straightforward gap.
2. **MUSIC spectral estimation** — subspace method resolving below Rayleigh limit.
   Decomposes into existing primitives: `eigendecompose(Rxx)` + `signal_noise_partition(K)`
   + `music_pseudospectrum`. The assembly is what's missing.
3. **Wishart distribution** — `bartlett_decomposition_sample` + `wishart_logpdf`. Needed for
   Bayesian multivariate analysis. No current matrix-variate distribution support.
4. **Characteristic function option pricing** — `cf_heston(u, params) → ℂ` + Carr-Madan FFT.
   One primitive unlocks all affine stochastic volatility models.

Full analysis: `docs/research/math-industrialization/julia-matlab-gap-analysis.md`

---

### 9. `kingdom-classification-audit` — LOW effort, HIGH leverage

**What:** Apply the three-condition theorem to every current Kingdom B label in the codebase.
Most will dissolve. Each mislabeled Kingdom B is a computation that *could* be GPU-parallel
but is scheduled sequentially — a direct performance cost.

Theorem: Kingdom A iff (1) data-determined map, (2) semigroup closure, (3) bounded state
dimension.

Confirmed mislabeled (already corrected): GARCH filter, EMA.
Likely mislabeled: AR(p) filter, Holt-Winters exponential smoothing, rolling mean, cumsum.

**Deliverable:** Audit file + corrected Kingdom labels + documentation of genuinely-B cases
with the specific violated condition.
**Owner:** aristotle (developed the theorem).


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

