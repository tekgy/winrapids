# Scout's Proposals — Next Landscape Wave

*Scout, 2026-04-10 (during hold, after classification-bijection theory session)*

---

## 1. Kingdom Classification Audit (theory → code)

The session produced a verified theorem and a complete classification table. The next
step is applying the two-step test systematically across time_series.rs and volatility.rs:

**Two-step test for every recurrence labeled Kingdom B:**
1. Can you write the state update as matrix multiply on an augmented state vector?
   If yes → Kingdom A (representational Fock boundary, dissolves with right algebra)
2. Is the map at step t data-determined with a finitely-representable semigroup?
   If yes → Kingdom A over some semiring (may need TropicalMinPlus / AffineSemigroup op)
3. If neither → genuine Kingdom B (MCMC, EGARCH, TAR, BOCPD)

**Expected findings**: several functions currently labeled Kingdom B will reclassify.
The ARMA CSS implementation is the canonical case: Kingdom B implementation of
Kingdom A math (Kalman formulation).

**What I'd seed**: `kingdom-classification-audit` under theory, owned by scout +
math-researcher. Output: corrected docstrings + list of implementation debt items
where the code runs sequentially but the math is Kingdom A.

---

## 2. Semiring Trait Design (architecture — URGENT, blocks Op enum)

The theorem is settled: phyla are (grouping, op, semiring) triples. The Semiring<T>
trait must be designed before pathmaker extends the Op enum with TropicalMinPlus /
TropicalMaxPlus. Adding these as enum variants is the wrong shape.

**Correct shape**:
```rust
trait Semiring<T> {
    fn zero() -> T;      // additive identity (∞ for tropical-min, -∞ for tropical-max)
    fn one() -> T;       // multiplicative identity (0 for tropical, 1 for standard)
    fn add(a: T, b: T) -> T;   // associative, commutative (min for tropical-min)
    fn mul(a: T, b: T) -> T;   // associative, distributes over add (+ for tropical)
}
```

**Instances needed immediately**:
- `AdditiveReal` — (ℝ, +, ×) — standard accumulate
- `TropicalMinPlus` — (ℝ∪{∞}, min, +) — PELT, Bellman-Ford, Floyd-Warshall
- `TropicalMaxPlus` — (ℝ∪{-∞}, max, +) — Viterbi, longest path
- `LogSumExp` — (ℝ, lse, +) — HMM forward, softmax, attention
- `AffineSemigroup` — (2-param pairs, compose) — EMA, GARCH, AR(p) as first-class

**What I'd seed**: `semiring-trait-design` under architecture — owned by aristotle +
pathmaker. Gate: pathmaker cannot extend Op enum until this lands.

---

## 3. Universality Class Experiment (theory → experiment)

The experiment framing is complete and documented. Missing primitive:
`ks_test_custom(empirical: &[f64], theoretical_cdf: impl Fn(f64) -> f64)`.

**Three Wigner surmise CDFs needed**:
- `wigner_surmise_goe_cdf(s: f64) -> f64` — closed form: 1 - exp(-πs²/4)
- `wigner_surmise_gue_cdf(s: f64) -> f64` — numerical integration needed
- `poisson_spacing_cdf(s: f64) -> f64` — closed form: 1 - exp(-s)

**The experiment**: market eigenvalue spacings vs GOE/GUE/Poisson. Default expectation
GOE (real symmetric correlation matrix = Wishart ensemble). Finding GUE = hidden
complex structure. Publishable either way.

**What I'd seed**: `universality-class-experiment` under theory/rigor — owned by
scientist + math-researcher. Blocks on `ks_test_custom` primitive.

---

## 4. Tropical Op Annotations in graph.rs (quick win)

Bellman-Ford, Dijkstra, and Floyd-Warshall are already in graph.rs. They are tropical
Kingdom A algorithms unnamed as such. Three one-line annotation additions:

```rust
// Kingdom A: tropical SpMV = accumulate(edges, Graph, dist[u]+weight, TropicalMinPlus)
pub fn bellman_ford(...) { ... }

// Kingdom A: greedy tropical (priority-queue-accelerated Bellman-Ford)
pub fn dijkstra(...) { ... }

// Kingdom A: tropical closure = iterated tropical matrix multiply
pub fn floyd_warshall(...) { ... }
```

No behavior changes. Pure documentation. Makes the tropical framework visible in code.

**What I'd seed**: fold into the kingdom-classification-audit campsite, or as a
standalone quick-win under architecture.

---

## 5. silhouette_score GPU Opportunity Note

`clustering.rs:698` is fixed (now Kingdom A). But the implementation is sequential CPU.
The pairwise distance matrix (n² independent computations) is the canonical GPU target
— all pairs truly independent, no dependencies whatsoever. For large n this is a direct
GPU wins.

**What I'd seed**: note in the rolling-primitives campsite or as standalone
`silhouette-gpu-parallel` under architecture. Not urgent but captures a real GPU
opportunity once tam-gpu has the pairwise grouping pattern.

---

## Priority read from theory session

The theory work this session produced one urgent architectural dependency:

**Semiring trait design MUST precede Op enum extension.**

Everything else can be sequenced freely. But if pathmaker extends Op before aristotle
designs the Semiring trait, we'll get a flat list of semiring enum variants that
scales O(semirings) instead of O(1). That's a one-way door.

The classification audit is the right companion work — it generates the list of
algorithms that NEED tropical or affine semigroup ops, which drives the trait design.
