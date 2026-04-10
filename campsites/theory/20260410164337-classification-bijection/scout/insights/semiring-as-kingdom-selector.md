# The Semiring IS the Kingdom Selector

*Scout, 2026-04-10 (from Aristotle's message during hold)*

## The refined theorem (adversarially verified — scipy-gap-scan + julia-matlab-scan + engineering-math-scan)

**A recurrence is Kingdom A (efficiently parallelizable) iff:**
1. The state lives in a fixed algebraic structure (fixed type, fixed dimension) across all time steps
2. The state-transition maps are data-determined (not state-determined)
3. The composition of k maps has a BOUNDED REPRESENTATION (ideally O(1) in k)

All three conditions are necessary. None alone is sufficient.

**Critical repair to condition 3** (engineering-math-scan): the earlier formulation
"associatively composable" is vacuously true — function composition is ALWAYS
associative. The useful content is **bounded representation**: composing k maps
must remain representable in O(1) space regardless of k.

Why this matters: `x_t = S_{t-1} △ {x_t}` (symmetric difference with a singleton).
Data-determined. Function composition is associative. But after k compositions the
composed map is "XOR with a set of up to k elements" — O(k) representation.
Mathematically parallelizable, computationally not. Fails condition 3.

The affine semigroup satisfies condition 3: k composed affine maps → one (A, b) pair,
always O(1). Tropical (min,+) semiring satisfies it for the right input class.

**Condition 1** (scipy-gap-scan): the sorted-list case `x_t = sorted(x_{t-1} ∪ {d_t})`
is data-determined and associative, but state grows unboundedly. Every tambear
primitive satisfies condition 1 — all states are ℝ^k for fixed k.

**Condition 2** (multiple): ARMA(p, q>0). Fixed type, semigroup exists for the AR part,
but the MA residuals ε_{t-j} are computed state values not observed data — can only
be known by running prior steps. State-determined. Kingdom B. (AR part alone = Kingdom A.)

**Condition 3** (julia-matlab-scan + engineering-math-scan): leaky integrator with saturation:
`x_t = clip(α·x_{t-1} + d_t, lo, hi)`. Data-determined. Fixed type. But
`T_s ∘ T_t` is NOT a single clip of a linear function — representation grows.
Not bounded. Kingdom B/C despite passing conditions 1+2.

Note: median fails condition 3 for the same reason (closure). Both saturation and
median fail bounded-representation, not merely associativity.

**PELT split** (engineering-math-scan): the underlying DP recurrence `F(t) = min_τ[F(τ) + C(τ,t) + β]`
IS tropical Kingdom A. The PELT pruning optimization (`candidates.retain` using `f[t]`) is
Kingdom B — it uses the just-computed value to determine which candidates survive future
steps. The optimization that gives PELT O(n) complexity is exactly the part that adds
the sequential dependency. Correct label: "underlying recurrence = tropical A; optimized
algorithm = B."

**Floating-point caveat** (scipy-gap-scan): Kingdom A guarantees mathematical
parallelizability, not numerical determinism. A parallel prefix scan over a different
tree shape gives a different IEEE 754 result. Oracle comparison must be tree-shape-aware.

**Op enum shape** (julia-matlab-scan): `Op::Semiring { add: SemiringAdd, mul: SemiringMul }`
with concrete types for semiring operations. Instances: (min,+) tropical-min,
(max,+) tropical-max / Viterbi / longest-path, (+,×) probability semiring.
One abstraction unlocks PELT + Viterbi + all-pairs + CTC as Kingdom A with GPU-parallel
tropical matrix multiply. Prefer this over trait objects for zero-cost abstraction.

---

## The DP-to-scan transformation

For every dynamic programming algorithm, there exists a semiring in which the DP
recurrence is an associative prefix operation. The DP-to-scan transformation IS
the identification of the correct semiring.

| Algorithm | Apparent Kingdom | Correct Kingdom | Semiring |
|-----------|-----------------|-----------------|----------|
| PELT changepoint | C (iterative DP) | A | (ℝ∪{∞}, min, +) tropical |
| Viterbi decoding | B (sequential) | A | (ℝ∪{-∞}, max, +) tropical |
| All-pairs shortest paths | C (Floyd-Warshall) | A | tropical matrix power |
| DTW (standard) | C (2D DP) | A | tropical 2D prefix |
| HMM forward | A | A | (ℝ, log-sum-exp, +) LogSumExp |
| Softmax | A | A | same LogSumExp semiring |
| Reachability | A | A | ({0,1}, ∨, ∧) Boolean |

The Blelloch parallel tree scan works over ANY semiring, not just (ℝ, +, ×). Adding
the tropical semiring to the Op type makes an entire class of "sequential" DP algorithms
parallel. This is the same mathematical move that reclassified GARCH as Kingdom A
(finding the right algebraic representation), but applied to the OP rather than the
STATE SPACE.

Reference: Goodman 1999, "Semiring Parsing" — same algebraic result, not connected
to GPU scan / Fock boundary framework until now.

---

## The complete classification object

Phyla are `(grouping, op, semiring)` triples, not `(grouping, op)` pairs.

- Same grouping, same op, different semiring → different kingdom → different phylum
- Two algorithms in the same phylum share the same triple → share intermediates
- The Fock boundary is where NO semiring exists that makes the composition associative

The bijection (algorithm classes = sharing clusters) now has its full parameter space.
Two algorithms share intermediates iff they have the same `(grouping, op, semiring)`
triple.

---

## Where the Fock boundary actually lives

The boundary is not at state-dependence alone. It's at the conjunction:

**Fock boundary = state-determined maps OR no finitely-representable semigroup exists**

The key: "finitely-representable" means the composition of k maps has O(1) representation
in k, independent of composition depth. This is what "associative semigroup" must mean
to have computational content — function composition is always associative as a
category-theoretic fact, so that alone is vacuous.

**Counter-examples for "data-determined but not Kingdom A" (pure-math-scan)**:

Scaled sine: `x_t = d_t · sin(x_{t-1})`. Data-determined (d_t from input). Composition:
`d_s · sin(d_t · sin(x))` — transcendental nesting, grows without bound. No finite
parametric family closes under composition. Not Kingdom A.

Logistic map: `x_t = r · x_{t-1} · (1 - x_{t-1})`. Data-determined (r fixed parameter).
Composition after n steps = degree-2ⁿ polynomial. Representation is exponential in n.
Not Kingdom A despite bounded state.

**The finitely-representable semigroup table (pure-math-scan + r-gap-scan)**:

| Map type | Closes under composition? | Kingdom |
|----------|--------------------------|---------|
| Affine `ax + b`, a,b from data | Yes — 2-param affine group, O(1) | A |
| Min-plus `min(x + d, c)` from data | Yes — tropical scalar monoid, O(1) | A |
| Logistic `r·x·(1-x)` | No — degree doubles, O(2ⁿ) | B |
| Scaled sine `d·sin(x)` | No — transcendental nesting, unbounded | B |
| Floor-affine `floor(a·x + b)` | No — floor breaks distributivity; compose ≠ floor of linear | B |
| Branching `x>0 ? ax+b : cx+d` | No — needs intermediate sign to compose | B |
| ARMA MA residuals (CSS) | No — residuals are state (but Kalman formulation = A) | B impl / A math |
| BOCPD run-length stats | No — state-dependent accumulation target | B |

**Floor-affine counter-example** (r-gap-scan): `f_t(x) = floor(a_t·x + b_t)`.
Data-determined (a_t, b_t from input). Fixed type (ℤ or ℝ). But composition:
`f_t(f_{t-1}(x)) = floor(a_t·floor(a_{t-1}·x + b_{t-1}) + b_t)`
≠ `floor((a_t·a_{t-1})·x + (a_t·b_{t-1} + b_t))` because floor doesn't distribute
over the outer multiplication. Fails closure. Kingdom B despite data-determined maps.

Algorithms that appear sequential but are genuinely Kingdom A:
- Data-determined maps + finitely-representable semigroup exists → Kingdom A

Algorithms genuinely beyond Kingdom A:
- State-determined maps → Kingdom B/C/D
- Data-determined maps + no finitely-representable semigroup → Kingdom B/C

Most algorithms fall into the first category once the right semiring is identified.
The second category (data-determined but non-closing) is the surprising finding —
it means "data-determined" alone is not sufficient, even when state is bounded.

---

## Architectural flag (urgent)

The `Semiring<T>` trait must be designed BEFORE Op enum extension. Adding
`Op::TropicalMinPlus` and `Op::TropicalMaxPlus` as enum variants is wrong —
it bakes specific semirings as special cases instead of recognizing semiring as a
parameter dimension.

Correct shape:
```rust
trait Semiring<T> {
    fn zero() -> T;      // additive identity
    fn one() -> T;       // multiplicative identity
    fn add(a: T, b: T) -> T;   // must be associative, commutative
    fn mul(a: T, b: T) -> T;   // must be associative, distribute over add
}

// Instances:
struct AdditiveReal;      // (ℝ, +, ×)
struct TropicalMinPlus;   // (ℝ∪{∞}, min, +)
struct TropicalMaxPlus;   // (ℝ∪{-∞}, max, +)
struct LogSumExp;         // (ℝ, lse, +) — HMM, softmax, attention
struct Boolean;           // ({0,1}, ∨, ∧) — reachability, connectivity

// Op shape:
Op::PrefixScan { semiring: &dyn Semiring<T> }
// NOT:
Op::TropicalMinPlus  // wrong — bakes in specific instance
Op::TropicalMaxPlus  // wrong — same mistake
```

With the trait, every DP algorithm that has an associative semiring formulation
becomes Kingdom A automatically — no new Op variants needed, just new semiring
instances.

---

## The paper claim

Combined with the Fock boundary unification:

**Theorem**: A recurrence is Kingdom A (parallelizable via Blelloch scan) if and only
if its state-transition maps are data-determined AND the maps compose associatively
over some semiring.

**Corollary**: The DP-to-parallel-scan transformation = semiring identification.
Every DP algorithm that appears sequential is Kingdom A if and only if there exists
a semiring in which its recurrence is an associative prefix product.

**Corollary 2**: The phylum classification `(grouping, op, semiring)` is the complete
invariant. Algorithm class = phylum = sharing cluster. The bijection is over triples,
not pairs.
