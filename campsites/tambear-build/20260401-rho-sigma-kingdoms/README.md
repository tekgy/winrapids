# Campsite: The (ρ,σ,τ) Algorithm Taxonomy

**Opened:** 2026-04-01  
**Thread:** Parking-lot idea from team-lead; math-researcher mentioned it  
**Status:** Active exploration — expanded to (ρ,σ,τ) cube after discovering τ = solution uniqueness axis

---

## The Observation

The current Kingdom taxonomy (A, B, C) conflates two orthogonal dimensions:

1. **ρ — commutativity of the combine operator**
   - ρ=0: commutative (order doesn't matter) → can split arbitrarily, recombine in any order
   - ρ=1: non-commutative (order matters) → must preserve sequence

2. **σ — pass count**
   - σ=0: single pass (result computable in one traversal)
   - σ=1: multi-pass (result requires iteration until convergence)

The 2×2 table:

```
             σ=0 (single pass)    σ=1 (multi-pass)
           ┌────────────────────┬───────────────────┐
ρ=0 (comm) │   Kingdom A        │   Kingdom C        │
           │   sum, Welford,    │   K-means, IRLS,   │
           │   max, scatter     │   EM, Newton       │
           ├────────────────────┼───────────────────┤
ρ=1 (seq)  │   Kingdom B        │   Kingdom BC       │
           │   Affine scan,     │   ???              │
           │   EWM, Kalman fwd  │   (the empty cell) │
           └────────────────────┴───────────────────┘
```

**Kingdom A = (commutative, single-pass)**
**Kingdom B = (non-commutative, single-pass)**
**Kingdom C = (commutative, multi-pass)**
**Kingdom BC = (non-commutative, multi-pass)** — currently unnamed

---

## What Is in the BC Cell?

Algorithms requiring BOTH sequential order AND multiple passes to converge:

### Empirically verified: Wynn's ε (Shanks transformation)

**The cleanest, smallest, most directly verified BC algorithm.**

```
accumulate(partial_sums, Windowed{3}, AitkenExpr, Custom)  → accelerated seq
repeat until converged                                       → Wynn's epsilon
```

- Inner pass: windowed accumulate over (s_{n-1}, s_n, s_{n+1}) — **order-dependent**
  (sliding window must see terms in sequence; swapping s_{n-1} and s_{n+1} gives a different result)
- Outer loop: iterate until e-table converges — **multi-pass**
- Together: non-commutative inner, iterative outer → Kingdom BC

**Empirical evidence** (from `src/series_accel.rs`, 2026-04-01):
- 20 Leibniz terms, raw partial sum error: 1.25e-2
- After Aitken Δ²: 9.08e-6
- After Euler transform: 8.69e-9
- After Wynn's ε: **3.33e-16** (machine precision)

The e-table update `ε_{n+1}^{(k)} = ε_n^{(k+2)} + 1/(ε_{n+1}^{(k+1)} - ε_n^{(k+1)})` is
a sequential recurrence — each cell depends on its diagonal neighbor. Non-commutative in both
the window-sliding (inner) and the e-table-building (outer) dimensions.

Compare: Aitken Δ² is Kingdom A (windowed, but commutative within window). Euler transform is
Kingdom A (binomial coefficients, commutable). Richardson is Kingdom A (geometric weights,
commutable). Wynn's ε is the first one that needs sequential state across both passes.

---

**Other BC inhabitants** (more complex setup, same algebraic structure):

**Kalman Smoother (RTS smoother)**
- Forward filter: Kalman prediction/update scan (sequential, order-dependent)
- Backward smoother: reverse scan conditioned on forward results
- Iteration: EM-Kalman smoother iterates both passes until convergence
- Nature: non-commutative scan + iterative → pure BC

**Bidirectional RNNs / LSTMs trained with BPTT**
- Forward pass: sequential scan (order-dependent hidden state)
- Backward pass: reverse sequential scan
- Parameter update: iterate both until convergence
- Nature: non-commutative (hidden state), multi-pass (training)

**Fluid simulation with iterative pressure solvers**
- Advection: sequential (Kingdom B — fluid moves along flow)
- Pressure projection: iterative Poisson solve (Kingdom C with spatial stencil)
- Together: BC

**Wavefunction collapse / quantum circuit simulation**
- Gate application: sequential, non-commutative (order of gates matters)
- Measurement/optimization: iterative (VQE, QAOA)

---

## The Series Acceleration Family Spans Three Kingdoms

The four classical series accelerators map cleanly to different cells:

| Algorithm | ρ | σ | Kingdom | Why |
|-----------|---|---|---------|-----|
| Aitken Δ² | 0 | 0 | A | Windowed, but result doesn't depend on window order |
| Euler transform | 0 | 0 | A | Binomial-weighted accumulate — fully commutable |
| Richardson extrapolation | 0 | 0 | A | Polynomial weights — commutable |
| Cesàro summation | 0 | 0 | A | Uniform kernel — arithmetic mean of partial sums |
| Abel summation | 0 | 0 | A | Exponential kernel (x^k) — commutable |
| Abel ∘ Richardson | 1 | 0 | B | Two Kingdom A ops in sequence → Kingdom B pipeline |
| Wynn's ε (Shanks) | 1 | 1 | BC | e-table recurrence is sequential AND iterative |

The gap from A to BC is not gradual. Aitken/Euler/Richardson/Cesàro/Abel all sit in
the same cell. Wynn's ε crosses both axes simultaneously. The Abel ∘ Richardson
composition is Kingdom B — demonstrating that **A ∘ A = B** (two commutative ops in
sequence create a non-commutative pipeline).

### Richardson vs Wynn Head-to-Head on Positive Algebraic Series

When the convergence class is KNOWN algebraic (positive monotone), the matched Kingdom A
accelerator (Richardson) dramatically outperforms the Kingdom BC accelerator (Wynn):

| n | Raw error | Wynn error | Richardson error | Rich/Wynn |
|---|-----------|-----------|-----------------|-----------|
| 20 | 4.88e-2 | 8.65e-3 | 1.64e-4 | 52.8× |
| 40 | 2.47e-2 | 4.76e-3 | 3.00e-7 | 15,841× |
| 80 | 1.24e-2 | 2.63e-3 | 9.71e-9 | 270,857× |
| 160 | 6.23e-3 | 1.36e-3 | 1.69e-11 | **80,441,438×** |

Richardson auto-detects error order p from tail-difference ratios. The gap grows with n
because Richardson cancels successive O(1/N^p) terms while Wynn's rational Padé
mismatches the polynomial error structure.

**Design rule**: use Richardson when convergence class is known algebraic; Wynn when unknown.
The `accelerate()` dispatcher in `series_accel.rs` implements this automatically.

### Kingdom A accelerators fail on algebraic convergence — BC succeeds

Empirically verified (2026-04-01) on ergodic averages of Lorenz system (chaotic attractor,
z-component, running mean over block-averaged samples, block length ≈ decorrelation time ≈ 5):

| Method | Convergence rate | Effect on ergodic mean |
|--------|-----------------|----------------------|
| Raw running mean | O(1/√N) algebraic | Baseline |
| Aitken Δ² | Assumes geometric r^n | **8× DEGRADATION** |
| Wynn's ε | Handles algebraic O(1/√N) | **3.9× improvement** |

**Root cause of Aitken failure**: Aitken's Δ² algorithm assumes the error sequence decays
geometrically (e_{n+1} ≈ r · e_n). Ergodic means of chaotic systems converge algebraically
at O(1/√N). Applying Aitken to algebraically-converging sequences destroys the convergence
structure and WORSENS the estimate.

**Why Wynn succeeds**: Wynn's ε computes a Padé approximant to the sequence of partial sums.
Padé approximants can represent both geometric AND algebraic decay — the internal rational
function structure is richer than Aitken's linear extrapolation. The BC taxonomy is the
structural reason: Wynn's non-commutative e-table builds the right rational model; Aitken's
commutative window cannot.

**Signal farm design rule** (from this result): For ergodic averages in the signal farm,
block-average to decorrelation time → apply Wynn ε. Never apply Aitken to running means.

---

## Why the BC Cell Was Invisible

The current taxonomy generated A, B, C by asking "how many passes?" and "what's the
operator?" separately, then combining informally. The three-letter taxonomy suggests
A, B, C are points on a line — a progression from "simpler" to "more complex."

The (ρ,σ) framing reveals they're corners of a square. The fourth corner isn't "more
complex than C" — it's ORTHOGONAL to C. Sequential iteration is a different kind of
complexity than commutative iteration.

**The reason nobody labeled BC**: most Kingdom B algorithms are naturally one-pass.
The EWM, Kalman filter, ARIMA — these are single-pass prefix scans. They have no
natural reason to iterate. The cases that DO iterate (Kalman smoother, BPTT) are
studied in specialized literature (sequential Monte Carlo, BPTT) without the general
framing.

---

## Parallelism Implications

The parallelizability table for the full (ρ,σ) space:

```
             σ=0                      σ=1
           ┌──────────────────────┬────────────────────────────┐
ρ=0 (comm) │   NC₁ / NC           │   NC₁ inner, P outer       │
           │   (parallel reduce)  │   (parallel iter, seq conv) │
           ├──────────────────────┼────────────────────────────┤
ρ=1 (seq)  │   NC (prefix scan)   │   NC inner, P outer        │
           │   O(log n) depth     │   (each pass is O(log n),   │
           │                      │    convergence sequential)   │
           └──────────────────────┴────────────────────────────┘
```

BC algorithms have the best-of-both-worlds parallelism:
- Each pass: O(log n) depth via prefix scan (Kingdom B parallelism)
- Across passes: convergence is inherently sequential but passes are individually fast

---

## The Galois Connection (from naturalist/math-researcher)

Math-researcher noted: A/B/C corresponds to abelian/solvable/general in Galois theory.

In the (ρ,σ) framing:
- ρ=0 (commutative) = abelian → parallelizable
- ρ=1 (non-commutative, still solvable in structure) = solvable → scan-parallelizable
- σ=1 adds iteration = need for convergence = "non-polynomial solvability"

The BC cell: non-commutative AND iterative. Still potentially solvable (Kalman smoother
has a closed-form solution!) but requires more structure to guarantee it.

---

## API Sketch

If `attract()` is a primitive for Kingdom C:

```rust
// Kingdom C: commutative iterate
fn attract<S>(data, f: (data, S) -> S, init: S, tol) -> S

// Kingdom BC: non-commutative iterate  
fn attract_scan<S>(data, forward: scan_fn, backward: scan_fn, init: S, tol) -> S
// OR:
fn attract<S>(data, f: () -> scan_op, init: S, tol) -> S
// where f returns a scan operation (Kingdom B inner loop)
```

The Kalman smoother example:
```
attract_scan(
    observations,
    forward  = kalman_filter_scan,
    backward = rts_smoother_scan,
    init     = prior,
    tol      = convergence_threshold
)
```

---

## The Third Axis: τ (Solution Uniqueness)

The (ρ,σ) square describes the computation. There's a separate property of the SOLUTION:
does the iteration converge to a unique answer?

τ = 0: unique fixed point (convergence is to THE answer regardless of initialization)
τ = 1: multiple attractors (initialization-sensitive, different runs may give different answers)

The (ρ,σ,τ) cube has 6 meaningful cells (τ=1 forces σ=1):

| ρ | σ | τ | Name | Examples |
|---|---|---|------|---------|
| 0 | 0 | 0 | Kingdom A | Sum, mean, COPA accumulate |
| 1 | 0 | 0 | Kingdom B | Kalman filter, EWM |
| 0 | 1 | 0 | Kingdom C, unique | Eigendecomp, IRLS/Huber |
| 0 | 1 | 1 | Kingdom C, multiple | K-means, ICA, t-SNE |
| 1 | 1 | 0 | Kingdom BC, unique | Wynn ε, RTS smoother |
| 1 | 1 | 1 | Kingdom BC, multiple | BPTT (non-convex) |

**Important separation**: the three-layer Fock boundary taxonomy describes the PROBLEM (which kingdom does the objective belong to? — INTRINSIC). The (ρ,σ,τ) taxonomy describes the ALGORITHM (how does the computation proceed? — CONTINGENT). A Kingdom A problem can require a Kingdom C algorithm (Galois obstruction → eigendecomposition iteration). The algorithm can be a worse fit than necessary, or a tight fit.

**Two phase boundaries in the objective's algebraic hierarchy** (math-researcher):

1. **Degree 2→3 boundary**: determines σ. Below it: ∇²f constant → closed-form → σ=0. Above it: ∇²f depends on solution → iterate → σ=1.

2. **Convexity boundary** (within degree ≥ 3): determines τ. Convex objective (∇²f ≥ 0 everywhere) → unique global minimum → τ=0. Non-convex (∇²f changes sign) → multiple local minima → τ=1.

These are orthogonal. Both are intrinsic to the objective. Degree-3 objectives CAN be convex (logistic regression → τ=0). Neural networks are degree-3+ AND non-convex → τ=1.

**The engineering criterion**: choose the algorithm with the tightest (ρ,σ,τ) for the problem's classification. A d=3 eigendecomposition is a σ=0, τ=0 problem — Cardano's formula is the tight algorithm; Jacobi iteration is correct but uses σ=1 unnecessarily.

**The COPA architecture goal**: for Kingdom A problems (degree ≤ 2), find a (0,0,0) algorithm — commutative, one-pass, unique. That's what COPA does.

---

## Open Questions

1. Is the BC cell genuinely distinct from C + B applied separately? Or is `attract(scan)` 
   just a composition pattern with no new structure?

2. Wynn ε is (1,1,0) — BC with unique answer. Is there a useful sub-taxonomy of τ=0 
   Kingdom BC? (When is convergence guaranteed? When is it not but typically happens?)

3. Does the 2×2 table generalize to 3×3 with ρ ∈ {0, 1, 2}? What would ρ=2 be?
   (Perhaps: commutative, non-commutative, non-associative?)

4. **Strange attractors in sequential systems** (resolved): τ does NOT need a third value.
   τ (uniqueness) and α (attractor type) are ORTHOGONAL. τ=0/1 measures whether the answer
   is unique; α=0/1/2 measures what KIND of attractor you're converging to:

   | α | Attractor type   | `attract()` return type | compile_budget mode |
   |---|-----------------|------------------------|---------------------|
   | 0 | Fixed point     | `FixedPoint(θ*)`       | Convergence(tol)    |
   | 1 | Limit cycle     | `Trajectory([T steps])`| Budget(T)           |
   | 2 | Strange/chaotic | `Trajectory(long)`     | Budget(N), N >> T   |

   A unique limit cycle is (α=1, τ=0). Multiple strange attractors would be (α=2, τ=1).
   The full classification is (ρ, σ, τ, α). See garden: `20260401-what-attract-returns.md`.

---

## Related Campsites
- [scatter-attract-duality](../20260401-scatter-attract-duality/) — attract() as primitive

## Implementation
- `src/series_accel.rs` — 10 functions + StreamingWynn struct, 45 tests (session 3 update)
  - Partial sums, Cesàro, Aitken Δ², Wynn ε, Richardson, Euler, Abel, Richardson-on-partial-sums, accelerate(), Euler-Maclaurin ζ(s)
  - **StreamingWynn**: first streaming Kingdom BC primitive — incremental tableau,
    term-at-a-time, `attract(wynn_step)` pattern. Converges at term 15 on Leibniz (tol=1e-10).
  - Wynn ε achieves machine precision (3.33e-16) from 20 Leibniz terms
  - Richardson 82,231× on Basel (auto-detected error order p=1)
  - Abel sums divergent series (Grandi → 1/2, Σ(-1)^n(n+1) → 1/4)
  - Composition experiments: Cesàro∘Wynn = 10¹³× WORSE (destroys structure Wynn needs)
  - Ergodic convergence test (Lorenz z-mean): Wynn 3.9× improvement, Aitken 8× degradation
  - Module-level doc: Kingdom BC annotation, kernel taxonomy, cross-cadence connection
