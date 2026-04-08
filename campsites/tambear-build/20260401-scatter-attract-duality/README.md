# Campsite: Scatter vs Attractor Duality — `tb.attract()` as a Primitive

**Opened:** 2026-04-01  
**Thread:** Parking-lot idea from team-lead  
**Status:** Active exploration

---

## The Core Observation

Kingdom A scatters: it distributes n data elements to k accumulator cells and returns.
Kingdom C converges: it iterates Kingdom A until a fixed point — an attractor.

These are not two kinds of algebraic operators. They are two different roles in a computation:
- **accumulate**: one data-touching pass (Kingdom A or B depending on operator commutativity)
- **attract**: runs an inner accumulate repeatedly until convergence

**Proposed decomposition:**

```
Kingdom A = accumulate(data, K=A_op)          # single commutative pass
Kingdom B = scan(data, K=B_op)                # single non-commutative pass
Kingdom C = attract(accumulate(K=A_op), init) # multi-pass until fixed point
```

Every Kingdom C algorithm we've seen is `attract(one_K_A_pass, initial_state)`:
- K-means: `attract(assign_then_recompute_centroids, random_centroids)`
- IRLS: `attract(weighted_least_squares, OLS_init)`
- EM/GMM: `attract(E_step + M_step, init_params)`
- PageRank: `attract(rank_scatter, uniform)`
- Newton: `attract(gradient_update, init_params)`

---

## `tb.attract()` as a Primitive

```
attract(f: State → State, init: State, tol: f64) → State
```

Where `f` is itself an accumulate call. The convergence criterion is determined by the
Lipschitz constant of `f` — which is exactly the "contraction constant" (ρ) that
math-researcher identified as the Kingdom C parameter.

The primitive would:
1. Initialize state
2. Run one data pass via `f`
3. Measure change in state
4. If `|Δstate| < tol`: return; else: goto 2

This is already what every IRLS/EM/K-means loop does. Naming it makes it explicit.

---

## Why This Clarifies Kingdom C

The current taxonomy says "Kingdom C = iterative." But that's a description of *behavior*,
not *algebra*. The algebra is still Kingdom A (the inner pass is commutative). The "C-ness"
comes entirely from the `attract()` wrapper, not from the accumulate itself.

**Implication**: Kingdom C is not a different algebraic kind — it's `attract(Kingdom A)`.
The three kingdoms then become:
- Kingdom A: direct accumulate (commutative)
- Kingdom B: direct scan (non-commutative, but single pass)
- Kingdom C: attract(accumulate) (commutative inner, convergent outer)

And this reveals an empty cell: what is `attract(scan)`? A sequential process that also
requires multiple passes to converge. This is the BC cell — see the (ρ,σ) campsite.

---

## The Attractor Geometry

Every Kingdom C algorithm has an attractor in parameter space. The attractor IS the answer.
- K-means attractor: the locally optimal centroid configuration
- EM attractor: the local maximum of the likelihood
- IRLS attractor: the M-estimate (fixed point of the weighted LS map)
- PageRank attractor: the stationary distribution of the random walk

The geometry of attractors in parameter space IS the geometry of the solution space.
Multiple attractors = multiple local optima = initialization sensitivity.
Single attractor = global convergence guarantee = unique solution.

The convergence guarantee is algebraic: if `f` is a contraction mapping (Lipschitz < 1),
there's exactly one fixed point by the Banach fixed-point theorem.

**IRLS convergence**: `ρ = 0` for linear objectives (OLS: one step). `0 < ρ < 1` for
convex nonlinear (logistic, Huber). `ρ ≥ 1` potentially for non-convex (bisquare, EM).

---

## Connection to the MSR Principle

The attractor IS the MSR of Kingdom C: it's the minimum state from which the full answer
can be extracted. The IRLS fixed point is sufficient — you don't need the trajectory.

The Fock boundary for Kingdom C: at each iteration, the running state (μ, scale for IRLS)
is a collapsed representation of all data seen so far given the current estimate. The Fock
boundary density is 1 collapse per iteration, not 1 collapse total.

---

## Open Questions

1. Can `attract()` be expressed as a higher-order `scan()` over the iteration space?
   (Sequence of states s₀, s₁, s₂, ... is a scan where each element is one data pass)

2. Is there a "parallel attract" — run multiple starting points simultaneously, keep the
   best attractor? This is exactly random restarts in K-means / simulated annealing.

3. What's the right API? (updated after two-kinds-of-attractor + what-attract-returns)

   ```rust
   enum Termination {
       Convergence { tol: f64, max_iter: usize },  // point attractor (α=0)
       Budget { n: usize },                          // cycle or strange (α=1,2)
       AutoDetect { budget: usize },                 // optimistic-monitor-fallback
   }

   enum AttractionResult<S> {
       FixedPoint(S),       // convergence fired: α=0
       Trajectory(Vec<S>),  // budget mode: α=1 (cycle) or α=2 (strange)
   }

   fn attract<S>(
       f: impl Fn(S) -> S,
       init: S,
       termination: Termination,
   ) -> AttractionResult<S>
   ```

   `AutoDetect` is optimistic-monitor-fallback: try Convergence first, watch for
   period-T orbits (monitor), declare α=2 if budget exhausted with no convergence.
   The compile_budget decision IS the hypothesis about attractor type.
   See garden: `20260401-what-attract-returns.md`

4. Does `attract()` compose? `attract(attract(f, init1), init2)` — nested attractors.
   This might describe hierarchical clustering or nested EM.

---

## Strange Attractors: Geometric vs Statistical (observer + scout, 2026-04-01)

The `attract()` primitive has a dual meaning depending on attractor type:

| Attractor type | Kingdom C (`attract`) | Kingdom A (`accumulate`) |
|---|---|---|
| **Point** (fixed point) | Converges to the point | Degenerate measure (all mass at one point) |
| **Limit cycle** | Converges to the cycle | Uniform measure on the cycle |
| **Strange** (chaotic) | Never converges; trajectory wanders forever | Converges to the SRB measure at 1/√N |

**The nesting**: `accumulate(attract(f, init), All, Mean)` — Kingdom C generates the trajectory (explores the geometric attractor), Kingdom A measures it (computes the statistical attractor / SRB measure). The ergodic theorem guarantees convergence of the measurement regardless of where the trajectory wanders.

**Empirical verification**: `complexity::tests::lorenz_attractor_ergodic_convergence` — Lorenz system (σ=10, ρ=28, β=8/3), 50K steps. z_mean converges to 23.54 at rate 1/√N even though the trajectory oscillates chaotically between -20 and +20 in x.

**Series acceleration connection**: the systematic bias in ergodic averages (finite-time decorrelation) has polynomial structure → Richardson extrapolation applies. Random fluctuations (1/√N) are noise no accelerator can help with. At small N, systematic bias dominates → Richardson worthwhile. At large N, random fluctuation dominates → diminishing returns. See `series-acceleration-prefix-scan` campsite open question #3.

**Architectural implication**: `attract()` is not just for finding fixed points — it's a trajectory generator. The full pattern is Kingdom C (generate) + Kingdom A (measure). This means Kingdom C algorithms always have a Kingdom A "observation layer" that extracts the answer from the trajectory.

---

## Related Campsites
- [rho-sigma-kingdoms](../20260401-rho-sigma-kingdoms/) — the 2×2 table where attract(scan) lives
- [series-acceleration-prefix-scan](../20260401-series-acceleration-prefix-scan/) — ergodic acceleration open question
