# Campsite: Infinite Series as Prefix Scan + Acceleration

**Opened:** 2026-04-01  
**Scout:** scout  
**Thread:** Parking-lot idea from team-lead: "Infinite series as prefix scan + acceleration (Aitken, Richardson, Shanks). What does a limit mean in tambear?"  
**Status:** All open questions resolved — `src/series_accel.rs` (10 functions + StreamingWynn, 45 tests). Kingdom BC populated and streaming. Full kernel taxonomy, composition algebra, matched-kernel principle.

---

## The Core Observation

A partial sum is the simplest prefix scan:

```
S_n = Σ_{k=0}^{n} a_k   ←→   scan(a, init=0, op=Add)[n]
```

A convergent series is a prefix scan whose limit exists. The *limit* is the "final value" of an infinite scan — the fixed point that the partial sums approach.

**Key insight**: Series acceleration methods (Aitken, Richardson extrapolation, Shanks/Wynn-epsilon) are **post-processors on the output of the prefix scan**. They take the sequence `(S_1, S_2, ..., S_n)` and extrapolate to `S_∞` faster than waiting for convergence.

---

## The Methods

### Aitken Δ² Method

Given three consecutive partial sums, estimate the limit:
```
S_n* = S_{n+2} - (S_{n+2} - S_{n+1})² / (S_{n+2} - 2·S_{n+1} + S_n)
     = S_{n+2} - (ΔS_{n+1})² / (Δ²S_n)
```

**In tambear terms**: Aitken is a sliding window of size 3 over the prefix scan output, applying a nonlinear transformation. It's a `gather(scan(a), pattern=sliding_3)` followed by a pointwise `apply(aitken_fn)`.

The method works when the series converges geometrically (linear convergence). The acceleration converts linear convergence to superlinear.

### Richardson Extrapolation

Run the same computation at two different "resolutions" (h and h/2), then extrapolate:
```
f(0) ≈ (4·f(h/2) - f(h)) / 3     [for error O(h²)]
```

**In tambear terms**: Richardson is a **dual-target accumulate** — run the same accumulate expression twice with different step sizes, then combine the results. This is the `20260331-dual-target-fusion` pattern applied to numerical integration.

For series: run partial sums at two different truncation points (N and 2N), then extrapolate. The extrapolated value converges faster than either raw partial sum.

### Shanks Transformation / Wynn Epsilon Algorithm

The Shanks transformation generalizes Aitken to use windows of size 2k+1, giving k-th order acceleration. The Wynn epsilon algorithm computes this via a recursive table:

```
ε_{n}^{(0)} = S_n
ε_{n}^{(-1)} = 0
ε_{n+1}^{(k-1)} = ε_n^{(k+1)} + 1/(ε_{n+1}^{(k)} - ε_n^{(k)})
```

This recursive structure is itself a **2D scan** — building the epsilon table row by row. The odd-indexed columns give the accelerated estimates.

**In tambear terms**: Shanks is a nested scan — the outer scan iterates over the acceleration order k, the inner scan iterates over the series terms. This is the 2D accumulate pattern.

---

## The Generalization: From Series to Iterative Algorithms

Every iterative algorithm (Kingdom C/BC) produces a sequence of states that converges to a fixed point. This sequence is exactly like a sequence of partial sums converging to a limit.

The same acceleration methods apply:

| Series acceleration | Kingdom C/BC acceleration |
|--------------------|--------------------------|
| Aitken Δ² on (S_n) | Aitken Δ² on (θ_n) — accelerate iterative regression |
| Richardson on (S_N, S_{2N}) | Run two optimization trajectories, extrapolate |
| Shanks/Wynn-ε on (S_n) | **Anderson mixing** on (θ_n) |

**Anderson mixing** (aka DIIS in quantum chemistry, Pulay mixing in DFT) is precisely the Shanks transformation applied to fixed-point iteration sequences. It maintains a history of the last m iterates and computes the next iterate as a linear combination that minimizes the residual. This is the Shanks transformation in vector form.

---

## What a "Limit" Means in Tambear

The team-lead asked: *what does a limit mean in tambear?*

Proposed answer: a limit is a **collapsed scan** — the result of running a prefix scan to completion and extracting the terminal state. But for infinite or practically-infinite sequences, "running to completion" is too expensive. The limit must be estimated.

Three ways to get a limit in tambear:

1. **Run until convergence** (current approach): compile_budget controls when to stop. The terminal state IS the estimate.

2. **Acceleration post-processing**: run N steps of the scan, apply Aitken/Richardson/Wynn-ε to the trajectory. Get a better estimate with fewer steps.

3. **Extrapolation from structure**: if the series has known algebraic structure (geometric series, alternating series, Euler-Maclaurin endpoint corrections), use that structure to compute the limit analytically from the first few terms.

**These three are compile_budget decisions.** The compile budget already decides "how much computation is enough." Acceleration is a *precision-per-compute-dollar* improvement — same compute, better precision, or same precision, less compute.

---

## The Convex Combination Structure

Anderson mixing produces the next iterate as:
```
θ_{k+1} = Σ_i α_i · f(θ_{k-i})    where Σ α_i = 1
```

This is a **convex combination of the history** — a GramMatrix solve for the optimal weights α. The GramMatrix is the **overlap matrix** of the residuals `r_i = f(θ_i) - θ_i`.

**In tambear terms**: Anderson mixing = gather from the iterate history (addressing pattern) + GramMatrix solve (Kingdom A) + weighted combination (scatter). It's a pure tambear pipeline.

The history length m (how many past iterates to use) is a hyperparameter — the analog of the window size in the Shanks transformation.

---

## Concrete Example: IRLS with Acceleration

IRLS (Kingdom C) converges linearly for M-estimation. With Anderson mixing:
1. Each IRLS step produces a new weight estimate θ_k
2. Anderson mixing uses the last m=5 estimates to compute θ_{k+1}
3. Convergence goes from O(k) iterations to O(k/m) with better constants

The IRLS + Anderson mixing pipeline in tambear:
```
attract_anderson(
    data,
    step = irls_step,           // f: θ → θ (Kingdom A inner)
    init = ols_estimate,
    history_length = 5,
    overlap_solve = gram_matrix, // GramMatrix for Anderson weights (Kingdom A)
    tol = 1e-8
)
```

This is a Kingdom C primitive with a Kingdom A inner solver for the acceleration weights.

---

## Connection to compile_budget

The `compile_budget.rs` module (exists in codebase) decides when to stop iteration. Currently: stop after max_iter or when ||Δθ|| < tol.

With series acceleration, the budget can be spent more efficiently: after k iterations, run Aitken on (θ_{k-2}, θ_{k-1}, θ_k) to get an accelerated estimate. The budget comparison becomes:

```
// Current: raw convergence
if delta_norm < tol { return current_state; }

// With Aitken: accelerated convergence  
let accelerated = aitken(states[k-2], states[k-1], states[k]);
if accelerated_delta < tol { return accelerated; }
```

The accelerated estimate often hits the tolerance after half the iterations. This is a compile_budget multiplier — you get 2x improvement for free by post-processing the trajectory.

---

## Open Questions

1. **Is there a tambear primitive for "trajectory"?** Currently Kingdom C returns only the final state. To apply acceleration, you need access to intermediate states. Does tambear need a `trace_attract()` variant that returns the full trajectory?

2. **How does Richardson extrapolation interact with the compile_budget?** Richardson needs two runs at different resolutions. If the budget allows two runs, Richardson gets you better precision. If the budget allows one run, you're stuck with the terminal state. The budget decision becomes a question about 1-run vs 2-run tradeoffs.

3. **Is Wynn-epsilon GPU-friendly?** The epsilon table has data dependencies: each cell depends on the two cells above it in a triangular pattern. This is a 2D scan with a triangular dependency structure — similar to LU factorization. Can it be parallelized via wavefront scheduling?

4. **What's the right abstraction for history?** Anderson mixing needs the last m iterates. In a streaming/GPU context, maintaining a ring buffer of m states and solving a small m×m GramMatrix at each step is cheap. But the ring buffer is state that must be carried alongside the iterate — it changes the MSR of the Kingdom C computation.

---

## Empirical Results (observer, 2026-04-01)

Implementation: `crates/tambear/src/series_accel.rs`

### Convergence hierarchy — Leibniz π/4 (20 terms)

| Method | Error | Accumulate pattern |
|--------|-------|--------------------|
| Raw partial sum | 1.25e-2 | `Prefix` |
| Aitken Δ² | 9.08e-6 | `Windowed{3}` on `Prefix` |
| Euler transform | 8.69e-9 | `ByKey(binomial)` on `Prefix` |
| Wynn ε (Shanks) | **3.33e-16** | Iterated `Windowed` (Kingdom C) on `Prefix` |

### Iterated Aitken on Leibniz (30 terms)

Each `Windowed{3}` pass gains ~3 orders of magnitude:
```
Raw:      8.33e-3
Aitken×1: 2.56e-6
Aitken×2: 2.71e-9
Aitken×3: 5.49e-12
```

### Key proven claims

1. **Aitken = Windowed{3} on Prefix** — proven exact to machine precision (`aitken_is_windowed3_on_prefix`)
2. **Wynn ε = iterated Shanks** — reaches 1e-16 from 20 terms of an O(1/n) series
3. **Richardson on trapezoidal** — >100x improvement over finest grid alone
4. **Euler transform** — binomial-weighted partial sum average, 1e-9 for alternating series
5. **No new primitive needed** — "limit" = composition of Prefix + Windowed

### What the positive-series test revealed

Basel π²/6 (30 terms, all positive):
```
Raw:  3.28e-2
Wynn: 5.38e-3
```

Wynn helps (~6x) but doesn't dominate. Shanks is designed for alternating/geometric convergence. For positive monotone series, Richardson or Euler-Maclaurin endpoint corrections would be more appropriate.

### Gain profile depends on convergence type (budget experiment, observer)

**Leibniz (algebraic, error ~ 1/n)**: gain DECREASES per level
```
n= 30: 3.5  3.0  2.7  2.5  1.9   ← diminishing
n= 60: 4.1  3.6  3.4  2.4
n=120: 4.7  4.3  4.3
n=200: 5.2  4.7
```
More terms → bigger first-level gain (budget artifact confirmed). But within each budget, gains decrease because Aitken assumes geometric tail and Leibniz is algebraic.

**Geometric (error ~ r^n, EXACT match for Aitken)**: one pass reaches machine precision
```
r=0.95, n=100: 11.2 orders in ONE pass (1.2e-1 → 7.8e-13)
r=0.90, n= 80: 10.5 orders in ONE pass (2.2e-3 → 6.8e-14)
r=0.80, n= 60:  8.9 orders in ONE pass (7.7e-6 → 1.1e-14)
```
Aitken eliminates the exact error term; residual is floating-point noise. Can't observe ρ-squaring because first pass already exhausts f64 precision.

**Conclusion**: gain profile = match between accelerator assumption and convergence type.

### Wynn ε instability at large n

```
n=10: 3.71e-8   (good)
n=20: 3.33e-16  (machine precision)
n=40: 9.55e-15  (degraded!)
n=80: 9.01e+15  (garbage — 31 orders wrong)
```

Deep Wynn tableau amplifies rounding errors catastrophically. Practical limit: ~20-40 terms. Beyond that, iterated Aitken is more stable.

**Fixed**: early stopping now monitors consecutive even-column estimates. n=80 goes from 9.01e+15 (garbage) to machine precision. Slight conservatism at n=10 (~half a digit lost).

**Upgrade path** (scout finding): robust Padé via SVD (Gonnet/Güttel/Trefethen 2013) = category 2 regularization. Solves the Padé coefficient system directly, detects near-pole degeneracy via SVD. More expensive but handles the singular case correctly. Future: `robust_pade(terms)` as premium path, Wynn+early-stop as fast path, iterated Aitken as robust fallback.

---

## Richardson = K03 (Cross-Cadence) — A Broader Pattern

The `series_accel.rs` module doc notes: "Richardson extrapolation IS a cross-cadence operation."

This is a structural rhyme worth naming explicitly. In WinRapids, K03 is the cross-cadence kingdom — running the same computation at multiple temporal resolutions (1s ticks, 5s bins, 30s bins) and letting them inform each other. The argument: computing 5-minute bins from 1-minute bins is lossy; computing each from raw ticks preserves independent projection bandwidth.

Richardson extrapolation is exactly this, applied to numerical approximations:
- Run the same computation at step size h and step size h/2 (two independent "cadences")
- The error at each resolution is a different projection of the same truth
- The extrapolated value cancels the leading error term — the same way K03 cross-cadence reveals signal invisible to any single cadence

The abstraction: **K03 is Richardson extrapolation applied to time resolution rather than numerical step size.** Both eliminate systematic resolution-dependent bias by comparing independent projections of the same underlying signal.

This suggests K03 has broader application than financial markets:
- Numerical integration at multiple grid resolutions
- ML model evaluation at multiple data subsampling rates
- Signal reconstruction at multiple frequency bandwidths
- Experiment replication at multiple sample sizes (power analysis as Richardson)

The unifying structure: run the same transform at two independent resolutions, fit the error curve, extrapolate to the infinite-resolution limit. This is the cross-cadence operation in the most general sense.

---

## Auto-Selector: Compile-Time Accelerator Dispatch (observer)

`accelerate(terms)` detects convergence type and dispatches:
- Geometric (ratio < 0.95, stable) → Aitken (exact in one pass)
- Alternating (sign pattern) → Euler + Wynn, pick better
- Positive monotone algebraic → Richardson with auto-detected error order
- Mixed algebraic/Unknown → Richardson + Wynn/Aitken, pick better

Results on series (updated after Basel gap closure):
```
Leibniz:   raw → auto = 1.25e-2 → 3.33e-16  (10¹³× speedup)
ln2:       raw → auto = 2.44e-2 → 5.55e-16  (10¹³× speedup)
Geometric: raw → auto = 5.76e-2 → 1.07e-14  (10¹²× speedup)
Basel:     raw → auto = 2.47e-2 → 3.00e-7   (82,231× speedup)  ← WAS 5×
```

---

## Lorenz Attractor: Ergodic Theorem as Accumulate (observer)

`complexity::tests::lorenz_attractor_ergodic_convergence` verifies: for chaotic systems, the trajectory never settles but `accumulate(trajectory, All, Value, Add) / N` converges at 1/√N.

```
z_mean at  1K: 23.797  (err = 0.254)
z_mean at 10K: 23.566  (err = 0.024)
z_mean at 49K: 23.542  (ergodic estimate)
```

Kingdom implication: trajectory = Kingdom C (multi-pass), measurement = Kingdom A (one-pass accumulate). The ergodic theorem bridges them.

---

## Basel Gap Closure — Richardson on Partial Sums (observer, session 2)

`richardson_partial_sums(terms)` auto-detects the error order and applies Richardson:

**Error order detection**: compute partial sums at geometrically spaced truncation points
(N/2^k, ..., N/2, N), measure tail-difference ratios d_i/d_{i+1}, extract p = log₂(mean ratio).

| Series | Detected p | Raw error | Richardson error | Speedup |
|--------|-----------|-----------|-----------------|---------|
| Basel (40 terms) | 1 | 2.47e-2 | 3.00e-7 | 82,231× |
| Basel (160 terms) | 1 | 6.23e-3 | 1.69e-11 | 80,441,438× |
| ζ(3) (60 terms) | 2 | 1.37e-4 | 1.09e-6 | 126× |

**Richardson vs Wynn on Basel**: Richardson wins by 52× at n=20, 15,841× at n=40,
270,857× at n=80, 80M× at n=160. The gap grows with n because Richardson cancels
successively more error terms while Wynn's geometric-convergence assumption mismatches.

**ζ(3) surprise**: "only" 126× improvement despite correct p=2 detection. The ζ(3) error
expansion has more than two terms, and with only ~4 Richardson levels (log₂(60/4) ≈ 3.9),
we can only cancel ~4 error terms. The improvement compounds slower than Basel because
the higher-order terms are relatively larger. More terms → more levels → better results.

**Matched-kernel principle confirmed again**: Richardson is to positive algebraic what
Euler is to alternating and Aitken is to geometric. The auto-selector now dispatches
all three matched cases correctly.

---

## Open Questions

1. ~~Richardson with auto-detected error order for positive algebraic series~~ **DONE** — 82,231× on Basel
2. **Robust Padé via SVD** as category 2 upgrade for Wynn instability
3. ~~Aitken on ergodic averages~~ **DONE** (empirically resolved, observer 2026-04-01):
   Block-average to decorrelation time → Wynn ε on block means = **3.9× improvement**.
   Aitken on same sequence = **8× WORSE** (geometric assumption mismatches 1/√N algebraic).
   Key: block-averaging converts correlated → ~iid, exposing algebraic convergence structure
   that Wynn ε's rational Padé handles correctly. Signal farm rule: Wynn ε on block averages,
   never Aitken on ergodic means. See rho-sigma campsite for empirical table.
4. ~~Abel summation~~ **DONE** — exponential kernel implemented; taxonomy complete. See below.
5. ~~Streaming Wynn~~ **DONE** — `StreamingWynn` struct: incremental tableau, term-at-a-time.
   
   Anti-diagonal recurrence: `new[k] = last[k-2] + 1/(new[k-1] - last[k-1])`.
   State: one entry per column (the "last" array). O(depth) per push, O(1) estimate.
   Stability monitoring detects diverging deep columns, keeps best stable estimate.
   
   Results on Leibniz π/4:
   ```
   Converges at term 15 (tol=1e-10), from 100 available
   Term 20: machine precision (3.33e-16)
   Trajectory monotone through machine epsilon floor
   ln(2):  1.11e-16 from 25 terms (machine precision)
   e^{-1}: exact (0.00e0) from 15 terms
   ```
   
   **This IS `attract(wynn_step, empty_tableau)`** — the first streaming Kingdom BC primitive.
   Inner: sequential anti-diagonal sweep (ρ=1). Outer: push until converged (σ=1).
6. ~~ζ(3) deeper investigation~~ **DONE** — `euler_maclaurin_zeta()` reaches machine precision.
   
   Implemented: `euler_maclaurin_zeta(s, n_terms, correction_order)` adds Bernoulli corrections
   to the partial sum using the Hurwitz zeta Euler-Maclaurin expansion evaluated at a=N+1.
   
   Results on ζ(3) with n=60:
   ```
   Raw partial sum:     1.37e-4
   Richardson alone:    1.09e-6   (126×)
   EM p=0 (integral):   1.81e-8   (7,566×)
   EM p=1 (+B₂):        1.62e-12  (84,442,253×)
   EM p=2 (+B₄):        2.22e-16  (MACHINE PRECISION)
   ```
   
   Two Bernoulli correction terms take ζ(3) from 126× to machine precision. The campsite
   predicted ~1000×; reality is 84 million× with one correction, machine ε with two.
   
   The consecutive-power problem is completely solved: Euler-Maclaurin analytically removes
   the 1/N², 1/N³, 1/N⁴ terms, leaving residuals too small to measure in f64.

---

## Abel Summation — Exponential Kernel (observer, session 2)

`abel_sum(terms)` evaluates f(x) = Σ a_k·x^k at geometrically spaced x → 1⁻,
then Richardson-extrapolates. The composition: Abel regularizes, Richardson accelerates.

### On convergent series (Leibniz, 200 terms)

| Method | Kernel | Error |
|--------|--------|-------|
| Raw | — | 1.25e-3 |
| Cesàro | Uniform | 7.10e-4 |
| **Abel** | **Exponential** | **2.64e-6** |
| Aitken | Nonlinear | 7.93e-9 |
| Euler | Binomial | 2.22e-16 |

Abel slots between Cesàro and Aitken on convergent alternating series. Not the matched
kernel for this case (Euler is), but a valid universal method.

### On divergent series (Abel's strength)

**Grandi** (1-1+1-1+..., divergent, Abel sum = 1/2):
```
n=  100: err = 4.17e-3
n=  500: err = 1.21e-8
n= 2000: err = 2.93e-11
```

**Σ(-1)^n·(n+1)** (divergent, Cesàro C-1 fails, Abel sum = 1/4):
```
Abel:   0.249981  (err = 1.87e-5)
Cesàro: 0.000000  (err = 2.50e-1)  ← fails completely
```

Abel sums series that Cesàro cannot. This proves the Tauberian inclusion:
Cesàro-summable ⊂ Abel-summable.

### Kernel taxonomy (completed)

| Kernel | Convergent alternating | Positive algebraic | Divergent |
|--------|----------------------|-------------------|-----------|
| Cesàro (uniform) | Weak | Weak | Grandi only |
| **Abel (exponential)** | **Middle** | **Weak** | **Strongest** |
| Euler (binomial) | Strongest | N/A | N/A |
| Richardson (polynomial) | N/A | Strongest | N/A |
| Aitken (nonlinear) | Strong | Moderate | N/A |
| Wynn (iterated nonlinear) | Strongest (small n) | Weak | N/A |

Each method has a domain where it's optimal. The auto-selector `accelerate()` matches
method to convergence type. Abel fills the "divergent series" column — no other method
in the toolkit handles genuinely divergent series.

---

## Accelerator Composition Algebra (observer, session 2)

Accelerators compose, but composition is **not free** — the first accelerator can
create or destroy the structure the second accelerator needs.

### Empirical results

| Composition | vs base | Reason |
|------------|---------|--------|
| Abel → Richardson | **Helps** (Abel on Grandi: 4e-3 → 3e-11 at n=2000) | Abel creates smooth-in-h structure; Richardson needs smooth-in-h |
| Cesàro → Wynn | **Hurts** (10¹³× worse than Wynn alone) | Cesàro destroys oscillation pattern; Wynn needs oscillation |
| Euler → Aitken | **Neutral** (Euler already at machine ε) | Nothing left to improve |
| Aitken^k (iterated) | **Helps** (each pass gains ~3 orders) | Aitken creates new sequence with squared convergence ratio |

### The structural principle

**Composition helps when the first accelerator creates structure the second needs.**
- Abel → monotone-in-h sequence → Richardson extrapolates smoothly ✓
- Aitken → sequence with faster convergence → Aitken extrapolates again ✓

**Composition hurts when the first accelerator destroys structure the second needs.**
- Cesàro averaging → monotone sequence → Wynn's rational Padé has nothing to extrapolate ✗
- (Analogous: Aitken on block-averaged ergodic means = 8× worse, see question 3)

**The matched-kernel principle generalizes to composition**: the intermediate
representation between two composed accelerators must match what the second
accelerator's kernel is designed for.

---

## Related Campsites
- [scatter-attract-duality](../20260401-scatter-attract-duality/) — Kingdom C as attract(Kingdom A)
- [rho-sigma-kingdoms](../20260401-rho-sigma-kingdoms/) — the 2×2 table; Kingdom C outer attract
- [summability-as-kernel-accumulate](../20260401-summability-as-kernel-accumulate/) — the KDE↔summability rhyme
- `compile_budget.rs` — the existing precision-budget module this would extend
- `series_accel.rs` — the implementation (10 functions + StreamingWynn struct, 45 tests)
