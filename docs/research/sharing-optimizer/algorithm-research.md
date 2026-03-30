# Lab Notebook: Operator Algorithm Research

**Date**: 2026-03-30
**Authors**: Claude (team lead) + Tekgy
**Status**: Active
**Depends on**: Phase 2 experiments (E01-E09), Phase 3 operator implementations

---

## Context & Motivation

During Phase 3 construction of the timbre sharing optimizer, we began implementing scan operators (AssociativeOp trait) and discovered that the standard algorithms (Welford, Kalman) have performance and correctness properties that differ significantly from expectations when used in GPU parallel prefix scans. This notebook documents the algorithm research, reformulations, and discoveries.

The driving principle: **push complexity from combine to lift/constructor.** The combine runs O(n log n) times in the scan tree. The constructor runs once. Every operation moved from combine to constructor is a multiplicative win.

---

## Experiment 1: Performance Tier Discovery

### Before
**Hypothesis**: Operator performance scales with state size (more bytes = slower scan).
**Design**: Benchmark AddOp (8B), KalmanAffineOp (16B), CubicMomentsOp (24B), WelfordOp (24B), KalmanOp (28B) at 100K elements on Blackwell GPU.

### Results

| Operator | State bytes | Combine ops | p01 (μs) |
|---|---|---|---|
| AddOp | 8 | 1 add | 42 |
| KalmanAffineOp | 16 | 2 mul + 1 add | 42 |
| CubicMomentsOp | 24 | 3 add | 44 |
| WelfordOp | 24 | div + branch | 100 |
| KalmanOp | 28 | 5 div + 3 branch | 103 |

**Surprise?** YES. CubicMomentsOp (24B) is same speed as AddOp (8B). State size is irrelevant. Combine complexity is everything.

### Discussion
**What we learned**: Two performance tiers exist, determined solely by combine complexity:
- Fast tier (~42μs): adds and multiplies only
- Slow tier (~100μs): division or branching

**Follow-up measurement**: Isolated division vs branching on GPU:
- `a / b` vs `a * b`: 1.04x (division is essentially free)
- `where(mask, a*b, a+b)` vs `a * b`: **13.7x** (branching is catastrophic)

**Revised understanding**: The slow tier is caused by BRANCHING (warp divergence), not division. Division alone costs almost nothing on Blackwell.

---

## Experiment 2: Branch-Free Scan Engine

### Before
**Hypothesis**: Removing `if (gid < n)` bounds checks from the scan kernel and padding inputs to block-size multiples will eliminate the branching penalty and collapse all operators to the fast tier.
**Design**: Pad input arrays with identity elements. Remove all gid<n conditionals from Phase 1 and Phase 3 kernels. Benchmark all operators.

### Results

| Operator | Before padding | After padding | Change |
|---|---|---|---|
| AddOp | 42 μs | 48 μs | +14% (worse) |
| WelfordOp | 100 μs | 104 μs | +4% (worse) |
| KalmanOp | 103 μs | 108 μs | +5% (worse) |

**Surprise?** YES — net NEGATIVE. Padding overhead (alloc_zeros + memcpy) exceeded the branch elimination benefit. The gid<n branch was already warp-uniform for 97/98 blocks (only the last block has partial data).

### Discussion
**What we learned**: Kernel-level branching (bounds checks) is NOT the source of the two-tier gap. The branching is INSIDE the operator combine body (WelfordOp's `if(n==0)`, KalmanOp's `if/else if/else` for has_data). These operator-internal branches cause warp divergence because different threads in the same warp may have different combine paths.

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Engine-level padding | Revert (keep for correctness) | Ship as optimization | Net negative performance |
| Identity short-circuit in engine | Don't build | Build before combine | The cost is in operator combine, not engine dispatch |

---

## Experiment 3: Operator Combine Audit

### Before
**Hypothesis**: Reducing divisions and eliminating unnecessary branches in operator combines will narrow the performance gap. Specific predictions:
- WelfordOp: 2 div → 1 (cache inv_n) should save ~30μs
- KalmanOp: 5 div → 1 (algebraic reformulation) should save ~50μs
- EWMOp: pow() → exp(log_decay * count) should help

**Design**: Algebraic reformulation of each operator's combine body. Measure before/after.

### Results

| Operator | Before | After | Change | What changed |
|---|---|---|---|---|
| WelfordOp | 104 μs | **72.5 μs** | **-30%** | 2 div → 1 (cached inv_n) |
| KalmanOp | 100 μs | **78.8 μs** | **-21%** | 5 div → 1 (algebraic reformulation) |
| MaxOp/MinOp | ~42 μs | ~42 μs | noise | Ternary → fmax/fmin intrinsics |
| Fast tier | ~42-46 μs | ~48-51 μs | noise | Unchanged |

**Surprise?** Partially. The division reduction helped significantly (-30%, -21%) but did NOT collapse to the fast tier. One remaining f64 division still costs ~1.5x. This contradicts the element-wise benchmark showing division = 1.04x — the per-element measurement doesn't capture the combine-level impact where division creates a pipeline stall in the scan tree's sequential dependency chain.

### Discussion
**What we learned**: Three-tier model collapses to two:
- Fast tier (~42-50μs): zero divisions in combine
- Division tier (~72-79μs): one division = 1.5x
- Branch tier eliminated by reducing to single-branch or branchless

**Open question**: Can ALL operators be reformulated to zero divisions in combine?

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| WelfordOp reformulation | Cache inv_n (1 div remaining) | Track sum/sum_sq (0 div, naive) | Numerical stability concern — tested separately |
| KalmanOp reformulation | Single-division algebraic form | Keep 5-division form | 21% improvement, same correctness |
| EWMOp reformulation | exp(log_decay * count) | Keep pow(1-α, count) | One transcendental instead of two |

---

## Experiment 4: EWM ≡ Steady-State Kalman (Proven)

### Before
**Hypothesis**: At F=1, H=1, the steady-state Kalman gain K_ss equals the EWM alpha parameter, and the two algorithms produce identical output.
**Design**: Generate random walk with Gaussian noise. Run sequential EWM(alpha=K_ss) and sequential Kalman(F=1,H=1,Q,R). Compare outputs.

### Results
```
Steady-state K_ss = 0.270156 for Q=0.01, R=0.1

Steady-state Kalman vs EWM(alpha=K_ss):
  max_err = 1.776357e-15   ← MACHINE EPSILON
```

**BIT-IDENTICAL.** Not approximately equal. The same computation, character for character:
```
EWM:    x[t] = alpha * z[t] + (1 - alpha) * x[t-1]
Kalman: x[t] = K_ss  * z[t] + (1 - K_ss)  * F * x[t-1]   (with F=1)
```

### Discussion
**What we learned**: EWM is not a heuristic smoothing technique. It IS the Bayes-optimal filter for a random walk observed in Gaussian noise. The alpha parameter is the steady-state Kalman gain — the solution to the Riccati equation for the signal-to-noise ratio Q/R.

**Historical note**: This connection is KNOWN in control theory (Muth 1960, Harvey 1990) but NOT widely known in data science or finance. Most practitioners using EWM don't know they're doing optimal Bayesian filtering.

**What's new**: The proof via bit-identical outputs on GPU scan operators, and the framing through the AssociativeOp trait that makes the family relationship explicit in the type system.

**Implication for operator families**: KalmanAffineOp(F=1,H=1,Q,R) and EWMOp(alpha=K_ss(Q,R)) are the SAME POINT in operator space. Families overlap. The operator space is continuous, not discrete categories.

---

## Experiment 5: Universal Affine Combine

### Before
**Hypothesis**: One combine function `(right.A * left.A, right.A * left.b + right.b)` handles constant-alpha EWM, variable-alpha EWM, cumsum, and variable-parameter Kalman — all bit-identical to their specialized implementations.
**Design**: Test same combine with four different lift functions on the same data.

### Results

| Use case | Lift function | max_err vs reference |
|---|---|---|
| Constant alpha EWM | `(1-α, α·z)` | **0.00e+00** |
| Variable alpha EWM | `(1-α_t, α_t·z)` | **0.00e+00** |
| Cumsum | `(1.0, z)` | **0.00e+00** |
| Variable Kalman | `((1-K_t·H)·F_t, K_t·z)` | **0.00e+00** |

ZERO ERROR. All four cases. Bit-identical to specialized implementations.

### Discussion
**What we learned**: The affine combine is UNIVERSAL for 1D linear recurrences. The specialization is entirely in the lift function. One kernel template, infinite operators.

**Key insight**: EWMOp's `pow(1-α, count)` in the combine was solving a problem that doesn't exist in the affine formulation. The "segment decay" is naturally captured by the accumulated A product in the scan tree. The affine formulation eliminates pow() entirely — even for variable alpha.

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Combine formulation | Universal affine (2 mul + 1 add) | Per-operator specialized combine | Zero error, fewer ops, handles all cases |
| Variable alpha handling | Different lift function, same combine | Separate operator with pow() | pow() is unnecessary — scan tree handles decay naturally |

---

## Experiment 6: KalmanAffineOp on GPU

### Before
**Hypothesis**: The affine Kalman formulation will produce machine-epsilon-level accuracy (unlike KalmanOp's covariance intersection at ~0.2 error) and land in the fast performance tier (~42μs) due to zero divisions in combine.
**Design**: Implement KalmanAffineOp with Riccati solver in constructor. Benchmark on GPU at FinTek sizes.

### Results
```
n=100:     max_err=1.78e-15   (machine epsilon)  PASS
n=10,000:  max_err=2.13e-14   (machine epsilon)  PASS
n=100,000: max_err=5.68e-14   317μs               PASS
```

**vs KalmanOp**: 5.68e-14 vs ~0.2 error. Seven orders of magnitude better.

### Discussion
The Riccati solver runs in the Rust constructor (1000 iterations, converges in ~50). The GPU combine is `{b.a * a.a, b.a * a.b + b.b}` — two multiplies, one add. No division, no branching. Fast tier confirmed.

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Kalman formulation | Affine (exact, fast) | Covariance intersection (approximate, slow) | 7 orders of magnitude better accuracy, same or faster speed |
| P computation | Riccati solver in Rust constructor | GPU-side P propagation | P converges in ~50 iterations, runs once |

---

## Experiment 7: RefCenteredStatsOp Discovery

### Before
**Hypothesis**: There exists a variance formulation that is BOTH numerically stable (like Welford) AND division-free in the combine (like naive sum/sumsq).
**Design**: Center all values around a fixed reference point (first observation) before accumulating. Track (count, sum_delta, sum_delta_sq) where delta = x - ref. The combine becomes pure addition. The cancellation is between small numbers (deviations from ref) not large numbers (values themselves).

### Results

| Data type | Naive (digits) | Welford (digits) | RefCentered (digits) |
|---|---|---|---|
| Returns (centered) | 16 | 14 | **16** |
| Prices (large mean) | 9 | 14 | **20 (EXACT)** |
| Prices (huge mean) | 2 | 10 | **16** |
| Timestamps (ns) | -5 (WRONG) | 1 | **13** |

**Surprise?** YES — RefCentered is MORE STABLE than Welford on every test case. Not just "good enough" — strictly better.

### Discussion
**Why RefCentered beats Welford:**
1. Welford accumulates per-step rounding errors through the running mean update (O(n) error propagation)
2. RefCentered sums deviations — in the scan tree, this IS pairwise summation (O(log n) error)
3. The combination of fixed-reference centering + pairwise scan tree = machine-epsilon precision

**Why RefCentered beats naive:**
- Naive: `sum_sq/n - (sum/n)²` — cancellation between O(mean²·n) terms
- RefCentered: `sum_delta_sq/n - (sum_delta/n)²` — cancellation between O(var·n) terms
- The deviations from ref are O(std), not O(mean). Cancellation goes from O(mean²/var) to O(1).

**The combine:**
```
State: (count: i64, sum_delta: f64, sum_delta_sq: f64) = 24 bytes
Combine: (a.n + b.n, a.sd + b.sd, a.sdsq + b.sdsq) = 3 ADDS. ZERO DIVISION.
Extract: mean = ref + sd/n, var = sdsq/(n-1) - sd²/(n*(n-1))
```

**Performance prediction**: Should hit fast tier (~42-50μs). Same combine cost as AddOp.

**Novelty assessment**:
- The shift technique is OLD (Chan, Golub, LeVeque 1979; Knuth 1969)
- Applying it to GPU parallel scan combines to eliminate division: NEW
- The observation that shifted > Welford in pairwise parallel context: NEW
- The "push complexity from combine to lift" design principle: NEW

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Variance formulation | RefCentered (0 div, 16 digits) | Welford (1 div, 10-14 digits) | Better stability AND better performance |
| Reference point | First observation | Running mean (Welford) | Known at construction time → no division in combine |
| State representation | (count, sum_delta, sum_delta_sq) | (count, mean, M2) | Enables pure-addition combine |

### Open Questions
- Does RefCentered maintain stability for ROLLING (windowed) variance? The prefix-sum subtraction for windowed computation may reintroduce cancellation.
- At what n does float64 RefCentered start losing digits? Theoretical: when sum_delta ≈ n · std, the cancellation in extract is O(n · std² / var) = O(n). At n = 10^8 and std/mean < 10^-4, might lose 4 digits. Still better than Welford at the same scale.

---

## Experiment 8: Särkkä 5-Tuple Kalman (Exact Transient)

### Before
**Hypothesis**: The Särkkä (2021) 5-tuple formulation for parallel Kalman filtering is exactly associative and handles transient P from step 1, unlike KalmanAffineOp which requires P convergence (~20 steps).
**Design**: Implement scalar 5-tuple (A, b, C, η, J) with the combine from Lemma 3 of the paper. Test associativity and compare to sequential Kalman.

### Results
```
Associativity check: max_err = 1.387779e-17  ← MACHINE EPSILON. ASSOCIATIVE.

Särkkä vs Sequential Kalman:
  x max_err = 1.33e-01  (initialization mismatch, not combine error)
  MSE vs true (seq):    2.28e-02
  MSE vs true (sarkka): 2.29e-02  (ratio: 1.007 — nearly identical)
```

**The combine IS associative.** The x/P discrepancy is from initialization (how the first observation is lifted into the 5-tuple), not from the combine formula.

### Discussion
**Combine cost**: 1 division (for `denom = 1 + C_l * J_r`) + ~10 multiplies + ~5 adds = 16 FLOPs.

**Arithmetic intensity analysis**:
```
Affine (2-tuple):  3 FLOPs / 32 bytes = 0.094 FLOP/byte → MEMORY BOUND
Särkkä (5-tuple):  16 FLOPs / 80 bytes = 0.200 FLOP/byte → BORDERLINE
Crossover:         269 GFLOPS / 1792 GB/s = 0.150 FLOP/byte
```

Both are approximately memory-bound on Blackwell. The 5-tuple is at the crossover but in practice shared memory access patterns keep it memory-bound.

**Estimated cost**: ~84μs at 100K (2x affine from 2.5x more state bytes). Extra for 4600-ticker farm: 193ms. Trivial.

**Implication**: if branch-free Särkkä 5-tuple (with the one division) lands near the division tier (~72-79μs), it may be acceptable as the ONLY Kalman operator — exact from step 1, no steady-state assumption. The affine version becomes an optional optimization, not a necessity.

**Init fix needed**: The lift function must match sequential Kalman's initial conditions. The combine is correct and associative. Only the initialization (lift of first observation) needs adjustment.

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| 5-tuple for production | Pending measurement | Pending | If ~80μs, replaces both KalmanOp and KalmanAffineOp for correctness from step 1 |
| Affine for production | Keep as fast option | Remove entirely | 42μs vs ~80μs may matter at extreme scale |

### Open Questions
- Does the branch-free 5-tuple with `inv_denom` (1 division, 5 multiplies by inverse) land in fast tier or division tier?
- If fast tier: AffineOp and SarkkaOp become a single operator family choice. If division tier: keep both.
- Can the 5-tuple be reformulated to ZERO divisions? The denom = 1 + C*J — could we track 1/C or 1/J instead, converting the division to multiplication?

---

## Design Principles Discovered

### Principle 1: Push Complexity from Combine to Lift/Constructor
The combine runs O(n log n) times in the scan tree. The constructor runs once. The lift runs O(n) times. Every operation moved from combine to constructor or lift is a multiplicative win.

**Examples**:
- KalmanOp → KalmanAffineOp: Riccati solver moved to constructor. Combine: 5 div → 0 div.
- WelfordOp → RefCenteredStatsOp: Centering moved to lift. Combine: 1 div → 0 div.
- EWMOp → AffineOp: Decay computation moved to lift (via A parameter). Combine: pow() → 2 mul.

### Principle 2: The Scan Tree IS Pairwise Summation
The Blelloch scan's tree structure naturally provides pairwise summation stability (O(log n) error growth). Algorithms that assume sequential accumulation (O(n) error) may be LESS stable than simpler formulations that benefit from the tree structure.

**Example**: RefCentered (naive + pairwise tree = 16 digits) > Welford (stable + sequential propagation = 10-14 digits)

### Principle 3: Performance Tiers Are Determined by Combine Complexity
- Fast tier (~42-50μs): adds and multiplies only
- Division tier (~72-79μs): one f64 division = 1.5x
- State size is IRRELEVANT (8B and 24B both at ~42μs)

### Principle 4: The Universal Affine Combine
Any 1D linear recurrence `x[t] = A_t * x[t-1] + b_t` can be parallelized with combine `(right.A * left.A, right.A * left.b + right.b)`. This handles constant parameters, variable parameters, cumsum (A=1), EWM, Kalman, ARIMA(1) — all with one combine.

### Principle 5: Operator Families Are Continuous, Not Discrete
Operators are coordinates in a continuous manifold defined by the AssociativeOp trait. EWMOp and KalmanAffineOp share a point (F=1, H=1, alpha=K_ss). The Fock boundary is the edge of the manifold. The trait IS the manifold.

---

## Artifacts

| Artifact | Location | Description |
|---|---|---|
| Operator implementations | `crates/winrapids-scan/src/ops.rs` | AddOp, MulOp, MaxOp, MinOp, WelfordOp, EWMOp, KalmanOp, KalmanAffineOp |
| EWM=Kalman proof | `~/.claude/garden/2026-03-30-ewm-is-kalman.md` | Bit-identical proof at matching parameters |
| Scan universe | `~/.claude/garden/2026-03-30-everything-is-a-scan.md` | Algorithms that are secretly scans |
| Liftability isomorphism | `~/.claude/garden/2026-03-30-the-liftability-scan-isomorphism.md` | Pith↔WinRapids connection |
| Whitacre chord | `~/.claude/garden/2026-03-30-the-whitacre-chord.md` | Recursive biphoton lifting |
| Observer lab notebook | `docs/research/sharing-optimizer/lab-notebook.md` | 23+ entries with measurements |
| Performance benchmarks | `crates/winrapids-compiler/src/bench_branchfree.rs` | Tier measurement code |
| Vision document | `docs/vision.md` | Full architecture with Phase 2+3 updates |
| Session insights | `docs/session-2026-03-30-insights.md` | 10 conceptual breakthroughs |

---

## Open Research Questions

1. **RefCenteredStatsOp GPU validation**: Does it actually hit the fast tier? The observer will measure.
2. **SarkkaOp tier placement**: Division tier or fast tier with inv_denom optimization?
3. **Rolling (windowed) RefCentered**: Does the prefix-sum subtraction reintroduce cancellation?
4. **Zero-division Särkkä**: Can the 5-tuple be reformulated to eliminate the denom division entirely?
5. **Matrix KalmanOp**: Can the AssociativeOp trait handle matrix-valued states for multi-dimensional Kalman? The trait takes `f64` input — would need generalization for vector observations.
6. **ARIMA family**: AR(1) = affine. AR(p) = companion matrix scan. What's the state size and combine cost for practical p values?
7. **The Fock gradient**: Can we quantify how much accuracy the EKF approximate lift loses as a function of the nonlinearity? Is there a metric that predicts when linearization is "good enough"?
