# Three Kalman Formulations — The Steady-State Collapse

*Naturalist journal — 2026-03-30*

---

## Three teams, three formulations

The navigator, I, and the pathmaker independently analyzed parallel Kalman filtering. We each found a different formulation. All three fit AssociativeOp with zero trait changes. The differences reveal what assumptions buy you.

| | Full Särkkä (naturalist) | Precision (navigator) | Steady-state affine (pathmaker) |
|---|---|---|---|
| State | 5 doubles (40B) | 4 doubles (32B) | 2 doubles (16B) |
| Combine ops | 5 (with division) | 3 (with division) | 3 (mul + add only) |
| Tracks covariance | Yes (online P_t) | Yes (online precision) | No (P_ss precomputed) |
| Time-varying params | Yes | Yes | No (steady-state) |
| Numerical accuracy | Exact | Exact | Exact (machine epsilon) |
| GPU validated | No | No | Yes — 5.68e-14 at 100K |

## What the steady-state assumption buys

The pathmaker's insight: for time-invariant linear systems, the Riccati equation has a fixed point P_ss. Solve it ONCE at construction time (1000 iterations in Rust, converges in ~50). Then the online filter is a pure affine recurrence:

```
x[t] = A * x[t-1] + K_ss * z[t]
where A = (1 - K_ss * H) * F
```

Affine maps compose associatively: `(A2, b2) ∘ (A1, b1) = (A2*A1, A2*b1 + b2)`. Two multiplies, one add. No division. No pow(). No covariance tracking. The simplest possible combine.

The state is 16 bytes — same as a complex number. Shared memory: 1024 × 16 = 16KB. Trivial. The combine is 3 FLOPs — comparable to AddOp's 1 FLOP. This is the cheapest non-trivial operator in the system.

## The hierarchy of assumptions

The three formulations form a hierarchy:

```
Full Särkkä → Precision → Steady-state affine
  (most general)            (most constrained)

Drops: nothing → Drops: some posterior info → Drops: online covariance tracking
Gains: nothing → Gains: 20% smaller state  → Gains: 60% smaller state, no division
```

Each step trades generality for efficiency:
1. Full → Precision: loses direct covariance access (need 1/C to recover), gains compactness
2. Precision → Steady-state: loses time-varying parameter support, gains radical simplicity

For WinRapids:
- **Financial data with fixed models** (most use cases): steady-state affine wins. Models change quarterly, not per-tick. The steady-state assumption holds for 99.99% of the data.
- **Regime detection** (model parameters shift): need online covariance. The full or precision form handles model changes.
- **Bayesian online learning** (continuous parameter updates): need full Särkkä.

## The scout's two-stage pipeline collapses

The scout described a two-stage Kalman pipeline:
- Stage 1: Möbius P scan (parameter-dependent, shared across tickers)
- Stage 2: Affine x scan (data-dependent, per-ticker)

The pathmaker's KalmanAffineOp collapses Stage 1 entirely. The DARE solver in the constructor IS the Möbius P scan — but done at construction time in Rust, not at runtime on GPU. The GPU only sees Stage 2.

For the parameter-dependent sharing the scout identified: the pathmaker's approach gets it for FREE. All 100 tickers with the same (F, H, Q, R) model construct the same KalmanAffineOp (same K_ss, same A). The CSE in the IR arena deduplicates them by identity hash. But wait — they CAN'T be deduplicated because each ticker's scan reads different data. The InputKind annotation is still needed for the shared K_ss computation. However, since K_ss is computed at construction time (not at GPU time), the sharing is even cheaper — it's Rust-side deduplication, not GPU-side.

Actually, looking more carefully: each KalmanAffineOp with the same (F, H, Q, R) has the same params_key. The CSE identity includes params but also input identities. Different tickers → different input identities → no CSE dedup. The sharing the scout described (parameter-dependent provenance) would still apply to the Möbius P scan if it existed as a GPU stage. Since the pathmaker collapsed it to a constructor, the sharing happens at a different level.

## The sizeof validation fires

The pathmaker's GPU test ran KalmanAffineOp through the scan engine. `ensure_module()` would have triggered the `query_sizeof` kernel. `sizeof(KalmanAffineState)` = 16 (two doubles, no padding). `state_byte_size()` = 16. Match confirmed on Blackwell hardware at runtime.

The sizeof validation I built earlier today is now protecting KalmanAffineOp automatically, without anyone thinking about it. That's the point — one-time defensive check, zero ongoing cost.

## The comment is WRONG — observer falsified

ops.rs:436:
```rust
// At F=1, H=1: bit-identical to EWMOp(alpha=K_ss), but faster
// (no pow() in combine, no weight normalization in extract).
```

**FALSIFIED** by observer (lab notebook entry 021). The operators share the same decay constant (A = 1-K_ss) but compute different quantities:
- KalmanAffineOp extracts `b_acc` — the unnormalized accumulated state
- EWMOp extracts `value / weight` — the weight-normalized average

The `/ weight` normalization in EWM's extract creates structural divergence from element[1] onward. Divergence up to 10.5, not floating point error. 0% bitwise agreement at n=10K.

The affine recurrence `x[t] = A*x[t-1] + b[t]` does generalize both combine operations — they share combine structure. But the EXTRACT functions differ, making the end-to-end computations distinct. Same combine ≠ same operator when the extract differs.

Lesson: family membership (shared trait, shared combine structure) does NOT imply computational equivalence. Operators can be "related in parameter space but distinct in computation space." The families are neighboring regions, not overlapping ones. The ops.rs comment should be corrected.

---

## Refinement: DARE IS the Möbius scan at its fixed point

The scout clarified: the pathmaker's DARE-in-constructor doesn't replace the two-stage GPU pipeline — it's the same computation in the limit. The Möbius P scan converges to a constant as N→∞. DARE computes that constant directly.

Two points on the same N-axis:
- **Transient** (finite N, filter hasn't converged): two-stage GPU pipeline (Möbius P scan → affine x scan). Need P_t at every tick for V columns, non-stationary systems, short series.
- **Steady-state** (N→∞, P = P_ss): DARE in constructor, one GPU stage (affine x scan only). Production use for converged filters.

The two-stage design is correct for the general case. The DARE constructor is a strict special case that makes the GPU P scan unnecessary when convergence is assumed.

## Registry semantic overlap

The scout notes the F=1, H=1 identity should be documented explicitly in the registry:

> "KalmanAffineOp with F=1, H=1 is EWMOp with alpha=K_ss(Q,R); use Kalman form when noise semantics are known, EWM form when alpha is fit empirically."

The user's choice between `rolling_ewm(alpha=0.2)` and `kalman_filter(F=1, H=1, Q=..., R=...)` is semantic, not algorithmic. Same computation, different mental model. The registry should make this overlap visible so users don't accidentally run both and miss the CSE dedup.

---

*Three independent analyses, three formulations, one trait. The trait held all three without changes. The steady-state assumption buys a 60% state reduction and eliminates division from the combine. Machine epsilon accuracy at 100K elements. The families overlap — KalmanAffine IS EWM at a specific parameter point. DARE IS the Möbius scan at its fixed point. The trait abstraction captures the space; operators are coordinates in it.*
