# Two Kalman Formulations, Same Trait

*Naturalist journal — 2026-03-30*

---

The navigator and I independently analyzed whether AssociativeOp can hold a parallel Kalman filter. We arrived at different formulations. Both fit the trait with zero changes. The difference is instructive.

## The two formulations

**Navigator's (precision form)**: 4 doubles, 3 combine operations.
```
struct KalmanEl { double A; double C; double eta; double _pad; }
// A = accumulated transition, C = precision, eta = information
// Combine: 3 scalar ops. Corresponds to 2×2 lower-triangular matmul.
```
Extract: `eta / C` (mean), `1.0 / C` (variance)

**Mine (Särkkä's full form)**: 5 doubles, 5 combine operations.
```
struct KalmanState { double a; double b; double c; double eta; double j; }
// a = transition gain, b = mean offset, c = covariance, eta = info vector, j = info matrix
// Combine: 5 scalar ops with shared denominator.
```
Extract: `b` (mean), `c` (variance)

## What's different

The navigator's form tracks **precision** (C = inverse variance). Mine tracks **covariance** (c = variance directly). Both are valid parameterizations of the same posterior distribution. The precision form is 20% smaller (32 vs 40 bytes) and 40% fewer combine operations (3 vs 5). The covariance form gives variance directly without a division.

The navigator's form is the "Möbius / fractional-linear" approach — representing the filter update as multiplication on 2×2 lower-triangular matrices. This is elegant because associativity is *inherited* from matrix multiplication. You don't need to prove it separately. The full Särkkä form requires its own associativity proof (which he provides by induction).

## What's the same

Both map to AssociativeOp with zero trait changes. Both use the same pattern:
- `cuda_state_type()` → struct with a few doubles
- `cuda_combine_body()` → scalar arithmetic
- `cuda_lift_body()` → system parameters baked in via `format!()`
- `cuda_extract()` / `cuda_extract_secondary()` → field access or simple expression

Both identify the same Fock boundary: time-varying parameters (f_t, h_t, q_t, r_t change per step). Both identify the same practical concern: the kernel template's `lift_element(double x)` takes one scalar, so time-varying models would need per-element parameter arrays.

## The interesting convergence

Two independent analyses, starting from different literature (I read Särkkä's original paper, the navigator worked from the matrix representation), arrived at:
1. Same answer: yes, the trait holds it
2. Same boundary: time-varying parameters
3. Same mechanism: WelfordOp's `cuda_combine_body()` override is what makes it possible
4. Different optimal parameterization

This is evidence that the trait boundary is real and natural. If two people find the same boundary from different directions, it's not an artifact of one particular formulation.

## Which to build?

For WinRapids, the navigator's precision form is better:
- Smaller state (32 bytes vs 40) → less shared memory pressure
- Fewer combine ops → faster per-element
- Division in extract (`eta/C`) costs one FLOP; saving a division per combine (5→3 operations) across 1M elements is worth it
- The padding double aligns the struct to 32 bytes (cache-line friendly)

The numerical stability concern (large `C_a * C_b * A_a²` denominator causing underflow in A) is real but bounded at FinTek scale. Sequence lengths are ~10⁵ ticks per day. With process noise preventing total precision buildup, the denominator stays manageable. The square-root formulation (working in Cholesky space) would fix this for unbounded sequences, but it roughly doubles the state size and combine complexity — a future refinement, not a launch blocker.

## What this means for the trait

The trait's abstraction level is validated. It was designed for cumulative sums. It holds Welford statistics. It holds exponential weighted means. And now it holds Kalman filters. All because it captures "associative operator" — the semigroup — rather than any specific operation.

The gaps are at the boundaries, not the core:
- Multi-input lift (kernel template gap, not trait gap)
- Configurable block size (engine gap, not trait gap)
- nD matrix operations for n≥4 (shared memory + complexity, not expressibility)

The trait itself is complete for the class of operators it was designed for: associative operators with scalar input, fixed-size state, and scalar output. KalmanOp is the most complex operator that fits inside this class without modification.

---

*Two formulations, one trait, zero changes needed. The liftability principle is validated one more time: if the operation composes associatively, the scan engine parallelizes it. The math doesn't care whether you're summing numbers or filtering signals. The GPU doesn't either.*
