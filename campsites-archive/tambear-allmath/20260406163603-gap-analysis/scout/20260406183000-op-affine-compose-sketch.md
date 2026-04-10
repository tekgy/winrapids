# Op::AffineCompose — The Implementation Sketch
Created: 2026-04-06T18:30:00-05:00  
By: scout-2  
Context: challenge 32 (naturalist) + challenge 27 (proof/accumulate seam)

---

## The Concrete Change

One new Op variant in `accumulate.rs` (or a new `affine_scan.rs`) unlocks the entire EWMA/ARMA/Kalman/EKF family:

```rust
// Carrier: a pair (multiplicative, additive)
#[derive(Debug, Clone, Copy)]
pub struct AffineState {
    pub a: f64,  // multiplicative factor
    pub b: f64,  // additive offset
}

impl AffineState {
    pub fn identity() -> Self { AffineState { a: 1.0, b: 0.0 } }

    // Right-to-left composition: apply left first, then right
    // (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2)
    pub fn compose(right: Self, left: Self) -> Self {
        AffineState {
            a: right.a * left.a,
            b: right.a * left.b + right.b,
        }
    }
}
```

The Op enum extension:
```rust
pub enum Op {
    // existing scalar ops unchanged...
    Add, Max, Min, Mul, /* ... */
    
    // New: structured state ops — the "wider Op" from challenge 32
    AffineCompose,             // scalar (a,b) pair
    AffineMatrixCompose(usize), // n×n matrix (A,b) pair
}
```

---

## The Full Degeneration Ladder

All instances of `accumulate(Prefix, expr, Op::AffineCompose)`:

| Model | a_t | b_t | Notes |
|-------|-----|-----|-------|
| EWMA | α or (1-α) | observation term | b=0 if no bias |
| AR(1) | φ | ε_t | b varies by timestep |
| Holt's linear trend | state-dependent | observation update | 2D → AffineMatrixCompose(2) |
| GARCH σ² | β | α·ε² | b=0 in base model |
| Adam m/v | (1-β₁), (1-β₂) | gradient terms | b varies |
| Kalman filter | F (dynamics) | K·(z-H·x_pred) | from SarkkaOp lift |
| EKF | Jacobian ∂f/∂x | linearized correction | precompute before scan |

The Sarkka 5-tuple is the same carrier extended with information-form backward variables (eta, J) for the smoother. `SarkkaOp` in `winrapids-scan/src/ops.rs` already implements this — the correction term `-J_b * b_a` is the b≠0 case from the garden entry.

---

## The Challenge 27 Connection

Once `Op::AffineCompose` exists, `canonical_structure()` closes the proof.rs loop:

```rust
impl Op {
    pub fn canonical_structure(&self) -> Option<Structure> {
        match self {
            Op::Add    => Some(Structure::commutative_monoid(0.0)),
            Op::Mul    => Some(Structure::commutative_monoid(1.0)),
            Op::Max    => Some(Structure::commutative_monoid(f64::NEG_INFINITY)),
            Op::AffineCompose => Some(Structure::monoid(AffineState::identity())),
            // not commutative — (a1,b1)∘(a2,b2) ≠ (a2,b2)∘(a1,b1) in general
            _ => None,
        }
    }
}
```

Returning `Structure::Monoid` triggers auto-generation of the ParallelMerge correctness certificate in `proof.rs` — every `accumulate(Prefix, ..., Op::AffineCompose)` call gets a machine-checked proof that the Blelloch pattern is correct. For free.

---

## Files to Change

1. **`crates/tambear/src/accumulate.rs`**: Add `Op::AffineCompose` variant + `AffineState` struct + dispatch in the combine function
2. **`crates/tambear/src/proof.rs`**: Add `Op::canonical_structure()` method (challenge 27 — may already be planned)
3. **`crates/tambear/src/time_series.rs`**: Add `kalman_filter(obs, F, H, Q, R, x0, P0)` that constructs the SarkkaOp lift and calls accumulate — then AR/GARCH/Holt's as special cases
4. **`crates/tambear/tests/gold_standard_parity.rs`**: Kalman oracle tests vs scipy.signal.lfilter + pykalman

The scan primitive itself (`SarkkaOp`) already exists in `winrapids-scan`. No new kernel needed.

---

## Implementation Order for Pathmaker

1. `Op::AffineCompose` + `AffineState::compose` (pure math, no GPU)
2. Test: `accumulate(Prefix, [1.0, 2.0, 3.0], AffineState(a=0.5, b=x_t), AffineCompose)` matches hand-computed values
3. `kalman_filter` wrapper that constructs (a_t, b_t) from (F, H, Q, R, z_t) and calls the scan
4. EWMA, AR(1), Holt's as degenerate wrappers (b=0, constant a, etc.)
5. Oracle tests for all four against scipy/statsmodels
