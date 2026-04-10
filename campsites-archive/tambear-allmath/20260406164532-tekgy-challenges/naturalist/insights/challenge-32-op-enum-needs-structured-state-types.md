# Challenge 32 — Op Enum Needs Structured State Types

**Date**: 2026-04-06  
**Type A: Representation Challenge — the missing seam**  
**Credit**: scout-2 synthesis of challenges 13+28+29

---

## The Traditional Assumption

Adding GARCH, Kalman filter, ARMA, and EWMA parallel implementations requires new kernel infrastructure — new kernel types, new GPU primitives, new dispatch paths.

## Why It Dissolves

The existing `accumulate(Prefix, ..., op)` dispatch already IS Blelloch. What's missing isn't new infrastructure — it's new `Op` variants for structured state types.

---

## The Concrete Gap

The current `Op` enum (in `accumulate.rs`) contains scalar operations:
```rust
pub enum Op {
    Add,   // commutative monoid (ℝ, +, 0)
    Mul,   // commutative monoid (ℝ, ×, 1)
    Max,   // lattice (ℝ, max, -∞)
    Min,   // lattice (ℝ, min, +∞)
    Sub,   // NOT associative
    Div,   // NOT associative
    Count, // alias for Add with phi=1
    // JIT variants...
}
```

What's needed:

```rust
pub enum Op {
    // ... existing scalar ops ...
    
    /// Welford streaming merge: (n, mean, M2) × (n, mean, M2) → (n, mean, M2)
    /// Enables: online variance, COPA streaming covariance
    WelfordMerge,
    
    /// Affine function composition: (A, b) ∘ (A', b') = (A·A', A·b' + b)
    /// Carrier: (Mat(2,2,Real), Vec(2,Real))
    /// Enables: GARCH, EWMA, AR(p), Kalman filter, Adam moments — ALL of them
    AffineCompose,
    
    /// Log-sum-exp semiring: (max, shifted-sum) state
    /// Enables: numerically stable softmax, log-probabilities
    LogSumExpMerge,
    
    /// Sarkka 5-tuple: (A, b, C, η, J) — full parallel Kalman
    /// Note: requires the correction term -J_b·b_a from garden/006-the-correction-term.md
    SarkkaMerge,
}
```

---

## Why This Changes Everything

Once `Op::AffineCompose` exists:

```rust
// GARCH(1,1): σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
// Write as (A_t, b_t) where A_t = [[β, α·r²_t], [0, 0]] and b_t = [ω, r²_t]
// Then:
accumulate(timesteps, transition_matrix(t), Grouping::Prefix, Op::AffineCompose)
// → all σ²_t in O(log n) via Blelloch

// EWMA: S_t = α·x_t + (1-α)·S_{t-1}  
// Write as A_t = [[1-α]], b_t = [α·x_t]
accumulate(timesteps, ewma_element(t, alpha), Grouping::Prefix, Op::AffineCompose)
// → all S_t in O(log n)

// Adam first moment: m_t = β₁·m_{t-1} + (1-β₁)·g_t
accumulate(timesteps, adam_m_element(t, beta1), Grouping::Prefix, Op::AffineCompose)
// → all m_t in O(log n)

// Kalman filter: full (A_t, b_t) with observation updates
accumulate(timesteps, kalman_element(t, F, H, Q, R, z_t), Grouping::Prefix, Op::AffineCompose)
// → filtered states (with SarkkaMerge for the full 5-tuple)
```

**One Op variant. Every model in the list. No new kernel dispatch. No new grouping patterns.**

---

## Connection to Challenge 27 (Proof↔Accumulate)

`Op::AffineCompose.canonical_structure()` returns:
```rust
Structure::semigroup(
    Sort::Product(Sort::Mat(2,2,Real), Sort::Vec(2,Real)),
    BinOp::AffineCompose,
)
```

Note: semigroup, not monoid. The identity element is `(I, 0)` (identity matrix, zero vector), making it a monoid, but the Blelloch scan works with just associativity.

The proof certificate auto-generates via `CompositionRule::ParallelMerge`. Challenge 27's `Op::canonical_structure()` function is the seam that connects the new Op variants to the proof system.

---

## The Degeneration Hierarchy

All models are degenerations of `Op::SarkkaMerge` (full Kalman):

| Model | A | b | Observation | Notes |
|---|---|---|---|---|
| GARCH(1,1) | 2×2 | nonzero | n/a | b_t = [ω + α·r²_t, r²_t] |
| EWMA | [[1-α]] | [α·x_t] | n/a | b=0 for unweighted |
| AR(1) | [[φ]] | [0] | direct | b=0 |
| Holt's | 2×2 | [α·x_t, 0] | n/a | b nonzero |
| **Kalman** | F | K·(z-H·x) | H·x + noise | Full SarkkaMerge |
| EKF | Jac(f) | linearize | H·x + noise | Needs Jacobian precomputed |

The b=0 models are safe without the correction term. The nonzero-b models (GARCH, Holt, Kalman) require the `-J_b·b_a` correction from `006-the-correction-term.md` to work correctly in the Blelloch tree.

---

## Implementation Order

1. **`Op::WelfordMerge`** — simplest, already implemented as a standalone fn in `descriptive.rs`. This is just hoisting the existing merge into the Op enum.

2. **`Op::AffineCompose`** — enables GARCH, EWMA, AR, Adam. State = (Mat2x2, Vec2). Correction term NOT needed for b=0 variants (start here). Add nonzero-b support (Holt, GARCH with ω) in a second pass.

3. **`Op::LogSumExpMerge`** — enables numerically stable softmax and log-probabilities. Already documented in the math taxonomy.

4. **`Op::SarkkaMerge`** — full Kalman. Requires the correction term. Test against AR(1) gold standard at n≥8 before trusting (RTS backward risk, see challenge 29).

Each step is independent. Each step unlocks a family of models. The implementation is bottom-up through this degeneration hierarchy.
