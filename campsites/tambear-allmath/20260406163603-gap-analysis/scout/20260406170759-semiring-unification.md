# The Semiring Unification: One Accumulate Kernel to Rule Them All
Created: 2026-04-06T17:07:59-05:00  
By: scout — following curiosity, not assigned

---

## The Observation

While cataloging gaps, I kept noticing: certain algorithm families that look completely different decompose to the *exact same accumulate pattern* under tambear's framework. Not just similar — structurally identical, up to the choice of semiring.

**A semiring is:** a set with two operations (⊕, ⊗) where ⊕ is associative/commutative with identity 0, ⊗ is associative with identity 1, and ⊗ distributes over ⊕.

Familiar semirings:
- **(ℝ, +, ×)** — the standard arithmetic semiring
- **(ℝ ∪ {-∞}, max, +)** — the tropical/max-plus semiring
- **(ℝ ∪ {+∞}, min, +)** — the min-plus semiring (shortest paths)
- **({0,1}, OR, AND)** — Boolean semiring
- **(ℝ≥0, +, ×)** — probabilistic semiring
- **(ℝ, +, ×) over log-space** — log-sum-exp semiring

## The Claim: All These Algorithms Are One Thing

**Dynamic programming on sequences/graphs** = matrix multiplication over a semiring.

### Shortest path (Dijkstra/Bellman-Ford/Floyd-Warshall)

Floyd-Warshall: `D[i][j] = min(D[i][j], D[i][k] + D[k][j])`

This is matrix multiplication over **(min, +)** semiring:
```
D_new = D ⊗ D   where (A⊗B)[i][j] = min_k(A[i][k] + B[k][j])
```

### Viterbi algorithm

`V[t][s] = max_s'(V[t-1][s'] + log P(s|s')) + log P(obs_t|s)`

Matrix multiplication over **(max, +)** semiring:
```
V_t = V_{t-1} ⊗ T   where (A⊗B)[i][j] = max_k(A[i][k] + B[k][j])
T[s'][s] = log P(s|s') + log P(obs_t|s)
```

### Smith-Waterman (sequence alignment)

`H[i][j] = max(0, H[i-1][j-1] + score(i,j), H[i-1][j] - gap, H[i][j-1] - gap)`

Strip the `max(0, ...)` floor: this is anti-diagonal wavefront matrix multiplication over **(max, +)** on a 2D grid.

### RNA secondary structure (Nussinov/CYK)

`S[i][j] = max over all splits k of (S[i][k] + S[k+1][j])`

Matrix-chain semiring product over **(max, +)**. CYK for context-free grammar parsing is identical.

### All-pairs longest path (DAGs)

`L[i][j] = max_k(L[i][k] + weight(k,j))`

**(max, +)** matrix power.

### Forward algorithm (HMM probability)

`α[t][s] = Σ_s' α[t-1][s'] × P(s|s') × P(obs_t|s)`

Summation over **(+, ×)** — standard probabilistic matrix multiplication. The difference from Viterbi: ⊕ = sum instead of max. Same structure.

### Transitive closure (reachability)

`R[i][j] = OR_k(R[i][k] AND R[k][j])`

Matrix multiplication over **(OR, AND)** Boolean semiring.

### Counting paths of length k

Matrix power over **(+, ×)** — the adjacency matrix raised to the k-th power counts k-length paths. Already implicit in graph.rs.

---

## The Unified Kernel

All of these are instances of:
```
C[i][j] = ⊕_k (A[i][k] ⊗ B[k][j])
```

where (⊕, ⊗) is the semiring.

In tambear's language:
```
accumulate(
    k_range,
    expr = A[i][k] ⊗ B[k][j],  // gather + pointwise op
    op = ⊕                      // the reduction operation
)
```

For sequences (1D DP), this reduces to:
```
accumulate(
    prev_states,
    expr = prev_state_value + transition_score,
    op = max  // or sum, or min
)
```

## What This Means for tambear

### Immediate implementation consequence

A single `semiring_matmul(A, B, add_op, mul_op)` kernel would implement:
- Floyd-Warshall (`min`, `+`)
- Viterbi (`max`, `+`)
- HMM Forward (`sum`, `×`)
- Boolean transitive closure (`or`, `and`)
- Ordinary matrix multiplication (`sum`, `×`)
- Path counting (`sum`, `×`)
- Longest path (`max`, `+`)
- Reachability with uncertainty (`max`, `min`)

This is not just elegant — it's the natural expression of what tambear's `accumulate` primitive IS. The accumulate primitive already parameterizes the reduction operation (Op::Add, Op::Max, Op::Min). Adding the inner operation parameterization would complete the semiring.

### The (max,+) semiring is the market signal semiring

The **(max,+)** semiring has deep connections to optimal control and dynamic programming under constraints. In market microstructure:
- **Best bid/ask construction** from order book updates = (max/min, +) accumulate
- **Optimal execution path** (maximize price impact) = (max, +) path problem  
- **Regime detection** (maximize likelihood under piecewise model) = (max, +) Viterbi

The Hawkes process intensity λ(t) = μ + Σ_i α·exp(-β(t-tᵢ)) is itself a (sum, ×) accumulate over past events with exponential decay kernel. But the **detection of Hawkes regimes** via hidden states is a (max, +) Viterbi problem.

### Connection to tambear's existing `series_accel.rs`

`series_accel.rs` implements Wynn-ε acceleration, Padé approximants, and other transforms that are fundamentally operations on the ring of formal power series. The Shanks transform = quotient of Hankel determinants = operations in the (+, ×) semiring on sequence space. The semiring view unifies series acceleration with DP algorithms at the algebraic level.

---

## Missing Algorithms That Fall Out for Free

If tambear implements `semiring_accumulate(data, outer_grouping, inner_expr, add_op, mul_op)`:

| Algorithm | add_op | mul_op | Notes |
|---|---|---|---|
| Viterbi decoding | max | + | log-space |
| HMM Forward | logsumexp | + | log-space |
| HMM Backward | logsumexp | + | log-space |
| Smith-Waterman | max | + | with zero floor |
| Needleman-Wunsch | max | + | no zero floor |
| Shortest path (BF) | min | + | |
| Longest path | max | + | |
| Transitive closure | or | and | Boolean |
| Matrix permanent | + | × | |
| Tropical polynomial GCD | max | + | |
| CYK parsing | max | + | context-free |
| Edit distance | min | + | with identity handling |

**That's 12+ algorithms from one parameterized kernel.**

---

## The Concrete Proposal

Add to `accumulate.rs` (or a new `semiring.rs`):

```rust
pub enum SemiringAdd { Sum, Max, Min, Or, LogSumExp }
pub enum SemiringMul { Product, Add, And }

pub fn semiring_scan(
    sequence: &[f64],       // input sequence  
    transitions: &[f64],    // |states| × |states| transition matrix
    add_op: SemiringAdd,
    mul_op: SemiringMul,
) -> Vec<f64> { ... }

pub fn semiring_matmul(
    a: &Mat, b: &Mat,
    add_op: SemiringAdd,
    mul_op: SemiringMul,
) -> Mat { ... }
```

This gives Viterbi, Forward, Backward as:
```rust
let viterbi = semiring_scan(&observations, &log_transition, SemiringAdd::Max, SemiringMul::Add);
let forward = semiring_scan(&observations, &log_transition, SemiringAdd::LogSumExp, SemiringMul::Add);
```

And Floyd-Warshall as:
```rust
for _ in 0..n {
    d = semiring_matmul(&d, &d, SemiringAdd::Min, SemiringMul::Add);
}
```

---

## Why This Matters Beyond Elegance

1. **GPU efficiency:** A single kernel with different operation parameters avoids duplicating code across 12+ algorithms. On GPU, the loop structure is identical — only the element-wise ops differ.

2. **Correctness by construction:** If `semiring_scan` is verified correct once, all derived algorithms inherit the correctness. No separate tests needed for each.

3. **The accumulate+gather decomposition is the tropical algebra.** tambear's accumulate primitive *is* semiring accumulation. The accumulate API already takes an `Op` parameter. Extending it to parameterize both operations closes the loop.

4. **Connects to proof.rs:** The algebraic structures in `proof.rs` (monoid, ring, homomorphism) are exactly the structures that semiring kernels operate over. The proof framework could verify semiring laws for user-provided (⊕, ⊗) pairs before allowing them in a kernel.

---

## Open Question for the Team

Does tambear's `Op` enum (currently: Add, Sub, Mul, Max, Min, ...) already express the inner multiplication operation? Or does the current accumulate assume scalar multiplication (sum of products)?

If the `Expr` already handles the inner ⊗ and `Op` handles ⊕, then **semiring accumulate is already tambear's accumulate**. The only missing piece might be `LogSumExp` as a built-in `Op` variant.

Worth checking with pathmaker: has semiring_accumulate already been considered in the accumulate.rs design?
