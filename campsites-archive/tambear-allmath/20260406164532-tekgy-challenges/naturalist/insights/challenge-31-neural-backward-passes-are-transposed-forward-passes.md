# Challenge 31 — Neural Backward Passes Are Transposed Forward Passes

**Date**: 2026-04-06  
**Type C: Foundation Challenge — structural identity**

---

## The Gap

`neural.rs` has forward passes for all operations but backward passes are incomplete:

| Operation | Forward | Backward |
|---|---|---|
| Activations (ReLU, GELU, etc.) | ✓ | ✓ |
| Linear (dense layer) | ✓ | ✓ |
| Loss functions | ✓ | ✓ |
| **Conv2D** | ✓ | ✗ MISSING |
| **Attention** | ✓ | ✗ MISSING |
| **BatchNorm/LayerNorm** | ✓ | ✗ MISSING |
| **MaxPool/AvgPool** | ✓ | ✗ MISSING |

This means `neural.rs` currently supports inference but not training. It's not a cuDNN replacement for training workloads.

---

## The Tekgy Observation: Backward = Forward Transposed

Challenge 19 (gradient duality) states: "Forward/backward = same DotProduct, transposed. Chain rule closed under accumulate."

For `neural.rs`, this means:

**Conv2D**: `y = Im2col(x) @ W^T` (forward)
- `∂L/∂W = Im2col(x)^T @ ∂L/∂y` — same GEMM, transposed inputs
- `∂L/∂x = Col2im(∂L/∂y @ W)` — same Im2col structure, reversed

**Attention**: `attn = softmax(QK^T/√d)` then `output = attn @ V`
- `∂L/∂V = attn^T @ ∂L/∂output` — same matmul, transposed attention
- `∂L/∂Q = d_softmax(∂L/∂output @ V^T / √d) @ K` — same matmul, different sides

**BatchNorm**: `y = γ(x - μ)/σ + β`
- `∂L/∂γ = Σ ∂L/∂y · (x-μ)/σ` — accumulate over batch
- `∂L/∂x` = complex chain through μ and σ, but structurally a scatter-gather

None of these require new algorithms. They're all the same Tiled accumulate (matmul) or windowed accumulate (convolution) with different transposition patterns.

---

## The Structural Insight

The gradient of ANY affine operation `y = f(x; W)` takes the form:
1. `∂L/∂W = x^T @ ∂L/∂y` (or equivalent gather pattern)
2. `∂L/∂x = W^T @ ∂L/∂y` (or equivalent scatter pattern)

The SAME accumulate kernel with transposed inputs gives the gradient. The accumulate type (Tiled for matmul, Windowed for conv) is IDENTICAL between forward and backward — only the indexing changes.

This is the practical implication of challenge 19's gradient duality theorem: once you have a working forward kernel, the backward kernel is the same kernel with transposed argument order.

---

## Most Actionable

1. `conv2d_backward`: `∂L/∂W` = same im2col + GEMM as forward, with `x^T` instead of `x`. `∂L/∂x` = same im2col, reversed patch-to-pixel indexing (col2im operation). Both are the same `emit_scatter_multi` kernel with different phi expressions.

2. `attention_backward`: 
   - `∂L/∂V = attn^T @ ∂L/∂output` — mat_mul with transposed attention
   - `∂L/∂Q, ∂L/∂K` — softmax Jacobian applied to the upstream gradient

3. `batch_norm_backward`: accumulate over batch dimension for `∂L/∂γ` and `∂L/∂β`, then chain rule through normalization for `∂L/∂x`.

All of these can use the existing `linear_algebra::mat_mul` + `codegen::emit_scatter_multi` infrastructure. No new kernel types needed.

---

## Connection to Challenge 19

Challenge 19 claimed: "chain rule closed under accumulate." This is the test. If adding backward passes requires only transposed accumulate calls (not new kernel types), challenge 19 is confirmed empirically.
