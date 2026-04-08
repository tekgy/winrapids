# Family 23: Neural Network Ops — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: REVIEWED
**Code**: `crates/tambear/src/neural.rs`

---

## Operations Tested

| Section | Operations | Lines | Verdict |
|---------|-----------|-------|---------|
| Activations | ReLU, LeakyReLU, ELU, SELU, GELU, Swish, Mish, Sigmoid, Tanh, Softplus, Softsign, HardSigmoid, HardSwish | 44-237 | OK |
| Softmax/LogSoftmax | softmax, log_softmax, softmax_backward | 239-261 | OK |
| Conv1D | conv1d, conv1d_multi | 267-327 | OK |
| Conv2D | conv2d (im2col + GEMM) | 329-390 | OK |
| Conv2D Transpose | conv2d_transpose | 392-437 | OK (LOW: square-kernel-only) |
| Max Pool | max_pool1d, max_pool2d | 444-519 | **MEDIUM: NaN→NEG_INFINITY** |
| Avg Pool | avg_pool1d, avg_pool2d, global_avg_pool2d | 467-562 | OK |
| Adaptive Pool | adaptive_avg_pool1d | 565-573 | OK (LOW: empty bins) |
| Batch Norm | batch_norm | 589-632 | OK (centered variance) |
| Layer Norm | layer_norm | 636-659 | OK (centered variance) |
| RMS Norm | rms_norm | 663-684 | OK |
| Group Norm | group_norm, instance_norm | 688-750 | OK (centered variance) |
| Dropout | dropout, dropout_backward | 757-785 | OK (inverted dropout) |
| Linear | linear, linear_backward | 793-828 | OK |
| Bilinear | bilinear | 832-860 | OK |
| Embedding | embedding, positional_encoding, rope | 866-922 | OK |
| Attention | scaled_dot_product, multi_head | 938-1045 | OK |
| MSE Loss | mse_loss, mse_loss_backward | 1052-1061 | OK |
| BCE Loss | bce_loss, bce_loss_backward | 1064-1081 | OK (clamped) |
| Cross-Entropy | cross_entropy_loss, cross_entropy_loss_backward | 1085-1110 | OK (log-softmax) |
| Huber Loss | huber_loss, huber_loss_backward | 1113-1136 | OK |
| Cosine Loss | cosine_similarity_loss | 1139-1146 | OK |
| Hinge Loss | hinge_loss | 1149-1152 | OK |
| Focal Loss | focal_loss | 1156-1164 | OK (clamped) |
| Utilities | clip_grad_norm/value, label_smooth, temperature_scale, top_k, top_p | 1170-1248 | OK |

---

## Finding F23-1: Max Pooling NaN Suppression (MEDIUM)

**Location**: `max_pool1d` line 455, `max_pool2d` line 506

**Bug**: Uses `if input[idx] > max_val` to track the maximum. Since `NaN > anything` is `false` in IEEE 754, NaN values are never selected. If ALL values in a pooling window are NaN, the output is `f64::NEG_INFINITY` (the initial value), silently converting NaN → −∞.

**Impact**: Silent data corruption. NaN should propagate through max pooling (as in PyTorch, TensorFlow, JAX). Downstream layers see finite values where they should see NaN, masking the original data issue.

**Fix**: Use `f64::max()` or check `is_nan()` explicitly:
```rust
// Option A: propagate NaN via is_nan check
if input[start + j].is_nan() || input[start + j] > max_val {
    max_val = input[start + j];
    max_idx = start + j;
}

// Option B: use f64::max (propagates NaN)
// Requires restructuring the argmax tracking
```

**Severity**: MEDIUM — silent incorrect output. Not a panic, but NaN suppression hides upstream data problems.

---

## Finding F23-2: conv2d_transpose Non-Square Kernel Padding (LOW)

**Location**: `conv2d_transpose` line 435

**Bug**: `full_pad = kh - 1 - padding.min(kh - 1)` uses `kh` for both height and width dimensions. For non-square kernels (kh ≠ kw) with padding > 0, the width padding is wrong — it should be `kw - 1 - padding.min(kw - 1)`.

**Impact**: Incorrect output for transposed convolutions with non-square kernels and non-zero padding. The underlying `conv2d` API only accepts a single padding parameter, so this is an API design limitation.

**Mitigation**: The API only supports symmetric stride/padding, so the inconsistency is contained within the constraint. Fixing requires adding separate height/width padding parameters to both `conv2d` and `conv2d_transpose`.

---

## Finding F23-3: Adaptive Pool Empty Bins (LOW)

**Location**: `adaptive_avg_pool1d` line 571

**Bug**: When `output_size > input.len()`, some bins have `start == end` (empty bin). Division by `(end - start) as f64` gives `0.0 / 0.0 = NaN`.

**Impact**: Produces NaN for empty bins. This is actually reasonable behavior (undefined average of no elements), but it's undocumented and may surprise callers. Extremely unlikely in practice since adaptive pooling almost always downsamples.

---

## Positive Findings

**ALL backward passes are mathematically correct.** Verified by hand:

| Backward | Formula | Verification |
|----------|---------|-------------|
| GELU | 0.5(1+tanh) + 0.5·x·sech²·d_inner | Product rule + chain rule on tanh approx ✓ |
| Swish | s + x·s·(1-s) | Product rule on x·σ(x) ✓ |
| Sigmoid | s·(1-s) | Standard derivative ✓ |
| Tanh | 1 - t² | Standard derivative ✓ |
| Softmax | s·(g - ⟨s,g⟩) | Jacobian-vector product ✓ |
| MSE | 2(p-t)/n | Standard ✓ |
| BCE | (-t/p + (1-t)/(1-p))/n | Standard ✓ |
| Cross-entropy | (softmax - one_hot)/n | Classic result ✓ |
| Huber | piecewise: d/n or δ·sign(d)/n | Standard ✓ |
| Linear | grad_input=g@W, grad_weight=g^T@X, grad_bias=sum(g,0) | Standard ✓ |
| Dropout | g·mask·scale | Correct for inverted dropout ✓ |

**Sigmoid is numerically stable.** Branch on x ≥ 0 avoids exp overflow in both directions. For x = 710 (near f64 exp overflow), uses 1/(1+exp(-710)) ≈ 1.0. For x = -710, uses exp(-710)/(1+exp(-710)) ≈ 0.0. ✓

**Softmax is numerically stable.** Subtract-max prevents exp overflow. ✓

**Log-softmax uses log-sum-exp trick.** Direct computation avoids log(softmax) cancellation. ✓

**Batch/Layer/Group/RMS normalization all use centered variance.** Two-pass approach: compute mean, then Σ(x-mean)². No naive formula bug. This is the strongest pattern in the file — consistent correct implementation across 4 normalization variants.

**BCE loss clamps predictions to (1e-12, 1-1e-12).** Prevents log(0). ✓

**Cross-entropy uses log_softmax internally.** Avoids separate softmax → log which would lose precision in the tail. ✓

**Focal loss clamps probabilities.** Same log(0) protection as BCE. ✓

**GELU uses the correct 0.044715 constant.** Matches Hendrycks & Gimpel 2016 and PyTorch's default `tanh` approximation.

**SELU uses exact Klambauer et al. 2017 constants.** α = 1.6732632423543772, λ = 1.0507009873554805. These are the self-normalizing fixed points.

**RoPE handles odd dimensions.** Last unpaired dimension passed through. ✓

**Positional encoding handles odd d_model.** Last sin component added without cos pair. ✓

**No NaN panics.** The file does NOT contain `partial_cmp().unwrap()`. Both `top_k_logits` and `top_p_logits` use `unwrap_or(std::cmp::Ordering::Equal)`. ✓

**Conv2D im2col indexing is correct.** Verified: patches extracted from padded input match the standard im2col layout, kernel matrix multiplication produces correct output shape.

**Inverted dropout scaling is correct.** Scale = 1/(1-p) applied during training ensures expected values match inference (where dropout is disabled). Edge cases p=0 (no drop) and p=1 (all drop) handled explicitly.

---

## Test Vectors

### TV-F23-MAXPOOL-NAN-01: NaN propagation in max pool
```
input = [1.0, NaN, 3.0, NaN, NaN]
kernel_size = 2, stride = 1
Expected (IEEE-compliant): [NaN, NaN, NaN, NaN]  (NaN in every window)
Current: [1.0, 3.0, 3.0, NEG_INFINITY]  (NaN suppressed)
Windows: [1,NaN], [NaN,3], [3,NaN], [NaN,NaN]
```

### TV-F23-MAXPOOL-NAN-02: All-NaN window
```
input = [NaN, NaN, NaN]
kernel_size = 3, stride = 1
Expected: [NaN]
Current: [NEG_INFINITY]
```

### TV-F23-GELU-BACKWARD-01: Gradient correctness
```
x = [0.0, 1.0, -1.0, 3.0]
Finite difference: (gelu(x+h) - gelu(x-h)) / (2h), h = 1e-7
Compare with gelu_backward(x, [1,1,1,1])
Expected: agreement to ~1e-6
```

### TV-F23-SOFTMAX-01: Extreme inputs
```
x = [1000.0, 1001.0, 999.0]
Expected: softmax still sums to 1.0 (subtract-max prevents overflow)
```

### TV-F23-SOFTMAX-02: All equal inputs
```
x = [c, c, c, c] for any c (including 1e300)
Expected: [0.25, 0.25, 0.25, 0.25]
```

### TV-F23-SIGMOID-01: Extreme inputs
```
sigmoid(710.0) ≈ 1.0 (no overflow)
sigmoid(-710.0) ≈ 0.0 (no underflow to -inf)
sigmoid(0.0) = 0.5
```

### TV-F23-CE-BACKWARD-01: Cross-entropy gradient
```
logits = [2.0, 1.0, 0.1], target = 0
grad = softmax(logits) - one_hot(0) = [s0-1, s1, s2]
Verify: sum(grad) ≈ 0 (conservation)
```

### TV-F23-BATCHNORM-01: Invariance
```
input: 100 samples, feature = offset + noise
After batch_norm: each feature has mean ≈ beta, std ≈ gamma
Verify: no naive formula error for offset = 1e8
```

### TV-F23-ROPE-01: Rotation property
```
For d=2: RoPE(x, pos) = rotation by pos·theta
After RoPE: ||output||² = ||input||² (norm-preserving)
```

### TV-F23-CONV2D-01: Identity kernel
```
input: 3x3 single channel, kernel: [[0,0,0],[0,1,0],[0,0,0]]
stride=1, padding=1
Expected: output = input (identity convolution)
```

### TV-F23-TRANSPOSE-01: Conv/ConvTranspose roundtrip
```
For stride=1, padding=0:
conv2d_transpose(conv2d(x, k), k_flipped) should recover x
(up to boundary effects)
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F23-1: Max pool NaN suppression | **MEDIUM** | Silent NaN → −∞ | `is_nan()` check or `f64::max()` |
| F23-2: conv2d_transpose non-square | **LOW** | Wrong padding for kh ≠ kw | Separate h/w padding params |
| F23-3: adaptive pool empty bins | **LOW** | NaN for output_size > input_len | Guard or document |

---

## Overall Assessment

**Second strongest module reviewed** (after special_functions.rs). The implementations are overwhelmingly correct with proper numerical stability patterns throughout. The single MEDIUM finding (NaN suppression in max pooling) is the only production-relevant issue. All backward passes verified correct by hand. The consistent use of centered variance in all normalization layers and the careful numerical treatment of loss functions show strong engineering discipline.
