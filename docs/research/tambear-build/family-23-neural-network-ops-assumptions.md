# Family 23: Neural Network Operations — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (convolutions, attention = parallel accumulates) + B (causal attention = masked scan)

---

## Core Insight: Three GPU Primitives Cover Everything

1. **GEMM** (matrix multiply): Linear layers, attention logits, 1x1 convolutions
2. **Convolution**: Feature extraction (direct/FFT/Winograd implementations)
3. **Reduction**: Pooling, normalization statistics, softmax denominator

Everything else (activations, dropout, embedding) is pointwise.

---

## 1. Convolution

### 1D Convolution
```
y[n] = Σ_{k=0}^{K-1} w[k] · x[n·s + k - p]
```
where K = kernel size, s = stride, p = padding.

Output size: ⌊(N + 2p - K) / s⌋ + 1.

### 2D Convolution
```
y[c_out, h, w] = Σ_{c_in} Σ_{kh} Σ_{kw} W[c_out, c_in, kh, kw] · x[c_in, h·s+kh-p, w·s+kw-p]
```

### Implementation Strategies

**Direct (im2col + GEMM)**:
1. Unfold input patches into columns (im2col): each patch becomes a row
2. Matrix multiply: Y = W_reshaped · X_unfolded
- Simple, leverages optimized GEMM
- Memory overhead: C_in · K_h · K_w × H_out · W_out matrix

**FFT-based**:
1. FFT(x), FFT(w), pointwise multiply, IFFT
- Better for large kernels (K > 7)
- Uses F03 FFT

**Winograd** (Lavin & Gray 2016):
Transform input and filter to Winograd domain, pointwise multiply, transform back.
For 3×3 filters (most common): F(2,3) uses 2.25× fewer multiplications.
```
Y = A' [(G·g·G') ⊙ (B'·d·B)] A
```
where G, B, A are fixed transform matrices.

**CRITICAL**: Winograd is numerically less stable than direct. Use for inference (fp16/fp32). Avoid for high-precision gradients.

### Depthwise Separable Convolution (MobileNet)
1. Depthwise: C_in independent K×K convolutions (one per channel)
2. Pointwise: 1×1 convolution (C_in → C_out)
Cost: C_in·K²·H·W + C_in·C_out·H·W vs C_in·C_out·K²·H·W (direct). Savings: ~K²/C_out + 1/K² ≈ 8-9x for 3×3.

### Transposed Convolution (Deconvolution)
Insert s-1 zeros between input elements, then convolve with flipped kernel. Used in upsampling / decoder networks.

### Dilated (Atrous) Convolution
Insert d-1 zeros between kernel elements. Effective receptive field = K + (K-1)(d-1) without increasing parameters.

---

## 2. Pooling

### Max Pooling
```
y[h,w] = max_{kh,kw} x[h·s+kh, w·s+kw]
```
Gradient: flows only to the max element (sparse gradient).

### Average Pooling
```
y[h,w] = (1/(K_h·K_w)) Σ_{kh,kw} x[h·s+kh, w·s+kw]
```
Gradient: distributed equally to all elements in the window.

### Global Average Pooling (GAP)
Average over entire spatial dimensions: output = 1 value per channel.
```
y[c] = (1/(H·W)) Σ_{h,w} x[c,h,w]
```
This is `accumulate(ByKey(channel), x, Mean)`.

### Adaptive Pooling
Specify output size, compute window sizes automatically. PyTorch standard.

---

## 3. Normalization

### 3a. Batch Normalization (Ioffe & Szegedy 2015)
```
ŷ = γ · (x - μ_B) / √(σ²_B + ε) + β
```
where μ_B, σ²_B computed across batch and spatial dimensions per channel.

Training: μ_B = mean over (N, H, W). σ²_B = variance over (N, H, W).
Inference: use running mean/variance (EMA from training).

### 3b. Layer Normalization (Ba et al. 2016)
```
μ, σ² computed across (C, H, W) per sample
```
No batch dependency → works with any batch size, sequential data, transformers.

### 3c. Group Normalization (Wu & He 2018)
```
μ, σ² computed across (C/G, H, W) per sample per group
```
G groups of C/G channels. G=1: LayerNorm. G=C: InstanceNorm.

### 3d. Instance Normalization
```
μ, σ² computed across (H, W) per sample per channel
```
Used in style transfer.

### 3e. RMS Normalization (Zhang & Sennrich 2019)
```
ŷ = γ · x / RMS(x)    where RMS(x) = √((1/d)Σ x²_i)
```
No mean subtraction. Simpler, often works as well as LayerNorm. Used in LLaMA, GPT-NeoX.

### All normalizations are: compute statistics (accumulate) → normalize (pointwise) → affine transform (pointwise).

---

## 4. Attention

### 4a. Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK'/√d_k) · V
```
where Q (n×d_k), K (m×d_k), V (m×d_v).

### Softmax
```
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```
**CRITICAL**: Subtract max for numerical stability. Without it, exp overflows for large x.

### 4b. Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
where head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)
```
h heads, each with d_k = d_model/h.

### 4c. Causal (Masked) Attention
For autoregressive models: mask future positions.
```
Mask[i,j] = {-∞  if j > i
            {0    otherwise
Attention = softmax((QK'/√d_k) + Mask) · V
```

### 4d. Flash Attention (Dao et al. 2022)
**Key insight**: Standard attention materializes the n×n attention matrix (O(n²) memory). Flash Attention computes attention in tiles, never materializing the full matrix.

Algorithm:
```
For each query tile Q_i:
  m_i = -∞, l_i = 0, O_i = 0
  For each key/value tile K_j, V_j:
    S_ij = Q_i · K_j' / √d_k
    m_new = max(m_i, rowmax(S_ij))
    P_ij = exp(S_ij - m_new)
    l_new = exp(m_i - m_new) · l_i + rowsum(P_ij)
    O_i = exp(m_i - m_new) · O_i + P_ij · V_j
    m_i = m_new, l_i = l_new
  O_i = O_i / l_i
```

This is an **online softmax** — a scan that accumulates numerator and denominator without materializing the full matrix.

**Memory**: O(n) instead of O(n²). **Speed**: 2-4x faster (less memory bandwidth).

### 4e. Sparse Attention
- **Local**: attend only to window of k neighbors
- **Strided**: attend to every k-th position
- **Block-sparse**: attend within blocks (BigBird, Longformer)
- **Linear attention**: replace softmax(QK') with φ(Q)·φ(K)' → O(n·d²) instead of O(n²·d)

### Kingdom: Dense attention = A (parallel matmuls). Causal attention = B (masked accumulate). Flash attention = B (tiled online scan).

---

## 5. Activations

### Standard
| Function | Formula | Derivative |
|----------|---------|-----------|
| ReLU | max(0, x) | I(x>0) |
| LeakyReLU | max(αx, x), α=0.01 | α if x<0, 1 if x>0 |
| GELU | x·Φ(x) ≈ 0.5x(1+tanh(√(2/π)(x+0.044715x³))) | complex |
| SiLU/Swish | x·σ(x) | σ(x) + x·σ(x)(1-σ(x)) |
| Mish | x·tanh(softplus(x)) | complex |
| Sigmoid | 1/(1+e^{-x}) | σ(x)(1-σ(x)) |
| Tanh | (e^x-e^{-x})/(e^x+e^{-x}) | 1-tanh²(x) |
| Softplus | log(1+e^x) | σ(x) |
| ELU | x if x>0, α(e^x-1) if x≤0 | 1 if x>0, f(x)+α if x≤0 |

**GELU is the default for transformers** (BERT, GPT). SiLU for vision (EfficientNet).

All activations are elementwise → embarrassingly parallel.

---

## 6. Dropout

### Training
```
mask ~ Bernoulli(1-p)
y = x · mask / (1-p)    (inverted dropout — scale at train time)
```

### Inference: y = x (no dropout).

### DropConnect: Drop weights instead of activations.
### DropPath: Drop entire residual branches (used in EfficientNet, vision transformers).

---

## 7. Embedding

### Token Embedding
```
E(token_id) = W_embed[token_id, :]    (lookup table, shape: vocab_size × d_model)
```
Gradient: sparse update to rows that were selected.

### Positional Encoding

**Sinusoidal (Vaswani et al. 2017)**:
```
PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```
Precomputed, no learnable parameters.

**Learned**: Trainable embedding matrix (max_len × d_model).

**RoPE (Rotary Position Embedding — Su et al. 2021)**:
```
q'_m = R_m · q_m    where R_m = block-diag rotation matrices at angle m·θ_i
```
θ_i = 10000^{-2i/d}. Applied to Q and K before attention.

**ALiBi (Press et al. 2022)**: Add linear bias to attention logits based on distance. No explicit positional encoding.

---

## 8. Backward Pass (Gradient Computation)

### Convolution Backward
- ∂L/∂W = conv(input, ∂L/∂y) — correlation of input with output gradient
- ∂L/∂x = conv_transpose(W, ∂L/∂y) — transposed convolution of weights with output gradient

### Attention Backward
```
∂L/∂V = P' · ∂L/∂O                                (P = softmax attention weights)
∂L/∂P = ∂L/∂O · V'
∂L/∂S = P ⊙ (∂L/∂P - (∂L/∂P ⊙ P)·1·1')         (softmax Jacobian)
∂L/∂Q = ∂L/∂S · K / √d_k
∂L/∂K = ∂L/∂S' · Q / √d_k
```

### BatchNorm Backward
Requires: stored pre-normalized x, μ_B, σ²_B from forward pass.

### CRITICAL: Flash Attention backward recomputes attention weights (not stored) — trades compute for memory.

---

## Sharing Surface

### Reuse from Other Families
- **F02 (Linear Algebra)**: GEMM for linear layers, attention, 1×1 conv
- **F03 (Signal Processing)**: FFT for large-kernel convolution
- **F05 (Optimization)**: ALL training uses F05 optimizers (Adam, SGD)
- **F06 (Descriptive)**: Normalization statistics = per-channel/layer means and variances

### Consumers of F23
- **F24 (Training)**: ALL training infrastructure consumes F23 ops
- **F22 (Dimensionality Reduction)**: Autoencoder architectures use conv/attention
- **F28 (Manifold)**: Parametric UMAP uses neural networks

### Structural Rhymes
- **Convolution = FIR filter (F03)**: same operation, different context
- **BatchNorm = standardization (F06)**: same μ/σ computation, applied per-channel
- **Attention = soft dictionary lookup**: Q queries, K keys, V values
- **Softmax = Boltzmann distribution**: same as statistical mechanics partition function
- **Flash Attention = online accumulate**: same principle as Welford (F06) — never materialize full intermediate
- **RoPE = Fourier features**: same sinusoidal basis as positional encoding

---

## Implementation Priority

**Phase 1** — Core ops (~300 lines):
1. Convolution 1D/2D (im2col + GEMM)
2. Max/Average/Global pooling
3. All activations (ReLU, GELU, SiLU, Sigmoid, Tanh, etc.)
4. LayerNorm / RMSNorm
5. Scaled dot-product attention (with causal mask)

**Phase 2** — Transformer ops (~200 lines):
6. Multi-head attention
7. Flash Attention (tiled online softmax)
8. Embedding + positional encoding (sinusoidal, learned, RoPE)
9. Dropout / DropPath
10. BatchNorm / GroupNorm / InstanceNorm

**Phase 3** — Advanced conv (~200 lines):
11. Winograd convolution (3×3)
12. FFT convolution (large kernels)
13. Depthwise separable convolution
14. Transposed convolution
15. Dilated convolution
16. 3D convolution

**Phase 4** — Backward passes (~200 lines):
17. Convolution backward (input grad + weight grad)
18. Attention backward (with Flash recompute)
19. Normalization backward
20. All activation gradients

---

## Composability Contract

```toml
[family_23]
name = "Neural Network Operations"
kingdom = "A (parallel matmuls, reductions) + B (causal attention, flash tiling)"

[family_23.shared_primitives]
conv = "im2col + GEMM, FFT, or Winograd"
attention = "Scaled dot-product, multi-head, flash"
normalization = "BatchNorm, LayerNorm, GroupNorm, RMSNorm"
activation = "ReLU, GELU, SiLU, Sigmoid, Tanh, etc."
pooling = "Max, Average, Global, Adaptive"

[family_23.reuses]
f02_linalg = "GEMM for linear layers and attention"
f03_signal = "FFT for large-kernel convolution"
f05_optimizer = "Training optimizers"
f06_descriptive = "Mean/variance for normalization"

[family_23.provides]
forward_ops = "All neural network forward operations"
backward_ops = "All gradient computations"
attention_variants = "Dense, causal, flash, sparse, linear"

[family_23.consumers]
f24_training = "All training loops use F23 ops"
f22_reduction = "Autoencoders"
f28_manifold = "Parametric UMAP"
```
