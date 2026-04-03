# Family 24: Training Infrastructure — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (loss = accumulate over batch) + B (LR schedules = time-indexed functions)

---

## Core Insight: Everything Composes from F05 + F23

Training = loss function (this family) + optimizer (F05) + forward/backward ops (F23). F24 provides the glue: losses, learning rate schedules, gradient manipulation, mixed precision, data augmentation.

---

## 1. Loss Functions

### 1a. Regression Losses

| Loss | Formula | Gradient | Properties |
|------|---------|----------|------------|
| MSE | (1/n)Σ(y-ŷ)² | -2(y-ŷ)/n | Standard, sensitive to outliers |
| MAE (L1) | (1/n)Σ\|y-ŷ\| | -sign(y-ŷ)/n | Robust, gradient discontinuous at 0 |
| Huber | δ²(√(1+((y-ŷ)/δ)²)-1) | Smooth L1 | Robust, smooth everywhere |
| Smooth L1 | 0.5x²/β if \|x\|<β, \|x\|-0.5β otherwise | Standard for object detection |
| Quantile | ρ_τ(y-ŷ) = (y-ŷ)(τ-I(y<ŷ)) | Asymmetric, for quantile regression |
| Log-Cosh | (1/n)Σlog(cosh(y-ŷ)) | tanh(y-ŷ)/n | Smooth approx to MAE |

### 1b. Classification Losses

**Binary Cross-Entropy**:
```
BCE = -(1/n)Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
**CRITICAL**: Use log-sum-exp trick for numerical stability:
```
BCE = (1/n)Σ[max(z,0) - z·y + log(1+exp(-|z|))]
```
where z = logit (pre-sigmoid).

**Categorical Cross-Entropy**:
```
CE = -(1/n)Σ_i Σ_c y_{ic}·log(ŷ_{ic})
```
Combined with softmax: CE = -(1/n)Σ_i [z_{i,y_i} - log(Σ_c exp(z_{ic}))] — log-softmax for stability.

**Focal Loss** (Lin et al. 2017):
```
FL = -(1/n)Σ α_t·(1-p_t)^γ·log(p_t)
```
where p_t = ŷ if y=1, 1-ŷ if y=0. Default γ=2, α=0.25.
Down-weights easy examples → better for class imbalance.

**Hinge Loss** (SVM):
```
L = (1/n)Σ max(0, 1 - y·z)    where y ∈ {-1, +1}, z = decision value
```

**Label Smoothing**:
Replace hard targets [0,0,1,0] with [ε/K, ε/K, 1-ε+ε/K, ε/K]. Standard ε=0.1.

### 1c. Metric Learning Losses

**Contrastive Loss** (Chopra et al. 2005):
```
L = (1/2)y·d² + (1/2)(1-y)·max(0, m-d)²
```
where d = ‖f(x₁)-f(x₂)‖, y=1 if same class, m = margin.

**Triplet Loss** (Schroff et al. 2015):
```
L = max(0, ‖f(a)-f(p)‖² - ‖f(a)-f(n)‖² + m)
```
where a=anchor, p=positive, n=negative, m=margin.

**Hard negative mining**: Select n = argmax_n d(a,n) within valid negatives. Critical for convergence.

**InfoNCE** (van den Oord et al. 2018):
```
L = -log(exp(sim(q,k⁺)/τ) / Σ_i exp(sim(q,k_i)/τ))
```
where τ = temperature. Used in SimCLR, CLIP, contrastive learning.

**CTC Loss** (Connectionist Temporal Classification — Graves et al. 2006):
For sequence-to-sequence alignment without explicit alignment:
```
L = -log P(label | input) = -log Σ_{π∈B⁻¹(label)} P(π | input)
```
where B⁻¹ collapses repeated characters and removes blanks. Computed via forward-backward algorithm.

---

## 2. Learning Rate Schedules

### All schedules map: (step_t, total_steps) → η_t

| Schedule | Formula | Parameters |
|----------|---------|------------|
| Constant | η₀ | η₀ |
| Step decay | η₀·γ^⌊t/s⌋ | γ=0.1, s=step_size |
| Exponential | η₀·γ^t | γ per step |
| Cosine anneal | η_min + ½(η₀-η_min)(1+cos(πt/T)) | η_min, T |
| Linear warmup | η₀·t/T_w for t<T_w, then base schedule | T_w |
| Cosine + warmup | Linear warmup → cosine anneal | T_w, T, η_min |
| Polynomial | η₀·(1-t/T)^p | p=1 (linear), p=2 (quadratic) |
| OneCycleLR | Ramp up then down (Smith 2019) | max_lr, div_factor, pct_start |
| Cyclic | Triangular cycle between η_min and η_max | cycle_size |
| ReduceOnPlateau | Reduce by factor when metric stops improving | patience, factor |

### CRITICAL: Warmup is essential for transformers
Without warmup, Adam's adaptive learning rates are poorly calibrated in early steps (small v → large updates → divergence). Linear warmup for ~5-10% of training steps is standard.

### Implementation: Pure function (step → lr). No state needed except ReduceOnPlateau.

---

## 3. Gradient Manipulation

### 3a. Gradient Clipping

**By global norm** (standard for transformers):
```
if ‖g‖ > max_norm:
    g = g · max_norm / ‖g‖
```
This preserves direction, only scales magnitude.

**By value**:
```
g = clamp(g, -max_val, max_val)
```
Changes direction — less principled but simpler.

### 3b. Gradient Accumulation
For effective batch size B_eff = B_actual × num_accumulation_steps:
```
g_accum += g_step
if step % num_accumulation_steps == 0:
    optimizer.step(g_accum / num_accumulation_steps)
    g_accum = 0
```

### 3c. Gradient Scaling (Mixed Precision)
FP16 gradients can underflow. Scale loss by factor S before backward:
```
loss_scaled = loss × S
g_fp16 = backward(loss_scaled)
g_fp32 = g_fp16.float() / S
```
Dynamic scaling: increase S until overflow detected, then halve S and skip step.

---

## 4. Mixed Precision Training (Micikevicius et al. 2018)

### Rules
1. Weights: store master copy in FP32
2. Forward pass: cast weights to FP16, compute in FP16
3. Loss: compute in FP32
4. Backward pass: FP16 activations, FP32 loss scaling
5. Weight update: FP32 (on master weights)

### Why FP16 for forward/backward
- 2× less memory → larger batch sizes
- 2× faster on tensor cores (NVIDIA)
- Gradient scaling prevents underflow

### BFloat16 vs Float16
- BF16: same exponent range as FP32 (8 bits), less mantissa (7 bits). No loss scaling needed.
- FP16: more mantissa (10 bits), smaller range (5-bit exponent). Needs loss scaling.
- **BF16 is preferred when available** (Ampere+, Apple M-series).

---

## 5. Data Augmentation (for future GPU implementation)

### Image
- Random crop, horizontal flip, rotation, color jitter
- Cutout (random erasing)
- Mixup: x̃ = λ·x₁ + (1-λ)·x₂, ỹ = λ·y₁ + (1-λ)·y₂, λ ~ Beta(α,α)
- CutMix: patch of one image pasted onto another
- RandAugment: randomly select N augmentations from a pool, each with magnitude M

### Tabular / Time Series (relevant for fintek)
- Random noise injection
- Time warping (stretch/compress segments)
- Window slicing
- Magnitude warping

---

## 6. Distributed Training Primitives

### AllReduce (gradient synchronization)
```
g_synchronized = (1/P) Σ_{p=1}^{P} g_p
```
where P = number of GPUs. Ring-AllReduce is O(N/P) bandwidth-optimal.

### Gradient Compression
- Top-K sparsification: send only largest K% of gradients
- Error feedback: accumulate unsent gradients for next round
- PowerSGD: low-rank approximation of gradient

---

## 7. Regularization Techniques

### Weight Decay (already in F05 AdamW)
### Dropout (already in F23)

### Label Smoothing (Section 1b above)

### Stochastic Depth (DropPath — Huang et al. 2016)
Randomly drop entire residual blocks during training:
```
x_out = x + drop_path(f(x))
```

### R-Drop (Liang et al. 2021)
Minimize KL divergence between two forward passes with different dropout masks:
```
L = CE(y, p₁) + CE(y, p₂) + α·KL(p₁ || p₂)
```

---

## Sharing Surface

### Reuse from Other Families
- **F05 (Optimization)**: ALL optimizers (Adam, AdamW, SGD, L-BFGS)
- **F23 (Neural Network Ops)**: ALL forward/backward operations
- **F06 (Descriptive)**: Running statistics for BatchNorm (EMA of mean/var)
- **F25 (Information Theory)**: Cross-entropy, KL divergence as loss functions
- **F04 (RNG)**: Dropout masks, data augmentation randomness

### Consumers of F24
- Everything that trains a neural network
- F28 (Manifold): Parametric UMAP training
- F22 (Dimensionality Reduction): Autoencoder training
- F21 (Classification): Neural classifier training

### Structural Rhymes
- **Cross-entropy = KL divergence + constant**: same as F25 information theory
- **Focal loss = reweighted CE**: same concept as F06 weighted statistics
- **Gradient clipping = projection onto norm ball**: same as F05 projected gradient
- **Gradient accumulation = distributed accumulate**: same as F06 distributed MomentStats
- **Mixup = interpolation between samples**: same as F31 linear interpolation
- **LR warmup = ramp function**: same as F03 FIR filter ramp

---

## Implementation Priority

**Phase 1** — Core training loop (~150 lines):
1. Loss functions: MSE, MAE, Huber, BCE (with log-sum-exp), CE (with log-softmax)
2. LR schedules: constant, step, cosine, linear warmup, cosine+warmup
3. Gradient clipping (by norm, by value)
4. Gradient accumulation
5. Basic training loop orchestration

**Phase 2** — Advanced losses (~100 lines):
6. Focal loss, label smoothing
7. Contrastive, triplet, InfoNCE
8. Hinge loss
9. Quantile loss

**Phase 3** — Mixed precision + regularization (~100 lines):
10. FP16/BF16 forward pass, FP32 weight update
11. Dynamic loss scaling
12. Stochastic depth (DropPath)
13. R-Drop

**Phase 4** — Data augmentation + distributed (~100 lines):
14. Mixup, CutMix
15. Random augmentation pipeline
16. AllReduce gradient synchronization
17. CTC loss

---

## Composability Contract

```toml
[family_24]
name = "Training Infrastructure"
kingdom = "A (loss = accumulate over batch) + B (LR schedule = time function)"

[family_24.shared_primitives]
loss_fn = "Loss function: data → scalar"
lr_schedule = "LR schedule: step → learning_rate"
gradient_clip = "Clip gradient by norm or value"
gradient_accumulate = "Accumulate gradients across micro-batches"
mixed_precision = "FP16 forward, FP32 update, dynamic loss scaling"

[family_24.reuses]
f05_optimizer = "Adam, AdamW, SGD — the optimization engine"
f23_nn_ops = "Forward/backward neural network operations"
f06_descriptive = "Running statistics for batch normalization"
f25_information = "Cross-entropy, KL divergence"
f04_rng = "Dropout, augmentation randomness"

[family_24.provides]
training_loop = "Complete training orchestration"
loss_functions = "MSE, MAE, BCE, CE, focal, contrastive, triplet, InfoNCE, CTC"
lr_schedules = "Constant, step, cosine, warmup, cyclic, OneCycle, plateau"
gradient_ops = "Clipping, accumulation, scaling"
augmentation = "Mixup, CutMix, RandAugment"

[family_24.consumers]
everything = "Any model that trains via gradient descent"
```
