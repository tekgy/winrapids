# Family 25: Information Theory — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (Commutative) — all discrete IT algorithms are `scatter_add` → `accumulate(entr_expr, Sum)`

---

## Kingdom A Classification

Per the three-kingdom taxonomy, Family 25 is Kingdom A (commutative). The MSR for discrete information theory is the **histogram** (joint distribution). All entropy/divergence measures are polynomial/transcendental extractions from the histogram.

**Structure**:
1. **Phase 1**: `scatter_add` — build histogram / contingency table (the MSR)
2. **Phase 2**: `accumulate(Contiguous, f(p_i), Sum)` — extract the measure

The structural rhyme: **Random forest split = Mutual information** (same histogram → entropy pipeline).

---

## The 6 Kernel Primitives

Every algorithm in this family needs exactly these 6 elementwise functions combined with `accumulate(..., Sum)`:

| Primitive | Formula | Used by |
|-----------|---------|---------|
| `entr(p)` | -p·log(p) if p>0, 0 if p=0 | Shannon, Joint, Conditional entropy |
| `rel_entr(p,q)` | p·(log(p)-log(q)) if p>0,q>0; 0 if p=0; ∞ if q=0,p>0 | KL divergence |
| `xlogy(x,y)` | x·log(y) if x>0; 0 if x=0 | Cross entropy |
| `pow(p,α)` | p^α | Rényi, Tsallis |
| `jsd_fused(p,q)` | p·log(2p/(p+q)) + q·log(2q/(p+q)) | Jensen-Shannon |
| `digamma(x)` | ψ(x) = d/dx log(Γ(x)) | KSG estimator (continuous MI) |

All 6 are fused element-wise operations inside an additive reduction. **One scatter_multi_phi with 6 expressions covers the entire family.**

---

## 1. Shannon Entropy

### Formula

```
H(X) = -Σ p(x)·log(p(x))
```

**Numerically superior form** (for counts nᵢ with total N):
```
H = log(N) - (1/N)·Σ nᵢ·log(nᵢ)
```
This avoids N divisions inside the sum. Better for GPU f32.

### Log Base

| Package | Default | Unit |
|---------|---------|------|
| scipy.stats.entropy | e (natural) | nats |
| sklearn MI | e (natural) | nats |
| R entropy | e (natural) | nats |

**Decision**: Compute in nats. Provide base parameter. Conversion: H_base2 = H_nat / ln(2).

### Convention: 0·log(0) = 0

By L'Hôpital: lim_{p→0+} p·log(p) = 0. ALL packages enforce this. The `entr()` kernel must handle p=0 explicitly.

### Bias Corrections (from finite samples)

| Method | Formula | Best for |
|--------|---------|----------|
| Plugin (naive) | H = -Σ(nᵢ/N)·log(nᵢ/N) | Large N |
| Miller-Madow | H_MM = H_plugin + (k-1)/(2N) | Simple correction |
| James-Stein | Shrinkage toward uniform | Best bias-variance |
| Chao-Shen | Coverage-adjusted Horvitz-Thompson | Many rare events |

### Bounds

```
0 ≤ H(X) ≤ log(|support|)
```
H=0 iff deterministic. H=log(k) iff uniform over k outcomes.

### Differential Entropy (continuous)

```
h(X) = -∫ f(x)·log(f(x))dx
```

CAN be negative. Requires density estimation. Spacing estimators (Vasicek, Van Es, Ebrahimi) work on sorted data.

**GPU decomposition**: `accumulate(Contiguous, entr(p_i), Sum)`

---

## 2. Rényi Entropy

### Formula

```
H_α(X) = 1/(1-α) · log(Σ p(x)^α)     for α ≥ 0, α ≠ 1
```

### Special Cases

| α | Name | Formula |
|---|------|---------|
| 0 | Hartley | log(|support|) |
| →1 | Shannon | -Σ p·log(p) |
| 2 | Collision | -log(Σ p²) |
| →∞ | Min-entropy | -log(max p) |

### CRITICAL: Numerical Instability Near α=1

Direct formula has 0/0 form. At α = 1+10⁻¹², relative error is 1.3×10⁻⁵ (VERIFIED).

**Solution**: Taylor expansion around α=1:
```
H_α ≈ H_Shannon - (α-1)/2 · Var_p[log p] + O((α-1)²)
```
**Rule**: if |α-1| < 0.01, use Taylor expansion. Otherwise direct formula.

### Ordering

```
H_∞ ≤ H_α ≤ H_β ≤ H₀    for ∞ > α > β > 0
```

### GPU decomposition

```
(1/(1-α)) · log(accumulate(Contiguous, pow(p_i, α), Sum))
```

For α=2: `-log(accumulate(Contiguous, p_i·p_i, Sum))` — avoids pow().

---

## 3. Tsallis Entropy

### Formula

```
S_q(X) = 1/(q-1) · (1 - Σ p(x)^q)     for q > 0, q ≠ 1
```

### Non-Extensivity (KEY PROPERTY)

For independent A, B:
```
S_q(A∪B) = S_q(A) + S_q(B) + (1-q)·S_q(A)·S_q(B)
```

This is **non-additive** (unlike Shannon/Rényi). q controls non-extensivity:
- q=1: additive (Shannon)
- q<1: super-additive
- q>1: sub-additive

### Relationship to Rényi

```
S_q = (1 - exp((1-q)·H_q)) / (q-1)
```

### Special Cases

| q | Value |
|---|-------|
| 0 | k-1 |
| →1 | Shannon |
| 2 | 1-Σp² = Gini-Simpson index |

### GPU decomposition

Same as Rényi inner sum, different outer scalar.

---

## 4. KL Divergence

### Formula

```
D_KL(P‖Q) = Σ p(x)·(log(p(x)) - log(q(x)))
```

**Use log-subtraction form**, NOT log(p/q) which overflows in f32 for p/q > e^88.

### Properties

- **NOT symmetric**: D_KL(P‖Q) ≠ D_KL(Q‖P)
- **NOT a metric**: fails triangle inequality
- Non-negative: D_KL ≥ 0 (Gibbs' inequality)
- D_KL = 0 iff P = Q
- No finite upper bound

### Edge Cases

| Condition | Value | Implementation |
|-----------|-------|----------------|
| p(x) = 0 | 0 | `rel_entr` returns 0 |
| q(x) = 0, p(x) > 0 | +∞ | `rel_entr` returns +∞ |
| q(x) = 0, p(x) = 0 | 0 | `rel_entr` returns 0 |

### GPU decomposition

```
accumulate(Contiguous, rel_entr(p_i, q_i), Sum)
```

---

## 5. Jensen-Shannon Divergence

### Formula

```
JSD(P,Q) = H(M) - ½H(P) - ½H(Q)     where M = (P+Q)/2
```

**Fused single-pass form** (preferred for GPU):
```
JSD = ½ · Σ [p·log(2p/(p+q)) + q·log(2q/(p+q))]
```

### Properties

- **Symmetric**: JSD(P,Q) = JSD(Q,P) ✓
- **Bounded**: 0 ≤ JSD ≤ log(2) (nats), 0 ≤ JSD ≤ 1 (bits)
- √JSD **IS a proper metric** (triangle inequality holds). Proven by Endres & Schindelin 2003.
- JSD itself is NOT a metric (13% triangle inequality violation rate).

### Edge Cases

- Disjoint support → JSD = log(2)
- Identical distributions → JSD = 0
- One distribution is delta → JSD is finite (unlike KL which is ∞)

### Generalization to n distributions

```
JSD_π(P₁,...,Pₙ) = H(Σ πᵢ·Pᵢ) - Σ πᵢ·H(Pᵢ)
```

### GPU decomposition

```
0.5 · accumulate(Contiguous, jsd_fused(p_i, q_i), Sum)
```

---

## 6. Mutual Information

### Discrete Formula

```
I(X;Y) = H(X) + H(Y) - H(X,Y)
       = Σ_x Σ_y p(x,y)·log(p(x,y)/(p(x)·p(y)))
```

### From Contingency Table (sklearn approach)

Given contingency matrix C where C_ij = count(x=i, y=j), N = ΣC_ij:
```
MI = Σ_ij (C_ij/N)·log(N·C_ij / (C_i.·C_.j))
```

### Normalized MI (4 variants)

```
NMI_arithmetic = I(X;Y) / ((H(X)+H(Y))/2)       ← sklearn default
NMI_geometric  = I(X;Y) / √(H(X)·H(Y))
NMI_min        = I(X;Y) / min(H(X), H(Y))
NMI_max        = I(X;Y) / max(H(X), H(Y))
```

Range: [0, 1].

### Adjusted MI (for clustering comparison)

```
AMI = (MI - E[MI]) / (avg(H(U),H(V)) - E[MI])
```

E[MI] computed under hypergeometric null. **Use gammaln for log-space computation to avoid overflow.**

AMI can be negative (-0.111 for random labelings). AMI=0 means chance. AMI=1 means perfect.

### KSG Estimator (Continuous, Kraskov et al. 2004)

**KSG Algorithm 1** (sklearn's implementation):
```
I_KSG1(X;Y) = ψ(k) + ψ(N) - ⟨ψ(nₓ(i)+1) + ψ(nᵧ(i)+1)⟩
```

where:
- ψ = digamma function
- k = number of nearest neighbors (sklearn default: 5)
- εᵢ = Chebyshev distance to k-th NN in joint (X,Y) space
- nₓ(i) = count of points j where |xⱼ-xᵢ| < εᵢ (strict inequality)
- nᵧ(i) = count of points j where |yⱼ-yᵢ| < εᵢ (strict inequality)

**KSG is biased low**: k=5, N=1000, ρ=0.7 Gaussian gives MI=0.294 vs theoretical 0.337 (12% underestimate). This is typical.

**GPU bottleneck**: k-NN search. For d≤3, brute-force tiled. For d>3, approximate NN.

### GPU decomposition

**Discrete**: scatter_add_2d (build contingency table) → map-reduce over nonzero entries.

**KSG**: k-NN search → parallel digamma evaluations → reduce.

---

## 7. Conditional Entropy

```
H(Y|X) = H(X,Y) - H(X)
```

Chain rule: H(X₁,...,Xₙ) = Σᵢ H(Xᵢ|X₁,...,Xᵢ₋₁)

Properties: H(Y|X) ≥ 0, H(Y|X) ≤ H(Y), H(Y|X)=0 iff Y is deterministic function of X.

### GPU: compute as difference of two entropy accumulations.

---

## 8. Joint Entropy

```
H(X,Y) = -Σ_x Σ_y p(x,y)·log(p(x,y))
```

Shannon entropy on the flattened joint distribution.

### GPU: scatter_add_2d → Shannon entropy on result.

---

## 9. Transfer Entropy

```
TE_{X→Y} = H(Yₜ|Y_past) - H(Yₜ|Y_past,X_past)
         = I(Yₜ ; X_past | Y_past)
```

### Relationship to Granger Causality

For linear Gaussian: TE = ½·log(var(Y|Y_past) / var(Y|Y_past,X_past)). Exactly Granger.

For nonlinear: TE captures what Granger misses.

### Properties

- NOT symmetric: directional information flow
- Non-negative
- TE=0 iff X provides no predictive information beyond Y's own past

### Estimation

**Binned**: build time-shifted state vectors → 4 joint entropies → combine.
**KSG-style**: conditional MI estimation with k-NN.
**Significance**: permutation testing (shuffle X to destroy coupling).

### Hyperparameters

k (Y history length), l (X history length), bins/k_nn for estimation.

### GPU: 4 parallel scatter_add → 4 parallel entropy computations → 3 scalar operations.

---

## 10. Cross Entropy

```
H(P,Q) = -Σ p(x)·log(q(x)) = H(P) + D_KL(P‖Q)
```

### Binary Cross Entropy (ML loss)

Numerically stable form (from logits z):
```
BCE(y,z) = max(z,0) - z·y + log(1+exp(-|z|))
```

### Categorical Cross Entropy

```
CCE(y,q) = -log(q_c)     where c is true class
```

With label smoothing ε: CCE = -(1-ε)·log(q_c) - (ε/K)·Σ_k log(q_k)

### GPU: `accumulate(Contiguous, xlogy(p_i, q_i), Sum)` negated.

---

## 11. F-Divergence Family (Unifying Framework)

ALL divergences are special cases of:
```
D_f(P‖Q) = Σ q(x)·f(p(x)/q(x))
```

| Divergence | f(t) | GPU kernel |
|-----------|------|------------|
| KL | t·log(t) | rel_entr(p,q) |
| Reverse KL | -log(t) | -q·log(p/q) |
| Hellinger² | (√t-1)² | (√p-√q)² |
| χ² | (t-1)² | (p-q)²/q |
| Total Variation | |t-1|/2 | |p-q|/2 |
| JSD | specialized | jsd_fused(p,q) |

**All are `accumulate(Contiguous, f_div_term(p_i,q_i), Sum)`.** Embarrassingly parallel.

---

## 12. Numerical Stability

### Float32 vs Float64

Shannon entropy with f32: relative error ~5.7×10⁻⁸ (within f32 epsilon). For N>100K, use Kahan summation or two-pass reduction.

### The log(p/q) Problem

For KL divergence: log(p/q) overflows if p/q > e^88 (f32 max ~1.6×10³⁸). **Always use log(p)-log(q) form.**

### The 0·log(0) Convention

Must be handled in every kernel. Branch: `if p > 0.0 { -p * log(p) } else { 0.0 }`.

---

## 13. Edge Cases to Test

| Algorithm | Edge case | Expected |
|-----------|----------|----------|
| Shannon | Single bin (k=1) | H = 0 |
| Shannon | Uniform over k bins | H = log(k) |
| Shannon | One bin has p=0 | 0·log(0) = 0, no NaN |
| Rényi α→1 | Direct formula | Must use Taylor expansion |
| Rényi α=2 | Collision entropy | -log(Σp²), avoid pow() |
| KL | q(x)=0, p(x)>0 | +∞ |
| KL | Identical P=Q | 0.0 exactly |
| JSD | Disjoint support | log(2) = 0.6931... |
| JSD | Identical P=Q | 0.0 exactly |
| MI | Independent X,Y | I ≈ 0 (bias from finite sample) |
| AMI | Random labelings | AMI ≈ 0 (can be slightly negative) |
| Cross entropy | q_c = 0 (underflow) | Clamp to ε before log |
| All | Empty distribution | NaN |

---

## 14. Implementation Priority

**Phase 1** — Core discrete measures (all `accumulate` + `Sum`):
1. Shannon entropy (+ Miller-Madow bias correction)
2. KL divergence
3. Jensen-Shannon divergence (fused kernel)
4. Cross entropy (binary + categorical)

**Phase 2** — Generalized entropies:
5. Rényi entropy (all α, with Taylor expansion near α=1)
6. Tsallis entropy
7. Joint entropy, Conditional entropy (from contingency table)
8. Mutual information (discrete, from contingency table)
9. NMI (4 normalization variants)

**Phase 3** — Advanced:
10. AMI (adjusted, requires hypergeometric E[MI])
11. Transfer entropy (time-lagged MI)
12. KSG estimator (continuous MI)
13. F-divergence family (Hellinger, χ², Total Variation)
14. Differential entropy (spacing estimators)

---

## 15. Composability Contract

```toml
[algorithm]
name = "shannon_entropy"
family = "information_theory.entropy"

[inputs]
required = ["probability_distribution"]  # or counts
optional = ["base"]

[outputs]
primary = "entropy"
secondary = ["entropy_bits", "entropy_nats"]

[sufficient_stats]
consumes = ["histogram"]
produces = ["histogram"]

[sharing]
provides_to_session = ["Histogram(data_id)"]
consumes_from_session = ["Histogram(data_id)"]
auto_insert_if_missing = "scatter_add(keys, ones)"

[assumptions]
requires_sorted = false
requires_positive = true  # probabilities must be non-negative
requires_no_nan = true
minimum_n = 1
sum_must_equal_one = false  # auto-normalize from counts
```
