# Information Theory — Complete Variant Catalog

## What Exists (tambear::information_theory)

### Entropy (3 implemented)
- `shannon_entropy(probs)` — H(X) = -Σ pᵢ log(pᵢ)
- `renyi_entropy(probs, alpha)` — H_α(X), handles α∈{0, 1-limit, 2, ∞}
- `tsallis_entropy(probs, q)` — S_q(X) = (1/(q-1))(1 - Σ pᵢ^q)

### Divergences (3 implemented)
- `kl_divergence(p, q)` — D_KL(P||Q)
- `js_divergence(p, q)` — D_JS(P,Q), symmetric
- `cross_entropy(p, q)` — H(P,Q) = -Σ pᵢ log(qᵢ)

### Mutual Information (6 implemented)
- `mutual_information(contingency, nx, ny)`
- `normalized_mutual_information(contingency, nx, ny, method)` — max/arithmetic/geometric
- `adjusted_mutual_info_score(labels_true, labels_pred)` — AMI with expected MI correction
- `variation_of_information(contingency, nx, ny)` — VI = H(X|Y) + H(Y|X)
- `conditional_entropy(contingency, nx, ny)` — H(Y|X)
- `mutual_info_miller_madow(contingency, nx, ny)` — bias-corrected MI

### Estimators (3 implemented)
- `entropy_histogram(data, n_bins)` — plug-in from histogram
- `fisher_information_histogram(values, n_bins)` — Fisher info + distance
- `grassberger_entropy(data, n_bins)` — digamma bias correction

### Other (3 implemented)
- `transfer_entropy(x, y, n_bins)` — Schreiber 2000
- `tfidf(counts, n_docs, n_terms, smooth, sublinear)` — TF-IDF
- `cosine_similarity(a, b)` / `cosine_similarity_matrix(data, n, d)`

---

## What's MISSING — Complete Catalog by Sub-Family

### A. Missing Entropy Variants

1. **Differential entropy** — H(X) = -∫ f(x) log f(x) dx for continuous distributions
   - Parameters: `data: &[f64]`, `bandwidth: Option<f64>` (KDE-based)
   - Primitives: KDE → integrate p(x) log p(x) dx
   - Assumptions: continuous distribution, sufficient samples for KDE
   - Note: negative values are valid (unlike discrete entropy)

2. **Joint entropy** — H(X,Y) = -Σ p(x,y) log p(x,y)
   - Already computable from contingency table but no standalone function
   - Should be a first-class primitive since MI = H(X) + H(Y) - H(X,Y)

3. **Conditional Rényi entropy** — H_α(Y|X)
   - Generalizes conditional Shannon entropy to Rényi order α
   - Parameters: `contingency`, `alpha`

4. **Conditional Tsallis entropy** — S_q(Y|X)
   - Parameters: `contingency`, `q`

5. **Permutation entropy** — already in complexity.rs but should be cross-referenced
   - Ordinal pattern distribution → Shannon entropy on pattern frequencies

6. **Weighted entropy** — H_w(X) = -Σ wᵢ pᵢ log(pᵢ)
   - Parameters: `probs`, `weights`
   - Use case: financial data where outcomes have different magnitudes

7. **Cumulative residual entropy** (CRE) — Rao et al. 2004
   - ε(X) = -∫₀^∞ P(X > x) log P(X > x) dx
   - Advantages: defined for distributions without density, more stable

8. **Extropy** — Lad, Sanfilippo, Agro 2015
   - J(X) = -Σ (1-pᵢ) log(1-pᵢ) / (k-1)
   - Complementary dual of entropy

9. **Quadratic entropy** (Rao) — Q(X) = Σᵢ Σⱼ dᵢⱼ pᵢ pⱼ
   - Parameters: `probs`, `distance_matrix`
   - Incorporates metric structure on the sample space

10. **Topological entropy** — rate of exponential growth of orbits
    - Relevant for dynamical systems (complement to Lyapunov exponents)

### B. Missing Divergences

1. **Hellinger distance** — H²(P,Q) = ½ Σ (√pᵢ - √qᵢ)²
   - Bounded [0,1], symmetric, satisfies triangle inequality
   - Shares intermediates: sqrt of probabilities

2. **Total variation distance** — TV(P,Q) = ½ Σ |pᵢ - qᵢ|
   - Upper bound for many statistical distances

3. **Chi-squared divergence** — χ²(P||Q) = Σ (pᵢ - qᵢ)² / qᵢ
   - Related to KL: D_KL ≤ χ²/2 (Pinsker-type bound)

4. **f-divergences** (general) — D_f(P||Q) = Σ qᵢ f(pᵢ/qᵢ)
   - Unifying family: KL, reverse KL, chi-squared, Hellinger, TV all special cases
   - Parameters: `p`, `q`, `f: Fn(f64) -> f64`
   - Named instances:
     - f(t) = t log t → KL
     - f(t) = -log t → reverse KL
     - f(t) = (t-1)² → chi-squared
     - f(t) = (√t - 1)² → squared Hellinger
     - f(t) = |t-1|/2 → total variation
     - f(t) = t^α / (α(α-1)) → alpha-divergence

5. **Rényi divergence** — D_α(P||Q) = (1/(α-1)) log Σ pᵢ^α qᵢ^(1-α)
   - Parameters: `p`, `q`, `alpha`
   - Limit α→1 gives KL divergence
   - Limit α→∞ gives max-divergence

6. **Wasserstein distance** (Earth Mover's Distance)
   - W_p(P,Q) = (inf_γ ∫ d(x,y)^p dγ(x,y))^(1/p)
   - For 1D: W₁ = ∫|F_P(x) - F_Q(x)|dx (from sorted CDFs)
   - Parameters: `p`, `q`, `p_order` (typically 1 or 2)
   - 1D case is O(n log n) via sorting — a clean primitive

7. **Maximum Mean Discrepancy** (MMD)
   - MMD²(P,Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
   - Parameters: `x_samples`, `y_samples`, `kernel` (default: Gaussian RBF)
   - Widely used in two-sample testing and generative model evaluation

8. **Energy distance** — E(P,Q) = 2E||X-Y|| - E||X-X'|| - E||Y-Y'||
   - Parameters: `x_samples`, `y_samples`
   - Metric, analogous to energy in physics

9. **Bregman divergence** — D_φ(p,q) = φ(p) - φ(q) - ⟨∇φ(q), p-q⟩
   - Parameters: `p`, `q`, `phi: Fn`, `grad_phi: Fn`
   - KL divergence is Bregman with φ = Σ pᵢ log pᵢ
   - Squared Euclidean is Bregman with φ = ||x||²

10. **Bhattacharyya distance** — D_B = -ln(Σ √(pᵢ qᵢ))
    - Related to Hellinger: H² = 1 - exp(-D_B)

### C. Missing Mutual Information Variants

1. **Interaction information** (co-information)
   - I(X;Y;Z) = I(X;Y) - I(X;Y|Z) — can be negative (synergy)
   - Parameters: `contingency_3d`, `nx`, `ny`, `nz`

2. **Total correlation** (multi-information)
   - TC(X₁,...,Xₙ) = Σ H(Xᵢ) - H(X₁,...,Xₙ)
   - Parameters: `joint_contingency`, `dims: &[usize]`

3. **Dual total correlation** (binding information)
   - DTC = H(X₁,...,Xₙ) - Σ H(Xᵢ|X_{-i})
   - Measures the shared, non-redundant information

4. **Partial information decomposition** (PID) — Williams & Beer 2010
   - Decomposes MI into: unique to X, unique to Y, redundant, synergistic
   - Multiple proposed definitions (I_min, I_broja, I_ccs)
   - Advanced but increasingly used in neuroscience & finance

5. **Directed information** — I(X^n → Y^n) = Σ I(X^t; Y_t | Y^{t-1})
   - Causal version of MI, measures directed channel capacity
   - Parameters: `x`, `y`, `max_lag`

6. **Pointwise mutual information** (PMI) — log p(x,y)/(p(x)p(y))
   - Returns per-cell values, not aggregated
   - PPMI (positive PMI) = max(0, PMI) — used in NLP

7. **Conditional MI** — I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
   - Parameters: `contingency_3d`, `nx`, `ny`, `nz`

### D. Missing Estimators

1. **KSG estimator** — Kraskov, Stögbauer, Grassberger 2004
   - k-nearest-neighbor MI estimator
   - Avoids binning entirely — works in continuous space
   - Two variants: KSG1 (max-norm digamma), KSG2 (average digamma)
   - Parameters: `x`, `y`, `k` (neighbors, default 3-5)
   - This is the gold-standard MI estimator for continuous data

2. **Kozachenko-Leonenko entropy** — k-NN entropy estimator
   - H_KL = (d/n) Σ log(2 ε_i) + log(n) - ψ(k) + log(V_d)
   - Parameters: `data: &[&[f64]]` (multivariate), `k`
   - Shares: distance computation, kd-tree

3. **Jackknife entropy** — bias correction via leave-one-out
   - H_jack = n × H_all - (n-1)/n × Σ H_{-i}
   - Parameters: `counts`, `method` (delete-one or delete-d)

4. **NSB estimator** — Nemenman, Shafee, Bialek 2001
   - Bayesian entropy estimator with Dirichlet prior
   - Best for very undersampled distributions
   - Parameters: `counts`, `prior_alpha` (default: automatic)

5. **Shrinkage entropy** — James-Stein estimator applied to probabilities
   - Shrinks plug-in frequencies toward uniform
   - Parameters: `counts`, `shrinkage_intensity` (default: auto via Hausser-Strimmer)

6. **Chao-Shen entropy** — coverage-adjusted estimator
   - Uses Good-Turing coverage: C = 1 - f₁/n
   - Adjusts for unseen species
   - Parameters: `counts`

### E. Missing Coding Theory / Compression

1. **Kolmogorov complexity approximation** — via compression ratio
   - K(x) ≈ length(compress(x)) / length(x)
   - Parameters: `data`, `compressor` (LZ77, gzip, bzip2)
   - Normalized compression distance: NCD(x,y) = (C(xy) - min(C(x),C(y))) / max(C(x),C(y))

2. **Normalized information distance** — NID(x,y) = max(K(x|y), K(y|x)) / max(K(x), K(y))
   - Approximated via compression

3. **Rate-distortion function** — R(D) = min_{p(y|x)} I(X;Y) s.t. E[d(X,Y)] ≤ D
   - Parameters: `source_dist`, `distortion_fn`, `max_distortion`
   - Iterative computation (Blahut-Arimoto algorithm)

4. **Blahut-Arimoto algorithm** — compute channel capacity
   - C = max_p(x) I(X;Y) for given channel p(y|x)
   - Parameters: `channel_matrix`, `tol`, `max_iter`

### F. Missing Cryptographic Entropy

1. **Min-entropy** — H_∞(X) = -log max pᵢ
   - Already handled as renyi_entropy(probs, f64::INFINITY) but deserves standalone

2. **Collision entropy** — H₂(X) = -log Σ pᵢ²
   - Already handled as renyi_entropy(probs, 2.0) but deserves standalone

3. **Guessing entropy** — G(X) = Σᵢ i × p_{(i)}
   - Expected number of guesses to determine X (sorted probabilities)

4. **Marginal guesswork** — min expected computation for brute force

---

## Intermediate Sharing Map

```
histogram counts ──┬── shannon_entropy
                   ├── renyi_entropy (all α)
                   ├── tsallis_entropy (all q)
                   ├── grassberger_entropy
                   ├── jackknife_entropy
                   ├── nsb_entropy
                   ├── shrinkage_entropy
                   ├── chao_shen_entropy
                   └── fisher_information

joint histogram ───┬── mutual_information
                   ├── conditional_entropy
                   ├── variation_of_information
                   ├── normalized_mi
                   ├── adjusted_mi
                   └── pointwise_mi

marginal probs ────┬── kl_divergence
                   ├── js_divergence
                   ├── hellinger_distance
                   ├── total_variation
                   ├── chi2_divergence
                   ├── renyi_divergence
                   ├── bhattacharyya_distance
                   ├── f_divergence (general)
                   └── cross_entropy

k-NN distances ────┬── ksg_mi (KSG1, KSG2)
                   ├── kozachenko_leonenko_entropy
                   └── transfer_entropy (k-NN variant)

sorted CDF ────────── wasserstein_1d

kernel matrix ─────┬── mmd
                   └── energy_distance

sqrt(probs) ───────┬── hellinger_distance
                   └── bhattacharyya_distance
```

## Priority for Implementation

**Tier 1 — High impact, straightforward** (primitives that unlock many consumers):
1. `joint_entropy` — trivial from contingency, completes H decomposition
2. `hellinger_distance` — simple formula, widely used
3. `total_variation_distance` — simple formula, widely used
4. `differential_entropy` — continuous analog, uses KDE (already have it)
5. `wasserstein_1d` — O(n log n) via sort, very useful distance
6. `pointwise_mi` — per-cell PMI, important for NLP/feature analysis

**Tier 2 — High impact, moderate complexity**:
7. `ksg_mi` — gold-standard continuous MI, needs k-NN infrastructure
8. `renyi_divergence` — generalizes KL, natural family extension
9. `f_divergence` — unifying framework, parameterized by generator
10. `conditional_mi` — needed for PID, causal analysis
11. `mmd` — two-sample testing, generative models
12. `energy_distance` — clean metric for distribution comparison

**Tier 3 — Important but specialized**:
13-17: NSB estimator, Chao-Shen, interaction information, directed information, Blahut-Arimoto

## Assumptions and Failure Modes

| Function | Key Assumption | Failure Mode |
|----------|---------------|--------------|
| shannon_entropy | probs sum to 1, ≥ 0 | Negative probs → nonsense |
| kl_divergence | Q covers P's support | p>0 where q=0 → +∞ |
| transfer_entropy | stationarity, sufficient bins | Too few bins → bias, too many → variance |
| ksg_mi | IID samples, smooth density | Discrete data → k-NN degeneracy |
| wasserstein_1d | Same support space | Different units → meaningless |
| mmd | Characteristic kernel | Non-characteristic kernel → can't distinguish |
| differential_entropy | Smooth density exists | Discrete/singular → undefined |
