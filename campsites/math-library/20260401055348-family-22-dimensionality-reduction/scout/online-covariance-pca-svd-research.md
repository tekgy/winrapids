# Online Covariance / Online PCA / Tambear-Native SVD — Research

Created: 2026-04-01
By: scout (research task #10)
Session: Tambear math library expedition

---

## Purpose

Research all streaming/online algorithms for covariance, PCA, and SVD that are
potentially expressible via tambear's accumulate(grouping, expr, op) primitive.
Analyze what sufficient statistics each needs, what the merge operators look like,
and whether each is structurally native to tambear or requires sequential iteration.

The central question: **can we compute PCA/SVD using only accumulate(grouping, expr, op)
plus a deterministic post-step (eigendecomposition / SVD of a small matrix)?**

Short answer: **yes, for the covariance path (exact PCA), and approximately yes for
randomized SVD. Oja's rule and incremental SVD are NOT accumulate-native — they require
sequential state propagation.**

---

## Part 1: Online Covariance — Fully Accumulate-Native

### 1.1 The Sufficient Statistics

To compute the full p×p covariance matrix of n observations x_i ∈ ℝᵖ, the exact
sufficient statistics are:

```
S = {
    n:   count (scalar)
    μ:   mean vector (p-vector) = (1/n) Σ x_i
    C:   cross-product accumulator (p×p matrix) = Σ (x_i - μ_n)(x_i - μ_n)'
}

From S, recover:
  μ   = S.mean
  Σ_pop  = C / n          (population covariance)
  Σ_samp = C / (n-1)      (sample covariance, Bessel-corrected)
```

That's it. Three objects total for the full covariance matrix. The mean is p numbers.
The cross-product accumulator C is p² numbers (symmetric, so p(p+1)/2 distinct).

### 1.2 Welford's Online Update (Sequential)

For a single new observation x_{n+1}:

```
Algorithm (Welford 1962, multivariate extension):

  δ₁ = x_{n+1} − μ_n           (delta before update)
  μ_{n+1} = μ_n + δ₁/(n+1)     (update mean)
  δ₂ = x_{n+1} − μ_{n+1}       (delta after update)

  C_{n+1} = C_n + δ₁ ⊗ δ₂      (outer product update of cross-product accumulator)

where ⊗ = outer product (δ₁ᵀ ⊗ δ₂ is p×p)
```

Key: δ₁ ⊗ δ₂ = (x - old_mean)(x - new_mean)' — uses BOTH before/after mean deltas.
This is numerically stable because the two deltas bracket the mean update.

Univariate case: M₂ += δ₁ · δ₂ (scalar; this is the Welford M₂ accumulator).
Sample variance = M₂/(n-1). Population variance = M₂/n.

### 1.3 Chan's Parallel Merge Formula (Tambear-Native)

This is the key formula that makes covariance accumulate-native. Given two sets A and B:

```
n_A, μ_A, C_A    (statistics for set A)
n_B, μ_B, C_B    (statistics for set B)

Merge into AB:
  n_AB = n_A + n_B
  δ    = μ_B − μ_A                     (mean difference vector, p-vector)
  μ_AB = μ_A + n_B/(n_A + n_B) · δ    (weighted mean update)
  C_AB = C_A + C_B + (n_A·n_B/n_AB) · δ ⊗ δ   (covariance merge)
```

The C_AB formula is exact — no approximation. It works for any partition of the data.
The δ ⊗ δ term corrects for the mean shift between the two groups.

**This is exactly the tambear accumulate combine operator for the GramMatrix-of-centered
data operation.** Derivation:

```
GramMatrix of centered X = X̃'X̃  where X̃ = X - 1μ'
                        = Σ_i (x_i - μ)(x_i - μ)'
                        = C (the cross-product accumulator)

So: C = accumulate(All, phi=(x - mean_x) ⊗ (x - mean_x), Add)

But this requires knowing the global mean first (two-pass for centering).
One-pass version: use the (n, μ, C) sufficient stats with the merge formula.
```

### 1.4 Tambear Expression of Covariance Accumulation

```
// Pass 1: accumulate sufficient statistics
stats = accumulate(
    grouping = All,           // or ByKey{group_id} for grouped covariance
    phi      = (n=1, mu=x, C=0),   // per-row: count 1, self-mean, zero cross-product
    op       = WelfordCovOp,  // merge via Chan formula above
)

// Post-step (not accumulate — deterministic given stats):
covariance_matrix = stats.C / (stats.n - 1)   // sample covariance

// Full PCA from covariance:
eigenvalues, eigenvectors = eigh(covariance_matrix)   // symmetric eigendecomposition
```

The WelfordCovOp is the combine operator: given (n_A, μ_A, C_A) and (n_B, μ_B, C_B),
produce (n_AB, μ_AB, C_AB) using Chan's formula. This IS a valid semigroup with
associativity and identity (n=0, μ=0, C=0).

**Proof it's a semigroup combine:**
- Associative: merge(merge(A,B), C) == merge(A, merge(B,C)) ✓ (Chan 1979 proves this)
- Identity: (0, 0, 0) is the left/right identity ✓
- No ordering requirement: merge(A,B) = merge(B,A) ✓ (commutative)

This makes it a **commutative monoid** → fully compatible with tambear's parallel prefix
scan model. Can be computed in O(log N) parallel steps.

### 1.5 What plm/sklearn Use for Covariance

```python
# sklearn.covariance.EmpiricalCovariance:
# One-pass with precomputed mean, then X̃'X̃/(n-1) — two arrays in memory
# NOT streaming

# numpy.cov():
# np.cov(X.T) = X̃'X̃/(n-1) — same, loads all data

# The streaming approach (Chan's formula) is used in:
# - sklearn.preprocessing.StandardScaler with partial_fit()
# - sklearn.decomposition.IncrementalPCA (for mean tracking)
# - Dask array statistics
# - TensorFlow streaming statistics
```

---

## Part 2: Streaming PCA — What's Accumulate-Native vs What's Not

### 2.1 The Covariance Path (Exact PCA — Accumulate-Native)

```
1. Accumulate: (n, μ, C) using Chan merge     ← ONE PASS, accumulate-native
2. Compute:    Σ = C/(n-1)                    ← O(p²), trivial
3. Decompose:  Σ = Q Λ Q'   (eigendecomposition)  ← O(p³), serial, small matrix
4. Output:     Q[:, :k] = top-k eigenvectors (principal components)
```

Steps 1 is accumulate-native. Steps 2-4 are a deterministic post-step on a p×p matrix.
For the WinRapids use case (p = number of features, typically p << n), this is optimal.

**This is the standard batch PCA algorithm**, and it's perfectly tambear-native for the
accumulation step. The bottleneck is step 1 (one GPU pass over n×p data). Step 3 is
negligible for p in the hundreds.

Memory: O(p²) for C. For p=1000, C is 1M floats = 4 MB. Trivial.

### 2.2 Oja's Rule (Online PCA — NOT Accumulate-Native)

Oja (1982) proposed the following online update for the leading eigenvector w ∈ ℝᵖ:

```
For each observation x_t:
  y_t = w_t' · x_t                    (scalar projection)
  Δw  = η_t · y_t · (x_t - y_t · w_t)   (Hebb-anti-Hebb update)
  w_{t+1} = w_t + Δw                  (normalized implicitly by the rule)

Continuous-time version:
  dw/dt = E[y·(x - y·w)] = E[xx'w] - E[y²]·w = (Σ - λ₁)·w
  where λ₁ = largest eigenvalue
```

Learning rate schedule: η_t = η₀/t satisfies Σηₜ = ∞ and Ση²_t < ∞ (Robbins-Monro).

Convergence: w_t → q₁ (first principal component, eigenvector of largest λ₁) as t→∞.

**Why NOT accumulate-native:**
- The update Δw depends on the current w_t, which changes after each observation
- This is sequential state — each step depends on the previous w
- You cannot reorder observations and get the same result
- It's an SGD-type algorithm, not a reduction

**Generalized Hebbian Algorithm (GHA/Sanger 1989) for top-k components:**
```
For each x_t:
  y = W_t' · x_t                         (k-vector of projections)
  Δw_j = η · y_j · (x_t - Σ_{l≤j} y_l · w_l)   (deflation of lower components)
  W_{t+1} = W_t + ΔW
```
Same sequential structure — NOT accumulate-native.

**Verdict**: Oja's rule is useful when p is very large (too large to store p×p covariance)
and only a few components are needed. For WinRapids (p moderate, n large), the covariance
path (Section 2.1) is faster, exact, and accumulate-native.

### 2.3 sklearn IncrementalPCA (Ross et al. 2008)

Implements Ross, Lim, Lin, Yang (2008) "Incremental Learning for Robust Visual Tracking."

**Algorithm (batch update):**

Given existing SVD of A_old → U_old, S_old, V_old (shape m×r), and new batch X_new (shape b×p):

```
Step 1: Update mean
  n_old = number of rows in A_old
  n_new = b (batch size)
  μ_new = (n_old·μ_old + n_new·μ_batch) / (n_old + n_new)

Step 2: Build augmented matrix
  mean_correction = sqrt(n_old·n_new/(n_old+n_new)) · (μ_old - μ_batch)
  X̃_new = X_new - μ_batch   (center new batch by its mean)

  Construct [X_old_approx; X̃_new; mean_correction] where
  X_old_approx = U_old · diag(S_old)   (reconstruct old data from SVD, shape m×r)

Step 3: SVD of augmented matrix
  [U_old·S_old | X̃_new | mean_correction]  → thin SVD → new U, S, V

Step 4: Truncate to k components
  Keep top k singular values/vectors
```

**Memory**: O(batch_size × p) — only the current batch is in memory, not all n rows.

**Why partially accumulate-native:**
- The mean update IS Chan's formula (step 1 is accumulate-native)
- The SVD augmentation (steps 2-4) requires maintaining the current SVD state (U, S, V)
  and doing a fresh SVD each batch — NOT a pure accumulate

This is closer to a sequential fold over batches than a parallel reduce. It CAN be
parallelized by merging SVDs of independent shards (Brand 2002), but that merge is
non-trivial.

---

## Part 3: Brand (2002) Incremental SVD — Sequential, But Mergeable

### 3.1 Core Idea

Brand (2002) "Incremental SVD of uncertain data" (ECCV 2002, MERL TR-2002-24).

Given existing SVD: A = U S V' (m×n matrix, rank r)
Adding new column c (m×1 vector):

```
[A | c] = [U S V' | c]

Decompose: c = U·(U'c) + p  where p = c - U·(U'c)   (orthogonal decomposition)
                               U'c  = U's projection onto existing basis
                               p    = residual perpendicular to existing basis

Let: a = U'c   (r-vector, projection onto existing basis)
     p = c - U·a   (residual perpendicular to U)
     p_norm = ||p||
     P = p/p_norm   (unit vector in residual direction, if p_norm > 0)

Construct small (r+1)×(r+1) extended matrix:
  K = [ S    a  ]
      [ 0  p_norm ]   // just the bottom-right scalar if adding one column

SVD of K: K = Ũ S̃ Ṽ'   (small, r+1 × r+1)

Update:
  U_new = [U | P] · Ũ     (m × (r+1))
  S_new = S̃                ((r+1) × (r+1))
  V_new = [V 0; 0 1] · Ṽ  ((n+1) × (r+1))
```

Then truncate to rank k if desired.

**Cost per update**: O(mr + r³) — linear in m×r (data dimension × rank), cubic in r (small).

### 3.2 Tambear Assessment

Brand's update IS sequential (each new column changes U), but the key structure is:
- The full U doesn't need to be stored for the sufficiency — only U (current subspace)
- Brand showed you can MERGE two SVDs: given SVD(A) and SVD(B), compute SVD([A|B])
  using the same small-matrix trick

The merge formula makes it compatible with tambear's parallel scan model IF:
- We treat each batch as a "block" with its own SVD
- Then merge-reduce using Brand's merge operator

**However**: the merge of two SVDs (Brand's pairwise merge) is NOT the same as Chan's
covariance merge. It requires a thin SVD of an (r₁+r₂)×(r₁+r₂) matrix — cheap for
small rank r, but not a simple monoid combine like covariance.

**Verdict**: Brand's algorithm CAN be implemented as a parallel-merge-reduce in tambear,
but it's not a clean semigroup (the merge cost scales with r², not O(1)). For typical
use: prefer the covariance path (Section 2.1) unless p is too large for p×p storage.

---

## Part 4: Halko-Martinsson-Tropp (2011) Randomized SVD

Reference: "Finding structure with randomness: Probabilistic algorithms for constructing
approximate matrix decompositions." SIAM Review 53(2), 2011.

### 4.1 Algorithm

Goal: approximate top-k SVD of A (n×p matrix), without forming the full n×p matrix.

```
Stage 1: Range Finder (Pass 1 over data)
  Generate: Ω ∈ ℝ^{p×(k+p)} — Gaussian random matrix, oversampling by p=5-10
  Compute:  Y = A·Ω   (n×(k+p) matrix — "sketched" data)
                       ← this requires one pass over A
  QR:       Y = Q·R   (thin QR, get Q ∈ ℝ^{n×(k+p)}, orthonormal columns)

Stage 2: Reduction (Pass 2 over data)
  Form:     B = Q'·A  ((k+p)×p matrix — tiny!)
                       ← this requires a second pass over A

Stage 3: Deterministic SVD (on tiny B)
  SVD:      B = Ũ·Σ·V'
  Recover:  U = Q·Ũ   (← the left singular vectors of A)
  Output:   A ≈ U[:, :k] · Σ[:k, :k] · V[:k, :]'
```

Total: **2 passes** over A. Space: O(n·(k+p) + p·(k+p)) — much less than n×p for large n.

### 4.2 Power Iteration for Accuracy (when singular values decay slowly)

For matrices with slowly decaying singular values (e.g., dense covariance matrices):

```
Replace Y = A·Ω  with  Y = (A·A')^q · A·Ω   (q power iterations)

In practice (with QR re-orthogonalization each step):
  Y₀ = A·Ω
  for q steps:
    (Q₀, _) = QR(Y₀)
    Y₁ = A'·Q₀
    (Q₁, _) = QR(Y₁)
    Y₀ = A·Q₁
  Final Y = Y₀

Each power iteration = 1 additional pass over A.
Total passes = 2 + 2q.
```

### 4.3 Tambear Assessment — Can This Be Done with accumulate?

Stage 1 (Y = A·Ω): This is a matrix multiply A·Ω. In tambear terms:
```
Y_j = accumulate(All, phi = x_i · Ω[i, j], Add)   for each column j of Ω

Or equivalently:
Y = accumulate(All, phi = outer(x_i, Omega_j), Add)   // where Omega_j is the j-th row of Ω
```
This IS accumulate-native. It's a GramMatrix-style accumulation:
```
Y = accumulate(Tiled(n, k+p), DotProduct(row_i, omega_j), Add)
```

Stage 2 (B = Q'·A): After computing Q (from QR of Y, a purely deterministic step),
```
B[l, j] = Σ_i Q[i,l] · A[i,j]
B = accumulate(All, phi = outer(Q_row_i, x_i), Add)   // outer product
```
This is ALSO accumulate-native (it's another GramMatrix-style operation, but with the
Q weights that come from the deterministic QR step).

**Key insight: both passes over A are accumulate(All, weighted_outer_product, Add).**

The QR step in between is a tiny (n×(k+p)) deterministic operation on the result.

So: **Randomized SVD is accumulate-native**, with two accumulation passes:
```
Pass 1:  Y    = accumulate(All, phi=outer(x, omega_j),  Add)   // n×(k+p) sketch
         Q, _ = qr(Y)                                          // deterministic, tiny
Pass 2:  B    = accumulate(All, phi=outer(q_i(x), x),   Add)   // (k+p)×p reduction
         U_small, S, V = svd(B)                                // deterministic, tiny
         U = Q · U_small                                        // final left sing. vecs
```

The "phi" in Pass 1 is x → outer(x, Ω·x) where Ω is fixed random projection.
The "phi" in Pass 2 is x → outer(Q'·x, x) where Q is the fixed orthonormal basis.

**Both passes are tambear GramMatrix-style accumulations.** This is the best answer to
"can we do SVD with accumulate primitives?" — YES, via randomized SVD, 2 passes.

### 4.4 Comparison: Exact PCA vs Randomized SVD

| Property | Exact PCA (covariance path) | Randomized SVD (HMT 2011) |
|----------|----------------------------|---------------------------|
| Passes over data | 1 | 2 (or 2+2q) |
| Accumulate-native | Yes (Chan covariance) | Yes (two weighted outer products) |
| Result | Exact | Approximate (error bounded) |
| Memory | O(p²) covariance matrix | O(n·(k+p)) sketch matrix |
| Good when | p moderate, want exact | p >> k, want fast approx |
| GPU-friendly | Yes (tiled GramMatrix) | Yes (two matrix multiplies) |
| Post-step | Eigendecomposition of p×p | SVD of (k+p)×p tiny matrix |

For WinRapids (p = feature count, k = components wanted):
- p < 5000: exact PCA dominates — one pass, exact, trivial post-step
- p > 50000, k << p: randomized SVD — two passes, excellent accuracy

---

## Part 5: Summary — Tambear-Native Classification

### Fully Accumulate-Native (one pass, parallel reduce)

```
┌──────────────────────────────────────────────────────────────────┐
│  Algorithm           │  Sufficient Stats      │  Merge Op         │
├──────────────────────┼────────────────────────┼───────────────────┤
│  Online covariance   │  (n, μ, C)             │  Chan's formula   │
│  Batch PCA (exact)   │  (n, μ, C) → eigdecomp │  Chan's formula   │
│  Randomized SVD p1   │  Y = A·Ω sketch        │  Sum outer prods  │
│  Randomized SVD p2   │  B = Q'·A              │  Sum outer prods  │
│  Weighted covariance │  (n, μ, C, weights)    │  Weighted Chan    │
└──────────────────────────────────────────────────────────────────┘
```

### NOT Accumulate-Native (require sequential state)

```
┌──────────────────────────────────────────────────────────────────┐
│  Algorithm            │  Why not                                  │
├───────────────────────┼───────────────────────────────────────────┤
│  Oja's rule           │  SGD: w_t depends on w_{t-1}             │
│  Sanger/GHA           │  SGD: deflation uses current components  │
│  Brand incremental    │  SVD state (U,S,V) propagates each step  │
│  sklearn IncrementalPCA│  Batch fold: SVD augment per batch      │
└──────────────────────────────────────────────────────────────────┘
```

Brand and IncrementalPCA CAN be expressed as parallel merge-reduces (Brand shows this),
but the merge cost scales with rank² — they're not clean monoids like Chan covariance.

---

## Part 6: Exact Formulas Reference

### Chan's Covariance Merge (the core combine operator)

```
Inputs:  (n_A, μ_A, C_A) and (n_B, μ_B, C_B)
         where C = Σ_i (x_i - μ)(x_i - μ)'  [cross-product accumulator]

δ    = μ_B - μ_A                              (p-vector)
n_AB = n_A + n_B                              (scalar)
μ_AB = μ_A + (n_B / n_AB) · δ               (p-vector)
C_AB = C_A + C_B + (n_A · n_B / n_AB) · δ·δ'  (p×p matrix, outer product of δ)
```

Sample covariance = C_AB / (n_AB - 1).
Population covariance = C_AB / n_AB.

### Welford Per-Element Update

```
Given current (n, μ, C) and new observation x:
  δ₁ = x - μ
  n'  = n + 1
  μ'  = μ + δ₁/n'
  δ₂  = x - μ'
  C'  = C + δ₁·δ₂'   (outer product — p×p rank-1 update)
```

Efficient implementation: δ₁·δ₂' = (x-μ)(x-μ')' — only two vector differences.

### Oja's Rule (for reference, not accumulate-native)

```
Given current w_t ∈ ℝᵖ (unit-norm approximate top eigenvector) and x_t:
  y_t    = w_t' · x_t                  (scalar projection)
  w_{t+1} = w_t + η_t · y_t · (x_t - y_t · w_t)

Learning rate: η_t ~ 1/t (decreasing, Robbins-Monro)
Convergence: w_t → q₁ (first principal component, ‖q₁‖=1)
```

For k components: use Sanger's GHA rule (deflation) or power method in batches.

### Randomized SVD — Pass 1 Accumulation

```
Ω ∈ ℝ^{p×ℓ}   (Gaussian random matrix, ℓ = k + oversampling)
Y = A·Ω        (n×ℓ, range sketch)

Y[i,:] = Σ_j A[i,j]·Ω[j,:]   (each row of Y = x_i' · Ω)

Accumulate formulation:
  Y = accumulate(All, phi = x_i ⊗ (Ω'·x_i), Add)   ← row x_i, column index j

Or per-column: Y[:,j] = Σ_i x_i · (Ω[i,j] inner products)  ← GramMatrix with Ω columns
```

Post-step: (Q, R) = QR(Y)  →  Q ∈ ℝ^{n×ℓ} (but Q is n×ℓ — don't store explicitly)

### Randomized SVD — Pass 2 Accumulation

```
B = Q'·A   (ℓ×p matrix)

B[l,j] = Σ_i Q[i,l] · A[i,j]

Accumulate formulation:
  B = accumulate(All, phi = Q_row_i ⊗ x_i, Add)
  where Q_row_i = Q[i,:] = the i-th row of Q (known from Pass 1 post-step)
```

Post-step: SVD(B) → Ũ, Σ, V' — tiny (ℓ×p) SVD
Final: U = Q·Ũ (reconstruct left singular vectors from Q and Ũ)

Note: Q is n×ℓ, which is too large to store if n is huge. In the streaming case, Q is
NOT materialized. Instead, we use a different approach: store only the QR factor R (ℓ×ℓ),
then in Pass 2 compute B directly without materializing Q. See Halko et al. Section 5.5
for the full streaming implementation.

---

## Part 7: Implementation Recommendations for Tambear

### For F22 (Dimensionality Reduction)

**Recommended PCA implementation path:**

```
Step 1: Add WelfordCovOp to the accumulate primitive registry
  State: (n: u64, mean: Vec<f32>, C: SymmetricMatrix<f32>)
  Identity: (0, zeros, zeros)
  Combine: Chan's formula above
  Update: Welford per-element formula above

Step 2: Define PCA as:
  (_, μ, C) = accumulate(All, phi=row_identity, op=WelfordCovOp)
  Σ = C / (n-1)
  (λ, Q) = eigh(Σ)   // symmetric eigendecomposition, O(p³)
  PC = Q[:, -k:]     // top-k eigenvectors (largest eigenvalues last in eigh)
  Z  = (X - μ) · PC // projection — another GramMatrix operation

Step 3: Explained variance and loadings:
  explained_var_ratio = λ[-k:] / sum(λ)   // simple division
  loadings = PC * sqrt(λ[-k:])            // scaled eigenvectors
```

**For randomized SVD** (when p is very large):

```
Step 1: Generate Ω ~ N(0, 1/p), shape (p, ℓ)
Step 2: Y = accumulate(All, phi=x·Ω, op=VectorAdd)  // Pass 1: matrix multiply
Step 3: (Q, R) = QR(Y)
Step 4: B = accumulate(All, phi=outer(Qtranspose·x, x), op=MatrixAdd)  // Pass 2
Step 5: (Ũ, Σ, V) = SVD(B)
Step 6: U = Q·Ũ   // full left singular vectors
```

### Gold Standard Validation References

```python
# Exact PCA vs sklearn:
from sklearn.decomposition import PCA
pca = PCA(n_components=k)
pca.fit(X)
pca.components_    # k×p principal component directions
pca.explained_variance_ratio_   # variance explained per component
pca.mean_          # data mean

# Incremental PCA vs sklearn:
from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=k)
for batch in batches:
    ipca.partial_fit(batch)
ipca.components_              # should converge to PCA.components_

# Randomized SVD vs sklearn:
from sklearn.utils.extmath import randomized_svd
U, s, Vt = randomized_svd(X, n_components=k, n_iter=4, random_state=42)
# n_iter=4 applies 4 power iterations for better accuracy
```

### Numerical Traps

1. **Sign ambiguity**: PCA eigenvectors are defined up to sign. Canonical form: flip so
   the component with largest absolute value is positive. sklearn does this.

2. **Covariance vs correlation matrix**: PCA on raw covariance is scale-sensitive. If
   features have very different variances, standardize first (divide by std dev). This
   is PCA on the correlation matrix.

3. **Centering for SVD vs covariance PCA**: SVD on centered X gives PCA. SVD on uncentered
   X gives a different decomposition (truncated SVD). Always center for PCA.

4. **Numerical stability of Chan's formula**: When n_A >> n_B or n_A << n_B, the
   weighting factor n_A·n_B/n_AB approaches zero correctly. No cancellation issues.

5. **Memory for C matrix**: p×p covariance accumulator. For p=10,000: 10⁸ f32 = 400 MB.
   For p=100,000: 40 GB — infeasible. Use randomized SVD beyond p≈5,000.

---

## Part 8: Citation Index

- Welford (1962) — online variance algorithm: "Note on a method for calculating corrected sums of squares and products." Technometrics 4(3):419-420.
- Chan, Golub, Leveque (1979) — parallel covariance merge: "Updating formulae and a pairwise algorithm for computing sample variances." Tech. Rep. STAN-CS-79-773.
- Brand (2002) — incremental SVD: "Incremental singular value decomposition of uncertain data with missing values." ECCV 2002, MERL TR-2002-24.
- Ross, Lim, Lin, Yang (2008) — IPCA: "Incremental learning for robust visual tracking." IJCV 77(1-3):125-141.
- Halko, Martinsson, Tropp (2011) — randomized SVD: "Finding structure with randomness." SIAM Review 53(2):217-288.
- Oja (1982) — online PCA: "Simplified neuron model as a principal component analyzer." J. Math. Biology 15:267-273.
- Sanger (1989) — generalized Hebbian algorithm: "Optimal unsupervised learning in a single-layer linear feedforward neural network." Neural Networks 2:459-473.
