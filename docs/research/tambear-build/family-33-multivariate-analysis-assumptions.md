# Family 33: Multivariate Analysis — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (Commutative) — ALL methods are eigendecompositions of degree-2 GramMatrices

---

## Core Insight: Everything is GramMatrix → Eigendecomposition

Every method in this family follows one pattern:
1. Compute cross-product matrices from data (GramMatrix subblocks)
2. Form a ratio or product of these matrices
3. Eigendecompose the result
4. Extract test statistics / canonical variates from eigenvalues/eigenvectors

**GramMatrix once → feed CCA, MANOVA, discriminant analysis, Hotelling's T², all multivariate tests.**

---

## 1. Canonical Correlation Analysis (CCA)

### Setup
Two sets of variables: X (n × p) and Y (n × q). Assume p ≤ q WLOG.

### Goal
Find linear combinations a'X and b'Y that maximize correlation:
```
ρ = cor(a'X, b'Y) = a'Σ_XY·b / √(a'Σ_XX·a · b'Σ_YY·b)
```

### Solution
The canonical correlations ρ₁ ≥ ρ₂ ≥ ... ≥ ρ_min(p,q) are the square roots of the eigenvalues of:
```
Σ_XX⁻¹ · Σ_XY · Σ_YY⁻¹ · Σ_YX
```

Equivalently (numerically better): SVD of the **whitened cross-covariance**:
```
Σ_XX^(-1/2) · Σ_XY · Σ_YY^(-1/2) = U · diag(ρ₁,...,ρₛ) · V'
```

Canonical variates: aₖ = Σ_XX^(-1/2) · uₖ, bₖ = Σ_YY^(-1/2) · vₖ.

### From GramMatrix
The full (p+q) × (p+q) GramMatrix of [X, Y]:
```
G = [Σ_XX  Σ_XY]
    [Σ_YX  Σ_YY]
```
All four blocks come from ONE tiled accumulate of the concatenated data. CCA extracts from subblocks.

### Significance Tests
**Wilks' Lambda** for testing ρₖ₊₁ = ... = ρₛ = 0:
```
Λₖ = Π_{i=k+1}^{s} (1 - ρᵢ²)
```

**Bartlett's approximation**: -((n-1) - (p+q+1)/2) · ln(Λₖ) ~ χ²((p-k)(q-k))

### Edge Cases
- p = 0 or q = 0: no canonical correlations
- Σ_XX or Σ_YY singular: regularize with ridge (Σ + λI)
- n < p + q: GramMatrix rank-deficient → use truncated SVD
- ρ₁ = 1 exactly: perfect linear dependency between X and Y sets

### Implementation: ~50 lines after F10 GramMatrix + F22 SVD exist
1. Compute GramMatrix of [X, Y] → subblocks Σ_XX, Σ_XY, Σ_YY
2. Cholesky of Σ_XX, Σ_YY (from F10)
3. Whitened cross-covariance via triangular solves
4. SVD of whitened matrix (from F22)
5. Extract canonical correlations and variates

---

## 2. MANOVA (Multivariate Analysis of Variance)

### Setup
k groups, p response variables. Group g has nₘ observations.

### Matrices
**Total SSCP** (Sum of Squares and Cross Products):
```
T = Σᵢ (xᵢ - x̄)(xᵢ - x̄)'    (p × p)
```

**Within-group SSCP**:
```
W = Σ_g Σ_{i∈g} (xᵢ - x̄_g)(xᵢ - x̄_g)'    (p × p)
```

**Between-group SSCP** (Hypothesis matrix):
```
H = B = Σ_g nₘ(x̄_g - x̄)(x̄_g - x̄)' = T - W    (p × p)
```

### From GramMatrix
Per-group GramMatrix gives Σ_XX per group → W = Σ_g(Σ_XX_g - nₘ·x̄_g·x̄_g').
Between: H = T - W. Same sharing as F07 ANOVA (which uses scalar versions of these).

### Four Test Statistics

All based on eigenvalues θ₁ ≥ ... ≥ θₛ of **H·W⁻¹** (or equivalently, eigenvalues λᵢ of H·E⁻¹ where E = W):

| Test | Formula | Distribution (approx) |
|------|---------|----------------------|
| **Wilks' Λ** | Λ = Π(1/(1+θᵢ)) = \|W\|/\|T\| | F via Rao's exact transform |
| **Pillai's trace** | V = Σ θᵢ/(1+θᵢ) | F(s·p_h, s·p_e) approx |
| **Hotelling-Lawley trace** | U = Σ θᵢ | F approx |
| **Roy's largest root** | θ_max = θ₁ | Special tables (Roy distribution) |

where s = min(p, k-1), p_h = k-1, p_e = N-k.

### Rao's F-Transform for Wilks' Λ
```
df₁ = p·(k-1)
df₂ = m·s - p·(k-1)/2 + 1
F = ((1-Λ^(1/s))/Λ^(1/s)) · (df₂/df₁)
```
where s = √((p²(k-1)² - 4)/(p² + (k-1)² - 5)) and m = N - 1 - (p + k)/2.

This is EXACT (not an approximation) for certain combinations of p and k-1:
- p=1 or k-1=1: exact with s=1 (reduces to univariate F)
- p=2 or k-1=2: exact with s=√((p²·(k-1)²-4)/(p²+(k-1)²-5))

### Recommendations
- **Pillai's trace**: most robust to violations of assumptions, recommended default
- **Wilks' Λ**: most commonly reported (SPSS default), good for balanced designs
- **Roy's largest root**: most powerful for single dominant effect, least robust
- **Hotelling-Lawley**: good power, assumes equal covariances

### Edge Cases
- W singular (N-k < p): eigenvalues of HW⁻¹ undefined. Regularize or reduce dimensionality.
- k = 2: all four tests give the same F-value (reduces to Hotelling's T²)
- p = 1: MANOVA reduces to ANOVA (scalar case)

---

## 3. Hotelling's T² Test

### One-Sample
Test H₀: μ = μ₀ (p-dimensional mean vector)
```
T² = n(x̄ - μ₀)'S⁻¹(x̄ - μ₀)
```
where S = sample covariance matrix.

**Distribution**: (n-1)p/(n-p) · F(p, n-p). Exact under normality.

### Two-Sample
```
T² = (n₁n₂/(n₁+n₂)) · (x̄₁ - x̄₂)'S_p⁻¹(x̄₁ - x̄₂)
```
where S_p = pooled covariance = ((n₁-1)S₁ + (n₂-1)S₂)/(n₁+n₂-2).

**Distribution**: ((n₁+n₂-2)p/(n₁+n₂-p-1)) · F(p, n₁+n₂-p-1).

### From Sufficient Statistics
Needs: per-group {nₘ, x̄_g (p-vector), S_g (p×p covariance)}. These ARE the GramMatrix subblocks.

### Edge Cases
- n < p: S is singular. Use regularized inverse or Moore-Penrose pseudoinverse.
- p = 1: T² = t² (reduces to square of Student's t-test). Same structural rhyme as ANOVA/F.

---

## 4. Linear Discriminant Analysis (LDA)

### Fisher's Criterion
Find projection w that maximizes:
```
J(w) = w'Hw / w'Ww = w'S_Bw / w'S_Ww
```
(between-class / within-class scatter ratio)

Solution: generalized eigenvalue problem S_B·w = λ·S_W·w.

### At most min(p, k-1) discriminant dimensions
S_B has rank ≤ k-1 (k centroids span at most k-1 dimensional subspace).

### From GramMatrix
S_W = W/(N-k), S_B = H/(k-1) — same matrices as MANOVA. LDA and MANOVA are the SAME decomposition, different extraction.

### Classification Rule (Bayes with equal priors and common covariance)
Assign x to group g that minimizes Mahalanobis distance:
```
g* = argmin_g (x - x̄_g)'S_p⁻¹(x - x̄_g)
```
Equivalently: linear discriminant function δ_g(x) = x'S_p⁻¹x̄_g - ½x̄_g'S_p⁻¹x̄_g + log πₘ.

### Quadratic Discriminant Analysis (QDA)
When covariances differ per group (Σ_g ≠ Σ_h):
```
δ_g(x) = -½ log|Σ_g| - ½(x-μ_g)'Σ_g⁻¹(x-μ_g) + log πₘ
```
Decision boundary is quadratic (not linear).

### Edge Cases
- Singular S_W: common in high-dimensional settings (p > n). Regularize: S_W + λI.
- Equal group means: J(w) = 0, no discrimination possible.
- Two groups (k=2): only one discriminant dimension. LDA = logistic regression boundary.

---

## 5. Multivariate Normality Tests

### 5a. Mardia's Tests

**Multivariate skewness**:
```
b₁,p = (1/n²) Σᵢ Σⱼ [(xᵢ-x̄)'S⁻¹(xⱼ-x̄)]³
```
Under H₀: n·b₁,p/6 ~ χ²(p(p+1)(p+2)/6)

**Multivariate kurtosis**:
```
b₂,p = (1/n) Σᵢ [(xᵢ-x̄)'S⁻¹(xᵢ-x̄)]²
```
Under H₀: (b₂,p - p(p+2)) / √(8p(p+2)/n) ~ N(0,1)

### 5b. Henze-Zirkler Test
Based on the empirical characteristic function. More powerful than Mardia for many alternatives.

### From Sufficient Statistics
Needs: Mahalanobis distances (xᵢ-x̄)'S⁻¹(xᵢ-x̄) for all points. This IS the squared Mahalanobis distance matrix — computed via TiledEngine after whitening (same as F01 Mahalanobis distance).

---

## 6. Multivariate Regression (Multiple Response)

### Model
Y (n × q) = X (n × p) · B (p × q) + E (n × q)

### Solution
```
B̂ = (X'X)⁻¹X'Y
```
This is p separate regressions sharing the same design matrix. X'X is the same GramMatrix as univariate regression. X'Y has q columns instead of 1.

### Multivariate Tests on Coefficients
Test H₀: C·B·M = 0 (general linear hypothesis).

Hypothesis matrix: H = (CBM)'[C(X'X)⁻¹C']⁻¹(CBM)
Error matrix: E = M'Y'(I - X(X'X)⁻¹X')Y·M

Then apply MANOVA test statistics (Wilks, Pillai, etc.) to H and E.

### Implementation: Trivial extension of F10
F10 already has (X'X)⁻¹X'y for one response. For q responses, replace y with Y (matrix). Same Cholesky, q forward/back substitutions instead of 1.

---

## Sharing Surface

### The GramMatrix IS the Family
Every algorithm in F33 decomposes as:
```
GramMatrix(data) → subblock extraction → eigendecomposition → test statistic / variates
```

| Algorithm | GramMatrix Used | Extraction |
|-----------|----------------|------------|
| CCA | [Σ_XX, Σ_XY; Σ_YX, Σ_YY] | SVD of whitened cross-covariance |
| MANOVA | H = T - W (between, within SSCP) | Eigenvalues of HW⁻¹ |
| Hotelling's T² | S⁻¹ (inverse covariance) | Quadratic form |
| LDA | S_B, S_W | Generalized eigenvalues |
| QDA | Per-group Σ_g | Log-determinant + quadratic form |
| Multivariate regression | X'X, X'Y | Cholesky solve (same as F10) |
| Mardia's tests | Mahalanobis distances | Scatter of cubed/squared distances |

### Reuse from Other Families
- F10 (Regression): GramMatrix (X'X) + Cholesky → shared
- F22 (Dimensionality Reduction): Eigendecomposition → shared (PCA on covariance = same as LDA without grouping)
- F07 (Hypothesis Testing): MANOVA Wilks' Λ → multivariate generalization of ANOVA F-test
- F06 (Descriptive Statistics): Per-group moments → MomentStats per group

### Structural Rhymes Confirmed
- **MANOVA = multivariate ANOVA**: scalar SS_B/SS_W → matrix H/W, F-test → eigenvalue tests
- **LDA = Fisher's criterion = MANOVA extraction**: same H, W matrices, different consumer
- **CCA = SVD of whitened cross-covariance = regression of whitened X on whitened Y**
- **Hotelling's T² = MANOVA with k=2**: all four test statistics give the same answer

---

## Kingdom Classification

**Everything is Kingdom A.** This is the purest Kingdom A family:
- GramMatrix: tiled_accumulate(ByKey(group), dot_product) — degree 2 polynomial
- Eigendecomposition: CPU (Lanczos or full LAPACK dsyev)
- Test statistics: scalar functions of eigenvalues

No iteration (not Kingdom C). No sequential scan (not Kingdom B). Pure commutative accumulation → linear algebra → extraction.

---

## Implementation Priority

**Phase 1** — Core tests (~250 lines after F10+F22):
1. Hotelling's T² (one-sample + two-sample)
2. MANOVA (all four test statistics: Wilks, Pillai, Hotelling-Lawley, Roy)
3. LDA (Fisher's discriminant)

**Phase 2** — CCA + extensions:
4. CCA (SVD of whitened cross-covariance)
5. Multivariate regression (trivial extension of F10)
6. QDA (per-group covariance)

**Phase 3** — Normality tests:
7. Mardia's skewness + kurtosis
8. Henze-Zirkler

---

## Composability Contract

```toml
[family_33]
name = "Multivariate Analysis"
kingdom = "A (Commutative)"

[family_33.shared_primitives]
gram_matrix = "tiled_accumulate(ByKey(group) or All, dot_product)"
eigendecomposition = "eigenvalues of HW⁻¹ or SVD of whitened matrix"
cholesky = "for S⁻¹ and whitening"

[family_33.reuses]
f10_gram_matrix = "X'X, X'Y — same tiled accumulate"
f10_cholesky = "Σ factorization, triangular solves"
f22_svd_eigendecomp = "eigenvalues/vectors for CCA, LDA, MANOVA"
f06_moment_stats = "per-group means and covariances"
f07_anova = "MANOVA generalizes ANOVA to p dimensions"

[family_33.session_intermediates]
gram_matrix = "GramMatrix(data_id) — full (p+q)×(p+q) cross-product"
within_sscp = "WithinSSCP(data_id, grouping) — pooled within-group"
between_sscp = "BetweenSSCP(data_id, grouping) — between-group"
canonical_correlations = "CanonicalCorr(data_id_x, data_id_y)"
```
