# Mathematical Catalog — Pith Ecosystem

Complete inventory of all mathematics used across the Pith document intelligence layer, TAMBEAR statistical pipeline, and photon field rendering research. Organized by mathematical family with cross-references to implementation location and pipeline usage.

Generated: 2026-04-08

---

## 1. Descriptive Statistics

### 1.1 Central Tendency

#### 1.1.1 Arithmetic Mean
- **Equation**: $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$
- **Implementation**: `src/math/descriptive.ts:9-14`
- **Used in**: All statistical tests, factor analysis, PCA, regression, clustering
- **Pipeline use**: Universal preprocessing; input to variance, covariance, correlation
- **Effect size**: Cohen's d computation relies on mean difference
- **Alias**: μ (population), M (sample)

#### 1.1.2 Geometric Mean
- **Equation**: $\sqrt[n]{\prod_{i=1}^n x_i}$ or $\exp(\frac{1}{n}\sum \ln x_i)$
- **Implementation**: Not directly in codebase; can be built from mean of log-transformed data
- **Pipeline use**: Proportional growth rates, fold-change normalization
- **Note**: Not used in TAMBEAR v0.1; relevant for genomics/time-series work

#### 1.1.3 Harmonic Mean
- **Equation**: $\frac{n}{\sum_{i=1}^n 1/x_i}$
- **Implementation**: Not in codebase
- **Pipeline use**: F-scores, weighted average rates

### 1.2 Dispersion

#### 1.2.1 Variance (Population & Sample)
- **Equation (population)**: $\sigma^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2$
- **Equation (sample, unbiased)**: $s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$ (Bessel correction)
- **Implementation**: `src/math/descriptive.ts:17-27`
- **Parameters**: `bias=true` for population, `bias=false` (default) for sample
- **Used in**: Standard deviation, t-tests, ANOVA, regression, clustering quality metrics (silhouette)
- **Pipeline steps**: Input to power analysis, confidence intervals, effect sizes
- **Note**: The Bessel correction (divide by n-1 not n) is always applied for sample variance

#### 1.2.2 Standard Deviation
- **Equation**: $s = \sqrt{s^2} = \sqrt{\frac{1}{n-1} \sum (x_i - \bar{x})^2}$
- **Implementation**: `src/math/descriptive.ts:30-32`
- **Used in**: t-test denominator, z-score computation, effect size normalization
- **Pipeline use**: Standardization (z-transformation), confidence interval width
- **Related**: Square root of variance

#### 1.2.3 Coefficient of Variation
- **Equation**: $CV = \frac{s}{\bar{x}} \times 100\%$
- **Implementation**: Not directly exposed; computed as SD/mean in method-specific code
- **Pipeline use**: Comparing variability across variables with different scales
- **Note**: Scale-free dispersion measure

### 1.3 Distributional Shape

#### 1.3.1 Sample Skewness (Fisher's Moment Coefficient)
- **Equation**: $\gamma_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3$
- **Implementation**: `src/math/descriptive.ts:34-43`
- **Interpretation**: γ₁ > 0 = right-skewed; γ₁ < 0 = left-skewed; |γ₁| < 0.5 = symmetric
- **Used in**: Normality assessment, data characterization
- **Pipeline steps**: Diagnostic output in descriptive statistics, feeds into assumption checking
- **Note**: Bias-corrected version suitable for small samples

#### 1.3.2 Sample Excess Kurtosis (Kurtosis - 3)
- **Equation**: $\gamma_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum \left(\frac{x_i-\bar{x}}{s}\right)^4 - 3\frac{(n-1)^2}{(n-2)(n-3)}$
- **Implementation**: `src/math/descriptive.ts:45-56`
- **Interpretation**: γ₂ > 0 = heavy-tailed (leptokurtic); γ₂ < 0 = light-tailed (platykurtic)
- **Used in**: Outlier detection, heavy-tail assessment
- **Pipeline steps**: Descriptive output; informs variance-stabilizing transformations

### 1.4 Association

#### 1.4.1 Covariance (Sample)
- **Equation**: $\text{cov}(X,Y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$
- **Implementation**: `src/math/descriptive.ts:58-67`
- **Used in**: Correlation computation, covariance matrix, regression (XtX matrix)
- **Pipeline use**: Input to PCA eigendecomposition, factor analysis
- **Dimension**: Standardized by product of SDs to get correlation

#### 1.4.2 Pearson Correlation Coefficient
- **Equation**: $r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$
- **Implementation**: `src/math/descriptive.ts:69-87`
- **Properties**: -1 ≤ r ≤ 1; r=0 means no linear association; r=±1 means perfect linear
- **Used in**: Correlation test (method "correlation"), matrix construction for PCA
- **Pipeline steps**: Correlation method step, visualization (scatterplot), assumption checking
- **Confidence interval**: Fisher z-transform for CI construction
- **P-value**: Via t-distribution with df=n-2

#### 1.4.3 Spearman Rank Correlation
- **Equation**: $r_s = \text{Pearson}(\text{rank}(X), \text{rank}(Y))$
- **Implementation**: `src/math/descriptive.ts:89-92`
- **Used in**: Non-parametric alternative when data is non-normal or skewed
- **Pipeline use**: Correlation method with `method="spearman"`
- **Advantage**: Robust to monotone transformations and outliers; does not assume linearity

#### 1.4.4 Rank Function (for Ties)
- **Equation**: Assign 1, 2, ..., n; for ties, average the ranks they would occupy
- **Implementation**: `src/math/descriptive.ts:94-109`
- **Example**: [10, 20, 20, 30] → ranks [1, 2.5, 2.5, 4]
- **Used in**: Spearman correlation, Wilcoxon signed-rank test, Mann-Whitney U

#### 1.4.5 Kendall's Tau (τ)
- **Equation**: $\tau = \frac{\text{# concordant pairs} - \text{# discordant pairs}}{n(n-1)/2}$
- **Implementation**: Not yet in codebase; available as `method="kendall"` in correlation options but not executed
- **Used in**: Correlation method (interface prepared, implementation pending)
- **Advantage**: Easier to interpret for non-linear relationships; robust to outliers

### 1.5 Correlation & Covariance Matrices

#### 1.5.1 Correlation Matrix
- **Equation**: Symmetric p×p matrix where R[i,j] = r(x_i, x_j)
- **Implementation**: `src/math/descriptive.ts:111-124`
- **Used in**: PCA (eigendecomposition input), factor analysis, path analysis
- **Properties**: Diagonal = 1; off-diagonal ∈ [-1,1]; symmetric

#### 1.5.2 Covariance Matrix
- **Equation**: Symmetric p×p matrix where C[i,j] = cov(x_i, x_j)
- **Implementation**: `src/math/descriptive.ts:126-138`
- **Used in**: Regression (XtX matrix in normal equations), MANOVA
- **Relationship**: R = D^(-1) C D^(-1) where D = diag(s₁,...,s_p)

### 1.6 Hypothesis Testing Support

#### 1.6.1 Fisher Z-Transform
- **Equation**: $z = \frac{1}{2} \ln\left(\frac{1+r}{1-r}\right)$
- **Implementation**: `src/math/descriptive.ts:148-151`
- **Inverse**: $r = \frac{e^{2z}-1}{e^{2z}+1}$
- **Used in**: Confidence intervals for Pearson r, Fisher method for combining p-values
- **Motivation**: Makes correlation approximately normal for CI construction

#### 1.6.2 Pearson Correlation P-Value (via t-distribution)
- **Equation**: $t = r \sqrt{\frac{n-2}{1-r^2}}$, then P(T > |t|) where T ~ t(df=n-2)
- **Implementation**: `src/math/descriptive.ts:140-146`
- **Used in**: Correlation test; tests H₀: ρ = 0
- **Note**: Uses `tDistPValue1Tail` from stat-functions.ts for p-value

#### 1.6.3 Pearson Correlation Confidence Interval
- **Equation**: Fisher z ± z_crit × SE, where SE = 1/√(n-3), then inverse-transform
- **Implementation**: `src/math/descriptive.ts:158-165`
- **Parameters**: `alpha=0.05` for 95% CI
- **Used in**: Correlation method output, determines CI bounds in HTML output

### 1.7 Power Analysis for Correlation

#### 1.7.1 Required N for Correlation
- **Equation**: $N = \left(\frac{z_\alpha + z_\beta}{z|H1|}\right)^2 + 3$ where z is Fisher Z-transform
- **Implementation**: `src/math/descriptive.ts:211-217`
- **Parameters**: `alpha=0.05`, `power=0.80` (default)
- **Used in**: Study planning, power analysis method
- **Output**: Minimum sample size to detect effect size r with specified power

#### 1.7.2 Observed Power for Correlation
- **Equation**: P(Z > z_α - |z| √(n-3)) where Z ~ Normal(0,1)
- **Implementation**: `src/math/descriptive.ts:222-230`
- **Used in**: Post-hoc power analysis, sensitivity of published findings

### 1.8 Power Analysis for t-Tests

#### 1.8.1 Required N for Two-Sample t-Test
- **Equation**: $N = 2 \left(\frac{z_\alpha + z_\beta}{|d|}\right)^2$ (per group)
- **Implementation**: `src/math/descriptive.ts:235-240`
- **Parameters**: `alpha=0.05`, `power=0.80`, `d=effect size`
- **Used in**: t-test planning step

#### 1.8.2 Observed Power for t-Test
- **Equation**: P(Z > z_α - |d|√(n/2)) where Z ~ Normal(0,1)
- **Implementation**: `src/math/descriptive.ts:245-251`
- **Used in**: Post-hoc power analysis after observing t-test result

#### 1.8.3 Error Function Approximation (for Normal CDF)
- **Equation**: Abramowitz-Stegun rational approximation to erf(x)
- **Implementation**: `src/math/descriptive.ts:254-258`
- **Used in**: Normal CDF for power calculation (P(Z > z) via 1 - erf)
- **Reference**: Abramowitz & Stegun (1964) formula 7.1.26

### 1.9 Reliability

#### 1.9.1 Cronbach's Alpha
- **Equation**: $\alpha = \frac{k}{k-1} \left(1 - \frac{\sum_{i=1}^k \text{var}(X_i)}{\text{var}(X_{total})}\right)$
- **Implementation**: `src/math/descriptive.ts:260-278`
- **Parameters**: k = number of items, var(X_i) = variance of item i
- **Interpretation**: α ≥ 0.70 = acceptable; α ≥ 0.80 = good; α ≥ 0.90 = excellent
- **Used in**: Cronbach method step, internal consistency assessment
- **Pipeline steps**: Scale validation before multi-item composite creation

---

## 2. Hypothesis Testing Distributions

### 2.1 Gamma Function & Log-Gamma

#### 2.1.1 Log-Gamma via Lanczos Approximation
- **Equation**: $\ln \Gamma(x)$ via Lanczos approximation with g=7
- **Implementation**: `src/math/stat-functions.ts:11-25`
- **Reference**: Numerical Recipes, Godfrey coefficients
- **Used in**: Beta function (for t, F, χ² p-values), regularized incomplete beta
- **Accuracy**: ~15 significant digits for x > 0.5

### 2.2 Regularized Incomplete Beta Function

#### 2.2.1 Regularized Incomplete Beta I_x(a,b)
- **Equation**: $I_x(a,b) = \frac{B_x(a,b)}{B(a,b)}$ where B_x is incomplete beta, B is complete beta
- **Implementation**: `src/math/stat-functions.ts:35-60`
- **Method**: Lentz's continued fraction with symmetry relation for numerical stability
- **Used in**: Core engine for t, F, χ² p-values (via incomplete beta)
- **Convergence**: Up to 200 iterations, tolerance 1e-10

### 2.3 t-Distribution

#### 2.3.1 One-Tailed p-Value: P(T > |t|)
- **Equation**: $P(T > |t|) = \frac{1}{2} I_{df/(df+t^2)}(df/2, 0.5)$
- **Implementation**: `src/math/stat-functions.ts:64-68`
- **Used in**: Correlation p-value, one-tailed t-test
- **Degrees of freedom**: n-2 for correlation, n-p-1 for regression

#### 2.3.2 Two-Tailed p-Value
- **Equation**: Two × P(T > |t|)
- **Implementation**: `src/math/stat-functions.ts:70-74`
- **Used in**: Pearson t-test, regression coefficient tests
- **Standard**: Two-tailed is default in TAMBEAR methods

### 2.4 F-Distribution

#### 2.4.1 Upper-Tail p-Value: P(F > f | df₁, df₂)
- **Equation**: $P(F > f) = I_{df_2/(df_2 + df_1 \cdot f)}(df_2/2, df_1/2)$
- **Implementation**: `src/math/stat-functions.ts:76-81`
- **Used in**: ANOVA (MS_between / MS_within), regression overall F-test
- **Degrees of freedom**: df₁ = number of groups - 1 (or predictors), df₂ = n - groups (or n - p - 1)

#### 2.4.2 F-Distribution CDF: P(F ≤ f)
- **Equation**: $1 - I_{df_2/(df_2 + df_1 f)}(df_2/2, df_1/2)$ (complement of p-value)
- **Implementation**: `src/math/stat-functions.ts:84-88`
- **Used in**: Power analysis for ANOVA, quantile functions

### 2.5 Normal Distribution

#### 2.5.1 Normal Quantile (Probit) - Beasley-Springer-Moro
- **Equation**: Inverse CDF Φ^(-1)(p)
- **Implementation**: `src/math/descriptive.ts:168-203`
- **Method**: Rational approximation, piecewise with 3 regions
- **Used in**: Power analysis (z_α, z_β), confidence interval critical values
- **Accuracy**: ~4.5 × 10^(-4) maximum relative error

#### 2.5.2 Error Function (erf) via Abramowitz-Stegun
- **Equation**: Approximation to 1/√π ∫₀^x e^(-t²) dt
- **Implementation**: `src/math/descriptive.ts:254-258`
- **Used in**: Normal CDF via 1 - erf(x/√2)
- **Reference**: Abramowitz & Stegun formula 7.1.26

---

## 3. Linear Algebra

### 3.1 Basic Matrix Operations

#### 3.1.1 Matrix Multiply
- **Equation**: C = A × B where C[i,j] = Σ_k A[i,k] × B[k,j]
- **Implementation**: `src/math/matrix.ts:30-41`
- **Complexity**: O(n³) for n×n matrices (naive); no fast algorithms used
- **Used in**: Regression (XtX, XtX⁻¹ Xt y), factor analysis (V * sqrt(D))
- **Note**: Assumes matrices are compatible dimensions

#### 3.1.2 Transpose
- **Equation**: T[j,i] = A[i,j]
- **Implementation**: `src/math/matrix.ts:20-28`
- **Complexity**: O(n²)
- **Used in**: Regression (Xt = transpose(X)), correlation matrix (symmetric structure)

#### 3.1.3 Identity Matrix
- **Equation**: I[i,j] = 1 if i=j, else 0
- **Implementation**: `src/math/matrix.ts:8-13`
- **Used in**: Initialization for eigen algorithms, matrix inversion

#### 3.1.4 Zero Matrix
- **Equation**: All elements = 0
- **Implementation**: `src/math/matrix.ts:15-18`
- **Used in**: Initialization in iterative algorithms

#### 3.1.5 Frobenius Norm
- **Equation**: $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$
- **Implementation**: `src/math/matrix.ts:48-53`
- **Used in**: Convergence checking in iterative algorithms (e.g., Jacobi rotation)
- **Property**: Equals √(trace(A^T A))

#### 3.1.6 Diagonal Extraction & Construction
- **Equation**: diag(A) = [A[0,0], A[1,1], ..., A[n,n]], diagMat(d) = diag matrix with d on diagonal
- **Implementation**: `src/math/matrix.ts:55-69`
- **Used in**: Eigenvalue extraction, variance-stabilizing transformations

### 3.2 Eigendecomposition

#### 3.2.1 Jacobi Eigendecomposition (Symmetric)
- **Equation**: A = V Λ V^T where Λ = diag(λ), V = [v₁ ... v_p]
- **Implementation**: `src/math/matrix.ts:78-149`
- **Algorithm**: Iterative Givens rotations targeting largest off-diagonal element
- **Convergence**: Stops when max |A[i,j]| < 1e-12 (for i ≠ j)
- **Max iterations**: 100 × n²
- **Complexity**: O(n³) per iteration, typically < 100 iterations
- **Used in**: PCA (covariance/correlation eigen), factor analysis, parallel analysis
- **Output**: Eigenvalues sorted descending; eigenvectors as columns of matrix V
- **References**: Numerical Recipes §11.1, QR algorithm alternative (not implemented)

### 3.3 Linear System Solving

#### 3.3.1 LU Decomposition with Partial Pivoting
- **Equation**: PA = LU where P is permutation, L is unit lower triangular, U is upper triangular
- **Implementation**: `src/math/matrix.ts:155-194`
- **Stability**: Partial pivoting for numerical stability
- **Singularity test**: max |LU[k,k]| < 1e-14 triggers singular matrix error
- **Complexity**: O(n³)
- **Used in**: Regression (solve XtX β = Xt y for β), matrix inversion
- **Output**: Solves Ax = b by forward/back substitution on LU form

#### 3.3.2 Matrix Inversion via LU
- **Equation**: A^(-1) computed by solving A X = I column-by-column
- **Implementation**: `src/math/matrix.ts:196-208`
- **Complexity**: O(n³) (same as LU solve n times)
- **Used in**: Computing standard errors in regression (SE = √(MSE × diag(XtX)^(-1)))
- **Note**: Avoid in production; use solve(A, I) instead for numerical stability

---

## 4. Factor Analysis & PCA

### 4.1 Principal Components Analysis

#### 4.1.1 PCA with Kaiser Normalization
- **Algorithm**: 
  1. Standardize data Z = (X - mean) / SD
  2. Compute correlation matrix R = Z^T Z / (n-1)
  3. Eigendecompose R → λ, V
  4. Loadings L = V sqrt(Λ) (first k columns)
  5. Optionally rotate L via varimax
- **Implementation**: `src/math/factor-analysis.ts:37-119`
- **Kaiser criterion**: nFactors = # eigenvalues > 1 (default)
- **User control**: Can specify nFactors explicitly
- **Output structure**:
  - loadings: p × k matrix
  - eigenvalues: p-vector (all eigenvalues)
  - communalities: p-vector (sum of squared loadings per variable)
  - varianceExplained: k-vector (% variance per component)
  - cumulativeVariance: cumulative sum
  - scores: n × k (component scores for observations)
  - rotationMatrix: k × k (if rotation applied)
- **Used in**: Factor analysis method, dimensionality reduction, multivariate assumption checking

#### 4.1.2 Factor Scores (Regression Method)
- **Equation**: Unrotated PCA: S = Z L (exact for PCA); Rotated: S ≈ Z L (approximation)
- **Note**: Full regression method S = Z R^(-1) L avoided for rotated PCA to skip matrix inversion
- **Implementation**: `src/math/factor-analysis.ts:100-106`
- **Bias**: Small negative bias after varimax rotation, negligible for high communalities

### 4.2 Varimax Rotation

#### 4.2.1 Varimax with Kaiser Normalization
- **Goal**: Maximize sum of variances of squared loadings (Σ_j Σ_i L[i,j]^4)
- **Kaiser normalization**: Rows normalized by communality h_i = √(Σ_j L[i,j]²) before rotation
- **Implementation**: `src/math/factor-analysis.ts:129-206`
- **Algorithm**: Pairwise Givens rotations until convergence
- **Convergence criterion**: maxRotation < tolerance (default 1e-6)
- **Max iterations**: 100
- **Output**: Rotated loadings A, rotation matrix T such that A = L_original T
- **Interpretation**: Simplifies factor structure; makes loadings closer to 0 or 1
- **Used in**: EFA (exploratory factor analysis) method

#### 4.2.2 Rotation Angle Calculation (Varimax)
- **Equation**: angle = atan2(num, den) / 4 where num, den involve sums of powers of loadings
- **Implementation**: `src/math/factor-analysis.ts:162-176`
- **Givens rotation**: Rotate columns i and j by angle in the loading matrix and accumulate in T

### 4.3 Parallel Analysis

#### 4.3.1 Parallel Analysis (Horn, 1965)
- **Algorithm**:
  1. Compute observed eigenvalues from real data
  2. For each of `nRep` iterations:
     a. Generate n × p random normal matrix
     b. Compute correlation matrix eigenvalues
     c. Store percentile (default 95th)
  3. Count how many observed λ exceed random 95th percentile
- **Implementation**: `src/math/factor-analysis.ts:212-248`
- **Random number generator**: Box-Muller (two Uniform → two Normal)
- **Parameters**: `nRep=100` (number of random permutations), uses 95th percentile
- **Output**: `nFactors` (number to retain), `randomEigenvalues` (95th percentiles for reference)
- **Used in**: Determining number of factors to extract (more conservative than Kaiser)
- **Reference**: Horn (1965), Hayton et al. (2004)

---

## 5. Clustering

### 5.1 K-Means Clustering

#### 5.1.1 Lloyd's Algorithm with k-means++ Initialization
- **Algorithm**:
  1. Initialize centroids via k-means++ (probabilistic seeding)
  2. Repeat until convergence (max 100 iterations):
     a. Assign each point to nearest centroid (Euclidean distance)
     b. Update centroids as mean of assigned points
- **Implementation**: `src/math/clustering.ts:32-99`
- **Initialization**: k-means++ (Appendix in Arthur & Vassilvitskii 2007)
- **Distance metric**: Squared Euclidean: d²(x,c) = Σ(x_i - c_i)²
- **Convergence**: Stops when no point changes cluster assignment
- **Complexity**: O(nkd × iterations) where n=samples, k=clusters, d=dimensions
- **Output structure**:
  - assignments: n-vector (cluster labels 0 to k-1)
  - centroids: k × d matrix
  - iterations: number of iterations until convergence
  - wcss: k-vector (within-cluster sum of squares per cluster)
  - totalWCSS: sum of WCSS (goodness metric; lower is better)
  - bcss: between-cluster sum of squares
  - silhouettes: n-vector (silhouette coefficient per observation)
  - meanSilhouette: average silhouette (quality metric; -1 to 1)
- **Used in**: K-means method step

#### 5.1.2 k-means++ Initialization
- **Algorithm**:
  1. Pick first centroid uniformly at random from data
  2. For c = 2 to k:
     a. Compute d[i] = distance from x_i to nearest existing centroid
     b. Pick new centroid with probability ∝ d[i]²
- **Implementation**: `src/math/clustering.ts:101-135`
- **Motivation**: Spreads initial centroids far apart, reducing chance of poor local minimum
- **Reference**: Arthur & Vassilvitskii (2007)

#### 5.1.3 Silhouette Coefficient
- **Equation**: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
  - a(i) = mean distance from x_i to other points in its cluster
  - b(i) = min (over other clusters) of mean distance to that cluster
- **Interpretation**: s(i) ∈ [-1, 1]; closer to 1 means well-clustered
- **Implementation**: `src/math/clustering.ts:147-178`
- **Used in**: Cluster quality assessment, k selection

#### 5.1.4 Elbow Method
- **Algorithm**: Compute totalWCSS for k = 1, 2, ..., maxK; plot to find elbow
- **Implementation**: `src/math/clustering.ts:184-191`
- **Used in**: Determining optimal k (visual inspection of plot)
- **Limitation**: Elbow may be ambiguous; combine with silhouette or domain knowledge

---

## 6. Regression

### 6.1 Ordinary Least Squares (OLS)

#### 6.1.1 OLS Regression: Normal Equations
- **Model**: y = Xβ + ε where β = [β₀, β₁, ..., β_k]^T
- **Normal equations**: (X^T X)^(-1) X^T y
- **Implementation**: `src/math/regression.ts:51-132`
- **Steps**:
  1. Construct design matrix X (n × p) with intercept column [1, x₁, x₂, ...]
  2. Compute X^T X and X^T y
  3. Solve (X^T X)β = X^T y via LU decomposition
  4. Singularity check: if det(X^T X) ≈ 0, error on collinearity
- **Output structure**:
  - coefficients: p-vector (β₀, β₁, ..., β_k)
  - standardErrors: p-vector (SE for each β)
  - tValues: p-vector (t = β / SE)
  - pValues: p-vector (two-tailed p-values via t-dist)
  - rSquared: R² = 1 - SS_res / SS_tot
  - adjRSquared: adjusted R² penalizing for degrees of freedom
  - fStatistic: (SS_model / df_model) / (SS_res / df_res)
  - fPValue: upper-tail p-value for F
  - residualSE: √(MS_res) = √(SS_res / df_res)
  - dfResidual: n - p
  - residuals: y - ŷ
  - fitted: ŷ = Xβ
  - vif: k-vector (variance inflation factors, one per predictor)
  - durbinWatson: autocorrelation diagnostic
  - nObs: n
  - nPredictors: k
- **Used in**: Regression method, mediation analysis, moderation analysis, HLM

#### 6.1.2 Sum of Squares Decomposition
- **Total SS**: SS_tot = Σ(y_i - ȳ)²
- **Residual SS**: SS_res = Σ(y_i - ŷ_i)² = Σ ε_i²
- **Model SS**: SS_model = SS_tot - SS_res
- **R²**: Proportion of variance explained = SS_model / SS_tot
- **Adjusted R²**: Penalizes for extra predictors: 1 - (1-R²)(n-1)/(n-p)
- **Mean squares**: MS_model = SS_model / df_model, MS_res = SS_res / df_res
- **F-statistic**: MS_model / MS_res ~ F(df_model, df_res) under H₀

#### 6.1.3 Standard Errors & t-Tests
- **SE(β_j)**: √(MS_res × (X^T X)^(-1)[j,j])
- **t-statistic**: t_j = β_j / SE(β_j) ~ t(df = n - p) under H₀: β_j = 0
- **p-value**: Two-tailed P(T > |t_j|) where T ~ t(n-p)
- **Implementation**: `src/math/regression.ts:98-107`

#### 6.1.4 Durbin-Watson Statistic
- **Equation**: $DW = \frac{\sum_{i=2}^n (e_i - e_{i-1})^2}{\sum_i e_i^2}$
- **Interpretation**: DW ≈ 2 = no autocorrelation; DW ≈ 0 = positive autocorr; DW ≈ 4 = negative
- **Range**: 0 to 4
- **Used in**: Time-series assumption checking, autocorrelation detection
- **Implementation**: `src/math/regression.ts:119-124`

#### 6.1.5 Variance Inflation Factor (VIF)
- **Equation**: VIF_j = 1 / (1 - R_j²) where R_j² is R² from regressing x_j on all other x's
- **Algorithm**: For each predictor j, fit regression of x_j on other predictors, get R²
- **Implementation**: `src/math/regression.ts:135-167`
- **Interpretation**: VIF ≈ 1 = no collinearity; VIF > 5 or 10 = concerning
- **Used in**: Multicollinearity diagnosis

---

## 7. Hypothesis Tests — Comparative

### 7.1 t-Test (Comparison of Means)

#### 7.1.1 Independent-Samples t-Test
- **Test statistic**: $t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2(1/n_1 + 1/n_2)}}$
- **Pooled variance**: $s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}$
- **df**: n₁ + n₂ - 2 (equal variances)
- **Effect size**: Cohen's d = (x̄₁ - x̄₂) / s_p
- **Assumptions**: Normality (robust via CLT for n > 30), independence, equal variances (testable)
- **Used in**: T-test method (test_type="independent")
- **Pipeline steps**: Specify t-test → validate → run → report d and p-value

#### 7.1.2 Welch-Satterthwaite Correction (Unequal Variances)
- **Test statistic**: $t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$
- **Welch df**: $df = \frac{(s_1^2/n_1 + s_2^2/n_2)^2}{(s_1^2/n_1)^2/(n_1-1) + (s_2^2/n_2)^2/(n_2-1)}$
- **Default in TAMBEAR**: var_equal=false (default) applies Welch correction
- **Advantage**: Valid even if variances differ; still valid if variances are equal (slightly less power)
- **Used in**: T-test with `var_equal=false`

#### 7.1.3 Paired-Samples t-Test
- **Difference scores**: D_i = x_i - y_i
- **Test statistic**: $t = \frac{\bar{D}}{s_D / \sqrt{n}}$ where s_D = SD of differences
- **df**: n - 1
- **Effect size**: Cohen's d_z = D̄ / s_D
- **Assumptions**: Paired observations, normality of differences
- **Used in**: T-test method (test_type="paired")

#### 7.1.4 One-Sample t-Test
- **Test statistic**: $t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$
- **df**: n - 1
- **Effect size**: Cohen's d = (x̄ - μ₀) / s
- **Tests H₀**: μ = μ₀ (population mean equals hypothesized value μ₀)
- **Used in**: T-test method (test_type="one-sample")
- **Parameters**: mu = μ₀ (hypothesized value)

#### 7.1.5 Cohen's d (Effect Size for t-Tests)
- **Equation (independent)**: $d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$
- **Equation (paired)**: $d_z = \frac{\bar{D}}{s_D}$
- **Interpretation**: |d| < 0.2 = negligible; 0.2-0.5 = small; 0.5-0.8 = medium; > 0.8 = large
- **CI for d**: Computed via noncentral t-distribution
- **Used in**: All t-test output

### 7.2 ANOVA (Comparison of ≥3 Means)

#### 7.2.1 One-Way ANOVA
- **Model**: y_ij = μ + α_i + ε_ij where i = 1..k groups, j = 1..n_i observations
- **Null hypothesis**: H₀: α₁ = ... = α_k = 0 (all group means equal)
- **Test statistic**: $F = \frac{MS_{between}}{MS_{within}} = \frac{SS_B / (k-1)}{SS_W / (n-k)}$
- **p-value**: Upper-tail P(F > f_obs | df₁ = k-1, df₂ = n-k)
- **SS decomposition**: 
  - SS_B = Σ n_i (ȳ_i - ȳ..)² (between-group variance)
  - SS_W = Σ Σ (y_ij - ȳ_i)² (within-group variance)
  - SS_T = SS_B + SS_W
- **Effect size**: $\eta^2 = SS_B / SS_T$ (proportion of variance explained)
- **Interpretation**: η² < 0.01 = negligible; 0.01-0.06 = small; 0.06-0.14 = medium; > 0.14 = large
- **Assumptions**: Normality (robust for balanced designs, n_i > 5), independence, homogeneity of variance (testable via Levene)
- **Used in**: ANOVA method step

#### 7.2.2 Post-Hoc Comparisons: Tukey HSD
- **Method**: Honest Significant Difference test
- **Statistic**: $q = \frac{|\bar{y}_i - \bar{y}_j|}{SE} \sim \text{Studentized Range}(k, df_{error})$
- **SE**: √(MS_within × (1/n_i + 1/n_j) / 2)
- **Controls**: Familywise Type I error rate at α
- **Used in**: ANOVA with post_hoc="tukey"

#### 7.2.3 Post-Hoc Comparisons: Bonferroni
- **Method**: Simple Bonferroni multiple comparison correction
- **Adjusted α**: α / m where m = number of comparisons
- **Conservative**: More stringent than Tukey; controls Type I but may miss true effects
- **Used in**: ANOVA with post_hoc="bonferroni"

### 7.3 Chi-Square Test (Categorical Association)

#### 7.3.1 Pearson Chi-Square Test of Independence
- **Contingency table**: r rows × c columns of counts
- **Expected frequency**: E_ij = (row_i total) × (col_j total) / N
- **Test statistic**: $\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$
- **df**: (r - 1)(c - 1)
- **p-value**: Upper-tail P(χ² > χ²_obs | df)
- **Null hypothesis**: H₀: row and column variables are independent
- **Assumptions**: E_ij ≥ 5 for all cells (use Yates or Fisher for smaller tables)
- **Used in**: Chi-square method step

#### 7.3.2 Cramér's V (Effect Size for Chi-Square)
- **Equation**: $V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}}$
- **Range**: 0 to 1
- **Interpretation**: V < 0.1 = negligible; 0.1-0.3 = small; 0.3-0.5 = medium; > 0.5 = large
- **Used in**: Chi-square method output

#### 7.3.3 Yates' Continuity Correction (2×2 Tables)
- **Equation**: $\chi^2_{Yates} = \sum \frac{(|O_{ij} - E_{ij}| - 0.5)^2}{E_{ij}}$
- **When to use**: Any expected count < 10 (conservative)
- **Effect**: Reduces χ² slightly, raises p-value
- **Parameter**: yates_correction=true in chi-square method
- **Used in**: 2×2 contingency tables with small expected counts

### 7.4 Non-Parametric Tests

#### 7.4.1 Mann-Whitney U Test (Non-Parametric Alternative to Independent t)
- **Hypothesis**: Tests whether two groups come from distributions with different medians
- **Algorithm**: Rank all n₁ + n₂ observations; U₁ = sum of ranks in group 1 - n₁(n₁+1)/2
- **Test statistic**: U = min(U₁, U₂)
- **Null distribution**: Discrete U distribution (tabulated for small n; normal approximation for large n)
- **Advantage**: No normality assumption; robust to outliers
- **Used in**: Mann-Whitney method step (not yet fully implemented)
- **Reference**: Mann & Whitney (1947)

#### 7.4.2 Wilcoxon Signed-Rank Test (Non-Parametric Paired t)
- **Hypothesis**: Tests whether paired differences have median = 0
- **Algorithm**: 
  1. Compute D_i = x_i - y_i, discard zeros
  2. Rank |D_i|, keeping signs
  3. T+ = sum of positive ranks
  4. T = min(T+, T-)
- **Used in**: Wilcoxon method step (not yet fully implemented)
- **Advantage**: No normality assumption; robust to extreme outliers
- **Reference**: Wilcoxon (1945)

#### 7.4.3 Kruskal-Wallis Test (Non-Parametric ANOVA)
- **Hypothesis**: Tests whether k groups come from distributions with different locations
- **Algorithm**: Rank all observations across groups; H = 12/(n(n+1)) × Σ R_i² / n_i - 3(n+1)
- **Null distribution**: χ² with df = k - 1 (asymptotic)
- **Used in**: Kruskal-Wallis method step (not yet fully implemented)
- **Reference**: Kruskal & Wallis (1952)

---

## 8. Correlation & Association

### 8.1 Bivariate Correlation (Already in Section 1.4)

### 8.2 Partial Correlation
- **Definition**: Correlation between X and Y holding Z constant
- **Equation**: $r_{XY.Z} = \frac{r_{XY} - r_{XZ} r_{YZ}}{\sqrt{(1-r_{XZ}^2)(1-r_{YZ}^2)}}$
- **Used in**: Partial correlation method step
- **Interpretation**: Controls for confounding or mediating variable Z

### 8.3 Polychoric & Tetrachoric Correlations
- **Polychoric**: Correlation for two ordinal variables (assumes underlying bivariate normal)
- **Tetrachoric**: Special case for two dichotomous variables
- **Used in**: SEM with ordinal/categorical observed variables
- **Implementation**: Not yet in codebase; relevant for advanced SEM
- **Note**: More complex than Pearson; typically requires iterative fitting

---

## 9. Resampling & Permutation Methods

### 9.1 Bootstrap

#### 9.1.1 Bootstrap Confidence Intervals
- **Algorithm**:
  1. Resample data with replacement: draw n samples from original n observations
  2. Compute statistic (e.g., mean, correlation) on each resample
  3. Repeat step 1-2 for B iterations (typically 1000-10000)
  4. Sort bootstrap statistics; CI = [percentile_α/2, percentile_(1-α/2)]
- **Used in**: Correlation method with bootstrap_n > 0
- **Advantage**: Nonparametric; valid for any statistic; works when parametric formula unavailable
- **Bias correction**: Percentile method; other methods (BCa, studentized) more complex
- **Parameters**: bootstrap_n = number of resamples; default = 0 (analytical CI instead)

#### 9.1.2 Parametric Bootstrap
- **Algorithm**: Fit model to data → simulate new data from fitted distribution → refit
- **Used in**: Not yet in TAMBEAR; relevant for likelihood-based inference

### 9.2 Permutation Tests

#### 9.2.1 Permutation Test (Randomization Test)
- **Algorithm**:
  1. Compute test statistic T_obs on original data
  2. Permute labels/assignments randomly
  3. Recompute T on permuted data
  4. Repeat step 2-3 for M times (1000-10000)
  5. p-value = (# times T_perm > T_obs) / M
- **Null hypothesis**: H₀: labels/group assignments don't matter
- **Advantage**: No parametric assumptions; valid for any test statistic
- **Used in**: Permutation test method step
- **Limitation**: Exact p-values require exponentially many permutations for large n

---

## 10. Dimensionality Reduction & Projection

### 10.1 Principal Components Analysis (Already in Section 4.1)

### 10.2 Linear Discriminant Analysis (LDA)
- **Goal**: Find linear combinations of variables that best separate k known groups
- **Computation**: Eigendecomposition of S_W^(-1) S_B where S_B = between-group, S_W = within-group covariance
- **Output**: k-1 discriminant functions (maximum)
- **Used in**: LDA method step (not yet fully implemented)
- **Difference from PCA**: PCA ignores group labels; LDA maximizes class separation

### 10.3 Grassmannian Geometry for Photon Field Rendering
- **Core idea**: The photon field renderer computes pushforward measure over R^N projected to R²
- **Mathematical framework**: 
  - μ = probability measure over R^N (per-photon distribution in high-dimensional space)
  - π = projection map R^N → R² (to screen coordinates)
  - Output image I(u,v) = π_*(μ) = marginal density of μ projected to screen
- **Kernel density estimation**: Additive KDE with bandwidth h: Î(u,v) = (1/n) Σ K_h(π(p_i) - (u,v))
- **Grassmannian connection**: The space of d-dimensional subspaces of R^N forms Gr(d,N); rendering at different dimensions explores different Grassmannian sections
- **Used in**: Photon field research (dimensional lifting proofs), rendering optimization
- **Key results**:
  - Theorem 16.1 (Liftability): Framework computes exactly linear functionals of per-photon measure
  - Theorem 22.1 (Accumulation Boundary): Order-dependent effects require non-additive accumulation
  - Theorem 23.1 (λ Continuum): Parameter λ continuously interpolates rendering modes

---

## 11. Advanced Regression Methods

### 11.1 Logistic Regression (Binary Outcome)
- **Model**: log(p/(1-p)) = β₀ + β₁x₁ + ... + β_kx_k (logit linear model)
- **Estimation**: Maximum likelihood (iteratively reweighted least squares)
- **Used in**: Logistic regression method step
- **Output**: β coefficients, odds ratios (exp(β)), p-values via Wald test

### 11.2 Poisson Regression (Count Outcome)
- **Model**: log(μ) = β₀ + β₁x₁ + ... (log-linear model)
- **Assumption**: Variance = mean (equidispersion)
- **Overdispersion**: If variance > mean, use quasi-Poisson or negative binomial instead
- **Used in**: Poisson regression method step

### 11.3 Quantile Regression
- **Model**: Quantile_τ(y | x) = β₀(τ) + β₁(τ)x₁ + ... (different intercepts/slopes per quantile)
- **Advantage**: Robust to outliers; models full distribution, not just mean
- **Used in**: Quantile regression method step
- **Interpretation**: β_j(0.5) = median regression; β_j(0.1) = 10th percentile regression

### 11.4 Ridge & Lasso Regression
- **Ridge**: Minimize SSE + λ Σ β_j² (shrinks large coefficients toward 0)
- **Lasso**: Minimize SSE + λ Σ |β_j| (shrinks some coefficients to exactly 0, feature selection)
- **Parameter**: λ ≥ 0 controls amount of shrinkage; choose via cross-validation
- **Used in**: Ridge/Lasso regression method steps
- **Relationship**: Bayesian interpretation as priors on β

### 11.5 Robust Regression (Huber, MM)
- **Goal**: Downweight outliers instead of leveraging them
- **Huber loss**: Quadratic for |residual| < k, linear beyond (combines MSE + MAD)
- **Used in**: Robust regression method step
- **Advantage**: Less affected by extreme observations than OLS

---

## 12. Mixed Models & Hierarchical Methods

### 12.1 Linear Mixed Models (HLM, Multilevel)
- **Model**: y_ij = β₀ + β₁x_ij + u_j + ε_ij where u_j ~ N(0, σ_u²), ε_ij ~ N(0, σ²)
- **Random intercept**: u_j allows cluster-specific baseline
- **Random slope**: u₁_j allows cluster-specific treatment effect
- **Estimation**: Restricted maximum likelihood (REML) or maximum likelihood (ML)
- **Used in**: HLM (hierarchical linear model) method step
- **Advantages**: Handles nested data (students in schools), unbalanced designs, missing data
- **Challenges**: Computational complexity (iterative optimization); many degrees of freedom

### 12.2 Generalized Linear Mixed Models (GLMM)
- **Extension**: Mixed models + non-normal outcomes (logistic, Poisson, etc.)
- **Used in**: GLMM method step
- **Challenges**: No closed-form likelihood; requires Laplace or Gauss-Hermite quadrature

---

## 13. Structural Equation Modeling (SEM)

### 13.1 Confirmatory Factor Analysis (CFA)
- **Model**: Observed variables = common factors + unique factors
- **Path diagram**: Latent factors with paths to observed indicators
- **Estimation**: Maximum likelihood on covariance matrix
- **Goodness of fit**: χ², RMSEA, CFI, TLI, SRMR
- **Used in**: CFA method step

### 13.2 Path Analysis
- **Regression chains**: y = f(x), x = f(w) — analyze indirect effects
- **Direct effect**: x → y
- **Indirect effect**: x → m → y (product of path coefficients)
- **Total effect**: Direct + indirect
- **Used in**: Mediation analysis (via path analysis framework)

### 13.3 Mediation Analysis
- **Model**: c = c' + ab where a = effect of X on mediator M, b = effect of M on Y, c = total effect, c' = direct effect
- **Test methods**: Sobel test (asymptotic), bootstrap CI (nonparametric)
- **Used in**: Mediation method step
- **Interpretation**: If 95% CI for ab excludes 0, significant indirect effect

### 13.4 Moderation Analysis
- **Model**: y = β₀ + β₁x + β₂z + β₃(x×z) + ε where z is moderator
- **Interpretation**: β₃ = effect of X depends on Z; simple slopes at Z = low/mean/high
- **Used in**: Moderation method step

---

## 14. Time-Series Methods

### 14.1 ARIMA (AutoRegressive Integrated Moving Average)
- **Model**: ARIMA(p,d,q) where p = autoregressive, d = differencing, q = moving average
- **Stationarity**: Differencing d times to make series stationary
- **Estimation**: Maximum likelihood (via ARSS or conditional sum of squares)
- **Used in**: ARIMA method step
- **Reference**: Box & Jenkins (1976)

### 14.2 Vector Autoregression (VAR)
- **Model**: Multivariate extension of AR; y_t = A₀ + A₁y_{t-1} + ... + A_py_{t-p} + ε_t
- **Interpretation**: Each variable's current value depends on past values of all variables
- **Lag order**: Choose p via AIC, BIC, or sequential testing
- **Used in**: VAR regression method step
- **Assumptions**: Stationarity of all series

---

## 15. Survival Analysis

### 15.1 Kaplan-Meier Estimator
- **Non-parametric survival curve**: Ŝ(t) = ∏_{t_i ≤ t} (1 - d_i / n_i)
- **d_i**: Number of events at time t_i
- **n_i**: Number at risk (not yet censored or experienced event) just before t_i
- **CI**: Via Greenwood variance: Var(log Ŝ) = Σ d_i / (n_i(n_i - d_i))
- **Used in**: Kaplan-Meier method step
- **Log-rank test**: Compares two KM curves; H₀: curves are equal

### 15.2 Cox Proportional Hazards
- **Model**: h(t|x) = h₀(t) exp(β'x) where h₀(t) is baseline hazard
- **Proportional hazards assumption**: HR doesn't change over time
- **Partial likelihood**: Estimate β without modeling h₀(t)
- **Interpretation**: exp(β_j) = hazard ratio per unit increase in x_j
- **Used in**: Cox regression method step
- **Schoenfeld residuals**: Test proportional hazards assumption

---

## 16. Bayesian Methods

### 16.1 Bayes Factor (Alternative to p-values)
- **Definition**: BF₁₀ = P(data | H₁) / P(data | H₀)
- **Interpretation**: BF > 10 = strong evidence for H₁; 1-3 = weak evidence; BF < 1 = evidence for H₀
- **Advantage**: Allows evidence for null hypothesis (p-values do not)
- **Used in**: Bayes factor method step
- **Computation**: Requires Bayesian model specification; complex for non-standard models

### 16.2 Bayesian t-Test
- **Prior**: Specify prior on effect size d
- **Likelihood**: Student's t model
- **Posterior**: Updated belief about d given data
- **Bayes factor**: Ratio of marginal likelihoods under H₀ and H₁
- **Used in**: Bayesian t-test method step

### 16.3 Bayesian Regression
- **Prior on β**: Often normal, N(0, Σ) (ridge regression interpretation)
- **Posterior**: N(mean, covariance) after observing data
- **Credible interval**: Analogous to confidence interval; directly interpretable as probability
- **Used in**: Bayesian regression method step

---

## 17. Effect Size Calculations

### 17.1 Cohen's d (Already in Section 7.1.5)

### 17.2 Eta-Squared (η²) for ANOVA
- **Equation**: η² = SS_between / SS_total
- **Interpretation**: Proportion of variance explained by group membership
- **Used in**: ANOVA method output

### 17.3 Omega-Squared (ω²) for ANOVA
- **Equation**: ω² = (SS_B - df_B × MS_error) / (SS_T + MS_error)
- **Advantage**: Less biased than η² (doesn't overestimate small effects)
- **Used in**: Can be requested in ANOVA advanced options

### 17.4 Cramér's V for Chi-Square (Already in Section 7.3.2)

### 17.5 Partial Eta-Squared
- **For mixed designs**: Variance explained by effect controlling for other factors
- **Used in**: ANOVA/MANOVA with multiple factors

---

## 18. Multiple Comparisons & Correction

### 18.1 Bonferroni Correction
- **Method**: Divide α by number of tests: α_adjusted = α / m
- **P-value correction**: Multiply observed p by m (cap at 1.0)
- **Conservative**: Controls familywise error rate; may lack power with many tests
- **Used in**: Multiple comparison correction method step, ANOVA post-hoc

### 18.2 Benjamini-Hochberg False Discovery Rate (FDR)
- **More liberal than Bonferroni**: Controls proportion of false discoveries
- **Method**: Sort p-values; find largest i where p_{(i)} ≤ (i/m)α; reject first i hypotheses
- **Controls**: Expected fraction of false discoveries among rejected hypotheses
- **Used in**: Multiple comparison correction method step

### 18.3 Tukey HSD (Already in Section 7.2.2)

---

## 19. Normality & Assumption Tests

### 19.1 Shapiro-Wilk Test
- **Null hypothesis**: Data come from normal distribution
- **Test statistic**: W = (Σ a_i x_{(i)})² / Σ(x_i - x̄)² where x_{(i)} are order statistics
- **Coefficients**: a_i precomputed (Shapiro & Wilk 1965)
- **Used in**: Normality assessment in assumption checking
- **Limitations**: Sensitive to even slight deviations; may reject normal data with n > 5000

### 19.2 Levene's Test (Homogeneity of Variance)
- **Null hypothesis**: All groups have equal variance
- **Method**: One-way ANOVA on absolute deviations from group medians
- **Used in**: Pre-test before independent t-test or ANOVA
- **Robustness**: Uses median (not mean) for robustness; detects heterogeneity well

### 19.3 Q-Q Plot
- **Graphical method**: Plot sample quantiles vs theoretical quantiles
- **Interpretation**: Points close to diagonal suggest normality
- **Used in**: Visual assumption checking (not a formal test)

### 19.4 Mahalanobis Distance (Multivariate Outliers)
- **Equation**: D² = (x - μ)^T Σ^(-1) (x - μ)
- **Threshold**: χ² critical value with df = p (number of variables)
- **Used in**: Bivariate outlier detection in correlation/regression
- **Interpretation**: D² > critical value = potential outlier

---

## 20. Effect Size Conversions

### 20.1 t to d Conversion
- **From t-statistic**: $d = 2t / \sqrt{df}$
- **Used in**: Meta-analysis pooling estimates
- **Note**: Requires sample size n or degrees of freedom

### 20.2 r to d Conversion
- **Equation**: $d = 2r / \sqrt{1 - r^2}$
- **Reverse**: $r = d / \sqrt{d^2 + 4}$
- **Used in**: Converting between correlation and standardized mean difference

### 20.3 χ² to V Conversion
- **From Cramér's V**: $\chi^2 = V^2 \cdot n \cdot \min(r-1, c-1)$
- **Used in**: Effect size comparisons across test types

---

## 21. Photon Field Rendering Mathematics

### 21.1 Kernel Density Estimation (KDE)
- **Continuous estimator**: $\hat{f}(u,v) = \frac{1}{nh^2} \sum_{i=1}^n K\left(\frac{u - x_i}{h}, \frac{v - y_i}{h}\right)$
- **Discrete (photon field)**: Splatting each photon p_i at screen position (u_i, v_i) with kernel K_h
- **Bandwidth h**: Controls kernel width; h → 0 gives noisy estimate, h → ∞ gives over-smoothed
- **Optimal bandwidth**: h* ∝ n^(-1/4) in 2D (minimizes MISE)
- **Used in**: Core rendering algorithm (splatting)

### 21.2 Gaussian Kernel
- **Equation**: $K(x,y) = \frac{1}{2\pi} \exp(-(x^2 + y^2)/2)$
- **In rendering**: Splat size (standard deviation = h) controls blur width
- **Average-case**: Over-smooth sharp edges; good for volumetric effects
- **Used in**: Default photon field kernel for smooth shading

### 21.3 Uniform Disk Kernel
- **Equation**: $K(x,y) = 1/(πh²)$ if √(x² + y²) ≤ h, else 0
- **Hard boundary**: Creates visible splat rings; sharp edge definition
- **Used in**: Educational demos (easier to understand)

### 21.4 Photon Path Space
- **Dimension**: Each photon carries coordinates (x, y, z, ω_x, ω_y, λ, t, k, w, ...)
  - (x,y,z) = 3D position (path dimension)
  - (ω_x, ω_y) = 2D direction on sphere (path dimension)
  - λ = wavelength (state dimension; transformed by dispersion)
  - t = emission time (path dimension; for motion blur)
  - k = bounce count (path dimension; for GI depth decomposition)
  - w = photon weight (state dimension; for gain media)
  - θ = polarization angle (state dimension; for Malus's law)
- **Projection**: π: R^N → R² selects screen coordinates
- **Output**: I(u,v) = density of photons at (u,v) after projection

### 21.5 Path vs State Dimension Classification
- **Path dimensions**: Describe where photon came from; kernel is *averaging* (integrating over paths)
  - Examples: position, time, bounce count
  - Kernel operation: Spread contribution over path variations
  - Examples: motion blur kernel (uniform [0,T]), GI bounce kernel (Gaussian in k), caustic depth kernel (exponential -λz)
  
- **State dimensions**: Internal degrees of freedom; kernel is *rejection sampling* or *weighted transmission*
  - Examples: wavelength, polarization, phase
  - Kernel operation: Transmit with weight based on state (Malus's law, Fresnel, Beer-Lambert)
  - Examples: polarization cos²(Δθ), absorption e^(-σz), reflection Fresnel(θ_i)

### 21.6 Green's Function & PDE Connection
- **Theorem**: Every kernel K is the Green's function of some differential operator L
- **Rendering as PDE**: Rendering with K solves L^(-1) ρ where ρ = photon density, L = differential operator
- **Examples**:
  - Gaussian kernel K(x) ∝ e^(-|x|²/h²) → Diffusion equation ∂ρ/∂t = ∇²ρ
  - Uniform disk K → Laplace equation ∇²ρ = 0
  - Exponential in z (depth): Hard shadow blending
- **Implications**: Different kernels compute different boundary-value problems

### 21.7 Fresnel Number & Diffraction Regime
- **Fresnel number**: F = D² / (λL) where D = feature size, λ = wavelength, L = propagation distance
- **Regimes**:
  - F << 1: Fraunhofer (far-field); shadow sharp-edged
  - F ~ 1: Fresnel (near-field); diffraction visible; penumbra structure
  - F >> 1: Geometric optics; shadow perfectly sharp
- **Rendering**: Per-photon wavelength λ as coordinate; kernel bandwidth = coherence length relates to F
- **Verified**: Airy disk demo (F ≈ 4.3 at default params); Airy null position matched to 0.9%

### 21.8 Liftability Principle (Theorem 16.1)
- **Statement**: The photon field framework computes exactly the set of quantities expressible as pushforward measures of per-photon functionals over finite-dimensional coordinate spaces
- **Consequence**: Cannot compute order-dependent effects (z-buffer, self-occlusion) exactly via additive accumulation
- **Partial lift**: z-buffer is MIN blend; can approximate via z-weighted accumulation with λ → ∞, converging as photon count N → ∞
- **Operator boundary**: Beyond liftability; requires non-additive per-pixel operations

### 21.9 Mode Dissolution (Theorem 27.1)
- **Statement**: Every discrete rendering mode is a limiting case of a continuous per-photon parameter
- **Example**: z-buffer (λ→+∞) and atmospheric perspective (λ→−∞) are the SAME parameter, opposite signs
- **Parameter λ**: Controls visibility blending: λ=0 (pure average/volumetric), λ>0 (surface occlusion), λ<0 (far-field dominance)
- **Consequence**: One renderer, all modes, one knob (λ) — no explicit mode switching needed
- **Novel**: Unifies discrete rendering concepts under continuous parameter space

### 21.10 Stimulated Emission (Fock Process, Non-Liftable)
- **Physics**: Photon triggers cascade in gain medium; energy released as multiple correlated photons
- **Not liftable**: Requires photon branching (non-additive); each interaction creates new photons whose contribution depends on N (total photon count)
- **Framework limitation**: Fock processes are the only hard boundary; all other effects liftable or partially liftable
- **Workaround**: Threshold fission (photon weight w → fission if w > threshold)

### 21.11 Subsurface Scattering (SSS) — Dipole Approximation
- **Full space**: 10D (entry point 2D, direction 2D, interior scatter 5D, exit point 2D)
- **Dipole reduction**: Assumes diffusion regime (many scatters, isotropic phase function)
- **Result**: R_d(r) = diffusion profile (depends only on |entry - exit| distance on surface)
- **Integral becomes 2D convolution**: L_out(Q) = ∫ R_d(|P-Q|) L_in(P) dP (surface integral only)
- **Error regimes**: Fails near absorption, thin geometry, edge curvature, anisotropic phase functions
- **In photon field framework**: Full 10D trajectory sampling + projection to surface represents exact SSS (no dipole approximation)

### 21.12 Polarization State Space (Stokes Vector)
- **Full state**: 4-component Stokes vector (S₀, S₁, S₂, S₃) or 2×2 complex coherence matrix
- **Simplified**: For linear polarization, use scalar θ ∈ [0°, 180°] (orientation)
- **Malus's law**: cos²(θ_photon - θ_polarizer)
- **Three-polarizer paradox**: Middle polarizer rotates photon state, opening non-zero projection to final filter; geometry resolves "paradox"
- **Full Stokes encoding**: S₁ = linear h/v, S₂ = linear ±45°, S₃ = circular (RGB channel mapping)
- **Extensions**: Birefringence (full Jones calculus), optical activity (rotation matrices)

### 21.13 Phase & Coherence
- **De Broglie wavelength = kernel bandwidth**: λ_dB = h/p (momentum); in photon field, bandwidth h sets coherence length
- **Phase φ**: Accumulated optical path length; relevant for interference (requires tracking complex amplitudes, not just real intensities)
- **Interference**: Two photons with phase difference Δφ; intensity = |e^(iφ₁) + e^(iφ₂)|² = 2 + 2cos(Δφ)
- **Framework extension**: Photon carries (phase, amplitude) pair; splat complex numbers instead of scalars
- **Used in**: Photon-field interference.html demo (Newton rings, Fabry-Perot)

---

## 22. Cross-Cutting Concepts

### 22.1 Standardization (z-Score)
- **Equation**: $z_i = (x_i - \bar{x}) / s$
- **Properties**: Mean = 0, SD = 1
- **Used in**: PCA input (correlation form), effect size comparisons, outlier detection (|z| > 3 = outlier)

### 22.2 Centering & Scaling
- **Centering**: x_c = x - mean (zero-center data)
- **Scaling**: x_s = x / SD (standardize variance to 1)
- **Combined**: Standardization = center + scale
- **Used in**: PCA, regression (reduces multicollinearity), power analysis

### 22.3 Orthogonalization (Gram-Schmidt)
- **Algorithm**: Make vectors orthogonal via successive projections
- **QR decomposition**: Factor matrix A = QR where Q is orthogonal, R is upper triangular
- **Used in**: Numerically stable regression (not directly implemented; could improve QR over LU)
- **Alternative**: Singular Value Decomposition (SVD) — more expensive but most stable

### 22.4 Centering, Scaling, & Rank in Clustering
- **k-means sensitive to scale**: Large-variance features dominate; standardize before clustering
- **Missing data**: Listwise deletion (entire row) vs pairwise (use available pairs)
- **Rank correlation option**: Use rank(x) instead of x for robustness

### 22.5 Confidence Interval Construction
- **Parametric (analytical)**: Assume distribution (t, normal, F); compute critical value; CI = estimate ± crit × SE
- **Nonparametric (bootstrap)**: Resample; compute percentile; CI = [percentile_α/2, percentile_(1-α/2)]
- **Used in**: All hypothesis tests (t, correlation, regression, mediation)
- **Level α**: Default 0.05; produces (1-α)×100% = 95% CI

### 22.6 Degrees of Freedom
- **Sample mean**: df = n - 1 (lose one df estimating mean from data)
- **Regression**: df = n - p where p = # predictors + intercept
- **t-test**: df = n₁ + n₂ - 2 (equal var) or Welch df (unequal var)
- **ANOVA**: df_between = k - 1, df_within = n - k, df_total = n - 1
- **Chi-square**: df = (r-1)(c-1) for r×c contingency table
- **Affects**: Critical values, p-values, power calculations

### 22.7 Multiple Comparisons Problem
- **Issue**: Running many tests inflates false positive rate
- **Example**: 20 tests at α=0.05 → expected false positives = 20 × 0.05 = 1 (by chance!)
- **Solutions**: Bonferroni, Benjamini-Hochberg FDR, Tukey HSD
- **In TAMBEAR**: Post-hoc corrections in ANOVA available; multiple correction method step

---

## 23. Missing Data Handling

### 23.1 Listwise Deletion
- **Method**: Remove entire row if any variable has missing data
- **Advantage**: Simple; preserves complete observations; unbiased under MCAR (missing completely at random)
- **Disadvantage**: Loss of power; biased if data MNAR (missing not at random)
- **Used in**: Default in TAMBEAR correlation/regression methods

### 23.2 Pairwise Deletion
- **Method**: Use all available pairs of variables; different analyses may use different n
- **Advantage**: Retains more data than listwise
- **Disadvantage**: Inconsistency (covariance matrix may not be positive definite); difficult to report unified n
- **Used in**: Optional in correlation method (listwise=false)

### 23.3 Multiple Imputation (MI)
- **Method**: Simulate missing values m times; analyze each complete dataset; pool results
- **Advantage**: Asymptotically unbiased under MAR (missing at random); valid inference
- **Disadvantage**: Complex implementation; requires assumption about missingness mechanism
- **Used in**: Not yet in TAMBEAR; relevant for missing data method step

---

## 24. Data Transformation

### 24.1 Log Transform
- **Equation**: y' = log(y)
- **When to use**: Right-skewed data (income, reaction time, RNA-seq counts)
- **Effect**: Compresses large values; stabilizes variance
- **In regression**: Coefficient interpretation changes: exp(β) = multiplicative effect

### 24.2 Square Root Transform
- **Equation**: y' = √y
- **When to use**: Count data (if Poisson model not appropriate)
- **Variance stabilization**: Var(√y) ≈ const for Poisson data

### 24.3 Box-Cox Transform
- **Equation**: y'(λ) = (y^λ - 1) / λ (λ ≠ 0); y' = log(y) (λ = 0)
- **Optimization**: Choose λ to maximize profile likelihood
- **Used in**: Automatic transformation selection in robust regression

### 24.4 Rank Transform
- **Method**: Replace data with ranks 1, 2, ..., n (handling ties as average ranks)
- **Effect**: Makes any distribution uniform; robust to outliers
- **Used in**: Non-parametric tests (Spearman, Wilcoxon, Kruskal-Wallis)

### 24.5 Fisher z-Transform (Already in Section 1.6.1)

---

## 25. Algorithmic Complexity & Computational Notes

| Operation | Complexity | Used In |
|-----------|-----------|---------|
| Mean/variance | O(n) | All descriptive |
| Correlation (single pair) | O(n) | Correlation method |
| Correlation matrix (p variables) | O(p²n) | PCA, CFA |
| Eigendecomposition (p×p) | O(p³) per iteration | PCA, factor analysis |
| Matrix invert (p×p) | O(p³) | Regression, cov. inversion |
| Regression (p predictors, n obs) | O(np²) | Regression, mediation |
| k-means (n points, k clusters, d dims) | O(nkd·iters) | K-means clustering |
| Silhouette (n points, k clusters) | O(n²) | Clustering quality |
| Permutation test (m tests) | O(m × n) | Permutation method |
| Bootstrap (B resamples) | O(B × computation) | Correlation CI, mediation |
| Parallel analysis (nRep perms) | O(nRep × p³) | Factor analysis |

---

## 27. References & Standards

### Canonical Publications
- Abramowitz & Stegun (1964) — "Handbook of Mathematical Functions" (erf, normal quantile)
- Arthur & Vassilvitskii (2007) — "k-means++: The Advantages of Careful Seeding"
- Box & Jenkins (1976) — "Time Series Analysis: Forecasting and Control" (ARIMA)
- Cohen (1988) — "Statistical Power Analysis for the Behavioral Sciences" (effect sizes)
- Horn (1965) — "A Rationale and Test for the Number of Factors in Factor Analysis"
- Kruskal & Wallis (1952) — "Use of Ranks in One-Criterion Variance Analysis"
- Mann & Whitney (1947) — "On a Test of Whether One of Two Random Variables is Stochastically Larger"
- Numerical Recipes (Press et al. 1992) — LU decomposition, Jacobi eigendecomposition, Lanczos gamma
- Shapiro & Wilk (1965) — "An Analysis of Variance Test for Normality"
- Veach (1997) — "Robust Monte Carlo Methods for Light Transport Simulation" (path space, path integral rendering)
- Wilcoxon (1945) — "Individual Comparisons by Ranking Methods"

### TAMBEAR Implementation Standards
- APA format for p-values, effect sizes, confidence intervals
- Two-tailed tests default; one-tailed available in advanced options
- Bessel correction (n-1) for sample variance/SD
- 95% CI default; α = 0.05 default significance level
- Power analysis: target power = 0.80, α = 0.05

### Photon Field Research References
- Jensen (1996) — "Photon maps in bidirectional Monte Carlo ray tracing" (photon mapping)
- Tymoczko (2011) — "A Geometry of Music" (Grassmannian geometry applications)
- Kajiya (1986) — "The Rendering Equation" (path integral rendering)

---

---

## 26. Document Analysis & Prose Intelligence

The prose intelligence layer (`src/editor/analyze.ts`, `src/layout/tracks/`, `src/editor/citations/`) provides real-time linguistic analysis of documents. Seven overlay layers + five timeline tracks + spatial readability gauges form the document intelligence layer. All metrics compute per-line (Audience layer) or per-paragraph (tracks) to reveal document structure and quality.

### 15.1 Readability Formulas (Audience Layer)

All formulas built via `text-readability` NPM module (wrapper around classical formulas) + custom syllable counting. Applied per-line; only outliers (±3 grades from document mean) are highlighted.

#### 15.1.1 Flesch-Kincaid Grade Level
- **Equation**: $\text{GradeLevel} = 0.39 \times \text{(words/sentence)} + 11.8 \times \text{(syllables/word)} - 15.59$
- **Implementation**: `src/editor/analyze.ts:425-437`, wrapped via `text-readability.fleschKincaidGrade()`
- **Range**: 0-18+ (0-6 = early elementary; 8-10 = general audience; 12+ = college/technical)
- **Used in**: Audience layer highlights, pacing track per-paragraph visualization, readability track over versions
- **Properties**: Balance of sentence complexity + word difficulty; penalizes short words and sentences (low grade)
- **Note**: Original formula by Flesch (1948, re-weighted by Kincaid et al. 1975) — the standard US readability metric

#### 15.1.2 Coleman-Liau Index
- **Equation**: $\text{CLI} = 0.0588 \times \text{L} - 0.296 \times \text{S} - 15.8$, where L = chars/100 words, S = sentences/100 words
- **Implementation**: `src/editor/analyze.ts:433`, via `text-readability.colemanLiauIndex()`
- **Range**: 0-18+
- **Advantage**: Uses character count instead of syllables (more robust to mispronunciation)
- **Comparison to FK**: Often similar; CLI avoids syllable ambiguity

#### 15.1.3 Gunning Fog Index
- **Equation**: $\text{FOG} = 0.4 \times \left(\text{words/sentence} + 100 \times \frac{\text{complex words}}{\text{words}}\right)$, where "complex" = 3+ syllables (excluding proper nouns, 3-syllable combinations)
- **Implementation**: `src/editor/analyze.ts:434`, via `text-readability.gunningFog()`
- **Range**: 0-18+
- **Emphasis**: Counts difficult (polysyllabic) words explicitly; designed for business writing
- **Interpretation**: Grade level (4 = fourth grade readable)

#### 15.1.4 SMOG Index (Simple Measure of Gobbledygook)
- **Equation**: $\text{SMOG} = 1.0430 \times \sqrt{3 \times \text{(polysyllabic words)}} + 3.1291$
- **Implementation**: `src/editor/analyze.ts:435`, via `text-readability.smogIndex()`
- **Range**: 0-20
- **Design**: Developed for healthcare writing; very conservative (tends to estimate higher grades than other formulas)
- **Accuracy**: Claims 68% accuracy (±1.5 grades)

#### 15.1.5 Automated Readability Index (ARI)
- **Equation**: $\text{ARI} = 4.71 \times \text{(chars/words)} + 0.5 \times \text{(words/sentences)} - 21.43$
- **Implementation**: `src/editor/analyze.ts:436`, via `text-readability.automatedReadabilityIndex()`
- **Range**: 0-18+
- **Advantage**: Character-based (no syllable counting); fast to compute
- **History**: Originally designed for typewriter-era readability assessment (DuBay, 2004)

#### 15.1.6 Dale-Chall Readability Score
- **Equation**: $\text{DC} = 0.1579 \times \text{(\% difficult words)} + 0.0496 \times \text{(words/sentence)}$; adjust if < 4.9 or > 9.9
- **Implementation**: `src/editor/analyze.ts:437`, via `text-readability.daleChallReadabilityScore()`
- **Range**: 0.9-10.0 (converted to grade level scale)
- **Reference**: Uses Dale-Chall 3000-word list of common English words; marks 3000+ not-in-list as "difficult"
- **Reliability**: Generally considered most reliable for K-12 range

#### 15.1.7 Consensus Grade (Median of Six Formulas)
- **Equation**: $\text{Grade} = \text{median}(\text{FK, CLI, FOG, SMOG, ARI, DC})$
- **Implementation**: `src/editor/analyze.ts:426`, via `text-readability.textMedian()`
- **Rationale**: Different formulas emphasize different aspects; median is robust to outlier formulas and smooths variations
- **Used in**: Primary display grade in Audience layer; stored per-line in `ReadabilityBreakdown`
- **Cache**: Per-line breakdowns cached in `_audienceBreakdowns` map; rebuilt if text changes

#### 15.1.8 Syllable Counting
- **Source**: `syllable` NPM module (accurate phonetic counting)
- **Fallback**: Simple heuristic: count vowel groups, subtract 1 if word ends in 'e', minimum 1
- **Implementation**: `src/editor/analyze.ts:345-349` (calls syllable package); `src/editor/citations/paper-analysis.ts:113-120` (heuristic fallback)
- **Used in**: FK, Gunning Fog, SMOG formulas
- **Accuracy**: Package uses phonetic rules; heuristic suitable for bulk analysis when package unavailable

### 15.2 Source & Claim Analysis Layer

Identifies cited claims vs. unsourced assertions; per-line inline highlighting.

#### 15.2.1 Citation Pattern Detection
- **Regex patterns**: Six patterns detect academic citations
  1. `\.\s*\(\d+(?:,\s*\d+)*\)` — `(1)`, `(1, 2)` after sentence end (superscript-style brackets)
  2. `\)\s*\(\d+(?:,\s*\d+)*\)` — `)(\d+)` (parenthetical citation after clause)
  3. `\[\d+(?:,\s*\d+)*\]` — `[1]`, `[1, 2]` (square bracket style)
  4. `[A-Z][a-z]+\s+et\s+al\.\s*\(` — `Smith et al. (` (Author et al.)
  5. `[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s*\(\d{4}\)` — `Smith and Jones (2020)` (Author and Author year)
  6. `[A-Z][a-z]{2,}\s*\(\d{4}\)` — `Smith (2020)` (Author year, 3+ char name required)
- **Implementation**: `src/editor/analyze.ts:190-197`
- **Matching**: Case-insensitive except author name (3+ chars, capitalized for pattern 6)
- **Score**: Lines with citations marked as category "cited" (score 1.0)

#### 15.2.2 Claim Pattern Detection
- **Regex patterns**: Nine patterns detect unsourced claim language
  1. `\bresults?\s+(demonstrate|show|indicate|suggest|reveal)\b`
  2. `\bevidence\s+(suggests?|indicates?|shows?)\b`
  3. `\bfindings?\s+(suggest|indicate|show|demonstrate|reveal)\b`
  4. `\bstudies?\s+(have\s+)?(shown|found|demonstrated|revealed)\b`
  5. `\bis\s+(associated|correlated|linked|related)\s+with\b`
  6. `\bsignificant(ly)?\s+(increase|decrease|difference|effect)\b`
  7. `\bclearly\b` (standalone assertion marker)
  8. `\bundeniably\b` (standalone assertion marker)
  9. `\b(it\s+is|there\s+is)\s+(clear|evident|obvious|well.?known)\b` (epistemic stance)
- **Implementation**: `src/editor/analyze.ts:199-209`
- **Matching**: Case-insensitive word boundary matching
- **Scoring**: Lines with claim markers but no citations marked as category "claim" (score 0.8); without citations, "unsourced claim"

#### 15.2.3 Source Analysis per-Line Logic
- **Algorithm**: `analyzeSource()` at `src/editor/analyze.ts:248-292`
1. Split text by lines; skip headings and blank lines
2. Find all citation + claim pattern matches per line with inline highlighting
3. Assign category based on presence/absence of citations:
   - `cited`: Has citation patterns → score 1.0 (well-sourced)
   - `claim`: Has claim markers but no citations → score 0.8 (unsourced assertion)
   - `neutral`: Neither → score 0.2
- **Highlights**: Inline decorations mark exact citation and claim phrases for click-to-source workflow

### 15.3 Certainty & Hedging Layer

Distinguishes between definitive (high confidence) vs. hedged (tentative) language; per-line inline highlighting.

#### 15.3.1 Hedge Patterns (Tentative Language)
- **Patterns**: 12 markers of qualification/uncertainty
  1. `\bmay\b` — Modal auxiliary of permission/possibility
  2. `\bmight\b` — Counterfactual possibility
  3. `\bcould\b` — Conditional capability
  4. `\bpossibly\b` — Adverbial qualifier
  5. `\bperhaps\b` — Uncertainty marker
  6. `\bsuggests?\b` — Evidential softening
  7. `\bappears?\b` — Perceptual qualification
  8. `\bseems?\b` — Epistemic uncertainty
  9. `\btends?\sto\b` — Frequency softening
  10. `\blikely\b` — Probabilistic qualifier
  11. `\bpotentially\b` — Potentiality marker
  12. `\bpresumably\b` — Assumption qualification
  + Phrasal patterns: `it is possible`, `it is plausible`
- **Implementation**: `src/editor/analyze.ts:213-219`
- **Matching**: Case-insensitive; word-boundary respecting
- **Score**: Lines with hedges scored 0.3 (tentative); count of hedge markers reported

#### 15.3.2 Definitive Patterns (High Confidence Language)
- **Patterns**: 8 markers of certainty/assertion
  1. `\bclearly\b` — Clarity assertion
  2. `\bdefinitely\b` — Absolute affirmation
  3. `\bundeniably\b` — Logical necessity
  4. `\bcertainly\b` — Certainty marker
  5. `\bdemonstrates?\b` — Proof language
  6. `\bproves?\b` — Logical entailment
  7. `\bestablishes?\b` — Foundation assertion
  8. `\bconfirms?\b` — Verification language
  + Modal assertions: `must`, `without doubt`
- **Implementation**: `src/editor/analyze.ts:221-226`
- **Matching**: Case-insensitive; full-word matching
- **Score**: Lines with definitive markers scored 0.9; count reported

#### 15.3.3 Certainty Analysis per-Line Logic
- **Algorithm**: `analyzeCertainty()` at `src/editor/analyze.ts:294-339`
1. Split by lines; skip headings and blank lines
2. Find all hedge + definitive pattern matches with inline highlighting
3. Assign category by dominance:
   - `hedged`: Hedge markers present → score 0.3
   - `definitive`: Definitive markers present → score 0.9
   - `neutral`: Neither → score 0.5
- **Detail**: Count of markers reported (e.g., "3 hedge markers", "2 definitive markers")

### 15.4 Coherence & Semantic Continuity

**Status**: Dark code. Placeholder functions in `analyze.ts:177-186` (`analyzeFlow`, `analyzeArgument`) have `TAMBEAR: removed prose module` comments. Full implementations exist in archived prose modules; reintegration pending. Intended to track:
- **Flow**: Sentence-to-sentence semantic drift; discourse connectors; paragraph-level cohesion
- **Argument**: Claim density per paragraph; evidence density ratio; transition quality between claims and support

### 15.5 Resonance & Internal Echo Detection

**Status**: Dark code stub at `src/editor/analyze.ts:498-500`. Full implementation removed in TAMBEAR transition.

#### 15.5.1 Intended Concept
- **Goal**: Detect word/phrase repetition patterns (dissonance) vs. rhetorical callbacks (harmony)
- **Inputs**: Lexical frequency vectors per paragraph
- **Outputs**: Harmony (strategic echo), dissonance (unintended repetition), silent (no echoes)
- **Used in**: TAMBEAR timeline/craft signal (future expedition seed)
- **Reference**: Memory file `metapca-writer-strategies.md` — future bridge via MetaPCA cross-commit analysis

### 15.6 Citation Density Track (Temporal)

Per-commit line chart showing citation support density over versions; integrates with git history.

#### 15.6.1 Citation Density Formula
- **Calculation**: (Number of lines with ≥1 citation) / (Total prose lines) × 100%
- **Input**: Citation pattern matches from `analyzeSource()` per commit
- **Visualization**: Line chart mapped to timeline history axis; color-coded (green = well-cited, yellow = moderate, orange = sparse, red = uncited)
- **Used in**: Quality track composite (15% weight)

### 15.7 Pacing Track (Spatial)

Per-paragraph readability waveform mapped to character offset within current document.

#### 15.7.1 Paragraph Segmentation
- **Delimiters**: Blank lines, headings (`^#+`), blockquotes (`^[>|]`)
- **Implementation**: `src/layout/tracks/pacing-track.ts:162-226` (`_compute()` method)
- **Output**: Array of `PacingDataPoint`: `{paraIdx, from, to, grade, sentenceCount, avgSentenceLen}`

#### 15.7.2 Per-Paragraph Grade Aggregation
- **Algorithm**:
  1. Segment text into paragraphs by line delimiters
  2. For each line in paragraph, fetch cached `ReadabilityBreakdown` (median grade)
  3. Compute paragraph grade = median of all line grades
  4. Count sentences via regex split `[.!?]+`
  5. Compute words/sentence = total words / sentence count
- **Implementation**: `src/layout/tracks/pacing-track.ts:176-202`
- **Grade Color**: Green (≤8), yellow (8-14), red (>14)

#### 15.7.3 Spatial Visualization
- **Axis**: Horizontal (left to right) maps document character range (0 to text.length)
- **Bar height**: FK grade (0-20 max, clipped); opacity gradient
- **Median line**: Dashed line at document average grade
- **HitTest**: Hover reveals paragraph grade, sentence/word stats, jump-to-paragraph link

### 15.8 Quality Track (Temporal)

Per-commit composite score integrating six prose dimensions; line chart over version history.

#### 15.8.1 Quality Composite Score Formula
- **Dimensions**: Argument (claim+evidence), coherence, vocabulary diversity, register consistency, rhetoric, citation density
- **Aggregation**: Weighted average or geometric mean of six normalized 0-1 scores
- **Implementation**: `src/layout/tracks/quality-track.ts` display only; computation upstream (likely in step-runner.ts or export pipeline)
- **Range**: 0-1 (0 = early draft, 1 = publication-ready)

#### 15.8.2 Quality Assessment Mapping
- **Score ≥ 0.7**: "strong" (green)
- **Score 0.5-0.7**: "developing" (yellow)
- **Score 0.3-0.5**: "needs work" (orange/peach)
- **Score < 0.3**: "early draft" (red)
- **Live badge**: Right-aligned "75% strong" (example) with git commit message on hover

### 15.9 Document Language Detection

Wrapper around `franc` language detection; guards readability formulas (English-only).

#### 15.9.1 Language Detection via Franc
- **Source**: `franc` NPM module (n-gram based language identification)
- **Input**: Full document text (franc requires ~100+ chars for accuracy)
- **Output**: ISO 639-3 language code (e.g., `eng`, `fra`, `deu`)
- **Implementation**: `src/editor/analyze.ts:370-376`
- **Caching**: Result cached per text; regenerated on text change

#### 15.9.2 Readability Guard Logic
- **Rules**: Apply readability analysis only if detected language is English (`eng`), Scottish English (`sco`), or Undetermined (`und`)
- **Non-English fallback**: Return neutral results (score 0) per line to avoid meaningless grades
- **Implementation**: `src/editor/analyze.ts:393-401` (`analyzeAudience()` guard)

### 15.10 Analysis Layer Registration System

Extensible layer architecture allowing runtime registration of new analysis layers.

#### 15.10.1 Layer Configuration
- **Interface**: `LayerConfig` specifies layer ID, label, symbol (emoji), description, category color map
- **Example**:
  ```typescript
  {
    id: "source", label: "Source", symbol: "📚",
    description: "Which claims have citations and which don't...",
    categories: {
      cited: { color: "#9ece6a", label: "Cited" },
      claim: { color: "#f7768e", label: "Unsourced claim" },
      neutral: { color: "#414868", label: "Neutral" },
    }
  }
  ```
- **Implementation**: `src/editor/analyze.ts:45-52`, 81-174`

#### 15.10.2 Layer Function Contract
- **Function type**: `LayerFn = (text: string) => Map<number, LayerResult>`
  - Input: Full document text
  - Output: Map of line index (0-based) → analysis result
  - `LayerResult`: score (0-1), category (string), detail (string), optional highlights array
- **Per-line assumption**: Analysis operates on `\n`-split lines; line indices are map keys

#### 15.10.3 Layer Registration & Retrieval
- **Registration**: `registerAnalysisLayer(config, fn, aiEnhancement?)` stores in module-private maps
- **Retrieval**: `getRegisteredLayers()` returns configs in insertion order
- **Initialization**: Built-in layers auto-registered in `initBuiltinLayers()` (called at startup)
- **Extensibility**: New layers (e.g., AI-powered summary, syntax errors) can register via public API

### 15.11 Paper Analysis (Citation Intelligence)

Separate module for analyzing research paper abstracts and metadata.

#### 15.11.1 Topic Extraction via Bigram Frequency
- **Algorithm**:
  1. Tokenize text into lowercase words; filter by length ≥3
  2. Extract consecutive word pairs (bigrams), skip stopwords
  3. Count frequency; filter bigrams appearing ≥2 times
  4. Sort by frequency; return top 8
- **Implementation**: `src/editor/citations/paper-analysis.ts:44-69` (`extractTopics()`)
- **Output**: Array of phrases (e.g., `["machine learning", "neural networks", ...]`)

#### 15.11.2 Key Term Extraction via Frequency
- **Algorithm**:
  1. Tokenize into lowercase words; filter by length > 3
  2. Skip stopwords (150-word list: common articles, verbs, connectors)
  3. Count term frequency in full text
  4. Sort by frequency; return top 15 terms
- **Implementation**: `src/editor/citations/paper-analysis.ts:13-42` (`extractKeyTerms()`)
- **Stopwords**: See embedded list (lines 22-30)

#### 15.11.3 Sentiment Score (Paper Abstract)
- **Algorithm**:
  1. Count positive words (23-word list: "good", "excellent", "novel", ...)
  2. Count negative words (24-word list: "bad", "failure", "limited", ...)
  3. Score = (pos - neg) / (pos + neg), range [-1, 1]
- **Implementation**: `src/editor/citations/paper-analysis.ts:71-96` (`computeSentiment()`)
- **Interpretation**: >0 = positive framing; <0 = cautious/critical framing; 0 = balanced

#### 15.11.4 Paper Readability (simplified FK)
- **Algorithm**:
  1. Count sentences via `[.!?]+` split
  2. Count words via whitespace split
  3. Count syllables: sum `countSyllables(word)` over all words
  4. Heuristic: vowel group count, minus 1 if ends in 'e', minimum 1
  5. FK formula: `0.39 × (words/sent) + 11.8 × (syl/word) - 15.59`
- **Implementation**: `src/editor/citations/paper-analysis.ts:98-111` (`computeReadability()`)
- **Used in**: `PaperRecord.pithAnalysis.readability` field

#### 15.11.5 Paper Analysis API
- **`analyzeText(text)`**: Run all four analyses; return `PaperAnalysis` object
- **`analyzePaper(key)`**: Fetch paper from corpus by key; analyze abstract or title; store result
- **`analyzeAllPapers()`**: Batch analyze all papers lacking analysis; return count analyzed
- **Implementation**: `src/editor/citations/paper-analysis.ts:122-163`

### 15.12 Readability Formats & Interpretations

Summary table for quick reference of grade-to-audience mapping:

| Grade | Audience | Example | Context |
|-------|----------|---------|---------|
| 0-6 | Early elementary | "The cat sat. It is big." | Picture books, children's content |
| 7-9 | Upper elementary | "The dog ran quickly to the store." | Middle school, general audience |
| 10-12 | High school | "Furthermore, empirical evidence suggests that..." | High school textbooks, general newspapers |
| 13-15 | College/advanced | "Methodologically, the intersection of X and Y reveals..." | College textbooks, academic journals |
| 16+ | Graduate/technical | "The aforementioned homomorphic properties necessitate..." | PhD-level, technical documentation |

---

## 28. Known Gaps & Future Work

- **Kendall's τ**: Interface prepared, implementation pending
- **Mann-Whitney U, Wilcoxon, Kruskal-Wallis**: Methods listed; implementations incomplete
- **Multiple imputation**: Not yet implemented; relevant for missing data step
- **Latent Dirichlet Allocation (LDA)**: Topic modeling; not in TAMBEAR v0.1
- **QR decomposition**: More numerically stable than LU; not yet implemented
- **Document analysis prose modules** (flow, argument, resonance): Dark code; archived implementations available; reintegration planned
- **MetaPCA writer-strategies**: Bridge incomplete; requires cross-commit feature vector accumulation; architectural design complete (see memory/metapca-writer-strategies.md)
- **SVD (Singular Value Decomposition)**: Most stable factorization; not implemented
- **Nonlinear regression**: No NLS (nonlinear least squares) yet
- **Bayesian hierarchical models**: Requires MCMC; computationally intensive; not in scope for v0.1
- **Interaction terms in ANOVA**: Only main effects fully supported
- **Post-hoc power analysis**: Can be computed; not yet integrated into UI
- **Photon field full polarization**: Currently scalar θ; full Stokes 4-vector pending

---

Generated by mathematical taxonomy of Pith ecosystem, 2026-04-08.
