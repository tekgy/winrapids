# F21 Sharing Surface: Classification as Distance + Kernel + Trees

Created: 2026-04-01T06:24:58-05:00
By: navigator

Prerequisites: F01 complete (DistancePairs), F10 complete (GramMatrix + logistic), F25 complete (entropy/information).

---

## The Classification Zoo: Three Architectural Groups

Classification algorithms split into three groups by their primitive consumption:

| Group | Algorithms | Primitives |
|-------|-----------|-----------|
| **Distance-based** | KNN, NearestCentroid, LVQ | DistancePairs trunk (F01) |
| **Kernel-based** | SVM, Kernel SVM, Gaussian Process | GramMatrix kernel + QP |
| **Tree/Ensemble** | Decision Tree, Random Forest, Gradient Boosting | F25 entropy + Kingdom C |

---

## Group 1: Distance-Based Classifiers

### KNN Classifier

**Pure DistancePairs extraction.** Already designed for KNN regressor in F20/F21.

```
Predict class(q) = majority_vote(labels[k_nearest_neighbors(q)])
```

When DistancePairs is cached in TamSession: zero new GPU compute.
The only new code: argpartition to find k nearest (ArgMinOp K times), majority vote aggregation.

**Weighted KNN**: weight by 1/distance. Same ArgMinOp + weighted majority via scatter_phi.

### Nearest Centroid (NCM)

```
Train: centroid_k = mean(X[labels==k]) — F06 grouped mean (ByKey)
Predict: class(q) = argmin_k ||q - centroid_k||² — DistancePairs on centroids
```

Train = F06 grouped mean (zero GPU, pure extraction from MomentStats).
Predict = K distances from query to K centroids — O(p×K) = trivial.
**Total new code: ~10 lines.** Most leveraged classifier in the library.

### Linear Discriminant Analysis (LDA)

```
W = Σ_w^{-1} Σ_b    (within-class vs between-class scatter ratio)
Projection: X_lda = X W
```

- `Σ_w` = within-class GramMatrix (centered by class mean) = RefCenteredStats grouped by class
- `Σ_b` = between-class GramMatrix = F10 GramMatrix on class centroids
- `W` = eigendecomp of `Σ_w^{-1} Σ_b` = generalized eigenvalue problem = F22 infrastructure

LDA spans F10 (GramMatrix) + F22 (EigenDecomp) + F06 (grouped centering). No new primitives.
**LDA IS discriminant CCA** (special case of F33 CCA with binary/categorical Y).

---

## Group 2: Kernel-Based Classifiers

### Linear SVM

Hard/soft margin SVM: minimize `0.5 ||w||² + C Σ ξ_i` subject to `y_i(w·x_i + b) ≥ 1 - ξ_i`.

Dual problem: maximize `Σ α_i - 0.5 Σ_i,j α_i α_j y_i y_j x_i·x_j`.

The key inner product `x_i·x_j` = GramMatrix entry K[i,j].

**Linear SVM = constrained QP on GramMatrix.** The QP solver is the only new infrastructure.
- Phase 1: coordinate descent / SMO (Sequential Minimal Optimization) — ~150 lines
- Phase 2: full QP solver (LP/QP, similar to F05 constrained optimization)

**Support vector prediction**: `f(x) = Σ_i α_i y_i K(x_i, x) + b` — inner product with support vectors.
When GramMatrix is cached: prediction is one row of GramMatrix (inner products of x with all x_i).

### Kernel SVM (RBF / Polynomial)

Same dual formulation, but `K(x_i, x_j) = kernel(x_i, x_j)` instead of `x_i·x_j`.

- **RBF kernel**: `exp(-γ ||x_i - x_j||²)` = `exp(-γ · D²[i,j])` — free from DistancePairs D²
- **Polynomial kernel**: `(x_i·x_j + c)^d` — polynomial in GramMatrix entries
- **Sigmoid kernel**: `tanh(α x_i·x_j + c)` — sigmoid of GramMatrix entries

**All three are O(N²) extractions from the cached kernel matrix.** The kernel matrix IS the
GramMatrix (for linear), or a function of DistancePairs (for RBF), or GramMatrix (for polynomial).

### Gaussian Process Classifier

`P(y=1|x) = Φ(f(x))` where f ~ GP(0, K) and K = kernel matrix.

GP classification requires: kernel matrix K = GramMatrix-based, Cholesky(K) for inference.
EP/Laplace approximation for the non-Gaussian likelihood.

**This is F34 (Bayesian) + kernel GramMatrix.** Defer to Phase 3.

---

## Group 3: Tree and Ensemble Methods

### Decision Tree (CART)

At each split node: choose feature j and threshold t to maximize information gain:
```
IG(j, t) = H(y_parent) - [N_left/N · H(y_left) + N_right/N · H(y_right)]
```

Where H = entropy (Shannon) = F25 infrastructure.

**Split finding** = scan over all (j, threshold) pairs, compute H for each split.

For continuous features: sort feature j, scan thresholds = O(N log N) per feature.
For bucketized features: histogram H (F25 PHI_COUNT) per bucket.

**Tambear decomposition**:
1. `scatter_phi("entropy", ByFeatureBucket)` — F25 entropy per split candidate (~zero new code)
2. Information gain = parent entropy - weighted child entropies — scalar arithmetic
3. Best split = `argmax_j argmax_t IG(j, t)` — ArgMaxOp over split candidates

**Tree building** = Kingdom C outer loop: recursively split until leaf condition.
Each split decision uses F25 entropy. No new primitive.

### Random Forest

N independent decision trees on bootstrapped data + random feature subsets.
Final prediction = majority vote (classification) or mean (regression).

**No new primitives beyond Decision Tree.**
Parallelism: trees are independent = massive data parallel. Each tree = one Kingdom C loop.
With tambear's GPU: all trees build simultaneously (if N_trees × N_samples fits GPU).

### Gradient Boosting (XGBoost-style)

```
Iterate r=1..R:
  Compute pseudo-residuals: r_i = -∂L/∂f(x_i)  (gradient of loss at current prediction)
  Fit decision tree to residuals
  Update: f(x) += η · tree_r(x)
```

**Kingdom C outer loop over decision trees.** Each iteration:
- Pseudo-residuals = gradient evaluation (GradientOracle from F05)
- Fit tree to residuals = one Decision Tree training run

The pseudo-residuals for different losses:
- MSE: `r_i = y_i - f(x_i)` (simple residuals)
- Binary cross-entropy: `r_i = y_i - σ(f(x_i)) = y_i - p_i` (logistic residuals, F10)
- Huber: `r_i = ψ(y_i - f(x_i))` (robust residuals, F09)

Gradient boosting "learns" a sequence of weighted residal regressors. Each is a Kingdom C step.

---

## MSR Types F21 Produces

```rust
pub struct ClassificationModel {
    pub n_obs: usize,
    pub n_features: usize,
    pub n_classes: usize,

    pub model_type: ClassifierType,
    pub train_accuracy: f64,
    pub train_auc: Option<f64>,    // binary only

    /// Prediction function (stores model parameters):
    pub params: ClassifierParams,
}

pub enum ClassifierType {
    Knn { k: usize, metric: DistanceMetric },
    NearestCentroid,
    Lda { n_components: usize },
    LinearSvm { c: f64 },
    KernelSvm { c: f64, kernel: KernelType },
    DecisionTree { max_depth: usize, min_samples_leaf: usize },
    RandomForest { n_trees: usize, max_features: usize },
    GradientBoosting { n_estimators: usize, learning_rate: f64, max_depth: usize },
}
```

---

## Build Order

**Phase 1 (distance-based + LDA)**:
1. KNN classifier: top-k ArgMinOp on DistancePairs + majority vote (~20 lines)
2. NearestCentroid: grouped mean (F06) + distance to centroids (~10 lines)
3. LDA: RefCenteredStats per class, generalized eigenvalue via F22 (~40 lines)
4. Tests: sklearn for all three; KNN exact match expected; LDA projection match within 1e-4

**Phase 2 (Decision Tree + Random Forest)**:
1. Feature split: scatter entropy (F25) over bucketed features per split candidate
2. Tree structure: recursive Kingdom C loop (~150 lines)
3. Random Forest: parallelize tree building (~50 lines orchestration)
4. Tests: sklearn with same random seed; tree structure won't match but accuracy should be within 1%

**Phase 3 (SVM, Gradient Boosting)**:
- SVM: SMO dual QP (~200 lines) — new for F21
- Gradient Boosting: F05 GradientOracle + tree fitting loop (~100 lines)
- Tests: sklearn with same hyperparameters; accuracy match ≥ 99% of sklearn on standard datasets

**Gold standards**:
- sklearn for all — `KNeighborsClassifier`, `LinearDiscriminantAnalysis`, `SVC`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`

---

## The Lab Notebook Claim

> Classification is three groups unified: distance-based (KNN, NCM, LDA) consume DistancePairs and GramMatrix from TamSession at zero new GPU cost; kernel-based (SVM) consume the kernel GramMatrix as the QP objective; tree-based (Decision Tree, Random Forest, Gradient Boosting) build on F25 entropy for split scoring and F05 gradient oracle for boosting. LDA is the structural bridge: it IS discriminant CCA (F33), consuming both GramMatrix (F10) and EigenDecomp (F22). Phase 1 classification (KNN + NCM + LDA) costs ~70 lines of new code.
