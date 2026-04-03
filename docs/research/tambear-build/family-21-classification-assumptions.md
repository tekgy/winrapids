# Family 21: Classification & Supervised Learning — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — A (KNN/NCM/SVM kernel = distance/GramMatrix), C (boosting = iterative)

---

## Core Insight: Most Classifiers Reduce to Existing Primitives

- KNN/NCM = F01 DistancePairs (distance matrix + argmin)
- LDA = F33 discriminant analysis (GramMatrix → generalized eigenvalues)
- SVM = GramMatrix QP (kernel trick)
- Decision Tree split = F25 entropy/gini
- GradientBoosting = F05 GradientOracle + sequential tree fitting
- Logistic Regression = F10 (already implemented)

~70 lines Phase 1 for KNN/NCM/LDA from DistancePairs.

---

## 1. K-Nearest Neighbors (KNN)

### Algorithm
For query x:
1. Compute distance d(x, x_i) for all training points
2. Find k nearest neighbors
3. Classify by majority vote (or weighted vote: w_i = 1/d_i)

### Distance Metrics
- Euclidean: √(Σ(x_i - y_i)²) — standard, from F01
- Manhattan: Σ|x_i - y_i|
- Minkowski: (Σ|x_i - y_i|^p)^{1/p}
- Mahalanobis: √((x-y)'Σ⁻¹(x-y)) — accounts for feature correlations

### Complexity
- Brute force: O(Nd) per query — GPU-parallelizable across queries
- KD-tree: O(d·log N) average for low d — CPU, bad for d > 20
- Ball tree: better for high d than KD-tree
- **GPU strategy**: brute-force distance matrix (F01 tiled) + top-k selection

### Choice of k
- k = 1: Bayes error rate ≤ R*_1NN ≤ 2·R* (Cover-Hart 1967)
- k = √N: common heuristic
- Cross-validation to select optimal k

### Edge Cases
- Ties in voting: random break or use distance-weighted voting
- k > N: use all training points (reduces to NCM)
- Feature scaling: CRITICAL — KNN is scale-dependent. Standardize features first (F06).

---

## 2. Nearest Centroid Classifier (NCM)

### Algorithm
```
ĝ(x) = argmin_g ‖x - μ̂_g‖²
```
where μ̂_g = mean of class g training points.

### Equivalent to LDA with equal covariance = σ²I (spherical).

### Shrunken Centroids (PAM — Tibshirani et al. 2002)
Shrink class centroids toward grand mean by feature:
```
μ̂_gj^s = μ̂_j + soft(μ̂_gj - μ̂_j, Δ)
```
where soft(x, Δ) = sign(x)·max(0, |x|-Δ). Features with all-zero shrunken differences are eliminated → automatic feature selection.

### Implementation: F06 per-class means → F01 distance to centroids → argmin. ~20 lines.

---

## 3. Linear Discriminant Analysis (LDA) — Classification

### Already covered in F33 (Multivariate Analysis).

As classifier: assign to class g that maximizes discriminant function:
```
δ_g(x) = x'Σ̂⁻¹μ̂_g - ½μ̂_g'Σ̂⁻¹μ̂_g + log π̂_g
```
where Σ̂ = pooled covariance, μ̂_g = class mean, π̂_g = class prior.

### Regularized LDA (Friedman 1989)
```
Σ̂_λ = (1-λ)·Σ̂ + λ·diag(Σ̂)    (shrink toward diagonal)
```
or Σ̂_λ = (1-λ)·Σ̂ + λ·I (shrink toward identity).

---

## 4. Support Vector Machine (SVM)

### Hard-Margin (linearly separable)
```
min_{w,b} ½‖w‖²
s.t.  y_i(w'x_i + b) ≥ 1  ∀i
```

### Soft-Margin (C-SVM)
```
min_{w,b,ξ} ½‖w‖² + C·Σ ξ_i
s.t.  y_i(w'x_i + b) ≥ 1 - ξ_i,  ξ_i ≥ 0
```

### Dual Form
```
max_α Σ α_i - ½ Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
s.t.  0 ≤ α_i ≤ C,  Σ α_i y_i = 0
```
where K(x_i, x_j) is the kernel (inner product in feature space).

### Kernel Trick
| Kernel | K(x, y) | Parameters |
|--------|---------|------------|
| Linear | x'y | — |
| RBF (Gaussian) | exp(-γ‖x-y‖²) | γ = 1/(2σ²) |
| Polynomial | (γx'y + r)^d | γ, r, d |
| Sigmoid | tanh(γx'y + r) | γ, r |

### CRITICAL: K(x_i, x_j) IS a GramMatrix (the kernel matrix). Computing it = tiled_accumulate with kernel function.

### Solving the QP (Quadratic Program)
- **SMO (Sequential Minimal Optimization — Platt 1999)**: Iteratively optimize pairs of α_i. Standard for SVM.
- **Interior point**: For moderate N (< 10K)
- **GPU**: Kernel matrix computation is embarrassingly parallel (F01 tiled). SMO is sequential (Kingdom C).

### Multi-class
- **One-vs-Rest (OvR)**: K binary classifiers, pick highest score
- **One-vs-One (OvO)**: K(K-1)/2 binary classifiers, majority vote
- **Direct multi-class**: Crammer-Singer formulation

### Edge Cases
- C → ∞: hard margin (overfits if not linearly separable)
- C → 0: all slack, classifier ignores data
- γ → ∞ (RBF): each point is its own support vector (overfit)
- γ → 0 (RBF): constant decision function (underfit)
- Unbalanced classes: use class-weighted C (C_g = C · n/n_g)

---

## 5. Decision Trees (CART)

### Splitting Criterion
For binary split of node t into t_L, t_R:
```
Gain = I(t) - (n_L/n)·I(t_L) - (n_R/n)·I(t_R)
```

### Impurity Measures
| Measure | Formula | Use |
|---------|---------|-----|
| **Gini** | Σ_k p_k(1-p_k) = 1 - Σ p_k² | Default for classification |
| **Entropy** | -Σ p_k log₂ p_k | Information gain |
| **MSE** | Σ(y_i - ȳ)² / n | Regression trees |

### CRITICAL: Gini and Entropy give nearly identical splits in practice. Gini is cheaper (no log).

### Best Split Search
For feature j, sorted values x_{(1)j} ≤ ... ≤ x_{(n)j}:
- Try all n-1 split points
- Compute impurity for each split via running counts
- **GPU**: Parallel across features. Within feature: sequential scan of sorted values (Kingdom B).

### Pruning
- **Pre-pruning**: Stop splitting when gain < threshold or min samples reached
- **Cost-complexity pruning (CART)**: Find subtree T_α minimizing R_α(T) = R(T) + α|T̃| via weakest link cutting. Select α by cross-validation.

### Edge Cases
- Constant feature: skip (no valid split)
- All samples same class: leaf node
- Missing values: surrogate splits (CART) or separate missing branch

---

## 6. Random Forest

### Algorithm (Breiman 2001)
```
For b = 1 to B:
    1. Bootstrap sample of size n from training data
    2. Grow tree on bootstrap sample, at each node:
       - Select m random features (m = √p for classification, p/3 for regression)
       - Find best split among m features
    3. Grow to maximum depth (no pruning)
Prediction: majority vote of B trees
```

### Out-of-Bag (OOB) Error
Each tree doesn't see ~37% of data (not in bootstrap). Predict those using trees that didn't see them → unbiased error estimate without cross-validation.

### Variable Importance
- **Permutation importance**: Permute feature j in OOB data, measure accuracy drop
- **Gini importance**: Total decrease in Gini from splits on feature j (biased toward high-cardinality features)

### GPU: Trees are independent → embarrassingly parallel across trees. Within tree: splits are parallel across features.

---

## 7. Gradient Boosting (GBDT)

### Algorithm
```
F₀(x) = argmin_c Σ L(y_i, c)    (initialize with constant)
For m = 1 to M:
    r_{im} = -∂L(y_i, F_{m-1}(x_i))/∂F_{m-1}    (pseudo-residuals)
    Fit tree h_m to pseudo-residuals
    F_m(x) = F_{m-1}(x) + η · h_m(x)
```

### This IS a GradientOracle pattern (F05)
The pseudo-residual r = -∂L/∂F is the gradient of the loss with respect to the function value. Each boosting iteration = one gradient step in function space.

### Loss Functions
| Task | Loss | Pseudo-residual |
|------|------|----------------|
| Regression | ½(y-F)² | y - F |
| Classification | log(1+e^{-yF}) | y·σ(-yF) |
| Quantile | ρ_τ(y-F) | τ - I(y<F) |
| Huber | Huber(y-F) | clamp(y-F, -δ, δ) |

### Key Hyperparameters
- **Learning rate η**: 0.01-0.3 (smaller = more trees needed but better)
- **Max depth**: 3-8 (shallow trees for boosting)
- **Subsample**: 0.5-0.8 (stochastic gradient boosting, like mini-batch SGD)
- **L2 regularization λ**: on leaf weights
- **Min split gain γ**: minimum loss reduction to split

### XGBoost Split Criterion
```
Gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
```
where G = Σ g_i (gradient sum), H = Σ h_i (Hessian sum) within each child.

### Kingdom: C (sequential boosting rounds — each depends on previous). Within each round: tree building = A (parallel split search).

---

## 8. Naive Bayes

### Formula
```
P(C|x) ∝ P(C) · Π_j P(x_j|C)
```

### Variants
| Variant | P(x_j\|C) | Use |
|---------|----------|-----|
| Gaussian | N(μ_{jC}, σ²_{jC}) | Continuous features |
| Multinomial | Multinomial(θ_C) | Text (word counts) |
| Bernoulli | Bernoulli(p_{jC}) | Binary features |
| Complement | 1 - MultinomialComplement | Imbalanced text |

### Implementation: F06 per-class MomentStats → Gaussian parameters. Classification = log-sum.

### Laplace Smoothing (for multinomial)
```
θ̂_{jC} = (N_{jC} + α) / (N_C + α·V)
```
where α = 1 (Laplace), V = vocabulary size.

---

## 9. Evaluation Metrics

### Confusion Matrix Derived
| Metric | Formula |
|--------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) |
| Precision | TP/(TP+FP) |
| Recall/Sensitivity | TP/(TP+FN) |
| Specificity | TN/(TN+FP) |
| F1 | 2·Precision·Recall/(Precision+Recall) |
| MCC | (TP·TN-FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)) |

### ROC / AUC
- ROC curve: TPR vs FPR at all thresholds
- AUC = P(score(positive) > score(negative)) = concordance
- **Computation**: Sort by score, count concordant/discordant pairs. O(n log n) via merge sort.

### Multi-class: Macro (average per class), Micro (pool all), Weighted (by class frequency).

### Calibration
- Platt scaling: fit logistic regression on SVM/tree scores → calibrated probabilities
- Isotonic regression: monotone fit to reliability diagram

---

## Sharing Surface

### Reuse from Other Families
- **F01 (Distance)**: KNN distance computation, NCM centroid distances
- **F06 (Descriptive)**: Per-class means (NCM, Naive Bayes), standardization
- **F10 (Regression)**: Logistic regression (classification), IRLS
- **F25 (Information Theory)**: Entropy/Gini for tree splits
- **F33 (Multivariate Analysis)**: LDA = discriminant analysis
- **F05 (Optimization)**: SVM QP, gradient boosting oracle

### Structural Rhymes
- **NCM = LDA with spherical covariance**: same as Mahalanobis with Σ=I
- **SVM kernel matrix = GramMatrix**: same tiled accumulate as F10
- **Gradient boosting = functional gradient descent**: same as F05 GD in function space
- **Random Forest = bagged decision trees**: bootstrap = F08 resampling
- **Tree split = greedy entropy reduction**: same as F25 mutual information

---

## Implementation Priority

**Phase 1** — Distance-based (~70 lines):
1. KNN (F01 distance matrix + top-k selection)
2. NCM (F06 class means + F01 distances)
3. LDA classifier (F33 discriminant + Bayes rule)

**Phase 2** — Trees (~200 lines):
4. Decision Tree (CART: Gini splits, cost-complexity pruning)
5. Random Forest (bootstrap + random feature subsets)
6. Gradient Boosting (pseudo-residuals + sequential trees)

**Phase 3** — SVM + others (~150 lines):
7. SVM (kernel matrix + SMO solver)
8. Naive Bayes (Gaussian, Multinomial, Bernoulli)
9. Shrunken Centroids (PAM)

**Phase 4** — Evaluation (~80 lines):
10. Confusion matrix + all derived metrics
11. ROC curve + AUC
12. Cross-validation (k-fold, stratified, LOO)
13. Calibration (Platt, isotonic)

---

## Composability Contract

```toml
[family_21]
name = "Classification & Supervised Learning"
kingdom = "A (distance-based) + C (boosting/SVM)"

[family_21.shared_primitives]
knn = "F01 distance matrix + argmin top-k"
tree_split = "F25 entropy/Gini impurity reduction"
svm_kernel = "GramMatrix with kernel function"
boosting = "F05 GradientOracle + sequential tree loop"

[family_21.reuses]
f01_distance = "KNN, NCM distance computation"
f06_descriptive = "Per-class means, standardization"
f10_regression = "Logistic regression, IRLS"
f25_information = "Entropy/Gini for splits"
f33_multivariate = "LDA discriminant functions"
f05_optimization = "SVM QP, boosting gradient"

[family_21.provides]
classifiers = "KNN, NCM, LDA, SVM, Tree, RF, GBDT, NB"
predictions = "Class labels + probabilities"
feature_importance = "Permutation, Gini, SHAP"
evaluation = "Accuracy, F1, AUC, calibration"

[family_21.consumers]
f20_clustering = "Cluster validation via classification"
fintek = "Signal classification tasks"
```
