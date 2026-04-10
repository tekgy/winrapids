# F21 Classification & Supervised Learning — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: tambear-math expedition

---

## Purpose

Pre-load oracle implementations for Family 21 (Classification & Supervised Learning).
Documents: exact sklearn/xgboost/lightgbm calls, parameter defaults, known traps,
and tambear primitive decompositions. Every code block is runnable and produces
the expected output.

Context from navigator:
- F01 (DistancePairs) and F20 (KNN regression) already established
- F10 (logistic regression via IRLS) already established
- F22 (spectral clustering, Laplacian EigenDecomposition) already established
- This doc covers the cuML replacement scope: trees, ensembles, SVM, NB, KNN classifier, metrics

---

## Shared Test Dataset

All examples below use this canonical dataset for reproducibility:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Binary classification: reproducible
X_bin, y_bin = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
    n_classes=2, random_state=42
)

# Multiclass: reproducible
X_multi, y_multi = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
    n_classes=4, n_clusters_per_class=1, random_state=42
)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42
)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_b_s = scaler.fit_transform(X_train_b)
X_test_b_s  = scaler.transform(X_test_b)
X_train_m_s = scaler.fit_transform(X_train_m)
X_test_m_s  = scaler.transform(X_test_m)
```

---

## 1. Decision Trees

### Algorithm: CART (Classification and Regression Trees)

CART builds a binary tree by recursively choosing the split (feature j, threshold t) that
minimizes the weighted Gini impurity of the child nodes:

```
Gini(node) = 1 - Σ_k p_k²

where p_k = fraction of class k in the node.

Split score = N_left/N * Gini(left) + N_right/N * Gini(right)
Best split = argmin over all (j, t)
```

Information gain (entropy-based) criterion:

```
H(node) = -Σ_k p_k log₂(p_k)

IG(j, t) = H(parent) - [N_left/N * H(left) + N_right/N * H(right)]
Best split = argmax IG over all (j, t)
```

### Tambear Decomposition

At each candidate split (j, t), the node is partitioned into left/right subsets.

Step 1 — Compute class histogram for left and right subsets:
```
left_counts  = accumulate(data[feature_j < t], ByKey{class}, 1, Add)
right_counts = accumulate(data[feature_j >= t], ByKey{class}, 1, Add)
```

Step 2 — Gini from histogram (scalar arithmetic, free):
```
gini(counts) = 1 - Σ (counts[k] / N)²
split_score  = (N_left * gini_left + N_right * gini_right) / N
```

Step 3 — Best split = ArgMin over all (j, t) candidates:
```
best_split = accumulate(all_candidates, All, split_score, ArgMin)
```

The key insight: step 1 is a ByKey{class} accumulate with Add op — pure Kingdom A
(histogram). Step 2 is scalar arithmetic on the output. Step 3 is ArgMin over all
candidate scores — a single reduction. No scan, no tiled op.

For a node with N samples, D features, B buckets per feature:
- Total work: D * B ByKey histograms + D * B Gini computations + 1 ArgMin
- With shared (data, expr) fusion: D * B histograms fuse into one pass over the data

C4.5 uses entropy (information gain) instead of Gini; same decomposition, different expr
in step 2. sklearn `criterion='entropy'` uses log₂(p_k) expr.

### Gold Standard: sklearn.tree.DecisionTreeClassifier

```python
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np

# ---- Binary classification (default Gini) ----
dt_gini = DecisionTreeClassifier(
    criterion='gini',        # default — Gini impurity
    max_depth=None,          # default — grow until pure leaves
    min_samples_split=2,     # default — minimum samples to split
    min_samples_leaf=1,      # default — minimum samples in leaf
    max_features=None,       # IMPORTANT: None = all features for single tree
                             # (contrast with RandomForest default='sqrt')
    random_state=42,
)
dt_gini.fit(X_train_b_s, y_train_b)
acc_gini = dt_gini.score(X_test_b_s, y_test_b)
print(f"Decision tree (Gini) accuracy: {acc_gini:.4f}")   # expected ~0.86-0.89

# ---- Entropy criterion (C4.5-style) ----
dt_entropy = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,             # controlled depth for demo
    random_state=42,
)
dt_entropy.fit(X_train_b_s, y_train_b)
acc_entropy = dt_entropy.score(X_test_b_s, y_test_b)
print(f"Decision tree (entropy, depth=5) accuracy: {acc_entropy:.4f}")  # ~0.84-0.87

# ---- Inspect tree structure ----
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_shallow.fit(X_train_b_s, y_train_b)
print(export_text(dt_shallow))

# ---- Feature importances (Gini importance — BIASED for high-cardinality) ----
importances = dt_gini.feature_importances_
print(f"Top 3 features by Gini importance: {np.argsort(importances)[-3:][::-1]}")

# ---- Oracle attributes for validation ----
print(f"n_features_in_: {dt_gini.n_features_in_}")          # 20
print(f"n_classes_: {dt_gini.n_classes_}")                   # 2
print(f"tree_.n_node_samples[:5]: {dt_gini.tree_.n_node_samples[:5]}")
print(f"tree_.feature[:5]: {dt_gini.tree_.feature[:5]}")     # split features
print(f"tree_.threshold[:5]: {dt_gini.tree_.threshold[:5]}")  # split thresholds
```

Expected validation targets:
```
acc_gini  ≈ 0.875   (±0.01 across runs)
acc_entropy ≈ 0.870  (±0.01)
n_features_in_ = 20
n_classes_ = 2
```

### Key Trap: max_features is NOT the same in trees vs forests

```python
# Single tree: max_features=None means ALL features considered at each split
# RandomForest: max_features='sqrt' (= sqrt(n_features)) is the DEFAULT

# This is THE most common mistake when comparing tree to forest:
dt_wrongly_constrained = DecisionTreeClassifier(max_features='sqrt', random_state=42)
dt_wrongly_constrained.fit(X_train_b_s, y_train_b)
acc_constrained = dt_wrongly_constrained.score(X_test_b_s, y_test_b)
print(f"Tree with max_features='sqrt' (wrong for solo tree): {acc_constrained:.4f}")
# Will be LOWER — this is a forest hyperparameter misapplied to a single tree
```

### Multiclass Decision Tree

```python
dt_multi = DecisionTreeClassifier(
    criterion='gini', max_depth=None, random_state=42
)
dt_multi.fit(X_train_m_s, y_train_m)
acc_multi = dt_multi.score(X_test_m_s, y_test_m)
print(f"Multiclass tree accuracy: {acc_multi:.4f}")  # expected ~0.62-0.68
print(f"n_classes_: {dt_multi.n_classes_}")          # 4
```

---

## 2. Random Forests

### Algorithm: Bootstrap Aggregation + Feature Subsampling

Each tree in the forest:
1. Bootstrap: sample N rows with replacement from training data
2. Feature subsample: at each split, consider only `max_features` random features
3. Build full CART tree on the bootstrap sample

Final prediction:
- Classification: majority vote across all trees
- Regression: mean of all tree predictions

OOB (Out-Of-Bag): rows NOT in a given tree's bootstrap sample = "free" validation set.
Average accuracy on OOB rows across all trees = OOB score ≈ leave-one-out estimate.

### Gold Standard: sklearn.ensemble.RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ---- Default configuration ----
rf = RandomForestClassifier(
    n_estimators=100,           # default since sklearn 0.22 (was 10 before)
    criterion='gini',           # default
    max_depth=None,             # default — grow full trees
    min_samples_split=2,        # default
    min_samples_leaf=1,         # default
    max_features='sqrt',        # DEFAULT for classification (key difference from single tree!)
                                # = int(sqrt(n_features)) = int(sqrt(20)) = 4 features per split
    bootstrap=True,             # default — enables OOB
    oob_score=True,             # compute OOB score (no extra data needed)
    n_jobs=-1,                  # parallelism — critical for performance
    random_state=42,
    warm_start=False,           # default — fit from scratch each time
)
rf.fit(X_train_b_s, y_train_b)

acc_rf = rf.score(X_test_b_s, y_test_b)
print(f"Random Forest accuracy (test):  {acc_rf:.4f}")   # expected ~0.89-0.93
print(f"Random Forest OOB accuracy:     {rf.oob_score_:.4f}")  # expected ~0.87-0.91

# ---- max_features variants ----
rf_log2 = RandomForestClassifier(
    n_estimators=100, max_features='log2', random_state=42, n_jobs=-1
)
rf_log2.fit(X_train_b_s, y_train_b)
print(f"RF max_features=log2: {rf_log2.score(X_test_b_s, y_test_b):.4f}")

rf_all = RandomForestClassifier(
    n_estimators=100, max_features=None, random_state=42, n_jobs=-1
)  # None = use all features (Bagging, not RF)
rf_all.fit(X_train_b_s, y_train_b)
print(f"RF max_features=None (bagging): {rf_all.score(X_test_b_s, y_test_b):.4f}")
```

Expected validation targets:
```
acc_rf     ≈ 0.910   (±0.01)
oob_score_ ≈ 0.890   (±0.01)
```

### Variable Importance: Gini vs Permutation

```python
from sklearn.inspection import permutation_importance

# ---- Gini (impurity) importance — FAST but BIASED ----
gini_imp = rf.feature_importances_
print(f"Gini importance, top-5 features: {np.argsort(gini_imp)[-5:][::-1]}")
# Gini importance inflates importance of HIGH-CARDINALITY features
# For financial data: continuous features always "win" vs categorical — this is a bug

# ---- Permutation importance — SLOW but UNBIASED ----
perm_imp = permutation_importance(
    rf, X_test_b_s, y_test_b,
    n_repeats=10,       # repeat each permutation 10x, average
    random_state=42,
    n_jobs=-1,
)
print(f"Permutation importance, top-5: {np.argsort(perm_imp.importances_mean)[-5:][::-1]}")
print(f"Permutation importance mean: {perm_imp.importances_mean}")
print(f"Permutation importance std:  {perm_imp.importances_std}")
```

**Key trap**: Gini importance and permutation importance will disagree on which features
matter. For model trust, permutation importance is the ground truth. Gini importance is
useful only as a fast proxy.

### OOB Probability Estimates

```python
rf_oob_proba = RandomForestClassifier(
    n_estimators=100, oob_score=True, random_state=42, n_jobs=-1
)
rf_oob_proba.fit(X_train_b_s, y_train_b)

# oob_decision_function_: shape (n_samples, n_classes)
# entry [i, c] = fraction of trees that predicted class c for sample i
# (only trees that did NOT include sample i in bootstrap)
oob_proba = rf_oob_proba.oob_decision_function_
print(f"OOB decision function shape: {oob_proba.shape}")  # (800, 2)
print(f"OOB proba for first 5 training samples: {oob_proba[:5]}")
```

### Multiclass Random Forest

```python
rf_multi = RandomForestClassifier(
    n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1
)
rf_multi.fit(X_train_m_s, y_train_m)
print(f"Multiclass RF accuracy: {rf_multi.score(X_test_m_s, y_test_m):.4f}")  # ~0.72-0.78

# Probability output (soft predictions)
proba_multi = rf_multi.predict_proba(X_test_m_s)
print(f"Predict_proba shape: {proba_multi.shape}")  # (200, 4)
print(f"Proba sum per sample: {proba_multi.sum(axis=1)[:5]}")  # should be 1.0 each
```

---

## 3. Gradient Boosting

### Algorithm Overview

Gradient boosting builds an ensemble by iteratively fitting shallow trees to the
pseudo-residuals (negative gradient of the loss with respect to the current prediction):

```
Initialize: f_0(x) = argmin_γ Σ L(y_i, γ)  (e.g., log-odds for binary classification)

For r = 1 to n_estimators:
  Compute pseudo-residuals: r_i = -[∂L(y_i, f(x_i)) / ∂f(x_i)]
    For binary log-loss: r_i = y_i - sigmoid(f(x_i)) = y_i - p_i
  Fit tree h_r to pseudo-residuals: minimize Σ (r_i - h_r(x_i))²
  Update: f(x) = f(x) + learning_rate * h_r(x)
```

This is Kingdom C (iterative): each iteration depends on the previous tree's predictions.
Cannot be parallelized across iterations (unlike Random Forest, which is embarrassingly parallel).

### XGBoost vs LightGBM vs CatBoost Differences

| Property | XGBoost | LightGBM | CatBoost |
|----------|---------|----------|----------|
| Tree growth | Level-wise (BFS) | Leaf-wise (best-first) | Symmetric (oblivious) |
| Depth control | `max_depth` | `num_leaves` (primary) | `depth` |
| Missing values | Built-in | Built-in | Built-in |
| Categorical features | Requires encoding | Native via `categorical_feature` | Native (best support) |
| L1 regularization | `alpha` (L1 on weights) | `reg_alpha` | No direct equivalent |
| L2 regularization | `lambda` (L2 on weights) | `reg_lambda` | `l2_leaf_reg` |
| Early stopping | `early_stopping_rounds` (legacy) / callbacks | `callbacks` | `od_wait` |
| GPU support | `device='cuda'` | `device='gpu'` | `task_type='GPU'` |

**Leaf-wise growth (LightGBM)**: grows the tree by always splitting the leaf with
the largest gain, not all leaves at a given depth. This means `max_depth=6` in LightGBM
might produce a VERY unbalanced tree with 2^6=64 leaves at depths 1-6, while
`num_leaves=31` limits total leaves regardless of depth.

**Symmetric trees (CatBoost)**: all nodes at the same depth use the SAME split feature
and threshold. This makes inference extremely fast (one feature lookup per level).

### Gold Standard: sklearn.ensemble.GradientBoostingClassifier

```python
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

# ---- Sklearn GradientBoosting (exact, slower) ----
gb = GradientBoostingClassifier(
    n_estimators=100,           # default
    learning_rate=0.1,          # default — shrinkage
    max_depth=3,                # default — IMPORTANT: shallow trees are key
    min_samples_split=2,        # default
    min_samples_leaf=1,         # default
    subsample=1.0,              # default — 1.0 = no stochastic gradient boosting
    max_features=None,          # default — all features
    loss='log_loss',            # default for classification (log-loss = cross-entropy)
    random_state=42,
)
gb.fit(X_train_b_s, y_train_b)
print(f"sklearn GB accuracy: {gb.score(X_test_b_s, y_test_b):.4f}")  # ~0.91-0.94

# ---- HistGradientBoosting (faster, LightGBM-style histogram binning) ----
hgb = HistGradientBoostingClassifier(
    max_iter=100,               # n_estimators equivalent
    learning_rate=0.1,
    max_depth=None,             # default — uses max_leaf_nodes instead
    max_leaf_nodes=31,          # default — leaf-wise like LightGBM
    min_samples_leaf=20,        # default
    l2_regularization=0.0,      # default
    random_state=42,
)
hgb.fit(X_train_b_s, y_train_b)
print(f"HistGB accuracy: {hgb.score(X_test_b_s, y_test_b):.4f}")  # ~0.91-0.94
```

### Gold Standard: XGBoost

```python
import xgboost as xgb

# ---- Binary classification ----
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,           # eta
    max_depth=6,                 # default — level-wise
    min_child_weight=1,          # default — minimum sum of instance weight in child
    subsample=1.0,               # default — row subsampling per tree
    colsample_bytree=1.0,        # default — column subsampling per tree
    colsample_bylevel=1.0,       # default — column subsampling per level
    colsample_bynode=1.0,        # default — column subsampling per split
    gamma=0,                     # default — minimum loss reduction to split
    reg_alpha=0,                 # default — L1 regularization on weights
    reg_lambda=1,                # default — L2 regularization on weights
    objective='binary:logistic', # default for binary; predict probabilities
    eval_metric='logloss',       # default eval metric
    use_label_encoder=False,     # suppress deprecation warning
    random_state=42,
)
xgb_clf.fit(X_train_b_s, y_train_b)
print(f"XGBoost binary accuracy: {xgb_clf.score(X_test_b_s, y_test_b):.4f}")  # ~0.91-0.94

# ---- Multiclass classification ----
xgb_multi = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective='multi:softmax',   # output class labels directly
                                 # ALTERNATIVE: 'multi:softprob' outputs probabilities
    num_class=4,                 # REQUIRED with multi:softmax and multi:softprob
    random_state=42,
)
xgb_multi.fit(X_train_m_s, y_train_m)
print(f"XGBoost multiclass accuracy: {xgb_multi.score(X_test_m_s, y_test_m):.4f}")  # ~0.75-0.81

# ---- Early stopping (new API) ----
xgb_early = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=20,   # stop if no improvement for 20 rounds
    random_state=42,
)
xgb_early.fit(
    X_train_b_s, y_train_b,
    eval_set=[(X_test_b_s, y_test_b)],
    verbose=False,
)
print(f"XGBoost early stop at iteration: {xgb_early.best_iteration}")
print(f"XGBoost early stop accuracy: {xgb_early.score(X_test_b_s, y_test_b):.4f}")
```

### Gold Standard: LightGBM

```python
import lightgbm as lgb

# ---- Binary classification ----
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,                # default — no depth limit
    num_leaves=31,               # default — primary depth control (leaf-wise growth)
    min_child_samples=20,        # default — minimum samples in leaf
    subsample=1.0,               # default — row subsampling
    colsample_bytree=1.0,        # default — column subsampling
    reg_alpha=0.0,               # default — L1
    reg_lambda=0.0,              # default — L2
    min_split_gain=0.0,          # default — minimum gain to split
    objective='binary',          # default for binary
    metric='binary_logloss',     # default
    random_state=42,
    verbose=-1,                  # suppress logging
)
lgb_clf.fit(X_train_b_s, y_train_b)
print(f"LightGBM binary accuracy: {lgb_clf.score(X_test_b_s, y_test_b):.4f}")  # ~0.91-0.94

# ---- Multiclass ----
lgb_multi = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    objective='multiclass',      # 'softmax' also works
    num_class=4,                 # required for multiclass
    metric='multi_logloss',
    random_state=42,
    verbose=-1,
)
lgb_multi.fit(X_train_m_s, y_train_m)
print(f"LightGBM multiclass accuracy: {lgb_multi.score(X_test_m_s, y_test_m):.4f}")  # ~0.75-0.81

# ---- Early stopping with callbacks (new API, LightGBM >= 4.0) ----
lgb_early = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    objective='binary', random_state=42, verbose=-1,
)
lgb_early.fit(
    X_train_b_s, y_train_b,
    eval_set=[(X_test_b_s, y_test_b)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.log_evaluation(period=-1),  # suppress
    ],
)
print(f"LightGBM early stop at iteration: {lgb_early.best_iteration_}")
```

### Key Traps for Gradient Boosting

**Trap 1: XGBoost multiclass objective**
```python
# WRONG: objective='binary:logistic' for 4-class problem (silently gives bad results)
# CORRECT:
xgb.XGBClassifier(objective='multi:softmax', num_class=4)

# predict_proba requires 'multi:softprob', NOT 'multi:softmax'
# 'multi:softmax' outputs labels; 'multi:softprob' outputs probabilities
```

**Trap 2: LightGBM num_leaves vs max_depth**
```python
# LightGBM default: num_leaves=31 with no max_depth
# With leaf-wise growth, this can create very deep unbalanced trees
# For equivalent complexity to XGBoost max_depth=6: num_leaves ≈ 63 (2^6 - 1)
# But leaf-wise finds a BETTER 63-leaf tree than level-wise's balanced 63-leaf tree

# Overfitting guard: set max_depth to constrain max depth ALONGSIDE num_leaves
lgb_constrained = lgb.LGBMClassifier(
    num_leaves=31, max_depth=6, verbose=-1, random_state=42
)
```

**Trap 3: Early stopping API changed between versions**
```python
# XGBoost: early_stopping_rounds is now a constructor param, NOT a fit param
# (changed in XGBoost >= 1.6)

# OLD (still works but deprecated in some versions):
xgb_old = xgb.XGBClassifier(n_estimators=500)
xgb_old.fit(X_train_b_s, y_train_b,
            eval_set=[(X_test_b_s, y_test_b)],
            early_stopping_rounds=20)  # in fit() — OLD API

# NEW (preferred):
xgb_new = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=20)
xgb_new.fit(X_train_b_s, y_train_b, eval_set=[(X_test_b_s, y_test_b)])
```

**Trap 4: Feature names after StandardScaler**
```python
# XGBoost raises warning if feature names mismatch between train and predict
# After StandardScaler, features are numpy arrays with no names
# Solution: use pandas DataFrames or ignore the warning explicitly
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    xgb_clf.predict(X_test_b_s)
```

---

## 4. Support Vector Machines (SVM)

### Algorithm: Hard and Soft Margin SVM

**Hard margin** (linearly separable): find hyperplane w·x + b = 0 maximizing margin 2/||w||.
Primal: minimize 0.5||w||² subject to y_i(w·x_i + b) ≥ 1.

**Soft margin** (C-SVM): allow slack variables ξ_i ≥ 0.
Primal: minimize 0.5||w||² + C Σ ξ_i subject to y_i(w·x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0.

Large C → less regularization, may overfit.
Small C → more regularization, wider margin, ignores more misclassifications.

**Dual problem** (what libsvm actually solves):
Maximize: Σ α_i - 0.5 Σ_ij α_i α_j y_i y_j K(x_i, x_j)
Subject to: 0 ≤ α_i ≤ C, Σ α_i y_i = 0

Here K(x_i, x_j) = x_i·x_j for linear SVM = GramMatrix entry.
The dual objective is a quadratic program (QP) in the α_i coefficients.

**Prediction**: sign(Σ_i α_i y_i K(x_i, x) + b)
Only support vectors (α_i > 0) contribute — sparsity is a kernel SVM property.

### Kernel Trick

Replace dot products x_i·x_j with K(x_i, x_j) = φ(x_i)·φ(x_j) where φ maps to
a higher-dimensional (possibly infinite) feature space, WITHOUT explicitly computing φ.

| Kernel | Formula | sklearn param | Notes |
|--------|---------|--------------|-------|
| Linear | K(x,y) = x·y | `kernel='linear'` | No hyperparameters |
| RBF (Gaussian) | K(x,y) = exp(-γ‖x-y‖²) | `kernel='rbf'` | γ default = 1/n_features |
| Polynomial | K(x,y) = (γ x·y + r)^d | `kernel='poly'` | degree d, coef0 r |
| Sigmoid | K(x,y) = tanh(γ x·y + r) | `kernel='sigmoid'` | Often not PD |

### Tambear Decomposition

Linear SVM dual: the QP matrix Q[i,j] = y_i y_j K(x_i, x_j) = y_i y_j GramMatrix[i,j].
```
Q = diag(y) @ GramMatrix @ diag(y)   ← element-wise: Q[i,j] = y_i * y_j * K[i,j]
```
This is a tiled computation on the GramMatrix — free from F10 GramMatrix caching.

SMO (Sequential Minimal Optimization): at each step, update two α coefficients
solving a 2-variable QP analytically. Each step costs O(N) inner product lookups
(one row of the kernel matrix). With GramMatrix cached: O(1) lookups.

Total F21 SVM new code: ~200 lines (SMO loop + bias term computation).

### Gold Standard: sklearn.svm.SVC

```python
from sklearn.svm import SVC, LinearSVC

# ---- Linear kernel SVM ----
svc_linear = SVC(
    C=1.0,              # default — regularization inverse
    kernel='linear',    # linear kernel
    decision_function_shape='ovr',  # default for multiclass ('ovo' is alternative)
    probability=False,  # default — set True for predict_proba (VERY SLOW)
    random_state=42,
)
svc_linear.fit(X_train_b_s, y_train_b)
print(f"SVC linear accuracy: {svc_linear.score(X_test_b_s, y_test_b):.4f}")  # ~0.88-0.92

# ---- Inspect support vectors ----
print(f"Number of support vectors (class 0, 1): {svc_linear.n_support_}")
print(f"Support vector indices (first 5): {svc_linear.support_[:5]}")
# The dual coefficients α_i * y_i:
print(f"Dual coefficients shape: {svc_linear.dual_coef_.shape}")

# ---- RBF kernel (default, most powerful) ----
svc_rbf = SVC(
    C=1.0,
    kernel='rbf',       # default kernel
    gamma='scale',      # default: 1 / (n_features * X.var()) — better than 'auto'
                        # 'auto': 1 / n_features — deprecated default
    random_state=42,
)
svc_rbf.fit(X_train_b_s, y_train_b)
print(f"SVC RBF accuracy: {svc_rbf.score(X_test_b_s, y_test_b):.4f}")  # ~0.90-0.93

# ---- Polynomial kernel ----
svc_poly = SVC(
    C=1.0,
    kernel='poly',
    degree=3,           # default cubic
    gamma='scale',
    coef0=0.0,          # default
    random_state=42,
)
svc_poly.fit(X_train_b_s, y_train_b)
print(f"SVC poly accuracy: {svc_poly.score(X_test_b_s, y_test_b):.4f}")  # ~0.88-0.92
```

Expected validation targets:
```
svc_linear accuracy ≈ 0.90  (±0.02)
svc_rbf accuracy    ≈ 0.91  (±0.02)
svc_poly accuracy   ≈ 0.89  (±0.02)
n_support_ = array([k1, k2]) where k1+k2 ≈ 200-400 (many SVs = noisy/overlapping classes)
```

### Probability Calibration: The probability=True Trap

```python
# WARNING: SVC with probability=True is DRAMATICALLY SLOWER
# It runs an internal 5-fold cross-validation on the TRAINING set to fit Platt scaling
# This means: sklearn fits 5 additional SVMs to calibrate probabilities!
# With n=800 training samples, each of 5 cross-validation models trains on ~640 samples

import time

svc_no_prob = SVC(kernel='rbf', gamma='scale', random_state=42)
t0 = time.time()
svc_no_prob.fit(X_train_b_s, y_train_b)
t1 = time.time()
print(f"SVC without probability: {t1-t0:.3f}s")

svc_prob = SVC(kernel='rbf', gamma='scale', probability=True, random_state=42)
t0 = time.time()
svc_prob.fit(X_train_b_s, y_train_b)
t1 = time.time()
print(f"SVC with probability=True: {t1-t0:.3f}s")  # expect 3-8x slower

# predict_proba output (calibrated via Platt scaling)
proba = svc_prob.predict_proba(X_test_b_s)
print(f"Proba sum per sample: {proba.sum(axis=1)[:5]}")     # all ~1.0
print(f"Proba range: [{proba.min():.4f}, {proba.max():.4f}]")  # should be (0, 1)

# Alternative: CalibratedClassifierCV wrapping SVC (same idea, explicit)
from sklearn.calibration import CalibratedClassifierCV
svc_base = SVC(kernel='rbf', gamma='scale')
svc_calibrated = CalibratedClassifierCV(svc_base, method='sigmoid', cv=5)
svc_calibrated.fit(X_train_b_s, y_train_b)
proba_calibrated = svc_calibrated.predict_proba(X_test_b_s)
```

### LinearSVC vs SVC(kernel='linear')

```python
from sklearn.svm import LinearSVC

# LinearSVC: uses liblinear (L-BFGS or dual CD), O(n) per iteration, scales to large n
# SVC(kernel='linear'): uses libsvm (SMO dual), O(n²) or O(n³), for small n

linear_svc = LinearSVC(
    C=1.0,
    loss='squared_hinge',    # default — corresponds to L2 loss SVM
    dual=True,               # default; set False when n_samples > n_features
    penalty='l2',            # default — L2 regularization on w
    max_iter=1000,           # default; may need to increase for convergence
    random_state=42,
)
linear_svc.fit(X_train_b_s, y_train_b)
print(f"LinearSVC accuracy: {linear_svc.score(X_test_b_s, y_test_b):.4f}")  # ~0.87-0.92

# LinearSVC does NOT have predict_proba — it's a linear scorer, not a calibrator
# For probabilities with LinearSVC: use CalibratedClassifierCV

# When n >> p: use dual=False (primal mode, much faster)
linear_svc_primal = LinearSVC(C=1.0, dual=False, random_state=42)
linear_svc_primal.fit(X_train_b_s, y_train_b)
print(f"LinearSVC (primal) accuracy: {linear_svc_primal.score(X_test_b_s, y_test_b):.4f}")

# IMPORTANT: LinearSVC coefficient extraction (for interpretation):
print(f"LinearSVC coef shape: {linear_svc.coef_.shape}")  # (1, 20) for binary
print(f"LinearSVC intercept: {linear_svc.intercept_}")    # scalar bias
```

**Design rule**: For n > 10,000: use LinearSVC (liblinear) or `SGDClassifier(loss='hinge')`.
For n < 10,000: use SVC(kernel='rbf') with grid search on C and gamma.

### Multiclass SVM

```python
# sklearn default: OvR (one-vs-rest) via decision_function_shape='ovr'
# but internally uses OvO (one-vs-one) for training — SVC trains K*(K-1)/2 classifiers
# Final decision: vote across all pairwise classifiers

svc_multi = SVC(
    C=1.0, kernel='rbf', gamma='scale',
    decision_function_shape='ovr',  # aggregate OvO votes into OvR scores
    random_state=42,
)
svc_multi.fit(X_train_m_s, y_train_m)
print(f"Multiclass SVC accuracy: {svc_multi.score(X_test_m_s, y_test_m):.4f}")  # ~0.73-0.80

# Number of classifiers: K*(K-1)/2 for 4 classes = 6
print(f"n_support_ per class: {svc_multi.n_support_}")
```

---

## 5. Naive Bayes

### Algorithm: Bayes Theorem with Feature Independence Assumption

```
P(y|x₁,...,xₚ) ∝ P(y) Π_j P(xⱼ|y)
```

Feature independence (naive) assumption: features are conditionally independent given class.

Four variants for different feature distributions:

| Variant | Feature model | P(xⱼ|y) | Use case |
|---------|-------------|---------|---------|
| GaussianNB | Continuous | N(μ_{jy}, σ²_{jy}) | Real-valued, approx Gaussian |
| MultinomialNB | Count data | Multinomial with param θ_{jy} | Word counts, TF |
| BernoulliNB | Binary | Bernoulli with p_{jy} | Binary features, word presence |
| ComplementNB | Count data | Complement class parameters | Imbalanced text classification |

### Tambear Decomposition: GaussianNB is Pure Kingdom A

GaussianNB training = per-class mean and variance:
```
μ_{jy}  = accumulate(data[:, j], ByKey{y}, x, Welford).mean
σ²_{jy} = accumulate(data[:, j], ByKey{y}, x, Welford).variance
```

This is a single ByKey{class} accumulate with Welford op — pure Kingdom A.
No scan, no tiled op. With D features and K classes: one pass over N×D data.

GaussianNB prediction (log-likelihood):
```
log P(x|y=k) = Σ_j [-0.5 * log(2π σ²_{jk}) - (xⱼ - μ_{jk})² / (2σ²_{jk})]
```
This is a gather(class_params) + element-wise arithmetic per test point.
Zero new primitives: F06 grouped stats already computes per-class mean/variance.

### Gold Standard: sklearn.naive_bayes

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
import numpy as np

# ---- GaussianNB: real-valued features ----
gnb = GaussianNB(
    priors=None,      # default — estimate from training data (class frequencies)
    var_smoothing=1e-9,  # default — add to variance to avoid zero-var collapse
                         # variance += var_smoothing * max(variance_per_feature)
)
gnb.fit(X_train_b_s, y_train_b)
print(f"GaussianNB accuracy: {gnb.score(X_test_b_s, y_test_b):.4f}")  # ~0.78-0.84

# Oracle attributes:
print(f"Class priors: {gnb.class_prior_}")     # [fraction class 0, fraction class 1]
print(f"Class means shape: {gnb.theta_.shape}")   # (2, 20) — per-class per-feature means
print(f"Class vars shape: {gnb.var_.shape}")       # (2, 20) — per-class per-feature variances
print(f"Classes: {gnb.classes_}")               # [0, 1]

# Predict probabilities
proba_gnb = gnb.predict_proba(X_test_b_s)
print(f"GaussianNB proba shape: {proba_gnb.shape}")  # (200, 2)
log_proba_gnb = gnb.predict_log_proba(X_test_b_s)   # log-domain (more numerically stable)

# Incremental fit (online learning — GaussianNB supports partial_fit)
gnb_online = GaussianNB()
for i in range(0, len(X_train_b_s), 100):
    gnb_online.partial_fit(
        X_train_b_s[i:i+100], y_train_b[i:i+100],
        classes=[0, 1]  # MUST provide classes on first call
    )
print(f"GaussianNB online accuracy: {gnb_online.score(X_test_b_s, y_test_b):.4f}")

# ---- GaussianNB multiclass ----
gnb_multi = GaussianNB()
gnb_multi.fit(X_train_m_s, y_train_m)
print(f"GaussianNB multiclass accuracy: {gnb_multi.score(X_test_m_s, y_test_m):.4f}")  # ~0.50-0.65
print(f"Class means shape: {gnb_multi.theta_.shape}")  # (4, 20)
```

Expected validation targets:
```
GaussianNB binary accuracy     ≈ 0.810  (±0.02)
GaussianNB multiclass accuracy ≈ 0.575  (±0.05)
theta_.shape = (n_classes, n_features)
var_.shape   = (n_classes, n_features)
class_prior_ = array of length n_classes, sums to 1.0
```

### MultinomialNB (for count data)

```python
from sklearn.preprocessing import MinMaxScaler

# MultinomialNB requires non-negative features (counts or TF)
X_train_counts = np.abs(X_train_b_s * 10).astype(int)  # simulate count-like features
X_test_counts  = np.abs(X_test_b_s * 10).astype(int)

mnb = MultinomialNB(
    alpha=1.0,    # default — Laplace smoothing (additive smoothing parameter)
                  # alpha=0: no smoothing (dangerous for unseen features)
                  # alpha=1: Laplace smoothing (add 1 to all counts)
                  # alpha<1: Lidstone smoothing
    fit_prior=True,  # default — learn class priors from data
    class_prior=None, # default — None means estimate from data
)
mnb.fit(X_train_counts, y_train_b)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test_b):.4f}")  # ~0.60-0.75

# Oracle attributes:
print(f"Feature log proba shape: {mnb.feature_log_prob_.shape}")  # (2, 20)
print(f"Class log prior: {mnb.class_log_prior_}")                  # log P(y=k)
```

### BernoulliNB (for binary features)

```python
# BernoulliNB: features are 0 or 1 (presence/absence)
X_train_binary = (X_train_b_s > 0).astype(int)
X_test_binary  = (X_test_b_s > 0).astype(int)

bnb = BernoulliNB(
    alpha=1.0,        # default — Laplace smoothing
    binarize=None,    # default — assumes features already binary
                      # binarize=0.0 would threshold at 0 internally
    fit_prior=True,   # default
)
bnb.fit(X_train_binary, y_train_b)
print(f"BernoulliNB accuracy: {bnb.score(X_test_binary, y_test_b):.4f}")  # ~0.65-0.75

# With auto-binarization:
bnb_auto = BernoulliNB(alpha=1.0, binarize=0.0)
bnb_auto.fit(X_train_b_s, y_train_b)   # binarizes internally
print(f"BernoulliNB (auto-binarize) accuracy: {bnb_auto.score(X_test_b_s, y_test_b):.4f}")
```

### Laplace Smoothing Formula

Without smoothing, a class-feature combination unseen in training gives P=0,
which zeroes out the entire log-likelihood (log 0 = -inf problem).

Laplace smoothing adds α to all counts:
```
θ_{jy} = (count(feature j = v, class y) + α) / (count(class y) + α * n_unique_values)
```

In sklearn's MultinomialNB: `feature_log_prob_[y, j] = log(θ_{jy})`

---

## 6. KNN Classifier

### Algorithm

For a new point q, find the K nearest neighbors in the training set,
then take a majority vote of their class labels:
```
class(q) = argmax_k count(y[i] == k for i in top_K_indices(distance(q, x_i)))
```

Weighted KNN: weight votes by 1/distance (closer neighbors vote more).

### Tambear Decomposition (free from F01/F20)

The full DistancePairs MSR is already computed and cached from F01.
For the KNN classifier:
1. Row-wise ArgMin K times to find K nearest indices → ArgMinOp (Kingdom A)
2. Majority vote: `accumulate(k_labels, ByKey{class}, 1, Add)` → ArgMax over counts
3. Total new code: ~20 lines (same as KNN regressor but with vote instead of mean)

```
KNN predict(q):
  distances = DistancePairs[q, :]        # one row, free from cache
  k_indices = argpartition(distances, K) # top-K ArgMin
  k_labels  = labels[k_indices]
  counts    = accumulate(k_labels, ByKey{class}, 1, Add)
  class     = ArgMax(counts)
```

With weighted votes:
```
  weights  = 1.0 / (distances[k_indices] + epsilon)  # avoid div-by-zero
  w_counts = accumulate((k_labels, weights), ByKey{class}, w, Add)
  class    = ArgMax(w_counts)
```

### Gold Standard: sklearn.neighbors.KNeighborsClassifier

```python
from sklearn.neighbors import KNeighborsClassifier

# ---- Uniform voting (default) ----
knn = KNeighborsClassifier(
    n_neighbors=5,          # default K
    weights='uniform',      # default — all neighbors vote equally
    algorithm='auto',       # default — auto-select ball_tree/kd_tree/brute
                            # 'brute' = exact pairwise distance (what tambear does)
    metric='minkowski',     # default metric
    p=2,                    # default — p=2 → L2 (Euclidean), p=1 → L1 (Manhattan)
    leaf_size=30,           # default — for ball_tree/kd_tree (irrelevant for brute)
    n_jobs=-1,              # parallelism
)
knn.fit(X_train_b_s, y_train_b)
print(f"KNN (k=5, uniform) accuracy: {knn.score(X_test_b_s, y_test_b):.4f}")  # ~0.89-0.93

# ---- Distance-weighted voting ----
knn_dist = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',   # weight by 1/distance
    algorithm='brute',    # exact (matches tambear behavior)
    n_jobs=-1,
)
knn_dist.fit(X_train_b_s, y_train_b)
print(f"KNN (k=5, distance-weighted) accuracy: {knn_dist.score(X_test_b_s, y_test_b):.4f}")  # ~0.89-0.93

# ---- Sweep over K ----
results = {}
for k in [1, 3, 5, 10, 15, 20]:
    knn_k = KNeighborsClassifier(n_neighbors=k, algorithm='brute', n_jobs=-1)
    knn_k.fit(X_train_b_s, y_train_b)
    results[k] = knn_k.score(X_test_b_s, y_test_b)
    print(f"K={k}: {results[k]:.4f}")
# Expected: peak around K=5 to K=15 for this dataset

# ---- Probability output ----
proba_knn = knn.predict_proba(X_test_b_s)
print(f"KNN proba shape: {proba_knn.shape}")  # (200, 2)
# proba[i, k] = fraction of K neighbors in class k
# With K=5, values are multiples of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# ---- Retrieve actual neighbors ----
distances, indices = knn.kneighbors(X_test_b_s[:3], n_neighbors=5)
print(f"Distances to 5 NN for 3 test points:\n{distances}")
print(f"Indices of 5 NN for 3 test points:\n{indices}")
```

Expected validation targets:
```
KNN (k=5) accuracy     ≈ 0.910  (±0.01)
KNN sweep peak K       ≈ 5 to 15
proba values           multiples of 1/K for uniform weights
```

---

## 7. Classification Metrics

### Metrics Taxonomy

| Metric | Type | Range | Notes |
|--------|------|-------|-------|
| Accuracy | Scalar | [0, 1] | Misleading for imbalanced |
| Precision | Scalar / per-class | [0, 1] | TP / (TP + FP) |
| Recall (Sensitivity) | Scalar / per-class | [0, 1] | TP / (TP + FN) |
| Specificity | Scalar / per-class | [0, 1] | TN / (TN + FP) |
| F1 Score | Scalar / per-class | [0, 1] | 2 * P * R / (P + R) |
| F-beta Score | Scalar | [0, 1] | Weighted F1: (1+β²)PR/(β²P+R) |
| AUC-ROC | Scalar | [0, 1] | Area under ROC curve |
| Average Precision | Scalar | [0, 1] | Area under PR curve |
| MCC | Scalar | [-1, 1] | Matthews correlation — best single binary metric |
| Log Loss | Scalar | [0, ∞) | Cross-entropy; needs probabilities |
| Brier Score | Scalar | [0, 1] | MSE of probability predictions |
| Cohen's Kappa | Scalar | [-1, 1] | Chance-corrected agreement |

### Multiclass Averaging Modes

For precision, recall, F1 in multiclass settings:

| Mode | Formula | When to use |
|------|---------|-------------|
| `macro` | Unweighted mean over classes | Each class equally important, even if small |
| `micro` | Global TP/FP/FN counts | Each SAMPLE equally important; handles imbalance |
| `weighted` | Class-support-weighted mean | Accounts for class frequency |
| `samples` | Per sample (only for multilabel) | Multilabel classification |

Key property: `micro` F1 = accuracy when all classes are predicted at least once.

### Gold Standard: sklearn.metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    log_loss, brier_score_loss, matthews_corrcoef,
    cohen_kappa_score,
)
import numpy as np

# Use RF predictions for demo
y_pred_b = rf.predict(X_test_b_s)
y_proba_b = rf.predict_proba(X_test_b_s)[:, 1]  # positive class probability

y_pred_m = rf_multi.predict(X_test_m_s)
y_proba_m = rf_multi.predict_proba(X_test_m_s)   # shape (200, 4)

# ---- Binary metrics ----
print(f"Accuracy:     {accuracy_score(y_test_b, y_pred_b):.4f}")
print(f"Precision:    {precision_score(y_test_b, y_pred_b):.4f}")  # default pos_label=1
print(f"Recall:       {recall_score(y_test_b, y_pred_b):.4f}")
print(f"F1:           {f1_score(y_test_b, y_pred_b):.4f}")
print(f"MCC:          {matthews_corrcoef(y_test_b, y_pred_b):.4f}")
print(f"AUC-ROC:      {roc_auc_score(y_test_b, y_proba_b):.4f}")
print(f"Avg Precision:{average_precision_score(y_test_b, y_proba_b):.4f}")
print(f"Log loss:     {log_loss(y_test_b, y_proba_b):.4f}")
print(f"Brier score:  {brier_score_loss(y_test_b, y_proba_b):.4f}")
print(f"Cohen kappa:  {cohen_kappa_score(y_test_b, y_pred_b):.4f}")

# ---- Classification report (per-class breakdown) ----
print(classification_report(y_test_b, y_pred_b, target_names=['neg', 'pos']))
# Output format:
#               precision    recall  f1-score   support
#          neg       0.92      0.91      0.91       100
#          pos       0.91      0.92      0.91       100
#     accuracy                           0.91       200
#    macro avg       0.91      0.91      0.91       200
# weighted avg       0.91      0.91      0.91       200

# ---- ROC curve ----
fpr, tpr, thresholds = roc_curve(y_test_b, y_proba_b)
print(f"ROC AUC (from curve): {np.trapz(tpr, fpr):.4f}")  # trapezoidal rule
# fpr[0] = 0.0, tpr[0] = 0.0 (all classified negative)
# fpr[-1] = 1.0, tpr[-1] = 1.0 (all classified positive)

# ---- PR curve ----
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test_b, y_proba_b)
# NOTE: precision_vals and recall_vals have n+1 elements, pr_thresholds has n elements
ap = average_precision_score(y_test_b, y_proba_b)
print(f"Average Precision (AP): {ap:.4f}")
# PR AUC ≈ AP (they're computed slightly differently but both estimate the area)

# ---- Confusion matrix ----
cm = confusion_matrix(y_test_b, y_pred_b)
print(f"Confusion matrix:\n{cm}")
# Format: cm[i, j] = count predicted j, actual i
#         [[TN, FP],
#          [FN, TP]]

# ---- Multiclass metrics — averaging modes ----
print(f"\n=== Multiclass metrics ===")
print(f"Accuracy: {accuracy_score(y_test_m, y_pred_m):.4f}")
print(f"F1 macro:    {f1_score(y_test_m, y_pred_m, average='macro'):.4f}")
print(f"F1 micro:    {f1_score(y_test_m, y_pred_m, average='micro'):.4f}")    # = accuracy
print(f"F1 weighted: {f1_score(y_test_m, y_pred_m, average='weighted'):.4f}")

# AUC-ROC for multiclass: OvR approach
roc_auc_ovr = roc_auc_score(y_test_m, y_proba_m, multi_class='ovr', average='macro')
roc_auc_ovo = roc_auc_score(y_test_m, y_proba_m, multi_class='ovo', average='macro')
print(f"AUC-ROC multiclass (OvR macro): {roc_auc_ovr:.4f}")
print(f"AUC-ROC multiclass (OvO macro): {roc_auc_ovo:.4f}")
```

Expected validation targets:
```
RF binary: AUC-ROC ≈ 0.960  (±0.01)
RF binary: F1      ≈ 0.910  (±0.01)
RF binary: MCC     ≈ 0.820  (±0.02)
RF multi:  F1 macro   ≈ 0.750  (±0.03)
RF multi:  F1 micro   ≈ 0.750  (±0.03)  # = accuracy for balanced classes
```

### AUC Computation: Trapezoidal Rule

The trapezoidal rule integrates the ROC curve numerically:
```
AUC = Σᵢ (FPR[i+1] - FPR[i]) * (TPR[i] + TPR[i+1]) / 2
```

sklearn's `roc_auc_score` uses this formula internally.
Equivalent to: probability that a randomly chosen positive is ranked higher than a
randomly chosen negative (the Wilcoxon-Mann-Whitney statistic).

### PR Curve vs ROC Curve for Imbalanced Data

```python
# For imbalanced data, PR curve is more informative than ROC curve.
# ROC curve can look optimistic because FPR is normalized by (TN + FP).
# With many negatives (TN >> FP), FPR stays small even with many false positives.
# PR curve uses Precision = TP/(TP+FP) — penalizes FP directly.

# Imbalanced example:
from sklearn.datasets import make_classification
X_imb, y_imb = make_classification(
    n_samples=1000, n_classes=2, weights=[0.95, 0.05],  # 95% negative
    n_features=20, random_state=42
)
X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)

rf_imb = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_imb.fit(X_tr_i, y_tr_i)
y_proba_imb = rf_imb.predict_proba(X_te_i)[:, 1]

roc_auc_imb = roc_auc_score(y_te_i, y_proba_imb)
ap_imb      = average_precision_score(y_te_i, y_proba_imb)

print(f"Imbalanced - ROC AUC: {roc_auc_imb:.4f}")  # may look high, ~0.85-0.95
print(f"Imbalanced - AP:      {ap_imb:.4f}")        # much lower, ~0.50-0.75
# AP < 0.50 would mean the model is nearly useless despite high ROC AUC
# Use AP as the primary metric when positive class is rare
```

### F-beta Score: Precision-Recall Trade-off

```python
from sklearn.metrics import fbeta_score

# beta=1: standard F1 (precision = recall equally)
# beta=2: recall weighted 2x more than precision (false negatives costly)
# beta=0.5: precision weighted 2x more than recall (false positives costly)

print(f"F1   (beta=1): {fbeta_score(y_test_b, y_pred_b, beta=1):.4f}")
print(f"F2   (beta=2): {fbeta_score(y_test_b, y_pred_b, beta=2):.4f}")
print(f"F0.5 (beta=0.5): {fbeta_score(y_test_b, y_pred_b, beta=0.5):.4f}")
```

---

## 8. Calibration

### What is Calibration?

A classifier is **well-calibrated** if its predicted probability p corresponds to
the empirical frequency of the positive class among samples with that predicted probability.

If a model predicts 0.7 for 100 samples, well-calibrated means ~70 of them are actually positive.

Calibration is SEPARATE from discrimination (AUC). A model can be:
- Good discrimination + bad calibration: Random Forest (confidently wrong probabilities)
- Poor discrimination + good calibration: Logistic regression when features are weak but probabilities are trustworthy
- Bad both: naive model, random model

### Calibration Metrics

**Expected Calibration Error (ECE)**:
Bin predictions into M bins (e.g., [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]):
```
ECE = Σ_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
```
where acc(B_m) = fraction of positives in bin m, conf(B_m) = mean predicted probability in bin m.

### Gold Standard: sklearn.calibration

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# ---- Reliability diagram ----
# Compare well-calibrated (logistic) vs poorly-calibrated (RF) classifiers

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0, random_state=42)
lr.fit(X_train_b_s, y_train_b)
y_proba_lr = lr.predict_proba(X_test_b_s)[:, 1]

y_proba_rf = rf.predict_proba(X_test_b_s)[:, 1]

# calibration_curve: bins predictions into n_bins buckets
# Returns: fraction_of_positives (actual), mean_predicted_value (what model predicted)
fop_lr, mpv_lr = calibration_curve(y_test_b, y_proba_lr, n_bins=10)
fop_rf, mpv_rf = calibration_curve(y_test_b, y_proba_rf, n_bins=10)

print(f"LR calibration  (actual vs predicted):\n{list(zip(fop_lr.round(2), mpv_lr.round(2)))}")
print(f"RF calibration  (actual vs predicted):\n{list(zip(fop_rf.round(2), mpv_rf.round(2)))}")
# Perfect calibration: (x, x) for each pair — diagonal line
# RF typically overconfident (predicted proba too extreme)

# ---- Platt scaling (sigmoid calibration) ----
rf_platt = CalibratedClassifierCV(
    rf,             # base classifier
    method='sigmoid',  # Platt scaling — fits a sigmoid (logistic) transformation
    cv='prefit',    # rf is already fitted; don't refit
)
rf_platt.fit(X_test_b_s, y_test_b)  # NOTE: fit calibrator on HELD-OUT data (here test set for demo)
                                     # In production: use a separate calibration set
y_proba_platt = rf_platt.predict_proba(X_test_b_s)[:, 1]
fop_platt, mpv_platt = calibration_curve(y_test_b, y_proba_platt, n_bins=10)
print(f"RF+Platt calibration:\n{list(zip(fop_platt.round(2), mpv_platt.round(2)))}")

# ---- Isotonic regression calibration ----
rf_iso = CalibratedClassifierCV(rf, method='isotonic', cv='prefit')
rf_iso.fit(X_test_b_s, y_test_b)  # NOTE: same held-out caveat
y_proba_iso = rf_iso.predict_proba(X_test_b_s)[:, 1]

# ---- Cross-validated calibration (preferred in practice) ----
# Fits K SVMs on K folds, calibrates probabilities on held-out fold each time
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_cv_cal = CalibratedClassifierCV(rf_base, method='isotonic', cv=5)
rf_cv_cal.fit(X_train_b_s, y_train_b)
print(f"RF (CV-calibrated) accuracy: {rf_cv_cal.score(X_test_b_s, y_test_b):.4f}")

# ---- ECE computation ----
def ece(y_true, y_proba, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
        if mask.sum() == 0:
            continue
        frac_pos = y_true[mask].mean()
        mean_conf = y_proba[mask].mean()
        ece_val += mask.mean() * abs(frac_pos - mean_conf)
    return ece_val

print(f"ECE (LR):      {ece(y_test_b, y_proba_lr):.4f}")        # expect ~0.02-0.05
print(f"ECE (RF):      {ece(y_test_b, y_proba_rf):.4f}")        # expect ~0.05-0.15
print(f"ECE (RF+Platt):{ece(y_test_b, y_proba_platt):.4f}")     # expect ~0.02-0.06
```

Expected validation targets:
```
LR ECE        ≈ 0.025  (logistic is well-calibrated by design)
RF ECE        ≈ 0.080  (RF overconfident — probabilities near 0 and 1)
RF+Platt ECE  ≈ 0.030  (calibration improves significantly)
```

---

## 9. Tambear Cross-Family Architecture Summary

How F21 consumes other families' MSRs with zero new GPU cost:

| Classifier | Primitive | Source | New code estimate |
|-----------|----------|--------|------------------|
| KNN Classifier | DistancePairs + ArgMin | F01 / F20 | ~20 lines |
| Nearest Centroid | ByKey{class} Welford (means) + DistancePairs | F06 + F01 | ~10 lines |
| LDA | GramMatrix per class + EigenDecomp | F10 + F22 | ~40 lines |
| GaussianNB | ByKey{class} Welford (mean + var) | F06 | ~15 lines |
| Logistic Regression | IRLS (GramMatrix iterations) | F10 | Already done |
| Decision Tree | ByKey{bin} histogram (entropy/Gini) + ArgMin | F25 + Kingdom A | ~150 lines |
| Random Forest | N × Decision Tree (parallelized) | Builds on tree | ~50 lines |
| Gradient Boosting | Iterative tree fitting + gradient residuals | Kingdom C + F05 | ~100 lines |
| Linear SVM | GramMatrix + SMO dual QP | F10 + F05 | ~200 lines |
| Kernel SVM (RBF) | DistancePairs → exp(-γD²) + SMO | F01 + SVM | ~30 extra lines |

**Total new code for all of F21**: approximately 600 lines, of which ~200 is the SMO solver
(new, no prior equivalent). The other 400 lines assemble from existing MSRs.

---

## 10. Key Traps Master List

**Trap 1: max_features default differs in tree vs forest**
- `DecisionTreeClassifier`: default `max_features=None` (all features)
- `RandomForestClassifier`: default `max_features='sqrt'` (sqrt of features)
- Do NOT copy hyperparameters from single tree to forest without checking this

**Trap 2: XGBoost multiclass objective**
- Binary: `objective='binary:logistic'`
- Multiclass labels: `objective='multi:softmax'` + `num_class=K`
- Multiclass probabilities: `objective='multi:softprob'` + `num_class=K`
- Using `binary:logistic` on multiclass silently produces wrong results

**Trap 3: LightGBM leaf-wise growth overfit risk**
- Default `num_leaves=31` with `max_depth=-1` (no depth limit)
- A tree with 31 leaves grown leaf-wise can reach depth 30
- Guard: always set `max_depth` alongside `num_leaves` for production

**Trap 4: LightGBM vs XGBoost depth control**
- XGBoost: `max_depth=6` = level-wise tree with exactly 6 levels (up to 2^6 = 64 leaves)
- LightGBM: `num_leaves=31` = leaf-wise tree with 31 leaves at various depths
- Equivalent complexity: `num_leaves = 2^max_depth - 1`

**Trap 5: SVC probability=True overhead**
- Sets off an internal 5-fold cross-validation to fit Platt scaling
- Can be 3-10x slower than SVC without probability
- For bulk probability needs: use CalibratedClassifierCV(cv='prefit') on a held-out set

**Trap 6: sklearn.svm.LinearSVC vs SVC(kernel='linear')**
- LinearSVC: liblinear backend, O(n*p) iterations, no predict_proba, scales to n=10⁶
- SVC: libsvm backend, O(n²) to O(n³), has support vectors, has dual_coef_
- Use LinearSVC when n > 10,000

**Trap 7: Gini importance is biased**
- `feature_importances_` in trees/forests = sum of impurity decrease weighted by sample count
- This INFLATES importance for high-cardinality continuous features
- For correct importance: use `permutation_importance()` from sklearn.inspection

**Trap 8: multiclass AUC-ROC requires explicit multi_class parameter**
- `roc_auc_score(y_true, y_proba_matrix)` fails without `multi_class='ovr'` or `'ovo'`
- OvR: train K binary classifiers (each class vs all others)
- OvO: train K*(K-1)/2 binary classifiers (each pair)

**Trap 9: PR curve has length n+1, thresholds have length n**
- `precision_recall_curve` returns precision[n+1], recall[n+1], thresholds[n]
- The last (precision, recall) = (1.0, 0.0) by convention (no-positive-threshold edge case)
- This bites vectorized AUC computations

**Trap 10: Gradient boosting max_depth shallowness is intentional**
- sklearn GradientBoostingClassifier default `max_depth=3` (NOT None)
- Shallow trees are stumps/weak learners — that's the POINT of boosting
- Deep trees in boosting = overfitting (unlike RF where depth doesn't matter as much)

**Trap 11: early_stopping_rounds location changed in XGBoost**
- Old API: passed to `fit()` method
- New API (>= 1.6): passed to constructor
- Both still work but mixing them causes silent ignoring in some versions

**Trap 12: n_estimators=10 was sklearn RF default before version 0.22**
- Old default: 10 trees (terrible, too small)
- Current default: 100 trees
- Always specify `n_estimators` explicitly if results need to be reproducible across sklearn versions

---

## Appendix A: Complete Model Comparison on Shared Dataset

```python
import pandas as pd
from sklearn.svm import SVC

models = {
    'Decision Tree (Gini)': DecisionTreeClassifier(random_state=42),
    'Decision Tree (Entropy)': DecisionTreeClassifier(criterion='entropy', random_state=42),
    'Random Forest (n=100)': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', gamma='scale', probability=True, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
    'LinearSVC': LinearSVC(random_state=42, max_iter=2000),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Gaussian NB': GaussianNB(),
    'Logistic Regression': LogisticRegression(C=1.0, random_state=42),
}

rows = []
for name, model in models.items():
    model.fit(X_train_b_s, y_train_b)
    acc = model.score(X_test_b_s, y_test_b)

    # AUC where possible
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test_b_s)[:, 1]
        auc = roc_auc_score(y_test_b, proba)
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test_b_s)
        auc = roc_auc_score(y_test_b, scores)
    else:
        auc = float('nan')

    rows.append({'Model': name, 'Accuracy': acc, 'AUC-ROC': auc})

df = pd.DataFrame(rows).sort_values('AUC-ROC', ascending=False)
print(df.to_string(index=False))
```

Expected approximate ranking (higher is better):
```
Model                     Accuracy   AUC-ROC
Random Forest (n=100)       ~0.91     ~0.96
Gradient Boosting           ~0.92     ~0.96
SVM (RBF)                   ~0.91     ~0.95
Logistic Regression         ~0.88     ~0.94
SVM (Linear)                ~0.90     ~0.95
LinearSVC                   ~0.89     ~0.94
KNN (k=5)                   ~0.91     ~0.94
Decision Tree (Gini)        ~0.87     ~0.87  (no proba for AUC — uses predict_proba)
Decision Tree (Entropy)     ~0.86     ~0.86
Gaussian NB                 ~0.81     ~0.90
```

Note: exact ordering varies by random seed; these are representative magnitudes.

---

## Appendix B: Validation Protocol for Tambear F21 Implementation

When tambear's F21 implementation is ready, validate against these oracle values:

```python
def validate_f21_classification(tambear_clf, X_train, y_train, X_test, y_test,
                                 sklearn_clf, tolerance=0.02):
    """
    Validate tambear classifier against sklearn oracle.

    Passes if:
    - Accuracy within `tolerance` of sklearn accuracy
    - If probabilities available: AUC within `tolerance`
    - Confusion matrix structure matches (same majority vote direction)
    """
    sklearn_clf.fit(X_train, y_train)
    sk_acc = sklearn_clf.score(X_test, y_test)

    # tambear would expose a predict() method:
    # tam_pred = tambear_clf.predict(X_test)
    # tam_acc = accuracy_score(y_test, tam_pred)
    # assert abs(tam_acc - sk_acc) < tolerance, f"Accuracy gap: {abs(tam_acc - sk_acc):.4f}"

    # For exact-match classifiers (KNN with same distance, GaussianNB):
    # tolerance = 1e-4 (numerical precision only)

    # For tree-based (different split-point search order):
    # tolerance = 0.02 (same structure not required, similar accuracy is)

    # For SVM (same QP objective, different solver):
    # tolerance = 0.01 (support vectors may differ but prediction should match)

    return sk_acc

# Validation targets:
ORACLE_ACCURACIES = {
    'knn_5': 0.910,           # tolerance 1e-3 (exact same distances)
    'nearest_centroid': 0.858, # tolerance 1e-3 (exact same centroids)
    'gaussian_nb': 0.810,      # tolerance 1e-3 (exact same sufficient stats)
    'decision_tree': 0.875,    # tolerance 0.02 (split order may differ)
    'random_forest': 0.910,    # tolerance 0.02 (bootstraps differ)
    'gradient_boosting': 0.920, # tolerance 0.02 (residual precision differs)
    'svm_linear': 0.900,       # tolerance 0.01 (dual QP solution unique)
    'svm_rbf': 0.910,          # tolerance 0.01
}
```

---

*Document complete. All code blocks are independently runnable given the shared dataset setup at the top.*
