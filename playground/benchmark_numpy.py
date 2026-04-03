"""
Tambear Benchmark: NumPy/SciPy competitor
Same 6-step pipeline, same data, fair comparison.
"""
import time
import numpy as np
from scipy import stats

N = 100_000
D = 50
K_PCA = 5
K_CLUSTERS = 3

print("=" * 60)
print(f"  NUMPY BENCHMARK")
print(f"  N={N}, D={D}, k_pca={K_PCA}, k_clusters={K_CLUSTERS}")
print(f"  Data size: {N*D*8/1e6:.1f} MB")
print("=" * 60)

# Generate same data structure (3 planted clusters)
t0 = time.perf_counter()
rng = np.random.default_rng(42)
centers = rng.standard_normal((3, D)) * 5
n_per = N // 3
data = np.vstack([
    rng.standard_normal((n_per, D)) + centers[i]
    for i in range(3)
])
true_labels = np.repeat([0, 1, 2], n_per)
t_gen = time.perf_counter() - t0
print(f"\n  Data generation: {t_gen:.3f}s")

# Step 1: Descriptive statistics
t1 = time.perf_counter()
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)
skews = stats.skew(data, axis=0)
kurts = stats.kurtosis(data, axis=0)
t_desc = time.perf_counter() - t1
print(f"  Step 1 (describe):     {t_desc:.3f}s  [4 separate passes]")
print(f"    mean[0]={means[0]:.4f}, std[0]={stds[0]:.4f}, skew[0]={skews[0]:.4f}, kurt[0]={kurts[0]:.4f}")

# Step 2: Correlation matrix
t2 = time.perf_counter()
corr = np.corrcoef(data, rowvar=False)
t_corr = time.perf_counter() - t2
print(f"  Step 2 (correlation):  {t_corr:.3f}s")
print(f"    corr[0,1]={corr[0,1]:.4f}, corr[0,2]={corr[0,2]:.4f}")

# Step 3: PCA (via eigendecomposition of correlation matrix)
t3 = time.perf_counter()
eigenvalues, eigenvectors = np.linalg.eigh(corr)
# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx][:K_PCA]
eigenvectors = eigenvectors[:, idx][:, :K_PCA]
t_pca = time.perf_counter() - t3
var_explained = eigenvalues.sum() / D
print(f"  Step 3 (PCA):          {t_pca:.3f}s")
print(f"    eigenvalues: {[f'{e:.2f}' for e in eigenvalues]}")
print(f"    variance explained: {var_explained*100:.1f}%")

# Projection
t_proj_start = time.perf_counter()
data_std = (data - means) / stds
projected = data_std @ eigenvectors
t_proj = time.perf_counter() - t_proj_start
print(f"    projection:          {t_proj:.3f}s")

# Step 4: KDE on first PC
t4 = time.perf_counter()
pc1 = projected[:, 0]
kde = stats.gaussian_kde(pc1)
grid = np.linspace(pc1.min() - 0.5, pc1.max() + 0.5, 200)
density = kde(grid)
t_kde = time.perf_counter() - t4
print(f"  Step 4 (KDE):          {t_kde:.3f}s")
print(f"    peak density: {density.max():.4f}")

# Step 5: K-means (using sklearn if available, else manual)
try:
    from sklearn.cluster import KMeans
    t5 = time.perf_counter()
    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=1, max_iter=100)
    assignments = km.fit_predict(projected)
    t_kmeans = time.perf_counter() - t5
except ImportError:
    # Manual k-means
    t5 = time.perf_counter()
    centers_km = projected[:K_CLUSTERS].copy()
    assignments = np.zeros(N, dtype=int)
    for _ in range(100):
        dists = np.array([np.sum((projected - c)**2, axis=1) for c in centers_km])
        new_assignments = np.argmin(dists, axis=0)
        if np.all(new_assignments == assignments):
            break
        assignments = new_assignments
        for c in range(K_CLUSTERS):
            mask = assignments == c
            if mask.any():
                centers_km[c] = projected[mask].mean(axis=0)
    t_kmeans = time.perf_counter() - t5

# Rand index (subsample for speed)
sub = min(N, 5000)
agree = 0
total = 0
for i in range(sub):
    for j in range(i+1, sub):
        same_pred = assignments[i] == assignments[j]
        same_true = true_labels[i] == true_labels[j]
        if same_pred == same_true:
            agree += 1
        total += 1
rand_index = agree / total
print(f"  Step 5 (k-means):      {t_kmeans:.3f}s")
print(f"    Rand index vs truth: {rand_index:.4f}")

# Step 6: ANOVA
t6 = time.perf_counter()
f_stats = []
for col in range(D):
    groups = [data[assignments == g, col] for g in range(K_CLUSTERS)]
    f, p = stats.f_oneway(*groups)
    f_stats.append(f)
f_stats = np.array(f_stats)
t_anova = time.perf_counter() - t6
significant = np.sum(f_stats > 3.0)
print(f"  Step 6 (ANOVA):        {t_anova:.3f}s")
print(f"    significant (F>3): {significant}/{D}")
print(f"    max F: {f_stats.max():.1f}, min F: {f_stats.min():.1f}")

# Summary
t_total = time.perf_counter() - t0
t_compute = t_desc + t_corr + t_pca + t_proj + t_kde + t_kmeans + t_anova

print(f"\n=== SUMMARY ===")
print(f"  Data:    {N} × {D} = {N*D*8/1e6:.1f} MB")
print(f"  Total:   {t_total:.3f}s (including generation)")
print(f"  Compute: {t_compute:.3f}s (pipeline only)")
print(f"\n  Step breakdown:")
print(f"    describe:    {t_desc:>7.3f}s  {t_desc/t_compute*100:>5.1f}%")
print(f"    correlation: {t_corr:>7.3f}s  {t_corr/t_compute*100:>5.1f}%")
print(f"    PCA:         {t_pca:>7.3f}s  {t_pca/t_compute*100:>5.1f}%")
print(f"    projection:  {t_proj:>7.3f}s  {t_proj/t_compute*100:>5.1f}%")
print(f"    KDE:         {t_kde:>7.3f}s  {t_kde/t_compute*100:>5.1f}%")
print(f"    k-means:     {t_kmeans:>7.3f}s  {t_kmeans/t_compute*100:>5.1f}%")
print(f"    ANOVA:       {t_anova:>7.3f}s  {t_anova/t_compute*100:>5.1f}%")
print(f"\n  Passes over data: 8+ (each np.mean, np.std, etc. is a separate pass)")
