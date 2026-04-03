# The Tambear Benchmark: One Pipeline, Every Platform, Honest Numbers

## The Pipeline

A complete exploratory data analysis that every data scientist recognizes:

```
Step 1: Descriptive statistics (mean, std, skew, kurt per column)
Step 2: Correlation matrix (all pairwise)
Step 3: PCA (eigendecomposition, top k components)
Step 4: KDE (density estimation of each PC)
Step 5: Clustering (k-means on PCA-reduced data)
Step 6: Hypothesis test (ANOVA across clusters, per original column)
```

This is what you'd write as:
```tbs
data.describe().correlation().pca(k=5).kde().kmeans(k=3).anova()
```

Six steps. One chain. Every step has a well-defined correct output.

## Why This Pipeline

1. **Steps 1-2 share accumulators.** mean/std/skew/kurt all need {Σx, Σx², Σx³, Σx⁴, n}. 
   Correlation needs {Σxᵢxⱼ}. Tambear computes all of this in ONE pass.
   Every competitor computes them separately (multiple passes over data).

2. **Step 3 is the hard part.** PCA on D×D covariance matrix. O(D³) eigendecomp.
   But the covariance matrix was ALREADY computed in step 2 (shared!).
   Competitors: recompute covariance from scratch for PCA.

3. **Steps 4-6 operate on reduced data.** After PCA, data is N×k (k<<D).
   KDE, k-means, ANOVA on reduced data = fast.
   The bottleneck is steps 1-3, which tambear fuses.

4. **Every step has a verifiable correct output.** Mean of N(0,1) = 0±ε.
   Eigenvalues of a known covariance = known values. ANOVA F-stat = known.

5. **Scales in both N and D.** 
   N = 10K, 100K, 1M, 10M, 100M, 1B rows
   D = 10, 50, 100, 500, 1000 columns
   Cross product: 30 scale points

## The Competitors

For EACH competitor: their BEST option, tuned, with tricks.

| Platform | Package | GPU? | Notes |
|----------|---------|------|-------|
| Python/NumPy | numpy + scipy + sklearn | CPU | The baseline everyone knows |
| Python/CuPy | cupy + cuml | GPU | NVIDIA's Python GPU stack |
| R | base R + Rfast | CPU | Optimized R packages |
| Julia | Statistics.jl + MultivariateStats | CPU | JIT-compiled |
| RAPIDS cuML | cuml.PCA, cuml.KMeans | GPU | NVIDIA's best |
| Rust/ndarray | ndarray + linfa | CPU | Same language, different approach |
| MATLAB | built-in | CPU | Commercial reference |
| Tambear CPU | CpuBackend | CPU | Our CPU fallback |
| Tambear CUDA | CudaBackend | GPU | Our primary target |
| Tambear Vulkan | VulkanBackend | GPU | Cross-platform |
| Tambear wgpu | WgpuBackend | GPU | WebGPU/universal |

## What We Measure

For each (platform, N, D) combination:

### Time
- T_launch: time from script start to first computation
- T_step[i]: time for each of the 6 pipeline steps
- T_total: end-to-end wall clock
- T_compile: (tambear only) JIT compilation time
- T_io: data loading / serialization

### Accuracy
- Mean: |computed - true_mean| per column (synthetic data with known mean)
- Std: |computed - true_std| per column
- Eigenvalues: |computed - true_eigenvalues| (synthetic with planted structure)
- Cluster assignments: Rand index vs known ground truth
- ANOVA F-stat: |computed - scipy.reference|

### Resources
- Peak GPU VRAM usage (MB)
- Peak CPU RAM usage (MB)
- CPU utilization % (per core)
- GPU utilization % 
- GPU temperature (°C) over time
- Power draw (W) if available

### Tambear-Specific
- Number of GPU kernel launches
- Number of passes over data
- Shared accumulator count (how many steps fused)
- JIT plan: which steps were GPU-accelerated vs CPU fallback

## The Dataset

Synthetic (reproducible, known ground truth):

```python
# 3 planted clusters, known covariance structure
rng = np.random.default_rng(42)
n_per_cluster = N // 3
centers = rng.standard_normal((3, D)) * 5
cov = make_spd_matrix(D, random_state=42)  # known positive definite
data = np.vstack([
    rng.multivariate_normal(centers[i], cov, n_per_cluster)
    for i in range(3)
])
labels = np.repeat([0, 1, 2], n_per_cluster)
```

Ground truth:
- True means = centers (per cluster)
- True covariance = cov
- True eigenvalues = eig(cov)
- True labels = labels
- True ANOVA: planted difference → F >> 1

## The One Number

After all measurements: compute the **Tambear Score**:

```
score = geometric_mean(
    T_competitor / T_tambear,      # speed ratio
    accuracy_tambear / accuracy_competitor,  # accuracy ratio  
    passes_competitor / passes_tambear,     # efficiency ratio
)
```

One number per competitor. geometric_mean so no single metric dominates.

## Phase 1 (now): Prototype

Build the benchmark for ONE scale point (N=1M, D=100) with:
- Tambear (CPU)
- Python/NumPy (CPU)

Verify correctness. Measure time. This is the smoke test.

## Phase 2: Full sweep

All platforms. All scale points. Automated. Published.

## Phase 3: The Killer Chart

The chart that goes in every presentation:

```
X axis: dataset size (N × D)
Y axis: time (log scale)
Lines: one per platform
Annotations: where each platform runs out of memory / time
The gap between tambear and everything else = the value proposition
```
