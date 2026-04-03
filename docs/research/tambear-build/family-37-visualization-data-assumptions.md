# Family 37: Visualization Data — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (all operations are embarrassingly parallel grid evaluations or accumulate reductions)

---

## Core Insight: Compute the Data, Not the Pixels

Family 37 is NOT a rendering engine. It computes the NUMERICAL DATA that visualization consumes:
- Function evaluation over grids (the curves)
- Contour extraction (the isolines)
- Color mapping (the heatmaps)
- Histogram binning (the bar charts)
- Statistical annotations (confidence bands, error bars)

Rendering is someone else's problem (WebGPU, SVG, Canvas). Our job: produce the arrays that renderers consume. This is pure Kingdom A — embarrassingly parallel evaluation over independent grid points.

The structural rhyme: **visualization data computation = batch function evaluation**. Every plot is an `accumulate(Tiled{grid_points, parameters}, eval_expr, Collect)`.

---

## 1. Grid Evaluation (the fundamental operation)

### Uniform Grid
Given f(x), domain [a,b], resolution N:
```
x_i = a + i·(b-a)/(N-1)    for i = 0, ..., N-1
y_i = f(x_i)
```

**GPU decomposition**: `accumulate(Contiguous, f(x_i), Identity)` — each point independent. Embarrassingly parallel.

### Adaptive Grid
Uniform grids waste resolution on flat regions. Adaptive refinement:
1. Evaluate on coarse grid
2. Where |f''(x)| > threshold: subdivide
3. Repeat until resolution sufficient

**Curvature estimation**: `f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²`

This is a gather (3 neighbors) + elementwise expression. Still parallel.

### 2D Grid (for surfaces/heatmaps)
```
z_{i,j} = f(x_i, y_j)    for i=0..M-1, j=0..N-1
```

**GPU decomposition**: `accumulate(Tiled{x_grid, y_grid}, f(x_i, y_j), Identity)`. M×N independent evaluations.

### Parametric Curves
```
x(t) = f(t),  y(t) = g(t)    for t ∈ [a,b]
```

Same as grid evaluation but output is 2D point per t.

---

## 2. Contour Extraction (Marching Squares)

### Problem
Given 2D scalar field z_{i,j}, find curves where z = c (contour level).

### Marching Squares Algorithm
For each 2×2 cell of the grid:
1. Classify each corner as above/below threshold c → 4-bit case index (16 cases)
2. Look up edge crossings from case table
3. Interpolate crossing position: `x_cross = x₁ + (c - z₁)/(z₂ - z₁) · (x₂ - x₁)`

**GPU decomposition**:
- Step 1: `accumulate(Contiguous, classify(z_corners, c), Identity)` — parallel per cell
- Step 2: gather from case lookup table
- Step 3: elementwise interpolation

**Edge cases**:
- Ambiguous case (corners 0101 or 1010): use center value to disambiguate
- Saddle points: average of 4 corners, classify center
- Exactly on threshold: perturb by ε to avoid degenerate topology

### Contour Levels
**Linear spacing**: levels = linspace(z_min, z_max, N)
**Log spacing**: levels = logspace(log10(z_min), log10(z_max), N)
**Quantile spacing**: levels at percentiles of the data (uses F06 quantile computation)
**Pretty spacing**: round to "nice" numbers (1, 2, 5 multiples of powers of 10)

---

## 3. Color Mapping

### Scalar-to-Color Mapping
Given scalar value v ∈ [v_min, v_max], map to color:
```
t = (v - v_min) / (v_max - v_min)    // normalize to [0,1]
color = colormap(t)                    // lookup + interpolate
```

**GPU decomposition**: `accumulate(Contiguous, colormap_lookup(normalized_v), Identity)` — parallel per point.

### Colormap Types

| Type | Examples | Interpolation |
|------|----------|--------------|
| Sequential | viridis, plasma, inferno, magma | Linear in perceptually uniform space |
| Diverging | RdBu, coolwarm, BrBG | Two ramps from center |
| Categorical | tab10, Set1, Paired | Nearest-neighbor (no interpolation) |
| Cyclic | twilight, hsv | Wraps around |

### Perceptual Uniformity (CRITICAL)
Rainbow/jet colormaps create false features. Viridis family designed to be:
- Perceptually uniform (equal steps in data → equal steps in perceived color)
- Readable in grayscale
- Accessible to colorblind viewers

**Decision**: Default to viridis. Provide all standard colormaps. Store as 256-entry LUT in SRGB.

### Interpolation in Color Space
Linear RGB interpolation produces muddy colors. Interpolate in:
- **CIELAB** (perceptual uniformity)
- **CIELUV** (for additive displays)
- **OKLab** (modern, better than CIELAB for wide gamut)

---

## 4. Histogram Visualization Data

### Already F06 (Descriptive Statistics) for computation
Histogram = `scatter_add(bin_indices, ones)`. The visualization layer adds:

### Bin Edge Computation
```
edges_i = min + i · (max - min) / N_bins    for i = 0, ..., N_bins
```

### Bin Width Rules

| Rule | Formula | Best for |
|------|---------|----------|
| Sturges | N_bins = ⌈log₂(n) + 1⌉ | Normal-ish data |
| Scott | width = 3.49·σ·n^(-1/3) | Normal data (MSE-optimal) |
| Freedman-Diaconis | width = 2·IQR·n^(-1/3) | Robust to outliers |
| Doane | Sturges + skewness correction | Skewed data |

### Kernel Density Curve (overlay)
Already F08 KDE. Provides smooth density estimate alongside histogram bars.

### Cumulative Histogram
`accumulate(Prefix(forward), bin_count_i, Sum)` — prefix sum of bin counts.

---

## 5. Statistical Annotations

### Confidence Bands
For regression line ŷ(x):
```
CI(x) = ŷ(x) ± t_{α/2,n-p} · SE(ŷ(x))
```
where SE(ŷ(x)) = σ̂ · √(x'(X'X)⁻¹x). Uses F10 regression infrastructure.

**GPU**: evaluate CI at each grid point — parallel.

### Prediction Intervals
```
PI(x) = ŷ(x) ± t_{α/2,n-p} · √(σ̂² + SE(ŷ(x))²)
```

Wider than CI (accounts for individual variation, not just mean uncertainty).

### Error Bars
- **Standard error**: SE = σ/√n (from F06 MomentStats)
- **Confidence interval**: x̄ ± t·SE
- **Standard deviation**: x̄ ± σ (wider, shows spread not precision)
- **Min-max**: [min, max] (from F06)
- **IQR**: [Q1, Q3] (from F06 quantiles)

### Box Plot Data
From F06:
```
{Q1, median, Q3, whisker_low, whisker_high, outliers[]}
```
whisker_low = max(min, Q1 - 1.5·IQR), whisker_high = min(max, Q3 + 1.5·IQR).
Outliers = points outside whiskers.

---

## 6. Curve Fitting Visualization

### Regression Lines
From F10: compute ŷ = Xβ̂ at grid points. Parallel evaluation.

### Smoothing Curves
- **LOWESS/LOESS**: local weighted regression at each grid point. Uses F10 WLS.
- **Spline**: from F31 interpolation infrastructure.
- **Moving average**: `accumulate(Windowed(k), x_i, Mean)` — from F06.

### Residual Plots
```
residual_i = y_i - ŷ_i
```
Element-wise subtraction. Parallel.

---

## 7. Transformation Functions

### Axis Transforms
```
log:    v → log(v)       (requires v > 0)
log2:   v → log₂(v)
log10:  v → log₁₀(v)
sqrt:   v → √v           (requires v ≥ 0)
logit:  v → log(v/(1-v)) (requires 0 < v < 1)
symlog: v → sign(v)·log(1+|v|)  (handles zero and negatives)
```

**Edge cases**:
- log(0): return -∞ or clip to log(ε)
- log(negative): NaN — warn user
- logit(0) or logit(1): ±∞

### Power Transforms (for normalization visualization)
- Box-Cox: (x^λ - 1)/λ (λ ≠ 0), log(x) (λ = 0). Requires x > 0.
- Yeo-Johnson: handles x ≤ 0.
- From F09 robust statistics.

---

## 8. Projection Data (for dimensionality reduction plots)

### 2D Embeddings
From F22: PCA, t-SNE, UMAP produce 2D coordinates for each data point. Visualization just needs the (x,y) arrays.

### Biplot Data
PCA biplot needs:
- Scores (projected data points): Z = X · V[:,:2]
- Loadings (variable arrows): V[:,:2] scaled by √(eigenvalue)

Both are matrix operations (F02).

---

## 9. Edge Cases

| Operation | Edge Case | Expected |
|-----------|----------|----------|
| Grid eval | f(x) = NaN for some x | Preserve NaN (gap in plot) |
| Grid eval | f(x) = ±∞ | Clip to axis limits |
| Contour | Flat field (all same value) | No contours (empty output) |
| Contour | Value exactly on threshold | Perturb by ε |
| Color map | v outside [v_min, v_max] | Clamp to endpoints |
| Color map | NaN input | Transparent or designated "bad" color |
| Histogram | Zero-width bin (min = max) | Single bin, full count |
| Histogram | Empty data | Zero bins |
| Log axis | Zero or negative values | Error with clear message |
| Confidence band | n < p (underdetermined) | Cannot compute, return NaN |

---

## Sharing Surface

### Reuses from Other Families
- **F06 (Descriptive)**: MomentStats for error bars, quantiles for box plots, histogram counts
- **F08 (Nonparametric)**: KDE for density overlay, bootstrap for CI
- **F10 (Regression)**: Fitted values, confidence/prediction intervals
- **F22 (Dim Reduction)**: PCA/t-SNE/UMAP embeddings
- **F31 (Interpolation)**: Spline curves for smooth plotting

### Provides to Other Families
- **All families**: standardized visualization data format for any computed result
- **F36 (Symbolic)**: expression evaluation over grids for symbolic function plots

### Structural Rhymes
- **Grid evaluation = batch function evaluation** (the fundamental insight)
- **Contour extraction = thresholded classification** (same as F21 decision boundary)
- **Color mapping = gather from LUT** (same as embedding lookup in F23)

---

## Implementation Priority

**Phase 1** — Core evaluation (~80 lines):
1. Uniform grid generation (1D, 2D)
2. Batch function evaluation over grid
3. Linear color mapping (normalize + LUT interpolation)
4. Histogram bin edges (Sturges, Scott, Freedman-Diaconis)

**Phase 2** — Statistical overlays (~100 lines):
5. Confidence bands (from F10 regression)
6. Error bar data (SE, CI, SD, IQR)
7. Box plot data extraction
8. Cumulative distribution curves

**Phase 3** — Contours and transforms (~120 lines):
9. Marching squares (contour extraction)
10. Axis transforms (log, sqrt, symlog, logit)
11. Adaptive grid refinement (curvature-based)

**Phase 4** — Advanced (~80 lines):
12. Parametric curve evaluation
13. Standard colormaps (viridis, plasma, etc. as LUTs)
14. Biplot data from PCA
15. Residual diagnostic data

---

## Composability Contract

```toml
[family_37]
name = "Visualization Data"
kingdom = "A (embarrassingly parallel grid evaluations)"

[family_37.shared_primitives]
grid_eval = "Batch function evaluation over uniform/adaptive grid"
contour = "Marching squares for isoline extraction"
color_map = "Scalar-to-color via LUT interpolation"
stat_annotation = "Error bars, confidence bands, box plot data"

[family_37.reuses]
f06_descriptive = "MomentStats, quantiles, histogram counts"
f08_nonparametric = "KDE, bootstrap CI"
f10_regression = "Fitted values, SE, confidence/prediction intervals"
f22_dim_reduction = "2D embeddings for scatter plots"
f31_interpolation = "Splines for smooth curves"

[family_37.provides]
grid_data = "Evaluated function values on structured grids"
contour_data = "Polyline segments for isolines"
color_data = "RGBA arrays for heatmaps"
annotation_data = "Statistical overlay arrays (CI, error bars, whiskers)"

[family_37.consumers]
tbs_ide = "Live visualization in .tbs editor"
export = "Data arrays for external renderers (SVG, Canvas, WebGPU)"
```
