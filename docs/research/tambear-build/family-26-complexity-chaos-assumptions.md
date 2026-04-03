# Family 26: Complexity & Chaos — Mathematical Assumptions Document

**Author**: Math Researcher (Claude)
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Family**: 26 — Complexity & Chaos (from the 35-family landscape)
**Kingdom**: Mixed — see per-algorithm classification below

---

## Kingdom Classification Summary

Family 26 is **not a single kingdom**. It spans all three:

| Algorithm | Kingdom | Why |
|-----------|---------|-----|
| Correlation dimension | A | Commutative pair-count over distance matrix |
| Sample entropy | A | Commutative template-match count |
| Approximate entropy | A | Commutative template-match count (with self-matches) |
| Permutation entropy | A | Histogram of ordinal patterns -> Shannon entropy |
| Lempel-Ziv complexity | B | Sequential left-to-right parse (inherently serial) |
| Hurst exponent (R/S) | A (per block) | Commutative stats within blocks + log-log regression |
| DFA | A (per segment) | Commutative regression within segments + log-log regression |
| MFDFA | A (per segment) | Same as DFA with q-th order generalization |
| Higuchi fractal dim | A | Commutative sums of absolute differences per stride |
| Box-counting dimension | A | Commutative bin counting at multiple scales |
| Katz dimension | A | Commutative distance sums |
| Petrosian dimension | A | Commutative sign-change counting |
| Lyapunov (Rosenstein) | A + B | Distance matrix (A) + sequential divergence tracking (log-log regression) |
| Lyapunov (Kantz) | A + B | Distance matrix (A) + neighborhood-averaged divergence |
| Lyapunov (Wolf) | C | Iterative re-orthogonalization (outer loop) |
| RQA | A + B | Distance matrix (A) + segmented scan for line statistics (B) |
| Symbolic dynamics | A | Histogram-based (same as permutation entropy) |

**Key structural insight**: The O(N^2) pairwise distance matrix on embedded vectors is the shared bottleneck for Lyapunov, correlation dimension, RQA, and sample entropy. This is the SAME `accumulate(Tiled(N,N), dist_expr, op)` as DBSCAN and KNN. **One GPU distance computation serves the entire family.**

---

## 0. Shared Foundation: Phase Space Reconstruction (Takens' Embedding)

Almost every algorithm in Family 26 operates on **embedded vectors**, not the raw time series.

### Takens' Embedding Theorem (1981)

**Theorem**: Let M be a compact manifold of dimension d, and let phi: M -> M be a smooth diffeomorphism with a smooth observation function h: M -> R. Then for generic (phi, h), the delay-coordinate map:

```
F: M -> R^m
F(x) = (h(x), h(phi(x)), h(phi^2(x)), ..., h(phi^(m-1)(x)))
```

is a diffeomorphism (smooth embedding) of M into R^m when m >= 2d + 1.

**In practice**: Given a scalar time series {x(t)}, construct m-dimensional vectors:

```
y(i) = [x(i), x(i+tau), x(i+2*tau), ..., x(i+(m-1)*tau)]
```

where:
- **m** = embedding dimension
- **tau** = time delay (in samples)
- **N_embed** = N - (m-1)*tau vectors produced from N original points

### Choosing tau: Time Delay

**Method 1: First minimum of mutual information** (Fraser & Swinney, 1986)

Compute the mutual information between x(t) and x(t+tau) as a function of tau:

```
I(tau) = SUM_ij p(x_i, x_{i+tau}_j) * log[p(x_i, x_{i+tau}_j) / (p(x_i) * p(x_{i+tau}_j))]
```

Choose tau = first local minimum of I(tau). This captures nonlinear independence (superior to autocorrelation).

**GPU decomposition**: For each candidate tau, compute 2D histogram via `scatter_add_2d` then entropy. This is Family 25 mutual information repeated for tau = 1, 2, ..., tau_max. Embarrassingly parallel across tau values.

**Method 2: First zero-crossing of autocorrelation** (simpler, linear-only)

```
R(tau) = (1/N) * SUM_{t=1}^{N-tau} (x(t) - mu)(x(t+tau) - mu) / sigma^2
```

Choose tau where R(tau) first crosses zero. Alternatively: first point where R(tau) drops below 1/e.

**GPU decomposition**: `accumulate(All, x(t)*x(t+tau), Add)` for each tau. Or via FFT: `R(tau) = IFFT(|FFT(x)|^2)` — one FFT pair computes all lags simultaneously.

**Standard choices**: tau = 1 for maps, tau from MI for flows. Financial data: tau = 1 is typical (tick data is already a map).

**Failure modes**:
- tau too small: embedded vectors are nearly collinear, attractor is "squished" along the diagonal
- tau too large: trajectories decorrelate, no dynamical structure preserved
- Strong periodicity: MI oscillates, take first minimum not a secondary one

### Choosing m: Embedding Dimension

**Method: False Nearest Neighbors (FNN)** (Kennel, Brown & Abarbanel, 1992)

For each point y(i) in m-dimensional embedding, find its nearest neighbor y(nn(i)). Check if they remain neighbors in (m+1)-dimensional embedding:

```
R_i(m) = |x(i + m*tau) - x(nn(i) + m*tau)| / ||y(i) - y(nn(i))||_m
```

A pair is a "false nearest neighbor" if:

```
R_i(m) > R_threshold    (typically R_threshold = 15)
```

AND/OR using Kennel's second criterion:

```
|x(i + m*tau) - x(nn(i) + m*tau)| / R_A > A_threshold    (typically A_threshold = 2)
```

where R_A = std(x) is the attractor size.

**FNN fraction**: F(m) = (number of false NNs at dimension m) / (total NNs at dimension m)

Choose m = smallest dimension where F(m) approximately equals 0 (typically < 1%).

**GPU decomposition**: For each candidate m, build embedded vectors and find nearest neighbors. The NN search is `accumulate(Tiled(N,N), (a_k-b_k)^2, Add)` + `reduce(All, ArgMin)` per row. Parallelizable across candidate m values.

**Standard choices**: m = 2 to 10 for financial data. Test by computing the target measure at multiple m values and checking convergence.

**Failure modes**:
- Noise floor: FNN never reaches zero for noisy data. Use a threshold like 1-5%.
- N too small for large m: need N >> 10^(m/2) points (Eckmann & Ruelle bound)

### TamSession Sharing Contract

```toml
[sharing]
produces_to_session = [
    "EmbeddedVectors(data_id, m, tau)",
    "DistanceMatrix(data_id, m, tau, metric)",
    "NearestNeighborIndex(data_id, m, tau)",
    "AutocorrelationFunction(data_id)",
    "MutualInformationProfile(data_id)",
]
consumes_from_session = [
    "EmbeddedVectors(data_id, m, tau)",
    "DistanceMatrix(data_id, m, tau, metric)",
]
```

---

## 1. Lyapunov Exponents

### 1.1 Physical Meaning

The Lyapunov exponent measures the average exponential rate of divergence (or convergence) of initially nearby trajectories in phase space. For a continuous-time dynamical system dx/dt = F(x):

```
||delta_x(t)|| ~ ||delta_x(0)|| * e^{lambda*t}
```

- lambda_1 > 0: chaos (sensitive dependence on initial conditions)
- lambda_1 = 0: marginally stable (limit cycle, quasiperiodic)
- lambda_1 < 0: stable fixed point (dissipative)
- **Sum of all Lyapunov exponents** = divergence of the flow (Liouville's theorem). For dissipative systems: SUM(lambda_i) < 0.

For an m-dimensional system, there are m Lyapunov exponents lambda_1 >= lambda_2 >= ... >= lambda_m. The **Lyapunov dimension** (Kaplan-Yorke):

```
D_KY = j + SUM_{i=1}^{j} lambda_i / |lambda_{j+1}|
```

where j is the largest integer such that SUM_{i=1}^{j} lambda_i >= 0.

### 1.2 Rosenstein et al. (1993) — Largest Lyapunov Exponent

**Reference**: Rosenstein, Collins & De Luca, "A practical method for calculating largest Lyapunov exponents from small data sets," Physica D 65:117-134.

**Algorithm**:

**Step 1**: Embed the time series. Construct y(i) in R^m for i = 1, ..., N_embed.

**Step 2**: For each reference point y(i), find its nearest neighbor y(j*) subject to temporal separation:

```
j*(i) = argmin_{j : |i-j| > w} ||y(i) - y(j)||_2
```

where w is the mean period of the data (Theiler window), estimated as:
- w = first zero-crossing of autocorrelation, or
- w = 1/f_dominant from FFT peak, or
- at minimum, w >= tau (the embedding delay)

**Step 3**: Track the divergence for each pair over dn = 0, 1, 2, ..., dn_max steps:

```
d_i(dn) = ||y(i + dn) - y(j*(i) + dn)||_2
```

**Step 4**: Average the log divergence across all reference points:

```
S(dn) = (1/N_ref) * SUM_{i=1}^{N_ref} ln(d_i(dn))
```

where N_ref = number of valid reference points (those with i + dn_max and j*(i) + dn_max still in bounds).

**Step 5**: The largest Lyapunov exponent is the slope of S(dn) vs dn*dt in the **linear scaling region**:

```
lambda_1 = d[S(dn)] / d[dn * dt]
```

where dt is the sampling interval of the original time series.

**Identifying the linear region**: Look for a plateau in the local slope dS/d(dn). The linear region typically spans from a few steps (after initial transients) to before saturation (when diverging trajectories wrap around the attractor). In practice: use the first 10-30% of the S(dn) curve.

**Inputs**:
- Time series x(t) of length N
- Embedding dimension m
- Time delay tau
- Sampling interval dt
- Theiler window w (default: mean period)
- Maximum divergence steps dn_max (default: N_embed / 4)

**Outputs**:
- lambda_1: largest Lyapunov exponent (in units of 1/dt)
- S(dn): the divergence curve (array)
- R^2: quality of linear fit
- Standard error of lambda_1

**Edge cases**:
- d_i(0) = 0: nearest neighbor is the point itself (temporal separation too small). Skip.
- d_i(dn) = 0: trajectories merged. Set ln(d) = -infinity, skip in average.
- No valid nearest neighbors: return NaN.
- S(dn) oscillates: no clear linear region -> unreliable lambda_1.

**Assumptions**:
- Time series is from a deterministic dynamical system (possibly with measurement noise)
- N is large enough: Eckmann-Ruelle criterion suggests N >> 10^(2+0.4m)
- Stationarity (at least approximate) over the data window
- The attractor is compact

**GPU decomposition**:
- **Step 2** (nearest neighbor search): `accumulate(Tiled(N,N), (a_k-b_k)^2, Add)` to get distance matrix, then `reduce(per_row, ArgMin)` with temporal exclusion mask. This is the O(N^2 * m) bottleneck. Shares DistanceMatrix with correlation dimension and RQA.
- **Steps 3-4** (divergence tracking): For each (i, j*(i)) pair, the divergence at step dn is a single distance computation. All pairs and all dn values are independent -> embarrassingly parallel. `gather(i+dn, source=embedded_vectors)` and `gather(j*(i)+dn, source=embedded_vectors)`, then L2 distance and log.
- **Step 5** (regression): `accumulate(All, x_k*y_k, Add)` and `accumulate(All, x_k^2, Add)` — two reductions for OLS slope.
- **Kingdom**: A for the distance matrix, then embarrassingly parallel for divergence tracking, then A for regression. No sequential dependency.

### 1.3 Kantz (1994) — Largest Lyapunov Exponent

**Reference**: Kantz, "A robust method to estimate the maximal Lyapunov exponent of a time series," Physics Letters A 185:77-87.

**Key difference from Rosenstein**: Instead of tracking only the single nearest neighbor, Kantz tracks ALL neighbors within radius epsilon and averages. This makes the estimate more robust to noise.

**Algorithm**:

**Step 1**: Embed as before.

**Step 2**: For each reference point y(i), find the epsilon-neighborhood:

```
U_epsilon(i) = {j : ||y(i) - y(j)||_2 < epsilon, |i - j| > w}
```

**Step 3**: Compute the average divergence from the neighborhood:

```
S(epsilon, dn) = (1/N_ref) * SUM_i ln[ (1/|U_epsilon(i)|) * SUM_{j in U_epsilon(i)} ||y(i+dn) - y(j+dn)||_2 ]
```

Note the order: average INSIDE the log for each reference point, THEN average over reference points. This is crucial — it differs from averaging log-distances.

**Step 4**: lambda_1 = slope of S(epsilon, dn) vs dn*dt. If the estimate is correct, it should be independent of epsilon (check multiple epsilon values).

**Inputs**: Same as Rosenstein, plus:
- epsilon: neighborhood radius (or multiple radii for robustness check)
- Typical: epsilon = 2-10% of the attractor diameter (std of embedded vectors)

**Outputs**: Same as Rosenstein. Additionally: S(epsilon, dn) for multiple epsilon values.

**Advantage over Rosenstein**: More robust to noise because averaging over a neighborhood reduces the effect of spurious nearest neighbors.

**Disadvantage**: Slower — O(N^2) for the neighborhood search at each dn (unless DistanceMatrix is precomputed).

**GPU decomposition**:
- **Step 2**: The epsilon-neighborhood test is `R(i,j) = ||y(i)-y(j)|| < epsilon` — the SAME recurrence matrix as RQA. Shares DistanceMatrix.
- **Step 3**: For each dn, compute divergences for all (i,j) pairs where j in U_epsilon(i), then reduce within each neighborhood (per-row reduce with mask), then log, then global average. This is `accumulate(Masked(recurrence_row_i), ||y(i+dn)-y(j+dn)||, Add)` for each i and dn.
- **Kingdom**: A (all reductions are commutative).

### 1.4 Wolf et al. (1985) — Full Lyapunov Spectrum

**Reference**: Wolf, Swift, Swinney & Vastano, "Determining Lyapunov exponents from a time series," Physica D 16:285-317.

**Purpose**: Estimates ALL m Lyapunov exponents, not just the largest.

**Algorithm**:

This is **Kingdom C** — inherently iterative.

**Step 1**: Choose a fiducial trajectory point and m-1 nearby points defining an initial tangent frame.

**Step 2**: Evolve all m points forward by one replacement time T_replace:

```
y_k(t + T_replace) for k = 0, 1, ..., m-1
```

where y_0 is the fiducial and y_1,...,y_{m-1} are the neighbors.

**Step 3**: Compute the displacement vectors:

```
delta_k(t + T_replace) = y_k(t + T_replace) - y_0(t + T_replace)
```

**Step 4**: Gram-Schmidt orthonormalize the displacement vectors to get the orthogonal frame {e_1, ..., e_{m-1}}.

Record the norms BEFORE orthonormalization: these contribute to the Lyapunov exponents.

```
lambda_k = (1/T_total) * SUM_{steps} ln(||delta_k before GS||)
```

But with the complication that GS ordering matters: lambda_1 gets the full stretching, lambda_2 gets the residual after removing the lambda_1 direction, etc.

**Step 5**: Replace the neighbors: find new points near the fiducial along the orthonormalized directions. Continue from Step 2.

**The Gram-Schmidt procedure** (exact formulas):

Given vectors {v_1, ..., v_{m-1}}:

```
u_1 = v_1
e_1 = u_1 / ||u_1||

u_k = v_k - SUM_{j=1}^{k-1} <v_k, e_j> * e_j
e_k = u_k / ||u_k||

lambda_k contribution = ln(||u_k||)
```

**Inputs**:
- Time series, m, tau, dt
- T_replace: replacement time (typically mean orbital period / 10)

**Outputs**:
- Full spectrum {lambda_1, lambda_2, ..., lambda_m}
- Kaplan-Yorke dimension D_KY

**Failure modes**:
- m too large relative to N: not enough neighbors for replacement
- Strong noise: spectrum dominated by noise dimensions (positive exponents from noise)
- Replacement failure: no suitable replacement point found. Algorithm stalls.

**Assumptions**:
- Data length: N must be much larger than for single-exponent methods
- Stationarity: the attractor must be stationary over the analysis window
- Typical requirement: at least several orbital periods of data

**GPU decomposition**:
- **Kingdom C**: outer iteration loop (cannot be parallelized)
- **Inner loop**: each iteration involves a nearest-neighbor search (A) and Gram-Schmidt (sequential over k, but each projection is an inner product -> A)
- The replacement step (finding new nearby points along orthogonal directions) is a constrained nearest-neighbor search
- **Not a good GPU target** in its pure form. Sano-Sawada (1985) and Eckmann-Ruelle variants may be more GPU-friendly.

**Alternative for GPU**: The Jacobian method (requires known equations of motion — not applicable to time series). For data-driven full spectrum: consider the Sano-Sawada method, which computes local Jacobians from the data and uses QR decomposition instead of neighbor replacement.

### 1.5 Standard Parameter Choices for Financial Data

| Parameter | Typical value | Rationale |
|-----------|---------------|-----------|
| m | 5-10 | FNN suggests m=5-7 for financial returns |
| tau | 1 | Tick data is already discrete; MI usually confirms tau=1 |
| w (Theiler) | 10-50 | Larger than any autocorrelation structure |
| dn_max | N_embed/4 | Conservative: don't track beyond 25% of data |
| Linear region | First 10-20% of S(dn) | Before attractor saturation |
| epsilon (Kantz) | 0.05*std to 0.2*std | Multiple values for robustness check |

---

## 2. Correlation Dimension (Grassberger-Procaccia, 1983)

**Reference**: Grassberger & Procaccia, "Characterization of strange attractors," Physical Review Letters 50:346-349.

### 2.1 Correlation Integral

Given N embedded vectors y(1), ..., y(N) in R^m, the correlation integral is:

```
C(r) = (2 / (N * (N-1))) * #{(i,j) : i < j, ||y(i) - y(j)|| < r}
```

This counts the fraction of all pairs of points that are within distance r of each other.

### 2.2 Scaling and Dimension

For a fractal attractor, the correlation integral scales as a power law for small r:

```
C(r) ~ r^{D_2}    as r -> 0
```

where D_2 is the **correlation dimension** (the second-order Renyi dimension). Therefore:

```
D_2 = lim_{r->0} d[ln C(r)] / d[ln r]
```

In practice, compute C(r) at many r values, plot ln C(r) vs ln r, and find the slope in the **scaling region** (the range of r where the relationship is approximately linear).

### 2.3 Complete Algorithm

**Step 1**: Embed the time series with dimension m and delay tau.

**Step 2**: Compute pairwise distances between all embedded vectors:

```
d(i,j) = ||y(i) - y(j)||_2    for all i < j
```

This produces N_embed * (N_embed - 1) / 2 distances.

**Step 3**: Choose r values. Typically 20-50 logarithmically spaced values between r_min and r_max:

```
r_min = max(median(d) * 0.001, machine_epsilon)
r_max = max(d) * 0.9
r_k = r_min * (r_max/r_min)^{k/(n_r-1)}    for k = 0, ..., n_r-1
```

**Step 4**: For each r, count pairs:

```
C(r_k) = (2 / (N*(N-1))) * |{(i,j) : i < j, d(i,j) < r_k}|
```

**Step 5**: Log-log regression in the scaling region:

```
ln C(r) = D_2 * ln(r) + const
```

**Identifying the scaling region**: Plot the local slope d[ln C(r)]/d[ln r] and look for a plateau. The plateau value is D_2. Alternatively: use the range where local slope varies by less than 10%.

### 2.4 Theiler Correction (1986)

**Reference**: Theiler, "Spurious dimension from correlation algorithms applied to limited time-series data," Physical Review A 34:2427-2432.

**Problem**: Temporally close points are correlated not because of attractor geometry but because of temporal proximity. This inflates C(r) at small r, biasing D_2 downward.

**Fix**: Exclude pairs with |i - j| <= w from the count:

```
C_w(r) = (2 / N_w) * #{(i,j) : i < j, |i-j| > w, ||y(i) - y(j)|| < r}
```

where N_w = number of valid pairs after temporal exclusion.

**Standard**: w = tau*m (one "orbit" in embedding space) or w estimated from autocorrelation decay.

### 2.5 Lacunarity

Lacunarity measures the "gappiness" or texture of a fractal at a given scale r:

```
Lambda(r) = <C(r)^2> / <C(r)>^2 - 1
```

More precisely, for box-counting: divide space into boxes of size r. Let n_i be the mass (point count) in box i.

```
Z(q, r) = SUM_i n_i^q / (SUM_i n_i)^q

Lambda(r) = Z(2,r) / Z(1,r)^2 - 1
```

High lacunarity: heterogeneous distribution (clumpy). Low lacunarity: homogeneous.

### 2.6 Inputs, Outputs, Edge Cases

**Inputs**:
- Time series x(t) of length N
- Embedding dimension m (or range of m for convergence check)
- Time delay tau
- Theiler window w
- Number of r values (default: 30)

**Outputs**:
- D_2: correlation dimension estimate
- R^2: quality of log-log fit
- ln C(r) vs ln r data (for visual inspection)
- D_2(m) for multiple m (convergence plot): D_2 should plateau as m increases

**Edge cases**:
- All points identical: C(r) = 1 for all r > 0 -> D_2 = 0 (point attractor)
- Uniform random data in R^m: D_2 -> m (fills the space)
- N too small: C(r) is noisy, D_2 unreliable. Minimum: N >> r^{-D_2} for the smallest r used.
- No scaling region: local slope never plateaus -> no well-defined dimension. Return NaN with diagnostic.

**Failure modes**:
- **Lacunarity bias**: for r too large, C(r) saturates. For r too small, finite-N effects dominate.
- **High-dimensional noise**: adds noise dimensions, inflating D_2.
- **Non-stationarity**: mixing different attractor states corrupts the integral.

### 2.7 GPU Decomposition

**Step 2** (pairwise distances): `accumulate(Tiled(N,N), (a_k-b_k)^2, Add)` -> N x N distance matrix. **This is shared with Lyapunov, RQA, and sample entropy.**

**Step 4** (pair counting at each r): Given the sorted distance list (or the full N x N matrix), counting pairs within r is:
- If distances are sorted: binary search for each r -> O(log(N^2) * n_r). CPU is fine.
- On GPU with full matrix: `accumulate(All, Theta(r - d(i,j)), Add)` for each r. With n_r=30 radii, this is 30 reductions over the same matrix — fuse into one kernel with 30 output counters.

**Step 5** (regression): Two reductions for OLS. Trivial.

**Kingdom**: A. Everything is commutative. The distance matrix is the MSR.

---

## 3. Sample Entropy (Richman & Moorman, 2000)

**Reference**: Richman & Moorman, "Physiological time-series analysis using approximate entropy and sample entropy," American Journal of Physiology 278:H2039-H2049.

### 3.1 Definition

Given a time series {x(1), ..., x(N)}, form templates of length m:

```
X_i^m = [x(i), x(i+1), ..., x(i+m-1)]    for i = 1, ..., N-m+1
```

(Note: templates are consecutive values, NOT delay-embedded with tau. Sample entropy traditionally uses tau=1.)

Define the distance between templates using the Chebyshev (L-infinity) norm:

```
d(X_i^m, X_j^m) = max_{k=0,...,m-1} |x(i+k) - x(j+k)|
```

### 3.2 Template Matching

Count the number of matches at dimension m, **excluding self-matches** (i != j):

```
B_i^m(r) = (1/(N-m)) * #{j : j != i, d(X_i^m, X_j^m) < r}
```

where the inequality is **strict** (< not <=). This is a convention choice — Richman & Moorman use strict.

The total count at dimension m:

```
B^m(r) = (1/(N-m+1)) * SUM_{i=1}^{N-m+1} B_i^m(r)
        = (2 / ((N-m+1)*(N-m))) * #{(i,j) : i != j, d(X_i^m, X_j^m) < r}
```

Similarly at dimension m+1:

```
A^m(r) = (2 / ((N-m)*(N-m-1))) * #{(i,j) : i != j, d(X_i^{m+1}, X_j^{m+1}) < r}
```

### 3.3 Sample Entropy Formula

```
SampEn(m, r, N) = -ln(A^m(r) / B^m(r))
```

Equivalently, using raw pair counts:

```
SampEn = -ln(count_A / count_B)
```

where:
- count_B = number of template pairs matching at length m
- count_A = number of template pairs matching at length m+1
- count_A <= count_B always (matching at m+1 implies matching at m)

### 3.4 Interpretation

- SampEn = 0: perfectly regular (every template matches at m+1 whenever it matches at m)
- SampEn -> infinity: no matches at m+1 even when matching at m (highly complex/random)
- Typical range for financial data: 0.1 - 2.5
- SampEn is the negative natural log of the conditional probability that templates matching for m points continue to match for m+1 points

### 3.5 Difference from Approximate Entropy

| Property | SampEn | ApEn |
|----------|--------|------|
| Self-matches | Excluded (j != i) | Included (j = i allowed) |
| Log placement | -ln(A/B) = single log | Phi^m - Phi^{m+1} = difference of mean logs |
| Bias | Less biased for short N | More biased (self-matches inflate counts) |
| Defined when | A > 0 | Always (self-matches guarantee C_i > 0) |
| Data dependence | Less dependent on N | Strongly N-dependent |

### 3.6 Standard Parameters

```
m = 2        (template length)
r = 0.2*sigma    (tolerance, where sigma = std(x))
```

The r = 0.2*sigma convention is from Pincus (1991), widely adopted. Some authors use r = 0.1*sigma to 0.25*sigma.

**CRITICAL**: r should be computed from the ORIGINAL time series, not the embedded vectors.

### 3.7 Multiscale Entropy (Costa et al., 2002)

**Reference**: Costa, Goldberger & Peng, "Multiscale entropy analysis of complex physiologic time series," Physical Review Letters 89:068102.

**Algorithm**: Coarse-grain the time series at multiple scales, then compute SampEn at each scale.

**Coarse-graining at scale s**:

```
y_j^(s) = (1/s) * SUM_{i=(j-1)*s+1}^{j*s} x(i)     for j = 1, ..., floor(N/s)
```

This is a non-overlapping block average of s consecutive points.

**MSE profile**: SampEn(m, r, y^(s)) for s = 1, 2, ..., s_max.

- White noise: MSE decreases monotonically (coarse-graining removes randomness)
- 1/f noise: MSE is approximately constant across scales
- Complex physiological signals: MSE is high and roughly constant (complexity preserved across scales)
- Pathological signals: MSE decreases at coarse scales (loss of complexity)

**GPU decomposition**: The coarse-graining is `accumulate(Segmented(block_boundaries), x_i, Add)` followed by division by s — a segmented reduction. Then SampEn computation for each scale. All scales are independent -> parallel.

### 3.8 Inputs, Outputs, Edge Cases

**Inputs**:
- Time series x(t), length N
- m: template length (default: 2)
- r: tolerance (default: 0.2*std(x))
- For MSE: s_max (default: 20)

**Outputs**:
- SampEn: scalar (for single-scale)
- MSE profile: SampEn(s) for s = 1, ..., s_max
- count_A, count_B: raw match counts (for diagnostics)

**Edge cases**:
- count_A = 0: SampEn = +infinity (undefined). Return +infinity with warning.
- count_B = 0: No matches at dimension m. Data may be too short or r too small. Return NaN.
- N < m+2: Not enough data to form templates. Return NaN.
- Constant signal: all templates match -> count_A = count_B -> SampEn = 0.
- r = 0: Only exact matches -> SampEn = +infinity for any non-constant signal.

**Failure modes**:
- N too small (< 200): SampEn is unreliable. Costa suggests N >= 750 for MSE.
- r sensitivity: SampEn can vary significantly with r. Report results for multiple r values.
- Non-stationarity: SampEn assumes stationarity. Pre-test or segment.

### 3.9 GPU Decomposition

The bottleneck is the O(N^2) pairwise template comparison.

**Approach 1: Full distance matrix**
```
D_m[i,j] = max_{k=0,...,m-1} |x(i+k) - x(j+k)|    (L-infinity norm)
```

This requires a **TiledOp with Max combine** instead of Sum:
```
accumulate(Tiled(N,N), |a_k - b_k|, Max)
```

Then: count_B = `accumulate(All, Theta(r - D_m[i,j]) * (1-delta_{ij}), Add)`

For m+1: extend each template by one element and recompute (or: note that d(X_i^{m+1}, X_j^{m+1}) = max(d(X_i^m, X_j^m), |x(i+m) - x(j+m)|), so you only need ONE additional comparison per pair).

**Approach 2: Fused kernel (avoids materializing full matrix)**

For each pair (i,j), compute both the m-dimensional and (m+1)-dimensional distances in the same kernel, and emit two counters:
```
d_m = max_{k=0,...,m-1} |x(i+k) - x(j+k)|
d_{m+1} = max(d_m, |x(i+m) - x(j+m)|)
count_B += (d_m < r) ? 1 : 0
count_A += (d_{m+1} < r) ? 1 : 0
```

This is ONE tiled kernel with two output counters. No intermediate matrix materialized.

**Kingdom**: A. The pair counting is commutative.

---

## 4. Approximate Entropy (Pincus, 1991)

**Reference**: Pincus, "Approximate entropy as a complexity measure," Proceedings of the National Academy of Sciences 88:2297-2301.

### 4.1 Definition

Given templates X_i^m as in SampEn, define the correlation count **including self-matches**:

```
C_i^m(r) = #{j : 1 <= j <= N-m+1, d(X_i^m, X_j^m) <= r} / (N-m+1)
```

Note: inequality is **<=** (not strict <), and j = i IS counted. This guarantees C_i^m(r) >= 1/(N-m+1) > 0, so the logarithm is always defined.

Define:

```
Phi^m(r) = (1/(N-m+1)) * SUM_{i=1}^{N-m+1} ln(C_i^m(r))
```

### 4.2 Approximate Entropy Formula

```
ApEn(m, r, N) = Phi^m(r) - Phi^{m+1}(r)
```

Note the structure: this is a **difference of average logarithms**, NOT the logarithm of a ratio (as in SampEn). By Jensen's inequality:

```
E[ln(X)] <= ln(E[X])
```

This means ApEn and SampEn are NOT algebraically equivalent even for the same match counts. ApEn systematically overestimates regularity (underestimates complexity) relative to SampEn.

### 4.3 Why Self-Matches Are Problematic

Self-matches create an artificial floor: C_i^m(r) >= 1/(N-m+1). For short data:
- This floor is significant (e.g., 1/100 for N=102, m=2)
- It makes ApEn strongly N-dependent
- It causes bias: ApEn is lower than expected for short sequences

**Consensus**: SampEn is preferred for all new work. ApEn is included for backward compatibility and comparison with legacy literature.

### 4.4 GPU Decomposition

Identical to SampEn but:
1. Count includes self-matches (no j != i exclusion)
2. Compute Phi^m(r): for each i, count C_i^m(r), take ln, then average
3. This requires a per-row reduction (count matches per reference point), then element-wise log, then global average

```
Step 1: C_i[j] = (d(X_i^m, X_j^m) <= r) ? 1 : 0    (N x N boolean matrix)
Step 2: count_i = accumulate(row_i, C_i[j], Add)      (per-row sum)
Step 3: ln_count_i = ln(count_i / (N-m+1))            (element-wise)
Step 4: Phi^m = accumulate(All, ln_count_i, Add) / (N-m+1)  (global average)
```

**Kingdom**: A. All reductions are commutative.

---

## 5. Permutation Entropy (Bandt & Pompe, 2002)

**Reference**: Bandt & Pompe, "Permutation entropy: A natural complexity measure for time series," Physical Review Letters 88:174102.

### 5.1 Ordinal Patterns

For an order m and delay tau, map each length-m subsequence to its ordinal pattern (rank order):

Given sub-sequence (x(t), x(t+tau), ..., x(t+(m-1)*tau)), compute the permutation pi in S_m such that:

```
x(t + pi(0)*tau) <= x(t + pi(1)*tau) <= ... <= x(t + pi(m-1)*tau)
```

That is, pi is the sorting permutation of the sub-sequence.

**Example** (m=3): The subsequence (4, 2, 7) has rank order (1, 0, 2) — the 0th element is second-smallest, the 1st element is smallest, the 2nd element is largest. The permutation is pi = (1, 0, 2).

There are m! possible ordinal patterns for order m.

### 5.2 Pattern Distribution

Count the frequency of each pattern over the time series:

```
n(pi) = #{t : the ordinal pattern at time t is pi}
```

Total patterns: N_pat = N - (m-1)*tau

Probability: p(pi) = n(pi) / N_pat

### 5.3 Permutation Entropy

Shannon entropy of the ordinal pattern distribution:

```
H_PE = -SUM_{pi in S_m} p(pi) * ln(p(pi))
```

with the convention 0*ln(0) = 0.

**Normalized permutation entropy**:

```
h_PE = H_PE / ln(m!)
```

Range: h_PE in [0, 1]. h_PE = 0 for monotonic sequences (only one pattern occurs). h_PE = 1 for fully random sequences (uniform pattern distribution).

### 5.4 Lehmer Code (Pattern to Index)

To convert a permutation to a unique integer index for histogramming:

```
lehmer(pi) = SUM_{i=0}^{m-1} c_i * (m-1-i)!
```

where c_i = #{j > i : pi(j) < pi(i)} is the number of elements to the right of position i that are smaller.

This maps each permutation to a unique integer in [0, m!-1].

### 5.5 Weighted Permutation Entropy (Fadlallah et al., 2013)

Standard PE ignores amplitude — it treats the pattern (1.001, 1.002, 1.003) the same as (1, 100, 10000). Weighted PE assigns a weight to each pattern occurrence based on the variance of the sub-sequence:

```
w(t) = (1/m) * SUM_{k=0}^{m-1} (x(t + k*tau) - X_bar_t)^2
```

where X_bar_t is the mean of the sub-sequence.

Then:

```
p_w(pi) = SUM_{t : pattern(t)=pi} w(t) / SUM_t w(t)
```

```
H_WPE = -SUM_pi p_w(pi) * ln(p_w(pi))
```

### 5.6 Standard Parameters

| m | Number of patterns (m!) | Minimum N for reliable estimation |
|---|------------------------|-----------------------------------|
| 3 | 6 | 120 (20 * m!) |
| 4 | 24 | 480 |
| 5 | 120 | 2400 |
| 6 | 720 | 14400 |
| 7 | 5040 | 100800 |

**Rule of thumb**: N >= 5*m! at minimum (Bandt recommends more).

tau = 1 for most applications. tau > 1 for multiscale analysis.

### 5.7 Forbidden Patterns

For deterministic systems, some ordinal patterns may never occur (or occur with probability 0). The fraction of missing (forbidden) patterns is itself a complexity measure:

```
f_forbidden = (m! - |{pi : p(pi) > 0}|) / m!
```

For random data: f_forbidden ~ 0 (all patterns occur). For deterministic maps: some patterns are structurally forbidden.

### 5.8 Inputs, Outputs, Edge Cases

**Inputs**:
- x(t), length N
- m: pattern order (default: 3, recommended range: 3-7)
- tau: delay (default: 1)

**Outputs**:
- H_PE: permutation entropy (nats)
- h_PE: normalized PE (dimensionless, in [0,1])
- Pattern counts: histogram of m! pattern frequencies
- f_forbidden: fraction of forbidden patterns

**Edge cases**:
- Constant signal: only one pattern (all equal). Handling of ties matters — see below.
- Ties: x(t+k*tau) = x(t+j*tau) for some k != j. Standard: break ties by temporal order (first occurrence is "smaller"). Some implementations randomize tie-breaking.
- m = 1: meaningless (1! = 1, only one trivial pattern). Require m >= 2.
- N < (m-1)*tau + 1: no patterns can be formed. Return NaN.

**Failure modes**:
- m too large for N: sparse pattern counts, unreliable probabilities
- tau inappropriate: can miss structure at the wrong time scale

### 5.9 GPU Decomposition

**Step 1**: For each t, compute the Lehmer code of the sub-sequence at position t. This is an element-wise operation (m comparisons per element). For m <= 7, this is a fixed-size sort of <= 7 elements — trivially parallelizable.

```
index[t] = lehmer_code(x[t], x[t+tau], ..., x[t+(m-1)*tau])
```

**Step 2**: Histogram the indices:

```
counts = accumulate(ByKey{index, m!}, 1, Add)
```

This is a scatter_add with m! bins.

**Step 3**: Shannon entropy on the histogram:

```
H_PE = -accumulate(Contiguous, entr(counts[k]/N_pat), Add)
```

**Kingdom**: A. Scatter-add histogram + entropy reduction. Identical structure to Family 25 Shannon entropy.

---

## 6. Lempel-Ziv Complexity (LZ76)

**Reference**: Lempel & Ziv, "On the Complexity of Finite Sequences," IEEE Transactions on Information Theory IT-22:75-81, 1976.

### 6.1 Algorithm (LZ76)

Given a binary sequence S = s(1)s(2)...s(N):

**Initialize**: complexity counter c = 1, current position pointer i = 1, length of current word l = 1, pointer to search window k = 1, maximum search extent q = 1.

**Parse**: Scan left-to-right. At each step, check if the substring S[k..k+l-1] has appeared as a substring in S[1..k+l-2] (the history preceding the current word, extended by l-1 characters of the current word):

- **If found**: extend the current word: l += 1
- **If NOT found**: register new word, increment complexity: c += 1; advance k += l; reset l = 1

Continue until k + l > N.

**After termination**: Add 1 to c if the last word was found (it was being extended when parsing ended).

### 6.2 Normalization

For a random binary sequence of length N, the expected number of distinct words is:

```
E[c(N)] ~ N / log_2(N)
```

(Lempel & Ziv, 1976, Theorem 2)

Normalized complexity:

```
LZ_norm = c(N) / (N / log_2(N))
```

For random binary: LZ_norm ~ 1.
For periodic: LZ_norm -> 0 as N -> infinity.
For chaotic systems: LZ_norm typically 0.5 - 1.0.

### 6.3 Binarization Strategy

For continuous-valued time series, binarize before applying LZ:

**Median threshold** (most common):

```
s(t) = 1 if x(t) > median(x), else 0
```

**Mean threshold**:

```
s(t) = 1 if x(t) > mean(x), else 0
```

**Derivatives** (slope-based):

```
s(t) = 1 if x(t+1) > x(t), else 0
```

**Multi-symbol** (extends to alphabet size > 2):

Quantize into k symbols using k-1 thresholds (quantiles). Then LZ still applies — the normalization becomes N / log_k(N).

### 6.4 Inputs, Outputs, Edge Cases

**Inputs**:
- x(t), length N (continuous — will be binarized)
- Binarization method (default: median threshold)
- Alternatively: pre-binarized sequence

**Outputs**:
- c(N): raw complexity (number of distinct words)
- LZ_norm: normalized complexity
- Word list (optional: the actual parsed words)

**Edge cases**:
- Constant signal: c = 1 (one "word"), LZ_norm -> 0
- Alternating 0-1: c = 2 (after initial transient), LZ_norm -> 0
- N < 3: meaningless. Return NaN.
- All same value after binarization (highly skewed data): c = 1

**Failure modes**:
- Short sequences (N < 100): normalization is inaccurate
- Binarization destroys amplitude information
- Sensitive to the binarization threshold

### 6.5 GPU Decomposition

**This is Kingdom B — inherently sequential.** The parsing depends on the history of what has been seen.

The substring search at each step (checking if the current word appears in the history) CAN be parallelized using string matching algorithms (suffix arrays, Aho-Corasick), but the overall parse is sequential.

**GPU strategy**:
- For a single long sequence: the parse is sequential. Use CPU or a single GPU thread.
- For MANY sequences in parallel (e.g., computing LZ for 500 tickers x 31 cadences simultaneously): parallelize across sequences. Each GPU thread handles one sequence.
- The batch parallelism is where GPU wins: 15,500 independent LZ computations running in parallel.

```
accumulate(ByKey{sequence_id}, lz_parse, Custom(LZ_state))
```

This is NOT a standard accumulate — the state is complex (the parsed word dictionary). For batch processing of short sequences, this is the right decomposition.

**Kingdom**: B for a single sequence. A (embarrassingly parallel) across independent sequences.

---

## 7. Recurrence Quantification Analysis (RQA)

**Reference**: Zbilut & Webber, "Embeddings and delays as derived from quantification of recurrence plots," Physics Letters A 171:199-203, 1992. Extended by Marwan et al. (2007).

### 7.1 Recurrence Matrix

Given N embedded vectors y(1), ..., y(N), the recurrence matrix is:

```
R[i,j] = Theta(epsilon - ||y(i) - y(j)||)
```

where Theta is the Heaviside step function: Theta(x) = 1 if x >= 0, else 0.

R is symmetric (R[i,j] = R[j,i]) and has R[i,i] = 1 always.

### 7.2 Choosing epsilon

**Method 1**: Fixed recurrence rate. Choose epsilon such that RR (see below) equals a target value (typically 1-5%).

**Method 2**: Fixed fraction of attractor diameter. epsilon = f * max_{i,j} ||y(i) - y(j)|| where f is typically 0.05-0.10.

**Method 3**: From the distance distribution. epsilon = percentile of the pairwise distance distribution (e.g., 10th percentile).

### 7.3 RQA Metrics

All metrics are computed from the structure of the recurrence matrix R.

#### 7.3.1 Recurrence Rate (RR)

Fraction of recurrence points:

```
RR = (1/N^2) * SUM_{i,j} R[i,j]
```

or excluding the main diagonal:

```
RR = (1/(N*(N-1))) * SUM_{i!=j} R[i,j]
```

#### 7.3.2 Diagonal Line Structures

A **diagonal line** of length l in R is a consecutive sequence of R[i,j] = 1 along a diagonal (i+k, j+k) for k = 0, 1, ..., l-1.

The diagonal line length distribution P(l) counts the number of diagonal lines of each length l.

**Minimum line length** l_min (typically 2): lines shorter than l_min are excluded from statistics.

**Determinism (DET)**:

```
DET = SUM_{l=l_min}^{N} l * P(l) / SUM_{l=1}^{N} l * P(l)
```

Fraction of recurrence points forming diagonal lines. High DET -> deterministic dynamics. Low DET -> stochastic.

**Average diagonal line length (L)**:

```
L = SUM_{l=l_min}^{N} l * P(l) / SUM_{l=l_min}^{N} P(l)
```

Mean duration of predictability.

**Longest diagonal line (L_max)**:

```
L_max = max{l : P(l) > 0}
```

Longest predictable segment. Related to the inverse of the largest Lyapunov exponent: L_max ~ 1/lambda_1.

**Entropy of diagonal lines (ENTR)**:

```
ENTR = -SUM_{l=l_min}^{N} p(l) * ln(p(l))
```

where p(l) = P(l) / SUM P(l). Shannon entropy of the line length distribution. Higher -> more complex recurrence structure.

**Divergence (DIV)**:

```
DIV = 1 / L_max
```

Inverse of longest diagonal. Estimates the largest Lyapunov exponent.

#### 7.3.3 Vertical Line Structures

A **vertical line** of length v in R is a consecutive sequence of R[i,j] = 1 along a column (i+k, j) for k = 0, 1, ..., v-1.

The vertical line length distribution P_v(v) counts vertical lines of each length.

**Laminarity (LAM)**:

```
LAM = SUM_{v=v_min}^{N} v * P_v(v) / SUM_{v=1}^{N} v * P_v(v)
```

Fraction of recurrence points forming vertical lines. High LAM -> laminar (slowly changing) states.

**Trapping Time (TT)**:

```
TT = SUM_{v=v_min}^{N} v * P_v(v) / SUM_{v=v_min}^{N} P_v(v)
```

Average length of vertical lines. How long the system "traps" in a state.

**Longest vertical line (V_max)**:

```
V_max = max{v : P_v(v) > 0}
```

#### 7.3.4 Ratio Measures

**Ratio DET/RR**: Distinguishes between random recurrences (low) and deterministic structure (high).

**DET/LAM ratio**: Diagonal vs vertical structure. Indicates transitions.

### 7.4 Cross-Recurrence Analysis

For two time series x(t) and z(t), the cross-recurrence matrix is:

```
CR[i,j] = Theta(epsilon - ||y_x(i) - y_z(j)||)
```

where y_x and y_z are the embeddings of x and z respectively. CR is NOT symmetric.

Same metrics (RR, DET, L, etc.) are computed from CR.

### 7.5 Inputs, Outputs, Edge Cases

**Inputs**:
- x(t), length N
- m, tau: embedding parameters
- epsilon: recurrence threshold
- l_min: minimum diagonal line length (default: 2)
- v_min: minimum vertical line length (default: 2)
- Metric: L2, L-infinity, or L1 norm for distance

**Outputs**:
- RR, DET, L, L_max, ENTR, LAM, TT, V_max, DIV
- R[i,j]: the recurrence matrix (N x N boolean, optional — may be too large)
- P(l), P_v(v): line length distributions

**Edge cases**:
- epsilon = 0: R = I (identity). RR = 1/N, DET = 0 (no diagonal lines).
- epsilon -> infinity: R = all ones. RR = 1, DET = 1, L = N.
- Periodic signal: regular diagonal line pattern. DET ~ 1, L = period.
- Random signal: sparse R with few/short diagonal lines. DET -> 0.
- N very large: R is N x N — memory issue. Subsample or use sparse representation.

**Failure modes**:
- epsilon too small: nearly empty R, metrics undefined or noisy
- epsilon too large: R nearly full, all metrics trivial
- Non-stationarity: R shows visible block structure (which can itself be analyzed)

### 7.6 GPU Decomposition

**Phase 1 — Distance matrix**: `accumulate(Tiled(N,N), (a_k-b_k)^2, Add)` -> shared with Lyapunov, correlation dimension.

**Phase 2 — Thresholding**: Element-wise: R[i,j] = (d[i,j] < epsilon^2). Trivially parallel.

**Phase 3 — RR**: `accumulate(All, R[i,j], Add) / N^2`. One reduction. Kingdom A.

**Phase 4 — Diagonal line detection**: This is the Kingdom B part. Scan along each diagonal of R:

For diagonal offset k, the diagonal elements are R[i, i+k] for i = 0, ..., N-1-k. A diagonal scan detects runs of consecutive 1's.

```
For each diagonal k:
    line_lengths = segmented_scan(R[i,i+k], Counting, boundaries_at_zeros)
```

This is `accumulate(Segmented(zero_boundaries), 1, Add)` per diagonal — counting consecutive ones, resetting at each zero.

There are 2N-1 diagonals, each of length up to N. These are independent -> parallel across diagonals.

**Phase 5 — Vertical line detection**: Same as diagonal but along columns.

```
For each column j:
    v_lengths = segmented_scan(R[i,j], Counting, boundaries_at_zeros)
```

N columns, each independent -> parallel.

**Phase 6 — Line statistics**: Once we have all line lengths, build histograms and compute DET, L, ENTR, LAM, TT from the histograms. This is scatter_add + entropy (Kingdom A).

**Total**: Phase 1-3 are Kingdom A (shared distance matrix + threshold + reduction). Phases 4-5 are Kingdom B (segmented scan). Phase 6 is Kingdom A (histogram + entropy).

**Memory**: The recurrence matrix R is N x N boolean. For N=10,000: 100MB. For N=50,000: 2.5GB. Consider sparse representation or streaming diagonal extraction.

---

## 8. Fractal Dimension

### 8.1 Box-Counting Dimension (Hausdorff Dimension Estimate)

**Reference**: Classic — Mandelbrot, "The Fractal Geometry of Nature," 1982.

**Definition**: Cover the set with boxes of side length epsilon. Let N(epsilon) be the minimum number of boxes needed. Then:

```
D_box = lim_{epsilon->0} ln(N(epsilon)) / ln(1/epsilon)
```

**Algorithm for time series** (embedded in m dimensions):

**Step 1**: Embed the time series to get points in R^m.

**Step 2**: For each scale epsilon (logarithmically spaced):
- Divide R^m into a grid of hypercubes with side length epsilon
- Count the number of non-empty boxes: N(epsilon)

**Step 3**: Fit ln(N(epsilon)) vs ln(1/epsilon). Slope = D_box.

**Grid box counting**: For a point y = (y_1, ..., y_m), its box index is:

```
box(y) = (floor(y_1/epsilon), floor(y_2/epsilon), ..., floor(y_m/epsilon))
```

N(epsilon) = number of unique box indices.

**GPU decomposition**:
- Compute box indices for all points (element-wise). For each epsilon, this is a quantization + unique count.
- `scatter(ByKey{box_hash}, 1, Add)` then count non-zero bins.
- For m > 3: the number of possible boxes explodes. Use hash-based counting (hash the m-dimensional box index to a manageable key space).
- **Kingdom**: A. Box counting is a scatter-add.

**Edge cases**:
- Grid alignment affects results. Standard: use multiple grid offsets and average.
- For very small epsilon: N(epsilon) = N (every point in its own box). For very large epsilon: N(epsilon) = 1.
- Scaling region: similar to correlation dimension — must identify the linear region in the log-log plot.

### 8.2 Higuchi Fractal Dimension (1988)

**Reference**: Higuchi, "Approach to an irregular time series on the basis of the fractal theory," Physica D 31:277-283.

**Algorithm**: Directly from the time series (no embedding needed).

For each scale k = 1, 2, ..., k_max and each starting offset m_offset = 1, 2, ..., k:

**Step 1**: Construct the sub-sampled sequence:

```
X_{m_offset}^k = {x(m_offset), x(m_offset+k), x(m_offset+2k), ..., x(m_offset+floor((N-m_offset)/k)*k)}
```

**Step 2**: Compute the "curve length" of each sub-sampled sequence:

```
L_{m_offset}(k) = [(1/k) * SUM_{i=1}^{floor((N-m_offset)/k)} |x(m_offset + i*k) - x(m_offset + (i-1)*k)|] * [(N-1) / (floor((N-m_offset)/k) * k)]
```

The first factor is the average absolute increment at stride k, divided by k (normalizing for step size). The second factor normalizes for the varying number of points across different m_offset values.

**Step 3**: Average over starting offsets:

```
L(k) = (1/k) * SUM_{m_offset=1}^{k} L_{m_offset}(k)
```

**Step 4**: The fractal dimension is the slope of ln(L(k)) vs ln(1/k):

```
D_H = -slope(ln(L(k)) vs ln(k))
```

(Negative because L(k) DECREASES with increasing k for fractal signals.)

**Interpretation**:
- D_H = 1.0: smooth (differentiable) signal
- D_H = 1.5: Brownian motion
- D_H = 2.0: space-filling (white noise)
- D_H in (1, 2) for fractal time series

**Standard**: k_max = N/4 or k_max in {4, 6, 8, 10, ..., 64}. Must test sensitivity to k_max.

**GPU decomposition**:
- For each (m_offset, k) pair: `gather(strided(m_offset, k), x)` then `accumulate(Contiguous, |x[i]-x[i-1]|, Add)`. This is a strided gather + reduce.
- All (m_offset, k) pairs are independent -> embarrassingly parallel.
- **Kingdom**: A. Each L_{m_offset}(k) is a commutative sum of absolute differences.

### 8.3 Katz Fractal Dimension (1988)

**Reference**: Katz, "Fractals and the analysis of waveforms," Computers in Biology and Medicine 18:145-156.

**Formula**:

```
D_K = log_10(n) / (log_10(n) + log_10(d/L))
```

where:
- n = N - 1 (number of consecutive points — number of steps)
- L = total path length = SUM_{i=1}^{N-1} |x(i+1) - x(i)| (sum of absolute increments)
- d = max_{i} |x(i) - x(1)| (maximum distance from first point — planar diameter of the curve)

**Alternatively** (for time series with equal time spacing):

```
d = max_{i} sqrt((i-1)^2 + (x(i) - x(1))^2)
```

But the common implementation for pure amplitude analysis uses d = max|x(i) - x(1)|.

**Properties**:
- Fast: O(N) computation, no embedding, no log-log regression
- D_K = 1.0 for straight line
- D_K -> 1.5 for Brownian motion
- Less accurate than Higuchi for estimating true fractal dimension
- Sensitive to total path length, which can be dominated by noise

**GPU decomposition**: Two reductions + one max:
```
L = accumulate(Contiguous, |x[i] - x[i-1]|, Add)
d = accumulate(Contiguous, |x[i] - x[0]|, Max)     (or with proper distance formula)
D_K = log(N-1) / (log(N-1) + log(d/L))
```
**Kingdom**: A. Three commutative reductions.

### 8.4 Petrosian Fractal Dimension (1995)

**Reference**: Petrosian, "Kolmogorov complexity of finite sequences and recognition of different preictal EEG patterns," Proceedings IEEE Symposium on Computer-Based Medical Systems, pp. 212-217.

**Formula**:

```
D_P = log_10(N) / (log_10(N) + log_10(N / (N + 0.4 * N_delta)))
```

where:
- N = length of time series
- N_delta = number of sign changes in the first difference: #{i : (x(i+1) - x(i)) * (x(i) - x(i-1)) < 0}

**Properties**:
- Fastest fractal dimension estimate: O(N)
- Approximation to the Katz dimension
- D_P increases with signal complexity
- Less accurate than Higuchi or box-counting
- Primarily used in EEG analysis for its speed

**GPU decomposition**:
```
diff[i] = x[i+1] - x[i]                          (element-wise)
sign_change[i] = (diff[i] * diff[i-1] < 0) ? 1 : 0  (element-wise)
N_delta = accumulate(Contiguous, sign_change[i], Add)    (one reduction)
```
**Kingdom**: A. Element-wise + one reduction.

---

## 9. Hurst Exponent

### 9.1 Rescaled Range (R/S) Analysis (Hurst, 1951)

**Reference**: Hurst, "Long-term storage capacity of reservoirs," Transactions of the American Society of Civil Engineers 116:770-808.

**Algorithm**:

For each block size s in {s_min, ..., s_max}:

**Step 1**: Divide the series into floor(N/s) non-overlapping blocks of size s.

**Step 2**: For each block b = 1, ..., floor(N/s):

a. Compute the block mean:
```
mu_b = (1/s) * SUM_{i=1}^{s} x((b-1)*s + i)
```

b. Compute the cumulative deviation profile:
```
Y(j) = SUM_{i=1}^{j} (x((b-1)*s + i) - mu_b)    for j = 1, ..., s
```

c. Compute the range:
```
R_b = max_{1<=j<=s} Y(j) - min_{1<=j<=s} Y(j)
```

d. Compute the standard deviation:
```
S_b = sqrt[(1/(s-1)) * SUM_{i=1}^{s} (x((b-1)*s + i) - mu_b)^2]
```

e. Rescaled range:
```
(R/S)_b = R_b / S_b    (if S_b > 0, else skip)
```

**Step 3**: Average over blocks:
```
<R/S>(s) = (1/n_valid) * SUM_{b : S_b > 0} (R/S)_b
```

**Step 4**: The Hurst exponent is the slope of log-log regression:
```
log(<R/S>(s)) = H * log(s) + c
```

**Interpretation**:
- H = 0.5: random walk / uncorrelated
- H > 0.5: persistent (positive long-range correlations, trends continue)
- H < 0.5: anti-persistent (negative long-range correlations, mean-reverting)
- H = 1: ballistic motion (perfectly correlated)

**Standard choices**:
- s_min = 10 (need enough points for meaningful range)
- s_max = N/4 (need enough blocks for averaging)
- Block sizes: geometric progression (e.g., s_k = s_min * 1.5^k)
- Typically 10-20 block sizes

**Edge cases**:
- S_b = 0: constant block -> skip (R/S undefined)
- All blocks have S_b = 0: return NaN
- Very short series (N < 20): not enough scale range
- H > 1 or H < 0: indicates the model doesn't fit -> non-stationarity or insufficient data

### 9.2 Detrended Fluctuation Analysis (DFA) — Peng et al., 1994

**Reference**: Peng, Buldyrev, Havlin, Simons, Stanley & Goldberger, "Mosaic organization of DNA nucleotides," Physical Review E 49:1685-1689.

**Algorithm**:

**Step 1**: Compute the cumulative profile:
```
Y(i) = SUM_{k=1}^{i} (x(k) - mu)    for i = 1, ..., N
```
where mu = (1/N)*SUM(x(k)).

**Step 2**: For each window size s:

a. Divide Y into floor(N/s) non-overlapping segments. To use ALL data, also do this from the END (giving 2*floor(N/s) segments total).

b. In each segment v, fit a polynomial trend y_v(i) of order p (DFA-p):
- DFA-1: linear trend (p=1)
- DFA-2: quadratic trend (p=2)
- DFA-q: polynomial of degree q

The fit minimizes SUM(Y(i) - y_v(i))^2 within the segment.

c. Compute the fluctuation (RMS of residuals) for segment v:
```
F^2(v,s) = (1/s) * SUM_{i=1}^{s} [Y((v-1)*s + i) - y_v(i)]^2
```

d. Average over all segments:
```
F(s) = sqrt[(1/(2*floor(N/s))) * SUM_{v=1}^{2*floor(N/s)} F^2(v,s)]
```

**Step 3**: The DFA scaling exponent alpha is the slope of log F(s) vs log s:

```
log F(s) = alpha * log(s) + c
```

**Relationship to Hurst exponent**:
- For stationary signals: alpha = H (Hurst exponent)
- For non-stationary signals: alpha != H in general. alpha characterizes the INTEGRATED process.
- For fractional Gaussian noise (fGn): alpha = H
- For fractional Brownian motion (fBm): alpha = H + 1

**DFA interpretation**:
- alpha = 0.5: uncorrelated (white noise)
- alpha = 1.0: 1/f noise (pink noise)
- alpha = 1.5: Brownian motion
- 0.5 < alpha < 1.0: persistent long-range correlated (stationary)
- 0 < alpha < 0.5: anti-persistent (stationary)
- alpha > 1.0: non-stationary, strongly correlated

**Why DFA beats R/S**: DFA can handle non-stationarity (the detrending removes polynomial trends of order p). R/S is biased by trends.

**GPU decomposition**:
- **Step 1**: Prefix sum -> `accumulate(Prefix(forward), x(i)-mu, Add)`. This is Kingdom B (scan).
- **Step 2**: For each window size s, per-segment polynomial regression. The regression within each segment is `accumulate(Segmented(segment_boundaries), Vandermonde_terms, Add)` -> Kingdom A. All segments at a given s are independent.
- Different window sizes are independent -> parallel across sizes.
- **Step 3**: Log-log regression. Kingdom A.

### 9.3 MFDFA — Multifractal DFA (Kantelhardt et al., 2002)

**Reference**: Kantelhardt, Zschiegner, Koscielny-Bunde, Buldyrev, Havlin & Stanley, "Multifractal detrended fluctuation analysis of nonstationary time series," Physica A 316:87-114.

**Extension of DFA**: Instead of the simple RMS (q=2), compute the q-th order fluctuation function:

```
F_q(s) = [(1/(2*N_s)) * SUM_{v=1}^{2*N_s} (F^2(v,s))^{q/2}]^{1/q}    for q != 0
```

For q = 0 (logarithmic average):
```
F_0(s) = exp[(1/(4*N_s)) * SUM_{v=1}^{2*N_s} ln(F^2(v,s))]
```

**The multifractal spectrum**:

For each q, the generalized Hurst exponent h(q) is:

```
F_q(s) ~ s^{h(q)}
```

(log-log slope of F_q(s) vs s)

The Renyi exponent (mass exponent):
```
tau(q) = q * h(q) - 1
```

The singularity (Holder) exponent:
```
alpha_holder = d(tau)/d(q) = h(q) + q * h'(q)
```

The multifractal spectrum (singularity spectrum):
```
f(alpha_holder) = q * alpha_holder - tau(q) = q * (alpha_holder - h(q)) + 1
```

**Interpretation**:
- If h(q) is constant for all q: monofractal (single scaling exponent). Classical DFA suffices.
- If h(q) varies with q: multifractal. Different moments scale differently.
- Width of f(alpha_holder) spectrum: degree of multifractality. Wider = more multifractal.
- h(2) = standard DFA exponent alpha.
- h(q) for q > 0: influenced by large fluctuations.
- h(q) for q < 0: influenced by small fluctuations (segments with small variance).

**Standard parameters**:
- q range: typically q in {-5, -4, -3, ..., 0, ..., 3, 4, 5} or q in [-10, 10]
- Same window sizes as DFA

**CRITICAL edge case for q < 0**: If any F^2(v,s) = 0, then (F^2(v,s))^{q/2} is infinite (division by zero). This happens when a segment is perfectly linear (detrending removes all variation). **Fix**: skip segments with F^2(v,s) < epsilon_machine, or use a floor.

**GPU decomposition**:
- Same as DFA, except Step 2d becomes: for each q, compute (F^2(v,s))^{q/2} (element-wise power), then reduce. Different q values are independent -> parallel.
- **Kingdom**: A (per-segment regressions) + parallel across (q, s) pairs.

---

## 10. Symbolic Dynamics

### 10.1 Symbolization Strategies

Convert continuous time series to a symbolic sequence S = s(1)s(2)...s(N) over a finite alphabet A = {0, 1, ..., k-1}.

**Strategy 1: Fixed thresholds**

```
s(t) = j    if c_j <= x(t) < c_{j+1}
```

where c_0 < c_1 < ... < c_k are threshold values. Common: c = quantiles of x.

**Strategy 2: Equiprobable bins**

Choose thresholds at the (1/k, 2/k, ..., (k-1)/k) quantiles. Each symbol has approximately equal probability. This maximizes the entropy of the marginal distribution, isolating temporal structure from amplitude effects.

**Strategy 3: Slope-based (derivative)**

```
s(t) = 0    if x(t+1) - x(t) < -delta    (decrease)
s(t) = 1    if |x(t+1) - x(t)| <= delta    (flat)
s(t) = 2    if x(t+1) - x(t) > delta     (increase)
```

Captures directional change rather than amplitude.

**Strategy 4: Ordinal patterns** (same as permutation entropy — Section 5)

### 10.2 Transfer Entropy from Symbols

Given symbolic sequences S_X and S_Y, transfer entropy is:

```
TE_{X->Y} = SUM p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) * ln [p(y_{t+1}|y_t^{(k)}, x_t^{(l)}) / p(y_{t+1}|y_t^{(k)})]
```

where y_t^{(k)} = (y_t, y_{t-1}, ..., y_{t-k+1}) is the k-history of Y, and x_t^{(l)} is the l-history of X.

This reduces to joint histogram counts over the 4-tuple (y_{t+1}, y_t^{(k)}, x_t^{(l)}) — identical to the transfer entropy in Family 25, but with pre-symbolized data.

**GPU decomposition**: Same as Family 25 transfer entropy. Symbolization simplifies the histogram step because keys are already discrete integers.

### 10.3 Forbidden Patterns

For a deterministic system of complexity d, the number of distinct length-L words grows polynomially, not exponentially. The **topological entropy** is:

```
h_top = lim_{L->infinity} (1/L) * ln |W_L|
```

where |W_L| is the number of distinct words of length L observed.

For completely random data: h_top = ln(k) (all k^L words occur).

**Forbidden words**: words of length L that never appear in the symbolic sequence. The fraction of forbidden words is:

```
f_forbidden(L) = 1 - |W_L| / k^L
```

For deterministic systems: f_forbidden > 0 for sufficiently large L (the dynamics constrains which transitions are possible). This is closely related to permutation entropy's forbidden patterns (Section 5.7).

### 10.4 GPU Decomposition

Symbolization: element-wise bin assignment -> parallel.
Word counting: hash the L-gram -> scatter_add -> exactly the same as n-gram histogram in NLP.
Transfer entropy: joint histogram -> entropy combination (Section 10.2, same as Family 25).

**Kingdom**: A. All operations are histograms (scatter_add) + entropy extractions (reduction).

---

## 11. Cross-Algorithm Sharing Surface

### 11.1 The MSR for Family 26

The Minimum Sufficient Representation for the chaos/complexity family is:

**Level 1 (expensive, O(N) or O(N^2))**:

| MSR Type | Contents | Cost | Used by |
|----------|----------|------|---------|
| `EmbeddedVectors(m,tau)` | N_embed x m matrix | O(N*m) — just gather | ALL algorithms below |
| `DistanceMatrix(m,tau,L2)` | N_embed x N_embed pairwise L2 distances | O(N^2*m) | Lyapunov, Correlation dim, RQA |
| `DistanceMatrix(m,tau,Linf)` | Same but L-infinity metric | O(N^2*m) | Sample entropy, Approx entropy |
| `NearestNeighborIndex(m,tau)` | argmin per row (with Theiler exclusion) | O(N^2*m) or from DistanceMatrix | Rosenstein Lyapunov, FNN |
| `RecurrenceMatrix(m,tau,epsilon)` | N x N boolean | From DistanceMatrix + threshold | RQA, Kantz neighborhoods |
| `OrdinalPatternHistogram(m,tau)` | m!-bin histogram | O(N*m) | Permutation entropy |
| `BinarySequence(threshold)` | N-bit vector | O(N) | Lempel-Ziv, symbolic dynamics |
| `CumulativeProfile` | prefix sum of (x-mu) | O(N) | DFA, MFDFA |

**Level 2 (cheaper, derived)**:

| Derived | Source | Cost | Used by |
|---------|--------|------|---------|
| `CorrelationIntegral(r)` | DistanceMatrix | O(N^2) per r | Correlation dimension |
| `DiagonalLineDistribution(epsilon)` | RecurrenceMatrix | O(N^2) segmented scan | RQA (DET, L, ENTR) |
| `VerticalLineDistribution(epsilon)` | RecurrenceMatrix | O(N^2) segmented scan | RQA (LAM, TT) |
| `DivergenceCurve(m,tau)` | NearestNeighborIndex + EmbeddedVectors | O(N*dn_max) | Rosenstein/Kantz Lyapunov |
| `FluctuationFunction(s,q)` | CumulativeProfile | O(N) per s | DFA, MFDFA, Hurst |
| `SegmentRegressions(s,p)` | CumulativeProfile | O(N) per s | DFA |

### 11.2 Sharing Opportunities

When the signal farm runs ALL of Family 26 on the same bin:

**One distance matrix serves**: Lyapunov (Rosenstein and Kantz), correlation dimension, and RQA. Cost: O(N^2*m) GPU time. Without sharing: 3x this cost.

**One embedded vector set serves**: everything. Cost: O(N*m) gather. Without sharing: 6+ repeated embeddings.

**DFA and R/S share**: both operate on block statistics of the time series. The cumulative profile (prefix sum) is shared.

**Permutation entropy and symbolic dynamics share**: both operate on ordinal patterns. The pattern histogram is shared.

### 11.3 Composability Contract

```toml
[family_26_session]
# On session.prime():
auto_compute = [
    "EmbeddedVectors(m=default, tau=default)",
    "CumulativeProfile",
    "OrdinalPatternHistogram(m=3)",
    "BinarySequence(threshold=median)",
]

# On first demand for any distance-dependent algorithm:
lazy_compute = [
    "DistanceMatrix(m,tau,L2)",    # O(N^2*m) - only when needed
    "DistanceMatrix(m,tau,Linf)",  # computed alongside L2 if both needed
]

# Ordering by cost (cheapest first):
# 1. Petrosian FD, Katz FD                 O(N)
# 2. Permutation entropy                    O(N*m)
# 3. Higuchi FD                             O(N*k_max)
# 4. Lempel-Ziv complexity                  O(N*log(N)) [serial]
# 5. Hurst R/S, DFA, MFDFA                  O(N*log(N)) [parallel across scales]
# 6. Sample entropy, Approx entropy          O(N^2) [pairwise distances]
# 7. Correlation dimension                   O(N^2) [pairwise distances, shared]
# 8. Lyapunov exponents                      O(N^2) [pairwise distances, shared]
# 9. RQA                                     O(N^2) [distance + segmented scan]
# 10. Wolf full spectrum                     O(N^2*iterations) [iterative, Kingdom C]
```

---

## 12. Numerical Stability

### 12.1 Log-Log Regression Quality

Many algorithms (correlation dimension, Hurst, DFA, Higuchi, Lyapunov) depend on the slope of a log-log regression. The quality of this regression determines the reliability of the output.

**Always report R^2** (coefficient of determination) alongside the main estimate. If R^2 < 0.9, flag the result as unreliable.

**Scaling region selection**: The most common source of error. Automated methods:
- Take the region where local slope (computed via moving window of 3-5 points) has the smallest variance
- Exclude the first and last 10% of the log-log range (boundary effects)
- Report the scaling region bounds alongside the estimate

### 12.2 Distance Matrix Precision

For N = 10,000: the distance matrix has 50 million entries. In f32, the L2 distance computation `||a-b||^2 = SUM(a_k-b_k)^2` can suffer from catastrophic cancellation if points are far from the origin.

**Fix**: Use the identity `||a-b||^2 = ||a||^2 + ||b||^2 - 2*<a,b>` and precompute norms. This is numerically equivalent but allows sharing norms across all pairs. For f32 with m <= 10 and typical financial returns (magnitude ~ 0.01), this is fine.

### 12.3 Entropy of Small Counts

For permutation entropy with m = 7 (5040 bins) and N = 10,000 patterns: average count per bin ~ 2. Many bins have count 0 or 1. The entropy estimate from such sparse histograms is biased downward.

**Fix**: Apply the same bias corrections as Family 25 (Miller-Madow or James-Stein shrinkage).

### 12.4 DFA Polynomial Regression in Short Segments

For small window sizes (s = 4-8), fitting a polynomial of degree p to s points can be numerically unstable. The Vandermonde matrix is ill-conditioned.

**Fix**: Use centered and scaled x values: x_centered = (i - s/2) / (s/2). Or use orthogonal polynomials (Legendre).

---

## 13. Edge Cases Summary

| Algorithm | Edge Case | Expected Result | Implementation |
|-----------|----------|-----------------|----------------|
| All embedding | N < (m-1)*tau + 2 | NaN | Check before embedding |
| All log-log | < 3 valid points | NaN | Check before regression |
| SampEn | count_B = 0 | NaN | No matches — r too small or N too small |
| SampEn | count_A = 0 | +infinity | Matches at m but not m+1 |
| SampEn | constant signal | 0 | All templates match |
| ApEn | constant signal | 0 | All templates match |
| PE | monotonic signal | 0 | Only one pattern |
| PE | constant signal | 0 | Only one pattern (tie handling required) |
| PE | N < 5*m! | unreliable | Warn, but compute |
| LZ | constant after binarize | 1 word -> LZ_norm ~ 0 | Expected |
| Hurst | S_b = 0 in all blocks | NaN | Skip blocks with zero std |
| DFA | F(v,s) = 0 for all v | NaN (for MFDFA q<0) | Floor at epsilon_machine |
| Corr dim | all distances equal | D_2 undefined | Return NaN |
| Lyapunov | no valid NN found | NaN | Theiler window too large for N |
| RQA | epsilon = 0 | R = I, metrics trivial | Return degenerate values |
| RQA | epsilon -> infinity | R = 1, metrics trivial | Return degenerate values |
| Higuchi | k_max >= N/2 | inaccurate | Clamp k_max to N/4 |
| Box-counting | m > 5 | memory explosion for grid | Use hash-based counting |

---

## 14. Implementation Priority

**Phase 1 — O(N) algorithms** (no distance matrix needed):
1. Permutation entropy (A: scatter_add + entropy — identical infrastructure to Family 25)
2. Higuchi fractal dimension (A: strided gather + reduction)
3. Katz fractal dimension (A: two reductions)
4. Petrosian fractal dimension (A: sign-change counting)
5. Lempel-Ziv complexity (B: sequential parse — batch-parallel across sequences)

**Phase 2 — O(N*log(N)) scaling algorithms**:
6. Hurst R/S (A: per-block statistics + log-log regression)
7. DFA (B scan + A per-segment regression)
8. MFDFA (extends DFA with q-th order — same infrastructure)

**Phase 3 — O(N^2) distance-dependent algorithms** (distance matrix = shared MSR):
9. Sample entropy (A: L-infinity pairwise + count — needs Max-combine TiledOp)
10. Approximate entropy (A: same as SampEn with self-matches)
11. Correlation dimension (A: L2 pairwise + count at multiple r)
12. Multiscale entropy (A: coarse-grain + SampEn at each scale)

**Phase 4 — O(N^2) with post-processing**:
13. Lyapunov exponent (Rosenstein) (A: distance matrix + divergence tracking + regression)
14. Lyapunov exponent (Kantz) (A: distance matrix + neighborhood averaging)
15. RQA (A: distance matrix -> B: segmented scan for line statistics -> A: histogram metrics)

**Phase 5 — Kingdom C / Advanced**:
16. Wolf full Lyapunov spectrum (C: iterative with Gram-Schmidt)
17. Symbolic dynamics / transfer entropy (A: reuse Family 25 infrastructure)
18. FNN for embedding dimension selection (A: distance matrix + next-dimension check)
19. Mutual information for tau selection (A: reuse Family 25 MI)

---

## 15. GPU Kernel Requirements Not Yet in TiledEngine

| Requirement | Used by | Description |
|-------------|---------|-------------|
| **Max-combine TiledOp** | SampEn, ApEn (L-infinity distance) | `accumulate(Tiled, |a_k-b_k|, Max)` instead of Sum |
| **Segmented scan** | RQA line detection | `accumulate(Segmented(zero_boundaries), Count, Add)` |
| **Multi-radius threshold** | Corr dimension | One distance matrix -> 30 threshold counts in one kernel |
| **Divergence tracking** | Lyapunov | Gather from two shifted positions + L2 distance |
| **Batch sequential parse** | Lempel-Ziv | One thread per sequence, 15,000 sequences in parallel |
| **Ordinal pattern computation** | Permutation entropy | Fixed-size sort of m <= 7 elements per position |

Of these, the **Max-combine TiledOp** and **Segmented scan** are the two genuinely new GPU primitives. Everything else composes from existing accumulate + gather.

---

## 16. Relationship to fintek.spec.toml

These algorithms map to the following leaves in the signal farm:

| Leaf (approximate) | Algorithm | Kingdom | cadences |
|---------------------|-----------|---------|----------|
| K02P12C01R01 | Hurst R/S | A | all |
| K02P12C01R02 | DFA | A+B | all |
| K02P12C01R03 | MFDFA | A | all |
| K02P12C02R01 | Higuchi FD | A | all |
| K02P12C02R02 | Katz FD | A | all |
| K02P12C02R03 | Petrosian FD | A | all |
| K02P12C02R04 | Box-counting FD | A | coarse |
| K02P12C03R01 | Correlation dimension | A | coarse |
| K02P12C04R01 | Largest Lyapunov (Rosenstein) | A+B | coarse |
| K02P12C04R02 | Largest Lyapunov (Kantz) | A+B | coarse |
| K02P13C01R01 | Sample entropy | A | all |
| K02P13C01R02 | Approximate entropy | A | all |
| K02P13C01R03 | Multiscale entropy | A | all |
| K02P13C02R01 | Permutation entropy | A | all |
| K02P13C03R01 | Lempel-Ziv complexity | B | all |
| K02P14C01R01 | RQA (full) | A+B | coarse |
| K02P14C02R01 | Symbolic dynamics | A | all |

Note "coarse" cadences for O(N^2) algorithms — these are too expensive for fine cadences (1s, 5s) where bin sizes are small and the algorithms are unreliable anyway.

---

## 17. Open Questions

1. **Should L-infinity distance be a separate `DistanceMatrix` MSR type or a flag on the L2 type?** L-infinity requires a Max-combine TiledOp, so it's a different kernel. But the sharing pattern is the same: one L-infinity matrix serves SampEn and ApEn.

2. **Can SampEn and ApEn avoid materializing the full N x N distance matrix?** A fused kernel that counts matches without storing all distances would reduce memory from O(N^2) to O(1). This is the "count-only" variant of TiledEngine.

3. **Is the Wolf algorithm worth implementing for financial data?** Full Lyapunov spectra are primarily useful for known dynamical systems. For financial data with substantial noise, the single largest exponent (Rosenstein/Kantz) may be all that's reliably estimable.

4. **What embedding parameters should the fintek signal farm use?** Fixed (m=5, tau=1) for production speed, or adaptive (FNN/MI per bin) for accuracy? The V column could carry the FNN-suggested m and whether the production m was sufficient.

5. **Can the DFA prefix sum share infrastructure with other prefix-sum-based algorithms?** The cumulative profile Y(i) = SUM(x(k)-mu) is a prefix scan. If other algorithms need the same prefix sum (e.g., R/S analysis uses cumulative deviations within blocks), they should share.

6. **For MFDFA: is there a closed-form relationship between h(q) and the multifractal spectrum f(alpha)?** The Legendre transform f(alpha) = q*alpha - tau(q) is the standard route. But numerical differentiation of tau(q) is noisy. Should we fit h(q) parametrically and compute f(alpha) analytically?
