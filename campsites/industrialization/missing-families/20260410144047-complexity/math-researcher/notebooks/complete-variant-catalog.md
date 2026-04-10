# Complexity — Complete Variant Catalog

## What Exists (tambear::complexity)

### Entropy-based Complexity (3)
- `sample_entropy(data, m, r)` — SampEn, Richman & Moorman 2000
- `approx_entropy(data, m, r)` — ApEn, Pincus 1991
- `permutation_entropy(data, m, tau)` / `normalized_permutation_entropy`

### Fractal (3)
- `hurst_rs(data)` — R/S analysis Hurst exponent
- `dfa(data, min_box, max_box)` — Detrended Fluctuation Analysis
- `higuchi_fd(data, k_max)` — Higuchi fractal dimension

### Symbolic (1)
- `lempel_ziv_complexity(data)` — LZ76 complexity

### Phase Space / Chaos (4)
- `correlation_dimension(data, m, tau)` — Grassberger-Procaccia
- `largest_lyapunov(data, m, tau, dt)` — Rosenstein algorithm
- `lyapunov_spectrum(data, m, tau, dt, n_exponents)` — full spectrum via QR
- `rqa(data, m, tau, epsilon, lmin)` — recurrence quantification (12 measures)

### Multifractal (1)
- `mfdfa(data, q_values, min_seg, max_seg)` — MFDFA with full Hurst spectrum h(q)

### Causality (1)
- `ccm(x, y, embed_dim, tau, k)` — convergent cross mapping (Sugihara 2012)

### Phase Transition / Criticality (1)
- `phase_transition(levels, q_values, scale)` — Fisher information + MFDFA SOC test

### Singular Value (2)
- `harmonic_r_stat(levels)` — Wigner-Dyson vs Poisson spacing
- `hankel_r_stat(data, embed_dim)` — SVD spacing of Hankel matrix

---

## What's MISSING — Complete Catalog

### A. Missing Entropy-based Complexity Measures

1. **Multiscale entropy** (MSE) — Costa, Goldberger, Peng 2002
   - Coarse-grain at scale τ, compute SampEn at each scale
   - MSE(τ) = SampEn(coarse_grain(data, τ), m, r)
   - Parameters: `data`, `m`, `r`, `max_scale`
   - Key insight: healthy complexity shows high entropy across scales
   - Primitives: coarse_graining (moving average + subsample) → sample_entropy

2. **Composite multiscale entropy** (CMSE) — Wu et al. 2013
   - Multiple coarse-grainings at each scale, average SampEn
   - More stable than MSE for short time series
   - Parameters: same as MSE

3. **Refined composite multiscale entropy** (RCMSE) — Wu et al. 2014
   - Further improvement: compute template matching across all coarse-grained series
   - Even more stable for short series

4. **Fuzzy entropy** — Chen et al. 2007
   - Replaces Heaviside matching with fuzzy membership function
   - FuzzyEn(m, r, n) uses exp(-(d/r)^n) instead of θ(r-d)
   - Parameters: `data`, `m`, `r`, `n` (fuzzy exponent, default 2)
   - More robust to noise and short series than SampEn

5. **Distribution entropy** — Li et al. 2015
   - Shannon entropy of the empirical distance distribution
   - DistEn = -Σ pᵢ log pᵢ where pᵢ = histogram of pairwise distances
   - Parameters: `data`, `m`, `tau`, `n_bins`
   - Advantage: single-scale, no tolerance parameter

6. **Dispersion entropy** — Rostaghi & Azami 2016
   - Map to classes via NCDF, compute Shannon entropy of dispersion patterns
   - Parameters: `data`, `m`, `c` (number of classes)
   - Faster than SampEn, O(n) vs O(n²)

7. **Bubble entropy** — Manis et al. 2017
   - Count of swaps in bubble sort of ordinal patterns
   - Parameter-free except embedding dimension
   - Parameters: `data`, `m`
   - Advantage: no threshold r to tune

8. **Slope entropy** — Cuesta-Frau 2019
   - Based on slope patterns rather than ordinal patterns
   - Differentiates equal-value segments
   - Parameters: `data`, `m`, `delta` (slope threshold)

9. **Increment entropy** — Liu et al. 2016
   - Entropy of increment sequence (first differences) patterns
   - Parameters: `data`, `m`, `tau`

10. **Spectral entropy** — already partially in time_series.rs as `spectral_flatness`
    - H_spec = -Σ p(f) log p(f) where p(f) = PSD(f) / Σ PSD
    - Should be a first-class complexity primitive

11. **Diversity entropy** — Pham 2017
    - Based on cosine similarity of delay vectors
    - Parameters: `data`, `m`, `tau`

### B. Missing Fractal / Self-Similarity Measures

1. **Multifractal DFA** — ✓ exists as `mfdfa`

2. **Katz fractal dimension** — Katz 1988
   - FD = log(L) / log(d) where L = path length, d = max excursion
   - Parameters: `data`
   - Simpler than Higuchi, single-pass

3. **Petrosian fractal dimension** — Petrosian 1995
   - FD = log(N) / (log(N) + log(N/(N + 0.4 N_δ)))
   - N_δ = number of sign changes in first differences
   - Parameters: `data`
   - Very fast, O(n)

4. **Sevcik fractal dimension** — Sevcik 1998
   - Normalized curve length in unit square
   - Parameters: `data`

5. **Box-counting dimension** — fundamental fractal dimension
   - D = lim_{ε→0} log N(ε) / log(1/ε)
   - Parameters: `data` (point cloud), `min_box`, `max_box`, `n_scales`
   - Primitives: grid binning at multiple scales → OLS log-log fit

6. **Information dimension** — D₁ = lim_{ε→0} Σ pᵢ log pᵢ / log ε
   - Parameters: similar to box-counting
   - D₁ ≤ D₀ (box-counting) always

7. **Generalized dimensions** D_q — Rényi spectrum
   - D_q = (1/(q-1)) lim_{ε→0} log(Σ pᵢ^q) / log ε
   - q=0: box-counting, q=1: information, q=2: correlation
   - Parameters: `data`, `q_values`, `scales`
   - Shares: box-counting grid at multiple scales

8. **Wavelet leaders multifractal** — Jaffard et al. 2006
   - More robust than MFDFA for short series
   - Uses DWT leaders instead of DFA fluctuations
   - Parameters: `data`, `q_values`, `wavelet` (default: Daubechies)

9. **Multifractal cross-correlation analysis** (MF-DCCA) — Zhou 2008
   - Cross-correlations between two multifractal series
   - Parameters: `x`, `y`, `q_values`, `min_seg`, `max_seg`
   - Extension of MFDFA to bivariate case

### C. Missing Recurrence-Based Measures

1. **Cross-recurrence quantification** (CRQA) — Zbilut et al. 1998
   - RQA between two time series
   - Parameters: `x`, `y`, `m`, `tau`, `epsilon`, `lmin`
   - Shares: delay embedding, distance computation

2. **Joint recurrence quantification** (JRQA)
   - Element-wise AND of two recurrence matrices
   - Detects simultaneous recurrences
   - Parameters: same as CRQA

3. **Recurrence network analysis** — Marwan et al. 2009
   - Interpret recurrence matrix as adjacency → graph metrics
   - Clustering coefficient, transitivity, average path length
   - Parameters: `data`, `m`, `tau`, `epsilon`
   - Shares: recurrence matrix from RQA

4. **Windowed RQA** — time-varying RQA
   - Sliding window → RQA metrics over time
   - Parameters: `data`, `m`, `tau`, `epsilon`, `lmin`, `window_size`, `step`

### D. Missing Symbolic Dynamics

1. **Symbolic transfer entropy** — Staniek & Lehnertz 2008
   - Transfer entropy using ordinal symbol sequences
   - Parameters: `x`, `y`, `m`, `tau`
   - More robust than bin-based TE

2. **Forbidden patterns** — Amigó et al. 2007
   - Count of ordinal patterns that never appear
   - Related to topological entropy
   - Parameters: `data`, `m`, `tau`
   - Theory: deterministic systems have forbidden patterns; stochastic don't

3. **Compression complexity** (ETC) — Nagaraj et al. 2013
   - Effort-to-compress: iterative substitution until constant
   - Parameters: `data`, `n_symbols`
   - Related to Kolmogorov complexity

4. **Block entropy** — Shannon entropy of blocks of length L
   - H(L) = -Σ p(b_L) log p(b_L)
   - entropy rate: h = lim_{L→∞} H(L)/L
   - Parameters: `data` (symbolic), `max_block_length`

5. **T-complexity** — Titchener 1998
   - Based on T-decomposition of strings
   - Parameters: `data` (symbolic)

### E. Missing Network / Graph Measures from Time Series

1. **Visibility graph entropy** — already have NVG/HVG degree
   - Missing: clustering coefficient, average path length, degree distribution entropy
   - Parameters: use existing nvg_degree/hvg_degree

2. **Ordinal partition network** — McCullough et al. 2015
   - Transition network between ordinal patterns
   - Network entropy, betweenness, clustering
   - Parameters: `data`, `m`, `tau`

3. **Horizontal visibility graph Lyapunov** — Luque et al. 2009
   - λ_HVG from degree distribution
   - For i.i.d.: P(k) = (1/3)(2/3)^{k-2}, λ_HVG = ln(3/2)

### F. Missing Phase Space Measures

1. **False nearest neighbors** (FNN) — Kennel et al. 1992
   - Determine optimal embedding dimension
   - Parameters: `data`, `tau`, `max_dim`, `rtol`, `atol`
   - Critical for: all delay embedding methods (SampEn, correlation_dim, Lyapunov)

2. **Average mutual information** for time delay — Fraser & Swinney 1986
   - First minimum of MI(τ) gives optimal delay
   - Parameters: `data`, `max_lag`, `n_bins`
   - Uses: mutual_information from information_theory.rs

3. **Kaplan-Glass test** for determinism
   - Tests if trajectory is deterministic vs stochastic
   - Parameters: `data`, `m`, `tau`

4. **Wayland test** for determinism — Wayland et al. 1993
   - E_trans statistic based on translation error
   - Parameters: `data`, `m`, `tau`

5. **0-1 test for chaos** — Gottwald & Melbourne 2004
   - K → 0 for regular, K → 1 for chaotic
   - Parameters: `data`, `c` (test frequency, default: random)
   - Advantage: no phase-space reconstruction needed

---

## Decomposition into Primitives

```
delay_embed(data, m, tau) ────┬── sample_entropy
                              ├── approx_entropy
                              ├── fuzzy_entropy
                              ├── distribution_entropy
                              ├── diversity_entropy
                              ├── correlation_dimension
                              ├── largest_lyapunov
                              ├── lyapunov_spectrum
                              ├── rqa / crqa / jrqa
                              ├── false_nearest_neighbors
                              ├── wayland_test
                              └── kaplan_glass_test

ordinal_pattern(data, m, tau) ┬── permutation_entropy
                              ├── forbidden_patterns
                              ├── symbolic_transfer_entropy
                              ├── ordinal_partition_network
                              └── slope_entropy (variant)

coarse_grain(data, scale) ────── multiscale_entropy (all variants)

sort(data) ───────────────────── hurst_rs (R/S range)

ols_fit(log_x, log_y) ───────┬── dfa
                              ├── higuchi_fd
                              ├── box_counting_dim
                              ├── correlation_dimension
                              └── generalized_dimensions

pairwise_distances(embedded) ─┬── correlation_dimension
                              ├── rqa (recurrence matrix)
                              ├── distribution_entropy
                              └── false_nearest_neighbors
```

## Intermediate Sharing Map

| Intermediate | Computed by | Shared with |
|---|---|---|
| Delay-embedded matrix | `delay_embed` | SampEn, ApEn, FuzzyEn, CorrDim, Lyapunov, RQA, FNN |
| Pairwise distance matrix | distance computation | CorrDim, RQA, DistEn, FNN, kNN-based |
| Ordinal patterns | `ordinal_pattern` | PermEn, forbidden patterns, symbolic TE, ordinal network |
| Coarse-grained series | `coarse_grain` | MSE, CMSE, RCMSE |
| Log-log regression | `ols_slope` | DFA, Higuchi, box-counting, Hurst |
| Recurrence matrix | threshold on distances | RQA, CRQA, JRQA, recurrence network |

## Priority

**Tier 1** — Should exist now:
1. `false_nearest_neighbors` — needed to validate ALL embedding-based methods
2. `multiscale_entropy` — most cited complexity measure after SampEn
3. `fuzzy_entropy` — better than SampEn in nearly all settings
4. `katz_fd` / `petrosian_fd` — fast, simple, complementary to Higuchi
5. `zero_one_test_chaos` — no embedding needed, very useful diagnostic

**Tier 2** — High value:
6. `dispersion_entropy` — O(n), parameter-light
7. `bubble_entropy` — parameter-free
8. `cross_rqa` — bivariate complexity
9. `mf_dcca` — bivariate multifractal
10. `forbidden_patterns` — determinism diagnostic

**Tier 3** — Specialist:
11-20: wavelet leaders, network entropy, compression complexity, etc.
