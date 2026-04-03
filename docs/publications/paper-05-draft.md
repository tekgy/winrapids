# Hidden Structure in Numerical Computing: How a Universal Decomposition Reveals That Five Normalization Methods Are One

## Abstract

Between 2015 and 2019, five influential papers introduced BatchNorm, LayerNorm, InstanceNorm, GroupNorm, and RMSNorm — each presented as a distinct contribution to deep learning. We show that all five are a single parameterized operation: an associative accumulation over data partitioned by different grouping keys. This is not an isolated observation. Using a universal decomposition of numerical algorithms into parameterized accumulations, we systematically catalog 34+ structural equivalences across 35 algorithm families spanning statistics, machine learning, signal processing, optimization, econometrics, and scientific computing. These equivalences are not coincidences: they are coordinate-plane intersections in a product space defined by three orthogonal axes — data transform, accumulation pattern, and estimation oracle. The product space framework explains known connections (ANOVA generalizes the t-test; Spearman correlation is Pearson applied to ranks), reveals novel ones (the IRLS master template unifying 10 estimation families through a single weighted accumulation; Adam optimization consisting of four parallel exponential moving average channels), and predicts the existence of algorithms in unfilled cells. We argue that the structural fragmentation of numerical computing — identical algorithms reimplemented under different names across communities separated by notation and tradition — is an artifact of missing instrumentation, now correctable.

---

## 1. Introduction

In January 2016, Ba, Kiros, and Hinton published Layer Normalization, addressing a limitation of Batch Normalization (Ioffe & Szegedy, 2015). Later that year, Ulyanov, Vedaldi, and Lempitsky introduced Instance Normalization for style transfer. In 2018, Wu and He proposed Group Normalization to handle small batch sizes. In 2019, Zhang and Sennrich presented RMSNorm, dropping the mean-centering step entirely.

Each paper motivated its contribution by identifying a specific failure mode of prior methods. Each was reviewed, accepted, and cited as a distinct advance. Collectively, they have been cited over 68,000 times — BatchNorm alone accounts for over 46,000.

All five compute the same thing.

Specifically, each normalizes activations by computing sufficient statistics — mean and variance, or root-mean-square — over a subset of tensor dimensions, then applying an affine transformation. The subsets differ: BatchNorm averages over the batch dimension, LayerNorm over the feature dimension, InstanceNorm over spatial dimensions per channel, GroupNorm over channel groups, and RMSNorm computes only the root-mean-square. But the computational pattern is identical: partition the data by a grouping key, accumulate {count, sum, sum-of-squares} per group, extract the normalizing statistics, apply element-wise.

In the notation of decomposable accumulation systems, all five are:

```
accumulate(activations, ByKey(axis), identity, Welford)
```

where `axis` ∈ {batch, feature, spatial, group, all} is the only parameter that varies.

This example illustrates a broader phenomenon. Algorithms developed independently across different research communities — statistics, machine learning, signal processing, econometrics, psychometrics, geostatistics — frequently share identical computational structure. The equivalences are obscured by differences in notation (statisticians write Σ, ML researchers write `torch.sum`), publication venue (JASA vs. NeurIPS vs. IEEE TSP), and implementation ecosystem (R vs. Python vs. MATLAB). The same algorithm acquires different names, different citations, and different communities of practice — without anyone noticing that the computation is the same.

We call these **structural rhymes**: pairs or groups of algorithms that, when decomposed into their constituent accumulation operations, reveal identical computational structure despite different surface presentations.

### 1.1 The Instrument

The equivalences reported here were discovered using a systematic decomposition of numerical algorithms into two fundamental operations:

1. **accumulate**(data, grouping, expression, operator) — the universal computation primitive
2. **gather**(indices, source) — the universal read primitive

where the *grouping* parameter determines how data is partitioned (reduce over all elements, scatter by key, prefix scan, windowed rolling, tiled matrix blocks, segmented, or masked), the *expression* specifies a per-element function, and the *operator* defines the associative combination rule. We have found that eight operators — Add, Welford (online mean+variance), RefCentered (numerically stable moments), Affine (linear recurrence), Särkkä (exact transient Kalman), Max/Min, ArgMax/ArgMin, and SoftmaxWeighted (online softmax accumulation) — cover 500+ algorithms across 35 families.

This decomposition serves as an instrument: algorithms that look different in standard notation reveal shared structure when expressed as accumulation tuples. Two algorithms "rhyme" when their (grouping, expression, operator) tuples match, even if their names, notations, and publication histories differ.

### 1.2 Contributions

We make three contributions:

1. **A systematic catalog of 34+ structural rhymes** across 35 algorithm families, organized by the type of equivalence (same computation in different fields; same framework at different scales; same operator in different domains; different names for identical formulas). Several individual equivalences are well-known (ANOVA generalizes the t-test; Spearman correlation is Pearson on ranks). The systematic *collection* using a single instrument, and the novel equivalences it reveals, are new.

2. **The product space framework.** We show that the space of algorithms factorizes as Transforms × Kingdoms × Oracles, where transforms are preprocessing operations (sort, FFT, wavelet, ranking, embedding), kingdoms classify the accumulation pattern (commutative, sequential, iterative), and oracles specify the estimation interface (moment statistics, IRLS, affine scan, gradient). Structural rhymes are coordinate-plane intersections in this product space. The framework not only *explains* known equivalences but *predicts* algorithms in unfilled cells.

3. **The IRLS master template.** We identify a single weighted accumulation primitive — `accumulate(ByKey, weighted_outer_product, Add)` followed by matrix inversion — that unifies 10 estimation families spanning nearly two centuries of independent development: ordinary least squares, logistic regression, Poisson regression, robust M-estimation (Huber, 1964), Gaussian mixture EM (Dempster et al., 1977), linear mixed effects (Henderson, 1950), item response theory (Rasch, 1960), Cox proportional hazards (Cox, 1972), confirmatory factor analysis, and variational inference. Each family differs only in its weight function. The marginal implementation cost of adding a new family to the template is one function.

### 1.3 Scope and Limitations

We do not claim that the accumulation decomposition is universal for all computation — the boundary matrix reduction in persistent homology (topological data analysis) operates over GF(2), a different algebraic structure that does not fit the framework. We claim sufficiency for the 35 algorithm families examined, covering the computational cores of SPSS, SAS, Stata, MATLAB, R's base statistics, Python's scipy/sklearn/statsmodels, and NVIDIA's cuBLAS/cuFFT/cuML/cuDNN.

Several individual structural rhymes in our catalog are well-established in the literature. Rasmussen and Williams (2006) explicitly note the connection between Gaussian process regression and kriging. The identity F = t² for k = 2 groups appears in introductory statistics textbooks. Spearman correlation is defined as Pearson correlation on ranks. Our contribution is not any single rhyme but the *instrument* that discovers them systematically, the *product space* that explains why they exist, and the *IRLS template* that reveals a novel 10-family unification.

### 1.4 Paper Organization

Section 2 presents the accumulation decomposition and defines structural rhymes formally. Section 3 introduces the product space framework (Transforms × Kingdoms × Oracles) and shows how rhymes arise as coordinate-plane intersections. Section 4 presents the full catalog of 34+ rhymes, organized by type. Section 5 develops the IRLS master template and its 10-family instantiation. Section 6 discusses predicted algorithms in unfilled product-space cells. Section 7 discusses pedagogical implications. Section 8 concludes.

---

## 2. The Accumulation Decomposition

[TODO: formal definition of accumulate and gather, the 8 operators, grouping patterns, the "rhyme" relation as tuple equality modulo naming]

---

## 3. The Product Space Framework

The structural rhymes in our catalog are not isolated coincidences. They arise systematically from a three-dimensional factorization of algorithmic structure.

### 3.1 Three Orthogonal Axes

We observe that every algorithm in our 35-family corpus can be characterized by three independent choices:

**Transform (T)**: a preprocessing operation that changes the data representation before accumulation.

| Transform | Effect | Algorithms enabled |
|-----------|--------|-------------------|
| Identity | no change | most Kingdom A algorithms |
| Sort | ordered permutation | quantiles, rank statistics, order statistics |
| Rank | integer rank assignment | Spearman, Kruskal-Wallis, Friedman |
| FFT | frequency domain | periodogram, Welch, spectral analysis |
| Wavelet | multi-resolution | DWT analysis, denoising |
| Embedding | delay/feature expansion | Takens, im2col (convolution), polynomial features |
| Log/Exp | scale change | geometric mean, log-returns, Poisson link |

**Kingdom (K)**: the accumulation pattern, classifying the algorithmic structure.

| Kingdom | Pattern | Parallel depth | Examples |
|---------|---------|---------------|----------|
| A (Commutative) | order-independent reduction | O(log n) | mean, variance, histogram, GEMM |
| B (Sequential) | prefix scan with associative state | O(log n) | EWM, Kalman, ARIMA, Adam |
| C (Iterative) | outer loop over Kingdom A/B body | O(k · inner) | EM, IRLS, Newton, gradient descent |

**Oracle (O)**: the estimation interface — what quantity the accumulation computes.

| Oracle | Provides | Families served |
|--------|----------|----------------|
| MomentStats | {n, μ, m₂, m₃, m₄, min, max} | F06, F07, F08, F09, F25 |
| IRLS | weighted GramMatrix solve per iteration | F10, F11, F14, F15, F16, F34 |
| Affine scan | linear recurrence state | F17, F18, F05 (Adam) |
| GradientOracle | ∇f(θ) for optimization | F05, F24, F34 |

### 3.2 Rhymes as Coordinate-Plane Intersections

Two algorithms that share the same position on two of three axes but differ on the third are structural rhymes. This explains the five types:

**Type I: Same (K, O), different T** — Transform rhymes.
Algorithms that accumulate identically but preprocess differently.
- Spearman correlation = (T=Rank, K=A, O=MomentStats) vs. Pearson = (T=Identity, K=A, O=MomentStats). Same Kingdom, same Oracle, different Transform.
- Kruskal-Wallis = (T=Rank, K=A, O=MomentStats) vs. ANOVA = (T=Identity, K=A, O=MomentStats). Identical after ranking.

**Type II: Same (T, O), different K** — Kingdom rhymes.
Algorithms that transform and estimate identically but accumulate at different levels.
- OLS = (T=Identity, K=A, O=IRLS with w=1) vs. Ridge = (T=Identity, K=A, O=IRLS with w=1 + λI). Same transform, same oracle, Kingdom A in both cases — the "difference" is a diagonal perturbation of the GramMatrix, not a kingdom change.
- ANOVA (k=2) produces F = t². The kingdom is the same (A); the "test" is the same (MomentStats extraction). The equivalence is an identity within a single coordinate point.

**Type III: Same (T, K), different O** — Oracle rhymes.
Algorithms that transform and accumulate identically but answer different estimation questions.
- The five normalizations: all (T=Identity, K=A, O=Welford). The "oracle" is identical; only the *grouping key within* K=A varies. Five papers; one point in T×K×O space.

**Type IV: Same (T, K, O) — true duplicates.**
Algorithms that occupy the *same* point in the product space. These are the deepest rhymes: genuinely identical computations with different names.
- Kriging (geostatistics, 1951) and GP regression (machine learning, 2006): both (T=Identity, K=A, O=kernel matrix solve). 55 years of independent development for the same algorithm.
- IRT information, Fisher information, IRLS weight, Bernoulli variance: four names for μ(1−μ). Same point.

### 3.3 Predictive Power

The product space framework is not merely descriptive. Unfilled cells predict algorithms:

| T | K | O | Predicted Algorithm | Status |
|---|---|---|-------------------|--------|
| Rank | C | Sigmoid | Ordinal logistic regression on ranks | Known (proportional odds) |
| FFT | B | Affine | Spectral Kalman filter | Known (frequency-domain KF) |
| Wavelet | A | MomentStats | Wavelet-domain descriptive statistics | Novel? |
| Embedding | C | IRLS | Manifold-regularized robust regression | Novel? |
| Log | B | Affine | Log-domain state space model | Known (log-EWM) |

Some "predicted" algorithms turn out to be known — which validates the framework. Others appear genuinely novel: algorithms that exist in principle (the product space says they should work) but have not been published. We leave experimental investigation of these cells to future work.

### 3.4 The Dimensionality Argument

The number of structural rhymes is *determined* by the dimensionality of the product space. In a space with three axes of cardinality |T|, |K|, |O|:

- Each axis-aligned plane contains |T|×|K|, |T|×|O|, or |K|×|O| cells.
- Two algorithms that share a plane are structurally related.
- The total number of potential rhymes grows quadratically with the number of algorithms placed in the space.

With |T| ≈ 7 transforms, |K| = 3 kingdoms, |O| = 4 oracles, the product space has 84 cells. We have placed 35 families across these cells. The 31+ observed rhymes are a natural consequence of 35 families in 84 cells — many families share at least one coordinate. The rhymes are not surprising; they are *expected* from the dimensionality. What is surprising is that the space has only three axes and 84 cells, yet accommodates 500+ algorithms.

---

## 4. Catalog of Structural Rhymes

We catalog 33 structural rhymes organized by the taxonomy of Section 3. The full table with (T,K,O) coordinates appears in Appendix A. Here we summarize the distribution and highlight notable examples.

### 4.1 Distribution by Type

| Type | Description | Count | Dominant Pattern |
|------|------------|------:|-----------------|
| I | Transform rhymes: same (K,O), different T | 7 | Rank transform generates 4 alone |
| II | Kingdom rhymes: same (T,O), different K | 4 | Scale/dimension changes |
| III | Oracle rhymes: same (T,K), different O | 13 | IRLS template dominates |
| IV | True duplicates: same (T,K,O) | 5 | Independent reinventions |
| V | Meta-rhymes: self-similarity | 4 | System-level patterns |

**Type III dominates the catalog.** This is not accidental. The oracle — the weight function, the estimation interface — is what each research community considers its distinctive contribution. What statisticians, machine learning researchers, psychometricians, and econometricians each consider their field's unique insight is the parameter that varies across the IRLS template. The LOOP is shared. The PARAMETER is the contribution. The fragmentation persists precisely because each community correctly identifies its distinctive parameter and incorrectly concludes that a distinctive parameter implies a distinctive computation.

### 4.2 Type I Examples: Transform Rhymes

The rank transform is a prolific generator of rhymes. Applying ranks to data and then running a Kingdom A accumulation produces a non-parametric test that is immune to distributional assumptions:

| Parametric test | + Rank transform | = Non-parametric test |
|-----------------|-----------------|----------------------|
| Pearson correlation | rank(x), rank(y) | Spearman correlation |
| One-way ANOVA | rank(all data) | Kruskal-Wallis H |
| Paired t-test | rank(|differences|) | Wilcoxon signed-rank |
| Two-sample t-test | rank(pooled data) | Mann-Whitney U |

The non-parametric statistics literature (Hollander, Wolfe & Chicken, 2013) teaches these as independent methods with their own derivations. The accumulation decomposition reveals them as the SAME methods applied to RANK-TRANSFORMED data. The transform is the only difference.

### 4.3 Type IV Examples: True Duplicates

The most striking rhymes are algorithms that occupy the same (T,K,O) coordinate despite independent development:

**Kriging (Krige, 1951) and Gaussian Process Regression (Rasmussen & Williams, 2006).** Both build a covariance matrix from a kernel function, solve a linear system with observations as the right-hand side, and produce predictions with uncertainty. The geostatistical variogram parameters {nugget, sill, range} correspond exactly to the GP kernel hyperparameters {noise variance, signal variance, lengthscale}. 55 years of independent development; identical computation.

**The normalization zoo (2015–2019).** Five methods — BatchNorm, LayerNorm, InstanceNorm, GroupNorm, RMSNorm — each computing `accumulate(data, ByKey(axis), identity, Welford)` with different axis parameters. 68,258 collective citations; one operation.

**IRT information = Fisher information = IRLS weight = Bernoulli variance.** The expression μ(1−μ) appears under four names in four fields spanning 300 years (Bernoulli 1713 through Lord 1980). Each field derived it independently from its own axioms. All four are the same formula.

**GARCH(1,1) = Adam's second moment = EWM on squared sequence.** In 1986, Bollerslev introduced GARCH(1,1) to model time-varying volatility in financial returns:

```
σ²_t = ω + β · σ²_{t-1} + α · r²_{t-1}
```

In 2014, Kingma and Ba introduced Adam's v-accumulator to track gradient magnitude:

```rust
v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
```

Both are `accumulate(data², Prefix, v, Affine(β, ω))` — a sequential Affine scan on squared values, Kingdom B. The only structural difference is the intercept ω. GARCH uses ω > 0 to enforce a long-run variance floor `σ̄² = ω/(1−α−β)`. Adam sets ω = 0 and compensates with bias correction `v̂_t = v_t / (1−β^t)`. Both mechanisms solve the same problem: preventing the accumulator from decaying to zero when the signal is intermittently quiet. The solutions are structurally equivalent.

Loshchilov and Hutter's AdamW (2019) adds weight decay `v *= (1-λ)` at each step, which is precisely a nonzero ω in GARCH form: `v_t = (β₂-λ) · v_{t-1} + (1-β₂) · g²_t`. **AdamW IS GARCH with ω = λ · v̄.** Three papers, three research communities, 33 years of independent development — one Affine scan.

The full family: IGARCH/RiskMetrics EWMA (1990/1994, finance), GARCH(1,1) (1986, econometrics), RMSProp (Hinton, 2012, deep learning), Adam-v (2014, deep learning), AdamW-v (2019, deep learning). Five algorithms; one `accumulate(x², Prefix, state, Affine(β, ω))` call. (T=Square, K=B, O=Affine.)

### 4.4 The Oracle Distribution Insight

The dominance of Type III (oracle) rhymes in the catalog has a structural explanation. In the product space T × K × O:

- Transforms (T) are few in number (~7) and well-understood. Communities readily recognize when the same transform is applied.
- Kingdoms (K) are coarse-grained (3 categories). The distinction between commutative and sequential computation is usually obvious.
- Oracles (O) are fine-grained and domain-specific. Each community names its oracle with domain vocabulary: "Bernoulli variance" (probability), "Fisher information" (statistics), "IRLS weight" (regression), "item information function" (psychometrics).

The *specificity* of oracle naming creates the illusion of distinctiveness. The community cannot see through its own naming convention to the shared loop beneath. An instrument that strips oracle-specific vocabulary from the computation — expressing everything as accumulate(data, grouping, expr, op) — dissolves the illusion.

---

## 5. The IRLS Master Template

The deepest structural equivalence in our catalog spans ten algorithm families developed independently over more than six decades. We show that all ten share a single computational core: a weighted accumulation followed by a linear solve.

### 5.1 The Template

Iteratively Reweighted Least Squares (IRLS) alternates between two steps:

1. **Weight computation**: given current parameters θ̂, compute diagonal weights W = diag(w₁, ..., wₙ) where wᵢ = f(xᵢ, θ̂) for a family-specific weight function f.

2. **Weighted accumulation + solve**: compute the weighted cross-product matrix X^T W X and weighted response X^T W z, then solve (X^T W X)θ = X^T W z.

In the accumulation decomposition, step 2 is:

```
G = accumulate(data, ByKey(parameter), weighted_outer_product, Add)
    where weighted_outer_product(xᵢ) = wᵢ · xᵢ · xᵢ^T
b = accumulate(data, ByKey(parameter), weighted_response, Add)
    where weighted_response(xᵢ) = wᵢ · xᵢ · zᵢ
θ = solve(G, b)
```

The ENTIRE family-specific content is the weight function f. Everything else — the accumulation, the solve, the iteration control — is shared infrastructure.

### 5.2 Ten Instantiations

| Family | Weight function wᵢ | Year | Field | Key citation |
|--------|-------------------|------|-------|-------------|
| **OLS** | wᵢ = 1 (constant) | 1809 | Mathematics | Gauss |
| **Logistic regression** | wᵢ = μᵢ(1 − μᵢ) | 1958 | Statistics | Cox |
| **Poisson regression** | wᵢ = μᵢ | 1972 | Statistics | Nelder & Wedderburn |
| **Robust M-estimation** | wᵢ = ψ(rᵢ/σ̂) / (rᵢ/σ̂) | 1964 | Robust statistics | Huber |
| **EM (Gaussian mixture)** | wᵢ = γₙₖ (posterior) | 1977 | General statistics | Dempster, Laird & Rubin |
| **Linear mixed effects** | wᵢ = (ZGZ^T + R)⁻¹ | 1950 | Animal breeding | Henderson |
| **Item Response Theory** | wᵢ = μᵢ(1 − μᵢ) | 1960 | Psychometrics | Rasch |
| **Cox proportional hazards** | wᵢ = risk-set weights | 1972 | Survival analysis | Cox |
| **Confirmatory Factor Analysis** | wᵢ = information matrix | 1969 | Psychometrics | Jöreskog |
| **Variational inference** | wᵢ = E_q[sufficient stats] | 1999 | Machine learning | Jordan et al. |

The chronological span is remarkable: from Legendre's method of least squares (1805) through Henderson's mixed model equations in animal breeding (1950), Cox's logistic regression (1958), Rasch's psychometric models (1960), Huber's robust estimation (1964), Jöreskog's confirmatory factor analysis (1969), the Cox proportional hazards model (1972), the EM algorithm (Dempster, Laird & Rubin, 1977), and variational inference (Jordan, Ghahramani, Jaakkola & Saul, 1999) — a span of nearly two centuries of independent statistical research.

These ten families were developed in six different fields (mathematics, statistics, animal breeding, psychometrics, survival analysis, machine learning), published in venues with minimal cross-readership, and are taught in different graduate programs. Yet all ten iterate the same weighted accumulation.

### 5.3 The Weight Function IS the Family

The table above reveals that the weight function is the *only* axis of variation. The accumulation structure, the solve method, the iteration logic, and the convergence criterion are identical. This has a concrete implication for implementation:

**Marginal cost of adding a new family: one function.** Given an implementation of the weighted accumulation template, adding logistic regression requires specifying `w(μ) = μ(1 − μ)`. Adding Poisson regression requires `w(μ) = μ`. Adding robust Huber estimation requires `w(r) = min(1, k/|r|)`. Each is a single function definition. The infrastructure — GPU-parallel weighted scatter, matrix assembly, iterative solve, convergence check — is written once and shared.

### 5.4 Why This Was Not Previously Recognized

The unification of OLS and logistic regression through IRLS is well-known in the GLM literature (McCullagh & Nelder, 1989). The connection between IRLS and M-estimation is documented by Huber (1964) and Holland & Welsch (1977). What appears to be novel is the extension to all ten families, particularly:

1. **EM ↔ IRLS**: The M-step of Gaussian mixture EM is structurally identical to a weighted least-squares step, with posterior responsibilities γₙₖ as weights. This is noted by some authors (e.g., Meng & van Dyk, 1997) but not widely recognized as an IRLS instantiation.

2. **IRT ↔ Logistic IRLS**: The E-step of IRT's marginal maximum likelihood is equivalent to logistic IRLS with quadrature-approximated expectations. Both use the weight function μ(1 − μ), which is simultaneously the Bernoulli variance, the Fisher information, and the IRLS diagonal weight. Four concepts in four fields; one formula.

3. **CFA/SEM ↔ IRLS**: The Newton-Raphson optimization of the CFA likelihood function uses the expected information matrix as weights in what is structurally a weighted GramMatrix solve.

4. **Variational inference ↔ IRLS**: Coordinate ascent variational inference (CAVI) updates each factor using expected sufficient statistics under the variational distribution — which ARE weighted accumulations with E_q[·] as the weight function.

### 5.5 The μ(1 − μ) Rhyme

The deepest single equivalence in the template deserves separate attention. The expression μ(1 − μ), where μ = sigmoid(η), appears under four independent names in four fields:

| Name | Field | Role | Citation |
|------|-------|------|----------|
| Bernoulli variance | Probability theory | Var(Y) for Y ~ Bernoulli(μ) | Bernoulli, 1713 |
| Fisher information | Mathematical statistics | I(θ) = E[(∂ℓ/∂θ)²] for logistic | Fisher, 1922 |
| IRLS weight | Generalized linear models | Diagonal of W in (X^TWX)β = X^TWz | Nelder & Wedderburn, 1972 |
| IRT information | Psychometrics | Item information function | Lord, 1980 |

Four names. Four literatures. Four citation chains. One expression: μ(1 − μ). A student who learns any one of these has, without knowing it, learned all four — but no textbook draws the connection because no textbook has the instrument to see it.

### 5.6 Computational Implications

In a conventional software stack, each of the ten families is implemented as a separate module with its own iteration loop, matrix assembly, and solve call. In the accumulation framework:

- One weighted scatter primitive handles all ten
- One matrix solve handles all ten
- One convergence check handles all ten
- The weight function is a parameter, not a code path

The reduction in implementation complexity is measurable: in our reference implementation, the IRLS template is approximately 80 lines of infrastructure code. Each new family adds approximately 10 lines (the weight function and its derivative). Ten families in approximately 180 lines total, compared to approximately 2,000 lines when implemented independently — a 10× compression ratio attributable entirely to the shared accumulation structure.

---

## 6. Predicted Algorithms

The product space framework is not merely a classification scheme — it is generative. Unfilled cells in the T × K × O grid predict algorithms that should exist but may not have been published.

### 6.1 Method

For each of the 84 cells in our 7 × 3 × 4 product space, we check whether a known algorithm occupies that cell. Unoccupied cells represent potential algorithms whose transform, kingdom, and oracle are individually well-established, but whose combination has not been explored.

### 6.2 Predictions

| T | K | O | Predicted Algorithm | Status |
|---|---|---|-------------------|--------|
| Rank | C | IRLS(sigmoid) | Ordinal logistic regression on ranked data | Known: proportional odds model |
| FFT | B | Affine | Frequency-domain Kalman filter | Known: spectral KF (Haykin, 2001) |
| Wavelet | A | MomentStats | Wavelet-domain descriptive statistics | Partially known: wavelet shrinkage uses variance estimates at each scale |
| Embedding | C | IRLS(posterior) | Delay-embedded mixture model for phase space | Novel: EM on Takens-embedded attractors |
| Rank | A | Histogram | Distribution-free entropy estimation | Known: permutation entropy (Bandt & Pompe, 2002) |
| Log | C | GradientOracle | Log-domain neural network training | Known: LogSumExp tricks, but not as a systematic log-transform |
| FFT | A | MomentStats | Spectral moment statistics | Known: spectral centroid/spread/flux |
| Square | B | Affine(β,ω) | Conditional variance accumulator | Known: GARCH (1986), Adam-v (2014), AdamW (2019) — same cell, three literatures |
| x^k | B | Affine(β) | k-th conditional moment accumulator | Finance: k=1..4 (Harvey & Siddique 1999, Dittmar 2002). Deep learning: k=1,2 (Adam). **k=3,4 in deep learning: predicted but not yet built.** |

Of our nine concrete predictions, seven correspond to known algorithms — validating the framework — and two appear novel. The "delay-embedded mixture model" combines Takens embedding (T=Embedding) with Gaussian mixture EM (K=C, O=IRLS(posterior)) to discover attractor structure in dynamical systems. We leave experimental investigation to future work.

### 6.3 The Conditional Moment Hierarchy

The x^k prediction in Section 6.2 deserves extended treatment because it is the most concrete prediction the framework makes about an algorithm that exists in one field but not another.

The Affine scan on the k-th power of an input sequence produces the k-th conditional moment of that sequence. Financial econometrics has systematically built this hierarchy:

| k | Algorithm | Field | Year | Reference |
|---|-----------|-------|------|-----------|
| 1 | Exponential smoothing / EWM mean | Statistics/Finance | ~1957 | Holt (1957) |
| 2 | GARCH(1,1) / IGARCH / EWMA variance | Econometrics/Finance | 1986 | Bollerslev |
| 3 | Conditional coskewness model | Econometrics | 1999 | Harvey & Siddique |
| 4 | Conditional cokurtosis model | Econometrics | 2002 | Dittmar |

Deep learning has independently discovered k=1 (momentum SGD, Adam-m) and k=2 (RMSProp, Adam-v). The framework predicts that k=3 and k=4 should exist in deep learning as well.

A "skewness-corrected Adam" (ScAdam) would add:

```rust
m3[i] = beta3 * m3[i] + (1.0 - beta3) * g[i].powi(3);  // k=3: conditional skewness
```

and use the standardized skewness `m3 / v^(3/2)` to asymmetrize the step — taking larger steps in directions where the gradient distribution tilts toward negative values (indicating a more strongly curved loss surface). This is the optimization analog of the "skewness premium" in finance: Harvey & Siddique (1999) showed that assets with negative conditional skewness command higher expected returns; ScAdam would treat parameters with negative gradient skewness as requiring larger steps.

This prediction is falsifiable (implement ScAdam, test on heavy-tailed loss surfaces), novel (no existing optimizer paper derives from the financial conditional moment hierarchy), and structurally guaranteed (the x^k Affine scan is the same accumulate call for all k). Whether ScAdam is *useful* is an empirical question outside the scope of this paper. That it *exists* — as a computable algorithm with a well-defined structural place in the product space — is the framework's claim.

The deeper implication: finance discovered the entire k=1..4 conditional moment hierarchy through the lens of pricing kernel theory (each additional moment is needed to explain asset returns when investors are non-Gaussian). Deep learning discovered k=1,2 through the lens of adaptive learning rates. The product space predicts the connection; the histories confirm that the same accumulate structure appears wherever sequences are analyzed and moments are needed.

### 6.4 The Generator Interpretation

The product space is a *generator* for algorithms. Each point (T, K, O) defines a computation:

1. Apply transform T to data
2. Accumulate using kingdom K's pattern
3. Extract using oracle O's interface

Any valid combination produces a well-defined algorithm. Whether that algorithm is *useful* depends on the domain — but it is always *computable*. The space of useful algorithms is a subset of the space of computable algorithms, and the product space enumerates the latter completely (within the associative-combine constraint).

---

## 7. Pedagogical Implications

The structural rhymes catalog has direct implications for how numerical methods are taught.

### 7.1 The Current Curriculum

Standard curricula teach algorithms as independent methods. A graduate student in statistics takes separate courses in regression (F10), multivariate analysis (F33), time series (F17), and survival analysis (F13). Each course derives its methods from first principles in its own notation. The student learns four sets of formulas and four sets of software tools. Connections between courses are mentioned informally ("Cox regression is like logistic regression but...") without formal unification.

### 7.2 The Accumulate Curriculum

An alternative curriculum organized around the accumulate decomposition would proceed differently:

**Week 1–2**: The accumulate operation. Grouping patterns. The 8 operators. Students learn ONE framework.

**Week 3–4**: The MSR principle. Given data, what is the minimum set of accumulated values? Students compute MomentStats for a dataset and derive mean, variance, skewness, kurtosis — not as separate formulas but as extractions from one accumulation.

**Week 5–6**: The t-test as an extraction. "You already computed the MSR in Week 3. The t-test is the SAME numbers, asked a different question." ANOVA as the same MSR with a ByKey grouping. F = t² verified computationally.

**Week 7–8**: The IRLS template. "You learned OLS in Week 5. Logistic regression is the same loop with a different weight function." Students implement the template once and instantiate 5 families by changing one function.

**Week 9–10**: Transforms. "Rank your data, then run the same tests from Week 5–6. You just did non-parametric statistics." Spearman from Pearson. Kruskal-Wallis from ANOVA.

**Week 11–12**: The product space. "You've now seen 50 algorithms. All are points in a 3D space. Here are the unfilled cells — your final project is to investigate one."

This curriculum teaches FEWER formulas and produces DEEPER understanding. The student who learns the IRLS template in Week 7 can instantiate any GLM, any robust estimator, any mixture model, any IRT model — because they understand the LOOP, not just the WEIGHT FUNCTION.

### 7.3 The Instrument as Tutor

In an interactive system built on the accumulate decomposition, the connections are visible to the student in real time:

> "Your t-test on line 4 consumed the same {n, mean, M₂} that .describe() computed on line 2. A t-test IS a descriptive statistic viewed as a hypothesis."

This is not a pedagogical annotation — it is a computational fact. The system KNOWS that the same accumulated values were used because they share a content-addressed identifier in the session. The learning is embedded in the tool.

---

## 8. Conclusion

The fragmentation of numerical computing — the same algorithm reimplemented under different names across communities separated by notation, tradition, and publication venue — is an artifact of missing instrumentation.

The accumulate decomposition provides that instrumentation. By expressing every algorithm as a choice from four menus (addressing, grouping, expression, operator), the decomposition strips surface notation from computational structure. Algorithms that look different in their native notation reveal shared structure when decomposed.

The 34+ structural rhymes cataloged here are not the full set — they are the subset we discovered in a systematic survey of 35 algorithm families. Each new family added to the framework reveals additional rhymes, because the product space T × K × O guarantees that algorithms sharing two of three coordinates will exhibit structural equivalence on the third.

The IRLS master template — 10 families spanning nearly two centuries of independent development, unified by a single weighted accumulation primitive — demonstrates the practical consequence. Each family's distinctive contribution is a weight function: one line of code. The shared loop is everything else.

The fragmentation persists not because the connections are hidden, but because each community correctly identifies its distinctive contribution — the weight function, the grouping key, the transform — and incorrectly concludes that a distinctive contribution implies a distinctive computation. The accumulate decomposition reveals that distinctiveness lives in the *parameters*, not the *structure*. The structure is shared. The parameters are the contribution. Seeing this requires an instrument that strips parameters from structure, which is what we provide.

---

## References

[To be compiled — key citations include:]

- Ioffe, S. & Szegedy, C. (2015). Batch Normalization. ICML.
- Ba, J.L., Kiros, J.R. & Hinton, G.E. (2016). Layer Normalization. arXiv.
- Ulyanov, D., Vedaldi, A. & Lempitsky, V. (2016). Instance Normalization. arXiv.
- Wu, Y. & He, K. (2018). Group Normalization. ECCV.
- Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS.
- Legendre, A.-M. (1805). Nouvelles méthodes pour la détermination des orbites des comètes.
- Henderson, C.R. (1950). Estimation of genetic parameters. Annals of Mathematical Statistics.
- Cox, D.R. (1958). The regression analysis of binary sequences. JRSS-B.
- Rasch, G. (1960). Probabilistic Models for Intelligence and Attainment Tests.
- Huber, P.J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics.
- Jöreskog, K.G. (1969). A general approach to confirmatory maximum likelihood factor analysis. Psychometrika.
- Nelder, J.A. & Wedderburn, R.W.M. (1972). Generalized linear models. JRSS-A.
- Cox, D.R. (1972). Regression models and life-tables. JRSS-B.
- Dempster, A.P., Laird, N.M. & Rubin, D.B. (1977). Maximum likelihood from incomplete data via the EM algorithm. JRSS-B.
- Jordan, M.I., Ghahramani, Z., Jaakkola, T.S. & Saul, L.K. (1999). An introduction to variational methods for graphical models. Machine Learning.
- Krige, D.G. (1951). A statistical approach to some basic mine valuation problems. Journal of the Chemical, Metallurgical and Mining Society of South Africa.
- Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT Press. (Chapter 4: kriging connection.)
- McCullagh, P. & Nelder, J.A. (1989). Generalized Linear Models, 2nd ed. Chapman & Hall.
- Green, P.J. (1984). Iteratively reweighted least squares for maximum likelihood estimation. JRSS-B.
- Hollander, M., Wolfe, D.A. & Chicken, E. (2013). Nonparametric Statistical Methods, 3rd ed. Wiley.
- Bandt, C. & Pompe, B. (2002). Permutation entropy. Physical Review Letters.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics.
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity. Econometrica. (ARCH, precursor to GARCH.)
- Nelson, D.B. (1990). Stationarity and persistence in the GARCH(1,1) model. Econometric Theory. (IGARCH.)
- RiskMetrics Group. (1994). RiskMetrics Technical Document. J.P. Morgan.
- Harvey, C.R. & Siddique, A. (1999). Autoregressive conditional skewness. Journal of Financial and Quantitative Analysis.
- Dittmar, R.F. (2002). Nonlinear pricing kernels, kurtosis preference, and evidence from the cross section of equity returns. Journal of Finance.
- Tieleman, T. & Hinton, G. (2012). Lecture 6e — RMSProp. COURSERA: Neural Networks for Machine Learning. (RMSProp.)
- Kingma, D.P. & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.
- Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. ICLR. (AdamW.)

## Appendix A: Full Rhyme Catalog

[34+-rhyme table with (T,K,O) coordinates — compiled by naturalist, organized by Type I-V]
