# Tambear Math Library — Complete Landscape

Every family below is a campsite. Every method within each family is a sub-campsite.
Every method has all flavors/variants. Every implementation is from tambear primitives.
No vendor libraries. No fallbacks. Any GPU. Bit-perfect cross-platform.

This replaces: cuBLAS, cuFFT, cuML, cuDNN, cuSOLVER, cuSPARSE, cuRAND,
SPSS, SAS, Stata, MATLAB, Mplus, R, Python scipy/sklearn/statsmodels.

---

## Foundational Primitives (build first — everything else uses these)

### 01. Distance & Similarity
L2, cosine, Mahalanobis, Poincaré, spherical geodesic, Minkowski,
Jaccard, Hamming, edit distance, DTW, Wasserstein, KL divergence,
all kernel functions (RBF, polynomial, sigmoid, Laplacian).
Lifted to its own family — KMeans, KNN, SVM, etc. use this, don't reimplement.

### 02. Linear Algebra (cuBLAS + cuSOLVER + cuSPARSE replacement)
GEMM, GEMV, triangular solve, Cholesky, LU, QR, SVD (full/truncated/randomized),
eigendecomposition (symmetric/general), least squares, tridiagonal solver,
matrix inversion, determinant, trace, norms, condition number.
Sparse: SpMV, SpMM, sparse triangular solve, format conversions (CSR/COO/BSR).
Batched small-matrix operations.

### 03. Signal Processing (cuFFT replacement + beyond)
FFT (Cooley-Tukey radix-2, split-radix, Bluestein chirp-z, arbitrary-N),
IFFT, real FFT (RFFT), 2D/3D FFT, STFT, NUFFT (non-uniform),
sparse FFT, DCT/DST, NTT (number theoretic), Hartley transform.
Wavelets: DWT, CWT, SWT, wavelet packets, Haar, Daubechies, Morlet, Gabor.
Filters: FIR, IIR, Butterworth, Chebyshev, Kalman, Savitzky-Golay, median.
Hilbert transform, cepstrum, Wigner-Ville, analytic signal.

### 04. Random Number Generation (cuRAND replacement)
Philox-4x32, MRG32k3a, MTGP32, XORWOW, PCG.
Distributions: uniform, normal, log-normal, exponential, gamma, beta,
Poisson, binomial, multinomial, Cauchy, Student-t, F, chi-square,
Weibull, Pareto, Dirichlet, von Mises, multivariate normal.
Sobol quasi-random sequences. Permutation/shuffle.

### 05. Optimization
SGD, Adam, AdamW, RMSprop, Adagrad, L-BFGS, conjugate gradient,
Newton's method, quasi-Newton (BFGS, SR1), trust region.
Constrained: linear programming, quadratic programming, ADMM.
Stochastic: simulated annealing, genetic algorithms, particle swarm.
Riemannian optimization (on manifolds).

---

## Descriptive & Inferential Statistics (SPSS/SAS/Stata replacement)

### 06. Descriptive Statistics
Central tendency: mean, median, mode, trimmed mean, winsorized mean,
geometric mean, harmonic mean, weighted mean.
Dispersion: variance, std, MAD, IQR, range, CV, Gini coefficient.
Shape: skewness (Pearson/Fisher/Bowley/L-moment/medcouple),
kurtosis (excess/Fisher/Bowley/L-moment), moments 1-8.
Quantiles: percentiles, quartiles, deciles, arbitrary, interpolated (9 methods).
All flavors. Population and sample versions.

### 07. Hypothesis Testing
t-tests: one-sample, two-sample (equal/unequal variance), paired, Welch's.
ANOVA: one-way, two-way, factorial, repeated measures, mixed, MANOVA, ANCOVA.
Chi-square: goodness of fit, independence, homogeneity, McNemar.
Proportion tests: z-test, Fisher exact, binomial test.
Equivalence: TOST, non-inferiority.
Multiple comparison: Bonferroni, Holm, BH (FDR), Tukey HSD, Dunnett, Scheffé.
Effect sizes: Cohen's d, eta-squared, omega-squared, Cramér's V, odds ratio.
Power analysis for all of the above.

### 08. Non-parametric Statistics
Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman.
Kolmogorov-Smirnov, Anderson-Darling, Shapiro-Wilk (normality).
Runs test, sign test, median test, Mood's median.
Bootstrap: percentile, BCa, studentized, parametric.
Permutation tests: exact, approximate, stratified.
Kernel density estimation (all kernels + bandwidth selection).
Rank-based methods: Spearman, Kendall, rank regression.

### 09. Robust Statistics
M-estimators (Huber, bisquare, Hampel). Trimmed/winsorized statistics.
Median absolute deviation (MAD). Breakdown point methods.
Least trimmed squares, least median squares. MM-estimators.
Robust PCA, robust covariance (MCD, MVE, S-estimators).

---

## Regression & Modeling (SAS/Stata replacement)

### 10. Regression
OLS, WLS, GLS, feasible GLS. Ridge, lasso, elastic net.
Logistic (binary, multinomial, ordinal, conditional).
Poisson, negative binomial, zero-inflated, hurdle models.
Quantile regression. Robust regression (M-estimation, LTS, LMS).
Polynomial regression. Stepwise selection (forward, backward, both).
Diagnostics: residuals, leverage, Cook's D, VIF, DW, Breusch-Pagan, White.

### 11. Mixed Effects & Multilevel
Linear mixed models (LME). Generalized linear mixed models (GLMM).
Random intercepts, random slopes, crossed random effects, nested.
REML vs ML estimation. Kenward-Roger / Satterthwaite df.
ICC (intraclass correlation). Multilevel SEM.

### 12. Panel Data & Econometrics
Fixed effects, random effects, first differences. Hausman test.
Instrumental variables (2SLS, GMM). Treatment effects (ATE, ATT).
Difference-in-differences. Regression discontinuity.
Heckman selection. Tobit. Duration models.

### 13. Survival Analysis
Kaplan-Meier estimator. Log-rank test. Cox proportional hazards.
Parametric survival (exponential, Weibull, log-normal, log-logistic).
Accelerated failure time. Competing risks (Fine-Gray).
Frailty models. Time-varying covariates. Concordance index.

---

## Latent Variable Models (Mplus replacement)

### 14. Factor Analysis & SEM
Exploratory FA: principal axis, ML, ULS, WLS, MINRES.
Rotation: varimax, promax, oblimin, geomin, target, bifactor.
Polychoric/polyserial/tetrachoric correlations.
Confirmatory FA. Full SEM (structural + measurement).
Path analysis. Mediation/moderation. Multi-group SEM.
Fit indices: chi-square, CFI, TLI, RMSEA, SRMR, AIC, BIC.

### 15. IRT & Psychometrics
1PL (Rasch), 2PL, 3PL, 4PL. Graded response model.
Partial credit model. Nominal response model.
DIF (differential item functioning). Test equating.
CAT (computerized adaptive testing). Item-person maps.

### 16. Mixture & Latent Class Models
Gaussian mixture models (EM). Latent class analysis.
Latent profile analysis. Growth mixture models.
Factor mixture models. Hidden Markov (with latent classes).
BIC/AIC/entropy for class enumeration.

---

## Time Series (MATLAB/EViews replacement + fintek)

### 17. Time Series Models
AR, MA, ARMA, ARIMA, SARIMA, ARIMAX.
Exponential smoothing (SES, Holt, Holt-Winters).
State space models (local level, local trend, BSM).
VAR, VECM, structural VAR. Granger causality.
Cointegration (Engle-Granger, Johansen). Unit root tests (ADF, KPSS, PP).
Structural breaks (CUSUM, Chow, Bai-Perron, PELT).
Changepoint detection (BOCPD, PELT, binary segmentation).

### 18. Volatility & Financial Time Series
GARCH, EGARCH, GJR-GARCH, TGARCH, FIGARCH.
Realized volatility, bipower variation, multipower variation.
Stochastic volatility (SV). Jump detection (BNS).
VPIN, Roll spread, Kyle's lambda. Market microstructure.
DFA, MF-DFA, Hurst R/S. Scaling exponents. Long memory.

### 19. Spectral Time Series
Periodogram, Welch PSD, multitaper PSD, Burg AR PSD.
Lomb-Scargle (irregular sampling). Cross-spectral density.
Coherence, transfer function, impulse response.
Spectral entropy. Energy band decomposition.

---

## Machine Learning (cuML replacement + beyond)

### 20. Clustering
KMeans (Lloyd, mini-batch, KMeans++, bisecting).
DBSCAN, HDBSCAN, OPTICS. Hierarchical (single/complete/average/Ward).
Spectral clustering. Gaussian mixture (EM). Mean-shift.
Affinity propagation. BIRCH. Agglomerative.
Cluster validation: silhouette, Davies-Bouldin, Calinski-Harabasz, gap statistic.

### 21. Classification & Supervised Learning
Logistic regression. SVM (linear, kernel). Decision trees (CART, C4.5).
Random forest. Gradient boosting (XGBoost-style, LightGBM-style).
Naive Bayes (Gaussian, multinomial, Bernoulli). KNN.
LDA, QDA. AdaBoost. Stacking, bagging.
Metrics: accuracy, precision, recall, F1, AUC-ROC, log-loss, confusion matrix.

### 22. Dimensionality Reduction
PCA (full, truncated, randomized, incremental, kernel).
ICA (FastICA, Infomax). NMF. Sparse PCA.
t-SNE (Barnes-Hut). UMAP. MDS (classical, metric, non-metric).
LDA (as dimensionality reduction). Isomap. LLE.
Autoencoders (linear, variational).

---

## Deep Learning Primitives (cuDNN replacement)

### 23. Neural Network Operations
Convolution: 1D, 2D, 3D (direct, FFT-based, Winograd). Transposed convolution.
Pooling: max, avg, global, adaptive. Unpooling.
Normalization: batch, layer, group, instance, RMS.
Attention: scaled dot-product, multi-head, causal, local, sparse, flash.
All activations: ReLU, LeakyReLU, GELU, SiLU, Swish, Mish, ELU, SELU, tanh, sigmoid, softmax, log-softmax.
Dropout, embedding lookup, positional encoding.

### 24. Training Infrastructure
Loss functions: MSE, MAE, cross-entropy, binary CE, focal, hinge, contrastive, triplet, InfoNCE, CTC.
Optimizers: SGD (momentum, Nesterov), Adam, AdamW, LAMB, LARS, Adafactor.
LR schedulers: cosine, linear warmup, step, exponential, cyclic, one-cycle.
Gradient clipping, gradient accumulation, mixed precision.
Data augmentation: random crop, flip, rotate, color jitter, mixup, cutout.

---

## Information Theory & Complexity

### 25. Information Theory
Shannon entropy, Rényi entropy (all orders), Tsallis entropy.
Mutual information (discrete, continuous/KSG estimator).
Conditional entropy, joint entropy. KL divergence, JS divergence.
Transfer entropy. Conditional mutual information.
Rate-distortion. Channel capacity.

### 26. Complexity & Chaos
Lyapunov exponents (Rosenstein, Kantz). Correlation dimension.
Sample entropy, approximate entropy, permutation entropy.
Lempel-Ziv complexity. Recurrence quantification analysis.
Fractal dimension (box-counting, Higuchi). Hurst exponent.
Kolmogorov complexity estimates. Symbolic dynamics.

---

## Geometry, Topology & Manifolds

### 27. Topological Data Analysis
Persistent homology (Vietoris-Rips, alpha complex, Čech).
Betti numbers, persistence diagrams, persistence landscapes.
Mapper algorithm. Euler characteristic curves.
Topological signatures for time series and point clouds.

### 28. Manifold Operations
Riemannian gradient, exponential map, logarithmic map.
Parallel transport, geodesic computation.
Curvature (sectional, Ricci, scalar).
Fréchet mean (on any manifold). Karcher mean.
Manifold-valued regression. Manifold interpolation.

### 29. Graph Algorithms
PageRank, HITS. Shortest paths (Dijkstra, Bellman-Ford, Floyd-Warshall).
Connected components. Community detection (Louvain, Leiden, label propagation).
Centrality (betweenness, closeness, eigenvector, Katz).
Graph Laplacian, spectral embedding. Triangle counting.
Random walks, graph kernels, Weisfeiler-Leman.

---

## Spatial & Specialized

### 30. Spatial Statistics
Kriging (ordinary, universal, co-kriging). Variograms.
Point processes (Poisson, Cox, Hawkes). Ripley's K/L functions.
Spatial autocorrelation (Moran's I, Geary's C). GWR.
Haversine distance, point-in-polygon, convex hull.

### 31. Interpolation & Approximation
Splines: cubic, B-spline, natural, clamped, NURBS.
Polynomial fitting (Vandermonde, orthogonal polynomials).
Gaussian process regression (with all kernels).
Radial basis function interpolation.
Chebyshev approximation. Padé approximation.

### 32. Numerical Methods
Quadrature: Gauss-Legendre, Simpson, adaptive (Gauss-Kronrod).
Root finding: Newton, bisection, Brent, secant.
ODE solvers: Euler, RK4, RK45 (adaptive), BDF (stiff).
Numerical differentiation (finite differences, complex step).
Monte Carlo integration.

---

## Multivariate & Specialized Statistics

### 33. Multivariate Analysis
MANOVA, MANCOVA. Canonical correlation analysis.
Discriminant analysis (Fisher's, regularized).
Correspondence analysis (simple, multiple).
Procrustes analysis. ANOSIM, PERMANOVA.

### 34. Bayesian Methods
MCMC: Metropolis-Hastings, Gibbs sampler, HMC, NUTS.
Variational inference: ADVI, normalizing flows.
Bayesian regression (linear, logistic, hierarchical).
Bayesian model comparison (Bayes factors, WAIC, LOO-CV).
Posterior predictive checks. Prior sensitivity analysis.
Bayesian nonparametrics (Dirichlet process, Gaussian process).

### 35. Causal Inference
Propensity score matching. Inverse probability weighting.
Instrumental variables. Regression discontinuity.
Difference-in-differences. Synthetic control.
Mediation analysis (Baron-Kenny, product of coefficients).
Do-calculus fundamentals. Causal forests.

---

## Total: 35 families, ~500+ individual algorithms, every flavor/variant.

Each family = one campsite.
Each algorithm = one sub-campsite.
Each implementation: from tambear primitives, any GPU, verified, lab-notebooked.
