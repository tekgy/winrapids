# WinRapids

Windows-native GPU-accelerated data science toolkit. Market Atlas signal farm.

---

## What This System IS

The market is a temporospatial system. Not a prediction problem. Not a game to win. A SYSTEM — with properties that can be described, measured, and farmed as signal.

The goal: build the instrument that describes the system. Farm every measurable signal. Let the system speak. When it has enough dimensional resolution, it stops being a prediction problem — you observe from close enough to the true structure that the behavior becomes legible.

The kingdoms are a dimensional ladder. Each kingdom adds one tensor cross-axis. K01 is 1D (ticks). K02 is 2D (bins). K03 is 3D (cross-cadence). K04 is 4D (cross-ticker). Higher kingdoms add spatial axes — the fault map, the phase space topology, the learned metric space where market coupling lives geometrically.

Traditional signals are 2D projections of higher-dimensional structure. We're building the projection apparatus to climb the dimensional ladder until the structure resolves.

---

## Irrevocable Architectural Principles

These are not preferences. They are enforced by the architecture. Violating them produces systems that create more problems than they solve. Every prior attempt learned this.

### node === node

Every node has identical structure regardless of scale — trunk, branch, leaf, all the same. Enforced by the CLI scaffold generator: you cannot build a half-node. The generated node is fully wired before any math is written.

**Why irrevocable**: node===node is what makes the agent-as-researcher pattern work. Give an agent the scaffold as its only context — it has everything to build novel math and have it be production-ready on arrival. No integration work. The architecture is the guarantee.

**In practice**: Generate the scaffold first. Get it running (even with `log()`). THEN write the math. Never build infrastructure by hand.

### No Tech Debt — Ever

See it, fix it. The codebase must be fully understandable at all times.

**Why irrevocable**: Tech debt in a complex system compounds. The cost of "fix it later" is not linear — it's multiplicative. One piece of debt makes the next piece harder to fix, which makes the next harder, until the system is incomprehensible. This has happened before. The zero-debt rule is what keeps the system explorable.

### Anti-YAGNI

If the principles structurally guarantee we'll need something, build it now. The question is not "do we need this today?" but "does the structure guarantee we'll need it?"

**Why irrevocable**: This system has proven empirically, repeatedly, that over-specification creates serendipitous value at every turn — things needed but not yet anticipated emerge for free. Under-specification means a format version bump later, a migration, a debt. The 64 custom compression slots cost nothing (header is fixed at 4096 bytes). Leaving them out costs a format change when we need slot 65.

**The reflex to fight**: Both human and AI are trained toward YAGNI and "minimum viable." That reflex is almost always wrong in this system. When you feel it: ask "does the structure guarantee we'll need this?" If yes, build it.

### Raw Ticks Always — No Derived Cadences

Every cadence (1s, 5s, 30s, 1min...) is computed directly from raw ticks. Never from a coarser cadence.

**Why irrevocable**: Computing 5-minute bins from 1-minute bins is lossy. The microstructure visible in raw ticks is destroyed before you start. More importantly: computing from derived cadences destroys the symplectic structure — you're working in a deformed phase space. K03 cross-cadence is only meaningful if each cadence is an independent projection of the same raw reality. Derived cadences compare summaries of summaries. Independent raw-tick cadences compare bandwidth-orthogonal observations of the same signal.

### Run Everything — Never Gate Production

The signal farm computes every leaf for every ticker, every day. V columns carry confidence metadata. Consumers decide what to trust.

**Why irrevocable**: Research findings change. Today's "unstable" signal might be stable under conditions we haven't seen yet. If we gate production, we lose data we can't recover. The cost of computing an extra leaf is tiny (ms of GPU time). The cost of not having it when you need it is losing irreplaceable information. The pattern: DO columns = the data, V columns = the signal about the signal.

### Preallocation — No Migration Paths

Decide max size at design time. Fix the preallocation. If world changes break the allocation: delete and recompute. No backward compatibility.

**Why irrevocable**: Migration paths are tech debt with a time bomb. Preallocation enforces the invariant that the system is always in a known good state. If something no longer fits, the right answer is to recompute — not to stretch the allocation around old assumptions.

---

## The Tambear Contract — Every Primitive, Every Time

Tambear owns ALL of mathematics. Every implementation must satisfy this contract. This is the filter test for any new function, any refactor, any "can we just use ___" reflex.

### 1. Custom implemented, our way

No wrapping vendor libraries. No calling scipy, LAPACK, BLAS, MKL, FFTW, cuDNN, cuBLAS, nalgebra, rust-blas, ndarray-linalg, or any "just use the X function" shortcut. We write the math from first principles with our own quality bar, our own parameterization, our own numerical stability, our own sharing contract. The math in the papers is canonical; our implementation is authoritative.

**Rationale**: wrapping libraries means inheriting their thread model, their memory model, their precision choices, their API shape, their error handling, their platform assumptions. Vendored math is a black box that can't share intermediates with our session, can't decompose into accumulate+gather, can't run on every device. First-principles math is legible, testable, shareable, and ours to optimize.

### 2. Accumulate + gather decomposition

Every primitive decomposes into tambear's operations: `accumulate(grouping, expr, op)` and `gather(addressing, ...)`. If a method has a natural accumulate+gather form, we express it that way. If it genuinely cannot (e.g., Kingdom B sequential recurrence, Kingdom C iterative fixed-point), we honestly declare the kingdom and TAM handles scheduling.

**Rationale**: the accumulate+gather skeleton is what makes everything composable across backends. A method expressed this way runs on CPU, GPU, NPU, or any future accelerator without rewrites. A method that hides its dependency structure behind opaque loops is a dead-end.

### 3. Shareable intermediates via TamSession — with compatibility enforcement

Every expensive intermediate — distance matrix, covariance, FFT, sorted order statistics, moment stats, QR factorization, eigendecomposition — registers in TamSession with a content-addressed IntermediateTag. Consumers pull from cache. Never recompute.

**But sharing is conditional, not automatic**. Two methods asking for "the distance matrix" may have different requirements. Method A may accept any Lp distance; method B may require a Mahalanobis-corrected distance against the pooled covariance; method C may require a phase-space delay embedding before computing distances. The sharing contract enforces compatibility — if the cached intermediate was computed under assumptions that don't match the consumer's method, it is NOT reusable, and the consumer gets a fresh computation tagged with its own assumptions. Sharing only happens when the upstream intermediate is *provably correct for the downstream method*, not merely "has the same shape and dtype."

This is a correctness invariant, not a performance hint. Incorrect sharing would silently return wrong answers. The IntermediateTag carries enough metadata (method signature, parameters, assumption fingerprint) that a consumer can check `is_compatible(tag, my_requirements)` before pulling.

**Rationale**: in a typical fintek bin, ~15 methods share the same FFT, ~15 share MomentStats, ~6 share the phase-space distance matrix — but only when they share compatible definitions. Computing each intermediate once per bin per session is the only way the library stays fast at scale; computing it *correctly shared* is the only way it stays right.

### 4. Every parameter tunable

No hardcoded alpha. No hardcoded threshold=4/n. No hardcoded bandwidth. No hardcoded tolerance. Every knob a domain expert would touch is an optional parameter with a documented default. Every override flows through `using()`.

**Rationale**: a fixed-parameter library is a curated subset. We don't curate — we expose. Users pick their own thresholds.

### 5. Every measure in every family

We don't pick favorites. For every quantity a researcher might measure, we have the primitive. For every estimator that exists, we have it. For every variant of every test, every version of every measure, every flavor of every transform — we own the catalog.

Fintek's leaves become one-liners calling our primitives. Auto-detection chains compose them into decision trees. Users get a buffet and pick based on their data and preferences. We have every answer; we don't have an opinion.

**The scope**: ALL of mathematics across ALL fields. This is an overly-exhaustive enumeration to anchor the shared vocabulary. It is not a whitelist or a roadmap — it's a starting context for what "all math" means. Every named field below, and every field not listed that exists in the mathematical/scientific literature, is in scope. For each field, we want the complete catalog of primitives, estimators, tests, transforms, and measures.

**Pure mathematics**:
- Algebra: group theory, ring theory, field theory, Galois theory, commutative algebra, homological algebra, linear algebra (every factorization, every decomposition, every iterative solver), multilinear algebra, tensor algebra, abstract algebra generally, representation theory, Lie algebras, universal algebra, category theory, topos theory, higher category theory
- Analysis: real analysis, complex analysis, harmonic analysis, functional analysis (every operator class), measure theory, integration theory, distribution theory, Fourier analysis, variational calculus, non-standard analysis, p-adic analysis
- Topology: point-set topology, algebraic topology, differential topology, geometric topology, knot theory, persistent homology, sheaf theory, topological data analysis
- Geometry: Euclidean geometry, non-Euclidean geometry (hyperbolic, elliptic), projective geometry, analytical geometry, algebraic geometry, differential geometry, Riemannian geometry, Kähler geometry, symplectic geometry, complex geometry, tropical geometry, convex geometry, computational geometry (hulls, Voronoi, Delaunay, visibility, arrangement)
- Number theory: elementary, analytic, algebraic, additive, multiplicative, computational (primality, factorization, discrete log, elliptic curves), Diophantine, transcendental, p-adic, modular forms, L-functions, sieve methods
- Combinatorics: enumerative, algebraic, extremal, probabilistic, analytic, graph theory (every algorithm: paths, flows, matchings, colorings, isomorphism, embedding), hypergraphs, matroids, design theory, Ramsey theory, partition theory, generating functions
- Logic and foundations: set theory, model theory, proof theory, computability theory, type theory, category-theoretic foundations, recursion theory
- Discrete mathematics: formal languages, automata theory (finite, pushdown, Turing), state machines, complexity theory, algorithm analysis

**Probability & Statistics**:
- Probability theory: classical, axiomatic (measure-theoretic), combinatorial, stochastic processes (Markov chains, martingales, random walks, Brownian motion, Lévy processes, point processes, Poisson/Hawkes/renewal/cluster processes), large deviations, concentration inequalities
- Statistics: descriptive (every measure of central tendency, dispersion, shape, association), inferential (every hypothesis test against every reference distribution), parametric, nonparametric, robust, Bayesian (conjugate, MCMC/HMC/NUTS, variational, ABC, particle methods), frequentist, empirical likelihood, bootstrap, permutation, jackknife
- Multivariate analysis: every factor/component method, CCA, MANOVA, LDA/QDA, partial least squares, discriminant analysis, copulas (Gaussian, Clayton, Gumbel, Frank, t-copula, mixed), vine copulas, tail dependence measures
- Time series: AR/MA/ARMA/ARIMA/SARIMA/SARIMAX/VAR/VECM/VARMA, every GARCH variant, state-space (linear Gaussian Kalman, EKF, UKF, particle filters, smoothers), spectral (periodogram, Welch, multitaper, Lomb-Scargle, wavelets, CWT, DWT, wavelet packets, synchrosqueeze, scattering, Wigner-Ville), nonlinear (sample entropy, permutation entropy, Lyapunov, correlation dimension, DFA, MFDFA, Hurst, visibility graphs, recurrence quantification), changepoint (CUSUM, PELT, BOCPD, binary segmentation, Zivot-Andrews), cointegration (Engle-Granger, Johansen), transfer entropy, causal methods (Granger, CCM, convergent cross mapping)
- Survival analysis: Kaplan-Meier, Nelson-Aalen, Cox PH, stratified Cox, time-varying coefficients, AFT models, competing risks, joint models
- Causal inference: potential outcomes, DAGs, IV, propensity score methods, synthetic control, DiD, RDD, matching, g-computation, IPW, doubly robust, mediation, moderation
- Sampling: importance, rejection, Gibbs, Metropolis-Hastings, HMC, NUTS, SMC, reversible jump, slice, nested, stratified, cluster, adaptive
- Experimental design: full factorial, fractional factorial, response surface, blocking, Latin squares, BIBD, orthogonal arrays, optimal design
- Meta-analysis, power analysis, sample size calculation for every test

**Optimization**:
- Unconstrained: gradient descent variants (SGD, momentum, Adam, AdaGrad, RMSProp, AMSGrad, Lookahead), Newton, quasi-Newton (BFGS, L-BFGS, DFP, SR1), conjugate gradient (Fletcher-Reeves, Polak-Ribière, Hestenes-Stiefel), trust region (dogleg, Steihaug, cauchy point), Nelder-Mead, Powell, golden section, Brent, simulated annealing, differential evolution, particle swarm, genetic, CMA-ES, Bayesian optimization, natural gradient, K-FAC, Shampoo
- Constrained: linear programming (simplex, interior point, ellipsoid), quadratic programming, SOCP, SDP, conic, convex optimization (ADMM, proximal gradient, Frank-Wolfe, mirror descent), nonlinear programming (SQP, augmented Lagrangian, barrier methods), integer programming (branch and bound, branch and cut, cutting planes), mixed integer, combinatorial
- Stochastic optimization, robust optimization, chance-constrained, multi-objective (Pareto, NSGA), online learning, bandit algorithms, RL

**Machine learning**:
- Supervised: every regression (OLS, WLS, GLS, ridge, lasso, elastic net, quantile, robust, LOESS, spline, GAM, GLM families, kernel ridge, Bayesian, quantile, composite), every classifier (logistic, SVM with every kernel, decision trees, random forests, extra trees, gradient boosting: XGBoost/LightGBM/CatBoost-style, AdaBoost, k-NN, naive Bayes variants, LDA/QDA, Gaussian process classifier)
- Unsupervised: every clustering (K-means, K-medoids, hierarchical with every linkage, DBSCAN, HDBSCAN, OPTICS, mean shift, affinity propagation, spectral, GMM EM, Bayesian GMM, DP mixture), every dimensionality reduction (PCA, kernel PCA, sparse PCA, robust PCA, ICA, factor analysis, MDS, Isomap, LLE, t-SNE, UMAP, diffusion maps, Laplacian eigenmaps, autoencoders), density estimation (histograms, KDE with every kernel and bandwidth rule, mixture models)
- Probabilistic: HMMs, Bayesian networks, Markov random fields, conditional random fields, PGMs generally, variational inference, expectation propagation, belief propagation
- Neural networks: every activation, every initialization, every optimizer, every layer type (dense, conv1d/2d/3d, transposed conv, pooling variants, batch/layer/group/instance/RMS norm, dropout variants, attention: self/cross/multi-head/causal/rotary/flash, transformer blocks, RNN/LSTM/GRU, graph neural networks, equivariant networks, neural ODEs), every loss function, forward and backward passes for all, training loops, inference paths, quantization, pruning, distillation
- RL: Q-learning, SARSA, DQN variants, policy gradient, actor-critic (A2C/A3C/PPO/TRPO/SAC/TD3), model-based RL, inverse RL, MCTS, AlphaZero-style
- Representation learning: Word2Vec, GloVe, FastText, BERT-style, contrastive (SimCLR, MoCo, BYOL, Barlow Twins), masked autoencoders
- NLP: TF-IDF, LSA, LDA (topic modeling), BM25, language models, sequence labeling (HMM, CRF, BiLSTM-CRF), parsing (CYK, Earley, Shift-Reduce), sentiment, NER, word embeddings
- Computer vision: every classical CV algorithm, feature detection (SIFT, SURF, ORB, Harris), matching, homography, fundamental/essential matrices, camera calibration, SfM, SLAM, optical flow, segmentation
- Fairness/explainability: SHAP, LIME, counterfactuals, disparate impact, equalized odds

**Information theory**:
- Every entropy (Shannon, Rényi, Tsallis, differential, conditional, joint, cross, relative), every divergence (KL, JS, Bregman, f-divergences, Wasserstein, MMD, Hellinger, total variation, energy distance), mutual information and variants (NMI, AMI, normalized, corrected), transfer entropy, directed information, channel capacity, coding bounds (Shannon, Hamming, Gilbert-Varshamov, Singleton), compression (Huffman, arithmetic, LZ, LZW, BWT, ANS), error correction (Reed-Solomon, BCH, Hamming, LDPC, turbo, polar, convolutional), cryptographic entropy, rate-distortion

**Cryptography and coding theory**: every cipher (stream, block, symmetric, asymmetric), every hash, every MAC, every signature scheme, every key exchange, every zero-knowledge proof system, elliptic curve cryptography, lattice cryptography, post-quantum, homomorphic encryption, secure multi-party computation, threshold schemes, blockchain primitives (Merkle trees, consensus algorithms, VDFs)

**Game theory**: normal form games, extensive form, Nash equilibrium computation, correlated equilibrium, evolutionary stable strategies, mechanism design, auctions, matching, cooperative game theory (Shapley, Banzhaf, nucleolus, core), algorithmic game theory, combinatorial game theory, fictitious play, regret minimization, counterfactual regret minimization (CFR)

**Dynamical systems**: ODE solvers (every Runge-Kutta variant, implicit methods, symplectic integrators, IMEX, multistep, BDF, Adams, Gauss), DAE solvers, PDE solvers (finite difference, finite element, spectral methods, finite volume, discontinuous Galerkin, mesh-free), SDE solvers (Euler-Maruyama, Milstein, Heun, stochastic Runge-Kutta), chaos (Poincaré sections, Lyapunov spectra, attractor reconstruction, fractal dimensions), bifurcation analysis, continuation methods, invariant manifolds, perturbation theory

**Control theory**: classical (PID, lead-lag, root locus, Bode, Nyquist), modern (state space, LQR, LQG, Kalman filtering for control), robust (H∞, μ-synthesis, sliding mode), adaptive (MRAC, STR, gain scheduling), optimal (Pontryagin, dynamic programming, HJB), nonlinear (feedback linearization, backstepping, Lyapunov), model predictive control, distributed control, stochastic control

**Numerical analysis**: interpolation (Lagrange, Newton, Hermite, splines, RBF, GP), approximation (Chebyshev, Padé, Remez, rational), quadrature (Gauss, Clenshaw-Curtis, adaptive, Monte Carlo), root finding (bisection, Newton, secant, Brent, Muller, Jenkins-Traub), series acceleration (Aitken, Shanks, Wynn, Richardson, Euler, Levin), arbitrary-precision arithmetic, interval arithmetic

**Physics and mathematical physics**:
- Classical mechanics: Lagrangian, Hamiltonian, symplectic integration, rigid body, multi-body, constraint solvers, N-body (direct, Barnes-Hut, Fast Multipole Method)
- Electrodynamics: Maxwell solvers (FDTD, FEM, method of moments), Green's functions, transmission line, waveguide modes
- Thermodynamics and stat mech: Ising model, Potts, XY, Heisenberg, molecular dynamics, Monte Carlo (Metropolis, cluster algorithms, Wang-Landau, replica exchange), Langevin dynamics, Fokker-Planck
- Quantum mechanics: Schrödinger equation solvers (split-step Fourier, Crank-Nicolson), density matrix evolution, quantum gates, VQE, QAOA, quantum simulation, tensor networks (MPS, MERA, PEPS)
- Relativity: geodesic equation, Christoffel symbols, Riemann/Ricci/Einstein tensors
- Fluid dynamics: Navier-Stokes (every discretization), lattice Boltzmann, vortex methods, SPH, turbulence models (RANS, LES, DNS)
- Continuum mechanics, plasma physics, optics (geometric, physical, quantum), acoustics, cosmology

**Engineering domains**:
- Civil: structural analysis, finite element for solids/beams/plates/shells, seismic response, hydraulic/hydrology, transportation network flow
- Electrical: circuit simulation (modified nodal analysis), power systems, semiconductor device modeling, RF/microwave, signal integrity
- Mechanical: CFD, FEA, vibration analysis, kinematics, dynamics, control systems
- Chemical: reaction engineering, process simulation, equilibrium calculations, distillation, mass/energy balances
- Aerospace: trajectory optimization, orbital mechanics, attitude control, aerodynamics

**Biology, chemistry, biochemistry**:
- Bioinformatics: sequence alignment (Needleman-Wunsch, Smith-Waterman, BLAST-style k-mer search), phylogenetics (parsimony, likelihood, Bayesian, neighbor-joining, UPGMA), genome assembly, variant calling, read mapping (Burrows-Wheeler, suffix arrays), HMM profiles, MSA
- Molecular dynamics, quantum chemistry (Hartree-Fock, DFT, post-HF), reaction kinetics (mass action, Michaelis-Menten, Gillespie SSA, tau-leaping, chemical Langevin), docking, virtual screening
- Systems biology: ODE models of signaling/metabolism, stoichiometric analysis, flux balance, gene regulatory networks
- Neuroscience: spike sorting, connectivity analysis, decoding, generalized linear models for spike trains, dynamic causal modeling, EEG/MEG source localization, fMRI GLM

**Finance (beyond fintek)**: every option pricing model (Black-Scholes, binomial tree, trinomial, Monte Carlo with variance reduction, finite difference for American/Bermudan/exotic), Greeks (analytical and numerical), interest rate models (Vasicek, CIR, Hull-White, HJM, LMM), credit models (Merton, reduced form, CreditMetrics), portfolio optimization (Markowitz, Black-Litterman, risk parity, minimum variance, maximum Sharpe, hierarchical risk parity), VaR/ES (parametric, historical, Monte Carlo), backtesting, market microstructure (Roll, Kyle, Glosten-Milgrom, PIN, VPIN, order book dynamics, impact models), algorithmic trading primitives

**Economics**: consumer theory, producer theory, general equilibrium, DSGE, agent-based models, matching markets, mechanism design, input-output, econometrics (every panel method, every time series method, GMM, maximum likelihood, IV, 2SLS, 3SLS, FGLS)

**Domain-specific**:
- Music theory: tuning systems (equal temperament, just intonation, meantone, well temperament), pitch class sets, neo-Riemannian transformations, Tonnetz, voice leading, harmonic function, rhythm analysis, spectral music theory, psychoacoustic models, audio feature extraction (chroma, MFCC, spectral flux/centroid/rolloff, onset detection, beat tracking, key detection), music information retrieval
- Sabermetrics: every baseball metric (OPS, OPS+, wRC+, FIP, xFIP, BABIP, ISO, wOBA, UZR, DRS, WAR variants, WPA, LI, clutch metrics), Pythagorean expectation, log5 matchup, aging curves, projection systems, park factors
- Environmental science: climate models, carbon cycle, hydrological modeling, ecological dynamics, population genetics
- Epidemiology: compartmental models (SIR, SEIR, SEIRD, agent-based), R0 estimation, contact tracing, survival analysis for disease, spatial epidemiology
- Seismology: Gutenberg-Richter, Omori's law, Bath's law, ETAS models, earthquake magnitude estimation
- Astronomy: ephemeris calculation, stellar modeling, N-body simulation, photometry, spectroscopy, transit detection, time-domain astronomy
- Psychology and psychometrics: every IRT model (Rasch, 1PL, 2PL, 3PL, GRM, PCM, GPCM, nominal), CFA, SEM, path analysis, mediation/moderation, mixed models, reliability (Cronbach, McDonald, Guttman)
- Social sciences: network analysis, social network metrics, longitudinal models, multilevel models, agent-based modeling

**Operations research**: scheduling, routing (TSP, VRP, VRPTW), assignment, transportation, facility location, inventory, queuing theory (every queue type, Jackson networks, BCMP, mean-field), reliability, maintenance, revenue management, simulation (discrete event, Monte Carlo, agent-based)

**Computer science primitives**: sorting (every comparison and non-comparison sort), searching, data structures (every tree/heap/hash), string algorithms (KMP, Boyer-Moore, Aho-Corasick, suffix automata, generalized suffix trees), compression (every algorithm), parsing (LL, LR, GLR, PEG, Earley, CYK, shift-reduce), compiler passes (every optimization), type inference, SMT/SAT solving, constraint propagation, linear algebra kernels (BLAS levels 1/2/3 from scratch), sparse linear algebra (every format and solver)

**The meta-rule**: if you can name a field that has computable math, tambear eventually implements the complete catalog of primitives for that field. This list is not a whitelist — it's an anchor. Fields not mentioned here are still in scope. Every piece of mathematics ever done on a computer (or that will be done in the future) is a tambear primitive, implemented from first principles, composable via accumulate+gather, shareable via TamSession, portable across every backend, with no vendor or OS lock-in.

### 6. Optimized for advanced machines in 2026

Fused passes over modern memory hierarchies. Cache-aware layouts. SIMD where natural. Wavefront scheduling for massively parallel hardware. Avoiding data copies. Minimizing synchronization. Designed for 128-core CPUs, Blackwell/M4 Pro/Grace Hopper, and future accelerators.

### 7. No vendor lock-in

The ONLY hardware dependency we accept is the kernel driver that lets us talk to an ALU — CPU instructions natively, GPU compute units via wgpu (Vulkan/Metal/DX12 cross-platform), NPU via open interfaces when they mature. `cudarc` is feature-gated behind a door, never in the core path.

**Rationale**: vendor lock-in means our library dies when the vendor does. wgpu means the same binary runs on NVIDIA, AMD, Intel, Apple, mobile — everywhere. The portable path is the fast path because we own the kernels.

### 8. No OS lock-in

Runs on Windows, Linux, macOS, mobile, embedded. No OS-specific syscalls in the core. No registry, no epoll, no Metal-only path. OS differences stay behind thin abstraction layers; the math is OS-agnostic.

### 9. Lifting to TAM

Fock boundary issues, orchestration, sequential dependencies, cross-device placement, parallel scheduling, memory management, work stealing, load balancing — all of these belong to TAM, not the primitive. The primitive expresses the math. TAM expresses the execution.

When something is genuinely hard (a Fock boundary, a non-liftable operation, a stateful iterator), the primitive honestly declares its constraints and TAM handles the resolution. The primitive is pure; TAM is the orchestrator.

### 10. Publication-grade rigor — prove every implementation

Every primitive is worked up as if preparing for a Nature paper. Not metaphorically — literally. Before we call anything "done":

- **Every assumption documented**. The math paper assumes X; we state X explicitly in the docs, the test, and the error conditions. If a user violates X, we detect it and say so, not silently produce garbage.
- **Every parameter documented**. Name, type, range, default, meaning, impact on output, when to tune it. A domain expert reading our docs should immediately see the full knob set without going to the source.
- **Benchmarked against every competing implementation**. scipy, R, MATLAB, Julia, Stata, SAS, NumPyro, Stan, PyMC, statsmodels, scikit-learn, Eigen, cuBLAS, cuDNN, LAPACK, FFTW, GSL, Armadillo, NLopt, CVXPY — whatever the reference is, we measure against it. Same inputs, same parameters, same assumptions.
- **Bit-perfect or bug-finding**. Under identical parameter assumptions, our answer matches theirs to numerical precision — OR we find the bug in theirs. We do not treat other implementations as ground truth. They are peers to be verified against mpmath/SymPy/closed-form analytical reference at 50-digit precision. Every bug we find in scipy/R/MATLAB gets filed upstream — even users who never switch to tambear benefit from our rigor.
- **Benchmarked at every scale**. From master-thesis-size datasets (the kind a graduate student runs on a laptop) up through the largest datasets in human or model history. Synthetic datasets with billions of rows, trillions of cells, hundreds of thousands of columns. Real benchmark datasets from every domain that has them (UCI, OpenML, LIBSVM, ImageNet-scale, LAION-scale, climate reanalyses, genomic corpora, high-frequency tick archives). We document the scaling law for every primitive: O(n) constant factor, break points where the algorithm changes regime, memory ceiling, GPU crossover point.
- **Proven correct under the tests that matter**. Not "tests pass" — *tests that would fail if the math were wrong*. Adversarial edge cases: singular matrices, perfect collinearity, zero variance, heavy tails, ties, missing values, extreme magnitudes, ill-conditioning, boundary conditions. Gold-standard oracles for every published result where one exists.

**Rationale**: the value proposition is correctness and depth, not velocity. A library that gets the answer wrong once loses its users forever. A library that finds bugs in its competitors becomes the reference. The work of proving every implementation is the product — not overhead on top of the product.

### The Filter Test

Before shipping any primitive, confirm:

- [ ] Written from first principles, not wrapping a library
- [ ] Expresses as accumulate + gather where possible (kingdom declared otherwise)
- [ ] Registers shareable intermediates via TamSession with compatibility tags
- [ ] Every parameter tunable and documented (name, type, range, default, meaning, when to tune)
- [ ] Assumptions explicit — documented, tested, detected at runtime
- [ ] Benchmarked against every competing implementation (bit-perfect or bug filed upstream)
- [ ] Benchmarked at multiple scales up through billion/trillion-row synthetic data
- [ ] Gold-standard oracle against mpmath/SymPy/closed-form at high precision
- [ ] Adversarial test suite exercises edge cases (singular, collinear, heavy tail, ties, missing, ill-conditioned)
- [ ] Hardware details hidden behind tambear-wgpu / tam-gpu
- [ ] Runs on CPU, GPU, and future accelerators with no code changes
- [ ] Honestly declares its Kingdom (A/B/C/D) so TAM knows how to schedule it
- [ ] Is one more piece of "every math, our way, everywhere, provably correct"

If all pass, ship. If any fail, fix first.

---

## Methods Are Compositions — Primitives Are the Atoms

Every method in tambear decomposes into global primitives + a formula. The method doesn't *own* its sub-algorithms — it *composes* them. The formula is the only thing the method uniquely contributes. Everything else is a call to a primitive that exists independently in the global catalog.

| Method | Primitives it composes | What the method uniquely owns |
|--------|----------------------|-------------------------------|
| Kendall tau | sort, inversion_count, tie_count | tau-b formula |
| Pearson r | moments (mean, variance, covariance) | r = cov/(sx*sy) |
| PCA | covariance_matrix, eigendecomposition | "top k eigenvectors" selection |
| DBSCAN | distance_matrix, neighborhood_count, union_find | density reachability rule |
| GARCH | log_likelihood, optimizer | variance recursion equation |
| Kaplan-Meier | sort, prefix_product | conditional survival formula |
| OLS regression | gram_matrix, cholesky_solve | (X'X)⁻¹X'y formula |

**A primitive is math that exists because it's math, not because some method needs it.** `inversion_count` is a primitive. `covariance_matrix` is a primitive. `sort` is a primitive. Each has its own implementations (mergesort vs fenwick for inversions), its own kingdom classification, its own parameterization.

**A method is a thin orchestration layer** — typically 20-50 lines of composition + formula. If a method is 200+ lines, it probably contains embedded primitives that should be extracted. Every private `fn` inside a method is a question: "is this math that exists independently?" Almost always yes.

### Why primitive decomposition is load-bearing

This isn't just clean code — it's what makes the entire architecture work:

- **`using()` flows through compositions.** When `kendall_tau` calls the global `inversion_count` primitive, the user's `using(inversion_method="fenwick")` reaches it. If the inversion count is a private copy inside Kendall, the `using()` hook can't reach it.
- **Sharing via TamSession.** If a method calls the global `covariance_matrix` primitive, the result registers in TamSession and PCA, factor analysis, LDA, Mahalanobis, and mixed effects all reuse it. If the covariance matrix is private inside each method, it gets computed 6 times per bin.
- **Linting and kingdom classification.** The lint system and TAM scheduler need to know each operation's kingdom independently. If a method hides its sub-operations, `kingdom_of()` can't see inside.
- **Every primitive is callable at Level 0 in TBS.** The user can call `inversion_count(col=0)` directly — not just Kendall. Same syntax, same `using()`, same output format at every level.

### The fractal catalog — flat, not nested

Every primitive is a family. `mean` alone has 15+ named variants: arithmetic, trimmed, winsorized, geometric, harmonic, power (generalized), Lehmer, Fréchet, contraharmonic, exponential moving, kernel-weighted, and more. Each has parameters. Each connects to other primitives (`geometric_mean = exp(mean(log(x)))` — a composition).

The catalog is fractal — zoom in on any entry and it expands — but bounded. Bounded by the finite set of named mathematical operations in the human literature. Families share structure (power mean parameterized by p gives 5 variants from 1 implementation). The work is proportional to the number of DISTINCT mathematical ideas, not the number of combinations.

**The catalog is flat, not hierarchically nested.** Families (statistics, algebra, spectral, etc.) are metadata tags on primitives, not namespaces that constrain access. In TBS, every path resolves to the same computation: `geometric_mean(col=0)`, `mean(col=0).using(method="geometric")`, `mean.geometric(col=0)` — all the same primitive. The user shouldn't need to know our internal module structure to find what they want. Nesting into families risks losing multi-composability — if geometric_mean lives "inside" the mean family, it becomes harder to discover from the geometry side or the transform side. Tags don't have this problem. A primitive can belong to multiple families simultaneously.

The process for growing the catalog:
1. For every method we implement, decompose it into named operations
2. For every named operation, check if it exists as a standalone primitive
3. If not, implement the primitive FIRST, then have the method call it
4. For every new primitive, ask: what's the family? What are the variants?
5. For every variant, check R/scipy/MATLAB/Julia for evidence of user demand
6. Implement the demanded variants

The catalog grows organically from decomposition, not from top-down planning.

### TBS is the universal surface

TBS exposes every level with the same syntax:

- **Level 0 — Primitives**: `mean(col=0)`, `inversion_count(col=0)`, `sort(col=0)`
- **Level 0+ — Compositions**: `rank(col=0).pearson_on_ranks(col_y=1)` (= Spearman, composed from primitives)
- **Level 1 — Methods**: `kendall_tau(col_x=0, col_y=1)` (internally: sort → inversion_count → tie_count → formula)
- **Level 2 — Pipelines**: `two_group_comparison(group_col=2)` (internally: shapiro_wilk → levene → [welch_t | mann_whitney] → cohens_d → ci)
- **Level 3 — Discovery**: `discover_correlation(col_x=0, col_y=1)` (runs every correlation in parallel, reports agreement)

`using()` flows DOWN through all levels. When `two_group_comparison` calls `shapiro_wilk`, the user's `using(alpha=0.01)` reaches it. The primitives don't know they're inside a method. The methods don't know they're inside a pipeline. Each queries the `using()` bag for its parameters and uses defaults if nothing is set.

**Every property propagates at every level.** Every node in the composition tree — from a Level 3 pipeline down to the lowest Level 0 primitive — is independently:

- **Addressable** — `kendall_tau.inversion_count` is a valid path. So is just `inversion_count`.
- **Decomposable** — `inversion_count` itself decomposes to `mergesort_with_count` or `fenwick_tree_count`. Each is a primitive. Each is addressable. The fractal goes all the way down.
- **Tunable** — `using(inversion_method="fenwick")` reaches inside `kendall_tau` to the specific sub-step.
- **Discoverable** — `discover_inversion_count` runs every algorithm, reports which is fastest, which gives which answer.
- **Sweepable** — `sweep(inversion_method=["mergesort","fenwick","brute"])` runs all three, returns all three.
- **Superpositionable** — don't collapse. Keep all results. The structural fingerprint of agreement IS the output.
- **using()-able** — every parameter at every depth is reachable from the top.
- **Shareable** — the result registers in TamSession. If Kendall already computed the inversion count, Spearman's footrule reuses it. First consumer pays; everyone else gets it free. One pass.

A composed method in TBS is not a new thing — it's a **named grouping of existing things**. `kendall_tau` is syntactic sugar for `sort → inversion_count → tie_count → tau_b_formula`. The method adds a name and a formula. It doesn't add new computation. Every sub-step is independently addressable at Level 0. The composition tree IS the computation graph. TBS IS the compiler's IR.

### TBS as completeness test

TBS is the forcing function for primitive completeness. When you try to express a function as a TBS statement, every symbol in the expression must resolve to a primitive. If you can't write a clean expression, you've found a gap. The language surface IS the completeness test for the flat catalog.

Three things TBS reveals:

**"This primitive does more than one thing."** If a function internally does multiple distinct computations, it's a compound primitive that breaks composability. A real primitive is one symbol, one operation, one result. If `ar_fit` internally computes ACF + Levinson-Durbin + coefficients + residuals + AIC, that's 5 primitives glued together. Each should be callable independently, shareable independently, liftable independently. Test: can you decompose the TBS expression into smaller expressions that each do exactly one thing? If yes, it should BE those smaller expressions composed.

**"What else computes this?"** For every primitive, the literature has alternative methods. `correlation` → Pearson, Spearman, Kendall, point-biserial, phi, polychoric, tetrachoric, distance, MIC, Hoeffding's D, Schweizer-Wolff, Blomqvist beta... Each is a flavor. Each is a `using(method=...)` key or a standalone primitive. If we only have 4, TBS makes the gap visible.

**"What parameters do people actually tune?"** What R, scipy, MATLAB, Julia, Stata expose for the same computation — every parameter they offer that we don't is a missing `using()` key. The full parameter space of every primitive should be discoverable through TBS, not hidden in source code.

**The primitive IS the library. Everything above is composition.**

---

## Layers Above the Math

Tambear's primitives are pure math. They do one thing, do it correctly, and return a result. That's it — no opinions, no diagnostics, no method-switching, no workflow assembly. A primitive like `spearman_correlation(x, y)` just computes Spearman's ρ.

The *smart* parts of tambear — the parts that make it feel like a $10M/year quant is sitting next to the user — live in **separate layers built on top of the math, not inside it**. These layers compose primitives, tune parameters, pick methods, and enforce rigor. They are first-class products in their own right, but they are categorically distinct from the math primitives below them. This separation is the reason the primitives stay clean, reusable, and composable across every possible higher-level consumer.

### The layers

**Layer 0 — Math primitives** (the Tambear Contract above).
Pure functions. One method per entry point. Accumulate + gather decomposition. Shareable intermediates. Benchmarked, oracled, adversarial-tested. No knowledge of pipelines, no knowledge of diagnostics, no knowledge of other primitives. `spearman_correlation(x, y) -> f64`.

**Layer 1 — Diagnostics and auto-selection**.
A separate layer that knows *which primitive to call for which data*. When a user asks for the generic thing (`correlation(x, y)`), Layer 1 runs the diagnostics a senior statistician would run (normality check, variable-type check, outlier influence, ties), picks the appropriate primitive (Pearson / Spearman / Kendall τ-b / polychoric / tetrachoric / point-biserial / distance correlation), calls it, and reports the decision + rationale in a structured output (`TbsStepAdvice`: `recommended`, `user_override`, `diagnostics`). Layer 1 is built *out of* Layer 0 primitives — it calls `shapiro_wilk`, `pearson_r`, `spearman_r`, etc. — but the primitives themselves know nothing about Layer 1.

**Layer 2 — `using()` override transparency**.
Built on Layer 1. When a user forces a specific method via `using(method="pearson")`, Layer 2 still runs the Layer 1 recommendation *and* the user's choice, and writes both results into the output alongside a warning if the override is statistically questionable. The user sees what they chose, what tambear would have chosen, and the numerical difference — so they disagree with their eyes open. Neither silently wins. Layer 2 is a policy layer that wraps Layer 1; the underlying math primitives are called twice from the same diagnostic-backed infrastructure.

**Layer 3 — Expert pipelines as persisted defaults**.
Curated workflows shipped as first-class artifacts — `two_group_comparison`, `regression_diagnostics`, `time_series_analysis`, `clustering_workflow`, `survival_analysis`, etc. Each is a composition of Layer 0 primitives orchestrated through Layer 1 diagnostics and Layer 2 override handling, tuned as if a decade-experienced domain expert was hired to build that one workflow from scratch. Users load a pipeline and run it; inside, it walks the decision tree (normality → variance homogeneity → t-test variant → effect size → power → CI → robustness) calling primitives as needed. Each pipeline has its own version, its own test suite against published benchmark datasets, and its own oracle against whatever the gold-standard workflow in that domain looks like. The pipeline is the product most users load — the toolkit stays available for people who want to assemble their own.

**Layer 4 — `.discover()` / superposition auto-discovery**.
The layer that runs *every plausible method simultaneously*, keeps all results in superposition, and reports structural fingerprints of agreement/disagreement across methods. When the user wants exploration rather than a single answer, Layer 4 surfaces "which views agree, which disagree, and what does that tell us about the data?" Layer 4 is the opposite philosophical stance from Layer 1 (Layer 1 picks; Layer 4 refuses to pick) but uses the same Layer 0 primitives underneath. Both layers exist because both stances are valid depending on the user's question.

### Why the separation matters

- **Primitives stay reusable**. `spearman_correlation` can be called from Layer 1 (pipeline picked it), Layer 2 (user forced it), Layer 3 (regression pipeline needs a rank correlation on residuals), Layer 4 (superposition ensemble), or a user's own custom code. Zero coupling to any one consumer.
- **Layers can be swapped or extended**. A new diagnostic layer that prefers Bayesian reasoning over frequentist can be built without touching a single math primitive. A new pipeline genre (e.g. experimental-design workflows) is just a new Layer 3 module that calls existing Layer 0 math.
- **Testing is cleaner**. A primitive is tested against its math. A diagnostic layer is tested against whether it picks the right primitive. A pipeline is tested against whether it produces the expert's workflow. Each layer has its own oracle.
- **The Contract stays scoped**. The Filter Test applies to Layer 0 only. Layer 1/2/3/4 have their own separate quality gates — they don't need to be bit-perfect vs scipy (because they're workflow orchestration, not math), but they do need to make defensible methodological choices, match expert judgment on benchmark scenarios, and preserve override transparency.

**The rule**: if you find yourself wanting a math primitive to "know about" normality checks, method switching, expert defaults, or diagnostic wiring — stop. That knowledge belongs in a layer above. The primitive stays pure. The layer above composes it.

---

## Architecture Reference

**Kingdoms**: Numeric always (K01, K02...). Kingdom = tensor rank. Never mix kingdom and representation in the name.

**KIKO**: KI = input representation, KO = output representation. Orthogonal to kingdom. KO00=columnar, KO01=FFT, KO04=wavelet, KO05=sufficient stats, KO06=correlation, KO07=eigenvectors. Both filename and header carry KO code (belt-and-suspenders).

**All kingdoms farm**: No "research kingdoms." Research is a leaf lifecycle stage (experimental → prune | keep | production). Every leaf in every kingdom runs on real data.

**Leaf lifecycle**: Predefined (known from start, production immediately). Experimental (discovered, runs on real data, not yet promoted). Status is on the leaf, not the kingdom.

**MKTF / MKTC**: MKTF is the sub-file format (Block 0, sections, ByteEntry columns). MKTC is the pre-allocated container (MKTF blobs at fixed byte offsets). Per-dtype MKTF blobs within container (float32.mktf, int32.mktf etc.) — independent compression per dtype per leaf.

**V columns**: Every leaf with a confidence dimension produces V columns alongside DO columns. Never suppress the leaf because V is low — suppress consumption, not production.

---

## What Good Looks Like

- A new signal hypothesis → generate scaffold → running in pipeline → write math. Infrastructure is free.
- A researcher agent with only the scaffold template produces production-ready math.
- The header IS the manifest. Reading the header is reading the data's identity.
- Complexity that captures a real phenomenon IS elegance. Don't simplify it away.
- When something feels over-engineered: ask whether the structure guarantees we'll need it. Usually yes.
