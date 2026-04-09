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

### 3. Shareable intermediates via TamSession

Every expensive intermediate — distance matrix, covariance, FFT, sorted order statistics, moment stats, QR factorization, eigendecomposition — registers in TamSession with a content-addressed IntermediateTag. Consumers pull from cache. Never recompute.

**Rationale**: in a typical fintek bin, ~15 methods share the same FFT, ~15 share MomentStats, ~6 share the phase-space distance matrix. Computing each intermediate once per bin per session is the only way the library stays fast at scale.

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

### The Filter Test

Before shipping any primitive, confirm:

- [ ] Written from first principles, not wrapping a library
- [ ] Expresses as accumulate + gather where possible (kingdom declared otherwise)
- [ ] Registers shareable intermediates via TamSession where appropriate
- [ ] Every parameter tunable, nothing hardcoded beyond documented defaults
- [ ] Hardware details hidden behind tambear-wgpu / tam-gpu
- [ ] Runs on CPU, GPU, and future accelerators with no code changes
- [ ] Honestly declares its Kingdom (A/B/C/D) so TAM knows how to schedule it
- [ ] Is one more piece of "every math, our way, everywhere"

If all eight pass, ship. If any fail, fix first.

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
