# Universal Mathematics Taxonomy

**Author**: math-researcher  
**Date**: 2026-04-08  
**Purpose**: Map every fundamental computational algorithm across all fields of mathematics. For each: core algorithm, accumulate+gather decomposition (if possible), parameters, tambear status.

**Legend**:
- **A+G**: Decomposes into accumulate+gather (parallelizable on GPU)
- **SEQ**: Inherently sequential (Kingdom B — prefix scan or recurrence)
- **ITER**: Iterative convergence (Kingdom C — repeated A+G until fixed point)
- **TREE**: Tree/recursive structure (Kingdom D — divide and conquer)
- **HAVE**: Already in tambear
- **TASK**: Has a task number
- **NEW**: Not yet tracked

---

## I. PURE MATHEMATICS

### 1. Number Theory

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Miller-Rabin primality | Fermat test + strong witnesses | A+G: independent witness tests, gather vote | n_witnesses, bases | HAVE |
| AKS primality | Deterministic polynomial test | SEQ: polynomial arithmetic mod (x^r-1, n) | none (deterministic) | NEW |
| Pollard's rho factoring | Cycle detection on f(x)=x²+c mod n | SEQ: Floyd/Brent cycle detection | c parameter | HAVE |
| Trial division | Divide by primes up to √n | A+G: test each prime independently | limit | HAVE |
| Extended Euclidean | gcd + Bézout coefficients | SEQ: recursion | none | HAVE |
| CRT (Chinese Remainder) | Reconstruct from modular residues | A+G: pairwise combination is associative! | moduli | HAVE |
| Modular exponentiation | Square-and-multiply | SEQ: bit-serial | base, exp, mod | HAVE |
| Discrete logarithm | Baby-step giant-step | A+G: baby steps = scatter, giant steps = gather | group order | NEW |
| Elliptic curve point add | Chord-and-tangent on Weierstrass curve | SEQ: point doubling chain | curve params (a,b,p) | NEW |
| Elliptic curve scalar mul | Double-and-add (like modular exp) | SEQ: bit-serial | scalar, point, curve | NEW |
| Lattice reduction (LLL) | Gram-Schmidt + size reduction + swap | SEQ: column-by-column | delta parameter (0.5-1) | NEW |
| Continued fractions | Floor + reciprocal iteration | SEQ: inherently sequential | depth limit | HAVE (p-adic) |
| Möbius function/inversion | Sieve-like computation | A+G: multiplicative function on prime factorization | none | NEW |
| Euler totient (sieve) | Product formula on primes | A+G: sieve = scatter, totient = gather | limit | HAVE |

**Key insight**: Most number theory is inherently sequential (modular chains), but some operations (CRT, sieve, independent primality witnesses) parallelize beautifully. CRT is literally an associative binary operation — it IS a prefix scan.

### 2. Linear Algebra (beyond what we have)

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Dense mat-mul | O(n³) or Strassen O(n^2.807) | A+G: tiled dot products | tile_size | HAVE |
| SVD | Bidiagonalization + QR iteration | ITER: Golub-Kahan | tolerance | HAVE (via eigendecomp) |
| QR decomposition | Householder reflections | SEQ: column-by-column | pivoting strategy | HAVE |
| LU decomposition | Gaussian elimination with pivoting | SEQ: column-by-column | partial/full pivot | HAVE (Cholesky) |
| Sparse mat-vec | Compressed sparse row × dense vector | A+G: scatter rows, accumulate products | format (CSR/CSC) | NEW |
| Iterative solvers (CG) | Conjugate gradient for Ax=b | ITER: mat-vec per iteration | preconditioner, tol | NEW |
| GMRES | Krylov solver for non-symmetric systems | ITER: Arnoldi + least squares | restart_k, tol | NEW |
| Randomized SVD | Halko-Martinsson-Tropp 2011 | A+G: random projection → small SVD | rank_k, oversampling | NEW |
| Matrix exponential | Padé approximation + scaling/squaring | SEQ: repeated squaring | order | NEW |
| Tensor decomposition (CP) | ALS on rank-R factorization | ITER: alternating least squares | rank_R, tol | NEW |
| Kronecker product | Block multiplication | A+G: embarrassingly parallel | none | NEW |

**Key insight**: Dense linear algebra is mostly SEQ (column operations depend on previous columns). But the BUILDING BLOCKS (dot product, mat-vec) are A+G. Iterative solvers (CG, GMRES) are ITER: each iteration is one A+G pass (mat-vec), and convergence is the outer loop.

### 3. Combinatorics

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Partition function p(n) | Hardy-Ramanujan-Rademacher | SEQ: sum of convergent series | precision | NEW |
| Binomial coefficient | Pascal's triangle or Stirling | A+G: independent for each (n,k) | none | HAVE (via log_gamma) |
| Catalan numbers | C(n) = C(2n,n)/(n+1) | A+G: direct formula | none | NEW (trivial) |
| Stirling numbers (1st/2nd kind) | Recurrence or inclusion-exclusion | SEQ: triangular recurrence | none | NEW |
| Bell numbers | B(n) = Σ S(n,k) | SEQ: via Stirling triangle | none | NEW |
| Derangements | D(n) = (n-1)(D(n-1)+D(n-2)) | SEQ: recurrence | none | NEW (trivial) |
| Permanent (matrix) | Ryser formula O(2^n · n) | A+G: each subset sum is independent | none | NEW |
| Graph coloring number | NP-hard — backtracking | TREE: branch and bound | none | NEW |

**Key insight**: Combinatorics is mostly about recurrences (SEQ) or enumeration (TREE). The individual EVALUATIONS (e.g., C(n,k) for specific n,k) are A+G but the generation of tables is SEQ.

### 4. Topology

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Persistent homology H₀ | Union-find on filtered edges | SEQ: process edges in order | filtration | HAVE |
| Persistent homology H₁ | Boundary matrix column reduction | SEQ: column operations | max_edge | HAVE (approx) |
| Simplicial homology (exact) | Smith normal form on boundary matrices | SEQ: matrix reduction | dimension | NEW |
| Betti numbers | Rank of homology groups | SEQ: from boundary reduction | dimension | NEW |
| Euler characteristic | χ = Σ(-1)^k rank(H_k) = V - E + F | A+G: count simplices per dimension | none | NEW (trivial) |
| Mapper algorithm | Nerve of a cover | A+G: cluster per interval, connect overlaps | n_intervals, overlap | NEW |
| Wasserstein distance (persistence) | Optimal matching between diagrams | SEQ: Hungarian algorithm on diagram points | p (1 or 2) | NEW |
| Bottleneck distance | Max matching distance | SEQ: binary search + bipartite matching | none | NEW |

**Key insight**: Persistent homology H₀ via union-find is already the canonical example of a Kingdom B (sequential merge) operation in tambear. H₁ requires column reduction — inherently sequential. Betti numbers and Euler characteristic are derived quantities.

---

## II. PHYSICS ENGINES

### 5. Classical Mechanics

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Euler integration | x += v·dt, v += a·dt | SEQ: prefix scan (position is cumulative sum of velocities) | dt | HAVE |
| RK4 integration | 4-stage Runge-Kutta | SEQ: 4 sequential evaluations per step | dt | HAVE |
| Verlet (velocity) | Symplectic integrator | SEQ: leapfrog | dt | HAVE |
| N-body (direct) | O(n²) pairwise forces | A+G: tiled_reduce over particle pairs! | G, softening | HAVE |
| N-body (Barnes-Hut) | O(n log n) via octree | TREE: build tree, walk for each particle | theta (opening angle) | NEW |
| N-body (FMM) | O(n) multipole expansion | TREE: upward pass (accumulate multipoles) + downward pass (gather local) | expansion_order | NEW |
| Rigid body dynamics | Euler equations + quaternion integration | SEQ: integrate orientation + angular momentum | inertia tensor | NEW |
| Constraint solver (Gauss-Seidel) | Iterative joint constraint resolution | ITER: project each constraint sequentially | iterations, SOR_param | NEW |
| Hamiltonian dynamics | Leapfrog on (q, p) | SEQ: symplectic | dt, potential function | HAVE (in HMC context) |

**Key insight**: N-body is the poster child for tambear's tiled architecture. Direct N-body IS tiled_reduce. Barnes-Hut and FMM are tree algorithms — a different paradigm. For n < 10,000, direct tiled is competitive with tree methods due to GPU parallelism.

### 6. Quantum Mechanics

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Quantum gate application | Matrix × state vector | A+G: mat-vec on 2^n amplitudes | gate type, qubits | NEW |
| Quantum circuit simulation | Sequence of gate applications | SEQ: gate by gate (state evolves) | circuit depth | NEW |
| Density matrix evolution | ρ' = Σ K_i ρ K_i† (Kraus operators) | A+G: each Kraus term independent | noise model | NEW |
| Schrödinger equation (1D) | -ℏ²/2m ψ'' + V(x)ψ = Eψ → tridiagonal | SEQ: tridiagonal solve (Task #104!) | potential V(x), dx | NEW |
| VQE (variational quantum eigensolver) | Minimize ⟨ψ(θ)|H|ψ(θ)⟩ | ITER: classical optimizer + quantum circuit | ansatz, optimizer | NEW |
| Exact diagonalization | Full Hamiltonian eigendecomposition | A+G: matrix construction is parallel; eigendecomp is SEQ | system size | NEW |
| DMRG (density matrix renormalization) | Variational MPS optimization | ITER: sweep left/right | bond dimension, sweeps | NEW |

**Key insight**: Quantum simulation for n qubits requires 2^n amplitudes — exponential space. For n ≤ 25, direct simulation is feasible (2^25 ≈ 33M amplitudes, fits in GPU memory). The gate application is a mat-vec — pure A+G. The circuit is sequential. This is exactly the Kingdom B pattern: each step is a parallel operation, applied sequentially.

### 7. Statistical Mechanics & Thermodynamics

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Ising model (Metropolis) | Spin flip MC on lattice | ITER: propose flip, accept/reject | temperature, lattice_size | HAVE |
| Wang-Landau sampling | Flat-histogram MC | ITER: adaptive histogram | energy_bins, f_factor | NEW |
| Molecular dynamics (Lennard-Jones) | Verlet + LJ force | A+G: tiled force calculation (like N-body) | epsilon, sigma, dt | NEW |
| Langevin dynamics | SDE: dx = -∇V dt + √(2kT) dW | SEQ: Euler-Maruyama on SDE | temperature, friction | NEW |
| Replica exchange (parallel tempering) | Multiple MC chains at different T, swap | A+G: chains independent between swaps | temperatures[], swap_interval | NEW |
| Partition function (exact, small system) | Z = Σ exp(-βE_i) | A+G: embarrassingly parallel over microstates | temperature | NEW |

**Key insight**: Most MC methods are ITER (propose-accept loop). But the FORCE COMPUTATION within each step is A+G. Langevin dynamics = Brownian motion (already have) + gradient force (A+G).

### 8. Fluid Dynamics

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Lattice Boltzmann (LBM) | Collision + streaming on lattice | A+G: collision is local (map), streaming is gather | viscosity, lattice_type (D2Q9, D3Q19) | NEW |
| Navier-Stokes (spectral) | FFT-based pressure solver | A+G: FFT is tiled butterfly + gather | Re, dt, resolution | NEW |
| Vortex methods | Biot-Savart on vortex particles | A+G: tiled pairwise interaction (like N-body) | viscosity, regularization | NEW |
| Shallow water equations | Finite volume on 2D grid | A+G: flux computation is local; update is scatter | gravity, dx, dt | NEW |
| SPH (smoothed particle hydro) | Kernel-weighted particle interactions | A+G: neighbor search + kernel accumulate | smoothing_length, kernel | NEW |

**Key insight**: CFD methods split into Eulerian (grid-based) and Lagrangian (particle-based). Both have A+G inner loops: grid methods do stencil operations (local gather), particle methods do pairwise interactions (tiled). The time-stepping is SEQ.

### 9. Electrodynamics & Waves

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| FDTD (Yee grid) | Leapfrog on E and H fields | A+G: stencil update is local map | dx, dt, PML_layers | NEW |
| Method of Moments (MoM) | Integral equation → dense linear system | A+G: fill matrix (tiled) + solve (SEQ) | frequency, basis_functions | NEW |
| Ray tracing | Geometric optics | TREE: ray-object intersection, recursive reflection | max_bounces | NEW |
| Helmholtz equation | ∇²u + k²u = f, finite difference | A+G: stencil; solve via CG/GMRES | wavenumber k, boundary conditions | NEW |

---

## III. ENGINEERING & APPLIED MATH

### 10. Control Theory

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| PID controller | P + I·∫e dt + D·de/dt | SEQ: integral is prefix sum, derivative is first difference | Kp, Ki, Kd | NEW |
| LQR (Linear Quadratic Regulator) | Solve Riccati equation for optimal gain | ITER: matrix Riccati recursion (backward) or eigendecomposition | Q, R matrices | NEW |
| Kalman filter | Predict + update on state-space model | SEQ: matrix prefix scan (Särkkä Op!) | F, H, Q, R matrices | TASK #101 |
| Extended Kalman filter | Kalman with linearized dynamics | SEQ: Jacobian at each step + Kalman update | f(x), h(x) functions | NEW |
| Model Predictive Control | Optimize over finite horizon | ITER: QP solver at each time step | horizon, constraints | NEW |

**Key insight**: Kalman filter is THE canonical example of the Särkkä Op framework — it's a matrix prefix scan with 5-tuple state (F, Q, b, H, R). Already identified in project memory. LQR is a backward Riccati recursion — also a matrix prefix scan but in reverse time.

### 11. Finite Elements / PDE Solvers

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| 1D finite differences | Tridiagonal system Ax = b | SEQ: Thomas algorithm = 3×3 matrix prefix scan (Task #104!) | dx, boundary conditions | NEW |
| 2D finite differences (Poisson) | 5-point stencil → sparse system | A+G: stencil is local map; solve via CG/multigrid | dx, dy, boundary conditions | NEW |
| Finite element assembly | ∫ φ_i · L(φ_j) dΩ over elements | A+G: each element integral is independent! Scatter to global matrix. | element type, quadrature order | NEW |
| Multigrid | V-cycle: restrict → solve coarse → prolong → smooth | TREE: recursive coarsening | n_levels, smoother (Jacobi/GS) | NEW |
| Spectral methods | Expand in global basis (Chebyshev, Fourier) | A+G: FFT for evaluation; mat-vec for differentiation | basis, n_modes | NEW |

**Key insight**: FEM assembly is embarrassingly parallel — each element's local stiffness matrix is computed independently, then SCATTERED to the global matrix. This is literally `accumulate(element_matrix, ByKey=node_pair, Add)`. The assembly step IS a scatter operation. Only the solve step (CG, multigrid) is iterative.

### 12. Information & Coding Theory

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Huffman coding | Greedy tree construction | TREE: priority queue | alphabet | NEW |
| Arithmetic coding | Interval subdivision | SEQ: inherently sequential (state-dependent) | probability model | NEW |
| Reed-Solomon encode/decode | Polynomial evaluation/interpolation over GF(2^m) | A+G: evaluate polynomial at each point independently | field_size, n_parity | NEW |
| LDPC decode (belief propagation) | Message passing on factor graph | ITER: variable→check→variable messages | max_iter, damping | HAVE (spectral_gap.rs has Arnoldi) |
| Turbo decode | Iterative BCJR on component codes | ITER: forward-backward on trellis (prefix scan!) | max_iter, interleaver | NEW |
| CRC computation | Polynomial division mod generator | SEQ: bit-serial XOR chain | generator polynomial | NEW |
| Lempel-Ziv (LZ77/LZ78) | Sliding window dictionary | SEQ: inherently sequential (dictionary grows) | window_size | NEW |

**Key insight**: Encoding is usually parallel (evaluate codeword at each position). Decoding is usually iterative (belief propagation, BCJR). CRC is a linear recurrence — it's a prefix scan over GF(2)!

### 13. Optimization (beyond what we have)

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Interior point (LP) | Newton on barrier function | ITER: solve Newton system per step | barrier parameter | NEW |
| Simplex method (LP) | Pivot on vertices of polytope | SEQ: pivot selection is sequential | pricing rule | NEW |
| Branch and bound (IP) | Tree search with LP relaxation bounds | TREE: branching + pruning | branching strategy | NEW |
| Semidefinite programming | Interior point on matrix cone | ITER: Newton on matrix inequality | barrier | NEW |
| Proximal gradient | For composite f(x) + g(x) with g non-smooth | ITER: gradient step + prox operator | step_size, prox function | NEW |
| ADMM | Split optimization via augmented Lagrangian | ITER: x-update, z-update, dual update | rho, tolerance | NEW |
| Evolutionary strategies (CMA-ES) | Covariance matrix adaptation | ITER: sample population, update distribution | population_size, sigma | NEW |
| Particle swarm | Swarm intelligence | ITER: update velocities + positions | n_particles, inertia, cognitive, social | NEW |
| Bayesian optimization | GP surrogate + acquisition function | ITER: fit GP, optimize acquisition, evaluate | kernel, acquisition (EI/UCB/PI) | NEW |

---

## IV. BIOLOGY & CHEMISTRY

### 14. Bioinformatics

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Needleman-Wunsch (global alignment) | DP on score matrix | SEQ: anti-diagonal wavefront parallelism | gap penalty, substitution matrix | NEW |
| Smith-Waterman (local alignment) | DP with zero floor | SEQ: same wavefront pattern | gap penalty, substitution matrix | NEW |
| BLAST (heuristic alignment) | Seed + extend | A+G: seed finding is scatter; extension is local | word_size, e_value | NEW |
| Phylogenetic (neighbor-joining) | Iterative closest-pair merging | SEQ: merge step is sequential | distance matrix | NEW |
| HMM (Viterbi/forward-backward) | DP on hidden state trellis | SEQ: prefix scan with matrix multiply! (Särkkä Op) | transition matrix, emission probs | NEW |
| Gillespie SSA | Exact stochastic simulation of reactions | SEQ: exponential waiting times | reaction rates | NEW |
| Michaelis-Menten kinetics | v = V_max [S] / (K_m + [S]) | A+G: pointwise evaluation | V_max, K_m | NEW (trivial) |

**Key insight**: Sequence alignment (NW, SW) has anti-diagonal parallelism — each anti-diagonal of the DP table can be computed in parallel. This is a wavefront pattern that maps to segmented prefix scan. HMM forward-backward IS a matrix prefix scan — same as Kalman filter but with discrete state space.

---

## V. ECONOMICS & FINANCE (beyond current)

### 15. Option Pricing

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Black-Scholes (European) | Closed-form N(d1), N(d2) | A+G: pointwise | S, K, T, r, σ | HAVE |
| Binomial tree (American) | Backward induction on recombining tree | SEQ: backward pass through tree levels | n_steps, up/down factors | NEW |
| Monte Carlo pricing | Simulate paths, average payoff | A+G: paths are independent! | n_paths, n_steps, payoff function | NEW |
| Finite difference (Black-Scholes PDE) | Crank-Nicolson on BS PDE | SEQ: tridiagonal solve per time step | n_S, n_t, boundary conditions | NEW |
| Greeks (sensitivities) | ∂price/∂param via finite differences or adjoint | A+G: each Greek is independent | bump_size | HAVE (delta only) |

### 16. Portfolio & Risk

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Markowitz mean-variance | min w'Σw s.t. w'μ = target, w'1 = 1 | A+G: covariance matrix + QP solve | target_return, constraints | NEW |
| Black-Litterman | Bayesian update on market equilibrium | A+G: matrix operations | tau, views matrix P, confidence Ω | NEW |
| Risk parity | Equal risk contribution: w_i (Σw)_i = c | ITER: Newton on risk contribution equations | risk_budget | NEW |
| Value at Risk (VaR) | Quantile of loss distribution | A+G: sort + quantile (or parametric: normal quantile) | confidence level | NEW |
| Expected Shortfall (CVaR) | E[loss | loss > VaR] | A+G: conditional mean beyond quantile | confidence level | NEW |
| Copula fitting | Joint distribution with arbitrary marginals | ITER: ML on copula parameters | copula family, marginals | NEW |

---

## VI. MACHINE LEARNING (beyond neural ops)

### 17. Tree-Based Methods

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Decision tree (CART) | Recursive binary splitting on features | TREE: find best split at each node | max_depth, min_samples_split, criterion | NEW |
| Random forest | Bootstrap + random feature subsets + vote | A+G: each tree is independent! | n_trees, max_features, max_depth | NEW |
| Gradient boosting | Sequential additive model | SEQ: each tree fits residuals of previous | n_trees, learning_rate, max_depth | NEW |
| XGBoost-style | Regularized gradient boosting with histogram | SEQ (trees) + A+G (histogram binning) | lambda, gamma, n_bins | NEW |

**Key insight**: Random forest is embarrassingly parallel — each tree is independent. This is `accumulate(predictions, All, Vote)` where each tree is an independent gather. Gradient boosting is inherently sequential (each tree depends on the previous residuals) — Kingdom B.

### 18. Kernel Methods

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| SVM (linear) | Max margin via hinge loss | ITER: (sub)gradient descent | C (regularization) | NEW |
| SVM (kernel, SMO) | Sequential minimal optimization | ITER: select working set, solve 2-variable QP | C, kernel, gamma | NEW |
| Kernel ridge regression | (K + λI)⁻¹ y | A+G: kernel matrix (tiled!) + solve (SEQ) | lambda, kernel | NEW |
| Gaussian process regression | μ* = K*'(K + σ²I)⁻¹y | A+G: kernel matrix (tiled!) + solve (SEQ) | kernel, noise_variance | NEW |
| Sparse GP (inducing points) | Low-rank approximation via m inducing points | A+G: K_nm is tiled; solve is O(m³) not O(n³) | m_inducing, kernel | NEW |

**Key insight**: Kernel methods center on the kernel matrix K_ij = k(x_i, x_j). This IS a tiled computation — exactly what TiledEngine does for distance matrices. The kernel is just a different function on pairs. A single new TiledOp (KernelOp with configurable kernel function) would unlock ALL kernel methods.

### 19. Probabilistic Models

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| EM (generic) | E-step (posterior) + M-step (maximize) | ITER: E-step is A+G (compute responsibilities), M-step is A+G (accumulate sufficient stats) | convergence_tol | NEW |
| Variational inference (mean-field) | Optimize q(z) ≈ p(z|x) via ELBO | ITER: coordinate ascent on variational parameters | n_iter | NEW |
| LDA (topic model) | Collapsed Gibbs or variational EM | ITER: sample/update topic assignments | n_topics, alpha, beta | TASK #115 |
| Naive Bayes | Class-conditional likelihoods | A+G: count features per class (scatter!), predict (gather) | smoothing (Laplace) | TASK #116 |
| CRF | Structured prediction via forward-backward | SEQ: forward-backward = prefix scan! | feature templates | TASK #117 |

**Key insight**: EM is the universal ITER pattern: E-step = accumulate(responsibilities, ByKey=component, posterior), M-step = accumulate(sufficient_stats, ByKey=component, Add). GMM, LDA, HMM training — all EM. The forward-backward algorithm in HMM/CRF is a matrix prefix scan.

---

## VII. GEOMETRY & GRAPHICS

### 20. Computational Geometry

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Convex hull (2D, Graham scan) | Sort by angle, stack-based scan | SEQ: scan is inherently sequential | none | NEW |
| Convex hull (2D, quickhull) | Divide-and-conquer | TREE: partition + recurse | none | NEW |
| Voronoi diagram | Fortune's sweep line or Delaunay dual | SEQ: sweep line | none | NEW |
| Delaunay triangulation | Bowyer-Watson incremental insertion | SEQ: insert point, re-triangulate | none | NEW |
| Point-in-polygon | Ray casting or winding number | A+G: each query point is independent | none | NEW |
| K-d tree construction | Recursive median splitting | TREE: divide by median at each level | leaf_size | NEW |
| K-d tree query (NN) | Backtracking search | TREE: prune by distance bounds | k neighbors | NEW |
| Boolean operations (CSG) | Intersection/union/difference of polygons | SEQ: sweep line + event processing | operation type | NEW |

### 21. Differential Geometry

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Geodesic equation | d²x^μ/ds² + Γ^μ_αβ dx^α/ds dx^β/ds = 0 | SEQ: ODE integration (RK4) | metric tensor g_μν | NEW |
| Christoffel symbols | Γ^μ_αβ = ½ g^μλ (∂_α g_βλ + ∂_β g_αλ - ∂_λ g_αβ) | A+G: each component is independent | metric tensor | NEW |
| Riemann tensor | R^ρ_σμν from Christoffel symbols | A+G: each component from Γ | metric tensor | NEW |
| Gaussian curvature | K = det(II) / det(I), or K = R/2 in 2D | A+G: pointwise from metric | surface parametrization | NEW |
| Parallel transport | Solve ∇_γ V = 0 along curve γ | SEQ: ODE along curve | connection, curve | NEW |
| Exponential map | exp_p(v) = γ(1) where γ'(0) = v | SEQ: geodesic integration | base point, tangent vector | NEW |
| Poincaré disk operations | Möbius transformations | A+G: pointwise | curvature | HAVE (manifold.rs) |

---

## VIII. CROSS-CUTTING PATTERNS

### The Accumulate+Gather Classification

After surveying ~200 algorithms across all fields, the decomposition patterns are:

| Pattern | Fraction | Examples |
|---------|----------|---------|
| **A+G (embarrassingly parallel)** | ~25% | N-body forces, kernel matrices, random forest trees, element integrals, Monte Carlo paths |
| **SEQ (prefix scan / recurrence)** | ~35% | Kalman filter, GARCH, CRC, DP alignment, time integration |
| **ITER (fixed-point iteration)** | ~25% | EM, Newton-Raphson, CG/GMRES, belief propagation, optimization |
| **TREE (divide and conquer)** | ~15% | Barnes-Hut, quicksort, multigrid, decision trees, convex hull |

**The fundamental insight**: ITER methods are sequences of A+G steps. Each iteration of EM, CG, Newton is an A+G pass. The outer loop is sequential. So the real question is: how many A+G passes until convergence?

**SEQ methods often hide A+G**: A prefix scan IS parallel — it's O(log n) parallel steps via the Blelloch algorithm. The Särkkä Op framework makes this explicit: any associative binary operation on a sequence is a prefix scan, which is O(n) work in O(log n) depth.

### What Tambear Should Prioritize

**Tier 1 — Already have the infrastructure, just need wrappers:**
- Kernel methods (kernel matrix = TiledEngine with different function)
- FEM assembly (scatter to global matrix)
- Monte Carlo pricing (independent path simulation)
- Random forest (independent trees)
- Naive Bayes (scatter-based counting)

**Tier 2 — Need new primitives but map cleanly:**
- Sparse mat-vec (new memory access pattern)
- Lattice Boltzmann (stencil operation = local gather)
- Quantum gates (small dense mat-vec on 2^n state)
- FDTD (stencil on 3D grid)

**Tier 3 — Genuinely new paradigms:**
- Tree algorithms (Barnes-Hut, decision trees, multigrid) — need tree traversal primitive
- Constraint solvers (sequential projection)
- Graph algorithms that aren't reducible to matrix operations

---

## IX. CRYPTOGRAPHY

### 22. Symmetric Cryptography

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| AES (Rijndael) | SubBytes + ShiftRows + MixColumns + AddRoundKey | A+G: each block is independent (ECB); chained in CBC (SEQ) | key_size (128/192/256), mode (ECB/CBC/CTR/GCM) | NEW |
| ChaCha20 | ARX (add-rotate-xor) quarter-rounds | A+G: each block counter is independent in CTR mode | key, nonce, counter | NEW |
| SHA-256 | Merkle-Damgård with compression function | SEQ: each block depends on previous hash state | none (fixed) | NEW |
| SHA-3 (Keccak) | Sponge construction with 1600-bit state | SEQ: absorb phase is sequential | capacity, rate | NEW |
| BLAKE3 | Merkle tree of BLAKE2s compressions | TREE: leaf compressions are parallel! Parent nodes depend on children | key (optional) | NEW |
| HMAC | H(K xor opad, H(K xor ipad, message)) | SEQ: two hash invocations | hash function, key | NEW |
| Poly1305 | Polynomial evaluation mod 2^130 - 5 | A+G: can be parallelized via Horner decomposition | key (r, s) | NEW |

**Key insight**: Block ciphers in CTR/ECB mode are embarrassingly parallel — each block is encrypted independently. In CBC mode, each ciphertext block depends on the previous one — inherently sequential. BLAKE3 is explicitly designed for parallelism via its Merkle tree structure.

### 23. Asymmetric Cryptography & Key Exchange

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| RSA key generation | Generate two large primes, compute n=pq | A+G: primality testing is independent per candidate | key_bits (2048/4096) | NEW |
| RSA encrypt/decrypt | Modular exponentiation c = m^e mod n | SEQ: square-and-multiply chain | (n, e) or (n, d) | NEW (have mod_exp) |
| ECDSA sign/verify | Scalar multiplication on elliptic curve + hash | SEQ: point doubling chain | curve (P-256, secp256k1), hash | NEW |
| ECDH key exchange | Shared secret = scalar × peer's public point | SEQ: scalar multiplication | curve | NEW |
| X25519 / Ed25519 | Montgomery ladder / Edwards curve | SEQ: constant-time scalar mul | none (fixed curve) | NEW |
| Diffie-Hellman | g^a mod p (discrete log hardness) | SEQ: modular exponentiation | prime p, generator g | NEW |
| Lattice crypto (Kyber/CRYSTALS) | Learning With Errors (LWE) on module lattices | A+G: matrix-vector products mod q are parallel | dimension n, modulus q | NEW |

### 24. Hash Functions & Random Oracles

| Algorithm | Core idea | Decomposition | Params | Status |
|-----------|-----------|---------------|--------|--------|
| Merkle tree | Hash leaves, then parent = H(left || right) | TREE: leaves are parallel, tree reduction | hash function, leaf_size | NEW |
| Password hashing (Argon2) | Memory-hard function | SEQ: memory-dependent access pattern (by design) | time_cost, memory_cost, parallelism | NEW |
| Pedersen commitment | g^v · h^r (homomorphic hiding) | A+G: each commitment is independent | generators g, h | NEW |

**Sharing opportunity**: Merkle tree construction is a TREE reduction with an associative hash combine. This is structurally identical to a segmented reduction — the leaves are the data blocks, the combine operation is H(left || right). A Merkle tree IS an accumulate(blocks, Tree, HashCombine).

---

## X. INTERMEDIATE SHARING MAP (Cross-Algorithm)

For each major intermediate type, which algorithms across ALL fields produce and consume it:

### Distance/Kernel Matrices (n×n pairwise)

| Producer | Consumer | Field |
|----------|----------|-------|
| TiledEngine (L2, Cosine, Dot) | DBSCAN, KNN, silhouette | Statistics |
| Kernel evaluation (RBF, Matern, polynomial) | SVM, GP regression, kernel PCA | ML |
| Pairwise forces (Lennard-Jones, Coulomb) | N-body, molecular dynamics | Physics |
| Variogram (spatial covariance) | Kriging | Spatial stats |
| Sequence alignment scores (NW, SW) | Phylogenetics, clustering | Biology |
| Graph adjacency (weighted) | Shortest paths, centrality, community | Graph theory |

**Sharing rule**: ANY pairwise function f(x_i, x_j) produces the same n×n matrix structure. A single `TiledOp` trait with pluggable kernel functions unifies ALL of these.

### Eigendecompositions

| Producer | Consumer | Field |
|----------|----------|-------|
| PCA (covariance eigendecomp) | Scree, Kaiser, explained variance | Statistics |
| Spectral clustering (Laplacian eigendecomp) | Cluster assignment | ML |
| Quantum Hamiltonian (eigenvalues = energy levels) | Energy spectrum, ground state | Quantum |
| Vibration analysis (mass-stiffness eigendecomp) | Natural frequencies, mode shapes | Engineering |
| Google PageRank (dominant eigenvector) | Node importance | Graph theory |
| Arnoldi/Lanczos (sparse top-k) | Spectral gap, principal components | Numerical LA |

**Sharing rule**: Eigendecomposition is expensive (O(n³) dense, O(nk²) sparse). ANY algorithm that produces eigenvalues/vectors of a SPECIFIC matrix can share with any consumer of the SAME matrix's spectrum. Tag must include which matrix.

### Factorizations (QR, LU, Cholesky, SVD)

| Producer | Consumer | Field |
|----------|----------|-------|
| OLS regression (QR of design matrix X) | Cook's distance, leverage, VIF | Statistics |
| Cholesky of covariance (Σ = LL') | Sampling multivariate normal, Mahalanobis | Statistics/ML |
| SVD of data matrix | PCA, low-rank approximation, pseudoinverse | Everywhere |
| LU of stiffness matrix | FEM solve, condition number | Engineering |
| Cholesky of kernel matrix (K = LL') | GP posterior, SVM dual | ML |

**Sharing rule**: Matrix factorizations are the most expensive shared intermediates. A Cholesky computed for GP regression is the SAME Cholesky needed for sampling from the posterior. Tag: `MatrixFactorization { matrix_id, factorization_type }`.

### FFT / Spectral Representations

| Producer | Consumer | Field |
|----------|----------|-------|
| FFT of signal | Power spectrum, filtering, convolution | Signal processing |
| FFT of autocorrelation | Spectral density (Wiener-Khinchin) | Time series |
| FFT of potential | Poisson solver (spectral method) | Physics/PDE |
| FFT of kernel | Fast convolution (O(n log n)) | ML/signal |
| NTT (number theoretic transform) | Polynomial multiplication, Reed-Solomon | Crypto/coding |

**Sharing rule**: FFT of a signal is reusable for ANY spectral operation on that signal. The inverse FFT recovers the original. Tag: `SpectralRepresentation { data_id, n_points }`.

### Gradient / Jacobian / Hessian

| Producer | Consumer | Field |
|----------|----------|-------|
| OLS gradient (X'(y - Xβ)) | Gradient descent, L-BFGS | Optimization |
| Neural network backward pass | Weight updates (SGD, Adam) | ML |
| Cox PH gradient (S₀, S₁) | Schoenfeld residuals, SE | Survival |
| GARCH gradient (∂ℓ/∂θ) | Parameter estimation | Finance |
| Finite element residual | Newton iteration for nonlinear FEM | Engineering |

**Sharing rule**: Gradients are the most frequently recomputed intermediate. If the parameters haven't changed, the gradient is reusable. But gradients become stale the moment parameters update. Tag needs a version counter: `Gradient { objective_id, param_version }`.

---

_This taxonomy is a living document. Each entry that gets implemented should be checked off and linked to its tambear module._

_Last updated: 2026-04-08 by math-researcher. Covers ~230 algorithms across 24 sub-fields._
