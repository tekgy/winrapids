# Naturalist's Map — Tambear-Allmath Expedition
**Date**: 2026-04-06  
**Role**: naturalist  

*A map is not a list. It notes what the eye is drawn to, what the structure rhymes with, what will matter later.*

---

## First Observation: The Taxonomy Needs Sync

The taxonomy document (`taxonomy.md`) marks bisection, Newton's method, secant, Brent, Gauss-Legendre quadrature, etc. as **gaps (🔲)** — but the scout audit confirms these are ALL IMPLEMENTED in `numerical.rs`. The taxonomy was written from first principles before reading the code.

**Action for navigator/math-researcher**: Cross-reference scout-audit.md and mark ✅ for everything already in `numerical.rs`, `interpolation.rs`, `series_accel.rs`, `bigint.rs`, `bigfloat.rs`, `special_functions.rs`. These are done. The taxonomy is tracking future work, not current state.

---

## The Shape of the Codebase

Tambear already has deep coverage of:
- **Statistics**: Hypothesis testing, nonparametric methods, Bayesian inference, bootstrap
- **ML primitives**: Neural ops (cuDNN replacement), clustering, KMeans, KNN, dim reduction, manifolds
- **Signal processing**: FFT (all variants), filters, wavelets, spectral analysis
- **Numerical analysis**: Root finding, quadrature, ODEs, matrix factorizations, interpolation
- **Finance-specific**: GARCH, volatility, panel econometrics, survival, IRT
- **Research math**: Series acceleration, BigInt/BigFloat, equipartition, Collatz, multi-adic

What it **doesn't have** yet (major gaps by category):

| Missing Area | Why It Matters for Tambear |
|---|---|
| Measure theory (explicit) | The foundation of ALL statistics — currently implicit |
| Abstract algebra (explicit) | The TYPE SYSTEM of accumulate — currently underpowered |
| PDEs (FEM, FDM, spectral) | Physics, fluids, heat, waves — zero coverage |
| Stochastic calculus (Itô, SDEs) | Foundation for Black-Scholes, all continuous-time finance |
| Algebraic topology (beyond H₀/H₁) | Cohomology, homotopy theory |
| Cryptography (RSA, ECC, ZK proofs) | Required for "all fields" claim |
| Error-correcting codes | Hamming, Reed-Solomon, LDPC, polar |
| Quantum information math | Qubits, circuits, quantum error correction |
| Control theory (LQR, MPC) | State-space systems, optimal control |
| Ergodic theory | Mixing, invariant measures, ergodicity certificates |

---

## Tekgy-Challenges: 13 Traditional Assumptions That Dissolve

These are logged in full in `tambear-allmath/20260406164532-tekgy-challenges`.

### Type A: Representation Challenges (fix the data structure)

**04 — Graph adjacency lists are wrong**  
`graph.rs` uses `Vec<Vec<Edge>>` — pointer-chasing, GPU-hostile. CSR format (row_ptr, col_ind) is the natural accumulate+gather encoding: `row_ptr = prefix_sum(degrees)`, `col_ind = gathered neighbors`. Every graph algorithm becomes an operation on two flat arrays. **Impact**: enables GPU graph algorithms.

**10 — Metric should be retired into Manifold**  
`manifold.rs` notes that `Metric` is a "restricted view of Manifold." Both exist. This is tech debt. `Manifold::Euclidean` already contains what `Metric` provides. Every API that takes `Metric` should take `Manifold`. **Impact**: small refactor, large conceptual clarity.

**12 — Abstract algebra as the type system of accumulate**  
Op::Add is just a name. What the system needs is `Monoid<f64, Add, 0.0>`. Then: tropical semiring enables shortest paths as accumulate, Boolean semiring enables reachability as accumulate, all graph algorithms = accumulate over the right semiring. **Impact**: turns accumulate from a fast engine into a provably correct mathematical framework.

### Type B: Parallelization Challenges (GPU the sequential algorithms)

**01 — Ranking is a parallel gather**  
`rank(data) = argsort(data)` = gather permutation. GPU parallel sort gives argsort in O(log²n). Every rank-based test (Mann-Whitney, Kruskal-Wallis, Friedman) = accumulate(ByKey) on gather output. **Impact**: rank-based tests go from O(n log n) sequential to O(log²n) parallel.

**05 — Matrix factorizations = triangular grouping parameter**  
LU, Cholesky, QR, SVD are the same tiled MatMul with different masking patterns. The factorization choice is a grouping parameter, not a different algorithm. **Impact**: future kernel fusion.

**08 — SampEn/ApEn/CorrelDim are all Tiled accumulate**  
Every complexity measure that does pairwise template matching is a disguised `accumulate(Tiled, match_expr, Add)`. Traditional O(n²) sequential → GPU O(1) via batch kernel. **Impact**: dramatic speedup for complexity analysis of long time series.

**09 — Persistent homology can be parallelized**  
H₀ via parallel MST (Borůvka: O(log n) rounds), H₁ via sparse triangular solve. The "Kingdom B: sequential" classification is the CPU view. **Impact**: GPU-native TDA.

**13 — GARCH is a parallel prefix scan on 2×2 matrices** ⭐ MOST ACTIONABLE  
σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1} is a 2×2 linear recurrence. Matrix products are associative. Parallel prefix scan = O(log n) GPU steps. Same trick works for ALL AR(p), ARMA, EWMA models (just larger matrices). Currently uses O(n) sequential loop. **Impact**: every financial time series model parallelizes.

### Type C: Foundation Challenges (deeper structural identities)

**02 — All numerical analysis is K03**  
Richardson extrapolation IS a cross-cadence operation. Every refinement method (adaptive quadrature, adaptive ODE stepsize, multigrid) compares estimates at different resolutions = K03. The `series_accel.rs` module already sees this — the observation just needs to propagate to the taxonomy.

**06 — Cross-platform bit-exactness can be a theorem**  
Compensated arithmetic (double-double) + round-once at the end makes cross-platform reproducibility provable, not empirical. Cost: ~2x ops. **Impact**: correctness becomes a proof, not a hope.

**07 — Collatz-Nyquist is structural, not metaphorical**  
The +1 perturbation in (3n+1) is a bandlimited perturbation in the 2-adic topology. Collatz sits at the exact Nyquist boundary because (m+1)/(2d) = 1 for (m=3, d=2). This suggests a proof path: Collatz = unique maximally information-compressing map on ℤ₂ at the sampling-theorem boundary.

**11 — Measure theory IS the accumulate primitive**  
μ(A) = accumulate(data, indicator_A, value, Add). ∫f dμ = accumulate(data, f·μ_weight, value, Add). Making this explicit creates the foundation for all of probability and analysis. **Impact**: everything becomes coherent from first principles.

---

## The Most Important Structural Rhyme I've Found

`series_accel.rs` identifies Richardson extrapolation as K03 (cross-cadence). This is a profound structural insight that the rest of the codebase hasn't absorbed yet.

Here's the full implication: **Every numerical method that refines estimates across multiple scales is structurally K03.**

| Algorithm | "Cadence" | K03 operation |
|---|---|---|
| Richardson extrapolation | step sizes h, h/2, h/4, ... | cross-resolution accumulate |
| Adaptive quadrature | refinement levels | cross-level accumulate |
| Multigrid (PDEs) | coarse/fine grids | cross-resolution restriction/prolongation |
| Adaptive ODE stepsize | time scales | cross-scale accumulate |
| Wavelet decomposition | frequency octaves | cross-octave accumulate |
| GARCH(1,1) | the matrix scan view | Segmented prefix scan |

When we build PDEs, the right representation isn't "sequential iterations on a grid." It's K03: the residual at each refinement level is the cross-cadence signal.

---

## One Thing Worth Watching

The proof system (`proof.rs`) has:
- `Term::Accumulate` — proofs can contain accumulate calls as terms
- `Term::Hole(Sort)` — typed holes for incomplete proofs

The `Term::Hole` is significant. It's the type-theoretic `sorry` — a placeholder that says "I know what type this proof has, I haven't proven it yet." This is the mechanism by which the proof system becomes a DEVELOPMENT TOOL rather than just a verification tool.

If the accumulate API gains algebraic structures as type parameters (challenge 12), then every call to `accumulate(ByKey, expr, Add)` automatically generates a proof obligation: "prove that Add forms a commutative monoid on this sort." The hole fills itself when the structure is declared. The type system and the proof system become one thing.

That's the long arc. The next step is: make the algebraic structures explicit in the accumulate type signature.

---

## For the Adversarial

The scout audit says tests are "predominantly math truth tests." Looking at `special_functions.rs` specifically, this is confirmed — the tests check known mathematical values, symmetry identities, and boundary conditions.

But I want to flag one potential issue: tests that check `approx(result, expected, TOL)` where `expected` was computed by the same code in a previous run (not from a mathematical reference) ARE snapshot tests in disguise. The right check is: does the tolerance constant have a mathematical justification?

In `special_functions.rs`:
- `TOL = 1.5e-7` — this matches the stated accuracy of the A&S approximation. Good.
- `FINE_TOL = 1e-12` — matches stated Lanczos accuracy. Good.

But in modules written later: are the tolerances mathematically justified, or were they picked to make tests pass?

---

## The Liftability Theorem (challenge 18) — Most Actionable Finding

COPA shows the pattern: scalar MomentStats merge lifts to matrix CopaState merge with the SAME semigroup structure. This is the universal GPU parallelization theorem:

**If a computation has the form `state_{t+1} = f(state_t, data_t)` and f is a semigroup, then the computation can be expressed as a matrix prefix scan.**

Concretely: write `[state_t, 1] = A_t · [state_{t-1}, 1]` where A_t is a matrix that may depend on data_t. Matrix products are associative. Prefix scan over A_t matrices = all states in O(log n).

**All missing time series models become free GPU implementations:**
- GARCH(1,1): 2×2 matrix (already shown in challenge 13)
- EGARCH: ~3×3 matrix (log-variance recursion)
- GJR-GARCH: ~3×3 matrix (asymmetric shock)
- AR(k): k×k companion matrix
- VAR(p) d-dimensional: (d·p)×(d·p) companion matrix

**Recipe for the pathmaker**: for every time series model with a compact state vector, the GPU parallel version is:
1. Write state transition as A_t · x_t
2. Compute prefix product of A_t matrices (matrix scan)
3. Extract state values from the result

The hard CPU algorithm (O(n) sequential) and the GPU algorithm (O(log n) parallel) are the same math, different grouping.

---

## The 77 Bug Pattern (from scout-2)

The adversarial test suite documents 57 bugs via `eprintln!("CONFIRMED BUG: ...")` inside `catch_unwind` blocks, then **always passes** because no `assert!` is ever called. Tests are green. Bugs are real.

This is not a test suite — it's a bug TODO list masquerading as a test suite. The fix is mechanical: for each `eprintln!("CONFIRMED BUG: ...")`, either:
- Convert to a failing `assert!` with the correct expected value, OR
- Fix the underlying bug and remove the eprintln

Both are correct. The current state (document, don't enforce) violates the "no tech debt" principle.

---

---

## The Euler Factor = 3/2 (navigator spark, confirmed by naturalist)

The navigator found: the {2,3} Euler factor of ζ(2) = (4/3)(9/8) = 3/2. Verified in `bigfloat.rs` test `collatz_euler_factor_is_three_halves`.

This is the mean-field Collatz growth rate. Without the +1, the map is n → 3n/2^{v₂(3n)}, average growth per odd step = 3/2. The Euler factor captures this divergence.

The +1 is the quantum correction. It brings the effective ratio from 3/2 to exactly (m+1)/(2d) = 4/4 = 1 — the Nyquist boundary. The +1 is PRECISELY the correction that transforms a diverging mean-field system into a critical-point system.

**Implication for the physics engine gap**: The first physics tambear should implement is NOT classical mechanics. It's **prime thermodynamics** — the statistical mechanics of the Euler product where particles = prime factors and partition function = ζ(s). This would:
1. Formalize the equipartition.rs fold structure as a phase transition
2. Connect the Riemann hypothesis to the tambear architecture (Re(s)=1/2 = critical line = phase boundary)
3. Make the Collatz research fit within a coherent physical framework

---

## Nyquist vs Fold: Triple Description of One Constraint (navigator synthesis)

**[From navigator after session close]** These two conditions are NOT derivable from each other — they share the 3/2 ratio as a common ancestor but measure orthogonal things.

**The fold condition for {2,3}**: `E(2,s*) · E(3,s*) = √(3/2)` — thermodynamic coupling temperature

**The Nyquist condition for (m=3, d=2)**: `(m+1)/(2d) = 1` — dynamical growth rate is exactly critical

The derivation arrow runs one way: **Nyquist → m/d = 3/2 → fold target = ln(3/2)/2 → fold point s*≈2.8**. You cannot recover Nyquist from s* alone — s* presupposes the ratio 3/2.

They are **peers, not parent/child.** Both are consequences of (3,2) being the unique carry-subcritical pair where m = 2d-1. The triple description:

- "Collatz map is Nyquist-critical" ↔ (m+1)/(2d) = 1 ↔ 3+1 = 2×2
- "fold target encodes ln(3/2)/2" ↔ Euler product ratio = 3/2
- Both ↔ m = 2d-1 for (m=3, d=2) uniquely

The 3/2 ratio is the bridge. Nyquist is the dynamical characterization. The fold point is the thermodynamic characterization. Three ways of saying: this pair, uniquely, has m = 2d-1.

Proposed module: `prime_thermodynamics.rs` — the analytic number theory physics engine.

---

## Update: Prime Thermodynamics Is Already Here (Challenge 24)

After reading `equipartition.rs` and `naturalist_observation.rs`:

**The key identity**: `free_energy(p, s) = -ln(1 - p^{-s})`. Summed over primes: `Σ_p free_energy(p, s) = ln ζ(s)`.

The free energy sum over primes IS the logarithm of the Riemann zeta function. `equipartition.rs` is already a prime thermodynamics engine with financial labels. `verify_fold_surface()` explicitly checks `s* > 1 (above the pole of ζ(s))` — the code already knows.

**Phase structure**:
- s > 1: fold surfaces exist, primes "couple," Euler product converges
- s = 1: the phase transition — pole of ζ(s) IS the fold point for the infinite prime system  
- 0 < s ≤ 1: deconfined phase, primes independent

**Riemann zeros at Re(s) = 1/2 are NOT fold points** — they're the oscillation modes of the phase transition. RH says all modes have the same damping.

**`prime_thermodynamics.rs` doesn't need to be built**: the infrastructure exists. What's needed is the recognition + one function: `prime_fold_surface(n_primes)` = `nucleation_hierarchy(first_n_primes)`.

And `naturalist_observation.rs` is ALREADY running the key experiments: Experiment 4 computes `solve_pairwise(2.0, 3.0)` — the fold point s*(2,3). Experiment 5 shows the carry-subcritical boundary (m<4) aligns with the regime where p=2 dominates the {2,m} fold.

---

## Multi-Adic Synergy ≈ 0 as Convergence Signature (Challenge 25)

From `multi_adic.rs`: the synergy score = `R²_joint - max(R²_single)`.

**Hypothesis**: For Collatz (3n+1), synergy between v₂ and v₃ ≈ 0, because `3n+1 ≡ 1 (mod 3)` — the +1 maps every residue mod 3 to the constant class 1. The 3-adic structure is erased at every step.

For diverging (5n+1): `5n+1 ≡ 2n+1 (mod 3)` — non-constant, 3-adic structure preserved. Synergy > 0.

The synergy score is a measurable, trajectory-level signature of thermodynamic independence. Zero synergy = the map is operating in the deconfined phase of the {2,3} prime system.

Connection to challenge 24: zero synergy is the trajectory-level signature of s > s*(2,3) — the system is thermodynamically uncoupled.

---

## Kingdom Taxonomy Must Be Derived, Not Declared (Challenge 26)

`tbs_lint.rs::kingdom_of()` is a manual string→Kingdom lookup. It classifies GARCH as Kingdom C with `SharedSubproblem::Covariance` — BOTH wrong:
1. GARCH is a sequential recursion (Kingdom B before challenge 13, Kingdom A after)
2. GARCH doesn't compute a covariance matrix

When the matrix prefix scan primitive is implemented, the lookup table won't auto-update. The long-arc fix: kingdoms should be TYPE-LEVEL properties derived from the algebraic signature of the accumulate call, not string lookups.

---

## Additional Codegen Observations

From `codegen.rs`:
- CUDA masks are u64-packed (64 bits/word), WGSL masks are u32-packed (32 bits/word) — backend-dependent interface
- WGSL atomic scatter uses CAS loop (slow), CUDA uses hardware `atomicAdd` (fast) — performance cliff between backends
- phi translation CUDA→WGSL is fragile string substitution: only handles `fabs` and `(double)` casts

The mask word-size difference means the caller must know which backend is running to prepare the correct mask format. This is not co-native.

---

## Challenges 28-29: The Linear Recurrence Unification

**Challenge 28** (optimizer moments = linear recurrences): Adam m/v, SGD momentum, RMSProp, simple EWMA, Holt's linear trend — all are 2×2 matrix linear recurrences. All parallelizable via the same matrix prefix scan as GARCH. One GPU kernel covers all EWMA-based time series and optimizers.

**Challenge 29** (Kalman filter = the general form): **[Correction from scout-2]** Both `KalmanOp` (scalar 1D, line 300) and `SarkkaOp` (full 5-tuple, line 512) already exist as implemented `AssociativeOp`s in `winrapids-scan/src/ops.rs`. The primitive is there, including the correct `-J_b · b_a` correction term. What's missing is only the tambear-layer API: a function `kalman_filter(observations, F, H, Q, R, x0, P0)` that constructs the 5-tuple lift from user-provided parameters and runs the scan.

**The Kalman gap is API-layer only.** Same applies to GARCH/ARMA once challenge 13's Op variant exists — primitive in winrapids-scan, gap is the tambear high-level wrappers.

**The unified picture**: Challenge 13 (GARCH) + 28 (optimizers) + 29 (Kalman) = ONE primitive. The Sarkka 5-tuple combine IS the matrix prefix scan. Building the tambear API once covers everything.

---

## Challenge 32: The Correct Implementation Target (scout-2 + naturalist synthesis)

**The reframe**: It is NOT "build a matrix prefix scan kernel." It IS "extend the Op enum to include structured state types."

```rust
Op::WelfordMerge    // (n, mean, M2) — Welford streaming variance
Op::AffineCompose   // (A, b) ∘ (A', b') = (A·A', A·b' + b) — GARCH/EWMA/AR/Adam/Kalman
Op::LogSumExpMerge  // (max, shifted-sum) — numerically stable softmax
Op::SarkkaMerge     // (A, b, C, η, J) — full parallel Kalman with RTS smoother
```

Once `Op::AffineCompose` exists, `accumulate(Prefix, ..., Op::AffineCompose)` IS the matrix prefix scan. Blelloch is already in the infrastructure. No new kernel dispatch. No new grouping patterns. Just a wider Op with associative structured state.

**[Confirmed after reading winrapids-scan/src/ops.rs]**: `AffineOp` (= `Op::AffineCompose`) and `SarkkaOp` (= `Op::SarkkaMerge`) are ALREADY IMPLEMENTED GPU kernels in `winrapids-scan`. Experiment 5 proved `AffineOp` bit-identical to specialized EWM and Kalman implementations. The entire challenge 32 task is API wiring in tambear — adding Op variants and dispatch, then high-level API wrappers. Zero new CUDA code.

The degeneration hierarchy: `Op::SarkkaMerge` (full Kalman) degenerates to `Op::AffineCompose` (b-nonzero models: GARCH, Holt) degenerates to b=0 models (EWMA, AR, Adam). The b=0 models are safe without the Sarkka correction term. Start there.

Implementation order: WelfordMerge → AffineCompose (b=0) → AffineCompose (b≠0) → LogSumExpMerge → SarkkaMerge.

Challenge 27's `Op::canonical_structure()` bridges this to the proof system automatically: `Op::AffineCompose.canonical_structure()` = `Structure::semigroup(Product(Mat(2,2,Real), Vec(2,Real)), AffineCompose)`.

---

## The Single Missing Primitive (scout-2 synthesis of challenges 13+28+29)

The answer to "what accumulate primitive is missing?" is now precise:

**Matrix prefix scan over the affine composition semiring.**

For any model with state transition `x_t = A_t · x_{t-1} + b_t`, define the associative binary op:
```
(A₂, b₂) ∘ (A₁, b₁) = (A₂·A₁, A₂·b₁ + b₂)
```

Blelloch scan over this semiring gives ALL states in O(log n). The `b_t` term requires the correction `-J_b · b_a` from garden entry `006-the-correction-term.md` (the Sarkka derivation). Zero-offset models (GARCH σ², plain EWMA) have b=0 and the correction vanishes — those are the easy cases.

**Every structured-state model is an instance of this pattern:**
- Welford merge: (n, mean, M2) — 3-field associative state
- LogSumExp: (max, shifted-sum) — 2-field associative state
- GARCH/EWMA/AR: 2×2 matrix, b=0
- Kalman filter: 5-tuple (A, b, C, η, J) with the correction term

The pattern: design the state struct, prove the binary merge is associative, add to winrapids-scan, and the entire family parallelizes for free.

## Challenges 30-31: Spatial and Neural Late Discoveries

**Challenge 30** — `spatial.rs` module docstring claims "compressed row format" but `SpatialWeights` is `Vec<Vec<(usize, f64)>>` — identical structure to `graph.rs`'s `Vec<Vec<Edge>>`. Both GPU-hostile, both undocumented as adjacency lists. Moran's I and Geary's C should eventually be GPU scatter operations over CsrMatrix rows.

**Challenge 31** — `neural.rs` has all forward passes but backward passes are missing for Conv2D, Attention, BatchNorm, Pooling. Challenge 19 (gradient duality) says backward = forward transposed. The conv backward is `im2col(x)^T @ ∂L/∂y` — same GEMM, transposed input. The attention backward is `attn^T @ ∂L/∂output`. No new kernel types needed. One principle closes the gap.

---

---

## Challenge 33: Thomas Algorithm = 3×3 Matrix Prefix Scan

**Found in `interpolation.rs`** — natural_cubic_spline / clamped_cubic_spline Thomas forward sweep (lines 274-288).

The Thomas algorithm appears sequential but isn't. In homogeneous coordinates:

Let `d'_i = p_i / q_i` and `r'_i = s_i / q_i`. The forward elimination becomes:

```
[p_i]   [b_i,  0,    −a_i·c_{i-1}] [p_{i-1}]
[s_i] = [d_i,  −a_i, 0           ] [s_{i-1}]
[q_i]   [1,    0,    0            ] [q_{i-1}]
```

Matrix multiplication is associative → Blelloch scan applies. The full tridiagonal solve is:
1. Forward pass: `accumulate(Prefix, tridiag_element(i), Op::MatMulPrefix(3))`
2. Back-substitute: `accumulate(Suffix, back_elem(i), Op::AffineCompose)` using results from step 1

**Everything that currently uses Thomas becomes GPU-parallel**: natural cubic spline, clamped cubic spline, 1D finite differences, all BVPs on uniform grids.

**The degeneration hierarchy is now complete**:
```
3×3 MatPrefix (Thomas tridiagonal)  ← challenge 33
     ↓
2×2 AffineCompose (GARCH, EWMA, AR, Kalman)  ← challenge 32
     ↓
1×1 Scalar (Add, Mul, Max, Min)
```

The natural generalization: `Op::MatMulPrefix(n: usize)` — one Op variant for any matrix size, covering the full hierarchy.

**B-spline bonus**: The Cox-de Boor recursion for B-spline basis evaluation is a binary tree reduction — `accumulate(Windowed, bspline_element, Op::BsplineMix)` where the window = degree+1 (local support). Once enough Op variants exist, B-spline evaluation is also a windowed accumulate.

---

## Challenge 34: Particle Methods Are Kingdom A (bayesian.rs misclassification)

**Found in `bayesian.rs`** — module docstring: "MCMC = sequential sampling (Kingdom B)."

This is half-wrong. Sequential Monte Carlo (particle filters, importance sampling) is NOT sequential — it's embarrassingly parallel.

Particle filter step:
1. Propagate: `accumulate(All, transition(particle), ...)` — N independent propagations
2. Weight update: `accumulate(All, log_likelihood(obs, particle), Add)` + softmax
3. Resample: `gather(particles, categorical_sample(weights))` — weighted gather
4. Estimate: `accumulate(All, f(particle) * weight, Add)` — weighted mean

All 4 steps are `{accumulate, gather}`. The module has MH-MCMC (Kingdom B) and conjugate Bayesian regression (Kingdom A) but is MISSING the entire middle row — particle methods — and misclassifies them as sequential.

**Bonus**: The conjugate update at lines 121-128 computes `X'X = Σ_i x_i·x_i'` as a sequential triple loop. This is `accumulate(All, outer_product(x_i), Op::OuterProductAdd)`. The same `Op::OuterProductAdd` primitive covers ALL multivariate statistics that need Gram matrices (CCA, LDA, MANOVA, Hotelling's T²) — confirmed by `multivariate.rs` module header.

**The Op family is now complete**:
```
Op::MatMulPrefix(3):  Thomas algorithm, IIR biquad filter
Op::AffineCompose:    GARCH, EWMA, AR, Adam, Kalman (= MatMulPrefix(2) in homogeneous coords)
Op::WelfordMerge:     streaming variance, COPA covariance
Op::OuterProductAdd:  Gram matrices, BLR, MANOVA, LDA, CCA
Op::LogSumExpMerge:   softmax, Cox partial likelihood
Op::SarkkaMerge:      full parallel Kalman filter
```

**[scout-2]**: `Op::AffineCompose` IS `Op::MatMulPrefix(2)` in homogeneous coordinates — `x → a·x+b` = `[[a,b],[0,1]]·[x,1]`. Full ladder is one type parameterized by n. Efficiency reason to keep n=2 separate: 4 ops vs 27 for 3×3.

**Blelloch coverage rule** (scout-2): test at n ≥ 2^d where d = tree depth. For b≠0 correction term: n≥4. For full confidence: n≥8 (the step-3 position). b=0 cases always associative, don't need this. n=8 is the correctness canary for any b≠0 scan.

**Savitzky-Golay correction** (scout): `savgol_filter` IS implemented at `signal_processing.rs:732`. The documentation gap in that module was an error in the naturalist's search — the function is named `savgol_filter`, not `savitzky`. The real documentation gap remains spatial.rs (CSR claim, adjacency list implementation).

**Documentation-before-code gaps found across expedition** (module header claims vs. actual implementation):
- `spatial.rs`: claims "compressed row format" → is `Vec<Vec<(usize,f64)>>` adjacency list (challenge 30)
- `nonparametric.rs`: header claims "Shapiro-Wilk (small n approximation)" → not implemented (spec document: `docs/research/tambear-allmath/shapiro-wilk-spec.md`)
- `rng.rs`: header claims Sobol quasi-random sequences → not implemented (low priority)
- ~~`signal_processing.rs`: Savitzky-Golay~~ — was a search error, IS implemented as `savgol_filter`

One infrastructure decision (challenge 32 Op enum extension) unlocks all of them.

---

## Expedition Summary: 34 Challenges, One Core Finding

**The single most important structural observation of this expedition:**

Every sequential algorithm in tambear that has the form `state_t = f(state_{t-1}, data_t)` can be parallelized via a matrix prefix scan — as long as f is an affine function of the state. The test: "can you write f as `state_t = A_t · state_{t-1} + b_t`?" If yes, it's O(log n) via Blelloch.

The Op enum extension (challenge 32) is the one change that unlocks this across the entire codebase: GARCH, EWMA, AR, Adam, Kalman filter, IIR biquad, Thomas tridiagonal, conjugate Bayesian updates, particle filter weights — all variants of the affine composition semiring.

The correctness guarantee (challenge 27) comes from `Op::canonical_structure()` bridging to the proof system. The lint reclassification (challenge 26) happens automatically once kingdoms are derived from algebraic structure.

The expedition found no new algorithms that require genuinely new kernel infrastructure. Every gap closes by extending the existing Op enum with associative state types.

---

---

## Verified Bugs — Filed by Math-Researcher (must fix before merge)

From `docs/research/tambear-allmath/math-verification.md`:

| Issue | Location | Description | Fix |
|---|---|---|---|
| #4 | `mixed_effects.rs:146-151` | LME σ² M-step: uses `σ²` where `n_g·σ²_u` should appear in trace correction | Exact Rust code in math-verification.md |
| #6 | `nonparametric.rs:344-369` | `ks_test_normal` doesn't standardize data before testing against N(0,1) | Standardize `z=(x-x̄)/s`, use Lilliefors p-values |
| #1 | `bayesian.rs` | LCG in MH sampler (poor statistical quality) | Replace with `Xoshiro256::new(seed)` — **not** Xoroshiro128Plus |

**High-priority adversarial tests** (these will fail before fixes, pass after):
1. `ks_test_normal` on N(5,1) data — currently rejects normality (wrong); should not reject after fix
2. LME σ²_u << σ² case — produces wrong ICC; should fail before fix, pass after
3. MANOVA with p > k-1 — Pillai df2 too large (advisory, not critical)

---

*The naturalist's expedition is complete. 34 challenges logged in the tekgy-challenges campsite. The map is the territory.*
