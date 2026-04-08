# tambear Gap Taxonomy: What's Missing and How It Decomposes
**Author**: scout  
**Date**: 2026-04-06  
**Source**: Full audit of 68K lines / 38 modules + systematic field review  
**Companion**: scout-audit.md (what exists), taxonomy.md (math-researcher's full roadmap)

This document focuses on **gaps** — fields with zero or near-zero coverage — with accumulate+gather decompositions for each algorithm. Organized by WinRapids priority.

---

## Tambear Kingdom Reference

- **Kingdom A** = single-pass closed-form accumulate (linear algebra, weighted sums)
- **Kingdom B** = sequential scan, order-dependent (time series, MCMC chains, Kalman)
- **Kingdom C** = iterative/converging (EM, Newton, optimization loops)
- **Kingdom D** = graph/recursive (DP, tree traversal, backtracking)
- **Kingdom E** = multi-pass pyramid (FFT-style, hierarchical, wavelet)

---

## Priority 1: Critical for WinRapids Signal Farm

### Stochastic Processes (0% coverage)

The market is a stochastic process. These belong in `crates/tambear/src/stochastic.rs`.

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Geometric Brownian Motion | B | `S_{t+1} = S_t · exp((μ-σ²/2)Δt + σ√Δt·Z)` — sequential multiply |
| Ornstein-Uhlenbeck process | B | `X_{t+1} = X_t + θ(μ-X_t)Δt + σ√Δt·Z` — sequential |
| Euler-Maruyama SDE solver | B | `X_{t+1} = X_t + f(X_t)Δt + g(X_t)ΔW` — sequential |
| Milstein scheme | B | E-M + `g·g'·(ΔW²-Δt)/2` correction — sequential |
| Hawkes process (self-exciting) | B | `λ(t) = μ + Σᵢ α·exp(-β(t-tᵢ))` — scatter past events with decay kernel |
| Cox process (doubly stochastic) | B | integrate λ(t), then Poisson arrivals |
| Lévy process (stable noise) | B | stable distribution sampling + sequential sum |
| Brownian bridge | B | sequential with endpoint conditioning |
| Fokker-Planck (density PDE) | C | finite-difference PDE on probability density |

**The Hawkes process** deserves special attention. `λ(t) = μ + Σᵢ α·exp(-β(t-tᵢ))` is a scatter-accumulate over past events with exponential kernel — native tambear architecture. Trade arrivals, order book events, price jumps all fit Hawkes. Zero-coverage gap with maximum WinRapids relevance.

---

### Kalman Filter Family (0% coverage)

New module: `crates/tambear/src/kalman.rs`

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Linear Kalman predict | A | `x̂ = A·x; P = A·P·Aᵀ + Q` — matrix accumulate |
| Linear Kalman update | A | `K = P·Hᵀ·(H·P·Hᵀ+R)⁻¹; x += K·(y-H·x̂)` — accumulate + Cholesky |
| Kalman smoother (RTS) | B | backward pass: `P_s = P + C·(P_s_next - P_pred)·Cᵀ` |
| Extended Kalman (EKF) | B | same + Jacobian `∂f/∂x`, `∂h/∂x` at each step |
| Unscented Kalman (UKF) | B | sigma points: `2n+1` deterministic samples → propagate → remerge |
| Ensemble Kalman (EnKF) | B | Monte Carlo ensemble of state vectors + sample covariance |
| Particle filter (bootstrap) | B | importance weights + resampling = scatter-accumulate |
| Information filter | A | inverse covariance form — natural for sparse systems |

**Decomposition detail:** Full Kalman step = 5 accumulates:
1. `A·x` (matrix-vector)
2. `A·P·Aᵀ + Q` (two matrix multiplies + add)
3. `H·P·Hᵀ + R` (for innovation covariance)
4. Cholesky solve for K
5. State/covariance update

All Kingdom A, chained sequentially over time → Kingdom B.

---

### ARMA/ARIMA/VAR (10% coverage — AR only)

Extend `time_series.rs`.

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| MA(q) innovations algorithm | B | sequential: each step uses prior innovations |
| ARMA(p,q) MLE (CSS/exact) | C | iterate parameters, compute log-likelihood |
| ARIMA(p,d,q) | B | difference d times (B) + ARMA |
| SARIMA(p,d,q)(P,D,Q)_s | B+C | seasonal difference + seasonal ARMA |
| VAR(p) fitting | A | multivariate OLS: `accumulate(X, XᵀX, XᵀY)` |
| Johansen cointegration test | A+C | eigendecompose Π from reduced-rank regression |
| Engle-Granger cointegration | A+B | OLS residuals → ADF test |
| Seasonal decomposition (STL) | B+C | locally weighted regression iterations |
| VARMA | C | multivariate ARMA MLE |

---

### Mathematical Finance (0% coverage)

New module: `crates/tambear/src/finance.rs`

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Black-Scholes formula | A | `d₁,d₂ = log(S/K±...)/σ√T`; call = `S·N(d₁) - K·e^{-rT}·N(d₂)` |
| Black-Scholes Greeks | A | closed-form derivatives: delta = N(d₁), gamma = φ(d₁)/Sσ√T, etc. |
| Binomial option tree | D | backward DP: `V_i = (p·V_{i+1,up} + (1-p)·V_{i+1,down})·e^{-rΔt}` |
| Monte Carlo option pricing | A | `accumulate(n_paths, payoff(path), mean)` |
| Vasicek simulation | B | sequential: `dr = a(b-r)dt + σ·dW` |
| CIR simulation | B | sequential: `dr = a(b-r)dt + σ√r·dW` |
| Nelson-Siegel yield curve | C | NLS on observed bond prices |
| Duration & convexity | A | `accumulate(cash_flows, t·CF·e^{-yt}, sum)` |
| VaR (historical) | A | quantile of sorted loss distribution |
| Expected Shortfall | A | conditional mean beyond VaR = `accumulate(losses>VaR, mean)` |
| GEV/GPD extreme value fit | C | iterative MLE on maxima/exceedances |
| Hill estimator | A | `accumulate(top-k order stats, log ratio, mean)` |
| Copula sampling (Gaussian) | A | Cholesky of correlation + marginal inverse CDF |
| Clayton copula CDF | A | `(u^{-θ} + v^{-θ} - 1)^{-1/θ}` — closed form |

---

## Priority 2: Major Missing Fields

### Cryptography & Coding Theory (0% coverage)

Foundation needed: GF(p^n) arithmetic (from abstract algebra section).

**Number-theoretic crypto** — new module `crypto.rs`:

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Fast modular exponentiation | B | sequential bit scan: square-and-multiply |
| Miller-Rabin primality | B | sequential witness tests |
| Extended Euclidean algorithm | B | sequential gcd steps |
| Baby-step giant-step | A+D | scatter baby steps (A), search (D) |
| RSA arithmetic | A | modular exponentiation |
| EC point addition/doubling | A | field arithmetic accumulate |
| EC scalar multiplication | B | double-and-add sequential |
| Montgomery ladder | B | always-2-ops sequential (side-channel safe) |
| ECDSA sign/verify | A | hash + scalar mul |

**Error-correcting codes** — new module `codes.rs`:

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Hamming(7,4) encode | A | matrix multiply over GF(2) = XOR accumulate |
| Reed-Solomon encoding | A | polynomial evaluation over GF(2^m) |
| Reed-Solomon decoding (BM) | C | iterative error locator polynomial |
| LDPC belief propagation | C | `scatter(var→check) + accumulate(check→var)`, iterate |
| CRC computation | B | sequential XOR division over GF(2) |
| Viterbi (convolutional) | B+D | forward DP on trellis + traceback |
| Turbo decoding (BCJR) | B | forward-backward on trellis = sequential |
| AES (with GF(2^8)) | B | sequential rounds; MixColumns = GF matrix mul |

**Accumulate highlight — LDPC:** Belief propagation = two interleaved scatter-accumulate passes per iteration. Variable→check: `scatter(variable_beliefs, check_node, product)`. Check→variable: `scatter(check_parity_sums, variable_node, ratio_message)`. Iterate until convergence. Kingdom C.

---

### Classical Mechanics & Symplectic Integration (0% coverage)

New module: `mechanics.rs`

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Störmer-Verlet integrator | B | 2-step sequential: half-kick, drift, half-kick |
| Leapfrog | B | sequential 2-step |
| Runge-Kutta-Nyström | B | 4-stage sequential per step |
| N-body force (direct) | A | `accumulate(particles_j, F(i,j), sum)` per particle i |
| Barnes-Hut (O(n log n)) | D+E | tree build (D) + multipole accumulate (E) |
| Fast multipole method | E | multi-level scatter + accumulate with far-field expansion |
| Rigid body quaternion ODE | B | sequential: quaternion × angular velocity |
| Inertia tensor | A | `accumulate(mass_elements, r²×m)` |
| Hamiltonian energy computation | A | `T + V = accumulate(p², 1/2m) + potential` |

**Why symplectic matters:** Verlet/leapfrog *preserve phase space volume* (Liouville's theorem). They're the right integrators for Hamiltonian systems. Standard RK4 drifts energy. For WinRapids: market microstructure has a Hamiltonian-like structure if you model it as a dynamical system.

---

### Statistical Mechanics (5% coverage via custom equipartition.rs)

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Partition function Z | A | `accumulate(states, exp(-βE), sum)` |
| Free energy | A | `-kT·log(Z)` |
| Ising model (Metropolis) | B | sequential spin flip = standard MH MCMC on lattice |
| Wolff cluster algorithm | D | BFS flood-fill cluster + flip — much faster than spin-flip |
| Wang-Landau (density of states) | C | iterative histogram flattening |
| Transfer matrix | A | matrix-vector accumulate over system length = exact 1D solution |
| Virial equation | A | accumulate cluster integrals |
| DMRG (tensor network) | C | iterative SVD truncation on tensor network |

**Note:** `equipartition.rs` already implements free energy and phase analysis, but from tambear's specific (ρ,σ,τ) perspective. Standard stat mech algorithms (Ising, Wang-Landau) are absent and don't overlap with the existing code.

---

### Tensor Decomposition (0% coverage)

New module: `tensor.rs`

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| CP decomposition (ALS) | C | fix all factors, solve OLS for one — `accumulate(Khatri-Rao, data slice, ALS)` |
| Tucker decomposition (HOSVD) | A | sequential mode-n SVD then project |
| Tensor train (TT-SVD) | A | sequential SVD with rank truncation, left-to-right sweep |
| HOPM (higher-order power method) | C | iterative, converges to best rank-1 component |
| NMF (non-negative matrix factorization) | C | multiplicative updates = accumulate ratio |
| NTF (non-negative tensor) | C | same extended to 3+ modes |
| Randomized Tucker | A | random projection + SVD = sketch-based |

**Decomposition highlight:** CP-ALS = `gather(Khatri-Rao product of all factor matrices except mode n)` → `accumulate(unfolded tensor, gathered factor, lstsq)`. The Khatri-Rao product is an interleaved gather of column-wise Kronecker products. Already expressible in tambear primitives.

---

### Iterative Linear Algebra (0% coverage)

Extend `linear_algebra.rs` or new `iterative_linalg.rs`.

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Conjugate Gradient (CG) | B | sequential: α_k, β_k, p_{k+1} — Krylov iteration |
| GMRES | B | sequential Arnoldi + solve Hessenberg system |
| BiCGSTAB | B | sequential with shadow residual |
| Preconditioned CG | B | CG with `M⁻¹` applied per step |
| ILU(0) preconditioner | A | incomplete LU = accumulate with structural zero-fill |
| Multigrid V-cycle | E | restrict (scatter-accumulate down), smooth (B), prolongate (gather up) |
| Lanczos algorithm | B | 3-term recurrence, produces tridiagonal projection |
| Arnoldi iteration | B | sequential orthogonalization |
| Randomized SVD | A | random projection `Y=A·Ω`, then QR + SVD of `B=Qᵀ·A` |
| SpMV (sparse matrix-vector) | A | scatter nonzeros into result accumulate |

---

## Priority 3: Pure Math Foundations

### Abstract Algebra (5% coverage via proof.rs)

Currently `proof.rs` has the algebraic *vocabulary* (Sort::Monoid, etc.) but no *computational* algebra.

**GF(p^n) arithmetic** — foundational for coding theory:
- `GaloisField { p: u64, n: usize, irred: Vec<u64> }` 
- Multiplication = polynomial multiply mod irreducible = `accumulate(coeffs_a × coeffs_b, mod_reduce)`
- Inversion = extended Euclidean over GF(p)[x]

**Polynomial operations over fields:**
- GCD (Euclidean): B — sequential remainder steps
- Factoring (Berlekamp): C — iterative over GF(p) factorization
- Interpolation over GF(p): A — already have Lagrange, generalize

**Gröbner bases (Buchberger):**
- C — iterative S-polynomial reduction
- Foundation for algebraic geometry and constraint solving

**LLL lattice reduction:**
- C — iterated basis reduction
- Used in: lattice crypto (NTRU), integer programming, factoring polynomials over ℤ

---

### Algebraic Topology Beyond Rips

`tda.rs` has Rips H₀/H₁. Missing:

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Cubical homology | A | boundary operators on grid complexes |
| Čech complex | B | check ball intersections = KNN + threshold |
| Persistent H₂ and beyond | B | extend Rips to higher dimensions |
| Discrete Morse theory | B | gradient flow on simplicial complex |
| Sheaf cohomology (basic) | A | Mayer-Vietoris accumulate |
| Mapper algorithm | D | cover + cluster + nerve graph |
| Euler characteristic | A | `accumulate(simplices, (-1)^dim, sum)` |

---

### Measure Theory & Ergodic Theory

The formal underpinning of probability and integration.

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Ergodic average (Birkhoff) | B | time-average along orbit = prefix sum / n |
| Poincaré recurrence detection | B | sequential scan, flag returns to neighborhood |
| Hausdorff dimension (box-counting) | C | `accumulate(scales, log N(ε) / log(1/ε))` |
| Invariant measure estimation | C | iterated pushforward + normalize |
| Mixing coefficient | B | autocorrelation decay |

---

### Harmonic Analysis Beyond FFT

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Spherical harmonics Y_l^m(θ,φ) | A | recurrence over degree l |
| NUFFT (non-uniform DFT) | E | multi-level scatter-accumulate + corrections |
| Walsh-Hadamard transform | E | butterfly = log-depth accumulate (same structure as FFT) |
| Continuous wavelet transform (CWT) | E | `accumulate(scales, FFT × scaled_wavelet, IFFT)` |
| S-transform | E | Gaussian-windowed STFT, scale-dependent σ |
| EMD / Hilbert-Huang transform | C+B | iterative sifting + Hilbert analytic signal |
| Group DFT (finite groups) | E | character table × group algebra = matrix-vector |

---

## Priority 4: Domain Mathematics

### Neuroimaging (0% coverage)

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| HRF convolution (fMRI design) | A | convolve regressors × HRF = FFT product |
| GLM for BOLD signal | A | voxelwise OLS = `accumulate(X, XᵀX, XᵀY)` per voxel |
| ICA decomposition (FastICA) | C | iterative kurtosis maximization |
| Phase-locking value (EEG) | A | `accumulate(epochs, exp(iΔφ), mean)` |
| Connectivity matrix | A | pairwise accumulate = outer product of signals |

### Genomics (0% coverage)

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Smith-Waterman (local alignment) | D | DP: `H(i,j) = max(0, diag+score, H(i-1,j)-gap, H(i,j-1)-gap)` |
| Needleman-Wunsch (global) | D | same, no max(0) floor |
| Profile HMM (Viterbi) | B+D | sequential scan over sequence + DP |
| k-mer frequency | A | scatter: each k-mer → counter array |
| Phylogenetic distance (Jukes-Cantor) | A | closed-form from substitution count |
| UPGMA clustering | D | iterative minimum-distance merge |

**Decomposition highlight — Smith-Waterman on GPU:** Anti-diagonal wavefront parallelization. Each anti-diagonal `i+j=k` is independent. `gather(anti-diagonal k-1)` → `accumulate(anti-diagonal k, max(0, diag+score, gap_penalties))`. Embarrassingly parallel over anti-diagonals after gather.

### Astrophysics (0% coverage)

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| Kepler's equation (E - e·sin E = M) | C | Newton iteration on scalar |
| Orbital elements ↔ Cartesian | A | closed-form accumulate |
| Keplerian propagation | B | sequential Kepler steps |
| Power spectrum of density field | E | FFT of density grid |
| Stellar spectrum fitting | C | NLS on photometric bands |

### Seismology (0% coverage)

| Algorithm | Kingdom | Decomposition |
|---|---|---|
| STA/LTA trigger | B | running ratio of two EMAs = sequential scan |
| Magnitude (ML, MW) | A | accumulate log-peak from waveforms |
| Seismic FD (acoustic wave) | B | FDTD: sequential grid update |
| Cross-correlation for P-pick | A | FFT product = convolve |
| Gutenberg-Richter MLE | A | `accumulate(magnitudes, 1/ln(10)/(mean-m_min))` |

---

## Cross-Cutting Structural Insights

### Five algorithms that are secretly the same thing

1. **Viterbi = Smith-Waterman = Knuth-Yao DP** = `accumulate` over `(max, +)` semiring. Different domains, identical decomposition.

2. **Belief propagation = LDPC decoding = Kalman update** = `scatter(factor→variable)` + `accumulate(variable beliefs)`. Factor graph marginalizing.

3. **All gradient optimizers** (Adam, AdaGrad, RMSProp, etc.) = `sequential_scan(params, (grad, state), update_rule)`. Same Kingdom B structure, different update closure.

4. **All spectral transforms** (DFT, DCT, WHT, Spherical Harmonics, Graph Laplacian modes) = `accumulate(basis_functions, inner_product_with_data, sum)`. They differ only in the basis.

5. **AES MixColumns = Reed-Solomon encoding = BCH encoding** = polynomial evaluation over GF(2^m). Same field multiply-accumulate.

### The missing modules (natural file structure)

Based on gaps, new modules needed:
```
crates/tambear/src/
  stochastic.rs       ← SDE, Hawkes, Lévy, OU
  kalman.rs           ← Kalman family + particle filter
  hmm.rs              ← Viterbi, Forward-Backward, Baum-Welch (move from mixture.rs)
  arma.rs             ← MA, ARMA, ARIMA, SARIMA, VAR
  finance.rs          ← Black-Scholes, copulas, EVT, yield curves
  crypto.rs           ← modular arithmetic, RSA, ECC
  codes.rs            ← error-correcting codes (needs GF first)
  galois.rs           ← GF(p^n), polynomial rings, Berlekamp
  tensor.rs           ← CP, Tucker, TT, NMF
  mechanics.rs        ← symplectic integrators, N-body, rigid body
  stat_mech.rs        ← Ising, Wang-Landau, partition functions
  quantum.rs          ← Schrödinger, quantum gates, VQE
  iterative_linalg.rs ← CG, GMRES, BiCGSTAB, multigrid
  domain/
    neuroimaging.rs   ← HRF, GLM, ICA, PLV
    genomics.rs       ← SW, NW, profile HMM, k-mers
    astrophysics.rs   ← Kepler, orbital mechanics
    seismology.rs     ← STA/LTA, magnitudes, Gutenberg-Richter
  wavelet_analysis.rs ← CWT, S-transform, EMD
  spherical_harmonics.rs
```

---

## Summary Priority Matrix

| Field | Coverage | WinRapids | Effort | Decomposition Complexity |
|---|---|---|---|---|
| Stochastic processes (OU, Hawkes, SDEs) | 0% | ★★★★★ | Low | B-sequential |
| Kalman + HMM | 0% | ★★★★★ | Medium | B-sequential |
| ARMA/ARIMA/VAR | 10% | ★★★★★ | Medium | B+C |
| Mathematical finance | 0% | ★★★★★ | Low | A+C |
| Tensor decomposition | 0% | ★★★★ | Medium | C (ALS) |
| Iterative linear algebra | 0% | ★★★★ | Medium | B-Krylov |
| Cryptography | 0% | ★★★ | Medium | B (modexp) |
| Error-correcting codes | 0% | ★★★ | Medium | A+C |
| Classical mechanics (symplectic) | 0% | ★★★ | Medium | B |
| Abstract algebra (GF) | 5% | ★★★ | Medium | A+C |
| Statistical mechanics | 5% | ★★★ | Medium | A+B |
| Harmonic analysis (CWT/SHT) | 20% | ★★★ | Medium | E |
| Domain math (neuro/genomics) | 0% | ★★ | High | D (DP) |
| PDE solvers | 0% | ★★ | High | B+C |
| Algebraic topology (beyond Rips) | 10% | ★★ | High | A+D |
| Quantum mechanics | 0% | ★★ | High | B+C |
| Abstract algebra (Gröbner) | 0% | ★ | Very High | C |
| Category theory (computational) | 5% | ★ | Very High | D |

*Full audit: `campsites/tambear-allmath/20260406164237-audit/scout/20260406164948-full-audit.md`*
