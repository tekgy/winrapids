# Physics Math Gaps — Tambear Coverage Analysis

**Date**: 2026-04-10  
**Method**: Full audit of `crates/tambear/src/physics.rs` (6 sections, ~1170 lines) plus
cross-checks against `stochastic.rs`, `numerical.rs`, `signal_processing.rs`, and `linear_algebra.rs`.
Web research used to ground each gap in domain literature.

---

## What We Already Have (physics.rs inventory)

### Section 1 — Classical Mechanics
- Particle struct (pos, vel, mass; KE, momentum, angular momentum)
- N-body gravitational via Velocity-Verlet (`nbody_gravity`)
- SHO exact solution, energy (`sho_exact`, `sho_energy`)
- Damped harmonic oscillator underdamped (`dho_underdamped`)
- Kepler orbital elements from state vectors (`kepler_orbit`, `vis_viva`)
- Double pendulum Lagrangian derivation, RK4, energy (`double_pendulum_*`)
- Torque-free rigid body Euler equations (`euler_rotation`, `rotational_kinetic_energy`)

### Section 2 — Thermodynamics
- Ideal gas (pressure, temperature, internal energy, entropy change)
- Van der Waals equation + critical constants
- Carnot efficiency, Otto efficiency, isothermal entropy
- Fourier heat flux, Newton cooling, Stefan-Boltzmann radiation

### Section 3 — Statistical Mechanics
- Canonical partition function, mean energy, heat capacity, Helmholtz free energy
- Boltzmann probabilities, Gibbs entropy
- QHO energy levels, Bose-Einstein occupation, Planck distribution, Wien displacement
- 1D Ising exact (transfer matrix), 2D Ising Metropolis MC
- Arrhenius rate, equilibrium constant from ΔG

### Section 4 — Quantum Mechanics
- Hydrogen energy levels, wavelengths (Rydberg)
- Particle-in-box energy and wavefunction
- Tunneling transmission (rectangular barrier)
- Amplitude/complex state, normalize, time-evolve, expectation value, uncertainty
- Density matrix trace, purity, von Neumann entropy (diagonal)
- 1D Schrödinger via finite-difference tridiagonal (`schrodinger1d`)
- Symmetric tridiagonal eigenproblem via Sturm bisection + inverse iteration (`sym_tridiag_eigvals`)

### Section 5 — Fluid Dynamics
- Dimensionless numbers: Reynolds, Mach, Prandtl, Nusselt (Dittus-Boelter)
- Bernoulli velocity, Poiseuille flow rate and profile
- 1D Euler equations (Lax-Friedrichs), CFL timestep
- 2D Navier-Stokes vorticity-streamfunction: Poisson SOR, vorticity advection step

### Section 6 — Special Relativity
- Lorentz factor, relativistic KE, momentum, mass-energy
- Time dilation, length contraction, velocity addition, Doppler

### Elsewhere in tambear that overlaps physics
- SDE primitives (Euler-Maruyama via `ito_integral`, OU process) — `stochastic.rs`
- ODE solvers: Euler, RK4, RK45, RK4-system — `numerical.rs`
- FFT, STFT, convolution, Hilbert — `signal_processing.rs`
- Tridiagonal scan/solve — `linear_algebra.rs`
- Brownian motion, GBM, Poisson processes — `stochastic.rs`

---

## Gap Analysis by Field

Legend: **difficulty** = trivial / moderate / complex  
"Composes from" = what tambear primitives this would call.

---

### A. SHARED FUNDAMENTAL OPERATIONS (cross-field)

These appear across multiple physics subfields and should be implemented as flat-catalog primitives first.

#### A.1 Symplectic Integrators — **MISSING**

What we have: Velocity-Verlet (inside `nbody_gravity`), generic RK4/RK45  
What we don't have: Symplectic methods as standalone primitives

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `leapfrog_step` | Störmer-Verlet position-velocity leapfrog (2nd order symplectic) — the split that preserves the symplectic 2-form | CM, MD | trivial |
| `ruth_forest_4th` | Forest-Ruth 4th-order symplectic (3 substeps, coefficients c₁..c₄ from Ruth 1983/Forest-Ruth 1990) | CM, particle physics | moderate |
| `yoshida_6th` | Yoshida 6th-order symplectic composition (11 substeps) | CM, planetary dynamics | moderate |
| `verlet_npt` | Velocity-Verlet with Nosé-Hoover thermostat (NPT ensemble) | MD | moderate |
| `hamiltonian_phase_space_volume` | Liouville check: phase-space volume conservation diagnostic via Jacobian determinant | CM | moderate |

**Accumulate+gather decomposition**: `leapfrog_step(positions, velocities, forces, dt)` = accumulate forces (same as Verlet kick), gather position/velocity update. The symplectic structure is in the *ordering* of the split — no new mathematical machinery needed, just exposing the split as named primitives.

**Composes from**: `rk4_system` for the substep structure; force accumulation is existing N-body pattern; no new shared intermediates needed.

#### A.2 SDE Solvers — **PARTIALLY MISSING**

What we have: `ito_integral`, `ornstein_uhlenbeck`, `brownian_motion` (path generation)  
What we don't have: General-purpose SDE stepping primitives

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `euler_maruyama_step` | X_{n+1} = X_n + f(X_n)dt + g(X_n)ΔW_n | Langevin, plasma, finance | trivial |
| `milstein_step` | Euler-Maruyama + g·g'·((ΔW)²-dt)/2 correction term (O(dt) strong order) | MD, finance | trivial |
| `stochastic_rk_step` | Runge-Kutta for SDEs (Kloeden-Platen Runge-Kutta, 2nd order) | general | moderate |
| `fokker_planck_1d_diffusion` | Finite-difference solve of ∂ₜp = -∂ₓ(f·p) + ½∂ₓₓ(g²·p) on grid | Brownian motion, Langevin | moderate |
| `langevin_dynamics` | mẍ = -γẋ + F(x) + √(2γkT)η(t), Störmer-Verlet discretization | MD, plasma | trivial |

**Accumulate+gather**: `euler_maruyama_step` is one vector add + scalar multiply. `fokker_planck_1d_diffusion` is a tridiagonal solve (composes from `solve_tridiagonal`).

**Composes from**: `rng::sample_normal`, `solve_tridiagonal`, `rk4_system`.

#### A.3 Fast Multipole Method (FMM) — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `multipole_expansion` | Multipole coefficients from source distribution P(r): M_l = Σ q_i r_i^l Y_l^m(θ,φ) | N-body, electrostatics, BEM | complex |
| `local_expansion` | Convert far-field multipole to near-field local expansion (M2L operator) | N-body, EM | complex |
| `fmm_nbody` | O(N log N) N-body force via FMM (vs O(N²) direct) | N-body, Coulomb, gravity | complex |
| `barnes_hut_tree` | Octree + θ-MAC (multipole acceptance criterion) for O(N log N) N-body | N-body | complex |

**Note**: FMM and Barnes-Hut are structurally accumulate+gather — the tree is just a spatial grouping. The grouping patterns are hierarchical instead of flat, but the skeleton is identical: accumulate multipoles up the tree, gather forces down. This is the cleanest example of a Kingdom A algorithm with a hierarchical grouping.

**Composes from**: tree data structure (not in tambear yet), `special_functions::legendre_p` (already have), `rng` for sampling.

---

### B. CLASSICAL MECHANICS — Gaps

#### B.1 Constraint Dynamics — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `shake_constraint` | SHAKE algorithm: iterative constraint correction for bond lengths r_{ij} = d | MD, robotics | moderate |
| `rattle_constraint` | RATTLE: SHAKE + velocity correction for holonomic constraints | MD | moderate |
| `lagrange_multipliers_constrained` | Solve constrained Euler-Lagrange: M·q̈ = Q + Jᵀλ, J·q̈ = -J̇q̇ | robotics, multibody | complex |

#### B.2 Orbital Mechanics — **PARTIALLY MISSING**

What we have: `kepler_orbit` (compute orbital elements from state vector), `vis_viva`  
What we don't have:

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `kepler_anomaly_solve` | Solve Kepler's equation M = E - e·sin(E) for eccentric anomaly E (Newton-Raphson) | astrodynamics | trivial |
| `orbital_state_from_elements` | Convert Keplerian elements (a, e, i, Ω, ω, ν) → (r, v) Cartesian state vectors | astrodynamics | moderate |
| `hohmann_transfer` | ΔV for Hohmann transfer between circular orbits | astrodynamics | trivial |
| `patched_conic_soi` | Sphere-of-influence radius r_SOI = a·(m/M)^{2/5} | astrodynamics | trivial |

#### B.3 Multi-body Rigid Dynamics — **PARTIALLY MISSING**

What we have: Euler's torque-free rotation equations  
What we don't have:

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `quaternion_rotate` | Quaternion product, rotation matrix from quaternion, SLERP | robotics, graphics, attitude control | trivial |
| `inertia_tensor_parallel_axis` | I = I_cm + m·d² (parallel-axis theorem) | rigid body | trivial |
| `euler_to_quaternion` | Convert Euler angles (roll, pitch, yaw) to quaternion | rigid body | trivial |

---

### C. ELECTRODYNAMICS — **MOSTLY MISSING**

This is the largest gap. Physics.rs has no electrodynamics beyond the constant ε₀.

#### C.1 Maxwell Solver (FDTD) — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `fdtd_1d_yee_step` | 1D Yee FDTD: update E from curl(H), update H from curl(E) on staggered grid | EM, photonics, antennas | moderate |
| `fdtd_2d_te_step` | 2D TE mode FDTD (Ex, Ey, Hz fields) — Yee cell update | EM | moderate |
| `fdtd_absorbing_pml` | PML (perfectly matched layer) absorbing boundary condition coefficients | EM | complex |
| `fdtd_dispersive_drude` | Drude dispersive material model for metals in FDTD | photonics | complex |

**Accumulate+gather**: Each FDTD step = accumulate curl contributions from neighbors, gather into E/H at each cell. Perfectly parallel — textbook Kingdom A.

**Composes from**: `fft` (for spectral post-processing), `convolve` (for dispersive materials).

#### C.2 Coulomb/Biot-Savart — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `coulomb_force` | F = kq₁q₂/r² × r̂ between point charges | EM | trivial |
| `biot_savart_segment` | dB = μ₀I/(4π) · (dl × r̂)/r² for a current segment | EM, MRI | trivial |
| `electric_dipole_field` | E-field of electric dipole (near and far field) | EM | trivial |
| `magnetic_dipole_field` | B-field of magnetic dipole (Helmholtz coil, magnetic moment) | EM | trivial |
| `laplace_finite_difference_2d` | Solve ∇²φ = 0 (Laplace) on 2D grid with Dirichlet BC via SOR | EM, heat | trivial |

**Composes from**: `poisson_sor` (already have! — just change the source term).

#### C.3 Green's Functions — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `greens_function_free_space_3d` | G(r,r') = 1/(4π|r-r'|) for Laplacian | EM, acoustics | trivial |
| `greens_function_helmholtz_3d` | G(r,r') = exp(ik|r-r'|)/(4π|r-r'|) for Helmholtz ∇²G + k²G = -δ | wave EM, acoustics | trivial |
| `method_of_moments_1d` | MoM: discretize ∫G(x,x')σ(x')dx' = -φ_inc(x) into linear system | EM, acoustic BEM | moderate |

**Composes from**: `solve` (already have), `lagrange` interpolation (already have).

#### C.4 Transmission Lines — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `transmission_line_impedance` | Z_in = Z₀·(Z_L + jZ₀tan(βl))/(Z₀ + jZ_L tan(βl)) | RF engineering | trivial |
| `reflection_coefficient` | Γ = (Z_L - Z₀)/(Z_L + Z₀) | RF | trivial |
| `smith_chart_point` | Map impedance Z to Smith chart coordinates (Γ plane) | RF | trivial |

---

### D. THERMODYNAMICS / STAT MECH — Gaps

#### D.1 Advanced Monte Carlo — **MISSING**

What we have: 2D Ising Metropolis  
What we don't have:

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `wolff_cluster_flip` | Single-cluster Wolff algorithm: build cluster via BFS with probability 1-exp(-2βJ), flip all spins | spin models, critical phenomena | moderate |
| `swendsen_wang_clusters` | Multi-cluster Swendsen-Wang: bond percolation + simultaneous flip of all clusters | Ising/Potts | moderate |
| `wang_landau_density_of_states` | Wang-Landau flat-histogram algorithm: iteratively refine g(E) until histogram flat | any model with discrete E | complex |
| `parallel_tempering_swap` | Replica exchange MC: propose swap between replicas at T_i, T_{i+1} with Metropolis acceptance | frustrated systems | moderate |
| `histogram_reweighting` | Ferrenberg-Swendsen histogram reweighting to extrapolate thermodynamics | stat mech | moderate |

**Accumulate+gather**: Wolff cluster build = BFS over spin lattice (graph traversal, composes from `bfs` structure). Wang-Landau density-of-states update is a histogram accumulate.

**Composes from**: `union_find` (already have via clustering.rs!), `rng::sample_bernoulli`, `histogram`.

#### D.2 Molecular Dynamics — **MOSTLY MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `lennard_jones_force` | F_LJ = 24ε/σ·[(2(σ/r)¹³ - (σ/r)⁷)] | MD, chemical physics | trivial |
| `lennard_jones_energy` | U_LJ = 4ε[(σ/r)¹² - (σ/r)⁶] | MD | trivial |
| `ewald_sum` | Ewald summation for long-range Coulomb in periodic systems: real + reciprocal space | MD, periodic EM | complex |
| `particle_mesh_ewald` | PME: Ewald via FFT on grid (O(N log N) vs O(N^{3/2})) | MD | complex |
| `nose_hoover_thermostat` | Nosé-Hoover chain equations of motion for NVT ensemble | MD | moderate |
| `pressure_tensor` | Virial pressure P = nkT + ⟨Σ r·F⟩/(3V) | MD | trivial |
| `radial_distribution_function` | g(r) histogram accumulation from particle positions | MD, liquid theory | trivial |
| `velocity_autocorrelation` | C(t) = ⟨v(0)·v(t)⟩/⟨v(0)²⟩ | MD, diffusion coefficient | trivial |
| `mean_square_displacement` | MSD(t) = ⟨|r(t) - r(0)|²⟩, Stokes-Einstein diffusion | MD | trivial |

**Composes from**: `ornstein_uhlenbeck` structure, `rng`, `histogram`, `autocorrelation` (already have in signal_processing.rs).

#### D.3 Fokker-Planck / Langevin — **MISSING**

(Partly covered by stochastic.rs, but Fokker-Planck PDE solver is absent)

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `fokker_planck_1d` | Crank-Nicolson solve of ∂p/∂t = -∂(fp)/∂x + D∂²p/∂x² | Brownian, diffusion | moderate |
| `fokker_planck_stationary` | Solve 0 = -∂(fp)/∂x + D∂²p/∂x²: p_∞(x) ∝ exp(-U(x)/D) | equilibrium distributions | trivial |
| `diffusion_equation_1d` | Special case: F=0, pure diffusion ∂p/∂t = D∂²p/∂x² | heat, diffusion | trivial |
| `drift_diffusion_flux` | Continuity: J = fp - D∂p/∂x (drift + diffusion components) | transport | trivial |

**Composes from**: `solve_tridiagonal` (already have), Crank-Nicolson = 2 tridiagonal solves.

#### D.4 Partition Functions / Thermodynamic Integration — **PARTIALLY MISSING**

What we have: `partition_function` for discrete energy levels  
What we don't have:

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `thermodynamic_integration` | ∂F/∂λ = ⟨∂H/∂λ⟩_λ integrated via quadrature | free energy, drug design | moderate |
| `free_energy_perturbation` | ΔF = -kT ln⟨exp(-ΔH/kT)⟩_A (Zwanzig formula) | free energy calculation | trivial |
| `grand_canonical_partition` | Ξ = Σ_N z^N Z_N, fugacity z = exp(βμ) | grand canonical ensemble | trivial |
| `density_of_states_dos` | g(E) from eigenvalue spectrum, g(E) = Σδ(E - Eₙ) broadened by Gaussian | condensed matter | trivial |

---

### E. QUANTUM MECHANICS — Gaps

#### E.1 Quantum Gates / Circuit Simulation — **MISSING**

What we have: Amplitude struct, time_evolve_state, density matrix basics  
What we don't have: any gate operations or circuit machinery

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `pauli_x`, `pauli_y`, `pauli_z` | Single-qubit Pauli gates: 2×2 complex matrix-vector | QC | trivial |
| `hadamard_gate` | H = (X+Z)/√2, creates superposition | QC | trivial |
| `cnot_gate` | Controlled-NOT on 2-qubit state (4-component complex vector) | QC | trivial |
| `rotation_x`, `rotation_y`, `rotation_z` | R_x(θ) = exp(-iθX/2), parameterized gates for VQE/QAOA | QC, VQE | trivial |
| `controlled_phase_gate` | CPhase(θ): |11⟩ → e^{iθ}|11⟩ | QC | trivial |
| `quantum_fourier_transform` | QFT on n-qubit state (recursive decomposition) | QC, Shor's algorithm | moderate |
| `apply_gate_to_qubit` | Apply 1-qubit gate to qubit k in n-qubit state vector (2^n complex amplitudes) | QC | moderate |
| `circuit_simulate` | Sequential gate application to state vector | QC | moderate |
| `vqe_expectation` | ⟨ψ(θ)|H|ψ(θ)⟩ for variational quantum eigensolver parameterized state | QC, chemistry | moderate |
| `qaoa_circuit` | p-layer QAOA circuit for MaxCut: alternating problem + mixer unitaries | combinatorial opt | moderate |
| `quantum_fidelity` | F(ρ,σ) = (Tr√(√ρ σ √ρ))² between two density matrices | QC benchmarking | moderate |
| `quantum_state_tomography` | Reconstruct density matrix from measurement outcomes (MLE or linear inversion) | QC | complex |

**Accumulate+gather**: applying a 1-qubit gate to qubit k = accumulate over the 2^(n-1) affected pairs, gather updates. QFT = recursive butterfly (same structure as FFT — literally composes from `fft` pattern).

**Composes from**: `linear_algebra::mat_mul` (complex version), `fft` pattern, `rng` for measurement simulation.

#### E.2 Tensor Networks — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `mps_contract_single` | Contract single MPS site tensor (left-right sweep step) | condensed matter, QC | moderate |
| `svd_truncate` | SVD with bond dimension cutoff χ (truncate to top-χ singular values) | MPS, DMRG | trivial |
| `mps_expectation_value` | ⟨ψ|O|ψ⟩ via sequential contraction of MPS + operator + MPS* | DMRG | moderate |
| `dmrg_sweep_step` | Single DMRG sweep step: solve 2-site eigenvalue problem, update MPS tensors | 1D quantum systems | complex |
| `tebd_trotter_step` | Time-evolving block decimation: apply 2-site gate via SVD truncation | quantum dynamics | complex |
| `mpo_apply` | Apply matrix product operator (MPO) to MPS | quantum chemistry | complex |

**Composes from**: `svd` (already have!), `mat_mul`, eigendecomposition primitives.

**Why important**: DMRG is the gold standard for 1D quantum systems. SVD truncation with bond dimension χ is the only new primitive needed — everything else composes from existing linear algebra.

#### E.3 Variational Methods — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `rayleigh_ritz` | Minimize ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ in subspace (Rayleigh-Ritz method) | QM, FEM | moderate |
| `hartree_fock_step` | One HF SCF iteration: build Fock matrix from density matrix, diagonalize | quantum chemistry | complex |
| `variational_monte_carlo` | VMC: estimate ⟨H⟩ via Metropolis sampling of |ψ(R)|² | quantum chemistry | moderate |
| `imaginary_time_evolution` | Propagate ψ → exp(-τH)ψ, normalize each step; converges to ground state | QM | trivial |

**Composes from**: `sym_eigen` (already have), `metropolis_hastings` (already have via `bayesian.rs`), `rk4_system`.

#### E.4 Schrödinger Equation — **PARTIALLY MISSING**

What we have: `schrodinger1d` (time-independent, finite-difference diagonalization)  
What we don't have:

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `split_step_fourier` | Propagate ψ(x,t): alternating real-space (V) and k-space (T) half-steps using FFT | TDSE, BEC, nonlinear optics | moderate |
| `crank_nicolson_tdse` | Implicit CN scheme for time-dependent Schrödinger: tridiagonal solve each step | TDSE | moderate |
| `wave_packet_gaussian` | ψ(x,t=0) = exp(-(x-x₀)²/4σ²)·exp(ik₀x), Gaussian wavepacket initial condition | QM demos | trivial |
| `schrodinger2d` | 2D TISE on grid (sparse Hamiltonian, power iteration or Lanczos) | 2D potentials, quantum dots | complex |

**Composes from**: `fft`/`ifft` (already have), `solve_tridiagonal` (already have).

---

### F. RELATIVITY — **MOSTLY MISSING**

What we have: Special relativity (Lorentz, time dilation, etc.)  
What we don't have: General relativity

#### F.1 General Relativity Tensors — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `metric_christoffel` | Γ^σ_{μν} = ½g^{σρ}(∂_μg_{νρ} + ∂_νg_{μρ} - ∂_ρg_{μν}) from metric tensor g_{μν} | GR, cosmology | moderate |
| `riemann_tensor` | R^ρ_{σμν} = ∂_μΓ^ρ_{νσ} - ∂_νΓ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} - Γ^ρ_{νλ}Γ^λ_{μσ} | GR | moderate |
| `ricci_tensor` | R_{μν} = R^λ_{μλν} (contraction of Riemann) | GR | trivial (given Riemann) |
| `ricci_scalar` | R = g^{μν}R_{μν} | GR | trivial (given Ricci) |
| `einstein_tensor` | G_{μν} = R_{μν} - ½g_{μν}R | GR | trivial (given Ricci) |
| `geodesic_ode` | d²x^μ/dτ² + Γ^μ_{αβ}(dx^α/dτ)(dx^β/dτ) = 0, as first-order system for ODE solver | GR, black hole orbits | trivial |
| `schwarzschild_metric` | ds² = -(1-rs/r)dt² + (1-rs/r)^{-1}dr² + r²dΩ² coefficients | GR, black holes | trivial |
| `kerr_metric` | Boyer-Lindquist Kerr metric coefficients for rotating black holes | GR | moderate |

**Accumulate+gather**: Christoffel symbols = tensor contraction (matrix products + derivatives). Riemann = sum of products of Christoffel symbols. These are symbolic/numerical tensor operations.

**Composes from**: `mat_mul`, `inv` (already have), `derivative_central` (already have).

**Difficulty note**: For a fixed metric (Schwarzschild, Kerr), these are trivial. For a general symbolic metric, it requires symbolic differentiation (not in tambear). Start with the named exact metrics.

#### F.2 Numerical Relativity — **COMPLEX**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `adm_3plus1_decompose` | ADM formalism: lapse N, shift β^i, induced metric γ_{ij} from full metric | numerical GR | complex |
| `bssn_evolve` | BSSN formulation time evolution step (standard modern NR) | numerical GR, binary merger | complex |

**Note**: Full numerical relativity (binary black hole mergers etc.) is essentially an entire research subfield. The tensor primitives above (Christoffel through geodesic) cover 90% of coursework and research applications. Full BSSN is complex but structurally just PDE time-stepping over tensor fields.

---

### G. FLUID DYNAMICS — Gaps

What we have: 1D Euler (Lax-Friedrichs), 2D NS vorticity-streamfunction, dimensionless numbers, pipe flow  
What we don't have:

#### G.1 Lattice Boltzmann — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `lbm_d2q9_equilibrium` | Maxwell-Boltzmann equilibrium distribution f_i^eq for D2Q9 lattice | LBM | trivial |
| `lbm_collision_step` | BGK collision: f_i → f_i + (f_i^eq - f_i)/τ | LBM | trivial |
| `lbm_streaming_step` | Stream populations along velocity vectors: f_i(r+c_i, t+1) = f_i*(r,t) | LBM | trivial |
| `lbm_macroscopic` | Recover ρ = Σf_i, ρu = Σc_i f_i from populations | LBM | trivial |
| `lbm_bounce_back` | Bounce-back boundary condition (no-slip walls) | LBM | trivial |

**Accumulate+gather**: LBM is textbook Kingdom A. Collision = local (no neighbors), streaming = shift (gather from neighbors). Each is a single-pass over the grid.

**Composes from**: nothing exotic — just array indexing and arithmetic.

#### G.2 Smoothed Particle Hydrodynamics — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `sph_kernel_cubic_spline` | W(r,h) = σ_d·f(r/h): cubic B-spline SPH kernel, normalized | SPH | trivial |
| `sph_kernel_gradient` | ∇W(r,h) = (dW/dr)·r̂/r | SPH | trivial |
| `sph_density` | ρ_i = Σ_j m_j W(r_{ij}, h): SPH density accumulation | SPH | trivial |
| `sph_pressure_force` | F_i^P = -m_i Σ_j m_j (P_i/ρ_i² + P_j/ρ_j²)∇W_{ij} | SPH | trivial |
| `sph_viscosity_force` | Π_{ij} artificial viscosity term | SPH | moderate |
| `sph_timestep` | dt = min(CFL·h/c_s, viscous dt) for SPH | SPH | trivial |

**Accumulate+gather**: SPH density = accumulate(particle pairs within h, kernel weight, sum) — literally the definition of accumulate over a spatial grouping. Perfect tambear primitive.

**Composes from**: `knn_adjacency` (neighbor lists), `euclidean_2d`, `rng`.

#### G.3 Vortex Methods — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `biot_savart_vortex_2d` | Velocity at x from vortex at y: u = Γ/(2π) (x-y)^⊥/|x-y|² | 2D vortex flow | trivial |
| `vortex_sheet_velocity` | Velocity from distribution of vortex elements (accumulate over all vortices) | aerodynamics | trivial |
| `vortex_merge_criterion` | Check vortex merging: merge if overlap exceeds threshold | vortex particle method | trivial |

#### G.4 Turbulence Models — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `smagorinsky_viscosity` | ν_t = (C_s Δ)² |S̄|, Smagorinsky subgrid viscosity for LES | LES turbulence | trivial |
| `kolmogorov_scales` | η = (ν³/ε)^{1/4}, v_η = (νε)^{1/4}, τ_η = (ν/ε)^{1/2} | turbulence theory | trivial |
| `energy_spectrum_inertial_range` | E(k) ≈ C·ε^{2/3}·k^{-5/3} (Kolmogorov -5/3 law) | turbulence | trivial |
| `structure_function_2nd` | S₂(r) = ⟨|u(x+r) - u(x)|²⟩ ≈ C·ε^{2/3}·r^{2/3} | turbulence experiments | trivial |

**Composes from**: `periodogram`/`welch` (power spectrum), `moments_ungrouped` (already have).

---

### H. OPTICS — **MOSTLY MISSING**

What we have: `fft`/`fft2d` in signal_processing.rs (foundational), `window_*` functions  
What we don't have: optical wave propagation, ray tracing, beam propagation

#### H.1 Ray Optics — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `ray_refract` | Snell's law: n₁sinθ₁ = n₂sinθ₂, returns refracted ray direction | optics, lens design | trivial |
| `ray_reflect` | Specular reflection: r = d - 2(d·n̂)n̂ | optics, mirrors | trivial |
| `abcd_matrix_ray` | ABCD ray transfer matrix for optical element (lens, free space, mirror) | Gaussian beam | trivial |
| `abcd_propagate` | Propagate ray/beam through sequence of ABCD matrices via matrix product | lens design | trivial |
| `gaussian_beam_waist` | w(z) = w₀√(1+(z/z_R)²), Rayleigh range z_R = πw₀²/λ | laser optics | trivial |
| `snell_vector_3d` | Vector form of Snell's law in 3D (handles TIR detection) | ray tracing | trivial |

**Composes from**: `mat_mul` (already have), `dot` (already have).

#### H.2 Wave/Fourier Optics — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `fresnel_propagate` | U(x,y,z) via Fresnel diffraction integral (paraxial approximation): FFT of U₀·exp(iπr²/λz) | physical optics | moderate |
| `fraunhofer_diffract` | Far-field (Fraunhofer) diffraction = 2D Fourier transform of aperture field | diffraction, crystallography | trivial (composes from fft2d) |
| `angular_spectrum_propagate` | Angular spectrum method: propagate field via FFT, multiply by transfer function exp(ikz√(1-kx²-ky²)), IFFT | physical optics | moderate |
| `optical_transfer_function` | H(ν) for a diffraction-limited lens: 1 inside cutoff ν_c = NA/λ, 0 outside | imaging | trivial |
| `zernike_polynomial` | Z_n^m(ρ,φ): radial + azimuthal aberration basis for wavefront | adaptive optics | moderate |
| `wavefront_from_zernike` | Reconstruct wavefront from Zernike coefficients | adaptive optics | trivial |

**Accumulate+gather**: `angular_spectrum_propagate` = FFT → pointwise multiply → IFFT. Exactly the pattern of `fft` + `convolve`. Composes directly.

**Composes from**: `fft2d` (already have!), `fft`/`ifft`, `window_*`, `legendre_p` (for Zernike radial part).

#### H.3 Nonlinear Optics / Fiber — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `nlse_split_step` | Nonlinear Schrödinger equation propagation: dA/dz = (i/2)β₂∂²A/∂t² + iγ|A|²A via split-step Fourier | fiber optics, BEC | moderate |
| `group_velocity_dispersion` | β₂ = d²k/dω² for optical pulse spreading | photonics | trivial |

---

### I. PLASMA PHYSICS — **MISSING**

#### I.1 Particle-in-Cell (PIC) — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `pic_charge_deposit_ngp` | Nearest-grid-point charge deposition: ρ(grid) from particle positions | PIC | trivial |
| `pic_charge_deposit_cic` | Cloud-in-cell (linear interpolation) charge deposition | PIC | trivial |
| `pic_field_interpolate` | Bilinear interpolation of E,B fields from grid to particle positions | PIC | trivial |
| `boris_pusher` | Boris algorithm for charged particle in EM field: γm(v_{n+1}-v_n)/dt = q(E + v×B) | PIC, plasma | moderate |
| `pic_poisson_solve` | ∇²φ = -ρ/ε₀ on periodic grid via FFT (spectral Poisson solver) | PIC | moderate |
| `pic_current_deposit` | Esirkepov current deposition (charge-conserving) | PIC | complex |

**Accumulate+gather**: charge deposition = accumulate(particle charges, scatter to nearby grid cells, sum). Field interpolation = gather(grid values, interpolate to particle position). This is literally the canonical scatter-gather pattern.

**Composes from**: `fft` (spectral Poisson), `poisson_sor` (iterative alternative), `rng`.

#### I.2 Magnetohydrodynamics — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `mhd_ideal_1d_roe` | 1D ideal MHD Roe solver: 7-wave structure (3 Alfvén, 2 fast, 2 slow, entropy) | space physics | complex |
| `alfven_speed` | v_A = B/√(μ₀ρ) | plasma, MHD | trivial |
| `plasma_frequency` | ω_p = √(nq²/(mε₀)) for electron plasma oscillation | plasma | trivial |
| `debye_length` | λ_D = √(ε₀kT/(nq²)) | plasma | trivial |
| `cyclotron_frequency` | ω_c = qB/m | plasma, MRI | trivial |
| `magnetosonic_speed` | v_ms = √(v_A² + c_s²) | MHD | trivial |
| `mhd_divergence_cleaning` | GLM divergence cleaning for ∇·B = 0 constraint | numerical MHD | complex |

#### I.3 Vlasov Equation — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `vlasov_bsl_advect` | Backward semi-Lagrangian advection step for f(x,v,t) in phase space | Vlasov-Poisson | complex |
| `vlasov_moment_density` | n(x) = ∫ f(x,v) dv (velocity-space integration) | Vlasov | trivial |
| `vlasov_moment_current` | J(x) = q∫ v f(x,v) dv | Vlasov | trivial |
| `landau_damping_rate` | γ = -π/2 · (ω_p²/k²) · ∂f₀/∂v|_{v=ω/k} (analytic Landau damping) | plasma | moderate |

---

### J. CONDENSED MATTER — **MOSTLY MISSING**

#### J.1 Band Structure — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `bloch_hamiltonian_1d` | H(k) = -2t·cos(k·a) for 1D tight-binding (TBM) | condensed matter | trivial |
| `tight_binding_chain` | Energy bands E(k) for N-site tight-binding chain with periodic BC | solid state | trivial |
| `tight_binding_2d_square` | E(kx,ky) = -2t(cos(kx·a) + cos(ky·a)) for square lattice | condensed matter | trivial |
| `density_of_states_numerical` | g(E) from eigenvalue histogram with Gaussian broadening | condensed matter | trivial |
| `brillouin_zone_1d_path` | Generate k-path through BZ: Γ→X→M→Γ for 2D square | condensed matter | trivial |
| `berry_phase_1d` | γ = Im(ln∏⟨u_k|u_{k+1}⟩) along BZ (topological invariant) | topological materials | moderate |

**Composes from**: `sym_eigen` (already have), `histogram_auto` (already have), `fft` (k-space operations).

#### J.2 Hubbard Model — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `hubbard_exact_diag` | Exact diagonalization of Hubbard model for small cluster (Fock space basis, sparse H) | condensed matter | complex |
| `hubbard_hartree_fock` | Mean-field HF solution of Hubbard model: n_σ self-consistency | condensed matter | moderate |
| `hubbard_local_dos` | Local density of states A(ω) = -(1/π)Im G^R(ω) from Green's function | condensed matter | moderate |

#### J.3 Superconductivity (BCS) — **MISSING**

| Primitive | What it computes | Fields | Difficulty |
|-----------|-----------------|--------|------------|
| `bcs_gap_equation` | Self-consistent solution of BCS gap equation: Δ = gΣ_k Δ/(2E_k)·tanh(E_k/2kT) | superconductivity | moderate |
| `bcs_quasiparticle_energy` | E_k = √(ξ_k² + Δ²) (Bogoliubov quasiparticle) | BCS | trivial |
| `bcs_condensation_energy` | E_cond = N(0)Δ²/2 per unit volume | BCS | trivial |
| `cooper_pair_density` | n_s = 2Σ_k v_k² (superfluid density) | BCS | trivial |

---

## Summary: Priority Ordering

### Tier 1 — Fundamental, broadly used, trivial-to-moderate difficulty

These appear across many physics subfields and compose cleanly from existing primitives:

1. **SDE primitives** (`euler_maruyama_step`, `milstein_step`, `langevin_dynamics`) — compose from `rng::sample_normal`, `ito_integral` pattern already in `stochastic.rs`
2. **Symplectic integrators** (`leapfrog_step`, `yoshida_6th`) — compose from existing `rk4_system` pattern
3. **Fokker-Planck 1D** (`fokker_planck_1d`, `diffusion_equation_1d`) — composes from `solve_tridiagonal`
4. **LBM core** (`lbm_d2q9_equilibrium`, `lbm_collision_step`, `lbm_streaming_step`, `lbm_macroscopic`, `lbm_bounce_back`) — pure arithmetic, no dependencies
5. **Coulomb/EM basics** (`coulomb_force`, `laplace_finite_difference_2d`, `coulomb_force`, `electric_dipole_field`) — trivial
6. **Ray optics** (`ray_refract`, `ray_reflect`, `abcd_matrix_ray`, `gaussian_beam_waist`) — trivial, compose from `mat_mul`
7. **Fraunhofer diffraction** (`fraunhofer_diffract`) — literally `fft2d`, already have
8. **Angular spectrum method** (`angular_spectrum_propagate`) — composes from `fft2d`
9. **SPH kernel + density** (`sph_kernel_cubic_spline`, `sph_density`) — pure arithmetic
10. **GR tensor basics** (`schwarzschild_metric`, `geodesic_ode`) — trivial given ODE solver

### Tier 2 — Domain-specific, moderate difficulty, compose from existing primitives

11. **Quantum gates** (`pauli_*`, `hadamard`, `cnot`, `rotation_*`, `apply_gate_to_qubit`) — compose from complex `mat_mul`
12. **MPS/SVD truncation** (`svd_truncate`) — trivial given `svd` already exists; enables DMRG
13. **Cluster MC** (`wolff_cluster_flip`, `wang_landau_density_of_states`) — compose from `union_find` (already have!)
14. **LJ potential** (`lennard_jones_force`, `lennard_jones_energy`) — trivial
15. **MD observables** (`radial_distribution_function`, `velocity_autocorrelation`, `msd`) — compose from `histogram`, `autocorrelation`
16. **Split-step Fourier TDSE** (`split_step_fourier`, `wave_packet_gaussian`) — compose from `fft`
17. **Band structure** (`tight_binding_chain`, `density_of_states_numerical`) — compose from `sym_eigen`, `histogram`
18. **GR tensors** (`metric_christoffel`, `riemann_tensor`, `ricci_tensor`, `einstein_tensor`) — compose from `mat_mul`, `inv`, `derivative_central`
19. **PIC charge deposition** (`pic_charge_deposit_ngp/cic`, `boris_pusher`) — core scatter-gather pattern
20. **Kepler anomaly** (`kepler_anomaly_solve`, `orbital_state_from_elements`) — compose from Newton root-finder

### Tier 3 — Complex, research-level, require new infrastructure

21. **FMM/Barnes-Hut** — need octree data structure (not in tambear)
22. **FDTD** (`fdtd_2d_te_step`, PML boundaries) — moderate-complex
23. **Ewald summation / PME** — complex, requires FFT + real-space summation
24. **VQE/QAOA** — circuit + optimization layer
25. **DMRG sweep** — tensor network infrastructure first
26. **MHD 1D Roe solver** — complex wave structure
27. **Vlasov BSL** — requires interpolation in 2D phase space
28. **HF SCF** — self-consistency loop

---

## Accumulate+Gather Decompositions Not Yet in tambear

These are new *grouping patterns* that would expand the accumulate+gather algebra:

| Pattern | Description | Example |
|---------|-------------|---------|
| Spatial neighbor grouping | Group particles by grid cell, accumulate over neighbors within radius h | SPH density, LBM streaming, PIC deposit |
| Hierarchical tree grouping | Group sources by octree level, accumulate far-field multipoles | Barnes-Hut, FMM |
| Phase-space grouping | Group over (x,v) pairs within Δx×Δv cell | Vlasov moments |
| Qubit-register grouping | Group 2^n amplitudes by which qubit k is 0 or 1 | Quantum gate application |
| Bond-list grouping | Group particle pairs within cutoff, accumulate pairwise forces | MD, SPH |

All of these are Kingdom A (fully parallel over groups) — the TAM scheduler can handle them without any new kingdom classification.

---

## Key Insight: High Leverage from Existing Primitives

The following primitives are already in tambear and would power many physics gaps with minimal new code:

| Existing primitive | Physics use cases unlocked |
|--------------------|--------------------------|
| `fft` / `fft2d` | Fraunhofer diffraction, angular spectrum, spectral Poisson (PIC), split-step TDSE, NLSE |
| `solve_tridiagonal` | Fokker-Planck CN, Crank-Nicolson TDSE, FEM 1D |
| `svd` | MPS bond truncation (DMRG), quantum state tomography |
| `sym_eigen` | Tight-binding bands, Hubbard exact diag, Rayleigh-Ritz |
| `union_find` | Wolff cluster MC, Swendsen-Wang bond percolation |
| `poisson_sor` | Laplace equation, electrostatics (reuse with zero source) |
| `autocorrelation` | Velocity autocorrelation (diffusion coefficient) |
| `histogram` | g(r) RDF, density of states, Wang-Landau accumulation |
| `rng::sample_normal` | Langevin noise, VMC walkers, QMC |
| `derivative_central` | Christoffel symbols from metric |
| `mat_mul` | ABCD matrices, quantum gates, tensor contractions |
| `metropolis_hastings` | PIMC, VMC, quantum MC |
