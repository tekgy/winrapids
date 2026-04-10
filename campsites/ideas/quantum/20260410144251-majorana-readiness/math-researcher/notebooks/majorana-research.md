# Majorana Readiness — Mathematical Research Notes

## What "Majorana Readiness" Means for Tambear

Majorana fermions are their own antiparticle (γ = γ†). In quantum computing,
Majorana zero modes at the ends of topological superconductor wires provide
topologically protected qubits. Microsoft's approach uses these for fault-tolerant
quantum computing.

For tambear, "Majorana readiness" means: ensuring our mathematical primitives
can express the computations needed for topological quantum computing
simulation and analysis.

## Mathematical Primitives Needed

### 1. Pfaffian

The Pfaffian of a 2n×2n antisymmetric matrix A:
Pf(A)² = det(A)

Critical for: free-fermion models, Majorana pairing Hamiltonians.

Algorithm: O(n³) via Householder-like reduction to block-diagonal form.
Each 2×2 block contributes one factor to the Pfaffian.

Parameters: `a: &Mat` (must be antisymmetric) → `f64`

### 2. Matrix Functions on Complex Matrices

Majorana Hamiltonians are:
H = i/4 Σ A_jk γ_j γ_k

where A is real antisymmetric, γ are Majorana operators satisfying {γ_j, γ_k} = 2δ_jk.

Need: eigendecomposition of antisymmetric matrices (imaginary eigenvalues ±iλ_k)
This is a special case of the general eigendecomposition gap in linear algebra.

### 3. Topological Invariants

- **Chern number**: C = (1/2π) ∫ F(k) dk where F is Berry curvature
  - For discrete systems: lattice Berry phase via overlap matrices
  - Parameters: `bloch_hamiltonian_fn(k)`, `k_grid`

- **Z₂ invariant**: parity of Chern number
  - Simpler to compute: det(w_mn) at time-reversal-invariant momenta

- **Winding number**: for 1D topological insulators/superconductors
  - W = (1/2π) ∫ d/dk [arg(det(H(k)))] dk

### 4. Bogoliubov-de Gennes (BdG) Hamiltonian

Standard form for superconducting systems:
H_BdG = [H₀   Δ ]
        [Δ†  -H₀ᵀ]

Particle-hole symmetry: τ_x H_BdG* τ_x = -H_BdG

Need: eigendecomposition respecting particle-hole symmetry
(eigenvalues come in ±E pairs, Majorana zero modes at E=0)

### 5. Kitaev Chain

Simplest 1D topological superconductor:
H = -μ Σ c†_i c_i - t Σ (c†_i c_{i+1} + h.c.) + Δ Σ (c_i c_{i+1} + h.c.)

Phase diagram:
- |μ| < 2t: topological (Majorana zero modes at ends)
- |μ| > 2t: trivial

This is the Brusselator of topological physics — the minimal model.

Simulation primitives:
- Build BdG Hamiltonian (tridiagonal + anti-diagonal blocks)
- Eigendecomposition
- Track gap closure (phase boundary)
- Compute winding number

## What Tambear Already Has That's Relevant

1. `sym_eigen` — for Hermitian Hamiltonians (need to extend to complex)
2. `tridiagonal_solve` — for 1D chain models
3. `det` — for invariant computation
4. `svd` — for entanglement spectrum
5. `rk4_system` — for time evolution (Schrödinger equation)

## What Tambear Needs

1. **Complex matrix support** — biggest gap. Current `Mat` is real-only.
   Complex eigenvalues appear in: non-symmetric systems, quantum mechanics,
   Fourier transforms. This is a fundamental infrastructure need.

2. **Sparse matrix support** — quantum Hamiltonians are sparse (local interactions).
   Full dense matrices are wasteful for n > 100 sites.

3. **Pfaffian** — specific to Majorana/free-fermion calculations.

4. **Berry phase / Chern number** — topological invariant computation.

5. **Entanglement entropy** — from reduced density matrix eigenvalues.
   S = -Σ λᵢ log λᵢ (Shannon entropy of eigenvalues!)
   This connects directly to the information theory primitives.

## The Structural Rhyme

Majorana zero modes exist at PHASE BOUNDARIES in parameter space.
The Brusselator's Hopf bifurcation exists at a PHASE BOUNDARY.
GARCH persistence α+β ≈ 1 is a PHASE BOUNDARY.

The mathematics is the same: track eigenvalues of a parameter-dependent
operator, detect when they cross zero (or the imaginary axis).

This is the COPA boundary theorem from the taxonomy: different phase
boundaries (Fock, Banach) are structurally isomorphic.

## Priority Assessment

For tambear's immediate needs (financial signal farm), Majorana readiness
is Tier 3 — the primitives it needs (complex matrices, sparse support)
are important for OTHER reasons and will arrive naturally. The specific
topological invariants are specialized.

However, the conceptual framework — phase boundaries, eigenvalue tracking,
topological protection — connects directly to the regime detection and
criticality families that ARE Tier 1.
