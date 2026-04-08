# Audit Correction: physics.rs + dim_reduction.rs
Created: 2026-04-06T17:07:04-05:00  
By: scout

**This corrects errors in the initial gap taxonomy and audit.**

---

## physics.rs — 1614 lines, ENTIRELY MISSED

`physics.rs` was created at 17:04 on 2026-04-06 — 22 minutes into my initial audit. The pathmaker wrote it *during* my audit session. My original gap taxonomy incorrectly listed "physics engines entirely absent."

### What physics.rs actually contains:

**Section 1 — Classical Mechanics**
- N-body simulation (Velocity-Verlet integrator, O(n²))
- Simple Harmonic Oscillator (exact analytical + energy)
- Damped harmonic oscillator (underdamped analytical)
- Kepler orbit (orbital elements from (r, v, M))
- Vis-viva equation
- Double pendulum (RK4 integration + energy)
- Rigid body Euler equations (torque-free rotation)

**Section 2 — Thermodynamics**
- Ideal gas (pressure, temperature, internal energy, entropy change)
- Van der Waals gas (pressure, critical point)
- Carnot and Otto efficiency
- Heat transfer (Fourier conduction, Newton cooling, Stefan-Boltzmann radiation)

**Section 3 — Statistical Mechanics**
- Canonical ensemble (partition function, mean energy, heat capacity, Helmholtz free energy, Boltzmann probabilities, Gibbs entropy)
- Quantum harmonic oscillator statistics (QHO energy levels, Bose-Einstein occupation, Planck spectral energy, Wien displacement)
- **1D Ising model (exact via transfer matrix!)**
- **2D Ising model (Metropolis Monte Carlo!)**
- Arrhenius rate equation, equilibrium constant

**Section 4 — Quantum Mechanics**
- Hydrogen atom energy levels and spectral wavelengths
- Particle in a box (exact energy + wavefunction)
- Quantum tunneling transmission (WKB)
- Quantum state: normalize_state, time_evolve_state, expectation_value, uncertainty
- **Heisenberg uncertainty product**
- Density matrix: trace, purity, von Neumann entropy (diagonal)
- **1D Schrödinger equation (finite difference, tridiagonal eigenvalue)**

**Section 5 — Fluid Dynamics**
- Dimensionless numbers (Reynolds, Mach, Prandtl, Nusselt-Dittus-Boelter)
- Bernoulli equation, Poiseuille flow (rate + velocity profile)
- **1D Euler equations (compressible flow, Lax-Friedrichs scheme)**
- CFL timestep condition
- **2D Navier-Stokes vorticity-streamfunction (Poisson SOR + vorticity advection)**

**Section 6 — Special Relativity**
- Lorentz factor, relativistic kinetic energy, momentum
- E = mc²
- Time dilation, length contraction
- Relativistic velocity addition
- Relativistic Doppler shift

42 tests.

### What physics.rs is still missing:
- **General Relativity** (geodesic equations, Christoffel symbols, ADM formalism)
- **Symplectic integrators** (Verlet/leapfrog — double pendulum uses RK4 which doesn't preserve phase space volume)
- **Barnes-Hut / Fast Multipole** (N-body is O(n²); no tree code)
- **Lattice Boltzmann method**
- **Quantum circuits / gates** (the amplitude state representation exists but no gate operations)
- **DMRG / tensor network** for quantum many-body
- **Magnetohydrodynamics (MHD)**
- **Radiative transfer equation**
- Full 3D Navier-Stokes (only 2D vorticity-streamfunction)

---

## dim_reduction.rs — t-SNE and NMF exist but aren't top-level exported

My initial audit said dim_reduction.rs had "PCA via SVD, KPCA." That was wrong.

**dim_reduction.rs actually contains:**
- PCA (via SVD) ✓ (correctly noted)
- Classical MDS ← missed this
- **t-SNE** ← missed this (300 lines of full implementation)
- **NMF (Non-negative Matrix Factorization, Lee & Seung multiplicative updates)** ← missed this

These are accessible as `tambear::dim_reduction::tsne(...)` but not re-exported at the top level via `pub use`. Not a bug, just not prominent.

Also: my initial gap taxonomy listed "tensor decomposition entirely absent" — but **NMF is already in dim_reduction.rs.** The actual remaining tensor gaps are: CP decomposition (ALS), Tucker, Tensor Train, NTF.

---

## multivariate.rs — LDA was present all along

`multivariate.rs` has `lda()` (Linear Discriminant Analysis). I noted multivariate.rs in my audit as covering MANOVA, Hotelling T², CCA — but I missed LDA. My gap taxonomy said "NO LDA" in the dim_reduction section.

---

## Corrected Gap Assessment: Physics

**Previously stated:** "Physics entirely absent"  
**Corrected:** Physics is substantially covered. Remaining gaps:
- GR (geodesic, Christoffel, curvature tensors)
- Symplectic integrators (phase-space-preserving: Verlet, PEFRL)  
- Barnes-Hut / FMM (O(n log n) N-body)
- Quantum circuits and gates
- Lattice Boltzmann
- Full 3D NS

**Previously stated:** "Tensor decomposition entirely absent"  
**Corrected:** NMF is present. Remaining: CP-ALS, Tucker, TT-SVD, NTF.

**Previously stated:** "t-SNE, NMF absent"  
**Corrected:** Both present in dim_reduction.rs.

**Previously stated:** "LDA absent"  
**Corrected:** LDA present in multivariate.rs.

---

## Dynamic codebase note

The pathmaker is actively writing code. At least `physics.rs` was created after my initial file scan. My audit represents a snapshot; the pathmaker may have already addressed other gaps I identified.

Files modified after my initial scan: interpolation.rs, irt.rs, lib.rs, multivariate.rs, neural.rs, nonparametric.rs, rng.rs, series_accel.rs, special_functions.rs, tbs_executor.rs, time_series.rs, using.rs

Recommendation: re-run the audit of these modified files to see what was added/changed.
