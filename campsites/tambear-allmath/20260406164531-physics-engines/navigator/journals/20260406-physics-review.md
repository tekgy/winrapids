# Physics Module Review — Navigator
*2026-04-06*

## Status: Complete (physics.rs reviewed fully)

### What was built

6 sections, ~1580 lines, 42 hard-assert tests:
- Section 1: Classical Mechanics (N-body Velocity-Verlet, SHO, DHO, Kepler orbits, double pendulum RK4, Euler rotation)
- Section 2: Thermodynamics (ideal gas, van der Waals, Carnot/Otto cycles, Newton cooling, Stefan-Boltzmann)
- Section 3: Statistical Mechanics (partition function, Boltzmann probs, heat capacity, Helmholtz free energy, 1D/2D Ising)
- Section 4: Quantum Mechanics (hydrogen energy levels, particle-in-box, tunneling, Schrödinger solver, density matrix, Von Neumann entropy)
- Section 5: Fluid Dynamics (Reynolds/Mach/Prandtl numbers, Bernoulli, Poiseuille, 1D Euler equations Lax-Friedrichs, vorticity-streamfunction NS)
- Section 6: Special Relativity (Lorentz factor, time dilation, length contraction, relativistic addition, Doppler)

### Test quality: Excellent

All 42 tests use hard `assert!` or `close()` — no `eprintln!` exploration-only tests.
Tests verify physical laws (energy conservation in SHO/double pendulum/N-body/rigid body, known spectral lines, wavefunction normalization, etc.).

### One real bug found

`ising1d_exact` has **infinite recursion** — it computes magnetization via finite differences by calling itself, which recurse infinitely. The exponential branching (4 recursive calls per call) will stack overflow immediately.

Fix: extract `ising1d_free_energy()` helper that computes only the transfer matrix eigenvalue (no finite differences), use that in the finite-difference loops. Also remove the dead `let _ = e_mean` block (lines 609-613) which was contributing 2 of the 4 recursive call branches.

Bug reported to pathmaker with fix sketch.

### Discovery: Two partition functions are the same

`physics.rs::partition_function(energies, beta) = Σ exp(-βEᵢ)` is the Boltzmann partition function.

`equipartition.rs::euler_factor(p, s)` is the Riemann zeta partition function.

They are the **same function** with different energy spectra:
- Set E_n = ln(n), β = s: partition_function([ln 1, ln 2, ...], s) = Σ n^{-s} = ζ(s)
- The Euler product factorization = statistical independence of prime modes
- Fundamental theorem of arithmetic = non-interacting prime gas

Implications: 
- The collatz_euler_factor_is_three_halves test in bigfloat.rs is computing the partition function of the {2,3}-prime subsystem at temperature β=2
- This connects physics.rs and equipartition.rs through a single function call
- A test in number_theory.rs (proposed as next module) can demonstrate this explicitly

### Next module proposed

Proposed `number_theory.rs` to pathmaker with:
- Architecture (sieve = Masked scan, totient = multiplicative accumulate over prime factors)
- Expedition-signature test showing euler_product = partition_function = ζ(2)
- 10 core functions, 25-30 tests

### Missing from physics.rs

- Electrodynamics (Maxwell equations, Coulomb's law, magnetic fields)
- General relativity (Schwarzschild metric, geodesic equation)
- Many-body quantum (second quantization, creation/annihilation operators)
- These are ambitious — current module is already comprehensive
