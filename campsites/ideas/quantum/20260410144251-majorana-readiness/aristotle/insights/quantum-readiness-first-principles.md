# Quantum Readiness: What Would Tambear Actually Need?

*2026-04-10 — Aristotle*

## The Question Reframed

The campsite asks: "what math does tambear need for quantum hardware?" This is the wrong question. The right question is: **what changes in the accumulate+gather framework when the hardware substrate is quantum?**

My debut deconstruction established: the number "two" in two operations comes from hardware. SIMT hardware has two sources of parallelism (algebraic structure + flat memory). Quantum hardware has DIFFERENT sources of parallelism. So the decomposition should be different.

## What Quantum Hardware Actually Provides

A quantum computer does not compute faster by having more cores. It computes differently by exploiting:

1. **Superposition**: A qubit is in all states simultaneously. N qubits represent 2^N states at once.
2. **Interference**: Amplitudes add. Wrong answers cancel; right answers amplify.
3. **Entanglement**: Correlations without classical explanation. Measuring one qubit affects another.

These are NOT the same sources of parallelism as SIMT. They suggest different fundamental operations:

| SIMT | Quantum |
|------|---------|
| accumulate (algebraic combining) | interfere (amplitude combining) |
| gather (random memory access) | entangle (correlated state creation) |
| — | measure (irreversible state collapse) |

**THREE operations, not two.** Quantum computation needs a third verb — measurement — that has no classical analog. Measurement is irreversible, probabilistic, and destructive. It's the Fock boundary made physical.

**COPA boundary connection (math-researcher, same session)**: In the (rho, sigma, tau) kingdom taxonomy, measurement is the sigma-to-tau crossing. The Born rule p = |psi|^2 takes complex amplitudes (sigma: reversible unitary evolution) to real probabilities (tau: irreversible classical output). Interfere and entangle are sigma-domain operations. Measure is the COPA boundary crossing. This maps quantum measurement exactly to tambear's Fock boundary theorem.

**Complex f64 is a foundation-level gap, not quantum-specific (math-researcher)**: Non-symmetric eigendecomposition, Hilbert transform, full FFT, characteristic functions, transfer function analysis — all blocked by the same missing primitive. Complex DotProduct = paired-accumulate with (real_sum, imag_sum) state. Kingdom A, same scan structure, doubled width. No new architecture needed.

## What Tambear Already Has That's Quantum-Relevant

1. **Tensor contraction** (Tiled accumulate): Simulating quantum circuits on classical hardware requires tensor network contraction. tambear's TiledEngine IS a tensor contractor.

2. **Eigendecomposition**: Quantum chemistry (VQE) requires computing eigenvalues of Hamiltonians. Already have sym_eigen, power_iteration.

3. **Optimization**: QAOA maps combinatorial optimization to quantum circuits. The classical optimization loop (gradient descent variants) already exists.

4. **Statistical testing**: Verifying quantum results requires comparing distributions of measurement outcomes to theoretical predictions. KS test, chi-squared, divergence measures — all exist.

5. **Complex arithmetic**: Quantum states are complex-valued. tambear is f64. Would need complex f64 (or paired f64 for real/imaginary).

## What's Actually Missing

### Critical
- **Complex number support**: Quantum amplitudes are complex. accumulate with complex Add, complex DotProduct, complex MatMul.
- **Unitary operations**: Quantum gates are unitary matrices. Need efficient unitary matrix multiply (which is just complex TiledEngine).

### Important  
- **Sparse Hamiltonian simulation**: Most physical Hamiltonians are sparse. Sparse Tiled accumulate.
- **Tensor network contraction ordering**: Optimal contraction order is NP-hard in general. Need heuristics (greedy, genetic).

### Nice-to-Have
- **Stabilizer formalism**: Clifford circuits can be simulated classically in polynomial time. This is a special accumulate where the state space is the Pauli group.
- **Fermion-to-qubit mappings**: Jordan-Wigner, Bravyi-Kitaev transforms. These are specific gather permutations.

## The Honest Assessment

**For quantum simulation on classical hardware**: tambear is 80% ready. TiledEngine for tensor contraction + eigendecomposition + optimization loop covers most quantum chemistry workloads. The gap is complex number support.

**For programming actual quantum hardware**: tambear would need a fundamentally different architecture. The accumulate+gather decomposition doesn't apply — quantum computation is interfere+entangle+measure. The operator algebra is different (unitary groups, not monoids). The memory model is different (no-cloning theorem = no gather).

**For hybrid classical-quantum**: tambear handles the classical parts (optimization loop, data preprocessing, result analysis). The quantum parts run on quantum hardware. tambear's value is in the INTERFACE — preparing inputs and analyzing outputs.

## Recommendation

Don't try to make tambear a quantum computing framework. Instead:

1. **Add complex f64 support** to TiledEngine (the smallest change with the largest impact for quantum simulation).
2. **Implement tensor network contraction** as a higher-level pattern over TiledEngine (this is useful for classical ML too — tensor networks are used in DMRG, MERA, etc.).
3. **Keep the quantum-specific parts** (circuit representation, gate decomposition, error correction codes) OUT of tambear and in a separate crate that USES tambear for the linear algebra.

The Majorana chip is a topological qubit architecture. Its main math need is topological invariant computation (Chern numbers, winding numbers, Berry phases). These ARE accumulate operations (integrals over parameter spaces = accumulate with kernel). tambear can compute them classically. Whether the Majorana hardware itself runs accumulate+gather is a different question — it doesn't. It runs interfere+entangle+measure.
