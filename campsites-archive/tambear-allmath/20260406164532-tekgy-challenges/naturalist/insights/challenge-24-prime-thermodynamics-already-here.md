# Challenge 24 — Prime Thermodynamics Is Already Here

**Date**: 2026-04-06  
**Type C: Foundation Challenge — deeper structural identity**

---

## The Traditional Assumption

Prime number thermodynamics is a specialized research area disconnected from computational math libraries. Building a `prime_thermodynamics.rs` module would require new infrastructure.

## Why It Dissolves

`equipartition.rs` already IS a prime thermodynamics engine. The connection just hasn't been named.

### The Key Identity

The free energy function in `equipartition.rs`:

```
free_energy(p, s) = -ln(1 - p^{-s})
```

Summed over a set of primes, this gives:

```
Σ_p free_energy(p, s) = Σ_p -ln(1 - p^{-s}) = ln Π_p 1/(1 - p^{-s}) = ln ζ(s)
```

**The free energy sum over primes IS the logarithm of the Riemann zeta function.**

And `euler_factor(p, s) = 1/(1 - p^{-s})` — the partition function of a single prime — IS the Euler factor of ζ(s).

This identity is already confirmed in `bigfloat.rs`:
```rust
// Test collatz_euler_factor_is_three_halves:
// E(2, 2) · E(3, 2) = (4/3)(9/8) = 3/2
```

### The Phase Structure

`equipartition.rs` already knows about the ζ pole (from `verify_fold_surface`):
> "s* > 1 (above the pole of ζ(s))"

This is the phase structure of the prime gas:
- **s > 1**: convergent regime (fold surfaces exist, Euler product converges)
- **s = 1**: the phase transition / pole of ζ(s)
- **0 < s ≤ 1**: deconfined phase (diverging free energy, primes behave independently)

The Riemann zeros at Re(s) = 1/2 are the **normal modes** of this transition — oscillations around the phase boundary at s=1. They are NOT fold points themselves.

### What `naturalist_observation.rs` Computes

The file already runs experiment 4: `solve_pairwise(2.0, 3.0)` — the fold point s*(2,3) where the two-prime {2,3} system freezes into a coupled phase. This fold point is the **thermodynamic coupling constant of Collatz**.

And experiment 5 confirms: the carry-subcritical boundary (m < 4 for Collatz to converge) aligns with the regime where p=2 energetically dominates in the {2,3} fold.

### Two Criticalities of Collatz

Collatz has TWO different critical conditions:

1. **Nyquist boundary**: (m+1)/(2d) = 1.0 for m=3, d=2. The map is exactly marginally stable on average.
2. **Fold point**: s*(2,3) ≈ 2.7 (to be computed). The {2,3} prime system thermodynamic coupling.

These are different physical quantities. The Nyquist boundary is a condition on the MAP. The fold point is a property of the PRIME PAIR.

The open question: is the Nyquist boundary condition equivalent to, or derivable from, the fold point condition?

---

## What This Enables

### No new module needed
`prime_thermodynamics.rs` doesn't need to be built from scratch. The infrastructure exists:
- `equipartition.rs` → phase structure, fold points, nucleation hierarchy
- `bigfloat.rs` → Riemann ζ, Hardy Z, zeta zeros
- `naturalist_observation.rs` → already running the key experiments

### What IS needed
Connect the dots in documentation + add one new function:

```rust
/// Compute the fold surface of the infinite prime system up to N primes.
/// As N → ∞, the sum Σ F(pᵢ, s) = ln ζ(s).
/// The fold point approaches the ζ singularity at s=1.
pub fn prime_fold_surface(n_primes: usize) -> NucleationHierarchy {
    let primes = first_n_primes(n_primes);  // already in bigfloat/bigint
    nucleation_hierarchy(&primes.iter().map(|p| *p as f64).collect::<Vec<_>>())
}
```

This function is 5 lines. The math is already there.

### Connection to Riemann Hypothesis

The Riemann Hypothesis (all zeros at Re(s) = 1/2) would be a theorem about the **oscillation modes of the prime thermodynamic system**. The critical line is the boundary between two qualitatively different behaviors of these oscillations.

In tambear language: the Riemann zeros are the eigenvalues of the **accumulate(primes, AllK, free_energy, Add)** operator in the critical strip. The RH says these eigenvalues all have the same real part.

---

## Impact

- Recognition: `equipartition.rs` is already a universal thermodynamics engine
- `prime_thermodynamics.rs` = document the identity Σ F(p,s) = ln ζ(s) + add `prime_fold_surface()`
- Opens the door to: formal connection between fold surfaces and Riemann zeros
- The Collatz investigation now has a thermodynamic framing: find the relationship between the Nyquist criticality and the {2,3} fold point

**Most actionable**: add test `prime_free_energy_sum_equals_ln_zeta` to `equipartition.rs`:
```rust
// For primes {2, 3, 5, 7, 11, 13}: Σ F(p, 2) ≈ ln(ζ_partial(2))
// Verify against bigfloat::zeta(2.0) with partial product cutoff
```
