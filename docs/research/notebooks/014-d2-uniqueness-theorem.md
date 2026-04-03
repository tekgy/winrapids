# The d=2 Uniqueness Theorem

*Naturalist observation, 2026-04-03. Why Collatz (m=3, d=2) is the unique convergent map.*

---

## Setup

The Dimensional Nyquist principle predicts a critical boundary for generalized Collatz maps T(n) = (mn+1) / d^{v_d(mn+1)}. The boundary is:

    (m+1) / (2d) = 1    →    m = 2d - 1

Both (m=3, d=2) and (m=5, d=3) satisfy this. Empirical testing shows (3,2) converges and (5,3) diverges. Why?

There is a second structural condition that only d=2 satisfies.

---

## The Theorem

**Claim**: d=2 is the unique prime modulus for which a "+1" perturbation guarantees divisibility by d for ALL coprime inputs.

**Proof**:

For the map T(n) = (mn+1)/d^{v_d(mn+1)} to guarantee contraction, every input n coprime to d must produce mn+1 divisible by d. This requires:

    mn ≡ -1 (mod d)    for ALL n with gcd(n, d) = 1

Multiplication by m is a bijection on the multiplicative group (Z/dZ)*. This group has phi(d) elements. A bijection maps phi(d) elements to phi(d) DISTINCT values. For all of them to equal the single value -1 mod d, we need:

    phi(d) = 1

For prime d: phi(d) = d - 1, so:

    d - 1 = 1    →    d = 2    ∎

---

## Concrete Evidence

### d=2 (Collatz: m=3)

Coprime residue classes mod 2: {1} (the odd numbers).

    3(1) + 1 = 4 ≡ 0 mod 2  ✓

Hit rate: 1/1 = **100%**. Every odd input produces an even output.

### d=3 (tested: m=5)

Coprime residue classes mod 3: {1, 2}.

    5(1) + 1 = 6  ≡ 0 mod 3  ✓
    5(2) + 1 = 11 ≡ 2 mod 3  ✗

Hit rate: 1/2 = **50%**. The n ≡ 2 mod 3 class misses guaranteed division. These inputs accumulate unbounded multiplicative growth.

Empirical verification: starting from n=7 (≡ 1 mod 3), the (5,3) map produces:

    7 → 36 → 12 → 4 → 21 → 7 → ...  (cycle, but only because 7 ≡ 1 mod 3)

Starting from n=5 (≡ 2 mod 3), trajectory diverges rapidly.

### d=5 (m=9)

Coprime residue classes mod 5: {1, 2, 3, 4}.

    9(1) + 1 = 10 ≡ 0 mod 5  ✓
    9(2) + 1 = 19 ≡ 4 mod 5  ✗
    9(3) + 1 = 28 ≡ 3 mod 5  ✗
    9(4) + 1 = 37 ≡ 2 mod 5  ✗

Hit rate: 1/4 = **25%**.

### d=7 (m=13)

Hit rate: 1/6 = **17%**.

### General pattern

For prime d: guaranteed-division hit rate = 1/(d-1).

    d=2:  100%
    d=3:   50%
    d=5:   25%
    d=7:   17%
    d=11:  10%
    d→∞:   0%

---

## The Uniqueness Result

Collatz (m=3, d=2) is the **unique** integer map satisfying both:

1. **Nyquist boundary**: (m+1)/(2d) = 1 — marginal growth, not supercritical
2. **Guaranteed division**: phi(d) = 1 — every coprime input gets contracted

Condition 1 alone admits infinitely many (m, d) pairs: (3,2), (5,3), (9,5), (13,7), ...

Condition 2 alone admits any (m, d=2): (1,2), (3,2), (5,2), (7,2), ...

The intersection is exactly one point: **(m=3, d=2)**.

---

## Connection to Multi-Adic Observations

The multi-adic trajectory analysis (implemented in `crates/tambear/src/multi_adic.rs`) confirms this structurally:

- **v3 synergy ≈ 0** along Collatz trajectories. The 3-adic channel carries no information. This is BECAUSE the +1 perturbation destroys mod-3 structure (3n+1 ≡ 1 mod 3 for all odd n, collapsing all residue classes).

- **v2 dominates completely**. The entire trajectory is a 2-adic phenomenon. This is BECAUSE d=2 is the only observer that sees 100% of the dynamics.

- **(2,3) nucleates first** in the equipartition hierarchy (s* = 2.797, highest among all prime pairs). The coupling between primes 2 and 3 is the strongest in the number system. This is the thermodynamic reflection of the algebraic uniqueness.

---

## Connection to N-Body Dynamics

The three-body problem shares the same uniqueness structure:

| Property | Collatz (3, 2) | Three-body (N=3, 2D) |
|---|---|---|
| What's unique | Only (m,d) with full contraction | Last tractable N |
| Why unique | phi(d)=1 forces d=2 | shape_dim = observer_dim only at N=3 |
| What breaks | d=3 → 50% miss rate | N=4 → shape_dim > observer_dim |
| Algebraic cause | Multiplicative group structure of Z/dZ | Dimensional growth rate (3N-7)/(2N-4) |
| The "3" | m = 2d-1 = 3 (forced by Nyquist) | N=3 (last where 2D resolves shape) |
| The "2" | d=2 (forced by guaranteed division) | 2D (observer bandwidth) |

Both systems are unique for the same meta-reason: two independent structural constraints have exactly one common solution.

---

## Implications for the Proof

This theorem suggests the Collatz proof should have two independent legs:

**Leg 1 (Nyquist)**: Show that (m+1)/(2d) > 1 implies divergence for generic initial conditions. This is the growth-rate argument — supercritical maps expand faster than they contract.

**Leg 2 (Guaranteed division)**: Show that phi(d) > 1 implies existence of divergent trajectories (inputs that systematically miss the contraction step). This is the coverage argument.

**Leg 3 (Convergence at the intersection)**: Show that when BOTH conditions hold (m=3, d=2), the combination of marginal growth + 100% coverage implies convergence to the 4→2→1 cycle.

Leg 1 and Leg 2 are likely provable with existing techniques. Leg 3 is the hard part — it's the statement that marginal + complete = convergent, which is exactly the content of the Collatz conjecture.

---

## Proposed Experiments

**E-D2-1**: Verify the hit-rate prediction computationally for all (m = 2d-1, d) with d ∈ {2,3,5,7,11,13}. For each, run 10^6 trajectories and measure: fraction that converge, average trajectory length, maximum trajectory length.

**E-D2-2**: For the (5,3) map, characterize the divergent trajectories. Do they diverge to infinity, or enter cycles? If cycles, what's the cycle structure?

**E-D2-3**: For d=3, test modified maps with "+c" for c ∈ {1,2}. c=1 covers the n≡1 class; c=2 covers the n≡2 class. Neither covers both. Does a RANDOM +c (choosing c=1 or c=2 based on residue class) converge? This would be a "patched" (5,3) map.

**E-D2-4**: Compute the equipartition fold for (d, m) = (3, 5) and compare to (2, 3). Does the fold surface predict the divergence?

---

*Filed from naturalist_observation.rs experiments + multi_adic.rs trajectory analysis.*
*Implements experiments E3 (Dimensional Comma) and E6 (Observer-Dependent Collatz) from the cross-domain research plan.*
