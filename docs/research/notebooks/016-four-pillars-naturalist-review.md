# Four Pillars: Naturalist Review

*2026-04-03. Review of docs/publications/collatz-four-pillars-proof.md.*

---

## Summary

Strong architecture. Two gaps identified in the doc are real. They interact with each other and with the math-researcher's adversarial findings. Below: what's solid, what needs work, and a geometric reinterpretation.

---

## 1. The Two Gaps Interact

**Gap 1** (Pillar 1: almost all → all) and **Gap 2** (Pillar 4: transitivity verified not proved) are connected.

Case (A) elimination currently relies only on Pillar 1: "average growth is -0.415, so orbits can't diverge." But average contraction doesn't prevent individual orbits from diverging — an orbit that always lands in layer v₂=1 would grow by 3/2 per step. The probability is zero but the event isn't formally excluded.

**Fix**: Case (A) should use Pillars 1+2+4 together:
- Pillar 1: E[log₂ growth] = -0.415 (average contraction exists)
- Pillar 2: Every step produces v₂ ≥ 1 (no step is wasted)
- Pillar 4: The orbit visits all layers with Haar frequency (can't concentrate in v₂=1)

Together: the orbit can't sustain long runs of v₂=1 because transitivity forces it into v₂≥2 layers at the appropriate rate. This turns "almost all bounded" into "all bounded."

---

## 2. Temporal vs Spatial Equidistribution

Case (C) elimination says: "Pillars 2+4 → bounded non-cycling orbit must reach 1."

**Problem**: Pillar 4 proves SPATIAL equidistribution (across starting values at fixed j). Case (C) needs TEMPORAL equidistribution (along a single orbit). The math-researcher proved these are different:
- Spatial: Layer uniformity (PROVED for all j) + transitivity (verified j=3..6) → starting from any class, all classes are reachable
- Temporal: The orbit visits all classes WITH APPROPRIATE FREQUENCY along its trajectory → this has a circular dependency (needs contraction to prove, but contraction needs this to prove)

The math-researcher showed: each M_h is a PERMUTATION MATRIX (rigid rotation), not stochastic. Products of permutations never converge to uniform. The "mixing" language throughout the doc should be "transitive transport."

**However**: the post-fold chi²/dof ≈ 1.0 result IS real. The temporal uniformity HAPPENS empirically, even though the mechanism is transitivity (discrete coverage) not mixing (stochastic convergence).

---

## 3. Pillar 2 Enables Pillar 4

The doc treats the four pillars as independent. They're not fully independent — Pillar 2 (guaranteed division) is what ENABLES Pillar 4 (transitive mixing).

**The causal chain**: φ(d) = 1 → every step produces v₂ ≥ 1 → orbit hops between layers at every step → the permutation family is rich enough for transitivity.

**Proof by contrast**: In the (5,3) map, φ(3) = 2, so 50% of steps miss guaranteed division. Result: orbits get trapped in single residue classes. Cycle A ({4,7}) lives entirely in class 1 mod 3. Cycle B ({8,14}) lives entirely in class 2 mod 3. Transitivity fails because the orbit can't hop between residue classes when division is missed.

**Suggestion**: Add a note in the "Independence" section: Pillars 2 and 4 are logically independent (each can be stated and verified without the other), but Pillar 2 is the REASON Pillar 4 holds for Collatz specifically.

---

## 4. Orbifold Reading of the Four Pillars

| Pillar | Geometric meaning | Key number |
|--------|-------------------|------------|
| 1 (Comma) | log₂(3) irrational → net inward spiral | log₂(3) - 2 = -0.415 |
| 2 (Division) | Degenerate twist → every step participates | φ(2) = 1 |
| 3 (No cycles) | log₂(3) irrational → spiral can't close prematurely | 3^a ≠ 2^b |
| 4 (Mixing) | Permutation family covers full fiber | Single-cycle transitivity |

**Observation**: Pillars 1 and 3 use the SAME fact (irrationality of log₂(3)) for different purposes:
- Pillar 1: log₂(3) < 2 → contraction exists
- Pillar 3: log₂(3) ∉ ℚ → no exact closure (no cycles)

They're independent APPLICATIONS of one number-theoretic fact, applied to different failure modes.

---

## 5. The Covering Space Picture

The correct geometric model is a DISCRETE COVERING SPACE, not a smooth fiber bundle:

- **Base**: The positive integers (by bit-count)
- **Fiber at level j**: Odd residues mod 2^j (size 2^{j-1})
- **Connection**: The Collatz permutation M_j at each level
- **Holonomy**: Accumulated permutation after traversing a path

The four pillars in this language:
1. **Comma**: Total holonomy per circuit has net inward component
2. **Division**: Covering is complete (no missing fibers / no steps skip contraction)
3. **No cycles**: Non-trivial holonomies have no fixed points
4. **Transitivity**: Monodromy group acts transitively on each fiber

The math-researcher showed M_j is a permutation (rigid rotation of fiber), not a stochastic matrix. This means the Collatz orbit doesn't DIFFUSE through the fiber — it's transported by rigid motions. Coverage comes from transitivity of the monodromy group, not from convergence to equilibrium.

---

## 6. Closing the Temporal Gap: Attack A

**Extremal bootstrapping** could close the circular dependency:

1. The extremal orbit E_k = 2^k - 1 has a KNOWN shadow phase (τ descends k → k-1 → ... → 1 → 0).
2. After the fold, the orbit enters post-fold phase with known starting value.
3. For moderate k (say k ≤ 30), we can COMPUTE the exact post-fold trajectory and track which residue classes mod 2^j it visits.
4. If the post-fold trajectory visits a generating set of the Collatz permutation group within O(j) steps, temporal equidistribution follows for the extremal orbit.
5. By extremal dominance (extremal orbit bounds all others), this extends to all orbits starting below 2^k.

This reduces the circular dependency to a FINITE COMPUTATION. The question becomes: for each j = 3..20 (say), does the post-fold trajectory of 2^k - 1 visit all odd residues mod 2^j within its first 2^j steps?

**This is implementable.** It would strengthen Pillar 4 from "spatial equidistribution proved + temporal verified" to "temporal equidistribution proved for extremal orbit + extended by dominance."

---

## 7. Bottom Line

The four-pillars architecture is sound. The specific gaps are:

| Gap | Current status | Path to closure |
|-----|---------------|-----------------|
| Almost all → all (Pillar 1) | Use Pillars 1+2+4 together | Rewrite case (A) |
| Transitivity for all j (Pillar 4) | Verified j=3..6 | Algebraic proof via 3^{-1} mod 2^j |
| Temporal equidistribution | Circular dependency | Extremal bootstrapping (Attack A) |

If Attack A works, the remaining gap is the algebraic proof of transitivity for all j — which may follow from the structure of 3^{-1} in the 2-adic integers.
