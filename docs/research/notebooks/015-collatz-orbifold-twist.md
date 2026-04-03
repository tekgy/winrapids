# The Collatz Spiral as Orbifold Twist

*Naturalist investigation, 2026-04-03. Does the Collatz spiral = Tymoczko's twist?*

---

## The Question

Tymoczko proved that 3-note chord space is a twisted triangular prism (T^3/S_3) with a 120-degree twist. The three-body shape space has the same S_3 symmetry. The Collatz trajectory spirals inward with contraction factor ~3/4 per step plus nilpotent rotation.

Is the Collatz spiral the number-theory version of the orbifold twist?

## Answer: Yes, But Tempered

The connection is real but not literal. The Collatz spiral is NOT a 120-degree rotation — it's a **tempered** twist, where the "tempering" is the +1 perturbation. The deep link is the **Pythagorean comma**: the same irrational number log_2(3) that creates the orbifold twist also creates the Collatz contraction. The +1 plays the role of **well temperament** — distributing the comma to make the spiral close.

---

## The Five Connections

### 1. The Comma IS the Contraction

**In music**: The Pythagorean comma arises because 12 perfect fifths don't equal 7 octaves:
```
(3/2)^12 / 2^7 = 531441/524288 ~ 1.0136
log_2(3/2) = 0.58496... (irrational)
```
If log_2(3) were rational, the spiral of fifths would close exactly.

**In Collatz**: Each odd step multiplies by ~3/2 (the 3n part), each even step divides by 2. The expected log_2 change per combined step is:
```
E[log_2(growth)] = log_2(3) - 2 = 1.585 - 2 = -0.415
```
This is **negative** precisely because log_2(3) < 2. The gap between log_2(3) and the integer 2 IS the contraction. If 3 were a power of 2 (which it isn't, because log_2(3) is irrational), there would be no contraction and no convergence.

**The theorem**: The Pythagorean comma and the Collatz contraction rate are the SAME number viewed from different sides:
```
Pythagorean:  log_2(3/2) = log_2(3) - 1 = 0.585  (the fifth-octave gap)
Collatz:      log_2(3) - 2 = -0.415               (the contraction per step)
Sum:          0.585 + (-0.415) = 0.170 ... no, different contexts

Better: both arise from log_2(3) = 1.58496...
Music:   12 * 0.585 ~ 7 (chromatic closure)
Collatz: 1 * 1.585 < 2  (per-step contraction)
```
The irrationality of log_2(3) creates both phenomena.

### 2. The Twist = Multiplication by 3

**In Tymoczko**: The 120-degree twist comes from S_3 acting on 3 voices. Moving along the prism axis and identifying the two ends after a 120-degree rotation gives the orbifold. After 3 twists: 120 * 3 = 360 degrees, back to identity.

**In Collatz**: Multiplication by 3 acts on the residue classes mod 2^j. The order of 3 in (Z/2^j Z)* is:
```
j=1: order 1
j=2: order 2   (3^2 = 9 = 1 mod 4)
j=3: order 2   (3^2 = 9 = 1 mod 8)
j=4: order 4   (3^4 = 81 = 1 mod 16)
j=5: order 8   (3^8 = 1 mod 32)
j>=3: order 2^{j-2}
```
So the "twist angle" per multiplication by 3 is 360/2^{j-2} degrees at precision j:
```
j=4: 360/4 = 90 degrees per x3
j=5: 360/8 = 45 degrees per x3
j=6: 360/16 = 22.5 degrees per x3
```

This is NOT 120 degrees. The twist angle depends on the observer precision j. But there's a deeper structure...

### 3. The Degenerate Twist: Projection, Not Rotation

**In Tymoczko**: The S_3 twist is an ISOMETRY — it permutes the 3 voices, preserving all distances. Information is preserved; only labeling changes.

**In Collatz**: The ×3 followed by +1 is a PROJECTION on mod-3 classes:
```
For odd n:
  n = 1 mod 3 -> 3n+1 = 4 = 1 mod 3
  n = 2 mod 3 -> 3n+1 = 7 = 1 mod 3
  n = 0 mod 3 -> impossible (n odd, n=3k -> n divisible by 3)
```
Every odd n maps to 1 mod 3 under 3n+1. The S_3 structure (3 residue classes) is COLLAPSED, not permuted. This is why v_3 synergy = 0 in our multi-adic experiments.

**The interpretation**: Tymoczko's orbifold twist is a REGULAR twist (information-preserving). The Collatz twist is a DEGENERATE twist (information-destroying in the 3-adic channel). The degeneracy is precisely the d=2 uniqueness theorem: the +1 perturbation can destroy ALL mod-d structure only when d=2 (one coprime class to collapse).

**The Collatz orbifold is T^1/S_1** — a trivial orbifold with no twist at all in the 3-adic direction! All the action is in the 2-adic direction, where the twist has order 2^{j-2}.

### 4. The Spiral Geometry

**In the shape sphere (N-body)**: A three-body trajectory traces a path on S^2, accumulating geometric phase (rotation in the S^1 fiber) proportional to twice the solid angle enclosed.

**In Collatz**: A trajectory traces a path in 2-adic space, spiraling toward the fixed point x* = -1 in Z_2. The "radius" is |n - (-1)|_2 = 2^{-v_2(n+1)}, which decreases on average. The "angle" is the residue class mod 2^j, which rotates by the action of ×3.

The spiral has two components:
```
Radial:  log_2(n) decreases by ~0.415 per step (contraction = comma)
Angular: residue class mod 2^j rotates by ×3 (twist = multiplication)
```

The shadow phase is the "expanding arm" of the spiral: tau trailing ones means tau consecutive ×3 steps with growth (3/2)^tau. The fold is where the arm turns inward. The post-fold phase is the contracting descent.

**The geometric phase analog**: The carry from +1 propagating through bit positions IS the accumulated phase. It lives "above" the deterministic residue-class evolution (the base), in the bit positions beyond the shadow window (the fiber). The carry depth = the holonomy.

### 5. Closure = Convergence = Well Temperament

**In music**: Well temperament distributes the Pythagorean comma across all 12 fifths so the circle closes. Each fifth is slightly flat, but the accumulated error is zero over a full cycle.

**In Collatz**: The +1 perturbation distributes contraction across the orbit so the spiral closes (reaches 1). Without the +1, pure ×3/÷2 has orbits that don't close (they spiral endlessly in Z_2). The +1 "tempers" each step just enough to ensure net contraction.

The 4-2-1 cycle IS the "closed circle":
```
4 -> 2 -> 1 -> 4   (the octave: back to where you started)
```
Just as well temperament allows returning to the starting key after modulating through all 12 keys, the +1 tempering allows returning to 1 after visiting all necessary residue classes.

---

## The j-1 = 3 Question

The team lead asks: for j=4, does j-1=3 match the "3 turns to close" in the orbifold?

**Direct answer**: The match at j=4 is suggestive but NOT structural. For j=5, the max shadow depth is 4, while the orbifold always closes in 3 turns. The numbers diverge.

**Deeper answer**: The "3 turns to close" in Tymoczko comes from the cyclic subgroup Z/3Z inside S_3 (the 120-degree rotation has order 3). In Collatz, the analogous "turns to close" is the order of 3 mod 2^j, which is 2^{j-2}. These grow with j.

But there IS a connection at a different level. The convergents of log_2(3/2) = 0.58496... are:
```
0/1, 1/1, 1/2, 3/5, 7/12, 24/41, 31/53, ...
```
The denominators 1, 2, 5, 12, 41, 53 are the preferred scale sizes in music (and predicted information-theoretic projection dimensions, per the reading notebooks). The 12 in "12-tone" comes from log_2(3/2) ~ 7/12.

For Collatz, the relevant convergent is the simplest one: log_2(3) ~ 2. This gives the contraction rate 2 - log_2(3) ~ 0.415. The "orbit" closes when the accumulated contraction equals the initial expansion, which takes roughly:
```
tau * log_2(3/2) / 0.415 ~ tau * 1.41 steps after the fold
```

The number 3 shows up not as "3 turns" but as:
- The multiplier m = 3 (forced by Nyquist: m = 2d-1)
- The S_3 symmetry destroyed by +1 (the degenerate twist)
- The denominator of the simplest convergent of log_2(3): 3/5 ~ 0.6 ~ log_2(3/2)

---

## The Unified Framework

```
Tymoczko          Collatz              N-Body              Music
---------         --------             --------            ------
T^3/S_3           Z_2 residues         Shape sphere S^2    Pitch space
120 twist         x3 (degenerate)      Geometric phase     Perfect fifth
Prism axis        log_2(n)             Time                Octave register
Cross-section     Residue class        Shape               Chord
Closure           4-2-1 cycle          Periodic orbit      Key return
Comma             +1 perturbation      Phase mismatch      Tempering
Tempering         +1 distribution      Bounded phase       Well temperament
Singularity       Extremal (2^k-1)     Central config      Augmented triad
```

The picture: **All four systems are fiber bundles where the base space has a twist with incommensurate period, and convergence/closure requires a tempering mechanism that distributes the accumulated error.**

- Music: The pitch circle (S^1) doesn't close under fifths. Well temperament distributes the comma.
- Collatz: The 2-adic integers don't close under x3+1. The +1 distributes the carry.
- N-body: The shape sphere accumulates geometric phase. The force law bounds the accumulation.
- Orbifold: The chord simplex doesn't close under transposition. The orbifold identification distributes the twist.

**The irrational number log_2(3) = 1.58496... is the engine of all four.**

---

## What This Means for the Proof

The Collatz proof IS a proof about orbifold geometry, but the orbifold is **degenerate** (the 3-adic fiber is collapsed to a point). The proof has two independent legs:

1. **The twist is well-tempered** (d=2 uniqueness): The +1 perturbation achieves 100% coverage because phi(2) = 1. This is the analog of well temperament being possible.

2. **The tempered spiral closes** (contraction dominance): The expected contraction log_2(3) - 2 < 0 ensures orbits shrink on average. This is the analog of the circle of fifths actually returning to the starting key under temperament.

Neither leg alone suffices:
- Leg 1 without Leg 2: 100% coverage but at the Nyquist boundary (marginal growth). Could oscillate forever.
- Leg 2 without Leg 1: Contraction on average but some inputs miss the division step. Could have divergent subsequences.

Together: every input gets tempered (Leg 1) and the tempering accumulates to net contraction (Leg 2). QED (modulo making this rigorous).

---

## New Experiment: E14 — Collatz Holonomy

Compute the "holonomy" of Collatz trajectories: for each starting value n, track the accumulated residue-class rotation mod 2^j through the full trajectory to 1. The holonomy = the total rotation accumulated.

Prediction: the holonomy should be related to the Pythagorean comma. Specifically, for a trajectory of length L with S odd steps:
```
holonomy mod 2^j = 3^S * n + (accumulated +1 terms) mod 2^j
```
If this holonomy has a universal distribution (like the Sato-Tate distribution for elliptic curves), that's the Collatz analog of the geometric phase distribution on the shape sphere.

This is implementable as a tambear accumulate:
```
accumulate(sequential, residue_class_mod_2j, compose_affine, collatz_trajectory)
```
where compose_affine tracks the accumulated (a, b) coefficients of the affine map.

---

*The spiral is the twist. The +1 is the temperament. The convergence is the closure. log_2(3) is the comma. The proof is geometry.*
