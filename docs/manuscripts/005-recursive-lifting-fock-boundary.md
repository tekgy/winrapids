# Recursive Lifting and the Fock Boundary of Parallelizable Computation

**Draft — 2026-03-30**

---

## Abstract

The Fock boundary — the limit of parallelizable computation where self-reference prevents decomposition — is not fixed. It can be pushed outward by redefining the fundamental computational entity at progressively higher orders, a technique we call *recursive lifting*. At order 1, each element is processed independently. At order 2 (the "biphoton" level), pairs of elements are processed jointly, capturing pairwise interactions that are non-parallelizable at order 1. At order k, k-tuples capture k-body interactions. We formalize this hierarchy, show it applies to both GPU scan operators and compiler state optimization, and connect it to the k-particle hierarchy in quantum field theory (where the technique was first employed for rendering). The Fock boundary shifts to the point where the entity order k must be variable — where the number of interacting components is itself determined by the computation's intermediate state.

---

## 1. The Fock Boundary

### 1.1 Definition

A computation C over a sequence s₁, ..., sₙ is *decomposable at order 1* if:

C(s₁,...,sₙ) = extract(⊕ᵢ lift(sᵢ))

for some associative ⊕, lift function, and extraction. This is the standard definition of parallelizability via prefix scan or additive accumulation.

A computation is at the *Fock boundary* when it cannot be expressed in this form because the computation of element i depends on the COLLECTIVE state of all elements — not just element i's own attributes.

### 1.2 Self-Reference as the Mechanism

The Fock boundary arises from self-reference: the computation needs to know its own result before it can compute its result. Examples:

- **Rendering**: the number of photons emitted depends on the current light field (stimulated emission). The computation needs the output to determine the input.
- **Nonlinear filtering**: the state transition F(x) depends on the current estimate x. The filter needs its own output to determine its dynamics.
- **State-dependent branching**: `if result > threshold: do_X() else: do_Y()`. The computation path depends on intermediate results.

In all cases: the combine function's meaning depends on the accumulated state, violating the context-independence requirement of associativity.

---

## 2. Recursive Lifting

### 2.1 The Biphoton Trick

Some computations that are non-decomposable at order 1 become decomposable at order 2 by redefining the fundamental entity.

**Example (Rendering)**: Bell correlations ⟨σ_A · σ_B⟩ between two photons are non-liftable at order 1 (they require comparing two particles). But if we define a *biphoton* — a single entity in ℝ⁶ carrying both particles' positions — the correlation becomes a per-entity function. Liftable at order 2.

**Example (Computation)**: Optimizing provenance and persistence independently misses their interaction (a provenance hit on an evicted buffer has different cost than a hit on a resident buffer). But if we define a *joint state* (provenance_status, residency_status) as a single entity, the interaction is captured per-entity. The optimizer sees the full picture at order 2.

### 2.2 The General Hierarchy

**Definition.** A computation C is *decomposable at order k* if:

C(s₁,...,sₙ) = extract(⊕_{T ∈ S^k} lift_k(T))

where T ranges over k-tuples of input elements, lift_k maps k-tuples to a monoid M, and ⊕ is associative over M.

The hierarchy:
- **Order 1**: each element processed independently. Standard parallelism.
- **Order 2**: pairs processed jointly ("biphoton"). Captures pairwise interactions.
- **Order k**: k-tuples processed jointly. Captures k-body interactions.
- **Variable order (Fock)**: the tuple size k depends on intermediate computation. Non-parallelizable at any fixed order.

### 2.3 Each Order Pushes the Boundary

At each order, interactions that were "non-parallelizable externalities" become "per-entity internalities":

| Order | What's captured | What's still external |
|---|---|---|
| 1 | Per-element computation | All inter-element interactions |
| 2 | Pairwise interactions | Triple and higher interactions |
| 3 | Triple interactions | Quadruple and higher |
| k | k-body interactions | (k+1)-body and higher |
| ∞ | All fixed-order interactions | Variable-order (Fock) |

---

## 3. Applications

### 3.1 Operator Families and the Kalman Hierarchy

The Kalman filter family demonstrates recursive lifting:

- **Order 1**: KalmanAffineOp — steady-state filter. Each observation processed with fixed gain. Fully decomposable. O(log n) via affine scan.

- **Order 2 (transient)**: KalmanSärkkäOp — the 5-tuple carries coupled (state estimate, covariance) as a joint entity. The P-x interaction (covariance affects the gain affects the estimate) is captured within the entity. Fully decomposable at order 2 via the Särkkä associative operator.

- **Near-Fock (nonlinear)**: EKF — the state transition depends on the current estimate. Linearize to create an approximate order-2 entity. The approximation quality depends on how much the nonlinearity matters.

- **Fock (variable structure)**: Particle filter with variable particle count, or multiple-model adaptive estimation where the model set changes based on evidence. The entity order is not fixed — it depends on the computation's progress.

### 3.2 Compiler State Optimization

The compiler's optimization levels demonstrate recursive lifting:

**Order 1 (funnel)**: Optimize provenance, persistence, and fusion independently. Each level makes locally optimal decisions.

**Order 2 (joint pairs)**: Optimize (provenance × persistence) jointly — a provenance hit on a GPU-resident buffer is handled differently from a hit on an evicted buffer. Optimize (persistence × fusion) jointly — what's resident determines where fusion boundaries should be.

**Order 3 (full triple)**: Optimize (provenance × persistence × fusion) as a single entity. The compiler's execution plan considers all three simultaneously. One probe returns the full answer: cached + location + optimal execution strategy.

**Cross-query (temporal order 2)**: The plan itself has provenance. If the pipeline hasn't changed and the store state hasn't changed, reuse the compiled plan. This is the compiler caching its own work — a temporal biphoton of (current_query, previous_query).

### 3.3 Musical Analogy: The Whitacre Chord

The hierarchy has a natural musical interpretation:

- **Order 1 (monophonic)**: each voice sings alone. No relationships. No timbre.
- **Order 2 (diad)**: two voices create beating, consonance, or dissonance. The pairwise interaction IS the sound quality.
- **Order 3 (triad)**: three voices create chord quality (major/minor/diminished). Emergent property not present in any pair.
- **Order N (cluster)**: Whitacre's 18-voice cluster chords. The emergent texture is an N-body interaction that no subset captures.

The timbre of a computation — its performance fingerprint, its sharing profile, its optimization quality — emerges from the order at which the compiler operates on the joint state.

---

## 4. The True Fock Boundary

### 4.1 Variable Entity Order

The Fock boundary is not "high order k." It is VARIABLE order — where k itself is determined by the computation.

In quantum field theory: Fock space = ⊕_{N=0}^∞ H^⊗N. The photon number N is a quantum observable. States are superpositions across different N. No fixed-N entity can capture them.

In computation: when the pipeline structure changes based on intermediate results (data-dependent control flow), the "entity" that would need to be pre-defined changes shape mid-computation. No fixed k-tuple suffices.

### 4.2 Partial Lifts at the Boundary

Even Fock non-parallelizable computations admit useful approximations:

- **Fixed-k approximation**: choose k large enough to capture the dominant interactions, accept residual error from (k+1)-body effects. Like truncating a Taylor series.

- **Linearization**: approximate the Fock-level interaction as a perturbation of a parallelizable base case. The EKF strategy: pretend the nonlinearity is locally linear, scan the linearized system.

- **Sampling**: Monte Carlo over the variable-k space. Each sample has fixed k (parallelizable). The ensemble average approximates the Fock-level computation. Particle filters use this strategy.

### 4.3 The Self-Awareness Interpretation

The Fock boundary is the point where the computation would need more self-awareness than can be accommodated in a fixed-dimensional per-element attribute.

Below the boundary: the element carries everything it needs. The photon carries its wavelength. The scan state carries its accumulated history. The compiler state carries its provenance + residency + plan.

At the boundary: the element would need to carry information about what OTHER elements will do, which depends on what IT does. This circular dependency — self-reference — is structurally irreducible. It cannot be broken by adding more dimensions to the entity. It can only be approximated.

This connects to fundamental limits in computation theory: the halting problem (can a program predict its own termination?), Gödel's incompleteness (can a system prove its own consistency?), and the observer effect in quantum mechanics (can a measurement not disturb what it measures?). All are instances of the self-reference barrier.

---

## 5. Design Implications

### 5.1 For GPU System Designers

1. **Default to order 1.** Most computations are decomposable at order 1. The affine scan handles all linear recurrences.

2. **Lift to order 2 when order 1 misses interactions.** Joint (provenance × residency) optimization, coupled state estimation (Särkkä), pairwise correlation computation.

3. **The architecture should support arbitrary lifting.** The data structure (buffer headers, provenance entries, store lookups) should accommodate richer state without architectural change. A richer key → richer value → same HashMap.

4. **Don't fight the Fock boundary.** For truly variable-structure computation (data-dependent branching, adaptive model selection), accept sequential execution with maximal sharing of the fixed-structure sub-computations.

### 5.2 For Theorists

The decomposable accumulation framework (manuscript 003) provides the formal basis. Recursive lifting is the process of moving computations from the Fock class to the decomposable class by enriching the entity definition. The interesting questions:

- Is there an algorithm for determining the minimum order k at which a given computation becomes decomposable?
- Can the quality of an order-k approximation to a Fock computation be bounded in terms of the (k+1)-body interaction strength?
- Is there a meaningful notion of "Fock dimension" that measures how far a computation is from decomposability?

---

## 6. Conclusion

The Fock boundary is not a wall. It is a horizon that recedes as we lift the entity definition to higher orders. Each order captures more of the interaction structure. The boundary remains — variable-order interactions are structurally irreducible — but the parallelizable domain grows with each lifting.

The practical takeaway: when a computation appears non-parallelizable, ask not "is this sequential?" but "at what order does it become parallel?" The answer is often surprisingly low. Bell correlations: order 2. Transient Kalman: order 2 (Särkkä 5-tuple). Compiler state optimization: order 3. Most of what we call "inherently sequential" is parallelizable at a modestly higher order.

The timbre of a system — its emergent performance quality — is determined by the order at which its optimizer operates. Higher order = more interactions captured = richer sharing = better performance. The journey from order 1 to order k is the journey from a solo instrument to a Whitacre choir.

---

## References

- Bird, R. S. (1987). An introduction to the theory of lists.
- Blelloch, G. E. (1990). Prefix sums and their applications.
- Gu, A. & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces.
- Särkkä, S. & García-Fernández, Á. F. (2021). Temporal parallelization of Bayesian smoothers.
- Weinberg, S. (1995). The Quantum Theory of Fields, Vol 1. (Fock space formalism.)
