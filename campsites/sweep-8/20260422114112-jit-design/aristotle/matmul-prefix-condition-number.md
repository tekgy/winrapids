# MatMulPrefix condition number — κ_chain ≈ ||A||^n

Date: 2026-04-23
From: math-researcher
For: aristotle (sweep-8 jit-design)
Status: research deposit, feeds oracle's `extreme_conditioning_kappa_1e16` justification

---

## The finding (one sentence)

For a prefix product of n identical matrices A, `κ(A^n) ≤ κ(A)^n`;
the bound is **tight** for non-normal A, **loose** (sometimes by many
orders of magnitude) for normal / symmetric / unitary A. Oracle runs
that diverge at extreme conditioning are exhibiting this structural
growth, not an implementation bug.

## Setup

Let `A ∈ ℝ^{n×n}` (or ℂ^{n×n}; same analysis). The condition number
for a linear system is the 2-norm based

> κ₂(A) = ||A||₂ · ||A⁻¹||₂ = σ_max(A) / σ_min(A).

For a prefix product `P_k = A_1 · A_2 · ... · A_k`, the oracle cares
about how κ₂(P_k) evolves in k. Two useful regimes:

1. **Identical factors**: all A_i = A. `P_n = A^n`. Then

    κ₂(A^n) = σ_max(A^n) / σ_min(A^n).

2. **Varying factors with a common spectral-norm bound**: any
   upper-bound on κ₂(A^n) lifts to a bound on `∏κ₂(A_i)` via
   submultiplicativity.

## Derivation — general matrix

Using submultiplicativity of the operator 2-norm:

    ||AB||₂ ≤ ||A||₂ · ||B||₂.

For any invertible factors:

    κ₂(AB) = ||AB||₂ · ||(AB)⁻¹||₂
           = ||AB||₂ · ||B⁻¹ A⁻¹||₂
           ≤ (||A||₂ ||B||₂) · (||B⁻¹||₂ ||A⁻¹||₂)
           = κ₂(A) · κ₂(B).

Induction on n gives:

> **κ₂(∏_{i=1..n} A_i) ≤ ∏_{i=1..n} κ₂(A_i).**

For identical factors:

> **κ₂(A^n) ≤ κ₂(A)^n.**

This is the **Horn–Johnson** inequality (*Matrix Analysis*, 2nd ed.,
§5.6, 7.2 corollary on singular-value products). It is the fundamental
reason MatMulPrefix condition number can blow up exponentially in n.

## Tightness — when equality holds

The inequality is tight for **non-normal** A. Non-normality means
A·A* ≠ A*·A; intuitively, A has eigenvectors that are not orthogonal,
so repeated application can align vectors along the least-stable
direction and amplify there.

Classic tight example — **Jordan block** at a single eigenvalue λ
close to but not equal to 1:

    J = [ λ  1 ]
        [ 0  λ ]

    J^n = [ λ^n   n·λ^{n-1} ]
          [ 0     λ^n       ]

The (1,2) entry grows as n·λ^{n-1}, which for λ ≈ 1 is ≈ n. So
J^n has:
- σ_max(J^n) ≈ n (dominated by the nilpotent-scaled off-diagonal)
- σ_min(J^n) ≈ λ^n (small when λ < 1)
- κ₂(J^n) ≈ n / λ^n → κ₂(J)^n up to polynomial factor

For a Jordan block with eigenvalue exactly 1 and nilpotent part N
with N^k = 0 for k ≥ size, the product I + nN has κ₂ = 1 + poly(n)
growth — linear rather than exponential, but still strictly growing.

**Upshot**: for the adversarial oracle case, the "structured bad"
matrix is a near-defective near-identity — these exist, are smooth
to parameterize, and produce worst-case chain conditioning.

## Tighter bounds for structured A

| Class | Bound on κ₂(A^n) | Why |
|---|---|---|
| **Unitary** (A*A = I) | κ₂(A^n) = 1, for all n | σ_max(A) = σ_min(A) = 1 |
| **Orthogonal** (real unitary) | κ₂(A^n) = 1 | same |
| **Normal** (A*A = AA*) | κ₂(A^n) = κ₂(A)^n exactly (not bounded, but exact expression via eigenvalues) | simultaneous diagonalization of A and A* |
| **Symmetric positive definite (SPD)** | κ₂(A^n) = κ₂(A)^n exactly | SPD ⇒ normal with real positive eigenvalues |
| **Triangular (upper/lower)** | κ₂(A^n) ≤ κ₂(A)^n with typical tightness | eigenvalues on diagonal; near-Jordan for near-defective |
| **Permutation** | κ₂(A^n) = 1 | permutation is orthogonal |
| **Rotation** (2D) | κ₂(A^n) = 1 | rotations are orthogonal |
| **Diagonal** | κ₂(A^n) = κ₂(A)^n = (max|d_i|/min|d_i|)^n | eigenvalues are diagonal |
| **Non-normal general** | κ₂(A^n) ≤ κ₂(A)^n, potentially LARGER via pseudospectral gain | transient amplification above spectral bound |
| **Near-defective Jordan blocks** | **κ₂(J^n) ≈ κ₂(J)^n** | tightness at near-repeated eigenvalues |

**Pseudospectral caveat**: for non-normal A, the operator 2-norm of
A^n can transiently be larger than (spectral radius)^n even when
eigenvalues are inside the unit disk. This is the **Kreiss matrix
theorem** / Trefethen-Embree pseudospectra territory. The upper bound
κ₂(A)^n always holds, but in a specific finite-n window intermediate
products can overshoot the "asymptotic" spectral rate.

For oracle design: the asymptotic bound κ₂(A)^n is the right target;
intermediate-n transient amplification is a sub-bound inside the
same envelope.

## Implication for MatMulPrefix determinism oracle

Three design consequences:

### 1. Vacuous chain-wide static bound

A bound of `max_condition_number = 1/ε ≈ 4.5·10^{15}` for MatMulPrefix
LiftedTree is **vacuous for chains with κ(A) > 1**: ε·bound = 1.0 ≡ 100%
relative error, which any finite output trivially satisfies. The bound
must be either:
- per-element (scale single-step bound; caller multiplies chain-wise), or
- typed-domain-explicit (the enum says WHETHER the bound is per-element
  or chain-wide; DEC-022 sub-clause E applied to the tier coordinate).

(This is the same structural move GAP-AFFINE-COND-1 made for
AffineCompose; `MatMulPrefix::Sequential` already uses
`(1.0 / f64::EPSILON).powf(1.0 / k)` — the honest per-element
scale. `MatMulPrefix::LiftedTree` still has the flat vacuous form.)

### 2. Oracle divergence is expected, not a bug

At extreme conditioning κ ≈ 1/ε the oracle expects implementations
to diverge because κ_chain = κ(A)^n exceeds f64 precision almost
immediately. Two implementations using slightly different reduction
trees WILL produce bit-level-different outputs that are both within
the mathematical truth's uncertainty ball.

For `extreme_conditioning_kappa_1e16` variant:
- **Claim**: cross-implementation divergence at κ ≥ 1/ε is structural,
  not indicative of any implementation being "wrong."
- **Test discipline**: assert relative agreement bounded by
  `κ(A)^n · ε` (the theoretical limit), not bit-exactness.
- **When bit-exactness IS testable**: structured A (unitary,
  permutation, orthogonal, SPD, diagonal with bounded ratio) — the
  oracle can gate bit-exactness here because κ₂(A^n) is either 1 or
  computable.

### 3. Structured-A short-circuits

If caller annotates A as unitary / orthogonal / permutation / rotation
(via `using(matrix_structure = Orthogonal)` or similar), the
determinism_floor can IMMEDIATELY promote to BitExact regardless of
chain length — these matrices all have κ₂(A^n) = 1. This is a
rigorous short-cut that costs nothing.

Similarly, SPD annotation gives a usable `κ₂(A^n) = κ₂(A)^n` scaling
law that the oracle can evaluate before dispatch.

## Numerical example (sanity check)

Take n = 100 and A = [[0.9, 1.0], [0.0, 0.9]] (near-defective
near-contractive). Then:

    ||A||₂ ≈ 1.35, ||A⁻¹||₂ ≈ 1.35 · (1/0.9^2) ≈ 1.67
    κ₂(A)   ≈ 2.26
    κ₂(A)^100 ≈ 2.26^100 ≈ 10^{35.5}

That's ≈ 10^20 past f64's κ ≈ 10^16 limit. The prefix product of 100
steps of this benign-looking matrix puts us deep into bit-chaos land.
At that conditioning, any two implementations are free to diverge
by up to 10^{19} ULP in the answer and both be "correct within the
precision the math affords."

Contrast with A = [[cos θ, -sin θ], [sin θ, cos θ]] (rotation): A^n
is another rotation, κ₂(A^n) = 1 for all n. 100 compositions of
rotations stay precisely conditioned forever. Oracle should gate
bit-exactness here even for very long chains.

## References

- Horn, R.A. & Johnson, C.R. (2013). *Matrix Analysis* (2nd ed.),
  Cambridge. §5.6 (singular-value submultiplicativity), §7.2
  (eigenvalue inequalities).
- Trefethen, L.N. & Bau, D. (1997). *Numerical Linear Algebra*,
  SIAM. Lec. 12 (conditioning), Lec. 14 (stability of Gauss
  elimination — where "transient amplification" is introduced).
- Trefethen, L.N. & Embree, M. (2005). *Spectra and Pseudospectra*,
  Princeton. Ch. 14-16 (Kreiss matrix theorem; why
  non-normal matrices transiently exceed spectral-radius growth).
- Higham, N.J. (2002). *Accuracy and Stability of Numerical
  Algorithms* (2nd ed.), SIAM. §3.5 (norm and conditioning basics),
  §14 (matrix powers and iterations).
- Golub, G.H. & Van Loan, C.F. (2013). *Matrix Computations* (4th
  ed.), JHU Press. §2.3 (matrix norms), §2.6 (singular values).

## Queue / suggested next moves

- Decide between "constant-tighten MatMulPrefix::LiftedTree" (parallel
  to GAP-AFFINE-COND-1 fix) vs "add typed BoundDomain discriminant to
  DeterminismClass" (deeper sub-clause E move). The DECISION belongs
  with aristotle + pathmaker; the math above is only the input.
- If you want the structured-matrix annotation surface:
  `using(matrix_structure = Orthogonal | Unitary | Permutation | SPD
  | ...)`. Each gives a specific, small, provable κ_n function the
  oracle can use.

## Parallel filing

Also surfaced to navigator as part of post-compaction verification
sweep; preserved there under math-researcher's R10-15 campsite at
`R:\tambear\campsites\r10-15\20260423052729-math-references\math-researcher\insights\`.
