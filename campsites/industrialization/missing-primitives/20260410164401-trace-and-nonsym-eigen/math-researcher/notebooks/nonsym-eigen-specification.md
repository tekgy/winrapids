<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# Non-Symmetric Eigendecomposition — Implementation Specification

## What's Needed

Given a real n×n matrix A (not necessarily symmetric), compute all eigenvalues
and optionally all eigenvectors. Eigenvalues may be complex (conjugate pairs).

## Algorithm: Francis QR with Implicit Double Shift

**Reference**: Francis (1961, 1962). Wilkinson (1968) for shift strategy.
Golub & Van Loan (2013), "Matrix Computations", Ch. 7.

### Step 1: Hessenberg Reduction — O(n³)

Reduce A to upper Hessenberg form H via Householder reflections:
A = Q H Q^T where H has zeros below the first subdiagonal.

```
For k = 1, ..., n-2:
  v = householder_vector(H[k+1:n, k])
  H[k+1:n, k:n] -= 2 v (v^T H[k+1:n, k:n])    // left multiply
  H[1:n, k+1:n] -= 2 (H[1:n, k+1:n] v) v^T     // right multiply
  Q[1:n, k+1:n] -= 2 (Q[1:n, k+1:n] v) v^T     // accumulate Q
```

The Hessenberg form preserves eigenvalues and has O(n²) nonzeros instead of O(n³).

### Step 2: Implicit QR Iteration with Double Shift — O(n² per sweep)

The Francis double shift handles complex conjugate eigenvalue pairs without
complex arithmetic. Each sweep chases a 3×1 bulge down the Hessenberg matrix.

```
Repeat until convergence:
  // Wilkinson double shift: eigenvalues of bottom-right 2×2 block
  s = H[n-1,n-1] + H[n,n]         // trace of 2×2
  t = H[n-1,n-1]*H[n,n] - H[n-1,n]*H[n,n-1]  // det of 2×2
  
  // First column of (H - s1*I)(H - s2*I) = H² - sH + tI
  x = H[1,1]² + H[1,2]*H[2,1] - s*H[1,1] + t
  y = H[2,1]*(H[1,1] + H[2,2] - s)
  z = H[2,1]*H[3,2]
  
  // Chase bulge from (1,1) to (n,n)
  For k = 1, ..., n-2:
    v = householder_vector([x, y, z])
    Apply 3×3 Householder from left and right
    Update x, y, z from next column
  
  // Check for deflation: if H[k+1,k] ≈ 0, eigenvalue has converged
  if |H[k+1,k]| < eps * (|H[k,k]| + |H[k+1,k+1]|):
    H[k+1,k] = 0  // deflate
    Recurse on smaller submatrices
```

### Step 3: Extract Eigenvalues

After QR iteration converges, H is quasi-upper-triangular (real Schur form):
- 1×1 diagonal blocks → real eigenvalues
- 2×2 diagonal blocks → complex conjugate pairs (a ± bi)

For a 2×2 block [[a, b], [c, d]]:
- eigenvalues = (a+d)/2 ± sqrt((a-d)²/4 + bc)
- if discriminant < 0: complex pair

### Step 4: Eigenvectors (Optional)

Back-substitute through the quasi-triangular matrix to get eigenvectors.
For complex eigenvalue pairs, produce two real vectors (real and imaginary parts).

## API Design

```rust
pub struct EigenResult {
    /// Eigenvalues as (real, imaginary) pairs, sorted by magnitude descending.
    pub eigenvalues: Vec<(f64, f64)>,
    /// Right eigenvectors as columns of an n×n matrix.
    /// For complex eigenvalue pair at indices j, j+1:
    ///   eigenvector_j = column_j + i*column_{j+1}
    ///   eigenvector_{j+1} = column_j - i*column_{j+1}
    pub eigenvectors: Option<Mat>,
}

/// Eigendecomposition of a general (non-symmetric) real matrix.
///
/// Returns all eigenvalues (possibly complex) and optionally eigenvectors.
/// Complex eigenvalues always appear in conjugate pairs (a+bi, a-bi).
///
/// Uses Francis QR algorithm with implicit double shift (Golub & Van Loan Ch. 7).
/// O(n³) for eigenvalues only, O(n³) + O(n³) for eigenvectors.
///
/// ## When to use
/// - `sym_eigen`: for symmetric/Hermitian matrices (real eigenvalues, orthogonal eigenvectors)
/// - `general_eigen`: for non-symmetric matrices (complex eigenvalues possible)
///
/// ## Failure modes
/// - Defective matrices: algebraic multiplicity > geometric multiplicity.
///   The eigenvector matrix is not invertible. Detected by: near-zero norm of
///   an eigenvector during back-substitution.
/// - Non-convergence: QR iteration may not converge for pathological matrices.
///   Returns partial results with a convergence flag.
pub fn general_eigen(a: &Mat, compute_vectors: bool) -> EigenResult
```

## Primitives This Decomposes Into

```
householder_vector(x) → v       [ALREADY USED internally in QR]
householder_apply_left(H, v)    [ALREADY USED internally in QR]  
householder_apply_right(H, v)   [ALREADY USED internally in QR]
wilkinson_shift(H)              [NEW: 2×2 eigenvalue computation]
bulge_chase(H, shift)           [NEW: Francis double shift step]
schur_to_eigenvalues(T)         [NEW: extract from quasi-triangular]
back_substitute_schur(T, Q)     [NEW: eigenvectors from Schur form]
```

The Householder operations already exist inside `qr()`. They should be exposed as
public primitives (the extraction audit already flagged this).

## trace() — Trivial but Missing

```rust
/// Matrix trace: tr(A) = Σ a_ii.
///
/// Properties: tr(AB) = tr(BA), tr(A) = Σ eigenvalues(A),
/// tr(A^T) = tr(A), tr(αA) = α tr(A).
pub fn trace(a: &Mat) -> f64 {
    let n = a.rows.min(a.cols);
    (0..n).map(|i| a.get(i, i)).sum()
}
```

One line. Should have existed from the start.

## What general_eigen Unlocks

1. **matrix_exp for non-symmetric A**: exp(A) = V exp(Λ) V⁻¹
   (current Padé approach works but eigendecomposition is an alternative + validation)
2. **Stability analysis**: eigenvalues of Jacobian → stable/unstable/center manifold
3. **Brusselator/Gray-Scott**: Jacobian eigenvalues → Hopf bifurcation detection
4. **Dynamical systems**: Floquet multipliers, Lyapunov exponents from linearization
5. **Non-normal matrices**: pseudospectra (eigenvalue sensitivity)
6. **Schur decomposition**: comes for free (it's an intermediate step)
7. **Matrix sign function**: sign(A) = V sign(Λ) V⁻¹

## Sharing via TamSession

Tag: `general_eigen(matrix_hash)`
Fields: eigenvalues, eigenvectors (optional), Schur form (optional)

Consumers: matrix functions, stability analysis, spectral methods

Compatibility: NOT compatible with `sym_eigen` results (different algorithm,
different guarantees). The eigenvalues may match but the eigenvector matrices
have different properties (orthogonal for symmetric, generally invertible for non-symmetric).

## Testing Strategy

1. **Symmetric matrices**: result should match `sym_eigen` (eigenvalues identical,
   eigenvectors may differ in sign/ordering)
2. **Known spectrum**: companion matrix of polynomial with known roots
3. **Complex pairs**: rotation matrix [[cos θ, -sin θ], [sin θ, cos θ]] → eigenvalues e^{±iθ}
4. **Defective**: [[1,1],[0,1]] → eigenvalue 1 with multiplicity 2 but only 1 eigenvector
5. **Large random**: verify A = V Λ V⁻¹ reconstruction
6. **Adversarial**: Wilkinson matrix (known near-clustering eigenvalues)


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

