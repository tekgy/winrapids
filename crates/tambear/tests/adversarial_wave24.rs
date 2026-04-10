//! Adversarial Wave 24 — Singularity-as-silent-wrong-answer in matrix_sqrt (Denman-Beavers)
//!
//! Target: `matrix_sqrt` / `matrix_sqrt_denman` in `linear_algebra.rs`.
//!
//! The Denman-Beavers iteration requires invertible X and Y at each step.
//! When the iteration encounters a singular intermediate (which happens for
//! matrices with zero or negative eigenvalues), it calls `inv()` which returns
//! `None`, and the code `break`s out of the loop — returning whatever partially-
//! converged `x` happens to be at that iteration. No NaN, no None, no error flag.
//!
//! ## The bug pattern (linear_algebra.rs:1676-1678)
//!
//! ```rust
//! let x_inv = inv(&x);
//! let y_inv = inv(&y);
//! if x_inv.is_none() || y_inv.is_none() { break; }   // ← silent break
//! ```
//!
//! The `break` exits the iteration with `x` in an arbitrary mid-iteration state.
//! The caller receives a plausible-looking matrix that satisfies neither
//! `result² ≈ A` nor any other mathematical criterion. The contract of
//! `matrix_sqrt(A)` — return M such that M² = A — is violated without any signal.
//!
//! ## What `matrix_sqrt` should return for singular/indefinite matrices:
//!
//! Per Higham (2008) "Functions of Matrices", §6.1:
//! - The principal square root exists iff A has no eigenvalues on the closed
//!   negative real axis (i.e., no zero or negative real eigenvalues).
//! - For A with eigenvalue 0 (singular): principal square root does not exist
//!   in general for non-nilpotent A. Result should be NaN-matrix.
//! - For A with negative real eigenvalue: sqrt involves complex numbers.
//!   Result should be NaN-matrix (indicating real-sqrt undefined).
//!
//! ## Bugs confirmed:
//!
//! 1. `matrix_sqrt` of singular matrix (zero eigenvalue) — returns wrong non-NaN
//!    matrix rather than NaN-matrix. Contract violated silently.
//!
//! 2. `matrix_sqrt` of negative-semidefinite matrix (negative eigenvalue) —
//!    Denman-Beavers diverges; intermediate becomes singular; break returns
//!    a meaningless partially-computed matrix, not NaN.
//!
//! 3. `matrix_sqrt` of NaN-input matrix — NaN propagates through mat_mul but
//!    may produce NaN output that doesn't clearly signal the input was NaN
//!    (minor: this might accidentally be correct).
//!
//! These tests FAIL with the current implementation and PASS after the fix.

use tambear::linear_algebra::{Mat, matrix_sqrt, mat_mul};

/// Check that every entry of a matrix is NaN.
fn all_nan(m: &Mat) -> bool {
    m.data.iter().all(|v| v.is_nan())
}

/// Frobenius norm of (A - B).
fn frob_diff(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    a.data.iter().zip(b.data.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ─── Baseline: correct cases must still work ────────────────────────────────

/// Identity matrix: sqrt(I) = I.
#[test]
fn matrix_sqrt_identity_baseline() {
    let eye = Mat::eye(2);
    let s = matrix_sqrt(&eye);
    // s² should equal I
    let s2 = mat_mul(&s, &s);
    assert!(frob_diff(&s2, &eye) < 1e-10,
        "sqrt(I)² should equal I, frob_diff={}", frob_diff(&s2, &eye));
}

/// Diagonal SPD matrix: sqrt([[4,0],[0,9]]) = [[2,0],[0,3]].
#[test]
fn matrix_sqrt_diagonal_spd_baseline() {
    let a = Mat::from_vec(2, 2, vec![4.0, 0.0, 0.0, 9.0]);
    let s = matrix_sqrt(&a);
    let s2 = mat_mul(&s, &s);
    assert!(frob_diff(&s2, &a) < 1e-8,
        "sqrt([[4,0],[0,9]])² should equal [[4,0],[0,9]], frob_diff={}", frob_diff(&s2, &a));
}

// ─── Bug 1: singular matrix (eigenvalue = 0) ────────────────────────────────

/// `matrix_sqrt` of a singular matrix (rank-1 matrix, zero eigenvalue).
///
/// A = [[1, 1], [1, 1]] has eigenvalues {2, 0}. Principal square root does not
/// exist in the Higham sense (singular matrices don't have a unique square root).
/// The correct behavior is to return a NaN-matrix, NOT a plausible-looking matrix
/// whose square doesn't equal A.
///
/// BUG: Denman-Beavers hits a singular intermediate during iteration and `break`s,
/// returning the current iterate X. X² ≠ A — the contract is violated silently.
#[test]
fn matrix_sqrt_singular_matrix_returns_nan_not_silent_wrong() {
    // rank-1 symmetric PSD matrix with eigenvalues {2, 0}
    let a = Mat::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);

    let s = matrix_sqrt(&a);

    // The correct behavior: return NaN-matrix signaling failure.
    // The bug: return a plausible non-NaN matrix X where X² ≠ A.
    let s2 = mat_mul(&s, &s);
    let sq_error = frob_diff(&s2, &a);

    // If s is a true square root, s² == a (error < 1e-8).
    // The bug is that s² ≠ a but no NaN was returned — silent wrong answer.
    // We assert that either: (a) all entries are NaN (correct failure signal),
    // or (b) s is actually a correct square root (sq_error < 1e-8).
    // The bug manifests as: s is non-NaN AND sq_error is large.
    assert!(
        all_nan(&s) || sq_error < 1e-8,
        "matrix_sqrt(singular): returned non-NaN matrix X where X²≠A. \
         frob(X²-A)={sq_error:.6e}. Must return NaN-matrix or a correct sqrt."
    );
}

/// `matrix_sqrt` of the zero matrix.
///
/// sqrt(0) = 0 is mathematically valid. This is a boundary case — the zero
/// matrix has only zero eigenvalues, but sqrt(0) = 0 is the correct answer.
/// Denman-Beavers likely fails here (Y_inv singular from iteration 1).
/// Should either return zero-matrix or NaN-matrix — NOT a wrong matrix.
#[test]
fn matrix_sqrt_zero_matrix_correct_or_nan() {
    let zero = Mat::zeros(2, 2);
    let s = matrix_sqrt(&zero);

    // Two acceptable outcomes:
    // 1. All-zero (correct: 0² = 0)
    // 2. All-NaN (failure signal: couldn't compute)
    // Unacceptable: non-zero non-NaN matrix (wrong answer with no signal)
    let s2 = mat_mul(&s, &s);
    let sq_error = frob_diff(&s2, &zero);

    assert!(
        all_nan(&s) || sq_error < 1e-10,
        "matrix_sqrt(zero): returned non-NaN X where X²≠0. \
         frob(X²)={sq_error:.6e}. Must return zero-matrix or NaN-matrix."
    );
}

// ─── Bug 2: negative (semi)definite matrix (eigenvalue < 0) ─────────────────

/// `matrix_sqrt` of a matrix with a negative eigenvalue.
///
/// A = [[-1, 0], [0, 1]] has eigenvalues {-1, 1}. The real principal square
/// root of -1 does not exist. Denman-Beavers diverges — the iteration encounters
/// large norms, hits a singular intermediate, breaks, and returns garbage.
///
/// BUG: returns non-NaN garbage matrix. X² ≠ A. No error signal.
#[test]
fn matrix_sqrt_negative_eigenvalue_returns_nan_not_silent_wrong() {
    // Diagonal with one negative eigenvalue: sqrt undefined over reals.
    let a = Mat::from_vec(2, 2, vec![-1.0, 0.0, 0.0, 1.0]);

    let s = matrix_sqrt(&a);

    let s2 = mat_mul(&s, &s);
    let sq_error = frob_diff(&s2, &a);

    // Correct behavior: NaN-matrix (real sqrt undefined for negative eigenvalue).
    // Bug: non-NaN matrix X where X² ≠ A.
    assert!(
        all_nan(&s) || sq_error < 1e-8,
        "matrix_sqrt(neg eigenvalue): returned non-NaN X where X²≠A. \
         frob(X²-A)={sq_error:.6e}. Real square root is undefined — must return NaN-matrix."
    );
}

/// `matrix_sqrt` of a negative definite matrix.
///
/// A = [[-4, 0], [0, -9]] has eigenvalues {-4, -9}. Both negative.
/// Denman-Beavers starts from a negative-definite matrix and immediately
/// has trouble inverting ill-conditioned intermediates.
///
/// BUG: breaks on singular intermediate, returns partially-converged garbage.
#[test]
fn matrix_sqrt_negative_definite_returns_nan() {
    let a = Mat::from_vec(2, 2, vec![-4.0, 0.0, 0.0, -9.0]);

    let s = matrix_sqrt(&a);

    let s2 = mat_mul(&s, &s);
    let sq_error = frob_diff(&s2, &a);

    // Only correct outcome: NaN-matrix (no real square root exists).
    // Note: sq_error < 1e-8 would mean s² = A, but s would have to be complex.
    // Any real matrix squaring to a negative-definite matrix is impossible.
    // So we ONLY accept NaN output.
    assert!(
        all_nan(&s),
        "matrix_sqrt(negative definite): must return NaN-matrix (no real sqrt exists). \
         Got non-NaN output with frob(X²-A)={sq_error:.6e}."
    );
}

/// `matrix_sqrt` of a nearly-singular matrix: eigenvalues {ε, 1} as ε→0.
///
/// Near-singular: A = [[ε, 0], [0, 1]]. As ε→0 the sqrt approaches singular.
/// For ε=1e-16 (near machine epsilon), Y_inv becomes ill-conditioned.
/// The correct sqrt is [[√ε, 0], [0, 1]].
///
/// This is a precision test: does the implementation return a matrix close to
/// the true sqrt, or does it silently return garbage from early break?
#[test]
fn matrix_sqrt_near_singular_matrix_precision() {
    let eps = 1e-10_f64;  // small but representable
    let a = Mat::from_vec(2, 2, vec![eps, 0.0, 0.0, 1.0]);

    let s = matrix_sqrt(&a);

    // True sqrt: [[sqrt(eps), 0], [0, 1]]
    let true_sqrt = Mat::from_vec(2, 2, vec![eps.sqrt(), 0.0, 0.0, 1.0]);

    // Either NaN-matrix (conservative failure) or close to true sqrt
    let s2 = mat_mul(&s, &s);
    let sq_error = frob_diff(&s2, &a);

    assert!(
        all_nan(&s) || sq_error < 1e-8,
        "matrix_sqrt(near-singular, eps={eps:.0e}): X² differs from A by {sq_error:.6e}. \
         True sqrt gives s²=A exactly. Denman-Beavers may have broken early."
    );
    // Also verify the returned matrix is close to true sqrt (not just any solution to s²=A)
    if !all_nan(&s) {
        let direct_diff = frob_diff(&s, &true_sqrt);
        assert!(direct_diff < 1e-6,
            "matrix_sqrt(near-singular): s is numerically wrong. \
             frob(s - true_sqrt)={direct_diff:.6e}. Expected s[0,0]≈{:.6e}, got {:.6e}.",
            eps.sqrt(), s.get(0, 0));
    }
}

// ─── Bug 3: NaN-input matrix propagation ────────────────────────────────────

/// `matrix_sqrt` of a matrix with NaN entries.
///
/// If A contains NaN, the result must also contain NaN (propagate, don't hide).
/// This may already work correctly via arithmetic propagation, but documents
/// the expected behavior explicitly.
#[test]
fn matrix_sqrt_nan_input_propagates_nan() {
    let a = Mat::from_vec(2, 2, vec![1.0, f64::NAN, 0.0, 1.0]);
    let s = matrix_sqrt(&a);
    // Must produce NaN output — either all-NaN or at least some NaN.
    let has_nan = s.data.iter().any(|v| v.is_nan());
    assert!(has_nan,
        "matrix_sqrt(NaN input): output must contain NaN. Got {:?}", s.data);
}

// ─── The key diagnostic test ─────────────────────────────────────────────────

/// Verify the silent-wrong-answer mechanism directly.
///
/// For a singular matrix, the current code breaks mid-iteration and returns
/// the current iterate X. We can verify this is wrong by checking X² ≠ A.
/// This test specifically documents that the returned value is neither
/// correct (X² = A) nor properly flagged (all-NaN).
///
/// This is the WORST FAILURE MODE: confident wrong answer with no signal.
#[test]
fn matrix_sqrt_singular_silent_wrong_answer_mechanism() {
    // Rank-2 symmetric matrix with one zero eigenvalue.
    // eigenvalues: 5, 0 (via characteristic polynomial)
    let a = Mat::from_vec(2, 2, vec![3.0, 2.0, 2.0, 2.0]);
    // det = 3*2 - 2*2 = 6 - 4 = 2 ≠ 0. Hmm, not singular.
    // Let's use a = [[4, 2], [2, 1]] = outer product of [2,1] with itself
    // eigenvalues: 5, 0 ✓ (det = 4*1 - 2*2 = 0 ✓)
    let a = Mat::from_vec(2, 2, vec![4.0, 2.0, 2.0, 1.0]);
    // Verify det = 0: 4*1 - 2*2 = 4 - 4 = 0 ✓

    let s = matrix_sqrt(&a);

    if all_nan(&s) {
        // Correct: implementation detected singularity and signaled failure.
        return;
    }

    // If we get here, the implementation returned a non-NaN matrix.
    // Verify it's actually wrong (this is the bug).
    let s2 = mat_mul(&s, &s);
    let sq_error = frob_diff(&s2, &a);

    // The principal square root of a singular rank-1 matrix is itself rank-1.
    // Denman-Beavers breaks mid-iteration and returns an unconverged state.
    // That state is wrong: sq_error will be large.
    assert!(
        sq_error < 1e-8,
        "matrix_sqrt(singular rank-1): returned non-NaN X but X²≠A. \
         frob(X²-A)={sq_error:.6e}. This is the silent-wrong-answer bug: \
         the implementation returned a plausible-looking but incorrect matrix \
         with no error signal. Fix: return NaN-matrix when inv() fails."
    );
}
