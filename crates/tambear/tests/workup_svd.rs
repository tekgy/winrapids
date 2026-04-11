//! Publication-grade workup for `linear_algebra::svd`.
//!
//! SVD is the foundation for `pinv`, `lstsq`, `rank`, and `cond`. The pinv
//! workup treats SVD as a trusted black box. This workup tests SVD directly
//! against analytical oracles, theorem properties, and adversarial inputs.
//!
//! # Coverage
//!
//! 1. Analytical oracles — matrices with known exact singular values
//! 2. Theorem tests — U orthogonality, V^T orthogonality, Σ positivity, ordering
//! 3. Reconstruction theorem — A = U Σ V^T at multiple scales
//! 4. Ill-conditioned matrices — Hilbert matrices, known condition numbers
//! 5. Near-rank-deficient — matrices where σ_k ≈ σ_{k+1} ≈ 0
//! 6. Adversarial edge cases — rank-0, rank-1, wide/tall, square
//!
//! # How analytical oracles are derived
//!
//! For a 2×2 symmetric matrix [[a,b],[b,d]], eigenvalues are:
//!   λ = ((a+d) ± sqrt((a-d)² + 4b²)) / 2
//! and since eigenvalues of A^T A for a symmetric PSD matrix equal squares of SVs,
//! the SVs are the square roots of these eigenvalues (for a PSD matrix, SVs = eigenvalues).
//!
//! For A^T A = [[a²+b², ab+bd],[ab+bd, b²+d²]] the exact SVs can be computed
//! analytically. All oracles below are derived from this formula or simpler cases.
//!
//! Run: `CARGO_TARGET_DIR=target3 cargo test --test workup_svd -- --nocapture`

use tambear::linear_algebra::{svd, Mat, mat_mul};

// ─── Helpers ────────────────────────────────────────────────────────────────

fn mat(rows: usize, cols: usize, data: &[f64]) -> Mat {
    Mat::from_vec(rows, cols, data.to_vec())
}

fn mat2(data: &[f64]) -> Mat {
    mat(2, 2, data)
}

fn max_abs_err(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    a.data.iter().zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs(a: &Mat) -> f64 {
    a.data.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

fn assert_close(got: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (got - expected).abs() <= tol,
        "{label}: got {got:.6e}, expected {expected:.6e}, tol {tol:.1e}"
    );
}

/// Check that M ≈ identity (for orthogonality tests)
fn assert_is_identity(m: &Mat, tol: f64, label: &str) {
    let eye = Mat::eye(m.rows);
    let err = max_abs_err(m, &eye);
    assert!(err <= tol, "{label}: max_err={err:.2e}, expected <= {tol:.1e}");
}

// ─── Section 1: Analytical oracles ──────────────────────────────────────────

/// Oracle 1: Identity matrix.
/// SVD(I_n) = I · diag(1,...,1) · I. All singular values = 1.0.
#[test]
fn svd_oracle_identity_3x3() {
    let a = Mat::eye(3);
    let r = svd(&a);
    assert_eq!(r.sigma.len(), 3);
    for (i, &s) in r.sigma.iter().enumerate() {
        assert_close(s, 1.0, 1e-14, &format!("I₃ σ[{i}]"));
    }
}

/// Oracle 2: Diagonal matrix with known entries.
/// SVD(diag(5, 3, 1)) = I · diag(5,3,1) · I. SVs = |diagonal entries|, sorted.
#[test]
fn svd_oracle_diagonal_3x3() {
    let a = mat(3, 3, &[5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
    let r = svd(&a);
    assert_eq!(r.sigma.len(), 3);
    assert_close(r.sigma[0], 5.0, 1e-14, "diag(5,3,1) σ[0]");
    assert_close(r.sigma[1], 3.0, 1e-14, "diag(5,3,1) σ[1]");
    assert_close(r.sigma[2], 1.0, 1e-14, "diag(5,3,1) σ[2]");
}

/// Oracle 3: Permuted diagonal (entries NOT in descending order).
/// SVD should sort singular values descending.
#[test]
fn svd_oracle_diagonal_unsorted_entries() {
    let a = mat(3, 3, &[1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0]);
    let r = svd(&a);
    // Sorted descending: 5, 3, 1
    assert_close(r.sigma[0], 5.0, 1e-14, "unsorted diag σ[0]");
    assert_close(r.sigma[1], 3.0, 1e-14, "unsorted diag σ[1]");
    assert_close(r.sigma[2], 1.0, 1e-14, "unsorted diag σ[2]");
}

/// Oracle 4: Negative entries — SVD takes absolute values.
/// diag(-4, 2): SVs = 4, 2 (not -4, 2).
#[test]
fn svd_oracle_negative_diagonal() {
    let a = mat2(&[-4.0, 0.0, 0.0, 2.0]);
    let r = svd(&a);
    assert_close(r.sigma[0], 4.0, 1e-14, "diag(-4,2) σ[0]");
    assert_close(r.sigma[1], 2.0, 1e-14, "diag(-4,2) σ[1]");
}

/// Oracle 5: 2×2 rotation matrix.
/// R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]] is orthogonal, so all SVs = 1.
#[test]
fn svd_oracle_rotation_matrix() {
    let theta = std::f64::consts::PI / 7.0;
    let c = theta.cos();
    let s = theta.sin();
    let a = mat2(&[c, -s, s, c]);
    let r = svd(&a);
    assert_close(r.sigma[0], 1.0, 1e-13, "rotation σ[0]");
    assert_close(r.sigma[1], 1.0, 1e-13, "rotation σ[1]");
}

/// Oracle 6: Rank-1 matrix uv^T.
/// For u = [3, 4]^T and v = [1, 0]^T: uv^T has exactly one nonzero SV = ||u||·||v|| = 5.
#[test]
fn svd_oracle_rank1_matrix() {
    // u = [3, 4], v = [1, 0] → uv^T = [[3,0],[4,0]]
    let a = mat2(&[3.0, 0.0, 4.0, 0.0]);
    let r = svd(&a);
    assert_close(r.sigma[0], 5.0, 1e-13, "rank-1 σ[0] = ||u||·||v||");
    // Second SV should be zero (rank-1 matrix has only one nonzero SV)
    assert!(r.sigma[1].abs() < 1e-13, "rank-1 σ[1] = {:.2e}, expected ≈ 0", r.sigma[1]);
}

/// Oracle 7: 2×2 general matrix — analytical singular values.
///
/// For A = [[1, 2], [3, 4]]:
///   A^T A = [[10, 14], [14, 20]]
///   eigenvalues of A^T A: λ = (30 ± sqrt(900 - 200 + 4*196))/2
///                             = (30 ± sqrt(700+784))/2 = 15 ± sqrt(221)/2 ... let me recompute
///
/// Actually: A^T A = [[1*1+3*3, 1*2+3*4], [2*1+4*3, 2*2+4*4]] = [[10,14],[14,20]]
/// Eigenvalues: trace=30, det=200-196=4. λ = (30 ± sqrt(900-16))/2 = 15 ± sqrt(221)/2 ...
/// Wait: det(A^T A - λI) = (10-λ)(20-λ) - 196 = λ² - 30λ + 200 - 196 = λ² - 30λ + 4
/// λ = (30 ± sqrt(900-16))/2 = (30 ± sqrt(884))/2 = 15 ± sqrt(221)
/// σ₁ = sqrt(15 + sqrt(221)), σ₂ = sqrt(15 - sqrt(221))
/// sqrt(221) ≈ 14.866...
/// σ₁ ≈ sqrt(29.866) ≈ 5.465...
/// σ₂ ≈ sqrt(0.134) ≈ 0.366...
///
/// Reference: NumPy np.linalg.svd([[1,2],[3,4]]) gives [5.46498..., 0.36596...]
#[test]
fn svd_oracle_2x2_general_analytical() {
    let a = mat2(&[1.0, 2.0, 3.0, 4.0]);
    let r = svd(&a);

    // σ₁ = sqrt(15 + sqrt(221))
    let sigma1 = (15.0 + 221.0_f64.sqrt()).sqrt();
    // σ₂ = sqrt(15 - sqrt(221))
    let sigma2 = (15.0 - 221.0_f64.sqrt()).sqrt();

    assert_close(r.sigma[0], sigma1, 1e-12, "[[1,2],[3,4]] σ₁");
    assert_close(r.sigma[1], sigma2, 1e-12, "[[1,2],[3,4]] σ₂");
}

/// Oracle 8: Hadamard-2 matrix [[1,1],[1,-1]] (scaled by 1/sqrt(2) would be orthogonal).
/// SVs of [[1,1],[1,-1]]: A^T A = [[2,0],[0,2]] → both SVs = sqrt(2).
#[test]
fn svd_oracle_hadamard_2x2() {
    let a = mat2(&[1.0, 1.0, 1.0, -1.0]);
    let r = svd(&a);
    let s2 = 2.0_f64.sqrt();
    assert_close(r.sigma[0], s2, 1e-13, "Hadamard-2 σ[0]");
    assert_close(r.sigma[1], s2, 1e-13, "Hadamard-2 σ[1]");
}

/// Oracle 9: Tall matrix (3×2) — min(m,n) = 2 singular values.
/// A = [[1,0],[0,1],[0,0]]: A^T A = I₂ → both SVs = 1.
#[test]
fn svd_oracle_tall_matrix_3x2_with_zero_row() {
    let a = mat(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let r = svd(&a);
    assert_eq!(r.sigma.len(), 2, "3×2 tall: should have 2 singular values");
    assert_close(r.sigma[0], 1.0, 1e-13, "tall 3x2 σ[0]");
    assert_close(r.sigma[1], 1.0, 1e-13, "tall 3x2 σ[1]");
}

/// Oracle 10: Wide matrix (2×3) — min(m,n) = 2 singular values.
/// A = [[1,0,0],[0,1,0]]: A A^T = I₂ → both SVs = 1.
#[test]
fn svd_oracle_wide_matrix_2x3_with_zero_col() {
    let a = mat(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let r = svd(&a);
    assert_eq!(r.sigma.len(), 2, "2×3 wide: should have 2 singular values");
    assert_close(r.sigma[0], 1.0, 1e-13, "wide 2x3 σ[0]");
    assert_close(r.sigma[1], 1.0, 1e-13, "wide 2x3 σ[1]");
}

// ─── Section 2: Theorem properties ──────────────────────────────────────────

/// Theorem 1: U is orthogonal — U^T U = I_m, U U^T = I_m.
#[test]
fn svd_theorem_u_is_orthogonal_3x2() {
    let a = mat(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = svd(&a);
    let utu = mat_mul(&r.u.t(), &r.u);
    let uut = mat_mul(&r.u, &r.u.t());
    assert_is_identity(&utu, 1e-12, "U^T U (3×2 tall)");
    assert_is_identity(&uut, 1e-12, "U U^T (3×2 tall)");
}

/// Theorem 2: V^T is orthogonal — V^T (V^T)^T = I_n, i.e., V^T V = I_n.
#[test]
fn svd_theorem_vt_is_orthogonal_3x2() {
    let a = mat(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = svd(&a);
    let vtv = mat_mul(&r.vt, &r.vt.t());
    let vtvt = mat_mul(&r.vt.t(), &r.vt);
    assert_is_identity(&vtv, 1e-12, "V^T V (= I_n)");
    assert_is_identity(&vtvt, 1e-12, "(V^T)^T V^T (= I_n)");
}

/// Theorem 3: U is orthogonal for a square 4×4 full-rank matrix.
/// Using a matrix with guaranteed full rank (Lehmer-style diagonally dominant).
#[test]
fn svd_theorem_u_orthogonal_4x4_square() {
    let a = mat(4, 4, &[
        4.0, 1.0, 0.5, 0.25,
        1.0, 3.0, 1.0, 0.5,
        0.5, 1.0, 4.0, 1.0,
        0.25, 0.5, 1.0, 3.0,
    ]);
    let r = svd(&a);
    let utu = mat_mul(&r.u.t(), &r.u);
    let uut = mat_mul(&r.u, &r.u.t());
    assert_is_identity(&utu, 1e-10, "U^T U (4×4 full-rank)");
    assert_is_identity(&uut, 1e-10, "U U^T (4×4 full-rank)");
}

/// Theorem 4: Singular values are non-negative.
#[test]
fn svd_theorem_sigma_nonneg_general_3x4() {
    let a = mat(3, 4, &[
        1.0, -2.0, 3.0, -4.0,
        5.0, -6.0, 7.0, -8.0,
        9.0, -10.0, 11.0, -12.0,
    ]);
    let r = svd(&a);
    for (i, &s) in r.sigma.iter().enumerate() {
        assert!(s >= 0.0, "σ[{i}] = {s:.4e} should be ≥ 0");
    }
}

/// Theorem 5: Singular values are in non-increasing order.
#[test]
fn svd_theorem_sigma_sorted_descending() {
    let a = mat(4, 3, &[
        4.0, 3.0, 1.0,
        2.0, 5.0, 6.0,
        1.0, 2.0, 3.0,
        7.0, 0.0, 1.0,
    ]);
    let r = svd(&a);
    for i in 1..r.sigma.len() {
        assert!(
            r.sigma[i - 1] >= r.sigma[i],
            "σ not sorted: σ[{}]={:.4e} < σ[{}]={:.4e}",
            i - 1, r.sigma[i - 1], i, r.sigma[i]
        );
    }
}

/// Theorem 6: Frobenius norm preserved — ||A||_F² = sum(σᵢ²).
/// The Frobenius norm is the L2 norm of the singular value vector.
#[test]
fn svd_theorem_frobenius_norm_equals_sv_norm() {
    let a = mat(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let r = svd(&a);
    let frob_sq: f64 = a.data.iter().map(|&x| x * x).sum();
    let sv_sq: f64 = r.sigma.iter().map(|&s| s * s).sum();
    assert!(
        (frob_sq - sv_sq).abs() < 1e-10,
        "||A||_F² = {frob_sq:.6}, sum(σᵢ²) = {sv_sq:.6}"
    );
}

/// Theorem 7: Spectral norm (σ_max) = max singular value.
/// For A = 5·I₂, σ_max = 5.
#[test]
fn svd_theorem_spectral_norm() {
    let a = mat2(&[5.0, 0.0, 0.0, 5.0]);
    let r = svd(&a);
    assert_close(r.sigma[0], 5.0, 1e-14, "spectral norm = σ_max");
}

/// Theorem 8: Nuclear norm (trace norm) = sum(σᵢ).
/// For diag(3, 1): nuclear norm = 4.
#[test]
fn svd_theorem_nuclear_norm() {
    let a = mat2(&[3.0, 0.0, 0.0, 1.0]);
    let r = svd(&a);
    let nuclear_norm: f64 = r.sigma.iter().sum();
    assert_close(nuclear_norm, 4.0, 1e-14, "nuclear norm");
}

// ─── Section 3: Reconstruction theorem ──────────────────────────────────────

/// Reconstruction: A = U Σ V^T (2×2 well-conditioned).
#[test]
fn svd_reconstruction_2x2() {
    let a = mat2(&[1.0, 2.0, 3.0, 4.0]);
    let r = svd(&a);
    let reconstructed = reconstruct_from_svd(&r.u, &r.sigma, &r.vt, 2, 2);
    let err = max_abs_err(&reconstructed, &a);
    assert!(err < 1e-13, "2×2 reconstruction err={err:.2e}");
}

/// Reconstruction: 3×2 tall matrix.
#[test]
fn svd_reconstruction_3x2_tall() {
    let a = mat(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = svd(&a);
    let reconstructed = reconstruct_from_svd(&r.u, &r.sigma, &r.vt, 3, 2);
    let err = max_abs_err(&reconstructed, &a);
    assert!(err < 1e-13, "3×2 tall reconstruction err={err:.2e}");
}

/// Reconstruction: 2×3 wide matrix.
#[test]
fn svd_reconstruction_2x3_wide() {
    let a = mat(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = svd(&a);
    let reconstructed = reconstruct_from_svd(&r.u, &r.sigma, &r.vt, 2, 3);
    let err = max_abs_err(&reconstructed, &a);
    assert!(err < 1e-13, "2×3 wide reconstruction err={err:.2e}");
}

/// Reconstruction at large scale: 1000x entries — tests numerical stability
/// when matrix entries are 1000x larger than unit scale.
#[test]
fn svd_reconstruction_scaled_1000x() {
    let data: Vec<f64> = (1..=4).map(|i| i as f64 * 1000.0).collect();
    let a = mat2(&data);
    let r = svd(&a);
    let reconstructed = reconstruct_from_svd(&r.u, &r.sigma, &r.vt, 2, 2);
    let err = max_abs_err(&reconstructed, &a);
    // At 1000x scale, absolute error should scale proportionally
    assert!(err < 1e-9, "1000x scaled reconstruction err={err:.2e}");
}

/// Reconstruction at small scale: 1e-8x entries.
#[test]
fn svd_reconstruction_scaled_1e_minus_8() {
    let data: Vec<f64> = (1..=4).map(|i| i as f64 * 1e-8).collect();
    let a = mat2(&data);
    let r = svd(&a);
    let reconstructed = reconstruct_from_svd(&r.u, &r.sigma, &r.vt, 2, 2);
    let rel_err = if max_abs(&a) > 0.0 {
        max_abs_err(&reconstructed, &a) / max_abs(&a)
    } else { 0.0 };
    assert!(rel_err < 1e-10, "1e-8 scaled reconstruction relative err={rel_err:.2e}");
}

/// Helper: reconstruct A = U Σ V^T from SVD components.
fn reconstruct_from_svd(u: &Mat, sigma: &[f64], vt: &Mat, m: usize, n: usize) -> Mat {
    let k = sigma.len();
    let mut result = Mat::zeros(m, n);
    for j in 0..k {
        // outer product u[:,j] * vt[j,:] * sigma[j]
        for i in 0..m {
            for l in 0..n {
                let val = result.get(i, l) + sigma[j] * u.get(i, j) * vt.get(j, l);
                result.set(i, l, val);
            }
        }
    }
    result
}

// ─── Section 4: Ill-conditioned matrices ────────────────────────────────────

/// Hilbert-2: H₂ = [[1, 1/2], [1/2, 1/3]].
/// Exact eigenvalues of H₂ are analytically computable.
/// H₂ is symmetric PSD, so SVs = eigenvalues.
/// trace = 4/3, det = 1/3 - 1/4 = 1/12.
/// λ = (4/3 ± sqrt(16/9 - 4/12)) / 2 = (4/3 ± sqrt(16/9 - 1/3)) / 2
///   = (4/3 ± sqrt(13/9)) / 2
/// σ₁ = (4/3 + sqrt(13)/3) / 2 = (4 + sqrt(13)) / 6
/// σ₂ = (4/3 - sqrt(13)/3) / 2 = (4 - sqrt(13)) / 6
/// sqrt(13) ≈ 3.60555...
/// σ₁ ≈ (4 + 3.60555) / 6 ≈ 1.26759...
/// σ₂ ≈ (4 - 3.60555) / 6 ≈ 0.06574...
/// Condition number κ = σ₁/σ₂ ≈ 19.28
#[test]
fn svd_ill_conditioned_hilbert_2x2() {
    let a = mat2(&[1.0, 0.5, 0.5, 1.0 / 3.0]);
    let r = svd(&a);

    let s13 = 13.0_f64.sqrt();
    let sigma1 = (4.0 + s13) / 6.0;
    let sigma2 = (4.0 - s13) / 6.0;

    assert_close(r.sigma[0], sigma1, 1e-13, "H₂ σ₁");
    assert_close(r.sigma[1], sigma2, 1e-13, "H₂ σ₂");
}

/// Hilbert-3: H₃ = [[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]].
/// Condition number κ(H₃) ≈ 524. Published values from Wilkinson (1965):
/// σ₁ ≈ 1.408..., σ₂ ≈ 1.220e-1, σ₃ ≈ 2.687e-3 (via A^T A eigenvalues).
/// We test the reconstruction and orthogonality, not exact SVs.
#[test]
fn svd_ill_conditioned_hilbert_3x3_reconstruction() {
    let a = mat(3, 3, &[
        1.0, 0.5, 1.0/3.0,
        0.5, 1.0/3.0, 0.25,
        1.0/3.0, 0.25, 0.2,
    ]);
    let r = svd(&a);

    // Reconstruction error should be < 1e-12
    let reconstructed = reconstruct_from_svd(&r.u, &r.sigma, &r.vt, 3, 3);
    let err = max_abs_err(&reconstructed, &a);
    assert!(err < 1e-12, "H₃ reconstruction err={err:.2e}");

    // Condition number check: σ_max / σ_min ≈ 524
    let kappa = r.sigma[0] / r.sigma[2];
    assert!(kappa > 100.0 && kappa < 10000.0,
        "H₃ condition number = {kappa:.1} (expected ~524)");

    // U and V^T must still be orthogonal despite ill-conditioning
    let utu = mat_mul(&r.u.t(), &r.u);
    assert_is_identity(&utu, 1e-10, "H₃ U^T U");
}

/// Hilbert-4: known condition number κ ≈ 15514.
/// Published value: σ₁ ≈ 1.50 (from spectral properties of Hilbert matrix).
#[test]
fn svd_ill_conditioned_hilbert_4x4() {
    let a = {
        let mut h = Mat::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                h.set(i, j, 1.0 / ((i + j + 1) as f64));
            }
        }
        h
    };
    let r = svd(&a);

    // Reconstruction error
    let reconstructed = reconstruct_from_svd(&r.u, &r.sigma, &r.vt, 4, 4);
    let err = max_abs_err(&reconstructed, &a);
    assert!(err < 1e-11, "H₄ reconstruction err={err:.2e}");

    // Condition number κ(H₄) ≈ 15514 — verify order of magnitude
    let kappa = r.sigma[0] / r.sigma[3];
    assert!(kappa > 1000.0 && kappa < 1e6,
        "H₄ condition number = {kappa:.1e} (expected ~15514)");

    // Orthogonality should hold even for ill-conditioned matrices
    let utu = mat_mul(&r.u.t(), &r.u);
    assert_is_identity(&utu, 1e-8, "H₄ U^T U");
}

/// Vandermonde-like matrix at [1, 2, 4]: very ill-conditioned.
/// V = [[1, 1, 1], [1, 2, 4], [1, 4, 16]].
/// Exact SVs: too complex to derive analytically. Test reconstruction only.
#[test]
fn svd_ill_conditioned_vandermonde_3x3() {
    let a = mat(3, 3, &[
        1.0, 1.0, 1.0,
        1.0, 2.0, 4.0,
        1.0, 4.0, 16.0,
    ]);
    let r = svd(&a);
    let reconstructed = reconstruct_from_svd(&r.u, &r.sigma, &r.vt, 3, 3);
    let err = max_abs_err(&reconstructed, &a);
    // Vandermonde at [1,2,4] has moderate condition (~100), so 1e-10 is achievable
    assert!(err < 1e-10, "Vandermonde reconstruction err={err:.2e}");
}

// ─── Section 5: Near-rank-deficient matrices ─────────────────────────────────

/// Rank-1 matrix: exactly one nonzero singular value.
/// A = [[2], [3], [6]] is a column vector; as 3×1 its SV = ||A|| = 7.
/// As a 3×3 rank-1 matrix: one SV = ||u|| * ||v||, rest zero.
#[test]
fn svd_near_rank_deficient_exact_rank1() {
    // uv^T where u=[2,3,6], v=[1,0,0] → first column is [2,3,6], rest zero
    let a = mat(3, 3, &[2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0]);
    let r = svd(&a);
    assert_close(r.sigma[0], 7.0, 1e-13, "rank-1 3x3 σ₁ = 7");
    assert!(r.sigma[1].abs() < 1e-13, "rank-1 σ[1] ≈ 0, got {:.2e}", r.sigma[1]);
    assert!(r.sigma[2].abs() < 1e-13, "rank-1 σ[2] ≈ 0, got {:.2e}", r.sigma[2]);
}

/// Near-rank-deficient: two nearly identical rows.
/// A = [[1, 0], [1, 0], [0, 1]] → rows 0 and 1 identical → rank 2 (as expected for 3×2).
#[test]
fn svd_near_rank_deficient_repeated_rows() {
    let a = mat(3, 2, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    let r = svd(&a);
    // SVs: A^T A = [[2,0],[0,1]] → σ₁ = sqrt(2), σ₂ = 1
    assert_close(r.sigma[0], 2.0_f64.sqrt(), 1e-13, "repeated rows σ₁");
    assert_close(r.sigma[1], 1.0, 1e-13, "repeated rows σ₂");
}

/// Near-rank-1 matrix: perturbed by ε in one entry.
/// A = [[1, ε], [1, ε]] is exactly rank-1 at ε=0. At ε=1e-8, it's full-rank but barely.
/// σ₂ should be ≈ ε * sqrt(2).
#[test]
fn svd_near_rank1_perturbed() {
    let eps = 1e-8_f64;
    // A = [[1, eps], [1, eps]] — rank 1 at eps=0
    // A^T A = [[2, 2*eps], [2*eps, 2*eps^2]]
    // det = 4*eps^2 - 4*eps^2 = 0 at eps=0 but eps^2 term matters
    // For small eps: σ₁ ≈ sqrt(2 + 2*eps^2) ≈ sqrt(2), σ₂ ≈ eps * sqrt(2)
    let a = mat2(&[1.0, eps, 1.0, eps]);
    let r = svd(&a);
    // σ₁ ≈ sqrt(2*(1+eps^2)) ≈ sqrt(2)
    assert_close(r.sigma[0], (2.0 * (1.0 + eps * eps)).sqrt(), 1e-10, "near-rank-1 σ₁");
    // σ₂ should be very small (≈ 0 for rank-1 matrix, eps*sqrt(2) for perturbed)
    assert!(r.sigma[1] < 1e-6, "near-rank-1 σ₂={:.2e}, expected small", r.sigma[1]);
}

// ─── Section 6: Adversarial edge cases ──────────────────────────────────────

/// Zero matrix: all singular values are zero.
#[test]
fn svd_adversarial_zero_matrix_3x3() {
    let a = Mat::zeros(3, 3);
    let r = svd(&a);
    let sv_max = r.sigma.iter().cloned().fold(0.0_f64, f64::max);
    assert!(sv_max == 0.0, "zeros(3,3): max SV = {sv_max:.2e}, expected 0");
}

/// Zero matrix 3×2: correct shape handling.
#[test]
fn svd_adversarial_zero_matrix_3x2() {
    let a = Mat::zeros(3, 2);
    let r = svd(&a);
    assert_eq!(r.sigma.len(), 2, "3×2 zero: should have 2 SVs");
}

/// 1×1 matrix: single singular value = |entry|.
#[test]
fn svd_adversarial_1x1() {
    let a = mat(1, 1, &[7.0]);
    let r = svd(&a);
    assert_eq!(r.sigma.len(), 1);
    assert_close(r.sigma[0], 7.0, 1e-15, "1×1 SV");
}

/// 1×1 zero: SV = 0.
#[test]
fn svd_adversarial_1x1_zero() {
    let a = mat(1, 1, &[0.0]);
    let r = svd(&a);
    assert_eq!(r.sigma.len(), 1);
    assert!(r.sigma[0] == 0.0, "1×1 zero SV should be 0");
}

/// Single column: SVD of a tall m×1 matrix.
/// A = [1,2,3]^T (3×1). SV = ||A|| = sqrt(14). U is 3×3, V^T is 1×1.
#[test]
fn svd_adversarial_single_column() {
    let a = mat(3, 1, &[1.0, 2.0, 3.0]);
    let r = svd(&a);
    assert_eq!(r.sigma.len(), 1, "3×1: should have 1 SV");
    let expected = 14.0_f64.sqrt();
    assert_close(r.sigma[0], expected, 1e-13, "3×1 column SV = ||A||");
    // U should be 3×3 orthogonal
    let utu = mat_mul(&r.u.t(), &r.u);
    assert_is_identity(&utu, 1e-12, "3×1 U^T U");
}

/// Single row: SVD of a 1×m matrix.
/// A = [1,2,3] (1×3). SV = ||A|| = sqrt(14). U is 1×1, V^T is 3×3.
#[test]
fn svd_adversarial_single_row() {
    let a = mat(1, 3, &[1.0, 2.0, 3.0]);
    let r = svd(&a);
    assert_eq!(r.sigma.len(), 1, "1×3: should have 1 SV");
    let expected = 14.0_f64.sqrt();
    assert_close(r.sigma[0], expected, 1e-13, "1×3 row SV = ||A||");
    // V^T should be 3×3 orthogonal
    let vtv = mat_mul(&r.vt, &r.vt.t());
    assert_is_identity(&vtv, 1e-12, "1×3 V^T V");
}

/// Large-entry matrix: all entries = 1e12.
/// Tests numerical stability at extreme scale.
/// [[s,s],[s,s]] = s * [[1,1],[1,1]].
/// [[1,1],[1,1]]^T [[1,1],[1,1]] = [[2,2],[2,2]], eigenvalues 4 and 0.
/// So SVs of [[s,s],[s,s]] are 2s and 0.
#[test]
fn svd_adversarial_large_entries_2x2() {
    let s = 1e12_f64;
    let a = mat2(&[s, s, s, s]); // rank-1 matrix at large scale
    let r = svd(&a);
    // σ₁ = 2s (not s*sqrt(2) — [[1,1],[1,1]] has σ₁=2, not sqrt(2))
    assert_close(r.sigma[0], 2.0 * s, 1.0, "large-entry σ₁ = 2s"); // 1.0 absolute tol at 1e12 scale
    assert!(r.sigma[1].abs() < 1e-3, "large-entry σ₂={:.2e}, expected ≈ 0", r.sigma[1]);
}

/// Orthogonal matrix (Householder): all SVs = 1.
/// H = I - 2 v v^T where v is a unit vector. For v = [1,0]/sqrt(1) = [1,0]:
/// H = I - 2[[1,0],[0,0]] = [[-1,0],[0,1]]. SVs = 1, 1.
#[test]
fn svd_adversarial_householder_2x2() {
    let a = mat2(&[-1.0, 0.0, 0.0, 1.0]);
    let r = svd(&a);
    assert_close(r.sigma[0], 1.0, 1e-14, "Householder σ[0]");
    assert_close(r.sigma[1], 1.0, 1e-14, "Householder σ[1]");
}

/// Antisymmetric matrix: [[0, a], [-a, 0]].
/// A^T A = [[a², 0], [0, a²]] → both SVs = |a|.
#[test]
fn svd_adversarial_antisymmetric_2x2() {
    let a_val = 3.7_f64;
    let a = mat2(&[0.0, a_val, -a_val, 0.0]);
    let r = svd(&a);
    assert_close(r.sigma[0], a_val, 1e-13, "antisymmetric σ[0]");
    assert_close(r.sigma[1], a_val, 1e-13, "antisymmetric σ[1]");
}
