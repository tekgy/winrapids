//! Parity tests for `linear_algebra::pinv`.
//!
//! Covers:
//! - NumPy oracle agreement (6 cases, all ≤ 1.11e-15 absolute error)
//! - CRITICAL: rcond fix — relative threshold zeroes noise SV on scaled matrix
//! - Moore-Penrose conditions 1–4 over random matrices
//! - Explicit rcond override
//! - Edge cases: zero matrix, identity, overdetermined, underdetermined
//!
//! See docs/research/atomic-industrialization/pinv.md for full workup.
//!
//! Run: cargo test --test workup_pinv -- --nocapture
//! (use CARGO_TARGET_DIR=target2 if main target dir has broken archive)

use tambear::linear_algebra::{Mat, mat_mul, pinv};

// ─── helpers ────────────────────────────────────────────────────────────────

fn mat(rows: usize, cols: usize, data: &[f64]) -> Mat {
    Mat::from_vec(rows, cols, data.to_vec())
}

fn mat2(data: &[f64]) -> Mat {
    mat(2, 2, data)
}

fn max_abs_err(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.rows, b.rows, "row mismatch");
    assert_eq!(a.cols, b.cols, "col mismatch");
    let mut m = 0.0_f64;
    for i in 0..a.rows {
        for j in 0..a.cols {
            m = m.max((a.get(i, j) - b.get(i, j)).abs());
        }
    }
    m
}

fn max_abs(a: &Mat) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..a.rows {
        for j in 0..a.cols {
            m = m.max(a.get(i, j).abs());
        }
    }
    m
}

fn mat_t(a: &Mat) -> Mat {
    a.t()
}

/// Pseudo-random f64 in [-1, 1] using a tiny LCG — no dependency on rand.
fn lcg_rand(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = ((*state >> 11) as u64) | 0x3FF0000000000000_u64;
    let v = f64::from_bits(bits) - 1.0; // [0, 1)
    v * 2.0 - 1.0 // [-1, 1)
}

fn random_mat(rows: usize, cols: usize, seed: u64) -> Mat {
    let mut state = seed;
    let data: Vec<f64> = (0..rows * cols).map(|_| lcg_rand(&mut state)).collect();
    Mat::from_vec(rows, cols, data)
}

// ─── Moore-Penrose condition checks ─────────────────────────────────────────

/// Condition 1: A A⁺ A = A
fn mp_cond1(a: &Mat, api: &Mat) -> f64 {
    let aapi = mat_mul(a, api);
    let aapi_a = mat_mul(&aapi, a);
    max_abs_err(&aapi_a, a)
}

/// Condition 2: A⁺ A A⁺ = A⁺
fn mp_cond2(a: &Mat, api: &Mat) -> f64 {
    let apia = mat_mul(api, a);
    let apia_api = mat_mul(&apia, api);
    max_abs_err(&apia_api, api)
}

/// Condition 3: (A A⁺)^T = A A⁺
fn mp_cond3(a: &Mat, api: &Mat) -> f64 {
    let aapi = mat_mul(a, api);
    let aapi_t = mat_t(&aapi);
    max_abs_err(&aapi, &aapi_t)
}

/// Condition 4: (A⁺ A)^T = A⁺ A
fn mp_cond4(a: &Mat, api: &Mat) -> f64 {
    let apia = mat_mul(api, a);
    let apia_t = mat_t(&apia);
    max_abs_err(&apia, &apia_t)
}

// ─── Oracle cases (NumPy reference) ─────────────────────────────────────────

/// Case 1: pinv(I₃) = I₃
/// Oracle: identity. Max error = 0.
#[test]
fn pinv_oracle_case1_identity() {
    let a = Mat::eye(3);
    let got = pinv(&a, None);
    let err = max_abs_err(&got, &a);
    assert!(err == 0.0, "Case 1 (I_3): expected 0, got {err:.2e}");
}

/// Case 2: pinv([[1,2],[3,4]]) = [[-2,1],[1.5,-0.5]]
/// Oracle: NumPy linalg.pinv. Max error ≤ 1.11e-15 (≤ 1 ULP).
#[test]
fn pinv_oracle_case2_full_rank_2x2() {
    let a = mat2(&[1.0, 2.0, 3.0, 4.0]);
    let oracle = mat2(&[-2.0, 1.0, 1.5, -0.5]);
    let got = pinv(&a, None);
    let err = max_abs_err(&got, &oracle);
    assert!(
        err < 2e-15,
        "Case 2 (2x2 full rank): max_err={err:.3e}, expected < 2e-15"
    );
}

/// Case 3: pinv([[1,2],[2,4]]) (rank-1) = [[0.04,0.08],[0.08,0.16]]
/// Oracle: NumPy. Max error = 0 (exact floating-point).
#[test]
fn pinv_oracle_case3_rank_deficient_2x2() {
    let a = mat2(&[1.0, 2.0, 2.0, 4.0]);
    let oracle = mat2(&[0.04, 0.08, 0.08, 0.16]);
    let got = pinv(&a, None);
    let err = max_abs_err(&got, &oracle);
    assert!(
        err < 1e-15,
        "Case 3 (rank-1): max_err={err:.3e}, expected < 1e-15"
    );
}

/// Case 4: pinv([[1,0],[0,1],[1,1]]) — 3×2 overdetermined
/// Oracle: NumPy [[2/3,-1/3,1/3],[-1/3,2/3,1/3]]. Max error < 4e-16.
#[test]
fn pinv_oracle_case4_overdetermined_3x2() {
    let a = mat(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    // NumPy oracle (2×3)
    let two_thirds = 2.0 / 3.0;
    let one_third = 1.0 / 3.0;
    let oracle = mat(
        2,
        3,
        &[two_thirds, -one_third, one_third, -one_third, two_thirds, one_third],
    );
    let got = pinv(&a, None);
    let err = max_abs_err(&got, &oracle);
    assert!(
        err < 5e-16,
        "Case 4 (3x2 overdetermined): max_err={err:.3e}, expected < 5e-16"
    );
}

/// Case 5 (CRITICAL — rcond fix verification):
/// pinv(diag(1e6, 5e-11)) must equal diag(1e-6, 0.0) with relative rcond.
///
/// Old behavior (absolute rcond=1e-12): 5e-11 > 1e-12 → kept → pinv(1,1)=2e10 WRONG.
/// New behavior (relative rcond): threshold = 2*eps*1e6 ≈ 4.44e-10 > 5e-11 → zeroed → 0.0.
///
/// This is the differentiating test between old and new rcond behavior.
#[test]
fn pinv_oracle_case5_rcond_fix_relative_threshold() {
    let a = mat2(&[1e6, 0.0, 0.0, 5e-11]);
    let got = pinv(&a, None);

    // (0,0): 1/1e6 = 1e-6
    let v00 = got.get(0, 0);
    assert!(
        (v00 - 1e-6).abs() < 1e-20,
        "Case 5: pinv(0,0)={v00:.3e}, expected 1e-6"
    );

    // (1,1): 5e-11 is below relative threshold → must be ZERO
    // Old behavior would give 2e10 here
    let v11 = got.get(1, 1);
    assert!(
        v11 == 0.0,
        "Case 5 (rcond fix): pinv(1,1)={v11:.3e}, expected 0.0 (relative rcond must zero noise SV)"
    );
}

/// Case 6: pinv(zeros(3×3)) = zeros(3×3)
/// Oracle: trivial. Max error = 0.
#[test]
fn pinv_oracle_case6_zero_matrix() {
    let a = Mat::zeros(3, 3);
    let got = pinv(&a, None);
    let err = max_abs(&got);
    assert!(err == 0.0, "Case 6 (zeros): max element={err:.2e}, expected 0");
}

// ─── Explicit rcond override ─────────────────────────────────────────────────

/// Explicit rcond=0.0 keeps all nonzero singular values (same as default for
/// well-conditioned matrices).
#[test]
fn pinv_explicit_rcond_zero_matches_default_for_full_rank() {
    let a = mat2(&[1.0, 2.0, 3.0, 4.0]);
    let default = pinv(&a, None);
    let explicit = pinv(&a, Some(0.0));
    let err = max_abs_err(&default, &explicit);
    // With rcond=0.0, threshold=0: all SVs kept. For full-rank 2x2 these agree.
    assert!(err < 2e-15, "explicit rcond=0 vs default: err={err:.2e}");
}

/// Explicit rcond=1.0 zeroes ALL singular values (max SV of [[1,2],[3,4]] < 6,
/// so threshold 1.0 keeps nothing) — should give zero pseudoinverse.
#[test]
fn pinv_explicit_rcond_large_zeroes_all_svs() {
    // [[1,2],[3,4]] has SVs ≈ 5.46 and 0.37. rcond=10 > both → all zeroed.
    let a = mat2(&[1.0, 2.0, 3.0, 4.0]);
    let got = pinv(&a, Some(10.0));
    let err = max_abs(&got);
    assert!(err == 0.0, "explicit rcond=10: all SVs should be zeroed, got max={err:.2e}");
}

/// Explicit rcond partitions which SVs to keep for a known diagonal matrix.
#[test]
fn pinv_explicit_rcond_selects_sv_boundary() {
    // diag(5.0, 0.1): with rcond=0.5, keep only SV=5.0 (0.1 < 0.5)
    let a = mat2(&[5.0, 0.0, 0.0, 0.1]);
    let got = pinv(&a, Some(0.5));
    let v00 = got.get(0, 0);
    let v11 = got.get(1, 1);
    assert!(
        (v00 - 0.2).abs() < 1e-15,
        "explicit rcond=0.5: pinv(0,0)={v00:.6}, expected 0.2"
    );
    assert!(
        v11 == 0.0,
        "explicit rcond=0.5: pinv(1,1)={v11:.6}, expected 0.0 (SV=0.1 < rcond=0.5)"
    );
}

// ─── Moore-Penrose conditions ────────────────────────────────────────────────

/// Case 2 (2x2 full rank): all four Moore-Penrose conditions.
#[test]
fn pinv_moore_penrose_2x2_full_rank() {
    let a = mat2(&[1.0, 2.0, 3.0, 4.0]);
    let api = pinv(&a, None);
    assert!(mp_cond1(&a, &api) < 1e-13, "MP cond 1 failed");
    assert!(mp_cond2(&a, &api) < 1e-13, "MP cond 2 failed");
    assert!(mp_cond3(&a, &api) < 1e-13, "MP cond 3 failed");
    assert!(mp_cond4(&a, &api) < 1e-13, "MP cond 4 failed");
}

/// Case 3 (rank-1 2x2): all four Moore-Penrose conditions.
#[test]
fn pinv_moore_penrose_rank1_2x2() {
    let a = mat2(&[1.0, 2.0, 2.0, 4.0]);
    let api = pinv(&a, None);
    assert!(mp_cond1(&a, &api) < 1e-13, "MP cond 1 failed");
    assert!(mp_cond2(&a, &api) < 1e-13, "MP cond 2 failed");
    assert!(mp_cond3(&a, &api) < 1e-13, "MP cond 3 failed");
    assert!(mp_cond4(&a, &api) < 1e-13, "MP cond 4 failed");
}

/// Case 4 (3×2 overdetermined): all four Moore-Penrose conditions.
#[test]
fn pinv_moore_penrose_overdetermined_3x2() {
    let a = mat(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let api = pinv(&a, None);
    assert!(mp_cond1(&a, &api) < 1e-13, "MP cond 1 failed");
    assert!(mp_cond2(&a, &api) < 1e-13, "MP cond 2 failed");
    assert!(mp_cond3(&a, &api) < 1e-13, "MP cond 3 failed");
    assert!(mp_cond4(&a, &api) < 1e-13, "MP cond 4 failed");
}

/// 2×3 underdetermined: all four Moore-Penrose conditions.
#[test]
fn pinv_moore_penrose_underdetermined_2x3() {
    let a = mat(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let api = pinv(&a, None);
    assert!(mp_cond1(&a, &api) < 1e-12, "MP cond 1 failed");
    assert!(mp_cond2(&a, &api) < 1e-12, "MP cond 2 failed");
    assert!(mp_cond3(&a, &api) < 1e-12, "MP cond 3 failed");
    assert!(mp_cond4(&a, &api) < 1e-12, "MP cond 4 failed");
}

/// 50 random 5×3 matrices — all four Moore-Penrose conditions.
/// Seed 54321 matches §5.3 in the workup document.
#[test]
fn pinv_moore_penrose_random_5x3_50_cases() {
    let mut seed: u64 = 54321;
    let mut max_e1 = 0.0_f64;
    let mut max_e2 = 0.0_f64;
    let mut max_e3 = 0.0_f64;
    let mut max_e4 = 0.0_f64;

    for _ in 0..50 {
        // Advance seed per matrix (each row is independently seeded)
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = random_mat(5, 3, seed);
        let api = pinv(&a, None);
        max_e1 = max_e1.max(mp_cond1(&a, &api));
        max_e2 = max_e2.max(mp_cond2(&a, &api));
        max_e3 = max_e3.max(mp_cond3(&a, &api));
        max_e4 = max_e4.max(mp_cond4(&a, &api));
    }

    assert!(max_e1 < 1e-12, "MP cond 1 (5x3 random): max_err={max_e1:.2e}");
    assert!(max_e2 < 1e-12, "MP cond 2 (5x3 random): max_err={max_e2:.2e}");
    assert!(max_e3 < 1e-12, "MP cond 3 (5x3 random): max_err={max_e3:.2e}");
    assert!(max_e4 < 1e-12, "MP cond 4 (5x3 random): max_err={max_e4:.2e}");
}

/// 50 random 3×5 matrices — all four Moore-Penrose conditions.
#[test]
fn pinv_moore_penrose_random_3x5_50_cases() {
    let mut seed: u64 = 99999;
    let mut max_e1 = 0.0_f64;
    let mut max_e2 = 0.0_f64;
    let mut max_e3 = 0.0_f64;
    let mut max_e4 = 0.0_f64;

    for _ in 0..50 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = random_mat(3, 5, seed);
        let api = pinv(&a, None);
        max_e1 = max_e1.max(mp_cond1(&a, &api));
        max_e2 = max_e2.max(mp_cond2(&a, &api));
        max_e3 = max_e3.max(mp_cond3(&a, &api));
        max_e4 = max_e4.max(mp_cond4(&a, &api));
    }

    assert!(max_e1 < 1e-12, "MP cond 1 (3x5 random): max_err={max_e1:.2e}");
    assert!(max_e2 < 1e-12, "MP cond 2 (3x5 random): max_err={max_e2:.2e}");
    assert!(max_e3 < 1e-12, "MP cond 3 (3x5 random): max_err={max_e3:.2e}");
    assert!(max_e4 < 1e-12, "MP cond 4 (3x5 random): max_err={max_e4:.2e}");
}

// ─── Invariant: pinv(A^T) = pinv(A)^T ───────────────────────────────────────

#[test]
fn pinv_transpose_invariant_2x2() {
    let a = mat2(&[1.0, 2.0, 3.0, 4.0]);
    let at = mat_t(&a);
    let pinv_a = pinv(&a, None);
    let pinv_at = pinv(&at, None);
    let pinv_a_t = mat_t(&pinv_a);
    let err = max_abs_err(&pinv_at, &pinv_a_t);
    assert!(err < 2e-15, "pinv(A^T) vs pinv(A)^T: err={err:.2e}");
}

#[test]
fn pinv_transpose_invariant_3x2() {
    let a = mat(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let at = mat_t(&a);
    let pinv_a = pinv(&a, None);
    let pinv_at = pinv(&at, None);
    let pinv_a_t = mat_t(&pinv_a);
    let err = max_abs_err(&pinv_at, &pinv_a_t);
    assert!(err < 2e-15, "pinv(A^T) vs pinv(A)^T (3x2): err={err:.2e}");
}

// ─── Adversarial inputs ───────────────────────────────────────────────────────

/// Zero matrix pseudoinverse is zero.
#[test]
fn pinv_adversarial_zero_matrix_2x3() {
    let a = Mat::zeros(2, 3);
    let got = pinv(&a, None);
    assert_eq!(got.rows, 3, "pinv of 2x3 should have 3 rows");
    assert_eq!(got.cols, 2, "pinv of 2x3 should have 2 cols");
    assert!(max_abs(&got) == 0.0, "pinv(zeros) should be zeros");
}

/// 1×1 matrix: pinv([v]) = [1/v]
#[test]
fn pinv_adversarial_1x1() {
    let a = mat(1, 1, &[7.0]);
    let got = pinv(&a, None);
    let v = got.get(0, 0);
    assert!((v - 1.0 / 7.0).abs() < 1e-15, "pinv([7]) = [1/7], got {v:.6}");
}

/// 1×1 zero matrix: pinv([0]) = [0]
#[test]
fn pinv_adversarial_1x1_zero() {
    let a = mat(1, 1, &[0.0]);
    let got = pinv(&a, None);
    assert!(got.get(0, 0) == 0.0, "pinv([0]) should be [0]");
}

/// Large-scale diagonal: diag(1e8, 1e4, 1e-8) with relative rcond.
/// SV=1e-8 should be KEPT (it's at the threshold boundary for a 3×3 matrix).
/// Threshold = 3 * eps * 1e8 ≈ 6.66e-8 > 1e-8 → zeroed.
#[test]
fn pinv_adversarial_scaled_diagonal_3x3() {
    let a = mat(
        3,
        3,
        &[1e8, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0, 0.0, 1e-8],
    );
    let got = pinv(&a, None);
    // SV₁=1e8: 1/1e8 = 1e-8
    let v00 = got.get(0, 0);
    assert!((v00 - 1e-8).abs() < 1e-22, "scaled diag (0,0): got {v00:.3e}");
    // SV₂=1e4: 1/1e4 = 1e-4
    let v11 = got.get(1, 1);
    assert!((v11 - 1e-4).abs() < 1e-18, "scaled diag (1,1): got {v11:.3e}");
    // SV₃=1e-8: below threshold (3*eps*1e8 ≈ 6.66e-8) → zeroed
    let v22 = got.get(2, 2);
    assert!(
        v22 == 0.0,
        "scaled diag (2,2): SV=1e-8 below threshold, got {v22:.3e}"
    );
}

/// Full-rank square: A A⁺ = I
#[test]
fn pinv_adversarial_full_rank_square_is_inverse() {
    let a = mat2(&[2.0, 1.0, 1.0, 3.0]);
    let api = pinv(&a, None);
    let product = mat_mul(&a, &api);
    let eye = Mat::eye(2);
    let err = max_abs_err(&product, &eye);
    assert!(err < 1e-14, "full rank: A A⁺ should = I, err={err:.2e}");
}

/// Orthonormal columns: A⁺ = A^T
#[test]
fn pinv_adversarial_orthonormal_columns() {
    // Q = [1/sqrt(2), 0; 0, 1; -1/sqrt(2), 0] — orthonormal columns
    let s2 = 2.0_f64.sqrt();
    let a = mat(3, 2, &[1.0 / s2, 0.0, 0.0, 1.0, -1.0 / s2, 0.0]);
    let at = mat_t(&a);
    let api = pinv(&a, None);
    let err = max_abs_err(&api, &at);
    assert!(
        err < 1e-14,
        "orthonormal columns: pinv should = A^T, err={err:.2e}"
    );
}
