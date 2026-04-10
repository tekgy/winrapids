//! Adversarial SVD tests + faer benchmark comparison
//!
//! Part 1: Adversarial accuracy tests targeting known failure modes
//!         (ill-conditioning, rank deficiency, edge shapes, near-zero entries)
//! Part 2: faer 0.24 as gold standard — singular value and reconstruction comparison
//! Part 3: Performance benchmarks (ignored by default)
//!
//! Run accuracy tests:  cargo test --test svd_adversarial
//! Run benchmarks:      cargo test --test svd_adversarial -- --ignored --nocapture

use tambear::linear_algebra::*;
use std::time::Instant;

// =========================================================================
// Helpers
// =========================================================================

const SEPARATOR: &str = "========================================================================";
const THIN_SEP: &str = "------------------------------------------------------------------------";

/// Frobenius norm of the difference A - B.
fn diff_fro(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let mut s = 0.0;
    for i in 0..a.rows {
        for j in 0..a.cols {
            let d = a.get(i, j) - b.get(i, j);
            s += d * d;
        }
    }
    s.sqrt()
}

/// Reconstruct A from SVD: U * diag(sigma) * VT.
fn reconstruct_svd(svd_res: &SvdResult, m: usize, n: usize) -> Mat {
    let k = svd_res.sigma.len();
    let mut result = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..k {
                s += svd_res.u.get(i, l) * svd_res.sigma[l] * svd_res.vt.get(l, j);
            }
            result.set(i, j, s);
        }
    }
    result
}

/// Check that U is orthogonal: U^T * U approx I (up to dimensions used).
fn check_orthogonal(u: &Mat, tol: f64, label: &str) {
    let utu = mat_mul(&u.t(), u);
    let eye = Mat::eye(utu.rows);
    let err = diff_fro(&utu, &eye);
    assert!(
        err < tol,
        "{}: U^T U not identity, error = {:.2e} (tol = {:.0e})",
        label, err, tol
    );
}

/// Hilbert matrix H(i,j) = 1 / (i + j + 1).
fn hilbert(n: usize) -> Mat {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = 1.0 / (i as f64 + j as f64 + 1.0);
        }
    }
    Mat::from_vec(n, n, data)
}

/// Deterministic pseudo-random matrix using modular arithmetic.
/// Values in [0, 1). seed_a and seed_b control the pattern.
fn pseudo_random_mat(rows: usize, cols: usize, seed_a: usize, seed_b: usize) -> Mat {
    let mut data = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            data[i * cols + j] = ((i * seed_a + j * seed_b + 3) % 97) as f64 / 97.0;
        }
    }
    Mat::from_vec(rows, cols, data)
}

/// Build a symmetric positive definite matrix: A^T * A + eps * I.
fn make_spd(base: &Mat, eps: f64) -> Mat {
    let ata = mat_mul(&base.t(), base);
    let n = ata.rows;
    let mut result = ata;
    for i in 0..n {
        let v = result.get(i, i);
        result.set(i, i, v + eps);
    }
    result
}

/// Create a faer Mat from a tambear Mat.
fn to_faer(m: &Mat) -> faer::Mat<f64> {
    faer::Mat::<f64>::from_fn(m.rows, m.cols, |i, j| m.get(i, j))
}

/// Extract sorted (descending) singular values from faer SVD.
fn faer_svd_sigma(fa: &faer::Mat<f64>) -> Vec<f64> {
    let mut sigma = fa.singular_values().expect("faer singular_values failed");
    sigma.sort_by(|a: &f64, b: &f64| b.partial_cmp(a).unwrap());
    sigma
}

// =========================================================================
// Part 1: Adversarial SVD Accuracy Tests
// =========================================================================

#[test]
fn svd_adversarial_diagonal_conditioning() {
    // Diagonal matrix with sigma = [1, 1e-8].
    // The old A^T*A approach gave ~41% error on the small singular value
    // because kappa^2 = 1e16 pushes the eigenvalue below machine epsilon.
    // One-sided Jacobi should recover both values to near machine precision.
    let a = Mat::diag(&[1.0, 1e-8]);
    let res = svd(&a);

    assert_eq!(res.sigma.len(), 2);

    let err_large = (res.sigma[0] - 1.0).abs();
    let err_small = (res.sigma[1] - 1e-8).abs();

    assert!(
        err_large < 1e-14,
        "large sigma error: {:.2e} (expected < 1e-14)", err_large
    );
    assert!(
        err_small < 1e-14 * 1e-8,
        "small sigma error: {:.2e} (expected < {:.0e})", err_small, 1e-14 * 1e-8
    );

    // Reconstruction check
    let recon = reconstruct_svd(&res, 2, 2);
    let err = diff_fro(&a, &recon);
    assert!(err < 1e-14, "reconstruction error: {:.2e}", err);
}

#[test]
fn svd_adversarial_extreme_diagonal() {
    // sigma = [1, 1e-15] -- near the edge of f64 precision.
    let a = Mat::diag(&[1.0, 1e-15]);
    let res = svd(&a);

    assert_eq!(res.sigma.len(), 2);
    // The large value must be exact
    assert!((res.sigma[0] - 1.0).abs() < 1e-14);
    // The small value: relative error should be reasonable
    let rel_err = (res.sigma[1] - 1e-15).abs() / 1e-15;
    assert!(
        rel_err < 1e-3,
        "relative error on sigma=1e-15: {:.2e} (got {:.6e})", rel_err, res.sigma[1]
    );
}

#[test]
fn svd_adversarial_hilbert_4x4() {
    // 4x4 Hilbert matrix: kappa ~ 1.55e4.
    // Well-known torture test for SVD accuracy.
    let a = hilbert(4);
    let res = svd(&a);

    // Reconstruction: ||A - U*Sigma*V^T|| should be very small
    let recon = reconstruct_svd(&res, 4, 4);
    let err = diff_fro(&a, &recon);
    assert!(
        err < 1e-12,
        "Hilbert 4x4 reconstruction error: {:.2e}", err
    );

    // Condition number: known to be ~15514
    let kappa = cond(&a);
    assert!(kappa > 1e3, "Hilbert kappa too small: {:.2e}", kappa);
    assert!(kappa < 1e5, "Hilbert kappa too large: {:.2e}", kappa);

    // Orthogonality of U and VT
    check_orthogonal(&res.u, 1e-12, "Hilbert U");
    check_orthogonal(&res.vt.t(), 1e-12, "Hilbert V");
}

#[test]
fn svd_adversarial_hilbert_6x6() {
    // 6x6 Hilbert: kappa ~ 1.5e7 -- significantly harder.
    let a = hilbert(6);
    let res = svd(&a);

    let recon = reconstruct_svd(&res, 6, 6);
    let err = diff_fro(&a, &recon);
    assert!(
        err < 1e-10,
        "Hilbert 6x6 reconstruction error: {:.2e}", err
    );

    let kappa = cond(&a);
    assert!(kappa > 1e6, "Hilbert 6x6 kappa too small: {:.2e}", kappa);
}

#[test]
fn svd_adversarial_rank_deficient() {
    // 3x3 matrix with exact rank 2.
    // Third row = first row + second row.
    let a = Mat::from_rows(&[
        &[1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
        &[5.0, 7.0, 9.0], // = row0 + row1
    ]);

    let res = svd(&a);
    assert_eq!(res.sigma.len(), 3);

    // Smallest singular value should be ~= 0
    assert!(
        res.sigma[2] < 1e-12,
        "rank-deficient: smallest sigma = {:.2e} (expected ~= 0)", res.sigma[2]
    );

    // rank() with a reasonable tolerance should return 2
    let r = rank(&a, 1e-10);
    assert_eq!(r, 2, "rank should be 2, got {}", r);

    // Reconstruction still works
    let recon = reconstruct_svd(&res, 3, 3);
    let err = diff_fro(&a, &recon);
    assert!(err < 1e-12, "rank-deficient reconstruction error: {:.2e}", err);
}

#[test]
fn svd_adversarial_near_zero_matrix() {
    // All entries ~= 1e-300. Must not crash, produce NaN, or inf.
    let tiny = 1e-300;
    let a = Mat::from_vec(3, 3, vec![
        tiny, 2.0 * tiny, 3.0 * tiny,
        4.0 * tiny, 5.0 * tiny, 6.0 * tiny,
        7.0 * tiny, 8.0 * tiny, 9.0 * tiny,
    ]);

    let res = svd(&a);
    assert_eq!(res.sigma.len(), 3);

    for (i, &s) in res.sigma.iter().enumerate() {
        assert!(s.is_finite(), "sigma[{}] is not finite: {}", i, s);
        assert!(!s.is_nan(), "sigma[{}] is NaN", i);
    }

    // U and VT should not contain NaN/inf
    for x in &res.u.data {
        assert!(x.is_finite(), "U contains non-finite: {}", x);
    }
    for x in &res.vt.data {
        assert!(x.is_finite(), "VT contains non-finite: {}", x);
    }
}

#[test]
fn svd_adversarial_wide_matrix() {
    // 2x5 -- tests the m < n transpose path.
    let a = Mat::from_rows(&[
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[6.0, 7.0, 8.0, 9.0, 10.0],
    ]);

    let res = svd(&a);
    assert_eq!(res.sigma.len(), 2, "wide matrix should have min(m,n)=2 singular values");

    // Both should be positive
    assert!(res.sigma[0] > 0.0);
    assert!(res.sigma[1] > 0.0);
    // Descending order
    assert!(res.sigma[0] >= res.sigma[1]);

    // Reconstruction
    let recon = reconstruct_svd(&res, 2, 5);
    let err = diff_fro(&a, &recon);
    assert!(err < 1e-12, "wide matrix reconstruction error: {:.2e}", err);

    // Dimensions
    assert_eq!(res.u.rows, 2);
    assert_eq!(res.u.cols, 2);
    assert_eq!(res.vt.rows, 5);
    assert_eq!(res.vt.cols, 5);
}

#[test]
fn svd_adversarial_single_column() {
    // 5x1 matrix
    let a = Mat::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let res = svd(&a);

    assert_eq!(res.sigma.len(), 1);
    // sigma should be the 2-norm of the column
    let expected_norm = (1.0 + 4.0 + 9.0 + 16.0 + 25.0_f64).sqrt();
    assert!(
        (res.sigma[0] - expected_norm).abs() < 1e-12,
        "single column sigma: got {:.15e}, expected {:.15e}", res.sigma[0], expected_norm
    );

    // Reconstruction
    let recon = reconstruct_svd(&res, 5, 1);
    let err = diff_fro(&a, &recon);
    assert!(err < 1e-12, "single column reconstruction error: {:.2e}", err);
}

#[test]
fn svd_adversarial_single_row() {
    // 1x5 matrix (wide, triggers transpose path)
    let a = Mat::from_vec(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let res = svd(&a);

    assert_eq!(res.sigma.len(), 1);
    let expected_norm = (1.0 + 4.0 + 9.0 + 16.0 + 25.0_f64).sqrt();
    assert!(
        (res.sigma[0] - expected_norm).abs() < 1e-12,
        "single row sigma: got {:.15e}, expected {:.15e}", res.sigma[0], expected_norm
    );

    let recon = reconstruct_svd(&res, 1, 5);
    let err = diff_fro(&a, &recon);
    assert!(err < 1e-12, "single row reconstruction error: {:.2e}", err);
}

#[test]
fn svd_adversarial_1x1() {
    let a = Mat::from_vec(1, 1, vec![42.0]);
    let res = svd(&a);

    assert_eq!(res.sigma.len(), 1);
    assert!((res.sigma[0] - 42.0).abs() < 1e-12);

    // Negative value
    let b = Mat::from_vec(1, 1, vec![-7.5]);
    let res_b = svd(&b);
    assert_eq!(res_b.sigma.len(), 1);
    assert!((res_b.sigma[0] - 7.5).abs() < 1e-12, "1x1 negative: sigma should be |value|");
}

#[test]
fn svd_adversarial_all_zero() {
    let a = Mat::zeros(3, 3);
    let res = svd(&a);

    assert_eq!(res.sigma.len(), 3);
    for (i, &s) in res.sigma.iter().enumerate() {
        assert!(
            s.abs() < 1e-14,
            "zero matrix sigma[{}] = {} (expected 0)", i, s
        );
    }

    // rank should be 0
    let r = rank(&a, 1e-10);
    assert_eq!(r, 0, "zero matrix rank should be 0, got {}", r);
}

#[test]
fn svd_adversarial_identity() {
    let a = Mat::eye(4);
    let res = svd(&a);

    assert_eq!(res.sigma.len(), 4);
    for (i, &s) in res.sigma.iter().enumerate() {
        assert!(
            (s - 1.0).abs() < 1e-14,
            "identity sigma[{}] = {:.15e} (expected 1.0)", i, s
        );
    }

    // U and VT should each be orthogonal
    check_orthogonal(&res.u, 1e-12, "identity U");
    check_orthogonal(&res.vt.t(), 1e-12, "identity V");

    // Reconstruction
    let recon = reconstruct_svd(&res, 4, 4);
    let err = diff_fro(&a, &recon);
    assert!(err < 1e-13, "identity reconstruction error: {:.2e}", err);
}

#[test]
fn svd_adversarial_pinv_overdetermined() {
    // Verify pseudoinverse for a tall matrix solves least squares.
    // A is 4x2, so A^+ is 2x4, and A^+ * A ~= I (2x2).
    let a = Mat::from_rows(&[
        &[1.0, 2.0],
        &[3.0, 4.0],
        &[5.0, 6.0],
        &[7.0, 8.0],
    ]);
    let a_pinv = pinv(&a, None);
    assert_eq!(a_pinv.rows, 2);
    assert_eq!(a_pinv.cols, 4);

    // A^+ * A should ~= I_2
    let product = mat_mul(&a_pinv, &a);
    let eye2 = Mat::eye(2);
    let err = diff_fro(&product, &eye2);
    assert!(err < 1e-10, "pinv: A^+ * A not identity, error = {:.2e}", err);
}

#[test]
fn svd_adversarial_cond_diagonal() {
    // Condition number of diag([10, 1]) = 10.
    let a = Mat::diag(&[10.0, 1.0]);
    let kappa = cond(&a);
    assert!(
        (kappa - 10.0).abs() < 1e-10,
        "cond(diag(10,1)) = {:.6e}, expected 10.0", kappa
    );
}

#[test]
fn svd_adversarial_cond_singular() {
    // Singular matrix -> infinite condition number.
    let a = Mat::from_rows(&[
        &[1.0, 2.0],
        &[2.0, 4.0],
    ]);
    let kappa = cond(&a);
    assert!(kappa > 1e12 || kappa == f64::INFINITY,
        "singular matrix cond should be very large or inf, got {:.2e}", kappa);
}

// =========================================================================
// Part 2: faer Comparison Tests
// =========================================================================

/// Compare tambear SVD singular values against faer's.
fn compare_svd_with_faer(a: &Mat, label: &str, sigma_tol: f64, recon_tol: f64) {
    let m = a.rows;
    let n = a.cols;

    // tambear SVD
    let tb_svd = svd(a);

    // faer SVD
    let fa = to_faer(a);
    let faer_sigma = faer_svd_sigma(&fa);

    let k = m.min(n);
    assert_eq!(tb_svd.sigma.len(), k, "{}: sigma length mismatch", label);
    assert_eq!(faer_sigma.len(), k, "{}: faer sigma length mismatch", label);

    // Compare singular values (both should be sorted descending)
    let mut max_sigma_err = 0.0_f64;
    for i in 0..k {
        let err = (tb_svd.sigma[i] - faer_sigma[i]).abs();
        let rel = if faer_sigma[i].abs() > 1e-15 {
            err / faer_sigma[i].abs()
        } else {
            err
        };
        max_sigma_err = max_sigma_err.max(rel.min(err));
    }

    assert!(
        max_sigma_err < sigma_tol,
        "{}: max sigma error vs faer = {:.2e} (tol = {:.0e})\n  tambear: {:?}\n  faer:    {:?}",
        label, max_sigma_err, sigma_tol, &tb_svd.sigma, &faer_sigma
    );

    // Reconstruction error
    let recon = reconstruct_svd(&tb_svd, m, n);
    let recon_err = diff_fro(a, &recon);
    assert!(
        recon_err < recon_tol,
        "{}: reconstruction error = {:.2e} (tol = {:.0e})", label, recon_err, recon_tol
    );
}

#[test]
fn svd_vs_faer_random_10x10() {
    let a = pseudo_random_mat(10, 10, 7, 13);
    compare_svd_with_faer(&a, "random 10x10", 1e-12, 1e-12);
}

#[test]
fn svd_vs_faer_hilbert_5x5() {
    let a = hilbert(5);
    compare_svd_with_faer(&a, "Hilbert 5x5", 1e-10, 1e-10);
}

#[test]
fn svd_vs_faer_ill_conditioned_diagonal() {
    // sigma = [1, 1e-4, 1e-8, 1e-12]
    let a = Mat::diag(&[1.0, 1e-4, 1e-8, 1e-12]);
    compare_svd_with_faer(&a, "ill-cond diagonal", 1e-10, 1e-12);
}

#[test]
fn svd_vs_faer_tall_20x5() {
    let a = pseudo_random_mat(20, 5, 11, 17);
    compare_svd_with_faer(&a, "tall 20x5", 1e-12, 1e-12);
}

#[test]
fn svd_vs_faer_wide_5x20() {
    let a = pseudo_random_mat(5, 20, 19, 23);
    compare_svd_with_faer(&a, "wide 5x20", 1e-12, 1e-12);
}

#[test]
fn svd_vs_faer_rank_deficient() {
    // 5x5 matrix of rank 2: two independent rows, rest are linear combos.
    let mut data = vec![0.0; 25];
    for j in 0..5 {
        data[0 * 5 + j] = j as f64 + 1.0;                     // row 0
        data[1 * 5 + j] = (j as f64 + 1.0) * 2.0 + 1.0;       // row 1
        data[2 * 5 + j] = data[0 * 5 + j] + data[1 * 5 + j];   // row 2 = r0 + r1
        data[3 * 5 + j] = data[0 * 5 + j] * 3.0;               // row 3 = 3*r0
        data[4 * 5 + j] = data[1 * 5 + j] * 0.5;               // row 4 = 0.5*r1
    }
    let a = Mat::from_vec(5, 5, data);

    // Both should agree on near-zero singular values
    let tb_svd = svd(&a);
    let fa = to_faer(&a);
    let faer_sigma = faer_svd_sigma(&fa);

    // First two should be significant, rest near zero
    assert!(tb_svd.sigma[0] > 1.0, "sigma[0] should be large");
    assert!(tb_svd.sigma[1] > 0.1, "sigma[1] should be nonzero");
    for i in 2..5 {
        assert!(
            tb_svd.sigma[i] < 1e-10,
            "sigma[{}] should be ~=0, got {:.2e}", i, tb_svd.sigma[i]
        );
        assert!(
            faer_sigma[i] < 1e-10,
            "faer sigma[{}] should be ~=0, got {:.2e}", i, faer_sigma[i]
        );
    }
}

#[test]
fn svd_vs_faer_various_shapes() {
    // Systematic test across shapes
    let shapes = [(3, 7), (7, 3), (1, 10), (10, 1), (4, 4), (8, 3)];
    for &(m, n) in &shapes {
        let a = pseudo_random_mat(m, n, 31, 37);
        compare_svd_with_faer(
            &a,
            &format!("shape {}x{}", m, n),
            1e-12,
            1e-12,
        );
    }
}

// =========================================================================
// Part 3: Performance Benchmarks (ignored by default)
// =========================================================================

/// Time a closure, returning elapsed in seconds.
fn timed<F: FnOnce()>(f: F) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64()
}

/// Time a closure over multiple iterations, returning median time in seconds.
fn timed_median<F: Fn()>(f: F, iters: usize) -> f64 {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        times.push(timed(|| f()));
    }
    times.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
    times[iters / 2]
}

#[test]
#[ignore]
fn bench_svd_scaling() {
    eprintln!("\n{}", SEPARATOR);
    eprintln!("SVD Performance: tambear vs faer (median of 3 runs)");
    eprintln!("{}", SEPARATOR);
    eprintln!("{:<8} {:>12} {:>12} {:>10}", "Size", "tambear(ms)", "faer(ms)", "Ratio");
    eprintln!("{}", THIN_SEP);

    let sizes = [10, 50, 100, 200];

    for &n in &sizes {
        let a = pseudo_random_mat(n, n, 7, 13);
        let fa = to_faer(&a);

        let iters = if n <= 50 { 5 } else { 3 };

        let tb_time = timed_median(|| { let _ = svd(&a); }, iters);
        let fa_time = timed_median(|| { let _ = fa.svd(); }, iters);

        let ratio = if fa_time > 0.0 { tb_time / fa_time } else { f64::NAN };
        eprintln!(
            "{:<8} {:>12.3} {:>12.3} {:>10.1}x",
            format!("{}x{}", n, n),
            tb_time * 1000.0,
            fa_time * 1000.0,
            ratio,
        );
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_svd_tall_matrices() {
    eprintln!("\n{}", SEPARATOR);
    eprintln!("SVD Performance on Tall Matrices: tambear vs faer");
    eprintln!("{}", SEPARATOR);
    eprintln!("{:<12} {:>12} {:>12} {:>10}", "Shape", "tambear(ms)", "faer(ms)", "Ratio");
    eprintln!("{}", THIN_SEP);

    let shapes = [(20, 5), (50, 10), (100, 20), (200, 50)];

    for &(m, n) in &shapes {
        let a = pseudo_random_mat(m, n, 11, 17);
        let fa = to_faer(&a);

        let iters = 3;
        let tb_time = timed_median(|| { let _ = svd(&a); }, iters);
        let fa_time = timed_median(|| { let _ = fa.svd(); }, iters);

        let ratio = if fa_time > 0.0 { tb_time / fa_time } else { f64::NAN };
        eprintln!(
            "{:<12} {:>12.3} {:>12.3} {:>10.1}x",
            format!("{}x{}", m, n),
            tb_time * 1000.0,
            fa_time * 1000.0,
            ratio,
        );
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_qr_comparison() {
    eprintln!("\n{}", SEPARATOR);
    eprintln!("QR Performance: tambear vs faer");
    eprintln!("{}", SEPARATOR);
    eprintln!("{:<8} {:>12} {:>12} {:>10}", "Size", "tambear(ms)", "faer(ms)", "Ratio");
    eprintln!("{}", THIN_SEP);

    for &n in &[100, 200] {
        let a = pseudo_random_mat(n, n, 7, 13);
        let fa = to_faer(&a);

        let iters = 3;
        let tb_time = timed_median(|| { let _ = qr(&a); }, iters);
        let fa_time = timed_median(|| { let _ = fa.qr(); }, iters);

        let ratio = if fa_time > 0.0 { tb_time / fa_time } else { f64::NAN };
        eprintln!(
            "{:<8} {:>12.3} {:>12.3} {:>10.1}x",
            format!("{}x{}", n, n),
            tb_time * 1000.0,
            fa_time * 1000.0,
            ratio,
        );
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_lu_comparison() {
    eprintln!("\n{}", SEPARATOR);
    eprintln!("LU Performance: tambear vs faer");
    eprintln!("{}", SEPARATOR);
    eprintln!("{:<8} {:>12} {:>12} {:>10}", "Size", "tambear(ms)", "faer(ms)", "Ratio");
    eprintln!("{}", THIN_SEP);

    for &n in &[100, 200] {
        let a = pseudo_random_mat(n, n, 11, 17);
        let fa = to_faer(&a);

        let iters = 3;
        let tb_time = timed_median(|| { let _ = lu(&a); }, iters);
        let fa_time = timed_median(|| { let _ = fa.partial_piv_lu(); }, iters);

        let ratio = if fa_time > 0.0 { tb_time / fa_time } else { f64::NAN };
        eprintln!(
            "{:<8} {:>12.3} {:>12.3} {:>10.1}x",
            format!("{}x{}", n, n),
            tb_time * 1000.0,
            fa_time * 1000.0,
            ratio,
        );
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_cholesky_comparison() {
    eprintln!("\n{}", SEPARATOR);
    eprintln!("Cholesky Performance: tambear vs faer");
    eprintln!("{}", SEPARATOR);
    eprintln!("{:<8} {:>12} {:>12} {:>10}", "Size", "tambear(ms)", "faer(ms)", "Ratio");
    eprintln!("{}", THIN_SEP);

    for &n in &[100, 200] {
        let base = pseudo_random_mat(n, n, 19, 23);
        let spd = make_spd(&base, 1.0);
        let fa = to_faer(&spd);

        let iters = 3;
        let tb_time = timed_median(|| { let _ = cholesky(&spd); }, iters);
        let fa_time = timed_median(|| { let _ = fa.llt(faer::Side::Lower); }, iters);

        let ratio = if fa_time > 0.0 { tb_time / fa_time } else { f64::NAN };
        eprintln!(
            "{:<8} {:>12.3} {:>12.3} {:>10.1}x",
            format!("{}x{}", n, n),
            tb_time * 1000.0,
            fa_time * 1000.0,
            ratio,
        );
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_sym_eigen_comparison() {
    eprintln!("\n{}", SEPARATOR);
    eprintln!("Symmetric Eigendecomposition: tambear vs faer");
    eprintln!("{}", SEPARATOR);
    eprintln!("{:<8} {:>12} {:>12} {:>10}", "Size", "tambear(ms)", "faer(ms)", "Ratio");
    eprintln!("{}", THIN_SEP);

    for &n in &[50, 100] {
        // Create symmetric matrix
        let base = pseudo_random_mat(n, n, 31, 37);
        let sym = make_spd(&base, 0.1);
        let fa = to_faer(&sym);

        let iters = 3;
        let tb_time = timed_median(|| { let _ = sym_eigen(&sym); }, iters);
        let fa_time = timed_median(|| { let _ = fa.self_adjoint_eigen(faer::Side::Lower); }, iters);

        let ratio = if fa_time > 0.0 { tb_time / fa_time } else { f64::NAN };
        eprintln!(
            "{:<8} {:>12.3} {:>12.3} {:>10.1}x",
            format!("{}x{}", n, n),
            tb_time * 1000.0,
            fa_time * 1000.0,
            ratio,
        );
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_summary() {
    eprintln!("\n{}", SEPARATOR);
    eprintln!("Summary: tambear is a from-scratch Jacobi/Householder implementation.");
    eprintln!("faer uses BLAS-3 blocked algorithms with SIMD. The ratio shows the");
    eprintln!("cost of correctness-first design. GPU kernels will close this gap.");
    eprintln!("{}\n", SEPARATOR);
}
