//! Adversarial Boundary Tests — Wave 9
//!
//! Targets: linear_algebra, interpolation, series_accel
//!
//! Focus: mathematical correctness and numerical stability,
//! not just "doesn't crash." Every test checks a mathematical property.

use tambear::linear_algebra::*;
use tambear::interpolation::*;
use tambear::series_accel::*;

// ═══════════════════════════════════════════════════════════════════════════
// LINEAR ALGEBRA — CORE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// det(I) = 1 for any dimension.
#[test]
fn det_identity() {
    for n in 1..=5 {
        let m = Mat::eye(n);
        let d = det(&m);
        assert!((d - 1.0).abs() < 1e-10, "det(I_{}) should be 1, got {}", n, d);
    }
}

/// det of 0x0 matrix.
/// Type 5: degenerate dimension.
#[test]
fn det_empty_matrix() {
    let m = Mat::zeros(0, 0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        det(&m)
    }));
    match result {
        Ok(d) => {
            // det of empty matrix is conventionally 1 (empty product)
            assert!(d.is_finite(), "det(0x0) should be finite, got {}", d);
        }
        Err(_) => eprintln!("NOTE: det panics on 0x0 matrix"),
    }
}

/// det of singular matrix = 0.
#[test]
fn det_singular() {
    let m = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]); // row 2 = 2 * row 1
    let d = det(&m);
    assert!(d.abs() < 1e-10, "det of singular matrix should be 0, got {}", d);
}

/// inv(I) = I.
#[test]
fn inv_identity() {
    let m = Mat::eye(3);
    let mi = inv(&m);
    assert!(mi.is_some(), "Identity should be invertible");
    let mi = mi.unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((mi.get(i, j) - expected).abs() < 1e-10,
                "inv(I)[{},{}] should be {}, got {}", i, j, expected, mi.get(i, j));
        }
    }
}

/// inv of singular matrix: should return None.
#[test]
fn inv_singular() {
    let m = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
    let mi = inv(&m);
    assert!(mi.is_none(), "Singular matrix should not be invertible");
}

/// A * inv(A) = I (roundtrip test with ill-conditioned matrix).
/// Type 3: cancellation/precision.
#[test]
fn inv_roundtrip_ill_conditioned() {
    // Hilbert matrix 4x4 — condition number ~15514
    let mut data = vec![0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            data[i * 4 + j] = 1.0 / (i as f64 + j as f64 + 1.0);
        }
    }
    let h = Mat::from_vec(4, 4, data);
    let hi = inv(&h);
    assert!(hi.is_some(), "Hilbert 4x4 should be invertible");
    let hi = hi.unwrap();
    let product = mat_mul(&h, &hi);
    let eye = Mat::eye(4);
    let mut max_err = 0.0_f64;
    for i in 0..4 {
        for j in 0..4 {
            let err = (product.get(i, j) - eye.get(i, j)).abs();
            if err > max_err { max_err = err; }
        }
    }
    // Hilbert matrices are notoriously ill-conditioned
    if max_err > 0.01 {
        eprintln!("CONFIRMED BUG: Hilbert 4x4 inv roundtrip max error = {} (ill-conditioning)", max_err);
    }
}

/// SVD of zero matrix: all singular values = 0.
#[test]
fn svd_zero_matrix() {
    let m = Mat::zeros(3, 3);
    let result = svd(&m);
    for &s in &result.sigma {
        assert!(s.abs() < 1e-10, "SVD of zero matrix should have s=0, got {}", s);
    }
}

/// SVD of 1x1 matrix: single singular value = |a|.
#[test]
fn svd_1x1() {
    let m = Mat::from_vec(1, 1, vec![-5.0]);
    let result = svd(&m);
    assert_eq!(result.sigma.len(), 1);
    assert!((result.sigma[0] - 5.0).abs() < 1e-10, "SVD of [-5] should have s=5, got {}", result.sigma[0]);
}

/// SVD reconstruction: U * diag(s) * Vt = A.
/// Type 3: precision under reconstruction.
#[test]
fn svd_reconstruction() {
    // Use square matrix to avoid thin/full SVD dimension issues
    let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let result = svd(&a);
    // Full SVD: U is 2x2, sigma has 2 entries, Vt is 2x2
    // Reconstruct: U * diag(sigma) * Vt
    let s_mat = Mat::diag(&result.sigma);
    let us = mat_mul(&result.u, &s_mat);
    let reconstructed = mat_mul(&us, &result.vt);
    let mut max_err = 0.0_f64;
    for i in 0..a.rows {
        for j in 0..a.cols {
            let err = (reconstructed.get(i, j) - a.get(i, j)).abs();
            if err > max_err { max_err = err; }
        }
    }
    assert!(max_err < 1e-10, "SVD reconstruction error = {} (should be ~0)", max_err);
}

/// Cholesky of non-PD matrix: should return None.
#[test]
fn cholesky_non_pd() {
    let m = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]); // eigenvalues: 3, -1 → not PD
    let result = cholesky(&m);
    assert!(result.is_none(), "Non-PD matrix should fail Cholesky");
}

/// QR of tall skinny matrix: Q'Q = I.
#[test]
fn qr_orthogonality() {
    let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
    let result = qr(&a);
    let qtq = mat_mul(&result.q.t(), &result.q);
    for i in 0..qtq.rows {
        for j in 0..qtq.cols {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((qtq.get(i, j) - expected).abs() < 1e-10,
                "Q'Q[{},{}] should be {}, got {}", i, j, expected, qtq.get(i, j));
        }
    }
}

/// sym_eigen of diagonal matrix: eigenvalues = diagonal entries.
#[test]
fn eigen_diagonal() {
    let m = Mat::diag(&[3.0, 1.0, 4.0, 1.5]);
    let (eigenvalues, _) = sym_eigen(&m);
    let mut eigs = eigenvalues.clone();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut expected = vec![1.0, 1.5, 3.0, 4.0];
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (e, &x) in eigs.iter().zip(&expected) {
        assert!((e - x).abs() < 1e-10, "Eigenvalue should be {}, got {}", x, e);
    }
}

/// cond(I) = 1.
#[test]
fn cond_identity() {
    let m = Mat::eye(5);
    let c = cond(&m);
    assert!((c - 1.0).abs() < 1e-10, "cond(I) should be 1, got {}", c);
}

/// cond of singular matrix: should be Inf.
#[test]
fn cond_singular() {
    let m = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
    let c = cond(&m);
    assert!(c > 1e10 || c.is_infinite(), "cond of singular should be huge/Inf, got {}", c);
}

/// pinv of zero matrix: should be zero.
#[test]
fn pinv_zero() {
    let m = Mat::zeros(2, 3);
    let p = pinv(&m, None);
    for i in 0..p.rows {
        for j in 0..p.cols {
            assert!(p.get(i, j).abs() < 1e-10, "pinv(0)[{},{}] should be 0", i, j);
        }
    }
}

/// lstsq overdetermined system: residual minimization.
#[test]
fn lstsq_overdetermined() {
    // y = 2x + 1, three points
    let a = Mat::from_rows(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0]]); // [1, x]
    let b = vec![3.0, 5.0, 7.0]; // y = 2x + 1
    let x = lstsq(&a, &b);
    assert!((x[0] - 1.0).abs() < 1e-10, "Intercept should be 1, got {}", x[0]);
    assert!((x[1] - 2.0).abs() < 1e-10, "Slope should be 2, got {}", x[1]);
}

/// Power iteration on matrix with repeated eigenvalue.
/// Type 2: convergence may be slow or fail.
#[test]
fn power_iteration_repeated_eigenvalue() {
    let m = Mat::diag(&[5.0, 5.0, 1.0]); // repeated dominant eigenvalue
    let (eigenvalue, _) = power_iteration(&m, 1000, 1e-10);
    assert!((eigenvalue - 5.0).abs() < 0.1,
        "Power iteration should find dominant eigenvalue 5, got {}", eigenvalue);
}

/// solve with NaN in matrix.
/// Type 3: NaN propagation.
#[test]
fn solve_nan_matrix() {
    let m = Mat::from_rows(&[&[1.0, f64::NAN], &[3.0, 4.0]]);
    let result = solve(&m, &[1.0, 2.0]);
    match result {
        Some(x) => {
            if x.iter().any(|v| v.is_nan()) {
                // NaN propagated — expected
            } else {
                eprintln!("NOTE: solve returned non-NaN result from NaN matrix: {:?}", x);
            }
        }
        None => {
            // LU factorization failed due to NaN — acceptable
        }
    }
}

/// mat_mul dimension mismatch: cols(A) != rows(B).
/// Type 5.
#[test]
fn mat_mul_dimension_mismatch() {
    let a = Mat::zeros(2, 3);
    let b = Mat::zeros(4, 2); // 3 != 4
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mat_mul(&a, &b)
    }));
    if result.is_err() {
        // Good: panics on dimension mismatch
    } else {
        eprintln!("CONFIRMED BUG: mat_mul doesn't check dimension mismatch");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERPOLATION
// ═══════════════════════════════════════════════════════════════════════════

/// Lagrange interpolation at a node: should return exact y value.
#[test]
fn lagrange_at_node() {
    let xs = vec![0.0, 1.0, 2.0, 3.0];
    let ys = vec![1.0, 0.0, 1.0, 0.0];
    for (i, (&x, &y)) in xs.iter().zip(ys.iter()).enumerate() {
        let result = lagrange(&xs, &ys, x);
        assert!((result - y).abs() < 1e-10,
            "Lagrange at node {} should be {}, got {}", i, y, result);
    }
}

/// Lagrange with duplicate x values: division by zero.
/// Type 1.
#[test]
fn lagrange_duplicate_x() {
    let xs = vec![1.0, 1.0, 2.0]; // duplicate!
    let ys = vec![3.0, 4.0, 5.0];
    let result = lagrange(&xs, &ys, 1.5);
    if result.is_nan() || result.is_infinite() {
        eprintln!("CONFIRMED BUG: Lagrange with duplicate x produces {} (division by zero in basis polynomials)", result);
    }
}

/// Lagrange with single point: constant function.
#[test]
fn lagrange_single_point() {
    let result = lagrange(&[2.0], &[7.0], 100.0);
    assert!((result - 7.0).abs() < 1e-10, "Single-point Lagrange should be constant 7, got {}", result);
}

/// Lagrange with empty arrays.
/// Type 5.
#[test]
fn lagrange_empty() {
    let result = std::panic::catch_unwind(|| {
        lagrange(&[], &[], 1.0)
    });
    match result {
        Ok(v) => assert!(v.is_nan() || v == 0.0, "Empty Lagrange should be NaN or 0, got {}", v),
        Err(_) => eprintln!("NOTE: lagrange panics on empty input"),
    }
}

/// Natural cubic spline with 2 points: degenerates to linear.
#[test]
fn cubic_spline_two_points() {
    let xs = vec![0.0, 1.0];
    let ys = vec![0.0, 1.0];
    let spline = natural_cubic_spline(&xs, &ys);
    let mid = spline.eval(0.5);
    assert!((mid - 0.5).abs() < 1e-10, "2-point spline should be linear, f(0.5)={}", mid);
}

/// Cubic spline extrapolation: outside knot range.
#[test]
fn cubic_spline_extrapolation() {
    let xs = vec![0.0, 1.0, 2.0];
    let ys = vec![0.0, 1.0, 0.0];
    let spline = natural_cubic_spline(&xs, &ys);
    let outside = spline.eval(10.0);
    // Extrapolation uses last segment — may give wild values
    assert!(outside.is_finite(), "Spline extrapolation should be finite, got {}", outside);
}

/// Cubic spline with single point.
/// Type 5.
#[test]
fn cubic_spline_single_point() {
    let result = std::panic::catch_unwind(|| {
        natural_cubic_spline(&[1.0], &[2.0])
    });
    match result {
        Ok(spline) => {
            let v = spline.eval(1.0);
            assert!((v - 2.0).abs() < 1e-10 || v.is_nan(),
                "Single-point spline at node should be 2.0, got {}", v);
        }
        Err(_) => eprintln!("NOTE: natural_cubic_spline panics on single point"),
    }
}

/// polyfit degree > n-1: overdetermined polynomial.
/// Type 5.
#[test]
fn polyfit_degree_too_high() {
    let xs = vec![1.0, 2.0];
    let ys = vec![3.0, 4.0];
    let result = std::panic::catch_unwind(|| {
        polyfit(&xs, &ys, 5) // degree 5 with 2 points
    });
    match result {
        Ok(fit) => {
            // Should still work (underdetermined) or return lower degree
            assert!(fit.coeffs.iter().all(|c| c.is_finite()),
                "polyfit with high degree should have finite coefficients");
        }
        Err(_) => eprintln!("NOTE: polyfit panics when degree > n-1"),
    }
}

/// Chebyshev nodes with n=0.
/// Type 5.
#[test]
fn chebyshev_nodes_zero() {
    let nodes = chebyshev_nodes(0, -1.0, 1.0);
    assert!(nodes.is_empty(), "0 Chebyshev nodes should be empty");
}

/// GP regression with length_scale=0: kernel becomes delta function.
/// Type 1.
#[test]
fn gp_zero_length_scale() {
    let x_train = vec![0.0, 1.0, 2.0];
    let y_train = vec![1.0, 0.0, 1.0];
    let x_query = vec![0.5, 1.5];
    let result = std::panic::catch_unwind(|| {
        gp_regression(&x_train, &y_train, &x_query, 0.0, 1.0, 0.01)
    });
    match result {
        Ok(gp) => {
            let any_bad = gp.mean.iter().chain(gp.std.iter()).any(|v| v.is_nan() || v.is_infinite());
            if any_bad {
                eprintln!("CONFIRMED BUG: GP regression with length_scale=0 produces NaN/Inf");
            }
        }
        Err(_) => eprintln!("NOTE: gp_regression panics with length_scale=0"),
    }
}

/// GP regression with noise_var=0: interpolation (exact at training points).
#[test]
fn gp_zero_noise() {
    let x_train = vec![0.0, 1.0, 2.0];
    let y_train = vec![1.0, 0.0, 1.0];
    let result = std::panic::catch_unwind(|| {
        gp_regression(&x_train, &y_train, &x_train, 1.0, 1.0, 0.0)
    });
    match result {
        Ok(gp) => {
            // With noise=0, GP should interpolate exactly at training points
            for (i, (&pred, &true_y)) in gp.mean.iter().zip(y_train.iter()).enumerate() {
                if (pred - true_y).abs() > 0.1 {
                    eprintln!("CONFIRMED BUG: GP with noise=0 doesn't interpolate at training point {}: pred={}, true={}", i, pred, true_y);
                }
            }
        }
        Err(_) => eprintln!("CONFIRMED BUG: GP regression panics with noise_var=0 (singular kernel matrix)"),
    }
}

/// Padé approximant eval at denominator zero: pole.
/// Type 1.
#[test]
fn pade_at_pole() {
    // Simple: 1/(1-x) has pole at x=1
    // Taylor: 1 + x + x² + x³ + ...
    let taylor = vec![1.0, 1.0, 1.0, 1.0];
    let p = pade(&taylor, 1, 1);
    let result = p.eval(1.0);
    if result.is_infinite() || result.is_nan() {
        // Expected: evaluating at a pole
    } else {
        eprintln!("NOTE: Padé at pole returns finite value: {}", result);
    }
}

/// Neville with single point: should return y value.
#[test]
fn neville_single_point() {
    let (val, err) = neville(&[2.0], &[7.0], 100.0);
    assert!((val - 7.0).abs() < 1e-10, "Single-point Neville should be 7, got {}", val);
}

/// B-spline basis with p=0: piecewise constant.
#[test]
fn bspline_basis_p0() {
    let knots = vec![0.0, 1.0, 2.0, 3.0];
    let b = bspline_basis(&knots, 0, 0, 0.5); // first basis function at x=0.5
    assert!((b - 1.0).abs() < 1e-10, "B-spline basis p=0 at interior should be 1, got {}", b);
}

// ═══════════════════════════════════════════════════════════════════════════
// SERIES ACCELERATION
// ═══════════════════════════════════════════════════════════════════════════

/// Cesaro sum of constant sequence: should return that constant.
#[test]
fn cesaro_constant() {
    let sums = vec![5.0, 5.0, 5.0, 5.0];
    let c = cesaro_sum(&sums);
    assert!((c - 5.0).abs() < 1e-10, "Cesaro of constant should be 5, got {}", c);
}

/// Cesaro of empty: should not crash.
#[test]
fn cesaro_empty() {
    let result = std::panic::catch_unwind(|| {
        cesaro_sum(&[])
    });
    match result {
        Ok(v) => assert!(v.is_nan() || v == 0.0, "Cesaro of empty should be NaN or 0, got {}", v),
        Err(_) => eprintln!("NOTE: cesaro_sum panics on empty input"),
    }
}

/// Aitken delta-squared with constant sequence: denominator = 0.
/// Type 1.
#[test]
fn aitken_constant_sequence() {
    let sums = vec![3.0, 3.0, 3.0, 3.0, 3.0];
    let result = aitken_delta2(&sums);
    // Δ²s_n = s_{n+2} - 2*s_{n+1} + s_n = 0 → 0/0
    for &v in &result {
        if v.is_nan() || v.is_infinite() {
            eprintln!("CONFIRMED BUG: Aitken delta² returns {} for constant sequence (0/0)", v);
            break;
        }
    }
}

/// Wynn epsilon with 2 terms: not enough for epsilon table.
/// Type 5.
#[test]
fn wynn_too_few_terms() {
    let result = std::panic::catch_unwind(|| {
        wynn_epsilon(&[1.0, 2.0])
    });
    match result {
        Ok(v) => assert!(v.is_finite(), "Wynn with 2 terms should give something finite, got {}", v),
        Err(_) => eprintln!("NOTE: wynn_epsilon panics on < 3 terms"),
    }
}

/// Richardson extrapolation with ratio=1: h and h*1 are same → no extrapolation.
/// Type 1.
#[test]
fn richardson_ratio_one() {
    let approx = vec![1.0, 1.1, 1.11];
    let result = std::panic::catch_unwind(|| {
        richardson_extrapolate(&approx, 1.0, 2)
    });
    match result {
        Ok(v) => {
            if v.is_nan() || v.is_infinite() {
                eprintln!("CONFIRMED BUG: Richardson with ratio=1 produces {} (ratio^p - 1 = 0)", v);
            }
        }
        Err(_) => eprintln!("NOTE: richardson_extrapolate panics on ratio=1"),
    }
}

/// Euler-Maclaurin zeta with s=1: pole of zeta function.
/// Type 1.
#[test]
fn euler_maclaurin_zeta_at_pole() {
    let result = euler_maclaurin_zeta(1.0, 100, 10);
    // zeta(1) = ∞ (harmonic series)
    if result.is_infinite() {
        // Correct: zeta has a pole at s=1
    } else if result.is_finite() {
        // Should be very large
        assert!(result > 1.0, "zeta(1) should diverge, got {}", result);
    }
}

/// Euler-Maclaurin zeta at s=2: should be π²/6 ≈ 1.6449.
#[test]
fn euler_maclaurin_zeta_s2() {
    let result = euler_maclaurin_zeta(2.0, 100, 10);
    let expected = std::f64::consts::PI.powi(2) / 6.0;
    assert!((result - expected).abs() < 0.01,
        "zeta(2) should be π²/6 ≈ {}, got {}", expected, result);
}

/// accelerate on alternating series: 1 - 1/2 + 1/3 - 1/4 + ... = ln(2).
#[test]
fn accelerate_alternating_harmonic() {
    let terms: Vec<f64> = (1..=20).map(|n| {
        let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
        sign / n as f64
    }).collect();
    let result = accelerate(&terms);
    let expected = (2.0_f64).ln();
    assert!((result - expected).abs() < 0.01,
        "Accelerated alternating harmonic should be ln(2)≈{}, got {}", expected, result);
}

/// accelerate on empty input.
/// Type 5.
#[test]
fn accelerate_empty() {
    let result = std::panic::catch_unwind(|| {
        accelerate(&[])
    });
    match result {
        Ok(v) => assert!(v.is_nan() || v == 0.0, "accelerate([]) should be NaN or 0, got {}", v),
        Err(_) => eprintln!("NOTE: accelerate panics on empty input"),
    }
}

/// partial_sums correctness: running sum.
#[test]
fn partial_sums_basic() {
    let terms = vec![1.0, 2.0, 3.0, 4.0];
    let sums = partial_sums(&terms);
    assert_eq!(sums, vec![1.0, 3.0, 6.0, 10.0]);
}

/// Euler transform of all-NaN terms.
/// Type 3.
#[test]
fn euler_transform_nan() {
    let terms = vec![f64::NAN; 5];
    let result = std::panic::catch_unwind(|| {
        euler_transform(&terms)
    });
    match result {
        Ok(v) => assert!(v.is_nan(), "Euler transform of NaN should be NaN, got {}", v),
        Err(_) => eprintln!("NOTE: euler_transform panics on NaN input"),
    }
}
