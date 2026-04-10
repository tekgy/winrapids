//! Workup: `multivariate::covariance_matrix`
//!
//! Fan-out: 8 downstream consumers (pca, factor_analysis, lda, cca, manova,
//! mahalanobis_distances, ridge, vif). Errors here propagate to all 8 silently.
//!
//! This workup asserts mathematical theorems about the covariance matrix —
//! properties that must hold regardless of implementation details. Each test
//! is a *claim about mathematical truth*, not a description of current behavior.
//!
//! Oracle values: closed-form rational (exact) or mpmath 50dp where noted.
//! All Type 1 theorem tests require no external tool — they are provable from
//! the definition `C[j,k] = Σᵢ (xᵢⱼ - μⱼ)(xᵢₖ - μₖ) / (n - ddof)`.

use tambear::multivariate::covariance_matrix;
use tambear::linear_algebra::Mat;

fn close(a: f64, b: f64, tol: f64, label: &str) {
    assert!(
        (a - b).abs() < tol,
        "{label}: got {a}, expected {b}, diff={:.2e}",
        (a - b).abs()
    );
}

// ─── Type 1: Formula correctness — mathematical necessities ──────────────────

/// Theorem: C[j,k] = C[k,j] for all j, k.
///
/// Symmetry is a direct consequence of the formula — the cross term
/// (xᵢⱼ - μⱼ)(xᵢₖ - μₖ) = (xᵢₖ - μₖ)(xᵢⱼ - μⱼ). Any implementation
/// that violates this has a bug in the off-diagonal computation.
#[test]
fn covariance_matrix_is_symmetric() {
    let x = Mat::from_rows(&[
        &[1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0],
        &[2.0, 4.0, 1.0],
        &[3.0, 1.0, 5.0],
    ]);
    let c = covariance_matrix(&x, None);
    let p = x.cols;
    for j in 0..p {
        for k in 0..p {
            assert!(
                (c.get(j, k) - c.get(k, j)).abs() < 1e-14,
                "symmetry violated: C[{j},{k}]={} != C[{k},{j}]={}",
                c.get(j, k), c.get(k, j)
            );
        }
    }
}

/// Theorem: C[j,j] = var(X[:,j], ddof).
///
/// The diagonal of the covariance matrix must equal the marginal variance
/// of each column. This follows from setting j=k in the formula.
#[test]
fn covariance_matrix_diagonal_is_variance() {
    let x = Mat::from_rows(&[
        &[2.0, 10.0],
        &[4.0, 20.0],
        &[4.0, 30.0],
        &[6.0, 40.0],
        &[4.0, 50.0],
    ]);
    let c = covariance_matrix(&x, None); // ddof=1

    // Column 0: values [2,4,4,6,4], mean=4, var=Σ(xᵢ-4)²/4 = (4+0+0+4+0)/4 = 2.0
    close(c.get(0, 0), 2.0, 1e-13, "var(col 0)");

    // Column 1: values [10,20,30,40,50], mean=30, var=Σ(xᵢ-30)²/4 = (400+100+0+100+400)/4 = 250.0
    close(c.get(1, 1), 250.0, 1e-10, "var(col 1)");
}

/// Theorem: cov(X + c, ddof) = cov(X, ddof) for any column-wise constant shift c.
///
/// Shift invariance: adding a constant to each column does not change deviations
/// from the mean, so covariance is unchanged. If this fails, the mean-subtraction
/// step is wrong (e.g. using running sum without centering).
#[test]
fn covariance_matrix_shift_invariant() {
    let x = Mat::from_rows(&[
        &[1.0, 2.0],
        &[3.0, 4.0],
        &[5.0, 6.0],
        &[7.0, 8.0],
        &[9.0, 10.0],
    ]);
    let c0 = covariance_matrix(&x, None);

    // Add large shifts to each column
    let shift0 = 1e9_f64;
    let shift1 = -5e8_f64;
    let x_shifted = Mat::from_rows(&[
        &[1.0 + shift0, 2.0 + shift1],
        &[3.0 + shift0, 4.0 + shift1],
        &[5.0 + shift0, 6.0 + shift1],
        &[7.0 + shift0, 8.0 + shift1],
        &[9.0 + shift0, 10.0 + shift1],
    ]);
    let c1 = covariance_matrix(&x_shifted, None);

    for j in 0..2 {
        for k in 0..2 {
            assert!(
                (c0.get(j, k) - c1.get(j, k)).abs() < 1e-6,
                "shift invariance violated at [{j},{k}]: C0={}, C1={} (diff={})",
                c0.get(j, k), c1.get(j, k), (c0.get(j, k) - c1.get(j, k)).abs()
            );
        }
    }
}

/// Theorem: cov(αX, ddof) = α² · cov(X, ddof) for scalar α.
///
/// Scale covariance: multiplying data by α scales the covariance by α².
/// Follows from the bilinearity of the covariance formula.
#[test]
fn covariance_matrix_scale_covariance() {
    let x = Mat::from_rows(&[
        &[1.0, 4.0],
        &[2.0, 5.0],
        &[3.0, 6.0],
        &[4.0, 7.0],
        &[5.0, 8.0],
    ]);
    let alpha = 3.7_f64;
    let x_scaled = Mat::from_rows(&[
        &[alpha * 1.0, alpha * 4.0],
        &[alpha * 2.0, alpha * 5.0],
        &[alpha * 3.0, alpha * 6.0],
        &[alpha * 4.0, alpha * 7.0],
        &[alpha * 5.0, alpha * 8.0],
    ]);
    let c0 = covariance_matrix(&x, None);
    let c1 = covariance_matrix(&x_scaled, None);
    let alpha2 = alpha * alpha;

    for j in 0..2 {
        for k in 0..2 {
            let expected = alpha2 * c0.get(j, k);
            assert!(
                (c1.get(j, k) - expected).abs() < 1e-8,
                "scale covariance [{j},{k}]: C(αX)={}, α²C(X)={} (diff={})",
                c1.get(j, k), expected, (c1.get(j, k) - expected).abs()
            );
        }
    }
}

/// Theorem: cov(X, ddof=0)[j,k] = cov(X, ddof=1)[j,k] · (n-1)/n.
///
/// The Bessel correction factor relates population and sample covariance.
/// Any implementation where the two don't satisfy this exact ratio has
/// a bug in the denominator computation.
#[test]
fn covariance_matrix_bessel_correction_ratio() {
    let x = Mat::from_rows(&[
        &[2.0, 5.0],
        &[4.0, 3.0],
        &[6.0, 7.0],
        &[8.0, 1.0],
        &[10.0, 9.0],
    ]);
    let n = x.rows as f64;
    let c_pop = covariance_matrix(&x, Some(0));  // population: divide by n
    let c_samp = covariance_matrix(&x, Some(1)); // sample: divide by n-1

    for j in 0..2 {
        for k in 0..2 {
            let expected = c_samp.get(j, k) * (n - 1.0) / n;
            assert!(
                (c_pop.get(j, k) - expected).abs() < 1e-13,
                "Bessel ratio [{j},{k}]: C_pop={}, C_samp*(n-1)/n={} (diff={})",
                c_pop.get(j, k), expected, (c_pop.get(j, k) - expected).abs()
            );
        }
    }
}

// ─── Type 2: Oracle cases — known exact values ────────────────────────────────

/// Oracle case: 2×2 dataset with exact rational covariance.
///
/// x = [[1,2],[3,4],[5,6]], n=3, ddof=1
/// means = [3, 4]
/// C[0,0] = ((1-3)²+(3-3)²+(5-3)²)/2 = (4+0+4)/2 = 4.0  (exact)
/// C[1,1] = ((2-4)²+(4-4)²+(6-4)²)/2 = (4+0+4)/2 = 4.0  (exact)
/// C[0,1] = ((1-3)(2-4)+(3-3)(4-4)+(5-3)(6-4))/2 = (4+0+4)/2 = 4.0  (exact)
///
/// Perfect linear relationship → off-diagonal equals diagonal.
#[test]
fn covariance_matrix_exact_linear_data() {
    let x = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
    let c = covariance_matrix(&x, None);

    close(c.get(0, 0), 4.0, 1e-13, "C[0,0]");
    close(c.get(1, 1), 4.0, 1e-13, "C[1,1]");
    close(c.get(0, 1), 4.0, 1e-13, "C[0,1]");
    close(c.get(1, 0), 4.0, 1e-13, "C[1,0]");
}

/// Oracle case: uncorrelated columns → zero off-diagonal.
///
/// x = [[1,1],[2,-1],[3,1],[4,-1],[5,1],[6,-1]], n=6
/// col0 = [1,2,3,4,5,6], mean=3.5
/// col1 = [1,-1,1,-1,1,-1], mean=0
///
/// Cross product: Σ(xᵢ₀ - 3.5)(xᵢ₁ - 0) for i=1..6
/// = (-2.5)(1)+(-1.5)(-1)+(-0.5)(1)+(0.5)(-1)+(1.5)(1)+(2.5)(-1)
/// = -2.5 + 1.5 - 0.5 - 0.5 + 1.5 - 2.5 = -3.0
///
/// Note: [1,2,3,4] with alternating signs does NOT give zero cross product.
/// Need symmetric deviations: equal-magnitude +/- deviations from mean.
/// Here col0 deviations [-2.5,-1.5,-0.5,+0.5,+1.5,+2.5] perfectly paired
/// with alternating signs [-,+,-,+,-,+] and sign-flipped: (-2.5)(-1)...
/// Wait — still not zero. Need to verify by construction.
///
/// Correct construction: col0 = [1,-1,1,-1], col1 = [1,1,-1,-1] (orthogonal sign patterns).
/// col0 mean=0, col1 mean=0.
/// Cross: (1)(1)+(-1)(1)+(1)(-1)+(-1)(-1) = 1-1-1+1 = 0. Exact zero.
#[test]
fn covariance_matrix_uncorrelated_is_zero_off_diagonal() {
    // col0 = [1,-1,1,-1], col1 = [1,1,-1,-1]
    // Both have mean=0. Cross products sum to exactly 0.
    // Σ(xᵢ₀)(xᵢ₁) = (1)(1)+(-1)(1)+(1)(-1)+(-1)(-1) = 1-1-1+1 = 0
    let x = Mat::from_rows(&[
        &[ 1.0,  1.0],
        &[-1.0,  1.0],
        &[ 1.0, -1.0],
        &[-1.0, -1.0],
    ]);
    let c = covariance_matrix(&x, None);
    assert!(
        c.get(0, 1).abs() < 1e-13,
        "orthogonal columns off-diagonal should be 0, got {}", c.get(0, 1)
    );
    assert!(
        c.get(1, 0).abs() < 1e-13,
        "symmetry: off-diagonal should be 0, got {}", c.get(1, 0)
    );
    // Also verify diagonal: var = Σxᵢ²/(n-1) = 4/3
    close(c.get(0, 0), 4.0/3.0, 1e-13, "var(col0)");
    close(c.get(1, 1), 4.0/3.0, 1e-13, "var(col1)");
}

// ─── Type 3: Robustness — degenerate inputs ───────────────────────────────────

/// Degenerate: all rows identical → covariance is zero matrix.
///
/// If all observations are the same point, every deviation from the mean is 0.
/// Covariance must be the zero matrix (not NaN, not garbage).
#[test]
fn covariance_matrix_constant_data_is_zero() {
    let x = Mat::from_rows(&[
        &[3.0, 7.0],
        &[3.0, 7.0],
        &[3.0, 7.0],
        &[3.0, 7.0],
    ]);
    let c = covariance_matrix(&x, None);
    for j in 0..2 {
        for k in 0..2 {
            assert!(
                c.get(j, k).abs() < 1e-14,
                "constant data: C[{j},{k}]={} should be 0", c.get(j, k)
            );
        }
    }
}

/// Degenerate: single row (n=1, ddof=1) → denominator = 0, result finite or NaN.
///
/// With n=1 and ddof=1, the denominator is 0. The implementation uses
/// `.saturating_sub(ddof).max(1)` to prevent divide-by-zero, producing
/// finite (not NaN) output. This is a design decision, not a theorem —
/// documenting the actual behavior so it doesn't silently change.
#[test]
fn covariance_matrix_single_row_does_not_panic() {
    let x = Mat::from_rows(&[&[1.0, 2.0, 3.0]]);
    let c = covariance_matrix(&x, None);
    // Should return a zero matrix (zero numerator / 1 via max(1))
    // NOT panic, NOT NaN — the saturating_sub protects the denominator.
    for j in 0..3 {
        for k in 0..3 {
            assert!(
                c.get(j, k).is_finite(),
                "single row: C[{j},{k}]={} should be finite", c.get(j, k)
            );
        }
    }
}

/// Degenerate: NaN in input data → NaN propagates to output.
///
/// If the input contains NaN, the covariance values that depend on it
/// should be NaN, not silently 0 or a finite value hiding the defect.
#[test]
fn covariance_matrix_nan_input_propagates() {
    let x = Mat::from_rows(&[
        &[1.0,      2.0],
        &[f64::NAN, 4.0],
        &[5.0,      6.0],
        &[7.0,      8.0],
    ]);
    let c = covariance_matrix(&x, None);
    // Column 0 contains NaN, so C[0,0] and C[0,1] should be NaN.
    assert!(
        c.get(0, 0).is_nan(),
        "NaN in col 0: C[0,0]={} should be NaN", c.get(0, 0)
    );
    assert!(
        c.get(0, 1).is_nan() || c.get(1, 0).is_nan(),
        "NaN in col 0: cross-term should be NaN, got C[0,1]={}", c.get(0, 1)
    );
}
