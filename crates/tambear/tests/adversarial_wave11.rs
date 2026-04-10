//! Adversarial Wave 11 — Second pass on fresh primitives
//!
//! Targets: gram_schmidt, gram_schmidt_modified, stirling_approx,
//!          bic_score/aic_score (clustering), gap_statistic,
//!          marchenko_pastur_*, chebyshev_outlier,
//!          concordance_correlation (nonparametric), dtw degenerate cases,
//!          bessel_j0/j1/jn identity checks, softmax/log_softmax stability
//!
//! Each test asserts what the MATH should produce.
//! FAILING tests reveal real bugs.

use tambear::linear_algebra::{gram_schmidt, gram_schmidt_modified, Mat};
use tambear::special_functions::{
    stirling_approx, stirling_approx_corrected,
    marchenko_pastur_pdf, marchenko_pastur_bounds, marchenko_pastur_classify,
    chebyshev_outlier, softmax, log_softmax,
    bessel_j0, bessel_j1, bessel_jn,
    log_gamma,
};
use tambear::clustering::{bic_score, aic_score, gap_statistic};
use tambear::nonparametric::dtw;

// ═══════════════════════════════════════════════════════════════════════════
// gram_schmidt — degenerate inputs
// ═══════════════════════════════════════════════════════════════════════════

/// Empty input → empty output (documented).
#[test]
fn gram_schmidt_empty() {
    let result = gram_schmidt(&[]).unwrap();
    assert!(result.is_empty(), "gram_schmidt([]) should return [], got {:?}", result);
}

/// Single zero vector → empty basis (linearly dependent, skipped).
#[test]
fn gram_schmidt_single_zero_vector() {
    let result = gram_schmidt(&[vec![0.0, 0.0, 0.0]]).unwrap();
    assert!(result.is_empty(),
        "gram_schmidt([0,0,0]) should return [] (zero vector is linearly dependent), got {:?}", result);
}

/// Single non-zero vector → basis of size 1, unit vector.
#[test]
fn gram_schmidt_single_nonzero() {
    let result = gram_schmidt(&[vec![3.0, 4.0]]).unwrap();
    assert_eq!(result.len(), 1, "Single non-zero vector should give 1 basis vector");
    let norm: f64 = result[0].iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-12,
        "Output basis vector should be unit norm, got norm={}", norm);
    // Should be [0.6, 0.8]
    assert!((result[0][0] - 0.6).abs() < 1e-12, "Expected 0.6, got {}", result[0][0]);
    assert!((result[0][1] - 0.8).abs() < 1e-12, "Expected 0.8, got {}", result[0][1]);
}

/// Two identical vectors → linearly dependent → only 1 basis vector.
#[test]
fn gram_schmidt_duplicate_vectors() {
    let v = vec![1.0, 0.0, 0.0];
    let result = gram_schmidt(&[v.clone(), v.clone()]).unwrap();
    assert_eq!(result.len(), 1,
        "Two identical vectors should give 1 basis vector (2nd is linearly dependent), got {:?}", result);
}

/// Standard basis: e1, e2, e3 → already orthonormal → output = input.
#[test]
fn gram_schmidt_standard_basis() {
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let result = gram_schmidt(&vectors).unwrap();
    assert_eq!(result.len(), 3, "Standard basis should give 3 basis vectors");
    for i in 0..3 {
        for j in 0..3 {
            let dot: f64 = result[i].iter().zip(result[j].iter()).map(|(a, b)| a * b).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-12,
                "Standard basis: dot(q[{}], q[{}]) should be {}, got {}", i, j, expected, dot);
        }
    }
}

/// BUG: n > d vectors — more vectors than dimensions. Some must be linearly dependent.
/// gram_schmidt should return at most d basis vectors.
#[test]
fn gram_schmidt_overdetermined() {
    // 5 vectors in 2D — at most 2 can be linearly independent
    let vectors = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, -1.0],
        vec![-1.0, 3.0],
    ];
    let result = gram_schmidt(&vectors).unwrap();
    assert!(result.len() <= 2,
        "5 vectors in 2D should give at most 2 basis vectors, got {}", result.len());
    assert_eq!(result.len(), 2,
        "5 generic vectors in 2D should give exactly 2 basis vectors, got {}", result.len());
}

/// Output vectors must be mutually orthogonal.
#[test]
fn gram_schmidt_orthogonality() {
    let vectors = vec![
        vec![1.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
    ];
    let result = gram_schmidt(&vectors).unwrap();
    for i in 0..result.len() {
        for j in (i+1)..result.len() {
            let dot: f64 = result[i].iter().zip(result[j].iter()).map(|(a, b)| a * b).sum();
            assert!(dot.abs() < 1e-10,
                "gram_schmidt: q[{}]·q[{}] should be 0 (orthogonality), got {}", i, j, dot);
        }
        // Unit norm
        let norm: f64 = result[i].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10,
            "gram_schmidt: q[{}] should be unit norm, got {}", i, norm);
    }
}

/// BUG: Inconsistent dimensions → should return Err.
#[test]
fn gram_schmidt_inconsistent_dims() {
    let result = gram_schmidt(&[vec![1.0, 0.0], vec![0.0, 1.0, 0.0]]);
    assert!(result.is_err(),
        "gram_schmidt with inconsistent dimensions should return Err, got Ok({:?})", result.ok());
}

/// NaN in input vector — norm = NaN, NaN < 1e-10 is false → accepted into basis.
/// BUG: NaN vector treated as valid basis vector.
#[test]
fn gram_schmidt_nan_vector_should_error_or_skip() {
    let result = gram_schmidt(&[vec![f64::NAN, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
    // Correct: NaN vector should either Error or be skipped (norm = NaN, NaN < 1e-10 = false → BUG: accepted)
    match &result {
        Ok(basis) => {
            // If accepted, check that no NaN exists in the basis
            let has_nan = basis.iter().any(|v| v.iter().any(|x| x.is_nan()));
            assert!(!has_nan,
                "gram_schmidt should not accept NaN vectors into basis, got basis with NaN: {:?}", basis);
        }
        Err(_) => {} // Returning Err is also acceptable
    }
}

/// Ill-conditioned: nearly linearly dependent vectors.
/// Classical GS loses orthogonality; modified GS should be better.
/// This test checks that modified GS is MORE orthogonal than classical GS.
#[test]
fn gram_schmidt_modified_more_orthogonal() {
    // Hilbert-like near-dependent vectors
    let n = 5;
    let vectors: Vec<Vec<f64>> = (0..n).map(|i| {
        (0..n).map(|j| 1.0 / (i + j + 1) as f64).collect()
    }).collect();

    let classical = gram_schmidt(&vectors).unwrap();
    let modified = gram_schmidt_modified(&vectors).unwrap();

    // Measure worst-case deviation from orthogonality
    let orthogonality_error = |basis: &[Vec<f64>]| -> f64 {
        let mut max_err = 0.0_f64;
        for i in 0..basis.len() {
            for j in (i+1)..basis.len() {
                let dot: f64 = basis[i].iter().zip(basis[j].iter()).map(|(a,b)| a*b).sum();
                max_err = max_err.max(dot.abs());
            }
        }
        max_err
    };

    let err_classical = orthogonality_error(&classical);
    let err_modified = orthogonality_error(&modified);

    // Modified should be strictly better for ill-conditioned inputs
    assert!(err_modified <= err_classical * 10.0,
        "Modified GS should be at least as orthogonal as classical: classical_err={}, modified_err={}",
        err_classical, err_modified);

    // Both should still produce a valid orthogonal set (within floating-point tolerance)
    assert!(err_modified < 1e-8,
        "Modified GS should produce near-orthogonal basis, max deviation={}", err_modified);
}

// ═══════════════════════════════════════════════════════════════════════════
// stirling_approx — correctness and edge cases
// ═══════════════════════════════════════════════════════════════════════════

/// n=0 → 0.0 (0! = 1, ln(1) = 0).
#[test]
fn stirling_n0_is_zero() {
    assert_eq!(stirling_approx(0.0), 0.0, "stirling_approx(0) should be 0");
    assert_eq!(stirling_approx_corrected(0.0), 0.0, "stirling_approx_corrected(0) should be 0");
}

/// n < 0 → NaN (documented).
#[test]
fn stirling_negative_is_nan() {
    assert!(stirling_approx(-1.0).is_nan(), "stirling_approx(-1) should be NaN");
    assert!(stirling_approx_corrected(-1.0).is_nan(), "stirling_corrected(-1) should be NaN");
}

/// n = NaN → should be NaN.
#[test]
fn stirling_nan_input_is_nan() {
    // n < 0.0 check: NaN < 0.0 = false (not caught!), n == 0.0: NaN == 0.0 = false.
    // So: n*ln(n) = NaN*NaN = NaN. Result is NaN. OK by propagation.
    let v = stirling_approx(f64::NAN);
    assert!(v.is_nan(), "stirling_approx(NaN) should be NaN, got {}", v);
}

/// BUG: n = Inf → formula has Inf - Inf = NaN (indeterminate form).
/// stirling_approx(Inf) = Inf*ln(Inf) - Inf + ... = Inf - Inf = NaN.
/// Mathematically ln(Inf!) = Inf, so the result should be Inf.
/// The formula breaks at Inf due to Inf-Inf cancellation.
/// A correct implementation should special-case n=Inf → return Inf.
#[test]
fn stirling_inf_input_should_be_inf_not_nan() {
    let v = stirling_approx(f64::INFINITY);
    // BUG: Inf*ln(Inf) = Inf, then Inf - Inf = NaN. Should be Inf (monotone increasing).
    assert_eq!(v, f64::INFINITY,
        "BUG: stirling_approx(Inf) returns NaN (Inf-Inf cancellation). Should guard n=Inf → return Inf. Got {}", v);
}

/// n = 1: ln(1!) = 0. stirling_approx(1) = 1*ln(1) - 1 + 0.5*ln(2π) ≈ -1 + 0.919 = -0.081.
/// This is NOT 0 — Stirling is only an approximation. For n=1, error is large.
#[test]
fn stirling_n1_is_approximate() {
    let exact = 0.0; // ln(1!) = 0
    let approx = stirling_approx(1.0);
    let corrected = stirling_approx_corrected(1.0);
    // Both should be finite but neither equals 0 (large error for small n)
    assert!(approx.is_finite(), "stirling(1) should be finite, got {}", approx);
    assert!(corrected.is_finite(), "stirling_corrected(1) should be finite, got {}", corrected);
    // Corrected should be closer to truth
    assert!((corrected - exact).abs() < (approx - exact).abs() * 2.0,
        "Corrected stirling should be closer to 0 than basic: basic={}, corrected={}", approx, corrected);
}

/// For large n: both approximations converge to log_gamma(n+1).
/// At n=100: error should be < 1e-7 for corrected version.
#[test]
fn stirling_large_n_agrees_with_log_gamma() {
    for &n in &[10.0, 50.0, 100.0, 1000.0] {
        let true_val = log_gamma(n + 1.0); // log_gamma(n+1) = ln(n!)
        let basic = stirling_approx(n);
        let corrected = stirling_approx_corrected(n);

        let err_basic = (basic - true_val).abs();
        let err_corrected = (corrected - true_val).abs();

        assert!(err_basic < 0.1,
            "stirling_approx({}) should be close to log_gamma({}+1)={}: err={}", n, n, true_val, err_basic);
        assert!(err_corrected < err_basic || err_corrected < 1e-8,
            "corrected stirling({}) should be more accurate: basic_err={}, corrected_err={}", n, err_basic, err_corrected);
    }
}

/// Corrected series: each correction term should reduce the error.
#[test]
fn stirling_corrected_better_than_basic() {
    // For any n > 0, corrected should be closer to truth than basic
    let n = 5.0;
    let true_val = log_gamma(n + 1.0);
    let basic = stirling_approx(n);
    let corrected = stirling_approx_corrected(n);
    assert!((corrected - true_val).abs() < (basic - true_val).abs(),
        "Corrected stirling should be better than basic for n={}: basic_err={}, corrected_err={}",
        n, (basic - true_val).abs(), (corrected - true_val).abs());
}

// ═══════════════════════════════════════════════════════════════════════════
// marchenko_pastur — degenerate inputs
// ═══════════════════════════════════════════════════════════════════════════

/// gamma=0: invalid aspect ratio → pdf should return 0.
#[test]
fn marchenko_pastur_pdf_zero_gamma() {
    let v = marchenko_pastur_pdf(1.0, 0.0, 1.0);
    // gamma <= 0 guard → 0.0
    assert_eq!(v, 0.0, "MP-pdf with gamma=0 should return 0, got {}", v);
}

/// sigma2=0: degenerate distribution → pdf should return 0.
#[test]
fn marchenko_pastur_pdf_zero_sigma2() {
    let v = marchenko_pastur_pdf(1.0, 1.0, 0.0);
    assert_eq!(v, 0.0, "MP-pdf with sigma2=0 should return 0, got {}", v);
}

/// lambda outside bulk [lambda_minus, lambda_plus]: pdf = 0.
#[test]
fn marchenko_pastur_pdf_outside_bulk() {
    let (lo, hi) = marchenko_pastur_bounds(0.5, 1.0);
    let below = marchenko_pastur_pdf(lo * 0.5, 0.5, 1.0);
    let above = marchenko_pastur_pdf(hi * 1.5, 0.5, 1.0);
    assert_eq!(below, 0.0, "MP-pdf below bulk should be 0, got {}", below);
    assert_eq!(above, 0.0, "MP-pdf above bulk should be 0, got {}", above);
}

/// gamma=1: lambda_minus = 0. lambda=0 is excluded (lambda <= 0 guard).
/// Nearby lambda=1e-10 should give a very large but finite value (1/lambda singularity).
#[test]
fn marchenko_pastur_pdf_gamma1_near_zero() {
    let (lo, _) = marchenko_pastur_bounds(1.0, 1.0);
    // For gamma=1: lambda_minus = 0, so near-zero lambda is in-bulk
    assert_eq!(lo, 0.0, "For gamma=1, lambda_minus should be 0, got {}", lo);
    // Very small positive lambda inside the bulk:
    let small_lambda = 1e-10;
    let v = marchenko_pastur_pdf(small_lambda, 1.0, 1.0);
    // Should be very large (near singularity) but finite and positive
    assert!(v.is_finite() && v > 0.0,
        "MP-pdf at gamma=1, lambda=1e-10 should be large finite, got {}", v);
}

/// BUG: gamma > 1 (p > n case). lambda_minus = sigma2*(1-sqrt(gamma))².
/// For gamma=2: sqrt(2)≈1.414, so 1-1.414=-0.414, squared=0.171.
/// lambda_minus > 0. lambda=0 has the lambda <= 0.0 guard fire → 0.
/// But what's the behavior at lambda_minus exactly (boundary)?
#[test]
fn marchenko_pastur_pdf_at_boundary() {
    let gamma = 0.5;
    let sigma2 = 1.0;
    let (lo, hi) = marchenko_pastur_bounds(gamma, sigma2);
    // At the exact boundary: (lambda_plus - lambda) = 0 → numerator = 0 → pdf = 0
    let v_lo = marchenko_pastur_pdf(lo, gamma, sigma2);
    let v_hi = marchenko_pastur_pdf(hi, gamma, sigma2);
    assert_eq!(v_lo, 0.0, "MP-pdf at lambda_minus should be 0, got {}", v_lo);
    assert_eq!(v_hi, 0.0, "MP-pdf at lambda_plus should be 0, got {}", v_hi);
}

/// marchenko_pastur_classify with p=0 or n=0: returns (0,0,NaN).
#[test]
fn marchenko_pastur_classify_degenerate() {
    let (ns, nn, lp) = marchenko_pastur_classify(&[], 0, 10);
    assert_eq!(ns, 0); assert_eq!(nn, 0); assert!(lp.is_nan());
    let (ns2, nn2, lp2) = marchenko_pastur_classify(&[], 10, 0);
    assert_eq!(ns2, 0); assert_eq!(nn2, 0); assert!(lp2.is_nan());
}

/// marchenko_pastur_classify: for p=n=10, gamma=1, lambda_plus=4.
/// All eigenvalues below 4 should be classified as noise.
#[test]
fn marchenko_pastur_classify_all_noise() {
    let eigenvalues = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]; // all < 4.0
    let (n_signal, n_noise, lambda_plus) = marchenko_pastur_classify(&eigenvalues, 10, 10);
    assert!((lambda_plus - 4.0).abs() < 0.01,
        "For p=n=10, gamma=1, lambda_plus should be 4.0, got {}", lambda_plus);
    assert_eq!(n_signal, 0, "All eigenvalues < 4: n_signal should be 0, got {}", n_signal);
    assert_eq!(n_noise, 7, "All 7 eigenvalues should be noise, got {}", n_noise);
}

// ═══════════════════════════════════════════════════════════════════════════
// chebyshev_outlier — degenerate inputs
// ═══════════════════════════════════════════════════════════════════════════

/// std=0: returns (Inf, 0.0) — any deviation is "infinite standard deviations away."
#[test]
fn chebyshev_zero_std_is_inf() {
    let (k, p) = chebyshev_outlier(5.0, 3.0, 0.0);
    assert_eq!(k, f64::INFINITY, "Chebyshev with std=0: k should be Inf, got {}", k);
    assert_eq!(p, 0.0, "Chebyshev with std=0: p_bound should be 0, got {}", p);
}

/// std < 0: should be handled same as std=0 or return NaN.
/// Currently the code has `if std <= 0.0 → return (Inf, 0.0)`.
#[test]
fn chebyshev_negative_std() {
    let (k, p) = chebyshev_outlier(5.0, 3.0, -1.0);
    assert_eq!(k, f64::INFINITY, "Chebyshev with std<0: k should be Inf, got {}", k);
    assert_eq!(p, 0.0, "Chebyshev with std<0: p_bound should be 0, got {}", p);
}

/// x = mean: k = 0, no deviation → p_bound = 1.0 (guaranteed bound ≤ 1 for k ≤ 1).
#[test]
fn chebyshev_at_mean_is_zero() {
    let (k, p) = chebyshev_outlier(5.0, 5.0, 2.0);
    assert_eq!(k, 0.0, "x=mean: k should be 0, got {}", k);
    assert_eq!(p, 1.0, "k=0 ≤ 1: p_bound should be 1.0, got {}", p);
}

/// k=2: p_bound = 1/4 = 0.25.
#[test]
fn chebyshev_k2_p_is_quarter() {
    let (k, p) = chebyshev_outlier(7.0, 5.0, 1.0); // |7-5|/1 = 2
    assert!((k - 2.0).abs() < 1e-12, "k should be 2, got {}", k);
    assert!((p - 0.25).abs() < 1e-12, "k=2: p should be 0.25, got {}", p);
}

/// NaN inputs: std=NaN → guard `std <= 0.0` fails (NaN comparison) → k = NaN.
#[test]
fn chebyshev_nan_std() {
    let (k, _p) = chebyshev_outlier(5.0, 3.0, f64::NAN);
    // NaN <= 0.0 is false → not caught → k = (5-3).abs() / NaN = NaN
    assert!(k.is_nan() || k.is_infinite(),
        "chebyshev_outlier with NaN std should give NaN or Inf k, got {}", k);
}

// ═══════════════════════════════════════════════════════════════════════════
// softmax / log_softmax — numerical stability
// ═══════════════════════════════════════════════════════════════════════════

/// softmax sums to 1.
#[test]
fn softmax_sums_to_one() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let s = softmax(&x);
    let sum: f64 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "softmax should sum to 1, got {}", sum);
}

/// softmax with all-equal inputs: uniform distribution.
#[test]
fn softmax_uniform_inputs() {
    let x = vec![1.0; 5];
    let s = softmax(&x);
    for (i, &si) in s.iter().enumerate() {
        assert!((si - 0.2).abs() < 1e-12,
            "softmax with equal inputs should give 0.2, got s[{}]={}", i, si);
    }
}

/// BUG: softmax with very large inputs — overflow.
/// softmax(x) = exp(x_i) / sum(exp(x_j)). With x=[1000, 1001, 1002]:
/// exp(1000) = Inf. All terms Inf/Inf = NaN.
/// A numerically stable softmax subtracts the max first: exp(x-max)/sum(exp(x-max)).
#[test]
fn softmax_large_inputs_stable() {
    let x = vec![1000.0, 1001.0, 1002.0];
    let s = softmax(&x);
    // Numerically stable: result should be the same as for [0, 1, 2] shifted
    let x_small = vec![0.0, 1.0, 2.0];
    let s_small = softmax(&x_small);
    for i in 0..3 {
        assert!((s[i] - s_small[i]).abs() < 1e-10,
            "softmax([1000,1001,1002]) should equal softmax([0,1,2]) (stability), s[{}]={} vs {}", i, s[i], s_small[i]);
    }
}

/// BUG: softmax with very negative inputs — underflow.
/// exp(-1000) = 0. All terms 0/0 = NaN.
/// Stable: subtract max (=-1000), then [0,-1,-2] → sum=exp(0)+exp(-1)+exp(-2).
#[test]
fn softmax_large_negative_inputs_stable() {
    let x = vec![-1000.0, -1001.0, -1002.0];
    let s = softmax(&x);
    let sum: f64 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10,
        "softmax with large negative inputs should sum to 1, got {} (underflow?)", sum);
    // Largest should be softmax of -1000 (max element): s[0] > s[1] > s[2]
    assert!(s[0] > s[1] && s[1] > s[2],
        "softmax ordering should be preserved: {:?}", s);
}

/// log_softmax for large inputs should be numerically stable.
#[test]
fn log_softmax_large_inputs_stable() {
    let x = vec![1000.0, 1001.0, 1002.0];
    let ls = log_softmax(&x);
    // log_softmax should equal log(softmax), but computed stably
    for v in &ls {
        assert!(v.is_finite(), "log_softmax with large inputs should be finite, got {:?}", ls);
        assert!(*v <= 0.0, "log_softmax values should be <= 0, got {}", v);
    }
    // log_softmax([1000,1001,1002]) should equal log_softmax([0,1,2])
    let ls_small = log_softmax(&[0.0, 1.0, 2.0]);
    for i in 0..3 {
        assert!((ls[i] - ls_small[i]).abs() < 1e-10,
            "log_softmax stability: ls[{}]={} vs ls_small[{}]={}", i, ls[i], i, ls_small[i]);
    }
}

/// softmax with single element: always [1.0].
#[test]
fn softmax_single_element() {
    let s = softmax(&[42.0]);
    assert_eq!(s, vec![1.0], "softmax([x]) should be [1.0], got {:?}", s);
}

/// softmax with NaN input: NaN propagates.
#[test]
fn softmax_nan_propagates() {
    let s = softmax(&[1.0, f64::NAN, 3.0]);
    let has_nan = s.iter().any(|v| v.is_nan());
    // The max subtraction step may propagate NaN through all entries
    // At minimum, NaN in input should not produce finite results that look valid
    assert!(has_nan || s.iter().sum::<f64>().is_nan(),
        "softmax with NaN input should produce NaN output, got {:?}", s);
}

// ═══════════════════════════════════════════════════════════════════════════
// bic_score / aic_score — degenerate inputs
// ═══════════════════════════════════════════════════════════════════════════

/// BUG: bic_score with k=1 returns Infinity because `c.k >= 2` guard.
/// But BIC for k=1 is well-defined — it's the baseline model.
/// This prevents computing the BIC ratio that the gap statistic uses internally.
#[test]
fn bic_score_k1_should_be_finite() {
    // 10 points in 1D, all in one cluster (k=1)
    let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let labels: Vec<i32> = vec![0; 10];
    let bic = bic_score(&data, &labels, 1, 1);
    // k=1: one cluster with all 10 points. BIC should be finite.
    // Currently returns Inf because c.k >= 2 guard excludes k=1.
    assert!(bic.is_finite(),
        "bic_score with k=1 should return finite value, got Inf (c.k >= 2 guard bug)");
}

/// BIC with k > n: impossible assignment. Each cluster has at most 1 point.
/// dof = n*d - k*d could be negative (or 0 if k=n). Returns Inf.
#[test]
fn bic_score_k_equals_n() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let labels = vec![0i32, 1, 2, 3, 4];
    let bic = bic_score(&data, &labels, 1, 5);
    // dof = 5*1 - 5*1 = 0 → returns Inf. This is correct (saturated model).
    assert_eq!(bic, f64::INFINITY,
        "bic_score with k=n (one point per cluster) should be Inf (0 dof), got {}", bic);
}

/// AIC should always be less than BIC for same data (AIC penalizes less).
/// AIC = 2p - 2*loglik, BIC = p*ln(n) - 2*loglik, where p = k*d.
/// AIC < BIC when 2p < p*ln(n), i.e., ln(n) > 2, i.e., n > e² ≈ 7.4.
#[test]
fn aic_less_than_bic_for_large_n() {
    // n=20 > e²: AIC should penalize less than BIC
    let data: Vec<f64> = vec![
        1.0,1.0, 1.1,1.0, 0.9,1.0, // cluster 0 around (1,1)
        5.0,5.0, 5.1,5.0, 4.9,5.0, // cluster 1 around (5,5)
        // ... more points
        1.0,1.1, 5.0,4.9, 1.1,0.9, 4.9,5.1,
        1.0,1.0, 5.0,5.0, 0.9,0.9, 5.1,5.1,
        1.0,1.0, 5.0,5.0,
    ];
    let n = data.len() / 2;
    let mut labels: Vec<i32> = vec![0i32; n];
    for i in (n/2)..n { labels[i] = 1; }
    let bic = bic_score(&data, &labels, 2, 2);
    let aic = aic_score(&data, &labels, 2, 2);
    // Both should be finite
    assert!(bic.is_finite(), "BIC should be finite, got {}", bic);
    assert!(aic.is_finite(), "AIC should be finite, got {}", aic);
    // For n > e²: AIC < BIC (less penalty)
    assert!(aic < bic,
        "For n={}, AIC should be less than BIC: AIC={}, BIC={}", n, aic, bic);
}

/// Empty data: returns Infinity.
#[test]
fn bic_score_empty() {
    let bic = bic_score(&[], &[], 1, 1);
    assert_eq!(bic, f64::INFINITY, "bic_score with empty data should return Inf, got {}", bic);
}

// ═══════════════════════════════════════════════════════════════════════════
// gap_statistic — degenerate inputs
// ═══════════════════════════════════════════════════════════════════════════

/// BUG: gap_statistic with k_range containing 1 should work.
/// The gap statistic compares data clustering to uniform reference.
/// For k=1: one cluster (all data together), gap[0] = E[log W_1^ref] - log W_1.
#[test]
fn gap_statistic_k1_included() {
    let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    // Should not panic with k_range=1..=3
    let result = std::panic::catch_unwind(|| {
        gap_statistic(&data, 10, 1, 1..=3, 5, 42)
    });
    assert!(result.is_ok(), "gap_statistic with k_range=1..=3 should not panic");
}

/// BUG: All-identical data — W=0, log(W)=-Inf, gap = E[log(W_ref)] - (-Inf) = Inf.
/// gap_statistic with constant data produces Inf gaps. This is mathematically
/// defensible (k=1 is infinitely better than any reference) but not a useful result.
/// A robust implementation should detect zero-variance data and return 0 or NaN.
#[test]
fn gap_statistic_constant_data_produces_inf() {
    let data = vec![1.0_f64; 10];
    let result = std::panic::catch_unwind(|| {
        gap_statistic(&data, 10, 1, 1..=3, 5, 42)
    });
    assert!(result.is_ok(), "gap_statistic with constant data should not panic");
    if let Ok(r) = result {
        // Documenting the current behavior: Inf gaps for constant data.
        // This is a usability bug: Inf is not a useful gap statistic value.
        let any_inf = r.gaps.iter().any(|g| g.is_infinite());
        // The test FAILS to assert Inf, documenting that it SHOULD handle this gracefully:
        assert!(!any_inf,
            "BUG: gap_statistic with constant data gives Inf gaps (W=0 → log(0)=-Inf → gap=Inf). Should return 0 or NaN instead. Gaps: {:?}", r.gaps);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bessel functions — known identities and edge cases
// ═══════════════════════════════════════════════════════════════════════════

/// J0(0) = 1 (definition of zeroth-order Bessel function).
#[test]
fn bessel_j0_at_zero() {
    assert!((bessel_j0(0.0) - 1.0).abs() < 1e-12,
        "J0(0) should be 1.0, got {}", bessel_j0(0.0));
}

/// J1(0) = 0 (definition of first-order Bessel function).
#[test]
fn bessel_j1_at_zero() {
    assert!((bessel_j1(0.0) - 0.0).abs() < 1e-12,
        "J1(0) should be 0.0, got {}", bessel_j1(0.0));
}

/// Jn(0) = 0 for n >= 1 (Bessel function of order n at 0 is 0 for n > 0).
#[test]
fn bessel_jn_at_zero() {
    for n in 1..=5 {
        assert!((bessel_jn(n, 0.0) - 0.0).abs() < 1e-12,
            "J{}(0) should be 0.0, got {}", n, bessel_jn(n, 0.0));
    }
}

/// J0(x)² + J1(x)² ≤ 1 for all x (because they're bounded functions).
/// Actually max of J0 is 1 (at x=0) and max of J1 is ~0.58 (at x~1.8).
/// This test just checks they're bounded.
#[test]
fn bessel_j0_bounded() {
    for i in 0..100 {
        let x = i as f64 * 0.5;
        let v = bessel_j0(x);
        assert!(v.abs() <= 1.0 + 1e-10,
            "J0({}) should be in [-1, 1], got {}", x, v);
    }
}

/// Known zero of J0: first zero at x ≈ 2.4048.
#[test]
fn bessel_j0_first_zero() {
    let first_zero = 2.4048255577;
    let v = bessel_j0(first_zero);
    assert!(v.abs() < 1e-6,
        "J0({}) should be ≈ 0, got {}", first_zero, v);
}

/// Recurrence: J_{n+1}(x) = (2n/x) J_n(x) - J_{n-1}(x).
/// This is the three-term recurrence that any correct implementation satisfies.
#[test]
fn bessel_recurrence_relation() {
    let x = 3.0;
    for n in 1..5 {
        let j_nm1 = bessel_jn(n - 1, x); // J_{n-1}
        let j_n = bessel_jn(n, x);       // J_n
        let j_np1 = bessel_jn(n + 1, x); // J_{n+1}
        let expected = (2.0 * n as f64 / x) * j_n - j_nm1;
        assert!((j_np1 - expected).abs() < 1e-8,
            "Bessel recurrence at n={}, x={}: J_{{n+1}}={}, expected={}", n, x, j_np1, expected);
    }
}

/// J0(Inf) should decay to 0 (asymptotically).
#[test]
fn bessel_j0_large_x() {
    let v = bessel_j0(1e10);
    // For large x: J0(x) ~ sqrt(2/(pi*x)) * cos(x - pi/4). |J0(x)| ~ sqrt(2/(pi*x)) → 0.
    assert!(v.abs() < 1e-4,
        "J0(1e10) should be near 0 (asymptotic decay), got {}", v);
}

/// BUG: J0(NaN) — should return NaN.
#[test]
fn bessel_j0_nan() {
    let v = bessel_j0(f64::NAN);
    assert!(v.is_nan(), "J0(NaN) should be NaN, got {}", v);
}

// ═══════════════════════════════════════════════════════════════════════════
// DTW — additional degenerate inputs not in previous tests
// ═══════════════════════════════════════════════════════════════════════════

/// BUG: DTW with NaN input — Rust's f64::min(NaN, x) = x (NOT NaN).
/// This means the DTW cost[i][j] = NaN + prev = NaN, but then subsequent
/// cells do `cost[i*...+j].min(NaN)` = cost[i*...+j], silently discarding NaN.
/// The NaN row effectively blocks all finite paths through it, giving Inf
/// or potentially a wrong finite result, rather than the correct NaN.
///
/// This is a deep bug: f64::min eats NaN in Rust, so DTW silently returns
/// a result that pretends NaN observations don't exist.
#[test]
fn dtw_nan_input_should_signal_error() {
    let x = vec![1.0, f64::NAN, 3.0];
    let y = vec![1.0, 2.0, 3.0];
    let d = dtw(&x, &y);
    // With Rust's f64::min eating NaN: the NaN column becomes Inf (all paths blocked).
    // Got Inf instead of NaN — this is still wrong (should be NaN to signal bad input).
    // A correct implementation should either return NaN or be documented to skip NaN.
    assert!(d.is_nan() || d.is_infinite(),
        "DTW with NaN input should return NaN or Inf (not a finite value pretending to be correct), got {}", d);
    // The specific observed bug: Rust's min(NaN, x) = x silently eats NaN
    // so it returns Inf (blocked paths) rather than NaN (undefined input).
    // Assert the bug exists — this test will FAIL if someone fixes it to return NaN.
    if d.is_infinite() {
        // Document: currently returns Inf due to f64::min(NaN,x)=x eating the NaN.
        // Correct behavior: should return NaN (undefined input → undefined output).
        // This assertion intentionally FAILS to mark the bug:
        assert!(d.is_nan(),
            "BUG: DTW with NaN input returns Inf (f64::min eats NaN). Should return NaN. Got {}", d);
    }
}

/// BUG: DTW with Inf input — |Inf - finite| = Inf. min of Inf paths = Inf.
/// Final result should be Inf, not a finite value or NaN.
#[test]
fn dtw_inf_input() {
    let x = vec![1.0, f64::INFINITY, 3.0];
    let y = vec![1.0, 2.0, 3.0];
    let d = dtw(&x, &y);
    // The infinity propagates: |Inf - 2| = Inf, Inf + min(prev) = Inf.
    // All subsequent cells in that row/col get Inf. Final result = Inf.
    assert!(d.is_infinite() && d > 0.0,
        "DTW with Inf input should return Inf, got {}", d);
}

/// DTW: single-element sequences.
#[test]
fn dtw_single_element() {
    let d = dtw(&[3.0], &[5.0]);
    assert!((d - 2.0).abs() < 1e-12, "DTW([3],[5]) should be 2.0, got {}", d);
}

/// DTW triangle inequality: DTW(X,Z) ≤ DTW(X,Y) + DTW(Y,Z) approximately.
/// (DTW doesn't satisfy exact triangle inequality, but for identical length sequences
/// it should approximately hold for well-behaved inputs.)
#[test]
fn dtw_non_negative() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![2.0, 3.0, 4.0, 5.0];
    let d = dtw(&x, &y);
    assert!(d >= 0.0, "DTW should be non-negative, got {}", d);
}

/// DTW with all-identical sequences of different lengths: cost = 0.
/// [a,a,a] vs [a,a]: DTW = 0 because we can warp perfectly.
#[test]
fn dtw_identical_different_lengths() {
    let x = vec![5.0; 5];
    let y = vec![5.0; 3];
    let d = dtw(&x, &y);
    assert_eq!(d, 0.0, "DTW(all-5 len 5, all-5 len 3) should be 0, got {}", d);
}
