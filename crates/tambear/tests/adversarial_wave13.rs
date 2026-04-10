//! Adversarial Wave 13 — Scan identity elements and associativity
//!
//! Targets Aristotle's hypothesis: if an Op variant's identity element is wrong,
//! the scan produces correct results for power-of-2 lengths (no padding) and
//! wrong results for all other lengths.
//!
//! Also targets: catastrophic cancellation in associative compositions,
//! NaN propagation through scan primitives, and the tridiagonal scan
//! zero-pivot identity-substitution bug.
//!
//! Tests assert mathematical truths. Failures are bugs.

use tambear::linear_algebra::{
    solve_tridiagonal, solve_tridiagonal_scan,
    tridiagonal_scan_element, tridiagonal_scan_compose,
};
use tambear::series_accel::{partial_sums, cumsum};
use tambear::signal_processing::{fft, ifft, Complex};

// ═══════════════════════════════════════════════════════════════════════════
// Tridiagonal scan — identity element bug
// ═══════════════════════════════════════════════════════════════════════════

/// When pivot b_i = 0, tridiagonal_scan_element returns the identity matrix.
/// This silently masks a singular matrix — the compose chain continues as if
/// the singular row doesn't exist, producing garbage rather than None.
///
/// BUG: tridiagonal_scan_element should signal failure (return NaN-filled
/// element, or the parallel scan infra should check for singularity),
/// not silently substitute the identity and continue.
#[test]
fn scan_element_zero_pivot_signals_failure() {
    // A row with b=0 and nonzero a,c means the system is singular.
    let elem = tridiagonal_scan_element(1.0, 0.0, 1.0); // b_i = 0: singular pivot
    // The identity matrix composed into the prefix scan is the identity element
    // e such that compose(e, x) = x. If a singular row produces the identity,
    // the scan doesn't notice the singularity — it just continues.
    //
    // Mathematical truth: a zero pivot means the matrix is singular or
    // numerically degenerate. The scan element MUST NOT be the identity
    // (which would silently skip this row). It should propagate a signal
    // (NaN or a special sentinel) that the system is singular.
    //
    // The identity matrix has det=1. A valid scan element for a singular
    // row should have det=0 or contain NaN to signal failure.
    //
    // Compute det of returned 3x3:
    let m = elem;
    // det(3x3) = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6])
    let det = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]);
    // If det=1 (identity returned), the bug is confirmed: singular row masked as identity.
    assert!(
        (det - 1.0).abs() > 1e-10,
        "BUG: tridiagonal_scan_element(a, b=0, c) returns 3x3 identity (det=1), \
         silently masking singular pivot. Should return NaN-filled element or \
         special sentinel, got det={:.6}", det
    );
}

/// When a zero pivot is in the middle of the chain, the scan-based solver
/// should agree with the Thomas solver: both should return None (singular).
/// But the scan element substitutes identity, allowing the compose to continue —
/// so the scan solver may return Some(garbage) while Thomas returns None.
///
/// BUG: solve_tridiagonal_scan and solve_tridiagonal must agree on singularity.
#[test]
fn scan_solver_agrees_with_thomas_on_singular_system() {
    // System where the second pivot becomes 0 after first elimination step.
    // b_1 = a_1 * c_0 / b_0 makes the pivot exactly zero.
    // Set b_0=1, c_0=2, a_1=3, b_1=6: pivot becomes 6 - 3*2 = 0.
    let lower = vec![3.0];
    let main  = vec![1.0, 6.0];
    let upper = vec![2.0];
    let rhs   = vec![1.0, 1.0];

    let thomas_result = solve_tridiagonal(&lower, &main, &upper, &rhs);
    let scan_result   = solve_tridiagonal_scan(&lower, &main, &upper, &rhs);

    // Both must agree: either both None (singular) or both Some with same answer.
    match (&thomas_result, &scan_result) {
        (None, None) => (), // correct — both detect singularity
        (Some(_), Some(xs_scan)) => {
            // Both found a solution — verify they match
            if let Some(xs_thomas) = &thomas_result {
                for (a, b) in xs_thomas.iter().zip(xs_scan.iter()) {
                    assert!((a - b).abs() < 1e-8,
                        "Thomas and scan solvers disagree: thomas={:?} scan={:?}",
                        thomas_result, scan_result);
                }
            }
        }
        (None, Some(xs)) => {
            assert!(false,
                "BUG: Thomas detected singular pivot (returned None) but \
                 scan solver returned Some({:?}) — scan's identity substitution \
                 masks the singularity and returns a garbage solution",
                xs);
        }
        (Some(_), None) => {
            assert!(false,
                "Thomas returned a solution but scan returned None — \
                 unexpected behavior, scan is more conservative");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tridiagonal scan vs Thomas: non-power-of-2 consistency
// ═══════════════════════════════════════════════════════════════════════════

/// For any valid (non-singular) tridiagonal system, solve_tridiagonal and
/// solve_tridiagonal_scan must produce identical results. This property
/// must hold for ALL input lengths, not just powers of 2.
///
/// If a parallel scan had the wrong identity element, the sequential version
/// here would still be correct (it doesn't pad), but this test ensures the
/// two sequential implementations are consistent — a prerequisite for
/// trusting either for the parallel version.
fn make_diagonally_dominant_tridiagonal(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Diagonally dominant → non-singular. lower, main, upper, rhs.
    // main[i] = 4.0, lower[i] = -1.0, upper[i] = -1.0, rhs[i] = 2.0 * (i+1) as f64
    let lower: Vec<f64> = vec![-1.0; n - 1];
    let main:  Vec<f64> = vec![4.0;  n];
    let upper: Vec<f64> = vec![-1.0; n - 1];
    let rhs:   Vec<f64> = (0..n).map(|i| 2.0 * (i + 1) as f64).collect();
    (lower, main, upper, rhs)
}

#[test]
fn tridiagonal_scan_vs_thomas_n16() {
    let (lower, main, upper, rhs) = make_diagonally_dominant_tridiagonal(16);
    let thomas = solve_tridiagonal(&lower, &main, &upper, &rhs).expect("singular");
    let scan   = solve_tridiagonal_scan(&lower, &main, &upper, &rhs).expect("singular");
    for (a, b) in thomas.iter().zip(scan.iter()) {
        assert!((a - b).abs() < 1e-10, "n=16: thomas={} scan={}", a, b);
    }
}

#[test]
fn tridiagonal_scan_vs_thomas_n17() {
    let (lower, main, upper, rhs) = make_diagonally_dominant_tridiagonal(17);
    let thomas = solve_tridiagonal(&lower, &main, &upper, &rhs).expect("singular");
    let scan   = solve_tridiagonal_scan(&lower, &main, &upper, &rhs).expect("singular");
    for (a, b) in thomas.iter().zip(scan.iter()) {
        assert!((a - b).abs() < 1e-10,
            "n=17 (non-power-of-2): thomas={} scan={} — mismatch may indicate identity bug",
            a, b);
    }
}

#[test]
fn tridiagonal_scan_vs_thomas_n31() {
    let (lower, main, upper, rhs) = make_diagonally_dominant_tridiagonal(31);
    let thomas = solve_tridiagonal(&lower, &main, &upper, &rhs).expect("singular");
    let scan   = solve_tridiagonal_scan(&lower, &main, &upper, &rhs).expect("singular");
    for (a, b) in thomas.iter().zip(scan.iter()) {
        assert!((a - b).abs() < 1e-10, "n=31: thomas={} scan={}", a, b);
    }
}

#[test]
fn tridiagonal_scan_vs_thomas_n100() {
    let (lower, main, upper, rhs) = make_diagonally_dominant_tridiagonal(100);
    let thomas = solve_tridiagonal(&lower, &main, &upper, &rhs).expect("singular");
    let scan   = solve_tridiagonal_scan(&lower, &main, &upper, &rhs).expect("singular");
    for (a, b) in thomas.iter().zip(scan.iter()) {
        assert!((a - b).abs() < 1e-10, "n=100: thomas={} scan={}", a, b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tridiagonal scan compose: associativity
// ═══════════════════════════════════════════════════════════════════════════

/// The scan compose operation must be strictly associative.
/// compose(compose(a,b), c) must equal compose(a, compose(b,c)).
/// Floating-point non-associativity is a concern for the Sarkka correction term;
/// for 3x3 matrix multiply it should hold to machine precision for normal inputs.
#[test]
fn scan_compose_associativity_normal() {
    let a = tridiagonal_scan_element(-0.5, 2.0, -0.3);
    let b = tridiagonal_scan_element(-0.7, 3.0, -0.1);
    let c = tridiagonal_scan_element(-0.4, 4.0, -0.6);

    let left  = tridiagonal_scan_compose(&tridiagonal_scan_compose(&a, &b), &c);
    let right = tridiagonal_scan_compose(&a, &tridiagonal_scan_compose(&b, &c));

    for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        assert!((l - r).abs() < 1e-12,
            "associativity failure at component {}: left={} right={}", i, l, r);
    }
}

/// Large dynamic range: mixing tiny and huge values in the compose chain.
/// Catastrophic cancellation in [-a/b + ...] terms.
/// For Aristotle's second hypothesis: large dynamic range amplifies
/// floating-point non-associativity beyond acceptable tolerance.
#[test]
fn scan_compose_associativity_large_dynamic_range() {
    // Deliberately large condition number: b ~ 1e8, a ~ 1, c ~ 1e-8
    let a_elem = tridiagonal_scan_element(1.0, 1e8, 1e-8);
    let b_elem = tridiagonal_scan_element(1e-8, 2.0, 1.0);
    let c_elem = tridiagonal_scan_element(1.0, 3e7, 1e-6);

    let left  = tridiagonal_scan_compose(
        &tridiagonal_scan_compose(&a_elem, &b_elem), &c_elem);
    let right = tridiagonal_scan_compose(
        &a_elem, &tridiagonal_scan_compose(&b_elem, &c_elem));

    // For non-pathological inputs, relative error should be < 1e-8.
    for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        let scale = l.abs().max(r.abs()).max(1e-300);
        let rel_err = (l - r).abs() / scale;
        assert!(rel_err < 1e-6,
            "Catastrophic cancellation in scan compose at component {}: \
             left={} right={} rel_err={:.2e} — tree evaluation order matters \
             for high-dynamic-range inputs",
            i, l, r, rel_err);
    }
}

/// Compose with the identity element: compose(identity, x) = x.
/// This verifies the identity element is correct for the 3×3 matrix monoid.
#[test]
fn scan_compose_identity_left() {
    let identity = [1.0f64, 0.0, 0.0,
                    0.0,    1.0, 0.0,
                    0.0,    0.0, 1.0];
    let x = tridiagonal_scan_element(-0.5, 3.0, -0.7);
    let result = tridiagonal_scan_compose(&identity, &x);
    for (i, (r, xi)) in result.iter().zip(x.iter()).enumerate() {
        assert!((r - xi).abs() < 1e-14,
            "compose(I, x) ≠ x at component {}: got {} expected {}", i, r, xi);
    }
}

/// Compose with the identity on the right: compose(x, identity) = x.
#[test]
fn scan_compose_identity_right() {
    let identity = [1.0f64, 0.0, 0.0,
                    0.0,    1.0, 0.0,
                    0.0,    0.0, 1.0];
    let x = tridiagonal_scan_element(-0.5, 3.0, -0.7);
    let result = tridiagonal_scan_compose(&x, &identity);
    for (i, (r, xi)) in result.iter().zip(x.iter()).enumerate() {
        assert!((r - xi).abs() < 1e-14,
            "compose(x, I) ≠ x at component {}: got {} expected {}", i, r, xi);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FFT: Parseval's theorem — energy preserved across non-power-of-2 lengths
// ═══════════════════════════════════════════════════════════════════════════

/// Parseval's theorem: Σ|x_k|² = (1/N) Σ|X_k|².
/// This must hold for ALL input lengths, including non-powers-of-2.
/// If the zero-padding identity for the butterfly is wrong (non-zero),
/// Parseval breaks for non-power-of-2 inputs.
fn parseval_error(data: &[Complex]) -> f64 {
    let n = data.len();
    let spectrum = fft(data);
    let n_fft = spectrum.len(); // next_pow2(n)

    let energy_time: f64 = data.iter().map(|&(re, im)| re*re + im*im).sum();
    let energy_freq: f64 = spectrum.iter().map(|&(re, im)| re*re + im*im).sum::<f64>()
        / n_fft as f64;

    // But wait: zero-padding changes the energy! Parseval holds for the
    // zero-padded version: sum over N_fft points in time = original N points
    // (the rest are zero, contributing 0 to time-domain energy).
    // So: Σ_{k=0}^{N-1} |x_k|² = (1/N_fft) Σ_{k=0}^{N_fft-1} |X_k|²
    (energy_time - energy_freq).abs()
}

#[test]
fn fft_parseval_power_of_2() {
    let data: Vec<Complex> = (0..16).map(|i| (i as f64, 0.0)).collect();
    let err = parseval_error(&data);
    assert!(err < 1e-8, "Parseval error for n=16 (power of 2): {}", err);
}

#[test]
fn fft_parseval_non_power_of_2_n17() {
    let data: Vec<Complex> = (0..17).map(|i| (i as f64, 0.0)).collect();
    let err = parseval_error(&data);
    assert!(err < 1e-8,
        "Parseval error for n=17 (non-power-of-2): {} — \
         zero-padding identity must be (0+0i) for energy conservation", err);
}

#[test]
fn fft_parseval_non_power_of_2_n31() {
    let data: Vec<Complex> = (0..31).map(|i| ((i as f64).sin(), 0.0)).collect();
    let err = parseval_error(&data);
    assert!(err < 1e-8, "Parseval error for n=31: {}", err);
}

#[test]
fn fft_parseval_non_power_of_2_n100() {
    let data: Vec<Complex> = (0..100).map(|i| ((i as f64 * 0.1).cos(), 0.0)).collect();
    let err = parseval_error(&data);
    assert!(err < 1e-8, "Parseval error for n=100: {}", err);
}

// ═══════════════════════════════════════════════════════════════════════════
// FFT: round-trip (fft then ifft) for non-power-of-2 inputs
// ═══════════════════════════════════════════════════════════════════════════

/// ifft(fft(x)) should recover x (within numerical precision).
/// For non-power-of-2 inputs, zero-padding adds extra frequency components;
/// the inverse must discard them (truncate to original length).
#[test]
fn fft_round_trip_n17() {
    let data: Vec<Complex> = (0..17).map(|i| ((i as f64).sin(), 0.0)).collect();
    let spectrum = fft(&data);
    let recovered = ifft(&spectrum);
    // ifft returns the full padded length; check first 17 elements
    for (i, (&(re_orig, im_orig), &(re_rec, im_rec))) in data.iter().zip(recovered.iter()).enumerate() {
        assert!((re_orig - re_rec).abs() < 1e-10,
            "Round-trip Re error at i={}: orig={} recovered={}", i, re_orig, re_rec);
        assert!(im_orig.abs() < 1e-14 && im_rec.abs() < 1e-10,
            "Round-trip Im error at i={}: orig={} recovered={}", i, im_orig, im_rec);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// cumsum / partial_sums: NaN propagation
// ═══════════════════════════════════════════════════════════════════════════

/// A NaN in the input at position k should propagate through all subsequent
/// prefix sums (positions k, k+1, ..., n-1 should all be NaN).
/// If NaN silently vanishes (due to Rust's `+` NaN behavior with the
/// accumulator), this test will catch it.
#[test]
fn cumsum_nan_propagates_forward() {
    let input = vec![1.0, 2.0, f64::NAN, 3.0, 4.0];
    let out = cumsum(&input);
    assert_eq!(out.len(), 5);
    assert!((out[0] - 1.0).abs() < 1e-14, "cumsum[0] should be 1, got {}", out[0]);
    assert!((out[1] - 3.0).abs() < 1e-14, "cumsum[1] should be 3, got {}", out[1]);
    assert!(out[2].is_nan(), "cumsum[2] should be NaN (NaN input at index 2), got {}", out[2]);
    assert!(out[3].is_nan(), "cumsum[3] should be NaN (NaN propagated), got {}", out[3]);
    assert!(out[4].is_nan(), "cumsum[4] should be NaN (NaN propagated), got {}", out[4]);
}

/// NaN at position 0 should make ALL prefix sums NaN.
#[test]
fn cumsum_nan_at_start_makes_all_nan() {
    let input = vec![f64::NAN, 1.0, 2.0, 3.0];
    let out = cumsum(&input);
    for (i, v) in out.iter().enumerate() {
        assert!(v.is_nan(), "cumsum[{}] should be NaN when input[0]=NaN, got {}", i, v);
    }
}

/// Inf in input: prefix sum should be Inf from that point on.
#[test]
fn cumsum_inf_propagates_forward() {
    let input = vec![1.0, 2.0, f64::INFINITY, 3.0, 4.0];
    let out = cumsum(&input);
    assert!((out[0] - 1.0).abs() < 1e-14);
    assert!((out[1] - 3.0).abs() < 1e-14);
    assert!(out[2].is_infinite() && out[2] > 0.0, "Inf should propagate: got {}", out[2]);
    assert!(out[3].is_infinite() && out[3] > 0.0, "Inf should propagate: got {}", out[3]);
    assert!(out[4].is_infinite() && out[4] > 0.0, "Inf should propagate: got {}", out[4]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Tridiagonal scan: NaN input
// ═══════════════════════════════════════════════════════════════════════════

/// NaN in main diagonal should produce NaN output or None, not silently
/// produce a finite result.
#[test]
fn tridiagonal_scan_element_nan_pivot_no_garbage() {
    let elem = tridiagonal_scan_element(0.5, f64::NAN, 0.5);
    // At least one component should be NaN (pivot is NaN → inv_b is NaN → element is NaN)
    let has_nan = elem.iter().any(|v| v.is_nan());
    assert!(has_nan,
        "tridiagonal_scan_element with NaN pivot should produce NaN component, \
         but all elements are finite: {:?}", elem);
}

/// If a NaN makes it into the compose chain, the result should propagate NaN,
/// not produce a finite value that looks valid.
#[test]
fn scan_compose_nan_propagates() {
    let nan_elem = tridiagonal_scan_element(0.5, f64::NAN, 0.5);
    let valid_elem = tridiagonal_scan_element(-0.5, 3.0, -0.5);

    // compose(nan_elem, valid_elem) should have NaN in result
    let result = tridiagonal_scan_compose(&nan_elem, &valid_elem);
    let has_nan = result.iter().any(|v| v.is_nan());
    assert!(has_nan,
        "compose(NaN-containing element, valid element) should produce NaN output, \
         but got all-finite result: {:?} — NaN swallowed in matrix multiply",
        result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scan correctness: verify against known analytic solution
// ═══════════════════════════════════════════════════════════════════════════

/// The 1D Poisson equation (-u'' = f) on [0,1] with Dirichlet BC u(0)=u(1)=0
/// discretized with h=1/(n+1) gives a tridiagonal system with:
///   main[i] = 2/h², lower[i] = upper[i] = -1/h²
///   rhs[i] = f(x_i)
///
/// For f(x) = 1, the analytic solution is u(x) = x(1-x)/2.
/// This is a mathematically meaningful test with known analytic answer.
#[test]
fn tridiagonal_poisson_1d_n15() {
    let n = 15; // non-power-of-2: 15 = 2^4 - 1
    let h = 1.0 / (n + 1) as f64;
    let diag_val = 2.0 / (h * h);
    let off_val  = -1.0 / (h * h);
    let lower = vec![off_val; n - 1];
    let main  = vec![diag_val; n];
    let upper = vec![off_val; n - 1];
    let rhs   = vec![1.0; n]; // f(x) = 1

    let sol = solve_tridiagonal(&lower, &main, &upper, &rhs).expect("should not be singular");
    let sol_scan = solve_tridiagonal_scan(&lower, &main, &upper, &rhs).expect("should not be singular");

    // Analytic: u(x_i) = x_i*(1-x_i)/2
    for i in 0..n {
        let xi = (i + 1) as f64 * h;
        let analytic = xi * (1.0 - xi) / 2.0;
        assert!((sol[i] - analytic).abs() < 1e-10,
            "Poisson n=15, i={}: thomas={} analytic={}", i, sol[i], analytic);
        assert!((sol_scan[i] - analytic).abs() < 1e-10,
            "Poisson n=15 (scan), i={}: scan={} analytic={}", i, sol_scan[i], analytic);
    }
}

#[test]
fn tridiagonal_poisson_1d_n32() {
    let n = 32; // exact power of 2
    let h = 1.0 / (n + 1) as f64;
    let diag_val = 2.0 / (h * h);
    let off_val  = -1.0 / (h * h);
    let lower = vec![off_val; n - 1];
    let main  = vec![diag_val; n];
    let upper = vec![off_val; n - 1];
    let rhs   = vec![1.0; n];

    let sol      = solve_tridiagonal(&lower, &main, &upper, &rhs).expect("should not be singular");
    let sol_scan = solve_tridiagonal_scan(&lower, &main, &upper, &rhs).expect("should not be singular");

    for i in 0..n {
        let xi = (i + 1) as f64 * h;
        let analytic = xi * (1.0 - xi) / 2.0;
        assert!((sol[i] - analytic).abs() < 1e-10,
            "Poisson n=32, i={}: thomas={} analytic={}", i, sol[i], analytic);
        assert!((sol_scan[i] - analytic).abs() < 1e-10,
            "Poisson n=32 (scan), i={}: scan={} analytic={}", i, sol_scan[i], analytic);
    }
}

#[test]
fn tridiagonal_poisson_1d_n1000() {
    let n = 1000; // scale test
    let h = 1.0 / (n + 1) as f64;
    let diag_val = 2.0 / (h * h);
    let off_val  = -1.0 / (h * h);
    let lower = vec![off_val; n - 1];
    let main  = vec![diag_val; n];
    let upper = vec![off_val; n - 1];
    let rhs   = vec![1.0; n];

    let sol      = solve_tridiagonal(&lower, &main, &upper, &rhs).expect("should not be singular");
    let sol_scan = solve_tridiagonal_scan(&lower, &main, &upper, &rhs).expect("should not be singular");

    // Check a few interior points + max deviation
    let max_err_thomas: f64 = (0..n).map(|i| {
        let xi = (i + 1) as f64 * h;
        (sol[i] - xi*(1.0-xi)/2.0).abs()
    }).fold(0.0_f64, f64::max);

    let max_err_scan: f64 = (0..n).map(|i| {
        let xi = (i + 1) as f64 * h;
        (sol_scan[i] - xi*(1.0-xi)/2.0).abs()
    }).fold(0.0_f64, f64::max);

    assert!(max_err_thomas < 1e-8,
        "Poisson n=1000: max Thomas error = {:.2e}", max_err_thomas);
    assert!(max_err_scan < 1e-8,
        "Poisson n=1000: max scan error = {:.2e}", max_err_scan);
    assert!((max_err_thomas - max_err_scan).abs() < 1e-10,
        "Thomas and scan disagree at scale: thomas_err={:.2e} scan_err={:.2e}",
        max_err_thomas, max_err_scan);
}
