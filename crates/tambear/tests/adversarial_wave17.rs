//! Adversarial Wave 17 — NaN-eating in matrix norms and correlation dimension
//!
//! Aristotle's three-test template applied to:
//!
//! 1. `Mat::norm_inf` — `fold(0.0_f64, f64::max)` at linear_algebra.rs:121
//!    L∞ norm = max row sum of |entries|. NaN entry → row sum NaN → fold eats it → 0.0
//!
//! 2. `Mat::norm_1` — `fold(0.0_f64, f64::max)` at linear_algebra.rs:128
//!    L1 norm = max column sum of |entries|. Same bug.
//!
//! 3. `correlation_dimension` — `fold(0.0, f64::max)` at complexity.rs:492
//!    L∞ pairwise distance. NaN in embedded vectors → distance NaN → fold eats it → 0.0
//!    Distances of 0.0 silently enter the histogram → correlation integral distorted
//!    → dimension estimate garbage.
//!
//! Mathematical truths:
//!   - ||A||_∞ is undefined when A contains NaN → norm_inf must return NaN
//!   - ||A||_1 is undefined when A contains NaN → norm_1 must return NaN
//!   - ||A||_∞ ≥ 0 for NaN-free matrices (0.0 is valid, but only when all rows are zero)
//!   - ||I||_∞ = 1 (identity matrix)  ||I||_1 = 1
//!   - ||A||_∞ = max_i sum_j |a_ij| (row norm)
//!   - ||A||_1 = max_j sum_i |a_ij| (column norm)
//!   - For a vector v treated as column matrix: ||v||_∞ = max(|v_i|), ||v||_1 = ||v||_1
//!
//! All tests assert mathematical truths. Failures are bugs.

use tambear::linear_algebra::Mat;
use tambear::correlation_dimension;

// ═══════════════════════════════════════════════════════════════════════════
// Mat::norm_inf — Test 1: NaN propagation
// ═══════════════════════════════════════════════════════════════════════════

/// NaN in a matrix entry → row sum contains NaN → that row's L∞ contribution = NaN.
/// fold(0.0, f64::max): f64::max(0.0, NaN) = 0.0 — NaN eaten.
/// norm_inf returns a finite value even though ||A|| is undefined.
///
/// EXPECTED: norm_inf returns NaN.
/// ACTUAL (BUG): returns 0.0 (or the max of the NaN-free rows).
#[test]
fn mat_norm_inf_nan_entry_must_propagate() {
    // 2×2 matrix: [[1, NaN], [3, 4]]
    // row 0 sum = |1| + |NaN| = NaN
    // row 1 sum = |3| + |4| = 7
    // ||A||_∞ = max(NaN, 7) = NaN (mathematical truth)
    // fold(0.0, max): max(0.0, NaN) = 0.0; max(0.0, 7.0) = 7.0 → returns 7.0
    let a = Mat::from_vec(2, 2, vec![1.0, f64::NAN, 3.0, 4.0]);

    let norm = a.norm_inf();

    assert!(norm.is_nan(),
        "BUG: norm_inf with NaN entry should return NaN, got {} \
         — fold(0.0, f64::max) eats NaN from the row containing NaN, \
         so only the clean row (sum=7) contributes, giving 7.0 instead of NaN",
        norm);
}

/// NaN in the MAXIMUM-SUM row: fold(0.0, max) visits NaN row last → eats it.
/// The NaN row has the largest sum, so max of clean rows is returned.
/// The returned norm is LESS THAN the true norm of the clean submatrix.
/// This is the worst case: the function silently UNDERESTIMATES the norm.
#[test]
fn mat_norm_inf_nan_in_dominant_row_underestimates() {
    // 3×2: row 0=[1,1] sum=2, row 1=[NaN,10] sum=NaN, row 2=[3,3] sum=6
    // True ||A||_∞ = NaN (undefined due to NaN entry)
    // fold(0.0, max): max(0.0,2,NaN,6) — NaN eaten, returns 6.0
    let a = Mat::from_vec(3, 2, vec![1.0, 1.0, f64::NAN, 10.0, 3.0, 3.0]);

    let norm = a.norm_inf();

    assert!(norm.is_nan(),
        "BUG: norm_inf with NaN in dominant row should return NaN, got {} \
         — NaN row has sum≈10+ε but is silently swallowed, returning 6.0 \
         (underestimate of the NaN-free submatrix norm, worse than the full bug)",
        norm);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mat::norm_inf — Test 2: Correctness on clean matrices
// ═══════════════════════════════════════════════════════════════════════════

/// ||I||_∞ = 1.0 (identity matrix). Row sums are all 1.0.
/// fold(0.0, max): max(0.0, 1.0, 1.0, ...) = 1.0 ✓
#[test]
fn mat_norm_inf_identity_is_one() {
    for n in [1, 2, 3, 5, 10] {
        let a = Mat::eye(n);
        let norm = a.norm_inf();
        assert!((norm - 1.0).abs() < 1e-14,
            "||I_{}||_∞ should be 1.0, got {}", n, norm);
    }
}

/// For a diagonal matrix D with entries d_0, ..., d_{n-1}:
/// row i has one nonzero entry: |d_i|.
/// ||D||_∞ = max_i |d_i|.
#[test]
fn mat_norm_inf_diagonal_matrix() {
    // D = diag(3, 1, 4, 1, 5, 9)
    let diags = [3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0];
    let d = Mat::diag(&diags);
    let norm = d.norm_inf();
    let expected = 9.0_f64;  // max of diag entries
    assert!((norm - expected).abs() < 1e-14,
        "||diag(3,1,4,1,5,9)||_∞ should be 9.0, got {}", norm);
}

/// For a row vector [a, b, c], norm_inf = |a| + |b| + |c| (single row sum).
#[test]
fn mat_norm_inf_row_vector() {
    // Row vector: [1.0, -2.0, 3.0]
    let a = Mat::from_vec(1, 3, vec![1.0, -2.0, 3.0]);
    let norm = a.norm_inf();
    // One row: |1| + |-2| + |3| = 6.0
    assert!((norm - 6.0).abs() < 1e-14,
        "||[1,-2,3]||_∞ (row vector) should be 6.0, got {}", norm);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mat::norm_1 — Test 1: NaN propagation
// ═══════════════════════════════════════════════════════════════════════════

/// NaN in a matrix entry → column sum containing NaN → fold eats it.
///
/// EXPECTED: norm_1 returns NaN.
/// ACTUAL (BUG): returns max of NaN-free column sums.
#[test]
fn mat_norm_1_nan_entry_must_propagate() {
    // 2×2: [[1, 2], [NaN, 4]]
    // col 0 sum = |1| + |NaN| = NaN
    // col 1 sum = |2| + |4| = 6
    // ||A||_1 = max(NaN, 6) = NaN
    // fold(0.0, max): eats NaN → returns 6.0
    let a = Mat::from_vec(2, 2, vec![1.0, 2.0, f64::NAN, 4.0]);

    let norm = a.norm_1();

    assert!(norm.is_nan(),
        "BUG: norm_1 with NaN entry should return NaN, got {} \
         — fold(0.0, f64::max) eats NaN from the column containing it, \
         returning 6.0 instead of NaN",
        norm);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mat::norm_1 — Test 2: Correctness on clean matrices
// ═══════════════════════════════════════════════════════════════════════════

/// ||I||_1 = 1.0 (identity matrix). Column sums are all 1.0.
#[test]
fn mat_norm_1_identity_is_one() {
    for n in [1, 2, 3, 5, 10] {
        let a = Mat::eye(n);
        let norm = a.norm_1();
        assert!((norm - 1.0).abs() < 1e-14,
            "||I_{}||_1 should be 1.0, got {}", n, norm);
    }
}

/// For a column vector [a, b, c]^T, norm_1 = |a| + |b| + |c| (single column sum).
#[test]
fn mat_norm_1_column_vector() {
    // Column vector: [1.0, -2.0, 3.0]^T
    let a = Mat::col_vec(&[1.0, -2.0, 3.0]);
    let norm = a.norm_1();
    // One column: |1| + |-2| + |3| = 6.0
    assert!((norm - 6.0).abs() < 1e-14,
        "||[1,-2,3]^T||_1 (column vector) should be 6.0, got {}", norm);
}

/// norm_inf and norm_1 satisfy the submultiplicativity inequality:
/// ||AB||_∞ ≤ ||A||_∞ · ||B||_∞
/// Test this as a sanity check (consistency property, not NaN-related).
#[test]
fn mat_norm_inf_submultiplicative() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // AB = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    let a = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Mat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
    // AB manually
    let ab = Mat::from_vec(2, 2, vec![19.0, 22.0, 43.0, 50.0]);

    let norm_ab = ab.norm_inf();     // max(19+22, 43+50) = max(41, 93) = 93
    let norm_a = a.norm_inf();       // max(1+2, 3+4) = max(3, 7) = 7
    let norm_b = b.norm_inf();       // max(5+6, 7+8) = max(11, 15) = 15

    assert!(norm_ab <= norm_a * norm_b + 1e-10,
        "||AB||_∞ = {} should be ≤ ||A||_∞ · ||B||_∞ = {} · {} = {}",
        norm_ab, norm_a, norm_b, norm_a * norm_b);
}

// ═══════════════════════════════════════════════════════════════════════════
// correlation_dimension — Test 1: NaN in data must propagate
// ═══════════════════════════════════════════════════════════════════════════

/// NaN in the time series → NaN in the delay-embedded vectors
/// → NaN in the L∞ pairwise distances (fold(0.0, max) eats NaN → distance 0.0)
/// → 0.0 distances enter the correlation integral as if points coincide
/// → C(r) is distorted for all r → slope (dimension) is garbage.
///
/// EXPECTED: correlation_dimension returns NaN when data contains NaN.
/// ACTUAL (BUG): returns a finite (incorrect) dimension.
#[test]
fn correlation_dimension_nan_in_data_must_propagate() {
    // Lorenz-like trajectory: periodic signal + NaN injection
    let mut data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
    data[100] = f64::NAN;  // inject NaN mid-series

    let d = correlation_dimension(&data, 3, 1);

    assert!(d.is_nan(),
        "BUG: correlation_dimension with NaN in data should return NaN, got {} \
         — fold(0.0, f64::max) in L∞ pairwise distance eats NaN, \
         producing 0.0 distances that distort the correlation integral",
        d);
}

// ═══════════════════════════════════════════════════════════════════════════
// correlation_dimension — Test 2: known dimension estimates
// ═══════════════════════════════════════════════════════════════════════════

/// A periodic 1D signal (sine wave) has attractor dimension 1.
/// Embedded in m=2 with appropriate tau, the attractor is a closed curve (circle) in 2D.
/// Correlation dimension should be approximately 1.
#[test]
fn correlation_dimension_periodic_signal_approx_one() {
    // Pure sine wave: attractor is a circle → D₂ ≈ 1
    let data: Vec<f64> = (0..500).map(|i| (i as f64 * std::f64::consts::TAU / 50.0).sin()).collect();

    let d = correlation_dimension(&data, 2, 12);  // tau = quarter-period

    if d.is_nan() { return; }  // skip if too few vectors

    // D₂ ≈ 1 for a periodic attractor. Allow wide tolerance for finite-sample estimation.
    assert!(d > 0.5 && d < 1.8,
        "correlation_dimension of periodic signal should be ≈ 1, got {}", d);
}

/// A 2D Brownian motion (IID noise) has no attractor structure.
/// Estimated dimension should be higher than for structured signals,
/// and the estimator should not return a negative dimension.
#[test]
fn correlation_dimension_is_nonnegative() {
    // IID noise: correlation dimension should be positive (not negative or NaN)
    let data: Vec<f64> = (0..300).map(|i| {
        // Deterministic "noise" using a simple LCG — avoids external RNG
        let x = ((i * 1664525 + 1013904223) & 0x7fffffff) as f64 / 2147483647.0;
        2.0 * x - 1.0  // uniform [-1, 1]
    }).collect();

    let d = correlation_dimension(&data, 3, 1);

    if d.is_nan() { return; }  // skip if insufficient data

    assert!(d >= 0.0,
        "correlation_dimension should be ≥ 0, got {}", d);
}

// ═══════════════════════════════════════════════════════════════════════════
// correlation_dimension — Test 3: wrong identity bites non-trivial geometry
// ═══════════════════════════════════════════════════════════════════════════

/// The L∞ distance fold starts from 0.0 instead of NEG_INFINITY.
/// For the L∞ norm of a difference vector v = xi - xj:
///   correct: fold(NEG_INFINITY, max) over |v_k| → max |v_k|
///   buggy:   fold(0.0, max) over |v_k| → max(0.0, |v_k|) = |v_k| (coincidentally correct)
///
/// Wait — the fold is over |v_k|, all of which are ≥ 0. So fold(0.0, max) over non-negative
/// values gives the same result as fold(NEG_INFINITY, max) over non-negative values.
/// The WRONG identity only bites when a value is NaN (eaten) or when the slice is EMPTY.
///
/// Test the empty-slice case: m=0 (embedding dimension 0) would give empty difference vectors.
/// Since m=0 is guarded (n_vectors < 10 would catch it for empty data), test m=1 directly.
#[test]
fn correlation_dimension_m1_gives_valid_result() {
    // m=1, tau=1: embedding is just the original series
    // L∞ distance in 1D = |xi - xj| (absolute difference)
    // fold(0.0, max) over {|xi - xj|} for m=1 gives the same as fold(NEG_INF, max)
    // since the difference is ≥ 0.
    // This confirms the wrong-identity does NOT bite for NaN-free, m≥1 inputs.
    let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();

    let d = correlation_dimension(&data, 1, 1);

    // D should be between 0 and 2 for a 1D time series
    if d.is_nan() { return; }
    assert!(d >= 0.0 && d <= 3.0,
        "correlation_dimension (m=1) should be in [0, 3], got {}", d);
}
