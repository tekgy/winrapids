//! Adversarial tests for correlation variants:
//! biserial, rank-biserial, tetrachoric

use tambear::nonparametric::*;

// ═══════════════════════════════════════════════════════════════════════════
// BISERIAL CORRELATION
// ═══════════════════════════════════════════════════════════════════════════

/// Biserial ≥ |point-biserial| for the same data.
/// Biserial assumes latent continuous, so it's always at least as large in magnitude.
#[test]
fn biserial_greater_equal_point_biserial() {
    let binary = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let cont = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.5, 5.5, 1.5, 4.5];
    let r_b = biserial_correlation(&binary, &cont);
    let r_pb = point_biserial(&binary, &cont);
    assert!(r_b.is_finite() && r_pb.is_finite());
    assert!(r_b.abs() >= r_pb.abs() - 1e-10,
        "Biserial |{}| should be >= |point-biserial| {}", r_b, r_pb);
}

/// Biserial with all-same binary: NaN.
#[test]
fn biserial_all_same_binary() {
    let binary = vec![1.0; 10];
    let cont = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let r = biserial_correlation(&binary, &cont);
    assert!(r.is_nan(), "Biserial all-same binary should be NaN, got {}", r);
}

/// Biserial with constant continuous: NaN (zero variance).
#[test]
fn biserial_constant_continuous() {
    let binary = vec![0.0, 0.0, 1.0, 1.0];
    let cont = vec![5.0, 5.0, 5.0, 5.0];
    let r = biserial_correlation(&binary, &cont);
    assert!(r.is_nan(), "Biserial constant continuous should be NaN, got {}", r);
}

/// Biserial with perfect separation (high group all above low group).
#[test]
fn biserial_perfect_separation() {
    let binary = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let cont = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let r = biserial_correlation(&binary, &cont);
    assert!(r.is_finite() && r > 0.9,
        "Biserial perfect separation should be strongly positive, got {}", r);
}

/// Biserial with single element: NaN.
#[test]
fn biserial_too_few() {
    let r = biserial_correlation(&[1.0], &[2.0]);
    assert!(r.is_nan(), "Biserial n=1 should be NaN, got {}", r);
}

/// Biserial with reversed association: negative.
#[test]
fn biserial_negative() {
    let binary = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let cont = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let r = biserial_correlation(&binary, &cont);
    assert!(r < 0.0, "Biserial reversed should be negative, got {}", r);
}

// ═══════════════════════════════════════════════════════════════════════════
// RANK-BISERIAL (Glass 1966)
// ═══════════════════════════════════════════════════════════════════════════

/// Rank-biserial with perfect separation: r = ±1.
#[test]
fn rank_biserial_perfect_separation() {
    let x = vec![10.0, 11.0, 12.0, 13.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let r = rank_biserial(&x, &y);
    // All x values > all y values → U (smaller) = 0 → r = 1 - 0 = 1
    assert!((r.abs() - 1.0).abs() < 1e-6,
        "Rank-biserial perfect separation should be ±1, got {}", r);
}

/// Rank-biserial with identical groups: r ≈ 0.
#[test]
fn rank_biserial_identical() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let r = rank_biserial(&x, &y);
    // With ties, U ≈ n1*n2/2 → r ≈ 0
    assert!(r.abs() < 0.5,
        "Rank-biserial identical groups should be near 0, got {}", r);
}

/// Rank-biserial range: [-1, 1].
#[test]
fn rank_biserial_range() {
    let x = vec![1.0, 5.0, 3.0, 7.0, 2.0];
    let y = vec![4.0, 6.0, 2.0, 8.0, 3.0];
    let r = rank_biserial(&x, &y);
    assert!(r.is_finite());
    assert!(r >= -1.0 && r <= 1.0,
        "Rank-biserial should be in [-1,1], got {}", r);
}

/// Rank-biserial with empty group: NaN.
#[test]
fn rank_biserial_empty() {
    let r = rank_biserial(&[], &[1.0, 2.0]);
    assert!(r.is_nan(), "Rank-biserial empty group should be NaN, got {}", r);
}

// ═══════════════════════════════════════════════════════════════════════════
// TETRACHORIC CORRELATION
// ═══════════════════════════════════════════════════════════════════════════

/// Tetrachoric on independent data (balanced 2x2): ≈ 0.
#[test]
fn tetrachoric_independent() {
    // Balanced table: no association between rows and columns
    let r = tetrachoric(&[10.0, 10.0, 10.0, 10.0]);
    assert!(r.abs() < 0.01,
        "Tetrachoric independent should be ~0, got {}", r);
}

/// Tetrachoric on strong diagonal: positive.
#[test]
fn tetrachoric_positive() {
    // Strong diagonal (both variables agree)
    let r = tetrachoric(&[40.0, 5.0, 5.0, 40.0]);
    assert!(r > 0.5,
        "Tetrachoric strong diagonal should be positive, got {}", r);
}

/// Tetrachoric on strong anti-diagonal: negative.
#[test]
fn tetrachoric_negative() {
    let r = tetrachoric(&[5.0, 40.0, 40.0, 5.0]);
    assert!(r < -0.5,
        "Tetrachoric anti-diagonal should be negative, got {}", r);
}

/// Tetrachoric with zero off-diagonal cell b: perfect positive association.
#[test]
fn tetrachoric_zero_b() {
    let r = tetrachoric(&[10.0, 0.0, 5.0, 10.0]);
    assert!((r - 1.0).abs() < 0.01,
        "Tetrachoric with b=0 should be 1.0 (perfect pos), got {}", r);
}

/// Tetrachoric with zero diagonal cell a: perfect negative association.
#[test]
fn tetrachoric_zero_a() {
    let r = tetrachoric(&[0.0, 10.0, 10.0, 0.0]);
    // ad = 0, bc = 100 → ratio = 0 → cos(π/(1+0)) = cos(π) = -1
    assert!((r - (-1.0)).abs() < 0.01,
        "Tetrachoric with a=0 should be -1.0, got {}", r);
}

/// Tetrachoric range: [-1, 1].
#[test]
fn tetrachoric_range() {
    for table in &[[20.0, 5.0, 10.0, 15.0], [1.0, 1.0, 1.0, 1.0], [100.0, 1.0, 1.0, 100.0]] {
        let r = tetrachoric(table);
        if r.is_finite() {
            assert!(r >= -1.0 && r <= 1.0,
                "Tetrachoric should be in [-1,1] for {:?}, got {}", table, r);
        }
    }
}
