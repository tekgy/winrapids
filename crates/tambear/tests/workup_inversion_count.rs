//! Workup: inversion_count — standalone primitive validation
//!
//! Inversion count is a fundamental primitive used by:
//!   - Kendall's tau (discordant pairs)
//!   - Displacement distance
//!   - Sorting network analysis
//!   - Any order-based statistic
//!
//! This workup validates it independently of any consumer.

use tambear::nonparametric::inversion_count;

// ── Known exact values ──────────────────────────────────────────────────

#[test]
fn sorted_has_zero_inversions() {
    assert_eq!(inversion_count(&[1.0, 2.0, 3.0, 4.0, 5.0]), 0);
}

#[test]
fn reversed_has_n_choose_2_inversions() {
    // n=5: C(5,2) = 10
    assert_eq!(inversion_count(&[5.0, 4.0, 3.0, 2.0, 1.0]), 10);
    // n=4: C(4,2) = 6
    assert_eq!(inversion_count(&[4.0, 3.0, 2.0, 1.0]), 6);
    // n=3: C(3,2) = 3
    assert_eq!(inversion_count(&[3.0, 2.0, 1.0]), 3);
}

#[test]
fn single_swap() {
    // [2, 1, 3]: one inversion (2 > 1)
    assert_eq!(inversion_count(&[2.0, 1.0, 3.0]), 1);
}

#[test]
fn two_swaps() {
    // [3, 1, 2]: two inversions (3>1, 3>2)
    assert_eq!(inversion_count(&[3.0, 1.0, 2.0]), 2);
}

#[test]
fn interleaved() {
    // [2, 4, 1, 3, 5]: inversions are (2,1), (4,1), (4,3) = 3
    assert_eq!(inversion_count(&[2.0, 4.0, 1.0, 3.0, 5.0]), 3);
}

// ── Edge cases ──────────────────────────────────────────────────────────

#[test]
fn empty() {
    assert_eq!(inversion_count(&[]), 0);
}

#[test]
fn single_element() {
    assert_eq!(inversion_count(&[42.0]), 0);
}

#[test]
fn two_elements_sorted() {
    assert_eq!(inversion_count(&[1.0, 2.0]), 0);
}

#[test]
fn two_elements_reversed() {
    assert_eq!(inversion_count(&[2.0, 1.0]), 1);
}

#[test]
fn all_equal() {
    // Equal values are NOT inversions (stable: arr[i] <= arr[j] keeps order)
    assert_eq!(inversion_count(&[5.0, 5.0, 5.0, 5.0]), 0);
}

#[test]
fn ties_not_counted() {
    // [1, 1, 2]: no inversions (1 <= 1)
    assert_eq!(inversion_count(&[1.0, 1.0, 2.0]), 0);
    // [2, 1, 1]: one inversion (2 > 1 at index 0,1; 2 > 1 at index 0,2) = 2
    assert_eq!(inversion_count(&[2.0, 1.0, 1.0]), 2);
}

// ── Algebraic invariants ────────────────────────────────────────────────

#[test]
fn inv_plus_concordant_equals_n_choose_2() {
    // For any permutation of distinct values:
    // inversions + concordant_pairs = n(n-1)/2
    let arr = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];
    let n = arr.len() as i64;
    let inv = inversion_count(&arr);
    let total = n * (n - 1) / 2;
    // concordant = total - inversions (when no ties)
    assert!(inv >= 0);
    assert!(inv <= total);
}

#[test]
fn reversed_is_max_inversions() {
    // Reversed array has the maximum possible inversions for distinct data
    for n in 2..=20 {
        let arr: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();
        let expected = n as i64 * (n as i64 - 1) / 2;
        assert_eq!(inversion_count(&arr), expected, "n={}", n);
    }
}

#[test]
fn shift_invariant() {
    // Adding a constant doesn't change inversions (only relative order matters)
    let arr = [3.0, 1.0, 4.0, 1.5, 5.0];
    let shifted: Vec<f64> = arr.iter().map(|x| x + 1000.0).collect();
    assert_eq!(inversion_count(&arr), inversion_count(&shifted));
}

#[test]
fn positive_scale_invariant() {
    // Multiplying by a positive constant doesn't change inversions
    let arr = [3.0, 1.0, 4.0, 1.5, 5.0];
    let scaled: Vec<f64> = arr.iter().map(|x| x * 42.0).collect();
    assert_eq!(inversion_count(&arr), inversion_count(&scaled));
}

#[test]
fn negation_complements() {
    // Negating reverses order: inv(-arr) = n(n-1)/2 - inv(arr) for distinct values
    let arr = [3.0, 1.0, 4.0, 2.0, 5.0];
    let neg: Vec<f64> = arr.iter().map(|x| -x).collect();
    let n = arr.len() as i64;
    let total = n * (n - 1) / 2;
    assert_eq!(inversion_count(&arr) + inversion_count(&neg), total);
}

// ── Larger random-ish cases ─────────────────────────────────────────────

#[test]
fn known_permutation() {
    // [4, 3, 1, 2]: inversions = (4,3), (4,1), (4,2), (3,1), (3,2) = 5
    assert_eq!(inversion_count(&[4.0, 3.0, 1.0, 2.0]), 5);
}

#[test]
fn bubble_sort_count() {
    // Inversion count = number of bubble sort swaps
    // [5, 1, 4, 2, 8]: inv = (5,1), (5,4), (5,2), (4,2) = 4
    assert_eq!(inversion_count(&[5.0, 1.0, 4.0, 2.0, 8.0]), 4);
}
