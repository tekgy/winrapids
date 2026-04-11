use super::*;

#[test]
fn mean_arithmetic_has_two_accumulates_one_gather() {
    let acc = MEAN_ARITHMETIC.accumulate_steps();
    assert_eq!(acc.len(), 2); // sum + count
    let gathers: Vec<_> = MEAN_ARITHMETIC.steps.iter()
        .filter(|s| matches!(s, Step::Gather { .. })).collect();
    assert_eq!(gathers.len(), 1);
}

#[test]
fn mean_and_variance_share_sum_and_count() {
    let shared = MEAN_ARITHMETIC.shared_with(&VARIANCE);
    // Both have: Accumulate(All, Value, Add) → "sum"
    // Both have: Accumulate(All, One, Add) → "count"
    assert_eq!(shared, 2, "mean and variance share sum + count");
}

#[test]
fn mean_geometric_has_ln_expr() {
    let acc = MEAN_GEOMETRIC.accumulate_steps();
    let has_ln = acc.iter().any(|s| {
        matches!(s, Step::Accumulate { expr: ExprKind::Ln, .. })
    });
    assert!(has_ln, "geometric mean should accumulate ln(x)");
}

#[test]
fn mean_harmonic_has_reciprocal_expr() {
    let acc = MEAN_HARMONIC.accumulate_steps();
    let has_recip = acc.iter().any(|s| {
        matches!(s, Step::Accumulate { expr: ExprKind::Reciprocal, .. })
    });
    assert!(has_recip, "harmonic mean should accumulate 1/x");
}

#[test]
fn cumsum_is_prefix_scan() {
    let acc = CUMSUM.accumulate_steps();
    assert_eq!(acc.len(), 1);
    assert!(matches!(acc[0],
        Step::Accumulate { grouping: GroupingKind::Prefix, op: OpKind::Add, .. }
    ));
}

#[test]
fn variance_shares_sum_sq_with_rms() {
    let shared = VARIANCE.shared_with(&MEAN_QUADRATIC);
    // Both have: Accumulate(All, ValueSq, Add) → sum_sq/sum_sq
    // Both have: Accumulate(All, One, Add) → count
    assert_eq!(shared, 2, "variance and RMS share sum_sq + count");
}

#[test]
fn no_sharing_between_unrelated() {
    let shared = CUMSUM.shared_with(&MEAN_HARMONIC);
    assert_eq!(shared, 0, "cumsum (Prefix) shares nothing with harmonic (All)");
}
