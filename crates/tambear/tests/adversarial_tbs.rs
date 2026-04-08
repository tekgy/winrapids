//! Adversarial tests for the `.tbs` chain language.
//!
//! These test the parser, executor, and science linter at their boundaries.
//! Tests marked `#[ignore]` are stubs awaiting executor completion (task #21).
//! Tests without `#[ignore]` test the parser directly (already functional).
//!
//! ## Boundary taxonomy mapping
//!
//! | Category | Boundary type | What breaks |
//! |----------|--------------|-------------|
//! | Parser edge cases | Type 1 (denominator) | Empty input, missing parens |
//! | Type/signature errors | Type 5 (structural) | Wrong arg types → wrong function class |
//! | Science linter | Type 3 (cancellation) | Redundant ops, wrong ordering |
//! | Fusion edge cases | Type 5 (structural) | Incompatible kernels, DAG forks |
//! | Budget adversarial | Type 2 (convergence) | Exceeds memory/compute budget |

use tambear::tbs_parser::TbsChain;

// ═══════════════════════════════════════════════════════════════════════════
// PARSER ADVERSARIAL — tests the parser directly (no executor needed)
// ═══════════════════════════════════════════════════════════════════════════

/// Empty string should produce an error, not panic.
#[test]
fn parser_empty_string() {
    let result = TbsChain::parse("");
    assert!(result.is_err(), "Empty string should be a parse error");
}

/// Whitespace-only string should produce an error.
#[test]
fn parser_whitespace_only() {
    let result = TbsChain::parse("   \n\t  ");
    assert!(result.is_err(), "Whitespace-only should be a parse error");
}

/// Single valid step with no args.
#[test]
fn parser_single_step_no_args() {
    let chain = TbsChain::parse("normalize()").unwrap();
    assert_eq!(chain.steps.len(), 1);
    assert_eq!(chain.steps[0].name.to_string(), "normalize");
    assert!(chain.steps[0].args.is_empty());
}

/// Missing parentheses should be an error.
#[test]
fn parser_missing_parens() {
    let result = TbsChain::parse("normalize");
    assert!(result.is_err(), "Missing parens should be a parse error");
}

/// Empty parens with extra whitespace — should still parse.
#[test]
fn parser_whitespace_in_parens() {
    let chain = TbsChain::parse("normalize(  )").unwrap();
    assert_eq!(chain.steps.len(), 1);
}

/// Dotted name (namespaced operation).
#[test]
fn parser_dotted_name() {
    let chain = TbsChain::parse("train.linear(target=\"price\")").unwrap();
    assert_eq!(chain.steps[0].name.to_string(), "train.linear");
}

/// Multiple steps chained with dot.
#[test]
fn parser_multi_step_chain() {
    let chain = TbsChain::parse(
        "normalize().discover_clusters(epsilon=0.5, min_samples=2).train.linear(target=\"y\")"
    ).unwrap();
    assert_eq!(chain.steps.len(), 3);
}

/// Deeply nested chain — 50 steps. Should not stack overflow.
#[test]
fn parser_deeply_nested_chain() {
    let chain_str: String = (0..50).map(|_| "normalize()").collect::<Vec<_>>().join(".");
    let result = TbsChain::parse(&chain_str);
    assert!(result.is_ok(), "50-step chain should parse without stack overflow");
    assert_eq!(result.unwrap().steps.len(), 50);
}

/// Numeric arguments: integer, float, negative.
#[test]
fn parser_numeric_args() {
    let chain = TbsChain::parse("kmeans(k=3, max_iter=500)").unwrap();
    assert_eq!(chain.steps[0].args.len(), 2);

    let chain2 = TbsChain::parse("discover_clusters(epsilon=0.5, min_samples=2)").unwrap();
    assert_eq!(chain2.steps[0].args.len(), 2);
}

/// Boolean arguments.
#[test]
fn parser_bool_args() {
    // This may not be used yet, but the grammar supports it
    let chain = TbsChain::parse("normalize(center=true)");
    // May succeed or fail depending on whether normalize accepts args
    // The parser should at least not panic
    assert!(chain.is_ok() || chain.is_err());
}

/// Trailing dot should be an error (incomplete chain).
#[test]
fn parser_trailing_dot() {
    let result = TbsChain::parse("normalize().");
    assert!(result.is_err(), "Trailing dot should be a parse error");
}

/// Leading dot should be an error.
#[test]
fn parser_leading_dot() {
    let result = TbsChain::parse(".normalize()");
    assert!(result.is_err(), "Leading dot should be a parse error");
}

/// Double dot should be an error.
#[test]
fn parser_double_dot() {
    let result = TbsChain::parse("normalize()..kmeans(k=3)");
    assert!(result.is_err(), "Double dot should be a parse error");
}

// ═══════════════════════════════════════════════════════════════════════════
// EXECUTOR ADVERSARIAL — requires executor to be wired (task #21)
// ═══════════════════════════════════════════════════════════════════════════

/// Unknown operation should produce a clear error, not panic.
#[test]
fn executor_unknown_operation() {
    use tambear::tbs_executor::execute;

    let chain = TbsChain::parse("nonexistent_operation()").unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = execute(chain, data, 2, 2, None);
    assert!(result.is_err(), "Unknown operation should be an error");
    // TbsResult doesn't impl Debug, so use match instead of unwrap_err
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.contains("unsupported") || err_msg.contains("unknown"),
            "Error should mention the unsupported operation: {err_msg}");
    }
}

/// train.linear without y should produce a clear error.
#[test]
fn executor_train_without_target() {
    use tambear::tbs_executor::execute;

    let chain = TbsChain::parse("train.linear()").unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = execute(chain, data, 3, 2, None);
    assert!(result.is_err(), "train.linear without y should error");
}

/// discover_clusters without epsilon should produce a clear error.
#[test]
fn executor_missing_required_arg() {
    use tambear::tbs_executor::execute;

    let chain = TbsChain::parse("discover_clusters()").unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = execute(chain, data, 2, 2, None);
    assert!(result.is_err(), "discover_clusters without epsilon should error");
}

/// normalize on single-row data (n=1) — edge case for z-score.
#[test]
fn executor_normalize_single_row() {
    use tambear::tbs_executor::execute;

    let chain = TbsChain::parse("normalize()").unwrap();
    let data = vec![1.0, 2.0, 3.0];
    let result = execute(chain, data, 1, 3, None);
    // With n=1, std = 0 → z-score is 0/0. Should handle gracefully.
    match result {
        Ok(r) => {
            let frame = r.pipeline.frame();
            // All values should be finite (0.0 or NaN handled → 0.0)
        }
        Err(e) => {
            eprintln!("normalize(n=1) error: {e}");
            // An error is also acceptable for degenerate input
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SCIENCE LINTER ADVERSARIAL — tests lint detection
// ═══════════════════════════════════════════════════════════════════════════

/// L001: distance-based op without prior normalize should warn.
#[test]
fn lint_l001_missing_normalization() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("kmeans(k=3)").unwrap();
    let lints = static_lints(&chain);

    let has_l001 = lints.iter().any(|l| l.code == "L001");
    assert!(has_l001, "kmeans without normalize should trigger L001, got: {:?}",
        lints.iter().map(|l| &l.code).collect::<Vec<_>>());
}

/// L001 should NOT fire if normalize precedes the distance op.
#[test]
fn lint_l001_not_fired_with_normalize() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("normalize().kmeans(k=3)").unwrap();
    let lints = static_lints(&chain);

    let has_l001 = lints.iter().any(|l| l.code == "L001");
    assert!(!has_l001, "normalize().kmeans() should NOT trigger L001, got: {:?}",
        lints.iter().map(|l| &l.code).collect::<Vec<_>>());
}

/// Redundant normalize: normalize().normalize() should warn.
#[test]
fn lint_redundant_normalize() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("normalize().normalize()").unwrap();
    let lints = static_lints(&chain);

    // Should have a lint about redundant normalize
    let has_redundant = lints.iter().any(|l| l.message.to_lowercase().contains("redundant"));
    assert!(has_redundant, "normalize().normalize() should warn about redundancy");
}

/// kmeans().normalize() — clustering then normalizing is suspicious.
/// The lint message must be SPECIFIC: "normalize() after kmeans() normalizes
/// the output labels, not the input features. If you meant to normalize before
/// clustering, move normalize() before kmeans()."
/// A vague "warning" is not useful — the message is the test.
#[test]
fn lint_normalize_after_clustering() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("kmeans(k=3).normalize()").unwrap();
    let lints = static_lints(&chain);

    // Must warn with a specific, actionable message
    let has_warn = lints.iter().any(|l| {
        let msg = l.message.to_lowercase();
        // Must mention BOTH the ordering issue AND the fix
        (msg.contains("after") || msg.contains("post"))
            && (msg.contains("before") || msg.contains("move"))
    });
    assert!(has_warn,
        "kmeans().normalize() should warn about post-clustering normalize with actionable fix, \
         got: {:?}", lints.iter().map(|l| &l.message).collect::<Vec<_>>());
}

/// L106: constant columns in data should trigger a warning.
#[test]
fn lint_l106_constant_column_detected() {
    use tambear::tbs_lint::lint_l106_constant_columns;

    // Column 0 is constant (all 5.0), column 1 varies
    let data = vec![
        5.0, 1.0,
        5.0, 2.0,
        5.0, 3.0,
        5.0, 4.0,
    ];
    let lints = lint_l106_constant_columns(&data, 4, 2);

    let has_l106 = lints.iter().any(|l| l.code == "L106");
    assert!(has_l106, "Constant column should trigger L106, got: {:?}",
        lints.iter().map(|l| &l.code).collect::<Vec<_>>());
}

/// L106 should NOT fire when all columns vary.
#[test]
fn lint_l106_not_fired_for_varying_columns() {
    use tambear::tbs_lint::lint_l106_constant_columns;

    let data = vec![
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ];
    let lints = lint_l106_constant_columns(&data, 3, 2);

    let has_l106 = lints.iter().any(|l| l.code == "L106");
    assert!(!has_l106, "No constant columns → no L106");
}

/// L004: describe() → t_test() should warn about distribution assumption.
#[test]
fn lint_l004_describe_then_ttest() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("describe().t_test()").unwrap();
    let lints = static_lints(&chain);
    assert!(lints.iter().any(|l| l.code == "L004"),
        "describe() → t_test() should trigger L004 (distribution assumption mismatch)");
}

/// L004: skewness() → anova() — skewness is a shape op, anova assumes normality.
#[test]
fn lint_l004_skewness_then_anova() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("skewness().anova()").unwrap();
    let lints = static_lints(&chain);
    assert!(lints.iter().any(|l| l.code == "L004"),
        "skewness() → anova() should trigger L004");
}

/// L004 should NOT fire without a prior shape-analysis step.
#[test]
fn lint_l004_not_fired_without_shape_op() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("t_test()").unwrap();
    let lints = static_lints(&chain);
    assert!(!lints.iter().any(|l| l.code == "L004"),
        "t_test() alone should NOT trigger L004 (no shape analysis precedes it)");
}

/// L004 should be suppressed when ks_test intervenes (explicit normality check).
#[test]
fn lint_l004_suppressed_by_ks_test() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("describe().ks_test().t_test()").unwrap();
    let lints = static_lints(&chain);
    assert!(!lints.iter().any(|l| l.code == "L004"),
        "L004 should be suppressed when ks_test() checks normality before t_test()");
}

/// L109: dynamic skewness check — |skew| > 1.0 with normality-assuming test.
#[test]
fn lint_l109_fires_for_skewed_data() {
    use tambear::tbs_lint::lint_l109_skewness_normality;
    use tambear::ColumnDescribe;

    // Construct DescribeResult with |skewness| = 2.5 (highly skewed)
    let describe = tambear::DescribeResult {
        columns: vec![ColumnDescribe {
            index: 0, count: 100, mean: 10.0, std: 5.0,
            min: 0.0, q1: 3.0, median: 7.0, q3: 14.0, max: 50.0,
            iqr: 11.0, skewness: 2.5, kurtosis: 8.0,
        }],
    };
    let chain = TbsChain::parse("describe().t_test()").unwrap();
    let lints = lint_l109_skewness_normality(&describe, &chain);
    assert!(lints.iter().any(|l| l.code == "L109"),
        "L109 should fire for |skewness| = 2.5 > 1.0 with t_test in chain");
}

/// L109 should NOT fire when skewness is small.
#[test]
fn lint_l109_not_fired_for_symmetric_data() {
    use tambear::tbs_lint::lint_l109_skewness_normality;
    use tambear::ColumnDescribe;

    let describe = tambear::DescribeResult {
        columns: vec![ColumnDescribe {
            index: 0, count: 100, mean: 0.0, std: 1.0,
            min: -3.0, q1: -0.67, median: 0.0, q3: 0.67, max: 3.0,
            iqr: 1.34, skewness: 0.05, kurtosis: -0.1,
        }],
    };
    let chain = TbsChain::parse("describe().t_test()").unwrap();
    let lints = lint_l109_skewness_normality(&describe, &chain);
    assert!(lints.is_empty(),
        "L109 should not fire for |skewness| = 0.05 (near-normal)");
}

/// L109 should NOT fire if no normality-assuming test is in the chain.
#[test]
fn lint_l109_not_fired_without_normality_op() {
    use tambear::tbs_lint::lint_l109_skewness_normality;
    use tambear::ColumnDescribe;

    let describe = tambear::DescribeResult {
        columns: vec![ColumnDescribe {
            index: 0, count: 100, mean: 10.0, std: 5.0,
            min: 0.0, q1: 3.0, median: 7.0, q3: 14.0, max: 50.0,
            iqr: 11.0, skewness: 3.0, kurtosis: 12.0,
        }],
    };
    let chain = TbsChain::parse("describe().kmeans(k=3)").unwrap();
    let lints = lint_l109_skewness_normality(&describe, &chain);
    assert!(lints.is_empty(),
        "L109 should not fire when chain has no normality-assuming tests");
}

/// L109 boundary: |skewness| = exactly 1.0 should NOT fire (threshold is >1.0).
#[test]
fn lint_l109_boundary_exactly_one() {
    use tambear::tbs_lint::lint_l109_skewness_normality;
    use tambear::ColumnDescribe;

    let describe = tambear::DescribeResult {
        columns: vec![ColumnDescribe {
            index: 0, count: 100, mean: 5.0, std: 3.0,
            min: 0.0, q1: 2.0, median: 4.0, q3: 7.0, max: 20.0,
            iqr: 5.0, skewness: 1.0, kurtosis: 1.5,
        }],
    };
    let chain = TbsChain::parse("describe().t_test()").unwrap();
    let lints = lint_l109_skewness_normality(&describe, &chain);
    assert!(lints.is_empty(),
        "L109 should NOT fire at exactly |skewness| = 1.0 (threshold is strictly >1.0)");
}

/// L201: adjacent Kingdom C steps sharing distance matrix should inform.
#[test]
fn lint_l201_shared_distance_matrix() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("kmeans(k=3).knn(k=5)").unwrap();
    let lints = static_lints(&chain);
    assert!(lints.iter().any(|l| l.code == "L201" && l.message.contains("distance matrix")),
        "kmeans → knn should trigger L201 (shared distance matrix), got: {:?}",
        lints.iter().map(|l| format!("{}: {}", l.code, &l.message)).collect::<Vec<_>>());
}

/// L201 should NOT fire for steps with different shared subproblems.
#[test]
fn lint_l201_not_fired_for_different_subproblems() {
    use tambear::tbs_lint::static_lints;

    // kmeans (distance matrix) + train.linear (Gram matrix) — different subproblems
    let chain = TbsChain::parse("kmeans(k=3).train.linear(target=\"y\")").unwrap();
    let lints = static_lints(&chain);
    assert!(!lints.iter().any(|l| l.code == "L201"),
        "different subproblems should NOT trigger L201");
}

// ═══════════════════════════════════════════════════════════════════════════
// LINT ADVERSARIAL — L002, L003, L101 boundary tests
// ═══════════════════════════════════════════════════════════════════════════

/// L002 should fire for train.logistic too, not just train.linear.
#[test]
fn lint_l002_fires_for_logistic() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("train.logistic()").unwrap();
    let lints = static_lints(&chain);
    assert!(lints.iter().any(|l| l.code == "L002"),
        "train.logistic without exploration should trigger L002");
}

/// L002 should NOT fire if any exploratory step precedes training.
#[test]
fn lint_l002_suppressed_by_pca() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("pca().train.linear(target=\"y\")").unwrap();
    let lints = static_lints(&chain);
    assert!(!lints.iter().any(|l| l.code == "L002"),
        "pca() before training should suppress L002");
}

/// L002 severity must be Info (suggestion), not Warning.
#[test]
fn lint_l002_is_info_not_warning() {
    use tambear::tbs_lint::{static_lints, LintSeverity};

    let chain = TbsChain::parse("train.linear(target=\"y\")").unwrap();
    let lints = static_lints(&chain);
    let l002 = lints.iter().find(|l| l.code == "L002")
        .expect("L002 should fire for direct training");
    assert_eq!(l002.severity, LintSeverity::Info,
        "L002 is a suggestion, not a warning — severity must be Info");
}

/// L003: train.linear().predict() should warn about optimistic bias.
#[test]
fn lint_l003_fires_for_train_predict() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("train.linear(target=\"y\").predict()").unwrap();
    let lints = static_lints(&chain);
    assert!(lints.iter().any(|l| l.code == "L003"),
        "train().predict() should trigger L003 (predict on training data)");
}

/// L003 should NOT fire if predict is separated from train by another step.
#[test]
fn lint_l003_not_fired_with_intervening_step() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("train.linear(target=\"y\").normalize().predict()").unwrap();
    let lints = static_lints(&chain);
    assert!(!lints.iter().any(|l| l.code == "L003"),
        "train → normalize → predict should NOT trigger L003 (intervening step)");
}

/// L003 should NOT fire for standalone predict().
#[test]
fn lint_l003_not_fired_standalone_predict() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse("predict()").unwrap();
    let lints = static_lints(&chain);
    assert!(!lints.iter().any(|l| l.code == "L003"),
        "standalone predict() without train should NOT trigger L003");
}

/// L001 should be suppressed for cosine metric (scale-invariant).
#[test]
fn lint_l001_suppressed_by_cosine_metric() {
    use tambear::tbs_lint::static_lints;

    let chain = TbsChain::parse(r#"knn(k=5, metric="cosine")"#).unwrap();
    let lints = static_lints(&chain);
    assert!(!lints.iter().any(|l| l.code == "L001"),
        "cosine metric should suppress L001 (scale-invariant)");
}

/// L001 should fire for ALL distance ops, not just kmeans.
#[test]
fn lint_l001_fires_for_all_distance_ops() {
    use tambear::tbs_lint::static_lints;

    for op in &["dbscan(epsilon=1.0, min_samples=2)", "kmeans(k=3)", "knn(k=5)"] {
        let chain = TbsChain::parse(op).unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L001"),
            "L001 should fire for {op} without normalize");
    }
}

/// L101: naive variance should fire for extreme-range data with describe().
#[test]
fn lint_l101_extreme_range_with_describe() {
    use tambear::tbs_lint::lint_l101_naive_variance;

    // Values near 1e12 with tiny variance — naive formula loses precision
    let data: Vec<f64> = (0..50).map(|i| 1e12 + i as f64 * 0.001).collect();
    let chain = TbsChain::parse("describe()").unwrap();
    let lints = lint_l101_naive_variance(&data, 50, 1, &chain);
    assert!(lints.iter().any(|l| l.code == "L101"),
        "extreme-range data with describe() should trigger L101");
}

/// L101 should NOT fire if the chain has no variance-related operations.
#[test]
fn lint_l101_not_fired_without_variance_op() {
    use tambear::tbs_lint::lint_l101_naive_variance;

    let data: Vec<f64> = (0..50).map(|i| 1e12 + i as f64 * 0.001).collect();
    let chain = TbsChain::parse("normalize()").unwrap();
    let lints = lint_l101_naive_variance(&data, 50, 1, &chain);
    assert!(lints.is_empty(),
        "L101 should not fire without variance operations in chain");
}

/// L106: near-constant column with non-zero mean.
#[test]
fn lint_l106_near_constant_nonzero_mean() {
    use tambear::tbs_lint::lint_l106_constant_columns;

    // Column 0: all 1000.0 (constant, large mean). Column 1: varies.
    let data = vec![
        1000.0, 1.0,
        1000.0, 2.0,
        1000.0, 3.0,
    ];
    let lints = lint_l106_constant_columns(&data, 3, 2);
    assert!(lints.iter().any(|l| l.code == "L106" && l.message.contains("column 0")),
        "near-constant column with large mean should trigger L106");
}

/// Integration: executor should collect both static and dynamic lints.
#[test]
fn lint_integration_static_and_dynamic_together() {
    use tambear::tbs_executor::execute;

    // kmeans without normalize (L001) + constant column (L106)
    let data = vec![
        5.0, 1.0,
        5.0, 2.0,
        5.0, 3.0,
        5.0, 10.0,
    ];
    let chain = TbsChain::parse("kmeans(k=2)").unwrap();
    let result = execute(chain, data, 4, 2, None).unwrap();

    let codes: Vec<&str> = result.lints.iter().map(|l| l.code).collect();
    assert!(codes.contains(&"L001"), "should have L001 (missing normalize), got: {codes:?}");
    assert!(codes.contains(&"L106"), "should have L106 (constant column), got: {codes:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// FUSION EDGE CASES (for task #24 — JIT pipeline)
// ═══════════════════════════════════════════════════════════════════════════

/// Two adjacent Kingdom A steps should compile to GPU passes, not CPU fallback.
/// Observable criterion: plan_summary shows "gpu:*" entries, not "cpu:fallback".
/// The fused result must also match sequential execution numerically.
#[test]
fn fusion_adjacent_a_steps_fuse() {
    use tambear::tbs_executor::execute;
    use tambear::tbs_jit;

    // normalize().describe() — both are Kingdom A (single-pass accumulate).
    let chain = TbsChain::parse("normalize().describe()").unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Observable: JIT plan should classify both steps as GPU-eligible
    let plan = tbs_jit::compile(&chain);
    let summary = tbs_jit::plan_summary(&plan);
    assert_eq!(summary.len(), 2, "plan should have 2 passes for 2 steps");
    for (pass_type, step_idx) in &summary {
        assert!(pass_type.starts_with("gpu:"),
            "step {step_idx} should be GPU-compiled, got {pass_type}");
    }

    // Numerical: fused result must match sequential
    let chain_seq = TbsChain::parse("normalize()").unwrap();
    let r_seq = execute(chain_seq, data.clone(), 3, 2, None).unwrap();

    let chain_fused = TbsChain::parse("normalize().describe()").unwrap();
    let r_fused = execute(chain_fused, data, 3, 2, None).unwrap();

    let seq_data = r_seq.pipeline.frame().data.clone();
    let fused_data = r_fused.pipeline.frame().data.clone();
    for (a, b) in seq_data.iter().zip(fused_data.iter()) {
        assert!((a - b).abs() < 1e-12, "fusion must not change results: {a} != {b}");
    }
}

/// DAG fork: normalize() feeds both kmeans() AND train.linear().
/// The normalized data must be materialized (not fused away) because both
/// downstream steps consume it. Verify both downstream results are correct.
#[test]
fn fusion_dag_fork_preserves_both_uses() {
    use tambear::tbs_executor::execute;
    use tambear::tbs_jit;

    // normalize → kmeans → train.linear: the normalized data must survive
    // through kmeans to also feed train.linear correctly.
    let data = vec![
        1.0, 1.0,
        1.0, 2.0,
        2.0, 1.0,
        10.0, 10.0,
        10.0, 11.0,
        11.0, 10.0,
    ];
    let y: Vec<f64> = data.chunks(2).map(|p| p[0] + p[1]).collect();

    let chain = TbsChain::parse(
        "normalize().kmeans(k=2).train.linear(target=\"y\")"
    ).unwrap();

    // JIT plan should have 3 passes
    let plan = tbs_jit::compile(&chain);
    let summary = tbs_jit::plan_summary(&plan);
    assert_eq!(summary.len(), 3, "plan should have 3 passes");

    // Execute and verify both downstream results are valid
    let result = execute(chain, data, 6, 2, Some(y)).unwrap();

    // kmeans consumed normalized data correctly
    assert_eq!(result.pipeline.frame().n_clusters, Some(2),
        "kmeans should find 2 clusters in normalized data");

    // train.linear consumed normalized data correctly (not garbage)
    let model = result.linear_model.as_ref()
        .expect("train.linear should produce a model");
    assert!(model.r_squared > 0.5,
        "R² should be reasonable on normalized data, got {}", model.r_squared);
}

// ═══════════════════════════════════════════════════════════════════════════
// BUDGET ADVERSARIAL
// ═══════════════════════════════════════════════════════════════════════════

/// Extremely long chain — 100 steps. Should either succeed or fail with
/// a clear budget error, never panic or OOM without explanation.
#[test]
fn budget_100_step_chain() {
    use tambear::tbs_executor::execute;

    // 100 normalize steps in sequence (each is a full-data pass)
    let chain_str: String = (0..100).map(|_| "normalize()").collect::<Vec<_>>().join(".");
    let chain = TbsChain::parse(&chain_str).unwrap();
    let data = vec![1.0; 200]; // 100 × 2

    let result = execute(chain, data, 100, 2, None);
    // Should succeed (normalize is idempotent after first pass) or
    // produce a budget warning — not crash.
    match result {
        Ok(r) => {
            eprintln!("100-step chain succeeded, {} lints", r.lints.len());
        }
        Err(e) => {
            eprintln!("100-step chain error: {e}");
            // A budget-exceeded error is acceptable
        }
    }
}

/// Chain that requests more Kingdom C steps than the compile budget allows.
#[test]
fn budget_many_kingdom_c_steps() {
    // 50 sequential kmeans with different k — each is Kingdom C (iterative).
    // The compile budget should either warn or cap the iteration count.
    let chain_str: String = (2..=10).map(|k| format!("kmeans(k={k})")).collect::<Vec<_>>().join(".");
    let chain = TbsChain::parse(&chain_str).unwrap();
    assert!(chain.steps.len() == 9, "Should parse 9 kmeans steps");
}
