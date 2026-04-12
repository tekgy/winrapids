//! Oracle runner integration tests (campsite 4.8).
//!
//! These tests verify the oracle runner infrastructure using `f64::exp` as the
//! calibration baseline (a known-good stand-in for `tam_exp` before the libm
//! implementation lands from Peak 2).
//!
//! ## What these tests prove
//!
//! 1. `load_oracle_entry` correctly parses `oracles/tam_exp.toml` including:
//!    - f64 literal corpus values
//!    - String expression corpus values ("inf", "-inf", "nan", "ln(2)", etc.)
//!    - Hex bit-pattern expected_bits ("0x0000000000000000")
//!    - "NONZERO_SUBNORMAL" pseudo-constraint
//!    - Identity check specs
//!
//! 2. `run_oracle` produces a correct `OracleReport` including:
//!    - Random sample ULP measurement (from exp-1k.bin)
//!    - Injection set ULP reports (NaN propagation, special values)
//!    - Bit-exact check results (signed-zero, NONZERO_SUBNORMAL class)
//!    - Identity check results (exp_negation, exp_one_returns_e)
//!
//! 3. The calibration baseline (`f64::exp`) passes the 1-ULP claimed bound.
//!
//! 4. The runner fails fast when the candidate is wrong: a deliberately broken
//!    candidate (returns 0.0 for all inputs) is detected as FAIL.
//!
//! ## Invariants
//!
//! - I9: `f64::exp` serves as calibration only — it is a PEER reference, not an
//!   oracle.  The oracle is mpmath via the exp-1k.bin reference file.
//!   When `tam_exp` is implemented, replace the `|x| x.exp()` candidate with the
//!   real implementation.
//! - I11: the oracle runner checks NaN propagation through the injection sets.

use tambear_tam_test_harness::oracle_runner::{load_oracle_entry, run_oracle};

/// Path to the expedition root (relative to the crate manifest dir).
/// CARGO_MANIFEST_DIR = R:\winrapids\crates\tambear-tam-test-harness
/// ../../ gets back to R:\winrapids, then campsites/expedition/...
const EXPEDITION_ROOT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../campsites/expedition/20260411120000-the-bit-exact-trek"
);

fn oracle_toml_path() -> String {
    format!("{}/oracles/tam_exp.toml", EXPEDITION_ROOT)
}

// ─────────────────────────────────────────────────────────────────────────────
// Load tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn oracle_entry_loads_without_error() {
    let entry = load_oracle_entry(&oracle_toml_path(), EXPEDITION_ROOT)
        .expect("oracle entry should load cleanly");

    assert_eq!(entry.function_name, "tam_exp");
    assert_eq!(entry.claimed_max_ulp, 1.0);
    assert!(!entry.reference_records.is_empty(),
        "reference records should be non-empty");

    // Injection sets: the TOML has special_values, cody_waite_exact, near_overflow, etc.
    assert!(!entry.injection_sets.is_empty(),
        "at least one injection set should be parsed");

    // Bit-exact checks
    assert!(!entry.bit_exact_checks.is_empty(),
        "bit-exact checks should be parsed");

    // Identity checks
    assert!(!entry.identity_checks.is_empty(),
        "identity checks should be parsed");

    // Verify the critical bit-exact check is present
    let exp_neg_inf_check = entry.bit_exact_checks.iter()
        .find(|c| c.name == "exp_neg_inf_is_positive_zero");
    assert!(exp_neg_inf_check.is_some(),
        "exp_neg_inf_is_positive_zero check must be present (adversarial B1 requirement)");

    // Verify NONZERO_SUBNORMAL is parsed (not errored)
    let subnormal_check = entry.bit_exact_checks.iter()
        .find(|c| c.name == "exp_subnormal_output_not_zero");
    assert!(subnormal_check.is_some(),
        "NONZERO_SUBNORMAL check must load without error");
}

#[test]
fn oracle_entry_parses_string_expressions() {
    let entry = load_oracle_entry(&oracle_toml_path(), EXPEDITION_ROOT)
        .expect("oracle entry should load");

    // Find the argument_reduction_boundaries injection set
    let arb = entry.injection_sets.iter()
        .find(|(name, _)| name == "argument_reduction_boundaries");
    assert!(arb.is_some(), "argument_reduction_boundaries set must be parsed");

    let (_, inputs) = arb.unwrap();
    // "ln(2)" should evaluate to std::f64::consts::LN_2
    let ln2 = inputs.iter().find(|&&x| (x - std::f64::consts::LN_2).abs() < 1e-15);
    assert!(ln2.is_some(), "ln(2) expression must evaluate correctly");
}

// ─────────────────────────────────────────────────────────────────────────────
// Calibration run: f64::exp as candidate
// ─────────────────────────────────────────────────────────────────────────────

/// Calibration: f64::exp must pass the 1-ULP oracle for tam_exp.
///
/// This establishes that the oracle runner infrastructure works end-to-end,
/// and that `f64::exp` is within 1 ULP of mpmath on the primary domain.
/// This is NOT a validation of tam_exp — it's a runner smoke test.
///
/// I9: mpmath at 50 digits is the oracle; f64::exp is the candidate.
#[test]
fn calibration_f64_exp_passes_claimed_bound() {
    let entry = load_oracle_entry(&oracle_toml_path(), EXPEDITION_ROOT)
        .expect("oracle entry should load");

    let report = run_oracle(&entry, "exp", |x| x.exp());

    println!("{}", report.summary());

    // The random sample (from exp-1k.bin) must be within 1 ULP
    assert!(report.random_sample_report.passes(entry.claimed_max_ulp as u64),
        "f64::exp must be ≤ {} ULP from mpmath on random sample; got max_ulp={}",
        entry.claimed_max_ulp, report.random_sample_report.max_ulp);

    // Bit-exact checks: exp(-inf) = +0.0, not -0.0 (I11 + IEEE 754)
    let bit_exact_failures: Vec<&str> = report.bit_exact_results.iter()
        .filter(|r| !r.passes)
        .map(|r| r.name.as_str())
        .collect();
    assert!(bit_exact_failures.is_empty(),
        "bit-exact checks must pass for f64::exp; failed: {:?}", bit_exact_failures);

    // Identity checks: exp_negation, exp_one_returns_e (available without tam_ln)
    for result in &report.identity_results {
        // Skip identities that require unavailable functions (NaN residual = not available)
        if result.max_residual_ulp.is_nan() { continue; }
        // exp_log_roundtrip requires ln — not available yet; it returns NaN residual.
        // The runner skips it gracefully. Only check the ones that ran.
        assert!(result.passes,
            "identity check '{}' must pass; max_residual={:.3} ULP, tolerance={} ULP; worst_case={:?}",
            result.name, result.max_residual_ulp, result.claimed_tolerance_ulp, result.worst_case_input);
    }
}

/// Verify the runner detects a broken candidate.
///
/// A candidate that always returns 0.0 should fail the oracle's 1-ULP bound
/// on the random sample AND fail the bit-exact checks (0.0 is not +inf for
/// exp(+inf), not 1.0 for exp(0.0), etc.).
#[test]
fn oracle_detects_broken_candidate() {
    let entry = load_oracle_entry(&oracle_toml_path(), EXPEDITION_ROOT)
        .expect("oracle entry should load");

    // Broken candidate: always returns 0.0
    let report = run_oracle(&entry, "exp", |_x| 0.0_f64);

    // The random sample must fail
    assert!(!report.random_sample_report.passes(entry.claimed_max_ulp as u64),
        "broken candidate must fail random sample ULP check");

    // At least some bit-exact checks must fail
    let bit_exact_failures = report.bit_exact_results.iter().filter(|r| !r.passes).count();
    assert!(bit_exact_failures > 0,
        "broken candidate (always 0.0) must fail at least one bit-exact check");

    // Overall must fail
    assert!(!report.passes, "overall oracle report must FAIL for a broken candidate");
}

/// Verify NaN propagation through the oracle (I11).
///
/// The special_values injection set includes NaN. The runner must detect that
/// a candidate which returns 0.0 for NaN input has a special_value_failure.
#[test]
fn oracle_nan_propagation_check() {
    let entry = load_oracle_entry(&oracle_toml_path(), EXPEDITION_ROOT)
        .expect("oracle entry should load");

    // Check that the special_values set contains NaN
    let special_set = entry.injection_sets.iter()
        .find(|(name, _)| name == "special_values");
    assert!(special_set.is_some(), "special_values injection set must exist");
    let (_, inputs) = special_set.unwrap();
    let has_nan = inputs.iter().any(|x| x.is_nan());
    assert!(has_nan, "special_values must include NaN (I11 test)");

    // A NaN-propagating candidate (f64::exp) must not have special_value_failures
    // on the NaN input
    let report = run_oracle(&entry, "exp", |x| x.exp());
    let special_report = report.injection_reports.iter()
        .find(|(name, _)| name == "special_values")
        .map(|(_, r)| r);
    if let Some(r) = special_report {
        assert_eq!(r.special_value_failures, 0,
            "f64::exp must propagate NaN correctly; got {} special_value_failures",
            r.special_value_failures);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// When tam_exp lands (placeholder)
// ─────────────────────────────────────────────────────────────────────────────

/// Placeholder: when tambear-libm tam_exp is implemented (Peak 2 campsite 2.6+),
/// replace f64::exp with the real implementation.
///
/// The test body is identical to calibration_f64_exp_passes_claimed_bound.
/// Scientist removes the #[ignore] annotation and swaps the candidate.
#[test]
#[ignore = "pending Peak 2 campsite 2.6+: tam_exp implementation in tambear-libm"]
fn tam_exp_passes_oracle() {
    // When tam_exp is available:
    // use tambear_libm::tam_exp;
    // let entry = load_oracle_entry(...);
    // let report = run_oracle(&entry, "tam_exp", |x| tam_exp(x));
    // assert!(report.passes, "{}", report.summary());
    todo!("wire up tambear_libm::tam_exp");
}
