//! Single source of truth for algorithm weakness — what fails on what data
//! and what to reach for instead.
//!
//! This table is consumed by several layers of tambear:
//!
//! 1. **Oracle test suite** (`primitives/oracle/mod.rs`) generates per-algorithm
//!    skip lists from this table. A test like `kahan_sum_passes_hard_sums` skips
//!    the cases where Kahan is mathematically known to lose bits; the list of
//!    cases comes from here, not from hardcoded strings in the test.
//!
//! 2. **tbs_lint** (`crates/tambear/src/tbs_lint.rs`) warns a `.tbs` script
//!    author when they call a primitive on data that matches a known weakness.
//!    It reads the `data_probe` field to decide what quick check to run and
//!    reads `explanation` + `recommended` to format the warning.
//!
//! 3. **TbsStepAdvice** (`crates/tambear/src/tbs_advice.rs`) populates its
//!    `TbsRecommendation` fields from this table when the executor decides to
//!    route around a known weakness automatically.
//!
//! 4. **IDE docstrings** — a build-time code generator can synthesize a
//!    "# Known limitations" section at the bottom of each primitive's rustdoc
//!    from this table so published API docs and test-suite truth stay
//!    synchronized.
//!
//! 5. **Layer 1 auto-detect dispatch** — when a recipe calls the generic
//!    `sum()` without an explicit `using()`, the dispatcher runs the listed
//!    data probes on the input and picks the tightest algorithm whose row
//!    does NOT match any detected weakness.
//!
//! # Adding a new entry
//!
//! 1. Document the algorithm and the data shape that breaks it.
//! 2. Add a `failing_case` name — this must match an entry in `hard_sums()` if
//!    you want the oracle test suite to exercise it automatically. If it is a
//!    novel pattern, add a new case to `hard_sums()` first, then reference it.
//! 3. Write the `explanation` as prose suitable for a user-facing advice
//!    message — avoid internal jargon.
//! 4. Choose the `recommended` alternative — this should name a primitive that
//!    actually handles the failing case.
//! 5. If possible, name a `data_probe` that can detect the failure mode at
//!    runtime. If no cheap probe exists, use `DataProbe::None` and rely on
//!    user opt-in or pattern-based lint warnings.
//!
//! # What this table is NOT for
//!
//! - Bugs in the implementation — those get fixed, not documented. This table
//!   is for *algorithmic* weakness inherent to the named method.
//! - Performance characteristics — speed/cost is documented in the primitive's
//!   rustdoc, not here.
//! - User preference defaults — this is for correctness, not style.

/// A single documented weakness of a named primitive.
#[derive(Debug, Clone, Copy)]
pub struct AlgorithmWeakness {
    /// Name of the primitive function (e.g., `"kahan_sum"`).
    pub algorithm: &'static str,

    /// Name of the `hard_sums()` case that triggers this weakness. Oracle
    /// tests for `algorithm` skip this case and list it in the failure
    /// explanation.
    pub failing_case: &'static str,

    /// Prose explanation suitable for a user-facing advice message.
    pub explanation: &'static str,

    /// Name of another primitive that handles the failing case correctly.
    pub recommended: &'static str,

    /// Optional runtime probe that can detect whether the user's input
    /// exhibits the failing pattern. Used by Layer 1 auto-dispatch.
    pub data_probe: DataProbe,
}

/// A cheap probe that checks whether an input slice exhibits a pattern that
/// triggers a known algorithm weakness.
///
/// These probes are designed to run in O(n) with small constants. They should
/// never allocate on the heap and should never overflow or panic on any input.
///
/// The Layer 1 dispatcher runs the relevant probe on the user's data before
/// deciding which summation primitive to invoke. If the probe fires, the
/// dispatcher picks the recommended alternative from the matching
/// [`AlgorithmWeakness`] row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataProbe {
    /// No cheap probe exists. Dispatch defaults to the recommended safer
    /// alternative whenever the data can't be pre-classified.
    None,

    /// True when the input contains both positive and negative values whose
    /// magnitudes exceed `max|xᵢ| · 1e15` — the threshold above which f64
    /// summation can lose a full unit of magnitude in the trailing residual.
    LargeCancellationRatio,

    /// True when any element exceeds `1e300` in magnitude or any non-zero
    /// element is below `1e-300`. Compensated summation degrades in these
    /// regimes because the compensation register overflows or underflows.
    NearFloatingPointLimits,

    /// True when the input has both very large and very small elements
    /// without prior ordering — the case where Kahan's `fast_two_sum`
    /// precondition fails and the compensation term captures the wrong
    /// residual.
    UnorderedMixedMagnitudes,
}

impl DataProbe {
    /// Run this probe on an f64 slice and return whether the weakness condition
    /// is present. Safe on any input, including empty, NaN, infinity.
    pub fn fires_on(self, xs: &[f64]) -> bool {
        match self {
            DataProbe::None => false,
            DataProbe::LargeCancellationRatio => large_cancellation_ratio(xs),
            DataProbe::NearFloatingPointLimits => near_floating_point_limits(xs),
            DataProbe::UnorderedMixedMagnitudes => unordered_mixed_magnitudes(xs),
        }
    }
}

fn large_cancellation_ratio(xs: &[f64]) -> bool {
    let mut max_pos = 0.0_f64;
    let mut max_neg = 0.0_f64;
    let mut min_nonzero = f64::INFINITY;
    for &x in xs {
        if !x.is_finite() {
            continue;
        }
        if x > max_pos {
            max_pos = x;
        }
        if x < 0.0 && -x > max_neg {
            max_neg = -x;
        }
        let ax = x.abs();
        if ax > 0.0 && ax < min_nonzero {
            min_nonzero = ax;
        }
    }
    let peak = max_pos.max(max_neg);
    // Both a positive and a negative peak exist (cancellation possible),
    // and there's some value at least 1e15 smaller than the peak.
    max_pos > 0.0 && max_neg > 0.0 && peak / min_nonzero > 1e15
}

fn near_floating_point_limits(xs: &[f64]) -> bool {
    for &x in xs {
        if !x.is_finite() {
            continue;
        }
        let ax = x.abs();
        if ax > 1e300 {
            return true;
        }
        if ax > 0.0 && ax < 1e-300 {
            return true;
        }
    }
    false
}

fn unordered_mixed_magnitudes(xs: &[f64]) -> bool {
    if xs.len() < 2 {
        return false;
    }
    // Look for a decrease-then-increase pattern in |xᵢ| that would violate
    // fast_two_sum's ordering precondition in a running-sum interpretation.
    let mut seen_decrease = false;
    let mut prev = xs[0].abs();
    for &x in &xs[1..] {
        let ax = x.abs();
        if ax.is_nan() {
            continue;
        }
        if ax < prev * 0.5 {
            seen_decrease = true;
        }
        if seen_decrease && ax > prev * 2.0 {
            return true;
        }
        prev = ax;
    }
    false
}

/// The canonical catalog of algorithm weaknesses. Keep this sorted by
/// algorithm name, then by failing case.
///
/// When adding a row, also ensure the named `failing_case` exists in
/// `hard_sums()` in `oracle::mod.rs`.
pub const ALGORITHM_WEAKNESSES: &[AlgorithmWeakness] = &[
    // ── kahan_sum ───────────────────────────────────────────────────────────
    AlgorithmWeakness {
        algorithm: "kahan_sum",
        failing_case: "perfect_cancellation",
        explanation: "\
Kahan's compensation register is rounded away when the running sum returns \
to near-zero after a large positive and an equal-magnitude negative value. \
A small value (e.g. `1e17 - 1e17 + 1`) gets lost because the compensation \
term itself underflows relative to the running sum during the cancellation \
step. This is a property of the Kahan-Babuška algorithm, not an implementation \
bug — every canonical Kahan implementation has the same weakness.",
        recommended: "neumaier_sum (compares magnitudes before compensation) \
or sum_k(xs, 2) for the same cost with stricter correctness",
        data_probe: DataProbe::LargeCancellationRatio,
    },
    AlgorithmWeakness {
        algorithm: "kahan_sum",
        failing_case: "rump_example_small",
        explanation: "\
When large cancelling pairs are interleaved with small values (the Rump \
example pattern), Kahan's single-level compensation cannot recover the \
trapped bits — each cancellation resets the compensation register before \
the next small value is added.",
        recommended: "neumaier_sum, or sum_k(xs, 3) for increased condition-number tolerance",
        data_probe: DataProbe::LargeCancellationRatio,
    },
    AlgorithmWeakness {
        algorithm: "kahan_sum",
        failing_case: "near_overflow_with_tiny",
        explanation: "\
For values near f64::MAX the compensation register cannot represent the \
small trapped values at all — their magnitudes differ from the running sum \
by more than 2^53, which is the full precision budget of a single f64.",
        recommended: "sum_k(xs, 3) or kulisch_oracle_sum (exact, slower)",
        data_probe: DataProbe::NearFloatingPointLimits,
    },
    // ── pairwise_sum ────────────────────────────────────────────────────────
    AlgorithmWeakness {
        algorithm: "pairwise_sum",
        failing_case: "perfect_cancellation",
        explanation: "\
Pairwise summation has no compensation mechanism at all. Its error bound \
is O(log n · ε · Σ|xᵢ|) which is tight for well-conditioned sums but does \
not recover trapped small values from cancellations of large terms.",
        recommended: "neumaier_sum for cheap compensation, or sum_k(xs, k) for tunable accuracy",
        data_probe: DataProbe::LargeCancellationRatio,
    },
    AlgorithmWeakness {
        algorithm: "pairwise_sum",
        failing_case: "rump_example_small",
        explanation: "\
Same root cause as perfect_cancellation — pairwise has no compensation, so \
any cancellation of large terms loses the trapped small values between them.",
        recommended: "neumaier_sum or sum_k(xs, 3)",
        data_probe: DataProbe::LargeCancellationRatio,
    },
    AlgorithmWeakness {
        algorithm: "pairwise_sum",
        failing_case: "near_overflow_with_tiny",
        explanation: "\
Pairwise splits the input into halves and sums each recursively. When one \
half's sum approaches f64::MAX, adding a 1e-300 element to that half's \
intermediate result is a no-op — the small value is lost immediately.",
        recommended: "sum_k(xs, 3) or kulisch_oracle_sum",
        data_probe: DataProbe::NearFloatingPointLimits,
    },
];

/// Return the list of `hard_sums()` case names that `algorithm` is known to
/// fail on. Oracle tests use this to generate their per-algorithm skip list
/// directly from the catalog.
pub fn skip_list_for(algorithm: &str) -> Vec<&'static str> {
    ALGORITHM_WEAKNESSES
        .iter()
        .filter(|w| w.algorithm == algorithm)
        .map(|w| w.failing_case)
        .collect()
}

/// Look up the weakness rows for a given algorithm.
pub fn weaknesses_of(algorithm: &str) -> Vec<&'static AlgorithmWeakness> {
    ALGORITHM_WEAKNESSES
        .iter()
        .filter(|w| w.algorithm == algorithm)
        .collect()
}

/// Check whether `algorithm` has a documented weakness that triggers on
/// `data`. Returns the matching weakness if so. Used by Layer 1 dispatch
/// and by tbs_lint.
pub fn weakness_for_data(algorithm: &str, data: &[f64]) -> Option<&'static AlgorithmWeakness> {
    ALGORITHM_WEAKNESSES
        .iter()
        .find(|w| w.algorithm == algorithm && w.data_probe.fires_on(data))
}

/// Format a user-facing advice message when `algorithm` has been requested
/// on `data` and a weakness probe has fired. Used by tbs_lint and the IDE
/// education surface.
pub fn format_advice(weakness: &AlgorithmWeakness) -> String {
    format!(
        "You selected `{algo}`. {explanation}\n\nFor your data, consider using `{recommended}` instead. \
         This may or may not be okay depending on your analysis — if you know that the pattern \
         doesn't affect your result, or you must use `{algo}` for comparison with an external \
         reference, you can suppress this advice with `.using(suppress_advice=true)`.",
        algo = weakness.algorithm,
        explanation = weakness.explanation,
        recommended = weakness.recommended,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_non_empty() {
        assert!(!ALGORITHM_WEAKNESSES.is_empty());
    }

    #[test]
    fn every_row_has_non_empty_fields() {
        for w in ALGORITHM_WEAKNESSES {
            assert!(!w.algorithm.is_empty(), "empty algorithm name");
            assert!(!w.failing_case.is_empty(), "empty failing case");
            assert!(!w.explanation.is_empty(), "empty explanation");
            assert!(!w.recommended.is_empty(), "empty recommendation");
        }
    }

    #[test]
    fn every_failing_case_exists_in_hard_sums() {
        use crate::primitives::oracle::hard_sums;
        let cases: Vec<&'static str> = hard_sums().into_iter().map(|(name, _)| name).collect();
        for w in ALGORITHM_WEAKNESSES {
            assert!(
                cases.contains(&w.failing_case),
                "weakness references unknown hard_sums case: {:?} in algorithm {:?}",
                w.failing_case,
                w.algorithm
            );
        }
    }

    #[test]
    fn skip_list_for_kahan_sum_includes_known_cases() {
        let list = skip_list_for("kahan_sum");
        assert!(list.contains(&"perfect_cancellation"));
        assert!(list.contains(&"rump_example_small"));
        assert!(list.contains(&"near_overflow_with_tiny"));
    }

    #[test]
    fn skip_list_for_unknown_algorithm_is_empty() {
        assert!(skip_list_for("nonexistent_algorithm").is_empty());
    }

    #[test]
    fn weaknesses_of_neumaier_is_empty() {
        // Neumaier has no documented weakness — if this starts failing, we
        // discovered a new one and need to document it.
        assert!(weaknesses_of("neumaier_sum").is_empty());
    }

    // ── Data probe behavior ────────────────────────────────────────────────

    #[test]
    fn large_cancellation_probe_fires_on_cancellation_input() {
        let xs = vec![1e17, 1.0, -1e17];
        assert!(DataProbe::LargeCancellationRatio.fires_on(&xs));
    }

    #[test]
    fn large_cancellation_probe_ignores_well_conditioned_input() {
        let xs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert!(!DataProbe::LargeCancellationRatio.fires_on(&xs));
    }

    #[test]
    fn large_cancellation_probe_ignores_same_sign_mixed_magnitude() {
        // All positive: no cancellation possible, even if magnitudes differ.
        let xs = vec![1e17, 1.0, 1e-17];
        assert!(!DataProbe::LargeCancellationRatio.fires_on(&xs));
    }

    #[test]
    fn near_limits_probe_fires_on_big_value() {
        assert!(DataProbe::NearFloatingPointLimits.fires_on(&[1e301]));
    }

    #[test]
    fn near_limits_probe_fires_on_tiny_value() {
        assert!(DataProbe::NearFloatingPointLimits.fires_on(&[1e-301]));
    }

    #[test]
    fn near_limits_probe_ignores_zero() {
        // Zero is not "small" in the subnormal sense — it has a legitimate
        // representation and should not fire the probe.
        assert!(!DataProbe::NearFloatingPointLimits.fires_on(&[0.0, 1.0, 2.0]));
    }

    #[test]
    fn near_limits_probe_ignores_normal_range() {
        assert!(!DataProbe::NearFloatingPointLimits.fires_on(&[1.0, 2.0, 1e100, -3.0]));
    }

    #[test]
    fn weakness_for_data_finds_kahan_cancellation() {
        let xs = vec![1e17, 1.0, -1e17];
        let w = weakness_for_data("kahan_sum", &xs);
        assert!(w.is_some());
        assert_eq!(w.unwrap().algorithm, "kahan_sum");
    }

    #[test]
    fn weakness_for_data_returns_none_on_safe_input() {
        let xs = vec![1.0, 2.0, 3.0];
        assert!(weakness_for_data("kahan_sum", &xs).is_none());
    }

    #[test]
    fn format_advice_mentions_both_algorithms() {
        let w = &ALGORITHM_WEAKNESSES[0];
        let advice = format_advice(w);
        assert!(advice.contains(w.algorithm));
        assert!(advice.contains(w.recommended));
    }

    #[test]
    fn none_probe_never_fires() {
        let cases: &[&[f64]] = &[&[], &[1.0, 2.0], &[1e17, -1e17, 1.0], &[f64::NAN]];
        for xs in cases {
            assert!(!DataProbe::None.fires_on(xs));
        }
    }
}
