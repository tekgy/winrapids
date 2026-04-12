//! Hard-cases suite: adversarial input generators (campsite 4.7).
//!
//! Every case here is designed to expose a specific class of numerical failure.
//! These tests fail immediately (no real backend exists yet), which is correct.
//! Each backend added by Peaks 1/3/5/7 must pass them on arrival.
//!
//! ## Why these specific inputs
//!
//! Each generator targets a known failure mode in floating-point computation:
//!
//! | Case                  | Failure mode it exposes                        |
//! |-----------------------|------------------------------------------------|
//! | catastrophic cancel   | `sum([1e16, 1, -1e16])` loses the `1` entirely |
//! | near-underflow        | subnormal arithmetic may flush to zero         |
//! | nan propagation       | NaN must infect all downstream results         |
//! | inf arithmetic        | inf - inf must produce NaN, not crash          |
//! | huge N                | reduction must not overflow accumulators       |
//! | empty                 | kernel must handle N=0 without panic           |
//! | single element        | variance of 1 element = divide by zero         |
//! | one-pass variance trap | Σx² - (Σx)²/n loses precision catastrophically |
//! | alternating signs     | naive accumulation has large rounding error    |
//! | denormals             | subnormal values must not flush to zero        |
//! | near f64::MAX         | overflow in accumulation                       |
//! | mixed magnitude       | alternating large/small: large swamps small    |
//!
//! ## Invariant I5 note
//!
//! Tests that check the final value of a reduction (sum, variance, etc.) are
//! annotated `#[ignore = "xfail_nondeterministic"]` until Peak 6 lands and
//! deterministic reductions are implemented.  After Peak 6, those annotations
//! are removed by the Test Oracle.

use crate::{Inputs};

/// Catastrophic cancellation: the `1.0` in the middle is lost in naive summation.
///
/// Exact result: 1.0
/// Naive f64 sum result: 0.0 (the `1.0` is below the precision of `1e16`)
///
/// A correct compensated summation (Kahan) gives 1.0.
pub fn catastrophic_cancellation() -> Inputs {
    Inputs::new().with_buf("x", vec![1e16_f64, 1.0, -1e16_f64])
}

/// Near-underflow: values in the subnormal range.
///
/// The hardware must not flush these to zero (FTZ mode would make all results 0).
/// Invariant I4 forbids flush-to-zero.
pub fn near_underflow(n: usize) -> Inputs {
    let v = f64::MIN_POSITIVE / 2.0; // subnormal
    Inputs::new().with_buf("x", vec![v; n])
}

/// NaN propagation: a single NaN must produce NaN in every downstream result.
///
/// This tests that the backend does not silently drop NaN through a select
/// or min/max operation.
pub fn nan_propagation() -> Inputs {
    Inputs::new().with_buf("x", vec![1.0, f64::NAN, 3.0])
}

/// Inf arithmetic: `+inf + (-inf) = NaN`.  Must not panic or produce garbage.
pub fn inf_arithmetic() -> Inputs {
    Inputs::new().with_buf("x", vec![f64::INFINITY, f64::NEG_INFINITY, 1.0])
}

/// Huge N: stress-tests the reduction path for overflow and throughput.
///
/// Generates N copies of 1.0, so the exact sum is N as f64.
pub fn huge_n(n: usize) -> Inputs {
    Inputs::new().with_buf("x", vec![1.0_f64; n])
}

/// Empty input: the kernel must handle zero-length buffers.
///
/// Expected sum = 0.0.  Expected variance = NaN (0 elements).
pub fn empty() -> Inputs {
    Inputs::new().with_buf("x", vec![])
}

/// Single element: variance is undefined (division by zero in the n-1 form).
///
/// Expected behavior: NaN or inf — but must not panic.
pub fn single_element(value: f64) -> Inputs {
    Inputs::new().with_buf("x", vec![value])
}

/// The classic one-pass variance trap.
///
/// All values are close to 1e6 (mean ≈ 1e6), but with small variations.
/// The naive formula Σx² - (Σx)²/n subtracts two nearly equal large numbers,
/// losing all precision.  Welford's algorithm avoids this.
///
/// This is the test that exposed the issue in `tambear_primitives::recipes::variance`.
pub fn one_pass_variance_trap(n: usize) -> Inputs {
    let mean = 1_000_000.0_f64;
    // Linear spread: mean - 1.0, mean - 0.99, ..., mean + 1.0
    let data: Vec<f64> = (0..n).map(|i| {
        mean + (i as f64 / n as f64) * 2.0 - 1.0
    }).collect();
    Inputs::new().with_buf("x", data)
}

/// Alternating signs: `+1, -1, +2, -2, ...`
///
/// Naive summation accumulates rounding errors from each sign flip.
/// Pairwise or tree summation fares much better.
pub fn alternating_signs(n: usize) -> Inputs {
    let data: Vec<f64> = (1..=(n as i64)).map(|i| {
        if i % 2 == 0 { i as f64 } else { -(i as f64) }
    }).collect();
    Inputs::new().with_buf("x", data)
}

/// Denormals: values in the subnormal range (strictly below `f64::MIN_POSITIVE`).
///
/// Hardware in FTZ mode flushes these to 0.  We must never enable FTZ (I4).
pub fn denormals(n: usize) -> Inputs {
    // Use fractions of MIN_POSITIVE, keeping all values strictly subnormal.
    // The largest value is MIN_POSITIVE / 2, well into subnormal territory.
    // We avoid dividing by zero by asserting n >= 1.
    assert!(n >= 1, "denormals requires n >= 1");
    let step = f64::MIN_POSITIVE / (n as f64 + 1.0);
    let data: Vec<f64> = (1..=n).map(|i| step * i as f64).collect();
    Inputs::new().with_buf("x", data)
}

/// Near f64::MAX: accumulation overflow.
///
/// Sum of N copies of f64::MAX / N should equal f64::MAX exactly.
/// Sum of N+1 copies overflows to inf.
pub fn near_max(n: usize) -> Inputs {
    Inputs::new().with_buf("x", vec![f64::MAX / n as f64; n])
}

/// Mixed magnitude: alternating 1e20 and 1e-20.
///
/// The large values swamp the small ones in naive summation.
/// Tests whether the kernel correctly handles large dynamic range.
pub fn mixed_magnitude(n: usize) -> Inputs {
    let data: Vec<f64> = (0..n).map(|i| {
        if i % 2 == 0 { 1e20_f64 } else { 1e-20_f64 }
    }).collect();
    Inputs::new().with_buf("x", data)
}

/// All cases as a named collection for property-based iteration.
pub fn all_cases() -> Vec<(&'static str, Inputs)> {
    vec![
        ("catastrophic_cancellation", catastrophic_cancellation()),
        ("near_underflow_100",        near_underflow(100)),
        ("nan_propagation",           nan_propagation()),
        ("inf_arithmetic",            inf_arithmetic()),
        ("huge_n_1m",                 huge_n(1_000_000)),
        ("empty",                     empty()),
        ("single_element_zero",       single_element(0.0)),
        ("single_element_five",       single_element(5.0)),
        ("one_pass_variance_trap_100", one_pass_variance_trap(100)),
        ("alternating_signs_1000",    alternating_signs(1000)),
        ("denormals_100",             denormals(100)),
        ("near_max_10",               near_max(10)),
        ("mixed_magnitude_200",       mixed_magnitude(200)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Validate the generators produce the right shapes and known values.
    // These tests don't run any backend — they test the generators themselves.

    #[test]
    fn catastrophic_cancellation_has_three_elements() {
        let inp = catastrophic_cancellation();
        let buf = &inp.buffers[0].1;
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 1e16_f64);
        assert_eq!(buf[1], 1.0_f64);
        assert_eq!(buf[2], -1e16_f64);
    }

    #[test]
    fn near_underflow_values_are_subnormal() {
        let inp = near_underflow(5);
        for v in &inp.buffers[0].1 {
            assert!(v.is_subnormal(), "{v} should be subnormal");
        }
    }

    #[test]
    fn nan_propagation_contains_nan() {
        let inp = nan_propagation();
        let has_nan = inp.buffers[0].1.iter().any(|v| v.is_nan());
        assert!(has_nan);
    }

    #[test]
    fn inf_arithmetic_contains_inf() {
        let inp = inf_arithmetic();
        let has_pos_inf = inp.buffers[0].1.iter().any(|v| *v == f64::INFINITY);
        let has_neg_inf = inp.buffers[0].1.iter().any(|v| *v == f64::NEG_INFINITY);
        assert!(has_pos_inf);
        assert!(has_neg_inf);
    }

    #[test]
    fn empty_has_zero_elements() {
        let inp = empty();
        assert_eq!(inp.buffers[0].1.len(), 0);
    }

    #[test]
    fn single_element_has_one_element() {
        let inp = single_element(42.0);
        assert_eq!(inp.buffers[0].1.len(), 1);
        assert_eq!(inp.buffers[0].1[0], 42.0);
    }

    #[test]
    fn one_pass_trap_has_correct_length() {
        let n = 100;
        let inp = one_pass_variance_trap(n);
        assert_eq!(inp.buffers[0].1.len(), n);
    }

    #[test]
    fn alternating_signs_alternates() {
        let inp = alternating_signs(6);
        let buf = &inp.buffers[0].1;
        // Odd indices: -(2k-1), even indices: +(2k)
        // i=1 → -1, i=2 → +2, i=3 → -3, i=4 → +4, i=5 → -5, i=6 → +6
        assert!(buf[0] < 0.0); // i=1 is odd → negative
        assert!(buf[1] > 0.0); // i=2 is even → positive
    }

    #[test]
    fn denormals_are_subnormal() {
        let inp = denormals(10);
        for v in &inp.buffers[0].1 {
            assert!(v.is_subnormal(), "{v} should be subnormal");
        }
    }

    #[test]
    fn near_max_sum_is_finite() {
        // The sum of N copies of MAX/N should be finite (= MAX exactly, or close)
        let n = 10;
        let inp = near_max(n);
        let sum: f64 = inp.buffers[0].1.iter().sum();
        assert!(sum.is_finite(), "sum of near_max should be finite, got {sum}");
    }

    #[test]
    fn mixed_magnitude_alternates() {
        let inp = mixed_magnitude(4);
        let buf = &inp.buffers[0].1;
        assert_eq!(buf[0], 1e20_f64);
        assert_eq!(buf[1], 1e-20_f64);
        assert_eq!(buf[2], 1e20_f64);
        assert_eq!(buf[3], 1e-20_f64);
    }

    #[test]
    fn all_cases_returns_expected_count() {
        let cases = all_cases();
        assert_eq!(cases.len(), 13);
        // All names are unique
        let names: std::collections::HashSet<&str> =
            cases.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), cases.len(), "duplicate case names");
    }

    // -----------------------------------------------------------------------
    // The tests below are the actual oracle tests that will fail until a
    // backend exists.  They are currently ignore-marked because no backend
    // has landed yet.  As each peak lands, the implementing backend's test
    // module will un-ignore these (or recreate them with the real backends).
    //
    // Note: reduction-based tests are also marked xfail_nondeterministic
    // until Peak 6 lands.  The Test Oracle removes those marks post-Peak-6.
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "no backend yet — will be activated when Peak 5 (cpu-interpreter) lands"]
    fn catastrophic_cancellation_sum_is_one() {
        // When CPU interpreter backend is available:
        // run sum_all program on catastrophic_cancellation() input
        // assert output[0] == 1.0 (requires compensated summation to pass)
        todo!("wire up CpuInterpreterBackend");
    }

    #[test]
    #[ignore = "no backend yet — will be activated when Peak 5 lands"]
    fn nan_propagates_through_sum() {
        // Sum of [1.0, NaN, 3.0] must be NaN
        todo!("wire up CpuInterpreterBackend");
    }

    #[test]
    #[ignore = "no backend yet; xfail_nondeterministic until Peak 6"]
    fn cross_backend_agree_on_sum_huge_n() {
        // I5: reduction must be deterministic.  This test is xfail until
        // Peak 6 replaces atomicAdd with fixed-order tree reduce.
        todo!("wire up CpuInterpreterBackend + CudaPtxRawBackend");
    }

    #[test]
    #[ignore = "no backend yet; xfail_nondeterministic until Peak 6"]
    fn cross_backend_agree_on_variance_one_pass_trap() {
        // This is the core correctness check for variance:
        // CPU and GPU must agree bit-exactly after Peak 6.
        todo!("wire up backends");
    }
}
