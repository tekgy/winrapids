//! Cross-platform bit-exact determinism tests for `tambear::math`.
//!
//! Every building block in `math::` must produce the same bit pattern on
//! adversarial inputs as a direct Kulisch reference computation. This is
//! the correctness gate that lets recipes compose math building blocks
//! without worrying about numerical drift across thread counts or backends.

use tambear::math;
use tambear::primitives::compensated::two_product_fma;
use tambear::primitives::specialist::kulisch_accumulator::KulischAccumulator;

// ── Adversarial corpora (same shape as determinism_contract.rs) ─────────────

fn corpus_small() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 4.0, 5.0]
}
fn corpus_signed() -> Vec<f64> {
    vec![3.7, -0.25, 1e10, -0.5, 1.25, -1e10, 42.0, -6.0]
}
fn corpus_cancellation() -> Vec<f64> {
    let mut xs = Vec::with_capacity(3000);
    for _ in 0..1000 {
        xs.push(1e17);
        xs.push(1.0);
        xs.push(-1e17);
    }
    xs
}
fn corpus_kahan_trap() -> Vec<f64> {
    let mut xs = vec![1.0];
    for _ in 0..10_000 {
        xs.push(1e-10);
    }
    xs
}
fn corpus_mixed_scale() -> Vec<f64> {
    (0..500)
        .map(|i| match i % 4 {
            0 => (i as f64) * 1e50,
            1 => -(i as f64),
            2 => (i as f64) * 1e-50,
            _ => (i as f64).sin() * 1e10,
        })
        .collect()
}
fn corpus_with_nan_inf() -> Vec<f64> {
    let mut xs = corpus_signed();
    xs.insert(2, f64::NAN);
    xs.insert(5, f64::INFINITY);
    xs.insert(7, f64::NEG_INFINITY);
    xs
}

fn all_corpora() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("small", corpus_small()),
        ("signed", corpus_signed()),
        ("cancellation", corpus_cancellation()),
        ("kahan_trap", corpus_kahan_trap()),
        ("mixed_scale", corpus_mixed_scale()),
        ("with_nan_inf", corpus_with_nan_inf()),
    ]
}

fn assert_bits_eq(got: f64, want: f64, label: &str) {
    // Allow NaN == NaN for ratio-of-zero cases.
    if got.is_nan() && want.is_nan() {
        return;
    }
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{label}: got {got:e} (bits {:#018x}), want {want:e} (bits {:#018x})",
        got.to_bits(),
        want.to_bits()
    );
}

// ── Sum ─────────────────────────────────────────────────────────────────────

#[test]
fn sum_matches_kulisch_reference() {
    for (name, xs) in all_corpora() {
        let mut k = KulischAccumulator::new();
        k.add_slice(&xs);
        let want = k.to_f64();
        let got = math::sum(&xs);
        assert_bits_eq(got, want, &format!("sum on '{name}'"));
    }
}

#[test]
fn sum_empty_returns_zero() {
    assert_eq!(math::sum(&[]).to_bits(), 0.0f64.to_bits());
}

#[test]
fn sum_all_nan_returns_zero() {
    let xs = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    assert_eq!(math::sum(&xs).to_bits(), 0.0f64.to_bits());
}

#[test]
fn sum_run_to_run_bit_exact() {
    for (name, xs) in all_corpora() {
        let a = math::sum(&xs);
        let b = math::sum(&xs);
        assert_bits_eq(a, b, &format!("sum run-to-run on '{name}'"));
    }
}

// ── Sum of squares ──────────────────────────────────────────────────────────

#[test]
fn sum_sq_matches_kulisch_of_two_product() {
    for (name, xs) in all_corpora() {
        let mut k = KulischAccumulator::new();
        for &v in &xs {
            if v.is_finite() {
                let (hi, lo) = two_product_fma(v, v);
                k.add_f64(hi);
                k.add_f64(lo);
            }
        }
        let want = k.to_f64();
        let got = math::sum_sq(&xs);
        assert_bits_eq(got, want, &format!("sum_sq on '{name}'"));
    }
}

// ── Mean ────────────────────────────────────────────────────────────────────

#[test]
fn mean_matches_kulisch_sum_over_count() {
    for (name, xs) in all_corpora() {
        let n = math::count_finite(&xs);
        let mut k = KulischAccumulator::new();
        k.add_slice(&xs);
        let want = if n == 0 {
            f64::NAN
        } else {
            k.to_f64() / n as f64
        };
        let got = math::mean(&xs);
        assert_bits_eq(got, want, &format!("mean on '{name}'"));
    }
}

#[test]
fn mean_empty_is_nan() {
    assert!(math::mean(&[]).is_nan());
}

#[test]
fn mean_all_nan_is_nan() {
    assert!(math::mean(&[f64::NAN, f64::INFINITY]).is_nan());
}

// ── Variance ────────────────────────────────────────────────────────────────

#[test]
fn variance_sample_matches_two_pass_kulisch() {
    for (name, xs) in all_corpora() {
        let n = math::count_finite(&xs);
        if n < 2 {
            continue;
        }
        // Reference: two-pass exact Kulisch.
        let mu = math::sum(&xs) / n as f64;
        let mut k = KulischAccumulator::new();
        for &v in &xs {
            if v.is_finite() {
                let d = v - mu;
                let (hi, lo) = two_product_fma(d, d);
                k.add_f64(hi);
                k.add_f64(lo);
            }
        }
        let want = k.to_f64() / (n as f64 - 1.0);
        let got = math::variance_sample(&xs);
        assert_bits_eq(got, want, &format!("variance_sample on '{name}'"));
    }
}

#[test]
fn variance_sample_n_lt_2_is_nan() {
    assert!(math::variance_sample(&[]).is_nan());
    assert!(math::variance_sample(&[1.0]).is_nan());
    assert!(math::variance_sample(&[f64::NAN, 1.0]).is_nan());
}

#[test]
fn variance_population_denominator_is_n() {
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mu = math::mean(&xs);
    let expected = (xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>()) / 5.0;
    let got = math::variance_population(&xs);
    // Bit-equality not expected here (reference uses naive f64), but close.
    assert!((got - expected).abs() < 1e-12);
}

// ── Dot product ─────────────────────────────────────────────────────────────

#[test]
fn dot_matches_kulisch_two_product_reference() {
    let a = corpus_mixed_scale();
    let b: Vec<f64> = a.iter().map(|x| x * 1.5 - 0.25).collect();
    let mut k = KulischAccumulator::new();
    for i in 0..a.len() {
        let (x, y) = (a[i], b[i]);
        if x.is_finite() && y.is_finite() {
            let (hi, lo) = two_product_fma(x, y);
            k.add_f64(hi);
            k.add_f64(lo);
        }
    }
    let want = k.to_f64();
    let got = math::dot(&a, &b);
    assert_bits_eq(got, want, "dot on mixed_scale");
}

#[test]
fn dot_run_to_run_bit_exact() {
    let a = corpus_cancellation();
    let b: Vec<f64> = a.iter().map(|x| x * 0.5 + 1.0).collect();
    let x = math::dot(&a, &b);
    let y = math::dot(&a, &b);
    assert_bits_eq(x, y, "dot run-to-run on cancellation");
}

#[test]
fn dot_skips_non_finite_pairs() {
    let a = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];
    let b = vec![2.0, 10.0, 4.0, 100.0, 6.0];
    // Only (1, 2), (3, 4), (5, 6) contribute → 2 + 12 + 30 = 44
    assert_eq!(math::dot(&a, &b).to_bits(), 44.0f64.to_bits());
}

// ── Weighted sum ────────────────────────────────────────────────────────────

#[test]
fn weighted_sum_equals_dot() {
    let values = corpus_small();
    let weights = vec![0.1, 0.2, 0.3, 0.2, 0.2];
    let a = math::weighted_sum(&values, &weights);
    let b = math::dot(&values, &weights);
    assert_bits_eq(a, b, "weighted_sum vs dot equivalence");
}

// ── Correlation / covariance ────────────────────────────────────────────────

#[test]
fn correlation_of_series_with_itself_is_one() {
    let xs = corpus_mixed_scale();
    let r = math::correlation(&xs, &xs);
    // For a series with itself, correlation is exactly 1.0 in exact arithmetic.
    // With finite-precision final division, it should be very close.
    assert!((r - 1.0).abs() < 1e-12, "self-correlation not ≈ 1: got {r}");
}

#[test]
fn correlation_of_antiseries_is_neg_one() {
    let xs = corpus_small();
    let neg: Vec<f64> = xs.iter().map(|x| -x).collect();
    let r = math::correlation(&xs, &neg);
    assert!((r - -1.0).abs() < 1e-12, "anti-correlation not ≈ -1: got {r}");
}

#[test]
fn covariance_sample_run_to_run_bit_exact() {
    let a = corpus_cancellation();
    let b: Vec<f64> = a.iter().map(|x| x * 0.5 + 1.0).collect();
    let x = math::covariance_sample(&a, &b);
    let y = math::covariance_sample(&a, &b);
    assert_bits_eq(x, y, "covariance_sample run-to-run on cancellation");
}

// ── Centered sum sq ─────────────────────────────────────────────────────────

#[test]
fn centered_sum_sq_at_zero_mean_equals_sum_sq() {
    // If mean is 0, centered_sum_sq should equal sum_sq.
    let xs = vec![1.0, -1.0, 2.0, -2.0]; // mean = 0
    let a = math::centered_sum_sq(&xs, 0.0);
    let b = math::sum_sq(&xs);
    assert_bits_eq(a, b, "centered_sum_sq at mean=0 equals sum_sq");
}
