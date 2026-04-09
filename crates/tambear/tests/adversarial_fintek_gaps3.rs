//! Adversarial tests for fintek gap implementations wave 3: RQA.
//!
//! Gold standards:
//! - Constant signal → RR = 1, DET = 1, LAM = 1.
//! - Periodic signal → high DET (most points on diagonal stripes).
//! - Random signal → low DET compared to periodic.
//! - Degenerate input (too short, m=0, tau=0) → NaN result.
//! - Lmax on constant signal = n_vec (entire diagonal is one run).
//! - Diagonal contributions are symmetric (upper == lower).

use tambear::complexity::{rqa, RqaResult};

// ═══════════════════════════════════════════════════════════════════════════
// Guards
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rqa_too_short_returns_nan() {
    let r = rqa(&[1.0, 2.0], 3, 1, 0.5, 2);
    assert!(r.rr.is_nan(), "too-short data should return NaN rr");
}

#[test]
fn rqa_zero_embedding_dim_returns_nan() {
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let r = rqa(&data, 0, 1, 0.5, 2);
    assert!(r.rr.is_nan(), "m=0 must produce NaN");
}

#[test]
fn rqa_zero_tau_returns_nan() {
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let r = rqa(&data, 3, 0, 0.5, 2);
    assert!(r.rr.is_nan(), "tau=0 must produce NaN");
}

#[test]
fn rqa_negative_epsilon_returns_nan() {
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let r = rqa(&data, 3, 1, -1.0, 2);
    assert!(r.rr.is_nan(), "epsilon<=0 must produce NaN");
}

#[test]
fn rqa_nan_epsilon_returns_nan() {
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let r = rqa(&data, 3, 1, f64::NAN, 2);
    assert!(r.rr.is_nan(), "NaN epsilon must produce NaN");
}

// ═══════════════════════════════════════════════════════════════════════════
// Constant signal
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rqa_constant_signal_full_recurrence() {
    let n = 50usize;
    let m = 3usize;
    let tau = 1usize;
    let data = vec![3.14_f64; n];
    let r = rqa(&data, m, tau, 0.1, 2);
    // Every embedded vector is identical → every pair recurs → RR = 1.
    assert!((r.rr - 1.0).abs() < 1e-12,
        "constant signal should have RR=1, got {}", r.rr);
    // Each upper diagonal k in 1..=n_vec-1 is a single contiguous run of
    // length n_vec-k. Lines of length 1 (k = n_vec-1) are excluded by lmin=2.
    // DET = 2*sum(l for l in 2..=n_vec-1) / (n_vec*(n_vec-1))
    //     = 2*((n_vec-1)*n_vec/2 - 1) / (n_vec*(n_vec-1))
    //     = 1 - 2/(n_vec*(n_vec-1))
    let n_vec = n - (m - 1) * tau;
    let expected_det = 1.0 - 2.0 / (n_vec * (n_vec - 1)) as f64;
    assert!((r.det - expected_det).abs() < 1e-12,
        "constant DET should be {} (1 - 2/{}), got {}", expected_det, n_vec*(n_vec-1), r.det);
    // Verticals: each column has an "above main diagonal" run and a "below"
    // run, split at the main diagonal (which is excluded). The columns at
    // j=1 and j=n_vec-2 produce a length-1 run each (excluded by lmin=2),
    // so LAM has the same 2-point deficit as DET.
    let expected_lam = 1.0 - 2.0 / (n_vec * (n_vec - 1)) as f64;
    assert!((r.lam - expected_lam).abs() < 1e-12,
        "constant signal LAM should be {} (1 - 2/{}), got {}",
        expected_lam, n_vec*(n_vec-1), r.lam);
}

#[test]
fn rqa_constant_signal_lmax_equals_nvec_minus_one() {
    let n = 30;
    let m = 3;
    let tau = 1;
    let data = vec![1.0; n];
    let r = rqa(&data, m, tau, 0.5, 2);
    // n_vec = n - (m-1)*tau = 30 - 2 = 28.
    // The longest non-main diagonal (k=1) has 27 elements, all recurrent.
    // Main diagonal (k=0) is excluded from diagonal line counting; the next
    // diagonal has length n_vec - 1 = 27.
    let n_vec = n - (m - 1) * tau;
    assert_eq!(r.lmax, n_vec - 1,
        "constant signal's longest off-diagonal should be n_vec-1={}, got {}",
        n_vec - 1, r.lmax);
}

// ═══════════════════════════════════════════════════════════════════════════
// Periodic vs random — structural DET comparison
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rqa_periodic_det_higher_than_random() {
    // Pure sinusoid — strong diagonal line structure at the period.
    let n = 400;
    let periodic: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();

    // Random signal via Xoshiro256.
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let random: Vec<f64> = (0..n)
        .map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0))
        .collect();

    // Match each signal's scale so epsilon picks comparable fraction of pairs.
    let std_p = {
        let mean = periodic.iter().sum::<f64>() / n as f64;
        (periodic.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    let std_r = {
        let mean = random.iter().sum::<f64>() / n as f64;
        (random.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
    };

    let rp = rqa(&periodic, 3, 1, 0.1 * std_p, 2);
    let rr = rqa(&random, 3, 1, 0.1 * std_r, 2);

    assert!(rp.det.is_finite() && rr.det.is_finite(),
        "DET must be finite; periodic={}, random={}", rp.det, rr.det);
    assert!(rp.det > rr.det,
        "periodic DET {} should exceed random DET {}", rp.det, rr.det);
}

#[test]
fn rqa_periodic_lmax_large() {
    // A pure period-T sinusoid produces diagonal lines of length ≈ (n_vec - kT)
    // at every multiple of T. The longest off-main diagonal is thus large.
    let n = 400;
    let period = 50.0_f64; // samples
    let data: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / period).sin())
        .collect();
    let std_d = {
        let mean = data.iter().sum::<f64>() / n as f64;
        (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    let r = rqa(&data, 3, 1, 0.05 * std_d, 2);
    // Should see at least one long diagonal (>> 10).
    assert!(r.lmax >= 20,
        "periodic signal should have long diagonals; got lmax={}", r.lmax);
}

// ═══════════════════════════════════════════════════════════════════════════
// RR within [0,1] and finiteness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rqa_rr_in_unit_interval() {
    let n = 200;
    let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.03).sin() + (i as f64 * 0.17).cos()).collect();
    let r = rqa(&data, 3, 2, 0.3, 2);
    assert!(r.rr >= 0.0 && r.rr <= 1.0,
        "RR must be in [0,1], got {}", r.rr);
    assert!(r.det >= 0.0 && r.det <= 1.0 + 1e-12,
        "DET must be in [0,1], got {}", r.det);
    assert!(r.lam >= 0.0 && r.lam <= 1.0 + 1e-12,
        "LAM must be in [0,1], got {}", r.lam);
}

#[test]
fn rqa_large_epsilon_full_recurrence() {
    // If epsilon exceeds the diameter of the phase-space cloud, every pair recurs.
    let data: Vec<f64> = (0..50).map(|i| i as f64 * 0.01).collect();
    let r = rqa(&data, 3, 1, 1e6, 2);
    assert!((r.rr - 1.0).abs() < 1e-12,
        "huge epsilon should give RR=1, got {}", r.rr);
}

#[test]
fn rqa_tiny_epsilon_no_recurrence() {
    // For strictly monotone data with distinct values, a tiny epsilon gives
    // RR = 0 off the main diagonal.
    let data: Vec<f64> = (0..80).map(|i| i as f64).collect();
    let r = rqa(&data, 3, 1, 1e-12, 2);
    assert!(r.rr < 1e-12,
        "tiny epsilon on monotone data should give RR≈0, got {}", r.rr);
}

// ═══════════════════════════════════════════════════════════════════════════
// lmin gating
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rqa_lmin_increase_decreases_det() {
    // Increasing lmin excludes shorter lines → DET monotonically non-increasing.
    let n = 300;
    let mut rng = tambear::rng::Xoshiro256::new(7);
    let data: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.1).sin() + 0.1 * tambear::rng::sample_normal(&mut rng, 0.0, 1.0))
        .collect();
    let std_d = {
        let mean = data.iter().sum::<f64>() / n as f64;
        (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    let r2 = rqa(&data, 3, 1, 0.2 * std_d, 2);
    let r5 = rqa(&data, 3, 1, 0.2 * std_d, 5);
    assert!(r2.det.is_finite() && r5.det.is_finite());
    assert!(r5.det <= r2.det + 1e-12,
        "DET(lmin=5)={} should not exceed DET(lmin=2)={}", r5.det, r2.det);
}

// ═══════════════════════════════════════════════════════════════════════════
// Entropy signs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rqa_entropy_nonnegative() {
    let n = 200;
    let data: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.07).sin()).collect();
    let r = rqa(&data, 3, 1, 0.1, 2);
    assert!(r.entr >= 0.0 || r.entr.is_nan(),
        "diagonal-length Shannon entropy should be ≥ 0, got {}", r.entr);
}

#[test]
fn rqa_constant_signal_entropy_maximal() {
    // A constant signal is pathological for RQA entropy: each diagonal k
    // contributes one run of unique length n_vec-k, so every length bin has
    // count 1 and P(l) = 1/(n_vec-2) for each of the (n_vec-2) included
    // lengths (k=1..n_vec-2; k=n_vec-1 is excluded at lmin=2).
    // Therefore ENTR = ln(n_vec - 2) exactly (natural log).
    let n = 60usize;
    let m = 3usize;
    let tau = 1usize;
    let data = vec![5.0_f64; n];
    let r = rqa(&data, m, tau, 0.5, 2);
    let n_vec = n - (m - 1) * tau;
    let expected_entr = ((n_vec - 2) as f64).ln();
    assert!((r.entr - expected_entr).abs() < 1e-10,
        "constant signal ENTR should be ln({}) = {}, got {}",
        n_vec - 2, expected_entr, r.entr);
}

// ═══════════════════════════════════════════════════════════════════════════
// Result struct is sane (no panics, nan() helper)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rqa_nan_helper_is_all_nan() {
    let r = RqaResult::nan();
    assert!(r.rr.is_nan() && r.det.is_nan() && r.lam.is_nan() && r.entr.is_nan());
    assert_eq!(r.lmax, 0);
    assert!(r.l_avg.is_nan() && r.tt.is_nan());
}
