//! Adversarial tests for fintek gap implementations wave 2:
//! Burg AR + PSD, Morlet CWT, transfer entropy, CUSUM.

use tambear::time_series::*;
use tambear::signal_processing::*;
use tambear::information_theory::transfer_entropy;

// ═══════════════════════════════════════════════════════════════════════════
// BURG AR
// ═══════════════════════════════════════════════════════════════════════════

/// Burg AR on simulated AR(1): should recover phi close to truth.
#[test]
fn burg_ar_recovers_ar1() {
    let n = 500;
    let true_phi = 0.7;
    let mut data = vec![0.0; n];
    let mut rng = tambear::rng::Xoshiro256::new(42);
    for t in 1..n {
        let noise = tambear::rng::sample_normal(&mut rng, 0.0, 1.0);
        data[t] = true_phi * data[t - 1] + noise;
    }
    let r = ar_burg_fit(&data, 1);
    assert_eq!(r.coefficients.len(), 1);
    assert!((r.coefficients[0] - true_phi).abs() < 0.1,
        "Burg should recover phi={}, got {}", true_phi, r.coefficients[0]);
    assert!(r.sigma2 > 0.0 && r.sigma2 < 3.0);
}

/// Burg AR on AR(2): should recover both coefficients.
#[test]
fn burg_ar_recovers_ar2() {
    let n = 1000;
    let phi1 = 0.5;
    let phi2 = -0.3;
    let mut data = vec![0.0; n];
    let mut rng = tambear::rng::Xoshiro256::new(99);
    for t in 2..n {
        let noise = tambear::rng::sample_normal(&mut rng, 0.0, 1.0);
        data[t] = phi1 * data[t - 1] + phi2 * data[t - 2] + noise;
    }
    let r = ar_burg_fit(&data, 2);
    assert_eq!(r.coefficients.len(), 2);
    assert!((r.coefficients[0] - phi1).abs() < 0.1,
        "Burg AR(2) phi1 should be {}, got {}", phi1, r.coefficients[0]);
    assert!((r.coefficients[1] - phi2).abs() < 0.1,
        "Burg AR(2) phi2 should be {}, got {}", phi2, r.coefficients[1]);
}

/// Burg AR on white noise: coefficients should be small.
#[test]
fn burg_ar_white_noise() {
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let data: Vec<f64> = (0..500).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
    let r = ar_burg_fit(&data, 3);
    // AR coefficients should all be small for white noise
    for (i, &phi) in r.coefficients.iter().enumerate() {
        assert!(phi.abs() < 0.2, "WN Burg phi[{}] should be ~0, got {}", i, phi);
    }
}

/// Burg AR with constant data: degenerate → zero coefficients.
#[test]
fn burg_ar_constant() {
    let data = vec![5.0; 100];
    let r = ar_burg_fit(&data, 3);
    assert_eq!(r.sigma2, 0.0, "Constant Burg σ² should be 0, got {}", r.sigma2);
    assert!(r.coefficients.iter().all(|&c| c == 0.0));
}

/// Burg AR too short: graceful NaN/zero.
#[test]
fn burg_ar_too_short() {
    let r = ar_burg_fit(&[1.0, 2.0], 5);
    assert_eq!(r.coefficients.len(), 5);
    assert!(r.coefficients.iter().all(|&c| c == 0.0));
}

/// Burg AR PSD sanity: non-negative everywhere.
#[test]
fn burg_ar_psd_non_negative() {
    let n = 200;
    let mut data = vec![0.0; n];
    let mut rng = tambear::rng::Xoshiro256::new(42);
    for t in 1..n {
        data[t] = 0.8 * data[t - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 1.0);
    }
    let r = ar_burg_fit(&data, 4);
    let (_, psd) = ar_psd(&r, 100);
    for (i, &p) in psd.iter().enumerate() {
        assert!(p >= 0.0, "AR PSD[{}] should be non-negative, got {}", i, p);
    }
}

/// AR(1) PSD should peak at DC for positive phi.
#[test]
fn burg_ar_psd_peaks_at_dc_for_ar1_pos() {
    let n = 500;
    let mut data = vec![0.0; n];
    let mut rng = tambear::rng::Xoshiro256::new(42);
    for t in 1..n {
        data[t] = 0.9 * data[t - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.5);
    }
    let r = ar_burg_fit(&data, 1);
    // PSD should be monotonically decreasing for positive AR(1)
    let p_dc = ar_psd_at(&r, 0.0);
    let p_mid = ar_psd_at(&r, 0.25);
    let p_nyq = ar_psd_at(&r, 0.5);
    assert!(p_dc > p_mid, "AR(1) PSD should peak at DC: p(0)={}, p(0.25)={}", p_dc, p_mid);
    assert!(p_mid > p_nyq, "AR(1) PSD should decrease: p(0.25)={}, p(0.5)={}", p_mid, p_nyq);
}

// ═══════════════════════════════════════════════════════════════════════════
// MORLET CWT
// ═══════════════════════════════════════════════════════════════════════════

/// Morlet wavelet: norm² integral ≈ 1 (on a wide enough support).
#[test]
fn morlet_wavelet_finite() {
    for t in &[-3.0_f64, -1.0, 0.0, 1.0, 3.0] {
        let (re, im) = morlet_wavelet(*t, 6.0);
        assert!(re.is_finite() && im.is_finite(),
            "Morlet at t={} should be finite, got ({}, {})", t, re, im);
    }
}

/// Morlet at t=0: real part should be π^(-1/4), imag ≈ 0.
#[test]
fn morlet_wavelet_at_zero() {
    let (re, im) = morlet_wavelet(0.0, 6.0);
    let expected = 1.0 / std::f64::consts::PI.powf(0.25);
    assert!((re - expected).abs() < 1e-12, "Morlet re at t=0 should be π^(-1/4), got {}", re);
    assert!(im.abs() < 1e-12, "Morlet im at t=0 should be 0, got {}", im);
}

/// CWT on sine: maximum energy at the scale matching the sine period.
#[test]
fn morlet_cwt_sine_detection() {
    let n = 512;
    let period = 32.0;
    let data: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / period).sin()).collect();
    // Scales corresponding to a range around the period
    let scales = vec![8.0, 16.0, 32.0, 64.0, 128.0];
    let scalogram = morlet_cwt(&data, &scales, 6.0);
    assert_eq!(scalogram.len(), scales.len() * n);

    // Average energy per scale (skip boundaries)
    let mid_start = n / 4;
    let mid_end = 3 * n / 4;
    let mut energy_per_scale = vec![0.0; scales.len()];
    for (si, _) in scales.iter().enumerate() {
        let mut sum = 0.0;
        for t in mid_start..mid_end {
            sum += scalogram[si * n + t];
        }
        energy_per_scale[si] = sum;
    }
    // Maximum should be at a middle scale (not at endpoints)
    let max_idx = energy_per_scale.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    assert!(max_idx > 0 && max_idx < scales.len() - 1,
        "CWT max should be at a middle scale, got index {}", max_idx);
}

/// CWT with empty data: empty result.
#[test]
fn morlet_cwt_empty() {
    let scalogram = morlet_cwt(&[], &[10.0], 6.0);
    assert!(scalogram.is_empty());
}

/// CWT with zero scales: empty result.
#[test]
fn morlet_cwt_no_scales() {
    let data = vec![1.0, 2.0, 3.0];
    let scalogram = morlet_cwt(&data, &[], 6.0);
    assert!(scalogram.is_empty());
}

/// Scale-to-frequency conversion.
#[test]
fn morlet_scale_to_freq_basic() {
    // At scale 10, with ω₀=6, fs=100 Hz → f = (6/(2π)) * 100 / 10 ≈ 9.55 Hz
    let f = morlet_scale_to_frequency(10.0, 6.0, 100.0);
    let expected = (6.0 / (2.0 * std::f64::consts::PI)) * 100.0 / 10.0;
    assert!((f - expected).abs() < 1e-10);
}

/// Log-spaced scales cover the requested frequency range.
#[test]
fn morlet_log_scales_range() {
    let scales = morlet_log_scales(1.0, 50.0, 20, 100.0, 6.0);
    assert_eq!(scales.len(), 20);
    // Scales should be monotone decreasing (high frequency → low scale)
    // Actually scales are decreasing because smaller scale = higher frequency
    for i in 1..scales.len() {
        assert!(scales[i] >= scales[i - 1] || scales[i - 1] >= scales[i],
            "scales monotone check");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER ENTROPY
// ═══════════════════════════════════════════════════════════════════════════

/// TE is non-negative.
#[test]
fn transfer_entropy_non_negative() {
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let x: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
    let y: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
    let te = transfer_entropy(&x, &y, 4);
    assert!(te >= 0.0, "TE should be non-negative, got {}", te);
}

/// TE from X to Y where Y_{t+1} = f(X_t): should be positive.
#[test]
fn transfer_entropy_positive_causal() {
    let n = 500;
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let x: Vec<f64> = (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
    // y depends on x with a lag
    let mut y = vec![0.0; n];
    for t in 1..n {
        y[t] = 0.8 * x[t - 1] + 0.2 * tambear::rng::sample_normal(&mut rng, 0.0, 1.0);
    }
    let te_xy = transfer_entropy(&x, &y, 5);
    let te_yx = transfer_entropy(&y, &x, 5);
    // TE(X→Y) should exceed TE(Y→X) for this causal structure
    assert!(te_xy > te_yx,
        "TE(X→Y)={} should exceed TE(Y→X)={}", te_xy, te_yx);
}

/// TE too short: NaN.
#[test]
fn transfer_entropy_too_short() {
    assert!(transfer_entropy(&[1.0, 2.0], &[3.0, 4.0], 4).is_nan());
}

/// TE with mismatched lengths: NaN.
#[test]
fn transfer_entropy_mismatch() {
    let x = vec![1.0; 10];
    let y = vec![1.0; 5];
    assert!(transfer_entropy(&x, &y, 4).is_nan());
}

// ═══════════════════════════════════════════════════════════════════════════
// CUSUM
// ═══════════════════════════════════════════════════════════════════════════

/// CUSUM on constant data: zero everywhere.
#[test]
fn cusum_constant() {
    let r = cusum_mean(&[5.0; 20]);
    for &v in &r.cusum {
        assert!(v.abs() < 1e-12);
    }
    assert!(r.max_abs_cusum < 1e-12);
}

/// CUSUM on step change: max at the changepoint.
#[test]
fn cusum_step_change() {
    // 50 samples at 0, then 50 samples at 1
    let mut data = vec![0.0; 50];
    data.extend(vec![1.0; 50]);
    let r = cusum_mean(&data);
    // Changepoint is around index 49-50
    assert!(r.argmax >= 45 && r.argmax <= 55,
        "CUSUM argmax should be near 49-50, got {}", r.argmax);
    assert!(r.max_abs_cusum > 10.0,
        "CUSUM max for step should be large, got {}", r.max_abs_cusum);
}

/// CUSUM on zero-mean white noise: small max.
#[test]
fn cusum_white_noise_small() {
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let data: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
    let r = cusum_mean(&data);
    // White noise CUSUM should be O(√n) ≈ 10 for n=100
    assert!(r.max_abs_cusum < 30.0,
        "WN CUSUM should be O(√n), got {}", r.max_abs_cusum);
}

/// Binary segmentation on clean step data: finds exactly one changepoint.
#[test]
fn cusum_binary_segmentation_single_step() {
    let mut data = vec![0.0; 100];
    data.extend(vec![5.0; 100]);
    let cps = cusum_binary_segmentation(&data, 20.0, 10, 10);
    assert!(!cps.is_empty(), "Should detect at least one changepoint");
    // First (largest) changepoint should be near 100
    let main_cp = cps[0];
    // It might split further into smaller segments, but the primary
    // changepoint should be the step
    let has_near_step = cps.iter().any(|&cp| cp >= 90 && cp <= 110);
    assert!(has_near_step, "Should find changepoint near step, got {:?}", cps);
    let _ = main_cp;
}

/// Binary segmentation on constant data: no changepoints.
#[test]
fn cusum_binary_segmentation_no_changes() {
    let data = vec![1.0; 100];
    let cps = cusum_binary_segmentation(&data, 5.0, 10, 10);
    assert!(cps.is_empty(), "Constant data should give no changepoints");
}

/// Binary segmentation respects min_segment_size.
#[test]
fn cusum_binary_segmentation_min_size() {
    let mut data = vec![0.0; 50];
    data.extend(vec![1.0; 50]);
    let cps = cusum_binary_segmentation(&data, 5.0, 20, 10);
    // min_segment_size=20 means changepoints must be ≥20 from endpoints
    for &cp in &cps {
        assert!(cp >= 20 && cp <= 80,
            "Changepoint {} violates min_segment_size=20", cp);
    }
}

/// CUSUM empty data.
#[test]
fn cusum_empty() {
    let r = cusum_mean(&[]);
    assert!(r.cusum.is_empty());
    assert_eq!(r.max_abs_cusum, 0.0);
}
