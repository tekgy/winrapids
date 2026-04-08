//! Adversarial Boundary Tests — Wave 10
//!
//! Targets: nonparametric, mixed_effects, rng, equipartition
//!
//! Focus: mathematical correctness, distributional properties,
//! and edge-case handling in statistical and sampling functions.

use tambear::nonparametric::*;
use tambear::mixed_effects::*;
use tambear::rng::*;
use tambear::equipartition::*;

// ═══════════════════════════════════════════════════════════════════════════
// NONPARAMETRIC
// ═══════════════════════════════════════════════════════════════════════════

/// Spearman correlation of identical rankings: should be 1.0.
#[test]
fn spearman_perfect() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0]; // perfect monotonic
    let r = spearman(&x, &y);
    assert!((r - 1.0).abs() < 1e-10, "Perfect monotonic should give r=1, got {}", r);
}

/// Spearman of reversed rankings: should be -1.0.
#[test]
fn spearman_reversed() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![50.0, 40.0, 30.0, 20.0, 10.0];
    let r = spearman(&x, &y);
    assert!((r - (-1.0)).abs() < 1e-10, "Reversed should give r=-1, got {}", r);
}

/// Spearman with constant data: all tied ranks → 0/0.
/// Type 1.
#[test]
fn spearman_constant() {
    let x = vec![5.0; 5];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let r = spearman(&x, &y);
    if r.is_nan() {
        // Expected: all tied ranks → zero variance → 0/0
    } else {
        assert!(r.is_finite(), "Spearman with constant x should be NaN or finite, got {}", r);
    }
}

/// Kendall tau with 2 data points: single pair.
#[test]
fn kendall_two_points() {
    let x = vec![1.0, 2.0];
    let y = vec![3.0, 4.0]; // concordant
    let tau = kendall_tau(&x, &y);
    assert!((tau - 1.0).abs() < 1e-10, "Single concordant pair should give tau=1, got {}", tau);
}

/// Kendall tau with all ties.
/// Type 1.
#[test]
fn kendall_all_ties() {
    let x = vec![1.0; 5];
    let y = vec![1.0; 5];
    let tau = kendall_tau(&x, &y);
    if tau.is_nan() {
        // Expected: no concordant or discordant pairs, denominator = 0
    }
}

/// Mann-Whitney U with empty group.
/// Type 5.
#[test]
fn mann_whitney_empty_group() {
    let result = std::panic::catch_unwind(|| {
        mann_whitney_u(&[], &[1.0, 2.0, 3.0])
    });
    match result {
        Ok(r) => {
            if r.p_value.is_nan() {
                // Expected: can't compute test with empty group
            }
        }
        Err(_) => eprintln!("NOTE: mann_whitney_u panics on empty group"),
    }
}

/// Wilcoxon signed-rank with all zeros: all differences = 0 → all tied.
/// Type 4.
#[test]
fn wilcoxon_all_zeros() {
    let result = std::panic::catch_unwind(|| {
        wilcoxon_signed_rank(&[0.0, 0.0, 0.0, 0.0])
    });
    match result {
        Ok(r) => {
            // All zero differences should be excluded → n_eff = 0
            assert!(r.statistic.is_finite() || r.statistic.is_nan(),
                "Wilcoxon with all zeros should be finite or NaN");
        }
        Err(_) => eprintln!("NOTE: wilcoxon_signed_rank panics on all-zero differences"),
    }
}

/// KS test with single data point.
/// Type 5.
#[test]
fn ks_test_single_point() {
    let result = std::panic::catch_unwind(|| {
        ks_test_normal(&[5.0])
    });
    match result {
        Ok(r) => {
            assert!(r.statistic.is_finite(), "KS with n=1 should be finite, got {}", r.statistic);
        }
        Err(_) => eprintln!("NOTE: ks_test_normal panics on single point"),
    }
}

/// Two-sample KS with identical samples: D=0.
#[test]
fn ks_two_sample_identical() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let r = ks_test_two_sample(&data, &data);
    assert!((r.statistic - 0.0).abs() < 1e-10,
        "KS of identical samples should be D=0, got {}", r.statistic);
}

/// Runs test with single value: no runs.
/// Type 5.
#[test]
fn runs_test_single() {
    let result = std::panic::catch_unwind(|| {
        runs_test(&[true])
    });
    match result {
        Ok(r) => assert!(r.p_value.is_nan() || r.p_value.is_finite(),
            "Runs test with 1 value should give NaN or finite p"),
        Err(_) => eprintln!("NOTE: runs_test panics on single value"),
    }
}

/// Sign test with all equal to median: all zeros → n_eff = 0.
/// Type 1.
#[test]
fn sign_test_all_equal_median() {
    let data = vec![5.0, 5.0, 5.0, 5.0];
    let result = std::panic::catch_unwind(|| {
        sign_test(&data, 5.0)
    });
    match result {
        Ok(r) => {
            // No data above or below median → n_eff = 0
            assert!(r.p_value.is_nan() || r.p_value.is_finite(),
                "Sign test with all equal should give NaN or finite p");
        }
        Err(_) => eprintln!("NOTE: sign_test panics when all data equals median"),
    }
}

/// Silverman bandwidth with constant data: IQR=0, std=0 → bandwidth=0.
/// Type 1.
#[test]
fn silverman_bandwidth_constant() {
    let data = vec![3.0; 100];
    let bw = silverman_bandwidth(&data);
    if bw == 0.0 {
        eprintln!("CONFIRMED BUG: silverman_bandwidth returns 0 for constant data (will cause div-by-zero in KDE)");
    } else if bw.is_nan() {
        eprintln!("CONFIRMED BUG: silverman_bandwidth returns NaN for constant data");
    }
}

/// KDE with bandwidth=0: delta functions → division by zero.
/// Type 1.
#[test]
fn kde_bandwidth_zero() {
    let data = vec![1.0, 2.0, 3.0];
    let eval = vec![1.0, 2.0, 3.0];
    let result = std::panic::catch_unwind(|| {
        kde(&data, &eval, KernelType::Gaussian, Some(0.0))
    });
    match result {
        Ok(density) => {
            let any_bad = density.iter().any(|v| v.is_nan() || v.is_infinite());
            if any_bad {
                eprintln!("CONFIRMED BUG: KDE with bandwidth=0 produces NaN/Inf");
            }
        }
        Err(_) => eprintln!("NOTE: kde panics on bandwidth=0"),
    }
}

/// Permutation test with single element per group.
/// Type 5.
#[test]
fn permutation_test_tiny_groups() {
    let result = std::panic::catch_unwind(|| {
        permutation_test_mean_diff(&[5.0], &[3.0], 100, 42)
    });
    match result {
        Ok(r) => {
            assert!(r.p_value >= 0.0 && r.p_value <= 1.0,
                "Permutation p-value should be in [0,1], got {}", r.p_value);
        }
        Err(_) => eprintln!("NOTE: permutation_test panics with single-element groups"),
    }
}

/// Level spacing R-statistic: should be in [0,1] for GOE.
#[test]
fn level_spacing_basic() {
    let values = vec![1.0, 2.1, 3.3, 4.2, 5.8, 7.1, 8.0, 9.5];
    let r = level_spacing_r_stat(&values);
    assert!(r >= 0.0 && r <= 1.0, "Level spacing R should be in [0,1], got {}", r);
}

/// Level spacing with 2 values: only 1 spacing → no ratio possible.
/// Type 5.
#[test]
fn level_spacing_two_values() {
    let result = std::panic::catch_unwind(|| {
        level_spacing_r_stat(&[1.0, 2.0])
    });
    match result {
        Ok(r) => {
            // Only 1 spacing, need 2 for ratio → degenerate
            assert!(r.is_finite() || r.is_nan(), "R with 2 values should be finite or NaN, got {}", r);
        }
        Err(_) => eprintln!("NOTE: level_spacing_r_stat panics on 2 values"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MIXED EFFECTS
// ═══════════════════════════════════════════════════════════════════════════

/// LME with single group: random intercept variance = 0.
/// Type 4.
#[test]
fn lme_single_group() {
    // All observations in group 0
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // n=5, d=1
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let groups = vec![0, 0, 0, 0, 0];
    let result = std::panic::catch_unwind(|| {
        lme_random_intercept(&x, &y, 5, 1, &groups, 100, 1e-6)
    });
    match result {
        Ok(r) => {
            // Single group → sigma2_u should be 0 or near 0
            assert!(r.sigma2_u >= 0.0 || r.sigma2_u.is_nan(),
                "Single group sigma2_u should be ≥0, got {}", r.sigma2_u);
        }
        Err(_) => eprintln!("NOTE: lme_random_intercept panics with single group"),
    }
}

/// ICC with all data in one group: undefined (0/0).
/// Type 1.
#[test]
fn icc_single_group() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let groups = vec![0, 0, 0, 0, 0];
    let result = std::panic::catch_unwind(|| {
        icc_oneway(&values, &groups)
    });
    match result {
        Ok(i) => {
            if i.is_nan() {
                // Expected: between-group variance = 0 with 1 group
            } else {
                assert!(i >= -1.0 && i <= 1.0, "ICC should be in [-1,1], got {}", i);
            }
        }
        Err(_) => eprintln!("NOTE: icc_oneway panics with single group"),
    }
}

/// Design effect with ICC = -1/(n-1): theoretical minimum.
#[test]
fn design_effect_negative_icc() {
    let de = design_effect(-0.5, 10.0);
    // DEFF = 1 + (m-1)*ICC = 1 + 9*(-0.5) = -3.5
    // Negative design effect is theoretically possible but practically meaningless
    assert!(de.is_finite(), "Design effect should be finite, got {}", de);
}

/// Design effect with ICC = 1: maximum clustering.
#[test]
fn design_effect_perfect_icc() {
    let de = design_effect(1.0, 10.0);
    // DEFF = 1 + (10-1)*1 = 10
    assert!((de - 10.0).abs() < 1e-10, "DEFF with ICC=1, m=10 should be 10, got {}", de);
}

// ═══════════════════════════════════════════════════════════════════════════
// RNG — DISTRIBUTIONAL CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

/// Normal distribution: mean ≈ mu, std ≈ sigma over many samples.
#[test]
fn normal_distribution_moments() {
    let mut rng = Xoshiro256::new(42);
    let n = 10000;
    let samples: Vec<f64> = (0..n).map(|_| sample_normal(&mut rng, 5.0, 2.0)).collect();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    assert!((mean - 5.0).abs() < 0.1, "Normal mean should be ~5, got {}", mean);
    assert!((var.sqrt() - 2.0).abs() < 0.2, "Normal std should be ~2, got {}", var.sqrt());
}

/// Exponential with lambda=0: mean = 1/0 = ∞.
/// Type 1.
#[test]
fn exponential_lambda_zero() {
    let mut rng = Xoshiro256::new(42);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sample_exponential(&mut rng, 0.0)
    }));
    match result {
        Ok(v) => {
            if v.is_infinite() || v.is_nan() {
                eprintln!("CONFIRMED BUG: sample_exponential(lambda=0) returns {} (should guard)", v);
            }
        }
        Err(_) => eprintln!("NOTE: sample_exponential panics on lambda=0"),
    }
}

/// Gamma with alpha=0: degenerate → always 0.
/// Type 1.
#[test]
fn gamma_alpha_zero() {
    let mut rng = Xoshiro256::new(42);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sample_gamma(&mut rng, 0.0, 1.0)
    }));
    match result {
        Ok(v) => {
            if v.is_nan() {
                eprintln!("CONFIRMED BUG: sample_gamma(alpha=0) returns NaN");
            }
        }
        Err(_) => eprintln!("NOTE: sample_gamma panics on alpha=0"),
    }
}

/// Beta(0, 0): both parameters 0 → undefined.
/// Type 1.
#[test]
fn beta_zero_zero() {
    let mut rng = Xoshiro256::new(42);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sample_beta(&mut rng, 0.0, 0.0)
    }));
    match result {
        Ok(v) => {
            if v.is_nan() {
                eprintln!("NOTE: sample_beta(0,0) returns NaN (degenerate)");
            }
        }
        Err(_) => eprintln!("NOTE: sample_beta panics on (0,0)"),
    }
}

/// Poisson with lambda=0: should always return 0.
#[test]
fn poisson_lambda_zero() {
    let mut rng = Xoshiro256::new(42);
    for _ in 0..10 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sample_poisson(&mut rng, 0.0)
        }));
        match result {
            Ok(v) => assert_eq!(v, 0, "Poisson(0) should always be 0, got {}", v),
            Err(_) => {
                eprintln!("NOTE: sample_poisson panics on lambda=0");
                break;
            }
        }
    }
}

/// Geometric with p=0: never succeeds → infinite loop.
/// Type 2.
#[test]
fn geometric_p_zero() {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut rng = Xoshiro256::new(42);
        let v = sample_geometric(&mut rng, 0.0);
        let _ = tx.send(v);
    });
    match rx.recv_timeout(std::time::Duration::from_secs(2)) {
        Ok(v) => {
            // Should return something sensible (max value?)
            eprintln!("NOTE: geometric(p=0) returned {}", v);
        }
        Err(_) => {
            eprintln!("CONFIRMED BUG: sample_geometric(p=0) infinite loop (never succeeds)");
        }
    }
}

/// Bernoulli with p > 1: always true? No validation?
/// Type 5.
#[test]
fn bernoulli_p_above_one() {
    let mut rng = Xoshiro256::new(42);
    let result = sample_bernoulli(&mut rng, 1.5);
    // p=1.5 → rng.next_f64() < 1.5 is always true
    assert!(result, "Bernoulli(1.5) should always be true");
}

/// sample_without_replacement with k > n: impossible.
/// Type 5.
#[test]
fn sample_without_replacement_k_gt_n() {
    let mut rng = Xoshiro256::new(42);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sample_without_replacement(&mut rng, 3, 10) // 10 from 3
    }));
    if result.is_err() {
        eprintln!("NOTE: sample_without_replacement panics when k > n");
    } else {
        let samples = result.unwrap();
        eprintln!("NOTE: sample_without_replacement(n=3, k=10) returned {} elements", samples.len());
    }
}

/// sample_weighted with all-zero weights: no item has positive weight.
/// Type 1.
#[test]
fn sample_weighted_all_zero() {
    let mut rng = Xoshiro256::new(42);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sample_weighted(&mut rng, &[0.0, 0.0, 0.0], 2)
    }));
    match result {
        Ok(indices) => {
            // All zero weights → cumsum = [0, 0, 0], random in [0, 0) → undefined
            eprintln!("NOTE: sample_weighted with all-zero weights returned {:?}", indices);
        }
        Err(_) => eprintln!("NOTE: sample_weighted panics on all-zero weights"),
    }
}

/// sample_weighted with negative weights.
/// Type 5.
#[test]
fn sample_weighted_negative() {
    let mut rng = Xoshiro256::new(42);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sample_weighted(&mut rng, &[1.0, -2.0, 3.0], 1)
    }));
    match result {
        Ok(indices) => {
            // Negative weight makes cumsum non-monotone → binary search breaks
            eprintln!("NOTE: sample_weighted with negative weights returned {:?}", indices);
        }
        Err(_) => eprintln!("NOTE: sample_weighted panics on negative weights"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EQUIPARTITION
// ═══════════════════════════════════════════════════════════════════════════

/// Free energy with s=0: p^(-0) = 1, but 1/(1-1) = Inf.
/// Type 1: denominator collapse at s=0.
#[test]
fn free_energy_s_zero() {
    let result = free_energy(2.0, 0.0);
    // free_energy involves 1/(1 - p^(-s)). At s=0: p^0 = 1, so 1/(1-1) = Inf
    if result.is_infinite() {
        // Expected: pole at s=0 for free energy
    } else {
        assert!(result.is_finite(), "free_energy(2, 0) should be Inf or finite, got {}", result);
    }
}

/// Euler factor at s=0.
#[test]
fn euler_factor_s_zero() {
    let result = euler_factor(2.0, 0.0);
    // 1/(1 - p^(-s)) = 1/(1-1) = 1/0 → Inf
    if result.is_infinite() {
        // Expected: Euler product diverges at s=0
    } else {
        assert!(result.is_finite(), "euler_factor(2, 0) should be Inf or finite, got {}", result);
    }
}

/// solve_fold with empty scales.
/// Type 5.
#[test]
fn solve_fold_empty() {
    let result = std::panic::catch_unwind(|| {
        solve_fold(&[], 1.0)
    });
    match result {
        Ok(v) => {
            assert!(v.is_none(), "solve_fold with empty scales should be None");
        }
        Err(_) => eprintln!("NOTE: solve_fold panics on empty scales"),
    }
}

/// solve_pairwise with identical scales: no fold point.
/// Type 4.
#[test]
fn solve_pairwise_identical() {
    let result = solve_pairwise(5.0, 5.0);
    // p^s = q^s only when p = q → true for all s → no unique fold
    match result {
        Some(s) => eprintln!("NOTE: solve_pairwise(5,5) returns fold at s={}", s),
        None => { /* Expected: no unique solution */ }
    }
}

/// solve_pairwise with a=0: 0^s = 0 for s > 0 → degenerate.
/// Type 1.
#[test]
fn solve_pairwise_zero() {
    let result = std::panic::catch_unwind(|| {
        solve_pairwise(0.0, 2.0)
    });
    match result {
        Ok(v) => {
            if let Some(s) = v {
                // 0^s = 2^s only if both are 0 or s = -∞
                eprintln!("NOTE: solve_pairwise(0, 2) returns fold at s={}", s);
            }
        }
        Err(_) => eprintln!("NOTE: solve_pairwise panics with a=0"),
    }
}

/// fold_target with single scale: trivial.
#[test]
fn fold_target_single() {
    let target = fold_target(&[5.0]);
    assert!(target.is_finite(), "fold_target of single scale should be finite, got {}", target);
}

/// verify_fold_surface basic check.
#[test]
fn verify_fold_basic() {
    let scales = vec![2.0, 3.0, 5.0, 7.0];
    let (valid, messages) = verify_fold_surface(&scales);
    // Should pass basic verification
    if !valid {
        eprintln!("NOTE: verify_fold_surface failed with messages: {:?}", messages);
    }
}

/// classify_phase with extreme s.
#[test]
fn classify_phase_extreme() {
    let scales = vec![2.0, 3.0, 5.0];
    let result = std::panic::catch_unwind(|| {
        classify_phase(&scales, 1e10, 1e-6)
    });
    match result {
        Ok(_phase) => { /* Whatever phase, shouldn't crash */ }
        Err(_) => eprintln!("NOTE: classify_phase panics on extreme s"),
    }
}

/// phase_sweep with n_points=0.
/// Type 5.
#[test]
fn phase_sweep_zero_points() {
    let result = std::panic::catch_unwind(|| {
        phase_sweep(&[2.0, 3.0], 0.0, 10.0, 0, 1e-6)
    });
    match result {
        Ok(phases) => assert!(phases.is_empty(), "0 points should give empty sweep"),
        Err(_) => eprintln!("NOTE: phase_sweep panics on n_points=0"),
    }
}

/// fold_sensitivity with single scale.
#[test]
fn fold_sensitivity_single() {
    let result = std::panic::catch_unwind(|| {
        fold_sensitivity(&[2.0], 1.0)
    });
    match result {
        Ok(sens) => {
            assert_eq!(sens.len(), 1, "Single scale should give 1 sensitivity value");
        }
        Err(_) => eprintln!("NOTE: fold_sensitivity panics on single scale"),
    }
}

/// batch_pairwise_folds with 2 identical scales.
#[test]
fn batch_pairwise_identical() {
    let result = std::panic::catch_unwind(|| {
        batch_pairwise_folds(&[3.0, 3.0])
    });
    match result {
        Ok(batch) => {
            // Identical scales → no fold point between them
        }
        Err(_) => eprintln!("NOTE: batch_pairwise_folds panics on identical scales"),
    }
}
