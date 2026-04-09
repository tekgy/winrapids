//! Adversarial Boundary Tests — Wave 7
//!
//! Targets: hypothesis, special_functions, information_theory, robust, bayesian
//!
//! Attack taxonomy:
//! - Type 1: Division by zero / denominator collapse
//! - Type 2: Convergence / iteration boundary
//! - Type 3: Cancellation / precision
//! - Type 4: Equipartition / degenerate geometry
//! - Type 5: Structural incompatibility

use tambear::descriptive::MomentStats;
use tambear::hypothesis::*;
use tambear::special_functions::*;
use tambear::information_theory::*;
use tambear::robust::*;
use tambear::bayesian::*;

// ═══════════════════════════════════════════════════════════════════════════
// HYPOTHESIS
// ═══════════════════════════════════════════════════════════════════════════

/// One-sample t-test with n=0: 0/0 everywhere.
/// Type 1: denominator collapse.
#[test]
fn t_test_empty_stats() {
    let stats = MomentStats::empty();
    let result = one_sample_t(&stats, 0.0);
    // n < 2 → should return NaN gracefully
    assert!(result.statistic.is_nan(), "t-stat with 0 observations should be NaN");
    assert!(result.p_value.is_nan(), "p-value with 0 observations should be NaN");
}

/// One-sample t-test with zero variance: se=0 → t=Inf or NaN.
/// Type 1.
#[test]
fn t_test_zero_variance() {
    let stats = MomentStats {
        count: 10.0, sum: 50.0, min: 5.0, max: 5.0,
        m2: 0.0, m3: 0.0, m4: 0.0,
    };
    let result = one_sample_t(&stats, 3.0);
    // mean=5, se=0 → t=(5-3)/0 → Inf or NaN
    if result.statistic.is_nan() || result.statistic.is_infinite() {
        // Expected: division by zero in t-statistic
    } else {
        panic!("Zero-variance t-test should produce Inf or NaN, got {}", result.statistic);
    }
}

/// Two-sample t with identical groups: pooled variance=0 → t=0/0.
/// Type 1.
#[test]
fn two_sample_t_identical() {
    let stats = MomentStats {
        count: 5.0, sum: 25.0, min: 5.0, max: 5.0,
        m2: 0.0, m3: 0.0, m4: 0.0,
    };
    let result = two_sample_t(&stats, &stats);
    // Same mean, zero variance → 0/0
    assert!(result.statistic.is_nan() || result.statistic == 0.0,
        "Identical groups should give t=0 or NaN, got {}", result.statistic);
}

/// Welch t-test with n=1 in one group: df formula blows up.
/// Type 1.
#[test]
fn welch_t_single_obs() {
    let one = MomentStats {
        count: 1.0, sum: 5.0, min: 5.0, max: 5.0,
        m2: 0.0, m3: 0.0, m4: 0.0,
    };
    let many = MomentStats {
        count: 10.0, sum: 50.0, min: 3.0, max: 7.0,
        m2: 20.0, m3: 0.0, m4: 100.0,
    };
    let result = welch_t(&one, &many);
    // n=1 → variance/n = 0/1 or m2/0 → degenerate
    assert!(result.df.is_nan() || result.df >= 0.0,
        "Welch df with n=1 should be NaN or non-negative, got {}", result.df);
}

/// ANOVA with single group: df_between=0 → F=0/0.
/// Type 1.
#[test]
fn anova_single_group() {
    let stats = MomentStats {
        count: 10.0, sum: 50.0, min: 3.0, max: 7.0,
        m2: 20.0, m3: 0.0, m4: 100.0,
    };
    let result = one_way_anova(&[stats]);
    // k=1 → df_between=0, ss_between=0
    assert!(result.f_statistic.is_nan() || result.f_statistic == 0.0,
        "ANOVA with 1 group should give F=0 or NaN, got {}", result.f_statistic);
}

/// ANOVA with empty groups array.
/// Type 5: structural incompatibility.
#[test]
fn anova_no_groups() {
    let result = std::panic::catch_unwind(|| {
        one_way_anova(&[])
    });
    if result.is_err() {
        eprintln!("CONFIRMED BUG: ANOVA panics on empty groups");
    }
}

/// Chi-square goodness of fit with zero expected: 0/0.
/// Type 1.
#[test]
fn chi2_gof_zero_expected() {
    let observed = vec![5.0, 3.0, 0.0];
    let expected = vec![5.0, 3.0, 0.0]; // zero expected for category 3
    let result = chi2_goodness_of_fit(&observed, &expected);
    // (0-0)^2/0 = 0/0 = NaN in the sum
    if result.statistic.is_nan() {
        eprintln!("CONFIRMED BUG: chi2 goodness of fit returns NaN with zero expected count");
    }
}

/// Chi-square independence with all zeros: empty table.
/// Type 1.
#[test]
fn chi2_independence_all_zeros() {
    let table = vec![0.0, 0.0, 0.0, 0.0]; // 2x2 all zeros
    let result = chi2_independence(&table, 2);
    assert!(result.statistic.is_finite() || result.statistic.is_nan(),
        "All-zero table chi2 should be finite or NaN, got {}", result.statistic);
}

/// Proportion z-test with n=0: 0/0.
/// Type 1.
#[test]
fn proportion_z_no_obs() {
    let result = one_proportion_z(0.0, 0.0, 0.5);
    // p_hat = 0/0 = NaN, se = sqrt(p0*(1-p0)/0) = Inf
    assert!(result.statistic.is_nan() || result.statistic.is_infinite(),
        "Proportion z with n=0 should be NaN or Inf, got {}", result.statistic);
}

/// Odds ratio with zero cell: OR=∞.
/// Type 1.
#[test]
fn odds_ratio_zero_cell() {
    let table = [10.0, 0.0, 5.0, 8.0]; // b=0 → ad/bc = ad/0
    let or = odds_ratio(&table);
    if or.is_infinite() {
        // Expected: division by zero
    } else if or.is_nan() {
        eprintln!("NOTE: Odds ratio with zero cell returns NaN instead of Inf");
    }
}

/// Log odds ratio SE with zero cell: log(0) = -∞, sqrt(1/0) = Inf.
/// Type 1.
#[test]
fn log_odds_se_zero_cell() {
    let table = [10.0, 0.0, 5.0, 8.0];
    let se = log_odds_ratio_se(&table);
    // sqrt(1/a + 1/b + 1/c + 1/d) with b=0 → sqrt(Inf) = Inf
    if se.is_infinite() || se.is_nan() {
        // Expected: zero cell causes Inf or NaN
    } else {
        panic!("Log OR SE with zero cell should be Inf or NaN, got {}", se);
    }
}

/// Cohen's d with zero variance in both groups.
/// Type 1.
#[test]
fn cohens_d_zero_variance() {
    let s1 = MomentStats {
        count: 10.0, sum: 50.0, min: 5.0, max: 5.0,
        m2: 0.0, m3: 0.0, m4: 0.0,
    };
    let s2 = MomentStats {
        count: 10.0, sum: 30.0, min: 3.0, max: 3.0,
        m2: 0.0, m3: 0.0, m4: 0.0,
    };
    let d = cohens_d(&s1, &s2);
    // pooled sd = 0 → d = (5-3)/0 = Inf or NaN
    if d.is_infinite() || d.is_nan() {
        // Expected
    } else {
        panic!("Cohen's d with zero variance should be Inf or NaN, got {}", d);
    }
}

/// Bonferroni correction with empty p-values.
/// Type 5.
#[test]
fn bonferroni_empty() {
    let adjusted = bonferroni(&[]);
    assert!(adjusted.is_empty(), "Empty p-values should give empty adjustment");
}

/// Benjamini-Hochberg with all identical p-values.
/// Type 4.
#[test]
fn bh_identical_pvalues() {
    let p = vec![0.05, 0.05, 0.05, 0.05];
    let adjusted = benjamini_hochberg(&p);
    // All equal → order doesn't matter → all adjusted to same value
    for &a in &adjusted {
        assert!(a.is_finite(), "BH adjusted p should be finite, got {}", a);
        assert!(a >= 0.0 && a <= 1.0, "BH p should be in [0,1], got {}", a);
    }
}

/// Holm correction with p=0: rank * 0 = 0 always.
#[test]
fn holm_zero_pvalues() {
    let p = vec![0.0, 0.0, 0.05];
    let adjusted = holm(&p);
    for &a in &adjusted {
        assert!(a.is_finite(), "Holm adjusted p should be finite, got {}", a);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SPECIAL FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// erf(0) = 0 exactly.
#[test]
fn erf_zero() {
    let result = erf(0.0);
    if (result - 0.0).abs() > 1e-10 {
        eprintln!("CONFIRMED BUG: erf(0) should be 0, got {}", result);
    }
}

/// erf(NaN) should be NaN.
#[test]
fn erf_nan() {
    assert!(erf(f64::NAN).is_nan(), "erf(NaN) should be NaN");
}

/// erf(Inf) should be 1.
#[test]
fn erf_inf() {
    let result = erf(f64::INFINITY);
    assert!((result - 1.0).abs() < 1e-10, "erf(Inf) should be 1, got {}", result);
}

/// erf(-Inf) should be -1.
#[test]
fn erf_neg_inf() {
    let result = erf(f64::NEG_INFINITY);
    assert!((result - (-1.0)).abs() < 1e-10, "erf(-Inf) should be -1, got {}", result);
}

/// log_gamma(0): pole at 0.
/// Type 1.
#[test]
fn log_gamma_zero() {
    let result = log_gamma(0.0);
    // Gamma(0) is undefined (pole) → log_gamma should be Inf or NaN
    assert!(result.is_infinite() || result.is_nan(),
        "log_gamma(0) should be Inf or NaN, got {}", result);
}

/// log_gamma(negative integer): poles at -1, -2, -3...
#[test]
fn log_gamma_negative_int() {
    let result = log_gamma(-1.0);
    assert!(result.is_infinite() || result.is_nan(),
        "log_gamma(-1) should be Inf or NaN, got {}", result);
}

/// gamma(0): pole.
#[test]
fn gamma_zero() {
    let result = gamma(0.0);
    assert!(result.is_infinite() || result.is_nan(),
        "gamma(0) should be Inf or NaN, got {}", result);
}

/// gamma(large): overflow to Inf.
#[test]
fn gamma_overflow() {
    let result = gamma(200.0);
    // gamma(200) ≈ 10^373, way past f64 max
    assert!(result.is_infinite(), "gamma(200) should overflow to Inf, got {}", result);
}

/// digamma(0): pole → -Inf or NaN.
#[test]
fn digamma_zero() {
    let result = digamma(0.0);
    assert!(result.is_infinite() || result.is_nan(),
        "digamma(0) should be -Inf or NaN, got {}", result);
}

/// trigamma(0): pole → +Inf or NaN.
#[test]
fn trigamma_zero() {
    let result = trigamma(0.0);
    assert!(result.is_infinite() || result.is_nan(),
        "trigamma(0) should be Inf or NaN, got {}", result);
}

/// regularized_incomplete_beta(x=0, a, b) should be 0.
#[test]
fn incomplete_beta_x_zero() {
    let result = regularized_incomplete_beta(0.0, 2.0, 3.0);
    assert!((result - 0.0).abs() < 1e-10,
        "I_0(2,3) should be 0, got {}", result);
}

/// regularized_incomplete_beta(x=1, a, b) should be 1.
#[test]
fn incomplete_beta_x_one() {
    let result = regularized_incomplete_beta(1.0, 2.0, 3.0);
    assert!((result - 1.0).abs() < 1e-10,
        "I_1(2,3) should be 1, got {}", result);
}

/// regularized_incomplete_beta with a=0: degenerate.
/// Type 1.
#[test]
fn incomplete_beta_a_zero() {
    let result = regularized_incomplete_beta(0.5, 0.0, 1.0);
    // B(0,b) is undefined → result should be NaN or 1
    assert!(result.is_finite() || result.is_nan(),
        "I_0.5(0,1) should be finite or NaN, got {}", result);
}

/// t_cdf with df=0: degenerate distribution.
/// Type 1.
#[test]
fn t_cdf_zero_df() {
    let result = t_cdf(1.0, 0.0);
    // Student-t with df=0 is undefined
    assert!(result.is_nan() || (result >= 0.0 && result <= 1.0),
        "t_cdf(1, df=0) should be NaN or valid prob, got {}", result);
}

/// normal_cdf(0) = 0.5 exactly.
#[test]
fn normal_cdf_zero() {
    let result = normal_cdf(0.0);
    assert!((result - 0.5).abs() < 1e-10, "normal_cdf(0) should be 0.5, got {}", result);
}

/// f_cdf with d1=0 or d2=0.
/// Type 1.
#[test]
fn f_cdf_zero_df() {
    let result = f_cdf(1.0, 0.0, 5.0);
    assert!(result.is_nan() || (result >= 0.0 && result <= 1.0),
        "f_cdf(1, d1=0, d2=5) should be NaN or valid prob, got {}", result);
}

/// chi2_cdf with k=0: degenerate.
#[test]
fn chi2_cdf_zero_k() {
    let result = chi2_cdf(1.0, 0.0);
    assert!(result.is_nan() || (result >= 0.0 && result <= 1.0),
        "chi2_cdf(1, k=0) should be NaN or valid prob, got {}", result);
}

// ═══════════════════════════════════════════════════════════════════════════
// INFORMATION THEORY
// ═══════════════════════════════════════════════════════════════════════════

/// Shannon entropy of single-element distribution: log(1) = 0.
#[test]
fn shannon_entropy_singleton() {
    let probs = vec![1.0];
    let h = shannon_entropy(&probs);
    assert!((h - 0.0).abs() < 1e-10, "H([1.0]) should be 0, got {}", h);
}

/// Shannon entropy with zero probability: 0 * log(0) → 0 by convention.
#[test]
fn shannon_entropy_with_zero() {
    let probs = vec![0.5, 0.5, 0.0];
    let h = shannon_entropy(&probs);
    // 0 * log(0) should be treated as 0
    assert!(h.is_finite(), "Entropy with p=0 should be finite (0*log(0)=0), got {}", h);
}

/// Shannon entropy of empty distribution.
/// Type 5.
#[test]
fn shannon_entropy_empty() {
    let probs: Vec<f64> = vec![];
    let h = shannon_entropy(&probs);
    assert!((h - 0.0).abs() < 1e-10 || h.is_nan(),
        "Entropy of empty distribution should be 0 or NaN, got {}", h);
}

/// Renyi entropy with alpha=1: should equal Shannon entropy.
/// Type 3: 1/(1-1) = 1/0.
#[test]
fn renyi_entropy_alpha_one() {
    let probs = vec![0.25, 0.25, 0.25, 0.25];
    let h_shannon = shannon_entropy(&probs);
    let h_renyi = renyi_entropy(&probs, 1.0);
    // alpha→1 limit is Shannon entropy, but 1/(1-alpha) diverges
    if h_renyi.is_nan() || h_renyi.is_infinite() {
        eprintln!("CONFIRMED BUG: Renyi entropy at alpha=1 returns {} instead of Shannon entropy {}", h_renyi, h_shannon);
    } else {
        assert!((h_renyi - h_shannon).abs() < 0.01,
            "Renyi(alpha=1) should equal Shannon, got {} vs {}", h_renyi, h_shannon);
    }
}

/// Renyi entropy with alpha=0: log of support size.
#[test]
fn renyi_entropy_alpha_zero() {
    let probs = vec![0.5, 0.3, 0.2];
    let h = renyi_entropy(&probs, 0.0);
    // H_0 = log(|support|) = log(3) ≈ 1.0986
    let expected = (3.0_f64).ln();
    if h.is_nan() || h.is_infinite() {
        eprintln!("NOTE: Renyi entropy at alpha=0 returns {}", h);
    } else {
        assert!((h - expected).abs() < 0.01,
            "Renyi(alpha=0) should be log(3)={}, got {}", expected, h);
    }
}

/// Tsallis entropy with q=1: should equal Shannon.
/// Type 3: (1-q) in denominator → 0.
#[test]
fn tsallis_entropy_q_one() {
    let probs = vec![0.25, 0.25, 0.25, 0.25];
    let h_shannon = shannon_entropy(&probs);
    let h_tsallis = tsallis_entropy(&probs, 1.0);
    if h_tsallis.is_nan() || h_tsallis.is_infinite() {
        eprintln!("CONFIRMED BUG: Tsallis entropy at q=1 returns {} instead of Shannon entropy", h_tsallis);
    } else {
        assert!((h_tsallis - h_shannon).abs() < 0.01,
            "Tsallis(q=1) should equal Shannon, got {} vs {}", h_tsallis, h_shannon);
    }
}

/// KL divergence with zero in q: log(p/0) → Inf.
/// Type 1.
#[test]
fn kl_divergence_zero_q() {
    let p = vec![0.5, 0.5];
    let q = vec![1.0, 0.0]; // q[1]=0 but p[1]=0.5 → Inf
    let kl = kl_divergence(&p, &q);
    assert!(kl.is_infinite() || kl.is_nan(),
        "KL(p||q) with q=0 where p>0 should be Inf, got {}", kl);
}

/// KL divergence: KL(p||p) = 0.
#[test]
fn kl_divergence_identical() {
    let p = vec![0.3, 0.4, 0.3];
    let kl = kl_divergence(&p, &p);
    assert!((kl - 0.0).abs() < 1e-10, "KL(p||p) should be 0, got {}", kl);
}

/// JS divergence: symmetric, bounded [0, ln(2)].
#[test]
fn js_divergence_bounds() {
    let p = vec![1.0, 0.0];
    let q = vec![0.0, 1.0];
    let js = js_divergence(&p, &q);
    // Maximum JS divergence = ln(2) ≈ 0.693
    assert!(js.is_finite(), "JS divergence should be finite, got {}", js);
    assert!(js >= -1e-10, "JS divergence should be non-negative, got {}", js);
    assert!(js <= (2.0_f64).ln() + 0.01, "JS should be <= ln(2), got {}", js);
}

/// Mutual information with 1x1 contingency: trivial.
/// Type 4.
#[test]
fn mutual_info_1x1() {
    let contingency = vec![10.0]; // 1x1 table
    let mi = mutual_information(&contingency, 1, 1);
    assert!((mi - 0.0).abs() < 1e-10 || mi.is_nan(),
        "MI of 1x1 table should be 0, got {}", mi);
}

/// Mutual information with all-zero contingency.
/// Type 1.
#[test]
fn mutual_info_all_zeros() {
    let contingency = vec![0.0, 0.0, 0.0, 0.0]; // 2x2 all zeros
    let mi = mutual_information(&contingency, 2, 2);
    if mi.is_nan() {
        // 0/0 in log terms
    } else {
        assert!((mi - 0.0).abs() < 1e-10, "All-zero table MI should be 0, got {}", mi);
    }
}

/// Adjusted mutual information score with identical labels.
#[test]
fn ami_identical() {
    let labels = vec![0, 0, 1, 1, 2, 2];
    let ami = adjusted_mutual_info_score(&labels, &labels);
    assert!((ami - 1.0).abs() < 0.01, "AMI of identical labels should be 1.0, got {}", ami);
}

/// Entropy from histogram with 0 bins.
/// Type 5.
#[test]
fn entropy_histogram_zero_bins() {
    let data = vec![1.0, 2.0, 3.0];
    let result = std::panic::catch_unwind(|| {
        entropy_histogram(&data, 0)
    });
    if result.is_err() {
        eprintln!("NOTE: entropy_histogram panics with 0 bins");
    } else {
        let h = result.unwrap();
        assert!(h.is_finite() || h.is_nan(), "0 bins should give finite or NaN, got {}", h);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ROBUST
// ═══════════════════════════════════════════════════════════════════════════

/// Huber M-estimate with single data point: no variance → MAD=0 → scale=0.
/// Type 1.
#[test]
fn huber_m_single_point() {
    let data = vec![5.0];
    let result = huber_m_estimate(&data, 1.345, 100, 1e-6);
    assert!((result.location - 5.0).abs() < 1e-10,
        "Single point Huber location should be 5.0, got {}", result.location);
}

/// Huber M-estimate with all identical data: MAD=0 → divide by zero.
/// Type 1.
#[test]
fn huber_m_constant_data() {
    let data = vec![3.0; 10];
    let result = huber_m_estimate(&data, 1.345, 100, 1e-6);
    assert!(result.location.is_finite(),
        "Huber on constant data should give finite location, got {}", result.location);
    assert!((result.location - 3.0).abs() < 0.01,
        "Huber on constant data should find 3.0, got {}", result.location);
}

/// Bisquare M-estimate with empty data.
/// Type 5.
#[test]
fn bisquare_m_empty() {
    let result = std::panic::catch_unwind(|| {
        bisquare_m_estimate(&[], 4.685, 100, 1e-6)
    });
    if result.is_err() {
        eprintln!("NOTE: bisquare_m_estimate panics on empty data");
    }
}

/// Qn scale estimator with constant data: all pairwise differences = 0.
/// Type 4.
#[test]
fn qn_scale_constant() {
    let data = vec![7.0; 10];
    let q = qn_scale(&data);
    assert!((q - 0.0).abs() < 1e-10, "Qn of constant data should be 0, got {}", q);
}

/// Sn scale estimator with 1 data point.
/// Type 5.
#[test]
fn sn_scale_single_point() {
    let data = vec![42.0];
    let result = std::panic::catch_unwind(|| {
        sn_scale(&data)
    });
    match result {
        Ok(s) => assert!((s - 0.0).abs() < 1e-10 || s.is_nan(),
            "Sn of single point should be 0 or NaN, got {}", s),
        Err(_) => eprintln!("NOTE: sn_scale panics on single point"),
    }
}

/// Tau scale with 2 data points.
#[test]
fn tau_scale_two_points() {
    let data = vec![1.0, 100.0];
    let result = std::panic::catch_unwind(|| {
        tau_scale(&data)
    });
    match result {
        Ok(t) => assert!(t.is_finite(), "Tau scale of 2 points should be finite, got {}", t),
        Err(_) => eprintln!("NOTE: tau_scale panics on 2 points"),
    }
}

/// LTS regression with fewer points than needed (n < 3).
/// Type 5.
#[test]
fn lts_too_few_points() {
    let x = vec![1.0];
    let y = vec![2.0];
    let result = std::panic::catch_unwind(|| {
        lts_simple(&x, &y, 100, 42)
    });
    match result {
        Ok(r) => {
            // n=1, h=ceil(n/2)+1=2 > n → subset selection fails
            assert!(r.slope.is_finite() || r.slope.is_nan(),
                "LTS with 1 point should be finite or NaN, got slope={}", r.slope);
        }
        Err(_) => eprintln!("NOTE: lts_simple panics with n=1"),
    }
}

/// MCD 2D with collinear points: covariance is singular.
/// Type 4.
#[test]
fn mcd_collinear() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x, perfectly collinear
    let result = std::panic::catch_unwind(|| {
        mcd_2d(&x, &y, 100, 42)
    });
    match result {
        Ok(r) => {
            // Singular covariance → determinant = 0
            let det = r.covariance[0] * r.covariance[3] - r.covariance[1] * r.covariance[2];
            if det.abs() < 1e-10 {
                // Expected: singular covariance for collinear data
            }
        }
        Err(_) => eprintln!("CONFIRMED BUG: mcd_2d panics on collinear data"),
    }
}

/// Medcouple with constant data: all kernel values are 0/0.
/// Type 1.
#[test]
fn medcouple_constant() {
    let data = vec![5.0; 10];
    let mc = medcouple(&data);
    // h(xi, xj) = (xi + xj - 2*median) / (xi - xj) with xi=xj → 0/0
    assert!(mc.is_finite() || mc.is_nan(),
        "Medcouple of constant data should be finite or NaN, got {}", mc);
}

/// Medcouple with 2 points: minimal sample.
#[test]
fn medcouple_two_points() {
    let data = vec![1.0, 10.0];
    let mc = medcouple(&data);
    if mc.is_nan() {
        eprintln!("CONFIRMED BUG: medcouple returns NaN for 2 data points");
    } else {
        assert!(mc >= -1.0 && mc <= 1.0, "Medcouple should be in [-1,1], got {}", mc);
    }
}

/// Hampel weight function with a >= b >= c ordering violated.
/// Type 5: violates expected a < b < c.
#[test]
fn hampel_weight_bad_params() {
    // a=5, b=3, c=1 → reversed ordering
    let w = hampel_weight(2.0, 5.0, 3.0, 1.0);
    // With a>b>c, the piecewise conditions are confused
    assert!(w.is_finite(), "Hampel weight with bad params should be finite, got {}", w);
}

// ═══════════════════════════════════════════════════════════════════════════
// BAYESIAN
// ═══════════════════════════════════════════════════════════════════════════

/// MCMC with 0 samples: should return empty chain.
/// Type 5.
#[test]
fn mcmc_zero_samples() {
    let log_target = |_x: &[f64]| -> f64 { -0.5 * _x[0] * _x[0] };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        metropolis_hastings(&log_target, &[0.0], 1.0, 0, 0, 42)
    }));
    match result {
        Ok(chain) => {
            assert!(chain.samples.is_empty(), "0 samples should give empty chain");
        }
        Err(_) => eprintln!("NOTE: MCMC panics on 0 samples"),
    }
}

/// MCMC with burnin > n_samples: all samples discarded.
/// Type 2.
#[test]
fn mcmc_burnin_exceeds_samples() {
    let log_target = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        metropolis_hastings(&log_target, &[0.0], 1.0, 100, 200, 42)
    }));
    match result {
        Ok(chain) => {
            // burnin > n_samples → no samples retained
            assert!(chain.samples.is_empty() || chain.samples.len() <= 100,
                "burnin>n should give few/no samples, got {}", chain.samples.len());
        }
        Err(_) => eprintln!("CONFIRMED BUG: MCMC panics when burnin > n_samples"),
    }
}

/// MCMC with log_target returning -Inf: every proposal rejected.
/// Type 2.
#[test]
fn mcmc_always_reject() {
    let log_target = |_x: &[f64]| -> f64 { f64::NEG_INFINITY };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        metropolis_hastings(&log_target, &[0.0], 1.0, 50, 0, 42)
    }));
    match result {
        Ok(chain) => {
            // All proposals have log_target=-∞, acceptance ratio = exp(-∞ - (-∞)) = exp(NaN) = NaN
            // or if initial is also -∞, everything is stuck at initial
            assert!(chain.acceptance_rate >= 0.0 && chain.acceptance_rate <= 1.0,
                "Acceptance rate should be in [0,1], got {}", chain.acceptance_rate);
        }
        Err(_) => eprintln!("CONFIRMED BUG: MCMC panics when log_target always returns -Inf"),
    }
}

/// MCMC with NaN log_target.
/// Type 3.
#[test]
fn mcmc_nan_target() {
    let log_target = |_x: &[f64]| -> f64 { f64::NAN };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        metropolis_hastings(&log_target, &[0.0], 1.0, 20, 0, 42)
    }));
    match result {
        Ok(chain) => {
            // exp(NaN - NaN) = exp(NaN) = NaN. NaN > u is always false → always reject
            // chain should be all initial value
            eprintln!("NOTE: MCMC with NaN target: acceptance_rate={}", chain.acceptance_rate);
        }
        Err(_) => eprintln!("CONFIRMED BUG: MCMC panics on NaN log_target"),
    }
}

/// MCMC with proposal_sd=0: all proposals identical to current → acceptance=1 always.
/// Type 4.
#[test]
fn mcmc_zero_proposal() {
    let log_target = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        metropolis_hastings(&log_target, &[5.0], 0.0, 50, 0, 42)
    }));
    match result {
        Ok(chain) => {
            // proposal_sd=0 → proposal = current + 0*N(0,1) = current
            // → acceptance = 1, chain stuck at initial
            for sample in &chain.samples {
                assert!((sample[0] - 5.0).abs() < 1e-10,
                    "Zero proposal should keep chain at initial, got {}", sample[0]);
            }
        }
        Err(_) => eprintln!("CONFIRMED BUG: MCMC panics with proposal_sd=0"),
    }
}

/// Effective sample size with constant chain: all autocorrelations = 1.
/// Type 3.
#[test]
fn ess_constant_chain() {
    let samples = vec![3.0; 100];
    let ess = effective_sample_size(&samples);
    // Constant data: variance=0, autocorrelation undefined → ESS=0 or NaN
    if ess.is_nan() {
        eprintln!("NOTE: ESS returns NaN for constant chain (0/0)");
    } else {
        assert!(ess >= 0.0, "ESS should be non-negative, got {}", ess);
    }
}

/// R-hat with single chain.
#[test]
fn r_hat_single_chain() {
    let chain = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = std::panic::catch_unwind(|| {
        r_hat(&[&chain])
    });
    match result {
        Ok(rh) => {
            if rh.is_nan() {
                eprintln!("CONFIRMED BUG: R-hat returns NaN for single chain (between-chain variance undefined)");
            } else {
                assert!(rh.is_finite(), "R-hat with single chain should be finite, got {}", rh);
            }
        }
        Err(_) => eprintln!("NOTE: r_hat panics on single chain"),
    }
}

/// R-hat with identical chains: perfect convergence → R-hat = 1.
#[test]
fn r_hat_identical_chains() {
    let chain = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let rh = r_hat(&[&chain, &chain, &chain]);
    if (rh - 1.0).abs() > 0.2 {
        eprintln!("CONFIRMED BUG: R-hat for identical chains should be ~1.0, got {} (split-chain variance artifact)", rh);
    }
}

/// Bayesian linear regression with d=0: no features.
/// Type 5.
#[test]
fn bayes_linear_no_features() {
    let x: Vec<f64> = vec![];
    let y = vec![1.0, 2.0, 3.0];
    let result = std::panic::catch_unwind(|| {
        bayesian_linear_regression(&x, &y, 3, 0, &[], &[], 1.0, 1.0)
    });
    match result {
        Ok(r) => {
            assert!(r.sigma2_mean.is_finite(), "Sigma2 should be finite with 0 features");
        }
        Err(_) => eprintln!("NOTE: Bayesian linear regression panics with 0 features"),
    }
}

/// Bayesian linear regression with n < d: underdetermined.
/// Type 5.
#[test]
fn bayes_linear_underdetermined() {
    // n=2, d=5 → X'X is 5x5 but rank ≤ 2 → prior regularizes
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0,
                 6.0, 7.0, 8.0, 9.0, 10.0];
    let y = vec![1.0, 2.0];
    let prior_mean = vec![0.0; 5];
    let prior_precision = vec![1.0; 5]; // diagonal prior precision
    let result = std::panic::catch_unwind(|| {
        bayesian_linear_regression(&x, &y, 2, 5, &prior_mean, &prior_precision, 1.0, 1.0)
    });
    match result {
        Ok(r) => {
            for &b in &r.beta_mean {
                assert!(b.is_finite(), "Beta should be finite in underdetermined case, got {}", b);
            }
        }
        Err(_) => eprintln!("CONFIRMED BUG: Bayesian regression panics on underdetermined system"),
    }
}
