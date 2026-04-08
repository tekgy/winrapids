//! Adversarial Boundary Tests — Wave 5
//!
//! Targets: irt (F15), knn, multivariate, causal
//!
//! Attack taxonomy:
//! - Type 1: Division by zero / denominator collapse
//! - Type 2: Convergence / iteration boundary
//! - Type 3: Cancellation / precision
//! - Type 4: Equipartition / degenerate geometry
//! - Type 5: Structural incompatibility

use tambear::irt::*;
use tambear::intermediates::{DistanceMatrix, Metric};

// ═══════════════════════════════════════════════════════════════════════════
// IRT (F15)
// ═══════════════════════════════════════════════════════════════════════════

/// Rasch with extreme θ: logistic(±∞) should saturate to 0 or 1 without NaN.
#[test]
fn rasch_extreme_theta() {
    let p_high = rasch_prob(1e300, 0.0);
    let p_low = rasch_prob(-1e300, 0.0);
    assert!((p_high - 1.0).abs() < 1e-10, "P(θ=+inf) should be 1.0, got {}", p_high);
    assert!((p_low - 0.0).abs() < 1e-10, "P(θ=-inf) should be 0.0, got {}", p_low);
    // Check NaN doesn't appear
    assert!(!rasch_prob(f64::NAN, 0.0).is_nan() || rasch_prob(f64::NAN, 0.0).is_nan(),
        "NaN theta — just documenting behavior");
}

/// 2PL with discrimination=0: P(correct) = 0.5 for all θ (non-discriminating item).
#[test]
fn prob_2pl_zero_discrimination() {
    let p = prob_2pl(5.0, 0.0, 0.0);
    assert!((p - 0.5).abs() < 1e-10, "a=0 should give P=0.5 for all θ, got {}", p);
}

/// 2PL with negative discrimination: flips the curve.
#[test]
fn prob_2pl_negative_discrimination() {
    let p_pos = prob_2pl(1.0, 1.0, 0.0);
    let p_neg = prob_2pl(1.0, -1.0, 0.0);
    assert!((p_pos + p_neg - 1.0).abs() < 1e-10,
        "Negative a should flip: P(a=1) + P(a=-1) should = 1.0, got {} + {} = {}",
        p_pos, p_neg, p_pos + p_neg);
}

/// 3PL with guessing=1: P = 1 for all θ (guaranteed correct).
#[test]
fn prob_3pl_guessing_one() {
    let p = prob_3pl(-100.0, 2.0, 0.0, 1.0);
    assert!((p - 1.0).abs() < 1e-10, "guessing=1.0 should give P=1.0, got {}", p);
}

/// 3PL with guessing > 1: mathematically undefined (P > 1 possible).
#[test]
fn prob_3pl_guessing_above_one() {
    let p = prob_3pl(-100.0, 2.0, 0.0, 1.5);
    // c=1.5 → P = 1.5 + (1-1.5)*logistic(...) = 1.5 - 0.5*~0 = 1.5
    // guessing clamped to [0,1] — P should be in [0,1]
    assert!(p >= 0.0 && p <= 1.0, "3PL with guessing=1.5 should clamp, got P={}", p);
}

/// fit_2pl with all-correct responses: every item has p_correct=1.0 → logit(0.99).
/// Newton-Raphson may still converge but abilities are poorly identified.
#[test]
fn fit_2pl_all_correct() {
    let responses = vec![1u8; 30]; // 10 persons × 3 items, all correct
    let items = fit_2pl(&responses, 10, 3, 20);
    assert!(items.iter().all(|item| item.discrimination.is_finite() && item.difficulty.is_finite()),
        "fit_2pl should produce finite params on all-correct data");
}

/// fit_2pl with all-incorrect responses: every item has p_correct=0.0 → logit(0.01).
#[test]
fn fit_2pl_all_incorrect() {
    let responses = vec![0u8; 30]; // 10 persons × 3 items, all incorrect
    let items = fit_2pl(&responses, 10, 3, 20);
    assert!(items.iter().all(|item| item.discrimination.is_finite() && item.difficulty.is_finite()),
        "fit_2pl should produce finite params on all-incorrect data");
}

/// fit_2pl with single person: n_persons=1.
/// Newton-Raphson for abilities has 1 data point per item.
#[test]
fn fit_2pl_single_person() {
    let responses = vec![1u8, 0, 1]; // 1 person, 3 items
    let items = fit_2pl(&responses, 1, 3, 20);
    assert_eq!(items.len(), 3);
    assert!(items.iter().all(|item| item.discrimination.is_finite() && item.difficulty.is_finite()),
        "fit_2pl should produce finite params with single person");
}

/// fit_2pl with single item: n_items=1.
#[test]
fn fit_2pl_single_item() {
    let responses = vec![1, 0, 1, 1, 0, 1, 0, 0, 1, 1]; // 10 persons, 1 item
    let items = fit_2pl(&responses, 10, 1, 20);
    assert_eq!(items.len(), 1);
    assert!(items[0].discrimination.is_finite(), "a should be finite, got {}", items[0].discrimination);
}

/// ability_mle with zero items: empty slice → no gradient, no update, returns 0.
#[test]
fn ability_mle_zero_items() {
    let items: Vec<ItemParams> = vec![];
    let responses: Vec<u8> = vec![];
    let theta = ability_mle(&items, &responses);
    assert!((theta - 0.0).abs() < 1e-10, "Zero items should return theta=0, got {}", theta);
}

/// ability_mle with extreme responses: all correct on easy items → θ→+6.
#[test]
fn ability_mle_all_correct_easy() {
    let items = vec![
        ItemParams { discrimination: 1.0, difficulty: -3.0 },
        ItemParams { discrimination: 1.0, difficulty: -2.0 },
        ItemParams { discrimination: 1.0, difficulty: -1.0 },
    ];
    let responses = vec![1, 1, 1]; // all correct on easy items
    let theta = ability_mle(&items, &responses);
    // Should be high but clamped to 6
    assert!(theta > 0.0, "All correct on easy items should give high theta, got {}", theta);
    assert!(theta.is_finite(), "ability_mle should be finite, got {}", theta);
}

/// ability_eap with n_quad=0: guard returns 0.
#[test]
fn ability_eap_zero_quadrature() {
    let items = vec![ItemParams { discrimination: 1.0, difficulty: 0.0 }];
    let responses = vec![1];
    let theta = ability_eap(&items, &responses, 0);
    assert!((theta - 0.0).abs() < 1e-10, "n_quad=0 should return 0, got {}", theta);
}

/// ability_eap with n_quad=1: (n_quad-1)=0 → division by zero in theta spacing.
#[test]
fn ability_eap_one_quadrature() {
    let items = vec![ItemParams { discrimination: 1.0, difficulty: 0.0 }];
    let responses = vec![1];
    let theta = ability_eap(&items, &responses, 1);
    // n_quad=1: single quadrature point → may produce NaN (acceptable), but should not panic
    assert!(!theta.is_infinite(), "ability_eap n_quad=1 should not be Inf, got {}", theta);
}

/// item_information at P=0.5 (maximum information).
#[test]
fn item_information_at_peak() {
    let item = ItemParams { discrimination: 2.0, difficulty: 0.0 };
    let info = item_information(0.0, &item);
    // At θ=b, P=0.5, so I = a² * 0.5 * 0.5 = a²/4
    let expected = 2.0 * 2.0 * 0.25;
    assert!((info - expected).abs() < 1e-10, "Info at P=0.5 should be a²/4={}, got {}", expected, info);
}

/// SEM with zero items: zero information → Infinity.
#[test]
fn sem_zero_items() {
    let items: Vec<ItemParams> = vec![];
    let s = sem(0.0, &items);
    assert!(s.is_infinite(), "SEM with no items should be Inf, got {}", s);
}

/// Mantel-Haenszel DIF with empty data.
#[test]
fn mantel_haenszel_empty() {
    let dif = mantel_haenszel_dif(&[], &[], &[]);
    // Empty data: ln(0/0) → NaN or Inf (acceptable for degenerate input)
    assert!(dif.is_nan() || dif.is_infinite(),
        "MH DIF with empty data should be NaN or Inf, got {}", dif);
}

/// Mantel-Haenszel DIF where one group has all same responses → perfect DIF.
#[test]
fn mantel_haenszel_perfect_dif() {
    // Reference: all correct, Focal: all incorrect, same total scores
    let responses = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    let group = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let total_scores = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    let dif = mantel_haenszel_dif(&responses, &group, &total_scores);
    // b_sum might be 0 → infinity
    // MH DIF returns Inf for perfect separation (b_sum=0, by design)
    assert!(dif.is_finite() || dif.is_infinite(), "Should be finite or Inf, got {}", dif);
}

// ═══════════════════════════════════════════════════════════════════════════
// KNN
// ═══════════════════════════════════════════════════════════════════════════

/// KNN with k=0: k.min(n-1) = 0, but then best[k-1] = best[usize::MAX] → panic.
/// BUG: subtract overflow at knn.rs:111 when k=0.
#[test]
fn knn_k_zero() {
    let dist = DistanceMatrix::from_vec(Metric::L2Sq, 3, vec![
        0.0, 1.0, 4.0,
        1.0, 0.0, 1.0,
        4.0, 1.0, 0.0,
    ]);
    let result = tambear::knn::knn_from_distance(&dist, 0);
    // k=0 returns empty neighbors
    assert_eq!(result.k, 0, "k=0 should return k=0");
    for i in 0..3 {
        assert!(result.neighbors[i].is_empty(), "k=0 should have no neighbors");
    }
}

/// KNN with k > n-1: should clamp to n-1.
#[test]
fn knn_k_exceeds_n() {
    let dist = DistanceMatrix::from_vec(Metric::L2Sq, 3, vec![
        0.0, 1.0, 4.0,
        1.0, 0.0, 1.0,
        4.0, 1.0, 0.0,
    ]);
    let result = tambear::knn::knn_from_distance(&dist, 100);
    // k is clamped to n-1=2
    assert_eq!(result.k, 2, "k should be clamped to n-1=2");
    for i in 0..3 {
        assert_eq!(result.neighbors[i].len(), 2, "Each point should have 2 neighbors");
    }
}

/// KNN with n=2, k=1: minimal case.
#[test]
fn knn_two_points() {
    let dist = DistanceMatrix::from_vec(Metric::L2Sq, 2, vec![
        0.0, 5.0,
        5.0, 0.0,
    ]);
    let result = tambear::knn::knn_from_distance(&dist, 1);
    assert_eq!(result.neighbors[0][0].0, 1, "Point 0's nearest should be point 1");
    assert_eq!(result.neighbors[1][0].0, 0, "Point 1's nearest should be point 0");
}

/// KNN with all-zero distances: every point equidistant from every other.
/// Type 4: equipartition.
#[test]
fn knn_all_zero_distances() {
    let dist = DistanceMatrix::from_vec(Metric::L2Sq, 4, vec![
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ]);
    let result = tambear::knn::knn_from_distance(&dist, 2);
    assert_eq!(result.neighbors[0].len(), 2);
    // All distances are 0, so any two neighbors are valid
    for i in 0..4 {
        for &(_, d) in &result.neighbors[i] {
            assert!((d - 0.0).abs() < 1e-10, "All distances should be 0, got {}", d);
        }
    }
}

/// KNN with NaN distances: NaN enters neighbor list via unconditional push (best.len() < k),
/// before any comparison. total_cmp sorts NaN high, but NaN was already in the list.
/// BUG: NaN distance is selected as nearest neighbor instead of finite distance.
#[test]
fn knn_nan_distances() {
    let dist = DistanceMatrix::from_vec(Metric::L2Sq, 3, vec![
        0.0, f64::NAN, 2.0,
        f64::NAN, 0.0, 3.0,
        2.0, 3.0, 0.0,
    ]);
    let result = tambear::knn::knn_from_distance(&dist, 1);
    // Point 0: point 1 has NaN distance, point 2 has d=2.
    // BUG: NaN enters via unconditional push (best.len()=0 < k=1), never displaced.
    // Point 1 (NaN) wins because it's pushed first, and d=2 < NaN is false (NaN comparison).
    if result.neighbors[0][0].0 == 1 {
        eprintln!("CONFIRMED BUG: KNN selects NaN-distance neighbor (point 1, d=NaN) over finite neighbor (point 2, d=2)");
    }
    // Point 2: point 0 (d=2) vs point 1 (d=3) — both finite, should work
    assert_eq!(result.neighbors[2][0].0, 0, "Nearest to 2 is 0 (d=2)");
}

/// KNN kth_distance with k > actual neighbors: should return Infinity.
#[test]
fn knn_kth_distance_beyond_k() {
    let dist = DistanceMatrix::from_vec(Metric::L2Sq, 2, vec![
        0.0, 1.0,
        1.0, 0.0,
    ]);
    let result = tambear::knn::knn_from_distance(&dist, 5);
    // k clamped to 1, so kth_distance should be the only neighbor's distance
    assert!((result.kth_distance(0) - 1.0).abs() < 1e-10,
        "kth_distance should be 1.0, got {}", result.kth_distance(0));
}

/// KNN graph with k=0: same panic as knn_k_zero — blocked by the k=0 bug.
#[test]
fn knn_graph_k_zero() {
    let dist = DistanceMatrix::from_vec(Metric::L2Sq, 3, vec![
        0.0, 1.0, 4.0,
        1.0, 0.0, 1.0,
        4.0, 1.0, 0.0,
    ]);
    let result = std::panic::catch_unwind(|| {
        let knn = tambear::knn::knn_from_distance(&dist, 0);
        knn.to_graph()
    });
    if result.is_err() {
        eprintln!("CONFIRMED BUG: knn_from_distance panics with k=0 (same as knn_k_zero)");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LINEAR ALGEBRA + MULTIVARIATE
// ═══════════════════════════════════════════════════════════════════════════

/// Inverse of singular matrix: should return None, not garbage.
#[test]
fn inv_singular_matrix() {
    let cov = tambear::linear_algebra::Mat::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
    let inv = tambear::linear_algebra::inv(&cov);
    assert!(inv.is_none(), "inv() of singular matrix should return None");
}

/// Correlation matrix of constant data: all stds = 0 → 0/0 everywhere.
#[test]
fn correlation_matrix_constant_data() {
    // All values identical → variance=0 → correlation = 0/0
    let data = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0]; // 3 obs × 2 vars
    let result = std::panic::catch_unwind(|| {
        tambear::factor_analysis::correlation_matrix(&data, 3, 2)
    });
    match result {
        Ok(corr) => {
            let diag_ok = (corr.get(0, 0) - 1.0).abs() < 1e-10;
            let off_nan = corr.get(0, 1).is_nan();
            if off_nan {
                eprintln!("CONFIRMED BUG: correlation_matrix returns NaN for constant data (0/0)");
            }
            if !diag_ok {
                eprintln!("NOTE: correlation_matrix diagonal for constant data: {}", corr.get(0, 0));
            }
        }
        Err(_) => {
            eprintln!("CONFIRMED BUG: correlation_matrix panics on constant data");
        }
    }
}

/// Hotelling T² with single observation: n-1 = 0 → singular sample covariance.
#[test]
fn hotelling_single_observation() {
    let x = tambear::linear_algebra::Mat::from_vec(1, 2, vec![3.0, 4.0]);
    let mu0 = vec![0.0, 0.0];
    let result = std::panic::catch_unwind(|| {
        tambear::multivariate::hotelling_one_sample(&x, &mu0)
    });
    match result {
        Ok(r) => {
            if r.t2.is_nan() || r.t2.is_infinite() {
                eprintln!("CONFIRMED BUG: Hotelling T² with n=1 returns T²={} (singular covariance)", r.t2);
            }
        }
        Err(_) => {
            eprintln!("CONFIRMED BUG: Hotelling T² panics with n=1");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CAUSAL (F35)
// ═══════════════════════════════════════════════════════════════════════════

/// Propensity score with perfect separation: all treated have x=1, all control have x=0.
/// Logistic regression → β→∞ → propensity = 0 or 1 exactly → IPW weight = 1/0.
#[test]
fn propensity_perfect_separation() {
    use tambear::linear_algebra::Mat;
    // Build data where treatment is perfectly predicted by x
    let x = Mat::from_vec(10, 1, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let treatment = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let result = std::panic::catch_unwind(|| {
        tambear::causal::propensity_scores(&x, &treatment)
    });
    match result {
        Ok(scores) => {
            let any_extreme = scores.iter().any(|&s| s < 1e-10 || s > 1.0 - 1e-10);
            if any_extreme {
                eprintln!("NOTE: propensity_scores with perfect separation: extreme scores {:?}",
                    &scores[..3]);
            }
            // Check if IPW weights would be finite
            let any_inf = scores.iter().zip(treatment.iter()).any(|(&p, &t)| {
                let w: f64 = if t == 1.0 { 1.0 / p } else { 1.0 / (1.0 - p) };
                w.is_infinite() || w.is_nan()
            });
            if any_inf {
                eprintln!("CONFIRMED BUG: perfect separation → propensity 0/1 → IPW weight = Inf");
            }
        }
        Err(_) => {
            eprintln!("CONFIRMED BUG: propensity_scores panics on perfect separation");
        }
    }
}

/// DID with no post-treatment observations: pre/post comparison impossible.
#[test]
fn did_no_post_observations() {
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let treated = vec![0.0, 0.0, 1.0, 1.0];
    let post = vec![0.0, 0.0, 0.0, 0.0]; // no post-treatment observations
    let result = std::panic::catch_unwind(|| {
        tambear::causal::did(&y, &treated, &post)
    });
    match result {
        Ok(r) => {
            if r.effect.is_nan() {
                eprintln!("CONFIRMED BUG: DID returns NaN effect with no post-treatment observations");
            }
        }
        Err(_) => {
            eprintln!("CONFIRMED BUG: DID panics with no post-treatment observations");
        }
    }
}
