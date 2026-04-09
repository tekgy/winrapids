//! Adversarial tests for fintek gap implementations wave 4: VPIN + NVG/HVG.

use tambear::volatility::{vpin_bvc, nvg_degree, hvg_degree, nvg_mean_degree, hvg_mean_degree};

// ═══════════════════════════════════════════════════════════════════════════
// VPIN
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn vpin_empty_returns_empty() {
    let r = vpin_bvc(&[], &[], 100.0, 10);
    assert_eq!(r.n_buckets, 0);
    assert!(r.vpin.is_empty());
}

#[test]
fn vpin_too_short_returns_empty() {
    let r = vpin_bvc(&[100.0], &[50.0], 100.0, 10);
    assert_eq!(r.n_buckets, 0);
}

#[test]
fn vpin_mismatched_lengths_returns_empty() {
    let r = vpin_bvc(&[100.0, 101.0], &[50.0], 100.0, 10);
    assert!(r.vpin.is_empty());
}

#[test]
fn vpin_zero_bucket_volume_returns_empty() {
    let r = vpin_bvc(&[100.0, 101.0], &[50.0, 50.0], 0.0, 10);
    assert!(r.vpin.is_empty());
}

#[test]
fn vpin_in_unit_interval() {
    // VPIN = |V_buy - V_sell| / V_bucket, averaged. Must be in [0, 1].
    let n = 1000;
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let mut prices = vec![100.0_f64; n];
    let volumes = vec![10.0_f64; n];
    for i in 1..n {
        prices[i] = prices[i - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.5);
    }
    let r = vpin_bvc(&prices, &volumes, 200.0, 10);
    assert!(r.n_buckets > 10, "should form enough buckets, got {}", r.n_buckets);
    for &v in &r.vpin {
        assert!(v >= 0.0 && v <= 1.0, "VPIN must be in [0,1], got {}", v);
    }
}

#[test]
fn vpin_trending_higher_than_random() {
    // A strong trend should produce higher VPIN than mean-reverting noise
    // because BVC classifies most volume as buy (or sell) in a trend.
    let n = 2000;
    let vol = vec![10.0_f64; n];

    // Trending: constant upward drift
    let trend: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();
    let r_trend = vpin_bvc(&trend, &vol, 200.0, 20);

    // Random walk: zero drift
    let mut rng = tambear::rng::Xoshiro256::new(99);
    let mut random = vec![100.0_f64; n];
    for i in 1..n {
        random[i] = random[i - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.5);
    }
    let r_rand = vpin_bvc(&random, &vol, 200.0, 20);

    if !r_trend.vpin.is_empty() && !r_rand.vpin.is_empty() {
        let mean_trend: f64 = r_trend.vpin.iter().sum::<f64>() / r_trend.vpin.len() as f64;
        let mean_rand: f64 = r_rand.vpin.iter().sum::<f64>() / r_rand.vpin.len() as f64;
        assert!(mean_trend > mean_rand,
            "trending VPIN {} should exceed random VPIN {}", mean_trend, mean_rand);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NVG
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nvg_empty_returns_empty() {
    assert!(nvg_degree(&[]).is_empty());
}

#[test]
fn nvg_single_element() {
    let d = nvg_degree(&[5.0]);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0], 0); // no edges possible
}

#[test]
fn nvg_two_elements_connected() {
    let d = nvg_degree(&[1.0, 2.0]);
    assert_eq!(d, vec![1, 1]); // single edge between them
}

#[test]
fn nvg_monotone_increasing_all_adjacent() {
    // Strictly increasing: each pair (a, a+1) has nothing between, so connected.
    // But (a, a+2) blocked by the intermediate point on the line? For strictly
    // increasing: y_c < y_a + (y_b-y_a)*(c-a)/(b-a). Linear interp equals y_c
    // when data is exactly linear. Strictly increasing with constant step: data[c]
    // equals the interpolation → NOT strictly less than → NOT visible.
    // So for [1,2,3,4], only adjacent pairs are visible.
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let d = nvg_degree(&data);
    // Endpoints have degree 1, interior have degree 2
    assert_eq!(d[0], 1);
    assert_eq!(d[1], 2);
    assert_eq!(d[2], 2);
    assert_eq!(d[3], 1);
}

#[test]
fn nvg_peak_sees_all() {
    // A single tall peak in the middle should see all other points.
    let data = vec![0.0, 0.0, 10.0, 0.0, 0.0];
    let d = nvg_degree(&data);
    // The peak at index 2 sees all 4 others (nothing blocks it).
    assert_eq!(d[2], 4, "peak should see all others, got {}", d[2]);
}

#[test]
fn nvg_random_mean_degree_near_4() {
    // Lacasa et al. 2008: for random iid, E[mean_degree] → 4.
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let data: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
    let md = nvg_mean_degree(&data);
    assert!(md > 2.0 && md < 8.0,
        "NVG mean degree for random should be near 4, got {}", md);
}

// ═══════════════════════════════════════════════════════════════════════════
// HVG
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hvg_empty_returns_empty() {
    assert!(hvg_degree(&[]).is_empty());
}

#[test]
fn hvg_monotone_all_adjacent() {
    // Strictly increasing: (a, b) visible iff all c in (a,b) have data[c] < min(data[a], data[b]).
    // For increasing, min(data[a], data[b]) = data[a]. data[a+1] > data[a] → blocks.
    // So only adjacent pairs are connected.
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let d = hvg_degree(&data);
    assert_eq!(d, vec![1, 2, 2, 1]);
}

#[test]
fn hvg_constant_only_adjacent() {
    // Constant: data[c] = data[a] = min(data[a], data[b]). NOT strictly less → NOT visible.
    // Only adjacent pairs (no intermediary) are visible.
    let data = vec![5.0; 5];
    let d = hvg_degree(&data);
    assert_eq!(d, vec![1, 2, 2, 2, 1]);
}

#[test]
fn hvg_valley_short_range() {
    // Deep valley: [5, 1, 5]. The valley at 1 sees both 5s (adjacent).
    // The two 5s: is 1 < min(5,5) = 5? Yes. So (0,2) are visible through the valley.
    let data = vec![5.0, 1.0, 5.0];
    let d = hvg_degree(&data);
    // 0-1: adjacent, visible. 1-2: adjacent, visible. 0-2: 1 < min(5,5), visible.
    assert_eq!(d, vec![2, 2, 2]);
}

#[test]
fn hvg_random_mean_degree_near_4() {
    // For random iid, E[HVG mean degree] = 4 exactly.
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let data: Vec<f64> = (0..300).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
    let md = hvg_mean_degree(&data);
    assert!(md > 2.5 && md < 6.0,
        "HVG mean degree for random should be near 4, got {}", md);
}

#[test]
fn hvg_degree_leq_nvg_degree() {
    // HVG is a subgraph of NVG — every HVG edge is also an NVG edge.
    // So HVG degree <= NVG degree for every node.
    let mut rng = tambear::rng::Xoshiro256::new(7);
    let data: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
    let nvg = nvg_degree(&data);
    let hvg = hvg_degree(&data);
    for i in 0..data.len() {
        assert!(hvg[i] <= nvg[i],
            "HVG degree {} > NVG degree {} at index {}", hvg[i], nvg[i], i);
    }
}
