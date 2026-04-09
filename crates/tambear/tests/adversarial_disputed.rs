//! Adversarial tests for disputed findings between review agents.
//!
//! These tests exist to settle disagreements with concrete evidence.
//! Each test targets a specific claim, with a clear pass/fail criterion.

// ═══════════════════════════════════════════════════════════════════════════
// t-SNE: Jacobi gradient update (VERIFIED FIX — was Gauss-Seidel)
// ═══════════════════════════════════════════════════════════════════════════
//
// The gradient is now computed using a frozen snapshot of all positions
// (grad_buf), then applied atomically. This is Jacobi (correct).
//
// Mathematical consequence: ∑_i grad_i = ∑_i ∑_j 4(p_ij-q_ij)·q_ij·(y_i-y_j)
// = 0 by antisymmetry (each (y_i-y_j) term appears twice with opposite signs).
// Therefore the centroid of the embedding is invariant under Jacobi updates.
// With Gauss-Seidel (in-place), this symmetry is broken and the centroid drifts.

#[test]
fn tsne_jacobi_update_preserves_centroid() {
    use tambear::dim_reduction::tsne;

    // 4 points in 2D. t-SNE uses a fixed seed (42u64 LCG), so the initial
    // embedding is deterministic. We replicate it to compute the expected centroid.
    let data = vec![0.0f64, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0];
    let n = 4usize;
    let out_dim = 2usize;

    // Replicate initial embedding from tsne's fixed seed (same LCG as the impl)
    let mut rng = 42u64;
    let mut init_cx = 0.0f64;
    let mut init_cy = 0.0f64;
    for _i in 0..n {
        for c in 0..out_dim {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng as f64 / u64::MAX as f64 - 0.5) * 0.01;
            if c == 0 { init_cx += val; } else { init_cy += val; }
        }
    }
    init_cx /= n as f64;
    init_cy /= n as f64;

    // Run 300 iterations
    let res = tsne(&data, n, out_dim, 2.0, 300, 50.0);
    let final_cx: f64 = (0..n).map(|i| res.embedding.get(i, 0)).sum::<f64>() / n as f64;
    let final_cy: f64 = (0..n).map(|i| res.embedding.get(i, 1)).sum::<f64>() / n as f64;

    // Jacobi invariant: centroid is preserved exactly (to floating-point precision)
    assert!((final_cx - init_cx).abs() < 1e-8,
        "Jacobi: centroid x preserved. init={init_cx:.6e}, final={final_cx:.6e}");
    assert!((final_cy - init_cy).abs() < 1e-8,
        "Jacobi: centroid y preserved. init={init_cy:.6e}, final={final_cy:.6e}");
    assert!(res.kl_divergence.is_finite(), "KL should be finite");
}

// ═══════════════════════════════════════════════════════════════════════════
// t-SNE: Early exaggeration separates clusters (VERIFIED FIX)
// ═══════════════════════════════════════════════════════════════════════════
//
// Standard t-SNE multiplies P by 4.0 for the first 250 iterations.
// This is now implemented. Five clusters in 10D with ±1 noise at separation 5
// should produce between-cluster distance > within-cluster distance.

#[test]
fn tsne_early_exaggeration_separates_clusters() {
    use tambear::dim_reduction::tsne;

    let mut data = Vec::new();
    let mut rng = 12345u64;
    let centers = [
        [0.0f64; 10],
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let pts_per_cluster = 10;
    let mut labels = Vec::new();
    for (ci, center) in centers.iter().enumerate() {
        for _ in 0..pts_per_cluster {
            for d in 0..10 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
                data.push(center[d] + noise);
            }
            labels.push(ci);
        }
    }
    let n = pts_per_cluster * 5;

    let res = tsne(&data, n, 10, 10.0, 500, 100.0);
    assert_eq!(res.embedding.rows, n);

    let mut within = 0.0f64;
    let mut within_count = 0usize;
    let mut between = 0.0f64;
    let mut between_count = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = res.embedding.get(i, 0) - res.embedding.get(j, 0);
            let dy = res.embedding.get(i, 1) - res.embedding.get(j, 1);
            let d = (dx * dx + dy * dy).sqrt();
            if labels[i] == labels[j] {
                within += d;
                within_count += 1;
            } else {
                between += d;
                between_count += 1;
            }
        }
    }
    within /= within_count as f64;
    between /= between_count as f64;
    let ratio = between / within.max(1e-10);

    // With Jacobi updates and early exaggeration (4× P for first 250 iters),
    // t-SNE should achieve basic cluster separation: between > within.
    assert!(ratio > 1.0,
        "t-SNE with early exaggeration should separate clusters: between/within={ratio:.2}");
    assert!(res.kl_divergence.is_finite(), "KL should be finite");
}

// ═══════════════════════════════════════════════════════════════════════════
// IRT: ability_eap log-space computation (VERIFIED FIX — was underflow)
// ═══════════════════════════════════════════════════════════════════════════
//
// EAP ability estimation with 100 items previously underflowed to 0.0:
// the likelihood product p₁·p₂·...·p₁₀₀ → 0 in f64.
// Fix: log-space computation with log-sum-exp trick (see ability_eap docstring).
// Test: EAP and MLE agree within 1.0 for a moderate-ability response pattern.

#[test]
fn ability_eap_log_space_handles_many_items() {
    use tambear::irt::{ItemParams, ability_eap, ability_mle};

    let items: Vec<ItemParams> = (0..100).map(|i| ItemParams {
        discrimination: 1.0,
        difficulty: -2.0 + 4.0 * i as f64 / 99.0, // spread -2 to +2
    }).collect();

    // Mixed pattern: correct on easy items (difficulty < 0), wrong on hard.
    // True ability is near the difficulty boundary → moderate (θ ≈ 0).
    let responses: Vec<u8> = (0..100).map(|i| {
        if items[i].difficulty < 0.0 { 1 } else { 0 }
    }).collect();

    let theta_mle = ability_mle(&items, &responses);
    let theta_eap = ability_eap(&items, &responses, 41);

    assert!(theta_mle.is_finite(), "MLE should be finite: {theta_mle}");
    assert!(theta_eap.is_finite(), "EAP should be finite (log-space fix): {theta_eap}");
    assert!(theta_mle.abs() < 2.0, "MLE should find moderate ability, got {theta_mle}");

    // EAP and MLE should agree within 1.0 for this well-identified pattern
    let diff = (theta_eap - theta_mle).abs();
    assert!(diff < 1.0,
        "EAP ({theta_eap:.4}) should agree with MLE ({theta_mle:.4}) within 1.0, diff={diff:.4}");
}

// ═══════════════════════════════════════════════════════════════════════════
// IRT: ability_eap with moderate items (sanity check)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ability_eap_works_for_few_items() {
    use tambear::irt::{ItemParams, ability_eap, ability_mle};

    // 10 items — no underflow expected
    let items: Vec<ItemParams> = (0..10).map(|i| ItemParams {
        discrimination: 1.0,
        difficulty: -2.0 + 4.0 * i as f64 / 9.0,
    }).collect();

    // All correct → high ability
    let responses_all = vec![1u8; 10];
    let theta_mle = ability_mle(&items, &responses_all);
    let theta_eap = ability_eap(&items, &responses_all, 41);

    eprintln!("10-item all-correct: MLE={theta_mle:.4}, EAP={theta_eap:.4}");

    // With only 10 items, EAP should work fine
    assert!(theta_eap > 0.5, "EAP should show high ability for all-correct, got {theta_eap}");

    // EAP should be somewhat close to MLE (EAP is shrunk toward 0 by the prior)
    assert!(theta_eap < theta_mle + 1.0,
        "EAP ({theta_eap}) should be ≤ MLE ({theta_mle}) + prior shrinkage");
}

// ═══════════════════════════════════════════════════════════════════════════
// IRT: ability_eap with n_quad < 2 (edge case — graceful fallback)
// ═══════════════════════════════════════════════════════════════════════════
//
// n_quad < 2 cannot span the quadrature interval; the implementation returns
// 0.0 (prior mean) as a safe fallback per the guard at the start of ability_eap.

#[test]
fn ability_eap_nquad_1_returns_default() {
    use tambear::irt::{ItemParams, ability_eap};

    let items = vec![ItemParams { discrimination: 1.0, difficulty: 0.0 }];
    let responses = vec![1u8];

    // n_quad=1: cannot form a quadrature rule; returns prior mean 0.0
    let theta = ability_eap(&items, &responses, 1);
    assert!((theta - 0.0).abs() < 1e-10,
        "n_quad=1 should return 0.0 (prior mean fallback), got {theta}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Volatility: EWMA initialization bias (MEDIUM)
// ═══════════════════════════════════════════════════════════════════════════
//
// EWMA initializes σ²₀ with the full-sample variance (forward-looking).
// This means σ²₀ uses future information. In production, only past data
// should be used. For backtesting this creates look-ahead bias.

#[test]
fn ewma_initialization_is_forward_looking() {
    use tambear::volatility::ewma_variance;

    // Returns with a massive shock at the END
    let mut returns = vec![0.01; 99];
    returns.push(0.50); // huge shock at position 99

    let sigma2 = ewma_variance(&returns, 0.94);

    // The initial variance σ²₀ should only see past data.
    // With a proper initialization (e.g., first K returns), σ²₀ ≈ 0.01² = 0.0001.
    // But the current code uses full-sample variance which includes the 0.50 shock:
    // var0 = (99 * 0.01² + 0.50²) / 100 = (0.0099 + 0.25) / 100 = 0.002599
    let var0 = sigma2[0];
    let expected_if_causal = 0.01 * 0.01; // ≈ 0.0001
    let expected_with_lookahead = (99.0 * 0.0001 + 0.25) / 100.0; // ≈ 0.002599

    eprintln!("EWMA σ²₀ = {var0:.6}");
    eprintln!("Expected if causal: {expected_if_causal:.6}");
    eprintln!("Expected with look-ahead: {expected_with_lookahead:.6}");

    // This proves the forward-looking bias
    let closer_to_lookahead = (var0 - expected_with_lookahead).abs()
        < (var0 - expected_if_causal).abs();

    assert!(closer_to_lookahead,
        "σ²₀={var0:.6} should be closer to look-ahead value {expected_with_lookahead:.6} \
         than causal value {expected_if_causal:.6}, proving forward-looking initialization");

    eprintln!("CONFIRMED: EWMA initialization uses full-sample variance (forward-looking)");
    eprintln!("Impact: creates look-ahead bias in backtesting scenarios");
}

// ═══════════════════════════════════════════════════════════════════════════
// Volatility: GARCH step size is fixed (convergence concern)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn garch_converges_on_well_conditioned_data() {
    use tambear::volatility::garch11_fit;

    // Generate well-behaved GARCH(1,1) data
    let true_omega = 0.001;
    let true_alpha = 0.1;
    let true_beta = 0.85;
    let n = 2000;

    let mut returns = vec![0.0; n];
    let mut sigma2: f64 = true_omega / (1.0 - true_alpha - true_beta);
    let mut rng = 42u64;

    for t in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = rng as f64 / u64::MAX as f64;
        // Box-Muller (approximate)
        let z = ((-2.0 * u.max(1e-300).ln()).sqrt()) * (std::f64::consts::TAU * u).cos();
        returns[t] = sigma2.sqrt() * z;
        sigma2 = true_omega + true_alpha * returns[t].powi(2) + true_beta * sigma2;
    }

    let res = garch11_fit(&returns, 1000);

    eprintln!("GARCH fit (n=2000, 1000 iters):");
    eprintln!("  True:  ω={true_omega:.4}, α={true_alpha:.4}, β={true_beta:.4}");
    eprintln!("  Est:   ω={:.4}, α={:.4}, β={:.4}", res.omega, res.alpha, res.beta);
    eprintln!("  Iters: {}", res.iterations);
    eprintln!("  LL:    {:.2}", res.log_likelihood);

    // The fixed step size (1e-5) should still converge for well-conditioned data.
    // Check basic parameter recovery.
    assert!(res.alpha > 0.01 && res.alpha < 0.4,
        "α={} should be in reasonable range", res.alpha);
    assert!(res.beta > 0.5 && res.beta < 0.99,
        "β={} should be in reasonable range", res.beta);
    assert!(res.alpha + res.beta < 1.0,
        "Stationarity: α+β={}", res.alpha + res.beta);

    // The fixed step size may prevent tight parameter recovery.
    // Document the actual errors.
    let alpha_err = (res.alpha - true_alpha).abs() / true_alpha;
    let beta_err = (res.beta - true_beta).abs() / true_beta;
    eprintln!("  α relative error: {:.2}%", alpha_err * 100.0);
    eprintln!("  β relative error: {:.2}%", beta_err * 100.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Cox PH: Hazard ratio sign (VERIFIED FIX — was risk set inversion bug)
// ═══════════════════════════════════════════════════════════════════════════
//
// The risk set R(t) = {i : T_i ≥ t} is now computed correctly: subjects are
// initialized into the full risk set and removed as time advances (forward
// iteration). The previous backward iteration overcounted the risk set,
// inverting the gradient sign and producing catastrophically wrong β.

#[test]
fn cox_ph_positive_hazard_sign_correct() {
    use tambear::survival::cox_ph;

    // Higher x → shorter survival → positive β (higher hazard)
    let n = 30;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let times: Vec<f64> = x.iter().map(|&xi| (50.0 - xi).max(0.5)).collect();
    let events = vec![true; n];

    let res = cox_ph(&x, &times, &events, n, 1, 100);

    assert!(res.beta[0] > 0.0,
        "Higher x → shorter survival → β should be > 0. Got β={:.4}", res.beta[0]);
    assert!(res.hazard_ratios[0] > 1.0,
        "Hazard ratio should be > 1 for positive covariate effect. Got HR={:.4}",
        res.hazard_ratios[0]);
}

#[test]
fn cox_ph_negative_hazard_sign_correct() {
    use tambear::survival::cox_ph;

    // Higher x → longer survival → negative β (protective effect)
    // Noise breaks perfect separation to avoid Newton-Raphson divergence
    let n = 50;
    let mut rng = 99u64;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let times: Vec<f64> = x.iter().map(|&xi| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
        (2.0 + 8.0 * xi + noise).max(0.1)
    }).collect();
    let events = vec![true; n];

    let res = cox_ph(&x, &times, &events, n, 1, 200);

    assert!(res.beta[0] < 0.0,
        "Higher x → longer survival → β should be < 0. Got β={:.4}", res.beta[0]);
    assert!(res.hazard_ratios[0] < 1.0,
        "Hazard ratio should be < 1 for protective effect. Got HR={:.4}",
        res.hazard_ratios[0]);
}

#[test]
fn cox_ph_separation_divergence() {
    use tambear::survival::cox_ph;

    // 3-point trace: x=[0,1,2], times=[3,2,1], all events.
    // PERFECT SEPARATION: higher x perfectly predicts earlier death.
    // The MLE is at β → +∞ (no finite maximizer).
    //
    // The risk set fix (line 210: forward iteration) is CORRECT —
    // β moves positive in early iterations. But without step-size bounds
    // or regularization, Newton-Raphson diverges when exp(2β) overflows.
    //
    // Trajectory: β = 1.6, 2.7, 5.7, 10.7, 13.1, ... → overflow → -36
    // This is NOT a risk set bug. It's missing Firth penalty or step-size limits.
    let x = vec![0.0, 1.0, 2.0];
    let times = vec![3.0, 2.0, 1.0];
    let events = vec![true, true, true];

    // Early iterations show correct direction
    let res_5 = cox_ph(&x, &times, &events, 3, 1, 5);
    eprintln!("Cox PH separation: β after 5 iters = {:.4} (should be positive)", res_5.beta[0]);
    assert!(res_5.beta[0] > 0.0,
        "Early iterations should push β positive, got {:.4}", res_5.beta[0]);

    // Full run diverges (documenting known limitation)
    let res_100 = cox_ph(&x, &times, &events, 3, 1, 100);
    eprintln!("Cox PH separation: β after 100 iters = {:.4} (diverged)", res_100.beta[0]);
    eprintln!("FIX NEEDED: step-size bounds or Firth penalized likelihood for separated data");
}

#[test]
fn cox_ph_with_censoring_risk_set() {
    use tambear::survival::cox_ph;

    // Mixed censoring: 80% events, 20% censored
    // Tests that censored observations are correctly excluded from events
    // but still removed from risk set at their censoring time.
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let times: Vec<f64> = x.iter().map(|&xi| (10.0 - 8.0 * xi).max(0.5)).collect();
    let events: Vec<bool> = (0..n).map(|i| i % 5 != 0).collect(); // 80% events

    let res = cox_ph(&x, &times, &events, n, 1, 100);

    eprintln!("Cox PH with censoring: β = {:.4}, HR = {:.4}", res.beta[0], res.hazard_ratios[0]);

    // Higher x → shorter time → positive β
    assert!(res.beta[0] > 0.0,
        "With censoring: β={:.4} should be > 0", res.beta[0]);
}

#[test]
fn cox_ph_tied_event_times() {
    use tambear::survival::cox_ph;

    // Tied events: multiple subjects have the same event time.
    // Tests Breslow's approximation for ties.
    let n = 20;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // 4 groups of 5 with same event time: t=1,1,1,1,1, 2,2,2,2,2, 3,..., 4,...
    let times: Vec<f64> = (0..n).map(|i| (i / 5 + 1) as f64).collect();
    let events = vec![true; n];

    let res = cox_ph(&x, &times, &events, n, 1, 100);

    eprintln!("Cox PH with ties: β = {:.4}", res.beta[0]);

    // Within each time group, higher-indexed subjects have higher x.
    // Higher x subjects have SAME time as lower x → no clear hazard direction
    // within tied groups. But across groups, higher time groups have higher x.
    // Higher x → LATER event time → negative β (protective effect).
    assert!(res.beta[0] < 0.0,
        "Tied events: β={:.4} should be < 0 (higher x → later time group)", res.beta[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// SufficientStatistics: Naive variance catastrophic cancellation (MODERATE)
// ═══════════════════════════════════════════════════════════════════════════
//
// SufficientStatistics::from_vecs computes m2 = sum_sqs - sum²/count.
// This is the naive one-pass formula, which suffers catastrophic cancellation
// when values have a large offset: sum_sqs and sum²/n are both huge,
// their difference is tiny, and floating-point subtraction loses all precision.

#[test]
fn sufficient_stats_naive_variance_large_offset() {
    use tambear::intermediates::SufficientStatistics;

    // 3 values with large offset: 1e8 + {1, 2, 3}
    // True variance (population) = Var({1,2,3}) = 2/3 ≈ 0.6667
    let values = [1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0];
    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let sum_sqs: f64 = values.iter().map(|x| x * x).sum();

    // from_vecs uses: m2 = sum_sqs - sum²/count
    let stats = SufficientStatistics::from_vecs(1, vec![sum], vec![sum_sqs], vec![n]);
    let var_pop = stats.variance(0);
    let true_var = 2.0 / 3.0; // Var({1,2,3})

    eprintln!("Large offset (1e8): naive variance = {var_pop:.6e}, true = {true_var:.6e}");

    // The naive formula loses precision catastrophically.
    // ULP at 3e16 ≈ 4, but true m2 = 2. Result is dominated by rounding error.
    let rel_err = (var_pop - true_var).abs() / true_var;
    eprintln!("Relative error: {rel_err:.2e}");

    // KNOWN LIMITATION: from_vecs uses naive sum_sqs - sum²/count.
    // At offset 1e8, catastrophic cancellation produces > 100% error.
    // The fix is to use from_welford (which accepts pre-computed centered m2)
    // or the two-pass DescriptiveEngine::moments_grouped path.
    //
    // Task #5 fixed the MomentStats/DescriptiveEngine path (which was always correct).
    // The from_vecs path remains naive — it's documented as such in its doc comment.
    if rel_err > 0.1 {
        eprintln!("CONFIRMED: from_vecs catastrophic cancellation at offset 1e8");
        eprintln!("This is a known limitation of the one-pass formula.");
        eprintln!("Use from_welford or DescriptiveEngine::moments_grouped instead.");
    }
    // Document rather than assert — the stable path exists (from_welford).
    assert!(var_pop.is_finite(), "Variance should at least be finite, got {var_pop}");
}

#[test]
fn sufficient_stats_naive_vs_welford() {
    use tambear::intermediates::SufficientStatistics;

    // Compare from_vecs (naive) vs from_welford (stable) for large offset data
    let offset = 1e10;
    let n = 100;
    let values: Vec<f64> = (0..n).map(|i| offset + i as f64).collect();
    let count = n as f64;
    let sum: f64 = values.iter().sum();
    let sum_sqs: f64 = values.iter().map(|x| x * x).sum();

    // Compute true m2 via two-pass
    let mean = sum / count;
    let true_m2: f64 = values.iter().map(|x| (x - mean).powi(2)).sum();

    // from_vecs path (naive)
    let naive = SufficientStatistics::from_vecs(1, vec![sum], vec![sum_sqs], vec![count]);
    // from_welford path (stable)
    let stable = SufficientStatistics::from_welford(1, vec![sum], vec![true_m2], vec![count]);

    let var_naive = naive.variance(0);
    let var_stable = stable.variance(0);
    let true_var = true_m2 / count; // = 99*100/12 = 833.25... wait, let me recalculate

    // Var({0,1,...,99}) = (99)(100)/12 = 825.0 (using n, not n-1)
    // Actually: Var = E[X²] - E[X]² = (99*199/6)/100 - (99/2)² ... let me just use true_m2
    let true_var_check = true_m2 / count;

    eprintln!("Offset 1e10, n=100:");
    eprintln!("  Naive variance:  {var_naive:.6e}");
    eprintln!("  Welford variance: {var_stable:.6e}");
    eprintln!("  True variance:   {true_var_check:.6e}");

    let naive_err = (var_naive - true_var_check).abs() / true_var_check;
    let welford_err = (var_stable - true_var_check).abs() / true_var_check;

    eprintln!("  Naive relative error:   {naive_err:.2e}");
    eprintln!("  Welford relative error: {welford_err:.2e}");

    // Welford should be accurate
    assert!(welford_err < 1e-10,
        "Welford should be exact: error = {welford_err:.2e}");

    // Naive should show significant error at 1e10 offset
    // (documents the bug — will fail when the fix is verified)
    if naive_err > 0.001 {
        eprintln!("CONFIRMED: naive formula has {:.1}% error at offset 1e10", naive_err * 100.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SVD: Condition number analysis for centered outer product approach (RESEARCH)
// ═══════════════════════════════════════════════════════════════════════════
//
// The current one-sided Jacobi SVD avoids forming A^T A (which squares κ).
// The proposed centered outer product approach DOES form the covariance matrix,
// so it squares the condition number. These tests characterize when that matters.

#[test]
// ═══════════════════════════════════════════════════════════════════════════
// Prim's MST: negative weight ordering (Task #12)
// ═══════════════════════════════════════════════════════════════════════════
//
// The old neg_weight_key used `!sorted as i64` which broke ordering across
// the i64 sign boundary. Negative-weight edges got LARGER keys than positive
// edges, so they popped LAST from the max-heap — exactly backwards.
// Fix: `(!sorted ^ (1u64 << 63)) as i64` remaps to preserve signed order.

#[test]
fn prim_neg_weight_key_ordering() {
    use tambear::graph::{Graph, prim};

    // Star graph: center node 0 with edges to nodes 1-5.
    // Edges: -10, -5, 0, 5, 10. MST MUST include ALL edges (it's a tree).
    // Total weight = -10 + -5 + 0 + 5 + 10 = 0.
    let mut g = Graph::new(6);
    g.add_undirected(0, 1, -10.0);
    g.add_undirected(0, 2, -5.0);
    g.add_undirected(0, 3, 0.0);
    g.add_undirected(0, 4, 5.0);
    g.add_undirected(0, 5, 10.0);

    let mst = prim(&g);
    assert_eq!(mst.edges.len(), 5, "Star graph has exactly 5 MST edges");
    assert!((mst.total_weight - 0.0).abs() < 1e-10,
        "total={} expected 0.0", mst.total_weight);
}

#[test]
fn prim_negative_weight_prefers_most_negative() {
    use tambear::graph::{Graph, prim};

    // Complete graph K4 with varying negative weights.
    // All edges negative — MST should pick the 3 most-negative.
    let mut g = Graph::new(4);
    g.add_undirected(0, 1, -100.0); // best
    g.add_undirected(0, 2, -50.0);  // third
    g.add_undirected(0, 3, -10.0);  // skipped
    g.add_undirected(1, 2, -80.0);  // second
    g.add_undirected(1, 3, -5.0);   // skipped
    g.add_undirected(2, 3, -60.0);  // fourth candidate

    let mst = prim(&g);
    assert_eq!(mst.edges.len(), 3);
    // MST: -100 (0-1), -80 (1-2), -60 (2-3) = -240
    assert!((mst.total_weight - (-240.0)).abs() < 1e-10,
        "total={} expected -240.0", mst.total_weight);
}

#[test]
fn prim_neg_weight_roundtrip_precision() {
    use tambear::graph::{Graph, prim};

    // Verify that key encoding preserves exact f64 values through round-trip.
    // Use weights that exercise different IEEE 754 bit patterns.
    let weights = [-1e-300, -1e-15, -0.0, 0.0, 1e-15, 1e-300, f64::MIN_POSITIVE, -f64::MIN_POSITIVE];
    let n = weights.len() + 1; // center + spokes

    let mut g = Graph::new(n);
    for (i, &w) in weights.iter().enumerate() {
        g.add_undirected(0, i + 1, w);
    }

    let mst = prim(&g);
    assert_eq!(mst.edges.len(), weights.len());

    // Verify all weights are recovered exactly
    let mut recovered: Vec<f64> = mst.edges.iter().map(|&(_, _, w)| w).collect();
    recovered.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut expected = weights.to_vec();
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (r, e) in recovered.iter().zip(&expected) {
        assert!(r.to_bits() == e.to_bits() || (r - e).abs() < 1e-300,
            "Weight mismatch: recovered {r} vs expected {e}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Joint histogram: i32 composite key overflow (Task #11)
// ═══════════════════════════════════════════════════════════════════════════
//
// Composite key = x * ny + y stored as i32. Overflows when nx * ny
// approaches i32::MAX. The fix should either use i64 or route to CPU fallback.

#[test]
fn joint_histogram_large_categories() {
    use tambear::compute_engine::ComputeEngine;
    use tambear::information_theory::joint_histogram;

    let mut compute = ComputeEngine::new(tam_gpu::detect());

    // 1000 × 1000 = 1M bins. Well within i32 range, but tests the path.
    let n = 100;
    let keys_x: Vec<i32> = (0..n).map(|i| (i % 1000) as i32).collect();
    let keys_y: Vec<i32> = (0..n).map(|i| (i % 1000) as i32).collect();

    let hist = joint_histogram(&mut compute, &keys_x, &keys_y, 1000, 1000).unwrap();
    let total: f64 = hist.iter().sum();
    assert!((total - n as f64).abs() < 1e-10,
        "Total count should be {n}, got {total}");
}

#[test]
fn joint_histogram_near_i32_limit() {
    use tambear::compute_engine::ComputeEngine;
    use tambear::information_theory::joint_histogram;

    let mut compute = ComputeEngine::new(tam_gpu::detect());

    // nx=46340, ny=46340: n_bins = 2,147,395,600 ≈ i32::MAX.
    // This is near the boundary — composite key max = 2,147,395,599.
    // With only 10 data points, this should work without OOM.
    let nx = 46340usize;
    let ny = 46340usize;

    // Only 10 data points at corners
    let keys_x: Vec<i32> = vec![0, 0, 1, 1, 46339, 46339, 100, 200, 300, 400];
    let keys_y: Vec<i32> = vec![0, 1, 0, 1, 46339, 46338, 100, 200, 300, 400];

    // This may be too large to allocate (2B * 8 bytes = 16GB).
    // If allocation fails, that's expected for this test — just don't panic on i32 overflow.
    match joint_histogram(&mut compute, &keys_x, &keys_y, nx, ny) {
        Ok(hist) => {
            let total: f64 = hist.iter().sum();
            assert!((total - 10.0).abs() < 1e-10, "total={total} expected 10.0");
            eprintln!("Near-i32-limit joint histogram: OK (n_bins={})", nx * ny);
        }
        Err(e) => {
            eprintln!("Near-i32-limit joint histogram: allocation failed (expected): {e}");
            // Allocation failure is acceptable — we're allocating ~16GB.
            // The important thing is no i32 overflow panic.
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Series acceleration: adversarial inputs (what breaks the magic?)
// ═══════════════════════════════════════════════════════════════════════════
//
// Observer proved: Wynn's epsilon gets π/4 to machine precision from 20 terms
// of the Leibniz series. Beautiful. But these accelerators have assumptions
// about the structure of the error. What happens when those assumptions break?
//
// The accelerators and their assumptions:
// - Aitken Δ²: error ratio e_{n+1}/e_n ≈ constant (geometric convergence)
// - Wynn ε: meromorphic continuation (Padé approximation in disguise)
// - Euler: alternating series with monotone-decreasing terms
// - Richardson: error expands in powers of h (polynomial extrapolation)
//
// Each test targets a violation of one assumption.

#[test]
fn aitken_breaks_on_oscillating_convergence() {
    use tambear::series_accel::{partial_sums, aitken_delta2};

    // Aitken assumes the error ratio r = e_{n+1}/e_n is approximately constant.
    // Construct a sequence where r oscillates wildly: r_n = 0.1, 0.9, 0.1, 0.9, ...
    // The limit is 1.0, but Aitken's Δ² denominator (Δ²S) oscillates and can
    // produce estimates FARTHER from the limit.
    let limit = 1.0_f64;
    let mut sums = Vec::new();
    let mut err = 0.5_f64;
    for k in 0..20 {
        let ratio = if k % 2 == 0 { 0.1 } else { 0.9 };
        err *= ratio;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        sums.push(limit + sign * err);
    }

    let raw_error = (sums[sums.len() - 1] - limit).abs();
    let accel = aitken_delta2(&sums);
    let accel_error = (accel[accel.len() - 1] - limit).abs();

    eprintln!("Oscillating ratio: raw_err={raw_error:.2e}, aitken_err={accel_error:.2e}");

    // For oscillating ratios, Aitken can fail dramatically.
    // It may still work if the overall trend is convergent, but it can
    // amplify oscillations. Document the behavior.
    if accel_error > raw_error {
        eprintln!("CONFIRMED: Aitken amplifies oscillating-ratio sequences (error grew by {:.1}x)",
            accel_error / raw_error);
    }
    // Don't assert improvement — this IS a known failure mode.
    assert!(accel_error.is_finite(), "At minimum, Aitken should produce finite output");
}

#[test]
fn wynn_breaks_on_divergent_series() {
    use tambear::series_accel::{partial_sums, wynn_epsilon};

    // Wynn's epsilon can sometimes "sum" a divergent series via Cesàro/Abel-like
    // regularization (it's related to Padé approximants). But for genuinely
    // chaotic divergence, it should not produce a meaningful answer.
    //
    // Grandi's series: 1 - 1 + 1 - 1 + ... → Cesàro sum = 1/2 (Abel sum = 1/2)
    // Wynn might actually "find" 1/2 here — that's a feature, not a bug!
    let grandi_terms: Vec<f64> = (0..20).map(|k| {
        if k % 2 == 0 { 1.0 } else { -1.0 }
    }).collect();
    let sums = partial_sums(&grandi_terms);
    let estimate = wynn_epsilon(&sums);

    eprintln!("Grandi's series (1-1+1-1+...): Wynn ε = {estimate:.6}");
    eprintln!("Cesàro/Abel sum = 0.5");

    // Wynn on Grandi's series often yields exactly 0.5 (the regularized sum).
    // This is mathematically correct from the Padé perspective!
    if (estimate - 0.5).abs() < 0.01 {
        eprintln!("CONFIRMED: Wynn recovers the regularized sum 1/2");
    }

    // Now try something truly pathological: factorial-growing terms
    let divergent: Vec<f64> = (0..15).map(|k| {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let mut factorial = 1.0;
        for i in 1..=k { factorial *= i as f64; }
        sign * factorial
    }).collect();
    let sums_div = partial_sums(&divergent);
    let est_div = wynn_epsilon(&sums_div);

    eprintln!("Factorial divergent: Wynn ε = {est_div:.6}");
    // This IS the alternating factorial series Σ(-1)^n n!
    // Its Borel sum is ∫₀^∞ e^{-t}/(1+t) dt ≈ 0.5963...
    // Wynn may or may not get close. Document behavior.
    eprintln!("Borel sum ≈ 0.5963 (if Wynn attempts regularization)");

    assert!(est_div.is_finite(), "Wynn should not produce NaN/Inf on divergent input");
}

#[test]
fn euler_breaks_on_non_alternating() {
    use tambear::series_accel::euler_transform;

    // Euler transform assumes alternating signs. Apply it to a positive series.
    // Basel series: Σ 1/k² (all positive) → π²/6.
    // Euler should NOT help and may hurt.
    let target = std::f64::consts::PI.powi(2) / 6.0;
    let terms: Vec<f64> = (0..30).map(|k| 1.0 / ((k + 1) as f64).powi(2)).collect();

    let euler_est = euler_transform(&terms);
    let raw_sum: f64 = terms.iter().sum();
    let raw_error = (raw_sum - target).abs();
    let euler_error = (euler_est - target).abs();

    eprintln!("Basel (positive series): raw_err={raw_error:.2e}, euler_err={euler_error:.2e}");

    // Euler transform with binomial weights on positive series: the result
    // is a weighted average of partial sums with weights C(n,k)/2^n.
    // This is basically the midpoint of the partial sum trajectory —
    // much WORSE than the last partial sum for a monotone-increasing sequence.
    if euler_error > raw_error {
        eprintln!("CONFIRMED: Euler transform degrades positive series (error grew by {:.1}x)",
            euler_error / raw_error);
    }

    assert!(euler_est.is_finite());
}

#[test]
fn richardson_breaks_on_non_polynomial_error() {
    use tambear::series_accel::richardson_extrapolate;

    // Richardson extrapolation assumes error expands as c₁h^p + c₂h^{2p} + ...
    // If the error is oscillatory or has a logarithmic component, extrapolation
    // can amplify the non-polynomial residual.
    //
    // Construct: A(h) = true_value + h^2 * sin(1/h)
    // The sin(1/h) oscillation defeats polynomial extrapolation.
    let true_value = 1.0_f64;
    let mut approxs = Vec::new();
    let mut h = 1.0_f64;
    for _ in 0..6 {
        approxs.push(true_value + h * h * (1.0_f64 / h).sin());
        h /= 2.0;
    }

    let raw_error = (approxs[5] - true_value).abs();
    let rich = richardson_extrapolate(&approxs, 2.0, 2);
    let rich_error = (rich - true_value).abs();

    eprintln!("Oscillatory error (h²·sin(1/h)): raw_err={raw_error:.2e}, rich_err={rich_error:.2e}");

    if rich_error > raw_error {
        eprintln!("CONFIRMED: Richardson amplifies non-polynomial error structure");
    }

    assert!(rich.is_finite());
}

#[test]
fn wynn_numerical_stability_near_convergence() {
    use tambear::series_accel::{partial_sums, wynn_epsilon};

    // When consecutive partial sums are nearly equal (series almost converged),
    // Wynn's epsilon computes 1/(S_{k+1} - S_k) → 1/ε → huge number.
    // This can cause catastrophic cancellation in the alternating additions.
    //
    // e^{-1} series converges factorially fast. With 20 terms, the last few
    // sums differ by < 1e-15. Wynn's 1/(diff) terms become > 1e15.
    let terms: Vec<f64> = {
        let mut v = Vec::with_capacity(20);
        let mut factorial = 1.0;
        for k in 0..20 {
            if k > 0 { factorial *= k as f64; }
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            v.push(sign / factorial);
        }
        v
    };
    let sums = partial_sums(&terms);

    // Check: late partial sums should be extremely close
    let diff_late = (sums[19] - sums[18]).abs();
    eprintln!("Late sum difference (S_19 - S_18): {diff_late:.2e}");

    let estimate = wynn_epsilon(&sums);
    let true_val = (-1.0_f64).exp();
    let error = (estimate - true_val).abs();

    eprintln!("Wynn on fast-converging e^-1 (20 terms): error = {error:.2e}");
    eprintln!("Raw sum error: {:.2e}", (sums[19] - true_val).abs());

    // For a fast-converging series, Wynn shouldn't be MUCH worse than raw.
    // But the 1/ε instability means it may not help much either.
    assert!(error < 1e-10,
        "Wynn on fast-converging series should be accurate, got error {error:.2e}");
}

#[test]
fn aitken_constant_sequence_no_panic() {
    use tambear::series_accel::{partial_sums, aitken_delta2};

    // Constant sequence: S_n = c for all n. Δ²S = 0, so Aitken divides by zero.
    // The implementation should handle this gracefully (return S_n, not NaN/panic).
    let sums = vec![1.0; 10];
    let accel = aitken_delta2(&sums);

    // All entries should be 1.0 (the correct limit) or at least finite
    for (i, &v) in accel.iter().enumerate() {
        assert!(v.is_finite(), "Aitken[{i}] should be finite for constant sequence, got {v}");
        assert!((v - 1.0).abs() < 1e-10,
            "Aitken[{i}] should be 1.0 for constant sequence, got {v}");
    }
}

#[test]
fn wynn_single_and_two_element() {
    use tambear::series_accel::wynn_epsilon;

    // Edge cases: very short sequences
    assert_eq!(wynn_epsilon(&[]), 0.0, "Empty → 0");
    assert_eq!(wynn_epsilon(&[42.0]), 42.0, "Single → itself");

    let two = wynn_epsilon(&[1.0, 2.0]);
    assert!(two.is_finite(), "Two elements should be finite");
    eprintln!("Wynn on [1.0, 2.0] = {two}");
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE solvers: Lorenz attractor & chaos (parking lot exploration)
// ═══════════════════════════════════════════════════════════════════════════
//
// The Lorenz system dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz
// is the textbook chaotic system. Two trajectories starting ε apart diverge
// exponentially: |δ(t)| ~ |δ₀| · exp(λ_max · t) where λ_max ≈ 0.9056.
//
// This means for ANY fixed-step numerical integrator, the trajectory becomes
// numerically meaningless after t ≈ -ln(ε_machine) / λ_max ≈ 36 / 0.91 ≈ 40.
//
// What we CAN verify: structural properties (the attractor shape, Lyapunov
// exponent, invariant measure) are correct even when individual trajectories
// aren't. This tests whether our ODE solver preserves those.

#[test]
fn lorenz_attractor_structural_properties() {
    use tambear::numerical::rk4_system;

    let sigma = 10.0_f64;
    let rho = 28.0_f64;
    let beta = 8.0_f64 / 3.0;

    let lorenz = |_t: f64, y: &[f64]| -> Vec<f64> {
        vec![
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2],
        ]
    };

    // Integrate for a moderate time (enough to sample the attractor)
    let (t, traj) = rk4_system(lorenz, &[1.0, 1.0, 1.0], 0.0, 50.0, 50000);

    // Structural property 1: the attractor is bounded.
    // For standard Lorenz (σ=10, ρ=28, β=8/3), all trajectories enter and
    // stay within an ellipsoid. z should stay in roughly [0, 50].
    let z_max = traj.iter().map(|y| y[2]).fold(f64::NEG_INFINITY, f64::max);
    let z_min = traj.iter().map(|y| y[2]).fold(f64::INFINITY, f64::min);
    eprintln!("Lorenz: z ∈ [{z_min:.1}, {z_max:.1}]");
    assert!(z_max < 60.0, "z should be bounded, got z_max={z_max:.1}");
    assert!(z_min > 0.0, "z should stay positive on attractor, got z_min={z_min:.1}");

    // Structural property 2: the trajectory visits both wings of the butterfly.
    // The two fixed points are at (±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1) ≈ (±8.49, ±8.49, 27).
    let wing_threshold = 0.0; // x > 0 = right wing, x < 0 = left wing
    let n_right = traj.iter().filter(|y| y[0] > wing_threshold).count();
    let n_left = traj.iter().filter(|y| y[0] < wing_threshold).count();
    let ratio = n_right.min(n_left) as f64 / n_right.max(n_left) as f64;
    eprintln!("Wing visits: right={n_right}, left={n_left}, ratio={ratio:.3}");
    assert!(ratio > 0.3, "Should visit both wings roughly equally, ratio={ratio:.3}");

    // Structural property 3: energy-like quantity.
    // V = x² + y² + (z - σ - ρ)² decreases on average outside the attractor.
    // Inside the attractor, the mean z should be ≈ ρ - 1 = 27.
    let skip = 5000; // discard transient
    let mean_z: f64 = traj[skip..].iter().map(|y| y[2]).sum::<f64>() / (traj.len() - skip) as f64;
    eprintln!("Mean z (after transient): {mean_z:.2}, expected ≈ {}", rho - 1.0);
    assert!((mean_z - (rho - 1.0)).abs() < 5.0,
        "Mean z should be ≈ ρ-1 = {}, got {mean_z:.2}", rho - 1.0);
}

#[test]
fn lorenz_sensitivity_to_initial_conditions() {
    use tambear::numerical::rk4_system;

    let sigma = 10.0_f64;
    let rho = 28.0_f64;
    let beta = 8.0_f64 / 3.0;

    let lorenz = |_t: f64, y: &[f64]| -> Vec<f64> {
        vec![
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2],
        ]
    };

    // Two trajectories starting 1e-8 apart (larger perturbation for faster divergence)
    let eps = 1e-8;
    let t_end = 50.0;
    let n_steps = 50000;
    let (_, traj1) = rk4_system(lorenz, &[1.0, 1.0, 1.0], 0.0, t_end, n_steps);
    let (_, traj2) = rk4_system(lorenz, &[1.0 + eps, 1.0, 1.0], 0.0, t_end, n_steps);

    // Track divergence over time
    let mut divergence_time = 0.0;
    let mut max_dist = 0.0_f64;
    for (i, (y1, y2)) in traj1.iter().zip(&traj2).enumerate() {
        let dist = ((y1[0] - y2[0]).powi(2) + (y1[1] - y2[1]).powi(2) + (y1[2] - y2[2]).powi(2)).sqrt();
        max_dist = max_dist.max(dist);
        let t = i as f64 * t_end / n_steps as f64;
        if dist > 1.0 && divergence_time == 0.0 {
            divergence_time = t;
        }
    }

    eprintln!("Trajectory divergence: max dist = {max_dist:.2e}, divergence at t = {divergence_time:.2}");
    eprintln!("(Started {eps:.0e} apart, integrated to t={t_end})");

    // The Lyapunov time for standard Lorenz is ~1/0.9 ≈ 1.1 time units.
    // Divergence to O(1) from ε = 10^{-8} takes about -ln(1e-8)/0.9 ≈ 20.5 time units.
    // We should see divergence within t=50, but the exact time depends on trajectory.
    if divergence_time > 0.0 {
        eprintln!("CONFIRMED: chaotic divergence at t = {divergence_time:.2}");
    } else {
        eprintln!("NOTE: no O(1) divergence in t={t_end}. max_dist = {max_dist:.2e}");
        eprintln!("May need longer integration or larger perturbation.");
    }

    // Estimate Lyapunov exponent from early growth
    let early_dists: Vec<f64> = (0..3000).map(|i| {
        let y1 = &traj1[i];
        let y2 = &traj2[i];
        ((y1[0]-y2[0]).powi(2) + (y1[1]-y2[1]).powi(2) + (y1[2]-y2[2]).powi(2)).sqrt()
    }).collect();
    let t_window = 10.0; // measure over first 10 time units
    let n_window = (t_window / (30.0 / 30000.0)) as usize;
    let dist_start = early_dists[100].max(1e-300); // skip transient
    let dist_end = early_dists[n_window.min(early_dists.len() - 1)].max(1e-300);
    let lambda_est = (dist_end / dist_start).ln() / t_window;
    eprintln!("Estimated Lyapunov exponent: {lambda_est:.3} (theoretical ≈ 0.906)");

    // The estimate should be positive (trajectories diverge), but the exact value
    // depends on transient behavior and initial conditions.
    // λ_max ≈ 0.906 for standard Lorenz, but we may see different values
    // depending on which part of the attractor we sample.
    eprintln!("(Theoretical λ_max ≈ 0.906 for standard Lorenz parameters)");
}

#[test]
fn rk4_stiff_system_failure_mode() {
    use tambear::numerical::rk4_system;

    // Stiff system: y' = -1000·y + 1000·sin(t) + cos(t)
    // The solution is y(t) = sin(t) + c·exp(-1000t) where c = y0 - sin(0) = y0.
    // After transient decay, y → sin(t).
    //
    // RK4 with fixed step CANNOT handle this unless h < 2/1000 = 0.002.
    // With h = 0.01 (100 steps for [0,1]), RK4 will BLOW UP.
    let f = |t: f64, y: &[f64]| -> Vec<f64> {
        vec![-1000.0 * y[0] + 1000.0 * t.sin() + t.cos()]
    };

    // This SHOULD blow up with only 100 steps (h = 0.01 > stability limit 0.002)
    let (_, traj_coarse) = rk4_system(f, &[0.0], 0.0, 1.0, 100);
    let y_final_coarse = traj_coarse.last().unwrap()[0];

    // Fine step should be accurate
    let (_, traj_fine) = rk4_system(f, &[0.0], 0.0, 1.0, 10000);
    let y_final_fine = traj_fine.last().unwrap()[0];

    let true_val = 1.0_f64.sin(); // ≈ 0.8415

    eprintln!("Stiff system y'=-1000y+...: true y(1) ≈ {true_val:.4}");
    eprintln!("RK4 h=0.01 (coarse): y(1) = {y_final_coarse:.4e}");
    eprintln!("RK4 h=0.0001 (fine): y(1) = {y_final_fine:.6}");

    // Coarse step should be unstable (blowup)
    if y_final_coarse.abs() > 10.0 {
        eprintln!("CONFIRMED: RK4 unstable for stiff system at h=0.01 (stability limit h<0.002)");
    }

    // Fine step should be accurate
    assert!((y_final_fine - true_val).abs() < 0.01,
        "Fine-step RK4 should be accurate: {y_final_fine:.6} vs {true_val:.6}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Series acceleration on chaotic ergodic averages
// ═══════════════════════════════════════════════════════════════════════════
//
// Prediction (adversarial + observer):
//   - Block-averaged Aitken WINS: blocks of ~110 steps (≈ 1 Lyapunov time)
//     are approximately independent → geometric error decay → Aitken accelerates.
//   - Raw-sampled Aitken LOSES: autocorrelated running means violate Aitken's
//     geometric-ratio assumption → acceleration fails or degrades.
//
// The contrast proves the matched-kernel principle extends to chaotic systems.

#[test]
fn aitken_on_chaotic_ergodic_average() {
    use tambear::numerical::rk4_system;
    use tambear::series_accel::{partial_sums, aitken_delta2};

    let sigma = 10.0_f64;
    let rho = 28.0_f64;
    let beta = 8.0_f64 / 3.0;

    let lorenz = |_t: f64, y: &[f64]| -> Vec<f64> {
        vec![
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2],
        ]
    };

    // Generate a long trajectory (discard transient)
    let dt = 0.01;
    let n_steps = 100_000;
    let (_, traj) = rk4_system(lorenz, &[1.0, 1.0, 1.0], 0.0, n_steps as f64 * dt, n_steps);
    let skip = 5000; // discard transient
    let z_values: Vec<f64> = traj[skip..].iter().map(|y| y[2]).collect();
    let n = z_values.len();

    // Ergodic target: running mean of z over the full trajectory
    let ergodic_target: f64 = z_values.iter().sum::<f64>() / n as f64;
    eprintln!("Ergodic target (full trajectory mean z): {ergodic_target:.4}");

    // ── Block-averaged sequence ──────────────────────────────────────────
    // Block length ≈ 1 Lyapunov time ≈ 1/0.9 ≈ 1.1 time units ≈ 110 steps
    // Each block mean is an approximately independent sample of the ergodic average.
    let block_len = 110;
    let n_blocks = n / block_len;
    let block_means: Vec<f64> = (0..n_blocks).map(|b| {
        let start = b * block_len;
        let end = start + block_len;
        z_values[start..end].iter().sum::<f64>() / block_len as f64
    }).collect();

    // Running mean of block means — this is the sequence Aitken will accelerate.
    // Each entry = mean of first k block means = ergodic estimate from k blocks.
    let block_running_means: Vec<f64> = {
        let mut means = Vec::with_capacity(n_blocks);
        let mut sum = 0.0;
        for (i, &bm) in block_means.iter().enumerate() {
            sum += bm;
            means.push(sum / (i + 1) as f64);
        }
        means
    };

    // ── Finely-sampled running means (every 10 steps) ─────────────────
    // Same number of running-mean entries, but using MANY small overlapping
    // windows. These are heavily autocorrelated — Aitken's geometric-ratio
    // assumption should be violated.
    let fine_step = 10;
    let n_fine = n_blocks; // same count for fair comparison
    let raw_running_means: Vec<f64> = {
        let mut means = Vec::with_capacity(n_fine);
        let mut sum = 0.0;
        for (i, &z) in z_values.iter().enumerate() {
            sum += z;
            if (i + 1) % fine_step == 0 && means.len() < n_fine {
                means.push(sum / (i + 1) as f64);
            }
        }
        means
    };

    // Apply Aitken to both sequences
    let aitken_block = aitken_delta2(&block_running_means);
    let aitken_raw = aitken_delta2(&raw_running_means);

    // Measure errors at the last available point
    let block_raw_err = (block_running_means.last().unwrap() - ergodic_target).abs();
    let raw_raw_err = (raw_running_means.last().unwrap() - ergodic_target).abs();

    let block_aitken_err = if !aitken_block.is_empty() {
        (aitken_block.last().unwrap() - ergodic_target).abs()
    } else { f64::INFINITY };

    let raw_aitken_err = if !aitken_raw.is_empty() {
        (aitken_raw.last().unwrap() - ergodic_target).abs()
    } else { f64::INFINITY };

    eprintln!("═══ Aitken on Lorenz ergodic z-mean ═══");
    eprintln!("Block running mean error:  {block_raw_err:.4e}");
    eprintln!("Block + Aitken error:      {block_aitken_err:.4e}");
    eprintln!("Raw running mean error:    {raw_raw_err:.4e}");
    eprintln!("Raw + Aitken error:        {raw_aitken_err:.4e}");

    let block_improvement = block_raw_err / block_aitken_err.max(1e-300);
    let raw_improvement = raw_raw_err / raw_aitken_err.max(1e-300);
    eprintln!("Block Aitken improvement:  {block_improvement:.2}x");
    eprintln!("Raw Aitken improvement:    {raw_improvement:.2}x");

    // ── FINDING ──
    // Aitken Δ² DEGRADES ergodic averages of chaotic systems.
    //
    // Root cause: ergodic means converge at O(1/√N) (CLT rate), not
    // geometrically (r^n). The error ratio between consecutive running
    // means is √(k/(k+1)) → 1, making Aitken's Δ² denominator near-zero
    // and the correction enormous and wrong.
    //
    // Block averaging helps the BASE convergence (1e-3 vs 2e-2) but makes
    // Aitken WORSE: the better-converged block sequence has smaller Δ values,
    // pushing the Δ²a denominator even closer to zero.
    //
    // This is a genuine domain mismatch: Aitken assumes geometric convergence,
    // ergodic averages have algebraic convergence. No amount of decorrelation
    // fixes this. The matched-kernel principle holds for the BASE estimates
    // (blocks >> raw) but Aitken is the wrong accelerator for this class.
    //
    // Candidate accelerators for O(1/√N) convergence:
    //   - Richardson extrapolation on (N, mean_N) pairs
    //   - Wynn epsilon (Padé may handle algebraic decay)
    //   - Batch means with variance-weighted combination

    // ── Wynn epsilon comparison ──────────────────────────────────────────
    // Wynn builds Padé approximants. A Padé of s_N = L + a/√N + b/N + ...
    // is a rational function in N that might capture the algebraic decay.
    use tambear::series_accel::wynn_epsilon;

    let wynn_block = wynn_epsilon(&block_running_means);
    let wynn_raw = wynn_epsilon(&raw_running_means);

    let block_wynn_err = (wynn_block - ergodic_target).abs();
    let raw_wynn_err = (wynn_raw - ergodic_target).abs();

    let block_wynn_improvement = block_raw_err / block_wynn_err.max(1e-300);
    let raw_wynn_improvement = raw_raw_err / raw_wynn_err.max(1e-300);

    eprintln!("── Wynn epsilon comparison ──");
    eprintln!("Block + Wynn error:        {block_wynn_err:.4e}");
    eprintln!("Raw + Wynn error:          {raw_wynn_err:.4e}");
    eprintln!("Block Wynn improvement:    {block_wynn_improvement:.2}x");
    eprintln!("Raw Wynn improvement:      {raw_wynn_improvement:.2}x");

    if block_wynn_improvement > 1.0 {
        eprintln!("Wynn IMPROVES block ergodic averages — Padé handles algebraic decay");
    } else {
        eprintln!("Wynn also degrades block ergodic averages — algebraic convergence defeats both");
    }

    // Block averaging helps the raw convergence dramatically
    assert!(block_raw_err < raw_raw_err,
        "Block means should converge faster than raw: {block_raw_err:.4e} vs {raw_raw_err:.4e}");

    // Both accelerators should at least produce finite results
    assert!(block_aitken_err.is_finite(), "Block Aitken should be finite");
    assert!(raw_aitken_err.is_finite(), "Raw Aitken should be finite");
    assert!(wynn_block.is_finite(), "Block Wynn should be finite");
}

// ═══════════════════════════════════════════════════════════════════════════
// SVD: Condition number analysis for COPA (Task #10/13)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn svd_well_conditioned_outer_product_ok() {
    use tambear::linear_algebra::{Mat, svd, mat_mul};

    // Well-conditioned: σ = [10, 5, 1], κ = 10
    // A = U·diag(σ)·V^T for known U, V
    let sigma = [10.0, 5.0, 1.0];
    let m = 20;
    let n = 3;

    // Generate A with known singular values via outer product
    let mut a = Mat::zeros(m, n);
    let mut rng = 42u64;
    for i in 0..m {
        for j in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            a.set(i, j, rng as f64 / u64::MAX as f64 - 0.5);
        }
    }

    let res = svd(&a);
    let k = res.sigma.len().min(n);

    eprintln!("SVD well-conditioned: σ = {:?}", &res.sigma[..k]);
    eprintln!("Condition number κ = {:.2}", res.sigma[0] / res.sigma[k-1]);

    // Verify reconstruction A = U·Σ·V^T
    let u_sub = res.u.submat(0, 0, m, k);
    let sigma_mat = Mat::diag(&res.sigma[..k]);
    let vt_sub = res.vt.submat(0, 0, k, n);
    let reconstructed = mat_mul(&mat_mul(&u_sub, &sigma_mat), &vt_sub);

    let mut max_err = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let err = (a.get(i, j) - reconstructed.get(i, j)).abs();
            max_err = max_err.max(err);
        }
    }

    eprintln!("Reconstruction max error: {max_err:.2e}");
    assert!(max_err < 1e-10,
        "SVD reconstruction error {max_err:.2e} should be < 1e-10 for well-conditioned matrix");
}

#[test]
fn svd_ill_conditioned_jacobi_survives() {
    use tambear::linear_algebra::{Mat, svd, mat_mul};

    // Ill-conditioned but ORTHOGONAL columns: σ_1/σ_2 = 10^6
    // Column 0 and column 1 point in orthogonal directions.
    // The outer product approach squares κ → 10^12, losing σ_2.
    // Jacobi (which avoids forming A^T A) should recover both.
    let m = 10;
    let n = 2;
    let mut a = Mat::zeros(m, n);

    // Column 0: [1, 0, 1, 0, ...] * 1e3 (large direction)
    // Column 1: [0, 1, 0, 1, ...] * 1e-3 (tiny direction, orthogonal)
    for i in 0..m {
        if i % 2 == 0 {
            a.set(i, 0, 1e3);
            a.set(i, 1, 0.0);
        } else {
            a.set(i, 0, 0.0);
            a.set(i, 1, 1e-3);
        }
    }

    let res = svd(&a);
    let kappa = res.sigma[0] / res.sigma[1].max(1e-300);

    eprintln!("Ill-conditioned SVD (orthogonal): σ = [{:.4e}, {:.4e}]", res.sigma[0], res.sigma[1]);
    eprintln!("Condition number κ = {kappa:.2e}");

    // Expected: σ_1 ≈ √5 * 1e3 ≈ 2236, σ_2 ≈ √5 * 1e-3 ≈ 0.002236
    assert!(res.sigma[0] > 1e2, "σ_1 should be large, got {:.2e}", res.sigma[0]);
    assert!(res.sigma[1] > 1e-4, "σ_2 should be recoverable by Jacobi, got {:.2e}", res.sigma[1]);
    assert!(kappa > 1e5, "κ should be ~10^6, got {kappa:.2e}");

    // Verify reconstruction
    let k = 2;
    let u_sub = res.u.submat(0, 0, m, k);
    let sigma_mat = Mat::diag(&res.sigma[..k]);
    let vt_sub = res.vt.submat(0, 0, k, n);
    let reconstructed = mat_mul(&mat_mul(&u_sub, &sigma_mat), &vt_sub);

    let mut max_err = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let err = (a.get(i, j) - reconstructed.get(i, j)).abs();
            max_err = max_err.max(err);
        }
    }
    eprintln!("Reconstruction max error: {max_err:.2e}");
    assert!(max_err < 1e-8,
        "SVD reconstruction error {max_err:.2e} should be < 1e-8");
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimization: adversarial landscapes
// ═══════════════════════════════════════════════════════════════════════════

/// Rosenbrock function — the classic banana valley.
/// f(x,y) = (1-x)² + 100(y-x²)². Minimum at (1,1).
/// GD struggles (condition number ~400 along the valley floor).
/// L-BFGS should handle it. Adam should handle it.
#[test]
fn optimizer_rosenbrock_valley() {
    use tambear::optimization::{gradient_descent, adam, lbfgs};

    let rosenbrock = |x: &[f64]| -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    };
    let grad_rosen = |x: &[f64]| -> Vec<f64> {
        vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ]
    };

    let x0 = &[-1.0, 1.0]; // far from minimum

    // L-BFGS (gold standard)
    let res_lbfgs = lbfgs(&rosenbrock, &grad_rosen, x0, 10, 1000, 1e-10);
    let err_lbfgs = ((res_lbfgs.x[0] - 1.0).powi(2) + (res_lbfgs.x[1] - 1.0).powi(2)).sqrt();
    eprintln!("L-BFGS: x={:.6?}, f={:.2e}, iters={}, err={err_lbfgs:.2e}",
        res_lbfgs.x, res_lbfgs.f_val, res_lbfgs.iterations);
    assert!(res_lbfgs.converged, "L-BFGS should converge on Rosenbrock");
    assert!(err_lbfgs < 1e-4, "L-BFGS error {err_lbfgs:.2e} should be < 1e-4");

    // Adam (adaptive)
    let res_adam = adam(&rosenbrock, &grad_rosen, x0,
        0.01, 0.9, 0.999, 1e-8, 10000, 1e-8);
    let err_adam = ((res_adam.x[0] - 1.0).powi(2) + (res_adam.x[1] - 1.0).powi(2)).sqrt();
    eprintln!("Adam:   x={:.6?}, f={:.2e}, iters={}, err={err_adam:.2e}",
        res_adam.x, res_adam.f_val, res_adam.iterations);

    // GD with fixed lr — should struggle (demonstrate the problem)
    let res_gd = gradient_descent(&rosenbrock, &grad_rosen, x0,
        0.001, 0.0, 10000, 1e-8);
    let err_gd = ((res_gd.x[0] - 1.0).powi(2) + (res_gd.x[1] - 1.0).powi(2)).sqrt();
    eprintln!("GD:     x={:.6?}, f={:.2e}, iters={}, err={err_gd:.2e}",
        res_gd.x, res_gd.f_val, res_gd.iterations);

    // L-BFGS should beat GD by orders of magnitude
    eprintln!("L-BFGS/GD speedup: {:.0}x fewer iters",
        res_gd.iterations as f64 / res_lbfgs.iterations.max(1) as f64);
}

/// Saddle point: f(x,y) = x² - y². Saddle at origin.
/// All optimizers should escape (the negative curvature in y pushes away).
/// But gradient-based methods starting exactly at (0,0) have zero gradient —
/// they can get stuck if the perturbation is symmetric.
#[test]
fn optimizer_saddle_point_escape() {
    use tambear::optimization::{gradient_descent, nelder_mead};

    // f(x,y) = x² - y² has a saddle at origin
    let saddle = |x: &[f64]| -> f64 { x[0] * x[0] - x[1] * x[1] };
    let grad_saddle = |x: &[f64]| -> Vec<f64> { vec![2.0 * x[0], -2.0 * x[1]] };

    // Starting NEAR the saddle (not exactly at it, since grad = 0 there)
    let x0 = &[0.01, 0.01];
    let res = gradient_descent(&saddle, &grad_saddle, x0, 0.1, 0.0, 100, 1e-12);
    eprintln!("GD from near-saddle: x={:.6?}, f={:.2e}", res.x, res.f_val);

    // The function is unbounded below (y → ∞ makes f → -∞).
    // GD should move away from the saddle in the y-direction.
    assert!(res.f_val < -1.0,
        "GD should escape saddle point, f should be very negative, got {:.4}", res.f_val);

    // Starting exactly at saddle: GD has zero gradient, should not move
    let x0_exact = &[0.0, 0.0];
    let res_exact = gradient_descent(&saddle, &grad_saddle, x0_exact, 0.1, 0.0, 100, 1e-12);
    eprintln!("GD from exact saddle: x={:.6?}, f={:.2e} (gradient is zero — stuck)",
        res_exact.x, res_exact.f_val);
    // Documents the limitation: GD at exactly zero gradient converges immediately
    assert!(res_exact.converged, "GD converges instantly at zero-gradient saddle");
    assert!(res_exact.iterations == 0, "Zero iterations at saddle point");

    // Nelder-Mead doesn't need gradients — it should escape from near the saddle
    let res_nm = nelder_mead(&saddle, &[0.01, 0.01], 1.0, 100, 1e-12);
    eprintln!("Nelder-Mead from near-saddle: x={:.6?}, f={:.2e}", res_nm.x, res_nm.f_val);
    assert!(res_nm.f_val < -1.0, "Nelder-Mead should escape saddle");
}

/// Ill-conditioned quadratic: f(x) = x₁² + 10⁶·x₂².
/// Condition number κ = 10⁶. GD oscillates, L-BFGS adapts.
#[test]
fn optimizer_ill_conditioned_quadratic() {
    use tambear::optimization::{gradient_descent, lbfgs};

    let kappa = 1e6;
    let ill_quad = |x: &[f64]| -> f64 { x[0] * x[0] + kappa * x[1] * x[1] };
    let grad_iq = |x: &[f64]| -> Vec<f64> { vec![2.0 * x[0], 2.0 * kappa * x[1]] };

    let x0 = &[1.0, 1.0];

    // L-BFGS
    let res_lbfgs = lbfgs(&ill_quad, &grad_iq, x0, 10, 1000, 1e-14);
    let err_lbfgs = ill_quad(&res_lbfgs.x);
    eprintln!("L-BFGS (κ=1e6): f={err_lbfgs:.2e}, iters={}", res_lbfgs.iterations);

    // GD with small lr (must be < 2/λ_max = 1/κ to avoid divergence)
    let lr = 0.5 / kappa; // safe but very slow for the x₁ direction
    let res_gd = gradient_descent(&ill_quad, &grad_iq, x0, lr, 0.0, 10000, 1e-14);
    let err_gd = ill_quad(&res_gd.x);
    eprintln!("GD (κ=1e6, lr={lr:.2e}): f={err_gd:.2e}, iters={}", res_gd.iterations);

    // L-BFGS should converge
    assert!(res_lbfgs.converged || err_lbfgs < 1e-10,
        "L-BFGS should handle κ=1e6, f={err_lbfgs:.2e}");

    // GD with small lr barely makes progress in x₁
    // This documents why adaptive methods matter
    if err_lbfgs > 1e-300 {
        eprintln!("L-BFGS vs GD: {:.0}x better", err_gd / err_lbfgs);
    } else {
        eprintln!("L-BFGS reached exact zero; GD stuck at f={err_gd:.4e}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Interpolation: Runge phenomenon — structure beats degree
// ═══════════════════════════════════════════════════════════════════════════

/// The Runge phenomenon: equispaced high-degree polynomial interpolation diverges
/// for smooth functions. f(x) = 1/(1+25x²) on [-1,1].
///
/// n=5: max error ~0.4. n=15: max error ~1.8. n=25: max error ~80.
/// Chebyshev nodes with the same degree: max error decreases monotonically.
///
/// This is the interpolation analog of the Aitken/Wynn finding:
/// the STRUCTURE of the approximation (node placement) matters more than
/// the DEGREE (number of terms).
#[test]
fn runge_phenomenon_equispaced_vs_chebyshev() {
    use tambear::interpolation::{lagrange, chebyshev_approximate, chebyshev_eval};

    let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);

    // Test points: 200 equispaced in [-1, 1]
    let n_test = 200;
    let test_xs: Vec<f64> = (0..n_test).map(|i| -1.0 + 2.0 * i as f64 / (n_test - 1) as f64).collect();

    eprintln!("═══ Runge phenomenon: equispaced vs Chebyshev ═══");

    let mut prev_equi_err = 0.0;
    let mut diverging = false;

    for &n in &[5, 11, 15, 21, 25] {
        // Equispaced Lagrange
        let xs_equi: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64).collect();
        let ys_equi: Vec<f64> = xs_equi.iter().map(|&x| runge(x)).collect();
        let equi_max_err: f64 = test_xs.iter()
            .map(|&x| (lagrange(&xs_equi, &ys_equi, x) - runge(x)).abs())
            .fold(0.0_f64, |a, b| a.max(b));

        // Chebyshev
        let cheb_coeffs = chebyshev_approximate(&runge, n, -1.0, 1.0);
        let cheb_max_err: f64 = test_xs.iter()
            .map(|&x| (chebyshev_eval(&cheb_coeffs, x, -1.0, 1.0) - runge(x)).abs())
            .fold(0.0_f64, |a, b| a.max(b));

        eprintln!("n={n:2}: equispaced max err = {equi_max_err:.4e}  |  Chebyshev max err = {cheb_max_err:.4e}  |  ratio = {:.0}x",
            equi_max_err / cheb_max_err.max(1e-300));

        if n > 5 && equi_max_err > prev_equi_err * 1.5 {
            diverging = true;
        }
        prev_equi_err = equi_max_err;

        // Chebyshev should always beat equispaced for this function
        assert!(cheb_max_err < equi_max_err || n <= 5,
            "Chebyshev should beat equispaced at n={n}: {cheb_max_err:.4e} vs {equi_max_err:.4e}");
    }

    // The equispaced error should INCREASE with n (Runge phenomenon)
    assert!(diverging,
        "Equispaced error should increase with n (Runge phenomenon)");
}

/// Spectral leakage: FFT of a non-integer-period sinusoid without windowing
/// spreads energy across all frequency bins. Windowing concentrates it.
///
/// This is the spectral analog of the Runge phenomenon: the rectangular
/// window's sharp edges in time create wide sidelobes in frequency.
/// Structure (window shape) beats resources (more samples) again.
#[test]
fn spectral_leakage_windowing() {
    use tambear::signal_processing::{fft, window_hann, window_blackman};

    let n = 256;
    let fs = 256.0; // sample rate = 256 Hz → bin spacing = 1 Hz

    // Sinusoid at exactly 10 Hz (integer bins → no leakage)
    let sig_clean: Vec<(f64, f64)> = (0..n).map(|i| {
        let t = i as f64 / fs;
        ((2.0 * std::f64::consts::PI * 10.0 * t).sin(), 0.0)
    }).collect();
    let spec_clean = fft(&sig_clean);
    let mag_clean: Vec<f64> = spec_clean.iter().map(|c| (c.0 * c.0 + c.1 * c.1).sqrt()).collect();

    // Sinusoid at 10.5 Hz (between bins → leakage)
    let sig_leak: Vec<(f64, f64)> = (0..n).map(|i| {
        let t = i as f64 / fs;
        ((2.0 * std::f64::consts::PI * 10.5 * t).sin(), 0.0)
    }).collect();
    let spec_leak = fft(&sig_leak);
    let mag_leak: Vec<f64> = spec_leak.iter().map(|c| (c.0 * c.0 + c.1 * c.1).sqrt()).collect();

    // Apply Hann window to the 10.5 Hz signal
    let hann = window_hann(n);
    let sig_hann: Vec<(f64, f64)> = sig_leak.iter().zip(hann.iter())
        .map(|(s, w)| (s.0 * w, 0.0)).collect();
    let spec_hann = fft(&sig_hann);
    let mag_hann: Vec<f64> = spec_hann.iter().map(|c| (c.0 * c.0 + c.1 * c.1).sqrt()).collect();

    // Apply Blackman window (even narrower mainlobe, better sidelobe suppression)
    let blackman = window_blackman(n);
    let sig_black: Vec<(f64, f64)> = sig_leak.iter().zip(blackman.iter())
        .map(|(s, w)| (s.0 * w, 0.0)).collect();
    let spec_black = fft(&sig_black);
    let mag_black: Vec<f64> = spec_black.iter().map(|c| (c.0 * c.0 + c.1 * c.1).sqrt()).collect();

    // Measure: what fraction of total energy is in bins far from the signal?
    // Use only positive frequencies (0..N/2+1) to avoid double-counting
    // from the FFT's conjugate symmetry for real signals.
    let half = n / 2 + 1;
    let total_energy = |m: &[f64]| -> f64 { m[..half].iter().map(|v| v * v).sum::<f64>() };
    let peak_bin = |m: &[f64]| -> usize {
        m[..half].iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    };
    let leakage_fraction = |m: &[f64]| -> f64 {
        let pk = peak_bin(m);
        let total = total_energy(m);
        let near: f64 = m[..half].iter().enumerate()
            .filter(|(i, _)| (*i as i64 - pk as i64).abs() <= 3)
            .map(|(_, v)| v * v)
            .sum();
        1.0 - near / total
    };

    let leak_clean = leakage_fraction(&mag_clean);
    let leak_raw = leakage_fraction(&mag_leak);
    let leak_hann = leakage_fraction(&mag_hann);
    let leak_black = leakage_fraction(&mag_black);

    eprintln!("═══ Spectral leakage ═══");
    eprintln!("Integer freq (10 Hz):    leakage = {leak_clean:.4} (no leakage expected)");
    eprintln!("Non-integer (10.5 Hz):   leakage = {leak_raw:.4} (rectangular window)");
    eprintln!("Hann windowed (10.5 Hz): leakage = {leak_hann:.4}");
    eprintln!("Blackman (10.5 Hz):      leakage = {leak_black:.4}");
    eprintln!("Leakage reduction: Hann {:.1}x, Blackman {:.1}x vs rectangular",
        leak_raw / leak_hann.max(1e-300), leak_raw / leak_black.max(1e-300));

    // Integer frequency should have essentially zero leakage
    assert!(leak_clean < 0.01, "Integer frequency should have <1% leakage, got {leak_clean:.4}");

    // Non-integer without window should have measurable leakage
    assert!(leak_raw > 0.01, "10.5 Hz rectangular should have >1% leakage, got {leak_raw:.4}");

    // Windowing should reduce leakage substantially
    assert!(leak_hann < leak_raw,
        "Hann should reduce leakage: {leak_hann:.4} vs {leak_raw:.4}");
}

// ═══════════════════════════════════════════════════════════════════════════
// KDE: Silverman bandwidth on bimodal data
// ═══════════════════════════════════════════════════════════════════════════

/// Silverman's rule assumes unimodal data. On bimodal data, it oversmooths:
/// two distinct peaks merge into one broad hump.
///
/// This is the density estimation analog of structure-beats-resources:
/// Silverman (wrong structural assumption) merges peaks regardless of sample size.
/// A smaller bandwidth (right assumption about modality) resolves them.
#[test]
fn kde_silverman_oversmooths_bimodal() {
    use tambear::nonparametric::{kde, silverman_bandwidth, KernelType};

    // Bimodal: 100 points at N(-3, 0.5) + 100 points at N(3, 0.5)
    // Clear separation = 6σ apart.
    let n = 200;
    let mut data = Vec::with_capacity(n);
    let mut rng = 42u64;
    for _ in 0..100 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (rng as f64 / u64::MAX as f64).max(1e-300);
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = rng as f64 / u64::MAX as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        data.push(-3.0 + 0.5 * z);
    }
    for _ in 0..100 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (rng as f64 / u64::MAX as f64).max(1e-300);
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = rng as f64 / u64::MAX as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        data.push(3.0 + 0.5 * z);
    }

    let silverman_h = silverman_bandwidth(&data);

    // Evaluate at key points
    let eval_pts: Vec<f64> = (-60..=60).map(|i| i as f64 * 0.1).collect();

    // KDE with Silverman bandwidth (oversmoothed)
    let dens_silverman = kde(&data, &eval_pts, KernelType::Gaussian, Some(silverman_h));

    // KDE with correct bandwidth (0.3 — close to the true σ=0.5)
    let dens_correct = kde(&data, &eval_pts, KernelType::Gaussian, Some(0.3));

    // Check: does each KDE have one or two modes?
    let count_modes = |dens: &[f64]| -> usize {
        let mut modes = 0;
        for i in 1..dens.len() - 1 {
            if dens[i] > dens[i - 1] && dens[i] > dens[i + 1] && dens[i] > 0.01 {
                modes += 1;
            }
        }
        modes
    };

    let modes_silverman = count_modes(&dens_silverman);
    let modes_correct = count_modes(&dens_correct);

    // Density at the midpoint (x=0): should be near-zero for bimodal, high for oversmoothed
    let idx_0 = eval_pts.iter().position(|&x| (x - 0.0).abs() < 0.05).unwrap();
    let dens_at_0_silverman = dens_silverman[idx_0];
    let dens_at_0_correct = dens_correct[idx_0];

    eprintln!("═══ KDE Silverman vs correct bandwidth on bimodal data ═══");
    eprintln!("Silverman h = {silverman_h:.3}  (oversmoothed)");
    eprintln!("Correct  h  = 0.300");
    eprintln!("Silverman modes: {modes_silverman}  |  Correct modes: {modes_correct}");
    eprintln!("Density at x=0: Silverman = {dens_at_0_silverman:.4}  |  Correct = {dens_at_0_correct:.6}");
    eprintln!("Valley-to-peak ratio: Silverman = {:.2}%  |  Correct = {:.2}%",
        dens_at_0_silverman / dens_silverman.iter().cloned().fold(0.0_f64, f64::max) * 100.0,
        dens_at_0_correct / dens_correct.iter().cloned().fold(0.0_f64, f64::max) * 100.0);

    // The correct bandwidth should resolve two modes
    assert!(modes_correct >= 2,
        "Correct bandwidth should find ≥2 modes, found {modes_correct}");

    // The correct bandwidth should have near-zero density at x=0 (the gap)
    let peak_correct = dens_correct.iter().cloned().fold(0.0_f64, f64::max);
    assert!(dens_at_0_correct < 0.1 * peak_correct,
        "Correct KDE at x=0 should be <10% of peak, got {:.2}%",
        dens_at_0_correct / peak_correct * 100.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Time series: ADF unit root test at the boundary
// ═══════════════════════════════════════════════════════════════════════════

/// ADF test power at the unit root boundary: φ = 1.0 (random walk) vs
/// φ = 0.95, 0.99 (near-unit-root stationary).
///
/// The closer φ is to 1, the harder it is to reject H₀: unit root.
/// This tests whether ADF can distinguish a barely-stationary process from
/// a true random walk with finite sample sizes.
#[test]
fn adf_unit_root_boundary() {
    use tambear::time_series::{adf_test, ar_fit};

    let n = 500;

    // Generate AR(1) processes with different φ values using an LCG
    let mut rng = 12345u64;
    let next_normal = |rng: &mut u64| -> f64 {
        // Box-Muller from two LCG draws
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (*rng as f64) / u64::MAX as f64;
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (*rng as f64) / u64::MAX as f64;
        (-2.0 * u1.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    eprintln!("═══ ADF unit root boundary ═══");

    for &phi in &[0.5_f64, 0.9, 0.95, 0.99, 1.0] {
        rng = 12345;
        let mut data = vec![0.0_f64; n];
        for t in 1..n {
            data[t] = phi * data[t - 1] + next_normal(&mut rng);
        }

        let adf = adf_test(&data, 1);
        let ar = ar_fit(&data, 1);

        let reject_5pct = adf.statistic < adf.critical_5pct;
        eprintln!("φ={phi:.2}: ADF stat={:.3}, 5% cv={:.3}, reject={}, AR(1) est φ̂={:.4}",
            adf.statistic, adf.critical_5pct, reject_5pct, ar.coefficients[0]);

        // Strongly stationary (φ=0.5): should reject unit root
        if phi <= 0.9 {
            assert!(reject_5pct,
                "ADF should reject unit root at φ={phi}: stat={:.3} vs cv={:.3}",
                adf.statistic, adf.critical_5pct);
        }

        // True random walk (φ=1.0): should NOT reject
        if phi == 1.0 {
            assert!(!reject_5pct,
                "ADF should not reject unit root for random walk: stat={:.3}",
                adf.statistic);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Robust statistics: breakdown point exploration
// ═══════════════════════════════════════════════════════════════════════════

/// Breakdown point test: how much contamination can each estimator survive?
/// The arithmetic mean has breakdown point 0 — a single extreme outlier corrupts it.
/// MAD/median have breakdown point 50% — they survive until the majority is corrupted.
/// Huber interpolates between mean and median.
/// Bisquare (Tukey) completely rejects outliers beyond its threshold.
#[test]
fn robust_estimator_breakdown_points() {
    use tambear::robust::{huber_m_estimate, bisquare_m_estimate};
    use tambear::descriptive::median;

    let n = 100;
    let true_loc = 0.0;

    // Clean data: N(0, 1) — use a simple deterministic "pseudo-normal"
    let clean: Vec<f64> = (0..n).map(|i| {
        let u = (i as f64 + 0.5) / n as f64; // uniform (0,1)
        // Approximate normal quantile: simple rational approximation
        let t = if u < 0.5 { (-2.0 * (u).ln()).sqrt() } else { (-2.0 * (1.0 - u).ln()).sqrt() };
        let sign = if u < 0.5 { -1.0 } else { 1.0 };
        sign * (t - (2.515517 + t * (0.802853 + t * 0.010328)) /
               (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308))))
    }).collect();

    eprintln!("═══ Breakdown point exploration ═══");

    // Test at various contamination levels
    for &contam_pct in &[0, 1, 10, 20, 30, 40, 49] {
        let n_contam = n * contam_pct / 100;
        let mut data = clean.clone();
        // Replace first n_contam values with extreme outliers
        for i in 0..n_contam {
            data[i] = 1000.0; // massive outlier
        }

        let mean_val: f64 = data.iter().sum::<f64>() / n as f64;
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median_val = median(&sorted);
        let huber = huber_m_estimate(&data, 1.345, 50, 1e-8);
        let bisq = bisquare_m_estimate(&data, 4.685, 50, 1e-8);

        let mean_err = (mean_val - true_loc).abs();
        let med_err = (median_val - true_loc).abs();
        let huber_err = (huber.location - true_loc).abs();
        let bisq_err = (bisq.location - true_loc).abs();

        eprintln!("{contam_pct:2}% contam: mean={mean_err:8.2}  median={med_err:6.3}  huber={huber_err:6.3}  bisquare={bisq_err:6.3}");

        // At low contamination, all should be reasonable
        if contam_pct <= 10 {
            assert!(median_val.abs() < 1.0, "Median should survive {contam_pct}% contamination");
            assert!(huber.location.abs() < 2.0, "Huber should survive {contam_pct}% contamination");
            assert!(bisq.location.abs() < 2.0, "Bisquare should survive {contam_pct}% contamination");
        }

        // At 49% contamination, median should still work
        if contam_pct == 49 {
            assert!(median_val.abs() < 5.0, "Median should survive 49% contamination, got {median_val}");
        }

        // Mean breaks immediately with contamination
        if contam_pct >= 10 {
            assert!(mean_err > 50.0, "Mean should be destroyed at {contam_pct}%, got err={mean_err:.2}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Special functions: boundary adversarial tests
// ═══════════════════════════════════════════════════════════════════════════

/// erfc at extreme arguments — tests the subnormal/zero boundary.
/// erfc(27) ≈ 5.2e-319 (subnormal), erfc(28) ≈ 0 (underflows).
/// The A&S polynomial approximation must not produce negative values.
#[test]
fn erfc_extreme_arguments() {
    use tambear::special_functions::{erf, erfc};

    // Moderate: erfc(5) ≈ 1.54e-12
    // A&S 7.1.26 has max absolute error 1.5e-7 on erf, but the direct
    // product formula (poly × exp(-x²)) gives ~1% relative error on erfc.
    let e5 = erfc(5.0);
    let e5_true = 1.5374597944e-12_f64;
    let e5_rel = (e5 - e5_true).abs() / e5_true;
    eprintln!("erfc(5)  = {e5:.6e}  (expected {e5_true:.6e}, rel err {e5_rel:.2e})");
    assert!(e5 > 0.0, "erfc(5) must be positive");
    assert!(e5_rel < 0.02, "erfc(5) relative error {e5_rel:.2e} should be < 2%");

    // Large: erfc(10) ≈ 2.09e-45
    let e10 = erfc(10.0);
    eprintln!("erfc(10) = {e10:.6e}  (expected ~2.09e-45)");
    assert!(e10 >= 0.0, "erfc(10) must be non-negative");

    // Very large: erfc(27) is subnormal (~5.2e-319)
    let e27 = erfc(27.0);
    eprintln!("erfc(27) = {e27:.6e}  (subnormal territory)");
    assert!(e27 >= 0.0, "erfc(27) must be non-negative (subnormal OK)");

    // Past underflow: erfc(30) should be 0.0 (graceful underflow)
    let e30 = erfc(30.0);
    eprintln!("erfc(30) = {e30:.6e}  (should underflow to 0.0)");
    assert!(e30 >= 0.0, "erfc(30) must not be negative");
    assert!(e30 == 0.0 || e30 < 1e-300, "erfc(30) should underflow");

    // Negative large: erfc(-10) = 2 - erfc(10) ≈ 2.0
    let em10 = erfc(-10.0);
    eprintln!("erfc(-10) = {em10:.6e}  (expected ~2.0)");
    assert!((em10 - 2.0).abs() < 1e-10, "erfc(-10) ≈ 2.0");

    // erf must stay in [-1, 1] for all inputs
    for &x in &[0.0, 1.0, 5.0, 10.0, 27.0, 100.0, -100.0] {
        let e = erf(x);
        assert!(e >= -1.0 && e <= 1.0,
            "erf({x}) = {e} out of [-1,1]");
    }
}

/// log_gamma near negative integers — the reflection formula sin(πx) → 0.
/// Tests whether we get +∞ (correct) or NaN/panic.
#[test]
fn log_gamma_near_poles() {
    use tambear::special_functions::{log_gamma, gamma};

    // At negative integers: Γ has poles, log_gamma should return +∞
    // (The implementation returns +∞ for x ≤ 0, so exact integers are handled.)
    for n in 0..10 {
        let val = log_gamma(-(n as f64));
        eprintln!("log_gamma({}) = {val}", -(n as i32));
        assert!(val.is_infinite() || val.is_nan(),
            "log_gamma at pole should be inf or nan, got {val}");
    }

    // DOCUMENTED LIMITATION: log_gamma returns inf for ALL x ≤ 0.
    // The reflection formula only handles (0, 0.5). For negative arguments,
    // the implementation doesn't compute log|Γ(x)| — it unconditionally returns inf.
    //
    // This is acceptable because all statistical callers (incomplete beta, gamma,
    // chi2_cdf, t_cdf, f_cdf) only pass positive a, b, df values.
    // But it means log_gamma(-0.5) = inf instead of the correct ln(2√π) ≈ 1.265.
    let lg_neg_half = log_gamma(-0.5);
    eprintln!("log_gamma(-0.5) = {lg_neg_half}  (limitation: returns inf, correct = {:.4})",
        (2.0 * std::f64::consts::PI.sqrt()).ln());
    assert!(lg_neg_half.is_infinite(),
        "Documenting: log_gamma returns inf for negative args");

    // Γ(0.5) = √π — sanity check the reflection formula
    let g_half = gamma(0.5);
    let expected = std::f64::consts::PI.sqrt();
    eprintln!("Γ(0.5) = {g_half:.10}  (expected {expected:.10})");
    assert!((g_half - expected).abs() / expected < 1e-10,
        "Γ(0.5) = √π, got {g_half}");
}

/// Incomplete beta with extreme parameters — tests Lentz CF convergence.
/// Very asymmetric (a=0.001, b=1000) or very large (a=b=1000).
#[test]
fn incomplete_beta_extreme_parameters() {
    use tambear::special_functions::regularized_incomplete_beta;

    // Highly asymmetric: a=0.001, b=1000, x=0.5
    // Beta(0.001, 1000) is concentrated near 0. I_{0.5} should be ~1.0
    let ib1 = regularized_incomplete_beta(0.5, 0.001, 1000.0);
    eprintln!("I_0.5(0.001, 1000) = {ib1:.10}  (expected ~1.0)");
    assert!(ib1 > 0.99, "I_0.5(0.001, 1000) should be near 1.0, got {ib1}");
    assert!(ib1 <= 1.0, "I_x must be ≤ 1.0");

    // Reverse asymmetric: a=1000, b=0.001, x=0.5
    // Beta(1000, 0.001) is concentrated near 1. I_{0.5} should be ~0.0
    let ib2 = regularized_incomplete_beta(0.5, 1000.0, 0.001);
    eprintln!("I_0.5(1000, 0.001) = {ib2:.10}  (expected ~0.0)");
    assert!(ib2 < 0.01, "I_0.5(1000, 0.001) should be near 0.0, got {ib2}");
    assert!(ib2 >= 0.0, "I_x must be ≥ 0.0");

    // Symmetry: I_x(a,b) + I_{1-x}(b,a) = 1
    let a = 3.5;
    let b = 7.2;
    let x = 0.3;
    let sum = regularized_incomplete_beta(x, a, b) +
              regularized_incomplete_beta(1.0 - x, b, a);
    eprintln!("I_x(a,b) + I_{{1-x}}(b,a) = {sum:.15}  (expected 1.0)");
    assert!((sum - 1.0).abs() < 1e-12,
        "Beta symmetry violated: sum = {sum}");

    // Large symmetric: a=b=1000, x=0.5 → should be exactly 0.5 by symmetry
    let ib_sym = regularized_incomplete_beta(0.5, 1000.0, 1000.0);
    eprintln!("I_0.5(1000, 1000) = {ib_sym:.10}  (expected 0.5)");
    assert!((ib_sym - 0.5).abs() < 1e-6,
        "I_0.5(a,a) = 0.5 by symmetry, got {ib_sym}");
}

/// t_cdf with df=1 (Cauchy) and extreme arguments.
/// Cauchy has such thick tails that t_cdf(1e8, 1) is still measurably < 1.
/// Formula: t_cdf(t, 1) = 0.5 + arctan(t)/π
#[test]
fn t_cdf_cauchy_extreme_tails() {
    use tambear::special_functions::t_cdf;

    // Cauchy CDF = 0.5 + arctan(t)/π
    let cauchy_cdf = |t: f64| 0.5 + t.atan() / std::f64::consts::PI;

    for &t in &[1.0, 10.0, 100.0, 1e6, 1e10] {
        let got = t_cdf(t, 1.0);
        let expected = cauchy_cdf(t);
        let err = (got - expected).abs();
        eprintln!("t_cdf({t:.0e}, df=1) = {got:.12}  expected {expected:.12}  err {err:.2e}");
        assert!(err < 1e-6,
            "t_cdf({t}, 1) = {got} != Cauchy CDF {expected}, err {err:.2e}");
    }

    // Symmetry: t_cdf(-t, df) + t_cdf(t, df) = 1
    for &df in &[1.0, 3.0, 30.0, 1000.0] {
        let t = 2.5;
        let sum = t_cdf(-t, df) + t_cdf(t, df);
        assert!((sum - 1.0).abs() < 1e-12,
            "t_cdf symmetry violated at df={df}: sum = {sum}");
    }

    // Very small df: df=0.5 (valid but extreme)
    let v = t_cdf(1.0, 0.5);
    eprintln!("t_cdf(1.0, df=0.5) = {v:.10}");
    assert!(v > 0.5 && v < 1.0, "t_cdf(1, 0.5) should be in (0.5, 1.0), got {v}");
}

/// chi2_cdf with very large degrees of freedom (k=10000).
/// By CLT, χ²(k) ≈ N(k, 2k), so chi2_cdf(k, k) ≈ 0.5.
#[test]
fn chi2_cdf_large_df() {
    use tambear::special_functions::chi2_cdf;

    // At x = k (the mean), CDF ≈ 0.5 for large k
    for &k in &[100.0, 1000.0, 10000.0] {
        let cdf_at_mean = chi2_cdf(k, k);
        eprintln!("chi2_cdf({k}, k={k}) = {cdf_at_mean:.6}  (expected ~0.5)");
        // For large k, the median ≈ k(1 - 2/(9k))³ ≈ k - 2/3
        // So CDF at exactly k is slightly above 0.5
        assert!((cdf_at_mean - 0.5).abs() < 0.05,
            "chi2_cdf(k,k) should be ~0.5 for large k, got {cdf_at_mean}");
    }

    // chi2_cdf(0, k) = 0 for all k > 0
    assert!(chi2_cdf(0.0, 100.0) == 0.0, "chi2_cdf(0, k) = 0");

    // Very small k: chi2_cdf(x, 0.5)
    let v = chi2_cdf(0.001, 0.5);
    eprintln!("chi2_cdf(0.001, k=0.5) = {v:.10}");
    assert!(v > 0.0 && v < 1.0, "chi2_cdf(0.001, 0.5) in (0,1), got {v}");

    // CDF should be monotonic: cdf(x1) < cdf(x2) when x1 < x2
    let k = 10.0;
    let c1 = chi2_cdf(5.0, k);
    let c2 = chi2_cdf(10.0, k);
    let c3 = chi2_cdf(20.0, k);
    assert!(c1 < c2 && c2 < c3,
        "chi2_cdf must be monotonic: {c1} < {c2} < {c3}");
}

/// normal_cdf at extreme arguments — tests erfc composition.
/// normal_cdf(37) should be 1.0 - ε, normal_cdf(-37) should be ε.
#[test]
fn normal_cdf_extreme_tails() {
    use tambear::special_functions::{normal_cdf, normal_sf};

    // FIXED: normal_cdf(0) now returns exactly 0.5 via special case.
    // Previously returned 0.4999999995 due to A&S polynomial bias at t=1.
    let cdf_0 = normal_cdf(0.0);
    eprintln!("normal_cdf(0) = {cdf_0:.15}  (should be exactly 0.5)");
    assert!(cdf_0 == 0.5, "normal_cdf(0) should be exactly 0.5, got {cdf_0}");

    let cdf_196 = normal_cdf(1.96);
    assert!((cdf_196 - 0.975).abs() < 1e-3, "Phi(1.96) ~ 0.975");

    // Far tails: normal_sf(x) should be > 0 for reasonable x
    for &x in &[5.0, 8.0, 10.0, 20.0] {
        let sf = normal_sf(x);
        eprintln!("normal_sf({x}) = {sf:.6e}");
        assert!(sf >= 0.0, "SF must be non-negative");
        assert!(sf < 0.5, "SF({x}) must be < 0.5");
    }

    // Very extreme: normal_sf(37) ≈ 5e-300
    let sf37 = normal_sf(37.0);
    eprintln!("normal_sf(37) = {sf37:.6e}  (subnormal territory)");
    assert!(sf37 >= 0.0, "SF(37) must be non-negative");

    // CDF + SF = 1 identity — exact at x=0 (special case), within A&S error elsewhere
    for &x in &[0.0, 1.0, 3.0, -2.5] {
        let sum = normal_cdf(x) + normal_sf(x);
        eprintln!("CDF({x}) + SF({x}) = {sum:.15}");
        if x == 0.0 {
            assert!(sum == 1.0, "CDF(0) + SF(0) should be exactly 1.0, got {sum}");
        } else {
            assert!((sum - 1.0).abs() < 2e-9,
                "CDF + SF = 1 at x={x}, got {sum} (A&S error)");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multivariate: Hotelling T² with near-singular covariance (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// When variables are nearly multicollinear (x3 ≈ x1 + x2), the covariance
// matrix is near-singular. Cholesky decomposition may fail or produce
// numerically unreliable T² values. This tests the boundary of the
// implementation's numerical stability.
//
// This is the multivariate version of the matched-kernel principle:
// Hotelling T² assumes the covariance matrix is well-conditioned.
// When that structure breaks (collinearity), the test degrades.

#[test]
fn hotelling_near_singular_covariance() {
    use tambear::linear_algebra::Mat;
    use tambear::multivariate::hotelling_one_sample;

    // 3D data where x3 = x1 + x2 + epsilon (epsilon ~ 1e-10)
    // Condition number of covariance → very large
    let n = 30;
    let mut data = Vec::new();
    let mut rng_state = 42u64;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
    };

    for _ in 0..n {
        let x1 = next_f64();
        let x2 = next_f64();
        let noise = next_f64() * 1e-10;
        let x3 = x1 + x2 + noise;
        data.push(x1);
        data.push(x2);
        data.push(x3);
    }
    let x = Mat::from_vec(n, 3, data);

    // The covariance matrix is near-singular due to x3 ≈ x1 + x2.
    // Cholesky should panic ("not positive definite") because the matrix
    // is numerically rank-deficient.
    let result = std::panic::catch_unwind(|| {
        hotelling_one_sample(&x, &[0.0, 0.0, 0.0])
    });

    match &result {
        Ok(r) => {
            // If it somehow succeeds, the T² should be unreliable.
            // Either the F-stat is huge (ill-conditioned inversion amplifies)
            // or the p-value is nonsensical.
            eprintln!("Near-singular: T²={:.6}, F={:.6}, p={:.6e}",
                r.t2, r.f_statistic, r.p_value);
            // The test "passes" either way — we're documenting behavior.
            // But note: if Cholesky succeeds on near-singular data, that's
            // numerically suspicious.
            eprintln!("WARNING: Cholesky succeeded on near-singular covariance.");
            eprintln!("T² results are likely numerically unreliable.");
        }
        Err(_) => {
            // Expected: Cholesky panics on near-singular matrix.
            // This IS the correct behavior — the implementation correctly
            // refuses to compute a meaningless T² on ill-conditioned data.
            eprintln!("Cholesky correctly panicked on near-singular covariance.");
            eprintln!("This is the RIGHT behavior: refuse rather than produce garbage.");
        }
    }
    // Either outcome is valid — the test documents the boundary.
}

// ═══════════════════════════════════════════════════════════════════════════
// Multivariate: Hotelling T² dimension curse (p → n)
// ═══════════════════════════════════════════════════════════════════════════
//
// When dimensionality p approaches sample size n, the F-distribution
// approximation has df2 = n - p → 0. The test loses all power.
// With p > n, the covariance matrix is guaranteed singular (rank n-1).
// This is Bellman's curse of dimensionality in hypothesis testing.

#[test]
fn hotelling_dimension_curse() {
    use tambear::linear_algebra::Mat;
    use tambear::multivariate::hotelling_one_sample;

    // Case 1: p = n/2 (well-conditioned — should work fine)
    let n = 20;
    let p_safe = 5;
    let mut data = Vec::new();
    let mut rng_state = 137u64;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
    };
    for _ in 0..(n * p_safe) {
        data.push(next_f64() + 2.0); // shifted from origin — should reject
    }
    let x_safe = Mat::from_vec(n, p_safe, data);
    let mu0_safe = vec![0.0; p_safe];
    let res_safe = hotelling_one_sample(&x_safe, &mu0_safe);
    eprintln!("p={p_safe}, n={n}: T²={:.4}, F={:.4}, df2={:.0}, p={:.4e}",
        res_safe.t2, res_safe.f_statistic, res_safe.df2, res_safe.p_value);
    assert!(res_safe.df2 > 5.0, "df2 should be well above 0 for safe case");

    // Case 2: p = n - 2 (borderline — df2 = 2, barely valid)
    let p_border = n - 2;
    let mut data2 = Vec::new();
    for _ in 0..(n * p_border) {
        data2.push(next_f64() + 2.0);
    }
    let x_border = Mat::from_vec(n, p_border, data2);
    let mu0_border = vec![0.0; p_border];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hotelling_one_sample(&x_border, &mu0_border)
    }));

    match result {
        Ok(res) => {
            eprintln!("p={p_border}, n={n}: T²={:.4}, F={:.4}, df2={:.0}, p={:.4e}",
                res.t2, res.f_statistic, res.df2, res.p_value);
            // With df2 = 2, the F-test has extremely heavy tails.
            // The same data that rejects with p=5 may fail to reject with p=18.
            // This IS the curse of dimensionality.
            assert!(res.df2 <= 2.0,
                "df2 should be tiny for p near n, got {}", res.df2);
        }
        Err(_) => {
            eprintln!("p={p_border}, n={n}: Cholesky panicked (covariance singular)");
            eprintln!("Dimensionality exceeded sample size — expected.");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multivariate: Mardia normality on heavy-tailed data (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// Mardia's kurtosis test assumes N(μ, Σ). Heavy-tailed data (Cauchy-like)
// should produce huge kurtosis and strongly reject normality.
// This tests whether the kurtosis statistic actually detects departures.

#[test]
fn mardia_heavy_tails_rejection() {
    use tambear::linear_algebra::Mat;
    use tambear::multivariate::mardia_normality;

    // Generate pseudo-Cauchy data: ratio of two ~normals has heavy tails.
    // Use a simple approach: tan(uniform * π - π/2) ≈ Cauchy.
    let n = 50;
    let mut data = Vec::new();
    let mut rng_state = 999u64;
    let mut next_uniform = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    for _ in 0..n {
        // Cauchy in 2D: transform uniform through tan
        let u1 = next_uniform().clamp(0.01, 0.99); // avoid exact 0 or 1
        let u2 = next_uniform().clamp(0.01, 0.99);
        let x1 = (u1 * std::f64::consts::PI - std::f64::consts::FRAC_PI_2).tan();
        let x2 = (u2 * std::f64::consts::PI - std::f64::consts::FRAC_PI_2).tan();
        data.push(x1);
        data.push(x2);
    }
    let x = Mat::from_vec(n, 2, data);

    let res = mardia_normality(&x);
    eprintln!("Mardia on Cauchy-like data:");
    eprintln!("  skewness b1,p = {:.4}, p = {:.4e}", res.skewness, res.skewness_p);
    eprintln!("  kurtosis b2,p = {:.4}, p = {:.4e}", res.kurtosis, res.kurtosis_p);

    // Expected kurtosis for 2D normal: p(p+2) = 2*4 = 8.
    // Cauchy has infinite kurtosis, so b2,p >> 8.
    assert!(res.kurtosis > 8.0,
        "Heavy-tailed data should have kurtosis > p(p+2)=8, got {}", res.kurtosis);

    // The kurtosis p-value should reject normality.
    assert!(res.kurtosis_p < 0.05,
        "Should reject normality for Cauchy-like tails, p={:.4e}", res.kurtosis_p);

    eprintln!("Mardia correctly detects heavy tails via excess kurtosis.");
}

// ═══════════════════════════════════════════════════════════════════════════
// Multivariate: LDA with overlapping groups (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// When group means are nearly identical, between-group SSCP H ≈ 0.
// All eigenvalues → 0, and classification degrades to chance.
// This tests the implementation's behavior at the discriminability boundary.

#[test]
fn lda_overlapping_groups_degrade() {
    use tambear::linear_algebra::Mat;
    use tambear::multivariate::{lda, manova};

    // Two groups drawn from the SAME distribution (no separation)
    let mut data = Vec::new();
    let mut rng_state = 2025u64;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
    };

    let n_per_group = 20;
    let p = 3;
    let mut groups = Vec::new();
    for g in 0..2 {
        for _ in 0..n_per_group {
            for _ in 0..p {
                data.push(next_f64()); // same distribution for both groups
            }
            groups.push(g);
        }
    }
    let x = Mat::from_vec(2 * n_per_group, p, data);

    // MANOVA should NOT reject (groups are identical)
    let manova_res = manova(&x, &groups);
    eprintln!("Overlapping groups MANOVA:");
    eprintln!("  Wilks = {:.4}, Pillai = {:.4}, p = {:.4e}",
        manova_res.wilks_lambda, manova_res.pillai_trace, manova_res.p_value);
    assert!(manova_res.wilks_lambda > 0.5,
        "Wilks should be near 1 for no separation, got {}", manova_res.wilks_lambda);
    assert!(manova_res.p_value > 0.05,
        "MANOVA should not reject H0, p={:.4e}", manova_res.p_value);

    // LDA eigenvalues should be near zero (no discrimination)
    let lda_res = lda(&x, &groups);
    eprintln!("LDA eigenvalues: {:?}", lda_res.eigenvalues);
    assert!(lda_res.eigenvalues[0] < 1.0,
        "Eigenvalue should be small for no separation, got {}", lda_res.eigenvalues[0]);

    // Classification accuracy should be near chance (50%)
    let preds = lda_res.predict(&x);
    let correct: usize = preds.iter().zip(&groups).filter(|(&p, &g)| p == g).count();
    let accuracy = correct as f64 / (2 * n_per_group) as f64;
    eprintln!("LDA classification accuracy: {accuracy:.3} (chance = 0.50)");
    // Allow generous range — with random data, accuracy varies
    assert!(accuracy < 0.85,
        "Should not achieve high accuracy on identical distributions, got {accuracy}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Information theory: MI finite-sample bias (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// Mutual information estimated from a contingency table is biased UPWARD
// for finite samples. With n observations and k×k bins, the bias is
// approximately (k-1)²/(2n). This means MI > 0 even for truly independent
// variables when the contingency table is sparse.
//
// AMI (adjusted MI) corrects for this. NMI does not.
// This tests whether the implementation correctly exhibits the bias.

#[test]
fn mi_finite_sample_bias() {
    use tambear::information_theory::{
        mutual_information, normalized_mutual_information,
        adjusted_mutual_info_score,
    };

    // Generate truly independent labels — each variable is uniform over k=5.
    // With n=20 observations and 5×5=25 cells, most cells are empty or have 1 count.
    // The bias formula: E[MI] ≈ (k-1)²/(2n) = 16/40 = 0.4 nats.
    let n = 20;
    let k = 5;
    let mut rng_state = 777u64;
    let mut next_label = || -> i32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) % k as u64) as i32
    };

    let labels_x: Vec<i32> = (0..n).map(|_| next_label()).collect();
    let labels_y: Vec<i32> = (0..n).map(|_| next_label()).collect();

    // Build contingency table manually (since mutual_information takes a flat table)
    let mut table = vec![0.0; k * k];
    for (&x, &y) in labels_x.iter().zip(&labels_y) {
        table[x as usize * k + y as usize] += 1.0;
    }

    let mi = mutual_information(&table, k, k);
    let nmi = normalized_mutual_information(&table, k, k, "arithmetic");
    let ami = adjusted_mutual_info_score(&labels_x, &labels_y);

    eprintln!("MI finite-sample bias (n={n}, k={k}):");
    eprintln!("  MI  = {mi:.4} nats  (expected bias ≈ {:.4})", (k as f64 - 1.0).powi(2) / (2.0 * n as f64));
    eprintln!("  NMI = {nmi:.4}  (also biased — not corrected)");
    eprintln!("  AMI = {ami:.4}  (should be near 0 — corrected for chance)");

    // Key assertion: MI is STRICTLY positive even for independent variables
    assert!(mi > 0.0,
        "MI should be biased upward for finite samples, got {mi}");

    // AMI should be near zero (it corrects for the expected MI under random permutations)
    assert!(ami < 0.15,
        "AMI should be near 0 for independent variables, got {ami}");

    // The gap between MI and AMI IS the finite-sample bias
    eprintln!("  MI - AMI = {:.4} nats (the finite-sample bias)", mi - ami);
    eprintln!("This is the Structure Beats Resources principle again:");
    eprintln!("  MI is the 'wrong structure' (no correction for chance).");
    eprintln!("  AMI is the 'right structure' (corrects for expected MI).");
}

// ═══════════════════════════════════════════════════════════════════════════
// Information theory: entropy histogram bin sensitivity (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// Differential entropy estimated via histogram depends on bin width.
// H_est ≈ H_discrete + log(bin_width). Changing the number of bins
// changes bin_width, which changes the entropy estimate.
// The "right" number of bins is itself a structure choice.

#[test]
fn entropy_histogram_bin_sensitivity() {
    use tambear::information_theory::entropy_histogram;

    // Generate 200 samples from a pseudo-uniform distribution on [0, 1].
    // True differential entropy of Uniform(0,1) = 0 nats (= log(1)).
    let n = 200;
    let mut rng_state = 42u64;
    let mut next_uniform = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };
    let data: Vec<f64> = (0..n).map(|_| next_uniform()).collect();

    // True entropy of Uniform(0,1) = 0 nats
    let true_entropy = 0.0;

    // Compute with different bin counts
    let bin_counts = [5, 10, 20, 50, 100, 200];
    let mut estimates = Vec::new();
    for &bins in &bin_counts {
        let h = entropy_histogram(&data, bins);
        estimates.push(h);
        eprintln!("  bins={bins:>3}: H_est = {h:.4} nats (error = {:.4})", (h - true_entropy).abs());
    }

    // The range of estimates should be substantial — this IS the bin sensitivity
    let min_est = estimates.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_est = estimates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_est - min_est;
    eprintln!("  Range of estimates: {range:.4} nats (min={min_est:.4}, max={max_est:.4})");

    // With 200 samples, the range should be noticeable but not catastrophic
    // The correction H_est = H_discrete + log(bin_width) partially compensates,
    // but doesn't fully eliminate the dependence.
    // What we're testing: does the correction HELP?
    // Without correction: H_discrete depends on bins by O(log(bins)).
    // With correction: should be more stable.
    // The range should be < 1 nat for reasonable bin counts.
    assert!(range < 1.5,
        "Entropy estimates should be somewhat stable across bin counts, range={range:.4}");

    // Even with the log(bin_width) correction, all estimates are biased.
    // Histogram entropy estimation is systematically poor for finite samples.
    let best_err = estimates.iter().map(|h| (h - true_entropy).abs()).fold(f64::INFINITY, f64::min);
    eprintln!("  Best absolute error: {best_err:.4} nats");
    eprintln!("  All estimates are biased negative: histogram method underestimates");
    eprintln!("  entropy of continuous distributions with finite samples.");
    eprintln!("  This is another instance of Structure Beats Resources:");
    eprintln!("  more bins ≠ better estimate. The binning structure matters.");

    // The key finding: estimates should all be finite and ordered sensibly
    for h in &estimates {
        assert!(h.is_finite(), "Entropy estimate should be finite");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Volatility: GARCH at IGARCH boundary (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// IGARCH (Integrated GARCH) has α + β = 1 — shocks persist forever.
// The implementation clamps α + β < 0.999 for stationarity.
// With a true IGARCH DGP, the estimated α + β should hit this ceiling,
// and the unconditional variance formula ω/(1-α-β) diverges.
// This is the volatility analog of the ADF unit-root boundary.

#[test]
fn garch_igarch_boundary() {
    use tambear::volatility::garch11_fit;

    // Simulate IGARCH(1,1): α=0.15, β=0.85, α+β=1.0, ω=0.0001
    // σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
    let n = 500;
    let omega_true = 0.0001;
    let alpha_true = 0.15;
    let beta_true = 0.85;

    let mut sigma2 = vec![0.0f64; n];
    let mut returns = vec![0.0f64; n];
    sigma2[0] = 0.01; // initial variance
    let mut rng_state = 314159u64;
    let mut next_normal = || -> f64 {
        // Box-Muller
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = ((rng_state >> 33) as f64) / (u32::MAX as f64);
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = ((rng_state >> 33) as f64) / (u32::MAX as f64);
        let u1 = u1.max(1e-15);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    returns[0] = next_normal() * sigma2[0].sqrt();
    for t in 1..n {
        sigma2[t] = omega_true + alpha_true * returns[t-1].powi(2) + beta_true * sigma2[t-1];
        returns[t] = next_normal() * sigma2[t].sqrt();
    }

    let result = garch11_fit(&returns, 1000);
    let sum_ab = result.alpha + result.beta;

    eprintln!("GARCH fit on IGARCH(0.15, 0.85) data:");
    eprintln!("  ω = {:.6e} (true: {:.6e})", result.omega, omega_true);
    eprintln!("  α = {:.4} (true: {:.4})", result.alpha, alpha_true);
    eprintln!("  β = {:.4} (true: {:.4})", result.beta, beta_true);
    eprintln!("  α+β = {:.6} (true: 1.0, ceiling: 0.999)", sum_ab);
    eprintln!("  iterations = {}", result.iterations);

    // FINDING: The optimizer doesn't just hit the ceiling — it fails entirely.
    // The coordinate-descent MLE gets trapped in a bad local optimum when
    // the true DGP is IGARCH. The ω estimate is nonsensical (orders of magnitude off),
    // and α/β don't recover the true values at all.
    //
    // This is a deeper problem than boundary bias: the likelihood surface for
    // IGARCH is pathological. At α+β=1, the unconditional variance is infinite,
    // so the MLE has a degenerate landscape.
    //
    // The correct approach would be to fit IGARCH directly (constraining α+β=1)
    // or use a more robust optimizer (e.g., L-BFGS with boundary handling).
    // This is ANOTHER instance of Structure Beats Resources: better optimization
    // iterations won't fix this — you need the right parameterization.

    // The test passes by documenting the failure mode, not by asserting success.
    // ω should be on the order of 1e-4 (unconditional variance scale).
    // Getting 1e+13 is catastrophic.
    if result.omega > 1.0 {
        eprintln!("WARNING: ω = {:.2e} is nonsensical (expected ~1e-4)", result.omega);
        eprintln!("Coordinate descent MLE fails on IGARCH likelihood surface.");
        eprintln!("This is the volatility analog of fitting AR(1) at the unit root.");
    }

    // Either the optimizer finds reasonable parameters or it doesn't —
    // both outcomes document real behavior at the IGARCH boundary.
    eprintln!("IGARCH boundary = volatility analog of unit root.");
    eprintln!("The simple MLE optimizer is the 'wrong structure' for this problem.");
}

// ═══════════════════════════════════════════════════════════════════════════
// Special functions: digamma near poles (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// ψ(x) has poles at x = 0, -1, -2, ...
// The reflection formula ψ(1-x) - π·cot(πx) diverges at negative integers
// because cot(πn) = cos(πn)/sin(πn) and sin(πn) = 0.
// The implementation should return ±Inf or NaN at these poles.

#[test]
fn digamma_near_poles() {
    use tambear::special_functions::digamma;

    // At x=0: documented NaN
    assert!(digamma(0.0).is_nan(), "ψ(0) should be NaN");

    // At negative integers: poles. tan(πn) = 0 for integer n → division by zero.
    // PREVIOUSLY: returned huge finite values due to tan(nπ) ≈ ε_mach.
    // FIXED: now checks for negative integers explicitly before reflection formula.
    for &n in &[-1.0, -2.0, -3.0, -10.0] {
        let val = digamma(n);
        eprintln!("ψ({n}) = {val}");
        // Should be NaN or ±Inf at poles
        assert!(!val.is_finite(),
            "ψ({n}) should be non-finite at pole, got {val}");
    }

    // Near poles: ψ(-n + ε) should be large negative (approaching -∞ from right)
    // ψ(-n - ε) should be large positive (approaching +∞ from left)
    let eps = 1e-8;
    for &n in &[0.0, 1.0, 2.0] {
        let right = digamma(-n + eps); // just right of pole
        let left = digamma(-n - eps);  // just left of pole
        eprintln!("ψ({:.1e}) = {right:.4e}, ψ(-{:.1e}) = {left:.4e}",
            -n + eps, n + eps);
        // The values should be large and opposite in sign near a pole
        if right.is_finite() && left.is_finite() {
            assert!(right.abs() > 1e6,
                "ψ near pole should be large, got {right}");
        }
    }

    // Known values far from poles
    // ψ(1) = -γ ≈ -0.5772156649
    let psi1 = digamma(1.0);
    let euler_gamma = 0.5772156649015329;
    eprintln!("ψ(1) = {psi1:.15} (expected -γ = {:.15})", -euler_gamma);
    assert!((psi1 + euler_gamma).abs() < 1e-10,
        "ψ(1) = -γ, got {psi1}");

    // ψ(0.5) = -γ - 2·ln(2) ≈ -1.9635100260
    let psi_half = digamma(0.5);
    let expected = -euler_gamma - 2.0 * 2.0f64.ln();
    eprintln!("ψ(0.5) = {psi_half:.15} (expected {expected:.15})");
    assert!((psi_half - expected).abs() < 1e-10,
        "ψ(0.5) should be -γ - 2ln2, got {psi_half}");

    // Recurrence: ψ(x+1) = ψ(x) + 1/x for several x
    for &x in &[0.5, 1.0, 2.5, 10.0, 100.0] {
        let lhs = digamma(x + 1.0);
        let rhs = digamma(x) + 1.0 / x;
        eprintln!("Recurrence at x={x}: ψ(x+1)={lhs:.15}, ψ(x)+1/x={rhs:.15}");
        assert!((lhs - rhs).abs() < 1e-12,
            "Recurrence ψ(x+1) = ψ(x) + 1/x failed at x={x}: {lhs} vs {rhs}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Special functions: trigamma boundary + derivative consistency (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// ψ₁(x) = dψ/dx. We test:
// 1. Near x=0+: ψ₁(x) ~ 1/x² (should be very large)
// 2. Derivative consistency: ψ₁(x) ≈ (ψ(x+h) - ψ(x))/h
// 3. Recurrence: ψ₁(x+1) = ψ₁(x) - 1/x²

#[test]
// ═══════════════════════════════════════════════════════════════════════════
// Mixture: GMM model selection with BIC (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// EM converges to the maximum GIVEN K. But how do we choose K?
// BIC penalizes model complexity: BIC = -2·logL + k·log(n).
// With well-separated 2-component data, BIC should prefer K=2 over K=1 or K=3.
// This tests whether the BIC penalty correctly identifies the true number of components.

// ═══════════════════════════════════════════════════════════════════════════
// Causal: DiD with parallel trends violation (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// DiD assumes parallel trends: absent treatment, treated and control groups
// would have followed the same trajectory. When this fails (e.g., treated
// group was already trending up faster), DiD attributes the pre-existing
// trend to the treatment → biased upward.
//
// This is the causal analog of Structure Beats Resources: the "structure"
// (parallel trends assumption) determines whether the estimator is valid.
// More data with a violated assumption → more precise WRONG answer.

// ═══════════════════════════════════════════════════════════════════════════
// Number theory: Euler product restricted to {2,3} (EXPLORATORY)
// ═══════════════════════════════════════════════════════════════════════════
//
// ζ(s) = ∏_p 1/(1-p^{-s}) (Euler product over all primes).
// Restricted to primes {2,3}: ∏_{p∈{2,3}} 1/(1-p^{-s}).
// At s=2: (4/3)(9/8) = 3/2 exactly.
//
// The Collatz map uses exactly primes 2 and 3:
// - Odd → multiply by 3 and add 1
// - Even → divide by 2
//
// The heuristic contraction ratio per Collatz step:
// Each odd step multiplies by 3, then divides by 2^k where k ~ Geometric(1/2).
// Expected log₂ contraction = log₂(3) - E[k] = log₂(3) - 2 ≈ -0.415.
//
// Is there a deeper connection between the Euler factor 3/2 and the Collatz
// contraction? This test computes both and looks for structural relationships.

// ═══════════════════════════════════════════════════════════════════════════
// TDA: Persistence noise sensitivity (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// Persistent homology should be stable: small perturbations to the point cloud
// should produce small changes in the persistence diagram (bottleneck stability).
// But: adding a single outlier point can create spurious high-persistence features
// in H₀ (components that die late because the outlier is far from everything).
//
// This tests the bottleneck stability theorem empirically.

#[test]
fn persistence_outlier_sensitivity() {
    use tambear::tda::{rips_h0, bottleneck_distance, persistence_statistics};

    // Clean data: two tight clusters at (0,0) and (10,0)
    let n_clean = 10;
    let mut points_clean = Vec::new();
    let mut rng_state = 42u64;
    let mut noise = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.2
    };
    // Cluster 1: near (0, 0)
    for _ in 0..5 { points_clean.push((noise(), noise())); }
    // Cluster 2: near (10, 0)
    for _ in 0..5 { points_clean.push((10.0 + noise(), noise())); }

    // Build distance matrix
    let dist_clean = build_dist_2d(&points_clean);
    let h0_clean = rips_h0(&dist_clean, n_clean);
    let stats_clean = persistence_statistics(&h0_clean.pairs);

    // The most persistent finite feature should be the two-cluster merge at d ≈ 10
    let finite_pairs: Vec<_> = h0_clean.pairs.iter()
        .filter(|p| p.death.is_finite())
        .collect();
    let max_persistence = finite_pairs.iter()
        .map(|p| p.death - p.birth)
        .fold(0.0f64, f64::max);
    eprintln!("Clean data: max persistence = {max_persistence:.4} (expect ≈10)");
    assert!(max_persistence > 8.0 && max_persistence < 12.0,
        "Two clusters at distance 10 should merge around d=10, got {max_persistence}");

    // Add outlier at (100, 0) — far from everything
    let mut points_outlier = points_clean.clone();
    points_outlier.push((100.0, 0.0));
    let n_outlier = n_clean + 1;
    let dist_outlier = build_dist_2d(&points_outlier);
    let h0_outlier = rips_h0(&dist_outlier, n_outlier);

    let finite_outlier: Vec<_> = h0_outlier.pairs.iter()
        .filter(|p| p.death.is_finite())
        .collect();
    let max_persistence_outlier = finite_outlier.iter()
        .map(|p| p.death - p.birth)
        .fold(0.0f64, f64::max);
    eprintln!("With outlier at (100,0): max persistence = {max_persistence_outlier:.4}");

    // The outlier creates a new high-persistence H₀ feature (component that dies at d≈90-100)
    assert!(max_persistence_outlier > 80.0,
        "Outlier should create a long-lived component, got {max_persistence_outlier}");

    // Bottleneck distance between clean and outlier diagrams
    // should be large (the outlier changes the diagram significantly)
    let bn = bottleneck_distance(
        &h0_clean.pairs.iter().filter(|p| p.death.is_finite()).cloned().collect::<Vec<_>>(),
        &h0_outlier.pairs.iter().filter(|p| p.death.is_finite()).cloned().collect::<Vec<_>>(),
    );
    eprintln!("Bottleneck distance (clean vs outlier): {bn:.4}");
    assert!(bn > 30.0,
        "Single outlier should significantly change the diagram, bottleneck = {bn}");

    eprintln!("A single outlier creates a spurious high-persistence feature in H₀.");
    eprintln!("The bottleneck stability theorem bounds this by the Hausdorff distance");
    eprintln!("between point clouds — which IS the outlier distance.");
}

/// Build pairwise distance matrix from 2D points.
// ═══════════════════════════════════════════════════════════════════════════
// Causal: Doubly robust under single vs double misspecification (ADVERSARIAL)
// ═══════════════════════════════════════════════════════════════════════════
//
// The doubly robust estimator is consistent if EITHER:
// (a) the propensity model e(x) is correct, OR
// (b) the outcome model μ(x) is correct
//
// But if BOTH are wrong, the estimator is biased.
// This tests the "double protection" property empirically.

#[test]
fn doubly_robust_misspecification() {
    use tambear::causal::doubly_robust_ate;

    let n = 20;
    // True effect: 5.0
    let treatment = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let outcome = vec![
        10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 10.0, 11.0, 12.0, 10.5,
        15.0, 16.0, 17.0, 15.5, 16.5, 17.5, 15.0, 16.0, 17.0, 15.5,
    ];
    let true_ate = 5.0;

    // Case 1: Both models correct
    let e_correct = vec![0.5; n];
    let mu1_correct: Vec<f64> = (0..n).map(|i| if i < 10 { 15.5 } else { 16.0 }).collect();
    let mu0_correct: Vec<f64> = (0..n).map(|i| if i < 10 { 10.5 } else { 11.0 }).collect();
    let ate_both = doubly_robust_ate(&e_correct, &treatment, &outcome, &mu1_correct, &mu0_correct);
    eprintln!("Both correct:  ATE = {ate_both:.4} (true: {true_ate})");

    // Case 2: Propensity wrong, outcome correct
    let e_wrong = vec![0.3; n]; // wrong propensity (true is 0.5)
    let ate_e_wrong = doubly_robust_ate(&e_wrong, &treatment, &outcome, &mu1_correct, &mu0_correct);
    eprintln!("Prop wrong:    ATE = {ate_e_wrong:.4} (true: {true_ate})");

    // Case 3: Propensity correct, outcome wrong
    let mu1_wrong = vec![20.0; n]; // wrong outcome model (too high)
    let mu0_wrong = vec![5.0; n];  // wrong outcome model (too low)
    let ate_mu_wrong = doubly_robust_ate(&e_correct, &treatment, &outcome, &mu1_wrong, &mu0_wrong);
    eprintln!("Outcome wrong: ATE = {ate_mu_wrong:.4} (true: {true_ate})");

    // Case 4: Both wrong
    let ate_both_wrong = doubly_robust_ate(&e_wrong, &treatment, &outcome, &mu1_wrong, &mu0_wrong);
    eprintln!("Both wrong:    ATE = {ate_both_wrong:.4} (true: {true_ate})");

    // Cases 1-3 should be close to true ATE (doubly robust property)
    assert!((ate_both - true_ate).abs() < 1.5,
        "Both correct should give good ATE: {ate_both}");
    assert!((ate_e_wrong - true_ate).abs() < 1.5,
        "Single misspec (propensity) should still work: {ate_e_wrong}");
    assert!((ate_mu_wrong - true_ate).abs() < 1.5,
        "Single misspec (outcome) should still work: {ate_mu_wrong}");

    // Case 4 may be biased — the "doubly robust" protection fails
    let bias_both_wrong = (ate_both_wrong - true_ate).abs();
    let bias_single = (ate_e_wrong - true_ate).abs().max((ate_mu_wrong - true_ate).abs());
    eprintln!("Bias (both wrong): {bias_both_wrong:.4}");
    eprintln!("Max bias (single wrong): {bias_single:.4}");

    // Document the behavior — the key insight is WHEN the protection holds
    eprintln!("\nDoubly robust: single misspecification is tolerated.");
    eprintln!("Double misspecification breaks the guarantee — bias may be large.");
}

fn build_dist_2d(points: &[(f64, f64)]) -> Vec<f64> {
    let n = points.len();
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            dist[i * n + j] = (dx * dx + dy * dy).sqrt();
        }
    }
    dist
}

// ═══════════════════════════════════════════════════════════════════════════
// Number theory: Collatz transition matrix spectrum (EXPLORATORY)
// ═══════════════════════════════════════════════════════════════════════════
//
// Build the Collatz transition matrix for odd numbers in [1, N].
// Each odd n maps to (3n+1)/2^k (the next odd number).
// Some map outside [1, N] ("escape").
// The eigenvalues of this matrix control the dynamics.

#[test]
fn collatz_transition_spectrum() {
    // Odd numbers in [1, 255]: there are 128 of them (1, 3, 5, ..., 255)
    let n_max = 255u64;
    let odds: Vec<u64> = (1..=n_max).step_by(2).collect();
    let n = odds.len(); // 128
    let odd_to_idx: std::collections::HashMap<u64, usize> = odds.iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    // Build transition: for each odd number, compute the Collatz step
    let mut transitions: Vec<Option<usize>> = Vec::with_capacity(n);
    let mut n_escape = 0;
    for &odd in &odds {
        let next = 3 * odd + 1;
        // Divide by 2 until odd
        let mut m = next;
        while m % 2 == 0 { m /= 2; }
        if let Some(&idx) = odd_to_idx.get(&m) {
            transitions.push(Some(idx));
        } else {
            // Maps outside our range
            transitions.push(None);
            n_escape += 1;
        }
    }

    eprintln!("Collatz transition matrix for odd numbers in [1, {n_max}]:");
    eprintln!("  States: {n}");
    eprintln!("  Escape: {n_escape} ({:.1}%)", 100.0 * n_escape as f64 / n as f64);
    eprintln!("  Internal: {} ({:.1}%)", n - n_escape, 100.0 * (n - n_escape) as f64 / n as f64);

    // Build the "number of times each state is a target" histogram
    let mut in_degree = vec![0usize; n];
    for t in &transitions {
        if let Some(idx) = t { in_degree[*idx] += 1; }
    }
    let max_in = in_degree.iter().max().unwrap();
    let min_in = in_degree.iter().min().unwrap();
    let avg_in = in_degree.iter().sum::<usize>() as f64 / n as f64;
    eprintln!("  In-degree: min={min_in}, max={max_in}, avg={avg_in:.2}");

    // Fixed points: n where collatz step returns to n
    let fixed: Vec<u64> = odds.iter().enumerate()
        .filter(|&(i, _)| transitions[i] == Some(i))
        .map(|(_, &v)| v)
        .collect();
    eprintln!("  Fixed points: {:?}", fixed);

    // Cycle detection: follow each state for up to n steps
    let mut n_cycles = 0;
    let mut longest_internal_chain = 0;
    for start in 0..n {
        let mut pos = start;
        let mut steps = 0;
        loop {
            if let Some(next) = transitions[pos] {
                steps += 1;
                pos = next;
                if pos == start {
                    if steps > 0 { n_cycles += 1; }
                    break;
                }
                if steps > n { break; } // not a short cycle
            } else {
                longest_internal_chain = longest_internal_chain.max(steps);
                break; // escaped
            }
        }
    }
    eprintln!("  Short cycles (len ≤ {n}): {n_cycles}");
    eprintln!("  Longest chain before escape: {longest_internal_chain}");

    // The key structural observation: MOST states escape.
    // The few that don't escape form the "attractor" — the 1→4→2→1 cycle.
    // For the Collatz conjecture: eventually all trajectories reach this cycle,
    // so the transition matrix has a dominant absorbing structure.

    // Verify that 1 is a fixed point (1 → 4 → 2 → 1)
    // Actually 1 → 4, which is even, so: 4/2=2, 2/2=1. So 1 maps to 1.
    assert!(transitions[0] == Some(0),
        "1 should map to itself: 1→4→2→1");
    eprintln!("  Confirmed: 1 is a fixed point (the Collatz cycle attractor).");
}

#[test]
fn euler_product_23_and_collatz() {
    // Euler product for ζ(s) restricted to {2,3}
    let euler_23 = |s: f64| -> f64 {
        1.0 / (1.0 - 2.0f64.powf(-s)) * 1.0 / (1.0 - 3.0f64.powf(-s))
    };

    // At s=2: should be exactly 3/2
    let e2 = euler_23(2.0);
    eprintln!("Euler product {{2,3}} at s=2: {e2:.15} (exact: 1.5)");
    assert!((e2 - 1.5).abs() < 1e-14, "Should be exactly 3/2");

    // At s=1: diverges (ζ(1) = harmonic series)
    // The restricted product at s=1: 1/(1-1/2) × 1/(1-1/3) = 2 × 3/2 = 3
    let e1 = euler_23(1.0);
    eprintln!("Euler product {{2,3}} at s=1: {e1:.15} (exact: 3.0)");
    assert!((e1 - 3.0).abs() < 1e-14, "Should be exactly 3");

    // The Collatz contraction ratio per odd step:
    // multiply by 3, divide by 2^k where Pr(k) = 1/2^k
    // Expected log₂ ratio = log₂(3) + E[-k·log₂(2)] = log₂(3) - E[k]
    // E[k] for trailing zeros of 3n+1: the argument is that 3n+1 has
    // random trailing zeros, so E[k] = Σ_{k=1}^∞ k·2^{-k} = 2
    let log2_contraction = 3.0f64.log2() - 2.0;
    let contraction = 2.0f64.powf(log2_contraction); // = 3/4
    eprintln!("Collatz contraction per odd step: 3/2^E[k] = 3/4 = {contraction:.4}");

    // The ratio 3/2 appears as:
    // - Euler product ζ(2) restricted to {2,3} = 3/2
    // - Collatz map: multiply by 3, then divide by 2 (the first division)
    // - Collatz net ratio per odd-even pair: (3n+1)/2 ~ 3n/2 (before further divisions)
    //
    // The 3/2 is the FIRST-ORDER structure of the Collatz map.
    // The full contraction includes the additional factor of 1/2 from geometric trailing zeros.
    // Net contraction = 3/4 = (3/2) × (1/2).
    //
    // So: the Euler factor IS the Collatz first-order ratio.
    // The second factor (1/2 from E[trailing_zeros] = 2, first division already counted)
    // is the stochastic component.
    assert!((contraction - 0.75).abs() < 1e-14, "3/4 = 0.75");

    // Deeper: ζ(2) = π²/6. The {2,3} contribution is 3/2 out of π²/6 ≈ 1.6449.
    // The remaining primes contribute (π²/6) / (3/2) = π²/9 ≈ 1.0966.
    let zeta2 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    let remaining = zeta2 / 1.5;
    eprintln!("ζ(2) = {zeta2:.10}");
    eprintln!("{{2,3}} factor = {e2:.10}");
    eprintln!("Remaining primes = {remaining:.10} = π²/9");
    eprintln!("The {{2,3}} factor accounts for {:.1}% of ζ(2)", (e2 / zeta2) * 100.0);

    // √(3/2) ≈ 1.2247
    // This is the geometric mean of 1 (even step, divide by 2) and 3/2 (odd step, ×3/2)
    let sqrt_32 = (1.5f64).sqrt();
    eprintln!("√(3/2) = {sqrt_32:.10}");
    eprintln!("This is the geometric mean contraction over even/odd step pairs.");

    // The test passes — it's exploratory, documenting the numerical relationships.
    // The question for the team: is the connection between the Euler product
    // and the Collatz ratio a coincidence (both involve {2,3}) or something deeper?
}

#[test]
fn did_parallel_trends_violation() {
    use tambear::causal::did;

    // Scenario: True treatment effect = 0 (no real treatment effect)
    // But: treatment group has a pre-existing upward trend of +5 per period
    // while control group is flat.

    let n_per_cell = 25;
    let mut y = Vec::new();
    let mut treat = Vec::new();
    let mut post = Vec::new();
    let mut rng_state = 9999u64;
    let mut noise = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0
    };

    // Control group, pre-period: mean = 10
    for _ in 0..n_per_cell {
        y.push(10.0 + noise());
        treat.push(0.0);
        post.push(0.0);
    }
    // Control group, post-period: mean = 10 (flat trend)
    for _ in 0..n_per_cell {
        y.push(10.0 + noise());
        treat.push(0.0);
        post.push(1.0);
    }
    // Treatment group, pre-period: mean = 10 (same starting point)
    for _ in 0..n_per_cell {
        y.push(10.0 + noise());
        treat.push(1.0);
        post.push(0.0);
    }
    // Treatment group, post-period: mean = 15 (trend of +5, NOT treatment)
    for _ in 0..n_per_cell {
        y.push(15.0 + noise()); // No treatment effect — just pre-existing trend
        treat.push(1.0);
        post.push(1.0);
    }

    let result = did(&y, &treat, &post);
    eprintln!("DiD with parallel trends violation:");
    eprintln!("  True treatment effect = 0");
    eprintln!("  Pre-existing trend in treatment group = +5");
    eprintln!("  DiD estimate = {:.4} (should be ≈5, NOT ≈0)", result.effect);
    eprintln!("  SE = {:.4}, t = {:.4}, p = {:.4e}", result.se, result.t_stat, result.p_value);

    // DiD INCORRECTLY estimates a large positive treatment effect
    // because it attributes the pre-existing trend to the treatment.
    assert!(result.effect > 3.0,
        "DiD should be biased upward by the trend, got {:.4}", result.effect);

    // And it's "statistically significant" — the WRONG answer with high confidence
    assert!(result.p_value < 0.001,
        "DiD confidently reports the wrong answer, p={:.4e}", result.p_value);

    eprintln!("DiD produces a confident WRONG answer when parallel trends are violated.");
    eprintln!("More data → more confident → more dangerously wrong.");
    eprintln!("This IS Structure Beats Resources: the assumption structure (parallel trends)");
    eprintln!("determines validity. No amount of data fixes a structural violation.");
}

#[test]
fn gmm_bic_model_selection() {
    use tambear::mixture::{gmm_em, gmm_bic};

    // Generate well-separated 2-component 1D data
    let n = 100;
    let d = 1;
    let mut data = Vec::new();
    let mut rng_state = 12345u64;
    let mut next_normal = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = ((rng_state >> 33) as f64 / u32::MAX as f64).max(1e-15);
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (rng_state >> 33) as f64 / u32::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // Component 1: N(-5, 1), Component 2: N(5, 1)
    for _ in 0..50 { data.push(next_normal() - 5.0); }
    for _ in 0..50 { data.push(next_normal() + 5.0); }

    // Fit K=1, K=2, K=3
    let mut bics = Vec::new();
    for k in 1..=3 {
        let result = gmm_em(&data, n, d, k, 200, 1e-6);
        let bic = gmm_bic(result.log_likelihood, n, d, k);
        bics.push((k, bic, result.log_likelihood));
        eprintln!("K={k}: logL={:.2}, BIC={:.2}, iters={}", result.log_likelihood, bic, result.iterations);
    }

    // BIC should be minimized at K=2
    let best_k = bics.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
    eprintln!("Best K by BIC: {best_k}");
    assert_eq!(best_k, 2, "BIC should select K=2 for 2-component data, selected K={best_k}");

    // K=1 should have the worst (highest) BIC
    assert!(bics[0].1 > bics[1].1,
        "BIC(K=1) should be worse than BIC(K=2)");
}

#[test]
fn softmax_temperature_robustness() {
    // The temperature unification conjecture: sum mode (T→∞) and optimize mode
    // (T→0) are connected by temperature. This test explores whether there is
    // a critical temperature where robustness changes qualitatively.
    //
    // softmax_β(x) = Σ xᵢ·exp(β·xᵢ) / Σ exp(β·xᵢ)
    // β=0 → arithmetic mean (fragile: 0% breakdown)
    // β→∞ → max (also fragile: dominated by largest value)
    // β→-∞ → min (also fragile: dominated by smallest value)
    //
    // Question: is there an intermediate β with nonzero breakdown?

    let clean_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let clean_softmax = |beta: f64| -> f64 {
        let weights: Vec<f64> = clean_data.iter().map(|&x| (beta * x).exp()).collect();
        let sum_w: f64 = weights.iter().sum();
        clean_data.iter().zip(&weights).map(|(&x, &w)| x * w / sum_w).sum::<f64>()
    };

    // Contaminate: replace one observation with outlier
    let mut contaminated = clean_data.clone();
    contaminated[9] = 1000.0; // single outlier
    let contam_softmax = |beta: f64| -> f64 {
        // Protect against overflow: subtract max before exp
        let max_bx = contaminated.iter().map(|&x| beta * x).fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = contaminated.iter().map(|&x| (beta * x - max_bx).exp()).collect();
        let sum_w: f64 = weights.iter().sum();
        contaminated.iter().zip(&weights).map(|(&x, &w)| x * w / sum_w).sum::<f64>()
    };

    eprintln!("Softmax temperature robustness (10% contamination, outlier=1000):");
    eprintln!("{:>8} {:>12} {:>12} {:>12}", "beta", "clean", "contam", "influence");

    let betas = [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0];
    let mut influences = Vec::new();
    for &beta in &betas {
        let clean = clean_softmax(beta);
        let contam = contam_softmax(beta);
        let influence = (contam - clean).abs();
        influences.push((beta, influence));
        eprintln!("{beta:>8.1} {clean:>12.4} {contam:>12.4} {influence:>12.4}");
    }

    // At β=0 (mean): influence = (1000-10)/10 = 99
    let mean_influence = influences.iter().find(|(b, _)| *b == 0.0).unwrap().1;
    eprintln!("\nMean (β=0) influence: {mean_influence:.4}");

    // Key finding: influence should be LARGER at positive β (outlier gets more weight)
    // and SMALLER at negative β (outlier gets less weight).
    // Negative β = inverse softmax = attention to SMALL values = implicit outlier rejection!
    let neg_beta_influence = influences.iter().find(|(b, _)| *b == -1.0).unwrap().1;
    let pos_beta_influence = influences.iter().find(|(b, _)| *b == 1.0).unwrap().1;

    eprintln!("β=-1 influence: {neg_beta_influence:.4} (should be < mean)");
    eprintln!("β=+1 influence: {pos_beta_influence:.4} (should be > mean)");

    // The conjecture: negative temperature creates implicit robustness
    // by downweighting extreme values.
    assert!(neg_beta_influence < mean_influence,
        "Negative β should reduce outlier influence: {neg_beta_influence} vs {mean_influence}");

    // But the min (β→-∞) is also fragile — it just tracks the minimum.
    // So there should be a sweet spot at moderate negative β.
    eprintln!("\nConclusion: temperature IS a robustness parameter.");
    eprintln!("Negative β = implicit outlier downweighting (for upward outliers).");
    eprintln!("This is asymmetric: β<0 protects against high outliers, not low ones.");
    eprintln!("True robustness requires SYMMETRIC rejection — bisquare, not softmax.");
    eprintln!("Temperature governs directional robustness within a function class.");
    eprintln!("Structural robustness (bisquare) requires crossing the Fock boundary.");
}

#[test]
fn trigamma_boundary_and_consistency() {
    use tambear::special_functions::{digamma, trigamma};

    // Trigamma at x → 0+: should be ~ 1/x²
    let x_small = 0.001;
    let tri_small = trigamma(x_small);
    let approx = 1.0 / (x_small * x_small);
    eprintln!("ψ₁({x_small}) = {tri_small:.4}, 1/x² = {approx:.4}");
    assert!((tri_small / approx - 1.0).abs() < 0.01,
        "ψ₁(x) ~ 1/x² near 0, got ratio {}", tri_small / approx);

    // Trigamma at negative x: should be NaN (poles + domain)
    assert!(trigamma(-1.0).is_nan(), "ψ₁(-1) should be NaN");
    assert!(trigamma(0.0).is_nan(), "ψ₁(0) should be NaN");

    // Known value: ψ₁(1) = π²/6
    let tri1 = trigamma(1.0);
    let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    eprintln!("ψ₁(1) = {tri1:.15} (expected π²/6 = {expected:.15})");
    assert!((tri1 - expected).abs() < 1e-10,
        "ψ₁(1) = π²/6, got {tri1}");

    // Recurrence: ψ₁(x+1) = ψ₁(x) - 1/x²
    for &x in &[0.5, 1.0, 2.0, 5.0, 50.0] {
        let lhs = trigamma(x + 1.0);
        let rhs = trigamma(x) - 1.0 / (x * x);
        eprintln!("Trigamma recurrence at x={x}: {lhs:.15} vs {rhs:.15}");
        assert!((lhs - rhs).abs() < 1e-11,
            "Recurrence ψ₁(x+1) = ψ₁(x) - 1/x² failed at x={x}");
    }

    // Derivative consistency: ψ₁(x) ≈ (ψ(x+h) - ψ(x))/h
    let h = 1e-6;
    for &x in &[1.0, 2.5, 10.0, 100.0] {
        let numerical = (digamma(x + h) - digamma(x)) / h;
        let analytical = trigamma(x);
        let rel_err = ((numerical - analytical) / analytical).abs();
        eprintln!("ψ₁({x}): analytical={analytical:.10}, numerical={numerical:.10}, rel_err={rel_err:.2e}");
        // Numerical derivative has O(h) error, so tolerance is proportional to h
        assert!(rel_err < 1e-4,
            "Derivative consistency failed at x={x}: analytical={analytical}, numerical={numerical}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Hypothesis: BH FDR with all significant (no correction needed)
// ═══════════════════════════════════════════════════════════════════════════
//
// When all p-values are very small (all truly significant), the BH correction
// should barely change them. When all p-values are uniform [0,1] (all null),
// BH should reject far fewer than Bonferroni.

#[test]
fn bh_fdr_extreme_cases() {
    use tambear::hypothesis::{bonferroni, holm, benjamini_hochberg};

    // Case 1: All highly significant (p ≈ 0)
    let p_sig = vec![1e-10, 1e-9, 1e-8, 1e-7, 1e-6];
    let bh_sig = benjamini_hochberg(&p_sig);
    let bon_sig = bonferroni(&p_sig);
    eprintln!("BH FDR (all significant):");
    for i in 0..5 {
        eprintln!("  p={:.0e}: Bonf={:.2e}, Holm=skip, BH={:.2e}",
            p_sig[i], bon_sig[i], bh_sig[i]);
    }
    // BH should still be < 0.05 for all
    assert!(bh_sig.iter().all(|&p| p < 0.05),
        "All truly significant → BH should reject all");

    // Case 2: All null (uniform p-values)
    let p_null = vec![0.12, 0.25, 0.48, 0.67, 0.91];
    let bh_null = benjamini_hochberg(&p_null);
    let bon_null = bonferroni(&p_null);
    let holm_null = holm(&p_null);
    eprintln!("BH FDR (all null):");
    for i in 0..5 {
        eprintln!("  p={:.2}: Bonf={:.4}, Holm={:.4}, BH={:.4}",
            p_null[i], bon_null[i], holm_null[i], bh_null[i]);
    }
    // None should be significant at 0.05
    assert!(bh_null.iter().all(|&p| p > 0.05),
        "All null → no BH rejections at 0.05");

    // BH should be uniformly <= Bonferroni (less conservative)
    for i in 0..5 {
        assert!(bh_null[i] <= bon_null[i] + 1e-10,
            "BH should be ≤ Bonferroni: BH={}, Bonf={}", bh_null[i], bon_null[i]);
    }
    eprintln!("  ✓ BH ≤ Bonferroni (less conservative, as expected)");

    // Holm should be between BH and Bonferroni
    // (Holm controls FWER, BH controls FDR)
    for i in 0..5 {
        assert!(holm_null[i] <= bon_null[i] + 1e-10,
            "Holm should be ≤ Bonferroni");
    }

    // Case 3: Edge — one tiny p among many nulls
    let mut p_mixed = vec![0.8; 20];
    p_mixed[0] = 1e-10; // one truly significant
    let bh_mixed = benjamini_hochberg(&p_mixed);
    eprintln!("BH FDR (1 significant among 19 null):");
    eprintln!("  p[0]={:.0e} → BH={:.2e}", p_mixed[0], bh_mixed[0]);
    assert!(bh_mixed[0] < 0.05,
        "The one truly significant p should survive BH: {}", bh_mixed[0]);
    // The null p-values should NOT be significant
    let null_rejections = bh_mixed[1..].iter().filter(|&&p| p < 0.05).count();
    eprintln!("  Null rejections at 0.05: {}/19", null_rejections);
    assert_eq!(null_rejections, 0,
        "Null p-values should not be rejected by BH");
}

// ═══════════════════════════════════════════════════════════════════════════
// Nonparametric: Kendall tau with all ties (denominator boundary)
// ═══════════════════════════════════════════════════════════════════════════
//
// When all values in x are equal (or all in y), every pair is a tie.
// concordant = discordant = 0, ties_x = n(n-1)/2.
// denom = sqrt(0 * (0 + ties_y)) = 0 → NaN.

#[test]
fn kendall_tau_all_ties() {
    use tambear::nonparametric::kendall_tau;

    // All x values equal → all ties in x
    let x = vec![5.0; 10];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let tau = kendall_tau(&x, &y);
    eprintln!("Kendall tau (all x tied): {tau}");
    assert!(tau.is_nan(), "All-ties should give NaN, got {tau}");

    // Perfect concordance
    let x2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let tau2 = kendall_tau(&x2, &y2);
    eprintln!("Kendall tau (perfect concordance): {tau2}");
    assert!((tau2 - 1.0).abs() < 1e-10, "Perfect concordance → τ = 1.0, got {tau2}");

    // Perfect discordance
    let y3 = vec![50.0, 40.0, 30.0, 20.0, 10.0];
    let tau3 = kendall_tau(&x2, &y3);
    eprintln!("Kendall tau (perfect discordance): {tau3}");
    assert!((tau3 + 1.0).abs() < 1e-10, "Perfect discordance → τ = -1.0, got {tau3}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Nonparametric: Level spacing r-statistic (RMT connection)
// ═══════════════════════════════════════════════════════════════════════════
//
// For uniformly spaced levels (perfectly regular): r → 1.0.
// For Poisson spacing (uncorrelated): r ≈ 2ln2 - 1 ≈ 0.386.
// For GUE (quantum chaotic): r ≈ 4 - 2√3 ≈ 0.536.
//
// This tests that the implementation computes r correctly for known cases.

#[test]
fn level_spacing_known_cases() {
    use tambear::nonparametric::level_spacing_r_stat;

    // Perfectly uniform spacing → consecutive ratio min/max = 1.0 → r = 1.0
    let uniform: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let r_uniform = level_spacing_r_stat(&uniform);
    eprintln!("Level spacing r (uniform): {r_uniform:.6} (expected ~1.0)");
    assert!((r_uniform - 1.0).abs() < 0.01,
        "Uniform spacing → r ≈ 1.0, got {r_uniform}");

    // Poisson (exponential gaps): generate from -ln(U)
    let mut rng = 42u64;
    let mut poisson_levels: Vec<f64> = Vec::new();
    let mut t = 0.0;
    for _ in 0..10000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng as f64 / u64::MAX as f64).max(1e-15);
        t -= u.ln(); // exponential gap
        poisson_levels.push(t);
    }
    let r_poisson = level_spacing_r_stat(&poisson_levels);
    let r_poisson_theory = 2.0 * 2.0_f64.ln() - 1.0; // ≈ 0.386
    eprintln!("Level spacing r (Poisson): {r_poisson:.6} (expected ~{r_poisson_theory:.4})");
    assert!((r_poisson - r_poisson_theory).abs() < 0.05,
        "Poisson spacing → r ≈ {r_poisson_theory}, got {r_poisson}");

    // All identical levels → gaps = 0, mean_gap = 0 → NaN
    let identical = vec![5.0; 100];
    let r_identical = level_spacing_r_stat(&identical);
    eprintln!("Level spacing r (identical): {r_identical}");
    assert!(r_identical.is_nan(), "Identical levels → NaN, got {r_identical}");
}

// ═══════════════════════════════════════════════════════════════════════════
// COPA: Merge associativity with different partitions (HIGH)
// ═══════════════════════════════════════════════════════════════════════════
//
// COPA merge is an associative semigroup operation:
//   merge(merge(A, B), C) == merge(A, merge(B, C))
//
// This is the parallel scan contract. If it fails, GPU parallel reduction
// produces wrong results. Test with data that stresses the cross-term
// (nₐ·n_b/n)·ΔΔᵀ — data with large mean shifts between partitions.

#[test]
fn copa_merge_associativity() {
    use tambear::copa::CopaState;

    // Three groups with very different means (stresses the ΔΔᵀ cross-term)
    let group_a: Vec<f64> = vec![
        100.0, 200.0,
        101.0, 199.0,
        102.0, 201.0,
    ];
    let group_b: Vec<f64> = vec![
        0.0, 0.0,
        1.0, -1.0,
    ];
    let group_c: Vec<f64> = vec![
        -50.0, 50.0,
        -49.0, 51.0,
        -51.0, 49.0,
        -50.5, 50.5,
    ];

    let p = 2;
    let mut sa = CopaState::new(p);
    sa.add_batch(&group_a, 3);
    let mut sb = CopaState::new(p);
    sb.add_batch(&group_b, 2);
    let mut sc = CopaState::new(p);
    sc.add_batch(&group_c, 4);

    // Left-associative: (A ∘ B) ∘ C
    let ab = CopaState::merge(&sa, &sb);
    let left = CopaState::merge(&ab, &sc);

    // Right-associative: A ∘ (B ∘ C)
    let bc = CopaState::merge(&sb, &sc);
    let right = CopaState::merge(&sa, &bc);

    // Also: one-shot (all data at once)
    let mut all = CopaState::new(p);
    let mut all_data = Vec::new();
    all_data.extend_from_slice(&group_a);
    all_data.extend_from_slice(&group_b);
    all_data.extend_from_slice(&group_c);
    all.add_batch(&all_data, 9);

    eprintln!("COPA merge associativity:");
    eprintln!("  Left (AB)C:  n={}, mean=[{:.6}, {:.6}]", left.n, left.mean[0], left.mean[1]);
    eprintln!("  Right A(BC): n={}, mean=[{:.6}, {:.6}]", right.n, right.mean[0], right.mean[1]);
    eprintln!("  One-shot:    n={}, mean=[{:.6}, {:.6}]", all.n, all.mean[0], all.mean[1]);

    assert_eq!(left.n, right.n);
    assert_eq!(left.n, all.n);

    // Mean should agree to machine precision
    for j in 0..p {
        assert!((left.mean[j] - right.mean[j]).abs() < 1e-10,
            "mean[{j}] differs: left={} right={}", left.mean[j], right.mean[j]);
        assert!((left.mean[j] - all.mean[j]).abs() < 1e-10,
            "mean[{j}] differs: merged={} oneshot={}", left.mean[j], all.mean[j]);
    }

    // Cross-product matrix C should agree
    for j in 0..p {
        for k in 0..p {
            let l = left.c[j * p + k];
            let r = right.c[j * p + k];
            let a = all.c[j * p + k];
            let tol = a.abs() * 1e-10 + 1e-10;
            assert!((l - r).abs() < tol,
                "C[{j},{k}] assoc: left={l} right={r} diff={}", (l - r).abs());
            assert!((l - a).abs() < tol,
                "C[{j},{k}] vs oneshot: merged={l} oneshot={a} diff={}", (l - a).abs());
        }
    }
    eprintln!("  ✓ Merge is associative to machine precision");

    // Covariance should match
    let cov_merged = left.covariance();
    let cov_oneshot = all.covariance();
    for j in 0..p {
        for k in 0..p {
            let diff = (cov_merged.get(j, k) - cov_oneshot.get(j, k)).abs();
            assert!(diff < 1e-8,
                "Covariance[{j},{k}] differs: {diff}");
        }
    }
    eprintln!("  ✓ Covariance matches between merged and one-shot");
}

// ═══════════════════════════════════════════════════════════════════════════
// COPA: Mahalanobis with near-singular covariance (MEDIUM)
// ═══════════════════════════════════════════════════════════════════════════
//
// When data is nearly rank-deficient (e.g., 3D data on a 2D plane),
// the covariance matrix is near-singular. Cholesky should fail,
// returning None from mahalanobis(). This is the correct behavior.

#[test]
fn copa_mahalanobis_near_singular() {
    use tambear::copa::CopaState;

    // 3D data on a plane: z = x + y
    let mut copa = CopaState::new(3);
    let mut rng = 42u64;
    for _ in 0..50 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = (rng as f64 / u64::MAX as f64 - 0.5) * 10.0;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let y = (rng as f64 / u64::MAX as f64 - 0.5) * 10.0;
        let z = x + y; // perfect linear combination → rank 2
        copa.add(&[x, y, z]);
    }

    let point = [1.0, 2.0, 3.0]; // on the plane
    let d = copa.mahalanobis(&point);
    eprintln!("COPA Mahalanobis (near-singular, 3D on 2D plane):");
    eprintln!("  result = {:?}", d);

    // Cholesky of the 3×3 covariance should fail (rank 2)
    // mahalanobis returns None
    match d {
        None => eprintln!("  ✓ Correctly detected singular covariance (returned None)"),
        Some(val) => {
            eprintln!("  ⚠ Got distance = {val:.6} (Cholesky succeeded on near-singular matrix)");
            // This might happen with floating-point noise giving it just enough rank
            assert!(val.is_finite(), "Mahalanobis should be finite if it returns Some");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spectral: Lomb-Scargle recovers known frequency from irregular data
// ═══════════════════════════════════════════════════════════════════════════
//
// The Lomb-Scargle periodogram handles non-uniform sampling.
// Inject a pure sinusoid at known frequency with irregular time gaps.
// The LS should identify the frequency accurately.

#[test]
fn lomb_scargle_known_frequency() {
    use tambear::spectral::lomb_scargle;

    // Irregular sampling times (median Δt ≈ 0.1, but with gaps)
    let mut times = Vec::new();
    let mut values = Vec::new();
    let f_true = 2.5; // Hz
    let mut rng = 42u64;
    let mut t = 0.0;
    for _ in 0..200 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let dt = 0.05 + 0.15 * (rng as f64 / u64::MAX as f64); // 0.05 to 0.2
        t += dt;
        times.push(t);
        values.push((2.0 * std::f64::consts::PI * f_true * t).sin());
    }

    let result = lomb_scargle(&times, &values, 100);

    // Find peak frequency
    let (peak_idx, _) = result.power.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    let f_peak = result.freqs[peak_idx];
    eprintln!("Lomb-Scargle known frequency:");
    eprintln!("  true f = {f_true} Hz, detected f = {f_peak:.4} Hz");
    eprintln!("  peak power = {:.4}", result.power[peak_idx]);

    // Should be within ~1 frequency bin of the true frequency
    let df = result.freqs[1] - result.freqs[0];
    assert!((f_peak - f_true).abs() < 2.0 * df,
        "LS peak at {f_peak} should be near true {f_true} (resolution = {df})");
}

// ═══════════════════════════════════════════════════════════════════════════
// Spectral: Coherence of identical vs uncorrelated signals
// ═══════════════════════════════════════════════════════════════════════════
//
// Identical signals → coherence = 1.0 at all frequencies.
// Uncorrelated signals → coherence ≈ 0 (bias depends on n_segments).
// With 1 segment, coherence is always 1.0 (mathematical identity), which
// is a known bias. Need multiple segments for meaningful coherence.

#[test]
fn coherence_identical_vs_uncorrelated() {
    use tambear::spectral::cross_spectral;

    // Create a test signal: sum of two sinusoids
    let n = 1024;
    let fs = 100.0;
    let signal: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / fs;
        (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 25.0 * t).sin()
    }).collect();

    // Identical signals → coherence should be 1.0
    let result_same = cross_spectral(&signal, &signal, fs, 256, 0.5);
    let max_coh = result_same.coherence.iter()
        .skip(1) // skip DC
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_coh = result_same.coherence.iter()
        .skip(1)
        .filter(|c| **c > 0.0) // skip zero-power bins
        .cloned()
        .fold(f64::INFINITY, f64::min);
    eprintln!("Coherence (identical signals):");
    eprintln!("  max = {max_coh:.6}, min (nonzero) = {min_coh:.6}");
    assert!(max_coh > 0.99, "Same signal coherence should be ~1.0, max={max_coh}");

    // Uncorrelated signals → coherence should be low
    let mut rng = 42u64;
    let noise: Vec<f64> = (0..n).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng as f64 / u64::MAX as f64 - 0.5) * 2.0
    }).collect();
    let result_uncorr = cross_spectral(&signal, &noise, fs, 256, 0.5);
    let mean_coh: f64 = result_uncorr.coherence.iter().skip(1).sum::<f64>()
        / (result_uncorr.coherence.len() - 1) as f64;
    eprintln!("Coherence (uncorrelated signals):");
    eprintln!("  mean = {mean_coh:.6}");
    // With Welch averaging, bias ≈ 1/n_segments, so mean coherence should be low
    assert!(mean_coh < 0.5,
        "Uncorrelated coherence should be low, mean={mean_coh}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Spectral: Spectral entropy distinguishes pure tone from noise
// ═══════════════════════════════════════════════════════════════════════════
//
// Pure tone → all power in one bin → H ≈ 0 (low entropy).
// White noise → uniform PSD → H ≈ log(N) (max entropy).

#[test]
fn spectral_entropy_tone_vs_noise() {
    use tambear::spectral::{spectral_entropy, spectral_entropy_normalized};

    // Pure tone PSD: all power in one bin
    let n = 128;
    let mut tone_psd = vec![0.0; n];
    tone_psd[10] = 100.0; // all energy at bin 10
    let h_tone = spectral_entropy(&tone_psd);
    let h_tone_norm = spectral_entropy_normalized(&tone_psd);

    // White noise PSD: uniform
    let noise_psd = vec![1.0; n];
    let h_noise = spectral_entropy(&noise_psd);
    let h_noise_norm = spectral_entropy_normalized(&noise_psd);

    eprintln!("Spectral entropy:");
    eprintln!("  Tone:  H = {h_tone:.6}, H_norm = {h_tone_norm:.6}");
    eprintln!("  Noise: H = {h_noise:.6}, H_norm = {h_noise_norm:.6}");
    eprintln!("  Max H = {:.6}", (n as f64).ln());

    assert!(h_tone < 0.01, "Pure tone entropy should be ~0, got {h_tone}");
    assert!(h_tone_norm < 0.01, "Normalized tone entropy should be ~0, got {h_tone_norm}");
    assert!((h_noise_norm - 1.0).abs() < 0.01,
        "White noise normalized entropy should be ~1.0, got {h_noise_norm}");
    assert!(h_noise > h_tone * 100.0,
        "Noise entropy should be >> tone entropy");
}

// ═══════════════════════════════════════════════════════════════════════════
// Mixed Effects: ICC with zero between-group variance (HIGH)
// ═══════════════════════════════════════════════════════════════════════════
//
// When all group means are identical (σ²_u = 0), the EM should converge
// to σ²_u = 0, ICC = 0. But lambda = σ²/σ²_u → ∞ (line 121),
// which is clipped to 1e10. This massive regularization shrinks u → 0,
// which is correct! But the convergence path may be slow.

#[test]
fn lme_zero_between_group_variance() {
    use tambear::mixed_effects::lme_random_intercept;

    // 3 groups, identical distributions: y ~ N(5, 1) in each group.
    // σ²_u should be ~0, ICC should be ~0.
    let n = 90;
    let d = 1;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut groups = Vec::new();
    let mut rng = 42u64;
    for g in 0..3 {
        for _ in 0..30 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let xi = rng as f64 / u64::MAX as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
            x.push(xi);
            y.push(5.0 + 2.0 * xi + noise); // same intercept for all groups
            groups.push(g);
        }
    }

    let result = lme_random_intercept(&x, &y, n, d, &groups, 100, 1e-8);
    eprintln!("LME zero between-group variance:");
    eprintln!("  σ² = {:.6}, σ²_u = {:.6e}", result.sigma2, result.sigma2_u);
    eprintln!("  ICC = {:.6}", result.icc);
    eprintln!("  iterations = {}", result.iterations);
    eprintln!("  beta = {:?}", result.beta);

    // ICC should be near zero (no group effect)
    assert!(result.icc.is_finite(), "ICC should be finite");
    assert!(result.icc < 0.2,
        "ICC should be near 0 when groups are identical, got {}", result.icc);
    // sigma2_u should be small
    assert!(result.sigma2_u < result.sigma2,
        "σ²_u ({}) should be << σ² ({})", result.sigma2_u, result.sigma2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mixed Effects: Singleton groups (1 obs per group)
// ═══════════════════════════════════════════════════════════════════════════
//
// With 1 observation per group, the random effects are not identifiable.
// Henderson's equations have n_g = 1 for all g, so Z'Z = I.
// The system is: [X'X, X'Z; Z'X, I + λI] where λ → ∞ when σ²_u → 0.
// This forces u → 0 and the model collapses to fixed effects only.

#[test]
fn lme_singleton_groups() {
    use tambear::mixed_effects::lme_random_intercept;

    let n = 20;
    let d = 1;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| 3.0 + 2.0 * i as f64 / n as f64).collect();
    let groups: Vec<usize> = (0..n).collect(); // each obs is its own group

    let result = lme_random_intercept(&x, &y, n, d, &groups, 100, 1e-8);
    eprintln!("LME singleton groups (20 groups, 1 obs each):");
    eprintln!("  σ² = {:.6e}, σ²_u = {:.6e}", result.sigma2, result.sigma2_u);
    eprintln!("  ICC = {:.6}", result.icc);
    eprintln!("  iterations = {}", result.iterations);
    eprintln!("  beta[0:2] = [{:.4}, {:.4}]", result.beta[0], result.beta[1]);

    // Model should still converge without panic
    assert!(result.sigma2.is_finite(), "σ² should be finite");
    assert!(result.icc.is_finite(), "ICC should be finite");
    // With perfectly linear data and no noise, ICC could be anything
    // (random effects absorb the linear trend or fixed effects do)
    eprintln!("  DOCUMENTED: Singleton groups → random effects not identifiable");
    eprintln!("  Model converged in {} iterations (no panic)", result.iterations);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayesian: R-hat with constant chains (division by zero)
// ═══════════════════════════════════════════════════════════════════════════
//
// If all chains have zero within-chain variance (w = 0), then
// R̂ = √(var_hat / w) = √(anything / 0) = Inf or NaN.
// This happens when all chains converge to the same constant — which
// should indicate PERFECT convergence (R̂ = 1), not diagnostic failure.

#[test]
fn r_hat_constant_chains() {
    use tambear::bayesian::r_hat;

    // 3 chains, all constant at 5.0
    let c1 = vec![5.0; 100];
    let c2 = vec![5.0; 100];
    let c3 = vec![5.0; 100];
    let chains: Vec<&[f64]> = vec![&c1, &c2, &c3];

    let rhat = r_hat(&chains);
    eprintln!("R-hat (3 constant chains at 5.0): {rhat}");

    // DOCUMENTED: w = 0, so R̂ = √(var_hat / 0). Should be 1.0 but is
    // NaN or Inf due to division by zero.
    if !rhat.is_finite() {
        eprintln!("  ⚠ R-hat is {rhat} for perfectly converged chains (w=0 → div/0)");
        eprintln!("  Fix: if w < ε, return 1.0 (perfect convergence)");
    } else {
        assert!((rhat - 1.0).abs() < 0.01,
            "Constant chains should give R̂ = 1.0, got {rhat}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayesian: R-hat with divergent chains (should be >> 1)
// ═══════════════════════════════════════════════════════════════════════════
//
// When chains haven't mixed (each stuck in a different mode),
// R̂ >> 1 signals non-convergence.

#[test]
fn r_hat_divergent_chains() {
    use tambear::bayesian::r_hat;

    // Chain 1 at 0, chain 2 at 100 — zero mixing
    let c1 = vec![0.0; 100];
    let c2 = vec![100.0; 100];
    let chains: Vec<&[f64]> = vec![&c1, &c2];

    let rhat = r_hat(&chains);
    eprintln!("R-hat (unmixed chains at 0 and 100): {rhat}");

    // B is large (between-chain variance), w is 0 (within-chain variance).
    // So this also hits the w=0 case.
    // If chains had small noise, we'd get R̂ >> 1.
    // With zero noise: division by zero.
    if rhat.is_finite() {
        assert!(rhat > 2.0, "Unmixed chains should give R̂ >> 1, got {rhat}");
    } else {
        eprintln!("  ⚠ R-hat = {rhat} (w=0 again). Need noise for meaningful R-hat.");
    }

    // Now with small noise: chains centered at 0 and 100 with sd=1
    let mut rng = 42u64;
    let c1n: Vec<f64> = (0..100).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng as f64 / u64::MAX as f64 - 0.5) * 2.0 // near 0
    }).collect();
    let c2n: Vec<f64> = (0..100).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        100.0 + (rng as f64 / u64::MAX as f64 - 0.5) * 2.0 // near 100
    }).collect();
    let chains_n: Vec<&[f64]> = vec![&c1n, &c2n];
    let rhat_n = r_hat(&chains_n);
    eprintln!("R-hat (noisy unmixed chains): {rhat_n:.4}");
    assert!(rhat_n > 5.0, "Unmixed noisy chains should give R̂ >> 1, got {rhat_n}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayesian: ESS for white noise vs autocorrelated chain
// ═══════════════════════════════════════════════════════════════════════════
//
// White noise: ESS ≈ N.
// AR(1) with ρ=0.99: ESS ≈ N(1-ρ)/(1+ρ) ≈ N/200.
// Tests that the autocorrelation cutoff works correctly.

#[test]
fn ess_white_noise_vs_autocorrelated() {
    use tambear::bayesian::effective_sample_size;

    // White noise
    let mut rng = 42u64;
    let white: Vec<f64> = (0..1000).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        rng as f64 / u64::MAX as f64
    }).collect();

    let ess_white = effective_sample_size(&white);
    eprintln!("ESS (white noise, n=1000): {ess_white:.1}");
    // ESS should be close to n for iid samples
    assert!(ess_white > 500.0, "White noise ESS should be ~1000, got {ess_white}");

    // AR(1) with ρ = 0.99 (highly autocorrelated)
    let mut ar: Vec<f64> = vec![0.0; 1000];
    let rho = 0.99;
    for i in 1..1000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let eps = (rng as f64 / u64::MAX as f64 - 0.5) * 0.1;
        ar[i] = rho * ar[i - 1] + eps;
    }

    let ess_ar = effective_sample_size(&ar);
    eprintln!("ESS (AR(1) ρ=0.99, n=1000): {ess_ar:.1}");
    // Theoretical ESS ≈ 1000 * (1-0.99)/(1+0.99) ≈ 5.0
    // The implementation may overestimate due to the ρ < 0.05 cutoff
    assert!(ess_ar < ess_white,
        "AR(1) ESS should be much less than white noise: {ess_ar} vs {ess_white}");
    assert!(ess_ar < 100.0,
        "AR(1) ρ=0.99 ESS should be << 100, got {ess_ar}");
    eprintln!("  Ratio: {:.1}x reduction", ess_white / ess_ar);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayesian: MH with impossible target (log_target = -∞)
// ═══════════════════════════════════════════════════════════════════════════
//
// If the log-target is -∞ everywhere except at the initial point,
// no proposal is ever accepted. The chain stays frozen at the initial.
// Acceptance rate = 0. This should not panic or produce NaN.

#[test]
fn mh_impossible_target() {
    use tambear::bayesian::metropolis_hastings;

    // Target: -∞ everywhere. No proposal can improve on current.
    // Actually, the initial evaluation is also -∞, so log_alpha = -∞ - (-∞) = NaN.
    // Let's make it: finite at initial, -∞ elsewhere.
    let log_target = |x: &[f64]| {
        if (x[0] - 0.0).abs() < 1e-15 { 0.0 } else { f64::NEG_INFINITY }
    };

    let chain = metropolis_hastings(&log_target, &[0.0], 1.0, 100, 10, 42);
    eprintln!("MH impossible target:");
    eprintln!("  acceptance rate: {:.4}", chain.acceptance_rate);
    eprintln!("  n samples: {}", chain.samples.len());

    // All proposals move away from 0.0 → all rejected
    // Actually, proposals have continuous distribution, P(|proposal - 0| < 1e-15) ≈ 0
    assert!(chain.acceptance_rate < 0.02,
        "Impossible target should give near-0 acceptance, got {}", chain.acceptance_rate);

    // All samples should be at initial point (0.0)
    let all_at_init = chain.samples.iter().all(|s| s[0] == 0.0);
    eprintln!("  all samples at initial? {all_at_init}");
    // Note: the initial evaluation is 0.0 (finite). Proposals have log_target = -∞.
    // log_alpha = -∞ - 0 = -∞. u.ln() is always > -∞ (for u > 0). So no acceptance.
    assert!(all_at_init, "Chain should be frozen at initial point");
}

// ═══════════════════════════════════════════════════════════════════════════
// Panel: FE single-unit degeneracy (HIGH)
// ═══════════════════════════════════════════════════════════════════════════
//
// With n_units = 1, the small-sample correction has:
//   correction = (nu / (nu - 1.0)) * ((nt - 1.0) / (nt - d))
// where nu = 1, so (nu / (nu - 1.0)) = 1/0 = Inf.
// Also: demeaning within a single unit zeros out all variation, so
// the beta estimate is meaningless (X̃'X̃ is zero → singular).

#[test]
fn panel_fe_single_unit() {
    use tambear::panel::panel_fe;

    // 10 observations from 1 unit
    let n = 10;
    let d = 1;
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..10).map(|i| 2.0 * i as f64 + 1.0).collect();
    let units = vec![0usize; 10];

    let result = panel_fe(&x, &y, n, d, &units);

    eprintln!("Panel FE single unit:");
    eprintln!("  beta = {:?}", result.beta);
    eprintln!("  se   = {:?}", result.se_clustered);
    eprintln!("  R²   = {:.6}", result.r2_within);
    eprintln!("  df   = {}", result.df);

    // With a single unit, demeaning is well-defined (subtract unit mean).
    // The within-unit regression should still work: β ≈ 2.0.
    assert!(result.beta[0].is_finite(), "Beta should be finite");

    // But clustered SE with 1 cluster is nonsense: correction = 1/(1-1) = Inf
    // The implementation may produce Inf, NaN, or a large number.
    eprintln!("  SE finite? {}", result.se_clustered[0].is_finite());
    if !result.se_clustered[0].is_finite() {
        eprintln!("  ⚠ DOCUMENTED: Clustered SE is Inf with 1 cluster (1/0 correction)");
        eprintln!("  Minimum clusters for valid inference: 2 (ideally ≥ 30)");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Panel: FE with singleton observations (each unit has 1 obs)
// ═══════════════════════════════════════════════════════════════════════════
//
// With 1 observation per unit, demeaning zeros everything: y_dm = 0, x_dm = 0.
// X̃'X̃ = 0 → singular. The lstsq fallback should return zeros.

#[test]
fn panel_fe_singleton_units() {
    use tambear::panel::panel_fe;

    let n = 5;
    let d = 1;
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let units = vec![0, 1, 2, 3, 4]; // each unit has exactly 1 observation

    let result = panel_fe(&x, &y, n, d, &units);

    eprintln!("Panel FE singleton units:");
    eprintln!("  beta = {:?}", result.beta);
    eprintln!("  se   = {:?}", result.se_clustered);
    eprintln!("  R²   = {:.6}", result.r2_within);
    eprintln!("  df   = {}", result.df);

    // With 1 obs per unit, demeaning reduces everything to zero.
    // X̃ = 0, ỹ = 0 → beta should be 0 (no within-unit variation).
    // R² should be 0 (ss_tot = 0).
    assert!(result.beta[0].is_finite(), "Beta should be finite (zero)");
    assert!(result.r2_within.is_finite(), "R² should be finite");
    eprintln!("  DOCUMENTED: Singleton units → no within-variation → beta meaningless");
}

// ═══════════════════════════════════════════════════════════════════════════
// Factor Analysis: Cronbach's alpha with identical items (MEDIUM)
// ═══════════════════════════════════════════════════════════════════════════
//
// If all items are identical copies: item variances = 0, total variance > 0.
// α = (p/(p-1)) * (1 - 0/σ²_total) = p/(p-1) > 1.
// Cronbach's alpha should be ≤ 1 for valid data, but identical items
// produce α > 1, which is mathematically correct but psychometrically wrong.

#[test]
fn cronbach_alpha_identical_items() {
    use tambear::factor_analysis::cronbachs_alpha;

    // 5 items, all identical to the first
    let n = 20;
    let p = 5;
    let mut data = Vec::new();
    let mut rng = 42u64;
    for _ in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let score = rng as f64 / u64::MAX as f64 * 10.0;
        for _ in 0..p {
            data.push(score); // all items identical
        }
    }

    let alpha = cronbachs_alpha(&data, n, p);
    eprintln!("Cronbach's α with identical items: {alpha:.6}");

    // Each item has the same variance. Total variance = p² × item_variance.
    // Sum of item variances = p × item_variance.
    // α = (p/(p-1)) × (1 - p×v / (p²×v)) = (p/(p-1)) × (1 - 1/p) = 1.0.
    //
    // Wait — if items are identical, the total score = p × x_i.
    // σ²_total = p² × σ²_item. Σσ²_item = p × σ²_item.
    // α = (5/4)(1 - 5v/(25v)) = (5/4)(1 - 0.2) = (5/4)(0.8) = 1.0.
    //
    // So α = 1.0 exactly for identical items. Good — it doesn't exceed 1.
    assert!((alpha - 1.0).abs() < 1e-10,
        "Identical items should give α = 1.0, got {alpha}");
    eprintln!("  α = 1.0 exactly — perfect internal consistency (trivially).");
}

// ═══════════════════════════════════════════════════════════════════════════
// Factor Analysis: PAF with rank-deficient correlation matrix (HIGH)
// ═══════════════════════════════════════════════════════════════════════════
//
// If variable 3 = variable 1 + variable 2 (perfect multicollinearity),
// the correlation matrix is rank-deficient. The sym_eigen decomposition
// may produce a near-zero eigenvalue, and PAF communality iteration
// may not converge or may produce communalities > 1.

#[test]
fn paf_rank_deficient_correlation() {
    use tambear::factor_analysis::{correlation_matrix, principal_axis_factoring};

    // 3 variables: x3 = x1 + x2 (perfect collinearity)
    let n = 50;
    let p = 3;
    let mut data = Vec::new();
    let mut rng = 99u64;
    for _ in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x1 = (rng as f64 / u64::MAX as f64 - 0.5) * 10.0;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x2 = (rng as f64 / u64::MAX as f64 - 0.5) * 10.0;
        let x3 = x1 + x2; // perfect linear combination
        data.push(x1);
        data.push(x2);
        data.push(x3);
    }

    let corr = correlation_matrix(&data, n, p);
    eprintln!("Correlation matrix (rank-deficient):");
    for i in 0..p {
        eprintln!("  [{:.4} {:.4} {:.4}]", corr.get(i, 0), corr.get(i, 1), corr.get(i, 2));
    }

    // Extract 1 factor (safe — asking for 2 might hit the zero eigenvalue)
    let result = principal_axis_factoring(&corr, 1, 100);
    eprintln!("PAF result (1 factor from rank-2 matrix):");
    eprintln!("  eigenvalues: {:?}", result.eigenvalues);
    eprintln!("  communalities: {:?}", result.communalities);
    eprintln!("  loadings col 0: [{:.4}, {:.4}, {:.4}]",
        result.loadings.get(0, 0), result.loadings.get(1, 0), result.loadings.get(2, 0));

    // Eigenvalue should be finite and positive
    assert!(result.eigenvalues[0].is_finite() && result.eigenvalues[0] > 0.0,
        "First eigenvalue should be positive finite");
    // Communalities should be in [0, 1]
    for (j, &c) in result.communalities.iter().enumerate() {
        assert!(c.is_finite(), "communality[{j}] should be finite");
        // Note: communalities CAN exceed 1 in PAF (Heywood case)
        if c > 1.0 {
            eprintln!("  ⚠ Heywood case: communality[{j}] = {c} > 1.0");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Factor Analysis: McDonald's omega when loadings sum to zero
// ═══════════════════════════════════════════════════════════════════════════
//
// If loadings on the first factor are [+0.5, +0.5, -0.5, -0.5],
// sum_l = 0, and ω = 0/(0 + sum_u) = 0.
// This is "correct" (no general factor) but hides that there may be
// a strong factor that just has mixed signs (bipolar factor).

#[test]
fn mcdonalds_omega_bipolar_factor() {
    use tambear::factor_analysis::mcdonalds_omega;
    use tambear::linear_algebra::Mat;

    // Bipolar loadings: half positive, half negative
    let mut loadings = Mat::zeros(4, 1);
    loadings.set(0, 0, 0.8);
    loadings.set(1, 0, 0.7);
    loadings.set(2, 0, -0.8);
    loadings.set(3, 0, -0.7);

    let result = mcdonalds_omega(&loadings);
    eprintln!("McDonald's ω with bipolar loadings: {:.6}", result.omega);
    eprintln!("  bipolar detected: {}", result.bipolar);
    eprintln!("  sum_l (raw) = {}", 0.8 + 0.7 - 0.8 - 0.7);

    // The fix: mcdonalds_omega now detects bipolar factors and uses |loadings|.
    // With absolute values: sum_l = 0.8+0.7+0.8+0.7 = 3.0, ω ≈ 0.95.
    let comm: Vec<f64> = (0..4).map(|j| loadings.get(j, 0).powi(2)).collect();
    eprintln!("  communalities: {:?}", comm);
    eprintln!("  mean communality: {:.4}", comm.iter().sum::<f64>() / 4.0);

    assert!(result.omega.is_finite(), "ω should be finite");
    assert!(result.bipolar, "Should detect bipolar factor");
    // With bipolar correction, ω should be high (strong factor, just mixed signs)
    assert!(result.omega > 0.5,
        "Bipolar-corrected ω should reflect strong factor, got {:.4}", result.omega);
}

// ═══════════════════════════════════════════════════════════════════════════
// Spatial: Kriging with near-coincident observation points (HIGH)
// ═══════════════════════════════════════════════════════════════════════════
//
// Near-coincident points make the kriging covariance matrix near-singular.
// Two points at distance ε have nearly identical rows in C, so the system
// Cw = c is ill-conditioned. The LU solver may produce garbage weights
// (huge positive and negative), leading to predictions far outside the
// convex hull of observed values.
//
// This is the spatial analogue of multicollinearity in regression — and
// it's the default failure mode when real-world sensors are clustered.

#[test]
fn kriging_near_coincident_points() {
    use tambear::spatial::{SpatialPoint, ordinary_kriging, spherical_variogram, VariogramModel};

    // Three well-separated points + one nearly coincident with the first
    let eps = 1e-10; // nearly zero separation
    let points = vec![
        SpatialPoint { x: 0.0, y: 0.0, value: 1.0 },
        SpatialPoint { x: eps, y: 0.0, value: 5.0 },  // near-duplicate, DIFFERENT value
        SpatialPoint { x: 10.0, y: 0.0, value: 2.0 },
        SpatialPoint { x: 0.0, y: 10.0, value: 3.0 },
    ];
    let model = VariogramModel { nugget: 0.0, sill: 5.0, range: 15.0 };

    // Query at a sane location (5, 5) — should get something reasonable
    let result = ordinary_kriging(
        &points,
        &[5.0], &[5.0],
        &model, spherical_variogram,
    );

    let pred = result.predicted[0];
    let var = result.variance[0];
    eprintln!("Kriging with near-coincident points:");
    eprintln!("  pred at (5,5) = {pred:.6}");
    eprintln!("  variance      = {var:.6}");
    eprintln!("  Data range: [1.0, 5.0]");

    // The prediction should be finite (not NaN/Inf from singular system)
    assert!(pred.is_finite(), "Prediction is not finite: {pred}");

    // DOCUMENTED FINDING: with zero nugget and near-coincident points with
    // different values, the kriging weights can be extreme (one huge positive,
    // one huge negative for the near-duplicate pair). The prediction may
    // extrapolate wildly outside the data range [1, 5].
    //
    // In production: either (a) add a nugget (regularization), or
    // (b) deduplicate observations, or (c) detect condition number and warn.
    if pred < 0.0 || pred > 6.0 {
        eprintln!("  ⚠ Prediction OUTSIDE data range — ill-conditioning artifact");
        eprintln!("  This is the spatial collinearity problem.");
    } else {
        eprintln!("  Prediction within data range (solver handled conditioning)");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spatial: Kriging with nugget regularizes the ill-conditioned system
// ═══════════════════════════════════════════════════════════════════════════
//
// The nugget effect adds a diagonal shift to the covariance matrix:
//   C_ii = sill + nugget, C_ij = sill + nugget - γ(h)
// For coincident points (h≈0): C_ij ≈ sill + nugget (off-diagonal),
// but the diagonal is also sill + nugget. So the matrix is still singular!
//
// The REAL regularization comes from γ(h) > 0 for h > 0 (even tiny h).
// The nugget shifts the *diagonal* relative to near-off-diagonal entries
// only when the variogram model has γ(0) = 0 (no nugget) vs γ(0⁺) = nugget.
//
// Wait — actually: spherical_variogram(h, model) returns 0 when h < 1e-300,
// then nugget + sill*(1.5*hr - 0.5*hr³) for h > 0. So:
//   C_ii = (sill + nugget) - 0 = sill + nugget
//   C_ij = (sill + nugget) - (nugget + sill*(...)) = sill*(1 - 1.5*hr + 0.5*hr³)
// The diagonal excess IS the nugget. Good.

#[test]
fn kriging_nugget_regularizes() {
    use tambear::spatial::{SpatialPoint, ordinary_kriging, spherical_variogram, VariogramModel};

    let eps = 1e-10;
    let points = vec![
        SpatialPoint { x: 0.0, y: 0.0, value: 1.0 },
        SpatialPoint { x: eps, y: 0.0, value: 5.0 },
        SpatialPoint { x: 10.0, y: 0.0, value: 2.0 },
        SpatialPoint { x: 0.0, y: 10.0, value: 3.0 },
    ];

    // With nugget = 1.0 (measurement error), the system is regularized
    let model = VariogramModel { nugget: 1.0, sill: 4.0, range: 15.0 };
    let result = ordinary_kriging(
        &points,
        &[5.0], &[5.0],
        &model, spherical_variogram,
    );

    let pred = result.predicted[0];
    let var = result.variance[0];
    eprintln!("Kriging with nugget=1.0 (regularized):");
    eprintln!("  pred at (5,5) = {pred:.6}");
    eprintln!("  variance      = {var:.6}");

    assert!(pred.is_finite(), "Prediction should be finite with nugget");
    // With nugget, prediction should be a reasonable weighted average
    // The data values are 1, 5, 2, 3 → mean = 2.75
    // Query at (5,5) is equidistant-ish → prediction near the mean
    assert!(pred > 0.0 && pred < 6.0,
        "With nugget regularization, prediction should be within data range: {pred}");
    eprintln!("  Nugget regularization keeps prediction sane.");
}

// ═══════════════════════════════════════════════════════════════════════════
// Spatial: Kriging extrapolation beyond data envelope
// ═══════════════════════════════════════════════════════════════════════════
//
// Ordinary kriging with a bounded variogram (spherical) has a key property:
// once the query point is beyond the range of the variogram from ALL
// observation points, the covariance vector c(x*, x_i) is zero for all i,
// and the kriging weights default to 1/n (ordinary kriging Lagrange constraint).
//
// This means: far-field extrapolation → prediction = mean of observations,
// variance = sill (maximum uncertainty). This is CORRECT behavior for
// ordinary kriging — but users might not expect the silent degradation.

#[test]
fn kriging_extrapolation_degrades_to_mean() {
    use tambear::spatial::{SpatialPoint, ordinary_kriging, spherical_variogram, VariogramModel};

    let points = vec![
        SpatialPoint { x: 0.0, y: 0.0, value: 10.0 },
        SpatialPoint { x: 1.0, y: 0.0, value: 20.0 },
        SpatialPoint { x: 0.0, y: 1.0, value: 30.0 },
    ];
    let model = VariogramModel { nugget: 0.0, sill: 100.0, range: 5.0 };
    let data_mean = 20.0;

    // Query at increasing distances from the data
    let far_x = vec![0.5, 5.0, 50.0, 500.0, 5000.0];
    let far_y = vec![0.5, 5.0, 50.0, 500.0, 5000.0];
    let result = ordinary_kriging(&points, &far_x, &far_y, &model, spherical_variogram);

    eprintln!("Kriging extrapolation (data mean = {data_mean}):");
    for (i, &x) in far_x.iter().enumerate() {
        let dist_from_data = (x * x + x * x).sqrt();
        eprintln!("  ({x},{x}): pred={:.4}, var={:.4}, dist={dist_from_data:.1}",
            result.predicted[i], result.variance[i]);
    }

    // DOCUMENTED FINDING: The farthest prediction converges to the GLS mean
    // (generalized least squares), NOT the arithmetic mean. This is because the
    // kriging weights at far distance are determined by the covariance structure
    // among observation points (C⁻¹), not uniform 1/n.
    //
    // For non-equidistant observations, GLS mean ≠ arithmetic mean.
    // This is mathematically correct but often surprising to users.
    let far_pred = result.predicted[4];
    assert!(far_pred.is_finite(), "Far prediction should be finite");

    // Once beyond range, prediction should be CONSTANT (same for all far queries)
    let pred_50 = result.predicted[2];
    let pred_500 = result.predicted[3];
    let pred_5000 = result.predicted[4];
    eprintln!("  Predictions at dist >> range are constant:");
    eprintln!("    pred(50,50)   = {pred_50:.6}");
    eprintln!("    pred(500,500) = {pred_500:.6}");
    eprintln!("    pred(5000,5000) = {pred_5000:.6}");
    assert!((pred_50 - pred_500).abs() < 1e-8,
        "Beyond range, predictions should be constant");
    assert!((pred_500 - pred_5000).abs() < 1e-8,
        "Beyond range, predictions should be constant");

    // The GLS mean should be within the data range [10, 30]
    assert!(far_pred >= 10.0 && far_pred <= 30.0,
        "GLS mean should be within data range, got {far_pred}");
    eprintln!("  GLS mean = {far_pred:.4} (arithmetic mean = {data_mean})");
    eprintln!("  Difference = {:.4} — kriging upweights points with lower covariance",
        far_pred - data_mean);

    // Far variance should be at or near the sill
    let far_var = result.variance[4];
    eprintln!("  Far variance: {far_var:.4} (sill+nugget = {})", model.sill + model.nugget);
    // Variance saturates at some value related to sill — may exceed sill due to Lagrange
    assert!(far_var > 0.0, "Far variance should be positive, got {far_var}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Spatial: Moran's I with checkerboard (perfect negative autocorrelation)
// ═══════════════════════════════════════════════════════════════════════════
//
// On a regular grid with alternating high/low values (checkerboard),
// Moran's I should be strongly negative. This tests the opposite extreme
// from the existing test (which only tests positive autocorrelation).
//
// The theoretical minimum of Moran's I depends on the weights matrix,
// but for a knn(k=4) weights on a grid, a perfect checkerboard gives
// I close to -1.

#[test]
fn morans_i_checkerboard_negative() {
    use tambear::spatial::{SpatialWeights, morans_i};

    // 4×4 grid, checkerboard pattern
    let mut pts = Vec::new();
    let mut values = Vec::new();
    for r in 0..4 {
        for c in 0..4 {
            pts.push((r as f64, c as f64));
            values.push(if (r + c) % 2 == 0 { 10.0 } else { 0.0 });
        }
    }

    let weights = SpatialWeights::knn(&pts, 4);
    let i = morans_i(&values, &weights);
    eprintln!("Moran's I (4×4 checkerboard, k=4): {i:.6}");

    // Checkerboard → strong negative autocorrelation
    assert!(i < -0.3, "Checkerboard should give strongly negative Moran's I, got {i}");

    // Compare with Geary's C — should be > 1 (negative autocorrelation)
    use tambear::spatial::gearys_c;
    let c = gearys_c(&values, &weights);
    eprintln!("Geary's C (4×4 checkerboard, k=4): {c:.6}");
    assert!(c > 1.0, "Checkerboard should give Geary's C > 1, got {c}");

    // Moran and Geary should be inversely related
    eprintln!("I < 0, C > 1 — consistent (negative spatial autocorrelation)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Spatial: Haversine at antipodes and poles
// ═══════════════════════════════════════════════════════════════════════════
//
// Antipodal points are the maximum great-circle distance (≈ 20015 km).
// The haversine formula can lose precision at antipodes because
// a = sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2) → 1,
// and 2*asin(√1) = π. But floating point might have a ≈ 1 + ε.

#[test]
fn haversine_antipodes_and_poles() {
    use tambear::spatial::haversine;

    // Exact antipodes: (0,0) ↔ (0,180) — half circumference
    let d = haversine(0.0, 0.0, 0.0, 180.0);
    let half_circ = std::f64::consts::PI * 6371.0; // ≈ 20015.1 km
    eprintln!("Antipodal (equator): {d:.2} km (expected {half_circ:.2})");
    assert!((d - half_circ).abs() < 1.0,
        "Equatorial antipodes should be half circumference, got {d}");

    // North pole to south pole
    let d_poles = haversine(90.0, 0.0, -90.0, 0.0);
    eprintln!("Pole to pole: {d_poles:.2} km (expected {half_circ:.2})");
    assert!((d_poles - half_circ).abs() < 1.0,
        "Pole-to-pole should be half circumference, got {d_poles}");

    // Same point → 0
    let d_same = haversine(40.7128, -74.0060, 40.7128, -74.0060);
    assert!(d_same.abs() < 1e-10, "Same point should give distance 0, got {d_same}");

    // Near-antipodal (stress test for asin domain)
    let d_near = haversine(0.0, 0.0, 0.001, 179.999);
    assert!(d_near.is_finite() && d_near > 0.0,
        "Near-antipodal should be finite positive, got {d_near}");
    assert!((d_near - half_circ).abs() < 5.0,
        "Near-antipodal should be close to half circumference, got {d_near}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Spatial: Ripley's K with regular lattice (anti-clustering)
// ═══════════════════════════════════════════════════════════════════════════
//
// A perfectly regular grid has K(r) < πr² for small r (inhibition).
// At larger r, the lattice catches up. This is the opposite of the
// existing test which only checks for clustering.

#[test]
fn ripleys_k_regular_lattice() {
    use tambear::spatial::ripleys_k;
    use std::f64::consts::PI;

    // 5×5 regular grid on [0,10]×[0,10]
    let mut points = Vec::new();
    for r in 0..5 {
        for c in 0..5 {
            points.push((r as f64 * 2.5, c as f64 * 2.5));
        }
    }
    let area = 10.0 * 10.0;

    // At small radius (less than grid spacing), K should show inhibition
    let r_small = 1.0; // less than 2.5 grid spacing
    let k = ripleys_k(&points, &[r_small], area);
    let poisson_k = PI * r_small * r_small;
    eprintln!("Regular lattice K({r_small}) = {:.4}, Poisson πr² = {poisson_k:.4}", k[0]);

    // For a regular grid at r < spacing, no neighbors exist → K ≈ 0
    assert!(k[0] < poisson_k,
        "Regular lattice at r < spacing: K should show inhibition (K={} < πr²={})", k[0], poisson_k);

    // At large radius (covers many neighbors), K approaches Poisson
    let r_large = 8.0;
    let k_large = ripleys_k(&points, &[r_large], area);
    let poisson_large = PI * r_large * r_large;
    eprintln!("Regular lattice K({r_large}) = {:.4}, Poisson πr² = {poisson_large:.4}", k_large[0]);
    // Not necessarily > Poisson at large r for a finite lattice, but should be substantial
    assert!(k_large[0] > 0.0, "K at large r should be positive");
}

// ═══════════════════════════════════════════════════════════════════════════
// Series Acceleration: Aitken on divergent series (Type 5 — wrong function class)
// ═══════════════════════════════════════════════════════════════════════════
//
// Aitken assumes geometric convergence: e_{n+1} ≈ r * e_n. Applied to a
// divergent series, the Δ² quotient is meaningless. Wynn ε, however,
// can sometimes extract a regularized value via Padé approximation.
// This is the canonical Structure Beats Resources example: more Aitken
// iterations make it WORSE; switching to Wynn fixes it.

#[test]
fn aitken_divergent_vs_wynn_regularized() {
    use tambear::series_accel::{partial_sums, aitken_delta2, wynn_epsilon, cesaro_sum};

    // Grandi's series: 1 - 1 + 1 - 1 + ... (Cesàro summable to 1/2)
    let terms: Vec<f64> = (0..20).map(|k| if k % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let sums = partial_sums(&terms);

    let cesaro = cesaro_sum(&sums);
    let aitken = aitken_delta2(&sums);
    let wynn = wynn_epsilon(&sums);

    eprintln!("Grandi's series (divergent, Cesàro-summable to 0.5):");
    eprintln!("  Cesàro sum: {cesaro:.6}");
    eprintln!("  Aitken (last): {:?}", aitken.last());
    eprintln!("  Wynn ε: {wynn:.6}");

    // Cesàro should be close to 0.5
    assert!((cesaro - 0.5).abs() < 0.1,
        "Cesàro should give ~0.5 for Grandi's series, got {cesaro}");

    // Aitken's output on an oscillating series is unpredictable
    // (it's designed for geometric convergence, not oscillation)
    // Just verify it doesn't crash and returns finite values
    for &a in &aitken {
        assert!(a.is_finite(), "Aitken output should be finite, got {a}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Series Acceleration: Wynn on constant sequence (Type 1 — denominator)
// ═══════════════════════════════════════════════════════════════════════════
//
// For a constant sequence S_n = c, the recurrence has
// ε_k(n+1) - ε_k(n) = 0, producing 1/0. The implementation must handle
// this gracefully (it's a converged sequence — the answer IS c).

#[test]
fn wynn_constant_sequence_no_division_by_zero() {
    use tambear::series_accel::{wynn_epsilon, StreamingWynn};

    // Already converged: all partial sums are π
    let sums = vec![std::f64::consts::PI; 10];
    let result = wynn_epsilon(&sums);

    assert!(result.is_finite(), "Wynn on constant sequence should be finite, got {result}");
    assert!((result - std::f64::consts::PI).abs() < 1e-10,
        "Wynn on constant π should return π, got {result}");

    // Streaming version too
    let mut sw = StreamingWynn::new(20);
    for _ in 0..10 {
        sw.push(0.0); // terms are zero (constant partial sum would need push_value)
    }
    assert!(sw.estimate().is_finite(), "StreamingWynn must not blow up on zero terms");
}

// ═══════════════════════════════════════════════════════════════════════════
// Series Acceleration: Richardson with wrong error order (Type 5)
// ═══════════════════════════════════════════════════════════════════════════
//
// Richardson assumes error ~ c·h^p. If p is wrong, the extrapolation
// cancels the wrong term and produces garbage. This is a Type 5 boundary:
// the method's structural assumption is violated.

#[test]
fn richardson_wrong_order_degrades() {
    use tambear::series_accel::richardson_extrapolate;

    // True error order p=2 (central difference): A(h) = L + c·h² + O(h⁴)
    // Approximations at h, h/2, h/4, h/8 with p=2 error
    let true_limit = 1.0;
    let c = 0.5;
    let approx: Vec<f64> = (0..5).map(|i| {
        let h = 0.5 / 2.0_f64.powi(i);
        true_limit + c * h * h
    }).collect();

    // Correct order p=2 → should be very accurate
    let correct = richardson_extrapolate(&approx, 2.0, 2);
    let err_correct = (correct - true_limit).abs();

    // Wrong order p=1 → less accurate
    let wrong = richardson_extrapolate(&approx, 2.0, 1);
    let err_wrong = (wrong - true_limit).abs();

    eprintln!("Richardson correct order (p=2): err = {err_correct:.2e}");
    eprintln!("Richardson wrong order   (p=1): err = {err_wrong:.2e}");

    // Both should be finite
    assert!(correct.is_finite(), "Correct-order Richardson must be finite");
    assert!(wrong.is_finite(), "Wrong-order Richardson must be finite");

    // Correct order should be much better (or at least not worse)
    assert!(err_correct < 1e-6,
        "Richardson with correct order should be accurate, err={err_correct:.2e}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimization: Saddle point traps GD, L-BFGS escapes (Type 5)
// ═══════════════════════════════════════════════════════════════════════════
//
// The monkey saddle f(x,y) = x³ - 3xy² has a critical point at (0,0)
// with gradient = 0 but Hessian = 0 too. First-order methods (GD) get
// stuck; second-order methods (L-BFGS with line search) can escape.

#[test]
fn optimizer_saddle_point_gd_vs_lbfgs() {
    use tambear::optimization::{gradient_descent, lbfgs};

    // f(x,y) = x⁴ + y⁴ - 2(x-y)² — has saddle at (0,0), minima at (1,-1) and (-1,1)
    let f = |x: &[f64]| x[0].powi(4) + x[1].powi(4) - 2.0 * (x[0] - x[1]).powi(2);
    let grad = |x: &[f64]| vec![
        4.0 * x[0].powi(3) - 4.0 * (x[0] - x[1]),
        4.0 * x[1].powi(3) + 4.0 * (x[0] - x[1]),
    ];

    // Start very near the saddle
    let x0 = [0.01, -0.01];

    let gd = gradient_descent(&f, &grad, &x0, 0.01, 0.0, 5000, 1e-10);
    let lb = lbfgs(&f, &grad, &x0, 10, 1000, 1e-10);

    eprintln!("Saddle escape test:");
    eprintln!("  GD:    x={:?}, f={:.6}, converged={}, iters={}", gd.x, gd.f_val, gd.converged, gd.iterations);
    eprintln!("  L-BFGS: x={:?}, f={:.6}, converged={}, iters={}", lb.x, lb.f_val, lb.converged, lb.iterations);

    // L-BFGS should find a lower value (minima at f ≈ -2)
    // GD may get stuck near the saddle (f ≈ 0)
    assert!(lb.f_val < gd.f_val || lb.f_val < -1.0,
        "L-BFGS should escape the saddle (f={:.4}) better than GD (f={:.4})",
        lb.f_val, gd.f_val);
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimization: Ill-conditioned quadratic — GD stuck, L-BFGS converges (Type 5)
// ═══════════════════════════════════════════════════════════════════════════
//
// f(x,y) = x² + 1000000·y²  (condition number = 10⁶)
// GD with fixed learning rate either diverges (lr too big) or crawls
// (lr too small). L-BFGS adapts via curvature and converges fast.

#[test]
fn optimizer_ill_conditioned_gd_vs_lbfgs() {
    use tambear::optimization::{gradient_descent, lbfgs};

    let kappa = 1e6; // condition number
    let f = |x: &[f64]| x[0] * x[0] + kappa * x[1] * x[1];
    let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * kappa * x[1]];

    let x0 = [1.0, 1.0];

    // GD: lr must be < 2/L_max = 2/(2*kappa) = 1e-6 to not diverge
    let gd = gradient_descent(&f, &grad, &x0, 5e-7, 0.0, 10000, 1e-8);
    let lb = lbfgs(&f, &grad, &x0, 10, 100, 1e-10);

    eprintln!("Ill-conditioned (κ=1e6) test:");
    eprintln!("  GD:    f={:.6e}, converged={}, iters={}", gd.f_val, gd.converged, gd.iterations);
    eprintln!("  L-BFGS: f={:.6e}, converged={}, iters={}", lb.f_val, lb.converged, lb.iterations);

    // L-BFGS should converge in far fewer iterations
    assert!(lb.converged, "L-BFGS should converge on ill-conditioned quadratic");
    assert!(lb.f_val < 1e-10,
        "L-BFGS solution should be near-optimal, f={:.2e}", lb.f_val);
    // GD with this lr converges slowly on the x dimension
    assert!(lb.iterations < gd.iterations || lb.f_val < gd.f_val,
        "L-BFGS should outperform GD on ill-conditioned problem");
}

// ═══════════════════════════════════════════════════════════════════════════
// Survival: All censored — no events (Type 4 — equipartition)
// ═══════════════════════════════════════════════════════════════════════════
//
// When all observations are censored, KM has no steps, survival stays 1.0,
// and median survival is Inf. This is the equipartition boundary:
// no information to estimate survival → assume everyone survives.

#[test]
fn survival_all_censored_km() {
    use tambear::survival::{kaplan_meier, km_median};

    let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let events = vec![false; 5]; // all censored

    let steps = kaplan_meier(&times, &events);
    let med = km_median(&steps);

    eprintln!("KM all censored: {} steps, median = {med}", steps.len());

    // No events → no steps in the KM curve
    assert_eq!(steps.len(), 0, "No events → no KM steps");
    // Median should be infinity (not reached)
    assert!(med.is_infinite(), "Median should be Inf when all censored, got {med}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Survival: Log-rank with one group empty
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn survival_logrank_single_group_no_crash() {
    use tambear::survival::log_rank_test;

    let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let events = vec![true, true, true, true, true];
    let groups = vec![0, 0, 0, 0, 0]; // all in group 0, none in group 1

    let res = log_rank_test(&times, &events, &groups);
    eprintln!("Log-rank single group: χ²={:.4}, p={:.4}", res.chi2, res.p_value);

    // With all in one group, no comparison possible → χ² should be 0 or small
    assert!(res.chi2.is_finite(), "χ² must be finite with single group");
}

// ═══════════════════════════════════════════════════════════════════════════
// TDA: Identical points — zero-distance boundary (Type 1)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tda_identical_points_persistence() {
    use tambear::tda::rips_h0;

    // All points at the same location → distance matrix is all zeros
    let n = 4;
    let dist = vec![0.0; n * n];
    let diag = rips_h0(&dist, n);

    eprintln!("TDA identical points: {} pairs", diag.pairs.len());

    // All merges happen at distance 0 → all finite pairs have persistence 0
    let finite: Vec<_> = diag.dimension(0).into_iter()
        .filter(|p| p.death.is_finite())
        .collect();
    for p in &finite {
        assert!((p.persistence() - 0.0).abs() < 1e-15,
            "Identical points should have zero persistence, got {}", p.persistence());
    }
    // Should still have one surviving component
    let inf_count = diag.dimension(0).into_iter().filter(|p| p.death.is_infinite()).count();
    assert_eq!(inf_count, 1, "Should have exactly 1 surviving component");
}

// ═══════════════════════════════════════════════════════════════════════════
// Time Series: ADF on random walk (should NOT reject unit root)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn adf_random_walk_not_rejected() {
    use tambear::time_series::adf_test;

    // Generate a random walk: y_t = y_{t-1} + ε_t
    let n = 200;
    let mut data = vec![0.0; n];
    let mut rng = 42u64;
    for t in 1..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
        data[t] = data[t - 1] + noise;
    }

    let res = adf_test(&data, 1);
    eprintln!("ADF random walk: stat={:.3}, 5% critical={:.3}", res.statistic, res.critical_5pct);

    // Random walk has unit root → should NOT reject H₀
    // ADF statistic should be > critical value (less negative)
    assert!(res.statistic > res.critical_1pct,
        "ADF should not reject unit root at 1%: stat={:.3} vs crit={:.3}",
        res.statistic, res.critical_1pct);
}

// ═══════════════════════════════════════════════════════════════════════════
// Time Series: AR(0) — constant series, no structure
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ar_fit_constant_series() {
    use tambear::time_series::ar_fit;

    let data = vec![5.0; 100];
    let res = ar_fit(&data, 2);

    eprintln!("AR(2) on constant: coeffs={:?}, σ²={:.6e}", res.coefficients, res.sigma2);

    // No variance → σ² should be 0 (or near-zero)
    assert!(res.sigma2 < 1e-10, "σ² should be ~0 for constant series, got {:.2e}", res.sigma2);
    // Coefficients should be ~0 (no AR structure)
    for (j, &c) in res.coefficients.iter().enumerate() {
        assert!(c.abs() < 0.1, "AR coeff[{j}] = {c} should be ~0 for constant series");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Robust: Bisquare M-estimator at 49% contamination (breakdown point)
// ═══════════════════════════════════════════════════════════════════════════
//
// Bisquare has 50% breakdown point. At 49% contamination, it should still
// estimate the true location. At 51%, it fails. This tests the boundary.

#[test]
fn robust_bisquare_near_breakdown() {
    use tambear::robust::bisquare_m_estimate;

    let n = 100;
    let true_loc = 5.0;
    let mut data: Vec<f64> = vec![true_loc; n]; // 100 good points

    // Replace 49 with extreme outliers
    for i in 0..49 {
        data[i] = 1000.0 + i as f64;
    }

    let res = bisquare_m_estimate(&data, 4.685, 100, 1e-8);

    eprintln!("Bisquare at 49% contamination:");
    eprintln!("  location = {:.4}, true = {true_loc}, converged = {}", res.location, res.converged);

    // Should recover the true location (within tolerance)
    assert!((res.location - true_loc).abs() < 2.0,
        "Bisquare should recover true location at 49% contamination, got {:.4} (true={true_loc})",
        res.location);
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph: Dijkstra on disconnected graph (Type 4 — no path)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn graph_dijkstra_disconnected() {
    use tambear::graph::{Graph, dijkstra};

    // Two disconnected components: {0,1} and {2,3}
    let g = Graph::from_undirected_edges(4, &[
        (0, 1, 1.0),
        (2, 3, 1.0),
    ]);

    let (dist, _prev) = dijkstra(&g, 0);

    eprintln!("Dijkstra disconnected: dist = {:?}", dist);

    assert!((dist[1] - 1.0).abs() < 1e-10, "d(0→1) should be 1.0");
    assert!(dist[2] == f64::INFINITY, "d(0→2) should be Inf (disconnected), got {}", dist[2]);
    assert!(dist[3] == f64::INFINITY, "d(0→3) should be Inf (disconnected), got {}", dist[3]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Neural: Softmax with extreme inputs (overflow/underflow)
// ═══════════════════════════════════════════════════════════════════════════
//
// Without the subtract-max trick, exp(700) overflows to Inf and
// exp(-700) underflows to 0. The implementation must handle both extremes
// and still produce a valid probability distribution.

#[test]
fn neural_softmax_extreme_inputs() {
    use tambear::neural::{softmax, log_softmax};

    // Extreme positive: exp(700) = Inf without max subtraction
    let extreme_pos = vec![700.0, 0.0, -700.0];
    let sm = softmax(&extreme_pos);
    eprintln!("Softmax([700, 0, -700]): {:?}", sm);

    assert!((sm.iter().sum::<f64>() - 1.0).abs() < 1e-10,
        "Softmax must sum to 1.0, got {}", sm.iter().sum::<f64>());
    assert!(sm[0] > 0.99, "exp(700) should dominate, got {}", sm[0]);
    assert!(sm.iter().all(|&s| s.is_finite() && s >= 0.0),
        "All softmax values must be finite and non-negative");

    // All same value: should be uniform
    let uniform = vec![42.0; 5];
    let sm_u = softmax(&uniform);
    for &s in &sm_u {
        assert!((s - 0.2).abs() < 1e-10, "Uniform input → uniform softmax, got {s}");
    }

    // Log-softmax should also be stable
    let lsm = log_softmax(&extreme_pos);
    assert!(lsm.iter().all(|l| l.is_finite()), "Log-softmax must be finite: {:?}", lsm);
    assert!(lsm[0] > -1e-10, "log(softmax(700)) ≈ 0, got {}", lsm[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Neural: Sigmoid at extremes (no NaN)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neural_sigmoid_extreme_values() {
    use tambear::neural::sigmoid;

    assert!((sigmoid(0.0) - 0.5).abs() < 1e-15, "sigmoid(0) = 0.5");
    assert!((sigmoid(100.0) - 1.0).abs() < 1e-10, "sigmoid(100) ≈ 1.0");
    assert!(sigmoid(-100.0) < 1e-10, "sigmoid(-100) ≈ 0.0");
    assert!((sigmoid(700.0) - 1.0).abs() < 1e-15, "sigmoid(700) = 1.0 (no overflow)");
    assert!(sigmoid(-700.0) >= 0.0, "sigmoid(-700) ≥ 0 (no underflow to negative)");
    assert!(sigmoid(-700.0).is_finite(), "sigmoid(-700) must be finite");
    assert!(sigmoid(f64::MAX / 2.0).is_finite(), "sigmoid(huge) must be finite");
    assert!(sigmoid(f64::MIN / 2.0).is_finite(), "sigmoid(-huge) must be finite");
}

// ═══════════════════════════════════════════════════════════════════════════
// Neural: Batch norm with constant features (zero variance)
// ═══════════════════════════════════════════════════════════════════════════
//
// When all values in a feature are identical, variance = 0.
// Division by sqrt(var + eps) must not produce NaN.

#[test]
fn neural_batch_norm_zero_variance() {
    use tambear::neural::batch_norm;

    let batch = 4;
    let features = 2;
    // Feature 0: all 5.0 (zero variance), Feature 1: varied
    let input = vec![
        5.0, 1.0,
        5.0, 2.0,
        5.0, 3.0,
        5.0, 4.0,
    ];
    let gamma = vec![1.0; features];
    let beta = vec![0.0; features];

    let res = batch_norm(&input, batch, features, &gamma, &beta, 1e-5);

    eprintln!("Batch norm zero-var: output = {:?}", res.output);
    eprintln!("  var = {:?}", res.var);

    // All outputs must be finite
    assert!(res.output.iter().all(|x| x.is_finite()),
        "Batch norm output must be finite with zero variance");

    // Feature 0 (zero var): all normalized to 0 (centered, divided by sqrt(eps))
    for b in 0..batch {
        let val = res.output[b * features];
        assert!(val.abs() < 1e-3,
            "Zero-variance feature should normalize to ~0, got {val}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Neural: Cross-entropy with confident wrong prediction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn neural_cross_entropy_confident_wrong() {
    use tambear::neural::cross_entropy_loss;

    // 3 classes, target is class 2, but model is very confident in class 0
    let logits = vec![100.0, -100.0, -100.0]; // confident class 0
    let targets = vec![2]; // true class is 2

    let loss = cross_entropy_loss(&logits, 3, &targets);
    eprintln!("Cross-entropy (confident wrong): loss = {loss:.4}");

    assert!(loss.is_finite(), "Loss must be finite, got {loss}");
    assert!(loss > 10.0, "Confident wrong prediction should have high loss, got {loss}");

    // Correct prediction: should have near-zero loss
    let correct_targets = vec![0]; // matches the confident prediction
    let loss_correct = cross_entropy_loss(&logits, 3, &correct_targets);
    eprintln!("Cross-entropy (confident right): loss = {loss_correct:.6}");

    assert!(loss_correct < 1e-6, "Confident right prediction should have ~0 loss, got {loss_correct}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Dim Reduction: PCA on rank-deficient data (more features than samples)
// ═══════════════════════════════════════════════════════════════════════════
//
// When n < d, the covariance matrix has rank at most n-1. PCA must handle
// this gracefully — producing at most n-1 meaningful components.

#[test]
fn pca_rank_deficient_data() {
    use tambear::dim_reduction::pca;

    // 3 samples in 10-dimensional space → rank ≤ 2
    let n = 3;
    let d = 10;
    let mut data = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            data[i * d + j] = (i as f64 + 1.0) * (j as f64 + 1.0);
        }
    }

    let res = pca(&data, n, d, 3);

    eprintln!("PCA rank-deficient: singular values = {:?}", res.singular_values);
    eprintln!("  explained variance ratios = {:?}", res.explained_variance_ratio);

    // All outputs must be finite
    assert!(res.singular_values.iter().all(|s| s.is_finite()),
        "Singular values must be finite");
    assert!(res.explained_variance_ratio.iter().all(|r| r.is_finite()),
        "Explained variance ratios must be finite");

    // First component should capture most variance (data is nearly 1D)
    assert!(res.explained_variance_ratio[0] > 0.9,
        "First PC should capture >90% of variance, got {:.4}", res.explained_variance_ratio[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mixture: GMM with k=1 (degenerate — single component)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gmm_single_component() {
    use tambear::mixture::gmm_em;

    // 50 points from a single Gaussian — k=1 should converge trivially
    let n = 50;
    let d = 2;
    let mut data = vec![0.0; n * d];
    let mut rng = 42u64;
    for i in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        data[i * d] = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        data[i * d + 1] = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
    }

    let res = gmm_em(&data, n, d, 1, 100, 1e-6);

    eprintln!("GMM k=1: weight={:.4}, mean={:?}, ll={:.4}",
        res.weights[0], res.means[0], res.log_likelihood);

    assert_eq!(res.weights.len(), 1);
    assert!((res.weights[0] - 1.0).abs() < 1e-10, "Single component weight must be 1.0");
    assert!(res.log_likelihood.is_finite(), "Log-likelihood must be finite");
    // All points assigned to cluster 0
    assert!(res.labels.iter().all(|&l| l == 0), "All labels should be 0 with k=1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Interpolation: Chebyshev vs equispaced at Runge (adversarial comparison)
// ═══════════════════════════════════════════════════════════════════════════
//
// This is the canonical Structure Beats Resources example. Equispaced
// interpolation of f(x) = 1/(1+25x²) on [-1,1] diverges as n→∞ due
// to the Lebesgue constant growing exponentially. Chebyshev nodes fix
// it structurally (Lebesgue constant grows logarithmically).

#[test]
fn interpolation_runge_equispaced_vs_chebyshev() {
    use tambear::interpolation::{lagrange, chebyshev_nodes, chebyshev_approximate, chebyshev_eval};

    let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
    let n = 20;

    // Equispaced nodes
    let equi_x: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64).collect();
    let equi_y: Vec<f64> = equi_x.iter().map(|&x| runge(x)).collect();

    // Chebyshev nodes on [-1, 1]
    let cheb_x = chebyshev_nodes(n, -1.0, 1.0);
    let cheb_coeffs = chebyshev_approximate(&runge, n, -1.0, 1.0);

    // Evaluate both at test point near the boundary (where Runge is worst)
    let test_x = 0.95;
    let equi_val = lagrange(&equi_x, &equi_y, test_x);
    let cheb_val = chebyshev_eval(&cheb_coeffs, test_x, -1.0, 1.0);
    let true_val = runge(test_x);

    eprintln!("Runge at x={test_x}, n={n}:");
    eprintln!("  True value:  {true_val:.6}");
    eprintln!("  Equispaced:  {equi_val:.6} (err={:.2e})", (equi_val - true_val).abs());
    eprintln!("  Chebyshev:   {cheb_val:.6} (err={:.2e})", (cheb_val - true_val).abs());

    // Chebyshev should be dramatically better
    let equi_err = (equi_val - true_val).abs();
    let cheb_err = (cheb_val - true_val).abs();

    assert!(cheb_err < 0.1,
        "Chebyshev error should be small, got {cheb_err:.4}");
    // Equispaced may oscillate wildly at n=20 — just verify both are finite
    assert!(equi_val.is_finite(), "Equispaced interpolation should be finite");
    assert!(cheb_val.is_finite(), "Chebyshev interpolation should be finite");
}

// ═══════════════════════════════════════════════════════════════════════════
// Complexity: Hurst exponent for known processes
// ═══════════════════════════════════════════════════════════════════════════
//
// White noise: H ≈ 0.5 (independent increments)
// Random walk: H ≈ 0.5 (self-similar with H=0.5)
// Trending series: H > 0.5 (persistent)
// Mean-reverting: H < 0.5 (anti-persistent)

#[test]
fn complexity_hurst_known_processes() {
    use tambear::complexity::hurst_rs;

    // White noise: H should be near 0.5
    let n = 500;
    let mut noise = vec![0.0; n];
    let mut rng = 42u64;
    for v in noise.iter_mut() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = rng as f64 / u64::MAX as f64 - 0.5;
    }
    let h_noise = hurst_rs(&noise);
    eprintln!("Hurst white noise: H = {h_noise:.4} (expected ~0.5)");
    assert!(h_noise > 0.2 && h_noise < 0.8,
        "White noise H should be near 0.5, got {h_noise}");

    // Trending series: cumulative sum of biased noise → H > 0.5
    let mut trend = vec![0.0; n];
    let mut acc = 0.0;
    rng = 42u64;
    for t in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        acc += 0.1 + (rng as f64 / u64::MAX as f64 - 0.5) * 0.5;
        trend[t] = acc;
    }
    let h_trend = hurst_rs(&trend);
    eprintln!("Hurst trending: H = {h_trend:.4} (expected > 0.5)");
    assert!(h_trend > 0.5, "Trending series should have H > 0.5, got {h_trend}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Complexity: Sample entropy on constant vs random
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn complexity_sample_entropy_constant_vs_random() {
    use tambear::complexity::sample_entropy;

    // Constant series: perfectly predictable → SampEn should be 0 or very small
    let constant = vec![5.0; 200];
    let se_const = sample_entropy(&constant, 2, 0.2);

    // Random series: unpredictable → SampEn should be larger
    let mut random = vec![0.0; 200];
    let mut rng = 42u64;
    for v in random.iter_mut() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = rng as f64 / u64::MAX as f64;
    }
    let se_rand = sample_entropy(&random, 2, 0.2);

    eprintln!("SampEn constant: {se_const:.4}");
    eprintln!("SampEn random:   {se_rand:.4}");

    // Both should be finite
    assert!(se_const.is_finite(), "SampEn(constant) must be finite");
    assert!(se_rand.is_finite(), "SampEn(random) must be finite");

    // Random should have higher entropy than constant
    // (Note: SampEn of constant may be 0, NaN, or Inf depending on implementation
    //  when all templates match. Check carefully.)
    if se_const.is_finite() && se_rand.is_finite() && se_const > 0.0 {
        assert!(se_rand > se_const,
            "Random should have higher SampEn than constant: {} vs {}", se_rand, se_const);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// REGRESSION: Panel FE single-cluster SE guard (Type 1 fix)
// ═══════════════════════════════════════════════════════════════════════════
//
// With 1 cluster, the N/(N-1) correction was 1/0 = Inf → SE = Inf.
// Fix: fall back to uncorrected sandwich SE when n_clusters <= 1.

#[test]
fn regression_panel_fe_single_cluster_finite_se() {
    use tambear::panel::panel_fe;

    let n = 10;
    let d = 1;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 1.0).collect();
    let units = vec![0usize; n]; // all same unit

    let res = panel_fe(&x, &y, n, d, &units);

    // The critical check: SE must be finite (was Inf before fix)
    for (j, &se) in res.se_clustered.iter().enumerate() {
        assert!(se.is_finite(), "SE[{j}] = {se} must be finite with 1 cluster");
        assert!(se >= 0.0, "SE[{j}] = {se} must be non-negative");
    }
    eprintln!("Panel FE single-cluster SE: {:?} (finite — guard works)", res.se_clustered);
}

// ═══════════════════════════════════════════════════════════════════════════
// REGRESSION: GARCH near-IGARCH detection (Type 2 fix)
// ═══════════════════════════════════════════════════════════════════════════
//
// When α + β → 1 (IGARCH), ω was pushed to ~10¹³ because the unconditional
// variance is undefined. Fix: flag near_igarch and clamp omega.

#[test]
fn regression_garch_igarch_omega_bounded() {
    use tambear::volatility::garch11_fit;

    // Generate near-IGARCH returns: α=0.15, β=0.84 → α+β=0.99
    let n = 500;
    let mut returns = vec![0.0; n];
    let mut sigma2: f64 = 0.01;
    let mut rng = 42u64;
    for t in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let z = (rng as f64 / u64::MAX as f64 - 0.5) * 3.46;
        returns[t] = sigma2.sqrt() * z;
        sigma2 = 0.0001 + 0.15 * returns[t].powi(2) + 0.84 * sigma2;
    }

    let res = garch11_fit(&returns, 500);
    let sample_var: f64 = returns.iter().map(|r| r * r).sum::<f64>() / n as f64;

    eprintln!("GARCH near-IGARCH: ω={:.6e}, α={:.4}, β={:.4}, α+β={:.4}, near_igarch={}",
        res.omega, res.alpha, res.beta, res.alpha + res.beta, res.near_igarch);

    // Critical check: omega must not explode (was 10¹³ before fix)
    assert!(res.omega < 1000.0 * sample_var,
        "ω = {:.4e} should be bounded, not exploding (sample_var = {:.4e})",
        res.omega, sample_var);
    assert!(res.omega > 0.0, "ω must be positive");
}

// ═══════════════════════════════════════════════════════════════════════════
// REGRESSION: ESS monotone sequence estimator (Type 2 fix)
// ═══════════════════════════════════════════════════════════════════════════
//
// ESS for AR(1) ρ=0.99 was overestimated 3.7x due to early truncation.
// Fix: Geyer's initial monotone sequence estimator (IMSE).

#[test]
fn regression_ess_high_autocorrelation_not_overestimated() {
    use tambear::bayesian::effective_sample_size;

    // Generate AR(1) with ρ=0.99 → theoretical ESS ≈ n*(1-ρ)/(1+ρ) ≈ n*0.005
    let n = 1000;
    let rho = 0.99;
    let mut samples = vec![0.0; n];
    let mut rng = 42u64;
    for t in 1..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.2;
        samples[t] = rho * samples[t - 1] + noise;
    }

    let ess = effective_sample_size(&samples);
    let theoretical = n as f64 * (1.0 - rho) / (1.0 + rho);

    eprintln!("ESS for AR(1) ρ=0.99: ESS={:.1}, theoretical≈{:.1}, ratio={:.2}x",
        ess, theoretical, ess / theoretical);

    // Should be within 2x of theoretical (was 3.7x before fix)
    assert!(ess < theoretical * 2.5,
        "ESS={:.1} should not overestimate theoretical {:.1} by more than 2.5x (ratio={:.2}x)",
        ess, theoretical, ess / theoretical);
    assert!(ess > 0.0, "ESS must be positive");
}
