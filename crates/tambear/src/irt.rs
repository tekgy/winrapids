//! # Family 15 — Item Response Theory & Psychometrics
//!
//! Rasch, 2PL, 3PL models, person ability estimation, test information.
//!
//! ## Architecture
//!
//! IRT models = iterative MLE/EM (Kingdom C).
//! Person estimation = Newton-Raphson or EAP (Kingdom C).
//! Test information = closed-form from item parameters (Kingdom A).

// ═══════════════════════════════════════════════════════════════════════════
// IRT probability models
// ═══════════════════════════════════════════════════════════════════════════

/// 1PL (Rasch) probability: P(correct | θ, b) = 1 / (1 + exp(-(θ-b)))
pub fn rasch_prob(theta: f64, difficulty: f64) -> f64 {
    logistic(theta - difficulty)
}

/// 2PL probability: P(correct | θ, a, b) = 1 / (1 + exp(-a(θ-b)))
pub fn prob_2pl(theta: f64, discrimination: f64, difficulty: f64) -> f64 {
    logistic(discrimination * (theta - difficulty))
}

/// 3PL probability with guessing: P = c + (1-c) / (1 + exp(-a(θ-b)))
pub fn prob_3pl(theta: f64, discrimination: f64, difficulty: f64, guessing: f64) -> f64 {
    let guessing = guessing.clamp(0.0, 1.0);
    guessing + (1.0 - guessing) * logistic(discrimination * (theta - difficulty))
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════
// Item parameters
// ═══════════════════════════════════════════════════════════════════════════

/// Item parameters for 2PL model.
#[derive(Debug, Clone)]
pub struct ItemParams {
    /// Discrimination (a).
    pub discrimination: f64,
    /// Difficulty (b).
    pub difficulty: f64,
}

/// Fit 2PL model via joint MLE (alternating Newton steps).
/// `responses`: n_persons × n_items binary matrix (1=correct, 0=incorrect), row-major.
/// Returns item parameters.
pub fn fit_2pl(responses: &[u8], n_persons: usize, n_items: usize, max_iter: usize) -> Vec<ItemParams> {
    assert_eq!(responses.len(), n_persons * n_items);

    let mut abilities: Vec<f64> = vec![0.0; n_persons];
    let mut items: Vec<ItemParams> = (0..n_items).map(|j| {
        let p_correct: f64 = (0..n_persons).map(|i| responses[i * n_items + j] as f64).sum::<f64>()
            / n_persons as f64;
        ItemParams {
            discrimination: 1.0,
            difficulty: -logit(p_correct.clamp(0.01, 0.99)),
        }
    }).collect();

    for _ in 0..max_iter {
        // E-step: estimate abilities given item parameters
        for i in 0..n_persons {
            // Newton-Raphson for θ_i
            let mut theta = abilities[i];
            for _ in 0..10 {
                let mut grad = 0.0;
                let mut hess = 0.0;
                for j in 0..n_items {
                    let a = items[j].discrimination;
                    let p = prob_2pl(theta, a, items[j].difficulty);
                    let r = responses[i * n_items + j] as f64;
                    grad += a * (r - p);
                    hess -= a * a * p * (1.0 - p);
                }
                // Prior: θ ~ N(0,1)
                grad -= theta;
                hess -= 1.0;
                if hess.abs() < 1e-10 { break; }
                let step = grad / hess;
                theta -= step;
                theta = theta.clamp(-6.0, 6.0);
                if step.abs() < 1e-6 { break; }
            }
            abilities[i] = theta;
        }

        // M-step: update item parameters given abilities.
        // Uses the diagonal Hessian approximation (Birnbaum 1968, Lord & Novick Ch.17):
        // ignores the off-diagonal H_ab = Σ_i (θ_i - b)·a·p·(1-p) which couples
        // discrimination and difficulty. This is the standard JMLE approximation —
        // a and b are updated independently via separate Newton steps.
        // The full 2×2 Newton step converges faster but adds complexity and
        // instability near flat likelihoods.
        for j in 0..n_items {
            let mut a = items[j].discrimination;
            let mut b = items[j].difficulty;
            for _ in 0..10 {
                let mut grad_a = 0.0;
                let mut grad_b = 0.0;
                let mut hess_aa = 0.0;
                let mut hess_bb = 0.0;
                for i in 0..n_persons {
                    let p = prob_2pl(abilities[i], a, b);
                    let r = responses[i * n_items + j] as f64;
                    let diff = abilities[i] - b;
                    grad_a += diff * (r - p);
                    grad_b += -a * (r - p);
                    hess_aa -= diff * diff * p * (1.0 - p);
                    hess_bb -= a * a * p * (1.0 - p);
                    // H_ab = Σ diff * a * p*(1-p) omitted (diagonal approx.)
                }
                if hess_aa.abs() > 1e-10 { a -= grad_a / hess_aa; }
                if hess_bb.abs() > 1e-10 { b -= grad_b / hess_bb; }
                a = a.clamp(0.1, 5.0);
                b = b.clamp(-5.0, 5.0);
            }
            items[j] = ItemParams { discrimination: a, difficulty: b };
        }
    }

    items
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

// ═══════════════════════════════════════════════════════════════════════════
// Person ability estimation
// ═══════════════════════════════════════════════════════════════════════════

/// MLE estimation of person ability given item parameters and responses.
/// MLE ability estimate via Newton-Raphson on the item response log-likelihood.
///
/// Iterates up to 50 Newton steps starting from θ=0.0; clamps θ to [-6, 6].
///
/// **Perfect scores (all 0 or all 1)**: The log-likelihood is monotone with no
/// interior maximum, so MLE does not exist. This function returns the boundary
/// value (near -6.0 for all-zero, near +6.0 for all-one). Use `ability_eap`
/// instead when perfect scores are possible — EAP with a N(0,1) prior always
/// has a finite posterior mean.
pub fn ability_mle(items: &[ItemParams], responses: &[u8]) -> f64 {
    let n = items.len();
    assert_eq!(responses.len(), n);

    let mut theta = 0.0;
    for _ in 0..50 {
        let mut grad = 0.0;
        let mut hess = 0.0;
        for j in 0..n {
            let a = items[j].discrimination;
            let p = prob_2pl(theta, a, items[j].difficulty);
            let r = responses[j] as f64;
            grad += a * (r - p);
            hess -= a * a * p * (1.0 - p);
        }
        if hess.abs() < 1e-10 { break; }
        let step = grad / hess;
        theta -= step;
        theta = theta.clamp(-6.0, 6.0);
        if step.abs() < 1e-6 { break; }
    }
    theta
}

/// EAP (Expected A Posteriori) ability estimate with N(0,1) prior.
///
/// Uses uniform quadrature over [-4, 4] (n_quad equally-spaced nodes), **not**
/// Gauss-Hermite quadrature. Uniform quadrature is adequate for smooth
/// posteriors within the typical ability range; Gauss-Hermite would be more
/// efficient but requires fixed node tables.
///
/// Computes in log-likelihood space using the log-sum-exp trick to avoid
/// underflow when the number of items is large (>50).
pub fn ability_eap(items: &[ItemParams], responses: &[u8], n_quad: usize) -> f64 {
    let n = items.len();
    assert_eq!(responses.len(), n);

    // Guard: need at least 2 quadrature points to span the interval.
    if n_quad < 2 {
        return 0.0;
    }

    // Simple quadrature over [-4, 4], computed in log space.
    let mut log_weights: Vec<f64> = Vec::with_capacity(n_quad);
    let mut thetas: Vec<f64> = Vec::with_capacity(n_quad);

    for q in 0..n_quad {
        let theta = -4.0 + 8.0 * q as f64 / (n_quad - 1) as f64;
        let log_prior = -0.5 * theta * theta;
        let mut log_lik = 0.0;
        for j in 0..n {
            let p = prob_2pl(theta, items[j].discrimination, items[j].difficulty);
            log_lik += if responses[j] == 1 { p.ln() } else { (1.0 - p).ln() };
        }
        thetas.push(theta);
        log_weights.push(log_lik + log_prior);
    }

    // Log-sum-exp trick: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
    let max_lw = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_lw == f64::NEG_INFINITY {
        return 0.0;
    }

    let mut numer = 0.0;
    let mut denom = 0.0;
    for q in 0..n_quad {
        let w = (log_weights[q] - max_lw).exp();
        numer += thetas[q] * w;
        denom += w;
    }
    if denom < 1e-300 { return 0.0; }
    numer / denom
}

// ═══════════════════════════════════════════════════════════════════════════
// Information functions
// ═══════════════════════════════════════════════════════════════════════════

/// Item information for 2PL: I(θ) = a² · P(θ) · (1-P(θ)).
pub fn item_information(theta: f64, item: &ItemParams) -> f64 {
    let p = prob_2pl(theta, item.discrimination, item.difficulty);
    item.discrimination * item.discrimination * p * (1.0 - p)
}

/// Test information: sum of item information functions.
pub fn test_information(theta: f64, items: &[ItemParams]) -> f64 {
    items.iter().map(|item| item_information(theta, item)).sum()
}

/// Standard error of measurement = 1/√I(θ).
/// Returns f64::INFINITY if items is empty (zero information).
pub fn sem(theta: f64, items: &[ItemParams]) -> f64 {
    let info = test_information(theta, items);
    if info <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / info.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// DIF (Differential Item Functioning)
// ═══════════════════════════════════════════════════════════════════════════

/// Mantel-Haenszel DIF statistic for one item.
/// `responses`: binary responses for this item across all persons.
/// `group`: 0=reference, 1=focal for each person.
/// `total_scores`: matching variable (total test scores).
/// Returns log odds ratio (positive = easier for reference group).
pub fn mantel_haenszel_dif(
    responses: &[u8], group: &[usize], total_scores: &[usize]
) -> f64 {
    let n = responses.len();
    assert_eq!(group.len(), n);
    assert_eq!(total_scores.len(), n);

    let max_score = *total_scores.iter().max().unwrap_or(&1);

    let mut a_sum: f64 = 0.0; // Σ A_k · D_k / N_k
    let mut b_sum: f64 = 0.0; // Σ B_k · C_k / N_k

    for score in 0..=max_score {
        // Count within this score stratum
        let mut a = 0.0; // ref correct
        let mut b = 0.0; // ref incorrect
        let mut c = 0.0; // focal correct
        let mut d = 0.0; // focal incorrect

        for i in 0..n {
            if total_scores[i] != score { continue; }
            if group[i] == 0 {
                if responses[i] == 1 { a += 1.0; } else { b += 1.0; }
            } else {
                if responses[i] == 1 { c += 1.0; } else { d += 1.0; }
            }
        }

        let nk = a + b + c + d;
        if nk < 1.0 { continue; }
        a_sum += a * d / nk;
        b_sum += b * c / nk;
    }

    if b_sum < 1e-10 { return f64::INFINITY; }
    (a_sum / b_sum).ln() // log MH odds ratio
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: {a} vs {b} (diff={})", (a - b).abs());
    }

    #[test]
    fn rasch_symmetry() {
        close(rasch_prob(0.0, 0.0), 0.5, 1e-10, "P(θ=b)=0.5");
        let p1 = rasch_prob(1.0, 0.0);
        let p2 = rasch_prob(-1.0, 0.0);
        close(p1 + p2, 1.0, 1e-10, "Symmetry");
    }

    #[test]
    fn prob_2pl_discrimination() {
        // Higher discrimination → steeper curve
        let p_high = prob_2pl(0.5, 2.0, 0.0);
        let p_low = prob_2pl(0.5, 0.5, 0.0);
        assert!(p_high > p_low, "Higher a → more separation");
    }

    #[test]
    fn prob_3pl_guessing_floor() {
        // 3PL with guessing: P should be ≥ c
        let c = 0.25;
        let p = prob_3pl(-10.0, 1.0, 0.0, c);
        assert!(p >= c - 1e-10, "P={p} should be ≥ c={c}");
    }

    #[test]
    fn fit_2pl_recovers_difficulty_order() {
        // Easy item (high proportion correct) should have lower difficulty
        let n_persons = 100;
        let n_items = 3;
        let mut responses = vec![0u8; n_persons * n_items];
        let mut rng = 42u64;

        // Item 0: easy (80% correct), Item 1: medium (50%), Item 2: hard (20%)
        for i in 0..n_persons {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            responses[i * n_items + 0] = if (rng as f64 / u64::MAX as f64) < 0.8 { 1 } else { 0 };
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            responses[i * n_items + 1] = if (rng as f64 / u64::MAX as f64) < 0.5 { 1 } else { 0 };
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            responses[i * n_items + 2] = if (rng as f64 / u64::MAX as f64) < 0.2 { 1 } else { 0 };
        }

        let items = fit_2pl(&responses, n_persons, n_items, 20);
        assert!(items[0].difficulty < items[2].difficulty,
            "Easy item b={} should be < hard item b={}", items[0].difficulty, items[2].difficulty);
    }

    #[test]
    fn ability_mle_high_scorer() {
        let items = vec![
            ItemParams { discrimination: 1.0, difficulty: -1.0 },
            ItemParams { discrimination: 1.0, difficulty: 0.0 },
            ItemParams { discrimination: 1.0, difficulty: 1.0 },
            ItemParams { discrimination: 1.0, difficulty: 2.0 },
        ];
        // All correct → high ability
        let theta_all = ability_mle(&items, &[1, 1, 1, 1]);
        // All wrong → low ability
        let theta_none = ability_mle(&items, &[0, 0, 0, 0]);
        assert!(theta_all > theta_none,
            "All correct θ={} should be > all wrong θ={}", theta_all, theta_none);
    }

    #[test]
    fn test_info_peak_at_difficulty() {
        let item = ItemParams { discrimination: 1.5, difficulty: 1.0 };
        // Information peaks at θ = b
        let info_at_b = item_information(1.0, &item);
        let info_away = item_information(3.0, &item);
        assert!(info_at_b > info_away,
            "Info at b={} should be > info away={}", info_at_b, info_away);
    }

    #[test]
    fn test_info_additive() {
        let items = vec![
            ItemParams { discrimination: 1.0, difficulty: 0.0 },
            ItemParams { discrimination: 1.5, difficulty: 0.5 },
        ];
        let total = test_information(0.0, &items);
        let sum = item_information(0.0, &items[0]) + item_information(0.0, &items[1]);
        close(total, sum, 1e-10, "Test info = sum of item info");
    }

    #[test]
    fn sem_inversely_related_to_info() {
        let items = vec![
            ItemParams { discrimination: 2.0, difficulty: 0.0 },
            ItemParams { discrimination: 2.0, difficulty: 0.5 },
            ItemParams { discrimination: 2.0, difficulty: -0.5 },
        ];
        let se = sem(0.0, &items);
        let info = test_information(0.0, &items);
        close(se, 1.0 / info.sqrt(), 1e-10, "SEM = 1/√I");
    }

    // ── Bug-fix regression tests ──────────────────────────────────────────

    #[test]
    fn ability_eap_many_items_no_underflow() {
        // 100 items: old code underflowed to 0.0 and returned the prior mean.
        // With log-space computation, it should produce a meaningful estimate.
        let items: Vec<ItemParams> = (0..100).map(|i| ItemParams {
            discrimination: 1.0,
            difficulty: -2.0 + 4.0 * i as f64 / 99.0,
        }).collect();
        // All correct → ability should be well above 0
        let responses_all: Vec<u8> = vec![1; 100];
        let theta = ability_eap(&items, &responses_all, 41);
        assert!(theta > 1.0, "100 all-correct items: θ={theta} should be > 1.0");
        assert!(theta.is_finite(), "θ should be finite, got {theta}");

        // All wrong → ability should be well below 0
        let responses_none: Vec<u8> = vec![0; 100];
        let theta_low = ability_eap(&items, &responses_none, 41);
        assert!(theta_low < -1.0, "100 all-wrong items: θ={theta_low} should be < -1.0");
        assert!(theta_low.is_finite(), "θ should be finite, got {theta_low}");
    }

    #[test]
    fn ability_eap_n_quad_zero_no_panic() {
        let items = vec![ItemParams { discrimination: 1.0, difficulty: 0.0 }];
        // n_quad=0 should not panic (usize underflow) — returns default 0.0
        let theta = ability_eap(&items, &[1], 0);
        close(theta, 0.0, 1e-10, "n_quad=0 returns 0.0");
    }

    #[test]
    fn ability_eap_n_quad_one_no_panic() {
        let items = vec![ItemParams { discrimination: 1.0, difficulty: 0.0 }];
        // n_quad=1 should not panic — returns default 0.0
        let theta = ability_eap(&items, &[1], 1);
        close(theta, 0.0, 1e-10, "n_quad=1 returns 0.0");
    }

    #[test]
    fn sem_empty_items_returns_infinity() {
        let se = sem(0.0, &[]);
        assert!(se.is_infinite() && se > 0.0,
            "SEM with no items should be +infinity, got {se}");
    }
}
