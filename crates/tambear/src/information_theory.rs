//! Information theory — entropy, divergence, mutual information.
//!
//! ## Architecture
//!
//! Every information-theoretic measure decomposes to:
//! 1. **Histogram** via `scatter(ByKey, "1.0", Add)` — O(n) GPU, one kernel
//! 2. **Derive** from counts — O(k) CPU where k = number of bins
//!
//! For joint measures (mutual information), a composite key `x * ny + y`
//! produces an (nx × ny) contingency table in a single scatter pass.
//!
//! ## Measures
//!
//! **Entropy**: Shannon, Rényi (all orders), Tsallis
//! **Divergence**: KL, JS, cross-entropy
//! **Mutual information**: MI, NMI (several normalizations), AMI, VI
//! **Conditional**: H(Y|X), H(X|Y)
//!
//! ## Clustering evaluation
//!
//! `mutual_info_score`, `normalized_mutual_info_score`, `adjusted_mutual_info_score`
//! — drop-in replacements for sklearn.metrics with identical semantics.
//!
//! ## Convention
//!
//! All logarithms are natural (base e) unless otherwise noted.
//! `0 * log(0)` is defined as 0 (the limit as p→0⁺).

use crate::compute_engine::ComputeEngine;
use crate::PHI_COUNT;

// ═══════════════════════════════════════════════════════════════════════════
// Core: probability distributions from counts
// ═══════════════════════════════════════════════════════════════════════════

/// Normalize counts to probabilities. Returns empty vec if total is 0.
pub fn probabilities(counts: &[f64]) -> Vec<f64> {
    let total: f64 = counts.iter().sum();
    if total == 0.0 { return vec![0.0; counts.len()]; }
    counts.iter().map(|&c| c / total).collect()
}

/// Safely compute p * ln(p), returning 0.0 when p ≤ 0.
#[inline]
fn p_log_p(p: f64) -> f64 {
    if p <= 0.0 { 0.0 } else { p * p.ln() }
}

/// Safely compute p * ln(p/q), returning 0.0 when p ≤ 0.
#[inline]
fn p_log_p_over_q(p: f64, q: f64) -> f64 {
    if p <= 0.0 { return 0.0; }
    if q <= 0.0 { return f64::INFINITY; }
    p * (p / q).ln()
}

// ═══════════════════════════════════════════════════════════════════════════
// Entropy variants
// ═══════════════════════════════════════════════════════════════════════════

/// Shannon entropy: H(X) = -Σ pᵢ log(pᵢ).
///
/// Input: probability distribution (non-negative, sums to 1).
/// Returns 0 for a degenerate distribution (one event with probability 1).
/// Maximum is log(k) for k equally likely events.
pub fn shannon_entropy(probs: &[f64]) -> f64 {
    -probs.iter().map(|&p| p_log_p(p)).sum::<f64>()
}

/// Shannon entropy directly from counts (convenience).
pub fn shannon_entropy_from_counts(counts: &[f64]) -> f64 {
    shannon_entropy(&probabilities(counts))
}

/// Rényi entropy of order α: H_α(X) = (1/(1-α)) × log(Σ pᵢ^α).
///
/// Special cases:
/// - α = 0 → log(|support|) (Hartley entropy)
/// - α → 1 → Shannon entropy (limit)
/// - α = 2 → -log(Σ pᵢ²) (collision entropy)
/// - α → ∞ → -log(max pᵢ) (min-entropy)
///
/// Panics if α < 0.
pub fn renyi_entropy(probs: &[f64], alpha: f64) -> f64 {
    assert!(alpha >= 0.0, "Rényi order α must be ≥ 0, got {}", alpha);

    if (alpha - 1.0).abs() < 1e-12 {
        // Limit α → 1: Shannon entropy
        return shannon_entropy(probs);
    }

    if alpha == 0.0 {
        // H_0 = log(|support|)
        let support = probs.iter().filter(|&&p| p > 0.0).count();
        return (support as f64).ln();
    }

    if alpha == f64::INFINITY {
        // Min-entropy: -log(max p)
        let max_p = probs.iter().cloned().fold(0.0f64, f64::max);
        return -max_p.ln();
    }

    let sum_p_alpha: f64 = probs.iter().map(|&p| {
        if p <= 0.0 { 0.0 } else { p.powf(alpha) }
    }).sum();

    (1.0 / (1.0 - alpha)) * sum_p_alpha.ln()
}

/// Tsallis entropy of order q: S_q(X) = (1/(q-1)) × (1 - Σ pᵢ^q).
///
/// Non-additive generalization of Shannon entropy.
/// Limit q → 1: Shannon entropy.
pub fn tsallis_entropy(probs: &[f64], q: f64) -> f64 {
    if (q - 1.0).abs() < 1e-12 {
        return shannon_entropy(probs);
    }

    let sum_p_q: f64 = probs.iter().map(|&p| {
        if p <= 0.0 { 0.0 } else { p.powf(q) }
    }).sum();

    (1.0 / (q - 1.0)) * (1.0 - sum_p_q)
}

// ═══════════════════════════════════════════════════════════════════════════
// Divergences
// ═══════════════════════════════════════════════════════════════════════════

/// KL divergence: D_KL(P || Q) = Σ pᵢ log(pᵢ/qᵢ).
///
/// NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P).
/// Returns +∞ if any pᵢ > 0 where qᵢ = 0 (Q doesn't cover P's support).
/// `p` and `q` must have the same length.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "KL divergence: p and q must have same length");
    p.iter().zip(q).map(|(&pi, &qi)| p_log_p_over_q(pi, qi)).sum()
}

/// Jensen-Shannon divergence: D_JS(P, Q) = 0.5 × D_KL(P||M) + 0.5 × D_KL(Q||M)
/// where M = (P + Q) / 2.
///
/// Symmetric, bounded: [0, ln(2)].
/// Square root of JS divergence is a true metric.
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "JS divergence: p and q must have same length");
    let m: Vec<f64> = p.iter().zip(q).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// Cross-entropy: H(P, Q) = -Σ pᵢ log(qᵢ).
///
/// Measures the expected message length when using Q to encode P.
/// H(P, Q) = H(P) + D_KL(P || Q).
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "cross-entropy: p and q must have same length");
    -p.iter().zip(q).map(|(&pi, &qi)| {
        if pi <= 0.0 { 0.0 }
        else if qi <= 0.0 { f64::INFINITY }
        else { pi * qi.ln() }
    }).sum::<f64>()
}

// ═══════════════════════════════════════════════════════════════════════════
// Mutual information
// ═══════════════════════════════════════════════════════════════════════════

/// Mutual information from a contingency table.
///
/// I(X; Y) = Σᵢⱼ p(i,j) × log(p(i,j) / (p(i) × p(j)))
///
/// `contingency`: row-major nx × ny table of counts.
/// Returns MI in nats (base e).
pub fn mutual_information(contingency: &[f64], nx: usize, ny: usize) -> f64 {
    assert_eq!(contingency.len(), nx * ny, "contingency must be nx × ny");
    let total: f64 = contingency.iter().sum();
    if total == 0.0 { return 0.0; }

    // Marginals
    let mut row_sums = vec![0.0f64; nx];
    let mut col_sums = vec![0.0f64; ny];
    for i in 0..nx {
        for j in 0..ny {
            let c = contingency[i * ny + j];
            row_sums[i] += c;
            col_sums[j] += c;
        }
    }

    let mut mi = 0.0f64;
    for i in 0..nx {
        for j in 0..ny {
            let nij = contingency[i * ny + j];
            if nij <= 0.0 { continue; }
            let pij = nij / total;
            let pi = row_sums[i] / total;
            let pj = col_sums[j] / total;
            mi += pij * (pij / (pi * pj)).ln();
        }
    }
    mi.max(0.0) // MI is non-negative; clamp numerical noise
}

/// Normalized mutual information.
///
/// `method`:
/// - `"arithmetic"` → 2 × I(X;Y) / (H(X) + H(Y))  (default, sklearn)
/// - `"geometric"` → I(X;Y) / √(H(X) × H(Y))
/// - `"min"` → I(X;Y) / min(H(X), H(Y))
/// - `"max"` → I(X;Y) / max(H(X), H(Y))
///
/// Range [0, 1]. 1 = perfect correlation.
pub fn normalized_mutual_information(
    contingency: &[f64], nx: usize, ny: usize, method: &str,
) -> f64 {
    let total: f64 = contingency.iter().sum();
    if total == 0.0 { return 0.0; }

    let mut row_sums = vec![0.0f64; nx];
    let mut col_sums = vec![0.0f64; ny];
    for i in 0..nx {
        for j in 0..ny {
            let c = contingency[i * ny + j];
            row_sums[i] += c;
            col_sums[j] += c;
        }
    }

    let hx = shannon_entropy(&probabilities(&row_sums));
    let hy = shannon_entropy(&probabilities(&col_sums));
    let mi = mutual_information(contingency, nx, ny);

    let denom = match method {
        "arithmetic" => (hx + hy) / 2.0,
        "geometric" => (hx * hy).sqrt(),
        "min" => hx.min(hy),
        "max" => hx.max(hy),
        _ => (hx + hy) / 2.0,
    };

    if denom == 0.0 { 0.0 } else { (mi / denom).min(1.0) }
}

/// Variation of information: VI(X, Y) = H(X|Y) + H(Y|X).
///
/// A true metric on clusterings. VI = 0 iff X = Y.
pub fn variation_of_information(contingency: &[f64], nx: usize, ny: usize) -> f64 {
    let total: f64 = contingency.iter().sum();
    if total == 0.0 { return 0.0; }

    let mut row_sums = vec![0.0f64; nx];
    let mut col_sums = vec![0.0f64; ny];
    for i in 0..nx {
        for j in 0..ny {
            let c = contingency[i * ny + j];
            row_sums[i] += c;
            col_sums[j] += c;
        }
    }

    let hx = shannon_entropy(&probabilities(&row_sums));
    let hy = shannon_entropy(&probabilities(&col_sums));
    let mi = mutual_information(contingency, nx, ny);

    // VI = H(X|Y) + H(Y|X) = H(X) - I(X;Y) + H(Y) - I(X;Y) = H(X) + H(Y) - 2*I
    (hx + hy - 2.0 * mi).max(0.0)
}

/// Conditional entropy H(Y|X) = H(X,Y) - H(X).
///
/// How much uncertainty about Y remains after observing X.
pub fn conditional_entropy(contingency: &[f64], nx: usize, ny: usize) -> f64 {
    let total: f64 = contingency.iter().sum();
    if total == 0.0 { return 0.0; }

    let mut row_sums = vec![0.0f64; nx];
    for i in 0..nx {
        for j in 0..ny {
            row_sums[i] += contingency[i * ny + j];
        }
    }

    let hx = shannon_entropy(&probabilities(&row_sums));
    let hxy = shannon_entropy(&probabilities(contingency));
    (hxy - hx).max(0.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Histogram computation (scatter-based)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute histogram (count per bin) via scatter.
///
/// `keys[i]` must be in [0, n_bins). Returns count per bin.
/// Uses ComputeEngine for multi-backend support.
pub fn histogram(
    compute: &mut ComputeEngine,
    keys: &[i32],
    n_bins: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let values = vec![0.0f64; keys.len()]; // dummy values; PHI_COUNT ignores them
    compute.scatter_phi(PHI_COUNT, keys, &values, None, n_bins)
}

/// Compute joint histogram (2D contingency table) via scatter.
///
/// Flattens (x, y) pairs into composite keys: `key = x * ny + y`.
/// Returns flat nx × ny array (row-major).
pub fn joint_histogram(
    compute: &mut ComputeEngine,
    keys_x: &[i32],
    keys_y: &[i32],
    nx: usize,
    ny: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    assert_eq!(keys_x.len(), keys_y.len(), "keys_x and keys_y must have same length");
    let n_bins = nx.checked_mul(ny).expect("nx * ny overflow");

    if n_bins <= i32::MAX as usize {
        // Composite key fits in i32 — use GPU scatter
        let composite: Vec<i32> = keys_x.iter().zip(keys_y)
            .map(|(&x, &y)| x as i32 * ny as i32 + y)
            .collect();
        let values = vec![0.0f64; composite.len()];
        compute.scatter_phi(PHI_COUNT, &composite, &values, None, n_bins)
    } else {
        // Composite key exceeds i32 range — CPU fallback (avoids overflow)
        let mut hist = vec![0.0f64; n_bins];
        for (&x, &y) in keys_x.iter().zip(keys_y) {
            let idx = x as usize * ny + y as usize;
            if idx < n_bins { hist[idx] += 1.0; }
        }
        Ok(hist)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Clustering evaluation metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Build a contingency table from two label vectors.
///
/// `labels_a[i]` must be in [0, na). `labels_b[i]` must be in [0, nb).
/// Returns (contingency, na, nb).
fn contingency_from_labels(labels_a: &[i32], labels_b: &[i32]) -> (Vec<f64>, usize, usize) {
    assert_eq!(labels_a.len(), labels_b.len(), "label vectors must have same length");
    let na = labels_a.iter().copied().max().unwrap_or(-1) as usize + 1;
    let nb = labels_b.iter().copied().max().unwrap_or(-1) as usize + 1;
    let mut table = vec![0.0f64; na * nb];
    for (&a, &b) in labels_a.iter().zip(labels_b) {
        table[a as usize * nb + b as usize] += 1.0;
    }
    (table, na, nb)
}

/// Mutual information score between two clusterings.
///
/// `labels_true` and `labels_pred` are integer label vectors (0-indexed).
/// Returns MI in nats. Matches `sklearn.metrics.mutual_info_score`.
pub fn mutual_info_score(labels_true: &[i32], labels_pred: &[i32]) -> f64 {
    let (table, na, nb) = contingency_from_labels(labels_true, labels_pred);
    mutual_information(&table, na, nb)
}

/// Normalized mutual information score.
///
/// `average_method`: "arithmetic" (default), "geometric", "min", "max".
/// Matches `sklearn.metrics.normalized_mutual_info_score`.
pub fn normalized_mutual_info_score(
    labels_true: &[i32], labels_pred: &[i32], average_method: &str,
) -> f64 {
    let (table, na, nb) = contingency_from_labels(labels_true, labels_pred);
    normalized_mutual_information(&table, na, nb, average_method)
}

/// Adjusted mutual information score (corrected for chance).
///
/// AMI = (MI - E[MI]) / (max(H_a, H_b) - E[MI])
///
/// Returns 0 for random labeling, 1 for perfect agreement.
/// Matches `sklearn.metrics.adjusted_mutual_info_score`.
pub fn adjusted_mutual_info_score(labels_true: &[i32], labels_pred: &[i32]) -> f64 {
    let (table, na, nb) = contingency_from_labels(labels_true, labels_pred);
    let n = labels_true.len() as f64;
    if n == 0.0 { return 0.0; }

    let mi = mutual_information(&table, na, nb);

    // Marginals
    let mut a_counts = vec![0.0f64; na];
    let mut b_counts = vec![0.0f64; nb];
    for i in 0..na {
        for j in 0..nb {
            let c = table[i * nb + j];
            a_counts[i] += c;
            b_counts[j] += c;
        }
    }

    let ha = shannon_entropy(&probabilities(&a_counts));
    let hb = shannon_entropy(&probabilities(&b_counts));

    // Expected MI under the hypergeometric model (approximation for large n)
    let emi = expected_mutual_info(&a_counts, &b_counts, n);

    let max_h = ha.max(hb);
    let denom = max_h - emi;
    if denom.abs() < 1e-15 { 0.0 } else { ((mi - emi) / denom).min(1.0) }
}

/// Expected mutual information under the hypergeometric model.
///
/// E[MI] = Σᵢⱼ Σ_{nij} p(nij) × nij/n × log(n × nij / (ai × bj))
///
/// Uses the exact formula with log-factorials for numerical stability.
fn expected_mutual_info(a: &[f64], b: &[f64], n: f64) -> f64 {
    let na = a.len();
    let nb = b.len();
    let n_int = n as usize;
    let log_n = n.ln();

    // Precompute log-factorials
    let max_val = n_int + 1;
    let mut log_fact = vec![0.0f64; max_val + 1];
    for i in 2..=max_val {
        log_fact[i] = log_fact[i - 1] + (i as f64).ln();
    }

    let mut emi = 0.0f64;
    for i in 0..na {
        let ai = a[i] as usize;
        if ai == 0 { continue; }
        for j in 0..nb {
            let bj = b[j] as usize;
            if bj == 0 { continue; }

            let nij_min = if ai + bj > n_int { ai + bj - n_int } else { 0 };
            let nij_max = ai.min(bj);

            for nij in nij_min..=nij_max {
                if nij == 0 { continue; }
                // Hypergeometric probability:
                // P(nij) = C(ai, nij) × C(n-ai, bj-nij) / C(n, bj)
                let log_p = log_fact[ai] - log_fact[nij] - log_fact[ai - nij]
                    + log_fact[n_int - ai] - log_fact[bj - nij] - log_fact[n_int - ai - bj + nij]
                    - log_fact[n_int] + log_fact[bj] + log_fact[n_int - bj];

                let term = (nij as f64) / n * ((n * nij as f64) / (ai as f64 * bj as f64)).ln();
                emi += log_p.exp() * term;
            }
        }
    }
    emi
}

// ═══════════════════════════════════════════════════════════════════════════
// Continuous entropy estimation
// ═══════════════════════════════════════════════════════════════════════════

/// Estimate Shannon entropy of continuous data via equal-width histogram.
///
/// Discretizes values into `n_bins` equal-width bins, then computes
/// differential entropy corrected for bin width:
///   H_continuous ≈ H_discrete + log(bin_width)
///
/// NaN values are excluded.
pub fn entropy_histogram(values: &[f64], n_bins: usize) -> f64 {
    assert!(n_bins > 0, "n_bins must be > 0");
    let clean: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if clean.is_empty() { return f64::NAN; }

    let min = clean.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = clean.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if min == max { return 0.0; } // zero entropy for constant data

    let bin_width = (max - min) / n_bins as f64;
    let mut counts = vec![0.0f64; n_bins];
    for &v in &clean {
        let bin = ((v - min) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1); // clamp max value to last bin
        counts[bin] += 1.0;
    }

    let probs = probabilities(&counts);
    shannon_entropy(&probs) + bin_width.ln()
}

// ═══════════════════════════════════════════════════════════════════════════
// InformationEngine — scatter-accelerated computation
// ═══════════════════════════════════════════════════════════════════════════

/// Information theory engine with GPU-accelerated histogram computation.
///
/// For small data or pre-computed counts, use the free functions directly.
/// The engine adds scatter-based histogram computation for large data.
pub struct InformationEngine {
    compute: ComputeEngine,
}

impl InformationEngine {
    pub fn new() -> Self {
        Self { compute: ComputeEngine::new(tam_gpu::detect()) }
    }

    /// Compute Shannon entropy from label keys via scatter histogram.
    pub fn entropy_from_keys(
        &mut self, keys: &[i32], n_bins: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let counts = histogram(&mut self.compute, keys, n_bins)?;
        Ok(shannon_entropy_from_counts(&counts))
    }

    /// Compute mutual information from two label vectors via scatter.
    pub fn mutual_info(
        &mut self, keys_x: &[i32], keys_y: &[i32], nx: usize, ny: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let table = joint_histogram(&mut self.compute, keys_x, keys_y, nx, ny)?;
        Ok(mutual_information(&table, nx, ny))
    }

    /// Compute NMI from two label vectors via scatter.
    pub fn nmi(
        &mut self, keys_x: &[i32], keys_y: &[i32], nx: usize, ny: usize,
        method: &str,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let table = joint_histogram(&mut self.compute, keys_x, keys_y, nx, ny)?;
        Ok(normalized_mutual_information(&table, nx, ny, method))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Transfer entropy (Schreiber 2000)
// ═══════════════════════════════════════════════════════════════════════════

/// Transfer entropy from X to Y (Schreiber 2000).
///
/// TE(X→Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
///
/// Measures the reduction in uncertainty about Y's next value given knowledge
/// of X's current value, beyond what Y's own past tells us. Asymmetric —
/// TE(X→Y) ≠ TE(Y→X) in general.
///
/// Uses bin-based estimation on quantile-symbolized data:
/// 1. Symbolize x and y into `n_bins` quantile bins
/// 2. Count joint/marginal histograms
/// 3. Compute TE from the counts
///
/// `x`, `y`: paired time series (same length).
/// `n_bins`: discretization bins (typically 3-10).
/// Returns TE in bits (log base 2).
pub fn transfer_entropy(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
    let n = x.len();
    if n < 3 || y.len() != n || n_bins < 2 { return f64::NAN; }

    // Symbolize both series into quantile bins
    let sym_x = crate::nonparametric::quantile_symbolize(x, n_bins);
    let sym_y = crate::nonparametric::quantile_symbolize(y, n_bins);

    // Build joint counts:
    // p(y_{t+1}, y_t, x_t) — 3D, shape (n_bins)³
    // p(y_t, x_t)         — 2D
    // p(y_{t+1}, y_t)     — 2D
    // p(y_t)              — 1D
    let b = n_bins;
    let mut p_yxx = vec![0.0_f64; b * b * b]; // index: y_next*b²+y_t*b+x_t
    let mut p_yx = vec![0.0_f64; b * b];      // index: y_t*b+x_t
    let mut p_yy = vec![0.0_f64; b * b];      // index: y_next*b+y_t
    let mut p_y = vec![0.0_f64; b];           // index: y_t

    let total = (n - 1) as f64;
    for t in 0..(n - 1) {
        let y_next = sym_y[t + 1] as usize;
        let y_t = sym_y[t] as usize;
        let x_t = sym_x[t] as usize;
        if y_next >= b || y_t >= b || x_t >= b { continue; }
        p_yxx[y_next * b * b + y_t * b + x_t] += 1.0;
        p_yx[y_t * b + x_t] += 1.0;
        p_yy[y_next * b + y_t] += 1.0;
        p_y[y_t] += 1.0;
    }
    if total < 1.0 { return f64::NAN; }

    // TE = Σ p(y_{t+1}, y_t, x_t) · log [ p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t) ]
    //    = Σ p(y_{t+1}, y_t, x_t) · log [ p(y_{t+1}, y_t, x_t) · p(y_t) / (p(y_t, x_t) · p(y_{t+1}, y_t)) ]
    let mut te = 0.0_f64;
    for y_next in 0..b {
        for y_t in 0..b {
            for x_t in 0..b {
                let idx3 = y_next * b * b + y_t * b + x_t;
                let n_yxx = p_yxx[idx3];
                if n_yxx < 1e-15 { continue; }
                let n_yx = p_yx[y_t * b + x_t];
                let n_yy = p_yy[y_next * b + y_t];
                let n_y = p_y[y_t];
                if n_yx < 1e-15 || n_yy < 1e-15 || n_y < 1e-15 { continue; }
                // Ratio inside log (counts cancel to probabilities)
                let ratio = (n_yxx * n_y) / (n_yx * n_yy);
                if ratio > 0.0 {
                    let p_joint = n_yxx / total;
                    te += p_joint * ratio.log2();
                }
            }
        }
    }
    te.max(0.0) // TE is non-negative by construction; guard numerical noise
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, msg: &str) {
        if a.is_nan() && b.is_nan() { return; }
        assert!((a - b).abs() < tol,
            "{}: expected {}, got {} (diff {})", msg, b, a, (a - b).abs());
    }

    // ── Shannon entropy ───────────────────────────────────────────────

    #[test]
    fn entropy_uniform() {
        // 4 equally likely outcomes → H = log(4)
        let p = vec![0.25, 0.25, 0.25, 0.25];
        close(shannon_entropy(&p), 4.0f64.ln(), 1e-10, "entropy_uniform4");
    }

    #[test]
    fn entropy_deterministic() {
        // One certain outcome → H = 0
        let p = vec![1.0, 0.0, 0.0];
        close(shannon_entropy(&p), 0.0, 1e-10, "entropy_deterministic");
    }

    #[test]
    fn entropy_binary() {
        // Fair coin: H = log(2) ≈ 0.6931
        let p = vec![0.5, 0.5];
        close(shannon_entropy(&p), 2.0f64.ln(), 1e-10, "entropy_binary");
    }

    #[test]
    fn entropy_from_counts() {
        let counts = vec![10.0, 10.0, 10.0, 10.0];
        close(shannon_entropy_from_counts(&counts), 4.0f64.ln(), 1e-10, "entropy_counts");
    }

    // ── Rényi entropy ─────────────────────────────────────────────────

    #[test]
    fn renyi_converges_to_shannon() {
        let p = vec![0.3, 0.5, 0.2];
        let h_shannon = shannon_entropy(&p);
        // α close to 1 should give Shannon
        close(renyi_entropy(&p, 0.9999), h_shannon, 1e-3, "renyi_near1_lo");
        close(renyi_entropy(&p, 1.0001), h_shannon, 1e-3, "renyi_near1_hi");
        close(renyi_entropy(&p, 1.0), h_shannon, 1e-10, "renyi_at1");
    }

    #[test]
    fn renyi_order_2() {
        // H_2 = -log(Σ p²)
        let p = vec![0.5, 0.3, 0.2];
        let sum_sq: f64 = p.iter().map(|x| x * x).sum();
        close(renyi_entropy(&p, 2.0), -sum_sq.ln(), 1e-10, "renyi_2");
    }

    #[test]
    fn renyi_uniform() {
        // For uniform distribution, all Rényi entropies equal Shannon = log(k)
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = 4.0f64.ln();
        close(renyi_entropy(&p, 0.5), h, 1e-10, "renyi_uniform_05");
        close(renyi_entropy(&p, 2.0), h, 1e-10, "renyi_uniform_2");
        close(renyi_entropy(&p, 10.0), h, 1e-10, "renyi_uniform_10");
    }

    // ── Tsallis entropy ───────────────────────────────────────────────

    #[test]
    fn tsallis_converges_to_shannon() {
        let p = vec![0.3, 0.5, 0.2];
        close(tsallis_entropy(&p, 1.0), shannon_entropy(&p), 1e-10, "tsallis_at1");
    }

    // ── KL divergence ─────────────────────────────────────────────────

    #[test]
    fn kl_identical() {
        let p = vec![0.3, 0.5, 0.2];
        close(kl_divergence(&p, &p), 0.0, 1e-10, "kl_identical");
    }

    #[test]
    fn kl_asymmetric() {
        let p = vec![0.9, 0.1];
        let q = vec![0.1, 0.9];
        let kl_pq = kl_divergence(&p, &q);
        let kl_qp = kl_divergence(&q, &p);
        // Should be equal for this symmetric swap, but KL itself is asymmetric in general
        close(kl_pq, kl_qp, 1e-10, "kl_symmetric_swap");
        assert!(kl_pq > 0.0, "KL must be non-negative");
    }

    #[test]
    fn kl_zero_in_q() {
        // q has zero where p has mass → KL = +∞
        let p = vec![0.5, 0.5];
        let q = vec![1.0, 0.0];
        assert_eq!(kl_divergence(&p, &q), f64::INFINITY);
    }

    // ── JS divergence ─────────────────────────────────────────────────

    #[test]
    fn js_identical() {
        let p = vec![0.3, 0.5, 0.2];
        close(js_divergence(&p, &p), 0.0, 1e-10, "js_identical");
    }

    #[test]
    fn js_symmetric() {
        let p = vec![0.9, 0.1];
        let q = vec![0.1, 0.9];
        close(js_divergence(&p, &q), js_divergence(&q, &p), 1e-10, "js_symmetric");
    }

    #[test]
    fn js_bounded() {
        // JS ∈ [0, ln(2)]
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let js = js_divergence(&p, &q);
        assert!(js <= 2.0f64.ln() + 1e-10, "JS must be ≤ ln(2)");
        close(js, 2.0f64.ln(), 1e-10, "js_max"); // disjoint supports → max
    }

    // ── Cross-entropy ─────────────────────────────────────────────────

    #[test]
    fn cross_entropy_self() {
        // H(P, P) = H(P)
        let p = vec![0.3, 0.5, 0.2];
        close(cross_entropy(&p, &p), shannon_entropy(&p), 1e-10, "ce_self");
    }

    #[test]
    fn cross_entropy_relation() {
        // H(P, Q) = H(P) + D_KL(P || Q)
        let p = vec![0.7, 0.2, 0.1];
        let q = vec![0.4, 0.4, 0.2];
        let expected = shannon_entropy(&p) + kl_divergence(&p, &q);
        close(cross_entropy(&p, &q), expected, 1e-10, "ce_relation");
    }

    // ── Mutual information ────────────────────────────────────────────

    #[test]
    fn mi_independent() {
        // Independent: X and Y have no shared information
        // p(x,y) = p(x) × p(y) for all x,y → MI = 0
        let table = vec![
            0.15, 0.10,  // row 0
            0.45, 0.30,  // row 1
        ];
        // row sums: [0.25, 0.75], col sums: [0.6, 0.4]
        // p(0,0) = 0.15, p(0)*p(0) = 0.25*0.6 = 0.15 ✓ independent
        close(mutual_information(&table, 2, 2), 0.0, 1e-10, "mi_independent");
    }

    #[test]
    fn mi_perfect() {
        // Perfect correlation: Y = X (2 classes)
        // contingency: [[5,0],[0,5]]
        let table = vec![5.0, 0.0, 0.0, 5.0];
        let mi = mutual_information(&table, 2, 2);
        // MI = H(X) = H(Y) = log(2)
        close(mi, 2.0f64.ln(), 1e-10, "mi_perfect");
    }

    #[test]
    fn nmi_perfect() {
        let table = vec![5.0, 0.0, 0.0, 5.0];
        close(normalized_mutual_information(&table, 2, 2, "arithmetic"), 1.0, 1e-10, "nmi_perfect");
    }

    #[test]
    fn vi_perfect() {
        let table = vec![5.0, 0.0, 0.0, 5.0];
        close(variation_of_information(&table, 2, 2), 0.0, 1e-10, "vi_perfect");
    }

    #[test]
    fn conditional_entropy_independent() {
        // H(Y|X) = H(Y) when independent
        let table = vec![0.15, 0.10, 0.45, 0.30];
        let hy = shannon_entropy(&probabilities(&[0.6, 0.4]));
        close(conditional_entropy(&table, 2, 2), hy, 1e-10, "cond_ent_indep");
    }

    // ── Clustering evaluation ─────────────────────────────────────────

    #[test]
    fn mi_score_perfect() {
        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let pred_labels = vec![0, 0, 1, 1, 2, 2];
        let mi = mutual_info_score(&true_labels, &pred_labels);
        assert!(mi > 0.0, "perfect clustering should have positive MI");
    }

    #[test]
    fn nmi_score_perfect() {
        let true_labels = vec![0, 0, 1, 1];
        let pred_labels = vec![0, 0, 1, 1];
        close(normalized_mutual_info_score(&true_labels, &pred_labels, "arithmetic"),
              1.0, 1e-10, "nmi_score_perfect");
    }

    #[test]
    fn nmi_score_permuted() {
        // Permuted labels should still give NMI = 1 (label names don't matter)
        let true_labels = vec![0, 0, 1, 1];
        let pred_labels = vec![1, 1, 0, 0];
        close(normalized_mutual_info_score(&true_labels, &pred_labels, "arithmetic"),
              1.0, 1e-10, "nmi_permuted");
    }

    #[test]
    fn ami_independent_is_nonpositive() {
        // Orthogonal labeling: MI = 0, E[MI] > 0 for small n → AMI ≤ 0
        let true_labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let pred_labels = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let ami = adjusted_mutual_info_score(&true_labels, &pred_labels);
        assert!(ami <= 0.01, "independent labeling AMI should be ≤ 0, got {}", ami);
    }

    #[test]
    fn ami_perfect() {
        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let pred_labels = vec![0, 0, 1, 1, 2, 2];
        close(adjusted_mutual_info_score(&true_labels, &pred_labels), 1.0, 1e-10, "ami_perfect");
    }

    // ── Histogram via scatter ─────────────────────────────────────────

    #[test]
    fn histogram_basic() {
        let mut compute = ComputeEngine::new(tam_gpu::detect());
        let keys = vec![0i32, 1, 1, 2, 2, 2];
        let counts = histogram(&mut compute, &keys, 3).unwrap();
        close(counts[0], 1.0, 1e-10, "hist_0");
        close(counts[1], 2.0, 1e-10, "hist_1");
        close(counts[2], 3.0, 1e-10, "hist_2");
    }

    #[test]
    fn joint_histogram_basic() {
        let mut compute = ComputeEngine::new(tam_gpu::detect());
        let kx = vec![0i32, 0, 1, 1];
        let ky = vec![0i32, 1, 0, 1];
        let table = joint_histogram(&mut compute, &kx, &ky, 2, 2).unwrap();
        // (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1
        for &c in &table { close(c, 1.0, 1e-10, "joint_hist"); }
    }

    // ── Continuous entropy ────────────────────────────────────────────

    #[test]
    fn entropy_histogram_constant() {
        close(entropy_histogram(&[5.0, 5.0, 5.0], 10), 0.0, 1e-10, "ent_hist_const");
    }

    #[test]
    fn entropy_histogram_with_nan() {
        let vals = vec![1.0, 2.0, f64::NAN, 3.0, 4.0];
        let h = entropy_histogram(&vals, 4);
        assert!(!h.is_nan(), "NaN values should be excluded");
    }

    // ── InformationEngine ─────────────────────────────────────────────

    #[test]
    fn engine_entropy() {
        let mut engine = InformationEngine::new();
        let keys = vec![0i32, 0, 1, 1]; // two equally likely bins
        let h = engine.entropy_from_keys(&keys, 2).unwrap();
        close(h, 2.0f64.ln(), 1e-10, "engine_entropy");
    }

    #[test]
    fn engine_mutual_info() {
        let mut engine = InformationEngine::new();
        // Perfect correlation: x=y
        let kx = vec![0i32, 0, 1, 1];
        let ky = vec![0i32, 0, 1, 1];
        let mi = engine.mutual_info(&kx, &ky, 2, 2).unwrap();
        close(mi, 2.0f64.ln(), 1e-10, "engine_mi");
    }
}
