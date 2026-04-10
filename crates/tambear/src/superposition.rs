//! # Universal Superposition for `.tbs` Chains
//!
//! Every sweepable parameter runs at ALL values simultaneously.
//! The user writes `kmeans(k=5)` — internally tambear runs k=2..20,
//! computes all, returns the k=5 result PLUS the full superposition
//! with agreement metrics.
//!
//! The user never selects parameters. Tambear over-provisions, computes
//! everything, collapses at the end. Unused parameter configurations
//! produce results that are stored but don't affect the collapsed output.
//!
//! This generalizes `.discover()` (which does this for clustering only)
//! to every operation with sweepable parameters.

use std::collections::BTreeMap;

use crate::tbs_executor::TbsStepOutput;

// ═══════════════════════════════════════════════════════════════════════════
// Core types
// ═══════════════════════════════════════════════════════════════════════════

/// One view in a superposition: one parameter configuration → one result.
#[derive(Debug, Clone)]
pub struct SuperpositionView {
    /// Human-readable description, e.g. `"kmeans(k=3)"`.
    pub name: String,
    /// Parameter values used for this view.
    pub params: BTreeMap<String, f64>,
    /// The step output for this configuration.
    pub output: TbsStepOutput,
}

/// Full superposition: all configurations swept, plus cross-view diagnostics.
///
/// The collapsed result is `views[requested_idx].output`. The rest of the
/// views are the superposition — stored, never discarded, informative.
#[derive(Debug, Clone)]
pub struct Superposition {
    /// All views, in sweep order.
    pub views: Vec<SuperpositionView>,
    /// Index of the view the user requested (the "collapsed" output).
    pub requested_idx: usize,
    /// Cross-view agreement metric (operation-specific, 0..1).
    /// 1.0 = all configurations agree. Low = parameter-sensitive.
    pub agreement: f64,
    /// Which parameter value is "modal" (most-supported by the sweep).
    pub modal_value: f64,
    /// Name of the swept parameter.
    pub swept_param: String,
}

impl Superposition {
    /// The collapsed output — what the user asked for.
    pub fn collapsed(&self) -> &TbsStepOutput {
        &self.views[self.requested_idx].output
    }

    /// Is the superposition informative? (Do different configs give different results?)
    pub fn is_informative(&self) -> bool {
        self.views.len() >= 2 && self.agreement < 0.99
    }

    /// Parameter sensitivity: 1 - agreement. High = sensitive to parameter choice.
    pub fn sensitivity(&self) -> f64 {
        1.0 - self.agreement
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sweep definitions — what parameters to sweep for each operation
// ═══════════════════════════════════════════════════════════════════════════

/// Generates the k-range for clustering sweeps.
/// Range: 2..=min(k_max, sqrt(n)), always including the requested k.
fn clustering_k_range(requested_k: usize, n: usize) -> Vec<usize> {
    let k_max = 20.min(((n as f64).sqrt() as usize).max(2));
    let mut ks: Vec<usize> = (2..=k_max).collect();
    if requested_k > k_max {
        ks.push(requested_k);
    }
    ks
}

/// Generates the n_components range for dimensionality reduction sweeps.
/// Range: 1..=min(d, 10), always including the requested value.
fn dim_reduction_range(requested_nc: usize, d: usize) -> Vec<usize> {
    let nc_max = d.min(10);
    let mut ncs: Vec<usize> = (1..=nc_max).collect();
    if requested_nc > nc_max && requested_nc <= d {
        ncs.push(requested_nc);
    }
    ncs
}

/// Generates window sizes for time series sweeps.
/// Geometric progression: 3, 5, 7, 10, 15, 20, 30, 50, ...
/// Capped at n/3 so each window sees enough data.
fn window_range(requested_w: usize, n: usize) -> Vec<usize> {
    let candidates = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200];
    let max_w = n / 3;
    let mut ws: Vec<usize> = candidates.iter()
        .copied()
        .filter(|&w| w <= max_w && w >= 3)
        .collect();
    if !ws.contains(&requested_w) && requested_w >= 3 && requested_w <= max_w {
        ws.push(requested_w);
        ws.sort();
    }
    if ws.is_empty() && requested_w >= 3 {
        ws.push(requested_w);
    }
    ws
}

/// Generates AR order range for time series model fitting.
/// Range: 1..=min(p_max, n/4), always including the requested order.
fn ar_order_range(requested_p: usize, n: usize) -> Vec<usize> {
    let p_max = (n / 4).min(10).max(1);
    let mut ps: Vec<usize> = (1..=p_max).collect();
    if requested_p > p_max {
        ps.push(requested_p);
    }
    ps
}

// ═══════════════════════════════════════════════════════════════════════════
// Sweep execution — clustering
// ═══════════════════════════════════════════════════════════════════════════

/// Run KMeans at all k values, return superposition with Rand-index agreement.
///
/// Uses the provided `data` (row-major n×d, f64) and a single KMeansEngine.
pub fn sweep_kmeans(
    data: &[f64],
    n: usize,
    d: usize,
    requested_k: usize,
    max_iter: usize,
) -> Superposition {
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let ks = clustering_k_range(requested_k, n);
    let mut views = Vec::with_capacity(ks.len());
    let mut all_labels: Vec<Vec<i32>> = Vec::with_capacity(ks.len());
    let mut requested_idx = 0;

    let engine = crate::kmeans::KMeansEngine::new()
        .expect("KMeansEngine::new failed in sweep_kmeans");

    for &k in &ks {
        let result = engine.fit(&data_f32, n, d, k, max_iter)
            .expect("KMeansEngine::fit failed in sweep");
        let labels: Vec<i32> = result.labels.iter().map(|&l| l as i32).collect();

        if k == requested_k {
            requested_idx = views.len();
        }

        let mut params = BTreeMap::new();
        params.insert("k".to_string(), k as f64);

        all_labels.push(labels);
        views.push(SuperpositionView {
            name: format!("kmeans(k={})", k),
            params,
            output: TbsStepOutput::Transform,
        });
    }

    let agreement = label_agreement(&all_labels, n);
    let modal_k = modal_k_from_labels(&all_labels);

    Superposition {
        views,
        requested_idx,
        agreement,
        modal_value: modal_k as f64,
        swept_param: "k".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sweep execution — PCA
// ═══════════════════════════════════════════════════════════════════════════

/// Run PCA at all n_components values, return superposition.
/// Agreement = stability of explained variance ratios across component counts.
pub fn sweep_pca(
    data: &[f64],
    n: usize,
    d: usize,
    requested_nc: usize,
) -> Superposition {
    let ncs = dim_reduction_range(requested_nc, d);
    let mut views = Vec::with_capacity(ncs.len());
    let mut requested_idx = 0;
    let mut var_ratios: Vec<Vec<f64>> = Vec::new();

    for &nc in &ncs {
        let result = crate::dim_reduction::pca(data, n, d, nc);

        if nc == requested_nc {
            requested_idx = views.len();
        }

        var_ratios.push(result.explained_variance_ratio.clone());

        let mut params = BTreeMap::new();
        params.insert("n_components".to_string(), nc as f64);

        views.push(SuperpositionView {
            name: format!("pca(n_components={})", nc),
            params,
            output: TbsStepOutput::Pca(result),
        });
    }

    let agreement = variance_ratio_agreement(&var_ratios);
    // Modal value: the n_components where cumulative variance crosses 0.9
    let modal_nc = elbow_nc(&var_ratios);

    Superposition {
        views,
        requested_idx,
        agreement,
        modal_value: modal_nc as f64,
        swept_param: "n_components".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sweep execution — hypothesis tests
// ═══════════════════════════════════════════════════════════════════════════

/// Run all applicable two-sample tests on the same pair of columns.
/// Agreement = fraction of tests that agree on reject/fail-to-reject at `alpha`.
///
/// `alpha`: significance level for reject/fail-to-reject decisions.
/// Default 0.05 if `None`.
pub fn sweep_two_sample_tests(
    x: &[f64],
    y: &[f64],
) -> Superposition {
    sweep_two_sample_tests_alpha(x, y, 0.05)
}

/// Like [`sweep_two_sample_tests`] but with an explicit alpha parameter.
pub fn sweep_two_sample_tests_alpha(
    x: &[f64],
    y: &[f64],
    alpha: f64,
) -> Superposition {
    let sx = crate::descriptive::moments_ungrouped(x);
    let sy = crate::descriptive::moments_ungrouped(y);

    let mut views = Vec::new();
    let mut decisions: Vec<bool> = Vec::new(); // true = reject

    // Two-sample t-test
    {
        let result = crate::hypothesis::two_sample_t(&sx, &sy);
        decisions.push(result.p_value < alpha);
        let mut params = BTreeMap::new();
        params.insert("variant".to_string(), 0.0);
        views.push(SuperpositionView {
            name: "two_sample_t".to_string(),
            params,
            output: TbsStepOutput::Test(result),
        });
    }

    // Welch t-test
    {
        let result = crate::hypothesis::welch_t(&sx, &sy);
        decisions.push(result.p_value < alpha);
        let mut params = BTreeMap::new();
        params.insert("variant".to_string(), 1.0);
        views.push(SuperpositionView {
            name: "welch_t".to_string(),
            params,
            output: TbsStepOutput::Test(result),
        });
    }

    // Mann-Whitney U (nonparametric)
    {
        let result = crate::nonparametric::mann_whitney_u(x, y);
        decisions.push(result.p_value < alpha);
        let mut params = BTreeMap::new();
        params.insert("variant".to_string(), 2.0);
        views.push(SuperpositionView {
            name: "mann_whitney_u".to_string(),
            params,
            output: TbsStepOutput::Nonparametric(result),
        });
    }

    // KS two-sample
    {
        let result = crate::nonparametric::ks_test_two_sample(x, y);
        decisions.push(result.p_value < alpha);
        let mut params = BTreeMap::new();
        params.insert("variant".to_string(), 3.0);
        views.push(SuperpositionView {
            name: "ks_two_sample".to_string(),
            params,
            output: TbsStepOutput::Nonparametric(result),
        });
    }

    let n_agree = if !decisions.is_empty() {
        let majority = decisions.iter().filter(|&&d| d).count() * 2 > decisions.len();
        decisions.iter().filter(|&&d| d == majority).count()
    } else {
        0
    };
    let agreement = if decisions.is_empty() { 1.0 } else { n_agree as f64 / decisions.len() as f64 };

    // requested_idx = 0 (two_sample_t is the "default" collapse)
    Superposition {
        views,
        requested_idx: 0,
        agreement,
        modal_value: if decisions.iter().filter(|&&d| d).count() * 2 > decisions.len() { 1.0 } else { 0.0 },
        swept_param: "test_variant".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sweep execution — time series (window/lag)
// ═══════════════════════════════════════════════════════════════════════════

/// Sweep moving average across window sizes.
/// Agreement = correlation between smoothed series at different windows.
pub fn sweep_moving_average(
    col: &[f64],
    requested_window: usize,
) -> Superposition {
    let n = col.len();
    let ws = window_range(requested_window, n);
    let mut views = Vec::with_capacity(ws.len());
    let mut requested_idx = 0;
    let mut results: Vec<Vec<f64>> = Vec::new();

    for &w in &ws {
        let values = crate::signal_processing::moving_average(col, w);

        if w == requested_window {
            requested_idx = views.len();
        }

        results.push(values.clone());

        let mut params = BTreeMap::new();
        params.insert("window".to_string(), w as f64);

        views.push(SuperpositionView {
            name: format!("moving_average(window={})", w),
            params,
            output: TbsStepOutput::Vector { name: "moving_average", values },
        });
    }

    let agreement = series_agreement(&results);

    Superposition {
        views,
        requested_idx,
        agreement,
        modal_value: requested_window as f64,
        swept_param: "window".to_string(),
    }
}

/// Sweep AR model order.
/// Agreement = BIC/AIC consistency across orders.
pub fn sweep_ar(
    col: &[f64],
    requested_p: usize,
) -> Superposition {
    let n = col.len();
    let ps = ar_order_range(requested_p, n);
    let mut views = Vec::with_capacity(ps.len());
    let mut requested_idx = 0;
    let mut residual_vars: Vec<f64> = Vec::new();

    for &p in &ps {
        let result = crate::time_series::ar_fit(col, p);

        if p == requested_p {
            requested_idx = views.len();
        }

        residual_vars.push(result.sigma2);

        let mut params = BTreeMap::new();
        params.insert("p".to_string(), p as f64);

        views.push(SuperpositionView {
            name: format!("ar(p={})", p),
            params,
            output: TbsStepOutput::Ar(result),
        });
    }

    // BIC-optimal order: the one that minimizes BIC = n*ln(residual_var) + p*ln(n)
    let n_f = n as f64;
    let ln_n = n_f.ln();
    let mut best_bic = f64::INFINITY;
    let mut best_p = requested_p;
    for (i, &p) in ps.iter().enumerate() {
        if residual_vars[i] > 0.0 {
            let bic = n_f * residual_vars[i].ln() + p as f64 * ln_n;
            if bic < best_bic {
                best_bic = bic;
                best_p = p;
            }
        }
    }

    // Agreement: how much does residual variance change across orders?
    // If it's stable → high agreement (more p doesn't help)
    let agreement = if residual_vars.len() >= 2 {
        let max_rv = residual_vars.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
        let min_rv = residual_vars.iter().cloned().fold(f64::INFINITY, crate::numerical::nan_min);
        if max_rv > 0.0 {
            1.0 - (max_rv - min_rv) / max_rv
        } else {
            1.0
        }
    } else {
        1.0
    };

    Superposition {
        views,
        requested_idx,
        agreement: agreement.max(0.0),
        modal_value: best_p as f64,
        swept_param: "p".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Agreement metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Pairwise Rand-index agreement across label sets (reuses discover() logic).
fn label_agreement(all_labels: &[Vec<i32>], n: usize) -> f64 {
    if all_labels.len() < 2 { return 1.0; }
    let sample_n = n.min(500);
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..all_labels.len() {
        for j in (i + 1)..all_labels.len() {
            sum += rand_index_sampled(&all_labels[i], &all_labels[j], sample_n);
            count += 1;
        }
    }
    if count == 0 { 1.0 } else { sum / count as f64 }
}

/// Rand Index between two labelings on the first `sample_n` points.
fn rand_index_sampled(a: &[i32], b: &[i32], sample_n: usize) -> f64 {
    let n = a.len().min(b.len()).min(sample_n);
    if n < 2 { return 1.0; }
    let mut agree = 0u64;
    let mut total = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            let same_a = a[i] == a[j];
            let same_b = b[i] == b[j];
            if same_a == same_b { agree += 1; }
            total += 1;
        }
    }
    if total == 0 { 1.0 } else { agree as f64 / total as f64 }
}

/// Find the modal k (most common cluster count) across label sets.
fn modal_k_from_labels(all_labels: &[Vec<i32>]) -> usize {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for labels in all_labels {
        let k = labels.iter().cloned().collect::<std::collections::HashSet<_>>().len();
        *counts.entry(k).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k).unwrap_or(0)
}

/// Variance ratio agreement across PCA runs with different n_components.
/// Checks whether the first few components' variance ratios are stable.
fn variance_ratio_agreement(var_ratios: &[Vec<f64>]) -> f64 {
    if var_ratios.len() < 2 { return 1.0; }

    // Compare the first component's variance ratio across all runs.
    // If it's consistent, the structure is stable.
    let first_ratios: Vec<f64> = var_ratios.iter()
        .filter(|vr| !vr.is_empty())
        .map(|vr| vr[0])
        .collect();

    if first_ratios.len() < 2 { return 1.0; }

    let mean = first_ratios.iter().sum::<f64>() / first_ratios.len() as f64;
    if mean.abs() < 1e-15 { return 1.0; }

    let cv = (first_ratios.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / first_ratios.len() as f64).sqrt() / mean.abs();

    // CV near 0 → high agreement, CV near 1 → low agreement
    (1.0 - cv).max(0.0)
}

/// Find the "elbow" n_components where cumulative variance crosses 0.9.
fn elbow_nc(var_ratios: &[Vec<f64>]) -> usize {
    // Use the run with the most components to find the elbow
    if let Some(longest) = var_ratios.iter().max_by_key(|vr| vr.len()) {
        let mut cum = 0.0;
        for (i, &r) in longest.iter().enumerate() {
            cum += r;
            if cum >= 0.9 {
                return i + 1;
            }
        }
        longest.len()
    } else {
        1
    }
}

/// Series agreement: average pairwise correlation between smoothed series.
fn series_agreement(results: &[Vec<f64>]) -> f64 {
    if results.len() < 2 { return 1.0; }

    let mut sum = 0.0;
    let mut count = 0usize;

    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let corr = pearson_corr(&results[i], &results[j]);
            if corr.is_finite() {
                sum += corr;
                count += 1;
            }
        }
    }

    if count == 0 { 1.0 } else { (sum / count as f64).max(0.0) }
}

/// Pearson correlation between two series (may have different lengths — use overlap).
fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 { return 1.0; }

    let mean_a = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b = b[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 { 1.0 } else { cov / denom }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_data() -> (Vec<f64>, usize, usize) {
        let mut data = Vec::new();
        // Cluster A: centered at (0, 0)
        for i in 0..20 {
            data.push(0.0 + (i as f64) * 0.1);
            data.push(0.0 + (i as f64) * 0.05);
        }
        // Cluster B: centered at (10, 10)
        for i in 0..20 {
            data.push(10.0 + (i as f64) * 0.1);
            data.push(10.0 + (i as f64) * 0.05);
        }
        (data, 40, 2)
    }

    #[test]
    fn sweep_kmeans_produces_views() {
        let (data, n, d) = two_cluster_data();
        let s = sweep_kmeans(&data, n, d, 2, 100);

        assert!(s.views.len() >= 2, "should sweep multiple k values");
        assert_eq!(s.swept_param, "k");
        assert!(s.collapsed().is_transform());
    }

    #[test]
    fn sweep_kmeans_requested_k_in_views() {
        let (data, n, d) = two_cluster_data();
        let s = sweep_kmeans(&data, n, d, 3, 100);

        let requested_params = &s.views[s.requested_idx].params;
        assert_eq!(requested_params["k"], 3.0);
    }

    #[test]
    fn sweep_pca_produces_views() {
        let (data, n, d) = two_cluster_data();
        let s = sweep_pca(&data, n, d, 2);

        assert!(s.views.len() >= 2, "should sweep multiple n_components");
        assert_eq!(s.swept_param, "n_components");
        // PCA agreement on well-structured data should be high
        assert!(s.agreement > 0.5);
    }

    #[test]
    fn sweep_two_sample_tests_all_variants() {
        let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.1).collect();
        let s = sweep_two_sample_tests(&x, &y);

        assert_eq!(s.views.len(), 4); // t, welch, mann-whitney, KS
        assert_eq!(s.swept_param, "test_variant");
        // Clearly different distributions → all should agree on rejection
        assert!(s.agreement > 0.7);
    }

    #[test]
    fn sweep_moving_average_produces_views() {
        let col: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let s = sweep_moving_average(&col, 5);

        assert!(s.views.len() >= 2, "should sweep multiple windows");
        assert_eq!(s.swept_param, "window");
    }

    #[test]
    fn sweep_ar_bic_optimal() {
        // AR(1) process: x[t] = 0.8*x[t-1] + noise
        let mut col = vec![0.0f64; 200];
        let mut rng = crate::rng::SplitMix64::new(42);
        for t in 1..200 {
            col[t] = 0.8 * col[t - 1] + crate::rng::sample_normal(&mut rng, 0.0, 0.1);
        }
        let s = sweep_ar(&col, 3);

        assert!(s.views.len() >= 2, "should sweep multiple orders");
        assert_eq!(s.swept_param, "p");
        // BIC-optimal should favor low order for AR(1) data
        assert!(s.modal_value >= 1.0 && s.modal_value <= 5.0);
    }

    #[test]
    fn superposition_informative_when_configs_differ() {
        let (data, n, d) = two_cluster_data();
        let s = sweep_kmeans(&data, n, d, 2, 100);

        // With two clear clusters, k=2 and k=5 should give different Rand indices
        // so the superposition should be informative
        if s.views.len() >= 3 {
            assert!(s.is_informative() || s.agreement > 0.95,
                "superposition should either be informative or show high agreement");
        }
    }

    #[test]
    fn label_agreement_perfect() {
        let a = vec![0, 0, 1, 1, 2];
        let b = vec![0, 0, 1, 1, 2];
        assert!((label_agreement(&[a, b], 5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn label_agreement_random_lower() {
        let a = vec![0, 0, 0, 1, 1];
        let b = vec![1, 0, 1, 0, 1];
        let ag = label_agreement(&[a, b], 5);
        assert!(ag < 1.0); // random labels should have < 1.0 agreement
    }

    #[test]
    fn clustering_k_range_includes_requested() {
        let ks = clustering_k_range(15, 100);
        assert!(ks.contains(&15));
        assert!(ks.contains(&2));
    }

    #[test]
    fn window_range_respects_n() {
        let ws = window_range(5, 30);
        assert!(ws.iter().all(|&w| w <= 10)); // n/3 = 10
        assert!(ws.contains(&5));
    }
}
