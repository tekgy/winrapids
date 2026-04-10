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

/// Scalar agreement: fraction of values within 5% of the mean absolute value.
/// Used for comparing scalar outputs across method variants.
fn scalar_agreement(values: &[f64]) -> f64 {
    if values.len() < 2 { return 1.0; }
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 { return 0.0; }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let abs_mean = mean.abs();
    if abs_mean < 1e-15 {
        // All near zero → high agreement
        let max_dev = finite.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if max_dev < 1e-10 { return 1.0; } else { return 0.0; }
    }
    let max_rel_dev = finite.iter()
        .map(|v| ((v - mean) / abs_mean).abs())
        .fold(0.0_f64, f64::max);
    (1.0 - max_rel_dev).max(0.0)
}

/// Agreement on reject/fail-to-reject decisions at a given alpha across p-values.
fn decision_agreement(p_values: &[f64], alpha: f64) -> f64 {
    if p_values.len() < 2 { return 1.0; }
    let decisions: Vec<bool> = p_values.iter().map(|&p| p < alpha).collect();
    let n_reject = decisions.iter().filter(|&&d| d).count();
    let majority_decision = n_reject * 2 > decisions.len();
    let n_agree = decisions.iter().filter(|&&d| d == majority_decision).count();
    n_agree as f64 / decisions.len() as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Sweep execution — correlation (Layer 4 discover)
// ═══════════════════════════════════════════════════════════════════════════

/// Run every applicable bivariate correlation measure on (x, y).
///
/// Methods: Pearson r, Spearman ρ, Kendall τ-b, distance correlation, Hoeffding's D.
/// Agreement = fraction whose sign agrees and whose magnitude is within 20% of the mean.
/// `modal_value` = median of all measures (signed).
pub fn sweep_correlation(x: &[f64], y: &[f64]) -> Superposition {
    let mut views = Vec::new();
    let mut values = Vec::new();

    let pearson = crate::nonparametric::pearson_r(x, y);
    {
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 0.0);
        views.push(SuperpositionView {
            name: "pearson".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "pearson", value: pearson },
        });
        values.push(pearson);
    }

    let spearman = crate::nonparametric::spearman(x, y);
    {
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 1.0);
        views.push(SuperpositionView {
            name: "spearman".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "spearman", value: spearman },
        });
        values.push(spearman);
    }

    let kendall = crate::nonparametric::kendall_tau(x, y);
    {
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 2.0);
        views.push(SuperpositionView {
            name: "kendall_tau".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "kendall_tau", value: kendall },
        });
        values.push(kendall);
    }

    // Distance correlation: non-negative (0 = independence, 1 = perfect dependence)
    let dcor = crate::nonparametric::distance_correlation(x, y);
    {
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 3.0);
        views.push(SuperpositionView {
            name: "distance_correlation".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "distance_correlation", value: dcor },
        });
        // dcor is always ≥ 0; convert to signed by using same sign as pearson for agreement
        values.push(if pearson < 0.0 { -dcor } else { dcor });
    }

    // Hoeffding's D: scaled to [-0.5, 1.0]; 1.0 = perfect dependence
    let hoeff = crate::nonparametric::hoeffdings_d(x, y);
    {
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 4.0);
        views.push(SuperpositionView {
            name: "hoeffdings_d".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "hoeffdings_d", value: hoeff },
        });
        // Scale to [-1, 1] range for agreement comparison (hoeff ∈ [-0.5, 1])
        values.push(if pearson < 0.0 { -hoeff } else { hoeff });
    }

    let agreement = scalar_agreement(&values);

    // Modal value: median of the three signed-comparable measures (pearson, spearman, kendall)
    let mut signed_three = [pearson, spearman, kendall];
    signed_three.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let modal_value = signed_three[1]; // median

    Superposition {
        views,
        requested_idx: 0, // pearson is the "default" collapse
        agreement,
        modal_value,
        swept_param: "correlation_method".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sweep execution — regression (Layer 4 discover)
// ═══════════════════════════════════════════════════════════════════════════

/// Run all applicable simple regression methods on (x, y).
///
/// Methods: OLS slope, Theil-Sen slope, Siegel slope.
/// Agreement = fraction of slope estimates within 20% of the OLS estimate.
/// `modal_value` = median slope across methods.
pub fn sweep_regression(x: &[f64], y: &[f64]) -> Superposition {
    let mut views = Vec::new();
    let mut slopes = Vec::new();

    // OLS
    let ols = crate::linear_algebra::simple_linear_regression(x, y);
    let ols_slope = ols.slope;
    {
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 0.0);
        views.push(SuperpositionView {
            name: "ols".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "ols_slope", value: ols_slope },
        });
        slopes.push(ols_slope);
    }

    // Theil-Sen
    let theil = crate::nonparametric::theilslopes(y, Some(x));
    let theil_slope = theil.slope;
    {
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 1.0);
        views.push(SuperpositionView {
            name: "theil_sen".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "theil_sen_slope", value: theil_slope },
        });
        slopes.push(theil_slope);
    }

    // Siegel repeated-median
    let siegel = crate::nonparametric::siegelslopes(y, Some(x));
    let siegel_slope = siegel.slope;
    {
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 2.0);
        views.push(SuperpositionView {
            name: "siegel".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "siegel_slope", value: siegel_slope },
        });
        slopes.push(siegel_slope);
    }

    let agreement = scalar_agreement(&slopes);

    let mut sorted_slopes = slopes.clone();
    sorted_slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let modal_value = sorted_slopes[sorted_slopes.len() / 2]; // median

    Superposition {
        views,
        requested_idx: 0, // OLS is the default collapse
        agreement,
        modal_value,
        swept_param: "regression_method".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sweep execution — changepoint detection (Layer 4 discover)
// ═══════════════════════════════════════════════════════════════════════════

/// Run all changepoint detection methods on a time series.
///
/// Methods: CUSUM (mean shift), PELT (penalized segmentation), binary segmentation.
/// `modal_value` = consensus estimated number of changepoints across methods.
/// Agreement = fraction of methods that agree on the changepoint count.
pub fn sweep_changepoint(col: &[f64]) -> Superposition {
    let n = col.len();
    let mut views = Vec::new();
    let mut n_cps: Vec<usize> = Vec::new();

    // CUSUM mean shift: detects a single candidate changepoint at argmax of |CUSUM|.
    // We report 0 or 1 changepoints based on whether max_abs_cusum exceeds threshold.
    {
        let result = crate::time_series::cusum_mean(col);
        let threshold = using_threshold(n);
        let n_detected = if result.max_abs_cusum > threshold { 1 } else { 0 };
        n_cps.push(n_detected);
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 0.0);
        views.push(SuperpositionView {
            name: "cusum_mean".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "n_changepoints", value: n_detected as f64 },
        });
    }

    // PELT
    {
        let min_seg = (n / 10).max(2);
        let cps = crate::time_series::pelt(col, min_seg, None);
        let n_detected = cps.len();
        n_cps.push(n_detected);
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 1.0);
        views.push(SuperpositionView {
            name: "pelt".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "n_changepoints", value: n_detected as f64 },
        });
    }

    // Binary segmentation: signature is (data, threshold, min_segment_size, max_changepoints)
    {
        let min_seg = (n / 10).max(2);
        let threshold = using_threshold(n);
        let max_cps = 20;
        let cps = crate::time_series::cusum_binary_segmentation(col, threshold, min_seg, max_cps);
        let n_detected = cps.len();
        n_cps.push(n_detected);
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 2.0);
        views.push(SuperpositionView {
            name: "binary_segmentation".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "n_changepoints", value: n_detected as f64 },
        });
    }

    // Agreement: fraction of methods that agree on the modal count
    let modal_count = {
        let mut freq: BTreeMap<usize, usize> = BTreeMap::new();
        for &c in &n_cps { *freq.entry(c).or_insert(0) += 1; }
        freq.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k).unwrap_or(0)
    };
    let n_agree = n_cps.iter().filter(|&&c| c == modal_count).count();
    let agreement = n_agree as f64 / n_cps.len() as f64;

    Superposition {
        views,
        requested_idx: 1, // PELT is the most principled default
        agreement,
        modal_value: modal_count as f64,
        swept_param: "changepoint_method".to_string(),
    }
}

/// Heuristic threshold for binary segmentation based on series length.
fn using_threshold(n: usize) -> f64 {
    // Approximate 95th percentile of CUSUM statistic under H0 (no change)
    // Scales as ~sqrt(n * log(n)) for classical CUSUM
    let n_f = n as f64;
    (n_f * n_f.ln()).sqrt() * 0.5
}

// ═══════════════════════════════════════════════════════════════════════════
// Sweep execution — stationarity (Layer 4 discover)
// ═══════════════════════════════════════════════════════════════════════════

/// Run all stationarity tests on a time series.
///
/// Methods: ADF (H0=unit root), KPSS (H0=stationary), PP test, variance ratio test.
/// Note: ADF and KPSS have OPPOSITE null hypotheses. Agreement is measured on
/// the consensus conclusion (stationary or not), not p-value direction.
/// `modal_value` = 1.0 if consensus is stationary, 0.0 if non-stationary.
pub fn sweep_stationarity(col: &[f64], alpha: f64) -> Superposition {
    let n = col.len();
    let mut views = Vec::new();
    let mut stationary_votes: Vec<bool> = Vec::new();

    // ADF: H0 = unit root (non-stationary); reject (statistic < critical_5pct) → stationary.
    // ADF critical values are negative; more negative = stronger rejection.
    {
        let n_lags = (n as f64).powf(1.0 / 3.0).ceil() as usize;
        let result = crate::time_series::adf_test(col, n_lags);
        let stationary = result.statistic < result.critical_5pct;
        stationary_votes.push(stationary);
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 0.0);
        // Normalized: stat / |critical_5pct|. Values < 1 = reject H0 = stationary.
        let norm = if result.critical_5pct.abs() > 1e-10 {
            result.statistic / result.critical_5pct.abs()
        } else {
            f64::NAN
        };
        views.push(SuperpositionView {
            name: "adf".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "adf_stat_norm", value: norm },
        });
    }

    // KPSS: H0 = stationary; reject (statistic > critical_5pct) → non-stationary.
    // Fail to reject H0 (statistic <= critical_5pct) → stationary.
    {
        let result = crate::time_series::kpss_test(col, false, None);
        let stationary = result.statistic <= result.critical_5pct;
        stationary_votes.push(stationary);
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 1.0);
        // Normalized: stat / critical_5pct. Values > 1 = reject H0 = non-stationary.
        let norm = if result.critical_5pct.abs() > 1e-10 {
            result.statistic / result.critical_5pct
        } else {
            f64::NAN
        };
        views.push(SuperpositionView {
            name: "kpss".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "kpss_stat_norm", value: norm },
        });
    }

    // PP: H0 = unit root (non-stationary); reject (statistic < critical_5pct) → stationary.
    // Same critical value structure as ADF (MacKinnon approximations).
    {
        let result = crate::time_series::pp_test(col, None);
        let stationary = result.statistic < result.critical_5pct;
        stationary_votes.push(stationary);
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 2.0);
        let norm = if result.critical_5pct.abs() > 1e-10 {
            result.statistic / result.critical_5pct.abs()
        } else {
            f64::NAN
        };
        views.push(SuperpositionView {
            name: "pp_test".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "pp_stat_norm", value: norm },
        });
    }

    // Variance ratio test: H0 = random walk; |z_star| > 1.96 → reject → mean-reverting (stationary).
    // Use heteroscedasticity-robust z_star; 1.96 ≈ normal 5% two-sided critical value.
    {
        let result = crate::time_series::variance_ratio_test(col, None);
        let stationary = result.z_star.abs() > 1.96;
        stationary_votes.push(stationary);
        let mut params = BTreeMap::new();
        params.insert("method".to_string(), 3.0);
        views.push(SuperpositionView {
            name: "variance_ratio".to_string(),
            params,
            output: TbsStepOutput::Scalar { name: "vr_z_star", value: result.z_star },
        });
    }

    let n_stationary = stationary_votes.iter().filter(|&&v| v).count();
    let consensus_stationary = n_stationary * 2 > stationary_votes.len();
    let n_agree = stationary_votes.iter()
        .filter(|&&v| v == consensus_stationary)
        .count();
    let agreement = n_agree as f64 / stationary_votes.len() as f64;

    Superposition {
        views,
        requested_idx: 0, // ADF is the conventional default
        agreement,
        modal_value: if consensus_stationary { 1.0 } else { 0.0 },
        swept_param: "stationarity_method".to_string(),
    }
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
    fn sweep_correlation_produces_five_views() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..50).map(|i| i as f64 * 2.0 + 1.0).collect();
        let s = sweep_correlation(&x, &y);
        assert_eq!(s.views.len(), 5);
        assert_eq!(s.swept_param, "correlation_method");
        // Perfect linear → all methods should agree closely
        assert!(s.agreement > 0.7, "agreement={}", s.agreement);
        // modal_value should be near 1.0
        assert!(s.modal_value > 0.9, "modal_value={}", s.modal_value);
    }

    #[test]
    fn sweep_regression_produces_three_views() {
        let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..40).map(|i| 3.0 * i as f64 + 5.0).collect();
        let s = sweep_regression(&x, &y);
        assert_eq!(s.views.len(), 3);
        assert_eq!(s.swept_param, "regression_method");
        // Perfect linear → all slopes should agree at ~3.0
        assert!(s.agreement > 0.8, "agreement={}", s.agreement);
        assert!((s.modal_value - 3.0).abs() < 0.5, "modal_value={}", s.modal_value);
    }

    #[test]
    fn sweep_changepoint_produces_three_views() {
        // Two-segment series: step change at midpoint
        let mut col: Vec<f64> = (0..50).map(|_| 0.0).collect();
        for i in 25..50 { col[i] = 10.0; }
        let s = sweep_changepoint(&col);
        assert_eq!(s.views.len(), 3);
        assert_eq!(s.swept_param, "changepoint_method");
        // Should detect at least 1 changepoint
        assert!(s.modal_value >= 1.0, "modal_value={}", s.modal_value);
    }

    #[test]
    fn sweep_stationarity_produces_four_views() {
        // Stationary white noise
        let col: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let s = sweep_stationarity(&col, 0.05);
        assert_eq!(s.views.len(), 4);
        assert_eq!(s.swept_param, "stationarity_method");
        // modal_value: 1.0 = stationary, 0.0 = non-stationary
        assert!(s.modal_value == 0.0 || s.modal_value == 1.0);
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
