//! Layer 1 auto-detection — diagnostic-driven method selection.
//!
//! Each function here answers the question "which primitive should I call for
//! this data?" without touching dispatch logic or output formatting.  The
//! executor calls these functions and receives a `(result, TbsStepAdvice)` pair;
//! it is then responsible for wrapping the result in the appropriate
//! `TbsStepOutput` variant.
//!
//! ## What lives here (Layer 1)
//! - Normality testing (SW vs D'Agostino-Pearson selection)
//! - Binary-variable detection
//! - Variance-equality testing (Levene/Brown-Forsythe)
//! - Method routing trees (correlation, two-sample test, ANOVA, PCA, volatility)
//!
//! ## What does NOT live here (Layer 0)
//! - Math primitives (`pearson_r`, `shapiro_wilk`, `garch11_fit`, …)
//!
//! ## What does NOT live here (Layer 2)
//! - `using()` persistence / consumption — that stays in the executor
//! - Output formatting / `TbsStepOutput` construction — that stays in the executor

use crate::tbs_advice::TbsStepAdvice;
use crate::using::UsingBag;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helper — normality test selection
// ─────────────────────────────────────────────────────────────────────────────

/// Choose and run a normality test appropriate for the sample size.
///
/// For `n < n_thresh`: Shapiro-Wilk (exact, powerful on small n).
/// For `n ≥ n_thresh`: D'Agostino-Pearson (O(n log n), scales well).
///
/// Returns `(p_value, test_name)`.  Returns `(NaN, "n<3")` for trivial inputs.
pub(crate) fn normality_test(col: &[f64], n_thresh: usize) -> (f64, &'static str) {
    if col.len() < 3 { return (f64::NAN, "n<3"); }
    if col.len() < n_thresh {
        let r = crate::nonparametric::shapiro_wilk(col);
        (r.p_value, "Shapiro-Wilk")
    } else {
        let r = crate::nonparametric::dagostino_pearson(col);
        (r.p_value, "D'Agostino-Pearson")
    }
}

/// Count distinct non-NaN values; returns early once count exceeds 2.
fn count_unique(col: &[f64]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for &v in col {
        if !v.is_nan() { seen.insert(v.to_bits()); }
        if seen.len() > 2 { return seen.len(); }
    }
    seen.len()
}

// ─────────────────────────────────────────────────────────────────────────────
// autodetect_correlation
// ─────────────────────────────────────────────────────────────────────────────

/// Layer 1 auto-detection for bivariate correlation.
///
/// Decision tree:
/// 1. Both binary (≤2 unique values) → Phi coefficient
/// 2. One binary + one continuous     → Point-biserial
/// 3. Both normal (p > `normality_alpha`) → Pearson r
/// 4. Otherwise                        → Spearman ρ
///
/// If `bag` contains `method`, the auto-recommendation is still computed
/// and recorded in the returned advice, but the caller is expected to apply
/// the override (Layer 2 responsibility — handled in the executor).
///
/// Returns `(auto_value, auto_name, advice)`.
pub(crate) fn autodetect_correlation(
    x: &[f64],
    y: &[f64],
    bag: &UsingBag,
) -> (f64, &'static str, TbsStepAdvice) {
    let normality_alpha = bag.get_f64("normality_alpha").unwrap_or(0.05);
    let n_thresh = bag.get_f64("normality_test_n_threshold")
        .map(|v| v as usize).unwrap_or(5000);

    let x_binary = count_unique(x) <= 2;
    let y_binary = count_unique(y) <= 2;

    if x_binary && y_binary {
        let phi = crate::nonparametric::phi_coefficient(x, y);
        let adv = TbsStepAdvice::accepted(
            "phi_coefficient",
            "both variables binary (≤2 unique values), phi coefficient is exact Pearson on binary data",
        );
        return (phi, "phi_coefficient", adv);
    }

    if x_binary || y_binary {
        let (bin, cont) = if x_binary { (x, y) } else { (y, x) };
        let rpb = crate::nonparametric::point_biserial(bin, cont);
        let adv = TbsStepAdvice::accepted(
            "point_biserial",
            "one variable binary and one continuous, using point-biserial (= Pearson on 0/1 indicator)",
        );
        return (rpb, "point_biserial", adv);
    }

    let (px, tn_x) = normality_test(x, n_thresh);
    let (py, tn_y) = normality_test(y, n_thresh);
    let x_norm = px > normality_alpha;
    let y_norm = py > normality_alpha;

    if x_norm && y_norm {
        let r = crate::nonparametric::pearson_r(x, y);
        let adv = TbsStepAdvice::accepted(
            "pearson",
            format!("both normal ({tn_x} p={px:.3}/{py:.3}), using Pearson r"),
        )
        .with_diagnostic(tn_x, px, "normal")
        .with_diagnostic(tn_y, py, "normal");
        (r, "pearson", adv)
    } else {
        let r = crate::nonparametric::spearman(x, y);
        let reason = if !x_norm && !y_norm {
            format!("both non-normal ({tn_x} p={px:.3}/{py:.3}), using Spearman ρ")
        } else if !x_norm {
            format!("x non-normal ({tn_x} p={px:.3}), using Spearman ρ")
        } else {
            format!("y non-normal ({tn_y} p={py:.3}), using Spearman ρ")
        };
        let adv = TbsStepAdvice::accepted("spearman", reason)
            .with_diagnostic(tn_x, px, if x_norm { "normal" } else { "non-normal" })
            .with_diagnostic(tn_y, py, if y_norm { "normal" } else { "non-normal" });
        (r, "spearman", adv)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// autodetect_t_test_2
// ─────────────────────────────────────────────────────────────────────────────

/// Layer 1 auto-detection for two-sample location tests.
///
/// Decision tree:
/// 1. Both normal (SW/D'A-P p > `normality_alpha`):
///    a. Equal variance (Levene p > `variance_alpha`) → pooled two-sample t
///    b. Unequal variance                              → Welch t
/// 2. Non-normal → Mann-Whitney U
///
/// Returns `(auto_method_name, advice)`.  The caller runs the test.
pub(crate) fn autodetect_t_test_2(
    x: &[f64],
    y: &[f64],
    bag: &UsingBag,
) -> (&'static str, TbsStepAdvice) {
    let normality_alpha = bag.get_f64("normality_alpha").unwrap_or(0.05);
    let variance_alpha  = bag.get_f64("variance_alpha").unwrap_or(0.05);
    let n_thresh = bag.get_f64("normality_test_n_threshold")
        .map(|v| v as usize).unwrap_or(5000);

    let (px, tn_x) = normality_test(x, n_thresh);
    let (py, tn_y) = normality_test(y, n_thresh);
    let x_norm = px > normality_alpha;
    let y_norm = py > normality_alpha;

    let levene_p = if x_norm && y_norm {
        let lev = crate::hypothesis::levene_test(
            &[x, y],
            crate::hypothesis::LeveneCenter::Median,
        );
        lev.p_value
    } else {
        f64::NAN
    };
    let equal_var = levene_p > variance_alpha;

    let (auto_method, auto_reason): (&'static str, String) = if x_norm && y_norm {
        if equal_var {
            ("two_sample_t", format!(
                "both normal ({tn_x} p={px:.3}/{py:.3}), equal variance (Levene p={levene_p:.3}): pooled t-test"
            ))
        } else {
            ("welch_t", format!(
                "both normal ({tn_x} p={px:.3}/{py:.3}), unequal variance (Levene p={levene_p:.3}): Welch t-test"
            ))
        }
    } else {
        ("mann_whitney_u", format!(
            "non-normal data ({tn_x} p={px:.3}/{py:.3}): Mann-Whitney U"
        ))
    };

    let adv = TbsStepAdvice::accepted(auto_method, auto_reason)
        .with_diagnostic(tn_x, px, if x_norm { "normal" } else { "non-normal" })
        .with_diagnostic(tn_y, py, if y_norm { "normal" } else { "non-normal" });

    (auto_method, adv)
}

// ─────────────────────────────────────────────────────────────────────────────
// autodetect_anova
// ─────────────────────────────────────────────────────────────────────────────

/// Layer 1 auto-detection for k-group location tests.
///
/// Decision tree:
/// 1. All groups normal (SW/D'A-P p > `normality_alpha`):
///    a. Equal variance (Levene p > `variance_alpha`) → one-way ANOVA
///    b. Unequal variance                              → Welch's ANOVA
/// 2. Any group non-normal → Kruskal-Wallis H
///
/// Returns `(auto_method_name, advice)`.  The caller runs the test.
pub(crate) fn autodetect_anova(
    group_vecs: &[Vec<f64>],
    bag: &UsingBag,
) -> (&'static str, TbsStepAdvice) {
    let normality_alpha = bag.get_f64("normality_alpha").unwrap_or(0.05);
    let variance_alpha  = bag.get_f64("variance_alpha").unwrap_or(0.05);
    let n_thresh = bag.get_f64("normality_test_n_threshold")
        .map(|v| v as usize).unwrap_or(5000);

    let norm_results: Vec<(f64, &'static str)> = group_vecs.iter()
        .map(|g| normality_test(g, n_thresh))
        .collect();
    let all_normal = norm_results.iter().all(|(p, _)| *p > normality_alpha);

    let group_slices: Vec<&[f64]> = group_vecs.iter().map(|v| v.as_slice()).collect();
    let levene = crate::hypothesis::levene_test(
        &group_slices,
        crate::hypothesis::LeveneCenter::Median,
    );
    let equal_var = levene.p_value > variance_alpha;

    let k = group_vecs.len();
    let (auto_method, auto_reason): (&'static str, String) = if all_normal {
        if equal_var {
            ("one_way_anova", format!(
                "all {k} groups normal, equal variance (Levene p={:.3}): classic ANOVA",
                levene.p_value
            ))
        } else {
            ("welch_anova", format!(
                "all {k} groups normal, unequal variance (Levene p={:.3}): Welch's ANOVA",
                levene.p_value
            ))
        }
    } else {
        let non_normal: Vec<usize> = norm_results.iter().enumerate()
            .filter(|(_, (p, _))| *p <= normality_alpha).map(|(i, _)| i).collect();
        ("kruskal_wallis", format!(
            "groups {non_normal:?} non-normal: Kruskal-Wallis H test"
        ))
    };

    let adv = TbsStepAdvice::accepted(auto_method, auto_reason)
        .with_diagnostic("Levene", levene.p_value,
            if equal_var { "equal variance" } else { "unequal variance" });

    (auto_method, adv)
}

// ─────────────────────────────────────────────────────────────────────────────
// autodetect_pca_components
// ─────────────────────────────────────────────────────────────────────────────

/// Layer 1 auto-detection for the number of PCA components.
///
/// Decision tree:
/// 1. KMO overall < `kmo_threshold` → warn, use Kaiser k anyway
/// 2. Bartlett p ≥ `bartlett_alpha` → warn, use Kaiser k anyway
/// 3. Otherwise → Kaiser criterion (eigenvalue > 1) → k components
///
/// Returns `(auto_k, advice)`.
pub(crate) fn autodetect_pca_components(
    data: &[f64],
    n: usize,
    d: usize,
    bag: &UsingBag,
) -> (usize, TbsStepAdvice) {
    let kmo_threshold  = bag.get_f64("kmo_threshold").unwrap_or(0.5);
    let bartlett_alpha = bag.get_f64("bartlett_alpha").unwrap_or(0.05);

    let corr_mat = crate::factor_analysis::correlation_matrix(data, n, d);
    let kb = crate::factor_analysis::kmo_bartlett(&corr_mat, n);

    let kmo = kb.kmo_overall;
    let bartlett_p = kb.bartlett_p_value;
    let pca_viable = kmo >= kmo_threshold && bartlett_p < bartlett_alpha;

    // Full PCA to obtain eigenvalues (σ² / (n-1))
    let full_pca = crate::dim_reduction::pca(data, n, d, d);
    let n_minus_1 = (n - 1).max(1) as f64;
    let eigenvalues: Vec<f64> = full_pca.singular_values.iter()
        .map(|sv| sv * sv / n_minus_1)
        .collect();

    // Kaiser criterion
    let kaiser_k = eigenvalues.iter().filter(|&&ev| ev > 1.0).count().max(1);
    let auto_k = kaiser_k.min(d).max(1);

    let (auto_method, auto_reason) = if !pca_viable {
        if kmo < kmo_threshold {
            ("pca_warn", format!(
                "KMO={kmo:.3} < {kmo_threshold}: data may not be suitable for PCA (sampling inadequacy)"
            ))
        } else {
            ("pca_warn", format!(
                "Bartlett p={bartlett_p:.3} ≥ {bartlett_alpha}: correlation matrix not significantly different from identity"
            ))
        }
    } else {
        ("pca", format!(
            "KMO={kmo:.3} ≥ {kmo_threshold}, Bartlett p={bartlett_p:.4} < {bartlett_alpha}: PCA viable; Kaiser criterion → {auto_k} components"
        ))
    };

    let adv = TbsStepAdvice::accepted(auto_method, auto_reason)
        .with_diagnostic("KMO", kmo,
            if kmo >= 0.9 { "marvellous" }
            else if kmo >= 0.8 { "meritorious" }
            else if kmo >= 0.7 { "middling" }
            else if kmo >= 0.6 { "mediocre" }
            else if kmo >= kmo_threshold { "miserable" }
            else { "unacceptable" })
        .with_diagnostic("Bartlett", bartlett_p,
            if bartlett_p < bartlett_alpha { "significant (correlations present)" } else { "not significant" });

    (auto_k, adv)
}

// ─────────────────────────────────────────────────────────────────────────────
// autodetect_regression_diagnostics
// ─────────────────────────────────────────────────────────────────────────────

/// Layer 1 regression diagnostics: VIF, residual normality, Breusch-Pagan,
/// Cook's distance.
///
/// Returns `(auto_method_name, advice)`.  The caller already has the fitted
/// model and residuals; it passes them in here.
pub(crate) fn autodetect_regression_diagnostics(
    x_pred: &crate::linear_algebra::Mat,
    x_aug: &crate::linear_algebra::Mat,
    residuals: &[f64],
    resid_norm_p: f64,
    resid_norm_test_name: &'static str,
    bag: &UsingBag,
) -> (&'static str, TbsStepAdvice) {
    let vif_threshold  = bag.get_f64("vif_threshold").unwrap_or(10.0);
    let normality_alpha = bag.get_f64("normality_alpha").unwrap_or(0.05);
    let variance_alpha  = bag.get_f64("variance_alpha").unwrap_or(0.05);

    let vif_vals = crate::multivariate::vif(x_pred);
    let max_vif = vif_vals.iter().cloned().fold(0.0_f64, f64::max);
    let multicollinear = max_vif > vif_threshold;

    let resid_normal = resid_norm_p > normality_alpha;

    let bp = crate::hypothesis::breusch_pagan(x_aug, residuals);
    let heteroscedastic = bp.p_value < variance_alpha;

    let influence_threshold = bag.get_f64("influence_threshold");
    let influence = crate::hypothesis::cooks_distance_with_threshold(
        x_aug, residuals, influence_threshold,
    );
    let n_influential = influence.n_influential;

    let (auto_method, auto_reason): (&'static str, String) = if multicollinear {
        ("ridge", format!(
            "max VIF={max_vif:.1} > {vif_threshold:.0} (multicollinearity detected): consider Ridge regression"
        ))
    } else if !resid_normal {
        ("quantile", format!(
            "residuals non-normal (p={resid_norm_p:.3}): consider quantile regression or robust OLS"
        ))
    } else if heteroscedastic {
        ("wls", format!(
            "heteroscedastic residuals (Breusch-Pagan p={:.3}): consider WLS or HC3 standard errors",
            bp.p_value
        ))
    } else {
        ("ols", format!(
            "normality OK (p={resid_norm_p:.3}), homoscedasticity OK (BP p={:.3}), VIF OK (max={max_vif:.1}): OLS assumptions satisfied",
            bp.p_value
        ))
    };

    let adv = TbsStepAdvice::accepted(auto_method, auto_reason)
        .with_diagnostic("VIF_max", max_vif,
            if multicollinear { "multicollinearity detected" } else { "OK" })
        .with_diagnostic(resid_norm_test_name, resid_norm_p,
            if resid_normal { "normal" } else { "non-normal" })
        .with_diagnostic("Breusch-Pagan", bp.p_value,
            if heteroscedastic { "heteroscedastic" } else { "homoscedastic" })
        .with_diagnostic("Cook's_D_n_influential", n_influential as f64,
            if n_influential > 0 {
                format!("{n_influential} influential observations (Cook's D > 4/n)")
            } else {
                "no influential observations".to_owned()
            });

    (auto_method, adv)
}

// ─────────────────────────────────────────────────────────────────────────────
// autodetect_volatility
// ─────────────────────────────────────────────────────────────────────────────

/// Layer 1 volatility model selection.
///
/// Decision tree:
/// 1. ARCH-LM p < `arch_alpha` → GARCH(1,1)
/// 2. No ARCH effects          → constant volatility (EWMA)
///
/// Returns `(auto_method_name, advice, Option<GarchFit>, Option<ewma_variance>)`.
/// The caller constructs the `TbsStepOutput` from the returned data.
pub(crate) struct VolatilityDecision {
    pub auto_method: &'static str,
    pub advice: TbsStepAdvice,
    /// Set when GARCH(1,1) was fitted: [omega, alpha, beta].
    pub garch_params: Option<[f64; 3]>,
    /// Set when EWMA was used.
    pub ewma_variance: Option<Vec<f64>>,
}

pub(crate) fn autodetect_volatility(col: &[f64], bag: &UsingBag) -> VolatilityDecision {
    let arch_lags_cap   = bag.get_f64("arch_lags_cap").map(|v| v as usize).unwrap_or(5);
    let arch_lags_denom = bag.get_f64("arch_lags_fraction_denom").map(|v| v as usize).unwrap_or(10);
    let arch_alpha      = bag.get_f64("arch_alpha").unwrap_or(0.05);
    let arch_lags = arch_lags_cap.min(col.len() / arch_lags_denom).max(1);

    let arch = crate::volatility::arch_lm_test(col, arch_lags);
    let arch_p = arch.as_ref().map_or(1.0, |a| a.p_value);
    let has_arch = arch_p < arch_alpha;

    if has_arch {
        let garch_max_iter = bag.get_f64("garch_max_iter").map(|v| v as usize).unwrap_or(200);
        let garch = crate::volatility::garch11_fit(col, garch_max_iter);
        let rec = if garch.near_igarch {
            "GARCH(1,1) near-integrated (α+β > 0.99): consider IGARCH or long-memory model"
        } else {
            "GARCH(1,1) fitted successfully"
        };
        let adv = TbsStepAdvice::accepted("garch11", rec.to_string())
            .with_diagnostic("ARCH-LM", arch_p, "ARCH effects present")
            .with_diagnostic("alpha+beta", garch.alpha + garch.beta,
                if garch.near_igarch { "near-integrated" } else { "stationary" });
        VolatilityDecision {
            auto_method: "garch11",
            advice: adv,
            garch_params: Some([garch.omega, garch.alpha, garch.beta]),
            ewma_variance: None,
        }
    } else {
        let adv = TbsStepAdvice::accepted(
            "constant_volatility",
            "no ARCH effects detected; constant volatility (EWMA/rolling std) sufficient",
        )
        .with_diagnostic("ARCH-LM", arch_p, "no ARCH effects");

        let ewma_lambda = bag.get_f64("ewma_lambda").unwrap_or(0.94);
        let ewma = crate::volatility::ewma_variance(col, ewma_lambda);

        VolatilityDecision {
            auto_method: "constant_volatility",
            advice: adv,
            garch_params: None,
            ewma_variance: Some(ewma),
        }
    }
}
