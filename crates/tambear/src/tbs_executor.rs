//! Interpreter for parsed `.tbs` chains — dispatches `TbsStep` to tambear functions.
//!
//! ## Coverage
//!
//! Every tambear module that operates on columnar (n×d) data is callable from
//! `.tbs`. Functions that require non-matrix inputs (graphs, interpolation
//! knot arrays, optimization objectives) are not directly chainable but can
//! be accessed through the Rust API.
//!
//! ## Column convention
//!
//! Steps that operate on a single column accept `col=<index>` (default 0).
//! Steps that need two columns accept `col_x` / `col_y` (defaults 0 / 1).
//!
//! ## Science linting
//!
//! The executor collects `TbsLint` warnings during chain execution:
//! - **naive_variance**: suggest `normalize()` before variance-sensitive ops
//! - **small_n**: warn when n < 20 for parametric tests
//! - **multiple_tests**: suggest correction when multiple hypothesis tests run
//! - **normality_unchecked**: suggest KS test before parametric tests

use crate::pipeline::TamPipeline;
use crate::tbs_lint::{self, LintSeverity, TbsLint};
use crate::tbs_parser::{TbsChain, TbsStep};
use crate::train::linear::LinearModel;
use crate::train::logistic::LogisticModel;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Output from a single `.tbs` step.
#[derive(Debug, Clone)]
pub enum TbsStepOutput {
    /// No output (transformation step like `normalize`).
    Transform,
    /// Single scalar value.
    Scalar { name: &'static str, value: f64 },
    /// Vector of values (ACF lags, quantiles, etc.).
    Vector { name: &'static str, values: Vec<f64> },
    /// Matrix result (correlation matrix, etc.) — row-major.
    Matrix { name: &'static str, data: Vec<f64>, rows: usize, cols: usize },
    /// Descriptive statistics summary.
    Descriptive(Vec<crate::descriptive::DescriptiveResult>),
    /// Hypothesis test result.
    Test(crate::hypothesis::TestResult),
    /// ANOVA result.
    Anova(crate::hypothesis::AnovaResult),
    /// Chi-square result.
    ChiSquare(crate::hypothesis::ChiSquareResult),
    /// Nonparametric test result.
    Nonparametric(crate::nonparametric::NonparametricResult),
    /// Bootstrap result.
    Bootstrap(crate::nonparametric::BootstrapResult),
    /// PCA result (components, variance ratios, transformed data).
    Pca(crate::dim_reduction::PcaResult),
    /// t-SNE embedding.
    Tsne(crate::dim_reduction::TsneResult),
    /// AR model fit.
    Ar(crate::time_series::ArResult),
    /// ADF stationarity test.
    Adf(crate::time_series::AdfResult),
    /// Factor analysis result.
    FactorAnalysis(crate::factor_analysis::FaResult),
}

impl TbsStepOutput {
    /// Is this a transform (no explicit output)?
    pub fn is_transform(&self) -> bool {
        matches!(self, TbsStepOutput::Transform)
    }
}

// ---------------------------------------------------------------------------
// Advice types — "tambear recommends X because Y, user forced Z"
// ---------------------------------------------------------------------------

/// A diagnostic data point computed during auto-detection.
#[derive(Debug, Clone)]
pub struct TbsDiagnostic {
    /// Name of the diagnostic check (e.g. "normality", "equal_variance").
    pub test_name: &'static str,
    /// Numeric result (e.g. p-value, statistic, count).
    pub result: f64,
    /// Human-readable conclusion (e.g. "p=0.23, normality not rejected").
    pub conclusion: String,
}

/// What tambear recommends and why.
#[derive(Debug, Clone)]
pub struct TbsRecommendation {
    /// Recommended method name (e.g. "welch_t", "mann_whitney").
    pub method: &'static str,
    /// Reason for the recommendation.
    pub reason: String,
}

/// What the user forced instead of the recommendation.
#[derive(Debug, Clone)]
pub struct TbsOverride {
    /// The method the user explicitly requested.
    pub method: String,
    /// The parameter/key that triggered this override (e.g. "using(method=…)").
    pub key: String,
    /// Warning about the override (e.g. "normality assumption may be violated").
    pub warning: Option<String>,
}

/// Per-step advice: recommendation + optional user override + diagnostics.
/// Populated by steps that have auto-detection logic.
#[derive(Debug, Clone)]
pub struct TbsStepAdvice {
    /// What tambear would have recommended.
    pub recommended: TbsRecommendation,
    /// What the user forced (None if they accepted the recommendation).
    pub user_override: Option<TbsOverride>,
    /// Supporting diagnostic checks.
    pub diagnostics: Vec<TbsDiagnostic>,
}

impl TbsStepAdvice {
    /// Build advice for a recommendation the user accepted (no override).
    pub fn accepted(method: &'static str, reason: impl Into<String>) -> Self {
        TbsStepAdvice {
            recommended: TbsRecommendation { method, reason: reason.into() },
            user_override: None,
            diagnostics: Vec::new(),
        }
    }

    /// Build advice for a recommendation the user overrode.
    pub fn overridden(
        recommended: &'static str,
        reason: impl Into<String>,
        forced: impl Into<String>,
        key: impl Into<String>,
        warning: Option<impl Into<String>>,
    ) -> Self {
        TbsStepAdvice {
            recommended: TbsRecommendation { method: recommended, reason: reason.into() },
            user_override: Some(TbsOverride {
                method: forced.into(),
                key: key.into(),
                warning: warning.map(|w| w.into()),
            }),
            diagnostics: Vec::new(),
        }
    }

    /// Attach a diagnostic to this advice.
    pub fn with_diagnostic(mut self, test_name: &'static str, result: f64, conclusion: impl Into<String>) -> Self {
        self.diagnostics.push(TbsDiagnostic { test_name, result, conclusion: conclusion.into() });
        self
    }
}

/// The output of executing a `.tbs` chain.
pub struct TbsResult {
    /// The pipeline after all steps have been applied.
    pub pipeline: TamPipeline,

    /// Set if the chain included a `train.linear(...)` step.
    pub linear_model: Option<LinearModel>,

    /// Set if the chain included a `train.logistic(...)` step.
    pub logistic_model: Option<LogisticModel>,

    /// Per-step outputs (one entry per step in the chain).
    pub outputs: Vec<TbsStepOutput>,

    /// Science lint warnings collected during execution.
    pub lints: Vec<TbsLint>,

    /// Per-step superpositions (parallel to `outputs`).
    /// `None` for steps that don't have sweepable parameters.
    /// When present, `outputs[i]` is the collapsed result and
    /// `superpositions[i]` contains the full sweep.
    pub superpositions: Vec<Option<crate::superposition::Superposition>>,

    /// Per-step advice (parallel to `outputs`).
    /// `None` for steps without auto-detection logic (transforms, basic stats).
    /// When present, shows what tambear recommended and whether the user overrode it.
    pub advice: Vec<Option<TbsStepAdvice>>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a single column from row-major n×d data.
fn extract_col(data: &[f64], n: usize, d: usize, col: usize) -> Vec<f64> {
    assert!(col < d, "column index {col} out of range (d={d})");
    (0..n).map(|i| data[i * d + col]).collect()
}

/// Extract two columns.
fn extract_two_cols(data: &[f64], n: usize, d: usize, cx: usize, cy: usize) -> (Vec<f64>, Vec<f64>) {
    (extract_col(data, n, d, cx), extract_col(data, n, d, cy))
}

/// Read `col` arg from step (named or positional[idx]), defaulting to `default`.
fn col_arg(step: &TbsStep, default: usize) -> usize {
    step.get_arg("col", 0).and_then(|v| v.as_usize()).unwrap_or(default)
}

/// Read a named-or-positional f64 arg with a default.
fn f64_arg(step: &TbsStep, name: &str, pos: usize, default: f64) -> f64 {
    step.get_arg(name, pos).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Read a named-or-positional usize arg with a default.
fn usize_arg(step: &TbsStep, name: &str, pos: usize, default: usize) -> usize {
    step.get_arg(name, pos).and_then(|v| v.as_usize()).unwrap_or(default)
}

/// Required f64 arg.
fn f64_req(step: &TbsStep, name: &str, pos: usize) -> Result<f64, Box<dyn std::error::Error>> {
    step.get_arg(name, pos)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("{}: {name} required (named or positional[{pos}])", step.name).into())
}

/// Required usize arg.
fn usize_req(step: &TbsStep, name: &str, pos: usize) -> Result<usize, Box<dyn std::error::Error>> {
    step.get_arg(name, pos)
        .and_then(|v| v.as_usize())
        .ok_or_else(|| format!("{}: {name} required (named or positional[{pos}])", step.name).into())
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Execute a parsed `.tbs` chain on `data` (n × d, row-major).
///
/// `y` is the target vector for supervised steps (`train.linear`, etc.).
/// If a supervised step is encountered without `y`, an error is returned.
pub fn execute(
    chain: TbsChain,
    data: Vec<f64>,
    n: usize,
    d: usize,
    y: Option<Vec<f64>>,
) -> Result<TbsResult, Box<dyn std::error::Error>> {
    // Static lints (parse-time, chain structure only)
    let mut lints = tbs_lint::static_lints(&chain);

    // Dynamic lints on input data (before pipeline consumes the Vec)
    lints.extend(tbs_lint::lint_l106_constant_columns(&data, n, d));
    lints.extend(tbs_lint::lint_l101_naive_variance(&data, n, d, &chain));

    let mut pipeline = TamPipeline::from_slice(data, n, d);
    let mut linear_model: Option<LinearModel> = None;
    let mut logistic_model: Option<LogisticModel> = None;
    let mut outputs: Vec<TbsStepOutput> = Vec::with_capacity(chain.steps.len());
    let mut advice: Vec<Option<TbsStepAdvice>> = Vec::with_capacity(chain.steps.len());
    let mut using_bag = crate::using::UsingBag::new();

    // Track state for science linting
    let mut normalized = false;
    let mut normality_checked = false;
    let mut n_hypothesis_tests = 0u32;

    for (step_idx, step) in chain.steps.iter().enumerate() {
        let fr = pipeline.frame();
        let (pn, pd) = (fr.n, fr.d);

        let mut step_advice: Option<TbsStepAdvice> = None;
        let output = match step.name.as_str() {
            // ══════════════════════════════════════════════════════════════
            // Preprocessing
            // ══════════════════════════════════════════════════════════════

            ("normalize", None) => {
                pipeline = pipeline.normalize();
                normalized = true;
                TbsStepOutput::Transform
            }

            // ══════════════════════════════════════════════════════════════
            // Descriptive statistics
            // ══════════════════════════════════════════════════════════════

            ("describe", None) => {
                let engine = crate::descriptive::DescriptiveEngine::new();
                let mut results = Vec::with_capacity(pd);
                for j in 0..pd {
                    let col = extract_col(&pipeline.frame().data, pn, pd, j);
                    results.push(engine.describe(&col));
                }
                TbsStepOutput::Descriptive(results)
            }

            ("mean", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let v = crate::descriptive::moments_ungrouped(&col).mean();
                TbsStepOutput::Scalar { name: "mean", value: v }
            }

            ("std", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let stats = crate::descriptive::moments_ungrouped(&col);
                TbsStepOutput::Scalar { name: "std", value: stats.std(1) }
            }

            ("variance", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let stats = crate::descriptive::moments_ungrouped(&col);
                if !normalized {
                    lints.push(TbsLint {
                        code: "L101",
                        step_index: Some(step_idx),
                        message: "Computing variance on unnormalized data. Consider normalize() first for numerical stability.".into(),
                        severity: LintSeverity::Info,
                    });
                }
                TbsStepOutput::Scalar { name: "variance", value: stats.variance(1) }
            }

            ("skewness", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let stats = crate::descriptive::moments_ungrouped(&col);
                TbsStepOutput::Scalar { name: "skewness", value: stats.skewness(false) }
            }

            ("kurtosis", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let stats = crate::descriptive::moments_ungrouped(&col);
                TbsStepOutput::Scalar { name: "kurtosis", value: stats.kurtosis(true, false) }
            }

            ("median", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sorted = crate::descriptive::sorted_nan_free(&col);
                TbsStepOutput::Scalar { name: "median", value: crate::descriptive::median(&sorted) }
            }

            ("quantile", None) => {
                let q = f64_req(step, "q", 0)?;
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sorted = crate::descriptive::sorted_nan_free(&col);
                let v = crate::descriptive::quantile(&sorted, q, crate::descriptive::QuantileMethod::Linear);
                TbsStepOutput::Scalar { name: "quantile", value: v }
            }

            ("quartiles", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sorted = crate::descriptive::sorted_nan_free(&col);
                let (q1, q2, q3) = crate::descriptive::quartiles(&sorted);
                TbsStepOutput::Vector { name: "quartiles", values: vec![q1, q2, q3] }
            }

            ("iqr", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sorted = crate::descriptive::sorted_nan_free(&col);
                TbsStepOutput::Scalar { name: "iqr", value: crate::descriptive::iqr(&sorted) }
            }

            ("mad", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sorted = crate::descriptive::sorted_nan_free(&col);
                TbsStepOutput::Scalar { name: "mad", value: crate::descriptive::mad(&sorted) }
            }

            ("gini", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sorted = crate::descriptive::sorted_nan_free(&col);
                TbsStepOutput::Scalar { name: "gini", value: crate::descriptive::gini(&sorted) }
            }

            ("geometric_mean", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "geometric_mean", value: crate::descriptive::geometric_mean(&col) }
            }

            ("harmonic_mean", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "harmonic_mean", value: crate::descriptive::harmonic_mean(&col) }
            }

            ("trimmed_mean", None) => {
                let frac = f64_arg(step, "fraction", 0, 0.1);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sorted = crate::descriptive::sorted_nan_free(&col);
                TbsStepOutput::Scalar { name: "trimmed_mean", value: crate::descriptive::trimmed_mean(&sorted, frac) }
            }

            ("winsorized_mean", None) => {
                let frac = f64_arg(step, "fraction", 0, 0.1);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sorted = crate::descriptive::sorted_nan_free(&col);
                TbsStepOutput::Scalar { name: "winsorized_mean", value: crate::descriptive::winsorized_mean(&sorted, frac) }
            }

            ("correlation_matrix", None) => {
                let mat = crate::factor_analysis::correlation_matrix(&pipeline.frame().data, pn, pd);
                TbsStepOutput::Matrix { name: "correlation_matrix", data: mat.data.clone(), rows: pd, cols: pd }
            }

            ("correlation", None) => {
                // Auto-detect the appropriate pairwise correlation method.
                // Decision tree (all branches produce &'static str name):
                //   using(method=X) → honour override and record recommendation
                //   Both binary (≤2 unique values): Phi coefficient
                //   One binary + one continuous: Point-biserial (= Pearson on binary+continuous)
                //   Both normal (SW or D'A-P, p > 0.05): Pearson r
                //   Otherwise: Spearman ρ
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);

                // ≤2 distinct non-NaN values → treat as binary
                let count_unique = |col: &[f64]| -> usize {
                    let mut seen = std::collections::HashSet::new();
                    for &v in col {
                        if !v.is_nan() { seen.insert(v.to_bits()); }
                        if seen.len() > 2 { return seen.len(); }
                    }
                    seen.len()
                };
                let x_binary = count_unique(&x) <= 2;
                let y_binary = count_unique(&yv) <= 2;

                // Normality test: SW for n < 5000, D'Agostino-Pearson for n ≥ 5000
                let normality = |col: &[f64]| -> (f64, &'static str) {
                    if col.len() < 5000 {
                        let r = crate::nonparametric::shapiro_wilk(col);
                        (r.p_value, "Shapiro-Wilk")
                    } else {
                        let r = crate::nonparametric::dagostino_pearson(col);
                        (r.p_value, "D'Agostino-Pearson")
                    }
                };

                // Check for user override first
                let user_method: Option<String> = using_bag.method().map(|s| s.to_owned());

                // Compute the auto-recommended method + value + advice
                let (auto_name, auto_val, auto_adv): (&'static str, f64, TbsStepAdvice) =
                    if x_binary && y_binary {
                        let phi = crate::nonparametric::phi_coefficient(&x, &yv);
                        (
                            "phi_coefficient", phi,
                            TbsStepAdvice::accepted("phi_coefficient",
                                "both variables binary (≤2 unique values), phi coefficient is exact Pearson on binary data"),
                        )
                    } else if x_binary || y_binary {
                        let (bin, cont) = if x_binary { (&x, &yv) } else { (&yv, &x) };
                        let rpb = crate::nonparametric::point_biserial(bin, cont);
                        (
                            "point_biserial", rpb,
                            TbsStepAdvice::accepted("point_biserial",
                                "one variable binary and one continuous, using point-biserial (= Pearson on 0/1 indicator)"),
                        )
                    } else {
                        let (px, tn_x) = normality(&x);
                        let (py, tn_y) = normality(&yv);
                        let x_norm = px > 0.05;
                        let y_norm = py > 0.05;
                        if x_norm && y_norm {
                            let r = crate::nonparametric::pearson_r(&x, &yv);
                            let adv = TbsStepAdvice::accepted("pearson",
                                format!("both normal ({} p={:.3}/{:.3}), using Pearson r", tn_x, px, py))
                                .with_diagnostic(tn_x, px, "normal")
                                .with_diagnostic(tn_y, py, "normal");
                            ("pearson", r, adv)
                        } else {
                            let r = crate::nonparametric::spearman(&x, &yv);
                            let reason = if !x_norm && !y_norm {
                                format!("both non-normal ({} p={:.3}/{:.3}), using Spearman ρ", tn_x, px, py)
                            } else if !x_norm {
                                format!("x non-normal ({} p={:.3}), using Spearman ρ", tn_x, px)
                            } else {
                                format!("y non-normal ({} p={:.3}), using Spearman ρ", tn_y, py)
                            };
                            let adv = TbsStepAdvice::accepted("spearman", reason)
                                .with_diagnostic(tn_x, px, if x_norm { "normal" } else { "non-normal" })
                                .with_diagnostic(tn_y, py, if y_norm { "normal" } else { "non-normal" });
                            ("spearman", r, adv)
                        }
                    };

                // If user forced a method, run that instead but record the override
                let (final_name, final_val, final_adv): (&'static str, f64, TbsStepAdvice) =
                    if let Some(ref forced) = user_method {
                        let forced_val = match forced.as_str() {
                            "spearman" => crate::nonparametric::spearman(&x, &yv),
                            "kendall" | "kendall_tau" => crate::nonparametric::kendall_tau(&x, &yv),
                            "phi" | "phi_coefficient" => crate::nonparametric::phi_coefficient(&x, &yv),
                            "point_biserial" => crate::nonparametric::point_biserial(&x, &yv),
                            _ => crate::nonparametric::pearson_r(&x, &yv),
                        };
                        let forced_name: &'static str = match forced.as_str() {
                            "spearman" => "spearman",
                            "kendall" | "kendall_tau" => "kendall_tau",
                            "phi" | "phi_coefficient" => "phi_coefficient",
                            "point_biserial" => "point_biserial",
                            _ => "pearson",
                        };
                        let warn = if forced_name != auto_name {
                            Some(format!(
                                "user forced {forced_name} but tambear recommends {auto_name}: {}",
                                auto_adv.recommended.reason
                            ))
                        } else { None };
                        let adv = TbsStepAdvice::overridden(
                            auto_name,
                            auto_adv.recommended.reason.clone(),
                            forced.clone(),
                            "method",
                            warn,
                        );
                        (forced_name, forced_val, adv)
                    } else {
                        (auto_name, auto_val, auto_adv)
                    };

                step_advice = Some(final_adv);
                TbsStepOutput::Scalar { name: final_name, value: final_val }
            }

            // ══════════════════════════════════════════════════════════════
            // Hypothesis testing
            // ══════════════════════════════════════════════════════════════

            ("test", Some("t")) | ("t_test", None) => {
                let mu = f64_arg(step, "mu", 0, 0.0);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let stats = crate::descriptive::moments_ungrouped(&col);
                if pn < 20 {
                    lints.push(TbsLint {
                        code: "L107",
                        step_index: Some(step_idx),
                        message: format!("t-test with n={pn} (<20): low statistical power."),
                        severity: LintSeverity::Warning,
                    });
                }
                if !normality_checked {
                    lints.push(TbsLint {
                        code: "L004",
                        step_index: Some(step_idx),
                        message: "Parametric test without normality check. Consider ks_test() first.".into(),
                        severity: LintSeverity::Info,
                    });
                }
                n_hypothesis_tests += 1;
                TbsStepOutput::Test(crate::hypothesis::one_sample_t(&stats, mu))
            }

            ("test", Some("t2")) | ("t_test_2", None) => {
                // Auto-detect the appropriate two-sample test.
                // Decision tree:
                //   using(method=X) → honour override
                //   Both normal (SW/D'A-P p > 0.05):
                //     Equal variance (Levene p > 0.05) → pooled two-sample t
                //     Unequal variance → Welch t
                //   Non-normal → Mann-Whitney U
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let sx = crate::descriptive::moments_ungrouped(&x);
                let sy = crate::descriptive::moments_ungrouped(&yv);

                // Normality
                let normality = |col: &[f64]| -> (f64, &'static str) {
                    if col.len() < 5000 {
                        let r = crate::nonparametric::shapiro_wilk(col);
                        (r.p_value, "Shapiro-Wilk")
                    } else {
                        let r = crate::nonparametric::dagostino_pearson(col);
                        (r.p_value, "D'Agostino-Pearson")
                    }
                };
                let (px, tn_x) = normality(&x);
                let (py, tn_y) = normality(&yv);
                let x_norm = px > 0.05;
                let y_norm = py > 0.05;

                // Variance equality via Brown-Forsythe (Levene with median)
                let levene_p = if x_norm && y_norm {
                    let lev = crate::hypothesis::levene_test(
                        &[x.as_slice(), yv.as_slice()],
                        crate::hypothesis::LeveneCenter::Median,
                    );
                    lev.p_value
                } else { f64::NAN };
                let equal_var = levene_p > 0.05;

                let user_method: Option<String> = using_bag.method().map(|s| s.to_owned());

                // Build auto recommendation
                let (auto_method, auto_reason): (&'static str, String) =
                    if x_norm && y_norm {
                        if equal_var {
                            ("two_sample_t", format!("both normal ({} p={:.3}/{:.3}), equal variance (Levene p={:.3}): pooled t-test", tn_x, px, py, levene_p))
                        } else {
                            ("welch_t", format!("both normal ({} p={:.3}/{:.3}), unequal variance (Levene p={:.3}): Welch t-test", tn_x, px, py, levene_p))
                        }
                    } else {
                        ("mann_whitney_u", format!("non-normal data ({} p={:.3}/{:.3}): Mann-Whitney U", tn_x, px, py))
                    };

                let run_test = |method: &str| -> crate::hypothesis::TestResult {
                    match method {
                        "welch_t" | "welch" => crate::hypothesis::welch_t(&sx, &sy),
                        "mann_whitney" | "mann_whitney_u" => {
                            let mw = crate::nonparametric::mann_whitney_u(&x, &yv);
                            crate::hypothesis::TestResult {
                                test_name: "Mann-Whitney U",
                                statistic: mw.statistic,
                                p_value: mw.p_value,
                                df: f64::NAN,
                                effect_size: f64::NAN,
                                effect_size_name: "",
                            }
                        }
                        _ => crate::hypothesis::two_sample_t(&sx, &sy),
                    }
                };

                let (final_method, test_result, adv) =
                    if let Some(ref forced) = user_method {
                        let forced_method = forced.as_str();
                        let warn = if forced_method != auto_method {
                            Some(format!("user forced {forced_method} but tambear recommends {auto_method}: {auto_reason}"))
                        } else { None };
                        let adv = TbsStepAdvice::overridden(
                            auto_method, auto_reason.clone(), forced.clone(), "method", warn)
                            .with_diagnostic(tn_x, px, if x_norm { "normal" } else { "non-normal" })
                            .with_diagnostic(tn_y, py, if y_norm { "normal" } else { "non-normal" });
                        (forced_method.to_owned(), run_test(forced_method), adv)
                    } else {
                        let adv = TbsStepAdvice::accepted(auto_method, auto_reason.clone())
                            .with_diagnostic(tn_x, px, if x_norm { "normal" } else { "non-normal" })
                            .with_diagnostic(tn_y, py, if y_norm { "normal" } else { "non-normal" });
                        (auto_method.to_owned(), run_test(auto_method), adv)
                    };
                let _ = final_method; // method name encoded in TestResult.test_name
                step_advice = Some(adv);
                n_hypothesis_tests += 1;
                TbsStepOutput::Test(test_result)
            }

            ("test", Some("welch")) | ("welch_t", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let sx = crate::descriptive::moments_ungrouped(&x);
                let sy = crate::descriptive::moments_ungrouped(&yv);
                n_hypothesis_tests += 1;
                TbsStepOutput::Test(crate::hypothesis::welch_t(&sx, &sy))
            }

            ("test", Some("paired_t")) | ("paired_t", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let diffs: Vec<f64> = x.iter().zip(yv.iter()).map(|(a, b)| a - b).collect();
                let ds = crate::descriptive::moments_ungrouped(&diffs);
                n_hypothesis_tests += 1;
                TbsStepOutput::Test(crate::hypothesis::paired_t(&ds))
            }

            ("test", Some("chi2")) | ("chi2", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (obs, exp) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                n_hypothesis_tests += 1;
                TbsStepOutput::ChiSquare(crate::hypothesis::chi2_goodness_of_fit(&obs, &exp))
            }

            ("test", Some("proportion")) | ("proportion_z", None) => {
                let successes = f64_req(step, "successes", 0)?;
                let total = f64_req(step, "n", 1)?;
                let p0 = f64_arg(step, "p0", 2, 0.5);
                n_hypothesis_tests += 1;
                TbsStepOutput::Test(crate::hypothesis::one_proportion_z(successes, total, p0))
            }

            // Effect sizes
            ("cohens_d", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let sx = crate::descriptive::moments_ungrouped(&x);
                let sy = crate::descriptive::moments_ungrouped(&yv);
                TbsStepOutput::Scalar { name: "cohens_d", value: crate::hypothesis::cohens_d(&sx, &sy) }
            }

            ("hedges_g", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let sx = crate::descriptive::moments_ungrouped(&x);
                let sy = crate::descriptive::moments_ungrouped(&yv);
                TbsStepOutput::Scalar { name: "hedges_g", value: crate::hypothesis::hedges_g(&sx, &sy) }
            }

            // Multiple comparison corrections
            ("bonferroni", None) => {
                let c = col_arg(step, 0);
                let pvals = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Vector { name: "corrected_p", values: crate::hypothesis::bonferroni(&pvals) }
            }

            ("holm", None) => {
                let c = col_arg(step, 0);
                let pvals = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Vector { name: "corrected_p", values: crate::hypothesis::holm(&pvals) }
            }

            ("benjamini_hochberg", None) => {
                let c = col_arg(step, 0);
                let pvals = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Vector { name: "corrected_p", values: crate::hypothesis::benjamini_hochberg(&pvals) }
            }

            // ══════════════════════════════════════════════════════════════
            // Nonparametric tests
            // ══════════════════════════════════════════════════════════════

            ("spearman", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "spearman", value: crate::nonparametric::spearman(&x, &yv) }
            }

            ("kendall", None) | ("kendall_tau", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "kendall_tau", value: crate::nonparametric::kendall_tau(&x, &yv) }
            }

            ("mann_whitney", None) | ("mann_whitney_u", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Nonparametric(crate::nonparametric::mann_whitney_u(&x, &yv))
            }

            ("wilcoxon", None) | ("wilcoxon_signed_rank", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Nonparametric(crate::nonparametric::wilcoxon_signed_rank(&col))
            }

            ("ks_test", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                normality_checked = true;
                TbsStepOutput::Nonparametric(crate::nonparametric::ks_test_normal(&col))
            }

            ("ks_test_2", None) | ("ks_test_two_sample", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Nonparametric(crate::nonparametric::ks_test_two_sample(&x, &yv))
            }

            ("sign_test", None) => {
                let median0 = f64_arg(step, "median", 0, 0.0);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Nonparametric(crate::nonparametric::sign_test(&col, median0))
            }

            ("runs_test", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Nonparametric(crate::nonparametric::runs_test_numeric(&col))
            }

            ("bootstrap", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let n_boot = usize_arg(step, "n_boot", 1, 1000);
                let ci = f64_arg(step, "ci", 2, 0.95);
                let seed = step.get_arg("seed", 3).and_then(|v| v.as_usize()).unwrap_or(42) as u64;
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                fn stat_mean(x: &[f64]) -> f64 { x.iter().sum::<f64>() / x.len() as f64 }
                TbsStepOutput::Bootstrap(crate::nonparametric::bootstrap_percentile(&col, stat_mean, n_boot, ci, seed))
            }

            ("permutation_test", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let n_perms = usize_arg(step, "n_perms", 2, 5000);
                let seed = step.get_arg("seed", 3).and_then(|v| v.as_usize()).unwrap_or(42) as u64;
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Nonparametric(crate::nonparametric::permutation_test_mean_diff(&x, &yv, n_perms, seed))
            }

            ("kde", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let n_grid = usize_arg(step, "n_grid", 1, 256);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let (grid, density) = crate::nonparametric::kde_fft(&col, n_grid, None);
                let mut flat = Vec::with_capacity(n_grid * 2);
                for i in 0..grid.len() {
                    flat.push(grid[i]);
                    flat.push(density[i]);
                }
                TbsStepOutput::Matrix { name: "kde", data: flat, rows: grid.len(), cols: 2 }
            }

            // ══════════════════════════════════════════════════════════════
            // Clustering
            // ══════════════════════════════════════════════════════════════

            ("discover_clusters", None) | ("dbscan", None) => {
                let epsilon = f64_req(step, "epsilon", 0)?;
                let min_samples = usize_req(step, "min_samples", 1)?;
                pipeline = pipeline.discover_clusters(epsilon, min_samples);
                TbsStepOutput::Transform
            }

            ("kmeans", None) => {
                let k = usize_req(step, "k", 0)?;
                let max_iter = usize_arg(step, "max_iter", 1, 300);
                pipeline = pipeline.kmeans(k, max_iter);
                TbsStepOutput::Transform
            }

            // ══════════════════════════════════════════════════════════════
            // Neighbors
            // ══════════════════════════════════════════════════════════════

            ("knn", None) => {
                let k = usize_req(step, "k", 0)?;
                pipeline = pipeline.knn(k);
                TbsStepOutput::Transform
            }

            // ══════════════════════════════════════════════════════════════
            // Training (supervised)
            // ══════════════════════════════════════════════════════════════

            ("train", Some("linear")) => {
                let y_ref = y.as_deref()
                    .ok_or("train.linear: target data (y) must be provided to executor")?;
                let (p, m) = pipeline.train_linear(y_ref)?;
                pipeline = p;
                linear_model = Some(m);
                TbsStepOutput::Transform
            }

            ("train", Some("logistic")) => {
                let y_ref = y.as_deref()
                    .ok_or("train.logistic: target data (y) must be provided to executor")?;
                let lr = f64_arg(step, "lr", 0, 1.0);
                let max_iter = usize_arg(step, "max_iter", 1, 500);
                let tol = f64_arg(step, "tol", 2, 1e-8);
                let (p, m) = pipeline.train_logistic(y_ref, lr, max_iter, tol)?;
                pipeline = p;
                logistic_model = Some(m);
                TbsStepOutput::Transform
            }

            // ══════════════════════════════════════════════════════════════
            // Dimensionality reduction
            // ══════════════════════════════════════════════════════════════

            ("pca", None) => {
                let n_components = usize_arg(step, "n_components", 0, 2);
                let result = crate::dim_reduction::pca(&pipeline.frame().data, pn, pd, n_components);
                TbsStepOutput::Pca(result)
            }

            ("tsne", None) => {
                let perplexity = f64_arg(step, "perplexity", 0, 30.0);
                let max_iter = usize_arg(step, "max_iter", 1, 1000);
                let lr = f64_arg(step, "lr", 2, 200.0);
                let result = crate::dim_reduction::tsne(&pipeline.frame().data, pn, pd, perplexity, max_iter, lr);
                TbsStepOutput::Tsne(result)
            }

            ("mds", None) => {
                let n2 = pn;
                let mut dist = vec![0.0; n2 * n2];
                let data = &pipeline.frame().data;
                for i in 0..n2 {
                    for j in (i + 1)..n2 {
                        let mut d2 = 0.0;
                        for k in 0..pd {
                            let diff = data[i * pd + k] - data[j * pd + k];
                            d2 += diff * diff;
                        }
                        dist[i * n2 + j] = d2.sqrt();
                        dist[j * n2 + i] = d2.sqrt();
                    }
                }
                let nc = usize_arg(step, "n_components", 0, 2);
                let result = crate::dim_reduction::classical_mds(&dist, n2, nc);
                TbsStepOutput::Matrix {
                    name: "mds",
                    data: result.embedding.data.clone(),
                    rows: result.embedding.rows,
                    cols: result.embedding.cols,
                }
            }

            ("nmf", None) => {
                let k = usize_req(step, "k", 0)?;
                let max_iter = usize_arg(step, "max_iter", 1, 200);
                let result = crate::dim_reduction::nmf(&pipeline.frame().data, pn, pd, k, max_iter);
                TbsStepOutput::Matrix {
                    name: "nmf_w",
                    data: result.w.data.clone(),
                    rows: result.w.rows,
                    cols: result.w.cols,
                }
            }

            // ══════════════════════════════════════════════════════════════
            // Factor analysis
            // ══════════════════════════════════════════════════════════════

            ("paf", None) | ("principal_axis_factoring", None) => {
                let n_factors = usize_arg(step, "n_factors", 0, 2);
                let max_iter = usize_arg(step, "max_iter", 1, 100);
                let corr = crate::factor_analysis::correlation_matrix(&pipeline.frame().data, pn, pd);
                let result = crate::factor_analysis::principal_axis_factoring(&corr, n_factors, max_iter);
                TbsStepOutput::FactorAnalysis(result)
            }

            ("cronbachs_alpha", None) => {
                let v = crate::factor_analysis::cronbachs_alpha(&pipeline.frame().data, pn, pd);
                TbsStepOutput::Scalar { name: "cronbachs_alpha", value: v }
            }

            ("mcdonalds_omega", None) => {
                let corr = crate::factor_analysis::correlation_matrix(&pipeline.frame().data, pn, pd);
                let fa = crate::factor_analysis::principal_axis_factoring(&corr, 1, 100);
                let result = crate::factor_analysis::mcdonalds_omega(&fa.loadings);
                TbsStepOutput::Scalar { name: "mcdonalds_omega", value: result.omega }
            }

            ("scree_elbow", None) => {
                let corr = crate::factor_analysis::correlation_matrix(&pipeline.frame().data, pn, pd);
                let (eigenvalues, _) = crate::linear_algebra::sym_eigen(&corr);
                let elbow = crate::factor_analysis::scree_elbow(&eigenvalues);
                TbsStepOutput::Scalar { name: "scree_elbow", value: elbow as f64 }
            }

            // ══════════════════════════════════════════════════════════════
            // Time series
            // ══════════════════════════════════════════════════════════════

            ("acf", None) => {
                let max_lag = usize_arg(step, "max_lag", 0, 20);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::time_series::acf(&col, max_lag);
                TbsStepOutput::Vector { name: "acf", values }
            }

            ("pacf", None) => {
                let max_lag = usize_arg(step, "max_lag", 0, 20);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::time_series::pacf(&col, max_lag);
                TbsStepOutput::Vector { name: "pacf", values }
            }

            ("ar", None) | ("ar_fit", None) => {
                let p = usize_arg(step, "p", 0, 3);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let result = crate::time_series::ar_fit(&col, p);
                TbsStepOutput::Ar(result)
            }

            ("difference", None) => {
                let d_order = usize_arg(step, "d", 0, 1);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::time_series::difference(&col, d_order);
                TbsStepOutput::Vector { name: "difference", values }
            }

            ("ses", None) | ("exponential_smoothing", None) => {
                let alpha = f64_arg(step, "alpha", 0, 0.3);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let result = crate::time_series::simple_exponential_smoothing(&col, alpha);
                TbsStepOutput::Vector { name: "ses_fitted", values: result.fitted }
            }

            ("holt", None) | ("holt_linear", None) => {
                let alpha = f64_arg(step, "alpha", 0, 0.3);
                let beta = f64_arg(step, "beta", 1, 0.1);
                let horizon = usize_arg(step, "horizon", 2, 10);
                let c = usize_arg(step, "col", 3, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::time_series::holt_linear(&col, alpha, beta, horizon);
                TbsStepOutput::Vector { name: "holt_forecast", values }
            }

            ("adf_test", None) | ("adf", None) => {
                let n_lags = usize_arg(step, "n_lags", 0, 5);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let result = crate::time_series::adf_test(&col, n_lags);
                TbsStepOutput::Adf(result)
            }

            // ══════════════════════════════════════════════════════════════
            // Signal processing
            // ══════════════════════════════════════════════════════════════

            ("fft", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let spectrum = crate::signal_processing::rfft(&col);
                let magnitudes: Vec<f64> = spectrum.iter().map(|c| (c.0 * c.0 + c.1 * c.1).sqrt()).collect();
                TbsStepOutput::Vector { name: "fft_magnitudes", values: magnitudes }
            }

            ("periodogram", None) => {
                let fs = f64_arg(step, "fs", 0, 1.0);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let (freqs, power) = crate::signal_processing::periodogram(&col, fs);
                let mut flat = Vec::with_capacity(freqs.len() * 2);
                for i in 0..freqs.len() {
                    flat.push(freqs[i]);
                    flat.push(power[i]);
                }
                TbsStepOutput::Matrix { name: "periodogram", data: flat, rows: freqs.len(), cols: 2 }
            }

            ("welch_psd", None) => {
                let seg = usize_arg(step, "segment_len", 0, 256);
                let overlap = usize_arg(step, "overlap", 1, seg / 2);
                let fs = f64_arg(step, "fs", 2, 1.0);
                let c = usize_arg(step, "col", 3, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let (freqs, power) = crate::signal_processing::welch(&col, seg, overlap, fs);
                let mut flat = Vec::with_capacity(freqs.len() * 2);
                for i in 0..freqs.len() {
                    flat.push(freqs[i]);
                    flat.push(power[i]);
                }
                TbsStepOutput::Matrix { name: "welch_psd", data: flat, rows: freqs.len(), cols: 2 }
            }

            ("moving_average", None) => {
                let window = usize_arg(step, "window", 0, 5);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::signal_processing::moving_average(&col, window);
                TbsStepOutput::Vector { name: "moving_average", values }
            }

            ("ema", None) => {
                let alpha = f64_arg(step, "alpha", 0, 0.1);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::signal_processing::ema(&col, alpha);
                TbsStepOutput::Vector { name: "ema", values }
            }

            ("savgol", None) | ("savgol_filter", None) => {
                let window = usize_arg(step, "window", 0, 5);
                let order = usize_arg(step, "order", 1, 2);
                let c = usize_arg(step, "col", 2, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::signal_processing::savgol_filter(&col, window, order);
                TbsStepOutput::Vector { name: "savgol", values }
            }

            ("median_filter", None) => {
                let window = usize_arg(step, "window", 0, 5);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::signal_processing::median_filter(&col, window);
                TbsStepOutput::Vector { name: "median_filter", values }
            }

            ("envelope", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::signal_processing::envelope(&col);
                TbsStepOutput::Vector { name: "envelope", values }
            }

            ("haar_dwt", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let (approx, detail) = crate::signal_processing::haar_dwt(&col);
                let mut flat = Vec::with_capacity(approx.len() + detail.len());
                flat.extend_from_slice(&approx);
                flat.extend_from_slice(&detail);
                TbsStepOutput::Matrix { name: "haar_dwt", data: flat, rows: 2, cols: approx.len() }
            }

            ("autocorrelation", None) => {
                let max_lag = usize_arg(step, "max_lag", 0, 20);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::signal_processing::autocorrelation(&col, max_lag);
                TbsStepOutput::Vector { name: "autocorrelation", values }
            }

            ("zero_crossing_rate", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "zero_crossing_rate", value: crate::signal_processing::zero_crossing_rate(&col) }
            }

            // ══════════════════════════════════════════════════════════════
            // Bayesian
            // ══════════════════════════════════════════════════════════════

            ("bayesian", Some("linear")) | ("bayesian_linear", None) => {
                let y_ref = y.as_deref()
                    .ok_or("bayesian.linear: target data (y) must be provided to executor")?;
                let prior_mean = vec![0.0; pd];
                let mut prior_prec = vec![0.0; pd * pd];
                for j in 0..pd { prior_prec[j * pd + j] = 0.01; }
                let alpha0 = f64_arg(step, "alpha0", 0, 1.0);
                let beta0 = f64_arg(step, "beta0", 1, 1.0);
                let result = crate::bayesian::bayesian_linear_regression(
                    &pipeline.frame().data, y_ref, pn, pd,
                    &prior_mean, &prior_prec, alpha0, beta0,
                );
                TbsStepOutput::Vector { name: "bayesian_beta_mean", values: result.beta_mean }
            }

            ("ess", None) | ("effective_sample_size", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "ess", value: crate::bayesian::effective_sample_size(&col) }
            }

            // ══════════════════════════════════════════════════════════════
            // Survival analysis
            // ══════════════════════════════════════════════════════════════

            ("kaplan_meier", None) => {
                let time_col = usize_arg(step, "time_col", 0, 0);
                let event_col = usize_arg(step, "event_col", 1, 1);
                let times = extract_col(&pipeline.frame().data, pn, pd, time_col);
                let event_vals = extract_col(&pipeline.frame().data, pn, pd, event_col);
                let events: Vec<bool> = event_vals.iter().map(|&v| v > 0.5).collect();
                let steps = crate::survival::kaplan_meier(&times, &events);
                let km_median = crate::survival::km_median(&steps);
                TbsStepOutput::Scalar { name: "km_median_survival", value: km_median }
            }

            ("cox", None) | ("cox_ph", None) => {
                let y_ref = y.as_deref()
                    .ok_or("cox: target data (y) must be provided as [time1, event1, time2, event2, ...]")?;
                let n_obs = y_ref.len() / 2;
                let times: Vec<f64> = (0..n_obs).map(|i| y_ref[i * 2]).collect();
                let events: Vec<bool> = (0..n_obs).map(|i| y_ref[i * 2 + 1] > 0.5).collect();
                let max_iter = usize_arg(step, "max_iter", 0, 50);
                let result = crate::survival::cox_ph(&pipeline.frame().data, &times, &events, pn, pd, max_iter);
                TbsStepOutput::Vector { name: "cox_coefficients", values: result.beta }
            }

            // ══════════════════════════════════════════════════════════════
            // Robust statistics
            // ══════════════════════════════════════════════════════════════

            ("huber", None) | ("huber_m", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let k = f64_arg(step, "k", 1, 1.345);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let result = crate::robust::huber_m_estimate(&col, k, 50, 1e-6);
                TbsStepOutput::Scalar { name: "huber_location", value: result.location }
            }

            ("bisquare", None) | ("bisquare_m", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let k = f64_arg(step, "k", 1, 4.685);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let result = crate::robust::bisquare_m_estimate(&col, k, 50, 1e-6);
                TbsStepOutput::Scalar { name: "bisquare_location", value: result.location }
            }

            ("qn_scale", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "qn_scale", value: crate::robust::qn_scale(&col) }
            }

            ("sn_scale", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "sn_scale", value: crate::robust::sn_scale(&col) }
            }

            ("medcouple", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "medcouple", value: crate::robust::medcouple(&col) }
            }

            ("lts", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let n_trials = usize_arg(step, "n_trials", 2, 500);
                let seed = step.get_arg("seed", 3).and_then(|v| v.as_usize()).unwrap_or(42) as u64;
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let result = crate::robust::lts_simple(&x, &yv, n_trials, seed);
                TbsStepOutput::Vector { name: "lts_coeffs", values: vec![result.intercept, result.slope] }
            }

            // ══════════════════════════════════════════════════════════════
            // Information theory
            // ══════════════════════════════════════════════════════════════

            ("entropy", None) | ("shannon_entropy", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let n_bins = usize_arg(step, "n_bins", 1, 10);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "entropy", value: crate::information_theory::entropy_histogram(&col, n_bins) }
            }

            ("kl_divergence", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (p, q) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "kl_divergence", value: crate::information_theory::kl_divergence(&p, &q) }
            }

            ("js_divergence", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (p, q) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "js_divergence", value: crate::information_theory::js_divergence(&p, &q) }
            }

            ("cross_entropy", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (p, q) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "cross_entropy", value: crate::information_theory::cross_entropy(&p, &q) }
            }

            ("mutual_info", None) | ("mutual_information", None) => {
                let labels = pipeline.frame().labels.as_ref()
                    .ok_or("mutual_info: run a clustering step first to produce labels")?;
                let c = usize_arg(step, "col", 0, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let n_bins = usize_arg(step, "n_bins", 1, 10);
                let min_v = col.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_v = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = (max_v - min_v).max(1e-15);
                let binned: Vec<i32> = col.iter().map(|&v| ((v - min_v) / range * (n_bins as f64 - 1.0)).round() as i32).collect();
                TbsStepOutput::Scalar {
                    name: "mutual_info",
                    value: crate::information_theory::mutual_info_score(labels, &binned),
                }
            }

            // ══════════════════════════════════════════════════════════════
            // Complexity measures
            // ══════════════════════════════════════════════════════════════

            ("sample_entropy", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let m = usize_arg(step, "m", 1, 2);
                let r = f64_arg(step, "r", 2, 0.2);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "sample_entropy", value: crate::complexity::sample_entropy(&col, m, r) }
            }

            ("permutation_entropy", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let m = usize_arg(step, "m", 1, 3);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let tau = usize_arg(step, "tau", 2, 1);
                TbsStepOutput::Scalar { name: "permutation_entropy", value: crate::complexity::permutation_entropy(&col, m, tau) }
            }

            ("hurst", None) | ("hurst_rs", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "hurst", value: crate::complexity::hurst_rs(&col) }
            }

            ("dfa", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let min_box = usize_arg(step, "min_box", 1, 4);
                let max_box = usize_arg(step, "max_box", 2, pn / 4);
                TbsStepOutput::Scalar { name: "dfa", value: crate::complexity::dfa(&col, min_box, max_box) }
            }

            ("higuchi_fd", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let kmax = usize_arg(step, "kmax", 1, 10);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "higuchi_fd", value: crate::complexity::higuchi_fd(&col, kmax) }
            }

            ("lempel_ziv", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "lempel_ziv", value: crate::complexity::lempel_ziv_complexity(&col) as f64 }
            }

            // ══════════════════════════════════════════════════════════════
            // Spatial statistics
            // ══════════════════════════════════════════════════════════════

            ("morans_i", None) => {
                if pd < 3 {
                    return Err("morans_i: need at least 3 columns (x, y, value)".into());
                }
                let points: Vec<(f64, f64)> = (0..pn).map(|i| {
                    let data = &pipeline.frame().data;
                    (data[i * pd], data[i * pd + 1])
                }).collect();
                let k = usize_arg(step, "k", 0, 4);
                let val_col = usize_arg(step, "val_col", 1, 2);
                let values = extract_col(&pipeline.frame().data, pn, pd, val_col);
                let weights = crate::spatial::SpatialWeights::knn(&points, k);
                TbsStepOutput::Scalar { name: "morans_i", value: crate::spatial::morans_i(&values, &weights) }
            }

            ("gearys_c", None) => {
                if pd < 3 {
                    return Err("gearys_c: need at least 3 columns (x, y, value)".into());
                }
                let points: Vec<(f64, f64)> = (0..pn).map(|i| {
                    let data = &pipeline.frame().data;
                    (data[i * pd], data[i * pd + 1])
                }).collect();
                let k = usize_arg(step, "k", 0, 4);
                let val_col = usize_arg(step, "val_col", 1, 2);
                let values = extract_col(&pipeline.frame().data, pn, pd, val_col);
                let weights = crate::spatial::SpatialWeights::knn(&points, k);
                TbsStepOutput::Scalar { name: "gearys_c", value: crate::spatial::gearys_c(&values, &weights) }
            }

            // ══════════════════════════════════════════════════════════════
            // Panel data
            // ══════════════════════════════════════════════════════════════

            ("panel", Some("fe")) | ("panel_fe", None) => {
                let y_ref = y.as_deref()
                    .ok_or("panel.fe: target data (y) must be provided")?;
                let unit_col = usize_arg(step, "unit_col", 0, pd - 1);
                let units: Vec<usize> = extract_col(&pipeline.frame().data, pn, pd, unit_col)
                    .iter().map(|&v| v as usize).collect();
                let x_cols: Vec<usize> = (0..pd).filter(|&j| j != unit_col).collect();
                let x_d = x_cols.len();
                let mut x_data = Vec::with_capacity(pn * x_d);
                for i in 0..pn {
                    for &j in &x_cols {
                        x_data.push(pipeline.frame().data[i * pd + j]);
                    }
                }
                let result = crate::panel::panel_fe(&x_data, y_ref, pn, x_d, &units);
                TbsStepOutput::Vector { name: "fe_beta", values: result.beta }
            }

            // ══════════════════════════════════════════════════════════════
            // Causal inference
            // ══════════════════════════════════════════════════════════════

            ("rdd", None) | ("rdd_sharp", None) => {
                let cutoff = f64_req(step, "cutoff", 0)?;
                let bw = f64_arg(step, "bw", 1, 1.0);
                let cx = usize_arg(step, "col_x", 2, 0);
                let cy = usize_arg(step, "col_y", 3, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let result = crate::causal::rdd_sharp(&x, &yv, cutoff, bw);
                TbsStepOutput::Scalar { name: "rdd_effect", value: result.effect }
            }

            // ══════════════════════════════════════════════════════════════
            // Mixed effects
            // ══════════════════════════════════════════════════════════════

            ("icc", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let labels = pipeline.frame().labels.as_ref()
                    .ok_or("icc: run a clustering step first to produce group labels")?;
                let groups: Vec<usize> = labels.iter().map(|&l| l.max(0) as usize).collect();
                TbsStepOutput::Scalar { name: "icc", value: crate::mixed_effects::icc_oneway(&col, &groups) }
            }

            // ══════════════════════════════════════════════════════════════
            // Volatility
            // ══════════════════════════════════════════════════════════════

            ("garch", None) => {
                let c = col_arg(step, 0);
                let max_iter = usize_arg(step, "max_iter", 1, 100);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let result = crate::volatility::garch11_fit(&col, max_iter);
                TbsStepOutput::Vector {
                    name: "garch_params",
                    values: vec![result.omega, result.alpha, result.beta],
                }
            }

            ("ewma_var", None) | ("ewma_variance", None) => {
                let lambda = f64_arg(step, "lambda", 0, 0.94);
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let values = crate::volatility::ewma_variance(&col, lambda);
                TbsStepOutput::Vector { name: "ewma_variance", values }
            }

            // ══════════════════════════════════════════════════════════════
            // Volatility (additional)
            // ══════════════════════════════════════════════════════════════

            ("realized_variance", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "realized_variance", value: crate::volatility::realized_variance(&col) }
            }

            ("realized_volatility", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "realized_volatility", value: crate::volatility::realized_volatility(&col) }
            }

            // ══════════════════════════════════════════════════════════════
            // Pipeline configuration
            // ══════════════════════════════════════════════════════════════

            ("using", None) => {
                // Accumulate all named args into the using bag.
                // Multiple .using() calls stack; the next computation step drains.
                for arg in &step.args {
                    if let crate::tbs_parser::TbsArg::Named { key, value } = arg {
                        let uv = match value {
                            crate::tbs_parser::TbsValue::Str(s) =>
                                crate::using::UsingValue::Str(s.clone()),
                            crate::tbs_parser::TbsValue::Float(f) =>
                                crate::using::UsingValue::Float(*f),
                            crate::tbs_parser::TbsValue::Int(i) =>
                                crate::using::UsingValue::Int(*i),
                            crate::tbs_parser::TbsValue::Bool(b) =>
                                crate::using::UsingValue::Bool(*b),
                        };
                        using_bag.set(key.clone(), uv);
                    }
                }
                TbsStepOutput::Transform
            }

            // ══════════════════════════════════════════════════════════════
            // Unknown
            // ══════════════════════════════════════════════════════════════

            _ => {
                return Err(format!("unsupported .tbs operation: {}", step.name).into());
            }
        };

        // Drain using bag after every non-using step.
        // Using steps accumulate; everything else consumes.
        if !matches!(step.name.as_str(), ("using", None)) {
            using_bag.clear();
        }

        outputs.push(output);
        advice.push(step_advice);
    }

    // ── Post-chain science lints ──────────────────────────────────────────
    if n_hypothesis_tests >= 3 {
        lints.push(TbsLint {
            code: "L005",
            step_index: Some(chain.steps.len() - 1),
            message: format!(
                "Ran {n_hypothesis_tests} hypothesis tests. Consider bonferroni() or benjamini_hochberg() for multiple comparison correction."
            ),
            severity: LintSeverity::Warning,
        });
    }

    let superpositions = vec![None; outputs.len()];
    Ok(TbsResult { pipeline, linear_model, logistic_model, outputs, lints, superpositions, advice })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tbs_parser::TbsChain;

    fn two_cluster_data() -> (Vec<f64>, usize, usize) {
        let data = vec![
            1.0, 1.0,
            1.0, 2.0,
            2.0, 1.0,
            10.0, 10.0,
            10.0, 11.0,
            11.0, 10.0,
        ];
        (data, 6, 2)
    }

    // ── Existing tests (preserved) ────────────────────────────────────────

    #[test]
    fn execute_normalize() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("normalize()").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        let col0_mean: f64 = result.pipeline.frame().data.iter().step_by(2).sum::<f64>() / 6.0;
        assert!(col0_mean.abs() < 1e-10, "col0_mean = {col0_mean}");
    }

    #[test]
    fn execute_discover_clusters_named_args() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("discover_clusters(epsilon=3.0, min_samples=1)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert_eq!(result.pipeline.frame().n_clusters, Some(2));
    }

    #[test]
    fn execute_discover_clusters_positional_args() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("discover_clusters(3.0, 1)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert_eq!(result.pipeline.frame().n_clusters, Some(2));
    }

    #[test]
    fn execute_full_chain_from_string() {
        let (data, n, d) = two_cluster_data();
        let y: Vec<f64> = data.chunks(2).map(|p| p[0] + p[1]).collect();
        let src = "normalize()\n  .discover_clusters(epsilon=3.0, min_samples=1)\n  .train.linear(target=\"y\")";
        let chain = TbsChain::parse(src).unwrap();
        let result = execute(chain, data, n, d, Some(y)).unwrap();
        assert!(result.linear_model.is_some());
        assert_eq!(result.pipeline.frame().n_clusters, Some(2));
        assert!(result.linear_model.unwrap().r_squared > 0.9);
    }

    #[test]
    fn execute_train_without_y_errors() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("train.linear(target=\"price\")").unwrap();
        let result = execute(chain, data, n, d, None);
        assert!(result.is_err());
    }

    #[test]
    fn execute_unsupported_step_errors() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("window(size=10)").unwrap();
        let result = execute(chain, data, n, d, None);
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("unsupported"), "error should mention 'unsupported': {msg}");
    }

    #[test]
    fn execute_dbscan_alias() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("dbscan(epsilon=3.0, min_samples=1)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert_eq!(result.pipeline.frame().n_clusters, Some(2));
    }

    #[test]
    fn execute_session_shared_after_chain() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("normalize().discover_clusters(epsilon=3.0, min_samples=1)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert!(result.pipeline.session_len() >= 1);
    }

    #[test]
    fn execute_train_logistic() {
        let n = 100;
        let d = 2;
        let mut data = vec![0.0f64; n * d];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let label = if i < n / 2 { 0.0 } else { 1.0 };
            let cx = if label == 0.0 { -2.0 } else { 2.0 };
            let offset = (i % (n / 2)) as f64 / (n as f64 / 2.0) - 0.5;
            data[i * d] = cx + offset * 0.5;
            data[i * d + 1] = cx + offset * 0.3;
            y[i] = label;
        }
        let chain = TbsChain::parse("train.logistic(lr=1.0, max_iter=500, tol=0.00000001)").unwrap();
        let result = execute(chain, data, n, d, Some(y)).unwrap();
        assert!(result.logistic_model.is_some());
        assert!(result.logistic_model.unwrap().accuracy > 0.9);
    }

    #[test]
    fn execute_train_logistic_defaults() {
        let n = 50;
        let d = 1;
        let data: Vec<f64> = (0..n).map(|i| i as f64 / 10.0 - 2.5).collect();
        let y: Vec<f64> = (0..n).map(|i| if i >= n / 2 { 1.0 } else { 0.0 }).collect();
        let chain = TbsChain::parse("train.logistic()").unwrap();
        let result = execute(chain, data, n, d, Some(y)).unwrap();
        assert!(result.logistic_model.is_some());
        assert!(result.logistic_model.unwrap().loss.is_finite());
    }

    #[test]
    fn execute_train_logistic_without_y_errors() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("train.logistic()").unwrap();
        assert!(execute(chain, data, n, d, None).is_err());
    }

    #[test]
    fn execute_knn() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("knn(k=2)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        let knn = result.pipeline.frame().knn_result.as_ref().expect("expected knn_result");
        assert_eq!(knn.n, 6);
        assert_eq!(knn.k, 2);
    }

    #[test]
    fn execute_knn_positional() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("knn(2)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert!(result.pipeline.frame().knn_result.is_some());
    }

    #[test]
    fn execute_knn_missing_k_errors() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("knn()").unwrap();
        assert!(execute(chain, data, n, d, None).is_err());
    }

    #[test]
    fn execute_cluster_then_knn_shares_session() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("discover_clusters(epsilon=3.0, min_samples=1).knn(k=2)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert_eq!(result.pipeline.frame().n_clusters, Some(2));
        assert!(result.pipeline.frame().knn_result.is_some());
        assert_eq!(result.pipeline.session_len(), 1);
    }

    #[test]
    fn execute_kmeans_named_k() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("kmeans(k=2)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert_eq!(result.pipeline.frame().n_clusters, Some(2));
    }

    #[test]
    fn execute_kmeans_positional_k() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("kmeans(2)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert_eq!(result.pipeline.frame().n_clusters, Some(2));
    }

    #[test]
    fn execute_kmeans_missing_k_errors() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("kmeans()").unwrap();
        assert!(execute(chain, data, n, d, None).is_err());
    }

    #[test]
    fn execute_kmeans_custom_max_iter() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("kmeans(k=2, max_iter=50)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert_eq!(result.pipeline.frame().n_clusters, Some(2));
    }

    // ── Descriptive statistics ─────────────────────────────────────────────

    #[test]
    fn execute_describe() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("describe()").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Descriptive(descs) => {
                assert_eq!(descs.len(), 2);
                assert!(descs[0].count == 6.0);
            }
            other => panic!("expected Descriptive, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn execute_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let chain = TbsChain::parse("mean()").unwrap();
        let result = execute(chain, data, 6, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!((*value - 3.5).abs() < 1e-10);
            }
            other => panic!("expected Scalar, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn execute_median() {
        let data = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let chain = TbsChain::parse("median()").unwrap();
        let result = execute(chain, data, 5, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!((*value - 3.0).abs() < 1e-10);
            }
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn execute_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let chain = TbsChain::parse("quantile(q=0.5)").unwrap();
        let result = execute(chain, data, 10, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!((*value - 5.5).abs() < 0.5, "median quantile = {value}");
            }
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn execute_correlation_auto_detect() {
        // Linearly structured data — auto-detect should pick Pearson and return r close to 1.
        // Use 10 points exactly on y = 2x + 1 (perfect linear).
        let data: Vec<f64> = (0..10)
            .flat_map(|i| vec![i as f64, i as f64 * 2.0 + 1.0])
            .collect();
        let n = 10;
        let d = 2;
        let chain = TbsChain::parse("correlation()").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!(*value > 0.99, "perfect linear data: correlation should be > 0.99, got {value}");
            }
            _ => panic!("expected Scalar from auto-detect correlation"),
        }
        assert!(result.advice[0].is_some(), "auto-detect should populate advice");
    }

    #[test]
    fn execute_correlation_matrix() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("correlation_matrix()").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Matrix { rows, cols, data, .. } => {
                assert_eq!(*rows, 2);
                assert_eq!(*cols, 2);
                assert!((data[0] - 1.0).abs() < 1e-10);
                assert!((data[3] - 1.0).abs() < 1e-10);
            }
            _ => panic!("expected Matrix from correlation_matrix"),
        }
    }

    #[test]
    fn execute_correlation_override() {
        // Linearly spaced data — auto would pick Pearson. User forces spearman.
        let data: Vec<f64> = (0..20).flat_map(|i| vec![i as f64, i as f64 * 2.0 + 1.0]).collect();
        let chain = TbsChain::parse("using(method=\"spearman\").correlation()").unwrap();
        let result = execute(chain, data, 20, 2, None).unwrap();
        // outputs[0] = using() Transform, outputs[1] = correlation() Scalar
        match &result.outputs[1] {
            TbsStepOutput::Scalar { name, value } => {
                assert_eq!(*name, "spearman");
                assert!((*value - 1.0).abs() < 1e-6, "spearman of linear data = {value}");
            }
            _ => panic!("expected Scalar at outputs[1]"),
        }
        // advice[1] = correlation step advice
        let adv = result.advice[1].as_ref().unwrap();
        assert!(adv.user_override.is_some(), "override should be recorded in advice");
    }

    // ── Hypothesis testing ────────────────────────────────────────────────

    #[test]
    fn execute_t_test() {
        let data: Vec<f64> = (0..10).map(|i| 5.0 + (i as f64) * 0.1).collect();
        let chain = TbsChain::parse("t_test(mu=0.0)").unwrap();
        let result = execute(chain, data, 10, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Test(t) => {
                assert!(t.p_value < 0.05, "p_value={}", t.p_value);
            }
            _ => panic!("expected Test"),
        }
    }

    #[test]
    fn execute_t_test_2_auto_detect_normal() {
        // Two groups from clearly different normals — auto-detect should run
        // t-test (pooled or Welch) and return a significant result.
        let mut data = Vec::new();
        for i in 0..20 { data.push(i as f64 * 0.1); }         // col 0: 0.0..2.0
        for i in 0..20 { data.push(5.0 + i as f64 * 0.1); }  // col 1: 5.0..7.0
        // Interleave into row-major 20×2
        let interleaved: Vec<f64> = (0..20).flat_map(|i| vec![data[i], data[20 + i]]).collect();
        let chain = TbsChain::parse("t_test_2()").unwrap();
        let result = execute(chain, interleaved, 20, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Test(t) => {
                assert!(t.p_value < 0.001, "groups far apart: p={}", t.p_value);
            }
            _ => panic!("expected Test"),
        }
        assert!(result.advice[0].is_some(), "auto-detect should produce advice");
    }

    #[test]
    fn execute_t_test_2_auto_detect_nonnormal() {
        // Exponentially distributed data — SW will flag as non-normal → Mann-Whitney
        let col_a: Vec<f64> = (0..15).map(|i| (i as f64 * 0.5).exp().min(100.0)).collect();
        let col_b: Vec<f64> = (0..15).map(|i| (i as f64 * 0.5 + 3.0).exp().min(1000.0)).collect();
        let interleaved: Vec<f64> = (0..15).flat_map(|i| vec![col_a[i], col_b[i]]).collect();
        let chain = TbsChain::parse("t_test_2()").unwrap();
        let result = execute(chain, interleaved, 15, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Test(t) => {
                // Mann-Whitney or t-test — just verify a result comes back
                assert!(t.p_value >= 0.0 && t.p_value <= 1.0, "p_value={}", t.p_value);
            }
            _ => panic!("expected Test"),
        }
        let adv = result.advice[0].as_ref().unwrap();
        // Check that at least one diagnostic was recorded
        assert!(!adv.diagnostics.is_empty(), "should have normality diagnostics");
    }

    #[test]
    fn execute_spearman() {
        let data = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
            4.0, 40.0,
            5.0, 50.0,
        ];
        let chain = TbsChain::parse("spearman()").unwrap();
        let result = execute(chain, data, 5, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!((*value - 1.0).abs() < 1e-10, "spearman={value}");
            }
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn execute_ks_test() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.123).sin()).collect();
        let chain = TbsChain::parse("ks_test()").unwrap();
        let result = execute(chain, data, 100, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Nonparametric(r) => {
                assert!(r.statistic >= 0.0);
            }
            _ => panic!("expected Nonparametric"),
        }
    }

    // ── Dimensionality reduction ──────────────────────────────────────────

    #[test]
    fn execute_pca() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("pca(n_components=1)").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Pca(pca) => {
                assert_eq!(pca.explained_variance_ratio.len(), 1);
                assert!(pca.explained_variance_ratio[0] > 0.8);
            }
            _ => panic!("expected Pca"),
        }
    }

    // ── Time series ───────────────────────────────────────────────────────

    #[test]
    fn execute_acf() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let chain = TbsChain::parse("acf(max_lag=10)").unwrap();
        let result = execute(chain, data, 50, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert!(values.len() >= 10, "acf should have at least 10 lags, got {}", values.len());
            }
            _ => panic!("expected Vector"),
        }
    }

    #[test]
    fn execute_adf_test() {
        // Random-walk-like data: cumulative sum of pseudo-random values
        let mut data = Vec::with_capacity(200);
        let mut rng = 42u64;
        let mut cumsum = 0.0;
        for _ in 0..200 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = rng as f64 / u64::MAX as f64 - 0.5;
            cumsum += u;
            data.push(cumsum);
        }
        let chain = TbsChain::parse("adf_test(n_lags=3)").unwrap();
        let result = execute(chain, data, 200, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Adf(adf) => {
                assert!(adf.statistic.is_finite());
            }
            _ => panic!("expected Adf"),
        }
    }

    // ── Signal processing ─────────────────────────────────────────────────

    #[test]
    fn execute_fft() {
        let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let chain = TbsChain::parse("fft()").unwrap();
        let result = execute(chain, data, 64, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert!(!values.is_empty());
                assert!(values.iter().all(|v| v.is_finite()));
            }
            _ => panic!("expected Vector"),
        }
    }

    #[test]
    fn execute_moving_average() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let chain = TbsChain::parse("moving_average(window=3)").unwrap();
        let result = execute(chain, data, 20, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert!(!values.is_empty());
            }
            _ => panic!("expected Vector"),
        }
    }

    // ── Robust statistics ─────────────────────────────────────────────────

    #[test]
    fn execute_huber() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let chain = TbsChain::parse("huber(col=0, k=1.345)").unwrap();
        let result = execute(chain, data, 6, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!(*value < 10.0, "huber location={value} should resist outlier");
            }
            _ => panic!("expected Scalar"),
        }
    }

    // ── Information theory ────────────────────────────────────────────────

    #[test]
    fn execute_entropy() {
        let data: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let chain = TbsChain::parse("entropy(col=0, n_bins=10)").unwrap();
        let result = execute(chain, data, 100, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!(*value > 0.0);
            }
            _ => panic!("expected Scalar"),
        }
    }

    // ── Complexity measures ───────────────────────────────────────────────

    #[test]
    fn execute_hurst() {
        let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin()).collect();
        let chain = TbsChain::parse("hurst()").unwrap();
        let result = execute(chain, data, 200, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!(*value > 0.0 && *value < 2.0, "hurst={value}");
            }
            _ => panic!("expected Scalar"),
        }
    }

    // ── Science linting ───────────────────────────────────────────────────

    #[test]
    fn lint_warns_on_unnormalized_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chain = TbsChain::parse("variance()").unwrap();
        let result = execute(chain, data, 5, 1, None).unwrap();
        assert!(!result.lints.is_empty(), "should lint on unnormalized variance");
        assert!(result.lints[0].message.contains("normalize"));
    }

    #[test]
    fn lint_no_warning_after_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chain = TbsChain::parse("normalize().variance()").unwrap();
        let result = execute(chain, data, 5, 1, None).unwrap();
        let variance_lints: Vec<_> = result.lints.iter()
            .filter(|l| l.message.contains("normalize"))
            .collect();
        assert!(variance_lints.is_empty(), "no normalize lint after normalize()");
    }

    #[test]
    fn lint_warns_on_small_n_t_test() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chain = TbsChain::parse("t_test(mu=0.0)").unwrap();
        let result = execute(chain, data, 5, 1, None).unwrap();
        let small_n_lints: Vec<_> = result.lints.iter()
            .filter(|l| l.message.contains("n=5"))
            .collect();
        assert!(!small_n_lints.is_empty(), "should warn on small n");
    }

    #[test]
    fn lint_warns_on_multiple_hypothesis_tests() {
        let data = vec![
            1.0, 10.0, 100.0,
            2.0, 20.0, 200.0,
            3.0, 30.0, 300.0,
            4.0, 40.0, 400.0,
            5.0, 50.0, 500.0,
        ];
        let chain = TbsChain::parse(
            "t_test(mu=0.0, col=0).t_test(mu=0.0, col=1).t_test(mu=0.0, col=2)"
        ).unwrap();
        let result = execute(chain, data, 5, 3, None).unwrap();
        let mc_lints: Vec<_> = result.lints.iter()
            .filter(|l| l.message.contains("multiple comparison"))
            .collect();
        assert!(!mc_lints.is_empty(), "should warn about multiple comparisons");
    }

    #[test]
    fn lint_suggests_normality_check() {
        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let chain = TbsChain::parse("t_test(mu=0.0)").unwrap();
        let result = execute(chain, data, 30, 1, None).unwrap();
        let normality_lints: Vec<_> = result.lints.iter()
            .filter(|l| l.message.contains("normality"))
            .collect();
        assert!(!normality_lints.is_empty(), "should suggest normality check");
    }

    #[test]
    fn lint_no_normality_warning_after_ks_test() {
        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let chain = TbsChain::parse("ks_test().t_test(mu=0.0)").unwrap();
        let result = execute(chain, data, 30, 1, None).unwrap();
        let normality_lints: Vec<_> = result.lints.iter()
            .filter(|l| l.message.contains("normality"))
            .collect();
        assert!(normality_lints.is_empty(), "no normality lint after ks_test()");
    }

    // ── Chain integration ─────────────────────────────────────────────────

    #[test]
    fn execute_outputs_per_step() {
        let (data, n, d) = two_cluster_data();
        let chain = TbsChain::parse("normalize().describe().spearman()").unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert_eq!(result.outputs.len(), 3);
        assert!(matches!(result.outputs[0], TbsStepOutput::Transform));
        assert!(matches!(result.outputs[1], TbsStepOutput::Descriptive(_)));
        assert!(matches!(result.outputs[2], TbsStepOutput::Scalar { .. }));
    }

    #[test]
    fn execute_full_analysis_chain() {
        let (data, n, d) = two_cluster_data();
        let src = "normalize().describe().ks_test().t_test(mu=0.0).pca(n_components=1)";
        let chain = TbsChain::parse(src).unwrap();
        let result = execute(chain, data, n, d, None).unwrap();
        assert_eq!(result.outputs.len(), 5);
        assert!(matches!(result.outputs[0], TbsStepOutput::Transform));
        assert!(matches!(result.outputs[1], TbsStepOutput::Descriptive(_)));
        assert!(matches!(result.outputs[2], TbsStepOutput::Nonparametric(_)));
        assert!(matches!(result.outputs[3], TbsStepOutput::Test(_)));
        assert!(matches!(result.outputs[4], TbsStepOutput::Pca(_)));
        let normality_lints: Vec<_> = result.lints.iter()
            .filter(|l| l.message.contains("normality"))
            .collect();
        assert!(normality_lints.is_empty());
    }
}
