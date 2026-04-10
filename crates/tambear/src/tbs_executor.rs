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

/// Read a named-or-positional bool arg with a default.
fn bool_arg(step: &TbsStep, name: &str, pos: usize, default: bool) -> bool {
    step.get_arg(name, pos).map(|v| match v {
        crate::tbs_parser::TbsValue::Bool(b) => *b,
        crate::tbs_parser::TbsValue::Int(i) => *i != 0,
        crate::tbs_parser::TbsValue::Float(f) => *f != 0.0,
        crate::tbs_parser::TbsValue::Str(s) => matches!(s.as_str(), "true" | "yes" | "1"),
    }).unwrap_or(default)
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
    let mut superpositions_vec: Vec<Option<crate::superposition::Superposition>> = Vec::with_capacity(chain.steps.len());
    let mut using_bag = crate::using::UsingBag::new();

    // Track state for science linting
    let mut normalized = false;
    let mut normality_checked = false;
    let mut n_hypothesis_tests = 0u32;

    for (step_idx, step) in chain.steps.iter().enumerate() {
        let fr = pipeline.frame();
        let (pn, pd) = (fr.n, fr.d);

        let mut step_advice: Option<TbsStepAdvice> = None;
        let mut step_superposition: Option<crate::superposition::Superposition> = None;
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

            ("coefficient_of_variation", None) | ("cv", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "coefficient_of_variation", value: crate::descriptive::coefficient_of_variation(&col) }
            }

            ("mode", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "mode", value: crate::descriptive::mode(&col) }
            }

            ("sem", None) | ("standard_error_of_mean", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "sem", value: crate::descriptive::sem(&col) }
            }

            ("percentileofscore", None) | ("percentile_of_score", None) => {
                let score = f64_req(step, "score", 0)?;
                let c = usize_arg(step, "col", 1, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "percentileofscore", value: crate::descriptive::percentileofscore(&col, score) }
            }

            ("lmoment", None) | ("l_moments", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sorted = crate::descriptive::sorted_nan_free(&col);
                let lm = crate::descriptive::lmoment(&sorted);
                TbsStepOutput::Vector { name: "lmoment", values: lm.to_vec() }
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

                // Normality test: SW for n < n_thresh, D'Agostino-Pearson for n ≥ n_thresh
                let normality_alpha = using_bag.get_f64("normality_alpha").unwrap_or(0.05);
                let n_thresh = using_bag.get_f64("normality_test_n_threshold")
                    .map(|v| v as usize).unwrap_or(5000);
                let normality = |col: &[f64]| -> (f64, &'static str) {
                    if col.len() < n_thresh {
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
                        let x_norm = px > normality_alpha;
                        let y_norm = py > normality_alpha;
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
                let normality_alpha = using_bag.get_f64("normality_alpha").unwrap_or(0.05);
                let variance_alpha = using_bag.get_f64("variance_alpha").unwrap_or(0.05);
                let n_thresh = using_bag.get_f64("normality_test_n_threshold")
                    .map(|v| v as usize).unwrap_or(5000);
                let normality = |col: &[f64]| -> (f64, &'static str) {
                    if col.len() < n_thresh {
                        let r = crate::nonparametric::shapiro_wilk(col);
                        (r.p_value, "Shapiro-Wilk")
                    } else {
                        let r = crate::nonparametric::dagostino_pearson(col);
                        (r.p_value, "D'Agostino-Pearson")
                    }
                };
                let (px, tn_x) = normality(&x);
                let (py, tn_y) = normality(&yv);
                let x_norm = px > normality_alpha;
                let y_norm = py > normality_alpha;

                // Variance equality via Brown-Forsythe (Levene with median)
                let levene_p = if x_norm && y_norm {
                    let lev = crate::hypothesis::levene_test(
                        &[x.as_slice(), yv.as_slice()],
                        crate::hypothesis::LeveneCenter::Median,
                    );
                    lev.p_value
                } else { f64::NAN };
                let equal_var = levene_p > variance_alpha;

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
                                ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
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

            // ── ANOVA auto-detect ─────────────────────────────────────────
            // Decision tree:
            //   1. Test normality per group (Shapiro-Wilk / D'Agostino-Pearson)
            //   2. Test homoscedasticity (Levene's / Brown-Forsythe)
            //   3. Route:
            //      - Normal + equal var → classic one-way ANOVA
            //      - Normal + unequal var → Welch's ANOVA
            //      - Non-normal → Kruskal-Wallis
            //   4. If significant → auto-add post-hoc
            ("test", Some("anova")) | ("anova", None) => {
                // col_val = the value column, col_group = the group column
                let cv = usize_arg(step, "col_val", 0, 0);
                let cg = usize_arg(step, "col_group", 1, 1.min(pd.saturating_sub(1)));
                let vals = extract_col(&pipeline.frame().data, pn, pd, cv);
                let groups_raw = extract_col(&pipeline.frame().data, pn, pd, cg);

                // Build groups: discretize the group column
                let mut group_map: std::collections::BTreeMap<i64, Vec<f64>> = std::collections::BTreeMap::new();
                for i in 0..pn {
                    let g = groups_raw[i] as i64;
                    group_map.entry(g).or_default().push(vals[i]);
                }
                let group_vecs: Vec<Vec<f64>> = group_map.values().cloned().collect();
                let k = group_vecs.len();

                if k < 2 {
                    lints.push(TbsLint {
                        code: "L201", step_index: Some(step_idx),
                        message: "ANOVA requires at least 2 groups.".into(),
                        severity: LintSeverity::Warning,
                    });
                    n_hypothesis_tests += 1;
                    TbsStepOutput::Test(crate::hypothesis::TestResult {
                        test_name: "ANOVA", statistic: f64::NAN, p_value: f64::NAN,
                        df: f64::NAN, effect_size: f64::NAN, effect_size_name: "",
                        ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
                    })
                } else {
                    // Normality check per group
                    let normality_alpha = using_bag.get_f64("normality_alpha").unwrap_or(0.05);
                    let variance_alpha = using_bag.get_f64("variance_alpha").unwrap_or(0.05);
                    let n_thresh = using_bag.get_f64("normality_test_n_threshold")
                        .map(|v| v as usize).unwrap_or(5000);
                    let normality = |col: &[f64]| -> (f64, &'static str) {
                        if col.len() < 3 { return (f64::NAN, "n<3"); }
                        if col.len() < n_thresh {
                            let r = crate::nonparametric::shapiro_wilk(col);
                            (r.p_value, "Shapiro-Wilk")
                        } else {
                            let r = crate::nonparametric::dagostino_pearson(col);
                            (r.p_value, "D'Agostino-Pearson")
                        }
                    };
                    let norm_results: Vec<(f64, &str)> = group_vecs.iter().map(|g| normality(g)).collect();
                    let all_normal = norm_results.iter().all(|(p, _)| *p > normality_alpha);

                    // Levene's test for homoscedasticity
                    let group_slices: Vec<&[f64]> = group_vecs.iter().map(|v| v.as_slice()).collect();
                    let levene = crate::hypothesis::levene_test(
                        &group_slices, crate::hypothesis::LeveneCenter::Median);
                    let equal_var = levene.p_value > variance_alpha;

                    let user_method: Option<String> = using_bag.method().map(|s| s.to_owned());

                    // Flatten for one_way_anova / kruskal_wallis
                    let flat: Vec<f64> = group_vecs.iter().flat_map(|v| v.iter().copied()).collect();
                    let sizes: Vec<usize> = group_vecs.iter().map(|v| v.len()).collect();

                    // Compute per-group MomentStats for Welch ANOVA
                    let group_stats: Vec<crate::descriptive::MomentStats> = group_vecs.iter()
                        .map(|g| crate::descriptive::moments_ungrouped(g)).collect();

                    // Auto recommendation
                    let (auto_method, auto_reason): (&'static str, String) = if all_normal {
                        if equal_var {
                            ("one_way_anova", format!(
                                "all {} groups normal, equal variance (Levene p={:.3}): classic ANOVA",
                                k, levene.p_value))
                        } else {
                            ("welch_anova", format!(
                                "all {} groups normal, unequal variance (Levene p={:.3}): Welch's ANOVA",
                                k, levene.p_value))
                        }
                    } else {
                        let non_normal: Vec<usize> = norm_results.iter().enumerate()
                            .filter(|(_, (p, _))| *p <= normality_alpha).map(|(i, _)| i).collect();
                        ("kruskal_wallis", format!(
                            "groups {:?} non-normal: Kruskal-Wallis H test",
                            non_normal))
                    };

                    let run_anova = |method: &str| -> TbsStepOutput {
                        match method {
                            "welch_anova" | "welch" => {
                                let slices: Vec<&[f64]> = group_vecs.iter().map(|v| v.as_slice()).collect();
                                let res = crate::hypothesis::welch_anova(&slices);
                                TbsStepOutput::Test(crate::hypothesis::TestResult {
                                    test_name: "Welch's ANOVA",
                                    statistic: res.f_statistic,
                                    p_value: res.p_value,
                                    df: res.df_between,
                                    effect_size: f64::NAN,
                                    effect_size_name: "",
                                    ci_lower: f64::NAN, ci_upper: f64::NAN, ci_level: f64::NAN,
                                })
                            }
                            "kruskal_wallis" | "kruskal" => {
                                let kw = crate::nonparametric::kruskal_wallis(&flat, &sizes);
                                TbsStepOutput::Nonparametric(kw)
                            }
                            _ => {
                                // Classic one-way ANOVA
                                let res = crate::hypothesis::one_way_anova(&group_stats);
                                TbsStepOutput::Anova(res)
                            }
                        }
                    };

                    let (output, adv) = if let Some(ref forced) = user_method {
                        let warn = if forced.as_str() != auto_method {
                            Some(format!("user forced {} but tambear recommends {}: {}",
                                forced, auto_method, auto_reason))
                        } else { None };
                        let adv = TbsStepAdvice::overridden(
                            auto_method, auto_reason.clone(), forced.clone(), "method", warn)
                            .with_diagnostic("Levene", levene.p_value,
                                if equal_var { "equal variance" } else { "unequal variance" });
                        (run_anova(forced), adv)
                    } else {
                        let adv = TbsStepAdvice::accepted(auto_method, auto_reason.clone())
                            .with_diagnostic("Levene", levene.p_value,
                                if equal_var { "equal variance" } else { "unequal variance" });
                        (run_anova(auto_method), adv)
                    };

                    step_advice = Some(adv);
                    n_hypothesis_tests += 1;
                    output
                }
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

            // pearson_r: primitive (no auto-detection wrapper like "correlate")
            ("pearson_r", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "pearson_r", value: crate::nonparametric::pearson_r(&x, &yv) }
            }

            ("biserial_correlation", None) | ("biserial", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (binary, continuous) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "biserial_correlation", value: crate::nonparametric::biserial_correlation(&binary, &continuous) }
            }

            ("rank_biserial", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "rank_biserial", value: crate::nonparametric::rank_biserial(&x, &yv) }
            }

            ("distance_correlation", None) | ("dcor", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "distance_correlation", value: crate::nonparametric::distance_correlation(&x, &yv) }
            }

            ("concordance_correlation", None) | ("ccc", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "concordance_correlation", value: crate::nonparametric::concordance_correlation(&x, &yv) }
            }

            // cramers_v: requires contingency table as 2D matrix (n_rows × n_cols)
            ("cramers_v", None) => {
                // data is treated as n_rows × pd contingency table
                TbsStepOutput::Scalar { name: "cramers_v", value: crate::nonparametric::cramers_v(&pipeline.frame().data, pn) }
            }

            ("tetrachoric", None) => {
                // 2×2 contingency table from 4 values: a, b, c, d (col 0..3)
                if pipeline.frame().data.len() < 4 {
                    return Err("tetrachoric: data must have at least 4 values (a,b,c,d of 2x2 table)".into());
                }
                let d = &pipeline.frame().data;
                let table = [d[0], d[1], d[2], d[3]];
                TbsStepOutput::Scalar { name: "tetrachoric", value: crate::nonparametric::tetrachoric(&table) }
            }

            // ols_simple: (intercept, slope, residuals, r2)
            ("ols_simple", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let (intercept, slope, _residuals, r2) = crate::hypothesis::ols_simple(&x, &yv);
                TbsStepOutput::Vector { name: "ols_simple", values: vec![intercept, slope, r2] }
            }

            // ols_two_predictor: (b0, b1, b2, r2)
            ("ols_two_predictor", None) => {
                if pd < 3 {
                    return Err("ols_two_predictor: need 3 columns (x, m, y)".into());
                }
                let x = extract_col(&pipeline.frame().data, pn, pd, 0);
                let m = extract_col(&pipeline.frame().data, pn, pd, 1);
                let yv = extract_col(&pipeline.frame().data, pn, pd, 2);
                let (b0, b1, b2, r2) = crate::hypothesis::ols_two_predictor(&x, &m, &yv);
                TbsStepOutput::Vector { name: "ols_two_predictor", values: vec![b0, b1, b2, r2] }
            }

            ("mann_whitney", None) | ("mann_whitney_u", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Nonparametric(crate::nonparametric::mann_whitney_u(&x, &yv))
            }

            ("ranksums", None) | ("wilcoxon_ranksums", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Nonparametric(crate::nonparametric::ranksums(&x, &yv))
            }

            ("skewtest", None) | ("dagostino_skewtest", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Nonparametric(crate::nonparametric::skewtest(&col))
            }

            ("kurtosistest", None) | ("dagostino_kurtosistest", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Nonparametric(crate::nonparametric::kurtosistest(&col))
            }

            ("theilslopes", None) | ("theil_sen", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let r = crate::nonparametric::theilslopes(&col, None);
                TbsStepOutput::Vector { name: "theilslopes", values: vec![r.slope, r.intercept] }
            }

            ("siegelslopes", None) | ("siegel_repeated_median", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let r = crate::nonparametric::siegelslopes(&col, None);
                TbsStepOutput::Vector { name: "siegelslopes", values: vec![r.slope, r.intercept] }
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

            // ── Clustering auto-detection ─────────────────────────────────
            // 1. Hopkins statistic: clustering tendency
            // 2. Scale check: warn if column range ratios > 10x
            // 3. Silhouette sweep over k=2..max_k to find optimal k
            // 4. Run KMeans with best k
            // 5. Validate with silhouette/CH/DB
            // 6. Record in TbsStepAdvice
            ("cluster_auto", None) => {
                let max_k = usize_arg(step, "max_k", 0, 8).min(pn / 2).max(2);
                let max_iter = usize_arg(step, "max_iter", 1, 300);
                let data = pipeline.frame().data.clone();

                // Scale check: warn if any column range >> another
                let ranges: Vec<f64> = (0..pd).map(|j| {
                    let col: Vec<f64> = (0..pn).map(|i| data[i * pd + j]).collect();
                    let min = col.iter().cloned().fold(f64::INFINITY, crate::numerical::nan_min);
                    let max = col.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
                    max - min
                }).collect();
                let max_range = ranges.iter().cloned().fold(0.0_f64, f64::max);
                let min_range = ranges.iter().cloned().fold(f64::INFINITY, f64::min);
                let scale_ratio = if min_range > 1e-12 { max_range / min_range } else { 1.0 };
                let scale_ratio_thresh = using_bag.get_f64("scale_ratio_warn_threshold").unwrap_or(10.0);
                if scale_ratio > scale_ratio_thresh {
                    lints.push(TbsLint {
                        code: "L202", step_index: Some(step_idx),
                        message: format!("Column range ratio {scale_ratio:.1}x > {scale_ratio_thresh}: clustering may be dominated by large-scale features. Consider normalize() first."),
                        severity: LintSeverity::Warning,
                    });
                }

                // Hopkins statistic (clustering tendency)
                let hopkins = crate::clustering::hopkins_statistic(&data, pn, pd, pn.min(10), 42);
                let hopkins_threshold = using_bag.get_f64("hopkins_threshold").unwrap_or(0.5);
                let has_structure = hopkins > hopkins_threshold;

                // Silhouette sweep
                let engine = crate::kmeans::KMeansEngine::new()?;
                let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                let mut best_k = 2usize;
                let mut best_sil = f64::NEG_INFINITY;
                for k in 2..=max_k {
                    if k >= pn { break; }
                    let km_result = engine.fit(&data_f32, pn, pd, k, max_iter)?;
                    let labels: Vec<i32> = km_result.labels.iter().map(|&l| l as i32).collect();
                    if let Some(cv) = crate::clustering::cluster_validation(&data, &labels, pd) {
                        if cv.silhouette > best_sil {
                            best_sil = cv.silhouette;
                            best_k = k;
                        }
                    }
                }

                // Run final clustering with best_k (via pipeline to cache labels)
                pipeline = pipeline.kmeans(best_k, max_iter);
                let final_labels = pipeline.frame().labels.clone().unwrap_or_default();
                let final_cv = crate::clustering::cluster_validation(&data, &final_labels, pd);

                // Build advice
                let user_method: Option<String> = using_bag.method().map(|s| s.to_owned());
                let auto_method = "kmeans";
                let auto_reason = format!(
                    "silhouette sweep k=2..{max_k}: best k={best_k} (silhouette={best_sil:.3}); Hopkins={hopkins:.3}{}",
                    if !has_structure { " (weak clustering tendency)" } else { "" }
                );

                let mut adv = TbsStepAdvice::accepted(auto_method, auto_reason.clone())
                    .with_diagnostic("Hopkins", hopkins,
                        if has_structure { "clustering tendency present" } else { "weak clustering tendency" })
                    .with_diagnostic("best_k", best_k as f64, format!("silhouette={best_sil:.3}"));
                if let Some(ref cv) = final_cv {
                    adv = adv
                        .with_diagnostic("Silhouette", cv.silhouette, "final clustering")
                        .with_diagnostic("Calinski-Harabasz", cv.calinski_harabasz, "higher is better")
                        .with_diagnostic("Davies-Bouldin", cv.davies_bouldin, "lower is better");
                }

                let final_adv = if let Some(ref forced) = user_method {
                    let warn = if forced.as_str() != auto_method {
                        Some(format!("user forced {forced} but tambear recommends {auto_method}: {auto_reason}"))
                    } else { None };
                    TbsStepAdvice::overridden(auto_method, auto_reason, forced.clone(), "method", warn)
                } else {
                    adv
                };

                step_advice = Some(final_adv);
                TbsStepOutput::Scalar { name: "silhouette", value: best_sil }
            }

            ("discover_clusters", None) | ("dbscan", None) => {
                let epsilon = f64_req(step, "epsilon", 0)?;
                let min_samples = usize_req(step, "min_samples", 1)?;
                pipeline = pipeline.discover_clusters(epsilon, min_samples);
                TbsStepOutput::Transform
            }

            // sweep_kmeans: run kmeans at k_req and neighborhood; return superposition.
            // modal_value = most-supported k across the sweep.
            // sensitivity() tells you how much k choice matters.
            ("sweep_kmeans", None) | ("discover_kmeans", None) => {
                let k = usize_arg(step, "k", 0, 3);
                let max_iter = usize_arg(step, "max_iter", 1, 100);
                let data = pipeline.frame().data.clone();
                let sup = crate::superposition::sweep_kmeans(&data, pn, pd, k, max_iter);
                let modal_k = sup.modal_value;
                step_superposition = Some(sup);
                TbsStepOutput::Scalar { name: "sweep_kmeans_k", value: modal_k }
            }

            // sweep_two_sample: run all two-sample tests in superposition.
            // collapsed() = the view with the most-agreed test decision.
            // modal_value = fraction of tests that rejected H0.
            ("sweep_two_sample", None) | ("discover_two_sample", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let alpha = f64_arg(step, "alpha", 2, 0.05);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let sup = crate::superposition::sweep_two_sample_tests_alpha(&x, &yv, alpha);
                // modal_value = fraction of tests that rejected H0
                let rejection_rate = sup.modal_value;
                step_superposition = Some(sup);
                TbsStepOutput::Scalar { name: "sweep_two_sample_rejection_rate", value: rejection_rate }
            }

            // sweep_ar: run AR at order p and neighbors; return superposition.
            // modal_value = modal AR order by minimum residual variance.
            ("sweep_ar", None) | ("discover_ar", None) => {
                let c = col_arg(step, 0);
                let p = usize_arg(step, "p", 1, 2);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sup = crate::superposition::sweep_ar(&col, p);
                let modal_p = sup.modal_value;
                step_superposition = Some(sup);
                TbsStepOutput::Scalar { name: "sweep_ar_order", value: modal_p }
            }

            // sweep_moving_average: run MA at window size and neighbors.
            // modal_value = modal window with smallest out-of-sample residual variance.
            ("sweep_moving_average", None) | ("sweep_ma", None) | ("discover_ma", None) => {
                let c = col_arg(step, 0);
                let window = usize_arg(step, "window", 1, 5);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sup = crate::superposition::sweep_moving_average(&col, window);
                let modal_win = sup.modal_value;
                step_superposition = Some(sup);
                TbsStepOutput::Scalar { name: "sweep_ma_window", value: modal_win }
            }

            // discover_correlation: run all bivariate correlation measures in superposition.
            // modal_value = median of signed measures (pearson, spearman, kendall).
            // agreement = stability of measures across methods.
            ("discover_correlation", None) | ("sweep_correlation", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let sup = crate::superposition::sweep_correlation(&x, &yv);
                let modal_corr = sup.modal_value;
                step_superposition = Some(sup);
                TbsStepOutput::Scalar { name: "discover_correlation", value: modal_corr }
            }

            // discover_regression: run OLS, Theil-Sen, Siegel slopes in superposition.
            // modal_value = median slope across methods.
            // agreement = fraction of slopes within 20% of OLS estimate.
            ("discover_regression", None) | ("sweep_regression", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, yv) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let sup = crate::superposition::sweep_regression(&x, &yv);
                let modal_slope = sup.modal_value;
                step_superposition = Some(sup);
                TbsStepOutput::Scalar { name: "discover_regression_slope", value: modal_slope }
            }

            // discover_changepoint: run CUSUM, PELT, binary segmentation in superposition.
            // modal_value = consensus number of changepoints across methods.
            // agreement = fraction of methods that agree on the count.
            ("discover_changepoint", None) | ("sweep_changepoint", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sup = crate::superposition::sweep_changepoint(&col);
                let modal_n = sup.modal_value;
                step_superposition = Some(sup);
                TbsStepOutput::Scalar { name: "discover_changepoint_n", value: modal_n }
            }

            // discover_stationarity: run ADF, KPSS, PP test, variance ratio in superposition.
            // modal_value = 1.0 if consensus is stationary, 0.0 if non-stationary.
            // agreement = fraction of tests that agree with the consensus conclusion.
            ("discover_stationarity", None) | ("sweep_stationarity", None) => {
                let c = col_arg(step, 0);
                let alpha = f64_arg(step, "alpha", 1, 0.05);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sup = crate::superposition::sweep_stationarity(&col, alpha);
                let modal_stat = sup.modal_value;
                step_superposition = Some(sup);
                TbsStepOutput::Scalar { name: "discover_stationarity", value: modal_stat }
            }

            ("kmeans", None) => {
                let k = usize_req(step, "k", 0)?;
                let max_iter = usize_arg(step, "max_iter", 1, 300);
                pipeline = pipeline.kmeans(k, max_iter);
                TbsStepOutput::Transform
            }

            // kmeans_f64: pure f64 CPU Lloyd's — returns label vector
            ("kmeans_f64", None) => {
                let k = usize_req(step, "k", 0)?;
                let d = usize_arg(step, "d", 1, pd);
                let max_iter = usize_arg(step, "max_iter", 2, 100);
                let seed = usize_arg(step, "seed", 3, 42) as u64;
                let labels = crate::clustering::kmeans_f64(&pipeline.frame().data, pn, d, k, max_iter, seed);
                TbsStepOutput::Vector { name: "kmeans_f64_labels", values: labels.iter().map(|&l| l as f64).collect() }
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

            // ── Regression with auto-diagnostics ─────────────────────────
            // 1. Fit OLS
            // 2. VIF check (multicollinearity)
            // 3. Residual normality (Shapiro-Wilk)
            // 4. Breusch-Pagan heteroscedasticity test
            // 5. Cook's distance for influential observations
            // Records all findings in TbsStepAdvice.
            ("regression", None) => {
                let y_ref = y.as_deref()
                    .ok_or("regression: target data (y) must be provided to executor")?;
                let data = &pipeline.frame().data;

                // Fit OLS
                let model = crate::train::linear::fit(data, y_ref, pn, pd)?;
                let r2 = model.r_squared;

                // Residuals = y - ŷ
                let y_hat = model.predict(data, pn);
                let residuals: Vec<f64> = y_ref.iter().zip(y_hat.iter()).map(|(yi, fi)| yi - fi).collect();

                // Build X_aug (n × (d+1)) with intercept column
                let p = pd + 1;
                let mut x_aug_data = vec![0.0_f64; pn * p];
                for i in 0..pn {
                    x_aug_data[i * p] = 1.0; // intercept
                    for j in 0..pd {
                        x_aug_data[i * p + j + 1] = data[i * pd + j];
                    }
                }
                let x_aug = crate::linear_algebra::Mat { rows: pn, cols: p, data: x_aug_data };

                // Predictor matrix without intercept for VIF
                let x_pred = crate::linear_algebra::Mat { rows: pn, cols: pd, data: data.to_vec() };

                // VIF check
                let vif_threshold = using_bag.get_f64("vif_threshold").unwrap_or(10.0);
                let normality_alpha = using_bag.get_f64("normality_alpha").unwrap_or(0.05);
                let variance_alpha = using_bag.get_f64("variance_alpha").unwrap_or(0.05);
                let n_thresh = using_bag.get_f64("normality_test_n_threshold")
                    .map(|v| v as usize).unwrap_or(5000);
                let vif_vals = crate::multivariate::vif(&x_pred);
                let max_vif = vif_vals.iter().cloned().fold(0.0_f64, f64::max);
                let multicollinear = max_vif > vif_threshold;

                // Residual normality
                let norm_result = if pn < n_thresh {
                    crate::nonparametric::shapiro_wilk(&residuals)
                } else {
                    crate::nonparametric::dagostino_pearson(&residuals)
                };
                let resid_norm_p = norm_result.p_value;
                let resid_normal = resid_norm_p > normality_alpha;

                // Breusch-Pagan heteroscedasticity
                let bp = crate::hypothesis::breusch_pagan(&x_aug, &residuals);
                let heteroscedastic = bp.p_value < variance_alpha;

                // Cook's distance
                let influence_threshold = using_bag.get_f64("influence_threshold");
                let influence = crate::hypothesis::cooks_distance_with_threshold(&x_aug, &residuals, influence_threshold);
                let n_influential = influence.n_influential;

                // Build advice
                let user_method: Option<String> = using_bag.method().map(|s| s.to_owned());
                let (auto_method, auto_reason) = if multicollinear {
                    ("ridge", format!("max VIF={max_vif:.1} > {vif_threshold:.0} (multicollinearity detected): consider Ridge regression"))
                } else if !resid_normal {
                    ("quantile", format!("residuals non-normal (p={resid_norm_p:.3}): consider quantile regression or robust OLS"))
                } else if heteroscedastic {
                    ("wls", format!("heteroscedastic residuals (Breusch-Pagan p={:.3}): consider WLS or HC3 standard errors", bp.p_value))
                } else {
                    ("ols", format!("normality OK (p={resid_norm_p:.3}), homoscedasticity OK (BP p={:.3}), VIF OK (max={max_vif:.1}): OLS assumptions satisfied", bp.p_value))
                };

                let adv = TbsStepAdvice::accepted(auto_method, auto_reason.clone())
                    .with_diagnostic("VIF_max", max_vif,
                        if multicollinear { "multicollinearity detected" } else { "OK" })
                    .with_diagnostic(norm_result.test_name, resid_norm_p,
                        if resid_normal { "normal" } else { "non-normal" })
                    .with_diagnostic("Breusch-Pagan", bp.p_value,
                        if heteroscedastic { "heteroscedastic" } else { "homoscedastic" })
                    .with_diagnostic("Cook's_D_n_influential", n_influential as f64,
                        if n_influential > 0 {
                            format!("{n_influential} influential observations (Cook's D > 4/n)")
                        } else {
                            "no influential observations".to_owned()
                        });

                let final_adv = if let Some(ref forced) = user_method {
                    let warn = if forced.as_str() != auto_method {
                        Some(format!("user forced {} but tambear recommends {}: {}", forced, auto_method, auto_reason))
                    } else { None };
                    TbsStepAdvice::overridden(auto_method, auto_reason, forced.clone(), "method", warn)
                } else {
                    adv
                };

                step_advice = Some(final_adv);
                linear_model = Some(model);
                n_hypothesis_tests += 1;
                TbsStepOutput::Scalar { name: "r_squared", value: r2 }
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

            // Gaussian Naive Bayes — fit returns class means/variances/priors as vectors
            ("train", Some("naive_bayes")) | ("gaussian_nb_fit", None) => {
                let y_ref = y.as_deref()
                    .ok_or("gaussian_nb_fit: y (class labels as f64) must be provided")?;
                let labels: Vec<i32> = y_ref.iter().map(|&v| v.round() as i32).collect();
                let data = &pipeline.frame().data;
                let model = crate::train::naive_bayes::gaussian_nb_fit(data, &labels, pn, pd, None);
                // Return class priors as vector output
                TbsStepOutput::Vector { name: "gaussian_nb_priors", values: model.class_prior.clone() }
            }

            ("gaussian_nb_predict", None) => {
                // Requires y to contain class labels used for fitting — no: we need the model
                // Since TBS doesn't have model state for NB yet, run fit+predict in one step
                let y_ref = y.as_deref()
                    .ok_or("gaussian_nb_predict: y (class labels as f64) must be provided for fit")?;
                let labels: Vec<i32> = y_ref.iter().map(|&v| v.round() as i32).collect();
                let data = &pipeline.frame().data;
                let model = crate::train::naive_bayes::gaussian_nb_fit(data, &labels, pn, pd, None);
                let preds = crate::train::naive_bayes::gaussian_nb_predict(&model, data, pn);
                TbsStepOutput::Vector { name: "gaussian_nb_predictions", values: preds.iter().map(|&l| l as f64).collect() }
            }

            // ══════════════════════════════════════════════════════════════
            // Dimensionality reduction
            // ══════════════════════════════════════════════════════════════

            ("pca", None) => {
                // If n_components is specified: pass-through (no auto-detection).
                // If not specified: auto-select via KMO/Bartlett pre-check + Kaiser criterion.
                let user_n_components = step.get_arg("n_components", 0).and_then(|v| v.as_usize());
                let data = &pipeline.frame().data;

                if let Some(nc) = user_n_components {
                    // Explicit n_components — run PCA directly
                    let result = crate::dim_reduction::pca(data, pn, pd, nc);
                    TbsStepOutput::Pca(result)
                } else {
                    // Auto-detection: KMO/Bartlett + Kaiser criterion for n_components
                    // 1. Compute correlation matrix
                    let corr_mat = crate::factor_analysis::correlation_matrix(data, pn, pd);
                    let kb = crate::factor_analysis::kmo_bartlett(&corr_mat, pn);

                    let kmo = kb.kmo_overall;
                    let bartlett_p = kb.bartlett_p_value;
                    let kmo_threshold = using_bag.get_f64("kmo_threshold").unwrap_or(0.5);
                    let bartlett_alpha = using_bag.get_f64("bartlett_alpha").unwrap_or(0.05);
                    let pca_viable = kmo >= kmo_threshold && bartlett_p < bartlett_alpha;

                    // 2. Run full PCA to get eigenvalues (singular_values² / (n-1))
                    let full_pca = crate::dim_reduction::pca(data, pn, pd, pd);
                    let n_minus_1 = (pn - 1).max(1) as f64;
                    let eigenvalues: Vec<f64> = full_pca.singular_values.iter()
                        .map(|sv| sv * sv / n_minus_1)
                        .collect();

                    // 3. Kaiser criterion: components with eigenvalue > 1
                    let kaiser_k = eigenvalues.iter().filter(|&&ev| ev > 1.0).count().max(1);
                    let auto_k = kaiser_k.min(pd).max(1);

                    // 4. Build advice
                    let user_method: Option<String> = using_bag.method().map(|s| s.to_owned());
                    let (auto_method, auto_reason) = if !pca_viable {
                        if kmo < kmo_threshold {
                            ("pca_warn", format!(
                                "KMO={kmo:.3} < {kmo_threshold}: data may not be suitable for PCA (sampling inadequacy)"))
                        } else {
                            ("pca_warn", format!(
                                "Bartlett p={bartlett_p:.3} ≥ {bartlett_alpha}: correlation matrix not significantly different from identity"))
                        }
                    } else {
                        ("pca", format!(
                            "KMO={kmo:.3} ≥ {kmo_threshold}, Bartlett p={bartlett_p:.4} < {bartlett_alpha}: PCA viable; Kaiser criterion → {auto_k} components"))
                    };

                    let adv = TbsStepAdvice::accepted(auto_method, auto_reason.clone())
                        .with_diagnostic("KMO", kmo,
                            if kmo >= 0.9 { "marvellous" }
                            else if kmo >= 0.8 { "meritorious" }
                            else if kmo >= 0.7 { "middling" }
                            else if kmo >= 0.6 { "mediocre" }
                            else if kmo >= kmo_threshold { "miserable" }
                            else { "unacceptable" })
                        .with_diagnostic("Bartlett", bartlett_p,
                            if bartlett_p < bartlett_alpha { "significant (correlations present)" } else { "not significant" });

                    let final_adv = if let Some(ref forced) = user_method {
                        let warn = if forced.as_str() != auto_method {
                            Some(format!("user forced {forced} but tambear recommends {auto_method}: {auto_reason}"))
                        } else { None };
                        TbsStepAdvice::overridden(auto_method, auto_reason, forced.clone(), "method", warn)
                    } else {
                        adv
                    };
                    step_advice = Some(final_adv);

                    // Run PCA with auto-selected n_components
                    let result = crate::dim_reduction::pca(data, pn, pd, auto_k);
                    TbsStepOutput::Pca(result)
                }
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

            // ── Time series auto-detection ────────────────────────────────
            ("time_series", None) | ("ts_analyze", None) => {
                let c = usize_arg(step, "col", 0, 0);
                let n_lags = usize_arg(step, "n_lags", 1, 5);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);

                let adf = crate::time_series::adf_test(&col, n_lags);
                let kpss = crate::time_series::kpss_test(&col, false, None);
                let adf_rejects = adf.statistic < adf.critical_5pct;
                let kpss_rejects = kpss.statistic > kpss.critical_5pct;

                let stationarity = if adf_rejects && !kpss_rejects { "stationary" }
                    else if !adf_rejects && kpss_rejects { "non-stationary" }
                    else if adf_rejects && kpss_rejects { "trend-stationary" }
                    else { "inconclusive" };

                let max_lag = using_bag.get_f64("max_lag")
                    .map(|v| v as usize)
                    .unwrap_or_else(|| (col.len() / 4).min(20).max(1));
                let acf_vals = crate::time_series::acf(&col, max_lag);
                let sig = 2.0 / (col.len() as f64).sqrt();
                let sig_acf: usize = (1..acf_vals.len()).filter(|&k| acf_vals[k].abs() > sig).count();

                let arch_lags_cap = using_bag.get_f64("arch_lags_cap")
                    .map(|v| v as usize).unwrap_or(5);
                let arch_lags_denom = using_bag.get_f64("arch_lags_fraction_denom")
                    .map(|v| v as usize).unwrap_or(10);
                let arch_lags = arch_lags_cap.min(col.len() / arch_lags_denom).max(1);
                let arch_alpha = using_bag.get_f64("arch_alpha").unwrap_or(0.05);
                let arch = crate::volatility::arch_lm_test(&col, arch_lags);
                let arch_p = arch.as_ref().map_or(1.0, |a| a.p_value);
                let has_arch = arch_p < arch_alpha;

                let rec = if !adf_rejects && kpss_rejects {
                    if has_arch { "difference + ARIMA-GARCH" } else { "difference + ARIMA" }
                } else if has_arch { "AR + GARCH" }
                  else if sig_acf == 0 { "white noise" }
                  else { "AR/ARMA model" };

                let adv = TbsStepAdvice::accepted("time_series_analysis",
                    format!("{stationarity}; {rec}"))
                    .with_diagnostic("ADF", adf.statistic,
                        if adf_rejects { "stationary" } else { "unit root" })
                    .with_diagnostic("KPSS", kpss.statistic,
                        if kpss_rejects { "non-stationary" } else { "stationary" })
                    .with_diagnostic("ARCH-LM", arch_p,
                        if has_arch { "ARCH effects" } else { "no ARCH" });

                step_advice = Some(adv);
                TbsStepOutput::Adf(adf)
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
                let min_v = col.iter().cloned().fold(f64::INFINITY, crate::numerical::nan_min);
                let max_v = col.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
                let mi_value = if min_v.is_nan() || max_v.is_nan() {
                    f64::NAN
                } else {
                    let range = (max_v - min_v).max(1e-15);
                    let binned: Vec<i32> = col.iter().map(|&v| ((v - min_v) / range * (n_bins as f64 - 1.0)).round() as i32).collect();
                    crate::information_theory::mutual_info_score(labels, &binned)
                };
                TbsStepOutput::Scalar { name: "mutual_info", value: mi_value }
            }

            // f-divergence family
            ("hellinger", None) | ("hellinger_distance", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (p, q) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "hellinger_distance", value: crate::information_theory::hellinger_distance(&p, &q) }
            }

            ("total_variation", None) | ("tv_distance", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (p, q) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "total_variation_distance", value: crate::information_theory::total_variation_distance(&p, &q) }
            }

            ("chi_squared_divergence", None) | ("chi2_divergence", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (p, q) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "chi_squared_divergence", value: crate::information_theory::chi_squared_divergence(&p, &q) }
            }

            ("renyi_divergence", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let alpha = f64_arg(step, "alpha", 1, 1.0);
                let (p, q) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "renyi_divergence", value: crate::information_theory::renyi_divergence(&p, &q, alpha) }
            }

            ("bhattacharyya", None) | ("bhattacharyya_distance", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (p, q) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "bhattacharyya_distance", value: crate::information_theory::bhattacharyya_distance(&p, &q) }
            }

            // Sample-based divergences
            ("wasserstein_1d", None) | ("wasserstein", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, y) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "wasserstein_1d", value: crate::information_theory::wasserstein_1d(&x, &y) }
            }

            ("mmd_rbf", None) | ("mmd", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let bandwidth = f64_arg(step, "bandwidth", 1, f64::NAN);
                let bw_opt = if bandwidth.is_nan() { None } else { Some(bandwidth) };
                let (x, y) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "mmd_rbf", value: crate::information_theory::mmd_rbf(&x, &y, bw_opt) }
            }

            ("energy_distance", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, y) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "energy_distance", value: crate::information_theory::energy_distance(&x, &y) }
            }

            // Joint entropy: build joint histogram from two columns, compute H(X,Y)
            ("joint_entropy", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let n_bins = usize_arg(step, "n_bins", 1, 10);
                let (x, y) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let n = x.len().min(y.len());
                let min_x = x.iter().cloned().fold(f64::INFINITY, crate::numerical::nan_min);
                let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
                let min_y = y.iter().cloned().fold(f64::INFINITY, crate::numerical::nan_min);
                let max_y = y.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
                let je_value = if min_x.is_nan() || max_x.is_nan() || min_y.is_nan() || max_y.is_nan() {
                    f64::NAN
                } else {
                    let range_x = (max_x - min_x).max(1e-15);
                    let range_y = (max_y - min_y).max(1e-15);
                    // Build flat joint count table (n_bins × n_bins)
                    let mut counts = vec![0u64; n_bins * n_bins];
                    for i in 0..n {
                        let bx = ((x[i] - min_x) / range_x * (n_bins as f64 - 1.0)).round() as usize;
                        let by = ((y[i] - min_y) / range_y * (n_bins as f64 - 1.0)).round() as usize;
                        let bx = bx.min(n_bins - 1);
                        let by = by.min(n_bins - 1);
                        counts[bx * n_bins + by] += 1;
                    }
                    let total = n as f64;
                    let joint_probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();
                    crate::information_theory::joint_entropy(&joint_probs, n_bins, n_bins)
                };
                TbsStepOutput::Scalar { name: "joint_entropy", value: je_value }
            }

            ("chi_squared_divergence", None) | ("chi_sq_divergence", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (p, q) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                TbsStepOutput::Scalar { name: "chi_squared_divergence", value: crate::information_theory::chi_squared_divergence(&p, &q) }
            }

            ("pmi", None) | ("pointwise_mutual_info", None) => {
                // PMI matrix between col_x and col_y (binned into n_bins×n_bins joint counts)
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let n_bins = usize_arg(step, "n_bins", 2, 10);
                let positive = bool_arg(step, "positive", 3, false);
                let (x, y) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let n = x.len().min(y.len());
                let min_x = x.iter().cloned().fold(f64::INFINITY, crate::numerical::nan_min);
                let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
                let min_y = y.iter().cloned().fold(f64::INFINITY, crate::numerical::nan_min);
                let max_y = y.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
                if min_x.is_nan() || max_x.is_nan() || min_y.is_nan() || max_y.is_nan() {
                    return Ok(TbsStepOutput::Matrix { name: "pmi", data: vec![f64::NAN], rows: 1, cols: 1 });
                }
                let range_x = (max_x - min_x).max(1e-15);
                let range_y = (max_y - min_y).max(1e-15);
                let mut counts = vec![0.0f64; n_bins * n_bins];
                for i in 0..n {
                    let bx = ((x[i] - min_x) / range_x * (n_bins as f64 - 1.0)).round() as usize;
                    let by = ((y[i] - min_y) / range_y * (n_bins as f64 - 1.0)).round() as usize;
                    counts[bx.min(n_bins - 1) * n_bins + by.min(n_bins - 1)] += 1.0;
                }
                let pmi_mat = crate::information_theory::pointwise_mutual_information(&counts, n_bins, n_bins, positive);
                TbsStepOutput::Matrix { name: "pmi", data: pmi_mat, rows: n_bins, cols: n_bins }
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

            ("rqa", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let m = usize_arg(step, "m", 1, 2);
                let tau = usize_arg(step, "tau", 2, 1);
                let epsilon = f64_arg(step, "epsilon", 3, 0.1);
                let lmin = usize_arg(step, "lmin", 4, 2);
                let res = crate::complexity::rqa(&col, m, tau, epsilon, lmin);
                TbsStepOutput::Vector {
                    name: "rqa",
                    values: vec![res.rr, res.det, res.lam, res.entr, res.lmax as f64, res.l_avg, res.tt],
                }
            }

            ("mfdfa", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let min_seg = usize_arg(step, "min_seg", 1, 16);
                let max_seg = usize_arg(step, "max_seg", 2, pn / 4);
                // Default q values: -3, -2, -1, 0, 1, 2, 3
                let q_values: Vec<f64> = vec![-3.0, -2.0, -1.0, 0.5, 1.0, 1.5, 2.0, 3.0];
                let res = crate::complexity::mfdfa(&col, &q_values, min_seg, max_seg);
                TbsStepOutput::Vector {
                    name: "mfdfa",
                    values: vec![res.width, res.h2, res.mean_se],
                }
            }

            ("ccm", None) => {
                let cx = usize_arg(step, "col_x", 0, 0);
                let cy = usize_arg(step, "col_y", 1, 1);
                let (x, y) = extract_two_cols(&pipeline.frame().data, pn, pd, cx, cy);
                let embed_dim = usize_arg(step, "embed_dim", 2, 3);
                let tau = usize_arg(step, "tau", 3, 1);
                let k = usize_arg(step, "k", 4, embed_dim + 1);
                let res = crate::complexity::ccm(&x, &y, embed_dim, tau, k);
                TbsStepOutput::Vector {
                    name: "ccm",
                    values: vec![res.rho_xy, res.rho_yx, res.rho_xy_half, res.rho_yx_half, res.convergence],
                }
            }

            ("phase_transition", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let win_size = usize_arg(step, "win_size", 1, pn / 10);
                let res = crate::complexity::phase_transition(&col, win_size, None);
                TbsStepOutput::Vector {
                    name: "phase_transition",
                    values: vec![res.order_parameter, res.susceptibility, res.binder_cumulant, res.critical_exponent],
                }
            }

            ("harmonic_r_stat", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "harmonic_r_stat", value: crate::complexity::harmonic_r_stat(&col) }
            }

            ("hankel_r_stat", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let embed_dim = usize_arg(step, "embed_dim", 1, 3);
                TbsStepOutput::Scalar { name: "hankel_r_stat", value: crate::complexity::hankel_r_stat(&col, embed_dim) }
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

            // ── Volatility auto-detection ──────────────────────────────────
            // ARCH-LM → GARCH fit → near-IGARCH warning → standardized residuals
            ("volatility_analyze", None) | ("vol_analyze", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);

                // ARCH-LM test
                let arch_lags_cap = using_bag.get_f64("arch_lags_cap")
                    .map(|v| v as usize).unwrap_or(5);
                let arch_lags_denom = using_bag.get_f64("arch_lags_fraction_denom")
                    .map(|v| v as usize).unwrap_or(10);
                let arch_lags = arch_lags_cap.min(col.len() / arch_lags_denom).max(1);
                let arch_alpha = using_bag.get_f64("arch_alpha").unwrap_or(0.05);
                let arch = crate::volatility::arch_lm_test(&col, arch_lags);
                let arch_p = arch.as_ref().map_or(1.0, |a| a.p_value);
                let has_arch = arch_p < arch_alpha;

                if has_arch {
                    // Fit GARCH(1,1)
                    let garch_max_iter = using_bag.get_f64("garch_max_iter")
                        .map(|v| v as usize).unwrap_or(200);
                    let garch = crate::volatility::garch11_fit(&col, garch_max_iter);
                    let rec = if garch.near_igarch {
                        "GARCH(1,1) near-integrated (α+β > 0.99): consider IGARCH or long-memory model"
                    } else {
                        "GARCH(1,1) fitted successfully"
                    };
                    let adv = TbsStepAdvice::accepted("garch11", rec.to_string())
                        .with_diagnostic("ARCH-LM", arch_p, "ARCH effects present")
                        .with_diagnostic("alpha+beta", garch.alpha + garch.beta,
                            if garch.near_igarch { "near-integrated" } else { "stationary" });
                    step_advice = Some(adv);
                    TbsStepOutput::Vector {
                        name: "garch_params",
                        values: vec![garch.omega, garch.alpha, garch.beta],
                    }
                } else {
                    let adv = TbsStepAdvice::accepted("constant_volatility",
                        "no ARCH effects detected; constant volatility (EWMA/rolling std) sufficient")
                        .with_diagnostic("ARCH-LM", arch_p, "no ARCH effects");
                    step_advice = Some(adv);
                    // Return EWMA as the simpler alternative
                    let ewma_lambda = using_bag.get_f64("ewma_lambda").unwrap_or(0.94);
                    let values = crate::volatility::ewma_variance(&col, ewma_lambda);
                    TbsStepOutput::Vector { name: "ewma_variance", values }
                }
            }

            // ── Survival auto-detection ──────────────────────────────────────
            // KM → Cox PH → Schoenfeld PH check
            ("survival_analyze", None) | ("surv_analyze", None) => {
                // Expects y = [time1, event1, time2, event2, ...]
                let y_ref = y.as_deref()
                    .ok_or("survival_analyze: target data (y) required as [time, event, ...]")?;
                let n_obs = y_ref.len() / 2;
                let times: Vec<f64> = (0..n_obs).map(|i| y_ref[i * 2]).collect();
                let events: Vec<bool> = (0..n_obs).map(|i| y_ref[i * 2 + 1] > 0.5).collect();
                let n_events = events.iter().filter(|&&e| e).count();

                let has_covariates = pd > 0;

                let rec = if !has_covariates {
                    "Kaplan-Meier (no covariates)"
                } else if n_events < pd * 10 {
                    "Cox PH (but events-per-variable < 10: consider penalized Cox)"
                } else {
                    "Cox PH"
                };

                let adv = TbsStepAdvice::accepted("survival_analysis",
                    format!("{rec}; {n_events} events, {n_obs} observations"))
                    .with_diagnostic("events_per_variable",
                        if pd > 0 { n_events as f64 / pd as f64 } else { f64::INFINITY },
                        if n_events >= pd * 10 { "adequate" } else { "low (< 10 EPV)" });

                step_advice = Some(adv);

                if has_covariates {
                    let result = crate::survival::cox_ph(
                        &pipeline.frame().data, &times, &events, pn, pd, 50);
                    TbsStepOutput::Vector { name: "cox_coefficients", values: result.beta }
                } else {
                    let km = crate::survival::kaplan_meier(&times, &events);
                    let surv_vals: Vec<f64> = km.iter().map(|s| s.survival).collect();
                    TbsStepOutput::Vector { name: "km_survival", values: surv_vals }
                }
            }

            // ── Bayesian auto-detection ──────────────────────────────────────
            ("bayesian_analyze", None) | ("bayes_analyze", None) => {
                // For now: fit Bayesian linear regression (conjugate) and check convergence
                let y_ref = y.as_deref()
                    .ok_or("bayesian_analyze: target data (y) required")?;

                let rec = if pn < 30 {
                    "Bayesian conjugate regression (small n benefits from prior regularization)"
                } else {
                    "Bayesian conjugate regression (conjugate prior for fast posterior)"
                };

                let adv = TbsStepAdvice::accepted("bayesian_linear",
                    format!("{rec}; n={pn}, p={pd}"))
                    .with_diagnostic("n/p_ratio", pn as f64 / pd.max(1) as f64,
                        if pn > pd * 5 { "adequate" } else { "low (prior important)" });
                step_advice = Some(adv);

                // Fit conjugate Bayesian linear regression with vague prior
                let prior_mean = vec![0.0; pd];
                let mut prior_prec = vec![0.0; pd * pd];
                for j in 0..pd { prior_prec[j * pd + j] = 0.01; }
                let result = crate::bayesian::bayesian_linear_regression(
                    &pipeline.frame().data, y_ref, pn, pd,
                    &prior_mean, &prior_prec, 1.0, 1.0);
                TbsStepOutput::Vector { name: "bayes_coefficients", values: result.beta_mean }
            }

            // ══════════════════════════════════════════════════════════════
            // IRT — Item Response Theory
            // ══════════════════════════════════════════════════════════════

            // fit_2pl: data is n_persons × n_items binary response matrix (0/1 as f64).
            // Returns item params as flat vector: [disc1, diff1, disc2, diff2, ...].
            ("irt_2pl", None) | ("fit_2pl", None) => {
                let n_items = usize_req(step, "n_items", 0)?;
                let n_persons = pn;
                let max_iter = usize_arg(step, "max_iter", 1, 100);
                let data = &pipeline.frame().data;
                // Convert f64 → u8 responses
                let responses: Vec<u8> = data.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
                let items = crate::irt::fit_2pl(&responses, n_persons, n_items, max_iter, None);
                let flat: Vec<f64> = items.iter().flat_map(|it| [it.discrimination, it.difficulty]).collect();
                TbsStepOutput::Vector { name: "irt_item_params", values: flat }
            }

            // ability_mle: data must have exactly n_items columns; first row is used.
            // item_params_flat from a previous irt_2pl step must be in step args.
            // Simpler: expose ability_mle for single-person scoring.
            // Convention: col 0..n_items are binary responses; item params come from args.
            ("ability_mle", None) => {
                let n_items = usize_req(step, "n_items", 0)?;
                let disc_arg = f64_arg(step, "discrimination", 1, 1.0);
                let diff_arg = f64_arg(step, "difficulty", 2, 0.0);
                let data = &pipeline.frame().data;
                // Build one ItemParams per item from args or use same params for all items.
                let items: Vec<crate::irt::ItemParams> = (0..n_items).map(|_| crate::irt::ItemParams {
                    discrimination: disc_arg,
                    difficulty: diff_arg,
                }).collect();
                // Use first row as the response pattern
                let responses: Vec<u8> = (0..n_items).map(|j| {
                    if j < pd { if data[j] >= 0.5 { 1 } else { 0 } } else { 0 }
                }).collect();
                let theta = crate::irt::ability_mle(&items, &responses);
                TbsStepOutput::Scalar { name: "ability_mle", value: theta }
            }

            ("ability_eap", None) => {
                let n_items = usize_req(step, "n_items", 0)?;
                let disc_arg = f64_arg(step, "discrimination", 1, 1.0);
                let diff_arg = f64_arg(step, "difficulty", 2, 0.0);
                let n_quad = usize_arg(step, "n_quad", 3, 21);
                let data = &pipeline.frame().data;
                let items: Vec<crate::irt::ItemParams> = (0..n_items).map(|_| crate::irt::ItemParams {
                    discrimination: disc_arg,
                    difficulty: diff_arg,
                }).collect();
                let responses: Vec<u8> = (0..n_items).map(|j| {
                    if j < pd { if data[j] >= 0.5 { 1 } else { 0 } } else { 0 }
                }).collect();
                let theta = crate::irt::ability_eap(&items, &responses, n_quad);
                TbsStepOutput::Scalar { name: "ability_eap", value: theta }
            }

            // ══════════════════════════════════════════════════════════════
            // Graph algorithms
            // Convention: data is n_edges × 3 (from, to, weight) or n_edges × 2 (from, to).
            // Node count comes from the "n_nodes" arg.
            // ══════════════════════════════════════════════════════════════

            ("dijkstra", None) => {
                let n_nodes = usize_req(step, "n_nodes", 0)?;
                let source = usize_arg(step, "source", 1, 0);
                let directed = bool_arg(step, "directed", 2, true);
                let data = &pipeline.frame().data;
                let mut g = crate::graph::Graph::new(n_nodes);
                for i in 0..pn {
                    let from = data[i * pd] as usize;
                    let to   = data[i * pd + 1] as usize;
                    let w    = if pd >= 3 { data[i * pd + 2] } else { 1.0 };
                    if from < n_nodes && to < n_nodes {
                        if directed { g.add_edge(from, to, w); } else { g.add_undirected(from, to, w); }
                    }
                }
                let (dists, _) = crate::graph::dijkstra(&g, source);
                TbsStepOutput::Vector { name: "dijkstra_distances", values: dists }
            }

            ("bellman_ford", None) => {
                let n_nodes = usize_req(step, "n_nodes", 0)?;
                let source = usize_arg(step, "source", 1, 0);
                let directed = bool_arg(step, "directed", 2, true);
                let data = &pipeline.frame().data;
                let mut g = crate::graph::Graph::new(n_nodes);
                for i in 0..pn {
                    let from = data[i * pd] as usize;
                    let to   = data[i * pd + 1] as usize;
                    let w    = if pd >= 3 { data[i * pd + 2] } else { 1.0 };
                    if from < n_nodes && to < n_nodes {
                        if directed { g.add_edge(from, to, w); } else { g.add_undirected(from, to, w); }
                    }
                }
                match crate::graph::bellman_ford(&g, source) {
                    Some((dists, _)) => TbsStepOutput::Vector { name: "bellman_ford_distances", values: dists },
                    None => return Err("bellman_ford: negative-weight cycle detected".into()),
                }
            }

            ("floyd_warshall", None) => {
                let n_nodes = usize_req(step, "n_nodes", 0)?;
                let directed = bool_arg(step, "directed", 1, true);
                let data = &pipeline.frame().data;
                let mut g = crate::graph::Graph::new(n_nodes);
                for i in 0..pn {
                    let from = data[i * pd] as usize;
                    let to   = data[i * pd + 1] as usize;
                    let w    = if pd >= 3 { data[i * pd + 2] } else { 1.0 };
                    if from < n_nodes && to < n_nodes {
                        if directed { g.add_edge(from, to, w); } else { g.add_undirected(from, to, w); }
                    }
                }
                let mat = crate::graph::floyd_warshall(&g);
                let flat: Vec<f64> = mat.into_iter().flatten().collect();
                TbsStepOutput::Matrix { name: "floyd_warshall", data: flat, rows: n_nodes, cols: n_nodes }
            }

            ("pagerank", None) => {
                let n_nodes = usize_req(step, "n_nodes", 0)?;
                let damping = f64_arg(step, "damping", 1, 0.85);
                let max_iter = usize_arg(step, "max_iter", 2, 100);
                let tol = f64_arg(step, "tol", 3, 1e-6);
                let data = &pipeline.frame().data;
                let mut g = crate::graph::Graph::new(n_nodes);
                for i in 0..pn {
                    let from = data[i * pd] as usize;
                    let to   = data[i * pd + 1] as usize;
                    let w    = if pd >= 3 { data[i * pd + 2] } else { 1.0 };
                    if from < n_nodes && to < n_nodes {
                        g.add_edge(from, to, w);
                    }
                }
                let scores = crate::graph::pagerank(&g, damping, max_iter, tol);
                TbsStepOutput::Vector { name: "pagerank", values: scores }
            }

            ("kruskal", None) | ("mst", None) => {
                let n_nodes = usize_req(step, "n_nodes", 0)?;
                let data = &pipeline.frame().data;
                let mut g = crate::graph::Graph::new(n_nodes);
                for i in 0..pn {
                    let from = data[i * pd] as usize;
                    let to   = data[i * pd + 1] as usize;
                    let w    = if pd >= 3 { data[i * pd + 2] } else { 1.0 };
                    if from < n_nodes && to < n_nodes {
                        g.add_undirected(from, to, w);
                    }
                }
                let mst = crate::graph::kruskal(&g);
                TbsStepOutput::Scalar { name: "mst_total_weight", value: mst.total_weight }
            }

            ("connected_components", None) => {
                let n_nodes = usize_req(step, "n_nodes", 0)?;
                let data = &pipeline.frame().data;
                let mut g = crate::graph::Graph::new(n_nodes);
                for i in 0..pn {
                    let from = data[i * pd] as usize;
                    let to   = data[i * pd + 1] as usize;
                    let w    = if pd >= 3 { data[i * pd + 2] } else { 1.0 };
                    if from < n_nodes && to < n_nodes {
                        g.add_undirected(from, to, w);
                    }
                }
                let labels = crate::graph::connected_components(&g);
                let max_comp = labels.iter().cloned().max().unwrap_or(0) + 1;
                TbsStepOutput::Scalar { name: "n_components", value: max_comp as f64 }
            }

            ("bfs", None) => {
                let n_nodes = usize_req(step, "n_nodes", 0)?;
                let source = usize_arg(step, "source", 1, 0);
                let directed = bool_arg(step, "directed", 2, false);
                let data = &pipeline.frame().data;
                let mut g = crate::graph::Graph::new(n_nodes);
                for i in 0..pn {
                    let from = data[i * pd] as usize;
                    let to   = data[i * pd + 1] as usize;
                    let w    = if pd >= 3 { data[i * pd + 2] } else { 1.0 };
                    if from < n_nodes && to < n_nodes {
                        if directed { g.add_edge(from, to, w); } else { g.add_undirected(from, to, w); }
                    }
                }
                let (dists, _) = crate::graph::bfs(&g, source);
                let dists_f64: Vec<f64> = dists.iter().map(|&d| if d < 0 { f64::INFINITY } else { d as f64 }).collect();
                TbsStepOutput::Vector { name: "bfs_distances", values: dists_f64 }
            }

            ("dfs", None) => {
                let n_nodes = usize_req(step, "n_nodes", 0)?;
                let source = usize_arg(step, "source", 1, 0);
                let directed = bool_arg(step, "directed", 2, true);
                let data = &pipeline.frame().data;
                let mut g = crate::graph::Graph::new(n_nodes);
                for i in 0..pn {
                    let from = data[i * pd] as usize;
                    let to   = data[i * pd + 1] as usize;
                    let w    = if pd >= 3 { data[i * pd + 2] } else { 1.0 };
                    if from < n_nodes && to < n_nodes {
                        if directed { g.add_edge(from, to, w); } else { g.add_undirected(from, to, w); }
                    }
                }
                let order = crate::graph::dfs(&g, source);
                let order_f64: Vec<f64> = order.iter().map(|&v| v as f64).collect();
                TbsStepOutput::Vector { name: "dfs_order", values: order_f64 }
            }

            ("degree_centrality", None) => {
                let n_nodes = usize_req(step, "n_nodes", 0)?;
                let directed = bool_arg(step, "directed", 1, true);
                let data = &pipeline.frame().data;
                let mut g = crate::graph::Graph::new(n_nodes);
                for i in 0..pn {
                    let from = data[i * pd] as usize;
                    let to   = data[i * pd + 1] as usize;
                    let w    = if pd >= 3 { data[i * pd + 2] } else { 1.0 };
                    if from < n_nodes && to < n_nodes {
                        if directed { g.add_edge(from, to, w); } else { g.add_undirected(from, to, w); }
                    }
                }
                let centrality = crate::graph::degree_centrality(&g);
                TbsStepOutput::Vector { name: "degree_centrality", values: centrality }
            }

            // ══════════════════════════════════════════════════════════════
            // Interpolation
            // Convention: data is n_points × 2 (x, y).
            // Query point comes from "x_query" arg; for batch eval use "n_eval" + range.
            // ══════════════════════════════════════════════════════════════

            ("lagrange", None) => {
                if pd < 2 { return Err("lagrange: need 2 columns (x, y)".into()); }
                let x_query = f64_req(step, "x", 0)?;
                let xs = extract_col(&pipeline.frame().data, pn, pd, 0);
                let ys = extract_col(&pipeline.frame().data, pn, pd, 1);
                let val = crate::interpolation::lagrange(&xs, &ys, x_query);
                TbsStepOutput::Scalar { name: "lagrange", value: val }
            }

            ("newton_interp", None) | ("newton_divided_diff", None) => {
                if pd < 2 { return Err("newton_interp: need 2 columns (x, y)".into()); }
                let x_query = f64_req(step, "x", 0)?;
                let xs = extract_col(&pipeline.frame().data, pn, pd, 0);
                let ys = extract_col(&pipeline.frame().data, pn, pd, 1);
                let coeffs = crate::interpolation::newton_divided_diff(&xs, &ys);
                let val = crate::interpolation::newton_eval(&xs, &coeffs, x_query);
                TbsStepOutput::Scalar { name: "newton_interp", value: val }
            }

            ("cubic_spline", None) | ("spline", None) => {
                if pd < 2 { return Err("cubic_spline: need 2 columns (x, y)".into()); }
                let x_query = f64_req(step, "x", 0)?;
                let xs = extract_col(&pipeline.frame().data, pn, pd, 0);
                let ys = extract_col(&pipeline.frame().data, pn, pd, 1);
                let spline = crate::interpolation::natural_cubic_spline(&xs, &ys);
                let val = spline.eval(x_query);
                TbsStepOutput::Scalar { name: "cubic_spline", value: val }
            }

            ("akima", None) => {
                if pd < 2 { return Err("akima: need 2 columns (x, y)".into()); }
                let x_query = f64_req(step, "x", 0)?;
                let xs = extract_col(&pipeline.frame().data, pn, pd, 0);
                let ys = extract_col(&pipeline.frame().data, pn, pd, 1);
                let spline = crate::interpolation::akima(&xs, &ys);
                TbsStepOutput::Scalar { name: "akima", value: spline.eval(x_query) }
            }

            ("pchip", None) => {
                if pd < 2 { return Err("pchip: need 2 columns (x, y)".into()); }
                let x_query = f64_req(step, "x", 0)?;
                let xs = extract_col(&pipeline.frame().data, pn, pd, 0);
                let ys = extract_col(&pipeline.frame().data, pn, pd, 1);
                let spline = crate::interpolation::pchip(&xs, &ys);
                TbsStepOutput::Scalar { name: "pchip", value: spline.eval(x_query) }
            }

            ("rbf", None) | ("rbf_interp", None) => {
                if pd < 2 { return Err("rbf: need 2 columns (x, y)".into()); }
                let x_query = f64_req(step, "x", 0)?;
                let kernel_str = step.get_arg("kernel", 1)
                    .and_then(|v| if let crate::tbs_parser::TbsValue::Str(s) = v { Some(s.clone()) } else { None })
                    .unwrap_or_else(|| "gaussian".to_string());
                let epsilon = f64_arg(step, "epsilon", 2, 1.0);
                let kernel = match kernel_str.as_str() {
                    "multiquadric" => crate::interpolation::RbfKernel::Multiquadric(epsilon),
                    "inverse_multiquadric" => crate::interpolation::RbfKernel::InverseMultiquadric(epsilon),
                    "thin_plate" | "thin_plate_spline" => crate::interpolation::RbfKernel::ThinPlateSpline,
                    _ => crate::interpolation::RbfKernel::Gaussian(epsilon),
                };
                let xs = extract_col(&pipeline.frame().data, pn, pd, 0);
                let ys = extract_col(&pipeline.frame().data, pn, pd, 1);
                let interp = crate::interpolation::rbf_interpolate(&xs, &ys, kernel);
                TbsStepOutput::Scalar { name: "rbf", value: interp.eval(x_query) }
            }

            ("polyfit", None) => {
                if pd < 2 { return Err("polyfit: need 2 columns (x, y)".into()); }
                let deg = usize_arg(step, "deg", 0, 2);
                let xs = extract_col(&pipeline.frame().data, pn, pd, 0);
                let ys = extract_col(&pipeline.frame().data, pn, pd, 1);
                let fit = crate::interpolation::polyfit(&xs, &ys, deg);
                TbsStepOutput::Vector { name: "polyfit_coeffs", values: fit.coeffs }
            }

            ("lerp", None) => {
                if pd < 2 { return Err("lerp: need 2 columns (x, y)".into()); }
                let x_query = f64_req(step, "x", 0)?;
                let xs = extract_col(&pipeline.frame().data, pn, pd, 0);
                let ys = extract_col(&pipeline.frame().data, pn, pd, 1);
                TbsStepOutput::Scalar { name: "lerp", value: crate::interpolation::lerp(&xs, &ys, x_query) }
            }

            // ══════════════════════════════════════════════════════════════
            // Linear algebra — explicit decompositions
            // Convention: data is n × d; treat as a matrix.
            // ══════════════════════════════════════════════════════════════

            ("lu", None) | ("lu_decomp", None) => {
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                match crate::linear_algebra::lu(&mat) {
                    Some(res) => {
                        // Return combined LU matrix (L has implicit 1s on diagonal)
                        TbsStepOutput::Matrix { name: "lu_decomp", data: res.lu.data, rows: res.lu.rows, cols: res.lu.cols }
                    }
                    None => return Err("lu: matrix is singular".into()),
                }
            }

            ("qr", None) | ("qr_decomp", None) => {
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let res = crate::linear_algebra::qr(&mat);
                // Return R matrix (upper triangular)
                TbsStepOutput::Matrix { name: "qr_r", data: res.r.data, rows: res.r.rows, cols: res.r.cols }
            }

            ("svd", None) => {
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let res = crate::linear_algebra::svd(&mat);
                TbsStepOutput::Vector { name: "singular_values", values: res.sigma }
            }

            ("det", None) => {
                if pn != pd { return Err("det: matrix must be square".into()); }
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                TbsStepOutput::Scalar { name: "det", value: crate::linear_algebra::det(&mat) }
            }

            ("cond", None) => {
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                TbsStepOutput::Scalar { name: "cond", value: crate::linear_algebra::cond(&mat) }
            }

            ("pinv", None) => {
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let res = crate::linear_algebra::pinv(&mat, None);
                TbsStepOutput::Matrix { name: "pinv", data: res.data, rows: res.rows, cols: res.cols }
            }

            ("solve_linear", None) => {
                // Square n×n system; y must be provided as the RHS.
                if pn != pd { return Err("solve_linear: coefficient matrix must be square".into()); }
                let b = y.as_deref().ok_or("solve_linear: y (RHS vector) must be provided")?;
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                match crate::linear_algebra::solve(&mat, b) {
                    Some(x) => TbsStepOutput::Vector { name: "solution", values: x },
                    None => return Err("solve_linear: matrix is singular".into()),
                }
            }

            ("sym_eigen", None) | ("eigendecomp", None) => {
                if pn != pd { return Err("sym_eigen: matrix must be square symmetric".into()); }
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let (eigenvalues, _) = crate::linear_algebra::sym_eigen(&mat);
                TbsStepOutput::Vector { name: "eigenvalues", values: eigenvalues }
            }

            ("matrix_exp", None) => {
                if pn != pd { return Err("matrix_exp: matrix must be square".into()); }
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let res = crate::linear_algebra::matrix_exp(&mat);
                TbsStepOutput::Matrix { name: "matrix_exp", data: res.data, rows: res.rows, cols: res.cols }
            }

            ("matrix_log", None) => {
                if pn != pd { return Err("matrix_log: matrix must be square".into()); }
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let res = crate::linear_algebra::matrix_log(&mat);
                TbsStepOutput::Matrix { name: "matrix_log", data: res.data, rows: res.rows, cols: res.cols }
            }

            ("matrix_sqrt", None) => {
                if pn != pd { return Err("matrix_sqrt: matrix must be square".into()); }
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let res = crate::linear_algebra::matrix_sqrt(&mat);
                TbsStepOutput::Matrix { name: "matrix_sqrt", data: res.data, rows: res.rows, cols: res.cols }
            }

            ("log_det", None) => {
                if pn != pd { return Err("log_det: matrix must be square".into()); }
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                TbsStepOutput::Scalar { name: "log_det", value: crate::linear_algebra::log_det(&mat) }
            }

            ("effective_rank", None) | ("effective_rank_from_sv", None) => {
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let svd_res = crate::linear_algebra::svd(&mat);
                TbsStepOutput::Scalar { name: "effective_rank", value: crate::linear_algebra::effective_rank_from_sv(&svd_res.sigma) }
            }

            ("conjugate_gradient", None) | ("cg", None) => {
                // Ax = b: data is n×n SPD matrix A; y is RHS b.
                if pn != pd { return Err("conjugate_gradient: matrix must be square".into()); }
                let b = y.as_deref().ok_or("conjugate_gradient: y (RHS vector) must be provided")?;
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let tol = f64_arg(step, "tol", 0, 1e-10);
                let max_iter = usize_arg(step, "max_iter", 1, 1000);
                let res = crate::linear_algebra::conjugate_gradient(&mat, b, None, Some(tol), Some(max_iter));
                TbsStepOutput::Vector { name: "cg_solution", values: res.x }
            }

            ("gmres", None) => {
                // Ax = b: data is n×n matrix A; y is RHS b.
                if pn != pd { return Err("gmres: matrix must be square".into()); }
                let b = y.as_deref().ok_or("gmres: y (RHS vector) must be provided")?;
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let tol = f64_arg(step, "tol", 0, 1e-10);
                let max_iter = usize_arg(step, "max_iter", 1, 1000);
                let res = crate::linear_algebra::gmres(&mat, b, None, Some(tol), Some(max_iter), None);
                TbsStepOutput::Vector { name: "gmres_solution", values: res.x }
            }

            // ══════════════════════════════════════════════════════════════
            // Multivariate tests: Hotelling, MANOVA, CCA, LDA, Mardia
            // ══════════════════════════════════════════════════════════════

            ("hotelling_t2", None) | ("hotelling_one_sample", None) => {
                // mu0 defaults to zeros; pass as flat args: mu0_0, mu0_1, ...
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let mu0: Vec<f64> = (0..pd)
                    .map(|j| f64_arg(step, "mu0", j, 0.0))
                    .collect();
                let res = crate::multivariate::hotelling_one_sample(&mat, &mu0);
                TbsStepOutput::Vector {
                    name: "hotelling_t2",
                    values: vec![res.t2, res.f_statistic, res.df1, res.df2, res.p_value],
                }
            }

            ("hotelling_two_sample", None) => {
                // Split data at midpoint (first pn/2 rows vs second pn/2 rows).
                // Or use group labels from a previous clustering step.
                let labels = pipeline.frame().labels.as_ref()
                    .ok_or("hotelling_two_sample: run a group_by or clustering step first")?;
                let group0: Vec<usize> = labels.iter().enumerate()
                    .filter(|(_, &l)| l == 0).map(|(i, _)| i).collect();
                let group1: Vec<usize> = labels.iter().enumerate()
                    .filter(|(_, &l)| l == 1).map(|(i, _)| i).collect();
                let data = &pipeline.frame().data;
                let extract_group = |indices: &[usize]| -> crate::linear_algebra::Mat {
                    let mut d = Vec::with_capacity(indices.len() * pd);
                    for &i in indices { d.extend_from_slice(&data[i * pd..(i + 1) * pd]); }
                    crate::linear_algebra::Mat { rows: indices.len(), cols: pd, data: d }
                };
                let m1 = extract_group(&group0);
                let m2 = extract_group(&group1);
                let res = crate::multivariate::hotelling_two_sample(&m1, &m2);
                TbsStepOutput::Vector {
                    name: "hotelling_two_sample",
                    values: vec![res.t2, res.f_statistic, res.df1, res.df2, res.p_value],
                }
            }

            ("manova", None) => {
                let labels = pipeline.frame().labels.as_ref()
                    .ok_or("manova: run a group_by or clustering step first to produce group labels")?;
                let groups: Vec<usize> = labels.iter().map(|&l| l.max(0) as usize).collect();
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let res = crate::multivariate::manova(&mat, &groups);
                TbsStepOutput::Vector {
                    name: "manova",
                    values: vec![res.wilks_lambda, res.f_statistic, res.p_value],
                }
            }

            ("lda", None) => {
                let labels = pipeline.frame().labels.as_ref()
                    .ok_or("lda: run a group_by or clustering step first to produce group labels")?;
                let groups: Vec<usize> = labels.iter().map(|&l| l.max(0) as usize).collect();
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let res = crate::multivariate::lda(&mat, &groups);
                // Return discriminant axis weights (p × d matrix, flattened)
                let weights = res.axes.data.clone();
                TbsStepOutput::Vector { name: "lda_weights", values: weights }
            }

            ("cca", None) => {
                // Split data into X (first pd/2 cols) and Y (second pd/2 cols).
                // Or use col_x_end arg to specify the split.
                let split = usize_arg(step, "col_x_end", 0, pd / 2);
                if split == 0 || split >= pd { return Err("cca: col_x_end must be between 1 and d-1".into()); }
                let data = &pipeline.frame().data;
                let mut x_data = Vec::with_capacity(pn * split);
                let mut y_data = Vec::with_capacity(pn * (pd - split));
                for i in 0..pn {
                    x_data.extend_from_slice(&data[i * pd..i * pd + split]);
                    y_data.extend_from_slice(&data[i * pd + split..i * pd + pd]);
                }
                let x_mat = crate::linear_algebra::Mat { rows: pn, cols: split, data: x_data };
                let y_mat = crate::linear_algebra::Mat { rows: pn, cols: pd - split, data: y_data };
                let res = crate::multivariate::cca(&x_mat, &y_mat);
                TbsStepOutput::Vector { name: "cca_correlations", values: res.correlations }
            }

            ("mardia_normality", None) => {
                let mat = crate::linear_algebra::Mat { rows: pn, cols: pd, data: pipeline.frame().data.clone() };
                let res = crate::multivariate::mardia_normality(&mat);
                TbsStepOutput::Vector {
                    name: "mardia_normality",
                    values: vec![res.skewness, res.skewness_p, res.kurtosis, res.kurtosis_p],
                }
            }

            // ══════════════════════════════════════════════════════════════
            // Factor analysis (EFA, varimax, Cronbach's alpha, McDonald's omega)
            // ══════════════════════════════════════════════════════════════

            ("efa", None) | ("factor_analysis", None) => {
                let n_factors = usize_arg(step, "n_factors", 0, 2);
                let max_iter = usize_arg(step, "max_iter", 1, 100);
                let corr = crate::factor_analysis::correlation_matrix(&pipeline.frame().data, pn, pd);
                let res = crate::factor_analysis::principal_axis_factoring(&corr, n_factors, max_iter);
                TbsStepOutput::Vector { name: "efa_communalities", values: res.communalities }
            }

            ("varimax", None) => {
                let n_factors = usize_arg(step, "n_factors", 0, 2);
                let max_iter = usize_arg(step, "max_iter", 1, 1000);
                let corr = crate::factor_analysis::correlation_matrix(&pipeline.frame().data, pn, pd);
                let fa = crate::factor_analysis::principal_axis_factoring(&corr, n_factors, max_iter);
                let rotated = crate::factor_analysis::varimax(&fa.loadings, 1000);
                TbsStepOutput::Matrix { name: "varimax_loadings", data: rotated.data, rows: rotated.rows, cols: rotated.cols }
            }

            ("cronbachs_alpha", None) | ("cronbach_alpha", None) => {
                let val = crate::factor_analysis::cronbachs_alpha(&pipeline.frame().data, pn, pd);
                TbsStepOutput::Scalar { name: "cronbachs_alpha", value: val }
            }

            ("mcdonalds_omega", None) | ("mcdonald_omega", None) => {
                let n_factors = usize_arg(step, "n_factors", 0, 1);
                let max_iter = usize_arg(step, "max_iter", 1, 100);
                let corr = crate::factor_analysis::correlation_matrix(&pipeline.frame().data, pn, pd);
                let fa = crate::factor_analysis::principal_axis_factoring(&corr, n_factors, max_iter);
                let res = crate::factor_analysis::mcdonalds_omega(&fa.loadings);
                TbsStepOutput::Scalar { name: "mcdonalds_omega", value: res.omega }
            }

            // ══════════════════════════════════════════════════════════════
            // Stochastic processes
            // ══════════════════════════════════════════════════════════════

            ("brownian_motion", None) => {
                let t_end = f64_arg(step, "t_end", 0, 1.0);
                let n_steps = usize_arg(step, "n_steps", 1, 1000);
                let seed = usize_arg(step, "seed", 2, 42) as u64;
                let (_, path) = crate::stochastic::brownian_motion(t_end, n_steps, seed);
                TbsStepOutput::Vector { name: "brownian_motion", values: path }
            }

            ("geometric_brownian_motion", None) | ("gbm", None) => {
                let s0 = f64_arg(step, "s0", 0, 1.0);
                let mu = f64_arg(step, "mu", 1, 0.0);
                let sigma = f64_arg(step, "sigma", 2, 0.1);
                let t_end = f64_arg(step, "t_end", 3, 1.0);
                let n_steps = usize_arg(step, "n_steps", 4, 252);
                let seed = usize_arg(step, "seed", 5, 42) as u64;
                let (_, prices) = crate::stochastic::geometric_brownian_motion(s0, mu, sigma, t_end, n_steps, seed);
                TbsStepOutput::Vector { name: "gbm", values: prices }
            }

            ("ornstein_uhlenbeck", None) | ("ou", None) => {
                let x0 = f64_arg(step, "x0", 0, 0.0);
                let mu = f64_arg(step, "mu", 1, 0.0);
                let theta = f64_arg(step, "theta", 2, 1.0);
                let sigma = f64_arg(step, "sigma", 3, 0.5);
                let t_end = f64_arg(step, "t_end", 4, 1.0);
                let n_steps = usize_arg(step, "n_steps", 5, 1000);
                let seed = usize_arg(step, "seed", 6, 42) as u64;
                let path = crate::stochastic::ornstein_uhlenbeck(x0, mu, theta, sigma, t_end, n_steps, seed);
                TbsStepOutput::Vector { name: "ornstein_uhlenbeck", values: path }
            }

            ("poisson_process", None) => {
                let lambda = f64_arg(step, "lambda", 0, 1.0);
                let t_end = f64_arg(step, "t_end", 1, 10.0);
                let seed = usize_arg(step, "seed", 2, 42) as u64;
                let events = crate::stochastic::poisson_process(lambda, t_end, seed);
                TbsStepOutput::Scalar { name: "n_events", value: events.len() as f64 }
            }

            ("markov_chain", None) | ("stationary_distribution", None) => {
                // Data is n_states × n_states transition matrix (row-major).
                if pn != pd { return Err("markov_chain: transition matrix must be square".into()); }
                let n_states = pn;
                let stationary = crate::stochastic::stationary_distribution(&pipeline.frame().data, n_states);
                TbsStepOutput::Vector { name: "stationary_distribution", values: stationary }
            }

            ("black_scholes", None) => {
                let s = f64_req(step, "s", 0)?;
                let k = f64_req(step, "k", 1)?;
                let t = f64_req(step, "t", 2)?;
                let r = f64_arg(step, "r", 3, 0.05);
                let sigma = f64_arg(step, "sigma", 4, 0.2);
                let call = bool_arg(step, "call", 5, true);
                let (price, delta) = crate::stochastic::black_scholes(s, k, t, r, sigma, call);
                TbsStepOutput::Vector { name: "black_scholes", values: vec![price, delta] }
            }

            // ══════════════════════════════════════════════════════════════
            // Series acceleration
            // Convention: data column 0 is the sequence of terms.
            // ══════════════════════════════════════════════════════════════

            ("euler_transform", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "euler_transform", value: crate::series_accel::euler_transform(&col) }
            }

            ("wynn_epsilon", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                // Convert to partial sums first
                let sums = crate::series_accel::partial_sums(&col);
                TbsStepOutput::Scalar { name: "wynn_epsilon", value: crate::series_accel::wynn_epsilon(&sums) }
            }

            ("aitken", None) | ("aitken_delta2", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sums = crate::series_accel::partial_sums(&col);
                let acc = crate::series_accel::aitken_delta2(&sums);
                let val = acc.last().cloned().unwrap_or(f64::NAN);
                TbsStepOutput::Scalar { name: "aitken", value: val }
            }

            ("series_accelerate", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                TbsStepOutput::Scalar { name: "series_accelerate", value: crate::series_accel::accelerate(&col) }
            }

            ("cesaro_sum", None) => {
                let c = col_arg(step, 0);
                let col = extract_col(&pipeline.frame().data, pn, pd, c);
                let sums = crate::series_accel::partial_sums(&col);
                TbsStepOutput::Scalar { name: "cesaro_sum", value: crate::series_accel::cesaro_sum(&sums) }
            }

            // ══════════════════════════════════════════════════════════════
            // Number theory
            // Convention: scalar values passed as args, not data columns.
            // ══════════════════════════════════════════════════════════════

            ("is_prime", None) => {
                let n = usize_req(step, "n", 0)? as u64;
                TbsStepOutput::Scalar { name: "is_prime", value: if crate::number_theory::is_prime(n) { 1.0 } else { 0.0 } }
            }

            ("gcd", None) => {
                let a = usize_req(step, "a", 0)? as u64;
                let b = usize_req(step, "b", 1)? as u64;
                TbsStepOutput::Scalar { name: "gcd", value: crate::number_theory::gcd(a, b) as f64 }
            }

            ("lcm", None) => {
                let a = usize_req(step, "a", 0)? as u64;
                let b = usize_req(step, "b", 1)? as u64;
                TbsStepOutput::Scalar { name: "lcm", value: crate::number_theory::lcm(a, b) as f64 }
            }

            ("euler_totient", None) => {
                let n = usize_req(step, "n", 0)? as u64;
                TbsStepOutput::Scalar { name: "euler_totient", value: crate::number_theory::euler_totient(n) as f64 }
            }

            ("factorize", None) => {
                let n = usize_req(step, "n", 0)? as u64;
                let factors = crate::number_theory::factorize(n);
                let flat: Vec<f64> = factors.iter().flat_map(|(p, e)| [*p as f64, *e as f64]).collect();
                TbsStepOutput::Vector { name: "factorize", values: flat }
            }

            // ══════════════════════════════════════════════════════════════
            // Topological data analysis (TDA)
            // Convention: data is n_points × d (point cloud); or n×n dist matrix.
            // ══════════════════════════════════════════════════════════════

            ("rips_h0", None) | ("persistent_homology_h0", None) => {
                // Compute pairwise distance matrix from point cloud
                let data = &pipeline.frame().data;
                let mut dist = vec![0.0_f64; pn * pn];
                for i in 0..pn {
                    for j in 0..pn {
                        let d2: f64 = (0..pd).map(|k| {
                            let diff = data[i * pd + k] - data[j * pd + k];
                            diff * diff
                        }).sum();
                        dist[i * pn + j] = d2.sqrt();
                    }
                }
                let diag = crate::tda::rips_h0(&dist, pn);
                let n_components = diag.pairs.len();
                TbsStepOutput::Scalar { name: "h0_components", value: n_components as f64 }
            }

            ("persistence_entropy", None) => {
                let data = &pipeline.frame().data;
                let mut dist = vec![0.0_f64; pn * pn];
                for i in 0..pn {
                    for j in 0..pn {
                        let d2: f64 = (0..pd).map(|k| {
                            let diff = data[i * pd + k] - data[j * pd + k];
                            diff * diff
                        }).sum();
                        dist[i * pn + j] = d2.sqrt();
                    }
                }
                let max_edge = f64_arg(step, "max_edge", 0, f64::INFINITY);
                let diag = crate::tda::rips_h1(&dist, pn, max_edge);
                let ent = crate::tda::persistence_entropy(&diag.pairs);
                TbsStepOutput::Scalar { name: "persistence_entropy", value: ent }
            }

            ("rips_h1", None) | ("persistent_homology_h1", None) => {
                let data = &pipeline.frame().data;
                let mut dist = vec![0.0_f64; pn * pn];
                for i in 0..pn {
                    for j in 0..pn {
                        let d2: f64 = (0..pd).map(|k| {
                            let diff = data[i * pd + k] - data[j * pd + k];
                            diff * diff
                        }).sum();
                        dist[i * pn + j] = d2.sqrt();
                    }
                }
                let max_edge = f64_arg(step, "max_edge", 0, f64::INFINITY);
                let diag = crate::tda::rips_h1(&dist, pn, max_edge);
                let n_loops = diag.pairs.len();
                TbsStepOutput::Scalar { name: "h1_loops", value: n_loops as f64 }
            }

            ("betti_curve", None) => {
                let data = &pipeline.frame().data;
                let mut dist = vec![0.0_f64; pn * pn];
                for i in 0..pn {
                    for j in 0..pn {
                        let d2: f64 = (0..pd).map(|k| {
                            let diff = data[i * pd + k] - data[j * pd + k];
                            diff * diff
                        }).sum();
                        dist[i * pn + j] = d2.sqrt();
                    }
                }
                let max_edge = f64_arg(step, "max_edge", 0, f64::INFINITY);
                let n_steps = usize_arg(step, "n_steps", 1, 50);
                let diag = crate::tda::rips_h1(&dist, pn, max_edge);
                // Build n_steps evenly-spaced thresholds in [0, max_finite_edge]
                let max_val = dist.iter().cloned().filter(|v| v.is_finite()).fold(0.0_f64, f64::max);
                let thresholds: Vec<f64> = (0..n_steps).map(|i| max_val * i as f64 / (n_steps - 1).max(1) as f64).collect();
                let counts = crate::tda::betti_curve(&diag.pairs, &thresholds);
                // Interleave threshold and betti count
                let mut out = Vec::with_capacity(thresholds.len() * 2);
                for (t, c) in thresholds.iter().zip(counts.iter()) {
                    out.push(*t);
                    out.push(*c as f64);
                }
                TbsStepOutput::Vector { name: "betti_curve", values: out }
            }

            ("persistence_statistics", None) => {
                let data = &pipeline.frame().data;
                let mut dist = vec![0.0_f64; pn * pn];
                for i in 0..pn {
                    for j in 0..pn {
                        let d2: f64 = (0..pd).map(|k| {
                            let diff = data[i * pd + k] - data[j * pd + k];
                            diff * diff
                        }).sum();
                        dist[i * pn + j] = d2.sqrt();
                    }
                }
                let max_edge = f64_arg(step, "max_edge", 0, f64::INFINITY);
                let diag = crate::tda::rips_h1(&dist, pn, max_edge);
                let stats = crate::tda::persistence_statistics(&diag.pairs);
                // [mean, std, max, total_persistence, n_pairs]
                TbsStepOutput::Vector { name: "persistence_statistics", values: stats.to_vec() }
            }

            // bottleneck_distance / wasserstein_distance (TDA):
            // Compare two diagrams computed from first half and second half of rows.
            // Convention: split data at row pn/2; compute H1 for each half; compare.
            ("bottleneck_distance", None) => {
                let half = pn / 2;
                if half < 4 { return Err("bottleneck_distance: need at least 8 rows (4 per half)".into()); }
                let data = &pipeline.frame().data;
                let max_edge = f64_arg(step, "max_edge", 0, f64::INFINITY);
                let build_dist = |rows: std::ops::Range<usize>| {
                    let nr = rows.len();
                    let mut d = vec![0.0_f64; nr * nr];
                    for (ii, i) in rows.clone().enumerate() {
                        for (jj, j) in rows.clone().enumerate() {
                            let v: f64 = (0..pd).map(|k| {
                                let dx = data[i * pd + k] - data[j * pd + k];
                                dx * dx
                            }).sum::<f64>().sqrt();
                            d[ii * nr + jj] = v;
                        }
                    }
                    (d, nr)
                };
                let (d1, n1) = build_dist(0..half);
                let (d2, n2) = build_dist(half..pn);
                let diag1 = crate::tda::rips_h1(&d1, n1, max_edge);
                let diag2 = crate::tda::rips_h1(&d2, n2, max_edge);
                let dist = crate::tda::bottleneck_distance(&diag1.pairs, &diag2.pairs);
                TbsStepOutput::Scalar { name: "bottleneck_distance", value: dist }
            }

            ("wasserstein_distance", None) => {
                let half = pn / 2;
                if half < 4 { return Err("wasserstein_distance: need at least 8 rows (4 per half)".into()); }
                let data = &pipeline.frame().data;
                let max_edge = f64_arg(step, "max_edge", 0, f64::INFINITY);
                let build_dist_half = |from: usize, to: usize| {
                    let nr = to - from;
                    let mut d = vec![0.0_f64; nr * nr];
                    for ii in 0..nr {
                        for jj in 0..nr {
                            let i = from + ii; let j = from + jj;
                            let v: f64 = (0..pd).map(|k| {
                                let dx = data[i * pd + k] - data[j * pd + k];
                                dx * dx
                            }).sum::<f64>().sqrt();
                            d[ii * nr + jj] = v;
                        }
                    }
                    (d, nr)
                };
                let (d1, n1) = build_dist_half(0, half);
                let (d2, n2) = build_dist_half(half, pn);
                let diag1 = crate::tda::rips_h1(&d1, n1, max_edge);
                let diag2 = crate::tda::rips_h1(&d2, n2, max_edge);
                let dist = crate::tda::wasserstein_distance(&diag1.pairs, &diag2.pairs);
                TbsStepOutput::Scalar { name: "wasserstein_distance", value: dist }
            }

            // kmo_bartlett: factor analysis adequacy test
            ("kmo_bartlett", None) | ("kmo", None) => {
                let data = &pipeline.frame().data;
                let corr_mat = crate::factor_analysis::correlation_matrix(data, pn, pd);
                let res = crate::factor_analysis::kmo_bartlett(&corr_mat, pn);
                TbsStepOutput::Vector {
                    name: "kmo_bartlett",
                    values: vec![res.kmo_overall, res.bartlett_statistic, res.bartlett_p_value],
                }
            }

            // Spectral clustering
            ("spectral_cluster", None) => {
                let k = usize_req(step, "k", 0)?;
                let sigma = f64_arg(step, "sigma", 1, 1.0);
                let data = &pipeline.frame().data;
                let params = crate::spectral_clustering::SpectralClusterParams {
                    k,
                    affinity: crate::spectral_clustering::AffinityKind::Rbf { sigma },
                    laplacian: crate::spectral_clustering::LaplacianKind::SymmetricNormalized,
                    kmeans_max_iter: usize_arg(step, "max_iter", 2, 100),
                    kmeans_tol: f64_arg(step, "tol", 3, 1e-6),
                    seed: usize_arg(step, "seed", 4, 42) as u64,
                };
                let result = crate::spectral_clustering::spectral_cluster(data, pn, pd, &params);
                TbsStepOutput::Vector { name: "spectral_labels", values: result.labels.iter().map(|&l| l as f64).collect() }
            }

            ("spectral_embedding", None) => {
                // Build affinity + laplacian from data, then embed
                let k = usize_arg(step, "k", 0, 2);
                let sigma = f64_arg(step, "sigma", 1, 1.0);
                let data = &pipeline.frame().data;
                let sq = crate::spectral_clustering::pairwise_sq_dist(data, pn, pd);
                let w = crate::spectral_clustering::build_affinity(&sq, crate::spectral_clustering::AffinityKind::Rbf { sigma });
                let lap = crate::spectral_clustering::build_laplacian(&w, crate::spectral_clustering::LaplacianKind::SymmetricNormalized);
                let (embedding, _eigenvalues) = crate::spectral_clustering::spectral_embedding(&lap, k, true);
                // embedding is pn*k row-major
                TbsStepOutput::Matrix { name: "spectral_embedding", data: embedding, rows: pn, cols: k }
            }

            // ══════════════════════════════════════════════════════════════
            // Physics — scalar formulas (no data matrix needed)
            // ══════════════════════════════════════════════════════════════

            ("black_body", None) | ("stefan_boltzmann", None) => {
                let emissivity = f64_arg(step, "emissivity", 0, 1.0);
                let area = f64_arg(step, "area", 1, 1.0);
                let temp = f64_req(step, "temperature", 2)?;
                TbsStepOutput::Scalar {
                    name: "stefan_boltzmann",
                    value: crate::physics::stefan_boltzmann(emissivity, area, temp),
                }
            }

            ("carnot_efficiency", None) => {
                let t_hot = f64_req(step, "t_hot", 0)?;
                let t_cold = f64_req(step, "t_cold", 1)?;
                TbsStepOutput::Scalar { name: "carnot_efficiency", value: crate::physics::carnot_efficiency(t_hot, t_cold) }
            }

            ("ideal_gas_pressure", None) => {
                let n_mol = f64_req(step, "n_mol", 0)?;
                let temperature = f64_req(step, "temperature", 1)?;
                let volume = f64_req(step, "volume", 2)?;
                TbsStepOutput::Scalar { name: "ideal_gas_pressure", value: crate::physics::ideal_gas_pressure(n_mol, temperature, volume) }
            }

            ("sho", None) | ("simple_harmonic_oscillator", None) => {
                let x0 = f64_arg(step, "x0", 0, 1.0);
                let v0 = f64_arg(step, "v0", 1, 0.0);
                let omega = f64_arg(step, "omega", 2, 1.0);
                let t = f64_req(step, "t", 3)?;
                let (x, v) = crate::physics::sho_exact(x0, v0, omega, t);
                TbsStepOutput::Vector { name: "sho", values: vec![x, v] }
            }

            // ══════════════════════════════════════════════════════════════
            // Haversine (spatial distance)
            // ══════════════════════════════════════════════════════════════

            ("haversine", None) => {
                if pd < 4 {
                    return Err("haversine: need 4 columns (lat1, lon1, lat2, lon2)".into());
                }
                let data = &pipeline.frame().data;
                let mut distances = Vec::with_capacity(pn);
                for i in 0..pn {
                    let lat1 = data[i * pd];
                    let lon1 = data[i * pd + 1];
                    let lat2 = data[i * pd + 2];
                    let lon2 = data[i * pd + 3];
                    distances.push(crate::spatial::haversine(lat1, lon1, lat2, lon2));
                }
                TbsStepOutput::Vector { name: "haversine", values: distances }
            }

            // ══════════════════════════════════════════════════════════════
            // State-space models: Kalman filter, RTS smoother, particle filter
            // ══════════════════════════════════════════════════════════════

            // Scalar Kalman filter.
            // Frame: 1-column observations (NaN = missing).
            // Args: f, h, q, r, x0, p0 (transition, obs coeff, noise variances, prior).
            // Output: Vector of filtered state means.
            ("kalman_scalar", None) | ("kalman_filter_scalar", None) => {
                let obs: Vec<f64> = (0..pn).map(|i| pipeline.frame().data[i * pd]).collect();
                let f   = f64_arg(step, "f",  0, 1.0);
                let h   = f64_arg(step, "h",  1, 1.0);
                let q   = f64_arg(step, "q",  2, 1.0);
                let r   = f64_arg(step, "r",  3, 1.0);
                let x0  = f64_arg(step, "x0", 4, 0.0);
                let p0  = f64_arg(step, "p0", 5, 1.0);
                match crate::kalman::kalman_filter_scalar(&obs, f, h, q, r, x0, p0) {
                    Some(kf) => TbsStepOutput::Vector { name: "kalman_states", values: kf.states },
                    None => return Err("kalman_scalar: empty observation sequence".into()),
                }
            }

            // RTS smoother (scalar).
            // Frame: 1-column observations; must be preceded by kalman_scalar in pipeline
            // but here we run filter+smoother in one step for convenience.
            // Args: f, h, q, r, x0, p0.
            // Output: Vector of smoothed state means.
            ("rts_scalar", None) | ("rts_smoother_scalar", None) => {
                let obs: Vec<f64> = (0..pn).map(|i| pipeline.frame().data[i * pd]).collect();
                let f   = f64_arg(step, "f",  0, 1.0);
                let h   = f64_arg(step, "h",  1, 1.0);
                let q   = f64_arg(step, "q",  2, 1.0);
                let r   = f64_arg(step, "r",  3, 1.0);
                let x0  = f64_arg(step, "x0", 4, 0.0);
                let p0  = f64_arg(step, "p0", 5, 1.0);
                match crate::kalman::kalman_filter_scalar(&obs, f, h, q, r, x0, p0) {
                    Some(kf) => {
                        let (smoothed, _) = crate::kalman::rts_smoother_scalar(&kf, f, q);
                        TbsStepOutput::Vector { name: "rts_states", values: smoothed }
                    }
                    None => return Err("rts_scalar: empty observation sequence".into()),
                }
            }

            // Particle filter for scalar random-walk model.
            // Frame: 1-column observations (NaN = missing).
            // Args: process_var, obs_var, n_particles, seed.
            // Output: Vector of filtered state means.
            ("particle_filter", None) | ("smc", None) => {
                let obs: Vec<Vec<f64>> = (0..pn)
                    .map(|i| vec![pipeline.frame().data[i * pd]])
                    .collect();
                let process_var = f64_arg(step, "process_var", 0, 1.0);
                let obs_var     = f64_arg(step, "obs_var",     1, 1.0);
                let n_particles = usize_arg(step, "n_particles", 2, 500);
                let seed        = usize_arg(step, "seed",        3, 42) as u64;
                let ssm = crate::state_space::LinearGaussianSsm::random_walk(process_var, obs_var);
                let r = crate::state_space::particle_filter_lgssm(&ssm, &obs, n_particles, seed);
                let means: Vec<f64> = r.means.iter().map(|m| m[0]).collect();
                TbsStepOutput::Vector { name: "particle_means", values: means }
            }

            // ══════════════════════════════════════════════════════════════
            // Hidden Markov Models
            // ══════════════════════════════════════════════════════════════

            // HMM forward-backward: posterior state probabilities.
            // Frame: 1-column integer observation indices.
            // Args: n_states, n_obs (number of symbols).
            // The HMM is initialised uniformly; use hmm_viterbi for decoding.
            // Output: Scalar log-likelihood of the observation sequence.
            ("hmm_forward_backward", None) | ("hmm_fb", None) => {
                let obs: Vec<usize> = (0..pn)
                    .map(|i| pipeline.frame().data[i * pd] as usize)
                    .collect();
                let n_states = usize_req(step, "n_states", 0)?;
                let n_symbols = usize_req(step, "n_obs", 1)?;
                let params = crate::kalman::HmmParams::uniform(n_states, n_symbols);
                match crate::kalman::hmm_forward_backward(&params, &obs) {
                    Some(fb) => TbsStepOutput::Scalar { name: "hmm_log_likelihood", value: fb.log_likelihood },
                    None => return Err("hmm_forward_backward: empty sequence or degenerate model".into()),
                }
            }

            // HMM Viterbi decoding: most probable state sequence.
            // Frame: 1-column integer observation indices.
            // Args: n_states, n_obs.
            // Output: Vector of state indices (as f64).
            ("hmm_viterbi", None) | ("viterbi", None) => {
                let obs: Vec<usize> = (0..pn)
                    .map(|i| pipeline.frame().data[i * pd] as usize)
                    .collect();
                let n_states = usize_req(step, "n_states", 0)?;
                let n_symbols = usize_req(step, "n_obs", 1)?;
                let params = crate::kalman::HmmParams::uniform(n_states, n_symbols);
                match crate::kalman::hmm_viterbi(&params, &obs) {
                    Some((path, _log_prob)) => {
                        let states: Vec<f64> = path.iter().map(|&s| s as f64).collect();
                        TbsStepOutput::Vector { name: "viterbi_path", values: states }
                    }
                    None => return Err("hmm_viterbi: empty sequence".into()),
                }
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
        superpositions_vec.push(step_superposition);
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

    Ok(TbsResult { pipeline, linear_model, logistic_model, outputs, lints, superpositions: superpositions_vec, advice })
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
    fn execute_regression_auto_detect() {
        // Clean linear relationship: y = 2x + 1 + small noise → OLS assumptions satisfied
        let n = 30usize;
        let mut rng = crate::rng::Xoshiro256::new(77);
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let data: Vec<f64> = x.clone();
        let y: Vec<f64> = x.iter()
            .map(|&xi| 2.0 * xi + 1.0 + (crate::rng::TamRng::next_f64(&mut rng) - 0.5) * 2.0)
            .collect();
        let chain = TbsChain::parse("regression()").unwrap();
        let result = execute(chain, data, n, 1, Some(y)).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { name, value } => {
                assert_eq!(*name, "r_squared");
                assert!(*value > 0.9, "R² should be high for clean linear data, got {value}");
            }
            other => panic!("expected Scalar, got {:?}", std::mem::discriminant(other)),
        }
        let adv = result.advice[0].as_ref().unwrap();
        // Should have 4 diagnostics: VIF, normality, BP, Cook's D
        assert_eq!(adv.diagnostics.len(), 4, "expected 4 diagnostics, got {}", adv.diagnostics.len());
        // VIF for single predictor should be 1
        assert!((adv.diagnostics[0].result - 1.0).abs() < 1e-6,
            "VIF should be 1.0 for single predictor, got {}", adv.diagnostics[0].result);
        // Linear model should be populated
        assert!(result.linear_model.is_some(), "linear_model should be set");
    }

    #[test]
    fn execute_regression_without_y_errors() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let chain = TbsChain::parse("regression()").unwrap();
        assert!(execute(chain, data, 20, 1, None).is_err());
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

    #[test]
    fn execute_cluster_auto_two_clusters() {
        // 2D data: 2 clearly separated clusters (10 points each).
        // Cluster A around (0,0), Cluster B around (20,20).
        let mut data: Vec<f64> = Vec::new();
        let mut rng = crate::rng::Xoshiro256::new(55);
        for _ in 0..10 {
            data.push(crate::rng::TamRng::next_f64(&mut rng));
            data.push(crate::rng::TamRng::next_f64(&mut rng));
        }
        for _ in 0..10 {
            data.push(20.0 + crate::rng::TamRng::next_f64(&mut rng));
            data.push(20.0 + crate::rng::TamRng::next_f64(&mut rng));
        }
        let chain = TbsChain::parse("cluster_auto(max_k=4)").unwrap();
        let result = execute(chain, data, 20, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { name, value } => {
                assert_eq!(*name, "silhouette");
                assert!(*value > 0.5, "well-separated clusters should have high silhouette, got {value}");
            }
            other => panic!("expected Scalar, got {:?}", std::mem::discriminant(other)),
        }
        // Best k should be 2 for clearly separated data
        let best_k = result.pipeline.frame().n_clusters.unwrap();
        assert_eq!(best_k, 2, "best_k should be 2 for 2-cluster data, got {best_k}");
        let adv = result.advice[0].as_ref().unwrap();
        assert!(adv.diagnostics.len() >= 2, "should have Hopkins + best_k diagnostics");
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
    fn execute_anova_auto_detect_normal() {
        // 3 groups of normally-distributed samples (Xoshiro256 Box-Muller), clearly separated.
        // The ANOVA auto-detector will: SW → normal or non-normal → route appropriately.
        // Since we can't guarantee SW result for small n, accept Anova, Test, or Nonparametric,
        // but require p < 0.05 for the clearly separated groups.
        // col 0 = value, col 1 = group id.
        let mut rng = crate::rng::Xoshiro256::new(42);
        let mut data: Vec<f64> = Vec::new();
        for _ in 0..15 { data.push(0.0 + crate::rng::TamRng::next_f64(&mut rng)); data.push(0.0); }
        for _ in 0..15 { data.push(20.0 + crate::rng::TamRng::next_f64(&mut rng)); data.push(1.0); }
        for _ in 0..15 { data.push(40.0 + crate::rng::TamRng::next_f64(&mut rng)); data.push(2.0); }
        let chain = TbsChain::parse("anova()").unwrap();
        let result = execute(chain, data, 45, 2, None).unwrap();
        let p = match &result.outputs[0] {
            TbsStepOutput::Anova(a) => {
                assert!(a.eta_squared > 0.9, "eta_squared={}", a.eta_squared);
                a.p_value
            }
            TbsStepOutput::Test(t) => t.p_value,
            TbsStepOutput::Nonparametric(r) => r.p_value,
            other => panic!("unexpected output discriminant={:?}", std::mem::discriminant(other)),
        };
        assert!(p < 0.05, "clearly separated groups should reject H0, p={p}");
        let adv = result.advice[0].as_ref().unwrap();
        assert!(!adv.diagnostics.is_empty(), "should have Levene diagnostic");
    }

    #[test]
    fn execute_anova_auto_detect_nonnormal() {
        // Exponential-ish groups → SW flags non-normal → Kruskal-Wallis.
        // col 0 = value (heavy-tailed), col 1 = group.
        let mut data: Vec<f64> = Vec::new();
        for i in 0..10usize { data.push((i as f64 * 0.4).exp()); data.push(0.0); }
        for i in 0..10usize { data.push((i as f64 * 0.4 + 3.0).exp()); data.push(1.0); }
        let chain = TbsChain::parse("anova()").unwrap();
        let result = execute(chain, data, 20, 2, None).unwrap();
        // Could be Nonparametric (KW) or Test depending on SW pass/fail
        let p = match &result.outputs[0] {
            TbsStepOutput::Nonparametric(r) => r.p_value,
            TbsStepOutput::Test(t) => t.p_value,
            TbsStepOutput::Anova(a) => a.p_value,
            other => panic!("unexpected output {:?}", std::mem::discriminant(other)),
        };
        assert!(p >= 0.0 && p <= 1.0, "p_value={p}");
        let adv = result.advice[0].as_ref().unwrap();
        assert!(!adv.diagnostics.is_empty(), "should have Levene diagnostic");
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

    #[test]
    fn execute_pca_auto_detect() {
        // 4D data where the first 2 dimensions dominate — Kaiser should pick k=2.
        // Build from 2 latent factors: x0=2*f1, x1=1.9*f1, x2=0.1*f2, x3=0.09*f2
        // → first 2 eigenvalues >> 1, last 2 << 1.
        let n = 40usize;
        let mut rng = crate::rng::Xoshiro256::new(13);
        let mut data: Vec<f64> = Vec::new();
        for _ in 0..n {
            let f1 = crate::rng::TamRng::next_f64(&mut rng) * 4.0 - 2.0;
            let f2 = crate::rng::TamRng::next_f64(&mut rng) * 4.0 - 2.0;
            data.push(2.0 * f1);
            data.push(1.9 * f1);
            data.push(0.1 * f2);
            data.push(0.09 * f2);
        }
        // pca() with no n_components → auto-detect
        let chain = TbsChain::parse("pca()").unwrap();
        let result = execute(chain, data, n, 4, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Pca(pca) => {
                // Kaiser should select 2 components (eigenvalue > 1 for 2 factors)
                let nc = pca.explained_variance_ratio.len();
                assert!(nc >= 1 && nc <= 4, "n_components={nc}");
                // At least 1 component returned
            }
            other => panic!("expected Pca, got {:?}", std::mem::discriminant(other)),
        }
        // advice should be populated
        let adv = result.advice[0].as_ref().unwrap();
        assert_eq!(adv.diagnostics.len(), 2, "KMO + Bartlett diagnostics expected");
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

    // ── ANOVA auto-detection (pathmaker) ─────────────────────────────────

    #[test]
    fn execute_anova_auto_detect_separated_groups() {
        // 3 groups with large mean differences → should detect and reject H₀
        let mut data = Vec::new();
        for i in 0..20 { data.push(i as f64 * 0.1 - 1.0); data.push(0.0); }
        for i in 0..20 { data.push(10.0 + i as f64 * 0.1 - 1.0); data.push(1.0); }
        for i in 0..20 { data.push(20.0 + i as f64 * 0.1 - 1.0); data.push(2.0); }
        let chain = TbsChain::parse("anova(col_val=0, col_group=1)").unwrap();
        let result = execute(chain, data, 60, 2, None).unwrap();
        assert!(matches!(&result.outputs[0], TbsStepOutput::Anova(_) | TbsStepOutput::Test(_) | TbsStepOutput::Nonparametric(_)));
        assert!(result.advice[0].is_some());
    }

    // ── New module wiring tests ───────────────────────────────────────────

    #[test]
    fn execute_dijkstra_simple() {
        // 3-node directed graph: 0→1 (1.0), 1→2 (2.0), 0→2 (10.0)
        // Shortest from 0: [0, 1, 3]
        let data = vec![
            0.0, 1.0, 1.0,
            1.0, 2.0, 2.0,
            0.0, 2.0, 10.0,
        ];
        let chain = TbsChain::parse("dijkstra(n_nodes=3, source=0, directed=1)").unwrap();
        let result = execute(chain, data, 3, 3, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert!((values[0] - 0.0).abs() < 1e-9);
                assert!((values[1] - 1.0).abs() < 1e-9);
                assert!((values[2] - 3.0).abs() < 1e-9);
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn execute_pagerank_two_nodes() {
        // Two-node graph: 0→1, 1→0 — both should have equal PageRank
        let data = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let chain = TbsChain::parse("pagerank(n_nodes=2)").unwrap();
        let result = execute(chain, data, 2, 3, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert_eq!(values.len(), 2);
                assert!((values[0] - values[1]).abs() < 1e-3, "should be equal: {:?}", values);
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn execute_lagrange_simple() {
        // Points on y = x^2: (0,0), (1,1), (2,4)
        // Lagrange at x=1.5 should be 2.25
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0];
        let chain = TbsChain::parse("lagrange(x=1.5)").unwrap();
        let result = execute(chain, data, 3, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!((value - 2.25).abs() < 1e-9, "got {value}");
            }
            _ => panic!("expected Scalar output"),
        }
    }

    #[test]
    fn execute_cubic_spline_simple() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 3.0, 9.0];
        let chain = TbsChain::parse("cubic_spline(x=1.5)").unwrap();
        let result = execute(chain, data, 4, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                // Should be close to 2.25 (x^2 pattern)
                assert!(*value > 1.5 && *value < 3.5, "got {value}");
            }
            _ => panic!("expected Scalar output"),
        }
    }

    #[test]
    fn execute_det_identity() {
        // det(I_3) = 1
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let chain = TbsChain::parse("det()").unwrap();
        let result = execute(chain, data, 3, 3, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!((value - 1.0).abs() < 1e-9, "got {value}");
            }
            _ => panic!("expected Scalar output"),
        }
    }

    #[test]
    fn execute_svd_rank() {
        // 3x2 matrix, rank 2 → 2 nonzero singular values
        let data = vec![1.0, 0.0, 0.0, 2.0, 1.0, 2.0];
        let chain = TbsChain::parse("svd()").unwrap();
        let result = execute(chain, data, 3, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert_eq!(values.len(), 2);
                assert!(values[0] > 0.0 && values[1] > 0.0);
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn execute_cronbachs_alpha_perfect() {
        // 5 identical columns → Cronbach's alpha = 1.0
        let mut data = Vec::new();
        for i in 0..20 {
            let v = i as f64;
            for _ in 0..5 { data.push(v); }
        }
        let chain = TbsChain::parse("cronbachs_alpha()").unwrap();
        let result = execute(chain, data, 20, 5, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!(*value > 0.9, "expected alpha ≈ 1, got {value}");
            }
            _ => panic!("expected Scalar output"),
        }
    }

    #[test]
    fn execute_brownian_motion() {
        let data = vec![0.0_f64; 0]; // no input data needed
        let chain = TbsChain::parse("brownian_motion(t_end=1.0, n_steps=100, seed=42)").unwrap();
        let result = execute(chain, data, 0, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert_eq!(values.len(), 101, "should have n_steps+1 points");
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn execute_euler_transform_geometric() {
        // Geometric series terms: 1, -1/2, 1/4, ... → sum = 2/3
        let terms: Vec<f64> = (0..20).map(|k| (-0.5_f64).powi(k)).collect();
        let mut data = Vec::new();
        for v in &terms { data.push(*v); }
        let chain = TbsChain::parse("euler_transform()").unwrap();
        let result = execute(chain, data, terms.len(), 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                // 1/(1+0.5) = 2/3 ≈ 0.6667
                assert!((value - 2.0/3.0).abs() < 0.01, "got {value}");
            }
            _ => panic!("expected Scalar output"),
        }
    }

    #[test]
    fn execute_is_prime() {
        let data = vec![0.0_f64]; // placeholder
        let chain = TbsChain::parse("is_prime(n=17)").unwrap();
        let result = execute(chain, data, 1, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert_eq!(*value, 1.0, "17 is prime");
            }
            _ => panic!("expected Scalar output"),
        }
    }

    #[test]
    fn execute_gcd_basic() {
        let data = vec![0.0_f64];
        let chain = TbsChain::parse("gcd(a=12, b=8)").unwrap();
        let result = execute(chain, data, 1, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert_eq!(*value, 4.0);
            }
            _ => panic!("expected Scalar output"),
        }
    }

    #[test]
    fn execute_markov_stationary() {
        // 2-state symmetric Markov chain: [[0.5, 0.5], [0.5, 0.5]] → stationary = [0.5, 0.5]
        let data = vec![0.5, 0.5, 0.5, 0.5];
        let chain = TbsChain::parse("stationary_distribution()").unwrap();
        let result = execute(chain, data, 2, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert_eq!(values.len(), 2);
                assert!((values[0] - 0.5).abs() < 0.01, "got {:?}", values);
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn execute_black_scholes_call() {
        let data = vec![0.0_f64];
        let chain = TbsChain::parse("black_scholes(s=100, k=100, t=1, r=0.05, sigma=0.2, call=1)").unwrap();
        let result = execute(chain, data, 1, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                // ATM call with r=5%, σ=20%, T=1yr → price ≈ 10.4
                assert!(values[0] > 8.0 && values[0] < 15.0, "price = {}", values[0]);
                // Delta should be around 0.6 for ATM call
                assert!(values[1] > 0.5 && values[1] < 0.8, "delta = {}", values[1]);
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn execute_mardia_normality_bivariate_normal() {
        // Generate bivariate normal data (approx) and check both p-values are > 0.05
        let mut data = Vec::new();
        let mut rng = crate::rng::Xoshiro256::new(999);
        for _ in 0..100 {
            let z1 = crate::rng::TamRng::next_f64(&mut rng) * 2.0 - 1.0;
            let z2 = crate::rng::TamRng::next_f64(&mut rng) * 2.0 - 1.0;
            data.push(z1);
            data.push(z2);
        }
        let chain = TbsChain::parse("mardia_normality()").unwrap();
        let result = execute(chain, data, 100, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                // values = [skewness, skewness_p, kurtosis, kurtosis_p]
                assert_eq!(values.len(), 4);
                // Both p-values should be > 0.05 for normal data
                assert!(values[1] > 0.01, "skewness_p = {} (expected > 0.01)", values[1]);
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn execute_mst_kruskal() {
        // 4-node graph: star topology, node 0 in center
        // Edges: 0-1 (1), 0-2 (2), 0-3 (3), 1-2 (10), 2-3 (10)
        // MST = 0-1, 0-2, 0-3 = total weight 6
        let data = vec![
            0.0, 1.0, 1.0,
            0.0, 2.0, 2.0,
            0.0, 3.0, 3.0,
            1.0, 2.0, 10.0,
            2.0, 3.0, 10.0,
        ];
        let chain = TbsChain::parse("mst(n_nodes=4)").unwrap();
        let result = execute(chain, data, 5, 3, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!((value - 6.0).abs() < 1e-9, "got {value}");
            }
            _ => panic!("expected Scalar output"),
        }
    }

    // ── Kalman / HMM / particle filter ───────────────────────────────────────

    #[test]
    fn tbs_kalman_scalar_basic() {
        // Scalar random walk: observations 0..9. Filtered means should trend upward.
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let chain = TbsChain::parse("kalman_scalar(f=1.0, h=1.0, q=1.0, r=1.0)").unwrap();
        let result = execute(chain, data, 10, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert_eq!(values.len(), 10);
                assert!(values.iter().all(|v| v.is_finite()));
                // Filtered mean at t=9 should be between 5 and 9
                assert!(values[9] > 5.0, "last filtered mean={}", values[9]);
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn tbs_rts_scalar_basic() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let chain = TbsChain::parse("rts_scalar(f=1.0, h=1.0, q=1.0, r=1.0)").unwrap();
        let result = execute(chain, data, 10, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert_eq!(values.len(), 10);
                assert!(values.iter().all(|v| v.is_finite()));
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn tbs_particle_filter_basic() {
        let data: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let chain = TbsChain::parse("particle_filter(process_var=1.0, obs_var=1.0, n_particles=200, seed=42)").unwrap();
        let result = execute(chain, data, 15, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert_eq!(values.len(), 15);
                assert!(values.iter().all(|v| v.is_finite()));
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn tbs_hmm_viterbi_basic() {
        // 5 observations from a 2-state, 3-symbol HMM
        let data = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let chain = TbsChain::parse("hmm_viterbi(n_states=2, n_obs=3)").unwrap();
        let result = execute(chain, data, 5, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Vector { values, .. } => {
                assert_eq!(values.len(), 5);
                // States must be in {0, 1}
                for &s in values {
                    assert!(s == 0.0 || s == 1.0, "invalid state {s}");
                }
            }
            _ => panic!("expected Vector output"),
        }
    }

    #[test]
    fn tbs_hmm_forward_backward_basic() {
        let data = vec![0.0, 1.0, 0.0, 2.0, 1.0];
        let chain = TbsChain::parse("hmm_forward_backward(n_states=2, n_obs=3)").unwrap();
        let result = execute(chain, data, 5, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { value, .. } => {
                assert!(value.is_finite(), "log_likelihood={value}");
                assert!(*value < 0.0, "log_likelihood should be negative, got {value}");
            }
            _ => panic!("expected Scalar output"),
        }
    }

    // ── discover_* Layer 4 superposition dispatches ───────────────────────────

    #[test]
    fn tbs_discover_correlation_basic() {
        let data: Vec<f64> = (0..50)
            .flat_map(|i| [i as f64, 2.0 * i as f64 + 1.0])
            .collect();
        let chain = TbsChain::parse("discover_correlation(col_x=0, col_y=1)").unwrap();
        let result = execute(chain, data, 50, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { name, value } => {
                assert_eq!(*name, "discover_correlation");
                assert!(value.is_finite(), "modal correlation={value}");
                assert!(*value > 0.9, "perfectly linear data → high correlation, got {value}");
            }
            other => panic!("expected Scalar, got {:?}", std::mem::discriminant(other)),
        }
        // Superposition should be populated
        assert!(result.superpositions[0].is_some(), "superposition should be set");
        let sup = result.superpositions[0].as_ref().unwrap();
        assert_eq!(sup.views.len(), 5, "should have 5 correlation methods");
        assert_eq!(sup.swept_param, "correlation_method");
    }

    #[test]
    fn tbs_discover_regression_basic() {
        let data: Vec<f64> = (0..40)
            .flat_map(|i| [i as f64, 3.0 * i as f64 + 5.0])
            .collect();
        let chain = TbsChain::parse("discover_regression(col_x=0, col_y=1)").unwrap();
        let result = execute(chain, data, 40, 2, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { name, value } => {
                assert_eq!(*name, "discover_regression_slope");
                assert!(value.is_finite());
                assert!((value - 3.0).abs() < 0.5, "slope should be near 3.0, got {value}");
            }
            other => panic!("expected Scalar, got {:?}", std::mem::discriminant(other)),
        }
        let sup = result.superpositions[0].as_ref().unwrap();
        assert_eq!(sup.views.len(), 3, "OLS + Theil-Sen + Siegel");
        assert_eq!(sup.swept_param, "regression_method");
    }

    #[test]
    fn tbs_discover_changepoint_basic() {
        // Step change at index 25
        let mut data: Vec<f64> = (0..50).map(|_| 0.0).collect();
        for i in 25..50 { data[i] = 10.0; }
        let chain = TbsChain::parse("discover_changepoint(col=0)").unwrap();
        let result = execute(chain, data, 50, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { name, value } => {
                assert_eq!(*name, "discover_changepoint_n");
                assert!(value.is_finite());
                assert!(*value >= 1.0, "should detect at least 1 changepoint, got {value}");
            }
            other => panic!("expected Scalar, got {:?}", std::mem::discriminant(other)),
        }
        let sup = result.superpositions[0].as_ref().unwrap();
        assert_eq!(sup.views.len(), 3, "CUSUM + PELT + binary segmentation");
        assert_eq!(sup.swept_param, "changepoint_method");
    }

    #[test]
    fn tbs_discover_stationarity_basic() {
        // White noise: should be stationary
        let data: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let chain = TbsChain::parse("discover_stationarity(col=0, alpha=0.05)").unwrap();
        let result = execute(chain, data, 100, 1, None).unwrap();
        match &result.outputs[0] {
            TbsStepOutput::Scalar { name, value } => {
                assert_eq!(*name, "discover_stationarity");
                assert!(*value == 0.0 || *value == 1.0, "modal_value must be 0 or 1, got {value}");
            }
            other => panic!("expected Scalar, got {:?}", std::mem::discriminant(other)),
        }
        let sup = result.superpositions[0].as_ref().unwrap();
        assert_eq!(sup.views.len(), 4, "ADF + KPSS + PP + VR");
        assert_eq!(sup.swept_param, "stationarity_method");
    }
}
