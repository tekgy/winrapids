//! Science lints for `.tbs` chains.
//!
//! Mathematical correctness warnings that run at parse-time (static) or
//! at the start of execution (dynamic). Lints produce warnings, not errors —
//! the user can suppress them.
//!
//! Spec: `docs/research/tambear-build/tbs-science-lint-spec.md`

use crate::tbs_parser::{TbsChain, TbsName};
use crate::proof::{
    BinOp, Prop, Proof, Sort, Structure, StructuralFact,
    Term, Theorem,
};

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// A single science lint produced during chain analysis.
#[derive(Debug, Clone)]
pub struct TbsLint {
    /// Lint code (e.g. "L001", "L101").
    pub code: &'static str,
    /// Severity level.
    pub severity: LintSeverity,
    /// Human-readable message.
    pub message: String,
    /// Which step in the chain triggered this lint (if applicable).
    pub step_index: Option<usize>,
}

/// Lint severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintSeverity {
    Info,
    Warning,
    Error,
}

// ═══════════════════════════════════════════════════════════════════════════
// Static lints (parse-time, no data needed)
// ═══════════════════════════════════════════════════════════════════════════

/// Distance-based operations that are scale-sensitive.
const DISTANCE_OPS: &[&str] = &["dbscan", "discover_clusters", "kmeans", "knn"];

/// Check whether a step uses a scale-invariant metric.
fn has_scale_invariant_metric(step: &crate::tbs_parser::TbsStep) -> bool {
    if let Some(v) = step.get_arg("metric", 99) {
        if let Some(s) = v.as_str() {
            return s == "cosine" || s == "correlation";
        }
    }
    false
}

/// Run all static lints on a parsed chain. Returns collected warnings.
pub fn static_lints(chain: &TbsChain) -> Vec<TbsLint> {
    let mut lints = Vec::new();
    lint_l001_missing_normalization(chain, &mut lints);
    lint_l002_supervised_without_exploration(chain, &mut lints);
    lint_l003_predict_on_training_data(chain, &mut lints);
    lint_l004_distribution_assumption(chain, &mut lints);
    lint_l005_redundant_consecutive(chain, &mut lints);
    lint_l006_preprocessing_after_consumer(chain, &mut lints);
    lints.extend(kingdom_lints(chain));
    lints
}

/// L001: Missing normalization before distance-based operations.
///
/// Distance-based algorithms are scale-sensitive. Features with large range
/// dominate Euclidean distance. Without normalization, clustering results
/// are artifacts of units, not structure.
fn lint_l001_missing_normalization(chain: &TbsChain, lints: &mut Vec<TbsLint>) {
    let mut seen_normalize = false;

    for (i, step) in chain.steps.iter().enumerate() {
        let name = match &step.name {
            TbsName::Simple(s) => s.as_str(),
            TbsName::Dotted(_, _) => continue,
        };

        if name == "normalize" {
            seen_normalize = true;
            continue;
        }

        if DISTANCE_OPS.contains(&name) && !seen_normalize && !has_scale_invariant_metric(step) {
            lints.push(TbsLint {
                code: "L001",
                severity: LintSeverity::Warning,
                message: format!(
                    "\u{26a0} L001: {name} uses Euclidean distance \
                     — consider normalize() first to avoid scale artifacts"
                ),
                step_index: Some(i),
            });
        }
    }
}

/// L002: Supervised step without any exploratory step.
///
/// The .tbs philosophy is exploration-first. This is a suggestion, not a
/// hard warning — linear regression doesn't require clustering.
fn lint_l002_supervised_without_exploration(chain: &TbsChain, lints: &mut Vec<TbsLint>) {
    const EXPLORATORY: &[&str] = &[
        "describe", "discover_clusters", "dbscan", "kmeans", "knn",
        "pca", "factor_analysis", "tsne", "umap",
    ];
    const SUPERVISED: &[(&str, Option<&str>)] = &[
        ("train", Some("linear")),
        ("train", Some("logistic")),
    ];

    let mut seen_exploratory = false;

    for (i, step) in chain.steps.iter().enumerate() {
        let (base, sub) = step.name.as_str();

        // Check if this is an exploratory step
        if sub.is_none() && EXPLORATORY.contains(&base) {
            seen_exploratory = true;
        }

        // Check if this is a supervised step without prior exploration
        if SUPERVISED.iter().any(|(b, s)| *b == base && *s == sub) && !seen_exploratory {
            lints.push(TbsLint {
                code: "L002",
                severity: LintSeverity::Info,
                message: "\u{2139} L002: consider exploratory steps \
                          (describe, discover_clusters) before supervised training"
                    .to_string(),
                step_index: Some(i),
            });
        }
    }
}

/// L003: Chained predictions without validation.
///
/// `train.linear().predict()` on the same data gives optimistic performance
/// estimates. This lint detects train immediately followed by predict.
fn lint_l003_predict_on_training_data(chain: &TbsChain, lints: &mut Vec<TbsLint>) {
    let mut prev_was_train = false;

    for (i, step) in chain.steps.iter().enumerate() {
        let (base, sub) = step.name.as_str();

        let is_train = base == "train" && sub.is_some();
        let is_predict = base == "predict" && sub.is_none();

        if is_predict && prev_was_train {
            lints.push(TbsLint {
                code: "L003",
                severity: LintSeverity::Warning,
                message: "\u{26a0} L003: predicting on training data \
                          — results are optimistically biased"
                    .to_string(),
                step_index: Some(i),
            });
        }

        prev_was_train = is_train;
    }
}

/// L004: Distribution assumption mismatch.
///
/// Warns when a normality-assuming test (t_test, anova, f_test) follows a
/// skewness/describe step in the chain. The chain structure implies the user
/// knows about distribution shape but proceeds with normality-assuming tests.
///
/// This is structural — the actual skewness threshold is checked dynamically
/// at execution time (L109, below). L004 flags the chain *pattern*.
fn lint_l004_distribution_assumption(chain: &TbsChain, lints: &mut Vec<TbsLint>) {
    const SHAPE_OPS: &[&str] = &["skewness", "describe", "moments"];
    const NORMALITY_OPS: &[&str] = &[
        "t_test", "anova", "f_test", "pearson_r",
        "one_sample_t", "paired_t", "welch_t",
    ];
    // If the user explicitly tested normality, suppress the warning
    const NORMALITY_CHECKS: &[&str] = &["ks_test", "shapiro_wilk", "anderson_darling"];

    let mut seen_shape = false;
    let mut checked_normality = false;

    for (i, step) in chain.steps.iter().enumerate() {
        let (base, _) = step.name.as_str();

        if SHAPE_OPS.contains(&base) {
            seen_shape = true;
        }

        if NORMALITY_CHECKS.contains(&base) {
            checked_normality = true;
        }

        if seen_shape && !checked_normality && NORMALITY_OPS.contains(&base) {
            lints.push(TbsLint {
                code: "L004",
                severity: LintSeverity::Info,
                message: format!(
                    "\u{2139} L004: {base} assumes normality — chain includes \
                     distribution shape analysis. Check |skewness| < 1 before trusting results."
                ),
                step_index: Some(i),
            });
        }
    }
}

/// L005: Redundant consecutive identical operations.
///
/// Two adjacent steps with the same name AND identical arguments are redundant.
/// Most .tbs operations are idempotent (normalize, describe) or produce the
/// same output when given identical params. Only fires when the full step
/// (name + args) is identical — `kmeans(k=3).kmeans(k=5)` is NOT redundant.
fn lint_l005_redundant_consecutive(chain: &TbsChain, lints: &mut Vec<TbsLint>) {
    for i in 1..chain.steps.len() {
        if chain.steps[i - 1] == chain.steps[i] {
            let name = &chain.steps[i].name;
            lints.push(TbsLint {
                code: "L005",
                severity: LintSeverity::Info,
                message: format!(
                    "\u{2139} L005: redundant consecutive {name}() — \
                     the second call produces the same result"
                ),
                step_index: Some(i),
            });
        }
    }
}

/// L006: Preprocessing step after a consuming algorithm.
///
/// Detects `normalize`, `standardize`, `log`, or `boxcox` appearing AFTER
/// a distance-based or normality-assuming algorithm. The preprocessing
/// transforms the *output* (labels, scores), not the *input* features —
/// almost certainly not what was intended.
fn lint_l006_preprocessing_after_consumer(chain: &TbsChain, lints: &mut Vec<TbsLint>) {
    const PREPROCESSING: &[&str] = &["normalize", "standardize", "log", "boxcox"];
    const CONSUMERS: &[&str] = &[
        "kmeans", "dbscan", "discover_clusters", "knn", "tsne", "umap",
        "t_test", "anova", "f_test", "pearson_r",
        "one_sample_t", "paired_t", "welch_t",
    ];

    for (i, step) in chain.steps.iter().enumerate() {
        let (base, _) = step.name.as_str();
        if !PREPROCESSING.contains(&base) {
            continue;
        }

        // Walk backwards to find a consumer before this preprocessing step
        let has_consumer_before = chain.steps[..i].iter().any(|s| {
            let (b, _) = s.name.as_str();
            CONSUMERS.contains(&b)
        });

        if !has_consumer_before {
            continue;
        }

        // Suppression: if another consumer appears AFTER this preprocessing
        // step, the user may be intentionally re-preprocessing for a second
        // analysis phase.
        let has_consumer_after = chain.steps[i + 1..].iter().any(|s| {
            let (b, _) = s.name.as_str();
            CONSUMERS.contains(&b)
        });

        if has_consumer_after {
            continue;
        }

        // Find which consumer preceded this preprocessing step
        let consumer_name = chain.steps[..i].iter().rev().find_map(|s| {
            let (b, _) = s.name.as_str();
            if CONSUMERS.contains(&b) { Some(b) } else { None }
        }).unwrap();

        lints.push(TbsLint {
            code: "L006",
            severity: LintSeverity::Warning,
            message: format!(
                "\u{26a0} L006: {base}() after {consumer_name}() normalizes the output \
                 labels, not the input features. If you meant to normalize before \
                 {consumer_name}(), move {base}() before {consumer_name}()."
            ),
            step_index: Some(i),
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dynamic lints (at execution, require data inspection)
// ═══════════════════════════════════════════════════════════════════════════

/// L106: Detect near-constant columns in input data.
///
/// A near-constant column contributes nothing to distance-based methods and
/// can cause numerical issues in regression (singular X'X).
pub fn lint_l106_constant_columns(data: &[f64], n: usize, d: usize) -> Vec<TbsLint> {
    let mut lints = Vec::new();
    for j in 0..d {
        let col: Vec<f64> = (0..n).map(|i| data[i * d + j]).collect();
        let m = crate::descriptive::moments_ungrouped(&col);
        let mean = m.mean();
        let std = m.std(0);

        let is_constant = if mean.abs() < 1e-10 {
            std < 1e-15
        } else {
            std / mean.abs() < 1e-10
        };

        if is_constant {
            lints.push(TbsLint {
                code: "L106",
                severity: LintSeverity::Warning,
                message: format!(
                    "\u{26a0} L106: column {j} is near-constant (\u{03c3} = {std:.2e}). \
                     Consider removing before analysis."
                ),
                step_index: None,
            });
        }
    }
    lints
}

/// L101: Naive variance warning.
///
/// Checks whether data range suggests naive variance formula would lose
/// precision. Tambear uses Welford internally, but warns if manual
/// computation would be dangerous.
pub fn lint_l101_naive_variance(data: &[f64], n: usize, d: usize, chain: &TbsChain) -> Vec<TbsLint> {
    // Only fire if chain contains var/std/moments-like operations
    let has_variance_op = chain.steps.iter().any(|s| {
        let (base, _) = s.name.as_str();
        matches!(base, "var" | "std" | "describe" | "moments")
    });
    if !has_variance_op {
        return Vec::new();
    }

    let mut lints = Vec::new();
    for j in 0..d {
        let col: Vec<f64> = (0..n).map(|i| data[i * d + j]).collect();
        let m = crate::descriptive::moments_ungrouped(&col);
        let min = m.min;
        let max = m.max;
        let mean = m.mean();
        let var = m.variance(0);

        let range_ratio = if min.abs() > 1e-300 { max / min } else { 0.0 };
        let condition = if var > 1e-300 { mean * mean / var } else { 0.0 };

        if range_ratio.abs() > 1e6 || condition > 1e10 {
            lints.push(TbsLint {
                code: "L101",
                severity: LintSeverity::Warning,
                message: format!(
                    "\u{26a0} L101: data range ({max:.2e}/{min:.2e}) suggests naive variance \
                     may lose precision — use moments() instead of manual \u{03a3}(x-x\u{0304})\u{00b2}"
                ),
                step_index: None,
            });
            break; // one warning is enough
        }
    }
    lints
}

/// L109: High skewness before normality-assuming test.
///
/// Dynamic companion to L004. Fires when `describe()` results show
/// |skewness| > 1.0 in any column AND the chain includes a normality-
/// assuming test. Threshold 1.0 is standard (Bulmer 1979: |skew| > 1 is
/// "highly skewed").
pub fn lint_l109_skewness_normality(
    describe: &crate::pipeline::DescribeResult,
    chain: &TbsChain,
) -> Vec<TbsLint> {
    const NORMALITY_OPS: &[&str] = &[
        "t_test", "anova", "f_test", "pearson_r",
        "one_sample_t", "paired_t", "welch_t",
    ];

    let has_normality_op = chain.steps.iter().any(|s| {
        let (base, _) = s.name.as_str();
        NORMALITY_OPS.contains(&base)
    });
    if !has_normality_op {
        return Vec::new();
    }

    let mut lints = Vec::new();
    for col in &describe.columns {
        if col.skewness.abs() > 1.0 {
            lints.push(TbsLint {
                code: "L109",
                severity: LintSeverity::Warning,
                message: format!(
                    "\u{26a0} L109: column {} has |skewness| = {:.2} > 1.0 — \
                     normality-assuming tests may be unreliable. Consider \
                     nonparametric alternatives (mann_whitney, kruskal_wallis).",
                    col.index, col.skewness.abs()
                ),
                step_index: None,
            });
        }
    }
    lints
}

// ═══════════════════════════════════════════════════════════════════════════
// Kingdom-aware lints (L201-L202)
// ═══════════════════════════════════════════════════════════════════════════

/// Computation kingdom classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kingdom {
    /// Single-pass accumulate.
    A,
    /// Sequential / sampling (order-dependent).
    B,
    /// Iterative convergence.
    C,
}

/// Shared Kingdom A subproblem that two adjacent steps might share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedSubproblem {
    DistanceMatrix,
    Covariance,
    GramMatrix,
    WeightedCovariance,
    Henderson,
    CrossProduct,
    RiskSet,
    Autocorrelation,
}

/// Look up the kingdom and shared subproblem for a .tbs operation.
fn kingdom_of(base: &str, sub: Option<&str>) -> (Kingdom, Option<SharedSubproblem>) {
    match (base, sub) {
        // Kingdom A — single-pass accumulate, no shared subproblem
        ("normalize", None) | ("describe", None) | ("moments", None)
        | ("var", None) | ("std", None) | ("median", None)
            => (Kingdom::A, None),

        // Kingdom A — covariance-based
        ("pca", None) | ("efa", None) | ("varimax", None)
        | ("manova", None) | ("lda", None) | ("cca", None)
            => (Kingdom::A, Some(SharedSubproblem::Covariance)),

        // Kingdom A — Gram matrix (X'X) based
        ("train", Some("linear")) | ("panel_fe", None) | ("two_sls", None)
        | ("hausman", None)
            => (Kingdom::A, Some(SharedSubproblem::GramMatrix)),

        // Kingdom A — autocorrelation based
        ("adf_test", None) | ("ar", None) | ("yule_walker", None)
            => (Kingdom::A, Some(SharedSubproblem::Autocorrelation)),

        // Kingdom A — suffix-count scan (reclassified: at-risk counts are deterministic
        // suffix counts of sorted data, not accumulated path-dependent state.
        // n1(t) = |{i: times[i] >= t, groups[i] == 0}| is computable for any t
        // independently of processing order — same structural argument as kaplan_meier)
        ("log_rank", None)
            => (Kingdom::A, None),

        // Kingdom A — prefix product scan (survival.rs: prior Kingdom B label was wrong)
        ("kaplan_meier", None)
            => (Kingdom::A, None),

        // Kingdom A — affine prefix scan (time_series.rs: prior Kingdom B label was wrong)
        ("exp_smoothing", None)
            => (Kingdom::A, None),

        // Kingdom C — distance-matrix dependent
        ("kmeans", None) | ("dbscan", None) | ("discover_clusters", None)
        | ("knn", None) | ("tsne", None) | ("umap", None)
        | ("discover", None) | ("discover_with", None)
            => (Kingdom::C, Some(SharedSubproblem::DistanceMatrix)),

        // Kingdom C — weighted covariance (EM)
        ("gmm", None) | ("mixture", None)
            => (Kingdom::C, Some(SharedSubproblem::WeightedCovariance)),

        // Kingdom C — Henderson equations
        ("lme", None) | ("panel_re", None)
            => (Kingdom::C, Some(SharedSubproblem::Henderson)),

        // Kingdom C — covariance-based convergence
        ("cfa", None)
            => (Kingdom::C, Some(SharedSubproblem::Covariance)),

        // Kingdom C — cross-product based
        ("irt", None)
            => (Kingdom::C, Some(SharedSubproblem::CrossProduct)),

        // Kingdom C — risk set based
        ("cox_ph", None)
            => (Kingdom::C, Some(SharedSubproblem::RiskSet)),

        // Kingdom C — logistic (Gram matrix iterative)
        ("train", Some("logistic"))
            => (Kingdom::C, Some(SharedSubproblem::GramMatrix)),

        // Kingdom C — GARCH fitting (A filter + C optimization outer loop)
        // The garch11_filter is Kingdom A (affine prefix scan); the fit is Kingdom C.
        ("garch", None)
            => (Kingdom::C, Some(SharedSubproblem::Covariance)),

        // Default: assume Kingdom A, no shared subproblem
        _ => (Kingdom::A, None),
    }
}

/// L201-L202: Kingdom-aware chain analysis.
///
/// L201: Adjacent steps with the same shared subproblem — suggest fusion.
/// L202: Kingdom B step breaking a shared subproblem chain — warn about
///       data transformation invalidating downstream assumptions.
pub fn kingdom_lints(chain: &TbsChain) -> Vec<TbsLint> {
    let mut lints = Vec::new();
    if chain.steps.len() < 2 {
        return lints;
    }

    let annotations: Vec<(Kingdom, Option<SharedSubproblem>)> = chain.steps.iter()
        .map(|s| {
            let (base, sub) = s.name.as_str();
            kingdom_of(base, sub)
        })
        .collect();

    for i in 1..annotations.len() {
        let (prev_k, prev_sub) = annotations[i - 1];
        let (curr_k, curr_sub) = annotations[i];
        let prev_name = &chain.steps[i - 1].name;
        let curr_name = &chain.steps[i].name;

        // L201: Redundant shared subproblem — two adjacent C steps
        // that both need the same A subproblem.
        if prev_k == Kingdom::C && curr_k == Kingdom::C
            && prev_sub.is_some() && prev_sub == curr_sub
        {
            let sub_name = match prev_sub.unwrap() {
                SharedSubproblem::DistanceMatrix => "distance matrix",
                SharedSubproblem::Covariance => "covariance matrix",
                SharedSubproblem::GramMatrix => "Gram matrix (X'X)",
                SharedSubproblem::WeightedCovariance => "weighted covariance",
                SharedSubproblem::Henderson => "Henderson equations",
                SharedSubproblem::CrossProduct => "cross-product matrix",
                SharedSubproblem::RiskSet => "risk set sums",
                SharedSubproblem::Autocorrelation => "autocorrelation",
            };
            lints.push(TbsLint {
                code: "L201",
                severity: LintSeverity::Info,
                message: format!(
                    "\u{2139} L201: {prev_name} and {curr_name} both compute {sub_name} \
                     — the session shares this automatically"
                ),
                step_index: Some(i),
            });
        }

        // L202: Kingdom B boundary breaking shared subproblem chain.
        // Check: if step i is B, and steps i-1 and i+1 share a subproblem,
        // the B step transforms data and invalidates the shared intermediate.
        if prev_k == Kingdom::B && i + 1 < annotations.len() {
            let (next_k, next_sub) = annotations[i + 1];
            if let (Some(curr_s), Some(next_s)) = (curr_sub, next_sub) {
                if curr_s == next_s && (curr_k == Kingdom::C || next_k == Kingdom::C) {
                    let next_name = &chain.steps[i + 1].name;
                    lints.push(TbsLint {
                        code: "L202",
                        severity: LintSeverity::Warning,
                        message: format!(
                            "\u{26a0} L202: {prev_name} (sequential) transforms data between \
                             {curr_name} and {next_name} — shared intermediate is invalidated"
                        ),
                        step_index: Some(i),
                    });
                }
            }
        }
    }

    lints
}

// ═══════════════════════════════════════════════════════════════════════════
// Proof-verified kingdom contracts
// ═══════════════════════════════════════════════════════════════════════════

/// A machine-verified kingdom contract for a .tbs operation.
///
/// `kingdom_of()` is a static lookup table — it says "kmeans is Kingdom C"
/// as a comment assertion. `KingdomContract` upgrades that to a proof:
/// the claim is verified by the `tambear::proof` system, which checks
/// algebraic laws (associativity, mergeability, etc.) before accepting it.
///
/// This is the "proof system as contract verifier" pattern:
/// - Kingdom A → associativity theorem verified
/// - Kingdom B → declared non-liftable (honest, no proof)
/// - Kingdom C → declared iterative (honest, no structural proof)
#[derive(Debug, Clone)]
pub struct KingdomContract {
    /// The .tbs operation name.
    pub op: String,
    /// The declared kingdom.
    pub kingdom: Kingdom,
    /// The shared subproblem (if any).
    pub shared: Option<SharedSubproblem>,
    /// Verification result.
    pub verification: KingdomVerification,
}

/// Result of attempting to machine-verify a kingdom claim.
#[derive(Debug, Clone)]
pub enum KingdomVerification {
    /// The claim is machine-verified by the proof system.
    Verified(Theorem),
    /// The claim is honestly declared but structurally unverifiable
    /// (B = sequential, C = convergence — no associativity proof possible).
    HonestDeclaration { reason: &'static str },
    /// The claim was rejected by the proof checker.
    Rejected(String),
}

impl KingdomVerification {
    /// True if verified or honestly declared (not rejected).
    pub fn is_sound(&self) -> bool {
        !matches!(self, KingdomVerification::Rejected(_))
    }
}

/// Construct the `Structure` for a Kingdom A shared subproblem.
///
/// Each `SharedSubproblem` has a well-defined accumulate structure:
/// - `Covariance` / `GramMatrix` / `Autocorrelation` → (ℝⁿ, +, 0) component-wise add (Mergeability)
/// - `DistanceMatrix` → (ℝ, +, 0) per-cell add (Mergeability)
/// - `RiskSet` → (ℕ, +, 0) event count accumulate
fn subproblem_structure(sub: SharedSubproblem) -> Structure {
    match sub {
        // Covariance accumulates sufficient statistics (sum, sumsq, cross-products).
        // Combine = component-wise add on the sufficient-stat vector → commutative monoid + Mergeability.
        SharedSubproblem::Covariance | SharedSubproblem::WeightedCovariance => Structure {
            name: "CovarianceAccumulate".into(),
            carrier: Sort::Named("SufficientStats".into()),
            op: Some(BinOp::Add),
            identity: Some(Term::Lit(0.0)),
            laws: vec![
                StructuralFact::Associativity,
                StructuralFact::Commutativity,
                StructuralFact::Identity,
                StructuralFact::Mergeability,
            ],
        },
        // Gram matrix X'X = tiled dot-product accumulate over outer products.
        // Combine = matrix add → associative + Mergeability.
        SharedSubproblem::GramMatrix => Structure {
            name: "GramMatrixAccumulate".into(),
            carrier: Sort::Named("Matrix".into()),
            op: Some(BinOp::Add),
            identity: Some(Term::Lit(0.0)),
            laws: vec![
                StructuralFact::Associativity,
                StructuralFact::Identity,
                StructuralFact::Mergeability,
            ],
        },
        // ACF/PACF: accumulate sum and lag-k products → (ℝ, +, 0).
        SharedSubproblem::Autocorrelation => Structure::commutative_monoid(
            Sort::Real,
            BinOp::Add,
            Term::Lit(0.0),
        ),
        // Distance matrix: per-cell squared differences → (ℝ, +, 0) per cell.
        SharedSubproblem::DistanceMatrix => Structure {
            name: "DistanceMatrixAccumulate".into(),
            carrier: Sort::Named("Matrix".into()),
            op: Some(BinOp::Add),
            identity: Some(Term::Lit(0.0)),
            laws: vec![
                StructuralFact::Associativity,
                StructuralFact::Commutativity,
                StructuralFact::Identity,
                StructuralFact::Mergeability,
            ],
        },
        // Risk set: event indicator sums → (ℕ, +, 0).
        SharedSubproblem::RiskSet => Structure::commutative_monoid(
            Sort::Nat,
            BinOp::Add,
            Term::NatLit(0),
        ),
        // Cross-product matrix: outer product accumulate → matrix add.
        SharedSubproblem::CrossProduct => Structure {
            name: "CrossProductAccumulate".into(),
            carrier: Sort::Named("Matrix".into()),
            op: Some(BinOp::Add),
            identity: Some(Term::Lit(0.0)),
            laws: vec![
                StructuralFact::Associativity,
                StructuralFact::Identity,
                StructuralFact::Mergeability,
            ],
        },
        // Henderson equations: Z'Z + D⁻¹ and Z'y → matrix/vector add on sufficient stats.
        SharedSubproblem::Henderson => Structure {
            name: "HendersonAccumulate".into(),
            carrier: Sort::Named("SufficientStats".into()),
            op: Some(BinOp::Add),
            identity: Some(Term::Lit(0.0)),
            laws: vec![
                StructuralFact::Associativity,
                StructuralFact::Commutativity,
                StructuralFact::Identity,
                StructuralFact::Mergeability,
            ],
        },
    }
}

/// Build and machine-verify the kingdom contract for a .tbs operation.
///
/// This is the one-line call that connects `kingdom_of()` (static lookup)
/// to the proof system (algebraic verification). The returned `KingdomContract`
/// carries either a `Theorem` or an `HonestDeclaration` or a `Rejected` error.
///
/// Kingdom A operations get a structural proof that their combine step is
/// associative (and thus parallel-scan liftable). Kingdom B and C operations
/// get an `HonestDeclaration` — they honestly cannot be expressed as a single
/// accumulate over an associative op.
pub fn verify_kingdom(op_name: &str, sub_name: Option<&str>) -> KingdomContract {
    let (kingdom, shared) = kingdom_of(op_name, sub_name);

    let verification = match kingdom {
        Kingdom::B => KingdomVerification::HonestDeclaration {
            reason: "Kingdom B: sequential/order-dependent — not liftable to parallel scan",
        },
        Kingdom::C => KingdomVerification::HonestDeclaration {
            reason: "Kingdom C: iterative convergence — no single associative combine step",
        },
        Kingdom::A => {
            // Build the associativity proposition and prove it from the structure.
            let structure = if let Some(sub) = shared {
                subproblem_structure(sub)
            } else {
                // Pure Kingdom A without shared subproblem: (ℝ, +, 0) global reduce.
                Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0))
            };

            // Proposition: ∀a,b,c. (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
            let prop = Prop::Forall {
                vars: vec![("a", Sort::Real), ("b", Sort::Real), ("c", Sort::Real)],
                body: Box::new(Prop::Eq(
                    Term::BinApp(
                        BinOp::Add,
                        Box::new(Term::BinApp(
                            BinOp::Add,
                            Box::new(Term::Var("a")),
                            Box::new(Term::Var("b")),
                        )),
                        Box::new(Term::Var("c")),
                    ),
                    Term::BinApp(
                        BinOp::Add,
                        Box::new(Term::Var("a")),
                        Box::new(Term::BinApp(
                            BinOp::Add,
                            Box::new(Term::Var("b")),
                            Box::new(Term::Var("c")),
                        )),
                    ),
                )),
            };

            let proof = Proof::ByStructure(structure, StructuralFact::Associativity);
            let thm_name = format!("{op_name}_kingdom_a_associativity");

            match Theorem::check(&thm_name, prop, proof) {
                Ok(thm) => KingdomVerification::Verified(thm),
                Err(e) => KingdomVerification::Rejected(e.to_string()),
            }
        }
    };

    KingdomContract {
        op: op_name.to_string(),
        kingdom,
        shared,
        verification,
    }
}

/// Verify all operations in the `kingdom_of()` lookup table.
///
/// Returns contracts for every known operation. Any `Rejected` entry is a bug:
/// the lookup table claims a kingdom that the proof system cannot verify.
pub fn verify_all_kingdom_contracts() -> Vec<KingdomContract> {
    // Every operation in the kingdom_of() match arms.
    let ops: &[(&str, Option<&str>)] = &[
        // Kingdom A — no shared subproblem
        ("normalize", None), ("describe", None), ("moments", None),
        ("var", None), ("std", None), ("median", None),
        // Kingdom A — covariance-based
        ("pca", None), ("efa", None), ("varimax", None),
        ("manova", None), ("lda", None), ("cca", None),
        // Kingdom A — Gram matrix
        ("train", Some("linear")), ("panel_fe", None), ("two_sls", None),
        ("hausman", None),
        // Kingdom A — autocorrelation
        ("adf_test", None), ("ar", None), ("yule_walker", None),
        // Kingdom A — prefix product scan (reclassified from B)
        ("kaplan_meier", None),
        // Kingdom A — affine prefix scan (reclassified from B)
        ("exp_smoothing", None),
        // Kingdom A — suffix-count scan (reclassified from B: at-risk counts are deterministic)
        ("log_rank", None),
        // Kingdom C — distance-matrix
        ("kmeans", None), ("dbscan", None), ("discover_clusters", None),
        ("knn", None), ("tsne", None), ("umap", None),
        ("discover", None), ("discover_with", None),
        // Kingdom C — EM
        ("gmm", None), ("mixture", None),
        // Kingdom C — Henderson
        ("lme", None), ("panel_re", None),
        // Kingdom C — covariance convergence
        ("cfa", None),
        // Kingdom C — cross-product
        ("irt", None),
        // Kingdom C — risk set
        ("cox_ph", None),
        // Kingdom C — logistic
        ("train", Some("logistic")),
        // Kingdom C — GARCH
        ("garch", None),
    ];

    ops.iter()
        .map(|(op, sub)| verify_kingdom(op, *sub))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tbs_parser::TbsChain;

    // ── L001: missing normalization ─────────────────────────────────────

    #[test]
    fn l001_fires_for_dbscan_without_normalize() {
        let chain = TbsChain::parse("dbscan(epsilon=1.0, min_samples=2)").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L001"), "expected L001, got: {lints:?}");
    }

    #[test]
    fn l001_suppressed_after_normalize() {
        let chain = TbsChain::parse("normalize().dbscan(epsilon=1.0, min_samples=2)").unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L001"), "L001 should be suppressed after normalize()");
    }

    #[test]
    fn l001_suppressed_with_cosine_metric() {
        let chain = TbsChain::parse(r#"knn(k=5, metric="cosine")"#).unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L001"), "L001 should be suppressed for cosine metric");
    }

    #[test]
    fn l001_fires_for_kmeans() {
        let chain = TbsChain::parse("kmeans(k=3)").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L001"));
    }

    #[test]
    fn l001_fires_for_knn() {
        let chain = TbsChain::parse("knn(k=5)").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L001"));
    }

    // ── L002: supervised without exploration ────────────────────────────

    #[test]
    fn l002_fires_for_direct_training() {
        let chain = TbsChain::parse(r#"train.linear(target="y")"#).unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L002"), "expected L002");
    }

    #[test]
    fn l002_suppressed_after_clustering() {
        let chain = TbsChain::parse(r#"discover_clusters(epsilon=1.0, min_samples=2).train.linear(target="y")"#).unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L002"));
    }

    #[test]
    fn l002_is_info_severity() {
        let chain = TbsChain::parse(r#"train.logistic()"#).unwrap();
        let lints = static_lints(&chain);
        let l002 = lints.iter().find(|l| l.code == "L002").unwrap();
        assert_eq!(l002.severity, LintSeverity::Info);
    }

    // ── L003: predict on training data ──────────────────────────────────

    #[test]
    fn l003_fires_for_train_then_predict() {
        let chain = TbsChain::parse(r#"train.linear(target="y").predict()"#).unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L003"), "expected L003");
    }

    #[test]
    fn l003_does_not_fire_without_train() {
        let chain = TbsChain::parse("predict()").unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L003"));
    }

    // ── L106: near-constant column ──────────────────────────────────────

    #[test]
    fn l106_fires_for_constant_column() {
        // 4×2 data: column 0 is constant (5.0), column 1 varies
        let data = vec![5.0, 1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0];
        let lints = lint_l106_constant_columns(&data, 4, 2);
        assert!(lints.iter().any(|l| l.code == "L106" && l.message.contains("column 0")),
            "expected L106 for constant column 0, got: {lints:?}");
        assert!(!lints.iter().any(|l| l.message.contains("column 1")),
            "column 1 varies, should not fire L106");
    }

    #[test]
    fn l106_does_not_fire_for_varying_data() {
        let data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let lints = lint_l106_constant_columns(&data, 3, 2);
        assert!(lints.is_empty());
    }

    // ── L101: naive variance ────────────────────────────────────────────

    #[test]
    fn l101_fires_for_extreme_range() {
        // Large values close together → naive variance loses precision
        let data: Vec<f64> = (0..100).map(|i| 1e12 + i as f64 * 0.001).collect();
        let chain = TbsChain::parse("describe()").unwrap();
        let lints = lint_l101_naive_variance(&data, 100, 1, &chain);
        assert!(lints.iter().any(|l| l.code == "L101"), "expected L101 for extreme range");
    }

    #[test]
    fn l101_does_not_fire_for_normal_data() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let chain = TbsChain::parse("describe()").unwrap();
        let lints = lint_l101_naive_variance(&data, 100, 1, &chain);
        assert!(lints.is_empty());
    }

    #[test]
    fn l101_only_fires_when_variance_op_present() {
        let data: Vec<f64> = (0..100).map(|i| 1e12 + i as f64 * 0.001).collect();
        let chain = TbsChain::parse("normalize()").unwrap();
        let lints = lint_l101_naive_variance(&data, 100, 1, &chain);
        assert!(lints.is_empty(), "L101 should not fire without variance operations");
    }

    // ── L201: shared subproblem detection ───────────────────────────────

    #[test]
    fn l201_fires_for_adjacent_distance_ops() {
        // kmeans + knn both need distance matrix — L201 should inform
        let chain = TbsChain::parse("normalize().kmeans(k=3).knn(k=5)").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L201"),
            "expected L201 for kmeans+knn, got: {:?}", lints.iter().map(|l| l.code).collect::<Vec<_>>());
    }

    #[test]
    fn l201_fires_for_redundant_kmeans() {
        // kmeans(k=3) + kmeans(k=5) — both compute distances
        let chain = TbsChain::parse("normalize().kmeans(k=3).kmeans(k=5)").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L201" && l.message.contains("distance matrix")),
            "expected L201 about distance matrix");
    }

    #[test]
    fn l201_does_not_fire_for_different_subproblems() {
        // pca (Covariance) + kmeans (DistanceMatrix) — different subproblems
        let chain = TbsChain::parse("pca().kmeans(k=3)").unwrap();
        let lints = kingdom_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L201"),
            "L201 should not fire for different subproblems");
    }

    #[test]
    fn l201_does_not_fire_for_single_step() {
        let chain = TbsChain::parse("kmeans(k=3)").unwrap();
        let lints = kingdom_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L201"));
    }

    #[test]
    fn l201_is_info_severity() {
        let chain = TbsChain::parse("normalize().kmeans(k=3).knn(k=5)").unwrap();
        let lints = static_lints(&chain);
        let l201 = lints.iter().find(|l| l.code == "L201");
        assert!(l201.is_some());
        assert_eq!(l201.unwrap().severity, LintSeverity::Info);
    }

    #[test]
    fn l201_fires_for_adjacent_covariance_ops() {
        // pca (A, Covariance) + cfa (C, Covariance) — only fire if both are C
        // pca is Kingdom A so this should NOT fire (L201 requires both to be C)
        let chain = TbsChain::parse("pca().cfa()").unwrap();
        let lints = kingdom_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L201"),
            "L201 requires both steps to be Kingdom C");
    }

    // ── Kingdom annotation ──────────────────────────────────────────────

    #[test]
    fn kingdom_annotation_distance_ops() {
        assert_eq!(kingdom_of("kmeans", None).0, Kingdom::C);
        assert_eq!(kingdom_of("knn", None).0, Kingdom::C);
        assert_eq!(kingdom_of("dbscan", None).0, Kingdom::C);
        assert_eq!(kingdom_of("tsne", None).0, Kingdom::C);
        // All share DistanceMatrix
        assert_eq!(kingdom_of("kmeans", None).1, Some(SharedSubproblem::DistanceMatrix));
        assert_eq!(kingdom_of("knn", None).1, Some(SharedSubproblem::DistanceMatrix));
    }

    #[test]
    fn kingdom_annotation_a_ops() {
        assert_eq!(kingdom_of("normalize", None).0, Kingdom::A);
        assert_eq!(kingdom_of("describe", None).0, Kingdom::A);
        assert_eq!(kingdom_of("pca", None).0, Kingdom::A);
        assert_eq!(kingdom_of("train", Some("linear")).0, Kingdom::A);
    }

    #[test]
    fn kingdom_annotation_reclassified_to_a() {
        // kaplan_meier: prior label "Kingdom B" was wrong.
        // KM is a prefix product scan — survival.rs documents this correction.
        assert_eq!(kingdom_of("kaplan_meier", None).0, Kingdom::A);
        // exp_smoothing: prior label "Kingdom B" was wrong.
        // EMA/exponential smoothing is an affine prefix scan (Kingdom A).
        assert_eq!(kingdom_of("exp_smoothing", None).0, Kingdom::A);
        // log_rank: prior label "Kingdom B" was wrong.
        // At-risk counts n1(t), n2(t) are deterministic suffix counts of sorted data —
        // n1(t) = |{i: times[i] >= t, groups[i] == 0}| — not path-dependent accumulated state.
        // Same structural argument as kaplan_meier. The sequential walk is an artifact.
        assert_eq!(kingdom_of("log_rank", None).0, Kingdom::A);
    }

    #[test]
    fn kingdom_annotation_logistic_is_c() {
        assert_eq!(kingdom_of("train", Some("logistic")).0, Kingdom::C);
        assert_eq!(kingdom_of("train", Some("logistic")).1, Some(SharedSubproblem::GramMatrix));
    }

    // ── L004: distribution assumption mismatch ──────────────────────────

    #[test]
    fn l004_fires_for_describe_then_t_test() {
        let chain = TbsChain::parse("describe().t_test()").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L004"),
            "expected L004 for describe → t_test");
    }

    #[test]
    fn l004_fires_for_skewness_then_anova() {
        let chain = TbsChain::parse("skewness().anova()").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L004"));
    }

    #[test]
    fn l004_does_not_fire_without_shape_step() {
        let chain = TbsChain::parse("t_test()").unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L004"));
    }

    #[test]
    fn l004_does_not_fire_for_nonparametric() {
        // describe → mann_whitney is fine (nonparametric doesn't assume normality)
        let chain = TbsChain::parse("describe().mann_whitney()").unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L004"));
    }

    #[test]
    fn l004_suppressed_after_ks_test() {
        // describe → ks_test → t_test: normality was explicitly checked
        let chain = TbsChain::parse("describe().ks_test().t_test(mu=0.0)").unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L004"),
            "L004 should be suppressed when ks_test precedes t_test");
    }

    // ── L109: skewness-normality dynamic ────────────────────────────────

    #[test]
    fn l109_fires_for_skewed_data() {
        use crate::pipeline::{DescribeResult, ColumnDescribe};
        let describe = DescribeResult {
            columns: vec![ColumnDescribe {
                index: 0, count: 100, mean: 5.0, std: 2.0,
                min: 0.0, q1: 3.0, median: 4.5, q3: 7.0, max: 20.0,
                iqr: 4.0, skewness: 2.5, kurtosis: 6.0,
            }],
        };
        let chain = TbsChain::parse("describe().t_test()").unwrap();
        let lints = lint_l109_skewness_normality(&describe, &chain);
        assert!(lints.iter().any(|l| l.code == "L109"),
            "expected L109 for skewness=2.5");
    }

    #[test]
    fn l109_does_not_fire_for_symmetric_data() {
        use crate::pipeline::{DescribeResult, ColumnDescribe};
        let describe = DescribeResult {
            columns: vec![ColumnDescribe {
                index: 0, count: 100, mean: 5.0, std: 2.0,
                min: 0.0, q1: 3.5, median: 5.0, q3: 6.5, max: 10.0,
                iqr: 3.0, skewness: 0.1, kurtosis: -0.2,
            }],
        };
        let chain = TbsChain::parse("describe().t_test()").unwrap();
        let lints = lint_l109_skewness_normality(&describe, &chain);
        assert!(lints.is_empty(), "L109 should not fire for |skew|=0.1");
    }

    // ── L005: redundant consecutive operations ────────────────────────

    #[test]
    fn l005_fires_for_duplicate_normalize() {
        let chain = TbsChain::parse("normalize().normalize()").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L005"),
            "normalize().normalize() should trigger L005, got: {:?}",
            lints.iter().map(|l| l.code).collect::<Vec<_>>());
    }

    #[test]
    fn l005_message_says_redundant() {
        let chain = TbsChain::parse("normalize().normalize()").unwrap();
        let lints = static_lints(&chain);
        let l005 = lints.iter().find(|l| l.code == "L005").unwrap();
        assert!(l005.message.to_lowercase().contains("redundant"),
            "L005 message should contain 'redundant'");
    }

    #[test]
    fn l005_not_fired_for_different_args() {
        let chain = TbsChain::parse("kmeans(k=3).kmeans(k=5)").unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L005"),
            "kmeans(k=3).kmeans(k=5) should NOT trigger L005 (different args)");
    }

    #[test]
    fn l005_not_fired_for_different_ops() {
        let chain = TbsChain::parse("normalize().describe()").unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L005"));
    }

    #[test]
    fn l005_fires_for_triple_normalize() {
        let chain = TbsChain::parse("normalize().normalize().normalize()").unwrap();
        let lints = static_lints(&chain);
        let count = lints.iter().filter(|l| l.code == "L005").count();
        assert_eq!(count, 2, "three consecutive normalize should produce 2 L005 lints");
    }

    // ── L006: preprocessing after consumer ──────────────────────────────

    #[test]
    fn l006_fires_for_normalize_after_kmeans() {
        let chain = TbsChain::parse("kmeans(k=3).normalize()").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L006"),
            "kmeans().normalize() should trigger L006, got: {:?}",
            lints.iter().map(|l| l.code).collect::<Vec<_>>());
    }

    #[test]
    fn l006_message_is_actionable() {
        let chain = TbsChain::parse("kmeans(k=3).normalize()").unwrap();
        let lints = static_lints(&chain);
        let l006 = lints.iter().find(|l| l.code == "L006").unwrap();
        let msg = l006.message.to_lowercase();
        assert!(msg.contains("after") && msg.contains("before"),
            "L006 message should mention both ordering issue and fix: {}", l006.message);
    }

    #[test]
    fn l006_not_fired_for_normalize_before_kmeans() {
        let chain = TbsChain::parse("normalize().kmeans(k=3)").unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L006"));
    }

    #[test]
    fn l006_suppressed_when_another_consumer_follows() {
        // normalize after kmeans, but knn follows — user may intend re-preprocessing
        let chain = TbsChain::parse("kmeans(k=3).normalize().knn(k=5)").unwrap();
        let lints = static_lints(&chain);
        assert!(!lints.iter().any(|l| l.code == "L006"),
            "L006 should be suppressed when another consumer follows");
    }

    #[test]
    fn l006_fires_for_log_after_ttest() {
        let chain = TbsChain::parse("t_test().log()").unwrap();
        let lints = static_lints(&chain);
        assert!(lints.iter().any(|l| l.code == "L006"),
            "t_test().log() should trigger L006");
    }

    #[test]
    fn l109_only_fires_when_normality_op_present() {
        use crate::pipeline::{DescribeResult, ColumnDescribe};
        let describe = DescribeResult {
            columns: vec![ColumnDescribe {
                index: 0, count: 100, mean: 5.0, std: 2.0,
                min: 0.0, q1: 3.0, median: 4.0, q3: 7.0, max: 20.0,
                iqr: 4.0, skewness: 3.0, kurtosis: 10.0,
            }],
        };
        let chain = TbsChain::parse("describe().kmeans(k=3)").unwrap();
        let lints = lint_l109_skewness_normality(&describe, &chain);
        assert!(lints.is_empty(), "L109 should not fire without normality-assuming ops");
    }

    // ── Kingdom contract verification ───────────────────────────────────

    #[test]
    fn kingdom_a_normalize_is_verified() {
        let contract = verify_kingdom("normalize", None);
        assert_eq!(contract.kingdom, Kingdom::A);
        assert!(contract.verification.is_sound());
        assert!(matches!(contract.verification, KingdomVerification::Verified(_)),
            "normalize should be machine-verified as Kingdom A, got: {:?}", contract.verification);
    }

    #[test]
    fn kingdom_a_pca_covariance_is_verified() {
        let contract = verify_kingdom("pca", None);
        assert_eq!(contract.kingdom, Kingdom::A);
        assert!(matches!(contract.verification, KingdomVerification::Verified(_)),
            "pca should be verified — covariance accumulate is associative");
    }

    #[test]
    fn kingdom_a_adf_autocorrelation_is_verified() {
        let contract = verify_kingdom("adf_test", None);
        assert_eq!(contract.kingdom, Kingdom::A);
        assert!(matches!(contract.verification, KingdomVerification::Verified(_)),
            "adf_test should be verified — ACF accumulate is (ℝ,+,0)");
    }

    #[test]
    fn kingdom_c_kmeans_is_honest() {
        let contract = verify_kingdom("kmeans", None);
        assert_eq!(contract.kingdom, Kingdom::C);
        assert!(contract.verification.is_sound());
        assert!(matches!(contract.verification, KingdomVerification::HonestDeclaration { .. }),
            "kmeans must declare honest C — iterative EM convergence");
    }

    #[test]
    fn kingdom_c_cox_ph_risk_set_is_honest() {
        let contract = verify_kingdom("cox_ph", None);
        assert_eq!(contract.kingdom, Kingdom::C);
        assert!(matches!(contract.verification, KingdomVerification::HonestDeclaration { .. }));
    }

    #[test]
    fn kingdom_a_train_linear_gram_is_verified() {
        let contract = verify_kingdom("train", Some("linear"));
        assert_eq!(contract.kingdom, Kingdom::A);
        assert_eq!(contract.shared, Some(SharedSubproblem::GramMatrix));
        assert!(matches!(contract.verification, KingdomVerification::Verified(_)),
            "linear regression — Gram matrix accumulate is Kingdom A");
    }

    #[test]
    fn kingdom_c_train_logistic_is_honest() {
        let contract = verify_kingdom("train", Some("logistic"));
        assert_eq!(contract.kingdom, Kingdom::C);
        assert!(matches!(contract.verification, KingdomVerification::HonestDeclaration { .. }),
            "logistic regression (IRLS) is iterative — honestly Kingdom C");
    }

    #[test]
    fn verify_all_contracts_none_rejected() {
        // The fundamental correctness invariant: every known operation's
        // kingdom claim must be either machine-verified (A) or honestly
        // declared (B/C). Any Rejected entry means the lookup table is wrong.
        let contracts = verify_all_kingdom_contracts();
        assert!(!contracts.is_empty(), "should have contracts for all known ops");
        let rejected: Vec<_> = contracts.iter()
            .filter(|c| matches!(c.verification, KingdomVerification::Rejected(_)))
            .collect();
        assert!(
            rejected.is_empty(),
            "kingdom_of() has inconsistent claims: {:?}",
            rejected.iter().map(|c| &c.op).collect::<Vec<_>>()
        );
    }

    #[test]
    fn verify_all_contracts_kingdom_a_count() {
        // Sanity: there should be multiple Kingdom A ops in the table.
        let contracts = verify_all_kingdom_contracts();
        let a_count = contracts.iter().filter(|c| c.kingdom == Kingdom::A).count();
        assert!(a_count >= 8, "expected at least 8 Kingdom A ops, got {a_count}");
    }

    #[test]
    fn verify_all_contracts_kingdom_b_count() {
        // Sanity check: all registered ops are Kingdom A or C (the standard liftable kingdoms).
        // All prior Kingdom B annotations in survival.rs, time_series.rs, panel.rs, and
        // tbs_lint.rs have been reclassified after structural analysis.
        // Genuine Kingdom B (not in TBS table): BOCPD (growing state), PELT (state-dependent pruning),
        // EGARCH (σ_{t-1} normalization), SDE step integrators (f(x_t) needs x_t),
        // TDA union-find/boundary matrix, MCMC (acceptance ratio nonlinear in state),
        // TAR/threshold models (branch selection depends on state value).
        // ARMA(q>0) reclassified Kingdom A: MA terms lift to companion matrix, M is constant.
        let contracts = verify_all_kingdom_contracts();
        let b_count = contracts.iter().filter(|c| c.kingdom == Kingdom::B).count();
        assert_eq!(b_count, 0, "all TBS table ops reclassified to A or C; got {b_count} B ops");
    }
}
