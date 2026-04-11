//! Layer 1 advice types — recommendation + diagnostics + user-override record.
//!
//! These types are produced by auto-detection logic in [`crate::tbs_autodetect`]
//! and surfaced in [`crate::tbs_executor::TbsResult`].  Keeping them in a
//! dedicated module breaks the otherwise-circular dependency:
//!
//! ```text
//! tbs_executor  ──imports──▶  tbs_advice   ◀──imports──  tbs_autodetect
//!      │                                                        ▲
//!      └────────────────────imports────────────────────────────┘
//! ```

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
