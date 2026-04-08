//! # Using — Conscious override of tambear defaults
//!
//! `.using()` says "I know what I want here." It overrides tambear's intelligent
//! defaults for the next step. Chainable. Accumulative. Consumed after use.
//!
//! `.discover()` is its complement — "find me the best one."
//!
//! ```text
//! tb.efa
//!   .using(correlation="polychoric")    // I know this
//!   .discover(rotation)                  // find me the best rotation
//!   .model(...)                          // define the model
//!   .input(...)                          // bind data
//!   .output(...)                         // where results go
//! ```
//!
//! The pair encodes the user's epistemic state at each point:
//! - `using()` = "I have knowledge"
//! - `discover()` = "I need knowledge"
//!
//! ```text
//! // I know everything
//! tb.efa.using(correlation="polychoric", rotation="promax", factors=4)
//!
//! // I know nothing
//! tb.efa.discover(correlation, rotation, factors)
//!
//! // I know some, discover the rest
//! tb.efa.using(correlation="polychoric").discover(rotation, factors)
//! ```
//!
//! ## What can be overridden
//!
//! Anything a method has a default for. The bag is open-ended — methods
//! query it for keys they care about and ignore the rest.

use std::collections::HashMap;

/// A bag of overrides accumulated by one or more `.using()` calls.
/// Consumed by the next computation step, then cleared.
#[derive(Debug, Clone, Default)]
pub struct UsingBag {
    overrides: HashMap<String, UsingValue>,
}

/// A using value — mirrors TbsValue but lives in the runtime.
#[derive(Debug, Clone, PartialEq)]
pub enum UsingValue {
    Str(String),
    Float(f64),
    Int(i64),
    Bool(bool),
}

impl UsingValue {
    pub fn as_str(&self) -> Option<&str> {
        match self { UsingValue::Str(s) => Some(s), _ => None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            UsingValue::Float(f) => Some(*f),
            UsingValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self { UsingValue::Bool(b) => Some(*b), _ => None }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self { UsingValue::Int(i) => Some(*i), _ => None }
    }
}

impl UsingBag {
    pub fn new() -> Self {
        Self { overrides: HashMap::new() }
    }

    /// Add an override. Later using() calls overwrite earlier ones for the same key.
    pub fn set(&mut self, key: impl Into<String>, value: UsingValue) {
        self.overrides.insert(key.into(), value);
    }

    /// Query an override. Returns None if not set (method uses its default).
    pub fn get(&self, key: &str) -> Option<&UsingValue> {
        self.overrides.get(key)
    }

    /// Query a string override.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }

    /// Query a float override.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    /// Query a bool override.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }

    /// True if the bag has any overrides.
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    /// Number of overrides in the bag.
    pub fn len(&self) -> usize {
        self.overrides.len()
    }

    /// Drain the bag — returns the overrides and clears the bag.
    /// Called by the executor after each computation step.
    pub fn drain(&mut self) -> HashMap<String, UsingValue> {
        std::mem::take(&mut self.overrides)
    }

    /// Clear without returning.
    pub fn clear(&mut self) {
        self.overrides.clear();
    }

    // -----------------------------------------------------------------------
    // Convenience: typed queries for common override keys
    // -----------------------------------------------------------------------

    /// Precision override: "fp32"/"f32" → F32, "fp64"/"f64" → F64.
    pub fn precision(&self) -> Option<crate::codegen::Precision> {
        match self.get_str("precision")? {
            "fp32" | "f32" => Some(crate::codegen::Precision::F32),
            "fp64" | "f64" => Some(crate::codegen::Precision::F64),
            _ => None,
        }
    }

    /// NaN policy override.
    pub fn nan_policy(&self) -> Option<crate::nan_guard::NanPolicy> {
        let s = self.get_str("nan")?;
        match s {
            "trim" | "omit" => Some(crate::nan_guard::NanPolicy::Omit),
            "kill" | "reject" | "error" => Some(crate::nan_guard::NanPolicy::Reject),
            s if s.starts_with("fill:") => {
                s[5..].parse::<f64>().ok().map(crate::nan_guard::NanPolicy::Replace)
            }
            _ => None,
        }
    }

    /// GPU-only override: using(gpu_only=true).
    pub fn gpu_only(&self) -> Option<bool> {
        self.get_bool("gpu_only")
    }

    /// Sweep override: using(sweep=false) to disable parameter sweep.
    pub fn sweep(&self) -> Option<bool> {
        self.get_bool("sweep")
    }

    /// Superposition override: using(superposition=false) to disable.
    pub fn superposition(&self) -> Option<bool> {
        self.get_bool("superposition")
    }

    /// Rotation override: using(rotation="promax") / "varimax" / "oblimin".
    pub fn rotation(&self) -> Option<&str> {
        self.get_str("rotation")
    }

    /// Clustering method override: using(method="kmeans") instead of tb default.
    pub fn method(&self) -> Option<&str> {
        self.get_str("method")
    }

    /// Stats mode override: using(stats="sufficient") for lossy compression.
    /// Default (None) = full data, full math.
    pub fn stats_mode(&self) -> Option<&str> {
        self.get_str("stats")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bag() {
        let bag = UsingBag::new();
        assert!(bag.is_empty());
        assert_eq!(bag.precision(), None);
        assert_eq!(bag.nan_policy(), None);
    }

    #[test]
    fn set_and_query() {
        let mut bag = UsingBag::new();
        bag.set("precision", UsingValue::Str("fp32".into()));
        bag.set("nan", UsingValue::Str("kill".into()));
        assert_eq!(bag.precision(), Some(crate::codegen::Precision::F32));
        assert_eq!(bag.nan_policy(), Some(crate::nan_guard::NanPolicy::Reject));
    }

    #[test]
    fn later_using_overwrites() {
        let mut bag = UsingBag::new();
        bag.set("precision", UsingValue::Str("fp32".into()));
        bag.set("precision", UsingValue::Str("fp64".into()));
        assert_eq!(bag.precision(), Some(crate::codegen::Precision::F64));
    }

    #[test]
    fn drain_clears() {
        let mut bag = UsingBag::new();
        bag.set("precision", UsingValue::Str("fp32".into()));
        bag.set("nan", UsingValue::Str("trim".into()));
        let drained = bag.drain();
        assert_eq!(drained.len(), 2);
        assert!(bag.is_empty());
    }

    #[test]
    fn nan_fill_parse() {
        let mut bag = UsingBag::new();
        bag.set("nan", UsingValue::Str("fill:0.0".into()));
        assert_eq!(bag.nan_policy(), Some(crate::nan_guard::NanPolicy::Replace(0.0)));
    }

    #[test]
    fn bool_overrides() {
        let mut bag = UsingBag::new();
        bag.set("gpu_only", UsingValue::Bool(true));
        bag.set("sweep", UsingValue::Bool(false));
        assert_eq!(bag.gpu_only(), Some(true));
        assert_eq!(bag.sweep(), Some(false));
    }

    #[test]
    fn unknown_keys_return_none() {
        let bag = UsingBag::new();
        assert_eq!(bag.get("nonexistent"), None);
        assert_eq!(bag.rotation(), None);
        assert_eq!(bag.method(), None);
    }

    #[test]
    fn multiple_using_accumulate() {
        let mut bag = UsingBag::new();
        bag.set("precision", UsingValue::Str("fp32".into()));
        bag.set("nan", UsingValue::Str("kill".into()));
        bag.set("rotation", UsingValue::Str("promax".into()));
        bag.set("sweep", UsingValue::Bool(false));

        assert_eq!(bag.len(), 4);
        assert_eq!(bag.precision(), Some(crate::codegen::Precision::F32));
        assert_eq!(bag.nan_policy(), Some(crate::nan_guard::NanPolicy::Reject));
        assert_eq!(bag.rotation(), Some("promax"));
        assert_eq!(bag.sweep(), Some(false));
    }

    #[test]
    fn stats_mode_default_is_full() {
        let bag = UsingBag::new();
        assert_eq!(bag.stats_mode(), None); // None = full data, full math
    }

    #[test]
    fn stats_mode_sufficient() {
        let mut bag = UsingBag::new();
        bag.set("stats", UsingValue::Str("sufficient".into()));
        assert_eq!(bag.stats_mode(), Some("sufficient"));
    }
}
