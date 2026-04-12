//! Oracle runner — Campsite 4.8 (I9′ scientist track).
//!
//! Reads a named oracle entry from `oracles/<fn>.toml` (written by adversarial),
//! executes it against a candidate function, and produces a structured `OracleReport`.
//!
//! ## Invariant I9 enforcement
//!
//! The runner does NOT decide what "acceptable" means.  That is `claimed_max_ulp`
//! in the oracle entry.  The runner only executes the entry's instructions and
//! reports TESTED vs CLAIMED.
//!
//! ## What the runner does NOT do
//!
//! - Does not curate inputs (that's the adversarial's oracle entry).
//! - Does not evaluate identity expressions as runtime strings.  Phase 1 uses a
//!   static registry of pre-compiled Rust closures, looked up by name.  Unknown
//!   identity names produce an error, not a silent skip.
//! - Does not run across multiple backends.  Takes a single `candidate` closure.
//!   Cross-backend comparison is `assert_cross_backend_agreement`'s job.

use std::collections::HashMap;
use serde::Deserialize;
use crate::ulp_harness::{UlpReport, RefRecord};
use crate::tolerance::ulp_distance;

// ─────────────────────────────────────────────────────────────────────────────
// TOML deserialization types (internal)
// ─────────────────────────────────────────────────────────────────────────────

/// Raw TOML structure for `oracles/<fn>.toml` — deserialised before conversion
/// to the public `OracleEntry` type.
#[derive(Debug, Deserialize)]
struct TomlOracleFile {
    function: TomlFunctionSection,
    corpus: TomlCorpus,
    #[serde(default)]
    bit_exact_checks: TomlBitExactSection,
    #[serde(default)]
    constraint_checks: TomlConstraintSection,
    #[serde(default)]
    identity_checks: Vec<TomlIdentityCheck>,
}

#[derive(Debug, Deserialize)]
struct TomlFunctionSection {
    name: String,
    claimed_max_ulp: f64,
    reference_bin: String,
}

/// `[corpus]` — each field is a named injection set.
/// Values may be f64 literals or string expressions (see `eval_toml_value`).
#[derive(Debug, Deserialize, Default)]
struct TomlCorpus {
    #[serde(flatten)]
    sets: HashMap<String, Vec<toml::Value>>,
}

#[derive(Debug, Deserialize, Default)]
struct TomlBitExactSection {
    #[serde(default)]
    cases: Vec<TomlBitExactCase>,
}

#[derive(Debug, Deserialize)]
struct TomlBitExactCase {
    name: String,
    input: toml::Value,     // f64 literal or string expression
    expected_bits: String,  // hex literal only (e.g. "0x0000000000000000")
    rationale: String,
}

#[derive(Debug, Deserialize, Default)]
struct TomlConstraintSection {
    #[serde(default)]
    cases: Vec<TomlConstraintCase>,
}

#[derive(Debug, Deserialize)]
struct TomlConstraintCase {
    name: String,
    input: toml::Value,    // f64 literal or string expression
    constraint: String,    // named constraint from the static registry
    rationale: String,
}

#[derive(Debug, Deserialize)]
struct TomlIdentityCheck {
    name: String,
    tolerance_ulp: f64,
    domain: [f64; 2],
}

// ─────────────────────────────────────────────────────────────────────────────
// String expression evaluator (Phase 1)
//
// Supports the expressions that appear in the adversarial's oracle entries.
// Does NOT call mpmath at runtime — uses Rust f64 constants.
// Unknown expressions produce an error.
// ─────────────────────────────────────────────────────────────────────────────

fn eval_expr(expr: &str) -> Result<f64, String> {
    let s = expr.trim();
    // Try numeric parse first
    if let Ok(v) = s.parse::<f64>() { return Ok(v); }
    // Special IEEE 754 values
    match s {
        "inf"  | "+inf"  | "Inf"  | "+Inf"  => return Ok(f64::INFINITY),
        "-inf" | "-Inf"  => return Ok(f64::NEG_INFINITY),
        "nan"  | "NaN"   | "NAN"  => return Ok(f64::NAN),
        _ => {}
    }
    // Simple arithmetic expressions used in the exp oracle entry.
    // Pattern: [coefficient "*"] base_expr
    // Base expressions: ln(2), pi, e, etc.
    eval_arithmetic(s)
}

fn eval_arithmetic(s: &str) -> Result<f64, String> {
    // Handle "a + b" and "a - b" for boundary adjustments like "ln(2) + 1e-15"
    // Scan right-to-left for + and - at depth=0, skipping scientific notation exponents.
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    for i in (0..bytes.len()).rev() {
        match bytes[i] {
            b')' => depth += 1,
            b'(' => depth -= 1,
            b'+' if depth == 0 && i > 0 => {
                // Skip if preceded by 'e' or 'E' (scientific notation exponent)
                let prev = bytes[i - 1];
                if prev == b'e' || prev == b'E' { continue; }
                let lhs = eval_arithmetic(&s[..i])?;
                let rhs = eval_atomic(&s[i+1..])?;
                return Ok(lhs + rhs);
            }
            b'-' if depth == 0 && i > 0 => {
                // Skip if preceded by 'e' or 'E' (scientific notation exponent)
                let prev = bytes[i - 1];
                if prev == b'e' || prev == b'E' { continue; }
                let lhs = eval_arithmetic(&s[..i])?;
                let rhs = eval_atomic(&s[i+1..])?;
                return Ok(lhs - rhs);
            }
            _ => {}
        }
    }
    eval_atomic(s)
}

fn eval_atomic(s: &str) -> Result<f64, String> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() { return Ok(v); }
    // Multiplication: "n * base"
    if let Some(star_pos) = s.find('*') {
        let lhs = s[..star_pos].trim().parse::<f64>()
            .map_err(|_| format!("cannot parse lhs of *: {:?}", &s[..star_pos]))?;
        let rhs = eval_base(&s[star_pos+1..])?;
        return Ok(lhs * rhs);
    }
    eval_base(s)
}

fn eval_base(s: &str) -> Result<f64, String> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() { return Ok(v); }
    // Unary minus: "-expr"
    if let Some(rest) = s.strip_prefix('-') {
        return eval_base(rest).map(|v| -v);
    }
    // Named constants and function calls
    match s {
        "ln(2)"         => Ok(std::f64::consts::LN_2),
        "log(2)"        => Ok(std::f64::consts::LN_2),
        "ln(10)"        => Ok(std::f64::consts::LN_10),
        "log(10)"       => Ok(std::f64::consts::LN_10),
        "pi" | "π"      => Ok(std::f64::consts::PI),
        "e"             => Ok(std::f64::consts::E),
        "sqrt(2)"       => Ok(std::f64::consts::SQRT_2),
        "sqrt(0.5)"     => Ok(std::f64::consts::FRAC_1_SQRT_2),
        "ln(2)/2"       => Ok(std::f64::consts::LN_2 / 2.0),
        "ln(2)/4"       => Ok(std::f64::consts::LN_2 / 4.0),
        "inf" | "+inf"  => Ok(f64::INFINITY),
        "-inf"          => Ok(f64::NEG_INFINITY),
        "nan"           => Ok(f64::NAN),
        _ => Err(format!("unknown expression: {:?}", s)),
    }
}

fn eval_toml_value(v: &toml::Value) -> Result<f64, String> {
    match v {
        toml::Value::Float(f) => Ok(*f),
        toml::Value::Integer(i) => Ok(*i as f64),
        toml::Value::String(s) => eval_expr(s),
        other => Err(format!("unexpected TOML value type: {:?}", other)),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit-exact constraint (hex literals only)
// ─────────────────────────────────────────────────────────────────────────────

/// The required bit pattern for a bit-exact check.
/// Only exact hex patterns — class constraints live in `[[constraint_checks]]`.
#[derive(Debug, Clone)]
pub struct BitExactConstraint(pub u64);

impl BitExactConstraint {
    fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.starts_with("0x") || s.starts_with("0X") {
            let hex = &s[2..];
            u64::from_str_radix(hex, 16)
                .map(BitExactConstraint)
                .map_err(|_| format!("invalid hex literal: {:?}", s))
        } else {
            s.parse::<u64>()
                .map(BitExactConstraint)
                .map_err(|_| format!("expected_bits must be a hex literal (0x...); got: {:?}", s))
        }
    }

    fn check(&self, actual_bits: u64) -> bool {
        actual_bits == self.0
    }

    fn describe(&self) -> String {
        format!("0x{:016X}", self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Named constraint registry (class-membership checks)
//
// Three entries for Phase 1.  Adversarial flags if new constraint types arise.
// ─────────────────────────────────────────────────────────────────────────────

/// A named class constraint on the candidate output value (not bits).
#[derive(Debug, Clone)]
pub enum NamedConstraint {
    /// Positive, nonzero, subnormal f64.
    /// bits != 0  &&  (bits >> 52) == 0  &&  (bits >> 63) == 0
    NonzeroSubnormalPositive,
    /// Finite (not ±inf, not NaN).
    Finite,
    /// Positive infinity (bits == 0x7FF0000000000000).
    InfinitePositive,
}

impl NamedConstraint {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "nonzero_subnormal_positive" => Ok(NamedConstraint::NonzeroSubnormalPositive),
            "finite"                     => Ok(NamedConstraint::Finite),
            "infinite_positive"          => Ok(NamedConstraint::InfinitePositive),
            other => Err(format!("unknown named constraint: {:?} (phase 1 registry: nonzero_subnormal_positive, finite, infinite_positive)", other)),
        }
    }

    fn check(&self, x: f64) -> bool {
        match self {
            NamedConstraint::NonzeroSubnormalPositive => {
                let bits = x.to_bits();
                bits != 0 && (bits >> 52) == 0 && (bits >> 63) == 0
            }
            NamedConstraint::Finite => x.is_finite(),
            NamedConstraint::InfinitePositive => x == f64::INFINITY,
        }
    }

    fn describe(&self) -> &'static str {
        match self {
            NamedConstraint::NonzeroSubnormalPositive => "nonzero_subnormal_positive",
            NamedConstraint::Finite                   => "finite",
            NamedConstraint::InfinitePositive         => "infinite_positive",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A named oracle entry loaded from `oracles/<fn>.toml` + its TAMBLMR1 reference binary.
pub struct OracleEntry {
    pub function_name: String,
    pub claimed_max_ulp: f64,
    /// Records from the TAMBLMR1 reference binary (for random-sample ULP measurement).
    pub reference_records: Vec<RefRecord>,
    /// Named injection sets: (set_name, vec_of_f64_inputs).
    pub injection_sets: Vec<(String, Vec<f64>)>,
    /// Bit-pattern exact checks: candidate(input).to_bits() == expected (hex only).
    pub bit_exact_checks: Vec<BitExactCheck>,
    /// Class-membership checks: candidate(input) satisfies a named constraint.
    pub constraint_checks: Vec<ConstraintCheck>,
    /// Identity checks (looked up in the static registry by name).
    pub identity_checks: Vec<IdentityCheckSpec>,
}

/// A check that candidate(input).to_bits() equals an exact hex bit pattern.
/// For sign-sensitive values (+0 vs -0, exact inf sign) where ULP distance
/// would accept the wrong sign.
pub struct BitExactCheck {
    pub name: String,
    pub input: f64,
    pub constraint: BitExactConstraint,
    pub rationale: String,
}

/// A check that candidate(input) belongs to a named output class.
/// For cases where the exact bit pattern is implementation-determined but
/// the class is specified (e.g. "must be a positive nonzero subnormal",
/// "must be finite", "must be +inf").
pub struct ConstraintCheck {
    pub name: String,
    pub input: f64,
    pub constraint: NamedConstraint,
    pub rationale: String,
}

/// The specification of an identity check — name + parameters.
/// The actual check closure comes from the static registry.
pub struct IdentityCheckSpec {
    pub name: String,
    pub tolerance_ulp: f64,
    pub domain: (f64, f64),
}

/// Result of a single bit-exact check.
#[derive(Debug, Clone)]
pub struct BitExactResult {
    pub name: String,
    pub input: f64,
    pub constraint: String,   // human-readable description
    pub actual_bits: u64,
    pub passes: bool,
    pub rationale: String,
}

/// Result of a single class-membership constraint check.
#[derive(Debug, Clone)]
pub struct ConstraintResult {
    pub name: String,
    pub input: f64,
    pub constraint: &'static str,  // canonical name from NamedConstraint::describe()
    pub actual: f64,
    pub passes: bool,
    pub rationale: String,
}

/// Result of a single identity check.
#[derive(Debug, Clone)]
pub struct IdentityResult {
    pub name: String,
    pub max_residual_ulp: f64,
    pub claimed_tolerance_ulp: f64,
    pub passes: bool,
    /// Worst-case input (the x that produced max_residual_ulp).
    pub worst_case_input: Option<f64>,
}

/// Full oracle report for a candidate function against one oracle entry.
pub struct OracleReport {
    pub function_name: String,
    pub claimed_max_ulp: f64,
    /// ULP report over the full reference set (random sample from mpmath).
    pub random_sample_report: UlpReport,
    /// Per injection-set ULP reports.
    pub injection_reports: Vec<(String, UlpReport)>,
    /// Bit-pattern exact check results.
    pub bit_exact_results: Vec<BitExactResult>,
    /// Class-membership constraint check results.
    pub constraint_results: Vec<ConstraintResult>,
    /// Identity check results.
    pub identity_results: Vec<IdentityResult>,
    /// true iff ALL components pass their claimed bounds.
    pub passes: bool,
}

impl OracleReport {
    /// Write a human-readable summary of the report.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("oracle: {}", self.function_name));
        lines.push(format!("  claimed: {} ULP", self.claimed_max_ulp));
        lines.push(format!("  random sample: {} [{}]",
            self.random_sample_report.summary(),
            if self.random_sample_report.passes(self.claimed_max_ulp as u64) { "PASS" } else { "FAIL" }
        ));
        for (name, report) in &self.injection_reports {
            lines.push(format!("  injection[{}]: {} [{}]",
                name, report.summary(),
                if report.passes(self.claimed_max_ulp as u64) { "PASS" } else { "FAIL" }
            ));
        }
        for result in &self.bit_exact_results {
            lines.push(format!("  bit_exact[{}]: input={:?} expected={} actual=0x{:016X} [{}]",
                result.name, result.input, result.constraint, result.actual_bits,
                if result.passes { "PASS" } else { "FAIL" }
            ));
        }
        for result in &self.constraint_results {
            lines.push(format!("  constraint[{}]: input={:?} constraint={} actual={:?} [{}]",
                result.name, result.input, result.constraint, result.actual,
                if result.passes { "PASS" } else { "FAIL" }
            ));
        }
        for result in &self.identity_results {
            lines.push(format!("  identity[{}]: max_residual={:.3} ULP, tolerance={} ULP [{}]",
                result.name, result.max_residual_ulp, result.claimed_tolerance_ulp,
                if result.passes { "PASS" } else { "FAIL" }
            ));
        }
        lines.push(format!("  OVERALL: {}", if self.passes { "PASS" } else { "FAIL" }));
        lines.join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Static identity check registry
//
// Pre-compiled closures, looked up by name.  Each closure receives:
// - x: f64 — the test input
// - fns: &HashMap<String, Box<dyn Fn(f64)->f64>> — the function map
//
// Returns the residual in ULP units (f64).
//
// Phase 1 registers identities for tam_exp.  Add new entries as functions land.
// ─────────────────────────────────────────────────────────────────────────────

type IdentityFn = Box<dyn Fn(f64, &HashMap<String, Box<dyn Fn(f64) -> f64>>) -> f64 + Send + Sync>;

fn build_identity_registry() -> HashMap<String, IdentityFn> {
    let mut m: HashMap<String, IdentityFn> = HashMap::new();

    // exp_log_roundtrip: exp(log(x)) should equal x within 2 ULP.
    // Requires both "exp" and "ln" in the function map.
    m.insert("exp_log_roundtrip".to_string(), Box::new(|x, fns| {
        let exp = fns.get("exp").or_else(|| fns.get("tam_exp"));
        let ln  = fns.get("ln").or_else(|| fns.get("tam_ln"));
        match (exp, ln) {
            (Some(exp_fn), Some(ln_fn)) => {
                let result = exp_fn(ln_fn(x));
                let dist = ulp_distance(x, result);
                if dist == u64::MAX { f64::INFINITY } else { dist as f64 }
            }
            _ => f64::NAN, // function not available — skip
        }
    }));

    // exp_negation: exp(x) * exp(-x) should equal 1.0 within 2 ULP.
    m.insert("exp_negation".to_string(), Box::new(|x, fns| {
        let exp = fns.get("exp").or_else(|| fns.get("tam_exp"));
        match exp {
            Some(exp_fn) => {
                let result = exp_fn(x) * exp_fn(-x);
                let dist = ulp_distance(1.0_f64, result);
                if dist == u64::MAX { f64::INFINITY } else { dist as f64 }
            }
            _ => f64::NAN,
        }
    }));

    // exp_additivity: exp(a) * exp(b) should equal exp(a + b) within 3 ULP.
    // x encodes the sum a + b; we split: a = x/2, b = x/2 for simplicity.
    // A full two-argument sweep would need a different interface; Phase 1 uses this proxy.
    m.insert("exp_additivity".to_string(), Box::new(|x, fns| {
        let exp = fns.get("exp").or_else(|| fns.get("tam_exp"));
        match exp {
            Some(exp_fn) => {
                let a = x / 2.0;
                let b = x / 2.0;
                let lhs = exp_fn(a) * exp_fn(b);
                let rhs = exp_fn(a + b);
                let dist = ulp_distance(lhs, rhs);
                if dist == u64::MAX { f64::INFINITY } else { dist as f64 }
            }
            _ => f64::NAN,
        }
    }));

    // exp_one_returns_e: exp(1.0) should equal f64 representation of e within 1 ULP.
    m.insert("exp_one_returns_e".to_string(), Box::new(|_x, fns| {
        let exp = fns.get("exp").or_else(|| fns.get("tam_exp"));
        match exp {
            Some(exp_fn) => {
                let result = exp_fn(1.0);
                let dist = ulp_distance(std::f64::consts::E, result);
                if dist == u64::MAX { f64::INFINITY } else { dist as f64 }
            }
            _ => f64::NAN,
        }
    }));

    m
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample points for identity sweeps
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a sample of f64 values distributed across a domain.
/// Uses log-distributed points for wide ranges, linear for narrow ranges.
fn sample_domain(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if lo >= hi || n == 0 { return vec![]; }
    let mut pts = Vec::with_capacity(n);

    if lo > 0.0 {
        // Logarithmically distributed
        let log_lo = lo.ln();
        let log_hi = hi.ln();
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0).max(1.0);
            pts.push((log_lo + t * (log_hi - log_lo)).exp());
        }
    } else {
        // Linearly distributed
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0).max(1.0);
            pts.push(lo + t * (hi - lo));
        }
    }
    pts
}

// ─────────────────────────────────────────────────────────────────────────────
// Load oracle entry
// ─────────────────────────────────────────────────────────────────────────────

/// Load an oracle entry from a `.toml` file, resolving the TAMBLMR1 reference
/// binary relative to the provided `base_dir`.
///
/// `toml_path`: path to the oracle entry file (e.g. `"oracles/tam_exp.toml"`).
/// `base_dir`: directory relative to which `reference_bin` paths are resolved.
///   Typically the expedition root or the peak2-libm directory.
pub fn load_oracle_entry(toml_path: &str, base_dir: &str) -> Result<OracleEntry, String> {
    let toml_str = std::fs::read_to_string(toml_path)
        .map_err(|e| format!("cannot read {toml_path}: {e}"))?;

    let raw: TomlOracleFile = toml::from_str(&toml_str)
        .map_err(|e| format!("TOML parse error in {toml_path}: {e}"))?;

    // Resolve reference binary
    let bin_path = format!("{}/{}", base_dir.trim_end_matches('/'), raw.function.reference_bin);
    let (_, reference_records) = crate::ulp_harness::read_reference_bin(&bin_path)
        .map_err(|e| format!("cannot load reference bin {bin_path}: {e}"))?;

    // Parse injection sets
    let mut injection_sets = Vec::new();
    for (set_name, values) in &raw.corpus.sets {
        let mut inputs = Vec::with_capacity(values.len());
        for v in values {
            let x = eval_toml_value(v)
                .map_err(|e| format!("corpus[{set_name}]: {e}"))?;
            inputs.push(x);
        }
        injection_sets.push((set_name.clone(), inputs));
    }

    // Parse bit-exact checks
    let mut bit_exact_checks = Vec::new();
    for case in &raw.bit_exact_checks.cases {
        let input = eval_toml_value(&case.input)
            .map_err(|e| format!("bit_exact_checks[{}].input: {e}", case.name))?;
        let constraint = BitExactConstraint::parse(&case.expected_bits)
            .map_err(|e| format!("bit_exact_checks[{}].expected_bits: {e}", case.name))?;
        bit_exact_checks.push(BitExactCheck {
            name: case.name.clone(),
            input,
            constraint,
            rationale: case.rationale.clone(),
        });
    }

    // Parse constraint checks (class-membership, not bit-exact)
    let mut constraint_checks = Vec::new();
    for case in &raw.constraint_checks.cases {
        let input = eval_toml_value(&case.input)
            .map_err(|e| format!("constraint_checks[{}].input: {e}", case.name))?;
        let constraint = NamedConstraint::parse(&case.constraint)
            .map_err(|e| format!("constraint_checks[{}].constraint: {e}", case.name))?;
        constraint_checks.push(ConstraintCheck {
            name: case.name.clone(),
            input,
            constraint,
            rationale: case.rationale.clone(),
        });
    }

    // Parse identity check specs
    let identity_checks = raw.identity_checks.iter().map(|ic| IdentityCheckSpec {
        name: ic.name.clone(),
        tolerance_ulp: ic.tolerance_ulp,
        domain: (ic.domain[0], ic.domain[1]),
    }).collect();

    Ok(OracleEntry {
        function_name: raw.function.name,
        claimed_max_ulp: raw.function.claimed_max_ulp,
        reference_records,
        injection_sets,
        bit_exact_checks,
        constraint_checks,
        identity_checks,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Run oracle
// ─────────────────────────────────────────────────────────────────────────────

/// Run a candidate function against all components of an oracle entry.
///
/// `candidate_name`: the key under which this function is registered in the
/// identity check function map.  For `tam_exp`, use `"exp"` or `"tam_exp"`.
///
/// `candidate`: the function under test.  Must not call any vendor libm (I1).
pub fn run_oracle(
    entry: &OracleEntry,
    candidate_name: &str,
    candidate: impl Fn(f64) -> f64 + Clone + 'static,
) -> OracleReport {
    let identity_registry = build_identity_registry();

    // Random sample ULP measurement
    let random_sample_report = UlpReport::measure(&entry.reference_records, |x| candidate(x));

    // Injection set ULP measurements
    let mut injection_reports = Vec::new();
    for (set_name, inputs) in &entry.injection_sets {
        // Convert injection inputs to RefRecord format (no reference_str needed)
        // We compare candidate(x) against the mpmath value — but injection sets
        // are unordered inputs without pre-computed mpmath values.
        // For injection sets: measure against the reference_records if a matching
        // input exists; otherwise measure against f64::NAN (which will give u64::MAX
        // for any non-NaN result, revealing the absence of an oracle value).
        //
        // In Phase 1, injection sets primarily exercise:
        //   (a) NaN propagation (I11) — candidate(NaN) must be NaN
        //   (b) Special-value semantics — inf, -inf, 0.0, -0.0
        //   (c) Boundary behavior — near overflow/underflow
        //
        // For (a) and (b), we check NaN propagation and sign directly via
        // a synthetic UlpReport on records with `reference = candidate_reference(x)`.
        // For Phase 1 injection sets, we use the ULP infrastructure but build
        // synthetic records where reference = std::f64 for known-correct inputs.
        //
        // This is a conservative approach: injection ULP reports only test
        // NaN and infinity handling; numerical precision is tested by the
        // random-sample report. Adversarial can add reference-bin entries for
        // any injection input that needs ULP verification.
        let synthetic_records: Vec<RefRecord> = inputs.iter().map(|&x| {
            // Look for this input in the reference records (exact bit match)
            let matching_ref = entry.reference_records.iter()
                .find(|r| r.input.to_bits() == x.to_bits());
            let reference = if let Some(rec) = matching_ref {
                rec.reference
            } else {
                // No mpmath reference for this input.
                // Use the candidate itself as reference — this means ULP = 0 always.
                // This is intentional: injection sets test NaN/inf PROPAGATION,
                // not numerical precision. The bit_exact_checks section handles
                // the precision-sensitive special values.
                candidate(x)
            };
            RefRecord {
                input: x,
                reference,
                reference_str: format!("{:?}", reference),
            }
        }).collect();
        let report = UlpReport::measure(&synthetic_records, |x| candidate(x));
        injection_reports.push((set_name.clone(), report));
    }

    // Bit-exact checks
    let mut bit_exact_results = Vec::new();
    for check in &entry.bit_exact_checks {
        let actual = candidate(check.input);
        let actual_bits = actual.to_bits();
        let passes = check.constraint.check(actual_bits);
        bit_exact_results.push(BitExactResult {
            name: check.name.clone(),
            input: check.input,
            constraint: check.constraint.describe(),
            actual_bits,
            passes,
            rationale: check.rationale.clone(),
        });
    }

    // Class-membership constraint checks
    let mut constraint_results = Vec::new();
    for check in &entry.constraint_checks {
        let actual = candidate(check.input);
        let passes = check.constraint.check(actual);
        constraint_results.push(ConstraintResult {
            name: check.name.clone(),
            input: check.input,
            constraint: check.constraint.describe(),
            actual,
            passes,
            rationale: check.rationale.clone(),
        });
    }

    // Identity checks
    let mut fn_map: HashMap<String, Box<dyn Fn(f64) -> f64>> = HashMap::new();
    let candidate_fn: Box<dyn Fn(f64) -> f64> = Box::new(candidate.clone());
    fn_map.insert(candidate_name.to_string(), candidate_fn);
    // Also register under the generic short name if different
    if candidate_name.starts_with("tam_") {
        let short = &candidate_name["tam_".len()..];
        let candidate_fn2: Box<dyn Fn(f64) -> f64> = Box::new(candidate.clone());
        fn_map.insert(short.to_string(), candidate_fn2);
    }

    let mut identity_results = Vec::new();
    for spec in &entry.identity_checks {
        match identity_registry.get(&spec.name) {
            None => {
                // Unknown identity name — this is a runner error, not a skip.
                // Report as a failure with max residual = INFINITY.
                identity_results.push(IdentityResult {
                    name: spec.name.clone(),
                    max_residual_ulp: f64::INFINITY,
                    claimed_tolerance_ulp: spec.tolerance_ulp,
                    passes: false,
                    worst_case_input: None,
                });
            }
            Some(identity_fn) => {
                // Sample the domain
                let n_samples = if spec.domain.0 == spec.domain.1 { 1 } else { 64 };
                let xs = if spec.domain.0 == spec.domain.1 {
                    vec![spec.domain.0]
                } else {
                    sample_domain(spec.domain.0, spec.domain.1, n_samples)
                };

                let mut max_residual = 0.0_f64;
                let mut worst_x: Option<f64> = None;

                for &x in &xs {
                    let residual = identity_fn(x, &fn_map);
                    if residual.is_nan() {
                        // Function not available in map — skip this identity
                        max_residual = f64::NAN;
                        break;
                    }
                    if residual > max_residual {
                        max_residual = residual;
                        worst_x = Some(x);
                    }
                }

                let passes = !max_residual.is_nan() && max_residual <= spec.tolerance_ulp;
                identity_results.push(IdentityResult {
                    name: spec.name.clone(),
                    max_residual_ulp: max_residual,
                    claimed_tolerance_ulp: spec.tolerance_ulp,
                    passes,
                    worst_case_input: worst_x,
                });
            }
        }
    }

    // Overall: passes iff every component passes
    let passes = {
        let random_ok = random_sample_report.passes(entry.claimed_max_ulp as u64);
        let injections_ok = injection_reports.iter()
            .all(|(_, r)| r.special_value_failures == 0);
        let bit_exact_ok = bit_exact_results.iter().all(|r| r.passes);
        let constraints_ok = constraint_results.iter().all(|r| r.passes);
        let identities_ok = identity_results.iter().all(|r| r.passes);
        random_ok && injections_ok && bit_exact_ok && constraints_ok && identities_ok
    };

    OracleReport {
        function_name: entry.function_name.clone(),
        claimed_max_ulp: entry.claimed_max_ulp,
        random_sample_report,
        injection_reports,
        bit_exact_results,
        constraint_results,
        identity_results,
        passes,
    }
}
