//! Associative operators for parallel scan.
//!
//! Each operator defines:
//! - A state type (what accumulates)
//! - An identity element (the "zero" of the operation)
//! - A combine function (how two partial results merge — MUST be associative)
//! - CUDA source fragments for GPU kernel generation
//!
//! The associativity requirement is the liftability test:
//! combine(combine(a, b), c) == combine(a, combine(b, c))
//!
//! If this holds, the scan can run in O(log n) parallel depth on GPU.
//! If it doesn't, you've hit the Fock boundary — the computation is
//! inherently sequential.

/// The scannability test as a trait. Implement this for your operator,
/// and the scan engine parallelizes it automatically.
pub trait AssociativeOp: Send + Sync {
    /// Name used for kernel cache key differentiation.
    fn name(&self) -> &'static str;

    /// CUDA type declaration for the state struct.
    /// e.g., "double" for AddOp, or a struct definition for WelfordOp.
    fn cuda_state_type(&self) -> String;

    /// CUDA expression for the identity element.
    /// e.g., "0.0" for AddOp, "{0, 0.0, 0.0}" for WelfordOp.
    fn cuda_identity(&self) -> String;

    /// CUDA expression that combines two states `a` and `b` into a merged state.
    /// This is the associative binary operator.
    /// Variables `a` and `b` are in scope as the state type.
    fn cuda_combine(&self) -> String;

    /// CUDA expression that creates a state from a single input element `x`.
    /// e.g., "x" for AddOp, "{1, x, 0.0}" for WelfordOp.
    fn cuda_lift_element(&self) -> String;

    /// CUDA expression that extracts the "primary" output from a state.
    /// e.g., "s" for AddOp (state IS the output),
    ///       "s.mean" for WelfordOp (extract mean from {count, mean, M2}).
    fn cuda_extract(&self) -> String;

    /// Number of output values per element (1 for most, 2 for Welford mean+var).
    fn output_width(&self) -> usize { 1 }

    /// Optional: additional CUDA expressions for secondary outputs.
    /// e.g., WelfordOp returns variance as second output.
    fn cuda_extract_secondary(&self) -> Vec<String> { vec![] }

    /// Parameters that affect the kernel (for cache key uniqueness).
    /// e.g., EWMOp includes alpha.
    fn params_key(&self) -> String { String::new() }
}

// ============================================================
// AddOp — cumulative sum
// ============================================================

pub struct AddOp;

impl AssociativeOp for AddOp {
    fn name(&self) -> &'static str { "add" }
    fn cuda_state_type(&self) -> String { "double".into() }
    fn cuda_identity(&self) -> String { "0.0".into() }
    fn cuda_combine(&self) -> String { "(a + b)".into() }
    fn cuda_lift_element(&self) -> String { "x".into() }
    fn cuda_extract(&self) -> String { "s".into() }
}

// ============================================================
// MulOp — cumulative product
// ============================================================

pub struct MulOp;

impl AssociativeOp for MulOp {
    fn name(&self) -> &'static str { "mul" }
    fn cuda_state_type(&self) -> String { "double".into() }
    fn cuda_identity(&self) -> String { "1.0".into() }
    fn cuda_combine(&self) -> String { "(a * b)".into() }
    fn cuda_lift_element(&self) -> String { "x".into() }
    fn cuda_extract(&self) -> String { "s".into() }
}

// ============================================================
// MaxOp — cumulative maximum
// ============================================================

pub struct MaxOp;

impl AssociativeOp for MaxOp {
    fn name(&self) -> &'static str { "max" }
    fn cuda_state_type(&self) -> String { "double".into() }
    fn cuda_identity(&self) -> String { "(-1.0/0.0)".into() } // -infinity
    fn cuda_combine(&self) -> String { "(a > b ? a : b)".into() }
    fn cuda_lift_element(&self) -> String { "x".into() }
    fn cuda_extract(&self) -> String { "s".into() }
}

// ============================================================
// MinOp — cumulative minimum
// ============================================================

pub struct MinOp;

impl AssociativeOp for MinOp {
    fn name(&self) -> &'static str { "min" }
    fn cuda_state_type(&self) -> String { "double".into() }
    fn cuda_identity(&self) -> String { "(1.0/0.0)".into() } // +infinity
    fn cuda_combine(&self) -> String { "(a < b ? a : b)".into() }
    fn cuda_lift_element(&self) -> String { "x".into() }
    fn cuda_extract(&self) -> String { "s".into() }
}

// ============================================================
// WelfordOp — online mean + variance (Welford's algorithm)
//
// State: { count: i64, mean: f64, m2: f64 }
// This is the SAME algebraic structure as FlashAttention's
// online softmax: carry (max, exp_sum, weighted_sum).
// Different domains, same semigroup.
// ============================================================

pub struct WelfordOp;

impl AssociativeOp for WelfordOp {
    fn name(&self) -> &'static str { "welford" }

    fn cuda_state_type(&self) -> String {
        r#"struct WelfordState { long long count; double mean; double m2; }"#.into()
    }

    fn cuda_identity(&self) -> String {
        "{0, 0.0, 0.0}".into()
    }

    fn cuda_combine(&self) -> String {
        // Parallel Welford merge: combine two partial aggregates.
        // This IS associative — proven by Chan et al. 1979.
        r#"({
            long long n = a.count + b.count;
            double delta = b.mean - a.mean;
            double mean = (n == 0) ? 0.0 : a.mean + delta * (double)b.count / (double)n;
            double m2 = a.m2 + b.m2 + delta * delta * (double)a.count * (double)b.count / (double)n;
            (WelfordState){n, mean, m2};
        })"#.into()
    }

    fn cuda_lift_element(&self) -> String {
        "(WelfordState){1, x, 0.0}".into()
    }

    fn cuda_extract(&self) -> String {
        "s.mean".into()
    }

    fn output_width(&self) -> usize { 2 }

    fn cuda_extract_secondary(&self) -> Vec<String> {
        vec!["(s.count > 1 ? s.m2 / (double)(s.count - 1) : 0.0)".into()] // variance
    }
}

// ============================================================
// EWMOp — exponential weighted mean
//
// State: { weight_sum: f64, value_sum: f64 }
// Recurrence: s[t] = alpha * x[t] + (1 - alpha) * s[t-1]
// Associative form: combine partial weighted sums.
// ============================================================

pub struct EWMOp {
    pub alpha: f64,
}

impl AssociativeOp for EWMOp {
    fn name(&self) -> &'static str { "ewm" }

    fn cuda_state_type(&self) -> String {
        r#"struct EWMState { double weight; double value; }"#.into()
    }

    fn cuda_identity(&self) -> String {
        "{0.0, 0.0}".into()
    }

    fn cuda_combine(&self) -> String {
        // EWM merge: the later segment's weights decay the earlier segment.
        // a is the earlier (left) partial, b is the later (right) partial.
        // The decay factor for a's contribution through b's length is b.weight.
        format!(
            r#"({{
                double decay = pow(1.0 - {alpha}, (double)1);
                (EWMState){{ a.weight * decay + b.weight, a.value * decay + b.value }};
            }})"#,
            alpha = self.alpha
        )
    }

    fn cuda_lift_element(&self) -> String {
        format!("(EWMState){{1.0, x * {alpha}}}", alpha = self.alpha)
    }

    fn cuda_extract(&self) -> String {
        "(s.weight > 0.0 ? s.value / s.weight : 0.0)".into()
    }

    fn params_key(&self) -> String {
        format!("alpha={:.10}", self.alpha)
    }
}
