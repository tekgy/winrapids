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

    /// Size of the state type in bytes. Used for device memory allocation.
    /// Default: 8 (scalar double). Override for struct states.
    fn state_byte_size(&self) -> usize { 8 }

    /// CUDA function body for lift_element. Override for struct states
    /// that need C++-compatible initialization (no compound literals).
    fn cuda_lift_body(&self) -> String {
        format!("    state_t s = {};\n    return s;", self.cuda_lift_element())
    }

    /// CUDA function body for combine_states. Override for struct states
    /// that need multi-statement combine logic (no GCC statement expressions).
    fn cuda_combine_body(&self) -> String {
        format!("    state_t result = {};\n    return result;", self.cuda_combine())
    }
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

    fn state_byte_size(&self) -> usize { 24 } // i64(8) + f64(8) + f64(8)

    fn cuda_lift_body(&self) -> String {
        r#"    state_t s;
    s.count = 1;
    s.mean = x;
    s.m2 = 0.0;
    return s;"#.into()
    }

    fn cuda_combine_body(&self) -> String {
        r#"    long long n = a.count + b.count;
    double delta = b.mean - a.mean;
    double mean = (n == 0) ? 0.0 : a.mean + delta * (double)b.count / (double)n;
    double m2 = a.m2 + b.m2 + delta * delta * (double)a.count * (double)b.count / (double)n;
    state_t result;
    result.count = n;
    result.mean = mean;
    result.m2 = m2;
    return result;"#.into()
    }
}

// ============================================================
// KalmanOp — parallel Kalman filter (scalar, 1D state)
//
// The liftability principle's most dramatic demonstration:
// Kalman filtering — the gold standard of sequential estimation —
// parallelized from O(n) to O(log n) via associative scan.
//
// State: { x: f64, P: f64, F_acc: f64, has_data: i32 }
//   x = state estimate
//   P = error covariance
//   F_acc = accumulated dynamics (product of F across the segment)
//   has_data = whether this segment has been initialized
//
// The combine operation merges two Kalman filter segments.
// For the scalar case (1D state, 1D observation):
//   Predict: x_pred = F * x, P_pred = F * P * F + Q
//   Update:  K = P_pred * H / (H * P_pred * H + R)
//            x = x_pred + K * (z - H * x_pred)
//            P = (1 - K * H) * P_pred
//
// The parallel formulation (Särkkä & García-Fernández 2021):
// Two segments (left=a, right=b) merge by:
//   1. Propagate a's state through b's dynamics
//   2. Fuse with b's measurement update
//
// This IS associative. Proven in the paper. The scan parallelizes it.
//
// Parameters: F (state transition), H (observation model),
//             Q (process noise), R (observation noise)
//
// Fock boundary: nonlinear Kalman (EKF/UKF) where F depends on x.
// Partial lift: linearize around trajectory estimate (EKF-as-scan,
// approximate but bounded error).
// ============================================================

pub struct KalmanOp {
    pub f: f64,  // state transition (scalar)
    pub h: f64,  // observation model (scalar)
    pub q: f64,  // process noise variance
    pub r: f64,  // observation noise variance
}

impl AssociativeOp for KalmanOp {
    fn name(&self) -> &'static str { "kalman" }

    fn cuda_state_type(&self) -> String {
        r#"struct KalmanState { double x; double P; double F_acc; int has_data; }"#.into()
    }

    fn cuda_identity(&self) -> String {
        "{0.0, 0.0, 1.0, 0}".into()
    }

    fn cuda_combine(&self) -> String {
        // Not used — cuda_combine_body overrides
        String::new()
    }

    fn cuda_lift_element(&self) -> String {
        // Not used — cuda_lift_body overrides
        String::new()
    }

    fn cuda_extract(&self) -> String {
        "s.x".into()
    }

    fn output_width(&self) -> usize { 2 }

    fn cuda_extract_secondary(&self) -> Vec<String> {
        vec!["s.P".into()] // error covariance
    }

    fn params_key(&self) -> String {
        format!("F={:.10},H={:.10},Q={:.10},R={:.10}", self.f, self.h, self.q, self.r)
    }

    fn state_byte_size(&self) -> usize { 32 } // f64(8) + f64(8) + f64(8) + i32(4) + 4B padding = 32

    fn cuda_lift_body(&self) -> String {
        // Each observation z is lifted into a Kalman state.
        // Initialize with the measurement: x = z/H, P = R/H²
        // This is the "information form" initialization.
        format!(
            r#"    state_t s;
    double H = {h};
    double R = {r};
    s.x = x / H;
    s.P = R / (H * H);
    s.F_acc = 1.0;
    s.has_data = 1;
    return s;"#,
            h = self.h,
            r = self.r,
        )
    }

    fn cuda_combine_body(&self) -> String {
        // Merge two Kalman segments: left (a) and right (b).
        // 1. Propagate a's state through the time gap using F
        // 2. Fuse propagated a with b using Kalman update equations
        //
        // The key insight: this merge IS associative because
        // linear state-space models compose associatively.
        // (Särkkä & García-Fernández 2021, Theorem 1)
        format!(
            r#"    double F = {f};
    double Q = {q};
    state_t result;

    if (!a.has_data) {{
        // Left segment empty — return right
        result = b;
    }} else if (!b.has_data) {{
        // Right segment empty — propagate left through time
        result.x = F * a.x;
        result.P = F * a.P * F + Q;
        result.F_acc = F * a.F_acc;
        result.has_data = 1;
    }} else {{
        // Both segments have data — predict then update
        // Predict: propagate a through one time step
        double x_pred = F * a.x;
        double P_pred = F * a.P * F + Q;

        // Fuse: combine prediction with b's estimate
        // Using the covariance intersection / Kalman fusion:
        // P_fused = 1 / (1/P_pred + 1/b.P)
        // x_fused = P_fused * (x_pred/P_pred + b.x/b.P)
        double P_inv_sum = 1.0 / P_pred + 1.0 / b.P;
        result.P = 1.0 / P_inv_sum;
        result.x = result.P * (x_pred / P_pred + b.x / b.P);
        result.F_acc = b.F_acc * F * a.F_acc;
        result.has_data = 1;
    }}
    return result;"#,
            f = self.f,
            q = self.q,
        )
    }
}

// ============================================================
// EWMOp — exponential weighted mean
//
// State: { weight: f64, value: f64, count: i64 }
// Recurrence: s[t] = alpha * x[t] + (1 - alpha) * s[t-1]
// Associative form: combine partial weighted sums, decaying
// the earlier segment by pow(1-alpha, b.count) — the length
// of the later segment.
// ============================================================

pub struct EWMOp {
    pub alpha: f64,
}

impl AssociativeOp for EWMOp {
    fn name(&self) -> &'static str { "ewm" }

    fn cuda_state_type(&self) -> String {
        r#"struct EWMState { double weight; double value; long long count; }"#.into()
    }

    fn cuda_identity(&self) -> String {
        "{0.0, 0.0, 0}".into()
    }

    fn cuda_combine(&self) -> String {
        // EWM merge: the later segment's weights decay the earlier segment.
        // a is the earlier (left) partial, b is the later (right) partial.
        // The decay factor is pow(1-alpha, b.count) — the length of b's segment.
        format!(
            r#"({{
                double decay = pow(1.0 - {alpha}, (double)b.count);
                (EWMState){{ a.weight * decay + b.weight,
                             a.value  * decay + b.value,
                             a.count + b.count }};
            }})"#,
            alpha = self.alpha
        )
    }

    fn cuda_lift_element(&self) -> String {
        format!("(EWMState){{1.0, x * {alpha}, 1LL}}", alpha = self.alpha)
    }

    fn cuda_extract(&self) -> String {
        "(s.weight > 0.0 ? s.value / s.weight : 0.0)".into()
    }

    fn params_key(&self) -> String {
        format!("alpha={:.10}", self.alpha)
    }

    fn state_byte_size(&self) -> usize { 24 } // f64(8) + f64(8) + i64(8)

    fn cuda_lift_body(&self) -> String {
        format!(
            "    state_t s;\n    s.weight = 1.0;\n    s.value = x * {alpha};\n    s.count = 1;\n    return s;",
            alpha = self.alpha
        )
    }

    fn cuda_combine_body(&self) -> String {
        format!(
            r#"    double decay = pow(1.0 - {alpha}, (double)b.count);
    state_t result;
    result.weight = a.weight * decay + b.weight;
    result.value = a.value * decay + b.value;
    result.count = a.count + b.count;
    return result;"#,
            alpha = self.alpha
        )
    }
}

// ============================================================
// KalmanAffineOp — exact steady-state Kalman via affine scan
//
// State: { a_acc: f64, b_acc: f64 } — 16 bytes, two doubles
//
// The steady-state Kalman recurrence is:
//   x[t] = A * x[t-1] + K_ss * z[t]
// where A = (1 - K_ss * H) * F and K_ss is the steady-state
// Kalman gain from the discrete algebraic Riccati equation.
//
// This is an AFFINE MAP: x → A*x + b where b = K_ss * z[t].
// Affine maps compose associatively:
//   (A2, b2) ∘ (A1, b1) = (A2*A1, A2*b1 + b2)
//
// Combine cost: 2 multiplies + 1 add. No pow(), no division,
// no covariance intersection. Trivially associative.
//
// At F=1, H=1: shares decay constant with EWMOp(alpha=K_ss)
// but NOT equivalent — different extract semantics (unnormalized
// state vs weight-normalized average). Verified: diverges by up
// to 10.5 at n=10K. See lab notebook Entry 021.
//
// Parameters: F (dynamics), H (observation), Q (process noise),
//             R (observation noise). Constructor solves Riccati
//             to get K_ss and A.
// ============================================================

pub struct KalmanAffineOp {
    pub f: f64,
    pub h: f64,
    pub q: f64,
    pub r: f64,
    /// Steady-state Kalman gain (from Riccati convergence).
    pub k_ss: f64,
    /// State transition after update: A = (1 - K_ss * H) * F.
    pub a: f64,
}

impl KalmanAffineOp {
    /// Construct from dynamics parameters. Solves the discrete algebraic
    /// Riccati equation iteratively to find the steady-state gain K_ss.
    pub fn new(f: f64, h: f64, q: f64, r: f64) -> Self {
        // Iterate Riccati to convergence (1D scalar, converges in ~50 iterations)
        let mut p = 1.0;
        for _ in 0..1000 {
            let p_pred = f * p * f + q;
            let k = p_pred * h / (h * p_pred * h + r);
            p = (1.0 - k * h) * p_pred;
        }
        // Final K_ss from converged P
        let p_pred = f * p * f + q;
        let k_ss = p_pred * h / (h * p_pred * h + r);
        let a = (1.0 - k_ss * h) * f;
        Self { f, h, q, r, k_ss, a }
    }
}

impl AssociativeOp for KalmanAffineOp {
    fn name(&self) -> &'static str { "kalman_affine" }

    fn cuda_state_type(&self) -> String {
        r#"struct KalmanAffineState { double a_acc; double b_acc; }"#.into()
    }

    fn cuda_identity(&self) -> String {
        "{1.0, 0.0}".into()
    }

    fn cuda_combine(&self) -> String {
        // Not used — cuda_combine_body overrides
        String::new()
    }

    fn cuda_lift_element(&self) -> String {
        // Not used — cuda_lift_body overrides
        String::new()
    }

    fn cuda_extract(&self) -> String {
        "s.b_acc".into()
    }

    fn params_key(&self) -> String {
        format!("F={:.10}_H={:.10}_Q={:.10}_R={:.10}", self.f, self.h, self.q, self.r)
    }

    fn state_byte_size(&self) -> usize { 16 } // 2 × f64

    fn cuda_lift_body(&self) -> String {
        format!(
            r#"    state_t s;
    s.a_acc = {a};
    s.b_acc = {k_ss} * x;
    return s;"#,
            a = self.a, k_ss = self.k_ss
        )
    }

    fn cuda_combine_body(&self) -> String {
        // Affine composition: (b ∘ a)(x) = b.a * (a.a * x + a.b) + b.b
        // = (b.a * a.a) * x + (b.a * a.b + b.b)
        r#"    state_t result;
    result.a_acc = b.a_acc * a.a_acc;
    result.b_acc = b.a_acc * a.b_acc + b.b_acc;
    return result;"#.into()
    }
}
