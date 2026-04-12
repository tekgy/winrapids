//! Codegen: TBS Expr + AccumulatePass → CUDA C source.
//!
//! This is the vendor door for NVIDIA. It takes a fused `AccumulatePass`
//! and emits a single CUDA C kernel that computes every slot's expression
//! per element and reduces into per-slot accumulators.
//!
//! Scope of this version:
//!   - `Grouping::All` only (N → 1 reductions).
//!   - `Op::Add` only (the fusion sweet spot — 47 of 55 slots in the catalog).
//!   - Up to 2 input buffers (`Val` = col 0, `Val2` = col 1).
//!
//! Everything else (Min/Max/Mul, Prefix/ByKey, Col(n>1)) is routed to the
//! CPU executor for now. The purpose of this module is to prove the door
//! works end-to-end on real silicon, not to cover every case.
//!
//! The output of `pass_to_cuda_kernel` is a complete, NVRTC-compilable
//! C source string. N is baked in via `#define TAM_N ...` — when N
//! changes the kernel recompiles. That's a fair tradeoff for JIT speed
//! (NVRTC cost ~40 ms) against per-dispatch arg overhead.

use crate::accumulates::{AccumulatePass, DataSource, Grouping, Op};
use crate::tbs::Expr;

// ═══════════════════════════════════════════════════════════════════
// Expr → CUDA C expression
// ═══════════════════════════════════════════════════════════════════

/// Render a TBS `Expr` as a CUDA C (double-precision) expression.
///
/// `val_var` is the name of the first-column variable (usually `"v"`),
/// `val2_var` is the second-column variable (usually `"v2"`),
/// `ref_var` is the reference scalar for centered transforms (usually `"r"`).
///
/// Only `Expr::Var` is forbidden here — gather-time variables bind later,
/// not in the per-element kernel body.
pub fn expr_to_cuda(
    expr: &Expr,
    val_var: &str,
    val2_var: &str,
    ref_var: &str,
) -> String {
    let rec = |e: &Expr| expr_to_cuda(e, val_var, val2_var, ref_var);
    match expr {
        Expr::Val        => val_var.to_string(),
        Expr::Val2       => val2_var.to_string(),
        Expr::Ref        => ref_var.to_string(),
        Expr::Col(0)     => val_var.to_string(),
        Expr::Col(1)     => val2_var.to_string(),
        Expr::Col(n)     => panic!("codegen: Col({n}) not supported in Phase 1"),
        Expr::Var(name)  => panic!("codegen: Var('{name}') belongs to gather, not accumulate body"),
        Expr::Lit(c)     => format!("({:.17e})", c),

        // Unary
        Expr::Neg(a)     => format!("(-{})", rec(a)),
        Expr::Abs(a)     => format!("fabs({})", rec(a)),
        Expr::Recip(a)   => format!("(1.0/{})", rec(a)),
        Expr::Sq(a)      => { let s = rec(a); format!("(({0})*({0}))", s) }
        Expr::Sqrt(a)    => format!("sqrt({})", rec(a)),
        Expr::Ln(a)      => format!("log({})", rec(a)),
        Expr::Exp(a)     => format!("exp({})", rec(a)),
        Expr::Floor(a)   => format!("floor({})", rec(a)),
        Expr::Ceil(a)    => format!("ceil({})", rec(a)),
        Expr::Sign(a)    => { let s = rec(a); format!("(({0})>0.0 ? 1.0 : (({0})<0.0 ? -1.0 : 0.0))", s) }
        Expr::IsFinite(a)=> format!("(isfinite({}) ? 1.0 : 0.0)", rec(a)),
        Expr::Sin(a)     => format!("sin({})", rec(a)),
        Expr::Cos(a)     => format!("cos({})", rec(a)),
        Expr::Tan(a)     => format!("tan({})", rec(a)),
        Expr::Asin(a)    => format!("asin({})", rec(a)),
        Expr::Acos(a)    => format!("acos({})", rec(a)),
        Expr::Atan(a)    => format!("atan({})", rec(a)),
        Expr::Sinh(a)    => format!("sinh({})", rec(a)),
        Expr::Cosh(a)    => format!("cosh({})", rec(a)),
        Expr::Tanh(a)    => format!("tanh({})", rec(a)),
        Expr::Round(a)   => format!("rint({})", rec(a)),
        Expr::Trunc(a)   => format!("trunc({})", rec(a)),

        // Binary arithmetic
        Expr::Add(a, b)  => format!("({} + {})", rec(a), rec(b)),
        Expr::Sub(a, b)  => format!("({} - {})", rec(a), rec(b)),
        Expr::Mul(a, b)  => format!("({} * {})", rec(a), rec(b)),
        Expr::Div(a, b)  => format!("({} / {})", rec(a), rec(b)),
        Expr::Pow(a, b)  => format!("pow({}, {})", rec(a), rec(b)),
        Expr::Min(a, b)  => format!("fmin({}, {})", rec(a), rec(b)),
        Expr::Max(a, b)  => format!("fmax({}, {})", rec(a), rec(b)),
        Expr::Atan2(a, b)=> format!("atan2({}, {})", rec(a), rec(b)),
        Expr::Mod(a, b)  => format!("fmod({}, {})", rec(a), rec(b)),

        // Ternary
        Expr::Clamp(a, lo, hi) => {
            format!("fmin(fmax({}, {}), {})", rec(a), rec(lo), rec(hi))
        }
        Expr::If(cond, t, e) => {
            format!("(({}) != 0.0 ? {} : {})", rec(cond), rec(t), rec(e))
        }

        // Comparisons → 0.0 / 1.0 (indicator values)
        Expr::Gt(a, b)   => format!("(({}) > ({}) ? 1.0 : 0.0)", rec(a), rec(b)),
        Expr::Lt(a, b)   => format!("(({}) < ({}) ? 1.0 : 0.0)", rec(a), rec(b)),
        Expr::Eq(a, b)   => format!("(fabs(({}) - ({})) < 1e-15 ? 1.0 : 0.0)", rec(a), rec(b)),
    }
}

// ═══════════════════════════════════════════════════════════════════
// AccumulatePass → kernel source
// ═══════════════════════════════════════════════════════════════════

/// Error returned when a pass cannot be lowered to this backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodegenError(pub String);

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "codegen error: {}", self.0)
    }
}

impl std::error::Error for CodegenError {}

/// What the generated kernel expects: number of input columns, number
/// of outputs, entry-point name, and the source string.
#[derive(Debug, Clone)]
pub struct CudaKernelSource {
    pub entry: String,
    pub source: String,
    /// How many input data buffers the kernel reads (1 or 2).
    pub n_inputs: usize,
    /// How many output slots the kernel writes.
    pub n_outputs: usize,
}

/// Lower an `AccumulatePass` to a CUDA C kernel source.
///
/// The generated kernel signature is (for a two-input pass):
/// ```c
/// extern "C" __global__ void {entry}(
///     const double* __restrict__ data_x,
///     const double* __restrict__ data_y,
///     double* out_slots
/// );
/// ```
/// with `TAM_N` baked in via `#define`. Launch with any `grid × block`
/// configuration — the kernel uses a grid-stride loop.
pub fn pass_to_cuda_kernel(
    pass: &AccumulatePass,
    entry: &str,
    n_elements: usize,
    reference: f64,
) -> Result<CudaKernelSource, CodegenError> {
    // ─── Validate scope ────────────────────────────────────────────
    if pass.grouping != Grouping::All {
        return Err(CodegenError(format!(
            "only Grouping::All supported in Phase 1, got {:?}", pass.grouping
        )));
    }
    if pass.op != Op::Add {
        return Err(CodegenError(format!(
            "only Op::Add supported in Phase 1, got {:?}", pass.op
        )));
    }
    if !matches!(pass.source, DataSource::Primary) {
        return Err(CodegenError(format!(
            "only DataSource::Primary supported in Phase 1, got {:?}", pass.source
        )));
    }
    // Detect whether we need a second input column
    let needs_val2 = pass.slots.iter().any(|(e, _)| mentions_val2(e));

    // ─── Header and constants ──────────────────────────────────────
    let n_slots = pass.slots.len();
    let mut src = String::new();
    src.push_str("// Tambear-generated CUDA kernel\n");
    src.push_str("// grouping: All  op: Add  source: Primary\n");
    src.push_str(&format!("#define TAM_N {}\n", n_elements));
    src.push_str(&format!("#define TAM_NSLOTS {}\n", n_slots));
    src.push_str(&format!("#define TAM_REF ({:.17e})\n", reference));
    src.push_str("\n");

    // ─── Kernel signature ──────────────────────────────────────────
    let n_inputs = if needs_val2 { 2 } else { 1 };
    src.push_str("extern \"C\" __global__ void ");
    src.push_str(entry);
    src.push_str("(\n");
    src.push_str("    const double* __restrict__ data_x,\n");
    if needs_val2 {
        src.push_str("    const double* __restrict__ data_y,\n");
    }
    src.push_str("    double* out_slots\n");
    src.push_str(") {\n");

    // ─── Thread-local accumulators ─────────────────────────────────
    src.push_str("    // Per-thread partial sums, one per slot.\n");
    for i in 0..n_slots {
        src.push_str(&format!("    double acc_{} = 0.0;\n", i));
    }
    src.push_str("\n");

    // ─── Grid-stride loop ──────────────────────────────────────────
    src.push_str("    const int gi     = blockIdx.x * blockDim.x + threadIdx.x;\n");
    src.push_str("    const int stride = gridDim.x  * blockDim.x;\n");
    src.push_str("    for (int i = gi; i < TAM_N; i += stride) {\n");
    src.push_str("        const double v  = data_x[i];\n");
    if needs_val2 {
        src.push_str("        const double v2 = data_y[i];\n");
    } else {
        // Supply 0.0 for kernels that don't use it (harmless in codegen).
        src.push_str("        const double v2 = 0.0;\n");
    }
    src.push_str("        const double r  = TAM_REF;\n");
    for (i, (e, name)) in pass.slots.iter().enumerate() {
        let expr_str = expr_to_cuda(e, "v", "v2", "r");
        src.push_str(&format!("        acc_{} += {};   // {}\n", i, expr_str, name));
    }
    src.push_str("    }\n\n");

    // ─── Atomic write-back to out_slots ────────────────────────────
    // atomicAdd(double*) requires sm_60+. Blackwell = sm_120, no problem.
    src.push_str("    // Atomic reduce across blocks. sm_60+ supports fp64 atomicAdd.\n");
    for i in 0..n_slots {
        src.push_str(&format!("    atomicAdd(&out_slots[{}], acc_{});\n", i, i));
    }
    src.push_str("}\n");

    Ok(CudaKernelSource {
        entry: entry.to_string(),
        source: src,
        n_inputs,
        n_outputs: n_slots,
    })
}

/// Does this expression reference `Val2` or `Col(1)` anywhere?
fn mentions_val2(expr: &Expr) -> bool {
    match expr {
        Expr::Val2 | Expr::Col(1) => true,
        Expr::Val | Expr::Ref | Expr::Col(_) | Expr::Lit(_) | Expr::Var(_) => false,
        Expr::Neg(a) | Expr::Abs(a) | Expr::Recip(a) | Expr::Sq(a)
        | Expr::Sqrt(a) | Expr::Ln(a) | Expr::Exp(a) | Expr::Floor(a)
        | Expr::Ceil(a) | Expr::Sign(a) | Expr::IsFinite(a)
        | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
        | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
        | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
        | Expr::Round(a) | Expr::Trunc(a) => mentions_val2(a),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b)
        | Expr::Pow(a, b) | Expr::Min(a, b) | Expr::Max(a, b)
        | Expr::Atan2(a, b) | Expr::Mod(a, b)
        | Expr::Gt(a, b) | Expr::Lt(a, b) | Expr::Eq(a, b)
            => mentions_val2(a) || mentions_val2(b),
        Expr::Clamp(a, b, c) | Expr::If(a, b, c)
            => mentions_val2(a) || mentions_val2(b) || mentions_val2(c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulates::fuse_passes;
    use crate::recipes::{variance, pearson_r, mean_arithmetic};

    #[test]
    fn simple_val_renders() {
        assert_eq!(expr_to_cuda(&Expr::val(), "v", "v2", "r"), "v");
    }

    #[test]
    fn square_renders() {
        assert_eq!(
            expr_to_cuda(&Expr::val().sq(), "v", "v2", "r"),
            "((v)*(v))"
        );
    }

    #[test]
    fn lit_uses_full_precision() {
        // Full 17-digit double rendering.
        let s = expr_to_cuda(&Expr::lit(1.0), "v", "v2", "r");
        assert!(s.contains("1.00000"));
    }

    #[test]
    fn cross_product_renders() {
        let e = Expr::val().mul(Expr::val2());
        assert_eq!(
            expr_to_cuda(&e, "v", "v2", "r"),
            "(v * v2)"
        );
    }

    #[test]
    fn variance_fused_pass_lowers() {
        let r = variance();
        let passes = fuse_passes(&r.slots);
        assert_eq!(passes.len(), 1);
        let src = pass_to_cuda_kernel(&passes[0], "var_all_add", 1024, 0.0).unwrap();
        assert_eq!(src.n_inputs, 1);
        assert_eq!(src.n_outputs, 3); // sum, sum_sq, count
        assert!(src.source.contains("extern \"C\" __global__"));
        assert!(src.source.contains("var_all_add"));
        assert!(src.source.contains("atomicAdd"));
        // Must reference v for sum, and (v)*(v) for sum_sq, and 1e0 for count.
        assert!(src.source.contains("acc_0 += v"));
        assert!(src.source.contains("acc_1 += ((v)*(v))"));
    }

    #[test]
    fn pearson_needs_two_inputs() {
        let r = pearson_r();
        let passes = fuse_passes(&r.slots);
        assert_eq!(passes.len(), 1);
        let src = pass_to_cuda_kernel(&passes[0], "pearson_add", 16, 0.0).unwrap();
        assert_eq!(src.n_inputs, 2, "pearson uses Val2 → needs data_y");
        assert!(src.source.contains("data_y"));
        assert!(src.source.contains("(v * v2)"));
    }

    #[test]
    fn mean_only_one_input() {
        let r = mean_arithmetic();
        let passes = fuse_passes(&r.slots);
        let src = pass_to_cuda_kernel(&passes[0], "mean_add", 16, 0.0).unwrap();
        assert_eq!(src.n_inputs, 1);
        assert!(!src.source.contains("data_y"));
    }

    #[test]
    fn unsupported_grouping_errors() {
        let pass = AccumulatePass {
            source: DataSource::Primary,
            grouping: Grouping::Prefix,
            op: Op::Add,
            slots: vec![(Expr::val(), "x".to_string())],
        };
        assert!(pass_to_cuda_kernel(&pass, "k", 16, 0.0).is_err());
    }

    #[test]
    fn unsupported_op_errors() {
        let pass = AccumulatePass {
            source: DataSource::Primary,
            grouping: Grouping::All,
            op: Op::Max,
            slots: vec![(Expr::val(), "x".to_string())],
        };
        assert!(pass_to_cuda_kernel(&pass, "k", 16, 0.0).is_err());
    }
}

