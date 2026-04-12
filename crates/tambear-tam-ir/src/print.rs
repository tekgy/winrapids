//! Text printer: AST → `.tam` text.
//!
//! `print_program(prog)` produces a canonical `.tam` text representation
//! that the parser can round-trip back to the same AST.
//!
//! Design decisions:
//! - Constants are printed with their full bit-exact hex representation
//!   (`0x<16-hex-digits>`) to guarantee round-trip equality even for edge
//!   cases like NaN bit patterns, -0.0, and subnormals.
//! - Registers always print with `%` prefix and optional `'` suffix.
//! - One op per line. Loop bodies are indented by 2 additional spaces.
//! - Comments in the source are NOT round-tripped (they're not in the AST).

use crate::ast::*;

// ═══════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════

/// Print a `Program` to a `.tam` text string.
///
/// The output is a canonical, human-readable `.tam` text that the parser
/// can parse back to an equivalent AST. Round-trip guarantee: for any
/// `Program p`, `parse(print(p)) == p` (modulo comments).
pub fn print_program(prog: &Program) -> String {
    let mut out = String::new();
    print_header(&mut out, &prog.version, &prog.target);
    out.push('\n');
    for func in &prog.funcs {
        print_func(&mut out, func, 0);
        out.push('\n');
    }
    for kernel in &prog.kernels {
        print_kernel(&mut out, kernel, 0);
        out.push('\n');
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// Header
// ═══════════════════════════════════════════════════════════════════

fn print_header(out: &mut String, version: &TamVersion, target: &Target) {
    out.push_str(&format!(".tam {}.{}\n", version.major, version.minor));
    let tgt = match target {
        Target::Cross => "cross",
        Target::Other(s) => s.as_str(),
    };
    out.push_str(&format!(".target {}\n", tgt));
}

// ═══════════════════════════════════════════════════════════════════
// Functions (libm)
// ═══════════════════════════════════════════════════════════════════

fn print_func(out: &mut String, func: &FuncDef, indent: usize) {
    let pad = spaces(indent);
    // Params: always f64
    let params: Vec<String> = func.params.iter()
        .map(|p| format!("f64 {}", p.reg.display()))
        .collect();
    out.push_str(&format!("{}func {}({}) -> f64 {{\n", pad, func.name, params.join(", ")));
    out.push_str(&format!("{}entry:\n", pad));
    for op in &func.body {
        print_op(out, op, indent + 2);
    }
    out.push_str(&format!("{}}}\n", pad));
}

// ═══════════════════════════════════════════════════════════════════
// Kernels
// ═══════════════════════════════════════════════════════════════════

fn print_kernel(out: &mut String, kernel: &KernelDef, indent: usize) {
    let pad = spaces(indent);
    // Emit kernel attributes before the kernel keyword.
    for attr in &kernel.attrs {
        match attr {
            KernelAttr::AccumulatorStateSize(n) => {
                out.push_str(&format!("{}@accumulator_state_size({})\n", pad, n));
            }
            KernelAttr::DefaultOrderStrategy(r) => {
                out.push_str(&format!("{}@default_order_strategy({})\n", pad, r.name()));
            }
        }
    }
    let params: Vec<String> = kernel.params.iter()
        .map(|p| {
            let ty_str = match &p.ty {
                Ty::BufF64 => "buf<f64>",
                Ty::I32 => "i32",
                Ty::I64 => "i64",
                Ty::F64 => "f64",
                Ty::Pred => "pred",
            };
            format!("{} {}", ty_str, p.reg.display())
        })
        .collect();
    out.push_str(&format!("{}kernel {}({}) {{\n", pad, kernel.name, params.join(", ")));
    out.push_str(&format!("{}entry:\n", pad));
    for stmt in &kernel.body {
        print_stmt(out, stmt, indent + 2);
    }
    out.push_str(&format!("{}}}\n", pad));
}

fn print_stmt(out: &mut String, stmt: &Stmt, indent: usize) {
    match stmt {
        Stmt::Op(op) => print_op(out, op, indent),
        Stmt::Loop(lp) => print_loop(out, lp, indent),
    }
}

fn print_loop(out: &mut String, lp: &LoopGridStride, indent: usize) {
    let pad = spaces(indent);
    out.push_str(&format!(
        "{}loop_grid_stride {} in [0, {}) {{\n",
        pad,
        lp.induction.display(),
        lp.limit.display()
    ));
    for op in &lp.body {
        print_op(out, op, indent + 2);
    }
    out.push_str(&format!("{}}}\n", pad));
}

// ═══════════════════════════════════════════════════════════════════
// Ops
// ═══════════════════════════════════════════════════════════════════

fn print_op(out: &mut String, op: &Op, indent: usize) {
    let pad = spaces(indent);
    let line = match op {
        // ── Constants ────────────────────────────────────────────────────────
        Op::ConstF64 { dst, value } => {
            // Bit-exact hex encoding: 0x<16 lowercase hex digits>
            // This guarantees round-trip for -0.0, NaN bit patterns, subnormals.
            format!("{} = const.f64 {}", dst.display(), f64_hex(*value))
        }
        Op::ConstI32 { dst, value } => {
            format!("{} = const.i32 {}", dst.display(), value)
        }

        // ── Buffer ops ───────────────────────────────────────────────────────
        Op::BufSize { dst, buf } => {
            format!("{} = bufsize {}", dst.display(), buf.display())
        }
        Op::LoadF64 { dst, buf, idx } => {
            format!("{} = load.f64 {}, {}", dst.display(), buf.display(), idx.display())
        }
        Op::StoreF64 { buf, idx, val } => {
            format!("store.f64 {}, {}, {}", buf.display(), idx.display(), val.display())
        }

        // ── Floating-point arithmetic ─────────────────────────────────────────
        Op::FAdd { dst, a, b } => format!("{} = fadd.f64 {}, {}", dst.display(), a.display(), b.display()),
        Op::FSub { dst, a, b } => format!("{} = fsub.f64 {}, {}", dst.display(), a.display(), b.display()),
        Op::FMul { dst, a, b } => format!("{} = fmul.f64 {}, {}", dst.display(), a.display(), b.display()),
        Op::FDiv { dst, a, b } => format!("{} = fdiv.f64 {}, {}", dst.display(), a.display(), b.display()),
        Op::FSqrt { dst, a }   => format!("{} = fsqrt.f64 {}", dst.display(), a.display()),
        Op::FNeg  { dst, a }   => format!("{} = fneg.f64 {}", dst.display(), a.display()),
        Op::FAbs  { dst, a }   => format!("{} = fabs.f64 {}", dst.display(), a.display()),

        // ── Integer arithmetic ────────────────────────────────────────────────
        Op::ConstI64 { dst, value } => format!("{} = const.i64 {}", dst.display(), value),
        Op::IAdd { dst, a, b } => format!("{} = iadd.i32 {}, {}", dst.display(), a.display(), b.display()),
        Op::ISub { dst, a, b } => format!("{} = isub.i32 {}, {}", dst.display(), a.display(), b.display()),
        Op::IMul { dst, a, b } => format!("{} = imul.i32 {}, {}", dst.display(), a.display(), b.display()),
        Op::ICmpLt { dst, a, b } => format!("{} = icmp_lt {}, {}", dst.display(), a.display(), b.display()),
        Op::IAdd64 { dst, a, b } => format!("{} = iadd.i64 {}, {}", dst.display(), a.display(), b.display()),
        Op::ISub64 { dst, a, b } => format!("{} = isub.i64 {}, {}", dst.display(), a.display(), b.display()),
        Op::AndI64 { dst, a, b } => format!("{} = and.i64 {}, {}", dst.display(), a.display(), b.display()),
        Op::OrI64  { dst, a, b } => format!("{} = or.i64 {}, {}", dst.display(), a.display(), b.display()),
        Op::XorI64 { dst, a, b } => format!("{} = xor.i64 {}, {}", dst.display(), a.display(), b.display()),
        Op::ShlI64 { dst, a, shift } => format!("{} = shl.i64 {}, {}", dst.display(), a.display(), shift.display()),
        Op::ShrI64 { dst, a, shift } => format!("{} = shr.i64 {}, {}", dst.display(), a.display(), shift.display()),

        // ── Float ↔ integer conversion ────────────────────────────────────────
        Op::LdExpF64 { dst, mantissa, exp } => format!("{} = ldexp.f64 {}, {}", dst.display(), mantissa.display(), exp.display()),
        Op::F64ToI32Rn { dst, a } => format!("{} = f64_to_i32_rn {}", dst.display(), a.display()),
        Op::BitcastF64ToI64 { dst, a } => format!("{} = bitcast.f64.i64 {}", dst.display(), a.display()),
        Op::BitcastI64ToF64 { dst, a } => format!("{} = bitcast.i64.f64 {}", dst.display(), a.display()),

        // ── Floating-point comparisons ────────────────────────────────────────
        Op::FCmpGt { dst, a, b } => format!("{} = fcmp_gt.f64 {}, {}", dst.display(), a.display(), b.display()),
        Op::FCmpLt { dst, a, b } => format!("{} = fcmp_lt.f64 {}, {}", dst.display(), a.display(), b.display()),
        Op::FCmpEq { dst, a, b } => format!("{} = fcmp_eq.f64 {}, {}", dst.display(), a.display(), b.display()),

        // ── Select ────────────────────────────────────────────────────────────
        Op::SelectF64 { dst, pred, on_true, on_false } => {
            format!("{} = select.f64 {}, {}, {}", dst.display(), pred.display(), on_true.display(), on_false.display())
        }
        Op::SelectI32 { dst, pred, on_true, on_false } => {
            format!("{} = select.i32 {}, {}, {}", dst.display(), pred.display(), on_true.display(), on_false.display())
        }

        // ── Transcendental stubs ─────────────────────────────────────────────
        Op::TamExp { dst, a } => format!("{} = tam_exp.f64 {}", dst.display(), a.display()),
        Op::TamLn  { dst, a } => format!("{} = tam_ln.f64 {}", dst.display(), a.display()),
        Op::TamSin { dst, a } => format!("{} = tam_sin.f64 {}", dst.display(), a.display()),
        Op::TamCos  { dst, a } => format!("{} = tam_cos.f64 {}", dst.display(), a.display()),
        Op::TamAtan { dst, a } => format!("{} = tam_atan.f64 {}", dst.display(), a.display()),
        Op::TamPow { dst, a, b } => format!("{} = tam_pow.f64 {}, {}", dst.display(), a.display(), b.display()),

        // ── Reduction ────────────────────────────────────────────────────────
        Op::ReduceBlockAdd { out_buf, slot_idx, val, order } => {
            format!(
                "reduce_block_add.f64 {}, {}, {} @order({})",
                out_buf.display(), slot_idx.display(), val.display(), order.name()
            )
        }

        // ── Return ────────────────────────────────────────────────────────────
        Op::RetF64 { val } => format!("ret.f64 {}", val.display()),
    };
    out.push_str(&format!("{}{}\n", pad, line));
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn spaces(n: usize) -> String {
    " ".repeat(n)
}

/// Format an f64 as a bit-exact hex literal: `0x<16 lowercase hex digits>`.
///
/// This is the canonical printer format. It round-trips perfectly for every
/// possible f64 value including -0.0, infinities, NaN bit patterns, and
/// subnormals. The parser must accept this form.
///
/// We also accept the human-readable decimal form in the parser (for
/// hand-written `.tam` files), but the printer always emits hex.
pub fn f64_hex(v: f64) -> String {
    format!("0x{:016x}", v.to_bits())
}

// ═══════════════════════════════════════════════════════════════════
// Tests (campsite 1.6 acceptance criteria)
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::variance_pass_program;

    fn make_const_f64_program() -> Program {
        Program {
            version: TamVersion::PHASE1,
            target: Target::Cross,
            funcs: vec![],
            kernels: vec![KernelDef {
                name: "test".into(),
                params: vec![],
                attrs: vec![],
                body: vec![
                    Stmt::Op(Op::ConstF64 { dst: Reg::new("x"), value: 1.0 }),
                ],
            }],
        }
    }

    #[test]
    fn print_header_format() {
        let prog = make_const_f64_program();
        let text = print_program(&prog);
        assert!(text.starts_with(".tam 0.1\n.target cross\n"), "got: {text:?}");
    }

    #[test]
    fn print_const_f64_uses_hex() {
        let prog = make_const_f64_program();
        let text = print_program(&prog);
        // 1.0 in IEEE 754 is 0x3ff0000000000000
        assert!(text.contains("0x3ff0000000000000"), "got: {text:?}");
    }

    #[test]
    fn print_f64_hex_zero() {
        assert_eq!(f64_hex(0.0), "0x0000000000000000");
    }

    #[test]
    fn print_f64_hex_neg_zero() {
        assert_eq!(f64_hex(-0.0f64), "0x8000000000000000");
    }

    #[test]
    fn print_f64_hex_inf() {
        assert_eq!(f64_hex(f64::INFINITY), "0x7ff0000000000000");
    }

    #[test]
    fn print_f64_hex_neg_inf() {
        assert_eq!(f64_hex(f64::NEG_INFINITY), "0xfff0000000000000");
    }

    #[test]
    fn print_variance_pass_kernel_structure() {
        let prog = variance_pass_program();
        let text = print_program(&prog);
        assert!(text.contains("kernel variance_pass("), "missing kernel decl: {text}");
        assert!(text.contains("buf<f64> %data"), "missing buf param: {text}");
        assert!(text.contains("loop_grid_stride %i in [0, %n)"), "missing loop: {text}");
        assert!(text.contains("reduce_block_add.f64"), "missing reduce: {text}");
    }
}
