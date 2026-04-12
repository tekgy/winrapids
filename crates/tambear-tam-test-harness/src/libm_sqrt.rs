//! Campsite 2.4 — `tam_sqrt` chain test.
//!
//! This is the **end-to-end chain test** that proves the full stack:
//!
//!   1. `peak2-libm/tam_sqrt.tam` source file
//!   2. `tambear_tam_ir::parse::parse_program` parses it
//!   3. `tambear_tam_ir::verify::verify` checks it
//!   4. `tambear_tam_ir::interp::Interpreter::call_func` executes it on one fp64 input
//!   5. Reference is `peak2-libm/sqrt-1k.bin` (mpmath at 50 digits, rounded to fp64)
//!   6. `UlpReport::measure` computes max/mean ULP distance
//!   7. `passes(0)` asserts IEEE 754 correctly-rounded equivalence (max_ulp == 0)
//!
//! tam_sqrt is the ONLY Phase 1 function with a `max_ulp = 0` bound because
//! IEEE 754 mandates that `fsqrt` is correctly rounded. The interpreter's
//! `Op::FSqrt` lowers to Rust stdlib `f64::sqrt()`, which lowers to LLVM's
//! `llvm.sqrt.f64` intrinsic, which lowers to the hardware's `sqrtsd` (x86)
//! or `fsqrt` (ARM) instruction. **No libm call on the I1 path.**
//!
//! If this test fails, the bug is either:
//!   - In the harness/reference pipeline (gen-reference.py, TAMBLMR1 reader, ulp_distance), OR
//!   - In the tambear-tam-ir parser or interpreter's FSqrt lowering, OR
//!   - On hardware that doesn't implement IEEE 754 fsqrt correctly (vanishingly unlikely).
//!
//! It CANNOT be an "algorithm bug" because there is no algorithm here beyond
//! `return fsqrt.f64 %x`. That's what makes it the ideal chain test per
//! navigator's 2026-04-12 sequencing decision.

use crate::ulp_harness::{read_reference_bin, UlpReport};

const TAM_SQRT_SRC: &str = r#"
.tam 0.1
.target cross

func tam_sqrt(f64 %x) -> f64 {
entry:
  %result = fsqrt.f64 %x
  ret.f64 %result
}
"#;

const SQRT_REF_PATH: &str = "../../campsites/expedition/20260411120000-the-bit-exact-trek/peak2-libm/sqrt-1k.bin";
const SQRT_REF_1M_PATH: &str = "../../campsites/expedition/20260411120000-the-bit-exact-trek/peak2-libm/sqrt-1m.bin";

/// Parse tam_sqrt.tam and build the interpreter-backed candidate closure.
///
/// Returns a tuple of (owned Program, a closure that runs it on one f64).
/// The caller passes the closure to `UlpReport::measure`.
fn build_candidate() -> Result<tambear_tam_ir::ast::Program, String> {
    let prog = tambear_tam_ir::parse::parse_program(TAM_SQRT_SRC)
        .map_err(|e| format!("parse error: {e}"))?;
    let errors = tambear_tam_ir::verify::verify(&prog);
    if !errors.is_empty() {
        let msg: Vec<String> = errors.iter()
            .map(|e| format!("[{}] {}", e.context, e.message))
            .collect();
        return Err(format!("verify errors:\n{}", msg.join("\n")));
    }
    Ok(prog)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tambear_tam_ir::interp::Interpreter;

    /// Campsite 2.4 acceptance test.
    ///
    /// Skipped if the reference file hasn't been generated yet (run
    /// `gen-reference.py --function sqrt --n 1000 --out sqrt-1k.bin`).
    #[test]
    fn campsite_2_4_tam_sqrt_zero_ulp() {
        if !std::path::Path::new(SQRT_REF_PATH).exists() {
            eprintln!("skipping: {} not present (run gen-reference.py to generate)", SQRT_REF_PATH);
            return;
        }

        // Step 1: parse and verify
        let prog = build_candidate().expect("tam_sqrt.tam must parse cleanly");
        assert_eq!(prog.funcs.len(), 1, "exactly one func in tam_sqrt.tam");
        assert_eq!(prog.funcs[0].name, "tam_sqrt");

        let interp = Interpreter::new(&prog);

        // Step 2: load the mpmath reference
        let (header, records) = read_reference_bin(SQRT_REF_PATH)
            .expect("sqrt-1k.bin must be readable");
        assert_eq!(header.function, "sqrt");
        assert_eq!(header.domain, "primary");
        assert_eq!(header.mpmath_digits, 50);
        assert_eq!(records.len(), 1000);

        // Step 3: measure via the .tam interpreter
        let report = UlpReport::measure(&records, |x| {
            interp.call_func("tam_sqrt", &[x])
                .expect("tam_sqrt should never fail for a finite positive input")
        });

        println!("tam_sqrt report: {}", report.summary());

        // Step 4: assert Phase 1 acceptance (max_ulp == 0, IEEE 754 mandated)
        assert!(
            report.passes(0),
            "tam_sqrt must be 0 ULP (IEEE 754 correctly-rounded fsqrt). \
             Got: {}. Worst case: {:?}",
            report.summary(),
            report.worst,
        );
    }

    /// Special-value coverage (the 5 cases navigator listed in the 2.4 acceptance criterion).
    ///
    /// These are injected explicitly because the 1k random sample does not
    /// cover them — specifically `-0.0`, `-1.0`, `+inf`, `nan` — which can
    /// only be forced via direct test cases.
    #[test]
    fn campsite_2_4_tam_sqrt_special_values() {
        let prog = build_candidate().expect("tam_sqrt.tam must parse cleanly");
        let interp = Interpreter::new(&prog);

        let call = |x: f64| interp.call_func("tam_sqrt", &[x])
            .expect("tam_sqrt call should not error");

        // sqrt(+0) = +0, bit-exact (including sign)
        let r = call(0.0);
        assert_eq!(r.to_bits(), 0.0f64.to_bits(),
            "sqrt(+0) must be +0 bit-exact, got {r}");

        // sqrt(-0) = -0, sign preserved per IEEE 754 §6.3
        let r = call(-0.0);
        assert_eq!(r.to_bits(), (-0.0f64).to_bits(),
            "sqrt(-0) must be -0 bit-exact (sign preserved per IEEE 754 §6.3), got {r}");

        // sqrt(+inf) = +inf
        let r = call(f64::INFINITY);
        assert_eq!(r, f64::INFINITY, "sqrt(+inf) must be +inf");

        // sqrt(-1) = NaN (invalid operation)
        let r = call(-1.0);
        assert!(r.is_nan(), "sqrt(-1) must be NaN");

        // sqrt(NaN) = NaN (I11: NaN propagates through every op)
        let r = call(f64::NAN);
        assert!(r.is_nan(), "sqrt(NaN) must be NaN (I11)");

        // sqrt(1) = 1, bit-exact (perfect square)
        let r = call(1.0);
        assert_eq!(r, 1.0);

        // sqrt(4) = 2, bit-exact (perfect square)
        let r = call(4.0);
        assert_eq!(r, 2.0);

        // sqrt(0.25) = 0.5, bit-exact
        let r = call(0.25);
        assert_eq!(r, 0.5);

        // sqrt(subnormal) should be approximately sqrt(sub), not flushed to 0.
        // Smallest positive subnormal: 5e-324. sqrt = ~2.22e-162.
        let sub = f64::from_bits(1);
        let r = call(sub);
        assert!(r > 0.0, "sqrt(subnormal) must not flush to 0, got {r}");
        assert!(r.is_finite(), "sqrt(subnormal) must be finite, got {r}");
        // Per IEEE 754 §5.4.1, sqrt of a subnormal is correctly rounded.
        // f64::sqrt inherits this directly.
        assert_eq!(r, sub.sqrt(),
            "sqrt(subnormal) via tam_sqrt must match f64::sqrt bit-exactly");
    }

    /// Full 1M-sample acceptance test, gated on the 48 MB `sqrt-1m.bin` file
    /// being present locally. The file is generated by running
    /// `gen-reference.py --function sqrt --n 1000000 --out sqrt-1m.bin` and is
    /// NOT committed to the repo (too large). This test runs as a local
    /// verification when the file is available.
    #[test]
    fn campsite_2_4_tam_sqrt_1m_sample_zero_ulp() {
        if !std::path::Path::new(SQRT_REF_1M_PATH).exists() {
            eprintln!("skipping: {} not present (run gen-reference.py --n 1000000 to generate)",
                SQRT_REF_1M_PATH);
            return;
        }

        let prog = build_candidate().expect("parse");
        let interp = Interpreter::new(&prog);
        let (header, records) = read_reference_bin(SQRT_REF_1M_PATH).expect("read 1m");
        assert_eq!(header.n_samples, 1_000_000);
        assert_eq!(records.len(), 1_000_000);

        let start = std::time::Instant::now();
        let report = UlpReport::measure(&records, |x| {
            interp.call_func("tam_sqrt", &[x]).unwrap()
        });
        let elapsed = start.elapsed();

        println!("tam_sqrt 1M-sample report: {} (elapsed: {:?})", report.summary(), elapsed);

        assert!(
            report.passes(0),
            "tam_sqrt at 1M samples must be 0 ULP, got {}. Worst: {:?}",
            report.summary(), report.worst,
        );
    }

    /// Parser smoke test: make sure tam_sqrt.tam has the expected shape.
    #[test]
    fn tam_sqrt_parses_and_verifies() {
        let prog = build_candidate().expect("tam_sqrt.tam must parse");
        assert_eq!(prog.funcs.len(), 1);
        let func = &prog.funcs[0];
        assert_eq!(func.name, "tam_sqrt");
        assert_eq!(func.params.len(), 1, "one f64 parameter");
        assert_eq!(func.body.len(), 2, "one FSqrt op + one RetF64 = 2 stmts");

        use tambear_tam_ir::ast::Op;
        assert!(matches!(func.body[0], Op::FSqrt { .. }),
            "first op should be FSqrt, got {:?}", func.body[0]);
        assert!(matches!(func.body[1], Op::RetF64 { .. }),
            "last op should be RetF64, got {:?}", func.body[1]);
    }
}
