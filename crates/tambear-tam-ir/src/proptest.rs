//! Property tests for the `.tam` IR.
//!
//! Campsite 1.8: round-trip 10,000 random well-typed programs.
//! Every program: print → parse → assert structurally equal (using bit-eq for f64).
//!
//! We use a simple linear congruential generator (LCG) instead of pulling
//! in a `rand` crate dependency. The goal is deterministic, comprehensive
//! coverage of the op space — not cryptographic randomness.

use crate::ast::*;
use crate::parse::parse_program;
use crate::print::print_program;

// ═══════════════════════════════════════════════════════════════════
// Pseudo-random generator (LCG, no deps)
// ═══════════════════════════════════════════════════════════════════

struct Rng {
    state: u64,
}

#[allow(dead_code)]
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_add(1) }
    }

    /// Next u64 in the sequence.
    fn next_u64(&mut self) -> u64 {
        // Knuth's multiplicative LCG (64-bit)
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 0
    }

    fn next_i32(&mut self) -> i32 {
        self.next_u64() as i32
    }

    fn next_f64_bits(&mut self) -> f64 {
        // Generate a random finite f64 (not NaN, not inf) for cleaner programs.
        // We want variety but also valid values.
        let choices: &[f64] = &[
            0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 42.0, -42.0,
            1e100, -1e100, 1e-100, f64::MIN_POSITIVE,
            std::f64::consts::PI, std::f64::consts::E,
        ];
        let idx = self.next_usize(choices.len());
        choices[idx]
    }

    fn next_name(&mut self, prefix: &str, max: usize) -> String {
        format!("{}{}", prefix, self.next_usize(max))
    }
}

// ═══════════════════════════════════════════════════════════════════
// Program generator
// ═══════════════════════════════════════════════════════════════════

/// Generate a random well-typed kernel.
///
/// A "well-typed" program here means:
/// - All register references are defined before use (no dangling references).
/// - All types match the ops that consume them.
/// - The structure is valid per the spec.
///
/// We generate simple programs: a setup block of consts, one optional loop,
/// then a reduce block. This covers the main structural patterns without
/// needing a full type inference engine.
fn random_kernel(rng: &mut Rng, kernel_idx: usize) -> KernelDef {
    use Op::*;

    let name = format!("kernel_{}", kernel_idx);
    let n_input_bufs = 1 + rng.next_usize(2); // 1 or 2 input buffers
    let n_output_slots = 1 + rng.next_usize(4); // 1 to 4 output slots

    let mut params: Vec<KernelParam> = (0..n_input_bufs)
        .map(|i| KernelParam {
            ty: Ty::BufF64,
            reg: Reg::new(format!("data{}", i)),
        })
        .collect();
    params.push(KernelParam {
        ty: Ty::BufF64,
        reg: Reg::new("out"),
    });

    let mut body: Vec<Stmt> = vec![];

    // Define n (buffer length)
    body.push(Stmt::Op(BufSize { dst: Reg::new("n"), buf: Reg::new("data0") }));

    // Define some f64 accumulators
    let n_acc = 1 + rng.next_usize(4);
    for j in 0..n_acc {
        body.push(Stmt::Op(ConstF64 {
            dst: Reg::new(format!("acc{}", j)),
            value: 0.0,
        }));
    }

    // Maybe add some constants
    let n_consts = rng.next_usize(3);
    for j in 0..n_consts {
        let v = rng.next_f64_bits();
        body.push(Stmt::Op(ConstF64 {
            dst: Reg::new(format!("c{}", j)),
            value: v,
        }));
    }

    // Loop body
    let mut loop_body: Vec<Op> = vec![];

    // Load from a buffer
    loop_body.push(LoadF64 {
        dst: Reg::new("v"),
        buf: Reg::new("data0"),
        idx: Reg::new("i"),
    });

    // If two input buffers, optionally load from second
    if n_input_bufs >= 2 && rng.next_bool() {
        loop_body.push(LoadF64 {
            dst: Reg::new("v2"),
            buf: Reg::new("data1"),
            idx: Reg::new("i"),
        });
    }

    // Build up some arithmetic using v
    // We chain operations to stress the phi pairs
    for j in 0..n_acc {
        let base = if j == 0 { Reg::new("v") } else { Reg::new("v") };
        let prev_acc = Reg::new(format!("acc{}", j));
        // Choose between fadd, fsub, fmul
        let op_choice = rng.next_usize(3);
        let new_prime = Reg::prime(format!("acc{}", j));
        match op_choice {
            0 => loop_body.push(FAdd { dst: new_prime, a: prev_acc, b: base }),
            1 => loop_body.push(FMul { dst: new_prime, a: prev_acc, b: base }),
            _ => loop_body.push(FSub { dst: new_prime, a: prev_acc, b: base }),
        }
    }

    body.push(Stmt::Loop(LoopGridStride {
        induction: Reg::new("i"),
        limit: Reg::new("n"),
        body: loop_body,
    }));

    // Reduction ops (capped at n_output_slots)
    for j in 0..n_output_slots.min(n_acc) {
        let slot_name = format!("s{}", j);
        body.push(Stmt::Op(ConstI32 {
            dst: Reg::new(&slot_name),
            value: j as i32,
        }));
        body.push(Stmt::Op(ReduceBlockAdd {
            out_buf: Reg::new("out"),
            slot_idx: Reg::new(&slot_name),
            val: Reg::prime(format!("acc{}", j)),
            order: OrderStrategy::SequentialLeft,
        }));
    }

    KernelDef { name, params, attrs: vec![], body }
}

/// Generate a minimal well-typed function.
fn random_func(rng: &mut Rng, func_idx: usize) -> FuncDef {
    use Op::*;
    let name = format!("func_{}", func_idx);
    let params = vec![FuncParam { reg: Reg::new("x") }];
    // Simple function: compute x*x and return it
    let ops = match rng.next_usize(4) {
        0 => vec![
            FMul { dst: Reg::new("r"), a: Reg::new("x"), b: Reg::new("x") },
            RetF64 { val: Reg::new("r") },
        ],
        1 => vec![
            FAdd { dst: Reg::new("r"), a: Reg::new("x"), b: Reg::new("x") },
            RetF64 { val: Reg::new("r") },
        ],
        2 => vec![
            FNeg { dst: Reg::new("r"), a: Reg::new("x") },
            RetF64 { val: Reg::new("r") },
        ],
        _ => vec![
            FSqrt { dst: Reg::new("r"), a: Reg::new("x") },
            RetF64 { val: Reg::new("r") },
        ],
    };
    FuncDef { name, params, body: ops }
}

fn random_program(rng: &mut Rng, prog_idx: usize) -> Program {
    let n_funcs = rng.next_usize(3);
    let n_kernels = 1 + rng.next_usize(3);

    let funcs = (0..n_funcs).map(|i| random_func(rng, i)).collect();
    let kernels = (0..n_kernels).map(|i| random_kernel(rng, prog_idx * 10 + i)).collect();

    Program {
        version: TamVersion::PHASE1,
        target: Target::Cross,
        funcs,
        kernels,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Bit-equality comparison (needed for f64 in AST)
// ═══════════════════════════════════════════════════════════════════

fn ops_bit_eq(a: &Op, b: &Op) -> bool {
    match (a, b) {
        (Op::ConstF64 { dst: da, value: va }, Op::ConstF64 { dst: db, value: vb }) => {
            da == db && va.to_bits() == vb.to_bits()
        }
        _ => a == b,
    }
}

fn stmts_bit_eq(a: &Stmt, b: &Stmt) -> bool {
    match (a, b) {
        (Stmt::Op(oa), Stmt::Op(ob)) => ops_bit_eq(oa, ob),
        (Stmt::Loop(la), Stmt::Loop(lb)) => {
            la.induction == lb.induction
                && la.limit == lb.limit
                && la.body.len() == lb.body.len()
                && la.body.iter().zip(lb.body.iter()).all(|(oa, ob)| ops_bit_eq(oa, ob))
        }
        _ => false,
    }
}

fn kernels_bit_eq(a: &KernelDef, b: &KernelDef) -> bool {
    a.name == b.name
        && a.params == b.params
        && a.body.len() == b.body.len()
        && a.body.iter().zip(b.body.iter()).all(|(sa, sb)| stmts_bit_eq(sa, sb))
}

fn funcs_bit_eq(a: &FuncDef, b: &FuncDef) -> bool {
    a.name == b.name
        && a.params == b.params
        && a.body.len() == b.body.len()
        && a.body.iter().zip(b.body.iter()).all(|(oa, ob)| ops_bit_eq(oa, ob))
}

fn programs_bit_eq(a: &Program, b: &Program) -> bool {
    a.version == b.version
        && a.target == b.target
        && a.funcs.len() == b.funcs.len()
        && a.kernels.len() == b.kernels.len()
        && a.funcs.iter().zip(b.funcs.iter()).all(|(fa, fb)| funcs_bit_eq(fa, fb))
        && a.kernels.iter().zip(b.kernels.iter()).all(|(ka, kb)| kernels_bit_eq(ka, kb))
}

// ═══════════════════════════════════════════════════════════════════
// Tests (campsite 1.8: 10,000 round-trips)
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Campsite 1.8: round-trip 10,000 random programs.
    ///
    /// For each: print → parse → assert structurally equal (bit-eq for f64).
    /// Zero divergences required.
    #[test]
    fn roundtrip_10000_random_programs() {
        let mut rng = Rng::new(0xdeadbeef_cafef00d);
        let mut divergences = 0usize;
        let n = 10_000;

        for i in 0..n {
            let original = random_program(&mut rng, i);
            let text = print_program(&original);
            match parse_program(&text) {
                Err(e) => {
                    eprintln!("program {}: parse error: {}\n\ntext:\n{}", i, e, text);
                    divergences += 1;
                }
                Ok(parsed) => {
                    if !programs_bit_eq(&original, &parsed) {
                        eprintln!("program {}: round-trip divergence\n\noriginal text:\n{}", i, text);
                        divergences += 1;
                    }
                }
            }
            if divergences >= 10 {
                panic!("too many divergences ({}), aborting at program {}", divergences, i);
            }
        }

        assert_eq!(divergences, 0,
            "{} / {} programs failed round-trip", divergences, n);
    }
}
