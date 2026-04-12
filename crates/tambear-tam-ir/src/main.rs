//! `.tam` CLI — load a `.tam` file and execute it on the CPU interpreter.
//!
//! Usage:
//!   cargo run -- <path/to/program.tam> [f64 inputs...]
//!
//! The first kernel in the program is run. The f64 arguments become the
//! first input buffer (`%data` or `%data0`). The output buffer is
//! pre-allocated based on the maximum slot index used in reduce_block_add,
//! or 16 slots if no reduction is found.
//!
//! Campsite 1.14 acceptance criterion:
//!   `cargo run -- sum_all_add.tam 1 2 3 4 5` prints `55`

use std::env;
use std::fs;
use std::process;
use tambear_tam_ir::ast::{Op, Stmt};
use tambear_tam_ir::interp::Interpreter;
use tambear_tam_ir::parse::parse_program;
use tambear_tam_ir::verify::verify;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: tam-ir <program.tam> [f64 inputs...]");
        process::exit(1);
    }

    let path = &args[1];
    let src = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error reading {path}: {e}");
            process::exit(1);
        }
    };

    let prog = match parse_program(&src) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("parse error: {e}");
            process::exit(1);
        }
    };

    let errors = verify(&prog);
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("verify error [{}]: {}", e.context, e.message);
        }
        process::exit(1);
    }

    if prog.kernels.is_empty() {
        eprintln!("error: program contains no kernels");
        process::exit(1);
    }
    let kernel = &prog.kernels[0];

    // Parse f64 inputs from argv[2..]
    let mut data: Vec<f64> = Vec::new();
    for raw in &args[2..] {
        match raw.parse::<f64>() {
            Ok(v) => data.push(v),
            Err(_) => {
                eprintln!("error: could not parse {raw:?} as f64");
                process::exit(1);
            }
        }
    }

    // Determine output buffer size: max slot index used in reduce_block_add + 1.
    // Walk the kernel body to find ConstI32 defs and ReduceBlockAdd slot references.
    let out_size = output_size(kernel);
    let mut out = vec![0.0f64; out_size];

    // Determine the name of the first input buffer param.
    // Kernels have params like (buf<f64> %data, buf<f64> %out).
    // We want the first buf<f64> param that isn't named "out".
    use tambear_tam_ir::ast::Ty;
    let input_buf_name: &str = kernel.params.iter()
        .filter(|p| p.ty == Ty::BufF64 && p.reg.name != "out")
        .map(|p| p.reg.name.as_str())
        .next()
        .unwrap_or("data");

    let interp = Interpreter::new(&prog);
    match interp.run_kernel(&kernel.name, &[(input_buf_name, &mut data), ("out", &mut out)]) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("runtime error: {e}");
            process::exit(1);
        }
    }

    // Print output: one value per line.
    for (i, v) in out.iter().enumerate() {
        println!("out[{i}] = {v}");
    }
}

/// Determine output buffer size by scanning kernel body for ConstI32 defs
/// that feed ReduceBlockAdd slot_idx, then returning max slot + 1.
/// Falls back to 16 if no reductions are found.
fn output_size(kernel: &tambear_tam_ir::ast::KernelDef) -> usize {
    let mut max_slot: Option<i32> = None;

    // Collect const.i32 definitions to know which register holds which slot index.
    let mut const_i32: std::collections::HashMap<String, i32> = std::collections::HashMap::new();

    for stmt in &kernel.body {
        match stmt {
            Stmt::Op(Op::ConstI32 { dst, value }) => {
                const_i32.insert(dst.name.clone(), *value);
            }
            Stmt::Op(Op::ReduceBlockAdd { slot_idx, .. }) => {
                if let Some(&slot) = const_i32.get(&slot_idx.name) {
                    max_slot = Some(match max_slot {
                        None => slot,
                        Some(prev) => prev.max(slot),
                    });
                }
            }
            _ => {}
        }
    }

    match max_slot {
        Some(s) if s >= 0 => (s as usize) + 1,
        _ => 16,
    }
}
