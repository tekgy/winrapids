//! Test fixtures: pre-built Programs for use in tests across all modules.
//!
//! This module is compiled only in test mode. It provides canonical
//! AST representations of the three reference programs, so tests don't
//! each have to re-build them.

use crate::ast::*;

/// Build the `variance_pass` kernel as an AST.
///
/// This is the canonical three-slot accumulation kernel:
/// out[0] = sum(x), out[1] = sum(x^2), out[2] = count
pub fn variance_pass_program() -> Program {
    use Op::*;
    Program {
        version: TamVersion::PHASE1,
        target: Target::Cross,
        funcs: vec![],
        kernels: vec![KernelDef {
            name: "variance_pass".into(),
            params: vec![
                KernelParam { ty: Ty::BufF64, reg: Reg::new("data") },
                KernelParam { ty: Ty::BufF64, reg: Reg::new("out") },
            ],
            attrs: vec![],
            body: vec![
                Stmt::Op(BufSize { dst: Reg::new("n"), buf: Reg::new("data") }),
                Stmt::Op(ConstF64 { dst: Reg::new("acc0"), value: 0.0 }),
                Stmt::Op(ConstF64 { dst: Reg::new("acc1"), value: 0.0 }),
                Stmt::Op(ConstF64 { dst: Reg::new("acc2"), value: 0.0 }),
                Stmt::Loop(LoopGridStride {
                    induction: Reg::new("i"),
                    limit: Reg::new("n"),
                    body: vec![
                        LoadF64 { dst: Reg::new("v"), buf: Reg::new("data"), idx: Reg::new("i") },
                        FMul { dst: Reg::new("v2"), a: Reg::new("v"), b: Reg::new("v") },
                        ConstF64 { dst: Reg::new("one"), value: 1.0 },
                        FAdd { dst: Reg::prime("acc0"), a: Reg::new("acc0"), b: Reg::new("v") },
                        FAdd { dst: Reg::prime("acc1"), a: Reg::new("acc1"), b: Reg::new("v2") },
                        FAdd { dst: Reg::prime("acc2"), a: Reg::new("acc2"), b: Reg::new("one") },
                    ],
                }),
                Stmt::Op(ConstI32 { dst: Reg::new("s0"), value: 0 }),
                Stmt::Op(ConstI32 { dst: Reg::new("s1"), value: 1 }),
                Stmt::Op(ConstI32 { dst: Reg::new("s2"), value: 2 }),
                Stmt::Op(ReduceBlockAdd {
                    out_buf: Reg::new("out"), slot_idx: Reg::new("s0"), val: Reg::prime("acc0"),
                    order: OrderStrategyRef::new("sequential_left"),
                }),
                Stmt::Op(ReduceBlockAdd {
                    out_buf: Reg::new("out"), slot_idx: Reg::new("s1"), val: Reg::prime("acc1"),
                    order: OrderStrategyRef::new("sequential_left"),
                }),
                Stmt::Op(ReduceBlockAdd {
                    out_buf: Reg::new("out"), slot_idx: Reg::new("s2"), val: Reg::prime("acc2"),
                    order: OrderStrategyRef::new("sequential_left"),
                }),
            ],
        }],
    }
}

/// Build the `sum_all_add` kernel as an AST.
pub fn sum_all_add_program() -> Program {
    use Op::*;
    Program {
        version: TamVersion::PHASE1,
        target: Target::Cross,
        funcs: vec![],
        kernels: vec![KernelDef {
            name: "sum_all_add".into(),
            params: vec![
                KernelParam { ty: Ty::BufF64, reg: Reg::new("data") },
                KernelParam { ty: Ty::BufF64, reg: Reg::new("out") },
            ],
            attrs: vec![],
            body: vec![
                Stmt::Op(BufSize { dst: Reg::new("n"), buf: Reg::new("data") }),
                Stmt::Op(ConstF64 { dst: Reg::new("acc"), value: 0.0 }),
                Stmt::Loop(LoopGridStride {
                    induction: Reg::new("i"),
                    limit: Reg::new("n"),
                    body: vec![
                        LoadF64 { dst: Reg::new("v"), buf: Reg::new("data"), idx: Reg::new("i") },
                        FAdd { dst: Reg::prime("acc"), a: Reg::new("acc"), b: Reg::new("v") },
                    ],
                }),
                Stmt::Op(ConstI32 { dst: Reg::new("s0"), value: 0 }),
                Stmt::Op(ReduceBlockAdd {
                    out_buf: Reg::new("out"), slot_idx: Reg::new("s0"), val: Reg::prime("acc"),
                    order: OrderStrategyRef::new("sequential_left"),
                }),
            ],
        }],
    }
}
