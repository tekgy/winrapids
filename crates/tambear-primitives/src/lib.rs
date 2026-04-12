//! # tambear-primitives
//!
//! The alphabet of computation. Three layers:
//!
//! - **TBS Expr** — the universal expression AST (one type for everything)
//! - **Accumulates** — (Grouping × Op) pairs that combine Expr-transformed elements
//! - **Gathers** — Expr over accumulated results
//!
//! Math is expressed as: Expr → Accumulate → Gather(Expr) chains.
//! Multiple chains fuse into single passes when they share (Grouping, Op).
//!
//! "Mean arithmetic" is not a primitive. It's a RECIPE:
//! ```text
//! Expr::val()    → Accumulate(All, Add) → "sum"
//! Expr::lit(1.0) → Accumulate(All, Add) → "count"    ← FUSES with above
//! Gather(Expr::var("sum").div(Expr::var("count")))
//! ```
//!
//! The atoms are: val, lit, sq, ln, exp, abs, add, sub, mul, div, pow.
//! The recipe composes them. TAM compiles and executes recipes.

pub mod tbs;
pub mod accumulates;
pub mod gathers;
pub mod recipes;
pub mod codegen;
