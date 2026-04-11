//! Coverage test: can the existing atoms express these recipes?
//! If this file COMPILES AND PASSES, the atoms are sufficient.

use crate::tbs::Expr;
use crate::accumulates::{AccumulateSlot, Grouping, Op, fuse_passes, execute_pass_cpu};
use std::collections::HashMap;

fn v() -> Expr { Expr::val() }
fn v2() -> Expr { Expr::val2() }
fn r() -> Expr { Expr::Ref }
fn c(x: f64) -> Expr { Expr::lit(x) }
fn var(s: &str) -> Expr { Expr::var(s) }

fn aa(expr: Expr, name: &str) -> AccumulateSlot {
    AccumulateSlot { expr, grouping: Grouping::All, op: Op::Add, output: name.into() }
}

#[test]
fn mean_atoms() {
    let slots = vec![aa(v(), "sum"), aa(c(1.0), "n")];
    let passes = fuse_passes(&slots);
    assert_eq!(passes.len(), 1);
    let results = execute_pass_cpu(&passes[0], &[1.0, 2.0, 3.0, 4.0, 5.0], 0.0, &[]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    let mean = crate::tbs::eval(&var("sum").div(var("n")), 0.0, 0.0, 0.0, &vars);
    assert!((mean - 3.0).abs() < 1e-14);
}

#[test]
fn variance_atoms() {
    let slots = vec![aa(v(), "s"), aa(v().sq(), "ss"), aa(c(1.0), "n")];
    let passes = fuse_passes(&slots);
    let results = execute_pass_cpu(&passes[0], &[1.0, 2.0, 3.0, 4.0, 5.0], 0.0, &[]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    let variance = crate::tbs::eval(
        &var("ss").sub(var("s").mul(var("s")).div(var("n"))).div(var("n").sub(c(1.0))),
        0.0, 0.0, 0.0, &vars,
    );
    assert!((variance - 2.5).abs() < 1e-14);
}

#[test]
fn skewness_needs_cubed() {
    let slots = vec![
        aa(v(), "s"), aa(v().sq(), "ss"),
        aa(v().pow(c(3.0)), "s3"), aa(c(1.0), "n"),
    ];
    let passes = fuse_passes(&slots);
    assert_eq!(passes.len(), 1);
    let results = execute_pass_cpu(&passes[0], &[1.0, 2.0, 3.0, 4.0, 5.0], 0.0, &[]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    assert_eq!(*vars.get("s3").unwrap(), 225.0);
}

#[test]
fn kurtosis_needs_fourth() {
    let slots = vec![
        aa(v(), "s"), aa(v().sq(), "ss"),
        aa(v().pow(c(3.0)), "s3"), aa(v().pow(c(4.0)), "s4"),
        aa(c(1.0), "n"),
    ];
    let passes = fuse_passes(&slots);
    assert_eq!(passes.len(), 1);
    let results = execute_pass_cpu(&passes[0], &[1.0, 2.0, 3.0], 0.0, &[]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    assert_eq!(*vars.get("s4").unwrap(), 98.0);
}

#[test]
fn pearson_cross_product() {
    let slots = vec![
        aa(v(), "sx"), aa(v().sq(), "ssx"),
        aa(v().mul(v2()), "sxy"), aa(c(1.0), "n"),
    ];
    let passes = fuse_passes(&slots);
    assert_eq!(passes.len(), 1);
    let results = execute_pass_cpu(&passes[0], &[1.0, 2.0, 3.0], 0.0, &[2.0, 4.0, 6.0]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    assert_eq!(*vars.get("sxy").unwrap(), 28.0);
}

#[test]
fn geometric_mean() {
    let slots = vec![aa(v().ln(), "ls"), aa(c(1.0), "n")];
    let passes = fuse_passes(&slots);
    let results = execute_pass_cpu(&passes[0], &[2.0, 8.0], 0.0, &[]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    let geo = crate::tbs::eval(&var("ls").div(var("n")).exp(), 0.0, 0.0, 0.0, &vars);
    assert!((geo - 4.0).abs() < 1e-14);
}

#[test]
fn harmonic_mean() {
    let slots = vec![aa(v().recip(), "rs"), aa(c(1.0), "n")];
    let passes = fuse_passes(&slots);
    let results = execute_pass_cpu(&passes[0], &[60.0, 40.0], 0.0, &[]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    let harm = crate::tbs::eval(&var("n").div(var("rs")), 0.0, 0.0, 0.0, &vars);
    assert!((harm - 48.0).abs() < 1e-12);
}

#[test]
fn l1_norm() {
    let slots = vec![aa(v().abs(), "l1")];
    let passes = fuse_passes(&slots);
    let results = execute_pass_cpu(&passes[0], &[-3.0, 4.0, -5.0], 0.0, &[]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    assert_eq!(*vars.get("l1").unwrap(), 12.0);
}

#[test]
fn count_positives() {
    let slots = vec![aa(Expr::Gt(Box::new(v()), Box::new(c(0.0))), "cp")];
    let passes = fuse_passes(&slots);
    let results = execute_pass_cpu(&passes[0], &[-2.0, -1.0, 0.0, 1.0, 2.0], 0.0, &[]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    assert_eq!(*vars.get("cp").unwrap(), 2.0);
}

#[test]
fn centered_squared_deviation() {
    let slots = vec![aa(v().sub(r()).sq(), "ss_dev")];
    let passes = fuse_passes(&slots);
    let results = execute_pass_cpu(&passes[0], &[1.0, 2.0, 3.0, 4.0, 5.0], 3.0, &[]);
    let vars: HashMap<String, f64> = results.into_iter().collect();
    assert_eq!(*vars.get("ss_dev").unwrap(), 10.0);
}

#[test]
fn eleven_stats_one_pass() {
    let slots = vec![
        aa(v(), "sum"),
        aa(c(1.0), "count"),
        aa(v().sq(), "sum_sq"),
        aa(v().pow(c(3.0)), "sum_cubed"),
        aa(v().pow(c(4.0)), "sum_4th"),
        aa(v().ln(), "log_sum"),
        aa(v().recip(), "recip_sum"),
        aa(v().abs(), "abs_sum"),
        aa(v().mul(v2()), "sum_xy"),
        aa(Expr::Gt(Box::new(v()), Box::new(c(0.0))), "count_pos"),
        aa(Expr::IsFinite(Box::new(v())), "count_finite"),
    ];
    let passes = fuse_passes(&slots);
    assert_eq!(passes.len(), 1, "ALL ELEVEN stats fuse into ONE data pass");
    assert_eq!(passes[0].slots.len(), 11);
}
