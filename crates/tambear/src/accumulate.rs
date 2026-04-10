//! Unified accumulate API — THE computation primitive.
//!
//! Every ML/DataFrame operation is:
//! ```text
//! accumulate(data, grouping, expr, op)
//! ```
//! — a choice from four menus: addressing × grouping × expr × op.
//!
//! This module provides a single dispatch surface that routes to the right
//! primitive (scatter, reduce, scan, tiled) based on the grouping pattern.
//!
//! # Architecture
//!
//! | Grouping | Dispatches to | Status |
//! |----------|--------------|--------|
//! | `All` + Add | scatter_phi(keys=0, n=1) | ✓ |
//! | `All` + ArgMin/ArgMax | ReduceOp | ✓ |
//! | `ByKey` + Add | ScatterJit::scatter_phi | ✓ |
//! | `ByKey` + multi-expr + Add | ScatterJit::scatter_multi_phi | ✓ |
//! | `Masked` + ByKey + Add | ScatterJit::scatter_phi_masked | ✓ |
//! | `Prefix` | winrapids-scan (AssociativeOp) | todo |
//! | `Segmented` | winrapids-scan segmented mode | todo |
//! | `Tiled` + DotProduct/Distance | winrapids-tiled::TiledEngine | ✓ |
//! | `Windowed` | prefix subtraction trick | todo |
//!
//! The `todo!()` groupings name the dispatch target so future wiring
//! is a one-line slot-in, not a design decision.

use std::sync::Arc;

use crate::compute_engine::ComputeEngine;
use crate::reduce_op::ReduceOp;
use winrapids_tiled::{TiledEngine, DotProductOp, DistanceOp};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// WHERE to write accumulation results / how to partition the data.
///
/// This is the single axis that distinguishes reduce from scatter from scan.
#[derive(Debug, Clone)]
pub enum Grouping<'a> {
    /// N → 1: all elements contribute to one accumulator.
    ///
    /// Dispatches to: scatter_phi(keys_all_zero, n_groups=1) for Add,
    /// or ReduceOp for ArgMin/ArgMax.
    All,

    /// N → K: scatter by integer key column.
    ///
    /// Dispatches to: ScatterJit::scatter_phi / scatter_multi_phi.
    /// `keys[i]` must be in `0..n_groups`.
    ByKey {
        keys: &'a [i32],
        n_groups: usize,
    },

    /// N → N (skip where mask bit is 0): filter fused into accumulation.
    ///
    /// Dispatches to: ScatterJit::scatter_phi_masked.
    /// `mask` is a packed u64 bitmask (1 bit per row, row-major).
    Masked {
        keys: &'a [i32],
        n_groups: usize,
        mask: &'a [u64],
    },

    /// N → N: prefix (forward scan). Requires winrapids-scan AssociativeOp.
    Prefix,

    /// N → N: prefix with reset at segment boundaries. Requires winrapids-scan.
    Segmented,

    /// M×K × K×N → M×N: tiled blocked accumulation via TiledEngine.
    ///
    /// `b` is the second matrix (row-major, K×N). The first matrix comes from
    /// the `values` parameter of `accumulate()`.
    Tiled { b: &'a [f64], m: usize, n: usize, k: usize },

    /// N → N: rolling window (prefix subtraction trick). Requires winrapids-scan.
    Windowed { size: usize },
}

/// WHAT to compute per element before combining.
///
/// Maps to the `phi_expr` parameter in ScatterJit — the lift function in
/// decomposable accumulation. Variable `v` = current element value,
/// `r` = reference value (per-group, e.g. group mean for centering).
#[derive(Debug, Clone)]
pub enum Expr<'a> {
    /// Raw value: `phi = "v"`. Identity lift.
    Value,
    /// Squared value: `phi = "v * v"`. For sum-of-squares.
    ValueSq,
    /// Constant 1: `phi = "1.0"`. For counting.
    One,
    /// Cross-expression: `phi = "v * r"`. For weighted sums.
    WeightedByRef,
    /// Custom CUDA expression string. `v` and `r` are in scope.
    Custom(&'a str),
}

impl Expr<'_> {
    fn as_phi(&self) -> &str {
        match self {
            Expr::Value        => "v",
            Expr::ValueSq      => "v * v",
            Expr::One          => "1.0",
            Expr::WeightedByRef => "v * r",
            Expr::Custom(s)    => s,
        }
    }
}

/// HOW to combine accumulators.
///
/// Each variant specifies a semiring over which the accumulation runs:
/// - `Add`/`Max`/`Min` inhabit the real-number semirings.
/// - `DotProduct`/`Distance` inhabit the matrix (bilinear) semiring.
/// - Future: `TropicalMinPlus` would inhabit the min-plus semiring (add=min, mul=+),
///   enabling Viterbi, PELT (unpruned), and all-pairs shortest-path as Kingdom A
///   computations. Its identity is (+∞, 0) — (additive identity, multiplicative identity).
///   Gap noted 2026-04-10 from tropical semiring analysis of PELT/Viterbi structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Additive monoid (ℝ, +). Maps to atomicAdd. The default.
    Add,
    /// Per-group maximum. CAS-loop f64 atomic. Initialises to -∞.
    Max,
    /// Per-group minimum. CAS-loop f64 atomic. Initialises to +∞.
    Min,
    /// Select minimum: (value, index) pair-reduction.
    ArgMin,
    /// Select maximum: (value, index) pair-reduction.
    ArgMax,
    /// Tiled dot product: C[i,j] = Σ_k A[i,k] * B[k,j]. For `Grouping::Tiled` only.
    DotProduct,
    /// Tiled L2Sq distance: C[i,j] = Σ_k (A[i,k] - B[k,j])². For `Grouping::Tiled` only.
    Distance,
}

impl Op {
    /// The neutral element (identity) of this monoid.
    ///
    /// `identity ⊕ x = x ⊕ identity = x` for all x.
    ///
    /// Used to pad non-power-of-2 scans in Blelloch prefix trees.
    /// MUST NOT be confused with `degenerate()` — padding with a degenerate
    /// value (e.g. NaN for Max) corrupts every element it touches.
    ///
    /// | Op      | identity      | Rationale                        |
    /// |---------|---------------|----------------------------------|
    /// | Add     | 0.0           | additive identity                |
    /// | Max     | -∞            | adding -∞ to a max leaves it unchanged |
    /// | Min     | +∞            | adding +∞ to a min leaves it unchanged |
    /// | ArgMin  | (+∞, MAX_IDX) | never beats a real minimum       |
    /// | ArgMax  | (-∞, MAX_IDX) | never beats a real maximum       |
    /// | DotProduct | 0.0        | additive identity of inner product |
    /// | Distance   | 0.0        | additive identity of L2Sq sum    |
    pub fn identity(&self) -> f64 {
        match self {
            Op::Add         => 0.0,
            Op::Max         => f64::NEG_INFINITY,
            Op::Min         => f64::INFINITY,
            Op::ArgMin      => f64::INFINITY,      // sentinel: never wins argmin
            Op::ArgMax      => f64::NEG_INFINITY,  // sentinel: never wins argmax
            Op::DotProduct  => 0.0,
            Op::Distance    => 0.0,
        }
    }

    /// The degenerate (invalid/empty) value for this Op.
    ///
    /// Returned when the input is empty, all-NaN, or otherwise uncomputable.
    /// Signals a computation failure — NOT a valid element of the monoid.
    ///
    /// MUST NOT be used for scan padding (use `identity()` instead).
    /// Consumers check for degenerate by testing `is_nan()` (scalar ops)
    /// or `value.is_nan()` (indexed ops).
    pub fn degenerate(&self) -> f64 {
        f64::NAN
    }
}

/// Output of an accumulate call — shape depends on grouping.
#[derive(Debug, Clone)]
pub enum AccResult {
    /// From `Grouping::All + Op::Add` — single scalar sum.
    Scalar(f64),
    /// From `Grouping::ByKey` — one value per group.
    PerGroup(Vec<f64>),
    /// From `Grouping::All + Op::ArgMin/ArgMax` — (value, index).
    IndexedPair(f64, usize),
    /// From `Grouping::Tiled` — M×N matrix (row-major).
    Matrix { data: Vec<f64>, m: usize, n: usize },
}

// ---------------------------------------------------------------------------
// AccumulateEngine
// ---------------------------------------------------------------------------

/// Unified GPU accumulation engine.
///
/// Owns the underlying primitives (ScatterJit, ReduceOp) and dispatches
/// `accumulate()` calls to the appropriate kernel based on the grouping pattern.
///
/// # Example — ByKey scatter
/// ```no_run
/// use tambear::AccumulateEngine;
/// use tambear::accumulate::{Grouping, Expr, Op, AccResult};
///
/// let mut engine = AccumulateEngine::new().unwrap();
/// let keys   = vec![0i32, 0, 1, 1, 2];
/// let values = vec![1.0,  2.0, 3.0, 4.0, 5.0];
///
/// let result = engine.accumulate(
///     &values,
///     Grouping::ByKey { keys: &keys, n_groups: 3 },
///     Expr::Value,
///     Op::Add,
/// ).unwrap();
///
/// if let AccResult::PerGroup(sums) = result {
///     // sums = [3.0, 7.0, 5.0]
/// }
/// ```
pub struct AccumulateEngine {
    compute: ComputeEngine,
    reduce: ReduceOp,
    tiled: TiledEngine,
}

impl AccumulateEngine {
    /// Initialise on GPU 0.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = tam_gpu::detect();
        Ok(Self {
            compute: ComputeEngine::new(gpu.clone()),
            reduce:  ReduceOp::with_backend(gpu.clone()),
            tiled: TiledEngine::new(gpu),
        })
    }

    /// The universal computation primitive.
    ///
    /// `accumulate(data, grouping, expr, op)` — routes to the appropriate kernel.
    pub fn accumulate(
        &mut self,
        values: &[f64],
        grouping: Grouping,
        expr: Expr,
        op: Op,
    ) -> Result<AccResult, Box<dyn std::error::Error>> {
        match (grouping, op) {
            // ── All + Add ────────────────────────────────────────────────
            // Reduce to scalar: scatter with all-zero keys, n_groups=1.
            (Grouping::All, Op::Add) => {
                let n = values.len();
                let keys = vec![0i32; n];
                let out = self.compute.scatter_phi(
                    expr.as_phi(), &keys, values, None, 1
                )?;
                Ok(AccResult::Scalar(out[0]))
            }

            // ── All + ArgMin / ArgMax ────────────────────────────────────
            (Grouping::All, Op::ArgMin) => {
                let (val, idx) = self.reduce.argmin(values)?;
                Ok(AccResult::IndexedPair(val, idx))
            }
            (Grouping::All, Op::ArgMax) => {
                let (val, idx) = self.reduce.argmax(values)?;
                Ok(AccResult::IndexedPair(val, idx))
            }

            // ── All + Max / Min ─────────────────────────────────────────
            (Grouping::All, Op::Max) => {
                let n = values.len();
                let keys = vec![0i32; n];
                let out = self.compute.scatter_extremum(true, &keys, values, 1)?;
                Ok(AccResult::Scalar(out[0]))
            }
            (Grouping::All, Op::Min) => {
                let n = values.len();
                let keys = vec![0i32; n];
                let out = self.compute.scatter_extremum(false, &keys, values, 1)?;
                Ok(AccResult::Scalar(out[0]))
            }

            // ── ByKey + Add ──────────────────────────────────────────────
            (Grouping::ByKey { keys, n_groups }, Op::Add) => {
                let out = self.compute.scatter_phi(
                    expr.as_phi(), keys, values, None, n_groups
                )?;
                Ok(AccResult::PerGroup(out))
            }

            // ── ByKey + Max / Min ────────────────────────────────────────
            (Grouping::ByKey { keys, n_groups }, Op::Max) => {
                let out = self.compute.scatter_extremum(true, keys, values, n_groups)?;
                Ok(AccResult::PerGroup(out))
            }
            (Grouping::ByKey { keys, n_groups }, Op::Min) => {
                let out = self.compute.scatter_extremum(false, keys, values, n_groups)?;
                Ok(AccResult::PerGroup(out))
            }

            // ── Masked ByKey + Add ───────────────────────────────────────
            (Grouping::Masked { keys, n_groups, mask }, Op::Add) => {
                let out = self.compute.scatter_phi_masked(
                    expr.as_phi(), keys, values, None, mask, n_groups
                )?;
                Ok(AccResult::PerGroup(out))
            }

            // ── ByKey + ArgMin/ArgMax: not yet supported ─────────────────
            (Grouping::ByKey { .. }, Op::ArgMin | Op::ArgMax) => {
                Err("ByKey + ArgMin/ArgMax: use scatter_phi with ReduceOp per group (not yet unified)".into())
            }

            // ── Prefix/Segmented/Tiled/Windowed: todo ────────────────────
            (Grouping::Prefix, _) => {
                Err("Grouping::Prefix: not yet wired — use winrapids-scan AssociativeOp directly".into())
            }
            (Grouping::Segmented, _) => {
                Err("Grouping::Segmented: not yet wired — use winrapids-scan segmented mode".into())
            }
            // ── Tiled + DotProduct / Distance ──────────────────────────
            (Grouping::Tiled { b, m, n, k }, Op::DotProduct) => {
                let result = self.tiled.run(&DotProductOp, values, b, m, n, k)?;
                Ok(AccResult::Matrix { data: result, m, n })
            }
            (Grouping::Tiled { b, m, n, k }, Op::Distance) => {
                let result = self.tiled.run(&DistanceOp, values, b, m, n, k)?;
                Ok(AccResult::Matrix { data: result, m, n })
            }
            (Grouping::Tiled { .. }, _) => {
                Err("Grouping::Tiled: only Op::DotProduct and Op::Distance supported".into())
            }
            (Grouping::Windowed { .. }, _) => {
                Err("Grouping::Windowed: not yet wired — use prefix subtraction trick via winrapids-scan".into())
            }

            _ => {
                Err(format!("accumulate: unsupported (grouping, op) combination — open a gap").into())
            }
        }
    }

    /// Fused multi-expression accumulate: same (data, grouping, op), multiple exprs.
    ///
    /// This is the compiler's fusion primitive: when N computations share the same
    /// data and grouping, compute them in one kernel pass instead of N passes.
    ///
    /// Only `Grouping::ByKey + Op::Add` supported for now.
    ///
    /// # Example — GroupBy mean in one pass
    /// ```no_run
    /// use tambear::AccumulateEngine;
    /// use tambear::accumulate::Grouping;
    /// use tambear::{PHI_SUM, PHI_COUNT};
    ///
    /// let mut engine  = AccumulateEngine::new().unwrap();
    /// let keys    = vec![0i32, 0, 1, 1];
    /// let values  = vec![1.0, 3.0, 2.0, 4.0];
    ///
    /// let results = engine.accumulate_multi(
    ///     &values,
    ///     Grouping::ByKey { keys: &keys, n_groups: 2 },
    ///     &[PHI_SUM, PHI_COUNT],
    /// ).unwrap();
    /// // results[0] = sums  = [4.0, 6.0]
    /// // results[1] = counts = [2.0, 2.0]
    /// // means = zip(sums, counts).map(|(s,c)| s/c) = [2.0, 3.0]
    /// ```
    pub fn accumulate_multi(
        &mut self,
        values: &[f64],
        grouping: Grouping,
        phi_exprs: &[&str],
    ) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        match grouping {
            Grouping::ByKey { keys, n_groups } => {
                self.compute.scatter_multi_phi(phi_exprs, keys, values, None, n_groups)
            }
            Grouping::Masked { keys, n_groups, mask } => {
                // Decompose into N separate masked scatter calls.
                // ComputeEngine doesn't have scatter_multi_phi_masked yet —
                // this trades the one-pass optimization for multi-backend.
                let mut results = Vec::with_capacity(phi_exprs.len());
                for &phi in phi_exprs {
                    let out = self.compute.scatter_phi_masked(phi, keys, values, None, mask, n_groups)?;
                    results.push(out);
                }
                Ok(results)
            }
            _ => Err("accumulate_multi: only ByKey and Masked groupings supported (others: todo)".into()),
        }
    }

    /// Fused dual-target accumulate: same `(data, expr)`, two groupings, one pass.
    ///
    /// When two accumulations share the same data and expression but differ in
    /// grouping, this computes the expression ONCE and scatters to both targets —
    /// halving memory bandwidth compared to two separate calls.
    ///
    /// Both groupings must use `Op::Add`. Only `Grouping::All` and `Grouping::ByKey`
    /// are supported for each target.
    ///
    /// # Example — total and per-ticker sum in one pass
    /// ```no_run
    /// use tambear::AccumulateEngine;
    /// use tambear::accumulate::{Grouping, Expr};
    ///
    /// let mut engine = AccumulateEngine::new().unwrap();
    /// let ticker_keys = vec![0i32, 0, 1, 1, 2];
    /// let prices      = vec![10.0, 11.0, 20.0, 21.0, 5.0];
    ///
    /// let (total, by_ticker) = engine.accumulate_dual(
    ///     &prices,
    ///     Grouping::All,                                          // → total sum
    ///     Grouping::ByKey { keys: &ticker_keys, n_groups: 3 },   // → per-ticker sum
    ///     Expr::Value,
    /// ).unwrap();
    /// // total = 67.0  (sum of all prices, ONE pass)
    /// // by_ticker = [21.0, 41.0, 5.0]
    /// ```
    pub fn accumulate_dual(
        &mut self,
        values: &[f64],
        grouping0: Grouping,
        grouping1: Grouping,
        expr: Expr,
    ) -> Result<(AccResult, AccResult), Box<dyn std::error::Error>> {
        let n = values.len();
        let phi = expr.as_phi();

        // Extract (keys, n_groups) from each grouping — only All and ByKey supported.
        let (keys0_owned, n_groups0) = grouping_to_keys(grouping0, n)?;
        let (keys1_owned, n_groups1) = grouping_to_keys(grouping1, n)?;

        // Decompose into two separate scatter_phi calls.
        // ScatterJit had a fused dual-target kernel (one memory pass),
        // but ComputeEngine dispatches through TamGpu for multi-backend.
        // Two passes is 2x bandwidth but enables CPU/Vulkan/Metal.
        let out0 = self.compute.scatter_phi(phi, &keys0_owned, values, None, n_groups0)?;
        let out1 = self.compute.scatter_phi(phi, &keys1_owned, values, None, n_groups1)?;

        let result0 = if n_groups0 == 1 {
            AccResult::Scalar(out0[0])
        } else {
            AccResult::PerGroup(out0)
        };
        let result1 = if n_groups1 == 1 {
            AccResult::Scalar(out1[0])
        } else {
            AccResult::PerGroup(out1)
        };

        Ok((result0, result1))
    }

    /// Compute the softmax of `values`: `output[i] = exp(v[i]) / sum(exp(v))`.
    ///
    /// Numerically stable: subtracts `max(values)` before exp (log-sum-exp trick).
    ///
    /// This is the reduce-broadcast-divide pattern from the accumulate unification:
    /// 1. `reduce(values, All, ArgMax)` → max_val
    /// 2. `reduce(exp(v - max_val), All, Add)` → sum_exp (scatter_phi with keys=0)
    /// 3. `map_phi("exp(v - max) / sum_exp", values)` → softmax probabilities
    ///
    /// Three NVRTC compiles on first call (~120ms total), then O(n) dispatch.
    pub fn softmax(&mut self, values: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = values.len();
        if n == 0 { return Ok(vec![]); }
        if n == 1 { return Ok(vec![1.0]); }

        // Step 1: max for numerical stability.
        let (max_val, _) = self.reduce.argmax(values)?;

        // Step 2: sum_exp = sum(exp(v - max_val)) — all elements into group 0.
        let keys = vec![0i32; n];
        let exp_phi = format!("exp(v - {:.17e})", max_val);
        let sum_vec = self.compute.scatter_phi(&exp_phi, &keys, values, None, 1)?;
        let sum_exp = sum_vec[0];

        // Step 3: map each element by exp(v - max_val) / sum_exp.
        let softmax_phi = format!("exp(v - {:.17e}) / {:.17e}", max_val, sum_exp);
        self.compute.map_phi(&softmax_phi, values)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract (owned_keys_vec, n_groups) from a Grouping for the dual-target case.
/// Only Grouping::All and Grouping::ByKey are supported.
fn grouping_to_keys(grouping: Grouping, n: usize) -> Result<(Vec<i32>, usize), Box<dyn std::error::Error>> {
    match grouping {
        Grouping::All => Ok((vec![0i32; n], 1)),
        Grouping::ByKey { keys, n_groups } => Ok((keys.to_vec(), n_groups)),
        other => Err(format!(
            "accumulate_dual: unsupported grouping {:?} — only All and ByKey for now",
            other
        ).into()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// phi expressions for testing (same as scatter_jit constants)
    const PHI_SUM: &str = "v";
    const PHI_COUNT: &str = "1.0";

    fn engine() -> AccumulateEngine {
        AccumulateEngine::new().expect("AccumulateEngine init failed")
    }

    #[test]
    fn all_add_reduces_to_sum() {
        let mut e = engine();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        match e.accumulate(&values, Grouping::All, Expr::Value, Op::Add).unwrap() {
            AccResult::Scalar(s) => assert!((s - 15.0).abs() < 1e-9, "sum={s}"),
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn all_add_value_sq_gives_sum_of_squares() {
        let mut e = engine();
        let values = vec![1.0, 2.0, 3.0];
        match e.accumulate(&values, Grouping::All, Expr::ValueSq, Op::Add).unwrap() {
            AccResult::Scalar(s) => assert!((s - 14.0).abs() < 1e-9, "sum_sq={s}"),
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn all_argmin_finds_minimum() {
        let mut e = engine();
        let values = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        match e.accumulate(&values, Grouping::All, Expr::Value, Op::ArgMin).unwrap() {
            AccResult::IndexedPair(v, i) => {
                assert!((v - 1.0).abs() < 1e-9, "min={v}");
                assert_eq!(i, 1, "idx={i}");
            }
            other => panic!("expected IndexedPair, got {other:?}"),
        }
    }

    #[test]
    fn all_argmax_finds_maximum() {
        let mut e = engine();
        let values = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        match e.accumulate(&values, Grouping::All, Expr::Value, Op::ArgMax).unwrap() {
            AccResult::IndexedPair(v, i) => {
                assert!((v - 5.0).abs() < 1e-9, "max={v}");
                assert_eq!(i, 1, "idx={i}");
            }
            other => panic!("expected IndexedPair, got {other:?}"),
        }
    }

    #[test]
    fn bykey_add_matches_scatter_sum() {
        let mut e = engine();
        let keys   = vec![0i32, 0, 1, 1, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        match e.accumulate(&values, Grouping::ByKey { keys: &keys, n_groups: 3 },
                           Expr::Value, Op::Add).unwrap() {
            AccResult::PerGroup(sums) => {
                assert!((sums[0] - 3.0).abs() < 1e-9);
                assert!((sums[1] - 7.0).abs() < 1e-9);
                assert!((sums[2] - 5.0).abs() < 1e-9);
            }
            other => panic!("expected PerGroup, got {other:?}"),
        }
    }

    #[test]
    fn accumulate_multi_fused_sum_count() {
        let mut e = engine();
        let keys   = vec![0i32, 0, 1, 1];
        let values = vec![1.0, 3.0, 2.0, 4.0];
        let results = e.accumulate_multi(
            &values,
            Grouping::ByKey { keys: &keys, n_groups: 2 },
            &[PHI_SUM, PHI_COUNT],
        ).unwrap();
        // sums  = [4.0, 6.0]
        // counts = [2.0, 2.0]
        assert!((results[0][0] - 4.0).abs() < 1e-9, "sum[0]={}", results[0][0]);
        assert!((results[0][1] - 6.0).abs() < 1e-9, "sum[1]={}", results[0][1]);
        assert!((results[1][0] - 2.0).abs() < 1e-9, "count[0]={}", results[1][0]);
        assert!((results[1][1] - 2.0).abs() < 1e-9, "count[1]={}", results[1][1]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut e = engine();
        let values = vec![1.0f64, 2.0, 3.0];
        let probs = e.softmax(&values).unwrap();
        assert_eq!(probs.len(), 3);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "sum={total}");
        // Probabilities should be in ascending order (values are ascending)
        assert!(probs[0] < probs[1] && probs[1] < probs[2], "probs={probs:?}");
    }

    #[test]
    fn softmax_known_values() {
        // softmax([1, 2, 3]) — stable formula: subtract max=3
        // exp(-2) + exp(-1) + exp(0) = 0.13534 + 0.36788 + 1.0 = 1.50321
        // p = [0.09003, 0.24473, 0.66524]
        let mut e = engine();
        let values = vec![1.0f64, 2.0, 3.0];
        let probs = e.softmax(&values).unwrap();
        assert!((probs[0] - 0.090030573).abs() < 1e-6, "p[0]={}", probs[0]);
        assert!((probs[1] - 0.244728471).abs() < 1e-6, "p[1]={}", probs[1]);
        assert!((probs[2] - 0.665240956).abs() < 1e-6, "p[2]={}", probs[2]);
    }

    #[test]
    fn softmax_single_element_is_one() {
        let mut e = engine();
        let probs = e.softmax(&[42.0]).unwrap();
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dual_target_all_and_bykey() {
        // The canonical multi-target fusion example:
        // total sum AND per-group sum — one kernel pass.
        let mut e = engine();
        let keys   = vec![0i32, 0, 1, 1, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (total, by_group) = e.accumulate_dual(
            &values,
            Grouping::All,
            Grouping::ByKey { keys: &keys, n_groups: 3 },
            Expr::Value,
        ).unwrap();

        // total = sum of all = 15
        match total {
            AccResult::Scalar(s) => assert!((s - 15.0).abs() < 1e-9, "total={s}"),
            other => panic!("expected Scalar, got {other:?}"),
        }
        // by_group = [3, 7, 5]
        match by_group {
            AccResult::PerGroup(v) => {
                assert!((v[0] - 3.0).abs() < 1e-9);
                assert!((v[1] - 7.0).abs() < 1e-9);
                assert!((v[2] - 5.0).abs() < 1e-9);
            }
            other => panic!("expected PerGroup, got {other:?}"),
        }
    }

    #[test]
    fn dual_target_matches_two_separate_calls() {
        // Verify dual == separate, to confirm fusion correctness.
        let mut e = engine();
        let keys   = vec![0i32, 0, 1, 1, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Fused
        let (dual_total, dual_by_group) = e.accumulate_dual(
            &values,
            Grouping::All,
            Grouping::ByKey { keys: &keys, n_groups: 3 },
            Expr::ValueSq,
        ).unwrap();

        // Separate
        let sep_total = e.accumulate(&values, Grouping::All, Expr::ValueSq, Op::Add).unwrap();
        let sep_by_group = e.accumulate(
            &values,
            Grouping::ByKey { keys: &keys, n_groups: 3 },
            Expr::ValueSq, Op::Add,
        ).unwrap();

        // Results must match
        match (dual_total, sep_total) {
            (AccResult::Scalar(a), AccResult::Scalar(b)) =>
                assert!((a - b).abs() < 1e-9, "total mismatch: {a} vs {b}"),
            _ => panic!("type mismatch"),
        }
        match (dual_by_group, sep_by_group) {
            (AccResult::PerGroup(a), AccResult::PerGroup(b)) => {
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    assert!((x - y).abs() < 1e-9, "group {i}: {x} vs {y}");
                }
            }
            _ => panic!("type mismatch"),
        }
    }

    #[test]
    fn map_phi_element_wise() {
        // Verify map_phi writes to output[i], not output[key]
        let mut e = engine();
        let values = vec![1.0f64, 4.0, 9.0];
        let roots = e.compute.map_phi("sqrt(v)", &values).unwrap();
        assert!((roots[0] - 1.0).abs() < 1e-9);
        assert!((roots[1] - 2.0).abs() < 1e-9);
        assert!((roots[2] - 3.0).abs() < 1e-9);
    }

    // ── Tiled accumulation tests ────────────────────────────────────

    #[test]
    fn tiled_dot_product_2x2() {
        // A (2×3) × B (3×2) → C (2×2)
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        // C = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //   = [[58, 64], [139, 154]]
        let mut e = engine();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let result = e.accumulate(
            &a,
            Grouping::Tiled { b: &b, m: 2, n: 2, k: 3 },
            Expr::Value,  // expr is ignored for tiled — op determines kernel
            Op::DotProduct,
        ).unwrap();

        match result {
            AccResult::Matrix { data, m, n } => {
                assert_eq!(m, 2);
                assert_eq!(n, 2);
                assert!((data[0] - 58.0).abs() < 1e-9, "C[0,0]={}", data[0]);
                assert!((data[1] - 64.0).abs() < 1e-9, "C[0,1]={}", data[1]);
                assert!((data[2] - 139.0).abs() < 1e-9, "C[1,0]={}", data[2]);
                assert!((data[3] - 154.0).abs() < 1e-9, "C[1,1]={}", data[3]);
            }
            other => panic!("expected Matrix, got {other:?}"),
        }
    }

    #[test]
    fn tiled_dot_product_vector() {
        // Matrix × vector: A (3×2) × B (2×1) → C (3×1)
        // A = [[1,2],[3,4],[5,6]], B = [[1],[1]]
        // C = [[3],[7],[11]]
        let mut e = engine();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 1.0];

        let result = e.accumulate(
            &a,
            Grouping::Tiled { b: &b, m: 3, n: 1, k: 2 },
            Expr::Value,
            Op::DotProduct,
        ).unwrap();

        match result {
            AccResult::Matrix { data, m, n } => {
                assert_eq!(m, 3);
                assert_eq!(n, 1);
                assert!((data[0] - 3.0).abs() < 1e-9);
                assert!((data[1] - 7.0).abs() < 1e-9);
                assert!((data[2] - 11.0).abs() < 1e-9);
            }
            other => panic!("expected Matrix, got {other:?}"),
        }
    }

    #[test]
    fn tiled_distance_self() {
        // Distance matrix: 3 points in 2D
        // Points: (0,0), (1,0), (0,2)
        // D[i,j] = L2Sq distance
        // Need B = transpose of A for distance computation
        let mut e = engine();
        let a = vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0]; // 3×2
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 2.0];  // 2×3 (transposed)

        let result = e.accumulate(
            &a,
            Grouping::Tiled { b: &b, m: 3, n: 3, k: 2 },
            Expr::Value,
            Op::Distance,
        ).unwrap();

        match result {
            AccResult::Matrix { data, m, n } => {
                assert_eq!(m, 3);
                assert_eq!(n, 3);
                // D[0,0] = 0 (self)
                assert!((data[0]).abs() < 1e-9, "D[0,0]={}", data[0]);
                // D[0,1] = (0-1)^2 + (0-0)^2 = 1
                assert!((data[1] - 1.0).abs() < 1e-9, "D[0,1]={}", data[1]);
                // D[0,2] = (0-0)^2 + (0-2)^2 = 4
                assert!((data[2] - 4.0).abs() < 1e-9, "D[0,2]={}", data[2]);
                // D[1,2] = (1-0)^2 + (0-2)^2 = 5
                assert!((data[5] - 5.0).abs() < 1e-9, "D[1,2]={}", data[5]);
            }
            other => panic!("expected Matrix, got {other:?}"),
        }
    }

    #[test]
    fn tiled_invalid_op_errors() {
        let mut e = engine();
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 1.0];
        let result = e.accumulate(
            &a,
            Grouping::Tiled { b: &b, m: 1, n: 1, k: 2 },
            Expr::Value,
            Op::Add,  // invalid for Tiled
        );
        assert!(result.is_err());
    }

    /// Proof: logistic regression gradient descent expressed as pure accumulate calls.
    ///
    /// The training loop:
    ///   Forward:  z = accumulate(X_aug, Tiled{β}, n,1,p, DotProduct)
    ///   Map:      p = sigmoid(z), r = p - y  (element-wise CPU, fuses in lazy)
    ///   Backward: ∇ = accumulate(X_aug_T, Tiled{r}, p,1,n, DotProduct)
    ///   Update:   β -= lr * ∇ / n  (element-wise CPU)
    ///
    /// Two accumulate calls per iteration. Same op (DotProduct), different grouping
    /// (different matrix shapes). The gradient duality IS the transpose of the
    /// tiled grouping.
    #[test]
    fn logistic_regression_via_accumulate() {
        let mut e = engine();

        // Simple 1D classification: x < 0 → class 0, x > 0 → class 1
        let n = 20;
        let d = 1;
        let p = d + 1; // augmented

        // Build augmented X and X_T
        let mut x_aug = vec![0.0f64; n * p];
        let mut x_aug_t = vec![0.0f64; p * n];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let xi = (i as f64 / n as f64) * 6.0 - 3.0; // -3 to 3
            x_aug[i * p] = xi;
            x_aug[i * p + d] = 1.0;
            x_aug_t[0 * n + i] = xi;
            x_aug_t[d * n + i] = 1.0;
            y[i] = if xi > 0.0 { 1.0 } else { 0.0 };
        }

        let mut beta = vec![0.0f64; p];
        let lr = 1.0;
        let n_f = n as f64;

        fn sigmoid(z: f64) -> f64 { 1.0 / (1.0 + (-z).exp()) }

        for _iter in 0..100 {
            // ── FORWARD: z = X_aug * β  via accumulate(Tiled, DotProduct) ──
            let z = match e.accumulate(
                &x_aug,
                Grouping::Tiled { b: &beta, m: n, n: 1, k: p },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── MAP: sigmoid + residual (CPU, fuses in lazy pipeline) ──
            let mut residual = vec![0.0f64; n];
            for i in 0..n {
                residual[i] = sigmoid(z[i]) - y[i];
            }

            // ── BACKWARD: ∇ = X_aug_T * residual  via accumulate(Tiled, DotProduct) ──
            let grad = match e.accumulate(
                &x_aug_t,
                Grouping::Tiled { b: &residual, m: p, n: 1, k: n },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── UPDATE: β -= lr * ∇ / n  (CPU) ──
            for j in 0..p {
                beta[j] -= lr * grad[j] / n_f;
            }
        }

        // Verify convergence: positive coefficient, correct predictions
        assert!(beta[0] > 0.0, "weight should be positive, got {}", beta[0]);

        // Check predictions
        let mut correct = 0;
        for i in 0..n {
            let z = beta[0] * x_aug[i * p] + beta[1];
            let pred = if sigmoid(z) >= 0.5 { 1.0 } else { 0.0 };
            if (pred - y[i]).abs() < 0.5 { correct += 1; }
        }
        let accuracy = correct as f64 / n_f;
        assert!(accuracy > 0.85, "accuracy={:.1}%, expected >85%", accuracy * 100.0);
    }

    /// Proof: Newton's method (second-order) via pure accumulate calls.
    ///
    /// The Hessian of logistic regression loss:
    ///   H = X' * diag(w) * X   where w[i] = p[i] * (1-p[i])
    ///
    /// Reformulated as: X_w[i,j] = X[i,j] * sqrt(w[i])
    ///   H = X_w' * X_w = accumulate(X_w_T, Tiled{X_w}, DotProduct)
    ///
    /// Newton's update: β -= H⁻¹ * ∇  (via Cholesky solve)
    ///
    /// THREE accumulate calls per iteration:
    ///   1. Forward:  z = accumulate(X, Tiled{β}, DotProduct)
    ///   2. Gradient: ∇ = accumulate(X', Tiled{residual}, DotProduct)
    ///   3. Hessian:  H = accumulate(X_w', Tiled{X_w}, DotProduct)
    ///
    /// All three are the SAME OPERATION (DotProduct) with different inputs.
    /// The Hessian is not "double-transposed" — it's a new DotProduct on
    /// weighted data, where the weights are element-wise (fuse in lazy pipeline).
    #[test]
    fn newton_method_via_accumulate() {
        let mut e = engine();

        let n = 30;
        let d = 1;
        let p = d + 1;

        let mut x_aug = vec![0.0f64; n * p];
        let mut x_aug_t = vec![0.0f64; p * n];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let xi = (i as f64 / n as f64) * 8.0 - 4.0; // -4 to 4
            x_aug[i * p] = xi;
            x_aug[i * p + d] = 1.0;
            x_aug_t[0 * n + i] = xi;
            x_aug_t[d * n + i] = 1.0;
            y[i] = if xi > 0.0 { 1.0 } else { 0.0 };
        }

        let mut beta = vec![0.0f64; p];
        let n_f = n as f64;

        fn sigmoid(z: f64) -> f64 { 1.0 / (1.0 + (-z).exp()) }

        // Newton's method should converge in very few iterations (quadratic convergence)
        for _iter in 0..10 {
            // ── 1. FORWARD: z = X_aug * β ──
            let z = match e.accumulate(
                &x_aug,
                Grouping::Tiled { b: &beta, m: n, n: 1, k: p },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── MAP: probabilities, residual, weights (element-wise, fuses) ──
            let mut residual = vec![0.0f64; n];
            let mut x_w = vec![0.0f64; n * p];      // weighted X
            let mut x_w_t = vec![0.0f64; p * n];    // transposed weighted X
            for i in 0..n {
                let pi = sigmoid(z[i]);
                residual[i] = pi - y[i];
                let wi = (pi * (1.0 - pi)).max(1e-12).sqrt();
                for j in 0..p {
                    x_w[i * p + j] = x_aug[i * p + j] * wi;
                    x_w_t[j * n + i] = x_aug[i * p + j] * wi;
                }
            }

            // ── 2. GRADIENT: ∇ = X_aug_T * residual ──
            let grad = match e.accumulate(
                &x_aug_t,
                Grouping::Tiled { b: &residual, m: p, n: 1, k: n },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── 3. HESSIAN: H = X_w_T * X_w  (p×n · n×p → p×p) ──
            let hessian = match e.accumulate(
                &x_w_t,
                Grouping::Tiled { b: &x_w, m: p, n: p, k: n },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── SOLVE: Δβ = H⁻¹ * ∇  (Cholesky, CPU) ──
            // For p=2, solve 2×2 system directly
            let a00 = hessian[0];
            let a01 = hessian[1];
            let a10 = hessian[2];
            let a11 = hessian[3];
            let det = a00 * a11 - a01 * a10;
            if det.abs() < 1e-15 { break; } // singular
            let delta0 = ( a11 * grad[0] - a01 * grad[1]) / det;
            let delta1 = (-a10 * grad[0] + a00 * grad[1]) / det;

            // ── UPDATE: β -= Δβ ──
            beta[0] -= delta0;
            beta[1] -= delta1;
        }

        // Newton's method with quadratic convergence should nail this
        assert!(beta[0] > 0.0, "weight should be positive, got {}", beta[0]);

        let mut correct = 0;
        for i in 0..n {
            let z = beta[0] * x_aug[i * p] + beta[1];
            let pred = if sigmoid(z) >= 0.5 { 1.0 } else { 0.0 };
            if (pred - y[i]).abs() < 0.5 { correct += 1; }
        }
        let accuracy = correct as f64 / n_f;
        assert!(accuracy > 0.9, "Newton accuracy={:.1}%, expected >90%", accuracy * 100.0);
    }

    /// Proof: 2-layer neural network forward+backward via pure accumulate calls.
    ///
    /// Architecture: X(n×d) → W1(d×h) → ReLU → W2(h×1) → sigmoid → loss
    ///
    /// Forward (2 accumulate calls):
    ///   z1 = accumulate(X, Tiled{W1}, DotProduct)     ← layer 1
    ///   a1 = ReLU(z1)                                   ← element-wise (fuses)
    ///   z2 = accumulate(a1, Tiled{W2}, DotProduct)    ← layer 2
    ///   a2 = sigmoid(z2)                                ← element-wise (fuses)
    ///
    /// Backward (2 accumulate calls for gradients, 1 for backprop through layer):
    ///   δ2 = a2 - y                                     ← element-wise
    ///   ∇W2 = accumulate(a1_T, Tiled{δ2}, DotProduct)  ← weight gradient layer 2
    ///   δ1_raw = accumulate(δ2, Tiled{W2_T}, DotProduct) ← backprop through matmul
    ///   δ1 = δ1_raw ⊙ relu_mask                         ← element-wise (fuses)
    ///   ∇W1 = accumulate(X_T, Tiled{δ1}, DotProduct)   ← weight gradient layer 1
    ///
    /// 5 accumulate calls total. ALL DotProduct. Activation derivatives are
    /// element-wise masks that fuse. The chain rule works through the framework.
    #[test]
    fn two_layer_network_via_accumulate() {
        let mut e = engine();

        // XOR-ish problem that requires hidden layer
        // Points at (±1, ±1), label = (x1 > 0) XOR (x2 > 0)
        let n = 8;
        let d = 2;
        let h = 4; // hidden units

        let x = vec![
            -1.0, -1.0,  // class 0
            -1.0,  1.0,  // class 1
             1.0, -1.0,  // class 1
             1.0,  1.0,  // class 0
            -0.5, -0.5,  // class 0
            -0.5,  0.5,  // class 1
             0.5, -0.5,  // class 1
             0.5,  0.5,  // class 0
        ];
        let y = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];

        // Initialize weights (small random-ish, deterministic)
        let mut w1 = vec![0.0f64; d * h]; // d×h
        let mut w2 = vec![0.0f64; h];     // h×1
        for i in 0..d*h {
            w1[i] = ((i as f64 * 7.3 + 1.1).sin()) * 0.5;
        }
        for i in 0..h {
            w2[i] = ((i as f64 * 3.7 + 2.2).sin()) * 0.5;
        }

        fn sigmoid(z: f64) -> f64 { 1.0 / (1.0 + (-z).exp()) }
        fn relu(z: f64) -> f64 { if z > 0.0 { z } else { 0.0 } }

        let lr = 0.5;
        let n_f = n as f64;

        let mut prev_loss = f64::INFINITY;
        for _iter in 0..200 {
            // ── FORWARD LAYER 1: z1 = X * W1  (n×d · d×h → n×h) ──
            let z1 = match e.accumulate(
                &x,
                Grouping::Tiled { b: &w1, m: n, n: h, k: d },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── ACTIVATION: a1 = ReLU(z1)  (element-wise, fuses) ──
            let a1: Vec<f64> = z1.iter().map(|&z| relu(z)).collect();
            let relu_mask: Vec<f64> = z1.iter().map(|&z| if z > 0.0 { 1.0 } else { 0.0 }).collect();

            // ── FORWARD LAYER 2: z2 = a1 * W2  (n×h · h×1 → n×1) ──
            let z2 = match e.accumulate(
                &a1,
                Grouping::Tiled { b: &w2, m: n, n: 1, k: h },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── OUTPUT + LOSS ──
            let mut loss = 0.0f64;
            let mut delta2 = vec![0.0f64; n]; // n×1
            for i in 0..n {
                let a2 = sigmoid(z2[i]);
                delta2[i] = a2 - y[i];
                let a2c = a2.clamp(1e-15, 1.0 - 1e-15);
                loss -= y[i] * a2c.ln() + (1.0 - y[i]) * (1.0 - a2c).ln();
            }
            loss /= n_f;

            // Check loss is decreasing (training signal)
            if _iter > 0 && _iter % 50 == 0 {
                assert!(loss < prev_loss + 0.1,
                    "loss not decreasing: iter={}, loss={:.4}, prev={:.4}", _iter, loss, prev_loss);
            }
            prev_loss = loss;

            // ── BACKWARD: ∇W2 = a1_T * δ2  (h×n · n×1 → h×1) ──
            let mut a1_t = vec![0.0f64; h * n];
            for i in 0..n { for j in 0..h { a1_t[j * n + i] = a1[i * h + j]; } }
            let grad_w2 = match e.accumulate(
                &a1_t,
                Grouping::Tiled { b: &delta2, m: h, n: 1, k: n },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── BACKWARD: δ1_raw = δ2 * W2_T  (n×1 · 1×h → n×h) ──
            // W2 is h×1, W2_T is 1×h
            let w2_t = w2.clone(); // h×1 transposed = 1×h (same data, different shape)
            let delta1_raw = match e.accumulate(
                &delta2,
                Grouping::Tiled { b: &w2_t, m: n, n: h, k: 1 },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── MASK: δ1 = δ1_raw ⊙ relu_mask  (element-wise, fuses) ──
            let delta1: Vec<f64> = delta1_raw.iter().zip(relu_mask.iter())
                .map(|(&d, &m)| d * m).collect();

            // ── BACKWARD: ∇W1 = X_T * δ1  (d×n · n×h → d×h) ──
            let mut x_t = vec![0.0f64; d * n];
            for i in 0..n { for j in 0..d { x_t[j * n + i] = x[i * d + j]; } }
            let grad_w1 = match e.accumulate(
                &x_t,
                Grouping::Tiled { b: &delta1, m: d, n: h, k: n },
                Expr::Value,
                Op::DotProduct,
            ).unwrap() {
                AccResult::Matrix { data, .. } => data,
                other => panic!("expected Matrix, got {other:?}"),
            };

            // ── UPDATE ──
            for j in 0..h {
                w2[j] -= lr * grad_w2[j] / n_f;
            }
            for j in 0..d*h {
                w1[j] -= lr * grad_w1[j] / n_f;
            }
        }

        // Verify loss decreased meaningfully
        assert!(prev_loss < 0.7, "final loss={:.4}, expected training signal", prev_loss);
    }

    // ── Max / Min scatter ────────────────────────────────────────────────────

    #[test]
    fn all_max_reduces_to_maximum() {
        let mut e = engine();
        let values = vec![3.0, 1.0, 7.0, 2.0, 5.0];
        match e.accumulate(&values, Grouping::All, Expr::Value, Op::Max).unwrap() {
            AccResult::Scalar(v) => assert!((v - 7.0).abs() < 1e-9, "max={v}"),
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn all_min_reduces_to_minimum() {
        let mut e = engine();
        let values = vec![3.0, 1.0, 7.0, 2.0, 5.0];
        match e.accumulate(&values, Grouping::All, Expr::Value, Op::Min).unwrap() {
            AccResult::Scalar(v) => assert!((v - 1.0).abs() < 1e-9, "min={v}"),
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn bykey_max_per_group() {
        let mut e = engine();
        // Bin 0: {3.0, 1.0, 4.0} → max 4.0
        // Bin 1: {1.5, 9.0, 2.6} → max 9.0
        // Bin 2: {5.0}            → max 5.0
        let keys   = vec![0i32, 0, 0, 1, 1, 1, 2];
        let values = vec![3.0,  1.0, 4.0, 1.5, 9.0, 2.6, 5.0];
        match e.accumulate(&values, Grouping::ByKey { keys: &keys, n_groups: 3 },
                           Expr::Value, Op::Max).unwrap() {
            AccResult::PerGroup(maxes) => {
                assert!((maxes[0] - 4.0).abs() < 1e-9, "max[0]={}", maxes[0]);
                assert!((maxes[1] - 9.0).abs() < 1e-9, "max[1]={}", maxes[1]);
                assert!((maxes[2] - 5.0).abs() < 1e-9, "max[2]={}", maxes[2]);
            }
            other => panic!("expected PerGroup, got {other:?}"),
        }
    }

    #[test]
    fn bykey_min_per_group() {
        let mut e = engine();
        // Bin 0: {3.0, 1.0, 4.0} → min 1.0
        // Bin 1: {1.5, 9.0, 2.6} → min 1.5
        // Bin 2: {5.0}            → min 5.0
        let keys   = vec![0i32, 0, 0, 1, 1, 1, 2];
        let values = vec![3.0,  1.0, 4.0, 1.5, 9.0, 2.6, 5.0];
        match e.accumulate(&values, Grouping::ByKey { keys: &keys, n_groups: 3 },
                           Expr::Value, Op::Min).unwrap() {
            AccResult::PerGroup(mins) => {
                assert!((mins[0] - 1.0).abs() < 1e-9, "min[0]={}", mins[0]);
                assert!((mins[1] - 1.5).abs() < 1e-9, "min[1]={}", mins[1]);
                assert!((mins[2] - 5.0).abs() < 1e-9, "min[2]={}", mins[2]);
            }
            other => panic!("expected PerGroup, got {other:?}"),
        }
    }

    #[test]
    fn bykey_max_min_single_element_groups() {
        // Each element is its own group — max[g] = min[g] = value[g]
        let mut e = engine();
        let keys   = vec![0i32, 1, 2, 3];
        let values = vec![7.0, 3.0, 9.0, 1.0];
        let max_result = e.accumulate(&values, Grouping::ByKey { keys: &keys, n_groups: 4 },
                                      Expr::Value, Op::Max).unwrap();
        let min_result = e.accumulate(&values, Grouping::ByKey { keys: &keys, n_groups: 4 },
                                      Expr::Value, Op::Min).unwrap();
        if let (AccResult::PerGroup(maxes), AccResult::PerGroup(mins)) = (max_result, min_result) {
            for (i, (&mx, &mn)) in maxes.iter().zip(mins.iter()).enumerate() {
                assert!((mx - values[i]).abs() < 1e-9, "max[{i}]={mx}");
                assert!((mn - values[i]).abs() < 1e-9, "min[{i}]={mn}");
            }
        } else {
            panic!("expected PerGroup");
        }
    }

    /// OHLCV decomposition proof: H and L computed via Max/Min scatter.
    /// O and C use gather (direct index at bin start/end) — no new op needed.
    #[test]
    fn ohlcv_high_low_via_max_min_scatter() {
        let mut e = engine();
        // 3 bins of tick prices:
        // Bin 0 (open=100, high=105, low=98,  close=102): ticks 100,105,98,102
        // Bin 1 (open=102, high=108, low=101, close=106): ticks 102,108,101,106
        // Bin 2 (open=106, high=107, low=103, close=104): ticks 106,107,103,104
        let keys   = vec![0i32,0,0,0, 1,1,1,1, 2,2,2,2];
        let prices = vec![100.0,105.0,98.0,102.0, 102.0,108.0,101.0,106.0, 106.0,107.0,103.0,104.0];

        let high = match e.accumulate(&prices, Grouping::ByKey { keys: &keys, n_groups: 3 },
                                       Expr::Value, Op::Max).unwrap() {
            AccResult::PerGroup(v) => v,
            other => panic!("{other:?}"),
        };
        let low = match e.accumulate(&prices, Grouping::ByKey { keys: &keys, n_groups: 3 },
                                      Expr::Value, Op::Min).unwrap() {
            AccResult::PerGroup(v) => v,
            other => panic!("{other:?}"),
        };

        assert!((high[0] - 105.0).abs() < 1e-9, "H[0]={}", high[0]);
        assert!((high[1] - 108.0).abs() < 1e-9, "H[1]={}", high[1]);
        assert!((high[2] - 107.0).abs() < 1e-9, "H[2]={}", high[2]);
        assert!((low[0]  -  98.0).abs() < 1e-9, "L[0]={}", low[0]);
        assert!((low[1]  - 101.0).abs() < 1e-9, "L[1]={}", low[1]);
        assert!((low[2]  - 103.0).abs() < 1e-9, "L[2]={}", low[2]);

        // O and C: gathers from bin_starts / bin_ends_minus_one
        let bin_starts = [0usize, 4, 8];
        let bin_ends_minus_one = [3usize, 7, 11];
        for g in 0..3 {
            let open  = prices[bin_starts[g]];
            let close = prices[bin_ends_minus_one[g]];
            assert!((open  - [100.0, 102.0, 106.0][g]).abs() < 1e-9, "O[{g}]={open}");
            assert!((close - [102.0, 106.0, 104.0][g]).abs() < 1e-9, "C[{g}]={close}");
        }
    }

    // ── Op::identity() and Op::degenerate() ─────────────────────────
    //
    // These tests enforce the monoid laws that prevent scan padding bugs.
    // Every scan over non-power-of-2 length pads with identity(), not degenerate().
    // Padding Max with NaN (degenerate) would corrupt every element it touched.

    #[test]
    fn op_identity_values_are_correct() {
        // Each identity must be the true neutral element.
        assert_eq!(Op::Add.identity(), 0.0, "Add identity");
        assert_eq!(Op::Max.identity(), f64::NEG_INFINITY, "Max identity");
        assert_eq!(Op::Min.identity(), f64::INFINITY, "Min identity");
        // ArgMin/ArgMax identities: sentinel values that never beat any real element.
        assert_eq!(Op::ArgMin.identity(), f64::INFINITY, "ArgMin identity");
        assert_eq!(Op::ArgMax.identity(), f64::NEG_INFINITY, "ArgMax identity");
        assert_eq!(Op::DotProduct.identity(), 0.0, "DotProduct identity");
        assert_eq!(Op::Distance.identity(), 0.0, "Distance identity");
    }

    #[test]
    fn op_degenerate_is_nan() {
        // degenerate() signals computation failure — always NaN for scalar ops.
        for op in [Op::Add, Op::Max, Op::Min, Op::ArgMin, Op::ArgMax, Op::DotProduct, Op::Distance] {
            assert!(op.degenerate().is_nan(), "{op:?}.degenerate() should be NaN");
        }
    }

    #[test]
    fn op_identity_and_degenerate_are_distinct() {
        // The key invariant: identity is a valid element, degenerate is not.
        // They must be different (identity must not be NaN for any Op).
        for op in [Op::Add, Op::Max, Op::Min, Op::DotProduct, Op::Distance] {
            let id = op.identity();
            assert!(!id.is_nan(), "{op:?}.identity() must not be NaN");
        }
    }

    /// Regression: Max over all-negative values padded to power-of-2 length.
    ///
    /// Before identity()/degenerate() distinction: padding used 0.0, causing
    /// max([-3, -1, -2]) to return 0.0 (the padding value) instead of -1.0.
    /// This test catches that class of bug.
    #[test]
    fn all_max_all_negative_values() {
        let mut e = engine();
        let values = vec![-3.0_f64, -1.0, -2.0]; // length 3 — non-power-of-2
        match e.accumulate(&values, Grouping::All, Expr::Value, Op::Max).unwrap() {
            AccResult::Scalar(s) => assert!((s - (-1.0)).abs() < 1e-9,
                "max of all-negative should be -1, got {s}"),
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    /// Regression: Min over all-positive values padded to power-of-2 length.
    ///
    /// Before the fix: padding used 0.0, causing min([3, 1, 2]) to return 0.0
    /// instead of 1.0. This test catches that class of bug.
    #[test]
    fn all_min_all_positive_values() {
        let mut e = engine();
        let values = vec![3.0_f64, 1.0, 2.0]; // length 3 — non-power-of-2
        match e.accumulate(&values, Grouping::All, Expr::Value, Op::Min).unwrap() {
            AccResult::Scalar(s) => assert!((s - 1.0).abs() < 1e-9,
                "min of all-positive should be 1, got {s}"),
            other => panic!("expected Scalar, got {other:?}"),
        }
    }
}
