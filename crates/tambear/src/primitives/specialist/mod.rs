//! Specialist primitives — slow but maximally accurate.
//!
//! These live outside the hot path. Their purpose is to serve as oracles
//! and reference implementations that the compensated and double-double
//! primitives are validated against. A recipe should not reach for these
//! in a production code path; it reaches for them at test time to prove
//! that the faster strategies give the right answer.
//!
//! # Inventory
//!
//! - `kulisch_accumulator`: an ~4352-bit signed fixed-point accumulator for
//!   exact summation of f64 values. No rounding, ever. Used as the gold
//!   standard when we need to know "what is `sum(xs)` really?"
//!
//! # Why separate from `compensated/` and `double_double/`
//!
//! The compensated primitives trade a constant factor of flops for bounded
//! error. The double-double type trades a larger constant factor for ~106
//! bits of precision. The specialist primitives trade a *much* larger
//! constant factor (often 10-100×) for **exact** results. They are
//! categorically different in cost and use.
//!
//! Recipes tagged `#[precision(correctly_rounded)]` rarely use these at
//! runtime — the double-double arithmetic is usually enough. But the
//! correctness *tests* for those recipes compare against the Kulisch
//! oracle to verify the final answer is correctly rounded.

pub mod kulisch_accumulator;
pub mod sum_k;

// Shared bookkeeping for streaming/mergeable accumulators (n / min /
// max plus the NaN/Inf-skip gate). Composed by every quantile sketch
// and any future accumulator that needs the same pattern.
pub mod observations;

// Quantile sketches — KLL (default), GK, t-digest. Common trait at
// quantile_sketch::QuantileSketch; concrete implementations selected
// at the recipe layer via using(sketch: "...").
pub mod quantile_sketch;
pub mod sketch_kll;
pub mod sketch_gk;
pub mod sketch_tdigest;
pub mod sketch_ddsketch;

pub use kulisch_accumulator::KulischAccumulator;
pub use sum_k::{sum_2, sum_3, sum_4, sum_k};
pub use observations::{
    FiniteObservations, MomentObservations, WeightedObservations, WelfordObservations,
};
pub use quantile_sketch::{QuantileSketch, SketchAlgorithm};
pub use sketch_kll::KllSketch;
pub use sketch_gk::GkSketch;
pub use sketch_tdigest::TdigestSketch;
pub use sketch_ddsketch::DdSketch;
