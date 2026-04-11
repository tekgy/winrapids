//! # tambear-primitives
//!
//! The alphabet of computation. Three categories of atoms:
//!
//! - **Transforms** — what to do to each element (exp, ln, square, abs, ...)
//! - **Accumulates** — how to combine elements (All+Add, ByKey+Max, Prefix+Semiring, Tiled+DotProduct, ...)
//! - **Gathers** — how to read results (scalar, per_group, per_element, formula, ...)
//!
//! Math is expressed as: Transform → Accumulate → Gather chains.
//! Multiple chains fuse into single passes when they share (Grouping, Op).
//!
//! "Mean arithmetic" is not a primitive. It's a RECIPE:
//! ```text
//! identity  → Accumulate(All, Add) → "sum"
//! const(1)  → Accumulate(All, Add) → "count"    ← FUSES with above
//! Gather(scalar, sum / count)
//! ```
//!
//! The primitives are: identity, const, All, Add, scalar, division.
//! The recipe is how they compose. TAM compiles and executes recipes.
//!
//! ## The alphabet
//!
//! ```text
//! transforms/     — element-wise operations (the φ in scatter_phi)
//! accumulates/    — (grouping × op) pairs
//! gathers/        — addressing patterns for reading results
//! recipes/        — named compositions (teaching names for chains)
//! ```

pub mod tbs;
pub mod transforms;
pub mod accumulates;
pub mod gathers;
pub mod recipes;
