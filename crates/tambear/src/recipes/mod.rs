//! Recipes — Layer 3 of the atoms/primitives/recipes architecture.
//!
//! A recipe is any piece of mathematics with a literature name. It is
//! implemented as a composition of primitives (`crate::primitives::*`)
//! and atoms (`accumulate`, `gather`). Recipes live here; nothing in
//! this module contains inline arithmetic that bypasses the primitive
//! layer.
//!
//! # Organization
//!
//! Recipes are organized into tagged families, not into nested modules.
//! The top-level groupings are convenient browsing directories, but the
//! same recipe may belong to multiple families via the `#[family(…)]`
//! attribute (once that macro lands). For now, modules are filed under
//! the family that best describes their primary intent.
//!
//! - `libm`: elementary transcendentals (exp, log, sin, cos, tan, asin,
//!   acos, atan, sinh, cosh, tanh, erf, erfc, gamma, log_gamma). Each is
//!   implemented with three lowering strategies: strict (fast, 1-3 ulps),
//!   compensated (compensated arithmetic, sub-ulp error), and
//!   correctly_rounded (double-double working precision, 0 ulps).
//!
//! # Lowering strategies
//!
//! Each recipe in `libm` exposes three entry points:
//!
//! ```text
//! recipes::libm::exp::exp_strict(x)            // fast, 1-3 ulps
//! recipes::libm::exp::exp_compensated(x)       // ~1 ulp via compensated Horner
//! recipes::libm::exp::exp_correctly_rounded(x) // 0 ulps via double-double
//! ```
//!
//! The three share the same mathematical recipe — range reduction,
//! polynomial approximation, reconstruction — but lower to different
//! primitive sets. This is the first pilot of the strategy-dispatch
//! pattern; Phase D will add a declarative `#[precision(...)]` attribute
//! that auto-generates the three entry points from a single source.

pub mod libm;
pub mod pipelines;
pub mod statistics;
