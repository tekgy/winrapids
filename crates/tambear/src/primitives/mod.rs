//! Terminal primitives layer.
//!
//! This module implements Layer 2 of the atoms/primitives/recipes architecture
//! described in `docs/architecture/atoms-primitives-recipes.md`.
//!
//! Primitives are **terminal operations**: either IEEE 754 hardware instructions
//! (`primitives::hardware`) or small sequences of hardware ops that implement
//! the compensated arithmetic foundations used by correctly-rounded recipes
//! (`primitives::compensated`, `primitives::double_double`).
//!
//! # The compositional rule
//!
//! Recipes (everything else in tambear) compose primitives via atoms
//! (`accumulate`, `gather`). A recipe may call primitives, other recipes,
//! and the atoms. A recipe MUST NOT contain inline arithmetic that bypasses
//! the primitive layer — `x * y + z` inside a recipe should be written as
//! `fmadd(x, y, z)` so the computation graph is explicit and single-rounding
//! semantics are preserved where available.
//!
//! # Why this matters
//!
//! 1. **Correctness by construction.** The primitive layer provides IEEE 754-2019
//!    semantics (notably `fmin`/`fmax` with NaN propagation). Recipes built on
//!    these primitives cannot have the NaN-eating bug class that affected
//!    ~11 functions in the 2026-04-10 adversarial sweep.
//! 2. **Backend portability.** Primitives are the one place where tambear touches
//!    `f64::mul_add` and friends. Porting to SPIR-V or PTX means replacing the
//!    primitive layer, not every recipe.
//! 3. **Precision strategy dispatch.** A recipe written as a tree of primitive
//!    calls can be lowered with different strategies (strict / compensated /
//!    correctly_rounded) without changing the source. The lowering pass walks
//!    the primitive tree and substitutes implementations.
//! 4. **Automatic differentiation.** Every primitive has a known derivative.
//!    Recipes built from primitives are differentiable by graph traversal.
//! 5. **Numerical error analysis.** Per-primitive ULP bounds compose predictably.
//!    Whole-recipe error bounds are derivable from the tree.

pub mod compensated;
pub mod constants;
pub mod double_double;
pub mod hardware;
pub mod oracle;
pub mod specialist;
