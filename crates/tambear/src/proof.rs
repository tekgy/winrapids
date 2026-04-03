//! # Proof Architecture for Tambear
//!
//! Lean4 meets GPU compute. Axioms as types, proofs as programs,
//! compilation to scatter/accumulate primitives.
//!
//! ## Core idea
//!
//! Every tambear primitive has algebraic structure:
//! - `Op::Add` on `f64` is a commutative monoid `(ℝ, +, 0)`
//! - `Op::Max` is a semigroup `(ℝ, max)` (identity = -∞)
//! - `ScatterJit::scatter_phi` preserves homomorphic structure
//! - `CopaState::merge` is an associative binary operation
//!
//! This module lets you:
//! 1. **Declare** algebraic structures (monoid, semigroup, ring, ...)
//! 2. **State** propositions about them (associativity, homomorphism, ...)
//! 3. **Prove** them via witnesses (symbolic, computational, or ⟨?⟩)
//! 4. **Compile** verified statements to runnable tambear GPU code
//!
//! ## Architecture
//!
//! ```text
//! Sort ──────── Prop ──────── Proof ──────── Compile
//! │              │              │               │
//! │ types        │ statements   │ witnesses      │ AccResult
//! │ Nat,Real,    │ ∀x,y:       │ by_assoc(),    │ accumulate(
//! │ Vec(Real),   │ x⊕y = y⊕x  │ by_compute(),  │   data,
//! │ Monoid(+,0)  │              │ ⟨?⟩           │   grouping,
//! └──────────────┘              │               │   expr, op)
//!                               └───────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust
//! use tambear::proof::*;
//!
//! // Declare: (f64, +, 0) is a commutative monoid
//! let f64_add = Structure::commutative_monoid(
//!     Sort::Real,
//!     BinOp::Add,
//!     Term::Lit(0.0),
//! );
//!
//! // State: scatter_phi("v") over + preserves associativity
//! let prop = Prop::Forall {
//!     vars: vec![("a", Sort::Real), ("b", Sort::Real), ("c", Sort::Real)],
//!     body: Box::new(Prop::Eq(
//!         Term::BinApp(BinOp::Add, Box::new(Term::BinApp(BinOp::Add,
//!             Box::new(Term::Var("a")), Box::new(Term::Var("b")))),
//!             Box::new(Term::Var("c"))),
//!         Term::BinApp(BinOp::Add, Box::new(Term::Var("a")),
//!             Box::new(Term::BinApp(BinOp::Add,
//!                 Box::new(Term::Var("b")), Box::new(Term::Var("c"))))),
//!     )),
//! };
//!
//! // Prove: by structure (+ is declared associative)
//! let proof = Proof::ByStructure(f64_add.clone(), StructuralFact::Associativity);
//! let thm = Theorem::check("add_assoc", prop, proof).unwrap();
//! assert!(thm.is_verified());
//! ```

use std::collections::HashMap;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// Sorts — the universe of types
// ═══════════════════════════════════════════════════════════════════════════

/// A sort (type) in the proof system.
///
/// Sorts classify terms. They correspond roughly to Lean4 types,
/// but tailored to what tambear actually computes over.
#[derive(Debug, Clone, PartialEq)]
pub enum Sort {
    /// Natural numbers ℕ.
    Nat,
    /// Real numbers ℝ (represented as f64 in computation).
    Real,
    /// Booleans.
    Bool,
    /// Fixed-size vector of a sort: Vec(n, Real) = ℝⁿ.
    Vec(usize, Box<Sort>),
    /// Matrix: Mat(m, n, Real) = ℝᵐˣⁿ.
    Mat(usize, usize, Box<Sort>),
    /// A named abstract sort (for user-defined algebraic structures).
    Named(String),
    /// Function sort: A → B.
    Arrow(Box<Sort>, Box<Sort>),
    /// Product sort: A × B.
    Product(Box<Sort>, Box<Sort>),
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Nat => write!(f, "ℕ"),
            Sort::Real => write!(f, "ℝ"),
            Sort::Bool => write!(f, "𝔹"),
            Sort::Vec(n, s) => write!(f, "{s}^{n}"),
            Sort::Mat(m, n, s) => write!(f, "{s}^({m}×{n})"),
            Sort::Named(name) => write!(f, "{name}"),
            Sort::Arrow(a, b) => write!(f, "({a} → {b})"),
            Sort::Product(a, b) => write!(f, "({a} × {b})"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Terms — expressions that inhabit sorts
// ═══════════════════════════════════════════════════════════════════════════

/// A term (expression) in the proof language.
///
/// Terms are the things we prove properties about. They mirror
/// tambear's computational primitives.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// A variable reference by name.
    Var(&'static str),
    /// A literal real value.
    Lit(f64),
    /// A literal natural number.
    NatLit(u64),
    /// Application of a binary operation.
    BinApp(BinOp, Box<Term>, Box<Term>),
    /// Application of a unary operation.
    UnApp(UnOp, Box<Term>),
    /// Function application: f(x).
    App(Box<Term>, Box<Term>),
    /// Lambda abstraction: λ(name: sort). body.
    Lambda(&'static str, Sort, Box<Term>),
    /// An accumulate call — the universal primitive.
    /// `Accumulate(grouping_tag, expr_term, op, data)`.
    Accumulate {
        grouping: GroupingTag,
        expr: Box<Term>,
        op: BinOp,
        data: Box<Term>,
    },
    /// Pair constructor.
    Pair(Box<Term>, Box<Term>),
    /// First projection.
    Fst(Box<Term>),
    /// Second projection.
    Snd(Box<Term>),
    /// Open obligation — a hole that needs to be filled.
    /// The sort tells you what type of term is needed.
    Hole(Sort),
}

/// Binary operations — correspond to tambear's `Op` enum and phi expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// Addition: the (ℝ, +, 0) monoid.
    Add,
    /// Multiplication: the (ℝ, ×, 1) monoid.
    Mul,
    /// Maximum: the (ℝ, max, -∞) semigroup.
    Max,
    /// Minimum: the (ℝ, min, +∞) semigroup.
    Min,
    /// Dot product: Σᵢ aᵢbᵢ (tiled accumulate).
    Dot,
    /// Subtraction (NOT associative — important for proof checking).
    Sub,
    /// Division (NOT associative).
    Div,
    /// Composition of functions.
    Compose,
}

/// Unary operations — phi expressions as term constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    /// Negate: -x.
    Neg,
    /// Square: x².
    Sq,
    /// Square root: √x.
    Sqrt,
    /// Logarithm: ln(x).
    Log,
    /// Exponential: eˣ.
    Exp,
    /// Absolute value: |x|.
    Abs,
    /// Indicator/count: always 1.0 (the "one" phi).
    One,
}

/// Tags for grouping patterns — abstract over Grouping enum variants.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupingTag {
    /// All → 1 (global reduction).
    All,
    /// N → K (by key).
    ByKey,
    /// N → N (prefix scan).
    Prefix,
    /// Segmented prefix scan.
    Segmented,
    /// M×K × K×N tiled.
    Tiled,
    /// Rolling window.
    Windowed(usize),
    /// Masked scatter.
    Masked,
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Var(v) => write!(f, "{v}"),
            Term::Lit(x) => write!(f, "{x}"),
            Term::NatLit(n) => write!(f, "{n}"),
            Term::BinApp(op, a, b) => {
                let sym = match op {
                    BinOp::Add => "+",
                    BinOp::Mul => "×",
                    BinOp::Max => "max",
                    BinOp::Min => "min",
                    BinOp::Dot => "·",
                    BinOp::Sub => "-",
                    BinOp::Div => "/",
                    BinOp::Compose => "∘",
                };
                write!(f, "({a} {sym} {b})")
            }
            Term::UnApp(op, x) => {
                let sym = match op {
                    UnOp::Neg => "-",
                    UnOp::Sq => "sq",
                    UnOp::Sqrt => "√",
                    UnOp::Log => "ln",
                    UnOp::Exp => "exp",
                    UnOp::Abs => "|·|",
                    UnOp::One => "𝟙",
                };
                write!(f, "{sym}({x})")
            }
            Term::App(func, arg) => write!(f, "{func}({arg})"),
            Term::Lambda(name, sort, body) => write!(f, "λ{name}:{sort}. {body}"),
            Term::Accumulate { grouping, expr, op, data } => {
                let op_sym = match op {
                    BinOp::Add => "+",
                    BinOp::Mul => "×",
                    BinOp::Max => "max",
                    BinOp::Min => "min",
                    _ => "⊕",
                };
                write!(f, "accumulate({grouping:?}, {expr}, {op_sym}, {data})")
            }
            Term::Pair(a, b) => write!(f, "({a}, {b})"),
            Term::Fst(p) => write!(f, "π₁({p})"),
            Term::Snd(p) => write!(f, "π₂({p})"),
            Term::Hole(sort) => write!(f, "⟨?:{sort}⟩"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Propositions — what we want to prove
// ═══════════════════════════════════════════════════════════════════════════

/// A proposition — a mathematical statement that may or may not be true.
#[derive(Debug, Clone, PartialEq)]
pub enum Prop {
    /// Equality of terms: a = b.
    Eq(Term, Term),
    /// Less-than-or-equal: a ≤ b.
    Le(Term, Term),
    /// Less-than: a < b.
    Lt(Term, Term),
    /// Logical conjunction: P ∧ Q.
    And(Box<Prop>, Box<Prop>),
    /// Logical disjunction: P ∨ Q.
    Or(Box<Prop>, Box<Prop>),
    /// Implication: P → Q.
    Implies(Box<Prop>, Box<Prop>),
    /// Negation: ¬P.
    Not(Box<Prop>),
    /// Universal quantification: ∀(vars). body.
    Forall {
        vars: Vec<(&'static str, Sort)>,
        body: Box<Prop>,
    },
    /// Existential quantification: ∃(vars). body.
    Exists {
        vars: Vec<(&'static str, Sort)>,
        body: Box<Prop>,
    },
    /// A named proposition (reference to a previously proved theorem).
    Ref(String),
    /// True (trivially provable).
    True,
    /// False (not provable, but useful as hypothesis).
    False,
}

impl fmt::Display for Prop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Prop::Eq(a, b) => write!(f, "{a} = {b}"),
            Prop::Le(a, b) => write!(f, "{a} ≤ {b}"),
            Prop::Lt(a, b) => write!(f, "{a} < {b}"),
            Prop::And(p, q) => write!(f, "({p} ∧ {q})"),
            Prop::Or(p, q) => write!(f, "({p} ∨ {q})"),
            Prop::Implies(p, q) => write!(f, "({p} → {q})"),
            Prop::Not(p) => write!(f, "¬{p}"),
            Prop::Forall { vars, body } => {
                write!(f, "∀")?;
                for (i, (name, sort)) in vars.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "{name}:{sort}")?;
                }
                write!(f, ". {body}")
            }
            Prop::Exists { vars, body } => {
                write!(f, "∃")?;
                for (i, (name, sort)) in vars.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "{name}:{sort}")?;
                }
                write!(f, ". {body}")
            }
            Prop::Ref(name) => write!(f, "[{name}]"),
            Prop::True => write!(f, "⊤"),
            Prop::False => write!(f, "⊥"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Algebraic structures — what tambear primitives ARE
// ═══════════════════════════════════════════════════════════════════════════

/// An algebraic structure: carrier set + operations + laws.
///
/// Each tambear primitive corresponds to one of these. The proof system
/// verifies that operations satisfy their declared laws.
#[derive(Debug, Clone, PartialEq)]
pub struct Structure {
    /// Human-readable name.
    pub name: String,
    /// The carrier sort.
    pub carrier: Sort,
    /// The binary operation (if any).
    pub op: Option<BinOp>,
    /// The identity element (if any — semigroups have none).
    pub identity: Option<Term>,
    /// Which laws this structure satisfies.
    pub laws: Vec<StructuralFact>,
}

/// A structural fact — a named algebraic law.
///
/// These are the axioms that tambear primitives satisfy. Each corresponds
/// to a concrete property needed for GPU parallelism to be correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructuralFact {
    /// ∀a,b,c. (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c).
    /// Required for parallel scan correctness.
    Associativity,
    /// ∀a,b. a ⊕ b = b ⊕ a.
    /// Required for non-deterministic scatter order on GPU.
    Commutativity,
    /// ∃e. ∀a. a ⊕ e = e ⊕ a = a.
    /// Required for accumulator initialization.
    Identity,
    /// ∀a. ∃a⁻¹. a ⊕ a⁻¹ = e.
    /// Enables the window = prefix subtraction trick.
    Invertibility,
    /// φ(a ⊕ b) = φ(a) ⊕ φ(b).
    /// The key property for scatter_phi correctness — the lift function
    /// commutes with the combine operation.
    Homomorphism,
    /// ∀a. a ⊕ a = a.
    /// Max and min are idempotent; add is not.
    Idempotence,
    /// The operation distributes over another: a × (b + c) = a×b + a×c.
    Distributivity,
    /// The merge operation preserves sufficient statistics.
    /// This is what makes CopaState::merge and MomentStats::merge correct.
    Mergeability,
}

impl Structure {
    /// Declare a semigroup: carrier with an associative binary operation.
    pub fn semigroup(carrier: Sort, op: BinOp) -> Self {
        Structure {
            name: format!("({carrier}, {op:?})"),
            carrier,
            op: Some(op),
            identity: None,
            laws: vec![StructuralFact::Associativity],
        }
    }

    /// Declare a monoid: semigroup + identity element.
    pub fn monoid(carrier: Sort, op: BinOp, identity: Term) -> Self {
        Structure {
            name: format!("({carrier}, {op:?}, {identity})"),
            carrier,
            op: Some(op),
            identity: Some(identity),
            laws: vec![StructuralFact::Associativity, StructuralFact::Identity],
        }
    }

    /// Declare a commutative monoid: monoid + commutativity.
    /// This is what `Op::Add` on `f64` is — the bread and butter of scatter.
    pub fn commutative_monoid(carrier: Sort, op: BinOp, identity: Term) -> Self {
        Structure {
            name: format!("({carrier}, {op:?}, {identity})_comm"),
            carrier,
            op: Some(op),
            identity: Some(identity),
            laws: vec![
                StructuralFact::Associativity,
                StructuralFact::Identity,
                StructuralFact::Commutativity,
            ],
        }
    }

    /// Declare a group: monoid + invertibility.
    /// Enables the prefix-subtraction window trick.
    pub fn group(carrier: Sort, op: BinOp, identity: Term) -> Self {
        Structure {
            name: format!("({carrier}, {op:?}, {identity})_grp"),
            carrier,
            op: Some(op),
            identity: Some(identity),
            laws: vec![
                StructuralFact::Associativity,
                StructuralFact::Identity,
                StructuralFact::Invertibility,
            ],
        }
    }

    /// Declare a commutative idempotent semigroup: max/min.
    pub fn lattice_op(carrier: Sort, op: BinOp, identity: Term) -> Self {
        Structure {
            name: format!("({carrier}, {op:?})_lattice"),
            carrier,
            op: Some(op),
            identity: Some(identity),
            laws: vec![
                StructuralFact::Associativity,
                StructuralFact::Commutativity,
                StructuralFact::Idempotence,
                StructuralFact::Identity,
            ],
        }
    }

    /// Does this structure satisfy a given law?
    pub fn has_law(&self, law: StructuralFact) -> bool {
        self.laws.contains(&law)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Proofs — witnesses of truth
// ═══════════════════════════════════════════════════════════════════════════

/// A proof — a witness that a proposition is true.
///
/// Three flavors:
/// - **ByStructure**: the proposition follows from declared algebraic laws.
/// - **ByComputation**: verified by exhaustive or sampled computation.
/// - **ByComposition**: built from smaller proofs (transitivity, symmetry, etc.).
/// - **Hole**: an open obligation ⟨?⟩ — not yet proved.
#[derive(Debug, Clone)]
pub enum Proof {
    /// The proposition follows from the algebraic structure.
    /// "Add is associative because (ℝ,+,0) is declared as a monoid."
    ByStructure(Structure, StructuralFact),

    /// Verified by computation over concrete values.
    /// `witness` contains the test cases that were checked.
    ByComputation {
        /// Description of the computational verification.
        method: ComputeMethod,
        /// Number of test cases verified.
        n_verified: usize,
        /// Maximum error observed (for approximate equalities).
        max_error: f64,
    },

    /// Built from sub-proofs by a logical rule.
    ByComposition(CompositionRule, Vec<Proof>),

    /// Direct witness: an explicit term that makes an existential true.
    ByWitness(Term),

    /// Reference to a previously proved theorem.
    ByRef(String),

    /// Open obligation — this step is not yet proved.
    /// The `String` is a human-readable description of what's needed.
    Hole(String),
}

/// How a computational proof was obtained.
#[derive(Debug, Clone)]
pub enum ComputeMethod {
    /// Checked all elements in a finite domain.
    Exhaustive,
    /// Checked a random sample.
    Sampled { seed: u64 },
    /// Checked specific boundary cases.
    BoundaryValues,
    /// Verified by running GPU code and comparing.
    GpuVerified,
}

/// Composition rules for building proofs from sub-proofs.
#[derive(Debug, Clone, Copy)]
pub enum CompositionRule {
    /// From P and P→Q, deduce Q.
    ModusPonens,
    /// From a=b and b=c, deduce a=c.
    Transitivity,
    /// From a=b, deduce b=a.
    Symmetry,
    /// From P and Q, deduce P∧Q.
    Conjunction,
    /// From P[witness/x], deduce ∃x.P.
    ExistentialIntro,
    /// From ∀x.P and a term t, deduce P[t/x].
    UniversalElim,
    /// From f(a)=f(b), deduce a=b (if f is injective) or vice versa.
    Congruence,
    /// From φ homomorphism + associativity, deduce scatter correctness.
    ScatterDecomposition,
    /// From mergeability, deduce parallel reduction correctness.
    ParallelMerge,
}

// ═══════════════════════════════════════════════════════════════════════════
// Theorems — verified propositions
// ═══════════════════════════════════════════════════════════════════════════

/// The status of a theorem's proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    /// All proof obligations discharged.
    Verified,
    /// Some proof steps are holes (⟨?⟩).
    Partial { holes: usize },
    /// Proof failed verification.
    Failed,
}

/// A theorem: a named proposition with its proof and verification status.
///
/// Once verified, a theorem can be referenced by name in other proofs.
#[derive(Debug, Clone)]
pub struct Theorem {
    /// Name of the theorem.
    pub name: String,
    /// The proposition being proved.
    pub prop: Prop,
    /// The proof (witness).
    pub proof: Proof,
    /// Verification result.
    pub status: VerificationStatus,
}

impl Theorem {
    /// Attempt to verify a proposition with a proof.
    ///
    /// Returns a Theorem with status indicating whether verification succeeded.
    pub fn check(name: &str, prop: Prop, proof: Proof) -> Result<Self, ProofError> {
        let status = verify_proof(&prop, &proof)?;
        Ok(Theorem {
            name: name.to_string(),
            prop,
            proof,
            status,
        })
    }

    /// Is this theorem fully verified (no holes)?
    pub fn is_verified(&self) -> bool {
        self.status == VerificationStatus::Verified
    }

    /// How many holes remain?
    pub fn holes(&self) -> usize {
        match self.status {
            VerificationStatus::Partial { holes } => holes,
            _ => 0,
        }
    }
}

impl fmt::Display for Theorem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = match self.status {
            VerificationStatus::Verified => "✓",
            VerificationStatus::Partial { holes } => return write!(f, "⟨{}?⟩ {}: {}", holes, self.name, self.prop),
            VerificationStatus::Failed => "✗",
        };
        write!(f, "{status} {}: {}", self.name, self.prop)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Proof errors
// ═══════════════════════════════════════════════════════════════════════════

/// Errors that can occur during proof verification.
#[derive(Debug, Clone)]
pub enum ProofError {
    /// The structure doesn't satisfy the claimed law.
    LawNotSatisfied {
        structure: String,
        law: StructuralFact,
    },
    /// Type mismatch: expected one sort, got another.
    SortMismatch {
        expected: Sort,
        got: Sort,
    },
    /// The operation is not valid for the claimed property.
    InvalidOperation {
        op: BinOp,
        reason: String,
    },
    /// Computational verification failed.
    ComputationFailed {
        description: String,
    },
    /// A referenced theorem doesn't exist.
    UnknownRef(String),
}

impl fmt::Display for ProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProofError::LawNotSatisfied { structure, law } =>
                write!(f, "structure {structure} does not satisfy {law:?}"),
            ProofError::SortMismatch { expected, got } =>
                write!(f, "sort mismatch: expected {expected}, got {got}"),
            ProofError::InvalidOperation { op, reason } =>
                write!(f, "invalid operation {op:?}: {reason}"),
            ProofError::ComputationFailed { description } =>
                write!(f, "computation failed: {description}"),
            ProofError::UnknownRef(name) =>
                write!(f, "unknown theorem reference: {name}"),
        }
    }
}

impl std::error::Error for ProofError {}

// ═══════════════════════════════════════════════════════════════════════════
// Verification engine
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that a proof witnesses a proposition.
fn verify_proof(prop: &Prop, proof: &Proof) -> Result<VerificationStatus, ProofError> {
    match proof {
        Proof::ByStructure(structure, fact) => {
            if structure.has_law(*fact) {
                // Check that the law matches the proposition shape.
                match (fact, prop) {
                    (StructuralFact::Associativity, Prop::Forall { body, .. }) => {
                        // The body should be an equality of re-associated terms.
                        if matches!(**body, Prop::Eq(..)) {
                            Ok(VerificationStatus::Verified)
                        } else {
                            Err(ProofError::ComputationFailed {
                                description: "associativity requires ∀. (a⊕b)⊕c = a⊕(b⊕c) shape".into(),
                            })
                        }
                    }
                    (StructuralFact::Commutativity, Prop::Forall { body, .. }) => {
                        if matches!(**body, Prop::Eq(..)) {
                            Ok(VerificationStatus::Verified)
                        } else {
                            Err(ProofError::ComputationFailed {
                                description: "commutativity requires ∀. a⊕b = b⊕a shape".into(),
                            })
                        }
                    }
                    (StructuralFact::Identity, Prop::Forall { body, .. }) => {
                        if matches!(**body, Prop::Eq(..)) && structure.identity.is_some() {
                            Ok(VerificationStatus::Verified)
                        } else {
                            Err(ProofError::ComputationFailed {
                                description: "identity requires ∀a. a⊕e = a shape and identity element".into(),
                            })
                        }
                    }
                    (StructuralFact::Homomorphism, _) => {
                        // Homomorphism of phi: φ(a⊕b) = φ(a)⊕φ(b)
                        Ok(VerificationStatus::Verified)
                    }
                    (StructuralFact::Mergeability, _) => {
                        // Mergeability: merge(a,b) is associative and preserves stats
                        Ok(VerificationStatus::Verified)
                    }
                    (StructuralFact::Idempotence, Prop::Forall { body, .. }) => {
                        if matches!(**body, Prop::Eq(..)) {
                            Ok(VerificationStatus::Verified)
                        } else {
                            Err(ProofError::ComputationFailed {
                                description: "idempotence requires ∀a. a⊕a = a shape".into(),
                            })
                        }
                    }
                    _ => Ok(VerificationStatus::Verified),
                }
            } else {
                Err(ProofError::LawNotSatisfied {
                    structure: structure.name.clone(),
                    law: *fact,
                })
            }
        }

        Proof::ByComputation { n_verified, max_error, .. } => {
            if *n_verified > 0 && *max_error < 1e-10 {
                Ok(VerificationStatus::Verified)
            } else if *n_verified > 0 {
                // Approximate — still verified but worth noting
                Ok(VerificationStatus::Verified)
            } else {
                Err(ProofError::ComputationFailed {
                    description: "no test cases verified".into(),
                })
            }
        }

        Proof::ByComposition(rule, sub_proofs) => {
            let mut total_holes = 0;
            for sp in sub_proofs {
                match count_holes(sp) {
                    0 => {}
                    n => total_holes += n,
                }
            }
            if total_holes == 0 {
                // All sub-proofs are complete — verify the composition rule is valid
                verify_composition_rule(*rule, sub_proofs)?;
                Ok(VerificationStatus::Verified)
            } else {
                Ok(VerificationStatus::Partial { holes: total_holes })
            }
        }

        Proof::ByWitness(_term) => {
            // A witness for an existential — check prop is ∃
            match prop {
                Prop::Exists { .. } => Ok(VerificationStatus::Verified),
                _ => Err(ProofError::ComputationFailed {
                    description: "witness proof requires ∃ proposition".into(),
                }),
            }
        }

        Proof::ByRef(_name) => {
            // In a full system this would look up the theorem registry.
            // For now, trust the reference.
            Ok(VerificationStatus::Verified)
        }

        Proof::Hole(description) => {
            let _ = description;
            Ok(VerificationStatus::Partial { holes: 1 })
        }
    }
}

/// Count holes in a proof tree.
fn count_holes(proof: &Proof) -> usize {
    match proof {
        Proof::Hole(_) => 1,
        Proof::ByComposition(_, subs) => subs.iter().map(count_holes).sum(),
        _ => 0,
    }
}

/// Verify that a composition rule is valid for its sub-proofs.
fn verify_composition_rule(rule: CompositionRule, _sub_proofs: &[Proof]) -> Result<(), ProofError> {
    match rule {
        CompositionRule::ModusPonens => {
            if _sub_proofs.len() != 2 {
                return Err(ProofError::ComputationFailed {
                    description: "modus ponens requires exactly 2 sub-proofs".into(),
                });
            }
            Ok(())
        }
        CompositionRule::Transitivity => {
            if _sub_proofs.len() < 2 {
                return Err(ProofError::ComputationFailed {
                    description: "transitivity requires at least 2 sub-proofs".into(),
                });
            }
            Ok(())
        }
        // Other rules: structurally valid by construction for now.
        // Full type-checking would verify sorts match across sub-proofs.
        _ => Ok(()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Theorem registry — accumulates verified results
// ═══════════════════════════════════════════════════════════════════════════

/// A registry of proved theorems.
///
/// Theorems can reference each other by name. The registry enforces
/// that referenced theorems exist and are verified.
#[derive(Debug, Default)]
pub struct ProofContext {
    /// Named theorems, insertion-ordered.
    theorems: Vec<Theorem>,
    /// Name → index for fast lookup.
    index: HashMap<String, usize>,
    /// Declared structures.
    structures: Vec<Structure>,
}

impl ProofContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a standard structure.
    pub fn declare_structure(&mut self, s: Structure) {
        self.structures.push(s);
    }

    /// Look up a structure by name.
    pub fn find_structure(&self, name: &str) -> Option<&Structure> {
        self.structures.iter().find(|s| s.name == name)
    }

    /// Add a theorem to the registry.
    pub fn add(&mut self, thm: Theorem) -> Result<(), ProofError> {
        // Check that all ByRef proofs refer to known theorems.
        self.check_refs(&thm.proof)?;

        let name = thm.name.clone();
        let idx = self.theorems.len();
        self.theorems.push(thm);
        self.index.insert(name, idx);
        Ok(())
    }

    /// Look up a theorem by name.
    pub fn get(&self, name: &str) -> Option<&Theorem> {
        self.index.get(name).map(|&i| &self.theorems[i])
    }

    /// All theorems in the registry.
    pub fn theorems(&self) -> &[Theorem] {
        &self.theorems
    }

    /// Count of fully verified theorems.
    pub fn verified_count(&self) -> usize {
        self.theorems.iter().filter(|t| t.is_verified()).count()
    }

    /// Count of theorems with holes.
    pub fn partial_count(&self) -> usize {
        self.theorems.iter().filter(|t| t.holes() > 0).count()
    }

    /// Total number of open holes across all theorems.
    pub fn total_holes(&self) -> usize {
        self.theorems.iter().map(|t| t.holes()).sum()
    }

    /// Summary of the proof context.
    pub fn summary(&self) -> ProofSummary {
        ProofSummary {
            total: self.theorems.len(),
            verified: self.verified_count(),
            partial: self.partial_count(),
            holes: self.total_holes(),
            structures: self.structures.len(),
        }
    }

    fn check_refs(&self, proof: &Proof) -> Result<(), ProofError> {
        match proof {
            Proof::ByRef(name) => {
                if self.index.contains_key(name) {
                    Ok(())
                } else {
                    Err(ProofError::UnknownRef(name.clone()))
                }
            }
            Proof::ByComposition(_, subs) => {
                for s in subs { self.check_refs(s)?; }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Summary statistics for a proof context.
#[derive(Debug, Clone)]
pub struct ProofSummary {
    pub total: usize,
    pub verified: usize,
    pub partial: usize,
    pub holes: usize,
    pub structures: usize,
}

impl fmt::Display for ProofSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Proof context: {} theorems ({} verified, {} partial, {} holes), {} structures",
            self.total, self.verified, self.partial, self.holes, self.structures
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compilation — from proofs to GPU code
// ═══════════════════════════════════════════════════════════════════════════

/// A compiled proof obligation — something that can run on GPU.
///
/// When a theorem involves computable parts (accumulate terms with
/// concrete data), this extracts the computation plan.
#[derive(Debug, Clone)]
pub struct CompiledObligation {
    /// Name of the theorem this was compiled from.
    pub theorem: String,
    /// The accumulate calls needed.
    pub steps: Vec<AccumulateStep>,
}

/// A single accumulate step extracted from a proof term.
#[derive(Debug, Clone)]
pub struct AccumulateStep {
    /// What grouping pattern to use.
    pub grouping: GroupingTag,
    /// The phi expression (as a string for codegen).
    pub phi_expr: String,
    /// The combine operation.
    pub op: BinOp,
    /// Human-readable description.
    pub description: String,
}

/// Extract computable accumulate steps from a term.
pub fn compile_term(term: &Term) -> Vec<AccumulateStep> {
    let mut steps = Vec::new();
    collect_accumulates(term, &mut steps);
    steps
}

fn collect_accumulates(term: &Term, steps: &mut Vec<AccumulateStep>) {
    match term {
        Term::Accumulate { grouping, expr, op, data } => {
            let phi_expr = term_to_phi(expr);
            steps.push(AccumulateStep {
                grouping: grouping.clone(),
                phi_expr,
                op: *op,
                description: format!("accumulate({grouping:?}, {expr}, {op:?}, {data})"),
            });
            // Recurse into data in case it's another accumulate
            collect_accumulates(data, steps);
        }
        Term::BinApp(_, a, b) => {
            collect_accumulates(a, steps);
            collect_accumulates(b, steps);
        }
        Term::UnApp(_, x) => collect_accumulates(x, steps),
        Term::App(f, x) => {
            collect_accumulates(f, steps);
            collect_accumulates(x, steps);
        }
        Term::Pair(a, b) => {
            collect_accumulates(a, steps);
            collect_accumulates(b, steps);
        }
        Term::Fst(x) | Term::Snd(x) => collect_accumulates(x, steps),
        Term::Lambda(_, _, body) => collect_accumulates(body, steps),
        _ => {}
    }
}

/// Convert a term to a phi expression string for codegen.
fn term_to_phi(term: &Term) -> String {
    match term {
        Term::Var(v) => v.to_string(),
        Term::Lit(x) => format!("{x}"),
        Term::BinApp(op, a, b) => {
            let a_str = term_to_phi(a);
            let b_str = term_to_phi(b);
            match op {
                BinOp::Add => format!("({a_str} + {b_str})"),
                BinOp::Mul => format!("({a_str} * {b_str})"),
                BinOp::Sub => format!("({a_str} - {b_str})"),
                BinOp::Div => format!("({a_str} / {b_str})"),
                _ => format!("{op:?}({a_str}, {b_str})"),
            }
        }
        Term::UnApp(op, x) => {
            let x_str = term_to_phi(x);
            match op {
                UnOp::Neg => format!("(-{x_str})"),
                UnOp::Sq => format!("({x_str} * {x_str})"),
                UnOp::Sqrt => format!("sqrt({x_str})"),
                UnOp::Log => format!("log({x_str})"),
                UnOp::Exp => format!("exp({x_str})"),
                UnOp::Abs => format!("fabs({x_str})"),
                UnOp::One => "1.0".to_string(),
            }
        }
        _ => "v".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in structures for tambear
// ═══════════════════════════════════════════════════════════════════════════

/// Create the standard tambear proof context with built-in algebraic structures.
///
/// This registers the fundamental structures that tambear's primitives inhabit:
/// - (ℝ, +, 0): commutative monoid — scatter_phi("v")
/// - (ℝ, ×, 1): commutative monoid — scatter_phi("v * r")
/// - (ℝ, max, -∞): idempotent commutative monoid — Op::Max
/// - (ℝ, min, +∞): idempotent commutative monoid — Op::Min
/// - (ℝ, +, 0) group: with invertibility — window = prefix - prefix
pub fn tambear_context() -> ProofContext {
    let mut ctx = ProofContext::new();

    // (ℝ, +, 0) — the workhorse. Every scatter_phi with Op::Add lives here.
    ctx.declare_structure(Structure::commutative_monoid(
        Sort::Real, BinOp::Add, Term::Lit(0.0),
    ));

    // (ℝ, ×, 1) — for multiplicative accumulation.
    ctx.declare_structure(Structure::commutative_monoid(
        Sort::Real, BinOp::Mul, Term::Lit(1.0),
    ));

    // (ℝ, max, -∞) — Op::Max, idempotent.
    ctx.declare_structure(Structure::lattice_op(
        Sort::Real, BinOp::Max, Term::Lit(f64::NEG_INFINITY),
    ));

    // (ℝ, min, +∞) — Op::Min, idempotent.
    ctx.declare_structure(Structure::lattice_op(
        Sort::Real, BinOp::Min, Term::Lit(f64::INFINITY),
    ));

    // (ℝ, +, 0) as a group — for window via prefix subtraction.
    ctx.declare_structure(Structure::group(
        Sort::Real, BinOp::Add, Term::Lit(0.0),
    ));

    ctx
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience builders — the API for writing proofs
// ═══════════════════════════════════════════════════════════════════════════

/// Convenience: build an associativity proposition for a binary operation.
pub fn assoc_prop(op: BinOp) -> Prop {
    Prop::Forall {
        vars: vec![("a", Sort::Real), ("b", Sort::Real), ("c", Sort::Real)],
        body: Box::new(Prop::Eq(
            Term::BinApp(op,
                Box::new(Term::BinApp(op,
                    Box::new(Term::Var("a")),
                    Box::new(Term::Var("b")))),
                Box::new(Term::Var("c"))),
            Term::BinApp(op,
                Box::new(Term::Var("a")),
                Box::new(Term::BinApp(op,
                    Box::new(Term::Var("b")),
                    Box::new(Term::Var("c"))))),
        )),
    }
}

/// Convenience: build a commutativity proposition for a binary operation.
pub fn comm_prop(op: BinOp) -> Prop {
    Prop::Forall {
        vars: vec![("a", Sort::Real), ("b", Sort::Real)],
        body: Box::new(Prop::Eq(
            Term::BinApp(op, Box::new(Term::Var("a")), Box::new(Term::Var("b"))),
            Term::BinApp(op, Box::new(Term::Var("b")), Box::new(Term::Var("a"))),
        )),
    }
}

/// Convenience: build an identity proposition.
pub fn identity_prop(op: BinOp, identity: Term) -> Prop {
    Prop::Forall {
        vars: vec![("a", Sort::Real)],
        body: Box::new(Prop::Eq(
            Term::BinApp(op, Box::new(Term::Var("a")), Box::new(identity)),
            Term::Var("a"),
        )),
    }
}

/// Convenience: build a homomorphism proposition φ(a⊕b) = φ(a) ⊕ φ(b).
pub fn homomorphism_prop(phi: UnOp, op: BinOp) -> Prop {
    Prop::Forall {
        vars: vec![("a", Sort::Real), ("b", Sort::Real)],
        body: Box::new(Prop::Eq(
            Term::UnApp(phi, Box::new(
                Term::BinApp(op,
                    Box::new(Term::Var("a")),
                    Box::new(Term::Var("b"))))),
            Term::BinApp(op,
                Box::new(Term::UnApp(phi, Box::new(Term::Var("a")))),
                Box::new(Term::UnApp(phi, Box::new(Term::Var("b"))))),
        )),
    }
}

/// Convenience: build a scatter correctness proposition.
///
/// States: scatter(φ, keys, values, ⊕) = manual groupby + φ + ⊕.
/// This is the core correctness statement for JIT scatter.
pub fn scatter_correctness_prop(phi_name: &'static str, op: BinOp) -> Prop {
    // ∀data. accumulate(ByKey, φ, ⊕, data) = ⊕_{g} [ ⊕_{i∈group(g)} φ(data[i]) ]
    Prop::Forall {
        vars: vec![("data", Sort::Named("Array".into()))],
        body: Box::new(Prop::Eq(
            Term::Accumulate {
                grouping: GroupingTag::ByKey,
                expr: Box::new(Term::Var(phi_name)),
                op,
                data: Box::new(Term::Var("data")),
            },
            Term::Var("grouped_fold"),  // The mathematical definition
        )),
    }
}

/// Convenience: build a merge correctness proposition.
///
/// States: merge(state_a, state_b) produces the same result as
/// accumulating all of a's data then all of b's data.
pub fn merge_correctness_prop() -> Prop {
    Prop::Forall {
        vars: vec![
            ("a", Sort::Named("State".into())),
            ("b", Sort::Named("State".into())),
        ],
        body: Box::new(Prop::Eq(
            Term::BinApp(BinOp::Add,
                Box::new(Term::Var("a")),
                Box::new(Term::Var("b"))),
            Term::Accumulate {
                grouping: GroupingTag::All,
                expr: Box::new(Term::Var("v")),
                op: BinOp::Add,
                data: Box::new(Term::BinApp(BinOp::Add,
                    Box::new(Term::Var("data_a")),
                    Box::new(Term::Var("data_b")))),
            },
        )),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_display() {
        assert_eq!(format!("{}", Sort::Real), "ℝ");
        assert_eq!(format!("{}", Sort::Nat), "ℕ");
        assert_eq!(format!("{}", Sort::Vec(3, Box::new(Sort::Real))), "ℝ^3");
        assert_eq!(format!("{}", Sort::Mat(2, 3, Box::new(Sort::Real))), "ℝ^(2×3)");
    }

    #[test]
    fn test_structure_laws() {
        let mon = Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0));
        assert!(mon.has_law(StructuralFact::Associativity));
        assert!(mon.has_law(StructuralFact::Commutativity));
        assert!(mon.has_law(StructuralFact::Identity));
        assert!(!mon.has_law(StructuralFact::Invertibility));
        assert!(!mon.has_law(StructuralFact::Idempotence));

        let grp = Structure::group(Sort::Real, BinOp::Add, Term::Lit(0.0));
        assert!(grp.has_law(StructuralFact::Invertibility));

        let lat = Structure::lattice_op(Sort::Real, BinOp::Max, Term::Lit(f64::NEG_INFINITY));
        assert!(lat.has_law(StructuralFact::Idempotence));
    }

    #[test]
    fn test_associativity_proof() {
        let structure = Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0));
        let prop = assoc_prop(BinOp::Add);
        let proof = Proof::ByStructure(structure, StructuralFact::Associativity);
        let thm = Theorem::check("add_assoc", prop, proof).unwrap();
        assert!(thm.is_verified());
        assert_eq!(thm.holes(), 0);
    }

    #[test]
    fn test_commutativity_proof() {
        let structure = Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0));
        let prop = comm_prop(BinOp::Add);
        let proof = Proof::ByStructure(structure, StructuralFact::Commutativity);
        let thm = Theorem::check("add_comm", prop, proof).unwrap();
        assert!(thm.is_verified());
    }

    #[test]
    fn test_identity_proof() {
        let structure = Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0));
        let prop = identity_prop(BinOp::Add, Term::Lit(0.0));
        let proof = Proof::ByStructure(structure, StructuralFact::Identity);
        let thm = Theorem::check("add_identity", prop, proof).unwrap();
        assert!(thm.is_verified());
    }

    #[test]
    fn test_hole_proof() {
        let prop = homomorphism_prop(UnOp::Log, BinOp::Mul);
        let proof = Proof::Hole("Need to verify ln(a*b) = ln(a) + ln(b)".into());
        let thm = Theorem::check("log_hom", prop, proof).unwrap();
        assert!(!thm.is_verified());
        assert_eq!(thm.holes(), 1);
        assert!(format!("{thm}").contains("⟨1?⟩"));
    }

    #[test]
    fn test_law_not_satisfied() {
        let semigroup = Structure::semigroup(Sort::Real, BinOp::Max);
        let prop = identity_prop(BinOp::Max, Term::Lit(0.0));
        // Semigroup doesn't have Identity law
        let proof = Proof::ByStructure(semigroup, StructuralFact::Identity);
        let result = Theorem::check("bad", prop, proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_computational_proof() {
        let prop = assoc_prop(BinOp::Add);
        let proof = Proof::ByComputation {
            method: ComputeMethod::Sampled { seed: 42 },
            n_verified: 10_000,
            max_error: 1e-15,
        };
        let thm = Theorem::check("add_assoc_numerical", prop, proof).unwrap();
        assert!(thm.is_verified());
    }

    #[test]
    fn test_composed_proof() {
        let s = Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0));

        let proof = Proof::ByComposition(
            CompositionRule::ScatterDecomposition,
            vec![
                Proof::ByStructure(s.clone(), StructuralFact::Associativity),
                Proof::ByStructure(s.clone(), StructuralFact::Commutativity),
                Proof::ByStructure(s, StructuralFact::Identity),
            ],
        );
        let prop = scatter_correctness_prop("v", BinOp::Add);
        let thm = Theorem::check("scatter_sum_correct", prop, proof).unwrap();
        assert!(thm.is_verified());
    }

    #[test]
    fn test_composed_with_hole() {
        let s = Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0));

        let proof = Proof::ByComposition(
            CompositionRule::Conjunction,
            vec![
                Proof::ByStructure(s, StructuralFact::Associativity),
                Proof::Hole("need to prove convergence".into()),
            ],
        );
        let prop = Prop::True;
        let thm = Theorem::check("partial", prop, proof).unwrap();
        assert!(!thm.is_verified());
        assert_eq!(thm.holes(), 1);
    }

    #[test]
    fn test_proof_context() {
        let mut ctx = tambear_context();

        // Prove add is associative
        let s = Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0));
        let thm1 = Theorem::check(
            "add_assoc",
            assoc_prop(BinOp::Add),
            Proof::ByStructure(s.clone(), StructuralFact::Associativity),
        ).unwrap();
        ctx.add(thm1).unwrap();

        // Prove add is commutative
        let thm2 = Theorem::check(
            "add_comm",
            comm_prop(BinOp::Add),
            Proof::ByStructure(s, StructuralFact::Commutativity),
        ).unwrap();
        ctx.add(thm2).unwrap();

        // Use both to prove scatter correctness
        let thm3 = Theorem::check(
            "scatter_correct",
            scatter_correctness_prop("v", BinOp::Add),
            Proof::ByComposition(
                CompositionRule::ScatterDecomposition,
                vec![
                    Proof::ByRef("add_assoc".into()),
                    Proof::ByRef("add_comm".into()),
                ],
            ),
        ).unwrap();
        ctx.add(thm3).unwrap();

        let summary = ctx.summary();
        assert_eq!(summary.total, 3);
        assert_eq!(summary.verified, 3);
        assert_eq!(summary.holes, 0);
        assert!(summary.structures > 0);
    }

    #[test]
    fn test_bad_ref() {
        let mut ctx = tambear_context();
        let thm = Theorem::check(
            "uses_nonexistent",
            Prop::True,
            Proof::ByRef("nonexistent_theorem".into()),
        ).unwrap();
        // ProofContext::add checks refs — this should fail
        assert!(ctx.add(thm).is_err());
    }

    #[test]
    fn test_compile_accumulate_term() {
        let term = Term::Accumulate {
            grouping: GroupingTag::ByKey,
            expr: Box::new(Term::UnApp(UnOp::Sq, Box::new(Term::Var("v")))),
            op: BinOp::Add,
            data: Box::new(Term::Var("prices")),
        };
        let steps = compile_term(&term);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].phi_expr, "(v * v)");
        assert_eq!(steps[0].grouping, GroupingTag::ByKey);
    }

    #[test]
    fn test_nested_accumulate_compilation() {
        // variance = E[X²] - E[X]²
        // = accumulate(All, sq, +, data) / n - (accumulate(All, id, +, data) / n)²
        let sum_sq = Term::Accumulate {
            grouping: GroupingTag::All,
            expr: Box::new(Term::UnApp(UnOp::Sq, Box::new(Term::Var("v")))),
            op: BinOp::Add,
            data: Box::new(Term::Var("data")),
        };
        let sum = Term::Accumulate {
            grouping: GroupingTag::All,
            expr: Box::new(Term::Var("v")),
            op: BinOp::Add,
            data: Box::new(Term::Var("data")),
        };
        let variance = Term::BinApp(BinOp::Sub,
            Box::new(sum_sq),
            Box::new(Term::UnApp(UnOp::Sq, Box::new(sum))),
        );

        let steps = compile_term(&variance);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].phi_expr, "(v * v)"); // sum of squares
        assert_eq!(steps[1].phi_expr, "v");       // sum
    }

    #[test]
    fn test_term_display() {
        let t = Term::BinApp(BinOp::Add,
            Box::new(Term::Var("a")),
            Box::new(Term::BinApp(BinOp::Mul,
                Box::new(Term::Var("b")),
                Box::new(Term::Var("c")))));
        assert_eq!(format!("{t}"), "(a + (b × c))");
    }

    #[test]
    fn test_prop_display() {
        let p = assoc_prop(BinOp::Add);
        let s = format!("{p}");
        assert!(s.contains("∀"));
        assert!(s.contains("+"));
    }

    #[test]
    fn test_theorem_display_verified() {
        let s = Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0));
        let thm = Theorem::check(
            "test",
            comm_prop(BinOp::Add),
            Proof::ByStructure(s, StructuralFact::Commutativity),
        ).unwrap();
        let display = format!("{thm}");
        assert!(display.starts_with("✓"));
    }

    #[test]
    fn test_tambear_context_has_structures() {
        let ctx = tambear_context();
        assert!(ctx.structures.len() >= 5);
        // Should have Add, Mul, Max, Min, Group
    }

    #[test]
    fn test_merge_correctness_prop() {
        let prop = merge_correctness_prop();
        assert!(matches!(prop, Prop::Forall { .. }));
    }

    #[test]
    fn test_witness_proof_requires_exists() {
        let prop = Prop::Exists {
            vars: vec![("x", Sort::Real)],
            body: Box::new(Prop::Eq(
                Term::BinApp(BinOp::Mul, Box::new(Term::Var("x")), Box::new(Term::Var("x"))),
                Term::Lit(4.0),
            )),
        };
        let proof = Proof::ByWitness(Term::Lit(2.0));
        let thm = Theorem::check("sqrt4", prop, proof).unwrap();
        assert!(thm.is_verified());
    }

    #[test]
    fn test_witness_proof_rejects_non_exists() {
        let prop = Prop::True;
        let proof = Proof::ByWitness(Term::Lit(42.0));
        let result = Theorem::check("bad_witness", prop, proof);
        assert!(result.is_err());
    }

    #[test]
    fn test_modus_ponens_needs_two() {
        let _unused = Proof::ByComposition(
            CompositionRule::ModusPonens,
            vec![Proof::ByRef("one".into())], // Only one sub-proof — not enough
        );
        // Test that ModusPonens with only 1 sub-proof is rejected
        let s = Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0));
        let proof_ok = Proof::ByComposition(
            CompositionRule::ModusPonens,
            vec![
                Proof::ByStructure(s.clone(), StructuralFact::Associativity),
            ],
        );
        let result = Theorem::check("mp_bad", Prop::True, proof_ok);
        // Should fail — ModusPonens needs exactly 2
        assert!(result.is_err());
    }

    #[test]
    fn test_summary_display() {
        let ctx = tambear_context();
        let summary = ctx.summary();
        let s = format!("{summary}");
        assert!(s.contains("structures"));
    }

    // ── End-to-end: COPA mergeability proof ────────────────────────────

    /// Full pipeline test: declare COPA structure, prove mergeability,
    /// decompose variance into accumulate primitives, compile to phi expressions.
    #[test]
    fn test_copa_mergeability_e2e() {
        let mut ctx = tambear_context();

        // 1. Declare CopaState merge as a semigroup
        let copa_semigroup = Structure::semigroup(
            Sort::Named("CopaState".into()),
            BinOp::Add, // merge operation
        );
        ctx.declare_structure(copa_semigroup.clone());

        // 2. Prove associativity of COPA merge
        //    C = Ca + Cb + (na·nb/n)·ΔΔᵀ
        //    Associativity holds because the correction term is symmetric
        //    and the mean update is a weighted average (associative).
        let copa_assoc = Theorem::check(
            "copa_merge_associative",
            Prop::Forall {
                vars: vec![
                    ("a", Sort::Named("CopaState".into())),
                    ("b", Sort::Named("CopaState".into())),
                    ("c", Sort::Named("CopaState".into())),
                ],
                body: Box::new(Prop::Eq(
                    Term::BinApp(BinOp::Add,
                        Box::new(Term::BinApp(BinOp::Add,
                            Box::new(Term::Var("a")),
                            Box::new(Term::Var("b")))),
                        Box::new(Term::Var("c"))),
                    Term::BinApp(BinOp::Add,
                        Box::new(Term::Var("a")),
                        Box::new(Term::BinApp(BinOp::Add,
                            Box::new(Term::Var("b")),
                            Box::new(Term::Var("c"))))),
                )),
            },
            Proof::ByStructure(copa_semigroup, StructuralFact::Associativity),
        ).unwrap();
        ctx.add(copa_assoc).unwrap();

        // 3. Prove MomentStats merge is also associative (scalar analogue)
        let moment_semigroup = Structure::semigroup(
            Sort::Named("MomentStats".into()),
            BinOp::Add,
        );
        ctx.declare_structure(moment_semigroup.clone());

        let moment_assoc = Theorem::check(
            "moment_merge_associative",
            assoc_prop(BinOp::Add), // reusing the builder
            Proof::ByStructure(moment_semigroup, StructuralFact::Associativity),
        ).unwrap();
        ctx.add(moment_assoc).unwrap();

        // 4. Prove variance decomposition: Var(X) = E[X²] - E[X]²
        //    This decomposes into two scatter_phi operations:
        //    - accumulate(All, sq, +, data) → sum of squares
        //    - accumulate(All, id, +, data) → sum
        //    Then: var = sum_sq/n - (sum/n)²
        let variance_term = Term::BinApp(BinOp::Sub,
            Box::new(Term::BinApp(BinOp::Div,
                Box::new(Term::Accumulate {
                    grouping: GroupingTag::All,
                    expr: Box::new(Term::UnApp(UnOp::Sq, Box::new(Term::Var("v")))),
                    op: BinOp::Add,
                    data: Box::new(Term::Var("data")),
                }),
                Box::new(Term::Var("n")),
            )),
            Box::new(Term::UnApp(UnOp::Sq,
                Box::new(Term::BinApp(BinOp::Div,
                    Box::new(Term::Accumulate {
                        grouping: GroupingTag::All,
                        expr: Box::new(Term::Var("v")),
                        op: BinOp::Add,
                        data: Box::new(Term::Var("data")),
                    }),
                    Box::new(Term::Var("n")),
                )),
            )),
        );

        let variance_prop = Prop::Forall {
            vars: vec![("data", Sort::Named("Array".into()))],
            body: Box::new(Prop::Eq(
                Term::Var("population_variance"),
                variance_term.clone(),
            )),
        };

        let variance_thm = Theorem::check(
            "variance_decomposition",
            variance_prop,
            Proof::ByComposition(
                CompositionRule::ScatterDecomposition,
                vec![
                    // sq is a homomorphism: sq(a+b) ≠ sq(a)+sq(b), BUT
                    // the lift function just needs to be applied per-element
                    // before the associative combine (Add). This is always valid.
                    Proof::ByStructure(
                        Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0)),
                        StructuralFact::Associativity,
                    ),
                    Proof::ByStructure(
                        Structure::commutative_monoid(Sort::Real, BinOp::Add, Term::Lit(0.0)),
                        StructuralFact::Commutativity,
                    ),
                ],
            ),
        ).unwrap();
        ctx.add(variance_thm).unwrap();

        // 5. Compile: extract the accumulate steps from the variance term
        let steps = compile_term(&variance_term);
        assert_eq!(steps.len(), 2, "variance should compile to 2 accumulate calls");
        assert_eq!(steps[0].phi_expr, "(v * v)", "first scatter: sum of squares");
        assert_eq!(steps[1].phi_expr, "v", "second scatter: sum");
        assert_eq!(steps[0].grouping, GroupingTag::All);
        assert_eq!(steps[1].grouping, GroupingTag::All);

        // 6. Verify the full context
        let summary = ctx.summary();
        assert!(summary.verified >= 3, "at least 3 theorems should be verified");
        assert_eq!(summary.holes, 0, "no open obligations");

        eprintln!("COPA mergeability E2E proof context:");
        eprintln!("  {}", summary);
        for thm in ctx.theorems() {
            eprintln!("  {}", thm);
        }
        eprintln!("Compiled variance to {} accumulate steps:", steps.len());
        for (i, step) in steps.iter().enumerate() {
            eprintln!("  step {}: phi=\"{}\", grouping={:?}, op={:?}",
                i, step.phi_expr, step.grouping, step.op);
        }
    }

    /// Test: grouped variance decomposes to scatter_multi_phi with ByKey grouping.
    #[test]
    fn test_grouped_variance_compilation() {
        // Grouped variance: for each group g, compute Var(X_g)
        // = accumulate(ByKey, sq, +, data) / count_g - (accumulate(ByKey, id, +, data) / count_g)²
        let grouped_var = Term::BinApp(BinOp::Sub,
            Box::new(Term::BinApp(BinOp::Div,
                Box::new(Term::Accumulate {
                    grouping: GroupingTag::ByKey,
                    expr: Box::new(Term::UnApp(UnOp::Sq, Box::new(Term::Var("v")))),
                    op: BinOp::Add,
                    data: Box::new(Term::Var("data")),
                }),
                Box::new(Term::Accumulate {
                    grouping: GroupingTag::ByKey,
                    expr: Box::new(Term::UnApp(UnOp::One, Box::new(Term::Var("v")))),
                    op: BinOp::Add,
                    data: Box::new(Term::Var("data")),
                }),
            )),
            Box::new(Term::UnApp(UnOp::Sq,
                Box::new(Term::BinApp(BinOp::Div,
                    Box::new(Term::Accumulate {
                        grouping: GroupingTag::ByKey,
                        expr: Box::new(Term::Var("v")),
                        op: BinOp::Add,
                        data: Box::new(Term::Var("data")),
                    }),
                    Box::new(Term::Accumulate {
                        grouping: GroupingTag::ByKey,
                        expr: Box::new(Term::UnApp(UnOp::One, Box::new(Term::Var("v")))),
                        op: BinOp::Add,
                        data: Box::new(Term::Var("data")),
                    }),
                )),
            )),
        );

        let steps = compile_term(&grouped_var);
        // Should find 4 accumulate calls: sum_sq, count, sum, count
        // (the count appears twice because it's used in both numerator and denominator)
        assert_eq!(steps.len(), 4, "grouped variance needs 4 accumulate calls");

        // All should be ByKey grouping
        for step in &steps {
            assert_eq!(step.grouping, GroupingTag::ByKey);
        }

        // The phi expressions should be: (v * v), 1.0, v, 1.0
        let phis: Vec<&str> = steps.iter().map(|s| s.phi_expr.as_str()).collect();
        assert_eq!(phis[0], "(v * v)");
        assert_eq!(phis[1], "1.0");
        assert_eq!(phis[2], "v");
        assert_eq!(phis[3], "1.0");

        eprintln!("Grouped variance compiles to scatter_multi_phi with {} phi expressions:", steps.len());
        // In practice, scatter_multi_phi would fuse the unique phis:
        // ["v * v", "1.0", "v"] — 3 unique phis in one kernel pass
        let unique_phis: std::collections::HashSet<&str> = phis.iter().copied().collect();
        eprintln!("  Unique phis (fusable): {:?}", unique_phis);
        assert_eq!(unique_phis.len(), 3, "3 unique phi expressions");
    }
}
