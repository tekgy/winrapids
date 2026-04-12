//! Metadata for Grouping and Op atoms.
//!
//! Same philosophy as Expr metadata: every atom is self-describing.
//! TAM, the proof engine, the optimizer, and the IDE all read this.

use super::{Grouping, Op};
use crate::tbs::metadata::{Property, NanBehavior, Cost};
use crate::tbs::shape::ShapeSig;

/// Complete metadata for a Grouping variant.
#[derive(Debug, Clone)]
pub struct GroupingMeta {
    pub syntax: &'static str,
    pub latex: &'static str,
    pub tambear: &'static str,
    pub description: &'static str,
    /// Input → output shape signature.
    pub shape: ShapeSig,
    /// Can slots with this grouping fuse with each other?
    pub fusable: bool,
    /// Does this grouping require sorted input?
    pub requires_sorted: bool,
    /// Does this grouping require keys/indices?
    pub requires_keys: bool,
    /// Parallelism model: how many threads work on this?
    pub parallelism: &'static str,
    /// TAM scheduling: any sequential dependency?
    pub sequential_dependency: bool,
}

/// Complete metadata for an Op variant.
#[derive(Debug, Clone)]
pub struct OpMeta {
    pub syntax: &'static str,
    pub latex: &'static str,
    pub tambear: &'static str,
    pub description: &'static str,
    /// Algebraic properties.
    pub properties: &'static [Property],
    /// Identity element: op(identity, x) = x.
    pub identity: f64,
    /// Absorbing element: op(absorbing, x) = absorbing. NaN if none.
    pub absorbing: f64,
    /// How NaN is handled in the reduction.
    pub nan_behavior: NanBehavior,
    /// Computational cost per combine step.
    pub cost: Cost,
    /// Does the proof engine need to verify associativity for parallel scan?
    pub requires_associativity_proof: bool,
    /// Tam instruction for the combine step.
    pub tam_combine: &'static str,
}

pub fn grouping_meta(g: &Grouping) -> GroupingMeta {
    match g {
        Grouping::All => GroupingMeta {
            syntax: "All", latex: r"\bigoplus_{i=1}^{n}", tambear: "⟨ · | All ⟩",
            description: "N → 1: all elements into one accumulator",
            shape: ShapeSig::vector_to_scalar(),
            fusable: true, requires_sorted: false, requires_keys: false,
            parallelism: "tree reduction: O(log n) depth, O(n) work",
            sequential_dependency: false,
        },
        Grouping::ByKey => GroupingMeta {
            syntax: "ByKey", latex: r"\bigoplus_{i \in G_k}", tambear: "⟨ · | G_k ⟩",
            description: "N → K: scatter by key into K accumulators",
            shape: ShapeSig::vector_to_groups(),
            fusable: true, requires_sorted: false, requires_keys: true,
            parallelism: "parallel scatter: O(1) depth, O(n) work",
            sequential_dependency: false,
        },
        Grouping::Prefix => GroupingMeta {
            syntax: "Prefix", latex: r"\bigoplus_{j=1}^{i}", tambear: "⟨ · | ← ⟩",
            description: "N → N: inclusive prefix scan (cumsum, etc)",
            shape: ShapeSig::vector_to_vector(),
            fusable: false, requires_sorted: false, requires_keys: false,
            parallelism: "Blelloch: O(log n) depth, O(n) work",
            sequential_dependency: true, // each output depends on all prior
        },
        Grouping::Segmented => GroupingMeta {
            syntax: "Segmented", latex: r"\bigoplus_{j=s_k}^{i}", tambear: "⟨ · | ←| ⟩",
            description: "N → N: prefix scan with resets at segment boundaries",
            shape: ShapeSig::vector_to_vector(),
            fusable: false, requires_sorted: false, requires_keys: true,
            parallelism: "segmented Blelloch: O(log n) depth",
            sequential_dependency: true,
        },
        Grouping::Windowed => GroupingMeta {
            syntax: "Windowed", latex: r"\bigoplus_{j=i-w}^{i}", tambear: "⟨ · | ↔w ⟩",
            description: "N → N: rolling window via prefix subtraction",
            shape: ShapeSig::vector_to_vector(),
            fusable: false, requires_sorted: false, requires_keys: false,
            parallelism: "prefix + subtraction: O(log n) depth, O(n) work",
            sequential_dependency: true,
        },
        Grouping::Tiled => GroupingMeta {
            syntax: "Tiled", latex: r"\sum_{k} A_{ik} B_{kj}", tambear: "⟨ · | ▦ ⟩",
            description: "M×K × K×N → M×N: blocked matrix accumulation",
            shape: ShapeSig::matmul(),
            fusable: true, requires_sorted: false, requires_keys: false,
            parallelism: "tiled: O(K/tile) depth, O(MNK) work",
            sequential_dependency: false,
        },
        Grouping::Graph => GroupingMeta {
            syntax: "Graph", latex: r"\bigoplus_{j \in N(i)}", tambear: "⟨ · | G ⟩",
            description: "N → N: scatter by graph adjacency",
            shape: ShapeSig::vector_to_vector(),
            fusable: false, requires_sorted: false, requires_keys: true,
            parallelism: "neighbor gather: O(max_degree) depth",
            sequential_dependency: false,
        },
    }
}

pub fn op_meta(op: &Op) -> OpMeta {
    match op {
        Op::Add => OpMeta {
            syntax: "+", latex: "+", tambear: "+",
            description: "Real addition",
            properties: &[Property::Associative, Property::Commutative],
            identity: 0.0,
            absorbing: f64::NAN, // no absorbing element
            nan_behavior: NanBehavior::Propagate,
            cost: Cost(1),
            requires_associativity_proof: true,
            tam_combine: "fadd",
        },
        Op::Max => OpMeta {
            syntax: "max", latex: r"\max", tambear: "∨",
            description: "Maximum",
            properties: &[Property::Associative, Property::Commutative, Property::Idempotent],
            identity: f64::NEG_INFINITY,
            absorbing: f64::INFINITY,
            nan_behavior: NanBehavior::Propagate,
            cost: Cost(1),
            requires_associativity_proof: true,
            tam_combine: "fmax",
        },
        Op::Min => OpMeta {
            syntax: "min", latex: r"\min", tambear: "∧",
            description: "Minimum",
            properties: &[Property::Associative, Property::Commutative, Property::Idempotent],
            identity: f64::INFINITY,
            absorbing: f64::NEG_INFINITY,
            nan_behavior: NanBehavior::Propagate,
            cost: Cost(1),
            requires_associativity_proof: true,
            tam_combine: "fmin",
        },
        Op::Mul => OpMeta {
            syntax: "*", latex: r"\times", tambear: "×",
            description: "Real multiplication",
            properties: &[Property::Associative, Property::Commutative],
            identity: 1.0,
            absorbing: 0.0,
            nan_behavior: NanBehavior::Propagate,
            cost: Cost(2),
            requires_associativity_proof: true,
            tam_combine: "fmul",
        },
        Op::And => OpMeta {
            syntax: "&&", latex: r"\land", tambear: "∧",
            description: "Logical AND (as 0/1 floats)",
            properties: &[Property::Associative, Property::Commutative, Property::Idempotent],
            identity: 1.0, // true AND x = x
            absorbing: 0.0, // false AND x = false
            nan_behavior: NanBehavior::Propagate,
            cost: Cost(1),
            requires_associativity_proof: false, // trivially associative
            tam_combine: "fmul", // 1*1=1, 0*anything=0
        },
        Op::Or => OpMeta {
            syntax: "||", latex: r"\lor", tambear: "∨",
            description: "Logical OR (as 0/1 floats)",
            properties: &[Property::Associative, Property::Commutative, Property::Idempotent],
            identity: 0.0, // false OR x = x
            absorbing: 1.0, // true OR x = true
            nan_behavior: NanBehavior::Propagate,
            cost: Cost(1),
            requires_associativity_proof: false,
            tam_combine: "fmax", // max(0,1)=1, max(0,0)=0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_grouping_has_scalar_output() {
        let meta = grouping_meta(&Grouping::All);
        assert_eq!(meta.shape.output, crate::tbs::shape::Shape::Scalar);
        assert!(meta.fusable);
        assert!(!meta.sequential_dependency);
    }

    #[test]
    fn prefix_has_sequential_dependency() {
        let meta = grouping_meta(&Grouping::Prefix);
        assert!(meta.sequential_dependency);
        assert!(!meta.fusable); // prefix scans don't fuse with each other
    }

    #[test]
    fn add_is_associative_commutative() {
        let meta = op_meta(&Op::Add);
        assert!(meta.properties.contains(&Property::Associative));
        assert!(meta.properties.contains(&Property::Commutative));
        assert_eq!(meta.identity, 0.0);
    }

    #[test]
    fn max_identity_is_neg_infinity() {
        let meta = op_meta(&Op::Max);
        assert_eq!(meta.identity, f64::NEG_INFINITY);
        assert_eq!(meta.absorbing, f64::INFINITY);
    }

    #[test]
    fn mul_absorbing_is_zero() {
        let meta = op_meta(&Op::Mul);
        assert_eq!(meta.identity, 1.0);
        assert_eq!(meta.absorbing, 0.0);
    }

    #[test]
    fn and_or_use_float_encoding() {
        let and_meta = op_meta(&Op::And);
        let or_meta = op_meta(&Op::Or);
        assert_eq!(and_meta.tam_combine, "fmul"); // 1*1=1, 0*x=0
        assert_eq!(or_meta.tam_combine, "fmax"); // max(1,x)=1, max(0,0)=0
    }

    #[test]
    fn tiled_is_matmul_shape() {
        let meta = grouping_meta(&Grouping::Tiled);
        assert!(meta.fusable);
        assert!(!meta.sequential_dependency);
    }

    #[test]
    fn tambear_notation_exists_for_all() {
        for g in &[Grouping::All, Grouping::ByKey, Grouping::Prefix,
                   Grouping::Segmented, Grouping::Windowed, Grouping::Tiled, Grouping::Graph] {
            let meta = grouping_meta(g);
            assert!(!meta.tambear.is_empty(), "missing tambear notation for {:?}", g);
        }
        for op in &[Op::Add, Op::Max, Op::Min, Op::Mul, Op::And, Op::Or] {
            let meta = op_meta(op);
            assert!(!meta.tambear.is_empty(), "missing tambear notation for {:?}", op);
        }
    }
}
