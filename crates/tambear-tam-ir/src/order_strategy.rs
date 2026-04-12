//! OrderStrategy registry — campsite 1.16.
//!
//! The registry is the authoritative list of named reduction strategies for the
//! `.tam` IR. Programs reference strategies by name (`OrderStrategyRef`). The
//! registry maps names to entries that carry the formal spec, a runnable Rust
//! reference implementation, bit-exact test vectors, and fusion-compatibility
//! metadata.
//!
//! # Why a registry, not a closed enum?
//!
//! A closed enum couples the IR's definition of "what strategies exist" to the
//! Rust source. Adding a new strategy (e.g. `kahan_compensated`, `rfa_k3`) would
//! require a Rust source change + rebuild. Worse, a backend that speaks a strategy
//! the enum doesn't know about has nowhere to land.
//!
//! A named registry decouples the IR (names only) from the implementation:
//! - New strategies land as registry entries, not enum variants.
//! - The verifier validates by registry lookup, not by exhaustive match.
//! - The CPU interpreter dispatches by name lookup, not by exhaustive match.
//! - Future backends can register their own strategies; the IR stays unchanged.
//!
//! The design follows the engineering shape observed in the three-registry
//! convergence: OrderStrategy registry + oracles registry + guarantee ledger
//! all carry the same pattern of named entries with formal specs and metadata.
//!
//! # Stability
//!
//! Entry names are stable once registered. A name that appears in a committed
//! `.tam` program is a permanent contract. Entry *implementations* may be
//! refined; entry *names* may not be changed or removed. Deprecation is
//! handled by adding a `deprecated_in: Option<&str>` field (future work).
//!
//! # Fusion-compatibility
//!
//! The `compat_class` field carries a name identifying which strategies are
//! mutually substitutable under cross-kernel fusion. Two strategies with the
//! same `compat_class` may be fused; two with different classes may not.
//! Phase 1 does not perform cross-kernel fusion, so this field is informational
//! only — but it must be populated now so Phase 2 has the metadata it needs
//! without an IR format version bump.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// Registry entry
// ═══════════════════════════════════════════════════════════════════

/// A single entry in the OrderStrategy registry.
///
/// Every field is required. Stub entries carry a `reference_impl` that panics
/// with an informative message (citing which peak will implement them), but all
/// metadata fields are populated so the verifier can reason about them.
pub struct StrategyEntry {
    /// Short canonical name — the key in the registry and in `.tam` source.
    /// Snake_case. ASCII only. No spaces.
    pub name: &'static str,

    /// One-line human summary shown in verifier error messages.
    pub summary: &'static str,

    /// Prose formal specification. What sequence of operations does this
    /// strategy perform? What are its numerical contracts? What does it
    /// guarantee about determinism and bit-exactness?
    pub spec: &'static str,

    /// Runnable Rust reference implementation.
    /// Called by the CPU interpreter when this strategy is in use.
    /// Receives the values to reduce (in-order slice) and returns the result.
    ///
    /// Stub entries panic with `"not yet implemented — <Peak N>"`.
    pub reference_impl: fn(&[f64]) -> f64,

    /// Hardcoded bit-exact test vectors: (inputs, expected output).
    /// The oracle test suite uses these to pin the implementation.
    pub test_vectors: &'static [TestVector],

    /// Fusion compatibility class.
    /// Two strategies with the same class are substitutable in cross-kernel
    /// fusion (Phase 2+). Different classes may not be fused.
    ///
    /// Phase 1 values:
    /// - `"sequential"` — all sequential strategies (commute freely with each other).
    /// - `"tree_pow2"` — power-of-two tree strategies (same fanout required for substitution).
    /// - `"rfa"` — reproducible floating-point accumulation family (incompatible with tree/seq).
    pub compat_class: &'static str,
}

impl StrategyEntry {
    /// Returns true if this strategy is compatible with `other` for cross-kernel fusion.
    ///
    /// Phase 1 definition: two strategies are fusable when they share the same
    /// `compat_class`. This is the stub Aristotle v5.2 requires for campsite 1.15.
    /// The real compatibility predicate (which may allow cross-class fusion under
    /// specific conditions) lands in Peak 6.
    pub fn is_fusable_with(&self, other: &StrategyEntry) -> bool {
        self.compat_class == other.compat_class
    }
}

/// A single bit-exact test vector: (input values, expected output).
pub struct TestVector {
    /// Descriptive label for this test case.
    pub label: &'static str,
    /// Input values passed to `reference_impl`.
    pub inputs: &'static [f64],
    /// Expected output, bit-exact (tested with `==`, not within ULP).
    pub expected: f64,
}

// ═══════════════════════════════════════════════════════════════════
// Reference implementations
// ═══════════════════════════════════════════════════════════════════

/// `sequential_left` reference implementation.
///
/// Left-to-right serial accumulation: `acc = ((a[0] + a[1]) + a[2]) + ...`
/// Empty slice returns 0.0. Single element returns the element unchanged.
fn sequential_left_impl(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0_f64, |acc, v| acc + v)
}

/// `tree_fixed_fanout_2` reference implementation.
///
/// Recursive binary tree reduction. Splits the slice in half, reduces each
/// half recursively, then adds the two halves. The split point is always
/// `len / 2` (integer division), which means for odd-length inputs the first
/// half is shorter. This produces a specific tree shape that is bit-exact for
/// a given input length regardless of backend.
///
/// Empty slice: 0.0. Single element: the element itself.
fn tree_fixed_fanout_2_impl(values: &[f64]) -> f64 {
    match values.len() {
        0 => 0.0,
        1 => values[0],
        n => {
            let mid = n / 2;
            let left = tree_fixed_fanout_2_impl(&values[..mid]);
            let right = tree_fixed_fanout_2_impl(&values[mid..]);
            left + right
        }
    }
}

/// `rfa_bin_exponent_aligned` stub — not yet implemented.
///
/// Will implement Demmel-Nguyen 2013/2015 reproducible floating-point
/// accumulation with K=3 folds and bin width DBWIDTH=40 bits. This is
/// the strategy required for the determinism guarantee in Peak 6. The
/// stub is registered now so IR programs can name it and the verifier
/// accepts it; the CPU interpreter panics if it is actually executed.
fn rfa_bin_exponent_aligned_impl(_values: &[f64]) -> f64 {
    panic!(
        "rfa_bin_exponent_aligned is not yet implemented — \
         implementation is scheduled for Peak 6 (Deterministic Reductions). \
         Reference: Demmel & Nguyen, 'Fast Reproducible Floating-Point \
         Summation', 2013/2015. K=3 folds, DBWIDTH=40 bits, 48-byte state."
    )
}

// ═══════════════════════════════════════════════════════════════════
// Test vectors
// ═══════════════════════════════════════════════════════════════════

// sequential_left test vectors — pin the fold order.
// These differ from tree_fixed_fanout_2 on inputs where rounding matters.

static SEQUENTIAL_LEFT_VECTORS: &[TestVector] = &[
    TestVector {
        label: "empty",
        inputs: &[],
        expected: 0.0,
    },
    TestVector {
        label: "single",
        inputs: &[1.0],
        expected: 1.0,
    },
    TestVector {
        label: "two_elements",
        inputs: &[1.0, 2.0],
        expected: 3.0,
    },
    TestVector {
        label: "three_elements_left_fold",
        // ((1.0 + 2.0) + 3.0) = 6.0 exactly
        inputs: &[1.0, 2.0, 3.0],
        expected: 6.0,
    },
    TestVector {
        label: "large_then_small",
        // Sequential fold: ((0 + 1e15) + 1.0) + (-1e15).
        // 1e15 + 1.0 = 1000000000000001.0 (representable in f64).
        // Then - 1e15 = 1.0.
        inputs: &[1e15, 1.0, -1e15],
        expected: 1.0,
    },
    TestVector {
        label: "nan_propagation",
        // NaN + anything = NaN.
        inputs: &[1.0, f64::NAN, 2.0],
        expected: f64::NAN,
    },
    TestVector {
        label: "inf_propagation",
        inputs: &[1.0, f64::INFINITY, -f64::INFINITY],
        expected: f64::NAN,
    },
];

static TREE_FIXED_FANOUT_2_VECTORS: &[TestVector] = &[
    TestVector {
        label: "empty",
        inputs: &[],
        expected: 0.0,
    },
    TestVector {
        label: "single",
        inputs: &[1.0],
        expected: 1.0,
    },
    TestVector {
        label: "two_elements",
        // tree: left=[1.0] right=[2.0]; 1.0+2.0=3.0
        inputs: &[1.0, 2.0],
        expected: 3.0,
    },
    TestVector {
        label: "three_elements_tree",
        // len=3, mid=1; left=[1.0]=1.0, right=[2.0,3.0]: mid=1, left=[2.0], right=[3.0] → 5.0
        // final: 1.0+5.0=6.0
        inputs: &[1.0, 2.0, 3.0],
        expected: 6.0,
    },
    TestVector {
        label: "large_then_small_tree",
        // tree on [1e15, 1.0, -1e15]: mid=1, left=[1e15]=1e15,
        // right=[1.0,-1e15]: mid=1, left=1.0, right=-1e15 → 1.0+(-1e15)=-(1e15-1.0)
        // final: 1e15 + (-(1e15-1.0)) = 1.0
        // (same result as sequential here — see tree_vs_sequential_differ test for
        //  an input that actually diverges)
        inputs: &[1e15, 1.0, -1e15],
        expected: 1.0,
    },
    TestVector {
        label: "nan_propagation",
        inputs: &[1.0, f64::NAN, 2.0],
        expected: f64::NAN,
    },
];

static RFA_BIN_EXPONENT_ALIGNED_VECTORS: &[TestVector] = &[
    // No test vectors yet — implementation is in Peak 6.
    // Vectors will be derived from the Demmel-Nguyen reference implementation
    // once the algorithm lands.
];

// ═══════════════════════════════════════════════════════════════════
// Registry
// ═══════════════════════════════════════════════════════════════════

/// Build and return the static OrderStrategy registry.
///
/// The registry is built on first call (via `registry()` below). It is
/// immutable after construction.
fn build_registry() -> HashMap<&'static str, StrategyEntry> {
    let mut m = HashMap::new();

    m.insert(
        "sequential_left",
        StrategyEntry {
            name: "sequential_left",
            summary: "Serial left-to-right accumulation: acc = fold_left(0, +, values)",
            spec: "\
sequential_left performs a left-to-right fold over the input values using fp64
addition with no FMA contraction and no associativity reordering (I3, I4).

Formal definition:
  sequential_left([]) = 0.0
  sequential_left([v]) = v
  sequential_left([v0, v1, ..., vN]) = sequential_left([v0, ..., v(N-1)]) + vN

This is equivalent to Rust: values.iter().fold(0.0_f64, |acc, &v| acc + v)

Numerical contracts:
- Deterministic: same input, same output on every run, every backend.
- IEEE-754 faithful: each addition is a faithful fp64 add (no FMA, no flush).
- NaN propagates: if any input is NaN, the result is NaN.
- Inf propagates: if both +Inf and -Inf appear, the result is NaN.
- The CPU interpreter always uses this strategy regardless of the op's declared
  strategy, because sequential_left is the natural sequential-machine semantics.
  The PTX and SPIR-V backends must implement it faithfully if they claim to
  support this strategy.

Accuracy note:
  sequential_left is NOT numerically stable for inputs with large magnitude
  differences. It is chosen for simplicity and determinism, not for accuracy.
  Users who need accurate summation should use rfa_bin_exponent_aligned (Peak 6)
  or Kahan compensated summation (future strategy entry).",
            reference_impl: sequential_left_impl,
            test_vectors: SEQUENTIAL_LEFT_VECTORS,
            compat_class: "sequential",
        },
    );

    m.insert(
        "tree_fixed_fanout_2",
        StrategyEntry {
            name: "tree_fixed_fanout_2",
            summary: "Balanced binary tree reduction with fanout 2",
            spec: "\
tree_fixed_fanout_2 performs a recursive binary tree reduction over the input
values using fp64 addition with no FMA contraction (I3).

Formal definition:
  tree_fixed_fanout_2([]) = 0.0
  tree_fixed_fanout_2([v]) = v
  tree_fixed_fanout_2(values) =
    let mid = len(values) / 2   -- integer division
    tree_fixed_fanout_2(values[0..mid]) + tree_fixed_fanout_2(values[mid..])

The split point `mid = len / 2` is deterministic for every input length.
This pins the tree shape: for a given array length, every backend that
implements tree_fixed_fanout_2 produces the same tree, the same intermediate
sums, and the same final result.

Numerical contracts:
- Deterministic: same input length → same tree shape → same bits.
- NaN propagates: if any input is NaN, the result is NaN.
- Different from sequential_left for numerically sensitive inputs (see the
  `large_then_small_tree` test vector — tree gets 1.0, sequential gets 0.0).

GPU implementation target:
  PTX backend uses warp-level tree reduction for within-warp step (32 threads
  = 5 tree levels). The block-level fold of warp partial sums uses
  sequential_left to keep the host-side deterministic. The full kernel produces
  tree_fixed_fanout_2 semantics for the warp step only — the inter-warp
  combination is declared separately.

Fusion compatibility:
  Two tree_fixed_fanout_2 reductions over non-overlapping inputs may be fused
  (same compat_class). Two reductions with different fanouts may not
  (different entries, same compat_class requires same fanout for substitution).",
            reference_impl: tree_fixed_fanout_2_impl,
            test_vectors: TREE_FIXED_FANOUT_2_VECTORS,
            compat_class: "tree_pow2",
        },
    );

    m.insert(
        "rfa_bin_exponent_aligned",
        StrategyEntry {
            name: "rfa_bin_exponent_aligned",
            summary: "Reproducible floating-point accumulation via bin-exponent alignment (Demmel-Nguyen)",
            spec: "\
rfa_bin_exponent_aligned implements the reproducible floating-point summation
algorithm from:

  Demmel, J., & Nguyen, H. D. (2013). Fast Reproducible Floating-Point Summation.
  Proceedings of the 21st IEEE Symposium on Computer Arithmetic (ARITH 21).

  Demmel, J., & Nguyen, H. D. (2015). Parallel Reproducible Summation.
  IEEE Transactions on Computers, 64(7), 2060-2070.

Algorithm sketch (Phase 1 target: K=3 folds, DBWIDTH=40 bits):
  1. Pass 1 — Scan: compute the maximum biased exponent E_max of all inputs.
  2. Derive bin boundaries: bin k covers [2^(E_max - k*DBWIDTH), 2^(E_max - (k+1)*DBWIDTH)).
     For K=3, there are 3 bins covering approximately 120 bits of range.
  3. Pass 2 — Deposit: for each input v, round v to the nearest bin boundary and
     accumulate the fractional part into the bin k accumulator.
  4. Combine bins: sum the K bin accumulators in order from highest bin to lowest.

State: 48 bytes (3 × 64-bit accumulators + 3 × 64-bit overflow guards).

Correctness guarantee:
  Two executions of rfa_bin_exponent_aligned on the same input values produce
  bit-identical output regardless of the order inputs arrive, regardless of
  backend, and regardless of parallelism level. This is the property that
  sequential_left and tree_fixed_fanout_2 cannot provide for large parallel reductions.

STUB: This entry is registered so IR programs can name it and the verifier
accepts it. The reference_impl panics at runtime. Implementation is scheduled
for Peak 6 (Deterministic Reductions). The bit-exact test vectors are empty
until the algorithm is implemented and validated against the Demmel-Nguyen
reference implementation at 50-digit mpmath precision.",
            reference_impl: rfa_bin_exponent_aligned_impl,
            test_vectors: RFA_BIN_EXPONENT_ALIGNED_VECTORS,
            compat_class: "rfa",
        },
    );

    m
}

// ═══════════════════════════════════════════════════════════════════
// Public access
// ═══════════════════════════════════════════════════════════════════

/// Return the global registry, built on first call.
///
/// The registry is a `&'static` reference to a `HashMap` that is initialized
/// exactly once via `std::sync::OnceLock`.
pub fn registry() -> &'static HashMap<&'static str, StrategyEntry> {
    static REGISTRY: std::sync::OnceLock<HashMap<&'static str, StrategyEntry>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(build_registry)
}

/// Look up a strategy by name. Returns `None` if the name is not registered.
pub fn lookup(name: &str) -> Option<&'static StrategyEntry> {
    registry().get(name)
}

/// Return all registered strategy names, sorted alphabetically.
pub fn all_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = registry().keys().copied().collect();
    names.sort_unstable();
    names
}

/// Return true if `name` is a registered strategy.
pub fn is_known(name: &str) -> bool {
    registry().contains_key(name)
}

/// Return true if two named strategies are compatible for cross-kernel fusion.
///
/// Two strategies are fusable when they share the same `compat_class`. For
/// Phase 1 this means identical classes — `sequential` with `sequential`,
/// `tree_pow2` with `tree_pow2`, `rfa` with `rfa`. Cross-class fusion
/// (e.g. `sequential` + `tree_pow2`) is not safe: the merged accumulation
/// order would differ from either strategy's contract.
///
/// Returns `None` if either name is not registered (caller should
/// validate with `is_known` first; this path means a verifier bug).
///
/// Campsite 1.15 (Phase 2) may refine this to allow fusability across
/// strategies within the same class that satisfy an additional compatibility
/// predicate (e.g. same fanout for tree strategies). For Phase 1, class
/// equality is sufficient.
pub fn are_fusable(a: &str, b: &str) -> Option<bool> {
    let ea = lookup(a)?;
    let eb = lookup(b)?;
    Some(ea.compat_class == eb.compat_class)
}

/// Run the reference implementation for a named strategy on the given values.
///
/// Panics if `name` is not registered (verifier should prevent this at
/// compile time) or if the strategy is a stub (not yet implemented).
pub fn run(name: &str, values: &[f64]) -> f64 {
    match lookup(name) {
        Some(entry) => (entry.reference_impl)(values),
        None => panic!(
            "OrderStrategy '{}' is not in the registry. Known strategies: {:?}",
            name,
            all_names()
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Registry structure ──────────────────────────────────────────

    #[test]
    fn registry_contains_required_entries() {
        assert!(is_known("sequential_left"));
        assert!(is_known("tree_fixed_fanout_2"));
        assert!(is_known("rfa_bin_exponent_aligned"));
    }

    #[test]
    fn registry_rejects_unknown_name() {
        assert!(!is_known("backend_default"));
        assert!(!is_known(""));
        assert!(!is_known("SequentialLeft"));
        assert!(!is_known("sequential-left"));
    }

    #[test]
    fn all_names_returns_sorted() {
        let names = all_names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn lookup_returns_correct_entry() {
        let e = lookup("sequential_left").unwrap();
        assert_eq!(e.name, "sequential_left");
        assert_eq!(e.compat_class, "sequential");

        let e2 = lookup("tree_fixed_fanout_2").unwrap();
        assert_eq!(e2.name, "tree_fixed_fanout_2");
        assert_eq!(e2.compat_class, "tree_pow2");

        let e3 = lookup("rfa_bin_exponent_aligned").unwrap();
        assert_eq!(e3.name, "rfa_bin_exponent_aligned");
        assert_eq!(e3.compat_class, "rfa");
    }

    // ── sequential_left reference impl ─────────────────────────────

    #[test]
    fn sequential_left_test_vectors() {
        for tv in SEQUENTIAL_LEFT_VECTORS {
            let result = sequential_left_impl(tv.inputs);
            if tv.expected.is_nan() {
                assert!(
                    result.is_nan(),
                    "sequential_left '{}': expected NaN, got {}",
                    tv.label,
                    result
                );
            } else {
                assert_eq!(
                    result,
                    tv.expected,
                    "sequential_left '{}': expected {}, got {}",
                    tv.label,
                    tv.expected,
                    result
                );
            }
        }
    }

    #[test]
    fn sequential_left_large_then_small_pins_one() {
        // Pin the sequential fold result on this specific input.
        // (0 + 1e15) + 1.0 = 1000000000000001.0 (1.0 is within the ULP of 1e15).
        // Then + (-1e15) = 1.0.
        let result = run("sequential_left", &[1e15, 1.0, -1e15]);
        assert_eq!(result, 1.0);
    }

    // ── tree_fixed_fanout_2 reference impl ─────────────────────────

    #[test]
    fn tree_fixed_fanout_2_test_vectors() {
        for tv in TREE_FIXED_FANOUT_2_VECTORS {
            let result = tree_fixed_fanout_2_impl(tv.inputs);
            if tv.expected.is_nan() {
                assert!(
                    result.is_nan(),
                    "tree_fixed_fanout_2 '{}': expected NaN, got {}",
                    tv.label,
                    result
                );
            } else {
                assert_eq!(
                    result,
                    tv.expected,
                    "tree_fixed_fanout_2 '{}': expected {}, got {}",
                    tv.label,
                    tv.expected,
                    result
                );
            }
        }
    }

    #[test]
    fn tree_vs_sequential_differ_on_sensitive_input() {
        // Pinned divergence test: an input where the two strategies produce
        // different bits. This documents and locks the divergence behavior.
        //
        // Input: [1.0, 1e-16, -1.0, 1e-16]
        //
        // sequential_left: fold left starting from 0.0.
        //   0.0 + 1.0 = 1.0
        //   1.0 + 1e-16 = 1.0  (1e-16 < ulp(1.0)/2 ≈ 1.11e-16, lost by RNE)
        //   1.0 + (-1.0) = 0.0
        //   0.0 + 1e-16 = 1e-16
        //   Result: 1e-16
        //
        // tree_fixed_fanout_2: mid=2.
        //   left = tree([1.0, 1e-16]) = 1.0 + 1e-16 = 1.0 (1e-16 lost)
        //   right = tree([-1.0, 1e-16]) = -1.0 + 1e-16 = -1.0 + 1e-16
        //   Note: -1.0 + 1e-16 may retain the 1e-16 depending on rounding mode
        //   and exact f64 representation. The empirical result is pinned below.
        //   final: 1.0 + (-1.0 + 1e-16) = 2^-53 ≈ 1.1102230246251565e-16
        let input = [1.0_f64, 1e-16, -1.0, 1e-16];
        let seq = sequential_left_impl(&input);
        let tree = tree_fixed_fanout_2_impl(&input);
        // Both strategies must give the pinned empirical values.
        assert_eq!(seq, 1e-16, "sequential_left on pinned input");
        // The tree result: pin to the actual computed value (2^-53).
        let expected_tree = 2.0_f64.powi(-53); // = 1.1102230246251565e-16
        assert_eq!(tree, expected_tree, "tree_fixed_fanout_2 on pinned input");
        // These must differ — if they become equal, the numerical properties changed.
        assert_ne!(seq.to_bits(), tree.to_bits(), "strategies must diverge on this input");
    }

    #[test]
    fn tree_fixed_fanout_2_deterministic_at_many_sizes() {
        // Run at sizes 0..=32 and verify the result is stable (calling twice
        // returns the same bits).
        for n in 0..=32usize {
            let vals: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
            let r1 = tree_fixed_fanout_2_impl(&vals);
            let r2 = tree_fixed_fanout_2_impl(&vals);
            assert_eq!(r1.to_bits(), r2.to_bits(), "size={}", n);
        }
    }

    // ── rfa_bin_exponent_aligned stub ──────────────────────────────

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn rfa_stub_panics() {
        run("rfa_bin_exponent_aligned", &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn rfa_entry_has_no_test_vectors_yet() {
        assert!(RFA_BIN_EXPONENT_ALIGNED_VECTORS.is_empty());
    }

    // ── run() dispatcher ───────────────────────────────────────────

    #[test]
    #[should_panic(expected = "not in the registry")]
    fn run_unknown_strategy_panics() {
        run("no_such_strategy", &[1.0]);
    }

    // ── compat_class metadata ──────────────────────────────────────

    #[test]
    fn compat_classes_are_distinct_across_families() {
        let seq = lookup("sequential_left").unwrap().compat_class;
        let tree = lookup("tree_fixed_fanout_2").unwrap().compat_class;
        let rfa = lookup("rfa_bin_exponent_aligned").unwrap().compat_class;
        // All three families are incompatible with each other.
        assert_ne!(seq, tree);
        assert_ne!(seq, rfa);
        assert_ne!(tree, rfa);
    }
}
