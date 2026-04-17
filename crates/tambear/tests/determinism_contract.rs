//! Cross-platform bit-exact determinism contract — the permanent CI gate.
//!
//! Codifies the Op contract documented in `src/accumulate.rs` and
//! `campsites/industrialization/architecture/2026-04-11-op-default-deterministic-plan.md`.
//!
//! Every `Op` variant must produce the same bit pattern regardless of
//! - thread count (1 vs N)
//! - execution order within parallel regions
//! - backend (CPU / CUDA / wgpu)
//! - CPU architecture
//!
//! for the same input.
//!
//! # Test categories
//!
//! 1. **Run-to-run stability** — the same call twice returns identical bits.
//!    For `Max`/`Min`/`ArgMax`/`ArgMin`: passes today (idempotent). For
//!    `Op::Add`: currently `#[ignore]`d. Today's engine auto-selects the GPU
//!    backend via `tam_gpu::detect()` when CUDA is available, and the GPU
//!    scatter_phi uses `atomicAdd` → non-deterministic run-to-run. Unignore
//!    once (a) backend pinning via `using(backend: "cpu")` is wired (plan
//!    step 4), and (b) either CPU Kulisch (step 3) or GPU deterministic
//!    scatter (step 7+) is active on the selected backend.
//!
//! 2. **Idempotent Ops (Max / Min / ArgMax / ArgMin)** — match a naive
//!    deterministic oracle. Idempotent + commutative → already cross-platform
//!    deterministic by construction (CAS on GPU, tree-reduction on CPU both
//!    give the same bits). Passes today.
//!
//! 3. **`Op::Add` matches a Kulisch-exact oracle** — the correctness gate
//!    for the Kulisch-backed default. Currently `#[ignore]`d; unignore when
//!    step 3 of the plan replaces `scatter_phi`'s atomicAdd-style `+=` with
//!    per-group Kulisch registers. Adversarial inputs (cancellation, Kahan
//!    traps, scale-mix) will flip from red to green when that lands.
//!
//! 4. **Kulisch primitive contract** — the merge operation is bit-exact
//!    associative over Op-relevant corpora. Already covered by
//!    `kulisch_accumulator.rs` unit tests; re-asserted here as the
//!    cross-file invariant. Passes today.

use tambear::primitives::specialist::kulisch_accumulator::KulischAccumulator;
use tambear::{AccResult, AccumulateEngine, Expr, Grouping, Op};

// ═══════════════════════════════════════════════════════════════════════════
// Corpora — adversarial inputs that stress associativity, compensation,
// and scale mixing. Kulisch sees through all of them; naive f64 sum does not.
// ═══════════════════════════════════════════════════════════════════════════

fn corpus_small() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 4.0, 5.0]
}

fn corpus_signed() -> Vec<f64> {
    vec![3.7, -0.25, 1e10, -0.5, 1.25, -1e10, 42.0, -6.0]
}

fn corpus_cancellation() -> Vec<f64> {
    // 1e17 + 1 + -1e17 = 1 (exactly). Naive f64 drops the 1 inside the sum.
    let mut xs = Vec::with_capacity(3000);
    for _ in 0..1000 {
        xs.push(1e17);
        xs.push(1.0);
        xs.push(-1e17);
    }
    xs
}

fn corpus_kahan_trap() -> Vec<f64> {
    // One large + many small. Naive drops all the small ones.
    let mut xs = vec![1.0];
    for _ in 0..10_000 {
        xs.push(1e-10);
    }
    xs
}

fn corpus_mixed_scale() -> Vec<f64> {
    (0..500)
        .map(|i| match i % 4 {
            0 => (i as f64) * 1e50,
            1 => -(i as f64),
            2 => (i as f64) * 1e-50,
            _ => (i as f64).sin() * 1e10,
        })
        .collect()
}

fn corpus_subnormal() -> Vec<f64> {
    let tiny = f64::from_bits(1);
    (0..2000).map(|_| tiny).collect()
}

fn all_corpora() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("small", corpus_small()),
        ("signed", corpus_signed()),
        ("cancellation", corpus_cancellation()),
        ("kahan_trap", corpus_kahan_trap()),
        ("mixed_scale", corpus_mixed_scale()),
        ("subnormal", corpus_subnormal()),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Oracles — the ground truth that `accumulate` must match bit-for-bit.
// ═══════════════════════════════════════════════════════════════════════════

/// Exact sum via Kulisch. Correctly-rounded final f64.
fn oracle_add_all(values: &[f64]) -> f64 {
    let mut acc = KulischAccumulator::new();
    acc.add_slice(values);
    acc.to_f64()
}

/// Per-group exact sum via Kulisch.
fn oracle_add_bykey(values: &[f64], keys: &[i32], n_groups: usize) -> Vec<f64> {
    let mut accs: Vec<KulischAccumulator> =
        (0..n_groups).map(|_| KulischAccumulator::new()).collect();
    for (v, &k) in values.iter().zip(keys.iter()) {
        accs[k as usize].add_f64(*v);
    }
    accs.iter().map(|a| a.to_f64()).collect()
}

/// Max over finite values, NaN-skip semantics.
fn oracle_max_all(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Min over finite values, NaN-skip semantics.
fn oracle_min_all(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn assert_bits_eq(got: f64, want: f64, label: &str) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{label}: got {got:e} (bits {:#018x}), want {want:e} (bits {:#018x})",
        got.to_bits(),
        want.to_bits()
    );
}

fn scalar_of(r: AccResult) -> f64 {
    match r {
        AccResult::Scalar(s) => s,
        other => panic!("expected Scalar, got {other:?}"),
    }
}

fn per_group_of(r: AccResult) -> Vec<f64> {
    match r {
        AccResult::PerGroup(v) => v,
        other => panic!("expected PerGroup, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category 1: run-to-run stability
// ═══════════════════════════════════════════════════════════════════════════
//
// The same call, same inputs, twice → identical bits. Passes today on the
// single-threaded CPU path; the gate for any future parallelism or GPU path.

#[test]
#[ignore = "FIXME(plan-step-3+4): engine auto-selects CUDA → atomicAdd → nondeterministic. Unignore after CPU Kulisch (step 3) + backend pinning via using() (step 4)."]
fn run_to_run_stable_add_all() {
    let mut engine = AccumulateEngine::new().expect("engine init");
    for (name, xs) in all_corpora() {
        let a = scalar_of(
            engine
                .accumulate(&xs, Grouping::All, Expr::Value, Op::Add)
                .expect("accumulate a"),
        );
        let b = scalar_of(
            engine
                .accumulate(&xs, Grouping::All, Expr::Value, Op::Add)
                .expect("accumulate b"),
        );
        assert_bits_eq(a, b, &format!("Add/All run-to-run on corpus '{name}'"));
    }
}

#[test]
fn run_to_run_stable_max_min_all() {
    let mut engine = AccumulateEngine::new().expect("engine init");
    for (name, xs) in all_corpora() {
        for (op_name, op) in [("Max", Op::Max), ("Min", Op::Min)] {
            let a = scalar_of(
                engine
                    .accumulate(&xs, Grouping::All, Expr::Value, op)
                    .expect("accumulate a"),
            );
            let b = scalar_of(
                engine
                    .accumulate(&xs, Grouping::All, Expr::Value, op)
                    .expect("accumulate b"),
            );
            assert_bits_eq(a, b, &format!("{op_name}/All run-to-run on '{name}'"));
        }
    }
}

#[test]
#[ignore = "FIXME(plan-step-3+4): engine auto-selects CUDA → atomicAdd → nondeterministic. Unignore after CPU Kulisch (step 3) + backend pinning via using() (step 4)."]
fn run_to_run_stable_add_bykey() {
    let mut engine = AccumulateEngine::new().expect("engine init");
    let xs = corpus_mixed_scale();
    let keys: Vec<i32> = (0..xs.len()).map(|i| (i % 7) as i32).collect();
    let grouping = Grouping::ByKey {
        keys: &keys,
        n_groups: 7,
    };

    let a = per_group_of(
        engine
            .accumulate(&xs, grouping.clone(), Expr::Value, Op::Add)
            .expect("accumulate a"),
    );
    let b = per_group_of(
        engine
            .accumulate(&xs, grouping, Expr::Value, Op::Add)
            .expect("accumulate b"),
    );
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert_bits_eq(a[i], b[i], &format!("Add/ByKey group {i} run-to-run"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category 2: idempotent Ops match a naive deterministic oracle
// ═══════════════════════════════════════════════════════════════════════════
//
// Max/Min are idempotent + commutative → any atomic / tree / serial order
// gives the same result. Match the straightforward reference implementation.

#[test]
fn op_max_all_matches_naive_oracle() {
    let mut engine = AccumulateEngine::new().expect("engine init");
    for (name, xs) in all_corpora() {
        let got = scalar_of(
            engine
                .accumulate(&xs, Grouping::All, Expr::Value, Op::Max)
                .expect("accumulate"),
        );
        let want = oracle_max_all(&xs);
        assert_bits_eq(got, want, &format!("Max/All vs oracle on '{name}'"));
    }
}

#[test]
fn op_min_all_matches_naive_oracle() {
    let mut engine = AccumulateEngine::new().expect("engine init");
    for (name, xs) in all_corpora() {
        let got = scalar_of(
            engine
                .accumulate(&xs, Grouping::All, Expr::Value, Op::Min)
                .expect("accumulate"),
        );
        let want = oracle_min_all(&xs);
        assert_bits_eq(got, want, &format!("Min/All vs oracle on '{name}'"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category 3: Op::Add matches the Kulisch-exact oracle
// ═══════════════════════════════════════════════════════════════════════════
//
// THE gate. These tests assert that `Op::Add` is correctly-rounded exact
// across adversarial inputs. Until step 3 of the plan lands
// (replace `scatter_phi`'s `+=` with per-group Kulisch registers + tree-merge),
// the cancellation / Kahan-trap corpora will lose bits via non-associative
// f64 add and these tests fail.
//
// `#[ignore]`d for now with a FIXME pointing at the plan. When step 3 lands,
// remove the `#[ignore]` attribute and the tests become permanent regression
// gates.

#[test]
#[ignore = "FIXME(plan-step-3): passes once scatter_phi uses Kulisch registers + merge"]
fn op_add_all_matches_kulisch_oracle() {
    let mut engine = AccumulateEngine::new().expect("engine init");
    for (name, xs) in all_corpora() {
        let got = scalar_of(
            engine
                .accumulate(&xs, Grouping::All, Expr::Value, Op::Add)
                .expect("accumulate"),
        );
        let want = oracle_add_all(&xs);
        assert_bits_eq(got, want, &format!("Add/All vs Kulisch oracle on '{name}'"));
    }
}

#[test]
#[ignore = "FIXME(plan-step-3): passes once scatter_phi uses Kulisch registers + merge"]
fn op_add_bykey_matches_kulisch_oracle() {
    let mut engine = AccumulateEngine::new().expect("engine init");
    let xs = corpus_mixed_scale();
    let n_groups = 8;
    let keys: Vec<i32> = (0..xs.len()).map(|i| (i % n_groups) as i32).collect();

    let got = per_group_of(
        engine
            .accumulate(
                &xs,
                Grouping::ByKey {
                    keys: &keys,
                    n_groups,
                },
                Expr::Value,
                Op::Add,
            )
            .expect("accumulate"),
    );
    let want = oracle_add_bykey(&xs, &keys, n_groups);
    assert_eq!(got.len(), want.len());
    for i in 0..got.len() {
        assert_bits_eq(got[i], want[i], &format!("Add/ByKey group {i} vs oracle"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category 4: Kulisch associativity invariant (direct, no engine)
// ═══════════════════════════════════════════════════════════════════════════
//
// Sanity: the Kulisch merge operation is bit-exact associative at the word
// level. This is what makes per-thread-then-merge parallelism safe.
// (Already covered in kulisch_accumulator.rs unit tests; re-asserted here
// as the cross-file contract between the primitive and the engine harness.)

#[test]
fn kulisch_merge_is_associative_over_corpora() {
    for (name, xs) in all_corpora() {
        if xs.len() < 30 {
            continue;
        }
        let (left, rest) = xs.split_at(xs.len() / 3);
        let (mid, right) = rest.split_at(rest.len() / 2);

        let mut a = KulischAccumulator::new();
        a.add_slice(left);
        let mut b = KulischAccumulator::new();
        b.add_slice(mid);
        let mut c = KulischAccumulator::new();
        c.add_slice(right);

        // (a ⊕ b) ⊕ c
        let mut lhs = a.clone();
        lhs.merge(&b);
        lhs.merge(&c);

        // a ⊕ (b ⊕ c)
        let mut bc = b.clone();
        bc.merge(&c);
        let mut rhs = a.clone();
        rhs.merge(&bc);

        assert_eq!(
            lhs.to_f64().to_bits(),
            rhs.to_f64().to_bits(),
            "Kulisch merge associativity broke on corpus '{name}'"
        );
    }
}
