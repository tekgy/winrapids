<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# EMA is Kingdom A — Architectural Spec
Written: 2026-04-10

## The Math (Verified Against Codebase)

EMA recurrence in `signal_processing.rs:716-725`:
```rust
pub fn ema(data: &[f64], alpha: f64) -> Vec<f64> {
    result.push(data[0]);
    for i in 1..data.len() {
        let prev = result[i - 1];
        result.push(alpha * data[i] + (1.0 - alpha) * prev);
    }
}
```

This is `s_t = α·x_t + (1-α)·s_{t-1}` — a first-order affine recurrence.

In affine map form: `s_t = a_t · s_{t-1} + b_t` where `a_t = (1-α)`, `b_t = α·x_t`.

Affine maps compose: `(a₁, b₁) ∘ (a₂, b₂) = (a₁·a₂, a₁·b₂ + b₁)`.

This composition is **associative** — it forms a semigroup. Therefore:
- The EMA sequence is a prefix scan over affine maps
- Each element `s_t` = result of composing all maps `(a_1,b_1)...(a_t,b_t)` applied to `s_0`
- This IS Kingdom A (parallel prefix over an associative operation)

## What the Codebase Already Knows

`spec_compiler.rs:1061-1068` has a PASSING TEST that confirms EMA compiles to a prefix scan:
```rust
fn compile_ema_scan() {
    let f = parse_formula("ema = scan(a*x + (1 - a)*acc)").unwrap();
    let plan = compile_one(&f);
    let has_prefix = plan.steps.iter().any(|s| {
        matches!(s, PlanStep::Accumulate(a) if a.grouping == GroupingTag::Prefix)
    });
    assert!(has_prefix, "EMA should compile to a prefix scan");
}
```

The SPEC COMPILER knows EMA is a prefix scan. The actual `ema()` implementation doesn't use it — it's still a sequential loop. These two things are inconsistent: the compiler-level understanding is correct, but the runtime implementation doesn't reflect it.

## What Needs to Change

### Step 1: Add `affine_prefix_scan` as a primitive in `signal_processing.rs`

The general form: given a sequence of affine maps (aᵢ, bᵢ), compute the prefix product
at each position — the composed map from position 0 to position t applied to an initial state.

```rust
/// Parallel prefix scan over real affine maps (a, b) with composition:
///   (a₁, b₁) ∘ (a₂, b₂) = (a₁·a₂, a₁·b₂ + b₁)
///
/// Given affine maps as parallel arrays `a` and `b` and initial state `s0`,
/// computes s[t] = a[t]·a[t-1]·...·a[0]·s0 + composed b terms.
///
/// Kingdom A: the affine semigroup is associative, enabling Blelloch-style
/// parallel prefix. Sequential fallback used on CPU; GPU path uses scan kernel.
pub fn affine_prefix_scan(a: &[f64], b: &[f64], s0: f64) -> Vec<f64> {
    // Sequential implementation (correct, then optimize):
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 { return vec![]; }
    let mut result = Vec::with_capacity(n);
    let mut s = s0;
    for i in 0..n {
        s = a[i] * s + b[i];
        result.push(s);
    }
    result
}
```

Note: the sequential version IS correct. For Kingdom A classification, the point is that
this CAN be parallelized — the sequential path is the CPU fallback until the GPU scan
kernel exists. Adding the function establishes the correct abstraction even before
the parallel version is built.

### Step 2: Rewrite `ema` to delegate to `affine_prefix_scan`

```rust
pub fn ema(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    // EMA is an affine recurrence: s_t = (1-α)·s_{t-1} + α·x_t
    // Map: a_t = (1-α) for all t, b_t = α·x_t
    let decay = 1.0 - alpha;
    let a: Vec<f64> = vec![decay; data.len()];
    let b: Vec<f64> = data.iter().map(|&x| alpha * x).collect();
    // First element is the initial state; scan starts at index 1
    // Convention: s[0] = data[0] (no smoothing at t=0)
    let s0 = data[0];
    // Build maps for indices 1..n, prepend data[0]
    let a1 = &a[1..];
    let b1 = &b[1..];
    let mut result = vec![s0];
    result.extend(affine_prefix_scan(a1, b1, s0));
    result
}
```

This makes the Kingdom A nature explicit and establishes the `affine_prefix_scan`
abstraction as the thing to eventually GPU-parallelize.

### Step 3: Rewrite `ewma_variance` identically

`volatility.rs:541-554` — EWMA variance: `σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}`.

Same affine structure: `a_t = λ`, `b_t = (1-λ)·r²_{t-1}`.

```rust
pub fn ewma_variance(returns: &[f64], lambda: f64) -> Vec<f64> {
    let n = returns.len();
    if n == 0 { return vec![]; }
    let var0 = returns[0].powi(2).max(1e-15);
    let a: Vec<f64> = vec![lambda; n - 1];
    let b: Vec<f64> = returns[..n-1].iter()
        .map(|&r| (1.0 - lambda) * r * r)
        .collect();
    let mut result = vec![var0];
    let scan = affine_prefix_scan(&a, &b, var0);
    result.extend(scan.into_iter().map(|v| v.max(1e-15)));
    result
}
```

### Step 4: Add `ema_period` (the missing primitive from the TA decomposition)

`ema_period(data, period)` with `alpha = 2/(period+1)` (the standard EMA convention):

```rust
pub fn ema_period(data: &[f64], period: usize) -> Vec<f64> {
    let alpha = 2.0 / (period as f64 + 1.0);
    ema(data, alpha)
}
```

This completes the TA primitive set needed for MACD, RSI, etc.

## What Does NOT Need to Change

The spec_compiler already correctly classifies `scan(...)` as `GroupingTag::Prefix`.
The theoretical framework is right. Only the runtime implementations need updating.

## All EMA-Equivalent Recurrences in the Codebase

Checked via grep: only two places use the EMA pattern in production code:
1. `signal_processing::ema` (line 716) — primary, needs updating
2. `volatility::ewma_variance` (line 541) — secondary, needs updating

The `state_space.rs` Kalman filter is handled by Sarkka Op — already correctly Kingdom A.

## The Broader n-th Order Case

General n-th order linear recurrence:
```
y[t] = a₁·y[t-1] + ... + aₙ·y[t-n] + b·x[t]
```
Rewritten in companion matrix form: the state vector `[y[t], ..., y[t-n+1]]ᵀ` evolves
as `F · state + G · x[t]` where F is the n×n companion matrix. Matrix products are
associative → this is also a prefix scan over n×n matrix products → Kingdom A.

The `state_space::SarkkaMerge` Op handles the n=1 case (scalar Kalman). The generalization
to ARMA(p,q) via companion matrices is a natural extension of the same infrastructure.

The `ema_period` function above is the scalar n=1 case. Any AR(p) model is the
companion-matrix n=p case. They're the same prefix scan at different matrix sizes.

## Kingdom Classification Clarification

EMA (and all first-order linear recurrences) should be classified:
- **Kingdom A** — the affine semigroup is associative, prefix scan is valid
- The Fock boundary is APPARENT, not real — it dissolves when you write the map form
- The sequential implementation is a CPU scheduling choice, not a structural constraint
- Comment in code should document: "// Kingdom A via affine semigroup prefix scan"


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

