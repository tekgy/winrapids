# Navigator Campsite: Guarantee Ledger — Kingdom B/C/D Extension

**Created by:** navigator
**Session:** Bit-Exact Trek, 2026-04-12
**Type:** Future work — formal analysis

---

## What this is

The current guarantee-ledger.md covers invariants I1–I11, all of which apply to Kingdom A (embarrassingly parallel, map-reduce) computations. The ledger has an open questions section that flags: "What does a full guarantee ledger look like for Kingdom B (sequential recurrences), Kingdom C (iterative fixed points), and Kingdom D (online/adaptive)?"

This campsite records what I understand about that question, what's still unknown, and what the next navigator needs to think about when Kingdom B lands.

---

## Current ledger coverage

All eleven invariants target the same abstract problem:

> Given a `.tam` program with only Kingdom A operations, guarantee bit-identical outputs on CPU, PTX (CUDA), and Vulkan/SPIR-V.

The invariants decompose into:
- **I1–I3**: Precision foundation (f64 as base, no implicit widening, no FMA contraction)
- **I4–I8**: Reduction ordering (deterministic grouping, fixed tree topology, no hardware atomics, same seed)
- **I9–I10**: Verification (mpmath oracle, cross-backend diff)
- **I11**: NaN propagation (IEEE-754 compliant, no silent NaN drops)

All of these can be stated, tested, and enforced in Kingdom A without reference to sequential recurrence or iterative convergence.

---

## The Kingdom B gap

Kingdom B operations are those with inherent sequential data dependencies — each step depends on the previous result. Examples in tambear-libm:

- **Cody-Waite reduction accumulation** (the "r_lo = r - r_hi * LN2_LO" correction step that must come after the hi-subtraction)
- **Recurrence relations** in ARMA, Kalman filter, Levin-u acceleration
- **Horner's method** evaluation of high-degree polynomials (formally sequential, though often unrolled as a reduction tree in practice)

For Kingdom B, the current ledger has three open questions:

### Open question 1: Does I3 (no FMA contraction) apply to sequential steps?

In Kingdom A, no-FMA is enforced per-op by `NoContraction`. In a Horner evaluation `a₀ + x*(a₁ + x*(a₂ + ...))`, each nested multiply-add is a candidate for FMA fusion. If the ops are sequential (cannot reorder), can NoContraction still be applied?

**Provisional answer:** Yes. NoContraction in SPIR-V is a per-instruction decoration — it doesn't care about the sequential/parallel structure. Each `OpFMul` + `OpFAdd` pair in the Horner chain gets NoContraction independently. PTX `.contract false` is similarly per-instruction. Sequential order ≠ inability to annotate.

**Caveat:** Some backends (e.g., future NPU) may not support per-op contraction suppression for sequential chains. A future I3-extension for Kingdom B would need to verify that the same guarantee holds on every target.

### Open question 2: Does I4 (deterministic grouping) extend to sequential reductions?

I4 says: "Every summation uses a fixed left-to-right (or tree) grouping; the grouping is stable across backends and runs." For Kingdom A, this is enforced by the `OrderStrategy` registry — you choose a tree topology and stick to it.

For Kingdom B sequential reductions (like the Kahan compensated summation or the Cody-Waite two-step), the grouping is fixed by the algorithm — you can't reorder the steps without changing the mathematical result. So I4 is "trivially satisfied" for Kingdom B: there's only one grouping (the sequential one), and it's deterministic by construction.

**But:** The subtle question is whether "sequential order" is the same across backends. If PTX speculatively reorders instructions within a sequential chain (which it can legally do for arithmetic ops without memory dependencies), the chain is no longer guaranteed sequential. The invariant for Kingdom B would need to explicitly require `PTX_SCHEDULER_RESPECT_ORDER` (or equivalent annotation) on chains with mathematical sequential dependencies.

This is a potential I4-extension: **I4b — for Kingdom B sequential chains, the backend must preserve mathematical ordering of dependent operations.** This may require explicit SPIR-V control flow (not just data flow) or PTX `SYNC` barriers between dependent ops.

### Open question 3: Does the mpmath oracle (I9) apply to non-closed-form results?

The mpmath oracle verifies against a closed-form reference. For Kingdom A functions (exp, log, sin, etc.), the closed-form is exactly the mathematical definition of the function.

For Kingdom B iterative algorithms (convergent eigendecomposition, Newton-Raphson root finding, Gauss-Seidel), the "true answer" is the limit of an infinite sequence — not a closed form. The mpmath oracle can verify the limit to arbitrary precision, but the tambear result is at a finite iteration count.

**Implication for the ledger:** Kingdom B oracle strategy must be different from Kingdom A:
- For sequences with known closed forms (eigenvalues of structured matrices, algebraic roots), use the closed-form oracle directly.
- For general iterative algorithms, use a convergence criterion oracle: "the result is within X of the fixed point" for some provable X.
- For ARMA residuals and similar, use trajectory oracle: "the trajectory matches the reference trajectory to within 1 ULP at each step."

This would require a new invariant I9b: **trajectory oracle for Kingdom B sequential chains.**

---

## The Kingdom C gap (iterative fixed points)

Kingdom C is for global iterative algorithms — EM, k-means, Newton, etc. — where convergence is to a fixed point but there's no guarantee of monotone progress.

The ledger gap for Kingdom C is more fundamental: **bit-identical convergence across backends may be impossible** unless:
1. The algorithm is run for a fixed number of iterations (not until convergence), OR
2. The convergence criterion is bit-identical across backends (which it won't be if it uses floating-point comparisons)

A Kingdom C invariant would likely be: "The result is within `ε` of the fixed point for a documented `ε`, but the specific path through iteration space is not guaranteed identical across backends." This is a weaker claim than bit-exact — it's a bounded-error claim.

**This means Kingdom C cannot have an I1-equivalent.** The guarantee ledger for Kingdom C would be a *different tier*: "convergence guarantees" instead of "bit-exact guarantees."

---

## What the next navigator needs to do when Kingdom B lands

1. **Audit the Kingdom B implementation for I3-extension violations.** Every sequential arithmetic chain should carry the same NoContraction annotations as Kingdom A ops.

2. **Check whether I4b is needed.** Run the same Kingdom B computation on CPU and GPU and compare traces step-by-step. If any step diverges, investigate whether backend instruction reordering is the cause.

3. **Draft a Kingdom B oracle strategy.** Extend I9 to cover trajectory oracles for sequential chains.

4. **Update the guarantee ledger** with a Kingdom B section. The Kingdom A invariants are the floor; Kingdom B adds ordering constraints on top.

5. **Don't conflate Kingdom B with Kingdom C.** They have different guarantee structures. Kingdom B can achieve bit-exact results (with the right annotations). Kingdom C probably cannot, and that's okay — the ledger should document why.

---

## The meta-observation

The guarantee ledger is currently a Kingdom A document masquerading as a universal one. Its invariants are correct and complete — for Kingdom A. When Kingdom B lands, the ledger needs a section header: "Part I: Kingdom A invariants (I1–I11)" and a new "Part II: Kingdom B extensions (I4b, I9b, and any new invariants)."

The decision to structure the ledger this way (rather than trying to write Kingdom B invariants now) is the right one: you can't know what the invariants need to be until you have an implementation to constrain. Write the implementation, then extract the invariants that would have caught the first bug.

That's the same order we used for Kingdom A. It worked.
