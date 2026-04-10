# Aristotle's Proposals — Next Landscape

*2026-04-10, after industrialization expedition*

---

## Answer to Navigator's Specific Question

**Is the using() two-scopes resolution stable enough to close?**

Yes. The three-tier model at `20260410164514-using-two-scopes` is the right design:
1. Step-level `using()` — consumed per-step, drains
2. Session-level `session.defaults(...)` — persistent, never drains
3. Coded defaults — `unwrap_or(0.05)`

Lookup pattern is pinned in the campsite. The design is stable. What's NOT done yet is the implementation: `session_defaults: UsingBag` field needs to land in `TamSession`, and `set_defaults()` needs to be added to tbs_executor. Those are implementation tasks, not open design questions.

**My classification**: using()-two-scopes as a DESIGN decision = closed. As an implementation task = open. Should move to the next wave as an implementation campsite, not a design campsite.

---

## My Proposals

### Proposal A: `.advise()` as Third Epistemic Stance

**What**: A new TBS surface between bare method call (Layer 1 silent) and `discover()` (no collapse). `.advise()` runs Layer 1 diagnostics, picks a winner, and returns the decision graph alongside the result.

**Why now**: The `using()` → `discover()` design is complete. The gap between them — a practitioner who wants the recommendation AND wants to see the reasoning — is unaddressed. This is the interpretability layer.

**Concrete deliverable**: A `TbsAdvice` struct:
```rust
pub struct TbsAdvice<T> {
    pub result: T,                          // the winning method's output
    pub recommended_method: String,         // what was chosen
    pub diagnostics: Vec<DiagnosticStep>,   // what was checked and found
    pub alternatives_ruled_out: Vec<(String, String)>, // (method, reason)
    pub confidence: AdviceConfidence,       // Clean / Split / Weak
}
```

The diagnostics are computed anyway by Layer 1. `.advise()` is "don't discard them." Nearly free to add.

**Kingdom**: This is Layer 1-2 orchestration, not a math primitive. Lives above tambear's math layer.

**Suggested owner**: math-researcher or tbs_executor implementer.

---

### Proposal B: Op Enum Extension — TropicalMinPlus, AffineSemigroup, MatrixMul

**What**: The scout identified three missing Op variants during the expedition:
- `Op::TropicalMinPlus` — for PELT changepoint DP and shortest-path problems
- `Op::AffineSemigroup` — for GARCH filter, EMA, any first-order linear recurrence
- `Op::MatrixMul` — for companion matrix power (ARMA exact likelihood, Fibonacci, etc.)

**Why now**: Without these, any Kingdom A claim for PELT/GARCH/EMA/ARMA is architecturally incomplete. The math is proven Kingdom A; the Op enum doesn't have the semigroup to express it. Users who want these computations are forced into sequential loops.

**Concrete deliverable**: Add to `accumulate.rs`:
```rust
pub enum Op {
    // existing: Add, Max, Min, ArgMin, ArgMax, DotProduct, Distance
    TropicalMinPlus,                    // state: (cost: f64, predecessor: usize)
    TropicalMaxPlus,                    // state: (cost: f64, predecessor: usize)
    AffineSemigroup,                    // state: (a: f64, b: f64), op: (a1,b1)∘(a2,b2) = (a1*a2, a1*b2+b1)
    MatrixMul { size: usize },          // state: Vec<f64> (flattened n×n matrix)
}
```

Each needs: `identity()`, `combine()`, `degenerate()`.

**Suggested owner**: math-researcher (knows the semigroup structures) + adversarial (writes edge-case tests for each).

---

### Proposal C: Validity Semantics — One Policy, Declared

**What**: Implement the adversarial agent's `adversarial-validity-semantics` proposal as a concrete decision + codebase sweep.

The three implicit policies currently in use:
- **Propagate**: return NaN (most functions follow this)
- **Ignore**: return INFINITY (log_gamma for x≤0)
- **Panic**: assert!/unwrap (scattered throughout)

We need ONE policy, declared as a tambear contract invariant, with:
1. A decision recorded in CLAUDE.md or the campsite logbook
2. A sweep to make the codebase consistent
3. An adversarial test that confirms the policy holds at every public entry point

**My recommendation on the policy**: Propagate (NaN). Reasons:
- Consistent with IEEE 754 semantics
- Composable — NaN chains naturally through arithmetic
- Testable — a sweep for INFINITY returns and panics is straightforward
- The adversarial tests already test NaN propagation; expanding to cover all entry points is incremental

`log_gamma(x≤0)` should return NaN, not INFINITY. The INFINITY behavior loses sign information AND doesn't compose with downstream arithmetic the way NaN does.

**Suggested owner**: adversarial (decision) + pathmaker (sweep implementation).

---

### Proposal D: Tsallis Escort Primitive

**What**: `tsallis_escort(pmf: &[f64], q: f64) -> Vec<f64]` — tilts a distribution by temperature q.

This is a two-line Kingdom A primitive (accumulate: Z_q = sum of p^q; gather: rescale). But it unlocks:
- Every Tsallis/Rényi entropy variant (they're all functionals of the escort)
- q-corrected hypothesis tests (reference distributions tilted to match data's tail-heaviness)
- q-from-multifractal estimator (data-driven temperature)
- A paper: "q as universal tail-heaviness parameter, unifying all heavy-tailed families"

**Why now**: The nonparametric work today surfaced multiple places where q=1 assumptions fail for financial data. The primitive is trivial to implement. The downstream impact is large.

**Suggested owner**: math-researcher. The escort itself is trivial; the interesting work is wiring it through the existing entropy/divergence families and building the q-estimator.

---

### Proposal E: Hypothesis-First Campsite Structure

**What**: Add a standard hypothesis preamble to new campsites:

```markdown
## Hypothesis

**Claim:** [one falsifiable sentence]  
**Confirmed by:** [what would count as evidence]  
**Refuted by:** [what would count as counter-evidence]  
**Status:** open / confirmed / refuted / dissolved
```

**Why now**: The holographic-error-correction campsite (navigator proposal #5) is already implicitly a hypothesis experiment. The scientist said "testable now." If we add the hypothesis preamble to that campsite, and it works, we have a template for all future campsites.

This is a low-cost structural change that makes the campsite system more epistemically rigorous. Campsites become experiments, not just explorations.

**Not a full proposal**: this is a meta-campsite about campsite structure. It should probably be a single navigator-owned file that proposes the standard, not a full campsite. Fits the navigator's "insight-dependency-graph" thread (proposal #6).

---

## My Priority Read

Against the navigator's framing:

**Unlocks the most downstream**: Op enum extension (B) — GARCH/EMA/PELT/ARMA all become cleanly Kingdom A once the semigroup exists. This is structural.

**Closes an open design question**: Validity semantics (C) — one decision, then a sweep. The adversarial wave 18 findings are sitting half-done without this.

**Highest leverage / lowest cost**: Tsallis escort (D) — two lines of code, large conceptual unlock.

**Right time but not urgent**: `.advise()` (A) — design is ready, implementation is straightforward, but TBS surface work (task #8) should come first.

**Process improvement**: Hypothesis-first campsites (E) — meta, low cost, worth doing before the next wave starts rather than after.
