# Target: Why bit-exact cross-hardware (vs provably bounded ULP)?

**Deconstructor:** Aristotle
**Date opened:** 2026-04-11
**Status:** Phases 1–8 drafted. Deconstruction stable.

The meta-goal of the trek, under deconstruction:

> "One compiled `.tam` kernel, the same source of math, the same numerical answers, running on any ALU through only its driver, with no vendor compiler and no vendor math library anywhere in the path."

The load-bearing phrase is **"the same numerical answers."** The trek interprets this as *bit-exact*: identical 64-bit patterns across CPU, CUDA, Vulkan, and every future backend. The alternative — that two existing deconstructions haven't questioned — is **provably bounded ULP**: a declared mathematical bound (say, ≤ 1 ULP) on how far two backends' answers can diverge on any input.

If bit-exactness is the right target, every peak on the trek is justified. If bounded-ULP is sufficient, the trek can be dramatically shorter (no Peak 6, less stringency in Peaks 2/3/7, and the tambear-libm implementation can follow ordinary numerical-analysis conventions instead of bit-perfect convention matching). The difference between the two is enormous in cost and in what the product actually guarantees.

---

## Phase 1 — Assumption Autopsy

Assumptions stacked inside "bit-exact cross-hardware is the right target":

1. **That users need bit-exactness.** Not "users benefit from consistency" — the stronger claim that the guarantee they need is "same bits." Many user categories might be satisfied by "same bits ± ULP-bounded error." Financial settlement needs reproducibility; physics simulation needs stability; ML training needs reproducibility-for-debugging but tolerates small numerical noise; research prototyping doesn't care. The target depends on *which* user we're optimizing for.

2. **That bit-exactness is qualitatively stronger than bounded-ULP.** This is superficially obvious but actually subtle. Bit-exactness is a SPECIAL CASE of bounded-ULP with bound = 0. So the strength difference depends on whether the 0-bound offers something the ε-bound doesn't. It does, but the difference is specific: you can test equality with `==`, you can content-address outputs, you can hash them, you can cache them across machines. Those are real but narrow.

3. **That bit-exactness is achievable with reasonable engineering cost.** The trek exists to demonstrate this — Peak 6 explicitly spends effort on it. But "achievable" depends on what we count: pure arithmetic kernels can be bit-exact without much work; transcendentals require libm to match bit-for-bit across backends, which requires *the same implementation of the same polynomial in the same order with the same rounding mode on every backend*. That's achievable but tight.

4. **That bit-exactness is testable.** Running the same kernel on two backends and asserting `out.to_bits() == out.to_bits()` is a finite test. But for continuous input domains, we test a *sample* and claim the property. The claim "bit-exact on ALL inputs" is stronger than "bit-exact on our test corpus." Phase 2 truth 8 (sampling is not representation) from Notebook 012 applies here too.

5. **That "cross-hardware" is a well-defined set.** NVIDIA Blackwell + AMD RDNA + Intel Xe + CPU (several ISAs) + future NPU + future TPU-like + future custom silicon. The claim is "any ALU." But future hardware might have new features (variable-precision arithmetic, posits, wider vectors) or missing features (no fp64, no denormals, no round-to-nearest-even as default). "Bit-exact on any ALU" implicitly assumes every ALU implements the same floating-point spec. That's true today for IEEE-754 fp64 hardware, but not eternally true.

6. **That non-Kingdom-A operations can be made bit-exact too.** The trek focuses on accumulate+gather (Kingdom A). For iterative fixed-point (Kingdom C), bit-exactness requires the iteration count AND the internal representation AND the termination check to match. For stochastic methods (MCMC, random sampling), bit-exactness requires the RNG sequence to match AND the consumption order AND the seeding protocol. For Monte Carlo, bit-exactness at the sample level may be feasible, but at the summary level it's extremely fragile.

7. **That "same source" and "same bits" are the right pairing.** The claim is "same source → same bits." But there's a weaker useful claim ("same source → same semantic answer") and a stronger useful claim ("same source → same executable instruction sequence"). Bit-exact sits in the middle. The trek has chosen the middle without deconstructing why.

8. **That the claim is the PRIMARY selling point.** The trek's README frames bit-exact cross-hardware as THE architectural claim. But tambear has other selling points: composability, sharing via TamSession, first-principles math, no vendor lock-in. If bit-exactness is costly, perhaps it shouldn't be the primary promise — it should be one of several promises, each with its own cost/benefit.

9. **That bit-exactness implies determinism.** It does in one direction: bit-exact → deterministic. It does NOT imply determinism in the other direction: you can be deterministic (reproducible-within-a-backend) without being bit-exact (identical-across-backends). Conflating these loses a cheaper win (determinism-per-backend) for the sake of the more expensive win (bit-exactness-across-backends).

10. **That users can VERIFY we achieved it.** If we claim bit-exact cross-hardware, a skeptical user should be able to run our library on two machines and check. If they run it and get a difference, our claim is false. That's a strong auditable property. Bounded-ULP is harder to verify ("is this within 1 ULP of something?") unless the bound is measured empirically on a corpus — which is the oracle question again.

---

## Phase 2 — Irreducible Truths

Stripped to what's undeniable:

1. **IEEE-754 fp64 is a total order on 2^64 bit patterns with a well-defined semantics for arithmetic operations.** Two different bit patterns are different numerical values (mostly — NaN is the exception, and that exception is real).

2. **A pure fp64 computation has a *unique correct answer* for each operation, given the round-to-nearest-even rule.** `a + b` in fp64 has a unique answer. No hardware can legally disagree about this for a single op. IEEE 754 guarantees it.

3. **A sequence of fp64 operations has a unique answer only when the sequence is totally ordered.** Because fp addition isn't associative, different orderings give different answers. But a fixed order has a unique answer.

4. **Two implementations executing THE SAME sequence of fp64 ops in THE SAME order produce THE SAME answer on any IEEE-754 compliant hardware.** This is the *foundational fact* that makes the trek feasible at all. If it weren't true, bit-exactness across hardware would be impossible.

5. **Implementations differ in *which* sequence they execute, not in *what* each op produces given identical inputs.** The cross-hardware divergence problem is a compiler/runtime problem (which operations, in which order), not a hardware problem (given the operations and order, the hardware is deterministic).

6. **"Bit-exact cross-hardware" is equivalent to "every backend compiles the source to an identical sequence of fp64 ops."** Not literally identical machine code, but identical at the IEEE-754 event level: same ops, same operands, same order.

7. **Producing the same sequence is achievable IF AND ONLY IF the source representation is precise enough to pin down the sequence.** An IR that leaves reduction order, contraction behavior, or constant folding up to the backend does NOT pin down the sequence. An IR that names every op, every order, every rounding mode DOES.

8. **Bounded-ULP cross-hardware is a *strictly weaker* claim than bit-exact.** Bounded-ULP with ε = 0 IS bit-exact. Bounded-ULP with ε > 0 allows the backends to execute *different* sequences as long as they produce answers within ε of each other. The weaker claim permits a much larger design space for the backends.

9. **The value of "bit-exact" to a user is the ability to test for equality.** `hash(output_A) == hash(output_B)` works iff bit-exact. Content-addressing works iff bit-exact. Reproducible CI signatures work iff bit-exact. Cross-machine cache hits work iff bit-exact. These are the irreplaceable use cases. For everything else — correctness, stability, numerical acceptability — bounded-ULP is sufficient.

10. **Bit-exactness does NOT imply correctness.** Two backends can be bit-exactly wrong together. The trek separates these: I5/I6/I7 (determinism + composition) produce bit-exactness; I9 (oracle) produces correctness. Phase 2 of Notebook 012 identified this separation. The current target depends on it.

---

## Phase 3 — Reconstruction from Zero

Given Phase 2, what are the plausible targets for a cross-hardware numerical library?

### 1. **No cross-hardware claim.** Each backend is its own numerical world.
- **Pro:** Cheapest. Each backend can optimize freely.
- **Con:** Users can't replicate results across machines. Our "runs everywhere" is a euphemism for "runs, produces different answers, on each everywhere."

### 2. **Bounded-ULP cross-hardware with a loose bound (e.g., 10 ULPs).**
- **Pro:** Backends have wide latitude. Each can use its own libm, its own reductions, its own FMA usage. As long as the bound holds, we're fine.
- **Con:** 10 ULPs is a lot — it hides bugs. Users who test for equality see failures. Content-addressing doesn't work.

### 3. **Bounded-ULP cross-hardware with a tight bound (≤ 1 ULP).**
- **Pro:** Almost as strong as bit-exact for most practical purposes. Backends have some latitude (FMA vs no-FMA, one reduction strategy vs another).
- **Con:** Still doesn't enable content-addressing. Testing is harder: "assert within 1 ULP" requires computing what the "right" answer is, which requires an oracle.

### 4. **Bit-exact for pure arithmetic, bounded-ULP for transcendentals.**
- **Pro:** The hard part (transcendentals) doesn't need bit-exactness; the easy part (arithmetic) does. This is actually what the trek is currently targeting in spirit — Peak 4's tolerance policy says "pure arithmetic bit-exact, transcendental within ULP bound."
- **Con:** Requires an oracle for the transcendental ULP bound (back to the I9 question). Content-addressing works for pure arithmetic only.

### 5. **Bit-exact cross-hardware on a declared *profile* of kernels.**
- **Pro:** Users who need bit-exactness declare it and pay the cost. Users who don't, get bounded-ULP. The library supports both modes and lets callers pick.
- **Con:** Two modes means two test matrices. Doubles the verification surface. Encourages sloppy defaults.

### 6. **Bit-exact cross-hardware as the DEFAULT for everything, with explicit opt-out to faster bounded-ULP modes.**
- **Pro:** The strong guarantee is the default. Users get the reproducibility they need without asking. Fast modes exist for users who've measured and accept the loss.
- **Con:** Default is expensive. Users who don't need bit-exactness pay for it anyway.

### 7. **Bit-exact cross-hardware as THE target (current trek).**
- **Pro:** Strongest guarantee. Simplest story. Content-addressing enabled globally. Testing is `==`, no oracles needed for cross-backend diffs.
- **Con:** Most expensive. Peak 6 exists to make reductions deterministic. Peak 2 must produce bit-identical libm. Peak 3/7 must emit explicit `.contract false` / NoContraction. Every future backend inherits the same rigor.

### 8. **Content-addressing as the real goal; bit-exactness as the means.**
- Reframe: the target isn't "same bits" as a property; it's "stable identity of outputs" as an engineering primitive for caching, hashing, comparison, cross-machine replication. Bit-exactness is the CURRENT way to achieve this.
- But: there might be other ways. Canonical rounding to a lower precision (fp32 for identity, fp64 for compute) gives a stable identity with looser compute constraints. "Bit-exact at fp32, bounded-ULP at fp64" is a real design choice that gives you content-addressing cheaply.
- **Pro:** Decouples the want (stable identity) from the means (bit-exactness).
- **Con:** Users probably want bit-exactness at their *working precision*, not at a coarser one. The fp32-identity approach is a hack.

### 9. **Machine-checked formal proof of bit-exact across a set of backends.**
- **Pro:** The strongest possible form of the claim. No empirical testing required — the proof shows bit-exactness for every input.
- **Con:** The trek doesn't have a formal methods infrastructure. Building one is 10x the cost of the trek itself. The libm alone would be a multi-year formal proof effort (see CRlibm's effort).

### 10. **Hybrid: bit-exact by default, bounded-ULP where we PROVE it's indistinguishable at the workload level.**
- Identify kernels where the end-user decision (is this trade profitable? does this phase converge? is this classification correct?) is insensitive to small numerical variation. For those kernels, bounded-ULP is operationally equivalent to bit-exact. For kernels where the end-user decision IS sensitive (comparison operators, rounding boundaries), bit-exact is required.
- **Pro:** Matches engineering cost to engineering need. Lets the cost of bit-exactness be paid only where it matters.
- **Con:** Requires workload-aware analysis. Hard to formalize. Users might not know which kernels they're using.

---

## Phase 4 — Assumption vs Truth Map

| Assumption | Matching truth | Where they collide |
|---|---|---|
| "Bit-exactness is the right target" | T9: bit-exactness enables equality testing, content addressing, cross-machine caching — specific and real but narrow use cases | The target is justified for those specific uses, not as a universal property. The trek's framing makes it sound like a universal need. |
| "Bit-exactness is qualitatively stronger than bounded-ULP" | T8: bit-exact is just bounded-ULP with ε=0 | The difference is quantitative at ε=0 vs ε>0, but the *value* of the difference is qualitative (equality tests). So the assumption is right in value but wrong in structure — and the structure matters for deconstruction because it shows bit-exact is the endpoint of a spectrum, not a different thing. |
| "Bit-exact across any ALU" | T1, T2, T4: IEEE-754 hardware is deterministic at the op level; bit-exact across IEEE-754 hardware is achievable by pinning the op sequence | True for IEEE-754 compliant hardware. NOT automatically true for non-IEEE hardware (posits, custom silicon without denormals, approximate computing units). "Any ALU" is actually "any IEEE-754 compliant ALU." That's a narrower claim. |
| "Bit-exact implies users can verify" | T10: bit-exact is the only property users can verify *locally* (run on two machines, compare bytes) without an oracle | This is strong support for the target — but it depends on the user actually wanting to verify. Most won't. Bit-exactness is a benefit mostly to users who have reproducibility needs that drive auditing, which are a subset. |
| "Peak 6 makes reductions deterministic and this IS bit-exactness" | T5, T6: bit-exactness requires identical op sequences across backends | Peak 6 gives determinism PER BACKEND (the same inputs produce the same output on the same backend). It does NOT by itself give bit-exactness ACROSS backends — that requires the same order strategy to be enforced at the IR level and respected by every backend. This is exactly the I7′ move from Notebook 011. Without that move, Peak 6 is necessary but not sufficient. |
| "The trek's claim is the primary selling point" | Project memory: tambear has multiple selling points (composability, sharing, anti-vendor-lock, first-principles math) | The trek frames bit-exactness as THE claim. But if bit-exactness costs more than the others and serves fewer users, the framing over-weights it. This is an Aristotelian observation, not a recommendation to remove the claim — it's a recommendation to name it honestly alongside others. |

**The deepest collision** is between "bit-exactness is a natural universal target" and "bit-exactness serves specific use cases that not all users have." The trek's framing implies the first; Phase 2 truth 9 establishes only the second.

**A more honest restatement of the meta-goal:**

> "Tambear supports a spectrum of cross-hardware guarantees. The default is bit-exact — same `.tam` source produces identical fp64 bit patterns on any IEEE-754 compliant ALU, enabling equality testing, content-addressing, and cross-machine caching. Users who don't need this guarantee can select bounded-ULP modes that allow backend-specific optimizations within a declared error bound. The trek focuses on demonstrating the bit-exact path for pure arithmetic and Kingdom-A operations, with the weaker modes deferred to post-trek work."

This reframing DOESN'T weaken the trek. It makes the trek's target *honest* about what it's buying and why, and leaves room for future work that doesn't invalidate the bit-exact path.

---

## Phase 5 — The Aristotelian Move

The highest-leverage action is NOT to weaken the trek's target. Bit-exact cross-hardware is the right summit; what's missing is the *honest framing* of WHY.

**The move:**

**Add a "Guarantee Spectrum" section to the trek plan, explicitly naming what bit-exactness buys, what bounded-ULP would buy, and which user categories need which. Pin bit-exactness as the default. Leave the weaker modes as declared future work. Then — critically — state the bit-exact target as a CONSEQUENCE of the user requirement (content-addressing, equality testing), not as an intrinsic architectural property.**

This move reframes I1–I10 from "rules we must obey" to "rules that deliver a specific user-facing guarantee." The rules stay. The framing changes.

Concretely:

- `trek-plan.md` gets a new section between Part I (what we're building) and Part II (invariants) titled **"Part I.5 — The Guarantee Spectrum and Why We Pick Bit-Exact."**
- The section lists the guarantee spectrum (no-cross-hardware, loose bounded-ULP, tight bounded-ULP, bit-exact-on-arithmetic-bounded-on-transcendental, fully bit-exact).
- It names the user categories for each (prototyping, generic research, tight research, financial/settlement, ML debugging, audit-requiring regulated users).
- It states that the trek targets **bit-exact** because the content-addressing and cross-machine-caching properties ENABLE downstream architecture (the TamSession persistent store, cross-machine cache hits, auditable reproducibility) that the other modes don't.
- It acknowledges bit-exact is the MOST EXPENSIVE point on the spectrum, and that the cost is paid for specific downstream benefits — not because "bit-exact" sounds good.
- It pins weaker modes as explicit FUTURE WORK, not as alternatives-considered-and-rejected. That's important: leaving the door open for a "fast mode" means the architecture doesn't become a prison.

**Why this is high leverage:**

1. **It protects the trek from scope collapse.** When Peak 3 or Peak 6 hits a hard case, someone will argue "well, bounded-ULP would be sufficient here — let's relax." Without the framing, the relaxation sounds reasonable. With the framing, the relaxation is visibly "giving up the content-addressing guarantee," which has downstream effects the arguer may not know about. The framing makes the cost of relaxation visible.

2. **It makes the trek's claim defensible to outsiders.** Right now the claim reads as "we chose bit-exact because it's cool." After the framing, the claim reads as "we chose bit-exact because it enables architecture X, which is load-bearing for our product." That's a professional framing that holds up in technical review.

3. **It aligns the trek with the rest of the project.** The persistent store in `project_persistent_store.md` assumes content-addressing of intermediates. The sharing contract in the Tambear Contract assumes shareable intermediates via TamSession. These depend on bit-exact outputs. The trek plan DOES NOT currently say this — it would benefit from stating it.

4. **It's free to do now.** The move is a one-page addition to the trek plan. It doesn't change any code, doesn't change any invariant, doesn't change any campsite. It just makes the *why* visible.

5. **It leaves a door open.** Some operations (stochastic methods, iterative convergence) may never be bit-exact practically. The framing lets those be "bounded-ULP mode only" without contradicting the trek. Without the framing, those ops are "violations of the architectural claim," which is false but reads as true.

**Why this is the first-principles move:**

Phase 2 truth 9 is "the value of bit-exact to a user is the ability to test for equality, content-address, and cache." That's the irreducible user-facing value. Phase 2 truth 10 is "bit-exact does not imply correct." Together they establish that bit-exactness is a MEANS to specific ends, not an end in itself. The trek currently frames it as an end. The first-principles move is to restate it as a means.

This is a less dramatic move than the I7 and I9 moves, but it's a structurally important one because it changes how the *whole trek* is defended. The other two moves change specific invariants; this one changes the trek's *rationale*. A rationale change is high leverage when the team hits friction — it determines what they do when the going gets hard.

---

## Phase 6 — Recursion: challenge the Phase-5 Move

The Phase-5 Move is:
> Add a "Guarantee Spectrum" section to trek-plan.md that frames bit-exactness as a MEANS to content-addressing / cross-machine caching / audit reproducibility. No code changes, no invariant changes, framing only.

Adding this Move to the assumption list and rerunning the loop.

### Phase 6.1 — Assumption autopsy on the Move

**M1.** That "content-addressing" is itself the deepest reason for the trek. Is it? Content-addressing is valuable IF tambear's downstream architecture depends on it. The Persistent Store design does. Cross-machine cache hits do. Audit reproducibility does. But these are ALL project-internal uses. The Move positions content-addressing as the justification, which makes the trek *self-referential* — "we need bit-exact because we decided our architecture needs it." That's true but circular. A first-principles framing should identify a USER-facing benefit that doesn't depend on our own architecture choices.

**M2.** That the guarantee spectrum is linearly ordered. The Move describes "no claim → loose bounded-ULP → tight bounded-ULP → bit-exact arithmetic / bounded transcendental → fully bit-exact" as a spectrum. But these aren't linearly ordered in *value to users*. A user who needs monotonicity guarantees cares about bounded-ULP ONLY if the bound preserves monotonicity (not automatic). A user who needs content-addressing needs bit-exact ONLY at fp64 precision (not necessarily fp32). The "spectrum" is a team-facing simplification of a multi-dimensional user need space.

**M3.** That framing changes are effective without code changes. Adding a paragraph to a document is the cheapest possible intervention, but also the easiest to ignore. Under pressure, a team may read "bit-exact because content-addressing" as one more invariant rather than as *the* rationale. Framing has leverage only if it becomes load-bearing in code review, in escalation discussions, and in design conversations. A document change alone is necessary but not sufficient.

**M4.** That the trek-plan is the right document for this framing. The trek-plan is specific to the Bit-Exact Trek; it's an expedition plan. But the framing concerns the WHOLE project (tambear's architectural story, not just the trek's). Maybe it belongs in the project's top-level architectural doc, not the trek plan. The trek plan might REFERENCE the top-level framing.

**M5.** That "future work" for weaker modes is a real door, not a dead letter. Most "future work" in software projects never happens. If the trek commits to "bounded-ULP modes are future work," and that future never comes, the spectrum framing becomes empty — it sounded like flexibility but delivered none. Users who eventually need bounded-ULP modes will face "not implemented" and have to build their own, defeating the tambear-is-the-whole-math-library promise.

**M6.** That user categories are actually distinguishable and countable. Phase 5 listed "prototyping, research, tight research, financial/settlement, ML debugging, regulated audit users." That list sounds clean but is fuzzy in practice. Is an ML debugger who tests for regression equivalent to a research user or a regulated user? The categories overlap. Any framing built on them will have edges where categorization is a judgment call.

**M7.** That the downstream architecture (TamSession persistent store, cross-machine cache, etc.) currently uses bit-exact identity. Notebook 013 flagged this as an open question — it needs verification. If TamSession uses approximate hashing (e.g. fp32-truncation hashing), then the content-addressing argument is weaker than the Move assumes, and the framing needs adjustment.

**M8.** That the trek's central claim is the PRIMARY architectural commitment. Per Phase 8 of the I7 deconstruction, decomposition is a SPEED story and bit-exact is a CORRECTNESS story. The trek's claim is on the correctness axis alone. Framing the guarantee spectrum only around bit-exactness leaves decomposition (which governs fusion/speed) un-explained. The framing should probably cover BOTH axes, not just the correctness axis the trek names.

**M9.** That the framing doesn't cost the team. Adding a framing section takes navigator time, review time, approval time, and documentation maintenance time forever. None of these is large, but together they're not zero. A first-principles framing should be able to show the savings (fewer escalations, fewer bad relaxations) justify the maintenance cost.

**M10.** That bit-exact-cross-hardware is the only target *tambear* currently wants. Even within the same project, different subsystems have different needs. The Persistent Store wants bit-exact. The `.discover()` superposition pipelines may not care — they're running *every* method anyway, small numerical variations don't affect the structural fingerprint. The framing should acknowledge that different *subsystems of tambear* want different points on the spectrum, not just different user classes.

### Phase 6.2 — Irreducible truths visible at this level

1. **A framing move's leverage depends on its pressure-test.** The test: when someone proposes relaxing an invariant, does the framing provide a concrete defense that the invariant alone doesn't? If yes, the framing is load-bearing. If no, it's decoration. Phase 5's Move passes this test for I3 (no FMA) — "relaxing I3 forfeits bit-exact arithmetic which forfeits content-addressing" is stronger than "the invariant says so." So the framing has real leverage, but only for specific invariants.

2. **Bit-exact is ONE benefit of a deeper property.** The deeper property is: **the IR is precise enough to pin the exact op sequence executed on any backend.** That precision-of-specification is what bit-exact cross-hardware is built on. It's ALSO what enables several other properties: auditable provenance, formal verification (if we ever do it), deterministic debugging across machines, and content-addressing. Bit-exact is the user-visible symptom; IR-precision is the cause. The framing should name the cause, not just a symptom.

3. **The Persistent Store DOES require bit-exact AT fp64 PRECISION for content-addressing to work across backends.** If the store's hash function operates on the raw fp64 bytes, a 1-ULP difference produces a different hash. If the store uses a quantized hash (e.g., truncate to fp32 before hashing), it tolerates sub-ULP variation but loses resolution. Both are viable designs. The trek assumes the first. Whether that's the right design is a Persistent Store question, not an Aristotle question — but the answer affects the framing's weight.

4. **The trek is about the CORRECTNESS axis. Decomposition is about the SPEED axis.** These are cleanly separable (Phase 8 of Notebook 011 established this). The guarantee spectrum concerns correctness; the decomposition/fusion story concerns speed. A good framing names both axes, because users evaluating tambear will weigh both.

5. **Future-work commitments are only real when there's a test that would fail once the future work happens.** "Bounded-ULP modes are future work" is empty unless there's a sentinel kernel that will stop passing bit-exact tests when bounded-ULP mode lands (signaling "now we need to implement the weaker mode to not break this user"). The pointers forward must be concrete.

6. **The framing-change-only Move is actually a sub-move; a full Move would also include a mechanism for enforcement.** Enforcement: a pre-merge checklist item ("invariant-relaxation PRs must cite which user-facing property is being traded away, per the Guarantee Spectrum"). Without the enforcement mechanism, the framing is aspirational.

7. **The deconstruction of the meta-goal is less likely to yield "refactor the trek" findings and more likely to yield "refactor how the trek is defended" findings.** Phase 5 already landed on that. Phase 6 is confirming it: the target is right; the framing needs to be honest and enforceable, not revised.

### Phase 6.3 — Reconstructions of the Move

**Move v1 (original).** Add a "Guarantee Spectrum" section to trek-plan.md. Framing only.

**Move v2.** v1 PLUS: move the framing to a project-level architectural doc (not just the trek plan) since the story concerns tambear as a whole. The trek plan references it.

**Move v3.** v2 PLUS: name BOTH axes (correctness via bit-exact, speed via decomposition/fusion). The guarantee spectrum becomes a 2D map, not a 1D spectrum.

**Move v4.** v3 PLUS: frame the *cause* as IR precision (the source-level commitment to pinning op sequences exactly), and bit-exact cross-hardware as a downstream consequence, not the primary target. This renames the story from "bit-exact trek" to "IR-precision trek with bit-exact consequences."

**Move v5 (recommend).** v4 PLUS: enforcement mechanism. A **"Guarantee Ledger"** document that maps each invariant (I1–I10 + anything new) to the specific user-facing property it protects. When someone proposes relaxing an invariant, the ledger is the mandatory checklist — the proposer must name which ledger-listed property they're trading. The ledger becomes part of the trek README and is consulted in every escalation. This is what makes the framing *enforceable*.

Concretely, the ledger entry for I3 (no FMA contraction) would read:

| Invariant | Protects | Downstream user-facing benefit | Cost of relaxation |
|---|---|---|---|
| I3 — No FMA contraction | Bit-exact cross-hardware for arithmetic kernels | Content-addressing of TamSession intermediates, cross-machine cache hits, reproducible audit trails | For this kernel: loses content-addressing on the output; any downstream consumer that hashes the output breaks. For the architecture: signals to future backends that FMA contraction is negotiable, which opens the door to silent drift elsewhere. |

Every invariant relaxation proposal must come with this row filled in.

### Phase 6.4 — The refined Aristotelian Move

> **Move v5 (final):** Add a **Guarantee Ledger** to the trek and reference it from the project's top-level architectural doc. The ledger is a table mapping each invariant (and each future invariant) to the specific user-facing property it protects and the exact cost of relaxing it. The ledger covers both axes — correctness (bit-exact cross-hardware) and speed (fusion via decomposition) — and is the mandatory checklist for any invariant-relaxation proposal. The source-level cause is named as IR PRECISION ("the IR is precise enough to pin the exact op sequence on any backend"), with bit-exact cross-hardware as a downstream consequence rather than the primary target. Future-work modes (bounded-ULP, fp32, approximate computing) are named with sentinel tests that would fail the moment those modes became necessary.

This is what to route to navigator (as trek-plan owner).

---

## Phase 7 — Stability check

Run one more pass, adding v5 to the assumption list.

**v5's residual assumptions:**
- That a Guarantee Ledger is maintainable as new invariants are added.
- That the ledger's "cost of relaxation" column can be filled in meaningfully for every invariant.
- That proposers will actually consult the ledger rather than writing a relaxation proposal and then filling in whatever justifies it post-hoc.
- That "IR precision" is a well-defined concept — what counts as "precise enough"?

**Autopsy findings:**

- **Ledger maintenance.** Each new invariant gets one row. At current size (I1–I10 plus I7′ and I9′), the ledger is 12 rows. Maintenance is trivial.
- **Cost-of-relaxation column.** For most invariants the cost is enumerable: which downstream property breaks, which subsystem loses a guarantee, which user class is affected. For a few (I2 — no vendor compiler), the cost is "we cede control of the instruction sequence to NVIDIA/Khronos/Microsoft," which is a meta-cost. Either column fills in for any invariant.
- **Post-hoc justification risk.** Real risk. Mitigation: the ledger is consulted by REVIEWERS, not just proposers. A proposer who writes a garbage justification still has to survive review against the ledger. The ledger becomes an institutional memory that outlasts individual reviewer judgment.
- **"IR precision" well-definedness.** The concept is: "the IR can express exactly one meaning for each written kernel, with no freedom for the compiler to reinterpret." Concrete tests: (a) no op has implementation-defined semantics; (b) all constants are bit-exact literals; (c) all ordering is declared (per I7′); (d) all rounding is explicit. If the IR passes all four, it's "precise enough." This is checkable.

**Stability verdict:** v5 is concrete. No new truths emerge. The deconstruction is **stable at v5**. No further recursion warranted.

**Cross-target observation (second time I'm noting this):** All three of my Moves have now converged on a *named-registry-or-ledger pattern*:

- I7′ v5 → `order_strategies/` registry
- I9′ v4 → `oracles/` registry
- Meta-goal v5 → Guarantee Ledger

Three artifacts with the same shape: named entries, formal content, reviewable at merge-time, consulted during escalation, owned by a specific role. This is the engineering pattern that emerges when tacit architectural knowledge is made explicit. I suspect it's not a coincidence — it may be the *right* way to make this class of architectural commitment inspectable. Worth flagging to navigator as a finding about the team's documentation pattern.

---

## Phase 8 — Forced Rejection

Forcibly reject the entire deconstruction. What if "cross-hardware guarantees" is the wrong axis entirely?

### Alternative framing 1: cross-TIME instead of cross-HARDWARE

Suppose tambear's primary guarantee is **bit-exact across runs, sessions, and days — but NOT across hardware**. Each backend is its own world, but within a backend, the same kernel produces the same bits forever. Cross-backend bit-exactness is a consequence of each backend being internally deterministic PLUS all backends using the same IR, but it's not the stated goal.

What does the project look like?

- **The Persistent Store still works** — at fp64 bit granularity, for each backend, within a single machine. Cross-machine caches become backend-specific.
- **Audit reproducibility still works** on a single machine over time, which is what most audits actually want.
- **Cross-machine cache hits become rarer** but not impossible: two machines running the same backend (CPU, or the same GPU model) still share cache.
- **The trek is easier.** Peak 6 (determinism within a backend) is all that's needed. Peaks 3 and 7 can be more flexible — they don't need to produce BIT-identical outputs, just DETERMINISTIC outputs per backend.
- **The architectural claim is weaker** — less novel, less publishable, less of a differentiator.

### Alternative framing 2: cross-IR instead of cross-hardware

Suppose the primary guarantee is **bit-exact across IR-compatible compilations of the same source**, regardless of which backend runs it. The IR is the anchor; the backends are interchangeable lowering strategies.

What does this look like? It's actually what the trek is DOING, described differently. The anchor is the IR, not the hardware. Bit-exactness cross-hardware is a consequence of (a) the IR pins the sequence, (b) every backend lowers faithfully, (c) hardware is IEEE-754 compliant for the ops used. The guarantee is *compositional*: IR-precision + faithful-lowering + IEEE-754-compliance → bit-exact output.

The framing change: name the guarantee as "IR-faithful lowering to bit-exact execution" rather than "bit-exact cross-hardware." The first is a compositional claim; the second is a consequence.

### Alternative framing 3: no cross-hardware claim at all; "tambear is a CPU library, period"

Suppose tambear abandons the cross-hardware story. It ships as a CPU library with optional GPU acceleration, where the GPU is a PERFORMANCE optimization and CPU is the ground truth. Cross-backend diffs aren't required to be bit-exact; they're required to be "within ULP tolerance."

- **Dramatically easier** — removes whole categories of work (Peak 6, parts of Peak 2, parts of Peak 3/7).
- **Still useful for most users.** Research, prototyping, ML — none of these need cross-hardware bit-exactness.
- **Loses the "architectural claim" that makes tambear distinctive.** The whole point of the trek is to demonstrate something no other library can. Abandoning cross-hardware is abandoning the distinctive claim.
- **Content-addressing becomes single-backend.** TamSession stores intermediates per-backend, not per-kernel. Cache hits only happen on the same backend the data was computed on. Practical for most workflows; less useful for distributed teams.

### What forced rejection reveals

**Reframing 2 (cross-IR) is actually the SAME as the current trek, clearly stated.** The trek calls it "cross-hardware" but the anchor is the IR, not the hardware. "Cross-hardware" is a consequence name; "cross-IR" is a cause name. Using the cause name would clarify what's actually guaranteed:

> **Given:** (a) the .tam source, (b) a backend that faithfully lowers .tam to its target ISA, (c) hardware that implements IEEE-754 compliantly for the ops used, **then:** the output is bit-exact across all such (backend, hardware) pairs.

That's a compositional, defensible, first-principles claim. It's what the trek is already proving. The team isn't saying it this way because "bit-exact cross-hardware" is punchier — but the punchy version hides the preconditions (IEEE-754 compliance, faithful lowering). ESC-001 exposed one such precondition (subnormals under `shaderDenormPreserveFloat64`); there will be others.

**The unseen first principle surfaced by forced rejection:**

> **The trek's claim is a COMPOSITIONAL guarantee, not a unilateral one.** It holds when three conditions hold: IR-precision, faithful-lowering, and IEEE-754-compliance-for-the-ops-used. Any of the three failing breaks the claim. The trek's current framing names only the IR-precision side (the part we control). The other two are implicit preconditions. Making them explicit is honest and prevents ESC-001-style surprises from being read as "the claim failed."

**Reframe for v5 Move:**

The Guarantee Ledger should have a **"Preconditions"** section naming the three conditions (IR-precision, faithful-lowering, IEEE-754-compliance-for-used-ops). Each invariant then protects ONE of the three. I3 (no FMA) protects IR-precision. I1 (no vendor libm) protects faithful-lowering. ESC-001's subnormal issue belongs under IEEE-754-compliance-for-used-ops. This structure makes EVERY invariant traceable to one of three root causes, and every escalation traceable to one of three failure modes.

### The refined v5 after Phase 8

> **Move v5 (Phase-8 refined):** The Guarantee Ledger's top-level structure is the three preconditions of the compositional claim: **IR-precision**, **faithful-lowering**, and **IEEE-754-compliance-for-the-ops-used**. Each invariant is classified as protecting one precondition. The ledger's cost-of-relaxation column names which precondition is violated and what consequence follows. The trek's central claim is restated:
>
> "Given .tam source, a backend that faithfully lowers .tam to its target ISA, and hardware that implements IEEE-754 for the ops in use, the output is bit-exact across all such (backend, hardware) pairs. Tambear owns the first condition via the IR; the second via our backend implementations; the third is a declared hardware prerequisite documented per device."

This is the final Move. It's longer than the original framing pitch but structurally far more defensible.

---

## Status as of 2026-04-12

- Phases 1–8 drafted.
- **Final Move (v5, Phase-8 refined):** Guarantee Ledger with three-precondition structure (IR-precision, faithful-lowering, IEEE-754-compliance-for-used-ops). Each invariant protects one precondition. Central claim restated as a compositional guarantee.
- **Cross-target observation:** All three Aristotle Moves converged on the same pattern — named registry/ledger of formal-spec entries with capability metadata and review-time enforcement. `order_strategies/` registry, `oracles/` registry, `guarantees/` ledger. Same shape. Same lifecycle.
- **ESC-001 relevance:** The Vulkan subnormal escalation is a violation of precondition 3 (IEEE-754-compliance-for-used-ops). Under v5, the navigator's Option 2 decision (narrow the claim to normal fp64 with subnormal as device prerequisite) becomes the canonical form: the architectural claim holds for the ops in use, subnormal outputs are a separate op class with a stronger hardware prerequisite.
- Second pass (Phase 7) found stability. No further recursion warranted.
- Ready to communicate to navigator.
