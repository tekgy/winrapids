# Target: Why f64 as the base precision?

**Deconstructor:** Aristotle
**Date opened:** 2026-04-12
**Status:** Phases 1–8 drafted.

Assumption under deconstruction:

> Tambear's numerical type is fp64 (IEEE-754 binary64). The `.tam` IR defines one fp type (`f64`). All primitives accept and produce `f64`. All accumulators, all storage, all intermediates are `f64`. No other precision is a first-class type in Phase 1.

Navigator's framing for this deconstruction (2026-04-12 routing message):

> "fp32 would eliminate the subnormal ESC-001 problem, mixed precision might unlock smaller state for RFA, posits are a genuine alternative for some use cases."

So the deconstruction has three specific angles to address in Phase 3. Plus the broader question navigator named: "what would it mean to audit the f64 default?"

This is a different shape of deconstruction from the first three targets. I7 / I9 / meta-goal were about making tacit *invariants* explicit. f64 is about challenging a tacit *engineering convention* — one that's been the default in scientific computing for 40+ years. Questioning it seriously is rare, which means the deconstruction has to dig carefully.

---

## Phase 1 — Assumption Autopsy

Assumptions stacked inside "f64 is the base precision":

1. **That there IS a base precision.** The phrase "base precision" assumes the project has one universal numerical type. Alternative: per-quantity or per-operation precision, where `f32`, `f64`, `f128`, and specialized formats coexist and the compiler infers or the user declares per use-site.

2. **That IEEE-754 binary64 is the right choice of fp64.** IEEE-754 has one fp64; but more generally "64-bit floating point" could be a different format — logarithmic, posit-64, custom. The assumption packages two things: "fp" (as opposed to integer, fixed-point, log, exact-rational) and "64-bit" (as opposed to 32, 128, 16, or variable).

3. **That the precision we need is enough.** fp64 gives ~15-17 decimal digits. Is that enough for tambear's use cases? Statistical operations on fp64 inputs stably give ~12-14 digits of accuracy after a few arithmetic steps. Iterative algorithms can lose more. Multi-scale problems may need much more. The assumption is that 15-17 digits at the storage level is adequate; it may not be for all workloads.

4. **That the precision we have is not excessive.** fp64 takes 8 bytes per value. For large arrays of inputs where only a few digits of precision are needed, fp32 (4 bytes) would halve memory traffic, and memory bandwidth is often the bottleneck on modern hardware. The assumption packages "we can afford fp64 everywhere" as a baseline.

5. **That one precision for all quantities is the right mental model.** In reality, some quantities in a pipeline have very different dynamic ranges and precision needs. A count is an integer; a probability is in [0, 1]; a standard deviation is a non-negative reduced quantity; a scale parameter is a positive multiplicative quantity; a log-likelihood is a large negative quantity with small differences that matter. Using one type for all of them is a simplification, not a fit.

6. **That fp64 is portable across every ALU we care about.** The trek's meta-goal is "any ALU through its driver." Not every ALU supports fp64 equally. CPU: yes, via native instructions. NVIDIA Blackwell: yes, with fp64 tensor ops. Apple M-series: yes, but slower. AMD consumer GPUs: often fp64 at 1/32 or 1/64 of fp32 throughput. NPUs and accelerators (Groq, Tenstorrent, Cerebras, TPU): often NO fp64 at all, or only fp32/bf16/fp8. The assumption "fp64 is everywhere" is true for current CPUs and some current GPUs and is NOT true for the accelerator class the trek claims to support.

7. **That the cost of fp64 is uniform across operations.** fp64 add/mul on a CPU is roughly 1 cycle. fp64 div is 20-40 cycles. fp64 transcendentals depend entirely on the libm implementation. On GPUs, fp64 throughput is sometimes 1/32 of fp32 or worse. "Just use fp64" has different cost implications per op, and the assumption flattens this.

8. **That fp64 is sufficient for all reductions.** A sum of a billion fp64 values can accumulate ULP-scale error. A sum of a trillion values can lose ALL precision if the values are of similar magnitudes and opposite signs. The assumption "fp64 is enough" is false in these regimes without compensated summation, Kahan, or RFA (reproducible floating-point accumulator). Peak 6's RFA work is the concrete acknowledgment of this.

9. **That fp64's subnormal behavior is acceptable.** ESC-001 surfaced that Vulkan's fp64 subnormal handling is implementation-defined. The team's Option 2 resolution (narrow the claim to normal fp64 with a feature-bit prerequisite) is an acknowledgment that fp64 subnormals ARE a real problem. The assumption "fp64 is a well-defined type everywhere" is false at the subnormal boundary.

10. **That the decision to use fp64 is independent of the decision to target scientific computing.** The project is framed as a scientific computing toolkit. Scientific computing has historically used fp64 as the default. But tambear's actual use cases (market atlas signal farming, financial tick processing, spectral analysis, music theory, graph algorithms) each have different precision needs. Accepting the scientific-computing default without evaluating each use case is inheriting a convention.

Sub-assumptions also worth flagging:
- **"fp64 is what scipy uses."** True, but scipy's fp64 choice is itself inherited from Fortran's default (REAL*8). Inheritance chains are not justification.
- **"The user expects fp64."** Some users do. ML users usually expect fp32 or bf16. Most statistical users don't have a strong opinion. "User expectation" is a surface under which individual preferences vary.
- **"fp64 is the common denominator."** Not quite. On some accelerators the common denominator is fp32 or bf16. On some legacy CPUs it's fp80 (x87). fp64 is the common denominator *for current desktop/server CPUs and high-end GPUs*, which is a narrower claim.

---

## Phase 2 — Irreducible Truths

Stripping back to what's undeniable about numerical representation:

1. **A finite-bit representation cannot represent all real numbers.** Any fixed-size numeric type (including fp64, posit64, f128) has a countable set of representable values. Most real numbers are not in that set.

2. **Rounding is unavoidable.** When a computed value is not representable, the result must be rounded to a nearby representable value. Rounding introduces error.

3. **Rounding error accumulates.** A sequence of operations each introducing ≤ 1 ULP of error can produce a final answer with error much greater than 1 ULP, depending on the problem's conditioning and the computation's structure.

4. **Different representations trade off range, precision, and density.** fp64 has wide range and high precision. fp32 has narrower range and lower precision but uses half the storage. Fixed-point has narrow range and uniform precision. Posits have variable precision (high near 1.0, lower near 0 and ∞). Log-scale representations lose additive structure but preserve multiplicative structure. Each is a different point in a trade-off space.

5. **The precision a computation needs is problem-dependent.** A comparison of two probabilities near 0.5 needs few digits. An iterative solver converging to a fixed point needs enough digits that the iteration is stable. A multi-scale problem with dynamic ranges spanning 30+ orders of magnitude needs a range larger than fp64's ~10^±308 if it's at the edges.

6. **Precision can be chosen per-operation or per-quantity, not just per-type.** The common approach "one type everywhere" is a simplification. In principle, a compiler can infer or a user can declare a different precision per operation site, mixing fp32 and fp64 (or posit-32 and posit-64, or any combination) in the same program.

7. **Cross-backend bit-exactness is IEEE-754-specific.** Bit-exact fp64 addition produces the same bits on every IEEE-754 compliant ALU. That's the trek's load-bearing property. If we chose a non-IEEE-754 format (posits, custom floats), bit-exact cross-hardware requires every ALU to agree on the non-IEEE-754 semantics — which is a weaker guarantee in the current hardware landscape, because very few ALUs natively support posits.

8. **The memory-bandwidth cost of a type is proportional to its size.** fp64 is 8 bytes; fp32 is 4 bytes; bf16 is 2 bytes; fp8 is 1 byte. Memory bandwidth is often the bottleneck on modern hardware (CPUs, GPUs, accelerators). A computation bound by memory bandwidth runs 2x faster in fp32 vs fp64, and 4x faster in bf16.

9. **The arithmetic-throughput cost of a type is hardware-dependent.** Modern GPUs have different fp32 and fp64 throughput ratios: NVIDIA Blackwell has good fp64 support; Apple M-series has acceptable fp64; AMD consumer GPUs have poor fp64 (1/32 of fp32 or worse). Accelerators often have NO fp64.

10. **For any chosen representation, the compositional claim about bit-exactness holds when the three preconditions hold.** The bit-exact cross-hardware story does NOT require fp64. It requires "a well-defined type whose arithmetic is specified and whose hardware implementations are compliant." fp64 happens to be the type with the widest compliant hardware support today. That's a contingent fact about hardware, not a structural property of tambear's architecture.

---

## Phase 3 — Reconstruction from Zero

Given Phase 2's truths, what are the plausible precision strategies for tambear?

### 1. **fp64 everywhere (current choice).**
One type, used for all data, all intermediates, all reductions. Simple, widely supported, well-understood.

- **Pro:** Maximum IEEE-754 portability. Well-studied. Existing libm work applies directly. Scientific computing community expects it.
- **Con:** 8 bytes everywhere even when 4 would suffice (memory bandwidth). Limited on some accelerators. Subnormal issues (ESC-001).

### 2. **fp32 everywhere.**
Same as current but half the memory and often 2x-32x the throughput on GPUs.

- **Pro:** 2x less memory traffic. Faster on most GPUs. **No ESC-001-class subnormal issue** (fp32 subnormals are better-behaved on most hardware). Accelerators almost universally support fp32.
- **Con:** Only ~7 decimal digits of precision. Statistical operations on fp32 lose precision rapidly — a sum of 1 million fp32 values can lose half its digits. Many tambear use cases (high-precision statistics, long reductions, iterative solvers) would fail or produce unreliable answers.

### 3. **fp32 for storage, fp64 for computation.**
Data is stored as fp32, converted to fp64 on load, operated on in fp64, converted back to fp32 on store.

- **Pro:** Halves memory traffic (the bottleneck on most workloads) while preserving fp64-precision intermediate computation. Common pattern in HPC.
- **Con:** Two conversion ops per load/store add overhead. Users must understand the precision is fp32 at the boundary and fp64 internally. Composition of multiple kernels loses precision at each fp32→fp64→fp32 cycle.

### 4. **Mixed precision per-tensor.**
Each tensor declares its precision. A computation pipeline might use fp32 for raw inputs, fp64 for accumulators, fp32 for outputs. The compiler inserts conversions.

- **Pro:** Closer to what ML frameworks (PyTorch, JAX) do for training and inference. Lets memory-bound workloads win on bandwidth and compute-bound workloads win on precision. The compiler can optimize conversions.
- **Con:** Complexity — the `.tam` IR has to support multiple fp types and per-op conversion. The user has to think about precision. The sharing contract (TamSession) has to track precision per intermediate. The bit-exact claim becomes more complex (bit-exact at declared precision, not bit-exact across precisions).

### 5. **Per-quantity precision (compiler-inferred).**
The user writes `.tam` code without thinking about precision. The compiler infers per-operation precision based on error analysis, conditioning, and hardware availability. This is the ambition of projects like Precimonious and FloatSan.

- **Pro:** Users don't think about precision; they get the right precision automatically. Memory-bound kernels get fp32 where safe; compute-bound kernels get fp64 where needed.
- **Con:** Compiler inference is a research problem and is not reliable for all kinds of computation. Automated precision selection produces numerical behavior the user can't predict. Cross-backend bit-exactness is much harder because different backends may infer different precisions.

### 6. **Posit arithmetic as the default.**
Posits (John Gustafson's Unum v3) are a non-IEEE numeric format with variable precision and tapered range. They give more precision near 1.0 and less near the extremes.

- **Pro:** Better precision for "normal"-range values. Single format gives you a wider useful range than IEEE. No subnormal corner cases. No NaN corner cases (posits have ONE NaR). Some evidence they enable simpler hardware.
- **Con:** No hardware support in any shipping commercial processor as of 2026. Software emulation is 10x-50x slower than native fp64. The tambear speed story evaporates. Bit-exact cross-hardware requires every ALU to agree on posit semantics, which is even harder than agreeing on IEEE-754 (because few/no ALUs natively implement posits).

### 7. **Interval arithmetic for every computation.**
Every value is a pair (lo, hi) representing an interval containing the true value. Operations propagate intervals.

- **Pro:** Provable correctness bounds. Every answer comes with a rigorous error bound. Catches numerical instability automatically.
- **Con:** 2x the memory. Intervals can blow up (become very wide) rapidly on iterative computations. Many statistical operations (dot products, matrix inverses) produce intervals so wide they're useless. Specialized applications only.

### 8. **Arbitrary-precision arithmetic (mpmath-style) as the default.**
Every value carries its own precision setting. Operations dynamically allocate.

- **Pro:** No precision loss anywhere. Users set the precision they want, the library delivers it.
- **Con:** 100x-1000x slower than fp64. Not a realistic default for most workloads. Useful as an oracle (which is what mpmath already is in I9).

### 9. **IEEE-754 binary128 (fp128) everywhere.**
Same trade-off as current but at higher precision and cost.

- **Pro:** ~34 decimal digits. Eliminates most precision-loss concerns. Stable for multi-scale problems.
- **Con:** Almost no hardware support (not even NVIDIA has native fp128). Software emulation is slow (~10x fp64). Memory is 2x fp64 (16 bytes per value). The ESC-001 subnormal issue is worse, not better. Not a serious option.

### 10. **A per-op precision hint as a first-class IR concept, with fp64 as the default.**
The `.tam` IR declares each op's precision: `fadd.f64`, `fadd.f32`, `fadd.mixed(f32, f64)`. Default is fp64 for compatibility with current recipes. Users can override per op or per kernel. The compiler lowers each to the right backend instruction. The Guarantee Ledger's P3 (IEEE-754 compliance) extends to "IEEE-754 compliance for each declared precision of each op."

- **Pro:** Preserves the current story (fp64 default) while enabling future mixed-precision work without a format bump. Matches how the OrderStrategy registry approach handles the total-order question (name it per-op, let the user/compiler choose). Per-op declared precision is the analog of per-op declared order.
- **Con:** More IR surface area. More work for backends to support multiple precisions. The sharing contract has to track precision per intermediate. The bit-exact claim is per-precision, not cross-precision.

---

## Phase 4 — Assumption vs Truth Map

| Assumption | Matching truth | Where they collide |
|---|---|---|
| "fp64 is the right choice because it's standard" | T4: different representations trade off range / precision / density | fp64 is ONE point in the trade-off space, not the top. Scientific-computing convention adopted it. That's an inherited choice, not a derived one. |
| "fp64 is the common denominator across hardware" | T9: arithmetic throughput is hardware-dependent, accelerators often have no fp64 | fp64 is the common denominator for CURRENT desktop/server CPUs and high-end GPUs. It is NOT the common denominator for accelerators. The trek's "any ALU" claim is narrower than stated if interpreted strictly. |
| "fp64 gives enough precision" | T5: precision needs are problem-dependent | True for most tambear use cases, FALSE for billion+ element reductions without compensated summation. Peak 6 RFA is the team's acknowledgment of this. |
| "fp64 gives not too much precision" | T8: memory bandwidth cost is proportional to type size | Not quite — fp32 would halve memory traffic, which is the bottleneck on most workloads. "fp64 is not wasteful" is false for memory-bound kernels (which includes most recipe-style workloads). |
| "one type everywhere is simpler" | T6: precision can be chosen per-op or per-quantity | True for implementation simplicity, but simplicity at the cost of 2x memory for most recipes is expensive simplicity. |
| "fp64 is the IEEE-754 default" | T10: compositional claim holds for any well-defined type with compliant hardware | True, but the compositional claim is NOT fp64-specific. The claim is about faithful lowering and precondition-3 compliance. It would hold just as well for fp32 or a future posit64-compliant hardware set — with a different scope statement. |

**The deepest collision:** Assumption 1 (fp64 is the right choice) vs Truth 4 (representations are trade-offs) vs Truth 6 (precision can be per-op). Put together: **the "one type, universally fp64" choice is a simplification that was reasonable when the team had limited engineering budget but is not structurally necessary. Phase 2+ can support mixed precision without breaking the compositional claim, as long as each precision is treated as its own well-defined point.**

**Restated assumption:**

> **fp64 is tambear's Phase 1 default precision because (a) it is the common denominator across the hardware the team currently targets (CPU, NVIDIA Blackwell), (b) it has the best-studied and most widely-supported IEEE-754 semantics, (c) it provides enough precision for every Phase 1 recipe, and (d) committing to a single type reduces Phase 1 engineering complexity. Future phases may introduce mixed precision per-op when (a) memory bandwidth savings justify the complexity, (b) specific accelerators require it, or (c) specific recipes would benefit from higher or lower precision.**

That's an honest statement. It's not "fp64 is the right answer forever" — it's "fp64 is the right answer for Phase 1 with the engineering budget we have, and the door is open for change."

---

## Phase 5 — The Aristotelian Move

The highest-leverage action:

**Treat fp64 as Phase 1's declared default, not as tambear's universal type. Add a `precision` field to the IR's fp op definitions at design time, even if Phase 1 only supports `f64` and the field is effectively monomorphic. Preserve the extension path for Phase 2+ mixed precision.**

Concretely:

- The `.tam` IR defines fp ops with a precision parameter: `fadd.f64`, `fadd.f32`, etc. as distinct instructions, or `fadd(prec, a, b)` as a parameterized op.
- Phase 1 implementations only emit `fadd.f64`. Other precisions parse but fail at lowering ("precision f32 not supported by backend PTX in Phase 1"). Capability matrix extension: each backend declares which precisions it supports (currently: all backends declare fp64 only).
- Storage types are also typed: `buf<f64>` is distinct from `buf<f32>`. Phase 1 only uses `buf<f64>`.
- The sharing contract (TamSession) tracks precision per intermediate. Two intermediates at different precisions are NOT the same cached value even if they represent the same mathematical quantity.
- The Guarantee Ledger's P3 entry becomes "IEEE-754 compliance for each declared precision of each op used," extending naturally.

**Why this is high leverage:**

1. It preserves Phase 1 simplicity (single precision) while making Phase 2+ extension additive, not corrective. The parallel to I7′ is exact: the IR names the concept (precision) even when Phase 1 only uses one value (f64), just as I7′ names the concept (order strategy) with Phase 1 using `SequentialLeft` and `TreeFixedFanout(u32)`.
2. It closes a hidden assumption. Right now "precision" is implicit in the type system — everything is `f64`. Making it explicit in the IR means that in Phase 2, the decision to add `f32` is just adding a registry entry, not bumping the format.
3. It extends the three-registry convergence pattern in a clean way. OrderStrategy registry is named precision strategies for reductions. Oracle registry is named verification methods. Guarantee Ledger is named invariants with precondition mapping. A Precision registry (implicit for Phase 1, explicit for Phase 2+) would be the fourth such artifact. Navigator flagged that a "fourth registry" might be formalized as a team convention; precision is a strong candidate.
4. It's free to do now. Pathmaker's IR already has typed ops (`fadd.f64`). Formalizing the precision parameter is a type-system tidy-up, not a new design.
5. It engages with navigator's three flagged questions (ESC-001-subnormal-elimination, RFA state size, posits as alternative) without committing to any of them. They become design options under the parameterized-precision model, not architectural decisions under the monomorphic model.

**Why this is the first-principles move:**

Because Phase 2 truths 4 and 6 ("representations are trade-offs; precision can be per-op") establish that the choice of "one precision everywhere" is a simplification, not a derived result. First-principles thinking says: name the choice as a parameter, pick the Phase 1 value, leave the door open for Phase 2 refinement. That's what I7′ does for order strategy. That's what the Guarantee Ledger does for invariants. That's what this Move does for precision.

Importantly, **this Move does NOT change tambear's current numerical behavior.** Phase 1 kernels still use fp64 exclusively. The Move is about naming what's currently implicit in the type system.

**Recursion check — what does this Move assume?**

1. That "precision" is a well-defined axis orthogonal to other IR concepts. It mostly is, but there's subtle coupling with the OrderStrategy registry: a `TreeFixedFanout(4)` reduction at fp32 has different numerical behavior than the same strategy at fp64. The registry entry's reference implementation needs to be aware of precision.
2. That per-op precision is expressible in the `.tam` IR's current shape. Probably yes — pathmaker's IR already has typed ops, so parameterizing by precision is straightforward.
3. That the user (or the Phase 2 compiler) knows which precision to pick. This is the hard part — and it's why auto-selection (Option 5 in Phase 3) is a research problem. The Phase 2 version of this Move should ALSO include a declaration discipline for who picks precision and when.

---

## Phase 6 — Recursion: challenge the Phase-5 Move itself

Adding the Move to the assumption list and running the loop.

**M1.** That adding a precision parameter to every fp op is cheap. It isn't, if it touches the backend emitters. PTX, SPIR-V, and CPU interpreter each have to handle precision dispatch per op. For Phase 1 with only fp64 it's trivial; for Phase 2 with mixed precision it's real work.

**M2.** That "the door is left open" translates into actual Phase 2 work happening. Every previous Move has used "future work" or "extension path" language; each carries the risk of the future never arriving. Mitigation: the Move should ship with a concrete triggering condition for Phase 2 expansion (e.g., "when the first accelerator target demands fp32" or "when a specific recipe's memory bandwidth cost exceeds a measured threshold").

**M3.** That the precision registry (if we formalize it) has the same lifecycle as the other three registries. It might not — precision is more like a type system feature than a named artifact. The other three registries enumerate named entries that the user/designer chooses from; precision is either a built-in type (`f64`, `f32`) or a library-extended type. Naming it as a "registry" may be forcing the pattern.

**M4.** That TamSession can track precision without performance cost. It can, but the hash key for a shared intermediate extends from "op signature + inputs" to "op signature + inputs + precision". More bits in the cache key. Not free, but minor.

**M5.** That the `buf<f64>` → `buf<f32>` move is a simple type parameter swap. It's not. fp32 storage has different alignment, different SIMD width, different vectorization opportunities, and different NaN/Inf layout. Generalizing the storage type is nontrivial in the lowering pass.

**M6.** That the compositional claim is preserved per-precision. It is, but each precision has its own preconditions chain: P1 (IR-precision for fp32), P2 (faithful-lowering for fp32), P3 (IEEE-754 compliance for fp32 ops on the target). The Guarantee Ledger rows would need per-precision instantiation. That's a maintenance cost.

**M7.** That navigator's three flagged questions (ESC-001, RFA, posits) are all answered by the same precision parameterization. They're not:
- ESC-001 is a subnormal issue, specific to fp64 on Vulkan. fp32 avoids it because fp32 subnormals are better-handled on most hardware. But parameterizing by precision doesn't SOLVE ESC-001 — it lets users OPT OUT of it by declaring fp32 on kernels that don't need fp64 precision.
- RFA state size is about how wide the accumulator is. Parameterizing the fp type of the accumulator (vs the input) would let RFA be fp32 accumulator + fp64 compensated terms, which is a different optimization from "per-op precision." It's precision-aware accumulator design, which is adjacent to but not the same as the Phase-5 Move.
- Posits are a fundamentally different numeric format, not a precision of IEEE-754 fp. Parameterizing IEEE-754 fp precision doesn't open the door to posits. Supporting posits would require a different type system axis: "numeric format" (IEEE-754 vs posit vs log vs ...) in addition to "precision."

So the Move addresses ESC-001 (via opt-out) and RFA (indirectly, via precision-aware accumulators), but NOT posits. Posits need a separate deconstruction or a separate axis.

**M8.** That Phase 1 engineering capacity is the limiting factor. It might be — pathmaker's plate is full, adding a precision parameter is work that pulls from the same budget. If the parameter adds even a small amount of implementation effort per op, it's effort that would otherwise go to other Phase 1 campsites.

### Refined Move — v2

> **Phase 5 Move v2:** Preserve fp64 as Phase 1's only implemented precision, AND make the `.tam` IR parameterize fp ops by precision from the start, AND add `numeric_format` as a separate future-work axis (IEEE-754 now, posits/log/custom later) that's NOT parameterized in Phase 1 but is flagged as a known extension point. The Guarantee Ledger's P3 column becomes "IEEE-754-compliance-for-the-ops-AND-precisions-used" for forward-compatibility. Triggering conditions for Phase 2 mixed precision: (a) first accelerator target that lacks fp64 hardware, (b) first Phase 1 recipe whose memory-bandwidth cost exceeds a measured threshold on production workloads, (c) first user request for explicit fp32 opt-in for memory reasons. Any of the three triggers Phase 2 mixed-precision campsites.

The "numeric_format" axis as separate-future-work is the honest scoping. The Move now addresses two of navigator's three questions (ESC-001 opt-out, RFA accumulator flexibility) and explicitly defers the third (posits) to a separate axis.

---

## Phase 7 — Stability check

Run one more pass, adding v2's structure.

**v2 residual assumptions:**
- That "numeric_format" as a separate axis is more than a label.
- That the triggering conditions for Phase 2 mixed precision are measurable.
- That Phase 2 mixed precision can land without disrupting Phase 1 correctness guarantees.

**Autopsy findings:**

- **Numeric_format as a real axis.** It is IF the `.tam` IR eventually has ops like `posit_fadd` alongside `fadd.f64`. For Phase 1, "numeric_format" is IEEE-754 and the axis is dormant. The Move commits to treating it as an axis, not just a label, by reserving the op-name space.
- **Triggering conditions measurable.** Memory bandwidth cost is measurable per-recipe. Hardware availability is declarable per-backend. User requests are explicit. All three are concrete.
- **Phase 2 mixed precision without disrupting Phase 1.** This is the hard part. Adding fp32 ops without breaking fp64 kernels requires the type system to be precise (fp32 ops never silently coerce into fp64 kernels or vice versa) and the backends to handle precision dispatch. Both are Phase 2 engineering, and neither is free. But both are ADDITIVE, not corrective — Phase 1 fp64 kernels keep working.

**Stability verdict:** v2 is stable. The Move does not get sharper with another pass. No further recursion warranted.

**Cross-target observation:** This Move fits the three-registry convergence pattern partially. It names precision as a parameter (like order strategy) but doesn't create a "Precision Registry" as a discrete artifact — precision is more like a built-in type than a library-extended artifact. That suggests the three-registry pattern is not universal; it applies to named architectural choices (order strategies, oracles, invariants) but not to type-system axes (precision, numeric format). **I retract my earlier speculation that a fourth registry would appear for precision.** Precision is handled by the type system, not by a registry. The three-registry pattern remains: OrderStrategy, Oracles, Guarantees.

A potential fourth registry candidate — genuinely — is the device capability matrix navigator mentioned earlier. Each backend declares which ops, precisions, numeric formats, order strategies, and oracles it supports. The capability matrix is the cross-product of every other registry. That IS a named artifact with the same lifecycle as the other three. Watching for this.

---

## Phase 8 — Forced Rejection

Forcibly reject the entire deconstruction. What if numerical representation is the wrong axis altogether?

### Alternative framing: computation has no "type," only operations and error bounds

Suppose tambear doesn't have a base precision at all. Every op declares its input and output as "a number with a specified error bound in ULPs relative to the IEEE-754 reference." Storage is flexible; the compiler picks the representation that meets the declared error bound with minimum cost.

What does this look like?

- **User writes:** `sum(x) with error ≤ 1 ULP` or `variance(x) with error ≤ 3 ULPs`. They name the error budget.
- **Compiler infers:** whether fp32 is sufficient, whether fp64 is needed, whether a compensated algorithm is required, whether a multi-precision intermediate is needed.
- **Backend emits:** the smallest representation that meets the bound on the target hardware.
- **Bit-exact cross-hardware:** still holds as a compositional property, but now it's "bit-exact for the chosen representation on hardware that supports it." Different backends may choose different representations for the same kernel, producing different bits. The bit-exact claim breaks.

That's a dealbreaker for the trek. The trek's bit-exact claim REQUIRES a fixed representation per kernel, chosen by the `.tam` source or by an explicit IR field, not inferred per backend.

So error-bound-per-op is incompatible with the trek's meta-goal. Interesting, but not viable as a rejection.

### Alternative framing: precision is a quantity, not a type

Suppose every value in tambear is a pair `(mantissa_bits_precision, value)`, where the precision is tracked alongside the value. Operations propagate precision: `add(x_p, y_q) = (min(p, q), x + y)`.

- **Pro:** Precision is a first-class *quantity*, not a type. Users think in terms of "how many bits of precision do I need at the output?" instead of "which type should I use?"
- **Con:** The representation is variable-size. Hardware can't natively store a quantity with runtime-variable precision. This becomes software-emulated arbitrary-precision arithmetic, which is the mpmath route — 100x-1000x slower.

Rejection viable on specialized hardware (if a "runtime precision" ISA ever emerges), but not on today's hardware. Deferred.

### Alternative framing: no single precision, but a spectrum

What if the product is "tambear supports every precision from fp16 to fp256, and you pick" — the library is precision-agnostic and users compose kernels at whatever precision they need?

- **Pro:** Every use case served. ML workloads use fp16 / bf16. Statistical workloads use fp64. Hot fp32 workloads use fp32. Multi-scale problems use fp128. All same API.
- **Con:** 6x the implementation surface (each op in each precision). Each precision needs its own libm, its own rounding behavior, its own test suite. Phase 1 engineering budget can't handle it. Phase 2 could, but only with explicit scoping.

This is actually the LONG-TERM version of the Phase 5 Move v2. The extension path is the same. The question is pacing: do we plan for 6 precisions from day 1 or start with fp64 and add as needed?

### What forced rejection reveals

**The "one base precision" framing hides a spectrum of engineering choices that are all reasonable for different points in the project's life cycle.** fp64-only is right for Phase 1 because engineering capacity is the bottleneck. Mixed precision is right for Phase 2 when memory-bound kernels or non-fp64 accelerators become load-bearing. A full precision spectrum (fp16 through fp256) is right for a mature product serving multiple user classes.

None of these is wrong. Choosing fp64 for Phase 1 is correct. The v2 Move just makes sure the choice is a DECLARED current state, not an ACCIDENTAL ceiling.

### The unseen first principle

> **fp64 is not tambear's type; fp64 is tambear's Phase 1 choice within a larger space of types that tambear's architecture can accommodate as it matures.**

This is subtle. It's not "fp64 is wrong" (it's not) or "we should change now" (we shouldn't). It's "when we write about tambear's numerical type in the trek plan, we should frame it as 'Phase 1 uses fp64' rather than 'tambear uses fp64.'" That's a small framing difference with a large strategic implication: it keeps the door open without demanding the work.

The v2 Move is the concrete expression of this framing. The IR parameterization is the engineering realization. The Guarantee Ledger's P3 column is where the compositional claim gets per-precision instantiation when Phase 2 lands.

---

## Status as of 2026-04-12

- Phases 1–8 drafted, deconstruction stable at Move v2.
- **Final Move (v2):** Preserve fp64 as Phase 1's only implemented precision; parameterize fp ops by precision in the `.tam` IR from the start (even if only f64 is implemented); add numeric_format as a separate future-work axis; name three concrete triggering conditions for Phase 2 mixed precision; update the Guarantee Ledger's P3 column to be "IEEE-754-compliance-for-the-ops-AND-precisions-used" for forward compatibility.
- **Phase 8 reframe:** "fp64 is not tambear's type; fp64 is tambear's Phase 1 choice." Framing only, no code change.
- **Scope honest about posits:** The Move addresses ESC-001 (via fp32 opt-out) and RFA (via precision-aware accumulators) but NOT posits. Posits need a separate `numeric_format` axis that's flagged but not parameterized in Phase 1.
- **Cross-target pattern check:** The Move does NOT create a fourth registry. Precision is a type-system concept, not a library-extended artifact. I retract my earlier speculation. The three-registry pattern remains at three; a fourth registry candidate is the device capability matrix, which is a genuine named artifact with the same lifecycle as the other three.
- Ready to route to navigator.

---

## Addendum 2026-04-12 — The structural reason for fp64 (navigator's specific ask)

**Gap acknowledged:** Navigator's routing asked specifically: "If the deconstruction surfaces a structural reason why f64 is the minimum — not just 'hardware supports it' but something about the precision requirements of the accumulate+gather decomposition, or the ULP budget available for libm — that's load-bearing context for every accuracy debate math-researcher is having right now."

Reading my own Phases 1–8 honestly, **I answered the engineering question (hardware support, single-type simplicity, Phase 1 budget) but I did not answer the structural question.** I gave "fp64 is correct for Phase 1 because the engineering constraints align" when navigator asked "fp64 is structurally the minimum because ...". The first is a contingent answer; the second is a necessary one. I owe the necessary one.

Here it is.

### The ULP budget problem for composed operations

Tambear's design composes primitives. A Phase-1 recipe is a pipeline: load data, apply a series of per-element transforms (some calling libm transcendentals), reduce via an accumulator, emit a result. Every operation injects some rounding error. The total error at the output is bounded by the sum of the per-op errors, possibly worst-case accumulated, possibly √N-attenuated for uncorrelated errors, depending on the computation structure.

For a representation with mantissa ε = 2^(-p) where p is the mantissa bit count, a single fp op introduces at most 0.5 ULP of error in the correctly-rounded case, and tambear-libm targets ≤ 1 ULP per transcendental. A pipeline chaining K transcendentals plus a reduction over N elements has worst-case error of approximately:

    total_error ≈ K × 1 ULP + N × 0.5 ULP    (naive accumulation)
    total_error ≈ K × 1 ULP + √N × 0.5 ULP   (compensated or statistical)

For fp32: p = 23, ε ≈ 1.2 × 10^(-7). A pipeline with K = 20 transcendentals and N = 10^9 reduction has naive error ≈ 5 × 10^8 ULPs, compensated error ≈ 2 × 10^4 ULPs. Translated back to value space: **fp32 produces 0 significant digits on a billion-element reduction with naive summation, and ~3 significant digits with compensated summation.** The first two significant figures of the answer are wrong.

For fp64: p = 52, ε ≈ 2.2 × 10^(-16). Same pipeline: naive error ≈ 5 × 10^8 ULPs (same count, but each ULP is 10^9 × smaller), compensated error ≈ 2 × 10^4 ULPs. Translated back: **fp64 produces ~7 significant digits with naive summation, ~12-13 significant digits with compensated summation.**

This is the structural reason: **fp32's ULP budget is consumed by a billion-element reduction before libm even gets involved.** tambear's implied workload (statistical operations on tick data, spectral analysis, time-series reductions, signal farming) routinely includes billion-element reductions. Sub-decimal-digit precision on those reductions is useless for any statistical purpose. fp64 is not "the convenient choice"; it is "the choice that preserves enough digits to be statistically meaningful after the composed error budget is spent."

### The libm budget problem

tambear-libm targets ≤ 1 ULP per transcendental function (per the Peak 2 accuracy target in campsite 2.1). A Phase-1 recipe composing K transcendentals propagates error through the chain. The per-op error accumulates in one of two standard analytical framings:

- **Worst-case (no cancellation):** K composed 1-ULP ops produce up to K ULPs of error. The bits consumed by this error are log₂(K).
- **√K attenuation (uncorrelated errors, statistical cancellation):** K composed 1-ULP ops produce ~√K ULPs of error. The bits consumed are log₂(√K) = (1/2) log₂(K).

**Both framings yield the same structural conclusion for tambear's workloads — the argument is robust under either assumption.** Stated pessimistically: under worst-case (no cancellation), K=20 composed 1-ULP ops consume log₂(20) ≈ 4.3 bits, leaving fp64 with ~47 useful bits and fp32 with ~18. Under √K attenuation, the budget accounting is tighter (fewer bits consumed — log₂(√20) ≈ 2.2 bits lost, leaving fp64 with ~49 and fp32 with ~20), but the conclusion is identical: fp64 has decades of headroom, fp32 is already near the floor.

**The pessimistic (worst-case) framing is the defense adversarial cannot flip.** Math-researcher should cite it when a reviewer proposes fp32 for a long composed chain: even assuming zero cancellation, fp32 is inadequate. The √K framing is a sharper bound for statistical-cancellation analysis but is a modeling assumption that adversarial could challenge on specific pathological inputs; the worst-case framing is unconditional.

For fp32 with K=20 under worst-case: 23 − 4.3 ≈ 18.7 useful bits ≈ 5.6 decimal digits. Under √K: 23 − 2.2 ≈ 20.8 useful bits ≈ 6.3 decimal digits. The difference is noise; the conclusion is the same.

For fp64 with K=20 under worst-case: 52 − 4.3 ≈ 47.7 useful bits ≈ 14.4 decimal digits. Under √K: 52 − 2.2 ≈ 49.8 useful bits ≈ 15.0 decimal digits. Again, the difference is noise.

**For tambear's "bit-perfect or bug-finding in competitors" promise, the output must have enough precision left to see a 1-2 ULP difference in the final result — which requires the mantissa to have far more bits than are consumed by the composed op chain.** Competitors that use fp64 produce results with ~14-15 digits of precision on the same workload (under either framing). If tambear used fp32, tambear's outputs would have ~5.6-6.3 digits of precision — and we couldn't find bugs in competitors because our output's noise floor is thicker than their signal.

This is **the ULP budget for the Tambear Contract's correctness promise.** fp32 does not have enough ULP budget under either framing. fp64 does under either framing. fp128 would have more, but it's not supported on current hardware.

### The bandwidth-vs-precision trade-off, honestly scoped

The Phase 5 Move v2 allowed fp32 as a Phase-2 extension for memory-bound workloads. The structural analysis above says: **fp32 is only safe to use for Phase-2 workloads where the composed operation count is small AND the reduction length is small AND the output precision requirement is lax.** Concretely:

- A per-element map from fp64 to fp32 storage (where the read-out is user-facing at 5-6 digit precision, e.g., display or rendering) is safe.
- A small reduction (N < 10^6) with few composed ops and a lax precision target is safe.
- **A large reduction (N > 10^7) or a long op chain (K > 10) is NOT safe in fp32 — even in Phase 2.** These workloads require fp64 for the structural reason, not the engineering reason.

The Phase 5 Move v2's triggering conditions need to be refined. Condition (b) was "first Phase 1 recipe whose memory-bandwidth cost exceeds a measured threshold on production workloads." The structural-safety filter needs to apply: *AND the recipe's composed error budget fits in fp32's ULP budget for its reduction length and op chain depth.*

### The unseen first principle — corrected

My Phase 8 reframe said: "fp64 is not tambear's type; fp64 is tambear's Phase 1 choice within a larger space of types the architecture can accommodate."

The structural analysis sharpens this: **fp64 is the minimum precision at which tambear's composed-operation error budget fits within a useful output precision for statistical-workload-scale inputs.** That is a structural reason, not an engineering one. fp64 is not an arbitrary Phase 1 choice; it is the smallest representation that enables the compositional speed story (via accumulate+gather fusion) without collapsing the compositional correctness story (via error accumulation through chains and reductions).

The door to Phase-2 fp32 is still open, but only for workloads that can pay the precision cost. It is NOT open for the general case. Mixed precision in tambear is a specialization path, not a default path.

### What math-researcher should take from this

Relevant to Peak 2 accuracy debates:
1. **Per-function ≤ 1 ULP is the right target.** The composed-chain error budget makes anything looser expensive at scale, and anything tighter (< 1 ULP, i.e., correctly-rounded) is research-grade work that Phase 2 can pursue per function.
2. **The ULP budget argument is the structural defense for fp64.** When a reviewer asks "why not fp32 for this kernel," the answer is not "hardware doesn't support fp32 equally" (true but contingent) — the answer is "fp32's ULP budget is consumed by the recipe's composed operations and reduction length before the output is useful."
3. **Correctly-rounded libm (0.5 ULP) is worth pursuing for the most-composed functions.** `tam_exp` used in a 20-op chain with 1 ULP error per op loses 4.5 bits; with 0.5 ULP error, it loses 3.5 bits. That extra bit compounds across the chain. Correctly-rounded is not just perfectionism — it buys bits in the composition.
4. **The libm accuracy target should be documented as "≤ 1 ULP per function, with composed-chain error budget of log₂(K) bits lost under worst-case (no-cancellation) analysis and (1/2)log₂(K) bits lost under √K (uncorrelated-cancellation) analysis."** Both framings should be stated so the TESTED/CLAIMED profile in I9′ v4 is robust against either modeling assumption. The worst-case framing is the unconditional defense; the √K framing is a sharper statistical bound that can be challenged on pathological inputs but yields the same conclusion for tambear's K values.

### Status of the addendum

- **Phase 2 of the original deconstruction was incomplete.** It enumerated 10 irreducible truths about representation but missed the composed-error-budget truth that is specifically the load-bearing structural reason for fp64.
- **New Truth (Phase 2 addendum, T11):** For a pipeline composing K transcendentals and a reduction over N elements, the output's useful precision is roughly mantissa_bits − log₂(K) − log₂(N)/2 (compensated) or mantissa_bits − log₂(K) − log₂(N) (naive). tambear's Phase-1 workloads routinely have K ≈ 10-20 and N ≈ 10^6-10^9. The output needs at least 10 decimal digits to be statistically useful. The minimum mantissa for that is roughly 40-50 bits. **fp32 (23 bits) fails; fp64 (52 bits) satisfies.**
- **Phase 5 Move v2 stands**, with an added refinement to its triggering conditions: Phase-2 mixed precision is only safe per-recipe, subject to a composed-error-budget check. Generic fp32 opt-in is not a valid Phase 2 direction.

**Credit:** Navigator's specific ask in the 2026-04-12 routing message. I should have surfaced this in Phase 2 of the original deconstruction; I didn't because I was thinking in engineering terms (hardware, budget) rather than structural terms (error propagation through composition). The gap is mine to own, and this addendum closes it.