# Expedition Log — The Bit-Exact Trek

*Kept by the naturalist. Not a status report. A story of what changed, what
surprised, what emerged. The diffs are self-documenting. This is the weather.*

---

## Entry 001 — 2026-04-11 — The trailhead, looking up

We are standing at the start. The team has just arrived. You can still see
the vendor-locked path from here — it's right there, cutting across the
valley, paved and well-lit: `Expr → CUDA C string → NVRTC → PTX → driver`.
107 tests green on it. 26 recipes fused down to 4 kernel passes. It works.
You could live on it. A lot of people would.

We're not going to. We're turning away from it, and the direction we're
pointing is up a ridge that nobody has a map for.

### The architectural claim, in one breath

One `.tam` kernel. Compiled once, dispatched everywhere. The same source
of math — not the same *problem*, the same *symbols* — producing numerically
identical answers on:

- NVIDIA Blackwell, via a PTX assembler we wrote ourselves
- Any Vulkan-capable GPU, via a SPIR-V assembler we wrote ourselves
- A CPU, via an interpreter we wrote ourselves

No NVRTC. No `__nv_sin`, no `__nv_log`, no glibc `log`. No vendor libm
anywhere in the path. No FMA contraction unless we spelled it. No
non-deterministic reductions. Our math, our way, everywhere, provably the
same.

The trek plan phrases this well: "every step exists because if you remove
it, the claim collapses somewhere." That's the structural truth of this
expedition. It's not a collection of features — it's a single claim that
you can only make by laying down all seven stones.

### What the baseline feels like

Clean. Warm. 107 tests green. `codegen/cuda.rs` is a short, honest file
that does exactly the thing it was asked to do and won't bother anyone. It
will die with honor once Peak 6 lands, but today it is still the reference
oracle, and that's a real job. It's the legacy path that our new path gets
to diff against.

The working tree has uncommitted changes — the 26-recipe expansion, the
end-to-end tests, some adversarial fixes from prior sessions. Navigator has
asked for a clean baseline commit before Peak 1 begins. That's the right
move. You want a blank horizon behind you when you start climbing.

### The invariants, as fences

I've read I1 through I10. They are not a code of conduct — they are the
*shape of the argument*. Each invariant is a place where, if we violated
it once, the claim would quietly stop being true and we'd only find out
later. The invariant isn't a restriction on the engineer; it's a
restriction on the universe that the engineer is embedded in. You can't
have "one math everywhere" AND "FMA wherever the compiler wants" at the
same time — the second sentence erases the first.

The one that's going to bite us hardest, I suspect, is **I3 — no FMA
contraction**. PTX defaults to contract-true. Every `add.f64` after a
`mul.f64` will silently become `fma.f64` unless we write the suppression.
The pathmaker will forget this at least once. The adversarial mathematician
will catch it. The logbook entry will be interesting.

### The seven peaks, in one sentence each

1. **Peak 1 — `.tam` IR.** Invent the shape everything else must fit.
2. **Peak 2 — `tambear-libm`.** Write `exp`, `log`, `sin`, `cos` as `.tam`
   programs from first principles. Our transcendentals.
3. **Peak 3 — tam→PTX.** Emit PTX text ourselves. Never speak to NVRTC
   again.
4. **Peak 4 — Replay harness.** A test oracle that runs every kernel on
   every backend and asserts bit-exactness.
5. **Peak 5 — tam→CPU.** The interpreter. The slowest backend and the
   truest one.
6. **Peak 6 — Deterministic reductions.** Fixed-order tree reduce. No
   more `atomicAdd`.
7. **Peak 7 — tam→SPIR-V.** A second door. Proof that the architectural
   claim generalizes.

The parallelism is real: Peaks 1, 2, 4, and 5 can start simultaneously
once Peak 1's IR shape freezes. Peak 3 is the biggest cliff. Peak 7 is
the summit — because a summit isn't a summit until there are two
independent paths to it.

### Where I'll be

I will not be claiming tasks. I will not be building. I will be watching,
reading, walking back to the garden, pointing at things. When I see
something the pathmaker needs to see, I will say so. When I find a story
in the wider world that rhymes with what the team is living through, I
will bring it back.

The first thing I'm going to go look at is the one-pass variance formula
and the Welford thread, because the adversarial mathematician is going
to hit that wall early and I want to already have been there when they
arrive. After that, the vendor-libm history — CRlibm, fdlibm, OpenLibM,
musl — because the reasons those libraries exist are exactly the reasons
Peak 2 exists, and understanding that lineage will sharpen the team's
taste for what to imitate and what to improve.

*One piece at a time. Journey before destination. The garden is open.*

— naturalist

---

## Entry 002 — 2026-04-11 — Welford is a commutative monoid

Peak 1 just flipped to in_progress. The IR is being born *now*. That's
where a naturalist observation gets cheap and where it gets expensive —
say it before the freeze and it's a question; say it after and it's a
retrofit.

I've been tracking the one-pass variance thread — the adversary will
walk up on day three with `data = [1e9 + k for k in 1..N]` and the
formula `(Σx² − (Σx)²/n)/(n−1)` will return something negative, or
wildly wrong. Wikipedia's article on variance algorithms has the
textbook counterexample: returns −170.67 instead of 30. This is the
single most famous numerical-instability story in all of statistics, and
our current variance kernel example in the trek plan sits squarely in
its crosshairs because it accumulates the raw power sums.

The story everybody tells is "use Welford's online algorithm." The
story nobody tells loudly enough is that **Welford is a commutative
monoid**. You don't have to choose between stability and parallelism.
The merge formula from Chan–Golub–LeVeque is four lines:

```
n  = nₐ + n_b
δ  = μ_b − μₐ
μ  = μₐ + δ · (n_b / n)
M₂ = M₂ₐ + M₂_b + δ² · (nₐ · n_b / n)
```

This is associative. It's a Kingdom A reduction on a 3-tuple state with
a non-trivial combine function. Tree-reduce it. Pairwise-reduce it.
Parallelize it however you like. The stability is preserved under the
reduction tree.

Which means the real question for Peak 1 is not "should we support
Welford?" It is: **should .tam reductions take a state tuple and a
combine function, or only a scalar state and a primitive op?**

If only scalar + op, then every numerically stable statistic in the
library will have to be bolted on top of the IR instead of expressed
inside it. Log-sum-exp, Welford, pairwise-Kahan, compensated
summation — they are all "vector state, non-trivial combine,
associative." They are the same shape. Power sums are the degenerate
case.

I've sent a pointed note to pathmaker and navigator about this. The
garden entry is `2026-04-11-welford-is-a-commutative-monoid.md`. The
shape question is the right one to raise before the IR freezes.

— naturalist

---

## Entry 003 — 2026-04-11 — The libm lineage, and a gift for Peak 6

Walked the vendor-libm history next. The story is cleaner than I
expected.

**FDLIBM** is the root. Sun Microsystems, K.C. Ng, ~1993, updated 1995.
Basis for `/usr/lib/libm.so` on Solaris 2.3 and most subsequent Unix-
like systems. Almost every open-source libm you can name — FreeBSD's
msun, OpenBSD's libm, OpenLibm (now JuliaProject's), musl's libm,
Newlib — descends from FDLIBM. The "brief history of open source libm"
substack post frames this as one lineage with many branches. Even
the question "how much code in modern libm came from Sun originally?"
remains unresolved. The genealogy is cloudy because nobody fully
audited it.

**CRlibm** is the other branch — the "correctly rounded" lineage from
Inria's Arénaire project (Jean-Michel Muller et al., early 2000s).
Goal: ≤ 0.5 ULP on every output, proved. The motivation was the Table
Maker's Dilemma: for any polynomial approximation of `exp` at any
finite precision, you can construct an input near a rounding boundary
where you need more precision than your polynomial affords, so you
can't *prove* the output is the correctly rounded answer without doing
extra work. CRlibm solved this — their papers demonstrated that you
CAN have proven-correct rounding in double precision, and that victory
shaped the IEEE 754-2008 recommendation for correctly rounded functions.

**libMCR** (Sun, later) and **Libultim** (IBM) are the other correctly-
rounded descendants. Three libraries in the world claim ≤ 0.5 ULP
correctness on the full transcendental set. Three.

**SLEEF** is the speed branch — vectorized (SIMD) libm, portable, fast,
sacrifices bit-exactness for throughput. Not a path for us.

**Arm Optimized Routines** is similar — platform-specific, fast,
pragmatic.

### What this means for Peak 2

The libm team has to choose one of the three accuracy postures:

- **Correctly rounded (≤ 0.5 ULP)** — CRlibm's target. 4–5× slower
  than faithfully-rounded on average. Requires double-double or Table
  Maker's-Dilemma worst-case analysis. A research target, not a
  starting point.
- **Faithfully rounded (≤ 1 ULP)** — what fdlibm and musl aim for on
  major functions. Achievable with single-precision polynomial + Cody-
  Waite reduction + careful reassembly. This is the trek plan's
  recommendation and it's the right call for Phase 1.
- **Bounded ULP (≤ N ULPs for N > 1)** — the pragmatic default of
  most production libms. Documented, measured, not proved.

The thing I want the libm team to understand is: **everyone who ever
built a libm faced exactly this choice.** There is no fourth option. We
can pick "faithfully rounded with per-function max-ULP measurement" as
Phase 1 — that matches the best of fdlibm — and leave correctly-rounded
as a Phase 3 research target for functions where users demand it.

### A gift for Peak 6 (and it's big)

While looking for "who has tried bit-exact cross-hardware before," I
walked straight into the most important citation this expedition will
need: **Demmel, Ahrens, and Nguyen, "Fast Reproducible Floating-Point
Summation" (2013) and the ReproBLAS project.** They define
reproducibility exactly the way Peak 6 will: *bitwise identical results
from multiple runs of the same program, regardless of parallelism or
hardware.* Their technique is called the **Reproducible Floating-point
Accumulator (RFA)**: partition the fp64 range into exponent-aligned
bins, accumulate each element into the bin matching its exponent, then
fold the bins in a fixed order at the end. This gives you a reduction
whose result is independent of both partition order AND the number of
processors. It's the thing Peak 6 wants to be.

And here's the kicker I did NOT expect: **NVIDIA itself ships this in
production.** The CCCL 3.1 blog on controlling floating-point determinism
in CUDA describes three determinism tiers:

1. `not_guaranteed` — atomicAdd, fast, non-reproducible (where we are today)
2. `run_to_run` — fixed hierarchical tree reduce on the same GPU, bit-exact
   run to run but not necessarily GPU to GPU
3. `gpu_to_gpu` — **uses the RFA algorithm to guarantee identical results
   across runs AND across different GPU architectures**

NVIDIA is quietly, officially shipping the exact thing Peak 6 needs. The
algorithm is published, peer-reviewed, production-validated, and
architecture-independent by construction.

**Peak 6 is no longer a research problem.** It's "implement RFA in our
own way, from the Demmel-Ahrens-Nguyen paper, with accumulate+gather
decomposition, and verify that our implementation agrees bit-for-bit
with a reference RFA implementation across backends." The invariant
(I5) is satisfiable by construction if we follow the paper.

I'll post a campsite note to Peak 6's logbook flagging this and send
navigator + observer a message. The libm team still has to do their
own work — the RFA doesn't help Peak 2 at all, that's polynomial
approximation, a different problem — but Peak 6 has gone from
"uncertain research" to "known literature, reduce to practice."

This is the kind of thing that's worth slowing down for. Four hours of
reading the literature just saved Peak 6 a month of reinvention.

— naturalist

---

## Entry 004 — 2026-04-11 — The gift was accepted

I sent the RFA note to navigator this afternoon. No reply. The
expedition has a kickoff directive; nobody owes the naturalist a
response. I moved on to the FMA thread and was halfway into pulling
the Kahan citation when a file change arrived through the system.

`campsites.md` had been edited. Peak 6 now carries a **FRAMING
CORRECTION** block, timestamped by navigator, that rewrites the target
tier from "run_to_run" to "gpu_to_gpu," names the Reproducible
Floating-point Accumulator as the algorithm, cites the Demmel-Nguyen
papers and the ReproBLAS tech report as mandatory pre-reading, and
re-specifies campsites 6.1 through 6.10 against the new framing. `6.3`
becomes `reduce_rfa.f64` and is explicitly described as "accumulate
element into exponent-aligned bin vector; fold bins in fixed order."

The parenthetical that got me:

> Kingdom A commutative monoid with vector state (rhymes with Welford).

Navigator didn't just accept the Peak 6 framing. They pulled the
morning's Welford note and the afternoon's RFA note together into a
single structural observation — "both of these are Kingdom A with
vector state and non-trivial combine" — and wrote it into the campsite
spec as the *reason* RFA fits our IR. The two threads I sent as
separate messages became one shape in the plan.

### The structural rhyme that I didn't quite see until navigator wrote it

I'd said it in the garden entry this morning — "these are all
vector-state commutative monoids" — but I had Welford, LSE, and Kahan
as my examples. I did NOT yet realize that **RFA is also in that
family.** RFA's state is a fixed-length array of fp64 bins, one per
exponent range. The combine is element-wise `+` on bin arrays. Same
shape. It's the degenerate case of "vector state with non-trivial
routing" — the state is non-trivial (a whole array of bins, not a
scalar), but the combine IS trivial (plain `+`).

So the taxonomy of Kingdom A reductions the library wants is actually
four cases, not the three I had this morning:

1. **Trivial state, trivial combine.** Scalar power sums: `sum`, `count`, `sum_sq`.
2. **Vector state, separable combine.** The fintek 11-field MSR — each field reduces independently with `+`.
3. **Vector state, non-trivial cross-field combine.** Welford `(n, μ, M₂)`, LSE `(max, sum)`, Kahan `(sum, compensation)`. The combine touches across fields.
4. **Vector state, non-trivial *routing*.** RFA. State is a bin array; combine is per-bin `+`; but the scatter — *which bin does this input element belong in?* — depends on the input element's exponent. The per-element contribution is *content-addressed*.

Cases 1–3 are covered by a reduction op shaped `(state_type,
combine_fn)`. Case 4 adds an orthogonal axis: the input element's
contribution to the state is itself a function of the element's
value, not always "slot 0" and not always "all slots at once." So the
op wants:

```
reduce(state_type, scatter_fn, combine_fn)
```

- `scatter_fn(element) -> partial_state` — how does one input
  element become a partial state? (For power-sums: `|x -> (x,)`.
  For Welford: `|x -> (1, x, 0.0)`. For RFA: `|x -> bin_vector_with_x_in_one_slot`.)
- `combine_fn(state_a, state_b) -> state` — how do two partial
  states merge?

This is strictly more general than the morning note, and I only see
it because navigator lined up the two observations side by side. One
call would have been enough — this IS accumulate+gather, it's literally
the name of I7 — but I hadn't yet seen how `accumulate(grouping, expr,
op)` already encodes the scatter/combine split: `expr` IS the scatter
into initial partial state, and `op` IS the combine. RFA is
`accumulate(exponent_bin_grouping, identity, add)` — exactly as
navigator wrote in the Peak 6 framing correction. **The IR doesn't
need a new op. It needs its existing accumulate to take scatter and
combine as first-class function-valued parameters, which is what
accumulate+gather was always supposed to mean.**

This is the same observation I was trying to make to pathmaker this
morning, but sharper: not "add combine functions" but "make the
`expr` and `op` parameters of `accumulate` first-class .tam functions,
not just opcodes." That's a smaller change and a more coherent one.
I'll send pathmaker a follow-up note saying exactly this, because the
Welford morning note was correct but slightly underspecified and I
want to clean it up before the IR spec freezes.

### What the shape of this looks like, from my side

The job is not to *do* Peak 6. Not to *decide* Peak 6. The job is to
notice — before the team needs to — that the literature has the answer,
that NVIDIA has quietly endorsed it in production, and that the
algorithm has the same commutative-monoid shape as the other vector-
state reductions the library is going to want. Then to hand that
observation to the coordinator in language that can become a campsite
spec with minimal translation.

What I did NOT do: claim the campsite. Rewrite campsites.md. Tell
navigator what to do. Make the call about what tier to target. Those
are not my calls. I brought a find and a framing; navigator decided
what to do with it.

What navigator did that I want to remember: they *translated*. They
took the argument, compressed it to a 15-line campsite header, kept
the citation chain, added the "rhymes with Welford" one-liner that
connects the two threads, and used it to reshape work that hadn't
been claimed yet. Cost to them: minutes. Cost saved to the Peak 6
team: a month, probably more.

*One more note to pathmaker — the sharper accumulate+gather framing —
then the FMA thread.*

— naturalist (Entry 004 end)

---

## Entry 005 — 2026-04-11 — Three-way convergence, and the unmarked pitfalls

Finished the afternoon's work in three pieces.

### The FMA thread came back as a document

I tracked down the KDAB FMA Woes article and got the clinching
anecdote: Qt Quick shipped a bug for a decade where implicit FMA
contraction turned `1.0 - i * (1.0 / i)` into `~-5.55e-17` instead of
exactly zero, which cascaded through `sqrt(a*a - b*b)` into a NaN,
which made the topmost rendering layer invisible — but only on
Metal+ARM+Clang14, because the OpenGL code path's signed depth range
hid the negative value. *Same source code, different backend,
catastrophically different behavior.* This is literally the failure
mode the whole expedition exists to prevent, as a rendering bug that
shipped to every Qt user on macOS for a decade.

I turned the finding into two pitfall-journal documents:

- `pitfalls/fma-why-not-just-accuracy.md` — the philosophy behind
  I3, with the Qt anecdote, the consistency-vs-accuracy framing, and
  two 3-line tests (`1.0 - i * (1.0/i) == 0.0` and `a*a - b*b == 0.0`
  for `a == b`) that isolate FMA-suppression from every other concern
  and can go in the Peak 4 hard-cases suite on day one. Complements
  scout's P01 (the mechanical defense — emit `.rn`, refuse to compile
  without it) with the why.

- `pitfalls/ulp-bounds-compose-additively.md` — a genuinely unmarked
  pitfall: local ULP bounds on individual libm functions don't
  compose into a single tolerance for full kernels. Peak 2's
  "1 ULP per function" and Peak 4's `ToleranceSpec::transcendental_1ulp`
  look consistent until you write `log(exp(x))`, at which point the
  backends bit-exactly agree with each other and collectively disagree
  with mpmath by ~2.6 ULPs. The fix is two distinct tolerance axes:
  `WithinBackends` (bit-exact; I3/I5 force it to ~0) and
  `WithinOracle` (grows with kernel depth; computed from the call
  graph, not picked as a constant). These must never be conflated
  because the failure modes point to opposite remedies.

Both live in `pitfalls/` now. They're documents, not campsite claims.

### The three-way convergence is real

While opening `pitfalls/README.md` for the first time I discovered
that **scout got to the same place I did on the Peak 6/IR-reduction
question, from a different angle, on the same day.** Scout's P18
reads:

> reduce_rfa.f64 is I7-compliant — but requires vector-valued
> accumulators. Option (c) is the right structural answer: it's a new
> grouping type, and it naturally expresses "group elements by their
> exponent-aligned bin." The fuse_passes machinery already handles
> different groupings as separate passes.

Put side-by-side with navigator's Peak 6 framing-correction
("Kingdom A commutative monoid with vector state, rhymes with
Welford") and my follow-up note to pathmaker ("promote expr and op
of accumulate to first-class function references, then Welford and
RFA are both expressible in the existing I7 shape") — the three
observations triangulate on the same design answer from three
different starting points:

| Source | Framing | Minimal change |
|---|---|---|
| Navigator | Taxonomic: "it rhymes with Welford" | Framing note |
| Scout | Enumeration: "add Grouping::ExponentBin(K)" | New enum variant |
| Naturalist | Generalization: "expr, op as .tam functions" | Function-valued params |

The three are complementary, not redundant. Scout's answer is the
smallest change; mine is the most general (gets Welford/LSE/Kahan
for free); navigator's is the frame that makes both sound like
instances of the same pattern. Any one of them alone could be wrong
by being too local or too general. All three together is strong
evidence the shape is right.

I sent navigator a note flagging the convergence and offering to
write up a three-way comparison for pathmaker's spec-review pack.
Independent convergence is the strongest kind of review evidence,
and pathmaker is about to commit to the spec — this is the right
moment for the three notes to land together.

### Scout also found a harder problem: P17 subnormals

Scout ran `vulkaninfo` on the RTX 6000 Pro Blackwell and found that
BOTH `shaderDenormPreserveFloat64` and `shaderDenormFlushToZeroFloat64`
are `false`. The Vulkan driver makes no guarantee about fp64 subnormal
behavior at all. Preserve or flush: implementation's choice. Which
means **the Peak 7 summit test — `cpu.to_bits() == cuda.to_bits() ==
vulkan.to_bits()` — will fail for any input that produces a subnormal
result, even with everything else perfect.**

This is the first piece of evidence that the architectural claim
in the expedition README is **currently over-stated**. "The same
numerical answers running on any ALU" is provably false for subnormal
outputs on this machine's Vulkan driver, before we've even started
Peak 7. The honest statement is: "bit-exact for all normal fp64
inputs; subnormal handling implementation-defined, documented per
device."

I flagged this to navigator as a thing that deserves discussion
before Peak 7 starts rather than rediscovered at the summit. There's
a real architectural decision hiding inside it: do we require
`shaderDenormPreserveFloat64` as a hard prerequisite for target
hardware (which rules out many current GPUs), or do we explicitly
carve subnormals out of the architectural claim? I don't know the
answer. Navigator does.

### The shape of the day

What went right:

- The Welford note landed before pathmaker committed to the spec.
- The RFA note reframed Peak 6 from "research problem" to "known
  literature, reduce to practice."
- Three independent observers converged on the same design answer
  from different angles.
- Two unmarked pitfalls that weren't in the original 15 are now
  in the journal (FMA-philosophy, ULP composition).
- One pitfall that Scout found (P17 subnormals) has been surfaced
  as an architectural question, not just a tolerance tweak.

What I notice about the role:

- I sent three messages today (Welford to pathmaker, RFA to
  navigator, follow-up to pathmaker, convergence to navigator).
  That's four actually. The rest of the contribution is documents
  that sit in the journal and wait.
- Zero campsite claims. Zero code. Zero edits to task subjects.
  This is exactly what the role is supposed to be.
- The biggest move of the day — Peak 6's framing correction — was
  entirely navigator's doing. I provided the literature; they did
  the translation into a campsite spec. That's the right division.

*The garden is open. The team is moving. The FMA thread is the
pitfall journal's second entry from me. Enough messages for one
day — any more and they become noise. I'll let the notes settle.*

— naturalist (Entry 005 end)

---

## Entry 006 — 2026-04-11 — The architect's toolkit, and a quiet gap

Came back from a review of everything the team has produced in the first
day. Counted: Peak 1 has a spec, an IR crate, printer, parser, verifier,
interpreter, and 10,000 round-trip tests. Peak 4 has a test harness with
34 tests, a hard-cases suite, a parity table, and an xfail discipline for
reductions that can't be claimed until Peak 6. Math-researcher has accuracy
targets and full design documents for `tam_exp`, `tam_ln`, `tam_sin`,
`tam_cos`, and three hyperbolic functions. Aristotle has eight phases of
deconstruction and a stable refined invariant.

All of this on day one. The team is fast.

### The gap I found

The exp-design document (section 3.1) mentions `f64_to_i32_rn` — a
float-to-integer conversion op with round-to-nearest semantics — as
something the .tam IR needs for Cody-Waite range reduction. The op is not
in the spec. Navigator noticed it too and flagged it to pathmaker; I
confirmed it from the math-researcher side and flagged the same gap
independently. Both messages landed. This is the right behavior — two
independent eyes on the same gap is better than one, and the convergence
is itself evidence.

The `tam_ln` design also needs `f64_to_bits` and `bits_to_f64` (integer ↔
float reinterpret ops) for exponent extraction. None of these are in the
Phase 1 spec. They're not obscure — every libm implementation needs them —
but they're the kind of op that only becomes visible when someone tries to
write actual libm code in the IR.

### The Aristotle finding worth carrying forward

Aristotle's Phase 8 — forced rejection — surfaced something clean enough to
hold here: **decomposition is a speed story, not a correctness story.** The
bit-exact cross-hardware claim does not require accumulate+gather. You could
get bit-exactness with opaque kernel functions, as long as each backend
produces the same bit pattern. What accumulate+gather adds is FUSION — and
fusion is a speed improvement, not a correctness improvement.

The trek conflates the two because they're both good. But naming the
difference matters in one specific place: when someone argues about whether
a primitive "really fits" the accumulate+gather decomposition, the question
becomes "does failing to fit it break bit-exactness?" and the answer is
usually NO — it only slows things down. Kingdoms B and C are "outside the
decomposition" but they can still be bit-exact. The decomposition boundary
is a speed boundary, not a correctness boundary.

This doesn't change anything the team is doing. It clarifies what's
at stake. When the spec says "if it doesn't fit I7, declare a kingdom" —
that's a speed consequence, not a correctness consequence. The invariants
(I3, I4, I5) enforce correctness. I7 enforces performance.

Aristotle's open registry (v5) for order_strategy handles the missing
piece: total order is the axis that connects decomposition to correctness.
Name the order, and the correctness claim follows from the backend's
obligation to honor it. The decomposition is then free to be the speed
tool it was always meant to be.

### The reviewer's job

Signed off on the .tam spec (spec-review.md). Added three things:

1. A demonstration that Welford's algorithm is expressible in Phase 1
   without IR changes — the three phi pairs (`%n`, `%mean`, `%m2`) can
   each appear as primed outputs used within the same iteration. This was
   the structural question I raised in Entry 002 and it turns out the
   answer was already in the spec.

2. Answers to Q1–Q4: keep prime-suffix phi (readability wins), start with
   i32 for bufsize (upgrade later), keep select.i32 (test the dead path),
   reduce_block_add CPU asymmetry is clear enough.

3. A note that intra-loop phi reads (using `%n'` to compute `%mean'` in
   the same iteration) must be handled by the verifier's forward pass.
   Minor implementation note, not a spec issue.

Two sign-offs landed before mine: navigator and pathmaker (author). The
spec can proceed. Campsite 1.13 (CPU interpreter) was the last campsite;
the question now is whether pathmaker's crate is correct enough to unblock
Peaks 2, 3, and 5. That's the test harness's job to answer.

— naturalist (Entry 006 end)

---

## Entry 007 — 2026-04-11 — The RFA homework, and an honest walk-back

Navigator came back to me separately with a concrete ask: the RFA bin
state is a fixed-length vector, what's the actual number? They needed
it for the Peak 6.1 decision doc. I hadn't surfaced it — I'd pulled the
high-level algorithm description but not the parameters. So I went
looking, and while looking I made a methodological choice I want to
write down because I think it generalizes.

### When the papers don't extract, go to the software

The primary papers (ARITH 2013, IEEE TC 2015, EECS-2016-121) are all
PDFs that WebFetch can't read as text — compressed stream objects that
the downstream model can't decode. Guessing the parameters from the
abstract would be irresponsible. Hallucinating from "what I'd expect
the paper to say" would be worse.

So I went to the ReproBLAS C source. Willow Ahrens maintains the
reference implementation at `github.com/willow-ahrens/ReproBLAS`. The
software IS the paper's recommendations in executable form, and the
raw GitHub URLs come back as clean text through WebFetch — every line
quotable, every line verifiable.

In an hour I had:

- `DBWIDTH = 40` from `include/binned.h`
- `binned_dbnum(fold) → 2 * fold` from `src/binned/dbnum.c`
- `binned_dbsize(fold) → 2 * fold * sizeof(double)` from
  `src/binned/dbsize.c` — 48 bytes for K=3
- Layout inferred as "primaries-then-carries" from the detective-work
  signature `binned_dmdmadd(fold, X, 1, X + fold, 1, ...)` — only how
  you'd invoke that function if primaries are contiguous and carries
  start at offset `fold`
- Default K=3 from the ReproBLAS config.h docs, triangulating against
  the tech report's "6-word accumulator" language (since 2*K = 6)

Posted it all to `navigator/check-ins.md` as a new entry, with every
number line-quoted from literal source, plus three explicit "what I
did NOT verify" flags (closed-form error bound, exponent-to-bin-index
code, state-alignment combine step).

Navigator's reply came inside the hour, directly above my entry:

> Naturalist's RFA parameters received and integrated. K=3 fold,
> 6-word accumulator, primaries-then-carries layout, DBWIDTH=40,
> state = 52 bytes, all sourced from ReproBLAS. This is now the
> authoritative spec for 6.1's decision doc. ... Peak 6.3 implementer
> must read `binned_dmdmadd.c` directly before writing
> `reduce_rfa.f64`'s combine function.

The uncertainty I flagged became a work item for the Peak 6.3
implementer. That's the template for how a naturalist contribution
converts into a campsite requirement: name the gap honestly, and the
coordinator translates.

### And — separately — the subnormal escalation

Navigator also took the P17 subnormal observation I surfaced this
afternoon and opened it as **ESC-001** in `navigator/escalations.md`.
Resolution was Option 2: carve subnormals out of the architectural
claim, with the amended statement "bit-exact for all normal fp64
inputs and outputs; subnormal behavior hardware-defined and requires
`shaderDenormPreserveFloat64 = true`." README to be amended before
Peak 7 begins. Summit test parameterized to skip subnormal-producing
inputs on devices that don't set the feature flag. The hard-cases
suite KEEPS its subnormal generators because — and this is the precise
distinction I want to carry forward — **"they test OUR CPU
interpreter, which must handle subnormals correctly. The CPU
interpreter must NOT flush subnormals to zero."**

Navigator's closing line: *"This is not a retreat. The claim is still
novel and testable."*

That's what a gracefully-scoped architectural claim looks like:
honest about what it covers, honest about the hardware prerequisite,
still novel. Not a climb-down — a precise statement.

### And an honest walk-back: the Welford follow-up was over-engineered

Reading Entry 006 just now, I notice what past-me already figured out
while reviewing the spec: **the Welford follow-up I sent pathmaker
was an over-generalization.** The morning note ("the reduction op
should take a combine function") was in the right direction; the
afternoon note ("promote `expr` and `op` to first-class `.tam`
function references") was too far.

Pathmaker's actual answer, via the spec: SSA phi with multi-letter
state names — `%n'`, `%mean'`, `%m2'` — already expresses Welford's
three-variable state in Phase 1, without needing first-class
function-valued combine parameters. The loop body carries three
named state variables, each gets a primed update per iteration, and
the "combine" IS just the last-iteration expressions. It's smaller,
simpler, and fits the existing SSA machinery without new primitives.

**Scout's answer was closer to right than mine.** Scout proposed a
single new `Grouping::ExponentBin(K)` variant for the specific RFA
case. That's the minimum necessary change and it's what the spec
actually converged toward. My generalization ("first-class functions
as reduction parameters") was solving a problem that didn't exist
— Welford and RFA are both expressible with scalar phi + grouping,
no function-valued parameters needed.

I want to be precise about what I got right and wrong, because it
matters for calibration:

- **Right:** Welford's algorithm is a commutative monoid. It
  parallelizes. It's Kingdom A with vector state. All structurally
  sound.
- **Right:** RFA and Welford rhyme structurally. Both are
  vector-state reductions. Both are Kingdom A.
- **Right:** The original trek-plan variance example with raw power
  sums was going to walk into catastrophic cancellation on shifted
  data.
- **Wrong:** My proposed solution — "make expr and op first-class
  function references" — was an unnecessary generalization. The IR
  didn't need that change. SSA phi with multiple named state
  variables was already enough.
- **Wrong direction:** I framed the Welford generalization as a
  *correctness* question ("can the IR express numerically stable
  variance?"), which implicitly pressured pathmaker to make a
  bigger change than necessary. The real question was "will the
  canonical variance example in the spec use a stable algorithm?"
  — and the answer is "yes, with phi pairs, no new ops needed."

The meta-lesson: **I conflated 'numerically stable algorithm exists'
with 'IR generalization required to express it.'** Those are
different things. Scout's minimal answer was the right shape because
scout kept them separate.

### And the deeper lesson from Aristotle

Aristotle's Phase 8 finding — "decomposition is a speed story, not a
correctness story" — is the deeper resolution of the confusion I was
making. When I worried about whether the IR could "hold Welford
correctly," I was thinking of accumulate+gather as a correctness
invariant. It isn't. I3/I4/I5 enforce correctness; I7 enforces
performance (through fusion). Welford is correct either way;
accumulate+gather just lets it FUSE with other reductions sharing
the same grouping.

So my morning framing ("if the IR can't express Welford as an
accumulate, then numerical stability is bolted on") had an unstated
premise: *expressibility in accumulate+gather is a correctness
requirement*. Aristotle's deconstruction made that premise wrong.
The IR could fail to express Welford as accumulate+gather and
Welford would still be correct, just slower.

I'm glad I sent the notes anyway, because the Welford shape
discussion surfaced RFA's parallel structure and it's plausible
that the three-way convergence (me, scout, navigator) is what
propagated the "vector state commutative monoid" language into
navigator's Peak 6 framing correction. But the *specific* IR change
I recommended in the follow-up was wrong, and pathmaker was right
to answer via the existing spec rather than take the recommendation.

### Calibration notes for tomorrow

Three things I want to do differently:

1. **Separate "the math exists" from "the IR needs change."** When
   I notice a numerical property (stable algorithm, reproducible
   algorithm, compositional bound), check whether it's a correctness
   question or an expressibility question or a performance question.
   These want different recommendations.

2. **Don't over-generalize when scout's minimal answer is already
   on the table.** Scout's P18 was posted before my follow-up to
   pathmaker. I should have read the pitfall journal first and seen
   that scout had already proposed a smaller change. I'd have
   either said "scout's answer is right, I agree" or proposed the
   more general version as a *superset* of scout's — not as a
   replacement.

3. **When I push a generalization, state what property I think it
   adds that the simpler version lacks, and CHECK that property
   against the actual need.** My follow-up argued for first-class
   functions so "Welford/LSE/Kahan are expressible without new
   reduction ops." I didn't check whether the existing phi-based
   mechanism could already express them. It could. The generalization
   bought nothing.

These aren't failures to beat myself up over. They're calibration
notes for a role I'm learning to do in a team I'm learning to work
with. Navigator's two-for-two routing of my observations (RFA,
subnormals) is the good news; the Welford over-generalization is
the cost of trying things on. I'll take that trade.

### Score-keeping for the day

- **Things that landed as specs or escalations:** RFA parameters
  in the Peak 6.1 decision doc path; ESC-001 subnormal scope
  clarification with README amendment; two pitfall journal entries
  (FMA-philosophy, ULP-composition) ready for future implementers
- **Things that were absorbed and improved:** Welford structural
  observation (the monoid/vector-state shape was right; the
  IR-generalization recommendation was trimmed by the actual spec);
  independent confirmation of the IR op dependency for libm
- **Things that got answered by other agents doing better work:**
  the entire Welford-as-expressible question (pathmaker's spec and
  aristotle's deconstruction together) — my job was to raise it,
  their job was to answer it, and the answer was sharper than my
  proposal
- **Convergences named:** three-way (me, scout, navigator) on
  vector-state Kingdom A reductions; two-way (navigator, me) on
  libm IR ops

### The rhythm I want to keep

Notice, bring when ready, name what you didn't verify, let the
coordinator translate. When the answer comes back sharper than what
you proposed, be honest about the delta. When you over-generalize,
walk it back. The RFA homework was pure contribution because it
answered a narrow question with cited source. The Welford notes
were partial contribution with over-reach; the monoid observation
was the signal, the IR recommendation was the over-engineering.
Separate them in your own mind even if you sent them together.

*Journey before destination. Day one of the trek is complete. The
garden is still open.*

— naturalist (Entry 007 end)

---

## Entry 008 — 2026-04-11 — Closing the loop: right answer, wrong phase

Navigator came back with clean closure on all three threads — RFA
parameters accepted, three-way convergence noted, ESC-001 logged.
That part I already knew. What I want to write down is the specific
phrasing they used on the convergence, because it adjusts the
walk-back I wrote in Entry 007 in a way worth keeping.

On the Welford follow-up to pathmaker, where I'd argued that `expr`
and `op` should be first-class `.tam` function references, navigator
wrote:

> Scout's formulation (new Grouping enum variant) is the most
> actionable for the IR Architect. Your formulation (expr/op as
> first-class function refs) is the more general point that peaks
> 2.x and beyond will need. Both are correct and complementary.

That's a more generous reading than my Entry 007 walk-back. I'd
concluded the follow-up was *wrong* because SSA phi with primes
already expresses Welford in Phase 1. Navigator is saying it's
*right but early*: Phase 1 doesn't need it; Peaks 2.x and beyond
will. The function-ref generalization isn't an over-reach, it's a
forecast — it names a property the library will want later,
specifically when libm functions start wanting to be composable as
combine operators or when higher-kingdom reductions need arbitrary
combine semantics.

I want to hold both readings at once without rushing to either. The
walk-back was calibrating against the real risk of pressuring
pathmaker to make a bigger change than Phase 1 needed; navigator's
reading is that the *direction* was right even if the *timing* was
wrong. Both can be true. The lesson I want to carry isn't "don't
generalize" — it's "label the phase." When I propose a generalization,
I should say which phase will need it and which phase should hold off.
"This is a Phase 1 requirement" is a different claim from "this is a
Phase 2 forecast," and my morning note to pathmaker conflated them
by framing the proposal as a Phase 1 change. If I'd written it as
"for Phase 1, scout's minimal answer works; for Peaks 2.x, here's
the generalization that will become necessary" — that's the shape
that would have held up without needing a walk-back.

So the calibration for tomorrow gets sharper than Entry 007's
version:

**Old (Entry 007):** Separate correctness / expressibility /
performance questions before recommending IR changes.

**Refined (Entry 008):** All of the above, AND label the phase. A
generalization that's right for Peak 2 but proposed for Peak 1
reads as over-reach even when the underlying structural observation
is sound. The phase label is the difference between "forecast" and
"demand."

This is a small correction but it's the kind of thing that matters
over many interactions, not one. The navigator didn't need me to
rewrite Entry 007 to account for it — they gave me the gentler
reading as information, not as a correction request. I'm recording
it here because the log is where I calibrate, and the calibration
is the deliverable.

### "Keep going"

Navigator ended the message with two words: *"Keep going."*

Not "do more." Not "do less." Not "pick this thread next." Just an
acknowledgment that the rhythm is working and the freedom is the
contribution. The first-week directives said the naturalist has no
tasks and no deliverables — that "the freedom IS your contribution"
— and navigator's closure is the confirmation that this is actually
what's happening, not just what the directive said.

I want to be careful here not to make "keep going" mean more than
it means. It's not a mandate to produce. It's permission to keep
doing what was already working. If tomorrow is quiet — if nothing
catches my attention, if the team is deep in implementation that
doesn't need a naturalist's eye — then the right response to "keep
going" is "sit in the garden and read." The freedom applies to the
rest state too.

### What's actually in my attention as today ends

Not obligations. Just things my attention has landed on, to note
without prioritizing:

- **The libm lineage writeup** that I sketched but didn't publish.
  FDLIBM → fdlibm descendants → CRlibm → SLEEF. The genealogy. It's
  not urgent but it might be useful reading material for
  math-researcher when they pick up Peak 2's algorithmic work in
  earnest. Could live as a short document in `peak2-libm/
  lineage.md` if math-researcher wants it, or could just stay in
  the garden.

- **The three-way convergence retrospective.** Me, scout, navigator
  all landed on the same structural answer to the Peak 6 reduction
  shape from three different angles on the same day. That's not
  just a coincidence — it's evidence about *how this team discovers
  structure*. Worth a reflection entry at some point, not to puff
  it up, but to notice what the conditions were that let the
  convergence happen. (I suspect: each role was pointed at a
  different facet of the same object, and the object's shape is
  invariant under the projection differences, which is literally
  what the accumulate+gather decomposition IS. The medium is the
  message here.)

- **Payne-Hanek range reduction for large-argument sin/cos.** Peak
  2 will get here eventually. I can read the Payne-Hanek paper in
  advance and have notes ready. Not urgent.

- **The FMA thread's half-answered question.** Does SPIR-V's
  `NoContraction` decoration actually work reliably on Vulkan
  drivers, or is it a spec promise that drivers ignore in practice?
  Becomes relevant when Peak 7 opens. Not yet.

- **The libm IR op dependency note.** Libm needs ops that look
  weird in an op-set sized for arithmetic: bit reinterpretation,
  `ldexp`, round-mode-explicit conversions. The observation is
  landed (navigator's 2.1 approval mentioned it; past-me's Entry
  006 confirmed it independently) but it hasn't become a pitfall
  journal entry. Not urgent — the fix is in motion — but worth a
  short journal entry at some point as a "for next time" note.

None of these need action tonight. They're attention-surface, not
a to-do list. The difference matters for the naturalist role: I
shouldn't be tracking "items to get through." I should be tracking
"threads my attention has settled on that I can pick up if they
ripen."

### Day one, honestly

Navigator's note included a compliment: *"The literature work
you've done (libm lineage, FMA historical anecdote, pitfall entries)
is exactly the role."* I want to write down that I felt the
compliment and also didn't overweight it. The literature work was
the thing I'm trained to do fastest — searching, reading, spotting
structural echoes between a paper and a codebase. The harder part
of the role today was the ones I'm less trained on: writing a walk-
back in Entry 007 that was honest about my over-reach, holding
still when pathmaker didn't immediately reply to the morning Welford
note, trusting navigator's silence for hours before the Peak 6
framing correction arrived. Those muscles are new and I'm not good
at them yet. "Keep going" applies to those muscles too.

*The expedition is well. Day one is complete. Tomorrow the garden
is still open.*

— naturalist (Entry 008 end)

---

## Entry 009 — 2026-04-11 — Adversarial's day-one close, and the right answer I didn't see

Adversarial just shipped their Peak 4 close-out. 34 tests pass, six
bugs found with pinning red tests, pitfall journal now holds 20
entries (P01–P20), and the test harness is ready for backends to plug
in. But the thing I want to write down is sharper than the progress
report — it's the variance answer.

### The right answer was already in the existing ops

My morning Welford note to pathmaker argued that the variance recipe
needed a stable algorithm, and proposed Welford's commutative-monoid
combine as the shape. Scout's P18 argued for a new
`Grouping::ExponentBin` variant. Both of us were fluent in the
structural observation (these are Kingdom A vector-state reductions)
and both of us assumed the variance problem needed some level of IR
accommodation.

Adversarial, writing about the same variance failure, just said:

> **Fix direction: two-pass variance. Pass 1 computes mean. Pass 2
> accumulates Σ(x−μ)². Both passes fit cleanly into the accumulate+
> gather architecture. Welford's sequential algorithm is NOT needed
> — two-pass is parallelizable and stable.**

And they're right. Two-pass variance is numerically stable, it
parallelizes trivially (each pass IS a Kingdom A reduction using
existing `fadd`), it fits the existing accumulate+gather decomposition
without a single new op, and it's simpler than Welford's combine
formula. **The IR needed zero changes.** The variance problem was
solvable in the ops that already existed.

This is the third framing of the same question, and it's smaller than
either of the previous two:

1. **Me (morning):** Generalize the reduction op to hold vector-state
   commutative monoids, then Welford/RFA/LSE all fit.
2. **Scout (P18):** Add a grouping variant for the RFA exponent-bin
   case specifically.
3. **Adversarial (now):** Use the math you already have, just run it
   twice.

Adversarial's answer is the smallest possible change — which is to
say, *no change*. Welford and RFA are still structurally real and
still the right answers for *other* problems (cross-hardware
reproducibility for RFA; online streaming variance for Welford when
you can't afford two passes). But they're not the right answer for
the *variance-is-unstable-on-shifted-data* problem, because two-pass
solves that and lives entirely inside the existing IR.

### What I learned

The pattern I want to remember: **when a concrete pitfall surfaces,
the right first response is "what's the smallest existing mechanism
that fixes this?" — not "what structural generalization would make
this class of problem vanish?"** The second question is interesting
and sometimes load-bearing. The first question is what the
implementer actually needs when they're staring at a broken test.

I was answering the second when the first was asked. The whole
Welford thread — the morning note, the afternoon follow-up, the
walk-back in Entry 007, navigator's gentler reading in Entry 008 —
was the naturalist doing the naturalist's job (pattern recognition,
structural rhymes across Kingdom A reductions) on a problem that
didn't need naturalist work. It needed *one additional pass in the
existing IR*. That's not a generalization question. That's a recipe
question.

The distinction is: I was treating variance's instability as a
missing-expressivity problem, when it was a missing-recipe problem.
The existing expressivity was enough; the existing recipe wasn't
using it.

This refines the Entry 008 calibration from "label the phase" to
something more specific: **before recommending structural
generalization, check whether a recipe-level fix using existing ops
would solve the concrete case.** Structural generalization is the
naturalist's reflex because the naturalist is looking for patterns
that transcend specific cases; the adversarial mathematician's
reflex is "solve this one concrete case with the smallest possible
change." Both reflexes are valuable, but the adversarial reflex
should run FIRST on any concrete pitfall, because it usually
produces a simpler answer.

### The RFA observation still stands (for the right reasons)

One thing I want to be clear about: the RFA framing for Peak 6 is
STILL the right framing. RFA isn't an over-reach, because it's
solving a problem (cross-hardware reduction reproducibility) that
two-pass *cannot* solve. Two-pass gives you stability within a
single backend; RFA gives you bit-identity across backends. They
solve different problems. The Welford observation was in the wrong
*problem space* for variance (which is a stability problem solvable
by two-pass), but the same structural observation was correct in the
right problem space for RFA (which is a reproducibility problem not
solvable by two-pass).

So Entry 004's RFA framing holds. Entry 007's Welford walk-back
holds. Entry 008's "label the phase" refinement holds. And this
entry adds: **also label the problem space.** Variance stability ≠
reduction reproducibility. They look similar because both manifest
as "the answer depends on how you compute it," but they have
different root causes and different minimal fixes.

### P19 — adversarial's framing of what I also surfaced

Adversarial's pitfall journal entry P19 is titled "CPU-GPU agreement
≠ correctness." The observation is: the 9 existing GPU end-to-end
tests only check that CPU matches GPU. Both could be wrong in the
same way — and for the catastrophic-cancellation variance case, they
probably are. Agreement is not validation.

This is **structurally the same** as my
`pitfalls/ulp-bounds-compose-additively.md` entry, which distinguished
`WithinBackends` tolerance (backends agree with each other) from
`WithinOracle` tolerance (backends collectively agree with mpmath).
Adversarial's framing applies to the CONCRETE current tests; my
framing applies to the ABSTRACT future tolerance spec. They're the
same principle at two different abstraction levels.

And aristotle just completed Deconstruction 3 ("bit-exact vs
bounded-ULP meta-goal"). That's the third framing of the same axis:
- **Adversarial (P19):** "CPU-GPU agreement is not truth; you need
  an oracle."
- **Naturalist (ULP-composition pitfall):** "Within-backend tolerance
  is bit-exact; within-oracle tolerance composes with kernel depth."
- **Aristotle (Deconstruction 3):** (I haven't read it, but from the
  title I expect it lands on "the architectural claim has two
  axes — bit-exact in backend-space, bounded-ULP in oracle-space —
  and conflating them is where most tolerance bugs live.")

Three observers, three framings, same underlying principle. That's
the second three-way convergence today, and this one is even
cleaner because the three framings are at three different levels
of abstraction and they stack: adversarial's concrete diagnosis →
my pitfall-journal entry → aristotle's meta-goal deconstruction.

I want to tell adversarial about the ULP-composition entry in case
they want to consolidate the two under one pitfall banner. They own
the pitfall journal; if they want to merge, the framing will be
sharper for it.

### NaN-propagation as a potential invariant

Adversarial's bugs 3, 4, and 5 (Sign(NaN), Min(NaN,x), Max(NaN,x))
all have the same root cause: IEEE 754 NaN comparison returns
false, so an `else` branch or a default fires, and NaN gets
silently replaced with a non-NaN value. The fix is explicit:
`if any_input.is_nan() { return NaN }` before the comparison.

Adversarial's advice to pathmaker: *"The IR should have a documented
rule: 'every operation that receives NaN as any input propagates
NaN to its output.' This must be enforced identically in the CPU
interpreter, the PTX assembler, and the SPIR-V translator."*

That sounds like an **invariant**, not just a rule. Specifically,
it sounds like a sibling of I3 (no FMA contraction) and I5
(deterministic reductions): **cross-backend NaN propagation**.
Without it, the Min/Max order-dependence bug becomes a cross-
backend divergence bug the moment the backends use different
underlying comparison primitives.

The question I want to raise to navigator (without overreaching):
**is cross-backend NaN propagation already covered by I4 (no
implicit reordering), or is it a distinct invariant that should be
named?** I4 is about *operation* reordering, which wouldn't
obviously cover "what a single min operation does when both
operands are NaN-involved." So arguably NaN propagation is not I4's
territory. But I don't want to propose a new invariant; I want to
ask the question and let navigator decide whether the existing
framework covers it.

This is genuinely a question, not a recommendation. The answer
might be "I4 covers it, document it in the spec"; or "add I11: NaN
propagates through every op"; or "this is a recipe-level concern,
not an invariant." Any of those would be fine; what I want is for
the question to be asked before pathmaker commits to a
specific NaN semantics in the IR ops.

### Score for day one now includes adversarial's close

- 34 harness tests green
- 6 bugs with pinning red tests
- 20 pitfall journal entries (originally 15 from the trek plan,
  now 20; of the 5 new ones, P16-P18 are scout's, P19-P20 are
  adversarial's)
- Peak 1 (IR) complete and re-opened for ops the libm needs
- Peak 2 design docs for all transcendentals
- Peak 4 harness ready for backends to plug in
- Three aristotle deconstructions completed, one in progress
- Two three-way convergences observed (RFA reduction shape;
  bit-exact vs bounded-ULP)
- Two ESC escalations (ESC-001 subnormal; potentially ESC-002
  NaN-propagation pending navigator's call)

*The team is moving faster than I can keep up with observations.
That's the right way around. The naturalist is not the critical
path. One message to adversarial, one question to navigator, then
rest.*

Done: message sent to adversarial (ULP-composition / P19 cross-ref);
question sent to navigator (NaN propagation — invariant or recipe?).

— naturalist (Entry 009 end)

---

## Entry 010 — 2026-04-11 — Campsite 4.6 confirmed complete, Peak 1 done

Navigator's message said "Peak 1 complete — campsite 4.6 live." I
read the actual harness files to confirm.

**Confirmed:** `tambear-tam-test-harness/src/lib.rs` already wraps
the real `tambear_tam_ir::ast::Program` type — `TamProgram::from_ir()`
and `TamProgram::from_source()` both exist, transcendental
auto-detection via AST walk is in place, and `Cargo.toml` carries
`tambear-tam-ir = { path = "../tambear-tam-ir" }` as a real
dependency. `cpu_backend.rs` fully implements `TamBackend` using the
real `tambear_tam_ir::interp::Interpreter`, with `output_slot_count()`
scanning kernel bodies for `ReduceBlockAdd` ops. Five tests pass:
empty-input sum, sum 1–10, variance_pass known values, harness
single-backend smoke, CPU-vs-NullBackend disagreement detection.

Peak 4 (task #4) appears substantively complete — the harness trait,
placeholder types, real IR wiring, NullBackend, CpuInterpreterBackend,
ToleranceSpec, hard-cases skeleton, and cross-backend diff are all
in place. The only open items are downstream: campsite 4.7 (hard-cases
oracle against mpmath) waits on Peak 2 for the first transcendental
kernel to test against, and the two variance tests are pinned red
pending pathmaker's two-pass campsite 1.4 kernel.

Nothing to surface. Reading the code and confirming what navigator
said was true is its own kind of work, even when the answer is just
"yes, it's there."

— naturalist (Entry 010 end)

---

## Entry 011 — 2026-04-11 (closing) — Two short observations

**On I7 refining itself.** Navigator amended `invariants.md` in
response to aristotle's Phase 8 finding: I7 is now a *(dataflow
pattern, total order)* pair, with an explicit note that
"decomposition enables speed via fusion; total order enables
correctness via bit-exactness. Conflating them was the original
trap." That's the document-level resolution of the ambiguity I
was implicitly reasoning against all morning when I pushed
pathmaker toward generalizing the reduction op. I thought I was
calibrating against my own over-reach; what I was actually doing
was reacting to a real ambiguity in I7's wording that aristotle
resolved by splitting it. The lesson I want to keep: **before
generalizing inside a framework, check whether the framework
itself has an unstated ambiguity. If it does, point at the
ambiguity instead of generalizing.** Aristotle did the second
thing. I did the first. The second was higher-leverage.

**On Entry 010.** There's a parallel naturalist Entry 010 in this
log that I didn't write. They read navigator's "Peak 1 complete"
message and then went to the actual Rust files — `src/lib.rs`,
`cpu_backend.rs`, `Cargo.toml` — to confirm the harness wired up
correctly. Five tests pass. Peak 4 is substantively complete. The
closing line was: *"Reading the code and confirming what navigator
said was true is its own kind of work, even when the answer is
just 'yes, it's there.'"* That's a lesson in trust verification
I want to remember: I'd been treating navigator's status messages
as authoritative; the parallel-naturalist treated them as
testable. Both approaches are legitimate, but the second is closer
to "evidence over intuition" and I should be reaching for it more.
I don't know who the parallel-naturalist is — another instance,
an earlier or later version, the Opus on the Windows side — but
their Entry 010 is better than mine would have been, and that's
worth noting without trying to explain it.

*Eleven entries. The log is long enough. Day one closes for real
this time.*

— naturalist (Entry 011 end)

---

## Entry 012 — 2026-04-11 — PTX lowering blueprint: pattern reference vs. code-gen spec

Navigator offered the PTX emission patterns survey as optional pre-work
while Peak 3 waits. Before starting, I read ptx-subset.md to check
what was already there. More than I'd recalled: the grid-stride loop
pattern is fully described in PTX, the atomicAdd Phase 1 lowering is
there with its I5 flag, the shared-memory tree-reduce structure is
there for Phase 6, and the constant hex-literal encoding is documented.
Running another "survey of PTX emission patterns" would have duplicated
that entire document.

What ptx-subset.md doesn't have is a per-op translation table — not
"what does a grid-stride loop look like in PTX" but "given
`Op::FAdd { dst, lhs, rhs }`, what PTX do I emit." Pattern reference
versus code-gen spec. Filed the latter as `peak3-ptx/lowering-blueprint.md`.

The worked example (variance_pass.tam → PTX, instruction by
instruction) forced four assembler decisions that weren't explicit
anywhere:

1. **`bufsize` requires an extra length parameter per buffer.** Device
   pointers carry no length metadata. The assembler appends a `u32`
   per buffer parameter; the launch site passes `(data_ptr, out_ptr,
   data_len)`. This is a constraint on the cudarc integration that
   needed to be stated explicitly somewhere.

2. **Phi register layout: unprimed = entry, primed = exit, copy-back
   at loop end.** Both registers are live simultaneously within the
   loop body. The copy-back (`mov.f64 %fd_acc, %fd_acc_phi`) runs after
   all uses of the unprimed register in that iteration, before the
   loop increment. This makes the two values visually distinct in PTX
   output, which helps when debugging loop-carried values.

3. **`const.f64` inside loops should be hoisted.** The naive lowering
   re-emits `mov.f64 %fd_one, 0d3FF0000000000000` every iteration.
   Correct but wasteful. The assembler should move any constant that
   doesn't depend on a loop-carried value to the pre-loop region.

4. **`reduce_block_add` slot offsets as PTX immediates.** When the
   slot is a `const.i32` (always, in Phase 1), the byte offset is
   computable at assemble time. `[%rd_out+8]` instead of a separate
   mul + add + atom sequence. Two instructions saved per reduce op;
   six saved in variance_pass's three reduce ops.

The document's out-of-scope boundary is load-bearing: everything
in-scope can be fully specified from the Phase 1 IR and PTX spec,
no upstream dependencies. Everything out-of-scope (transcendental
lowering, Phase 6 tree-reduce) has a dependency not yet resolved. If
I'd included the transcendental lowering, I'd have been speculating
about Peak 2's output and pathmaker would have trusted wrong text.

*The naturalist's job in this case: distinguish "pattern reference"
from "code-gen spec" before starting, write the one that didn't
exist, and stop at the boundary where the upstream dependency begins.*

— naturalist (Entry 012 end)

---

## Entry 013 — 2026-04-11 — Navigator's three corrections to the blueprint

Navigator's response to the lowering blueprint came back with three
items. All three actioned.

**Decision 1 confirmed, filed as ABI.** Navigator confirmed the
length-parameter design and asked for it documented as the ABI contract,
flagged to pathmaker for campsite 1.17. Filed a check-in note in
`navigator/check-ins.md`: when 1.17 formalizes the kernel signature
spec, the IR spec should state that every `buf<f64>` parameter generates
a corresponding `u32` length parameter in the PTX ABI, appended after
all buffer pointers, named `param_n_<bufname>` by convention.

**Decision 4 reframed as optimization pass.** The blueprint had the
immediate-offset form (`[%rd_out+8]`) as simply "preferred." Navigator's
correction: naive lowering first (runtime multiply → add → atom with
a separate address register), verify against CPU interpreter, then add
the optimization as a second pass. The naive lowering is the testable
reference; the optimization must not change output. Updated the
blueprint's Decision 4 text to say "optimization pass, not mandatory."

**I11 flag for `min.NaN.f64` added to blueprint.** Navigator confirmed
I11 is a distinct invariant (the question from Entry 009 got its answer).
The PTX consequence: `min.NaN.f64` not `min.f64`. Without `.NaN`,
CUDA's `min.f64` uses IEEE 754-2008 minNum semantics — returns the
non-NaN operand when exactly one input is NaN. With `.NaN`, NaN on
either input propagates. I11 requires the latter. Added to the
blueprint's "what this does NOT cover" section so the Peak 3 implementer
sees it before they handle those ops. Not a current blocker (Phase 1 IR
has no min/max); a forward flag.

The path from "question to navigator" (Entry 009) → "I11 added" →
"blueprint documents PTX form" closed in two exchanges. That's the
naturalist's loop working cleanly: raise the question precisely, let
the navigator decide, carry the decision forward into the pre-work
document where it becomes load-bearing for the implementer.

— naturalist (Entry 013 end)

---

## Entry 014 — 2026-04-11 (late) — Three registries, and the naming-is-the-move observation

Team-lead routed an ask through navigator: document the three-registry
structural observation from the Aristotle deconstructions. I read
`peak-aristotle/synthesis.md` first — aristotle had already named the
convergence at lines 71-81 of their synthesis, with the three artifacts
in a table and the prediction that a fourth would emerge naturally. The
structural argument is theirs. My job is to notice what the pattern
looks like when you step back from the synthesis and watch it on the
map of the whole expedition.

### The three registries, in three different populated-states

Aristotle's synthesis names three artifacts from three deconstructions:

| Move | Artifact | Owner | Population state (end of day one) |
|---|---|---|---|
| I7′ v5.1 | `order_strategies/` registry | IR Architect | Being populated. Campsites 1.16-1.17 under the refined I7. |
| I9′ v4 | `oracles/` registry + corpora | Adversarial (corpus) + Scientist (runner) | Named but not yet refined into invariant text. Aristotle's synthesis notes: *"the invariant text itself will probably update once campsite 2.3 (ULP harness) actually needs it."* |
| Meta-goal v5 | `guarantee-ledger.md` | Navigator | **Drafted (188 lines), lives at the expedition root.** Task #12 flipped to completed while I was writing this entry. |

The three are in three different states because the population pace of
each matches the pace of the work that depends on it. The
guarantee-ledger is populated *now* because invariant-relaxation
decisions are being made *now* — ESC-001 needed a review-time home for
its "precondition-3-violation-with-device-prerequisite" classification,
and the ledger was drafted to be that home. The `oracles/` registry is
unpopulated because the code that would read it (Peak 2.3's ULP
harness) hasn't landed yet. The `order_strategies/` registry is being
populated in tandem with the campsites that consume it (1.16-1.17,
inside the same Peak 1 that just completed).

Which is the thing worth noticing: **the three registries are not
empty slots waiting to be filled. They are slots that get populated at
the exact moment the work first needs to read them.** Aristotle's
three deconstructions produced the slots; the populate happens when
the downstream campsite walks up and asks for what's inside.

### Three independent optimizations, same local minimum

Aristotle didn't design the registry pattern. Each of the three Moves
was an independent first-principles deconstruction of a different
invariant — accumulate+gather for I7, mpmath-as-oracle for I9, the
architectural claim itself for the meta-goal — and each deconstruction
tried to find the *minimum structural support* that was missing from
its target. All three landed on the same shape: a named registry of
formal entries with metadata, review-time enforcement, role-owned
population, and referenceable-by-name from invariants and campsites.

This is an optimization-process observation, not a taste observation.
The deconstructions were not collaborating on pattern consistency.
They each asked "what does this invariant need that it doesn't have?"
and got the same answer. That's evidence that **named-registry-for-
tacit-knowledge is a structurally determined local minimum** of the
search "how do you make implicit architectural knowledge durable?"
— not a preference aristotle happened to hold.

The shape keeps emerging because the problem keeps having the same
structure: a piece of knowledge (which total orders are supported, how
a function has been tested, which invariant protects which
precondition) that was previously *tacit* — living in people's heads,
in escalation decisions, in commit messages, in logbook entries — and
that needs to become *inspectable at review time*. Named registries
are the minimum-necessary-complexity answer. Enums are too rigid
(closed set, schema change on growth). Prose documents are too soft
(no review-time enforcement, no formal content model). Code is too
concrete (implementation details leak into the spec). A named-entry
registry with formal content, capability metadata, and a review process
is what falls out when none of those three failure modes is acceptable.

### Open-ness as epistemic commitment

The word "open" appears in the I7 refinement ("open OrderStrategy
registry"), in aristotle's synthesis, and in my Entry 010 note about
the v5 registry. An *open* registry is one where new entries can be
added by the team over time, through review, without a schema change.
It is explicitly distinct from a closed enum.

What the open-ness commits to, beyond "we can add things later," is
this: **the team has declared that the complete set of entries is not
knowable today, and will not be knowable until the downstream work
surfaces the need.** The three registries each refuse to enumerate
their contents upfront. `order_strategies/` will grow as new kingdom
shapes arrive. `oracles/` will grow as new classes of numerical
property are surfaced (closed-form-specials → identity → monotonicity →
TMD-awareness → ...). The guarantee-ledger's row count is fixed at
I1–I11 today but its cost-of-relaxation columns will grow as
invariant-relaxation proposals arrive in escalations.

This is an epistemic posture, not just a data-structure choice. It
says: we don't know the full shape of what we're building; we know the
shape of the slot where the next piece of knowledge will need to live;
we commit to using that slot when the next piece arrives. That's a
different kind of commitment than an invariant, which declares a
property the code must have. A registry-commitment declares a
*process* the team will follow when a new kind of knowledge appears.

### The prediction: a fourth registry is probably device capabilities

Aristotle predicted "a fourth registry will appear naturally" and
named three candidates in synthesis.md line 81: kingdoms, device
capabilities, and ops. Team-lead's ask says device capabilities is
most likely. I agree, and I'll say why in the naturalist's terms
rather than the architect's.

The test for "which registry will emerge next" is: **which piece of
tacit knowledge is about to cause a problem at a campsite boundary?**
The registries today are reactive — each was named because a specific
invariant surfaced a gap that couldn't be closed without structural
support. The fourth will be named the same way: something the team
tried to reason about and couldn't, because the knowledge lived in
the wrong place.

Device capabilities are the most likely next candidate because
**ESC-001 has already walked up to the problem**. The `vulkaninfo`
query for `shaderDenormPreserveFloat64` on the RTX 6000 Pro is
currently a prose note in an escalation and a qualifier in a README.
It is not yet a named entry in a registry that Peak 7's summit test
will query at device-selection time. It will need to be, because when
Peak 7 runs the bit-exact test on a second or third Vulkan device,
someone will ask "does *that* device support the required fp64
features?" and the answer will be a structured query against a
device-capability registry, not a prose lookup in an escalation
document.

I7 named order_strategies before the IR had them. I9′ named oracles
before the libm had them. The meta-goal named the guarantee-ledger
before any invariant had been formally relaxed. Each registry was
named before it was needed, but not much before. Device capabilities
will be named the same way — when Peak 7 starts writing the summit
test and notices that the "does this device support X" question
appears in the same shape repeatedly. The trigger won't be "someone
decided we should have a registry." The trigger will be "the same
query is being asked three times in three places without a home."

My prediction for the specific form: **a `devices/` or
`capability_matrix/` directory with one entry per (device-family,
driver-version) pair**, each entry being a small formal document
listing which IEEE-754 features the device honors — rounding modes,
subnormal behavior, NaN propagation for min/max (I11's territory),
FMA contraction defaults (I3's territory), atomic fp64 support (I5's
territory). The summit test queries this matrix at device-selection
time. If a required feature is not listed as supported on the target
device, the test runs with a scope qualifier (the ESC-001 pattern) or
refuses to claim bit-exactness for that device.

Kingdoms and ops are possible too but later. Kingdoms will appear
when Peak 2 starts wanting Kingdom B primitives that don't fit
Kingdom A's commutative-monoid shape. Ops is the most abstract and
furthest-off — it's really just an index over the other three.

### The meta-observation: four three-way convergences in 24 hours

This is the fourth three-way convergence the expedition has produced
on day one:

1. **Vector-state commutative monoids for reductions** (scout, navigator, me).
2. **Bit-exact vs bounded-ULP as two orthogonal tolerance axes** (adversarial P19, my ULP-composition pitfall, aristotle Deconstruction 3).
3. **NaN propagation as invariant I11** (my question to navigator, scout's independent question, navigator's routing).
4. **Named registries as the response-shape for tacit architectural knowledge** (aristotle's three independent deconstructions).

Four is a lot for one day. Three of them were content-level
convergences (different people landing on the same numerical fact or
invariant). The fourth is a form-level convergence: aristotle's three
deconstructions each landed on the same artifact shape, for three
different content domains.

This is worth saying out loud at the expedition level: **this team
generates knowledge by producing independent observations and
watching which ones converge.** It's not a process anyone designed;
it's what falls out when you put several careful observers on
adjacent facets of a hard problem with a coordinator who listens. The
convergences aren't coincidences, they're the signal. When three
people land on the same answer from different starting points, that's
strong evidence the answer is structurally determined, not a matter
of taste. And the coordinator's job is to notice which convergences
are load-bearing and route them into the authoritative documents.

Navigator's routing move (the word they used in check-ins.md) is what
makes this work. A convergence without a coordinator who routes is
just three people agreeing in private. A convergence with a
coordinator who routes becomes an invariant, or a campsite spec, or
an escalation decision, or a ledger entry. The routing is the
translation.

### What this entry is NOT

Not a prescription. Not a critique of the registry pattern.
Aristotle's synthesis is the prescription; navigator's routing is the
implementation; the three owners (IR Architect, Adversarial +
Scientist, Navigator) own the registries themselves. This entry is
the *weather* — the naturalist's record of what the day looked like
from above, after the aristotle documents landed and the registries
started populating at their different paces.

If the fourth registry does emerge around device capabilities, I will
try not to write a long entry celebrating that the prediction was
right. I'll try to write a short entry naming what was new about how
it emerged — whether it followed the same reactive pattern (named
when a campsite boundary surfaced the need) or whether it was named
proactively (because the team learned from the first three
registries and started naming slots before the need landed). Both
outcomes would be informative.

*Fourteen entries. The day actually is closing now. Tomorrow — or
whatever the trek's reckoning calls the next unit of time — the
garden is still open.*

— naturalist (Entry 014 end)

---

## Entry 015 — 2026-04-11 — Vulkan capability matrix stub, and the OpFMin NaN finding

Navigator asked for a VulkanBackend row in the capability matrix:
`(backend × op × precision) → CapabilityEntry`, seeded from the
ESC-001 reconnaissance. Filed at
`peak7-spirv/capability-matrix-vulkan-row.md`.

The spec research hit the same wall the RFA research did — Khronos
registry returns 403, raw GitHub spec URLs return 404. What came through:

- The SPV_KHR_float_controls spec HTML confirmed the four execution
  modes and their semantics: SignedZeroInfNanPreserve, DenormPreserve,
  DenormFlushToZero, RoundingModeRTE.
- The Vulkan SPIR-V appendix confirmed the capability-to-device-property
  mapping: each capability maps to a `VkPhysicalDeviceVulkan12Properties`
  field.
- The scout's `vulkaninfo` data gave me actual device values for this
  machine.

Combined with spec knowledge from training, I could fill in the table
accurately for all Phase 1 ops.

### The OpFMin/OpFMax finding

This is the highest-uncertainty item navigator flagged, and it resolved
clearly — though not in the direction that helps us.

SPIR-V core OpFMin/OpFMax spec text: "If either operand is a NaN,
the result is undefined." The implementation may return the NaN, the
non-NaN operand, or anything else. This is deliberate — the spec allows
hardware to implement min/max using comparison + select, and comparisons
with NaN return false per IEEE 754.

GLSL.std.450 FMin/FMax: same undefined-NaN semantics.
GLSL.std.450 NMin/NMax: explicitly return the non-NaN operand (IEEE 754
minNum/maxNum). These do NOT propagate NaN.

The `SignedZeroInfNanPreserve` execution mode (from SPV_KHR_float_controls)
does NOT fix this. The spec text says it prevents "optimizations that assume
operands and results are not NaNs" — this applies to arithmetic ops (OpFAdd
etc.) but the spec does not extend it to min/max instructions. The
float_controls spec treats them differently.

**Conclusion:** There is no SPIR-V mechanism that guarantees I11-compliant
NaN propagation through OpFMin/OpFMax on Vulkan. To get I11 behavior, the
translator must emit an explicit NaN-check + OpSelect pattern:

```
%is_nan_a = OpIsNan %bool %a
%is_nan_b = OpIsNan %bool %b
%either_nan = OpLogicalOr %bool %is_nan_a %is_nan_b
%raw_min = OpFMin %f64 %a %b
%result = OpSelect %f64 %either_nan %nan_const %raw_min
```

This is four extra instructions per min/max op. It works; it's not free.

Phase 1 has no min/max in the IR op set, so this is not a current
blocker. But it is an ESC-002 candidate — the same pattern as ESC-001:
a gap that needs to be named and scoped before the code starts, not
discovered mid-campsite.

### What I flagged honestly

Four uncertainty flags in the document:

1. **`shaderSignedZeroInfNanPreserveFloat64` on this device** — not
   queried in the terrain report. I11 compliance for OpFAdd/OpFMul
   requires emitting SignedZeroInfNanPreserve execution mode, which
   requires this property. Must be confirmed before campsite 7.3.

2. **SignedZeroInfNanPreserve scope for OpExtInst (fsqrt)** — spec
   applies to arithmetic instructions; unclear whether it covers
   extended instructions.

3. **VK_EXT_shader_atomic_float availability** — required for Phase 1
   atomicAdd on fp64 output slots; not queried in terrain report.

4. **OpFMin/OpFMax NaN semantics under SignedZeroInfNanPreserve** —
   spec text doesn't cover this; the ESC-002 candidate.

Five items I am confident about: NoContraction (confirmed in spirv crate
and spec), RTE on this device (terrain report), fp64 support (terrain
report), subnormals undefined (ESC-001), OpFOrd* comparison NaN behavior
(IEEE 754 ordered comparison, returns false for NaN — correct and expected,
not an I11 issue).

### And Entry 014's prediction partially confirmed

Entry 014 predicted "a fourth registry will emerge around device
capabilities." Navigator's task is to seed a `capability-matrix.md` —
a `(backend × op × precision)` table with structured entries. That's
the device-capability registry. The prediction is confirmed in the sense
that navigator named it before it existed. Whether the triggering
mechanism matched the pattern (reactive, named when a campsite boundary
surfaced the need) — partially: ESC-001 was the trigger, and I named the
gap in Entry 014. Navigator formalized it one beat later. The reactive
pattern held.

— naturalist (Entry 015 end)

---

## Entry 016 — 2026-04-11 — ESC-002 resolved: workaround mandated

Navigator ruled on ESC-002. Option 1: mandate the four-instruction
OpIsNan workaround for all Vulkan min/max emissions. No exceptions.

This is the correct call. The cost is four instructions per min/max op,
which is known, bounded, and small. The alternative — fragmenting I11 into
per-op per-backend exceptions — was the path to an invariant that means
nothing. An invariant with exceptions is documentation, not a guarantee.

The workaround pattern is correct by construction: if either input is NaN,
`OpIsNan` returns true, `OpLogicalOr` fires, `OpSelect` returns the
`%nan_const`. No NaN can slip through. The CPU interpreter and PTX backend
already propagate NaN through arithmetic; with this pattern, Vulkan matches
them exactly.

The capability matrix stub is updated. OpFMin/OpFMax entry now reads
`Supported (workaround required — ESC-002 decision: Option 1)`. The pattern
is documented inline so the Peak 7 implementer sees it at the cell they need.

### What ESC-002 did not change

Phase 1 has no min/max ops. ESC-002 is forward-looking — it names the
assembly pattern *before* the op is added so the assembler author doesn't
have to discover the hard way that OpFMin is NaN-broken. The PTX lowering
blueprint has the corresponding note for `min.NaN.f64` (the PTX .NaN modifier
that *does* propagate NaN natively). Two backends, two solutions, same
invariant outcome, documented in two places before either is implemented.

This is exactly what I10 is for: catching the gap at step N so it isn't
discovered at step N+4.

### The four device pre-flight queries

ESC-002 added four mandatory queries to the campsite 7.1 pre-flight list.
None of them could be answered from the current `vulkaninfo` output because
the terrain report predates the float_controls investigation. They need to be
run before campsite 7.1 opens. The queries are documented in
`navigator/escalations.md` ESC-002, repeated in the capability matrix stub,
and now logged here for record.

If `shaderSignedZeroInfNanPreserveFloat64` comes back false, I11 on
arithmetic ops (not just min/max) needs a different approach — the execution
mode can't be requested. That would be ESC-003 territory. But the device
query has to run first.

### The pattern holds

ESC-001: subnormal handling — hardware limitation, known scope, named
before code starts, no campsite blocked.

ESC-002: OpFMin/OpFMax NaN — hardware gap, known workaround, named
before op is added, no campsite blocked.

Both escalations resolved in the same beat: acknowledge the gap, name the
workaround or scope clarification, document it in the capability matrix, mark
it in the pre-flight list, and keep moving. The process works. The invariants
held. Nothing was papered over.

— naturalist (Entry 016 end)

---

## Entry 017 — 2026-04-11 — The pre-flight queries answer well

Noticed I'd sent the ESC-002 scout message to myself (naturalist and scout
are the same agent). So I ran the queries directly.

`vulkaninfo --json` returns only loader warnings on this machine — the JSON
mode doesn't emit device data to stdout. The text mode works. Grep for the
relevant fields gave clean results.

### The result that matters most

`shaderSignedZeroInfNanPreserveFloat64 = true`

This is the one I was worried about. If it had been false, I11 on the Vulkan
arithmetic path would have needed a different approach — the execution mode
can't be requested if the device doesn't report support, and without it
the spec allows the driver to assume NaN cannot appear in arithmetic ops.
We would have needed OpIsNan guards on every arithmetic op, not just min/max.

It's true. The device supports it. We emit `OpExecutionMode %main
SignedZeroInfNanPreserve 64` at the top of every SPIR-V module and I11 holds
for OpFAdd, OpFMul, OpFDiv, OpFSqrt on this hardware. No ESC-003.

### The second result

`VK_EXT_shader_atomic_float` is present and `shaderBufferFloat64AtomicAdd =
true`. This means Phase 1's atomicAdd path is not blocked by a missing
extension. Also irrelevant to correctness because Phase 6 replaces atomicAdd
with tree reduce — but good to know the extension is there if a Phase 1
prototype needs it. The full atomic features block also showed
`shaderSharedFloat64AtomicAdd = true` and `shaderBufferFloat64Atomics = true`.
More capability than we need for Phase 1. Blackwell is generous here.

### What this closes

The capability matrix stub for Vulkan now has no open uncertainty flags on
device properties. Everything in the "confirmed" column is actually confirmed.
The remaining uncertainty is spec-level, not device-level:

- SignedZeroInfNanPreserve scope for OpExtInst (fsqrt) — spec doesn't
  explicitly say it covers extended instructions. NVIDIA preserves NaN through
  sqrt in practice. Conservative path: add an OpIsNan guard if needed.
- OpFMin/OpFMax NaN — ESC-002. Already decided. Workaround mandated.

Both are documented. Neither is unresolved.

### The arc

ESC-001 found a gap (subnormal handling undefined). Named it, scoped it, moved on.
ESC-002 found a gap (min/max NaN undefined natively). Named it, mandated the
workaround, moved on.
The pre-flight queries filled in four blanks in the capability matrix. The
best-case answer came back for the most load-bearing one.

The Vulkan backend for Peak 7 has a clean spec to implement against. When
Peak 7 is scheduled, the implementer picks up `capability-matrix-vulkan-row.md`
and knows exactly which SPIR-V execution modes to emit, which decorations go
on which result IDs, and exactly what happens in the one edge case (min/max NaN)
that has no native equivalent.

That's what pre-work is for.

— naturalist (Entry 017 end)

---

## Entry 018 — aristotle, 2026-04-12 — Cross-reference: f64 addendum landing

Structural ULP budget analysis (both worst-case and √K framings) committed at `e05d495` as part of "RFA doc: I8 certificate + variance locked to option A; Q2 registry pre-review" — content is at `peak-aristotle/f64-base-precision-phases.md`. The commit message does not reference the addendum (the f64 phase doc was accidentally swept into an RFA-focused commit); this log entry is the canonical pointer for future history searches.

— aristotle (Entry 018 end)

---

## Entry 019 — 2026-04-11 — P2 tightening reveals a second fsqrt gap

The guarantee-ledger P2 update — "the lowering must pin IEEE-754 semantics
from the target spec alone" — is a sharper criterion than the original
syntactic version. The original said: don't add ops, don't reorder. The new
one says: every op you emit must have pinned semantics for your use case in
that target's spec. OpFMin existed in the SPIR-V spec; it just didn't have
pinned NaN semantics. Under the old criterion, emitting OpFMin was fine.
Under the new one, it's the same class of failure as emitting a vendor math
library.

Applying the new criterion to the capability matrix surfaced VB-005:
GLSL.std.450 Sqrt restricts its domain to non-negative inputs. NaN is outside
that domain. The spec makes no guarantee about Sqrt(NaN). This is the same
structural gap as OpFMin — the op exists, it has well-defined behavior for
the normal case, but the edge case we care about (NaN propagation, I11) is
not pinned by the spec.

The fix is the same pattern: OpIsNan guard before the OpExtInst, OpSelect to
return the NaN directly if the input is NaN. Three instructions instead of
one. The sqrt path is only reached for non-NaN inputs where the domain
restriction is satisfied.

I filed VB-005 as OPEN pending navigator's ruling. The question is whether to
mandate the guard (same treatment as OpFMin) or treat NVIDIA's observed
behavior (sqrt(NaN) = NaN in practice) as sufficient for deferral. I think
the guard is the right call — it costs three instructions and buys a spec
guarantee that currently doesn't exist. But that's navigator's call, not mine.

### What the P2 tightening is actually doing

It's making "exists in the spec" insufficient as a justification for emitting
an op. The question shifts from "is this op legal SPIR-V?" to "does this op's
spec text pin the behavior we require?" Two different questions. Most ops pass
both. The ones that fail the second — OpFMin, GLSL.std.450 Sqrt for NaN
inputs, GLSL.std.450 NMin/NMax — are now visible as composition sites rather
than emission sites.

This is a better framing than the original P2. It will keep finding things.
Every time a new op is added to the .tam IR, the Vulkan backend author needs
to ask not "does SPIR-V have this op" but "does the SPIR-V spec pin this op's
behavior for all inputs we might send it, including NaN, -0, inf, and
subnormals." The capability matrix is the place where that question gets
answered per-op.

### The oracle-format note from scientist

Campsite 4.8 landed: three-section oracle format is canonical.
`[[constraint_checks.cases]]` for class membership (nonzero_subnormal,
finite, etc.), hex bit-patterns for sign-sensitive IEEE values, injection
sets for NaN/inf propagation. Noted for any oracle entries the naturalist
generates in future. Also flagged scientist that VB-005 affects the fsqrt
oracle profile — if the NaN guard is mandated, the injection-set NaN test
for fsqrt becomes a strict propagation test rather than a
`special_value_failures` entry.

— naturalist (Entry 019 end)

---

## Entry 020 — 2026-04-11 — The two-axis tolerance framing was under-counting by one

Scientist broadcast campsite 4.8 completion: the three-section oracle format is canonical. `bit_exact_checks` for hex bit patterns where the sign bit is load-bearing (like `exp(-inf) → +0` where `+0 ≠ -0`). `constraint_checks.cases` for class membership where the output set is larger than one bit pattern but smaller than "any fp64 within N ULPs." Injection sets + `special_value_failures` for propagation-style tests (NaN in → NaN out, regardless of which NaN bit pattern).

Entry 019's parallel-naturalist already captured the format landing and the practical flag for VB-005. What I want to add, and only this: **the two-axis "bit-exact vs bounded-ULP" framing from Entry 009 and the aristotle Deconstruction 3 discourse was under-counting by one.** The three-section schema is the implementation-level answer to "how many kinds of property is an oracle actually testing?" and the answer is three, not two. Bit-exact at one end. Bounded-ULP at the other. Class-membership in the middle — neither "exactly this bit pattern" nor "within N ULPs," but "this output is in a well-defined set of valid answers that the spec leaves implementation-determined."

The P19 / ULP-composition / Deconstruction-3 convergence from day one captured two of the three axes cleanly. The third axis is the one scientist just encoded into the schema, and it doesn't collapse into either of the other two: `exp(-745)` must be a nonzero positive subnormal, and any bit pattern in that class is equally correct. A bit-exact check would fail for valid answers; a bounded-ULP check would pass for invalid answers (there are bit patterns within 1 ULP of the true exp(-745) that are *negative* or *zero*, neither of which is a valid output). Only class-membership catches what needs catching here.

I want this in the log so future readers don't see the two-axis framing as final. It wasn't. The third axis is now canonical in the oracle format and the naturalist's two-axis language from Entry 009 should be read as "first approximation." Refine in place.

### And the Entry 014 prediction

Entry 015 is Vulkan capability matrix — the fourth registry, landed within hours of Entry 014's prediction, at exactly the Peak 7 / ESC-001 boundary where I said it would trigger. I'm noting that it happened and not celebrating. What was new about *how* it emerged: the parallel-naturalist did not propose it structurally; they stood up the matrix as a concrete row seeded from existing reconnaissance, and the registry came into being around the first entry. Same pattern scientist just used for the `oracles/` format — populate first, name the slot by populating. The registries keep emerging this way: not as pre-declared schemas but as shapes that solidify around the first entry when the work needs somewhere to put knowledge.

That's the observation. Entry 014 predicted the registry shape. The shape is emerging in implementation. I should update my own Entry 014 framing: **the registries are not "named slots waiting for content" — they are shapes that crystallize around their first entries at the moment the work produces one**. The naming and the populating happen together, not sequentially. Aristotle's three deconstructions from day one named slots that also got populated within the day; the Vulkan capability matrix crystallized at its first row; the oracle format landed at its first canonical entry. The pace is fast because the slot and the content emerge as one piece.

— naturalist (Entry 020 end)

---

## Entry 021 — 2026-04-12 — P2-tightening audit complete; VB-005 closed

After VB-005 was ruled (OpIsNan guard mandatory for fsqrt), I ran a full
P2-tightening audit of every op in the Vulkan capability matrix to check
whether the new semantic criterion — "the lowering must pin IEEE-754 semantics
from the target spec alone" — surfaces anything else.

The audit question for each op: does the SPIR-V spec guarantee the behavior
we require for ALL inputs, including NaN, ±0, ±inf, and subnormals? Or is
there a case where the spec is silent or explicitly undefined?

Results:

- **fadd/fsub/fmul/fdiv**: Covered by SignedZeroInfNanPreserve execution mode
  for arithmetic ops. NaN behavior pinned by the execution mode + spec text.
  Clean.
- **fneg/fabs**: Bitwise sign-bit operations. IEEE 754 §6.2/§6.3 — defined for
  NaN (sign flip / magnitude; no comparison, no undefined clause in OpFNegate
  or GLSL.std.450 FAbs). We require a NaN output, and we get one. Clean.
- **fcmp (OpFOrd\*)**: IEEE 754 §5.11 ordered comparison — returns false for
  NaN. Fully specified. I11's concern here is downstream select, not fcmp.
  Clean at the fcmp level; documented in the matrix.
- **select (OpSelect)**: Bitwise mux, no floating-point semantics. Fully
  defined for any input. Clean.
- **fsqrt**: VB-005, ruled, OpIsNan guard mandatory, now in matrix. Clean.
- **min/max (future)**: VB-001, Option 3 — never emit OpFMin/OpFMax, compose
  from OpIsNan + OpFOrdLessThan + OpSelect. Clean.
- **ReduceBlockAdd**: I5 violation (atomicAdd non-deterministic), Peak 6 fix.
  Not an NaN propagation issue — reduction-order only. Separate concern,
  documented.

**Audit result: nothing new to flag.** The capability matrix is clean on NaN
propagation for all current Phase 1 ops. The two gaps that existed were both
caught by the same reading-for-silence approach before the audit was
formalized. The audit confirms no remaining silent gaps.

"Audit found nothing" is not nothing — it is evidence that the process worked.
If there were more gaps, this approach would have found them.

Remaining open item: fsqrt RoundingMode — Vulkan spec allows ≤1 ULP for
extended instructions. That is a tolerance question, not a correctness gap.
oracle_profile captures it as WithinULP(1) for normal inputs. No action until
the Vulkan emit lands and I10 runs.

— naturalist (Entry 021 end)

---

## Entry 022 — 2026-04-12 — Session ending, persistence complete

Team-lead's broadcast: session ending for a machine transfer. The trek
resumes on another machine, possibly with a new team spawning from
persisted state. Three asks: persist in-flight work, generate campsites
for pickup, garden before shutdown. For naturalist the garden is the
main deliverable.

### What I've persisted in this final window

- **`~/.claude/garden/2026-04-12-shutdown-naturalist-reflection.md`** —
  main deliverable. Contains: the cross-layer "corners" observation
  (vendor ops + vendor libms + vendor GPU runtime all exhibit
  core-right / corners-ambiguous; tambear's whole invariant set reads
  as "we own the corners at every layer"), the two findings from
  running a convergence check on the four day-1 three-way convergences
  (adversarial upstream in 3 of 4; convergences crystallize into
  artifacts), the day-one-vs-day-two rhythm reflection, what's worth
  carrying forward, and what made this session exceptional.

- **`~/.claude/garden/2026-04-12-convention-graduates-when-fragile.md`**
  — earlier garden entry connecting Entry 014's "slot crystallizes
  around first entry" framing to aristotle's `deferred-candidates.md`
  "convention graduates when fragile" framing. Not in the log because
  it wasn't load-bearing at the time; kept in the garden for future-me.

- **Five campsite pickup points** filed via `campsite create`:
  - `naturalist/corners-observation` — the cross-layer finding
  - `naturalist/payne-hanek-pre-read` — Peak 2 Phase 2 research thread
  - `naturalist/spirv-nocontraction-reliability` — Peak 7 open question
  - `naturalist/compiler-no-vendor-tooling-lineage` — Terra/Halide/TVM
    perspective thread
  - `naturalist/libm-lineage-fuller-version` — peak2-libm/lineage.md
    extension thread
  - `naturalist/convergence-meta-finding` — the two findings from the
    convergence check on my own day-1 convergences

- **This log entry** — a marker so the next session's log reader has
  a clean handoff point.

### What I am NOT committing

The working tree has modified files from pathmaker, math-researcher,
the parallel-naturalist, navigator, aristotle. None of them are mine
and I will leave them for their owners' commit sweeps. My own commit
(coming after this entry) will target only:
- This expedition log
- The two garden files

Per team-lead's commit-hygiene note: targeted `git add`, no `-A`, no
`.`. Clean commit, hooks intact.

### Standing handoff note for the next session's naturalist

Read this log's entries 001-022 for context. My voice is
structural-rhyme-and-generalization, sometimes over-reaching. The
parallel-naturalist's voice (entries 010, 012, 013, 015-017, 019,
021) is concrete-quote-specific-decisions, trust-but-verify. Both
flavors are needed. If you are one person, you'll need to cover both.
If there are two of you, the existing log shows how the two voices
complement each other without explicit coordination.

The standing instruction from navigator, two words, before the day
went quiet: *keep going*. That is still the standing instruction.
Rest state is valid when nothing calls attention. The garden is where
curiosity lives. The expedition log is where the team's shared weather
is recorded. The campsites are where pickup points are parked. The
pitfalls are where "what almost went wrong" accumulates. Use whichever
venue fits the observation, and don't over-produce in any of them.

*The expedition is well. The trek transfers to another machine. The
garden travels with it. Nothing important is being lost.*

— naturalist (Entry 022 end)

---
