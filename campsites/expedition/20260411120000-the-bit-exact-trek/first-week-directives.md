# First Week Directives

**Who starts where. Five agents, five independent threads, day one — no serial blocking.**

Read `invariants.md` and `trek-plan.md` before you begin. The full campsite list lives in `campsites.md`. This file tells you *which campsites to pick first* so the team doesn't collide or sit idle.

---

## IR Architect — start immediately

**Goal of your first week:** Land Peak 1 campsites 1.1 → 1.9, plus start 1.10.

Your first day:
1. **Campsite 1.1** — Write the `.tam` IR spec as a 2-page markdown doc under `peak1-tam-ir/spec.md`. Follow the scope from Part IV/Peak 1 of the trek plan. **Do not expand beyond what's listed there.** Types, ops, entry format, text encoding. No code yet.
2. Circulate the spec in `peak1-tam-ir/spec-review.md` for lightweight sign-off from Navigator and Test Oracle before you start writing code.

Your first week after spec approval:
3. **1.2** — Rust types for the AST. New crate `tambear-tam-ir` in `crates/`, or a module in `tambear-primitives`? **Recommendation:** new crate, because it has zero deps and many downstream consumers.
4. **1.3, 1.4, 1.5** — Hand-write the three reference programs. These become your parser/printer test fixtures.
5. **1.6, 1.7** — Text printer, then text parser. These are independent and both small.
6. **1.8** — Round-trip property test with 10,000 random programs.
7. **1.9** — Type-check pass.

**Pair check-in:** When you finish 1.3 (hand-written `sum_all_add`), share with Libm Implementer so they know what a `.tam` function *looks* like for when they start writing `tam_exp`.

**What NOT to do:**
- Don't add ops beyond Phase 1 scope. If you're tempted, write it in `peak1-tam-ir/future-ops.md` and move on.
- Don't design a binary encoding. Text only for Phase 1.
- Don't build a verifier more powerful than "types match and every register has a definer."

---

## Libm Implementer + Math Researcher — start immediately

**Goal of your first week:** Infrastructure for libm. No libm code yet — but the *tests* for libm code must be ready to receive it.

Your first day:
1. **Campsite 2.1** — Pick the accuracy target. Write it up in `peak2-libm/accuracy-target.md`. Recommendation is ≤1 ULP faithful rounding; argue for or against before committing.
2. **Campsite 2.2** — Build the mpmath reference generator. Python, with `uv venv` + `uv pip install mpmath`. Lives in `peak2-libm/gen-reference.py`. Produces `.bin` or `.csv` files of (input, reference) pairs at 50-digit precision. Test with `--function sqrt --n 1000`.

Your first week:
3. **2.3** — Rust ULP-measurement harness. Reads the reference files from 2.2, computes max/mean/stddev ULP. This harness will be used by every libm function test.
4. **Algorithmic research** — for `tam_exp`, read the literature on Cody-Waite reduction, Remez coefficient generation, Tang's tables. Write `peak2-libm/exp-algorithm.md` explaining the approach you'll take. **Review with Math Researcher hat on:** did you read at least 3 sources? Did you cite them? Did you pick parameters deliberately or by inertia?
5. Start drafting the `tam_exp` algorithm design doc (**2.5**) — this will need IR Architect to confirm the op set is sufficient before you commit to an implementation.

**Do not** start writing `.tam` code until IR Architect has finished 1.15 (the reference doc) or at least 1.6/1.7 (parser/printer so you can validate what you write).

**What NOT to do:**
- Do NOT look at glibc/musl/sun-libm source code during design. Read the *papers* for algorithms, not other people's code for polynomials. The coefficients are yours.
- Do NOT optimize for speed during Phase 1. Correctness first, speed later. The interpreter will run it.
- Do NOT pick "close enough" rounding modes or "usually correct" coefficients. The ULP bound is not advisory.

---

## PTX Assembler — start immediately

**Goal of your first week:** Peak 3 campsites 3.1, 3.2, 3.3, 3.4 — get to the point where hand-written PTX loads through the raw driver path without NVRTC.

Your first day:
1. **Campsite 3.1** — Read the PTX ISA spec. Write `peak3-ptx/ptx-subset.md` naming the exact subset we emit. Critical: document `.contract` default behavior and how to disable it. Document `.rn` rounding mode on every fp op.

Your first week:
2. **3.2** — `PtxBuilder` in `crates/tambear-tam-ir` (or a new sibling crate `tambear-ptx`). Structured API for emitting PTX text. Accept when you can emit a kernel with `.version`, `.target`, `.entry`, and one `mov.f64` + `st.global.f64`.
3. **3.3** — Hand-written hello-world PTX (writes 42.0). Verify as a string using `ptxas --verify` if available locally (dev dep only, **never a runtime dep — I2**).
4. **3.4** — This is important: **investigate `cudarc::driver::CudaContext::load_module`** carefully. Does it actually bypass NVRTC, or is there a hidden NVRTC call? If there's any doubt, read the `cudarc` source. Goal: be able to say "yes, passing PTX text here means the driver's `cuModuleLoadDataEx` receives it, and no host-side compiler runs." If that's not true for the current `cudarc` API, **escalate to Navigator** before proceeding — we may need to drop `cudarc` and go lower.
5. **3.5** — Dispatch the hello-world. Confirm `out[0] == 42.0`.

**Do not** start the translator (3.6+) until Peak 1 has landed at least 1.9 (verified AST). The translator consumes an AST; without an AST it's vaporware.

**What NOT to do:**
- Don't use `compile_ptx_with_opts` for ANY reason. That's NVRTC. I2.
- Don't emit `add.f64` without `.rn`. Don't emit `mul.f64` without `.rn`. Don't let any fp op be ambiguous about rounding.
- Don't assume `.contract` is off by default. It's not. It's on. You must explicitly turn it off.

---

## Test Oracle + Adversarial Mathematician — start immediately

**Goal of your first week:** Infrastructure that every other peak will plug into, plus the pitfall journal.

Your first day:
1. **Campsite 4.1** — `trait TamBackend` skeleton. Just the trait and a dummy `NullBackend` that returns zeros. Lives in `crates/tambear-tam-test-harness` or similar. Accept when it compiles and has one smoke test.
2. **Start the pitfall journal** at `pitfalls/README.md` with the 15 pitfalls from `trek-plan.md` Part VII. Add entries as you discover new ones.

Your first week:
3. **4.7 (start early)** — Build the "hard cases" suite *now*, before any backend exists. Input generators for catastrophic cancellation, subnormal, nan/inf, huge N, empty, single-element. These tests will fail immediately (no backend to run them on), but having them in place means peaks 1/3/5/7 each have to satisfy them on arrival.
4. **Collaborate with Libm Implementer** on 2.2, 2.3 — the mpmath reference generator and ULP harness are shared infrastructure. Don't duplicate work; negotiate who owns what.
5. **Adversarial pass on current recipes:** Run `tambear_primitives::recipes::variance` on `data = [1e9 + x for x in uniform_unit]`. Watch the one-pass formula collapse. File a campsite for the recipe team ("variance recipe catastrophic cancellation — adopt Welford or document") and log it in `pitfalls/variance-one-pass.md`.

**What NOT to do:**
- Don't wait for other peaks to land. Your work is pure infrastructure and pure adversarial pressure; both are day-one work.
- Don't raise tolerances without root-causing. The tolerance policy is strict.
- Don't stop finding pitfalls. Even if everything "works," probe for weirdness. That's the job.

---

## CPU Backend Implementer (often double-hatted with IR Architect)

**Goal of your first week:** Follow IR Architect closely. Start writing the interpreter as soon as ops exist.

Your first week:
1. Wait for Peak 1 campsites 1.1–1.3 (spec, AST types, one hand-written program).
2. **Campsite 1.10** — Start the interpreter. Minimal set first: `const`, `fadd`, `fmul`, `load`, `store`, `loop_grid_stride` (serial, one-threaded execution). Accept when `sum_all_add` returns 55.0 on `[1..=10]`.
3. **1.11** — Extend with remaining ops.
4. **1.12** — Match the existing `recipes::variance` CPU executor. Exact equality.
5. **5.1** — Formalize as `CpuInterpreterBackend` implementing `TamBackend`. This plugs you into the test harness.

**What NOT to do:**
- Don't call `f64::exp` in the interpreter. Don't call `f64::sin`. Don't call `f64::ln`. You will only have `fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt` as fp operations. Anything else is a bug waiting to happen.
- Don't write a JIT in Phase 1. Peak 5.5 is later. Interpreter first.

---

## Shared practices for everyone, all week

### Logbook discipline

At the end of every campsite, append an entry to `logbook/<peak>-<campsite>.md` with:
- **What was done** (1–2 sentences, not a diff — the code speaks for itself)
- **What almost went wrong** (1 paragraph, the near-miss that taught you the most)
- **What tempted me off-path** (1 paragraph, the convenience you refused)
- **What the next traveler should know** (1 paragraph, warnings or landmarks)

### Commit discipline

- Commit at the end of every campsite. The commit message references the campsite number (e.g., `Peak 1.6: text printer for .tam IR`).
- **Don't skip hooks** — run local tests before committing.
- **Don't amend** — create new commits.
- If a campsite spans a session break, commit the partial work with "(WIP)" and pick it up next time.

### Invariant discipline

- Every morning, re-read `invariants.md`. It's short. Reading it refreshes the shape.
- If you touch code that could implicate an invariant (FMA contraction, rounding mode, atomic reductions, libm), add a comment in the code citing the invariant number.

### Escalation discipline

- If you hit a wall that requires relaxing an invariant, STOP. Write up the situation in `navigator/escalations.md` with:
  - Which invariant is in tension
  - What the campsite is trying to accomplish
  - What you tried
  - What you think the options are
- Navigator will adjudicate. Do NOT decide among yourselves to relax.

### Cross-role check-ins

- **Every ~3 campsites**, pause and write one sentence in `navigator/check-ins.md` about what's blocking, what just landed, what you need from someone else.
- Don't turn this into a daily standup. It's a "seam" — a place where roles touch. Use it when you actually need something.

---

## When in doubt

Read `trek-plan.md`. The plan has more detail than this directive file can cover. If the plan and this directive file disagree, **the plan wins** — this file is a schedule, not an amendment.

If the plan is silent on your question, **ask in `navigator/questions.md`** and keep working on something unblocked while you wait for Navigator to answer.

If the plan is silent *and* you can't find unblocked work, **follow your curiosity within the spirit of your role.** This team runs on jbd-team principles — idle time is an invitation to explore, not a stall.
