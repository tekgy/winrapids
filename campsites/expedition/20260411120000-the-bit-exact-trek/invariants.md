# Invariants (I1–I10) — the fences around the trek

**Print this. Read it at the start of every work session. Every campsite obeys these. If a campsite begins to require stepping over one, halt and escalate — never paper over.**

| # | Invariant | What it forbids | Why |
|---|---|---|---|
| **I1** | No vendor math library in any path | `__nv_sin`, `__nv_log`, glibc `log`, `metal::exp`, cuMath, cuBLAS, cuDNN, cuFFT, any third-party `*m` | The whole point |
| **I2** | No vendor source compiler in any path | NVRTC, nvcc, clang, llvm (via FFI), SPIRV-Cross, DXC, Metal Shading Compiler | Removes the control we need |
| **I3** | No FMA contraction, ever, unless explicit | PTX `.contract true` (default), `a*b+c` silently becoming `fma` | Different hardware contracts differently → bit drift |
| **I4** | No implicit reordering of floating-point operations | Compiler-freedom flags like `-ffast-math`, `.ftz`, `-Ofast`, `/fp:fast` | Associativity is false for fp; reordering means different answers |
| **I5** | No non-deterministic reductions | `atomicAdd` for user-visible final values, parallel reduce with variable block count | Same input → same output on every run, every hardware |
| **I6** | No silent fallback when a target is missing a feature | "if fp64 unavailable, use fp32" | Silent precision loss — user didn't ask for it |
| **I7** | Every primitive still decomposes into accumulate + gather | New ops added to `.tam` IR that don't fit this pattern | The library's compositional property depends on this |
| **I8** | First-principles only for transcendentals | Porting glibc / musl / sun-libm code | Our implementations are our responsibility; borrowed code carries borrowed assumptions |
| **I9** | mpmath (or equivalent arbitrary-precision reference) is the oracle | Comparing against another libm to "validate" | Two libms can both be wrong; arbitrary-precision is the ground truth |
| **I10** | Cross-backend diff is continuous, not a final audit | "we'll validate Vulkan later" | Drift compounds; catching it at step 3 is cheap, catching it at step 7 is expensive |

## Escalation protocol

If a campsite hits an invariant wall:

1. **Halt the campsite immediately.** Do not relax the invariant to get unblocked.
2. **Write the escalation** in `navigator/escalations.md` with:
   - Which invariant is in tension
   - What the campsite is trying to do
   - What the perceived conflict is
   - What the team member has tried
3. **Surface to Navigator** (the primary Claude session). Navigator adjudicates within one session; if they can't, fresh eyes are brought in.
4. **Never paper over an invariant violation by quiet mutual agreement.**

## The soft principles (binding, but less absolute)

- **Incrementality.** Every campsite is small enough to finish in one focused session, test in isolation, and commit. If it feels bigger, split it.
- **Incremental oracle.** Every new op, every new function, every new backend ends with a test against a previously-trusted reference.
- **Refuse convenience.** When you want to cut a corner, the convenience is almost always the direction of the pitfalls you're avoiding. Assume the easy path is the wrong path and have to prove otherwise.
- **Text-first, binary-later.** Every format we design has a human-readable text encoding first.
- **Shapes over contents.** The IR Architect decides what *kinds* of things exist before anyone decides their contents.
- **Document the weather, not the work.** Logbook entries record what *almost* went wrong, what tempted the implementer off-path. The near-misses are what help the next person.
- **Adversarial pressure from day one.** The Adversarial Mathematician role isn't a final check — it's a continuous pressure.
