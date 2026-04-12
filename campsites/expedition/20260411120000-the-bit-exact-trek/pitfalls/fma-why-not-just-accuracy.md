# Pitfall: Thinking I3 is about accuracy (it isn't, it's about consistency)

**Discovered by**: naturalist, 2026-04-11. Complement to P01 (the mechanical defense) and P02 (rounding modes). This entry is about the *philosophy* behind the rule, because whoever implements Peak 3 WILL be tempted to relax it, and the defense is only durable if they understand *why*.

---

## The temptation

You're writing the tam→PTX assembler. You translate `fmul.f64 %c = %a, %b` then `fadd.f64 %d = %c, %x` and you realize: *on NVIDIA hardware, these can be fused into `fma.rn.f64 %d, %a, %b, %x` — one instruction instead of two, strictly more accurate (one rounding step instead of two), almost certainly faster*. Every FP textbook from Kahan onward will tell you FMA is the better instruction. The PTX default is to fuse. The Moments in Graphics post ("fma: a faster, more accurate instruction") makes an eloquent argument.

So you think: maybe I3 should have an exception for tam_exp's polynomial evaluation, where FMA would shave an extra ULP off our error bound.

**No.** The reason is not what you think.

## The Qt Quick anecdote

KDAB's "FMA Woes" post walks through a bug that lived in Qt Quick for a decade:

```cpp
const double scale = 1.0 / i;
const double r     = 1.0 - i * scale;
```

Mathematically, `r` is exactly zero. Without FMA contraction, it IS exactly zero on every IEEE 754 fp64 platform — because `1.0/5 → 0.2` (rounded), then `5 * 0.2 → 1.0` (rounded), then `1.0 - 1.0 → 0.0` exactly. The round-off in the first step is compensated by the round-off in the second step.

*With* FMA contraction, the compiler emits `fma(i, scale, -1.0)`, which uses the *unrounded* product `i * scale`. For `i = 5`, that unrounded product is *not quite* 1.0 (because 0.2 isn't exactly representable in binary fp), so the fma returns approximately `-5.55e-17`. Negative.

Downstream, Qt's code fed that `r` into `sqrt(a*a - b*b)` where `a == b`, and a similar FMA-contraction ate the cancellation that should have produced exactly zero, so `sqrt` received a tiny negative number and returned NaN, and an entire rendering layer became invisible.

**The bug shipped for a decade.** It was invisible on OpenGL (which has a signed depth range that accepted the negative value) and appeared only when Qt moved to Metal+ARM8+Clang14 (which has an unsigned depth range). *Same source code, different backend, catastrophically different behavior.*

**This is exactly the failure mode tambear exists to prevent.** If we let FMA contract, we inherit Qt's failure mode at a lower layer. The expedition has two cross-backend invariants (I3 and I5) protecting against the exact same class of bug at two scales:

- **I5 (deterministic reductions)** — the reduction order is backend-independent.
- **I3 (no FMA contraction)** — the per-instruction rounding count is backend-independent.

Both say "the answer is backend-independent by construction." Relax either and the expedition's summit claim collapses at that layer.

## The paradox to internalize

Under FMA, Qt's `1.0 - i * (1.0/i)` is *strictly more accurate* than under non-FMA — closer to the true mathematical value of zero. That's not in dispute. The FMA version has fewer rounding steps.

And it breaks a program that was relying on the *compensating* roundings of non-FMA. The program was correct for its platform and the platform's behavior changed under it.

**The accuracy gain broke the program.**

This is the thing to internalize: *accuracy and consistency are different goals, and they can conflict.* Tambear is optimizing for **consistency**, not accuracy. When they conflict, consistency wins. Every time. No exceptions.

- Accuracy optimization: every operation produces the fp value closest to the true mathematical answer.
- Consistency optimization: every run of the same program produces the same fp value, regardless of hardware/backend/parallelism.

I3 exists because consistency dominates accuracy in tambear's value function. The CPU interpreter doesn't contract. The PTX backend COULD contract (NVIDIA's default is on). The Vulkan backend could contract per-op (SPIR-V `NoContraction` is an opt-out decoration). Allowing any of them to contract gives us three backends producing three different answers from the same .tam source. We built the whole RFA + IR + replay-harness apparatus to prevent exactly that, and FMA contraction would re-introduce it one layer down.

## The cost is real

Here's the part I want to be honest about: the CPU interpreter is the slowest backend AND the one that never contracts. We're treating it as the oracle of truth. That means the GPU backends have to match the slower backend's numerical behavior, which forces them to be *less accurate than they could be* on every polynomial-heavy operation.

Some of tambear-libm's function will have max-ULP bounds measurably worse than a hypothetical "PTX-with-FMA" version could achieve. This is not free. The pathmaker and the libm implementer will both notice it. They will be tempted to say "the PTX backend could be 0.5 ULP more accurate here if we just let this one add contract."

**The answer has to be no.** An accuracy gain that is backend-dependent is worse than an accuracy loss that is uniform. In this project's value function. Every time.

## The two tests every backend must pass

Both from the FMA Woes anecdote, both isolate FMA-suppression from every other concern (no libm, no reductions, no higher-level primitives):

```rust
// Test 1: classic "should be exactly zero"
let i: f64 = 5.0;
let scale = 1.0 / i;
let r = 1.0 - i * scale;
assert_eq!(r.to_bits(), 0.0f64.to_bits());  // passes without FMA; fails with FMA (r ≈ -5.55e-17)

// Test 2: difference of squares
let a: f64 = 1.23456789;
let b: f64 = 1.23456789;
let d = a * a - b * b;
assert_eq!(d.to_bits(), 0.0f64.to_bits());  // passes without FMA; fails with FMA
```

Three lines each. No dependencies on libm or reductions. Suitable for Peak 4's hard-cases suite from day one. Every backend that tambear adds must pass both, and if any backend produces anything other than `0.0f64.to_bits()`, it is contracting somewhere and I3 is violated.

The beauty of these tests: they don't require any higher-level infrastructure. A one-kernel .tam program with three ops (`load, fmul, fsub`) or (`fmul, fmul, fsub`) can run them. They belong in the Peak 4 infrastructure being built right now, and they will run before any backend can claim "I1–I5 compliant."

## The structural rhyme

I3 and I5 are the same invariant at different scales:

| Temptation | "More accurate locally" version | "Less accurate locally but consistent globally" version |
|---|---|---|
| FMA vs separate mul+add | FMA (one rounding) | Mul+add (two roundings, but every backend agrees) |
| `atomicAdd` vs RFA | `atomicAdd` (any tree) | RFA bins (fixed order, every hardware agrees) |

Both choose consistency over local accuracy. Both exist because cross-backend determinism requires it. Both will feel wrong to a numerical-analysis instinct ("but FMA is *better*!", "but atomicAdd is *faster*!"). Both are correct in tambear's value function.

If you ever find yourself wanting to relax I3 or I5 "just for one op," go back to the Qt story and ask: would you have wanted the Qt team to relax their equivalent rule "just for this one z-value calculation"? They did. It cost them a decade.

---

**See also**:
- `P01` (in README.md): the mechanical defense — emit `.rn`, forbid bare `add.f64`, reject `compile_ptx_with_opts`.
- `P02`: the rounding-mode companion to P01.
- The expedition log, Entry 003 and the related garden entry `2026-04-11-fma-consistency-not-accuracy.md`, for the longer-form argument.
