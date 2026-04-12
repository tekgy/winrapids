# Pitfall: ULP bounds compose additively (and nobody remembers)

**Discovered by**: naturalist, 2026-04-11. Unmarked in the original trek plan; not covered by P01–P18. Affects Peak 2 (tambear-libm accuracy target), Peak 4 (tolerance spec), and the architectural claim itself.

---

## The temptation

You (math-researcher, working on Peak 2) have picked "≤ 1 ULP faithful rounding" as tambear-libm's accuracy target for `tam_exp`. You've tested: for 1M random fp64 inputs in `exp`'s domain, the max observed error vs mpmath is 0.87 ULP. You write "`tam_exp` max ULP: 0.87" in `peak2-libm/accuracy-target.md` and move on.

You (scientist, working on Peak 4's `ToleranceSpec`) see the 1 ULP target and write:

```rust
ToleranceSpec::transcendental_1ulp()  // tolerate up to 1 ULP for ops using tam_exp
```

You add a test: a kernel that computes `exp(log(x))` on a random input and compares CPU interpreter vs PTX vs Vulkan. The test fails. All three backends agree with each other (I3/I5 are holding), but all three disagree with mpmath by ~2.6 ULPs. You scratch your head. The libm team says their functions are 1 ULP. The cross-backend diff harness says the backends agree. But the absolute answer is 2.6 ULPs off the oracle.

**Nobody is wrong. The pitfall is that nobody warned you ULPs *compose additively*.**

## The theorem

If a function `f` has a worst-case error of `ε_f` ULPs over its domain, and a function `g` has a worst-case error of `ε_g` ULPs, then the worst-case error of `g(f(x))` is (roughly) **`ε_g + ε_f * |f'(x)| * (2^e_f / 2^e_g)`** ULPs, where the exponent-scaling factor is small for most well-behaved compositions but NOT always small.

Informally: **bounded errors on individual functions add up when you compose them**, with an extra multiplicative factor if the second function is sensitive to its input (large derivative).

For `log(exp(x))`:
- `exp` has error `ε_exp` ≤ 1 ULP.
- `log` has error `ε_log` ≤ 1 ULP.
- `log` applied near `y ≈ 1` has `|log'(y)| = 1`, so the error from exp propagates almost 1:1.
- Total bound: approximately **2–3 ULPs**, not 1.

For `sin(x) + cos(x)` (simple addition of transcendentals):
- Each transcendental: ≤ 1 ULP.
- The `fadd` introduces one more rounding: ≤ 0.5 ULP.
- The relative cancellation factor when `sin(x) ≈ -cos(x)`: can blow up.
- Total bound: ≤ 2.5 ULPs in well-behaved regions, **unbounded near cancellation**.

For a Horner polynomial `P(x) = c_N*x^N + ... + c_1*x + c_0` with degree N, each `fmul + fadd` step contributes ≤ 1 ULP (under non-FMA), and the errors compose additively through the evaluation. A degree-10 polynomial has a ULP bound of roughly **N+1 ≈ 11 ULPs**, not 1.

## Why this is a pitfall, not just a fact

Because it bites twice, from opposite sides:

**Bite 1 (Peak 2, math-researcher)**: The per-function ULP bound is not the number users care about. Users care about the bound on their *full computation*, which is the composition of many primitives. If `tam_exp` is documented as 1 ULP and the user sees 3 ULP error in `exp(log(x))`, they will blame the libm. The libm is correct. The user's expectation is wrong. The docs must state the composition rule explicitly, or this confusion will be reported as a libm bug.

**Bite 2 (Peak 4, scientist)**: The `ToleranceSpec` for cross-backend diff must distinguish between:
- *Backend disagreement tolerance* — backends should agree bit-exactly on pure arithmetic, and within the documented per-function ULP bound on transcendentals. This is what P01 and I3 enforce.
- *Oracle disagreement tolerance* — backends collectively agree with mpmath oracle within some bound. This bound is NOT the per-function ULP bound; it's the *compositional* bound for whatever expression the kernel computes.

Conflating these two is the natural mistake, because both use "ULP" as their unit. The first is small and fixed (or zero for arithmetic). The second grows with the kernel depth.

## The trap pattern

This is how the pitfall manifests in practice:

1. Peak 2 measures `tam_exp` at 0.87 ULP max error. Documents it as "1 ULP bound."
2. Peak 2 measures `tam_log` at 0.93 ULP max error. Documents it as "1 ULP bound."
3. Peak 4 writes `ToleranceSpec::transcendental(1_ulp)` for any kernel that calls either.
4. A test kernel computes `y = log(exp(x))`, which should equal `x`.
5. The test runs. CPU, PTX, Vulkan all agree with each other (I3/I5 holding).
6. Their shared answer differs from mpmath's answer by 2.2 ULPs.
7. The tolerance is 1 ULP. Test fails.
8. Debugging spiral: libm team measures their functions, all under 1 ULP. Backend team measures cross-backend, all bit-exact. Everything is "correct." The test still fails.
9. Eventually someone realizes the tolerance should have been 2-3 ULPs for this kernel.
10. "Fix": raise the tolerance. But by how much? Random guessing. Adopting the new tolerance without a principled reason violates the "refuse convenience" rule from the trek plan.

## The right shape for the tolerance spec

`ToleranceSpec` should be computed from the kernel, not picked as a number:

```rust
enum ToleranceSpec {
    BitExact,                        // pure arithmetic, post-I3/I5
    WithinOracle { max_ulps: u32 },  // vs mpmath reference, computed from kernel
    WithinBackends { max_ulps: u32 }, // backends agree with each other, not oracle
}

fn compute_oracle_tolerance(kernel: &TamProgram) -> u32 {
    // Walk the kernel's call graph. For each call site, add the libm function's
    // documented per-function ULP bound. For each fp op, add 0.5 ULP.
    // Sum. Round up.
}
```

The key distinction: **`WithinBackends` is small (0–1 ULP for transcendentals, 0 for arithmetic) because it's testing whether the backends agree, which is a cross-backend-consistency property and I3/I5 force it to ~0**. **`WithinOracle` is computed from the kernel because it's testing whether the *composed* computation agrees with the true mathematical answer, which grows with kernel depth**.

Most tambear tests should assert:
- `cpu.to_bits() == cuda.to_bits() == vulkan.to_bits()` (bit-exact cross-backend).
- `|cpu - mpmath| ≤ compute_oracle_tolerance(kernel)` (compositional ULP bound).

Only the FIRST of these is the architectural claim. The second is a separate, weaker claim that lets users trust the library's absolute accuracy. They should not be conflated.

## Defense

1. **Peak 2**: Document both the per-function bound AND the "composition rule" in `peak2-libm/accuracy-target.md`. At minimum: "if a user computes `f(g(x))`, the bound on the full computation is at most `ε_f + ε_g * |f'| * scale`, where `scale` depends on the ranges involved. See `compositional-ulp.md` for the derivation."

2. **Peak 4**: `ToleranceSpec` must have both `WithinBackends` and `WithinOracle` variants, and the cross-backend diff harness must call each test with the right one. A helper function computes the oracle tolerance from the kernel's call graph.

3. **Peak 4 hard-cases suite**: include `y = log(exp(x))` and `y = exp(log(x))` specifically, because they are the most likely to expose this confusion. Both should backend-agree bit-exactly and both should diff against mpmath within the compositional bound.

4. **Test Oracle invariant**: *two distinct tolerance axes, never conflated*. If a test fails the cross-backend check, the fix is not "raise the tolerance" — it's "find the backend that's contracting an op it shouldn't." If a test fails the oracle check, the fix is not "raise the tolerance" — it's "the kernel's compositional bound is larger than you thought; update the expected bound."

## The general pattern this belongs to

This pitfall belongs to the family of **"local invariants with non-local consequences."** The per-function ULP bound is a local claim that seems to compose naturally. It doesn't. The same pattern shows up in:

- FMA contraction (P01, fma-why-not-just-accuracy.md): a local accuracy gain compromises global consistency.
- Reduction order (P04, P09, P10, RFA framing): a local parallelism optimization compromises global determinism.
- Polynomial evaluation order (P14 partial): a local Horner-vs-Estrin choice changes the composition of per-step errors.

Every one of these is "locally the obvious move" and "globally wrong." The project keeps encountering this family because the project's value function is global consistency, while the per-op reasoning that humans and compilers do is local.

**Rule of thumb**: whenever you find yourself reasoning about a single op or a single function "in isolation," stop and ask: *what's the compositional consequence?* The question almost always reveals a bound you hadn't been tracking.

---

**See also**:
- `fma-why-not-just-accuracy.md` — the FMA version of the same lesson at the instruction level
- `P14` in `README.md` — polynomial coefficient precision, the related concern inside a single libm function
- `P16`, `P18` in `README.md` — Welford/RFA versions of the same lesson at the reduction level
- The expedition log, Entries 002–004, for the consistency-vs-accuracy framing across all three scales
- `P19` in `peak4-oracle/pitfalls.md` — the concrete current diagnosis: 9 GPU tests only check WithinBackends, have no WithinOracle assertions
- `P21` in `peak4-oracle/pitfalls.md` — the architectural principle stated directly: agreement ≠ correctness, the two tolerance axes are orthogonal properties
