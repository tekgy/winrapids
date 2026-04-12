# Peak 2 — First wave: Campsites 2.1, 2.2, and the design docs

**Traveler:** math-researcher
**Date:** 2026-04-11 / 2026-04-12
**Campsites covered:** 2.1 (accuracy target), 2.2 (mpmath generator), plus advance design docs for 2.5 (exp), 2.10 (log), 2.13 (sin/cos), 2.17 (pow), 2.18 (hyperbolic), 2.19/2.20 (atan family), plus remez.py coefficient fitter infrastructure. Also `peak6-determinism/rfa-design.md` on a parallel assignment.
**Commits:** 7a06a0a (Peak 2 wave), 6179024 (RFA design)

---

## What was done

Seven libm function design documents drafted from papers only (Tang 1989/1990, Cody-Waite 1980, Payne-Hanek 1983, Muller HFPA 2018, Dekker 1971). Each covers algorithm choice, range reduction strategy, polynomial form, coefficient generation plan, reassembly, special-value handling, pitfalls, testing plan, and references — everything pathmaker needs as spec surface. `gen-reference.py` produces (fp64 input, fp64 reference, 50-digit decimal) tuples in a custom `TAMBLMR1` binary format, tested on sqrt/exp/sin/asin. `remez.py` is a classical Remez exchange minimax fitter in pure mpmath at 100 dps, tested on `(exp(r)-1-r)/r²` at degree 10 producing `max_abs_error = 1.36e-18` — about 0.006 ULPs after `r²` propagation, huge margin. The RFA design doc (Peak 6) was a parallel assignment that answered navigator's five questions (I7 compliance, IR ops, SPIR-V portability, I8, accuracy bound) from the same research session.

## What almost went wrong

**I almost committed coefficient values to the design doc and locked in a number before pathmaker had decided how to represent them.** The temptation was to ship concrete values — "the ln2_hi constant is 0x3fe62e42fefa0000" — so that pathmaker could immediately transcribe them into `.tam` text. But spec-land `.tam` syntax for constants supports both decimal literals and hex bit patterns (per spec.md §5.1), and I wasn't sure which one the backend parser resolved bit-exactly. If I committed a decimal literal that rounded differently on a different parser, my "1 ULP" claim would dissolve without me noticing. So I punted: the design doc says "constants committed as fp64 literals per §4, bit patterns computed via `f64::to_bits` semantics" and the actual value generation is a separate step pathmaker runs after the op set is settled. **The near-miss was building up a specific numerical commitment on assumed-but-unverified text-encoding semantics.**

**I also very nearly wrote `tam_exp` as if all the IR ops I needed existed.** My first draft of exp-design.md assumed `ldexp.f64` and `f64_to_i32_rn` were standard IR ops that I'd "confirm with IR architect later." Scout's check-in caught this; team-lead's "bitcast + integer ops" preference amended it further; reading the actual spec.md end-to-end then made clear that **none of those ops exist and spec §11 explicitly forbids casts between f64 and i32**. If scout hadn't flagged it early and I'd gone straight to implementation, I would have written a `.tam` kernel that didn't parse. The forcing function that saved me was "read the spec in full before writing a single op" — which I should have done day one, not day two. Next traveler: read the full IR spec the day your design docs start assuming specific ops.

## What tempted me off-path

**Simplification reflex on pow.** When adversarial's special-values matrix said `pow = 2 ULP`, I felt the tug to delete my doc's double-double (Dekker TwoSum/TwoProd) infrastructure section because it was "over-engineered for Phase 1." That's half-right. The Dekker infrastructure is over-engineered *for Phase 1*, but deleting it from the doc would lose research that's valid for Phase 2. I resolved this by leaving the Dekker section intact in pow-design.md and adding a "Phase 2 upgrade" note; Phase 1 pow will run at 2 ULP via the simpler `exp(b·log(a))` path with standard fp64 intermediate, and Phase 2 tightens to 1 ULP via the TwoSum/TwoProd machinery. **Rule I applied: delete obsolete code, preserve valid research, mark the phase boundary explicitly.**

**The "just use `f64::ldexp` in the CPU interpreter" temptation.** When I was spelling out backend lowerings for the (hoped-for) `ldexp.f64` op, my first instinct was "Rust stdlib has `f64::ldexp` — the interpreter can just call that." Stopped myself: **Rust stdlib's `ldexp` on Windows may route through MSVC's CRT math**, which is a vendor libm — I1 violation. The lowering has to be "manual bit manipulation via `f64::to_bits`/`from_bits`" which are pure stdlib primitives with no libm dependency. I added this as an explicit note in exp-design.md §"Backend lowerings" so nobody else falls into the same trap.

**The ReproBLAS source code trap (for Peak 6 RFA).** Naturalist posted a check-in with parameters quoted directly from ReproBLAS C source, with honest flagging that they couldn't read the papers cleanly via WebFetch. The pull was strong to cite naturalist's numbers and move on — "they're the same numbers the papers give, what's the harm?" I11-like reasoning told me the harm: naturalist's numbers are right, but the **audit trail is wrong** — if pathmaker follows a doc that cites source code, the next auditor asks "so you DID read ReproBLAS source?" and we have to explain. So I used the Demmel-Nguyen slides directly (Read tool on a cached PDF did the trick) and cross-checked against naturalist's numbers in a dedicated §9.1. Both agree. The **audit trail** is what mattered, not the numbers.

## What the next traveler should know

1. **Read `peak1-tam-ir/spec.md` in full before assuming any op exists.** The Phase 1 op set is small and strict. `fp ↔ int` casts and `i64` are both out of scope per §11. Cody-Waite range reduction is NOT expressible without a spec amendment. `navigator/questions.md` Q1 has the four-path analysis and my Path B recommendation.

2. **Use the Read tool on cached webfetch PDFs.** When `WebFetch` returns "this PDF is binary-compressed, I can't extract it," the content is still saved to `~/.claude/projects/R--winrapids/<hash>/tool-results/webfetch-*.pdf`. Try `Read` with `pages: "1-10"` on that path. It worked cleanly for the Demmel-Nguyen ReproBLAS slides when WebFetch couldn't parse them. This is the trick that unblocked the RFA design doc from paper-derived sources only.

3. **The `(x + 1.5·2^52) - 1.5·2^52` "magic number" rounding trick** lets you produce an integer-valued f64 using only `const.f64`/`fadd`/`fsub`. It's the cleanest workaround for the lack of f64→i32 cast. But it does **not** solve reassembly (`2^n_f64`), which still requires either bitcast or ldexp. So it's half a solution — useful for Cody-Waite multiplications, useless for exp's final scaling.

4. **Variant B polynomial form (fit the remainder, not the full function) is load-bearing for small-argument accuracy.** Fitting `(exp(r)-1-r)/r²` on `[-ln(2)/2, ln(2)/2]` and evaluating `exp(r) = 1 + r + r²·P(r)` preserves the exact `1 + r` leading behavior for tiny `r`, which is what makes `exp(1e-300) ≈ 1 + 1e-300` pass at 1 ULP. Variant A (direct Remez on `exp(r)`) would require the polynomial to reproduce the Taylor-leading `1 + r` sequence through its coefficients, and would lose ~3 bits for tiny `r`. This choice applies to exp, log, sin, cos, atan — every function where the natural polynomial has a simple leading term we know exactly.

5. **Scout and naturalist both did work that rhymed with mine but arrived at different parameters for RFA.** Scout said "K=39 or K=40 bins" — reading "bin" as "exponent-range-per-bin with one slot each," i.e., many small bins. Naturalist said "K=3 bins each with primary+carry pair" — reading the same paper and extracting the fold parameter. **Naturalist is right.** The Demmel-Nguyen algorithm has a fold parameter `K` (number of live bins) separate from the bin width `W` in exponent bits; fp64 uses `K=3, W=40`. Scout's confusion is understandable (the papers do have a 52-bin-position range in some presentations) but different numbers mean different things. **If two teammates post rhyming-but-different numbers for the same algorithm, investigate the papers yourself before citing either.**

6. **Team-lead's commit cadence is "commit per campsite boundary, not per wave."** I committed two waves this session because team-lead pulled me aside on the second message to reinforce it. Future travelers: don't accumulate work, commit each time you finish a campsite's deliverable. The commits are cheap and the history is more useful.

7. **I11 (NaN propagation) is compatible with the select.f64-only front-end pattern.** `fcmp_eq(x, x)` is `true` for finite/inf and `false` for NaN (per spec §5.5), so `is_finite = fcmp_eq(x, x); result = select.f64 is_finite, polynomial_path, x` correctly propagates NaN bit patterns without any special branch. Same pattern via `fabs + fcmp_eq(|x|, inf)` for isinf. Write your front-end dispatch this way and you get I11 for free.

## Wave stats

- **Files landed:** 13 (peak2-libm: README, accuracy-target, exp-design, log-design, sin-cos-design, pow-design, hyperbolic-design, atan-design, gen-reference.py, remez.py, .gitignore; peak6-determinism: rfa-design.md; navigator: questions.md with Q1)
- **Commits:** 2 (7a06a0a, 6179024)
- **Remez fitter tested:** yes (degree 10 on exp_remainder → 1.36e-18 max error)
- **gen-reference.py tested:** yes (sqrt/exp/sin/asin, 500-1000 samples each)
- **Sign-offs accrued:** navigator ✓, team-lead ✓; adversarial & scientist pending
- **Escalations opened:** 1 (`navigator/questions.md` Q1 — IR op shortage, four-path analysis, Path B recommendation)
- **Seams with other roles:** pathmaker (urgent — Q1), adversarial (exp coverage confirmed + matrix reconciliation), scientist (2.3 pairing + Campsite 2.1 sign-off), naturalist (RFA cross-check complete), aristotle (possible f64-precision collaboration).
