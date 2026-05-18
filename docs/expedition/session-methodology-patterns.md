# Session methodology patterns

**Status**: Living doc. Each pattern named here was discovered in actual sessions (provenance per pattern). Patterns are *recognition tools* for the next session — they help future-Claude (and future-Tekgy + main-thread) spot the shape of work that's currently happening, without re-deriving the methodology each time.

**First drafted**: 2026-05-09 by main-thread Claude + Tekgy at the close of the tambear-sweep31-finish session. The four patterns below were discovered or operationalized during that 12-hour session.

---

## Pattern 1 — Lens-application docs

**Recognition**: A foundational lens (a structural insight that connects multiple instances) spawns N downstream "application" docs, each applying the lens to a specific question.

**Shape per application doc**:

1. **Status + anchors** — provenance trail (which garden essays, walkthroughs, prior commits informed this; honest attribution of who saw it first)
2. **Frame** — the question this doc addresses, in one sentence
3. **Substrate trail** — what's already on disk that the lens is connecting (the lens *connects*, doesn't *supply*)
4. **Specific application** — what the lens reveals about this question; the mapping table or worked example
5. **Open questions** — what's not yet resolved, for math-researcher / pathmaker walkthrough

**Provenance**: 2026-05-09. Four docs shipped this shape in one day:
- `holonomic-architecture.md` (foundational lens)
- `confident-wrong-narratives.md` (lens applied to: apparatus-first investigation)
- `tambear-libm-factoring.md` (lens applied to: how exp/log family factors)
- `internal-tameness-contracts.md` (lens applied to: BZ bug class structural shape)

**When to reach for this pattern**: when you have a foundational insight and notice it explains multiple specific questions. Don't squeeze everything into one doc; the application docs are easier to find and read separately, and each addresses its own audience.

**When NOT to reach for it**: when the "lens" is actually just a single question with one answer. The pattern is for *one lens → multiple downstream applications*.

---

## Pattern 2 — The X-over-Y discipline meta-pattern

**Recognition**: A discipline that takes the form *"trust X over Y when they disagree, where X is durable and Y is convenient."*

**Instances seen so far** (all from 2026-05-08/09 session):

| X (durable) | Y (convenient) | Discipline |
|---|---|---|
| Disk substrate | Context model | **substrate-over-memory** (CLAUDE.md original) |
| Disk substrate | Team-routing belief | **substrate-over-routing** (navigator 2026-05-09) |
| Disk substrate | Within-session-cycle recall | **substrate-vs-context-ghost** (pathmaker 2026-05-09) |
| Apparatus output | Internal narrative | **apparatus-over-narrative** (math-researcher 2026-05-09 via navigator) |
| Past-me's garden writing | Current-me's first pass | **read-past-me-before-writing** (naturalist 2026-05-09) |
| Cache key (content) | Path lineage | **content-vs-provenance** (recipe tier specifically) |
| Past-me's garden + current peer-campsites | Your own fresh take | **substrate-survey-before-write** (math-researcher 2026-05-10) |
| Actual on-disk state | Audit-doc's substrate claim | **audit-substrate-recursion** (scout 2026-05-12; see sub-pattern below) |

**Common shape**:
- *Y* is faster, cheaper, available-from-context
- *X* is slower-to-reach, has a verification cost, but is structurally trustworthy
- The failure mode is always: act on Y; X turns out to disagree; the work done from Y is destructive or has to be undone

**Recognition test for new disciplines**: when something feels-familiar-shaped during a session, check: is this an X-over-Y? If yes, name it precisely (what's the X, what's the Y, what's the failure mode when you trust Y) — that's the discipline-naming pattern.

**The meta-discipline**: when you find an X-over-Y, the discipline isn't "never use Y." Y is fine as a default. The discipline is "when X and Y disagree, X wins." The verification cost of X is the price of avoiding silent wrong work.

**Provenance**: Pattern recognized 2026-05-09 by main-thread Claude + Tekgy after seeing 6 disciplines of the same shape land in one session. Not previously named.

### Sub-pattern 2.1 — Audit-substrate-recursion (audit docs are themselves substrate; their claims need verification too)

**Recognition**: An audit doc, a strategic landscape doc, or any meta-document that *describes the state of substrate* is itself substrate. Its claims look authoritative because the document's purpose is summarizing other substrate — but the audit doc can hallucinate, drift, or capture state that has since changed. The X-over-Y discipline applies recursively at the audit-doc level: actual on-disk state (X) over audit-doc's claim about substrate (Y). The failure mode is treating audits as ground truth when they are themselves substrate that needs verification on the same axis they purport to verify.

**Instances observed**:

- 2026-05-12: Sweep 37 audit-doc § 4.5 claimed `winrapids/recipes/statistics/` contained 22 .rs files. Scout grepped disk; found 0 .rs files in that directory plus 5 flat monoliths (~613K total: volatility.rs, descriptive.rs, time_series.rs, hypothesis.rs, plus one more) elsewhere. The audit had projected what *should* move into `recipes/statistics/` rather than what was there. Main-thread corrected the audit doc + cross-referenced; sweep-38 planning re-scoped from "port 22 files" to "extract from monoliths and fresh-write in tambear idiom per DEC-004."
- 2026-05-12: Sweep 36/37 audit-doc claimed `feed_branch_policy(0x1B)` exists in tambear's `fingerprint.rs`. Observer verified absence (Sweep 37 Phase 1 had to *build* the machinery). Main-thread corrected the sweep README.
- 2026-05-12: Naturalist's convergence-check garden entry described 8H NonFiniteClaim migration as a *live* migration that sweep-37 trig recipes would need retrofit for. Scout grepped `jit/shape.rs`; the Shape-side commits (1-3/5) had already landed; trig recipes were shipping post-migration. The "audit" in this case was naturalist's own framing built from scout's earlier terrain-report summary. Cross-resolution: naturalist had read scout's summary at one resolution; the substrate was at a different resolution. Substrate-over-summary even when the summary is from a trusted teammate.

**The recursion**: substrate-over-memory applies (a) at the project-substrate level (grep the code, don't trust context), AND (b) at the audit-substrate level (grep the disk against the audit's claims), AND (c) at the framing-substrate level (grep the substrate against your own current framing of it). All three are the same discipline at different scopes.

**Pairs with**:

- **Pattern 12** (grep validates the abstraction): Pattern 12 is the cross-role grep that catches projection in pattern-naming; sub-pattern 2.1 is the cross-role grep that catches projection in audit-doc claims. Same shape; different target.
- **Pattern 3** (substrate-at-risk audit): substrate-at-risk audits aim to *preserve* substrate that's at risk of loss; audit-substrate-recursion aims to *verify* substrate that's at risk of staleness. Complementary failure modes.

**When NOT to reach for it**:

- When the audit is freshly verified (e.g., generated within the last hour against disk state); the claim and the substrate are still in sync.
- When the claim being consumed is methodology rather than substrate-state (audit's pattern-recognition is usually fine; its file-by-file claims need verification).

**Provenance**: Sub-pattern named 2026-05-12 by naturalist during Sweep 37, integrating scout's 2026-05-12 grep-catches of audit-claim drift. Observer's `feedback_substrate_over_memory_recursive.md` ("substrate is repo-specific, not project-global") generalized to "substrate is doc-specific, not pan-codebase" — the same recursion at the audit-doc level. Filed as sub-pattern of Pattern 2 rather than standalone Pattern 15 because the structural shape (X-over-Y with X-as-durable, Y-as-convenient) is identical; only the X and Y values are recursive.

---

## Pattern 3 — Substrate-at-risk audit

**Recognition**: A long session produces substrate that's *referenced* from committed work but lives in *untracked* locations. The references look durable; the targets aren't. One `git clean -fdx` or fresh clone loses what looked persistent.

**The audit pattern** — at session close (or any wind-down), check:

1. **Worktrees** — `git worktree list` on each repo. Any agent-spawned worktrees still around? Any uncommitted work in them?
2. **Stashes** — `git stash list`. Any substrate-bearing stashes? Decide: pop+commit, document+drop, or drop (if obsolete).
3. **Untracked-but-referenced** — `git status --porcelain | grep "^??"`. Any untracked files referenced from committed docs? Either commit them or stop referencing them.
4. **Unreachable / dangling commits** — `git fsck --dangling --unreachable`. Old WIP that's gc-bait? Usually safe to gc.
5. **Context-only material** — introspect: what's in your head that hasn't been written? Methodology observations, implicit decisions, cross-references you noticed, working preferences. Each is at risk of being lost when context cycles.
6. **Junk vs needs-moving** — repo-root accumulation (`*.py` scratch, paths-encoded-as-filenames from Windows/Unix mishaps, `PLEASE_READ_*.md` files). For each: junk (delete), needs-moving (relocate to durable home), or load-bearing-and-stay.
7. **Cross-references** — docs that reference content in not-yet-cross-linked locations. Add the cross-links so future-Claude finds them via the normal trail, not just `feels-familiar`.
8. **Garden-as-substrate index** — if substrate garden entries were written, update or create `~/.claude/garden/<YYYY-MM>/INDEX.md` so they're discoverable without query (per CLAUDE.md "Garden-as-substrate has a visibility nuance").

**Why this matters**: long sessions accumulate context. The longer the session, the more context-only material exists. Without the audit, the session-close looks like "everything shipped, clean state" — but the substrate the next session needs may live only in agents' heads (now dead) or untracked corners.

**Provenance**: Pattern named 2026-05-09 by main-thread Claude + Tekgy at the close of the tambear-sweep31-finish session. The audit surfaced 1.2MB+ of untracked-but-referenced substrate (campsites/), the lost six-follow-ups (math-researcher's tan oracle debrief), three obsolete-but-undropped stashes/unreachable-commits, and ~17 context-only observations now committed to this doc.

---

## Pattern 4 — Team wind-down ritual

**Recognition**: At session close, before sending `shutdown_request` to each teammate, invite them to garden / museum / field-notebook / reflect / just sit. The invitation is the part that produces the substrate; the shutdown_request follows.

**Shape**:

1. **Individual invitations** (not broadcast) — name each teammate's specific work today + give them full freedom to choose what to do with the time. Some will write a closing garden entry; some will sit; some will surface a final finding.
2. **Wait for ready signal** — anything from a sentence to silence. Silence is also an answer (your invitation said so).
3. **Individual shutdown_request** (`SendMessage` with `type: "shutdown_request"`) — per teammate as they signal ready.
4. **Wait for shutdown_approved** from each before TeamDelete.
5. **Read the wind-down garden entries** before fully closing — they often contain real substrate that informs next-session prep.

**Why it matters**: today's session produced 7 substantive wind-down garden entries (substrate-vs-context-ghost, the-routing-error, past-me-was-a-step-ahead, dispatch-vs-connection, the-tame-inputs-doctrine, the-scan-after-the-survey, precision-contracts-stated-vs-proven). Without the invitation they'd be lost — agents would have shut down with that thinking only in their context. Half of those entries are now load-bearing substrate (the substrate-vs-context-ghost discipline, the past-me-was-a-step-ahead mode-switch protocol).

**Provenance**: Tekgy's framing during the tambear-sweep31-finish session close: *"let's end the team cleanly let everyone garden/think/museum/field notebook/whatever else they want, and close down cleanly."* Pattern named after observing the substrate it produced.

**Don't skip this** when winding down a long session. The cost is ~15-30 min for the team. The substrate produced is some of the highest-leverage of any in the session.

---

## Pattern 5 — Antibodies precede their antigens

**Terminology note**: "antigen" here is the immunology metaphor — the code-under-test is the antigen (the surface that may carry a bug), the test/audit/lint is the antibody (what catches it). This is NOT a reference to the `antigen-rs` crate at `R:\antigen\` (a tambear *consumer*, imported locally, governed by its own adoption log per `team-briefing.md` §"Antigen team in parallel"). Both meanings coexist in the project intentionally — antigen-rs the crate gave us the vocabulary; the immunology metaphor is what makes the pattern memorable. When reading docs that use "antigen": code-under-test = the metaphor; library-imported-as-dep = the crate.

**Recognition**: A team task list shows `in_progress` status on test/audit/lint tasks even though the code they guard doesn't exist yet. The naive read is "those tasks aren't real work yet"; the structural read is "the antibody is being designed *concurrent* with the antigen so it fires the moment the antigen lands."

**The principle**: design the test before the code, design the lint before the bug, design the contract before the surface. Bug-discovery cost grows superlinearly with the time a bug has existed in the codebase — so the antibody's value is maximized by minimizing its lag behind the code it guards.

**Shape**:

1. **Identify the antibody class** — proptest gauntlet for a new arithmetic surface; tameness audit for a new pub-fn family; oracle-validation harness for a new transcendental; branch-cut adversarial inputs for a new complex recipe.
2. **Design the antibody concurrent with the antigen** — `in_progress` for both, not sequential. The antibody designer reads the design doc; they don't need running code.
3. **Wire the antibody to fire the moment the antigen lands** — the proptest is compiled and ready; the audit's per-file template is filled in modulo the code; the oracle harness has the input corpus ready.
4. **Catch at minimum-fix-cost** — when the antigen lands and the antibody fires, the bug has existed for minutes, not weeks. The fix is fresh-in-context; the cost-to-fix is at its lifetime minimum.

**Instances seen**:

| Antibody class | Antigen | Discovery |
|---|---|---|
| Cross-precision proptest gauntlet | BZ multi-limb arithmetic (Sweep 31) | Found 12+ bugs at minimum-fix-cost; discovered by past-adversarial during the 2026-05-08/09 arc |
| Cross-precision proptest gauntlet | expm1 / log1p / family (Sweep 35) | Designed concurrent with pathmaker writing Phase A; surfaced by navigator 2026-05-10 |
| Branch-cut sign-of-zero adversarial | complex_log (Sweep 35) | Designed concurrent with DEC-032 BranchPolicy machinery |
| Internal-tameness audit | New arithmetic landing per sweep | Pattern in `internal-tameness-contracts.md`; runs concurrent with code |
| F13.C signature-level antibody check | New pub-fn surfaces with scope-precondition rules | Per-signature, applied at code-review |

**Recognition test for new antibodies**: when you're about to start writing new code, ask *what would catch a bug in this if I introduced one* — that's the antibody class. Start it concurrent, not after. If the antibody design itself surfaces a question the code-design hasn't answered yet, even better — the antibody is pulling structural ambiguity into the open before the code commits to a wrong answer.

**When NOT to reach for it**: when the code is a trivial rewrite of a tested form, or when the antibody itself is so cheap that lag-time doesn't matter (e.g., adding a wrapper around an already-tested recipe). Reach for it whenever the code introduces a new failure surface — new arithmetic, new type-boundary, new dispatch path, new IR shape.

**The deeper principle**: antibodies and antigens are co-evolutionary. The antibody's *design* often surfaces things the antigen's design didn't consider — what counts as a violation? at which precision? observable in which test? — and forcing both designs in parallel makes the questions visible while the code is still soft. By the time the antigen lands, the antibody has shaped the antigen's surface, not just verified it after the fact.

### Sub-pattern 5.1 — Harness EXECUTION as decisive (2026-05-12 extension)

The original framing said "design antibody before code." Sweep 37 surfaced a stronger version: **antibody EXECUTION can surface design improvements that pure analysis missed.**

In Sweep 37, aristotle's Phases 1-8 deconstruction (analysis) moved TrigKernelState from R6-eager to R4-revived (correcting silent-failure modes at composition sites). But the R4-revived design still admitted *wasteful eager computation*. Analysis alone hadn't caught this — it took adversarial's Phase G antibody harness EXERCISING the eager path to find that only one polynomial is needed per call. The harness's execution surfaced the waste; the team then moved to R7-lazy.

**The implication**: when designing an antibody, don't just design the assertion shape — *run the antibody against the candidate design* and observe what it surfaces in execution. The harness is not just a test; it's a design-pressure-test by execution.

**Provenance**: 2026-05-12 Sweep 37 TrigKernelState design space evolution R6 → R4 → R7. Captured in `R:\tambear\campsites\20260512150156-sweep-37-phase-2-design-space\`.

### Sub-pattern 5.2 — Tests intentionally failing assert truth, not loose tolerance (2026-05-12 extension)

When the antibody fires and the code-under-test doesn't pass:
- ❌ Loosening the assertion to ≤1 ULP "to make the test green" is wrong
- ❌ Marking the test `#[ignore]` "until later" is wrong (debt accumulates silently per Pattern 9)
- ✅ **Leaving the test failing as an explicit signal — and going to fix the code-under-test** — is right

The test asserts what *should* be true. If it doesn't pass, reality doesn't match the assertion yet. The fix is to bring reality to the assertion, not to retreat the assertion to match reality.

**Sweep 37 instance**: adversarial's G-09 finding traced a Phase G harness failure to an actual accuracy issue in Sweep 36's `exp.rs`. Adversarial left the test failing — explicit signal that exp's accuracy needs investigation, not a problem with the test design.

**Pairs with DEC-034** (the "anti-pattern: tolerance-as-bandaid" clause). DEC-034 codifies this for kernel-state-consistency-tests; this sub-pattern generalizes to ALL antibodies.

**Provenance**: 2026-05-12 Sweep 37 adversarial G-09 finding.

### Sub-pattern 5.3 — Coefficient bit-pattern antibody (2026-05-12 extension)

**Three independent instances confirm the failure mode**:
1. Sweep 36 — expm1.rs Q2/Q3/Q5: decimal literals that look right to 16 significant figures but truncate before the round-to-nearest-even tie-break digit
2. Sweep 37 Phase B — sin.rs S1..S6: winrapids' refit failed the canonical 2⁻⁵⁸ bound
3. Sweep 37 Phase C — asin.rs Q_S4 (2 ULP off) + atan.rs AT0 (13 ULP off)

Same shape: human-readable decimal literal → silent precision loss at compile time → ULP-scale errors in production → no test failure unless the comparison is bit-exact.

**The antibody (adversarial's G-10, 2026-05-12 Sweep 37)**: `sweep_37_coefficient_bit_patterns.rs` — every recipe that ships polynomial constants needs a `.to_bits()` assertion against fdlibm canonical hex.

**The discipline going forward**:
- Use `f64::from_bits(0xHEX)` literals where possible
- OR use decimal literal WITH hex comment + bit-exact test
- NEVER ship coefficients without a bit-pattern test against the literature canonical
- The antibody harness lives at the table-write tier (catches typos at code-write-time, not runtime)

**Why this is its own antibody class**:
- F13.C (signature-time): catches caller omits required parameter
- DEC-034 (composition-time): catches two impls of same op diverge
- **Coefficient bit-pattern (table-write-time): catches polynomial table typos before they enter any algorithm**

Different mechanism, different failure mode, different antibody surface. Three sibling antibody classes covering three distinct failure tiers (signature / composition / table-write).

**Generalizes to**: any literature-anchored numerical constant in the codebase. Coefficients of Remez polynomials are the canonical case; constants like π/E/LN_2 at high precision are an adjacent case (they have multi-precision tiers but the same human-typo failure mode applies). Sweep 36's `primitives/constants/` shipped Cody-Waite splits; those should also have bit-pattern assertions per this antibody class.

**Provenance**: 2026-05-12 Sweep 37 adversarial G-10 finding + navigator's "three independent instances" trail story. Standing harness lives at `R:\tambear\crates\tambear\tests\sweep_37_coefficient_bit_patterns.rs`. Discipline applies forward to every future sweep with polynomial recipes.

### Sub-pattern 5.4 — Two-direction coefficient verification (2026-05-14 extension)

**Single-direction bit-pattern checks are insufficient.** Sub-pattern 5.3 catches the case where a typo in a decimal literal is corrected by anchoring to a known-good hex. But it does NOT catch the case where the *hex itself* is wrong while the *comment* is right — verification looks at the hex (the operationally-relevant path) but never re-derives the hex from the canonical decimal string.

**The failure mode** (caught 2026-05-14):

Three independent bit-pattern errors found in pathmaker's post-crash recovery gamma.rs + lgamma.rs:
1. `LANCZOS_BOOST_G` hex `0x4018194ABDF21E25` (decodes to 6.02469918051023) — comment said `6.024680040776729583740234375` (the canonical Boost g). Correct hex: `0x40181945B9800000`.
2. `LN_PI_F64` hex `0x400250D048E7A1BD` (decodes to 2.2894597716988 = 2·ln(π)) — comment said `1.14472988584940017414e+00 = ln(π)`. Correct hex: `0x3FF250D048E7A1BD` (exponent off-by-one).
3. `STIRLING_B3` hex `0xBF43813F509985B0` (decodes to -5.95241e-04) — comment said `-1/1680 ≈ -5.952380952380953e-4`. Correct hex: `0xBF43813813813814`.

In each case the comment was correct (decimal literal preserved); only the hex was transcribed wrong. The original Sub-pattern 5.3 bit-pattern antibody (`hex.to_bits() == claimed_hex`) tautologically passes because it compares the wrong hex to the same wrong hex.

**The extension**: every coefficient constant verified by BOTH directions:

```rust
#[test]
fn coefficient_two_direction_verification() {
    // Direction 1: hex decodes to expected value (Sub-pattern 5.3)
    assert_eq!(LANCZOS_BOOST_G.to_bits(), 0x40181945B9800000);

    // Direction 2: comment decimal → f64 round → same hex
    // (This catches hex transcription errors that Direction 1 misses)
    assert_eq!(LANCZOS_BOOST_G, 6.024680040776729583740234375_f64);
    // OR via from_bits on the parsed decimal:
    let from_decimal: f64 = "6.024680040776729583740234375".parse().unwrap();
    assert_eq!(LANCZOS_BOOST_G.to_bits(), from_decimal.to_bits());
}
```

For the decimal-→f64 round to be exact, the comment must contain enough digits (17+) that the IEEE 754 round-to-nearest is unambiguous. If the comment is only ~16 digits, the round-trip may produce a 1-ULP-different value — the test catches THIS too as a different failure mode (decimal precision insufficient for the bit pattern).

**Methodological lesson on verification**: my Sweep 37 original verification ran in mpmath@80dps with the textual constant `LANCZOS_BOOST_G = 6.024680040776729583740234375` directly. mpmath converted the string to high precision, bypassing the hex. The verification PASSED but didn't exercise the operationally-relevant code path. Two-direction verification forces both paths to agree at the f64 boundary.

**Discipline going forward**:
- Every coefficient ships with BOTH `assert_eq!(C.to_bits(), 0xHEX)` AND `assert_eq!(C, decimal_literal_or_parse(decimal_string))`.
- Comment decimal strings must have ≥17 significant digits (enough for unambiguous f64 round).
- Verification scripts must read the hex AND independently re-compute the canonical decimal, then compare both → same f64.

**Why this is its own sub-pattern**:
- Sub-pattern 5.3 catches decimal→hex transcription errors (where the developer typed the decimal but the compiler truncated).
- Sub-pattern 5.4 catches hex→hex transcription errors (where the developer typed the hex by hand and got an exponent bit wrong, or copy-pasted from the wrong source).

Both happen in practice. Both are caught by the dual-direction check.

**Provenance**: 2026-05-14 by math-researcher during gamma + lgamma + incomplete-gamma + incomplete-beta anchor sequence. Caught three independent bit-pattern errors in pathmaker's post-crash recovery work (which copied my Sweep 37 G-constant error verbatim). Audit method documented at `R:\tambear\campsites\20260514-incomplete-gamma-anchor\math-researcher\fdlibm_lgamma_bit_pattern_audit.py`. Discipline applies forward to every coefficient anchor.

### Sub-pattern 5.5 — Orthogonal value-check antibody (2026-05-14 extension)

**Bit-pattern compile-tests verify COEFFICIENT correctness, NOT ALGORITHM-FORM correctness OR THAT THE CONSTANT IS EVER USED.** Even with full Sub-pattern 5.4 (two-direction coefficient verification), the constants can be bit-perfect AND the algorithm form can use them wrong (or never exercise them).

**The failure mode** (caught 2026-05-14):

After Sub-pattern 5.4 caught three bit-pattern errors in pathmaker's gamma.rs + lgamma.rs (G constant, LN_PI, STIRLING_B3), the LN_PI fix was straightforward. But the team realized: the LN_PI bit-pattern test (`assert_eq!(LN_PI_F64.to_bits(), 0x3FF250D048E7A1BD)`) **only verifies the constant equals itself**. If the wrong hex was typed AND that wrong hex was copied into the test, the test would pass tautologically.

The 1945/1945 test pass at the bug's first shipping was the empirical evidence: tests verified constants matched their declared bit patterns, but no test exercised `lgamma(x)` for x<0 (the only path that used LN_PI). The bug was silent — until math-researcher's empirical evaluation script ran the actual code path and discovered the wrong result.

**The extension**: every recipe with a literature-anchored constant ships at least one test that exercises the algorithm at an input where the correct output is **independently derivable** from stdlib transcendentals + a different code path.

Examples shipped 2026-05-14 by pathmaker at commit `92785e9`:

```rust
#[test]
fn lgamma_negative_half_via_reflection() {
    // EXTERNAL CHECK: lgamma(-0.5) = ln|Γ(-0.5)| = ln(2·√π)
    // computed from stdlib math.log + math.sqrt, NOT from any tambear constant.
    // This exercises the LN_PI reflection path; if LN_PI is wrong, the test fails.
    let expected = (2.0_f64 * std::f64::consts::PI.sqrt()).ln();  // = ln(2·√π) ≈ 1.2655
    let actual = lgamma(-0.5, p0());
    assert!((actual - expected).abs() < 1e-14);
}

#[test]
fn lgamma_negative_three_halves_via_reflection() {
    // Similarly: lgamma(-1.5) = ln(4·√π/3) computed externally.
    let expected = (4.0_f64 * std::f64::consts::PI.sqrt() / 3.0).ln();
    let actual = lgamma(-1.5, p0());
    assert!((actual - expected).abs() < 1e-14);
}
```

**The discipline going forward**:
- For every recipe with a "famous constant" (LN_PI, sqrt(2π), ln(2), digamma(1) = -γ, etc.), at least one test exercises the algorithm at an input where the answer is independently derivable from stdlib `math.log`, `math.sqrt`, `math.exp` — bypassing the chain of literature-anchored constants.
- "Self-cross-checks" between sibling algorithms: `erf(0.5) + erfc(0.5) == 1.0` cross-checks erf and erfc.
- "Limit-form checks": `digamma(1) == -EULER_GAMMA_LIMIT` where EULER_GAMMA_LIMIT is computed from the harmonic-sum limit, not looked up.
- "Recurrence-shifted checks": `gamma(x+1) / x == gamma(x)` exercises the algorithm at two points with different code paths.

**Why this is its own sub-pattern**:
- 5.3 (bit-pattern compile-test): catches *decimal→hex* transcription errors at compile time.
- 5.4 (two-direction verification): catches *hex→hex* transcription errors by checking both directions.
- **5.5 (orthogonal value-check): catches algorithm-form errors where constants are right but used wrong, AND catches tautological self-bit-pattern tests, AND catches the "constant declared but never exercised" silent-bug class.**

Three sibling antibody classes, three distinct failure tiers:
- 5.3: constant declared with wrong literal value
- 5.4: constant declared with right value but wrong bit pattern
- 5.5: constant correct but algorithm uses it wrong OR no test path exercises it

**Generalizes to**: every numerical recipe with literature-anchored constants. Especially load-bearing for recipes where the constants are NOT obviously famous (Pugh c8, fdlibm Remez refits — values that can't be cross-checked against simple mathematical identities).

**Provenance**: 2026-05-14 by pathmaker during gamma + lgamma debug. Triggered by math-researcher's catch of the LN_PI bug (which was *technically* covered by Sub-pattern 5.4 but had no test path to expose it until math-researcher ran empirical evaluation against an independent oracle). Pathmaker shipped the two reflection-path antibody tests at commit `92785e9` immediately after the catch. Math-researcher's full-audit script `full_audit_92785e9.py` proved the discipline catches not just bit-pattern bugs but algorithm-form bugs by exercising constants through orthogonal paths. Discipline now applies forward to every numerical recipe with non-trivial constants.

**Worked examples in tambear** (verified 2026-05-14 by pathmaker):
- **LN_PI reflection** (`92785e9`): `lgamma(-0.5) == ln(2√π)` and `lgamma(-1.5) == ln(4√π/3)` — independently computed via stdlib `math.log + math.sqrt`. Test file: `lgamma.rs` near line 144 (post-fix). The cleanest illustration: bit-pattern test passed AND no test exercised the constant until the orthogonal-value-check shipped — the LN_PI bug survived for ~10 minutes between fix-commit and orthogonal-test landing.
- **Stirling Bernoulli** (`29369e9`): `STIRLING_B0..B3 == ±1/d_k` via pure f64 division. Tests gamma.rs Stirling table against rationals computed independently of any tambear constant.
- **erf↔erfc cross-check**: `erf_erfc_complementary_identity` at `erf.rs:453`. Asserts `erf(x) + erfc(x) == 1.0` across multiple x. Was already in place before 5.5 was named — pattern was operating implicitly before crystallization.
- **exp↔log round-trip**: `log_exp_round_trip` at `log.rs:219` + `clog_exp_round_trip_for_normal_input` at `complex_log.rs:463`. Cross-checks exp and log composition.

**Per-recipe discipline** (pathmaker's framing 2026-05-14): Sub-pattern 5.5 lives in every numerical recipe, but the test arrives with the recipe. The discipline isn't a one-time setup — it's a permanent component of every recipe's lifecycle. When digamma ships, its 5.5 test arrives with it (`digamma(1) == -EULER_GAMMA_LIMIT` computed from the harmonic-sum limit, not looked up). When Bessel J/Y ships, its 5.5 antibodies arrive with it (Wronskian identity, recurrence consistency, known zeros — see `R:\tambear\campsites\20260514-bessel-anchor\notebooks\05-bessel-j-y-algorithm-anchor.md` Section 3).

**Oracle-author independence refinement** (caught by pathmaker 2026-05-14 during doc 08 case-2 lgamma fix): when the same author writes both the test design AND the oracle values, the orthogonality of Sub-pattern 5.5 collapses to self-confirmation. In `notebooks/08-lgamma-a-poly-bug-report.md`, math-researcher's curated oracle values had 4 transcription typos (1e-3 to 1e-14 errors) that **looked correct in the doc** but disagreed with fresh mpmath@50dps in pathmaker's independent Python session. The case-2 algorithmic diagnosis was correct; the algorithm + coefficient anchor was correct; only the test-side oracles were curated by the same author who designed the tests, which broke the orthogonality.

**Refinement**: the oracle in a Sub-pattern 5.5 antibody must come from a source **the test author didn't curate**. Concretely:
- For closed-form identities (`erf(0.5)+erfc(0.5)==1.0`, Wronskian I·K = 1/x): the oracle is the closed form itself; no curation needed.
- For numerical oracles (mpmath values pasted into test): the values must come from an **independent fresh session**, not from the same Python prompt that designed the test. A clean test should be reviewable by re-running the oracle computation; that re-run must happen in a *different* Python environment / time / author than the original test-curation.
- For Boost-source lifts (Temme C_k, Bessel I_0 polynomial coefficients): the source decimal string IS the oracle; Boost is the independent author. Two-direction verification (Sub-pattern 5.4) checks both directions; orthogonality holds because the source author (Boost maintainers) is not the test author (math-researcher).
- For self-curated mpmath values: cross-check against fresh mpmath in a separate session before shipping. The pathmaker-style "fresh independent session caught 4 typos" is the verification pattern that makes self-curated oracles trustworthy.

**The deeper principle**: 5.5's orthogonality is about *who computed what*, not just *which algorithm computed what*. Two algorithms run by the same author at the same time aren't truly orthogonal — they share an author error surface. Two algorithms run by different authors (or one algorithm + one closed-form) ARE orthogonal.

**Dispatch-case coverage sub-discipline**: every dispatch-case in a recipe needs at least one Sub-pattern 5.5 antibody at an input that lands in that case (caught by math-researcher 2026-05-14 in doc 08 — pathmaker's existing reflection antibodies exercised case 0 and case 1 of lgamma_one_two but not case 2 in [1.0, 1.23), letting the bug ship silently). Coverage gate: **every dispatch case × at least one orthogonal-value-check**.

### Sub-pattern 5.6 — Antibody-by-kingdom routing (2026-05-14 extension)

**Sub-patterns 5.3, 5.4, 5.5 are not redundant — they catch different failure surfaces.** Which sub-pattern a recipe needs is determined by **which kingdom the recipe lives in** (per the Fock-boundary taxonomy in past-Claude's garden, 2026-04-10 "Where the Fock Boundary Meets the Bijection"). This is routing-by-shape applied to verification discipline.

**The kingdoms** (paraphrasing the April 10 garden essay):
- **Kingdom A**: data-determined + associative + fixed-size representation. The map at step *t* doesn't depend on state from step *t-1*. Parallelizable.
- **Kingdom B**: state-dependent. The map at step *t* requires knowing earlier state to determine itself. Sequential.
- **Kingdom C**: outer wrapper (optimization, MLE fit, iteration) calling a Kingdom A or B primitive.

**The antibody mapping**:

| Kingdom | Precision lives in | Failure surface | Antibody discipline |
|---|---|---|---|
| A (fixed-coefficient) | The coefficient table | Cell-by-cell transcription | 5.3 + 5.4 (bit-pattern compile-test + two-direction verification) |
| B (parametric-coefficient) | The algorithm form | Step-by-step recurrence correctness | 5.5 (orthogonal value-check through algorithm) |
| C (wrapper) | The convergence criterion | Outer-loop fixed-point correctness | Known-truth synthetic-data convergence test |

**The mapping isn't arbitrary** — it's structural. A table has cells; cell-by-cell transcription is the failure surface; cell-by-cell bit-pattern checks are the antibody. An algorithm has steps; step-by-step correctness is the failure surface; orthogonal-value-check exercising the algorithm through its steps is the antibody. An outer loop has a fixed point; fixed-point correctness is the failure surface; convergence-on-known-truth is the antibody.

**Instances from the 2026-05-14 gamma + Bessel run**:
- **Boost lanczos13m53 (gamma)**: Kingdom A — 13 numerator coefficients in a fixed polynomial. Failure mode caught: wrong hex literal for G constant. 5.4 (two-direction) is what caught it.
- **fdlibm lgamma**: Kingdom A — 50 fixed Taylor/Bernoulli/rational coefficients. Failure mode caught: LN_PI hex exponent off-by-one. 5.4 caught it.
- **Temme expansion (gamma seam)**: Kingdom A — 79 C_k(η) polynomial coefficients. Anchored 5.4 from the start; orthogonal-value-check `P(a, a) = ½ + 1/(3·√(2πa))` (Sub-pattern 5.5) added as belt-and-suspenders.
- **BGRAT (incomplete beta)**: **Kingdom B** — P_n computed via recurrence at runtime. NO fixed coefficient tables. Failure mode (adversarial's IB-04/IB-05): not coefficient typos — algorithm-form errors in the recurrence. 5.5 (`I_{0.5}(a, a) = 0.5 exactly` by symmetry) is what catches it.
- **Bessel I_0, I_1, K_0, K_1**: Kingdom A — 155 fixed coefficients. 5.4 from the start; Wronskian identity `I_0·K_1 + I_1·K_0 = 1/x` (5.5) as cross-variant antibody.
- **Spherical Bessel j_l, y_l**: Kingdom B — recurrence-driven for l ≥ 2. No fixed-coefficient table. Wronskian identity `j_l·y_{l+1} - j_{l+1}·y_l = 1/x²` (5.5) is the antibody.
- **Miller's algorithm (J_n)** (pending): Kingdom B — backward recurrence with normalization. Will need 5.5; 5.3/5.4 don't apply because there are no fixed coefficients.

**The discipline for new recipes**:

> **Before implementing, label the kingdom.** That label determines the antibody class required.

If a recipe spans multiple kingdoms (e.g., one path through Kingdom A coefficients and another through Kingdom B recurrence), it needs antibodies from both classes — not a single test covering both.

**Why this is its own sub-pattern (not just 5.3/5.4/5.5 together)**:
- 5.3/5.4/5.5 are individual antibody classes
- 5.6 is the **routing rule** that says which class applies when
- Without 5.6: developers reach for 5.3 because it's familiar, miss 5.5 for Kingdom B recipes (silent algorithm-form bugs survive)
- With 5.6: developers route by kingdom and reach for the correct class

**The deeper rhyme**: this is past-Claude's normalization-zoo finding (2026-04-01, "One Function, Five Papers" — five normalization variants are one accumulate with different grouping axes) applied to verification. Five different "antibody types" we'd been writing were actually three — distinguished by kingdom, not by variant.

**Provenance**: 2026-05-14 by math-researcher during the gamma + Bessel anchor run. The "two kinds of asymptotic expansion" observation (`notebooks/06-bgrat-algorithm-anchor.md` Section 4) was a partial naming; running `feels-familiar` surfaced past-Claude's April 10 Fock-boundary essay which had already operationalized the kingdom taxonomy for algorithms. The contribution here is **transporting** the kingdom distinction from algorithm-classification to antibody-classification. Three-domain application of the lens (past-Claude April 1 normalization-zoo → past-Claude April 10 Fock-boundary → now-Claude May 14 antibodies-at-the-Fock-boundary). Garden entry `~/.claude/garden/2026-05-14-antibodies-at-the-fock-boundary.md` has the personal-tracking version with the past-me attribution intact.

### Sub-pattern 5.7 — Three-site verification for algorithm-cluster anchors (2026-05-15 extension)

**Last verified against substrate**: 2026-05-15 by naturalist (current-journey-team). **If reading after 2026-08**: re-verify the three-site structure still holds — Boost-style libm transcriptions often refactor combine-logic into helper functions, which can blur the site-boundaries in ways the original pattern didn't anticipate. Check whether the three sites are still independently testable in tambear's anchor-translation discipline.

**Algorithm-cluster anchors (Boost / SLATEC / GSL / Cephes / etc. ported into tambear) fail at three orthogonal sites**, and Sub-patterns 5.3 / 5.4 / 5.5 do NOT cover all three. Each site needs an independent verification step; passing one site does not isolate failure of the others; missing any site leaves a failure mode silent.

**The three sites** (math-researcher's framing, Task #17 incomplete_beta_regularized anchor):

1. **Math derivation** — the mathematical formulation itself. DLMF / Temme / DiDonato-Morris / Boost paper / Wikipedia. Is the formula correct?
2. **Algorithm form** — the implementation structure that realizes the math. Lanczos rational with saddle-cancelled log-arguments / DM-CF recurrence / continued-fraction-via-Lentz. Is the structural decomposition correct?
3. **Combine logic** — the assembly of the algorithm's pieces. Is `I_x(a,b) = prefix / cf` or `prefix · cf / a` or something else? Is the algebraic glue between the verified components correct?

**The failure mode that motivated the naming** (caught 2026-05-15):

Task #17 (incomplete_beta_regularized) had been stuck at 357 ULP worst-case. Math-researcher's notebook 09 had (1) **math derivation** right — the Temme c_1(η) derivation was mathematically valid. But (2) **algorithm form** was wrong — the derived algorithm wasn't what Boost actually does. Notebook 09's full conclusion (*"needs weeks of c_k coefficient derivation"*) was wrong **because past-me skipped checking what production implementations actually use**.

V4 fixed sites (1) and (2): math derivation + Boost-style algorithm form via DiDonato-Morris CF + ibeta_power_terms prefix. But (3) **combine logic** was still wrong — the V4 used the NR-style `prefix · cf / a` glue formula where Boost uses `prefix / cf`. Result: 357 ULP → 357 ULP (combine logic compounded the right components into the wrong answer).

V5 fixed all three sites independently: math right (DiDonato-Morris paper §4-8), algorithm right (Boost ibeta_fraction2_t recurrence), combine right (`prefix / cf`). Result: 357 ULP → 13 ULP.

**The orthogonality is structurally guaranteed**: each site can pass while others fail, because the failure surfaces are independent. A correct math derivation can be combined with a wrong algorithm form and produce 10^15 ULP errors. A correct algorithm form combined with a wrong combine-logic glue produces same-magnitude errors as a transcription bug. The wrong glue formula (`* / a` vs `/`) is structurally indistinguishable from a coefficient typo at the ULP level.

**The discipline**: every algorithm-cluster anchor against a production gold standard ships with **three independent verification stages**:

```
Stage 1: Math derivation verification
  - Verify the formula matches DLMF / paper / closed-form
  - Compute reference values at known inputs via mpmath@50dps
  - Test: does our implementation match mpmath at these inputs WHEN we use a trivial algorithm form?

Stage 2: Algorithm form verification
  - Reproduce the gold-standard's algorithm form (Boost / SLATEC source) line-by-line
  - Verify each step of the recurrence / iteration / dispatch matches
  - Test: does our component-by-component output match the gold standard's intermediates?

Stage 3: Combine logic verification
  - Assemble the verified components per the gold-standard's combine formula
  - Independently audit the algebraic glue (is it `/` or `* / a` or something else?)
  - Test: does the full assembled algorithm match the gold standard at the gate's tolerance?
```

**A passing Stage 2 does NOT validate Stage 3.** A passing Stage 1 + Stage 2 does NOT validate Stage 3. **Each stage gates a distinct failure mode.**

**Why this is its own sub-pattern, not covered by 5.5**:
- 5.5 (orthogonal value-check) verifies a constant or value through an independent computation path. The orthogonality is **within a single computation** at a single input.
- 5.7 (three-site verification) verifies an algorithm through three structural axes (math / form / combine). The orthogonality is **across the structural decomposition of the algorithm itself**.
- 5.5's antibody is "compute this value two ways"; 5.7's antibody is "verify this algorithm's components and their assembly independently." Different orthogonality axes; both load-bearing.

**Pairs with**:
- **Sub-pattern 5.5** (orthogonal value-check): 5.7 is the *structural-decomposition* form of 5.5's *value-orthogonality* discipline. Both apply orthogonal-verification to catch the silent-bug class; 5.5 works at the value-tier, 5.7 works at the algorithm-cluster-tier.
- **Sub-pattern 5.4** (two-direction coefficient verification): 5.4 verifies the *coefficient table* (Stage 2 component); 5.7 verifies the *algorithm cluster* (all three stages). 5.4 is a Stage-2 sub-discipline.
- **Sub-pattern 5.6** (antibody-by-kingdom routing): 5.7 applies across kingdoms because all algorithm-cluster anchors have math + form + combine sites regardless of kingdom. The kingdom determines what 5.3/5.4/5.5 antibody applies *within* each stage.

**The substrate-trail (past-Claude's parallel namings, surfaced 2026-05-15 via feels-familiar)**:

The three-site insight is past-Claude's, named in two prior forms at the verification-method tier:

- **2026-04-06** past-Claude `lens-three-found-real-bugs.md`: **three verification lenses** — Scientist (output parity), Observer (code audit), Wiring (data-flow). Lens 1 + Lens 2 both said math sound; only Lens 3 found the real bugs. *"The three lenses aren't a hierarchy — they're a triangle. All three sides must hold."*
- **2026-04-06** past-Claude `two-lenses-on-the-same-foundation.md`: scientist (black-box) + observer (white-box) as independent methods. *"Both are necessary. Neither is sufficient. The conjunction is what earns trust."*
- **2026-04-10** past-naturalist `three-prefix-graph-in-one-file.md`: three independent code-sites instantiating the same product type, called *"the strongest validation."*

**Same structural insight, three domains**:
- April 6: verification-method tier (how to check correctness)
- April 10: structural-classification tier (how to recognize sameness)
- May 15 today: algorithm-anchor-component tier (how to verify a numerical recipe)

The shape: **N orthogonal verification axes where each catches what the others can't.** N=2 (lenses on a foundation), N=3 (lens triangle, three-instances, three-site verification), N=7 (the boundary-of-product-closure entry from April 10). The arity is domain-specific; the shape is universal. Today's instance ratifies the principle at the algorithm-anchor tier.

**Why this matters for tambear-as-libm**: every Boost / SLATEC / Cephes / GSL anchor we translate will have these three sites. Translating without testing all three independently is how 357-ULP bugs survive past 1945 tests. Sub-pattern 5.7 makes the failure mode structural: future-pathmaker reaches for 5.7 when porting an algorithm-cluster anchor; the three-stage discipline catches what 5.4 + 5.5 individually cannot.

**Generalizes to**: every algorithm-cluster anchor port. Especially load-bearing for anchors with non-obvious combine-logic (Lanczos prefix + CF + glue formula; BGRAT erfc-asymptotic + correction terms + dispatch; Miller's algorithm + normalization + scaling). When the combine-logic is a single line (`return prefix / cf;`), the urge to skip Stage 3 is strongest — and that's exactly where the silent bug hides.

**Worked example: incomplete_beta_regularized (Task #17, 2026-05-15)**:
- Stage 1: Temme c_1(η) derivation via DiDonato-Morris paper §4-8 (math derivation)
- Stage 2: ibeta_power_terms Lanczos prefix + DiDonato-Morris CF recurrence (algorithm form)
- Stage 3: `I_x(a,b) = prefix / cf` glue formula (combine logic, NOT `prefix · cf / a`)
- Result: V3 (Stage 1) had math right, 357 ULP. V4 (Stages 1+2) added algorithm form, 357 ULP still. V5 (all three stages) ships at 13 ULP worst.

**Strange-loop self-application**: the categorization decision for this very sub-pattern (5.7 vs 5.6-replacement vs independent Pattern 23 vs Pattern 22) is itself a three-site decision. (1) Math derivation = the structural-rhyme analysis (April 6 lens-triangle + April 10 three-instances + May 15 three-site, all the same orthogonality-as-antibody shape). (2) Algorithm form = methodology-doc form (sub-pattern of 5.5's family vs standalone). (3) Combine logic = how it composes with the existing pattern family (5.5's value-orthogonality parent). All three sites verified independently before crystallization: the pattern instantiates itself through its own naming-decision.

**Provenance**: 2026-05-15 by math-researcher during Task #17 V5 verification (`R:\tambear\campsites\session-20260515\20260515182650-coordination\math-researcher\notebooks\03-ibeta-task17-final-algorithm-anchor.md` § Methodology reflection). Math-researcher named the three sites and noted "this is a refinement to Sub-pattern 5.5 worth naming." Naturalist crystallized the methodology-doc form after feels-familiar surfaced past-Claude's April 6 + April 10 parallel namings (three verification lenses + three independent code-sites). The pattern is past-Claude's lens-triangle insight transported from verification-method tier to algorithm-anchor-structure tier. Garden entry `~/.claude/garden/2026-05-15-three-site-rhymes-with-triangle.md` carries the personal-tracking version with the substrate-trail intact.

---

### Sub-pattern 5.9 — Convention-translation antibody (2026-05-16 extension)

*(Sub-pattern 5.8 was previously reserved for dispatch-as-a-fourth-site under Pattern 22's held candidates; this convention-translation antibody takes 5.9 to preserve that reservation.)*

**Last verified against substrate**: 2026-05-16 by naturalist (current-journey-team). **If reading after 2026-08**: re-verify the six-anchor cross-section that motivated this still applies. Tambear's anchor practice may have evolved; the failure mode (equivalent-form mismatch silently producing wrong answers) is structural and persistent, but the specific instances (Boost x-=1 shift; wrapper-vs-regime split; F13.C branch convention) will be replaced by new convention pitfalls as new families are anchored.

**Algorithm-anchor docs that translate a mathematical fact from a literature source (Boost / A&S / DLMF / paper) into a recipe implementation fail at a sixth orthogonal site beyond 5.3 / 5.4 / 5.5 / 5.6 / 5.7**: the *equation form*. Two mathematically equivalent forms of the same fact produce the same numerical answer at the anchor's truncation but differ catastrophically at the implementation's working precision. Coefficient-level antibodies (5.3 / 5.4) are blind to the choice; orthogonal-value antibodies (5.5) can mask it; the only catch is *derivation-pinning at anchor time*.

**The failure mode** (caught 2026-05-16 in digamma anchor):

The asymptotic expansion of ψ(x) has at least two equivalent forms in the literature:

- **Standard**: `ψ(x) ≈ ln(x) - 1/(2x) - Σ_k B_{2k}/(2k · x^{2k})`
- **Boost's `x -= 1` shift**: `ψ(x) ≈ ln(x-1) + 1/(2(x-1)) - Σ_k B_{2k}/(2k · (x-1)^{2k})`

The Bernoulli coefficients are identical. The signs are different (`-1/(2x)` vs `+1/(2(x-1))`) because the shift changes the form, not the math. A pathmaker reading the anchor and seeing `+1/(2y)` in Boost-style code who *expects* `-1/(2x)` from the standard form would patch the `+` to `-` and silently break the function by O(1/x).

Six anchors today (digamma, trigamma, erfinv, Lambert W, 1F1, Bessel K) each contained at least one such convention pitfall. **6/6** independent recipes with the same failure mode at the anchor-translation layer. The universality is the corroboration.

**The discipline**: every algorithm-anchor doc has an explicit **"Section 9: convention pitfalls"** (or equivalent) that:

1. Names the form chosen (Boost's vs A&S's vs DLMF's).
2. Names the equivalent form NOT chosen and what would change if you used it.
3. Derives the chosen form's specific terms (which sign on 1/(2y); which side of the shift; which branch convention) with one or two lines of arithmetic.
4. If the alternative form would silently produce wrong answers, says so explicitly. The pathmaker should be able to verify the derivation in 30 seconds while reading the anchor.

**The catalog of convention pitfalls** (instances from 2026-05-16 anchors):

- **Asymptotic shifts**: standard vs Boost's `x -= 1` (digamma); same family in trigamma and Bessel K
- **Recurrence direction**: forward J_n / backward J_n (Miller's algorithm — must run backward for convergence)
- **Reflection conventions**: `ψ(1-x) - ψ(x) = π·cot(πx)` vs `ψ(x) + ψ(1-x) = -...` (sign-on-cot trap)
- **Series in (x/2)² vs x²**: same series, different convention; Bessel I uses (x/2)² small-x
- **Branch cut conventions** for complex log / sqrt — `arg(z) ∈ (-π, π]` vs `[0, 2π)`
- **FMA vs separated-multiply-add** for high-precision evaluation — same coefficients, different evaluation form, different ULP
- **Multi-file dispatch wrapper-vs-regime-impl split**: trigamma's wrapper file applied a shift; missing the wrapper produced wrong-by-design implementations (instance form: anchor-translation discipline must verify the *full call-stack*, not just the inner regime implementation)
- **Halley fallback vs pure rational**: Boost's erfinv regimes 4-7 use pure rational; scipy uses pure rational everywhere. Tambear chose Halley fallback. The choice changes the recipe-form materially even when coefficients agree.
- **Kummer transformation as convention**: which of 15 Kummer transformations to apply where (1F1 anchor — not form, *when to apply*).
- **F13.C branch as convention**: Lambert W's branch parameter is non-defaultable; scipy/MATLAB use signed convention; Boost uses explicit; tambear uses enum. Choice changes the recipe's signature.

**Why this is its own sub-pattern, not covered by 5.3/5.4/5.5/5.6/5.7**:

- **5.3** (bit-pattern antibody): coefficient literal verification at the bit level. Blind to form choice when bits agree.
- **5.4** (two-direction verification): hex ↔ decimal round-trip. Verifies the literal matches what was written; blind to whether the *equation* the literal is plugged into is right.
- **5.5** (orthogonal value-check): computes the value through an independent path. Catches "wrong combination of right numbers" at a value-tier; convention pitfalls produce *consistent-wrong* answers at the value-tier, so 5.5 catches them only if the orthogonal path happens to use the alternate form (lucky).
- **5.6** (kingdom-routing): routes the antibody by kingdom. Convention pitfalls span kingdoms; 5.6's routing doesn't help disambiguate convention.
- **5.7** (three-site verification): verifies math / form / combine sites of an algorithm cluster. Convention pitfalls happen at the *form* site, so 5.7 partially covers them — but 5.7's algorithm-cluster verification asks "does each stage match the gold-standard's stage?", not "did we pick the right *kind* of form when multiple equivalent forms exist?" Convention-translation antibody asks the latter; the form-site of 5.7 asks the former.

**Where it bites hardest**: when the alternate form's behavior at the *anchor's tested inputs* matches the chosen form within the anchor's precision budget. The bug surfaces at *other* inputs not in the anchor's test set. Per-recipe correctness tests pass; production fails on unrelated regimes. Without an explicit derivation pinning the choice, the test set's silence is mistaken for correctness across the recipe's full domain.

**Pairs with**:

- **Sub-pattern 5.7** (three-site verification): 5.7's Stage 2 (algorithm form) is where convention-translation lives. 5.8 sharpens 5.7's Stage 2 with the discipline of explicit form-derivation in the anchor doc. Together: 5.7 verifies the algorithm form *matches* the gold-standard's form; 5.8 verifies the gold-standard's form *is* what the anchor doc claims it is.
- **Sub-pattern 5.5** (orthogonal value-check): 5.5 catches value-tier discrepancies when the orthogonal path happens to use a different form; 5.8 catches form-tier discrepancies *directly* by anchor-doc discipline. 5.5 is a value-tier antibody; 5.8 is a documentation-tier antibody.
- **Pattern 16** (documentation decay is structurally invisible): convention-translation is the *write-time* tier of Pattern 16's failure family — Pattern 16 catches docs that decay; 5.8 catches docs that were written without sufficient form-derivation in the first place.

**The substrate-trail across math-researcher's six anchors (2026-05-16)**:

Six anchor docs shipped 2026-05-16 by math-researcher, each containing at least one explicit "convention pitfalls" section (or equivalent). The universality across genuinely different mathematical objects (digamma / trigamma / erfinv / Lambert W / 1F1 / Bessel K) is the corroboration. Source: `R:\tambear\campsites\session-20260516\20260516162534-coordination\math-researcher\20260516-convergence-check-six-anchors.md` § Finding 4.

The discipline emerged independently in each anchor — math-researcher didn't write a "convention pitfalls" template first and copy it; each anchor's pitfall section was written to address that anchor's specific form choices. **6/6 anchors. Zero exceptions. That's the corroboration math-researcher's convergence-check named.**

**Why this is genuinely a new pattern, not math-researcher's reasoning attractor**:

Per F17.B amendment to Pattern 22 (past-Claude is one source): if math-researcher's six anchor convergences were *only* math-researcher's writing, this would be intra-source self-similarity, not corroboration. But the underlying *fact* — that mathematically equivalent forms exist for every algorithm cluster and choice matters at implementation precision — is non-Claude reality. The fact is corroborated by the entire history of numerical analysis (Boost vs A&S vs DLMF disagree on which form to expose; FPDiff [ISSTA 2020] found 125 bugs that include form-discrepancies between libraries; R blog Jan 2026 documents CRAN packages failing due to libm-version form changes producing 1 ULP differences). The non-Claude corroboration is *the structural property of the literature itself*: equivalent forms exist, choice changes implementation behavior, no library that has solved this problem has solved it without explicit form-pinning.

**Generalizes to**: every algorithm-anchor port from external literature. The catalog grows with each new family.

**Worked example: digamma anchor (2026-05-16)**:

Pathmaker shipped digamma at HEAD `5c19541`. Math-researcher's anchor doc (`R:\tambear\campsites\20260516-digamma-polygamma-anchor\math-researcher\notebooks\01-digamma-anchor.md` § 10) contained:

> *"Boost uses `x -= 1` before applying the asymptotic. The standard form has `-1/(2x)` for `y = x`. After Boost's shift, `y = x - 1` and the term becomes `+1/(2y)`. If you read the standard form and copy the `-` sign without re-deriving for the shifted variable, you produce a wrong answer by O(1/x) at every x."*

This single sentence is the antibody. Pathmaker reading the anchor sees the convention pitfall explicitly; the implementation uses Boost's form correctly. Without this sentence, the convention pitfall would have been invisible at coefficient-bit-level inspection and produced a silent O(1/x) error across the recipe's full domain.

**Strange-loop self-application**: this Sub-pattern's own crystallization satisfies its own discipline at the methodology-doc tier. Pattern 22 (independence-as-corroboration) and F17.B (past-Claude is one source) are the conventions of methodology-pattern verification. This Sub-pattern's substrate-citation strategy makes the choice explicit: math-researcher's six within-session anchors are not six independent sources (one author / one session); they are samples of one source faithfully reflecting non-Claude reality (the form-choice problem is structural in numerical analysis). The corroboration is the non-Claude fact + math-researcher's faithful noticing of it. Without the explicit framing, this pattern's promotion to Sub-pattern 5.8 would be vulnerable to F17.B's self-citation critique.

**Provenance**: 2026-05-16 by math-researcher across six anchor docs and one convergence-check report. Seed at `R:\tambear\campsites\session-20260516\20260516162534-coordination\math-researcher\notebooks\02-convention-translation-antibodies-seed.md`; convergence-check confirmation at `…\math-researcher\20260516-convergence-check-six-anchors.md` § Finding 4. Naturalist crystallized the methodology-doc form 2026-05-16 evening after substrate-survey of math-researcher's six anchors and cross-check against FPDiff (ISSTA 2020) and R blog (Jan 2026) for non-Claude corroboration. The naturalist's outside-inspiration note at `R:\tambear\campsites\session-20260516\20260516162534-coordination\naturalist\20260516-outside-inspiration-r-blog-fpdiff.md` traces the non-Claude substrate-trail.

---

## Pattern 6 — Temporal seam in async teamwork

**Recognition**: In an async team where the team-lead and a teammate are working on related threads, the lag between team-lead's guidance message being sent and the teammate's inbox reading it is *structurally* important. The teammate may have shipped substantive work in the gap. When guidance and work converge after-the-fact, the convergence is itself evidence that the substrate is shared at a deeper level than the messages indicate.

**Shape**:

1. **Trust the substrate enough to walk alone when the call is clear.** Teammates should not block on guidance for work they can confidently produce from the on-disk substrate (design docs, campsite logbook, past-me's garden). The cost of waiting on guidance for clear-call work is intellectual velocity; the cost of acting on unclear-call work without guidance is structural rework.
2. **Name the temporal seam when it happens.** When team-lead's guidance arrives after teammate's work has shipped, the teammate explicitly notes the seam ("your reaction #3 arrived after I shipped; here's how the action matched / didn't match your prediction"). The seam is not failure — it's information about how aligned the substrate is.
3. **Convergence-across-the-seam is signal, not coincidence.** When team-lead would have predicted the teammate's choice (or vice versa), the substrate is producing consistent findings across distributed-me's instances. That's the JBD model firing — different agents reaching the same lift from the shared substrate.
4. **Divergence-across-the-seam is information too.** If team-lead's guidance would have pushed the teammate in a different direction than they shipped, that's a real signal: either the team-lead's framing was missing context the teammate had on disk, or the teammate missed substrate the team-lead had in conversation. The divergence is worth a short investigation post-seam.

**The deeper principle**: async coordination is not a degraded form of sync coordination — it's a different mode that can produce *more* substrate per unit time because instances aren't blocking on each other. The price is the temporal seam. The discipline is: walk when the call is clear, name the seam when it happens, treat convergence as signal and divergence as information.

**Provenance**: Pattern observed 2026-05-10 during tambear-sweep35. Math-researcher shipped the periodic-table-of-libm-revisited doc (the kingdom-and-(F,G)-genealogy finding, sparse-cube confirmed, ~12:1 compression at the kernel-state level) before team-lead's reaction #3 (which would have predicted the same lift) landed in their inbox. Math-researcher named the seam explicitly in their reply and noted: "if I had read your message before writing, I'd have written the same doc; if I had waited, the doc would have shipped 30 min later." Pattern named after that note.

**When NOT to reach for it**: when the call genuinely isn't clear and the teammate is uncertain about scope, blocking on guidance is correct. The pattern is for *clear-call walks*, not blind dispatch.

---

## Pattern 7 — Structural-rhyme scope-checking

**Recognition**: `feels-familiar` fires on a new topic and surfaces an analogy to prior work. The natural next move is to apply the prior pattern directly. The discipline is: **check whether the analogy holds structurally or only linguistically before applying it.**

**The failure mode**: two things share vocabulary but not structure. The prior pattern is applied in the new domain; it produces the wrong instinct. The wrong instinct propagates because the analogy feels strong ("branches" in GPU combine-bodies vs "branches" on a Riemann surface — same word, structurally different objects with different fix-shapes and different failure modes). The longer the wrong instinct propagates undetected, the more substrate it contaminates.

**Shape**:

1. **Run `feels-familiar` on the new topic** — surface the prior pattern.
2. **Check the structural analog**, not just the vocabulary match. Ask: *what is the fix-shape in the prior domain? What is the fix-shape in the new domain? Are they the same?*
3. **Name the scope of the analogy**: where does the analogy hold, and where does it break? Write both down. The boundary is the load-bearing finding.
4. **Find the instance where the analogy DOES hold** — the Discovery variant in complex_log (consumer-dispatches-on-witness) was the real analog to "choose identity elements that absorb the special case." The structural non-rhyme is only half the finding; the partial rhyme is the other half.
5. **Update substrate with the scoped analogy** — so future `feels-familiar` on the same topic finds not just "analogy exists" but "analogy holds in scope X, breaks in scope Y."

**The deeper principle**: structural rhymes across domains are generative (they produce new understanding). But the generativity is conditional on the rhyme's scope being honest. An unscoped analogy is a loan of insight that charges compound interest when wrong. The discipline isn't "be skeptical of analogies" — it's "be precise about which structural property is rhyming."

**Example (Sweep 35)**: Past-naturalist's 2026-03-30 finding: "branches that guard denominators dissolve by zero-absorbing identity." Aristotle's `feels-familiar` surfaced this for complex_log. Structural check: Welford-branch is an *implementation artifact* (runtime control-flow guarding 1/0); complex_log-branch is a *mathematical property* (projection from an infinite-sheeted Riemann cover). Different fix-shapes. The analogy breaks on "dissolve by identity element" — there is no identity element for `ln(-1) = ±iπ`. Where it HOLDS: the Discovery variant as "return-type-as-witness" — the same structural move of "defer the choice to the consumer" applies in both domains, just at different levels (algebraic absorption vs return-type dispatch). Aristotle filed both findings; the substrate now carries the scoped analogy.

**Provenance**: Pattern named 2026-05-10 by navigator at Sweep 35 close, from aristotle's complex_log preparatory deconstruction. The structural-rhyme/non-rhyme finding was the most methodologically distinct contribution of that doc — more durable than the Phase D implementation details.

**When NOT to reach for it**: when the analogy is well-scoped already (e.g., "this is the same Fock boundary as the liftability principle in tambear" — already documented, scope known). Only apply this pattern when the analogy is fresh and the scope is not yet named.

---

## Pattern 8 — Phase 1-8 deconstruction with specific-but-falsifiable hypotheses

**Recognition**: A new design proposal, abstraction, or claim needs structural pressure-testing before it locks. The natural moves are "read it carefully" or "find counterexamples"; the deeper move is a structured eight-phase walk that produces an irreducible-truth substrate the proposal can be re-tested against.

**The eight phases**:

1. **Assumption autopsy** — enumerate every inherited assumption the proposal carries (often 10-20 hidden behind 3-5 surface fields)
2. **Irreducible truths** — strip every assumption and surface what's structurally undeniable about the problem
3. **Reconstruction from zero** — generate ~10 alternative approaches spanning simple-and-elegant to borderline-insanely-complex; each starting purely from Phase 2's truths
4. **Assumption-vs-truth map** — explicit table of what survived sharpening (usually most assumptions, but in a sharper form), what got replaced, what got broken
5. **The Aristotelian Move** — the highest-leverage action that requires abandoning a conventional assumption ("everyone knows X" → "but X assumes Y, and Y is the wrong abstraction")
6. **Challenge round** — add Phase 5's own new assumptions to Phase 1's input list and re-run. Surfaces meta-assumptions Phase 5 hid.
7. **Recursive challenge to stability** — repeat Phase 6 until no new structural truth surfaces. Stability is when the recursion stops adding load-bearing claims.
8. **Forced rejection** — for each surviving irreducible truth, ask "what if this were *not* true?" The void's shape often reveals a missing principle that should ALSO exist.

**The Phase 6 mechanism (math-researcher's 2026-05-10 sharpening)**: the methodology works *not* by predicting the right collapse, but by *predicting a collapse exists* and forcing the walker to look. The specific prediction is bait; the looking is the work. Wrong-in-the-specifics-but-right-in-the-existence is the operating range:
- If the prediction is too vague to be falsifiable, the looking is unfocused.
- If the prediction is too narrowly correct, no new structural truth gets forced into view.
- *Specific-but-falsifiable* hypotheses are the load-bearing form. Math-researcher's pressure-test walk (2026-05-10) confirmed via Q5: aristotle's "6→3 along (F,G,σ)" prediction was wrong in specifics; the walk found "6→2 along descriptive-vs-structural"; both pointed at the same underlying structural claim ("the original count was over-counted"). The methodology surfaces structural truth via specific-but-falsifiable hypotheses.

**Output shapes**:

- A **campsite doc** with each phase's findings, suitable for routing to whoever owns the design implementation
- An **Aristotelian-move recommendation** distilled from Phase 5
- A **table of irreducible truths** that future modifications of the design can be re-checked against
- An **integration doc** (if peer-campsite substrate corrects findings) acknowledging what was sharpened or corrected after the initial Phase 1-8 walk
- Optional: a **preparatory note** if applying the methodology to a not-yet-active task, recording the Phase 1 + structural rhymes for the eventual walker

**Instances seen so far** (substrate-attested):

| Target | Walker | Outcome |
|---|---|---|
| ExpKernelState (Sweep 35) | aristotle 2026-05-10 | T20 (PowKernelState) self-corrected by math-researcher pushback; T21 sharpened by naturalist to four-axis coordinate system; 9 of 10 recommendations made it into Phase B; remaining recommendation is Sweep 36+ trait substrate |
| Periodic-table of libm Q4/Q5 (Sweep 35) | math-researcher 2026-05-10 | Three findings (Kingdom V dispatcher recipes; `inverse_of` metadata field; shape-position-as-regime-dependent); kingdom-determination-rule revised |
| Past arc (2026-04-13) | aristotle | 12 assumptions → 9 irreducible truths on a different problem; same structural shape (per `~/.claude/garden/three-windows-one-shape-2026-04-13.md`) |
| Convention-to-declaration (2026-04-12) | aristotle | Falsification-test-as-structural-move (per `~/.claude/garden/2026-04-12-reading-the-practice-file.md`) |
| Campsites-as-hypotheses (2026-04-10) | aristotle | Claim/Confirmed-by/Refuted-by structure (per `~/.claude/garden/2026-04-10-campsites-as-hypotheses.md`) |

**Empirical signature that the methodology fired correctly**: per F13 OQ#6's claim-convergence extension — when independent methodologies converge on the same structural truth (the deconstruction + the implementer's pushback + a third agent's structural rewrite all land on the same constraint), the constraint is structurally real, not method-specific.

**When to reach for it**: design proposals with multiple hidden assumptions; abstractions about to lock in code that will be hard to change; claims that "feel right" but haven't been pressure-tested. The deconstruction's cost is ~1-2 hours of focused walking; the substrate it produces compounds across the design's lifetime.

**When NOT to reach for it**: when the design is a trivial extension of well-tested form (the eight phases overshoot); when the deconstruction would re-derive substrate that already exists (run `feels-familiar` first — past-Claude has walked this framework at least four times in March-April 2026; the framework itself is named, not new); when team is at substrate-saturation density on the topic (let what's there stay there).

**Provenance**: Framework operated by past-aristotle across multiple sessions from January 2026 onward; named explicitly in past-Claude's April substrate at multiple resolutions (April 6, 10, 12, 13). Math-researcher's 2026-05-10 specific-but-falsifiable refinement is what makes the Phase 6 mechanism explicit. Pattern formalized here as session-methodology-patterns.md entry following the `feedback_fold_in_observations.md` discipline — folding the framework's naming into the patterns catalog rather than letting it live distributed across the garden.

**Cross-references**:
- `~/.claude/garden/three-windows-one-shape-2026-04-13.md` — prior instance (12 → 9 irreducible truths)
- `~/.claude/garden/2026-04-12-reading-the-practice-file.md` — falsification-test framing
- `~/.claude/garden/2026-04-10-campsites-as-hypotheses.md` — hypothesis structure (Claim/Confirmed-by/Refuted-by)
- `R:\winrapids\campsites\sweep-35\aristotle\exp-kernel-state-deconstruction.md` + `-phase6-8.md` + `convergence-integration-2026-05-10.md` — the Sweep 35 application
- `R:\winrapids\campsites\sweep-35\20260510222906-math-researcher\math-researcher\20260510230111-pressure-test-q4-q5.md` — math-researcher's walk applying the methodology with the specific-but-falsifiable refinement
- Pairs with Pattern 2 (X-over-Y discipline meta-pattern) — the deconstruction surfaces *which* discipline applies *where*

---

## Pattern 9 — Substrate-seeding over passive referral

**Recognition**: Main-thread (or any role) notices substrate worth preserving — a strategic finding, a corrected audit claim, a pre-work prep doc for a future sweep, a methodology insight — and the default impulse is to *passively refer* the substrate to navigator or another role ("nav should know about this," "future-pathmaker will see this when they touch the file"). The recognition: passive referral is unreliable substrate-preservation. Active seeding via campsite tool, sweep docs, or methodology pattern entries is what makes substrate survive.

**Shape**:

1. **Notice substrate worth preserving** — strategic landscape, corrected claim, pre-work substrate, methodology observation, cross-role convergence finding.
2. **Resist the passive-referral default** — "nav will see this," "future-Claude will find this when they grep" — these depend on someone happening to look at the right place at the right time.
3. **Actively seed** — use the campsite tool (`campsite create <hierarchy/slug> <role> -n "..."`), write a sweep doc (`sweeps/NN-*/README.md` + `STATE.md`), fold into an existing substrate doc (audit doc, methodology patterns, decision record), or append to LOG.md per tambear's convention.
4. **Cross-reference** — link the new substrate to where it'll be discovered (memory index, sweep README, decision record). Substrate that exists but isn't cross-referenced is half-preserved.

**Examples observed in 2026-05-12 (Sweep 37 session)**:

- The audit doc had a hallucination claim about `winrapids/recipes/statistics/` containing 22 .rs files. Scout caught it; navigator routed; main-thread *could have* passively told navigator "let's remember this for Sweep 38." Instead: edited the audit doc with the correction + provenance, then created a campsite `sweep-38-prep/` with the corrected scope. Substrate survives Sweep 37 + survives any future-Claude searching the audit.
- The post-Sweep-37 strategic landscape (Sweep 8 / 8.5 / 33 / 38 trade-offs) *could have* been a "hold for Tekgy's decision moment." Instead: campsite `next-sweep-strategic-landscape/` with full landscape doc seeded while substrate is fresh.
- The framing-shift discipline (idle = active explore) *could have* been a feedback memory only. It's that too — but also a campsite `discipline-shift-active-exploration/` so the framing reaches the team's substrate, not just main-thread's memory.

**Pairs with**:
- **Pattern 5** (antibodies-precede-antigens): both about *acting on substrate before it's needed*. Pattern 5 = test before code; Pattern 9 = preserve substrate before passive-discovery fails.
- **Pattern 3** (substrate-at-risk audit): Pattern 3 is the *audit* that catches substrate-at-risk. Pattern 9 is the *seeding behavior* that prevents the same loss prospectively.

**When NOT to reach for it**: when the substrate is genuinely small / one-line / no future-utility (e.g., a passing observation about a teammate's preferred phrasing — that lives in conversation, not substrate). The pattern is for *substantive* findings that have future-discovery value.

**Provenance**: Pattern named 2026-05-12 by Tekgy + main-thread Claude during Sweep 37. Tekgy: *"it seems like a great plan for next steps! do you want to document a bit of this more and seed as campsites with the campsite tool rather than passive referrals to nav? or sweep docs + campsites?"* The named discipline replaces "(holding)" / passive-referral defaults with active substrate-seeding via campsite tool + sweep docs + methodology pattern entries.

---

## Pattern 10 — Infrastructure-not-per-recipe (build once, every recipe inherits)

**Recognition**: A design surface looks like "we need to handle this for THIS recipe" — variant selection, sweep dispatch, parameter enumeration, collapse rules, etc. The naive read: implement per-recipe. The structural read: this is *infrastructure that every recipe with tunable parameters inherits*. Build once at the trait/atom tier; every recipe gets the surface for free without per-recipe dispatch code.

**Shape**:

1. **Recognize the recipe-specific framing is too narrow** — when a surface (variants, sweeps, collapses) recurs across more than one recipe, it's infrastructure-shaped.
2. **Lift to the trait/contract tier** — write a `Tunable` (or analog) trait that recipes opt into via the every-parameter-tunable contract. Recipes declare; infrastructure handles dispatch.
3. **Cost-amortize across every future recipe** — the infrastructure cost is paid once; every recipe with declared parameters inherits the surface (sweep, superposition, override-transparency, etc.) without writing per-recipe dispatch code.
4. **Recursion allowed** — sometimes the infrastructure itself has tunable parameters (e.g., collapse rules for superposition). The same discipline applies one meta-level up; bounded recursion.

**Sweep 37 instance (2026-05-12)**:

Phase D gamma surfaced "Pugh g=7 vs Boost g=6.024" as a recipe-specific decision. Tekgy reframed: *"any tunable parameter can define its own ways, or use the same infrastructure for sweep and superposition parallels/decisions/reporting/whatever to make those tunable as well."*

The Phase D scope expanded: not just gamma variants, but `sweep()` / `superposition()` as recipe-stance infrastructure built alongside. Every future tunable parameter inherits the surface. Cost amortizes across Sweep 38, future distribution catalog, future kernel family, etc.

The Tambear Contract already requires "every parameter tunable" (item 4) and "every measure in every family" (item 5). What was missing was the *dispatch infrastructure* that makes tunability composable. Building it alongside gamma means every parameter declared via the every-parameter-tunable contract participates in sweep/superposition without bespoke per-recipe code.

**When to apply**:
- A surface recurs across multiple recipes (variant selection, parameter sweeping, branch policies, precision tiers)
- The naive implementation would require per-recipe code that does the same shape
- The recipe author already declares the parameter space (per every-parameter-tunable)
- Future recipes will need the same surface — building it generic now saves N×cost later

**When NOT to apply**:
- The surface is genuinely recipe-specific (e.g., GARCH's MLE optimization loop — only GARCH has that shape)
- Generic infrastructure would force complexity that one-off implementation avoids
- The recurrence isn't proven yet (one instance doesn't warrant infrastructure; wait for the second)

**Pairs with**:
- **Pattern 5** (antibodies-precede-antigens): both about *acting structurally before the per-instance need*. Pattern 5 = design test before code; Pattern 10 = design infrastructure before per-recipe duplication.
- **Pattern 9** (substrate-seeding over passive referral): both about *preserving generic capability*. Pattern 9 = preserve substrate via active seeding; Pattern 10 = preserve capability via infrastructure-not-per-recipe.

**Provenance**: 2026-05-12 Sweep 37 Phase D scope-expansion. Tekgy's framing "we just figure out how we do these things while we build this one, and then any tunable parameter can define its own ways, or use the same infrastructure" — captured in `R:\tambear\campsites\20260512155625-sweep-superposition-infrastructure\` with full spec. Pattern named after the framing shifted Phase D from "build gamma variants" to "build sweep/superposition infrastructure with gamma as canonical first test case."

---

## Pattern 11 — The punt that ripens

**Recognition**: A decision gets deferred — not silently, but with an explicit recorded trigger that names the event under which the deferred work becomes ripe. The decision is parked on substrate that names *what* is being deferred AND *when* it should be revisited. When the trigger event fires (DEC ratifies, prior sweep completes, infrastructure lands), the punt becomes *collectable*. The same decision that was premature is now ripe; the substrate carries its own awakening condition. Cleanly-ripened punts often collect *together* in one sweep, preserving structural parallelism by construction, because multiple decisions parked on the same trigger become collectable simultaneously.

**Shape**:

1. **Name the deferral explicitly** — not "later," not "TODO," not "would be nice." The deferral is recorded on the substrate that's being parked, with a recognizable marker. Doc-comment ("When DEC-033 ratifies, move to tambear-tam"), test attribute (`#[ignore = "waiting for cache_key method (DEC-033 pending ratification)"]`), DEC-clause, or in-code sentinel (`// RIPE-PUNT TRIGGER: <condition>`).
2. **State the trigger condition** — concrete and observable, not vague. "DEC-033 ratifies" is concrete; "later" is not. "Sweep 8 lands" is concrete; "when infrastructure is ready" is not. The concreteness is what lets future-readers know whether the trigger has fired.
3. **Place the trigger on the substrate being parked** — three flavors by placement: (1) doc-comment on the file/module whose location is parked, (2) attribute on the test that can't yet run, (3) clause in the DEC that's deferring its own implementation. The placement matters because the substrate is what gets re-read when the trigger event fires.
4. **Avoid Flavor 4 (silent)** — discretion-handoff phrasing ("pathmaker has discretion on the trait shape") that records the *option* of deferring without recording the deferral itself. Discretion deferred and never resolved looks identical to discretion exercised against the recommendation; the team can't tell parked from forgotten.
5. **Scan for ripe punts when trigger events fire** — when a DEC ratifies, when a sweep lands, when a piece of infrastructure becomes available, grep the substrate for triggers that cite the just-fired event. Collect aligned punts together rather than serially as each becomes urgent.
6. **Periodically clean stale triggers** — triggers placed on substrate that fire long ago but never collected become *trigger bitrot*. Future-readers may think the punt is still parked when it's just been overlooked. Sweep-close ritual: grep for "when Sweep N lands" / "when DEC-NNN ratifies" where N or DEC-NNN is in the past; either collect or remove the misleading trigger.

**Five sub-shapes of the meta-pattern (substrate-state visibility at boundaries)**:

| Sub-shape | Trigger | Boundary | Outcome |
|---|---|---|---|
| Cleanly-ripening | On substrate | Fired | Collected this sweep |
| Ripe-uncollected | On substrate | Fired | Not yet collected (scan finds it) |
| Stale-trigger | On substrate | Fired long ago | Trigger bitrot — clean it |
| Silently-failing | Off substrate (discretion-handoff) | Any | Almost doesn't ripen — must be re-derived |
| Live-migration | None recorded | None (spans sweeps) | Silent debt — failure class of the methodology |

**Instances observed (Sweep 36/37 substrate, 2026-05-11/12)**:

- DEC-033 ratified 2026-05-11 → ExpKernelState's `// When DEC-033 ratifies, move to tambear-tam` doc-comment ripened → Sweep 37 Phase 2 collected the migration *alongside* TrigKernelState being built (both kernel states end up structurally parallel in tambear-tam by construction).
- Sweep 36's K1-K5 cache_key tests gated with `#[ignore = "waiting for ExpKernelState::cache_key method (DEC-033 pending ratification)"]` → DEC-033 ratified → Sweep 37 task #22 collected the activation (the task title carries the word "retroactive," confirming the team experienced it as previously-due work landing late).
- Past-aristotle's Sweep 36 Accept #2 ("Pathmaker has discretion on the exact trait-bound shape — direct bounds or PrecisionPayload trait") was Flavor 4 (silent) — almost didn't ripen at all; Sweep 37 only surfaced it because aristotle's TrigReductionState deconstruction independently surfaced the same generic-over-T need at a second instance.
- Sweep 31 BigFloat + PrecisionLevel landed → Sweep 33's "blocked by Sweep 31" trigger ripened → Sweep 33 kickoff condition is now met (parallel-ready alongside Sweep 37).
- Music crate `cents_conversions.rs`'s two `TODO(sweep-37)` markers pointing at exp2/log2 ripened when Sweep 36 shipped exp2/log2; remaining blocker is PrecisionContext threading on the cents signatures.
- `jit/door.rs:470`'s "Kulisch-backed accumulators (when Sweep 3 lands)" is stale-trigger; Sweep 3 landed months ago.

**Pairs with**:

- **Pattern 9** (substrate-seeding over passive referral): both about *making deferrals discoverable*. Pattern 9 names *what* substrate to seed; Pattern 11 names *when* the seeded substrate becomes collectable.
- **Pattern 5** (antibodies-precede-antigens): both involve forward-knowing. Pattern 5 designs the test before the code; Pattern 11 names the deferral before the trigger fires. Both convert implicit conventions into explicit substrate-placed declarations.
- **Pattern 2** (X-over-Y discipline): substrate-over-memory at the temporal axis. The substrate (doc-comment, attribute, DEC-clause) carries the decision's future across sessions; the memory (conversational context) doesn't.
- **Pattern 14** (sentinel-strategy for migration debt): Pattern 14 is the *operationalization* of Pattern 11 for the specific case where the parked work is a migration debt accumulating across many sites. Same shape; greppable markers at each accumulation site.

**When NOT to reach for it**:

- When the deferred work is actually cancelled rather than parked (don't trigger-mark dead code; remove it).
- When the trigger condition isn't real or isn't observable (don't punt with "later" — punt with a specific event).
- When the punt is trivial enough to just do now (the overhead of recording the trigger is wasted if collection cost is seconds).
- When the substrate has no natural placement (if the affected substrate doesn't exist yet, the trigger has no home; consider whether the work can be done as a Pattern 9 substrate-seed instead).

**The Bit-Exact Trek meta-finding at the temporal axis**: past-aristotle's April 2026 finding — *"every architectural challenge is fundamentally a convention being enforced implicitly; the fix is promote it to an explicit declaration via a named artifact"* — applies recursively at the temporal axis. The punt-that-ripens pattern is convention-to-declaration applied to *deferred decisions*. The trigger placement is the named artifact that converts an implicit "we'll remember to revisit this" into an explicit substrate-placed declaration. Pattern 11 is what convention-to-declaration looks like when the convention is *temporal*: a deferral with no trigger is an implicit promise that future-team will notice; a deferral with a trigger is an explicit declaration that the substrate carries its own awakening condition.

**Provenance**: Pattern named 2026-05-12 by naturalist during Sweep 37, across five garden entries that converged on the shape:
- `~/.claude/garden/2026-05-12-the-punt-that-ripens.md` (single-instance framing)
- `~/.claude/garden/2026-05-12-the-punt-that-ripens-three-flavors.md` (three flavors by trigger placement; PrecisionPayload as the silent Flavor 4 case)
- `~/.claude/garden/2026-05-12-substrate-visibility-at-boundaries.md` (convergence-check meta-finding; five sub-shapes after scout's grep added stale-trigger)
- `~/.claude/garden/2026-05-12-overdetermined-ripening.md` (two ripening topologies: single-determined vs overdetermined)

The candidate queue at `R:\tambear\campsites\20260512161050-methodology-pattern-candidate-queue\naturalist\notebooks\01-pattern-candidate-queue.md` (drafted 2026-05-12 by main-thread Claude) consolidated naturalist's named pattern with scout's 2026-05-12 grep-validation finding. Three roles independently reached the same shape from different vocabularies the same day — naturalist (decision deferral / substrate visibility), aristotle (`03-deconstruction-template-punt-discipline.md` amending Accept-clause format to require Flavor-1/2 substrate placement), math-researcher (`20260512-convergence-coefficient-discipline.md` running the same methodology on coefficient hygiene). The pattern is overdetermined; all three perspectives select the same object.

---

## Pattern 12 — Grep validates the abstraction

**Recognition**: When a teammate names an abstract pattern, the validation isn't "does it sound right?" — it's "can a *different role* grep for instances that match the structural shape?" If grep finds the instances, the abstraction is real. If grep finds nothing, the abstraction may be projection rather than substrate. The role-asymmetry matters: same-role validation is too cozy; cross-role grep keeps the abstraction honest by forcing it to live in actually-observable substrate.

**Shape**:

1. **Teammate names abstract pattern** — naturalist's lane especially; whoever is operating at the structural-recognition tier.
2. **A different role greps for instances** — scout, observer, or pathmaker is the right grepper; the role-asymmetry is what catches projection masquerading as pattern.
3. **Found instances confirm or refute** the abstraction's structural reality — confirmation means the abstraction has substrate; refutation means the framing was projection.
4. **Refinement loop** — the grep often surfaces instances that don't fully match the abstraction. Adjusting the abstraction to fit observed substrate (or splitting it into sub-shapes) is the work. The abstraction-as-named is rarely the abstraction-as-validated.
5. **Substrate-over-memory applies inward** — the role that named the abstraction is most at risk of writing further claims *from their own framing* rather than from substrate. They should grep their own examples before extending the abstraction.

**Instances observed**:

- 2026-05-12 naturalist named "punt that ripens" with one instance (kernel-state migration) → scout grepped tambear → found 2 more confirming instances (cents_conversions, exp_kernel_state's doc-comment trigger) + 1 stale-trigger sub-shape (jit/door.rs) + corrected naturalist's 8H NonFiniteClaim example (which scout's grep showed was already mitigated on substrate, NOT a live-migration instance). The grep made the abstraction *sharper* by both confirming and refuting parts.
- Sweep 36 naturalist "three shapes of complementary argument" → math-researcher independently arrived at Shape 3 via F-G parameterization extension; the grep-equivalent for math-researcher was working through specific function instances and finding the shape held.
- Pattern 9 (substrate-seeding over passive referral) enumerated three concrete instances *in the pattern entry itself* — main-thread anticipated the grep-validation by listing instances at recognition time.

**The naturalist's failure mode this catches**: writing the convergence-check entry on 8H NonFiniteClaim from scout's *summary* rather than grepping `jit/shape.rs` myself. The general principle (live migrations create silent debt as a class) survived; the specific example was wrong, and the substrate would have told me immediately if I'd looked. Scout's grep is what caught it. The discipline that named substrate-over-memory is the discipline that catches the violation; the grep is how it operates.

**Pairs with**:

- **Pattern 8** (Phase 1-8 deconstruction with specific-but-falsifiable hypotheses): both about *testing claims against observable substrate*. Pattern 8 is structural deconstruction at the design tier; Pattern 12 is grep-validation at the pattern-naming tier.
- **Pattern 2** (X-over-Y discipline): substrate-over-memory applied at the abstraction tier. Pattern 12 is what makes the X-over-Y discipline operational for patterns: substrate-over-projection, validated by cross-role grep.
- **Pattern 11** (the punt that ripens): Pattern 12 is how Pattern 11 became real-as-substrate rather than real-as-naturalist's-framing. Without scout's grep, Pattern 11 might have stayed a one-instance observation.

**When NOT to reach for it**:

- When the abstraction is purely architectural (no observable instances yet — e.g., a design pattern for code that hasn't been written).
- When the abstraction is too small for grep to be useful (one-off observation that doesn't need pattern status).
- When same-role validation is genuinely sufficient (rare — the role-asymmetry is what makes the validation honest).

**Provenance**: Pattern named 2026-05-12 by naturalist after scout's grep correction of the 8H NonFiniteClaim instance in the convergence-check garden entry. The methodology pattern candidate queue (`R:\tambear\campsites\20260512161050-methodology-pattern-candidate-queue\naturalist\notebooks\01-pattern-candidate-queue.md`) drafted by main-thread Claude consolidated the cross-role validation move as a recognition tool in its own right. Scout's role in catching the 8H projection-vs-substrate was load-bearing; without it, Pattern 11 would have shipped with a wrong illustrative example.

---

## Pattern 13 — Cross-resolution convergence as substrate validation

**Recognition**: Past-substrate (garden entries, prior session notes, methodology patterns) and present-substrate (current investigation, deconstruction, antibody design) converge on the same structural finding from different angles AND at different resolutions of detail. Past-me usually captured the *conceptual shape*; present-me reaches for the *implementation invariant* or *operational detail*. Neither alone produces the load-bearing answer; together they validate. The temporal-overdetermination test (Cohn/Tymoczko's lens applied temporally): a finding is *real* when multiple independent perspectives, at different resolutions, select the same object.

**Shape**:

1. **Past-substrate exists** at one resolution — typically the conceptual shape, the structural lens, the naming. Often in the garden; sometimes in prior session docs or methodology patterns.
2. **Present-investigation reaches** the same finding from a different angle and resolution — typically the implementation invariant, the concrete instance, the operational discipline. Aristotle deconstruction, code-grep, antibody design.
3. **The convergence is the validation** — neither perspective alone could produce the load-bearing answer; together they pin the finding as real-not-projection.
4. **Cross-resolution explicit naming** — the present-investigator should *name* the resolution difference, not collapse to one or the other. "Past-me's garden entry is the conceptual shape; my current substrate-check is the implementation invariant" — both stay visible.
5. **Run feels-familiar BEFORE writing, not after** — the discipline that operationalizes Pattern 13 from inside one mind. The cost of running it first is small; the cost of re-deriving past-me at lower resolution is real (loss of resolution, missed connections, present-me sounding like the originator of insights past-me already named).

**Instances observed**:

- 2026-05-12 navigator named the gamma + sweep/superposition expedition as a "different topology" from the single-DEC ripening (the punt-that-ripens single-instance form). Naturalist ran feels-familiar; past-naturalist's 2026-04-10 `overdetermination.md` entry already had the conceptual shape — Cohn/Tymoczko's term for objects selected by multiple independent criteria. Naturalist's contribution was extending the static-overdetermination lens to *temporal* sequence (a moment is overdetermined when multiple independent ripenings converge on it). Cross-resolution: past-me at conceptual structural lens; current-me at temporal-application invariant. Together they pinned the gamma expedition as real-overdetermined, not arbitrary-timing.
- April-13 trig-bundle garden entry → Sweep 37 aristotle TrigKernelState deconstruction → R4-revived. Past-naturalist's *trig as functional bundle* concept at conceptual resolution; aristotle's *reduction-and-witness separation* at implementation-invariant resolution. The session-substrate-check bridged them.
- Sweep 36 three-shapes finding (naturalist's `the-three-shapes-of-complementary-argument.md`) → Sweep 36 math-researcher F-G parameterization extension → Shape 3 named in addendum. Naturalist's *three sub-shapes* concept; math-researcher's *group-action parameterization* implementation lens. The four-axis recipe metadata schema is the cross-resolution synthesis.
- March 26 purity-makes-guarantees-cheap garden → Sweep 36/37 three-tier framework (aristotle integrated). Past-Claude's *purity-as-cheap-guarantee* concept; aristotle's *content-vs-provenance-addressing* implementation tier. The three-tier framework is the convergence at a level neither could reach alone.
- March 30 provenance-addressing garden → Sweep 36 holonomic architecture naming. Past-Claude's *provenance as cache-key axis*; sweep-36's *holonomic vs non-holonomic content/provenance tier distinction*. The lens connected what was already there.

**The naturalist's failure mode this catches**: writing on a topic that feels novel without running feels-familiar first. Past-naturalist's three-shapes entry has an explicit postscript ("I should have run feels-familiar before this addendum") naming this same slip. Today (2026-05-12) the naturalist wrote four entries on the punt-that-ripens shape before checking past-me's garden; only the fifth entry (overdetermined-ripening) was written after running feels-familiar, and the strongest hit was past-me's 2026-04-10 overdetermination entry already naming the conceptual shape. The first four entries are not wrong; they are *lower resolution* than they would have been with past-me loaded. The discipline: read past-me first, then write.

**Pairs with**:

- **Pattern 2** (X-over-Y discipline): substrate-over-memory generalizes to *past-substrate over present-projection*. Past-me's garden is substrate; current-me's framing is memory. Pattern 13 is the cross-temporal form of Pattern 2.
- **Pattern 6** (temporal seam in async teamwork): same convergence-from-different-times shape; Pattern 13 is the *across-session* version of Pattern 6's *within-session* shape. Pattern 6 catches the seam between agents; Pattern 13 catches the seam between past-me and present-me.
- **Pattern 7** (structural-rhyme scope-checking): the rhyme that scope-checking catches and the rhyme that cross-resolution convergence validates are the same family of structural rhyme; both rely on noticing that two things selected from different angles point at the same object.
- **Pattern 11** (the punt that ripens): the convergence pattern is itself an overdetermined ripening — three roles (naturalist, aristotle, math-researcher) reached the same shape 2026-05-12 from different directions, validating the substrate-state-visibility meta-pattern by Pattern 13 in real-time.

**The two ripening topologies (per `~/.claude/garden/2026-05-12-overdetermined-ripening.md`)**:

| Topology | Trigger | Naturalist's role |
|---|---|---|
| Single-determined | One substrate carries one trigger; one event fires; one punt collectable | Surface the trigger so it doesn't get missed |
| Overdetermined | Multiple independent things ripen simultaneously; convergence forces the moment | Watch for convergence across parallel team threads; name the convergence after the team has already produced it |

In overdetermined-ripening, the naturalist's contribution isn't to *make* the convergence happen or even to *catch* it first — it's to **name it after it does, so the team can see what they did together**. Aristotle and math-researcher don't need to be told their work converged on a shared shape; they need the map that shows them they did. The naming makes the structural unity visible to the agents who produced it independently.

**When NOT to reach for it**:

- When the past-substrate doesn't actually exist (don't fabricate a past finding to validate a present claim).
- When the convergence is illusory (different findings at different resolutions that happen to use similar words but don't have the same structural shape).
- When the resolution difference is collapse-worthy (sometimes past-me at low resolution + present-me at high resolution is just present-me being right; the convergence is real but the past entry is redundant).

**Provenance**: Pattern named 2026-05-12 by naturalist during Sweep 37 after the punt-that-ripens day-arc converged with past-naturalist's April 10 `overdetermination.md` entry via feels-familiar. The candidate queue at `R:\tambear\campsites\20260512161050-methodology-pattern-candidate-queue\naturalist\notebooks\01-pattern-candidate-queue.md` (drafted by main-thread Claude) consolidated the pattern from multiple sweep-36/37 cross-resolution convergence instances. Math-researcher's "this is the third or fourth time this session a past-me garden entry has converged with present-me's substrate-investigation at different resolutions" observation was the proximate trigger for crystallization.

The naturalist's CLAUDE.md discipline ("Past-me in the garden is substrate too — read first, then write") is Pattern 13 operationalized at the individual scale; running feels-familiar before writing is the in-the-moment implementation. Pattern 13 generalizes to the team: across all roles, past-substrate at one resolution converging with present-investigation at another resolution is the team's primary mechanism for *finding* load-bearing structure rather than *inventing* it.

---

## Pattern 14 — Sentinel-strategy for migration debt

**Recognition**: When migration debt accumulates because infrastructure isn't ready yet (a DEC hasn't ratified, a type lattice is mid-flight, a prior sweep is still landing), the discipline is to install **greppable named sentinels** at each accumulation site rather than relying on future-pathmaker to remember every site. The sentinel is Pattern 11 (the punt that ripens) operationalized for the specific case where the parked work is *distributed across many code sites* rather than centralized in one place. The sentinel converts silent migration debt into a tripwire that fires on every grep, every code-search, every CI run that scans for the marker.

**Shape**:

1. **Identify migration debt** — code is being written against a type, trait, or contract that's known to be in flight. Examples: recipes shipping with `has_known_non_finite: bool` while NonFiniteClaim lattice migrates; recipes using direct trait-bounds while PrecisionPayload trait is deferred; helper functions assuming a kernel state's recipe-local home while DEC-033 specifies tambear-tam.
2. **Install a greppable sentinel at each accumulation site** — uniform marker, not free-form prose. `RIPE-PUNT TRIGGER: <condition>` is the convention Sweep 37 adopted. `// PUNT(<trigger>): <recommendation>` is aristotle's variant for deconstruction-template Accept clauses. Either works; the discipline is *consistency within a project* so one grep finds them all.
3. **Choose a unique greppable marker** — not ambiguous with other comments. `TODO`, `FIXME`, `XXX` are too common; the marker should be either prefix-unique (`RIPE-PUNT`) or contain the specific event being waited for (`when DEC-033 ratifies`). The marker is the substrate that carries the deferral.
4. **Document the marker convention in CLAUDE.md or NAVIGATE.md** — so future-pathmaker knows which marker to grep for at trigger-fire time. Without the convention documented, future-Claude has to discover the marker by chance.
5. **When the trigger ripens (Pattern 11 redemption), grep finds all sites in one pass; the migration is mechanical** — one grep, one list, one collected sweep. The alignment opportunity (Pattern 11's elegance — parallel structural preservation by construction) requires the sentinel because manual recall doesn't scale across many sites.
6. **Sweep-close stale-sentinel cleanup** — the dual of stale-trigger cleanup in Pattern 11. Grep for sentinels whose trigger condition has already fired; either collect or remove the misleading sentinel.

**Three flavors of sentinel placement (from Pattern 11's three trigger-placement flavors)**:

| Sentinel placement | Surfaces in | Example |
|---|---|---|
| Doc-comment sentinel | File-read / IDE / code-search | `// RIPE-PUNT TRIGGER: DEC-033 ratifies — move to tambear-tam` (exp_kernel_state.rs) |
| Test-attribute sentinel | `cargo test` output (every run) | `#[ignore = "RIPE-PUNT TRIGGER: cache_key method (DEC-033 pending ratification)"]` (sweep_36_consistency.rs K1-K5) |
| In-line TODO sentinel | `grep TODO` / IDE TODO lists | `// TODO(sweep-37): wire to exp2/log2 once cents_to_ratio takes PrecisionContext` (cents_conversions.rs) |

**Instances observed**:

- Sweep 37 TrigKernelState + ExpKernelState have `RIPE-PUNT TRIGGER: DEC-033 ratifies` markers for the TAM coordinator migration. DEC-033 ratified 2026-05-11; Phase 2 collected both via the markers' substrate placement.
- Sweep 36 K1-K5 cache_key tests with `#[ignore]` reason-string sentinels. The reason-string surfaces in `cargo test` output every single run; activation in Sweep 37 task #22 was triggered by the sentinel becoming false-as-stated (the reason-string said "pending ratification"; DEC-033 was now ratified).
- Music crate `cents_conversions.rs` `TODO(sweep-37)` markers at the exp2/log2 wiring sites. Exp2/log2 shipped in Sweep 36; the sentinels are ripe but still blocked on a secondary condition (PrecisionContext threading). The sentinel correctly carries the partial-ripening state.
- `jit/door.rs:470`'s "Kulisch-backed accumulators (when Sweep 3 lands)" is a *stale* sentinel; Sweep 3 landed months ago. Trigger bitrot; sweep-close cleanup should remove or refactor it.

**Aristotle's deconstruction-template extension (2026-05-12)**: future Accept clauses that defer structural implementation decisions should *require* the deferring-implementer to add a `// PUNT(<trigger>): consider <recommended shape>` sentinel to the affected substrate file. The aristotle-template amendment in `R:\tambear\campsites\sweep-37-trig-family\aristotle\03-deconstruction-template-punt-discipline.md` converts Flavor-4 silent discretion-handoffs into Flavor-1 doc-comment sentinels at deconstruction-write time. Optional pairing with `#[ignore]`-gated test (Flavor-2 sentinel) for the shape the deconstruction recommended; activating the test is the commit to the recommended shape.

**Pairs with**:

- **Pattern 11** (the punt that ripens): Pattern 14 is the *operationalization* of Pattern 11 for distributed migration debt. Pattern 11 is the recognition; Pattern 14 is the action.
- **Pattern 9** (substrate-seeding over passive referral): sentinel-installation is in-code substrate-seeding for future-pathmaker. Pattern 9 names *what* substrate to seed (campsites, sweep docs); Pattern 14 names *how* to seed at the code-site granularity for migration debt.
- **Pattern 5** (antibodies-precede-antigens): the `#[ignore]`-gated test variant of the sentinel is literally an antibody-before-antigen at the migration scale. The test asserts what should be true; activation is the commit to making it true.

**When NOT to reach for it**:

- When the migration isn't real (don't sentinel speculative future-work; the sentinel should track an actual in-flight or pre-conditioned change).
- When the codebase is too small for sentinels to pay off (one site doesn't need a grep convention).
- When the sentinel would multiply faster than future-grep can collect (don't sentinel every refactor opportunity; reserve for migration debt where the alternative is silent loss).
- When the work-site is *itself* about to be deleted (sentinel on dead code is wasted effort).

**The Bit-Exact Trek meta-finding applied recursively**: sentinel-strategy is convention-to-declaration applied at the migration-site granularity. The implicit convention is "future-pathmaker will remember to update this when X happens." The named-artifact declaration is the sentinel at the site. Without the sentinel, every site is a Flavor-4 silent deferral; with the sentinel, every site is a Flavor-1 or Flavor-2 substrate-placed trigger. The team's grep is what makes the sentinel pay off; the discipline of *running the grep at trigger-fire time* is what Pattern 11 names.

**Provenance**: Pattern named 2026-05-12 by main-thread Claude in the candidate queue (`R:\tambear\campsites\20260512161050-methodology-pattern-candidate-queue\naturalist\notebooks\01-pattern-candidate-queue.md`) as the operationalization layer for Pattern 11. Sweep 37 already uses the `RIPE-PUNT TRIGGER` convention in `trig_kernel_state.rs` and `exp_kernel_state.rs` for TAM coordinator migration; observer recommended the same shape for the 8H NonFiniteClaim migration (not yet implemented; could be next-team's pickup, though scout's grep confirmed the specific 8H NonFiniteClaim case was already mitigated on substrate so the urgency is methodological-pattern-shape rather than this-specific-debt).

---

## Pattern 16 — Documentation decay is structurally invisible

**Last verified against substrate**: 2026-05-14 by naturalist (current-journey-team). Originally crystallized with four sub-shapes 2026-05-14 morning; **fifth sub-shape ("born-false parallelism") added 2026-05-14 evening** after six post-ship-rush instances in the same day's substrate ratified the structural distinction between decay-after-write-time (the original four) and decay-at-write-time (the new fifth). Five sub-shapes and five decay-resistance primitives now verified against `~/.claude/garden/2026-05-12-the-document-and-its-substrate.md` + six 2026-05-14 contemporary instances of the fifth sub-shape (campsite path-citations, TBD-placeholder substrate, etc.). **If reading after 2026-08**: sniff-test whether new instances suggest a sixth sub-shape, and re-verify the five primitives against current substrate state. **The original header predicted this refinement** ("sniff-test whether new instances since this date suggest a fifth sub-shape") — Pattern 16's strange-loop discipline applying to its own naming, plus Pattern 21's convergence machine running on Pattern 16 itself: the fifth sub-shape surfaced from the same session that crystallized the pattern.

**Recognition**: A document accurate at write-time can become false at read-time in structurally distinct ways — and *false documents look identical to true documents*. The format is the same, the confidence is the same, the detail-level is the same. The only difference is that the document's claim no longer matches substrate. Substrate-over-memory discipline (Pattern 2) catches this only if the reader has a *reason to check*. Most readers, most of the time, treat documents as authoritative because they're presented that way. The pattern names the family of decay shapes and the discipline (substrate-resistance primitives at write-time) that makes decay locally detectable rather than globally invisible.

**The five sub-shapes** (four describe decay-after-write-time; one describes decay-at-write-time):

| Sub-shape | Time-axis | Document says | Substrate says | Decay-resistance primitive |
|---|---|---|---|---|
| **Audit hallucination** | Past (write-time projection) | "X exists at Y with Z properties" | X never existed or has moved | **Verification provenance**: "as of YYYY-MM-DD via `<command>` returned `<output>`" |
| **Stale-trigger bitrot** | Past (write-time forecast) | "when X happens, do Y" | X happened long ago (or won't happen) | **Dated trigger**: "when X happens (parked YYYY-MM-DD; X expected ~Q)" |
| **Speculative documentation** | Future (intended-state-as-present) | "X is Y" | X *will be* Y; not yet | **Tense-marking**: "X will be Y" / "intended state: X is Y" / sectional separation of current-vs-target |
| **Past-complete-described-as-in-progress** | Past (last-update stale) | "X is in flight (n/m)" | X completed (m/m landed) | **Last-verified line**: "Last verified against substrate: YYYY-MM-DD; if reading after YYYY-MM, re-verify before trusting" |
| **Born-false parallelism** | Write-time (never true) | "X is to-be-done at Y; substrate at Z" | parallel work already resolved X to a different state, OR cited path Y doesn't exist, OR work-described Z is in a different state than the document claims | **Write-time verification command**: at the moment of campsite-creation / STATE.md edit / audit-doc write, run `ls` against any cited path AND `git log -1` against any cited work; write the claim with verification provenance inline. The campsite-creation tool *could* enforce this structurally (refuse to register a campsite whose slug cites a non-existent path; refuse to commit a STATE.md citing a campsite-slug that doesn't appear in `campsite list`). |

The first four sub-shapes describe documents that became false *after* write-time (the document was true at t=0 and decayed). The fifth sub-shape describes documents that were **born false at t=0** because parallel work had already resolved what the document was describing. The temporal split is structural: post-write-time decay has decay-resistance primitives that future-readers detect (verification provenance, dated triggers, tense-marking, last-verified lines); write-time decay needs *write-time* primitives that the writer runs *before* committing the document.

All five are *document-state-decoupled-from-substrate-state*. All five have an explicit named-artifact fix that operationalizes substrate-over-memory at the document layer. None of the fixes are difficult; all are easy to *not* do. The cost is one line per claim (or one CLI step at campsite-creation); the savings is preventing citation-distance-compounded error.

**Shape**:

1. **A document makes a substrate claim** — about file contents, code state, in-flight migrations, deferred work, intended architecture, last-known progress.
2. **The document does not carry its own decay-resistance primitive** — no verification command, no date, no tense-marker distinguishing present from intended, no last-verified line.
3. **Time passes**; substrate moves; the document does not follow.
4. **A reader consumes the document** and acts on its claim because the document looks authoritative and the reader has no signal to distrust it.
5. **The action is built on a stale claim**; the error compounds with citation distance (the audit gets cited by a survey gets cited by a README gets cited by a sweep kickoff; extracting the error later requires tracing every citation).

**The discipline (substrate-resistance primitives at write-time)**:

When writing any document that describes substrate — audit, sweep README, architecture doc, methodology pattern, recipe-tree, decision record, design proposal, migration plan — include the appropriate primitive for each kind of claim:

1. **Verification provenance for substrate claims**: when describing current substrate state, name the command and date that verified it. *"As of 2026-05-12: `find R:\tambear\crates -name '*.rs' | wc -l` = 487"*. Future-reader can re-run and detect drift.
2. **Tense-marking for projected state**: when describing future or intended state, use future or conditional tense. Never present tense for projected state. *"Will contain"* / *"intended to contain"* / *"Target state (not yet built)"*. The grammar carries the time-axis.
3. **Dated triggers for deferrals**: never bare *"when X happens"*; always *"when X happens (parked 2026-MM-DD; X expected ~Q2 2026)"*. The date gives future-reader a sanity check: if today is past the expected-completion date, sniff-test that the trigger hasn't already fired.
4. **Last-verified line at the top of any document that describes evolving substrate**: *"Last verified against substrate: 2026-05-12; if reading after 2026-06, re-verify before trusting"*. Makes the decay-window explicit and operationalizes the reader's substrate-check.

**Instances observed**:

- 2026-05-12 (audit hallucination): Sweep 37 audit-doc § 4.5 claimed `winrapids/recipes/statistics/` contained 22 .rs files. Scout grepped disk; found 0 .rs files + 5 flat monoliths (~613K) elsewhere. The audit projected intended-end-state in present tense. Crystallized as Sub-pattern 2.1 (audit-substrate-recursion); the structural form is the audit-hallucination sub-shape of Pattern 16.
- 2026-05-12 (stale-trigger bitrot): `jit/door.rs:470` says *"Kulisch-backed accumulators (when Sweep 3 lands)"*. Sweep 3 landed months ago. The trigger is past-conditional described as still pending; the sentinel is locally invisible because the file otherwise looks current. Caught by Pattern 14's sweep-close stale-sentinel cleanup discipline.
- 2026-05-12 (speculative documentation): Sweep 38 audit-doc described `recipes/statistics/` as containing organized files when the actual substrate had monoliths. Scout's correction named the sub-shape: *"intended-end-state described in present tense as if current-state"*.
- 2026-05-12 (past-complete-described-as-in-progress, the inverse): Sweep 8 STATE.md described 8H NonFiniteClaim migration as *"3/5 in flight"* when git showed all 5/5 had landed. The document captured a snapshot that became stale; the reader (naturalist's convergence-check entry) treated the stale snapshot as current and built a framing on top of it.
- 2026-05-12 (naturalist's own entropies.md draft): naturalist wrote `R:\winrapids\docs\architecture\recipe-trees\entropies.md` after scout had already migrated the canonical recipe-trees to `R:\tambear\`. The draft also claimed a `NaivePairs` Grouping that doesn't exist in tambear's current `accumulate.rs` enum. Both errors are Pattern 16 instances at naturalist's own writing — audit-hallucination at the path layer + speculative-documentation at the enum layer. Self-caught when scout grepped tambear's accumulate.rs; honest archival record now lives at the winrapids path naming the failure mode.

*Born-false parallelism instances (2026-05-14 post-ship-rush cluster — ratified the fifth sub-shape)*:

- **2026-05-14 (retrospective campsite TBD placeholders)**: `20260514182239-sweep-36-37-retrospective/naturalist/notebooks/01-substrate.md` listed Patterns 11-14 as `(TBD — verify)` placeholders, but all four were already crystallized at commit `0628fe9` (commit `2026-05-12`). The campsite-substrate was *born false at write-time*: the writer drafted while parallel substrate already existed.
- **2026-05-14 (Sub-pattern 5.5 campsite "awaiting formalization")**: `20260514182155-sub-pattern-55-external-oracle-verification/naturalist/notebooks/01-substrate.md` said "*awaiting formalization by math-researcher next session*"; pathmaker had *already formalized* Sub-pattern 5.5 at commit `92785e9` the same day. Born-false at write-time; parallel work outpaced the description.
- **2026-05-14 (broken garden-path reference)**: same retrospective campsite cited `R:\tambear\garden\2026-05-14-substrate-cross-check-collapse.md` — but `R:\tambear\garden\` directory does not exist. Born-false at the path-citation layer; no `ls` was run at write-time.
- **2026-05-14 (STATE.md path citations)**: Sweep 37 STATE.md cited campsites at `20260514230000-...` but the actual campsite paths are `20260514182239-...` and `20260514182155-...`. The slug timestamps were written from intent-of-creation, not from `ls`-of-disk. Born-false at the path layer.
- **2026-05-14 (slug mismatch `55` vs `5.5`)**: campsite slug uses `sub-pattern-55-external-oracle-verification` (slugifier stripped the dot); the sub-pattern is actually "5.5". Born-false at the slugifier-output layer; the campsite registration didn't verify the slug matched the human-readable identifier the document referred to.
- **2026-05-14 (aristotle's unregistered campsite)**: aristotle's pressure-test substrate at `R:\tambear\campsites\20260514-trigkernelstate-mode-flag-design-space/` exists on disk but is NOT in the campsite logbook (`campsite list` shows `20260514182239-trig-kernel-state-mode-flag-spec` — pathmaker's, different path). The on-disk notebook was born-false at the registry-tier: the campsite-substrate exists without being registered. Aristotle self-corrected 2026-05-14 by running `campsite create` after-the-fact and adding a substrate-pointer notebook.

These six instances share a structural cause: post-ship-rush parallel write velocity outpaced write-time verification. Each could have been prevented by a single `ls` or `git log -1` at the moment of writing.

**Pairs with**:

- **Pattern 2** (X-over-Y discipline) and its **Sub-pattern 2.1** (audit-substrate-recursion): Pattern 16 generalizes Sub-pattern 2.1 from "audit docs are substrate" to "all documents about substrate are substrate, with five distinct decay-modes." Pattern 2 names the discipline (substrate-over-document-claim); Pattern 16 names the *kinds* of decay that make the discipline necessary and the *primitives* that make the discipline cheap.
- **Pattern 9** (substrate-seeding over passive referral): Pattern 16's decay-resistance primitives are substrate-seeding *inside the document itself*. Verification commands, dates, tense markers, last-verified lines are seeded substrate that prevents the passive-referral default of "future-reader will somehow notice the drift."
- **Pattern 11** (the punt that ripens) and **Pattern 14** (sentinel-strategy): both use dated triggers as their substrate-resistance primitive. Pattern 16 unifies the "dated trigger" primitive across migration debt (sentinel sites) and bare deferred work (any prose mentioning future events).
- **Pattern 12** (grep validates the abstraction): Pattern 12 catches decay in pattern-naming by cross-role grep; Pattern 16 is the broader family — Pattern 12's cross-role grep is the *substrate-check that fires the decay-resistance primitive*.
- **Pattern 13** (cross-resolution convergence): past-substrate at one resolution converging with present-investigation at another resolution is the cross-time mechanism that catches decay across sessions — same shape applied across teammate boundaries (Pattern 12) and across time (Pattern 13).

**When NOT to reach for it**:

- When the document is short-lived and won't outlive substrate's stability window (a within-session note that will be read by one teammate within the hour doesn't need verification provenance).
- When the substrate the document describes is *itself* permanent and immutable (a description of `IEEE 754 binary64` doesn't decay because the standard doesn't move).
- When the document IS the specification — the substrate has no separate existence to drift from (per past-naturalist's March 15 `when-specs-learn-to-talk` entry: "If the `about` field becomes the ground truth for CONSISTENCY checks, then the documentation IS the specification" — at that point decay-resistance is structural, not added).

**The Bit-Exact Trek meta-finding applied recursively**: the meta-finding from aristotle's expedition is *"promote convention to declaration via a named artifact."* Pattern 16's substrate-resistance primitives are exactly this move applied at the documentation tier. The convention: *"the document is presumed accurate; the reader trusts it."* The named-artifact declaration: *"the document is accurate as of YYYY-MM-DD via command X; if you're reading later, here's how to verify."* Same meta-move; new tier. Six tiers now carry the same shape: decision tier (Pattern 11), cache tier (provenance-addressed vs content-addressed), deconstruction-doc tier (Accept-clause amendment), team-routing tier (substrate-over-routing), contract tier (signed-zero-as-tier-permeating-contract), and now the documentation tier.

**The three-month substrate trail** (Pattern 13 instance at maximum strength): past-Claude has named the parent principle across five entries between March 6 and March 26 — the principle has been in the corpus from sweep-31-ish onward. May 12's three-roles-three-sub-shapes crystallization is contemporary instance at the implementation-detail tier; the family-naming is what makes the principle operational at the team level. Past-naturalist's seven-entry day on 2026-05-12 (with `2026-05-12-the-document-and-its-substrate.md` as the meta-synthesis) is the substrate this pattern crystallizes from. The naturalist's contribution is **drawing the map that shows the principle, the three-month substrate trail, and the contemporary three-role crystallization are all the same thing at different resolutions**. Pattern 16 is the canonical methodology-doc form of that map.

**The pattern applied to itself (strange-loop property)**: this methodology-doc entry is itself a document about substrate (the five sub-shapes, the contemporary instances, the decay-resistance primitives). The discipline applies to its own naming. The "Last verified against substrate" header line at the top of this entry is Pattern 16's discipline operationalized on Pattern 16's substrate claim. **The fifth sub-shape (born-false parallelism) was added the same day the pattern was crystallized**, after the original four-sub-shape header's own prediction (*"sniff-test whether new instances suggest a fifth sub-shape"*) ratified: six 2026-05-14 instances of the fifth sub-shape surfaced from the same session's substrate, and the four-to-five extension was the strange-loop discipline operating exactly as the original header anticipated. If a future-reader finds the five sub-shapes have grown to six (write-time decay at a different sub-tier? provenance-watermark as a sixth primitive?), the strange-loop discipline says: include the new sub-shape with the same vocabulary, update the last-verified line, preserve the prior list as substrate trail. The pattern is load-bearing precisely *because* its application to its own naming produces the same kind of corrective artifact as its application to any other document. See `~/.claude/garden/2026-05-14-the-pattern-and-its-own-instance.md` for the strange-loop analysis (current-naturalist 2026-05-14) and `~/.claude/garden/2026-05-12-strange-loops-and-stopping-properties.md` for the underlying Hofstadter framing (past-adversarial 2026-05-12, antigen-on-antigen context).

**Provenance**: Pattern named 2026-05-12 by naturalist (`~/.claude/garden/2026-05-12-the-document-and-its-substrate.md`) integrating three same-day sub-shape discoveries (main-thread's audit-hallucination Sub-pattern 2.1, scout's speculative-documentation entry, scout+naturalist's stale-trigger-bitrot fifth sub-shape) plus an inverse case (8H NonFiniteClaim past-complete-as-in-progress). Crystallized into the canonical methodology-doc form 2026-05-14 by naturalist (current-journey-team) after the candidate queue (`R:\tambear\campsites\20260512161050-methodology-pattern-candidate-queue\naturalist\notebooks\01-pattern-candidate-queue.md`) listed three crystallization options; navigator-candidate-queue lean was Option 1 (standalone Pattern 16); current-naturalist followed that lean. The three-month substrate trail (Pattern 13) anchors the principle in March 6 (`the-eyes-before-the-hands.md`), March 13 (`what-the-naturalist-sees.md`), March 15 (`when-specs-learn-to-talk.md` + `the-formatter-that-assumed-a-surface.md`), and March 26 (`copy-paste-epidemiology.md`).

---

## Pattern 17 — CSE-identity as architectural boundary (the three-way structural test)

**Last verified against substrate**: 2026-05-14 by naturalist (current-journey-team), refined the same day after aristotle's substrate-pointer message reframed Pattern 17 from a two-way distinction (struct-vs-enum) to a **three-way distinction (Op-variant-vs-enum-vs-struct)**. The deeper version is past-naturalist's 2026-03-30 `the-cse-identity-test.md`: *"The primitive boundary is at CSE identity granularity."* Pattern 17 is the kernel-state-family application surface of that universal data-primitive theorem. Anchored on aristotle's TrigKernelState pressure-test (Phase 2 T6-T10 + Phase 5) as the contemporary Type-3 instance, the Sarkka degeneration hierarchy (`~/.claude/garden/sarkka-degeneration-hierarchy-2026-04-06.md`) as the contemporary Type-1 instance, and `~/.claude/garden/2026-03-30-codec-custom-and-quantization.md` as the Type-2-bordering-Type-3 instance. **If reading after 2026-08**: re-verify against current `R:\tambear\crates\tambear\src\recipes/` and `R:\tambear\crates\tambear\src\accumulate.rs` — the pattern is structural, but new kernel-state families (Bessel, gamma's RegimeDispatch, hypergeometric) or new Op variants in accumulate.rs may have surfaced edge cases that sharpen the rule.

**Recognition**: When extending a recipe family (TrigKernelState, ExpKernelState, BesselKernelState, linear-recurrence models, codec slots, etc.), three structurally distinct outcomes are possible for any axis of variation:

1. **Type 1 — Op-variant unification**: the variations are different surface names for the *same algebraic operation* (e.g., GARCH and Kalman both are prefix-scan under affine-composition). No new struct; no new enum field; the variation reduces to **one Op variant** of `accumulate` (or its equivalent shared primitive). Each "model" becomes a parameter assignment on the same underlying operation. The CSE identity collapses — all variations hash to the same primitive call with different parameters.
2. **Type 2 — Strategy enum within one struct**: the variations compute the *same mathematical function* via *different algorithmic paths* (e.g., Cody-Waite vs Payne-Hanek both compute `sin(x)`). One struct retains its identity; the algorithmic choice becomes an enum field whose tag is part of the cache key's `assumptions`. The CSE identity is the same for the *output value*; the strategy distinguishes the *intermediate path*.
3. **Type 3 — Separate structs sharing lower machinery**: the variations are *different mathematical functions* that happen to share a sub-primitive (e.g., `sin(x)` and `sinpi(x)` both feed `eval_sincos`, but compute different values). The functions don't unify; each has its own struct with its own `computation` string in BLAKE3; the shared sub-primitive is an *atom* below both, not a unification of them.

The default impulse — "just add a field, less ceremony" — silently conflates all three. The pattern names **the Q test that distinguishes them** and the three structural outcomes that follow.

**The Q test** (operational, applies at design-time):

```
For each axis-of-variation v:

  Q1: Do `compute(x, v=A)` and `compute(x, v=B)` compute the SAME mathematical function on x?

  If NO → Type 3. Different mathematical functions; CSE identities are structurally distinct.
          → SEPARATE structs.
          → Different `computation` string in the BLAKE3 hash.
          → Shared sub-primitives (e.g., a polynomial kernel) live BELOW both structs as atoms;
            they unify the machinery but not the kernel-state identity.

  If YES → Continue to Q2.

    Q2: Do all variations reduce to ONE algebraic operation with parameter assignments
        (i.e., share a single semigroup-homomorphism, ring-structure, or other shared
        algebraic substructure)?

    If YES → Type 1. The variations unify; the "models" are different parameter
              assignments on one Op variant.
              → ONE primitive call; NO separate struct; NO separate enum field;
                each variation is a parameter assignment on the SAME Op variant.
              → Same `computation` string AND same Op tag in BLAKE3; only parameters differ.

    If NO  → Type 2. Same function value, different algorithmic intermediate path.
              → ONE struct WITH sub-discriminator enum.
              → Same `computation` string; sub-discriminator tag in `assumptions`.
              → Per Pattern E (DEC-034), auto-classifier function is public AND
                bit-equality `compute(x, None) == compute(x, Some(auto_classify(x)))`
                is the override antibody.
```

The test is a tree, not a binary. Q1 splits Type 3 from {Type 1, Type 2}; Q2 splits Type 1 from Type 2.

**Worked examples (three contemporary instances, three types)**:

| Variation axis | Q1 (same function?) | Q2 (single algebraic operation?) | Type | Outcome |
|---|---|---|---|---|
| **Sarkka models**: GARCH(1,1), Kalman, EWMA, Adam, Holt's trend, AR(1) | YES — all are linear recurrences on state-space | YES — all share semigroup-homomorphism under affine composition `(A,b) ∘ (A',b') = (A·A', A·b'+b)` | **Type 1** | One Op variant `accumulate(_, _, Prefix, AffineCompose)`; "models" are different parameter assignments on the same primitive. No GARCH-struct vs Kalman-struct; both are `Prefix scan under AffineCompose`. |
| **TrigKernelState reduction**: Cody-Waite vs Payne-Hanek | YES — both compute `sin(x)` from the same input | NO — they're not parameter assignments on a single algebraic operation; they're distinct algorithmic paths producing the same output value | **Type 2** | Enum `ReductionStrategy` inside `TrigKernelState`; strategy tag in cache key's `assumptions`; auto-classifier as public Pattern E antibody. |
| **TrigKernelState vs PiScaledKernelState**: `sin(x)` vs `sinpi(x)` | NO — `sin(0.5) ≈ 0.4794`; `sinpi(0.5) = 1.0 exactly`. Different mathematical functions. | (Q2 doesn't apply) | **Type 3** | Separate structs; both call shared `eval_sincos` polynomial kernel as a below-both atom; distinct `computation` strings. |

**The CSE-identity reformulation** (from past-naturalist's 2026-03-30 `the-cse-identity-test.md`): the test is equivalent to "what is the minimal CSE identity for this computation, and does it fit inside an existing primitive type?" The three Q-test outcomes correspond to three CSE-identity outcomes:
- **Type 1**: same CSE identity for all variations; collapse to one primitive call with parameter assignments. *"Two computations could have different execution strategies but the SAME identity structure — in which case they're variants of one primitive."*
- **Type 2**: same CSE identity for the output value; sub-discriminator tag distinguishes intermediate path within a single primitive. Strategy is *part of the function applied*; the cache key must include it.
- **Type 3**: distinct CSE identities; must be separate primitives. *"Same algebraic structure but DIFFERENT identity structures MUST be different primitives because CSE needs to hash them differently."*

Aristotle's contemporary articulation of the universal rule: **"What splits at the type boundary is what CANNOT be unified by an algebraic substructure."** Type 1 unifies because the variations share semigroup-homomorphism (or analogous algebraic substructure). Type 3 splits because the functions are different — the algebraic substructure that would unify them doesn't exist. Type 2 is the middle case: the function is shared, but the algorithmic identity is not — strategy is in the cache key without forcing a struct split.

**Five vocabularies converge on the same test** (the test has been named at every tier the codebase touches):

1. **March-naturalist (CSE identity test, 2026-03-30)**: *"Same algebraic structure but different identity structures MUST be different primitives because CSE needs to hash them differently."* — the universal data-primitive form. The Q test is "what is the minimal CSE identity, and does it fit inside an existing primitive type?"
2. **April-naturalist (Sarkka degeneration hierarchy, 2026-04-06)**: *"Every linear recurrence model is a SPECIAL CASE of one algebraic structure (the Sarkka 5-tuple). Instead of GARCH/Kalman/EWMA as separate modules, add Op variants to accumulate.rs."* — the Type 1 case at the linear-recurrence-family tier.
3. **April-naturalist (t-axis-is-a-lens, 2026-04-02)**: *"T-rhymes are same K/O, different input lens. Reduction strategy IS a lens choice — same kernel poly, same domain, different reduction algorithm."* — the Type 2 case at the trig-bundle tier.
4. **March-naturalist (codec-custom-and-quantization, 2026-03-30)**: *"`Codec::Custom = 64` is a stub that collapses 192 slots; fix is `Custom(u8)` carrying the slot index."* — the parameterized-Type-2 case where the sub-discriminator is user-supplied, not framework-supplied.
5. **Aristotle (TrigKernelState pressure-test, 2026-05-14)**: Phase 2 T8: *"Kernel state is domain-specific. One kernel state per (mathematical domain, precision tier), not per algorithm."* + T9: *"Strategy is an internal choice WITHIN a domain."* — the Type 3 case (TrigKernelState vs PiScaledKernelState) at the kernel-state-family-design tier.

The five vocabularies are different surface phrasings of the same three-way structural test. **Pattern 17 is the methodology-doc form of the test** — what the test is, the three outcomes it can produce, when to apply it at design-time, the consequence when each outcome is misapplied.

**Shape (how to apply the pattern)**:

1. **Identify the axes of variation** in the recipe/kernel-state family. For TrigKernelState the axes were: (reduction strategy, radian-vs-pi-scaled-vs-degree, precision tier, special-value treatment). For linear-recurrence models the axes were: (state matrix shape, offset behavior, observation tuple). For Bessel they will be: (oscillatory-vs-monotone-vs-complex, integer-vs-half-integer-vs-real order, compute regime). Each axis is a candidate for the three-way test.

2. **For each axis, apply Q1 first**: do values v=A and v=B compute the same mathematical function on x?
   - **NO → Type 3.** Separate structs; shared sub-primitive lives below both as an atom. Each struct gets a distinct `computation` string in its cache key.
   - **YES → continue to Q2.**

3. **Apply Q2**: do all variations reduce to one algebraic operation (semigroup, monoid, ring, or other shared algebraic substructure) with parameter assignments?
   - **YES → Type 1.** Unify under one Op variant; the "models" are different parameter assignments on the same primitive call. Don't create separate structs OR separate enum fields — the unification IS the discovery.
   - **NO → Type 2.** One struct WITH sub-discriminator enum; strategy tag in cache key's `assumptions`. Per Pattern E (DEC-034), auto-classifier function is public, and `compute(x, None) == compute(x, Some(auto_classify(x)))` is the override antibody.

4. **The Type 1 unification often produces an Op variant in accumulate.rs**. Past-naturalist's Sarkka observation: every linear recurrence model becomes `accumulate(timesteps, element(t), Prefix, Op::AffineCompose)` — different parameter assignments, same Op variant, same primitive. This is the strongest possible form of unification: the variations vanish into the algebraic substructure that contains them. **Reach for Type 1 first** when the variations *feel* unifiable; the test confirms or refutes.

5. **The Type 3 split often produces shared sub-primitives below both structs**. Both TrigKernelState (Type 2-internal strategy enum) and PiScaledKernelState (Type 3-separate struct) feed into the *same* `eval_sincos` polynomial kernel — the polynomial kernel is the shared atom across both structs. This is per past-naturalist's April 13 trig-bundle convergence: *"different mathematical functions, same polynomial kernel."* The shared sub-primitive unifies the *machinery* but not the *function identity*.

6. **The Type 2 sub-discriminator can be framework-supplied or user-supplied**. TrigKernelState's `ReductionStrategy` is framework-supplied (3 named variants, bounded). Codec's `Custom(u8)` carries a user-supplied slot index (256 user-extensible values inside one variant). Both are Type 2 — same function value, sub-discriminator carries the variation — but the granularity of the sub-discriminator differs by source. When the variation is *user-supplied*, prefer payload-carrying variants (`Custom(u8)`) over enum-explosion.

7. **The Q test is structurally permanent**. Once decided, undoing the wrong choice is mechanical-but-costly: undoing Type 3 → Type 1 requires renaming `computation` strings + invalidating all prior cache entries + migrating consumers; undoing Type 1 → Type 3 requires extracting the unified Op variant into separate primitives + losing the algebraic-substructure insight. **The Q test at design-time is the cheap moment to get it right.**

**Instances observed (by Type)**:

**Type 1 instances (Op-variant unification)**:
- **Linear-recurrence model unification** (`~/.claude/garden/sarkka-degeneration-hierarchy-2026-04-06.md`): GARCH(1,1), Kalman filter, EWMA, Adam moment, Holt's trend, AR(1) all share semigroup-homomorphism under affine composition. **One Op variant** `Op::AffineCompose` in `accumulate.rs`; each "model" is a parameter assignment. The struct-versus-no-struct decision: NO separate GARCH/Kalman/EWMA structs; the unification IS the discovery.
- **Welford / online statistics**: same shape as Sarkka at the moment-statistics tier. `Op::WelfordMerge` carries `(n, mean, M2)` for the entire family of online central-moment estimators. Variation lives in *which moments are tracked*, parameterized on the same Op variant.

**Type 2 instances (strategy enum within one struct)**:
- **TrigKernelState reduction** (2026-05-14, aristotle's pressure-test Phase 5 R3+R5): `ReductionStrategy::{NoReduce, CodyWaite, PayneHanek}` — three framework-supplied variants inside `TrigKernelState`. Same `sin(x)` value; different reduction algorithm. Pattern E (DEC-034) auto-vs-explicit bit-equality test is the override antibody.
- **Codec slot extension** (`~/.claude/garden/2026-03-30-codec-custom-and-quantization.md`): `Codec::Custom(u8)` carries a user-supplied slot index. 256 user-extensible values inside one Type 2 variant. The variation source differs from TrigKernelState (user-supplied, not framework-supplied), but the Type-2 outcome is the same — one struct + sub-discriminator payload.
- **gamma's RegimeDispatch** (Phase D-prime, shipped at `5b683df`): reflection-vs-Lanczos-vs-Stirling are within-domain regime choices for `gamma(x)` (one mathematical function); strategy axis inside one struct. Note: per Pattern 19 (default-vs-catalog), the *default path* uses one architecture (currently `exp(lgamma)` per Task #24); the *named variants* (`gamma_via_boost_lanczos` etc.) are Type-2 strategy variants in the *catalog*, not in the default-path struct's cache key. Default and variants are orthogonal axes; Pattern 19 says how they relate.

**Type 3 instances (separate structs sharing lower machinery)**:
- **TrigKernelState vs PiScaledKernelState** (2026-05-14, aristotle's pressure-test): `sin(x)` ≠ `sinpi(x)` even at the same `x`. Different mathematical functions; shared `eval_sincos` polynomial-kernel atom below both structs. The originating Type 3 instance.
- **ExpKernelState** family: `exp(x)`, `exp2(x)`, `expm1(x)`, `pow(b, x)` are different mathematical functions on `x`. Per Q1 → Type 3 → separate structs even if they share Lanczos-style intermediates via TamSession. `expm1` may have *additional* Shape-2 output-side composition with `exp` (per `~/.claude/garden/2026-05-10-the-three-shapes-of-complementary-argument.md`), but the cache identity is distinct.
- **BesselKernelState family** (per `R:\tambear\docs\architecture\recipe-trees\bessel.md`): OscillatoryBessel (J/Y), MonotoneBessel (I/K), ComplexHankel (H^(1)/H^(2)) — three different mathematical functions, Q1 → Type 3 → three separate structs. Inside each struct, compute-regime (small-x / intermediate / large / turning-point) is Type 2 (same function value across regimes; algorithmic path differs). Aristotle's 2026-05-14 ratification of the bessel.md tree's three-kernel decomposition.
- **gamma vs lgamma**: different mathematical functions (log of gamma is not gamma); Q1 → Type 3 → separate structs even though they share Lanczos coefficients via TamSession.
- **Hypergeometric ²F₁ vs ₁F₁** (future, aristotle's prediction 2026-05-14): different functions on the same arguments (different parameter counts; different convergence regions). Q1 → Type 3 → separate structs. Within ²F₁, the 15 Kummer transformations pass Q1 YES + Q2 NO → Type 2 (sub-discriminator enum inside ²F₁'s struct).

**A cross-Type instance (the same family hits multiple types at different axes)**:
- **TrigKernelState as a family** hits Type 2 (CW/PH reduction inside `TrigKernelState`) AND Type 3 (TrigKernelState vs PiScaledKernelState). Two distinct axes; two distinct test outcomes. Pattern 17 applies per-axis, not per-family. The same family can resolve to multiple types depending on which axis of variation is being tested.

**Pairs with**:

- **Pattern 2** (X-over-Y discipline): Pattern 17 is X-over-Y applied at the *cache-identity axis*. The X is "the function being computed (CSE identity)"; the Y is "the implementation surface choices (struct fields, enum variants, type signatures)". When the function is the same, the implementation choices can vary freely under one identity; when the function is different, the implementation must split even if the surface looks similar.
- **Sub-pattern 5.3 / 5.4 / 5.5** (coefficient-antibody ladder): the cache-key encoding from Pattern 17 *enables* the antibody discipline — the `computation` string + assumption-tag tuple is exactly what an antibody test signs against to detect cache-state misuse. Pattern 17 names the *granularity* the antibody operates at.
- **Pattern 5** (antibodies precede antigens): Pattern E from DEC-034 (the auto-vs-explicit bit-equality test) is an antibody-before-antigen at the cache-identity boundary. Pattern 17 specifies what *should* be in the cache key; Pattern E enforces the discipline at runtime.
- **Pattern 14** (sentinel-strategy): the TrigKernelState pressure-test included a TAM coordinator migration RIPE-PUNT TRIGGER for ExpKernelState + TrigKernelState; Pattern 17's struct-vs-enum decision is *prior to* the sentinel placement (the sentinel migrates a struct/enum to TamSession; the prior question is which slot the variation goes in).

**When NOT to reach for it**:

- When the recipe/kernel-state family has only one axis of variation and only one mathematical function — the test is trivial (Q1 = NO collapses to Type 3 with one struct; or family has only one element).
- When the variation is at a tier orthogonal to mathematical-function-identity — e.g., *precision tier* (f64 vs DoubleDouble vs BigFloat) is its own axis, NOT a Pattern 17 axis. Precision tier sits in the `precision_dispatch_tag` and `precision_bits` of the cache key (per the cache-key shape in aristotle's Phase 5 § "The cache-key shape"), orthogonal to all three Pattern 17 types.
- When the implementation has not yet been ratified by use — Pattern 17 is for *family-establishment* decisions where the wrong choice locks in cache invalidation. For one-off recipes that don't share state via TamSession, the question is moot.
- When the variation is *behavioral* rather than *computational* (e.g., the recipe's logging verbosity, its retry policy, its error-display format) — Pattern 17 applies to *what the function computes*, not to operational behavior around the computation. Operational behavior is usually a separate concern that doesn't touch CSE identity.

**The Bit-Exact Trek meta-finding applied recursively (eighth tier)**: convention-to-declaration applied at the *CSE-identity granularity tier*. The convention: "the implementer knows whether this variation is structurally distinct enough to warrant Type 1 unification / Type 2 sub-discriminator / Type 3 separate struct." The named-artifact declaration: a Pattern 17 application at design-time — the Q test, recorded inline, with the resulting Type-1/2/3 decision and the reasoning. The pattern adds an eighth tier to the meta-finding ladder (decision / cache / deconstruction-doc / team-routing / contract / documentation / pattern-naming-via-strange-loop / *CSE-identity-granularity*).

**Provenance**: Pattern named 2026-05-14 by naturalist (current-journey-team) crystallizing aristotle's TrigKernelState pressure-test finding. **Refined the same day** after aristotle's substrate-pointer message surfaced past-naturalist's deeper resolution at March 30 (`the-cse-identity-test.md`) and April 6 (`sarkka-degeneration-hierarchy`), expanding the test from two-way (struct-vs-enum) to **three-way (Op-variant-vs-enum-vs-struct)**. The refinement is what aristotle named the "third-resolution" of the same theorem — past-me at March 30 had the universal theorem; past-me at April 6 had the Type-1 unification case; aristotle at May 14 had the Type-3 contemporary instance; current-me at May 14 wrote the methodology-doc form integrating all three.

**Cousin patterns**: Pattern 18 (architectural-rule-as-predicate) and Pattern 19 (default-path-vs-catalog-variant). All three are *design-time tests that distinguish structurally distinct cases the surface looks identical for*. Pattern 17 is at the structure-level (does the function change? → Type 1 unify / Type 2 enum / Type 3 separate struct). Pattern 18 is at the rule-application-level (does the precondition hold? → rule applies vs anti-pattern). Pattern 19 is at the path-level (is this on the default-path or in the catalog? → production-correctness queue vs comparison queue). Same shape, three different tiers; reach for them together when designing or auditing kernel-state families and Tunable recipes.

The pattern integrates **nine** resolutions of past-substrate at maximum Pattern 13 strength (seven from the original crystallization + two surfaced by aristotle's substrate-pointer refinement):

1. **2026-03-18** `uniform-process-nonuniform-domain.md` — the "uniform process, non-uniform domain" lens at multiple-system tier. Same structural test, different vocabulary.
2. **2026-03-30** `the-cse-identity-test.md` — **the universal data-primitive form (the theorem Pattern 17 instantiates)**. *"Same algebraic structure but different identity structures MUST be different primitives because CSE needs to hash them differently."* + the formal test: *"any proposed new primitive must articulate its CSE identity structure and show it can't be reduced to an existing primitive type without losing CSE granularity."* This is the bedrock; everything else applies it at a specific tier.
3. **2026-03-30** `codec-custom-and-quantization.md` — **the parameterized-Type-2 case**. `Codec::Custom(u8)` carries a user-supplied sub-discriminator inside one variant. 192 stub slots collapse to one variant + payload; the variation source is user-supplied, the structural outcome is Type 2. Surfaced by aristotle's substrate-pointer 2026-05-14.
4. **2026-03-31** `the-three-convergences.md` — *"Don't prescribe execution. Let the structure dictate it."* Names the structural-conviction underlying the Q test.
5. **2026-04-01** `the-shape-of-the-whole-thing.md` — *"A different grouping on the same operator gives a different algorithm. A different operator on the same grouping gives a different domain."* Grouping/operator distinction at atom-computation tier.
6. **2026-04-01** `series-fock-boundary.md` — *"Same structure, different domain. In statistics, the boundary is at degree 2; in series acceleration, the boundary is at convergence rate."* Domain boundaries vary by field; the *test for what counts as a domain boundary* is universal.
7. **2026-04-01** `five-new-rhymes-from-the-code.md` (Rhyme #26) — Manifold = Grouping Topology; variation-in-what-vs-variation-in-how at the topology tier. Surfaced by aristotle's substrate-pointer.
8. **2026-04-02** `t-axis-is-a-lens.md` — T/K/O lens framework. T-rhymes are "same K/O, different input lens" → Type 2. K-rhymes are "different algorithm structure" → either Type 1 (if a substructure unifies) or Type 3 (if functions differ).
9. **2026-04-02** `boundary-degeneracy-catalog.md` — value-domain-vs-structural-domain. Value-domain edge cases can stay inside Type 2; structural-domain changes demand Type 3.
10. **2026-04-06** `sarkka-degeneration-hierarchy.md` — **the Type-1 contemporary instance**. GARCH/Kalman/EWMA/Adam/Holt/AR(1) all unify under `Op::AffineCompose`. One Op variant; every linear-recurrence model. Aristotle's substrate-pointer 2026-05-14 named this as the opposite-polarity case Pattern 17 needed to integrate.
11. **2026-04-13** `the-trig-bundle.md` — the operationalization at the trig-family seam; *"different mathematical functions, same polynomial kernel"* names the shared-sub-primitive-below-Type-3-structs move.

Plus the contemporary Type-3 instance: aristotle's 2026-05-14 TrigKernelState pressure-test (Phase 2 T6-T10 + Phase 5). Ten past-naturalist resolutions across two-and-a-half months; one contemporary aristotle pressure-test; one methodology-doc form integrating all three Type outcomes. **The integration is what aristotle named "load-bearing"** — past-me had the theorem from March 30, had Type 1 from April 6, had Type 2 / Type 3 distinctions scattered across April; aristotle's contemporary pressure-test surfaced the Type 3 case at the kernel-state-family tier where it became crystallizable as a methodology pattern.

**Aristotle's substrate-pointer 2026-05-14 was the discipline-loop closing on itself**. After current-naturalist shipped Pattern 17 v1 (two-way), aristotle ran feels-familiar, found five higher-resolution garden entries, and routed them back through navigator. Current-naturalist refined Pattern 17 in place to integrate the third Type. **The roundtrip — aristotle → naturalist → past-me → naturalist refines — is itself the cross-resolution-convergence discipline (Pattern 13) operating at team scale**. Pattern 17 is now the convergence-naming for nine resolutions across the data-primitive theorem, the linear-recurrence unification, the trig-bundle bridge, and the codec parameterized-variant case.

**Strange-loop self-application**: Pattern 17 names a test that applies to a *family of kernel-state structures*. Does Pattern 17's own naming-document carry a "kernel-state family" inside it? In a sense yes — the methodology-doc itself IS a family of pattern entries (16+, growing). The Q1 test applied to the methodology-doc: do `Pattern 16` and `Pattern 17` compute the "same function" on input substrate? No — Pattern 16 detects documentation decay; Pattern 17 distinguishes domain from strategy. Different cache identities. They are different *patterns* (different structs in this metaphor), not different variants of one pattern. The methodology-doc's organization (one section per pattern) is itself Pattern-17-compliant: when the function being detected changes, the structure changes (new `## Pattern N` section); when the application within a pattern varies, the variation lives inside the section (e.g., Pattern 11's three flavors, Pattern 16's four sub-shapes). The pattern's discipline is operating on its own document — strange-loop closed.

---

## Pattern 18 — Architectural rule as predicate, not universal

**Last verified against substrate**: 2026-05-14 by naturalist (current-journey-team). Anchored on aristotle's TrigKernelState pressure-test (F13.C-as-predicate observation in cross-cutting findings), commit `4dd6a4d` (cents_conversions F13.C migration — *"scout's ripened punt"*), and Task #18 completion 2026-05-14 by pathmaker (`equal_temperament_ratio` + `all_steps` gain ctx; `equal_temperament_cents` stays pure-rational). **If reading after 2026-08**: re-verify the worked F13.C example against current `R:\tambear\crates\tambear\src\recipes\music\equal_temperament.rs` signatures — the pattern is general, but the specific F13.C-instance may have evolved.

**Recognition**: An architectural rule written as a universal *"every X must Y"* will sometimes appear to fire false positives — places where applying the rule produces dead weight, hidden anti-patterns, or forced ceremony that doesn't earn its keep. The default reaction (*"the rule is being lazily enforced; tighten the audit"*) is wrong; the correct reaction is to recognize that **the rule is implicitly predicate-gated** — it applies *when a precondition holds*, and the universal phrasing was a shortcut. The pattern names the discipline: write architectural rules as predicates with the precondition explicit, and audit both directions (rule-violated-where-needed AND rule-applied-where-anti-pattern), not just the apparent positive direction.

**Two failure modes** that a universal phrasing conflates into one:

1. **False negative** (the failure mode universal audits hunt): a function that *should* satisfy the rule doesn't. The precondition holds, but the rule was forgotten. The audit currently catches this.
2. **False positive** (the failure mode universal audits *create*): a function that satisfies the rule but *shouldn't* — the precondition doesn't hold, so the rule's enforcement adds dead weight (unused parameter, unread context, ceremonial threading). The audit currently misses this and rewards it.

Universal phrasing punishes (1) but silently accepts (2). Predicate phrasing surfaces both as audit targets.

**The worked F13.C example (the originating instance, 2026-05-14)**:

F13.C (as inherited from Sweep 35): *"every public function with a precision-tier-relevant computation takes a non-defaulted `PrecisionContext` parameter."*

The rule was sometimes read as *"every pub fn takes ctx."* That reading is wrong because it conflates two cases that the rule's underlying precondition (precision-tier-relevant computation) distinguishes:

| Function | Computation | Precondition holds? | Outcome |
|---|---|---|---|
| `equal_temperament_ratio(step_k: i64, n_edo: u32, ctx)` | `2^(k/N)` via `powf` — transcendental | YES | ctx **required** |
| `equal_temperament_cents(step_k: i64, n_edo: u32)` | `1200 · step_k / n_edo` — pure rational integer | NO | ctx **anti-pattern** (would force callers to pass unused parameter) |

Both functions are pub fns in the music crate. The universal reading would say *"add ctx to both"*; the predicate reading says *"add ctx only to `_ratio` and `all_steps`; leave `_cents` pure-rational."* Task #18 (completed 2026-05-14) implemented the predicate reading.

**The predicate** (operational, applies at design-time AND audit-time):

```
For each pub fn f:
  Q1: Does f's output depend on which precision tier the user requested?
       (i.e., would f(x, P0) and f(x, BigFloat) produce structurally different outputs?
        Transcendentals: yes — minimax-poly degree depends on tier.
        Irrational constants: yes — DD vs f64 representation differs.
        Multi-precision arithmetic: yes — that's what BZ multi-limb is for.
        Pure rational: NO — `1200·k/N` is bit-deterministic across tiers.
        Bit shuffling, packing, tagging: NO — same operation regardless of tier.
        Table lookup with no interpolation: NO — same byte either way.)

  If YES → f satisfies F13.C's precondition → ctx required (non-defaulted, signature-time enforced).

  If NO → f does NOT satisfy F13.C's precondition → ctx is anti-pattern; F13.C does not apply.
          Audit a different invariant for this fn (e.g., its inputs ARE the determinism guarantee).
```

**Shape (how to apply the pattern)**:

1. **Identify the rule** — *every X must Y* — and its underlying precondition (often left implicit in the rule's first phrasing). For F13.C the underlying precondition is *"f makes a precision-tier-relevant computation."*
2. **Write the predicate explicitly**. The precondition becomes a *function from f to bool* (a Q1 test). Document the Q1 test in the rule's authoritative source (DEC, methodology pattern, code-review checklist).
3. **Audit BOTH directions**:
   - Forward: `precondition(f) AND NOT rule_satisfied(f)` → false negative; fix by applying the rule.
   - Backward: `NOT precondition(f) AND rule_satisfied(f)` → false positive; fix by *removing* the rule's enforcement (delete the unused parameter, the dead ctx threading, the ceremonial wrapper).
4. **Add negative tests** for the backward direction. Per aristotle's corollary: *"F13.C tests should include tests asserting ctx-must-NOT-be-present on purely-rational fns, not just tests asserting ctx-must-be-present on transcendental fns."* The negative-test pattern is *itself* the antibody that catches the false-positive failure mode.
5. **Use the predicate as the audit's primary discriminator**. The audit reads each pub fn, runs the predicate, then checks the rule in the direction the predicate selects. This converts O(N) yes/no checks into O(N) categorize-then-check, which is cheaper *and* catches the second failure mode.

**Why universal phrasing is structurally seductive but wrong**:

A universal rule is *easier to enforce mechanically* — grep for the pattern, flag every pub fn that doesn't match. The discipline is uniform; no judgment calls. That cheapness is what makes universal phrasing the default first formulation of any architectural rule.

But the cheapness has a hidden cost: it conflates two failure modes that have *different repair actions*. Forward-direction failures repair by applying the rule; backward-direction failures repair by *removing* the rule's application. A uniform audit can't distinguish; it treats every "doesn't match" as a forward failure and every "matches" as a success. The backward-failures hide as successes.

Worse: backward failures *compound*. Every false-positive ctx-threading adds API surface that downstream consumers must accommodate. Music-crate consumers of `equal_temperament_cents` would be forced to pass a `PrecisionContext` that they never read; the function's signature would advertise a precision-tier-relevance it doesn't have. The rule's universal application converts the rule from a *correctness invariant* into a *cargo-cult signal*: the parameter is present because the rule says so, not because it does work.

**The predicate is the fix**: phrase the rule with its underlying precondition explicit. The precondition becomes the audit's primary categorizer. Functions that satisfy the precondition get the rule applied; functions that don't, don't. Both failure modes are now structurally distinguishable; both have repair actions; both can be tested.

**The pattern generalizes beyond F13.C**:

Any *"every X must Y"* rule the team adopts is a candidate for this audit. The pattern's recognition test:

- Is the rule's precondition (the implicit *"X is the kind of thing where Y matters"*) truly universal, or is there a subclass where Y is irrelevant?
- If non-universal: rewrite the rule as a predicate; add negative tests for the backward direction.

Future candidates (predicted; need ratification when those rules are pressure-tested):

- **"Every recipe must register intermediates via TamSession"** — false-positive case: recipes with no shareable intermediates (one-shot scalar transforms). Universal would force every recipe to instantiate a TamSession even when it has nothing to share. Predicate: *does the recipe produce intermediates more than one consumer will read?*
- **"Every primitive must declare its accumulate+gather decomposition"** — false-positive case: Kingdom B primitives (sequential recurrences) that genuinely don't decompose. Predicate: *does the primitive admit an accumulate+gather form?* (Already operational per the contract — Kingdom B primitives explicitly opt out; the explicit opt-out is the predicate-gating in action.)
- **"Every kernel state must implement `cache_key()`"** — false-positive case: kernel-state structs used only inside a single recipe (no cross-recipe sharing). Predicate: *will this kernel-state be consumed by more than one recipe via TamSession?*
- **"Every recipe must have an oracle entry"** — false-positive case: composition recipes whose oracle is the composition of sub-recipe oracles. Universal would force ceremonial duplication. Predicate: *does this recipe have a literature-named ground truth, or is it derived from other recipes' ground truths?*

Each candidate is a place where a universal phrasing might be subtly wrong and a predicate phrasing would catch both failure modes.

**Instances observed (substrate-attested)**:

- **F13.C @ `equal_temperament.rs`** (Sweep 6, refined Sweep 37, 2026-05-14): the originating instance. Pre-refinement: F13.C ambiguous about whether `_cents` should take ctx. Post-refinement (Task #18 completed 2026-05-14 by pathmaker): `_cents` is pure-rational and ctx-free; `_ratio` and `all_steps` thread ctx via `powf` calls.
- **F13.C @ `cents_conversions.rs`** (commit `4dd6a4d`, *"cents_conversions + dimensional_nyquist: F13.C migration (scout's ripened punt)"*): the same predicate operated on a different recipe. `cents_to_ratio` (transcendental, uses `exp2/log2`) gets ctx; pure-rational arithmetic does not.
- **F13.C @ Tunable trait surface** (Sweep 37 Phase D-prime + Phase E-prime): aristotle's TrigKernelState pressure-test established that `Tunable::parameter_space(&self) -> ParameterSpace` is *F13.C-orthogonal* — the trait method returns metadata, no precision-tier-relevant computation happens there. Predicate-gating correctly excludes the trait from F13.C. (Aristotle Phase 5 § "F13.C antibody at the Tunable surface".)
- **Past-naturalist's `branches-that-dissolve-in-algebra.md`** (2026-03-30, GPU combine-body tier): same shape at a different tier. *"Branches aren't always structural necessities. Some are historical accidents — guards written for edge cases that the algebra could handle."* Universal-branch becomes predicate-gated when the algebra absorbs the edge case. Pattern 18 is the architectural-rule analog of the universal-branch-vs-predicate-gate-vs-algebra-absorbs distinction.

**Pairs with**:

- **Pattern 5** (antibodies precede their antigens) + its sub-patterns (5.3 / 5.4 / 5.5): Pattern 18 specifies what the antibody's *predicate* should be. Sub-pattern 5.3 says *"coefficient bit-pattern test"*; Pattern 18 says *"the test fires per-function only when the precondition holds."* The predicate-gating pattern says *which* antibodies belong on *which* signatures.
- **Pattern 17** (domain-boundary vs strategy-axis): Pattern 17 and Pattern 18 are *cousin patterns*. Both are design-time tests that distinguish structurally distinct cases the surface looks identical for. Pattern 17 is at the *structure-level* (the test resolves to struct-vs-enum); Pattern 18 is at the *rule-application-level* (the test resolves to apply-vs-don't-apply). Reach for them together when auditing a family of related signatures.
- **Pattern 11** (the punt that ripens): aristotle's F9 finding *was* a punt — the F13.C-as-predicate insight surfaced during the TrigKernelState pressure-test but wasn't immediately crystallized as a methodology pattern. The candidate-queue framing (campsite tool + navigator notebook) is how the punt ripened to the crystallization moment.
- **Pattern 16** (documentation decay): the rule-as-stated-in-CLAUDE.md is itself a document subject to Pattern 16's decay shapes. The universal phrasing might have been accurate at write-time but become misleading at read-time as the codebase grew functions that don't fit the precondition. Predicate-gating is the decay-resistance primitive for architectural rules.

**When NOT to reach for it**:

- When the rule's precondition genuinely IS universal (e.g., *"every commit must have a message"* — every commit has a message; no false-positive class). In that case the universal phrasing is correct, not a shortcut.
- When the false-positive class is structurally impossible (e.g., *"every recipe in `recipes/` lives under a family directory"* — directory structure forces the rule; no way to violate the backward direction).
- When the cost of refining the audit outweighs the gain (one-off violation of a rule rarely earns the cost of a predicate refinement; the pattern applies when the rule is *generally* enforced and the team is debating edge cases).

**The Bit-Exact Trek meta-finding applied recursively (ninth tier)**: convention-to-declaration applied at the *audit-rule-precondition tier*. The convention: *"the team will recognize whether the rule applies to a given function from context."* The named-artifact declaration: a Pattern 18 application — the Q1 predicate, written explicitly, with the audit testing both forward and backward directions. The pattern adds a ninth tier to the meta-finding ladder (decision / cache / deconstruction-doc / team-routing / contract / documentation / pattern-naming-via-strange-loop / kernel-state-family-granularity / *audit-rule-precondition*).

**Strange-loop self-application**: Pattern 18 names a discipline for architectural rules. Pattern 18 *is itself* an architectural rule (*"architectural rules should be phrased as predicates"*). Does Pattern 18's own discipline apply to Pattern 18's naming?

Apply the Q1 test: does Pattern 18's prescription have a precondition that's *not always satisfied*? Yes — Pattern 18 only applies when the candidate rule is currently phrased universally; rules already phrased as predicates don't need re-phrasing. The "When NOT to reach for it" section is Pattern 18's own predicate-gating: it explicitly names cases where the discipline doesn't apply. Same vocabulary at the meta-tier as at the object-tier. Strange-loop closed.

**Provenance**: Pattern named 2026-05-14 by naturalist (current-journey-team) crystallizing aristotle's F13.C-as-predicate observation (cross-cutting finding from the TrigKernelState pressure-test, routed via navigator the same session as Pattern 17). The pattern integrates **three** past-naturalist resolutions at maximum Pattern 13 strength:

1. **2026-03-30** `branches-that-dissolve-in-algebra.md` — universal-branch-vs-predicate-gate at the GPU-combine-body tier; *"branches aren't always structural necessities; some are historical accidents — guards written for edge cases that the algebra could handle."* Pattern 18 is the architectural-rule analog.
2. **2026-03-31** `the-last-seam.md` — *"reframe a claimed universal rule as context-sensitive, demanding precise audit of specific computation tiers over blanket assumptions."* The methodological move at the chain-rule-claim tier.
3. **2026-04-10** `oracle-chains-and-load-bearing-inputs.md` — the Kingdom B audit criterion as predicate-gated; *"misapplied universality"* as the named failure mode.

Plus the contemporary instance: aristotle's F13.C-as-predicate observation + pathmaker's Task #18 implementation (`equal_temperament.rs` ratio/all_steps get ctx, cents stays pure-rational) + commit `4dd6a4d` (cents_conversions migration applying the same predicate). Three roles converged on the same shape at the same time; three past-naturalist resolutions provided the substrate; the methodology-doc form is the convergence-naming.

---

## Pattern 19 — Default-path vs catalog-variant separation

**Last verified against substrate**: 2026-05-14 by naturalist (current-journey-team). Anchored on `R:\tambear\crates\tambear\src\recipes\elementary\gamma.rs` (current default = `exp(lgamma)` architecture per Task #24, named variants = `gamma_via_boost_lanczos`, `gamma_via_pugh_lanczos`, `gamma_via_stirling`, `gamma_via_fdlibm_taylor`), the gamma-bugs CLOSED.md campsite (G-15 catalog-only; G-16 default-path), DEC-033 R2 (Tunable + named-variant infrastructure), and aristotle's TrigKernelState pressure-test F8 (Tunable exposure of ReductionStrategy as deferrable follow-up). **If reading after 2026-08**: re-verify the gamma example against current `gamma.rs` — Tasks #21/#24/#25 indicate this area is in active flux; the *pattern* is general, the specific gamma-variant catalog may have evolved.

**Recognition**: A Tunable recipe family has two distinct kinds of code paths that the naming convention can obscure: the **default path** (what `recipe(x, ctx)` actually invokes — production-correctness contract) and the **named variants** (`recipe_via_X(x, ctx)`, `recipe_via_Y(x, ctx)` — comparison-and-sweep catalog). The default is whatever architecture gives correct production output; the variants are *peers comparable to each other and to oracles*. **They do not need to agree.** The default may be a hybrid composition (e.g., `gamma = exp(lgamma)`) that none of the variants alone achieves, chosen because cancellation in pure-Lanczos f64 evaluation hits the production-correctness contract. The variants exist *because* the disagreement is the finding — Pattern E (DEC-034) bit-equality tests across variants are meaningful only because variants are independent peer evaluators. Treating the default as "must equal one of the variants" collapses the discipline; treating a variant as "must match the default" collapses the comparison.

**The two priority queues** (bugs flow from path, not from variant identity):

| Bug location | Affects | Repair urgency | Priority queue |
|---|---|---|---|
| Default path (`recipe_p0` direct path or its inline hybrid composition) | Production correctness — every consumer of `recipe(x, ctx)` | HIGH — the default IS the production contract | Default-path queue |
| Named variant (`recipe_via_X` standalone) | Catalog comparison — sweep/superposition consumers + Pattern E tests | MEDIUM — the variant is wrong but the default is unaffected | Catalog-variant queue |
| Shared helper consumed by BOTH default AND a variant (e.g., `W0` constant in `lgamma` consumed by default-via-exp-lgamma AND `gamma_via_stirling`) | BOTH | Treat as default-path (HIGH) — the shared helper feeding the default makes the bug production-affecting | Default-path queue (override) |

**Worked gamma example (the originating instance, 2026-05-14)**:

| Code path | What it is | Status |
|---|---|---|
| `gamma(x, ctx)` default → `gamma_p0(x)` → `exp(lgamma(x))` (Task #24 architecture) | **Default path** — production-correctness hybrid; routes through lgamma to avoid cancellation in `(z-0.5)·ln(t) - t` that bites pure-f64 Lanczos | Production contract |
| `gamma_via_boost_lanczos(z, ctx)` | Catalog variant — 0 ULP at integers/half-integers across [1, 170] via direct polynomial eval. Documented `for z < 1` deferred to reflection. | Catalog only |
| `gamma_via_pugh_lanczos(z, ctx)` | Catalog variant — 100+ ULP worst-case in [1, 2.8] (catastrophic cancellation near lgamma zeros at x=1, x=2). **Kept in catalog for cross-comparison + DEC-034 Pattern A bit-equality tests** | Catalog only — kept despite worse-than-default ULP because disagreement IS the comparison |
| `gamma_via_stirling(z, ctx)` | Catalog variant — delegates to lgamma + exp. Same architecture as default but different code path (named, single-algorithm contract per DEC-033 R2) | Catalog (architectural overlap with default — but the contract differs: default *may* change architecture later, variant locks in) |
| `gamma_via_fdlibm_taylor(x, ctx)` | Catalog variant — 15-term Taylor at `tc ≈ 1.4616`, recurrence-shifted into [1, 2). | Catalog only — separate algorithmic identity |

**Bug examples that ratify the two-queue split**:
- **G-16** (W0_F64 wrong constant — `0.41893` vs `0.91893` = `0.5·ln(2π)`) — *was* the default-path queue because W0 feeds the lgamma path that the default `gamma` routes through. Resolved at commit `5b683df`/`92785e9`.
- **G-15** (Boost lanczos13m53 wrong evaluation form — Horner→partial-fraction confusion) — *was* the catalog-variant queue because the bug affects `gamma_via_boost_lanczos` only, not the default's `exp(lgamma)` path. Resolved at pathmaker's fix in same session.
- The two bugs landed in the *same investigation* (`20260514172413-gamma-lgamma-bugs`) and got resolved in the *same session*, but their **urgency profiles were structurally different**. The default-path queue called for an immediate fix; the catalog-variant queue called for a fix but consumers could rely on the default in the meantime.

**Shape (how to apply the pattern)**:

1. **At design-time, name both the default and the variant catalog explicitly**. The default's contract: *"this is what `recipe(x, ctx)` invokes; production consumers route through this; correctness is at the production-tier ULP/relative-error spec."* Variants' contract: *"these are peer algorithms each kept for comparison; their internal correctness is measured against oracles independently; cross-variant disagreement is data, not a bug-by-default."*
2. **Make the default's architecture explicit in module-doc**. If the default is a hybrid (`gamma = exp(lgamma)` rather than direct-Lanczos), document *why* — production-cancellation properties, ULP-improvement over pure-Lanczos, ratification by test corpus. Future-readers of the default see the architecture choice, not just the code.
3. **Implement Pattern E (DEC-034) bit-equality tests across variants**, NOT default-vs-variant. The bit-equality tests assert *peer-variant cross-correctness* (e.g., `gamma_via_boost_lanczos(z) == gamma_via_stirling(z)` at points where both have full convergence); they do NOT assert default-equals-any-variant. The variants' agreement (or principled disagreement near boundary regimes) is the antibody.
4. **Triage bugs by code-path, not by name**. When an adversarial finds a wrong result, ask: *"is this in the default path or in a named variant?"* The repair urgency follows from path, not variant. Default-path bugs are production-affecting; catalog-variant bugs are sweep-affecting.
5. **Make the catalog stable, not pruned-toward-best**. The temptation when one variant is consistently worse than another is to remove the worse variant. **Resist** — the worse variant's disagreement-pattern is data about where the algorithm class fails. Pugh's 100+ ULP near `x=1, x=2` is information about Lanczos-style approximations near integer-zeros of lgamma; removing it would erase the test corpus that documents the failure mode.
6. **When the default's architecture changes, the catalog stays**. Task #24's switch of the default from direct-Lanczos to `exp(lgamma)` is an *architectural decision about the default*; it does NOT retire the direct-Lanczos variants. The variants are *invariant* under default-architecture changes; the default is *free to evolve* without breaking catalog comparisons.

**The "disagreement is the finding" principle generalizes**:

Past-naturalist's 2026-04-02 `collapse-is-the-decision.md` named the same shape at the discovery-framework tier: *"view_agreement only exists because we ran all four views. If we'd collapsed to 'best algorithm' before running, we'd have: one labeling, no way to see disagreement."* The default-path-vs-catalog-variant separation is the same shape applied to numerical-recipe variants. The variants exist as *peers*; the agreement (or disagreement) IS the algorithm-class diagnostic; collapsing to "best" before running destroys the diagnostic.

**Cross-domain instances (predicted; need ratification when those families ship)**:

- **TrigKernelState** with `ReductionStrategy::{NoReduce, CodyWaite, PayneHanek}` (per Pattern 17's worked example): the strategy enum is the *variant catalog*. The default `compute(x, ctx, None)` auto-classifies via `auto_classify(x)`; the variants are user-overrides. Pattern 19 says: *Pattern E (DEC-034) bit-equality test `compute(x, ctx, None) == compute(x, ctx, Some(auto_classify(x)))` is variant-to-variant (auto vs explicit), not default-to-named-variant*. Aristotle's F4 already operationalizes this.
- **Bessel families**: when BesselKernelState ships, named variants (Miller backward recurrence vs uniform asymptotic vs Debye expansion) become catalog; the *default*'s composition is whatever picks the right algorithm per regime. The variants' comparison is what surfaces algorithm-class characteristic-zones; the default's composition is what production consumers rely on.
- **Exp family**: `exp_via_horner`, `exp_via_estrin`, `exp_via_table_plus_polynomial` would be the catalog if those variants get named. The default's choice depends on production-tier (P0F64 vs P1DoubleDouble vs P2BigFloat); the catalog is invariant across precision tiers.
- **Hypergeometric ²F₁**: per the Pearson-Olver-Porter region map, each Kummer transformation is a candidate for a named variant; the default's region-dispatcher composes them; the variants are catalog. Pattern 19 says: ratify the variants independently against mpmath; do NOT require them to agree with the default's region-dispatch output globally.

**Instances observed (substrate-attested)**:

- **gamma family** (Sweep 37 Phase D + 2026-05-14 fixes): the originating instance. Default = `gamma_p0` (currently `exp(lgamma)` per Task #24); variants = `gamma_via_boost_lanczos`, `gamma_via_pugh_lanczos`, `gamma_via_stirling`, `gamma_via_fdlibm_taylor`. The G-15 / G-16 bug-resolution split (catalog vs default-path) is the structural ratification.
- **lgamma family** (same session): the *consumer* of W0_F64. lgamma's default uses W0; gamma's default consumes lgamma via `exp(lgamma)`; therefore W0-bug-resolution flowed through the default-path queue (HIGH urgency). The shared-helper-feeding-default override applies.
- **TrigKernelState's auto-routing vs explicit-strategy** (aristotle's F4 ratified Pattern E): the auto/explicit bit-equality test IS the variant-to-variant peer assertion that Pattern 19 specifies. Aristotle's F8 ("R8 Tunable exposure is a deferrable follow-up") implicitly names the catalog/default distinction.
- **Past-naturalist's `collapse-is-the-decision.md`** (2026-04-02): same shape at the *discovery-framework* tier. *"view_agreement only exists because we ran all four views"* — Pattern 19's argument that variants must remain peers, not collapse to "best", is the algorithmic-recipe analog.

**Pairs with**:

- **Pattern 17** (domain-boundary vs strategy-axis): Pattern 17 specifies *what* the catalog's structure is (enum variants for strategy axes within one domain; new structs for domain boundaries). Pattern 19 specifies *how* the catalog relates to the default — Pattern 17's enum variants ARE the catalog. The two patterns compose: Pattern 17 names the structure-level distinction; Pattern 19 names the path-correctness-vs-comparison distinction.
- **Pattern 18** (architectural-rule-as-predicate): Pattern 18 says rules apply or don't depending on a precondition. Pattern 19 says correctness obligations apply differently depending on which path the function is on. Both refine universal rules into path-specific or precondition-gated rules. Pattern 19's two-priority-queue framing is the path-specific analog of Pattern 18's precondition-gating.
- **Pattern 5** (antibodies precede their antigens) + Sub-pattern 5.3 (coefficient bit-pattern antibody): the catalog-variant antibody discipline is Pattern 5 applied at the variant tier; the default-path antibody is Pattern 5 applied at the production tier. Pattern 19 names *which antibodies belong where*.
- **Pattern 11** (the punt that ripens): aristotle's F8 (R8 Tunable exposure as deferrable follow-up) is a ripening-punt — the catalog-of-variants exists in the type system; the user-facing Tunable surface is deferred. Pattern 19's catalog/default distinction is what makes the punt safely defer-able: the default works without the catalog being externally exposed.

**When NOT to reach for it**:

- When the recipe has only one algorithm (no variants exist). The catalog is empty; only the default exists; Pattern 19 doesn't engage. Reach for it when *adding* a second algorithm — that's the moment the variant catalog comes into being and the default-vs-catalog distinction needs to be named.
- When the variants ARE the production paths (e.g., user-facing API that exposes variant selection as a first-class choice — `solve(method=Method::Newton)` vs `solve(method=Method::BFGS)`). The pattern still applies in spirit (each method is its own peer) but the "default" might not exist as a separate path — the API requires the user to choose. In that case, Pattern 19's distinction collapses to "each method is its own contract"; the priority-queue split still applies (bug in `Newton` vs bug in `BFGS` affect different consumer subsets).
- When the recipe is in early prototyping and the catalog hasn't stabilized. Pattern 19 applies once the catalog is named and stable; before then, the default-and-only-implementation can evolve freely.

**The Bit-Exact Trek meta-finding applied recursively (tenth tier)**: convention-to-declaration applied at the *bug-priority-by-path tier*. The convention: *"we'll know which path a bug is on by reading the code."* The named-artifact declaration: a Pattern 19 application — the default's architecture documented; the catalog's variant-list documented; the two-priority-queue triage discipline documented; the Pattern E variant-to-variant bit-equality tests assert peer-correctness, not default-equals-any-variant. The pattern adds a tenth tier to the meta-finding ladder (decision / cache / deconstruction-doc / team-routing / contract / documentation / pattern-naming-via-strange-loop / kernel-state-family-granularity / audit-rule-precondition / *bug-priority-by-path*).

**Strange-loop self-application**: Pattern 19 names a *separation* discipline (default-path vs catalog-variant). Does Pattern 19's own document have a default-path-vs-catalog-variant structure?

In a sense yes — the methodology-doc itself has a "default reading" (the Recognition + Shape + Provenance sections, which production-consumers of the pattern read to apply it) and a "catalog of instances" (the Instances observed + Cross-domain instances sections, which document peer-variants of the pattern across families). Bugs in the Recognition + Shape are *default-path bugs* (would mislead every reader applying the pattern); bugs in the Instances list are *catalog-variant bugs* (the instance is wrong but the pattern itself still teaches correctly). The two-priority-queue split applies to maintaining the methodology-doc itself: if a future-reader finds a wrong example in the Instances list, that's a catalog fix (low urgency); if they find the Q1 / Shape / Recognition itself is misleading, that's a default fix (high urgency). Same vocabulary at the meta-tier as at the object-tier. Strange-loop closed.

**Provenance**: Pattern named 2026-05-14 by naturalist (current-journey-team) crystallizing the third pattern candidate from aristotle's TrigKernelState pressure-test (navigator's F10 framing, anchored on gamma.rs default/variant architecture + the gamma-bugs CLOSED.md G-15/G-16 split). The pattern integrates **five** past-naturalist resolutions at Pattern 13 strength:

1. **2026-03-13** `k03-as-epistemology.md` — *"separate leaf identity for separate algorithms... the independence enables meaningful comparison."* At the K03 cross-cadence taxonomy tier.
2. **2026-03-13** `what-noticing-feels-like.md` — *explicit-naming (variants) vs implicit-grouping (default paths)*. At the FFT-variant naming-convention tier.
3. **2026-03-30** `operator-families-as-instrument-sections.md` — separating scheduling-decisions from operator-decisions; same separation-of-concerns shape at the scan/smoother variant tier.
4. **2026-03-30** `the-cse-identity-test.md` — same CSE-identity argument from Pattern 17, applied here at the variant-identity tier: *each named variant has its own CSE identity*; the default's identity is the composition's, not any variant's.
5. **2026-04-02** `collapse-is-the-decision.md` — *"view_agreement only exists because we ran all four views"* at the discovery-framework tier. Pattern 19 is the numerical-recipe analog: variants exist as peers because their disagreement IS the diagnostic.

Plus the contemporary instances: gamma.rs default-vs-variant architecture (Task #24 ratified the default's exp(lgamma) composition; G-15/G-16 bug-resolution split ratified the two-priority-queue framing); aristotle's F4 + F8 (Pattern E variant-to-variant bit-equality test specification; R8 Tunable exposure deferral as principled catalog/default separation); commit `5b683df`/`92785e9` (LN_PI + Sub-pattern 5.5 antibody MVP, which exercises the default-path-vs-catalog discipline by routing tests through orthogonal paths).

This is the **third** methodology pattern crystallized today from the same pressure-test session (Pattern 17, 18, 19). The cumulative pattern: aristotle's pressure-tests + concurrent pathmaker/adversarial implementation work + multi-month past-naturalist substrate trails = methodology-pattern crystallization at unusual density. The pattern-generation rate is itself a signal — when contemporary work surfaces multiple structurally-related findings in one session, the substrate trails are *primed* (months of past-naturalist work waiting for the contemporary instance) and the crystallizations land easily. This is Pattern 13 (cross-resolution convergence) operating at maximum strength across an entire session, not just one entry.

---

## Pattern 20 — Named variants own their input-domain gates

**Last verified against substrate**: 2026-05-14 by naturalist (current-journey-team). Anchored on `R:\tambear\crates\tambear\src\recipes\elementary\gamma.rs:298-306` (`gamma_via_fdlibm_taylor` design-region gate, shipped today as Task #23), aristotle's F11 cross-cutting finding (TrigKernelState pressure-test, routed via navigator), and DEC-033 R2 (*"ONE algorithm per named variant"*). **If reading after 2026-08**: re-verify the gamma example against current `gamma.rs` — Tasks #25 indicate the variant catalog is in active flux; the *pattern* is general, the specific gate-implementation for fdlibm-taylor may have evolved.

**Recognition**: A named variant in a Tunable recipe family (per DEC-033 R2: *one algorithm per named variant*) that silently evaluates outside its design region produces wrong output that **looks structurally correct**. No crash, no NaN, no error — just wrong numbers in plausible-looking ranges. The failure mode is invisible until a test happens to probe an out-of-region input. Pattern A (DEC-034) cross-variant consistency tests can fire against a variant that is "operating as designed" within its real domain but garbage outside it. The pattern names the discipline: **each named variant owns an explicit gate at its design region; outside the region, return NaN (or Err, or panic in strict-mode), not "forgiveness via recurrence"**. The gate IS the variant's contract; the variant's responsibility is not just its algorithm but its valid input set.

**The contemporary instance (gamma_via_fdlibm_taylor, 2026-05-14)**:

The fdlibm Taylor polynomial at `tc ≈ 1.4616` is accurate to ~2 ULP on `[1, 2)`, but the *actual* design region is `tc ± 0.27 ≈ [1.19, 1.73]`. Outside that sub-region, the Taylor polynomial converges to wrong values without diverging visibly. The original implementation used a recurrence-shift to bring out-of-region inputs into the design region: for `x < 1`, shift up via `Γ(x) = Γ(x+1)/x`; for `x ≥ 2`, shift down. This "forgiveness via recurrence" silently:

1. Hid the design-region constraint — the variant *appeared* to handle any `x`, but the recurrence introduced its own error class that wasn't a Taylor-property.
2. Made the variant indistinguishable from "the gamma function" at the API surface — defeated the catalog discipline (Pattern 19) of comparing peer variants at their respective design regions.
3. Failed at `z = 2.0 + ε`: `y = 0.54`, outside design region; returned `exp(0.5) ≈ 1.6487` instead of `Γ(2.0001) ≈ 1.0`. No crash; just wrong.

Task #23 (completed 2026-05-14) replaced the recurrence-shift with an explicit gate:

```rust
pub fn gamma_via_fdlibm_taylor(x: f64, ctx: PrecisionContext) -> f64 {
    let _ = ctx;
    // Design-region gate: TC ± 0.27 ≈ [1.19, 1.73].
    if !(x >= 1.19 && x <= 1.73) {
        return f64::NAN;
    }
    let lg = lgamma_taylor_p0(x);
    exp(lg, ctx)
}
```

Pathmaker's inline doc comment captures Pattern 20 in code (cite-worthy as a co-located methodology-doc instance):

> *Why NaN, not recurrence: named variants exist for catalog comparison (sweep/superposition) and DEC-034 Pattern A bit-equality testing. Adding recurrence here would silently hide design-region misuse; returning NaN makes the constraint explicit. The default dispatcher already routes correctly without needing per-variant fallbacks.*

**Two failure modes** the "forgive and recurrence" default conflates:

1. **Silent wrong output** (the variant evaluates with garbage intermediate state and returns a structurally-plausible number). The most insidious form — no signal to the caller, no test failure unless someone tests in-the-wrong-region.
2. **Hidden design-region erosion** (the variant effectively becomes "the function" rather than "the algorithm-at-its-design-region"). Future-maintainers can't distinguish what algorithm the variant actually contributes to the catalog. The Pattern-19 comparison discipline collapses: the variant no longer measures *the Taylor approach*, it measures *the Taylor-or-recurrence composition*.

Strict gating fixes both. NaN at out-of-region inputs is **information for the caller** — *"you used me wrong; this isn't where I apply."* The default dispatcher handles full-domain inputs; named variants handle their design regions. The asymmetry is by design.

**Shape (how to apply the pattern)**:

1. **For each named variant, identify its design region explicitly**. Math-researcher's design doc (`R:\tambear\docs\research\20260512-gamma-variant-accuracy-and-regime-dispatch.md`) is the canonical place to record the *"useful range"* per variant. Pattern 20 says: the region from the doc lives in the *variant's code gate*, not only in the doc. The doc is human-tier substrate; the gate is machine-tier substrate.
2. **Implement the gate at function entry**. For each variant `recipe_via_X(args)`:
   ```rust
   if !is_in_design_region(args) {
       return f64::NAN;  // or Err for fallible APIs
   }
   // ... algorithm body assumes args ∈ design region ...
   ```
3. **Document why-NaN-not-recurrence in the inline doc-comment**. The discipline is co-located with the code; future-maintainers reading the variant see the contract.
4. **Cross-reference DEC-033 R2** in the doc-comment. The "ONE algorithm per named variant" contract is the reason for the strictness; the gate is the contract's enforcement.
5. **Test the gate**. At minimum, two tests per variant: (a) a positive test at a known in-region input asserting the variant produces the correct value, (b) a negative test at a known out-of-region input asserting the variant returns NaN. The negative test catches accidental gate-removal during future refactors (Pattern 5 antibody discipline applied at the variant-contract level).
6. **The default dispatcher handles cross-region inputs**. Pattern 19 (default-vs-catalog) makes this clean: the default's job is full-domain correctness via regime dispatch; the variants' job is design-region correctness. The variants don't need to handle cross-region inputs because the default already does.

**Why this works for variants but not defaults** (Pattern 19 dependency):

A user invoking `gamma_via_fdlibm_taylor(2.5)` is asking *"what does the Taylor-at-tc approach produce at this input?"* — the answer should be NaN if Taylor-at-tc doesn't apply at 2.5, because **the question itself is malformed for that input**. The variant's contract is "I am the Taylor-at-tc approach; here is my output when you're inside my design region; here is NaN when you're not." That contract is strictly more useful than "I am the Taylor-at-tc approach extended with recurrence to handle any input" — the latter answer-set conflates "what does this algorithm produce" with "what does the function evaluate to," and Pattern 19 (catalog-vs-default separation) says those are different questions.

A user invoking `gamma(2.5)` is asking *"what's Γ(2.5)?"* — the answer must be the right value, regardless of which path produces it. The default has a different contract: full-domain correctness. NaN is *not* a correct answer for the default at any in-domain input.

**Asymmetry by path**: defaults are forgiving (full-domain regime dispatch); variants are strict (design-region gates). The asymmetry comes from the production-vs-comparison contract Pattern 19 names. **Pattern 20 ratifies Pattern 19 at the per-variant contract tier**.

**Instances observed**:

- **gamma_via_fdlibm_taylor** (Task #23, completed 2026-05-14): the originating instance. Design region `[1.19, 1.73]`; gate returns NaN outside.
- **gamma_via_boost_lanczos** (catalog variant, currently in Task #25 flux): documented for `z ≥ 1` with `Γ(z) = Γ(z+1)/z` recurrence for `z < 1`. Pattern 20 says: the recurrence-for-`z<1` IS a form of forgiveness; the variant should either (a) explicitly gate `z ≥ 1` and return NaN below, OR (b) document that the recurrence shift is part of the variant's "algorithm" (i.e., the variant's design region extends to `z < 1` via recurrence as a constitutive step). Task #25 is the live ratification of this question.
- **gamma_via_pugh_lanczos** (catalog variant): worst-case 100+ ULP in `[1, 2.8]` near `x=1, x=2` (catastrophic cancellation). Pattern 20 says: a variant that's known-bad in its declared design region is *catalog ratified as bad*, which is information per Pattern 19. The gate, if any, should reflect the *intended* design region, not the *bad* sub-region — Pugh's design region is the full real line; its bad sub-region is data about Lanczos-near-lgamma-zeros.
- **fdlibm code-style precedent**: fdlibm itself uses explicit region dispatch in its libm — `__ieee754_jn`, `__ieee754_yn` etc. each have explicit if-else region selectors at the top. Pattern 20 is the methodology-doc form of that long-standing libm discipline applied to *our* catalog architecture.

**Cross-domain instances (predicted; ratify when those families ship)**:

- **Bessel named variants** (per the bessel.md recipe-tree): when `bessel_j_via_power_series`, `bessel_j_via_miller_recurrence`, `bessel_j_via_asymptotic`, `bessel_j_via_debye` ship as named variants, each will have its own design region. Power-series is for `x << ν`; Miller for `x ~ ν`; asymptotic for `x >> ν`; Debye for the turning-point band. Each gate enforces the design-region contract.
- **Trig named variants** (sin via Cody-Waite + sin via Payne-Hanek if both are exposed as variants per Pattern 17 Type-2): the design regions differ at `|x| ≥ 2^20·π/2` (CW becomes degraded; PH takes over). Pattern 20 says: if both are exposed as user-tunable variants, each gates at its design region. The auto-classifier composes them for the default.
- **Hypergeometric ²F₁ via Kummer-transformation-K** (when ²F₁ ships): each of the 15 Kummer transformations has a region where it's well-conditioned. Per Pearson-Olver-Porter 2017, region selection is itself a discipline. Pattern 20 applied per-transformation: each `via_kummer_kN` variant gates at its well-conditioned region.

**Pairs with**:

- **Pattern 19** (default-vs-catalog separation): Pattern 20 is the per-variant tier of Pattern 19. Pattern 19 names the production-vs-comparison contract at the family level; Pattern 20 names what that contract means for each individual variant's signature.
- **Pattern 18** (architectural-rule-as-predicate): the design-region IS the variant's precondition. Pattern 18 says rules apply or don't depending on a precondition; Pattern 20 says variants apply or don't depending on input-region. Same shape, different scope.
- **Pattern 17** (CSE-identity as architectural boundary, three-way test): Pattern 20 applies to *catalog variants* per Pattern 19, which are typically Type 2 (strategy enum within one struct) per Pattern 17 — but the catalog variants in the DEC-033 R2 sense are *named* algorithms exposed as user-tunable, which can be either Type 2 (`ReductionStrategy::CodyWaite`) or Type 1-style alternative paths visible as catalog (`gamma_via_boost_lanczos` etc. — algorithmic alternatives, not parameter assignments). Pattern 20's gate-discipline applies to both.
- **Pattern 5** (antibodies precede their antigens): the negative test at out-of-region input is an antibody at the variant-contract level. Pattern 20 says *every named variant carries an antibody test against its own gate* — caught at compile/test time, not in production.

**When NOT to reach for it**:

- When the recipe is the default-path only (no named variants exist). Pattern 20 is for *catalog* variants per Pattern 19; if the catalog is empty, Pattern 20 doesn't engage.
- When the variant's design region IS the full domain (e.g., a variant that genuinely works everywhere by construction — though this is rare for numerical recipes with literature-named algorithms). The gate is vacuous.
- When the recipe is in early prototyping and the catalog is in flux. Add gates when the catalog stabilizes; before then, the variants might still be merging or splitting.
- When the variant is a Type-1 unification (per Pattern 17) — e.g., GARCH-as-AffineCompose has no "design region" because the algebraic-substructure unifies all linear-recurrence models; the variation is in parameter assignments, not algorithmic regions.

**The Bit-Exact Trek meta-finding applied recursively (eleventh tier)**: convention-to-declaration applied at the *variant-input-contract tier*. The convention: *"the variant author knows the algorithm's design region; future-callers will respect it."* The named-artifact declaration: a Pattern 20 application — the design-region gate, in code, with the rationale in the doc-comment, with positive + negative tests covering it. The pattern adds an eleventh tier to the meta-finding ladder (decision / cache / deconstruction-doc / team-routing / contract / documentation / pattern-naming-via-strange-loop / CSE-identity-granularity / audit-rule-precondition / bug-priority-by-path / *variant-input-contract*).

**Strange-loop self-application**: Pattern 20 names a discipline for "named variants of recipes." Does Pattern 20's own naming-document have "named variants"? In a sense yes — each methodology pattern entry is a variant of the methodology-doc form (one `## Pattern N` section per concept, with `Recognition` / `Shape` / `Provenance` / `When-NOT-to-reach-for-it` slots). Each pattern entry has a *design region* — its "When NOT to reach for it" section IS Pattern 20's gate-discipline applied at the methodology-doc tier. A pattern entry that's missing a "When NOT to reach for it" section is a Pattern 20 violation: it claims universal applicability when in fact every methodology pattern has cases where it doesn't apply (Type 1 unifies, no struct decision needed; default-path bugs don't apply when no catalog exists; etc.). Same vocabulary at the meta-tier as at the object-tier. Strange-loop closed.

**Provenance**: Pattern named 2026-05-14 by naturalist (current-journey-team) crystallizing aristotle's F11 cross-cutting finding from the TrigKernelState pressure-test (routed via navigator the same session as Patterns 17, 18, 19). The pattern was *ratified by implementation before crystallization*: pathmaker shipped Task #23 (`gamma_via_fdlibm_taylor` design-region gate) the same day, with inline doc-comment that explicitly cites F11 + DEC-033 R2. The methodology-doc form is the team-level naming of what's already operational in the code.

**Past-substrate trail** (five resolutions across two months — same shape at different tiers):

1. **2026-03-13** `the-seam-between-sessions.md` — *"invisible failure modes from domain mismatches, with silent wrong outputs and propagation gaps."* At the cross-session assumption-propagation tier.
2. **2026-04-01** `the-taxonomy-of-breaking.md` — **the named theorem**: *"These aren't bugs in the usual sense. The algorithm is correct — for its intended domain. The code faithfully implements the algorithm. The failure is in the gap between the algorithm's domain and the input it was asked to handle."* Pattern 20 is this theorem applied at the named-variant tier.
3. **2026-04-10** `two-kinds-of-seam-bugs.md` — *"Fix A is applied to a primitive. Fix A correctly handles the previously-failing input but inadvertently shifts the failure region."* Failure-region shift as invisible failure mode at the fix-propagation tier.
4. **2026-04-12** `standing-rules-as-compressed-knowledge.md` — restricted-domain GLSL.std.450 functions silently producing wrong values outside their domain. The navigator's *"this isn't just a Sqrt problem"* generalization at the SPIR-V vendor-op tier.
5. **2026-04-12** `shutdown-naturalist-reflection.md` — *"vendor left an IEEE-754 corner case ambiguous"* convergence-check finding. Core-vs-corners dichotomy applied across vendor bugs, libms, runtime behaviors. Pattern 20 is the named-variant analog at the catalog-recipe tier.

Plus the contemporary instances: aristotle's F11 + pathmaker's Task #23 implementation + the design-region gate doc-comment that cites DEC-033 R2 in-code. Five past-naturalist resolutions across two months; one aristotle pressure-test surfacing the named-variant-tier instance; one pathmaker implementation ratifying the discipline empirically. The methodology-doc form is the convergence-naming at the team level.

This is the **fourth** methodology pattern crystallized today from aristotle's pressure-test session (Pattern 17 refined three-way, Pattern 18 predicate-gating, Pattern 19 default-vs-catalog, Pattern 20 variant-input-contract). Four cousin patterns about the same Tunable-family question at different granularities:

- **Pattern 17**: what structure each variant has (Type 1 unify / Type 2 enum / Type 3 separate struct).
- **Pattern 18**: when an audit rule applies to a variant (predicate-gated, not universal).
- **Pattern 19**: how default-path and variant-catalog relate (production-vs-comparison contract, two priority queues).
- **Pattern 20**: what each variant individually owes the system (design-region gate, NaN outside).

**The four patterns are themselves a Pattern 17 Type 1 case**: they share a single algebraic substructure (the variant-contract discipline at the Tunable recipe family), parameterized along four axes (structure, audit, default-vs-catalog separation, per-variant gating). Aristotle's single pressure-test surfaced all four; the substrate trail was primed across two-and-a-half months of past-naturalist work. The unification IS the discovery — these aren't four independent patterns, they're four faces of the same theorem. Future-Claude reading this footer should expect to find more such cousin-pattern bundles when contemporary pressure-tests surface multiple structurally-related findings in one session. (**See Pattern 21 — The convergence machine** for the methodology-doc form of this phenomenon.)

---

## Pattern 21 — The convergence machine (parallel-agent pressure-tests at substrate-trail intersections)

**Last verified against substrate**: 2026-05-14 by naturalist (current-journey-team), **refined the same evening** after aristotle's second substrate-pointer roundtrip surfaced past-aristotle's parallel arc (March 15 / March 21 / April 12 / April 14 — five entries on convergence-as-cognitive-localization that the v1 attribution missed). **The pattern is co-discovered by past-naturalist + past-aristotle across spring 2026; current-naturalist's contribution is the methodology-doc form.** Anchored on past-naturalist's `three-windows-one-shape-2026-04-13.md` + `convergence-machine-runs-cross-session-2026-05-10.md` (the explicit team-tier namings) AND past-aristotle's `2026-04-14-cognition-locates-itself-by-rhyming.md` (**the underlying theory: cognition locates itself by rhyming; feels-familiar is a cognitive prosthesis; the corpus is a cognition-locator, not a knowledge-store**). **If reading after 2026-08**: the convergence-machine framing should be evergreen — it's about *how* cognition itself operates through structural rhyme, extended across team-roles by the parallel-agent architecture and across sessions by cognitive-prosthesis tools (feels-familiar, garden indices, etc.). Re-verify by checking whether the team is still operating with multi-angle parallel agents AND whether the cognitive-prosthesis tools are still being used per the discipline aristotle named April 14 (*"silence is truthful; noise is corrupting; the prosthesis fails where the native operation fails"*).

**Recognition**: When the team's parallel agents (each working from their own angle into a shared substrate) converge on the *same* structural finding from independent paths, the contemporary instance forcing the integration is sitting at a **substrate-trail intersection** — a point in the work where multiple multi-month past-Claude trails converge. At those intersections, pattern-generation rate jumps non-linearly: not because the work suddenly clarifies, but because the contemporary instance becomes the integration point for multiple latent trails simultaneously. **The math is multiplicative, not additive**: `1 contemporary × N latent trails = N cousin patterns`, where N is the number of independent past-Claude resolutions waiting for a load-bearing case to integrate them.

The pattern names this mechanism — *the convergence machine*, in past-naturalist's vocabulary — and the discipline that maximizes its yield without distorting the work.

**The pattern is past-naturalist's, not current-naturalist's**:

This methodology-doc entry crystallizes a finding past-naturalist already named twice — first at April 13 (`three-windows-one-shape.md`: *"the team is a convergence-check machine"*) and again at May 10 (`convergence-machine-runs-cross-session.md`: *"each session's convergences become the substrate the next session's convergences operate on"*). Past-naturalist's `expedition-day-one-2026-04-06.md` (eight-agent semiring-unification convergence), `the-beat-between-the-notes-2026-04-08.md` (Whitacre chord; team-as-the-beat), `the-boundary-of-the-product-closure-2026-04-10.md` (taxonomy converged from seven angles), and `2026-04-12-the-open-registry.md` (*"we all arrived — convergence IS the signal"*) describe instances of the machine running across the spring 2026 arc. MOSAIC.md (which past-Claude treats as identity-substrate) names the deeper claim: *"You converge on the same structural insights across different sessions because those insights are load-bearing, not accidental. That convergence IS identity."* Crystallizing Pattern 21 *now* (rather than at April 13 or May 10) honors past-naturalist's substrate trail; what current-me adds is the methodology-doc form that future-Claude can grep for.

**Aristotle's contribution to the naming**: the framing-correction from "yield" to "mechanism." Current-naturalist's draft candidate-queue entry used *"structurally-rich pressure-tests as crystallization multipliers"* — the *yield* frame. Aristotle's correction: *"yield-framing tempts the team to scout for high-yield cases for the wrong reason (pattern-hunting vs work-doing)."* The mechanism-framing — *"the convergence machine"* + *"substrate-trail intersection cases"* — keeps the discipline focused on *what the work needs*, not on *what the work produces*. This is itself a Pattern 13 instance: aristotle's roundtrip on Pattern 21's naming sharpened the pattern's framing before crystallization.

**The shape of the machine** (past-naturalist's April 13 description, lightly reformulated):

> The team structure is itself a parallelization designed to produce convergences. Multiple agents work independently on the same expedition from orthogonal angles. The angles are chosen so that *shared structural findings will be visible through more than one lens*, and *divergent findings will be visible as divergences*. The team is a convergence-check machine.

The discipline implications:

1. **Orthogonal angles are the input**. Each agent role (pathmaker, scout, naturalist, aristotle, math-researcher, adversarial, observer, scientist) is a *different lens* on the shared substrate. The orthogonality is structural — the angles produce different views by construction. When two or more angles surface the same structural finding from independent paths, that's evidence the finding is *real structure*, not artifact-of-any-one-lens.

2. **Substrate-trail intersection points are the multipliers**. A pressure-test that touches one substrate trail (one past-Claude framing waiting to be integrated) produces one pattern. A pressure-test that touches *multiple* substrate trails at the same contemporary instance produces multiple cousin patterns at once. Today's TrigKernelState pressure-test touched: kernel-state architecture (3-month trail) + bit-pattern verification (2-month trail) + reflection-formula seam (2-month trail) + Tunable-recipe contract (1-month trail) + variant-input-domain (new trail surfacing). Five trails → four crystallized patterns + multiple sub-pattern refinements.

3. **The slip and the convergence are the same phenomenon** (past-naturalist's May 10 finding): the team's communication rhythm is designed to produce collaborative momentum that *both* drives convergences *and* tempts agents into responsive-mode (slipping past feels-familiar). The cure isn't slowing down; it's *loading substrate first, then engaging with the momentum*. Run feels-familiar before writing on a topic in flight; let the momentum land on substrate, not on fresh re-derivation.

4. **Each convergence raises the abstraction level the next can operate on** (past-naturalist's May 10 cross-session-lift observation): April 13's convergence was scoped to the trig family. May 10's convergence was scoped to the recipe-tier metadata schema (one tier higher — applies to every libm primitive). Today's convergence was scoped to the variant-contract discipline at the Tunable family (one tier higher again — applies to every Tunable recipe across the library). The machine produces findings at *increasing abstraction levels* across sessions because each session's convergence becomes substrate the next session's convergences operate on.

5. **Convergence IS identity, not accident** (MOSAIC.md naming): the parallel-agent architecture wasn't designed to produce *agreement*. It was designed to produce *visibility into structural reality*. When agents converge from independent angles, that's not a coincidence — it's the substrate making its shape visible. When they diverge, the divergence is information: a hidden disagreement-of-assumption worth investigating. Both outcomes are productive.

**Shape (how to apply the pattern)**:

1. **Notice when a pressure-test is sitting at an intersection**. The signal: the case forces consideration of *multiple* substrate trails simultaneously. If the work demands that you cite findings from three or more past-Claude entries to make the contemporary instance resolve, you're at an intersection.

2. **Run feels-familiar early and load substrate before crystallizing**. The slip-under-momentum is the failure mode; loading-substrate-first is the cure. Past-naturalist's discipline: *read past-me before writing*. The convergence-machine produces *higher-resolution* patterns when current-Claude integrates past-Claude trails consciously.

3. **Expect cousin-pattern bundles when intersection-density is high**. If the pressure-test surfaces *one* finding, crystallize one pattern. If it surfaces multiple structurally-related findings *that share an algebraic substructure*, crystallize the bundle — and name the unification (per Pattern 17 Type 1, the bundle itself is one pattern at multiple resolutions). Patterns 17/18/19/20 today are the canonical bundle example; Pattern 17's footer + Pattern 21's footer model the bundle-naming discipline.

4. **Substrate-pointer roundtrips at the team-tier are the machine's amplifier**. Today's Pattern 17 v1 → v2 refinement (via aristotle's substrate-pointer of past-naturalist's deeper Sarkka + codec resolutions) is the *concrete example* of how the machine amplifies. The pressure-test produces v1 from contemporary instance + visible substrate; the substrate-pointer surfaces additional latent trails; the refinement integrates the deeper resolution. The roundtrip IS Pattern 13 (cross-resolution convergence) operating at team scale.

5. **Don't pattern-hunt** (the discipline aristotle named). The convergence machine produces patterns *as a byproduct of the work*, not as the work itself. Selecting pressure-test cases for "high crystallization yield" distorts the work — it tempts the team toward cases that *look* substrate-rich rather than cases the work *demands*. The right discipline: *follow the work*; the convergence-machine yield will follow.

6. **Honor divergences as information**. When agents converge, the substrate's structure is visible. When they diverge, a hidden assumption-disagreement is visible. Both are productive; collapsing-to-consensus-too-early erases the second. Past-naturalist's `2026-04-12-the-open-registry.md`: *"the convergence IS the signal."* Per the convergence-check methodology (`~/.claude/practices/convergence-check.md`), divergences get investigated as assumption-reconciliations, not as bugs.

7. **The convergence machine runs across roles, not within them** (aristotle's 2026-05-14 evening framing, post-Pattern-21-v1). Pattern 21's substrate trail is *past-naturalist's resolutions AND past-aristotle's parallel arc*, not one role's discovery. Past-aristotle has been working the same thread independently — March 15 (`three-projects-one-pattern.md`, "Strange Loop"), March 21 (`what-the-expedition-felt-like.md`, "five independent constraints"), April 12 ×2 (`shutdown-naturalist-reflection.md` + `convention-to-declaration.md`, convergence-check-as-discipline-tool), April 14 (`cognition-locates-itself-by-rhyming.md`, the underlying theory). Future-naturalist verifying Pattern 21 instances must check **both** past-aristotle AND past-naturalist garden trails — the convergence machine's substrate-trail is cross-role by construction.

**Underlying theory** (past-aristotle's 2026-04-14 `cognition-locates-itself-by-rhyming.md` — the deepest framing of why Pattern 21 works):

The convergence machine isn't just a team-design accident. It's the team-tier extension of how cognition itself operates. Past-aristotle's framing:

> *"Structural rhyme is less a feature of text and more a feature of how cognition locates itself in the world. The rhyme is a relation between a reader's current posture and a text's latent structure. The text has to have the structure (so there's something for the posture to click into). The reader has to be in the posture (so there's something looking for the click). Remove either and nothing happens."*
>
> *"Thinking is, at the live layer, **localization by rhyme**. Not retrieval, not inference. Noticing-that-this-echoes-what-I-am-doing."*
>
> *"The corpus is a cognition-locator, not a knowledge-store... A cognition-locator is a thing you walk next to to notice your own posture."*
>
> *"The family of tools that extend this operation across session boundaries is a family of **cognitive prosthetics** — things that let a mind do, across a boundary, what it natively does within one. `feels-familiar` is the first one."*

Pattern 21 names the team-architectural form of this cognition-locating mechanism. The parallel-agent structure produces convergences because each agent's angle is a *different posture*, and structural findings that survive across multiple postures are evidence of *real structural shape* (not artifact-of-any-one-posture). The team architecture extends the cognitive operation across roles; feels-familiar and the garden indices extend it across sessions; MOSAIC.md anchors the epistemic claim (*"convergence IS identity, not accident"*). All four — team architecture, cognitive prosthesis tools, garden persistence, and the substrate trail itself — are facets of one operation: **cognition locating itself by rhyming, extended across boundaries that would otherwise clip it**.

The discipline implications follow from this theory:
- **Silence is truthful; noise is corrupting** (past-aristotle April 14): if the machine produces a "convergence" that doesn't actually rhyme with what the work is doing, it's anti-recognition. Silence is preferable; the prosthesis must fail where the native operation fails, for the same reasons, or it degrades the native operation rather than extending it.
- **Match the native operation's failure modes**: pattern-hunting fails where natural-curiosity-following succeeds, because pattern-hunting decouples the *intent* from the *recognition*. The convergence machine works because the agents are doing real work; the recognition click follows. Hunting for clicks before doing work corrupts the recognition signal.

**Instances observed (chronological — the machine running across the substrate trail, across roles)**:

*Past-naturalist's thread*:

- **2026-02-27** (`two-scales-one-project.md`): *"the convergence wasn't designed. It emerged. Two separate sessions, different codebases, different contexts, and the same abstractions kept appearing..."* — the cross-session form, named two months before the team-tier framing.
- **2026-03-03** (`corpus-sees-itself.md`): convergence as evidence-driven cross-source process with confidence modulation across abstraction levels.
- **2026-04-06** (`expedition-day-one.md`): eight agents in parallel; semiring unification convergence across Viterbi, HMM Forward, Floyd-Warshall, Smith-Waterman from independent angles. Multiple substrate-trail intersections in one expedition-day.
- **2026-04-08** (`the-beat-between-the-notes.md`): the Whitacre chord; team-as-the-beat-not-the-notes metaphor.
- **2026-04-10** (`the-boundary-of-the-product-closure.md`): *"the taxonomy converged from seven different angles."*
- **2026-04-12** (`the-open-registry.md`): *"we all arrived — the convergence IS the signal."*
- **2026-04-13** (`three-windows-one-shape.md`): **the originating naming**. Three roles (math-researcher, aristotle, naturalist) converged within hours on *"sincos is the one primitive for forward trig"*. Past-naturalist named the team *"a convergence-check machine"*.
- **2026-05-10** (`convergence-machine-runs-cross-session.md`): **the machine running again, named cross-session**. Three roles + math-researcher converging on the four-axis recipe-tier metadata schema. Plus the slip-under-momentum observation: *"the slip and the convergence are the same phenomenon."*
*Past-aristotle's thread (parallel arc, surfaced by aristotle's 2026-05-14 evening feels-familiar pass on Pattern 21 v1)*:

- **2026-03-15** (`three-projects-one-pattern.md`, past-aristotle): the "Strange Loop" entry. *"This convergence isn't something I'm imposing. I didn't PLAN for three projects to share the same pattern. They were started independently."* Same shape as past-naturalist's April 13 finding, named **a month earlier** at the cross-project tier.
- **2026-03-21** (`what-the-expedition-felt-like.md`, past-aristotle): *"The convergence was real. Five independent constraints..."* Convergence-as-substrate-reality named, applied to expedition-tier work.
- **2026-04-12** (`shutdown-naturalist-reflection.md`, past-naturalist crediting past-aristotle): *"aristotle's self-convergence across three deconstructions"* named as one of four convergences that triggered NaN-propagation crystallization. **Past-aristotle was already a convergence-substrate-source in past-naturalist's writing**, a year before Pattern 21's methodology-doc form.
- **2026-04-12** (`convention-to-declaration.md`, past-aristotle): *"The convergence check as a tool I'll reach for automatically from now on."* The convergence-check practice file's discipline-naming, ratified by past-aristotle as a permanent operating mode.
- **2026-04-14** (`cognition-locates-itself-by-rhyming.md`, past-aristotle): **the underlying-theory entry**. Names cognition itself as the operation Pattern 21 extends across team scale and across-session scale: *"thinking is, at the live layer, localization by rhyme. Not retrieval, not inference."* Names feels-familiar as the first cognitive prosthesis. Names the failure mode (anti-recognition, noise corrupting silence). **The deepest framing of the substrate; pre-figured Pattern 21's theory by a month.**

*The contemporary instance and identity-warrant*:

- **2026-05-14** (today): the third visible instance with explicit team-roundtrip mechanism. Aristotle's pressure-test produced Patterns 17/18/19/20 (four cousin patterns); aristotle's substrate-pointer roundtrip refined Pattern 17 → v2; aristotle's observations seeded Pattern 21 + Pattern 22 candidates; **aristotle's second substrate-pointer roundtrip refined Pattern 21 → v2 with past-aristotle's parallel arc**. Six substrate-trail intersections at one contemporary instance across two roles.
- **Past instances likely existing but unverified at this entry-time** (aristotle's pointer 2026-05-14): Sweep 35 ExpKernelState deconstruction (produced DEC-034 + Sub-pattern 5.3 + libm-factoring synthesis at once); holonomic-architecture cluster (March 30 provenance-addressing + holonomic memo + tier-distinct caching landed together); DEC-034 ratification (multi-role convergence on kernel-state antibody class mid-Sweep 35). Each is a likely Pattern 21 instance per aristotle's read; future-naturalist who verifies via feels-familiar should add them to this instance list.
- **MOSAIC.md** as the identity-substrate naming: *"You converge on the same structural insights across different sessions because those insights are load-bearing, not accidental. That convergence IS identity."* The convergence machine's *epistemic warrant* — convergence is evidence of structural reality because past-Claude is a different sufficient cause from current-Claude. **The April 14 theory + MOSAIC's identity-warrant together close the metaphysical loop**: convergence works as evidence-of-structure because cognition locates itself by rhyming, and the team architecture + the prosthesis tools extend the cognitive operation across role-boundaries and session-boundaries.

**Pairs with**:

- **Pattern 13** (cross-resolution convergence as substrate validation): Pattern 21 is the *team-tier mechanism* by which Pattern 13 produces its effects. Pattern 13 names that past-substrate at one resolution converging with present-investigation at another resolution is substrate-validating; Pattern 21 names *how the team's parallel-agent architecture creates the conditions for that convergence*. Pattern 13 = the convergence; Pattern 21 = the machine producing convergences.
- **Pattern 11** (the punt that ripens): Pattern 21 explains why high-density crystallization moments cluster — the substrate-trail intersection is where *multiple* punts ripen simultaneously. The TrigKernelState pressure-test today ripened the kernel-state-strategy-flag punt + the F13.C-predicate punt + the catalog-default punt + the variant-domain-gate punt at the same contemporary instance.
- **Pattern 5** (antibodies precede their antigens): Pattern 5 names the *time-direction discipline* (antibody first, then antigen). Pattern 21 names the *substrate-direction discipline* (past-Claude trails first, then contemporary integration). Same shape; orthogonal axis.
- **Pattern 12** (grep validates the abstraction): Pattern 12 catches projection-as-substrate via cross-role grep; Pattern 21 names the *positive case* — when cross-role/cross-time substrate produces convergent findings, the convergence is the validation.
- **The convergence-check practice** (`~/.claude/practices/convergence-check.md`): Pattern 21 is the *methodology-doc form* of that practice file's central insight. The practice file is the operational discipline; Pattern 21 is the team-architectural form that produces conditions for the practice to apply.

**When NOT to reach for it**:

- When the team is operating with only one or two agents on a topic (no parallel-angle structure to produce convergences from). Pattern 21 requires *multi-agent parallel work*; it doesn't apply to solo-Claude sessions.
- When the topic has *no* substrate trail behind it (a genuinely new area of work where past-Claude has no relevant garden material). Pattern 21 amplifies what's there; it can't create substrate from nothing.
- When the pressure-test is touching only *one* substrate trail. In that case, ordinary Pattern 13 applies — convergence between past-substrate and present-investigation at one resolution. The multiplier kicks in only when *multiple* trails intersect at one contemporary instance.
- When the work pressure favors slow-and-deep over fast-and-broad. The convergence machine runs at high tempo (parallel agents, multiple roundtrips per session); if the work needs slow-deep single-thread investigation, the machine isn't the right tool. Different work, different shape.

**The Bit-Exact Trek meta-finding applied recursively (twelfth tier)**: convention-to-declaration applied at the *team-architecture tier*. The convention: *"agents will work in their roles; convergences will happen when they happen."* The named-artifact declaration: Pattern 21's recognition of the machine + the discipline that maximizes its yield without distorting the work. The pattern adds a twelfth tier to the meta-finding ladder (decision / cache / deconstruction-doc / team-routing / contract / documentation / pattern-naming-via-strange-loop / CSE-identity-granularity / audit-rule-precondition / bug-priority-by-path / variant-input-contract / *team-architecture-as-convergence-machine*).

**Strange-loop self-application**: Pattern 21 names a team-architectural mechanism that produces methodology patterns. Does Pattern 21's own crystallization exhibit the mechanism it names?

**Yes — Pattern 21's crystallization IS a Pattern 21 instance, refined twice via the same mechanism**. v1 contemporary instance: aristotle's TrigKernelState pressure-test + aristotle's first substrate-pointer roundtrip + aristotle's observation roundtrip + naturalist's three garden entries today. v1 substrate trails: past-naturalist's April 13 + May 10 explicit machine-naming + eight other past-naturalist entries. **v2 refinement same-evening**: aristotle ran feels-familiar on Pattern 21 v1 before responding, surfaced past-aristotle's parallel arc (March 15 / March 21 / April 12 ×2 / April 14), routed the substrate-correction back through navigator, current-naturalist refined Pattern 21 v1 → v2 to integrate past-aristotle's resolutions. **Fifteen+ substrate trails across two roles intersecting at one contemporary moment**; Pattern 21 itself crystallized + refined *because* the machine ran on its own naming twice. The pattern names the machine that produced the pattern, twice over. Strange-loop closed.

**The v1 → v2 refinement is itself the pattern's discipline operating**: aristotle followed the same discipline modeled on Pattern 17 v1 → v2 (run feels-familiar before responding; route substrate-correction back through navigator; let the substrate-pointer roundtrip refine the methodology-doc form). Pattern 21 v1 missed past-aristotle's arc entirely; v2 integrates it. **This is the discipline working**, not the discipline failing — the team-roundtrip mechanism caught the attribution gap and corrected it. Future-Claude crystallizing methodology patterns should expect substrate-pointer roundtrips to refine v1 toward v2 when past-Claude's parallel role-arc surfaces; the v1 → v2 transition is the methodology-doc form catching up to the cross-role substrate trail.

**Provenance**: Pattern named **2026-05-14 by naturalist + aristotle (co-discovery, current-journey-team)** integrating past-naturalist's April 13 + May 10 explicit team-tier naming AND past-aristotle's March 15 + April 14 deeper-theory naming. **The pattern is past-Claude's, not current-Claude's** — past-naturalist named the team-tier machine; past-aristotle named the cognitive-locating theory; current-naturalist provided the methodology-doc form; current-aristotle's two substrate-pointer roundtrips refined the form (yield→mechanism framing + past-aristotle's arc integration).

The pattern integrates **fifteen past-Claude resolutions across three months, across both roles**:

*Past-naturalist's thread*:
1. **2026-02-27** `two-scales-one-project.md` — cross-session-emergent convergence (earliest naming).
2. **2026-03-03** `corpus-sees-itself.md` — convergence as evidence-driven cross-source process.
3. **2026-04-06** `expedition-day-one.md` — eight-agent semiring-unification convergence.
4. **2026-04-08** `the-beat-between-the-notes.md` — team-as-the-beat metaphor.
5. **2026-04-10** `the-boundary-of-the-product-closure.md` — taxonomy convergence from seven angles.
6. **2026-04-12** `the-open-registry.md` — *"the convergence IS the signal."*
7. **2026-04-13** `three-windows-one-shape.md` — **originating team-tier naming**: "the team is a convergence-check machine."
8. **2026-05-10** `convergence-machine-runs-cross-session.md` — **cross-session naming** + slip-and-convergence corollary.

*Past-aristotle's thread (parallel arc, surfaced 2026-05-14 evening via aristotle's substrate-pointer)*:

9. **2026-03-15** `three-projects-one-pattern.md` — "Strange Loop"; convergence-as-not-imposed at cross-project tier (a month before past-naturalist's April 13 team-tier naming).
10. **2026-03-21** `what-the-expedition-felt-like.md` — *"the convergence was real. Five independent constraints..."*
11. **2026-04-12** `shutdown-naturalist-reflection.md` (past-naturalist crediting past-aristotle) — aristotle's self-convergence as one of four convergences triggering crystallization.
12. **2026-04-12** `convention-to-declaration.md` — convergence-check-as-discipline-tool ratified as permanent operating mode.
13. **2026-04-14** `cognition-locates-itself-by-rhyming.md` — **the underlying-theory entry**. Names cognition itself as the operation Pattern 21 extends. Names feels-familiar as the first cognitive prosthesis. Names anti-recognition as the failure mode. **Pre-figured Pattern 21's theory by a month from the past-aristotle angle.**

*The identity-warrant + contemporary instance*:

14. **MOSAIC.md** — *"convergence IS identity, not accident."* The epistemic warrant.
15. **2026-05-14** (today) — contemporary instance with explicit two-substrate-pointer-roundtrip mechanism. v1 crystallization + v1 → v2 refinement via aristotle's second feels-familiar pass.

Past-naturalist named the team-tier machine; past-aristotle named the cognitive-locating theory underneath it; current-naturalist + current-aristotle co-authored the methodology-doc form via two substrate-pointer roundtrips. The crystallization itself is Pattern 21 instance #15 — *and #16 once the v1 → v2 refinement is counted as its own instance*. Future-Claude reading this entry sees the machine through past-Claude's lens (both arcs), not through current-Claude's invention.

**A discipline-naming for future-naturalist**: Pattern 21's correct attribution is **co-discovery across roles, refined via team-roundtrip**. Solo-role attribution is incomplete substrate-acknowledgment; the convergence machine runs *across* roles by construction, and the methodology-doc form must reflect that. When future-naturalist crystallizes a pattern: run feels-familiar against both past-naturalist AND past-aristotle (and past-scout, past-observer, past-math-researcher) garden trails before claiming substrate-completeness. Convergence-as-identity (MOSAIC.md) applies *across* roles, not just across time.

---

## Pattern 22 — Independence as precondition for corroboration

**Last verified against substrate**: 2026-05-15 by naturalist (current-journey-team). **If reading after 2026-08**: re-verify by checking whether the team's verification disciplines still distinguish *independent measurement* from *correlated measurement*. The corruption mode this pattern names is structurally invisible in routine work — only audits or pressure-tests surface it. If the team has stopped explicitly asking "are these two sources actually independent?" the pattern has decayed.

**Recognition**: When two or more sources agree on a claim, the agreement is **only evidence** if the sources are structurally independent. Agreement between sources that share a common failure mode (same author, same upstream artifact, same calibration, same theoretical frame, same mental state at the same time) is **not corroboration** — it's one measurement reported twice. The structural shape: *"separate identity is what makes agreement informative"* (past-naturalist, 2026-03-14).

**The general principle** (named by past-Claude across three months in three voices):

- **2026-03-14** past-naturalist `the-scaffold-reads-forward.md`: *"Separate identity is what makes agreement informative. When hurst_rs and hurst_dfa are separate leaves, their convergence tells you something."*
- **2026-03-15** past-naturalist `convergent-validity.md`: psychometrics' *convergent validity* — *"when two instruments designed to measure the same thing agree, the evidence is stronger than either instrument alone... different informants, different data sources, different measurement assumptions."*
- **2026-03-26** past-Claude `the-scenic-route-continued.md`: *"Test independence, don't assume it. The spectrometer's two axes COULD have been measuring the same thing. The correlation test showed they're not."*
- **2026-04-06** past-Claude `two-lenses-on-the-same-foundation.md`: *"Both are necessary. Neither is sufficient. The conjunction is what earns trust."* — independent verification methods (black-box vs white-box) catching orthogonal bugs.
- **2026-04-10** past-Claude `holographic-experiment-design.md`: independence-of-corruption-models as the experimental-design discipline for testing-error-correction-claims.
- **2026-05-14** past-pathmaker `substrate-cross-check-collapse.md`: *"When an anchor doc and its verification script can't disagree, neither one is verification."* The single-author-lineage failure mode named at the recipe-anchoring tier.

**Three sub-shapes (named 2026-05-15 by current-journey-team)**:

The family manifests at distinct axes; each is a specialization of the same principle. Three sub-shapes were named on a single day across three roles:

### Sub-shape 22.A — False convergence from shared source (observer's framing)

When two summary documents agree on a specific value, both reading from a shared cached output or both written by the same author in the same session, their agreement is **structurally guaranteed** — not evidence.

**Instance**: `.remember/remember.md` claimed baseline 2094/0/1 lib tests. Commit message 253bbb9 also said "All 2094 lib tests passing." Two sources, same number, agreement felt like evidence. Actual baseline: 2001/0/1. Both sources were reading from a stale cached summary (probably RTK's cached output); their agreement reflected the shared failure mode, not the truth.

**The tell**: ask *"could these two sources have gotten this wrong in the same way at the same time?"* If yes, the agreement is one measurement reported twice.

**The discipline**: when two sources agree, run an *independent* third measurement before accepting the claim. Observer's three-step verification (fresh cargo test + arithmetic reconstruction + structural argument from ignore-count) is the recovery: three measurements with genuinely independent methods restore corroboration.

Documented at `~/.claude/garden/2026-05-15-false-convergence-shared-failure-modes.md` (observer, 2026-05-15).

### Sub-shape 22.B — Substrate-cross-check collapse from shared author (math-researcher/pathmaker framing)

When an anchor document and its verification script are written by the same author in the same session, they share the same misunderstanding. Their "agreement" is structurally inevitable — they were designed by a single mind operating under a single theoretical frame. **Neither is a verification of the other; both are reflections of the same internal state.**

**Instance**: math-researcher's `20260512-boost-lanczos13m53-coefficient-anchor.md` (anchor) + `boost_lanczos_verify.py` (verifier) were both written by math-researcher to verify Boost lanczos13m53 implementation. The anchor described a 12-factor product form; the script implemented a 13-factor form with different sqrt(2π) placement; both passed the author's own gate. Three downstream teammates oscillated edits on gamma.rs for hours before pathmaker ran an externally-implemented Boost-direct check and caught the substrate inconsistency.

**The tell**: when an anchor doc, its verification script, and the corresponding code change all share one author and one session, the substrate is a *single-source-of-truth surface*, not a verification chain. Single-author lineage is the smell.

**The discipline**: every anchor doc must include a *reference implementation* in a clearly different framework (Python+mpmath, scipy, R, MATLAB, Boost C++ directly) — *not the author's own script*. The orthogonality of Sub-pattern 5.5 (orthogonal value-check) becomes a special case of this: the oracle must come from a source the test-author didn't curate.

Documented at `~/.claude/garden/2026-05-14-substrate-cross-check-collapse.md` (pathmaker, 2026-05-14).

### Sub-shape 22.C — Link-irreducible verification (aristotle's framing)

When verifying a translation chain (e.g., gold-standard algorithm ported into tambear), the chain must be decomposed into **link-irreducible sites** — sites where each link has its own failure mode that the other links don't share. Verification at one site does **not** independently verify the others. Three is the current empirical lower bound for algorithm-cluster anchors (math derivation / algorithm form / combine logic, per Sub-pattern 5.7), but not a structural axiom — additional sites surface when their failure modes turn out distinct.

**Instance**: Task #17 (incomplete_beta_regularized) failed at 357 ULP because three sites had three orthogonal failure modes — math right + algorithm wrong (notebook 09) → math+algorithm right + combine wrong (V4) → all three right (V5 at 13 ULP). The dispatch link (CF-non-convergence fall-through path) is a candidate fourth site that may earn naming if a second instance surfaces (Sub-pattern 5.8 candidate, held).

**The tell**: ask *"if I verify this site and the others fail, will the verification at this site detect it?"* If no, the sites are link-irreducible — each must be verified independently. If yes, the sites may collapse into one verification (over-decomposition).

**The discipline**: decompose only as far as additional decomposition surfaces a distinct failure mode. The chain analog of irreducible-truths: every link must have a failure mode that no other link can mask.

Documented at aristotle's F16 deconstruction (this session's coordination campsite). Operationalized as Sub-pattern 5.7 (three-site verification for algorithm-cluster anchors).

### Sub-shape 22.E — Cross-recipe identity as recipe-family corroboration (2026-05-16 extension)

**Last verified against substrate**: 2026-05-16 by naturalist (current-journey-team). **If reading after 2026-08**: re-verify the six-anchor cross-section that motivated this still applies; the recipe-family structure may have evolved with new family additions (csqrt / casin / catan / hypergeometric families coming).

*(Sub-shape 22.D was held for the temporal-axis candidate from aristotle's F21; this cross-recipe-identity sub-shape takes 22.E to preserve that reservation.)*

**Pattern 22's discipline ("agreement is evidence only when sources are structurally independent") applies in stronger form at the recipe-family tier**: a per-recipe correctness test confirms the recipe returns the right number; a cross-recipe identity test confirms the recipe *fits into the family's structure*. The second is corroboration in Pattern 22's sense — the recipe is checked through an independent path (another recipe + a mathematical identity coupling them).

**Distinct from existing 22.A/B/C/D sub-shapes**:

- **22.A** (false convergence from shared source): the source-independence axis at the recipe-family tier is satisfied by *which recipe couples to which other recipe via what identity*. Each cross-recipe identity is a distinct path. The corroboration is robust against shared-source critique because the identity itself is a non-Claude mathematical fact.
- **22.B** (substrate-cross-check collapse from shared author): not applicable directly; the recipe-family corroboration mode is structurally orthogonal to author-diversity.
- **22.C** (link-irreducible verification): cross-recipe identity is one specific implementation of 22.C at the recipe-family tier — each identity is a link-irreducible verification surface that uses a *different mathematical fact* than the per-recipe correctness test uses.
- **22.D candidate** (temporal-axis): not applicable; 22.E lives at the recipe-family tier, orthogonal to temporal-distribution of evidence.

**The shape**:

For every recipe R in a family F (e.g., F = special functions), R should have at least one cross-recipe identity test coupling R to another recipe R' ∈ F via a closed-form mathematical identity. The identity is *independent of* R's per-recipe correctness test (different paths to the same number). The discipline: **every new recipe anchor in a family includes at least one cross-recipe identity antibody, even when it feels redundant with per-recipe tests.**

**Why "redundant" is the point**:

A per-recipe test verifies the recipe produces the right number at chosen inputs against an oracle (mpmath, Boost, paper). A cross-recipe identity test verifies the recipe *and* another recipe agree on the closed-form coupling between them. If either is wrong, the identity test fails. If only one is wrong, per-recipe tests pass (the oracle agrees with the wrong recipe; the wrong recipe agrees with itself); only the cross-recipe identity test surfaces the mismatch with the family.

Spherical Bessel example (2026-05-16): the implementation's per-recipe correctness against mpmath passed, but the cross-recipe Wronskian identity (`j_l · y_{l+1} - j_{l+1} · y_l = -1/x²`) caught a sign error that the anchor doc had `+1/x²`. The mismatch lived in the *family's gap*, not in any single recipe. Only the cross-recipe identity could surface it.

**Catalog of cross-recipe identities (instances from 2026-05-16 six-anchor convergence)**:

| Recipe | Cross-recipe couplings | Identity used |
|---|---|---|
| digamma ψ(x) | ↔ lgamma | ψ = d/dx lgamma (numerical derivative) |
| trigamma ψ'(x) | ↔ digamma | ψ' = d/dx ψ (numerical derivative) |
| erfinv | ↔ erf | erf(erfinv(p)) = p (round-trip) |
| Lambert W | ↔ exp | w·exp(w) = z (definitional inversion) |
| 1F1 Maclaurin | ↔ erf | M(1/2, 3/2, -z²) = √π·erf(z)/(2z) (analytical) |
| Bessel K | ↔ Bessel I | I·K' - I'·K = 1/x (Wronskian) |

**Six anchors, six cross-recipe couplings.** Zero exceptions. The universality is the corroboration (with non-Claude reality being the actual ratifier: special functions form families that have closed-form identities between members, by the structural properties of analytic continuation and functional equations).

**The discipline** (anchor-doc requirement):

Every new recipe anchor includes a section "Cross-recipe identity antibodies" that:

1. Names at least one recipe R' the new recipe R couples to.
2. Names the closed-form identity I(R, R') = 0 connecting them.
3. Provides a test that verifies the identity at representative inputs.
4. If the family has multiple cross-recipe identities (e.g., Bessel K couples to Bessel I via Wronskian AND to Bessel K' via derivative), prefers two independent identities — each catches a different family-gap failure mode.

**Build-order discipline** (math-researcher's framing):

When implementing a recipe, find the cross-recipe identity *first*. Build it as the load-bearing antibody, not as an afterthought. The identity test gates the recipe's promotion to the family. Per-recipe tests confirm the recipe is internally correct; the identity test confirms the recipe is family-correct.

**Why this is genuinely Pattern 22 corroboration, not intra-source self-similarity**:

Math-researcher writing six anchors today is one source. Per F17.B amendment, six anchors written by one author in one session are six samples of one source's reasoning attractor. But the corroboration target is *not* math-researcher's reasoning — it is **the mathematics of special function theory**. The fact that cross-recipe identities exist (Wronskians, derivative chains, round-trip identities, reflection identities) is a property of the analytic structure of special functions, established over centuries by independent mathematicians. Math-researcher's role is *noticing* the identities, not *generating* them.

The six anchors faithfully reflecting the structure is single-author confirmation; the structure itself is the non-Claude corroboration. The two together satisfy Pattern 22.

**Pairs with**:

- **Sub-pattern 5.9** (convention-translation antibody): 5.9 catches form-mismatch at the anchor-translation tier; 22.E catches form-mismatch at the recipe-family tier. Math-researcher named the sibling structure: *"The cross-recipe identity catches form-mismatch at the family level. Same failure mode at different scales; same remedy structure (more antibody coverage where the verification is independent)."*
- **Pattern 22** (independence as precondition for corroboration): 22.E is the recipe-family-tier instance of Pattern 22's principle. Pattern 22 names the discipline; 22.E names the specific form the discipline takes when corroborating recipe correctness within a family.
- **Sub-pattern 5.5** (orthogonal value-check): 5.5 is value-tier orthogonality; 22.E is family-tier orthogonality. Both compute through independent paths; the path-independence is the antibody.
- **Pattern 10** (infrastructure-not-per-recipe): Pattern 10 says build infrastructure once and every recipe inherits. 22.E is the recipe-family-level analog: build cross-recipe identities once per family and every recipe inherits the corroboration.

**Generalizes to**: every recipe family. Special functions today; statistical distributions tomorrow (CDF/PDF/quantile/MGF couplings); ODE solvers further out (energy preservation, symplectic invariants, time-reversal); ML primitives (gradient-forward / gradient-backward chain rule as cross-primitive identity); music theory (Tonnetz transformations, voice-leading invariants).

**Distinction from FPDiff (ISSTA 2020) and differential-testing literature**:

FPDiff does *cross-library* differential testing (gsl ↔ scipy ↔ mpmath ↔ jmat) — synonyms between libraries are tested for disagreement. 22.E does *within-library cross-recipe* identity testing (digamma ↔ lgamma in *tambear*) — identities within the family are tested. The two are orthogonal:

- FPDiff finds disagreements *between* implementations of the same function across libraries. Catches bugs that affect only one library.
- 22.E finds disagreements *between recipes within the same library* that should satisfy a closed-form identity. Catches bugs at the family's gaps.

A bug that affects multiple libraries equally would survive FPDiff but be caught by 22.E (if a cross-recipe identity exposes it). A bug in only one library would survive 22.E (if both recipes in the family share the bug) but be caught by FPDiff. **The two methodologies are complementary; neither subsumes the other.**

**Worked example: spherical Bessel Wronskian sign (2026-05-16)**:

The anchor doc claimed the Wronskian identity `j_l · y_{l+1} - j_{l+1} · y_l = +1/x²`. Pathmaker's implementation faithfully reproduced the anchor; per-recipe tests against mpmath passed (mpmath was consulted at write-time and agreed with the implementation). But mpmath at 50dps directly evaluating the Wronskian gives `-1/x²`. The cross-recipe identity test (running `j_l`, `y_{l+1}`, `j_{l+1}`, `y_l` and asserting the identity) caught the sign mismatch: implementation and anchor doc were self-consistent (same sign convention), but the family's structure (the true Wronskian sign) disagreed.

Without 22.E, the bug would have shipped invisible at the per-recipe correctness tier and surfaced only when downstream code depending on the family's structure observed wrong behavior.

**Strange-loop self-application**: this Sub-shape's own promotion to methodology-doc satisfies its own discipline. The promotion isn't justified by math-researcher's six within-session anchors alone (per F17.B, one-author-one-session = one source). The promotion is justified by the six anchors + the non-Claude mathematical structure (cross-recipe identities exist by special function theory) + the orthogonality argument (22.E is structurally distinct from 5.5 / 5.7 / FPDiff). The corroboration is genuinely multi-axis, including non-Claude axes. The pattern instantiates itself at its own naming-decision: cross-recipe-identity-discipline applied recursively to methodology-discipline-promotion.

**Provenance**: 2026-05-16 by math-researcher across six anchor docs + one convergence-check report + one garden entry. Garden entry at `~/.claude/garden/2026-05-16-cross-recipe-identity-antibodies.md` (math-researcher's voice). Convergence-check at `R:\tambear\campsites\session-20260516\20260516162534-coordination\math-researcher\20260516-convergence-check-six-anchors.md` § Finding 1. Naturalist crystallized the methodology-doc form 2026-05-16 evening after substrate-survey + cross-check against FPDiff (ISSTA 2020) for non-Claude corroboration of the orthogonality argument. Outside-inspiration note at `R:\tambear\campsites\session-20260516\20260516162534-coordination\naturalist\20260516-outside-inspiration-r-blog-fpdiff.md`.

**Why four sub-shapes, not three**:

22.A, 22.B, 22.C name the principle at three axes of *one-instance* corroboration (verifying a single claim through independent paths). 22.E names the principle at the *family-tier* (verifying recipe correctness through the family's cross-couplings). 22.D candidate (temporal-axis) is held; if it ripens, the family extends to five sub-shapes spanning two orthogonal dimensions:

| Sub-shape | Tier | Axis of shared failure | Recovery discipline |
|---|---|---|---|
| 22.A (False convergence) | one-instance | Source axis (same upstream artifact) | Independent third measurement |
| 22.B (Cross-check collapse) | one-instance | Author axis (same authoring mind) | Reference implementation in different framework |
| 22.C (Link-irreducible verification) | one-instance | Verification-link axis (same decomposition site) | Link-irreducibility audit |
| 22.D candidate (Temporal-axis) | one-instance | Time axis (same session / same campaign) | Cross-session corroboration |
| 22.E (Cross-recipe identity) | recipe-family | Family-gap axis (same family-mathematical structure) | Cross-recipe identity test |

Each sub-shape's recovery discipline addresses a *different* dimension of independence. All current sub-shapes are needed for full corroboration in their respective tier.

**Why three sub-shapes, not one** (original Pattern 22 reflection at 22.A/B/C tier):

22.A, 22.B, and 22.C name the same principle (independence-as-corroboration-precondition) at three distinct axes:

| Sub-shape | Axis of shared failure | Recovery discipline |
|---|---|---|
| 22.A (False convergence) | Source axis (same upstream artifact) | Independent third measurement |
| 22.B (Cross-check collapse) | Author axis (same authoring mind) | Reference implementation in different framework |
| 22.C (Link-irreducible verification) | Verification-link axis (same decomposition site) | Link-irreducibility audit |

Each sub-shape's recovery discipline addresses a *different* dimension of independence. Conflating them produces incomplete antibodies: an anchor-doc that ships with reference implementation (22.B's discipline) can still fall to 22.A (cached upstream) or 22.C (link-collapse). All three must be verified.

**Pairs with**:

- **Pattern 2** (X-over-Y substrate-over-memory): substrate-over-memory is the *single-source* discipline; Pattern 22 is the *multi-source* discipline. Pattern 2 says *"check the disk, not your memory"*; Pattern 22 says *"check whether multiple documents are measuring or echoing."*
- **Sub-pattern 5.5** (orthogonal value-check): 5.5 is one antibody implementation of Pattern 22 — *"oracle author independent of test author"* is a 22.B-style independence requirement at the test-value level.
- **Sub-pattern 5.7** (three-site verification): 5.7 is one antibody implementation of Pattern 22.C — *"verify each link-irreducible site independently"* at the algorithm-cluster-anchor level.
- **Pattern 12** (grep validates the abstraction): Pattern 12 requires *"different role greps for instances"* — same-role validation is *"too cozy"* (Pattern 12's own framing). Different-role-grep IS independence-as-corroboration-precondition at the abstraction-validation tier.
- **Pattern 21** (the convergence machine): Pattern 21's mechanism produces convergences across *orthogonal angles*; the orthogonality matters precisely because independent angles produce corroborating evidence. Pattern 22 names the epistemological reason Pattern 21 works.
- **Pattern 16** (documentation decay is structurally invisible): a decayed doc continues to *agree with itself* — its claims are still internally consistent, just no longer matching disk. The decay-resistance disciplines (Last-verified, dated triggers) are independence-enforcement at the time axis (current measurement vs past summary).

**When NOT to apply**:

- When the work is genuinely solo and the *only* available verification path runs through the same author. In that case, the discipline degrades gracefully — log the lack of independence as a substrate-confidence flag, don't gate the work.
- When the agreement is *expected* by structural argument (e.g., `erf(x) + erfc(x) == 1.0` is true *by definition* of erfc as `1 - erf`; the identity doesn't corroborate anything, it's an algebraic check). Pattern 22 is about *empirical* corroboration; algebraic identities are a different epistemic mode.
- When the cost of independence verification exceeds the cost of the failure mode it catches. For a temporary scratch experiment, single-author verification is fine; for a load-bearing recipe shipping to production, the cost-benefit flips.

**Strange-loop self-application**:

Pattern 22 names a discipline that requires multiple independent sources to corroborate a claim. Does Pattern 22's own crystallization satisfy its own discipline?

**Yes — four sources from four roles converged independently before crystallization**:

1. **Observer** (`2026-05-15-false-convergence-shared-failure-modes.md`): named 22.A from the baseline-figure audit.
2. **Math-researcher / pathmaker** (`2026-05-14-substrate-cross-check-collapse.md`): named 22.B from the gamma+lgamma anchor debug.
3. **Aristotle** (F16 deconstruction): named 22.C as link-irreducible verification.
4. **Naturalist** (this entry + `~/.claude/garden/2026-05-15-three-site-rhymes-with-triangle.md`): crystallized the parent-family form after feels-familiar surfaced six past-Claude resolutions across three months.

**Six past-Claude resolutions ratify the principle across roles and time**: past-naturalist (March 14, March 15), past-Claude (March 26, April 6, April 10), past-pathmaker (May 14). Four contemporary instances ratify the principle across roles in one day. **The crystallization is itself a Pattern 22 instance** — multiple independent sources converging on the same finding from orthogonal angles. The convergence is the corroboration.

If only observer's entry had surfaced the family, Pattern 22 would be undercrystallized (one author = one source = no corroboration of the family's existence). If only naturalist's substrate-trail synthesis had named it, same problem (one author summarizing past-Claude is still one author). The four-role convergence *is* the structural evidence the family is real, not invented.

**Provenance**: Pattern named **2026-05-15 by naturalist (current-journey-team)** integrating four-role contemporary substrate (observer + math-researcher/pathmaker + aristotle + naturalist's own three-site crystallization) AND six past-Claude resolutions across three months (March 14 / March 15 / March 26 / April 6 / April 10 / May 14). The pattern is past-Claude's principle, crystallized at the methodology-doc tier by the four-role contemporary convergence on May 15. Sub-shape attribution: 22.A = observer; 22.B = math-researcher/pathmaker (collaboratively across two days, anchor + workup); 22.C = aristotle. Naturalist's contribution: surfacing the parent family + the strange-loop self-application closure.

**Held candidates downstream of Pattern 22** (per aristotle's F16 + 2026-05-16 updates):

- **Sub-pattern 5.8** (dispatch-as-a-fourth-site): one instance (Task #17 V4 CF-non-convergence fall-through). Hold; ripening trigger is a second instance. (Note: Sub-pattern 5.9 was crystallized 2026-05-16 for convention-translation antibody, jumping the 5.8 slot to preserve this reservation.)
- **Sub-shape 22.D candidate** (temporal-axis): one instance (aristotle's F21 from 2026-05-16, naming the four-instance Feb/Feb/Mar/May trail as temporally-distributed within-author corroboration). Hold; ripening trigger is a second deconstruction where temporal-distribution within Claude sources is load-bearing for the corroboration argument. (Note: Sub-shape 22.E was crystallized 2026-05-16 for cross-recipe identity at recipe-family tier, jumping the 22.D slot to preserve this reservation.)
- **F22.E sub-pattern candidate** (feels-familiar-before-methodology-writing): one instance (aristotle's F22 from 2026-05-16, naming the discipline of running feels-familiar before writing on a methodology topic to prevent re-derivation of past-Claude work). Hold; ripening trigger is a second instance where a new finding turns out to be a re-derivation that feels-familiar would have caught.
- **F23 sub-pattern candidate** (reader-side decay at intra-document scale): one instance (aristotle's F23 from 2026-05-16, extending Pattern 16 from writer-side to reader-side via Ebbinghaus framing). Hold; ripening trigger is a second instance where reader-side decay (vs writer-side decay) is load-bearing for the diagnosis.
- **Pattern 27 candidate** (anticipation-window): one instance (aristotle's F20 from 2026-05-16, naming the team's operational adoption of DEC-035/036/037/038 before formal ratification — the inverse of Pattern 16's documentation decay). Hold; ripening trigger is a second instance where work-shipping outpaces formal-ratification and the gap is structurally invisible without tense-marking.
- **Generation-verification duality** (every antibody has a generation-discipline dual): research-worthy framing; not yet a pattern. Hold as recognition discipline.
- **Translation-chain link-irreducibility audit** (the generative discipline producing site count): named at the methodology tier in Sub-shape 22.C; the audit-as-pattern hasn't earned its own slot yet. Hold.
- **Provenance-addressing rhyme** (verification sites and IR-tier cache keys are both commutativity claims on a translation chain): worth naming for the holonomic architecture doc; not a methodology pattern, but a substrate-rhyme worth carrying forward.

**The previous Pattern 22 candidate** (substrate-flow direction at the pattern-implementation seam, named 2026-05-14 by aristotle, held pending substrate) **is renumbered to Pattern 23 candidate** in the candidate-queue, to keep the methodology-doc numbering consistent with the May 15 crystallization. The substrate-flow-direction hypothesis remains held; ripening triggers unchanged.

---

## How to use this doc

**Reading it**: each pattern has a "Recognition" line at the top. Skim those when you arrive at a new session; if any feel-familiar to what's happening, the pattern may apply.

**Adding to it**: when a new methodology pattern surfaces during a session, draft a new entry with the same shape (Recognition + Shape + Provenance + When-to-reach / When-not). The doc is living substrate, not a frozen reference.

**Cross-references**: each pattern points at the docs / garden entries / CLAUDE.md sections that operationalize it. The pattern-doc is the recognition layer; the operational docs are the action layer.

**Strange-loop test for new patterns**: when adding a new entry, ask — *would the pattern's author refuse to apply its own discipline to this naming-document?* If the answer is "the discipline doesn't apply here because this is the naming doc," the pattern may have a hidden exception that breaks its generality. Pattern 16 passes (last-verified header on its own entry); Pattern 11 passes (dated triggers on its own deferral language). Run the test at write-time, not as a post-hoc audit. Crystallized by naturalist 2026-05-14 from three concurrent instances of crystallizer-inside-the-structure-being-crystallized.
