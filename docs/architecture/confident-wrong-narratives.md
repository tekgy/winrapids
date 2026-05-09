# Confident Wrong Narratives — apparatus-first investigation

**Status**: Methodology note, drafted 2026-05-09 by main-thread Claude + Tekgy. Anchored on math-researcher's sin-2^1000 false-narrative episode (same day) and navigator's `~/.claude/garden/2026-05/2026-05-09-confident-wrong-narratives.md` essay analyzing it.

**Companion to**: `R:\tambear\docs\TESTING.md` two-axis taxonomy (algorithm claim vs test structure). This doc names a *failure mode* of the taxonomy — the case where an investigator with domain knowledge constructs a sophisticated Axis-1 narrative for a problem that's actually Axis-2 apparatus failure.

---

## The episode

Math-researcher built the sin libm oracle harness. Apparent finding at corpus entries near `x = 2^1000`:

- 9.2 × 10¹⁸ ULP difference from MSVC libm
- Wrong sign
- Completely different value

Math-researcher then wrote a detailed, technically principled explanation:

> MSVC libm's Payne-Hanek implementation truncates the 2/π table below 1200 bits. For `x ≈ 2^1000`, you need precision from bit 1000 down to bit 947, plus guard bits. A 256-bit table only covers bits ~256 to 0, which is far too short. The catastrophic failures are MSVC's reduction running out of table.

The explanation:
- Cited Sun Microsystems' libm reference correctly
- Walked through the bit-extraction algorithm correctly
- Made a specific, falsifiable prediction about the failure mechanism
- Was internally coherent, technically sophisticated, and **entirely wrong**

The actual issue: the sin corpus loader was calling `serde_json::Value::as_f64()` on a decimal string that didn't round-trip correctly through Rust's float parser. `"1.0715086071862673e301"` was being parsed to `0x7e6fffffffffffff` instead of the intended `0x7e70000000000000` — losing the LSB. The "phenomenon" being explained didn't exist.

The corrected harness uses `parse_f64_bits_field` on the hex bit-pattern field instead of `as_f64()` on the decimal-string field. Round-tripping through hex bits is bit-exact by construction; round-tripping through decimal strings is not. **MSVC libm sin is ≤ 1 ULP everywhere across the 275-entry corpus, including 355/113 adversarial inputs and Payne-Hanek targets up to 2^1023.** The platform baseline was nearly perfect; the Tang-style criticism in the false narrative misallocated to the wrong implementation entirely.

---

## The specific failure mode

**Domain knowledge made the false narrative more convincing, not less.**

A naive investigator would say *"that's weird, let me double-check the apparatus"* — they have no ready-made explanation, so the apparatus check is the natural next step. A knowledgeable investigator has a ready-made explanation: Payne-Hanek table truncation, k-multiplier degradation, catastrophic-cancellation-in-reduction. The explanation is so plausible — so consistent with how transcendental approximations actually fail in practice — that **it bypasses the apparatus check entirely.**

This is different from ordinary overconfidence. The Sun Microsystems citation was accurate. The bit-extraction walkthrough was correct. Every individual piece of technical knowledge deployed in the narrative was real. What was wrong was the inference from *"this explanation is consistent with what I know"* to *"therefore the apparatus is fine."*

The confidence gradient runs the wrong way:

- **Low-knowledge interpretation**: "I don't know what this means; let me check the inputs."
- **Knowledgeable interpretation**: "I have a ready explanation that fits the symptoms; the algorithm is the suspect."

The knowledgeable path is *more* likely to skip the apparatus check, not less. The narratives an investigator can construct most confidently — citing real references, walking through real algorithms — are precisely the ones most likely to survive internal coherence testing even when wrong. The failure mode is that **the wrong explanation passes all the checks the investigator can apply from inside it.**

---

## Connection to substrate-over-memory

The same discipline applies inward.

`CLAUDE.md` says: *"My conversation context is a snapshot — not authoritative. When I think X is true about the project, the disk and git and campsite logbook are the source of truth. Stale state in my own context window is a real bug, not a minor inefficiency."*

This is the apparatus-first discipline applied to project state — the substrate is more reliable than the model. The model is just more convenient.

Apparatus-first is the same principle applied to investigation: when an interpretive model of a finding conflicts with what the apparatus is actually producing, **investigate the apparatus**. The internal model of "what MSVC libm would do at 2^1000" can be plausible-but-wrong. The external check is the apparatus, not the explanation.

> *Substrate over memory* (CLAUDE.md) :: *Apparatus over narrative* (this doc).
> Both: the substrate is more reliable than the model. The model is just more convenient.

---

## The discipline

**When a finding is too dramatic to be true, investigate the apparatus first.**

"Too dramatic to be true" is rough but operational: when a finding would require a specific, large, unexpected failure in a widely-deployed, heavily-tested system (MSVC libm, Python's float parser, `serde_json`, the OS scheduler, Linux kernel timing) — **the prior should be high that the apparatus is wrong, not the system.**

Order of investigation, in order of decreasing prior-probability-of-being-the-culprit:

1. **Did the gold standard compute the right thing?** Re-run the oracle at higher precision. Use a different oracle. If two independent oracles agree on the "correct" answer and the system disagrees, *then* you have a real discrepancy. If they disagree with each other, the gold standard was the bug.
2. **Was the input to the system what you intended?** The bit-field check. Read the raw bytes of the input as the system received them, not as the input source claims they were. This is what would have caught the sin bug at step 2.
3. **Was the comparison done correctly?** ULP computation can fail near zero, near infinity, near sign-boundary, near the cut of an inverse function. Tolerance computation can use the wrong reference scale.
4. **Then investigate the algorithm.**

The sin bug would have been caught at step 2. Math-researcher went straight to step 4 and built a technically sophisticated wrong narrative. **The reasoning chain was sound; the input to the reasoning chain wasn't.**

---

## What made it catchable

A linter. Not detective work by anyone on the investigation chain — a mechanical tool that noticed `parse_f64_field` was being used where `parse_f64_bits_field` should be.

There's something humbling there. The wrong narrative was sophisticated enough that it survived investigator review. It was a tool check, not a reasoning check, that caught it. Which is an argument for **more mechanical checks on apparatus assumptions**, not just "do the tests pass."

> "Are the assumptions in the test setup auditable?" is a different question from "do the tests pass?"

A test passing means the apparatus + the algorithm + the comparison all agreed. It does not mean the apparatus was *right*. If the apparatus loads garbage and the algorithm processes garbage and they happen to agree on the garbage, the test passes. The lint catches mismatches between apparatus assumptions and apparatus implementation; the test only catches algorithm output disagreements with oracle output.

---

## The proposed tooling: oracle-loader round-trip assertions

Every oracle corpus loader should have an explicit assertion that a known bit-pattern round-trips correctly through the loader. Not just *"does the loader run without panicking,"* but *"does `load(corpus); verify_known_entry_bit_exact()` pass."*

Concrete shape:

```rust
#[cfg(test)]
mod corpus_loader_apparatus {
    use super::*;

    /// Apparatus check: known input bit-patterns round-trip through the
    /// loader bit-exact. If this fails, the loader is corrupting inputs
    /// before the algorithm under test sees them.
    ///
    /// This is the apparatus-first antibody (see
    /// `R:\winrapids\docs\architecture\confident-wrong-narratives.md`):
    /// any "dramatic finding" reported by tests using this corpus may
    /// be apparatus failure, not algorithm failure. This test catches
    /// apparatus failure independently.
    #[test]
    fn loader_round_trips_known_bit_patterns() {
        let corpus = load_corpus("test-fixtures/known-bit-patterns.json");

        // For each known entry, the loader's output bit-pattern must
        // equal the entry's stated bit-pattern field.
        for entry in &corpus {
            let loaded_bits = entry.x.to_bits();
            let stated_bits = entry.x_bits_hex_field;
            assert_eq!(
                loaded_bits, stated_bits,
                "loader corrupted bit pattern: stated = 0x{:016x}, \
                 loaded = 0x{:016x}, decimal source = {:?}",
                stated_bits, loaded_bits, entry.x_decimal_field
            );
        }
    }
}
```

The test fixture (`known-bit-patterns.json`) contains entries whose hex bit-pattern field is hand-asserted, including:
- Tame values (1.0, 2.0, π, e — round-trip is trivially fine)
- Subnormals
- Values near the f64 limits (largest finite, smallest positive normal)
- Values with extreme exponents (the sin-2^1000 case — verify both `as_f64` parser path and `parse_f64_bits` path)
- NaN with non-zero payload
- Sign-bit edge cases (signed zeros, negative subnormals)

The cost of adding this lint is small (one test per oracle); the benefit is catching apparatus drift before it produces sophisticated wrong narratives. **Apparatus integrity becomes auditable rather than assumed.**

---

## What this does NOT mean

- **Not "always doubt the system."** When apparatus checks pass, real bugs in well-deployed systems are *also* possible. The discipline isn't "assume the apparatus is wrong" — it's "check the apparatus *first* before constructing the narrative." Once the apparatus is known good, sophisticated explanations about real algorithm bugs are the right move.
- **Not "domain knowledge is a liability."** Domain knowledge is what lets you write `parse_f64_bits_field` in the first place. The discipline is about *when* the knowledge gets deployed in the investigation order, not whether to use it at all.
- **Not a replacement for the two-axis taxonomy.** TESTING.md's Axis-1 / Axis-2 distinction is the diagnostic frame; this doc names a specific *bias* in how investigators traverse that frame. Both apply.

---

## Cross-references

- `R:\tambear\docs\TESTING.md` §"Two-axis test quality" — the diagnostic frame this doc supplements
- `R:\winrapids\CLAUDE.md` §"Substrate over memory" — the structurally-identical principle applied to project state
- `R:\winrapids\docs\architecture\holonomic-architecture.md` — written the same day; treats the cache-discipline question with the same "the substrate already had it; what was missing was the test" logic
- Math-researcher's triage doc: `R:\tambear\oracle\sin\disagreements\20260509-harness-bug-as_f64-loses-lsb-at-large-magnitudes.md` — the lesson record for the specific sin-2^1000 episode
- Navigator's analysis: `~/.claude/garden/2026-05/2026-05-09-confident-wrong-narratives.md` — the garden essay this doc memorializes
