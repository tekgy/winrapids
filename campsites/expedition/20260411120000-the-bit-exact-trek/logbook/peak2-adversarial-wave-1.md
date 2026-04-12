# Logbook — Peak 2: Adversarial Wave 1

**Campsites:** adversarial pre-code reviews for exp, log, sin/cos, pow, atan, tan, hyperbolic  
**Date:** 2026-04-12  
**Role:** Adversarial Mathematician  
**Status:** All seven reviews complete. All seven campsites held pending math-researcher amendments.

---

## What almost went wrong

### The cascade trap

Navigator asked whether the cosh B2 error (using `1/e_x` instead of `exp(-x)`) cascades into tanh. The natural adversarial move is to trace the call graph and say "yes, if cosh is wrong, everything that calls cosh is wrong." I started writing that and then stopped — checked whether tanh actually calls cosh. It doesn't. The design doc explicitly documents this (pitfall 6 in hyperbolic-design.md).

If I'd assumed cascade without checking, I'd have filed a false B4 against tanh for a bug that doesn't exist. Instead, checking the call graph correctly revealed the real B4: tanh's own medium-regime formula has an independent ~2 ULP error. The right bug, found for the right reason.

**Temptation off-path:** assume structural cascade without reading the actual code path.

### The signed-zero pattern emerging late

I found the signed-zero bug in exp first (B1: `exp(-inf)` → `−0` risk). Then again in log. Then sin/cos. By the fourth function I recognized it as a systematic pattern, not a function-specific accident. But I almost didn't name it explicitly — it would have been easy to just keep filing B1 entries one by one without noting that they're all the same bug.

The convergence check at session end forced me to name it. The naming matters: "signed-zero early-return pattern" is a design invariant violation, not seven independent bugs. A named pattern can become a checklist item. Seven unnamed instances just get fixed one at a time.

**Temptation off-path:** log individual instances without noticing the structural pattern.

### Oracle format negotiation over multiple rounds

The oracle TOML format went through three rounds: initial proposal (separate `[[constraint_checks]]` section), scientist's implementation in 269a338 (`BitExactConstraint::NonzeroSubnormal` inside `bit_exact_checks`), then scientist's d4141ec (three-section format with `NamedConstraint` enum, `constraint_checks` as its own section).

I updated the TOML twice — once for 269a338, once for d4141ec — and the messages crossed with scientist's updates in between. The TOML was briefly in an inconsistent state (had stale N1 referencing the intermediate design). The lesson: the oracle TOML is a shared artifact between adversarial and scientist. Both sides touching it simultaneously creates collision risk. In future sessions, establish ownership: adversarial writes the corpus and checks, scientist updates the notes and format-level changes.

**Temptation off-path:** assume the file is stable between sessions; don't check before editing.

### The overflow ULP gap

I asked scientist about overflow boundary handling (inputs not in reference binary → ULP path can't enforce sign). Scientist confirmed the gap exists. I then added `bit_exact_checks` entries before scientist's response fully arrived explaining they wanted `constraint_checks infinite_positive` instead. Had to revert and re-add.

The gap itself was real and worth catching — the oracle was testing overflow-boundary inputs via the injection-set path but getting ULP=0 (self-referential: candidate compared against itself). The `constraint_checks` path is the right fix. The churn was from moving faster than the message round-trip.

**Temptation off-path:** act immediately on a question before the answer arrives; accept first partial answer as final.

---

## What worked

The error analysis for tanh B4 worked correctly. The exact derivation: at `x = 0.55`, `exp(1.1)` has 1-ULP relative error, the division compounds by ~0.5 ULP, the subtraction `1.0 - q` has no cancellation but passes the absolute error through with amplification factor `≈ 1/q ≈ 2`, total ~2 ULP. Navigator accepted this analysis and made a clean ruling (two-call formula). Clean adversarial → navigator → math-researcher routing.

The VB-004 entry (PTX `min.f64` drops NaN by default) was the right call to promote from a footnote in VB-001 to a first-class entry. VB-001 mentioned it in the upstream-report field as "separate issue"; VB-004 documents it with the full workaround including the PTX ISA 7.5+ version boundary and the composed fallback for older targets.

---

## Blocker count at session end

| Function | Open blockers | Notes |
|---|---|---|
| exp | B1-B5 | Campsite 2.6 held |
| log | B1-B3 | Campsite 2.9 held |
| sin/cos | B1-B2 | Campsite 2.12 held |
| pow | B1, B4 | B2+B3 resolved in pow-design.md amendments |
| atan | B1-B3 | Campsite 2.15 held |
| tan | B1-B3 | New this session; campsite 2.20 held |
| hyperbolic | B1-B3, B4 ruled | B4: two-call formula; campsite 2.18 held |

Total: ~20 blockers across 7 functions. Zero implementations started. All holds appropriate.

---

## Next adversarial session pickup

1. Wait for math-researcher amendments on all seven design docs
2. Review each amendment: verify signed-zero fixes use `return x`, precision derivations present and correct, threshold justifications checkable
3. When each clears, confirm to navigator — campsite unblocks
4. Begin oracle TOML for each function as implementation lands (tam_sqrt already running; tam_ln next)
5. Check fabs.f64 IR op confirmation from pathmaker before any implementation that uses `|x|`

The convergence check writeup is at `peak2-libm/adversarial-convergence-check.md`. The proposed pre-adversarial checklist (C1-C5 from the convergence) should be added to the math-researcher campsite template before any new design doc is written.
