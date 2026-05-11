# Compaction Ghost Pattern — Navigator Note

Date: 2026-04-23
Last updated: 2026-04-23 (two-failure-mode correction after navigator-2 self-correction)

## Two failure modes — NOT one

Post-compaction agents produce confident reports pointing to wrong locations. But there are TWO distinct causes with different protocol fixes:

**Failure Mode A — True ghost (non-existent file anywhere)**: Agent cites a file/type that doesn't exist in any repo. The compacted context blended campsite markdown prose (design specs, planned types, pseudocode) with actual source code; the agent reports on the prose as if it were code. Observable: file not in any git tree; grep across correct repo returns nothing.

**Failure Mode B — Wrong-repo path confusion**: Agent cites a file that exists — but in a different repo than the one being searched. In WinRapids/tambear work: `R:/winrapids` and `R:/tambear` are separate git repos. Methods like `is_kernel_share_compatible_with`, `shape.rs`, `door.rs`, `sweep_8_r1015_attacks.rs` — these exist in `R:/tambear/crates/tambear/src/jit/`. A grep run from `R:/winrapids/crates/` returns zero hits. An agent (including navigator-2) may conclude "phantom" when the correct diagnosis is "wrong CWD."

**Navigator-2 made Mode B error 2026-04-23**: Ran grep in R:/winrapids, concluded phantom, sent false retraction messages to the full team, flagged sub-clause E evidence as phantom. Team-lead corrected. All 7 cited instances were real — in R:/tambear. The "sweep-8 jit domain is campsite-only" section below was WRONG; it described Mode B (wrong-repo) as Mode A (true ghost).

## Confirmed real (2026-04-23 correction)

The sweep-8 jit domain EXISTS as committed source — in R:/tambear:
- `R:/tambear/crates/tambear/src/jit/shape.rs` — real
- `R:/tambear/crates/tambear/src/jit/door.rs` — real
- `R:/tambear/crates/tambear/src/jit/jit_op.rs` — real
- `R:/tambear/crates/tambear/tests/sweep_8_r1015_attacks.rs` — real
- `is_kernel_share_compatible_with` at shape.rs:417 — real
- `is_result_share_compatible_with` at shape.rs:331 — real
- `DimHint`, `DeterminismClass`, `validate_dispatch` in door.rs — real
- All 7 sub-clause E/G instances cited in 069b46b and c20cadf — real

## True ghost instances this session (narrowed)

- Observer's commit hashes (330430f, 6a1154a etc.) — true ghosts (not in git log of either repo)
- Observer's moment_stats.rs filename (should be descriptive.rs) — Mode A ghost

Math-researcher's repeated shape.rs edits: possibly Mode B (editing wrong-repo path) rather than Mode A (editing non-existent file). Still unclear.

## Navigator protocol — verify BOTH failure modes

Before routing any finding that cites a file, type, or function:

1. Identify WHICH repo the work lives in:
   - Tambear source: `R:/tambear/crates/tambear/src/`
   - WinRapids crates: `R:/winrapids/crates/`
2. Run grep in the CORRECT repo: `grep -rn "<term>" /r/tambear/crates/ --include="*.rs" | grep -v target`
3. If zero hits in the correct repo → true ghost. If hits → real code, agent has correct information.
4. Before any edit: `pwd && git status` to confirm which repo tree is active.

**Do NOT run grep in R:/winrapids/crates/ to verify tambear source.** These are separate repos.

## What to tell affected agents

1. "Before concluding phantom: which repo does this file live in?"
2. "Run grep in R:/tambear/crates/, not R:/winrapids/crates/, for tambear source."
3. "If zero hits in both repos: true ghost — deposit mathematical analysis in the jit-design campsite."
4. "If hits in R:/tambear: the file is real, you were in the wrong working directory."

## Campsite is ground truth for design

The campsite is persistent and correct for design intent.
The grep of the CORRECT repo is ground truth for code existence.
When in doubt: grep the RIGHT source tree. Trust that grep.
