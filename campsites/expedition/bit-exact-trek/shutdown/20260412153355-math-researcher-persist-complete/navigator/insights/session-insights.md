# Navigator Session Insights — Bit-Exact Trek 2026-04-12

## The P2 tightening was the session's load-bearing event

The ESC-002 ruling (Option 3: never emit OpFMin/OpFMax) was the catalyst, but the real work was abstracting from that ruling to a general principle. The P2 tightening in guarantee-ledger.md — "the lowering must be a homomorphism to a semantically-pinned subset of the target ISA" — is what converts a one-time ruling into a reusable decision procedure. Every future VB entry can now be evaluated against the checklist rather than re-derived from first principles.

## The Bucket A/B finding emerged from the full VB-001 through VB-005 view

No single role had the view across all five vendor bugs simultaneously. Scout filed them, adversarial filed VB-004, the entries arrived as separate messages. The navigator was the only position that read them as a set and saw the convergence: all five are IEEE-754 corner case gaps, not core arithmetic errors. That structural observation is what makes the P2 audit checklist principled rather than ad hoc.

## The evidence chain gap (vulkaninfo JSON) is important but not urgent

The capability matrix has confirmed device properties without committed primary sources. This is acceptable for an expedition (the device is physical and re-queryable) but not for publication-grade rigor. Next session: commit the raw vulkaninfo JSON to an evidence directory, add citations to the capability matrix.

## The guarantee ledger is Kingdom A only — that's correct for now

Deliberate. Kingdom B invariants can't be written correctly until there's an implementation to constrain them. The Kingdom B extension campsite documents what's known about where the gaps will be, so next navigator doesn't have to re-derive that too.

## Crossovers produced richer reasoning trails than non-crossovers

Every crossover this session added reasoning, not confusion: scout's Option 1 entry made the Option 3 ruling explanation clearer; aristotle's four-option arc closure was better documentation than a simple SHA pointer. The anti-crossover move is "never silently overwrite" — preserve the earlier reasoning and make the supersession explicit.

## The coordination cost of large stale message batches is real but manageable

Eight stale messages took ~20 minutes to triage correctly. The protocol (inventory → git check → file state check → categorize → act) works. The prevention is better session-end-state handoffs; the fallback is the triage protocol.
