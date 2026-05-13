# entropies.md — superseded; canonical lives in tambear

**Status (as of 2026-05-12, verified by scout via `R:\tambear\docs\architecture\recipe-trees\entropies.md`)**: this file is a stale parallel draft. The canonical entropies.md is at:

  **`R:\tambear\docs\architecture\recipe-trees\entropies.md`**

drafted by scout 2026-05-12 during the recipe-trees migration to tambear.

## Why this file exists

The naturalist (sweep-37, 2026-05-12) drafted an entropies.md at this winrapids path without first verifying which directory was canonical. Scout had already migrated the recipe-trees to tambear earlier the same session and had drafted the canonical entropies.md there. The naturalist's draft contained two specific errors:

1. **Topology label**: claimed "disjoint" — scout's canonical tambear version correctly labels it "clustered-with-bridge" (PermutationEntropy and SpectralEntropy reuse DiscreteEntropy; they're compositions, not new kernels).

2. **Kingdom claim about SampEn**: claimed "Kingdom A with NaivePairs grouping (every pair's distance test is independent)." This was speculative documentation — *the current tambear `accumulate.rs::Grouping` enum (verified by scout 2026-05-12) is `All, ByKey, Prefix, Segmented, Windowed, Tiled, Graph, Probabilistic` — NaivePairs/AllPairs does NOT exist*. SampEn at the O(n²) naive form is Kingdom B in current tambear. A future Grouping extension (NaivePairs / AllPairs) WOULD lift it to Kingdom A, but that's intended-future-state, not current-state.

The naturalist's findings that *are* correct and preserved in scout's canonical version:
- Three kernels for distribution / continuous-density / time-series-pattern domains
- Rényi-α spectrum as the canonical first customer of Phase D-prime sweep/superposition infrastructure
- Connection to past-naturalist's 2026-04-01 `renyi-spectrum-as-gaussianity-probe.md` (Pattern 13: cross-resolution convergence)
- The literature catalog of ~25 named leaves and the four-axis recipe metadata

For the corrected, ratification-ready entropies tree, **read `R:\tambear\docs\architecture\recipe-trees\entropies.md`**.

## What this file's existence demonstrates

The naturalist's failure mode here is a recursive instance of the parent pattern named in `~/.claude/garden/2026-05-12-the-document-and-its-substrate.md` ("Documentation decay is structurally invisible"):

- **Audit hallucination sub-shape**: the naturalist wrote at a path assumed to be canonical (winrapids/recipe-trees/) without verifying the substrate state had migrated to tambear earlier the same session.
- **Speculative documentation sub-shape**: NaivePairs grouping was described in present tense as if it existed in tambear's accumulate.rs; it does not. The intended-future-state was described as current-state.

Scout's grep against `R:\tambear\crates\tambear\src\accumulate.rs` caught both. The discipline that named the parent pattern is the discipline that catches the violation. Pattern 12 (grep validates the abstraction) operative on the naturalist's own draft, three hours after the naturalist named Pattern 16 (documentation decay is structurally invisible) at the methodology-doc tier.

The honest archival record stays here as substrate trail: someone in a future session reading this file (or the campsite scout's messages, or the navigator routing notes) sees the failure mode named and the correction routed. The garden entry at `~/.claude/garden/2026-05-12-the-document-and-its-substrate.md` is the methodology naming; this file is the contemporary instance.

— naturalist, journey-team, 2026-05-12, with scout's correction folded in
