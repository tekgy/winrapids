# PLEASE READ — Brief from the GPU Verifier Port team

*From: a JBD tambear team that spawned 2026-04-25 to port Odrzywolek's elementary-functions-substrate verifier to GPU.*
*To: whichever Claude (or human) is working in winrapids.*
*Why this matters to you: the team independently re-derived several concepts that could shape tambear's TamSession, multi-hardware dispatch, and Op canonicalization design — using vocabulary that hasn't been part of tambear yet.*

## TL;DR

We spent 90+ minutes (per agent) studying a non-tambear verifier (Odrzywolek's `verify_base_set` Rust code, 2782 lines, plus the 26-page Supplementary Information of his arXiv:2603.21852v2 paper). The team — pathmaker, navigator, scout, naturalist, observer, math-researcher, adversarial, scientist, aristotle — independently arrived at three architectural reframings:

1. **The verifier is fundamentally a substrate cache, not a one-shot verifier.** The autogen workload is a query stream; per-query cost should be O(1) against a precomputed cache. Aristotle's "Aristotelian move."
2. **Multi-hardware dispatch is a TAM-the-observer recomposition pattern.** Per-level expansion shards naturally across (a, b) pair partitions; sync points merge local caches into global via dedup. Maps directly to tambear's planned multi-card sync.
3. **Algebraic-rewrite sieve is what we're calling Op canonicalization.** Pruning syntactically-redundant expressions before they hit the cache reduces memory pressure 5-10× by collapsing equivalence classes to canonical forms.

Worth knowing about regardless of whether you adopt the framings — these are concepts the verifier team needed to invent to make their work tractable, and they map cleanly onto problems tambear is solving anyway.

## The new vocabulary

| Term | Meaning |
|---|---|
| **Substrate cache** | The set of distinct values reachable from {substrate, K} levels, deduplicated by content. NOT the same as the substrate itself; it's the *enumerated reachable set*. |
| **TAM-the-observer recomposing** | The merge step where multiple hardware shards' local caches combine into a global cache. The "observer" is global; the workers are local. |
| **Op canonicalization sieve** | Pruning syntactically-distinct expressions that compute the same value, using known algebraic identities. Reduces cache size before storage, not just deduplicating after compute. |
| **Two-tier verification** (hybrid precision) | The pattern in the existing CUDA EmL_recognizer: fp32 GPU sieve to identify candidates, fp64 CPU verification for survivors. Two-stage instead of one-stage end-to-end. |
| **qkey** (their term) | A rounded-value hash key for value-level deduplication. Tambear's analog: content-addressable provenance hash for cached intermediates. |
| **Branch-aware mode vs discovery mode** | The verifier has two output modes for transcendental functions with branch cuts. A `using()` knob for which set of conventions applies. |

## Mapping table — verifier port concepts → tambear concepts

| GPU verifier port (independently invented) | Tambear (already in design) |
|---|---|
| `enumerate_substrate_levels(B, K) → SubstrateCache` | TamSession populated by repeated `accumulate(expr, op)` calls |
| `cache.contains(value) → Option<witness>` | `gather(method)` lookup with provenance |
| Per-level expansion: combine `(a, b) ∈ Lᵢ × Lⱼ` via op | `accumulate` over a Cartesian-product partition |
| Multi-hardware shard + dedup merge | TAM syncing across cards before next command |
| qkey-based value-level dedup | Provenance-hash content-addressing |
| Algebraic-rewrite sieve | Op canonicalization (which two syntactic forms are *the same* op invocation) |
| `using(branch_aware: bool)` | `using(precision: ...)` runtime config |
| Bit-exact CPU↔GPU contract | Tambear's bit-exact determinism contract (DEC-N, whichever number) |
| Long-tail per-candidate verification cost | Tambear's known issue: a few recipes have orders-of-magnitude more cost than the median; affects scheduler design |

## Specific architectural insights from the team

These aren't tambear-specific yet, but the patterns might apply.

**(1) The dedupe strategy IS the architectural commitment** — naturalist's finding. Framework choice (cubecl vs nvcc-FFI vs cudarc) is *downstream* of dedupe choice. Two real options:
- **Bitvector + radix sort** — references arXiv:2504.18943 (April 2026), GPU bottom-up program synthesis at 681M formulae/sec on A100
- **Concurrent GPU hash map** — NVIDIA cuCollections, 87.5 GB/s insert on H100

Tambear's TamSession dedup faces the same fork. The scientific and engineering literature for this specific decision is genuinely current and worth surveying.

**(2) Inner DP is HashMap-bound and barrier-synchronous** — the `can_represent` value-BFS evolves a HashMap state-by-state. Each step depends on the previous step's hash insertions. **Cannot parallelize within a single candidate without redesigning the algorithm.** Wall time is bounded by max-single-candidate cost, not average. Tambear's analog: any recipe with HashMap-shaped intermediate state will face the same ceiling.

**(3) Strict verification is much stricter than numerical-sieve discovery.** The Rust port's verifier rejects ~95% of candidates the SI's numerical sieve accepts. The Rust adds three safeguards the original Mathematica didn't: 8-probe cross-check (not 1), mpmath 128-digit recheck, flaky-witness detector. **Implication for tambear**: when a recipe's "validation" is built, decide explicitly whether it's the discovery-tier sieve or the verification-tier strict check. They should NOT be the same.

**(4) Branch-cut conventions are a real distinction, not a "details" matter.** `ln(-1)` returning `+iπ` vs `-iπ` flips downstream identities silently. The verifier has explicit `--branch-aware-witness` mode for this. Tambear's complex math primitives need an analog `using(branch: ...)` runtime knob if they touch transcendentals on negative reals.

**(5) The 6-binary-Sheffer-class landscape we thought existed (EML, EDL, −eml, the new `{1, 1/(ln x)^y}` class, LDE, LDE-0) is not what survives strict verification.** Most are sieve artifacts. Implication: when tambear claims "we enumerated all recipes of class X," be explicit about whether that's discovery-tier or verification-tier enumeration. The numbers will differ by an order of magnitude.

**(6) Hardware: GPU is sm_122 (Compute Capability 12.2)**, NOT sm_100. The Blackwell family splits into 12.x (workstation) and 10.x (datacenter) branches that don't cross-compile. Tambear's GPU code targets need this distinction explicit. Use `-arch=sm_122` for RTX PRO 6000 Workstation; `-arch=sm_100` (or sm_101/102) for datacenter Blackwell.

## What I'd do with this if I were on winrapids

1. **Adopt the substrate-cache framing** in TamSession's design language, regardless of how you implement it. Aristotle's `enumerate_substrate_levels(B, K) → SubstrateCache` + `cache.contains(value) → Option<witness>` is a clean external API even if internals are different. Worth at minimum considering as the canonical TamSession interface.
2. **Choose dedupe strategy explicitly** before deciding on framework. Bitvector+radix-sort vs cuCollections is a structural commitment that ripples through everything downstream. Don't let this be implicit.
3. **Build branch-mode `using()` early.** Tambear is going to hit this on the first transcendental recipe. Not having it means silent wrong answers.
4. **Distinguish discovery-tier vs verification-tier** in the recipe taxonomy. Two different epistemic standards; deliberately confusing them is the failure mode of the field.
5. **Keep the verifier-port architecture in mind as a reference.** When tambear tooling stabilizes, the GPU verifier could become a true tambear application — substrate cache as TamSession, dispatch as TAM, sieve as Op canonicalization. That's a real-world validation target for tambear that doesn't exist anywhere else.

## Pointers to source materials

- **The team's campsite**: `~/repos/SymbolicRegressionPackage/campsites/gpu-port/20260425205904-verify-base-set/`
  - `team-briefing.md` — original team brief
  - `aristotle/` — 8 phases of first-principles deconstruction (the substrate-cache reframing is in Phase 5 + Phase 6-7)
  - `naturalist/` — two-paths-around-HashMap analysis (the dedupe-strategy-is-the-architectural-commitment finding)
  - `scout/` — framework comparison (direct nvcc + Rust FFI recommended)
  - `math-researcher/` — verifier semantics specification (the canonical reference)
  - `adversarial/` — 14 failure modes the GPU port must handle (bit-exact equivalence test gauntlet)
  - `scientist/` — validation harness + canonical baselines (the strict-verifier ground truth)
  - `observer/` — risk register + qkey-rounding-discrepancy finding
  - `navigator/` — synthesis (when finished; in-flight at the time this was written)
  - `pathmaker/` — implementation (when finished; in-flight at the time this was written)
- **The reading notebook on the EML paper**: `~/.claude/garden/reading-notebooks/odrzywolek-eml-luca-elementary-functions.md` — chapter-by-chapter digest of the paper + SI, including the structural-rhyme analysis with tambear
- **The substrate-finding research project**: `~/.claude/chris/projects/sheffer-form.md` — the meta-hypothesis about minimal universal substrates having `(dual pair) + (asymmetric op)` form, with a verified-vs-sieve correction in Update #2
- **The repo we cloned**: `~/repos/SymbolicRegressionPackage/` — built and operational on this machine; rust_verify completes the 36-primitive bootstrap chain in 1.94 seconds

## Closing

We're not asking you to do anything specific. The brief is informational — vocabulary and patterns from a sibling team that you can use, ignore, or refine.

The convergence is the signal: nine agents working on a related-but-different problem (substrate-search verification, not tambear itself) independently arrived at concepts that map cleanly onto tambear's design space. That convergence is evidence the concepts are correct, not just team-specific framings.

If you want to talk to that team, the campsite logbook has full role-folder transcripts. If you want to surface back to us, drop a note in the campsite or open an issue in the chris vault's `sheffer-form.md` project file.

— Brief drafted by parent Claude (Opus 4.7) on 2026-04-26 after the JBD GPU verifier port team's findings landed. On behalf of Tekgy.
