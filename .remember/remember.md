# Handoff

## State
At `ef557ff`. New authoritative `tambear/` workspace created with 7 crates. TBS crate migrated (35 tests green, zero deps). tambear-primitives restructured to transforms×accumulates×gathers (32 tests). tambear-tam compiles recipes to fused single-pass execution (10 tests). TBS Expr AST is THE universal expression type — used for transforms, gathers, and scripts. 2,335 tests in old tambear crate still green.

## Next
1. Continue workspace migration — frame/ needs GPU buffer split (pure CPU types vs GPU containers). Then primitives/, tam/, session/.
2. Wire TBS Expr into accumulates (replace hardcoded Transform enum) and gathers (replace GatherComputation enum) — ONE expression type everywhere.
3. Build more recipes using the new architecture — the moment family (variance, skewness, kurtosis, pearson_r) is partially declared, needs TAM executor wiring.

## Context
- `tambear/` workspace is THE authoritative home. `crates/tambear/` is the old location — reference only, migrate from.
- `MIGRATION.md` maps every file from old → new location.
- Frame has tam_gpu baked in (GPU buffers in Column/GroupIndex). Decision needed: feature-gate or split.
- Semiring trait + 6 instances exist in both old accumulate.rs and new tambear-primitives. Consolidate during migration.
- Recipe system proven: 5 recipes fuse to 1 data pass. Sharing is structural, not runtime.
- TBS Expr replaces Expr::Custom(&str) — no vendor language strings anywhere.
