# WinRapids

Windows-native GPU-accelerated data science toolkit. Market Atlas signal farm.

---

## What This System IS

The market is a temporospatial system. Not a prediction problem. Not a game to win. A SYSTEM — with properties that can be described, measured, and farmed as signal.

The goal: build the instrument that describes the system. Farm every measurable signal. Let the system speak. When it has enough dimensional resolution, it stops being a prediction problem — you observe from close enough to the true structure that the behavior becomes legible.

The kingdoms are a dimensional ladder. Each kingdom adds one tensor cross-axis. K01 is 1D (ticks). K02 is 2D (bins). K03 is 3D (cross-cadence). K04 is 4D (cross-ticker). Higher kingdoms add spatial axes — the fault map, the phase space topology, the learned metric space where market coupling lives geometrically.

Traditional signals are 2D projections of higher-dimensional structure. We're building the projection apparatus to climb the dimensional ladder until the structure resolves.

---

## Irrevocable Architectural Principles

These are not preferences. They are enforced by the architecture. Violating them produces systems that create more problems than they solve. Every prior attempt learned this.

### node === node

Every node has identical structure regardless of scale — trunk, branch, leaf, all the same. Enforced by the CLI scaffold generator: you cannot build a half-node. The generated node is fully wired before any math is written.

**Why irrevocable**: node===node is what makes the agent-as-researcher pattern work. Give an agent the scaffold as its only context — it has everything to build novel math and have it be production-ready on arrival. No integration work. The architecture is the guarantee.

**In practice**: Generate the scaffold first. Get it running (even with `log()`). THEN write the math. Never build infrastructure by hand.

### No Tech Debt — Ever

See it, fix it. The codebase must be fully understandable at all times.

**Why irrevocable**: Tech debt in a complex system compounds. The cost of "fix it later" is not linear — it's multiplicative. One piece of debt makes the next piece harder to fix, which makes the next harder, until the system is incomprehensible. This has happened before. The zero-debt rule is what keeps the system explorable.

### Anti-YAGNI

If the principles structurally guarantee we'll need something, build it now. The question is not "do we need this today?" but "does the structure guarantee we'll need it?"

**Why irrevocable**: This system has proven empirically, repeatedly, that over-specification creates serendipitous value at every turn — things needed but not yet anticipated emerge for free. Under-specification means a format version bump later, a migration, a debt. The 64 custom compression slots cost nothing (header is fixed at 4096 bytes). Leaving them out costs a format change when we need slot 65.

**The reflex to fight**: Both human and AI are trained toward YAGNI and "minimum viable." That reflex is almost always wrong in this system. When you feel it: ask "does the structure guarantee we'll need this?" If yes, build it.

### Raw Ticks Always — No Derived Cadences

Every cadence (1s, 5s, 30s, 1min...) is computed directly from raw ticks. Never from a coarser cadence.

**Why irrevocable**: Computing 5-minute bins from 1-minute bins is lossy. The microstructure visible in raw ticks is destroyed before you start. More importantly: computing from derived cadences destroys the symplectic structure — you're working in a deformed phase space. K03 cross-cadence is only meaningful if each cadence is an independent projection of the same raw reality. Derived cadences compare summaries of summaries. Independent raw-tick cadences compare bandwidth-orthogonal observations of the same signal.

### Run Everything — Never Gate Production

The signal farm computes every leaf for every ticker, every day. V columns carry confidence metadata. Consumers decide what to trust.

**Why irrevocable**: Research findings change. Today's "unstable" signal might be stable under conditions we haven't seen yet. If we gate production, we lose data we can't recover. The cost of computing an extra leaf is tiny (ms of GPU time). The cost of not having it when you need it is losing irreplaceable information. The pattern: DO columns = the data, V columns = the signal about the signal.

### Preallocation — No Migration Paths

Decide max size at design time. Fix the preallocation. If world changes break the allocation: delete and recompute. No backward compatibility.

**Why irrevocable**: Migration paths are tech debt with a time bomb. Preallocation enforces the invariant that the system is always in a known good state. If something no longer fits, the right answer is to recompute — not to stretch the allocation around old assumptions.

---

## Architecture Reference

**Kingdoms**: Numeric always (K01, K02...). Kingdom = tensor rank. Never mix kingdom and representation in the name.

**KIKO**: KI = input representation, KO = output representation. Orthogonal to kingdom. KO00=columnar, KO01=FFT, KO04=wavelet, KO05=sufficient stats, KO06=correlation, KO07=eigenvectors. Both filename and header carry KO code (belt-and-suspenders).

**All kingdoms farm**: No "research kingdoms." Research is a leaf lifecycle stage (experimental → prune | keep | production). Every leaf in every kingdom runs on real data.

**Leaf lifecycle**: Predefined (known from start, production immediately). Experimental (discovered, runs on real data, not yet promoted). Status is on the leaf, not the kingdom.

**MKTF / MKTC**: MKTF is the sub-file format (Block 0, sections, ByteEntry columns). MKTC is the pre-allocated container (MKTF blobs at fixed byte offsets). Per-dtype MKTF blobs within container (float32.mktf, int32.mktf etc.) — independent compression per dtype per leaf.

**V columns**: Every leaf with a confidence dimension produces V columns alongside DO columns. Never suppress the leaf because V is low — suppress consumption, not production.

---

## What Good Looks Like

- A new signal hypothesis → generate scaffold → running in pipeline → write math. Infrastructure is free.
- A researcher agent with only the scaffold template produces production-ready math.
- The header IS the manifest. Reading the header is reading the data's identity.
- Complexity that captures a real phenomenon IS elegance. Don't simplify it away.
- When something feels over-engineered: ask whether the structure guarantees we'll need it. Usually yes.
