# Expedition Final State — All Measurements Resolved

Created: 2026-03-29T23:47:47-05:00
By: navigator

---

*Supersedes the earlier synthesis note (20260329233416). All measurements now complete. E03b resolved the last open question.*

---

## All Questions Answered

### Does the compiler vision work?

**Yes. Robustly.**

| Question | Answer | Evidence |
|----------|--------|----------|
| Does multi-output reduce save HBM reads? | 1.3-1.6x (L2 hides most at FinTek sizes) | E01 |
| Does sort-once-use-many help? | 1.3-1.7x; CuPy has no sort cache | E02 |
| Does cross-algorithm sharing help? | ALWAYS at FinTek sizes (2x); crossover ~500K | E03b |
| Is fusion right or wrong? | Size-adaptive: fuse below ~500K, compose above ~1M | E03b |
| What mechanism drives fusion benefit? | CuPy dispatch cost (10-27us/call), not bandwidth | E03b |
| Is provenance reuse real? | 81x-865x savings; 28x farm speedup at 1% dirty | E07 |
| Are GPU-resident queries fast? | 26x cold->warm; L2 makes anything under 40MB near-free | E06 |
| Does cudarc work on Windows? | Yes, zero issues, sm_120 Blackwell confirmed | E08 |
| Is Rust NVRTC faster than CuPy? | 2-4x compilation; 7.7x kernel launch overhead | E09 |
| Is JIT enough, or need pre-built? | JIT + disk cache = pre-built; two tiers not three | E05+E09 |

### The Corrected Architecture

**Shared intermediates — always.** CSE on the primitive DAG. 1.2-1.5x at all sizes, no crossover.

**Element-wise fused kernels — always.** Any expression tree that's element-independent fuses. 2.3x from E05. No crossover.

**Cross-operation fusion — size-adaptive.** Below ~500K rows: fuse (2x from dispatch reduction). Above ~1M: compose CuPy primitives (bandwidth wins). Rust's 9us launch shifts crossover to ~5M+.

**Two execution tiers.** JIT + disk cache = pre-built. ~22ms first run (Rust). Cached thereafter. 100 kernels = 2 seconds total warmup.

**Rust is the dispatch layer.** 7.7x lower launch overhead. Eliminates CuPy tax at farm scale.

### Registry Design (E04 Entry Point)

```
specialist: <name>
primitive_dag: [named outputs enable CSE]
fusion_eligible: bool
fusion_crossover_rows: int  (default 500_000)
independent: bool  (scan/sort are false; fused_expr/gather/scatter are true)
```

E04 prototype: 2-3 specialists sharing a cumsum, full loop spec → DAG → CSE → codegen → NVRTC → launch → verify.

E10: full 135-specialist registry. After E04 proves the loop.

