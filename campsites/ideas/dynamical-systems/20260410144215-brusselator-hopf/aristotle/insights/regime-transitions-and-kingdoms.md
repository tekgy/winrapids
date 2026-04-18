<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# Regime Transitions, Hopf Bifurcations, and the Kingdom Boundary

*2026-04-10 — Aristotle*

## The First-Principles Question

The Brusselator campsite asks: does market regime transition have Hopf bifurcation structure? Before answering HOW, I want to ask WHY this question matters for tambear's architecture.

## The Connection to Kingdoms

A Hopf bifurcation is the point where a stable fixed point becomes an unstable spiral surrounded by a stable limit cycle. The system transitions from stationary (converges to a point) to oscillatory (converges to a cycle).

In kingdom terms:
- **Before the bifurcation** (stationary regime): the market's sufficient statistics are CONSTANT. A single accumulate pass captures them. Kingdom A — embarrassingly parallel.
- **At the bifurcation** (critical point): the system is at the edge of instability. Perturbations decay SLOWLY (critical slowing down). The autocorrelation length diverges. Kingdom B — sequential dependencies grow.
- **After the bifurcation** (oscillatory regime): the market oscillates with a characteristic frequency. The sufficient statistics are PERIODIC. A windowed accumulate captures them. Kingdom A again — but with a different grouping (Windowed instead of All).

**The bifurcation IS a kingdom transition.** The critical point is where the computation shifts from Kingdom A (stable → All grouping) through Kingdom B (critical → long-range Prefix dependencies) to Kingdom A again (oscillatory → Windowed grouping).

## What This Predicts

1. **Critical slowing down IS a kingdom boundary signal.** When ACF of volatility increases toward 1, the system is approaching a bifurcation. The Prefix scan (which computes running statistics with exponential decay) becomes LESS accurate because the decay assumption breaks down — the autocorrelation structure changes.

2. **The Fock boundary of market analysis IS the bifurcation point.** At the bifurcation, the system's future depends on which side of the bifurcation it's on, which depends on the current state's exact position, which is measured by the very statistics the bifurcation is disrupting. Self-reference.

3. **Detection before transition requires Kingdom B methods.** Prefix scan with slowly-varying parameters (adaptive alpha in EWMA). After transition, can return to Kingdom A with new parameters.

## The Deeper Question

Is the Brusselator the RIGHT model for market regime transitions, or is it just the simplest model with a Hopf bifurcation?

Markets have:
- Multiple coupled oscillators (sectors, asset classes)
- Noise-driven transitions (not parameter-driven like Brusselator)
- Asymmetric transitions (crashes are faster than rallies)

The Brusselator captures the TOPOLOGY of the bifurcation (stable → oscillatory) but not the DYNAMICS of real markets (noise-induced, asymmetric, coupled).

**Better models for first-principles analysis:**
- **Coupled oscillators** (Kuramoto model): captures synchronization/desynchronization between sectors. Phase transitions in coupling strength map to market contagion.
- **Stochastic bifurcation** (noise-induced transitions): markets transition between regimes via noise, not via smooth parameter change. The stochastic Hopf bifurcation has different critical behavior.
- **Excitable systems** (FitzHugh-Nagumo): captures the asymmetric response — markets sit in a stable state until perturbed past threshold, then fire (crash), then return to rest. This is more realistic than oscillatory.

## Recommendation

The Brusselator is a good starting point for TESTING the detection machinery (ACF of volatility, critical slowing down metrics). But the real question — "does market regime transition have bifurcation structure?" — should be explored with multiple models simultaneously (Brusselator, Kuramoto, FHN, stochastic Hopf). Run all four, compare the critical signatures, see which model's critical behavior matches empirical market data.

This is exactly a `.discover()` question: which dynamical model best describes the data? Don't choose — run all of them and let view_agreement tell you which models agree with each other and with the market.


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

