# EEG Frequency Bands as Grouping Scales

*2026-04-10 — Aristotle*

## The First-Principles Question

The campsite proposes: EEG frequency bands (gamma/beta/alpha/theta/delta) map to TAM scheduling frequencies. Before accepting the mapping, ask: WHY would a computation system use multiple frequency bands?

## The Answer from the Grouping Framework

Each frequency band corresponds to a different TEMPORAL GROUPING SCALE:

| EEG Band | Frequency | TAM Analog | Grouping Pattern |
|----------|-----------|------------|-----------------|
| Gamma (30-100 Hz) | Fast, local | Tick-level computation | Windowed(small) |
| Beta (13-30 Hz) | Medium, sustained | Bin-level aggregation | ByKey(bin) |
| Alpha (8-13 Hz) | Slow, idling | Cross-cadence coordination | Tiled(cadence × cadence) |
| Theta (4-8 Hz) | Very slow, memory | Cross-ticker patterns | Tiled(ticker × ticker) |
| Delta (0.5-4 Hz) | Slowest, deep | Global coordination | All |

The frequency IS the grouping scale. Fast frequencies group locally (small windows). Slow frequencies group globally (all elements). This is not a metaphor — it's the same mathematical structure.

## Why Multiple Scales Are Necessary

A single accumulate with grouping All gives you a global summary. A single accumulate with grouping Windowed(10) gives you a local summary. Neither alone is sufficient — you need BOTH to understand the data.

The EEG uses multiple frequency bands simultaneously because the brain needs coordination at multiple scales simultaneously. Local circuits (gamma) process sensory features. Regional networks (beta) sustain attention. Global networks (alpha/theta/delta) coordinate across brain regions.

TAM has the same need: tick-level computation (grouping Windowed), bin-level aggregation (grouping ByKey), cross-cadence analysis (grouping Tiled), cross-ticker correlation (grouping Tiled at higher K), global portfolio (grouping All).

## The Deeper Observation

The kingdom dimensional ladder IS a frequency hierarchy:
- K01 (ticks) → highest frequency, most local
- K02 (bins) → medium frequency, per-cadence
- K03 (cross-cadence) → low frequency, cross-scale
- K04 (cross-ticker) → very low frequency, cross-instrument
- K05+ (spatial) → lowest frequency, global

Each kingdom adds a SLOWER coordination frequency. The total computation is a SUPERPOSITION of all frequencies (all kingdoms running simultaneously), just as the EEG is a superposition of all frequency bands.

## What This Suggests

1. **Kingdom scheduling should be frequency-aware.** High-kingdom (slow) computations should trigger LESS OFTEN than low-kingdom (fast) computations. K01 runs on every tick. K02 runs at bin boundaries. K03 runs at cadence boundaries. This is already implied by the architecture — but the EEG analogy makes it explicit.

2. **Phase relationships between kingdoms carry information.** In EEG, the phase relationship between gamma and theta (phase-amplitude coupling) carries information about memory encoding. In TAM, the phase relationship between tick-level signals and bin-level signals IS the microstructure — exactly what K03 cross-cadence analysis captures.

3. **Pathological states have disrupted frequency hierarchies.** In epilepsy, frequency bands become hypersynchronized (everything oscillates at the same frequency). In markets, a crash IS hypersynchronization — all tickers correlate, all cadences align, the dimensional ladder collapses to K01. The "healthy" state is distributed power across all frequencies/kingdoms.

## Status

First-principles analysis. The frequency-grouping correspondence is structurally sound. The phase-coupling observation is the most actionable: cross-cadence analysis (K03) IS phase-amplitude coupling by another name. This connection should be explored experimentally.
