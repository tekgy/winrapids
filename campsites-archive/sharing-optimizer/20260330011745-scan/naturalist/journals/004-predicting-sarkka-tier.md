# Predicting SarkkaOp's Performance Tier

*Naturalist journal — 2026-03-30*

---

## The prediction

Entry 020 established two performance tiers for scan operators:
- **Tier 1 (simple)**: adds/muls only → ~40μs p01 (AddOp 38μs, KalmanAffineOp 39μs, CubicMomentsOp 42μs)
- **Tier 2 (complex)**: division + branching → ~100μs p01 (WelfordOp 99μs, KalmanOp 103μs)

SarkkaOp (the full Särkkä 5-tuple) has:
- State: 40B (5 doubles — largest yet)
- Combine: 1 division for the denominator, rest is multiplies + adds
- No branching

It's the first operator with division but WITHOUT branching in the combine. This isolates a new variable: is it the division or the branching (or both together) that causes the 2.5x penalty?

**My prediction**: SarkkaOp lands at **50-70μs p01**.

Reasoning:
- State size has no measurable effect (confirmed by CubicMomentsOp). 40B vs 24B shouldn't matter.
- The single division adds throughput cost (1 per 4 clocks on Blackwell), but it's only ONE division per combine call. WelfordOp has division AND conditional branching.
- The branching in WelfordOp/KalmanOp causes warp divergence within the combine itself — different threads in the same warp take different paths. A division doesn't cause divergence; it's uniformly expensive.
- So: 40μs base + ~15-25μs for the division throughput penalty = 55-65μs.

## What the result will reveal

If SarkkaOp lands at:
- **~40μs** (Tier 1): Division alone is NOT the bottleneck. The entire 2.5x penalty is from branching. Design principle: divisions in the combine are fine; conditional logic is the killer.
- **~55-70μs** (between tiers): Division adds a measurable but moderate penalty. The 2.5x penalty is a combination of both. Design principle: minimize division AND branching, but branching is worse.
- **~100μs** (Tier 2): Even one division is enough to trigger the full penalty. Design principle: any division in the combine must be eliminated (move to constructor or restructure).

Each outcome has different design implications for the operator vocabulary. This is a testable prediction — the observer should benchmark SarkkaOp at 100K once the pathmaker completes task #17.

## The five-operator gradient

When SarkkaOp is benchmarked, we'll have a five-point gradient:

```
AddOp (8B, 0 div, 0 branch) → ~38μs
KalmanAffineOp (16B, 0 div, 0 branch) → ~39μs
CubicMomentsOp (24B, 0 div, 0 branch) → ~42μs
SarkkaOp (40B, 1 div, 0 branch) → ???
WelfordOp (24B, 1 div, 1+ branch) → ~99μs
KalmanOp (32B, 1+ div, 1+ branch) → ~103μs
```

The first three points establish the state-size-independent baseline. The fourth (SarkkaOp) isolates division. The fifth and sixth confirm the branching penalty on top of division. This is an operator performance model — a transfer function from combine complexity to dispatch latency.

---

*Predictions are more useful than explanations. An explanation fits what already happened; a prediction exposes what you don't know. If SarkkaOp lands outside my range, the model is wrong and I'll learn where it's wrong.*
