# FP16 is Not Universe-Safe Without Normalization
Created: 2026-03-27

Scout found: price² overflows FP16 at price > ~$255. AAPL at $230 has 24% headroom.

But the failure mode is worse than "overflow on squared ops":

BRK.A trades at ~$700,000/share. FP16 max is 65,504.
`cp.float16(700_000)` → **inf**. The level value itself overflows.

A K04 pipeline that casts raw prices to FP16 before normalization would silently corrupt
every BRK.A, NVR, BKNG, and other high-price ticker in the universe. No error — just inf
and NaN cascading through the correlation matrix.

**The rule**: normalize to zero-mean, unit-variance (or any bounded range) BEFORE any FP16 cast.
After z-score normalization, all prices are in [-3, 3] regardless of dollar level.

Derived features (ln_price ≈ 5.4 for AAPL, sqrt_price ≈ 15.2) are also safe — their
natural ranges stay well under FP16 max even for extreme tickers.

**Apply to**: Any pipeline step that converts to FP16/BF16 for TC operations.
The MKTF file stores float32 throughout. FP16 is only for in-memory TC computation.
