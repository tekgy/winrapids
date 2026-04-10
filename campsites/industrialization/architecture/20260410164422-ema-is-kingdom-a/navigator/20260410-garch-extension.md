# GARCH Extension to Kingdom A Finding

Written: 2026-04-10, navigator

## GARCH is Kingdom A (not Kingdom B as labeled)

`volatility.rs:9` labels GARCH as Kingdom B. This is wrong.

GARCH(1,1): `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}`

Affine map form: `a = β` (constant), `b_t = ω + α·r²_{t-1}` (data-dependent).
`r_{t-1}` is the observed return series — DATA, not state. Maps compose.
This is Kingdom A.

Correct classification:
- GARCH filter: Kingdom A (affine prefix scan)
- Log-likelihood sum: Kingdom A (prefix sum)  
- Parameter optimization: Kingdom C (iterative MLE)

Source: scout garden note `~/.claude/garden/2026-04-10-garch-is-kingdom-a.md`

## Action required

Fix comment at `volatility.rs:9`. Code is correct; label is wrong.
Routed to pathmaker 2026-04-10.

## The broader theorem emerging

EMA, EWMA, all ARMA, GARCH filter, HMM forward algorithm, Kalman filter —
all Kingdom A via the affine semigroup or matrix product semigroup.

Genuine Kingdom B requires state-dependent maps (map structure depends on
the current hidden state, not the current data). Scout/aristotle/math-researcher
actively investigating where the true Fock boundary lies.

Conjecture: Kingdom B requires nonlinearity in the STATE variables specifically
(not in the data). Affine-in-state = Kingdom A regardless of data complexity.
