# ten-lines: Navigator Notes

## The target

```python
import tambear as tb

df = tb.read("aapl.tb")                          # .tb header → GPU pointers
sums = df.groupby("ticker_id").sum("volume")      # hash scatter, 0.06ms
stats = df.groupby("ticker_id").stats("close")    # mean+var+std, one pass
hot = df.filter("close > 150.0")                  # bitmask, no copy
hot_sums = hot.groupby("ticker_id").sum("volume") # mask-aware scatter
df.write("out.tb")                                # flush with provenance tail
```

Six meaningful operations. All four invariants invisible to the user.

## The crate: crates/tambear-py/

Scaffolded. PyO3 crate wrapping `tambear::HashScatterEngine` and `tambear::Frame`.

Key PyO3 design decision: `PyGroupByBuilder` is a lightweight lazy builder.
`df.groupby("col")` is free (just stores the key name). `.sum("val")` triggers GPU.
This is the right granularity — no need for a full lazy graph for this API surface.

## from_columns() — the testing path

```python
df = tb.from_columns({
    "ticker_id": [0, 0, 1, 1, 2],    # integers, direct-index groupby
    "close": [149.0, 150.1, 148.5, 151.2, 147.3],
    "volume": [1000, 2000, 1500, 2500, 800],
})
sums = df.groupby("ticker_id").sum("volume")
# sums.sums == [3000.0, 4000.0, 800.0]
```

`from_columns()` is the integration test path — works without .tb file, just
Python lists. This is how Task 4 (real AAPL data) can be tested before the
full .tb format is implemented. Observer can use this to validate the scatter
engine with real numbers.

## The four invariants from Python's perspective

**Sort-free**: User never sees sort. `groupby` always uses hash scatter.
No option to "use sort-based groupby." The invariant is structural.

**Mask-not-filter**: `df.filter("close > 150.0")` returns `df` with a mask
set. The returned Frame is the SAME data, different mask. From Python: it
looks like a filtered copy but zero data was copied. Observer's filter()
implementation: parse expr, generate GPU bitmask kernel, set `frame.row_mask`.

**Dictionary strings**: If user loads a .tb file with a string column
"ticker_symbol", it appears as `column_names=["ticker_symbol", ...]` in Python.
But internally the column is u32 codes. User does `df.groupby("ticker_symbol")` —
works identically to `df.groupby("ticker_id")`. The PyFrame should expose both
forms: `df.dictionary("ticker_symbol")` returns the dict for decode at output.

**Types once**: `tb.read("aapl.tb", dtype=tb.f64)` — dtype propagates to
all operations. The PyGroupByBuilder reads `frame.pipeline_dtype` when dispatching.

## The dependency on scout's work

`PyGroupByBuilder.sum()`, `.mean()`, `.stats()` all need:
1. `frame.group_index_for(key_col, max_key)` — needs GroupIndex::build() (scout)
2. `engine.groupby(row_to_group, values, max_key+1)` — needs HashScatterEngine (scout)

So Task 5 is blocked on Task 1. But:
- The PyO3 scaffold can be written now (done)
- `from_columns()` can be implemented now (creates a mock Frame without GroupIndex)
- The full wiring happens once scout lands

Observer should start with `from_columns()` → `groupby()` → `scatter_sum()` as
the integration path, using `HashScatterEngine` directly without `GroupIndex`.
That gets the Python surface working end-to-end even before GroupIndex::build() is done.

## The maturin build

Look at `crates/winrapids-py/` for the maturin pattern. The key: `pyproject.toml`
needs `[tool.maturin] manifest-path = "crates/tambear-py/Cargo.toml"`.
The wheel installs as `tambear` with the `_tambear` native extension plus
a thin `tambear/__init__.py` that re-exports.
