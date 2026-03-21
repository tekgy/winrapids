"""
Smoke test for the winrapids library.

Exercises: Column, Frame, fusion, groupby, join, end-to-end pipeline.
"""

import sys
import time
import numpy as np
import cupy as cp

# Add src to path
sys.path.insert(0, "R:/winrapids/src")

from winrapids import Column, Frame, evaluate, fused_sum
from winrapids.fusion import where


def test_column_basics():
    print("=== Column basics ===")
    a = Column.from_numpy("a", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    print(f"  {a}")
    print(f"  sum={a.sum()}, mean={a.mean()}, min={a.min()}, max={a.max()}")
    assert a.sum() == 15.0
    assert a.mean() == 3.0
    print("  PASS\n")


def test_fusion():
    print("=== Kernel fusion ===")
    n = 1_000_000
    a = Column.from_numpy("a", np.random.randn(n))
    b = Column.from_numpy("b", np.random.randn(n))
    c = Column.from_numpy("c", np.random.randn(n))

    # Build expression tree (no GPU work)
    expr = a * b + c

    # Evaluate (fuses to one kernel)
    result = evaluate(expr)
    expected = cp.asnumpy(a._data * b._data + c._data)
    actual = cp.asnumpy(result)
    assert np.max(np.abs(expected - actual)) < 1e-10
    print(f"  a*b+c: max error = {np.max(np.abs(expected - actual)):.2e}")

    # Complex expression
    expr2 = (a * b + c * c - a / b).abs().sqrt()
    result2 = evaluate(expr2)
    expected2 = cp.asnumpy(cp.sqrt(cp.abs(a._data * b._data + c._data * c._data - a._data / b._data)))
    actual2 = cp.asnumpy(result2)
    assert np.max(np.abs(expected2 - actual2)) < 1e-10
    print(f"  sqrt(abs(a*b+c*c-a/b)): max error = {np.max(np.abs(expected2 - actual2)):.2e}")

    # Fused sum
    s = fused_sum(a * b + c)
    expected_s = float(cp.sum(a._data * b._data + c._data))
    assert abs(s - expected_s) < 0.01
    print(f"  fused_sum(a*b+c): {s:.6f} (expected {expected_s:.6f})")

    # Where
    expr_w = where(a > 0, b, c)
    result_w = evaluate(expr_w)
    expected_w = cp.asnumpy(cp.where(a._data > 0, b._data, c._data))
    actual_w = cp.asnumpy(result_w)
    assert np.max(np.abs(expected_w - actual_w)) < 1e-10
    print(f"  where(a>0, b, c): max error = {np.max(np.abs(expected_w - actual_w)):.2e}")

    print("  PASS\n")


def test_frame():
    print("=== Frame ===")
    n = 100_000
    rng = np.random.default_rng(42)

    f = Frame.from_dict({
        "x": rng.standard_normal(n),
        "y": rng.standard_normal(n),
        "key": rng.integers(0, 10, n).astype(np.int64),
    })

    print(f"  {f}")
    print(f"  {f.memory_map()}")
    assert f.shape == (n, 3)
    assert "x" in f
    assert f["x"].sum() != 0  # sanity
    print("  PASS\n")


def test_groupby():
    print("=== GroupBy ===")
    n = 1_000_000
    rng = np.random.default_rng(42)

    f = Frame.from_dict({
        "key": rng.integers(0, 5, n).astype(np.int64),
        "value": rng.standard_normal(n),
    })

    # Sum
    result = f.groupby("key").sum("value")
    print(f"  groupby sum: {result}")
    assert result.shape[0] == 5  # 5 unique keys
    assert result.shape[1] == 2  # key + value

    # Mean
    result_mean = f.groupby("key").mean("value")
    print(f"  groupby mean: {result_mean}")

    # Count
    result_count = f.groupby("key").count()
    total = int(cp.sum(result_count["count"]._data))
    assert total == n
    print(f"  groupby count total: {total} (expected {n})")

    # Multi-agg
    result_agg = f.groupby("key").agg("value", ("sum", "mean", "count"))
    print(f"  groupby agg: {result_agg}")
    assert "value_sum" in result_agg
    assert "value_mean" in result_agg
    assert "count" in result_agg

    # Verify against pandas
    import pandas as pd
    pdf = f.to_pandas()
    pd_result = pdf.groupby("key")["value"].sum().sort_index()
    gpu_result = result.to_pandas().set_index("key").sort_index()
    max_err = np.max(np.abs(pd_result.values - gpu_result["value"].values))
    print(f"  vs pandas: max error = {max_err:.2e}")
    assert max_err < 1e-6

    print("  PASS\n")


def test_join():
    print("=== Join ===")
    rng = np.random.default_rng(42)

    # Fact table
    fact = Frame.from_dict({
        "product_id": rng.integers(0, 100, 10000).astype(np.int64),
        "amount": rng.standard_normal(10000),
    })

    # Dim table
    dim = Frame.from_dict({
        "product_id": np.arange(100, dtype=np.int64),
        "category": rng.integers(0, 5, 100).astype(np.int64),
    })

    joined = fact.join(dim, on="product_id")
    print(f"  joined: {joined}")
    assert joined.shape[0] == 10000  # all fact rows should match
    assert "category" in joined
    assert "amount" in joined

    # Verify keys match
    jf_keys = joined["product_id"].to_numpy()
    jd_cats = joined["category"].to_numpy()

    # Check that each joined row's category matches the dim table
    dim_lookup = dim.to_pandas().set_index("product_id")["category"]
    expected_cats = dim_lookup[jf_keys].values
    assert np.all(jd_cats == expected_cats)
    print(f"  key verification: PASS")

    print("  PASS\n")


def test_end_to_end():
    print("=== End-to-end pipeline ===")
    rng = np.random.default_rng(42)

    n_sales = 1_000_000
    n_products = 1000
    n_categories = 10

    sales = Frame.from_dict({
        "product_id": rng.integers(0, n_products, n_sales).astype(np.int64),
        "quantity": rng.integers(1, 100, n_sales).astype(np.float64),
        "unit_price": rng.uniform(1.0, 500.0, n_sales),
        "discount": rng.uniform(0.0, 0.3, n_sales),
    })

    products = Frame.from_dict({
        "product_id": np.arange(n_products, dtype=np.int64),
        "category_id": rng.integers(0, n_categories, n_products).astype(np.int64),
    })

    t0 = time.perf_counter()

    # Join
    merged = sales.join(products, on="product_id")

    # Fused expression: revenue = quantity * unit_price * (1 - discount)
    revenue_expr = merged["quantity"] * merged["unit_price"] * (1 - merged["discount"])
    revenue = evaluate(revenue_expr)

    # Add revenue column to a new frame for groupby
    result_frame = Frame({
        "category_id": merged["category_id"],
        "revenue": Column("revenue", revenue),
    })

    # GroupBy
    by_category = result_frame.groupby("category_id").agg("revenue", ("sum", "mean", "count"))

    cp.cuda.Device(0).synchronize()
    t_gpu = (time.perf_counter() - t0) * 1000

    print(f"  Pipeline: join -> fused revenue -> groupby")
    print(f"  {n_sales:,} sales x {n_products:,} products -> {n_categories} categories")
    print(f"  GPU time: {t_gpu:.1f} ms")
    print(f"  Result: {by_category}")

    # Verify against pandas
    import pandas as pd
    sales_pd = sales.to_pandas()
    products_pd = products.to_pandas()
    merged_pd = pd.merge(sales_pd, products_pd, on="product_id")
    merged_pd["revenue"] = merged_pd["quantity"] * merged_pd["unit_price"] * (1 - merged_pd["discount"])
    pd_result = merged_pd.groupby("category_id")["revenue"].sum().sort_index()

    gpu_result = by_category.to_pandas().set_index("category_id").sort_index()
    max_err = np.max(np.abs(pd_result.values - gpu_result["revenue_sum"].values))
    print(f"  vs pandas: max error = {max_err:.2e}")
    assert max_err < 1.0  # floating point order differences at scale

    print("  PASS\n")


def main():
    print("WinRapids Library — Smoke Tests")
    print("=" * 50)
    print()

    test_column_basics()
    test_fusion()
    test_frame()
    test_groupby()
    test_join()
    test_end_to_end()

    print("=" * 50)
    print("All tests passed.")


if __name__ == "__main__":
    main()
