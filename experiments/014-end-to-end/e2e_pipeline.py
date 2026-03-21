"""
WinRapids Experiment 014: End-to-End GPU Analytics Pipeline

This ties together everything from the expedition into one complete pipeline:

    Parquet file -> Arrow -> GPU DataFrame -> Join -> Fused Expression -> GroupBy -> Result

The scenario: a sales analytics pipeline.
- Fact table: 10M sales records (product_id, quantity, unit_price, discount)
- Dimension table: 10K products (product_id, category_id, brand_id)
- Query: "Revenue by category, where revenue = quantity * unit_price * (1 - discount)"

This is the kind of query a data scientist writes every day in pandas.
We build it entirely on GPU, on Windows, without cuDF.

Components used:
- Arrow/Parquet I/O (Experiment 003)
- GPU DataFrame (Experiment 004)
- Kernel fusion for revenue expression (Experiment 010b)
- Direct-index join (Experiment 013)
- Sort-based GroupBy (Experiment 011)
"""

from __future__ import annotations

import os
import time
import tempfile

import numpy as np
import cupy as cp
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


# ============================================================
# Data Generation
# ============================================================

def generate_data(n_sales: int, n_products: int, n_categories: int, n_brands: int):
    """Generate realistic sales data."""
    rng = np.random.default_rng(42)

    # Dimension table: products
    products = pa.table({
        "product_id": pa.array(np.arange(n_products, dtype=np.int64)),
        "category_id": pa.array(rng.integers(0, n_categories, n_products).astype(np.int64)),
        "brand_id": pa.array(rng.integers(0, n_brands, n_products).astype(np.int64)),
    })

    # Fact table: sales
    sales = pa.table({
        "product_id": pa.array(rng.integers(0, n_products, n_sales).astype(np.int64)),
        "quantity": pa.array(rng.integers(1, 100, n_sales).astype(np.int64)),
        "unit_price": pa.array((rng.uniform(1.0, 500.0, n_sales)).astype(np.float64)),
        "discount": pa.array((rng.uniform(0.0, 0.3, n_sales)).astype(np.float64)),
    })

    return sales, products


def write_parquet(table: pa.Table, path: str):
    """Write Arrow table to Parquet file."""
    pq.write_table(table, path, compression="snappy")


# ============================================================
# Pandas Pipeline (baseline)
# ============================================================

def pandas_pipeline(sales_path: str, products_path: str) -> pd.DataFrame:
    """Complete pandas pipeline: read -> join -> compute -> groupby."""
    # Read
    sales = pd.read_parquet(sales_path)
    products = pd.read_parquet(products_path)

    # Join
    merged = pd.merge(sales, products, on="product_id", how="inner")

    # Compute: revenue = quantity * unit_price * (1 - discount)
    merged["revenue"] = merged["quantity"] * merged["unit_price"] * (1 - merged["discount"])

    # GroupBy: revenue by category
    result = merged.groupby("category_id")["revenue"].agg(["sum", "mean", "count"])

    return result


# ============================================================
# GPU Pipeline
# ============================================================

def gpu_pipeline(sales_path: str, products_path: str, n_products: int):
    """
    Complete GPU pipeline: read -> GPU transfer -> join -> fused compute -> groupby.

    Every step tracked individually for performance profiling.
    """
    timings = {}

    # Step 1: Read Parquet via Arrow (CPU)
    t0 = time.perf_counter()
    sales_table = pq.read_table(sales_path)
    products_table = pq.read_table(products_path)
    timings["read_parquet"] = (time.perf_counter() - t0) * 1000

    # Step 2: Arrow -> NumPy -> CuPy (H2D transfer)
    t0 = time.perf_counter()
    s_product_id = cp.asarray(sales_table.column("product_id").to_numpy())
    s_quantity = cp.asarray(sales_table.column("quantity").to_numpy().astype(np.float64))
    s_unit_price = cp.asarray(sales_table.column("unit_price").to_numpy())
    s_discount = cp.asarray(sales_table.column("discount").to_numpy())

    p_product_id = cp.asarray(products_table.column("product_id").to_numpy())
    p_category_id = cp.asarray(products_table.column("category_id").to_numpy())
    cp.cuda.Device(0).synchronize()
    timings["h2d_transfer"] = (time.perf_counter() - t0) * 1000

    # Step 3: Join (direct-index for dense integer keys)
    t0 = time.perf_counter()
    # Build lookup: product_id -> product row index
    lookup = cp.full(n_products, -1, dtype=cp.int32)
    lookup[p_product_id.astype(cp.int64)] = cp.arange(len(p_product_id), dtype=cp.int32)

    # Probe
    fact_keys_bounded = cp.clip(s_product_id, 0, n_products - 1)
    dim_indices = lookup[fact_keys_bounded.astype(cp.int64)]
    valid = (s_product_id >= 0) & (s_product_id < n_products) & (dim_indices >= 0)
    fact_idx = cp.where(valid)[0]
    dim_idx = dim_indices[fact_idx]

    # Gather joined columns
    j_quantity = s_quantity[fact_idx]
    j_unit_price = s_unit_price[fact_idx]
    j_discount = s_discount[fact_idx]
    j_category = p_category_id[dim_idx]
    cp.cuda.Device(0).synchronize()
    timings["join"] = (time.perf_counter() - t0) * 1000

    # Step 4: Fused expression: revenue = quantity * unit_price * (1 - discount)
    # Using CuPy RawKernel codegen for fusion
    t0 = time.perf_counter()
    _revenue_kernel = cp.RawKernel(r"""
    extern "C" __global__
    void compute_revenue(const double* quantity, const double* unit_price,
                         const double* discount, double* revenue, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            revenue[idx] = quantity[idx] * unit_price[idx] * (1.0 - discount[idx]);
        }
    }
    """, "compute_revenue")

    n_joined = len(j_quantity)
    revenue = cp.empty(n_joined, dtype=cp.float64)
    threads = 256
    blocks = (n_joined + threads - 1) // threads
    _revenue_kernel((blocks,), (threads,), (j_quantity, j_unit_price, j_discount, revenue, n_joined))
    cp.cuda.Device(0).synchronize()
    timings["fused_compute"] = (time.perf_counter() - t0) * 1000

    # Step 5: GroupBy category_id -> sum, mean, count
    t0 = time.perf_counter()
    sort_idx = cp.argsort(j_category)
    sorted_cats = j_category[sort_idx]
    sorted_revenue = revenue[sort_idx]

    boundaries = cp.concatenate([cp.array([True]), sorted_cats[1:] != sorted_cats[:-1]])
    boundary_idx = cp.where(boundaries)[0]
    unique_cats = sorted_cats[boundary_idx]
    n_groups = len(unique_cats)

    # Sum via cumsum
    cumsum = cp.cumsum(sorted_revenue)
    end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([n_joined - 1])])
    group_sums = cumsum[end_idx].copy()
    group_sums[1:] -= cumsum[boundary_idx[1:] - 1]

    # Count
    group_counts = cp.diff(cp.concatenate([boundary_idx, cp.array([n_joined])]))

    # Mean
    group_means = group_sums / group_counts.astype(cp.float64)

    cp.cuda.Device(0).synchronize()
    timings["groupby"] = (time.perf_counter() - t0) * 1000

    # Step 6: D2H transfer (result only — small)
    t0 = time.perf_counter()
    result = {
        "category_id": cp.asnumpy(unique_cats),
        "sum": cp.asnumpy(group_sums),
        "mean": cp.asnumpy(group_means),
        "count": cp.asnumpy(group_counts),
    }
    timings["d2h_result"] = (time.perf_counter() - t0) * 1000

    timings["total_gpu"] = sum(v for k, v in timings.items() if k != "read_parquet")

    return result, timings


# ============================================================
# Main
# ============================================================

def main():
    print("WinRapids Experiment 014: End-to-End GPU Analytics Pipeline")
    print("=" * 70)

    n_sales = 10_000_000
    n_products = 10_000
    n_categories = 50
    n_brands = 200

    print(f"\nScenario: Sales Analytics")
    print(f"  Fact table:  {n_sales:,} sales records")
    print(f"  Dim table:   {n_products:,} products")
    print(f"  Categories:  {n_categories}")
    print(f"  Query:       revenue by category, where revenue = qty * price * (1-discount)")
    print()

    # Generate data
    print("Generating data...", end=" ", flush=True)
    sales, products = generate_data(n_sales, n_products, n_categories, n_brands)
    print("done.")

    # Write Parquet
    tmpdir = tempfile.mkdtemp(prefix="winrapids_")
    sales_path = os.path.join(tmpdir, "sales.parquet")
    products_path = os.path.join(tmpdir, "products.parquet")
    write_parquet(sales, sales_path)
    write_parquet(products, products_path)
    sales_mb = os.path.getsize(sales_path) / 1e6
    products_mb = os.path.getsize(products_path) / 1e6
    print(f"Parquet files: sales={sales_mb:.1f} MB, products={products_mb:.1f} MB")
    print()

    # ---- Pandas Pipeline ----
    print("Running pandas pipeline...")
    # Warmup
    _ = pandas_pipeline(sales_path, products_path)

    t0 = time.perf_counter()
    pd_result = pandas_pipeline(sales_path, products_path)
    t_pandas = (time.perf_counter() - t0) * 1000
    print(f"  Pandas total:  {t_pandas:.1f} ms")
    print()

    # ---- GPU Pipeline ----
    print("Running GPU pipeline...")
    # Warmup
    _, _ = gpu_pipeline(sales_path, products_path, n_products)

    gpu_result, timings = gpu_pipeline(sales_path, products_path, n_products)

    print(f"  Read Parquet:    {timings['read_parquet']:8.2f} ms  (Arrow, CPU)")
    print(f"  H2D Transfer:   {timings['h2d_transfer']:8.2f} ms  (Arrow -> CuPy)")
    print(f"  Join:            {timings['join']:8.2f} ms  (direct-index)")
    print(f"  Fused Compute:   {timings['fused_compute']:8.2f} ms  (qty * price * (1-disc))")
    print(f"  GroupBy:         {timings['groupby']:8.2f} ms  (sort + cumsum)")
    print(f"  D2H Result:      {timings['d2h_result']:8.2f} ms  ({n_categories} groups)")
    print(f"  ----------------------------------------")
    print(f"  GPU total:       {timings['total_gpu']:8.2f} ms  (excl. Parquet read)")
    print(f"  Full pipeline:   {timings['total_gpu'] + timings['read_parquet']:8.2f} ms")
    print()

    # ---- Comparison ----
    gpu_total_with_io = timings["total_gpu"] + timings["read_parquet"]
    gpu_compute_only = timings["total_gpu"]
    print(f"=== Comparison ===")
    print(f"  Pandas:           {t_pandas:8.1f} ms")
    print(f"  GPU (full):       {gpu_total_with_io:8.1f} ms  ({t_pandas/gpu_total_with_io:.1f}x faster)")
    print(f"  GPU (excl. I/O):  {gpu_compute_only:8.1f} ms  ({t_pandas/gpu_compute_only:.1f}x faster)")
    print()

    # ---- Verify correctness ----
    print("=== Correctness Verification ===")
    pd_sorted = pd_result.sort_index()

    max_sum_err = 0
    max_mean_err = 0
    for i, cat in enumerate(gpu_result["category_id"]):
        if cat in pd_sorted.index:
            sum_err = abs(gpu_result["sum"][i] - pd_sorted.loc[cat, "sum"])
            mean_err = abs(gpu_result["mean"][i] - pd_sorted.loc[cat, "mean"])
            max_sum_err = max(max_sum_err, sum_err)
            max_mean_err = max(max_mean_err, mean_err)

    print(f"  Max sum error:    {max_sum_err:.2e}")
    print(f"  Max mean error:   {max_mean_err:.2e}")
    print(f"  Categories match: {len(gpu_result['category_id']) == len(pd_sorted)}")

    # Show top 5 categories by revenue
    print(f"\n=== Top 5 Categories by Revenue ===")
    sort_idx_result = np.argsort(gpu_result["sum"])[::-1]
    print(f"  {'Category':>10s}  {'Revenue':>15s}  {'Avg Revenue':>15s}  {'Count':>10s}")
    for i in sort_idx_result[:5]:
        print(f"  {gpu_result['category_id'][i]:10d}  {gpu_result['sum'][i]:15,.2f}  "
              f"{gpu_result['mean'][i]:15,.2f}  {gpu_result['count'][i]:10,d}")

    # Cleanup
    os.unlink(sales_path)
    os.unlink(products_path)
    os.rmdir(tmpdir)

    print(f"\n{'=' * 70}")
    print("Experiment 014 complete.")


if __name__ == "__main__":
    main()
