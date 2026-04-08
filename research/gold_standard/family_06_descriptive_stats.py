"""
Gold Standard Oracle: Family 06 — Descriptive Statistics

Generates expected values for ALL descriptive statistics that should be derivable
from the 11-field MSR: {n, Σp, Σp², max, min, Σsz, Σ(p·sz), Σr, Σr², Σr³, Σr⁴}

For each test dataset, outputs a JSON dictionary of expected values computed by
scipy/numpy/statsmodels. Tambear's implementation must match these within stated tolerance.

Usage:
    python research/gold_standard/family_06_descriptive_stats.py > research/gold_standard/family_06_expected.json

Verification chain: scipy (trusted) → tambear CPU (reference) → tambear CUDA (production)
"""

import json
import numpy as np
from scipy import stats as sp_stats
import sys

def compute_all_descriptive_stats(data: np.ndarray, name: str) -> dict:
    """Compute every descriptive statistic from scipy/numpy as gold standard."""
    n = len(data)
    result = {"dataset": name, "n": n}

    # === CENTRAL TENDENCY ===
    result["mean"] = float(np.mean(data))
    result["median"] = float(np.median(data))
    # Trimmed mean (10% trim each side)
    result["trimmed_mean_10"] = float(sp_stats.trim_mean(data, 0.1))

    # === DISPERSION ===
    result["variance_population"] = float(np.var(data, ddof=0))
    result["variance_sample"] = float(np.var(data, ddof=1))
    result["std_population"] = float(np.std(data, ddof=0))
    result["std_sample"] = float(np.std(data, ddof=1))
    result["range"] = float(np.ptp(data))
    result["min"] = float(np.min(data))
    result["max"] = float(np.max(data))
    result["iqr"] = float(sp_stats.iqr(data))
    result["mad"] = float(sp_stats.median_abs_deviation(data))
    # Coefficient of variation
    if np.mean(data) != 0:
        result["cv"] = float(np.std(data, ddof=1) / np.mean(data))
    else:
        result["cv"] = float("nan")

    # === SHAPE ===
    # Skewness: Fisher (bias=True = population, bias=False = sample correction)
    result["skewness_fisher_biased"] = float(sp_stats.skew(data, bias=True))
    result["skewness_fisher_unbiased"] = float(sp_stats.skew(data, bias=False))
    # Kurtosis: Fisher convention (normal=0), excess kurtosis
    result["kurtosis_fisher_biased"] = float(sp_stats.kurtosis(data, bias=True))
    result["kurtosis_fisher_unbiased"] = float(sp_stats.kurtosis(data, bias=False))
    # Kurtosis: Pearson convention (normal=3)
    result["kurtosis_pearson_biased"] = float(sp_stats.kurtosis(data, bias=True, fisher=False))
    result["kurtosis_pearson_unbiased"] = float(sp_stats.kurtosis(data, bias=False, fisher=False))

    # === MOMENTS ===
    for k in range(1, 9):
        result[f"central_moment_{k}"] = float(sp_stats.moment(data, moment=k))
    for k in range(1, 5):
        result[f"raw_moment_{k}"] = float(np.mean(data ** k))

    # === QUANTILES ===
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        result[f"quantile_{q:.2f}"] = float(np.quantile(data, q))
    # Deciles
    for d in range(1, 10):
        result[f"decile_{d}"] = float(np.quantile(data, d / 10))

    # === MSR FIELDS (what tambear accumulates) ===
    # These are the raw accumulator values — tambear computes these on GPU
    result["msr_n"] = n
    result["msr_sum_p"] = float(np.sum(data))
    result["msr_sum_p2"] = float(np.sum(data ** 2))
    result["msr_max"] = float(np.max(data))
    result["msr_min"] = float(np.min(data))
    # Returns (log returns for financial data, regular differences otherwise)
    if np.all(data > 0):
        returns = np.diff(np.log(data))
    else:
        returns = np.diff(data)
    if len(returns) > 0:
        result["msr_sum_r"] = float(np.sum(returns))
        result["msr_sum_r2"] = float(np.sum(returns ** 2))
        result["msr_sum_r3"] = float(np.sum(returns ** 3))
        result["msr_sum_r4"] = float(np.sum(returns ** 4))
    else:
        result["msr_sum_r"] = 0.0
        result["msr_sum_r2"] = 0.0
        result["msr_sum_r3"] = 0.0
        result["msr_sum_r4"] = 0.0

    # === DERIVED FROM MSR (what tambear extracts) ===
    # These are the extraction formulas — same math, computed from sufficient stats
    mean_from_msr = result["msr_sum_p"] / n
    var_from_msr = result["msr_sum_p2"] / n - mean_from_msr ** 2
    result["mean_from_msr"] = float(mean_from_msr)
    result["variance_pop_from_msr"] = float(var_from_msr)
    result["std_pop_from_msr"] = float(np.sqrt(max(0, var_from_msr)))

    # Verify MSR extraction matches direct computation
    result["msr_mean_matches"] = abs(mean_from_msr - result["mean"]) < 1e-12
    result["msr_var_matches"] = abs(var_from_msr - result["variance_population"]) < 1e-10

    # === NORMALITY TESTS ===
    if n >= 8:
        stat, pval = sp_stats.shapiro(data[:min(n, 5000)])  # Shapiro limited to 5000
        result["shapiro_stat"] = float(stat)
        result["shapiro_pval"] = float(pval)
    if n >= 20:
        stat, pval = sp_stats.normaltest(data)
        result["dagostino_stat"] = float(stat)
        result["dagostino_pval"] = float(pval)
    stat, pval = sp_stats.jarque_bera(data)
    result["jarque_bera_stat"] = float(stat)
    result["jarque_bera_pval"] = float(pval)

    # === ENTROPY ===
    # Differential entropy estimate (assuming normal)
    if result["std_sample"] > 0:
        result["entropy_normal"] = float(0.5 * np.log(2 * np.pi * np.e * result["variance_sample"]))

    return result


def make_test_datasets() -> list:
    """Generate test datasets covering normal, adversarial, and real-world-like cases."""
    np.random.seed(42)  # Reproducible
    datasets = []

    # 1. Standard normal (baseline — every stat has known expected value)
    datasets.append(("standard_normal_1000", np.random.randn(1000)))

    # 2. Uniform [0,1] — known theoretical moments
    datasets.append(("uniform_01_1000", np.random.uniform(0, 1, 1000)))

    # 3. Exponential (λ=1) — known positive skew
    datasets.append(("exponential_1000", np.random.exponential(1.0, 1000)))

    # 4. Heavy-tailed: t-distribution (df=3) — known excess kurtosis
    datasets.append(("t_dist_df3_1000", np.random.standard_t(3, 1000)))

    # 5. Bimodal: mixture of two normals
    mix = np.concatenate([np.random.normal(-3, 0.5, 500), np.random.normal(3, 0.5, 500)])
    datasets.append(("bimodal_1000", mix))

    # 6. Lognormal — financial return-like
    datasets.append(("lognormal_1000", np.random.lognormal(0, 0.5, 1000)))

    # === ADVERSARIAL CASES ===

    # 7. All identical values (zero variance)
    datasets.append(("all_same_100", np.full(100, 42.0)))

    # 8. Single element
    datasets.append(("single_element", np.array([3.14])))

    # 9. Two elements (minimum for variance)
    datasets.append(("two_elements", np.array([1.0, 2.0])))

    # 10. Alternating extreme values
    datasets.append(("alternating_extreme_100",
                     np.array([(-1)**i * 1e15 for i in range(100)])))

    # 11. Near-overflow values
    datasets.append(("near_overflow_10",
                     np.array([1e300, 1e300, 1e300, -1e300, -1e300,
                               1e-300, 1e-300, 0, 0, 1e150])))

    # 12. Contains NaN (tambear should handle or document behavior)
    datasets.append(("with_nan_10",
                     np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])))

    # 13. Contains Inf
    datasets.append(("with_inf_10",
                     np.array([1.0, 2.0, np.inf, 4.0, 5.0, -np.inf, 7.0, 8.0, 9.0, 10.0])))

    # 14. Monotonically increasing (perfect rank correlation)
    datasets.append(("monotone_100", np.arange(1.0, 101.0)))

    # 15. Financial-like: tick prices with microstructure noise
    base = 100 + np.cumsum(np.random.randn(10000) * 0.01)
    noise = np.random.randn(10000) * 0.001  # microstructure noise
    datasets.append(("tick_prices_10k", base + noise))

    # 16. Large dataset for accumulation precision test
    datasets.append(("large_normal_100k", np.random.randn(100_000) * 1000 + 1e6))

    return datasets


def run_oracle():
    """Generate all gold standard values."""
    datasets = make_test_datasets()
    results = []

    for name, data in datasets:
        try:
            result = compute_all_descriptive_stats(data, name)
            results.append(result)
            n_stats = len([k for k in result.keys() if k != "dataset"])
            print(f"  {name}: {n_stats} statistics computed", file=sys.stderr)
        except Exception as e:
            print(f"  {name}: FAILED — {e}", file=sys.stderr)
            results.append({"dataset": name, "error": str(e)})

    # Output as JSON
    print(json.dumps(results, indent=2, allow_nan=True))

    # Summary
    total_stats = sum(len([k for k in r.keys() if k not in ("dataset", "error")]) for r in results)
    print(f"\nTotal: {len(results)} datasets, {total_stats} gold standard values", file=sys.stderr)


if __name__ == "__main__":
    run_oracle()
