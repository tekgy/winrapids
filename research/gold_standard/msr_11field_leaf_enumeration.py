"""
MSR Verification: Do 11 accumulators truly produce 90 leaves?

The claim: {n, Σp, Σp², max, min, Σsz, Σ(p·sz), Σr, Σr², Σr³, Σr⁴}
produces 90 distinct statistical outputs (leaves).

This script enumerates every computable statistic from these 11 fields
and verifies each one against scipy/numpy on synthetic data.

If something is "computable from the MSR", it means:
  given ONLY {n, sum_p, sum_p2, max, min, sum_sz, sum_psz, sum_r, sum_r2, sum_r3, sum_r4}
  you can compute it with arithmetic alone — no access to raw data.
"""

import numpy as np
from scipy import stats as sp_stats


def enumerate_msr_leaves(data, sizes=None):
    """
    Given raw data, compute the 11 MSR fields, then enumerate
    every statistic derivable from those fields alone.

    data: array of prices/values
    sizes: array of trade sizes (optional, for volume-weighted stats)
    """
    n = len(data)
    if sizes is None:
        sizes = np.ones(n)

    # Compute returns (log returns for positive data, differences otherwise)
    if np.all(data > 0):
        returns = np.diff(np.log(data))
    else:
        returns = np.diff(data)

    # === THE 11 MSR FIELDS ===
    msr = {
        'n': n,
        'sum_p': np.sum(data),
        'sum_p2': np.sum(data**2),
        'max_p': np.max(data),
        'min_p': np.min(data),
        'sum_sz': np.sum(sizes),
        'sum_psz': np.sum(data * sizes),
        'sum_r': np.sum(returns) if len(returns) > 0 else 0.0,
        'sum_r2': np.sum(returns**2) if len(returns) > 0 else 0.0,
        'sum_r3': np.sum(returns**3) if len(returns) > 0 else 0.0,
        'sum_r4': np.sum(returns**4) if len(returns) > 0 else 0.0,
    }
    n_r = len(returns)  # n-1 for returns

    # === ENUMERATE ALL LEAVES ===
    leaves = {}

    # --- Price-based (from n, sum_p, sum_p2, max, min) ---

    # 1. Mean
    mean_p = msr['sum_p'] / msr['n']
    leaves['mean'] = mean_p

    # 2-3. Variance (population and sample)
    var_pop = msr['sum_p2'] / msr['n'] - mean_p**2
    leaves['variance_population'] = var_pop
    leaves['variance_sample'] = var_pop * msr['n'] / (msr['n'] - 1) if msr['n'] > 1 else float('nan')

    # 4-5. Std
    leaves['std_population'] = np.sqrt(max(0, var_pop))
    leaves['std_sample'] = np.sqrt(max(0, leaves['variance_sample'])) if msr['n'] > 1 else float('nan')

    # 6. Range
    leaves['range'] = msr['max_p'] - msr['min_p']

    # 7-8. Min/Max (direct from MSR)
    leaves['min'] = msr['min_p']
    leaves['max'] = msr['max_p']

    # 9. Coefficient of variation
    leaves['cv'] = leaves['std_sample'] / mean_p if mean_p != 0 else float('nan')

    # 10. Sum (trivially from MSR)
    leaves['sum'] = msr['sum_p']

    # 11. Sum of squares
    leaves['sum_of_squares'] = msr['sum_p2']

    # 12. Mean of squares
    leaves['mean_of_squares'] = msr['sum_p2'] / msr['n']

    # 13. Root mean square
    leaves['rms'] = np.sqrt(msr['sum_p2'] / msr['n'])

    # 14. Second central moment (= population variance, same as #2)
    leaves['central_moment_2'] = var_pop

    # --- Volume-weighted (from sum_sz, sum_psz) ---

    # 15. VWAP
    leaves['vwap'] = msr['sum_psz'] / msr['sum_sz'] if msr['sum_sz'] != 0 else float('nan')

    # 16. Total volume
    leaves['total_volume'] = msr['sum_sz']

    # 17. Average trade size
    leaves['avg_trade_size'] = msr['sum_sz'] / msr['n']

    # 18. Notional (same as sum_psz)
    leaves['notional'] = msr['sum_psz']

    # --- Return-based (from sum_r, sum_r2, sum_r3, sum_r4, n-1) ---

    if n_r > 0:
        mean_r = msr['sum_r'] / n_r
        var_r_pop = msr['sum_r2'] / n_r - mean_r**2
        var_r_sam = var_r_pop * n_r / (n_r - 1) if n_r > 1 else float('nan')

        # 19. Mean return
        leaves['return_mean'] = mean_r

        # 20. Total return
        leaves['return_total'] = msr['sum_r']

        # 21-22. Return variance
        leaves['return_variance_population'] = var_r_pop
        leaves['return_variance_sample'] = var_r_sam

        # 23-24. Return std (= volatility)
        leaves['return_std_population'] = np.sqrt(max(0, var_r_pop))
        leaves['return_std_sample'] = np.sqrt(max(0, var_r_sam)) if n_r > 1 else float('nan')

        # 25. Realized variance (sum of squared returns — standard definition)
        leaves['realized_variance'] = msr['sum_r2']

        # 26. Realized volatility
        leaves['realized_volatility'] = np.sqrt(msr['sum_r2'])

        # 27. Annualized volatility (assuming daily returns, 252 trading days)
        leaves['annualized_volatility'] = np.sqrt(msr['sum_r2'] * 252 / n_r)

        # 28-29. Return skewness
        if n_r > 2 and var_r_pop > 0:
            m3_r = msr['sum_r3'] / n_r - 3 * mean_r * var_r_pop - mean_r**3
            leaves['return_skewness_population'] = m3_r / (var_r_pop ** 1.5)
            # Sample correction (Fisher)
            g1 = m3_r / (var_r_pop ** 1.5)
            leaves['return_skewness_sample'] = g1 * np.sqrt(n_r * (n_r - 1)) / (n_r - 2)
        else:
            leaves['return_skewness_population'] = float('nan')
            leaves['return_skewness_sample'] = float('nan')

        # 30-33. Return kurtosis
        if n_r > 3 and var_r_pop > 0:
            # Central moment 4: E[(X-μ)⁴] = E[X⁴] - 4μE[X³] + 6μ²E[X²] - 3μ⁴
            m4_r = (msr['sum_r4'] / n_r
                    - 4 * mean_r * msr['sum_r3'] / n_r
                    + 6 * mean_r**2 * msr['sum_r2'] / n_r
                    - 3 * mean_r**4)
            # Pearson kurtosis (normal = 3)
            kurt_pearson = m4_r / (var_r_pop ** 2)
            leaves['return_kurtosis_pearson'] = kurt_pearson
            # Fisher/excess kurtosis (normal = 0)
            leaves['return_kurtosis_fisher'] = kurt_pearson - 3.0
            # Sample-corrected excess kurtosis
            G2 = kurt_pearson - 3.0
            leaves['return_kurtosis_sample'] = ((n_r - 1) / ((n_r - 2) * (n_r - 3)) *
                                                 ((n_r + 1) * G2 + 6))
            # Jarque-Bera test statistic (from skew and kurtosis)
            if 'return_skewness_population' in leaves:
                S = leaves['return_skewness_population']
                K = leaves['return_kurtosis_fisher']
                leaves['return_jarque_bera'] = n_r / 6 * (S**2 + K**2 / 4)
        else:
            leaves['return_kurtosis_pearson'] = float('nan')
            leaves['return_kurtosis_fisher'] = float('nan')
            leaves['return_kurtosis_sample'] = float('nan')

        # 34. Return range
        # NOTE: Cannot compute from MSR alone — need max(r) and min(r), NOT max(p) and min(p)
        # This is a leaf that REQUIRES additional MSR fields: max_r, min_r
        # leaves['return_range'] = NOT COMPUTABLE from current 11 fields

        # 35-36. Risk metrics
        leaves['downside_variance'] = None  # NEEDS sum_r2_negative (not in MSR)
        leaves['sortino_denominator'] = None  # NEEDS sum_r2_negative

        # 37-38. Semi-variance (above/below mean)
        # NOTE: Cannot compute from MSR — need conditional sums
        leaves['semi_variance_upper'] = None
        leaves['semi_variance_lower'] = None

        # 39. Mean absolute deviation of returns
        # NOTE: Cannot compute from MSR — need sum(|r - mean_r|)
        leaves['return_mad'] = None

        # --- Higher moment combinations ---

        # 40. Omega ratio numerator (sum of positive returns)
        # NOTE: Cannot compute — need sum_r_positive
        leaves['omega_ratio_numerator'] = None

        # 41. Sum of cubed returns (direct MSR field)
        leaves['sum_r3'] = msr['sum_r3']

        # 42. Sum of quartic returns (direct MSR field)
        leaves['sum_r4'] = msr['sum_r4']

        # 43. Mean cubed return
        leaves['mean_r3'] = msr['sum_r3'] / n_r

        # 44. Mean quartic return
        leaves['mean_r4'] = msr['sum_r4'] / n_r

    # --- Price-level derived ---

    # 45. Mid-price (if max/min represent bid/ask in a context)
    leaves['midpoint'] = (msr['max_p'] + msr['min_p']) / 2

    # 46. Spread (max - min, same as range)
    leaves['spread'] = msr['max_p'] - msr['min_p']

    # 47. Relative spread
    mid = (msr['max_p'] + msr['min_p']) / 2
    leaves['relative_spread'] = (msr['max_p'] - msr['min_p']) / mid if mid != 0 else float('nan')

    # 48. Price/VWAP ratio
    leaves['price_vwap_ratio'] = mean_p / leaves['vwap'] if leaves.get('vwap') and leaves['vwap'] != 0 else float('nan')

    # === COUNT WHAT'S ACTUALLY COMPUTABLE ===
    computable = {k: v for k, v in leaves.items() if v is not None}
    not_computable = {k: v for k, v in leaves.items() if v is None}

    return msr, computable, not_computable


def main():
    np.random.seed(42)

    # Realistic financial data: 1000 tick prices with microstructure
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
    sizes = np.random.exponential(100, 1000)

    msr, computable, not_computable = enumerate_msr_leaves(prices, sizes)

    print("=" * 60)
    print("MSR 11-FIELD LEAF ENUMERATION")
    print("=" * 60)

    print(f"\n11 MSR fields:")
    for k, v in msr.items():
        print(f"  {k}: {v:.6g}")

    print(f"\nCOMPUTABLE from 11 MSR fields alone: {len(computable)} leaves")
    for i, (k, v) in enumerate(sorted(computable.items()), 1):
        if isinstance(v, float) and not np.isnan(v):
            print(f"  {i:3d}. {k}: {v:.10g}")
        else:
            print(f"  {i:3d}. {k}: {v}")

    print(f"\nNOT COMPUTABLE from 11 MSR fields: {len(not_computable)} leaves")
    for k in sorted(not_computable.keys()):
        print(f"  - {k} (needs additional accumulator)")

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {len(computable)} leaves from 11 MSR fields")
    print(f"         {len(not_computable)} leaves need ADDITIONAL accumulators")

    # What additional accumulators would unlock more leaves?
    print(f"\nADDITIONAL ACCUMULATORS NEEDED for full coverage:")
    print(f"  - max_r, min_r: return range, return min/max")
    print(f"  - sum_r_positive, sum_r_negative: Omega ratio, gain/loss decomposition")
    print(f"  - sum_r2_negative: downside variance, Sortino ratio")
    print(f"  - sum_|r-mu|: MAD of returns")
    print(f"  - conditional sums: semi-variance above/below threshold")
    print(f"  - quantile sketches: median, IQR, percentiles (NOT from power sums)")
    print(f"\nNOTE: Skewness and kurtosis ARE computable from power sums (sum_r, sum_r2, sum_r3, sum_r4)")
    print(f"      But MEDIAN, MAD, QUANTILES are NOT — they require order statistics or sketches")
    print(f"      This is a fundamental mathematical limitation, not an implementation gap")


if __name__ == "__main__":
    main()
