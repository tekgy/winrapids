"""
Investigation: Why does the merge formula fail for m3?
The merge of 3 partitions at offset 1e6 shows 88% error in m3.
"""
import math
import random

random.seed(42)

def moments_ungrouped(values):
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {'count': 0, 'sum': 0, 'min': float('inf'), 'max': float('-inf'),
                'm2': 0, 'm3': 0, 'm4': 0}
    count = float(len(clean))
    s = sum(clean)
    mn = min(clean)
    mx = max(clean)
    mean = s / count

    m2 = m3 = m4 = 0.0
    for v in clean:
        d = v - mean
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2

    return {'count': count, 'sum': s, 'min': mn, 'max': mx,
            'm2': m2, 'm3': m3, 'm4': m4}

def merge(a, b):
    na, nb = a['count'], b['count']
    n = na + nb
    if n == 0: return moments_ungrouped([])
    if na == 0: return dict(b)
    if nb == 0: return dict(a)

    s = a['sum'] + b['sum']
    delta = b['sum']/nb - a['sum']/na
    d2 = delta*delta
    d3 = d2*delta
    d4 = d2*d2

    m2 = a['m2'] + b['m2'] + d2*na*nb/n
    m3 = (a['m3'] + b['m3'] + d3*na*nb*(na-nb)/(n*n)
          + 3*delta*(na*b['m2'] - nb*a['m2'])/n)
    m4 = (a['m4'] + b['m4']
          + d4*na*nb*(na*na - na*nb + nb*nb)/(n*n*n)
          + 6*d2*(na*na*b['m2'] + nb*nb*a['m2'])/(n*n)
          + 4*delta*(na*b['m3'] - nb*a['m3'])/n)

    return {'count': n, 'sum': s,
            'min': min(a['min'], b['min']),
            'max': max(a['max'], b['max']),
            'm2': m2, 'm3': m3, 'm4': m4}

print("=" * 80)
print("INVESTIGATION: Merge m3 failure")
print("=" * 80)

# First: what is the actual m3 for the data?
data = [1e6 + i*0.1 + (-1)**i * 0.05 for i in range(1000)]
ms_full = moments_ungrouped(data)
print(f"\nFull m3:    {ms_full['m3']:.10e}")
print(f"Full m2:    {ms_full['m2']:.10e}")
print(f"Full m4:    {ms_full['m4']:.10e}")

# Is m3 supposed to be near zero? (nearly symmetric data)
print(f"\nIs data symmetric? m3/m2^1.5 = {ms_full['m3'] / ms_full['m2']**1.5:.6e}")
print("(Near-zero skewness -> m3 is small relative to m2)")

# The issue: m3 is tiny (1.16e-4) because the data is nearly symmetric.
# When merging, the delta terms involve subtracting large numbers to get small results.
# This is EXACTLY the catastrophic cancellation pattern!

# Let's test with clearly skewed data
print("\n--- Test with clearly skewed data ---")
data_skew = [1e6 + i**2 * 0.01 for i in range(100)]  # skewed right
ms_skew = moments_ungrouped(data_skew)
ms_a = moments_ungrouped(data_skew[:50])
ms_b = moments_ungrouped(data_skew[50:])
ms_merged_skew = merge(ms_a, ms_b)

print(f"Skewed full m3:    {ms_skew['m3']:.10e}")
print(f"Skewed merged m3:  {ms_merged_skew['m3']:.10e}")
err = abs(ms_skew['m3'] - ms_merged_skew['m3']) / abs(ms_skew['m3'])
print(f"Skewed m3 err:     {err:.2e}")

# Test at zero offset
print("\n--- Test with zero offset ---")
data_zero = [i*0.1 + (-1)**i * 0.05 for i in range(1000)]
ms_zero = moments_ungrouped(data_zero)
ms_za = moments_ungrouped(data_zero[:300])
ms_zb = moments_ungrouped(data_zero[300:700])
ms_zc = moments_ungrouped(data_zero[700:])
ms_zero_merged = merge(merge(ms_za, ms_zb), ms_zc)

print(f"Zero-offset full m3:    {ms_zero['m3']:.10e}")
print(f"Zero-offset merged m3:  {ms_zero_merged['m3']:.10e}")
err = abs(ms_zero['m3'] - ms_zero_merged['m3']) / abs(ms_zero['m3']) if ms_zero['m3'] != 0 else 0
print(f"Zero-offset m3 err:     {err:.2e}")

# Test at 1e6 offset with 2 partitions
print("\n--- Test at 1e6 with 2 partitions ---")
data_2 = [1e6 + i*0.1 + (-1)**i * 0.05 for i in range(1000)]
ms_2a = moments_ungrouped(data_2[:500])
ms_2b = moments_ungrouped(data_2[500:])
ms_2_merged = merge(ms_2a, ms_2b)

print(f"2-part full m3:    {ms_full['m3']:.10e}")
print(f"2-part merged m3:  {ms_2_merged['m3']:.10e}")
err = abs(ms_full['m3'] - ms_2_merged['m3']) / abs(ms_full['m3']) if ms_full['m3'] != 0 else 0
print(f"2-part m3 err:     {err:.2e}")

# Now decompose: what are the delta terms?
print("\n--- Decomposing the merge for 3 partitions at 1e6 ---")
ms_a_info = moments_ungrouped(data[:300])
ms_b_info = moments_ungrouped(data[300:700])
ms_c_info = moments_ungrouped(data[700:])

print(f"Partition A: mean={ms_a_info['sum']/ms_a_info['count']:.10f}, m3={ms_a_info['m3']:.6e}")
print(f"Partition B: mean={ms_b_info['sum']/ms_b_info['count']:.10f}, m3={ms_b_info['m3']:.6e}")
print(f"Partition C: mean={ms_c_info['sum']/ms_c_info['count']:.10f}, m3={ms_c_info['m3']:.6e}")

# Merge A+B
ms_ab = merge(ms_a_info, ms_b_info)
print(f"\nAfter merge(A,B): m3={ms_ab['m3']:.6e}")

# Then merge (A+B)+C
ms_abc = merge(ms_ab, ms_c_info)
print(f"After merge(A+B,C): m3={ms_abc['m3']:.6e}")
print(f"Full computation:   m3={ms_full['m3']:.6e}")

err_ab = abs(ms_ab['m3'] - moments_ungrouped(data[:700])['m3']) / abs(moments_ungrouped(data[:700])['m3']) if moments_ungrouped(data[:700])['m3'] != 0 else 0
print(f"\nMerge(A,B) m3 error: {err_ab:.2e}")

# The real question: is the m3 just too small to survive merging?
print(f"\n--- Relative magnitude analysis ---")
print(f"m3 = {ms_full['m3']:.6e}")
print(f"m2 = {ms_full['m2']:.6e}")
print(f"m4 = {ms_full['m4']:.6e}")
print(f"m3/m2 ratio: {ms_full['m3']/ms_full['m2']:.6e}")
print(f"Skewness = {ms_full['m3']/ms_full['count'] / (ms_full['m2']/ms_full['count'])**1.5:.6e}")
print()
print("The m3 is ~1e-4 while m2 is ~8e5. The ratio is ~1e-10.")
print("The merge formula adds large terms that cancel to produce this tiny result.")
print("This is the SAME cancellation pattern as naive variance, but for m3 merge.")
print()
print("VERDICT: The merge formula is mathematically correct but numerically fragile")
print("when m3 is tiny relative to m2. For NEARLY SYMMETRIC DATA at large offset,")
print("the merged m3 loses precision. The skewness derived from it will be wrong.")
print()
print("This affects: any distributed/parallel computation where data is split")
print("across partitions and moments are merged. If the data is nearly symmetric")
print("(|skewness| < 1e-6) AND at large offset, the merged skewness will be")
print("unreliable.")
print()
print("Severity: MEDIUM. The full two-pass computation is correct. Only the merge")
print("path is affected, and only for near-zero skewness at large offset.")
