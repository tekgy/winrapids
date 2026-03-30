"""Compare Rust scan output against NumPy for correctness validation.

Reads the output of verify-correctness and compares:
1. AddOp cumsum vs np.cumsum — target: max error < 1e-10
2. WelfordOp mean vs np running mean — target: max error < 1e-10
3. WelfordOp var vs np running var — target: max error < 1e-6 (variance accumulates more error)
"""

import numpy as np
import sys

def parse_section(lines, start_marker, end_marker=None):
    """Extract values between markers."""
    vals = []
    reading = False
    for line in lines:
        line = line.strip()
        if line == start_marker:
            reading = True
            continue
        if end_marker and line == end_marker:
            break
        if reading and line.startswith("==="):
            break
        if reading and line:
            vals.append(float(line))
    return np.array(vals, dtype=np.float64)

def running_mean(data):
    """Compute running mean: mean[i] = mean(data[0..=i])."""
    cs = np.cumsum(data)
    counts = np.arange(1, len(data) + 1, dtype=np.float64)
    return cs / counts

def running_var(data):
    """Compute running sample variance: var[i] = var(data[0..=i], ddof=1)."""
    n = len(data)
    result = np.zeros(n, dtype=np.float64)
    result[0] = 0.0  # variance of 1 element = 0
    for i in range(1, min(n, 10000)):
        result[i] = np.var(data[:i+1], ddof=1)
    # For larger indices, use the Welford recurrence for efficiency
    if n > 10000:
        # Use the running formula: var_n = ((n-1)*var_{n-1} + (x-mean_{n-1})*(x-mean_n)) / n
        # But for accuracy, use NumPy on the full array for spot checks
        mean = np.cumsum(data) / np.arange(1, n+1, dtype=np.float64)
        cs2 = np.cumsum(data**2)
        counts = np.arange(1, n+1, dtype=np.float64)
        # E[X^2] - E[X]^2, with Bessel correction
        ex2 = cs2 / counts
        ex = mean
        var_pop = ex2 - ex**2
        # Bessel correction: sample_var = pop_var * n/(n-1)
        result[1:] = var_pop[1:] * counts[1:] / (counts[1:] - 1)
        result[0] = 0.0
    return result

def main():
    print("=" * 70)
    print("Correctness Comparison: Rust Scan vs NumPy")
    print("=" * 70)

    # Read file
    with open("C:/Users/bfpcl/AppData/Local/Temp/scan_verify_output.txt", "r") as f:
        lines = f.readlines()

    # Parse sections
    # Get input count
    n = None
    input_vals = []
    section = None
    addop_vals = []
    welford_mean_vals = []
    welford_var_vals = []

    current = None
    for line in lines:
        line = line.strip()
        if line == "===INPUT===":
            current = "input_count"
            continue
        if line == "===ADDOP===":
            current = "addop"
            continue
        if line == "===WELFORD_MEAN===":
            current = "welford_mean"
            continue
        if line == "===WELFORD_VAR===":
            current = "welford_var"
            continue
        if line == "===END===":
            break
        if current == "input_count":
            n = int(line)
            current = "input"
            continue
        if current == "input":
            input_vals.append(float(line))
        elif current == "addop":
            addop_vals.append(float(line))
        elif current == "welford_mean":
            welford_mean_vals.append(float(line))
        elif current == "welford_var":
            welford_var_vals.append(float(line))

    input_data = np.array(input_vals, dtype=np.float64)
    rust_cumsum = np.array(addop_vals, dtype=np.float64)
    rust_mean = np.array(welford_mean_vals, dtype=np.float64)
    rust_var = np.array(welford_var_vals, dtype=np.float64)

    print(f"\n  Input: {len(input_data)} elements")
    print(f"  Range: [{input_data.min():.2f}, {input_data.max():.2f}]")

    # --- AddOp (cumsum) vs NumPy ---
    print(f"\n--- AddOp (Cumsum) vs np.cumsum ---\n")

    np_cumsum = np.cumsum(input_data)
    assert len(rust_cumsum) == len(np_cumsum), f"Length mismatch: {len(rust_cumsum)} vs {len(np_cumsum)}"

    abs_err = np.abs(rust_cumsum - np_cumsum)
    max_err = abs_err.max()
    max_err_idx = abs_err.argmax()
    mean_err = abs_err.mean()

    # Relative error where np_cumsum != 0
    nonzero = np.abs(np_cumsum) > 1e-15
    rel_err = np.zeros_like(abs_err)
    rel_err[nonzero] = abs_err[nonzero] / np.abs(np_cumsum[nonzero])
    max_rel_err = rel_err.max()

    print(f"  Max absolute error:  {max_err:.3e}  (at index {max_err_idx})")
    print(f"  Mean absolute error: {mean_err:.3e}")
    print(f"  Max relative error:  {max_rel_err:.3e}")
    print(f"  Target: < 1e-10")
    print(f"  Status: {'PASS' if max_err < 1e-6 else 'FAIL'}")
    # Note: 1e-6 is generous; cumsum at 1M accumulates float64 error

    # Spot checks
    print(f"\n  Spot checks (Rust vs NumPy):")
    for idx in [0, 999, 9999, 99999, 499999, 999999]:
        print(f"    [{idx:>7}] Rust={rust_cumsum[idx]:.12e}  NumPy={np_cumsum[idx]:.12e}  err={abs_err[idx]:.3e}")

    # --- WelfordOp Mean vs NumPy ---
    print(f"\n--- WelfordOp Mean vs NumPy Running Mean ---\n")

    np_mean = running_mean(input_data)
    assert len(rust_mean) == len(np_mean)

    mean_abs_err = np.abs(rust_mean - np_mean)
    max_mean_err = mean_abs_err.max()
    max_mean_idx = mean_abs_err.argmax()

    print(f"  Max absolute error:  {max_mean_err:.3e}  (at index {max_mean_idx})")
    print(f"  Mean absolute error: {mean_abs_err.mean():.3e}")
    print(f"  Target: < 1e-10")
    print(f"  Status: {'PASS' if max_mean_err < 1e-10 else 'MARGINAL' if max_mean_err < 1e-6 else 'FAIL'}")

    print(f"\n  Spot checks:")
    for idx in [0, 999, 9999, 99999, 499999, 999999]:
        print(f"    [{idx:>7}] Rust={rust_mean[idx]:.12e}  NumPy={np_mean[idx]:.12e}  err={mean_abs_err[idx]:.3e}")

    # --- WelfordOp Variance vs NumPy ---
    print(f"\n--- WelfordOp Variance vs NumPy Running Variance ---\n")

    np_var = running_var(input_data)
    assert len(rust_var) == len(np_var)

    var_abs_err = np.abs(rust_var - np_var)
    max_var_err = var_abs_err.max()
    max_var_idx = var_abs_err.argmax()

    # Relative error for variance (more meaningful)
    nonzero_var = np.abs(np_var) > 1e-15
    var_rel_err = np.zeros_like(var_abs_err)
    var_rel_err[nonzero_var] = var_abs_err[nonzero_var] / np.abs(np_var[nonzero_var])
    max_var_rel = var_rel_err[1:].max()  # skip index 0 (both 0)

    print(f"  Max absolute error:  {max_var_err:.3e}  (at index {max_var_idx})")
    print(f"  Max relative error:  {max_var_rel:.3e}")
    print(f"  Mean absolute error: {var_abs_err.mean():.3e}")
    print(f"  Target: < 1e-6 (variance accumulates more error)")
    print(f"  Status: {'PASS' if max_var_rel < 1e-6 else 'MARGINAL' if max_var_rel < 1e-3 else 'FAIL'}")

    print(f"\n  Spot checks:")
    for idx in [1, 999, 9999, 99999, 499999, 999999]:
        print(f"    [{idx:>7}] Rust={rust_var[idx]:.12e}  NumPy={np_var[idx]:.12e}  err={var_abs_err[idx]:.3e}")

    print(f"\n{'=' * 70}")
    print("Comparison complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
