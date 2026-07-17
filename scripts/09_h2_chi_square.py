"""
09_h2_chi_square.py

Weighted chi-square test for H2's positional-clustering claim.

The null hypothesis is: for each case, the divergence step is drawn
uniformly from {1, 2, ..., n_steps} for that case's chain length.
Because chains have different lengths (3 to 15), a flat uniform over
all observed step values would be wrong — a step-10 divergence is only
possible in cases with ≥ 10 steps.

Approach:
  For each step value j in {1, ..., max_steps}:
    E_j = sum over all 30 cases of (1/n_steps_i) for cases where j <= n_steps_i
    O_j = observed count of divergences at step j

  Then chi-square = sum_j (O_j - E_j)^2 / E_j, with df = (number of bins used) - 1.

  Bins with E_j < 1 are merged into an "overflow" bin to satisfy chi-square
  assumptions (expected count ≥ 1 per bin; ideally ≥ 5, but we note when
  this isn't met).

Also reports a Monte Carlo permutation p-value (10,000 draws) as a
non-parametric cross-check, in case the chi-square approximation is
questionable with these small expected counts.

Usage:
  python scripts/09_h2_chi_square.py
"""

import json
import sys
import random
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

SHEET_PATH = "data/h2_manual_annotation_sheet.jsonl"
KEY_PATH = "data/h2_manual_answer_key.jsonl"


def load_cases():
    """Returns list of (observed_step, n_steps) for all numeric-guess cases."""
    with open(SHEET_PATH) as f:
        sheet = [json.loads(l) for l in f if l.strip()]
    with open(KEY_PATH) as f:
        key = {json.loads(l)["annotation_index"]: json.loads(l) for l in f if l.strip()}

    cases = []
    for r in sheet:
        g = r.get("CAREFUL_first_divergence_step_guess")
        if isinstance(g, int):
            n = key[r["annotation_index"]]["n_steps"]
            cases.append((g, n))
    return cases


def compute_expected(cases, max_step):
    """Expected count at each step j under H0: uniform(1..n_steps_i) per case."""
    expected = {}
    for j in range(1, max_step + 1):
        expected[j] = sum(1.0 / n for _, n in cases if j <= n)
    return expected


def chi_square_stat(observed, expected, bins):
    """Compute chi-square statistic for the given bins."""
    stat = 0.0
    for b in bins:
        o = observed.get(b, 0)
        e = expected[b]
        stat += (o - e) ** 2 / e
    return stat


def run():
    cases = load_cases()
    n_cases = len(cases)
    max_step = max(n for _, n in cases)

    observed = Counter(g for g, _ in cases)
    expected = compute_expected(cases, max_step)

    print(f"N = {n_cases} cases with numeric divergence guess")
    print(f"Chain lengths: {Counter(n for _, n in cases)}")
    print(f"Max step: {max_step}")

    # Show raw observed vs expected
    print(f"\n{'Step':<6}{'Obs':>6}{'Exp':>8}{'O-E':>8}{'(O-E)²/E':>10}")
    print("-" * 40)
    for j in range(1, max_step + 1):
        o = observed.get(j, 0)
        e = expected[j]
        if e > 0:
            contrib = (o - e) ** 2 / e
        else:
            contrib = 0
        flag = " *" if 0 < e < 1 else ("  **" if e < 0.5 and e > 0 else "")
        print(f"{j:<6}{o:>6}{e:>8.2f}{o - e:>8.2f}{contrib:>10.3f}{flag}")

    # Merge bins with E < 1 into overflow
    main_bins = []
    overflow_obs = 0
    overflow_exp = 0.0
    for j in range(1, max_step + 1):
        if expected[j] >= 1.0:
            main_bins.append(j)
        else:
            overflow_obs += observed.get(j, 0)
            overflow_exp += expected[j]

    # Build merged observed/expected
    merged_obs = {b: observed.get(b, 0) for b in main_bins}
    merged_exp = {b: expected[b] for b in main_bins}

    if overflow_exp > 0:
        main_bins.append("overflow")
        merged_obs["overflow"] = overflow_obs
        merged_exp["overflow"] = overflow_exp

    print(f"\n--- After merging bins with E < 1 into overflow ---")
    print(f"Bins used: {main_bins}")
    low_exp_bins = [b for b in main_bins if merged_exp[b] < 5]
    if low_exp_bins:
        print(f"WARNING: bins with E < 5 (chi-square approximation may be weak): {low_exp_bins}")

    chi2 = chi_square_stat(merged_obs, merged_exp, main_bins)
    df = len(main_bins) - 1

    # p-value from scipy if available, else from chi-square table
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df)
        print(f"\nChi-square = {chi2:.3f}, df = {df}, p = {p_value:.6f}")
        if p_value < 0.001:
            print(f"  → p < 0.001 — reject H0 (uniform) at any conventional α")
        elif p_value < 0.01:
            print(f"  → p < 0.01 — reject H0 at α = 0.01")
        elif p_value < 0.05:
            print(f"  → p < 0.05 — reject H0 at α = 0.05")
        else:
            print(f"  → p ≥ 0.05 — fail to reject H0")
    except ImportError:
        print(f"\nChi-square = {chi2:.3f}, df = {df}")
        print("  (scipy not installed — skipping analytical p-value)")
        p_value = None

    # Monte Carlo permutation test as cross-check
    print("\n--- Monte Carlo permutation test (10,000 draws) ---")
    rng = random.Random(42)
    n_sims = 10_000
    more_extreme = 0

    for _ in range(n_sims):
        # For each case, draw uniformly from 1..n_steps
        sim_obs = Counter()
        for _, n in cases:
            sim_obs[rng.randint(1, n)] += 1

        # Compute chi-square for this draw using same bins
        sim_merged_obs = {b: sim_obs.get(b, 0) for b in main_bins if b != "overflow"}
        if "overflow" in main_bins:
            sim_merged_obs["overflow"] = sum(
                sim_obs.get(j, 0) for j in range(1, max_step + 1)
                if j not in [b for b in main_bins if b != "overflow"]
            )
        sim_chi2 = chi_square_stat(sim_merged_obs, merged_exp, main_bins)
        if sim_chi2 >= chi2:
            more_extreme += 1

    mc_p = more_extreme / n_sims
    print(f"Observed chi-square: {chi2:.3f}")
    print(f"Simulations with chi2 >= observed: {more_extreme}/{n_sims}")
    print(f"Monte Carlo p-value: {mc_p:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    analytical = f"p = {p_value:.4f}" if p_value is not None else "N/A"
    print(f"  Chi-square({df}) = {chi2:.3f}")
    print(f"  Analytical p:    {analytical}")
    print(f"  Monte Carlo p:   {mc_p:.4f}")

    if (p_value is not None and p_value < 0.05) or mc_p < 0.05:
        print("  → The positional distribution is significantly non-uniform.")
        print("    Divergences cluster early (Steps 1-3), not spread evenly.")
    else:
        print("  → Cannot reject a uniform distribution at α = 0.05.")
        print("    The early-clustering pattern may be due to chance.")


if __name__ == "__main__":
    run()
