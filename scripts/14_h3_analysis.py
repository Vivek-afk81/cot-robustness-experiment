"""
14_h3_analysis.py

H3 analysis: computes per-condition accuracy and Robustness_tau for
qwen/qwen3.6-27b, then produces a direct comparison table against the
Llama-3.1-8B-Instant (Trial 2 average) numbers from 04_analysis.py.

McNemar's exact test is NOT run across models (different problems-sets would
apply if Stage 1 eligible sets differ); it is run for within-model condition
comparisons (same logic as 04_analysis.py, Bonferroni n=6).

Also notes the H3 interpretation clearly:
  - 27B > 8B in parameter count, so a LOWER drop in tau is the H3-direction
    result (larger model is MORE robust).
  - A *higher* drop would be surprising and worth reporting as an anomaly.
"""

import json
from scipy.stats import binomtest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

H3_STAGE1_PATH          = "data/h3_stage1_baseline.jsonl"
H3_RESULTS_PATH         = "results/h3_stage2_results.jsonl"
H3_BASELINE_CTRL_PATH   = "results/h3_stage2_baseline_control.jsonl"

# Llama reference numbers from 04_analysis.py (Trial 2 — use the averaged
# tau from notebook.md: Reversed 58.4% reported as 2-trial average).
# Hard-coded here because the Llama pipeline already ran and these are
# the paper-facing numbers.  Update if 04_analysis.py is re-run.
LLAMA_REF = {
    "baseline_control": {"n": None, "acc": None},   # filled at runtime from files if available
    "reversed":         {"acc": 0.584, "tau": None},
    "shuffled":         {"acc": None,  "tau": None},
    "partial_nd":       {"acc": None,  "tau": None},
}

LLAMA_RESULTS_PATH      = "results/stage2_results_v2_trial2.jsonl"
LLAMA_BASELINE_CTRL_PATH = "results/stage2_baseline_control_v2_trial2.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_correctness_by_id(records, condition_filter=None):
    out = {}
    for r in records:
        if condition_filter and r.get("condition") != condition_filter:
            continue
        out[r["problem_id"]] = r["correct"]
    return out


def load_degenerate_flags(results_records):
    out = {}
    for r in results_records:
        if r.get("condition") == "partial":
            out[r["problem_id"]] = r["degenerate"]
    return out


def acc(d):
    if not d:
        return 0.0, 0, 0
    total = len(d)
    correct = sum(d.values())
    return correct / total, correct, total


def mcnemar_exact(correct_a, correct_b, label_a, label_b, alpha=0.05, bonferroni_n=1):
    common_ids = set(correct_a) & set(correct_b)
    b = sum(1 for pid in common_ids if correct_a[pid] and not correct_b[pid])
    c = sum(1 for pid in common_ids if (not correct_a[pid]) and correct_b[pid])
    n_discordant = b + c

    print(f"\n  --- McNemar's test: {label_a} vs {label_b} ---")
    print(f"  Matched: {len(common_ids)}  |  b={b}  |  c={c}  |  discordant={n_discordant}")

    if n_discordant == 0:
        print("  No discordant pairs — p=1.0")
        return None

    result = binomtest(min(b, c), n_discordant, p=0.5, alternative="two-sided")
    p = result.pvalue
    ca = alpha / bonferroni_n
    sig_raw = "SIGNIFICANT" if p < alpha else "not significant"
    sig_cor = "SIGNIFICANT" if p < ca else "not significant"
    print(f"  p={p:.4f}  raw({alpha}): {sig_raw}  Bonferroni({ca:.4f}): {sig_cor}")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    # ---- H3 model results ----
    h3_results   = load_records(H3_RESULTS_PATH)
    h3_baseline  = load_records(H3_BASELINE_CTRL_PATH)

    h3_bc   = load_correctness_by_id(h3_baseline)
    h3_rev  = load_correctness_by_id(h3_results, "reversed")
    h3_shuf = load_correctness_by_id(h3_results, "shuffled")
    h3_part = load_correctness_by_id(h3_results, "partial")
    h3_deg  = load_degenerate_flags(h3_results)

    h3_part_nd = {pid: c for pid, c in h3_part.items() if not h3_deg.get(pid, True)}

    bc_acc,  bc_n,  bc_t   = acc(h3_bc)
    rev_acc, rev_n, rev_t  = acc(h3_rev)
    shuf_acc,shuf_n,shuf_t = acc(h3_shuf)
    pa_acc,  pa_n,  pa_t   = acc(h3_part)
    nd_acc,  nd_n,  nd_t   = acc(h3_part_nd)

    tau_rev  = rev_acc  / bc_acc if bc_acc else float("nan")
    tau_shuf = shuf_acc / bc_acc if bc_acc else float("nan")
    tau_pa   = pa_acc   / bc_acc if bc_acc else float("nan")
    tau_nd   = nd_acc   / bc_acc if bc_acc else float("nan")

    print("=" * 65)
    print("H3 ANALYSIS — qwen/qwen3.6-27b")
    print("=" * 65)
    print(f"\n{'Condition':<28}{'Accuracy':<20}{'Robustness_tau'}")
    print(f"{'Baseline-control':<28}{f'{bc_n}/{bc_t} ({bc_acc:.2%})':<20}{'1.000 (ref)'}")
    print(f"{'Reversed':<28}{f'{rev_n}/{rev_t} ({rev_acc:.2%})':<20}{tau_rev:.3f}")
    print(f"{'Shuffled':<28}{f'{shuf_n}/{shuf_t} ({shuf_acc:.2%})':<20}{tau_shuf:.3f}")
    print(f"{'Partial (blended)':<28}{f'{pa_n}/{pa_t} ({pa_acc:.2%})':<20}{tau_pa:.3f}")
    print(f"{'Partial (non-degenerate)':<28}{f'{nd_n}/{nd_t} ({nd_acc:.2%})':<20}{tau_nd:.3f}")

    print(f"\n  Within-model significance tests (McNemar, Bonferroni n=6):")
    N = 6
    mcnemar_exact(h3_bc, h3_rev,  "Baseline-control", "Reversed", bonferroni_n=N)
    mcnemar_exact(h3_bc, h3_shuf, "Baseline-control", "Shuffled", bonferroni_n=N)
    mcnemar_exact(h3_bc, h3_part_nd, "Baseline-control", "Partial (non-deg)", bonferroni_n=N)
    mcnemar_exact(h3_rev,  h3_shuf,    "Reversed", "Shuffled", bonferroni_n=N)
    mcnemar_exact(h3_rev,  h3_part_nd, "Reversed", "Partial (non-deg)", bonferroni_n=N)
    mcnemar_exact(h3_shuf, h3_part_nd, "Shuffled", "Partial (non-deg)", bonferroni_n=N)

    # ---- Cross-model comparison table ----
    try:
        llama_results  = load_records(LLAMA_RESULTS_PATH)
        llama_baseline = load_records(LLAMA_BASELINE_CTRL_PATH)

        llama_bc   = load_correctness_by_id(llama_baseline)
        llama_rev  = load_correctness_by_id(llama_results, "reversed")
        llama_shuf = load_correctness_by_id(llama_results, "shuffled")
        llama_part = load_correctness_by_id(llama_results, "partial")
        llama_deg  = load_degenerate_flags(llama_results)
        llama_nd   = {pid: c for pid, c in llama_part.items() if not llama_deg.get(pid, True)}

        lbc_a, _, _  = acc(llama_bc)
        lrev_a, lrev_n, lrev_t  = acc(llama_rev)
        lshuf_a,lshuf_n,lshuf_t = acc(llama_shuf)
        lnd_a,  lnd_n, lnd_t    = acc(llama_nd)

        ltau_rev  = lrev_a  / lbc_a if lbc_a else float("nan")
        ltau_shuf = lshuf_a / lbc_a if lbc_a else float("nan")
        ltau_nd   = lnd_a   / lbc_a if lbc_a else float("nan")

        print("\n\n" + "=" * 65)
        print("CROSS-MODEL COMPARISON (H3 direction check)")
        print("=" * 65)
        print("H3 prediction: 27B model should be MORE robust (higher tau)\n")
        print(f"{'Condition':<28}{'Llama-3.1-8B tau':<20}{'Qwen-27B tau':<20}{'Direction'}")
        print("-" * 65)

        def direction(llama_tau, h3_tau):
            if abs(llama_tau - h3_tau) < 0.01:
                return "≈ same"
            if h3_tau > llama_tau:
                return "✓ H3 direction (27B more robust)"
            return "✗ opposite (27B less robust)"

        print(f"{'Reversed':<28}{ltau_rev:<20.3f}{tau_rev:<20.3f}{direction(ltau_rev, tau_rev)}")
        print(f"{'Shuffled':<28}{ltau_shuf:<20.3f}{tau_shuf:<20.3f}{direction(ltau_shuf, tau_shuf)}")
        print(f"{'Partial (non-deg)':<28}{ltau_nd:<20.3f}{tau_nd:<20.3f}{direction(ltau_nd, tau_nd)}")

        print("\nNote: this is a cross-model size comparison (27B vs 8B), not a formal")
        print("H3 test (H3 called for smaller, not larger, model vs 8B baseline).")
        print("A consistent direction across conditions is suggestive but descriptive.")

    except FileNotFoundError as e:
        print(f"\n[Llama reference files not found — skipping cross-model table: {e}]")


if __name__ == "__main__":
    run()
