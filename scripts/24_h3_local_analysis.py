"""
24_h3_local_analysis.py — accuracy table, Robustness_tau, McNemar pairwise
tests, and a character-similarity bypass spotcheck for a local model's
Stage 2 results.

Run this AFTER 23_local_run_conditions.py. Reports Partial accuracy split
by the `degenerate` flag (from get_partial), not blended — matching the
11-07-2026 decision in your project log. The bypass spotcheck is not
optional; every prior cross-model block in this project needed one before
its accuracy numbers meant anything.

UPDATED: bypass_spotcheck now also breaks Partial down by exact step count
(3-step / 4-step / 5+-step), not just the True/False degenerate flag — a
3-step chain is degenerate because it's IDENTICAL to baseline by
construction (get_partial returns it completely unchanged), while a 4-step
chain is degenerate because its only non-identity middle permutation is a
reversal (real, if limited, disruption). Blending those two together under
one "degenerate=True" bucket inflates the apparent identical-response rate
with cases that were never going to differ in the first place. Also prints
the problem_id of every identical/near-identical case per condition so you
can go read the actual raw_response pairs by hand.

Usage (can be run from anywhere, paths are resolved relative to this file):
    python 24_h3_local_analysis.py phi3-mini
"""

import sys
import json
import difflib
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import binomtest
except ImportError:
    binomtest = None

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def load_results(model_key: str):
    path = RESULTS_DIR / f"h3_local_{model_key}_stage2_results.jsonl"
    if not path.exists():
        print(f"Missing {path} — run 23_h3_local_run_conditions.py first.")
        sys.exit(1)
    return [json.loads(line) for line in open(path)]


def accuracy_table(records):
    by_cond = defaultdict(lambda: [0, 0])  # condition -> [correct, clean_total]
    partial_by_degenerate = defaultdict(lambda: [0, 0])  # degenerate flag -> [correct, total]

    for r in records:
        if r["correct"] is None:
            continue
        by_cond[r["condition"]][1] += 1
        if r["correct"]:
            by_cond[r["condition"]][0] += 1
        if r["condition"] == "partial":
            key = r.get("degenerate", False)
            partial_by_degenerate[key][1] += 1
            if r["correct"]:
                partial_by_degenerate[key][0] += 1

    print("\n=== Accuracy by condition (clean calls only) ===")
    for cond in ("baseline_control", "reversed", "shuffled"):
        correct, total = by_cond.get(cond, [0, 0])
        pct = 100 * correct / total if total else float("nan")
        print(f"  {cond:16s}: {correct}/{total} ({pct:.2f}%)")

    print("\n  partial (blended, NOT recommended for reporting — see split below):")
    correct, total = by_cond.get("partial", [0, 0])
    pct = 100 * correct / total if total else float("nan")
    print(f"    {correct}/{total} ({pct:.2f}%)")

    print("\n  partial, split by degenerate flag:")
    for key, label in ((False, "non-degenerate (>=5-step chains)"), (True, "degenerate (3-4 step chains)")):
        correct, total = partial_by_degenerate.get(key, [0, 0])
        pct = 100 * correct / total if total else float("nan")
        print(f"    {label:38s}: {correct}/{total} ({pct:.2f}%)")

    return by_cond


def robustness_tau(by_cond):
    base_correct, base_total = by_cond.get("baseline_control", [0, 0])
    base_acc = base_correct / base_total if base_total else float("nan")
    print(f"\n=== Robustness_tau (perturbed accuracy / baseline_control accuracy) ===")
    for cond in ("reversed", "shuffled", "partial"):
        correct, total = by_cond.get(cond, [0, 0])
        acc = correct / total if total else float("nan")
        tau = acc / base_acc if base_acc else float("nan")
        print(f"  {cond:16s}: tau = {tau:.3f}")


def mcnemar_pairwise(records, cond_a, cond_b):
    by_pid = defaultdict(dict)
    for r in records:
        by_pid[r["problem_id"]][r["condition"]] = r["correct"]

    only_a_right = 0
    only_b_right = 0
    for pid, conds in by_pid.items():
        a, b = conds.get(cond_a), conds.get(cond_b)
        if a is None or b is None:
            continue
        if a and not b:
            only_a_right += 1
        elif b and not a:
            only_b_right += 1

    n_discordant = only_a_right + only_b_right
    if n_discordant == 0:
        print(f"  {cond_a} vs {cond_b}: b={only_a_right}, c={only_b_right} — "
              f"zero discordant pairs, McNemar undefined/trivial. Check the "
              f"bypass spotcheck below before treating this as 'no difference'.")
        return None

    if binomtest is None:
        print(f"  {cond_a} vs {cond_b}: b={only_a_right}, c={only_b_right} "
              f"(scipy not installed — install scipy for a p-value)")
        return None

    result = binomtest(min(only_a_right, only_b_right), n_discordant, 0.5)
    print(f"  {cond_a} vs {cond_b}: b={only_a_right}, c={only_b_right}, p={result.pvalue:.4f}")
    return result.pvalue


def _step_count_bucket(n_steps):
    if n_steps == 3:
        return "3-step"
    if n_steps == 4:
        return "4-step"
    return "5+-step"


def bypass_spotcheck(records, similarity_threshold=0.90, max_ids_printed=20):
    by_pid_response = defaultdict(dict)   # pid -> {condition: raw_response}
    n_steps_by_pid = {}                   # pid -> len(original_steps), from whichever record has it

    for r in records:
        pid = r["problem_id"]
        by_pid_response[pid][r["condition"]] = r["raw_response"]
        if pid not in n_steps_by_pid:
            n_steps_by_pid[pid] = len(r.get("original_steps") or [])

    print("\n=== Bypass spotcheck (character similarity vs baseline_control) ===")
    for cond in ("reversed", "shuffled", "partial"):
        identical_pids, near_identical_pids, different_pids = [], [], []

        for pid, resp in by_pid_response.items():
            base = resp.get("baseline_control")
            other = resp.get(cond)
            if base is None or other is None:
                continue
            ratio = difflib.SequenceMatcher(None, base, other).ratio()
            if base == other:
                identical_pids.append(pid)
            elif ratio >= similarity_threshold:
                near_identical_pids.append(pid)
            else:
                different_pids.append(pid)

        n = len(identical_pids) + len(near_identical_pids) + len(different_pids)
        if n == 0:
            continue

        print(f"  {cond:10s}: identical={len(identical_pids)} ({100*len(identical_pids)/n:.1f}%), "
              f"near-identical={len(near_identical_pids)} ({100*len(near_identical_pids)/n:.1f}%), "
              f"different={len(different_pids)} ({100*len(different_pids)/n:.1f}%)  [n={n}]")

        flagged = identical_pids + near_identical_pids
        if flagged:
            shown = sorted(flagged)[:max_ids_printed]
            more = f" (+{len(flagged) - len(shown)} more)" if len(flagged) > len(shown) else ""
            print(f"    identical/near-identical problem_ids: {shown}{more}")

        # Partial-specific: break the identical/near-identical set down by
        # actual step count, since 3-step chains are identical to baseline
        # BY CONSTRUCTION (get_partial can't touch a 0-element middle) and
        # tell you nothing about model behavior, while 4-step and 5+-step
        # identical cases are genuinely informative — the prompt WAS
        # different and the model still produced the same output.
        if cond == "partial":
            bucket_counts = defaultdict(lambda: {"identical": 0, "near_identical": 0, "different": 0, "total": 0})
            for pid in identical_pids:
                b = _step_count_bucket(n_steps_by_pid.get(pid, 0))
                bucket_counts[b]["identical"] += 1
                bucket_counts[b]["total"] += 1
            for pid in near_identical_pids:
                b = _step_count_bucket(n_steps_by_pid.get(pid, 0))
                bucket_counts[b]["near_identical"] += 1
                bucket_counts[b]["total"] += 1
            for pid in different_pids:
                b = _step_count_bucket(n_steps_by_pid.get(pid, 0))
                bucket_counts[b]["different"] += 1
                bucket_counts[b]["total"] += 1

            print("    partial, broken down by actual step count:")
            for bucket_label in ("3-step", "4-step", "5+-step"):
                c = bucket_counts.get(bucket_label)
                if not c or c["total"] == 0:
                    continue
                note = "  <- identical here is GUARANTEED by construction, not model behavior" \
                    if bucket_label == "3-step" else ""
                print(f"      {bucket_label:8s}: identical={c['identical']}, "
                      f"near-identical={c['near_identical']}, different={c['different']}, "
                      f"total={c['total']}{note}")

                if bucket_label != "3-step" and (c["identical"] or c["near_identical"]):
                    bucket_flagged = sorted(
                        [pid for pid in identical_pids + near_identical_pids
                         if _step_count_bucket(n_steps_by_pid.get(pid, 0)) == bucket_label]
                    )[:max_ids_printed]
                    print(f"        problem_ids to read by hand: {bucket_flagged}")

    print("\n  If identical+near-identical is high (>~50%) for a condition, "
          "manually read a sample of that condition's raw_response before "
          "trusting its accuracy number — see the Mistral bypass "
          "investigation for what to look for. For 'partial', ignore the "
          "3-step bucket when judging bypass risk — it can't be anything "
          "but identical by construction. The 4-step and 5+-step buckets "
          "are the ones that actually tell you something.")


def main(model_key: str):
    records = load_results(model_key)
    by_cond = accuracy_table(records)
    robustness_tau(by_cond)

    print("\n=== McNemar pairwise (exact, uncorrected — apply Bonferroni "
          "yourself across however many comparisons you actually report) ===")
    pairs = [
        ("baseline_control", "reversed"),
        ("baseline_control", "shuffled"),
        ("baseline_control", "partial"),
        ("reversed", "shuffled"),
        ("reversed", "partial"),
        ("shuffled", "partial"),
    ]
    for a, b in pairs:
        mcnemar_pairwise(records, a, b)

    bypass_spotcheck(records)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("phi3-mini", "qwen2.5-3b"):
        print("Usage: python 24_local_analysis.py [phi3-mini|qwen2.5-3b]")
        sys.exit(1)
    main(sys.argv[1])