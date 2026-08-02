"""
25_h3_local_overlap.py — cross-condition overlap analysis for the bypass
spotcheck. 24_h3_local_analysis.py tells you the identical/near-identical
RATE per condition; this tells you WHICH problems overlap across
conditions, which is what actually distinguishes competing explanations:

  - Full bypass (Mistral-style): identical in reversed AND shuffled AND
    partial, for the same problems, regardless of how disrupted the
    presented order is. The model isn't using the given steps at all.

  - Reversed-specific disruption: identical in shuffled AND partial, but
    genuinely DIFFERENT under reversed. This is NOT simple bypass — the
    model IS sensitive to something about reversed order specifically that
    random shuffling doesn't trigger. Matches a documented finding
    elsewhere in this project (Llama H1): reversed order can hurt more
    than random shuffling, hypothesized as the model trying to follow a
    reversed-but-still-coherent-looking sequence literally, compounding
    errors, versus a shuffle being obviously broken and prompting a
    from-scratch re-solve instead.

  - Genuinely order-sensitive: different under ALL THREE conditions. These
    are the cleanest evidence of real step-order sensitivity, if any exist.

  - Everything else: mixed patterns that don't cleanly fit the above —
    worth a look but not the first thing to read.

This does NOT decide the mechanism for you. It narrows down which
problem_ids are worth reading by hand, and roughly how many of each
pattern exist, so the manual read is targeted instead of a blind sample.

Usage:
    python 25_h3_local_overlap.py phi3-mini
    python 25_h3_local_overlap.py qwen2.5-3b
"""

import sys
import json
import difflib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

SIMILARITY_THRESHOLD = 0.90
CONDITIONS = ("reversed", "shuffled", "partial")


def load_results(model_key: str):
    path = RESULTS_DIR / f"h3_local_{model_key}_stage2_results.jsonl"
    if not path.exists():
        print(f"Missing {path} — run 23_h3_local_run_conditions.py first.")
        sys.exit(1)
    return [json.loads(line) for line in open(path)]


def is_identical_or_near(base, other, threshold=SIMILARITY_THRESHOLD):
    if base is None or other is None:
        return None  # missing data, can't judge
    if base == other:
        return "identical"
    ratio = difflib.SequenceMatcher(None, base, other).ratio()
    return "near_identical" if ratio >= threshold else "different"


def step_bucket(n_steps):
    if n_steps == 3:
        return "3-step"
    if n_steps == 4:
        return "4-step"
    return "5+-step"


def main(model_key: str):
    records = load_results(model_key)

    by_pid = defaultdict(dict)       # pid -> {condition: raw_response}
    correct_by_pid = defaultdict(dict)  # pid -> {condition: correct}
    n_steps_by_pid = {}

    for r in records:
        pid = r["problem_id"]
        by_pid[pid][r["condition"]] = r["raw_response"]
        correct_by_pid[pid][r["condition"]] = r["correct"]
        if pid not in n_steps_by_pid:
            n_steps_by_pid[pid] = len(r.get("original_steps") or [])

    # classification per problem, per condition
    status = defaultdict(dict)  # pid -> {condition: "identical"/"near_identical"/"different"/None}
    for pid, resp in by_pid.items():
        base = resp.get("baseline_control")
        for cond in CONDITIONS:
            status[pid][cond] = is_identical_or_near(base, resp.get(cond))

    def matches_flag(pid, want_flag):
        """True if the problem's status is in {'identical','near_identical'} == want_flag for all given conditions is decided by caller."""
        return status[pid]

    all_pids = sorted(by_pid.keys())

    def is_same(pid, cond):
        return status[pid][cond] in ("identical", "near_identical")

    def is_diff(pid, cond):
        return status[pid][cond] == "different"

    full_bypass = [pid for pid in all_pids if all(is_same(pid, c) for c in CONDITIONS)]
    reversed_specific = [pid for pid in all_pids
                          if is_diff(pid, "reversed") and is_same(pid, "shuffled") and is_same(pid, "partial")]
    genuinely_sensitive = [pid for pid in all_pids if all(is_diff(pid, c) for c in CONDITIONS)]
    classified = set(full_bypass) | set(reversed_specific) | set(genuinely_sensitive)
    mixed_other = [pid for pid in all_pids if pid not in classified]

    print(f"=== Cross-condition overlap analysis: {model_key} ===")
    print(f"Total problems: {len(all_pids)}\n")

    print(f"FULL BYPASS candidates (identical/near-identical in ALL of reversed, "
          f"shuffled, partial): {len(full_bypass)}/{len(all_pids)}")
    print(f"  {full_bypass}\n")

    print(f"REVERSED-SPECIFIC DISRUPTION (different under reversed, but "
          f"identical/near-identical under BOTH shuffled and partial): "
          f"{len(reversed_specific)}/{len(all_pids)}")
    print(f"  {reversed_specific}\n")

    print(f"GENUINELY ORDER-SENSITIVE (different under all three conditions): "
          f"{len(genuinely_sensitive)}/{len(all_pids)}")
    print(f"  {genuinely_sensitive}\n")

    print(f"MIXED / other patterns (doesn't cleanly fit the above): "
          f"{len(mixed_other)}/{len(all_pids)}")
    print(f"  {mixed_other}\n")

    # Reading order suggestion
    print("=== Suggested reading order ===")
    if full_bypass:
        sample = full_bypass[:5]
        print(f"1. Full bypass candidates first (strongest, condition-independent "
              f"signal) — read: {sample}")
    if reversed_specific:
        sample = reversed_specific[:5]
        print(f"2. Reversed-specific cases (tests the 'reversed disrupts more than "
              f"shuffle' hypothesis) — read: {sample}")
    if genuinely_sensitive:
        sample = genuinely_sensitive[:5]
        print(f"3. Genuinely order-sensitive cases (the real positive evidence for "
              f"H1/H3, if the mechanism checks out) — read: {sample}")
    if not full_bypass and not reversed_specific and not genuinely_sensitive:
        print("No cases cleanly fall into the three patterns above — read a sample "
              "from 'mixed / other' instead.")

    # Breakdown of full-bypass and reversed-specific sets by step count, since
    # step count changes which mechanisms are even possible (e.g. a 3-step
    # Partial can't be anything BUT identical, so its presence in these sets
    # isn't informative the way a 5+-step case is).
    print("\n=== Step-count breakdown of the two most informative sets ===")
    for label, pid_list in (("Full bypass", full_bypass), ("Reversed-specific", reversed_specific)):
        buckets = defaultdict(list)
        for pid in pid_list:
            buckets[step_bucket(n_steps_by_pid.get(pid, 0))].append(pid)
        print(f"{label}:")
        for b in ("3-step", "4-step", "5+-step"):
            if buckets[b]:
                note = "  <- note: identical-under-partial is guaranteed here regardless of mechanism" \
                    if b == "3-step" else ""
                print(f"    {b:8s}: {len(buckets[b])} — {buckets[b]}{note}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("phi3-mini", "qwen2.5-3b"):
        print("Usage: python 25_h3_local_overlap.py [phi3-mini|qwen2.5-3b]")
        sys.exit(1)
    main(sys.argv[1])