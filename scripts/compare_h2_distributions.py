"""
compare_h2_distributions.py

The real test of H2: does the divergence-position clustering found in the
self-report data (Step 3, not Step 2) hold up when using YOUR manual
judgment as ground truth instead of the model's self-report?

This is a stronger result than the self-report agreement-rate check alone —
that check told you HOW MUCH to trust the self-report; this tells you
whether the underlying positional pattern is real, using human judgment
directly, independent of self-report reliability.

CONFIG: adjust these paths if your actual filenames differ — I don't have
visibility into the exact key/mapping file your existing comparison script
used (it printed problem_id/condition per annotation_index, so that mapping
exists somewhere on your end).
"""

import json
from collections import Counter

MANUAL_SHEET_PATH = "data/h2_manual_annotation_sheet.jsonl"   
ANNOTATION_KEY_PATH = "data/h2_manual_answer_key.jsonl"          
                                                        
SELF_REPORT_PATH = "results/h2_self_report_trial1.jsonl"


def load_manual_annotations():
    """Returns list of dicts with annotation_index and the manual guess."""
    out = []
    with open(MANUAL_SHEET_PATH) as f:
        for line in f:
            r = json.loads(line)
            out.append({
                "annotation_index": r["annotation_index"],
                "manual_guess": r["YOUR_first_divergence_step_guess"],
            })
    return out


def load_annotation_key():
    """Returns {annotation_index: (problem_id, condition)}."""
    key = {}
    with open(ANNOTATION_KEY_PATH) as f:
        for line in f:
            r = json.loads(line)
            key[r["annotation_index"]] = (r["problem_id"], r["condition"])
    return key


def load_self_report():
    """Returns {(problem_id, condition): self_reported_step}."""
    out = {}
    with open(SELF_REPORT_PATH) as f:
        for line in f:
            r = json.loads(line)
            out[(r["problem_id"], r["condition"])] = r["permuted_divergence_step"]
    return out


def tally(values):
    """Splits into a Counter of numeric steps + a count of 'none'/None."""
    numeric = Counter()
    none_count = 0
    for v in values:
        if v in (None, "none"):
            none_count += 1
        else:
            numeric[v] += 1
    return numeric, none_count


def run():
    manual = load_manual_annotations()
    key = load_annotation_key()
    self_report = load_self_report()

    manual_values = []
    self_report_values_matched = []  # only the ones with a manual counterpart
    per_condition_manual = {"reversed": [], "shuffled": [], "partial": []}

    unmatched = []
    for a in manual:
        idx = a["annotation_index"]
        if idx not in key:
            unmatched.append(idx)
            continue
        problem_id, condition = key[idx]
        manual_values.append(a["manual_guess"])
        per_condition_manual.setdefault(condition, []).append(a["manual_guess"])

        sr_value = self_report.get((problem_id, condition))
        self_report_values_matched.append(sr_value)

    if unmatched:
        print(f"WARNING: {len(unmatched)} annotation_index values had no key match: {unmatched}")
        print("Check ANNOTATION_KEY_PATH — these were excluded from the comparison.\n")

    manual_numeric, manual_none = tally(manual_values)
    sr_numeric, sr_none = tally(self_report_values_matched)

    print(f"Total matched annotations: {len(manual_values)}")
    print("\n--- Divergence-position distribution: MANUAL (ground truth) vs SELF-REPORT ---")
    all_steps = sorted(set(manual_numeric) | set(sr_numeric))
    print(f"{'Step':<8}{'Manual':<10}{'Self-report':<12}")
    for step in all_steps:
        print(f"{step:<8}{manual_numeric.get(step, 0):<10}{sr_numeric.get(step, 0):<12}")
    print(f"{'none':<8}{manual_none:<10}{sr_none:<12}")

    print("\n--- Manual distribution by condition ---")
    for cond, values in per_condition_manual.items():
        numeric, none_c = tally(values)
        print(f"  {cond}: {dict(sorted(numeric.items()))}  (none: {none_c})")

    # Which step is the mode (most common) for each?
    if manual_numeric:
        manual_mode = manual_numeric.most_common(1)[0]
        print(f"\nManual mode: Step {manual_mode[0]} ({manual_mode[1]} occurrences)")
    if sr_numeric:
        sr_mode = sr_numeric.most_common(1)[0]
        print(f"Self-report mode: Step {sr_mode[0]} ({sr_mode[1]} occurrences)")

    print("\nIf both modes land on the same step, that's real support for H2's")
    print("clustering claim (revised to whichever step it actually is), independent")
    print("of self-report reliability. If they diverge, the self-report-based")
    print("clustering finding should NOT be trusted as reflecting real behavior —")
    print("lean on the manual distribution as the honest result instead.")


if __name__ == "__main__":
    run()