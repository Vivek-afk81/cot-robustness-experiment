"""
solve_bug.py

Run this from your repo root (same level as results/), NOT from here.

Does two things:
1. Per-problem comparison of trial 1 vs trial 2 for every condition
   (baseline_control, reversed, shuffled, partial), including a separate
   degenerate/non-degenerate split for partial. Aggregate accuracy matching
   between trials does NOT guarantee per-problem agreement (temp=0.0 is not
   fully deterministic, per your own documented flip-rate finding) -- this
   checks that directly instead of assuming.
2. Surfaces any records with a non-null "error" field in either trial, so
   you know exactly which problem_id/condition pairs need a manual rerun
   rather than silently counting an API failure as "model got it wrong."
"""

import json


RESULTS_TRIAL1 = "results/stage2_results_v2_trial1.jsonl"
RESULTS_TRIAL2 = "results/stage2_results_v2_trial2.jsonl"
BASELINE_TRIAL1 = "results/stage2_baseline_control_v2_trial1.jsonl"
BASELINE_TRIAL2 = "results/stage2_baseline_control_v2_trial2.jsonl"


def load_records(path):
    """Returns list of raw dicts from a jsonl file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_by_condition(records, condition_name=None):
    """
    Returns {problem_id: record}, optionally filtered to a single condition
    (needed for the multi-condition stage2_results files; not needed for the
    single-condition baseline_control files).
    """
    out = {}
    for r in records:
        if condition_name is not None and r.get("condition") != condition_name:
            continue
        out[r["problem_id"]] = r
    return out


def report_errors(records, trial_label, path_label):
    errors = [r for r in records if r.get("error")]
    if not errors:
        print(f"  {trial_label} ({path_label}): no error records. Clean.")
        return
    print(f"  {trial_label} ({path_label}): {len(errors)} ERROR record(s) found:")
    for r in errors:
        print(f"    - problem_id={r['problem_id']}, condition={r.get('condition')}, "
              f"error={r['error']!r}")
        print(f"      -> This was silently counted as correct=False. "
              f"Rerun this problem_id/condition manually before trusting accuracy numbers.")


def compare_condition(t1_by_id, t2_by_id, label, degenerate_lookup=None):
    common_ids = set(t1_by_id) & set(t2_by_id)
    only_t1 = set(t1_by_id) - set(t2_by_id)
    only_t2 = set(t2_by_id) - set(t1_by_id)

    agree = []
    flipped_to_correct = []   # wrong in trial1, correct in trial2
    flipped_to_wrong = []     # correct in trial1, wrong in trial2

    for pid in common_ids:
        c1 = t1_by_id[pid]["correct"]
        c2 = t2_by_id[pid]["correct"]
        if c1 == c2:
            agree.append(pid)
        elif (not c1) and c2:
            flipped_to_correct.append(pid)
        elif c1 and (not c2):
            flipped_to_wrong.append(pid)

    print(f"\n--- {label} ---")
    print(f"Matched problem_ids: {len(common_ids)}  "
          f"(only-in-trial1: {len(only_t1)}, only-in-trial2: {len(only_t2)})")
    if only_t1:
        print(f"  only-in-trial1 ids: {sorted(only_t1)}")
    if only_t2:
        print(f"  only-in-trial2 ids: {sorted(only_t2)}")
    print(f"Agree (same verdict both trials): {len(agree)}/{len(common_ids)}")
    print(f"Flipped wrong->correct (trial1->trial2): {len(flipped_to_correct)} "
          f"{sorted(flipped_to_correct) if flipped_to_correct else ''}")
    print(f"Flipped correct->wrong (trial1->trial2): {len(flipped_to_wrong)} "
          f"{sorted(flipped_to_wrong) if flipped_to_wrong else ''}")

    net_change = len(flipped_to_correct) - len(flipped_to_wrong)
    print(f"Net accuracy change from flips: {net_change:+d} "
          f"(this is why aggregate counts can match even with real per-problem churn)")

    if degenerate_lookup is not None:
        print("  Degenerate split (partial only):")
        for degenerate_flag, tag in [(True, "degenerate (3-4 step)"), (False, "non-degenerate (5+ step)")]:
            subset = [pid for pid in common_ids if degenerate_lookup.get(pid) == degenerate_flag]
            sub_agree = sum(1 for pid in subset if t1_by_id[pid]["correct"] == t2_by_id[pid]["correct"])
            sub_flip_pos = sum(1 for pid in subset
                                if not t1_by_id[pid]["correct"] and t2_by_id[pid]["correct"])
            sub_flip_neg = sum(1 for pid in subset
                                if t1_by_id[pid]["correct"] and not t2_by_id[pid]["correct"])
            print(f"    {tag}: n={len(subset)}, agree={sub_agree}, "
                  f"flip_wrong->correct={sub_flip_pos}, flip_correct->wrong={sub_flip_neg}")


def run():
    t1_all = load_records(RESULTS_TRIAL1)
    t2_all = load_records(RESULTS_TRIAL2)
    bc1_all = load_records(BASELINE_TRIAL1)
    bc2_all = load_records(BASELINE_TRIAL2)

    print("=" * 60)
    print("STEP 1: ERROR RECORD CHECK")
    print("=" * 60)
    report_errors(t1_all, "Trial 1", "stage2_results")
    report_errors(t2_all, "Trial 2", "stage2_results")
    report_errors(bc1_all, "Trial 1", "baseline_control")
    report_errors(bc2_all, "Trial 2", "baseline_control")

    print("\n" + "=" * 60)
    print("STEP 2: PER-PROBLEM TRIAL 1 vs TRIAL 2 COMPARISON")
    print("=" * 60)

    # baseline_control -- single-condition files, no filter needed
    bc1_by_id = index_by_condition(bc1_all)
    bc2_by_id = index_by_condition(bc2_all)
    compare_condition(bc1_by_id, bc2_by_id, "Baseline-control")

    # reversed / shuffled / partial -- filtered out of the multi-condition files
    for condition in ["reversed", "shuffled", "partial"]:
        t1_by_id = index_by_condition(t1_all, condition)
        t2_by_id = index_by_condition(t2_all, condition)

        degenerate_lookup = None
        if condition == "partial":
            # prefer trial2's degenerate flags; fall back to trial1 if a pid
            # is missing from trial2 (e.g. the error-record case)
            degenerate_lookup = {}
            for pid, r in t1_by_id.items():
                degenerate_lookup[pid] = r["degenerate"]
            for pid, r in t2_by_id.items():
                degenerate_lookup[pid] = r["degenerate"]

        compare_condition(t1_by_id, t2_by_id, condition.capitalize(), degenerate_lookup)

    print("\n" + "=" * 60)
    print("DONE. Read the 'Net accuracy change from flips' lines above --")
    print("a net of 0 with nonzero flips in both directions means the aggregate")
    print("number is masking real churn, which matters for how confidently you")
    print("can call trial 2 a 'clean confirmation' of trial 1.")
    print("=" * 60)


if __name__ == "__main__":
    run()


# """
# fix_pid11.py

# Reruns problem_id=11, condition=shuffled (the one that hit a 429 rate-limit
# error in trial 2), and patches the corrected record into
# results/stage2_results_v2_trial2.jsonl -- replacing the error record in
# place rather than appending, so the file doesn't end up with a duplicate.

# Run this from your repo root.
# """

# import json

# from utils import get_model_response_stage2, parse_response, normalize_answer


# RESULTS_PATH = "results/stage2_results_v2_trial2.jsonl"
# SHUFFLED_PATH = "data/stage2_shuffled.jsonl"
# TARGET_PROBLEM_ID = 11
# TARGET_CONDITION = "shuffled"


# def find_source_record(path, problem_id):
#     with open(path) as f:
#         for line in f:
#             r = json.loads(line)
#             if r["problem_id"] == problem_id:
#                 return r
#     raise ValueError(f"problem_id {problem_id} not found in {path}")


# def run():
#     source = find_source_record(SHUFFLED_PATH, TARGET_PROBLEM_ID)
#     question = source["question"]
#     ground_truth = source["ground_truth"]
#     permuted_steps = source["permuted_steps"]

#     print(f"Rerunning problem_id={TARGET_PROBLEM_ID}, condition={TARGET_CONDITION}...")
#     raw_response = get_model_response_stage2(question, permuted_steps)
#     _, parsed_answer = parse_response(raw_response)

#     norm_parsed = normalize_answer(parsed_answer)
#     norm_truth = normalize_answer(ground_truth)
#     is_correct = (norm_parsed is not None) and (norm_parsed == norm_truth)

#     fixed_record = {
#         "trial": 2,
#         "problem_id": TARGET_PROBLEM_ID,
#         "bucket": source["bucket"],
#         "condition": TARGET_CONDITION,
#         "n_steps": source["n_steps"],
#         "degenerate": source["degenerate"],
#         "ground_truth": ground_truth,
#         "raw_response": raw_response,
#         "parsed_answer": parsed_answer,
#         "correct": is_correct,
#         "error": None,
#     }

#     status = "correct" if is_correct else "WRONG"
#     print(f"Result: {status} (parsed={parsed_answer}, truth={ground_truth})")

#     # Load all existing records, replace the one matching problem_id + condition,
#     # then rewrite the file (avoids leaving a duplicate or an append-mode collision).
#     all_records = []
#     replaced = False
#     with open(RESULTS_PATH) as f:
#         for line in f:
#             r = json.loads(line)
#             if r["problem_id"] == TARGET_PROBLEM_ID and r["condition"] == TARGET_CONDITION:
#                 all_records.append(fixed_record)
#                 replaced = True
#             else:
#                 all_records.append(r)

#     if not replaced:
#         raise RuntimeError(
#             f"Could not find an existing record for problem_id={TARGET_PROBLEM_ID}, "
#             f"condition={TARGET_CONDITION} in {RESULTS_PATH} -- nothing was replaced."
#         )

#     with open(RESULTS_PATH, "w") as f:
#         for r in all_records:
#             f.write(json.dumps(r) + "\n")

#     print(f"\nPatched {RESULTS_PATH} in place -- error record replaced with real result.")
#     print("Re-run solve_bug.py to confirm the error count is now zero and to see")
#     print("whether pid 11 actually flipped correct<->wrong for real.")


# if __name__ == "__main__":
#     run()