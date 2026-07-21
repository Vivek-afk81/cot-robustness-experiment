"""
17_rescore_with_fixed_normalizer.py

Re-scores every existing results file using the FIXED normalize_answer
(handles "2.00" / "6.00" style multi-zero decimals, not just ".0") without
making any new API calls -- re-parses the already-saved raw_response text
and recomputes correctness/accuracy.

Run this AFTER updating normalize_answer in both utils.py and
utils_h3.py (see normalize_answer_utils.py / normalize_answer_utils_h3.py
for the corrected function -- copy-paste it in to replace the old version
in each file).

Checks every results file this project has produced, not just Qwen's --
the same bug lived in utils.py's original normalize_answer, so Llama's and
Mistral's saved results could show the same silent undercount even if it
hasn't been noticed yet.

For each file, reports:
  - old accuracy (using the 'correct' field already stored on disk)
  - new accuracy (re-parsed and re-scored with the fixed normalizer)
  - which specific problem_ids flipped from wrong -> correct (should never
    flip correct -> wrong, since the fix only makes matching MORE lenient
    on formatting, not stricter -- flag it loudly if any DO flip that way,
    since that would indicate a bug in the fix itself, not the original)

Does not overwrite any files -- prints a report only. Decide separately
whether/how to update the stored 'correct' fields once you've reviewed
the flips.
"""

import json
import os

# (label, path) -- add more rows here as needed if other results files exist
FILES_TO_CHECK = [
    ("Llama Stage 1 baseline",              "data/stage1_baseline_v2.jsonl"),
    ("Llama Stage 2 results (trial 1)",     "results/stage2_results_v2_trial1.jsonl"),
    ("Llama Stage 2 results (trial 2)",     "results/stage2_results_v2_trial2.jsonl"),
    ("Llama baseline-control (trial 1)",    "results/stage2_baseline_control_v2_trial1.jsonl"),
    ("Llama baseline-control (trial 2)",    "results/stage2_baseline_control_v2_trial2.jsonl"),
    ("Mistral Stage 1 baseline",            "data/stage1_baseline_mistral.jsonl"),
    ("Mistral baseline-control",            "results/stage2_baseline_control_mistral.jsonl"),
    ("Mistral Stage 2 results (reversed)",  "results/stage2_results_mistral.jsonl"),
    ("Qwen 27B Stage 1 baseline",           "data/h3_stage1_baseline.jsonl"),
    ("Qwen 27B baseline-control",           "results/h3_stage2_baseline_control.jsonl"),
    ("Qwen 27B Stage 2 results (all conditions)", "results/h3_stage2_results.jsonl"),
]


def normalize_answer_fixed(answer):
    """The corrected normalizer -- must match what you paste into
    utils.py / utils_h3.py exactly, so this report reflects reality."""
    if answer is None:
        return None
    answer = str(answer).strip()
    answer = answer.replace("$", "").replace(",", "")
    answer = answer.rstrip(".")
    try:
        value = float(answer)
        if value == int(value):
            return str(int(value))
        return str(value)
    except (ValueError, OverflowError):
        if answer.endswith(".0"):
            answer = answer[:-2]
        return answer


def normalize_answer_old(answer):
    """The ORIGINAL (buggy) normalizer, reproduced here only so this
    script can show the old-vs-new comparison without needing to import
    a stale version of utils.py."""
    if answer is None:
        return None
    answer = str(answer).strip()
    answer = answer.replace("$", "").replace(",", "")
    answer = answer.rstrip(".")
    if answer.endswith(".0"):
        answer = answer[:-2]
    return answer


def rescore_file(label, path):
    if not os.path.exists(path):
        print(f"\n[{label}] SKIPPED -- file not found: {path}")
        return

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"\n[{label}] SKIPPED -- file is empty: {path}")
        return

    # Only rescore records that actually have a ground_truth/parsed_answer
    # pair to compare (skips error records where raw_response was None)
    scorable = [r for r in records if r.get("parsed_answer") is not None
                and r.get("ground_truth") is not None]

    old_correct = 0
    new_correct = 0
    flips_to_correct = []
    flips_to_wrong = []  # should be empty -- flag loudly if not

    for r in scorable:
        parsed = r["parsed_answer"]
        truth = r["ground_truth"]

        old_norm_p = normalize_answer_old(parsed)
        old_norm_t = normalize_answer_old(truth)
        old_is_correct = (old_norm_p is not None) and (old_norm_p == old_norm_t)

        new_norm_p = normalize_answer_fixed(parsed)
        new_norm_t = normalize_answer_fixed(truth)
        new_is_correct = (new_norm_p is not None) and (new_norm_p == new_norm_t)

        if old_is_correct:
            old_correct += 1
        if new_is_correct:
            new_correct += 1

        pid = r.get("problem_id", "?")
        cond = r.get("condition", "")
        if (not old_is_correct) and new_is_correct:
            flips_to_correct.append((pid, cond, parsed, truth))
        if old_is_correct and (not new_is_correct):
            flips_to_wrong.append((pid, cond, parsed, truth))

    total = len(scorable)
    old_acc = old_correct / total if total else 0
    new_acc = new_correct / total if total else 0

    print(f"\n[{label}]  ({path})")
    print(f"  Scorable records: {total}")
    print(f"  OLD accuracy: {old_correct}/{total} ({old_acc:.2%})")
    print(f"  NEW accuracy: {new_correct}/{total} ({new_acc:.2%})")

    if flips_to_correct:
        print(f"  Flipped WRONG -> CORRECT ({len(flips_to_correct)}):")
        for pid, cond, parsed, truth in flips_to_correct:
            cond_str = f" [{cond}]" if cond else ""
            print(f"    problem {pid}{cond_str}: parsed={parsed!r}, truth={truth!r}")
    else:
        print("  No flips (fixed normalizer made no difference for this file).")

    if flips_to_wrong:
        print(f"  *** WARNING *** Flipped CORRECT -> WRONG ({len(flips_to_wrong)}) --")
        print("  this should NEVER happen with this fix (it only relaxes matching).")
        print("  Investigate immediately if this list is non-empty:")
        for pid, cond, parsed, truth in flips_to_wrong:
            cond_str = f" [{cond}]" if cond else ""
            print(f"    problem {pid}{cond_str}: parsed={parsed!r}, truth={truth!r}")


def run():
    print("Re-scoring all existing results with the fixed normalize_answer.")
    print("No API calls made -- re-parsing already-saved raw_response/parsed_answer data.\n")
    print("=" * 70)

    for label, path in FILES_TO_CHECK:
        rescore_file(label, path)

    print("\n" + "=" * 70)
    print("DONE. Review any flips above, then decide whether to update the")
    print("stored 'correct' fields in each file (or just use these corrected")
    print("numbers directly in your analysis scripts / readme / paper).")
    print("\nRemember: also paste the corrected normalize_answer into utils.py")
    print("AND utils_h3.py so any FUTURE runs use the fix too -- this script")
    print("only corrects the scoring of data already collected.")


if __name__ == "__main__":
    run()