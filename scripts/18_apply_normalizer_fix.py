"""
18_apply_normalizer_fix.py

17_rescore_with_fixed_normalizer.py was deliberately READ-ONLY -- it
reported old-vs-new accuracy and which problems flip, but never wrote
anything back to disk. That's why 04_analysis.py's most recent run still
showed the OLD numbers (baseline-control 85/89, matching the pre-fix
figure) -- 04_analysis.py reads the stored 'correct' field directly, it
does not re-derive it from raw_response.

This script actually REWRITES each results file in place, updating the
'correct' field (parsed_answer is left as-is, only 'correct' is
recomputed) using the FIXED normalize_answer. No new API calls -- purely
re-scoring already-saved raw_response/parsed_answer data, same as script
17, but this one persists the fix instead of just reporting it.

Run this ONCE, then re-run 04_analysis.py -- its output should now show
baseline-control 86/89 (96.63%) for both trials, per the numbers already
confirmed by 17_rescore_with_fixed_normalizer.py.

SAFETY: writes a `.bak` backup of each file before overwriting, so this
is reversible if anything looks wrong afterward.
"""

import json
import os
import shutil


FILES_TO_FIX = [
    "results/stage2_results_v2_trial1.jsonl",
    "results/stage2_results_v2_trial2.jsonl",
    "results/stage2_baseline_control_v2_trial1.jsonl",
    "results/stage2_baseline_control_v2_trial2.jsonl",
    # Qwen files too, so any H3 analysis script also reflects the fix:
    "results/h3_stage2_baseline_control.jsonl",
    "results/h3_stage2_results.jsonl",
]


def normalize_answer_fixed(answer):
    """Must match what's now in utils.py / utils_h3.py exactly."""
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


def fix_file(path):
    if not os.path.exists(path):
        print(f"SKIPPED -- file not found: {path}")
        return

    backup_path = path + ".bak"
    shutil.copy2(path, backup_path)

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_flipped = 0
    for r in records:
        parsed = r.get("parsed_answer")
        truth = r.get("ground_truth")
        if parsed is None or truth is None:
            continue  # leave error records untouched

        old_correct = r.get("correct", False)
        new_norm_p = normalize_answer_fixed(parsed)
        new_norm_t = normalize_answer_fixed(truth)
        new_correct = (new_norm_p is not None) and (new_norm_p == new_norm_t)

        if new_correct != old_correct:
            n_flipped += 1
        r["correct"] = new_correct

    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[{path}]  {len(records)} records, {n_flipped} 'correct' field(s) updated. "
          f"Backup saved to {backup_path}")


def run():
    print("Applying normalize_answer fix IN PLACE to stored 'correct' fields.")
    print("Backups (.bak) written before each file is overwritten.\n")

    for path in FILES_TO_FIX:
        fix_file(path)

    print("\nDONE. Re-run 04_analysis.py (and any Qwen/H3 analysis script) now --")
    print("their output should reflect the corrected numbers.")
    print("\nIf anything looks wrong, restore from the .bak files:")
    for path in FILES_TO_FIX:
        print(f"  copy {path}.bak {path}")


if __name__ == "__main__":
    run()