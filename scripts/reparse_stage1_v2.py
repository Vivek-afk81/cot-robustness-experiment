"""
Re-parses every stored Stage 1 raw response using the FIXED parse_response()
(group(2) bug fix). Does NOT call the API again — uses the raw_response text
already saved in stage1_baseline_reparsed.jsonl.

Why this script exists: parse_response() had a bug where current_step pulled
from the wrong regex capture group, silently reducing every "N." or "(N)."
step to just its own number (e.g. ['1','2','3','4','5']) instead of the real
step text. This corrupted 63/89 eligible records. This script fixes that
retroactively without burning API calls, since the raw text was always saved.

Output: data/stage1_baseline_v2.jsonl — the new source of truth going forward.
Old file (stage1_baseline_reparsed.jsonl) is left untouched, for the record —
worth keeping as evidence of the bug for the eventual paper's error-analysis
or reproducibility appendix.
"""

import json
from utils import parse_response, normalize_answer


INPUT_PATH = "data/stage1_baseline_reparsed.jsonl"
OUTPUT_PATH = "data/stage1_baseline_v2.jsonl"


def run():
    total = 0
    correct_count = 0
    changed_step_count = 0

    with open(INPUT_PATH) as f_in, open(OUTPUT_PATH, "w") as f_out:
        for line in f_in:
            r = json.loads(line)
            total += 1

            old_steps = r["parsed_steps"]
            old_n = len(old_steps) if old_steps else 0

            # Re-parse from the raw text using the FIXED parser.
            new_steps, new_final_answer = parse_response(r["raw_response"])

            # parsed_answer should be unaffected by this bug (different regex,
            # different code path) — but we recompute it anyway as a safety
            # check, and flag if it unexpectedly changes.
            norm_new_answer = normalize_answer(new_final_answer)
            norm_truth = normalize_answer(r["ground_truth"])
            new_correct = (norm_new_answer is not None) and (norm_new_answer == norm_truth)

            if new_correct:
                correct_count += 1

            if old_steps != new_steps:
                changed_step_count += 1

            # Flag anything where the previously-stored answer/correctness
            # doesn't match the freshly recomputed one — should not happen,
            # but worth catching rather than assuming.
            answer_mismatch = (r["parsed_answer"] != new_final_answer)
            correct_mismatch = (r["correct"] != new_correct)

            new_record = {
                "problem_id": r["problem_id"],
                "bucket": r["bucket"],
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "raw_response": r["raw_response"],
                "parsed_steps": new_steps,
                "parsed_answer": new_final_answer,
                "correct": new_correct,
                "error": r.get("error"),
                "_old_parsed_steps_for_audit": old_steps,  # kept temporarily for comparison
                "_answer_mismatch_flag": answer_mismatch,
                "_correct_mismatch_flag": correct_mismatch,
            }

            f_out.write(json.dumps(new_record) + "\n")

    print(f"Total records re-parsed: {total}")
    print(f"Records where parsed_steps changed: {changed_step_count}")
    print(f"New accuracy: {correct_count}/{total} ({correct_count/total:.2%})")
    print(f"(Old accuracy was 90/100 (90.00%) — should match unless something else broke)")


if __name__ == "__main__":
    run()