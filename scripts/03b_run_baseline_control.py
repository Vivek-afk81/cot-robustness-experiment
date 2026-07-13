"""
03b_run_baseline_control.py

Stage 2 control condition, added after discovering that switching from Stage 1's
free-generation prompt to Stage 2's "use these given steps, just answer" prompt
costs accuracy on its own — independent of any step reordering. (Found via the
degenerate Partial group: 84.21%, not ~100%, despite near-zero order change.)

This script feeds each eligible problem's ORIGINAL, UNPERMUTED Stage 1 steps
through the exact same Stage 2 prompt format used for Reversed/Shuffled/Partial.
Zero reordering happens here — this isolates the task-format-switch effect so it
can be separated out of the other three conditions' accuracy numbers.

Once this is run, the correct comparison becomes:
  Robustness_tau(condition) = Accuracy(condition) / Accuracy(Baseline-control)
rather than dividing by Stage 1's raw accuracy, which used a different prompt
entirely and isn't a fair "clean" reference point for Stage 2 conditions.

Same eligibility filter as 02_permute_conditions.py: correct AND >=3 steps (n=89).
"""

import json
import time

from utils import get_model_response_stage2, parse_response, normalize_answer


INPUT_PATH = "data/stage1_baseline_v2.jsonl"   #also updated this, same as Step 4
TRIAL = 2
OUTPUT_PATH = f"results/stage2_baseline_control_v2_trial{TRIAL}.jsonl"
MIN_STEPS = 3
SLEEP_SECONDS = 2.5


def load_eligible_records(path, min_steps):
    """Same filter as 02_permute_conditions.py — correct AND >=min_steps parsed steps."""
    eligible = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_steps = len(r["parsed_steps"]) if r["parsed_steps"] else 0
            if r["correct"] and n_steps >= min_steps:
                eligible.append(r)
    return eligible


def run():
    eligible = load_eligible_records(INPUT_PATH, MIN_STEPS)
    total = len(eligible)
    print(f"Eligible problems (correct AND >={MIN_STEPS} steps): {total}")

    correct_count = 0

    # 'a' mode — resumable. Delete/rename existing output file for a clean run.
    with open(OUTPUT_PATH, "a") as out_file:
        for i, record in enumerate(eligible, 1):
            question = record["question"]
            ground_truth = record["ground_truth"]
            original_steps = record["parsed_steps"]  # UNPERMUTED — this is the control

            print(f"[{i}/{total}] (baseline-control, problem {record['problem_id']}) "
                  f"Calling model...", end=" ", flush=True)

            try:
                raw_response = get_model_response_stage2(question, original_steps)
            except Exception as e:
                print(f"API ERROR: {e}")
                error_record = {
                    "trial": TRIAL,
                    "problem_id": record.get("problem_id", i),
                    "bucket": record["bucket"],
                    "condition": "baseline_control",
                    "n_steps": len(original_steps),
                    "degenerate": False,  # not applicable — no permutation happened
                    "ground_truth": ground_truth,
                    "raw_response": None,
                    "parsed_answer": None,
                    "correct": False,
                    "error": str(e),
                }
                out_file.write(json.dumps(error_record) + "\n")
                out_file.flush()
                time.sleep(SLEEP_SECONDS)
                continue

            _, parsed_answer = parse_response(raw_response)

            norm_parsed = normalize_answer(parsed_answer)
            norm_truth = normalize_answer(ground_truth)
            is_correct = (norm_parsed is not None) and (norm_parsed == norm_truth)

            if is_correct:
                correct_count += 1

            result_record = {
                "trial": TRIAL,
                "problem_id": record.get("problem_id", i),
                "bucket": record["bucket"],
                "condition": "baseline_control",
                "n_steps": len(original_steps),
                "degenerate": False,
                "ground_truth": ground_truth,
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "correct": is_correct,
                "error": None,
            }

            out_file.write(json.dumps(result_record) + "\n")
            out_file.flush()

            status = "correct" if is_correct else "WRONG"
            print(f"{status} (parsed={parsed_answer}, truth={ground_truth})")

            time.sleep(SLEEP_SECONDS)

    accuracy = correct_count / total
    # Compute Stage 1 accuracy dynamically instead of hardcoding it — a hardcoded
    # number here is exactly what caused a stale/misleading print statement last time.
    stage1_total = 0
    stage1_correct = 0
    with open(INPUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            stage1_total += 1
            if r["correct"]:
                stage1_correct += 1
    stage1_accuracy = stage1_correct / stage1_total
 
    print("\n--- DONE ---")
    print(f"Baseline-control accuracy: {correct_count}/{total} ({accuracy:.2%})")
    print("\nThis is the task-format-switch effect, isolated from any reordering.")
    print("Compare against:")
    print(f"  Stage 1 (free generation): {stage1_correct}/{stage1_total} ({stage1_accuracy:.2%})")
    print("\nUse the baseline-control number above (not Stage 1's), as the denominator")
    print("for Robustness_tau when evaluating Reversed / Shuffled / Partial(non-degenerate).")
    print("(Partial degenerate/non-degenerate split is NOT computed here — run that")
    print("check separately against results/stage2_results_v2_trial1.jsonl.)")
 
 
if __name__ == "__main__":
    run()
