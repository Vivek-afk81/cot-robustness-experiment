"""
03_run_conditions.py

Stage 2, part 2 of the CoT robustness experiment.
Reads the three permuted-condition files (reversed, shuffled, partial),
feeds each problem's permuted steps back to the model (Stage 2 prompt —
final-answer-only, not free re-reasoning), scores against ground truth,
and writes all results incrementally to results/stage2_results.jsonl.

Runs sequentially: all 89 Reversed problems, then all 89 Shuffled, then all
89 Partial. 267 total API calls. The `degenerate` flag from
02_permute_conditions.py is carried through untouched so Stage 4 analysis
can split/exclude degenerate Partial cases without re-deriving anything.
"""

import json
import time

from utils import get_model_response_stage2, parse_response, normalize_answer


CONDITION_FILES = {
    "reversed": "data/stage2_reversed.jsonl",
    "shuffled": "data/stage2_shuffled.jsonl",
    "partial": "data/stage2_partial.jsonl",
}
TRIAL = 2  # bump this each time you re-run — keeps trials distinguishable, never overwrites past runs
OUTPUT_PATH = f"results/stage2_results_v2_trial{TRIAL}.jsonl"
SLEEP_SECONDS = 2.5  # same conservative gap as Stage 1, Groq free tier


def load_condition_records(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run():
    all_records = []
    for condition_name, path in CONDITION_FILES.items():
        records = load_condition_records(path)
        all_records.extend(records)
        print(f"Loaded {len(records)} records for condition '{condition_name}'")

    total = len(all_records)
    correct_count = 0
    condition_correct = {"reversed": 0, "shuffled": 0, "partial": 0}
    condition_total = {"reversed": 0, "shuffled": 0, "partial": 0}

    # 'a' mode — if a run dies partway, rerun resumes by appending.
    # Delete/rename any existing results/stage2_results.jsonl first for a clean run.
    with open(OUTPUT_PATH, "a") as out_file:
        for i, record in enumerate(all_records, 1):
            condition = record["condition"]
            question = record["question"]
            ground_truth = record["ground_truth"]
            permuted_steps = record["permuted_steps"]

            print(f"[{i}/{total}] ({condition}, problem {record['problem_id']}) "
                  f"Calling model...", end=" ", flush=True)

            try:
                raw_response = get_model_response_stage2(question, permuted_steps)
            except Exception as e:
                print(f"API ERROR: {e}")
                error_record = {
                    "trial": TRIAL,
                    "problem_id": record["problem_id"],
                    "bucket": record["bucket"],
                    "condition": condition,
                    "n_steps": record["n_steps"],
                    "degenerate": record["degenerate"],
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

            # We only need the final answer here, not a step re-parse —
            # Stage 2 responses aren't guaranteed to contain a clean numbered
            # list since the model was told to just use the given steps.
            _, parsed_answer = parse_response(raw_response)

            norm_parsed = normalize_answer(parsed_answer)
            norm_truth = normalize_answer(ground_truth)
            is_correct = (norm_parsed is not None) and (norm_parsed == norm_truth)

            if is_correct:
                correct_count += 1
                condition_correct[condition] += 1
            condition_total[condition] += 1

            result_record = {
                "trial": TRIAL,
                "problem_id": record["problem_id"],
                "bucket": record["bucket"],
                "condition": condition,
                "n_steps": record["n_steps"],
                "degenerate": record["degenerate"],
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

    print("\n--- DONE ---")
    print(f"Overall: {correct_count}/{total} ({correct_count/total:.2%})")
    print("\nPer-condition accuracy:")
    for condition in CONDITION_FILES:
        c = condition_correct[condition]
        t = condition_total[condition]
        print(f"  {condition}: {c}/{t} ({c/t:.2%})")

    print("\nReminder: 'partial' accuracy above BLENDS degenerate (3-4 step,")
    print("Partial==Baseline-equivalent) and non-degenerate (5+ step) cases.")
    print("Re-run analysis splitting on the 'degenerate' field before drawing")
    print("conclusions about H1 (monotonic accuracy drop) for the Partial condition.")


if __name__ == "__main__":
    run()