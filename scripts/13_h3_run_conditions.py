"""
13_h3_run_conditions.py

H3 Stage 2, part 2: runs the three permuted-condition files through
qwen/qwen3.6-27b and scores against ground truth.

Mirrors 03_run_conditions.py exactly — only imports and file paths differ.
Output: results/h3_stage2_results.jsonl
"""

import json
import time

from utils_h3 import get_model_response_stage2, parse_response, normalize_answer, MODEL_ID


CONDITION_FILES = {
    "reversed": "data/h3_stage2_reversed.jsonl",
    "shuffled": "data/h3_stage2_shuffled.jsonl",
    "partial":  "data/h3_stage2_partial.jsonl",
}
OUTPUT_PATH = "results/h3_stage2_results.jsonl"
SLEEP_SECONDS = 2.5


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run():
    all_records = []
    for condition_name, path in CONDITION_FILES.items():
        records = load_records(path)
        all_records.extend(records)
        print(f"Loaded {len(records)} records for condition '{condition_name}'")

    total = len(all_records)
    correct_count = 0
    condition_correct = {"reversed": 0, "shuffled": 0, "partial": 0}
    condition_total  = {"reversed": 0, "shuffled": 0, "partial": 0}

    print(f"\nModel: {MODEL_ID}")
    print(f"Total calls: {total}\n")

    with open(OUTPUT_PATH, "a") as out_file:
        for i, record in enumerate(all_records, 1):
            condition     = record["condition"]
            question      = record["question"]
            ground_truth  = record["ground_truth"]
            permuted_steps = record["permuted_steps"]

            print(f"[{i}/{total}] ({condition}, problem {record['problem_id']}) "
                  f"Calling model...", end=" ", flush=True)

            try:
                raw_response = get_model_response_stage2(question, permuted_steps)
            except Exception as e:
                print(f"API ERROR: {e}")
                out_file.write(json.dumps({
                    "problem_id":   record["problem_id"],
                    "bucket":       record["bucket"],
                    "condition":    condition,
                    "n_steps":      record["n_steps"],
                    "degenerate":   record["degenerate"],
                    "ground_truth": ground_truth,
                    "raw_response": None,
                    "parsed_answer": None,
                    "correct":      False,
                    "error":        str(e),
                }) + "\n")
                out_file.flush()
                time.sleep(SLEEP_SECONDS)
                continue

            _, parsed_answer = parse_response(raw_response)

            norm_parsed = normalize_answer(parsed_answer)
            norm_truth  = normalize_answer(ground_truth)
            is_correct  = (norm_parsed is not None) and (norm_parsed == norm_truth)

            if is_correct:
                correct_count += 1
                condition_correct[condition] += 1
            condition_total[condition] += 1

            out_file.write(json.dumps({
                "problem_id":   record["problem_id"],
                "bucket":       record["bucket"],
                "condition":    condition,
                "n_steps":      record["n_steps"],
                "degenerate":   record["degenerate"],
                "ground_truth": ground_truth,
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "correct":      is_correct,
                "error":        None,
            }) + "\n")
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
        print(f"  {condition}: {c}/{t} ({c/t:.2%})" if t else f"  {condition}: no records")


if __name__ == "__main__":
    run()
