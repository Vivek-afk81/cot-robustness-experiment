"""
13b_h3_run_baseline_control.py

H3 Stage 2 control: feeds each eligible problem's ORIGINAL (unpermuted)
Stage 1 steps through the Stage 2 prompt using qwen/qwen3.6-27b.
Isolates the task-format-switch cost (Stage 1 free-generation → Stage 2
scaffold) from any order-disruption effect.

Mirrors 03b_run_baseline_control.py exactly — only imports and paths differ.
Output: results/h3_stage2_baseline_control.jsonl
"""

import json
import time

from utils_h3 import get_model_response_stage2, parse_response, normalize_answer, MODEL_ID


INPUT_PATH  = "data/h3_stage1_baseline.jsonl"
OUTPUT_PATH = "results/h3_stage2_baseline_control.jsonl"
MIN_STEPS   = 3
SLEEP_SECONDS = 2.5


def load_eligible_records(path, min_steps):
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
    print(f"Model: {MODEL_ID}")
    print(f"Eligible problems (correct AND >={MIN_STEPS} steps): {total}\n")

    correct_count = 0

    with open(OUTPUT_PATH, "a") as out_file:
        for i, record in enumerate(eligible, 1):
            question       = record["question"]
            ground_truth   = record["ground_truth"]
            original_steps = record["parsed_steps"]

            print(f"[{i}/{total}] (baseline-control, problem {record['problem_id']}) "
                  f"Calling model...", end=" ", flush=True)

            try:
                raw_response = get_model_response_stage2(question, original_steps)
            except Exception as e:
                print(f"API ERROR: {e}")
                out_file.write(json.dumps({
                    "problem_id":   record.get("problem_id", i),
                    "bucket":       record["bucket"],
                    "condition":    "baseline_control",
                    "n_steps":      len(original_steps),
                    "degenerate":   False,
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

            out_file.write(json.dumps({
                "problem_id":   record.get("problem_id", i),
                "bucket":       record["bucket"],
                "condition":    "baseline_control",
                "n_steps":      len(original_steps),
                "degenerate":   False,
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

    accuracy = correct_count / total
    stage1_total   = 0
    stage1_correct = 0
    with open(INPUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            stage1_total += 1
            if r["correct"]:
                stage1_correct += 1
    stage1_accuracy = stage1_correct / stage1_total if stage1_total else 0

    print("\n--- DONE ---")
    print(f"Model: {MODEL_ID}")
    print(f"Baseline-control accuracy: {correct_count}/{total} ({accuracy:.2%})")
    print(f"Stage 1 (free generation): {stage1_correct}/{stage1_total} ({stage1_accuracy:.2%})")
    print("\nUse baseline-control (not Stage 1) as the Robustness_tau denominator.")


if __name__ == "__main__":
    run()
