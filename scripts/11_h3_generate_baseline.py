"""
11_h3_generate_baseline.py

H3 Stage 1: free-generation baseline using qwen/qwen3.6-27b via Groq.
Identical task design to 01_generate_baseline.py — same 100-problem subset,
same prompt, same parser, same scorer.  Only the model changes.

Output: data/h3_stage1_baseline.jsonl
"""

import json
import time

from utils_h3 import get_model_response, parse_response, normalize_answer, MODEL_ID


INPUT_PATH = "data/day27_gsm8k_subset.json"
OUTPUT_PATH = "data/h3_stage1_baseline.jsonl"
SLEEP_SECONDS = 2.5   # conservative — Groq free tier, preview model


def run_baseline():
    with open(INPUT_PATH) as f:
        problems = json.load(f)

    total = len(problems)
    correct_count = 0

    print(f"Model: {MODEL_ID}")
    print(f"Running {total} problems → {OUTPUT_PATH}\n")

    with open(OUTPUT_PATH, "a") as out_file:
        for i, problem in enumerate(problems, 1):
            question = problem["question"]
            ground_truth = problem["final_answer"]

            print(f"[{i}/{total}] Calling model...", end=" ", flush=True)

            try:
                raw_response = get_model_response(question)
            except Exception as e:
                print(f"API ERROR: {e}")
                error_record = {
                    "problem_id": problem.get("problem_id", i),
                    "bucket": problem.get("bucket"),
                    "question": question,
                    "ground_truth": ground_truth,
                    "raw_response": None,
                    "parsed_steps": None,
                    "parsed_answer": None,
                    "correct": False,
                    "error": str(e),
                }
                out_file.write(json.dumps(error_record) + "\n")
                out_file.flush()
                time.sleep(SLEEP_SECONDS)
                continue

            steps, parsed_answer = parse_response(raw_response)

            norm_parsed = normalize_answer(parsed_answer)
            norm_truth = normalize_answer(ground_truth)
            is_correct = (norm_parsed is not None) and (norm_parsed == norm_truth)

            if is_correct:
                correct_count += 1

            record = {
                "problem_id": problem.get("problem_id", i),
                "bucket": problem.get("bucket"),
                "question": question,
                "ground_truth": ground_truth,
                "raw_response": raw_response,
                "parsed_steps": steps,
                "parsed_answer": parsed_answer,
                "correct": is_correct,
                "error": None,
            }

            out_file.write(json.dumps(record) + "\n")
            out_file.flush()

            status = "correct" if is_correct else "WRONG"
            print(f"{status} (parsed={parsed_answer}, truth={ground_truth})")

            time.sleep(SLEEP_SECONDS)

    accuracy = correct_count / total
    print("\n--- DONE ---")
    print(f"Model: {MODEL_ID}")
    print(f"Correct: {correct_count}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nLlama-3.1-8B-Instant reference: 90/100 (90%)")
    print("If this model's accuracy is close, the Stage 2 robustness comparison")
    print("will be interpretable as a model-size effect, not a capability gap artifact.")


if __name__ == "__main__":
    run_baseline()
