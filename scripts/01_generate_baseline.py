"""
01_generate_baseline.py

Stage 1 of the CoT robustness experiment.
Loops over all 100 problems in the stratified GSM8K subset, calls the model
once per problem, parses its self-generated CoT steps, normalizes the answer,
compares to ground truth, and writes results incrementally to
data/stage1_baseline.jsonl.

This file is the baseline / raw material that Stage 2 will later permute.
"""

import json
import time

from utils import get_model_response, parse_response, normalize_answer


INPUT_PATH = "data/day27_gsm8k_subset.json"
OUTPUT_PATH = "data/stage1_baseline.jsonl"
SLEEP_SECONDS = 2.5  # conservative gap between calls, to avoid hitting rate limits (poor me using groq free tier)


def run_baseline():
    with open(INPUT_PATH) as f:
        problems = json.load(f)

    total = len(problems)
    correct_count = 0

    # 'a' mode so a partial run can be resumed without losing earlier lines.

    
    with open(OUTPUT_PATH, "a") as out_file:
        for i, problem in enumerate(problems, 1):
            question = problem["question"]
            ground_truth = problem["final_answer"]

            print(f"[{i}/{total}] Calling model...", end=" ", flush=True)

            try:
                raw_response = get_model_response(question)
            except Exception as e:
                
                # Log the failure as its own record instead of silently skipping it,
                # so a rate-limit or API hiccup is visible in the data, not just the terminal.
                print(f"API ERROR: {e}")
                error_record = {
                    "problem_id": problem["id"],
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

            # Write immediately, one line per problem — never held in memory until the end.
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()

            status = "correct" if is_correct else "WRONG"
            print(f"{status} (parsed={parsed_answer}, truth={ground_truth})")

            time.sleep(SLEEP_SECONDS)

    accuracy = correct_count / total
    print("\n--- DONE ---")
    print(f"Correct: {correct_count}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print("Sanity check against external anchors:")
    print("  Fragile Thoughts (Llama-3.1-8B-Instruct, clean): ~87-96%")
    print("  GSM-Symbolic (Llama3-8b-instruct, 100-q subset): 74%")


if __name__ == "__main__":
    run_baseline()