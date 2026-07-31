"""
21_local_baseline.py — Stage 1 baseline for a local model, over the same
100-problem subset (data/day27_gsm8k_subset.json) used for Llama/Mistral/Qwen.

Mirrors your 01_generate_baseline.py: incremental writes (one line per
problem, flushed immediately), API-error records kept rather than dropped,
resumable via append mode. No sleep needed for local inference, but a tiny
pause is kept so a Ctrl-C lands cleanly.

Schema note: "problem_id" here IS the real GSM8K subset id (problem["id"]),
not a row-enumeration. This differs from your existing Llama/Mistral/Qwen
files, where "problem_id" is actually just row position 1-100 due to a
pre-existing bug (problem.get("problem_id", i) always misses and falls back
to i). Deliberately not replicating that bug here.

Usage (can be run from anywhere, paths are resolved relative to this file):
    python 21_local_baseline.py phi3-mini
    python 21_local_baseline.py qwen2.5-3b

Output:
    data/local_{model_key}_stage1_baseline.jsonl
"""

import sys
import json
import time
from pathlib import Path

import utils_local_models as u

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "day27_gsm8k_subset.json"
DATA_DIR = ROOT / "data"
SLEEP_SECONDS = 0.1


def main(model_key: str):
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / f"h3_local_{model_key}_stage1_baseline.jsonl"

    with open(INPUT_PATH) as f:
        problems = json.load(f)

    total = len(problems)
    correct_count = 0

    # 'a' mode, matching 01_generate_baseline.py, so a partial run is resumable.
    with open(output_path, "a") as out_file:
        for i, problem in enumerate(problems, 1):
            question = problem["question"]
            ground_truth = problem["final_answer"]

            print(f"[{i}/{total}] Calling {model_key}...", end=" ", flush=True)

            try:
                raw_response = u.get_model_response(question, model_key=model_key)
            except Exception as e:
                print(f"ERROR: {e}")
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

            steps, parsed_answer = u.parse_response(raw_response)

            norm_parsed = u.normalize_answer(parsed_answer)
            norm_truth = u.normalize_answer(ground_truth)
            is_correct = (norm_parsed is not None) and (norm_parsed == norm_truth)

            if is_correct:
                correct_count += 1

            record = {
                # "problem_id": problem["id"],
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
            print(f"{status} (parsed={parsed_answer}, truth={ground_truth}, steps={len(steps)})")

            time.sleep(SLEEP_SECONDS)

    accuracy = correct_count / total
    print("\n--- DONE ---")
    print(f"Model: {model_key}")
    print(f"Correct: {correct_count}/{total}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("phi3-mini", "qwen2.5-3b"):
        print("Usage: python 21_local_baseline.py [phi3-mini|qwen2.5-3b]")
        sys.exit(1)
    main(sys.argv[1])