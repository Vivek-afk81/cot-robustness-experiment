"""
20_h3_local_feasibility.py — run ONE problem through a local model before
committing to a full Stage 1 run. Matches your project's standing rule
(every new model gets a single-problem check first).

Usage (can be run from anywhere, paths are resolved relative to this file):
    python 20_h3_local_feasibility.py phi3-mini
    python 20_h3_local_feasibility.py qwen2.5-3b
"""

import sys
import json
from pathlib import Path

import utils_local_models as u

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "day27_gsm8k_subset.json"


def main(model_key: str):
    with open(DATA_PATH) as f:
        subset = json.load(f)

    problem = subset[0]
    question = problem["question"]
    ground_truth = problem["final_answer"]  # real key: "final_answer", not "answer"

    print(f"--- Feasibility test: {model_key} ---")
    print(f"Problem id: {problem['id']}")
    print(f"Ground truth: {ground_truth}\n")

    raw = u.get_model_response(question, model_key=model_key)
    print("=== RAW MODEL OUTPUT ===")
    print(raw)
    print("=== END RAW OUTPUT ===\n")

    steps, raw_answer = u.parse_response(raw)
    norm_pred = u.normalize_answer(raw_answer)
    norm_truth = u.normalize_answer(ground_truth)
    correct = (norm_pred is not None) and (norm_pred == norm_truth)

    print(f"Parsed step count: {len(steps)}")
    print(f"Raw final answer: {raw_answer!r}")
    print(f"Normalized: {norm_pred!r} vs ground truth {norm_truth!r}")
    print(f"Correct: {correct}")

    if raw_answer is None:
        print("\n*** FEASIBILITY FAIL: no 'Final answer: <number>' line found. ***")
        print("Check the chat template / stop tokens before running a full batch.")
        sys.exit(1)
    if len(steps) < 2:
        print("\n*** FEASIBILITY WARNING: fewer than 2 steps parsed on an easy problem. ***")

    print("\nFeasibility check passed — safe to proceed to 21_h3_local_baseline.py.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("phi3-mini", "qwen2.5-3b"):
        print("Usage: python 20_h3_local_feasibility.py [phi3-mini|qwen2.5-3b]")
        sys.exit(1)
    main(sys.argv[1])