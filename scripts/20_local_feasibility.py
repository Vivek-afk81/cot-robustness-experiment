"""
20_local_feasibility.py — run a small batch (default 5) through a local
model before committing to a full Stage 1 run.

Usage:
    python 20_local_feasibility.py phi3-mini          # 5 problems
    python 20_local_feasibility.py qwen2.5-3b 10      # 10 problems
"""

import sys
import json
from pathlib import Path

import utils_local_models as u

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "day27_gsm8k_subset.json"

DEFAULT_N = 5
MAX_N = 10


def main(model_key: str, n: int):
    with open(DATA_PATH) as f:
        subset = json.load(f)

    # Spread picks evenly across the dataset for variety
    indices = [int(i * len(subset) / n) for i in range(n)]
    problems = [subset[i] for i in indices]

    print(f"--- Feasibility test: {model_key} | {n} problems ---\n")

    results = []
    parse_fails = 0

    for idx, problem in enumerate(problems, 1):
        question = problem["question"]
        ground_truth = problem["final_answer"]
        pid = problem["id"]

        print(f"[{idx}/{n}] Problem id={pid}  (ground truth: {ground_truth})")

        raw = u.get_model_response(question, model_key=model_key)

        steps, raw_answer = u.parse_response(raw)
        norm_pred = u.normalize_answer(raw_answer)
        norm_truth = u.normalize_answer(ground_truth)
        correct = (norm_pred is not None) and (norm_pred == norm_truth)

        status = "✓" if correct else ("⚠ PARSE FAIL" if raw_answer is None else "✗")
        print(f"       {status}  steps={len(steps)}  answer={norm_pred!r} vs {norm_truth!r}")

        if raw_answer is None:
            parse_fails += 1
            print(f"       RAW OUTPUT (first 300 chars):\n       {raw[:300]}\n")

        results.append({
            "pid": pid, "correct": correct, "steps": len(steps),
            "pred": norm_pred, "truth": norm_truth, "raw_answer": raw_answer,
        })

    # ── Summary ──
    n_correct = sum(r["correct"] for r in results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"SUMMARY: {model_key}")
    print(f"{'='*50}")
    print(f"Accuracy:         {n_correct}/{n} ({n_correct/n:.0%})")
    print(f"Avg parsed steps: {avg_steps:.1f}")
    print(f"Parse failures:   {parse_fails}/{n}")

    if parse_fails > 0:
        print(f"\n*** {parse_fails} problem(s) had no parseable 'Final answer:' line. ***")
        print("Check the chat template / stop tokens before running a full batch.")
    elif n_correct == 0:
        print("\n*** Zero correct answers — model may not be suitable for this task. ***")
    else:
        print(f"\nFeasibility check passed — safe to proceed to 21_local_baseline.py.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("phi3-mini", "qwen2.5-3b"):
        print("Usage: python 20_local_feasibility.py [phi3-mini|qwen2.5-3b] [n=5]")
        sys.exit(1)

    model = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    n = max(1, min(n, MAX_N))
    main(model, n)