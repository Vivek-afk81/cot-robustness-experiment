"""
23_h3_local_run_conditions.py — execute Stage 2 for a local model using the
same prompt (build_stage2_prompt) as your Llama/Mistral/Qwen pipeline: model
is given the question + a presented step list and told not to skip/reorder/
add steps, only to produce "Final answer: <number>".

This is the WITH-question format — same starting point as your first Mistral
attempt. Phi-3-Mini and Qwen2.5-3B are much weaker than Mistral-8B or Qwen
3.6-27B, so full-bypass risk (ignoring the given steps, re-deriving from the
question) is lower but not zero. Run the bypass spotcheck in
24_local_analysis.py before trusting any accuracy number here — same rule
that applied to every other cross-model block in this project.

Reads condition records from data/ (written by 22_h3_local_permute.py), writes
actual model-response results to results/ — matching your Llama pipeline's
data/ vs results/ split.

Usage (can be run from anywhere, paths are resolved relative to this file):
    python 23_h3_local_run_conditions.py phi3-mini

Output:
    results/h3_local_{model_key}_stage2_results.jsonl
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import utils_local_models as u

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SLEEP_SECONDS = 0.1


def main(model_key: str):
    in_path = DATA_DIR / f"h3_local_{model_key}_stage2_conditions.jsonl"
    out_path = RESULTS_DIR / f"h3_local_{model_key}_stage2_results.jsonl"

    if not in_path.exists():
        print(f"Missing {in_path} — run 22_h3_local_permute.py first.")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    records = [json.loads(line) for line in open(in_path)]
    print(f"Running Stage 2 for {model_key}: {len(records)} (problem, condition) calls.")

    counts = defaultdict(lambda: [0, 0])  # condition -> [correct, clean_total]
    n_error = 0

    with open(out_path, "w") as out_f:
        for i, rec in enumerate(records):
            try:
                raw_response = u.get_model_response_stage2(
                    rec["question"], rec["presented_steps"], model_key=model_key
                )
            except Exception as e:
                print(f"[{i+1}/{len(records)}] problem_id={rec['problem_id']} "
                      f"cond={rec['condition']} API ERROR: {e}")
                out_f.write(json.dumps({**rec, "raw_response": None, "parsed_answer": None,
                                         "correct": False, "error": str(e)}) + "\n")
                out_f.flush()
                n_error += 1
                time.sleep(SLEEP_SECONDS)
                continue

            # Stage 2 responses aren't expected to contain a fresh numbered
            # chain (the model was told to use the GIVEN steps), but
            # parse_response's final-answer regex still works standalone —
            # any numbered lines it happens to emit are just discarded here.
            _, parsed_answer = u.parse_response(raw_response)
            norm_parsed = u.normalize_answer(parsed_answer)
            norm_truth = u.normalize_answer(rec["ground_truth"])
            correct = (norm_parsed is not None) and (norm_parsed == norm_truth)

            if parsed_answer is None:
                n_error += 1
            else:
                counts[rec["condition"]][1] += 1
                if correct:
                    counts[rec["condition"]][0] += 1

            out_record = {
                **rec,
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "correct": correct if parsed_answer is not None else None,
                "error": None,
            }
            out_f.write(json.dumps(out_record) + "\n")
            out_f.flush()

            status = "OK" if correct else ("ERR" if parsed_answer is None else "WRONG")
            print(f"[{i+1}/{len(records)}] problem_id={rec['problem_id']} "
                  f"cond={rec['condition']} {status}")
            time.sleep(SLEEP_SECONDS)

    print(f"\nDone. {n_error} error calls (unparseable). Clean-call accuracy by condition:")
    for cond in ("baseline_control", "reversed", "shuffled", "partial"):
        correct, total = counts.get(cond, [0, 0])
        pct = (100 * correct / total) if total else float("nan")
        print(f"  {cond:16s}: {correct}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("phi3-mini", "qwen2.5-3b"):
        print("Usage: python 23_local_run_conditions.py [phi3-mini|qwen2.5-3b]")
        sys.exit(1)
    main(sys.argv[1])