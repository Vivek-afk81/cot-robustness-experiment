"""
03b_run_baseline_control_mistral.py

Day 37 — Stage 2 baseline-control for the cross-model block
(ministral-8b-2512). Same purpose as the Llama pipeline's
03b_run_baseline_control.py: isolate the task-format-switch effect (free
generation -> "use these given steps" prompt) from any effect of
reordering, so Reversed's accuracy can be divided by THIS model's own
baseline-control accuracy, not by Stage 1's raw accuracy or by Llama's
baseline-control number.

Same eligibility filter as 02b_permute_conditions_mistral.py: correct AND
>=3 steps. Feeds each eligible problem's ORIGINAL, UNPERMUTED steps
through the Stage 2 prompt -- zero reordering happens here.

--- ESTIMATED RUNTIME ---
Formula: eligible_count * (SLEEP_SECONDS + typical_api_latency)
With SLEEP_SECONDS = 0.5 and typical latency for this small model of
roughly 1-2s per call (network + generation time for a ~200-400 token
response), expect roughly 1.5-2.5s per problem.

Example: if eligible_count comes out similar to Llama's own 89 (Llama and
Mistral scored close on Stage 1 -- 90% vs 95% -- so a similar eligible
count is a reasonable planning assumption, though the ACTUAL number will
print when you run 02b_permute_conditions_mistral.py):
  89 problems * ~2s/problem  =~ 3 minutes total

This is far faster than the equivalent Llama run (which used a more
conservative 2.5s sleep for Groq's free tier) -- Mistral's published rate
limits (625,000 TPM, 3.13 RPS) give far more headroom, so the sleep here
is much shorter.

Input:  data/stage1_baseline_mistral.jsonl
Output: results/stage2_baseline_control_mistral.jsonl
"""

import json
import time

from utils_mistral import get_model_response_stage2_mistral, MISTRAL_MODEL
from utils import parse_response, normalize_answer


INPUT_PATH = "data/stage1_baseline_mistral.jsonl"
OUTPUT_PATH = "results/stage2_baseline_control_mistral.jsonl"
MIN_STEPS = 3
SLEEP_SECONDS = 0.5  # same conservative-but-generous spacing as Stage 1 Mistral run


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
    print(f"Eligible problems (correct AND >={MIN_STEPS} steps): {total}")
    print(f"Estimated runtime: ~{total * 2 / 60:.1f} minutes "
          f"(at ~2s/problem average; actual may vary with API latency)\n")

    correct_count = 0

    with open(OUTPUT_PATH, "a") as out_file:
        for i, record in enumerate(eligible, 1):
            question = record["question"]
            ground_truth = record["ground_truth"]
            original_steps = record["parsed_steps"]  # UNPERMUTED -- this is the control

            print(f"[{i}/{total}] (baseline-control, problem {record['problem_id']}) "
                  f"Calling {MISTRAL_MODEL}...", end=" ", flush=True)

            try:
                raw_response = get_model_response_stage2_mistral(question, original_steps)
            except Exception as e:
                print(f"API ERROR: {e}")
                error_record = {
                    "model": MISTRAL_MODEL,
                    "problem_id": record.get("problem_id", i),
                    "bucket": record["bucket"],
                    "condition": "baseline_control",
                    "n_steps": len(original_steps),
                    "degenerate": False,
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
                "model": MISTRAL_MODEL,
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
    print("\n--- DONE ---")
    print(f"Model: {MISTRAL_MODEL}")
    print(f"Baseline-control accuracy: {correct_count}/{total} ({accuracy:.2%})")
    print("\nUse this as the denominator for Robustness_tau when evaluating")
    print("this model's Reversed condition -- not Stage 1's raw accuracy.")


if __name__ == "__main__":
    run()