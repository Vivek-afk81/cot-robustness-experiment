"""
03_run_conditions_mistral.py

Day 37 — Stage 2, Reversed condition only, for the cross-model block
(ministral-8b-2512). Per Day 37's scope, only Reversed is run this stage
(strongest raw signal from the Llama run) -- Shuffled and Partial are
deferred to Day 39, pending Day 38's decision.

Reads the Reversed permutation file built by 02b_permute_conditions_mistral.py,
feeds each problem's reversed steps to the model via the same Stage 2
prompt format used for Llama, scores against ground truth using the same
parse_response()/normalize_answer() from utils.py.

--- ESTIMATED RUNTIME ---
Same formula and reasoning as 03b_run_baseline_control_mistral.py:
  eligible_count * ~2s/problem (0.5s sleep + ~1-2s typical API latency)
If eligible_count is similar to Llama's 89: ~89 * 2s =~ 3 minutes.
Run 02b_permute_conditions_mistral.py first to see the ACTUAL eligible
count for this model before estimating your own runtime.

Total Day 37 API-calling time (baseline-control + reversed combined):
roughly DOUBLE the single-script estimate above, since both scripts iterate
over the same eligible set once each -- so budget ~6 minutes total if the
eligible count is Llama-comparable (~89), not ~3 minutes for both together.

Input:  data/stage2_reversed_mistral.jsonl
Output: results/stage2_results_mistral.jsonl
"""

import json
import time

from utils_mistral import get_model_response_stage2_mistral, MISTRAL_MODEL
from utils import parse_response, normalize_answer


INPUT_PATH = "data/stage2_reversed_mistral.jsonl"
OUTPUT_PATH = "results/stage2_results_mistral.jsonl"
SLEEP_SECONDS = 0.5


def load_condition_records(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run():
    records = load_condition_records(INPUT_PATH)
    total = len(records)
    print(f"Loaded {total} Reversed-condition records for {MISTRAL_MODEL}")
    print(f"Estimated runtime: ~{total * 2 / 60:.1f} minutes "
          f"(at ~2s/problem average; actual may vary with API latency)\n")

    correct_count = 0

    with open(OUTPUT_PATH, "a") as out_file:
        for i, record in enumerate(records, 1):
            condition = record["condition"]  # will be "reversed" for every row this stage
            question = record["question"]
            ground_truth = record["ground_truth"]
            permuted_steps = record["permuted_steps"]

            print(f"[{i}/{total}] ({condition}, problem {record['problem_id']}) "
                  f"Calling {MISTRAL_MODEL}...", end=" ", flush=True)

            try:
                raw_response = get_model_response_stage2_mistral(question, permuted_steps)
            except Exception as e:
                print(f"API ERROR: {e}")
                error_record = {
                    "model": MISTRAL_MODEL,
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

            _, parsed_answer = parse_response(raw_response)

            norm_parsed = normalize_answer(parsed_answer)
            norm_truth = normalize_answer(ground_truth)
            is_correct = (norm_parsed is not None) and (norm_parsed == norm_truth)

            if is_correct:
                correct_count += 1

            result_record = {
                "model": MISTRAL_MODEL,
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

    accuracy = correct_count / total
    print("\n--- DONE ---")
    print(f"Model: {MISTRAL_MODEL}")
    print(f"Reversed accuracy: {correct_count}/{total} ({accuracy:.2%})")


if __name__ == "__main__":
    run()