"""
02b_permute_conditions_mistral.py

Day 37 — eligibility filtering + Reversed-only permutation for the
cross-model block (ministral-8b-2512).

Deliberately generates ONLY the Reversed condition, per the roadmap's Day
37 scope ("Run Stage 2 for Baseline-control vs. Reversed only -- your
strongest raw signal from the Llama run -- deliberately not the full
4-condition matrix yet, to protect the timeline"). Shuffled and Partial
are NOT generated here; Day 38 will decide whether to expand based on
whether Reversed shows a directionally comparable pattern to Llama's.

Same eligibility rule as the Llama pipeline (02_permute_conditions.py):
correct == True AND len(parsed_steps) >= 3 (MIN_STEPS = 3). Reuses
get_reversed() from utils.py UNCHANGED -- same permutation logic, same
seeding-by-problem_id, applied to a different model's Stage 1 output.

No API calls in this script -- pure local computation, reading Stage 1
results and writing a permutation file. Runs in well under a second.

Input:  data/stage1_baseline_mistral.jsonl
Output: data/stage2_reversed_mistral.jsonl
"""

import json

from utils import get_reversed


INPUT_PATH = "data/stage1_baseline_mistral.jsonl"
OUTPUT_PATH = "data/stage2_reversed_mistral.jsonl"
MIN_STEPS = 3


def load_eligible_records(path, min_steps):
    """Correct answer AND >=min_steps parsed steps -- same rule as Llama's pipeline."""
    eligible = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_steps = len(r["parsed_steps"]) if r["parsed_steps"] else 0
            if r["correct"] and n_steps >= min_steps:
                eligible.append(r)
    return eligible


def build_condition_record(source_record, permuted_steps, condition_name):
    return {
        "problem_id": source_record["problem_id"],
        "bucket": source_record["bucket"],
        "question": source_record["question"],
        "ground_truth": source_record["ground_truth"],
        "original_steps": source_record["parsed_steps"],
        "condition": condition_name,
        "permuted_steps": permuted_steps,
        "degenerate": False,  # Reversed is never degenerate -- well-defined for any n>=2
        "n_steps": len(source_record["parsed_steps"]),
    }


def run():
    eligible = load_eligible_records(INPUT_PATH, MIN_STEPS)
    print(f"Eligible problems for ministral-8b-2512 (correct AND >={MIN_STEPS} steps): "
          f"{len(eligible)}")

    with open(OUTPUT_PATH, "w") as out_file:
        for record in eligible:
            steps = record["parsed_steps"]
            reversed_steps = get_reversed(steps)
            reversed_record = build_condition_record(record, reversed_steps, "reversed")
            out_file.write(json.dumps(reversed_record) + "\n")

    print(f"Wrote {len(eligible)} records to {OUTPUT_PATH}")
    print("\nNote: this is Reversed-only, per Day 37 scope. Shuffled and Partial")
    print("are deferred to Day 39, pending Day 38's decision on whether they're")
    print("worth running for this model.")
    print(f"\nUse this eligible count ({len(eligible)}) to estimate Stage 2 runtime")
    print("for both 03b_run_baseline_control_mistral.py and")
    print("03_run_conditions_mistral.py -- see each script's docstring for the formula.")


if __name__ == "__main__":
    run()