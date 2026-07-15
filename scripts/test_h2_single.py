"""
test_h2_single.py

Sanity check for the H2 self-report prompt/parser on ONE wrong record before
trusting it for the full batch in 05_h2_self_report.py. Same discipline as
test_single.py was used for Stage 1 and the Stage 2 prompt originally.

Picks the first wrong "reversed" record it finds, runs the self-report
prompt, and prints both the raw response and the parsed result so you can
manually confirm the parser is reading it correctly.
"""

import json

from utils import get_h2_self_report, parse_h2_response


RESULTS_PATH = "results/stage2_results_v2_trial1.jsonl"
PERMUTATION_PATH = "data/stage2_reversed.jsonl"
CONDITION = "reversed"


def find_first_wrong_record():
    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] == CONDITION and not r["correct"]:
                return r
    return None


def find_permutation_data(problem_id):
    with open(PERMUTATION_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["problem_id"] == problem_id:
                return r
    return None


def run():
    wrong_record = find_first_wrong_record()
    if wrong_record is None:
        print(f"No wrong '{CONDITION}' records found.")
        return

    problem_id = wrong_record["problem_id"]
    perm_data = find_permutation_data(problem_id)
    if perm_data is None:
        print(f"No permutation data found for problem_id={problem_id}")
        return

    print(f"Testing on problem_id={problem_id} (condition={CONDITION})")
    print(f"Question: {perm_data['question']}")
    print(f"Ground truth: {wrong_record['ground_truth']}")
    print(f"Model's wrong answer: {wrong_record['parsed_answer']}")
    print(f"\nPermuted steps shown to model:")
    for i, step in enumerate(perm_data["permuted_steps"], 1):
        print(f"  {i}. {step}")
    print(f"\nOriginal (correct-order) steps:")
    for i, step in enumerate(perm_data["original_steps"], 1):
        print(f"  {i}. {step}")

    print("\n--- Calling model for self-report ---")
    raw_h2_response = get_h2_self_report(
        perm_data["question"], perm_data["permuted_steps"], wrong_record["raw_response"]
    )

    print("\nRAW H2 RESPONSE:")
    print(raw_h2_response)

    permuted_div, justification = parse_h2_response(raw_h2_response)
    print(f"\nPARSED permuted_divergence_step: {permuted_div}")
    print(f"PARSED justification: {justification}")


if __name__ == "__main__":
    run()