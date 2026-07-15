"""
05_h2_self_report.py

H2 analysis, part 1: self-report pass.
Loops over every WRONG record in results/stage2_results_v2_trial1.jsonl
(conditions: reversed, shuffled, partial), re-joins each one against its
corresponding permutation file (data/stage2_{condition}.jsonl) to recover
the exact permuted_steps and original_steps the model saw, and asks the
model to self-report which permuted-position step is where its reasoning
first diverged.

Tracks BOTH:
  - permuted_divergence_step: position in the order the model actually saw
    (this is what H2's hypothesis literally refers to — "Step 2 position")
  - original_divergence_step: the SAME content's position in the original,
    correct-order chain — a secondary check against the confound where a
    positional finding could actually be a content-fragility finding in
    disguise (see project decision log)

Deliberately does NOT reveal ground truth in the self-report prompt (see
utils.py / build_h2_prompt docstring) — this is a genuine introspection
attempt, not answer-key-assisted rationalization. Reliability is UNKNOWN
until validated against a manual spot-check (separate script) — do not
treat self-report agreement as proven just because responses look confident.

Output: results/h2_self_report_trial1.jsonl
"""

import json
import time

from utils import get_h2_self_report, parse_h2_response


RESULTS_PATH = "results/stage2_results_v2_trial1.jsonl"
PERMUTATION_FILES = {
    "reversed": "data/stage2_reversed.jsonl",
    "shuffled": "data/stage2_shuffled.jsonl",
    "partial": "data/stage2_partial.jsonl",
}
OUTPUT_PATH = "results/h2_self_report_trial1.jsonl"
SLEEP_SECONDS = 2.5


def load_permutation_lookup(path):
    """Returns {problem_id: {"original_steps": [...], "permuted_steps": [...]}}."""
    lookup = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            lookup[r["problem_id"]] = {
                "original_steps": r["original_steps"],
                "permuted_steps": r["permuted_steps"],
                "degenerate": r["degenerate"],
            }
    return lookup


def map_to_original_position(permuted_divergence_step, permuted_steps, original_steps):
    """
    Given a 1-indexed permuted-position divergence, find that same step's
    text and return its 1-indexed position in the ORIGINAL step order.
    Returns None if divergence is "none" or unparseable, or if the text
    match is ambiguous/fails (flagged, not silently guessed).
    """
    if permuted_divergence_step in (None, "none"):
        return permuted_divergence_step  # pass through "none"/None as-is

    if not isinstance(permuted_divergence_step, int):
        return None

    if permuted_divergence_step < 1 or permuted_divergence_step > len(permuted_steps):
        return None  # out-of-range self-report, can't map

    step_text = permuted_steps[permuted_divergence_step - 1]

    # Sanity check: original_steps should have no duplicate text, or this
    # mapping is ambiguous. Flag rather than silently pick the first match.
    if original_steps.count(step_text) != 1:
        return "AMBIGUOUS_MAPPING"

    return original_steps.index(step_text) + 1  # 1-indexed


def run():
    permutation_lookups = {
        cond: load_permutation_lookup(path) for cond, path in PERMUTATION_FILES.items()
    }

    wrong_records = []
    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] in PERMUTATION_FILES and not r["correct"]:
                wrong_records.append(r)

    total = len(wrong_records)
    print(f"Wrong records to process: {total}")
    for cond in PERMUTATION_FILES:
        n = sum(1 for r in wrong_records if r["condition"] == cond)
        print(f"  {cond}: {n}")

    with open(OUTPUT_PATH, "w") as out_file:
        for i, record in enumerate(wrong_records, 1):
            condition = record["condition"]
            problem_id = record["problem_id"]
            question = record["question"] if "question" in record else None

            perm_data = permutation_lookups[condition].get(problem_id)
            if perm_data is None:
                print(f"[{i}/{total}] problem {problem_id} ({condition}): "
                      f"NO PERMUTATION DATA FOUND — skipping")
                continue

            permuted_steps = perm_data["permuted_steps"]
            original_steps = perm_data["original_steps"]

            # question wasn't saved in stage2_results — pull it from the
            # permutation file's parent record if needed. Re-check: permutation
            # files DO carry "question" (see 02_permute_conditions.py output
            # schema) — use that as the reliable source.
            with open(PERMUTATION_FILES[condition]) as pf:
                for line in pf:
                    r = json.loads(line)
                    if r["problem_id"] == problem_id:
                        question = r["question"]
                        break

            print(f"[{i}/{total}] problem {problem_id} ({condition})...", end=" ", flush=True)

            try:
                raw_h2_response = get_h2_self_report(
                    question, permuted_steps, record["raw_response"]
                )
            except Exception as e:
                print(f"API ERROR: {e}")
                error_record = {
                    "problem_id": problem_id,
                    "condition": condition,
                    "n_steps": record["n_steps"],
                    "degenerate": perm_data["degenerate"],
                    "permuted_divergence_step": None,
                    "original_divergence_step": None,
                    "justification": None,
                    "raw_h2_response": None,
                    "error": str(e),
                }
                out_file.write(json.dumps(error_record) + "\n")
                out_file.flush()
                time.sleep(SLEEP_SECONDS)
                continue

            permuted_div, justification = parse_h2_response(raw_h2_response)
            original_div = map_to_original_position(permuted_div, permuted_steps, original_steps)

            result_record = {
                "problem_id": problem_id,
                "condition": condition,
                "n_steps": record["n_steps"],
                "degenerate": perm_data["degenerate"],
                "permuted_divergence_step": permuted_div,
                "original_divergence_step": original_div,
                "justification": justification,
                "raw_h2_response": raw_h2_response,
                "error": None,
            }

            out_file.write(json.dumps(result_record) + "\n")
            out_file.flush()

            print(f"permuted_step={permuted_div}, original_step={original_div}")
            time.sleep(SLEEP_SECONDS)

    print("\n--- DONE ---")
    print(f"Output: {OUTPUT_PATH}")
    print("Next: run the manual spot-check annotation script on a 15-20 case")
    print("sample before trusting this self-report data's agreement rate.")


if __name__ == "__main__":
    run()