"""
12_h3_permute_conditions.py

H3 Stage 2, part 1: generate the three permuted-step conditions for the Qwen
27B run.  Reads data/h3_stage1_baseline.jsonl, applies the same eligibility
filter (correct AND >=3 parsed steps), and writes reversed / shuffled / partial
condition files to data/.

Identical logic to 02_permute_conditions.py — only file paths change.
"""

import json

from utils_h3 import get_reversed, get_shuffled, get_partial


INPUT_PATH = "data/h3_stage1_baseline.jsonl"
MIN_STEPS = 3

OUTPUT_PATHS = {
    "reversed": "data/h3_stage2_reversed.jsonl",
    "shuffled": "data/h3_stage2_shuffled.jsonl",
    "partial":  "data/h3_stage2_partial.jsonl",
}


def load_eligible_records(path, min_steps):
    eligible = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_steps = len(r["parsed_steps"]) if r["parsed_steps"] else 0
            if r["correct"] and n_steps >= min_steps:
                eligible.append(r)
    return eligible


def build_condition_record(source_record, permuted_steps, degenerate, condition_name):
    return {
        "problem_id":     source_record["problem_id"],
        "bucket":         source_record["bucket"],
        "question":       source_record["question"],
        "ground_truth":   source_record["ground_truth"],
        "original_steps": source_record["parsed_steps"],
        "condition":      condition_name,
        "permuted_steps": permuted_steps,
        "degenerate":     degenerate,
        "n_steps":        len(source_record["parsed_steps"]),
    }


def run():
    eligible = load_eligible_records(INPUT_PATH, MIN_STEPS)
    print(f"Eligible problems (correct AND >={MIN_STEPS} steps): {len(eligible)}")

    degenerate_counts = {"reversed": 0, "shuffled": 0, "partial": 0}
    out_files = {name: open(path, "w") for name, path in OUTPUT_PATHS.items()}

    try:
        for record in eligible:
            steps = record["parsed_steps"]
            seed = record["problem_id"]

            reversed_steps = get_reversed(steps)
            out_files["reversed"].write(
                json.dumps(build_condition_record(record, reversed_steps, False, "reversed")) + "\n"
            )

            shuffled_steps, shuffled_deg = get_shuffled(steps, seed)
            if shuffled_deg:
                degenerate_counts["shuffled"] += 1
            out_files["shuffled"].write(
                json.dumps(build_condition_record(record, shuffled_steps, shuffled_deg, "shuffled")) + "\n"
            )

            partial_steps, partial_deg = get_partial(steps, seed)
            if partial_deg:
                degenerate_counts["partial"] += 1
            out_files["partial"].write(
                json.dumps(build_condition_record(record, partial_steps, partial_deg, "partial")) + "\n"
            )
    finally:
        for f in out_files.values():
            f.close()

    print("\n--- DONE ---")
    print(f"Wrote {len(eligible)} records to each condition file.")
    print("Degenerate counts (flagged, not dropped):")
    for condition, count in degenerate_counts.items():
        pct = count / len(eligible) * 100 if eligible else 0
        print(f"  {condition}: {count}/{len(eligible)} ({pct:.1f}%)")


if __name__ == "__main__":
    run()
