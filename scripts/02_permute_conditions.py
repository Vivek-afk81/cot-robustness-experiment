"""
02_permute_conditions.py

Stage 2, part 1 of the CoT robustness experiment.
Reads the corrected Stage 1 baseline, filters to problems that were answered
correctly AND have >=3 model-generated steps (per locked scoping decision:
n=89), and generates three permuted step orderings per problem: Reversed,
Shuffled, and Partial (first/last fixed, middle shuffled).

Each output record carries a `degenerate` flag. Degenerate cases are not
dropped — they're kept and flagged so the analysis stage can decide how to
handle them (e.g. exclude from accuracy calc, or report separately), per the
project's documented limitation: 3-step chains make Partial identical to
Baseline by construction.
"""

import json

from utils import get_reversed, get_shuffled, get_partial


INPUT_PATH = "data/stage1_baseline_reparsed.jsonl"
MIN_STEPS = 3

OUTPUT_PATHS = {
    "reversed": "data/stage2_reversed.jsonl",
    "shuffled": "data/stage2_shuffled.jsonl",
    "partial": "data/stage2_partial.jsonl",
}


def load_eligible_records(path, min_steps):
    """Correct answer AND >=min_steps parsed steps — the locked Stage 2 scope (n=89)."""
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
        "problem_id": source_record["problem_id"],
        "bucket": source_record["bucket"],
        "question": source_record["question"],
        "ground_truth": source_record["ground_truth"],
        "original_steps": source_record["parsed_steps"],
        "condition": condition_name,
        "permuted_steps": permuted_steps,
        "degenerate": degenerate,
        "n_steps": len(source_record["parsed_steps"]),
    }


def run():
    eligible = load_eligible_records(INPUT_PATH, MIN_STEPS)
    print(f"Eligible problems (correct AND >={MIN_STEPS} steps): {len(eligible)}")

    degenerate_counts = {"reversed": 0, "shuffled": 0, "partial": 0}

    out_files = {name: open(path, "w") for name, path in OUTPUT_PATHS.items()}

    try:
        for record in eligible:
            steps = record["parsed_steps"]
            seed = record["problem_id"]  # reproducible per-problem seed, per locked design

            # Reversed — deterministic, no collision risk possible.
            reversed_steps = get_reversed(steps)
            reversed_record = build_condition_record(record, reversed_steps, False, "reversed")
            out_files["reversed"].write(json.dumps(reversed_record) + "\n")

            # Shuffled — collision-checked against original and reversed.
            shuffled_steps, shuffled_degenerate = get_shuffled(steps, seed)
            if shuffled_degenerate:
                degenerate_counts["shuffled"] += 1
            shuffled_record = build_condition_record(
                record, shuffled_steps, shuffled_degenerate, "shuffled"
            )
            out_files["shuffled"].write(json.dumps(shuffled_record) + "\n")

            # Partial — first/last fixed, middle collision-checked.
            partial_steps, partial_degenerate = get_partial(steps, seed)
            if partial_degenerate:
                degenerate_counts["partial"] += 1
            partial_record = build_condition_record(
                record, partial_steps, partial_degenerate, "partial"
            )
            out_files["partial"].write(json.dumps(partial_record) + "\n")
    finally:
        for f in out_files.values():
            f.close()

    print("\n--- DONE ---")
    print(f"Wrote {len(eligible)} records to each of: {list(OUTPUT_PATHS.values())}")
    print("Degenerate counts (flagged, not dropped):")
    for condition, count in degenerate_counts.items():
        pct = count / len(eligible) * 100
        print(f"  {condition}: {count}/{len(eligible)} ({pct:.1f}%)")
    print("\nNote: 'partial' degenerate count should include every exactly-3-step")
    print("chain in the eligible set — this is the documented, expected limitation,")
    print("not a bug. Cross-check this number against your step-count distribution.")


if __name__ == "__main__":
    run()