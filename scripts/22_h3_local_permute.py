"""
22_h3_local_permute.py — build Stage 2 condition records (baseline-control,
reversed, shuffled, partial) from a model's own Stage 1 baseline, using the
same get_reversed / get_shuffled / get_partial logic as utils.py (seeded per
problem by its id, matching your original seeding convention).

Eligibility: correct == True in Stage 1 AND >= 3 parsed steps. Step count is
computed here as len(parsed_steps) — the baseline file no longer stores a
separate n_parsed_steps field. Computed per-model, not reused from Llama's
eligible set — a weaker model gets a different subset of the 100 problems
right.

Reads from data/ and writes back to data/, matching your Llama pipeline's
convention (stage1_baseline_v2.jsonl and stage2_{reversed,shuffled,partial}.jsonl
both live in data/; results/ is reserved for actual model-response outputs).

Usage (can be run from anywhere, paths are resolved relative to this file):
    python 22_local_permute.py phi3-mini

Output:
    data/h3_local_{model_key}_stage2_conditions.jsonl
    One record per (problem, condition): problem_id, question, ground_truth,
    original_steps, condition, presented_steps, degenerate.
"""

import sys
import json
from pathlib import Path

import utils_local_models as u

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def main(model_key: str):
    baseline_path = DATA_DIR / f"h3_local_{model_key}_stage1_baseline.jsonl"
    out_path = DATA_DIR / f"h3_local_{model_key}_stage2_conditions.jsonl"

    if not baseline_path.exists():
        print(f"Missing {baseline_path} — run 21_local_baseline.py first.")
        sys.exit(1)

    eligible = []
    with open(baseline_path) as f:
        for line in f:
            rec = json.loads(line)
            n_steps = len(rec["parsed_steps"]) if rec.get("parsed_steps") else 0
            if rec.get("correct") is True and n_steps >= 3:
                eligible.append(rec)

    print(f"{model_key}: {len(eligible)} of {sum(1 for _ in open(baseline_path))} "
          f"Stage-1 problems are eligible for Stage 2 (correct AND >=3 parsed steps).")

    n_written = 0
    n_shuffle_degenerate = 0
    n_partial_degenerate = 0

    with open(out_path, "w") as out_f:
        for rec in eligible:
            steps = rec["parsed_steps"]
            seed = rec["problem_id"]  # seeded by problem id, matching utils.py's convention
            base = {
                "problem_id": rec["problem_id"],
                "question": rec["question"],
                "ground_truth": rec["ground_truth"],
                "original_steps": steps,
            }

            # baseline-control: correct order, run through the Stage-2 "steps
            # only" prompt format, isolating the format-switch effect from
            # the order effect.
            out_f.write(json.dumps({**base, "condition": "baseline_control",
                                     "presented_steps": steps, "degenerate": False}) + "\n")

            reversed_steps = u.get_reversed(steps)
            out_f.write(json.dumps({**base, "condition": "reversed",
                                     "presented_steps": reversed_steps, "degenerate": False}) + "\n")

            shuffled, shuf_degenerate = u.get_shuffled(steps, seed)
            if shuf_degenerate:
                n_shuffle_degenerate += 1
            out_f.write(json.dumps({**base, "condition": "shuffled",
                                     "presented_steps": shuffled, "degenerate": shuf_degenerate}) + "\n")

            partial, part_degenerate = u.get_partial(steps, seed)
            if part_degenerate:
                n_partial_degenerate += 1
            out_f.write(json.dumps({**base, "condition": "partial",
                                     "presented_steps": partial, "degenerate": part_degenerate}) + "\n")

            n_written += 4

    print(f"Wrote {n_written} condition records ({len(eligible)} problems x 4 conditions) to {out_path}")
    print(f"Shuffle degeneracy (collision-exhausted, MAX_SHUFFLE_ATTEMPTS reached): "
          f"{n_shuffle_degenerate}/{len(eligible)}")
    print(f"Partial degeneracy (chain too short to have a scramblable middle, "
          f"or collision-exhausted): {n_partial_degenerate}/{len(eligible)} "
          f"— report Partial accuracy split by this flag, not blended.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("phi3-mini", "qwen2.5-3b"):
        print("Usage: python 22_local_permute.py [phi3-mini|qwen2.5-3b]")
        sys.exit(1)
    main(sys.argv[1])