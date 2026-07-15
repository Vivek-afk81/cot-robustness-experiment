"""
06_prepare_blind_sample.py

H2 manual spot-check, part 1: build the BLIND annotation sample.

Takes all 35 wrong Stage 2 records, re-joins them against the permutation
files to recover the exact problem/steps/model-response the model saw, and
writes out an annotation sheet that deliberately OMITS:
  - the model's self-reported divergence step (permuted_divergence_step)
  - the model's self-report justification
  - the ground-truth answer

...so you can read each case cold and record your own judgment of where the
reasoning first went wrong, without being anchored by the model's guess or
able to work backward from the answer key.

The real answers (model's self-report + ground truth) are written to a
SEPARATE file (data/h2_manual_answer_key.jsonl) that 07_compare_annotations.py
reads later. Don't open that file until after you've annotated everything --
that's the whole point of a blind spot-check.

Sampling:
  - By default includes ALL 35 wrong records (recommended: n=35 gives you
    the full picture, not just a subset -- this is a small enough set that
    subsampling doesn't save much time and costs you statistical power on
    the agreement-rate estimate).
  - If you pass --sample N, it takes a random subset of size N, but always
    force-includes a few "priority" cases flagged as worth checking:
      - every 'reversed' record where the model's self-report was step 1
        (looked like a lazy/default answer rather than real introspection)
      - problem_id 82 (long chain, 12 original steps -- outlier worth
        double-checking the step mapping on)
  - Order is fully shuffled (not grouped by condition), so you can't
    unconsciously pattern-match "oh this is the reversed batch, must be
    early steps."

Output:
  data/h2_manual_annotation_sheet.jsonl   <- what you read and fill in
  data/h2_manual_answer_key.jsonl         <- hidden until you're done
"""

import json
import random
import argparse

RESULTS_PATH = "results/stage2_results_v2_trial1.jsonl"
H2_SELF_REPORT_PATH = "results/h2_self_report_trial1.jsonl"
PERMUTATION_FILES = {
    "reversed": "data/stage2_reversed.jsonl",
    "shuffled": "data/stage2_shuffled.jsonl",
    "partial": "data/stage2_partial.jsonl",
}

SHEET_PATH = "data/h2_manual_annotation_sheet.jsonl"
ANSWER_KEY_PATH = "data/h2_manual_answer_key.jsonl"

RANDOM_SEED = 42  # fixed so the shuffle order is reproducible if you re-run

# Cases worth force-including regardless of sample size (see docstring)
PRIORITY_PROBLEM_IDS = {82}  # long-chain outlier


def load_permutation_lookup(path):
    lookup = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            lookup[r["problem_id"]] = r
    return lookup


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_sample(sample_n=None):
    permutation_lookups = {
        cond: load_permutation_lookup(path) for cond, path in PERMUTATION_FILES.items()
    }

    results = load_jsonl(RESULTS_PATH)
    h2_reports = load_jsonl(H2_SELF_REPORT_PATH)

    # index h2 self-reports by (problem_id, condition) for easy join
    h2_by_key = {(r["problem_id"], r["condition"]): r for r in h2_reports}

    wrong_records = [
        r for r in results
        if r["condition"] in PERMUTATION_FILES and not r["correct"]
    ]

    combined = []
    for r in wrong_records:
        key = (r["problem_id"], r["condition"])
        h2 = h2_by_key.get(key)
        if h2 is None:
            print(f"WARNING: no H2 self-report found for {key}, skipping")
            continue

        perm_data = permutation_lookups[r["condition"]].get(r["problem_id"])
        if perm_data is None:
            print(f"WARNING: no permutation data for {key}, skipping")
            continue

        combined.append({
            "problem_id": r["problem_id"],
            "condition": r["condition"],
            "question": perm_data["question"],
            "permuted_steps": perm_data["permuted_steps"],
            "n_steps": r["n_steps"],
            "degenerate": r["degenerate"],
            "model_wrong_response": r["raw_response"],
            # answer-key fields, stashed separately below -- not in the sheet
            "_self_report_permuted_step": h2["permuted_divergence_step"],
            "_self_report_original_step": h2["original_divergence_step"],
            "_self_report_justification": h2["justification"],
            "_ground_truth": r["ground_truth"],
        })

    print(f"Total eligible wrong-answer cases available: {len(combined)}")

    rng = random.Random(RANDOM_SEED)

    if sample_n is not None and sample_n < len(combined):
        priority = [c for c in combined if c["problem_id"] in PRIORITY_PROBLEM_IDS]
        priority += [
            c for c in combined
            if c["condition"] == "reversed" and c["_self_report_permuted_step"] == 1
            and c not in priority
        ]
        # de-dupe while preserving order
        seen = set()
        priority_deduped = []
        for c in priority:
            k = (c["problem_id"], c["condition"])
            if k not in seen:
                seen.add(k)
                priority_deduped.append(c)

        remaining_pool = [
            c for c in combined
            if (c["problem_id"], c["condition"]) not in seen
        ]
        rng.shuffle(remaining_pool)

        n_fill = max(0, sample_n - len(priority_deduped))
        sample = priority_deduped + remaining_pool[:n_fill]
        print(f"Forced-priority cases included: {len(priority_deduped)}")
    else:
        sample = combined

    rng.shuffle(sample)  # final shuffle so priority cases aren't clustered at the top

    return sample


def write_outputs(sample):
    with open(SHEET_PATH, "w") as sheet_file, open(ANSWER_KEY_PATH, "w") as key_file:
        for i, case in enumerate(sample, 1):
            numbered_steps = "\n".join(
                f"{j}. {step}" for j, step in enumerate(case["permuted_steps"], 1)
            )

            sheet_record = {
                "annotation_index": i,   # use THIS to refer to a case, not problem_id,
                                          # so you're not tempted to cross-reference
                "problem_id_HIDDEN_UNTIL_DONE": None,  # filled back in at compare time
                "question": case["question"],
                "steps_as_shown_to_model": numbered_steps,
                "model_final_response": case["model_wrong_response"],
                "YOUR_first_divergence_step_guess": None,  # <-- fill this in
                "YOUR_confidence_1to5": None,               # <-- optional, fill this in
                "YOUR_notes": None,                         # <-- optional, fill this in
            }
            sheet_file.write(json.dumps(sheet_record) + "\n")

            key_record = {
                "annotation_index": i,
                "problem_id": case["problem_id"],
                "condition": case["condition"],
                "n_steps": case["n_steps"],
                "degenerate": case["degenerate"],
                "ground_truth": case["_ground_truth"],
                "model_self_report_permuted_step": case["_self_report_permuted_step"],
                "model_self_report_original_step": case["_self_report_original_step"],
                "model_self_report_justification": case["_self_report_justification"],
            }
            key_file.write(json.dumps(key_record) + "\n")

    print(f"\nWrote {len(sample)} cases to:")
    print(f"  {SHEET_PATH}  <- open and annotate THIS one")
    print(f"  {ANSWER_KEY_PATH}  <- do NOT open until you've finished annotating")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                         help="Number of cases to include (default: all 35). "
                              "Priority cases are force-included regardless.")
    args = parser.parse_args()

    sample = build_sample(sample_n=args.sample)
    write_outputs(sample)

    print("\n--- NEXT STEP ---")
    print(f"Open {SHEET_PATH} in a text editor (or write a tiny script to loop")
    print("through it). For each record, read the question + steps-as-shown +")
    print("model's final response, and fill in YOUR_first_divergence_step_guess")
    print("(an integer 1..n_steps, or 'none' if you don't think there's a clear")
    print("single divergence point). Save your edits, THEN run")
    print("07_compare_annotations.py to see how you compare to the model.")