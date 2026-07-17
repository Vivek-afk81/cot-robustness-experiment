"""
select_h2_recheck_sample.py

Picks a RANDOM (not cherry-picked) sample of 8 cases from the full 35, for a
blind intra-rater reliability check: same annotator (you), same cases, but
this time judged carefully against the explicit divergence rule instead of
a fast gut call.

Deliberately NOT sampling only from the "biggest disagreement" cases — that
would bias the recheck toward cases that might just be genuinely ambiguous,
not necessarily where your original judgment was noisy. A plain random
sample of 8 out of 35 gives an honest read on how much your fast-pass
annotations differ from careful ones, across the full range of cases.

Your ORIGINAL guess is deliberately left out of the printed output below —
re-annotate blind, then compare afterward.

Adjust MANUAL_SHEET_PATH / ANNOTATION_KEY_PATH if they don't match your
actual filenames.
"""

import json
import random

MANUAL_SHEET_PATH = "data/h2_manual_annotation_sheet.jsonl"
ANNOTATION_KEY_PATH = "data/h2_manual_answer_key.jsonl"
OUTPUT_PATH = "results/h2_recheck_sample_blind.jsonl"
SEED = 77  # different from the earlier (contaminated) sample

# These 8 were already revealed to you in the earlier (invalid) recheck --
# explicitly excluded so there's zero chance of overlap, rather than relying
# on a different seed alone (which doesn't guarantee no overlap by itself).
ALREADY_REVEALED_INDICES = {2, 4, 5, 8, 16, 18, 24, 35}


def load_all():
    sheet = {}
    with open(MANUAL_SHEET_PATH) as f:
        for line in f:
            r = json.loads(line)
            sheet[r["annotation_index"]] = r
    return sheet


def run():
    sheet = load_all()
    all_indices = sorted(sheet.keys())
    eligible_indices = [i for i in all_indices if i not in ALREADY_REVEALED_INDICES]

    rng = random.Random(SEED)
    sample_indices = rng.sample(eligible_indices, 8)
    sample_indices.sort()

    print(f"Excluded (already revealed to you): {sorted(ALREADY_REVEALED_INDICES)}")
    print(f"Selected 8 FRESH random indices (seed={SEED}, reproducible): {sample_indices}\n")
    print("=" * 70)

    output_records = []
    for idx in sample_indices:
        r = sheet[idx]
        print(f"\n--- annotation_index {idx} ---")
        print(f"Question: {r['question']}")
        print(f"\nSteps as shown to model:\n{r['steps_as_shown_to_model']}")
        print(f"\nModel's final response:\n{r['model_final_response']}")
        print("\n(Original gut guess intentionally hidden — judge fresh against the rule.)")
        print("-" * 70)

        output_records.append({
            "annotation_index": idx,
            "question": r["question"],
            "steps_as_shown_to_model": r["steps_as_shown_to_model"],
            "model_final_response": r["model_final_response"],
            "CAREFUL_first_divergence_step_guess": None,  # fill in after reading
            "CAREFUL_which_rule_applied": None,  # a/b/c/d/none
            "CAREFUL_confidence_1to5": None,
        })

    with open(OUTPUT_PATH, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n\nBlank recheck sheet written to {OUTPUT_PATH} — fill in the three")
    print("CAREFUL_ fields for each of the 8 cases, applying the explicit rule,")
    print("without looking at your original annotation sheet until you're done.")


if __name__ == "__main__":
    run()