"""
07_compare_annotations.py

H2 manual spot-check, part 2: compare YOUR blind judgments (from
data/h2_manual_annotation_sheet.jsonl, after you've filled it in) against
the model's self-reported divergence step (from the hidden answer key,
data/h2_manual_answer_key.jsonl).

This is the step that actually answers the open question from the readme:
"we still don't know how trustworthy the self-reports themselves are."

Reports:
  1. Exact agreement rate (your guess == model's self-report, on the
     PERMUTED-position numbering, since that's what both of you saw).
  2. "Close" agreement (off by exactly 1 step) -- useful because reasoning
     errors often cascade, so being one step off isn't obviously "wrong,"
     it just means you and the model located the same failure region.
  3. Per-condition breakdown (reversed / shuffled / partial), since the
     readme flagged reversed's step-1 self-reports as looking like a lazy
     default rather than real introspection -- if human agreement is much
     LOWER specifically on those reversed/step-1 cases, that's evidence
     the model really was confabulating there.
  4. A flagged list of the biggest disagreements (|your guess - model's
     guess| >= 2, or one of you said "none" and the other gave a number)
     for you to re-read individually -- these are the most informative
     cases, more so than the overall percentage.

Run this only after YOUR_first_divergence_step_guess is filled in for every
row of the sheet -- rows still marked null are skipped and reported as
incomplete.
"""

import json

SHEET_PATH = "data/h2_manual_annotation_sheet.jsonl"
ANSWER_KEY_PATH = "data/h2_manual_answer_key.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_guess(value):
    """Accepts int, numeric string, or 'none'/None. Returns int or 'none'."""
    if value is None:
        return None  # unfilled
    if isinstance(value, str) and value.strip().lower() == "none":
        return "none"
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def run():
    sheet = load_jsonl(SHEET_PATH)
    key = load_jsonl(ANSWER_KEY_PATH)
    key_by_index = {k["annotation_index"]: k for k in key}

    incomplete = []
    joined = []

    for row in sheet:
        idx = row["annotation_index"]
        your_guess = normalize_guess(row.get("YOUR_first_divergence_step_guess"))

        if your_guess is None:
            incomplete.append(idx)
            continue

        answer = key_by_index.get(idx)
        if answer is None:
            print(f"WARNING: no answer-key entry for annotation_index {idx}")
            continue

        joined.append({
            "annotation_index": idx,
            "problem_id": answer["problem_id"],
            "condition": answer["condition"],
            "your_guess": your_guess,
            "model_guess": normalize_guess(answer["model_self_report_permuted_step"]),
            "model_justification": answer["model_self_report_justification"],
            "n_steps": answer["n_steps"],
        })

    if incomplete:
        print(f"NOTE: {len(incomplete)} rows still unfilled, skipped: {incomplete}")
        print("Fill in YOUR_first_divergence_step_guess for these and re-run for")
        print("a complete picture. Continuing with the rows that ARE filled in.\n")

    total = len(joined)
    if total == 0:
        print("No completed rows to compare yet. Fill in the sheet first.")
        return

    exact_agree = 0
    close_agree = 0  # off by exactly 1, both numeric
    both_none = 0
    one_none_other_number = 0
    disagreements = []

    for j in joined:
        yg, mg = j["your_guess"], j["model_guess"]

        if yg == "none" and mg == "none":
            both_none += 1
            exact_agree += 1
            continue
        if yg == "none" or mg == "none":
            one_none_other_number += 1
            disagreements.append(j)
            continue

        # both numeric from here
        if yg == mg:
            exact_agree += 1
        else:
            diff = abs(yg - mg)
            if diff == 1:
                close_agree += 1
            if diff >= 2 or (yg == "none") != (mg == "none"):
                disagreements.append(j)

    print(f"--- H2 Manual Spot-Check Results (n={total} completed) ---\n")
    print(f"Exact agreement (incl. both='none'): {exact_agree}/{total} ({exact_agree/total:.1%})")
    print(f"Off-by-one (both numeric):           {close_agree}/{total} ({close_agree/total:.1%})")
    print(f"One said 'none', other gave a step:  {one_none_other_number}/{total} ({one_none_other_number/total:.1%})")
    print(f"Combined exact+off-by-one:            {(exact_agree+close_agree)/total:.1%}")

    print("\n--- Per-condition agreement ---")
    for cond in ["reversed", "shuffled", "partial"]:
        cond_rows = [j for j in joined if j["condition"] == cond]
        if not cond_rows:
            continue
        cond_exact = sum(
            1 for j in cond_rows
            if j["your_guess"] == j["model_guess"]
        )
        print(f"  {cond}: {cond_exact}/{len(cond_rows)} exact ({cond_exact/len(cond_rows):.1%})")

    # Specifically check the flagged "reversed + model said step 1" cases
    reversed_step1 = [
        j for j in joined
        if j["condition"] == "reversed" and j["model_guess"] == 1
    ]
    if reversed_step1:
        rs1_agree = sum(1 for j in reversed_step1 if j["your_guess"] == 1)
        print(f"\n--- Flagged check: reversed cases where model self-reported step 1 ---")
        print(f"  n={len(reversed_step1)}, your agreement: {rs1_agree}/{len(reversed_step1)} "
              f"({rs1_agree/len(reversed_step1):.1%})")
        print("  If this agreement rate is much lower than the overall rate above,")
        print("  that's evidence these specific self-reports were a default/lazy")
        print("  answer rather than genuine introspection, as flagged earlier.")
        for j in reversed_step1:
            match = "AGREE" if j["your_guess"] == 1 else f"DISAGREE (you said {j['your_guess']})"
            print(f"    problem {j['problem_id']}: {match}")

    print(f"\n--- Biggest disagreements to re-read individually (n={len(disagreements)}) ---")
    for j in sorted(disagreements, key=lambda x: x["annotation_index"]):
        print(f"  [idx {j['annotation_index']}] problem {j['problem_id']} ({j['condition']}, "
              f"n_steps={j['n_steps']}): you={j['your_guess']}, model={j['model_guess']}")
        print(f"    model's justification: {j['model_justification']}")

    print("\n--- Interpretation guide ---")
    print("There's no universal 'good enough' agreement threshold, but as a rough")
    print("anchor: exact+off-by-one agreement above ~70% is generally considered")
    print("reasonable evidence the self-report tracks something real, not noise.")
    print("Below ~50% (near what you'd expect from partial chance overlap on a")
    print("small step range) means the self-report shouldn't be trusted as a")
    print("stand-in for ground truth, and H2's story should lean on the")
    print("aggregate positional pattern only, with self-report reliability")
    print("reported explicitly as unresolved/weak.")


if __name__ == "__main__":
    run()