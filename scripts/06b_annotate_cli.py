"""
06b_annotate_cli.py

Interactive annotation loop for data/h2_manual_annotation_sheet.jsonl.

Instead of hand-editing JSONL, this shows you one case at a time
(question, steps-as-shown-to-model, model's final wrong response),
prompts you to type your guess, and writes it back into the sheet file
immediately after each answer -- so you can quit partway (Ctrl+C or
type 'quit') and resume later without losing progress.

Skips any record that's already been filled in (so re-running resumes
from where you left off, doesn't make you redo finished cases).

Usage:
  python scripts/06b_annotate_cli.py
"""

import json

SHEET_PATH = "data/h2_manual_annotation_sheet.jsonl"


def load_sheet():
    with open(SHEET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_sheet(records):
    with open(SHEET_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def prompt_guess(n_steps):
    while True:
        raw = input(f"Your guess (1-{n_steps}, or 'none', or 'quit'): ").strip().lower()
        if raw == "quit":
            return "QUIT"
        if raw == "none":
            return "none"
        if raw.isdigit() and 1 <= int(raw) <= n_steps:
            return int(raw)
        print(f"  Not valid -- enter an integer 1-{n_steps}, 'none', or 'quit'.")


def prompt_confidence():
    while True:
        raw = input("Confidence 1-5 (optional, Enter to skip): ").strip()
        if raw == "":
            return None
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  Enter 1-5, or just press Enter to skip.")


def run():
    records = load_sheet()

    remaining = [r for r in records if r.get("YOUR_first_divergence_step_guess") is None]
    done_count = len(records) - len(remaining)

    print(f"Total cases: {len(records)}  |  Already annotated: {done_count}  |  "
          f"Remaining: {len(remaining)}")
    if not remaining:
        print("Everything is already annotated. Run 07_compare_annotations.py next.")
        return
    print("Type 'quit' at any prompt to stop -- your progress is saved after each case.\n")

    for i, record in enumerate(remaining, 1):
        idx = record["annotation_index"]
        steps = record["steps_as_shown_to_model"]
        n_steps = steps.count("\n") + 1  # steps are newline-joined, 1 per line

        print("=" * 70)
        print(f"Case {i}/{len(remaining)}  (annotation_index={idx})")
        print("=" * 70)
        print(f"\nQUESTION:\n{record['question']}\n")
        print(f"STEPS AS SHOWN TO MODEL:\n{steps}\n")
        print(f"MODEL'S FINAL RESPONSE:\n{record['model_final_response']}\n")

        guess = prompt_guess(n_steps)
        if guess == "QUIT":
            print("\nStopping. Progress so far has been saved.")
            break

        confidence = prompt_confidence()
        notes = input("Notes (optional, Enter to skip): ").strip()

        record["YOUR_first_divergence_step_guess"] = guess
        record["YOUR_confidence_1to5"] = confidence
        record["YOUR_notes"] = notes if notes else None

        # write back immediately so nothing is lost if you stop
        save_sheet(records)
        print()

    remaining_after = sum(
        1 for r in records if r.get("YOUR_first_divergence_step_guess") is None
    )
    print(f"\nDone for now. {len(records) - remaining_after}/{len(records)} annotated.")
    if remaining_after == 0:
        print("All cases annotated! Run: python scripts/07_compare_annotations.py")
    else:
        print(f"{remaining_after} left -- re-run this script anytime to continue.")


if __name__ == "__main__":
    run()