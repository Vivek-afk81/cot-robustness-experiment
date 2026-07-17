"""
06d_annotate_full_careful_cli.py

Interactive annotation loop for data/h2_manual_annotation_sheet.jsonl.

Adds a SECOND, careful, rule-based judgment (CAREFUL_ fields) to each of the
35 records, alongside your original fast gut guess (YOUR_ fields) already in
the same file. Stays blind by design -- never prints or reads the YOUR_
fields while you're annotating, so this pass is genuinely independent.

Same resumable pattern as 06b/06c: writes back after every answer, skips
records that already have a CAREFUL_ answer, quit anytime with 'quit'.

Rule reminder, shown before each case:
  (a) MISUSE      - step uses a value that contradicts something already given
  (b) INVENTED     - step uses a number/operation not supported yet
  (c) SELF-BREAK   - model says something looks wrong, then proceeds badly anyway
  (d) SKIP/IGNORE  - step is never actually used, and skipping it changes the answer
Mark the FIRST step where any of these applies. If none apply anywhere and
the answer is still wrong, guess = "none" (a legitimate, honest answer).

Usage:
  python scripts/06d_annotate_full_careful_cli.py
"""

import json

SHEET_PATH = "data/h2_manual_annotation_sheet.jsonl"

RULE_TEXT = """
  (a) MISUSE      - step uses a value that contradicts something already given
  (b) INVENTED     - step uses a number/operation not supported yet
  (c) SELF-BREAK   - model says something looks wrong, then proceeds badly anyway
  (d) SKIP/IGNORE  - step is never actually used, and skipping it changes the answer
"""


def load_sheet():
    with open(SHEET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_sheet(records):
    with open(SHEET_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def prompt_guess(n_steps):
    while True:
        raw = input(f"Your CAREFUL guess (1-{n_steps}, or 'none', or 'quit'): ").strip().lower()
        if raw == "quit":
            return "QUIT"
        if raw == "none":
            return "none"
        if raw.isdigit() and 1 <= int(raw) <= n_steps:
            return int(raw)
        print(f"  Not valid -- enter an integer 1-{n_steps}, 'none', or 'quit'.")


def prompt_rule(guess):
    if guess == "none":
        return None
    while True:
        raw = input("Which rule applied (a/b/c/d): ").strip().lower()
        if raw in ("a", "b", "c", "d"):
            return raw
        print("  Enter one of: a, b, c, d")


def prompt_confidence():
    while True:
        raw = input("Confidence 1-5: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  Enter 1-5 -- even a low number is a valid, honest answer for a murky case.")


def run():
    records = load_sheet()

    # Ensure every record has the CAREFUL_ fields, even if this is the first run.
    for r in records:
        r.setdefault("CAREFUL_first_divergence_step_guess", None)
        r.setdefault("CAREFUL_which_rule_applied", None)
        r.setdefault("CAREFUL_confidence_1to5", None)
        r.setdefault("CAREFUL_notes", None)

    remaining = [r for r in records if r.get("CAREFUL_first_divergence_step_guess") is None]
    done_count = len(records) - len(remaining)

    print(f"Total cases: {len(records)}  |  Already careful-annotated: {done_count}  |  "
          f"Remaining: {len(remaining)}")
    if not remaining:
        print("Everything is already careful-annotated. Run compare_gut_vs_careful.py next.")
        return

    print("This pass stays BLIND to your original YOUR_ guesses -- they're not shown.")
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
        print(RULE_TEXT)

        guess = prompt_guess(n_steps)
        if guess == "QUIT":
            print("\nStopping. Progress so far has been saved.")
            break

        rule = prompt_rule(guess)
        confidence = prompt_confidence()
        notes = input("Notes (what specifically happened at that step): ").strip()

        record["CAREFUL_first_divergence_step_guess"] = guess
        record["CAREFUL_which_rule_applied"] = rule
        record["CAREFUL_confidence_1to5"] = confidence
        record["CAREFUL_notes"] = notes if notes else None

        # write back immediately so nothing is lost if you stop
        save_sheet(records)
        print()

    remaining_after = sum(
        1 for r in records if r.get("CAREFUL_first_divergence_step_guess") is None
    )
    print(f"\nDone for now. {len(records) - remaining_after}/{len(records)} careful-annotated.")
    if remaining_after == 0:
        print("All 35 done! Run: python compare_gut_vs_careful.py")
    else:
        print(f"{remaining_after} left -- re-run this script anytime to continue.")


if __name__ == "__main__":
    run()