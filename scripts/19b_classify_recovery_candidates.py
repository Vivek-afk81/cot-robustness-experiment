"""
19b_classify_recovery_candidates.py

Interactive classification loop for results/recovery_candidates.jsonl.

NOT blind (unlike the H2 annotation sheet) -- there's no self-report or
ground truth to hide here. The task is simply: read each flagged
response and judge whether the matched keyword(s) reflect a GENUINE
detected-and-recovered inconsistency, or an incidental/false-positive
match (e.g. "actually" used as an ordinary transition word, with no real
error being flagged).

Shows one candidate at a time: the question (if available), the matched
keyword(s) that triggered the flag, and the full raw response. Prompts
for a classification and optional notes, then saves immediately -- same
resumable design as 06b_annotate_cli.py, so you can stop and continue
without losing progress or redoing finished cases.

Usage:
  python scripts/19b_classify_recovery_candidates.py
"""

import json

SHEET_PATH = "results/recovery_candidates.jsonl"


def load_sheet():
    with open(SHEET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_sheet(records):
    with open(SHEET_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def prompt_classification():
    while True:
        raw = input("Genuine recovery? (y/n/u=unclear/quit): ").strip().lower()
        if raw in ("quit", "q"):
            return "QUIT"
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        if raw in ("u", "unclear"):
            return "unclear"
        print("  Enter y, n, u, or quit.")


def run():
    records = load_sheet()

    remaining = [r for r in records if r.get("MANUAL_is_genuine_recovery") is None]
    done_count = len(records) - len(remaining)

    print(f"Total candidates: {len(records)}  |  Already classified: {done_count}  |  "
          f"Remaining: {len(remaining)}")
    if not remaining:
        print("Everything is already classified.")
        _print_summary(records)
        return
    print("Type 'quit' at any prompt to stop -- progress is saved after each case.\n")

    for i, record in enumerate(remaining, 1):
        pid = record["problem_id"]
        cond = record["condition"]
        keywords = record.get("matched_keywords", [])

        print("=" * 70)
        print(f"Case {i}/{len(remaining)}  (problem_id={pid}, condition={cond})")
        print("=" * 70)
        print(f"\nMatched keyword(s): {keywords}\n")
        print(f"RAW RESPONSE:\n{record['raw_response']}\n")

        classification = prompt_classification()
        if classification == "QUIT":
            print("\nStopping. Progress so far has been saved.")
            break

        notes = input("Notes (optional, Enter to skip): ").strip()

        record["MANUAL_is_genuine_recovery"] = classification
        record["MANUAL_notes"] = notes if notes else None

        save_sheet(records)
        print()

    remaining_after = sum(
        1 for r in records if r.get("MANUAL_is_genuine_recovery") is None
    )
    print(f"\nDone for now. {len(records) - remaining_after}/{len(records)} classified.")
    if remaining_after == 0:
        _print_summary(records)
    else:
        print(f"{remaining_after} left -- re-run this script anytime to continue.")


def _print_summary(records):
    genuine = sum(1 for r in records if r.get("MANUAL_is_genuine_recovery") is True)
    false_pos = sum(1 for r in records if r.get("MANUAL_is_genuine_recovery") is False)
    unclear = sum(1 for r in records if r.get("MANUAL_is_genuine_recovery") == "unclear")
    total = len(records)

    print("\n--- FINAL CLASSIFICATION SUMMARY ---")
    print(f"Genuine recoveries: {genuine}/{total} ({genuine/total:.1%})")
    print(f"False positives:    {false_pos}/{total} ({false_pos/total:.1%})")
    print(f"Unclear:            {unclear}/{total} ({unclear/total:.1%})")

    # SELF-BREAK count is fixed from H2's canonical annotation (15/30).
    self_break_failures = 15
    recovery_rate = genuine / (genuine + self_break_failures)
    print(f"\nUpdated conditional recovery rate (confirmed genuine only):")
    print(f"  {genuine} genuine recoveries / ({genuine} + {self_break_failures} SELF-BREAK failures) "
          f"= {recovery_rate:.1%}")
    print("\nThis replaces the earlier 13-case-extrapolated 80.5% figure with a")
    print("fully validated count. Use THIS number in the paper, not the earlier")
    print("extrapolated one.")
    if unclear > 0:
        print(f"\nNote: {unclear} case(s) marked 'unclear' were excluded from the")
        print("rate above. Consider a second read of these before finalizing.")


if __name__ == "__main__":
    run()