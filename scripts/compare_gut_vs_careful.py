"""
compare_gut_vs_careful.py

The real intra-rater check: for every case in data/h2_manual_annotation_sheet.jsonl
that has BOTH a YOUR_ (fast gut) and CAREFUL_ (rule-based) guess, compares
them directly -- same person, same cases, same file, no join needed since
06d_annotate_full_careful_cli.py added CAREFUL_ fields onto the same records.
"""

import json

SHEET_PATH = "data/h2_manual_annotation_sheet.jsonl"


def run():
    exact = 0
    off_by_one = 0
    bigger_gap = 0
    total = 0

    print(f"{'idx':<6}{'gut':<8}{'careful':<10}{'rule':<6}{'result'}")
    print("-" * 60)

    with open(SHEET_PATH) as f:
        for line in f:
            r = json.loads(line)
            idx = r["annotation_index"]
            gut_guess = r.get("YOUR_first_divergence_step_guess")
            careful = r.get("CAREFUL_first_divergence_step_guess")
            rule = r.get("CAREFUL_which_rule_applied")

            if gut_guess is None or careful is None:
                continue  # not yet annotated on one side or the other

            total += 1

            if gut_guess == careful:
                result = "EXACT"
                exact += 1
            elif (isinstance(gut_guess, int) and isinstance(careful, int)
                  and abs(gut_guess - careful) == 1):
                result = "off-by-one"
                off_by_one += 1
            else:
                result = "DISAGREE"
                bigger_gap += 1

            print(f"{idx:<6}{str(gut_guess):<8}{str(careful):<10}{str(rule):<6}{result}")

    print("-" * 60)
    print(f"\nTotal compared: {total}")
    if total == 0:
        print("No cases have both YOUR_ and CAREFUL_ filled in yet.")
        return
    print(f"Exact agreement: {exact}/{total} ({exact/total:.1%})")
    print(f"Off-by-one: {off_by_one}/{total} ({off_by_one/total:.1%})")
    print(f"Combined exact+off-by-one: {(exact+off_by_one)}/{total} ({(exact+off_by_one)/total:.1%})")
    print(f"Bigger disagreement: {bigger_gap}/{total} ({bigger_gap/total:.1%})")


if __name__ == "__main__":
    run()