"""
19_recovery_analysis.py

Scans ALL models and trials for recovery-language candidates among CORRECT
perturbed responses. Produces a single combined output file for manual
review, plus per-model/trial summary statistics.

Models/trials covered:
  - Llama-3.1-8B trial 1 & trial 2 (reversed, shuffled, partial)
  - Ministral-8B (reversed only -- only condition run for Mistral)
  - Qwen-27B (reversed, shuffled, partial)
"""

import json
import re

# ── input files ──────────────────────────────────────────────────────────
INPUT_FILES = [
    ("Llama-8B trial 1",  "results/stage2_results_v2_trial1.jsonl"),
    ("Llama-8B trial 2",  "results/stage2_results_v2_trial2.jsonl"),
    ("Ministral-8B",      "results/stage2_results_mistral.jsonl"),
    ("Qwen-27B",          "results/h3_stage2_results.jsonl"),
]

OUTPUT_PATH = "results/recovery_candidates_all.jsonl"

# ── keyword screen ───────────────────────────────────────────────────────
RECOVERY_KEYWORDS = [
    r"\bwait\b", r"\bactually\b", r"\bhold on\b", r"\bre-?check\b",
    r"\bre-?consider(ing)?\b", r"\bon second thought\b",
    r"\bthat (doesn'?t|does not) (match|seem|make sense)\b",
    r"\bcorrection:?\b", r"\blet me (re-?check|re-?verify|reconsider)\b",
    r"\bI (made|need to correct) an? (error|mistake)\b",
    r"\bthis (contradicts|conflicts with)\b",
    r"\bskip (this|it|step)\b",
    r"\bcome back to (this|it|step)\b",
    r"\bignore (this|the first|step)\b",
    r"\bincorrect\b",
    r"\bout of order\b",
    r"\bnot (yet |been )?calculated\b",
    r"\bwe (don'?t|do not) have\b",
]
COMBINED_PATTERN = re.compile("|".join(RECOVERY_KEYWORDS), re.IGNORECASE)

PERTURBED_CONDITIONS = {"reversed", "shuffled", "partial"}


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def scan(text):
    if not text:
        return []
    return COMBINED_PATTERN.findall(text)


def run():
    all_candidates = []
    print("=" * 70)
    print("RECOVERY-LANGUAGE SCAN — all models & trials")
    print("=" * 70)

    for label, path in INPUT_FILES:
        try:
            records = load_records(path)
        except FileNotFoundError:
            print(f"\n[{label}] FILE NOT FOUND — skipping: {path}")
            continue

        correct_perturbed = [
            r for r in records
            if r.get("correct") and r.get("condition") in PERTURBED_CONDITIONS
        ]

        # per-condition breakdown
        cond_counts = {}
        for r in correct_perturbed:
            c = r["condition"]
            cond_counts[c] = cond_counts.get(c, 0) + 1

        flagged = []
        for r in correct_perturbed:
            raw = r.get("raw_response", "") or ""
            matches = scan(raw)
            if matches:
                flagged.append({
                    "model": label,
                    "problem_id": r["problem_id"],
                    "condition": r["condition"],
                    "n_steps": r.get("n_steps"),
                    "ground_truth": r["ground_truth"],
                    "raw_response": raw,
                    "matched_keywords": [m if isinstance(m, str) else m[0] for m in matches],
                    "MANUAL_is_genuine_recovery": None,
                    "MANUAL_notes": None,
                })

        all_candidates.extend(flagged)

        total = len(correct_perturbed)
        pct = len(flagged) / total * 100 if total else 0
        print(f"\n[{label}]  ({path})")
        print(f"  Correct perturbed responses: {total}")
        for c in sorted(cond_counts):
            print(f"    {c}: {cond_counts[c]}")
        print(f"  Flagged candidates: {len(flagged)}/{total} ({pct:.1f}%)")

        # per-condition flagged breakdown
        flag_by_cond = {}
        for f_ in flagged:
            c = f_["condition"]
            flag_by_cond[c] = flag_by_cond.get(c, 0) + 1
        for c in sorted(flag_by_cond):
            print(f"    {c}: {flag_by_cond[c]} flagged")

    # write combined output
    with open(OUTPUT_PATH, "w") as f:
        for c in all_candidates:
            f.write(json.dumps(c) + "\n")

    print("\n" + "=" * 70)
    print(f"TOTAL candidates across all models: {len(all_candidates)}")
    print(f"Written to: {OUTPUT_PATH}")

    # model-level summary table
    models = {}
    for c in all_candidates:
        m = c["model"]
        models[m] = models.get(m, 0) + 1
    print("\nPer-model candidate counts:")
    for m, n in models.items():
        print(f"  {m}: {n}")

    print("\n--- NEXT STEP ---")
    print("Open the output file, read each raw_response, and fill in")
    print("MANUAL_is_genuine_recovery (true / false / unclear).")
    print("Then compute conditional recovery rate per model:")
    print("  recoveries / (recoveries + SELF-BREAK failures)")


if __name__ == "__main__":
    run()