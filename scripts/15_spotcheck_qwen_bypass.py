"""
15_spotcheck_qwen_bypass.py

Validation check for the qwen/qwen3.6-27b H3 run's clean baseline-control
vs. Reversed result (97.8% both conditions, per the run summary).

Same rationale as 09_spotcheck_mistral_bypass.py: a perfect or
near-perfect Robustness_tau is CONSISTENT with genuine order-robustness,
but it is equally consistent with the model bypassing the given steps
entirely and re-deriving the answer from the question -- exactly what was
found for ministral-8b-2512 (84/89 character-identical responses between
baseline-control and Reversed). An aggregate accuracy number cannot tell
these two explanations apart; only reading the actual response text can.

This script pulls every problem_id present in BOTH
results/h3_stage2_baseline_control.jsonl and results/h3_stage2_results.jsonl
(condition == "reversed"), reports a full character-similarity pass across
ALL matched cases (not just a small sample -- cheap to do exhaustively
since it's just text comparison, no API calls), and prints a random sample
of full response pairs for manual reading.

Verdict categories (same thresholds as the Mistral spot-check):
  - IDENTICAL: baseline and reversed responses are exact character matches
  - NEAR-IDENTICAL: >90% similarity (SequenceMatcher ratio) -- e.g. trivial
    formatting differences only
  - DIFFERENT: below 90% similarity -- read these manually; genuine
    difference is necessary but not sufficient for "genuine engagement"
    (see the Mistral stepwise-prompt finding, where different-looking
    responses turned out to reflect a different artifact, not real
    order-sensitivity)
"""

import json
from difflib import SequenceMatcher
import random

BASELINE_PATH = "results/h3_stage2_baseline_control.jsonl"
RESULTS_PATH = "results/h3_stage2_results.jsonl"
REVERSED_INPUT_PATH = "data/h3_stage2_reversed.jsonl"  # for question/original_steps/permuted_steps context

SAMPLE_SIZE = 8  # how many full response pairs to print for manual reading
RANDOM_SEED = 13
NEAR_IDENTICAL_THRESHOLD = 0.90


def load_by_problem_id(path, condition_filter=None):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if condition_filter and r.get("condition") != condition_filter:
                continue
            out[r["problem_id"]] = r
    return out


def similarity(a, b):
    return SequenceMatcher(None, a or "", b or "").ratio()


def run():
    baseline = load_by_problem_id(BASELINE_PATH)
    reversed_results = load_by_problem_id(RESULTS_PATH, condition_filter="reversed")
    reversed_inputs = load_by_problem_id(REVERSED_INPUT_PATH)

    common_ids = sorted(set(baseline) & set(reversed_results))
    print(f"Model: qwen/qwen3.6-27b")
    print(f"Problems with both baseline-control and reversed results: {len(common_ids)}\n")

    identical = []
    near_identical = []
    different = []

    for pid in common_ids:
        b_text = (baseline[pid].get("raw_response") or "").strip()
        r_text = (reversed_results[pid].get("raw_response") or "").strip()

        if b_text == r_text:
            identical.append(pid)
        else:
            ratio = similarity(b_text, r_text)
            if ratio >= NEAR_IDENTICAL_THRESHOLD:
                near_identical.append((pid, ratio))
            else:
                different.append((pid, ratio))

    total = len(common_ids)
    print("--- FULL PASS: character-similarity across ALL matched problems ---\n")
    if total:
        print(f"IDENTICAL (exact char match):        {len(identical)}/{total} "
              f"({len(identical)/total:.1%})")
        print(f"NEAR-IDENTICAL (>={NEAR_IDENTICAL_THRESHOLD:.0%} similarity): "
              f"{len(near_identical)}/{total} ({len(near_identical)/total:.1%})")
        print(f"DIFFERENT (<{NEAR_IDENTICAL_THRESHOLD:.0%} similarity):       "
              f"{len(different)}/{total} ({len(different)/total:.1%})")
    else:
        print("No matched problems found -- check file paths.")
        return

    bypass_signature_rate = (len(identical) + len(near_identical)) / total
    print(f"\nCombined IDENTICAL + NEAR-IDENTICAL (bypass signature): "
          f"{len(identical) + len(near_identical)}/{total} ({bypass_signature_rate:.1%})")

    if bypass_signature_rate >= 0.7:
        print("\n>>> This matches the Mistral bypass pattern (84/89 = 94.4% there).")
        print(">>> Strong evidence Qwen is also re-deriving from the question rather")
        print(">>> than using the given step order. Read the sample below to confirm")
        print(">>> before writing this up -- don't skip the manual read.")
    elif bypass_signature_rate <= 0.3:
        print("\n>>> LOW bypass-signature rate -- genuinely different from Mistral's")
        print(">>> pattern. This does NOT automatically mean genuine order-engagement")
        print(">>> though -- recall the Mistral stepwise-prompt finding, where")
        print(">>> different-looking responses still turned out to reflect an unrelated")
        print(">>> artifact (positional selection), not real reasoning-order sensitivity.")
        print(">>> Read the sample below carefully before concluding anything.")
    else:
        print("\n>>> Mixed result. Read the sample below before drawing any conclusion.")

    # Print a random sample of full response pairs for manual reading
    print("\n" + "=" * 70)
    print(f"SAMPLE OF {min(SAMPLE_SIZE, total)} FULL RESPONSE PAIRS (for manual reading)")
    print("=" * 70)

    rng = random.Random(RANDOM_SEED)
    sample_ids = rng.sample(common_ids, min(SAMPLE_SIZE, total))

    for pid in sample_ids:
        b_record = baseline[pid]
        r_record = reversed_results[pid]
        inp = reversed_inputs.get(pid, {})

        print(f"\n{'='*70}\nPROBLEM {pid}\n{'='*70}")
        print(f"\nQUESTION:\n{inp.get('question', b_record.get('question', '(not found)'))}")

        print(f"\n--- REVERSED STEPS AS SHOWN TO MODEL ---")
        for i, step in enumerate(inp.get("permuted_steps", []), 1):
            print(f"  {i}. {step}")

        print(f"\n--- BASELINE-CONTROL RESPONSE ---\n{b_record.get('raw_response')}")
        print(f"\n--- REVERSED RESPONSE ---\n{r_record.get('raw_response')}")

        b_text = (b_record.get("raw_response") or "").strip()
        r_text = (r_record.get("raw_response") or "").strip()
        if b_text == r_text:
            print("\n>>> IDENTICAL TEXT.")
        else:
            ratio = similarity(b_text, r_text)
            print(f"\n>>> DIFFER (similarity={ratio:.2f}). Read both above: does the")
            print(">>> reversed response show genuine engagement with the reversed step")
            print(">>> content/order, or is it a fresh solve (or a transcription/selection")
            print(">>> artifact, per the Mistral stepwise-prompt finding)?")

    print("\n" + "=" * 70)
    print("Form a judgment from the sample above before updating the readme/paper.")
    print("Do not report tau=1.000-style results as 'robust' without this check --")
    print("that mistake is exactly what the Mistral investigation exists to prevent.")


if __name__ == "__main__":
    run()