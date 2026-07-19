"""
10_recheck_mistral_stepwise.py

Run this from your repo root. Requires utils_mistral.py, utils.py, and
utils_mistral_stepwise.py all present in scripts/ (or on your PYTHONPATH).

Tests the new step-wise, steps-only prompt on a FRESH 8-case sample before
scaling to all 89 -- same instinct that caught the original bypass finding.
Fresh seed (99) deliberately chosen to avoid overlap with seed=42 (H2
annotation sheet) and seed=77 (second intra-rater check) used elsewhere in
this project, so this sample isn't accidentally correlated with anything
already looked at closely.

For each sampled problem:
  - Calls the new step-wise no-question prompt with the ORIGINAL (baseline)
    step order.
  - Calls the same prompt with the REVERSED step order.
  - Scores both responses for step-engagement (does the response actually
    reference each given step's content?) and checks whether the two
    responses are near-identical (a bypass red flag, same as the original
    89-problem quantitative pass).
  - Reports correctness for both, plus the two automatic bypass signals.

ADDED vs. the original version: full results (including raw response text)
are written to results/stepwise_recheck_sample.jsonl, so you can go back
and manually read any case's actual model output without re-running the
API calls -- the console summary alone doesn't show response text, and a
manual read is exactly what this screening pass is supposed to lead to.

This is a screening pass. Per the caveat in utils_mistral_stepwise.py,
manually read the actual raw responses for at least a couple of flagged
AND unflagged cases before trusting the aggregate -- don't let this become
a rubber stamp the way "steps parsed successfully" briefly masked Bug B.
"""

import json
import random
import time

from utils import parse_response, normalize_answer
from utils_mistral_stepwise import (
    get_model_response_stage2_mistral_stepwise_no_question,
    score_step_engagement,
    responses_near_identical,
)


REVERSED_MISTRAL_PATH = "data/stage2_reversed_mistral.jsonl"
OUTPUT_PATH = "results/stepwise_recheck_sample.jsonl"
SAMPLE_SIZE = 8
SAMPLE_SEED = 99
SLEEP_SECONDS = 0.5  # Mistral's published limits (625K TPM, 3.13 RPS) give plenty of headroom


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run():
    all_records = load_records(REVERSED_MISTRAL_PATH)
    # Prefer non-degenerate cases if the field exists -- a genuinely reordered
    # step list is a more informative test of order-sensitivity than a
    # degenerate one. Falls back to the full set if 'degenerate' isn't present.
    candidates = [r for r in all_records if not r.get("degenerate", False)]
    if not candidates:
        candidates = all_records

    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(candidates, min(SAMPLE_SIZE, len(candidates)))

    print(f"Sampled {len(sample)} problems (seed={SAMPLE_SEED}): "
          f"{[r['problem_id'] for r in sample]}\n")

    results = []

    with open(OUTPUT_PATH, "w") as out_file:
        for i, record in enumerate(sample, 1):
            pid = record["problem_id"]
            ground_truth = record["ground_truth"]
            original_steps = record["original_steps"]
            reversed_steps = record["permuted_steps"]
            question = record.get("question")

            print(f"[{i}/{len(sample)}] problem_id={pid} -- calling baseline (original order)...",
                  end=" ", flush=True)
            baseline_response = get_model_response_stage2_mistral_stepwise_no_question(original_steps)
            print("done.")
            time.sleep(SLEEP_SECONDS)

            print(f"[{i}/{len(sample)}] problem_id={pid} -- calling reversed order...",
                  end=" ", flush=True)
            reversed_response = get_model_response_stage2_mistral_stepwise_no_question(reversed_steps)
            print("done.")
            time.sleep(SLEEP_SECONDS)

            _, baseline_answer = parse_response(baseline_response)
            _, reversed_answer = parse_response(reversed_response)

            baseline_correct = normalize_answer(baseline_answer) == normalize_answer(ground_truth)
            reversed_correct = normalize_answer(reversed_answer) == normalize_answer(ground_truth)

            baseline_engagement = score_step_engagement(baseline_response, original_steps)
            reversed_engagement = score_step_engagement(reversed_response, reversed_steps)

            near_identical, similarity_ratio = responses_near_identical(
                baseline_response, reversed_response
            )

            result = {
                "problem_id": pid,
                "question": question,
                "ground_truth": ground_truth,
                "original_steps": original_steps,
                "reversed_steps": reversed_steps,
                "baseline_response": baseline_response,
                "reversed_response": reversed_response,
                "baseline_correct": baseline_correct,
                "reversed_correct": reversed_correct,
                "baseline_engagement": baseline_engagement,
                "reversed_engagement": reversed_engagement,
                "near_identical": near_identical,
                "similarity_ratio": similarity_ratio,
            }
            results.append(result)

            out_file.write(json.dumps(result) + "\n")
            out_file.flush()

    print(f"\nFull results (including raw response text) written to: {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"\nproblem_id={r['problem_id']}")
        print(f"  Baseline: {'correct' if r['baseline_correct'] else 'WRONG'}  |  "
              f"engagement={r['baseline_engagement']['engagement_ratio']:.0%}  |  "
              f"trace_lines={r['baseline_engagement']['trace_step_count']}")
        print(f"  Reversed: {'correct' if r['reversed_correct'] else 'WRONG'}  |  "
              f"engagement={r['reversed_engagement']['engagement_ratio']:.0%}  |  "
              f"trace_lines={r['reversed_engagement']['trace_step_count']}")
        flag = " <-- BYPASS SUSPECTED (near-identical responses)" if r["near_identical"] else ""
        print(f"  Response similarity: {r['similarity_ratio']:.2f}{flag}")
        if r["reversed_engagement"]["engagement_ratio"] < 0.5:
            print(f"  <-- LOW ENGAGEMENT on reversed response -- read this one manually")

    n_bypass_flagged = sum(1 for r in results if r["near_identical"])
    n_low_engagement = sum(1 for r in results if r["reversed_engagement"]["engagement_ratio"] < 0.5)
    print(f"\n{n_bypass_flagged}/{len(results)} flagged as likely bypass (near-identical responses)")
    print(f"{n_low_engagement}/{len(results)} flagged as low step-engagement on the reversed response")
    print(f"\nRaw response text for every case is in {OUTPUT_PATH} -- read at least")
    print("2-3 flagged AND 2-3 unflagged cases from that file before deciding whether")
    print("this prompt actually fixes the bypass, or just changes its shape. Only")
    print("then decide whether to scale to the full 89.")


if __name__ == "__main__":
    run()