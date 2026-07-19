"""
09_spotcheck_mistral_bypass.py

Day 37/38 — qualitative check on the ministral-8b-2512 τ=1.000 result.

Both baseline-control and Reversed came back at 89/89 (100%) -- identical.
This script pulls a random sample of raw responses from BOTH conditions
side by side so you can read them and judge which explanation fits:

  (a) GENUINE ROBUSTNESS: the model's response actually engages with the
      given step list -- e.g. references specific numbers/phrasing from
      the steps it was handed, follows their structure, and would plausibly
      look different if fed a different (wrong) set of steps.
  (b) BYPASS / RE-DERIVATION: the model's response reads like a fresh,
      from-scratch solve of the original question -- ignoring the given
      step list's content and order entirely. If this is happening, the
      response for a given problem_id should look nearly IDENTICAL between
      baseline-control and Reversed, since the model isn't using either
      version of the steps as input, just the question text.

This is a manual read, not an automated classifier -- deliberately so,
same reasoning as the H2 annotation work: a script can't judge "is this
model actually using its inputs" reliably, a human reading the actual text
can.

WHAT TO LOOK FOR when reading each pair:
  - Do the baseline-control and Reversed responses for the SAME problem_id
    look substantively different in their reasoning steps/wording, or
    nearly identical? Nearly identical = strong evidence for bypass.
  - Does the response ever reference something that could only have come
    from the GIVEN steps (e.g. an intermediate number or phrasing that
    matches the steps but wouldn't be the "natural" way to solve it fresh)?
  - Does the response acknowledge the given steps at all (e.g. "using the
    steps provided...") or does it just launch into solving the question?
"""

import json
import random

BASELINE_PATH = "results/stage2_baseline_control_mistral.jsonl"
REVERSED_PATH = "results/stage2_results_mistral.jsonl"
REVERSED_INPUT_PATH = "data/stage2_reversed_mistral.jsonl"  # has permuted_steps for context

SAMPLE_SIZE = 8
RANDOM_SEED = 7


def load_by_problem_id(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["problem_id"]] = r
    return out


def run():
    baseline = load_by_problem_id(BASELINE_PATH)
    reversed_results = load_by_problem_id(REVERSED_PATH)
    reversed_inputs = load_by_problem_id(REVERSED_INPUT_PATH)  # for permuted_steps + question

    common_ids = sorted(set(baseline) & set(reversed_results))
    print(f"Problems with both baseline-control and reversed results: {len(common_ids)}")

    rng = random.Random(RANDOM_SEED)
    sample_ids = rng.sample(common_ids, min(SAMPLE_SIZE, len(common_ids)))

    for pid in sample_ids:
        b = baseline[pid]
        r = reversed_results[pid]
        inp = reversed_inputs.get(pid, {})

        print("\n" + "=" * 70)
        print(f"PROBLEM {pid}")
        print("=" * 70)
        print(f"\nQUESTION:\n{inp.get('question', '(not found)')}")

        print(f"\n--- REVERSED STEPS AS SHOWN TO MODEL ---")
        permuted = inp.get("permuted_steps", [])
        for i, step in enumerate(permuted, 1):
            print(f"  {i}. {step}")

        print(f"\n--- BASELINE-CONTROL RESPONSE (correct-order steps given) ---")
        print(b["raw_response"])

        print(f"\n--- REVERSED RESPONSE (reversed-order steps given) ---")
        print(r["raw_response"])

        # crude same-ness check: how much of the response text is literally
        # identical, character for character, ignoring whitespace differences
        b_text = (b["raw_response"] or "").strip()
        r_text = (r["raw_response"] or "").strip()
        if b_text == r_text:
            print("\n>>> IDENTICAL TEXT between baseline-control and reversed responses.")
            print(">>> Strong evidence the model ignored the given step order entirely.")
        else:
            print(f"\n>>> Responses differ (baseline: {len(b_text)} chars, "
                  f"reversed: {len(r_text)} chars). Read both above and judge:")
            print(">>> does the difference look like genuine step-order engagement,")
            print(">>> or just superficial rewording of the same from-scratch solve?")

    print("\n" + "=" * 70)
    print("AFTER READING: form a judgment on genuine robustness vs. bypass.")
    print("This determines how τ=1.000 should be reported in the paper --")
    print("as a real cross-family robustness finding, or as a construct-validity")
    print("limitation (the Stage 2 prompt doesn't constrain a stronger model the")
    print("way it constrains Llama). Either conclusion is a legitimate, reportable")
    print("result -- but they are NOT the same claim, so don't report one as if")
    print("it were the other.")


if __name__ == "__main__":
    run()