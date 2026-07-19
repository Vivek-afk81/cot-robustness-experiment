"""
10_test_noquestion_prompt_mistral.py

Follow-up to the bypass finding (09_spotcheck_mistral_bypass.py). Tests
whether withholding the original question -- forcing the model to work
ONLY from the given step list -- actually eliminates the bypass, before
committing to a full 89-problem re-run.

Runs on a SMALL RANDOM SAMPLE (default 15 of the 89 eligible problems),
both baseline-control (correct order) and Reversed, using the new
steps-only prompt (build_stage2_prompt_no_question / 
get_model_response_stage2_mistral_no_question in utils_mistral.py).

--- ESTIMATED RUNTIME ---
15 problems x 2 conditions x ~2s/call = ~60 seconds total. Deliberately
small and fast -- this is a go/no-go check, not the real data collection.

Reports, per problem:
  - baseline-control answer, reversed answer
  - whether the two responses are char-for-char identical (the same bypass
    signature found before) or genuinely different
Then an overall verdict:
  - If most/all pairs are STILL identical: withholding the question did
    NOT fix the bypass (the model may be reconstructing the question from
    the steps' content, or defaulting to some other shortcut). Don't
    proceed to a full run; log this as a second, deeper limitation.
  - If most pairs are NOW different: the fix worked. Proceed to a full
    89-problem run under this new prompt as a separate, labeled condition
    (not merged with the original baseline-control/reversed numbers).
"""

import json
import random

from utils import normalize_answer, parse_response
from utils_mistral import get_model_response_stage2_mistral_no_question, MISTRAL_MODEL


BASELINE_INPUT_PATH = "data/stage1_baseline_mistral.jsonl"  # for original_steps + ground_truth
REVERSED_INPUT_PATH = "data/stage2_reversed_mistral.jsonl"  # for permuted_steps

SAMPLE_SIZE = 15
RANDOM_SEED = 11


def load_by_problem_id(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["problem_id"]] = r
    return out


def run():
    baseline_source = load_by_problem_id(BASELINE_INPUT_PATH)
    reversed_source = load_by_problem_id(REVERSED_INPUT_PATH)

    common_ids = sorted(set(baseline_source) & set(reversed_source))
    rng = random.Random(RANDOM_SEED)
    sample_ids = rng.sample(common_ids, min(SAMPLE_SIZE, len(common_ids)))

    print(f"Testing steps-only prompt on {len(sample_ids)} problems "
          f"({MISTRAL_MODEL})\n")

    identical_count = 0
    results = []

    for i, pid in enumerate(sample_ids, 1):
        b_record = baseline_source[pid]
        r_record = reversed_source[pid]

        original_steps = b_record["parsed_steps"]
        permuted_steps = r_record["permuted_steps"]
        ground_truth = b_record["ground_truth"]

        print(f"[{i}/{len(sample_ids)}] problem {pid}...", end=" ", flush=True)

        try:
            baseline_response = get_model_response_stage2_mistral_no_question(original_steps)
            reversed_response = get_model_response_stage2_mistral_no_question(permuted_steps)
        except Exception as e:
            print(f"API ERROR: {e}")
            continue

        b_text = (baseline_response or "").strip()
        r_text = (reversed_response or "").strip()
        is_identical = (b_text == r_text)
        if is_identical:
            identical_count += 1

        _, b_answer = parse_response(baseline_response)
        _, r_answer = parse_response(reversed_response)
        b_correct = normalize_answer(b_answer) == normalize_answer(ground_truth)
        r_correct = normalize_answer(r_answer) == normalize_answer(ground_truth)

        status = "IDENTICAL (bypass signature)" if is_identical else "DIFFERENT"
        print(f"{status}  (baseline={b_answer}/{'ok' if b_correct else 'wrong'}, "
              f"reversed={r_answer}/{'ok' if r_correct else 'wrong'})")

        results.append({
            "problem_id": pid,
            "baseline_response": baseline_response,
            "reversed_response": reversed_response,
            "identical": is_identical,
            "baseline_correct": b_correct,
            "reversed_correct": r_correct,
        })

    print(f"\n--- VERDICT ---")
    if results:
        print(f"Identical (bypass signature) responses: {identical_count}/{len(results)} "
              f"({identical_count/len(results):.0%})")
    else:
        print("No results collected.")

    if not results:
        print("No successful calls -- check API errors above before concluding anything.")
        return

    identical_rate = identical_count / len(results)
    if identical_rate >= 0.7:
        print("\nBypass STILL present at a similar rate to before. Withholding the")
        print("question did not fix it -- the model may be reconstructing enough of")
        print("the original problem from the steps' own content/numbers to solve it")
        print("independently of their given order. Do NOT proceed to a full run under")
        print("this prompt; log this as a deeper limitation instead (see readme).")
    elif identical_rate <= 0.3:
        print("\nBypass rate dropped substantially. The steps-only prompt appears to")
        print("be forcing genuine engagement with the given step order. Consider")
        print("proceeding to a full 89-problem run under this prompt as a SEPARATE,")
        print("labeled condition (e.g. 'Stage 2b: steps-only') -- do not merge these")
        print("results with the original baseline-control/reversed numbers, since")
        print("they test a different (stricter, question-withheld) scenario.")
    else:
        print("\nMixed result -- bypass reduced but not eliminated. Worth reading a")
        print("few of the still-identical cases individually before deciding whether")
        print("a full run is worthwhile, or whether a further-hardened prompt is needed.")


if __name__ == "__main__":
    run()