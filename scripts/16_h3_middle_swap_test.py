"""
16_h3_middle_swap_test.py

Targeted follow-up to the qwen/qwen3.6-27b cross-size finding. Isolates
ONE specific confound from the Reversed-condition result: Reversed always
places the true final step (which, in this project's Stage-1 CoT format,
almost always states the answer directly) at position 1. High Reversed
accuracy is therefore consistent with EITHER genuine order-robust
reasoning OR a much simpler shortcut -- reading whatever is at position 1
and treating it as the answer.

THIS TEST: a minimal, surgical manipulation, not a new full condition.
For each eligible problem, take the ORIGINAL, correct-order steps and swap
ONLY the final step (index -1, the answer-bearing step) with the step at
the middle index. Every other step stays in its natural, original,
logical order. This is deliberately NOT a full reversal or shuffle -- it
changes exactly one thing: where the answer-bearing step sits in the
sequence, while leaving the rest of the reasoning chain undisturbed and
still in its natural forward order.

Why this isolates the confound cleanly:
  - If the model's Reversed-condition success were mostly a "read
    whatever's at the position where the answer tends to be" shortcut,
    moving the answer-bearing step to the MIDDLE (neither the first nor
    last position, and with everything else still in natural order --
    i.e. no other reason to expect it there) should hurt accuracy
    significantly, since there's no positional convenience to exploit and
    no full reordering to trigger extra-careful dependency reconciliation
    either.
  - If accuracy holds up close to baseline-control, that's real evidence
    the model is doing genuine content-level reasoning (reading and using
    each step's content correctly wherever it sits), not just exploiting
    an answer-position shortcut.

Sample size: deliberately small (15 problems) -- this is a targeted
diagnostic, not a full condition needing 80+ statistical power. Reuses
the eligible-problem pool from h3_stage1_baseline.jsonl (same filter as
02_permute_conditions.py-style scripts: correct AND >=3 steps... but here
we also require >=5 steps, since a "middle" position isn't well-defined
or meaningfully different from "near the end" for very short chains).

--- ESTIMATED RUNTIME ---
15 problems x ~1 call each x ~2.5s/call (same conservative Groq sleep as
the rest of the H3 pipeline) = well under a minute of API time, plus
generation latency. Expect 1-3 minutes total.
"""

import json
import time

from utils_h3 import get_model_response_stage2, parse_response, normalize_answer, MODEL_ID


INPUT_PATH = "data/h3_stage1_baseline.jsonl"
OUTPUT_PATH = "results/h3_middle_swap_test.jsonl"
MIN_STEPS_FOR_MIDDLE = 5  # need a real "middle" distinct from first/last
SAMPLE_SIZE = 15
SLEEP_SECONDS = 2.5


def load_eligible_records(path, min_steps):
    eligible = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_steps = len(r["parsed_steps"]) if r["parsed_steps"] else 0
            if r["correct"] and n_steps >= min_steps:
                eligible.append(r)
    return eligible


def build_middle_swap_steps(steps):
    """
    Swaps the FINAL step (answer-bearing, index -1) with the step at the
    middle index. All other steps remain in their original order and
    position. Returns the new step list and the middle index used (for
    logging/reference).
    """
    steps = list(steps)
    n = len(steps)
    middle_idx = n // 2  # e.g. n=6 -> index 3; n=7 -> index 3

    swapped = list(steps)
    swapped[middle_idx], swapped[-1] = swapped[-1], swapped[middle_idx]
    return swapped, middle_idx


def run():
    eligible = load_eligible_records(INPUT_PATH, MIN_STEPS_FOR_MIDDLE)
    print(f"Eligible problems (correct AND >={MIN_STEPS_FOR_MIDDLE} steps): {len(eligible)}")

    sample = eligible[:SAMPLE_SIZE]  # deterministic slice -- fine for a small diagnostic run;
                                      # switch to random.sample if you want a randomized pick
    total = len(sample)
    print(f"Testing on {total} problems\n")
    print(f"Estimated runtime: ~{total * 2.5 / 60:.1f}-{total * 4 / 60:.1f} minutes\n")

    correct_count = 0

    with open(OUTPUT_PATH, "w") as out_file:
        for i, record in enumerate(sample, 1):
            question = record["question"]
            ground_truth = record["ground_truth"]
            original_steps = record["parsed_steps"]

            middle_swap_steps, middle_idx = build_middle_swap_steps(original_steps)

            print(f"[{i}/{total}] problem {record['problem_id']} "
                  f"(n_steps={len(original_steps)}, middle_idx={middle_idx})...",
                  end=" ", flush=True)

            try:
                raw_response = get_model_response_stage2(question, middle_swap_steps)
            except Exception as e:
                print(f"API ERROR: {e}")
                out_file.write(json.dumps({
                    "problem_id": record["problem_id"],
                    "n_steps": len(original_steps),
                    "middle_idx": middle_idx,
                    "original_steps": original_steps,
                    "middle_swap_steps": middle_swap_steps,
                    "ground_truth": ground_truth,
                    "raw_response": None,
                    "parsed_answer": None,
                    "correct": False,
                    "error": str(e),
                }) + "\n")
                out_file.flush()
                time.sleep(SLEEP_SECONDS)
                continue

            _, parsed_answer = parse_response(raw_response)
            norm_parsed = normalize_answer(parsed_answer)
            norm_truth = normalize_answer(ground_truth)
            is_correct = (norm_parsed is not None) and (norm_parsed == norm_truth)

            if is_correct:
                correct_count += 1

            out_file.write(json.dumps({
                "problem_id": record["problem_id"],
                "n_steps": len(original_steps),
                "middle_idx": middle_idx,
                "original_steps": original_steps,
                "middle_swap_steps": middle_swap_steps,
                "ground_truth": ground_truth,
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "correct": is_correct,
                "error": None,
            }) + "\n")
            out_file.flush()

            status = "correct" if is_correct else "WRONG"
            print(f"{status} (parsed={parsed_answer}, truth={ground_truth})")

            time.sleep(SLEEP_SECONDS)

    accuracy = correct_count / total
    print("\n--- DONE ---")
    print(f"Model: {MODEL_ID}")
    print(f"Middle-swap accuracy: {correct_count}/{total} ({accuracy:.2%})")
    print("\nCompare against:")
    print("  Baseline-control (natural order, answer step at true end): 97.8%")
    print("  Reversed (answer step at position 1, full reversal):       97.8%")
    print(f"  Middle-swap (answer step buried in middle, rest natural):  {accuracy:.1%}")
    print("\nInterpretation:")
    print("  - If middle-swap accuracy is CLOSE to 97.8%: real evidence the model")
    print("    is doing genuine content-level reasoning, not exploiting an")
    print("    answer-position shortcut. Strengthens the 'genuine engagement'")
    print("    reading of the Reversed result.")
    print("  - If middle-swap accuracy DROPS substantially: evidence the model")
    print("    relies on some positional convenience (e.g. expecting the answer")
    print("    near the start or end) that a full reversal or shuffle didn't")
    print("    actually eliminate. Weakens the 'genuine robustness' reading and")
    print("    supports reporting Reversed's result as confounded, not settled.")
    print("\nRead a few raw responses in the output file regardless of the")
    print("aggregate number -- same standing rule as every other check this")
    print("project has done.")


if __name__ == "__main__":
    run()