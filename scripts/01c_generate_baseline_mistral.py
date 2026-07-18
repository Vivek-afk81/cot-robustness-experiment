"""
01c_generate_baseline_mistral.py

Day 36 (final) — Stage 1 baseline for the second model in the cross-model
block: ministral-8b-2512 via Mistral AI's own API, run against the SAME
100-problem stratified GSM8K subset used for Llama.

History, for the paper's methods section: three earlier candidates were
tried and abandoned before this one --
  1. Gemini 3.1 Flash-Lite: scored 98% on this same Stage 1 baseline,
     STRONGER than Llama-3.1-8B-Instant's 90% -- defeats the premise of
     comparing a smaller/weaker model.
  2. google/gemma-2-2b-it (HF Inference Providers): failed feasibility --
     "not deployed by any Inference Provider" per its own HF model page.
  3. microsoft/Phi-3.5-mini-instruct (HF Inference Providers): also not
     deployed by any provider on the account used.
  (Qwen/Qwen2.5-3B-Instruct via featherless-ai DID pass feasibility, but
  HF's free-tier rate limits proved too restrictive for a full 189-call
  run in practice.)
ministral-8b-2512 was chosen for its generous published rate limits
(625,000 TPM, 3.13 RPS) via Mistral's own API. NOTE: this model is
comparable in scale to Llama-3.1-8B-Instant, not smaller -- so this run
tests cross-family generalization of the step-order effect at a similar
parameter scale, not H3's original smaller-model-shows-bigger-drop claim.
Report it as that in the paper, not as an H3 confirmation/refutation.

Mirrors 01_generate_baseline.py's structure (same fields, same resumable-
append behavior) and reuses parse_response()/normalize_answer() from
utils.py UNCHANGED -- required for any cross-model comparison to be clean:
accuracy differences should reflect the models' actual reasoning, not a
parsing inconsistency between extraction methods.
"""

import json
import time

from utils import parse_response, normalize_answer
from utils_mistral import get_model_response_mistral, MISTRAL_MODEL


INPUT_PATH = "data/day27_gsm8k_subset.json"
OUTPUT_PATH = "data/stage1_baseline_mistral.jsonl"
SLEEP_SECONDS = 0.5  # Mistral's published limit is 3.13 RPS (~0.32s floor);
                      # padding a bit above that for safety margin. Far less
                      # conservative than Groq (2.5s) or HF (3.0s) since the
                      # published headroom here is much larger.


def run_baseline():
    with open(INPUT_PATH) as f:
        problems = json.load(f)

    total = len(problems)
    correct_count = 0

    # 'a' mode so a partial run can be resumed without losing earlier lines.
    with open(OUTPUT_PATH, "a") as out_file:
        for i, problem in enumerate(problems, 1):
            question = problem["question"]
            ground_truth = problem["final_answer"]

            print(f"[{i}/{total}] Calling {MISTRAL_MODEL}...", end=" ", flush=True)

            try:
                raw_response = get_model_response_mistral(question)
            except Exception as e:
                print(f"API ERROR: {e}")
                error_record = {
                    "model": MISTRAL_MODEL,
                    "problem_id": problem["id"],
                    "bucket": problem.get("bucket"),
                    "question": question,
                    "ground_truth": ground_truth,
                    "raw_response": None,
                    "parsed_steps": None,
                    "parsed_answer": None,
                    "correct": False,
                    "error": str(e),
                }
                out_file.write(json.dumps(error_record) + "\n")
                out_file.flush()
                time.sleep(SLEEP_SECONDS)
                continue

            steps, parsed_answer = parse_response(raw_response)

            norm_parsed = normalize_answer(parsed_answer)
            norm_truth = normalize_answer(ground_truth)
            is_correct = (norm_parsed is not None) and (norm_parsed == norm_truth)

            if is_correct:
                correct_count += 1

            record = {
                "model": MISTRAL_MODEL,
                "problem_id": problem.get("problem_id", i),
                "bucket": problem.get("bucket"),
                "question": question,
                "ground_truth": ground_truth,
                "raw_response": raw_response,
                "parsed_steps": steps,
                "parsed_answer": parsed_answer,
                "correct": is_correct,
                "error": None,
            }

            out_file.write(json.dumps(record) + "\n")
            out_file.flush()

            status = "correct" if is_correct else "WRONG"
            print(f"{status} (parsed={parsed_answer}, truth={ground_truth})")

            time.sleep(SLEEP_SECONDS)

    accuracy = correct_count / total
    print("\n--- DONE ---")
    print(f"Model: {MISTRAL_MODEL}")
    print(f"Correct: {correct_count}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nCompare against:")
    print("  Llama-3.1-8B-Instant (Groq):        90/100 (90%)")
    print("  Gemini 3.1 Flash-Lite (abandoned):  98/100 (98%) -- too strong")
    print("(same 100-problem subset, same prompt template, same parser)")
    print("\nNOTE: ministral-8b-2512 is comparable in scale to Llama-3.1-8B,")
    print("not smaller. This result speaks to cross-family generalization of")
    print("the step-order effect at similar model scale -- NOT to H3's original")
    print("smaller-model-shows-bigger-drop claim. Frame it that way in the paper.")


if __name__ == "__main__":
    run_baseline()