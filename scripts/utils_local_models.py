"""
utils_local_models.py

Same parsing / normalization / permutation logic as your utils.py, with the
model call swapped from Groq's API to a local llama.cpp model via
local_model.py. Every function below other than the *_response ones is a
direct port of your original code, unchanged in behavior, so Stage 1/Stage 2
results are governed by the same rules as Llama/Mistral/Qwen.

Field-naming note: your data file (data/day27_gsm8k_subset.json) uses "id"
and "final_answer" as keys, not "problem_id"/"answer". This file and the
scripts that use it are written against the real keys. Your original
01_generate_baseline.py has `problem.get("problem_id", i)`, which always
misses (the key doesn't exist) and silently falls back to the loop index —
worth checking whether that's already caused any id-mismatch issues in your
existing Llama/Mistral/Qwen results before drawing conclusions that involve
matching problems by id across files.
"""

import re
import random

from local_model import call_model

MODEL_KEY = None  # set by caller (e.g. "phi3-mini" or "qwen2.5-3b") before use



# Stage 1

def get_model_response(question, model_key=None):
    """Send one GSM8K question to the local model and return the raw response text."""
    key = model_key or MODEL_KEY
    if key is None:
        raise ValueError("model_key must be set (pass it in or set utils_local_models.MODEL_KEY)")

    prompt = f"""Solve this problem step by step. Number each step (1., 2., 3., ...).
End your response with exactly: "Final answer: <number>"

Problem: {question}"""

    return call_model(key, prompt, max_tokens=600, temperature=0.0)


def parse_response(raw_text):
    """Identical logic to utils.py's parse_response — stateful line-by-line
    parser, accepts 'N.' and '(N).' numbering, merges un-numbered
    continuation lines into the step currently being built, and extracts
    the final answer via a separate regex pass so numbering format doesn't
    affect answer extraction."""
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    steps = []
    final_answer = None
    current_step = None

    for line in lines:
        answer_match = re.search(r"final answer:?\s*\$?(-?[\d,]+\.?\d*)", line, re.IGNORECASE)
        step_match = re.match(r"^\(?(\d+)\)?\.\s*(.+)", line)

        if answer_match:
            final_answer = answer_match.group(1).replace(",", "")
            continue

        if step_match:
            if current_step is not None:
                steps.append(current_step)
            current_step = step_match.group(2)
        else:
            if current_step is not None:
                current_step += " " + line

    if current_step is not None:
        steps.append(current_step)

    return steps, final_answer


def normalize_answer(answer):
    """Identical to utils.py's fixed normalizer (handles multi-zero
    decimals, commas, $ signs; falls back to minimal cleanup on
    non-numeric input rather than crashing)."""
    if answer is None:
        return None
    answer = str(answer).strip()
    answer = answer.replace("$", "").replace(",", "")
    answer = answer.rstrip(".")

    try:
        value = float(answer)
        if value == int(value):
            return str(int(value))
        return str(value)
    except (ValueError, OverflowError):
        if answer.endswith(".0"):
            answer = answer[:-2]
        return answer



# Stage 2 — permutation engine 


MAX_SHUFFLE_ATTEMPTS = 100


def get_reversed(steps):
    return list(reversed(steps))


def get_shuffled(steps, seed):
    original = list(steps)
    reversed_order = list(reversed(steps))

    if len(steps) < 2:
        return list(steps), True

    rng = random.Random(seed)
    candidate = list(steps)

    for _ in range(MAX_SHUFFLE_ATTEMPTS):
        rng.shuffle(candidate)
        if candidate != original and candidate != reversed_order:
            return candidate, False

    return candidate, True


def get_partial(steps, seed):
    if len(steps) < 3:
        return list(steps), True

    first, middle, last = steps[0], steps[1:-1], steps[-1]

    if len(middle) < 2:
        return list(steps), True

    original_middle = list(middle)
    reversed_middle = list(reversed(middle))

    rng = random.Random(seed)
    candidate_middle = list(middle)

    for _ in range(MAX_SHUFFLE_ATTEMPTS):
        rng.shuffle(candidate_middle)
        if candidate_middle != original_middle and candidate_middle != reversed_middle:
            return [first] + candidate_middle + [last], False

    return [first] + candidate_middle + [last], True



# Stage 2 — prompt + model call 

def build_stage2_prompt(question, steps):
    numbered_steps = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))

    prompt = f"""Below is a step-by-step reasoning process for a math problem. \
Use the steps exactly as given, in the order given, to determine the final answer. \
Do not skip, reorder, or add steps.
End your response with exactly: "Final answer: <number>"

Problem: {question}

Steps:
{numbered_steps}"""

    return prompt


def get_model_response_stage2(question, steps, model_key=None):
    key = model_key or MODEL_KEY
    if key is None:
        raise ValueError("model_key must be set (pass it in or set utils_local_models.MODEL_KEY)")

    prompt = build_stage2_prompt(question, steps)
    return call_model(key, prompt, max_tokens=600, temperature=0.0)


# Self-test

if __name__ == "__main__":
    sample = """Sure, let's work through this.
1. Tom has 5 apples.
2. He buys 3 more.
   3 + 5 = 8
3. He gives away 2.
(4). Final count: 8 - 2 = 6
Final answer: 6"""

    steps, final_answer = parse_response(sample)
    print("Parsed steps:", steps)
    print("Final answer:", final_answer)

    assert len(steps) == 4
    assert "3 + 5 = 8" in steps[1]
    assert final_answer == "6"
    assert normalize_answer("62.00") == "62"
    assert normalize_answer("114,200") == "114200"
    assert normalize_answer("$31") == "31"

    print("\nAll self-tests passed.")