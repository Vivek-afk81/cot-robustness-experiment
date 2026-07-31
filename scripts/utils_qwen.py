"""
utils_h3.py

Utility functions for the H3 cross-size comparison run.
Model: qwen/qwen3.6-27b via Groq  (~27B parameters, preview)
Reference model: llama-3.1-8b-instant (~8B) — see utils.py

Mirrors utils.py exactly in prompt design, parsing, and scoring so the
two models are compared on strictly identical task definitions.  The
Groq client is intentionally shared (same API key, same rate-limit tier)
so the only thing that varies between runs is the model ID.

H3 hypothesis: a larger model should be MORE robust (i.e., smaller
accuracy drop) to step-order permutation than the 8B baseline.  A 27B
model is not the original H3 design (which called for a *smaller* model),
but it gives a cross-size directional check available within the current
free-tier constraint.  Label this clearly in the paper as
"cross-size replication at > 8B scale", not as an H3 confirmation.
"""

import os
import re
import random

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL_ID = "qwen/qwen3.6-27b"   # change here only if switching models

# ---------------------------------------------------------------------------
# Stage 1 — free-generation baseline
# ---------------------------------------------------------------------------

def get_model_response(question):
    """Send one GSM8K question to the model and return the raw response text."""
    prompt = f"""Solve this problem step by step. Number each step (1., 2., 3., ...).
End your response with exactly: "Final answer: <number>"

Problem: {question}"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        temperature=0.7,       # Groq docs: non-thinking mode best practice (temp=0.0 causes degenerate output)
        reasoning_effort="none",  # disable <think> block; forces direct formatted response
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def parse_response(raw_text):
    """Stateful line-by-line parser — same logic as utils.py.
    Accepts both N. and (N). step numbering, same as the reference parser."""
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    steps = []
    final_answer = None
    current_step = None

    for line in lines:
        answer_match = re.search(r"final answer:?\s*\$?(-?[\d,]+\.?\d*)", line, re.IGNORECASE)
        # Matches: "1. ...", "(1). ...", "(1) ..." — same two patterns as utils.py
        step_match = re.match(r"^\(?\d+\)?[.)\s]\s*(.+)", line)
        # Also accept "Step N:" prefix that Qwen sometimes emits
        if not step_match:
            step_match = re.match(r"^step\s+\d+[:.]\s*(.+)", line, re.IGNORECASE)

        if answer_match:
            final_answer = answer_match.group(1).replace(",", "")
            continue

        if step_match:
            if current_step is not None:
                steps.append(current_step)
            current_step = step_match.group(1)
        else:
            if current_step is not None:
                current_step += " " + line

    if current_step is not None:
        steps.append(current_step)

    return steps, final_answer


def normalize_answer(answer):
    """
    Same normalizer as utils.py — kept as a separate copy here per
    utils_h3.py's original isolation design (no cross-import from utils.py
    for this project's H3 module), but MUST stay in sync with utils.py's
    version. If you ever change one, change both.

    FIXED (see readme log): the original version only stripped a single
    trailing ".0" (e.g. "62.0" -> "62"), missing multi-zero decimals like
    "2.00" or "6.00". Confirmed cause of 5/273 Qwen 27B Stage-2 "errors"
    being false negatives (model answered "2.00", ground truth "2").

    Fix: convert to float and back to a canonical string, collapsing any
    number of trailing zeros to the same normalized value. Falls back to
    the original minimal cleanup for non-numeric/malformed input.
    """
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


# ---------------------------------------------------------------------------
# Permutation helpers (identical to utils.py — copied here for isolation)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 2 — scaffold prompt (identical design to utils.py)
# ---------------------------------------------------------------------------

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


def get_model_response_stage2(question, steps):
    """Send a permuted step sequence to the model and return the raw response text."""
    prompt = build_stage2_prompt(question, steps)
    response = client.chat.completions.create(
        model=MODEL_ID,
        temperature=0.7,       # consistent with Stage 1 call above
        reasoning_effort="none",  # disable <think> block
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
