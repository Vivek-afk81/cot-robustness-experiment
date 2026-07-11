import os
import re
import random

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


#stage 1

def get_model_response(question):
    """Send one GSM8K question to the model and return the raw response text."""
    prompt = f"""Solve this problem step by step. Number each step (1., 2., 3., ...).
End your response with exactly: "Final answer: <number>"

Problem: {question}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def parse_response(raw_text):
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    steps = []
    final_answer = None
    current_step = None  # holds the step we're currently building

    for line in lines:
        answer_match = re.search(r"final answer:?\s*\$?(-?[\d,]+\.?\d*)", line, re.IGNORECASE)
        step_match = re.match(r"^\(?(\d+)\)?\.\s*(.+)", line)

        if answer_match:
            final_answer = answer_match.group(1).replace(",", "")
            continue  # don't treat this line as step content

        if step_match:
            # A new numbered step started — save the previous one first
            if current_step is not None:
                steps.append(current_step)
            current_step = step_match.group(1)
        else:
            # This line has no number — it's a continuation of the current step
            if current_step is not None:
                current_step += " " + line

    # Don't forget the last step being built when the loop ends
    if current_step is not None:
        steps.append(current_step)

    return steps, final_answer


def normalize_answer(answer):
    """Basic normalization so parsed and ground-truth answers compare cleanly.
    Deliberately basic for now — exact edge cases (units, currency) deferred."""
    if answer is None:
        return None
    answer = str(answer).strip()
    answer = answer.replace("$", "").replace(",", "")
    answer = answer.rstrip(".")
    if answer.endswith(".0"):
        answer = answer[:-2]
    return answer


#stage 2
"""
generate the three Stage 2 perturbation conditions from a problem's
Stage 1 parsed steps, with a safeguard against accidental identity/collision
(a "shuffle" that happens to land back on the original or reversed order).
"""

MAX_SHUFFLE_ATTEMPTS = 100


def get_reversed(steps):
    """Reverse step order. Well-defined for any step count >= 2."""
    return list(reversed(steps))


def get_shuffled(steps, seed):
    """
    Randomly permute step order, seeded by problem_id for reproducibility.
    Re-shuffles (deterministically, using the same seeded RNG) until the
    result differs from both the original order and the fully-reversed order.
    If it can't find a non-colliding permutation within MAX_SHUFFLE_ATTEMPTS
    (only possible for very short chains with few valid permutations),
    returns the original order with degenerate=True rather than silently
    accepting a collision.
    """
    original = list(steps)
    reversed_order = list(reversed(steps))

    if len(steps) < 2:
        return list(steps), True  # can't meaningfully shuffle

    rng = random.Random(seed)
    candidate = list(steps)

    for _ in range(MAX_SHUFFLE_ATTEMPTS):
        rng.shuffle(candidate)
        if candidate != original and candidate != reversed_order:
            return candidate, False

    # Exhausted attempts without finding a non-colliding permutation.
    return candidate, True


def get_partial(steps, seed):
    """
    First and last step fixed in place; only the middle sub-list is shuffled.
    Same collision safeguard applied to the middle sub-list only (checked
    against the middle's original order and the middle's reversed order).

    Degenerate cases (flagged, not silently produced):
      - Exactly 3 steps: middle has exactly 1 step, nothing to shuffle —
        Partial is mathematically identical to Baseline. Known, documented
        limitation (see README log).
      - Middle shuffle can't find a non-colliding permutation within
        MAX_SHUFFLE_ATTEMPTS.
    """
    if len(steps) < 3:
        # Shouldn't happen given the >=3-step Stage 2 filter, but guard anyway.
        return list(steps), True

    first, middle, last = steps[0], steps[1:-1], steps[-1]

    if len(middle) < 2:
        # Exactly 3-step chain: no meaningful middle to shuffle.
        return list(steps), True

    original_middle = list(middle)
    reversed_middle = list(reversed(middle))

    rng = random.Random(seed)
    candidate_middle = list(middle)

    for _ in range(MAX_SHUFFLE_ATTEMPTS):
        rng.shuffle(candidate_middle)
        if candidate_middle != original_middle and candidate_middle != reversed_middle:
            return [first] + candidate_middle + [last], False

    # Exhausted attempts — return best-effort with degenerate flag.
    return [first] + candidate_middle + [last], True


"""Stage 2 differs from Stage 1: instead of asking the model to generate its own
CoT, we hand it a pre-built (permuted) step sequence and ask ONLY for the
final answer. This is deliberate — if we let the model re-reason freely, it
could silently regenerate a correct-order chain and mask the effect we're
testing (whether step ORDER, not content, affects the final answer).
"""


def build_stage2_prompt(question, steps):
    """
    Presents the given steps in the order provided (already permuted by the
    caller) as a sequential numbered list, and asks the model to use them
    to reach the final answer — not to re-derive or reorder them itself.
    """
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
        model="llama-3.1-8b-instant",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content