"""
utils_mistral_stepwise.py

ADDITIVE module -- does not replace anything in utils_mistral.py. Import
these alongside the existing functions. Built in response to the
Days 37-38 bypass finding: ministral-8b-2512 ignored the given step list
94.4% of the time under the original no-question prompt, re-solving from
the numbers embedded in the steps rather than actually using them in the
given order.

Two things this adds:
1. A stricter prompt that forces a visible per-step computation trace,
   making silent bypass much harder to pull off undetected, and giving a
   free-ish H2-style divergence signal (the line where the trace breaks
   is the divergence point) without a separate self-report call.
2. An automatic engagement scorer -- checks whether the response actually
   references each given step's content, in order -- so you don't have to
   manually re-read all 89 responses to catch a repeat of the bypass
   pattern. This is a screening signal, not a replacement for a manual
   spot-check -- always eyeball a handful of flagged AND unflagged cases
   before trusting the aggregate number, same lesson as Bug A/B.

IMPORTANT CAVEAT before you scale this to all 89: GSM8K steps can still be
individually self-contained arithmetic facts (e.g. "Step 3: 5 * 2 = 10")
that don't require the PRIOR step's output to compute. If most of your
steps are like that, a model could still correctly reproduce a step-wise-
LOOKING trace while secretly re-deriving each line's answer independently
rather than genuinely depending on the given order. Spot-check a few of
your actual permuted step lists for genuine forward-reference language
("using the result from the previous step...") before trusting this
prompt's stricter framing to actually be measuring order-sensitivity and
not just verbosity-shaped bypass.
"""

import re


def build_stage2_prompt_stepwise_no_question(steps):
    """
    Stricter successor to build_stage2_prompt_no_question. Forces a visible
    per-step trace instead of a single final answer, and explicitly forbids
    reordering/skipping/combining steps.

    Report results from this as a SEPARATE labeled condition, same rule as
    the original no-question variant -- do not merge with baseline-control/
    reversed numbers from the standard (with-question) pipeline.
    """
    numbered_steps = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))

    prompt = f"""Below is a step-by-step reasoning process for a math problem. \
You are NOT given the original question -- only these steps. Use ONLY the \
information contained in these steps, in the EXACT order given, to determine \
the final numeric answer.

Rules:
- Do not reinterpret the problem, do not re-derive it from any assumed context, \
and do not add information not present in the steps below.
- Do not reorder, skip, or combine steps out of sequence, even if a different \
order seems more natural to you.
- For each step below, in the order given, write one line in the format \
"Step N: <your computation for this step, using only information available \
from the problem's given steps up to and including this one>". Do not use any \
number or fact that hasn't appeared yet in the steps above.

After all step lines, end your response with exactly: "Final answer: <number>"

Steps:
{numbered_steps}"""

    return prompt


def get_model_response_stage2_mistral_stepwise_no_question(steps):
    """Stage 2 call using the stricter step-wise, steps-only prompt above."""
    from utils_mistral import _call_mistral  # reuse existing low-level call
    prompt = build_stage2_prompt_stepwise_no_question(steps)
    return _call_mistral(prompt)


# ---------------------------------------------------------------------------
# Engagement scoring -- automatic screen for bypass, not a manual replacement
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "has",
    "was", "were", "are", "will", "each", "than", "then", "into", "your",
    "step", "steps", "using", "used", "which", "there", "their", "does",
}


def _significant_words(text):
    """Lowercased words of length >=4, minus common stopwords. Used as a
    crude but cheap proxy for 'does the response actually mention this
    step's content', without needing embeddings or an extra model call."""
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def score_step_engagement(response_text, steps, overlap_threshold=0.25):
    """
    For each given step, checks whether the response's text contains a
    meaningful overlap of that step's significant words. Returns:
      - per_step: list of bools, same order as `steps`, True = referenced
      - engagement_ratio: fraction of steps referenced
      - has_stepwise_trace: whether the response contains explicit "Step N:"
        labels at all (a weaker but very cheap secondary signal)
      - trace_step_count: how many "Step N:" lines were found

    A LOW engagement_ratio (few steps referenced) on a response that still
    reaches the correct final answer is exactly the pattern that flagged
    the original bypass -- flag these for manual review rather than trusting
    the aggregate.
    """
    response_words = _significant_words(response_text)

    per_step = []
    for step_text in steps:
        step_words = _significant_words(step_text)
        if not step_words:
            per_step.append(True)  # nothing distinctive to check -- don't penalize
            continue
        overlap = len(step_words & response_words) / len(step_words)
        per_step.append(overlap >= overlap_threshold)

    engagement_ratio = sum(per_step) / len(per_step) if per_step else 0.0

    trace_lines = re.findall(r"step\s*(\d+)\s*:", response_text, re.IGNORECASE)
    has_stepwise_trace = len(trace_lines) > 0

    return {
        "per_step": per_step,
        "engagement_ratio": engagement_ratio,
        "has_stepwise_trace": has_stepwise_trace,
        "trace_step_count": len(trace_lines),
    }


def responses_near_identical(response_a, response_b, threshold=0.90):
    """
    Cheap char-level similarity check, same spirit as the original
    09_spotcheck_mistral_bypass.py comparison -- if baseline-control and
    a permuted-condition response are near-identical, that's a strong sign
    the model ignored the given order (or the given steps) entirely.
    Uses difflib's SequenceMatcher ratio rather than exact string equality,
    since trivial formatting differences (e.g. trailing punctuation) already
    showed up as a separate category in the original 89-problem pass.
    """
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, response_a, response_b).ratio()
    return ratio >= threshold, ratio