"""
utils_mistral.py

Day 36 (final model choice) — cross-model block.

Model-specific client + response function for ministral-8b-2512 via
Mistral AI's own API (not a third-party gateway) -- chosen after Gemini
3.1 Flash-Lite (too strong, 98% vs Llama's 90%) and two Hugging Face
Inference Providers candidates (gemma-2-2b-it, Phi-3.5-mini-instruct --
neither deployed by any provider) didn't pan out, and HF's own free-tier
rate limits proved too restrictive for a 189-call run anyway.

NOTE ON WHAT THIS MODEL ACTUALLY TESTS: ministral-8b-2512 is comparable in
size to Llama-3.1-8B-Instant (both ~8B dense), NOT smaller. This means the
comparison no longer directly tests H3's original premise ("does a
SMALLER/WEAKER model show a larger accuracy drop") -- it instead tests a
related but distinct question: does the step-order-sensitivity effect
generalize across model families/providers at a comparable parameter
scale? That's still a legitimate, useful result -- just log it accurately
in the paper as a same-scale cross-family comparison, not evidence for or
against H3's scaling claim specifically. If H3 itself still needs testing,
a genuinely smaller model remains an open item for a future extension.

Rate limits (per Mistral's published figures at time of writing):
  ministral-8b-2512: 625,000 TPM, 3.13 RPS
  This is far more headroom than either Groq's free tier or HF's Inference
  Providers gateway -- no meaningful risk of hitting limits across a
  ~189-call Stage 1 + Stage 2 (Baseline-control vs Reversed) run.

Requires:
  pip install mistralai
  MISTRAL_API_KEY set in your .env (from Mistral AI's La Plateforme --
  note a credit card is required even for the free development tier, which
  provides a small amount of test credit; verify your account has enough
  before a full run).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

MISTRAL_MODEL = "ministral-8b-2512"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

_api_key = os.environ["MISTRAL_API_KEY"]


def _call_mistral(prompt, max_tokens=1024):
    """
    Shared low-level call. Uses raw `requests` against Mistral's REST API
    directly, rather than the `mistralai` SDK -- the SDK's import
    (`from mistralai import Mistral`) raised an ImportError with an
    "unknown location" for this project's environment (likely a stray
    local file/folder shadowing the package, or an outdated SDK version
    still on the old v0.x import path). A raw `requests` call to the
    documented REST endpoint was already confirmed working manually before
    this rewrite, so this sidesteps the SDK issue entirely rather than
    debugging it further.
    """
    response = requests.post(
        MISTRAL_API_URL,
        headers={
            "Authorization": f"Bearer {_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MISTRAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
    )
    response.raise_for_status()  # raises for 4xx/5xx so callers' try/except catches it
    data = response.json()
    return data["choices"][0]["message"]["content"]


def build_stage2_prompt_no_question(steps):
    """
    EXPLORATORY, POST-BYPASS-FINDING VARIANT -- not used for the original
    H1/H2 pipeline or for Llama. Deliberately withholds the original
    question, showing ONLY the (possibly reordered) step list. The
    standard Stage 2 prompt (build_stage2_prompt in utils.py) always
    includes the original question alongside the steps -- this gives a
    capable model an escape hatch to just re-solve the problem from
    scratch and ignore the given steps entirely, which is what the spot-
    check (09_spotcheck_mistral_bypass.py) found ministral-8b-2512 doing
    in 89/89 eligible cases.

    By removing the question, the model has nothing to fall back to: it
    must either use the given steps (in whatever order they're in) to
    produce an answer, or fail to answer at all. This is a much stricter
    test of whether step order actually matters to the model's reasoning.

    NOTE: this changes what's being measured. The original Stage 2 prompt
    tests "does step order matter when the model ALSO has the original
    question available" (arguably a more realistic scenario, closer to how
    a person might skim reordered notes while still knowing the problem).
    This variant tests "does step order matter when the steps are the
    model's ONLY information" (a stricter, more artificial scenario, but
    one that can't be bypassed the way the original was). Report results
    from this variant as a SEPARATE, labeled condition -- do not merge
    these numbers with the original baseline-control/reversed results.
    """
    numbered_steps = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))

    prompt = f"""Below is a step-by-step reasoning process for a math problem. \
You are NOT given the original question -- only these steps. Use ONLY the \
information contained in these steps, in the order given, to determine the \
final numeric answer. Do not reinterpret the problem, do not re-derive it \
from any assumed context, and do not add information not present in the \
steps below.
End your response with exactly: "Final answer: <number>"

Steps:
{numbered_steps}"""

    return prompt


def get_model_response_stage2_mistral_no_question(steps):
    """Stage 2 call using the steps-only (no question) prompt variant above."""
    prompt = build_stage2_prompt_no_question(steps)
    return _call_mistral(prompt)


def get_model_response_mistral(question):
    """
    Send one GSM8K question to ministral-8b-2512 and return the raw
    response text. Prompt is IDENTICAL to the Llama/Gemini/HF Stage 1
    prompt -- only the model differs, to keep the comparison clean.
    """
    prompt = f"""Solve this problem step by step. Number each step (1., 2., 3., ...).
End your response with exactly: "Final answer: <number>"

Problem: {question}"""

    return _call_mistral(prompt)


def get_model_response_stage2_mistral(question, steps):
    """Stage 2 equivalent -- same build_stage2_prompt format as the other models."""
    from utils import build_stage2_prompt  # local import, same reasoning as utils_gemini.py

    prompt = build_stage2_prompt(question, steps)
    return _call_mistral(prompt)