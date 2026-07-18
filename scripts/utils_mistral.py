"""
utils_mistral.py

REST implementation for Ministral 3 8B.
Avoids SDK import/version issues by calling the HTTP API directly.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

MISTRAL_MODEL = "ministral-8b-2512"
API_URL = "https://api.mistral.ai/v1/chat/completions"

API_KEY = os.environ["MISTRAL_API_KEY"]


def _call_mistral(prompt: str) -> str:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MISTRAL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.0,
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def get_model_response_mistral(question):
    """
    Stage 1 prompt.
    Matches the Llama prompt exactly.
    """
    prompt = f"""Solve this problem step by step. Number each step (1., 2., 3., ...).
End your response with exactly: "Final answer: <number>"

Problem: {question}"""

    return _call_mistral(prompt)


def get_model_response_stage2_mistral(question, steps):
    """
    Stage 2 prompt.
    Reuses the same prompt builder as the Llama experiment.
    """
    from utils import build_stage2_prompt

    prompt = build_stage2_prompt(question, steps)
    return _call_mistral(prompt)