import os
import re

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


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
        step_match = re.match(r"^\d+\.\s*(.+)", line)

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