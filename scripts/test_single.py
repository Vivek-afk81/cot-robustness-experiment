import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

# Load your API key from .env into the environment
load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Load just the first problem from your subset, to test on one
with open("data/day27_gsm8k_subset.json") as f:
    problems = json.load(f)

# sanity check
# print(f"Loaded {len(problems)} problems")

test_problem = problems[5]
print("QUESTION:", test_problem["question"])
print("GROUND TRUTH ANSWER:", test_problem["final_answer"])
print("---")

# Build the prompt
prompt = f"""Solve this problem step by step. Number each step (1., 2., 3., ...).
End your response with exactly: "Final answer: <number>"

Problem: {test_problem['question']}"""

# Call the model
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    messages=[{"role": "user", "content": prompt}]
)

# Print the raw output so we can inspect it
print("\nRAW RESPONSE:")
print(response.choices[0].message.content)

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

# Test it on the response we just got
steps, parsed_answer = parse_response(response.choices[0].message.content)
print("PARSED STEPS:")
for i, s in enumerate(steps, 1):
    print(f"  {i}: {s}")
print("PARSED FINAL ANSWER:", parsed_answer)
