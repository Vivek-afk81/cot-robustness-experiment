import re
import json
from collections import defaultdict
from collections import Counter

"""
Fixed regex to accept both formats (N) and N. 
"""

# import json
# from utils import parse_response, normalize_answer

# INPUT = "data/stage1_baseline.jsonl"
# OUTPUT = "data/stage1_baseline_reparsed.jsonl"

# with open(INPUT) as fin, open(OUTPUT, "w") as fout:
#     for line in fin:
#         record = json.loads(line)

#         if record["raw_response"] is None:
#             fout.write(json.dumps(record) + "\n")
#             continue

#         steps, parsed_answer = parse_response(record["raw_response"])

#         record["parsed_steps"] = steps
#         record["parsed_answer"] = parsed_answer

#         record["correct"] = (
#             normalize_answer(parsed_answer)
#             == normalize_answer(record["ground_truth"])
#         )

#         fout.write(json.dumps(record) + "\n")

"""Verify the fix """

# import re

# def parse_response(raw_text):
#     lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
#     steps = []
#     final_answer = None
#     current_step = None
#     for line in lines:
#         answer_match = re.search(r"final answer:?\s*\$?(-?[\d,]+\.?\d*)", line, re.IGNORECASE)
#         step_match = re.match(r"^\(?(\d+)\)?\.\s*(.+)", line)
#         if answer_match:
#             final_answer = answer_match.group(1).replace(",", "")
#             continue
#         if step_match:
#             if current_step is not None:
#                 steps.append(current_step)
#             current_step = step_match.group(2)  # <-- the fix
#         else:
#             if current_step is not None:
#                 current_step += " " + line
#     if current_step is not None:
#         steps.append(current_step)
#     return steps, final_answer

# test = """1. We know that Jenna's mom picked 20 apples.
# 2. Jenna picked half as many apples as her mom.
# 3. So, Jenna picked 20 / 2 = 10 apples.
# Final answer: 30"""

# steps, answer = parse_response(test)
# print(steps)
# print(answer)

import json

with open("results/stage2_results_v2_trial1.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["problem_id"] == 4 and r["condition"] == "reversed":
            print(r["raw_response"])
            break