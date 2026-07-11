"""
Fixed regex to accept both formats (N) and N. 
"""

import json
from utils import parse_response, normalize_answer

INPUT = "data/stage1_baseline.jsonl"
OUTPUT = "data/stage1_baseline_reparsed.jsonl"

with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        record = json.loads(line)

        if record["raw_response"] is None:
            fout.write(json.dumps(record) + "\n")
            continue

        steps, parsed_answer = parse_response(record["raw_response"])

        record["parsed_steps"] = steps
        record["parsed_answer"] = parsed_answer

        record["correct"] = (
            normalize_answer(parsed_answer)
            == normalize_answer(record["ground_truth"])
        )

        fout.write(json.dumps(record) + "\n")