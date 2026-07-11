"""
script to check how many questions are correct and have step_count >=3 out of all 100
"""

import json

def count_records(path="data/stage1_baseline.jsonl"):
    correct_and_3plus = 0
    step3plus = 0

    with open(path) as f:
        for line in f:
            record = json.loads(line)

            if len(record["parsed_steps"]) >= 3:
                step3plus += 1

                if record["correct"]:
                    correct_and_3plus += 1

    return {
        "step3plus": step3plus,
        "correct_and_3plus": correct_and_3plus,
    }

counts = count_records()
print(counts)