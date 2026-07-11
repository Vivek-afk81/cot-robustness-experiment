"""
script to check how many questions are correct and have step_count >=3 out of all 100
"""

import json
from collections import defaultdict
from collections import Counter

def get_records_by_step_count(min_steps=3, path="data/day27_gsm8k_subset.json"):
    with open(path, "r") as f:
        records = json.load(f)

    filtered_records = [
        record for record in records
        if record["step_count"] == min_steps
    ]

    return filtered_records
# records = get_records_by_step_count(3)
# print(len(records))          # Number of records with >=3 steps
# print(records[0]["id"])      # First record's id
# print(records[0]["step_count"])

def count_records(path="data/stage1_baseline_reparsed.jsonl"):
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

# counts = count_records()
# print(counts)



def get_incorrect_records(path="data/stage1_baseline.jsonl"):
    incorrect = []

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if not record["correct"]:
                incorrect.append(record)

    return incorrect

incorrect_records = get_incorrect_records()





# with open("data/day27_gsm8k_subset.json") as f:
#     subset = json.load(f)
# bucket_by_id = {p.get("problem_id", i+1): p.get("bucket") for i, p in enumerate(subset)}

# crosstab = defaultdict(lambda: defaultdict(int))
# with open("data/stage1_baseline.jsonl") as f:
#     for line in f:
#         r = json.loads(line)
#         orig_bucket = bucket_by_id.get(r["problem_id"])
#         model_steps = len(r["parsed_steps"]) if r["parsed_steps"] else 0
#         crosstab[orig_bucket][model_steps] += 1

# for bucket, counts in sorted(crosstab.items()):
#     print(bucket, dict(sorted(counts.items())))


# with open("data/day27_gsm8k_subset.json") as f:
#     subset = json.load(f)
# bucket_by_id = {p.get("problem_id", i+1): p.get("bucket") for i, p in enumerate(subset)}

# with open("data/stage1_baseline.jsonl") as f:
#     for line in f:
#         r = json.loads(line)
#         orig_bucket = bucket_by_id.get(r["problem_id"])
#         model_steps = len(r["parsed_steps"]) if r["parsed_steps"] else 0
#         if orig_bucket == "5+-step" and model_steps == 0:
#             print("PROBLEM ID:", r["problem_id"])
#             print("QUESTION:", r["question"])
#             print("CORRECT:", r["correct"])
#             print("\nRAW RESPONSE:\n", r["raw_response"])

empty_count = 0

with open("data/stage1_baseline_reparsed.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if len(r["parsed_steps"]) == 0:
            empty_count += 1

# print(f"empty_count: {empty_count}")



with open("data/stage1_baseline_reparsed.jsonl") as f:
    records = [json.loads(line) for line in f]

# print("Total records:", len(records))
# print("Empty parsed_steps:", sum(len(r["parsed_steps"]) == 0 for r in records))
# print("Correct:", sum(r["correct"] for r in records))



with open("data/stage1_baseline_reparsed.jsonl") as f:
    records = [json.loads(line) for line in f]

# print(Counter(r["bucket"] for r in records))
# print(Counter(len(r["parsed_steps"]) for r in records))



with open("data/stage1_baseline_reparsed.jsonl") as f:
    records = [json.loads(line) for line in f]
    assert all(len(r["parsed_steps"]) >= 2 for r in records)

for bucket in ["2-step", "3-step", "4-step", "5+-step"]:
    subset = [r for r in records if r["bucket"] == bucket]
    lengths = [len(r["parsed_steps"]) for r in subset]

    # print(
    #     bucket,
    #     "mean =", sum(lengths)/len(lengths),
    #     "min =", min(lengths),
    #     "max =", max(lengths),
    # )



with open("data/stage1_baseline_reparsed.jsonl") as f:
    for line in f:
        r = json.loads(line)
        n_steps = len(r["parsed_steps"]) if r["parsed_steps"] else 0
        resp_len = len(r["raw_response"]) if r["raw_response"] else 0
        if n_steps <= 2 and resp_len > 800:  # long response, suspiciously few parsed steps
            print(r["problem_id"], n_steps, resp_len)