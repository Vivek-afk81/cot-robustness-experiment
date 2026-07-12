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



def get_incorrect_records(path="data/stage1_baseline_buggy.jsonl"):
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



# with open("data/stage1_baseline_reparsed.jsonl") as f:
#     for line in f:
#         r = json.loads(line)
#         n_steps = len(r["parsed_steps"]) if r["parsed_steps"] else 0
#         resp_len = len(r["raw_response"]) if r["raw_response"] else 0
#         if n_steps <= 2 and resp_len > 800:  # long response, suspiciously few parsed steps
#             print(r["problem_id"], n_steps, resp_len)

#actual step-count breakdown for just the 89 eligible problems and see if it lines up:


with open("data/stage1_baseline_reparsed.jsonl") as f:
    counts = Counter()
    for line in f:
        r = json.loads(line)
        n = len(r["parsed_steps"]) if r["parsed_steps"] else 0
        if r["correct"] and n >= 3:
            counts[n] += 1

# print(dict(sorted(counts.items())))
# print("3-step + 4-step count:", counts[3] + counts[4])


"""split the Partial number — because it's mixing two very different populations, and doing that first will change we read everything else."""

correct_by_group = defaultdict(int)
total_by_group = defaultdict(int)

with open("results/stage2_results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["condition"] != "partial":
            continue
        key = "degenerate" if r["degenerate"] else "non-degenerate"
        total_by_group[key] += 1
        if r["correct"]:
            correct_by_group[key] += 1

for key in total_by_group:
    c, t = correct_by_group[key], total_by_group[key]
    # print(f"partial ({key}): {c}/{t} ({c/t:.2%})")

"""three records show parsed=None — problems 13, 28, and 67. A None means the parser found no "Final answer: X" line at all, not that it found a wrong number. """

with open("results/stage2_baseline_control.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["problem_id"]== 67 and r["parsed_answer"] is None:
            print("PROBLEM", r["problem_id"])
            print(r["raw_response"])
            # print("---")


"""script to get the real flip-rate, per condition, which is more informative than any of the aggregate deltas """

def load_by_problem(path, condition_filter=None):
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if condition_filter and r.get("condition") != condition_filter:
                continue
            records[r["problem_id"]] = r["correct"]
    return records

# def compare_trials(trial1_path, trial2_path, condition_filter=None, label=""):
#     t1 = load_by_problem(trial1_path, condition_filter)
#     t2 = load_by_problem(trial2_path, condition_filter)
#     common_ids = set(t1) & set(t2)

#     flipped = [pid for pid in common_ids if t1[pid] != t2[pid]]
#     print(f"{label}: {len(flipped)}/{len(common_ids)} problems flipped "
#           f"({len(flipped)/len(common_ids):.1%})")
#     if flipped:
#         print(f"  flipped problem_ids: {sorted(flipped)}")

# compare_trials("results/stage2_results.jsonl", "results/stage2_results_trial2.jsonl",
#                "reversed", "Reversed")
# compare_trials("results/stage2_results.jsonl", "results/stage2_results_trial2.jsonl",
#                "shuffled", "Shuffled")
# compare_trials("results/stage2_results.jsonl", "results/stage2_results_trial2.jsonl",
#                "partial", "Partial")
# compare_trials("results/stage2_baseline_control.jsonl", "results/stage2_baseline_control_trial2.jsonl",
#                None, "Baseline-control")

"""check problem 67's actual question/steps"""

with open("data/stage1_baseline_reparsed.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["problem_id"] == 67:
            print("QUESTION:", r["question"])
            print("GROUND TRUTH:", r["ground_truth"])
            print("\nORIGINAL STAGE 1 STEPS (in order):")
            for i, step in enumerate(r["parsed_steps"], 1):
                print(f"  {i}. {step}")
            break