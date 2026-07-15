"""
analyze_h2_distribution.py

Computes the exact frequency distribution of self-reported divergence
positions — both permuted (what H2's hypothesis literally refers to) and
original (the content-fragility check) — plus flags None/unparseable cases
and any AMBIGUOUS_MAPPING records for manual review.
"""

import json
from collections import Counter

with open("results/h2_self_report_trial1.jsonl") as f:
    records = [json.loads(line) for line in f]

permuted_counter = Counter()
original_counter = Counter()
none_or_unparsed = []
ambiguous = []

for r in records:
    pd = r["permuted_divergence_step"]
    od = r["original_divergence_step"]

    if pd in (None, "none"):
        none_or_unparsed.append((r["problem_id"], r["condition"], pd))
    else:
        permuted_counter[pd] += 1

    if od == "AMBIGUOUS_MAPPING":
        ambiguous.append((r["problem_id"], r["condition"]))
    elif od not in (None, "none"):
        original_counter[od] += 1

print(f"Total records: {len(records)}")
print(f"\n--- Permuted-position divergence distribution ---")
for step in sorted(permuted_counter):
    print(f"  Step {step}: {permuted_counter[step]}")
print(f"  None/unparsed: {len(none_or_unparsed)} -> {none_or_unparsed}")

print(f"\n--- Original-position divergence distribution ---")
for step in sorted(original_counter):
    print(f"  Step {step}: {original_counter[step]}")
print(f"  Ambiguous mapping: {len(ambiguous)} -> {ambiguous}")

# Per-condition breakdown for permuted position, since conditions have
# different chain-length distributions and might cluster differently
print(f"\n--- Permuted-position by condition ---")
for cond in ["reversed", "shuffled", "partial"]:
    cond_counter = Counter(
        r["permuted_divergence_step"] for r in records
        if r["condition"] == cond and r["permuted_divergence_step"] not in (None, "none")
    )
    print(f"  {cond}: {dict(sorted(cond_counter.items()))}")