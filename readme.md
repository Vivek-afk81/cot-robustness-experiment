# Step Order Sensitivity in Chain-of-Thought Reasoning
### An Empirical Study on Small Open-Weight LLMs

This README is a running methodology log, written as the experiment progresses —
not a polished final writeup. Entries are dated and kept in the order they happened,
including dead ends and corrected mistakes. This becomes the backbone of the paper's
methods and related-work sections later.

---

## Research question

When Chain-of-Thought reasoning steps are reordered (reversed, shuffled, partially
permuted) but step *content* stays correct, does Llama 3.1 8B still reach the right
answer on GSM8K? Where does it fail first?

## Novelty (related work)

| Paper | What varies | What's fixed |
|---|---|---|
| GSM-Symbolic (Mirzadeh et al., ICLR 2025) | Problem surface: names, numbers, clause count | Model generates own CoT fresh each time — no existing trace touched |
| Fragile Thoughts (Aravindan & Kejriwal, arXiv 2603.03332, 2026) | Reasoning chain *content*: wrong math, wrong units, sycophancy, skipped/extra steps | Step *order* — always preserved |
| **Ours** | Reasoning chain *order*: reversed, shuffled, partial permutation | Step *content* — always correct |

We adopt Fragile Thoughts' robustness formula for comparability:
`Robustness_τ(M) = Accuracy(perturbed) / Accuracy(clean)`

## Hypotheses (locked before running anything)

- H1: Accuracy drops monotonically Baseline → Reversed → Partial → Shuffled
- H2: First error concentrates at Step 2 position when shuffled
- H3: Smaller model (cross-model) shows larger accuracy drop
- H4 (null): No significant difference between Baseline and Reversed

## Architecture note

Local laptop (Windows), not Colab — every model call is an HTTP request to Groq's
API, so no local GPU/compute is used and Colab offers no benefit here.

---

## Log

### [] — Dataset construction
Pulled GSM8K test split (1,319 problems) from `github.com/openai/grade-school-math`.
Built a stratified 100-problem subset, 25 each from 2-step / 3-step / 4-step / 5+-step
buckets (bucket = calculator-annotated line count in the *reference* solution — a proxy
measure, stated here as a methodology caveat, not a formal definition). Pool sizes before
sampling: 326/371/297/325. Seed=42 for reproducibility. Saved to
`data/day27_gsm8k_subset.json`.

### [Date] — Stage 1 prompt + parser development
Wrote `scripts/test_single.py` to test the Stage 1 prompt and response format on one
problem before committing to a full run.

Prompt template:
```
Solve this problem step by step. Number each step (1., 2., 3., ...).
End your response with exactly: "Final answer: <number>"

Problem: {question}
```
Model: `llama-3.1-8b-instant` via Groq, temperature=0.0.

**Bug found:** first version of `parse_response()` only captured lines starting with
"N." and silently dropped un-numbered continuation lines (sub-calculations under a
step), losing real step content. **Fix:** stateful line-by-line parser that appends
non-numbered lines onto the step currently being built. Verified on 2 problems across
different step-count buckets before trusting it for the full run.

### [10-07-2026] — Stage 1 full baseline run
Ran `scripts/01_generate_baseline.py` over all 100 problems (2.5s sleep between calls,
Groq free tier). Results saved incrementally to `data/stage1_baseline.jsonl` — one line
per problem, including full raw response for later debugging, not just parsed output.

**Result: 90/100 correct (90% accuracy).**

**Sanity check against external anchors:**
- Fragile Thoughts (Llama-3.1-8B-Instruct, clean): ~87–96% → our 90% falls inside this range
- GSM-Symbolic (Llama3-8b-instruct, their 100-q subset): 74% → our number is higher, but
  this is a different subset with a different sampling/difficulty distribution, not a
  directly comparable baseline — not treated as a red flag on its own

**Manually inspected all "WRONG" cases and 3 "correct" cases to rule out parsing bugs
before trusting the 90% figure.** Findings:
- Problem #6 (Stephen's loan): model reached the *correct* intermediate answer at step 4
  ($31), then second-guessed itself, invented an unrequested compounding-interest scheme,
  and reasoned its way to a wrong final answer ($315). Genuine reasoning failure, not a
  parsing artifact — the parser faithfully captured every step. Notable failure mode:
  passing through a correct answer before reasoning past it.
- Problem #69 (Britany's TikTok time): model's arithmetic was internally correct (18.8
  hours) but failed to convert the final answer into the units the question required
  (minutes; ground truth 1128 min = 18.8 hr). A unit-tracking failure, not a math error
  or parsing bug.
- Spot-checked "correct" cases with comma-formatted ground truth (e.g. 114,200 vs 114200)
  — normalizer handled these correctly.

**Conclusion: 90% accuracy is trustworthy, not a pipeline artifact.**

### [11-07-2026] — Stage 2 scoping decisions
Decision: only permute problems Stage 1 got correct (90 of 100). Permuting an already-
wrong chain doesn't isolate order effects, since content was already broken — matches
Fragile Thoughts' approach.

Decision: minimum 3 steps required for a problem to enter Stage 2. Checked actual counts
on the correct subset: `{'step3plus': 98, 'correct_and_3plus': 88}` — only 2 problems
lost by this cutoff, negligible impact on bucket representation.

finding:found parser bug: model occasionally used (N). step numbering instead of N., causing one record to silently parse to zero steps despite a correct final answer. Fixed regex to accept both formats. Re-parsed into stage1_baseline_reparsed.jsonl. step3plus 98→99, correct_and_3plus 88→89.


Finding: Although the evaluation set was stratified by reference solution length (25 problems per bucket), the model generated substantially longer reasoning chains, with mean parsed lengths of 4.48, 4.72, 5.12, and 6.92 steps for the 2-, 3-, 4-, and 5+-step buckets respectively.

known limitation : Partial-permutation degeneracy is not limited to exactly-3-step chains. A 4-step chain's middle sub-list has only 2 elements, and the only non-identity permutation of 2 elements is equivalent to reversing them — so Partial collapses into either Baseline or a middle-reversal for 4-step chains too, not just 3-step ones. Confirmed empirically: 38/89 eligible problems (all 3-step and 4-step chains) are flagged degenerate for Partial. Partial is only meaningfully distinct from Baseline/Reversed for chains with ≥5 steps (51/89 problems). Stage 4 analysis will report Partial accuracy separately for degenerate vs. non-degenerate chains rather than blending them into one aggregate, to avoid biasing the reported Partial-vs-Baseline gap toward looking artificially small.

### [11-07-2026] — Stage 2 permutation engine


### [12-07-2026]

 Re-run stability check, due to a concern that temp=0.0 does not guarantee identical outputs across separate API calls — re-ran all 4 Stage 2 conditions once more (Trial 2) to check. Core finding held: Reversed < Shuffled < Partial in both trials. Per-problem flip rate (correct↔wrong between trials) scales with order disruption: Baseline-control 2.2%, Partial 5.6%, Shuffled 6.7%, Reversed 13.5% — disrupted order doesn't just lower accuracy, it lowers reproducibility. Reporting Reversed accuracy as a 2-trial average (58.4%), not a single-run point estimate, given its noise level.
---

## Open items / deferred decisions

- Cross-model choice for H3 (Mixtral-8x7b vs. Gemma2-9b-it) — not yet decided
- Exact answer-normalization edge cases (units, currency symbols) — deliberately minimal
  for now, will expand if a real mismatch is found
- Rate-limit backoff strategy specifics — not yet needed, current sleep(2.5) sufficient