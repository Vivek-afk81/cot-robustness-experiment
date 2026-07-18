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
### [17-07-2026]

#### 1. Methodology change (record this explicitly in Methods)
 
- Original annotation pass (Day 33/34) used a free-form "gut call" guess
  for divergence step.
- Replaced with a **rule-based taxonomy**, four categories:
  - (a) MISUSE — model misuses/misreads a step it was given
  - (b) INVENTED — model fabricates a transition/content not in the given steps
  - (c) SELF-BREAK — model explicitly flags something as off, then continues
    anyway with degraded reasoning
  - (d) SKIP/IGNORE — model skips or ignores a given step
- `08_h2_careful_analysis.py` is the canonical analysis script going forward.
  Uses `CAREFUL_first_divergence_step_guess` and `CAREFUL_which_rule_applied`
  fields, not the earlier gut-call fields.
- **Why this matters for the paper**: rule-based annotation is more
  defensible than free-form guessing (reviewers can audit against a fixed
  rubric) and it changed the numbers meaningfully vs. the gut-call pass —
  see Section 3 below. Report both passes existed; state the rule-based
  pass as the one used for reported results.
---
 
#### 2. THE CONTRADICTION (headline item — do not lose this)
 
- **Model self-report** (Day 32, `analyze_h2_distribution.py`,
  n=35 wrong answers): divergence clusters at **Step 3** (11/35, ~31%).
- **Careful, rule-based human annotation** (this pass, n=30 non-"none"
  cases): divergence clusters **earlier** — co-modal at **Step 1 and Step 2**
  (8/30 each), median = 2, mean = 3.03, and **73% of divergences occur
  within the first 3 steps**.
- **Interpretation**: this is not just "the self-report is unreliable" (a
  vague claim) — it is a **specific, directional bias**: the model's
  self-reported divergence point is systematically **later** than where a
  blind human annotator locates the actual break. Frame this in the paper
  as a directional finding, not just a reliability caveat.
- Numeric agreement with self-report (careful vs. self-report, n=35):
  - Exact: 10/35 (28.6%)
  - Off-by-one: 11/35 (31.4%)
  - Combined exact+off-1: 21/35 (60.0%)
  - Bigger disagreement: 14/35 (40.0%)
  - (Note: this 60% combined is HIGHER than the earlier gut-call comparison,
    which had given 45.7% combined — the rule-based pass agrees with the
    model somewhat more than the gut-call pass did. Reconcile/mention both
    numbers if the gut-call pass is kept as supplementary material.)
---
 
#### 3. Known bug / caveat to fix before quoting "the mode"
 
- `08_h2_careful_analysis.py`'s summary section calls Step 1 "the modal
  divergence position" using `Counter.most_common(1)`, which silently
  breaks ties by insertion order.
- **Actual data: Step 1 and Step 2 are tied at 8/30 each.** This is a TIE,
  not a clean single mode.
- **Action item**: do not state "modal position was Step 1" verbatim in
  the paper. Correct phrasing: "co-modal at Steps 1–2 (8/30 each)."
- Also worth fixing in the script itself (detect ties, report all tied
  values) before generating any more paper-facing summaries from it.
---
 
#### 4. New finding not in original H2 hypothesis: failure-mode taxonomy
 
- Rule breakdown across 30 classified cases:
  - (a) MISUSE: 0 (0.0%)
  - (b) INVENTED: 6 (20.0%)
  - (c) SELF-BREAK: 15 (50.0%)
  - (d) SKIP/IGNORE: 9 (30.0%)
- **SELF-BREAK is the dominant failure mode (50%)**: the model notices
  something is inconsistent and says so, but continues reasoning anyway
  with degraded output rather than correcting course or stopping.
- **Zero MISUSE cases** — the model never simply misreads a step it was
  given; failures are about what it does *after* noticing a problem, not
  about basic misreading. Worth a sentence in Discussion.
- This taxonomy is an original contribution beyond the original H2
  hypothesis (which only asked "where," not "how"). Consider giving it its
  own subsection in the paper rather than folding it into H2 as a footnote.
---
 
#### 5. Condition-level patterns (n small — report as suggestive, not confirmed)
 
| Condition | n (numeric) | Mean step | Median | Mode | none count |
|---|---|---|---|---|---|
| reversed  | 10 | 2.10 | 2 | 1 | 3 |
| shuffled  | 9  | 2.89 | 2 | 1 | 1 |
| partial   | 11 | 4.00 | 3 | 2 | 1 |
 
- **Reversed** divergences happen earliest (mean 2.1) — consistent with a
  conclusion-first ordering immediately disorienting the model.
- **Partial** divergences happen latest (mean 4.0) — consistent with the
  fixed first/last steps giving the model a stable anchor before things
  break down in the shuffled middle.
- **Reversed has the most "none" cases (3/10)** — i.e., annotator (not just
  model) sometimes couldn't identify one clear divergence point in reversed
  chains. Possible implication: reversed-order reasoning is confusing
  enough that "first divergence" may not be a well-defined concept for
  some of these cases — worth a limitations note.
- **Rule × Condition** (small n, flag as suggestive only):
  - partial: 4/11 INVENTED vs. reversed 1/10, shuffled 1/9 — partial
    condition may push the model toward *fabricating* a bridging step more
    than reversed/shuffled do. Plausible mechanism: fixed first/last steps
    give the model an anchor to fabricate a plausible-sounding link into,
    whereas reversed/shuffled have no stable anchor to bridge from.
    **Not statistically tested — n too small. State as a hypothesis for
    future work, not a finding.**
---
 
#### 6. Confidence data
 
- Self-rated annotation confidence: mean 4.20/5, N=35.
- Distribution skews high (18/35 rated confidence=5; none rated 1).
- **Caveat for paper**: high self-rated confidence in the human annotator
  does not by itself validate the annotations — worth noting as a
  limitation (annotator confidence is not the same as annotator accuracy,
  and there's no second annotator / inter-rater check yet).
---
 
#### 7. Significance test on positional distribution — RESOLVED (was open)
 
**Status update: this item is no longer open.** Ran a length-aware
significance test on the careful-annotation positional distribution
(n=30 numeric-divergence cases; chain lengths ranged 3–15 steps, e.g.
Counter({5:6, 6:6, 4:5, 7:5, 10:2, 15:2, 8:2, 9:1, 3:1}), max step=15).

**Method:** null hypothesis = divergence position uniform over 1..n_steps
*per individual case* (not a flat uniform over all observed step values —
that would be wrong given varying chain lengths). Expected counts per step
computed accordingly; bins with E<1 merged into an overflow bin before the
chi-square approximation, per standard practice with sparse expected counts.
Cross-checked the analytical result with a 10,000-draw Monte Carlo
permutation test, since several bins still had E<5 even after merging
(chi-square approximation flagged as weak in that regime).

**Result:**
- Chi-square(7) = 6.513, analytical p = 0.4812
- Monte Carlo p = 0.4178 (4,178/10,000 simulated draws met or exceeded the
  observed chi-square)
- **Fail to reject H0 at α=0.05 by both methods.**

**Interpretation — important, do not overstate Section 2's clustering
claim without this context:** the early-clustering pattern described in
Section 2 (co-modal Steps 1–2, 73% within first 3 steps) is visually
consistent but **not statistically distinguishable from a uniform
distribution** once chain-length is properly accounted for. This mirrors
H1's own result (a consistent directional pattern that doesn't survive
correction at this sample size) — report both as the same honest kind of
finding: real-looking pattern, underpowered to confirm at n=30/n=89.
**Do not claim statistically significant positional clustering in the
paper.** The taxonomy finding (Section 4, SELF-BREAK dominant) is
unaffected by this — it's a categorical result, not a positional one.
---
 
#### 8. Single-sentence summary to reuse in the paper draft — REVISED
 
> ~~Rule-based human annotation of 35 Stage-2 failure cases shows reasoning
> divergence concentrated in the first 3 steps (73%, co-modal at Steps 1–2),
> earlier than the position the model itself self-reports (mode Step 3)~~
 
**Revised per item 7 above — the original summary sentence overstates the
positional finding and should not be used as-is.** Corrected version:
 
> Rule-based human annotation of 35 Stage-2 failure cases shows an
> early-leaning divergence pattern (co-modal at Steps 1–2, 73% within the
> first 3 steps) that is directionally earlier than the position the model
> itself self-reports (mode Step 3) — but a length-aware significance test
> (chi-square + Monte Carlo, n=30) fails to distinguish this pattern from a
> uniform distribution, so the positional claim is reported as suggestive,
> not confirmed. The dominant failure mode (50%) is the model explicitly
> flagging an inconsistency and continuing anyway rather than correcting or
> misreading a step outright (0% MISUSE) — this categorical finding is
> unaffected by the positional test.
---

#### 9. Decision: dropped dedicated error-analysis case studies

Originally scoped (Day 33 roadmap) as individual deep-dive write-ups on
problem_ids #6 (Stephen's loan), #67 (Jenna's apples), #69 (Britany's
TikTok time). **Decision: dropped, not carried forward.**

Two of the three (#6, #69) already received qualitative write-ups above
under [10-07-2026], as part of Stage 1 validation — those descriptions
stand and are not being redone. #67 never received individual treatment
and will not now, either.

**Rationale:** the rule-based H2 taxonomy (item 4 above) already
generalizes what three hand-picked anecdotes were meant to illustrate,
across all 30 classified cases rather than 2–3 examples. Given the null
result on positional significance (item 7), effort is better spent on
writeup synthesis than additional qualitative case studies that wouldn't
change the statistical picture. If the paper draft later needs a specific
illustrative quote, pull from the existing 35-case annotation sheet rather
than reopening this as a separate task.

---

### [18-07-2026] — Day 36: cross-model model-selection saga + explicit H3 scoping decision

**Decision, logged explicitly:** the cross-model block is being split into
two separate, clearly labeled things in the paper, not conflated:

1. **Cross-family generalization check (what we're actually running):**
   does the step-order-sensitivity effect (H1's core finding) replicate
   with a different model family at comparable parameter scale? This is
   what ministral-8b-2512 (via Mistral AI's own API) vs. Llama-3.1-8B-
   Instant (via Groq) actually tests. Both are ~8B dense models. Report
   this as its own labeled result — "cross-family replication at matched
   scale" — not as an H3 test.

2. **H3 (smaller/weaker model shows a larger accuracy drop):** remains
   **open and untested**. None of the models trialed this week satisfy
   H3's premise of being genuinely smaller than Llama-3.1-8B-Instant.
   Explicitly flagged as a **future-work item**, not silently dropped or
   quietly reframed as answered by the ministral run.

**Why this distinction matters for the paper:** conflating "a different
model also shows the effect" with "H3 is confirmed" would be a scoping
error — same mistake in spirit as the Step-3 self-report issue from
Day 34 (a real-looking pattern getting overclaimed past what the data
actually supports). Keeping the two claims separate protects both: the
generalization result stands on its own merits, and H3 stays honestly
marked as not yet tested rather than either confirmed or refuted by proxy.

---

**Model-selection history (for Methods section):**

| Attempt | Model | Source | Outcome |
|---|---|---|---|
| 1 | Gemini 3.1 Flash-Lite | Google AI Studio (free tier) | Ran full Stage 1 (98/100, 98%) — too strong vs. Llama's 90%, defeats size-comparison premise. Abandoned. |
| 2 | google/gemma-2-2b-it | HF Inference Providers | Failed feasibility test — "not deployed by any Inference Provider." Never ran. |
| 3 | microsoft/Phi-3.5-mini-instruct | HF Inference Providers | Failed feasibility test — same reason. Never ran. |
| 4 | Qwen/Qwen2.5-3B-Instruct | HF Inference Providers (featherless-ai) | Passed feasibility test (single call, correct answer, clean parse). HF free-tier rate limits judged too restrictive for a full 189-call run. Abandoned before full Stage 1. |
| 5 | **ministral-8b-2512** | **Mistral AI API (direct)** | **Selected and run.** Stage 1 result: **95/100 (95%)**. Sits between Llama's 90% and the abandoned Gemini's 98%. Generous published rate limits (625,000 TPM, 3.13 RPS) comfortably cleared the run with a 0.5s sleep and zero errors. ~8B scale — comparable to, not smaller than, Llama-3.1-8B. |

**Implementation note:** the `mistralai` Python SDK raised an unresolvable
`ImportError` ("cannot import name 'Mistral' from 'mistralai' (unknown
location)") in this project's environment — likely a stray local
file/folder shadowing the package, or a stale v0.x install still on the
old import path. Rather than debug the SDK further, `utils_mistral.py`
calls Mistral's REST endpoint directly via `requests` (already manually
confirmed working). No functional difference in what's being tested — the
model, prompt, and scoring are identical either way.

---

**Day 36 Stage 1 result (confirmed):**

| Model | Accuracy |
|---|---|
| Llama-3.1-8B-Instant (Groq) | 90/100 (90%) |
| Gemini 3.1 Flash-Lite (abandoned) | 98/100 (98%) |
| **ministral-8b-2512 (Mistral AI)** | **95/100 (95%)** |

ministral-8b-2512 sits between the two — not lower than Llama's baseline,
which is worth noting plainly: at matched Stage-1 accuracy this close
(90% vs. 95%), a Stage 2 accuracy-drop comparison between the two models
should still be interpretable, since neither model is so much stronger at
the base task that a difference in perturbed-condition accuracy would just
reflect "one model is generally better at GSM8K" rather than "one model is
more sensitive to step-order disruption." Day 37 (eligibility filtering +
Stage 2 Baseline-control vs. Reversed) is the next actionable step.



> A genuinely smaller model was not available within this study's
> free/low-cost tier constraints at the time of writing: candidates in
> the 2–4B range either lacked any active Inference Provider deployment
> (gemma-2-2b-it, Phi-3.5-mini-instruct) or were feasible but rate-limited
> below what a full experimental run required (Qwen2.5-3B-Instruct via
> HF). H3 (smaller models show larger step-order sensitivity) therefore
> remains an open hypothesis, not evaluated in this study, and is flagged
> as a natural extension for future work with access to a broader range of
> model sizes.

## Open items / deferred decisions
- Exact answer-normalization edge cases (units, currency symbols) — deliberately minimal
  for now, will expand if a real mismatch is found
- Rate-limit backoff strategy specifics — not yet needed, current sleep(2.5) sufficient
- Days 36–40 cross-model block — not started