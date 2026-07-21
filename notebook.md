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

### [19-07-2026] — Day 37-38 Cross-model block: bypass finding, Mistral block closed early

**Result that looked like a finding, but isn't one:** ministral-8b-2512
scored 89/89 (100%) on BOTH baseline-control and Reversed conditions --
Robustness_tau = 1.000 exactly.

**Investigated via manual spot-check (09_spotcheck_mistral_bypass.py,
full 89-problem pass, not just the 8-case sample):**

| Category | n | % |
|---|---|---|
| Char-for-char IDENTICAL response, baseline vs. reversed | 84 | 94.4% |
| Near-identical (trivial formatting diff only, e.g. trailing `%`) | 1 | 1.1% |
| "Substantively different" text | 4 | 4.5% |

**The 4 "substantively different" cases do NOT rescue a robustness
claim.** On inspection (problems 15, 56, 97 and one more): baseline-control
produced a terse one-line answer, Reversed produced a full verbose
re-derivation -- but the verbose version is a FRESH solve of the original
question, not an engagement with the given (reversed) step content or
order. Different verbosity, same underlying bypass. Effectively **89/89
(100%) show no evidence the model used the provided step list as input at
all**, regardless of which of the two literal-similarity buckets they
landed in.

**Conclusion: tau=1.000 must NOT be reported as a robustness finding.**
The correct interpretation is a construct-validity limitation: Mistral is
apparently confident enough to solve GSM8K problems directly from the
question text, and does so regardless of what step list (correct-order or
reversed) it's handed in the Stage 2 prompt. The prompt's "use these exact
steps, do not reorder" instruction does not behaviorally constrain this
model the way it constrains Llama-3.1-8B-Instant (90% Stage 1 accuracy,
vs. Mistral's 95% -- only a 5-point gap in raw capability, but apparently
enough to cross a threshold where the model stops needing the scaffolded
input).

**McNemar's test is not being run for this pair.** With zero discordant
pairs (b=0, c=0 -- baseline-control and Reversed agree on literally every
problem), the test is undefined / trivially p=1.0. This is not "no
significant difference from reordering" in the H1 sense -- it's a symptom
of the bypass, and reporting a p-value here without this context would be
misleading.

**Decision: closing the Mistral cross-model block here, not expanding to
Shuffled + Partial.** Running the remaining two conditions would almost
certainly reproduce the identical bypass pattern (both are equally
"not the correct order" from the model's perspective) -- more API calls
on a contaminated condition adds no information. Day 38/39's originally
planned scope (decide whether to expand conditions based on directional
match with Llama) is superseded by this finding: there is no meaningful
directional comparison to make when the denominator condition itself
isn't testing what it's supposed to test.

**Reframed contribution to the paper:** this becomes a documented
methodological limitation/finding in its own right, not a null result to
bury. Candidate framing for Discussion/Limitations:

> A Stage-2 prompt design that scaffolds reasoning via a provided step
> list implicitly assumes the model needs and uses that scaffold. For a
> sufficiently capable model, this assumption can silently fail: rather
> than following (or being disrupted by) the given step order, the model
> may bypass the provided steps entirely and re-derive the answer from the
> original question, producing near-identical outputs regardless of step
> order or content. This was observed in 89/89 (100%) of eligible cases for
> ministral-8b-2512, evidenced by character-for-character identical
> responses between the baseline-control (correct order) and Reversed
> conditions in the large majority of cases, and by verbosity-only
> differences (not content-order engagement) in the remainder. This
### suggests that step-order-sensitivity experiments of this design are only
> informative for models operating below their own solve-from-scratch
> confidence threshold on the target task, and that apparent perfect
> robustness (tau=1.000) should be verified qualitatively before being
> reported as a substantive finding -- a purely quantitative accuracy
> comparison would have missed this entirely.

**Open item flagged for future work, not resolved here:** a cleaner
Stage 2 prompt design might mitigate bypass (e.g. withholding the original
question and showing ONLY the step list, forcing the model to work
strictly from the given steps rather than allowing a fallback re-derivation
path) -- worth considering for any follow-up study, but out of scope for
the current pipeline/timeline.


---

### [Follow-up] — Steps-only prompt test: bypass persists, block closed

**Tested whether withholding the original question (forcing the model to
work ONLY from the given step list) would eliminate the bypass, before
committing to a full re-run.** Ran on a random 15-problem sample
(10_test_noquestion_prompt_mistral.py), both baseline-control and Reversed
conditions, using a hardened prompt that explicitly states the question is
not provided and instructs the model not to reinterpret or re-derive
outside the given steps.

**Result: 14/15 (93.3%) still produced character-identical responses
between baseline-control and Reversed.** The fix did not work. This
narrows down the mechanism: it's not simply that the model has the
original question as a fallback -- ministral-8b-2512 is apparently capable
of reconstructing enough problem context from the step list's own content
(the numbers and phrasing embedded in the steps themselves) to solve the
problem independently of what order those steps are shown in. Removing
the question removed one bypass route; the model used another.

**One exception, flagged but not over-interpreted (n=1):** problem 71 --
baseline-control answered incorrectly (0.25), Reversed answered correctly
(25). This is the only case across both the original 89-problem run and
this 15-problem follow-up where order appeared to matter at all for this
model. With a single data point, this is consistent with ordinary noise
and should not be read as evidence of partial order-sensitivity without
further investigation (out of scope for now, given the block is closing).

**Decision: cross-model block (ministral-8b-2512) is closed.** No further
prompt-hardening attempts planned -- diminishing returns, and the finding
itself (capable models bypass artificial reasoning-order manipulations via
multiple independent routes) is now well-evidenced across two separate
tests. This strengthens, not weakens, the paper's methodological
contribution: it's not a one-off prompt quirk, it's a robust bypass
behavior that survived a deliberate attempt to close the most obvious
escape hatch.

**Updated framing for paper's Discussion/Limitations section:**

> Two independent attempts to constrain a capable model
> (ministral-8b-2512) to genuinely engage with a provided, potentially
> reordered reasoning chain both failed to prevent bypass: providing the
> original question alongside the steps, and withholding the question
> entirely, both resulted in over 90% character-identical responses
> between correct-order and reversed-order conditions. This suggests the
> bypass is not attributable to a single fixable prompt weakness, but to
> the model's general capability exceeding what the experimental
> manipulation requires it to rely on -- it can reconstruct sufficient
> problem context from the steps' own content regardless of the
> scaffolding's completeness. Step-order-sensitivity designs of this kind
> are therefore only informative below some model-specific capability
> threshold, and confirming genuine engagement (not just comparing
> accuracy numbers) is a necessary check, not an optional one, before
> reporting a robustness result.

**Status: H3 remains untested (unchanged). Cross-family generalization
check is now this two-part bypass finding, not an accuracy-drop
comparison, and is closed as of this entry.**

---

### [Final] — Third prompt attempt: bypass broken, but replaced by a positional-selection artifact. Thread closed.

**Attempt 3: stepwise-trace, steps-only prompt** (utils_mistral_stepwise.py,
10_recheck_mistral_stepwise.py). Forces the model to emit one "Step N:"
line per given step, in the given order, before stating a final answer --
designed to make silent full-response bypass much harder to pull off
undetected than either of the first two attempts.

**Ran on a fresh 8-problem sample (seed=99, non-degenerate only).
Full raw response text saved and manually read for all 8 cases** (not
just the summary scores) -- per the project's standing rule that an
automatic screen is never trusted without a manual read alongside it.

**Result: baseline-control 8/8 correct, Reversed 3/8 correct.** On the
surface this looks like the order-sensitivity signal the experiment was
designed to detect. Reading the actual response text tells a different
story.

**Mechanism identified (confirmed across all 8 cases, no exceptions):**
The model's "Step N:" lines are not performing computation -- they are
transcribing the numeric conclusion already embedded in each given step's
own text, in whatever order the steps are presented. This is possible
because the Stage 1-generated steps already contain their answers inline
(e.g. "...Height after second bounce = 48 feet x (2/3) = 32 feet"), so
there is nothing left to compute -- only to copy forward in sequence.

Since the Reversed condition always places the TRUE final step (the one
containing the actually-correct answer) at position 1, whether the model
gets Reversed correct or wrong reduces to a single question: does it treat
its own FIRST transcribed line or its LAST transcribed line as "the final
answer"? Checked across all 8 cases:

| Problem | Model's answer taken from | Correct? |
|---|---|---|
| 59 | last | wrong |
| 56 | last | wrong |
| 29 | last | wrong |
| 87 | last (despite Step 1 line containing the right number) | wrong |
| 19 | last | wrong |
| 25 | first | correct |
| 34 | first | correct |
| 36 | first | correct |

Zero exceptions: every wrong answer took the last transcribed line, every
correct answer took the first. This is a clean, deterministic-looking
split with no counterexamples in this sample, and it has nothing to do
with multi-step reasoning dependency -- it is a positional artifact in
which line the model treats as "the final answer" when producing a
transcript-shaped response.

**Conclusion: bypass (in the sense of "ignoring the given steps entirely")
IS broken by this prompt design** -- the model is now genuinely
transcribing content from each given step in the given order, which
attempts 1 and 2 did not achieve. **However, the resulting accuracy number
does not measure reasoning-order sensitivity.** It measures an unrelated,
seemingly arbitrary first-vs-last selection instability. Scaling this to
the full 89-problem set would produce a precise, reproducible number, but
that number would answer "how often does this model pick the first vs.
last transcribed line as its final answer" -- not "is this model's
reasoning disrupted by step-order permutation" -- and would NOT be
comparable to Llama's Reversed accuracy under the original H1 design.

**Decision: closing the Mistral thread here. No fourth prompt attempt.**
Three attempts have now each surfaced a distinct, well-evidenced failure
mode:
  1. With-question prompt -> full re-derivation bypass (94.4% identical
     responses, baseline-control vs. Reversed)
  2. No-question, single-answer prompt -> bypass persists via
     self-reconstruction from the steps' own embedded numbers (93.3%
     identical on a fresh sample)
  3. No-question, stepwise-trace prompt -> bypass broken (confirmed
     genuine per-step transcription), but replaced by a positional
     selection artifact unrelated to order-sensitivity

Continuing to iterate on prompt design risks an open-ended search for "the
prompt that finally produces a clean, comparable number" -- a different
flavor of the scope creep already correctly cut once before (Day 33's
case-study decision). Three independently-evidenced failure modes is a
stronger, more honest contribution than a fourth attempt chasing a clean
result.

**Framing for the paper (Discussion/Limitations section) -- supersedes
the two-attempt framing logged earlier:**

> Three independent attempts to elicit genuine step-order engagement from
> a capable model (ministral-8b-2512) each surfaced a distinct failure
> mode rather than a clean measurement. Providing the original question
> alongside the steps allowed full bypass via re-derivation from the
> question (89/89 cases indistinguishable from a re-solve). Withholding
> the question did not eliminate bypass; the model instead reconstructed
> sufficient context from the steps' own embedded numeric content,
> independent of their order (14/15 cases in a follow-up sample). Forcing
> an explicit per-step transcription broke this second bypass -- the model
> genuinely transcribed each given step's content in the order
> presented -- but surfaced a third confound: because GSM8K steps
### typically embed their own computed values inline, transcription alone
> (without independent computation) suffices to reproduce the correct
> answer when steps are in the correct order, and the resulting accuracy
> under reordering was governed by a positional artifact (a first-vs-last
> selection instability when identifying "the final answer" from the
> transcript) rather than by genuine sensitivity to reasoning dependency.
> This suggests that step-order-sensitivity experiments relying on models
> to freely reproduce or transcribe reasoning are vulnerable to multiple,
> independent failure modes as model capability increases, and that
> verifying the actual mechanism behind an accuracy number -- not merely
> the number itself -- is necessary at each stage of such a design.

**Status: H3 remains untested (unchanged throughout this entire
investigation). Cross-family generalization check for ministral-8b-2512
is now this three-attempt bypass-mechanism finding, not an accuracy-drop
comparison of any kind, and is CLOSED as of this entry. No further model
attempts planned for the cross-model block absent a specific new idea
that avoids all three identified failure modes simultaneously (e.g.
steps that require withheld intermediate values not stated inline --
out of scope for the current timeline).**


### [19-07-2026] — H3 cross-size comparison: qwen/qwen3.6-27b (trial run)

**Decision:** The original H3 called for a *smaller* model than Llama-3.1-8B.
No sub-8B model is currently available on Groq's free tier.
Instead, running qwen/qwen3.6-27b (~27B) as a **cross-size check in the
other direction** (larger, not smaller).  Framing: does a stronger model show
more or less robustness to step-order permutation?  This is NOT a H3 test --
it is a **cross-size directional check** and will be labeled as such in the
paper.

**Model: qwen/qwen3.6-27b** (preview, Groq, ~27B dense parameters)

**Implementation note -- reasoning_effort fix (critical):**
Qwen 3.6 27B is a hybrid thinking/non-thinking model.  Without explicit
configuration, it operates in thinking mode and outputs only a `<think>...</think>`
block, never emitting the "Final answer: <number>" line the parser expects.
Fix: `reasoning_effort="none"` in every API call disables the think block and
forces direct formatted output.  Temperature set to 0.7 per Groq's
non-thinking mode best-practice (temperature=0.0 produces degenerate output in
non-thinking mode for this model family).  The key consistency note: Stage 1
and Stage 2 calls both use the same temperature/reasoning_effort pair, so the
comparison is internally consistent even if not identical to the Llama 0.0
setting.

**Feasibility test result (1 problem before full run):**
- Problem 0 (Maggie's oven), ground truth 500
- Parsed 8 steps cleanly, Final answer = 500, correct = True
- No `<think>` block in output after fix

**Pipeline scripts created:**
- `scripts/utils_h3.py` — model utilities (MODEL_ID = qwen/qwen3.6-27b)
- `scripts/11_h3_generate_baseline.py` — Stage 1 baseline (100 problems)
- `scripts/12_h3_permute_conditions.py` — generate reversed/shuffled/partial condition files
- `scripts/13_h3_run_conditions.py` — Stage 2: run 3 conditions
- `scripts/13b_h3_run_baseline_control.py` — Stage 2: baseline-control (unpermuted steps)
- `scripts/14_h3_analysis.py` — accuracy, Robustness_tau, McNemar, cross-model table

**Status: Stage 1 run in progress.** Results will be added here once complete.

What we actually got from the run

| Condition | Records | Clean calls | Error calls | Correct |
|-----------|--------:|------------:|------------:|---------:|
| Baseline-control | 91 | 91 | 0 | 89/91 (97.8%) |
| Reversed | 91 | 90 | 1 | 89/91 (97.8%) |
| Shuffled | 91 | 48 | 43 | 48/48 of clean |
| Partial | 57 | 3 | 54 | 3/3 of clean |

The shuffled and partial conditions got hammered by rate limits (preview model on free tier). But the baseline-control and reversed are complete and clean.

### [20-07-2026] - [Cross-size check] — qwen/qwen3.6-27b: a third, distinct mechanism (neither bypass nor clean robustness)

**Setup:** qwen/qwen3.6-27b via Groq, framed explicitly as a cross-size
directional check (larger than Llama-3.1-8B, not the smaller model H3
originally called for -- no sub-8B model is available on Groq's free
tier). NOT an H3 test; labeled as such throughout.
 
**Implementation note:** Qwen 3.6 27B is a hybrid thinking/non-thinking
model. Left unconfigured, it emits only a `<think>` block and never a
parseable final answer. Fixed via `reasoning_effort="none"` (disables the
think block) and `temperature=0.7` (0.0 produced degenerate output in
non-thinking mode, per the model family's own guidance). Both Stage 1 and
Stage 2 calls use this same setting, keeping the comparison internally
consistent.
 
**Aggregate result (baseline-control vs. Reversed, n=91 matched
problems):** 97.8% both conditions, Robustness_tau = 1.000 -- superficially
identical to the ministral-8b-2512 result that turned out to be bypass.
 
**This did NOT turn out to be bypass.** Ran the same validation used for
Mistral (15_spotcheck_qwen_bypass.py): full character-similarity pass
across all 91 matched problems, plus manual reading of an 8-case sample.
 
| Category | n | % |
|---|---|---|
| IDENTICAL (exact char match) | 0 | 0.0% |
| NEAR-IDENTICAL (>=90% similarity) | 8 | 8.8% |
| DIFFERENT (<90% similarity) | 83 | 91.2% |
 
This is the opposite pattern from Mistral (94.4% identical/near-identical
there vs. 8.8% here). Ruling out simple bypass required reading the actual
text, not just this number -- consistent with the standing project rule
that surface differentness doesn't by itself prove genuine engagement
(the Mistral stepwise-prompt attempt showed different-looking responses
can still hide an unrelated artifact).
 
**Manual read of 8 sampled response pairs revealed TWO distinct
sub-mechanisms, neither of them bypass:**
 
**Mechanism A -- genuine dependency reconstruction (5/8 sampled cases:
39, 43, 34, 94, 20).** The model recognizes the TRUE logical dependency
order of the steps regardless of presentation order, and silently
resequences its own computation to match that true order, while labeling
each block with the position number it held in the (reversed) list shown.
Problem 39 is the clearest example: presented in order [reward-addition,
total-salary, weeks, new-salary, raise, initial-salary], the model computed
in the CORRECT dependency order (initial-salary -> raise -> new-salary ->
weeks -> total-salary -> reward-addition), labeling each block "Step 6,"
"Step 5," etc. to match its position in the presented list. This reflects
real content-level engagement, not a fresh from-scratch solve and not a
literal-order transcription.
 
**Mechanism B -- pre-baked value extraction (3/8 sampled cases: 97, 25,
92).** The model processes steps in literal presented order and succeeds,
but succeeds trivially: each Stage-1-generated GSM8K step already states
its own computed numeric result inline (a known property of this
project's CoT generation format), so "using" a step in presented order
requires reading off an already-present number, not resolving any real
dependency.
 
**The confound that limits how strongly this can be reported:** Reversed
always places the ORIGINAL FINAL step -- which in GSM8K CoT chains
generated by this pipeline almost always states the answer directly -- at
position 1 of the presented list. A model with strong reading
comprehension can therefore get Reversed correct by locating and reading
position 1's stated conclusion alone, regardless of whether it does real
dependency reasoning (Mechanism A) or trivial extraction (Mechanism B).
Problem 39 is reassuring on this point -- the model did a full 6-step
reconciliation rather than taking the position-1 shortcut that was sitting
right there -- but this does not rule out the shortcut being taken in
other, unsampled cases. This is structurally the same underlying design
weakness that undid Mistral's third (stepwise) prompt attempt -- steps
that embed their own computed values -- manifesting differently here:
Mistral exploited it via blind transcription; Qwen's high accuracy is
consistent with genuine reasoning, trivial extraction, or some mix, and
the current data cannot cleanly separate these from each other.
 
**Conclusion: this is a third, distinct mechanism from Mistral's clean
bypass, and it should NOT be reported as either (a) "confirmed genuine
robustness" or (b) "the same bypass pattern as Mistral."** Both of those
framings would overclaim past what an 8-case manual sample can support.
The honest framing is a middle position: Qwen's Reversed accuracy is
CONSISTENT WITH genuine order-robustness at this model scale, but not
cleanly separable from a step-self-containment confound inherent to this
project's Stage-1 CoT generation format.
 
**Decision: closing the cross-size check here on the current sample.**
Same reasoning as the Mistral closure -- avoiding an open-ended chase for
a "clean" number. However, unlike Mistral (where three attempts
converged on "this model bypasses, full stop"), Qwen's result is
genuinely more interesting and less settled, and is worth strengthening
if reasonably cheap to do so (see recommendations below) before finalizing
how it's reported in the paper.
 
**Shuffled and Partial conditions were rate-limited (48/91 and 3/57 clean
calls respectively) and are NOT being chased to completion** -- consistent
with the decision not to spend further budget completing conditions once
the core mechanism question (bypass vs. not) is already answered from
Reversed alone.
 
**Status: H3 remains untested (unchanged). Cross-size check for
qwen/qwen3.6-27b is a third, distinct finding -- neither bypass (Mistral)
nor clean robustness -- and is provisionally closed pending the
strengthening steps below.**
 
---
 
### What would make this finding more robust (recommendations, not yet done)
 
#### the middle-swap test

Design: take each eligible problem's original, correct-order steps and swap only the final step (which states the answer) with the step sitting at the middle index — everything else stays in its natural, logical order. This isolates exactly one variable: does moving the answer-bearing step away from its natural end position (without introducing any other disruption) hurt accuracy? If accuracy holds, that's real evidence against "the model just reads whatever's at the position where the answer usually sits." If it collapses, that's evidence for the shortcut concern.

#### What to do with Shuffled/Partial once rate limits clear

they can each serve a distinct, useful role in this same investigation, essentially for free once the data exists.

1. Shuffled — bin by where the true final step landed. Unlike Reversed (which always puts it at position 1) and unlike this new middle-swap test (which always puts it at the middle), Shuffled randomizes the final step's landing position across the whole range. Once you have clean Shuffled data, you can group per-problem correctness by "what position did the true final/answer step land at in this problem's shuffle" and check whether accuracy correlates with that position. This is essentially a natural, higher-powered version of the middle-swap test, using data you were going to collect anyway — no extra API calls beyond what's already planned. I can write that binning/analysis script now if you want, so it's ready the moment Shuffled clears.

2. Partial — actually a cleaner comparison than either. By design, Partial keeps the first and last step fixed — meaning the true final step never moves in this condition. So Partial is structurally immune to the position-1 confound entirely: any accuracy drop or hold in Partial reflects purely "does scrambling the middle, non-answer-bearing steps matter," with zero interference from where the answer sits. If Partial accuracy comes back high, that's a genuinely clean piece of evidence for order-insensitivity on the non-critical steps — no confound to explain away.

### [21-07-2026] — Parser fix applied in-place; H1 stats reconfirmed; H2 integrity check flagged

**Covers work not logged on 20-07-2026** (the normalize_answer fix was
identified and validated read-only on 20-07-2026 via
17_rescore_with_fixed_normalizer.py; applying it in-place and
re-running downstream analysis happened today).

**1. Applied the normalizer fix in place (18_apply_normalizer_fix.py).**
17_rescore_with_fixed_normalizer.py was read-only by design -- it reported
old-vs-new accuracy but never wrote anything back to disk. Today's script
actually rewrites the stored `correct` field in every affected results
file (Llama trial 1/2 Stage 2 + baseline-control, Qwen 27B
baseline-control + Stage 2), with automatic `.bak` backups before each
overwrite.

**2. Re-ran 04_analysis.py with corrected data. H1's conclusion is
UNCHANGED.** Corrected baseline-control: 86/89 (96.63%) both trials
(up from 85/89). No McNemar comparison survives Bonferroni correction in
either trial -- tightest p-value is now 0.0215 (Baseline-control vs.
Partial non-degenerate, trial 1; previously 0.0391), still far above the
corrected threshold of 0.0083. Direction of effect (every perturbed
condition scores below baseline-control) holds in both trials, matching
the pre-fix finding. **This is the final, corrected version of H1's
numbers -- use these, not the pre-fix numbers, in the paper.**

**Note on why one comparison's p-value didn't move at all:** problem 5
(the recurring "62.00" vs "62" formatting case) flipped wrong->correct in
baseline-control, reversed, AND shuffled simultaneously for trial 1 --
since it flipped in both columns of the Baseline-vs-Reversed comparison,
it moved from "both wrong" to "both correct" without ever becoming a
discordant pair, so that specific p-value (0.0352) is byte-identical
before and after the fix. Baseline-vs-Partial DID shift, because problem
5's Partial-condition flip only occurred in trial 2, not trial 1 --
creating a new discordant pair in trial 1 specifically. Worth this exact
explanation if anyone asks why the correction affected some comparisons
and not others.

**3. Cross-trial comparison also updated.** Partial's trial1-vs-trial2
flip list now correctly includes problem 5 in the wrong->correct set
(previously showed 3 flips [20, 42, 68]; now shows 4 [5, 20, 42, 68]),
consistent with problem 5's Partial-condition fix landing only in trial 2.

**4. FLAGGED, NOT YET RESOLVED -- possible H2 integrity issue.**
05_h2_self_report.py pulled its 35 wrong-answer cases specifically from
`results/stage2_results_v2_trial1.jsonl` -- the exact file just corrected.
If problem 5 (reversed) and/or problem 5 (shuffled) from trial 1 were
among the original 35 "wrong" records used for the blind manual
annotation and rule-based taxonomy (Days 33-34), those records are no
longer wrong post-fix and should be REMOVED from the H2 analysis pool --
which would shift n=35 downward, and potentially affect:
  - the Step 1/2 co-modal positional finding
  - the SELF-BREAK (50%) failure-mode taxonomy percentages
  - the chi-square/Monte Carlo null result on positional clustering
**Action required before finalizing H2's numbers for the paper:** check
whether problem_id=5 (condition reversed or shuffled) appears in
`results/h2_self_report_trial1.jsonl` or
`data/h2_manual_annotation_sheet.jsonl`. If it does, remove that record
and recompute H2's distributions/percentages/chi-square with n=33 or
n=34 (however many records remain). If it does not appear (e.g. it wasn't
sampled among the wrong answers for some other reason), H2 is unaffected
and this item can be closed as a non-issue.

**Status: H1 numbers are now final and corrected. H2 numbers are
PROVISIONAL pending the integrity check above -- do not finalize the H2
section of the paper until this is resolved.**

### Parser fix applied in-place; H1 stats reconfirmed; H2 integrity check flagged
 
**Covers work not logged on 20-07-2026** (the normalize_answer fix was
identified and validated read-only on 20-07-2026 via
17_rescore_with_fixed_normalizer.py; applying it in-place and
re-running downstream analysis happened today).
 
**1. Applied the normalizer fix in place (18_apply_normalizer_fix.py).**
17_rescore_with_fixed_normalizer.py was read-only by design -- it reported
old-vs-new accuracy but never wrote anything back to disk. Today's script
actually rewrites the stored `correct` field in every affected results
file (Llama trial 1/2 Stage 2 + baseline-control, Qwen 27B
baseline-control + Stage 2), with automatic `.bak` backups before each
overwrite.
 
**2. Re-ran 04_analysis.py with corrected data. H1's conclusion is
UNCHANGED.** Corrected baseline-control: 86/89 (96.63%) both trials
(up from 85/89). No McNemar comparison survives Bonferroni correction in
either trial -- tightest p-value is now 0.0215 (Baseline-control vs.
Partial non-degenerate, trial 1; previously 0.0391), still far above the
corrected threshold of 0.0083. Direction of effect (every perturbed
condition scores below baseline-control) holds in both trials, matching
the pre-fix finding. **This is the final, corrected version of H1's
numbers -- use these, not the pre-fix numbers, in the paper.**
 
**Note on why one comparison's p-value didn't move at all:** problem 5
(the recurring "62.00" vs "62" formatting case) flipped wrong->correct in
baseline-control, reversed, AND shuffled simultaneously for trial 1 --
since it flipped in both columns of the Baseline-vs-Reversed comparison,
it moved from "both wrong" to "both correct" without ever becoming a
discordant pair, so that specific p-value (0.0352) is byte-identical
before and after the fix. Baseline-vs-Partial DID shift, because problem
5's Partial-condition flip only occurred in trial 2, not trial 1 --
creating a new discordant pair in trial 1 specifically. Worth this exact
explanation if anyone asks why the correction affected some comparisons
and not others.
 
**3. Cross-trial comparison also updated.** Partial's trial1-vs-trial2
flip list now correctly includes problem 5 in the wrong->correct set
(previously showed 3 flips [20, 42, 68]; now shows 4 [5, 20, 42, 68]),
consistent with problem 5's Partial-condition fix landing only in trial 2.
 
**4. RESOLVED — H2 integrity check.** Checked both files directly:
  - `results/h2_self_report_trial1.jsonl`: problem_id=5 DOES appear, for
    both `condition: reversed` and `condition: shuffled` (expected -- this
    file was built from ALL wrong Stage-2-trial-1 records at the time,
    before the normalizer fix).
  - `data/h2_manual_annotation_sheet.jsonl`: problem_id=5 does NOT appear
    (the 35-case sample happened not to draw it).
**Conclusion: the CANONICAL H2 numbers are unaffected.**
`08_h2_careful_analysis.py` -- the script designated canonical for the
paper -- reads from `data/h2_manual_annotation_sheet.jsonl`, not the
self-report file. Since problem 5 was never in that 35-case sample, none
of the following require any correction:
  - the co-modal Step 1/2 positional finding (8/30 each)
  - the SELF-BREAK (50%) failure-mode taxonomy
  - the careful-vs-self-report agreement rates (60% combined)
  - the chi-square/Monte Carlo null result on positional clustering
**Minor, non-headline staleness (optional cleanup, not urgent):** the
VERY FIRST self-report-only distribution reported on Day 32 (before the
careful-annotation pass existed) -- "divergence clusters at Step 3,
11/35" via `analyze_h2_distribution.py` -- technically includes problem
5's two now-invalid "wrong" entries. This number was already superseded
in the writeup by the careful-annotation contradiction finding and is not
used as a headline result, so this is a footnote-level correction, not a
blocking one. If revisited: recompute against n=33 (35 minus the 2 now-
invalid problem-5 entries) for full internal consistency.
 
**Status: BOTH H1 and H2 numbers are now final.** No further corrections
pending.
 
---
 
### What's next
 
1. **(Optional, low priority)** Recompute the Day-32 self-report-only
   distribution against n=33, for internal consistency -- not blocking,
   since this number isn't used as a headline finding in the paper
2. Update `readme.md`'s paper-ready summary sentences (Section 8 of the
   17-07-2026 entry) to cite the corrected H1 numbers (86/89 baseline-
   control, tightest p=0.0215) instead of the pre-fix ones
3. Resume the Days 36-40 cross-model/cross-size thread status: Mistral
   and Qwen 27B blocks are both closed (see their respective readme
   entries); no further model attempts planned
4. **Both H1 and H2 are now fully finalized. Begin paper draft assembly**
   (originally scoped as Day 40) -- abstract, intro, related work, and
   methods are largely already written via the README log; results and
   discussion sections can now be drafted against final, confirmed numbers
---

### What's next

1. **Resolve the H2 integrity check above** (highest priority --
   determines whether any already-reported H2 numbers need revision)
2. If H2 needs revision: re-run the relevant portions of
   `08_h2_careful_analysis.py` and the chi-square test with the corrected
   pool size
3. Once both H1 and H2 are confirmed final: update `readme.md`'s
   paper-ready summary sentences (Section 8 of the 17-07-2026 entry) to
   cite the corrected numbers, not the pre-fix ones
4. Resume the Days 36-40 cross-model/cross-size thread status: Mistral
   and Qwen 27B blocks are both closed (see their respective readme
   entries); no further model attempts planned
5. Begin paper draft assembly (originally scoped as Day 40) now that H1
   is fully finalized and H2 is one integrity check away from the same

## Open items / deferred decisions

- Exact answer-normalization edge cases (units, currency symbols) — deliberately minimal- done updated the normalize_answer
  for now, will expand if a real mismatch is found
- Rate-limit backoff strategy specifics — not yet needed, current sleep(2.5) sufficient
---