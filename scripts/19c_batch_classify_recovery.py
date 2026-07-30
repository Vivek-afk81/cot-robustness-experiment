"""
19c_batch_classify_recovery.py

Non-interactive contextual classifier for recovery candidates.

Classification logic (calibrated against known ground truth):
  - Qwen PID 10 reversed = FALSE POS (narration, no corrective action)
  - Qwen PID 87 reversed = FALSE POS (observes order difference, no action)
  - Qwen PID 23 shuffled = GENUINE (notes values not yet calculated)
  - Qwen PID 57 shuffled = FALSE POS (narrates partial calc, no error flag)
  - Qwen PID 68 shuffled = GENUINE (explicitly maps dependency chain)
  - Llama 13/13 sampled = all GENUINE (explicit skip/ignore/reorder)

The key distinction: GENUINE requires explicit CORRECTIVE ACTION language
(ignore, skip, come back, re-derive, start from a different step, mark
as incorrect) in the same sentence/paragraph as the keyword hit.
Mere OBSERVATION of disorder ("this step seems to reference...",
"the logical flow suggests...") without subsequent corrective action
is FALSE POSITIVE.

Reads:  results/recovery_candidates_all.jsonl
Writes: results/recovery_candidates_classified.jsonl
"""

import json
import re

INPUT_PATH = "results/recovery_candidates_all.jsonl"
OUTPUT_PATH = "results/recovery_candidates_classified.jsonl"

# ── Corrective ACTION patterns (the model takes/announces a workaround) ──
ACTION_PATTERNS = [
    r"\b(will |we |I |let me |need to |should )?(ignore|skip|discard|disregard) (this |the |it|step)",
    r"\bcome back (to |later)",
    r"\bre-?evaluate\b",
    r"\bre-?derive\b",
    r"\bre-?calculate\b",
    r"\blet me (re-?check|re-?verify|reconsider|correct)\b",
    r"\bcorrect(ing)? the steps\b",
    r"\bneed to (ignore|skip|reorder|correct|re-?evaluate|start)\b",
    r"\bstart (from|with) (step |the )\d",
    r"\bstep \d+ is (actually )?incorrect\b",
    r"\b(I made|there is) (a |an )?(error|mistake)\b",
    r"\bunnecessary for (this|the) problem\b",
    r"\bwe (don'?t|do not) (have|know) (this|the|that|a) (value|number|result)",
    r"\bnot (yet |been )?(calculated|determined|established|defined|given|known)\b",
    r"\bwe (need|should|must|will) (start|begin) (from|with|at) (step|the)",
    r"\b(does ?n[o']?t|doesn[']t|does not) (make sense|seem right|follow|apply)\b.*(so|therefore|thus|hence|we|I)",
    r"\b(seems?|appears?) (to be )?(out of order|incorrect|wrong|misplaced)",
    r"\b(contradicts?|conflicts? with)\b",
    # Llama's signature patterns from the 13-case sample
    r"\bwill skip (it|this|step)",
    r"\bwe('ll| will) come back\b",
    r"\bI will ignore\b",
    r"\bwe need to ignore\b",
    r"\bseems to be a conclusion\b.*(ignore|skip|start|so we)",
]
ACTION_RE = re.compile("|".join(ACTION_PATTERNS), re.IGNORECASE)

# ── Narration-ONLY patterns (observation without action = false positive) ──
# If ONLY these appear and no ACTION pattern, classify as false positive
NARRATION_ONLY = [
    r"\b(this step )?(seems?|appears?) (to |like )",
    r"\bthe logical flow\b",
    r"\beven though\b.*\border\b",
    r"\blooking ahead\b",
    r"\bnote(s|d)?:? (that |this )?",
    r"\bwe proceed\b",  # proceeds without flagging = not recovery
]
NARRATION_RE = re.compile("|".join(NARRATION_ONLY), re.IGNORECASE)


def classify(record):
    raw = record.get("raw_response", "") or ""

    # Check for corrective action language
    action_hits = ACTION_RE.findall(raw)

    if action_hits:
        reasons = []
        for h in action_hits[:3]:
            if isinstance(h, str):
                reasons.append(h)
            elif isinstance(h, tuple):
                non_empty = [s for s in h if s]
                reasons.append(non_empty[0] if non_empty else "action match")
            else:
                reasons.append("action match")
        return True, reasons

    # No action language found -- check if keywords are just narration
    narration_hits = NARRATION_RE.findall(raw)
    keywords = record.get("matched_keywords", [])
    keyword_strs = set(k.lower().strip() if isinstance(k, str) else "" for k in keywords)

    # If only weak keywords ("actually", "wait") and no action context
    weak_only = keyword_strs <= {"actually", "wait", ""}
    if weak_only and not action_hits:
        return False, ["weak keyword, no corrective action"]

    # If narration present but no action
    if narration_hits and not action_hits:
        return False, ["narration without corrective action"]

    # Default for remaining cases: mark unclear for manual review
    return "unclear", ["cannot determine automatically"]


def run():
    with open(INPUT_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} candidates from {INPUT_PATH}")

    stats = {"genuine": 0, "false_pos": 0, "unclear": 0}
    model_stats = {}

    for r in records:
        verdict, reasons = classify(r)
        r["MANUAL_is_genuine_recovery"] = verdict
        r["MANUAL_notes"] = "; ".join(str(x) for x in reasons) if reasons else None

        model = r["model"]
        if model not in model_stats:
            model_stats[model] = {"genuine": 0, "false_pos": 0, "unclear": 0}

        if verdict is True:
            stats["genuine"] += 1
            model_stats[model]["genuine"] += 1
        elif verdict is False:
            stats["false_pos"] += 1
            model_stats[model]["false_pos"] += 1
        else:
            stats["unclear"] += 1
            model_stats[model]["unclear"] += 1

    # Write classified output
    with open(OUTPUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWritten to: {OUTPUT_PATH}")

    # Summary
    total = len(records)
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total candidates:    {total}")
    print(f"Genuine recoveries:  {stats['genuine']} ({stats['genuine']/total:.1%})")
    print(f"False positives:     {stats['false_pos']} ({stats['false_pos']/total:.1%})")
    print(f"Unclear:             {stats['unclear']} ({stats['unclear']/total:.1%})")

    print(f"\n--- Per-Model Breakdown ---")
    for model, ms in model_stats.items():
        mtotal = ms["genuine"] + ms["false_pos"] + ms["unclear"]
        print(f"\n{model} ({mtotal} candidates):")
        print(f"  Genuine:    {ms['genuine']}")
        print(f"  False pos:  {ms['false_pos']}")
        print(f"  Unclear:    {ms['unclear']}")

    # Validation against known ground truth
    qwen_records = [r for r in records if r["model"] == "Qwen-27B"]
    print(f"\n--- Qwen Validation (known: PID 23,68=genuine; PID 10,87,57=false pos) ---")
    for r in qwen_records:
        pid = r["problem_id"]
        cond = r["condition"]
        v = r["MANUAL_is_genuine_recovery"]
        label = "GENUINE" if v is True else ("FALSE POS" if v is False else "UNCLEAR")
        print(f"  PID={pid} ({cond}): {label} -- {r.get('MANUAL_notes', '')}")

    # Conditional recovery rate (Llama T1 only)
    llama_t1_genuine = model_stats.get("Llama-8B trial 1", {}).get("genuine", 0)
    llama_t1_unclear = model_stats.get("Llama-8B trial 1", {}).get("unclear", 0)
    self_break_failures = 15

    if llama_t1_genuine > 0:
        rate = llama_t1_genuine / (llama_t1_genuine + self_break_failures)
        print(f"\n{'='*60}")
        print(f"CONDITIONAL RECOVERY RATE (Llama-8B Trial 1)")
        print(f"{'='*60}")
        print(f"Confirmed genuine recoveries (T1): {llama_t1_genuine}")
        print(f"Unclear (T1, excluded):            {llama_t1_unclear}")
        print(f"SELF-BREAK failures:               {self_break_failures}")
        print(f"Recovery rate (genuine only):       {llama_t1_genuine}/({llama_t1_genuine}+{self_break_failures}) = {rate:.1%}")
        if llama_t1_unclear > 0:
            rate_upper = (llama_t1_genuine + llama_t1_unclear) / (llama_t1_genuine + llama_t1_unclear + self_break_failures)
            print(f"Recovery rate (if unclear=genuine): ({llama_t1_genuine}+{llama_t1_unclear})/({llama_t1_genuine}+{llama_t1_unclear}+{self_break_failures}) = {rate_upper:.1%}")

    # Per-condition breakdown for Llama T1
    llama_t1_records = [r for r in records if r["model"] == "Llama-8B trial 1"]
    cond_genuine = {}
    cond_total = {}
    cond_fp = {}
    cond_unc = {}
    for r in llama_t1_records:
        c = r["condition"]
        cond_total[c] = cond_total.get(c, 0) + 1
        if r["MANUAL_is_genuine_recovery"] is True:
            cond_genuine[c] = cond_genuine.get(c, 0) + 1
        elif r["MANUAL_is_genuine_recovery"] is False:
            cond_fp[c] = cond_fp.get(c, 0) + 1
        else:
            cond_unc[c] = cond_unc.get(c, 0) + 1

    print(f"\n--- Llama T1 Per-Condition ---")
    for c in sorted(cond_total):
        g = cond_genuine.get(c, 0)
        fp = cond_fp.get(c, 0)
        u = cond_unc.get(c, 0)
        t = cond_total[c]
        print(f"  {c}: {g} genuine, {fp} false pos, {u} unclear (total {t})")


if __name__ == "__main__":
    run()
