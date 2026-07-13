"""
04_analysis.py

Stage 4 -- statistical analysis on the corrected (post-bugfix) Stage 2 results.

UPDATED to handle multiple trials (was hardcoded to trial 1 only). Design
decision: trials are reported SEPARATELY, not silently averaged/pooled into
one number. Trial 2 confirmed real per-problem churn under temp=0.0 (see
solve_bug.py output -- Reversed 2/89 flips, Shuffled 6/89 flips, Partial
4/89 flips with a net +2 shift concentrated in the non-degenerate subgroup).
Averaging that away would hide a real, already-documented limitation
(temp=0.0 non-determinism) behind one clean-looking number. Reporting both
trials plus the flip rate keeps that limitation visible, per the project's
standing preference for the honest read over the flattering one.

For each trial, computes:
1. Per-condition accuracy and Robustness_tau (denominator = that trial's own
   baseline-control accuracy -- NOT Stage 1, and NOT mixed across trials).
2. McNemar's exact test: baseline-control vs. each condition, plus all
   pairwise condition comparisons (6 comparisons total, Bonferroni-corrected).
3. Partial split: degenerate (3-4 step) vs. non-degenerate (5+ step) --
   only non-degenerate is used in the McNemar comparison.

Then, across all trials present, reports:
4. Per-problem agreement/flip counts between trials for every condition
   (folds in what solve_bug.py checked separately, so this becomes the one
   script that produces paper-ready numbers).

Does NOT compute H2 (first-error-position analysis) -- flagged as follow-up
work at the end, same as before.


"""

import json
from scipy.stats import binomtest


STAGE1_PATH = "data/stage1_baseline_v2.jsonl"
TRIALS = [1, 2]  # add more here if a trial 3 etc. is ever run

RESULTS_PATH_TEMPLATE = "results/stage2_results_v2_trial{trial}.jsonl"
BASELINE_CONTROL_PATH_TEMPLATE = "results/stage2_baseline_control_v2_trial{trial}.jsonl"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_correctness_by_id(records, condition_filter=None):
    """Returns {problem_id: correct (bool)} from a list of records, optionally
    filtered to a specific condition (needed for the multi-condition results
    file, not needed for the single-condition baseline_control file)."""
    out = {}
    for r in records:
        if condition_filter and r.get("condition") != condition_filter:
            continue
        out[r["problem_id"]] = r["correct"]
    return out


def load_degenerate_flags(results_records):
    """Returns {problem_id: degenerate (bool)} from a trial's partial records."""
    out = {}
    for r in results_records:
        if r.get("condition") == "partial":
            out[r["problem_id"]] = r["degenerate"]
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def mcnemar_exact(correct_a, correct_b, label_a, label_b, alpha=0.05, bonferroni_n=1):
    """
    Paired McNemar's exact test on matched problem_ids present in both dicts.
    b = A correct, B wrong. c = A wrong, B correct.
    Reports significance against both raw alpha and Bonferroni-corrected alpha
    (alpha / bonferroni_n).
    """
    common_ids = set(correct_a) & set(correct_b)
    b = sum(1 for pid in common_ids if correct_a[pid] and not correct_b[pid])
    c = sum(1 for pid in common_ids if (not correct_a[pid]) and correct_b[pid])
    n_discordant = b + c

    print(f"\n  --- McNemar's test: {label_a} vs {label_b} ---")
    print(f"  Matched problems: {len(common_ids)}")
    print(f"  {label_a} correct, {label_b} wrong (b): {b}")
    print(f"  {label_a} wrong, {label_b} correct (c): {c}")
    print(f"  Discordant pairs (b+c): {n_discordant}")

    if n_discordant == 0:
        print("  No discordant pairs -- conditions agree on every problem. p=1.0 (no difference).")
        return None

    result = binomtest(min(b, c), n_discordant, p=0.5, alternative="two-sided")
    p_value = result.pvalue
    corrected_alpha = alpha / bonferroni_n

    print(f"  p-value: {p_value:.4f}")
    print(f"  Raw threshold (alpha={alpha}): "
          f"{'SIGNIFICANT' if p_value < alpha else 'not significant'}")
    print(f"  Bonferroni-corrected threshold (alpha={alpha}/{bonferroni_n}={corrected_alpha:.4f}): "
          f"{'SIGNIFICANT' if p_value < corrected_alpha else 'not significant'}")
    return p_value


# ---------------------------------------------------------------------------
# Per-trial analysis
# ---------------------------------------------------------------------------

def analyze_trial(trial):
    print("\n" + "#" * 60)
    print(f"# TRIAL {trial}")
    print("#" * 60)

    results_records = load_records(RESULTS_PATH_TEMPLATE.format(trial=trial))
    baseline_records = load_records(BASELINE_CONTROL_PATH_TEMPLATE.format(trial=trial))

    baseline_control = load_correctness_by_id(baseline_records)
    reversed_c = load_correctness_by_id(results_records, "reversed")
    shuffled_c = load_correctness_by_id(results_records, "shuffled")
    partial_c = load_correctness_by_id(results_records, "partial")
    degenerate_flags = load_degenerate_flags(results_records)

    partial_nondegenerate_c = {
        pid: correct for pid, correct in partial_c.items()
        if not degenerate_flags.get(pid, True)
    }

    def acc(d):
        return sum(d.values()) / len(d), sum(d.values()), len(d)

    bc_acc, bc_n, bc_t = acc(baseline_control)
    rev_acc, rev_n, rev_t = acc(reversed_c)
    shuf_acc, shuf_n, shuf_t = acc(shuffled_c)
    part_all_acc, part_all_n, part_all_t = acc(partial_c)
    part_nd_acc, part_nd_n, part_nd_t = acc(partial_nondegenerate_c)

    print(f"\n{'Condition':<30}{'Accuracy':<18}{'Robustness_tau':<15}")
    print(f"{'Baseline-control':<30}{f'{bc_n}/{bc_t} ({bc_acc:.2%})':<18}{'1.000 (ref)':<15}")
    print(f"{'Reversed':<30}{f'{rev_n}/{rev_t} ({rev_acc:.2%})':<18}{rev_acc/bc_acc:<15.3f}")
    print(f"{'Shuffled':<30}{f'{shuf_n}/{shuf_t} ({shuf_acc:.2%})':<18}{shuf_acc/bc_acc:<15.3f}")
    print(f"{'Partial (blended)':<30}{f'{part_all_n}/{part_all_t} ({part_all_acc:.2%})':<18}{part_all_acc/bc_acc:<15.3f}")
    print(f"{'Partial (non-degenerate)':<30}{f'{part_nd_n}/{part_nd_t} ({part_nd_acc:.2%})':<18}{part_nd_acc/bc_acc:<15.3f}")

    print(f"\n  Significance tests (McNemar's exact, paired, two-sided, Bonferroni n=6):")
    N_COMPARISONS = 6
    mcnemar_exact(baseline_control, reversed_c, "Baseline-control", "Reversed", bonferroni_n=N_COMPARISONS)
    mcnemar_exact(baseline_control, shuffled_c, "Baseline-control", "Shuffled", bonferroni_n=N_COMPARISONS)
    mcnemar_exact(baseline_control, partial_nondegenerate_c, "Baseline-control", "Partial (non-degenerate)", bonferroni_n=N_COMPARISONS)
    mcnemar_exact(reversed_c, shuffled_c, "Reversed", "Shuffled", bonferroni_n=N_COMPARISONS)
    mcnemar_exact(reversed_c, partial_nondegenerate_c, "Reversed", "Partial (non-degenerate)", bonferroni_n=N_COMPARISONS)
    mcnemar_exact(shuffled_c, partial_nondegenerate_c, "Shuffled", "Partial (non-degenerate)", bonferroni_n=N_COMPARISONS)

    # Return the raw per-id dicts so the cross-trial comparison section can reuse them
    return {
        "baseline_control": baseline_control,
        "reversed": reversed_c,
        "shuffled": shuffled_c,
        "partial": partial_c,
        "degenerate_flags": degenerate_flags,
    }


# ---------------------------------------------------------------------------
# Cross-trial comparison
# ---------------------------------------------------------------------------

def compare_trials(trial_data, t1, t2, condition_key, label, degenerate_flags=None):
    a = trial_data[t1][condition_key]
    b = trial_data[t2][condition_key]
    common_ids = set(a) & set(b)

    agree = sum(1 for pid in common_ids if a[pid] == b[pid])
    flip_pos = [pid for pid in common_ids if (not a[pid]) and b[pid]]
    flip_neg = [pid for pid in common_ids if a[pid] and (not b[pid])]
    net = len(flip_pos) - len(flip_neg)
    flip_rate = (len(flip_pos) + len(flip_neg)) / len(common_ids)

    print(f"\n  --- {label}: Trial {t1} vs Trial {t2} ---")
    print(f"  Agree: {agree}/{len(common_ids)}  |  "
          f"Flip rate: {flip_rate:.1%}  |  "
          f"Flips wrong->correct: {len(flip_pos)} {flip_pos}  |  "
          f"Flips correct->wrong: {len(flip_neg)} {flip_neg}  |  "
          f"Net: {net:+d}")

    if degenerate_flags is not None:
        for deg_flag, tag in [(True, "degenerate"), (False, "non-degenerate")]:
            subset = [pid for pid in common_ids if degenerate_flags.get(pid) == deg_flag]
            if not subset:
                continue
            sub_agree = sum(1 for pid in subset if a[pid] == b[pid])
            sub_pos = sum(1 for pid in subset if (not a[pid]) and b[pid])
            sub_neg = sum(1 for pid in subset if a[pid] and (not b[pid]))
            print(f"    {tag}: n={len(subset)}, agree={sub_agree}, "
                  f"flip_wrong->correct={sub_pos}, flip_correct->wrong={sub_neg}")


def run():
    trial_data = {trial: analyze_trial(trial) for trial in TRIALS}

    if len(TRIALS) >= 2:
        print("\n" + "#" * 60)
        print("# CROSS-TRIAL COMPARISON (all trial pairs)")
        print("#" * 60)
        print("Reported as a stated limitation (temp=0.0 non-determinism), not")
        print("averaged away. See project handoff Section 6 for background.")

        for i in range(len(TRIALS)):
            for j in range(i + 1, len(TRIALS)):
                t1, t2 = TRIALS[i], TRIALS[j]
                compare_trials(trial_data, t1, t2, "baseline_control", "Baseline-control")
                compare_trials(trial_data, t1, t2, "reversed", "Reversed")
                compare_trials(trial_data, t1, t2, "shuffled", "Shuffled")
                # use trial t1's degenerate flags for the split (should match t2's,
                # since permutation assignment is deterministic/seeded, not model-dependent)
                compare_trials(
                    trial_data, t1, t2, "partial", "Partial",
                    degenerate_flags=trial_data[t1]["degenerate_flags"],
                )

    print("\n" + "#" * 60)
    print("# NOT YET IMPLEMENTED: H2 (first-error-position analysis)")
    print("#" * 60)
    print("Requires knowing which step position the model's reasoning first")
    print("diverged at for the Shuffled condition -- needs either manual")
    print("annotation of a sample, or a separate model call asking it to")
    print("identify where it got confused. Design as a follow-up, not skipped.")


if __name__ == "__main__":
    run()