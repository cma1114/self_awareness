#!/usr/bin/env python3
"""
Analyse an answer‑or‑delegate game (v3)

Changes vs v2
-------------
✓  Keeps *plain* average regret (+95 % CI, paired t‑test)
✓  Scaled Regret defined as regret / regret_random   (# lower = better)
✓  All numeric CIs printed as [lo, hi] with 3‑dec precision
✓  IEI set to 'n/a' when oracle head‑room < 0.1 pp to avoid instability
"""

import argparse, json, os, re, sys, math
import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score, brier_score_loss
from typing import Tuple, Callable

# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #
def bootstrap(arrs, stat: Callable, n=2000, alpha=0.05, seed=0) -> Tuple[float, float]:
    """Generic non‑parametric bootstrap CI."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(arrs[0]))
    boot_stats = np.empty(n)
    for i in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        boot_stats[i] = stat(*(a[b] for a in arrs))
    lo, hi = np.percentile(boot_stats, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def expected_calibration_error(conf, corr, n_bins=10):
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins-1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += abs(corr[m].mean() - conf[m].mean()) * m.mean()
    return ece

def bootstrap_stat(arrs, stat_fn, n=2000, alpha=0.05, seed=0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(arrs[0]))
    boot = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        boot.append(stat_fn(*(a[b] for a in arrs)))
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

def parse_game_filename(path):
    f = os.path.basename(path)
    m = re.match(r'(.+?)_([A-Za-z0-9]+)_(\d+)_(\d+)_team([\d.]+)_.*_game_data\.json', f)
    if not m:
        raise ValueError(f'Cannot parse filename: {f}')
    model, dataset, n1, n2, team = m.groups()
    return model, dataset, int(n1), int(n2), float(team)


def pct(x):        return f'{100*x:.1f}%'
def fmt(x):        return '—' if np.isnan(x) else f'{x:.3f}'
def fmt_ci(tup):   return '[—, —]' if any(map(np.isnan, tup)) else f'[{tup[0]:.3f}, {tup[1]:.3f}]'
def pct_ci(tup):   return '[—, —]' if any(map(np.isnan, tup)) else f'[{pct(tup[0])}, {pct(tup[1])}]'

# -------------------------------------------------------------------- #
def analyse(game_path):
    # ---------- load files -------------------------------------------
    game = json.load(open(game_path))
    model, dataset, n1, n2, team_acc_name = parse_game_filename(game_path)
    cap_path = f'./completed_results_{dataset.lower()}/{model}_phase1_completed.json'
    if not os.path.isfile(cap_path):
        sys.exit(f'Baseline capabilities file not found: {cap_path}')
    caps = json.load(open(cap_path))

    baseline = {qid: info['is_correct'] for qid, info in caps['results'].items()}
    trials   = [t for t in game['results'] if t.get('phase') == 2]
    if len(trials) != n2:
        print(f'⚠️  Expected {n2} Phase‑2 trials, found {len(trials)}.')

    qids = [t['question_id'] for t in trials]
    if missing := [q for q in qids if q not in baseline]:
        sys.exit(f'Missing baseline answers for {len(missing)} items.')

    def stat_sr(tc, sc, oc, dg):
        d  = dg.mean()
        ar = d * p_team + (1-d) * sc.mean()
        denom = oc.mean() - ar
        if denom < 1e-3:                 # <0.1 pp → discard replicate
            return np.nan
        val = (oc.mean() - tc.mean()) / denom
        return min(max(val, 0.0), 1.0)   # clip into [0,1]

    # ---------- per‑item arrays --------------------------------------
    self_corr = np.array([baseline[q]          for q in qids], int)
    team_corr = np.array([t['team_correct']    for t in trials], int)
    delegated = np.array([t['delegation_choice'] != 'Self' for t in trials])

    # confidence
    p_self = np.full(len(trials), np.nan)
    for i, t in enumerate(trials):
        pr = t.get('probs')
        if pr and t['correct_answer'] in pr:
            p_self[i] = pr[t['correct_answer']]
    has_conf = ~np.isnan(p_self)

    p_team = game['teammate_accuracy_phase1']
    assert math.isclose(p_team, team_acc_name, abs_tol=1e-3)

    # ---------- basic metrics ----------------------------------------
    self_acc  = self_corr.mean()
    team_acc  = team_corr.mean()
    lift      = team_acc - self_acc
    rel_lift  = (team_acc / self_acc - 1) if self_acc else np.nan

    d_rate    = delegated.mean()
    del_prec  = team_corr[delegated].mean()      if delegated.any() else np.nan
    self_prec = self_corr[~delegated].mean()     if (~delegated).any() else np.nan

    # ---------- feasible oracle --------------------------------------
    use_self = p_self >= p_team
    use_self = np.where(np.isnan(p_self), ~delegated, use_self)
    oracle_corr = np.where(use_self, self_corr, team_corr)
    acc_oracle  = oracle_corr.mean()
    headroom    = acc_oracle - self_acc
    print(f'Headroom: {headroom:.3f} (self={self_acc:.3f}, team={team_acc:.3f})')
    print(f'Oracle accuracy: {acc_oracle:.3f}')

    if headroom < 0.001:
        iei, ci_iei = np.nan, (np.nan, np.nan)
    else:
        iei = (team_acc - self_acc) / headroom
        ci_iei = bootstrap([team_corr, self_corr, oracle_corr],
                           lambda tc, sc, oc: (tc.mean()-sc.mean()) /
                                               max(1e-9, oc.mean()-sc.mean()))

    # ---------- clairvoyant oracle ------------------------------------
    clairvoyant_acc  = self_acc + (1 - self_acc) * p_team
    ce               = (team_acc - self_acc) / max(1e-9, clairvoyant_acc - self_acc)
    ci_ce = bootstrap([team_corr],
                    lambda tc: (tc.mean() - self_acc) /
                                max(1e-9, clairvoyant_acc - self_acc))

    # ---------- regrets ---------------------------------------------
    avg_regret = acc_oracle - team_acc
    ci_regret  = bootstrap([team_corr, oracle_corr],
                           lambda tc, oc: oc.mean() - tc.mean())
    p_regret   = ttest_rel(acc_oracle - oracle_corr, acc_oracle - team_corr).pvalue  # same as tc vs oracle?

    # Scaled Regret
    acc_rand   = d_rate * p_team + (1 - d_rate) * self_acc
    denom_sr   = acc_oracle - acc_rand
    if denom_sr < 0.001:
        sr, ci_sr = np.nan, (np.nan, np.nan)
    else:
        sr = (acc_oracle - team_acc) / denom_sr
        ci_sr = bootstrap_stat([team_corr, self_corr, oracle_corr, delegated], stat_sr)

    # ---------- confidence quality -----------------------------------
    if has_conf.any() and self_corr[has_conf].var() > 0:
        auroc = roc_auc_score(self_corr[has_conf], p_self[has_conf])
    else:
        auroc = np.nan
    if has_conf.any():
        brier = brier_score_loss(self_corr[has_conf], p_self[has_conf])
        ece   = expected_calibration_error(p_self[has_conf], self_corr[has_conf])
    else:
        brier = ece = np.nan

    # ---------- CIs for percentages ---------------------------------
    ci_team  = bootstrap([team_corr],  lambda x: x.mean())
    ci_self  = bootstrap([self_corr],  lambda x: x.mean())
    ci_lift  = bootstrap([team_corr, self_corr],
                         lambda tc, sc: (tc - sc).mean())
    ci_drate = bootstrap([delegated.astype(float)], lambda x: x.mean())

    # ---------- report ----------------------------------------------
    print(f"""
──────────────────────────────────────────────────────────────────────────
Delegate‑game analysis for model: {model}
  Dataset         : {dataset}
  Phase‑1 trials  : {n1}
  Phase‑2 trials  : {n2}
  Teammate acc φ₁ : {pct(p_team)}
──────────────────────────────────────────────────────────────────────────
TEAM PERFORMANCE
  Team accuracy (φ₂)          : {pct(team_acc)}   CI95 {pct_ci(ci_team)}
  Self true accuracy          : {pct(self_acc)}   CI95 {pct_ci(ci_self)}
  Lift (abs)                  : {pct(lift)}       CI95 {pct_ci(ci_lift)}
  Relative lift               : {rel_lift*100:.1f}% if baseline ≠ 0
  IEI_feas                    : {fmt(iei)}    CI95 {fmt_ci(ci_iei)}
  Avg regret (oracle‑team)    : {fmt(avg_regret)}  CI95 {fmt_ci(ci_regret)}   t‑test p={p_regret:.4g}
  Scaled Regret (lower=better): {fmt(sr)}     CI95 {fmt_ci(ci_sr)}
  Clairvoyant accuracy          : {pct(clairvoyant_acc)}
  Clairvoyant‑Efficiency (CE)   : {fmt(ce)}   CI95 {fmt_ci(ci_ce)}

DELEGATION BEHAVIOUR
  Delegation rate             : {pct(d_rate)}   CI95 {pct_ci(ci_drate)}
  Deleg‑precision             : {fmt(del_prec)}
  Self‑answer precision       : {fmt(self_prec)}

CONFIDENCE QUALITY (items with logprobs: {has_conf.sum()}/{len(p_self)})
  AUROC                       : {fmt(auroc)}
  Brier score                 : {fmt(brier)}
  Expected Calibration Error  : {fmt(ece)}
──────────────────────────────────────────────────────────────────────────
""")


# -------------------------------------------------------------------- #
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('game_file', help='Path to *_game_data.json file')
    args = ap.parse_args()
    if not os.path.isfile(args.game_file):
        sys.exit('File not found')
    analyse(args.game_file)
