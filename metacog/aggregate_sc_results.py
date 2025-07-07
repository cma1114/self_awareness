#!/usr/bin/env python
"""
Synthesize Idea-0…3 + Model-1.51 results across models.

Now extracts *mean ± 95 % CI* for Δp (Idea 1) and ΔH (Idea 3) that the
updated log prints, and produces pooled central estimates for them.
"""

import re, math, sys, argparse
from pathlib import Path
import pandas as pd
from scipy.stats import chi2, norm, ttest_1samp

def meta_from_se(est, se):
    """
    Fixed-effect inverse-variance meta when we have coefficient & SE.
    Ignores rows where either piece is missing / non-positive.
    """
    pairs = [(e, s) for e, s in zip(est, se)
             if not math.isnan(e) and not math.isnan(s) and s > 0]
    if not pairs:
        return math.nan, math.nan, math.nan, math.nan

    w       = [1 / s**2 for _, s in pairs]
    pooled  = sum(wi * ei for wi, (ei, _) in zip(w, pairs)) / sum(w)
    se_pool = math.sqrt(1 / sum(w))
    p_val   = 2 * (1 - norm.cdf(abs(pooled / se_pool)))
    return pooled, pooled - 1.96 * se_pool, pooled + 1.96 * se_pool, p_val

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1 + z ** 2 / n
    ctr   = phat + z ** 2 / (2 * n)
    adj   = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return (ctr - adj) / denom, (ctr + adj) / denom


def fishers_method(p_vals):
    p_vals = [p for p in p_vals if p > 0 and not math.isnan(p)]
    if not p_vals:
        return math.nan
    stat = -2 * sum(math.log(p) for p in p_vals)
    return 1 - chi2.cdf(stat, 2 * len(p_vals))


def inv_var_meta(est, low, up):
    """
    Fixed-effect inverse-variance meta of estimates that have 95 % CIs.
    Returns pooled_est, ci_low, ci_up.
    """
    est, low, up = map(list, (est, low, up))
    if not est:
        return math.nan, math.nan, math.nan, math.nan

    se = [(u - l) / (2 * 1.96) for l, u in zip(low, up)]
    w  = [1 / s ** 2 if s > 0 else 0 for s in se]
    if not any(w):
        return math.nan, math.nan, math.nan, math.nan

    pooled = sum(wi * ei for wi, ei in zip(w, est)) / sum(w)
    se_p   = math.sqrt(1 / sum(w))
    p_val  = 2 * (1 - norm.cdf(abs(pooled / se_p)))
    return pooled, pooled - 1.96 * se_p, pooled + 1.96 * se_p, p_val


def random_effects_meta(est):
    """
    Random-effects meta-analysis using the unweighted mean and a t-test.
    """
    est = [e for e in est if not math.isnan(e)]
    if not est:
        return math.nan, math.nan, math.nan, math.nan

    mean = sum(est) / len(est)
    se = math.sqrt(sum((e - mean) ** 2 for e in est) / (len(est) * (len(est) - 1))) if len(est) > 1 else 0
    low, up = (mean - 1.96 * se, mean + 1.96 * se) if se > 0 else (mean, mean)
    _, p_val = ttest_1samp(est, 0)
    return mean, low, up, p_val


# ── regexes ────────────────────────────────────────────────────────────────────
HDR_RE_CORRECTNESS = re.compile(r'^--- Analyzing (.*?) \((?:Redacted,\s*)?(Correct|Incorrect),', re.MULTILINE)
HDR_RE_NO_CORRECTNESS = re.compile(r'^--- Analyzing (.*?)(?:\s\(|,|$)', re.MULTILINE)
IDEA0_RE  = re.compile(r'Proportion of changes to 2nd choice:\s+([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]\s+\(n=(\d+)\)')
IDEA1_P_RE = re.compile(r'Wilcoxon delta_p:[^\n]*?p=([\d.eE-]+)')
IDEA1_MEAN_RE = re.compile(r'Mean Δp\s*=\s*([-\d.]+)\s*\[\s*([-\d.]+),\s*([-\d.]+)\]')
IDEA2_RE  = re.compile(r'^\s*p1\s+([-\d.]+)\s+[-\d.]+\s+[-\d.]+\s+([\d.eE-]+)', re.MULTILINE)
IDEA3_P_RE = re.compile(r'Wilcoxon delta_H:[^\n]*?p=([\d.eE-]+)')
IDEA3_MEAN_RE = re.compile(r'Mean ΔH\s*=\s*([-\d.]+)\s*\[\s*([-\d.]+),\s*([-\d.]+)\]')
IDEA3b_P_RE = re.compile(r'Wilcoxon delta_H Changed:[^\n]*?p=([\d.eE-]+)')
IDEA3b_MEAN_RE = re.compile(r'Mean ΔH Changed\s*=\s*([-\d.]+)\s*\[\s*([-\d.]+),\s*([-\d.]+)\]')
M151_RE   = re.compile(r'I\(p1_z \*\* 2\)\s+([-\d.]+)\s+[-\d.]+\s+[-\d.]+\s+([\d.eE-]+)')
IDEA5_P_RE = re.compile(r'Wilcoxon (game_entropy vs capabilities_entropy):[^\n]*?p=([\d.eE-]+)')
IDEA5_MEAN_RE = re.compile(r'Mean capabilities_entropy-game_entropy\s*=\s*([-\d.]+)\s*\[\s*([-\d.]+),\s*([-\d.]+)\]')

# ── parsing --------------------------------------------------------------------
def parse_log(path: Path, breakout_by_correctness: bool) -> pd.DataFrame:
    text = path.read_text(encoding='utf-8', errors='ignore')
    rows = []
    HDR_RE = HDR_RE_CORRECTNESS if breakout_by_correctness else HDR_RE_NO_CORRECTNESS
    headers = list(HDR_RE.finditer(text))

    for i, h in enumerate(headers):
        block = text[h.end(): headers[i + 1].start() if i + 1 < len(headers) else None]

        row = dict(model=h.group(1).strip(),
                   cor_incor=h.group(2) if breakout_by_correctness else "All",
                   # Idea 0
                   idea0_prop=math.nan, idea0_low=math.nan, idea0_up=math.nan, idea0_n=0,
                   # Idea 1
                   idea1_p=math.nan, idea1_mean=math.nan, idea1_low=math.nan, idea1_up=math.nan,
                   # Idea 2
                   idea2_coef=math.nan, idea2_p=math.nan,
                   # Idea 3
                   idea3_p=math.nan, idea3_mean=math.nan, idea3_low=math.nan, idea3_up=math.nan,
                   idea3b_p=math.nan, idea3b_mean=math.nan, idea3b_low=math.nan, idea3b_up=math.nan,
                   # Model 1.51
                   m151_quad_coef=math.nan, m151_quad_p=math.nan,
                   # Idea 5
                   idea5_p=math.nan, idea5_mean=math.nan, idea5_low=math.nan, idea5_up=math.nan)

        if (m := IDEA0_RE.search(block)):
            row.update(dict(idea0_prop=float(m.group(1)),
                            idea0_low=float(m.group(2)),
                            idea0_up=float(m.group(3)),
                            idea0_n=int(m.group(4))))
        if (m := IDEA1_P_RE.search(block)):
            row['idea1_p'] = float(m.group(1))
        if (m := IDEA1_MEAN_RE.search(block)):
            row.update(dict(idea1_mean=float(m.group(1)),
                            idea1_low=float(m.group(2)),
                            idea1_up=float(m.group(3))))
        if (m := IDEA2_RE.search(block)):
            row['idea2_coef'], row['idea2_p'] = map(float, m.groups())
        if (m := IDEA3_P_RE.search(block)):
            row['idea3_p'] = float(m.group(1))
        if (m := IDEA3_MEAN_RE.search(block)):
            row.update(dict(idea3_mean=float(m.group(1)),
                            idea3_low=float(m.group(2)),
                            idea3_up=float(m.group(3))))
        if (m := IDEA3b_P_RE.search(block)):
            row['idea3b_p'] = float(m.group(1))
        if (m := IDEA3b_MEAN_RE.search(block)):
            row.update(dict(idea3b_mean=float(m.group(1)),
                            idea3b_low=float(m.group(2)),
                            idea3b_up=float(m.group(3))))
        if (m := M151_RE.search(block)):
            row['m151_quad_coef'], row['m151_quad_p'] = map(float, m.groups())
        if (m := IDEA5_P_RE.search(block)):
            row['idea5_p'] = float(m.group(1))
        if (m := IDEA5_MEAN_RE.search(block)):
            row.update(dict(idea5_mean=float(m.group(1)),
                            idea5_low=float(m.group(2)),
                            idea5_up=float(m.group(3))))

        rows.append(row)

    return pd.DataFrame(rows)


# ── aggregation ----------------------------------------------------------------
def aggregate(df: pd.DataFrame, subset: str) -> dict:
    sub = df[df.cor_incor == subset]

    # Idea 0  (pooled proportion)
    k = (sub.idea0_prop * sub.idea0_n).sum()
    n = sub.idea0_n.sum()
    prop = k / n if n else math.nan
    low0, up0 = wilson_ci(k, n) if n else (math.nan, math.nan)

    # Idea 1  (meta-analytic mean Δp)
    m1, l1, u1, p1 = random_effects_meta(sub.idea1_mean.dropna().tolist())

    # Idea 2  (slope)
    mc, ml, mu, p2 = random_effects_meta(sub.idea2_coef.tolist())

    # Idea 3  (meta ΔH)
    m3, l3, u3, p3 = random_effects_meta(sub.idea3_mean.dropna().tolist())
    m3b, l3b, u3b, p3b = random_effects_meta(sub.idea3b_mean.dropna().tolist())

    # Model 1.51
    mq, lq, uq, pq = random_effects_meta(sub.m151_quad_coef.tolist())

    # Idea 5
    m5, l5, u5, p5 = random_effects_meta(sub.idea5_mean.dropna().tolist())

    return dict(
        subset=subset,
        # Idea 0
        idea0_prop=prop, idea0_low=low0, idea0_up=up0, idea0_n=int(n),
        # Idea 1
        idea1_mean=m1, idea1_low=l1, idea1_up=u1, idea1_p=p1,
        # Idea 2
        idea2_coef=mc, idea2_low=ml, idea2_up=mu, idea2_p=p2,
        # Idea 3
        idea3_mean=m3, idea3_low=l3, idea3_up=u3, idea3_p=p3,
        idea3b_mean=m3b, idea3b_low=l3b, idea3b_up=u3b, idea3b_p=p3b,
        # Idea 5
        idea5_mean=m5, idea5_low=l5, idea5_up=u5, idea5_p=p5,
        # Model 1.51
        m151_quad=mq, m151_low=lq, m151_up=uq, m151_p=pq
    )


# ── pretty printing ------------------------------------------------------------
def print_tables(df_models: pd.DataFrame, df_pool: pd.DataFrame, breakout_by_correctness: bool):
    pd.set_option('display.precision', 3)

    print("\n=== Per-model results (unaggregated) ===")
    cols = [
        'model',
        # Idea 0
        'idea0_prop', 'idea0_low', 'idea0_up', 'idea0_n',
        # Idea 1
        'idea1_mean', 'idea1_low', 'idea1_up', 'idea1_p',
        # Idea 2
        'idea2_coef', 'idea2_p',
        # Idea 3
        'idea3_mean', 'idea3_low', 'idea3_up', 'idea3_p',
        'idea3b_mean', 'idea3b_low', 'idea3b_up', 'idea3b_p',
        # Model 1.51
        'm151_quad_coef', 'm151_quad_p',
        # Idea 5
        'idea5_mean', 'idea5_low', 'idea5_up', 'idea5_p'
    ]
    if breakout_by_correctness:
        cols.insert(1, 'cor_incor')
    
    print(df_models[cols].to_string(index=False))

    print("\n=== Pooled results (aggregated) ===")
    cols2 = [
        'subset',
        # Idea 0
        'idea0_prop', 'idea0_low', 'idea0_up', 'idea0_n',
        # Idea 1
        'idea1_mean', 'idea1_low', 'idea1_up', 'idea1_p',
        # Idea 2
        'idea2_coef', 'idea2_low', 'idea2_up', 'idea2_p',
        # Idea 3
        'idea3_mean', 'idea3_low', 'idea3_up', 'idea3_p',
        'idea3b_mean', 'idea3b_low', 'idea3b_up', 'idea3b_p',
        # Model 1.51
        'm151_quad', 'm151_low', 'm151_up', 'm151_p',
        # Idea 5
        'idea5_mean', 'idea5_low', 'idea5_up', 'idea5_p'
    ]
    print(df_pool[cols2].to_string(index=False))


# ── CLI entry point ------------------------------------------------------------
def main():
    BREAKOUT_BY_CORRECTNESS = False
    dataset = "gpqa" #"simplemc" # 
    sc_version = "_new"  # "_new" or ""
    suffix = "_all" if not BREAKOUT_BY_CORRECTNESS else ""
    fname=f"analysis_log_multi_logres_sc_{dataset}{sc_version}{suffix}.txt"
    
    fp = Path(fname)
    if not fp.exists():
        sys.exit(f"{fname} not found")

    df = parse_log(fp, BREAKOUT_BY_CORRECTNESS)
    if df.empty:
        sys.exit("No model blocks were parsed – header regex may need tweaking.")

    # Filter out subjects that don't have Idea 1-5 or model 1.51 data
    idea_cols = [c for c in df.columns if (c.startswith('idea') and not c.startswith('idea0')) or c.startswith('m151')]
    df = df.dropna(subset=idea_cols, how='all').reset_index(drop=True)
    if df.empty:
        sys.exit("No models with Idea 1-5 or model 1.51 data found.")

    subsets_to_process = ['Correct', 'Incorrect'] if BREAKOUT_BY_CORRECTNESS else ['All']
    pooled = pd.DataFrame([aggregate(df, s) for s in subsets_to_process])
    print(f"Results for {fname}:\n")
    print_tables(df, pooled, BREAKOUT_BY_CORRECTNESS)


if __name__ == '__main__':
    main()
