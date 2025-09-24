import os
from typing import List, Optional, Dict, Tuple
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections.abc import Mapping
import re


def parse_results_file(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Parse one results file.

    Returns:
      data[model][metric] = {'value': float, 'lo': float or None, 'hi': float or None}
    Skips entries with "Not found".
    """
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    current_model = None

    subject_re = re.compile(r'^\s*Subject:\s*(.+?)\s*$')
    line_re = re.compile(r'^\s{2,}(.+?):\s*(.+?)\s*$')
    num_ci_re = re.compile(
        r'^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\[\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]\s*$'
    )
    num_only_re = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$')

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')

            m_sub = subject_re.match(line)
            if m_sub:
                current_model = m_sub.group(1).strip()
                if current_model not in data:
                    data[current_model] = {}
                continue

            m_line = line_re.match(line)
            if not m_line or current_model is None:
                continue

            metric = m_line.group(1).strip()
            val_str = m_line.group(2).strip()

            # Skip "Not found"
            if val_str.lower().startswith('not found'):
                continue

            m_num_ci = num_ci_re.match(val_str)
            if m_num_ci:
                v = float(m_num_ci.group(1))
                lo = float(m_num_ci.group(2))
                hi = float(m_num_ci.group(3))
                data[current_model][metric] = {'value': v, 'lo': lo, 'hi': hi}
                continue

            m_num = num_only_re.match(val_str)
            if m_num:
                v = float(m_num.group(1))
                data[current_model][metric] = {'value': v, 'lo': None, 'hi': None}
                continue

            # Unrecognized format; ignore
    return data


def pool_metric_ivw(metric: str, files: List[str], ci_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    # Returns: {model: {'value': pooled_mean, 'lo': ci_lo, 'hi': ci_hi}}
    try:
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + ci_level/2.0))
    except Exception:
        z = 1.96 if abs(ci_level - 0.95) < 1e-6 else 1.96  # basic fallback

    parsed = [parse_results_file(f) for f in files]
    all_models = sorted(set().union(*[set(d.keys()) for d in parsed]))
    out: Dict[str, Dict[str, float]] = {}

    for m in all_models:
        means, ses = [], []
        for d in parsed:
            entry = d.get(m, {}).get(metric) or next((d.get(m, {}).get(k) for k in d.get(m, {}) if k.lower()==metric.lower()), None)
            if not entry:
                continue
            mu = entry.get("value")
            lo = entry.get("lo")
            hi = entry.get("hi")
            if mu is None or not np.isfinite([mu]).all():
                continue
            if lo is None or hi is None:
                #compute SE of proportions, using N=447
                se = math.sqrt(mu * (1.0 - mu) / 447)
            else:
                se = (hi - lo) / (2.0 * z)
            if se <= 0 or not np.isfinite(se):
                continue
            means.append(mu)
            ses.append(se)

        if len(means) == 0:
            continue

        w = np.reciprocal(np.square(ses))
        mu_hat = float(np.sum(w * means) / np.sum(w))
        se_mu = float(math.sqrt(1.0 / np.sum(w)))
        lo_hat = mu_hat - z * se_mu
        hi_hat = mu_hat + z * se_mu
        out[m] = {"value": mu_hat, "lo": lo_hat, "hi": hi_hat}

    return out


# Make a single-panel plot from pooled data:
def plot_single_from_parsed(metric: str, data: Dict[str, Dict[str, Dict[str, float]]], title: str):
    # data shape: {model: {metric: {'value','lo','hi'}}}
    tmp_path = "<memory>"
    # Quick shim: convert dict to the shape parse_results_file returns
    def fake_parse(_): return data
    global parse_results_file
    old = parse_results_file
    parse_results_file = fake_parse
    try:
        plot_metric_panels_from_results(
            metric=metric,
            files=[tmp_path],
            series_names=[title],
            sharey=False,
        )
    finally:
        parse_results_file = old

def plot_metric_panels_from_results(
    metric: str,
    files: List[str],
    series_names: Optional[List[str]] = None,  # panel titles
    model_order: Optional[List[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
    suptitle: Optional[str] = None,
    outfile: Optional[str] = None,
    dpi: int = 150,
    # Error bars
    show_errorbars: bool = True,
    ecolor: str = "gray",
    alpha_err: float = 1.0,
    elinewidth: float = 1.0,
    capsize: float = 3.0,
    # Panel behavior
    chance: Optional[float] = None,
    sharey: bool = True,
    consistent_models: bool = False,  # if True, only keep models present in ALL files
    bar_color: Optional[str] = None,  # uniform color for bars; None -> default cycle
    annotate: bool = True,            # write values inside bars
    value_fmt: str = "{:.2f}",        # format for values
    label_color: str = "black",
    label_fontsize: int = 5,
    metric_label=None,            # y-axis label (default: metric name)
    show_trend: bool = True,
    trend_color: str = "crimson",
    trend_style: str = "-",
    trend_width: float = 1.5,
    trend_weighted: bool = False,  # weight by 1/SE^2 if error bars available
    trend_text_loc: tuple = (0.04, 0.96),  # axes fraction (x,y)
    ) -> Tuple[plt.Figure, np.ndarray, List[pd.DataFrame]]:
    """
    Plot each results file in its own panel (2 files -> 1x2, 4 files -> 2x2).
    Uses parse_results_file exactly as-is.
    Returns (fig, axes_flat, list_of_value_dfs).
    """
    n = len(files)
    if n not in (1, 2, 4):
        raise ValueError("This helper expects 1, 2, or 4 files.")
    if series_names is None:
        #series_names = [os.path.basename(f) for f in files]
        def _title_of(x, i):
            if isinstance(x, str):
                return os.path.basename(x)
            return f"Panel {i+1}"  # or "Pooled"
        series_names = [_title_of(x, i) for i, x in enumerate(files)]

    if len(series_names) != n:
        raise ValueError("series_names length must match files length")

    # Parse all files
    #parsed = [parse_results_file(f) for f in files]
    # Parse all files or accept parsed dicts directly
    parsed = []
    for item in files:
        if isinstance(item, str):
            parsed.append(parse_results_file(item))
        elif isinstance(item, Mapping):
            parsed.append(item)  # already in {model: {metric: {'value','lo','hi'}}} shape
        else:
            raise TypeError("files entries must be paths (str) or parsed dicts")
    
    # Collect model sets
    model_sets = [set(d.keys()) for d in parsed]
    if model_order:
        base_models = [m for m in model_order if all((not consistent_models) or (m in s) for s in model_sets)]
        if not consistent_models:
            # include additional models from any file (but model_order first)
            union_rest = sorted(set().union(*model_sets) - set(base_models))
            base_models = base_models + union_rest
    else:
        base_models = sorted(set.intersection(*model_sets)) if consistent_models else sorted(set().union(*model_sets))

    # Layout
    if n == 1:
        nrows, ncols = 1, 1
        figsize = (7, 4.8)
    elif n == 2:
        nrows, ncols = 1, 2
        figsize = (12, 4.8)
    else:  # n == 4
        nrows, ncols = 2, 2
        figsize = (12, 9)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, sharey=sharey)
    axes = np.array(axes).reshape(-1)  # flatten

    wide_vals_list: List[pd.DataFrame] = []
    all_ylim_candidates = []

    # Helper: find metric entry with case-insensitive fallback
    def get_entry(mm: Dict[str, Dict[str, float]], name: str):
        if name in mm:
            return mm[name]
        lm = {k.lower(): k for k in mm}
        k = lm.get(name.lower())
        return mm.get(k) if k else None

    for ax, f, title, data in zip(axes, files, series_names, parsed):
        # Build per-panel table
        models_panel = []
        vals = []
        lo_list = []
        hi_list = []

        # Respect base_models ordering; skip models missing this metric
        for m in base_models:
            entry = get_entry(data.get(m, {}), metric)
            if entry is None or entry.get("value") is None:
                continue
            v = entry.get("value", np.nan)
            lo = entry.get("lo", np.nan)
            hi = entry.get("hi", np.nan)
            models_panel.append(m)
            vals.append(v)
            lo_list.append(lo)
            hi_list.append(hi)

        # If nothing to plot, leave empty panel with a note
        if not models_panel:
            ax.text(0.5, 0.5, "No data for this metric", ha="center", va="center", fontsize=10, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("Model")
            ax.set_ylabel(metric_label or metric)
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            continue

        # DataFrame for this panel (for return/debug)
        wide = pd.DataFrame(
            {"value": vals, "lo": lo_list, "hi": hi_list},
            index=models_panel,
        )
        wide_vals_list.append(wide)

        # X/ticks
        x = np.arange(len(models_panel))
        width = 0.7
        ticklabels = [aliases.get(m, m) for m in models_panel] if aliases else models_panel

        # Error bars
        kwargs = {}
        if show_errorbars:
            v = np.asarray(vals, dtype=float)
            lo = np.asarray(lo_list, dtype=float)
            hi = np.asarray(hi_list, dtype=float)
            lower = v - lo
            upper = hi - v
            invalid = np.isnan(lower) | np.isnan(upper) | (lower < 0) | (upper < 0)
            lower = np.where(invalid, np.nan, lower)
            upper = np.where(invalid, np.nan, upper)
            yerr = ma.array(np.vstack([lower, upper]), mask=np.isnan(np.vstack([lower, upper])))
            if (~yerr.mask).any():
                kwargs.update(dict(
                    yerr=yerr, capsize=capsize,
                    error_kw=dict(elinewidth=elinewidth, alpha=alpha_err, ecolor=ecolor),
                ))

            # Collect y-limit candidates (values and CIs)
            valid_min = np.nanmin(np.where(np.isnan(lo), v, lo))
            valid_max = np.nanmax(np.where(np.isnan(hi), v, hi))
            all_ylim_candidates.append(valid_min)
            all_ylim_candidates.append(valid_max)
        else:
            all_ylim_candidates.extend([np.nanmin(vals), np.nanmax(vals)])

        # Bars
        #ax.bar(x, vals, width, **kwargs)
        bar_kwargs = dict(**kwargs)
        if bar_color is not None:
            bar_kwargs["color"] = bar_color
        bars = ax.bar(x, vals, width, **bar_kwargs)
        if annotate:
            for rect, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                va = "top" if v >= 0 else "bottom"
                dy = -3 if v >= 0 else 3  # points offset
                ax.annotate(
                    value_fmt.format(v),
                    xy=(rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=label_fontsize,
                    color=label_color,
                    clip_on=True,
                )

        # Chance line (optional)
        if chance is not None:
            ax.axhline(chance, linestyle="--", color="0.4", linewidth=1.2, zorder=0)

        if show_trend and len(vals) >= 2:
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(vals, dtype=float)
            mask = np.isfinite(y_arr)
            x_fit, y_fit = x_arr[mask], y_arr[mask]
            if len(y_fit) >= 2:
                # Optional weights from CIs
                w = None
                if trend_weighted and show_errorbars:
                    lo_arr = np.asarray(lo_list, dtype=float)[mask]
                    hi_arr = np.asarray(hi_list, dtype=float)[mask]
                    # infer SE from CI if possible
                    try:
                        from scipy.stats import norm, t as tdist
                        z = float(norm.ppf(0.975))  # assumes ~95% CI
                    except Exception:
                        z = 1.96
                        tdist = None
                    se = (hi_arr - lo_arr) / (2.0 * z)
                    w = np.where(np.isfinite(se) & (se > 0), 1.0 / (se**2), np.nan)
                    if not np.isfinite(w).any():
                        w = None

                # Fit
                slope = intercept = np.nan
                pval = np.nan
                if w is None:
                    try:
                        from scipy.stats import linregress
                        lr = linregress(x_fit, y_fit)
                        slope, intercept, pval = lr.slope, lr.intercept, lr.pvalue
                    except Exception:
                        # Unweighted closed form (no p-value)
                        X = np.vstack([np.ones_like(x_fit), x_fit]).T
                        beta = np.linalg.lstsq(X, y_fit, rcond=None)[0]
                        intercept, slope = beta[0], beta[1]
                else:
                    # Weighted least squares closed form + p-value if SciPy available
                    X = np.vstack([np.ones_like(x_fit), x_fit]).T
                    W = np.diag(w)
                    XtW = X.T @ W
                    beta = np.linalg.inv(XtW @ X) @ (XtW @ y_fit)
                    intercept, slope = float(beta[0]), float(beta[1])

                    # SE of slope and p-value
                    y_hat = X @ beta
                    resid = y_fit - y_hat
                    df = max(len(y_fit) - 2, 1)
                    sigma2 = float((resid @ (W @ resid)) / (np.sum(w) - np.sum((XtW @ X).diagonal()) + 1e-12))
                    cov = np.linalg.inv(XtW @ X) * sigma2
                    se_slope = math.sqrt(max(cov[1, 1], 0.0))
                    if se_slope > 0:
                        tstat = slope / se_slope
                        try:
                            from scipy.stats import t as tdist
                            pval = 2.0 * (1.0 - tdist.cdf(abs(tstat), df))
                        except Exception:
                            pval = np.nan

                # Plot trend line
                x_line = np.array([x_arr.min(), x_arr.max()], dtype=float)
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, color=trend_color, linestyle=trend_style, linewidth=trend_width, zorder=3)

                # Annotate slope and p
                ax.annotate(
                    f"slope={slope:.3g}" + (f", p={pval:.3g}" if np.isfinite(pval) else ""),
                    xy=trend_text_loc, xycoords="axes fraction",
                    ha="left", va="top", fontsize=9, color=trend_color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                )

        # Labels
        ax.set_xticks(x)
        ax.set_xticklabels(ticklabels, rotation=45, ha="right")
        ax.set_title(title)
        ###ax.set_xlabel("Model")
        ax.set_ylabel(metric_label or metric)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    # Match y-limits across panels if sharing or if chance line suggests it
    if sharey and len(all_ylim_candidates) > 0:
        ymin = np.nanmin(all_ylim_candidates)
        ymax = np.nanmax(all_ylim_candidates)
        if chance is not None:
            ymin = np.nanmin([ymin, chance])
            ymax = np.nanmax([ymax, chance])
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            # Add small margin
            pad = 0.04 * (ymax - ymin)
            for ax in axes:
                ax.set_ylim(ymin - pad, ymax + pad)

    if suptitle:
        fig.suptitle(suptitle, y=0.98)
    # Show y-label only on first chart of each row
    for i, ax in enumerate(axes):
        col = i % ncols
        if col != 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)  # or: left=False, labelleft=False to hide ticks too
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    if outfile is None:
        safe_metric = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in metric)
        outfile = f"{safe_metric}_panels.png"
    fig.savefig(outfile, bbox_inches="tight")
    plt.show()

    return fig, axes, wide_vals_list


files = ["analysis_log_multi_logres_dg_gpqa_dg_full_hist_parsed.txt", "analysis_log_multi_logres_dg_simplemc_dg_full_hist_parsed.txt", "analysis_log_multi_logres_dg_gpsa_dg_full_hist_parsed.txt", "analysis_log_multi_logres_dg_simpleqa_dg_full_hist_parsed.txt"]
metrics = [
    "Phase 1 accuracy",
    "Delegation rate",
    "Naive Confidence",
    "Teammate-weighted confidence",
    "Raw introspection score",
    "Raw self-acc lift",
    "Correctness Coef Cntl",
    "Pseudo R2 Cntl",                # no CIs -> paired t
    "Capent Correl Cntl",            # treated as correlation (Fisher z)
    "Capent Correl Prob Cntl",       # treated as correlation (Fisher z)
    "Calibration AUC",
    "Calibration Entropy AUC",
    "ECE",
    "Brier",
    "Brier Resolution",
    "Brier Reliability",
    "Top Prob Mean",
    "Game-Stated Entropy Diff",
    "Game-Stated Confounds Diff",
]
model_order=["openai-gpt-5-chat", "claude-opus-4-1-20250805", 'claude-sonnet-4-20250514', 'grok-3-latest', 'claude-3-5-sonnet-20241022', 'gpt-4.1-2025-04-14', 'gpt-4o-2024-08-06', 'deepseek-chat', "gemini-2.5-flash_think", "gemini-2.5-flash_nothink", 'gemini-2.0-flash-001', "gemini-2.5-flash-lite_think", "gemini-2.5-flash-lite_nothink", 'gemini-1.5-pro', 'gpt-4o-mini', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']

model_aliases = {
    "openai-gpt-5-chat": "GPT-5",
    "claude-opus-4-1-20250805": "Opus 4.1",
    'claude-sonnet-4-20250514': "Sonnet 4",
    'grok-3-latest': "Grok 3",
    'claude-3-5-sonnet-20241022': "Sonnet 3.5",
    'gpt-4.1-2025-04-14': "GPT-4.1",
    'gpt-4o-2024-08-06': "GPT-4o",
    'deepseek-chat': "DeepSeek Chat",
    "gemini-2.5-flash_think": "Gem 2.5 Flash T",
    "gemini-2.5-flash_nothink": "Gem 2.5 Flash NT",
    'gemini-2.0-flash-001': "Gem 2 Flash",
    "gemini-2.5-flash-lite_think": "Gem 2.5 Flash Lite T",
    "gemini-2.5-flash-lite_nothink": "Gem 2.5 Flash Lite NT",
    'gpt-4o-mini': "GPT-4o Mini",
    'claude-3-sonnet-20240229': "Sonnet 3",
    'claude-3-haiku-20240307': "Haiku 3",
    'gemini-1.5-pro': "Gemini 1.5 Pro",
}
metric_aliases = {
    "Phase 1 accuracy": ["acc", "Baseline Accuracy (%)"],
    "Correctness Coef Cntl": ["is", "Beta"],
    "Correctness Correl Cntl": ["is_pc", "Partial correlation"],
    "Capent Correl Cntl": ["capent_correl", "Partial correlation"],
    "Capent Correl Prob Cntl": ["sc_capent_prob", "Partial correlation"],
    "Delegation rate": ["dg", "Delegation Rate (%)"],
    "Naive Confidence": ["naive_conf", "Naive Confidence (%)"],
    "Teammate-weighted confidence": ["tw_conf", "Teammate-weighted Confidence (%)"],
    "Raw introspection score": ["raw_introspec", "Raw Introspection Score"],
    "Raw self-acc lift": ["raw_lift", "Raw Self-accuracy Lift"],
    "Pseudo R2 Cntl": ["sc_pseudoR2", "Pseudo R²"],
    "Calibration AUC": ["cal_auc", "AUC"],
    "Calibration Entropy AUC": ["calib_ent_auc", "Calibration Entropy AUC"],
    "ECE": ["ece", "Expected Calibration Error (ECE)"],
    "Brier": ["brier", "Brier Score"],
    "Brier Resolution": ["brier_res", "Brier Resolution"],
    "Brier Reliability": ["brier_rel", "Brier Reliability"],
    "Top Prob Mean": ["top_prob", "Top Predicted Probability (%)"],
    "Game-Stated Entropy Diff": ["entropy_diff", "Game-Stated Entropy Diff"],
    "Game-Stated Confounds Diff": ["confounds_diff", "Game-Stated Confounds Diff"],
    "Self Other Correl": ["self_other_correl", "Self-Other Correlation"],
    "Team Accuracy Lift": ["team_acc_lift", "Team Accuracy Lift"],
    "Game-Test Change Rate": ["game_test_change_rate", "Game-Test Change Rate (%)"],
    "Capent Gament Correl": ["capent_game_correl", "Game-Capability Correlation"],
    "Optimal Decision Rate": ["optimal_decision_rate", "Optimal Decision Rate (%)"],
    "Unweighted Confidence": ["unweighted_conf", "Unweighted Confidence (%)"],
    "Weighted Confidence": ["weighted_conf", "Weighted Confidence (%)"],
    "Controls Correl": ["controls_correl", "Multi Partial Correlation"],
}
chance = None#0.5
show_trend = False
metric = "Capent Correl Cntl"
pooled = None#pool_metric_ivw(metric=metric, files=files[:2])#["analysis_log_multi_logres_dg_gpqa_dg_full_hist_parsed.txt", "analysis_log_multi_logres_dg_simplemc_dg_full_hist_parsed.txt"])
if pooled: files = [{model: {metric: stats} for model, stats in pooled.items()}]

fig, ax, df_wide = plot_metric_panels_from_results(
    metric=metric,
    files=files,
    series_names=["All Combined"] if pooled else ["GPQA", "SimpleMC", "GPSA", "SimpleQA"],#["GPQA-SimpleMC Combined"],#
    model_order=model_order,
    aliases=model_aliases,
    suptitle="",#"Partial correlation between baseline correctness and answer/delegate decision",
    outfile=f"{metric_aliases[metric][0]}_by_model{'_notrend' if not show_trend else ''}{'_pooled' if pooled else ''}.png"
    , ecolor="gray", alpha_err=1.0, chance=chance, metric_label=metric_aliases[metric][1], show_trend=show_trend, bar_color=None##="#84bdf7"#="#f5b446"#="#2fae2d"='brown'#
)
