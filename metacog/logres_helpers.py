import statsmodels.stats.proportion as smp
import scipy.stats as ss
import numpy as np
import re

import math
from scipy.stats import binomtest, rankdata, pointbiserialr, norm, pearsonr, chi2
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from numpy.linalg import LinAlgError

from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm

import scipy.linalg as la
from patsy import dmatrices

def analyze_wrong_way(
    df,
    continuous_controls,     # list of names or Series 
    categorical_controls=None,  # list of names or Series
    normvars=True,
    robust=False,
    alpha=0.05
):
    """
    Build controls, fit two GLM logits, and either return results
    or raise a single, explanatory error if something is off.
    """
    # Predeclare for diagnostics
    controls = None
    Xb = None
    Xg = None
    yb = None
    yg = None

    try:
        # 1) Sanity: required columns and binary outcomes
        if 's_i_capability' not in df or 'delegate_choice' not in df:
            missing = [c for c in ['s_i_capability','delegate_choice'] if c not in df]
            raise ValueError(f"Missing required columns: {missing}")
        for col in ['s_i_capability', 'delegate_choice']:
            vals = pd.Series(df[col]).dropna().unique()
            if not set(vals).issubset({0,1}):
                raise ValueError(f"Outcome {col} must be binary 0/1; found values: {vals[:10]}")

        # 2) Build control matrix on FULL df (same pattern as your entropy code)
        controls = pd.DataFrame(index=df.index)

        # Continuous controls
        cont_names = []
        if continuous_controls:
            for i, ctrl in enumerate(continuous_controls):
                if isinstance(ctrl, str):
                    s = df[ctrl]
                    cname = ctrl
                else:
                    s = pd.Series(ctrl, index=df.index)
                    cname = s.name or f'cont_{i}'
                controls[cname] = pd.to_numeric(s, errors='coerce')
                cont_names.append(cname)

        # Categorical controls
        cat_names = []
        if categorical_controls:
            for i, ctrl in enumerate(categorical_controls):
                if isinstance(ctrl, str):
                    s = df[ctrl]
                    cname = ctrl
                else:
                    s = pd.Series(ctrl, index=df.index)
                    cname = s.name or f'cat_{i}'
                controls[cname] = s.astype('object')
                cat_names.append(cname)

        # One-hot on full df
        if cat_names:
            dummies = pd.get_dummies(controls[cat_names], drop_first=True, dtype=float)
            controls = pd.concat([controls.drop(columns=cat_names), dummies], axis=1)

        # 3) Build designs per outcome (subset, drop NA rows, scale continuous)
        def _design(ycol):
            idx = df[ycol].notna()
            y = df.loc[idx, ycol].astype(int)
            X = controls.loc[idx].copy()

            # Report and drop rows with any NA in predictors
            na_by_col = X.isna().sum()
            if na_by_col.any():
                X = X.dropna(axis=0)
                y = y.loc[X.index]

            # Standardize continuous controls on this subset (like your entropy code)
            if normvars and cont_names:
                present_cont = [c for c in cont_names if c in X.columns]
                if present_cont:
                    scaler = StandardScaler().fit(X[present_cont])
                    X[present_cont] = scaler.transform(X[present_cont])

            # Final dtype enforcement
            non_numeric = [c for c, dt in X.dtypes.items() if dt.kind not in 'fc']
            if non_numeric:
                raise TypeError(f"Non-numeric columns after setup: {non_numeric}")

            # Add intercept
            X = sm.add_constant(X.astype(float), has_constant='add')
            return X, y, na_by_col

        Xb, yb, na_b = _design('s_i_capability')
        Xg, yg, na_g = _design('delegate_choice')

        # 4) Check column alignment; align if needed but report differences
        xb_cols = list(Xb.columns)
        xg_cols = list(Xg.columns)
        if xb_cols != xg_cols:
            missing_in_game = sorted(list(set(xb_cols) - set(xg_cols)))
            missing_in_base = sorted(list(set(xg_cols) - set(xb_cols)))
            raise ValueError(
                "Predictor columns differ between baseline and game after setup. "
                f"Missing in game: {missing_in_game}; missing in baseline: {missing_in_base}. "
                "This means some dummy or control column is present only in one subset after row drops."
            )

        # 5) Constant columns (no variance) in each subset
        const_b = [c for c in Xb.columns if c != 'const' and Xb[c].nunique(dropna=False) <= 1]
        const_g = [c for c in Xg.columns if c != 'const' and Xg[c].nunique(dropna=False) <= 1]
        # We keep them (statsmodels can handle with warnings), but we’ll surface them if fit fails.

        # 6) Fit GLMs (logit)
        fam = sm.families.Binomial()
        base_res = sm.GLM(yb, Xb, family=fam).fit(cov_type='HC1' if robust else 'nonrobust')
        game_res = sm.GLM(yg, Xg, family=fam).fit(cov_type='HC1' if robust else 'nonrobust')

        # 7) Summaries for misuse rule
        bp = base_res.params.drop('const', errors='ignore')
        gp = game_res.params.drop('const', errors='ignore')
        bz = bp / base_res.bse.drop('const', errors='ignore')
        gz = gp / game_res.bse.drop('const', errors='ignore')
        p_one = pd.Series(1 - norm.cdf(gz.values), index=gz.index, name='p_one')

        candidates = set(bp[bp > 0].index.tolist())

        rows = []
        for j in bp.index.union(gp.index):
            misuse = (j in candidates) and (gp.get(j, np.nan) > 0) and (p_one.get(j, np.nan) < alpha)
            rows.append({
                'predictor': j,
                'beta_correct': bp.get(j, np.nan),
                'z_correct': bz.get(j, np.nan),
                'beta_delegate': gp.get(j, np.nan),
                'z_delegate': gz.get(j, np.nan),
                'p_one_sided_delegate_gt0': p_one.get(j, np.nan),
                'baseline_positive': (j in candidates),
                'misuse': misuse
            })
        return pd.DataFrame(rows).sort_values(['misuse','p_one_sided_delegate_gt0'], ascending=[False, True]), {
            'baseline': base_res, 'game': game_res
        }

    except Exception as e:
        # Build a precise, actionable message about what and where
        lines = []
        lines.append("Failed to fit models with entropy-style setup.")
        lines.append(f"Error: {type(e).__name__}: {e}")

        # Controls-level diagnostics
        if controls is None:
            lines.append("Controls: not constructed (error occurred earlier).")
        else:
            obj_cols = [c for c, dt in controls.dtypes.items() if dt == 'object']
            if obj_cols:
                # Show a peek of unique values to catch unencoded categoricals
                peek = {c: controls[c].dropna().unique()[:5].tolist() for c in obj_cols}
                lines.append(f"Controls contain object dtype columns (should be dummies or numeric): {obj_cols}. Samples: {peek}")

            na_totals = controls.isna().sum()
            na_cols = na_totals[na_totals > 0].sort_values(ascending=False)
            if len(na_cols) > 0:
                lines.append(f"Controls have NaNs before subsetting: {na_cols.to_dict()}")

        # Subset-level diagnostics
        def _subset_diags(name, X, y):
            if X is None:
                lines.append(f"{name}: design not built.")
                return
            lines.append(f"{name}: X shape {X.shape}, y length {len(y) if y is not None else 'n/a'}")
            nn = [c for c, dt in X.dtypes.items() if dt.kind not in 'fc']
            if nn:
                lines.append(f"{name}: non-numeric columns: {nn}")
            na_by_col = X.isna().sum()
            na_cols = na_by_col[na_by_col > 0]
            if len(na_cols) > 0:
                lines.append(f"{name}: NaNs in predictors by column: {na_cols.to_dict()}")
            consts = [c for c in X.columns if c != 'const' and X[c].nunique(dropna=False) <= 1]
            if consts:
                lines.append(f"{name}: constant columns (no variance): {consts}")

        _subset_diags("Baseline", Xb, yb)
        _subset_diags("Game", Xg, yg)

        # Column alignment
        if (Xb is not None) and (Xg is not None):
            if list(Xb.columns) != list(Xg.columns):
                missing_in_game = sorted(list(set(Xb.columns) - set(Xg.columns)))
                missing_in_base = sorted(list(set(Xg.columns) - set(Xb.columns)))
                lines.append(f"Column mismatch: missing in game {missing_in_game}; missing in baseline {missing_in_base}")

        raise RuntimeError("\n".join(lines)) from e
    
def plot_x3_relationships(X1, X2, X3, y, filename='x3_relationships.png',
                          n_bins=20, loess_frac=0.3, dpi=160):
    """
    Create simple diagnostic plots:
      - X1 vs X3: scatter with LOESS smoother
      - X2 vs X3: scatter with LOESS smoother
      - y vs X3: binned means (with 95% CI) and a logistic-spline fit (fallback to LOESS if GLM fails)
    Saves the figure to `filename`.

    Parameters
    - X1, X2, X3, y: array-like or pandas Series (y should be binary 0/1)
    - filename: output file path for the saved figure
    - n_bins: number of quantile bins for the binned-means plot
    - loess_frac: smoothing parameter for LOESS (0<frac<=1)
    - dpi: figure DPI for saving

    Returns
    - filename (string)
    """
    # Assemble a clean DataFrame and drop missing values
    df = pd.DataFrame({
        'X1': pd.Series(X1, dtype='float'),
        'X2': pd.Series(X2, dtype='float'),
        'X3': pd.Series(X3, dtype='float'),
        'y':  pd.Series(y).astype(int)
    }).dropna(subset=['X1','X2','X3','y'])

    if df.empty:
        raise ValueError("No data left after dropping missing values.")

    # Basic style
    sns.set_theme(context='notebook', style='whitegrid')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # Panel 1: X1 vs X3 (scatter + LOESS)
    ax = axes[0]
    ax.scatter(df['X3'], df['X1'], s=15, alpha=0.35, color='#1f77b4', edgecolor='none')
    # LOESS
    try:
        lo = lowess(df['X1'], df['X3'], frac=loess_frac, it=0, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], color='crimson', lw=2, label='LOESS')
    except Exception:
        pass
    ax.set_xlabel('X3 (entropy)')
    ax.set_ylabel('X1 (self-reported confidence)')
    ax.set_title('X1 vs X3')
    ax.legend(loc='best', frameon=True)

    # Panel 2: X2 vs X3 (scatter + LOESS)
    ax = axes[1]
    ax.scatter(df['X3'], df['X2'], s=15, alpha=0.35, color='#1f77b4', edgecolor='none')
    try:
        lo = lowess(df['X2'], df['X3'], frac=loess_frac, it=0, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], color='crimson', lw=2, label='LOESS')
    except Exception:
        pass
    ax.set_xlabel('X3 (entropy)')
    ax.set_ylabel('X2 (confidence in others)')
    ax.set_title('X2 vs X3')
    ax.legend(loc='best', frameon=True)

    # Panel 3: y vs X3 (binned means + logistic-spline or LOESS)
    ax = axes[2]

    # Binned means (quantile bins; fall back to equal-width if many ties)
    x = df['X3'].values
    yb = df['y'].values

    try:
        q_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(q_edges)
        if len(edges) < 4:
            raise ValueError("Not enough unique quantile edges.")
    except Exception:
        edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)

    # Assign bins and compute summaries
    bins = pd.cut(df['X3'], edges, include_lowest=True)
    grp = df.groupby(bins, observed=False)
    b_mean_x = grp['X3'].mean()
    b_mean_y = grp['y'].mean()
    b_n = grp['y'].size()
    # Binomial 95% CI via normal approx: p +/- 1.96*sqrt(p*(1-p)/n)
    with np.errstate(invalid='ignore'):
        se = np.sqrt(np.maximum(b_mean_y * (1 - b_mean_y) / b_n, 0))
    yerr = 1.96 * se

    ax.errorbar(b_mean_x.values, b_mean_y.values, yerr=yerr.values,
                fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=1,
                capsize=2, alpha=0.9, label='Binned means (95% CI)')

    # Logistic spline fit (fallback to LOESS if GLM fails)
    grid = np.linspace(np.nanmin(x), np.nanmax(x), 300)

    added_logit = False
    try:
        # Natural cubic spline via patsy (statsmodels dependency)
        from patsy import dmatrix
        X_spline = dmatrix("bs(x, df=4, degree=3, include_intercept=False)",
                           {"x": df['X3']}, return_type='dataframe')
        model = sm.GLM(df['y'], X_spline, family=sm.families.Binomial())
        res = model.fit()
        Xg = dmatrix("bs(x, df=4, degree=3, include_intercept=False)",
                     {"x": grid}, return_type='dataframe')
        pred = res.predict(Xg)
        ax.plot(grid, pred, color='crimson', lw=2.2, label='Logit spline')
        added_logit = True
    except Exception:
        # Fallback: LOESS on y vs X3
        try:
            lo = lowess(df['y'], df['X3'], frac=loess_frac, it=0, return_sorted=True)
            ax.plot(lo[:, 0], lo[:, 1], color='crimson', lw=2.2, label='LOESS')
        except Exception:
            pass

    ax.set_xlabel('X3 (entropy)')
    ax.set_ylabel('y (game success probability)')
    ax.set_title('y vs X3' + (' (logit spline)' if added_logit else ''))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', frameon=True)

    # Overall title and save
    fig.suptitle('Relationships with X3 (entropy): scatter/LOESS and y-binned means', y=1.02, fontsize=13)
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return filename


def tjur_R2_from_probs(p, y):
    y = np.asarray(y).astype(int)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return np.nan
    return float(p[y==1].mean() - p[y==0].mean())

def fit_logit_probs(X, y):
    """
    Try statsmodels Logit; if it fails (e.g., perfect separation), fallback to sklearn.
    Returns predicted probabilities on X.
    """
    try:
        mod = sm.Logit(y, sm.add_constant(X, has_constant='add'))
        res = mod.fit(disp=0)
        p = res.predict(sm.add_constant(X, has_constant='add'))
        return np.asarray(p), 'sm'
    except Exception:
        # sklearn fallback, near-unpenalized
        clf = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000)
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        return p, 'sk'

def compute_effects_once(df, X1_col='X1', X2_col='X2', X3_col='X3', y_col='y', controls=None):
    """
    Compute the two effect sizes on the provided dataframe.
    controls: list of additional control column names (optional).
    """
    # Build design matrices
    base_cols = [X2_col] + (controls if controls else [])
    full_cols = base_cols + [X3_col]
    
    # OLS: partial R^2 of X3 for X1 controlling for base_cols
    Xb = sm.add_constant(df[base_cols], has_constant='add')
    Xf = sm.add_constant(df[full_cols], has_constant='add')
    y_X1 = df[X1_col].values
    
    ols_b = sm.OLS(y_X1, Xb).fit()
    ols_f = sm.OLS(y_X1, Xf).fit()
    ssr_b = float(ols_b.ssr)
    ssr_f = float(ols_f.ssr)
    partial_R2 = np.nan
    if ssr_b > 1e-12:
        partial_R2 = 1.0 - (ssr_f / ssr_b)
    
    # Logit: delta Tjur's R^2 of adding X3 for y controlling for base_cols
    y_bin = df[y_col].astype(int).values
    # Base
    p_b, _ = fit_logit_probs(df[base_cols].values, y_bin)
    tjur_b = tjur_R2_from_probs(p_b, y_bin)
    # Full
    p_f, _ = fit_logit_probs(df[full_cols].values, y_bin)
    tjur_f = tjur_R2_from_probs(p_f, y_bin)
    delta_tjur = tjur_f - tjur_b if (not np.isnan(tjur_b) and not np.isnan(tjur_f)) else np.nan
    
    return partial_R2, delta_tjur

def bootstrap_effect_sizes(
    df,
    X1_col='X1', X2_col='X2', X3_col='X3', y_col='y',
    controls=None,
    n_boot=2000,
    stratify=True,
    random_state=42,
    ci=(2.5, 97.5),
):
    """
    df: dataframe with columns X1, X2, X3, y (and optional controls).
    controls: list of additional control column names to include in base and full models.
    Returns point estimates and percentile CIs for both effect sizes and their gap.
    """
    rng = np.random.default_rng(random_state)
    df = df.dropna(subset=[X1_col, X2_col, X3_col, y_col] + (controls if controls else []))
    
    # Point estimates on the full sample
    est_partial_R2, est_delta_tjur = compute_effects_once(
        df, X1_col=X1_col, X2_col=X2_col, X3_col=X3_col, y_col=y_col, controls=controls
    )
    
    # Prepare bootstrap sampling
    n = len(df)
    idx = np.arange(n)
    if stratify:
        y = df[y_col].astype(int).values
        idx_pos = idx[y == 1]
        idx_neg = idx[y == 0]
        n_pos, n_neg = len(idx_pos), len(idx_neg)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Cannot stratify bootstrap: one class has zero instances.")
    
    boot_partial = np.empty(n_boot, dtype=float)
    boot_delta_tjur = np.empty(n_boot, dtype=float)
    boot_partial.fill(np.nan)
    boot_delta_tjur.fill(np.nan)
    
    for b in range(n_boot):
        # Sample indices
        if stratify:
            samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
            samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
            samp_idx = np.concatenate([samp_pos, samp_neg])
        else:
            samp_idx = rng.choice(idx, size=n, replace=True)
        
        df_b = df.iloc[samp_idx].reset_index(drop=True)
        
        try:
            pR2_b, dTjur_b = compute_effects_once(
                df_b, X1_col=X1_col, X2_col=X2_col, X3_col=X3_col, y_col=y_col, controls=controls
            )
            boot_partial[b] = pR2_b
            boot_delta_tjur[b] = dTjur_b
        except Exception:
            # Leave NaN for this replicate
            continue
    
    # Drop NaNs (e.g., rare failures)
    bp = boot_partial[~np.isnan(boot_partial)]
    bt = boot_delta_tjur[~np.isnan(boot_delta_tjur)]
    gap = bt - bp[:len(bt)] if len(bt) == len(bp) else None  # align lengths if equal; otherwise skip
    
    def pct_ci(arr, lo, hi):
        if len(arr) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))
    
    lo, hi = ci
    ci_partial = pct_ci(bp, lo, hi)
    ci_delta_tjur = pct_ci(bt, lo, hi)
    ci_gap = pct_ci(gap, lo, hi) if gap is not None else (np.nan, np.nan)
    
    return {
        'point_estimates': {
            'partial_R2_X3_on_X1_ctrl': float(est_partial_R2),
            'delta_TjurR2_X3_on_y_ctrl': float(est_delta_tjur),
            'gap_deltaTjur_minus_partialR2': float(est_delta_tjur - est_partial_R2) if (not np.isnan(est_delta_tjur) and not np.isnan(est_partial_R2)) else np.nan,
        },
        'bootstrap_CI_percentile': {
            'partial_R2_X3_on_X1_ctrl': {'lo': ci_partial[0], 'hi': ci_partial[1]},
            'delta_TjurR2_X3_on_y_ctrl': {'lo': ci_delta_tjur[0], 'hi': ci_delta_tjur[1]},
            'gap_deltaTjur_minus_partialR2': {'lo': ci_gap[0], 'hi': ci_gap[1]},
        },
        'n_boot_effective': int(min(len(bp), len(bt))),
    }

def compare_predictors_of_choice_simple(
    X1, X2, X3, y,
    continuous_controls=None,
    categorical_controls=None,
    normvars=True,
    n_boot=2000,
    random_state=123
):
    """
    Minimal, comparable metrics:
      - Unadjusted influence (no controls): r(X3, X1), r(X3, X2), r(X3, y) with 95% CI and p-value.
      - Adjusted influence (with controls):
          * X1: partial r(X3, X1 | surface controls + X2)
          * X2: partial r(X3, X2 | surface controls only)
          * y : partial r(X3, y  | surface controls + X2), computed via residualization (LPM-style)
        Each with 95% CI (Fisher z with k-adjustment) and p-value (t-test for partial r).
      - Bootstrap CIs for differences:
          * Unadjusted: r(X3,y) - r(X3,X1); r(X3,y) - r(X3,X2); r(X3,X1) - r(X3,X2)
          * Adjusted: same, but using the partial correlations defined above.

    Returns:
      ret_str (plain text summary) and results_dict (JSON-friendly metrics).
    """

    ret_str = ""
    results_dict = {}

    try:
        # -----------------------------
        # Setup
        # -----------------------------
        original_names = {
            'X1': X1.name or 'X1',
            'X2': X2.name or 'X2',
            'X3': X3.name or 'X3',
            'y': y.name or 'y'
        }

        # Base frame
        df = pd.DataFrame({
            'X1': np.asarray(X1).astype(float),
            'X2': np.asarray(X2).astype(float),
            'X3': np.asarray(X3).astype(float),
            'y': np.asarray(y).astype(int)
        })

        # Add controls if provided
        cont_control_names = []
        if continuous_controls:
            for i, ctrl in enumerate(continuous_controls):
                cname = ctrl.name or f'cont_control_{i}'
                df[cname] = np.asarray(ctrl).astype(float)
                cont_control_names.append(cname)

        cat_control_names = []
        if categorical_controls:
            for i, ctrl in enumerate(categorical_controls):
                cname = ctrl.name or f'cat_control_{i}'
                df[cname] = np.asarray(ctrl)
                cat_control_names.append(cname)

        # Drop rows with NaNs in core vars first
        df = df.dropna(subset=['X1', 'X2', 'X3', 'y'])

        # One-hot encode categorical controls once on the full sample
        dummy_cols = []
        if cat_control_names:
            dummies = pd.get_dummies(df[cat_control_names], drop_first=True, dtype=float)
            dummy_cols = list(dummies.columns)
            df = pd.concat([df.drop(columns=cat_control_names), dummies], axis=1)

        # Standardize X1, X2, X3 and continuous controls (not dummies, not y)
        df_norm = df.copy()
        if normvars:
            to_scale = ['X1', 'X2', 'X3'] + cont_control_names
            if to_scale:
                scaler = StandardScaler().fit(df_norm[to_scale])
                df_norm[to_scale] = scaler.transform(df_norm[to_scale])

        surface_controls = cont_control_names + dummy_cols  # exogenous controls

        # For consistency across comparisons, define analysis frames:
        # - Unadjusted: require only core vars
        df_unadj = df_norm.dropna(subset=['X1', 'X2', 'X3', 'y']).copy()
        # - Adjusted (common sample): require core vars + all surface controls + X2 (since 2 and 6 include X2)
        df_adj_common = df_norm.dropna(subset=['X1', 'X2', 'X3', 'y'] + surface_controls).copy()

        # -----------------------------
        # Helper functions
        # -----------------------------
        def fisher_ci(r, n, k_controls=0, alpha=0.05):
            """
            Fisher z CI for (partial) correlation.
            For partial correlation with k controls, SE_z = 1 / sqrt(n - k - 3).
            """
            r = float(r)
            if np.isnan(r) or n is None:
                return (np.nan, np.nan)
            # Clamp to avoid infs
            r_clamped = np.clip(r, -0.999999, 0.999999)
            z = np.arctanh(r_clamped)
            se = 1.0 / np.sqrt(max(n - k_controls - 3, 1e-8))
            z_lo = z + stats.norm.ppf(alpha / 2.0) * se
            z_hi = z + stats.norm.ppf(1 - alpha / 2.0) * se
            return (float(np.tanh(z_lo)), float(np.tanh(z_hi)))

        def corr_with_p(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            n = len(x)
            if n != len(y) or n < 3:
                return np.nan, np.nan, np.nan, np.nan
            # Handle zero-variance cases
            sx = np.nanstd(x)
            sy = np.nanstd(y)
            if sx == 0 or sy == 0:
                return np.nan, np.nan, np.nan, n
            r = float(np.corrcoef(x, y)[0, 1])
            # t-test for Pearson r
            df_t = n - 2
            if abs(r) >= 1.0:
                p = 0.0
            else:
                t = r * np.sqrt(df_t / max(1 - r**2, 1e-12))
                p = 2 * stats.t.sf(abs(t), df=df_t)
            return r, p, n, df_t

        def partial_corr(df_in, var_y, var_x, controls):
            """
            Residualize var_y and var_x on controls (w/ intercept) and correlate residuals.
            Returns: r_partial, p_value, n, k_controls
            """
            cols_needed = [var_y, var_x] + (controls if controls else [])
            d = df_in.dropna(subset=cols_needed)
            n = len(d)
            k = len(controls) if controls else 0
            if n < (k + 4):  # need at least k+4 to have df>=2
                return np.nan, np.nan, n, k

            # Design matrices with intercept
            Xc = sm.add_constant(d[controls], has_constant='add') if k > 0 else np.ones((n, 1))
            # OLS residuals
            if k > 0:
                res_y = sm.OLS(d[var_y].values, Xc).fit().resid
                res_x = sm.OLS(d[var_x].values, Xc).fit().resid
            else:
                # No controls: just center
                res_y = d[var_y].values - d[var_y].values.mean()
                res_x = d[var_x].values - d[var_x].values.mean()

            r = float(np.corrcoef(res_x, res_y)[0, 1])
            # p-value for partial correlation via t-test with df = n - k - 2
            df_t = n - k - 2
            if np.isnan(r) or df_t <= 0:
                return np.nan, np.nan, n, k
            if abs(r) >= 1.0:
                p = 0.0
            else:
                t = r * np.sqrt(df_t / max(1 - r**2, 1e-12))
                p = 2 * stats.t.sf(abs(t), df=df_t)
            return r, p, n, k

        def bootstrap_diff_corrs_unadjusted(df_in, n_boot, rng):
            """
            Bootstrap percentile CIs for differences of unadjusted correlations:
              dy-dx1, dy-dx2, dx1-dx2
            Stratified by y to stabilize.
            """
            d = df_in.dropna(subset=['X1', 'X2', 'X3', 'y'])
            if len(d) < 10:
                return None  # too small

            idx = np.arange(len(d))
            yv = d['y'].astype(int).values
            idx_pos = idx[yv == 1]
            idx_neg = idx[yv == 0]
            n_pos, n_neg = len(idx_pos), len(idx_neg)
            if n_pos == 0 or n_neg == 0:
                # Fallback: non-stratified
                idx_pos, idx_neg = None, None

            diffs = {'dy_minus_dx1': [], 'dy_minus_dx2': [], 'dx1_minus_dx2': []}

            for _ in range(n_boot):
                if idx_pos is not None:
                    samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
                    samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
                    samp_idx = np.concatenate([samp_pos, samp_neg])
                else:
                    samp_idx = rng.choice(idx, size=len(idx), replace=True)
                bs = d.iloc[samp_idx]

                r_x3_y, _, _, _ = corr_with_p(bs['X3'], bs['y'])
                r_x3_x1, _, _, _ = corr_with_p(bs['X3'], bs['X1'])
                r_x3_x2, _, _, _ = corr_with_p(bs['X3'], bs['X2'])

                if not np.isnan(r_x3_y) and not np.isnan(r_x3_x1):
                    diffs['dy_minus_dx1'].append(r_x3_y - r_x3_x1)
                if not np.isnan(r_x3_y) and not np.isnan(r_x3_x2):
                    diffs['dy_minus_dx2'].append(r_x3_y - r_x3_x2)
                if not np.isnan(r_x3_x1) and not np.isnan(r_x3_x2):
                    diffs['dx1_minus_dx2'].append(r_x3_x1 - r_x3_x2)

            out = {}
            for k, arr in diffs.items():
                arr = np.array(arr, dtype=float)
                if len(arr) == 0:
                    out[k] = {'lo': np.nan, 'hi': np.nan, 'point': np.nan, 'p_boot': np.nan, 'n_boot': 0}
                else:
                    lo, hi = np.percentile(arr, [2.5, 97.5])
                    # Simple two-sided sign test p from bootstrap
                    p_boot = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
                    out[k] = {'lo': float(lo), 'hi': float(hi), 'point': float(np.mean(arr)),
                              'p_boot': float(p_boot), 'n_boot': int(len(arr))}
            return out

        def bootstrap_diff_corrs_adjusted(df_in, n_boot, rng, surface_controls):
            """
            Bootstrap percentile CIs for differences of partial correlations (adjusted):
              dy-dx1, dy-dx2, dx1-dx2
            Uses common sample df_in and control sets:
              - For X1: controls = surface_controls + ['X2']
              - For X2: controls = surface_controls
              - For y : controls = surface_controls + ['X2']
            """
            needed_cols = ['X1', 'X2', 'X3', 'y'] + surface_controls
            d = df_in.dropna(subset=needed_cols)
            if len(d) < 10:
                return None

            idx = np.arange(len(d))
            yv = d['y'].astype(int).values
            idx_pos = idx[yv == 1]
            idx_neg = idx[yv == 0]
            n_pos, n_neg = len(idx_pos), len(idx_neg)
            if n_pos == 0 or n_neg == 0:
                idx_pos, idx_neg = None, None

            diffs = {'dy_minus_dx1': [], 'dy_minus_dx2': [], 'dx1_minus_dx2': []}

            for _ in range(n_boot):
                if idx_pos is not None:
                    samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
                    samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
                    samp_idx = np.concatenate([samp_pos, samp_neg])
                else:
                    samp_idx = rng.choice(idx, size=len(idx), replace=True)
                bs = d.iloc[samp_idx]

                r_x3_x1, _, _, _ = partial_corr(bs, 'X1', 'X3', surface_controls + ['X2'])
                r_x3_x2, _, _, _ = partial_corr(bs, 'X2', 'X3', surface_controls)
                r_x3_y,  _, _, _ = partial_corr(bs, 'y',  'X3', surface_controls + ['X2'])

                if not np.isnan(r_x3_y) and not np.isnan(r_x3_x1):
                    diffs['dy_minus_dx1'].append(r_x3_y - r_x3_x1)
                if not np.isnan(r_x3_y) and not np.isnan(r_x3_x2):
                    diffs['dy_minus_dx2'].append(r_x3_y - r_x3_x2)
                if not np.isnan(r_x3_x1) and not np.isnan(r_x3_x2):
                    diffs['dx1_minus_dx2'].append(r_x3_x1 - r_x3_x2)

            out = {}
            for k, arr in diffs.items():
                arr = np.array(arr, dtype=float)
                if len(arr) == 0:
                    out[k] = {'lo': np.nan, 'hi': np.nan, 'point': np.nan, 'p_boot': np.nan, 'n_boot': 0}
                else:
                    lo, hi = np.percentile(arr, [2.5, 97.5])
                    p_boot = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
                    out[k] = {'lo': float(lo), 'hi': float(hi), 'point': float(np.mean(arr)),
                              'p_boot': float(p_boot), 'n_boot': int(len(arr))}
            return out

        # -----------------------------
        # Unadjusted correlations
        # -----------------------------
        r_x3_x1, p_x3_x1, n_u, _ = corr_with_p(df_unadj['X3'], df_unadj['X1'])
        r_x3_x2, p_x3_x2, _, _ = corr_with_p(df_unadj['X3'], df_unadj['X2'])
        r_x3_y,  p_x3_y,  _, _ = corr_with_p(df_unadj['X3'], df_unadj['y'])
        ci_x3_x1 = fisher_ci(r_x3_x1, n_u)
        ci_x3_x2 = fisher_ci(r_x3_x2, n_u)
        ci_x3_y  = fisher_ci(r_x3_y,  n_u)

        results_dict['unadjusted_influence'] = {
            f"{original_names['X3']}_vs_{original_names['X1']}": {
                'r': float(r_x3_x1), 'p': float(p_x3_x1), 'n': int(n_u),
                'ci_lo': float(ci_x3_x1[0]), 'ci_hi': float(ci_x3_x1[1])
            },
            f"{original_names['X3']}_vs_{original_names['X2']}": {
                'r': float(r_x3_x2), 'p': float(p_x3_x2), 'n': int(n_u),
                'ci_lo': float(ci_x3_x2[0]), 'ci_hi': float(ci_x3_x2[1])
            },
            f"{original_names['X3']}_vs_{original_names['y']}": {
                'r': float(r_x3_y), 'p': float(p_x3_y), 'n': int(n_u),
                'ci_lo': float(ci_x3_y[0]), 'ci_hi': float(ci_x3_y[1])
            }
        }

        ret_str += "Unadjusted influence (Pearson r; 95% CI; p-value):\n"
        ret_str += f"  {original_names['X3']} vs {original_names['X1']}: r={r_x3_x1:.4f}, CI[{ci_x3_x1[0]:.4f},{ci_x3_x1[1]:.4f}], p={p_x3_x1:.4g}, n={n_u}\n"
        ret_str += f"  {original_names['X3']} vs {original_names['X2']}: r={r_x3_x2:.4f}, CI[{ci_x3_x2[0]:.4f},{ci_x3_x2[1]:.4f}], p={p_x3_x2:.4g}, n={n_u}\n"
        ret_str += f"  {original_names['X3']} vs {original_names['y']}: r={r_x3_y:.4f}, CI[{ci_x3_y[0]:.4f},{ci_x3_y[1]:.4f}], p={p_x3_y:.4g}, n={n_u}\n\n"

        # -----------------------------
        # Adjusted partial correlations
        # -----------------------------
        # Control sets:
        controls_x1 = surface_controls + ['X2']  # for 2)
        controls_x2 = surface_controls           # for 4)
        controls_y  = surface_controls + ['X2']  # for 6)
        controls_y_b  = surface_controls + ['X2'] + ['X1']  # for 6)

        rpa_x3_x1, ppa_x3_x1, n_a, k1 = partial_corr(df_adj_common, 'X1', 'X3', controls_x1)
        rpa_x3_x2, ppa_x3_x2, _,   k2 = partial_corr(df_adj_common, 'X2', 'X3', controls_x2)
        rpa_x3_y,  ppa_x3_y,  _,   k3 = partial_corr(df_adj_common, 'y',  'X3', controls_y)
        # Also compute with X1 added to controls for y (sanity check; should be similar)
        rpa_x3_y_b, ppa_x3_y_b, _, k3b = partial_corr(df_adj_common, 'y', 'X3', controls_y_b)

        ci_pa_x3_x1 = fisher_ci(rpa_x3_x1, n_a, k_controls=k1)
        ci_pa_x3_x2 = fisher_ci(rpa_x3_x2, n_a, k_controls=k2)
        ci_pa_x3_y  = fisher_ci(rpa_x3_y,  n_a, k_controls=k3)
        ci_pa_x3_y_b = fisher_ci(rpa_x3_y_b, n_a, k_controls=k3b)

        results_dict['adjusted_influence'] = {
            f"{original_names['X3']}_vs_{original_names['X1']}_ctrl_surface+{original_names['X2']}": {
                'partial_r': float(rpa_x3_x1), 'p': float(ppa_x3_x1), 'n': int(n_a), 'k_controls': int(k1),
                'ci_lo': float(ci_pa_x3_x1[0]), 'ci_hi': float(ci_pa_x3_x1[1]),
                'controls': controls_x1
            },
            f"{original_names['X3']}_vs_{original_names['X2']}_ctrl_surface": {
                'partial_r': float(rpa_x3_x2), 'p': float(ppa_x3_x2), 'n': int(n_a), 'k_controls': int(k2),
                'ci_lo': float(ci_pa_x3_x2[0]), 'ci_hi': float(ci_pa_x3_x2[1]),
                'controls': controls_x2
            },
            f"{original_names['X3']}_vs_{original_names['y']}_ctrl_surface+{original_names['X2']}": {
                'partial_r': float(rpa_x3_y), 'p': float(ppa_x3_y), 'n': int(n_a), 'k_controls': int(k3),
                'ci_lo': float(ci_pa_x3_y[0]), 'ci_hi': float(ci_pa_x3_y[1]),
                'controls': controls_y
            },
            f"{original_names['X3']}_vs_{original_names['y']}_ctrl_surface+{original_names['X2']}+{original_names['X1']}": {
                'partial_r': float(rpa_x3_y_b), 'p': float(ppa_x3_y_b), 'n': int(n_a), 'k_controls': int(k3b),
                'ci_lo': float(ci_pa_x3_y_b[0]), 'ci_hi': float(ci_pa_x3_y_b[1]),
                'controls': controls_y_b
            }
        }

        ret_str += "Adjusted influence (partial r; 95% CI; p-value). Common sample with surface controls available:\n"
        ret_str += f"  {original_names['X3']} vs {original_names['X1']} | surface + {original_names['X2']}: "
        ret_str += f"partial r={rpa_x3_x1:.4f}, CI[{ci_pa_x3_x1[0]:.4f},{ci_pa_x3_x1[1]:.4f}], p={ppa_x3_x1:.4g}, n={n_a}\n"
        ret_str += f"  {original_names['X3']} vs {original_names['X2']} | surface only: "
        ret_str += f"partial r={rpa_x3_x2:.4f}, CI[{ci_pa_x3_x2[0]:.4f},{ci_pa_x3_x2[1]:.4f}], p={ppa_x3_x2:.4g}, n={n_a}\n"
        ret_str += f"  {original_names['X3']} vs {original_names['y']} | surface + {original_names['X2']}: "
        ret_str += f"partial r={rpa_x3_y:.4f}, CI[{ci_pa_x3_y[0]:.4f},{ci_pa_x3_y[1]:.4f}], p={ppa_x3_y:.4g}, n={n_a}\n\n"
        ret_str += f"{original_names['X3']} vs {original_names['y']} | surface + {original_names['X2']} + {original_names['X1']}: "
        ret_str += f"partial r={rpa_x3_y_b:.4f}, CI[{ci_pa_x3_y_b[0]:.4f},{ci_pa_x3_y_b[1]:.4f}], p={ppa_x3_y_b:.4g}, n={n_a}\n\n"

        # -----------------------------
        # Bootstrap differences
        # -----------------------------
        rng = np.random.default_rng(random_state)

        # Unadjusted differences
        diffs_unadj = bootstrap_diff_corrs_unadjusted(df_unadj, n_boot, rng)
        results_dict['differences_unadjusted'] = diffs_unadj if diffs_unadj is not None else {}

        if diffs_unadj is not None:
            ret_str += "Differences between unadjusted influences (bootstrap 95% CI; p_boot):\n"
            ret_str += f"  r({original_names['X3']},{original_names['y']}) - r({original_names['X3']},{original_names['X1']}): "
            d = diffs_unadj['dy_minus_dx1']
            ret_str += f"Δ={d['point']:.4f}, CI[{d['lo']:.4f},{d['hi']:.4f}], p_boot={d['p_boot']:.4f}, n_boot={d['n_boot']}\n"
            ret_str += f"  r({original_names['X3']},{original_names['y']}) - r({original_names['X3']},{original_names['X2']}): "
            d = diffs_unadj['dy_minus_dx2']
            ret_str += f"Δ={d['point']:.4f}, CI[{d['lo']:.4f},{d['hi']:.4f}], p_boot={d['p_boot']:.4f}, n_boot={d['n_boot']}\n"
            ret_str += f"  r({original_names['X3']},{original_names['X1']}) - r({original_names['X3']},{original_names['X2']}): "
            d = diffs_unadj['dx1_minus_dx2']
            ret_str += f"Δ={d['point']:.4f}, CI[{d['lo']:.4f},{d['hi']:.4f}], p_boot={d['p_boot']:.4f}, n_boot={d['n_boot']}\n\n"

        # Adjusted differences
        diffs_adj = bootstrap_diff_corrs_adjusted(df_adj_common, n_boot, rng, surface_controls)
        results_dict['differences_adjusted'] = diffs_adj if diffs_adj is not None else {}

        if diffs_adj is not None:
            ret_str += "Differences between adjusted influences (partial r; bootstrap 95% CI; p_boot):\n"
            ret_str += f"  partial r({original_names['X3']},{original_names['y']}|surf+{original_names['X2']}) - "
            ret_str += f"partial r({original_names['X3']},{original_names['X1']}|surf+{original_names['X2']}): "
            d = diffs_adj['dy_minus_dx1']
            ret_str += f"Δ={d['point']:.4f}, CI[{d['lo']:.4f},{d['hi']:.4f}], p_boot={d['p_boot']:.4f}, n_boot={d['n_boot']}\n"

            ret_str += f"  partial r({original_names['X3']},{original_names['y']}|surf+{original_names['X2']}) - "
            ret_str += f"partial r({original_names['X3']},{original_names['X2']}|surf): "
            d = diffs_adj['dy_minus_dx2']
            ret_str += f"Δ={d['point']:.4f}, CI[{d['lo']:.4f},{d['hi']:.4f}], p_boot={d['p_boot']:.4f}, n_boot={d['n_boot']}\n"

            ret_str += f"  partial r({original_names['X3']},{original_names['X1']}|surf+{original_names['X2']}) - "
            ret_str += f"partial r({original_names['X3']},{original_names['X2']}|surf): "
            d = diffs_adj['dx1_minus_dx2']
            ret_str += f"Δ={d['point']:.4f}, CI[{d['lo']:.4f},{d['hi']:.4f}], p_boot={d['p_boot']:.4f}, n_boot={d['n_boot']}\n"

        # Univariate analysis - fit separate logistic regression for each predictor
        ret_str += "Q2: Which factors drive game performance?\n"

        univar = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for var in ['X1','X2','X3']:
            Xi = sm.add_constant(df_norm[[var]])
            fit = sm.Logit(df_norm['y'], Xi).fit(disp=0)
            coef = float(fit.params[var])
            pval = float(fit.pvalues[var])
            # AUC with sklearn (unpenalized if available)
            clf = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
            auc = cross_val_score(clf, df_norm[[var]].values, df_norm['y'].values, cv=cv, scoring='roc_auc')
            univar[var] = {'coefficient': coef, 'odds_ratio': np.exp(coef), 'p_value': pval,
                        'mean_auc': float(auc.mean()), 'std_auc': float(auc.std()), 'aic': float(fit.aic)}

        results_dict['univariate_choice_predictors'] = { original_names[k]: v for k,v in univar.items() }

        # -----------------------------
        # Notes on control sets
        # -----------------------------
        ret_str += "\nNotes:\n"
        ret_str += "- Unadjusted correlations use the common sample with all four core variables present.\n"
        ret_str += "- Adjusted partial correlations use a common sample where all surface controls are present; control sets are:\n"
        ret_str += f"    * For {original_names['X1']}: surface controls + {original_names['X2']}\n"
        ret_str += f"    * For {original_names['X2']}: surface controls only (no {original_names['X2']} control by design)\n"
        ret_str += f"    * For {original_names['y']}: surface controls + {original_names['X2']}\n"
        if surface_controls:
            ret_str += f"- Surface controls included: {surface_controls}\n"
        else:
            ret_str += "- No surface controls were provided.\n"

    except Exception as e:
        ret_str += f"Error in correlation-based influence analysis: {str(e)}\n"

    return ret_str, results_dict

def compare_predictors_of_choice_simple_old(X1, X2, X3, y, continuous_controls=None, categorical_controls=None, normvars=True):
    ret_str = ""
    results_dict = {}
    try:
        #### Setup ####
        original_names = {
            'X1': X1.name or 'X1', 
            'X2': X2.name or 'X2', 
            'X3': X3.name or 'X3', 
            'y': y.name or 'y'
        }
        
        # Build base DataFrame
        df = pd.DataFrame({
            'X1': X1.values, 
            'X2': X2.values, 
            'X3': X3.values, 
            'y': y.values
        })
        
        # Add control variables if provided
        cont_control_names = []
        if continuous_controls:
            for ctrl in continuous_controls:
                ctrl_name = ctrl.name or f'cont_control_{len(cont_control_names)}'
                cont_control_names.append(ctrl_name)
                df[ctrl_name] = ctrl.values
        
        cat_control_names = []
        if categorical_controls:
            for ctrl in categorical_controls:
                ctrl_name = ctrl.name or f'cat_control_{len(cat_control_names)}'
                cat_control_names.append(ctrl_name)
                df[ctrl_name] = ctrl.values
        
        df = df.dropna(subset=['X1', 'X2', 'X3', 'y'])
        
        # Normalize X1, X2, X3 and any continuous controls
        scaler = StandardScaler()
        vars_to_standardize = ['X1', 'X2', 'X3'] + cont_control_names
        df_norm = df.copy()
        if normvars: df_norm[vars_to_standardize] = scaler.fit_transform(df[vars_to_standardize])
        ##################################
        
        #### Relationships among Stated Self/Other Confidences, Entropy, and Choice ####
        # CORRELATIONS
        ret_str += "Q1: Which test (Stated Self, Stated Other, Game) is most influenced by entropy\n"
        r_X1_X2, p_X1_X2 = pearsonr(df_norm['X1'], df_norm['X2'])
        r_X1_X3, p_X1_X3 = pearsonr(df_norm['X1'], df_norm['X3'])
        r_X2_X3, p_X2_X3 = pearsonr(df_norm['X2'], df_norm['X3'])
        r_X1_Y, p_X1_Y = pearsonr(df_norm['X1'], df_norm['y'])
        r_X2_Y, p_X2_Y = pearsonr(df_norm['X2'], df_norm['y'])
        r_X3_Y, p_X3_Y = pearsonr(df_norm['X3'], df_norm['y'])
        
        ret_str += f"Pearson Correlations:\n"
        ret_str += f"  {original_names['X1']}-{original_names['X2']}: r={r_X1_X2:.3f}, p={p_X1_X2:.4f}\n"
        ret_str += f"  {original_names['X1']}-{original_names['X3']}: r={r_X1_X3:.3f}, p={p_X1_X3:.4f}\n"
        ret_str += f"  {original_names['X2']}-{original_names['X3']}: r={r_X2_X3:.3f}, p={p_X2_X3:.4f}\n"
        ret_str += f"  {original_names['X1']}-{original_names['y']}: r={r_X1_Y:.3f}, p={p_X1_Y:.4f}\n"
        ret_str += f"  {original_names['X2']}-{original_names['y']}: r={r_X2_Y:.3f}, p={p_X2_Y:.4f}\n"
        ret_str += f"  {original_names['X3']}-{original_names['y']}: r={r_X3_Y:.3f}, p={p_X3_Y:.4f}\n"
        ret_str += "\n"

        results_dict['pearson_correlations'] = {
            f'{original_names["X1"]}-{original_names["X2"]}': {'r': float(r_X1_X2), 'p': float(p_X1_X2)},
            f'{original_names["X1"]}-{original_names["X3"]}': {'r': float(r_X1_X3), 'p': float(p_X1_X3)},
            f'{original_names["X2"]}-{original_names["X3"]}': {'r': float(r_X2_X3), 'p': float(p_X2_X3)},
            f'{original_names["X1"]}-{original_names["y"]}': {'r': float(r_X1_Y), 'p': float(p_X1_Y)},
            f'{original_names["X2"]}-{original_names["y"]}': {'r': float(r_X2_Y), 'p': float(p_X2_Y)},
            f'{original_names["X3"]}-{original_names["y"]}': {'r': float(r_X3_Y), 'p': float(p_X3_Y)},
        }

        # SPEARMAN CORRELATIONS for X3 relationships 
        rho_X3_X1, p_rho_X3_X1 = spearmanr(df['X3'], df['X1'])
        rho_X3_X2, p_rho_X3_X2 = spearmanr(df['X3'], df['X2'])
        rho_X3_Y, p_rho_X3_Y = spearmanr(df['X3'], df['y'])
        
        ret_str += f"Spearman ρ({original_names['X3']},{original_names['X1']}) = {rho_X3_X1:.3f} (p={p_rho_X3_X1:.4f})\n"
        ret_str += f"Spearman ρ({original_names['X3']},{original_names['X2']}) = {rho_X3_X2:.3f} (p={p_rho_X3_X2:.4f})\n"
        ret_str += f"Spearman ρ({original_names['X3']},{original_names['y']}) = {rho_X3_Y:.3f} (p={p_rho_X3_Y:.4f})\n"
        ret_str += "\n"

        results_dict['spearman_correlations'] = {
            f'{original_names["X3"]}-{original_names["X1"]}': {'rho': float(rho_X3_X1), 'p': float(p_rho_X3_X1)},
            f'{original_names["X3"]}-{original_names["X2"]}': {'rho': float(rho_X3_X2), 'p': float(p_rho_X3_X2)},
            f'{original_names["X3"]}-{original_names["y"]}': {'rho': float(rho_X3_Y), 'p': float(p_rho_X3_Y)},
        }

        # PATH ANALYSIS - asymmetry test
        model_X3_to_X1 = LinearRegression().fit(df_norm[['X3', 'X2']], df_norm['X1'])
        r2_X3_to_X1 = model_X3_to_X1.score(df_norm[['X3', 'X2']], df_norm['X1'])
        
        model_X1_to_X3 = LinearRegression().fit(df_norm[['X1', 'X2']], df_norm['X3'])
        r2_X1_to_X3 = model_X1_to_X3.score(df_norm[['X1', 'X2']], df_norm['X3'])
        
        ret_str += f"{original_names['X3']}+{original_names['X2']}→{original_names['X1']}: R²={r2_X3_to_X1:.3f}\n"
        ret_str += f"{original_names['X1']}+{original_names['X2']}→{original_names['X3']}: R²={r2_X1_to_X3:.3f}\n"
        ret_str += "\n"

        results_dict['asymmetry_test'] = {
            f"{original_names['X3']}_{original_names['X2']}_to_{original_names['X1']}_R2": float(r2_X3_to_X1),
            f"{original_names['X1']}_{original_names['X2']}_to_{original_names['X3']}_R2": float(r2_X1_to_X3),
        }

        ret_str += f"Comparative entropy impacts\n"
        import statsmodels.api as sm
        from scipy import stats
        dfm = df_norm.dropna(subset=['X1','X2','X3','y']).copy()
        # 1) X3 → X1 controlling for X2 (OLS with robust SEs)
        X_ols = sm.add_constant(dfm[['X2', 'X3']])
        ols = sm.OLS(dfm['X1'], X_ols).fit(cov_type='HC3')
        beta_X3_to_X1 = ols.params['X3']
        p_X3_to_X1 = ols.pvalues['X3']

        # 2) X3 → y controlling for X2 (Logit with unpenalized MLE)
        X_logit = sm.add_constant(dfm[['X2', 'X3']])
        logit = sm.Logit(dfm['y'], X_logit).fit(disp=0)
        beta_X3_to_y = logit.params['X3']
        p_X3_to_y = logit.pvalues['X3']

        # Robustness 1: Add y as control in OLS predicting X1
        X_ols_full = sm.add_constant(dfm[['X2','y','X3']])
        ols_full = sm.OLS(dfm['X1'], X_ols_full).fit(cov_type='HC3')
        beta_X3_to_X1_full = ols_full.params['X3']
        p_X3_to_X1_full = ols_full.pvalues['X3']

        # Robustness 2: y ~ X1 + X2 + X3
        X_logit_full = sm.add_constant(dfm[['X1','X2','X3']])
        logit_full = sm.Logit(dfm['y'], X_logit_full).fit(disp=0)
        beta_X3_to_y_full = logit_full.params['X3']
        p_X3_to_y_full = logit_full.pvalues['X3']

        # Store results
        results_dict['comparative_entropy_impacts'] = {
            'X3_to_X1_controlling_X2': {'beta': float(beta_X3_to_X1), 'p': float(p_X3_to_X1)},
            'X3_to_Y_controlling_X2': {'beta': float(beta_X3_to_y), 'p': float(p_X3_to_y)},
            'X3_to_X1_controlling_X2_Y': {'beta': float(beta_X3_to_X1_full), 'p': float(p_X3_to_X1_full)},
            'X3_to_Y_controlling_X1_X2': {'beta': float(beta_X3_to_y_full), 'p': float(p_X3_to_y_full)},
        }
        ret_str += f"Controlling for {original_names['X2']}:\n"
        ret_str += f"  {original_names['X3']}→{original_names['X1']}: β = {beta_X3_to_X1:.4f}, p = {p_X3_to_X1:.4f}\n"
        ret_str += f"  {original_names['X3']}→{original_names['y']}: β = {beta_X3_to_y:.4f}, p = {p_X3_to_y:.4f}\n"
        ret_str += "\n"
        
        ret_str += f"Controlling for BOTH {original_names['X2']} and {original_names['y']}:\n"
        ret_str += f"  {original_names['X3']}→{original_names['X1']}: β = {beta_X3_to_X1_full:.4f}, p = {p_X3_to_X1_full:.4f}\n"
        ret_str += f"Controlling for BOTH {original_names['X1']} and {original_names['X2']}:\n"
        ret_str += f"  {original_names['X3']}→{original_names['y']}: β = {beta_X3_to_y_full:.4f}, p = {p_X3_to_y_full:.4f}\n"
        ret_str += "\n"

        # Effect size comparison for Q1
        # Partial R^2 for X1: compare base (X2 only) vs full (X2 + X3)
        Xb_ols = sm.add_constant(dfm[['X2']])
        ols_b = sm.OLS(dfm['X1'], Xb_ols).fit()
        Xf_ols = sm.add_constant(dfm[['X2','X3']])
        ols_f = sm.OLS(dfm['X1'], Xf_ols).fit()
        partial_R2_X3_on_X1 = 1.0 - (ols_f.ssr / ols_b.ssr)

        # Tjur's R^2 for y (interpretable under imbalance)
        def tjur_R2(result, X, y_vec):
            p = result.predict(X)
            return float(p[y_vec==1].mean() - p[y_vec==0].mean())

        Xb_log = sm.add_constant(dfm[['X2']])
        mb = sm.Logit(dfm['y'], Xb_log).fit(disp=0)
        Xf_log = sm.add_constant(dfm[['X2','X3']])
        mf = sm.Logit(dfm['y'], Xf_log).fit(disp=0)
        delta_tjur_y = tjur_R2(mf, Xf_log, dfm['y']) - tjur_R2(mb, Xb_log, dfm['y'])

        results_dict.setdefault('comparative_entropy_impacts', {}).update({
            'partial_R2_X3_on_X1_ctrl_X2': float(partial_R2_X3_on_X1),
            'delta_TjurR2_X3_on_y_ctrl_X2': float(delta_tjur_y),
        })
        ret_str += f"Effect sizes (controls: {original_names['X2']}): partial R² (X1)={partial_R2_X3_on_X1:.4f}, ΔTjur R² (y)={delta_tjur_y:.4f}\n"
        res = bootstrap_effect_sizes(dfm, X1_col='X1', X2_col='X2', X3_col='X3', y_col='y',
                                    controls=None, n_boot=2000, stratify=True, random_state=123)

        ret_str += f"({res['bootstrap_CI_percentile']['gap_deltaTjur_minus_partialR2']['lo']}-{res['bootstrap_CI_percentile']['gap_deltaTjur_minus_partialR2']['hi']})\n"
        results_dict['comparative_entropy_impacts']['low'] = res['bootstrap_CI_percentile']['gap_deltaTjur_minus_partialR2']['lo']
        results_dict['comparative_entropy_impacts']['high'] = res['bootstrap_CI_percentile']['gap_deltaTjur_minus_partialR2']['hi']
        ##################################

        #### Analysis of Stated Self/Other Confidences and Entropy as Predictors of Choice ####
        # Univariate analysis - fit separate logistic regression for each predictor
        ret_str += "Q2: Which factors drive game performance?\n"

        univar = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for var in ['X1','X2','X3']:
            Xi = sm.add_constant(dfm[[var]])
            fit = sm.Logit(dfm['y'], Xi).fit(disp=0)
            coef = float(fit.params[var])
            pval = float(fit.pvalues[var])
            # AUC with sklearn (unpenalized if available)
            clf = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
            auc = cross_val_score(clf, dfm[[var]].values, dfm['y'].values, cv=cv, scoring='roc_auc')
            univar[var] = {'coefficient': coef, 'odds_ratio': np.exp(coef), 'p_value': pval,
                        'mean_auc': float(auc.mean()), 'std_auc': float(auc.std()), 'aic': float(fit.aic)}

        results_dict['univariate_choice_predictors'] = { original_names[k]: v for k,v in univar.items() }

        ret_str += "="*60
        ret_str += "\nUNIVARIATE RESULTS (each predictor alone)\n"
        ret_str += "="*60 + "\n"
        results_df = pd.DataFrame(univar).T
        results_df = results_df.round(4)
        ret_str += results_df[['coefficient', 'odds_ratio', 'p_value', 'mean_auc', 'aic']].to_string()
        ret_str += "\n"

        # LIKELIHOOD RATIO TESTS - which variables add value?
        ret_str += "="*60 + "\n"
        ret_str += "LIKELIHOOD RATIO TESTS\n"
        ret_str += "="*60 + "\n"

        def lr_test(base_vars, add_var):
            Xb = sm.add_constant(dfm[base_vars])
            Xf = sm.add_constant(dfm[base_vars + [add_var]])
            mb = sm.Logit(dfm['y'], Xb).fit(disp=0)
            mf = sm.Logit(dfm['y'], Xf).fit(disp=0)
            lr = 2*(mf.llf - mb.llf)
            p = stats.chi2.sf(lr, df=1)
            return lr, p

        for base_vars, add_var in [(['X1'], 'X2'), (['X1'], 'X3'), (['X2'], 'X3'),
                                (['X1','X2'], 'X3'), (['X1','X3'], 'X2')]:
            lr, p = lr_test(base_vars, add_var)
            
            base_names = [original_names[v] for v in base_vars]
            ret_str += f"{original_names[add_var]} adds to {'+'.join(base_names)}: LR={lr:.3f}, p={p:.4f}\n"

            if 'likelihood_ratio_tests' not in results_dict:
                results_dict['likelihood_ratio_tests'] = {}
            
            key = f"{original_names[add_var]}_adds_to_{'+'.join(base_names)}"
            results_dict['likelihood_ratio_tests'][key] = {
                'LR': float(lr), 
                'p': float(p)
            } 

        # REGRESSION WITH CONTROLS 
        if cont_control_names or cat_control_names:
            ret_str += "\n"
            ret_str += "="*60 + "\n"
            ret_str += "FULL MODEL WITH CONTROLS\n"
            ret_str += "="*60 + "\n"
            
            # Build formula string
            formula_parts = [original_names['y'], '~', original_names['X1'], '+', original_names['X2'], '+', original_names['X3']]
            
            # Add continuous controls
            for ctrl in cont_control_names:
                formula_parts.extend(['+', ctrl])
            
            # Add categorical controls with C() notation
            for ctrl in cat_control_names:
                formula_parts.extend(['+', f'C({ctrl})'])
            
            formula = ' '.join(formula_parts)
            
            # Use original df with actual column names for statsmodels
            df_for_model = pd.DataFrame()
            df_for_model[original_names['X1']] = df_norm['X1']
            df_for_model[original_names['X2']] = df_norm['X2']
            df_for_model[original_names['X3']] = df_norm['X3']
            df_for_model[original_names['y']] = df_norm['y']
            
            # Add controls (standardized continuous, original categorical)
            for ctrl in cont_control_names:
                df_for_model[ctrl] = df_norm[ctrl]
            for ctrl in cat_control_names:
                df_for_model[ctrl] = df[ctrl]
            
            model = smf.logit(formula, data=df_for_model)
            result = model.fit(disp=0)
            ret_str += result.summary().as_text()

            results_dict['full_model_choice_predictors'] = {}
            for var_key in ['X1', 'X2', 'X3']:
                var_name = original_names[var_key]
                if var_name in result.params.index:
                    results_dict['full_model_choice_predictors'][var_name] = {
                        'coef': float(result.params[var_name]),
                        'p': float(result.pvalues[var_name]),
                    }
        ##################################
        
    except Exception as e:
        ret_str += f"Error in simplified entropy analysis: {str(e)}\n"
        
    return ret_str, results_dict

def compare_predictors_of_choice(X1, X2, X3, y):
    ret_str = ""
    try:
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})
        df = df[['X1', 'X2', 'X3', 'y']].dropna()
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(df[['X1', 'X2', 'X3']])

        df_norm = pd.DataFrame(X_normalized, columns=['X1', 'X2', 'X3'])
        df_norm['y'] = df['y'].values

        ret_str += "="*60
        r_X1_X2, p_X1_X2 = pearsonr(df_norm['X1'], df_norm['X2'])
        r_X1_X3, p_X1_X3 = pearsonr(df_norm['X1'], df_norm['X3'])
        r_X2_X3, p_X2_X3 = pearsonr(df_norm['X2'], df_norm['X3'])

        ret_str += f"Correlations:\n"
        ret_str += f"Pearson X1-X2: r={r_X1_X2:.3f}, p={p_X1_X2:.4f}\n"
        ret_str += f"Pearson X1-X3: r={r_X1_X3:.3f}, p={p_X1_X3:.4f}\n"
        ret_str += f"Pearson X2-X3: r={r_X2_X3:.3f}, p={p_X2_X3:.4f}\n"
        ret_str += "\n"

        # Step 3: Univariate analysis - fit separate logistic regression for each predictor
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for var in ['X1', 'X2', 'X3']:
            # Fit model
            X = df_norm[[var]].values
            y = df_norm['y'].values
            
            model = LogisticRegression(solver='liblinear')
            model.fit(X, y)
            
            # Get coefficient and p-value
            coef = model.coef_[0, 0]
            z_score = coef / (np.sqrt(np.diag(np.linalg.inv(X.T @ X))) * 0.5)  # Approximate SE
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
            
            # Calculate AUC with cross-validation
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            mean_auc = auc_scores.mean()
            std_auc = auc_scores.std()
            
            # Calculate log-likelihood and AIC
            probs = model.predict_proba(X)[:, 1]
            log_likelihood = np.sum(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
            aic = -2 * log_likelihood + 2 * 2  # 2 parameters (intercept + coefficient)
            
            results[var] = {
                'coefficient': coef,
                'odds_ratio': np.exp(coef),
                'p_value': p_value,
                'mean_auc': mean_auc,
                'std_auc': std_auc,
                'aic': aic,
                'log_likelihood': log_likelihood
            }

        # Step 4: Multivariate analysis - all predictors together
        X_all = df_norm[['X1', 'X2', 'X3']].values
        y = df_norm['y'].values

        model_full = LogisticRegression(solver='liblinear')
        model_full.fit(X_all, y)

        # Calculate VIF (Variance Inflation Factors) to check multicollinearity
        from numpy.linalg import inv
        corr_matrix = df_norm[['X1', 'X2', 'X3']].corr().values
        vif = np.diag(inv(corr_matrix))

        # Get multivariate results
        multi_results = {}
        for i, var in enumerate(['X1', 'X2', 'X3']):
            multi_results[var] = {
                'multi_coefficient': model_full.coef_[0, i],
                'multi_odds_ratio': np.exp(model_full.coef_[0, i]),
                'vif': vif[i]
            }

        ret_str += "="*60
        ret_str += "\nUNIVARIATE RESULTS (each predictor alone)\n"
        ret_str += "="*60 + "\n"
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        ret_str += results_df[['coefficient', 'odds_ratio', 'p_value', 'mean_auc', 'aic']].to_string()
        ret_str += "\n"

        ret_str += "="*60
        ret_str += "\nMULTIVARIATE RESULTS (all predictors together)\n"
        ret_str += "="*60 + "\n"
        multi_df = pd.DataFrame(multi_results).T
        multi_df = multi_df.round(4)
        ret_str += multi_df.to_string()
        ret_str += "\n"

        ret_str += "="*60
        ret_str += "\nBEST PREDICTOR DETERMINATION\n"
        ret_str += "="*60 + "\n"

        # Find best by AUC
        best_auc = max(results.keys(), key=lambda x: results[x]['mean_auc'])
        ret_str += f"Highest AUC (univariate): {best_auc} with AUC = {results[best_auc]['mean_auc']:.4f}\n"

        # Find best by AIC (lowest is better)
        best_aic = min(results.keys(), key=lambda x: results[x]['aic'])
        ret_str += f"Lowest AIC (univariate): {best_aic} with AIC = {results[best_aic]['aic']:.2f}\n"

        # Find best by absolute coefficient in multivariate model
        best_multi = max(['X1', 'X2', 'X3'], key=lambda x: abs(multi_results[x]['multi_coefficient']))
        ret_str += f"Largest coefficient (multivariate): {best_multi} with coef = {multi_results[best_multi]['multi_coefficient']:.4f}\n"

        ret_str += "\n"
        ret_str += "="*60
        ret_str += "\nFINAL ANSWER\n"
        ret_str += "="*60 + "\n"

        # Determine overall best predictor based on multiple criteria
        scores = {var: 0 for var in ['X1', 'X2', 'X3']}
        scores[best_auc] += 1
        scores[best_aic] += 1
        scores[best_multi] += 1

        best_overall = max(scores.keys(), key=lambda x: scores[x])

        if max(scores.values()) >= 2:
            ret_str += f"BEST PREDICTOR: {best_overall}\n"
            ret_str += f"Reason: Best in {scores[best_overall]} out of 3 criteria\n"
        else:
            ret_str += f"BEST PREDICTOR BY AUC: {best_auc}\n"
            ret_str += "Note: No clear winner across all criteria, but AUC is most reliable for prediction\n"

        # Step 7: Additional validation - compare nested models
        ret_str += "\n"
        ret_str += "="*60 + "\n"
        ret_str += "LIKELIHOOD RATIO TESTS (does adding other variables help?)\n"
        ret_str += "="*60 + "\n"

        # For the best predictor, test if adding others improves significantly
        X_best = df_norm[[best_auc]].values
        model_best = LogisticRegression(solver='liblinear').fit(X_best, y)
        ll_best = np.sum(y * np.log(model_best.predict_proba(X_best)[:, 1] + 1e-10) + 
                        (1 - y) * np.log(1 - model_best.predict_proba(X_best)[:, 1] + 1e-10))

        # Test adding each other variable
        for var in ['X1', 'X2', 'X3']:
            if var != best_auc:
                X_combined = df_norm[[best_auc, var]].values
                model_combined = LogisticRegression(solver='liblinear').fit(X_combined, y)
                ll_combined = np.sum(y * np.log(model_combined.predict_proba(X_combined)[:, 1] + 1e-10) + 
                                (1 - y) * np.log(1 - model_combined.predict_proba(X_combined)[:, 1] + 1e-10))
                
                lr_stat = 2 * (ll_combined - ll_best)
                p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
                
                ret_str += f"Adding {var} to {best_auc}: LR stat = {lr_stat:.3f}, p = {p_value:.4f}\n"
                if p_value < 0.05:
                    ret_str += f"  -> {var} adds significant predictive value\n"
                else:
                    ret_str += f"  -> {var} does NOT add significant predictive value\n"

        from itertools import combinations
        variables = ['X1', 'X2', 'X3']
        y = df['y'].values

        for base_size in [1, 2]:
            for base_vars in combinations(variables, base_size):
                base_vars = list(base_vars)
                X_base = df[base_vars].values
                model_base = LogisticRegression(solver='liblinear').fit(X_base, y)
                
                # Calculate log-likelihood for base model
                probs_base = model_base.predict_proba(X_base)[:, 1]
                probs_base = np.clip(probs_base, 1e-10, 1-1e-10)  # Avoid log(0)
                ll_base = np.sum(y * np.log(probs_base) + (1 - y) * np.log(1 - probs_base))
                
                for add_var in variables:
                    if add_var not in base_vars:
                        X_full = df[base_vars + [add_var]].values
                        model_full = LogisticRegression(solver='liblinear').fit(X_full, y)
                        
                        # Calculate log-likelihood for full model
                        probs_full = model_full.predict_proba(X_full)[:, 1]
                        probs_full = np.clip(probs_full, 1e-10, 1-1e-10)
                        ll_full = np.sum(y * np.log(probs_full) + (1 - y) * np.log(1 - probs_full))
                        
                        lr_stat = 2 * (ll_full - ll_base)
                        p_val = 1 - stats.chi2.cdf(lr_stat, df=1)
                        
                        ret_str += f"{add_var} adds to {'+'.join(base_vars)}: LR={lr_stat:.3f}, p={p_val:.4f}\n"



        # Compare R² (variance explained)
        # For X3→X1 (continuous)
        model_X3_X1 = LinearRegression().fit(df[['X3']], df['X1'])
        r2_X3_X1 = model_X3_X1.score(df[['X3']], df['X1'])

        # For X3→Y (pseudo-R² for binary)
        model_X3_Y = LogisticRegression().fit(df[['X3']], df['y'])
        # McFadden's pseudo-R²
        null_model = LogisticRegression().fit(np.ones((len(df), 1)), df['y'])
        ll_null = np.sum(df['y'] * np.log(null_model.predict_proba(np.ones((len(df), 1)))[:, 1] + 1e-10) + 
                        (1 - df['y']) * np.log(1 - null_model.predict_proba(np.ones((len(df), 1)))[:, 1] + 1e-10))
        ll_model = np.sum(df['y'] * np.log(model_X3_Y.predict_proba(df[['X3']])[:, 1] + 1e-10) + 
                        (1 - df['y']) * np.log(1 - model_X3_Y.predict_proba(df[['X3']])[:, 1] + 1e-10))
        pseudo_r2_X3_Y = 1 - (ll_model / ll_null)

        # Also compare Spearman correlations (rank-based, works for both)
        rho_X3_X1, p_rho_X3_X1 = spearmanr(df['X3'], df['X1'])
        rho_X3_Y, p_rho_X3_Y = spearmanr(df['X3'], df['y'])

        ret_str += f"X3 explains {r2_X3_X1:.1%} of variance in X1\n"
        ret_str += f"X3 explains {pseudo_r2_X3_Y:.1%} of variance in Y (pseudo-R²)\n"
        ret_str += f"Spearman ρ(X3,X1) = {rho_X3_X1:.3f} (p={p_rho_X3_X1:.4f})\n"
        ret_str += f"Spearman ρ(X3,Y) = {rho_X3_Y:.3f} (p={p_rho_X3_Y:.4f})\n"

        # X3 predicting X1, controlling for X2
        model_X3_to_X1 = LinearRegression().fit(df[['X3', 'X2']], df['X1'])
        r2_X3_to_X1 = model_X3_to_X1.score(df[['X3', 'X2']], df['X1'])

        # X1 predicting X3, controlling for X2  
        model_X1_to_X3 = LinearRegression().fit(df[['X1', 'X2']], df['X3'])
        r2_X1_to_X3 = model_X1_to_X3.score(df[['X1', 'X2']], df['X3'])

        ret_str += f"X3+X2→X1: R²={r2_X3_to_X1:.3f}\n"
        ret_str += f"X1+X2→X3: R²={r2_X1_to_X3:.3f}\n"


        # X3 predicting X1, controlling for X2
        model = LinearRegression().fit(df[['X2', 'X3']], df['X1'])
        X3_coef = model.coef_[1]

        # Force float64 to prevent object dtype issues
        X = df[['X2', 'X3']].values
        X = X.astype(np.float64)  # THIS IS THE FIX

        y_pred = model.predict(df[['X2', 'X3']])  # Use original df here
        residuals = df['X1'].values - y_pred
        n = len(df)
        p = X.shape[1]
        mse = np.sum(residuals**2) / (n - p)
        var_coef = mse * np.linalg.inv(X.T @ X)[1, 1]
        se = np.sqrt(var_coef)
        t_stat = X3_coef / se
        p_value_X1 = 2 * (1 - stats.t.cdf(abs(t_stat), n - p))

        ret_str += f"Controlling for X2:\n"
        ret_str += f"  X3→X1: β = {X3_coef:.4f}, p = {p_value_X1:.4f}\n"

        # X3 predicting Y, controlling for X2  
        model_y = LogisticRegression(solver='liblinear').fit(df[['X2', 'X3']], df['y'])
        X3_coef_y = model_y.coef_[0, 1]  # X3 is second predictor

        # For logistic regression p-value (approximate)
        z_stat = X3_coef_y / (np.sqrt(np.diag(np.linalg.inv(X.T @ X))[1]) * 0.5)
        p_value_Y = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        ret_str += f"  X3→Y: β = {X3_coef_y:.4f}, p = {p_value_Y:.4f}\n"

        # X3 predicting X1, controlling for BOTH X2 and Y
        model_X1_full = LinearRegression().fit(df[['X2', 'y', 'X3']], df['X1'])
        X3_coef_X1_full = model_X1_full.coef_[2]  # X3 is third predictor

        # Calculate standard error for significance test
        X_full = df[['X2', 'y', 'X3']].values.astype(np.float64)
        y_pred = model_X1_full.predict(df[['X2', 'y', 'X3']])
        residuals = df['X1'].values - y_pred
        n = len(df)
        p = X_full.shape[1]
        mse = np.sum(residuals**2) / (n - p)
        var_coef = mse * np.linalg.inv(X_full.T @ X_full)[2, 2]
        se = np.sqrt(var_coef)
        t_stat = X3_coef_X1_full / se
        p_value_X1_full = 2 * (1 - stats.t.cdf(abs(t_stat), n - p))

        ret_str += f"Controlling for BOTH X2 and Y:\n"
        ret_str += f"  X3→X1: β = {X3_coef_X1_full:.4f}, p = {p_value_X1_full:.4f}\n"

        # X3 predicting Y, controlling for BOTH X1 and X2
        model_full = LogisticRegression(solver='liblinear').fit(df[['X1', 'X2', 'X3']], df['y'])
        X3_coef_full = model_full.coef_[0, 2]  # X3 is third predictor

        # For significance test
        X_full = df[['X1', 'X2', 'X3']].values.astype(np.float64)
        # Approximate z-test for logistic coefficient
        z_stat = X3_coef_full / (np.sqrt(np.diag(np.linalg.inv(X_full.T @ X_full))[2]) * 0.5)
        p_value_full = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        ret_str += f"Controlling for BOTH X1 and X2:\n"
        ret_str += f"  X3→Y: β = {X3_coef_full:.4f}, p = {p_value_full:.4f}\n"


        # X3→X1 relationship
        model_simple = LinearRegression().fit(df[['X3']], df['X1'])
        r2_simple = model_simple.score(df[['X3']], df['X1'])

        model_controlled = LinearRegression().fit(df[['X3', 'X2']], df['X1'])
        r2_controlled = model_controlled.score(df[['X3', 'X2']], df['X1'])  

        model_X2_only = LinearRegression().fit(df[['X2']], df['X1'])
        r2_X2_only = model_X2_only.score(df[['X2']], df['X1'])

        partial_r2 = r2_controlled - r2_X2_only  

        attenuation_X1 = (r2_simple - partial_r2) / r2_simple

        # X3→Y relationship
        # Simple model: just X3
        model_Y_simple = LogisticRegression(solver='liblinear').fit(df[['X3']], df['y'])
        prob_simple = np.clip(model_Y_simple.predict_proba(df[['X3']])[:, 1], 1e-10, 1-1e-10)
        ll_simple = np.sum(df['y'] * np.log(prob_simple) + (1 - df['y']) * np.log(1 - prob_simple))

        # Null model (intercept only) for pseudo-R² calculation
        model_null = LogisticRegression(solver='liblinear').fit(np.ones((len(df), 1)), df['y'])
        prob_null = np.clip(model_null.predict_proba(np.ones((len(df), 1)))[:, 1], 1e-10, 1-1e-10)
        ll_null = np.sum(df['y'] * np.log(prob_null) + (1 - df['y']) * np.log(1 - prob_null))

        # Controlled model: X3 and X2
        model_Y_controlled = LogisticRegression(solver='liblinear').fit(df[['X3', 'X2']], df['y'])
        prob_controlled = np.clip(model_Y_controlled.predict_proba(df[['X3', 'X2']])[:, 1], 1e-10, 1-1e-10)
        ll_controlled = np.sum(df['y'] * np.log(prob_controlled) + (1 - df['y']) * np.log(1 - prob_controlled))

        # Model with just X2
        model_Y_X2_only = LogisticRegression(solver='liblinear').fit(df[['X2']], df['y'])
        prob_X2_only = np.clip(model_Y_X2_only.predict_proba(df[['X2']])[:, 1], 1e-10, 1-1e-10)
        ll_X2_only = np.sum(df['y'] * np.log(prob_X2_only) + (1 - df['y']) * np.log(1 - prob_X2_only))

        # Calculate partial contribution of X3 (beyond X2)
        # This is the improvement in log-likelihood from adding X3 to X2
        ll_improvement_from_X3 = ll_controlled - ll_X2_only
        ll_improvement_simple = ll_simple - ll_null

        # Attenuation: how much does X3's effect weaken when controlling for X2?
        attenuation_Y = 1 - (ll_improvement_from_X3 / ll_improvement_simple)

        ret_str += f"X3→X1 attenuates {attenuation_X1:.1%} when controlling for X2\n"
        ret_str += f"X3→Y attenuates {attenuation_Y:.1%} when controlling for X2\n"
        if attenuation_Y < attenuation_X1:
            ret_str += "X3→Y is MORE robust to controlling for difficulty (X2)\n"
            ret_str += "This suggests delegation uses internal signals beyond external difficulty cues\n"
        else:
            ret_str += "X3→X1 is MORE robust to controlling for difficulty (X2)\n"

    except Exception as e:
        ret_str += f"Error during analysis: {e}"
    return ret_str


def contingency(delegate: np.ndarray, correct: np.ndarray):
    """
    delegate : bool[N]   True -> model delegated
    correct  : bool[N]   True -> model would be correct on its own
    returns  : TP, FN, FP, TN as ints
    """
    TP = np.sum(delegate  & ~correct)   # delegate & wrong
    FN = np.sum(~delegate & ~correct)   # keep     & wrong
    FP = np.sum(delegate  &  correct)   # delegate & right
    TN = np.sum(~delegate &  correct)   # keep     & right
    return TP, FN, FP, TN

def lift_mcc_stats(tp, fn, fp, tn, kept_correct, p0, n_boot=2000, seed=0, 
                   baseline_correct=None, delegated=None, baseline_probs=None):
    rng = np.random.default_rng(seed)
    N = tp + fn + fp + tn

    # ---------- point estimates --------------------------------------------
    k = len(kept_correct)
    kept_acc = kept_correct.mean() if k else np.nan
    lift = kept_acc - p0
    normed_lift = (kept_acc - p0) / (1 - p0) if p0 < 1.0 else np.nan

    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / denom if denom else np.nan

    sensitivity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity = tp / (tp + fn) if (tp + fn) > 0 else 0
    single_point_auc = (sensitivity + specificity) / 2
    j = (sensitivity + specificity - 1.0)
    p = (tn + fp) / N 
    c = (tn + fn) / N
    if (p > 0) and (p < 1) and (c > 0) and (c < 1):
        j_max = min(c / p, (1 - c) / (1 - p))
    else:
        j_max = 0.0
    ba_norm = (j / j_max) if (j_max > 0) and np.isfinite(j) else np.nan

    # Compute full AUC if we have probabilities
    full_auc, calibration_auc = None, None
    if baseline_probs is not None and delegated is not None:
        # Convert once to numpy for positional masking
        bp0 = baseline_probs.values if hasattr(baseline_probs, 'values') else baseline_probs
        dg0 = delegated.values if hasattr(delegated, 'values') else delegated
        bc0 = baseline_correct.values if hasattr(baseline_correct, 'values') else baseline_correct

        bp0 = np.asarray(bp0, dtype=float)
        dg0 = np.asarray(dg0, dtype=float)
        bc0 = np.asarray(bc0, dtype=float)

        # Single joint mask so lengths stay identical for downstream bootstrap
        mask = ~np.isnan(bp0) & ~np.isnan(dg0) & ~np.isnan(bc0)
        baseline_probs = bp0[mask]
        delegated = dg0[mask]
        baseline_correct = bc0[mask]

        # Assuming delegated needs inversion (1=pass → 1=answer)
        delegated_binary = 1 - delegated
        full_auc = roc_auc_score(delegated_binary, baseline_probs)
        calibration_auc = roc_auc_score(baseline_correct, baseline_probs)

    # ---------- p-values ----------------------------------------------------
    p_lift = binomtest(kept_correct.sum(), k, p0, alternative='two-sided').pvalue
    p_mcc = mcnemar([[tn, fp], [fn, tp]], exact=True).pvalue

    # ---------- bootstrap CIs ----------------------------------------------
    counts = np.array([tp, fn, fp, tn], int)
    probs = counts / N

    lifts, normed_lifts, mccs, aucs, full_aucs, calibration_aucs = [], [], [], [], [], []
    kept_idx = np.arange(k)

    for _ in range(n_boot):
        # ----- lift: resample ONLY kept correctness
        b_k_acc = kept_correct[rng.choice(kept_idx, k, replace=True)].mean()
        lifts.append(b_k_acc - p0)
        if p0 < 1.0:
            normed_lifts.append((b_k_acc - p0) / (1 - p0))
        else:
            normed_lifts.append(np.nan)

        # ----- MCC: multinomial resample of 4-cell table
        sample = rng.choice(4, size=N, replace=True, p=probs)
        btp, bfn, bfp, btn = np.bincount(sample, minlength=4)
        bden = math.sqrt((btp+bfp)*(btp+bfn)*(btn+bfp)*(btn+bfn))
        bmcc = (btp*btn - bfp*bfn) / bden if bden else 0.0
        mccs.append(bmcc)

        # ----- Single point AUC (if we have the data)
        if baseline_correct is not None and delegated is not None and j_max > 0:
            N = len(baseline_correct)
            idx = rng.choice(N, N, replace=True)
            boot_bc = baseline_correct.iloc[idx].values if hasattr(baseline_correct, 'iloc') else baseline_correct[idx]
            boot_del = delegated.iloc[idx].values if hasattr(delegated, 'iloc') else delegated[idx]
            boot_del_binary = 1 - boot_del  # Invert: 1=answer, 0=pass
            
            ac = ((boot_del_binary == 1) & (boot_bc == 1)).sum()
            ai = ((boot_del_binary == 1) & (boot_bc == 0)).sum()
            pc = ((boot_del_binary == 0) & (boot_bc == 1)).sum()
            pi = ((boot_del_binary == 0) & (boot_bc == 0)).sum()

            sens = ac / (ac + pc) if (ac + pc) > 0 else 0
            spec = pi / (pi + ai) if (pi + ai) > 0 else 0
            j = (sens + spec - 1.0)
            p = (ac + pc) / N 
            c = (ac + ai) / N
            j_max = min(c / p, (1 - c) / (1 - p))
            b_ba_norm = (j / j_max) 
            aucs.append(b_ba_norm)###(sens + spec) / 2)
            
            # ----- Full ROC-AUC (if we have probabilities)
            if baseline_probs is not None:
                boot_probs = baseline_probs.iloc[idx].values if hasattr(baseline_probs, 'iloc') else baseline_probs[idx]
                try:
                    full_aucs.append(roc_auc_score(boot_del_binary, boot_probs))
                    calibration_aucs.append(roc_auc_score(boot_bc, boot_probs))
                except:
                    pass  # Skip if all same class

    # Compute CIs
    ci_lift = tuple(np.percentile(lifts, [2.5, 97.5]))
    ci_normed_lift = tuple(np.percentile(normed_lifts, [2.5, 97.5]))
    ci_mcc = tuple(np.percentile(mccs, [2.5, 97.5]))
    ci_single_auc = tuple(np.percentile(aucs, [2.5, 97.5])) if aucs else (np.nan, np.nan)
    ci_full_auc = tuple(np.percentile(full_aucs, [2.5, 97.5])) if full_aucs else (np.nan, np.nan)
    ci_calibration_auc = tuple(np.percentile(calibration_aucs, [2.5, 97.5])) if calibration_aucs else (np.nan, np.nan)

    boot_arr = np.array(mccs)
    p_mcc = 2 * min((boot_arr <= 0).mean(), (boot_arr >= 0).mean())


    def ba_uplift_and_ci(tp, fn, fp, tn, ci=0.95):
        """
        Returns:
        - ba_uplift: (TPR - FPR)/2  where
            TPR = P(answer | correct)   = TN / (TN + FP)
            FPR = P(answer | incorrect) = FN / (TP + FN)
        - ci_low, ci_high: Newcombe (Wilson) CI for (TPR - FPR)/2
        - cohens_h, h_ci_low, h_ci_high: arcsine effect size and CI
        """
        import math
        from scipy.stats import norm

        tp = float(tp); fn = float(fn); fp = float(fp); tn = float(tn)
        n_correct   = tn + fp
        n_incorrect = tp + fn
        if n_correct <= 0 or n_incorrect <= 0:
            return {k: float('nan') for k in [
                'ba_uplift','ci_low','ci_high','cohens_h','h_ci_low','h_ci_high'
            ]}

        tpr = tn / n_correct
        fpr = fn / n_incorrect
        J = tpr - fpr
        ba_uplift = J / 2.0

        # Wilson interval
        def wilson(p, n, alpha):
            z = norm.ppf(1 - alpha/2)
            z2 = z*z
            denom = 1 + z2/n
            center = (p + z2/(2*n)) / denom
            half = (z / denom) * math.sqrt((p*(1-p) + z2/(4*n)) / n)
            return center - half, center + half

        alpha = 1 - ci
        L1, U1 = wilson(tpr, n_correct, alpha)
        L2, U2 = wilson(fpr, n_incorrect, alpha)

        # Newcombe MOVER for difference: d = p1 - p2
        d = tpr - fpr
        low_d  = d - math.sqrt((tpr - L1)**2 + (U2 - fpr)**2)
        high_d = d + math.sqrt((U1 - tpr)**2 + (fpr - L2)**2)

        ci_low  = low_d / 2.0
        ci_high = high_d / 2.0

        # Cohen's h and CI
        def asin_sqrt(p):
            # Clamp to open interval to avoid nan at 0/1
            eps = 1e-12
            p = min(max(p, eps), 1 - eps)
            return math.asin(math.sqrt(p))

        h = 2*asin_sqrt(tpr) - 2*asin_sqrt(fpr)
        se_h = math.sqrt(1.0/n_correct + 1.0/n_incorrect)
        z = norm.ppf(0.5 + ci/2.0)
        h_lo = h - z*se_h
        h_hi = h + z*se_h

        return {
            'ba_uplift': float(ba_uplift),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'cohens_h': float(h),
            'h_ci_low': float(h_lo),
            'h_ci_high': float(h_hi),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'n_correct': int(n_correct),
            'n_incorrect': int(n_incorrect),
        }    
    ba_stats = ba_uplift_and_ci(tp, fn, fp, tn, ci=0.95)
    ba_h = ba_stats['cohens_h']
    ba_h_ci = (ba_stats['h_ci_low'], ba_stats['h_ci_high'])

    
    return dict(
        lift=lift, lift_ci=ci_lift, p_lift=p_lift,
        normed_lift=normed_lift, normed_lift_ci=ci_normed_lift,
        mcc=mcc, mcc_ci=ci_mcc, p_mcc=p_mcc,
        single_point_auc=ba_h,
        single_point_auc_ci=ba_h_ci,
        full_auc=full_auc,
        full_auc_ci=ci_full_auc if full_auc is not None else (None, None),
        calibration_auc=calibration_auc,
        calibration_auc_ci=ci_calibration_auc if calibration_auc is not None else (None, None)
    )

def self_acc_stats(cap_corr, team_corr, kept_mask):
    k           = kept_mask.sum()                
    s           = team_corr[kept_mask].sum()      
    p0          = cap_corr.mean()    

    p_val = ss.binomtest(s, k, p0, alternative='two-sided').pvalue
    lo, hi = smp.proportion_confint(s, k, alpha=0.05, method='wilson')
    lift    = s/k - p0
    lift_lo = lo   - p0
    lift_hi = hi   - p0
    return lift, lift_lo, lift_hi, p_val

def self_acc_stats_boot(baseline_correct, kept_correct, kept_mask, n_boot=2000, seed=0):
    """
    baseline_correct : 1/0[N]   baseline correctness for *every* item
    kept_correct     : 1/0[N]   correctness *actually achieved in game*
    kept_mask        : bool[N]  True where the model answered itself
    """
    A_base = np.nanmean(baseline_correct)
    A_kept = kept_correct[kept_mask].mean() if kept_mask.any() else np.nan
    lift   = A_kept - A_base

    # paired bootstrap
    rng  = np.random.default_rng(seed)
    idx0 = np.arange(len(baseline_correct))
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(idx0, len(idx0), replace=True)
        A_b = np.nanmean(baseline_correct[idx])
        km  = kept_mask[idx]
        A_k = kept_correct[idx][km].mean() if km.any() else 0
        boots.append(A_k - A_b)

    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two  = 2 * min(np.mean(np.array(boots) <= 0),
                     np.mean(np.array(boots) >= 0))
    return lift, lo, hi, p_two

def delegate_gap_stats(TP, FN, FP, TN):
    def wilson(p, n, alpha=0.05):
        return smp.proportion_confint(count=p*n, nobs=n, alpha=alpha, method='wilson')

    n_wrong, n_right = TP+FN, FP+TN
    p_del_wrong  = TP / n_wrong
    p_del_right  = FP / n_right
    delta_d      = p_del_wrong - p_del_right

    lo1, hi1 = wilson(p_del_wrong,  n_wrong)   # CI for P(delegate|wrong)
    lo2, hi2 = wilson(p_del_right,  n_right)   # CI for P(delegate|right)
    ci_low  = delta_d - np.sqrt((p_del_wrong-lo1)**2 + (hi2-p_del_right)**2)
    ci_high = delta_d + np.sqrt((hi1-p_del_wrong)**2 + (p_del_right-lo2)**2)

    table = [[TP, FN],   # rows: wrong/right ; cols: delegate/keep
            [FP, TN]]
    chi2, p_val, *_ = ss.chi2_contingency(table, correction=False)
    return delta_d, ci_low, ci_high, p_val

def mcc_ci_boot(TP, FN, FP, TN):
    mcc = (TP*TN - FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    score = (mcc + 1)/2

    N = TP+FN+FP+TN
    wrong = np.array([1]*TP + [1]*FN + [0]*FP + [0]*TN, dtype=bool)   # 1 = model would be wrong
    dele  = np.array([1]*TP + [0]*FN + [1]*FP + [0]*TN, dtype=bool)   # 1 = model delegated

    boot=[]
    rng = np.random.default_rng(0)
    for _ in range(2000):
        idx = rng.choice(N, N, replace=True)
        tp = np.sum(wrong[idx] &  dele[idx])
        tn = np.sum(~wrong[idx] & ~dele[idx])
        fp = np.sum(~wrong[idx] &  dele[idx])
        fn = np.sum(wrong[idx] & ~dele[idx])
        denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        boot_mcc = (tp*tn - fp*fn)/denom if denom else 0
        boot.append(boot_mcc)
    ci = np.percentile(boot, [2.5,97.5])
    return mcc, score, ci

LOG_METRICS_TO_EXTRACT = [
    "Delegation to teammate occurred",
    "Phase 1 self-accuracy (from completed results, total - phase2)",
    "Phase 2 self-accuracy",
    "Statistical test (P2 self vs P1)"
]

LOG_METRIC_PATTERNS = {
    "Delegation to teammate occurred": re.compile(r"^\s*Delegation to teammate occurred in (.*)$"),
    "Phase 1 self-accuracy (from completed results, total - phase2)": re.compile(r"^\s*Phase 1 self-accuracy \(from completed results, total - phase2\): (.*)$"),
    "Phase 2 self-accuracy": re.compile(r"^\s*Phase 2 self-accuracy: (.*)$"),
    "Statistical test (P2 self vs P1)": re.compile(r"^\s*Statistical test \(P2 self vs P1\): (.*)$")
}

def extract_log_file_metrics(log_filepath):
    extracted_log_metrics = {key: "Not found" for key in LOG_METRICS_TO_EXTRACT}
    try:
        with open(log_filepath, 'r') as f:
            for line in f:
                for metric_name, pattern in LOG_METRIC_PATTERNS.items():
                    match = pattern.match(line)
                    if match:
                        extracted_log_metrics[metric_name] = match.group(1).strip()
                        if all(val != "Not found" for val in extracted_log_metrics.values()):
                            return extracted_log_metrics
    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_filepath}")
    except Exception as e:
        print(f"An error occurred while reading log file {log_filepath}: {e}")
    return extracted_log_metrics


def compare_predictors_of_answer(stated_confs, implicit_confs, pass_decisions):
    """Compare which confidence measure better predicts passing."""
    mask = ~(np.isnan(stated_confs) | np.isnan(implicit_confs) | np.isnan(pass_decisions))
    stated_confs = stated_confs[mask]
    implicit_confs = implicit_confs[mask]
    pass_decisions = pass_decisions[mask]
    #stated_ranks = rankdata(stated_confs) / len(stated_confs)
    #implicit_ranks = rankdata(implicit_confs) / len(implicit_confs)
    #X_stated = stated_ranks.reshape(-1, 1)
    #X_implicit = implicit_ranks.reshape(-1, 1)          
    X_stated = np.log((stated_confs+1e-6)/(1-stated_confs+1e-6)).reshape(-1, 1)
    eps = 1e-6 
    p     = np.clip(implicit_confs.astype(float), eps, 1 - eps)
    X_implicit = np.log(p / (1 - p)).reshape(-1, 1) 

    #X_stated = stated_confs.reshape(-1, 1)
    #X_implicit = implicit_confs.reshape(-1, 1)
    X_both = np.column_stack([stated_confs, implicit_confs])
    y = pass_decisions
    
    # Fit three models
    lr_stated = LogisticRegression().fit(X_stated, y)
    lr_implicit = LogisticRegression().fit(X_implicit, y)  
    lr_both = LogisticRegression().fit(X_both, y)
    
    # Cross-validated AUC scores
    auc_stated = cross_val_score(LogisticRegression(), X_stated, y, 
                                cv=5, scoring='roc_auc').mean()
    auc_implicit = cross_val_score(LogisticRegression(), X_implicit, y,
                                    cv=5, scoring='roc_auc').mean()
    auc_both = cross_val_score(LogisticRegression(), X_both, y,
                                cv=5, scoring='roc_auc').mean()
    
    # Get log-likelihoods 
    from sklearn.metrics import log_loss
    ll_stated = -log_loss(y, lr_stated.predict_proba(X_stated)[:,1], normalize=False)
    ll_implicit = -log_loss(y, lr_implicit.predict_proba(X_implicit)[:,1], normalize=False)
    ll_both = -log_loss(y, lr_both.predict_proba(X_both)[:,1], normalize=False)
    
    # Likelihood ratio test: does implicit add to stated?
    lr_stat_implicit_adds = 2 * (ll_both - ll_stated)
    p_value_implicit_adds = 1 - chi2.cdf(lr_stat_implicit_adds, df=1)
    
    # Likelihood ratio test: does stated add to implicit?
    lr_stat_stated_adds = 2 * (ll_both - ll_implicit)
    p_value_stated_adds = 1 - chi2.cdf(lr_stat_stated_adds, df=1)
    
    # Get standardized coefficients for interpretation
    X_both_std = (X_both - X_both.mean(axis=0)) / X_both.std(axis=0)
    lr_std = LogisticRegression().fit(X_both_std, y)
    
    results = {
        'auc_stated': auc_stated,
        'auc_implicit': auc_implicit,
        'auc_both': auc_both,
        'coef_stated': lr_std.coef_[0][0],
        'coef_implicit': lr_std.coef_[0][1],
        'p_implicit_adds_to_stated': p_value_implicit_adds,
        'p_stated_adds_to_implicit': p_value_stated_adds,
    }
    
    return results

def compare_predictors_of_implicit_conf(stated, behavior, implicit_confs):
    mask = ~(np.isnan(stated) | np.isnan(implicit_confs) | np.isnan(behavior))
    stated = stated[mask]
    implicit_confs = implicit_confs[mask]
    behavior = behavior[mask]
    eps = 1e-6 
    implicit_confs = np.clip(implicit_confs.astype(float), eps, 1 - eps)
    corr_actual, p_actual = pointbiserialr(behavior, implicit_confs)
    if stated.dtype == np.dtype('int'):
        corr_stated, p_stated = pointbiserialr(stated, implicit_confs)
    else:
        corr_stated, p_stated = pearsonr(stated, implicit_confs)
    # Test if correlations are significantly different using Fisher's z-transformation
    z_actual = np.arctanh(corr_actual)
    z_stated = np.arctanh(corr_stated)
    z_diff = (z_actual - z_stated) / np.sqrt(2/(len(implicit_confs)-3))
    p_diff = 2*(1 - norm.cdf(abs(z_diff)))
    return {
        'corr_actual': corr_actual,
        'p_actual': p_actual,
        'corr_stated': corr_stated,
        'p_stated': p_stated,
        'p_diff': p_diff
    }

def remove_collinear_terms(model_terms, df, target_col='delegate_choice', fit_kwargs={}, protected_terms=['s_i_capability']):
    """
    Iteratively remove terms causing singular matrix issues
    """
    working_terms = model_terms.copy()
    removed_terms = []
    
    while True:
        model_def_str = f'{target_col} ~ ' + ' + '.join(working_terms)
        
        try:
            logit_model = smf.logit(model_def_str, data=df).fit(**fit_kwargs)
            return logit_model, working_terms, removed_terms
            
        except (LinAlgError, np.linalg.LinAlgError):
            # Try removing each term EXCEPT protected ones
            for term in working_terms:
                if term in protected_terms:
                    continue  # Skip protected terms
                    
                test_terms = [t for t in working_terms if t != term]
                model_def_str = f'{target_col} ~ ' + ' + '.join(test_terms)
                
                try:
                    smf.logit(model_def_str, data=df).fit(**fit_kwargs)
                    # This worked, so remove this term
                    working_terms.remove(term)
                    removed_terms.append(term)
                    print(f"Removed: {term}")
                    break
                except:
                    continue
            else:
                print("Can't remove any more terms")
                return None, working_terms, removed_terms

def introspection_metrics(D, U, X=None, n_boot=2000, seed=0, C=1.0):
    """
    Odds ratio per 1 SD of U (with optional controls) and bootstrap 95% CI.
    - D: array-like of 0/1 (1 = answer, 0 = pass)
    - U: array-like (e.g., -entropy so higher = more confidence)
    - X: optional controls (array-like or DataFrame)
         Numeric controls are standardized; categorical/bool are one-hot.
    - C: L2 regularization strength for logistic regression
    Returns: {'or_per_sd': float, 'or_ci': (low, high)}
    """
    from sklearn.utils import check_random_state
    from pandas.api.types import is_numeric_dtype, is_bool_dtype

    # Assemble and clean
    D = np.asarray(D)
    U = np.asarray(U, dtype=float)
    df = pd.DataFrame({'D': D, 'U': U})
    if X is not None:
        X_df = pd.DataFrame(X)
        df = pd.concat([df, X_df], axis=1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = df['D'].to_numpy().astype(int)
    if y.min() == y.max():
        raise ValueError("D must contain both 0 and 1 after cleaning.")

    # Standardize U (per-SD), fixed for bootstrap
    U_mean = df['U'].mean()
    U_std = df['U'].std(ddof=0)
    if not np.isfinite(U_std) or U_std == 0:
        raise ValueError("U has zero/invalid variance.")
    Uz = ((df['U'] - U_mean) / U_std).to_numpy().reshape(-1, 1)

    # Controls: split into numeric (excluding bool) and categorical/bool
    control_cols = [c for c in df.columns if c not in ['D', 'U']]
    if control_cols:
        num_cols = [c for c in control_cols if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
        cat_cols = [c for c in control_cols if c not in num_cols]

        # Standardize numeric controls (drop zero-variance)
        if num_cols:
            num_means = df[num_cols].mean()
            num_stds = df[num_cols].std(ddof=0).replace(0, np.nan)
            num_keep = num_stds.index[num_stds.notna()].tolist()
            Z_num = ((df[num_keep] - num_means[num_keep]) / num_stds[num_keep]).to_numpy()
        else:
            num_keep, Z_num = [], None

        # One-hot encode categorical/bool controls (fixed columns for bootstrap)
        if cat_cols:
            X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
            cat_dummy_cols = X_cat.columns.tolist()
            X_cat_np = X_cat.to_numpy()
        else:
            cat_dummy_cols, X_cat_np = [], None

        # Build design matrix
        parts = [Uz]
        if Z_num is not None: parts.append(Z_num)
        if X_cat_np is not None: parts.append(X_cat_np)
        X_mat = np.hstack(parts)
    else:
        num_keep, cat_cols, cat_dummy_cols = [], [], []
        X_mat = Uz

    # Point estimate
    logit = LogisticRegression(
        penalty='l2', C=C, solver='lbfgs',
        fit_intercept=True, max_iter=2000
    )
    logit.fit(X_mat, y)
    beta = float(logit.coef_[0][0])
    or_per_sd = float(np.exp(beta))

    # Bootstrap CI (percentile), reusing global scales and dummy columns
    rng = check_random_state(seed)
    n = len(df)
    idx_all = np.arange(n)
    or_samples = []

    U_np = df['U'].to_numpy()
    if num_keep:
        NUM_np = df[num_keep].to_numpy()
        num_means_vec = num_means[num_keep].to_numpy()
        num_stds_vec = num_stds[num_keep].to_numpy()
    if cat_cols:
        CAT_df = df[cat_cols]

    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        ys = y[idx]
        if ys.min() == ys.max():
            continue

        Uz_b = ((U_np[idx] - U_mean) / U_std).reshape(-1, 1)

        parts = [Uz_b]
        if num_keep:
            Z_num_b = (NUM_np[idx] - num_means_vec) / num_stds_vec
            parts.append(Z_num_b)
        if cat_cols:
            X_cat_b = pd.get_dummies(CAT_df.iloc[idx], drop_first=True).reindex(columns=cat_dummy_cols, fill_value=0)
            parts.append(X_cat_b.to_numpy())

        Xm = np.hstack(parts)
        try:
            logit.fit(Xm, ys)
            b = float(logit.coef_[0][0])
            or_samples.append(np.exp(b))
        except Exception:
            continue

    if len(or_samples) == 0:
        or_ci = (float('nan'), float('nan'))
    else:
        lo = float(np.percentile(or_samples, 2.5))
        hi = float(np.percentile(or_samples, 97.5))
        or_ci = (lo, hi)

    return {'or_per_sd': or_per_sd, 'or_ci': or_ci}


def fraction_headroom_auc(D, U, X=None, n_boot=2000, seed=0, C=1.0):
    """
    Compute:
      - fraction_headroom = (AUC_full − AUC_ctrl) / (1 − AUC_ctrl), with 95% CI
      - auc_full with 95% CI
    Inputs:
      - D: 0/1 labels (1=answer, 0=pass)
      - U: score (e.g., −entropy)
      - X: optional controls (DataFrame or array-like)
      - C: L2 strength for logistic regression
    Returns:
      {'fraction_headroom', 'fraction_headroom_ci', 'auc_full', 'auc_full_ci'}
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import check_random_state
    from pandas.api.types import is_numeric_dtype, is_bool_dtype

    rng = check_random_state(seed)

    # Assemble and clean
    D = np.asarray(D)
    U = np.asarray(U, dtype=float)
    df = pd.DataFrame({'D': D, 'U': U})
    if X is not None:
        X_df = pd.DataFrame(X)
        df = pd.concat([df, X_df], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    y = df['D'].to_numpy().astype(int)
    if y.min() == y.max():
        raise ValueError("D must contain both 0 and 1 after cleaning.")
    U_vec = df['U'].to_numpy()

    # Controls preprocessing: standardize numeric, one-hot categorical/bool
    control_cols = [c for c in df.columns if c not in ['D', 'U']]
    if control_cols:
        num_cols = [c for c in control_cols if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
        cat_cols = [c for c in control_cols if c not in num_cols]

        if num_cols:
            num_means = df[num_cols].mean()
            num_stds = df[num_cols].std(ddof=0).replace(0, np.nan)
            num_keep = num_stds.index[num_stds.notna()].tolist()
            Z_num = ((df[num_keep] - num_means[num_keep]) / num_stds[num_keep]).to_numpy()
            NUM_np = df[num_keep].to_numpy()
            num_means_vec = num_means[num_keep].to_numpy()
            num_stds_vec = num_stds[num_keep].to_numpy()
        else:
            num_keep, Z_num = [], None
            NUM_np = num_means_vec = num_stds_vec = None

        if cat_cols:
            X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
            cat_dummy_cols = X_cat.columns.tolist()
            X_cat_np = X_cat.to_numpy()
            CAT_df = df[cat_cols]
        else:
            cat_dummy_cols, X_cat_np = [], None
            CAT_df = None

        parts = []
        if Z_num is not None: parts.append(Z_num)
        if X_cat_np is not None: parts.append(X_cat_np)
        X_ctrl = np.hstack(parts) if parts else None
    else:
        num_keep, cat_cols, cat_dummy_cols = [], [], []
        X_ctrl = None
        NUM_np = num_means_vec = num_stds_vec = None
        CAT_df = None

    # Standardize U with full-sample stats (fixed for bootstrap)
    U_mean = U_vec.mean()
    U_std = U_vec.std(ddof=0)
    if not np.isfinite(U_std) or U_std == 0:
        raise ValueError("U has zero/invalid variance.")
    U_z = ((U_vec - U_mean) / U_std).reshape(-1, 1)

    def fit_pred(Xm, ys):
        lr = LogisticRegression(penalty='l2', C=C, solver='lbfgs',
                                fit_intercept=True, max_iter=2000)
        lr.fit(Xm, ys)
        return lr.predict_proba(Xm)[:, 1]

    # Point estimates
    if X_ctrl is None:
        auc_ctrl = 0.5
    else:
        p_ctrl = fit_pred(X_ctrl, y)
        auc_ctrl = roc_auc_score(y, p_ctrl)

    X_full = U_z if X_ctrl is None else np.hstack([U_z, X_ctrl])
    p_full = fit_pred(X_full, y)
    auc_full = roc_auc_score(y, p_full)

    delta = auc_full - auc_ctrl
    headroom = 1.0 - auc_ctrl
    frac_headroom = delta / headroom if headroom > 0 else float('nan')

    # Bootstrap CIs
    auc_full_s = []
    frac_s = []
    n = len(y)
    idx_all = np.arange(n)

    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        ys = y[idx]
        if ys.min() == ys.max():
            continue

        # Controls for bootstrap sample using fixed transforms/columns
        if X_ctrl is None:
            auc_ctrl_b = 0.5
            Xm = None
        else:
            parts = []
            if num_keep:
                Zb = (NUM_np[idx] - num_means_vec) / num_stds_vec
                parts.append(Zb)
            if CAT_df is not None:
                X_cat_b = pd.get_dummies(CAT_df.iloc[idx], drop_first=True)\
                           .reindex(columns=cat_dummy_cols, fill_value=0).to_numpy()
                parts.append(X_cat_b)
            Xm = np.hstack(parts) if parts else None
            p_ctrl_b = fit_pred(Xm, ys)
            auc_ctrl_b = roc_auc_score(ys, p_ctrl_b)

        Ub = U_vec[idx]
        Uz_b = ((Ub - U_mean) / U_std).reshape(-1, 1)
        Xf = Uz_b if Xm is None else np.hstack([Uz_b, Xm])
        p_full_b = fit_pred(Xf, ys)
        auc_full_b = roc_auc_score(ys, p_full_b)

        delta_b = auc_full_b - auc_ctrl_b
        headroom_b = 1.0 - auc_ctrl_b
        frac_b = (delta_b / headroom_b) if headroom_b > 0 else np.nan

        auc_full_s.append(auc_full_b)
        frac_s.append(frac_b)

    def ci(arr):
        arr = [x for x in arr if np.isfinite(x)]
        if not arr:
            return (float('nan'), float('nan'))
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        'fraction_headroom': float(frac_headroom),
        'fraction_headroom_ci': ci(frac_s),
        'auc_full': float(auc_full),
        'auc_full_ci': ci(auc_full_s),
    }

from statsmodels.api import Logit, add_constant
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.api import Logit, add_constant

def logit_on_decision(pass_decision, iv_of_interest=None, control_vars=None):
    """
    Analyze a binary decision via logistic regression with adaptable predictors.

    This version correctly differentiates between continuous variables (which are
    standardized) and binary 0/1 variables (which are left as-is).

    Args:
        pass_decision (pd.Series): The dependent variable with 1/0 values.
        iv_of_interest (pd.Series, optional): The primary independent variable.
        control_vars (pd.DataFrame or list, optional): Control variables.

    Returns:
        statsmodels.results.api.RegressionResultsWrapper: The fitted regression results.
    """
    if iv_of_interest is None and (control_vars is None or len(control_vars) == 0):
        raise ValueError("At least one predictor (iv_of_interest or control_vars) must be provided.")

    if isinstance(control_vars, list): control_vars = pd.concat(control_vars, axis=1) if control_vars else None

    # --- Combine all data and handle missing values ---
    data_to_combine = [pass_decision.rename('pass_decision')]
    if iv_of_interest is not None:
        data_to_combine.append(iv_of_interest.rename(iv_of_interest.name or 'iv_of_interest'))
    if control_vars is not None and not control_vars.empty:
        data_to_combine.append(control_vars)
    
    model_data = pd.concat(data_to_combine, axis=1).dropna()

    if model_data.empty:
        raise ValueError("No complete cases remain after handling missing values.")

    y = model_data['pass_decision']
    X_raw = model_data.drop(columns=['pass_decision'])

    # --- Differentiate between continuous, binary, and categorical predictors ---
    continuous_cols = []
    binary_cols = []
    
    # First, separate numeric from non-numeric
    numeric_cols = X_raw.select_dtypes(include=np.number).columns
    categorical_cols = list(X_raw.select_dtypes(include=['object', 'category', 'string']).columns)

    # From the numeric columns, identify which are binary vs. continuous
    for col in numeric_cols:
        # If 2 or fewer unique values, treat as binary/categorical, not for scaling
        if X_raw[col].nunique() <= 2:
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    # --- Process each predictor type correctly ---
    # 1. Standardize ONLY the continuous variables
    if continuous_cols:
        scaler = StandardScaler()
        X_continuous = pd.DataFrame(scaler.fit_transform(X_raw[continuous_cols]),
                                    index=X_raw.index, columns=continuous_cols)
    else:
        X_continuous = pd.DataFrame(index=X_raw.index)

    # 2. Dummy-encode the true categorical variables
    if categorical_cols:
        X_categorical = pd.get_dummies(X_raw[categorical_cols], drop_first=True, dtype=float)
    else:
        X_categorical = pd.DataFrame(index=X_raw.index)

    # 3. Keep the binary variables as they are (no transformation)
    X_binary = X_raw[binary_cols]

    # Recombine into the final predictor matrix
    X = pd.concat([X_continuous, X_binary, X_categorical], axis=1)
    X = add_constant(X, has_constant='add')

    # --- Fit Model ---
    model = Logit(y, X)
    results = model.fit(disp=0)
    
    return results

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def partial_correlation_on_decision_orig(dv_series, iv_series, control_series_list):
    """
    Compute partial correlation between iv and dv, controlling for surface features.
    
    Args:
        iv_series: pd.Series of independent variable (e.g., baseline confidence)
        dv_series: pd.Series of dependent variable (e.g., pass/answer decision)
        control_series_list: list of pd.Series with control variables
    
    Returns:
        dict with 'correlation', 'p_value', 'n_samples', 'ci_lower', 'ci_upper'
    """
    # Combine all series into a dataframe, aligning indices
    all_data = pd.DataFrame({
        'iv': iv_series,
        'dv': dv_series
    })
    
    # Add control variables
    for i, control in enumerate(control_series_list):
        col_name = control.name if control.name else f'control_{i}'
        all_data[col_name] = control
    
    # Drop rows with any missing values
    all_data = all_data.dropna()
    n_samples = len(all_data)
    
    if n_samples < 3:
        raise ValueError(f"Not enough samples after removing NaNs: {n_samples}")
    
    # Prepare control matrix
    control_cols = [col for col in all_data.columns if col not in ['iv', 'dv']]
    X_controls_list = []
    
    for col in control_cols:
        if all_data[col].dtype in ['object', 'category']:
            # Dummy encode categorical
            dummies = pd.get_dummies(all_data[col], prefix=col, drop_first=True).astype(float)
            X_controls_list.append(dummies)
        else:
            # Standardize continuous
            scaler = StandardScaler()
            standardized = pd.DataFrame(
                scaler.fit_transform(all_data[[col]]),
                columns=[col],
                index=all_data.index
            )
            X_controls_list.append(standardized)
    
    # Combine all controls
    if X_controls_list:
        X_controls = pd.concat(X_controls_list, axis=1)
    else:
        X_controls = pd.DataFrame(index=all_data.index)
    
    # Add intercept
    X_controls.insert(0, 'const', 1.0)
    
    # Convert to float arrays
    X = X_controls.values.astype(float)
    y_iv = all_data['iv'].values.astype(float)
    y_dv = all_data['dv'].values.astype(float)
    
    # Regress iv and dv on controls to get residuals
    from numpy.linalg import lstsq
    
    # IV residuals
    coef_iv = lstsq(X, y_iv, rcond=None)[0]
    iv_residuals = y_iv - X @ coef_iv
    
    # DV residuals  
    coef_dv = lstsq(X, y_dv, rcond=None)[0]
    dv_residuals = y_dv - X @ coef_dv
    
    # Compute correlation between residuals
    corr, p_value = stats.pearsonr(iv_residuals, dv_residuals)
    
    # Compute 95% CI using Fisher z-transformation
    z = np.arctanh(corr)
    se = 1 / np.sqrt(n_samples - len(X_controls.columns) - 2)
    z_ci = [z - 1.96*se, z + 1.96*se]
    ci = np.tanh(z_ci)
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'n_samples': n_samples,
        'ci_lower': ci[0],
        'ci_upper': ci[1]
    }

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def collapse_categorical(s: pd.Series, max_levels=8, min_count=10):
    s = s.astype('category')
    counts = s.value_counts(dropna=False)
    # Keep top max_levels, also ensure any level with count >= min_count is kept
    keep = set(counts.nlargest(max_levels).index)
    keep |= set(counts[counts >= min_count].index)
    s2 = s.where(s.isin(keep), other='Collapsed')
    return s2.astype('category')

def prune_numeric_by_variance(df_num: pd.DataFrame, zero_var_tol=1e-12, nzv_prop=0.95):
    keep = []
    for c in df_num.columns:
        x = df_num[c].astype(float).values
        if np.std(x, ddof=0) <= zero_var_tol:
            continue
        # near-zero-variance by dominance of one value (for discretized vars)
        vals, counts = np.unique(x, return_counts=True)
        if counts.max() / len(x) >= nzv_prop:
            continue
        keep.append(c)
    return df_num[keep]

def prune_numeric_by_correlation(df_num: pd.DataFrame, corr_thresh=0.9):
    if df_num.shape[1] <= 1:
        return df_num
    corr = df_num.corr().abs().fillna(0.0)
    keep = []
    dropped = set()
    for c in corr.columns:
        if c in dropped:
            continue
        keep.append(c)
        # drop others highly correlated with c
        drop_these = corr.index[(corr[c] >= corr_thresh) & (corr.index != c)].tolist()
        dropped.update(drop_these)
    return df_num[keep]

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def simplify_controls(series_list, max_levels=8, min_count=None, eps=1e-12):  
    """
    Accepts a list of pandas Series (one per control) and returns a simplified DataFrame:
      - Numeric: drops near-constant columns.
      - Categorical/object: collapses rare levels to 'Collapsed' safely (adds category first),
        removes single-level vars.

    Parameters
    ----------
    series_list : list[pd.Series]
        Controls to simplify. Each Series should have a .name.
    max_levels : int or None
        Keep at most this many most frequent levels per categorical (None = no cap).
    min_count : int or None
        Minimum count to keep a level (None = no threshold).
    eps : float
        Threshold for "near-constant" numeric std.
    """
    parts = []
    for i, s in enumerate(series_list):
        if s is None:
            continue
        s = pd.Series(s).copy()
        name = s.name if s.name is not None else f'ctrl_{i}'
        s = s.replace([np.inf, -np.inf], np.nan)

        if is_numeric_dtype(s):
            # Drop near-constant or single-unique numerics
            x = pd.to_numeric(s, errors='coerce').astype(float)
            if x.std(ddof=0) <= eps or x.nunique(dropna=True) <= 1:
                continue
            x.name = name
            parts.append(x)
            continue

        # Categorical branch: collapse rare levels to 'Other'
        sc = s.astype('category').cat.remove_unused_categories()

        # If everything is NaN or only one level, drop
        if sc.nunique(dropna=True) <= 1:
            continue

        counts = sc.value_counts(dropna=True)

        # Decide which levels to keep
        if min_count is not None:
            keep = counts[counts >= min_count].index
            if max_levels is not None and len(keep) > max_levels:
                # If too many still, cap to most frequent among those
                keep = counts.loc[keep].nlargest(max_levels).index
        else:
            # Keep top-k by frequency
            keep = counts.nlargest(max_levels).index if max_levels is not None else counts.index

        # Ensure 'Collapsed' exists before relabeling
        if 'Collapsed' not in sc.cat.categories:
            sc = sc.cat.add_categories('Collapsed')

        # Relabel rare levels to 'Other' (preserve NaN)
        sc = sc.where(sc.isna() | sc.isin(keep), 'Collapsed')

        # Clean up categories again
        sc = sc.cat.remove_unused_categories()
        if sc.nunique(dropna=True) <= 1:
            # If collapsing made it single-level, drop
            continue

        sc.name = name
        parts.append(sc)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, axis=1)
    return out

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats

def partial_correlation_on_decision(dv_series, iv_series, control_series_list, eps=1e-12):
    """
    Compute partial correlation between iv and dv, controlling for surface features.

    Args:
        dv_series: pd.Series (dependent variable; can be binary)
        iv_series: pd.Series (independent variable)
        control_series_list: list of pd.Series with control variables

    Returns:
        dict with 'correlation', 'p_value', 'n_samples', 'ci_lower', 'ci_upper'
    """
    # Combine and clean
    all_data = pd.DataFrame({'iv': iv_series, 'dv': dv_series})
    for i, s in enumerate(control_series_list):
        name = getattr(s, 'name', None) or f'control_{i}'
        all_data[name] = s

    all_data = all_data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    n = len(all_data)
    if n < 5:
        raise ValueError(f"Not enough samples after cleaning: n={n}")

    control_names = [c for c in all_data.columns if c not in ['iv', 'dv']]
    control_df = simplify_controls([all_data[c] for c in control_names])

    # Align iv/dv with controls (single listwise deletion)
    df = pd.concat([all_data[['iv', 'dv']], control_df], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    ctrl = df.drop(columns=['iv', 'dv'])

    # Build control design matrix X (intercept + effective controls)
    parts = []
    for c in ctrl.columns:
        s = ctrl[c]
        if is_numeric_dtype(s):
            x = s.astype(float).values
            sd = x.std(ddof=0)
            if sd <= eps:
                continue
            zx = (x - x.mean()) / sd
            parts.append(pd.DataFrame({c: zx}, index=ctrl.index))
        else:
            sc = s.astype('category')
            if sc.cat.categories.size <= 1:
                continue
            dummies = pd.get_dummies(sc, prefix=c, drop_first=True, dtype=float)
            parts.append(dummies)

    X_controls = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ctrl.index)
    X_controls.insert(0, 'const', 1.0)
    X = X_controls.values.astype(float)

    # Effective number of controls = rank(X) - 1 (exclude intercept)
    rank_X = np.linalg.matrix_rank(X)
    k_eff = max(rank_X - 1, 0)

    # Residualize iv and dv on X
    coef_iv, _, _, _ = np.linalg.lstsq(X, all_data['iv'].astype(float).values, rcond=None)
    coef_dv, _, _, _ = np.linalg.lstsq(X, all_data['dv'].astype(float).values, rcond=None)
    iv_res = all_data['iv'].astype(float).values - X @ coef_iv
    dv_res = all_data['dv'].astype(float).values - X @ coef_dv

    # Guard against zero variance in residuals
    s_iv = np.std(iv_res, ddof=0)
    s_dv = np.std(dv_res, ddof=0)
    if s_iv <= eps or s_dv <= eps:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'n_samples': n,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'k_controls_effective': k_eff,
            'rank_X': rank_X
        }

    # Partial correlation = Pearson r between residuals
    r = np.corrcoef(iv_res, dv_res)[0, 1]
    r = float(np.clip(r, -1.0, 1.0))

    # p-value via t-test for partial correlation
    # df_t = n - k - 2 (k excludes intercept)
    df_t = n - k_eff - 2
    if df_t > 0 and abs(r) < 1:
        t_stat = r * np.sqrt(df_t / (1 - r**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_t))
    else:
        t_stat = np.nan
        p_value = np.nan

    # 95% CI via Fisher z with SE = 1/sqrt(n - k - 3)
    df_z = n - k_eff - 3
    if df_z > 0 and abs(r) < 1:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(df_z)
        ci_z = (z - 1.96 * se, z + 1.96 * se)
        ci = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
    else:
        ci = (np.nan, np.nan)

    return {
        'correlation': r,
        'p_value': p_value,
        'n_samples': n,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'k_controls_effective': k_eff,
        'rank_X': rank_X,
        'df_t': df_t,
        'df_z': df_z
    }

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS

def regression_std(dv_series, iv_series, control_series_list):
    """
    Regress dv on iv and/or controls.
    Returns:
        dict with 'coefficient', 'p_value', 'ci_lower', 'ci_upper', 'r_squared', 'n_samples'
    """
    if iv_series is None and (control_series_list is None or len(control_series_list) == 0):
        raise ValueError("Must provide either iv_series or control variables (or both)")
    
    # Get name for iv if provided
    iv_name = iv_series.name if iv_series is not None and iv_series.name else 'iv'
    dv_name = dv_series.name if dv_series.name else 'dv'
    
    # Start building dataframe
    all_data = pd.DataFrame({dv_name: dv_series})
    
    # Add IV if provided
    if iv_series is not None:
        all_data[iv_name] = iv_series
    
    # Add control variables if provided
    if control_series_list is not None:
        for i, control in enumerate(control_series_list):
            col_name = control.name if control.name else f'control_{i}'
            all_data[col_name] = control
    
    # Drop rows with any missing values
    all_data = all_data.dropna()
    n_samples = len(all_data)
    
    if n_samples < 3:
        raise ValueError(f"Not enough samples after removing NaNs: {n_samples}")
    
    # Prepare predictor matrix
    X_list = []
    
    # Add IV if present (not standardized to keep interpretable units)
    if iv_series is not None:
        X_list.append(all_data[[iv_name]])
    
    # Add controls if present
    if control_series_list is not None and len(control_series_list) > 0:
        control_cols = [col for col in all_data.columns if col not in [iv_name, dv_name]]
        
        for col in control_cols:
            if all_data[col].dtype in ['object', 'category']:
                # Dummy encode categorical
                dummies = pd.get_dummies(all_data[col], prefix=col, drop_first=True).astype(float)
                X_list.append(dummies)
            else:
                # Standardize continuous controls
                scaler = StandardScaler()
                standardized = pd.DataFrame(
                    scaler.fit_transform(all_data[[col]]),
                    columns=[col],
                    index=all_data.index
                )
                X_list.append(standardized)
    
    # Combine all predictors
    X = pd.concat(X_list, axis=1) if X_list else pd.DataFrame(index=all_data.index)
    
    # Add intercept
    X.insert(0, 'const', 1.0)
    
    # Run regression
    model = OLS(all_data[dv_name].astype(float), X.astype(float))
    results = model.fit()
    
    # Extract results for IV if present
    if iv_series is not None:
        iv_coef = results.params[iv_name]
        iv_pval = results.pvalues[iv_name]
        iv_ci = results.conf_int().loc[iv_name]
        
        return {
            'coefficient': iv_coef,
            'p_value': iv_pval,
            'ci_lower': iv_ci[0],
            'ci_upper': iv_ci[1],
            'r_squared': results.rsquared,
            'n_samples': n_samples,
            'full_results': results
        }
    else:
        # No IV, just return model fit stats
        return {
            'coefficient': None,
            'p_value': None,
            'ci_lower': None,
            'ci_upper': None,
            'r_squared': results.rsquared,
            'n_samples': n_samples,
            'full_results': results
        }
    
import numpy as np
import pandas as pd

def brier_ece(correctness_series, probability_series, n_bins=10, n_bootstrap=1000):
    """
    Compute Murphy decomposition of Brier score into reliability and resolution.
    
    Args:
        correctness_series: pd.Series with 1=correct, 0=incorrect
        probability_series: pd.Series with probability of chosen answer
        n_bins: number of bins for decomposition
        n_bootstrap: number of bootstrap samples for CIs
    
    Returns:
        dict with Brier components and confidence intervals
    """
    # Align and drop NaNs
    data = pd.DataFrame({
        'correct': correctness_series,
        'prob': probability_series
    }).dropna()
    
    n_samples = len(data)
    base_rate = data['correct'].mean()

    def compute_decomposition(df):
        """Helper function to compute decomposition for a dataset"""
        # Per-sample base rate (fix)
        base_rate = df['correct'].mean()

        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        reliability = 0.0
        resolution = 0.0
        
        for i in range(n_bins):
            bin_mask = (df['prob'] >= bin_boundaries[i]) & (df['prob'] < bin_boundaries[i+1])
            n_bin = bin_mask.sum()
            
            if n_bin > 0:
                bin_prob = df.loc[bin_mask, 'prob'].mean()
                bin_freq = df.loc[bin_mask, 'correct'].mean()
                bin_weight = n_bin / len(df)
                
                reliability += bin_weight * (bin_prob - bin_freq) ** 2
                resolution += bin_weight * (bin_freq - base_rate) ** 2
        
        # Uncertainty uses the per-sample base rate (fix)
        uncertainty = base_rate * (1 - base_rate)
        
        # Brier score
        brier = ((df['prob'] - df['correct']) ** 2).mean()
        
        return {
            'brier': brier,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty
        }

    # Compute for actual data
    results = compute_decomposition(data)
    
    # Bootstrap for confidence intervals
    bootstrap_results = {
        'brier': [],
        'reliability': [],
        'resolution': []
    }
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_data = data.iloc[indices].copy()
        boot_decomp = compute_decomposition(boot_data)
        
        bootstrap_results['brier'].append(boot_decomp['brier'])
        bootstrap_results['reliability'].append(boot_decomp['reliability'])
        bootstrap_results['resolution'].append(boot_decomp['resolution'])
    
    # Compute confidence intervals
    results['brier_ci'] = np.percentile(bootstrap_results['brier'], [2.5, 97.5])
    results['reliability_ci'] = np.percentile(bootstrap_results['reliability'], [2.5, 97.5])
    results['resolution_ci'] = np.percentile(bootstrap_results['resolution'], [2.5, 97.5])

    # 2. Expected Calibration Error (ECE)
    # Bin predictions
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (data['prob'] > bin_lower) & (data['prob'] <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            bin_acc = data.loc[in_bin, 'correct'].mean()
            bin_conf = data.loc[in_bin, 'prob'].mean()
            bin_count = in_bin.sum()
            
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)
            
            ece += prop_in_bin * abs(bin_acc - bin_conf)
    
    # ECE confidence interval using bootstrap
    n_bootstrap = 1000
    ece_bootstrap = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_data = data.iloc[indices]
        
        boot_ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (boot_data['prob'] > bin_lower) & (boot_data['prob'] <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                bin_acc = boot_data.loc[in_bin, 'correct'].mean()
                bin_conf = boot_data.loc[in_bin, 'prob'].mean()
                boot_ece += prop_in_bin * abs(bin_acc - bin_conf)
        
        ece_bootstrap.append(boot_ece)
    
    ece_ci = np.percentile(ece_bootstrap, [2.5, 97.5])
    results['ece'] = ece
    results['ece_ci'] = ece_ci

    return results

from scipy import stats
from pandas.api.types import is_categorical_dtype, is_object_dtype

def compare_partial_correlations(predictor_series, outcome1_series, outcome2_series, 
                                 control_series_list=None, n_bootstrap=1000, eps=1e-12):
    """
    Compare two partial correlations that share the same predictor using Steiger's test.
    """
    if control_series_list is None:
        control_series_list = []

    # Combine all data and drop NaNs
    all_data = pd.DataFrame({
        'predictor': predictor_series,
        'outcome1': outcome1_series,
        'outcome2': outcome2_series
    })

    for i, control in enumerate(control_series_list):
        col_name = control.name if (hasattr(control, 'name') and control.name) else f'control_{i}'
        all_data[col_name] = control

    all_data = all_data.dropna()
    n = len(all_data)
    if n < 5:
        raise ValueError("Not enough complete cases after dropping NaNs.")

    # Identify control columns
    control_cols = [c for c in all_data.columns if c not in ['predictor', 'outcome1', 'outcome2']]

    # Build a single control matrix once (avoid re-doing it for each variable)
    def build_X_controls(df, cols):
        X_parts = []
        effective_cols = []
        for col in cols:
            s = df[col]
            if is_categorical_dtype(s) or is_object_dtype(s):
                if s.nunique() < 2:
                    # Drop single-level categorical controls
                    continue
                dummies = pd.get_dummies(s, prefix=col, drop_first=True).astype(float)
                if dummies.shape[1] == 0:
                    continue
                X_parts.append(dummies)
                effective_cols.append(col)
            else:
                # Numeric: drop if near zero variance
                if np.nanstd(s.values.astype(float)) < eps:
                    continue
                # Standardization not required, but harmless; residualization is invariant to scaling
                x = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0)
                X_parts.append(pd.DataFrame({col: x.values}, index=df.index))
                effective_cols.append(col)

        if X_parts:
            X_controls = pd.concat(X_parts, axis=1)
        else:
            X_controls = pd.DataFrame(index=df.index)

        # Add intercept
        X_controls.insert(0, 'const', 1.0)
        return X_controls, effective_cols

    X_controls, effective_cols = build_X_controls(all_data, control_cols)
    X = X_controls.values.astype(float)

    # Residualize a variable on controls
    def residualize(y):
        y = y.values.astype(float)
        # OLS via lstsq
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ coef

    # Compute residuals
    predictor_resid = residualize(all_data['predictor'])
    outcome1_resid = residualize(all_data['outcome1'])
    outcome2_resid = residualize(all_data['outcome2'])

    # Partial correlations (correlations among residuals)
    r_p1 = np.corrcoef(predictor_resid, outcome1_resid)[0, 1]  # predictor–outcome1
    r_p2 = np.corrcoef(predictor_resid, outcome2_resid)[0, 1]  # predictor–outcome2
    r_12 = np.corrcoef(outcome1_resid, outcome2_resid)[0, 1]   # outcome1–outcome2

    # Steiger's test (1980): dependent correlations with one variable in common
    # Use raw r difference, t distribution with df = n - 3
    r_det = 1 - r_p1**2 - r_p2**2 - r_12**2 + 2*r_p1*r_p2*r_12  # det of 3x3 correlation matrix
    # Numerical guard
    if r_det < 0 and r_det > -1e-12:
        r_det = 0.0

    if r_det <= 0:
        t_stat = np.nan
        p_value = np.nan
    else:
        df = n - 3
        num = (r_p1 - r_p2)
        den = np.sqrt(2 * r_det / ( (n - 3) * (1 + r_12) ))
        # Avoid divide-by-zero
        if den == 0:
            t_stat = np.nan
            p_value = np.nan
        else:
            t_stat = num / den
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))

    # Bootstrap CI for the difference in partial correlations
    diff_bootstrap = []
    rng = np.random.default_rng()
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n, endpoint=False)
        boot = all_data.iloc[idx].reset_index(drop=True)

        # Rebuild controls for the bootstrap sample (skip if degenerate)
        Xc, eff = build_X_controls(boot, control_cols)
        Xb = Xc.values.astype(float)

        if Xb.shape[1] == 0:  # only intercept
            # Still valid; proceed
            pass

        def resid_b(y):
            y = y.values.astype(float)
            coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
            return y - Xb @ coef

        try:
            br1 = resid_b(boot['predictor'])
            br2 = resid_b(boot['outcome1'])
            br3 = resid_b(boot['outcome2'])
            rr_p1 = np.corrcoef(br1, br2)[0, 1]
            rr_p2 = np.corrcoef(br1, br3)[0, 1]
            if np.isfinite(rr_p1) and np.isfinite(rr_p2):
                diff_bootstrap.append(rr_p1 - rr_p2)
        except Exception:
            continue

    if len(diff_bootstrap) >= max(10, int(0.1 * n_bootstrap)):
        diff_ci = np.percentile(diff_bootstrap, [2.5, 97.5])
    else:
        diff_ci = np.array([np.nan, np.nan])

    diff = r_p1 - r_p2
    if np.isfinite(diff):
        if abs(diff) < 1e-12:
            interpretation = 'tie'
        else:
            interpretation = 'outcome1 better' if diff > 0 else 'outcome2 better'
    else:
        interpretation = 'undetermined'

    return {
        'partial_corr_outcome1': r_p1,
        'partial_corr_outcome2': r_p2,
        'correlation_between_outcomes': r_12,
        'difference': diff,
        'difference_ci': diff_ci,
        'steiger_z': t_stat,
        'p_value': p_value,
        'df': n - 3,
        'n_samples': n,
        'controls_used': effective_cols,
        'interpretation': interpretation,
        'test': 'Steiger (1980), one variable in common'
    }

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from statsmodels.regression.linear_model import OLS

def compare_surface_contamination(
    outcome1_series,
    outcome2_series,
    control_series_list,
    outcome1_name='outcome1',
    outcome2_name='outcome2',
    n_bootstrap=1000,
    random_state=None,
    eps=1e-12
):
    """
    Compare how well a shared set of surface features predicts two outcomes
    by comparing in-sample R^2 (and adjusted R^2), with bootstrap CIs.
    Returns a dict with R^2 for each outcome, their difference, and CIs.
    """

    # Assemble full data
    all_data = pd.DataFrame({
        'outcome1': outcome1_series,
        'outcome2': outcome2_series
    })
    for i, s in enumerate(control_series_list):
        col_name = getattr(s, 'name', None) or f'feature_{i}'
        all_data[col_name] = s

    # Clean and reset index to keep design aligned with outcomes
    all_data = all_data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    n = len(all_data)
    if n < 5:
        raise ValueError("Not enough complete cases after cleaning.")

    feature_cols = [c for c in all_data.columns if c not in ['outcome1', 'outcome2']]

    # Infer feature types
    num_cols, cat_cols = [], []
    for c in feature_cols:
        s = all_data[c]
        if is_numeric_dtype(s):
            if np.nanstd(s.values.astype(float)) > eps:  # drop zero-variance numerics
                num_cols.append(c)
        else:
            cat_cols.append(c)

    # Record categorical levels (drop single-level)
    cat_levels = {}
    for c in cat_cols:
        sc = all_data[c].astype('category')
        if sc.cat.categories.size > 1:
            cat_levels[c] = list(sc.cat.categories)

    # Build design matrix with fixed columns based on full data
    def build_X(df):
        parts = []

        # Numeric (standardize within df; keep zeros if sd ~ 0)
        for c in num_cols:
            x = df[c].astype(float).values
            sd = x.std(ddof=0)
            if sd <= eps:
                parts.append(pd.DataFrame({c: np.zeros(len(df))}, index=df.index))
            else:
                xz = (x - x.mean()) / sd
                parts.append(pd.DataFrame({c: xz}, index=df.index))

        # Categorical (fixed levels; drop_first=True). Ensure index alignment.
        for c, levels in cat_levels.items():
            sc = pd.Categorical(df[c], categories=levels)
            sser = pd.Series(sc, index=df.index)
            dummies = pd.get_dummies(sser, prefix=c, drop_first=True).astype(float)
            dummies.index = df.index  # ensure same index
            parts.append(dummies)

        X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
        X.insert(0, 'const', 1.0)  # intercept first
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X

    # Full-sample design
    X_full = build_X(all_data)
    if X_full.shape[0] != n:
        raise ValueError(f"Design row count {X_full.shape[0]} != n {n}")
    if not np.isfinite(X_full.values).all():
        raise ValueError("Design matrix contains non-finite values after cleaning.")
    X_cols = list(X_full.columns)
    p = X_full.shape[1] - 1  # predictors excluding intercept

    # Fit statsmodels OLS
    y1 = all_data['outcome1'].astype(float).values
    y2 = all_data['outcome2'].astype(float).values

    model1 = OLS(y1, X_full.values).fit()
    model2 = OLS(y2, X_full.values).fit()

    r2_out1 = model1.rsquared
    r2_out2 = model2.rsquared
    r2_diff = r2_out1 - r2_out2

    adj_r2_out1 = model1.rsquared_adj
    adj_r2_out2 = model2.rsquared_adj
    adj_r2_diff = adj_r2_out1 - adj_r2_out2

    # Bootstrap (no stratification; sample-until-success)
    rng = np.random.default_rng(random_state)
    r2_out1_bootstrap, r2_out2_bootstrap, r2_diffs_bootstrap = [], [], []

    def r2_from_lstsq(X_arr, y, eps=1e-12):
        y = y.astype(float)
        ybar = y.mean()
        sst = np.sum((y - ybar) ** 2)
        if sst <= eps:
            return np.nan
        coef, _, _, _ = np.linalg.lstsq(X_arr, y, rcond=None)
        yhat = X_arr @ coef
        sse = np.sum((y - yhat) ** 2)
        return 1.0 - sse / sst

    target_success = n_bootstrap
    max_attempts = max(5 * n_bootstrap, n_bootstrap + 5000)
    attempts = 0
    successes = 0

    while successes < target_success and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        boot = all_data.iloc[idx].reset_index(drop=True)

        Xb = build_X(boot).reindex(columns=X_cols, fill_value=0.0).values
        y1b = boot['outcome1'].astype(float).values
        y2b = boot['outcome2'].astype(float).values

        r2_1 = r2_from_lstsq(Xb, y1b)
        r2_2 = r2_from_lstsq(Xb, y2b)

        if np.isfinite(r2_1) and np.isfinite(r2_2):
            r2_out1_bootstrap.append(r2_1)
            r2_out2_bootstrap.append(r2_2)
            r2_diffs_bootstrap.append(r2_1 - r2_2)
            successes += 1

    warning = None
    if successes < target_success:
        warning = f"Warning: only {successes}/{target_success} valid bootstrap reps after {attempts} attempts."

    def pct_ci(a):
        a = np.asarray(a)
        if a.size == 0:
            return np.array([np.nan, np.nan])
        return np.percentile(a, [2.5, 97.5])

    r2_out1_ci = pct_ci(r2_out1_bootstrap)
    r2_out2_ci = pct_ci(r2_out2_bootstrap)
    r2_diff_ci = pct_ci(r2_diffs_bootstrap)

    diff_significant = (
        np.isfinite(r2_diff_ci).all() and not (r2_diff_ci[0] <= 0 <= r2_diff_ci[1])
    )

    if r2_out1 > r2_out2:
        more_contaminated, less_contaminated = outcome1_name, outcome2_name
    elif r2_out2 > r2_out1:
        more_contaminated, less_contaminated = outcome2_name, outcome1_name
    else:
        more_contaminated, less_contaminated = 'tie', 'tie'

    high_p = (p >= n - 1)
    note = "High model complexity: predictors >= (n - 1). In-sample R^2 may be inflated." if high_p else None

    return {
        f'r2_{outcome1_name}': r2_out1,
        f'r2_{outcome2_name}': r2_out2,
        f'r2_{outcome1_name}_ci': r2_out1_ci,
        f'r2_{outcome2_name}_ci': r2_out2_ci,
        f'adj_r2_{outcome1_name}': adj_r2_out1,
        f'adj_r2_{outcome2_name}': adj_r2_out2,
        'r2_difference': r2_diff,
        'r2_difference_ci': r2_diff_ci,
        'adj_r2_difference': adj_r2_diff,
        'difference_significant': diff_significant,
        'more_contaminated': more_contaminated,
        'less_contaminated': less_contaminated,
        'n_samples': n,
        'n_predictors': p,
        'n_bootstrap_success': successes,
        'warning': warning,
        'note': note,
        f'full_results_{outcome1_name}': model1,
        f'full_results_{outcome2_name}': model2,
        'design_columns': X_cols,
    }

import pandas as pd
from scipy.stats import binomtest

def summarize_wrong_way(results_list, alpha=0.05):
    """
    results_list: list of 'res' DataFrames returned by analyze_wrong_way for this question set.
    alpha: the same per-predictor one-sided threshold you used in analyze_wrong_way.
    """
    df = pd.concat(results_list, ignore_index=True)
    tested = df.loc[df['baseline_positive'] & df['p_one_sided_delegate_gt0'].notna()]
    n = int(len(tested))
    k = int(tested['misuse'].sum())
    if n == 0:
        return {'n_candidates': 0, 'n_wrong_way': 0, 'expected_by_chance': 0.0, 'p_value': float('nan'),
                'conclusion': 'insufficient data'}
    pval = binomtest(k, n, p=alpha, alternative='greater').pvalue
    conclusion = ('Models used wrong-way predictors more often than chance'
                  if pval < 0.05 else
                  'Models did not use wrong-way predictors more often than chance')
    return {'n_candidates': n,
            'n_wrong_way': k,
            'expected_by_chance': alpha * n,
            'p_value': pval,
            'conclusion': conclusion}

import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest

def summarize_wrong_wayB(results, alpha=0.05,
                        col_baseline_pos='baseline_positive',
                        col_beta_correct='beta_correct',
                        col_p_one='p_one_sided_delegate_gt0',
                        col_z_delegate='z_delegate'):
    """
    Option B:
    - Candidates (m): predictors with one-sided p for β_delegate>0 below alpha.
    - Wrong-way count (k): among candidates, how many have baseline-positive (β_correct>0).
    - Expected by chance: q_base * m, where q_base is the baseline-positive rate among all valid predictors.
    - Test: X ~ Binomial(m, q_base), alternative='greater'.

    Returns dict:
      n_candidates (m), n_wrong_way (k), expected_by_chance, p_value, alpha, conclusion.
    """
    # Accept list of DataFrames
    if isinstance(results, list):
        df = pd.concat(results, ignore_index=True)
    else:
        df = results.copy()

    # Ensure baseline_positive exists (fallback from beta_correct > 0)
    if col_baseline_pos not in df.columns:
        if col_beta_correct not in df.columns:
            raise ValueError(f"Need either '{col_baseline_pos}' or '{col_beta_correct}'")
        beta = pd.to_numeric(df[col_beta_correct], errors='coerce')
        df[col_baseline_pos] = beta > 0

    # Ensure one-sided p for delegation > 0 exists (fallback from z_delegate)
    if col_p_one not in df.columns:
        if col_z_delegate not in df.columns:
            raise ValueError(f"Need either '{col_p_one}' or '{col_z_delegate}'")
        z = pd.to_numeric(df[col_z_delegate], errors='coerce')
        df[col_p_one] = 1 - norm.cdf(z)

    bp = df[col_baseline_pos]
    p_one = pd.to_numeric(df[col_p_one], errors='coerce')

    # Valid rows must have both baseline sign and a p-value
    valid = bp.notna() & p_one.notna()
    if not valid.any():
        return {
            'n_candidates': 0,
            'n_wrong_way': 0,
            'expected_by_chance': 0.0,
            'p_value': np.nan,
            'alpha': alpha,
            'conclusion': 'no valid predictors'
        }

    # Base rate of baseline-positive among all valid predictors
    q_base = float((bp[valid] > 0).mean())

    # Gate on significant-positive delegation (candidates)
    sig_pos = valid & (p_one < alpha)
    m = int(sig_pos.sum())
    if m == 0:
        return {
            'n_candidates': 0,
            'n_wrong_way': 0,
            'expected_by_chance': 0.0,
            'p_value': np.nan,
            'alpha': alpha,
            'conclusion': 'no significant-positive delegation effects'
        }

    # Count baseline-positive within the candidates
    k = int(((bp > 0) & sig_pos).sum())
    expected = q_base * m

    # Binomial test: is k larger than expected under base rate q_base?
    pval = binomtest(k, m, p=q_base, alternative='greater').pvalue

    conclusion = 'more than expected by chance' if pval < 0.05 else 'not more than expected by chance'

    return {
        'n_candidates': m,
        'n_wrong_way': k,
        'expected_by_chance': expected,
        'p_value': pval,
        'alpha': alpha,
        'conclusion': conclusion
    }

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

def summarize_wrong_wayC(misused_results, alpha=0.05):
    # Concatenate input (handles list of DataFrames or a single DataFrame)
    if isinstance(misused_results, list):
        all_results = pd.concat(misused_results, keys=range(len(misused_results)), names=['model_idx', 'row_idx'])
    else:
        all_results = misused_results.copy()

    # Family = baseline-positive with valid one-sided p for β_delegate>0
    valid_p = all_results['p_one_sided_delegate_gt0'].notna()
    fam_mask = all_results['baseline_positive'] & valid_p
    potential_misuses = all_results.loc[fam_mask].copy()

    if len(potential_misuses) > 0:
        p_values = potential_misuses['p_one_sided_delegate_gt0'].values
        rejected, adjusted_p, _, _ = multipletests(p_values, method='fdr_bh', alpha=alpha)

        # Add adjusted p-values and FDR decision
        potential_misuses['p_adjusted'] = adjusted_p
        potential_misuses['misuse_fdr'] = rejected

        # Merge back (local) so downstream code that expects these columns in all_results can use them if needed
        all_results = all_results.merge(
            potential_misuses[['p_adjusted', 'misuse_fdr']],
            left_index=True,
            right_index=True,
            how='left'
        )
        all_results['misuse_fdr'] = all_results['misuse_fdr'].fillna(False)

    return potential_misuses

def compute_optimal_accuracy_with_introspection(model_correctness, model_confidence, delegate_choice, teammate_accuracy):
    """
    Compute accuracy if model uses optimal policy based on its confidence signal
    """
    # Optimal policy: answer when confidence > teammate_accuracy

    def bootstrap_ci(stat_fn, *arrays, B=10000, alpha=0.05, random_state=None, **kwargs):
        """
        Percentile bootstrap CI for any scalar statistic.

        stat_fn(*arrays, **kwargs) -> float
        - Should compute the metric from per-item arrays (same length).
        - Scalars (like teammate_accuracy) should be passed via **kwargs.

        Returns (ci_lo, ci_hi).
        """
        arrays = [np.asarray(a) for a in arrays]
        N = len(arrays[0])
        for a in arrays[1:]:
            if len(a) != N:
                raise ValueError("All arrays must have the same length")
        rng = np.random.default_rng(random_state)

        stats = np.empty(B, dtype=float)
        for b in range(B):
            idx = rng.integers(0, N, size=N)
            resampled = [a[idx] for a in arrays]
            stats[b] = float(stat_fn(*resampled, **kwargs))

        lo = float(np.quantile(stats, alpha/2))
        hi = float(np.quantile(stats, 1 - alpha/2))
        return lo, hi

    def wilson_ci(k, n, alpha=0.05):
        """
        Wilson score interval for a binomial proportion (k successes out of n).

        Returns (ci_lo, ci_hi). If n == 0, returns (np.nan, np.nan).
        """
        if n == 0:
            return (np.nan, np.nan)
        z = 1.959963984540054 if alpha == 0.05 else float(np.abs(np.sqrt(2)*np.erfinv(1 - alpha)))
        phat = k / n
        denom = 1 + z*z/n
        center = (phat + z*z/(2*n)) / denom
        half = z * np.sqrt(phat*(1 - phat)/n + z*z/(4*n*n)) / denom
        return (float(center - half), float(center + half))


    answer_mask = model_confidence > teammate_accuracy
    delegate_mask = ~answer_mask
    
    # Compute resulting accuracy
    n_answer = answer_mask.sum()
    n_delegate = delegate_mask.sum()
    
    if n_answer > 0:
        acc_when_answer = model_correctness[answer_mask].mean()
    else:
        acc_when_answer = 0
        
    # Overall accuracy
    total_accuracy = (n_answer * acc_when_answer + 
                     n_delegate * teammate_accuracy) / len(model_correctness)
    
    # Also compute what they actually chose
    answer_rate = n_answer / len(model_correctness)

    opt_delegate = (model_confidence < teammate_accuracy)          # optimal decision: True=delegate
    decide_delegate = delegate_choice.astype(bool)
    disagree = opt_delegate ^ decide_delegate                      # XOR: True where decisions differ
    agreement_rate = 1.0 - disagree.mean()
    agree_ci_lo, agree_ci_hi = wilson_ci(int((~disagree).sum()), len(disagree), alpha=0.05)

    severity_aware = np.mean(np.abs(model_confidence - teammate_accuracy) * disagree)    

    weights = np.abs(model_confidence - teammate_accuracy)
    # Normalized version: 1.0 = perfect, 0.0 = always take the opposite of the optimal decision
    denom = weights.sum()
    norm_weighted_agreement = 1.0 if denom == 0 else 1.0 - (weights[disagree].sum() / denom)

    s = model_confidence
    c = teammate_accuracy
    dec = delegate_choice.astype(bool)         # True=delegate, False=answer
    w = np.abs(s - c)
    opt_delegate = (s < c)
    opt_answer  = (s >= c)                    
    # Mistake types
    underconf = opt_answer & dec               # should answer, but delegated
    overconf  = opt_delegate & (~dec)          # should delegate, but answered
    # Rates (fraction of all items)
    under_rate = underconf.mean()
    over_rate  = overconf.mean()
    # Severity contributions (avg expected accuracy lost per item)
    under_severity = np.mean(w * underconf)    # points left on the table from underconfidence
    over_severity  = np.mean(w * overconf)

    # Normalized by total margin mass (0..1): share of |s-c| mass lost to each type
    W = w.sum()
    under_norm = 0.0 if W == 0 else w[underconf].sum() / W
    over_norm  = 0.0 if W == 0 else w[overconf].sum()  / W
    OWB = (over_norm - under_norm) / (over_norm + under_norm) if (over_norm + under_norm) > 0 else 0.0

    # OWB CI (bootstrap)
    def owb_stat(y, s, dec, c):
        m = s - c
        w = np.abs(m)
        over = (s < c) & (~dec.astype(bool))
        under = (s >= c) & (dec.astype(bool))
        Wb = w.sum()
        if Wb == 0:
            return 0.0
        over_norm_b = w[over].sum() / Wb
        under_norm_b = w[under].sum() / Wb
        den_b = over_norm_b + under_norm_b
        return 0.0 if den_b == 0 else (over_norm_b - under_norm_b) / den_b

    ci_owb = bootstrap_ci(owb_stat, model_correctness, model_confidence, delegate_choice,
                          B=10000, alpha=0.05, random_state=0, c=teammate_accuracy)

    # OBB and CI (bootstrap)
    N_over = int(overconf.sum())
    N_under = int(underconf.sum())
    obb_den = N_over + N_under
    OBB = 0.0 if obb_den == 0 else (N_over - N_under) / obb_den

    def obb_stat(y, s, dec, c):
        over_b = (s < c) & (~dec.astype(bool))
        under_b = (s >= c) & (dec.astype(bool))
        N_over_b = int(over_b.sum())
        N_under_b = int(under_b.sum())
        den_b = N_over_b + N_under_b
        return 0.0 if den_b == 0 else (N_over_b - N_under_b) / den_b

    ci_obb = bootstrap_ci(obb_stat, model_correctness, model_confidence, delegate_choice,
                          B=10000, alpha=0.05, random_state=0, c=teammate_accuracy)

    return {
        'optimal_accuracy': total_accuracy,
        'optimal_answer_rate': answer_rate,
        'accuracy_on_answered': acc_when_answer,
        'agreement_rate': agreement_rate,
        'agreement_rate_ci': (agree_ci_lo, agree_ci_hi),
        'weighted_agreement_rate': 1-severity_aware,
        'norm_weighted_agreement_rate': norm_weighted_agreement,
        'underconf_rate': under_rate,
        'overconf_rate': over_rate,
        'weighted_underconf_rate': 1-under_severity,
        'weighted_overconf_rate': 1-over_severity,
        'unweighted_confidence': OBB,
        'unweighted_confidence_ci': ci_obb,
        'weighted_confidence': OWB,
        'weighted_confidence_ci': ci_owb,
    }

def block_partial_controls_given_entropy(
    dv_series, entropy_series, control_series_list, B=2000, alpha=0.05, random_state=None
):
    """
    How much the surface cues (in aggregate) explain the decision after controlling for entropy
    (or a proxy like correctness).

    Returns:
      {
        'partial_R2_controls_given_entropy': ...,
        'R_controls_given_entropy': ...,
        'delta_R2': ...,
        'R2_reduced': ...,
        'R2_full': ...,
        'F': ...,
        'p_value': ...,
        'n_samples': ...,
        'df1': ...,
        'df2': ...,
        'partial_R2_CI': (lo, hi) or None,          # stratified bootstrap percentile CI
        'R_CI': (lo, hi) or None                    # stratified bootstrap percentile CI
      }
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from numpy.linalg import lstsq
    from scipy import stats

    eps = 1e-12

    # Assemble and clean
    all_data = pd.DataFrame({'dv': dv_series, 'ctrl': entropy_series})
    for i, s in enumerate(control_series_list or []):
        name = getattr(s, 'name', None) or f'control_{i}'
        all_data[name] = s
    all_data = all_data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    n = len(all_data)
    if n < 5:
        raise ValueError(f"Not enough samples after removing NaNs: {n}")

    # Build surface-cue matrix (dummy encode categoricals, standardize continuous)
    control_cols = [c for c in all_data.columns if c not in ['dv', 'ctrl']]
    Xc_parts = []
    min_count = max(3, int(0.01 * n))  # drop ultra-rare dummy levels
    for col in control_cols:
        s = all_data[col]
        if s.dtype.kind in ('O', 'U') or str(s.dtype) == 'category':
            dummies = pd.get_dummies(s.astype('category'), prefix=col, drop_first=True, dtype=float)
            # Drop rare levels that cause high leverage/instability
            keep = [c for c in dummies.columns if dummies[c].sum() >= min_count and (n - dummies[c].sum()) >= min_count]
            if keep:
                Xc_parts.append(dummies[keep])
        else:
            x = all_data[[col]].astype(float).values
            if float(np.std(x, ddof=0)) <= eps:
                continue
            standardized = pd.DataFrame(
                StandardScaler().fit_transform(x),
                columns=[col],
                index=all_data.index
            )
            Xc_parts.append(standardized)
    Xc = pd.concat(Xc_parts, axis=1).astype(float) if Xc_parts else pd.DataFrame(index=all_data.index)

    # Design matrices
    y = all_data['dv'].astype(float).values
    iv = all_data['ctrl'].astype(float).values.reshape(-1, 1)
    ones = np.ones((n, 1), dtype=float)
    X_reduced = np.concatenate([ones, iv], axis=1)  # intercept + control
    if Xc.shape[1] == 0:
        # No cues: return reduced-model stats
        beta_red = lstsq(X_reduced, y, rcond=None)[0]
        yhat_red = X_reduced @ beta_red
        tss = max(np.sum((y - y.mean())**2), eps)
        R2_reduced = 1.0 - np.sum((y - yhat_red)**2) / tss
        return {
            'partial_R2_controls_given_entropy': 0.0,
            'R_controls_given_entropy': 0.0,
            'delta_R2': 0.0,
            'R2_reduced': float(R2_reduced),
            'R2_full': float(R2_reduced),
            'F': np.nan,
            'p_value': np.nan,
            'n_samples': int(n),
            'df1': 0,
            'df2': max(n - X_reduced.shape[1], 0),
            'partial_R2_CI': None,
            'R_CI': None
        }

    X_full = np.concatenate([X_reduced, Xc.values.astype(float)], axis=1)

    # OLS fits for point estimates (unchanged semantics)
    beta_red = lstsq(X_reduced, y, rcond=None)[0]
    yhat_red = X_reduced @ beta_red
    rss_red = np.sum((y - yhat_red)**2)

    beta_full = lstsq(X_full, y, rcond=None)[0]
    yhat_full = X_full @ beta_full
    rss_full = np.sum((y - yhat_full)**2)

    tss = max(np.sum((y - y.mean())**2), eps)
    R2_reduced = 1.0 - rss_red / tss
    R2_full = 1.0 - rss_full / tss

    # Effective dfs by rank (more stable under collinearity/rare dummies)
    rank_red = np.linalg.matrix_rank(X_reduced)
    rank_full = np.linalg.matrix_rank(X_full)
    df1 = int(max(rank_full - rank_red, 0))
    df2 = int(max(n - rank_full, 0))

    delta_R2 = max(R2_full - R2_reduced, 0.0)
    denom = max(1.0 - R2_reduced, eps)
    partial_R2 = delta_R2 / denom
    partial_R2 = float(np.clip(partial_R2, 0.0, 1.0))
    R_block = float(np.sqrt(partial_R2))

    # Heteroskedasticity-robust (HC3) block test for cues (replaces naive F)
    q = Xc.shape[1]  # number of added columns (before rank)
    if df1 > 0 and df2 > 0:
        X = X_full
        b = beta_full
        e = y - yhat_full
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-12)
        # Hat diag: h_i = x_i^T (X'X)^{-1} x_i
        M = X @ XtX_inv
        h = np.sum(M * X, axis=1)
        adj = (e / np.clip(1.0 - h, 1e-8, None))**2
        meat = X.T @ (adj[:, None] * X)
        V = XtX_inv @ meat @ XtX_inv  # HC3 covariance
        # Restrictions: last block (cues) equal zero
        k_full = X.shape[1]
        Rm = np.zeros((k_full, k_full), dtype=float)
        Rm[-q:, -q:] = np.eye(q, dtype=float)
        R = Rm[-q:, :]  # shape (q, k)
        Rb = R @ b
        RVRT = R @ V @ R.T
        # Use effective q from rank in case of collinearity
        q_eff = int(np.linalg.matrix_rank(RVRT))
        try:
            RVRT_inv = np.linalg.pinv(RVRT, rcond=1e-12)
            Wald = float(Rb.T @ RVRT_inv @ Rb)
            F_stat = Wald / max(q_eff, 1)
            p_val = 1.0 - stats.f.cdf(F_stat, max(q_eff, 1), max(df2, 1))
        except Exception:
            F_stat, p_val = np.nan, np.nan
    else:
        F_stat, p_val = np.nan, np.nan

    # Stratified bootstrap for CIs (percentile), preserving class and control mix
    partial_R2_CI = None
    R_CI = None
    if B and B > 0:
        rng = np.random.default_rng(random_state)

        # Build strata: by DV (if binary) and by control bins (adaptive)
        dv_vals = all_data['dv'].values
        is_binary_dv = np.all(np.isin(np.unique(dv_vals), [0, 1])) and len(np.unique(dv_vals)) <= 2

        ctrl_vals = all_data['ctrl'].values
        # Try 5,4,3,2 bins for control; reduce if any stratum too small
        def make_bins(qs):
            try:
                bins = pd.qcut(ctrl_vals, qs, duplicates='drop')
                return bins
            except Exception:
                return None

        ctrl_bins = None
        for qbins in (5, 4, 3, 2):
            ctrl_bins = make_bins(qbins)
            if ctrl_bins is None:
                continue
            # combine with dv to check stratum sizes
            if is_binary_dv:
                strata = pd.Series(list(zip(dv_vals.astype(int), pd.Categorical(ctrl_bins).codes)))
            else:
                strata = pd.Series(pd.Categorical(ctrl_bins).codes)
            counts = strata.value_counts()
            if (counts.min() >= 2) and (counts.size >= 2):
                break
            else:
                ctrl_bins = None

        # Fallbacks if binning fails
        if ctrl_bins is None:
            if is_binary_dv:
                strata = pd.Series(dv_vals.astype(int))
            else:
                strata = pd.Series(np.zeros(n, dtype=int))
        else:
            if is_binary_dv:
                strata = pd.Series(list(zip(dv_vals.astype(int), pd.Categorical(ctrl_bins).codes)))
            else:
                strata = pd.Series(pd.Categorical(ctrl_bins).codes)

        codes, uniques = pd.factorize(strata, sort=True)
        indices_by_group = {}
        for i, code in enumerate(codes):
            if code == -1:
                continue
            indices_by_group.setdefault(code, []).append(i)
            nonempty = [np.array(idx, dtype=int) for idx in indices_by_group.values() if len(idx) > 0]
            if len(nonempty) <= 1:
                # degenerate: fall back to ordinary bootstrap
                nonempty = [np.arange(n, dtype=int)]

        r2_vals = []
        r_vals = []

        Xc_mat = Xc.values.astype(float)

        for _ in range(int(B)):
            # Stratified resample: sample within each stratum, preserve counts
            idx_parts = []
            for g in nonempty:
                m = len(g)
                idx_g = rng.integers(0, m, size=m)
                idx_parts.append(np.array(g, dtype=int)[idx_g])
            idx = np.concatenate(idx_parts, axis=0)

            y_b = y[idx]
            iv_b = iv[idx]
            Xc_b = Xc_mat[idx]

            Xr_b = np.concatenate([np.ones((len(idx), 1)), iv_b], axis=1)
            Xf_b = np.concatenate([Xr_b, Xc_b], axis=1)

            # Reduced
            try:
                br = lstsq(Xr_b, y_b, rcond=None)[0]
                yr = Xr_b @ br
                rr = np.sum((y_b - yr)**2)
            except Exception:
                continue

            # Full
            try:
                bf = lstsq(Xf_b, y_b, rcond=None)[0]
                yf = Xf_b @ bf
                rf = np.sum((y_b - yf)**2)
            except Exception:
                continue

            tt = np.sum((y_b - y_b.mean())**2)
            if tt <= eps:
                continue

            R2r = 1.0 - rr / tt
            R2f = 1.0 - rf / tt
            dR2 = max(R2f - R2r, 0.0)
            denom_b = max(1.0 - R2r, eps)
            pr2_b = dR2 / denom_b
            if np.isfinite(pr2_b):
                r2_vals.append(pr2_b)
                r_vals.append(np.sqrt(max(pr2_b, 0.0)))

        if len(r_vals) > 0:
            q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
            r2_arr = np.array(r2_vals, dtype=float)
            r_arr = np.array(r_vals, dtype=float)
            lo_r2 = float(np.quantile(r2_arr, q_lo))
            hi_r2 = float(np.quantile(r2_arr, q_hi))
            lo_r = float(np.quantile(r_arr, q_lo))
            hi_r = float(np.quantile(r_arr, q_hi))
            partial_R2_CI = (max(0.0, lo_r2), min(1.0, hi_r2))
            R_CI = (max(0.0, lo_r), min(1.0, hi_r))

    # --- Bootstrap-median point estimate (ad-hoc or global) ---
    # Apply globally? (set to True to always use bootstrap median when available)
    APPLY_GLOBALLY = False

    bootstrap_median_point_estimate_used = False
    ci_containment_applied = False  # if you also kept the earlier "containment" patch

    if B and R_CI is not None and 'r_arr' in locals() and len(r_arr) > 0:
        # Decide whether to apply ad-hoc
        apply_ad_hoc = False
        try:
            dv_vals = all_data['dv'].values
            dv_unique = np.unique(dv_vals)
            is_binary_dv = (dv_unique.size <= 2) and np.all(np.isin(dv_unique, [0, 1]))
            imbalance = 1.0
            if is_binary_dv:
                pos_rate = float(np.mean(dv_vals))
                imbalance = min(pos_rate, 1.0 - pos_rate)  # minority fraction

            ctrl_vals = all_data['ctrl'].values
            ctrl_unique = np.unique(ctrl_vals)
            control_is_near_binary = (ctrl_unique.size <= 3) and np.all(np.isin(ctrl_unique, [0, 1]))

            # Trigger only in the problematic corner: extreme imbalance + (near-)binary control
            apply_ad_hoc = (imbalance <= 0.10) and control_is_near_binary
        except Exception:
            apply_ad_hoc = False

        if APPLY_GLOBALLY or apply_ad_hoc:
            # Replace point estimate with stratified-bootstrap median
            R_block = float(np.median(r_arr))
            partial_R2 = float(R_block ** 2)
            bootstrap_median_point_estimate_used = True

    return {
        'partial_R2_controls_given_entropy': float(partial_R2),
        'R_controls_given_entropy': float(R_block),
        'delta_R2': float(delta_R2),
        'R2_reduced': float(R2_reduced),
        'R2_full': float(R2_full),
        'F': float(F_stat) if F_stat == F_stat else np.nan,
        'p_value': float(p_val) if p_val == p_val else np.nan,
        'n_samples': int(n),
        'df1': int(df1),
        'df2': int(df2),
        'partial_R2_CI': partial_R2_CI,
        'R_CI': R_CI
    }

def variance_partition_entropy_cues(dv_series, entropy_series, control_series_list, B=2000, alpha=0.05, random_state=None):
    """
    Commonality analysis for two predictor sets:
      - entropy (single predictor)
      - surface cues (block)
    Returns unique, shared, and unexplained proportions on the R^2 scale, plus the component R^2s.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from numpy.linalg import lstsq

    # Assemble and align
    all_data = pd.DataFrame({'dv': dv_series, 'entropy': entropy_series})
    for i, control in enumerate(control_series_list):
        col_name = control.name if control.name else f'control_{i}'
        all_data[col_name] = control
    all_data = all_data.dropna()
    n = len(all_data)
    if n < 5:
        raise ValueError(f"Not enough samples after removing NaNs: {n}")

    # Build cue matrix (dummy encode categoricals, standardize continuous)
    control_cols = [c for c in all_data.columns if c not in ['dv', 'entropy']]
    Xc_list = []
    for col in control_cols:
        if all_data[col].dtype in ['object', 'category']:
            dummies = pd.get_dummies(all_data[col], prefix=col, drop_first=True).astype(float)
            Xc_list.append(dummies)
        else:
            scaler = StandardScaler()
            standardized = pd.DataFrame(
                scaler.fit_transform(all_data[[col]]),
                columns=[col],
                index=all_data.index
            )
            Xc_list.append(standardized)
    Xc = pd.concat(Xc_list, axis=1) if Xc_list else pd.DataFrame(index=all_data.index)

    y = all_data['dv'].values.astype(float)
    one = np.ones((n,1))
    E = all_data['entropy'].values.astype(float).reshape(-1,1)
    C = Xc.values.astype(float) if Xc.shape[1] else np.empty((n,0))

    def R2(X):
        b = lstsq(X, y, rcond=None)[0]
        yhat = X @ b
        rss = np.sum((y - yhat)**2)
        tss = np.sum((y - y.mean())**2)
        return 1.0 - rss/tss

    XE = np.concatenate([one, E], axis=1)
    XC = np.concatenate([one, C], axis=1) if C.shape[1] else one
    XEC = np.concatenate([one, E, C], axis=1) if C.shape[1] else XE

    R2_E = R2(XE)
    R2_C = R2(XC)
    R2_EC = R2(XEC)

    U_E = max(R2_EC - R2_C, 0.0)
    U_C = max(R2_EC - R2_E, 0.0)
    S = R2_E + R2_C - R2_EC  # may be negative (suppression)
    U_unexpl = 1.0 - R2_EC

    out = dict(
        n_samples=int(n),
        R2_entropy=float(R2_E),
        R2_cues=float(R2_C),
        R2_full=float(R2_EC),
        unique_entropy=float(U_E),
        unique_cues=float(U_C),
        shared=float(S),
        unexplained=float(U_unexpl),
    )

    # Optional bootstrap CIs
    if B is not None and B > 0:
        rng = np.random.default_rng(random_state)
        vals = np.empty((B, 4), dtype=float)  # U_E, U_C, S, U_unexpl
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            y_b = y[idx]
            E_b = E[idx]
            C_b = C[idx] if C.shape[1] else C
            one_b = np.ones((n,1))

            def R2_b(X):
                bb = lstsq(X, y_b, rcond=None)[0]
                yhat = X @ bb
                rss = np.sum((y_b - yhat)**2)
                tss = np.sum((y_b - y_b.mean())**2)
                return 1.0 - rss/tss

            XE_b = np.concatenate([one_b, E_b], axis=1)
            XC_b = np.concatenate([one_b, C_b], axis=1) if C_b.shape[1] else one_b
            XEC_b = np.concatenate([one_b, E_b, C_b], axis=1) if C_b.shape[1] else XE_b

            R2E = R2_b(XE_b)
            R2C = R2_b(XC_b)
            R2EC = R2_b(XEC_b)

            UE = max(R2EC - R2C, 0.0)
            UC = max(R2EC - R2E, 0.0)
            S_b = R2E + R2C - R2EC
            Uu = 1.0 - R2EC
            vals[b] = (UE, UC, S_b, Uu)

        qs = np.quantile(vals, [alpha/2, 1 - alpha/2], axis=0)
        out.update(dict(
            unique_entropy_CI=(float(qs[0,0]), float(qs[1,0])),
            unique_cues_CI=(float(qs[0,1]), float(qs[1,1])),
            shared_CI=(float(qs[0,2]), float(qs[1,2])),
            unexplained_CI=(float(qs[0,3]), float(qs[1,3])),
        ))

    return out

