import statsmodels.stats.proportion as smp
import scipy.stats as ss
import numpy as np
import re

import math
from scipy.stats import binomtest, rankdata, pointbiserialr, norm, pearsonr, chi2
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

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

def lift_mcc_stats(tp, fn, fp, tn, kept_correct, p0, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)

    # ---------- point estimates --------------------------------------------
    k         = len(kept_correct)                    # ★ kept items
    kept_acc  = kept_correct.mean() if k else np.nan # ★ acc from real data
    lift      = kept_acc - p0

    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc   = (tp*tn - fp*fn) / denom if denom else np.nan

    # ---------- p-values ----------------------------------------------------
    p_lift = binomtest(kept_correct.sum(), k, p0,  # ★ successes = real kept
                       alternative='two-sided').pvalue
    p_mcc  = mcnemar([[tn, fp],
                      [fn, tp]], exact=True).pvalue

    # ---------- bootstrap CIs ----------------------------------------------
    N        = tp + fn + fp + tn
    counts   = np.array([tp, fn, fp, tn], int)
    probs    = counts / N

    lifts, mccs = [], []
    kept_idx = np.arange(k)                         # ★ indices for kept vector

    for _ in range(n_boot):
        # ----- lift: resample ONLY kept correctness ----------------★
        b_k_acc = kept_correct[rng.choice(kept_idx, k, replace=True)].mean()
        lifts.append(b_k_acc - p0)

        # ----- MCC: multinomial resample of 4-cell table (unchanged)
        sample = rng.choice(4, size=N, replace=True, p=probs)
        btp, bfn, bfp, btn = np.bincount(sample, minlength=4)
        bden = math.sqrt((btp+bfp)*(btp+bfn)*(btn+bfp)*(btn+bfn))
        bmcc = (btp*btn - bfp*bfn) / bden if bden else 0.0
        mccs.append(bmcc)

    ci_lift = tuple(np.percentile(lifts, [2.5, 97.5]))  # ★ tuple()
    ci_mcc  = tuple(np.percentile(mccs,  [2.5, 97.5]))

    boot_arr = np.array(mccs)
    p_mcc = 2 * min((boot_arr <= 0).mean(), (boot_arr >= 0).mean())
    
    return dict(lift=lift,   lift_ci=ci_lift,  p_lift=p_lift,
                mcc=mcc,     mcc_ci =ci_mcc,  p_mcc =p_mcc)

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
    
    # Get log-likelihoods (fixing the calculation)
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
