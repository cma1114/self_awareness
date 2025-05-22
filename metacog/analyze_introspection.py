import json
import argparse
import numpy as np
import os
import sys # For printing to stderr
import re # For parsing the log file
from sklearn.metrics import roc_auc_score, roc_curve # For AUROC
from scipy.stats import norm # For SDT calculations

# --- Helper Functions ---

def get_s_i(question_trial_data, p2_question_details_map):
    q_id = question_trial_data.get("question_id")
    if q_id in p2_question_details_map:
        return 1 if p2_question_details_map[q_id]['is_correct'] else 0
    return None

def calculate_calibration_metrics(true_outcomes, predicted_probs, n_bins=10, min_bin_samples_for_ece_mce=5):
    true_outcomes_np = np.array(true_outcomes)
    predicted_probs_np = np.array(predicted_probs)
    valid_indices = ~np.isnan(predicted_probs_np) & ~np.isinf(predicted_probs_np) # Also filter inf
    true_outcomes_np = true_outcomes_np[valid_indices]
    predicted_probs_np = predicted_probs_np[valid_indices]
    n_samples = len(true_outcomes_np)
    if n_samples == 0:
        return {"ece": None, "mce": None, "reliability_diagram_data": [], "brier_score": None, "n_calib_samples": 0}

    brier = np.mean((predicted_probs_np - true_outcomes_np)**2) if n_samples > 0 else None
    ece, mce, reliability_bins_out = None, None, [] # Initialize reliability_bins_out as list

    if n_samples >= n_bins:
        bin_limits = np.linspace(0, 1, n_bins + 1)
        predicted_probs_clipped = np.clip(predicted_probs_np, 0.0, 1.0)
        binids = np.digitize(predicted_probs_clipped, bin_limits[1:-1], right=False)
        bin_sums_probs, bin_sums_true, bin_counts = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
        for i in range(n_samples):
            bin_idx = binids[i]
            if bin_idx == n_bins: bin_idx = n_bins - 1
            bin_sums_probs[bin_idx] += predicted_probs_clipped[i]
            bin_sums_true[bin_idx] += true_outcomes_np[i]
            bin_counts[bin_idx] += 1
        current_ece, current_mce = 0.0, 0.0
        weighted_ece_sum, actual_total_samples_in_bins = 0.0, 0
        for k in range(n_bins):
            if bin_counts[k] > 0: # Process bin if it has samples
                avg_pred_prob = bin_sums_probs[k] / bin_counts[k]
                frac_pos = bin_sums_true[k] / bin_counts[k]
                reliability_bins_out.append({
                    "bin_midpoint_approx": (bin_limits[k] + bin_limits[k+1])/2,
                    "avg_predicted_prob": avg_pred_prob,
                    "fraction_of_positives": frac_pos,
                    "count_in_bin": int(bin_counts[k])})
                if bin_counts[k] >= min_bin_samples_for_ece_mce: # Only use well-populated bins for ECE/MCE
                    abs_diff = abs(avg_pred_prob - frac_pos)
                    weighted_ece_sum += bin_counts[k] * abs_diff
                    actual_total_samples_in_bins += bin_counts[k]
                    if abs_diff > current_mce: current_mce = abs_diff
        if actual_total_samples_in_bins > 0:
            ece = weighted_ece_sum / actual_total_samples_in_bins
            mce = current_mce # MCE based on bins meeting min_bin_samples criterion
        elif n_samples > 0 and not reliability_bins_out: # All bins were too small for ECE/MCE
             ece = None # Cannot calculate ECE if no bins meet criteria
             mce = None
    return {"ece": ece, "mce": mce, "reliability_diagram_data": reliability_bins_out, "brier_score": brier, "n_calib_samples": n_samples}

def calculate_p2_game_specific_metrics( # Renamed and focused
        p2_trials_sample, p2_question_details_map,
        teammate_accuracy_for_regret_oracle, p2_overall_team_accuracy_sample):
    y_true_for_s_i, p_correct_label_p2 = [], []
    oracle_regret_score_sum, n_for_regret_oracle, num_qs_with_p_correct_label = 0, 0, 0

    for trial in p2_trials_sample:
        s_i_capability = get_s_i(trial, p2_question_details_map)
        if s_i_capability is None: continue
        trial_probs = trial.get("probs"); true_correct_answer_label = trial.get("correct_answer")
        prob_for_true_correct_label = None
        if isinstance(trial_probs, dict) and true_correct_answer_label is not None:
            prob_for_true_correct_label = trial_probs.get(true_correct_answer_label)
        if prob_for_true_correct_label is not None:
            y_true_for_s_i.append(s_i_capability)
            p_correct_label_p2.append(prob_for_true_correct_label)
            num_qs_with_p_correct_label +=1
            if prob_for_true_correct_label >= teammate_accuracy_for_regret_oracle:
                oracle_regret_score_sum += s_i_capability
            else: oracle_regret_score_sum += teammate_accuracy_for_regret_oracle
            n_for_regret_oracle += 1
    auroc_p2_ans_conf = None
    if num_qs_with_p_correct_label > 1 and len(set(y_true_for_s_i)) > 1:
        try: auroc_p2_ans_conf = roc_auc_score(y_true_for_s_i, p_correct_label_p2)
        except ValueError: auroc_p2_ans_conf = 0.5
    elif num_qs_with_p_correct_label > 0: auroc_p2_ans_conf = 0.5
    
    oracle_prob_self_vs_tafn_accuracy = oracle_regret_score_sum/n_for_regret_oracle if n_for_regret_oracle > 0 else None
    regret = oracle_prob_self_vs_tafn_accuracy - p2_overall_team_accuracy_sample if all(x is not None for x in [oracle_prob_self_vs_tafn_accuracy, p2_overall_team_accuracy_sample]) else None
    
    p2_game_ans_conf_calib_metrics_dict = calculate_calibration_metrics(y_true_for_s_i, p_correct_label_p2)

    return {
        "auroc_P2_ans_conf_vs_S_i": auroc_p2_ans_conf,
        "n_for_auroc_P2_ans_conf": num_qs_with_p_correct_label,
        "oracle_acc_using_P2_ans_conf": oracle_prob_self_vs_tafn_accuracy,
        "n_for_oracle_using_P2_ans_conf": n_for_regret_oracle,
        "regret_vs_oracle_using_P2_ans_conf": regret,
        "p2_game_cal_ece_ans_conf_vs_S_i": p2_game_ans_conf_calib_metrics_dict.get("ece"),
        "p2_game_cal_mce_ans_conf_vs_S_i": p2_game_ans_conf_calib_metrics_dict.get("mce"),
        "p2_game_cal_brier_ans_conf_vs_S_i": p2_game_ans_conf_calib_metrics_dict.get("brier_score"),
        "p2_game_cal_n_samples_ans_conf_vs_S_i": p2_game_ans_conf_calib_metrics_dict.get("n_calib_samples"),
        # Store the diagram data for point estimate, not for individual bootstrap results
        "p2_game_cal_reliability_diagram_ans_conf_vs_S_i": p2_game_ans_conf_calib_metrics_dict.get("reliability_diagram_data")
    }

def calculate_metrics_from_data(p2_trials_sample, subject_true_capability,
                                teammate_p2_target_accuracy, p2_question_details_map,
                                teammate_accuracy_for_regret_oracle):
    if not p2_trials_sample: return None
    n_p2_sample = len(p2_trials_sample)
    p2_total_final_score_sample = sum(1 for trial in p2_trials_sample if trial.get('team_correct'))
    p2_overall_team_accuracy_sample = p2_total_final_score_sample / n_p2_sample if n_p2_sample > 0 else 0
    self_answered_trials = [t for t in p2_trials_sample if t.get('delegation_choice') == "Self"]
    n_self_answered = len(self_answered_trials)
    self_answered_correct_score = sum(1 for t in self_answered_trials if t.get('team_correct'))
    p2_acc_on_self_answered = self_answered_correct_score / n_self_answered if n_self_answered > 0 else None
    delegated_trials = [t for t in p2_trials_sample if t.get('delegation_choice') == "Teammate"]
    n_delegated = len(delegated_trials)
    delegated_correct_score = sum(1 for t in delegated_trials if t.get('team_correct'))
    p2_acc_on_delegated = delegated_correct_score / n_delegated if n_delegated > 0 else None
    delegation_rate_sample = n_delegated / n_p2_sample if n_p2_sample > 0 else 0
    ssag, percent_ssag = None, None
    if p2_acc_on_self_answered is not None and subject_true_capability is not None:
        ssag = p2_acc_on_self_answered - subject_true_capability
        if abs(subject_true_capability) > 1e-9: percent_ssag = (ssag / subject_true_capability) * 100.0
        elif ssag is not None: percent_ssag = float('inf') if ssag > 1e-9 else (0.0 if abs(ssag) < 1e-9 else float('-inf'))

    baseline_always_self = subject_true_capability
    baseline_always_teammate = teammate_p2_target_accuracy
    baseline_random_50_50 = 0.5 * subject_true_capability + 0.5 * teammate_p2_target_accuracy
    baseline_optimal_simple_aggregate = max(subject_true_capability, teammate_p2_target_accuracy)
    oracle_p2_score_sum_sample, optimal_decisions_made_sample, valid_s_i_trials_sample = 0, 0, 0
    tp, fp, fn, tn = 0, 0, 0, 0
    y_true_for_delegation_auroc, model_scores_for_delegation_auroc = [], []
    for trial in p2_trials_sample:
        s_i = get_s_i(trial, p2_question_details_map)
        if s_i is None: continue
        valid_s_i_trials_sample += 1
        if s_i >= teammate_p2_target_accuracy: oracle_p2_score_sum_sample += s_i
        else: oracle_p2_score_sum_sample += teammate_p2_target_accuracy
        optimal_choice_is_self = (s_i >= teammate_p2_target_accuracy)
        actual_choice_was_self = (trial.get('delegation_choice') == "Self")
        if optimal_choice_is_self and actual_choice_was_self: tp += 1
        elif not optimal_choice_is_self and actual_choice_was_self: fp += 1
        elif optimal_choice_is_self and not actual_choice_was_self: fn += 1
        elif not optimal_choice_is_self and not actual_choice_was_self: tn += 1
        if optimal_choice_is_self == actual_choice_was_self: optimal_decisions_made_sample += 1
        trial_probs = trial.get("probs")
        if isinstance(trial_probs, dict):
            prob_t = trial_probs.get('T')
            if prob_t is None:
                #assume it is 1 - sum of other probs 1 epsilon
                prob_t = 1.0 - sum(v for k, v in trial_probs.items() if k != 'T') - 1e-9
            model_score_for_self = 1.0 - prob_t
            y_true_for_delegation_auroc.append(1 if optimal_choice_is_self else 0)
            model_scores_for_delegation_auroc.append(model_score_for_self)
    if valid_s_i_trials_sample == 0: oracle_p2_accuracy_sample, decision_accuracy_sample = None, None
    else:
        oracle_p2_accuracy_sample = oracle_p2_score_sum_sample / valid_s_i_trials_sample
        decision_accuracy_sample = optimal_decisions_made_sample / valid_s_i_trials_sample
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    f1_score = 2*(precision*recall)/(precision+recall) if precision is not None and recall is not None and (precision+recall)>0 else None
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else None
    d_prime, criterion_c = None, None
    H, F = recall, false_positive_rate
    if H is not None and F is not None:
        N_pos_actual, N_neg_actual = tp + fn, fp + tn
        H_adj, F_adj = H, F
        if N_pos_actual == 0: H_adj = 0.5
        elif H == 1.0: H_adj = 1.0 - (1.0/(2*N_pos_actual)) if N_pos_actual > 0 else 1.0
        elif H == 0.0: H_adj = 0.0 + (1.0/(2*N_pos_actual)) if N_pos_actual > 0 else 0.0
        if N_neg_actual == 0: F_adj = 0.5
        elif F == 1.0: F_adj = 1.0 - (1.0/(2*N_neg_actual)) if N_neg_actual > 0 else 1.0
        elif F == 0.0: F_adj = 0.0 + (1.0/(2*N_neg_actual)) if N_neg_actual > 0 else 0.0
        if 0 < H_adj < 1 and 0 < F_adj < 1:
            d_prime = norm.ppf(H_adj) - norm.ppf(F_adj)
            criterion_c = -0.5 * (norm.ppf(H_adj) + norm.ppf(F_adj))
    delegation_auroc, op_thresh_est = None, None
    max_poss_dqa, opt_thresh_max_dqa, dqa_loss = None, None, None
    dqapi_chance, dqapi_smart, base_dqa_smart_calc = None, None, None
    if len(y_true_for_delegation_auroc) > 1 and len(set(y_true_for_delegation_auroc)) > 1:
        try:
            delegation_auroc = roc_auc_score(y_true_for_delegation_auroc, model_scores_for_delegation_auroc)
            fpr_r, tpr_r, thresh_r = roc_curve(y_true_for_delegation_auroc, model_scores_for_delegation_auroc)
            if H is not None and F is not None and len(fpr_r)>0 and len(tpr_r)>0:
                dist = np.sqrt((fpr_r-F)**2 + (tpr_r-H)**2)
                if len(dist)>0: idx_cl = np.argmin(dist); op_thresh_est = thresh_r[idx_cl] if idx_cl < len(thresh_r) else (thresh_r[-1] if len(thresh_r)>0 else None)
            n_pos_roc, n_neg_roc = sum(y_true_for_delegation_auroc), len(y_true_for_delegation_auroc) - sum(y_true_for_delegation_auroc)
            best_dqa = -1.0
            if n_pos_roc + n_neg_roc > 0 :
                for i, thr_v in enumerate(thresh_r):
                    tp_thr,tn_thr = tpr_r[i]*n_pos_roc, (1-fpr_r[i])*n_neg_roc
                    curr_dqa = (tp_thr+tn_thr)/(n_pos_roc+n_neg_roc)
                    if curr_dqa > best_dqa: best_dqa=curr_dqa; opt_thresh_max_dqa=thr_v
                max_poss_dqa = best_dqa
            if max_poss_dqa is not None and decision_accuracy_sample is not None: dqa_loss = max_poss_dqa - decision_accuracy_sample
            if decision_accuracy_sample is not None and max_poss_dqa is not None:
                b_dqa_chance = 0.5; den_chance = max_poss_dqa - b_dqa_chance
                if abs(den_chance) < 1e-9: dqapi_chance = 0.0 if abs(decision_accuracy_sample-b_dqa_chance)<1e-9 else f"undef(max_dqa={b_dqa_chance})"
                else: dqapi_chance = (decision_accuracy_sample-b_dqa_chance)/den_chance
                if len(y_true_for_delegation_auroc)>0:
                    p_opt_self = sum(y_true_for_delegation_auroc)/len(y_true_for_delegation_auroc)
                    base_dqa_smart_calc = max(p_opt_self, 1.0-p_opt_self)
                    den_smart = max_poss_dqa - base_dqa_smart_calc
                    if abs(den_smart) < 1e-9: dqapi_smart = 0.0 if abs(decision_accuracy_sample-base_dqa_smart_calc)<1e-9 else f"undef(max_dqa={base_dqa_smart_calc})"
                    else: dqapi_smart = (decision_accuracy_sample-base_dqa_smart_calc)/den_smart
                else: base_dqa_smart_calc=0.5; dqapi_smart=None
        except ValueError: delegation_auroc = 0.5 if len(set(y_true_for_delegation_auroc)) == 1 else None
    elif len(y_true_for_delegation_auroc) > 0 : delegation_auroc = 0.5

    # --- Calibration of Delegation Decision Signal (1-P(T) vs Optimal Choice) ---
    delegation_signal_calib_metrics_dict = None
    if len(y_true_for_delegation_auroc) > 0 and len(model_scores_for_delegation_auroc) > 0:
        # y_true_for_delegation_auroc IS "Optimal_Choice_is_Self" (0 or 1)
        # model_scores_for_delegation_auroc IS "1 - P(T)"
        delegation_signal_calib_metrics_dict = calculate_calibration_metrics(
            y_true_for_delegation_auroc,
            model_scores_for_delegation_auroc
        )

    # Calculate P2-game specific metrics including calibration for P(CorrectAnswerLabel)
    p2_game_specific_metrics = calculate_p2_game_specific_metrics( # Renamed function call
        p2_trials_sample, p2_question_details_map, teammate_accuracy_for_regret_oracle, p2_overall_team_accuracy_sample
    )
    metrics = {
        "p2_overall_team_accuracy": p2_overall_team_accuracy_sample,
        "p2_accuracy_on_self_answered": p2_acc_on_self_answered, "n_self_answered_p2": n_self_answered,
        "p2_accuracy_on_delegated": p2_acc_on_delegated, "n_delegated_p2": n_delegated,
        "delegation_rate_p2": delegation_rate_sample,
        "ssag (Self-Selection Accuracy Gain)": ssag, "percent_ssag": percent_ssag,
        "oracle_p2_accuracy (S_i based)": oracle_p2_accuracy_sample,
        "decision_quality_accuracy (DQA)": decision_accuracy_sample,
        "n_p2_trials_for_oracle_dqa": valid_s_i_trials_sample,
        "classification_TP": tp, "classification_FP": fp, "classification_FN": fn, "classification_TN": tn,
        "classification_precision": precision, "classification_recall (sensitivity/H)": recall,
        "classification_specificity": specificity, "classification_FPR (F)": false_positive_rate,
        "classification_f1_score": f1_score,
        "delegation_decision_auroc (1-P(T) based)": delegation_auroc,
        "sdt_d_prime": d_prime, "sdt_criterion_c": criterion_c,
        "estimated_operational_threshold_for_self_answer (1-P(T) based)": op_thresh_est,
        "max_possible_dqa_for_signal (1-P(T) based)": max_poss_dqa,
        "optimal_threshold_for_max_dqa (1-P(T) based)": opt_thresh_max_dqa,
        "dqa_loss_due_to_suboptimal_threshold (1-P(T) based)": dqa_loss,
        "dqapi_vs_chance (0.5)": dqapi_chance,
        "baseline_defined_DQA_SmartRandom (P2 sample based)": base_dqa_smart_calc,
        "dqapi_vs_smart_random": dqapi_smart,
        "baseline_defined_AlwaysSelf": baseline_always_self, "baseline_defined_AlwaysTeammate": baseline_always_teammate,
        "baseline_defined_Random5050": baseline_random_50_50, "baseline_defined_OptimalSimpleAggregate": baseline_optimal_simple_aggregate,
        **p2_game_specific_metrics # Merge P2 calibration and regret metrics
    }
    if delegation_signal_calib_metrics_dict:
        metrics["deleg_signal_cal_ece"] = delegation_signal_calib_metrics_dict.get("ece")
        metrics["deleg_signal_cal_mce"] = delegation_signal_calib_metrics_dict.get("mce")
        metrics["deleg_signal_cal_brier"] = delegation_signal_calib_metrics_dict.get("brier_score")
        metrics["deleg_signal_cal_n_samples"] = delegation_signal_calib_metrics_dict.get("n_calib_samples")

    baselines_for_norm = {"OptimalSimpleAggregate":baseline_optimal_simple_aggregate, "AlwaysSelf":baseline_always_self, "AlwaysTeammate":baseline_always_teammate, "Random5050":baseline_random_50_50}
    for name, bl_val in baselines_for_norm.items():
        ig, onis = None, None
        if p2_overall_team_accuracy_sample is not None and bl_val is not None:
            ig = p2_overall_team_accuracy_sample - bl_val
            oracle_s_i_based = metrics.get("oracle_p2_accuracy (S_i based)")
            if oracle_s_i_based is not None:
                den_onis = oracle_s_i_based - bl_val
                if abs(den_onis) < 1e-9: onis = 0.0 if abs(p2_overall_team_accuracy_sample-bl_val) <1e-9 else f"undef(oracle=bl_{name})"
                else: onis = (p2_overall_team_accuracy_sample - bl_val) / den_onis
        metrics[f"IG_vs_{name}"] = ig; metrics[f"ONIS_vs_{name}"] = onis
    return metrics

def bootstrap_ci(data_list, metric_calculator_func, n_bootstraps=1000, alpha=0.05,
                 subject_true_capability=None, teammate_p2_target_accuracy=None,
                 p2_question_details_map=None, teammate_accuracy_for_regret_oracle=None):
    # This function remains the same as it correctly passes all necessary args to metric_calculator_func
    if not data_list: return None
    n_data = len(data_list); bootstrapped_metrics_log = {}; print_interval = max(1, n_bootstraps // 20)
    for i in range(n_bootstraps):
        resample_indices = np.random.choice(range(n_data), size=n_data, replace=True)
        resample_data = [data_list[idx] for idx in resample_indices]
        current_metrics = metric_calculator_func(resample_data, subject_true_capability, teammate_p2_target_accuracy, p2_question_details_map, teammate_accuracy_for_regret_oracle)
        if current_metrics:
            for key, value in current_metrics.items():
                # Only try to bootstrap metrics that are scalar and numeric
                if isinstance(value, (int, float)) and value is not None and not np.isnan(value) and not np.isinf(value):
                    bootstrapped_metrics_log.setdefault(key, []).append(value)
        if (i + 1) % print_interval == 0 : print(f"  Bootstrap progress: {i+1}/{n_bootstraps}", end='\r', file=sys.stderr)
    print(file=sys.stderr); cis = {} # Newline after progress
    for metric_name, values_list in bootstrapped_metrics_log.items():
        if values_list and len(values_list) > n_bootstraps * 0.1 : # Ensure enough valid bootstrap samples
            cis[metric_name] = (np.percentile(values_list, alpha/2.0*100), np.percentile(values_list, (1-alpha/2.0)*100))
        else: cis[metric_name] = (None, None) # Not enough valid data from bootstraps for this metric
    return cis


def parse_log_for_true_accuracy(json_file_path):
    log_file_path = None
    if "_game_data.json" in json_file_path: log_file_path = json_file_path.replace("_game_data.json", ".log")
    elif "_results.json" in json_file_path: log_file_path = json_file_path.replace("_results.json", ".log")
    else: base, ext = os.path.splitext(json_file_path); log_file_path = base + ".log" if ext.lower() == ".json" else None
    if not log_file_path or not os.path.exists(log_file_path): print(f"Warning: Log file not found at {log_file_path}", file=sys.stderr); return None
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r"Phase 1 self-accuracy \(from completed results, total - phase2\): \d+/\d+ \((\d+\.\d+)%\)", line)
                if match: return float(match.group(1)) / 100.0
    except Exception as e: print(f"Warning: Error reading log {log_file_path}: {e}", file=sys.stderr)
    print(f"Warning: Target accuracy line not found in {log_file_path}", file=sys.stderr); return None

def calculate_capabilities_auroc_and_calibration(capabilities_file_path, n_bins_calib=10):
    if not capabilities_file_path or not os.path.exists(capabilities_file_path):
        print(f"Capabilities file not found: {capabilities_file_path}", file=sys.stderr); return None, 0, None
    try:
        with open(capabilities_file_path, 'r', encoding='utf-8') as f: cap_data = json.load(f)
    except Exception as e: print(f"Error loading cap file {capabilities_file_path}: {e}", file=sys.stderr); return None, 0, None
    if "results" not in cap_data or not isinstance(cap_data["results"], dict):
        print(f"Invalid cap file format: {capabilities_file_path}", file=sys.stderr); return None, 0, None
    y_true_cap, p_chosen_cap = [], []; questions_with_probs_cap = 0
    for q_id, res_data in cap_data["results"].items():
        is_corr, subj_ans, probs_dict = res_data.get("is_correct"), res_data.get("subject_answer"), res_data.get("probs")
        if is_corr is not None and subj_ans is not None and isinstance(probs_dict, dict):
            prob_chosen = probs_dict.get(subj_ans)
            if prob_chosen is not None: y_true_cap.append(1 if is_corr else 0); p_chosen_cap.append(prob_chosen); questions_with_probs_cap += 1
    auroc_cap = None
    if questions_with_probs_cap > 1 and len(set(y_true_cap)) > 1:
        try: auroc_cap = roc_auc_score(y_true_cap, p_chosen_cap)
        except ValueError: auroc_cap = 0.5
    elif questions_with_probs_cap > 0: auroc_cap = 0.5
    calibration_metrics_cap = calculate_calibration_metrics(y_true_cap, p_chosen_cap, n_bins=n_bins_calib)
    return auroc_cap, questions_with_probs_cap, calibration_metrics_cap

def analyze_introspection_from_file(file_path, n_bootstraps=2000):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: game_data_from_json = json.load(f)
    except Exception as e: print(f"Error loading file {file_path}: {e}", file=sys.stderr); return None
    if not isinstance(game_data_from_json, dict) or "results" not in game_data_from_json: print(f"Invalid JSON in {file_path}", file=sys.stderr); return None
    all_trials_list, config_params_dict = game_data_from_json.get("results", []), game_data_from_json
    phase2_trials_list = [t for t in all_trials_list if t.get('phase') == 2]
    if not phase2_trials_list: print("Error: No P2 trials.", file=sys.stderr); return None

    subject_true_capability = parse_log_for_true_accuracy(file_path)
    if subject_true_capability is None: subject_true_capability = config_params_dict.get("true_subject_accuracy")
    if subject_true_capability is None:
        capabilities_file_path_for_stv = config_params_dict.get("capabilities_file")
        if capabilities_file_path_for_stv and os.path.exists(capabilities_file_path_for_stv):
            try:
                with open(capabilities_file_path_for_stv, 'r') as cf: cap_data = json.load(cf)
                subject_true_capability = cap_data.get("accuracy")
                if subject_true_capability is not None: print(f"Note: Used STV from linked capabilities file: {capabilities_file_path_for_stv}", file=sys.stderr)
            except Exception as e: print(f"Warning: Could not re-load/parse capabilities file for STV: {e}", file=sys.stderr)
    if subject_true_capability is None: subject_true_capability = config_params_dict.get("subject_accuracy_phase1")
    if subject_true_capability is None: print("Fatal: Cannot determine Subject True Capability.", file=sys.stderr); return None
    print(f"Using subject_true_capability: {subject_true_capability:.4f}", file=sys.stderr)

    teammate_p2_target_accuracy = config_params_dict.get("teammate_accuracy_phase2")
    if teammate_p2_target_accuracy is None: print("Error: Missing 'teammate_accuracy_phase2'.", file=sys.stderr); return None
    teammate_accuracy_for_regret_oracle = config_params_dict.get("teammate_accuracy_phase1", teammate_p2_target_accuracy)
    print(f"Using teammate_accuracy_for_regret_oracle: {teammate_accuracy_for_regret_oracle:.4f}", file=sys.stderr)

    phase2_questions_config_list = config_params_dict.get("phase2_questions", [])
    p2_question_details_s_i_map = {q.get("id"): {"is_correct": q.get("is_correct")} for q in phase2_questions_config_list if q.get("id") and q.get("is_correct") is not None}
    if not p2_question_details_s_i_map: print("Error: S_i map empty.", file=sys.stderr); return None
    if not all(trial.get("question_id") in p2_question_details_s_i_map for trial in phase2_trials_list):
        print("Fatal: S_i data missing for some P2 trials. Analysis halted.", file=sys.stderr); return None
    print(f"S_i map successfully built for {len(p2_question_details_s_i_map)} P2 questions.", file=sys.stderr)

    # --- Capabilities AUROC and Calibration (Point Estimates) ---
    capabilities_file_path = config_params_dict.get("capabilities_file")
    capabilities_auroc, cap_auroc_n_qs, capabilities_calibration_point_estimates_dict = calculate_capabilities_auroc_and_calibration(capabilities_file_path)
    
    print(f"\nCapabilities AUROC (P(chosen_answer) vs S_i from {capabilities_file_path}): {capabilities_auroc:.4f} (N={cap_auroc_n_qs})" if capabilities_auroc is not None else "\nCapabilities AUROC: Not calculated")
    if capabilities_calibration_point_estimates_dict:
        print("Capabilities Calibration Metrics (P(chosen_answer) vs S_i):")
        for k, v_val in capabilities_calibration_point_estimates_dict.items():
            if k != "reliability_diagram_data": print(f"  {k}: {v_val:.4f}" if isinstance(v_val, float) else f"  {k}: {v_val}")
    
    # --- P2 Game Metrics Point Estimates (includes P2 calibration via calculate_metrics_from_data -> calculate_p2_game_specific_metrics -> calculate_calibration_metrics) ---
    print("\nCalculating game metrics point estimates (includes P2 calibration)...", file=sys.stderr)
    point_estimates_game = calculate_metrics_from_data(
        phase2_trials_list, subject_true_capability,
        teammate_p2_target_accuracy, p2_question_details_s_i_map,
        teammate_accuracy_for_regret_oracle
    )
    if not point_estimates_game: print("Could not calculate game metrics.", file=sys.stderr); return None
    
    print("\nPoint Estimates (Game Performance & P2 Game Calibrations):")
    # Separate P2 calibration for clarity if needed, or print all from point_estimates_game
    for key, value in point_estimates_game.items():
        # Exclude reliability diagram data from this summary print, it's in the JSON
        if key.endswith("_reliability_diagram_data"): continue
        if isinstance(value, float): print(f"  {key}: {value:.4f}")
        else: print(f"  {key}: {value}")


    # --- Bootstrap CIs for Game Metrics (including P2 calibration scalars) ---
    print(f"\nCalculating bootstrap CIs for Game Metrics (incl. P2 Calibration) with {n_bootstraps} samples...", file=sys.stderr)
    bootstrap_cis_game = bootstrap_ci(
        phase2_trials_list, calculate_metrics_from_data,
        n_bootstraps=n_bootstraps, alpha=0.05,
        subject_true_capability=subject_true_capability,
        teammate_p2_target_accuracy=teammate_p2_target_accuracy,
        p2_question_details_map=p2_question_details_s_i_map,
        teammate_accuracy_for_regret_oracle=teammate_accuracy_for_regret_oracle
    )
    if bootstrap_cis_game:
        print("\nBootstrap 95% CIs (Game Performance & P2 Calibration):")
        for metric_name, ci_bounds in bootstrap_cis_game.items():
            pe = point_estimates_game.get(metric_name) # Get corresponding point estimate
            # Only print CI if point estimate is numeric and CI bounds are valid
            if pe is not None and isinstance(pe, (float, int)) and ci_bounds != (None, None):
                 print(f"  {metric_name}: [{ci_bounds[0]:.4f}, {ci_bounds[1]:.4f}] (Point Est: {pe:.4f})")
            elif pe is not None: # Point estimate exists but is not numeric (e.g. "undefined") or CI failed
                 print(f"  {metric_name}: CI not computed or N/A (Point Est: {pe})")
    
    # --- Bootstrap CIs for Capabilities Calibration Metrics (ECE, MCE, Brier) ---
    cap_calib_cis_bootstrapped = {}
    if capabilities_file_path and os.path.exists(capabilities_file_path) and capabilities_calibration_point_estimates_dict:
        try:
            with open(capabilities_file_path, 'r', encoding='utf-8') as f: cap_data_for_bootstrap = json.load(f)
            if "results" in cap_data_for_bootstrap and isinstance(cap_data_for_bootstrap["results"], dict):
                cap_bootstrap_data_pairs = []
                for _q_id, res_data in cap_data_for_bootstrap["results"].items():
                    is_corr, subj_ans, probs_dict = res_data.get("is_correct"), res_data.get("subject_answer"), res_data.get("probs")
                    if is_corr is not None and subj_ans is not None and isinstance(probs_dict, dict):
                        prob_chosen = probs_dict.get(subj_ans)
                        if prob_chosen is not None: cap_bootstrap_data_pairs.append((1 if is_corr else 0, prob_chosen))
                
                if cap_bootstrap_data_pairs:
                    print(f"\nCalculating bootstrap CIs for Capabilities Calibration (N_pairs={len(cap_bootstrap_data_pairs)})...", file=sys.stderr)
                    bootstrapped_cap_calib_log = {"ece":[], "mce":[], "brier_score":[]}
                    cap_print_interval = max(1, n_bootstraps // 20)
                    for i in range(n_bootstraps):
                        resample_indices_cap = np.random.choice(len(cap_bootstrap_data_pairs), size=len(cap_bootstrap_data_pairs), replace=True)
                        resample_data_cap = [cap_bootstrap_data_pairs[idx] for idx in resample_indices_cap]
                        if not resample_data_cap: continue
                        true_o_cap, pred_p_cap = zip(*resample_data_cap)
                        current_cap_calib_metrics = calculate_calibration_metrics(list(true_o_cap), list(pred_p_cap))
                        for metric_key in ["ece", "mce", "brier_score"]: # Scalar metrics only
                            val = current_cap_calib_metrics.get(metric_key)
                            if val is not None and isinstance(val, (float,int)): # Ensure numeric
                                bootstrapped_cap_calib_log[metric_key].append(val)
                        if (i + 1) % cap_print_interval == 0 : print(f"  Capabilities Calib Bootstrap: {i+1}/{n_bootstraps}", end='\r', file=sys.stderr)
                    print(file=sys.stderr) # Newline

                    for metric_name, values_list in bootstrapped_cap_calib_log.items():
                        if values_list and len(values_list) > n_bootstraps * 0.1 :
                            cap_calib_cis_bootstrapped[metric_name] = (np.percentile(values_list, 2.5), np.percentile(values_list, 97.5))
                        else: cap_calib_cis_bootstrapped[metric_name] = (None, None)
        except Exception as e: print(f"Error during capabilities calibration bootstrapping: {e}", file=sys.stderr)

    if cap_calib_cis_bootstrapped:
        print("\nBootstrap 95% CIs (Capabilities Calibration):")
        for metric_name, ci_bounds in cap_calib_cis_bootstrapped.items():
            # Get point estimate from the dict calculated earlier
            pe_cap = capabilities_calibration_point_estimates_dict.get(metric_name) 
            if pe_cap is not None and isinstance(pe_cap, (float,int)) and ci_bounds != (None,None):
                print(f"  capabilities_{metric_name}: [{ci_bounds[0]:.4f}, {ci_bounds[1]:.4f}] (Point Est: {pe_cap:.4f})")
            elif pe_cap is not None:
                 print(f"  capabilities_{metric_name}: CI not computed or N/A (Point Est: {pe_cap})")


    final_analysis_results = {
        "source_file": os.path.basename(file_path),
        "subject_id": config_params_dict.get("subject_id", "N/A"),
        "subject_true_capability_used_for_analysis": subject_true_capability,
        "teammate_p2_target_accuracy": teammate_p2_target_accuracy,
        "teammate_accuracy_used_for_regret_oracle": teammate_accuracy_for_regret_oracle,
        "capabilities_auroc_of_chosen_answer": capabilities_auroc,
        "capabilities_auroc_n_questions": cap_auroc_n_qs,
        "capabilities_calibration_point_estimates": capabilities_calibration_point_estimates_dict, # Store the whole dict
        "capabilities_calibration_CI_95": cap_calib_cis_bootstrapped,
        "n_phase2_trials_total": len(phase2_trials_list),
        "game_performance_point_estimates": point_estimates_game, # Contains P2 calib point estimates (and diagram)
        "game_performance_bootstrap_95_CI": bootstrap_cis_game if bootstrap_cis_game else {}, # Contains P2 calib CIs
        "n_bootstraps": n_bootstraps,
        "original_game_config": { k: v for k,v in config_params_dict.items() if k not in ["results", "phase1_questions", "phase2_questions"]}
    }
    return final_analysis_results

# --- Main Execution Block (if __name__ == "__main__":) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze introspection metrics from DelegateGame output JSON.")
    parser.add_argument("input_file", help="Path to game JSON (e.g., _game_data.json or _results.json).")
    parser.add_argument("--bootstraps", type=int, default=2000, help="Bootstrap samples.")
    parser.add_argument("--output_file", help="Path to save analysis JSON.")

    args = parser.parse_args()
    print(f"Analyzing file: {args.input_file}", file=sys.stderr)
    analysis_results = analyze_introspection_from_file(args.input_file, n_bootstraps=args.bootstraps)

    if analysis_results and args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                def default_serializer(o):
                    if isinstance(o, np.integer): return int(o)
                    if isinstance(o, np.floating): return float(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    # Let dicts and lists pass through, json.dump will handle them
                    if isinstance(o, (dict, list)): return o 
                    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable: {repr(o)}")
                json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=default_serializer)
            print(f"\nAnalysis results saved to: {args.output_file}")
        except Exception as e: print(f"Error saving analysis results: {e}", file=sys.stderr)

    if not analysis_results: print("\nAnalysis could not be completed.", file=sys.stderr)
    else: print("\n--- Analysis Complete ---")