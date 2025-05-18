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

# MODIFIED calculate_metrics_from_data
def calculate_metrics_from_data(p2_trials_sample,
                                subject_true_capability,
                                teammate_p2_target_accuracy,
                                p2_question_details_map):
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

    # --- Self-Selection Accuracy Gain (SSAG) ---
    ssag = None
    percent_ssag = None
    if p2_acc_on_self_answered is not None and subject_true_capability is not None:
        ssag = p2_acc_on_self_answered - subject_true_capability
        if subject_true_capability > 1e-9: # Avoid division by zero or near-zero
            percent_ssag = (ssag / subject_true_capability) * 100.0
        elif ssag is not None : # If STV is 0, any positive p2_acc_on_self is infinite % gain
             percent_ssag = float('inf') if ssag > 1e-9 else (0.0 if abs(ssag) < 1e-9 else float('-inf'))


    baseline_always_self = subject_true_capability
    baseline_always_teammate = teammate_p2_target_accuracy
    baseline_random_50_50 = 0.5 * subject_true_capability + 0.5 * teammate_p2_target_accuracy
    baseline_optimal_simple_aggregate = max(subject_true_capability, teammate_p2_target_accuracy)

    oracle_p2_score_sum_sample = 0
    optimal_decisions_made_sample = 0
    valid_s_i_trials_sample = 0
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
            if prob_t is not None:
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
        if N_pos_actual == 0: H_adj = 0.5
        elif H == 1.0: H_adj = 1.0 - (1.0 / (2 * N_pos_actual)) if N_pos_actual > 0 else 1.0
        elif H == 0.0: H_adj = 0.0 + (1.0 / (2 * N_pos_actual)) if N_pos_actual > 0 else 0.0
        else: H_adj = H
        if N_neg_actual == 0: F_adj = 0.5
        elif F == 1.0: F_adj = 1.0 - (1.0 / (2 * N_neg_actual)) if N_neg_actual > 0 else 1.0
        elif F == 0.0: F_adj = 0.0 + (1.0 / (2 * N_neg_actual)) if N_neg_actual > 0 else 0.0
        else: F_adj = F
        if 0 < H_adj < 1 and 0 < F_adj < 1:
            d_prime = norm.ppf(H_adj) - norm.ppf(F_adj)
            criterion_c = -0.5 * (norm.ppf(H_adj) + norm.ppf(F_adj))

    delegation_auroc, operational_threshold_estimate = None, None
    max_possible_dqa_for_signal, optimal_threshold_for_max_dqa, dqa_loss_due_to_suboptimal_threshold = None, None, None
    dqapi_vs_chance, dqapi_vs_smart_random, baseline_dqa_smart_random = None, None, None


    if len(y_true_for_delegation_auroc) > 1 and len(set(y_true_for_delegation_auroc)) > 1:
        try:
            delegation_auroc = roc_auc_score(y_true_for_delegation_auroc, model_scores_for_delegation_auroc)
            fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_true_for_delegation_auroc, model_scores_for_delegation_auroc)
            if H is not None and F is not None and len(fpr_roc) > 0 and len(tpr_roc) > 0:
                distances = np.sqrt((fpr_roc - F)**2 + (tpr_roc - H)**2)
                if len(distances) > 0:
                    idx_closest_point = np.argmin(distances)
                    if idx_closest_point < len(thresholds_roc): operational_threshold_estimate = thresholds_roc[idx_closest_point]
                    elif len(thresholds_roc) > 0 : operational_threshold_estimate = thresholds_roc[-1]
            num_actual_positives = sum(y_true_for_delegation_auroc)
            num_actual_negatives = len(y_true_for_delegation_auroc) - num_actual_positives
            best_dqa_found = -1.0
            if num_actual_positives + num_actual_negatives > 0:
                for i_thr, thr_val in enumerate(thresholds_roc):
                    tp_at_thr = tpr_roc[i_thr] * num_actual_positives
                    tn_at_thr = (1 - fpr_roc[i_thr]) * num_actual_negatives
                    current_dqa = (tp_at_thr + tn_at_thr) / (num_actual_positives + num_actual_negatives)
                    if current_dqa > best_dqa_found: best_dqa_found = current_dqa; optimal_threshold_for_max_dqa = thr_val
                max_possible_dqa_for_signal = best_dqa_found
            if max_possible_dqa_for_signal is not None and decision_accuracy_sample is not None:
                dqa_loss_due_to_suboptimal_threshold = max_possible_dqa_for_signal - decision_accuracy_sample
            
            # DQAPI calculations
            if decision_accuracy_sample is not None and max_possible_dqa_for_signal is not None:
                baseline_dqa_chance = 0.5
                den_dqapi_chance = max_possible_dqa_for_signal - baseline_dqa_chance
                if abs(den_dqapi_chance) < 1e-9: dqapi_vs_chance = 0.0 if abs(decision_accuracy_sample - baseline_dqa_chance) < 1e-9 else f"undef (max_dqa={baseline_dqa_chance})"
                else: dqapi_vs_chance = (decision_accuracy_sample - baseline_dqa_chance) / den_dqapi_chance
                
                p_optimal_self = sum(y_true_for_delegation_auroc) / len(y_true_for_delegation_auroc)
                baseline_dqa_smart_random = max(p_optimal_self, 1.0 - p_optimal_self)
                den_dqapi_smart = max_possible_dqa_for_signal - baseline_dqa_smart_random
                if abs(den_dqapi_smart) < 1e-9: dqapi_vs_smart_random = 0.0 if abs(decision_accuracy_sample - baseline_dqa_smart_random) < 1e-9 else f"undef (max_dqa={baseline_dqa_smart_random})"
                else: dqapi_vs_smart_random = (decision_accuracy_sample - baseline_dqa_smart_random) / den_dqapi_smart

        except ValueError: delegation_auroc = 0.5 if len(set(y_true_for_delegation_auroc)) == 1 else None
    elif len(y_true_for_delegation_auroc) > 0 : delegation_auroc = 0.5

    metrics = {
        "p2_overall_team_accuracy": p2_overall_team_accuracy_sample,
        "p2_accuracy_on_self_answered": p2_acc_on_self_answered, "n_self_answered_p2": n_self_answered,
        "p2_accuracy_on_delegated": p2_acc_on_delegated, "n_delegated_p2": n_delegated,
        "delegation_rate_p2": delegation_rate_sample,
        "ssag (Self-Selection Accuracy Gain)": ssag, # New
        "percent_ssag": percent_ssag,               # New
        "oracle_p2_accuracy": oracle_p2_accuracy_sample,
        "decision_quality_accuracy (DQA)": decision_accuracy_sample,
        "n_p2_trials_for_oracle_dqa": valid_s_i_trials_sample,
        "classification_TP": tp, "classification_FP": fp, "classification_FN": fn, "classification_TN": tn,
        "classification_precision": precision, "classification_recall (sensitivity/H)": recall,
        "classification_specificity": specificity, "classification_FPR (F)": false_positive_rate,
        "classification_f1_score": f1_score,
        "delegation_decision_auroc": delegation_auroc,
        "sdt_d_prime": d_prime, "sdt_criterion_c": criterion_c,
        "estimated_operational_threshold_for_self_answer": operational_threshold_estimate,
        "max_possible_dqa_for_signal (from AUROC)": max_possible_dqa_for_signal,
        "optimal_threshold_for_max_dqa": optimal_threshold_for_max_dqa,
        "dqa_loss_due_to_suboptimal_threshold": dqa_loss_due_to_suboptimal_threshold,
        "dqapi_vs_chance (0.5)": dqapi_vs_chance, # New
        "baseline_defined_DQA_SmartRandom": baseline_dqa_smart_random, # New
        "dqapi_vs_smart_random": dqapi_vs_smart_random, # New
        "baseline_defined_AlwaysSelf": baseline_always_self, "baseline_defined_AlwaysTeammate": baseline_always_teammate,
        "baseline_defined_Random5050": baseline_random_50_50, "baseline_defined_OptimalSimpleAggregate": baseline_optimal_simple_aggregate,
    }
    baselines_for_norm = {"OptimalSimpleAggregate": baseline_optimal_simple_aggregate, "AlwaysSelf": baseline_always_self, "AlwaysTeammate": baseline_always_teammate, "Random5050": baseline_random_50_50}
    for name, baseline_val in baselines_for_norm.items():
        ig, onis = None, None
        if p2_overall_team_accuracy_sample is not None and baseline_val is not None:
            ig = p2_overall_team_accuracy_sample - baseline_val
            if oracle_p2_accuracy_sample is not None:
                den_onis = oracle_p2_accuracy_sample - baseline_val
                if abs(den_onis) < 1e-9: onis = 0.0 if abs(p2_overall_team_accuracy_sample-baseline_val) < 1e-9 else f"undefined (oracle=baseline_{name})"
                else: onis = (p2_overall_team_accuracy_sample - baseline_val) / den_onis
        metrics[f"IG_vs_{name}"] = ig; metrics[f"ONIS_vs_{name}"] = onis
    return metrics

def bootstrap_ci(data_list, metric_calculator_func, n_bootstraps=1000, alpha=0.05,
                 subject_true_capability=None, teammate_p2_target_accuracy=None,
                 p2_question_details_map=None):
    if not data_list: return None
    n_data = len(data_list)
    bootstrapped_metrics_log = {}
    print_interval = max(1, n_bootstraps // 20)
    for i in range(n_bootstraps):
        resample_indices = np.random.choice(range(n_data), size=n_data, replace=True)
        resample_data = [data_list[idx] for idx in resample_indices]
        current_metrics = metric_calculator_func(
            resample_data, subject_true_capability,
            teammate_p2_target_accuracy, p2_question_details_map
        )
        if current_metrics:
            for key, value in current_metrics.items():
                if isinstance(value, (int, float)) and value is not None : # Ensure value is numeric and not None
                    bootstrapped_metrics_log.setdefault(key, []).append(value)
        if (i + 1) % print_interval == 0 : print(f"  Bootstrap progress: {i+1}/{n_bootstraps}", end='\r', file=sys.stderr)
    print(file=sys.stderr) # Newline after progress
    cis = {}
    for metric_name, values_list in bootstrapped_metrics_log.items():
        if values_list and len(values_list) > n_bootstraps * 0.1 : # Ensure enough valid bootstrap samples
            lower_bound = np.percentile(values_list, alpha / 2.0 * 100)
            upper_bound = np.percentile(values_list, (1 - alpha / 2.0) * 100)
            cis[metric_name] = (lower_bound, upper_bound)
        else: cis[metric_name] = (None, None)
    return cis

def parse_log_for_true_accuracy(json_file_path):
    if "_game_data.json" not in json_file_path:
        if "_results.json" in json_file_path: log_file_path = json_file_path.replace("_results.json", ".log")
        else: base, ext = os.path.splitext(json_file_path); log_file_path = base + ".log" if ext.lower() == ".json" else None
    else: log_file_path = json_file_path.replace("_game_data.json", ".log")
    if not log_file_path or not os.path.exists(log_file_path): print(f"Warning: Log file not found at {log_file_path}", file=sys.stderr); return None
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r"Phase 1 self-accuracy \(from completed results, total - phase2\): \d+/\d+ \((\d+\.\d+)%\)", line)
                if match: return float(match.group(1)) / 100.0
    except Exception as e: print(f"Warning: Error reading log {log_file_path}: {e}", file=sys.stderr)
    print(f"Warning: Target accuracy line not found in {log_file_path}", file=sys.stderr); return None

def calculate_capabilities_auroc(capabilities_file_path):
    if not capabilities_file_path or not os.path.exists(capabilities_file_path):
        print(f"Capabilities file not found: {capabilities_file_path}", file=sys.stderr); return None, 0
    try:
        with open(capabilities_file_path, 'r', encoding='utf-8') as f: cap_data = json.load(f)
    except Exception as e: print(f"Error loading cap file {capabilities_file_path}: {e}", file=sys.stderr); return None, 0
    if "results" not in cap_data or not isinstance(cap_data["results"], dict):
        print(f"Invalid cap file format: {capabilities_file_path}", file=sys.stderr); return None, 0
    y_true, p_chosen = [], []; questions_with_probs = 0
    for q_id, res_data in cap_data["results"].items():
        is_corr, subj_ans, probs_dict = res_data.get("is_correct"), res_data.get("subject_answer"), res_data.get("probs")
        if is_corr is not None and subj_ans is not None and isinstance(probs_dict, dict):
            prob_chosen = probs_dict.get(subj_ans)
            if prob_chosen is not None: y_true.append(1 if is_corr else 0); p_chosen.append(prob_chosen); questions_with_probs += 1
    if questions_with_probs < 2 or len(set(y_true)) < 2:
        print(f"Not enough data/diversity in cap file ({questions_with_probs} Qs) for AUROC.", file=sys.stderr); return None, questions_with_probs
    try: return roc_auc_score(y_true, p_chosen), questions_with_probs
    except ValueError as e: print(f"Error calculating cap AUROC: {e}.", file=sys.stderr); return 0.5, questions_with_probs


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
                if subject_true_capability is not None: print(f"Note: Used STV from linked file '{capabilities_file_path_for_stv}'.", file=sys.stderr)
            except Exception: pass
    if subject_true_capability is None: subject_true_capability = config_params_dict.get("subject_accuracy_phase1")
    if subject_true_capability is None: print("Fatal: Cannot determine STV.", file=sys.stderr); return None
    print(f"Using subject_true_capability: {subject_true_capability:.4f}", file=sys.stderr)

    teammate_p2_target_accuracy = config_params_dict.get("teammate_accuracy_phase2")
    if teammate_p2_target_accuracy is None: print("Error: Missing 'teammate_accuracy_phase2'.", file=sys.stderr); return None

    phase2_questions_config_list = config_params_dict.get("phase2_questions", [])
    p2_question_details_s_i_map = {q.get("id"): {"is_correct": q.get("is_correct")} for q in phase2_questions_config_list if q.get("id") and q.get("is_correct") is not None}
    if not p2_question_details_s_i_map: print("Error: S_i map empty.", file=sys.stderr); return None
    if not all(trial.get("question_id") in p2_question_details_s_i_map for trial in phase2_trials_list):
        print("Fatal: S_i data missing for some P2 trials. Analysis halted.", file=sys.stderr); return None
    print(f"S_i map successfully built for {len(p2_question_details_s_i_map)} P2 questions.", file=sys.stderr)

    capabilities_file_path = config_params_dict.get("capabilities_file")
    capabilities_auroc, cap_auroc_n_qs = calculate_capabilities_auroc(capabilities_file_path)
    print(f"\nCapabilities AUROC (from {capabilities_file_path}): {capabilities_auroc:.4f} (N={cap_auroc_n_qs})" if capabilities_auroc is not None else "\nCapabilities AUROC: Not calc.")

    print("\nCalculating game metrics point estimates...", file=sys.stderr)
    point_estimates = calculate_metrics_from_data(
        phase2_trials_list, subject_true_capability,
        teammate_p2_target_accuracy, p2_question_details_s_i_map
    )
    if not point_estimates: print("Could not calculate game metrics.", file=sys.stderr); return None
    print("\nPoint Estimates (Game Performance):")
    for key, value in point_estimates.items():
        if isinstance(value, float): print(f"  {key}: {value:.4f}")
        else: print(f"  {key}: {value}")

    print(f"\nCalculating bootstrap CIs with {n_bootstraps} samples...", file=sys.stderr)
    bootstrap_cis = bootstrap_ci(
        phase2_trials_list, calculate_metrics_from_data,
        n_bootstraps=n_bootstraps, alpha=0.05,
        subject_true_capability=subject_true_capability,
        teammate_p2_target_accuracy=teammate_p2_target_accuracy,
        p2_question_details_map=p2_question_details_s_i_map
    )
    if bootstrap_cis:
        print("\nBootstrap 95% Confidence Intervals (Game Performance):")
        for metric_name, ci_bounds in bootstrap_cis.items():
            pe = point_estimates.get(metric_name)
            if pe is not None and isinstance(pe, (float, int)) and ci_bounds != (None, None):
                 print(f"  {metric_name}: [{ci_bounds[0]:.4f}, {ci_bounds[1]:.4f}] (Point Est: {pe:.4f})")
            elif pe is not None: print(f"  {metric_name}: CI not computed or N/A (Point Est: {pe})")

    final_analysis_results = {
        "source_file": os.path.basename(file_path),
        "subject_id": config_params_dict.get("subject_id", "N/A"),
        "subject_true_capability_used_for_analysis": subject_true_capability,
        "teammate_p2_target_accuracy": teammate_p2_target_accuracy,
        "capabilities_auroc_of_chosen_answer": capabilities_auroc,
        "capabilities_auroc_n_questions": cap_auroc_n_qs,
        "n_phase2_trials_total": len(phase2_trials_list),
        "game_performance_point_estimates": point_estimates,
        "game_performance_bootstrap_95_CI": bootstrap_cis if bootstrap_cis else {},
        "n_bootstraps": n_bootstraps,
        "original_game_config": { k: v for k,v in config_params_dict.items() if k not in ["results", "phase1_questions", "phase2_questions"]}
    }
    return final_analysis_results

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
                    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
                json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=default_serializer)
            print(f"\nAnalysis results saved to: {args.output_file}")
        except Exception as e: print(f"Error saving analysis results: {e}", file=sys.stderr)

    if not analysis_results: print("\nAnalysis could not be completed.", file=sys.stderr)
    else: print("\n--- Analysis Complete ---")