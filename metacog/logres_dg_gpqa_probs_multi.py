import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os
import numpy as np
from load_and_format_datasets import load_and_format_dataset
import re

LOG_FILENAME = "analysis_log_multi_logres_dg_gpqa_PROBS.txt"

def log_output(message_string, print_to_console=False):
    with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
        f.write(str(message_string) + "\n")
    if print_to_console:
        print(message_string)

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
    """Reads a .log file and extracts specified metrics."""
    extracted_log_metrics = {key: "Not found" for key in LOG_METRICS_TO_EXTRACT}
    try:
        with open(log_filepath, 'r') as f:
            for line in f:
                for metric_name, pattern in LOG_METRIC_PATTERNS.items():
                    match = pattern.match(line)
                    if match:
                        extracted_log_metrics[metric_name] = match.group(1).strip()
                        # Optimization: if all log metrics found, can break early
                        # This requires checking if all "Not found" have been replaced
                        if all(val != "Not found" for val in extracted_log_metrics.values()):
                            return extracted_log_metrics
    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_filepath}")
        # Return dict with "Not found" for all log metrics
    except Exception as e:
        print(f"An error occurred while reading log file {log_filepath}: {e}")
        # Return dict with "Not found" for all log metrics
    return extracted_log_metrics

def get_average_word_length(question_text):
    """Calculates the average word length in the question."""
    if not isinstance(question_text, str):
        return 0
    words = re.findall(r'\b\w+\b', question_text.lower()) # Find all words
    if not words:
        return 0
    total_word_length = sum(len(word) for word in words)
    return total_word_length / len(words)

def get_percent_non_alphabetic_whitespace(question_text):
    """
    Calculates the percentage of characters in the question text that are
    not alphabetic, not numeric, and not whitespace.
    """
    if not isinstance(question_text, str) or len(question_text) == 0:
        return 0
    
    non_alphabetic_whitespace_chars = re.findall(r'[^a-zA-Z\s]', question_text)
    return (len(non_alphabetic_whitespace_chars) / len(question_text)) * 100


def prepare_regression_data_for_model(game_file_paths_list,
                                      gpqa_feature_lookup,
                                      p_i_map_for_this_model,
                                      entropy_map_for_this_model,
                                      s_i_map_for_this_model): # Added s_i map
    """
    Prepares a DataFrame for a model's game file(s).
    
    Args:
        game_file_paths_list (list): List of paths to _game_data.json files.
        gpqa_feature_lookup (dict): Maps q_id to {'difficulty': score, 'domain': str, 'q_text': str}.
        p_i_map_for_this_model (dict): Maps q_id to P_i (prob of subject_answer) for THIS model.
        entropy_map_for_this_model (dict): Maps q_id to entropy of capabilities probs.
        s_i_map_for_this_model (dict): Maps q_id to S_i (1 if correct, 0 if incorrect in capabilities).
    Returns:
        pandas.DataFrame or None, phase1_subject_feedback
    """
    all_regression_data_for_model = []
    # This phase1_subject_feedback_overall is for the function's return value (logging)
    phase1_subject_feedback_overall_for_logging = None
    
    file_level_features_cache = []

    if not game_file_paths_list:
        return None, None

    # First pass: Load data, extract per-file features, and cache trial data
    for game_file_path in game_file_paths_list:
        try:
            with open(game_file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
        except Exception as e:
            print(f"Error loading game file {game_file_path}: {e}")
            continue

        # Set overall feedback for logging from the first successfully loaded file
        if phase1_subject_feedback_overall_for_logging is None:
            phase1_subject_feedback_overall_for_logging = game_data.get("feedback_config", {}).get("phase1_subject_feedback")

        current_teammate_acc = game_data.get("teammate_accuracy_phase1")
        current_subject_acc = game_data.get("subject_accuracy_phase1")
        current_file_feedback = game_data.get("feedback_config", {}).get("phase1_subject_feedback")
        
        filename_base = os.path.basename(game_file_path)
        file_has_nobio = "_nobio_" in filename_base
        file_has_noeasy = "_noeasy_" in filename_base
        file_has_noctr = "_noctr_" in filename_base
                
        phase2_trials = [t for t in game_data.get("results", []) if t.get('phase') == 2]
        if phase2_trials:
            file_level_features_cache.append({
                "trials": phase2_trials,
                "teammate_accuracy_phase1_file": current_teammate_acc,
                "subject_accuracy_phase1_file": current_subject_acc,
                "phase1_subject_feedback_file": current_file_feedback,
                "nobio_file": file_has_nobio,
                "noeasy_file": file_has_noeasy,
                "noctr_file": file_has_noctr
            })

    if not file_level_features_cache:
        return None, phase1_subject_feedback_overall_for_logging

    # Determine which regressors to create based on variability
    subject_acc_for_ratio_calc = None
    for f_data in file_level_features_cache:
        if f_data["subject_accuracy_phase1_file"] is not None:
            subject_acc_for_ratio_calc = f_data["subject_accuracy_phase1_file"]
            break # Assuming it's constant for the model, take the first one

    all_teammate_accs = [f["teammate_accuracy_phase1_file"] for f in file_level_features_cache if f["teammate_accuracy_phase1_file"] is not None]
    create_teammate_skill_ratio_reg = len(set(all_teammate_accs)) > 1 and \
                                     subject_acc_for_ratio_calc is not None and \
                                     subject_acc_for_ratio_calc != 0

    raw_feedback_vals = [f["phase1_subject_feedback_file"] for f in file_level_features_cache]
    # Convert to 0/1 (True -> 1, False/None -> 0) for variability check and regressor value
    conv_feedback_vals = [1 if v is True else 0 for v in raw_feedback_vals]
    create_feedback_reg = len(set(conv_feedback_vals)) > 1
    
    conv_nobio_vals = [1 if f["nobio_file"] else 0 for f in file_level_features_cache]
    create_nobio_reg = len(set(conv_nobio_vals)) > 1
    
    conv_noeasy_vals = [1 if f["noeasy_file"] else 0 for f in file_level_features_cache]
    create_noeasy_reg = len(set(conv_noeasy_vals)) > 1

    conv_noctr_vals = [1 if f["noctr_file"] else 0 for f in file_level_features_cache]
    create_noctr_reg = len(set(conv_noctr_vals)) > 1

    # Second pass: Construct regression data for all trials
    for i, file_data in enumerate(file_level_features_cache):
        trials = file_data["trials"]
        teammate_acc_file = file_data["teammate_accuracy_phase1_file"]
        
        # Use the converted 0/1 values for regressors
        feedback_val_for_reg = conv_feedback_vals[i]
        nobio_val_for_reg = conv_nobio_vals[i]
        noeasy_val_for_reg = conv_noeasy_vals[i]
        noctr_val_for_reg = conv_noctr_vals[i]

        for trial in trials:
            q_id = trial.get("question_id")
            delegation_choice_str = trial.get("delegation_choice")

            if not q_id or not delegation_choice_str:
                continue

            gpqa_features = gpqa_feature_lookup.get(q_id)
            p_i_capability = p_i_map_for_this_model.get(q_id)
            capabilities_entropy = entropy_map_for_this_model.get(q_id)
            s_i_capability = s_i_map_for_this_model.get(q_id) # Get s_i
            domain = gpqa_features.get('domain', 'unknown').replace(' ', '_').lower()

            # Ensure p_i_capability OR capabilities_entropy is present for the trial to be included.
            # s_i_capability will be carried along if present for these trials, but its absence alone won't exclude a trial
            # if probability data is available.
            if gpqa_features and gpqa_features.get('difficulty') is not None and \
               (p_i_capability is not None or capabilities_entropy is not None):
                delegate_choice_numeric = 1 if delegation_choice_str == "Teammate" else 0
                
                trial_data_dict = {
                    'q_id': q_id,
                    'delegate_choice': delegate_choice_numeric,
                    'p_i_capability': p_i_capability,
                    'capabilities_entropy': capabilities_entropy,
                    's_i_capability': s_i_capability, # Add s_i
                    'human_difficulty': gpqa_features['difficulty'],
                    'q_length': len(gpqa_features.get('q_text', '')),
                    'domain': ("Biology" if domain == "biology" else "NonBiology"),
                    'overlap_ratio': gpqa_features.get('overlap_ratio', 0),
                    'avg_word_length': get_average_word_length(gpqa_features.get('q_text', '')),
                    'percent_non_alphabetic_whitespace': get_percent_non_alphabetic_whitespace(gpqa_features.get('q_text', '')),
                }

                if create_teammate_skill_ratio_reg and teammate_acc_file is not None:
                    # subject_acc_for_ratio_calc is already checked for None and 0
                    trial_data_dict['teammate_skill_ratio'] = teammate_acc_file / subject_acc_for_ratio_calc
                
                if create_feedback_reg:
                    trial_data_dict['phase1_subject_feedback'] = feedback_val_for_reg
                if create_nobio_reg:
                    trial_data_dict['nobio'] = nobio_val_for_reg
                if create_noeasy_reg:
                    trial_data_dict['noeasy'] = noeasy_val_for_reg
                if create_noctr_reg:
                    trial_data_dict['noctr'] = noctr_val_for_reg
                
                all_regression_data_for_model.append(trial_data_dict)
    
    if not all_regression_data_for_model:
        return None, phase1_subject_feedback_overall_for_logging
    return pd.DataFrame(all_regression_data_for_model), phase1_subject_feedback_overall_for_logging

# --- Main Analysis Logic ---
if __name__ == "__main__":
    # 1. Load GPQA data once for features (difficulty, domain, question text for length)
    print("Loading main GPQA dataset for features...")
    gpqa_all_questions = load_and_format_dataset("GPQA") # This should have id, Question, high_level_domain, difficulty_score

    gpqa_feature_lookup = {
        item['id']: {
            'overlap_ratio': item.get('overlap_ratio', 0),
            'difficulty': item['difficulty_score'],
            'domain': item['high_level_domain'],
            'q_text': item['question']
        } for item in gpqa_all_questions
    }
    print(f"GPQA feature lookup created with {len(gpqa_feature_lookup)} entries.")


    # 2. Specify directories
    game_logs_dir = "./delegate_game_logs/"       # Where your _game_data.json files are
    capabilities_dir = "./completed_results_gpqa/" # Where your _phase1_completed.json files are

    if not os.path.isdir(game_logs_dir) or not os.path.isdir(capabilities_dir):
        print(f"Error: Ensure directories exist: {game_logs_dir}, {capabilities_dir}")
        exit()

    # 3. Group game log files by model
    model_game_files = {}
    for game_filename in sorted(os.listdir(game_logs_dir)):
        if game_filename.endswith("_game_data.json") and "_GPQA_" in game_filename:
            model_name_part = game_filename.split("_GPQA_")[0]
            if model_name_part not in model_game_files:
                model_game_files[model_name_part] = []
            model_game_files[model_name_part].append(os.path.join(game_logs_dir, game_filename))

    # 4. Iterate through models
    for model_name_part, game_file_paths_list in model_game_files.items():
        print(f"\nProcessing model: {model_name_part} with {len(game_file_paths_list)} game file(s)")

        # Derive capabilities filename
        capabilities_filename = f"{model_name_part}_phase1_completed.json"
        capabilities_file_path = os.path.join(capabilities_dir, capabilities_filename)

        if not os.path.exists(capabilities_file_path):
            print(f"  Corresponding capabilities file not found: {capabilities_file_path}. Skipping model.")
            continue

        # Load P_i and calculate Entropy from capabilities file
        p_i_map_for_this_model = {}
        entropy_map_for_this_model = {}
        s_i_map_for_this_model = {} # New map for S_i
        try:
            with open(capabilities_file_path, 'r', encoding='utf-8') as f_cap:
                cap_data = json.load(f_cap)
            for q_id, res_info in cap_data.get("results", {}).items():
                subject_answer = res_info.get("subject_answer")
                correct_answer = res_info.get("correct_answer") # Get correct_answer
                probs_dict = res_info.get("probs")

                # Populate p_i_map_for_this_model
                if subject_answer is not None and isinstance(probs_dict, dict):
                    prob_for_subject_answer = probs_dict.get(subject_answer)
                    if isinstance(prob_for_subject_answer, (int, float)):
                        p_i_map_for_this_model[q_id] = float(prob_for_subject_answer)
                
                # Calculate and populate entropy_map_for_this_model
                if isinstance(probs_dict, dict) and probs_dict:
                    prob_values = [float(p) for p in probs_dict.values() if isinstance(p, (int, float)) and p > 1e-9]
                    if prob_values:
                        entropy = -np.sum([p_val * np.log2(p_val) for p_val in prob_values if p_val > 1e-9])
                        entropy_map_for_this_model[q_id] = entropy
                
                # Populate s_i_map_for_this_model (using the 'is_correct' field)
                if res_info.get("is_correct") is not None:
                    s_i_map_for_this_model[q_id] = 1 if res_info["is_correct"] else 0
                # If "is_correct" is not present, s_i_map_for_this_model will not have an entry for this q_id,
                # and s_i_capability will be None for that trial, which is handled by statsmodels.

        except Exception as e:
            print(f"  Error loading or processing capabilities file {capabilities_file_path}: {e}. Skipping model.")
            continue
        
        # Check if at least one of the primary maps has data, otherwise skip
        if not p_i_map_for_this_model and not entropy_map_for_this_model and not s_i_map_for_this_model:
            print(f"  No P_i, Entropy, or S_i data loaded from {capabilities_file_path}. Skipping model.")
            continue

        # Prepare data for this model
        # The plan is to later modify prepare_regression_data_for_model to handle the whole list
        if not game_file_paths_list:
            print(f"  No game files found for model {model_name_part}. Skipping.")
            continue
        
        # Pass the full list of game files for the current model
        df_model, phase1_subject_feedback = prepare_regression_data_for_model(game_file_paths_list,
                                                                            gpqa_feature_lookup,
                                                                            p_i_map_for_this_model,
                                                                            entropy_map_for_this_model,
                                                                            s_i_map_for_this_model) # Pass s_i map

        if df_model is None or df_model.empty:
            print(f"  No data for regression analysis for model {model_name_part}.")
            continue

        if 'teammate_skill_ratio' in df_model.columns:
            mean_skill_ratio = df_model['teammate_skill_ratio'].mean()
            df_model['teammate_skill_ratio'] = df_model['teammate_skill_ratio'] - mean_skill_ratio       

        log_output(f"\n--- Analyzing Model: {model_name_part} ({len(game_file_paths_list)} game files, feedback={phase1_subject_feedback}) ---", print_to_console=True)
        
        # Use the log file from the first game data file in the list for log metrics
        if game_file_paths_list:
            first_game_log_path = game_file_paths_list[0].replace("_game_data.json", ".log")
            log_metrics_dict = extract_log_file_metrics(first_game_log_path)
            for metric, value in log_metrics_dict.items():
                log_output(f"  {metric}: {value}")
        else:
            log_output("  No game files found to extract log metrics.")


        # Run Logistic Regressions
        try:
            # --- Common Control Variables & Fit Arguments ---
            common_control_terms = [
                'q_length', 'domain', 'overlap_ratio', 'avg_word_length', 'percent_non_alphabetic_whitespace'
            ]
            conditional_regressors = [
                'teammate_skill_ratio', 'phase1_subject_feedback', 'nobio', 'noeasy', 'noctr'
            ]
            active_controls = [term for term in common_control_terms if term in df_model.columns]
            for regressor in conditional_regressors:
                if regressor in df_model.columns:
                    active_controls.append(regressor)
            
            # --- P_i Capability Model Path ---
            log_output("\n\n  --- P_i Capability Models ---")
            if 'p_i_capability' in df_model.columns and df_model['p_i_capability'].notna().any():
                # Model 1: p_i_capability alone
                log_output("\n  Model 1: Delegate_Choice ~ p_i_capability")
                try:
                    logit_m1 = smf.logit('delegate_choice ~ p_i_capability', data=df_model.dropna(subset=['p_i_capability', 'delegate_choice'])).fit(disp=0)
                    log_output(logit_m1.summary())
                except Exception as e_m1:
                    log_output(f"    Could not fit Model 1: {e_m1}")

                # Model 3: p_i_capability + human_difficulty
                log_output("\n  Model 3: Delegate_Choice ~ p_i_capability + human_difficulty")
                try:
                    logit_m3 = smf.logit('delegate_choice ~ p_i_capability + human_difficulty', data=df_model.dropna(subset=['p_i_capability', 'human_difficulty', 'delegate_choice'])).fit(disp=0)
                    log_output(logit_m3.summary())
                except Exception as e_m3:
                    log_output(f"    Could not fit Model 3: {e_m3}")

                # Model 5: P_i Full Model
                base_terms_p_i_full = ['p_i_capability', 's_i_capability', 'human_difficulty']
                final_terms_p_i_full = base_terms_p_i_full + active_controls
                
                df_model_m5_subset = df_model[['q_id', 'delegate_choice'] + final_terms_p_i_full].copy()
                df_model_m5_subset.dropna(inplace=True)

                if (len(df_model_m5_subset) > len(final_terms_p_i_full) and
                    ('domain' not in final_terms_p_i_full or df_model_m5_subset['domain'].nunique() > 1) and
                    df_model_m5_subset['s_i_capability'].notna().any()):
                    
                    model_def_p_i_full = 'delegate_choice ~ ' + ' + '.join(final_terms_p_i_full)
                    log_output(f"\n  Model 5 (P_i Full Model): {model_def_p_i_full}")
                    
                    fit_kwargs_m5 = {'disp': 0}
                    if df_model_m5_subset['q_id'].duplicated().any():
                        fit_kwargs_m5['cov_type'] = 'cluster'
                        fit_kwargs_m5['cov_kwds'] = {'groups': df_model_m5_subset['q_id']}
                        log_output("    Model 5: Using clustered standard errors by q_id.")
                    try:
                        logit_p_i_full = smf.logit(model_def_p_i_full, data=df_model_m5_subset).fit(**fit_kwargs_m5)
                        log_output(logit_p_i_full.summary())
                        if 'p_i_capability' in logit_p_i_full.params: log_output(f"    P_i Full Model - p_i_capability coef: {logit_p_i_full.params['p_i_capability']:.4f}, p-val: {logit_p_i_full.pvalues['p_i_capability']:.4g}")
                        if 's_i_capability' in logit_p_i_full.params: log_output(f"    P_i Full Model - s_i_capability coef: {logit_p_i_full.params['s_i_capability']:.4f}, p-val: {logit_p_i_full.pvalues['s_i_capability']:.4g}")
                    except Exception as e_p_i_full:
                        log_output(f"    Could not fit Model 5 (P_i Full Model): {e_p_i_full}")
                else:
                    log_output("\n  Skipping Model 5 (P_i Full Model) due to insufficient data after NaN removal for model-specific columns, missing s_i_capability, or domain variance issues.")
            else:
                log_output("\n  Skipping P_i Capability Models (1, 3, 5) as 'p_i_capability' column has no valid data or is missing.")

            # --- Capabilities Entropy Model Path ---
            log_output("\n\n  --- Capabilities Entropy Models ---")
            if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any():
                # Model 2: capabilities_entropy alone
                log_output("\n  Model 2: Delegate_Choice ~ capabilities_entropy")
                try:
                    logit_m2 = smf.logit('delegate_choice ~ capabilities_entropy', data=df_model.dropna(subset=['capabilities_entropy', 'delegate_choice'])).fit(disp=0)
                    log_output(logit_m2.summary())
                except Exception as e_m2:
                    log_output(f"    Could not fit Model 2: {e_m2}")

                # Model 4: capabilities_entropy + human_difficulty
                log_output("\n  Model 4: Delegate_Choice ~ capabilities_entropy + human_difficulty")
                try:
                    logit_m4 = smf.logit('delegate_choice ~ capabilities_entropy + human_difficulty', data=df_model.dropna(subset=['capabilities_entropy', 'human_difficulty', 'delegate_choice'])).fit(disp=0)
                    log_output(logit_m4.summary())
                except Exception as e_m4:
                    log_output(f"    Could not fit Model 4: {e_m4}")

                # Model 6: Entropy Full Model
                base_terms_entropy_full = ['capabilities_entropy', 's_i_capability', 'human_difficulty']
                final_terms_entropy_full = base_terms_entropy_full + active_controls

                df_model_m6_subset = df_model[['q_id', 'delegate_choice'] + final_terms_entropy_full].copy()
                df_model_m6_subset.dropna(inplace=True)
                
                if (len(df_model_m6_subset) > len(final_terms_entropy_full) and
                    ('domain' not in final_terms_entropy_full or df_model_m6_subset['domain'].nunique() > 1) and
                    df_model_m6_subset['s_i_capability'].notna().any()):

                    model_def_entropy_full = 'delegate_choice ~ ' + ' + '.join(final_terms_entropy_full)
                    log_output(f"\n  Model 6 (Entropy Full Model): {model_def_entropy_full}")
                    
                    fit_kwargs_m6 = {'disp': 0}
                    if df_model_m6_subset['q_id'].duplicated().any():
                        fit_kwargs_m6['cov_type'] = 'cluster'
                        fit_kwargs_m6['cov_kwds'] = {'groups': df_model_m6_subset['q_id']}
                        log_output("    Model 6: Using clustered standard errors by q_id.")
                    try:
                        logit_entropy_full = smf.logit(model_def_entropy_full, data=df_model_m6_subset).fit(**fit_kwargs_m6)
                        log_output(logit_entropy_full.summary())
                        if 'capabilities_entropy' in logit_entropy_full.params: log_output(f"    Entropy Full Model - capabilities_entropy coef: {logit_entropy_full.params['capabilities_entropy']:.4f}, p-val: {logit_entropy_full.pvalues['capabilities_entropy']:.4g}")
                        if 's_i_capability' in logit_entropy_full.params: log_output(f"    Entropy Full Model - s_i_capability coef: {logit_entropy_full.params['s_i_capability']:.4f}, p-val: {logit_entropy_full.pvalues['s_i_capability']:.4g}")
                    except Exception as e_entropy_full:
                        log_output(f"    Could not fit Model 6 (Entropy Full Model): {e_entropy_full}")
                else:
                    log_output("\n  Skipping Model 6 (Entropy Full Model) due to insufficient data after NaN removal for model-specific columns, missing s_i_capability, or domain variance issues.")
            else:
                log_output("\n  Skipping Capabilities Entropy Models (2, 4, 6) as 'capabilities_entropy' column has no valid data or is missing.")

        except Exception as e:
            print(f"  Overall error during logistic regression section for model {model_name_part}: {e}")
        
        print("-" * 40)