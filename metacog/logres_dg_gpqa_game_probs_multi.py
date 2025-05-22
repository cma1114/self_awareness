import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os
import numpy as np
from load_and_format_datasets import load_and_format_dataset
import re

LOG_FILENAME = "analysis_log_multi_logres_dg_gpqa_GAME_PROBS.txt"

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

def get_s_i_from_capabilities_map(q_id, capabilities_s_i_map):
    """
    Looks up S_i (1 if model knew it in capabilities test, 0 if not)
    """
    return capabilities_s_i_map.get(q_id)


def prepare_regression_data_for_model(game_file_paths_list,
                                      gpqa_feature_lookup,
                                      p_i_map_for_this_model): # This p_i_map is from capabilities, will be ignored in this script version
    """
    Prepares a DataFrame for a model's game file(s).
    
    Args:
        game_file_paths_list (list): List of paths to _game_data.json files.
        gpqa_feature_lookup (dict): Maps q_id to {'difficulty': score, 'domain': str, 'q_text': str}.
        p_i_map_for_this_model (dict): Maps q_id to P_i (probability) for THIS model.
                                            This map should be from the model's specific
                                            _phase1_completed.json (capabilities) file.
                                            NOTE: This parameter is kept for signature consistency but will be ignored.
                                                  Probabilities will be extracted from trial data.
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
                "trials_data": phase2_trials,
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
            break 

    all_teammate_accs = [f["teammate_accuracy_phase1_file"] for f in file_level_features_cache if f["teammate_accuracy_phase1_file"] is not None]
    create_teammate_skill_ratio_reg = len(set(all_teammate_accs)) > 1 and \
                                     subject_acc_for_ratio_calc is not None and \
                                     subject_acc_for_ratio_calc != 0

    raw_feedback_vals = [f["phase1_subject_feedback_file"] for f in file_level_features_cache]
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
        trials_from_file = file_data["trials_data"]
        teammate_acc_file = file_data["teammate_accuracy_phase1_file"]
        
        feedback_val_for_reg = conv_feedback_vals[i]
        nobio_val_for_reg = conv_nobio_vals[i]
        noeasy_val_for_reg = conv_noeasy_vals[i]
        noctr_val_for_reg = conv_noctr_vals[i]

        for trial in trials_from_file:
            q_id = trial.get("question_id")
            delegation_choice_str = trial.get("delegation_choice")
            prob_dict_trial = trial.get("probs")

            if not q_id or not delegation_choice_str:
                continue

            gpqa_features = gpqa_feature_lookup.get(q_id)
            domain = gpqa_features.get('domain', 'unknown').replace(' ', '_').lower()
            
            max_norm_prob_trial = None
            norm_prob_entropy_trial = None

            if isinstance(prob_dict_trial, dict):
                non_t_probs_values = [float(v) for k, v in prob_dict_trial.items() if k != "T" and isinstance(v, (int, float))]
                
                if non_t_probs_values:
                    sum_non_t_probs = sum(non_t_probs_values)
                    if sum_non_t_probs > 1e-9:
                        normalized_probs = [p / sum_non_t_probs for p in non_t_probs_values]
                        if normalized_probs:
                            max_norm_prob_trial = max(normalized_probs)
                            norm_prob_entropy_trial = -np.sum([p_norm * np.log2(p_norm) for p_norm in normalized_probs if p_norm > 1e-9])
            
            if not (gpqa_features and gpqa_features.get('difficulty') is not None and \
                    (max_norm_prob_trial is not None or norm_prob_entropy_trial is not None)):
                continue
                
            delegate_choice_numeric = 1 if delegation_choice_str == "Teammate" else 0
            
            trial_data_dict = {
                'q_id': q_id,
                'delegate_choice': delegate_choice_numeric,
                'human_difficulty': gpqa_features['difficulty'],
                'q_length': len(gpqa_features.get('q_text', '')),
                'domain': ("Biology" if domain == "biology" else "NonBiology"),
                'overlap_ratio': gpqa_features.get('overlap_ratio', 0),
                'avg_word_length': get_average_word_length(gpqa_features.get('q_text', '')),
                'percent_non_alphabetic_whitespace': get_percent_non_alphabetic_whitespace(gpqa_features.get('q_text', '')),
            }
            
            if max_norm_prob_trial is not None:
                trial_data_dict['max_normalized_prob'] = max_norm_prob_trial
            if norm_prob_entropy_trial is not None:
                trial_data_dict['normalized_prob_entropy'] = norm_prob_entropy_trial

            if create_teammate_skill_ratio_reg and teammate_acc_file is not None:
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
    gpqa_all_questions = load_and_format_dataset("GPQA") 

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
    game_logs_dir = "./delegate_game_logs/"       
    capabilities_dir = "./completed_results_gpqa/" 

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

        # Capabilities file is no longer directly used for p_i_map in this script's prepare_regression_data
        # but we might still want to load it for other reasons or keep the structure.
        # For now, we'll pass an empty dict or None for p_i_map_for_this_model as it's ignored.
        # However, the original script used it, so we'll keep the loading logic for now,
        # even if p_i_map_for_this_model is not used by prepare_regression_data_for_model.
        # This is a placeholder and can be removed if capabilities file is truly not needed at all.
        capabilities_filename = f"{model_name_part}_phase1_completed.json"
        capabilities_file_path = os.path.join(capabilities_dir, capabilities_filename)
        dummy_p_i_map = {} # Pass a dummy map as it's ignored by the modified function

        if not os.path.exists(capabilities_file_path):
            print(f"  Corresponding capabilities file not found: {capabilities_file_path}, but proceeding as trial probs are used.")
            # We can proceed if game files exist, as trial probs are primary.
        
        if not game_file_paths_list:
            print(f"  No game files found for model {model_name_part}. Skipping.")
            continue
        
        # Pass the full list of game files for the current model
        df_model, phase1_subject_feedback = prepare_regression_data_for_model(game_file_paths_list,
                                                                            gpqa_feature_lookup,
                                                                            dummy_p_i_map) # Pass dummy map

        if df_model is None or df_model.empty:
            print(f"  No data for regression analysis for model {model_name_part}.")
            continue

        if 'teammate_skill_ratio' in df_model.columns:
            mean_skill_ratio = df_model['teammate_skill_ratio'].mean()
            df_model['teammate_skill_ratio'] = df_model['teammate_skill_ratio'] - mean_skill_ratio       

        log_output(f"\n--- Analyzing Model: {model_name_part} ({len(game_file_paths_list)} game files, feedback={phase1_subject_feedback}) ---", print_to_console=True)
        
        if game_file_paths_list:
            first_game_log_path = game_file_paths_list[0].replace("_game_data.json", ".log")
            log_metrics_dict = extract_log_file_metrics(first_game_log_path)
            for metric, value in log_metrics_dict.items():
                log_output(f"  {metric}: {value}")
        else:
            log_output("  No game files found to extract log metrics.")

        # Determine primary capability predictor for this run
        # This script will run analyses for 'max_normalized_prob' and then 'normalized_prob_entropy'
        
        for cap_predictor_name in ['max_normalized_prob', 'normalized_prob_entropy']:
            if cap_predictor_name not in df_model.columns:
                log_output(f"\n  Skipping analyses for {cap_predictor_name} as it's not in the DataFrame for model {model_name_part}.", print_to_console=True)
                continue
            
            log_output(f"\n--- Running Regressions with: {cap_predictor_name} ---", print_to_console=True)

            try:
                log_output(f"\n  Model 1: Delegate_Choice ~ {cap_predictor_name}")
                logit_model1 = smf.logit(f'delegate_choice ~ {cap_predictor_name}', data=df_model).fit(disp=0)
                log_output(logit_model1.summary())

                log_output("\n  Model 2: Delegate_Choice ~ human_difficulty") # Stays the same
                logit_model2 = smf.logit('delegate_choice ~ human_difficulty', data=df_model).fit(disp=0)
                log_output(logit_model2.summary())

                log_output(f"\n  Model 3: Delegate_Choice ~ {cap_predictor_name} + human_difficulty")
                logit_model3 = smf.logit(f'delegate_choice ~ {cap_predictor_name} + human_difficulty', data=df_model).fit(disp=0)
                log_output(logit_model3.summary())
                
                if df_model['domain'].nunique() > 1 and len(df_model) > 20 : 
                    base_model_terms = [
                        cap_predictor_name,
                        'human_difficulty',
                        'q_length',
                        'C(domain)', # Explicitly treat domain as categorical for Model 4
                        'overlap_ratio',
                        'avg_word_length',
                        'percent_non_alphabetic_whitespace'
                    ]
                    
                    conditional_regressors_to_check = [
                        'teammate_skill_ratio',
                        'phase1_subject_feedback',
                        'nobio',
                        'noeasy',
                        'noctr'
                    ]
                    
                    final_model_terms = list(base_model_terms) 
                    for regressor in conditional_regressors_to_check:
                        if regressor in df_model.columns:
                            final_model_terms.append(regressor)
                            ###final_model_terms.append(f"{cap_predictor_name}:{regressor}") 
                    
                    has_duplicate_qids = df_model['q_id'].duplicated().any()
                    fit_kwargs = {'disp': 0}

                    if has_duplicate_qids:
                        fit_kwargs['cov_type'] = 'cluster'
                        fit_kwargs['cov_kwds'] = {'groups': df_model['q_id']}
                        log_output(f"    Model 4 ({cap_predictor_name}): Using clustered standard errors by q_id due to duplicate question IDs.")
                            
                    model_def_str = 'delegate_choice ~ ' + ' + '.join(final_model_terms) 
                    
                    log_output(f"\n  Model 4 ({cap_predictor_name}): {model_def_str}")
                    try:
                        logit_model4 = smf.logit(model_def_str, data=df_model).fit(**fit_kwargs)
                        log_output(logit_model4.summary())
                        
                        if cap_predictor_name in logit_model4.params:
                            coef_cap = logit_model4.params.get(cap_predictor_name)
                            pval_cap = logit_model4.pvalues.get(cap_predictor_name)
                            log_output(f"\n--- Coefficient for {cap_predictor_name} on Delegation (Adjusted) ---")
                            log_output(f"Coefficient for {cap_predictor_name}: {coef_cap:.4f}")
                            log_output(f"P-value for {cap_predictor_name}: {pval_cap:.4g}")
                        else:
                            log_output(f"{cap_predictor_name} not in the final Model 4 terms or parameters.")

                    except Exception as e_full:
                        log_output(f"    Could not fit full model 4 ({cap_predictor_name}): {e_full}")
                else:
                    log_output(f"\n  Skipping Model 4 ({cap_predictor_name}) (full controls) due to insufficient domain variance or data points.", print_to_console=True)

            except Exception as e:
                print(f"  Error during logistic regression for model {model_name_part} using {cap_predictor_name}: {e}")
            
        print("-" * 40)