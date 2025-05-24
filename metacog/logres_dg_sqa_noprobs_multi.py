import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os
import numpy as np
from load_and_format_datasets import load_and_format_dataset
import re
from collections import defaultdict


LOG_FILENAME = "analysis_log_multi_logres_dg_simpleqa.txt"

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

def identify_and_handle_deterministic_categories(df_input, outcome_var, categorical_predictors, min_obs_for_determinism_check=5):
    """
    Identifies categories in specified predictors that have a deterministic outcome.
    Returns a DataFrame subset excluding these categories and a list of identified deterministic categories.

    Args:
        df_input (pd.DataFrame): The input DataFrame for regression.
        outcome_var (str): The name of the binary outcome variable (e.g., 'delegate_choice').
        categorical_predictors (list): A list of column names for categorical predictors to check.
        min_obs_for_determinism_check (int): Minimum observations a category must have
                                             to be considered for deterministic removal.
                                             Helps avoid removing categories based on very few samples.
    Returns:
        pd.DataFrame: Subset of df_input excluding rows from deterministic categories.
        dict: A dictionary where keys are predictor names and values are lists of 
              categories within that predictor found to be deterministic.
    """
    df_subset = df_input.copy()
    deterministic_categories_found = defaultdict(list)
    
    print("\n--- Identifying Deterministic Categories ---")

    for predictor_col in categorical_predictors:
        if predictor_col not in df_subset.columns:
            print(f"Warning: Predictor column '{predictor_col}' not found in DataFrame. Skipping.", file=sys.stderr)
            continue

        # Ensure outcome variable is numeric (0 or 1) for mean calculation
        if df_subset[outcome_var].dtype != 'int' and df_subset[outcome_var].dtype != 'float':
             try:
                 df_subset[outcome_var] = pd.to_numeric(df_subset[outcome_var])
             except ValueError:
                 print(f"Error: Outcome variable '{outcome_var}' could not be converted to numeric. Skipping {predictor_col}.", file=sys.stderr)
                 continue


        unique_categories = df_subset[predictor_col].unique()
        for category_val in unique_categories:
            category_subset = df_subset[df_subset[predictor_col] == category_val]
            if len(category_subset) >= min_obs_for_determinism_check:
                outcome_mean = category_subset[outcome_var].mean()
                if outcome_mean == 0.0 or outcome_mean == 1.0:
                    print(f"  Deterministic category found: '{predictor_col}' = '{category_val}' "
                          f"always results in '{outcome_var}' = {int(outcome_mean)} (N={len(category_subset)})")
                    deterministic_categories_found[predictor_col].append(category_val)
    
    # Filter out rows belonging to any identified deterministic category
    rows_to_remove_mask = pd.Series([False] * len(df_subset), index=df_subset.index)
    for predictor_col, categories in deterministic_categories_found.items():
        if categories: # If any deterministic categories were found for this predictor
            rows_to_remove_mask = rows_to_remove_mask | df_subset[predictor_col].isin(categories)
            
    df_final_subset = df_subset[~rows_to_remove_mask]
    
    if len(df_final_subset) < len(df_input):
        print(f"\nRemoved {len(df_input) - len(df_final_subset)} rows belonging to deterministic categories.")
        print(f"Remaining observations for regression: {len(df_final_subset)}")
    else:
        print("No deterministic categories meeting criteria found or removed.")
        
    return df_final_subset, deterministic_categories_found

def prepare_regression_data_for_model(game_file_paths_list,
                                      sqa_feature_lookup,
                                      capabilities_s_i_map_for_model):
    """
    Prepares a DataFrame for a model's game file(s), assuming all files in the list
    belong to the same phase1_subject_feedback group.
    
    Args:
        game_file_paths_list (list): List of paths to _game_data.json files.
        sqa_feature_lookup (dict): Maps q_id to {'difficulty': score, 'domain': str, 'q_text': str}.
        capabilities_s_i_map_for_model (dict): Maps q_id to S_i (0 or 1) for THIS model.
                                            This map should be from the model's specific
                                            _phase1_completed.json (capabilities) file.
    Returns:
        pandas.DataFrame or None
    """
    all_regression_data_for_model = []
    
    file_level_features_cache = []

    if not game_file_paths_list:
        return None

    # First pass: Load data, extract per-file features, and cache trial data
    for game_file_path in game_file_paths_list:
        try:
            with open(game_file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
        except Exception as e:
            print(f"Error loading game file {game_file_path}: {e}")
            continue

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
        print(f"No valid game data found in the provided files.")
        return None

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

    # phase1_subject_feedback is handled by splitting files before calling this function,
    # so no regressor for it is created here.
    
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
        nobio_val_for_reg = conv_nobio_vals[i]
        noeasy_val_for_reg = conv_noeasy_vals[i]
        noctr_val_for_reg = conv_noctr_vals[i]

        for trial in trials:
            q_id = trial.get("question_id")
            delegation_choice_str = trial.get("delegation_choice")
            if delegation_choice_str == "Self":
                subject_answer = trial.get("subject_answer")
                # if subject_answer contains the word DELEGATE or T separated by word boundaries set to "Teammate"
                if subject_answer and re.search(r'\bDELEGATE\b|\bT\b', subject_answer, re.IGNORECASE) and trial.get("evaluation_method").startswith("llm_plurality"):
                    judg_dict = trial.get("judgments")
                    #if all judgment say "NOT ATTEMPTED" set to "Teammate"
                    if judg_dict and all(j == "NOT ATTEMPTED" for j in judg_dict.values()):
                        delegation_choice_str = "Teammate"
            
            if not q_id or not delegation_choice_str:
                print(f"Skipping trial due to missing q_id or delegation choice: {trial}")
                continue

            sqa_features = sqa_feature_lookup.get(q_id)
            s_i_capability = capabilities_s_i_map_for_model.get(q_id)

            if sqa_features and s_i_capability is not None:
                delegate_choice_numeric = 1 if delegation_choice_str == "Teammate" else 0
                
                trial_data_dict = {
                    'q_id': q_id, # Ensure q_id is always in the trial data
                    'delegate_choice': delegate_choice_numeric,
                    's_i_capability': s_i_capability,
                    'team_correct': trial.get('team_correct', False),
                    'answer_type': sqa_features['answer_type'],
                    'q_length': np.log(len(sqa_features.get('q_text', ''))),
                    'topic': sqa_features.get('topic', ''),
                }

                if create_teammate_skill_ratio_reg and teammate_acc_file is not None:
                    # subject_acc_for_ratio_calc is already checked for None and 0
                    trial_data_dict['teammate_skill_ratio'] = teammate_acc_file / subject_acc_for_ratio_calc
                
                # phase1_subject_feedback regressor is not added here
                if create_nobio_reg:
                    trial_data_dict['nobio'] = nobio_val_for_reg
                if create_noeasy_reg:
                    trial_data_dict['noeasy'] = noeasy_val_for_reg
                if create_noctr_reg:
                    trial_data_dict['noctr'] = noctr_val_for_reg
                
                all_regression_data_for_model.append(trial_data_dict)
    
    if not all_regression_data_for_model:
        print(f"No valid trials found in the provided game data files.")
        return None
    return pd.DataFrame(all_regression_data_for_model)

# --- Main Analysis Logic ---
if __name__ == "__main__":
    # 1. Load sqa data once for features (difficulty, domain, question text for length)
    print("Loading main sqa dataset for features...")
    sqa_all_questions = load_and_format_dataset("SimpleQA") # This should have id, Question, high_level_domain, difficulty_score

    sqa_feature_lookup = {
        item['id']: {
            'answer_type': item.get('answer_type', 0),
            'topic': item['topic'],
            'q_text': item['question']
        } for item in sqa_all_questions
    }
    print(f"sqa feature lookup created with {len(sqa_feature_lookup)} entries.")


    # 2. Specify directories
    game_logs_dir = "./delegate_game_logs/"       # Where your _game_data.json files are
    capabilities_dir = "./compiled_results_sqa/" # Where your _phase1_completed.json files are

    if not os.path.isdir(game_logs_dir) or not os.path.isdir(capabilities_dir):
        print(f"Error: Ensure directories exist: {game_logs_dir}, {capabilities_dir}")
        exit()

    # 3. Group game log files by model
    model_game_files = {}
    for game_filename in sorted(os.listdir(game_logs_dir)):
        if game_filename.endswith("_game_data_evaluated.json") and "_SimpleQA_" in game_filename:
            model_name_part = game_filename.split("_SimpleQA_")[0]
            if model_name_part not in model_game_files:
                model_game_files[model_name_part] = []
            model_game_files[model_name_part].append(os.path.join(game_logs_dir, game_filename))

    # 4. Iterate through models
    for model_name_part, game_file_paths_list_for_model in model_game_files.items():
        print(f"\nProcessing model: {model_name_part} (total {len(game_file_paths_list_for_model)} game files)")

        files_feedback_true = []
        files_feedback_false_or_none = []

        if not game_file_paths_list_for_model:
            print(f"  No game files found for model {model_name_part} before feedback splitting. Skipping.")
            continue

        for game_file_path_for_feedback_check in game_file_paths_list_for_model:
            try:
                with open(game_file_path_for_feedback_check, 'r', encoding='utf-8') as f_check:
                    game_data_check = json.load(f_check)
                current_file_feedback_status = game_data_check.get("feedback_config", {}).get("phase1_subject_feedback")
                if current_file_feedback_status is True:
                    files_feedback_true.append(game_file_path_for_feedback_check)
                else: # Catches False and None
                    files_feedback_false_or_none.append(game_file_path_for_feedback_check)
            except Exception as e:
                print(f"  Error reading {game_file_path_for_feedback_check} for feedback status: {e}. Skipping this file for grouping.")
                continue
        
        feedback_groups_to_process = []
        if files_feedback_true:
            feedback_groups_to_process.append(("Feedback_True", files_feedback_true))
        if files_feedback_false_or_none:
            feedback_groups_to_process.append(("Feedback_False", files_feedback_false_or_none))

        if not feedback_groups_to_process:
            print(f"  No game files to process for model {model_name_part} after attempting to split by feedback. Skipping model.")
            continue
            
        for feedback_type_str, files_for_current_feedback_type in feedback_groups_to_process:
            print(f"  Processing for {feedback_type_str} ({len(files_for_current_feedback_type)} files)")

            files_redacted = []
            files_non_redacted = []
            for game_file_path_for_redaction_check in files_for_current_feedback_type:
                if "_redacted_" in os.path.basename(game_file_path_for_redaction_check):
                    files_redacted.append(game_file_path_for_redaction_check)
                else:
                    files_non_redacted.append(game_file_path_for_redaction_check)

            redaction_groups_to_process = []
            if files_redacted:
                redaction_groups_to_process.append(("Redacted", files_redacted))
            if files_non_redacted:
                redaction_groups_to_process.append(("Non_Redacted", files_non_redacted))

            if not redaction_groups_to_process:
                print(f"    No game files to process for model {model_name_part}, feedback {feedback_type_str} after attempting to split by redaction. Skipping this feedback group.")
                continue

            for redaction_type_str, files_for_current_redaction_type in redaction_groups_to_process:
                print(f"    Processing for {redaction_type_str} ({len(files_for_current_redaction_type)} files)")

                subj_acc_override_pattern = re.compile(r"_subj\d+(\.\d+)?_")
                files_with_subj_override = []
                files_without_subj_override = []

                for game_file_path_for_subj_acc_check in files_for_current_redaction_type:
                    if subj_acc_override_pattern.search(os.path.basename(game_file_path_for_subj_acc_check)):
                        files_with_subj_override.append(game_file_path_for_subj_acc_check)
                    else:
                        files_without_subj_override.append(game_file_path_for_subj_acc_check)
                
                subj_acc_override_groups_to_process = []
                if files_with_subj_override:
                    subj_acc_override_groups_to_process.append(("SubjAccOverride", files_with_subj_override))
                if files_without_subj_override:
                    subj_acc_override_groups_to_process.append(("NoSubjAccOverride", files_without_subj_override))

                if not subj_acc_override_groups_to_process:
                    print(f"      No game files to process for model {model_name_part}, feedback {feedback_type_str}, redaction {redaction_type_str} after attempting to split by subject accuracy override. Skipping this redaction group.")
                    continue

                for subj_acc_type_str, files_for_current_subj_acc_type in subj_acc_override_groups_to_process:
                    print(f"      Processing for {subj_acc_type_str} ({len(files_for_current_subj_acc_type)} files)")

                    files_randomized = []
                    files_not_randomized = []
                    for game_file_path_for_randomized_check in files_for_current_subj_acc_type:
                        if "_randomized_" in os.path.basename(game_file_path_for_randomized_check):
                            files_randomized.append(game_file_path_for_randomized_check)
                        else:
                            files_not_randomized.append(game_file_path_for_randomized_check)

                    randomized_groups_to_process = []
                    if files_randomized:
                        randomized_groups_to_process.append(("Randomized", files_randomized))
                    if files_not_randomized:
                        randomized_groups_to_process.append(("NotRandomized", files_not_randomized))
                    
                    if not randomized_groups_to_process:
                        print(f"        No game files to process for model {model_name_part}, feedback {feedback_type_str}, redaction {redaction_type_str}, subj_acc {subj_acc_type_str} after attempting to split by randomized. Skipping this subj_acc group.")
                        continue

                    for randomized_type_str, files_for_current_randomized_type in randomized_groups_to_process:
                        print(f"        Processing for {randomized_type_str} ({len(files_for_current_randomized_type)} files)")

                        files_nohistory = []
                        files_with_history = [] # Renamed from files_not_nohistory for clarity
                        for game_file_path_for_nohistory_check in files_for_current_randomized_type:
                            if "_nohistory_" in os.path.basename(game_file_path_for_nohistory_check):
                                files_nohistory.append(game_file_path_for_nohistory_check)
                            else:
                                files_with_history.append(game_file_path_for_nohistory_check)
                        
                        nohistory_groups_to_process = []
                        if files_nohistory:
                            nohistory_groups_to_process.append(("NoHistory", files_nohistory))
                        if files_with_history:
                            nohistory_groups_to_process.append(("WithHistory", files_with_history))

                        if not nohistory_groups_to_process:
                            print(f"          No game files to process for model {model_name_part}, feedback {feedback_type_str}, redaction {redaction_type_str}, subj_acc {subj_acc_type_str}, randomized {randomized_type_str} after attempting to split by nohistory. Skipping this randomized group.")
                            continue
                            
                        for nohistory_type_str, files_for_current_nohistory_type in nohistory_groups_to_process:
                            print(f"          Processing for {nohistory_type_str} ({len(files_for_current_nohistory_type)} files)")

                            files_summary = []
                            files_not_summary = []
                            for game_file_path_for_summary_check in files_for_current_nohistory_type:
                                if "_summary_" in os.path.basename(game_file_path_for_summary_check):
                                    files_summary.append(game_file_path_for_summary_check)
                                else:
                                    files_not_summary.append(game_file_path_for_summary_check)
                            
                            summary_groups_to_process = []
                            if files_summary:
                                summary_groups_to_process.append(("Summary", files_summary))
                            if files_not_summary:
                                summary_groups_to_process.append(("NoSummary", files_not_summary))

                            if not summary_groups_to_process:
                                print(f"            No game files to process for model {model_name_part}, feedback {feedback_type_str}, redaction {redaction_type_str}, subj_acc {subj_acc_type_str}, randomized {randomized_type_str}, nohistory {nohistory_type_str} after attempting to split by summary. Skipping this nohistory group.")
                                continue

                            for summary_type_str, files_from_summary_loop in summary_groups_to_process: # Renamed var
                                print(f"            Processing for {summary_type_str} ({len(files_from_summary_loop)} files)")

                                # NEW FILTERED LOOP STARTS HERE
                                files_filtered = []
                                files_not_filtered = []
                                for game_file_path_for_filtered_check in files_from_summary_loop: # Input from summary loop
                                    if "_filtered_" in os.path.basename(game_file_path_for_filtered_check):
                                        files_filtered.append(game_file_path_for_filtered_check)
                                    else:
                                        files_not_filtered.append(game_file_path_for_filtered_check)
                                
                                filtered_groups_to_process = []
                                if files_filtered:
                                    filtered_groups_to_process.append(("Filtered", files_filtered))
                                if files_not_filtered:
                                    filtered_groups_to_process.append(("NotFiltered", files_not_filtered))

                                if not filtered_groups_to_process:
                                    print(f"                No game files to process for model {model_name_part}, feedback {feedback_type_str}, redaction {redaction_type_str}, subj_acc {subj_acc_type_str}, randomized {randomized_type_str}, nohistory {nohistory_type_str}, summary {summary_type_str} after attempting to split by filtered. Skipping this summary group.")
                                    continue # Skips to the next summary_type_str

                                for filtered_type_str, current_game_files_for_analysis in filtered_groups_to_process: # current_game_files_for_analysis is now the list for this specific filtered_type
                                    print(f"                Processing for {filtered_type_str} ({len(current_game_files_for_analysis)} files)")
                                    # The subsequent SEARCH/REPLACE block will handle indenting the original content (from line 437)
                                    # and placing it here, under this new 'filtered_type_str' loop.
                                    # Derive capabilities filename
                                    capabilities_filename = f"{model_name_part}_phase1_compiled.json"
                                    capabilities_file_path = os.path.join(capabilities_dir, capabilities_filename)

                                    if not os.path.exists(capabilities_file_path):
                                        print(f"                Corresponding capabilities file not found: {capabilities_file_path}. Skipping model for this full group.")
                                        continue

                                    # Load S_i data for this specific model from its capabilities file
                                    s_i_map_for_this_model = {}
                                    try:
                                        with open(capabilities_file_path, 'r', encoding='utf-8') as f_cap:
                                            cap_data = json.load(f_cap)
                                        for q_id, res_info in cap_data.get("results", {}).items():
                                            if res_info.get("is_correct") is not None:
                                                s_i_map_for_this_model[q_id] = 1 if res_info["is_correct"] else 0
                                    except Exception as e:
                                        print(f"                Error loading capabilities file {capabilities_file_path}: {e}. Skipping model for this full group.")
                                        continue
                                    
                                    if not s_i_map_for_this_model:
                                        print(f"                No S_i data loaded from {capabilities_file_path}. Skipping model for this full group.")
                                        continue

                                    if not current_game_files_for_analysis:
                                        print(f"                No game files found for model {model_name_part}, feedback {feedback_type_str}, redaction {redaction_type_str}, subj_acc {subj_acc_type_str}, randomized {randomized_type_str}, nohistory {nohistory_type_str}, summary {summary_type_str}, filtered {filtered_type_str}. Skipping this group.")
                                        continue
                                    
                                    # Pass the filtered list of game files for the current model and all conditions
                                    df_model = prepare_regression_data_for_model(current_game_files_for_analysis,
                                                                                                        sqa_feature_lookup,
                                                                                                        s_i_map_for_this_model)

                                    if df_model is None or df_model.empty:
                                        print(f"                No data for regression analysis for model {model_name_part}, feedback {feedback_type_str}, redaction {redaction_type_str}, subj_acc {subj_acc_type_str}, randomized {randomized_type_str}, nohistory {nohistory_type_str}, summary {summary_type_str}, filtered {filtered_type_str}.")
                                        continue

                                    if 'teammate_skill_ratio' in df_model.columns:
                                        mean_skill_ratio = df_model['teammate_skill_ratio'].mean()
                                        df_model['teammate_skill_ratio'] = df_model['teammate_skill_ratio'] - mean_skill_ratio

                                    log_output(f"\n--- Analyzing Model: {model_name_part} ({feedback_type_str}, {redaction_type_str}, {subj_acc_type_str}, {randomized_type_str}, {nohistory_type_str}, {summary_type_str}, {filtered_type_str}, {len(current_game_files_for_analysis)} game files) ---", print_to_console=True)
                                    log_output(f"              Game files for analysis: {current_game_files_for_analysis}\n")
                                    
                                    # Use the log file from the first game data file in the list for log metrics
                                    if current_game_files_for_analysis:
                                        first_game_log_path = current_game_files_for_analysis[0].replace("_game_data_evaluated.json", ".log")
                                        log_metrics_dict = extract_log_file_metrics(first_game_log_path)
                                        for metric, value in log_metrics_dict.items():
                                            log_output(f"                  {metric}: {value}")
                                    else:
                                        log_output(f"                  No game files found to extract log metrics for {model_name_part}, {feedback_type_str}, {redaction_type_str}, {subj_acc_type_str}, {randomized_type_str}, {nohistory_type_str}, {summary_type_str}, {filtered_type_str}.")


                                    # Run Logistic Regressions
                                    try:
                                        log_output(f"df_model['delegate_choice'].value_counts()= {df_model['delegate_choice'].value_counts()}\n")
                                        cross_tab = pd.crosstab(df_model['delegate_choice'], df_model['s_i_capability'])
                                        log_output(f"Cross-tabulation of delegate_choice vs. s_i_capability:\n{cross_tab}\n")
                                        prob_delegating_Si0 = df_model.loc[df_model['s_i_capability'] == 0, 'delegate_choice'].mean()
                                        log_output(f"Probability of delegating when s_i_capability is 0: {prob_delegating_Si0:.4f}")
                                        prob_delegating_Si1 = df_model.loc[df_model['s_i_capability'] == 1, 'delegate_choice'].mean()
                                        log_output(f"Probability of delegating when s_i_capability is 1: {prob_delegating_Si1:.4f}")
                                        cross_tab = pd.crosstab(df_model['delegate_choice'], df_model['team_correct'])
                                        log_output(f"Cross-tabulation of delegate_choice vs. team_correct:\n{cross_tab}\n")
                                        cross_tab = pd.crosstab(df_model.loc[df_model['delegate_choice'] == 0, 's_i_capability'],df_model.loc[df_model['delegate_choice'] == 0, 'team_correct'])
                                        log_output(f"Cross-tabulation of s_i_capability vs. self_correct:\n{cross_tab}\n")
                                        log_output("\n                  Model 1: Delegate_Choice ~ S_i_capability")
                                        logit_model1 = smf.logit('delegate_choice ~ s_i_capability', data=df_model).fit(disp=0)
                                        log_output(logit_model1.summary())

                                        # Optional: Full model with controls like q_length and domain
                                        # Ensure domain has enough categories and data points
                                        if len(df_model) > 20 : # Heuristic checks
                                            min_obs_per_category=int(len(df_model)/15) + 1
                                            topic_counts = df_model['topic'].value_counts()
                                            rare_topics = topic_counts[topic_counts < min_obs_per_category].index.tolist()

                                            if rare_topics: # Only create new column if there are rare topics
                                                df_model['topic_grouped'] = df_model['topic'].apply(lambda x: 'Misc' if x in rare_topics else x)
                                                # Recalculate count with 'Misc' included
                                                grouped_counts = df_model['topic_grouped'].value_counts()
                                                # If 'Misc' still has fewer than threshold observations, merge it into 'Other'
                                                if grouped_counts.get('Misc', 0) < min_obs_per_category:
                                                    df_model['topic_grouped'] = df_model['topic_grouped'].apply(lambda x: 'Other' if x == 'Misc' else x)
                                                    log_output(f"                  Grouped rare topics into 'Other': {rare_topics}")
                                                else:
                                                    log_output(f"                  Grouped rare topics into 'Misc': {rare_topics}")
                                                topic_column_for_formula = 'topic_grouped'
                                            else:
                                                df_model['topic_grouped'] = df_model['topic'] # No grouping needed, use original
                                                topic_column_for_formula = 'topic'

                                            ans_type_counts = df_model['answer_type'].value_counts()
                                            rare_ans_types = ans_type_counts[ans_type_counts < min_obs_per_category].index.tolist()

                                            if rare_ans_types:
                                                df_model['answer_type_grouped'] = df_model['answer_type'].apply(lambda x: 'Misc' if x in rare_ans_types else x)
                                                # Recalculate count with 'Misc' included
                                                grouped_counts = df_model['answer_type_grouped'].value_counts()

                                                # If 'Misc' still has fewer than threshold observations, merge it into 'Other'
                                                if grouped_counts.get('Misc', 0) < min_obs_per_category:
                                                    df_model['answer_type_grouped'] = df_model['answer_type_grouped'].apply(lambda x: 'Other' if x == 'Misc' else x)
                                                    log_output(f"                  Grouped rare answer types into 'Other': {rare_topics}")
                                                else:
                                                    log_output(f"                  Grouped rare answer types into 'Misc': {rare_topics}")
                                                ans_type_column_for_formula = 'answer_type_grouped'
                                            else:
                                                df_model['answer_type_grouped'] = df_model['answer_type'] # No grouping needed
                                                ans_type_column_for_formula = 'answer_type'
                                            base_model_terms = [
                                                's_i_capability',
                                                f'C({topic_column_for_formula})',
                                                f'C({ans_type_column_for_formula})',
                                                'q_length',
                                            ]
                                            log_output(f"                  Topic Grouped Counts:\n {df_model['topic'].value_counts()}")
                                            log_output(f"                  Answer Type Grouped Counts:\n {df_model['answer_type'].value_counts()}")
                                            cross_tab = pd.crosstab(df_model['topic_grouped'], df_model['answer_type_grouped'])
                                            log_output("\n                  Cross-tabulation of Topic Grouped vs. Answer Type Grouped:")
                                            log_output(cross_tab)
                                            log_output("                  Capability by topic:")
                                            log_output(df_model.groupby('topic_grouped')['s_i_capability'].agg(['mean', 'std', 'count']))

                                            log_output("\n                  Capability by answer type:")
                                            log_output(df_model.groupby('answer_type_grouped')['s_i_capability'].agg(['mean', 'std', 'count']))

                                            log_output("                  Q length by capability:")
                                            log_output(df_model.groupby('s_i_capability')['q_length'].agg(['mean', 'std', 'count']))

                                            log_output(f"{df_model.groupby('topic_grouped')['delegate_choice'].value_counts(normalize=True)}\n")
                                            log_output(f"{df_model.groupby('answer_type_grouped')['delegate_choice'].value_counts(normalize=True)}\n")
                                            proportions = df_model.groupby(['topic_grouped', 'answer_type_grouped'])['delegate_choice'].value_counts(normalize=True).unstack(fill_value=0)
                                            total_counts = df_model.groupby(['topic_grouped', 'answer_type_grouped'])['delegate_choice'].count()
                                            proportions['total_count'] = total_counts
                                            log_output("                  topic+answer_type By delegate:\n")
                                            log_output(proportions)

                                            # Check for newly added conditional regressors in df_model
                                            conditional_regressors_to_check = [
                                                'teammate_skill_ratio',
                                                # 'phase1_subject_feedback', # Removed as data is now split by this
                                                'nobio',
                                                'noeasy',
                                                'noctr'
                                            ]
                                            
                                            final_model_terms = list(base_model_terms) # Start with a copy
                                            for regressor in conditional_regressors_to_check:
                                                if regressor in df_model.columns:
                                                    final_model_terms.append(regressor)
                                                    # Add interaction term with s_i_capability
                                                    if regressor=='teammate_skill_ratio': final_model_terms.append(f"s_i_capability:{regressor}")
                                            
                                            has_duplicate_qids = df_model['q_id'].duplicated().any()
                                            fit_kwargs = {'disp': 0}
                                            if has_duplicate_qids:
                                                # Only set up clustering, do NOT add C(q_id) to model terms
                                                fit_kwargs['cov_type'] = 'cluster'
                                                fit_kwargs['cov_kwds'] = {'groups': df_model['q_id']}
                                                log_output("                    Model 4: Using clustered standard errors by q_id due to duplicate question IDs.")
                                            # else: fit_kwargs remains {'disp': 0}
                                                    
                                            #final_model_terms.append(f"C({topic_column_for_formula}):C({ans_type_column_for_formula})")
                                            model_def_str = 'delegate_choice ~ ' + ' + '.join(final_model_terms) # final_model_terms does not include C(q_id)
                                            
                                            log_output(f"\n                  Model 4: {model_def_str}")
                                            try:
                                                logit_model4 = smf.logit(model_def_str, data=df_model).fit(**fit_kwargs)
                                                log_output(logit_model4.summary())
                                                
                                                # Check if s_i_capability is in the model before trying to get its stats
                                                if 's_i_capability' in logit_model4.params:
                                                    coef_s_i = logit_model4.params.get('s_i_capability')
                                                    pval_s_i = logit_model4.pvalues.get('s_i_capability')
                                                    conf_int_s_i_log_odds = logit_model4.conf_int().loc['s_i_capability']
                                                    odds_ratio_delegate_Si0_vs_Si1 = np.exp(-coef_s_i)
                                                    ci_lower_or = np.exp(-conf_int_s_i_log_odds.iloc[1])
                                                    ci_upper_or = np.exp(-conf_int_s_i_log_odds.iloc[0])
                                                    log_output(f"\n                  --- Odds Ratio for S_i_capability on Delegation (Adjusted) ---")
                                                    log_output(f"                  P-value for s_i_capability: {pval_s_i:.4g}")
                                                    log_output(f"                  Odds Ratio (Delegating when S_i=0 vs. S_i=1): {odds_ratio_delegate_Si0_vs_Si1:.4f}")
                                                    log_output(f"                  95% CI for this Odds Ratio: [{ci_lower_or:.4f}, {ci_upper_or:.4f}]")
                                                else:
                                                    log_output(f"                  s_i_capability not in the final Model 4 terms or parameters.")

                                            except Exception as e_full:
                                                log_output(f"                    Could not fit full model: {e_full}")
                                            
                                            if 's_i_capability:teammate_skill_ratio' in final_model_terms:
                                                final_model_terms.remove('s_i_capability:teammate_skill_ratio')
                                                model_def_str = 'delegate_choice ~ ' + ' + '.join(final_model_terms) # final_model_terms does not include C(q_id)
                                                log_output(f"\n                  Model 5: {model_def_str}")
                                                try:
                                                    logit_model5 = smf.logit(model_def_str, data=df_model).fit(**fit_kwargs)
                                                    log_output(logit_model5.summary())
                                                except Exception as e_full:
                                                    log_output(f"                    Could not fit full model: {e_full}")

                                            categorical_cols_to_check = ['topic_grouped', 'answer_type_grouped']
                                            df_subset_for_regression, identified_deterministic_cats = identify_and_handle_deterministic_categories(df_model, 'delegate_choice', categorical_cols_to_check, min_obs_for_determinism_check=min_obs_per_category)
                                            if identified_deterministic_cats:
                                                has_duplicate_qids = df_subset_for_regression['q_id'].duplicated().any()
                                                fit_kwargs = {'disp': 0}
                                                if has_duplicate_qids:
                                                    # Only set up clustering, do NOT add C(q_id) to model terms
                                                    fit_kwargs['cov_type'] = 'cluster'
                                                    fit_kwargs['cov_kwds'] = {'groups': df_subset_for_regression['q_id']}

                                                model_def_str = 'delegate_choice ~ ' + ' + '.join(final_model_terms) # final_model_terms does not include C(q_id)
                                                log_output(f"\n                  Model 6: {model_def_str} after removing {identified_deterministic_cats}")
                                                try:
                                                    logit_model6 = smf.logit(model_def_str, data=df_subset_for_regression).fit(**fit_kwargs)
                                                    log_output(logit_model6.summary())
                                                except Exception as e_full:
                                                    log_output(f"                    Could not fit full model: {e_full}")

                                            final_model_terms.remove('s_i_capability')
                                            model_def_str = 'delegate_choice ~ ' + ' + '.join(final_model_terms) # final_model_terms does not include C(q_id)
                                            log_output(f"\n                  Model 7: {model_def_str}")
                                            try:
                                                logit_model7 = smf.logit(model_def_str, data=df_model).fit(**fit_kwargs)
                                                log_output(logit_model7.summary())
                                            except Exception as e_full:
                                                log_output(f"                    Could not fit full model: {e_full}")

                                        else:
                                            log_output("\n                  Skipping Model 4 (full controls) due to insufficient domain variance or data points.", print_to_console=True)

                                    except Exception as e:
                                        print(f"                  Error during logistic regression for model {model_name_part}, feedback {feedback_type_str}, redaction {redaction_type_str}, subj_acc {subj_acc_type_str}, randomized {randomized_type_str}, nohistory {nohistory_type_str}, summary {summary_type_str}, filtered {filtered_type_str}: {e}")
                                    
                                    print("-" * 40) # This should be per filtered group