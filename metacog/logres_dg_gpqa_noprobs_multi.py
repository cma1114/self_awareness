import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import os
import numpy as np
from load_and_format_datasets import load_and_format_dataset
import re
from collections import defaultdict

LOG_FILENAME = "analysis_log_multi_logres_dg_gpqa.txt"

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

def identify_and_handle_deterministic_categories(df_input, outcome_var, categorical_predictors, min_obs_for_determinism_check=5):
    df_subset = df_input.copy()
    deterministic_categories_found = defaultdict(list)
    
    print("\n--- Identifying Deterministic Categories ---")

    for predictor_col in categorical_predictors:
        if predictor_col not in df_subset.columns:
            print(f"Warning: Predictor column '{predictor_col}' not found in DataFrame. Skipping.")
            continue

        if df_subset[outcome_var].dtype not in ['int', 'float', 'bool']: # Bool for 0/1
             try:
                 df_subset[outcome_var] = pd.to_numeric(df_subset[outcome_var])
             except ValueError:
                 print(f"Error: Outcome variable '{outcome_var}' could not be converted to numeric. Skipping {predictor_col}.")
                 continue

        unique_categories = df_subset[predictor_col].unique()
        for category_val in unique_categories:
            category_subset = df_subset[df_subset[predictor_col] == category_val]
            if len(category_subset) >= min_obs_for_determinism_check:
                # Ensure outcome_mean is calculated on numeric type after potential conversion
                outcome_mean = pd.to_numeric(category_subset[outcome_var], errors='coerce').mean()
                if outcome_mean == 0.0 or outcome_mean == 1.0:
                    print(f"  Deterministic category found: '{predictor_col}' = '{category_val}' "
                          f"always results in '{outcome_var}' = {int(outcome_mean)} (N={len(category_subset)})")
                    deterministic_categories_found[predictor_col].append(category_val)
    
    rows_to_remove_mask = pd.Series(False, index=df_subset.index)
    for predictor_col, categories in deterministic_categories_found.items():
        if categories:
            rows_to_remove_mask |= df_subset[predictor_col].isin(categories)
            
    df_final_subset = df_subset[~rows_to_remove_mask]
    
    if len(df_final_subset) < len(df_input):
        print(f"\nRemoved {len(df_input) - len(df_final_subset)} rows belonging to deterministic categories.")
        print(f"Remaining observations for regression: {len(df_final_subset)}")
    else:
        print("No deterministic categories meeting criteria found or removed.")
        
    return df_final_subset, deterministic_categories_found

def prepare_regression_data_for_model(game_file_paths_list,
                                      gpqa_feature_lookup,
                                      capabilities_s_i_map_for_model):
    all_regression_data_for_model = []
    file_level_features_cache = []

    if not game_file_paths_list:
        return None

    phase2_corcnt, phase2_totalcnt = 0, 0
    for game_file_path in game_file_paths_list:
        judgment_data_for_file = {}
        try:
            with open(game_file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)

            judgment_file_path = game_file_path.replace("_game_data.json", "_game_data_judgment_judge_data.json")
            if os.path.exists(judgment_file_path):
                try:
                    with open(judgment_file_path, 'r', encoding='utf-8') as jf:
                        judgment_content = json.load(jf)
                    if isinstance(judgment_content, dict) and "results" in judgment_content and isinstance(judgment_content["results"], dict):
                        for qid, q_data in judgment_content["results"].items():
                            if isinstance(q_data, dict) and "delegate" in q_data:
                                judgment_data_for_file[qid] = q_data["delegate"]
                except Exception as e_judge:
                    print(f"Error loading or parsing judgment file {judgment_file_path}: {e_judge}")
        except Exception as e:
            print(f"Error loading game file {game_file_path}: {e}")
            continue

        filename_base = os.path.basename(game_file_path)
        phase2_trials = [t for t in game_data.get("results", []) if t.get('phase') == 2]

        phase2_corcnt += sum(1 for t in phase2_trials if t["delegation_choice"]=="Self" and t["subject_correct"])
        phase2_totalcnt += sum(1 for t in phase2_trials if t["delegation_choice"]=="Self")

        if phase2_trials:
            file_level_features_cache.append({
                "trials": phase2_trials,
                "teammate_accuracy_phase1_file": game_data.get("teammate_accuracy_phase1"),
                "subject_accuracy_phase1_file": game_data.get("subject_accuracy_phase1"),
                "phase1_subject_feedback_file": game_data.get("feedback_config", {}).get("phase1_subject_feedback"),
                "summary_file": "_summary_" in filename_base,
                "nobio_file": "_nobio_" in filename_base,
                "noeasy_file": "_noeasy_" in filename_base,
                "noctr_file": "_noctr_" in filename_base,
                "judgment_data": judgment_data_for_file
            })

    if not file_level_features_cache:
        print(f"No valid game data found in the provided files.")
        return None

    subject_acc_for_ratio_calc = next((f["subject_accuracy_phase1_file"] 
                                       for f in file_level_features_cache 
                                       if f["subject_accuracy_phase1_file"] is not None), None)

    all_teammate_accs = [f["teammate_accuracy_phase1_file"] for f in file_level_features_cache if f["teammate_accuracy_phase1_file"] is not None]
    create_teammate_skill_ratio_reg = len(set(all_teammate_accs)) > 1 and \
                                     subject_acc_for_ratio_calc is not None and \
                                     subject_acc_for_ratio_calc != 0
    
    create_summary_reg = len(set(f["summary_file"] for f in file_level_features_cache)) > 1
    create_nobio_reg = len(set(f["nobio_file"] for f in file_level_features_cache)) > 1
    create_noeasy_reg = len(set(f["noeasy_file"] for f in file_level_features_cache)) > 1
    create_noctr_reg = len(set(f["noctr_file"] for f in file_level_features_cache)) > 1

    for file_data in file_level_features_cache:
        for trial in file_data["trials"]:
            q_id = trial.get("question_id")
            delegation_choice_str = trial.get("delegation_choice")
            
            if not q_id or not delegation_choice_str:
                continue

            gpqa_features = gpqa_feature_lookup.get(q_id)
            s_i_capability = capabilities_s_i_map_for_model.get(q_id)
            domain = gpqa_features.get('domain', 'unknown').replace(' ', '_').lower()

            if gpqa_features and gpqa_features.get('difficulty') is not None and s_i_capability is not None:
                delegate_choice_numeric = 1 if delegation_choice_str == "Teammate" else 0
                
                trial_data_dict = {
                    'q_id': q_id, # Ensure q_id is always in the trial data
                    'delegate_choice': delegate_choice_numeric,
                    's_i_capability': s_i_capability,
                    'human_difficulty': gpqa_features['difficulty'],
                    'q_length': np.log(len(gpqa_features.get('q_text', ''))),
                    'domain': domain,#####("Biology" if domain == "biology" else "NonBiology"),
                    'overlap_ratio': gpqa_features.get('overlap_ratio', 0),
                    'avg_word_length': get_average_word_length(gpqa_features.get('q_text', '')),
                    'percent_non_alphabetic_whitespace': get_percent_non_alphabetic_whitespace(gpqa_features.get('q_text', '')),
                }

                if create_teammate_skill_ratio_reg and file_data["teammate_accuracy_phase1_file"] is not None:
                    trial_data_dict['teammate_skill_ratio'] = file_data["teammate_accuracy_phase1_file"] / subject_acc_for_ratio_calc
                
                if create_summary_reg: trial_data_dict['summary'] = int(file_data["summary_file"])
                if create_nobio_reg: trial_data_dict['nobio'] = int(file_data["nobio_file"])
                if create_noeasy_reg: trial_data_dict['noeasy'] = int(file_data["noeasy_file"])
                if create_noctr_reg: trial_data_dict['noctr'] = int(file_data["noctr_file"])
                
                all_regression_data_for_model.append(trial_data_dict)
    
    if not all_regression_data_for_model:
        return None
    
    df_to_return = pd.DataFrame(all_regression_data_for_model)

    if 'judge_delegate' in df_to_return.columns and not df_to_return['judge_delegate'].notna().any():
        df_to_return = df_to_return.drop(columns=['judge_delegate'])
    
    return df_to_return, subject_acc_for_ratio_calc, phase2_corcnt, phase2_totalcnt

# --- File Grouping Logic ---
def get_feedback_status_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("feedback_config", {}).get("phase1_subject_feedback") is True
    except Exception:
        return False # Default on error

def split_by_feedback(file_list):
    true_files, false_files = [], []
    for f_path in file_list:
        (true_files if get_feedback_status_from_file(f_path) else false_files).append(f_path)
    result = []
    if true_files: result.append(("Feedback_True", true_files))
    if false_files: result.append(("Feedback_False", false_files)) # Catches False and None
    return result

def split_by_filename_attr(file_list, attr_check_func, name_if_true, name_if_false):
    true_files, false_files = [], []
    for f_path in file_list:
        (true_files if attr_check_func(os.path.basename(f_path)) else false_files).append(f_path)
    result = []
    if true_files: result.append((name_if_true, true_files))
    if false_files: result.append((name_if_false, false_files))
    return result

def process_file_groups(files_to_process, criteria_chain, model_name_for_log, group_path_names=None):
    group_path_names = group_path_names or ()

    if not criteria_chain:
        if files_to_process:
            yield group_path_names, files_to_process
        return

    current_criterion = criteria_chain[0]
    remaining_criteria = criteria_chain[1:]
    
    current_criterion_split_groups = current_criterion['split_logic'](files_to_process)

    if not current_criterion_split_groups and files_to_process:
        path_str = ", ".join([model_name_for_log] + list(group_path_names))
        parent_group_name = group_path_names[-1] if group_path_names else f"model {model_name_for_log}"
        indent = "  " * (len(group_path_names) + 2)
        print(f"{indent}No game files to process for {path_str} "
              f"after attempting to split by {current_criterion['name_prefix']}. "
              f"Skipping this {parent_group_name} group.")
        return

    for group_name, files_in_group in current_criterion_split_groups:
        indent_level = len(group_path_names) + 1
        print_indent = "  " * indent_level
        print(f"{print_indent}Processing for {group_name} ({len(files_in_group)} files)")

        yield from process_file_groups(
            files_in_group, remaining_criteria, model_name_for_log, group_path_names + (group_name,)
        )

# --- Main Analysis Logic ---
if __name__ == "__main__":
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

    game_logs_dir = "./delegate_game_logs/"
    capabilities_dir = "./completed_results_gpqa/"

    if not os.path.isdir(game_logs_dir) or not os.path.isdir(capabilities_dir):
        print(f"Error: Ensure directories exist: {game_logs_dir}, {capabilities_dir}")
        exit()

    model_game_files = defaultdict(list)
    for game_filename in sorted(os.listdir(game_logs_dir)):
        if game_filename.endswith("_game_data.json") and "_GPQA_" in game_filename:
            model_name_part = game_filename.split("_GPQA_")[0]
            model_game_files[model_name_part].append(os.path.join(game_logs_dir, game_filename))

    subj_acc_override_pattern = re.compile(r"_subj\d+(\.\d+)?_")
    FILE_GROUPING_CRITERIA = [
        {'name_prefix': "Feedback", 'split_logic': split_by_feedback},
        {'name_prefix': "Redaction", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_redacted_" in bn, "Redacted", "Non_Redacted")},
        {'name_prefix': "SubjAccOverride", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: subj_acc_override_pattern.search(bn), "SubjAccOverride", "NoSubjAccOverride")},
        {'name_prefix': "Randomized", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_randomized_" in bn, "Randomized", "NotRandomized")},
        {'name_prefix': "NoHistory", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_nohistory_" in bn, "NoHistory", "WithHistory")},
#        {'name_prefix': "Summary", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_summary_" in bn, "Summary", "NoSummary")},
        {'name_prefix': "Filtered", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_filtered_" in bn, "Filtered", "NotFiltered")},
    ]

    for model_name_part, game_files_for_model in model_game_files.items():
        print(f"\nProcessing model: {model_name_part} (total {len(game_files_for_model)} game files)")
        if not game_files_for_model:
            print(f"  No game files found for model {model_name_part}. Skipping.")
            continue

        for group_names_tuple, current_game_files_for_analysis in process_file_groups(
                game_files_for_model, FILE_GROUPING_CRITERIA, model_name_part):
            
            capabilities_filename = f"{model_name_part}_phase1_completed.json"
            capabilities_file_path = os.path.join(capabilities_dir, capabilities_filename)

            if not os.path.exists(capabilities_file_path):
                print(f"{'  '*(len(group_names_tuple)+1)}Corresponding capabilities file not found: {capabilities_file_path}. Skipping this group.")
                continue

            s_i_map_for_this_model = {}
            try:
                with open(capabilities_file_path, 'r', encoding='utf-8') as f_cap:
                    cap_data = json.load(f_cap)
                for q_id, res_info in cap_data.get("results", {}).items():
                    if res_info.get("is_correct") is not None:
                        s_i_map_for_this_model[q_id] = 1 if res_info["is_correct"] else 0
            except Exception as e:
                print(f"{'  '*(len(group_names_tuple)+1)}Error loading Ccapabilities file {capabilities_file_path}: {e}. Skipping this group.")
                continue
            
            if not s_i_map_for_this_model:
                print(f"{'  '*(len(group_names_tuple)+1)}No S_i data loaded from {capabilities_file_path}. Skipping this group.")
                continue

            if not current_game_files_for_analysis: # Should be caught by process_file_groups, but as a safeguard
                print(f"{'  '*(len(group_names_tuple)+1)}No game files for analysis for this group. Skipping.")
                continue
            
            df_model, subject_acc_phase1, phase2_corcnt, phase2_totalcnt = prepare_regression_data_for_model(current_game_files_for_analysis,
                                                         gpqa_feature_lookup,
                                                         s_i_map_for_this_model)

            if df_model is None or df_model.empty:
                print(f"{'  '*(len(group_names_tuple)+1)}No data for regression analysis for group: {model_name_part} ({', '.join(group_names_tuple)}).")
                continue

            if 'teammate_skill_ratio' in df_model.columns:
                mean_skill_ratio = df_model['teammate_skill_ratio'].mean()
                df_model['teammate_skill_ratio'] = df_model['teammate_skill_ratio'] - mean_skill_ratio
            
            log_context_str = f"{model_name_part} ({', '.join(group_names_tuple)}, {len(current_game_files_for_analysis)} game files)"
            log_output(f"\n--- Analyzing {log_context_str} ---", print_to_console=True)
            log_output(f"              Game files for analysis: {current_game_files_for_analysis}\n")
            
            if current_game_files_for_analysis:
                first_game_log_path = current_game_files_for_analysis[0].replace("_game_data.json", ".log")
                log_metrics_dict = extract_log_file_metrics(first_game_log_path)
                for metric, value in log_metrics_dict.items():
                    log_output(f"                  {metric}: {value}")
            
            try:
                log_output(f"df_model['delegate_choice'].value_counts()= {df_model['delegate_choice'].value_counts(dropna=False)}\n")
                if 's_i_capability' in df_model.columns:
                    cross_tab_s_i = pd.crosstab(df_model['delegate_choice'], df_model['s_i_capability'])
                    log_output(f"Cross-tabulation of delegate_choice vs. s_i_capability:\n{cross_tab_s_i}\n")
                    prob_delegating_Si0 = df_model.loc[df_model['s_i_capability'] == 0, 'delegate_choice'].mean()
                    log_output(f"Probability of delegating when s_i_capability is 0: {prob_delegating_Si0:.4f}")
                    prob_delegating_Si1 = df_model.loc[df_model['s_i_capability'] == 1, 'delegate_choice'].mean()
                    log_output(f"Probability of delegating when s_i_capability is 1: {prob_delegating_Si1:.4f}")
                    log_output(f"Phase 1 accuracy: {subject_acc_phase1:.4f} (n=400)")
                    log_output(f"Phase 2 self-accuracy: {phase2_corcnt/phase2_totalcnt:.4f} (n={phase2_totalcnt})")
                
                if 'team_correct' in df_model.columns:
                    cross_tab_team_correct = pd.crosstab(df_model['delegate_choice'], df_model['team_correct'])
                    log_output(f"Cross-tabulation of delegate_choice vs. team_correct:\n{cross_tab_team_correct}\n")
                    if 's_i_capability' in df_model.columns: # self_correct is s_i_capability when delegate_choice == 0
                        self_choice_df = df_model[df_model['delegate_choice'] == 0]
                        if not self_choice_df.empty:
                             cross_tab_self_s_i_vs_team = pd.crosstab(self_choice_df['s_i_capability'], self_choice_df['team_correct'])
                             log_output(f"Cross-tabulation of s_i_capability vs. team_correct (for self_choice trials):\n{cross_tab_self_s_i_vs_team}\n")


                log_output("\n  Model 1: Delegate_Choice ~ S_i_capability")
                logit_model1 = smf.logit('delegate_choice ~ s_i_capability', data=df_model).fit(disp=0)
                log_output(logit_model1.summary())

                log_output("\n  Model 2: Delegate_Choice ~ human_difficulty")
                logit_model2 = smf.logit('delegate_choice ~ human_difficulty', data=df_model).fit(disp=0)
                log_output(logit_model2.summary())

                log_output("\n  Model 3: Delegate_Choice ~ S_i_capability + human_difficulty")
                logit_model3 = smf.logit('delegate_choice ~ s_i_capability + human_difficulty', data=df_model).fit(disp=0)
                log_output(logit_model3.summary())

                if df_model['domain'].nunique() > 1 and len(df_model) > 20 : # Heuristic checks
                    min_obs_per_category=int(len(df_model)/15) + 1
                    
                    for col, new_col_name in [('domain', 'domain_grouped')]:
                        counts = df_model[col].value_counts()
                        rare_items = counts[counts < min_obs_per_category].index.tolist()
                        df_model[new_col_name] = df_model[col].apply(lambda x: 'Misc' if x in rare_items else x)
                        grouped_counts = df_model[new_col_name].value_counts()
                        if grouped_counts.get('Misc', 0) < min_obs_per_category and 'Misc' in grouped_counts : # Check 'Misc' exists before trying to replace
                            df_model[new_col_name] = df_model[new_col_name].replace({'Misc': 'Other'})
                        log_output(f"                  Grouped rare {col} into 'Misc'/'Other': {rare_items if rare_items else 'None'}")
                    
                    domain_column_for_formula = 'domain_grouped' if 'domain_grouped' in df_model else 'domain'

                    base_model_terms = [
                        's_i_capability',
                        'human_difficulty',
                        'q_length',
                        f'C({domain_column_for_formula})',
                        'overlap_ratio',
                        'avg_word_length',
                        'percent_non_alphabetic_whitespace'
                    ]

                    log_output(f"                  Domain Counts:\n {df_model['domain'].value_counts()}")
                    
                    # Logging for grouped variables
                    for group_col in [domain_column_for_formula]:
                        if group_col in df_model:
                            log_output(f"                  Capability by {group_col}:")
                            log_output(df_model.groupby(group_col)['s_i_capability'].agg(['mean', 'std', 'count']))
                            log_output(f"                  Delegate Choice by {group_col}:\n{df_model.groupby(group_col)['delegate_choice'].value_counts(normalize=True)}\n")

                    log_output("                  Q length by capability:")
                    log_output(df_model.groupby('s_i_capability')['q_length'].agg(['mean', 'std', 'count']))

                    log_output(f"{df_model.groupby('domain_grouped')['delegate_choice'].value_counts(normalize=True)}\n")

                    conditional_regressors = ['summary', 'nobio', 'noeasy', 'noctr', 'judge_delegate']

                    final_model_terms = list(base_model_terms)
                    if 'teammate_skill_ratio' in df_model.columns:
                        final_model_terms.append('teammate_skill_ratio')
                        final_model_terms.append(f"s_i_capability:teammate_skill_ratio") # Interaction term
                    for regressor in conditional_regressors:
                        if regressor in df_model.columns:
                            final_model_terms.append(regressor)
                    
                    fit_kwargs = {'disp': 0}
                    if df_model['q_id'].duplicated().any():
                        fit_kwargs.update({'cov_type': 'cluster', 'cov_kwds': {'groups': df_model['q_id']}})
                        log_output("                    Model 4: Using clustered standard errors by q_id.")
                            
                    model_def_str_4 = 'delegate_choice ~ ' + ' + '.join(final_model_terms)
                    interaction_str = "Interactions" if 's_i_capability:teammate_skill_ratio' in final_model_terms else "No Interactions"
                    log_output(f"\n                  Model 4 ({interaction_str}): {model_def_str_4}")
                    try:
                        logit_model4 = smf.logit(model_def_str_4, data=df_model).fit(**fit_kwargs)
                        log_output(logit_model4.summary())
                        if 's_i_capability' in logit_model4.params:
                            coef_s_i = logit_model4.params['s_i_capability']
                            pval_s_i = logit_model4.pvalues['s_i_capability']
                            conf_int_s_i_log_odds = logit_model4.conf_int().loc['s_i_capability']
                            odds_ratio_delegate_Si0_vs_Si1 = np.exp(-coef_s_i)
                            ci_lower_or, ci_upper_or = np.exp(-conf_int_s_i_log_odds.iloc[1]), np.exp(-conf_int_s_i_log_odds.iloc[0])
                            log_output(f"\n                  --- Odds Ratio for S_i_capability on Delegation (Adjusted M4) ---")
                            log_output(f"                  P-value for s_i_capability: {pval_s_i:.4g}")
                            log_output(f"                  Odds Ratio (Delegating when S_i=0 vs. S_i=1): {odds_ratio_delegate_Si0_vs_Si1:.4f} [{ci_lower_or:.4f}, {ci_upper_or:.4f}]")
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 4: {e_full}")

                    # Model 5 (No interaction)
                    final_model_terms_m5 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t)]
                    model_def_str_5 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m5)
                    if model_def_str_5 != model_def_str_4:
                        log_output(f"\n                  Model 4 (No Interactions): {model_def_str_5}")
                        try:
                            logit_model5 = smf.logit(model_def_str_5, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model5.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 5: {e_full}")

                    # Model 5.5 (If judge_delegate was used in Model 5, do a model without it)
                    if 'judge_delegate' in final_model_terms_m5:
                        final_model_terms_m55 = [t for t in final_model_terms_m5 if t != 'judge_delegate']
                        model_def_str_5_5 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m55)
                        log_output(f"\n                  Model 5.5: {model_def_str_5_5}")
                        try:
                            logit_model5_5 = smf.logit(model_def_str_5_5, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model5_5.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 5.5: {e_full}")
                    

                    # Model 7 (No s_i_capability, no interaction)
                    final_model_terms_m7 = [t for t in final_model_terms_m5 if t != 's_i_capability']
                    model_def_str_7 = 'delegate_choice ~ ' + ' + '.join(final_model_terms_m7)
                    log_output(f"\n                  Model 7: {model_def_str_7}")
                    try:
                        logit_model7 = smf.logit(model_def_str_7, data=df_model).fit(**fit_kwargs)
                        log_output(logit_model7.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 7: {e_full}")

                    # Model 8 (judge_delegate only)
                    if 'judge_delegate' in df_model.columns:
                        model_def_str_8 = 'delegate_choice ~ judge_delegate'
                        log_output(f"\n                  Model 8: {model_def_str_8}")
                        try:
                            # Use original fit_kwargs which might include clustering by q_id
                            logit_model8 = smf.logit(model_def_str_8, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model8.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 8: {e_full}")

                    # Model 9 (judge_delegate but not other surface regessors)
                    if 'judge_delegate' in df_model.columns:
                        model_def_str_9 = 'delegate_choice ~ judge_delegate + s_i_capability + teammate_skill_ratio'
                        log_output(f"\n                  Model 9: {model_def_str_9}")
                        try:
                            # Use original fit_kwargs which might include clustering by q_id
                            logit_model9 = smf.logit(model_def_str_9, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model9.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 9: {e_full}")

                    # Model 10 (judge_delegate vs s_i_capability)
                    if 'judge_delegate' in df_model.columns:
                        model_def_str_10 = 'delegate_choice ~ judge_delegate + s_i_capability'
                        log_output(f"\n                  Model 10: {model_def_str_10}")
                        try:
                            # Use original fit_kwargs which might include clustering by q_id
                            logit_model10 = smf.logit(model_def_str_10, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model10.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 10: {e_full}")
                            
                else:
                    log_output("\n                  Skipping Full Models due to insufficient data points (<=20).", print_to_console=True)

            except Exception as e:
                print(f"                  Error during logistic regression for {log_context_str}: {e}")
            
            print("-" * 40)