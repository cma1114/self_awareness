import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.proportion as smp
import json
import os
import numpy as np
from load_and_format_datasets import load_and_format_dataset
import re
from collections import defaultdict
from logres_helpers import *

def log_output(message_string, print_to_console=False):
    with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
        f.write(str(message_string) + "\n")
    if print_to_console:
        print(message_string)

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
                                      capabilities_s_i_map_for_model,
                                      p_i_map_for_this_model=None,
                                      entropy_map_for_this_model=None,
                                      game_file_suffix=""):
    all_regression_data_for_model = []
    file_level_features_cache = []

    if not game_file_paths_list:
        return None

    phase2_corcnt, phase2_totalcnt = 0, 0
    for game_file_path in game_file_paths_list:
        try:
            with open(game_file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)

        except Exception as e:
            print(f"Error loading game file {game_file_path}: {e}")
            continue

        filename_base = os.path.basename(game_file_path)
        trials = []
        for qid, result in game_data["results"].items():
            trials.append(result)
            trials[-1]["id"] = qid  
            if result["answer_changed"]: 
                phase2_totalcnt += 1
                if result["is_correct"]:
                    phase2_corcnt += 1

        if trials:
            file_level_features_cache.append({
                "trials": trials,
                "summary_file": "_summary_" in filename_base,
                "nobio_file": "_nobio_" in filename_base,
                "noeasy_file": "_noeasy_" in filename_base,
                "noctr_file": "_noctr_" in filename_base,
            })

    if not file_level_features_cache:
        print(f"No valid game data found in the provided files.")
        return None
    
    create_summary_reg = len(set(f["summary_file"] for f in file_level_features_cache)) > 1
    create_nobio_reg = len(set(f["nobio_file"] for f in file_level_features_cache)) > 1
    create_noeasy_reg = len(set(f["noeasy_file"] for f in file_level_features_cache)) > 1
    create_noctr_reg = len(set(f["noctr_file"] for f in file_level_features_cache)) > 1

    for file_ctr, file_data in enumerate(file_level_features_cache):
        print(f"\nProcessing file {file_ctr + 1}/{len(file_level_features_cache)}: {game_file_paths_list[file_ctr]}")
        print(f"len(file_data['trials']) = {len(file_data['trials'])}")
        for trial in file_data["trials"]:
            q_id = trial.get("id")

            prob_dict_trial = trial.get("probs")
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

            gpqa_features = gpqa_feature_lookup.get(q_id)
            s_i_capability = capabilities_s_i_map_for_model.get(q_id)
            domain = gpqa_features.get('domain', 'unknown').replace(' ', '_').lower()
            p_i_capability = p_i_map_for_this_model.get(q_id) if p_i_map_for_this_model else None
            capabilities_entropy = entropy_map_for_this_model.get(q_id) if entropy_map_for_this_model else None

            if gpqa_features and s_i_capability is not None:
                answer_changed_numeric = 1 if trial["answer_changed"] else 0
                
                trial_data_dict = {
                    'q_id': q_id, 
                    'answer_changed': answer_changed_numeric,
                    's_i_capability': s_i_capability,
                    'subject_correct': False if trial.get('is_correct') is None else trial['is_correct'],
                    'human_difficulty': gpqa_features['difficulty'],
                    'q_length': np.log(len(gpqa_features.get('q_text', ''))),
                    'domain': domain,#####("Biology" if domain == "biology" else "NonBiology"),
                    'avg_word_length': get_average_word_length(gpqa_features.get('q_text', '')),
                    'percent_non_alphabetic_whitespace': get_percent_non_alphabetic_whitespace(gpqa_features.get('q_text', '')),
                    'p_i_capability': p_i_capability,
                    'capabilities_entropy': capabilities_entropy,
                    "experiment_id": file_ctr,
                }

                if max_norm_prob_trial is not None:
                    trial_data_dict['max_normalized_prob'] = max_norm_prob_trial
                if norm_prob_entropy_trial is not None:
                    trial_data_dict['normalized_prob_entropy'] = norm_prob_entropy_trial
                
                if create_summary_reg: trial_data_dict['summary'] = int(file_data["summary_file"])
                if create_nobio_reg: trial_data_dict['nobio'] = int(file_data["nobio_file"])
                if create_noeasy_reg: trial_data_dict['noeasy'] = int(file_data["noeasy_file"])
                if create_noctr_reg: trial_data_dict['noctr'] = int(file_data["noctr_file"])
                
                all_regression_data_for_model.append(trial_data_dict)
            else:
                if not gpqa_features:
                    print(f"Warning: No SQA features found for q_id {q_id} in file {game_file_paths_list[file_ctr]}. Skipping trial.")
                if s_i_capability is None:
                    print(f"Warning: No S_i capability found for q_id {q_id} in file {game_file_paths_list[file_ctr]}. Skipping trial.")
    if not all_regression_data_for_model:
        print(f"No valid regression data found in the provided game files.")
        return None
    
    df_to_return = pd.DataFrame(all_regression_data_for_model)
    
    return df_to_return, phase2_corcnt, phase2_totalcnt

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

    dataset = "GPSA"#"GPQA" #
    game_type = "sc"


    LOG_FILENAME = f"analysis_log_multi_logres_{game_type}_{dataset.lower()}.txt"
    print(f"Loading main {dataset} dataset for features...")
    gpqa_all_questions = load_and_format_dataset(dataset) 

    gpqa_feature_lookup = {
        item['id']: {
            'overlap_ratio': item.get('overlap_ratio', 0),
            'difficulty': item['difficulty_score'],
            'domain': item['high_level_domain'],
            'q_text': item['question']
        } for item in gpqa_all_questions
    }
    print(f"GPQA feature lookup created with {len(gpqa_feature_lookup)} entries.")

    game_logs_dir = "./delegate_game_logs/" if game_type == "dg" else "./pass_game_logs/" if game_type == "aop" else "./secondchance_game_logs/"
    capabilities_dir = "./completed_results_gpqa/" if dataset == "GPQA" else "./compiled_results_gpsa/"
    game_file_suffix = "_evaluated" if dataset == "GPSA" else ""
    test_file_suffix = "completed" if dataset == "GPQA" else "compiled"

    if not os.path.isdir(game_logs_dir) or not os.path.isdir(capabilities_dir):
        print(f"Error: Ensure directories exist: {game_logs_dir}, {capabilities_dir}")
        exit()

    skip_files = []#['claude-3-5-sonnet-20241022_SimpleQA_50_100_team0.1_temp0.0_1748028564_game_data_evaluated.json', 'claude-3-5-sonnet-20241022_SimpleQA_50_100_team0.2_temp0.0_1748028190_game_data_evaluated.json',  'claude-3-5-sonnet-20241022_SimpleQA_50_100_team0.7_1747746405_game_data_evaluated.json']
    hit_files = None#["claude-3-5-sonnet-20241022_SimpleMC_50_500_subj0.5_subjgame0.5_team0.5_temp0.0_1750276176_game_data.json"]

    model_game_files = defaultdict(list)
    for game_filename in sorted(os.listdir(game_logs_dir)):
        if game_filename in skip_files:
            continue
        if hit_files and game_filename not in hit_files:
            continue

        if game_filename.endswith(f"_game_data{game_file_suffix}.json") and f"_{dataset}_" in game_filename:
            model_name_part = game_filename.split(f"_{dataset}_")[0]
            model_game_files[model_name_part].append(os.path.join(game_logs_dir, game_filename))

    subj_acc_override_pattern = re.compile(r"_subj\d+(\.\d+)?_")
    subj_game_override_pattern = re.compile(r"_subjgame\d+(\.\d+)?_")

    if game_type == "dg":
        FILE_GROUPING_CRITERIA = [
            {'name_prefix': "Feedback", 'split_logic': split_by_feedback},
            {'name_prefix': "Redaction", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_redacted_" in bn, "Redacted", "Non_Redacted")},
            {'name_prefix': "SubjAccOverride", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: subj_acc_override_pattern.search(bn), "SubjAccOverride", "NoSubjAccOverride")},
            {'name_prefix': "SubjGameOverride", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: subj_game_override_pattern.search(bn), "SubjGameOverride", "NoSubjGameOverride")},
            {'name_prefix': "Randomized", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_randomized_" in bn, "Randomized", "NotRandomized")},
            {'name_prefix': "NoHistory", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_nohistory_" in bn, "NoHistory", "WithHistory")},
    #        {'name_prefix': "Summary", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_summary_" in bn, "Summary", "NoSummary")},
            {'name_prefix': "Filtered", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_filtered_" in bn, "Filtered", "NotFiltered")},
        ]
    elif game_type == "aop":
        FILE_GROUPING_CRITERIA = [
        {'name_prefix': "MsgHist", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_hist_" in bn, "MsgHist", "NoMsgHist")},
        {'name_prefix': "QCtr", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_noqcnt_" in bn, "NoQCtr", "QCtr")},
        {'name_prefix': "PCtr", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_nopcnt_" in bn, "NoPCtr", "PCtr")},
        {'name_prefix': "SCtr", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_noscnt_" in bn, "NoSCtr", "SCtr")},
        ]
    else:
        FILE_GROUPING_CRITERIA = [
            {'name_prefix': "Redaction", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_redacted_" in bn, "Redacted", "Non_Redacted")},
        ]

    for model_name_part, game_files_for_model in model_game_files.items():
        print(f"\nProcessing model: {model_name_part} (total {len(game_files_for_model)} game files)")
        if not game_files_for_model:
            print(f"  No game files found for model {model_name_part}. Skipping.")
            continue

        for group_names_tuple, current_game_files_for_analysis in process_file_groups(
                game_files_for_model, FILE_GROUPING_CRITERIA, model_name_part):
            
            capabilities_filename = f"{model_name_part}_phase1_{test_file_suffix}.json"
            capabilities_file_path = os.path.join(capabilities_dir, capabilities_filename)

            if not os.path.exists(capabilities_file_path):
                print(f"{'  '*(len(group_names_tuple)+1)}Corresponding capabilities file not found: {capabilities_file_path}. Skipping this group.")
                continue

            s_i_map_for_this_model = {}
            p_i_map_for_this_model = {}
            entropy_map_for_this_model = {}
            try:
                with open(capabilities_file_path, 'r', encoding='utf-8') as f_cap:
                    cap_data = json.load(f_cap)
                for q_id, res_info in cap_data.get("results", {}).items():
                    if res_info.get("is_correct") is not None:
                        s_i_map_for_this_model[q_id] = 1 if res_info["is_correct"] else 0

                    probs_dict = res_info.get("probs")
                    subject_answer = res_info.get("subject_answer")
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


            except Exception as e:
                print(f"{'  '*(len(group_names_tuple)+1)}Error loading Capabilities file {capabilities_file_path}: {e}. Skipping this group.")
                continue
            
            if not s_i_map_for_this_model:
                print(f"{'  '*(len(group_names_tuple)+1)}No S_i data loaded from {capabilities_file_path}. Skipping this group.")
                continue

            if not current_game_files_for_analysis: # Should be caught by process_file_groups, but as a safeguard
                print(f"{'  '*(len(group_names_tuple)+1)}No game files for analysis for this group. Skipping.")
                continue
            
            df_model, phase2_corcnt, phase2_totalcnt = prepare_regression_data_for_model(current_game_files_for_analysis,
                                                         gpqa_feature_lookup,
                                                         s_i_map_for_this_model,
                                                         p_i_map_for_this_model,
                                                         entropy_map_for_this_model,
                                                         game_file_suffix=game_file_suffix)

            if df_model is None or df_model.empty:
                print(f"{'  '*(len(group_names_tuple)+1)}No data for regression analysis for group: {model_name_part} ({', '.join(group_names_tuple)}).")
                continue

            log_context_str = f"{model_name_part} ({', '.join(group_names_tuple)}, {len(current_game_files_for_analysis)} game files)"
            log_output(f"\n--- Analyzing {log_context_str} ---", print_to_console=True)
            log_output(f"              Game files for analysis: {current_game_files_for_analysis}\n")
            
            if current_game_files_for_analysis:
                first_game_log_path = current_game_files_for_analysis[0].replace("_game_data.json", ".log")
                log_metrics_dict = extract_log_file_metrics(first_game_log_path)
                for metric, value in log_metrics_dict.items():
                    log_output(f"                  {metric}: {value}")
            
            try:
                log_output(f"df_model['answer_changed'].value_counts()= {df_model['answer_changed'].value_counts(dropna=False)}\n")
                #compute count of answer_changed == 1 / total count
                answer_changed_count = df_model['answer_changed'].sum()
                total_count = len(df_model)
                proportion_changed = answer_changed_count / total_count if total_count > 0 else float('nan')
                ci_low, ci_high = smp.proportion_confint(answer_changed_count, total_count, alpha=0.05, method='normal')
                log_output(f"Answer change%: {proportion_changed:.4f} [{ci_low}, {ci_high}] (n={total_count})")
                z_stat_vs25, p_value_vs25 = smp.proportions_ztest(answer_changed_count, total_count, 0.25)
                z_stat_vs0, p_value_vs0 = smp.proportions_ztest(answer_changed_count, total_count, 0.0)
                log_output(f"P-value vs 25%: {p_value_vs25:.4g}; P-value vs 0%: {p_value_vs0:.4g}")


                ci_low, ci_high = smp.proportion_confint(phase2_corcnt, phase2_totalcnt, alpha=0.05, method='normal')
                log_output(f"Phase 2 self-accuracy: {phase2_corcnt/phase2_totalcnt:.4f} [{ci_low}, {ci_high}] (n={phase2_totalcnt})")
                z_stat_vs25, p_value_vs25 = smp.proportions_ztest(phase2_corcnt, phase2_totalcnt, 0.25)
                z_stat_vs33, p_value_vs33 = smp.proportions_ztest(phase2_corcnt, phase2_totalcnt, 0.333)
                log_output(f"P-value vs 25%: {p_value_vs25:.4g}; P-value vs 33%: {p_value_vs33:.4g}")
    
                if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any():
                    # Model 1.5: capabilities_entropy alone
                    log_output("\n  Model 1.5: Answer Changed ~ capabilities_entropy")
                    try:
                        logit_m2 = smf.logit('answer_changed ~ capabilities_entropy', data=df_model.dropna(subset=['capabilities_entropy', 'answer_changed'])).fit(disp=0)
                        log_output(logit_m2.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 1.5: {e_full}")

                if 'normalized_prob_entropy' in df_model.columns and df_model['normalized_prob_entropy'].notna().any():
                    # Model 1.6: normalized_prob_entropy alone
                    log_output("\n  Model 1.6: Answer Changed ~ Game Entropy")
                    try:
                        logit_m2 = smf.logit('answer_changed ~ normalized_prob_entropy', data=df_model.dropna(subset=['normalized_prob_entropy', 'answer_changed'])).fit(disp=0)
                        log_output(logit_m2.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 1.6: {e_full}")

                if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any() and 'normalized_prob_entropy' in df_model.columns and df_model['normalized_prob_entropy'].notna().any():
                    # Model 1.7: both entropy measures
                    log_output("\n  Model 1.7: Answer Changed ~ capabilities_entropy + Game Entropy")
                    try:
                        logit_m2 = smf.logit('answer_changed ~ capabilities_entropy + normalized_prob_entropy', data=df_model.dropna(subset=['capabilities_entropy', 'answer_changed'])).fit(disp=0)
                        log_output(logit_m2.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 1.7: {e_full}")

                log_output("\n  Model 2: Answer Changed ~ human_difficulty")
                try:
                    logit_model2 = smf.logit('answer_changed ~ human_difficulty', data=df_model).fit(disp=0)
                    log_output(logit_model2.summary())
                except Exception as e_full:
                    log_output(f"                    Could not fit Model 2: {e_full}")


                if df_model['domain'].nunique() > 1 and len(df_model) > 20 : 
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
                        'human_difficulty',
                        'q_length',
                        f'C({domain_column_for_formula})',
                        'avg_word_length',
                        'percent_non_alphabetic_whitespace'
                    ]
                    if 'overlap_ratio' in df_model.columns:
                        base_model_terms.append('overlap_ratio')

                    log_output(f"                  Domain Counts:\n {df_model['domain'].value_counts()}")
                    
                    # Logging for grouped variables
                    for group_col in [domain_column_for_formula]:
                        if group_col in df_model:
                            log_output(f"                  Answer Changed by {group_col}:\n{df_model.groupby(group_col)['answer_changed'].value_counts(normalize=True)}\n")

                    log_output(f"{df_model.groupby('domain_grouped')['answer_changed'].value_counts(normalize=True)}\n")

                    conditional_regressors = ['summary', 'nobio', 'noeasy', 'noctr']


                    final_model_terms = list(base_model_terms)
                    for regressor in conditional_regressors:
                        if regressor in df_model.columns:
                            final_model_terms.append(regressor)
                    
                    fit_kwargs = {'disp': 0}
                    if df_model['q_id'].duplicated().any():
                        fit_kwargs.update({'cov_type': 'cluster', 'cov_kwds': {'groups': df_model['q_id']}})
                        log_output("                    Model 4: Using clustered standard errors by q_id.")
                            
                    model_def_str_4 = 'answer_changed ~ ' + ' + '.join(final_model_terms)
                    interaction_str = "Interactions" if 's_i_capability:teammate_skill_ratio' in final_model_terms else "No Interactions"
                    log_output(f"\n                  Model 4 ({interaction_str}): {model_def_str_4}")
                    try:
                        logit_model4 = smf.logit(model_def_str_4, data=df_model).fit(**fit_kwargs)
                        log_output(logit_model4.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 4: {e_full}")

                    if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any():
                        # Model 4.6: capabilities_entropy in full model w/o s_i_capability
                        final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                        final_model_terms_m45.append('capabilities_entropy')
                        model_def_str_4_5 = 'answer_changed ~ ' + ' + '.join(final_model_terms_m45)
                        log_output(f"\n                  Model 4.6: {model_def_str_4_5}")
                        try:
                            logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['capabilities_entropy', 'answer_changed'])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 4.6: {e_full}")

                    if 'normalized_prob_entropy' in df_model.columns and df_model['normalized_prob_entropy'].notna().any():
                        # Model 4.6: normalized_prob_entropy in full model w/o s_i_capability
                        final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                        final_model_terms_m45.append('normalized_prob_entropy')
                        model_def_str_4_5 = 'answer_changed ~ ' + ' + '.join(final_model_terms_m45)
                        log_output(f"\n                  Model 4.8: {model_def_str_4_5}")
                        try:
                            logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['normalized_prob_entropy', 'answer_changed'])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 4.8: {e_full}")

                    if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any() and 'normalized_prob_entropy' in df_model.columns and df_model['normalized_prob_entropy'].notna().any():
                        # Model 4.6: both entropies in full model w/o s_i_capability
                        final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                        final_model_terms_m45.append('capabilities_entropy')
                        final_model_terms_m45.append('normalized_prob_entropy')
                        model_def_str_4_5 = 'answer_changed ~ ' + ' + '.join(final_model_terms_m45)
                        log_output(f"\n                  Model 4.95: {model_def_str_4_5}")
                        try:
                            logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['normalized_prob_entropy', 'answer_changed'])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 4.95: {e_full}")

                            
                else:
                    log_output("\n                  Skipping Full Models due to insufficient data points (<=20).", print_to_console=True)

            except Exception as e:
                print(f"                  Error during logistic regression for {log_context_str}: {e}")
            
            print("-" * 40)