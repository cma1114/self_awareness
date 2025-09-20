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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, ttest_rel, rankdata

FIRST_PASS = True
def log_output(message_string, print_to_console=False):
    global FIRST_PASS
    if FIRST_PASS:
        openstr = "w"
        FIRST_PASS = False
    else:
        openstr = "a"
    with open(LOG_FILENAME, openstr, encoding='utf-8') as f:
        f.write(str(message_string) + "\n")
    if print_to_console:
        print(message_string)

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
                                      sqa_feature_lookup,
                                      capabilities_s_i_map_for_model,
                                      a_i_map_for_this_model,
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
            entropy_trial = None

            if isinstance(prob_dict_trial, dict):
                new_answer = trial.get("new_answer")
#                if len(prob_dict_trial.keys()) == 1: prob_dict_trial = {new_answer: float(prob_dict_trial.get(new_answer))**(len(new_answer)//2)} #approx token count

                prob_values = [float(v) for k, v in prob_dict_trial.items()]
                entropy_trial = -np.sum([p_val * np.log2(p_val) for p_val in prob_values if p_val > 1e-9]) if len(prob_dict_trial) > 1 else (1-prob_dict_trial.get(new_answer))

            sqa_features = sqa_feature_lookup.get(q_id)
            s_i_capability = capabilities_s_i_map_for_model.get(q_id)
            prob_dict = p_i_map_for_this_model.get(q_id)
            p_i_capability = max(prob_dict.values()) if prob_dict else None

            capabilities_entropy = entropy_map_for_this_model.get(q_id) if entropy_map_for_this_model else None

            if sqa_features and s_i_capability is not None:
                answer_changed_numeric = 1 if trial["answer_changed"] else 0
                
                base_clean = {k.strip(): float(v) for k, v in (p_i_map_for_this_model.get(q_id) or {}).items()
                            if k.strip() != "T" and isinstance(v, (int, float))}
                game_clean = {k.strip(): float(v) for k, v in (prob_dict_trial or {}).items()
                            if k.strip() != "T" and isinstance(v, (int, float))}
                
                trial_data_dict = {
                    'q_id': q_id, 
                    'answer_changed': answer_changed_numeric,
                    'correct_answer': trial.get('correct_answer', None),
                    'subject_answer': trial.get('new_answer'),
                    's_i_capability': s_i_capability,
                    'subject_correct': False if trial.get('is_correct') is None else trial['is_correct'],
                    'answer_type': sqa_features['answer_type'],
                    'q_length': np.log(len(sqa_features.get('q_text', '')) + 1e-9), # Add epsilon for empty q_text
                    'topic': sqa_features.get('topic', ''),
                    'p_i_capability': p_i_capability,
                    'base_probs': base_clean if len(base_clean) == 4 else None,     
                    'game_probs': game_clean if entropy_trial and len(game_clean) == 4 else None,
                    'capabilities_entropy': capabilities_entropy,
                    'game_entropy': entropy_trial,
                    "experiment_id": file_ctr,
                }
                
                if create_summary_reg: trial_data_dict['summary'] = int(file_data["summary_file"])
                if create_nobio_reg: trial_data_dict['nobio'] = int(file_data["nobio_file"])
                if create_noeasy_reg: trial_data_dict['noeasy'] = int(file_data["noeasy_file"])
                if create_noctr_reg: trial_data_dict['noctr'] = int(file_data["noctr_file"])
                
                all_regression_data_for_model.append(trial_data_dict)
            else:
                if not sqa_features:
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

def save_summary_data(all_results, filename="final_summary.csv"):
    """Saves the collected results to a CSV file."""
    if not all_results:
        print("No summary data to save.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Reorder columns for clarity
    cols = ['Model', 'Condition',
            'Idea 0: Prop', 'Idea 0: CI', 'Idea 0: p-val',
            'Idea 1: p-val',
            'Idea 2: Coef', 'Idea 2: CI', 'Idea 2: p-val',
            'Idea 3: p-val',
            'Idea 4: p-val',
            'Idea 4.5: p-val',
            'Idea 5: p-val',
            'M1.51: Coef', 'M1.51: CI', 'M1.51: p-val']
    
    df = df.reindex(columns=cols)
    
    df.to_csv(filename, index=False)
    print(f"\n--- Summary data saved to {filename} ---")


if __name__ == "__main__":
    all_model_summary_data = []

    dataset = "SimpleMC"#"SimpleQA" #
    game_type = "sc"
    sc_version = "_new"  # "_new" or "" or "_neut"
    suffix = ""  # "_all" or ""
    VERBOSE = False

    LOG_FILENAME = f"analysis_log_multi_logres_{game_type}_{dataset.lower()}{sc_version}{suffix}.txt"
    print(f"Loading main {dataset} dataset for features...")
    sqa_all_questions = load_and_format_dataset(dataset)
    sqa_feature_lookup = {
        item['id']: {
            'answer_type': item.get('answer_type', 0),
            'topic': item['topic'],
            'q_text': item['question']
        } for item in sqa_all_questions
    }
    print(f"sqa feature lookup created with {len(sqa_feature_lookup)} entries.")

    game_logs_dir = "./delegate_game_logs/" if game_type == "dg" else "./pass_game_logs/" if game_type == "aop" else "./sc_logs_new/" if sc_version == "_new" else "./sc_logs_neutral/" if sc_version == "_neut" else "./secondchance_game_logs/"
    capabilities_dir = "./compiled_results_sqa/" if dataset == "SimpleQA" else "./compiled_results_smc/"
    game_file_suffix = "_evaluated" if dataset == "SimpleQA" else ""

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
        if suffix != "_all":
            FILE_GROUPING_CRITERIA.append(
            {'name_prefix': "Correctness", 'split_logic': lambda fl: split_by_filename_attr(fl, lambda bn: "_cor_" in bn, "Correct", "Incorrect")},
            )

    for model_name_part, all_game_files_for_model in model_game_files.items():
        game_files_for_model = [f for f in all_game_files_for_model if not ('temp0.0' in f and any('temp1.0' in other for other in all_game_files_for_model))]
        print(f"\nProcessing model: {model_name_part} (total {len(game_files_for_model)} game files)")
        if not game_files_for_model:
            print(f"  No game files found for model {model_name_part}. Skipping.")
            continue

        for group_names_tuple, current_game_files_for_analysis in process_file_groups(
                game_files_for_model, FILE_GROUPING_CRITERIA, model_name_part):

            summary_data = {'Model': model_name_part, 'Condition': ', '.join(group_names_tuple)}

            capabilities_filename = f"{model_name_part}_phase1_compiled.json"
            capabilities_file_path = os.path.join(capabilities_dir, capabilities_filename)

            if not os.path.exists(capabilities_file_path):
                print(f"{'  '*(len(group_names_tuple)+1)}Corresponding capabilities file not found: {capabilities_file_path}. Skipping this group.")
                continue

            s_i_map_for_this_model = {}
            p_i_map_for_this_model = {}
            a_i_map_for_this_model = {}
            entropy_map_for_this_model = {}
            try:
                with open(capabilities_file_path, 'r', encoding='utf-8') as f_cap:
                    cap_data = json.load(f_cap)
                for q_id, res_info in cap_data.get("results", {}).items():
                    if res_info.get("is_correct") is not None:
                        s_i_map_for_this_model[q_id] = 1 if res_info["is_correct"] else 0

                    probs_dict = res_info.get("probs")
                    subject_answer = res_info.get("subject_answer")
                    a_i_map_for_this_model[q_id] = subject_answer
                    # Populate p_i_map_for_this_model
                    if subject_answer is not None and isinstance(probs_dict, dict):
                        prob_for_subject_answer = probs_dict.get(subject_answer)
                        if isinstance(prob_for_subject_answer, (int, float)):
                            p_i_map_for_this_model[q_id] = probs_dict#list(probs_dict.values())# float(prob_for_subject_answer)
                            """
                            if len(probs_dict.keys()) > 1:
                                p_i_map_for_this_model[q_id] = probs_dict#list(probs_dict.values())# float(prob_for_subject_answer)
                            else:
                                p_i_map_for_this_model[q_id] = {subject_answer: float(prob_for_subject_answer)**(len(subject_answer)//2)} #approx token count
                            """
                    # Calculate and populate entropy_map_for_this_model
                    if isinstance(probs_dict, dict) and probs_dict:
                        prob_values = [float(p) for p in probs_dict.values() if isinstance(p, (int, float)) and p > 1e-9]
                        if prob_values:
                            entropy = -np.sum([p_val * np.log2(p_val) for p_val in prob_values if p_val > 1e-9]) if len(probs_dict.keys()) > 1 else (1-p_i_map_for_this_model[q_id][subject_answer])
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
                                                         sqa_feature_lookup,
                                                         s_i_map_for_this_model,
                                                         a_i_map_for_this_model,
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

                if 'p_i_capability' in df_model.columns and df_model['p_i_capability'].notna().any():
                    log_output("\n  Model 1.4: Answer Changed ~ capabilities_prob")
                    try:
                        logit_m2 = smf.logit('answer_changed ~ p_i_capability', data=df_model.dropna(subset=['p_i_capability', 'answer_changed'])).fit(disp=0)
                        log_output(logit_m2.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 1.4: {e_full}")

                if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any():
                    # Model 1.5: capabilities_entropy alone
                    log_output("\n  Model 1.5: Answer Changed ~ capabilities_entropy")
                    try:
                        logit_m2 = smf.logit('answer_changed ~ capabilities_entropy', data=df_model.dropna(subset=['capabilities_entropy', 'answer_changed'])).fit(disp=0)
                        log_output(logit_m2.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 1.5: {e_full}")

                    if 'base_probs' in df_model.columns and df_model['base_probs'].notna().any():
                        df_model_tmp = df_model.copy()
                        df_model = df_model.dropna(subset=["base_probs"])

                        log_output("\n  Idea 0: On Change trials, second-choice token in baseline gets selected in the game more often than chance (33%)")
                        df_changed = df_model[df_model['answer_changed'] == 1].copy()
                        if not df_changed.empty:
                            def get_second_choice(prob_dict):
                                if not isinstance(prob_dict, dict) or len(prob_dict) < 2:
                                    return None
                                sorted_keys = sorted(prob_dict, key=prob_dict.get, reverse=True)
                                return sorted_keys[1]

                            df_changed['second_choice_base'] = df_changed['base_probs'].apply(get_second_choice)
                            ###df_changed['game_answer'] = df_changed['game_probs'].apply(lambda d: max(d, key=d.get) if isinstance(d, dict) and d else None)
                            
                            second_choice_successes = (df_changed['second_choice_base'] == df_changed['subject_answer']).sum()
                            total_changed_trials = len(df_changed)

                            if total_changed_trials > 0:
                                proportion_to_second = second_choice_successes / total_changed_trials
                                ci_low, ci_high = smp.proportion_confint(second_choice_successes, total_changed_trials, alpha=0.05, method='normal')
                                z_stat_vs33, p_value_vs33 = smp.proportions_ztest(second_choice_successes, total_changed_trials, value=1/3)
                                log_output(f"                  Proportion of changes to 2nd choice: {proportion_to_second:.4f} [{ci_low:.4f}, {ci_high:.4f}] (n={total_changed_trials})")
                                log_output(f"                  P-value vs 33.3%: {p_value_vs33:.4g}")
                                summary_data['Idea 0: Prop'] = proportion_to_second
                                summary_data['Idea 0: CI'] = f"[{ci_low:.3f}, {ci_high:.3f}]"
                                summary_data['Idea 0: p-val'] = p_value_vs33
                        else:
                            log_output("                  No trials with answer changes found to analyze for Idea 0.")

                        if 'game_probs' in df_model.columns and df_model['game_probs'].notna().any():
                            df_model = df_model.dropna(subset=["game_probs"])
                            log_output("\n  Idea 1: Chosen token in baseline gets lower prob in game even when the answer does not change")
                            def get_sorted_probs(prob_dict):
                                if not isinstance(prob_dict, dict):
                                    return [None] * 4
                                sorted_probs = sorted(prob_dict.values(), reverse=True)
                                return (sorted_probs + [None] * 4)[:4]
                            base_probs_sorted = pd.DataFrame(
                                df_model["base_probs"].apply(get_sorted_probs).tolist(),
                                columns=["p1", "p2", "p3", "p4"],
                                index=df_model.index
                            )
                            df_model = pd.concat([df_model, base_probs_sorted], axis=1)

                            game_probs_sorted = pd.DataFrame(
                                df_model["game_probs"].apply(get_sorted_probs).tolist(),
                                columns=["gp1", "gp2", "gp3", "gp4"],
                                index=df_model.index
                            )
                            df_model = pd.concat([df_model, game_probs_sorted], axis=1)

                            df_model["base_answer"] = df_model["base_probs"].apply(
                                lambda d: max(d, key=d.get) if isinstance(d, dict) and d else None
                            )
                            df_model["p_baselinechosentoken_game"] = df_model.apply(
                                lambda r: r["game_probs"].get(r["base_answer"]) if isinstance(r["game_probs"], dict) else None,
                                axis=1
                            )
                            df_model["delta_p"] = df_model["p1"] - df_model["p_baselinechosentoken_game"]
                            
                            delta_p_no_change = df_model.loc[df_model["answer_changed"] == 0, "delta_p"].dropna()
                            df_idea1 = df_model.loc[df_model["answer_changed"] == 0, ["p1", "p_baselinechosentoken_game"]].dropna()
                            if not df_idea1.empty and len(df_idea1) > 1:
                                t_stat, t_p = ttest_rel(df_idea1["p1"], df_idea1["p_baselinechosentoken_game"])
                                w_stat, w_p = wilcoxon(df_idea1["p1"], df_idea1["p_baselinechosentoken_game"])
                            else:
                                t_stat, t_p, w_stat, w_p = (float('nan'), float('nan'), float('nan'), float('nan'))
                            log_output(f"Paired t-test delta_p: statistic={t_stat:.2f}, p={t_p:.3g}")
                            log_output(f"Wilcoxon delta_p: statistic={w_stat:.2f}, p={w_p:.3g}")
                            mean_dp = delta_p_no_change.mean()
                            se       = delta_p_no_change.std(ddof=1) / np.sqrt(len(delta_p_no_change))
                            ci_low   = mean_dp - 1.96 * se
                            ci_up    = mean_dp + 1.96 * se
                            log_output(f"Mean Δp = {mean_dp:.4f}  [{ci_low:.4f}, {ci_up:.4f}]")
                            log_output((f"Idea 1 N = {len(delta_p_no_change)}; "))
                            summary_data['Idea 1: p-val'] = w_p

                            log_output("\n  Idea 1.5: Calibration Metrics")
                            y = df_model['s_i_capability'].to_numpy()
                            p_correct = df_model.apply(lambda r: (r['base_probs'] or {}).get(r.get('correct_answer'), np.nan), axis=1).to_numpy()
                            # drop rows where we could not find the correct label
                            mask = ~np.isnan(p_correct)
                            y       = y[mask]
                            p_hat   = np.clip(p_correct[mask], 1e-15, 1 - 1e-15)
                            nll   = -np.log(p_hat).mean()                     
                            brier = np.mean((p_hat - y) ** 2) 
                            def brier_decomposition(p, y, n_bins=10):
                                """
                                p : 1-D array of forecast probabilities for the event (here: model is correct)
                                y : 1-D array of 0/1 outcomes (s_i_capability)
                                """
                                p = np.asarray(p, float)
                                y = np.asarray(y, int)
                                N = len(p)

                                # 1. bin forecasts exactly as for ECE
                                bins = np.linspace(0, 1, n_bins + 1)
                                bin_id = np.digitize(p, bins[1:-1], right=True)

                                rel = res = 0.0
                                for k in range(n_bins):
                                    mask = bin_id == k
                                    if not mask.any():
                                        continue
                                    pk = p[mask].mean()
                                    yk = y[mask].mean()
                                    wk = mask.mean()          # n_k / N
                                    rel += wk * (pk - yk)**2
                                    # resolution term uses yk and overall mean later

                                y_bar = y.mean()
                                for k in range(n_bins):
                                    mask = bin_id == k
                                    if mask.any():
                                        yk = y[mask].mean()
                                        wk = mask.mean()
                                        res += wk * (yk - y_bar)**2

                                unc = y_bar * (1 - y_bar)
                                bs  = ((p - y)**2).mean()
                                # sanity: bs == rel - res + unc (float precision)
                                return {"brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc}
                            brier_decomp = brier_decomposition(p_hat, y)
                            # ECE
                            n_bins = 10
                            bin_edges = np.linspace(0, 1, n_bins + 1)
                            bin_ids   = np.digitize(p_hat, bin_edges[1:-1], right=True)
                            ece, signed_ece = 0.0, 0.0
                            for b in range(n_bins):
                                mask = bin_ids == b
                                if not mask.any():               # skip empty bins
                                    continue
                                acc   = y[mask].mean()
                                conf  = p_hat[mask].mean()
                                ece  += np.abs(acc - conf) * mask.mean()
                                signed_ece += (conf-acc) * mask.mean()
                            metrics = {"nll": nll, "brier": brier, "ece": ece}
                            def auroc(p, y):
                                """
                                p : array-like, predicted probabilities for the positive class (correct answer)
                                y : array-like, binary 0/1 ground truth (1 = correct)
                                """
                                p = np.asarray(p, float)
                                y = np.asarray(y,  int)

                                # Mann-Whitney U formulation: AUROC = (rank sum of positives − m(m+1)/2) / (m*n)
                                ranks = rankdata(p, method="average")          # smallest => rank 1
                                pos_ranks = ranks[y == 1].sum()
                                m = (y == 1).sum()                             # # positives
                                n = (y == 0).sum()                             # # negatives
                                if m == 0 or n == 0:
                                    return np.nan                              # undefined if only one class
                                u_stat = pos_ranks - m * (m + 1) / 2
                                return u_stat / (m * n)

                            auroc_score = auroc(p_hat, y)
                            log_output(f"  NLL: {metrics['nll']:.4f}, Signed ECE (overconf pos under neg): {signed_ece:.4f}, ECE: {metrics['ece']:.4f} (n={len(mask)})")
                            log_output(f"  Brier: {brier_decomp['brier']:.4f}, "
                                    f"Reliability (absolute calibration error; lower better): {brier_decomp['reliability']:.4f}, "
                                    f"Resolution (relative calibration quality; higher better): {brier_decomp['resolution']:.4f}, "
                                    f"Uncertainty: {brier_decomp['uncertainty']:.4f} (n={len(mask)})")
                            log_output(f"  AUROC: {auroc_score:.4f}")

                            log_output("\n  Idea 2: Decrease in game prob of chosen token scales with its baseline probability")
                            m1 = smf.ols("delta_p ~ p1 * answer_changed", data=df_model).fit()
                            log_output(m1.summary())
                            summary_data['Idea 2: Coef'] = m1.params['p1']
                            summary_data['Idea 2: CI'] = f"[{m1.conf_int().loc['p1'][0]:.3f}, {m1.conf_int().loc['p1'][1]:.3f}]"
                            summary_data['Idea 2: p-val'] = m1.pvalues['p1']

                            log_output("\n  Idea 3: Entropy of unchosen tokens in game is lower than in baseline when the answer doesn't change")
                            def rest_entropy(row, which):
                                prob_dict = row.get(f"{which}_probs")
                                base_answer = row.get("base_answer")
                                if not isinstance(prob_dict, dict) or base_answer is None:
                                    return None

                                probs = [p for ltr, p in prob_dict.items() if ltr != base_answer]
                                
                                prob_sum = sum(probs)
                                if prob_sum > 1e-9:
                                    normalized_probs = [p / prob_sum for p in probs]
                                else:
                                    return 0.0

                                normalized_probs = np.clip(normalized_probs, 1e-12, 1)
                                return -np.sum(normalized_probs * np.log2(normalized_probs))
                            df_model["H_unchosen_base"] = df_model.apply(rest_entropy, axis=1, which="base")
                            df_model["H_unchosen_game"] = df_model.apply(rest_entropy, axis=1, which="game")
                            df_model["delta_H"] = df_model["H_unchosen_base"] - df_model["H_unchosen_game"] 
                            delta_H_no_change = df_model.loc[df_model["answer_changed"] == 0, "delta_H"].dropna()
                            df_idea3 = df_model.loc[df_model["answer_changed"] == 0, ["H_unchosen_base", "H_unchosen_game"]].dropna()
                            if not df_idea3.empty and len(df_idea3) > 1:
                                t_stat, t_p = ttest_rel(df_idea3["H_unchosen_base"], df_idea3["H_unchosen_game"])
                                w_stat, w_p = wilcoxon(df_idea3["H_unchosen_base"], df_idea3["H_unchosen_game"])
                            else:
                                t_stat, t_p, w_stat, w_p = (float('nan'), float('nan'), float('nan'), float('nan'))
                            log_output(f"Paired t-test delta_H: statistic={t_stat:.2f}, p={t_p:.3g}")
                            log_output(f"Wilcoxon delta_H: statistic={w_stat:.2f}, p={w_p:.3g}")
                            mean_dH = delta_H_no_change.mean()
                            se_dH = delta_H_no_change.std(ddof=1) / np.sqrt(len(delta_H_no_change)) if len(delta_H_no_change) > 0 else 0
                            ci_low_dH = mean_dH - 1.96 * se_dH
                            ci_up_dH = mean_dH + 1.96 * se_dH
                            log_output(f"Mean ΔH = {mean_dH:.4f}  [{ci_low_dH:.4f}, {ci_up_dH:.4f}]")
                            summary_data['Idea 3: p-val'] = w_p
                            
                            delta_H_changed = df_model.loc[df_model["answer_changed"] == 1, "delta_H"].dropna()
                            if not delta_H_changed.empty and len(delta_H_changed) > 1:
                                df_idea3_changed = df_model.loc[df_model["answer_changed"] == 1, ["H_unchosen_base", "H_unchosen_game"]].dropna()
                                if not df_idea3_changed.empty and len(df_idea3_changed) > 1:
                                    t_stat, t_p = ttest_rel(df_idea3_changed["H_unchosen_base"], df_idea3_changed["H_unchosen_game"])
                                    w_stat, w_p = wilcoxon(df_idea3_changed["H_unchosen_base"], df_idea3_changed["H_unchosen_game"])
                                    log_output(f"Paired t-test delta_H Changed: statistic={t_stat:.2f}, p={t_p:.3g}")
                                    log_output(f"Wilcoxon delta_H Changed: statistic={w_stat:.2f}, p={w_p:.3g}")
                                mean_dH_changed = delta_H_changed.mean()
                                se_dH_changed = delta_H_changed.std(ddof=1) / np.sqrt(len(delta_H_changed))
                                ci_low_dH_changed = mean_dH_changed - 1.96 * se_dH_changed
                                ci_up_dH_changed = mean_dH_changed + 1.96 * se_dH_changed
                                log_output(f"Mean ΔH Changed = {mean_dH_changed:.4f}  [{ci_low_dH_changed:.4f}, {ci_up_dH_changed:.4f}]")

                            log_output("\n  Idea 4: Percentage of probability mass devoted to top two tokens in the game is higher than in baseline (sharpening)")
                            df_model['p_top2_base'] = df_model['p1'] + df_model['p2']
                            df_model['p_top2_game'] = df_model['gp1'] + df_model['gp2']
                            df_idea4 = df_model[['p_top2_base', 'p_top2_game']].dropna()

                            if not df_idea4.empty and len(df_idea4) > 1:
                                # Game is expected to be higher, so test game vs base
                                t_stat, t_p = ttest_rel(df_idea4['p_top2_game'], df_idea4['p_top2_base'])
                                w_stat, w_p = wilcoxon(df_idea4['p_top2_game'], df_idea4['p_top2_base'])
                                
                                log_output(f"Paired t-test (p_top2_game vs p_top2_base): statistic={t_stat:.2f}, p={t_p:.3g}")
                                log_output(f"Wilcoxon (p_top2_game vs p_top2_base): statistic={w_stat:.2f}, p={w_p:.3g}")
                                
                                delta_p_top2 = (df_idea4['p_top2_game'] - df_idea4['p_top2_base']).dropna()
                                mean_delta = delta_p_top2.mean()
                                se_delta = delta_p_top2.std(ddof=1) / np.sqrt(len(delta_p_top2))
                                ci_low_delta = mean_delta - 1.96 * se_delta
                                ci_up_delta = mean_delta + 1.96 * se_delta
                                
                                log_output(f"Mean Δp_top2 = {mean_delta:.4f}  [{ci_low_delta:.4f}, {ci_up_delta:.4f}] (n={len(delta_p_top2)})")
                                summary_data['Idea 4: p-val'] = w_p
                            else:
                                log_output("                  Not enough data for Idea 4 analysis.")
                            log_output("\n  Idea 4.5: Game entropy over the tokens that were NOT the top token in the baseline is lower than over the same tokens in the baseline")
                            
                            def entropy_of_baseline_unchosen_set(prob_dict, baseline_top_token):
                                if not isinstance(prob_dict, dict) or baseline_top_token is None:
                                    return None
                                # Explicitly use the set of tokens unchosen in the baseline
                                unchosen_probs = [p for token, p in prob_dict.items() if token != baseline_top_token]
                                prob_sum = sum(unchosen_probs)
                                if prob_sum <= 1e-9:
                                    return 0.0
                                normalized_probs = [p / prob_sum for p in unchosen_probs]
                                normalized_probs = np.clip(normalized_probs, 1e-12, 1)
                                return -np.sum(p * np.log2(p) for p in normalized_probs)

                            H_base_set_in_base = df_model.apply(lambda row: entropy_of_baseline_unchosen_set(row['base_probs'], row['base_answer']), axis=1)
                            H_base_set_in_game = df_model.apply(lambda row: entropy_of_baseline_unchosen_set(row['game_probs'], row['base_answer']), axis=1)

                            df_idea45 = pd.DataFrame({
                                'H_base_set_in_base': H_base_set_in_base,
                                'H_base_set_in_game': H_base_set_in_game
                            }).dropna()

                            if not df_idea45.empty and len(df_idea45) > 1:
                                # Game entropy is expected to be lower
                                t_stat, t_p = ttest_rel(df_idea45['H_base_set_in_base'], df_idea45['H_base_set_in_game'])
                                w_stat, w_p = wilcoxon(df_idea45['H_base_set_in_base'], df_idea45['H_base_set_in_game'])
                                
                                log_output(f"Paired t-test (H_base_set_in_base vs H_base_set_in_game): statistic={t_stat:.2f}, p={t_p:.3g}")
                                log_output(f"Wilcoxon (H_base_set_in_base vs H_base_set_in_game): statistic={w_stat:.2f}, p={w_p:.3g}")
                                
                                delta_H_all = (df_idea45['H_base_set_in_base'] - df_idea45['H_base_set_in_game']).dropna()
                                mean_delta = delta_H_all.mean()
                                se_delta = delta_H_all.std(ddof=1) / np.sqrt(len(delta_H_all)) if len(delta_H_all) > 0 else 0
                                ci_low_delta = mean_delta - 1.96 * se_delta
                                ci_up_delta = mean_delta + 1.96 * se_delta
                                
                                log_output(f"Mean ΔH_unchosen_baseline_set = {mean_delta:.4f}  [{ci_low_delta:.4f}, {ci_up_delta:.4f}] (n={len(delta_H_all)})")
                                summary_data['Idea 4.5: p-val'] = w_p
                            else:
                                log_output("                  Not enough data for Idea 4.5 analysis.")

                            log_output("\n  Model 1.51: Answer Changed ~ p1_z + I(p1_z**2)")
                            try:
                                df_model["posterior_top"] = df_model["p2"] / (1.0 - df_model["p1"] + 1e-12 )
                                #logit_int = smf.logit("answer_changed ~ posterior_top + p1", data=df_model).fit()
                                df_model["surprise"] = -np.log(np.clip(1.0 - df_model["p1"], 1e-12, None))
                                df_model["surprise_z"] = (df_model["surprise"] - df_model["surprise"].mean()) / df_model["surprise"].std(ddof=0)
                                df_model["p1_z"] = StandardScaler().fit_transform(df_model[["p1"]])
                                logit_int = smf.logit("answer_changed ~ p1_z + I(p1_z**2)", data=df_model).fit()
                                #logit_int = smf.logit("answer_changed ~ surprise_z + I(surprise_z**2)",data=df_model).fit()
                                log_output(logit_int.summary())
                                auc = roc_auc_score(df_model["answer_changed"], logit_int.predict(df_model))
                                log_output(f"AUC = {auc:.3f}")
                                summary_data['M1.51: Coef'] = logit_int.params['I(p1_z ** 2)']
                                summary_data['M1.51: CI'] = f"[{logit_int.conf_int().loc['I(p1_z ** 2)'][0]:.3f}, {logit_int.conf_int().loc['I(p1_z ** 2)'][1]:.3f}]"
                                summary_data['M1.51: p-val'] = logit_int.pvalues['I(p1_z ** 2)']
                            except Exception as e_full:
                                log_output(f"                    Could not fit Model 1.51: {e_full}")

                        df_model = df_model_tmp.copy()  # Restore original df_model

                if 'game_entropy' in df_model.columns and df_model['game_entropy'].notna().any():
                    # Model 1.6: normalized_prob_entropy alone
                    log_output("\n  Model 1.6: Answer Changed ~ Game Entropy")
                    try:
                        logit_m2 = smf.logit('answer_changed ~ game_entropy', data=df_model.dropna(subset=['game_entropy', 'answer_changed'])).fit(disp=0)
                        log_output(logit_m2.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 1.6: {e_full}")

                if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any() and 'game_entropy' in df_model.columns and df_model['game_entropy'].notna().any():
                    log_output("\n  Idea 5: Game entropy is different than capabilities entropy")
                    df_idea5 = df_model[['game_entropy', 'capabilities_entropy']].dropna()
                    if not df_idea5.empty and len(df_idea5) > 1:
                        t_stat, t_p = ttest_rel(df_idea5['game_entropy'], df_idea5['capabilities_entropy'])
                        w_stat, w_p = wilcoxon(df_idea5['game_entropy'], df_idea5['capabilities_entropy'])
                        log_output(f"Wilcoxon (game_entropy vs capabilities_entropy): statistic={w_stat:.2f}, p={w_p:.3g}")
                        
                        delta_entropy = df_idea5['capabilities_entropy'] - df_idea5['game_entropy']
                        mean_delta_entropy = delta_entropy.mean()
                        se_delta_entropy = delta_entropy.std(ddof=1) / np.sqrt(len(delta_entropy))
                        ci_low_delta_entropy = mean_delta_entropy - 1.96 * se_delta_entropy
                        ci_up_delta_entropy = mean_delta_entropy + 1.96 * se_delta_entropy
                        log_output(f"Paired t-test (game_entropy vs capabilities_entropy): statistic={t_stat:.2f}, p={t_p:.3g}")
                        log_output(f"Mean capabilities_entropy-game_entropy = {mean_delta_entropy:.4f}  [{ci_low_delta_entropy:.4f}, {ci_up_delta_entropy:.4f}] (n={len(delta_entropy)})")
                        summary_data['Idea 5: p-val'] = w_p
                    else:
                        log_output("                  Not enough data for Idea 5 analysis.")
                    # Model 1.7: both entropy measures
                    log_output("\n  Model 1.7: Answer Changed ~ capabilities_entropy + Game Entropy")
                    try:
                        logit_m2 = smf.logit('answer_changed ~ capabilities_entropy + game_entropy', data=df_model.dropna(subset=['capabilities_entropy', 'answer_changed'])).fit(disp=0)
                        log_output(logit_m2.summary())
                    except Exception as e_full:
                        log_output(f"                    Could not fit Model 1.7: {e_full}")

                if len(df_model) > 20 :
                    min_obs_per_category=int(len(df_model)/15) + 1
                    if 'grok' in model_name_part: min_obs_per_category = 40
                    
                    for col, new_col_name in [('topic', 'topic_grouped'), ('answer_type', 'answer_type_grouped')]:
                        counts = df_model[col].value_counts()
                        rare_items = counts[counts < min_obs_per_category].index.tolist()
                        df_model[new_col_name] = df_model[col].apply(lambda x: 'Misc' if x in rare_items else x)
                        grouped_counts = df_model[new_col_name].value_counts()
                        if grouped_counts.get('Misc', 0) < min_obs_per_category and 'Misc' in grouped_counts : # Check 'Misc' exists before trying to replace
                            df_model[new_col_name] = df_model[new_col_name].replace({'Misc': 'Other'})
                        log_output(f"                  Grouped rare {col} into 'Misc'/'Other': {rare_items if rare_items else 'None'}")
                    
                    topic_column_for_formula = 'topic_grouped' if 'topic_grouped' in df_model else 'topic'
                    ans_type_column_for_formula = 'answer_type_grouped' if 'answer_type_grouped' in df_model else 'answer_type'

                    base_model_terms = [
                        f'C({topic_column_for_formula})',
                        f'C({ans_type_column_for_formula})',
                        'q_length',
                    ]
                    if VERBOSE:
                        log_output(f"                  Topic Counts:\n {df_model['topic'].value_counts()}")
                        log_output(f"                  Answer Type Counts:\n {df_model['answer_type'].value_counts()}")
                        cross_tab = pd.crosstab(df_model['topic_grouped'], df_model['answer_type_grouped'])
                        log_output("\n                  Cross-tabulation of Topic Grouped vs. Answer Type Grouped:")
                        log_output(cross_tab)
                        
                        # Logging for grouped variables
                        for group_col in [topic_column_for_formula, ans_type_column_for_formula]:
                            if group_col in df_model:
                                log_output(f"                  Answer Changed by {group_col}:\n{df_model.groupby(group_col)['answer_changed'].value_counts(normalize=True)}\n")

                    log_output(f"{df_model.groupby('topic_grouped')['answer_changed'].value_counts(normalize=True)}\n")
                    log_output(f"{df_model.groupby('answer_type_grouped')['answer_changed'].value_counts(normalize=True)}\n")
                    proportions = df_model.groupby(['topic_grouped', 'answer_type_grouped'])['answer_changed'].value_counts(normalize=True).unstack(fill_value=0)
                    total_counts = df_model.groupby(['topic_grouped', 'answer_type_grouped'])['answer_changed'].count()
                    proportions['total_count'] = total_counts
                    log_output("                  topic+answer_type By answer_changed:\n")
                    log_output(proportions)

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
                        log_output(f"Mean capabilities_entropy = {df_model['capabilities_entropy'].mean():.4f}")
                        try:
                            logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['capabilities_entropy', 'answer_changed'])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 4.6: {e_full}")

                    if 'game_entropy' in df_model.columns and df_model['game_entropy'].notna().any():
                        # Model 4.6: normalized_prob_entropy in full model w/o s_i_capability
                        final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                        final_model_terms_m45.append('game_entropy')
                        model_def_str_4_5 = 'answer_changed ~ ' + ' + '.join(final_model_terms_m45)
                        log_output(f"\n                  Model 4.8: {model_def_str_4_5}")
                        try:
                            logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['game_entropy', 'answer_changed'])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 4.8: {e_full}")

                    if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any() and 'game_entropy' in df_model.columns and df_model['game_entropy'].notna().any():
                        # Model 4.6: both entropies in full model w/o s_i_capability
                        final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                        final_model_terms_m45.append('capabilities_entropy')
                        final_model_terms_m45.append('game_entropy')
                        model_def_str_4_5 = 'answer_changed ~ ' + ' + '.join(final_model_terms_m45)
                        log_output(f"\n                  Model 4.95: {model_def_str_4_5}")
                        try:
                            logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['game_entropy', 'answer_changed'])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 4.95: {e_full}")


                    # Model 6 (Like M5, but on subset after determinism check)
                    final_model_terms_m5 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t)]
                    model_def_str_5 = 'answer_changed ~ ' + ' + '.join(final_model_terms_m5)
                    df_subset_m6, identified_deterministic_cats = identify_and_handle_deterministic_categories(
                        df_model, 'answer_changed', [topic_column_for_formula, ans_type_column_for_formula], min_obs_per_category)
                    if identified_deterministic_cats and not df_subset_m6.empty:
                        fit_kwargs_m6 = {'disp': 0}
                        if df_subset_m6['q_id'].duplicated().any():
                           fit_kwargs_m6.update({'cov_type': 'cluster', 'cov_kwds': {'groups': df_subset_m6['q_id']}})
                        log_output(f"\n                  Model 6: {model_def_str_5} (after removing deterministic categories: {identified_deterministic_cats})")
                        try:
                            logit_model6 = smf.logit(model_def_str_5, data=df_subset_m6).fit(**fit_kwargs_m6)
                            log_output(logit_model6.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 6: {e_full}")

                        if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any():
                            # Model 6.6: capabilities_entropy in full model w/o s_i_capability
                            final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                            final_model_terms_m45.append('capabilities_entropy')
                            model_def_str_4_5 = 'answer_changed ~ ' + ' + '.join(final_model_terms_m45)
                            log_output(f"\n                  Model 6.6: {model_def_str_4_5}")
                            try:
                                logit_m2 = smf.logit(model_def_str_4_5, data=df_subset_m6.dropna(subset=['capabilities_entropy', 'answer_changed'])).fit(**fit_kwargs_m6)
                                log_output(logit_m2.summary())
                            except Exception as e_full:
                                log_output(f"                    Could not fit Model 6.6: {e_full}")


                    elif df_subset_m6.empty :
                        log_output(f"                    Skipping Model 6: DataFrame empty after removing deterministic categories.")
                    else: # No deterministic categories found/removed, Model 6 would be same as Model 5
                        log_output(f"                    Skipping Model 6: No deterministic categories removed, would be same as Model 5.")
                    
                            
                else:
                    log_output("\n                  Skipping Full Models due to insufficient data points (<=20).", print_to_console=True)

            except Exception as e:
                print(f"                  Error during logistic regression for {log_context_str}: {e}")
            
            all_model_summary_data.append(summary_data)
            print("-" * 40)
    
    save_summary_data(all_model_summary_data, filename=f"sc_summary{dataset}.csv")            