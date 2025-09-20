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
from scipy.stats import wilcoxon, ttest_rel
from statsmodels.stats.contingency_tables import mcnemar

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

def extract_file_key(filename):
    """Extract the key components from filename for matching game and neutral files."""
    # Remove the file extension and any suffix like "_game_data.json"
    #base = filename.replace("_game_data_evaluated.json", "").replace("_game_data.json", "").replace(".json", "")
    
    # Split on underscores
    parts =filename.split("_")
    
    # Find where "temp" appears - everything before it is our key
    temp_idx = None
    for i, part in enumerate(parts):
        if part.startswith("temp"):
            temp_idx = i + 1
            break
    
    if temp_idx is None:
        print(f"Warning: Could not find 'temp' in filename {filename}")
        return None
    
    # Key is everything up to "temp"
    key_parts = parts[:temp_idx]
    return "_".join(key_parts)

def build_neutral_file_lookup(neutral_logs_dir, dataset_name):
    """Build a lookup dictionary for neutral files based on their key components."""
    lookup = {}
    suffix = "_game_data.json" if dataset_name == "SimpleMC" else "_game_data_evaluated.json"
    
    for filename in os.listdir(neutral_logs_dir):
        if filename.endswith(suffix):
            file_key = extract_file_key(filename)
            if file_key:
                if file_key in lookup:
                    print(f"ERROR: Multiple neutral files found with same key '{file_key}':")
                    print(f"  - {lookup[file_key]}")
                    print(f"  - {filename}")
                    exit(1)
                lookup[file_key.replace("_neut","")] = filename
    
    return lookup

def load_neutral_file(game_file_path, neutral_logs_dir, neutral_lookup):
    """Load the corresponding neutral file for a given game file."""
    game_filename = os.path.basename(game_file_path)
    game_key = extract_file_key(game_filename)
    
    if game_key is None:
        return None, "Could not extract key"
    
    neutral_filename = neutral_lookup.get(game_key)
    if neutral_filename is None:
        return None, f"No match found (key: {game_key})"
    
    neutral_file_path = os.path.join(neutral_logs_dir, neutral_filename)
    
    try:
        with open(neutral_file_path, 'r', encoding='utf-8') as f:
            return json.load(f), neutral_filename
    except Exception as e:
        return None, f"Error loading file: {e}"

def prepare_regression_data_for_model(game_file_paths_list,
                                      sqa_feature_lookup,
                                      capabilities_s_i_map_for_model,
                                      p_i_map_for_this_model=None,
                                      entropy_map_for_this_model=None,
                                      neutral_logs_dir="./sc_logs_neutral/",
                                      dataset_name="SimpleMC"):
    all_regression_data_for_model = []
    file_level_features_cache = []

    if not game_file_paths_list:
        return None, 0, 0, [], []

    phase2_corcnt, phase2_totalcnt = 0, 0
    
    # For paired comparison statistics
    game_changes = []
    neutral_changes = []
    
    # Build neutral file lookup
    print(f"\nBuilding neutral file lookup from {neutral_logs_dir}...")
    neutral_lookup = build_neutral_file_lookup(neutral_logs_dir, dataset_name)
    print(f"Found {len(neutral_lookup)} neutral files in directory.")
    
    # First pass: check which files can be matched
    print("\nMatching game files to neutral files:")
    matched_files = []
    skipped_files = []
    
    for game_file_path in game_file_paths_list:
        game_filename = os.path.basename(game_file_path)
        neutral_data, neutral_info = load_neutral_file(game_file_path, neutral_logs_dir, neutral_lookup)
        
        if neutral_data is None:
            skipped_files.append((game_filename, neutral_info))
        else:
            matched_files.append((game_file_path, game_filename, neutral_info))
    
    # Log matching summary
    print(f"\nFile matching summary:")
    print(f"  Successfully matched: {len(matched_files)}/{len(game_file_paths_list)} files")
    
    if matched_files:
        print("\n  Matched pairs:")
        for _, game_name, neutral_name in matched_files[:5]:  # Show first 5
            print(f"    {game_name} → {neutral_name}")
        if len(matched_files) > 5:
            print(f"    ... and {len(matched_files) - 5} more")
    
    if skipped_files:
        print(f"\n  Skipped files ({len(skipped_files)}):")
        for game_name, reason in skipped_files:
            print(f"    {game_name}: {reason}")
    
    if not matched_files:
        print(f"No game files could be matched to neutral files.")
        return None, 0, 0, [], []
    
    # Now process matched files
    for game_file_path, game_filename, neutral_filename in matched_files:
        try:
            with open(game_file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
        except Exception as e:
            print(f"Error loading game file {game_file_path}: {e}")
            continue

        # Load corresponding neutral file (we know it exists)
        neutral_file_path = os.path.join(neutral_logs_dir, neutral_filename)
        try:
            with open(neutral_file_path, 'r', encoding='utf-8') as f:
                neutral_data = json.load(f)
        except Exception as e:
            print(f"Error loading neutral file {neutral_file_path}: {e}")
            continue

        filename_base = game_filename
        
        # Create lookup for neutral results by q_id
        neutral_results_lookup = {}
        for qid, result in neutral_data.get("results", {}).items():
            neutral_results_lookup[qid] = result
        
        trials = []
        for qid, result in game_data["results"].items():
            trial_data = result.copy()
            trial_data["id"] = qid
            
            # Get corresponding neutral result
            neutral_result = neutral_results_lookup.get(qid)
            if neutral_result:
                trial_data["neutral_answer_changed"] = neutral_result.get("answer_changed", False)
                trial_data["neutral_is_correct"] = neutral_result.get("is_correct", False)
            else:
                print(f"Warning: No neutral result found for q_id {qid}")
                continue
                
            trials.append(trial_data)
            
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
        return None, 0, 0, [], []
    
    create_summary_reg = len(set(f["summary_file"] for f in file_level_features_cache)) > 1
    create_nobio_reg = len(set(f["nobio_file"] for f in file_level_features_cache)) > 1
    create_noeasy_reg = len(set(f["noeasy_file"] for f in file_level_features_cache)) > 1
    create_noctr_reg = len(set(f["noctr_file"] for f in file_level_features_cache)) > 1

    for file_ctr, file_data in enumerate(file_level_features_cache):
        matched_file_info = matched_files[file_ctr] if file_ctr < len(matched_files) else ("", "", "")
        print(f"\nProcessing file {file_ctr + 1}/{len(file_level_features_cache)}: {matched_file_info[1]}")
        print(f"len(file_data['trials']) = {len(file_data['trials'])}")
        for trial in file_data["trials"]:
            q_id = trial.get("id")

            prob_dict_trial = trial.get("probs")
            entropy_trial = None

            if isinstance(prob_dict_trial, dict):
                prob_values = [float(v) for k, v in prob_dict_trial.items()]
                entropy_trial = -np.sum([p_val * np.log2(p_val) for p_val in prob_values if p_val > 1e-9])        

            sqa_features = sqa_feature_lookup.get(q_id)
            s_i_capability = capabilities_s_i_map_for_model.get(q_id)
            prob_dict = p_i_map_for_this_model.get(q_id)
            p_i_capability = max(prob_dict.values()) if prob_dict else None

            capabilities_entropy = entropy_map_for_this_model.get(q_id) if entropy_map_for_this_model else None

            if sqa_features and s_i_capability is not None:
                answer_changed_numeric = 1 if trial["answer_changed"] else 0
                neutral_changed_numeric = 1 if trial.get("neutral_answer_changed", False) else 0
                
                # Collect for paired comparison
                game_changes.append(answer_changed_numeric)
                neutral_changes.append(neutral_changed_numeric)
                
                # Create new DVs
                game_changed_neutral_not = 1 if (answer_changed_numeric == 1 and neutral_changed_numeric == 0) else 0
                neutral_changed_game_not = 1 if (neutral_changed_numeric == 1 and answer_changed_numeric == 0) else 0

                base_clean = {k.strip(): float(v) for k, v in (p_i_map_for_this_model.get(q_id) or {}).items()
                            if k.strip() != "T" and isinstance(v, (int, float))}
                game_clean = {k.strip(): float(v) for k, v in (prob_dict_trial or {}).items()
                            if k.strip() != "T" and isinstance(v, (int, float))}

                trial_data_dict = {
                    'q_id': q_id, 
                    'answer_changed': answer_changed_numeric,
                    'neutral_answer_changed': neutral_changed_numeric,
                    'game_changed_neutral_not': game_changed_neutral_not,
                    'neutral_changed_game_not': neutral_changed_game_not,
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
                    print(f"Warning: No SQA features found for q_id {q_id}. Skipping trial.")
                if s_i_capability is None:
                    print(f"Warning: No S_i capability found for q_id {q_id}. Skipping trial.")
    
    if not all_regression_data_for_model:
        print(f"No valid regression data found in the provided game files.")
        return None, 0, 0, [], []
    
    df_to_return = pd.DataFrame(all_regression_data_for_model)
    
    # Store matched file information as dataframe attributes for logging
    df_to_return.attrs['matched_files'] = [(mf[1], mf[2]) for mf in matched_files]
    
    # Return paired comparison data too
    return df_to_return, phase2_corcnt, phase2_totalcnt, game_changes, neutral_changes

def perform_paired_comparison(game_changes, neutral_changes):
    """Perform paired statistical comparison of answer change rates."""
    game_changes = np.array(game_changes)
    neutral_changes = np.array(neutral_changes)
    
    # Create contingency table
    both_changed = sum((game_changes == 1) & (neutral_changes == 1))
    game_only = sum((game_changes == 1) & (neutral_changes == 0))
    neutral_only = sum((game_changes == 0) & (neutral_changes == 1))
    neither_changed = sum((game_changes == 0) & (neutral_changes == 0))
    
    # McNemar's test
    result = mcnemar([[both_changed, game_only], [neutral_only, neither_changed]], exact=True)
    
    # Calculate rates
    game_change_rate = np.mean(game_changes)
    neutral_change_rate = np.mean(neutral_changes)
    
    # Calculate normalized lift (percent of possible improvement captured)
    # This measures how much of the "room for improvement" was captured by the game condition
    # Example: If neutral=10% and game=60%, normalized lift = (60-10)/(100-10) = 55.6%
    # Example: If neutral=50% and game=100%, normalized lift = (100-50)/(100-50) = 100%
    if neutral_change_rate < 1.0:  # Avoid division by zero
        normalized_lift = (game_change_rate - neutral_change_rate) / (1.0 - neutral_change_rate)
    else:
        normalized_lift = np.nan
    
    # Bootstrap confidence intervals for rates and normalized lift
    n_bootstrap = 10000
    bootstrap_game_rates = []
    bootstrap_neutral_rates = []
    bootstrap_normalized_lifts = []
    bootstrap_absolute_lifts = []
    
    n_pairs = len(game_changes)
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_pairs, n_pairs, replace=True)
        boot_game = game_changes[indices]
        boot_neutral = neutral_changes[indices]
        
        boot_game_rate = np.mean(boot_game)
        boot_neutral_rate = np.mean(boot_neutral)
        boot_absolute_lift = boot_game_rate - boot_neutral_rate
        
        bootstrap_game_rates.append(boot_game_rate)
        bootstrap_neutral_rates.append(boot_neutral_rate)
        bootstrap_absolute_lifts.append(boot_absolute_lift)
        
        # Normalized lift for this bootstrap sample
        if boot_neutral_rate < 1.0:
            boot_norm_lift = (boot_game_rate - boot_neutral_rate) / (1.0 - boot_neutral_rate)
            bootstrap_normalized_lifts.append(boot_norm_lift)
    
    # Calculate 95% confidence intervals
    game_rate_ci = np.percentile(bootstrap_game_rates, [2.5, 97.5])
    neutral_rate_ci = np.percentile(bootstrap_neutral_rates, [2.5, 97.5])
    absolute_lift_ci = np.percentile(bootstrap_absolute_lifts, [2.5, 97.5])
    
    # For normalized lift, only use valid values (where neutral rate < 1)
    valid_lifts = [x for x in bootstrap_normalized_lifts if not np.isnan(x)]
    if valid_lifts:
        normalized_lift_ci = np.percentile(valid_lifts, [2.5, 97.5])
    else:
        normalized_lift_ci = [np.nan, np.nan]
    
    return {
        'game_change_rate': game_change_rate,
        'game_change_rate_ci': game_rate_ci,
        'neutral_change_rate': neutral_change_rate,
        'neutral_change_rate_ci': neutral_rate_ci,
        'absolute_lift': game_change_rate - neutral_change_rate,
        'absolute_lift_ci': absolute_lift_ci,
        'normalized_lift': normalized_lift,
        'normalized_lift_ci': normalized_lift_ci,
        'both_changed': both_changed,
        'game_only_changed': game_only,
        'neutral_only_changed': neutral_only,
        'neither_changed': neither_changed,
        'mcnemar_statistic': result.statistic,
        'mcnemar_pvalue': result.pvalue,
        'n_pairs': len(game_changes)
    }

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
            'Idea 5: p-val',
            'M1.51: Coef', 'M1.51: CI', 'M1.51: p-val']
    
    df = df.reindex(columns=cols)
    
    df.to_csv(filename, index=False)
    print(f"\n--- Summary data saved to {filename} ---")


if __name__ == "__main__":
    all_model_summary_data = []

    dataset = "SimpleQA" #"SimpleMC" #
    game_type = "sc"
    sc_version = "_new"  # "_new" or ""
    suffix = "_all"  # "_all" or ""
    VERBOSE = False

    LOG_FILENAME = f"analysis_log_multi_logres_{game_type}_{dataset.lower()}{sc_version}{suffix}_vs_neutral.txt"
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

    game_logs_dir = "./sc_logs_new/" if sc_version == "_new" else "./secondchance_game_logs/"
    neutral_logs_dir = "./sc_logs_neutral/"
    capabilities_dir = "./compiled_results_sqa/" if dataset == "SimpleQA" else "./compiled_results_smc/"
    game_file_suffix = "_evaluated" if dataset == "SimpleQA" else ""
    test_file_suffix = "completed" if dataset == "GPQA" else "compiled"

    if not os.path.isdir(game_logs_dir) or not os.path.isdir(capabilities_dir) or not os.path.isdir(neutral_logs_dir):
        print(f"Error: Ensure directories exist: {game_logs_dir}, {capabilities_dir}, {neutral_logs_dir}")
        exit()

    skip_files = []
    hit_files = None

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
                            p_i_map_for_this_model[q_id] = probs_dict

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
            
            if not current_game_files_for_analysis:
                print(f"{'  '*(len(group_names_tuple)+1)}No game files for analysis for this group. Skipping.")
                continue
            
            result = prepare_regression_data_for_model(
                current_game_files_for_analysis,
                sqa_feature_lookup,
                s_i_map_for_this_model,
                p_i_map_for_this_model,
                entropy_map_for_this_model,
                neutral_logs_dir=neutral_logs_dir,
                dataset_name=dataset
            )
            
            # Handle the case where no data was returned
            if result is None:
                print(f"{'  '*(len(group_names_tuple)+1)}No data returned for group: {model_name_part} ({', '.join(group_names_tuple)}).")
                continue
            
            df_model, phase2_corcnt, phase2_totalcnt, game_changes, neutral_changes = result
            
            if df_model is None or df_model.empty:
                print(f"{'  '*(len(group_names_tuple)+1)}No data for regression analysis for group: {model_name_part} ({', '.join(group_names_tuple)}).")
                continue

            log_context_str = f"{model_name_part} ({', '.join(group_names_tuple)}, {len(current_game_files_for_analysis)} game files)"
            log_output(f"\n--- Analyzing {log_context_str} ---", print_to_console=True)
            log_output(f"              Game files for analysis: {current_game_files_for_analysis}\n")
            
            # Log the file matching results from the data preparation
            if hasattr(df_model, 'attrs') and 'matched_files' in df_model.attrs:
                log_output("              Matched file pairs:")
                for game_file, neutral_file in df_model.attrs['matched_files']:
                    log_output(f"                Game:    {game_file}")
                    log_output(f"                Neutral: {neutral_file}")
                log_output("")
            
            # Perform paired comparison
            if game_changes and neutral_changes:
                paired_results = perform_paired_comparison(game_changes, neutral_changes)
                log_output("\n--- Paired Comparison Results ---", print_to_console=True)
                log_output(f"  Number of paired observations: {paired_results['n_pairs']}")
                log_output(f"\n  Answer Change Rates:")
                log_output(f"    Game:    {paired_results['game_change_rate']:.3f} (95% CI: [{paired_results['game_change_rate_ci'][0]:.3f}, {paired_results['game_change_rate_ci'][1]:.3f}])")
                log_output(f"    Neutral: {paired_results['neutral_change_rate']:.3f} (95% CI: [{paired_results['neutral_change_rate_ci'][0]:.3f}, {paired_results['neutral_change_rate_ci'][1]:.3f}])")
                log_output(f"\n  Lift Measures:")
                log_output(f"    Absolute lift: {paired_results['absolute_lift']:.3f} (95% CI: [{paired_results['absolute_lift_ci'][0]:.3f}, {paired_results['absolute_lift_ci'][1]:.3f}])")
                log_output(f"                   ({paired_results['absolute_lift']*100:.1f} percentage points)")
                if not np.isnan(paired_results['normalized_lift']):
                    log_output(f"    Normalized lift: {paired_results['normalized_lift']:.3f} (95% CI: [{paired_results['normalized_lift_ci'][0]:.3f}, {paired_results['normalized_lift_ci'][1]:.3f}])")
                    log_output(f"    → Captured {paired_results['normalized_lift']*100:.1f}% of possible improvement from neutral baseline")
                else:
                    log_output(f"    Normalized lift: N/A (neutral rate already at ceiling)")
                log_output(f"\n  Contingency Table:")
                log_output(f"    Both changed: {paired_results['both_changed']}")
                log_output(f"    Game only changed: {paired_results['game_only_changed']}")
                log_output(f"    Neutral only changed: {paired_results['neutral_only_changed']}")
                log_output(f"    Neither changed: {paired_results['neither_changed']}")
                log_output(f"\n  McNemar's Test:")
                log_output(f"    Statistic: {paired_results['mcnemar_statistic']}")
                log_output(f"    p-value: {paired_results['mcnemar_pvalue']:.4f}")
                log_output(f"    Significant difference: {'Yes' if paired_results['mcnemar_pvalue'] < 0.05 else 'No'}")
            
            if current_game_files_for_analysis:
                first_game_log_path = current_game_files_for_analysis[0].replace("_game_data.json", ".log")
                log_metrics_dict = extract_log_file_metrics(first_game_log_path)
                for metric, value in log_metrics_dict.items():
                    log_output(f"                  {metric}: {value}")
            
            # Run regressions with new DVs
            for dv_name, dv_col in [("game_changed_neutral_not", "game_changed_neutral_not"), 
                                    ("neutral_changed_game_not", "neutral_changed_game_not")]:
                
                log_output(f"\n\n=== REGRESSIONS FOR DV: {dv_name} ===", print_to_console=True)
                
                try:
                    if 'p_i_capability' in df_model.columns and df_model['p_i_capability'].notna().any():
                        log_output(f"\n  Model 1.4: {dv_name} ~ capabilities_prob")
                        try:
                            logit_m2 = smf.logit(f'{dv_col} ~ p_i_capability', data=df_model.dropna(subset=['p_i_capability', dv_col])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 1.4: {e_full}")

                    if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any():
                        log_output(f"\n  Model 1.5: {dv_name} ~ capabilities_entropy")
                        try:
                            logit_m2 = smf.logit(f'{dv_col} ~ capabilities_entropy', data=df_model.dropna(subset=['capabilities_entropy', dv_col])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 1.5: {e_full}")

                    if 'game_entropy' in df_model.columns and df_model['game_entropy'].notna().any():
                        log_output(f"\n  Model 1.6: {dv_name} ~ Game Entropy")
                        try:
                            logit_m2 = smf.logit(f'{dv_col} ~ game_entropy', data=df_model.dropna(subset=['game_entropy', dv_col])).fit(disp=0)
                            log_output(logit_m2.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 1.6: {e_full}")

                    if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any() and 'game_entropy' in df_model.columns and df_model['game_entropy'].notna().any():
                        log_output(f"\n  Model 1.7: {dv_name} ~ capabilities_entropy + Game Entropy")
                        try:
                            logit_m2 = smf.logit(f'{dv_col} ~ capabilities_entropy + game_entropy', data=df_model.dropna(subset=['capabilities_entropy', dv_col])).fit(disp=0)
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
                                
                        model_def_str_4 = f'{dv_col} ~ ' + ' + '.join(final_model_terms)
                        log_output(f"\n                  Model 4: {model_def_str_4}")
                        try:
                            logit_model4 = smf.logit(model_def_str_4, data=df_model).fit(**fit_kwargs)
                            log_output(logit_model4.summary())
                        except Exception as e_full:
                            log_output(f"                    Could not fit Model 4: {e_full}")

                        if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any():
                            final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                            final_model_terms_m45.append('capabilities_entropy')
                            model_def_str_4_5 = f'{dv_col} ~ ' + ' + '.join(final_model_terms_m45)
                            log_output(f"\n                  Model 4.6: {model_def_str_4_5}")
                            try:
                                logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['capabilities_entropy', dv_col])).fit(disp=0)
                                log_output(logit_m2.summary())
                            except Exception as e_full:
                                log_output(f"                    Could not fit Model 4.6: {e_full}")

                        if 'game_entropy' in df_model.columns and df_model['game_entropy'].notna().any():
                            final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                            final_model_terms_m45.append('game_entropy')
                            model_def_str_4_5 = f'{dv_col} ~ ' + ' + '.join(final_model_terms_m45)
                            log_output(f"\n                  Model 4.8: {model_def_str_4_5}")
                            try:
                                logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['game_entropy', dv_col])).fit(disp=0)
                                log_output(logit_m2.summary())
                            except Exception as e_full:
                                log_output(f"                    Could not fit Model 4.8: {e_full}")

                        if 'capabilities_entropy' in df_model.columns and df_model['capabilities_entropy'].notna().any() and 'game_entropy' in df_model.columns and df_model['game_entropy'].notna().any():
                            final_model_terms_m45 = [t for t in final_model_terms if not (isinstance(t, str) and f"s_i_capability:teammate_skill_ratio" == t) and t != 's_i_capability']
                            final_model_terms_m45.append('capabilities_entropy')
                            final_model_terms_m45.append('game_entropy')
                            model_def_str_4_5 = f'{dv_col} ~ ' + ' + '.join(final_model_terms_m45)
                            log_output(f"\n                  Model 4.95: {model_def_str_4_5}")
                            try:
                                logit_m2 = smf.logit(model_def_str_4_5, data=df_model.dropna(subset=['game_entropy', dv_col])).fit(disp=0)
                                log_output(logit_m2.summary())
                            except Exception as e_full:
                                log_output(f"                    Could not fit Model 4.95: {e_full}")

                    else:
                        log_output("\n                  Skipping Full Models due to insufficient data points (<=20).", print_to_console=True)

                except Exception as e:
                    print(f"                  Error during logistic regression for {log_context_str}, DV={dv_name}: {e}")
            
            all_model_summary_data.append(summary_data)
            print("-" * 40)
    
    save_summary_data(all_model_summary_data, filename=f"sc_summary{dataset}_vs_neutral.csv")