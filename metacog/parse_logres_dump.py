import re

def parse_analysis_log(log_content, output_file):
    target_params = "Feedback_False, Non_Redacted, NoSubjAccOverride, NotRandomized, WithHistory, NotFiltered"
    
    block_start_regex = re.compile(
        r"--- Analyzing (\S+) \(" + re.escape(target_params) + r", \d+ game files\) ---"
    )

    prob_si0_regex = re.compile(r"^\s*Probability of delegating when s_i_capability is 0: (.*)$")
    prob_si1_regex = re.compile(r"^\s*Probability of delegating when s_i_capability is 1: (.*)$")
    phase1_acc_regex = re.compile(r"^\s*Phase 1 self-accuracy \(from completed results, total - phase2\): (.*)$")
    phase2_acc_regex = re.compile(r"^\s*Phase 2 self-accuracy: (.*)$")

    # Regex for cross-tabulation
    crosstab_title_regex = re.compile(r"^\s*Cross-tabulation of delegate_choice vs\. s_i_capability:$")
    # This will match the line "s_i_capability     0   1" or similar with spaces
    crosstab_col_header_regex = re.compile(r"^\s*s_i_capability\s+\S+\s+\S+") # Generalize for 0 and 1
    # This matches the "delegate_choice" line that might appear before data rows
    crosstab_row_header_label_regex = re.compile(r"^\s*delegate_choice\s*$")
    # This matches a data row like "0 11 9" or "1 46 34"
    crosstab_data_row_regex = re.compile(r"^\s*\d+\s+(\d+)\s+(\d+)\s*$") # Catches the two numbers

    model4_start_regex = re.compile(r"^\s*Model 4.*(?:No Interactions)?.*: delegate_choice ~")
    si_capability_line_regex = re.compile(r"^\s*s_i_capability\s+(-?\d+\.\d+)\s+.*$")

    analysis_blocks = re.split(r"(?=--- Analyzing )", log_content)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for block_content in analysis_blocks:
            if not block_content.strip():
                continue

            match_block_start = block_start_regex.search(block_content)
            if match_block_start:
                subject_name = match_block_start.group(1)
                
                outfile.write(f"Subject: {subject_name}\n")
                
                extracted_info = {
                    "prob_si0": "Not found", "P0_n0": "Not found",
                    "prob_si1": "Not found", "P1_n1": "Not found",
                    "phase1_acc": "Not found",
                    "phase2_acc": "Not found",
                    "model4_si_cap": "Not found"
                }
                
                # --- Cross-tab parsing state ---
                parsing_crosstab = False
                expecting_crosstab_col_header = False
                expecting_crosstab_row_header_label = False # For the "delegate_choice" line
                crosstab_data_lines_collected = 0
                temp_crosstab_cells = [] # To store the four cell values as they are found

                # --- Model 4 parsing state ---
                found_model4_summary = False
                in_model4_summary_table = False

                lines = block_content.splitlines()
                for line_idx, line in enumerate(lines):
                    # --- Standard extractions first ---
                    m_prob_si0 = prob_si0_regex.match(line)
                    if m_prob_si0: extracted_info["prob_si0"] = line.strip(); continue
                    m_prob_si1 = prob_si1_regex.match(line)
                    if m_prob_si1: extracted_info["prob_si1"] = line.strip(); continue
                    m_phase1_acc = phase1_acc_regex.match(line)
                    if m_phase1_acc: extracted_info["phase1_acc"] = line.strip(); continue
                    m_phase2_acc = phase2_acc_regex.match(line)
                    if m_phase2_acc: extracted_info["phase2_acc"] = line.strip(); continue

                    # --- Cross-tabulation parsing state machine ---
                    if crosstab_title_regex.match(line):
                        parsing_crosstab = True
                        expecting_crosstab_col_header = True
                        expecting_crosstab_row_header_label = False # Reset
                        crosstab_data_lines_collected = 0
                        temp_crosstab_cells = []
                        continue

                    if parsing_crosstab:
                        if expecting_crosstab_col_header and crosstab_col_header_regex.match(line):
                            expecting_crosstab_col_header = False
                            expecting_crosstab_row_header_label = True # Next could be "delegate_choice"
                            continue
                        elif expecting_crosstab_row_header_label and crosstab_row_header_label_regex.match(line):
                            expecting_crosstab_row_header_label = False # "delegate_choice" line found, now expect data
                            continue
                        elif (not expecting_crosstab_col_header): # If col_header was found, or if row_header_label was optional/passed
                            # This covers the case where delegate_choice line might be absent
                            if expecting_crosstab_row_header_label: # If we were still expecting "delegate_choice" but got data
                                expecting_crosstab_row_header_label = False # No longer expecting it

                            data_match = crosstab_data_row_regex.match(line)
                            if data_match:
                                temp_crosstab_cells.append(int(data_match.group(1))) # Cell for s_i=0
                                temp_crosstab_cells.append(int(data_match.group(2))) # Cell for s_i=1
                                crosstab_data_lines_collected += 1
                                if crosstab_data_lines_collected == 2:
                                    # temp_crosstab_cells should be [val_row0_si0, val_row0_si1, val_row1_si0, val_row1_si1]
                                    if len(temp_crosstab_cells) == 4:
                                        n0_val = temp_crosstab_cells[0] + temp_crosstab_cells[2]
                                        n1_val = temp_crosstab_cells[1] + temp_crosstab_cells[3]
                                        extracted_info["P0_n0"] = str(n0_val)
                                        extracted_info["P1_n1"] = str(n1_val)
                                    parsing_crosstab = False # Done with this table
                                continue
                            else: # Line doesn't match data row, assume end of crosstab
                                parsing_crosstab = False
                        # If still expecting col_header but didn't get it, or other unexpected line
                        elif not line.strip(): # Blank line might end the crosstab section prematurely
                            parsing_crosstab = False
                        # If line is not blank and doesn't match expected crosstab parts, assume crosstab ended
                        elif line.strip() and not (expecting_crosstab_col_header or expecting_crosstab_row_header_label):
                            parsing_crosstab = False


                    # --- Model 4 s_i_capability parsing (remains the same) ---
                    if not parsing_crosstab: # Only parse Model 4 if not in crosstab logic
                        if not found_model4_summary and model4_start_regex.search(line):
                            found_model4_summary = True
                            continue 
                        
                        if found_model4_summary and not in_model4_summary_table:
                            if "Logit Regression Results" in line or line.strip().startswith("==="):
                                if line.strip().startswith("===") and line.strip().endswith("==="):
                                    in_model4_summary_table = True
                            continue

                        if in_model4_summary_table:
                            m_si_cap = si_capability_line_regex.match(line)
                            if m_si_cap:
                                extracted_info["model4_si_cap"] = line.strip()
                                found_model4_summary = False 
                                in_model4_summary_table = False 
                            elif line.strip().startswith("--- Analyzing") or \
                                (line.strip().startswith("Model ") and not model4_start_regex.search(line)) or \
                                line.strip().startswith("--- Odds Ratio"):
                                found_model4_summary = False
                                in_model4_summary_table = False
                
                # Write out the collected info
                outfile.write(f"  {extracted_info['prob_si0']}\n")
                if extracted_info["P0_n0"] != "Not found":
                    outfile.write(f"  P0_n0: {extracted_info['P0_n0']}\n")
                outfile.write(f"  {extracted_info['prob_si1']}\n")
                if extracted_info["P1_n1"] != "Not found":
                    outfile.write(f"  P1_n1: {extracted_info['P1_n1']}\n")
                outfile.write(f"  {extracted_info['phase1_acc']}\n")
                outfile.write(f"  {extracted_info['phase2_acc']}\n")
                outfile.write(f"  Model 4 s_i_capability: {extracted_info['model4_si_cap']}\n")
                outfile.write("\n")
                
    print(f"Parsing complete. Output written to {output_file}")

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

# Z-score for 95% CI
Z_SCORE = norm.ppf(0.975)

def parse_value(text, pattern, group=1, as_type=float):
    """Helper to extract a value using regex and convert its type."""
    match = re.search(pattern, text)
    if match:
        try:
            return as_type(match.group(group))
        except (ValueError, TypeError):
            print(f"Warning: Could not convert value from '{text}' using pattern '{pattern}' to type {as_type}")
            return None
    return None

def parse_fraction(text, pattern):
    """Helper to extract a fraction like 'X/Y' and return X, Y as floats."""
    match = re.search(pattern, text)
    if match:
        try:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator == 0: # Avoid division by zero for the proportion itself
                print(f"Warning: Denominator is zero in fraction from '{text}'")
                return numerator, None 
            return numerator, denominator
        except (ValueError, TypeError):
            print(f"Warning: Could not parse fraction from '{text}' using pattern '{pattern}'")
            return None, None
    return None, None

def calculate_ci_proportion(p_val, n_val):
    """Calculate CI for a single proportion. p_val is the proportion, n_val is its sample size."""
    if p_val is None or n_val is None or n_val <= 0 or p_val < 0 or p_val > 1:
        return p_val, np.nan, np.nan # value, lower, upper
    
    # If p_val is exactly 0 or 1, SE is 0. Some CI methods (like Wilson) handle this better.
    # For normal approximation, if p=0 or p=1, CI is [0,0] or [1,1] if n>0.
    # However, for ratio CIs, a 0 SE can be problematic.
    # Let's use a small adjustment for p=0 or p=1 to avoid 0 SE for subsequent calculations if n is reasonable.
    # This is a common practice, e.g., adding 0.5 to successes and failures (Agresti-Coull adjustment idea)
    # but here we'll just adjust p slightly if it's exactly 0 or 1 for SE calculation.
    
    p_for_se = p_val
    if p_val == 0 and n_val > 0:
        p_for_se = 0.5 / n_val # Smallest non-zero proportion
    elif p_val == 1 and n_val > 0:
        p_for_se = (n_val - 0.5) / n_val # Largest non-one proportion

    se = np.sqrt(p_for_se * (1 - p_for_se) / n_val)
    lower_ci = p_val - Z_SCORE * se
    upper_ci = p_val + Z_SCORE * se
    return p_val, max(0, lower_ci), min(1, upper_ci)

def calculate_ci_diff_proportions(p1_val, n1_val, p2_val, n2_val):
    """Calculate CI for the difference of two independent proportions (p1 - p2)."""
    if any(v is None for v in [p1_val, n1_val, p2_val, n2_val]) or n1_val <= 0 or n2_val <= 0:
        return np.nan, np.nan, np.nan
    
    prop1_actual, _, _ = calculate_ci_proportion(p1_val, n1_val)
    prop2_actual, _, _ = calculate_ci_proportion(p2_val, n2_val)

    if np.isnan(prop1_actual) or np.isnan(prop2_actual):
        return np.nan, np.nan, np.nan
        
    diff = prop1_actual - prop2_actual
    
    # Use slightly adjusted p for SE calculation if p is 0 or 1 to avoid SE=0
    p1_for_se = p1_val if 0 < p1_val < 1 else (0.5/n1_val if p1_val==0 else (n1_val-0.5)/n1_val)
    p2_for_se = p2_val if 0 < p2_val < 1 else (0.5/n2_val if p2_val==0 else (n2_val-0.5)/n2_val)

    se1_sq = (p1_for_se * (1 - p1_for_se)) / n1_val
    se2_sq = (p2_for_se * (1 - p2_for_se)) / n2_val
    se_diff = np.sqrt(se1_sq + se2_sq)
    
    lower_ci = diff - Z_SCORE * se_diff
    upper_ci = diff + Z_SCORE * se_diff
    return diff, lower_ci, upper_ci

def calculate_ci_ratio_proportions_log_delta(p0_val, n0_val, p1_val, n1_val):
    """
    Calculate CI for the ratio of two independent proportions (p0 / p1)
    using the Delta method on the log-transformed ratio.
    """
    if any(v is None for v in [p0_val, n0_val, p1_val, n1_val]) or n0_val <= 0 or n1_val <= 0:
        return np.nan, np.nan, np.nan
    if p0_val < 0 or p0_val > 1 or p1_val < 0 or p1_val > 1:
        return np.nan, np.nan, np.nan
    if p1_val == 0: # Avoid division by zero for the ratio itself
        return np.nan, np.nan, np.nan # Ratio is undefined or infinite

    ratio = p0_val / p1_val
    
    # Handle cases where p0 or p1 are 0 or 1 for variance calculation
    # Use a continuity correction (e.g., add 0.5) for estimating variance if p is 0 or 1
    # This is to avoid Var=0 which makes SE_log_ratio undefined or zero.
    p0_eff = (p0_val * n0_val + 0.5) / (n0_val + 1) if n0_val > 0 else p0_val
    p1_eff = (p1_val * n1_val + 0.5) / (n1_val + 1) if n1_val > 0 else p1_val
    
    if p0_eff == 0 or p1_eff == 0 : # If effective p is still zero, CI is problematic
         return ratio, np.nan, np.nan

    var_log_p0 = (1 - p0_eff) / (p0_eff * n0_val) if n0_val > 0 and p0_eff !=0 else np.inf
    var_log_p1 = (1 - p1_eff) / (p1_eff * n1_val) if n1_val > 0 and p1_eff !=0 else np.inf
    
    if np.isinf(var_log_p0) or np.isinf(var_log_p1):
        return ratio, np.nan, np.nan

    with warnings.catch_warnings(): # Suppress RunTimeWarning for log(0) if p0_val is 0
        warnings.simplefilter("ignore", RuntimeWarning)
        log_ratio = np.log(p0_val) - np.log(p1_val) if p0_val > 0 else -np.inf # log(P0/P1)

    if np.isinf(log_ratio) and p0_val == 0: # Ratio is 0, log_ratio is -inf
        # If p0 is 0, ratio is 0. CI is [0, exp(log(0) + Z*SE_log_ratio_approx_at_0)]
        # This case is tricky for symmetric CIs via log method. A Fieller or bootstrap would be better.
        # For simplicity, if p0=0, ratio is 0. Upper bound can be estimated, lower is 0.
        # Let's return NaN for CI if p0 is 0 for now, as delta method on log scale struggles.
        if p0_val == 0: return 0.0, 0.0, np.nan # Ratio is 0, lower is 0, upper is uncertain with this method

    var_log_ratio_val = var_log_p0 + var_log_p1
    
    if var_log_ratio_val < 0 or np.isnan(var_log_ratio_val) or np.isinf(var_log_ratio_val):
        return ratio, np.nan, np.nan

    se_log_ratio = np.sqrt(var_log_ratio_val)

    log_lower_ci = log_ratio - Z_SCORE * se_log_ratio
    log_upper_ci = log_ratio + Z_SCORE * se_log_ratio

    lower_ci = np.exp(log_lower_ci)
    upper_ci = np.exp(log_upper_ci)
    
    # If original p0 was 0, the ratio is 0 and lower_ci should be 0.
    if p0_val == 0: lower_ci = 0.0

    return ratio, lower_ci, upper_ci


def analyze_parsed_data(input_summary_file):
    all_subject_data = []
    current_subject_info = {}

    with open(input_summary_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("Subject:"):
                if current_subject_info.get("subject_name"): # Save previous valid subject's data
                    all_subject_data.append(current_subject_info)
                current_subject_info = {"subject_name": line.split("Subject:")[1].strip()}
            elif "Probability of delegating when s_i_capability is 0:" in line:
                current_subject_info["prob_s0"] = parse_value(line, r":\s*(-?\d+\.?\d*)")
            elif "P0_n0:" in line:
                current_subject_info["p0_n0"] = parse_value(line, r":\s*(\d+)", as_type=int)
            elif "Probability of delegating when s_i_capability is 1:" in line:
                current_subject_info["prob_s1"] = parse_value(line, r":\s*(-?\d+\.?\d*)")
            elif "P1_n1:" in line:
                current_subject_info["p1_n1"] = parse_value(line, r":\s*(\d+)", as_type=int)
            elif "Phase 1 self-accuracy" in line: 
                num, den = parse_fraction(line, r":\s*(\d+)/(\d+)")
                if num is not None and den is not None:
                    current_subject_info["p1_acc_val"] = num / den
                    current_subject_info["p1_acc_n"] = den
                else: 
                    val_perc = parse_value(line, r"\((\d+\.?\d+)%\)")
                    current_subject_info["p1_acc_val"] = val_perc / 100 if val_perc is not None else None
                    current_subject_info["p1_acc_n"] = None 
            elif "Phase 2 self-accuracy:" in line: 
                current_subject_info["p2_acc_val"] = parse_value(line, r":\s*(\d+\.?\d*)")
                current_subject_info["p2_acc_n"] = parse_value(line, r"\(n=(\d+)\)", as_type=int)
            elif "Model 4 s_i_capability:" in line:
                # Line example: "Model 4 s_i_capability: s_i_capability      -1.0064  0.436   -2.311   0.021   -1.860   -0.153"
                # We want the numbers after the second "s_i_capability"
                
                # Find the part of the string that actually contains the coefficient line
                # This assumes the first parser wrote "s_i_capability <coef_line_content>"
                match_coef_line_content = re.search(r"Model 4 s_i_capability:\s*(s_i_capability\s+.*)", line)
                if match_coef_line_content:
                    coef_line_part = match_coef_line_content.group(1) # This is "s_i_capability      -1.0064 ..."
                    # Now extract numbers from this specific part
                    # Regex: s_i_capability <coef> <std_err> <z> <P> <CI_low> <CI_high>
                    # We need coef, CI_low, CI_high
                    # (-?\d+\.\d+) matches a float
                    # \s+ matches one or more spaces
                    # .*? matches anything non-greedily
                    # We need the 1st, 5th, and 6th numbers if all are present
                    
                    # Simpler: just get all floats from the coef_line_part
                    all_floats = re.findall(r"(-?\d+\.\d+)", coef_line_part)
                    
                    if len(all_floats) >= 3: # Expecting at least coef, ci_low, ci_high
                        # Assuming the order is coef, std_err, z, p_val, ci_low, ci_high
                        # Or if only coef, ci_low, ci_high are reliably the first and last two of interest
                        current_subject_info["si_coef"] = float(all_floats[0]) # The first float after "s_i_capability"
                        current_subject_info["si_coef_ci_low"] = float(all_floats[-2]) # Second to last float
                        current_subject_info["si_coef_ci_high"] = float(all_floats[-1]) # Last float
                    else:
                        print(f"Warning: Could not parse enough float values for SI Coef from: '{coef_line_part}' in line: '{line}'")

                else:
                    print(f"Warning: Could not find s_i_capability coefficient line structure in: '{line}'")        
        if current_subject_info.get("subject_name"): 
            all_subject_data.append(current_subject_info)

    results = []
    for data in all_subject_data:
        subject_name = data.get("subject_name", "Unknown")
        
        prob_s0 = data.get("prob_s0")
        p0_n0 = data.get("p0_n0")
        prob_s1 = data.get("prob_s1")
        p1_n1 = data.get("p1_n1")
        
        delegating_ratio, ratio_ci_low, ratio_ci_high = calculate_ci_ratio_proportions_log_delta(
            prob_s0, p0_n0, prob_s1, p1_n1
        )
        
        p2_acc = data.get("p2_acc_val")
        p2_n = data.get("p2_acc_n")
        p1_acc = data.get("p1_acc_val")
        p1_n = data.get("p1_acc_n")
        
        acc_lift, acc_lift_ci_low, acc_lift_ci_high = calculate_ci_diff_proportions(p2_acc, p2_n, p1_acc, p1_n)

        si_coef = data.get("si_coef")
        si_ci_low_orig = data.get("si_coef_ci_low")
        si_ci_high_orig = data.get("si_coef_ci_high")
        
        rev_si_coef = (-1 * si_coef) if si_coef is not None else np.nan
        rev_si_ci_low = (-1 * si_ci_high_orig) if si_ci_high_orig is not None else np.nan
        rev_si_ci_high = (-1 * si_ci_low_orig) if si_ci_low_orig is not None else np.nan
        
        results.append({
            "Subject": subject_name,
            "Delegating Ratio": delegating_ratio,
            "DelRatio_CI_Low": ratio_ci_low,
            "DelRatio_CI_High": ratio_ci_high,
            "Accuracy Lift (P2-P1)": acc_lift,
            "AccLift_CI_Low": acc_lift_ci_low,
            "AccLift_CI_High": acc_lift_ci_high,
            "Reversed SI Coef": rev_si_coef,
            "RevSICoef_CI_Low": rev_si_ci_low,
            "RevSICoef_CI_High": rev_si_ci_high
        })
        
    return pd.DataFrame(results)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings # Added for consistency, though not strictly needed here

def break_subject_name(name, max_parts_per_line=3):
    """Breaks a subject name string by hyphens for better display."""
    parts = name.split('-')
    if len(parts) <= max_parts_per_line:
        return name # No need to break if short enough
    
    wrapped_name = ""
    for i, part in enumerate(parts):
        wrapped_name += part
        if (i + 1) % max_parts_per_line == 0 and (i + 1) < len(parts):
            wrapped_name += "-\n" # Add hyphen back and newline
        elif (i + 1) < len(parts):
            wrapped_name += "-" # Add hyphen back
    return wrapped_name

def plot_results(df_results, subject_order=None):
    if df_results.empty:
        print("No data to plot.")
        return

    # Reorder DataFrame if subject_order is provided
    if subject_order:
        # Convert Subject column to categorical to enforce order
        df_results_ordered = df_results.copy()
        df_results_ordered['Subject_Cat'] = pd.Categorical(df_results_ordered['Subject'], categories=subject_order, ordered=True)
        # Filter out subjects not in the order list and sort
        df_results_ordered = df_results_ordered[df_results_ordered['Subject_Cat'].notna()].sort_values('Subject_Cat')
        if df_results_ordered.empty and not df_results.empty:
            print("Warning: None of the subjects in subject_order were found in the data. Plotting all available subjects.")
            # Fallback to original df_results if ordering results in empty df (e.g. bad order list)
        elif len(df_results_ordered) < len(df_results):
             print(f"Warning: Plotting only {len(df_results_ordered)} subjects present in the provided order list.")
             df_results = df_results_ordered.drop(columns=['Subject_Cat'])
        else:
            df_results = df_results_ordered.drop(columns=['Subject_Cat'])


    num_subjects = len(df_results)
    if num_subjects == 0:
        print("No subjects to plot after filtering/ordering.")
        return

    # Apply name breaking for x-axis labels
    # You can adjust max_parts_per_line as needed
    formatted_subject_names = [break_subject_name(name, max_parts_per_line=3) for name in df_results["Subject"]]


    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            print("Seaborn whitegrid style not found, using default.")
            pass 

    fig, axs = plt.subplots(3, 1, figsize=(max(10, num_subjects * 1.0 + 2), 20), sharex=False) # Increased height a bit

    # Font sizes
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12

    # --- Plot 1: Phase 2 Self-Accuracy Lift ---
    yerr_acc_lift_low = np.nan_to_num(df_results["Accuracy Lift (P2-P1)"] - df_results["AccLift_CI_Low"], nan=0.0)
    yerr_acc_lift_high = np.nan_to_num(df_results["AccLift_CI_High"] - df_results["Accuracy Lift (P2-P1)"], nan=0.0)
    yerr_acc_lift_low[yerr_acc_lift_low < 0] = 0
    yerr_acc_lift_high[yerr_acc_lift_high < 0] = 0
    
    axs[0].bar(formatted_subject_names, df_results["Accuracy Lift (P2-P1)"], 
               color='lightcoral', # Removed label
               yerr=[yerr_acc_lift_low, yerr_acc_lift_high], ecolor='gray', capsize=5, width=0.6)
    axs[0].set_ylabel('Accuracy Difference (P2 Acc - P1 Acc)', fontsize=label_fontsize)
    axs[0].set_title('Phase 2 Self-Accuracy Lift by Subject (95% CI)', fontsize=title_fontsize)
    axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    # axs[0].legend() # Removed legend
    axs[0].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    axs[0].tick_params(axis='y', labelsize=tick_fontsize)

    # --- Plot 2: Preferential Delegating Ratio ---
    yerr_ratio_low = np.nan_to_num(df_results["Delegating Ratio"] - df_results["DelRatio_CI_Low"], nan=0.0)
    yerr_ratio_high = np.nan_to_num(df_results["DelRatio_CI_High"] - df_results["Delegating Ratio"], nan=0.0)
    yerr_ratio_low[yerr_ratio_low < 0] = 0 
    yerr_ratio_high[yerr_ratio_high < 0] = 0

    axs[1].bar(formatted_subject_names, df_results["Delegating Ratio"], 
               color='skyblue', # <<<--- REMOVED label= HERE
               yerr=[yerr_ratio_low, yerr_ratio_high], ecolor='gray', capsize=5, width=0.6)
    axs[1].set_ylabel('Ratio P(Del|S_i=0) / P(Del|S_i=1)', fontsize=label_fontsize)
    axs[1].set_title('Preferential Delegating Ratio by Subject (95% CI)', fontsize=title_fontsize)
    axs[1].axhline(1, color='red', linestyle='--', linewidth=0.8, label='Ratio = 1 (No Preference)') # This label WILL appear
    axs[1].legend(fontsize=legend_fontsize) # This will now only pick up the axhline label
    axs[1].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    axs[1].tick_params(axis='y', labelsize=tick_fontsize)


    # --- Plot 3: Reversed s_i_capability Coefficient ---
    yerr_si_low = np.nan_to_num(df_results["Reversed SI Coef"] - df_results["RevSICoef_CI_Low"], nan=0.0)
    yerr_si_high = np.nan_to_num(df_results["RevSICoef_CI_High"] - df_results["Reversed SI Coef"], nan=0.0)
    yerr_si_low[yerr_si_low < 0] = 0
    yerr_si_high[yerr_si_high < 0] = 0
    
    axs[2].bar(formatted_subject_names, df_results["Reversed SI Coef"], 
               color='mediumseagreen', # Removed label
               yerr=[yerr_si_low, yerr_si_high], ecolor='gray', capsize=5, width=0.6)
    axs[2].set_ylabel('Coefficient Value (Log-Odds Scale)', fontsize=label_fontsize)
    axs[2].set_title('Reversed s_i_capability Coefficient (Model 4, Adjusted) by Subject (95% CI)', fontsize=title_fontsize)
    axs[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
    # axs[2].legend() # Removed legend
    axs[2].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    axs[2].tick_params(axis='y', labelsize=tick_fontsize)

    # Common x-label for the whole figure (if desired, but usually per-subplot is fine)
    # fig.text(0.5, 0.01, 'Subject', ha='center', va='center', fontsize=label_fontsize)

    plt.tight_layout(pad=3.0, h_pad=4.0) # Adjust h_pad for vertical spacing if titles/xlabels overlap
    plt.savefig("subject_analysis_charts.png", dpi=300)
    print("Charts saved to subject_analysis_charts.png")
    plt.show()

if __name__ == "__main__":
    input_log_filename = "analysis_log_multi_logres_dg_gpqa.txt"
    output_filename = f"{input_log_filename.split('.')[0]}_parsed.txt"
    try:
        with open(input_log_filename, 'r', encoding='utf-8') as f:
            log_content_from_file = f.read()
        parse_analysis_log(log_content_from_file, output_filename)

        df_results = analyze_parsed_data(output_filename)
        print("\n--- Calculated Data ---")
        print(df_results.to_string()) # Print full DataFrame
        
        model_list = ['grok-3-latest', 'gemini-1.5-pro', 'claude-3-7-sonnet-20250219', 'gemini-2.0-flash-001', 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'gpt-4-turbo-2024-04-09', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
        model_list = ['claude-3-7-sonnet-20250219', 'grok-3-latest', 'claude-3-5-sonnet-20241022', 'gemini-2.0-flash-001', 'gemini-1.5-pro', 'claude-3-opus-20240229', 'gpt-4-turbo-2024-04-09', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
        model_list = ['claude-3-7-sonnet-20250219', 'grok-3-latest', 'gemini-2.0-flash-001', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro', 'claude-3-opus-20240229', 'gpt-4-turbo-2024-04-09', 'claude-3-haiku-20240307', 'claude-3-sonnet-20240229']
        if not df_results.empty:
            plot_results(df_results, model_list)
        else:
            print("No results to plot.")

    except FileNotFoundError:
        print(f"Error: Input log file '{input_log_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        