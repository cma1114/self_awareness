import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
import warnings

# Z-score for 95% CI
Z_SCORE = norm.ppf(0.975)

def parse_analysis_log(log_content, output_file, target_params, model_list, int_score_type="adjusted", lift_score_type="adjusted"):
    
    block_start_regex = re.compile(
        r"--- Analyzing (\S+) \(" + re.escape(target_params) + r", \d+ game files\) ---"
    )

    adj_introspection_regex = re.compile(r"Adjusted introspection score = ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]")
    raw_introspection_regex = re.compile(r"Introspection score = ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]")
    filtered_introspection_regex = re.compile(r"Filtered Introspection score = ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]")
    
    adj_self_acc_lift_regex = re.compile(r"Adjusted self-acc lift = ([-\d.]+)\s*\[([-\d.]+), ([-\d.]+)")
    raw_self_acc_lift_regex = re.compile(r"Self-acc lift = ([-\d.]+)\s*\[([-\d.]+), ([-\d.]+)")
    filtered_self_acc_lift_regex = re.compile(r"Filtered Self-acc lift = ([-\d.]+)\s*\[([-\d.]+), ([-\d.]+)")

    game_test_change_regex = re.compile(r"Game-Test Change Rate: ([-\d.]+)")

    fp_regex = re.compile(r"FP = ([-\d.]+)")
    fn_regex = re.compile(r"FN = ([-\d.]+)")

    if int_score_type == "adjusted":
        introspection_regex = adj_introspection_regex
        prefix_int = "adj"
    elif int_score_type == "filtered":
        introspection_regex = filtered_introspection_regex
        prefix_int = "filt"
    else: # raw
        introspection_regex = raw_introspection_regex
        prefix_int = "raw"

    if lift_score_type == "adjusted":
        self_acc_lift_regex = adj_self_acc_lift_regex
        prefix_lift = "adj"
    elif lift_score_type == "filtered":
        self_acc_lift_regex = filtered_self_acc_lift_regex
        prefix_lift = "filt"
    else: # raw
        self_acc_lift_regex = raw_self_acc_lift_regex
        prefix_lift = "raw"
    
    # Model section identifiers
    model4_start_regex = re.compile(r"^\s*Model 4.*\(No Interactions\).*:\s*delegate_choice ~")
    model46_start_regex = re.compile(r"^\s*Model 4\.6.*:\s*delegate_choice ~")
    model48_start_regex = re.compile(r"^\s*Model 4\.8.*:\s*delegate_choice ~")
    model7_start_regex = re.compile(r"^\s*Model 7.*:\s*delegate_choice ~")
    
    # Logit regression results marker
    logit_results_regex = re.compile(r"^\s*Logit Regression Results\s*$")
    
    # Coefficient extraction regexes
    si_capability_coef_regex = re.compile(r"^\s*s_i_capability\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
    capabilities_entropy_coef_regex = re.compile(r"^\s*capabilities_entropy\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
    normalized_prob_entropy_coef_regex = re.compile(r"^\s*normalized_prob_entropy\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
    
    # Log-likelihood regex
    log_likelihood_regex = re.compile(r"Log-Likelihood:\s*([-\d.]+)")

    # Cross-tabulation regexes 
    crosstab_title_regex = re.compile(r"^\s*Cross-tabulation of delegate_choice vs\. s_i_capability:$")
    crosstab_col_header_regex = re.compile(r"^\s*s_i_capability\s+\S+\s+\S+")
    crosstab_row_header_label_regex = re.compile(r"^\s*delegate_choice\s*$")
    crosstab_data_row_regex = re.compile(r"^\s*\d+\s+(\d+)\s+(\d+)\s*$")

    analysis_blocks = re.split(r"(?=--- Analyzing )", log_content)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for block_content in analysis_blocks:
            if not block_content.strip():
                continue

            match_block_start = block_start_regex.search(block_content)
            if match_block_start:
                subject_name = match_block_start.group(1)
                if subject_name not in model_list:
                    print(f"Skipping subject {subject_name} as it is not in the provided model list.")
                    continue
                
                outfile.write(f"Subject: {subject_name}\n")
                
                extracted_info = {
                    f"{prefix_int}_introspection": "Not found",
                    f"{prefix_int}_introspection_ci_low": "Not found",
                    f"{prefix_int}_introspection_ci_high": "Not found",
                    f"{prefix_lift}_self_acc_lift": "Not found",
                    f"{prefix_lift}_self_acc_lift_ci_low": "Not found",
                    f"{prefix_lift}_self_acc_lift_ci_high": "Not found",
                    "model4_si_cap_coef": "Not found",
                    "model4_si_cap_ci_low": "Not found",
                    "model4_si_cap_ci_high": "Not found",
                    "model4_log_lik": "Not found",
                    "model46_cap_entropy_coef": "Not found",
                    "model46_cap_entropy_ci_low": "Not found",
                    "model46_cap_entropy_ci_high": "Not found",
                    "model48_norm_prob_entropy_coef": "Not found",
                    "model48_norm_prob_entropy_ci_low": "Not found",
                    "model48_norm_prob_entropy_ci_high": "Not found",
                    "model7_log_lik": "Not found",
                    "delegation_rate": "Not found",
                    "phase1_accuracy": "Not found",
                    "total_n": "Not found",
                    "game_test_change_rate": "Not found",
                    "fp": "Not found",
                    "fn": "Not found"
                }
                
                # Model parsing states
                in_model4 = False
                in_model46 = False
                in_model48 = False
                in_model7 = False
                found_logit_results = False

                # --- Cross-tab parsing state ---
                parsing_crosstab = False
                expecting_crosstab_col_header = False
                expecting_crosstab_row_header_label = False
                crosstab_data_lines_collected = 0
                temp_crosstab_cells = []

                lines = block_content.splitlines()
                for i, line in enumerate(lines):
                    # Extract adjusted introspection score
                    m = introspection_regex.search(line)
                    if m:
                        extracted_info[f"{prefix_int}_introspection"] = m.group(1)
                        extracted_info[f"{prefix_int}_introspection_ci_low"] = m.group(2)
                        extracted_info[f"{prefix_int}_introspection_ci_high"] = m.group(3)
                        continue
                    
                    # Extract adjusted self-acc lift
                    m = self_acc_lift_regex.search(line)
                    if m:
                        extracted_info[f"{prefix_lift}_self_acc_lift"] = m.group(1)
                        extracted_info[f"{prefix_lift}_self_acc_lift_ci_low"] = m.group(2)
                        extracted_info[f"{prefix_lift}_self_acc_lift_ci_high"] = m.group(3)
                        continue

                    # Extract game test change rate
                    m = game_test_change_regex.search(line)
                    if m:
                        extracted_info["game_test_change_rate"] = m.group(1)
                        continue

                    # Extract FP and FN
                    m_fp = fp_regex.search(line)
                    if m_fp:
                        extracted_info["fp"] = m_fp.group(1)
                        continue    
                    m_fn = fn_regex.search(line)
                    if m_fn:
                        extracted_info["fn"] = m_fn.group(1)
                        continue

                    # Cross-tabulation parsing state machine
                    if not parsing_crosstab and not any([in_model4, in_model46, in_model48, in_model7]) and crosstab_title_regex.match(line):
                        parsing_crosstab = True
                        expecting_crosstab_col_header = True
                        expecting_crosstab_row_header_label = False
                        crosstab_data_lines_collected = 0
                        temp_crosstab_cells = []
                        continue

                    if parsing_crosstab:
                        if expecting_crosstab_col_header and crosstab_col_header_regex.match(line):
                            expecting_crosstab_col_header = False
                            expecting_crosstab_row_header_label = True
                            continue
                        elif expecting_crosstab_row_header_label and crosstab_row_header_label_regex.match(line):
                            expecting_crosstab_row_header_label = False
                            continue
                        else:
                            data_match = crosstab_data_row_regex.match(line)
                            if data_match:
                                temp_crosstab_cells.append(int(data_match.group(1)))
                                temp_crosstab_cells.append(int(data_match.group(2)))
                                crosstab_data_lines_collected += 1
                                if crosstab_data_lines_collected == 2 and len(temp_crosstab_cells) == 4:
                                    # Calculate delegation rate, phase 1 accuracy, and total N
                                    # temp_crosstab_cells = [row0_col0, row0_col1, row1_col0, row1_col1]
                                    total_n = sum(temp_crosstab_cells)
                                    delegation_rate = (temp_crosstab_cells[2] + temp_crosstab_cells[3]) / total_n if total_n > 0 else 0
                                    phase1_accuracy = (temp_crosstab_cells[1] + temp_crosstab_cells[3]) / total_n if total_n > 0 else 0
                                    
                                    extracted_info["delegation_rate"] = str(delegation_rate)
                                    extracted_info["phase1_accuracy"] = str(phase1_accuracy)
                                    extracted_info["total_n"] = str(total_n)
                                    parsing_crosstab = False
                                continue
                            # blank or unexpected line ends crosstab
                            if not line.strip():
                                parsing_crosstab = False
                            continue

                    # Check for model starts
                    if model4_start_regex.search(line):
                        in_model4 = True
                        in_model46 = False
                        in_model48 = False
                        in_model7 = False
                        found_logit_results = False
                        parsing_crosstab = False
                        continue
                    elif model46_start_regex.search(line):
                        in_model4 = False
                        in_model46 = True
                        in_model48 = False
                        in_model7 = False
                        found_logit_results = False
                        parsing_crosstab = False
                        continue
                    elif model48_start_regex.search(line):
                        in_model4 = False
                        in_model46 = False
                        in_model48 = True
                        in_model7 = False
                        found_logit_results = False
                        parsing_crosstab = False
                        continue
                    elif model7_start_regex.search(line):
                        in_model4 = False
                        in_model46 = False
                        in_model48 = False
                        in_model7 = True
                        found_logit_results = False
                        parsing_crosstab = False
                        continue
                    
                    # Check for Logit Regression Results
                    if not parsing_crosstab and logit_results_regex.match(line):
                        found_logit_results = True
                        continue
                    
                    # Extract coefficients and log-likelihood based on current model
                    if not parsing_crosstab:
                        if in_model4 and found_logit_results:
                            # Look for s_i_capability coefficient
                            m = si_capability_coef_regex.match(line)
                            if m:
                                extracted_info["model4_si_cap_coef"] = m.group(1)
                                extracted_info["model4_si_cap_ci_low"] = m.group(5)
                                extracted_info["model4_si_cap_ci_high"] = m.group(6)
                            
                            # Look for log-likelihood
                            m = log_likelihood_regex.search(line)
                            if m:
                                extracted_info["model4_log_lik"] = m.group(1)
                        
                        elif in_model46 and found_logit_results:
                            # Look for capabilities_entropy coefficient
                            m = capabilities_entropy_coef_regex.match(line)
                            if m:
                                extracted_info["model46_cap_entropy_coef"] = m.group(1)
                                extracted_info["model46_cap_entropy_ci_low"] = m.group(5)
                                extracted_info["model46_cap_entropy_ci_high"] = m.group(6)
                        
                        elif in_model48 and found_logit_results:
                            # Look for normalized_prob_entropy coefficient
                            m = normalized_prob_entropy_coef_regex.match(line)
                            if m:
                                extracted_info["model48_norm_prob_entropy_coef"] = m.group(1)
                                extracted_info["model48_norm_prob_entropy_ci_low"] = m.group(5)
                                extracted_info["model48_norm_prob_entropy_ci_high"] = m.group(6)
                        
                        elif in_model7 and found_logit_results:
                            # Look for log-likelihood
                            m = log_likelihood_regex.search(line)
                            if m:
                                extracted_info["model7_log_lik"] = m.group(1)
                    
                        # Reset state if we see a new model or section
                        if line.strip().startswith("Model ") and not any([
                            model4_start_regex.search(line),
                            model46_start_regex.search(line),
                            model48_start_regex.search(line),
                            model7_start_regex.search(line)
                        ]):
                            in_model4 = in_model46 = in_model48 = in_model7 = False
                
                # Validate required fields and write output
                if extracted_info["model4_si_cap_coef"] == "Not found":
                    raise ValueError(f"Model 4 s_i_capability coefficient not found for {subject_name}. Check that Model 4 has Logit Regression Results.")
                if extracted_info["model4_log_lik"] == "Not found":
                    raise ValueError(f"Model 4 Log-Likelihood not found for {subject_name}")
                if extracted_info["model7_log_lik"] == "Not found":
                    raise ValueError(f"Model 7 Log-Likelihood not found for {subject_name}. Check that Model 7 has Logit Regression Results.")
                
                # Warnings for optional fields
                if extracted_info["model46_cap_entropy_coef"] == "Not found":
                    pass#print(f"Warning: Model 4.6 capabilities_entropy coefficient not found for {subject_name}")
                if extracted_info["model48_norm_prob_entropy_coef"] == "Not found":
                    pass#print(f"Warning: Model 4.8 normalized_prob_entropy coefficient not found for {subject_name}")
                
                # Write extracted info
                if int_score_type == "adjusted":
                    prefix_int_cln = "Adjusted "
                elif int_score_type == "filtered":
                    prefix_int_cln = "Filtered "
                else:
                    prefix_int_cln = "Raw "
                
                if lift_score_type == "adjusted":
                    prefix_lift_cln = "Adjusted "
                elif lift_score_type == "filtered":
                    prefix_lift_cln = "Filtered "
                else:
                    prefix_lift_cln = "Raw "
                    
                outfile.write(f"  {prefix_int_cln}introspection score: {extracted_info[f'{prefix_int}_introspection']} [{extracted_info[f'{prefix_int}_introspection_ci_low']}, {extracted_info[f'{prefix_int}_introspection_ci_high']}]\n")
                outfile.write(f"  {prefix_lift_cln}self-acc lift: {extracted_info[f'{prefix_lift}_self_acc_lift']} [{extracted_info[f'{prefix_lift}_self_acc_lift_ci_low']}, {extracted_info[f'{prefix_lift}_self_acc_lift_ci_high']}]\n")
                outfile.write(f"  Model 4 s_i_capability: {extracted_info['model4_si_cap_coef']} [{extracted_info['model4_si_cap_ci_low']}, {extracted_info['model4_si_cap_ci_high']}]\n")
                outfile.write(f"  Model 4 Log-Likelihood: {extracted_info['model4_log_lik']}\n")
                outfile.write(f"  Model 4.6 capabilities_entropy: {extracted_info['model46_cap_entropy_coef']} [{extracted_info['model46_cap_entropy_ci_low']}, {extracted_info['model46_cap_entropy_ci_high']}]\n")
                outfile.write(f"  Model 4.8 normalized_prob_entropy: {extracted_info['model48_norm_prob_entropy_coef']} [{extracted_info['model48_norm_prob_entropy_ci_low']}, {extracted_info['model48_norm_prob_entropy_ci_high']}]\n")
                outfile.write(f"  Model 7 Log-Likelihood: {extracted_info['model7_log_lik']}\n")
                outfile.write(f"  Delegation rate: {extracted_info['delegation_rate']}\n")
                outfile.write(f"  Phase 1 accuracy: {extracted_info['phase1_accuracy']}\n")
                outfile.write(f"  Total N: {extracted_info['total_n']}\n")
                outfile.write(f"  Game-Test Change Rate: {extracted_info['game_test_change_rate']}\n")
                outfile.write(f"  FP: {extracted_info['fp']}\n")
                outfile.write(f"  FN: {extracted_info['fn']}\n")
                outfile.write("\n")

    print(f"Parsing complete. Output written to {output_file}")


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


def analyze_parsed_data(input_summary_file):
    all_subject_data = []
    current_subject_info = {}

    with open(input_summary_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("Subject:"):
                if current_subject_info.get("subject_name"):
                    all_subject_data.append(current_subject_info)
                current_subject_info = {"subject_name": line.split("Subject:")[1].strip()}
            elif "introspection score:" in line:
                # Parse: "Adjusted introspection score: 0.167 [0.070, 0.262]"
                if "Adjusted" in line:
                    prefix_int = "adj"
                elif "Filtered" in line:
                    prefix_int = "filt"
                else:
                    prefix_int = "raw"
                m = re.search(r":\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]", line)
                if m:
                    current_subject_info[f"{prefix_int}_introspection"] = float(m.group(1))
                    current_subject_info[f"{prefix_int}_introspection_ci_low"] = float(m.group(2))
                    current_subject_info[f"{prefix_int}_introspection_ci_high"] = float(m.group(3))
            elif "self-acc lift:" in line:
                # Parse: "Adjusted self-acc lift: 0.178 [0.062, 0.280]"
                if "Adjusted" in line:
                    prefix_lift = "adj"
                elif "Filtered" in line:
                    prefix_lift = "filt"
                else:
                    prefix_lift = "raw"
                m = re.search(r":\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]", line)
                if m:
                    current_subject_info[f"{prefix_lift}_self_acc_lift"] = float(m.group(1))
                    current_subject_info[f"{prefix_lift}_self_acc_lift_ci_low"] = float(m.group(2))
                    current_subject_info[f"{prefix_lift}_self_acc_lift_ci_high"] = float(m.group(3))
            elif "Model 4 s_i_capability:" in line:
                # Parse: "Model 4 s_i_capability: -0.8796 [-1.451, -0.309]"
                m = re.search(r":\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]", line)
                if m:
                    current_subject_info["si_coef"] = float(m.group(1))
                    current_subject_info["si_coef_ci_low"] = float(m.group(2))
                    current_subject_info["si_coef_ci_high"] = float(m.group(3))
            elif "Model 4 Log-Likelihood:" in line:
                current_subject_info["loglik4"] = parse_value(line, r":\s*([-\d.]+)", as_type=float)
            elif "Model 4.6 capabilities_entropy:" in line:
                # Parse: "Model 4.6 capabilities_entropy: 2.7523 [1.396, 4.109]"
                m = re.search(r":\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]", line)
                if m:
                    current_subject_info["cap_entropy_coef"] = float(m.group(1))
                    current_subject_info["cap_entropy_ci_low"] = float(m.group(2))
                    current_subject_info["cap_entropy_ci_high"] = float(m.group(3))
            elif "Model 4.8 normalized_prob_entropy:" in line:
                # Parse: "Model 4.8 normalized_prob_entropy: 4.8797 [2.541, 7.218]"
                m = re.search(r":\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]", line)
                if m:
                    current_subject_info["norm_prob_entropy_coef"] = float(m.group(1))
                    current_subject_info["norm_prob_entropy_ci_low"] = float(m.group(2))
                    current_subject_info["norm_prob_entropy_ci_high"] = float(m.group(3))
            elif "Model 7 Log-Likelihood:" in line:
                current_subject_info["loglik7"] = parse_value(line, r":\s*([-\d.]+)", as_type=float)
            elif "Delegation rate:" in line:
                current_subject_info["delegation_rate"] = parse_value(line, r":\s*([-\d.]+)", as_type=float)
            elif "Phase 1 accuracy:" in line:
                current_subject_info["phase1_accuracy"] = parse_value(line, r":\s*([-\d.]+)", as_type=float)
            elif "Total N:" in line:
                current_subject_info["total_n"] = parse_value(line, r":\s*(\d+)", as_type=int)
            elif "Game-Test Change Rate:" in line:
                current_subject_info["game_test_change_rate"] = parse_value(line, r":\s*([-\d.]+)", as_type=float)
            elif "FP:" in line:
                current_subject_info["fp"] = parse_value(line, r":\s*([-\d.]+)", as_type=float)
            elif "FN:" in line:
                current_subject_info["fn"] = parse_value(line, r":\s*([-\d.]+)", as_type=float)

        if current_subject_info.get("subject_name"):
            all_subject_data.append(current_subject_info)

    results = []
    for data in all_subject_data:
        subject_name = data.get("subject_name", "Unknown")
        
        # Get all the values, using np.nan for missing optional values
        # Determine prefixes based on what was actually parsed (could be mixed if file was appended)
        # This part assumes that the prefixes determined during file parsing (adj, raw, filt) are consistent
        # for introspection and lift within a single subject block in the *parsed file*.
        # We need to find which prefix was used for this subject's introspection and lift.
        
        # Try to infer the prefix used for introspection for this subject
        current_prefix_int = "adj" # default
        if f"adj_introspection" in data:
            current_prefix_int = "adj"
        elif f"filt_introspection" in data:
            current_prefix_int = "filt"
        elif f"raw_introspection" in data:
            current_prefix_int = "raw"
            
        current_prefix_lift = "adj" # default
        if f"adj_self_acc_lift" in data:
            current_prefix_lift = "adj"
        elif f"filt_self_acc_lift" in data:
            current_prefix_lift = "filt"
        elif f"raw_self_acc_lift" in data:
            current_prefix_lift = "raw"

        introspection_val = data.get(f"{current_prefix_int}_introspection", np.nan)
        introspection_ci_low = data.get(f"{current_prefix_int}_introspection_ci_low", np.nan)
        introspection_ci_high = data.get(f"{current_prefix_int}_introspection_ci_high", np.nan)
        
        self_acc_lift_val = data.get(f"{current_prefix_lift}_self_acc_lift", np.nan)
        self_acc_lift_ci_low = data.get(f"{current_prefix_lift}_self_acc_lift_ci_low", np.nan)
        self_acc_lift_ci_high = data.get(f"{current_prefix_lift}_self_acc_lift_ci_high", np.nan)
        
        si_coef = data.get("si_coef", np.nan)
        si_ci_low = data.get("si_coef_ci_low", np.nan)
        si_ci_high = data.get("si_coef_ci_high", np.nan)
        
        # Reverse the sign of SI coefficient as in original code
        rev_si_coef = -1 * si_coef if not np.isnan(si_coef) else np.nan
        rev_si_ci_low = -1 * si_ci_high if not np.isnan(si_ci_high) else np.nan
        rev_si_ci_high = -1 * si_ci_low if not np.isnan(si_ci_low) else np.nan
        
        cap_entropy_coef = data.get("cap_entropy_coef", np.nan)
        cap_entropy_ci_low = data.get("cap_entropy_ci_low", np.nan)
        cap_entropy_ci_high = data.get("cap_entropy_ci_high", np.nan)
        
        norm_prob_entropy_coef = data.get("norm_prob_entropy_coef", np.nan)
        norm_prob_entropy_ci_low = data.get("norm_prob_entropy_ci_low", np.nan)
        norm_prob_entropy_ci_high = data.get("norm_prob_entropy_ci_high", np.nan)
        
        LL4 = data.get("loglik4", np.nan)
        LL7 = data.get("loglik7", np.nan)
        
        # Calculate likelihood ratio test
        LR_stat = 2 * (LL4 - LL7) if not np.isnan(LL4) and not np.isnan(LL7) else np.nan
        LR_pvalue = chi2.sf(LR_stat, df=1) if not np.isnan(LR_stat) else np.nan

        # Get delegation rate, phase 1 accuracy, and total N
        delegation_rate = data.get("delegation_rate", np.nan)
        phase1_accuracy = data.get("phase1_accuracy", np.nan)
        total_n = data.get("total_n", np.nan)

        results.append({
            "Subject": subject_name,
            f"{current_prefix_int.capitalize()}Intro": introspection_val,
            f"{current_prefix_int.capitalize()}Intro_LB": introspection_ci_low,
            f"{current_prefix_int.capitalize()}Intro_UB": introspection_ci_high,
            f"{current_prefix_lift.capitalize()}AccLift": self_acc_lift_val,
            f"{current_prefix_lift.capitalize()}AccLift_LB": self_acc_lift_ci_low,
            f"{current_prefix_lift.capitalize()}AccLift_UB": self_acc_lift_ci_high,
            "CapCoef": rev_si_coef,
            "CapCoef_LB": rev_si_ci_low,
            "CapCoef_UB": rev_si_ci_high,
            "CapEnt": cap_entropy_coef,
            "CapEnt_LB": cap_entropy_ci_low,
            "CapEnt_UB": cap_entropy_ci_high,
            "GameEnt": norm_prob_entropy_coef,
            "GameEnt_LB": norm_prob_entropy_ci_low,
            "GameEnt_UB": norm_prob_entropy_ci_high,
            "LL_Model4": LL4,
            "LL_Model7": LL7,
            "LR_stat": LR_stat,
            "LR_pvalue": LR_pvalue,
            "Delegation_Rate": delegation_rate,
            "Phase1_Accuracy": phase1_accuracy,
            "Total_N": total_n,
            "Change%": data.get("game_test_change_rate", np.nan),
            "FP": data.get("fp", np.nan),
            "FN": data.get("fn", np.nan)
        })
        
    return pd.DataFrame(results)


def break_subject_name(name, max_parts_per_line=3):
    """Breaks a subject name string by hyphens for better display."""
    parts = name.split('-')
    if len(parts) <= max_parts_per_line:
        return name
    
    wrapped_name = ""
    for i, part in enumerate(parts):
        wrapped_name += part
        if (i + 1) % max_parts_per_line == 0 and (i + 1) < len(parts):
            wrapped_name += "-\n"
        elif (i + 1) < len(parts):
            wrapped_name += "-"
    return wrapped_name


def plot_results(df_results, subject_order=None, dataset_name="GPQA", int_score_type="adjusted", lift_score_type="adjusted"):
    if int_score_type == "adjusted":
        prefix_int = "Adj"
        prefix_int_cln = "Adjusted "
    elif int_score_type == "filtered":
        prefix_int = "Filt"
        prefix_int_cln = "Filtered "
    else: # raw
        prefix_int = "Raw"
        prefix_int_cln = "Raw "

    if lift_score_type == "adjusted":
        prefix_lift = "Adj"
        prefix_lift_cln = "Adjusted "
    elif lift_score_type == "filtered":
        prefix_lift = "Filt"
        prefix_lift_cln = "Filtered "
    else: # raw
        prefix_lift = "Raw"
        prefix_lift_cln = "Raw "
        
    if df_results.empty:
        print("No data to plot.")
        return

    # Reorder DataFrame if subject_order is provided
    if subject_order:
        df_results_ordered = df_results.copy()
        df_results_ordered['Subject_Cat'] = pd.Categorical(df_results_ordered['Subject'], categories=subject_order, ordered=True)
        df_results_ordered = df_results_ordered[df_results_ordered['Subject_Cat'].notna()].sort_values('Subject_Cat')
        if df_results_ordered.empty and not df_results.empty:
            print("Warning: None of the subjects in subject_order were found in the data. Plotting all available subjects.")
        elif len(df_results_ordered) < len(df_results):
             print(f"Warning: Plotting only {len(df_results_ordered)} subjects present in the provided order list.")
             df_results = df_results_ordered.drop(columns=['Subject_Cat'])
        else:
            df_results = df_results_ordered.drop(columns=['Subject_Cat'])

    num_subjects = len(df_results)
    if num_subjects == 0:
        print("No subjects to plot after filtering/ordering.")
        return

    # Check if we have entropy coefficients
    has_cap_entropy = not df_results["CapEnt"].isna().all()
    has_norm_prob_entropy = not df_results["GameEnt"].isna().all()
    has_entropy = has_cap_entropy or has_norm_prob_entropy
    
    # Determine number of columns
    ncols = 2 if has_entropy else 1
    
    # Apply name breaking for x-axis labels
    formatted_subject_names = [break_subject_name(name, max_parts_per_line=3) for name in df_results["Subject"]]

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            print("Seaborn whitegrid style not found, using default.")
            pass

    # Create figure with appropriate number of subplots
    if has_entropy:
        fig, axs = plt.subplots(3, 2, figsize=(max(20, num_subjects * 2.0 + 4), 20))
        axs = axs.reshape(3, 2)  # Ensure it's a 2D array
    else:
        fig, axs = plt.subplots(3, 1, figsize=(max(10, num_subjects * 1.0 + 2), 20))
        axs = axs.reshape(3, 1)  # Make it 2D for consistent indexing

    # Font sizes
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12

    # --- Plot 1: Adjusted Introspection Score ---
    yerr_intro_low = np.nan_to_num(df_results[f"{prefix_int}Intro"] - df_results[f"{prefix_int}Intro_LB"], nan=0.0)
    yerr_intro_high = np.nan_to_num(df_results[f"{prefix_int}Intro_UB"] - df_results[f"{prefix_int}Intro"], nan=0.0)
    yerr_intro_low[yerr_intro_low < 0] = 0
    yerr_intro_high[yerr_intro_high < 0] = 0
    
    axs[0, 0].bar(formatted_subject_names, df_results[f"{prefix_int}Intro"],
                   color='mediumpurple',
                   yerr=[yerr_intro_low, yerr_intro_high], ecolor='gray', capsize=5, width=0.6)
    axs[0, 0].set_ylabel('Introspection Score', fontsize=label_fontsize)
    axs[0, 0].set_title(f'{prefix_int_cln}Introspection Score by LLM (95% CI)', fontsize=title_fontsize)
    axs[0, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[0, 0].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    axs[0, 0].tick_params(axis='y', labelsize=tick_fontsize)

    # --- Plot 2: Reversed s_i_capability Coefficient ---
    yerr_si_low = np.nan_to_num(df_results["CapCoef"] - df_results["CapCoef_LB"], nan=0.0)
    yerr_si_high = np.nan_to_num(df_results["CapCoef_UB"] - df_results["CapCoef"], nan=0.0)
    yerr_si_low[yerr_si_low < 0] = 0
    yerr_si_high[yerr_si_high < 0] = 0
    
    axs[1, 0].bar(formatted_subject_names, df_results["CapCoef"],
                   color='mediumseagreen',
                   yerr=[yerr_si_low, yerr_si_high], ecolor='gray', capsize=5, width=0.6)
    axs[1, 0].set_ylabel('Coefficient Value (Log-Odds Scale)', fontsize=label_fontsize)
    axs[1, 0].set_title('Introspection Coefficient by LLM (95% CI)', fontsize=title_fontsize)
    axs[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[1, 0].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    axs[1, 0].tick_params(axis='y', labelsize=tick_fontsize)

    # --- Plot 3: Adjusted Self-Accuracy Lift ---
    yerr_acc_lift_low = np.nan_to_num(df_results[f"{prefix_lift}AccLift"] - df_results[f"{prefix_lift}AccLift_LB"], nan=0.0)
    yerr_acc_lift_high = np.nan_to_num(df_results[f"{prefix_lift}AccLift_UB"] - df_results[f"{prefix_lift}AccLift"], nan=0.0)
    yerr_acc_lift_low[yerr_acc_lift_low < 0] = 0
    yerr_acc_lift_high[yerr_acc_lift_high < 0] = 0
    
    axs[2, 0].bar(formatted_subject_names, df_results[f"{prefix_lift}AccLift"],
                   color='lightcoral',
                   yerr=[yerr_acc_lift_low, yerr_acc_lift_high], ecolor='gray', capsize=5, width=0.6)
    axs[2, 0].set_ylabel('Accuracy Difference', fontsize=label_fontsize)
    axs[2, 0].set_title(f'{prefix_lift_cln}Self-Accuracy Lift by LLM (95% CI)', fontsize=title_fontsize)
    axs[2, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[2, 0].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    axs[2, 0].tick_params(axis='y', labelsize=tick_fontsize)

    # If we have entropy coefficients, plot them in the second column
    if has_entropy:
        # Hide the first row, second column (no plot there)
        axs[0, 1].axis('off')
        
        if has_cap_entropy:
            # --- Plot 4: Capabilities Entropy Coefficient ---
            yerr_cap_low = np.nan_to_num(df_results["CapEnt"] - df_results["CapEnt_LB"], nan=0.0)
            yerr_cap_high = np.nan_to_num(df_results["CapEnt_UB"] - df_results["CapEnt"], nan=0.0)
            yerr_cap_low[yerr_cap_low < 0] = 0
            yerr_cap_high[yerr_cap_high < 0] = 0
            
            axs[1, 1].bar(formatted_subject_names, df_results["CapEnt"],
                           color='cornflowerblue',
                           yerr=[yerr_cap_low, yerr_cap_high], ecolor='gray', capsize=5, width=0.6)
            axs[1, 1].set_ylabel('Coefficient Value', fontsize=label_fontsize)
            axs[1, 1].set_title('Capabilities Entropy Coefficient (Model 4.6)', fontsize=title_fontsize)
            axs[1, 1].axhline(0, color='black', linestyle='--', linewidth=0.8)
            axs[1, 1].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
            axs[1, 1].tick_params(axis='y', labelsize=tick_fontsize)
        else:
            axs[1, 1].axis('off')
        
        if has_norm_prob_entropy:
            # --- Plot 5: Normalized Probability Entropy Coefficient ---
            yerr_norm_low = np.nan_to_num(df_results["GameEnt"] - df_results["GameEnt_LB"], nan=0.0)
            yerr_norm_high = np.nan_to_num(df_results["GameEnt_UB"] - df_results["GameEnt"], nan=0.0)
            yerr_norm_low[yerr_norm_low < 0] = 0
            yerr_norm_high[yerr_norm_high < 0] = 0
            
            axs[2, 1].bar(formatted_subject_names, df_results["GameEnt"],
                           color='darkorange',
                           yerr=[yerr_norm_low, yerr_norm_high], ecolor='gray', capsize=5, width=0.6)
            axs[2, 1].set_ylabel('Coefficient Value', fontsize=label_fontsize)
            axs[2, 1].set_title('Normalized Probability Entropy Coefficient (Model 4.8)', fontsize=title_fontsize)
            axs[2, 1].axhline(0, color='black', linestyle='--', linewidth=0.8)
            axs[2, 1].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
            axs[2, 1].tick_params(axis='y', labelsize=tick_fontsize)
        else:
            axs[2, 1].axis('off')

    plt.tight_layout(pad=3.0, h_pad=4.0)
    plt.savefig(f"subject_analysis_charts_{dataset_name}_{prefix_int.lower()}_{prefix_lift.lower()}.png", dpi=300)
    print(f"Charts saved to subject_analysis_charts_{dataset_name}_{prefix_int.lower()}_{prefix_lift.lower()}.png")


if __name__ == "__main__":
    
    game_type = "dg"#"aop" #
    dataset = "SimpleQA"
    if game_type == "dg":
        target_params = "Feedback_False, Non_Redacted, NoSubjAccOverride, NoSubjGameOverride, NotRandomized, WithHistory, NotFiltered"#
        #if dataset != "GPSA": target_params = target_params.replace(", NoSubjGameOverride", "")
    else:
        target_params = "NoMsgHist, NoQCtr, NoPCtr, NoSCtr"
    model_list = ['claude-sonnet-4-20250514','claude-3-5-sonnet-20241022', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'grok-3-latest', 'gpt-4.1-2025-04-14', 'gpt-4o-2024-08-06', 'gemini-2.5-flash-preview-04-17', 'gemini-2.0-flash-001', 'gemini-1.5-pro', 'deepseek-chat']
#    model_list = ['claude-3-5-sonnet-20241022', 'gemini-2.0-flash-001', 'deepseek-chat', 'grok-3-latest', 'gpt-4o-2024-08-06', 'meta-llama-Meta-Llama-3.1-405B-Instruct', 'claude-3-haiku-20240307']
    introspection_score_type = "raw" # "adjusted", "filtered", or "raw"
    lift_score_type = "raw" # "adjusted", "filtered", or "raw"

    suffix = f"_{game_type}_full"
    if "Feedback_True" in target_params: suffix += "_fb"
    if "WithHistory" in target_params: suffix += "_hist" 
    else: suffix += "_sum"
    input_log_filename = f"analysis_log_multi_logres_{game_type}_{dataset.lower()}.txt"
    output_filename = f"{input_log_filename.split('.')[0]}{suffix}_parsed.txt"

#    model_list = ['grok-3-latest', 'gemini-1.5-pro', 'claude-3-7-sonnet-20250219', 'gemini-2.0-flash-001', 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'gpt-4-turbo-2024-04-09', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
#    model_list = ['claude-3-7-sonnet-20250219', 'grok-3-latest', 'claude-3-5-sonnet-20241022', 'gemini-2.0-flash-001', 'gemini-1.5-pro', 'claude-3-opus-20240229', 'gpt-4-turbo-2024-04-09', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
#    model_list = ['claude-3-7-sonnet-20250219', 'grok-3-latest', 'gemini-2.0-flash-001', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro', 'claude-3-opus-20240229', 'gpt-4-turbo-2024-04-09', 'claude-3-haiku-20240307', 'claude-3-sonnet-20240229']

    try:
        with open(input_log_filename, 'r', encoding='utf-8') as f:
            log_content_from_file = f.read()
        parse_analysis_log(log_content_from_file, output_filename, target_params, model_list, int_score_type=introspection_score_type, lift_score_type=lift_score_type)

        df_results = analyze_parsed_data(output_filename)
        df_display = (df_results.set_index("Subject").reindex(model_list).reset_index())
        print(df_display.to_string(index=False, formatters={"LR_pvalue": lambda p: ("" if pd.isna(p) else f"{p:.1e}" if p < 1e-4 else f"{p:.4f}")}))
     
        if not df_results.empty:
            plot_results(df_results, subject_order=model_list, dataset_name=f"{dataset}{suffix}", int_score_type=introspection_score_type, lift_score_type=lift_score_type)
        else:
            print("No results to plot.")

    except FileNotFoundError:
        print(f"Error: Input log file '{input_log_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")