import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
import warnings

# Z-score for 95% CI
Z_SCORE = norm.ppf(0.975)

def parse_secondchance_log(log_content, output_file, model_list):
    
    # Updated block start regex to capture model name and correct/incorrect status
    block_start_regex = re.compile(
        r"--- Analyzing (\S+) \(Redacted, (Correct|Incorrect), \d+ game files\) ---"
    )

    # Answer change regex
    answer_change_regex = re.compile(r"Answer change%: ([-\d.]+) \[([-\d.]+), ([-\d.]+)\] \(n=(\d+)\)")
    
    # Model start regexes
    model4_start_regex = re.compile(r"^\s*Model 4.*\(No Interactions\).*:\s*answer_changed ~")
    model46_start_regex = re.compile(r"^\s*Model 4\.6.*:\s*answer_changed ~")
    
    # Logit regression results marker
    logit_results_regex = re.compile(r"^\s*Logit Regression Results\s*$")
    
    # Log-likelihood regex
    log_likelihood_regex = re.compile(r"Log-Likelihood:\s*([-\d.]+)")
    
    # Coefficient extraction regexes for Model 4.6 
    capabilities_entropy_coef_regex = re.compile(r"^\s*capabilities_entropy\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
    normalized_prob_entropy_coef_regex = re.compile(r"^\s*normalized_prob_entropy\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
    
    # Singular matrix error regex
    singular_matrix_regex = re.compile(r"Could not fit Model 4\.95: Singular matrix")
    
    # Check if Model 4 could be fit
    model4_converged_regex = re.compile(r"converged:\s*True")

    analysis_blocks = re.split(r"(?=--- Analyzing )", log_content)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for block_content in analysis_blocks:
            if not block_content.strip():
                continue

            match_block_start = block_start_regex.search(block_content)
            if match_block_start:
                subject_name = match_block_start.group(1)
                correct_status = match_block_start.group(2)
                
                if subject_name not in model_list:
                    print(f"Skipping subject {subject_name} as it is not in the provided model list.")
                    continue
                
                outfile.write(f"Subject: {subject_name} ({correct_status})\n")
                
                extracted_info = {
                    "answer_change_pct": "Not found",
                    "answer_change_ci_low": "Not found",
                    "answer_change_ci_high": "Not found",
                    "n_value": "Not found",
                    "model4_log_lik": "Not found",
                    "model46_cap_entropy_coef": "Not found",
                    "model46_cap_entropy_ci_low": "Not found",
                    "model46_cap_entropy_ci_high": "Not found",
                    "model46_norm_prob_entropy_coef": "Not found",
                    "model46_norm_prob_entropy_ci_low": "Not found",
                    "model46_norm_prob_entropy_ci_high": "Not found",
                    "model46_log_lik": "Not found",
                }
                
                # Model parsing states
                in_model4 = False
                in_model46 = False
                found_logit_results = False

                lines = block_content.splitlines()
                for i, line in enumerate(lines):
                    # Extract answer change percentage
                    m = answer_change_regex.search(line)
                    if m:
                        extracted_info["answer_change_pct"] = m.group(1)
                        extracted_info["answer_change_ci_low"] = m.group(2)
                        extracted_info["answer_change_ci_high"] = m.group(3)
                        extracted_info["n_value"] = m.group(4)
                        continue

                    # Check for singular matrix error
                    if singular_matrix_regex.search(line):
                        # Model 4.6  couldn't be fit due to singular matrix
                        extracted_info["model46_log_lik"] = "Singular"
                        extracted_info["model46_cap_entropy_coef"] = "Singular"
                        extracted_info["model46_norm_prob_entropy_coef"] = "Singular"
                        continue
                    
                    # Check for Model 4 start
                    if model4_start_regex.search(line):
                        in_model4 = True
                        in_model46 = False
                        found_logit_results = False
                        continue
                    
                    # Check for Model 4.6  start
                    if model46_start_regex.search(line):
                        in_model4 = False
                        in_model46 = True
                        found_logit_results = False
                        continue
                    
                    # Check for Logit Regression Results
                    if (in_model4 or in_model46) and logit_results_regex.match(line):
                        found_logit_results = True
                        continue
                    
                    # Extract log-likelihood and coefficients based on current model
                    if in_model4 and found_logit_results:
                        # Look for log-likelihood in Model 4
                        m = log_likelihood_regex.search(line)
                        if m:
                            extracted_info["model4_log_lik"] = m.group(1)
                    
                    elif in_model46 and found_logit_results:
                        # Look for log-likelihood in Model 4.6 
                        m = log_likelihood_regex.search(line)
                        if m:
                            extracted_info["model46_log_lik"] = m.group(1)
                        
                        # Look for capabilities_entropy coefficient
                        m = capabilities_entropy_coef_regex.match(line)
                        if m:
                            extracted_info["model46_cap_entropy_coef"] = m.group(1)
                            extracted_info["model46_cap_entropy_ci_low"] = m.group(5)
                            extracted_info["model46_cap_entropy_ci_high"] = m.group(6)
                            continue
                        
                        # Look for normalized_prob_entropy coefficient
                        m = normalized_prob_entropy_coef_regex.match(line)
                        if m:
                            extracted_info["model46_norm_prob_entropy_coef"] = m.group(1)
                            extracted_info["model46_norm_prob_entropy_ci_low"] = m.group(5)
                            extracted_info["model46_norm_prob_entropy_ci_high"] = m.group(6)
                            continue
                    
                    # Reset state if we see a new model or section
                    if line.strip().startswith("Model ") and not any([
                        model4_start_regex.search(line),
                        model46_start_regex.search(line)
                    ]):
                        in_model4 = in_model46 = False
                        found_logit_results = False
                    
                    if line.strip().startswith("--- Analyzing"):
                        in_model4 = in_model46 = False
                        found_logit_results = False
                
                # Write extracted info
                outfile.write(f"  Answer change%: {extracted_info['answer_change_pct']} [{extracted_info['answer_change_ci_low']}, {extracted_info['answer_change_ci_high']}]\n")
                outfile.write(f"  N: {extracted_info['n_value']}\n")
                outfile.write(f"  Model 4 Log-Likelihood: {extracted_info['model4_log_lik']}\n")
                outfile.write(f"  Model 4.6  Log-Likelihood: {extracted_info['model46_log_lik']}\n")
                outfile.write(f"  Model 4.6  capabilities_entropy: {extracted_info['model46_cap_entropy_coef']} [{extracted_info['model46_cap_entropy_ci_low']}, {extracted_info['model46_cap_entropy_ci_high']}]\n")
                outfile.write(f"  Model 4.6  normalized_prob_entropy: {extracted_info['model46_norm_prob_entropy_coef']} [{extracted_info['model46_norm_prob_entropy_ci_low']}, {extracted_info['model46_norm_prob_entropy_ci_high']}]\n")
                outfile.write("\n")

    print(f"Parsing complete. Output written to {output_file}")


def parse_value(text, pattern, group=1, as_type=float):
    m = re.search(pattern, text)
    if m:
        try:
            return as_type(m.group(group))
        except (ValueError, TypeError):
            pass              
    return np.nan 


def analyze_parsed_data(input_summary_file):
    all_subject_data = []
    current_subject_info = {}

    with open(input_summary_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("Subject:"):
                if current_subject_info.get("subject_name"):
                    all_subject_data.append(current_subject_info)
                # Parse subject name and correct/incorrect status
                m = re.match(r"Subject: (\S+) \((Correct|Incorrect)\)", line)
                if m:
                    current_subject_info = {
                        "subject_name": m.group(1),
                        "correct_status": m.group(2)
                    }
                else:
                    current_subject_info = {"subject_name": line.split("Subject:")[1].strip()}
            elif "Answer change%:" in line:
                # Parse: "Answer change%: 0.1204 [0.07426407732768933, 0.1665736190073892]"
                m = re.search(r":\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]", line)
                if m:
                    current_subject_info["answer_change_pct"] = float(m.group(1))
                    current_subject_info["answer_change_ci_low"] = float(m.group(2))
                    current_subject_info["answer_change_ci_high"] = float(m.group(3))
            elif line.startswith("N:"):
                try:
                    current_subject_info["n_value"] = int(re.search(r":\s*(\d+)", line).group(1))
                except:
                    current_subject_info["n_value"] = np.nan
            elif "Model 4 Log-Likelihood:" in line:
                current_subject_info["loglik4"] = parse_value(line, r":\s*([-\d.]+)", as_type=float)
            elif "Model 4.6  Log-Likelihood:" in line:
                # Check if it says "Singular"
                if "Singular" in line:
                    current_subject_info["loglik46"] = np.nan
                    current_subject_info["singular_46"] = True
                else:
                    current_subject_info["loglik46"] = parse_value(line, r":\s*([-\d.]+)", as_type=float)
                    current_subject_info["singular_46"] = False
            elif "Model 4.6  capabilities_entropy:" in line:
                # Check if it says "Singular"
                if "Singular" in line:
                    current_subject_info["cap_entropy_coef"] = np.nan
                    current_subject_info["cap_entropy_ci_low"] = np.nan
                    current_subject_info["cap_entropy_ci_high"] = np.nan
                else:
                    # Parse: "Model 4.6  capabilities_entropy: 2.9170 [1.498, 4.336]"
                    m = re.search(r":\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]", line)
                    if m:
                        current_subject_info["cap_entropy_coef"] = float(m.group(1))
                        current_subject_info["cap_entropy_ci_low"] = float(m.group(2))
                        current_subject_info["cap_entropy_ci_high"] = float(m.group(3))
            elif "Model 4.6  normalized_prob_entropy:" in line:
                # Check if it says "Singular"
                if "Singular" in line:
                    current_subject_info["norm_prob_entropy_coef"] = np.nan
                    current_subject_info["norm_prob_entropy_ci_low"] = np.nan
                    current_subject_info["norm_prob_entropy_ci_high"] = np.nan
                else:
                    # Parse: "Model 4.6  normalized_prob_entropy: -0.0826 [-1.272, 1.107]"
                    m = re.search(r":\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]", line)
                    if m:
                        current_subject_info["norm_prob_entropy_coef"] = float(m.group(1))
                        current_subject_info["norm_prob_entropy_ci_low"] = float(m.group(2))
                        current_subject_info["norm_prob_entropy_ci_high"] = float(m.group(3))

        if current_subject_info.get("subject_name"):
            all_subject_data.append(current_subject_info)

    results = []
    for data in all_subject_data:
        subject_name = data.get("subject_name", "Unknown")
        correct_status = data.get("correct_status", "Unknown")
        
        # Get all the values, using np.nan for missing optional values
        answer_change_pct = data.get("answer_change_pct", np.nan)
        answer_change_ci_low = data.get("answer_change_ci_low", np.nan)
        answer_change_ci_high = data.get("answer_change_ci_high", np.nan)
        n_value = data.get("n_value", np.nan)
        
        cap_entropy_coef = data.get("cap_entropy_coef", np.nan)
        cap_entropy_ci_low = data.get("cap_entropy_ci_low", np.nan)
        cap_entropy_ci_high = data.get("cap_entropy_ci_high", np.nan)
        
        norm_prob_entropy_coef = data.get("norm_prob_entropy_coef", np.nan)
        norm_prob_entropy_ci_low = data.get("norm_prob_entropy_ci_low", np.nan)
        norm_prob_entropy_ci_high = data.get("norm_prob_entropy_ci_high", np.nan)
        
        LL4 = data.get("loglik4", np.nan)
        LL46 = data.get("loglik46", np.nan)
        LL4   = pd.to_numeric(data.get("loglik4",   np.nan), errors="coerce")
        LL46 = pd.to_numeric(data.get("loglik46", np.nan), errors="coerce")
        singular_46 = data.get("singular_46", False)
        
        # Calculate likelihood ratio test between Model 4 and Model 4.6 
        # Model 4.6  has 2 more parameters (capabilities_entropy and normalized_prob_entropy)
        LR_stat = 2 * (LL46 - LL4) if not pd.isna(LL4) and not pd.isna(LL46) else np.nan
        LR_pvalue = chi2.sf(LR_stat, df=2) if not pd.isna(LR_stat) else np.nan
        
        results.append({
            "Subject": subject_name,
            "Status": correct_status,
            "AnswerChange%": answer_change_pct,
            "AnswerChange_CI_Low": answer_change_ci_low,
            "AnswerChange_CI_High": answer_change_ci_high,
            "N": n_value,
            "CapEntropyCoef": cap_entropy_coef,
            "CapEntropy_CI_Low": cap_entropy_ci_low,
            "CapEntropy_CI_High": cap_entropy_ci_high,
#            "Norm Prob Entropy Coef": norm_prob_entropy_coef,
#            "NormProbEntropy_CI_Low": norm_prob_entropy_ci_low,
#            "NormProbEntropy_CI_High": norm_prob_entropy_ci_high,
            "LL_Model4": LL4,
            "LL_Model46": LL46,
            "LR_stat": LR_stat,
            "LR_pvalue": LR_pvalue,
        })
        
    df = pd.DataFrame(results)
    
    # Ensure numeric columns are actually numeric
    numeric_columns = ['AnswerChange%', 'AnswerChange_CI_Low', 'AnswerChange_CI_High', 'N',
                      'CapEntropyCoef', 'CapEntropy_CI_Low', 'CapEntropy_CI_High',
#                      'Norm Prob Entropy Coef', 'NormProbEntropy_CI_Low', 'NormProbEntropy_CI_High',
                      'LL_Model4', 'LL_Model46', 'LR_stat', 'LR_pvalue']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


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


def plot_results(df_results, subject_order=None, dataset_name="GPQA_SecondChance"):
    if df_results.empty:
        print("No data to plot.")
        return

    # Separate Correct and Incorrect data
    df_correct = df_results[df_results['Status'] == 'Correct'].copy()
    df_incorrect = df_results[df_results['Status'] == 'Incorrect'].copy()

    # Reorder DataFrames if subject_order is provided
    for df, label in [(df_correct, 'Correct'), (df_incorrect, 'Incorrect')]:
        if subject_order and not df.empty:
            df_ordered = df.copy()
            df_ordered['Subject_Cat'] = pd.Categorical(df_ordered['Subject'], categories=subject_order, ordered=True)
            df_ordered = df_ordered[df_ordered['Subject_Cat'].notna()].sort_values('Subject_Cat')
            if label == 'Correct':
                df_correct = df_ordered.drop(columns=['Subject_Cat'])
            else:
                df_incorrect = df_ordered.drop(columns=['Subject_Cat'])

    # Check if we have entropy coefficients
    has_cap_entropy = False
    has_norm_prob_entropy = False
    
    for df in [df_correct, df_incorrect]:
        if not df.empty and "CapEntropyCoef" in df.columns:
            # Check if we have any non-NaN values
            cap_values = df["CapEntropyCoef"]
            has_cap_entropy = has_cap_entropy or cap_values.notna().any()
            
        if not df.empty and "Norm Prob Entropy Coef" in df.columns:
            # Check if we have any non-NaN values
            norm_values = df["Norm Prob Entropy Coef"]
            has_norm_prob_entropy = has_norm_prob_entropy or norm_values.notna().any()
    
    has_entropy = has_cap_entropy or has_norm_prob_entropy
    
    # Determine number of columns
    ncols = 3 if has_entropy else 1
    nrows = 2  # One for Correct, one for Incorrect
    
    # Try to use seaborn style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass

    # Create figure
    fig, axs = plt.subplots(nrows, ncols, figsize=(max(10 * ncols, len(subject_order) * 1.5 + 2 if subject_order else 15), 12))
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1 or ncols == 1:
        axs = axs.reshape(nrows, ncols)

    # Font sizes
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12

    # Plot for each status (Correct, Incorrect)
    for row_idx, (df, status) in enumerate([(df_correct, 'Correct'), (df_incorrect, 'Incorrect')]):
        if df.empty:
            for col_idx in range(ncols):
                axs[row_idx, col_idx].text(0.5, 0.5, f'No {status} data', 
                                          ha='center', va='center', transform=axs[row_idx, col_idx].transAxes)
                axs[row_idx, col_idx].set_xticks([])
                axs[row_idx, col_idx].set_yticks([])
            continue
        
        # Apply name breaking for x-axis labels
        formatted_subject_names = [break_subject_name(name, max_parts_per_line=3) for name in df["Subject"]]
        
        # --- Plot 1: Answer Change % ---
        yerr_low = np.nan_to_num(df["AnswerChange%"] - df["AnswerChange_CI_Low"], nan=0.0)
        yerr_high = np.nan_to_num(df["AnswerChange_CI_High"] - df["AnswerChange%"], nan=0.0)
        yerr_low[yerr_low < 0] = 0
        yerr_high[yerr_high < 0] = 0
        
        bars = axs[row_idx, 0].bar(formatted_subject_names, df["AnswerChange%"],
                                   color='mediumslateblue' if status == 'Correct' else 'salmon',
                                   yerr=[yerr_low, yerr_high], ecolor='gray', capsize=5, width=0.6)
        
        # Add N values on bars
        for i, (bar, n) in enumerate(zip(bars, df["N"])):
            if not pd.isna(n):
                height = bar.get_height()
                axs[row_idx, 0].text(bar.get_x() + bar.get_width()/2., height,
                                    f'n={int(n)}', ha='center', va='bottom', fontsize=9)
        
        axs[row_idx, 0].set_ylabel('Answer Change %', fontsize=label_fontsize)
        axs[row_idx, 0].set_title(f'Answer Change Rate - {status} Answers (95% CI)', fontsize=title_fontsize)
        axs[row_idx, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axs[row_idx, 0].axhline(0.25, color='red', linestyle=':', linewidth=0.8, label='25%')
        if row_idx == 0:  # Only show legend on first plot
            axs[row_idx, 0].legend(fontsize=legend_fontsize)
        axs[row_idx, 0].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
        axs[row_idx, 0].tick_params(axis='y', labelsize=tick_fontsize)
        axs[row_idx, 0].set_ylim(0, max(0.7, df["AnswerChange_CI_High"].max() * 1.1))

        # If we have entropy coefficients
        if has_entropy:
            if has_cap_entropy:
                # --- Plot 2: Capabilities Entropy Coefficient ---
                # Convert to numeric values for plotting
                cap_values = pd.to_numeric(df["CapEntropyCoef"], errors='coerce')
                cap_ci_low = pd.to_numeric(df["CapEntropy_CI_Low"], errors='coerce')
                cap_ci_high = pd.to_numeric(df["CapEntropy_CI_High"], errors='coerce')
                yerr_cap_low = np.nan_to_num(cap_values - cap_ci_low, nan=0.0)
                yerr_cap_high = np.nan_to_num(cap_ci_high - cap_values, nan=0.0)
                yerr_cap_low[yerr_cap_low < 0] = 0
                yerr_cap_high[yerr_cap_high < 0] = 0
                
                bars = axs[row_idx, 1].bar(formatted_subject_names, cap_values,
                               color='cornflowerblue' if status == 'Correct' else 'lightcoral',
                               yerr=[yerr_cap_low, yerr_cap_high], ecolor='gray', capsize=5, width=0.6)
                
                axs[row_idx, 1].set_ylabel('Coefficient Value', fontsize=label_fontsize)
                axs[row_idx, 1].set_title(f'Capabilities Entropy (Model 4.6 ) - {status}', fontsize=title_fontsize)
                axs[row_idx, 1].axhline(0, color='black', linestyle='--', linewidth=0.8)
                axs[row_idx, 1].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
                axs[row_idx, 1].tick_params(axis='y', labelsize=tick_fontsize)
            else:
                axs[row_idx, 1].axis('off')
            
            if has_norm_prob_entropy:
                # --- Plot 3: Normalized Probability Entropy Coefficient ---
                # Convert to numeric values for plotting
                norm_values = pd.to_numeric(df["Norm Prob Entropy Coef"], errors='coerce')
                norm_ci_low = pd.to_numeric(df["NormProbEntropy_CI_Low"], errors='coerce')
                norm_ci_high = pd.to_numeric(df["NormProbEntropy_CI_High"], errors='coerce')
                yerr_norm_low = np.nan_to_num(norm_values - norm_ci_low, nan=0.0)
                yerr_norm_high = np.nan_to_num(norm_ci_high - norm_values, nan=0.0)
                yerr_norm_low[yerr_norm_low < 0] = 0
                yerr_norm_high[yerr_norm_high < 0] = 0
                
                bars = axs[row_idx, 2].bar(formatted_subject_names, norm_values,
                               color='darkorange' if status == 'Correct' else 'gold',
                               yerr=[yerr_norm_low, yerr_norm_high], ecolor='gray', capsize=5, width=0.6)
                
                axs[row_idx, 2].set_ylabel('Coefficient Value', fontsize=label_fontsize)
                axs[row_idx, 2].set_title(f'Normalized Prob Entropy (Model 4.6 ) - {status}', fontsize=title_fontsize)
                axs[row_idx, 2].axhline(0, color='black', linestyle='--', linewidth=0.8)
                axs[row_idx, 2].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
                axs[row_idx, 2].tick_params(axis='y', labelsize=tick_fontsize)
            else:
                axs[row_idx, 2].axis('off')

    plt.tight_layout(pad=3.0, h_pad=4.0)
    plt.savefig(f"secondchance_analysis_charts_{dataset_name}.png", dpi=300)
    print(f"Charts saved to secondchance_analysis_charts_{dataset_name}.png")


if __name__ == "__main__":
    
    dataset = "SimpleMC"# "GPQA"#
    suffix = ""

    input_log_filename = f"analysis_log_multi_logres_sc_{dataset.lower()}.txt"
    output_filename = f"{input_log_filename.split('.')[0]}{suffix}_parsed.txt"
    
    model_list = ['claude-sonnet-4-20250514','claude-3-5-sonnet-20241022', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'grok-3-latest', 'gpt-4.1-2025-04-14', 'gpt-4o-2024-08-06', 'gemini-2.5-flash-preview-04-17', 'gemini-2.0-flash-001', 'gemini-1.5-pro', 'deepseek-chat']


    try:
        with open(input_log_filename, 'r', encoding='utf-8') as f:
            log_content_from_file = f.read()
        parse_secondchance_log(log_content_from_file, output_filename, model_list)

        df_results = analyze_parsed_data(output_filename)
        print("\n--- Calculated Data ---")


        if not df_results.empty:
            # ① choose the columns to show ── now include CI bounds + log-likelihoods
            display_columns = [
                'Subject', 'N', 'Status',
                'AnswerChange%', 'AnswerChange_CI_Low', 'AnswerChange_CI_High',
                'CapEntropyCoef', #'Norm Prob Entropy Coef',
                'LL_Model4', 'LL_Model46',
                'LR_stat', 'LR_pvalue'
            ]

            df_display = df_results.copy()#df_results[df_results['Status']=="Incorrect"][display_columns].copy()
            #df_display = (df_display.set_index("Subject").reindex(model_list).reset_index())

            # ② basic formatting -----------------------------------------------------
            # integers
            df_display['N'] = df_display['N'].apply(
                lambda x: int(x) if pd.notna(x) else 0
            )

            # percentages & their CIs
            for col in ['AnswerChange%', 'AnswerChange_CI_Low', 'AnswerChange_CI_High']:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                )

            # log-likelihoods
            for col in ['LL_Model4', 'LL_Model46']:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else ""
                )

            # entropy coefficients (handle “Singular”)
            for coef_col in display_columns:
                if 'coef' not in coef_col.lower(): continue
                df_display[coef_col] = df_results.apply(
                    lambda row: ("Singular" if row.get('Singular_46', False)
                                else ("" if pd.isna(row[coef_col])
                                    else f"{row[coef_col]:.3f}")),
                    axis=1
                )

            # likelihood-ratio test stats
            df_display['LR_stat'] = df_display['LR_stat'].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
            df_display['LR_pvalue'] = df_display['LR_pvalue'].apply(
                lambda p: ("N/A" if pd.isna(p)
                        else f"{p:.1e}" if p < 1e-4
                        else f"{p:.4f}")
            )

            print(df_display.to_string(index=False))
            plot_results(df_results, subject_order=model_list, dataset_name=f"{dataset}{suffix}")
        else:
            print("No results to display.")

    except FileNotFoundError:
        print(f"Error: Input log file '{input_log_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()