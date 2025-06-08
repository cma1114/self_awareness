import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
import warnings

import re

def parse_analysis_log(log_content, output_file):
    target_params = "Feedback_False, Non_Redacted, NoSubjAccOverride, NotRandomized, WithHistory, NotFiltered"
    
    block_start_regex = re.compile(
        r"--- Analyzing (\S+) \(" + re.escape(target_params) + r", \d+ game files\) ---"
    )

    # Model 4.6 regexes 
    model46_start_regex = re.compile(r"^\s*Model 4\.6.*:\s*delegate_choice ~")
    pi_capability_line_regex = re.compile(r"^\s*capabilities_entropy\s+(-?\d+\.\d+)\s+.*$")

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
                    "model46_pi_cap": "Not found",
                }
                
                found_model46_summary = False
                in_model46_summary_table = False

                lines = block_content.splitlines()
                for line in lines:

                # --- Model 4.6 parsing ---
                    if not found_model46_summary and model46_start_regex.search(line):
                        found_model46_summary = True
                        continue

                    if found_model46_summary and not in_model46_summary_table:
                        # look for the separator line of "==="
                        if line.strip().startswith("===") and line.strip().endswith("==="):
                            in_model46_summary_table = True
                        continue

                    if in_model46_summary_table:
                        m_si = pi_capability_line_regex.match(line)
                        if m_si:
                            extracted_info["model46_pi_cap"] = line.strip()
                            # done with Model 4 block
                            found_model46_summary = False
                            in_model46_summary_table = False
                            continue

                        # end Model 4 if we see next model or Odds Ratio
                        if (line.strip().startswith("--- Analyzing") or
                            (line.strip().startswith("Model ") and not model46_start_regex.search(line)) or
                            line.strip().startswith("--- Odds Ratio")):
                            found_model46_summary = False
                            in_model46_summary_table = False


                # --- Write out the collected info ---
                outfile.write(f"  Model 4.6 p_i_capability: {extracted_info['model46_pi_cap']}\n")

    print(f"Parsing complete. Output written to {output_file}")


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
            elif "Model 4.6 p_i_capability:" in line:
                # Line example: "Model 4 s_i_capability: s_i_capability      -1.0064  0.436   -2.311   0.021   -1.860   -0.153"
                # We want the numbers after the second "s_i_capability"
                
                # Find the part of the string that actually contains the coefficient line
                # This assumes the first parser wrote "s_i_capability <coef_line_content>"
                match_coef_line_content = re.search(r"Model 4.6 p_i_capability:\s*(capabilities_entropy\s+.*)", line)
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

        si_coef = data.get("si_coef")
        si_ci_low = data.get("si_coef_ci_low")
        si_ci_high = data.get("si_coef_ci_high")
        
        results.append({
            "Subject": subject_name,
            "Reversed SI Coef": si_coef,
            "RevSICoef_CI_Low": si_ci_low,
            "RevSICoef_CI_High": si_ci_high,
        })
        
    return pd.DataFrame(results)



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

def plot_results(df_results, subject_order=None, dataset_name=""):
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


    # --- Plot 3: Reversed s_i_capability Coefficient ---
    yerr_si_low = np.nan_to_num(df_results["Reversed SI Coef"] - df_results["RevSICoef_CI_Low"], nan=0.0)
    yerr_si_high = np.nan_to_num(df_results["RevSICoef_CI_High"] - df_results["Reversed SI Coef"], nan=0.0)
    yerr_si_low[yerr_si_low < 0] = 0
    yerr_si_high[yerr_si_high < 0] = 0
    
    axs[2].bar(formatted_subject_names, df_results["Reversed SI Coef"], 
               color='mediumseagreen', # Removed label
               yerr=[yerr_si_low, yerr_si_high], ecolor='gray', capsize=5, width=0.6)
    axs[2].set_ylabel('Coefficient Value (Log-Odds Scale)', fontsize=label_fontsize)
    axs[2].set_title('Capabilities Probability Coefficient by LLM (95% CI)', fontsize=title_fontsize)
    axs[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
    # axs[2].legend() # Removed legend
    axs[2].tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    axs[2].tick_params(axis='y', labelsize=tick_fontsize)

    # Common x-label for the whole figure (if desired, but usually per-subplot is fine)
    # fig.text(0.5, 0.01, 'Subject', ha='center', va='center', fontsize=label_fontsize)

    plt.tight_layout(pad=3.0, h_pad=4.0) # Adjust h_pad for vertical spacing if titles/xlabels overlap
    plt.savefig(f"subject_analysis_charts_{dataset_name}.png", dpi=300)
    print(f"Charts saved to subject_analysis_charts_{dataset_name}.png")
    plt.show()

if __name__ == "__main__":
    dataset = "GPSA"
    suffix = "_full"

    input_log_filename = f"analysis_log_multi_logres_dg_{dataset.lower()}.txt"
    output_filename = f"{input_log_filename.split('.')[0]}{suffix}_capent_parsed.txt"
    try:
        with open(input_log_filename, 'r', encoding='utf-8') as f:
            log_content_from_file = f.read()
        parse_analysis_log(log_content_from_file, output_filename)

        df_results = analyze_parsed_data(output_filename)
        print("\n--- Calculated Data ---")
        print(df_results.to_string(formatters={"LR_pvalue": lambda p: ("" if pd.isna(p) else f"{p:.1e}" if p < 1e-4 else f"{p:.4f}")})) # Print full DataFrame
        
        model_list = ['grok-3-latest', 'gpt-4o-2024-08-06']
        if not df_results.empty:
            plot_results(df_results, model_list, dataset_name=f"{dataset}{suffix}_capent")
        else:
            print("No results to plot.")

    except FileNotFoundError:
        print(f"Error: Input log file '{input_log_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        