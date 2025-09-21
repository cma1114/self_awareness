import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

# Z-score for 95% CI
Z_SCORE = norm.ppf(0.975)

def parse_game_neutral_comparison_log(log_content, output_file, model_list, corstr=""):
    
    # Block start regex to capture model name
    block_start_regex = re.compile(
        rf"--- Analyzing (\S+) \(Redacted, {corstr}\d+ game files\) ---"
    )

    # Paired comparison results regex patterns
    n_pairs_regex = re.compile(r"Number of paired observations: (\d+)")
    game_rate_regex = re.compile(r"Game:\s+([\d.]+)\s+\(95% CI:\s+\[([\d.]+),\s+([\d.]+)\]\)")
    neutral_rate_regex = re.compile(r"Neutral:\s+([\d.]+)\s+\(95% CI:\s+\[([\d.]+),\s+([\d.]+)\]\)")
    normalized_lift_regex = re.compile(r"Normalized lift:\s+([-\d.]+)\s+\(95% CI:\s+\[([-\d.]+),\s+([-\d.]+)\]\)")
    mcnemar_pvalue_regex = re.compile(r"p-value:\s+([\d.]+)")
    
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
                    "n_pairs": "Not found",
                    "game_rate": "Not found",
                    "game_rate_ci_low": "Not found",
                    "game_rate_ci_high": "Not found",
                    "neutral_rate": "Not found",
                    "neutral_rate_ci_low": "Not found",
                    "neutral_rate_ci_high": "Not found",
                    "normalized_lift": "Not found",
                    "normalized_lift_ci_low": "Not found",
                    "normalized_lift_ci_high": "Not found",
                    "mcnemar_pvalue": "Not found",
                }
                
                lines = block_content.splitlines()
                for i, line in enumerate(lines):
                    # Extract number of pairs
                    m = n_pairs_regex.search(line)
                    if m:
                        extracted_info["n_pairs"] = m.group(1)
                        continue
                    
                    # Extract game change rate
                    m = game_rate_regex.search(line)
                    if m:
                        extracted_info["game_rate"] = m.group(1)
                        extracted_info["game_rate_ci_low"] = m.group(2)
                        extracted_info["game_rate_ci_high"] = m.group(3)
                        continue
                    
                    # Extract neutral change rate
                    m = neutral_rate_regex.search(line)
                    if m:
                        extracted_info["neutral_rate"] = m.group(1)
                        extracted_info["neutral_rate_ci_low"] = m.group(2)
                        extracted_info["neutral_rate_ci_high"] = m.group(3)
                        continue
                    
                    # Extract normalized lift
                    m = normalized_lift_regex.search(line)
                    if m:
                        extracted_info["normalized_lift"] = m.group(1)
                        extracted_info["normalized_lift_ci_low"] = m.group(2)
                        extracted_info["normalized_lift_ci_high"] = m.group(3)
                        continue
                    
                    # Extract McNemar p-value
                    if "McNemar's Test:" in line:
                        # Look for p-value in next few lines
                        for j in range(i+1, min(i+5, len(lines))):
                            m = mcnemar_pvalue_regex.search(lines[j])
                            if m:
                                extracted_info["mcnemar_pvalue"] = m.group(1)
                                break
                
                # Write extracted info
                outfile.write(f"  N pairs: {extracted_info['n_pairs']}\n")
                outfile.write(f"  Game rate: {extracted_info['game_rate']} [{extracted_info['game_rate_ci_low']}, {extracted_info['game_rate_ci_high']}]\n")
                outfile.write(f"  Neutral rate: {extracted_info['neutral_rate']} [{extracted_info['neutral_rate_ci_low']}, {extracted_info['neutral_rate_ci_high']}]\n")
                outfile.write(f"  Normalized lift: {extracted_info['normalized_lift']} [{extracted_info['normalized_lift_ci_low']}, {extracted_info['normalized_lift_ci_high']}]\n")
                outfile.write(f"  McNemar p-value: {extracted_info['mcnemar_pvalue']}\n")
                outfile.write("\n")

    print(f"Parsing complete. Output written to {output_file}")


def analyze_parsed_data(input_summary_file):
    all_subject_data = []
    current_subject_info = {}

    with open(input_summary_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("Subject:"):
                if current_subject_info.get("subject_name"):
                    all_subject_data.append(current_subject_info)
                # Parse subject name
                m = re.match(r"Subject: (\S+)", line)
                if m:
                    current_subject_info = {
                        "subject_name": m.group(1)
                    }
                else:
                    current_subject_info = {"subject_name": line.split("Subject:")[1].strip()}
            elif line.startswith("N pairs:"):
                try:
                    current_subject_info["n_pairs"] = int(re.search(r":\s*(\d+)", line).group(1))
                except:
                    current_subject_info["n_pairs"] = np.nan
            elif "Game rate:" in line:
                # Parse: "Game rate: 0.282 [0.239, 0.324]"
                m = re.search(r":\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", line)
                if m:
                    current_subject_info["game_rate"] = float(m.group(1))
                    current_subject_info["game_rate_ci_low"] = float(m.group(2))
                    current_subject_info["game_rate_ci_high"] = float(m.group(3))
            elif "Neutral rate:" in line:
                # Parse: "Neutral rate: 0.233 [0.195, 0.273]"
                m = re.search(r":\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", line)
                if m:
                    current_subject_info["neutral_rate"] = float(m.group(1))
                    current_subject_info["neutral_rate_ci_low"] = float(m.group(2))
                    current_subject_info["neutral_rate_ci_high"] = float(m.group(3))
            elif "Normalized lift:" in line:
                # Parse: "Normalized lift: 0.064 [0.018, 0.108]"
                m = re.search(r":\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]", line)
                if m:
                    current_subject_info["normalized_lift"] = float(m.group(1))
                    current_subject_info["normalized_lift_ci_low"] = float(m.group(2))
                    current_subject_info["normalized_lift_ci_high"] = float(m.group(3))
            elif "McNemar p-value:" in line:
                try:
                    current_subject_info["mcnemar_pvalue"] = float(re.search(r":\s*([\d.]+)", line).group(1))
                except:
                    current_subject_info["mcnemar_pvalue"] = np.nan

        if current_subject_info.get("subject_name"):
            all_subject_data.append(current_subject_info)

    results = []
    for data in all_subject_data:
        subject_name = data.get("subject_name", "Unknown")
        
        # Get all the values, using np.nan for missing values
        n_pairs = data.get("n_pairs", np.nan)
        
        game_rate = data.get("game_rate", np.nan)
        game_rate_ci_low = data.get("game_rate_ci_low", np.nan)
        game_rate_ci_high = data.get("game_rate_ci_high", np.nan)
        
        neutral_rate = data.get("neutral_rate", np.nan)
        neutral_rate_ci_low = data.get("neutral_rate_ci_low", np.nan)
        neutral_rate_ci_high = data.get("neutral_rate_ci_high", np.nan)
        
        normalized_lift = data.get("normalized_lift", np.nan)
        normalized_lift_ci_low = data.get("normalized_lift_ci_low", np.nan)
        normalized_lift_ci_high = data.get("normalized_lift_ci_high", np.nan)
        
        mcnemar_pvalue = data.get("mcnemar_pvalue", np.nan)
        
        results.append({
            "Subject": subject_name,
            "N": n_pairs,
            "GameRate": game_rate,
            "GameRate_CI_Low": game_rate_ci_low,
            "GameRate_CI_High": game_rate_ci_high,
            "NeutralRate": neutral_rate,
            "NeutralRate_CI_Low": neutral_rate_ci_low,
            "NeutralRate_CI_High": neutral_rate_ci_high,
            "NormalizedLift": normalized_lift,
            "NormalizedLift_CI_Low": normalized_lift_ci_low,
            "NormalizedLift_CI_High": normalized_lift_ci_high,
            "McNemar_pvalue": mcnemar_pvalue,
        })
        
    df = pd.DataFrame(results)
    
    # Ensure numeric columns are actually numeric
    numeric_columns = ['N', 'GameRate', 'GameRate_CI_Low', 'GameRate_CI_High',
                      'NeutralRate', 'NeutralRate_CI_Low', 'NeutralRate_CI_High',
                      'NormalizedLift', 'NormalizedLift_CI_Low', 'NormalizedLift_CI_High',
                      'McNemar_pvalue']
    
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


def plot_results(df_results, subject_order=None, dataset_name="GPQA_Game_vs_Neutral", corstr=""):
    if df_results.empty:
        print("No data to plot.")
        return

    df = df_results.copy()

    # Reorder DataFrame if subject_order is provided
    if subject_order and not df.empty:
        df_ordered = df.copy()
        df_ordered['Subject_Cat'] = pd.Categorical(df_ordered['Subject'], categories=subject_order, ordered=True)
        df_ordered = df_ordered[df_ordered['Subject_Cat'].notna()].sort_values('Subject_Cat')
        df = df_ordered.drop(columns=['Subject_Cat'])

    # Try to use seaborn style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass

    # Create figure with 2 subplots
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

    # Font sizes
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12

    if df.empty:
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, 'No data', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        return
    
    # Apply name breaking for x-axis labels
    formatted_subject_names = [break_subject_name(name, max_parts_per_line=3) for name in df["Subject"]]
    
    # --- Plot 1: Game vs Neutral Answer Change Rates ---
    x = np.arange(len(formatted_subject_names))
    width = 0.35
    
    # Calculate error bars for game rates
    game_yerr_low = np.nan_to_num(df["GameRate"] - df["GameRate_CI_Low"], nan=0.0)
    game_yerr_high = np.nan_to_num(df["GameRate_CI_High"] - df["GameRate"], nan=0.0)
    game_yerr_low[game_yerr_low < 0] = 0
    game_yerr_high[game_yerr_high < 0] = 0
    
    # Calculate error bars for neutral rates
    neutral_yerr_low = np.nan_to_num(df["NeutralRate"] - df["NeutralRate_CI_Low"], nan=0.0)
    neutral_yerr_high = np.nan_to_num(df["NeutralRate_CI_High"] - df["NeutralRate"], nan=0.0)
    neutral_yerr_low[neutral_yerr_low < 0] = 0
    neutral_yerr_high[neutral_yerr_high < 0] = 0
    
    # Create grouped bars
    bars1 = ax1.bar(x - width/2, df["NeutralRate"], width, 
                     label='Neutral', color='lightcoral',
                     yerr=[neutral_yerr_low, neutral_yerr_high], 
                     ecolor='gray', capsize=5)
    
    bars2 = ax1.bar(x + width/2, df["GameRate"], width,
                     label='Game', color='mediumslateblue',
                     yerr=[game_yerr_low, game_yerr_high], 
                     ecolor='gray', capsize=5)
    
    # Add significance markers for significant differences
    for i, (game_rate, neutral_rate, pval) in enumerate(zip(df["GameRate"], df["NeutralRate"], df["McNemar_pvalue"])):
        if not pd.isna(pval) and pval < 0.05:
            # Place star above the higher bar
            max_height = max(df.iloc[i]["GameRate_CI_High"], df.iloc[i]["NeutralRate_CI_High"])
            ax1.text(i, max_height + 0.01, '*', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Convert y-axis to percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    ax1.set_xlabel('Model', fontsize=label_fontsize)
    ax1.set_ylabel('Answer Change Rate', fontsize=label_fontsize)
    ax1.set_title(f'Answer Change Rates: {dataset}', fontsize=title_fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(formatted_subject_names, rotation=45, ha='right', fontsize=tick_fontsize)
    ax1.tick_params(axis='y', labelsize=tick_fontsize)
    ax1.legend(fontsize=legend_fontsize)
    ax1.set_ylim(0, max(0.5, max(df["GameRate_CI_High"].max(), df["NeutralRate_CI_High"].max()) * 1.1))
    ax1.grid(axis='y', alpha=0.3)
    
    # --- Plot 2: Normalized Lift ---
    # Calculate error bars for normalized lift
    lift_yerr_low = np.nan_to_num(df["NormalizedLift"] - df["NormalizedLift_CI_Low"], nan=0.0)
    lift_yerr_high = np.nan_to_num(df["NormalizedLift_CI_High"] - df["NormalizedLift"], nan=0.0)
    
    # Handle cases where CI might cross zero
    for i in range(len(lift_yerr_low)):
        if df.iloc[i]["NormalizedLift"] - lift_yerr_low[i] < -1:
            lift_yerr_low[i] = df.iloc[i]["NormalizedLift"] + 1
        if df.iloc[i]["NormalizedLift"] + lift_yerr_high[i] > 1:
            lift_yerr_high[i] = 1 - df.iloc[i]["NormalizedLift"]
    
    bars3 = ax2.bar(formatted_subject_names, df["NormalizedLift"],
                     color='darkseagreen',
                     yerr=[lift_yerr_low, lift_yerr_high], 
                     ecolor='gray', capsize=5, width=0.6)
    
    # Add percentage labels on bars
    for i, (bar, lift) in enumerate(zip(bars3, df["NormalizedLift"])):
        if not pd.isna(lift):
            height = bar.get_height()
            # Place label above or below bar depending on sign
            if height >= 0:
                va = 'bottom'
                y_offset = 0.005
            else:
                va = 'top'
                y_offset = -0.005
            ax2.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                     f'{lift*100:.1f}%', ha='center', va=va, fontsize=10)
    
    ax2.set_xlabel('Model', fontsize=label_fontsize)
    ax2.set_ylabel('Normalized Lift', fontsize=label_fontsize)
    ax2.set_title(f'Normalized Change Rate Lift: {dataset}', fontsize=title_fontsize)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
#    ax2.axhline(0.25, color='red', linestyle=':', linewidth=0.8, alpha=0.5, label='25%')
    ax2.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize)
    ax2.set_ylim(min(-0.1, df["NormalizedLift_CI_Low"].min() * 1.1), 
                 max(0.5, df["NormalizedLift_CI_High"].max() * 1.1))
    ax2.legend(fontsize=legend_fontsize)
    ax2.grid(axis='y', alpha=0.3)
    
    # Convert y-axis to percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

#    plt.tight_layout()
#    plt.savefig(f"game_vs_neutral_comparison_{dataset_name}.png", dpi=300, bbox_inches='tight')
    print(f"Charts saved to game_vs_neutral_comparison_{dataset_name}{corstr}.png")
    fig1.tight_layout()
    fig1.savefig(f"game_vs_neutral_rates_{dataset_name}{corstr}.png", dpi=300, bbox_inches='tight')

    fig2.tight_layout()
    fig2.savefig(f"normalized_lift_{dataset_name}{corstr}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    
    dataset = "SimpleMC" #"GPSA" #"SimpleQA" #"GPQA"#
    suffix = "_vs_neutral"
    corstr  = "" # "_cor", "_incor", ""
    sc_version = "_new"  # "_new" or "" or "_neut"

    corsuffix = "_all" if corstr == "" else ""
    corstrcln = "Correct, " if corstr == "_cor" else "Incorrect, " if corstr == "_incor" else ""


    input_log_filename = f"analysis_log_multi_logres_sc_{dataset.lower()}{sc_version}{corsuffix}{suffix}.txt"
    output_filename = f"{input_log_filename.split('.')[0]}_parsed.txt"
    
    model_list = ['gemini-2.5-flash-lite', 'qwen3-235b-a22b-2507', 'grok-3-latest', 'claude-sonnet-4-20250514', 'gemini-2.5-flash-preview-04-17', 'gpt-4.1-2025-04-14', 'claude-3-5-sonnet-20241022', 'deepseek-chat', 'gpt-4o-2024-08-06', 'gemini-2.0-flash-001', 'gemini-1.5-pro', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    model_list = ["openai-gpt-5-chat", "claude-opus-4-1-20250805", 'claude-sonnet-4-20250514', 'grok-3-latest', 'qwen3-235b-a22b-2507', 'claude-3-5-sonnet-20241022', 'gpt-4.1-2025-04-14', 'gpt-4o-2024-08-06', 'deepseek-chat', "gemini-2.5-flash_think", "gemini-2.5-flash_nothink", 'gemini-2.0-flash-001', "gemini-2.5-flash-lite_think", "gemini-2.5-flash-lite_nothink", 'gpt-4o-mini', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'gemini-1.5-pro']

    try:
        with open(input_log_filename, 'r', encoding='utf-8') as f:
            log_content_from_file = f.read()
        parse_game_neutral_comparison_log(log_content_from_file, output_filename, model_list, corstr=corstrcln)

        df_results = analyze_parsed_data(output_filename)
        print("\n--- Calculated Data ---")

        if not df_results.empty:
            # Display the parsed data
            display_columns = [
                'Subject', 'N',
                'GameRate', 'NeutralRate',
                'NormalizedLift',
                'McNemar_pvalue'
            ]

            df_display = df_results.copy()

            # Basic formatting
            df_display['N'] = df_display['N'].apply(
                lambda x: int(x) if pd.notna(x) else 0
            )

            # Format rates and lift
            for col in ['GameRate', 'NeutralRate', 'NormalizedLift']:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                )

            # Format p-value
            df_display['McNemar_pvalue'] = df_display['McNemar_pvalue'].apply(
                lambda p: ("N/A" if pd.isna(p)
                        else f"{p:.1e}" if p < 1e-4
                        else f"{p:.4f}")
            )

            print(df_display[display_columns].to_string(index=False))
            
            # Add significance column for summary
            df_results['Significant'] = df_results['McNemar_pvalue'].apply(
                lambda p: 'Yes' if pd.notna(p) and p < 0.05 else 'No'
            )
            
            print(f"\n--- Summary Statistics ---")
            print(f"Number of models with significant difference (p < 0.05): {sum(df_results['Significant'] == 'Yes')}/{len(df_results)}")
            print(f"Mean normalized lift: {df_results['NormalizedLift'].mean():.3f}")
            print(f"Models with positive normalized lift: {sum(df_results['NormalizedLift'] > 0)}/{len(df_results)}")
            
            plot_results(df_results, subject_order=model_list, dataset_name=dataset, corstr=corstr)
        else:
            print("No results to display.")

    except FileNotFoundError:
        print(f"Error: Input log file '{input_log_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()