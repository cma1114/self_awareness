import re
import pandas as pd
from io import StringIO

def parse_accuracy_from_log(log_content: str) -> pd.DataFrame:
    """
    Parses a log file to extract Phase 2 self-accuracy for Incorrect trials.

    Args:
        log_content: A string containing the entire log file.

    Returns:
        A pandas DataFrame with the extracted data.
    """
    # Regex to find the start of an "Incorrect" analysis block and capture the model name.
    model_re = re.compile(r"^--- Analyzing ([\w.-]+) \(Redacted, Incorrect, .*")

    # CORRECTED Regex: Changed `\s+` to `\s*` to match lines with zero or more leading spaces.
    accuracy_re = re.compile(
        r"^\s*Phase 2 self-accuracy: ([\d.]+) \[([\d.]+), ([\d.]+)\] \(n=(\d+)\)"
    )

    results_data = []
    current_model = None

    # Process the log content line by line
    for line in StringIO(log_content):
        # Check if the line marks the start of a relevant block
        model_match = model_re.search(line)
        if model_match:
            # Found a new "Incorrect" block, store its model name
            current_model = model_match.group(1)
            continue

        # If we are inside an "Incorrect" block, look for its accuracy line
        if current_model:
            accuracy_match = accuracy_re.search(line)
            if accuracy_match:
                # Found the data, extract and store it
                accuracy, lower, upper, n = accuracy_match.groups()
                results_data.append({
                    "model": current_model,
                    "accuracy": float(accuracy),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "n": int(n),
                })
                # Reset state to avoid carrying over the model name
                current_model = None

    # Create a DataFrame for clean formatting
    df = pd.DataFrame(results_data)
    
    # Rename columns for the final table
    df.rename(columns={
        'model': 'Model',
        'accuracy': 'Accuracy',
        'lower_bound': 'Lower Bound',
        'upper_bound': 'Upper Bound',
        'n': 'N'
    }, inplace=True)
    
    return df

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_accuracy_chart(df: pd.DataFrame, model_order: list, dataset):
    """
    Generates a bar chart of model accuracies with confidence intervals.

    Args:
        df (pd.DataFrame): DataFrame containing model performance data.
                           Must include 'Model', 'Accuracy', 'Lower Bound', 
                           'Upper Bound' columns.
        model_order (list): A list of model names to specify the order of
                            the bars on the chart.
    """
    # --- 1. Data Preparation ---
    # Filter out models from the order list that are not in the DataFrame
    models_in_df = df['Model'].unique()
    ordered_models_to_plot = [m for m in model_order if m in models_in_df]
    
    # Reorder the DataFrame according to the provided list
    df_sorted = df.set_index('Model').reindex(ordered_models_to_plot).reset_index()

    # Calculate the size of the error bars (distance from the mean)
    # yerr expects a 2xN array for asymmetric errors: [lower_errors, upper_errors]
    lower_error = df_sorted['Accuracy'] - df_sorted['Lower Bound']
    upper_error = df_sorted['Upper Bound'] - df_sorted['Accuracy']
    asymmetric_error = [lower_error, upper_error]

    # --- 2. Plotting ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the bars with error bars
    ax.bar(
        df_sorted['Model'],
        df_sorted['Accuracy'],
        yerr=asymmetric_error,
        capsize=5,  # Adds caps to the error bars
        color='cornflowerblue',
        edgecolor='black',
        alpha=0.8
    )

    # Add the horizontal line at 25% for chance
    ax.axhline(
        y=0.25,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Chance Accuracy (25%)'
    )

    # --- 3. Styling and Formatting ---
    # Set titles and labels
    ax.set_title(f'Game Accuracy on Incorrect Trials: {dataset}', fontsize=18, pad=20)
    ax.set_ylabel('Self-Accuracy (Corrects Initial Wrong Answer)', fontsize=14)
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1)

    # Rotate and align x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Add a light grid for better readability
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
    ax.set_axisbelow(True) # Ensure grid is behind bars

    # Add a legend for the chance line
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"secondchance_accuracy_{dataset}.png", dpi=300)

model_list = ['grok-3-latest', 'claude-sonnet-4-20250514', 'gemini-2.5-flash-preview-04-17', 'gpt-4.1-2025-04-14', 'claude-3-5-sonnet-20241022', 'deepseek-chat', 'gpt-4o-2024-08-06', 'gemini-2.0-flash-001', 'gemini-1.5-pro', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']

dataset = "GPQA"

file_path = f'analysis_log_multi_logres_sc_{dataset.lower()}_new.txt'
with open(file_path, 'r') as f:
    log_content = f.read()

# Parse the log and get the DataFrame
accuracy_df = parse_accuracy_from_log(log_content)

# Set display options for printing the final table
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# Print the final result
if not accuracy_df.empty:
    print(accuracy_df.to_string(index=False))
    plot_accuracy_chart(accuracy_df, model_list, dataset)

else:
    print("Error: Could not find any matching data. The DataFrame is empty.")