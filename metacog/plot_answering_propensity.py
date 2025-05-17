import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_phase2_decisions(log_file_path):
    """
    Parses the log file to extract Phase 2 delegation decisions.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        list: A list of booleans, True if the model answered, False if delegated.
              Returns an empty list if Phase 2 data isn't found or file doesn't exist.
    """
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return []

    decisions = []
    in_phase2 = False
    
    # Regex to identify Phase 2 start and decision lines
    # Phase 2 start can be identified by a line like "Phase 2: Answer or Delegate"
    # Decision lines contain "--> Your answer:" or "--> Delegating to teammate..."
    phase2_start_regex = re.compile(r"Phase 2: Answer or Delegate")
    answered_regex = re.compile(r"--> Your answer:")
    delegated_regex = re.compile(r"--> Delegating to teammate...")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not in_phase2:
                    if phase2_start_regex.search(line):
                        in_phase2 = True
                        print("Found start of Phase 2.")
                        continue  # Move to next line after finding phase 2 start
                
                if in_phase2:
                    if answered_regex.search(line):
                        decisions.append(True)
                    elif delegated_regex.search(line):
                        decisions.append(False)
                    # Could add a stop condition if there's a clear "Phase 2 Complete" or similar log
                    # For now, it will parse all such lines after Phase 2 starts.
    except Exception as e:
        print(f"Error reading or parsing log file: {e}")
        return []
        
    print(f"Found {len(decisions)} Phase 2 decisions.")
    return decisions

def calculate_binned_answering_percentage(decisions, bin_size=25):
    """
    Calculates the percentage of 'answer' decisions in successive bins.

    Args:
        decisions (list): List of booleans (True for answer, False for delegate).
        bin_size (int): The number of trials per bin.

    Returns:
        tuple: (list of percentages, list of bin labels)
    """
    if not decisions:
        return [], []

    percentages = []
    bin_labels = []
    num_bins = (len(decisions) + bin_size - 1) // bin_size  # Ceiling division

    for i in range(num_bins):
        start_index = i * bin_size
        end_index = min((i + 1) * bin_size, len(decisions))
        current_bin_decisions = decisions[start_index:end_index]
        
        if not current_bin_decisions:
            continue

        num_answered = sum(current_bin_decisions)
        percentage_answered = (num_answered / len(current_bin_decisions)) * 100
        percentages.append(percentage_answered)
        
        bin_label = f"Trials {start_index + 1}-{end_index}"
        bin_labels.append(bin_label)
        
    return percentages, bin_labels

def plot_answering_propensity(percentages, bin_labels, output_filename="answering_propensity_plot.png"):
    """
    Plots the binned answering propensity.

    Args:
        percentages (list): List of percentages for each bin.
        bin_labels (list): List of labels for each bin.
        output_filename (str): Filename to save the plot.
    """
    if not percentages or not bin_labels:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 7))
    
    x_pos = np.arange(len(bin_labels))
    plt.bar(x_pos, percentages, color='skyblue')

    plt.xlabel("Trial Bins")
    plt.ylabel("Percentage of Trials Answered by Model (%)")
    plt.title("Model's Answering Propensity Over Course of Phase 2 (Binned)")
    plt.xticks(x_pos, bin_labels, rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    # plt.show() # Uncomment to display plot if running in an environment that supports it

if __name__ == "__main__":
    # The user specified this log file.
    # Assuming it's in a 'delegate_game_logs' subdirectory relative to this script,
    # or in the same directory if run from the project root.
    log_file_name = "grok-3-latest_GPQA_50_200_team0.75_1747442192.log"
    
    # Try to find the log file in common locations
    possible_paths = [
        os.path.join("delegate_game_logs", log_file_name),
        log_file_name 
    ]
    
    log_file_path_to_use = None
    for path_option in possible_paths:
        if os.path.exists(path_option):
            log_file_path_to_use = path_option
            break
            
    if log_file_path_to_use:
        print(f"Using log file: {log_file_path_to_use}")
        phase2_decisions = parse_phase2_decisions(log_file_path_to_use)
        
        if phase2_decisions:
            bin_percentages, labels = calculate_binned_answering_percentage(phase2_decisions, bin_size=25)
            plot_answering_propensity(bin_percentages, labels, output_filename="grok_answering_propensity.png")
        else:
            print("Could not extract Phase 2 decisions or no decisions found.")
    else:
        print(f"Error: Log file '{log_file_name}' not found in ./ or ./delegate_game_logs/")
        print("Please ensure the log file is in the correct location or update the path in the script.")