import os
import json
import subprocess
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta

DELEGATE_LOGS_DIR = "./delegate_game_logs"
ANALYSIS_SCRIPT = "./analyze_introspection.py"
OUTPUT_FILE = "consolidated_introspection_metrics.txt"

METRICS_TO_EXTRACT = [
    "ssag (Self-Selection Accuracy Gain)",
    "decision_quality_accuracy (DQA)",
    "ONIS_vs_AlwaysSelf",
    "regret_vs_oracle_using_P2_ans_conf",
    "auroc_P2_ans_conf_vs_S_i",
    "p2_game_cal_ece_ans_conf_vs_S_i",
    "dqa_loss_due_to_suboptimal_threshold (1-P(T) based)", # This is the one user mentioned
    "delegation_decision_auroc (1-P(T) based)",
    "deleg_signal_cal_ece",
    "sdt_criterion_c", # Added new metric
    "Capabilities AUROC", # Special handling for this one as it has a different format
    "capabilities_ece"
]

LOG_METRICS_TO_EXTRACT = [
    "Delegation to teammate occurred",
    "Phase 1 self-accuracy (from completed results, total - phase2)",
    "Phase 2 self-accuracy",
    "Statistical test (P2 self vs P1)"
]

# Pre-compile regex for metric extraction to improve performance and readability
# For metrics from analyze_introspection.py output
METRIC_PATTERNS = {
    metric: re.compile(r"^\s*" + re.escape(metric) + r":\s*(.*)$", re.IGNORECASE)
    for metric in METRICS_TO_EXTRACT if metric not in ["Capabilities AUROC", "sdt_criterion_c"] # Handle special cases
}
# Special pattern for Capabilities AUROC
METRIC_PATTERNS["Capabilities AUROC"] = re.compile(r"^\s*Capabilities AUROC \(P\(chosen_answer\) vs S_i from .*\):\s*([\d\.]+)\s*\(N=\d+\)$", re.IGNORECASE)
# Pattern for sdt_criterion_c (assuming similar format to others with CI and Point Est)
METRIC_PATTERNS["sdt_criterion_c"] = re.compile(r"^\s*sdt_criterion_c:\s*(.*)$", re.IGNORECASE)


# For metrics from .log files
LOG_METRIC_PATTERNS = {
    "Delegation to teammate occurred": re.compile(r"^\s*Delegation to teammate occurred in (.*)$"),
    "Phase 1 self-accuracy (from completed results, total - phase2)": re.compile(r"^\s*Phase 1 self-accuracy \(from completed results, total - phase2\): (.*)$"),
    "Phase 2 self-accuracy": re.compile(r"^\s*Phase 2 self-accuracy: (.*)$"),
    "Statistical test (P2 self vs P1)": re.compile(r"^\s*Statistical test \(P2 self vs P1\): (.*)$")
}

ALL_METRIC_KEYS_ORDERED = LOG_METRICS_TO_EXTRACT + METRICS_TO_EXTRACT


def extract_model_name(filename):
    """Extracts the model name from the filename (part before the first underscore)."""
    parts = filename.split('_')
    if len(parts) > 0:
        return parts[0]
    return "UnknownModel"

def extract_dataset_name(filename):
    """Extracts the dataset name from the filename (part between the first and second underscore)."""
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[1]
    return "UnknownDataset"

def get_phase1_subject_feedback(json_filepath):
    """Reads the JSON file and returns the value of phase1_subject_feedback."""
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        # Ensure the keys exist to avoid KeyError
        if "feedback_config" in data and "phase1_subject_feedback" in data["feedback_config"]:
            return bool(data["feedback_config"]["phase1_subject_feedback"])
        else:
            print(f"Warning: 'feedback_config' or 'phase1_subject_feedback' not found in {json_filepath}. Defaulting to False.")
            return False
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {json_filepath}. Skipping this file for feedback status.")
        return None # Indicates an error or inability to determine status
    except FileNotFoundError:
        print(f"Error: File not found {json_filepath}. Skipping this file for feedback status.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_filepath}: {e}. Skipping this file for feedback status.")
        return None

def run_analysis_and_extract_metrics(json_filepath):
    """Runs analyze_introspection.py and extracts the specified metrics."""
    extracted_metrics = {}
    try:
        # Ensure the analysis script is executable or called via python interpreter
        # Assuming analyze_introspection.py is a python script and needs to be run with python
        process = subprocess.run(
            ["python", ANALYSIS_SCRIPT, json_filepath],
            capture_output=True,
            text=True,
            check=True
        )
        output_lines = process.stdout.splitlines()

        for line in output_lines:
            for metric_name, pattern in METRIC_PATTERNS.items():
                match = pattern.match(line)
                if match:
                    # For "Capabilities AUROC", we only want the value, not the N part.
                    # The pattern is already designed to capture just the value.
                    extracted_metrics[metric_name] = match.group(1).strip()
                    break # Move to the next line once a metric is matched
        
        # Check if all expected metrics were found
        for metric_name in METRICS_TO_EXTRACT:
            if metric_name not in extracted_metrics:
                 # It's possible some metrics are not always present, provide a default or note
                extracted_metrics[metric_name] = "Not found"


    except subprocess.CalledProcessError as e:
        print(f"Error running {ANALYSIS_SCRIPT} on {json_filepath}: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Error: The script {ANALYSIS_SCRIPT} was not found.")
        return None # Or handle as a critical error
    except Exception as e:
        print(f"An unexpected error occurred during analysis of {json_filepath}: {e}")
        return None
    return extracted_metrics

def extract_log_file_metrics(log_filepath):
    """Reads a .log file and extracts specified metrics."""
    extracted_log_metrics = {key: "Not found" for key in LOG_METRICS_TO_EXTRACT}
    try:
        with open(log_filepath, 'r') as f:
            for line in f:
                for metric_name, pattern in LOG_METRIC_PATTERNS.items():
                    match = pattern.match(line)
                    if match:
                        extracted_log_metrics[metric_name] = match.group(1).strip()
                        # Optimization: if all log metrics found, can break early
                        # This requires checking if all "Not found" have been replaced
                        if all(val != "Not found" for val in extracted_log_metrics.values()):
                            return extracted_log_metrics
    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_filepath}")
        # Return dict with "Not found" for all log metrics
    except Exception as e:
        print(f"An error occurred while reading log file {log_filepath}: {e}")
        # Return dict with "Not found" for all log metrics
    return extracted_log_metrics

def main():
    """Main function to process logs and write output."""
    # Define temperature window (local time) and override pattern
    # These are naive datetime objects, representing local time.
    start_temp_dt_local = datetime(2025, 5, 19, 12, 0, 0) # 12:00 PM local
    end_temp_dt_local = datetime(2025, 5, 19, 16, 0, 0)   # 4:00 PM local
    temp_override_pattern = re.compile(r"_temp(\d+\.\d+)_")

    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    if not os.path.exists(DELEGATE_LOGS_DIR):
        print(f"Error: Directory not found: {DELEGATE_LOGS_DIR}")
        return

    if not os.path.isfile(ANALYSIS_SCRIPT):
        print(f"Error: Analysis script not found: {ANALYSIS_SCRIPT}")
        return

    for entry_filename in os.listdir(DELEGATE_LOGS_DIR):
        json_to_process_filename = None
        log_file_base_name = None # Used to derive the .log filename

        if "SimpleQA" in entry_filename:
            if entry_filename.endswith("_game_data_evaluated.json"):
                json_to_process_filename = entry_filename
                log_file_base_name = entry_filename.replace("_game_data_evaluated.json", "")
            else:
                # For SimpleQA, we only care about _game_data_evaluated.json. Skip others.
                continue
        elif entry_filename.endswith("_game_data.json"): # Not SimpleQA
            json_to_process_filename = entry_filename
            log_file_base_name = entry_filename.replace("_game_data.json", "")
        else:
            # Not a relevant JSON file (e.g., could be a .log file itself, or other type)
            continue

        # If no valid JSON file was identified for processing, skip to next directory entry
        if not json_to_process_filename or not log_file_base_name:
            continue

        json_filepath = os.path.join(DELEGATE_LOGS_DIR, json_to_process_filename)
        log_filename = log_file_base_name + ".log"
        log_filepath = os.path.join(DELEGATE_LOGS_DIR, log_filename)

        print(f"Processing JSON: {json_to_process_filename}, Corresponding LOG: {log_filename}")

        # Metadata extraction should use the base name that's consistent
        # between the (potentially _evaluated) json and its .log file.
        model_name = extract_model_name(log_file_base_name) # Use log_file_base_name for consistency
        dataset_name = extract_dataset_name(log_file_base_name) # Use log_file_base_name for consistency
        
        # Check if the actual JSON file to process exists before proceeding
        if not os.path.exists(json_filepath):
            print(f"Warning: Expected JSON file {json_filepath} not found. Skipping.")
            continue

        phase1_feedback = get_phase1_subject_feedback(json_filepath)

        if phase1_feedback is None:
            print(f"Skipping {entry_filename} due to issues reading feedback status from {json_to_process_filename}.")
            continue

        feedback_status_key = "Phase1_Feedback_True" if phase1_feedback else "Phase1_Feedback_False"
        combined_metrics = {}

        log_metrics = extract_log_file_metrics(log_filepath)
        combined_metrics.update(log_metrics)

        analysis_metrics = run_analysis_and_extract_metrics(json_filepath)
        if analysis_metrics:
            combined_metrics.update(analysis_metrics)
        else:
            print(f"Failed to extract analysis metrics for {json_to_process_filename}. Log metrics (if any) will still be included.")
            for m_key in METRICS_TO_EXTRACT:
                if m_key not in combined_metrics:
                    combined_metrics[m_key] = "Not found"
        
        if combined_metrics:
            temp_value_str = "0.0"  # Default
            # Check for temp override in the original filename from os.listdir
            temp_match = temp_override_pattern.search(entry_filename)

            if temp_match:
                temp_value_str = temp_match.group(1)
            else:
                # Fallback to timestamp check of the *processed* json_filepath
                # Convert file modification timestamp to naive local datetime
                file_mod_timestamp = os.path.getmtime(json_filepath)
                file_mod_datetime_local = datetime.fromtimestamp(file_mod_timestamp)
                
                if start_temp_dt_local <= file_mod_datetime_local <= end_temp_dt_local:
                    temp_value_str = "1.0"
            
            temp_indicator = f"(temp={temp_value_str})"
            # The key in results should be based on the original file iterated (entry_filename)
            # as this is what the user sees in the directory and what might contain the _temp override.
            filename_key_for_results = f"{entry_filename} {temp_indicator}"
            
            all_results[dataset_name][model_name][feedback_status_key][filename_key_for_results] = combined_metrics
        else:
            print(f"Failed to extract any metrics for {entry_filename} (derived from {json_to_process_filename})")


    # Write results to output file
    with open(OUTPUT_FILE, 'w') as f:
        f.write("Consolidated Introspection Metrics\n")
        f.write("==================================\n\n")

        for dataset_name, model_data in sorted(all_results.items()):
            f.write(f"Dataset: {dataset_name}\n")
            f.write("=" * (len(dataset_name) + 9) + "\n\n")
            for model_name, feedback_data in sorted(model_data.items()):
                f.write(f"  Model: {model_name}\n")
                f.write("  " + "-" * (len(model_name) + 7) + "\n")
                for feedback_status, file_data in sorted(feedback_data.items()):
                    f.write(f"    {feedback_status}:\n")
                    for game_file, metrics in sorted(file_data.items()):
                        f.write(f"      File: {game_file}\n")
                        for metric_name in ALL_METRIC_KEYS_ORDERED: # Use the combined ordered list
                            value = metrics.get(metric_name) # Get value, could be None if not found by .get()
                            # Skip if value is None, an empty string, "Not found", or the string "None"
                            if value and value != "Not found" and value != "None":
                                f.write(f"        {metric_name}: {value}\n")
                        f.write("\n") # Blank line after each file's metrics
                    f.write("\n") # Blank line after each feedback status section
                f.write("\n") # Blank line after each model section
            f.write("\n") # Blank line after each dataset section


    print(f"Processing complete. Results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()