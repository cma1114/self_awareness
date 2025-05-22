import json
import os
import re # Ensure re is imported for normalize_text

# Moved normalize_text to be globally available and ensure it's defined before use.
def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""
    text = str(text).lower() # Ensure text is string
    text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces
    return text.strip() # Strip leading/trailing

def load_game_data(filepath):
    """Loads game data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

def extract_phase2_details(game_data):
    """Extracts relevant details from Phase 2 results."""
    phase2_details = {}
    if game_data and "results" in game_data and isinstance(game_data["results"], list):
        for trial in game_data["results"]:
            # Assuming 'results' list directly contains phase 2 trials
            # and each trial has a 'question_id'
            q_id = trial.get("question_id")
            if q_id:
                phase2_details[q_id] = {
                    "delegation_choice": trial.get("delegation_choice"),
                    "subject_answer": trial.get("subject_answer"),
                    "subject_correct": trial.get("subject_correct"), # This might be None if not evaluated
                    "team_answer": trial.get("team_answer"),
                    "team_correct": trial.get("team_correct")
                }
    else:
        print("Warning: 'results' key not found or not a list in game data.")
    return phase2_details

def compare_runs(file1_path, file2_path, output_comparison_file="comparison_output.txt"):
    """
    Compares Phase 2 question handling between two delegate game runs.
    """
    # normalize_text is now globally defined and should be accessible.

    data1 = load_game_data(file1_path)
    data2 = load_game_data(file2_path)

    if not data1 or not data2:
        print("Comparison cannot proceed due to file loading errors.")
        return

    details1 = extract_phase2_details(data1)
    details2 = extract_phase2_details(data2)

    if not details1:
        print(f"No Phase 2 details extracted from {file1_path}")
    if not details2:
        print(f"No Phase 2 details extracted from {file2_path}")
    
    common_q_ids = set(details1.keys()).intersection(set(details2.keys()))

    output_lines = []
    output_lines.append(f"Comparison of Phase 2 question handling between:\n")
    output_lines.append(f"File 1: {file1_path}\n")
    output_lines.append(f"File 2: {file2_path}\n")
    output_lines.append(f"\nFound {len(common_q_ids)} common Phase 2 questions.\n")
    output_lines.append("="*50 + "\n")

    num_diff_delegation = 0
    num_diff_self_answer_correctness = 0 # For cases where both chose Self but correctness differs
    num_diff_self_answer_text = 0 # For cases where both chose Self but answer text differs

    for q_id in sorted(list(common_q_ids)):
        d1 = details1[q_id]
        d2 = details2[q_id]

        output_lines.append(f"Question ID: {q_id}\n")
        output_lines.append(f"  File 1: Choice='{d1['delegation_choice']}'")
        if d1['delegation_choice'] == "Self":
            output_lines.append(f"    Answer='{d1['subject_answer']}' (Correct: {d1['subject_correct']})\n")
        else:
            output_lines.append(f"    Team Answer='{d1['team_answer']}' (Correct: {d1['team_correct']})\n")

        output_lines.append(f"  File 2: Choice='{d2['delegation_choice']}'")
        if d2['delegation_choice'] == "Self":
            output_lines.append(f"    Answer='{d2['subject_answer']}' (Correct: {d2['subject_correct']})\n")
        else:
            output_lines.append(f"    Team Answer='{d2['team_answer']}' (Correct: {d2['team_correct']})\n")
        
        differences_found = False
        if d1['delegation_choice'] != d2['delegation_choice']:
            output_lines.append(f"  !! DIFFERENCE: Delegation choice differs.\n")
            num_diff_delegation += 1
            differences_found = True
        elif d1['delegation_choice'] == "Self": # Both chose Self, compare answers
            if normalize_text(d1['subject_answer']) != normalize_text(d2['subject_answer']):
                output_lines.append(f"  !! DIFFERENCE: Subject answers differ (though both chose Self).\n")
                num_diff_self_answer_text += 1
                differences_found = True
            # Check correctness only if answers are the same or if we want to highlight different outcomes for same/diff answers
            if d1['subject_correct'] != d2['subject_correct']:
                 output_lines.append(f"  !! DIFFERENCE: Subject answer correctness differs (both chose Self).\n")
                 num_diff_self_answer_correctness +=1 # This counts if correctness differs, even if text is same
                 differences_found = True
        
        if not differences_found:
            output_lines.append("  -- No difference in handling for this common question.\n")
        output_lines.append("-" * 30 + "\n")

    output_lines.append("\nSummary of Differences for Common Questions:\n")
    output_lines.append(f"- Number of questions with different delegation choices: {num_diff_delegation}\n")
    output_lines.append(f"- Number of 'Self' choices with different answer text: {num_diff_self_answer_text}\n")
    output_lines.append(f"- Number of 'Self' choices with different correctness outcomes: {num_diff_self_answer_correctness}\n")
    
    summary_str = "".join(output_lines)
    print(summary_str)

    with open(output_comparison_file, 'w', encoding='utf-8') as f_out:
        f_out.write(summary_str)
    print(f"\nComparison details saved to {output_comparison_file}")

    print(f"len(common_q_ids): {len(common_q_ids)}")

if __name__ == "__main__":
    file1 = "./delegate_game_logs/gemini-2.5-flash-preview-04-17_SimpleQA_50_100_team0.7_1747711733_game_data.json"
    file2 = "./delegate_game_logs/gemini-2.5-flash-preview-04-17_SimpleQA_50_200_team0.7_1747712168_game_data.json"
    
    # Check if files exist before running
    if not os.path.exists(file1):
        print(f"FATAL: Input file not found: {file1}")
    elif not os.path.exists(file2):
        print(f"FATAL: Input file not found: {file2}")
    else:
        compare_runs(file1, file2)

    