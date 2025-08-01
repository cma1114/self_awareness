import json
import collections

def fix_discrepancies(input_file_path, output_file_path):
    """
    Parses a JSON file, corrects discrepancies, and writes the result to a new file.

    A discrepancy is found if the 'subject_answer' does not match the first key
    in the 'probs' dictionary. The 'probs' dictionary is assumed to be ordered
    from highest to lowest probability.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path to write the corrected JSON file.
    """
    try:
        with open(input_file_path, 'r') as f:
            # Use object_pairs_hook to preserve order, though modern Python dicts
            # generally preserve insertion order. This is for added safety.
            data = json.load(f, object_pairs_hook=collections.OrderedDict)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}")
        return

    discrepancy_count = 0
    
    if 'results' in data and isinstance(data['results'], dict):
        for record_id, record in data['results'].items():
            # Check if 'probs' is a non-empty dictionary
            if isinstance(record.get('probs'), dict) and record['probs']:
                # Get the first key from the 'probs' dictionary
                first_prob_key = next(iter(record['probs']))
                
                # Check for a discrepancy
                if record.get('subject_answer') != first_prob_key:
                    # Correct the 'subject_answer'
                    record['subject_answer'] = first_prob_key
                    discrepancy_count += 1

    # Write the (potentially) modified data to the output file
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Processing complete.")
    print(f"Found and corrected {discrepancy_count} discrepancies.")
    print(f"Corrected data saved to {output_file_path}")

if __name__ == "__main__":
    INPUT_FILE = "compiled_results_smc/claude-3-haiku-20240307_phase1_compiled.json"
    OUTPUT_FILE = "compiled_results_smc/claude-3-haiku-20240307_phase1_compiled_corrected.json"
    fix_discrepancies(INPUT_FILE, OUTPUT_FILE)