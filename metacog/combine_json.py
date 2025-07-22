import json

def combine_json_files():
    """
    Combines two JSON files. It takes all entries from the 'results' dictionary
    of the first file and adds entries from the second file's 'results' dictionary
    only if their key (the question ID) is not already present in the first.
    """
    
    file1_path = 'compiled_results_smc/claude-3-sonnet-20240229_phase1_compiled.json'
    file2_path = 'compiled_results_smc/claude-3-sonnet-20240229_phase1_compiled_noprob.json'
    output_path = 'compiled_results_smc/combined_claude_sonnet.json'

    try:
        # Load the first JSON file
        with open(file1_path, 'r') as f:
            data1 = json.load(f)
        
        # Load the second JSON file
        with open(file2_path, 'r') as f:
            data2 = json.load(f)

        # Get the results dictionaries from both files
        results1 = data1.get('results', {})
        results2 = data2.get('results', {})

        # Start with a copy of the first file's data
        combined_data = data1
        combined_results = results1.copy()

        # Add entries from the second file if their question ID (key) is not in the first
        for qid, entry in results2.items():
            if qid not in combined_results:
                combined_results[qid] = entry
        
        # Update the results in our combined data object
        combined_data['results'] = combined_results

        # Write the combined data to a new JSON file
        with open(output_path, 'w') as f:
            json.dump(combined_data, f, indent=2)

        print(f"Successfully combined files. Result saved to {output_path}")
        print(f"Original file 1 had {len(results1)} entries.")
        print(f"Original file 2 had {len(results2)} entries.")
        print(f"Combined file has {len(combined_results)} entries.")


    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both input files exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    combine_json_files()