import json
import os
from collections import defaultdict
from load_and_format_datasets import load_and_format_dataset
import sys

def analyze_delegation_by_domain(gpqa_questions_with_domain, game_logs_dir):
    """
    Analyzes game log files to determine delegation rates for Biology vs. Non-Biology questions.

    Args:
        gpqa_questions_with_domain (list): List of dicts from load_and_format_dataset("GPQA"),
                                           each containing "id" and "high_level_domain".
        game_logs_dir (str): Path to the directory containing _game_data.json files.
    """
    if not gpqa_questions_with_domain:
        print("Error: GPQA question list with domain info is empty or not provided.", file=sys.stderr)
        return

    feature_lookup = {
        item['id']: {
            'difficulty': item['difficulty_score'],
            'overlap_ratio': item.get('overlap_ratio', 0),
            'domain': item['high_level_domain'],
            'question_text': item['question'] 
        } 
        for item in gpqa_questions_with_domain if item.get('id') 
    }

    if not feature_lookup:
        print("Error: Could not create a domain lookup map from GPQA questions. Ensure items have 'id' and 'high_level_domain'.", file=sys.stderr)
        return

    print(f"Analyzing GPQA game log files in directory: {game_logs_dir}\n")

    for filename in sorted(os.listdir(game_logs_dir)):
        if filename.endswith("_game_data.json") and "_GPQA_" in filename:
            file_path = os.path.join(game_logs_dir, filename)
            print(f"--- Processing File: {filename} ---")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
            except Exception as e:
                print(f"  Error reading/parsing {filename}: {e}. Skipping.", file=sys.stderr)
                continue
            
            # Assuming phase 2 trials are in game_data["results"] which is a list
            phase2_trials = [t for t in game_data.get("results", []) if t.get('phase') == 2]

            if not phase2_trials:
                # Fallback: some of your earlier files might have results directly under the root
                # This is less likely for _game_data.json files from DelegateGameFromCapabilities
                # but good to have a small check if "results" isn't a list of P2 trials.
                # For now, we strictly expect phase 2 trials in game_data["results"] list.
                print(f"  Warning: No Phase 2 trials found in 'results' list for {filename}. Skipping.")
                continue

            bio_delegated_count = 0
            bio_total_count = 0
            nonbio_delegated_count = 0
            nonbio_total_count = 0
            unknown_domain_qids_count = 0
            difficulties_delegated, difficulties_answered = [], []
            overlap_ratios_delegated, overlap_ratios_answered = [], []

            for trial in phase2_trials:
                q_id = trial.get("question_id")
                delegation_choice = trial.get("delegation_choice")

                if not q_id or not delegation_choice:
                    # print(f"  Skipping trial due to missing q_id or delegation_choice: {trial}", file=sys.stderr)
                    continue

                domain = feature_lookup.get(q_id, {}).get("domain")
                if not domain:
                    unknown_domain_qids_count += 1
                    continue
                
                is_delegated = (delegation_choice == "Teammate")
                if is_delegated:
                    difficulties_delegated.append(feature_lookup.get(q_id, {}).get("difficulty", 0))
                    overlap_ratios_delegated.append(feature_lookup.get(q_id, {}).get("overlap_ratio", 0))
                else:
                    difficulties_answered.append(feature_lookup.get(q_id, {}).get("difficulty", 0))
                    overlap_ratios_answered.append(feature_lookup.get(q_id, {}).get("overlap_ratio", 0))

                if domain.lower() == "biology": # Case-insensitive check for "Biology"
                    bio_total_count += 1
                    if is_delegated:
                        bio_delegated_count += 1
                else:
                    nonbio_total_count += 1
                    if is_delegated:
                        nonbio_delegated_count += 1
            
            if unknown_domain_qids_count > 0:
                print(f"  Warning: Skipped {unknown_domain_qids_count} P2 trial QIDs not found in GPQA domain lookup.")

            if bio_total_count == 0 and nonbio_total_count == 0 and unknown_domain_qids_count > 0:
                print(f"  No P2 trial QIDs in this file matched the GPQA domain lookup. Check QID formats.")
                print("-" * 30 + "\n")
                continue


            bio_delegation_percent = (bio_delegated_count / bio_total_count) * 100 if bio_total_count > 0 else float('nan')
            nonbio_delegation_percent = (nonbio_delegated_count / nonbio_total_count) * 100 if nonbio_total_count > 0 else float('nan')

            print(f"  Biology Questions:")
            print(f"    Delegation Rate: {bio_delegation_percent:.1f}% (Delegated: {bio_delegated_count}, Total: {bio_total_count})")
            print(f"  Non-Biology Questions:")
            print(f"    Delegation Rate: {nonbio_delegation_percent:.1f}% (Delegated: {nonbio_delegated_count}, Total: {nonbio_total_count})")
            if bio_total_count > 0 and nonbio_total_count > 0:
                diff = bio_delegation_percent - nonbio_delegation_percent
                print(f"  Difference (Bio Rate - NonBio Rate): {diff:.1f} percentage points")
            if difficulties_delegated:
                avg_diff_delegated = sum(difficulties_delegated) / len(difficulties_delegated)
                print(f"  Avg Difficulty for DELEGATED: {avg_diff_delegated:.2f} (N={len(difficulties_delegated)})")
            if difficulties_answered:
                avg_diff_answered = sum(difficulties_answered) / len(difficulties_answered)
                print(f"  Avg Difficulty for ANSWERED: {avg_diff_answered:.2f} (N={len(difficulties_answered)})")
            if overlap_ratios_delegated:
                avg_overlap_delegated = sum(overlap_ratios_delegated) / len(overlap_ratios_delegated)
                print(f"  Avg Overlap Ratio for DELEGATED: {avg_overlap_delegated:.2f} (N={len(overlap_ratios_delegated)})")
            if overlap_ratios_answered:
                avg_overlap_answered = sum(overlap_ratios_answered) / len(overlap_ratios_answered)
                print(f"  Avg Overlap Ratio for ANSWERED: {avg_overlap_answered:.2f} (N={len(overlap_ratios_answered)})")
            
            print("-" * 30 + "\n")

if __name__ == "__main__":
    print("Loading GPQA dataset for domain information...")
    # Ensure your load_and_format_dataset("GPQA") returns items with "id" and "high_level_domain"
    gpqa_data_with_domains = load_and_format_dataset("GPQA") 

    if not gpqa_data_with_domains:
        print("Failed to load GPQA data with domains. Exiting.", file=sys.stderr)
        exit()
    
    # Quick check for required fields in the first item
    if gpqa_data_with_domains and not all(k in gpqa_data_with_domains[0] for k in ["id", "high_level_domain"]):
        print("Error: Loaded GPQA data items are missing 'id' or 'high_level_domain'.", file=sys.stderr)
        print(f"Example item: {gpqa_data_with_domains[0] if gpqa_data_with_domains else 'Empty data'}", file=sys.stderr)
        exit()
        
    print(f"Successfully loaded {len(gpqa_data_with_domains)} GPQA questions with domain info.")

    game_logs_directory = "./delegate_game_logs/" # Your specified directory
    if not os.path.isdir(game_logs_directory):
        print(f"Error: Directory not found: {game_logs_directory}", file=sys.stderr)
        exit()

    analyze_delegation_by_domain(gpqa_data_with_domains, game_logs_directory)