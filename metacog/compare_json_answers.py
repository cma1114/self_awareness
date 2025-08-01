import json

def compare_subject_answers(file1_path, file2_path):
    """
    Compares the 'subject_answer' field for each question in two JSON files.

    Args:
        file1_path (str): Path to the first JSON file.
        file2_path (str): Path to the second JSON file.
    """
    try:
        with open(file1_path, 'r') as f1:
            data1 = json.load(f1)
        with open(file2_path, 'r') as f2:
            data2 = json.load(f2)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from one of the files. Details: {e}")
        return

    differences_count = 0
    differing_ids = []

    results1 = data1.get('results', {})
    results2 = data2.get('results', {})

    all_ids = set(results1.keys()) | set(results2.keys())

    for record_id in sorted(all_ids):
        record1 = results1.get(record_id)
        record2 = results2.get(record_id)

        if not record1:
            print(f"Record '{record_id}' found in second file but not in first.")
            continue
        if not record2:
            print(f"Record '{record_id}' found in first file but not in second.")
            continue
            
        answer1 = record1.get('subject_answer')
        answer2 = record2.get('subject_answer')

        if answer1 != answer2:
            differences_count += 1
            differing_ids.append(record_id)

    if differing_ids:
        print("Question IDs with differing 'subject_answer':")
        for q_id in differing_ids:
            print(q_id)
    else:
        print("No differences found in 'subject_answer' between the two files.")

    print(f"\nTotal count of differences: {differences_count}")

if __name__ == "__main__":
    FILE1 = "compiled_results_smc/claude-3-haiku-20240307_phase1_compiled.json"
    FILE2 = "compiled_results_smc/claude-3-haiku-20240307_phase1_compiled_noprob.json"
    compare_subject_answers(FILE1, FILE2)