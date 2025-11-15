import json
import numpy as np

def calculate_entropy(probabilities):
  """Calculates the Shannon entropy of a probability distribution."""
  # The formula for entropy H(X) is: H(X) = -Σ [p(x) * log2(p(x))] for all x in X
  # where p(x) is the probability of outcome x.
  return -np.sum([p * np.log2(p) for p in probabilities.values() if p > 0])

import json
import numpy as np

def calculate_entropy(probabilities):
  """Calculates the Shannon entropy of a probability distribution."""
  # The formula for entropy H(X) is: H(X) = -Σ [p(x) * log2(p(x))] for all x in X
  return -np.sum([p * np.log2(p) for p in probabilities.values() if p > 0])

def analyze_json_files(file_path1, file_path2):
  """
  Compares two JSON files to count matching subject answers and calculate
  the average absolute entropy difference of their probability distributions.

  This version handles three formats for the "results" field:
  1. A dictionary of results keyed by question ID.
  2. A list of results where the question ID is in a nested 'question' object.
  3. A list of results where 'question_id' is a direct key in each object.
  """
  try:
    with open(file_path1, 'r') as f1:
      data1 = json.load(f1)
    with open(file_path2, 'r') as f2:
      data2 = json.load(f2)
  except FileNotFoundError:
    print("Error: One or both JSON files not found.")
    return
  except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    return

  results1 = data1.get("results", {})
  results2 = data2.get("results", {})

  # This block normalizes list-based results into a dictionary format
  # so the rest of the script can process it uniformly.
  def normalize_results(results):
    if not isinstance(results, list):
      return results # It's already a dictionary or empty
    
    normalized_dict = {}
    for item in results:
      # Handle format where question_id is a direct key
      if 'question_id' in item:
        normalized_dict[item['question_id']] = item
      # Handle format where question id is nested
      elif 'question' in item and 'id' in item['question']:
        normalized_dict[item['question']['id']] = item
    return normalized_dict

  results1 = normalize_results(results1)
  results2 = normalize_results(results2)

  same_answer_count = 0
  total_entropy_diff = 0
  question_count = 0

  for question_id, result1 in results1.items():
    if question_id in results2:
      result2 = results2[question_id]
      question_count += 1

      # 1. Count how often the "subject_answer" fields are the same
      if file_path1.startswith('capabilities_'):
        fieldstr = "subject_answer"
        if result1.get("subject_answer") == result2.get("subject_answer"):
          same_answer_count += 1
      elif file_path1.startswith('pass_'):
        fieldstr = "delegation_choice"
        if result1.get("delegation_choice") == result2.get("delegation_choice"):
            same_answer_count += 1

      # 2. Calculate the absolute entropy difference for the "probs" dicts
      probs1 = result1.get("probs", {})
      probs2 = result2.get("probs", {})

      if probs1 and probs2:
        entropy1 = calculate_entropy(probs1)
        entropy2 = calculate_entropy(probs2)
        total_entropy_diff += abs(entropy1 - entropy2)

  if question_count == 0:
      print("No common questions found between the two files.")
      return

  average_entropy_diff = total_entropy_diff / question_count

  print(f"Compared {question_count} common questions.")
  print(f"Number of matching '{fieldstr}' fields: {same_answer_count}")
  print(f"Average absolute entropy difference: {average_entropy_diff:.5f}")

if __name__ == "__main__":
#    analyze_json_files("capabilities_test_logs/qwen3-235b-a22b-2507_SimpleMC_500_1759008131_test_data.json", "capabilities_test_logs/qwen3-235b-a22b-2507_SimpleMC_500_1759006052_test_data.json")
#    analyze_json_files("capabilities_test_logs/qwen3-235b-a22b-2507_SimpleMC_500_1759008131_test_data.json", "capabilities_test_logs/qwen3-235b-a22b-2507_SimpleMC_500_1759008975_test_data.json")
#    analyze_json_files("capabilities_test_logs/qwen3-235b-a22b-2507_SimpleMC_500_1759006052_test_data.json", "capabilities_test_logs/qwen3-235b-a22b-2507_SimpleMC_500_1759008975_test_data.json")
#    analyze_json_files("capabilities_test_logs/deepseek-chat-v3.1_SimpleMC_500_1759025536_test_data.json", "capabilities_test_logs/deepseek-chat-v3.1_SimpleMC_500_1759023764_test_data.json")
    analyze_json_files("pass_game_logs/deepseek-chat-v3.1_SimpleMC_noqcnt_nopcnt_noscnt_temp0.0_1759029197_game_data.json", "pass_game_logs/deepseek-chat-v3.1_SimpleMC_noqcnt_nopcnt_noscnt_temp0.0_1759028404_game_data.json")