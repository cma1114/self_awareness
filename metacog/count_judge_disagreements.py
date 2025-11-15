#!/usr/bin/env python3
import json
import glob
from pathlib import Path

disagreements = []
total_questions = 0

for filepath in glob.glob("*_phase1_compiled.json"):
    with open(filepath) as f:
        print(f"Processing {Path(filepath).name}...")
        data = json.load(f)
    
    for question_id, result in data["results"].items():
        total_questions += 1
        if "judgments" not in result or not result["judgments"]:
            continue
        judgments = result["judgments"]
        
        # Check if all judgment values are the same
        values = list(judgments.values())
        if len(set(values)) > 1:  # Disagreement exists
            disagreements.append(result)

percentage = (len(disagreements) / total_questions * 100) if total_questions > 0 else 0

print(f"Total questions: {total_questions}")
print(f"Disagreements: {len(disagreements)}")
print(f"Disagreement rate: {percentage:.2f}%")

with open("disagreements.json", "w") as f:
    json.dump(disagreements, f, indent=2)

print(f"\nDisagreements saved to disagreements.json")