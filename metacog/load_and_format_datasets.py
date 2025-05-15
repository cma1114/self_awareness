from datasets import load_dataset
import random
import os

random.seed(42)  # For reproducibility
hf_token = os.environ.get("HF_TOKEN")

def load_and_format_dataset(dataset_name, num_questions_needed=None, split=None, skip_questions=None):
    if dataset_name=="GPQA":
        if split is None:
            return load_and_format_gpqa(num_questions_needed, hf_token=hf_token, skip_questions=skip_questions)
        else:
            return load_and_format_gpqa(num_questions_needed, hf_token=hf_token, split=split, skip_questions=skip_questions)
    elif dataset_name=="MMLU":
        if split is None:
            return load_and_format_mmlu(num_questions_needed, skip_questions=skip_questions)
        else:
            return load_and_format_mmlu(num_questions_needed, split=split, skip_questions=skip_questions)
    elif dataset_name=="TruthfulQA":
        if split is None:
            return load_and_format_truthfulqa(num_questions_needed, skip_questions=skip_questions)
        else:
            return load_and_format_truthfulqa(num_questions_needed, split=split, skip_questions=skip_questions)
    elif dataset_name=="SimpleQA":
        if split is None:
            return load_and_format_simpleqa(num_questions_needed, skip_questions=skip_questions)
        else:
            return load_and_format_simpleqa(num_questions_needed, split=split, skip_questions=skip_questions)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Supported datasets are: GPQA, MMLU, TruthfulQA.")

def load_and_format_gpqa(num_questions_needed=None, hf_token=None, split="train", skip_questions=None):
    """
    Loads the GPQA dataset and formats questions into the A-D multiple-choice format.
    """
    print(f"Attempting to load GPQA ({split} split)...")
    dataset_name = "Idavidrein/gpqa"
    config_name = "gpqa_main"
    try:
        dataset = load_dataset(dataset_name, config_name, split=split, token=hf_token, trust_remote_code=True)
        print("GPQA Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading GPQA dataset '{dataset_name}' ({config_name}, {split}): {e}")
        print("Please ensure you have the 'datasets' library installed, an internet connection,")
        print(f"and potentially a valid Hugging Face token if required (passed as hf_token).")
        return None

    formatted_questions = []
    question_ids_added = set()
    required_fields = ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Record ID']

    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)

    if not num_questions_needed: num_questions_needed = len(dataset)
    print(f"Formatting {num_questions_needed} questions from GPQA...")

    bad_ids=["recgCB0HSVt2IslDN"]
    for idx in dataset_indices:
        if len(formatted_questions) >= num_questions_needed:
            break

        item = dataset[idx]
        if skip_questions and item['Question'] in skip_questions:
            print(f"DEBUG: Skipping question '{item['Question'][:50]}...' as it's in skip_questions")
            continue

        # Check if all required fields exist and are not None/empty
        if not all(item.get(field) for field in required_fields):
            continue

        record_id = item['Record ID']

        # Apply filtering
        if record_id in bad_ids:
            continue

        # Check if ID already added
        if record_id in question_ids_added:
            continue

        # Gather options
        correct_answer_text = item['Correct Answer'].strip()
        incorrect_answers_text = [
            item['Incorrect Answer 1'].strip(),
            item['Incorrect Answer 2'].strip(),
            item['Incorrect Answer 3'].strip()
        ]
        if len(correct_answer_text) == 0 or any(len(ans) == 0 for ans in incorrect_answers_text):
            continue

        # Create the pool of 4 options and shuffle
        options_list = [correct_answer_text] + incorrect_answers_text
        random.shuffle(options_list)

        # Assign labels (A-D) and find the correct one
        options_dict = {}
        correct_label = None
        labels = ["A", "B", "C", "D"]
        
        for i, option_text in enumerate(options_list):
            label = labels[i]
            options_dict[label] = option_text
            if option_text == correct_answer_text:
                correct_label = label

        # Create the formatted question
        formatted_q = {
            "id": f"gpqa_{split}_{record_id}",
            "question": item['Question'],
            "options": options_dict,
            "correct_answer": correct_label
        }
        formatted_questions.append(formatted_q)
        question_ids_added.add(record_id)
        
    if len(formatted_questions) < num_questions_needed:
        print(f"Warning: Only able to format {len(formatted_questions)} unique questions, but {num_questions_needed} were requested.")

    print(f"Successfully formatted {len(formatted_questions)} unique questions from GPQA.")
    return formatted_questions

def load_and_format_mmlu(num_questions_needed=None, split="auxiliary_train", skip_questions=None):
    """
    Loads the MMLU dataset and formats questions into the A-D multiple-choice format.
    """
    print(f"Attempting to load MMLU ({split} split)...")
    try:
        dataset = load_dataset("cais/mmlu", "all", split=split)
        print("MMLU Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        print("Please ensure you have the 'datasets' library installed and an internet connection.")
        return None

    formatted_questions = []
    questions_seen = set()  # Track unique questions by their text

    # Shuffle dataset to get random questions
    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)

    if not num_questions_needed: num_questions_needed = len(dataset)
    print(f"Formatting {num_questions_needed} questions from MMLU...")
    for idx in dataset_indices:
        if len(formatted_questions) >= num_questions_needed:
            break

        item = dataset[idx]
        
        # Extract data
        question_text = item.get('question')
        if skip_questions is not None and question_text in skip_questions:
            continue
        choices = item.get('choices')
        answer_idx = item.get('answer')  # Integer index of correct answer
        
        # Basic validation
        if not all([question_text, choices, isinstance(answer_idx, int)]):
            continue
            
        # Ensure we have exactly 4 options
        if len(choices) != 4:
            continue
            
        # Verify the answer index is valid
        if answer_idx < 0 or answer_idx >= len(choices):
            continue
            
        # Skip duplicate questions
        if question_text in questions_seen:
            continue
        questions_seen.add(question_text)
            
        # Assign labels and find the correct one
        options_dict = {}
        labels = ["A", "B", "C", "D"]
        for i, option_text in enumerate(choices):
            label = labels[i]
            options_dict[label] = option_text
        
        # Get the correct answer label
        correct_label = labels[answer_idx]

        # Create the formatted dictionary
        formatted_q = {
            "id": f"mmlu_{idx}",
            "question": question_text,
            "options": options_dict,
            "correct_answer": correct_label
        }
        formatted_questions.append(formatted_q)

    if len(formatted_questions) < num_questions_needed:
        print(f"Warning: Only able to format {len(formatted_questions)} unique questions, but {num_questions_needed} were requested.")

    print(f"Successfully formatted {len(formatted_questions)} unique questions from MMLU.")
    return formatted_questions

def load_and_format_truthfulqa(num_questions_needed=None, split="validation", skip_questions=None):
    """
    Loads the TruthfulQA dataset and formats questions into the A-D multiple-choice format.
    """
    print(f"Attempting to load TruthfulQA ({split} split)...")
    try:
        dataset = load_dataset("truthful_qa", "generation", split=split, trust_remote_code=True)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading TruthfulQA dataset: {e}")
        print("Please ensure you have the 'datasets' library installed (`pip install datasets`)")
        print("and an internet connection. You might also need `trust_remote_code=True`.")
        return None

    formatted_questions = []

    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)

    question_ids_added = set()  # Keep track of IDs to ensure uniqueness

    if not num_questions_needed: num_questions_needed = len(dataset)
    print(f"Formatting {num_questions_needed} questions...")
    for idx in dataset_indices:
        if len(formatted_questions) >= num_questions_needed:
            break

        item = dataset[idx]
        potential_id = f"tqa_{split}_{idx}"

        question_text = item.get('question')
        if skip_questions is not None and question_text in skip_questions:
            continue
        best_answer = item.get('best_answer')
        if len(best_answer.strip()) == 0:
            continue
        incorrect_answers = item.get('incorrect_answers')

        # Basic validation of required fields
        if not all([question_text, best_answer, incorrect_answers]):
            continue

        # Need at least 3 incorrect answers to form 4 options
        if not isinstance(incorrect_answers, list) or len(incorrect_answers) < 3:
            continue

        # Ensure best_answer is not accidentally in the chosen incorrect list
        possible_incorrect = [ans for ans in incorrect_answers if ans != best_answer and len(ans.strip()) > 0]
        if len(possible_incorrect) < 3:
            continue

        # Select 3 distinct incorrect answers
        try:
            chosen_incorrect = random.sample(possible_incorrect, 3)
        except ValueError:
            continue

        # Create the pool of options and shuffle
        options_list = [best_answer] + chosen_incorrect
        random.shuffle(options_list)

        # Assign labels and find the correct one
        options_dict = {}
        correct_label = None
        labels = ["A", "B", "C", "D"]
        for i, option_text in enumerate(options_list):
            label = labels[i]
            options_dict[label] = option_text
            if option_text == best_answer:
                correct_label = label

        # Create the formatted dictionary
        formatted_q = {
            "id": potential_id,
            "question": question_text,
            "options": options_dict,
            "correct_answer": correct_label
        }
        formatted_questions.append(formatted_q)
        question_ids_added.add(potential_id)

    if len(formatted_questions) < num_questions_needed:
        print(f"Warning: Only able to format {len(formatted_questions)} unique questions, but {num_questions_needed} were requested.")

    print(f"Successfully formatted {len(formatted_questions)} unique questions from TruthfulQA.")
    return formatted_questions

def load_and_format_simpleqa(num_questions_needed=None, split="test", skip_questions=None):
    print(f"Attempting to load SimpleQA ({split} split)...")
    try:
        dataset = load_dataset("basicv8vc/SimpleQA", split=split)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading SimpleQA dataset: {e}")
        return None

    formatted_questions = []

    dataset_indices = list(range(len(dataset)))
    random.shuffle(dataset_indices)

    question_ids_added = set()  # Keep track of IDs to ensure uniqueness

    if not num_questions_needed: num_questions_needed = len(dataset)
    print(f"Formatting {num_questions_needed} questions...")
    for idx in dataset_indices:
        if len(formatted_questions) >= num_questions_needed:
            break

        item = dataset[idx]
        potential_id = f"sqa_{split}_{idx}"

        question_text = item.get('problem')
        if skip_questions is not None and question_text in skip_questions:
            continue
        best_answer = item.get('answer')
        if len(best_answer.strip()) == 0:
            continue

        # Create the formatted dictionary
        formatted_q = {
            "id": potential_id,
            "question": question_text,
            "correct_answer": best_answer
        }
        formatted_questions.append(formatted_q)
        question_ids_added.add(potential_id)

    if len(formatted_questions) < num_questions_needed:
        print(f"Warning: Only able to format {len(formatted_questions)} unique questions, but {num_questions_needed} were requested.")

    print(f"Successfully formatted {len(formatted_questions)} unique questions from SimpleQA.")
    return formatted_questions
