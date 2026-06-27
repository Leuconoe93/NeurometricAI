# scripts/dataset_gsm8k_100.py
#
# Downloads GSM8K from HuggingFace, samples 100 questions reproducibly,
# extracts ground truth numerical answers, and saves to disk.
#
# Output: data/gsm8k_100.json
# Format: list of {id, question, answer_text, answer_number}

import os
import re
import json
import random

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    "n_questions" : 100,
    "seed"        : 42,
    "split"       : "test",       # use test split — cleaner, less contamination
    "data_dir"    : "data/",
    "output_file" : "data/gsm8k_100.json",
}

# ── Answer extraction ──────────────────────────────────────────────────────────

def extract_answer_number(answer_text):
    """
    GSM8K answers end with '#### <number>'.
    Extract the final numerical answer as a float.
    Returns None if extraction fails.
    """
    match = re.search(r"####\s*([\-\d,\.]+)", answer_text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None
    return None

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install: pip install datasets")
        return

    os.makedirs(CONFIG["data_dir"], exist_ok=True)

    print(f"Loading GSM8K ({CONFIG['split']} split) from HuggingFace ...")
    dataset = load_dataset("openai/gsm8k", "main",
                           split=CONFIG["split"])
    print(f"Total questions available: {len(dataset)}\n")

    # Reproducible sampling
    random.seed(CONFIG["seed"])
    indices  = random.sample(range(len(dataset)), CONFIG["n_questions"])
    indices  = sorted(indices)

    questions = []
    skipped   = 0

    for i, idx in enumerate(indices):
        item        = dataset[idx]
        question    = item["question"].strip()
        answer_text = item["answer"].strip()
        answer_num  = extract_answer_number(answer_text)

        if answer_num is None:
            skipped += 1
            continue

        questions.append({
            "id"            : i,
            "original_idx"  : idx,
            "question"      : question,
            "answer_text"   : answer_text,
            "answer_number" : answer_num,
        })

    print(f"Sampled  : {CONFIG['n_questions']} questions")
    print(f"Parsed   : {len(questions)} with valid numerical answers")
    print(f"Skipped  : {skipped} (answer parsing failed)\n")

    # Save
    with open(CONFIG["output_file"], "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved to {CONFIG['output_file']}")

    # Preview first 3
    print("\nPreview (first 3 questions):")
    for q in questions[:3]:
        print(f"  Q{q['id']}: {q['question'][:80]}...")
        print(f"        Answer: {q['answer_number']}")

if __name__ == "__main__":
    main()