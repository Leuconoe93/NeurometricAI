# scripts/dataset_mmlu_100.py
#
# Downloads MMLU from HuggingFace, samples 100 questions
# stratified across subject categories.
#
# Output: data/mmlu_100.json
# Format: list of {id, question, choices, answer_idx, answer_text, subject}

import os
import json
import random
from collections import defaultdict

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CONFIG = {
    "n_questions"  : 100,
    "seed"         : 42,
    "output_file"  : "data/mmlu_100.json",
    # Broad subject categories to sample from
    "subjects"     : [
        "abstract_algebra", "anatomy", "astronomy",
        "college_biology", "college_chemistry", "college_physics",
        "college_mathematics", "college_medicine",
        "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_mathematics",
        "high_school_physics", "high_school_psychology",
        "human_aging", "human_sexuality",
        "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning",
        "medical_genetics", "miscellaneous",
        "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory",
        "professional_accounting", "professional_law",
        "professional_medicine", "professional_psychology",
        "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ],
}

CHOICE_LABELS = ["A", "B", "C", "D"]

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install: pip install datasets")
        return

    os.makedirs("data/", exist_ok=True)

    random.seed(CONFIG["seed"])

    all_items    = []
    failed_subjs = []

    print("Loading MMLU subjects ...")
    for subject in CONFIG["subjects"]:
        try:
            ds = load_dataset(
                "cais/mmlu", subject,
                split="validation", streaming = True,
            )
            for item in ds:
                choices = item["choices"]
                ans_idx = item["answer"]          # integer 0-3
                all_items.append({
                    "question"   : item["question"].strip(),
                    "choices"    : choices,
                    "answer_idx" : int(ans_idx),
                    "answer_text": choices[ans_idx],
                    "answer_label": CHOICE_LABELS[ans_idx],
                    "subject"    : subject,
                })
                if len(all_items) >= 2000:   # safety cap
                    break
        except Exception as e:
            failed_subjs.append(subject)
            continue

    print(f"Loaded {len(all_items)} questions across "
          f"{len(CONFIG['subjects']) - len(failed_subjs)} subjects")
    if failed_subjs:
        print(f"Failed subjects: {failed_subjs}")

    # Stratified sample across subjects
    by_subject  = defaultdict(list)
    for item in all_items:
        by_subject[item["subject"]].append(item)

    n_subjects  = len(by_subject)
    per_subject = max(1, CONFIG["n_questions"] // n_subjects)
    sampled     = []

    for subj, subj_items in by_subject.items():
        sampled.extend(
            random.sample(subj_items, min(per_subject, len(subj_items)))
        )

    random.shuffle(sampled)
    sampled   = sampled[:CONFIG["n_questions"]]

    # Top up if needed
    if len(sampled) < CONFIG["n_questions"]:
        remaining = [i for i in all_items if i not in sampled]
        random.shuffle(remaining)
        sampled += remaining[:CONFIG["n_questions"] - len(sampled)]

    questions = [{"id": i, **q} for i, q in enumerate(sampled)]

    with open(CONFIG["output_file"], "w") as f:
        json.dump(questions, f, indent=2)
    print(f"\nSaved {len(questions)} questions to {CONFIG['output_file']}")

    # Distribution
    subjects = defaultdict(int)
    for q in questions:
        subjects[q["subject"]] += 1
    print(f"\nSubjects represented: {len(subjects)}")
    print("\nPreview (first 3):")
    for q in questions[:3]:
        print(f"  Q{q['id']} [{q['subject']}]:")
        print(f"    {q['question'][:65]}...")
        for label, choice in zip(CHOICE_LABELS, q["choices"]):
            print(f"    {label}) {choice[:50]}")
        print(f"    → Correct: {q['answer_label']}) {q['answer_text']}")

if __name__ == "__main__":
    main()