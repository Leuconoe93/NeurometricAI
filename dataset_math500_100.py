# scripts/dataset_math500_100.py
#
# Downloads MATH-500 from HuggingFace, samples 100 problems
# stratified across difficulty levels and subjects.
#
# Output: data/math500_100.json
# Format: list of {id, problem, solution, answer, subject, level}

import os
import json
import random
from collections import defaultdict

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CONFIG = {
    "n_questions" : 100,
    "seed"        : 42,
    "output_file" : "data/math500_100.json",
}

def extract_boxed_answer(solution):
    """
    MATH-500 answers are wrapped in \\boxed{}.
    Extract the final boxed expression as the ground truth.
    """
    import re
    matches = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if matches:
        return matches[-1].strip()
    # Fallback: last number in solution
    numbers = re.findall(r"-?\d+\.?\d*", solution)
    return numbers[-1] if numbers else None

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install: pip install datasets")
        return

    os.makedirs("data/", exist_ok=True)

    print("Loading MATH-500 from HuggingFace ...")
    # MATH-500 is available as hendrycks/competition_math or
    # lighteval/MATH-Hard — try both
    try:
        dataset = load_dataset(
            "HuggingFaceH4/MATH-500",
            split     = "test",
            streaming = True,
        )
        has_level   = "level"   in dataset.column_names
        has_subject = "subject" in dataset.column_names
        solution_key = "solution"
        answer_key   = "answer" if "answer" in dataset.column_names \
                       else None
    except Exception:
        dataset = load_dataset(
            "hendrycks/competition_math", split="test"
        )
        has_level    = True
        has_subject  = True
        solution_key = "solution"
        answer_key   = None

    print("Streaming MATH-500 — collecting problems ...\n")

    # Build items with extracted answers
    items = []
    for item in dataset:
        solution = item.get(solution_key, "")
        answer   = (item.get(answer_key) if answer_key
                    else extract_boxed_answer(solution))
        if answer is None:
            continue

        items.append({
            "problem" : item.get("problem", item.get("question", "")),
            "solution": solution,
            "answer"  : answer,
            "subject" : item.get("subject", "unknown") if has_subject
                        else "unknown",
            "level"   : item.get("level", "unknown") if has_level
                        else "unknown",
        })

        if len(items) >= 500:
            break

    print(f"Problems with valid answers: {len(items)}\n")

    # Stratified sample across subjects
    random.seed(CONFIG["seed"])
    by_subject = defaultdict(list)
    for i, item in enumerate(items):
        subject = item.get("subject", "unknown")
        by_subject[subject].append(i)

    n_subjects  = len(by_subject)
    per_subject = max(1, CONFIG["n_questions"] // n_subjects)

    sampled_indices = []
    for subj, indices in by_subject.items():
        sampled_indices.extend(
            random.sample(indices, min(per_subject, len(indices)))
        )
    sampled = [items[i] for i in sampled_indices]

    # Top up to exactly n_questions if needed
    sampled_idx_set = set(sampled_indices)
    remaining       = [items[i] for i in range(len(items))
                       if i not in sampled_idx_set]
    random.shuffle(remaining)
    sampled = sampled[:CONFIG["n_questions"]]
    if len(sampled) < CONFIG["n_questions"]:
        sampled += remaining[:CONFIG["n_questions"] - len(sampled)]

    questions = [{"id": i, **q} for i, q in enumerate(sampled)]

    with open(CONFIG["output_file"], "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} problems to {CONFIG['output_file']}")

    # Distribution summary
    subjects = defaultdict(int)
    for q in questions:
        subjects[q["subject"]] += 1
    print("\nSubject distribution:")
    for subj, count in sorted(subjects.items()):
        print(f"  {subj:<30} {count}")

    print("\nPreview (first 3):")
    for q in questions[:3]:
        print(f"  Q{q['id']} [{q['subject']}]: "
              f"{q['problem'][:60]}...")
        print(f"        Answer: {q['answer']}")

if __name__ == "__main__":
    main()