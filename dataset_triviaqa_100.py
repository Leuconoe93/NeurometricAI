# scripts/dataset_triviaqa_100.py

import os
import json
import random

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CONFIG = {
    "n_questions"  : 100,
    "seed"         : 42,
    "output_file"  : "data/triviaqa_100.json",
    "buffer_factor": 5,    # collect n*5 candidates then sample
}

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install: pip install datasets")
        return

    os.makedirs("data/", exist_ok=True)

    print("Loading TriviaQA (unfiltered, streaming) ...")
    dataset = load_dataset(
        "trivia_qa", "unfiltered",
        split             = "validation",
        streaming         = True,
        trust_remote_code = True,
    )

    # Collect candidates via streaming — no full download
    target  = CONFIG["n_questions"] * CONFIG["buffer_factor"]
    filtered = []

    for item in dataset:
        answer     = item["answer"]
        value      = answer.get("value", "").strip()
        aliases    = answer.get("aliases", [])
        normalized = answer.get("normalized_aliases", [])

        # Keep only short, unambiguous answers
        if not value or len(value.split()) > 5:
            continue

        filtered.append({
            "question"       : item["question"].strip(),
            "answer"         : value,
            "answer_aliases" : list(set(aliases + normalized)),
        })

        if len(filtered) >= target:
            break

    print(f"Candidates collected: {len(filtered)}")

    # Reproducible random sample
    random.seed(CONFIG["seed"])
    sampled   = random.sample(
        filtered,
        min(CONFIG["n_questions"], len(filtered))
    )
    sampled   = sorted(sampled, key=lambda x: x["question"])
    questions = [{"id": i, **q} for i, q in enumerate(sampled)]

    with open(CONFIG["output_file"], "w") as f:
        json.dump(questions, f, indent=2)

    print(f"Saved {len(questions)} questions to {CONFIG['output_file']}")
    print("\nPreview (first 3):")
    for q in questions[:3]:
        print(f"  Q{q['id']}: {q['question'][:70]}...")
        print(f"        Answer: {q['answer']}")

if __name__ == "__main__":
    main()