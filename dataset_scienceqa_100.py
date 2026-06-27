# scripts/dataset_scienceqa_100.py
#
# Downloads ScienceQA from HuggingFace, samples 100 questions
# with images, stratified across subjects and grade levels.
#
# Output: data/scienceqa_100.json
# Format: list of {id, question, choices, answer_idx, answer_text,
#                  subject, topic, image_path}

import os
import json
import random
from collections import defaultdict

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CONFIG = {
    "n_questions" : 100,
    "seed"        : 42,
    "split"       : "validation",
    "images_dir"  : "data/scienceqa/images/",
    "output_file" : "data/scienceqa_100.json",
}

CHOICE_LABELS = ["A", "B", "C", "D", "E"]

def main():
    try:
        from datasets import load_dataset
        from PIL import Image
        import io
    except ImportError:
        print("Please install: pip install datasets Pillow")
        return

    os.makedirs(CONFIG["images_dir"], exist_ok=True)

    print("Loading ScienceQA from HuggingFace ...")
    dataset = load_dataset(
        "derek-thomas/ScienceQA",
        split     = CONFIG["split"],
        streaming = True,
    )
    print("Streaming ScienceQA — collecting questions ...\n")

    # Filter to questions that have images
    items = []
    for item in dataset:
        image   = item.get("image", None)
        choices = item.get("choices", [])
        ans_idx = item.get("answer", None)

        if image is None:
            continue
        if ans_idx is None or ans_idx >= len(choices):
            continue

        items.append({
            "question"    : item.get("question", "").strip(),
            "choices"     : choices,
            "answer_idx"  : int(ans_idx),
            "answer_text" : choices[ans_idx],
            "answer_label": CHOICE_LABELS[ans_idx] if ans_idx < 5 else str(ans_idx),
            "subject"     : item.get("subject", "general"),
            "topic"       : item.get("topic", ""),
            "image"       : image,
        })
        if len(items) >= 500:        # collect 5x buffer
            break
    print(f"Questions with images: {len(items)}")

    # Stratified sample across subjects
    random.seed(CONFIG["seed"])
    by_subject = defaultdict(list)
    for item in items:
        by_subject[item["subject"]].append(item)

    n_subjects  = len(by_subject)
    per_subject = max(1, CONFIG["n_questions"] // n_subjects)
    sampled     = []

    for subj, subj_items in by_subject.items():
        sampled.extend(
            random.sample(subj_items, min(per_subject, len(subj_items)))
        )

    random.shuffle(sampled)
    sampled = sampled[:CONFIG["n_questions"]]

    if len(sampled) < CONFIG["n_questions"]:
        remaining = [i for i in items if i not in sampled]
        random.shuffle(remaining)
        sampled += remaining[:CONFIG["n_questions"] - len(sampled)]

    # Save images and build question list
    questions = []
    print(f"\nSaving {len(sampled)} images ...")
    for idx, item in enumerate(sampled):
        img_filename = f"scienceqa_{idx:03d}.png"
        img_path     = os.path.join(CONFIG["images_dir"], img_filename)

        try:
            if hasattr(item["image"], "save"):
                item["image"].save(img_path)
            else:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(item["image"]))
                img.save(img_path)
        except Exception as e:
            print(f"  Warning: could not save image {idx}: {e}")
            img_path = None

        questions.append({
            "id"          : idx,
            "question"    : item["question"],
            "choices"     : item["choices"],
            "answer_idx"  : item["answer_idx"],
            "answer_text" : item["answer_text"],
            "answer_label": item["answer_label"],
            "subject"     : item["subject"],
            "topic"       : item["topic"],
            "image_path"  : img_path,
        })

    with open(CONFIG["output_file"], "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved to {CONFIG['output_file']}")

    # Distribution
    subjects = defaultdict(int)
    for q in questions:
        subjects[q["subject"]] += 1
    print("\nSubject distribution:")
    for subj, count in sorted(subjects.items()):
        print(f"  {subj:<30} {count}")

    print("\nPreview (first 3):")
    for q in questions[:3]:
        print(f"  Q{q['id']} [{q['subject']}]:")
        print(f"    {q['question'][:65]}...")
        for label, choice in zip(CHOICE_LABELS, q["choices"]):
            print(f"    {label}) {choice[:50]}")
        print(f"    → Correct: {q['answer_label']}) {q['answer_text']}")

if __name__ == "__main__":
    main()