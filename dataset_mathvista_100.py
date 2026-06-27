# scripts/dataset_mathvista_100.py
#
# Downloads MathVista from HuggingFace, samples 100 problems
# with visual input, stratified across task types.
#
# Output: data/mathvista_100.json
# Format: list of {id, question, image_path, answer, task_type}

import os
import json
import random
from collections import defaultdict
from pathlib import Path

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CONFIG = {
    "n_questions"  : 100,
    "seed"         : 42,
    "split"        : "testmini",    # 1000-item mini test set
    "images_dir"   : "data/mathvista/images/",
    "output_file"  : "data/mathvista_100.json",
}

def main():
    try:
        from datasets import load_dataset
        from PIL import Image
        import io
    except ImportError:
        print("Please install: pip install datasets Pillow")
        return

    os.makedirs(CONFIG["images_dir"], exist_ok=True)

    print("Loading MathVista from HuggingFace ...")
    dataset = load_dataset(
        "AI4Math/MathVista",
        split     = CONFIG["split"],
        streaming = True,
    )
    print("Streaming MathVista — collecting problems ...\n")

    # Filter to problems with images and valid answers
    items = []
    for item in dataset:
        answer    = item.get("answer", "")
        if not answer:
            continue

        # Task type for stratification
        task_type = item.get("task", item.get("metadata", {})
                    .get("task", "general")
                    if isinstance(item.get("metadata"), dict)
                    else "general")

        items.append({
            "question"  : item.get("question", "").strip(),
            "answer"    : str(answer).strip(),
            "task_type" : task_type,
            "image_idx" : item.get("pid", len(items)),
            "image"     : item.get("image", None),
        })
        if len(items) >= 500:   # safety cap
            break

    print(f"Problems with valid answers: {len(items)}")

    # Stratified sample across task types
    random.seed(CONFIG["seed"])
    by_task = defaultdict(list)
    for item in items:
        by_task[item["task_type"]].append(item)

    n_tasks     = len(by_task)
    per_task    = max(1, CONFIG["n_questions"] // n_tasks)
    sampled     = []

    for task, task_items in by_task.items():
        sampled.extend(
            random.sample(task_items, min(per_task, len(task_items)))
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
        img_filename = f"mathvista_{idx:03d}.png"
        img_path     = os.path.join(CONFIG["images_dir"], img_filename)

        # Save image if available
        if item["image"] is not None:
            # image saving
            try:
                img_obj = item["image"]
                if img_obj is None:
                    img_path = None
                elif hasattr(img_obj, "save"):
                    # PIL Image object
                    img_obj.save(img_path)
                elif isinstance(img_obj, bytes):
                    # Raw bytes
                    from PIL import Image
                    import io
                    Image.open(io.BytesIO(img_obj)).save(img_path)
                elif isinstance(img_obj, str):
                    # Base64 string or file path
                    import base64
                    from PIL import Image
                    import io
                    try:
                        img_data = base64.b64decode(img_obj)
                        Image.open(io.BytesIO(img_data)).save(img_path)
                    except Exception:
                        # Treat as file path
                        from shutil import copyfile
                        if os.path.exists(img_obj):
                            copyfile(img_obj, img_path)
                        else:
                            img_path = None
                else:
                    img_path = None
            except Exception as e:
                print(f"  Warning: could not save image {idx}: {e}")
                img_path = None

        questions.append({
            "id"        : idx,
            "question"  : item["question"],
            "answer"    : item["answer"],
            "task_type" : item["task_type"],
            "image_path": img_path,
        })

    with open(CONFIG["output_file"], "w") as f:
        # Cannot serialize PIL images — exclude them
        json.dump(questions, f, indent=2)
    print(f"Saved to {CONFIG['output_file']}")

    # Task distribution
    tasks = defaultdict(int)
    for q in questions:
        tasks[q["task_type"]] += 1
    print("\nTask distribution:")
    for task, count in sorted(tasks.items()):
        print(f"  {task:<30} {count}")

    print("\nPreview (first 3):")
    for q in questions[:3]:
        print(f"  Q{q['id']} [{q['task_type']}]: "
              f"{q['question'][:60]}...")
        print(f"        Answer: {q['answer']}")


if __name__ == "__main__":
    main()