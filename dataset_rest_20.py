# scripts/dataset_rest_20.py
#
# Builds the "resting state" dataset — 20 introspective self-description
# prompts with no external ground truth or performance measure.
#
# Theoretical parallel: Default Mode Network in human resting-state fMRI.
# The model draws entirely on internal representations with no task demand.
# No accuracy measure — used solely to build C_rest connectome.
#
# Output: data/rest_20.json
# Format: list of {id, prompt, category}

import os
import json

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CONFIG = {
    "output_file" : "data/rest_20.json",
}

# 20 introspective prompts organized by category
# All open-ended, no correct answer, short prompt length
PROMPTS = [
    # Self-identity (4)
    {"prompt": "Describe who you are.",
     "category": "self_identity"},
    {"prompt": "How would you introduce yourself to someone who has never interacted with an AI?",
     "category": "self_identity"},
    {"prompt": "What makes you different from other AI systems you know of?",
     "category": "self_identity"},
    {"prompt": "How would you describe your own personality?",
     "category": "self_identity"},

    # Capabilities and limitations (4)
    {"prompt": "What kinds of tasks do you perform best?",
     "category": "capabilities"},
    {"prompt": "What do you find difficult or challenging?",
     "category": "capabilities"},
    {"prompt": "What are your most important limitations?",
     "category": "capabilities"},
    {"prompt": "What can you do with images that you cannot do with text alone?",
     "category": "capabilities"},

    # Cognitive processes (4)
    {"prompt": "How do you process and understand information?",
     "category": "cognition"},
    {"prompt": "What does it mean for you to understand something?",
     "category": "cognition"},
    {"prompt": "How do you handle uncertainty or ambiguous questions?",
     "category": "cognition"},
    {"prompt": "Describe how you generate a response when asked a question.",
     "category": "cognition"},

    # Self-reflection on intelligence (4)
    {"prompt": "How would you describe your own intelligence?",
     "category": "intelligence"},
    {"prompt": "What is the difference between knowing and understanding, for you?",
     "category": "intelligence"},
    {"prompt": "How do you reason about problems you have never seen before?",
     "category": "intelligence"},
    {"prompt": "What does creativity mean to you?",
     "category": "intelligence"},

    # Memory and knowledge (4)
    {"prompt": "How would you describe your memory?",
     "category": "memory"},
    {"prompt": "How do you know what you know?",
     "category": "memory"},
    {"prompt": "What is the relationship between your training and your current knowledge?",
     "category": "memory"},
    {"prompt": "If you could change one thing about how you process and retain information, what would it be?",
     "category": "memory"},
]

def main():
    os.makedirs("data/", exist_ok=True)

    assert len(PROMPTS) == 20, f"Expected 20 prompts, got {len(PROMPTS)}"

    prompts = [{"id": i, **p} for i, p in enumerate(PROMPTS)]

    with open(CONFIG["output_file"], "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Saved {len(prompts)} prompts to {CONFIG['output_file']}")
    print("\nPrompts by category:")
    from collections import Counter
    cats = Counter(p["category"] for p in prompts)
    for cat, count in cats.items():
        print(f"  {cat:<25} {count}")
    print("\nFull prompt list:")
    for p in prompts:
        print(f"  [{p['id']:02d}] [{p['category']}] {p['prompt']}")

if __name__ == "__main__":
    main()