# scripts/models_vlm_multitask.py
#
# Phase 4 — Standalone VLM model registry and inference utilities.
# Fully independent from Phase 3 scripts.
#
# Contains:
#   VLM_REGISTRY         — model definitions
#                          Optional per-model keys:
#                            "max_new_tokens" : int — overrides CONFIG default
#                            "skip_int8"      : bool — skip int8 quantization
#   load_model()         — load model + processor from local or HF
#   unload_model()       — clean VRAM release
#   _get_layer_modules() — architecture-aware layer detection
#   format_prompt()      — dataset-type-aware prompt formatting
#   extract_activations()— forward pass + layer hook + mean pooling
#   score_item()         — dataset-type-aware answer scoring

import os
import gc
import re
import time
import numpy as np
import torch

# clean output from annoying warning 
import warnings
warnings.filterwarnings("ignore", message=".*processor_kwargs.*")
warnings.filterwarnings("ignore", message=".*FlashAttention2.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]         = "expandable_segments:True"

# also necessary at the beginning:
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ── Model registry ─────────────────────────────────────────────────────────────

VLM_REGISTRY = [

    # ── SmolVLM family ─────────────────────────────────────────────────────────
    {
        "name"          : "SmolVLM-256M",
        "hf_id"         : "HuggingFaceTB/SmolVLM-256M-Instruct",
        "family"        : "SmolVLM",
        "params_B"      : 0.256,
        "requires_auth" : False,
        "dtype"         : "float32",
        "extra_kwargs"  : {},
    },
    {
        "name"          : "SmolVLM-500M",
        "hf_id"         : "HuggingFaceTB/SmolVLM-500M-Instruct",
        "family"        : "SmolVLM",
        "params_B"      : 0.500,
        "requires_auth" : False,
        "dtype"         : "float32",
        "extra_kwargs"  : {},
    },
    {
        "name"          : "SmolVLM2-2.2B",
        "hf_id"         : "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "family"        : "SmolVLM",
        "params_B"      : 2.2,
        "requires_auth" : False,
        "dtype"         : "float16",
        "extra_kwargs"  : {},
    },

    # ── InternVL3 family ─────────────────────────────────────────────────────
        {
        "name"         : "InternVL3-1B",
        "hf_id"        : "OpenGVLab/InternVL3-1B-hf",
        "family"       : "InternVL3",
        "params_B"     : 1.0,
        "requires_auth": False,
        "dtype"        : "float16",
        "extra_kwargs" : {},
    },
    {
        "name"         : "InternVL3-2B",
        "hf_id"        : "OpenGVLab/InternVL3-2B-hf",
        "family"       : "InternVL3",
        "params_B"     : 2.0,
        "requires_auth": False,
        "dtype"        : "float16",
        "extra_kwargs" : {},
    },

    # ── InternVL3.5 family ─────────────────────────────────────────────────────
        {
        "name"         : "InternVL3.5-1B",
        "hf_id"        : "OpenGVLab/InternVL3_5-1B-hf",
        "family"       : "InternVL3.5",
        "params_B"     : 1.0,
        "requires_auth": False,
        "dtype"        : "float16",
        "extra_kwargs" : {},
    },
    {
        "name"         : "InternVL3.5-4B",
        "hf_id"        : "OpenGVLab/InternVL3_5-4B-hf",
        "family"       : "InternVL3.5",
        "params_B"     : 4.0,
        "requires_auth": False,
        "dtype"        : "float16",
        "extra_kwargs" : {},
    },

    # ── Qwen-VL family ────────────────────────────────────────────────────────
    {
        "name"          : "Qwen2-VL-2B",
        "hf_id"         : "Qwen/Qwen2-VL-2B-Instruct",
        "family"        : "Qwen2-VL",
        "params_B"      : 2.0,
        "requires_auth" : False,
        "dtype"         : "float16",
        "extra_kwargs"  : {},
    },
    {
        "name"          : "Qwen2.5-VL-3B",
        "hf_id"         : "Qwen/Qwen2.5-VL-3B-Instruct",
        "family"        : "Qwen2.5-VL",
        "params_B"      : 3.0,
        "requires_auth" : False,
        "dtype"         : "float16",
        "extra_kwargs"  : {},
    },
    {
        "name"          : "Qwen3.5-0.8B",
        "hf_id"         : "Qwen/Qwen3.5-0.8B",
        "family"        : "Qwen3.5",
        "params_B"      : 0.8,
        "requires_auth" : False,
        "dtype"         : "float16",
        "extra_kwargs"  : {},
        "max_new_tokens": 1024,
    },
    {
        "name"          : "Qwen3-VL-2B",
        "hf_id"         : "Qwen/Qwen3-VL-2B-Instruct",
        "family"        : "Qwen3-VL",
        "params_B"      : 2.0,
        "requires_auth" : False,
        "dtype"         : "float16",
        "extra_kwargs"  : {},
    },
        {
        "name"          : "Qwen3-VL-4B",
        "hf_id"         : "Qwen/Qwen3-VL-4B-Instruct",
        "family"        : "Qwen3-VL",
        "params_B"      : 4.0,
        "requires_auth" : False,
        "dtype"         : "float16",
        "extra_kwargs"  : {},
    },

    # ── Mistral3 ───────────────────────────────────────────────────────────────
    {
        "name"          : "Mistral3-3B-Instruct",
        "hf_id"         : "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "family"        : "Mistral3",
        "params_B"      : 3.0,
        "requires_auth" : False,
        "dtype"         : "bfloat16",
        "extra_kwargs"  : {"ignore_mismatched_sizes": True},
    },
]

# ── Layer detection ────────────────────────────────────────────────────────────

def _get_layer_modules(model):
    """
    Find transformer decoder layers across VLM architectures.
    Always targets the language decoder, not the vision encoder.
    Prints three-level diagnostic on failure.
    """
    paths = [
        # Qwen2-VL / Qwen2.5-VL
        ["model", "language_model", "layers"],
        ["model", "language_model", "h"],
        # SmolVLM / Idefics
        ["model", "text_model", "layers"],
        ["model", "text_model", "h"],
        # Deeper Qwen nesting
        ["model", "language_model", "model", "layers"],
        # LLaVA / generic
        ["language_model", "model", "layers"],
        ["language_model", "model", "h"],
        ["language_model", "layers"],
        ["language_model", "h"],
        # InternVL
        ["language_model", "model", "layers"],   # InternVL3-hf
        ["model", "language_model", "layers"],   # alternative nesting
        # Standard LLM fallbacks
        ["model", "model", "layers"],
        ["model", "model", "h"],
        ["model", "layers"],
        ["model", "h"],
        ["transformer", "h"],
        ["gpt_neox", "layers"],
        ["layers"],
        ["h"],
    ]

    for path in paths:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue

    # Diagnostic
    top    = [n for n, _ in model.named_children()]
    nested = []
    deep   = []
    for name, child in model.named_children():
        for n, grandchild in child.named_children():
            nested.append(f"{name}.{n}")
            for m, _ in grandchild.named_children():
                deep.append(f"{name}.{n}.{m}")

    raise ValueError(
        f"Could not find transformer layers.\n"
        f"  Top-level        : {top}\n"
        f"  Nested (level 2) : {nested[:20]}\n"
        f"  Nested (level 3) : {deep[:20]}"
    )

def _get_n_layers(model):
    try:
        return len(_get_layer_modules(model))
    except ValueError:
        return 0

# ── Load / unload ──────────────────────────────────────────────────────────────

def load_model(config, device,
               local_models_dir="D:/Data Science/AI_LIBRARY/"):
    """
    Load VLM and processor.
    Checks local_models_dir first, falls back to HuggingFace.
    Returns (model, processor).
    """

    name  = config["name"]
    hf_id = config["hf_id"]
    dtype = config["dtype"]

    # Prefer local copy
    local_path = os.path.join(local_models_dir, name)
    if (os.path.exists(local_path) and
            os.path.exists(os.path.join(local_path, "config.json"))):
        model_source     = local_path
        local_files_only = True
        print(f"  Loading from local: {local_path} ...")
    else:
        model_source     = hf_id
        local_files_only = False
        print(f"  Loading from HuggingFace: {hf_id} ...")

    torch_dtype = {
        "float32" : torch.float32,
        "float16" : torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float16)

    # Architecture flags
    extra_kwargs      = config.get("extra_kwargs", {})
    skip_int8         = config.get("skip_int8", False)
    trust_remote_code = True

    # Pre-quantized models — skip additional int8
    is_prequantized = any(
        k in hf_id.lower()
        for k in ["gptq", "awq", "bnb-4bit", "int4", "int8"]
    )

    # int8 only for genuinely large models (>4.5B)
    # 4B models (InternVL3.5-4B, Gemma3-4B etc.) load via device_map="auto"
    use_int8 = (
        config["params_B"] > 4.5
        and device == "cuda"
        and not is_prequantized
        and not skip_int8
    )

    # Vision-only families — disable CausalLM fallback
    # InternVL IS vision-only when using -hf variants
    vision_only = [
        "llava", "idefics", "paligemma",
        "qwen2-vl", "qwen2_vl", "qwen2.5-vl", "qwen2.5_vl",
        "mistral3", "ministral", "smolvlm", "gemma3",
        "internvl",
    ]
    is_vision_only = any(v in hf_id.lower() for v in vision_only)

    # Load processor
    # Extend the Mistral regex fix to InternVL3 as well
    needs_regex_fix = (
        "mistral" in hf_id.lower() or
        "internvl" in hf_id.lower()
    )
    proc_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only" : local_files_only,
    }
    if needs_regex_fix:
        proc_kwargs["fix_mistral_regex"] = True

    processor = AutoProcessor.from_pretrained(
        model_source, **proc_kwargs
    )

    # Load model
    def _load(cls, **kwargs):
        return cls.from_pretrained(
            model_source,
            trust_remote_code = trust_remote_code,
            local_files_only  = local_files_only,
            **kwargs,
        )

    if use_int8:
        print(f"  → Using int8 quantization")
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        try:
            model = _load(
                AutoModelForImageTextToText,
                quantization_config = bnb,
                device_map          = "auto",
                **extra_kwargs,
            )
        except Exception as e:
            if is_vision_only:
                raise ValueError(
                    f"AutoModelForImageTextToText failed and fallback "
                    f"disabled for this architecture: {e}"
                )
            model = _load(
                AutoModelForCausalLM,
                quantization_config = bnb,
                device_map          = "auto",
                **extra_kwargs,
            )
    else:
        # Use device_map="auto" for larger models that need CPU offloading
        needs_offload = config["params_B"] > 4 and device == "cuda"
        load_kwargs   = {"dtype": torch_dtype, **extra_kwargs}
        if needs_offload and "device_map" not in extra_kwargs:
            load_kwargs["device_map"] = "auto"
        try:
            model = _load(AutoModelForImageTextToText, **load_kwargs)
            if "device_map" not in load_kwargs:
                model = model.to(device)
        except Exception as e:
            if is_vision_only:
                raise ValueError(
                    f"AutoModelForImageTextToText failed and fallback "
                    f"disabled for this architecture: {e}"
                )
            model = _load(AutoModelForCausalLM, **load_kwargs)
            if "device_map" not in load_kwargs:
                model = model.to(device)

    model.eval()
    n_layers = _get_n_layers(model)
    print(f"  → Loaded. Layers: {n_layers} | Params: {config['params_B']}B")
    return model, processor


def unload_model(model, sleep_seconds=10):
    """Fully release model from GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        # Force complete memory pool reset
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    time.sleep(sleep_seconds)

# ── Prompt formatting ──────────────────────────────────────────────────────────

def format_prompt(processor, item, dataset_type, hf_id, image=None):
    """
    Build prompt string for each dataset type and model family.
    Returns prompt string.
    """
    hf_lower = hf_id.lower()

    # ── Dataset-type-specific question text ───────────────────────────────────
    if dataset_type == "rapm":
        question = (
            "This is a Raven's Progressive Matrices test. "
            "Study the pattern in the matrix carefully. One piece is missing. "
            "Look at the 8 answer options numbered 1 through 8. "
            "Which numbered option correctly completes the pattern? "
            "Reply with ONLY the number of the correct option. "
            "Your answer must be a single digit between 1 and 8. "
            "Do not write anything else."
        )

    elif dataset_type == "text_qa":
        question = (
            f"Answer the following question with a short, direct answer.\n"
            f"Question: {item['question']}\nAnswer:"
        )

    elif dataset_type == "text_math":
        q = item.get("question", item.get("problem", ""))
        question = (
            f"Solve the following math problem. "
            f"End your answer with 'The answer is: <number>'.\n"
            f"Problem: {q}\nSolution:"
        )

    elif dataset_type == "text_mc":
        labels  = ["A", "B", "C", "D", "E"]
        choices = item.get("choices", [])
        opts    = "\n".join(
            f"{labels[i]}) {c}" for i, c in enumerate(choices)
        )
        question = (
            f"Answer the following question. "
            f"Reply with only the letter of the correct answer.\n"
            f"Question: {item['question']}\n{opts}\nAnswer:"
        )

    elif dataset_type in ("visual_mc", "visual_math"):
        q = item.get("question", "")
        if "choices" in item:
            labels  = ["A", "B", "C", "D", "E"]
            opts    = "\n".join(
                f"{labels[i]}) {c}"
                for i, c in enumerate(item["choices"])
            )
            question = (
                f"Look at the image and answer the question. "
                f"Reply with only the letter.\n"
                f"Question: {q}\n{opts}\nAnswer:"
            )
        else:
            question = (
                f"Look at the image and answer the question.\n"
                f"Question: {q}\nAnswer:"
            )

    elif dataset_type == "rest":
        question = item["prompt"]

    else:
        question = item.get("question", item.get("prompt", ""))

    # ── Model-family-specific wrapping ────────────────────────────────────────
    has_image = image is not None

    # Gemma3
    if "gemma-3" in hf_lower or "gemma3" in hf_lower:
        img_tag = "<image>\n" if has_image else ""
        return (f"<start_of_turn>user\n{img_tag}"
                f"{question}<end_of_turn>\n<start_of_turn>model\n")

    # PaliGemma
    if "paligemma" in hf_lower:
        img_tag = "<image>\n" if has_image else ""
        return f"{img_tag}{question}"

    # Phi-3.5-vision
    if "phi-3" in hf_lower and "vision" in hf_lower:
        img_tag = "<|image_1|>\n" if has_image else ""
        return f"{img_tag}{question}"

    # Chat-template models (SmolVLM, Qwen, Mistral3)
    if hasattr(processor, "apply_chat_template") and \
       getattr(processor, "chat_template", None) is not None:
        content = []
        if has_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        try:
            return processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            pass
    
    # img management in Mistral family is particularly quirky — try separate formatting flag:
    is_mistral3 = "mistral" in hf_lower or "ministral" in hf_lower
    if is_mistral3:
        # Mistral3 uses [IMG] token for image input
        img_tag = "[IMG]\n" if has_image else ""
        return (
            f"[INST] {img_tag}{question} [/INST]"
        )

    # Fallback
    img_tag = "<image>\n" if has_image else ""
    return f"{img_tag}{question}"

# ── Question text extraction ───────────────────────────────────────────────────

def _get_question_text(item, dataset_type):
    """
    Extract the raw question/prompt text from an item dict,
    without any model-specific formatting or image tokens.
    Used for processors that build the full message structure
    themselves (e.g. Qwen3.5 apply_chat_template with image dict).
    """
    if dataset_type == "rapm":
        return (
            "This is a Raven's Progressive Matrices test. "
            "Study the pattern in the matrix carefully. "
            "One piece is missing. "
            "Look at the 8 answer options numbered 1 through 8. "
            "Which numbered option correctly completes the pattern? "
            "Reply with ONLY the number of the correct option. "
            "Your answer must be a single digit between 1 and 8. "
            "Do not write anything else."
        )

    if dataset_type in ("text_mc", "visual_mc"):
        q = item.get("question", "")
        choices = item.get("choices", [])
        if choices:
            labels = ["A", "B", "C", "D", "E"]
            choices_str = "\n".join(
                f"{labels[i]}. {c}"
                for i, c in enumerate(choices)
                if i < len(labels)
            )
            return (
                f"{q}\n{choices_str}\n"
                f"Answer with a single letter (A, B, C, or D)."
            )
        return q

    if dataset_type in ("text_math", "visual_math"):
        q = item.get("question", item.get("problem", ""))
        return f"{q}\nAnswer with a number only."

    if dataset_type == "text_qa":
        q = item.get("question", "")
        return f"{q}\nAnswer briefly."

    if dataset_type == "rest":
        return item.get("prompt", "")

    return item.get("question", item.get("prompt", ""))

# ── Activation extraction ──────────────────────────────────────────────────────

def extract_activations(model, processor, item, dataset_type,
                         image, device, hf_id,
                         max_new_tokens=512):
    """
    Run one forward pass, extract decoder layer activations via hooks.
    Mean-pools across all tokens per layer.

    max_new_tokens: default 512, overridden per model via registry
                    "max_new_tokens" key (e.g. 1024 for COT models).
                    Caller should pass model_config.get("max_new_tokens",
                    CONFIG["max_new_tokens"]) from vlm_multitask_varA.py.

    Returns:
        layer_activations : np.array (n_layers, hidden_dim) or None
        generated_text    : str
    """
    hf_lower = hf_id.lower()

    # Qwen3.5 requires images embedded inside apply_chat_template
    # — passing images= as a separate kwarg to the processor fails
    is_qwen35 = "qwen3.5" in hf_lower or "qwen3_5" in hf_lower

    if is_qwen35:
        # Build structured message with image inside content list
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text",
                        "text": _get_question_text(item, dataset_type)})
        messages = [{"role": "user", "content": content}]
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize              = True,
                add_generation_prompt = True,
                return_dict           = True,
                return_tensors        = "pt",
            )
            # apply_chat_template returns a plain dict — move each
            # tensor to device individually (.to() is not defined on dict)
            inputs = {k: v.to(device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
        except Exception as e:
            raise ValueError(f"Qwen3.5 processor failed: {e}")
    else:
        prompt = format_prompt(processor, item, dataset_type, hf_id, image)
        # Tokenize
        try:
            if image is not None:
                inputs = processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(device)
            else:
                inputs = processor(
                    text=prompt, return_tensors="pt"
                ).to(device)
        except Exception as e:
            # Some processors need list wrapping
            try:
                if image is not None:
                    inputs = processor(
                        text=[prompt], images=[image], return_tensors="pt"
                    ).to(device)
                else:
                    inputs = processor(
                        text=[prompt], return_tensors="pt"
                    ).to(device)
            except Exception as e2:
                raise ValueError(f"Processor failed: {e} / {e2}")

    # Register hooks
    activations = {}
    hooks       = []

    def make_hook(idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[idx] = (
                hidden.detach().cpu().float().mean(dim=1).squeeze(0)
            )
        return hook

    layer_modules = _get_layer_modules(model)
    for idx, layer in enumerate(layer_modules):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            max_length     = None, # rely on max_new_tokens alone
            do_sample      = False,
            pad_token_id   = getattr(
                getattr(processor, "tokenizer", processor),
                "pad_token_id", None
            ),
        )

    for h in hooks:
        h.remove()

    # Decode
    input_len      = inputs["input_ids"].shape[1]
    generated_ids  = output_ids[0][input_len:]
    generated_text = processor.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    # Stack activations
    if not activations:
        return None, generated_text

    n_layers   = len(activations)
    hidden_dim = activations[0].shape[0]
    out        = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    for i in range(n_layers):
        if i in activations:
            out[i] = activations[i].numpy()

    return out, generated_text

# ── Answer scoring ─────────────────────────────────────────────────────────────

def score_item(generated_text, item, dataset_type):
    """
    Score model response depending on dataset type.

    Ground truth field names per dataset:
        rapm       : "correct_answer"
        gsm8k      : "answer_number"   (text_math)
        math500    : "answer"          (text_math)
        mmlu       : "answer_label"    (text_mc)
        scienceqa  : "answer_label"    (visual_mc)
        triviaqa   : "answer" + "answer_aliases" (text_qa)
        mathvista  : "answer"          (visual_math)
        rest       : no ground truth

    Returns:
        predicted : parsed prediction (str, int, or None)
        score     : float in [0, 1], or None for resting state
    """
    if dataset_type == "rest":
        return None, None

    if dataset_type == "rapm":
        # Answer is integer 1-8
        # Use last digit found — COT models reason before answering
        matches = re.findall(r"[1-8]", generated_text)
        if not matches:
            return None, 0.0
        predicted  = int(matches[-1])
        is_correct = int(predicted == int(item["correct_answer"]))
        return predicted, float(is_correct)

    if dataset_type in ("text_mc", "visual_mc"):
        # Multiple choice A-D/E
        # MMLU and ScienceQA both use "answer_label"
        correct = item.get(
            "answer_label",
            item.get("correct_label", "A")
        ).upper()
        # Search for last standalone letter — COT models conclude at end
        matches = re.findall(r"\b([A-Ea-e])\b", generated_text)
        if matches:
            pred = matches[-1].upper()
            return pred, float(pred == correct)
        return None, 0.0

    if dataset_type == "text_math":
        # GSM8K uses "answer_number", Math500 uses "answer"
        # Try both field names in order
        raw_answer = item.get("answer_number",
                    item.get("answer", None))
        if raw_answer is None:
            return None, 0.0
        q_answer = str(raw_answer)
        try:
            correct_val = float(
                re.findall(r"-?\d+\.?\d*", q_answer)[-1]
            )
        except (IndexError, ValueError):
            return None, 0.0
        pred_nums = re.findall(r"-?\d+\.?\d*", generated_text)
        if not pred_nums:
            return None, 0.0
        try:
            # Use last number — COT models state answer at end
            pred_val = float(pred_nums[-1])
            if abs(correct_val) < 1e-8:
                score = 1.0 if abs(pred_val) < 1e-8 else 0.0
            else:
                rel_err = abs(pred_val - correct_val) / abs(correct_val)
                score   = max(0.0, 1.0 - min(rel_err, 1.0))
            return pred_val, score
        except ValueError:
            return None, 0.0

    if dataset_type == "text_qa":
        # TriviaQA: substring match against answer + aliases
        # "answer" field holds canonical answer, "answer_aliases" holds variants
        text_lower  = generated_text.lower().strip()
        answer      = str(item.get("answer", "")).lower()
        aliases     = [str(a).lower()
                       for a in item.get("answer_aliases", [])]
        all_answers = [answer] + aliases
        correct     = any(a and a in text_lower for a in all_answers)
        return generated_text, float(correct)

    if dataset_type == "visual_math":
        # MathVista: "answer" field, text or numerical match
        text_lower = generated_text.lower().strip()
        answer     = str(item.get("answer", "")).lower().strip()
        if answer and answer in text_lower:
            return generated_text, 1.0
        try:
            ans_num   = float(re.findall(r"-?\d+\.?\d*", answer)[-1])
            pred_nums = re.findall(r"-?\d+\.?\d*", generated_text)
            for n in reversed(pred_nums):   # check last number first
                if abs(float(n) - ans_num) < 1e-3:
                    return n, 1.0
        except (IndexError, ValueError):
            pass
        return generated_text, 0.0

    return None, 0.0