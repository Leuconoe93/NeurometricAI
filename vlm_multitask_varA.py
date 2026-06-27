# scripts/vlm_multitask_varA.py
#
# Phase 4 — Main multi-task VLM inference loop.
# Variant A: models as subjects.
#
# For each model:
#   1. Load model + processor
#   2. Run RAPM first (behavioral criterion)
#   3. Run each test dataset (connectome sources)
#   4. Unload model
#
# Resume logic: per-model JSON failure report tracks status
# of each (model, dataset) pair. Skips 'success', retries
# 'failed' and 'pending'.
#
# Pause: create file named 'PAUSE.txt' in project root.

import os
import sys
import gc
import json
import time
import datetime
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message=".*Kwargs passed to.*processor_kwargs.*",)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["HF_TOKEN"] = "YOUR_TOKEN_HERE"  # Replace with your Hugging Face token
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]         = "expandable_segments:True"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models_vlm_multitask import (
    VLM_REGISTRY,
    load_model,
    unload_model,
    extract_activations,
    score_item,
)

# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    "device"           : "cuda" if torch.cuda.is_available() else "cpu",
    "local_models_dir" : "D:/Data Science/AI_LIBRARY/",
    "results_dir"      : "results/multitask/",
    "failures_dir"     : "results/failures/",
    "pause_file"       : "PAUSE.txt",
    "max_new_tokens"   : 128,
    "unload_sleep"     : 30,   # seconds to wait after unloading model before next load

    # None = all registry models
    "models_to_run"    : [
        "SmolVLM-256M", "SmolVLM-500M", "SmolVLM2-2.2B",
        "Qwen2-VL-2B", "Qwen2.5-VL-3B", "Qwen3.5-0.8B", 
        "Mistral3-3B-Instruct",
        "InternVL3-1B", "InternVL3-2B",
        "InternVL3.5-1B","InternVL3.5-4B",
    ],

    # Dataset registry — RAPM first (behavioral criterion),
    # then test datasets (connectome sources).
    # criterion=True marks the behavioral measure dataset.
    "datasets": {
        "rapm"      : {
            "path"      : "data/rapm_36.json",
            "type"      : "rapm",
            "criterion" : True,
            "has_image" : True,
        },
        "triviaqa"  : {
            "path"      : "data/triviaqa_100.json",
            "type"      : "text_qa",
            "criterion" : False,
            "has_image" : False,
        },
        "gsm8k"     : {
            "path"      : "data/gsm8k_100.json",
            "type"      : "text_math",
            "criterion" : False,
            "has_image" : False,
        },
        "math500"   : {
            "path"      : "data/math500_100.json",
            "type"      : "text_math",
            "criterion" : False,
            "has_image" : False,
        },
        "mmlu"      : {
            "path"      : "data/mmlu_100.json",
            "type"      : "text_mc",
            "criterion" : False,
            "has_image" : False,
        },
        "mathvista" : {
            "path"      : "data/mathvista_100.json",
            "type"      : "visual_math",
            "criterion" : False,
            "has_image" : True,
        },
        "scienceqa" : {
            "path"      : "data/scienceqa_100.json",
            "type"      : "visual_mc",
            "criterion" : False,
            "has_image" : True,
        },
        "rest"      : {
            "path"      : "data/rest_20.json",
            "type"      : "rest",
            "criterion" : False,
            "has_image" : False,
        },
    },
}

# ── Failure report ─────────────────────────────────────────────────────────────

def failure_path(model_name):
    return os.path.join(
        CONFIG["failures_dir"], f"failure_{model_name}.json"
    )

def load_report(model_name):
    path = failure_path(model_name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "model_name"   : model_name,
        "last_updated" : None,
        "datasets"     : {
            ds: {
                "status"    : "pending",
                "n_items"   : None,
                "error"     : None,
                "timestamp" : None,
            }
            for ds in CONFIG["datasets"]
        },
    }

def save_report(report):
    os.makedirs(CONFIG["failures_dir"], exist_ok=True)
    report["last_updated"] = datetime.datetime.now().isoformat()
    with open(failure_path(report["model_name"]), "w") as f:
        json.dump(report, f, indent=2)

def mark_success(report, ds, n_items):
    report["datasets"][ds].update({
        "status"    : "success",
        "n_items"   : n_items,
        "error"     : None,
        "timestamp" : datetime.datetime.now().isoformat(),
    })
    save_report(report)

def mark_failed(report, ds, error):
    report["datasets"][ds].update({
        "status"    : "failed",
        "error"     : str(error)[:500],
        "timestamp" : datetime.datetime.now().isoformat(),
    })
    save_report(report)

def should_skip(report, ds):
    return report["datasets"][ds]["status"] == "success"

# ── Pause ──────────────────────────────────────────────────────────────────────

def check_pause():
    if os.path.exists(CONFIG["pause_file"]):
        print(f"\n⏸  PAUSED — delete '{CONFIG['pause_file']}' to resume ...")
        while os.path.exists(CONFIG["pause_file"]):
            time.sleep(5)
        print("▶  Resuming ...\n")

# ── Dataset loading ────────────────────────────────────────────────────────────

def load_all_datasets():
    """
    Pre-load all dataset JSON files into memory once.
    Returns dict: {ds_name: (items, ds_config)}
    """
    loaded = {}
    print("Pre-loading datasets ...")
    for ds_name, ds_cfg in CONFIG["datasets"].items():
        path = ds_cfg["path"]
        if not os.path.exists(path):
            print(f"  {ds_name:<12} ✗ file not found: {path}")
            continue
        with open(path) as f:
            items = json.load(f)
        loaded[ds_name] = (items, ds_cfg)
        print(f"  {ds_name:<12} ✓ {len(items)} items  "
              f"[{ds_cfg['type']}]"
              f"{'  ← criterion' if ds_cfg['criterion'] else ''}")
    print()
    return loaded

# ── Run one dataset ────────────────────────────────────────────────────────────

def run_dataset(model, processor, model_config,
                ds_name, items, ds_cfg, model_dir, device):
    """
    Run model on all items of one dataset.
    Saves to model_dir/ds_name/:
        activations.npy  (n_items, n_layers, hidden_dim)
        scores.npy       (n_items,)
        metadata.json

    Returns (n_items, mean_score).
    Raises RuntimeError if no valid activations produced.
    """
    ds_type   = ds_cfg["type"]
    has_image = ds_cfg["has_image"]
    hf_id     = model_config["hf_id"]

    out_dir = os.path.join(model_dir, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    all_acts     = []
    all_scores   = []
    all_predicted = []
    n_failed     = 0
    response_log = []   # universal response log — saved as responses.csv per dataset

    bar = tqdm(items, desc=f"    {ds_name:<12}", unit="item", leave=False)

    for item in bar:
        try:
            # Load image if needed
            image = None
            if has_image:
                img_path = item.get("image_path")
                if img_path and os.path.exists(img_path):
                    image = Image.open(img_path).convert("RGB")

            # Extract activations + generate text
            layer_acts, generated_text = extract_activations(
                model, processor,
                item, ds_type, image,
                device, hf_id,
                max_new_tokens=CONFIG["max_new_tokens"],
            )

            # Score
            predicted, score = score_item(generated_text, item, ds_type)

            # Universal response log — ground truth field varies by dataset type
            if ds_type == "rapm":
                ground_truth = item.get("correct_answer", "")
            elif ds_type in ("text_mc", "visual_mc"):
                # MMLU and ScienceQA use "answer_label"
                ground_truth = item.get(
                    "answer_label", item.get("correct_label", "")
                )
            elif ds_type == "text_math":
                # GSM8K uses "answer_number", Math500 uses "answer"
                ground_truth = str(item.get(
                    "answer_number", item.get("answer", "")
                ))
            elif ds_type == "visual_math":
                # MathVista uses "answer"
                ground_truth = str(item.get("answer", ""))
            elif ds_type == "text_qa":
                # TriviaQA uses "answer" (aliases handled in score_item)
                ground_truth = item.get("answer", "")
            elif ds_type == "rest":
                ground_truth = ""   # no ground truth for resting state
            else:
                ground_truth = str(item.get("answer", ""))

            response_log.append({
                "item_id"      : item.get("id", len(response_log) + 1),
                "ground_truth" : str(ground_truth),
                "generated"    : generated_text,
                "predicted"    : str(predicted) if predicted is not None else "",
                "score"        : score if score is not None else "",
            })

            all_acts.append(layer_acts)
            all_scores.append(score if score is not None else 0.0)
            all_predicted.append(predicted)

            # Progress display
            if score is not None:
                bar.set_postfix(score=f"{score:.2f}")
            else:
                bar.set_postfix(score="—")    # resting state

        except Exception as e:
            all_acts.append(None)
            all_scores.append(0.0)
            all_predicted.append(None)
            n_failed += 1
            response_log.append({
                "item_id"      : item.get("id", len(response_log) + 1),
                "ground_truth" : str(item.get(
                    "correct_answer",
                    item.get("answer_label",
                    item.get("answer_number",
                    item.get("answer", ""))))),
                "generated"    : f"[ERROR: {str(e)}]",
                "predicted"    : "",
                "score"        : 0,
            })

    # Check we have at least one valid activation
    valid = [a for a in all_acts if a is not None]
    if len(valid) == 0:
        raise RuntimeError(
            f"No valid activations for {ds_name} "
            f"({n_failed}/{len(items)} items failed)"
        )

    # Stack into (n_items, n_layers, hidden_dim)
    n_layers   = valid[0].shape[0]
    hidden_dim = valid[0].shape[1]
    act_matrix = np.zeros(
        (len(items), n_layers, hidden_dim), dtype=np.float32
    )
    for i, a in enumerate(all_acts):
        if a is not None:
            act_matrix[i] = a

    scores = np.array(all_scores, dtype=np.float32)

    # Save
    np.save(os.path.join(out_dir, "activations.npy"), act_matrix)
    np.save(os.path.join(out_dir, "scores.npy"),      scores)

    # Metadata
    meta = {
        "model_name"    : model_config["name"],
        "dataset"       : ds_name,
        "dataset_type"  : ds_type,
        "is_criterion"  : ds_cfg["criterion"],
        "n_items"       : len(items),
        "n_layers"      : n_layers,
        "hidden_dim"    : hidden_dim,
        "n_valid_acts"  : len(valid),
        "n_failed_items": n_failed,
        "mean_score"    : float(np.nanmean(scores)),
    }
    # Extra fields for criterion dataset
    if ds_cfg["criterion"]:
        valid_scores = scores[scores > 0]
        meta["total_score"] = int(np.sum(scores))
        meta["accuracy"]    = float(np.mean(scores))

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save response log for all datasets
    if response_log:
        pd.DataFrame(response_log).to_csv(
            os.path.join(out_dir, "responses.csv"), index=False
        )

    return len(items), float(np.nanmean(scores))

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["results_dir"],  exist_ok=True)
    os.makedirs(CONFIG["failures_dir"], exist_ok=True)

    device = CONFIG["device"]
    print(f"Device     : {device}")
    print(f"Pause file : '{os.path.abspath(CONFIG['pause_file'])}'\n")

    # Filter registry
    registry = [
        m for m in VLM_REGISTRY
        if CONFIG["models_to_run"] is None
        or m["name"] in CONFIG["models_to_run"]
    ]
    print(f"Models     : {len(registry)}")

    # Pre-load all datasets once
    dataset_items = load_all_datasets()
    if not dataset_items:
        print("No datasets found. Run dataset scripts first.")
        return

    # Build ordered list: RAPM first, then test datasets
    criterion_ds = [
        ds for ds in CONFIG["datasets"]
        if CONFIG["datasets"][ds]["criterion"]
        and ds in dataset_items
    ]
    test_ds = [
        ds for ds in CONFIG["datasets"]
        if not CONFIG["datasets"][ds]["criterion"]
        and ds in dataset_items
    ]
    run_order = criterion_ds + test_ds
    print(f"Run order  : {run_order}\n")

    # ── Model loop ─────────────────────────────────────────────────────────────
    for m_idx, model_config in enumerate(registry):
        model_name = model_config["name"]
        model_dir  = os.path.join(CONFIG["results_dir"], model_name)
        os.makedirs(model_dir, exist_ok=True)

        check_pause()

        print(f"{'='*60}")
        print(f"Model {m_idx+1}/{len(registry)}: {model_name}")
        print(f"{'='*60}")

        report = load_report(model_name)

        # Check if everything already done
        pending = [
            ds for ds in run_order
            if not should_skip(report, ds)
        ]
        if not pending:
            print("  All datasets completed — skipping.\n")
            continue
        print(f"  Pending : {pending}\n")

        # Load model
        try:
            model, processor = load_model(
                model_config, device,
                local_models_dir=CONFIG["local_models_dir"],
            )
        except Exception as e:
            print(f"  ✗ Load failed: {e}\n")
            for ds in pending:
                mark_failed(report, ds, f"load_failed: {e}")
            continue

        # ── Dataset loop ───────────────────────────────────────────────────────
        for ds_name in run_order:
            if should_skip(report, ds_name):
                print(f"  ✓ {ds_name:<12} already done — skipping")
                continue
            if ds_name not in dataset_items:
                print(f"  ⚠ {ds_name:<12} not loaded — skipping")
                continue

            check_pause()

            items, ds_cfg = dataset_items[ds_name]
            is_criterion  = ds_cfg["criterion"]
            tag           = " ← criterion" if is_criterion else ""
            print(f"\n  Running {ds_name}{tag} "
                  f"({len(items)} items, {ds_cfg['type']}) ...")

            try:
                n_items, mean_score = run_dataset(
                    model, processor, model_config,
                    ds_name, items, ds_cfg,
                    model_dir, device,
                )
                mark_success(report, ds_name, n_items)

                if ds_cfg["criterion"]:
                    total = int(round(mean_score * n_items))
                    print(f"    ✓ score = {total}/{n_items} "
                          f"(acc={mean_score:.3f})")
                elif ds_cfg["type"] == "rest":
                    print(f"    ✓ resting state complete "
                          f"({n_items} prompts)")
                else:
                    print(f"    ✓ mean_score = {mean_score:.3f}")

            except Exception as e:
                mark_failed(report, ds_name, str(e))
                print(f"    ✗ {ds_name} failed: {e}")

        # ── Unload ─────────────────────────────────────────────────────────────
        print(f"\n  Unloading {model_name} ...")
        unload_model(model, sleep_seconds=CONFIG["unload_sleep"])
        print(f"  Unloaded.\n")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Done':>5} {'Failed':>7} {'Pending':>8}")
    print("-" * 50)
    for model_config in registry:
        name    = model_config["name"]
        report  = load_report(name)
        ds_info = report["datasets"]
        n_ok    = sum(1 for s in ds_info.values()
                      if s["status"] == "success")
        n_fail  = sum(1 for s in ds_info.values()
                      if s["status"] == "failed")
        n_pend  = sum(1 for s in ds_info.values()
                      if s["status"] == "pending")
        print(f"  {name:<23} {n_ok:>5} {n_fail:>7} {n_pend:>8}")

    print(f"\nFailure reports: {os.path.abspath(CONFIG['failures_dir'])}")
    print(f"Results        : {os.path.abspath(CONFIG['results_dir'])}")

if __name__ == "__main__":
    main()