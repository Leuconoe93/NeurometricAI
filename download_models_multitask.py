# scripts/download_models_multitask.py
#
# Phase 4 — Download all VLMs to local AI_LIBRARY.
# Fully standalone — no imports from other phase scripts.
#
# Usage:
#   python scripts/download_models_multitask.py
#   python scripts/download_models_multitask.py --models SmolVLM-256M Qwen2-VL-2B

import os
import sys
import json
import argparse

os.environ["HF_TOKEN"] = "YOUR_TOKEN_HERE"  # Replace with your Hugging Face token
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models_vlm_multitask import VLM_REGISTRY

LOCAL_DIR = "D:/Data Science/AI_LIBRARY/"
LOG_PATH  = os.path.join(LOCAL_DIR, "download_log_multitask.json")

# ── Verification ───────────────────────────────────────────────────────────────

def is_downloaded(model_path):
    if not os.path.exists(model_path):
        return False, "folder missing"
    files = os.listdir(model_path)
    if not files:
        return False, "folder empty"
    if "config.json" not in files:
        return False, "config.json missing"
    has_weights = any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in files
    )
    if not has_weights:
        return False, "no weight files"
    return True, "ok"

def folder_size_gb(path):
    if not os.path.exists(path):
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / (1024 ** 3)

# ── Download ───────────────────────────────────────────────────────────────────

def download_model(config):
    from huggingface_hub import snapshot_download

    name       = config["name"]
    hf_id      = config["hf_id"]
    model_path = os.path.join(LOCAL_DIR, name)

    ok, reason = is_downloaded(model_path)
    if ok:
        size = folder_size_gb(model_path)
        print(f"  ✓ Already downloaded ({size:.1f} GB) — skipping\n")
        return model_path, True, "already_downloaded"

    if reason not in ("folder missing", "folder empty"):
        print(f"  ⚠ Incomplete ({reason}) — redownloading ...")

    print(f"  Downloading: {hf_id}")
    print(f"  Saving to  : {model_path}")
    os.makedirs(model_path, exist_ok=True)

    try:
        snapshot_download(
            repo_id         = hf_id,
            local_dir       = model_path,
            ignore_patterns = [
                "*.msgpack", "*.h5",
                "flax_model*", "tf_model*", "rust_model*",
            ],
        )
        ok, reason = is_downloaded(model_path)
        if ok:
            size = folder_size_gb(model_path)
            print(f"  ✓ Done ({size:.1f} GB)\n")
            return model_path, True, "downloaded"
        else:
            print(f"  ✗ Verification failed: {reason}\n")
            return model_path, False, f"verify_failed: {reason}"
    except Exception as e:
        print(f"  ✗ Failed: {e}\n")
        return None, False, f"error: {e}"

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model names to download. Default: all in registry."
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check status, do not download."
    )
    args = parser.parse_args()

    os.makedirs(LOCAL_DIR, exist_ok=True)

    registry = VLM_REGISTRY
    if args.models:
        registry = [m for m in VLM_REGISTRY if m["name"] in args.models]
        missing  = set(args.models) - {m["name"] for m in registry}
        if missing:
            print(f"Warning: not in registry: {missing}\n")

    print(f"{'='*60}")
    print(f"Phase 4 VLM Download Manager")
    print(f"Library: {os.path.abspath(LOCAL_DIR)}")
    print(f"Models : {len(registry)}")
    print(f"{'='*60}\n")

    # Load log
    log = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)

    if args.check_only:
        print("Download status:\n")
        for m in registry:
            path = os.path.join(LOCAL_DIR, m["name"])
            ok, reason = is_downloaded(path)
            size   = folder_size_gb(path) if ok else 0
            status = f"✓ Downloaded ({size:.1f} GB)" if ok \
                     else f"✗ {reason}"
            print(f"  {m['name']:<25} {status}")
        return

    results = []
    for config in registry:
        name = config["name"]
        print(f"{'─'*60}")
        print(f"Model : {name} ({config['params_B']}B | {config['family']})")
        if config["requires_auth"]:
            print(f"  ⚠ Requires HF login: huggingface-cli login")

        path, success, status = download_model(config)
        log[name] = {
            "hf_id"   : config["hf_id"],
            "path"    : path,
            "success" : success,
            "status"  : status,
        }
        with open(LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)
        results.append((name, success, path))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    ok_list   = [(n, p) for n, s, p in results if s]
    fail_list = [(n,)   for n, s, p in results if not s]

    print(f"  Succeeded : {len(ok_list)}")
    print(f"  Failed    : {len(fail_list)}")

    total_gb = sum(folder_size_gb(p) for _, p in ok_list if p)
    print(f"  Total size: {total_gb:.1f} GB")

    if fail_list:
        print(f"\n  Failed models:")
        for (n,) in fail_list:
            print(f"    - {n}")

    print(f"\n  Log: {LOG_PATH}")

if __name__ == "__main__":
    main()