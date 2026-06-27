# scripts/plot_connectomes_vlm_multitask.py
#
# Phase 4 — Connectome visualization for PCA-enriched connectomes.
# Reads FC matrices from connectivity_pca/.
#
# Plots:
#   1. Per-model: one row of heatmaps, one per dataset
#      (how connectivity changes across task contexts for one model)
#   2. Per-dataset: grid of heatmaps for all models,
#      sorted by RAPM score best → worst
#
# Output: results/multitask/plots/connectomes_pca/

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    "connectivity_dir" : "results/multitask/connectivity_pca/",
    "output_dir"       : "results/multitask/plots/connectomes_pca/",
    "cmap"             : "RdBu_r",
    "vmin"             : -1.0,
    "vmax"             :  1.0,
    "dpi"              : 150,

    "connectome_ds"    : [
        "triviaqa", "gsm8k", "math500", "mmlu",
        "mathvista", "scienceqa", "rest",
    ],
    "ds_labels"        : {
        "triviaqa"  : "TriviaQA\n(Memory)",
        "gsm8k"     : "GSM8K\n(Math)",
        "math500"   : "MATH-500\n(Hard Math)",
        "mmlu"      : "MMLU\n(Knowledge)",
        "mathvista" : "MathVista\n(Visual Math)",
        "scienceqa" : "ScienceQA\n(Visual Science)",
        "rest"      : "Resting State",
    },

    "models_to_plot"   : None,   # None = all
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset_data(ds_name, config):
    fc_path  = os.path.join(
        config["connectivity_dir"], f"{ds_name}_fc_matrices.npy"
    )
    idx_path = os.path.join(
        config["connectivity_dir"], f"{ds_name}_indices.csv"
    )
    if not os.path.exists(fc_path) or not os.path.exists(idx_path):
        return None, [], []
    fc_arr = np.load(fc_path)
    df     = pd.read_csv(idx_path)
    names  = df["model_name"].tolist()
    scores = df["rapm_score"].tolist()
    return fc_arr, names, scores

def unpad_fc(fc_row):
    nonzero = np.any(fc_row != 0, axis=1)
    n = int(nonzero.sum())
    if n < 2:
        return fc_row
    return fc_row[:n, :n]

# ── Plot 1: Per-dataset grid (all models, sorted by RAPM) ─────────────────────

def plot_per_dataset(ds_name, fc_arr, names, scores, config):
    """
    One heatmap per model for a single dataset.
    Sorted by RAPM score best → worst.
    """
    n      = len(names)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3.8))
    fig.suptitle(
        f"PCA Connectomes — "
        f"{config['ds_labels'].get(ds_name, ds_name)}\n"
        f"(sorted by RAPM score, best → worst)",
        fontsize=11, y=1.01
    )
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.4
    )

    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    for plot_idx, data_idx in enumerate(order):
        row = plot_idx // n_cols
        col = plot_idx  % n_cols
        ax  = fig.add_subplot(gs[row, col])

        fc   = unpad_fc(fc_arr[data_idx])
        name = names[data_idx]
        sc   = scores[data_idx]
        n_n  = fc.shape[0]

        ax.imshow(
            fc, cmap=config["cmap"],
            vmin=config["vmin"], vmax=config["vmax"],
            aspect="auto", interpolation="nearest",
        )
        ax.set_title(f"{name}\nRAMP={sc}/36 | nodes={n_n}",
                     fontsize=7, pad=3)
        ax.set_xlabel("PCA node", fontsize=5)
        ax.set_ylabel("PCA node", fontsize=5)
        ticks = list(range(0, n_n, max(1, n_n//4)))
        ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=4)
        ax.set_yticks(ticks); ax.set_yticklabels(ticks, fontsize=4)

    # Empty panels
    for plot_idx in range(n, n_rows * n_cols):
        row = plot_idx // n_cols
        col = plot_idx  % n_cols
        ax  = fig.add_subplot(gs[row, col])
        ax.axis("off")

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        cmap=config["cmap"],
        norm=plt.Normalize(vmin=config["vmin"], vmax=config["vmax"])
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Pearson r")

    out = os.path.join(
        config["output_dir"], f"grid_{ds_name}.png"
    )
    plt.savefig(out, dpi=config["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ── Plot 2: Per-model row (all datasets in one row) ───────────────────────────

def plot_per_model(all_data, config):
    """
    For each model: one row of connectome heatmaps, one per dataset.
    Shows how connectivity changes across task contexts.
    """
    all_names = set()
    for ds_name in config["connectome_ds"]:
        _, names, _ = all_data.get(ds_name, (None, [], []))
        all_names.update(names)

    models = sorted(all_names)
    if config["models_to_plot"]:
        models = [m for m in models if m in config["models_to_plot"]]

    n_datasets = len(config["connectome_ds"])

    for model_name in models:
        fig, axes = plt.subplots(
            1, n_datasets, figsize=(n_datasets * 3.5, 4)
        )
        if n_datasets == 1:
            axes = [axes]
        fig.suptitle(
            f"{model_name} — PCA Connectomes Across Datasets",
            fontsize=11
        )

        for col, ds_name in enumerate(config["connectome_ds"]):
            ax = axes[col]
            fc_arr, names, scores = all_data.get(
                ds_name, (None, [], [])
            )

            if fc_arr is None or model_name not in names:
                ax.text(0.5, 0.5, "No data",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.set_title(
                    config["ds_labels"].get(ds_name, ds_name),
                    fontsize=7
                )
                ax.axis("off")
                continue

            idx   = names.index(model_name)
            fc    = unpad_fc(fc_arr[idx])
            n_n   = fc.shape[0]
            score = scores[idx]

            mean_r = np.abs(fc[~np.eye(n_n, dtype=bool)]).mean()

            ax.imshow(
                fc, cmap=config["cmap"],
                vmin=config["vmin"], vmax=config["vmax"],
                aspect="auto", interpolation="nearest",
            )
            ax.set_title(
                f"{config['ds_labels'].get(ds_name, ds_name)}\n"
                f"mean|r|={mean_r:.3f} | nodes={n_n}",
                fontsize=7
            )
            ax.set_xlabel("PCA node", fontsize=5)
            ax.set_ylabel("PCA node", fontsize=5)
            ticks = list(range(0, n_n, max(1, n_n//4)))
            ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=4)
            ax.set_yticks(ticks); ax.set_yticklabels(ticks, fontsize=4)

        # Shared colorbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(
            cmap=config["cmap"],
            norm=plt.Normalize(
                vmin=config["vmin"], vmax=config["vmax"]
            )
        )
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Pearson r")

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        safe_name = model_name.replace("/","_").replace(" ","_")
        out = os.path.join(
            config["output_dir"], f"model_{safe_name}.png"
        )
        plt.savefig(out, dpi=config["dpi"], bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Load all datasets
    print("Loading FC matrices ...")
    all_data = {}
    for ds_name in CONFIG["connectome_ds"]:
        fc_arr, names, scores = load_dataset_data(ds_name, CONFIG)
        if fc_arr is None:
            print(f"  {ds_name:<12} ✗ not found — skipping")
            continue
        if CONFIG["models_to_plot"]:
            keep   = [i for i, n in enumerate(names)
                      if n in CONFIG["models_to_plot"]]
            fc_arr = fc_arr[keep]
            names  = [names[i]  for i in keep]
            scores = [scores[i] for i in keep]
        all_data[ds_name] = (fc_arr, names, scores)
        print(f"  {ds_name:<12} ✓ {len(names)} models")
    print()

    # Plot 1: per-dataset grids
    print("Plotting per-dataset connectome grids ...")
    for ds_name, (fc_arr, names, scores) in all_data.items():
        plot_per_dataset(ds_name, fc_arr, names, scores, CONFIG)

    # Plot 2: per-model rows
    print("\nPlotting per-model connectome rows ...")
    plot_per_model(all_data, CONFIG)

    print(f"\nDone. All plots saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()