# scripts/plot_graphs_vlm_multitask.py
#
# Phase 4 — Graph visualization for PCA-enriched connectomes.
# Reads fingerprints and model selection from graph_pca/.
#
# Plots:
#   1. Per-model: bar plots of key graph metrics across datasets
#      (one subplot per metric, dataset on X axis)
#   2. Per-dataset: radar grid (all models, one dataset)
#   3. WS win rate summary bar chart
#   4. Small-world sigma heatmap (model × dataset)
#
# Output: results/multitask/plots/graphs_pca/

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    "graph_dir"       : "results/multitask/graph_pca/",
    "output_dir"      : "results/multitask/plots/graphs_pca/",
    "dpi"             : 150,

    "connectome_ds"   : [
        "triviaqa", "gsm8k", "math500", "mmlu",
        "mathvista", "scienceqa", "rest",
    ],
    "ds_labels"       : {
        "triviaqa"  : "TriviaQA",
        "gsm8k"     : "GSM8K",
        "math500"   : "MATH-500",
        "mmlu"      : "MMLU",
        "mathvista" : "MathVista",
        "scienceqa" : "ScienceQA",
        "rest"      : "Rest",
    },

    # Radar metrics
    "radar_metrics_pos": [
        "pos_clustering", "pos_global_efficiency",
        "pos_small_world_sigma", "pos_modularity_Q",
        "pos_assortativity", "pos_algebraic_connectivity",
    ],
    "radar_metrics_neg": [
        "neg_clustering", "neg_global_efficiency",
        "neg_small_world_sigma", "neg_modularity_Q",
        "neg_assortativity", "neg_algebraic_connectivity",
    ],
    "radar_labels"     : [
        "Clust.", "Effic.", "SW-σ",
        "Mod.", "Assort.", "AlgConn"
    ],

    # Per-model bar plot metrics
    "bar_metrics"     : [
        ("pos_small_world_sigma", "Small-world σ"),
        ("pos_modularity_Q",      "Modularity Q"),
        ("pos_global_efficiency", "Global Efficiency"),
        ("pos_algebraic_connectivity", "AlgConn"),
    ],

    "models_to_plot"  : None,
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_fingerprints(ds_name, config):
    path = os.path.join(
        config["graph_dir"], f"fingerprints_{ds_name}.csv"
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if config["models_to_plot"]:
        df = df[df["model_name"].isin(config["models_to_plot"])]
    return df

def load_model_selection(ds_name, config):
    path = os.path.join(
        config["graph_dir"], f"model_selection_{ds_name}.csv"
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

# ── Radar helpers ──────────────────────────────────────────────────────────────

def get_normalized_radar_vals(row, metric_list):
    vals = []
    for m in metric_list:
        v = row.get(m, 0.0)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        vals.append(float(v))
    arr = np.clip(np.array(vals), 0, None)
    mx  = arr.max()
    if mx > 0:
        arr = arr / mx
    return arr.tolist() + [arr[0]]

# ── Plot 1: Per-model bar plots across datasets ────────────────────────────────

def plot_per_model_metrics(all_fp, config):
    """
    For each model: bar plots of key graph metrics across datasets.
    One subplot per metric, dataset on X axis.
    Resting state bar highlighted in red.
    """
    all_names = set()
    for df in all_fp.values():
        if df is not None:
            all_names.update(df["model_name"].tolist())

    models = sorted(all_names)
    if config["models_to_plot"]:
        models = [m for m in models if m in config["models_to_plot"]]

    ds_list  = config["connectome_ds"]
    ds_short = [config["ds_labels"].get(d, d) for d in ds_list]
    metrics  = config["bar_metrics"]

    for model_name in models:
        fig, axes = plt.subplots(
            1, len(metrics), figsize=(len(metrics) * 4, 4)
        )
        if len(metrics) == 1:
            axes = [axes]
        fig.suptitle(
            f"{model_name} — Graph Metrics Across Datasets "
            f"(PCA connectomes)",
            fontsize=11
        )

        for ax, (metric, label) in zip(axes, metrics):
            vals   = []
            colors = []
            for ds_name in ds_list:
                df = all_fp.get(ds_name)
                if df is None:
                    vals.append(np.nan)
                else:
                    sub = df[df["model_name"] == model_name]
                    vals.append(
                        float(sub[metric].values[0])
                        if len(sub) > 0 else np.nan
                    )
                colors.append(
                    "#e74c3c" if ds_name == "rest" else "#3498db"
                )

            ax.bar(range(len(ds_list)), vals,
                   color=colors, alpha=0.85)
            ax.set_title(label, fontsize=9)
            ax.set_xticks(range(len(ds_list)))
            ax.set_xticklabels(ds_short, rotation=40,
                               ha="right", fontsize=7)
            ax.set_ylabel(label, fontsize=7)

            # Reference line at resting state value
            rest_idx = ds_list.index("rest") \
                if "rest" in ds_list else None
            if rest_idx is not None and \
               not np.isnan(vals[rest_idx]):
                ax.axhline(vals[rest_idx], color="#e74c3c",
                           linewidth=1, linestyle="--", alpha=0.5)

        legend_elements = [
            Patch(color="#e74c3c", label="Resting state"),
            Patch(color="#3498db", label="Task datasets"),
        ]
        fig.legend(handles=legend_elements,
                   loc="lower right", fontsize=8)
        plt.tight_layout()

        safe_name = model_name.replace("/","_").replace(" ","_")
        out = os.path.join(
            config["output_dir"],
            f"metrics_{safe_name}.png"
        )
        plt.savefig(out, dpi=config["dpi"], bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")

# ── Plot 2: Radar grid per dataset ────────────────────────────────────────────

def plot_radar_grid_per_dataset(ds_name, fp_df, config):
    """Radar chart grid — all models for one dataset, sorted by RAPM."""
    fp_df  = fp_df.sort_values("rapm_score", ascending=False)
    models = fp_df["model_name"].tolist()
    scores = fp_df["rapm_score"].tolist()
    n      = len(models)
    if n == 0:
        return

    n_cols  = min(4, n)
    n_rows  = (n + n_cols - 1) // n_cols
    angles  = np.linspace(0, 2*np.pi,
                          len(config["radar_labels"]),
                          endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 4.5))
    fig.suptitle(
        f"Graph Fingerprints — "
        f"{config['ds_labels'].get(ds_name, ds_name)} "
        f"(PCA connectomes)\n"
        f"red=positive | blue=negative | sorted by RAPM",
        fontsize=11, y=1.01
    )

    for idx, (model_name, score) in enumerate(zip(models, scores)):
        ax  = fig.add_subplot(n_rows, n_cols, idx+1, polar=True)
        row = fp_df[fp_df["model_name"] == model_name].iloc[0].to_dict()

        vp = get_normalized_radar_vals(
            row, config["radar_metrics_pos"]
        )
        vn = get_normalized_radar_vals(
            row, config["radar_metrics_neg"]
        )

        ax.plot(angles, vp, "o-", color="#d62728",
                linewidth=1.5, markersize=3)
        ax.fill(angles, vp, color="#d62728", alpha=0.15)
        ax.plot(angles, vn, "o-", color="#1f77b4",
                linewidth=1.5, markersize=3)
        ax.fill(angles, vn, color="#1f77b4", alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(config["radar_labels"], fontsize=6)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["0.5", "1.0"], fontsize=4)
        ax.set_title(f"{model_name}\nRAMP={score}/36",
                     fontsize=7, pad=8)

    # Empty panels
    for idx in range(n, n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, idx+1, polar=True)
        ax.axis("off")

    legend_elements = [
        Line2D([0],[0], color="#d62728", lw=2, label="Positive"),
        Line2D([0],[0], color="#1f77b4", lw=2, label="Negative"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()

    out = os.path.join(
        config["output_dir"], f"radar_{ds_name}.png"
    )
    plt.savefig(out, dpi=config["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ── Plot 3: WS win rate summary ───────────────────────────────────────────────

def plot_ws_win_rate(all_ms, config):
    ds_list = [d for d in config["connectome_ds"]
               if all_ms.get(d) is not None]
    if not ds_list:
        return

    ws_pos, ws_neg, totals = [], [], []
    for ds_name in ds_list:
        df = all_ms[ds_name]
        n  = len(df)
        totals.append(n)
        ws_pos.append(
            (df["pos_best_BIC"] == "WS").sum() / n if n > 0 else 0
        )
        ws_neg.append(
            (df["neg_best_BIC"] == "WS").sum() / n if n > 0 else 0
        )

    x      = np.arange(len(ds_list))
    width  = 0.35
    labels = [config["ds_labels"].get(d, d) for d in ds_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, ws_pos, width, label="Positive subgraph",
           color="#d62728", alpha=0.8)
    ax.bar(x + width/2, ws_neg, width, label="Negative subgraph",
           color="#1f77b4", alpha=0.8)
    ax.axhline(1.0, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.5)

    ax.set_title(
        "Watts-Strogatz Win Rate by Dataset (PCA connectomes)\n"
        "(proportion of models best fit by WS, by BIC)",
        fontsize=11
    )
    ax.set_ylabel("WS win rate", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)

    for i, (n, wp, wn) in enumerate(zip(totals, ws_pos, ws_neg)):
        ax.text(i, 1.08, f"N={n}", ha="center",
                fontsize=7, color="gray")

    plt.tight_layout()
    out = os.path.join(config["output_dir"], "ws_win_rate.png")
    plt.savefig(out, dpi=config["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ── Plot 4: Small-world sigma heatmap ─────────────────────────────────────────

def plot_sigma_heatmap(all_fp, config):
    """Heatmap: small-world sigma per (model × dataset)."""
    ds_list = [d for d in config["connectome_ds"]
               if all_fp.get(d) is not None]
    if not ds_list:
        return

    all_names = set()
    for ds_name in ds_list:
        all_names.update(all_fp[ds_name]["model_name"].tolist())
    models = sorted(all_names)

    matrix = np.full((len(models), len(ds_list)), np.nan)
    for j, ds_name in enumerate(ds_list):
        df = all_fp[ds_name]
        for i, model in enumerate(models):
            sub = df[df["model_name"] == model]
            if len(sub) > 0:
                matrix[i, j] = float(
                    sub["pos_small_world_sigma"].values[0]
                )

    ds_labels = [config["ds_labels"].get(d, d) for d in ds_list]

    fig, ax = plt.subplots(
        figsize=(len(ds_list) * 1.5 + 2, len(models) * 0.7 + 2)
    )
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=0.5, vmax=3.0)

    ax.set_xticks(range(len(ds_list)))
    ax.set_xticklabels(ds_labels, rotation=35,
                       ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)

    for i in range(len(models)):
        for j in range(len(ds_list)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}",
                        ha="center", va="center",
                        fontsize=7,
                        color="black" if 0.8 < matrix[i,j] < 2.5
                        else "white")

    plt.colorbar(im, ax=ax, label="Small-world σ (pos subgraph)")
    ax.set_title(
        "Small-World σ — PCA Connectomes\n"
        "σ > 1 = small-world | green=higher σ",
        fontsize=10
    )
    plt.tight_layout()
    out = os.path.join(config["output_dir"], "sigma_heatmap.png")
    plt.savefig(out, dpi=config["dpi"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("Loading graph analysis data ...")
    all_fp = {}
    all_ms = {}
    for ds_name in CONFIG["connectome_ds"]:
        fp = load_fingerprints(ds_name, CONFIG)
        ms = load_model_selection(ds_name, CONFIG)
        if fp is None:
            print(f"  {ds_name:<12} ✗ fingerprints not found")
        else:
            all_fp[ds_name] = fp
            print(f"  {ds_name:<12} ✓ {len(fp)} models")
        all_ms[ds_name] = ms
    print()

    # Plot 1: per-model metric bars
    print("Plotting per-model graph metrics ...")
    plot_per_model_metrics(all_fp, CONFIG)

    # Plot 2: radar grids per dataset
    print("\nPlotting radar grids per dataset ...")
    for ds_name, fp_df in all_fp.items():
        plot_radar_grid_per_dataset(ds_name, fp_df, CONFIG)

    # Plot 3: WS win rate
    print("\nPlotting WS win rates ...")
    plot_ws_win_rate(all_ms, CONFIG)

    # Plot 4: sigma heatmap
    print("\nPlotting small-world sigma heatmap ...")
    plot_sigma_heatmap(all_fp, CONFIG)

    print(f"\nDone. All plots saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()