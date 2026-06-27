# scripts/connectivity_vlm_multitask_pca.py
#
# Phase 4 — Multi-task connectivity analysis with Layer-PCA enrichment.
# Approach A: fixed-k PCA — extract exactly k components per layer.
#
# For each layer of each model on each dataset:
#   - Extract top-k PCA components → (n_items, k) per layer
#     where k = CONFIG["pca_n_components"] (fixed, same for all layers)
#
# Concatenate across layers → (n_items, n_layers × k) activation matrix
# Pad to max node count across models → harmonized edge matrix
#
# Then: CPM edge selection (p < threshold) + LOOCV + NI
# Behavioral criterion: RAPM score (independent)
#
# Rationale for fixed k:
#   Parallel analysis answers "how many components are statistically
#   meaningful in this data?" — which with 100 items and hidden_dim=2048
#   yields k≈20, producing ~258k edges at N=10 models (intractable).
#   Fixed k=5 directly addresses the real constraint: statistical power
#   given N, giving ~16k edges — a 16x reduction.
#
# Output:
#   results/multitask/connectivity_pca/
#       <dataset>_NI.npy
#       <dataset>_sigma_pos.npy
#       <dataset>_sigma_neg.npy
#       <dataset>_fc_matrices.npy
#       <dataset>_indices.csv
#       <dataset>_edge_selection.csv
#       <dataset>_pca_log.csv       ← k and explained_var per (model, layer)
#       <dataset>_sensitivity.csv
#       summary_pca.csv

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    "results_dir"      : "results/multitask/",
    "output_dir"       : "results/multitask/connectivity_pca/",
    "pause_file"       : "PAUSE.txt",

    # Edge selection thresholds
    "p_threshold"      : 0.05,
    "p_thresholds"     : [0.05, 0.01, 0.001],

    # Fixed-k PCA settings
    # k=5 per layer: n_layers(24-36) × 5 = 120-180 nodes per model
    # max_nodes ≈ 180 → ~16k edges (vs ~258k with adaptive PA)
    "pca_n_components" : 6,
    "pca_seed"         : 42,

    "criterion_ds"     : "rapm",
    "connectome_ds"    : [
        "triviaqa",
        "gsm8k",
        "math500",
        "mmlu",
        "mathvista",
        "scienceqa",
        "rest",
    ],
}

# ── Load data ──────────────────────────────────────────────────────────────────

def load_model_list(results_dir):
    models = []
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results dir not found: {results_dir}")
    for model_name in sorted(os.listdir(results_dir)):
        model_dir  = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        rapm_meta  = os.path.join(model_dir, "rapm", "metadata.json")
        if not os.path.exists(rapm_meta):
            continue
        with open(rapm_meta) as f:
            meta = json.load(f)
        rapm_score = meta.get("total_score", None)
        if rapm_score is None:
            continue
        models.append({
            "name"      : model_name,
            "model_dir" : model_dir,
            "rapm_score": int(rapm_score),
        })
    print(f"Models with valid RAPM scores: {len(models)}")
    for m in models:
        print(f"  {m['name']:<25} RAPM: {m['rapm_score']}/36")
    print()
    return models

def load_activations(model_dir, dataset_name):
    path = os.path.join(model_dir, dataset_name, "activations.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)

# ── Fixed-k PCA enrichment ─────────────────────────────────────────────────────

def extract_pca_signals(acts, config, model_name, dataset_name):
    """
    For each layer: extract exactly k PCA components (fixed k).

    Args:
        acts : (n_items, n_layers, hidden_dim) activation array

    Returns:
        pca_signals : (n_items, n_layers × k) concatenated components
        pca_log     : list of dicts {model, dataset, layer, k, explained_var}
        k_per_layer : list of int (all equal to pca_n_components or less
                      if matrix rank is insufficient)
    """
    n_items, n_layers, hidden_dim = acts.shape
    k_target   = config["pca_n_components"]
    seed       = config["pca_seed"]
    pca_log    = []
    components = []

    for l in range(n_layers):
        X = acts[:, l, :].astype(np.float32)   # (n_items, hidden_dim)

        # Cap k at matrix rank — cannot exceed min(n_items-1, hidden_dim)
        k = min(k_target, n_items - 1, hidden_dim)

        pca = PCA(n_components=k, random_state=seed)
        try:
            scores        = pca.fit_transform(X)   # (n_items, k)
            explained_var = float(pca.explained_variance_ratio_.sum())
        except Exception:
            # Fallback to layer mean if PCA fails
            scores        = X.mean(axis=1, keepdims=True)
            k             = 1
            explained_var = float("nan")

        components.append(scores)
        pca_log.append({
            "model"        : model_name,
            "dataset"      : dataset_name,
            "layer"        : l,
            "k"            : k,
            "explained_var": round(explained_var, 4),
        })

    # Concatenate across layers → (n_items, Σk_l)
    pca_signals = np.concatenate(components, axis=1)
    k_per_layer = [entry["k"] for entry in pca_log]

    return pca_signals, pca_log, k_per_layer

# ── FC matrix from PCA signals ─────────────────────────────────────────────────

def compute_fc_from_pca(pca_signals):
    """
    (n_items, total_nodes) → (total_nodes, total_nodes) FC matrix.
    Pearson correlation across items for each pair of PCA nodes.
    """
    fc = np.corrcoef(pca_signals.T)   # (total_nodes, total_nodes)
    fc = np.nan_to_num(fc, nan=0.0)
    np.fill_diagonal(fc, 0.0)
    return fc

def fc_to_edge_vector(fc, max_edges):
    idx  = np.triu_indices(fc.shape[0], k=1)
    edge = fc[idx]
    if len(edge) < max_edges:
        edge = np.pad(edge, (0, max_edges - len(edge)))
    return edge[:max_edges]

# ── Build edge matrix across models ───────────────────────────────────────────

def build_edge_matrix_pca(models, dataset_name, config):
    """
    Build (n_valid_models, n_edges) edge matrix using PCA-enriched signals.
    Returns edge_matrix, fc_list, valid_idx, all_pca_logs, max_nodes.
    """
    fc_list      = []
    valid_idx    = []
    all_pca_logs = []
    node_counts  = []

    print(f"  Running layer-PCA ...")
    for i, m in enumerate(models):
        acts = load_activations(m["model_dir"], dataset_name)
        if acts is None:
            fc_list.append(None)
            continue

        pca_signals, pca_log, k_per_layer = extract_pca_signals(
            acts, config, m["name"], dataset_name
        )
        fc = compute_fc_from_pca(pca_signals)

        fc_list.append(fc)
        valid_idx.append(i)
        all_pca_logs.extend(pca_log)
        node_counts.append(fc.shape[0])

        total_nodes = sum(k_per_layer)
        print(f"    {m['name']:<25} "
              f"n_layers={acts.shape[1]} "
              f"total_nodes={total_nodes} "
              f"(k={config['pca_n_components']} per layer)")

    if not valid_idx:
        return None, fc_list, valid_idx, all_pca_logs, 0

    max_nodes = max(node_counts)
    max_edges = max_nodes * (max_nodes - 1) // 2
    print(f"  Max nodes across models: {max_nodes} "
          f"→ {max_edges} edges\n")

    # Build edge vectors with padding to max_edges
    edge_rows = []
    for i in valid_idx:
        fc   = fc_list[i]
        edge = fc_to_edge_vector(fc, max_edges)
        edge_rows.append(edge)

    edge_matrix = np.array(edge_rows)   # (n_valid, max_edges)
    return edge_matrix, fc_list, valid_idx, all_pca_logs, max_nodes

# ── CPM pipeline (identical to standard connectivity script) ──────────────────

def correlate_edges_with_pvalues(X, y):
    r_arr = np.zeros(X.shape[1])
    p_arr = np.ones( X.shape[1])
    for j in range(X.shape[1]):
        edge = X[:, j]
        if edge.std() < 1e-10:
            continue
        r, p       = stats.pearsonr(edge, y)
        r_arr[j]   = r
        p_arr[j]   = p
    return r_arr, p_arr

def select_edges(r_arr, p_arr, threshold):
    pos_mask = (r_arr > 0) & (p_arr < threshold)
    neg_mask = (r_arr < 0) & (p_arr < threshold)
    return pos_mask, neg_mask

def compute_cpm_index(edge_vec, pos_mask, neg_mask):
    sigma_pos = float(np.sum(edge_vec[pos_mask]))
    sigma_neg = float(np.sum(edge_vec[neg_mask]))
    return sigma_pos, sigma_neg, sigma_pos - sigma_neg

def run_loocv(X, y, threshold):
    n                  = X.shape[0]
    sigma_pos          = np.zeros(n)
    sigma_neg          = np.zeros(n)
    NI                 = np.zeros(n)
    edge_selection_log = []

    print(f"  LOOCV ({n} folds, p < {threshold}) ...")
    for i in range(n):
        train        = np.delete(np.arange(n), i)
        r_arr, p_arr = correlate_edges_with_pvalues(X[train], y[train])
        pos, neg     = select_edges(r_arr, p_arr, threshold)

        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        if n_pos + n_neg == 0:
            print(f"    ⚠ Fold {i}: no edges selected")

        edge_selection_log.append({
            "fold"           : i,
            "n_edges_total"  : X.shape[1],
            "n_pos_selected" : n_pos,
            "n_neg_selected" : n_neg,
        })
        sigma_pos[i], sigma_neg[i], NI[i] = compute_cpm_index(
            X[i], pos, neg
        )

    mean_pos = np.mean([s["n_pos_selected"] for s in edge_selection_log])
    mean_neg = np.mean([s["n_neg_selected"] for s in edge_selection_log])
    print(f"  Mean edges selected: "
          f"pos={mean_pos:.1f}  neg={mean_neg:.1f} "
          f"/ {X.shape[1]} total\n")

    return sigma_pos, sigma_neg, NI, edge_selection_log

def fit_linear(x, y, label="NI"):
    if x.std() < 1e-10:
        print(f"  [{label}] constant — skipping")
        return {"r": 0.0, "p": 1.0, "R2": 0.0}
    r, p = stats.pearsonr(x, y)
    lm   = LinearRegression().fit(x.reshape(-1,1), y)
    R2   = lm.score(x.reshape(-1,1), y)
    sig  = "*" if p < 0.05 else ("†" if p < 0.10 else "")
    print(f"  [{label}] r={r:+.4f}  p={p:.4e}  R²={R2:.4f} {sig}")
    return {"r": float(r), "p": float(p), "R2": float(R2)}

def min_r_for_p(n, p_threshold):
    t_crit = stats.t.ppf(1 - p_threshold/2, df=n-2)
    return float(t_crit / np.sqrt(t_crit**2 + n - 2))

# ── Pause mechanism ────────────────────────────────────────────────────────────

def check_pause():
    """
    Check for PAUSE.txt in project root.
    If present, halt execution until the file is deleted.
    Create PAUSE.txt to pause between datasets.
    Delete PAUSE.txt to resume.
    """
    pause_file = CONFIG["pause_file"]
    if os.path.exists(pause_file):
        print(f"\n⏸  PAUSED — delete '{pause_file}' to resume ...")
        while os.path.exists(pause_file):
            time.sleep(5)
        print("▶  Resuming ...\n")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    models = load_model_list(CONFIG["results_dir"])
    if len(models) < 3:
        print("Need at least 3 models. Exiting.")
        return

    y_all = np.array([m["rapm_score"] for m in models], dtype=float)

    print(f"Behavioral criterion : RAPM score")
    print(f"Connectivity method  : Layer-PCA (fixed k={CONFIG['pca_n_components']})")
    print(f"PCA components/layer : {CONFIG['pca_n_components']} "
          f"(seed={CONFIG['pca_seed']})")
    print(f"Expected nodes/model : n_layers × {CONFIG['pca_n_components']} "
          f"(e.g. 36 layers → 180 nodes → 16110 edges)")
    print(f"Edge selection       : p < {CONFIG['p_threshold']}")
    print(f"Min |r| for p<{CONFIG['p_threshold']} "
          f"(N-1={len(models)-1}): "
          f"{min_r_for_p(len(models)-1, CONFIG['p_threshold']):.3f}\n")

    summary_rows  = []
    all_pca_logs  = []

    for ds_name in CONFIG["connectome_ds"]:
        print(f"{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # Build PCA-enriched edge matrix
        edge_matrix, fc_list, valid_idx, pca_logs, max_nodes = \
            build_edge_matrix_pca(models, ds_name, CONFIG)

        all_pca_logs.extend(pca_logs)

        if edge_matrix is None or len(valid_idx) < 3:
            print(f"  Insufficient data — skipping.\n")
            summary_rows.append({
                "dataset" : ds_name,
                "n_models": len(valid_idx),
                "r_NI"    : np.nan,
                "p_NI"    : np.nan,
                "R2_NI"   : np.nan,
                "status"  : "insufficient_data",
            })
            continue

        valid_models = [models[i] for i in valid_idx]
        y            = np.array(
            [m["rapm_score"] for m in valid_models], dtype=float
        )
        names        = [m["name"] for m in valid_models]

        print(f"  Valid models : {len(valid_models)}")
        print(f"  Edge matrix  : {edge_matrix.shape}")
        print(f"  RAPM scores  : {y.astype(int).tolist()}\n")

        # Primary LOOCV
        sp, sn, NI, sel_log = run_loocv(
            edge_matrix, y, CONFIG["p_threshold"]
        )

        print(f"  NI    range: [{NI.min():.4f}, {NI.max():.4f}]")
        print(f"  Σ_pos range: [{sp.min():.4f}, {sp.max():.4f}]")
        print(f"  Σ_neg range: [{sn.min():.4f}, {sn.max():.4f}]\n")

        print(f"  Predictive validity (~ RAPM score):")
        stats_NI  = fit_linear(NI, y, "NI")
        stats_pos = fit_linear(sp, y, "Σ_pos")
        stats_neg = fit_linear(sn, y, "Σ_neg")

        # Sensitivity analysis
        print(f"\n  Sensitivity across thresholds:")
        sens_rows = []
        for thr in CONFIG["p_thresholds"]:
            sp_t, sn_t, NI_t, _ = run_loocv(edge_matrix, y, thr)
            r_t, p_t = (stats.pearsonr(NI_t, y)
                        if NI_t.std() > 1e-10 else (0.0, 1.0))
            sig = "*" if p_t < 0.05 else ("†" if p_t < 0.10 else "")
            print(f"    p<{thr:.2f}: r={r_t:+.4f}  p={p_t:.4e} {sig}")
            sens_rows.append({
                "threshold": thr,
                "r_NI"     : float(r_t),
                "p_NI"     : float(p_t),
            })

        # ── Save ──────────────────────────────────────────────────────────────
        prefix = os.path.join(CONFIG["output_dir"], ds_name)

        np.save(f"{prefix}_NI.npy",       NI)
        np.save(f"{prefix}_sigma_pos.npy", sp)
        np.save(f"{prefix}_sigma_neg.npy", sn)
        np.save(f"{prefix}_y_rapm.npy",    y)

        # FC matrices — padded to (n_models, max_nodes, max_nodes)
        valid_fcs = [fc_list[i] for i in valid_idx]
        fc_arr    = np.zeros(
            (len(valid_fcs), max_nodes, max_nodes), dtype=np.float32
        )
        for k_idx, fc in enumerate(valid_fcs):
            nl = fc.shape[0]
            fc_arr[k_idx, :nl, :nl] = fc
        np.save(f"{prefix}_fc_matrices.npy", fc_arr)

        # Indices CSV
        pd.DataFrame({
            "model_name": names,
            "rapm_score": y.astype(int),
            "NI"        : NI,
            "sigma_pos" : sp,
            "sigma_neg" : sn,
        }).to_csv(f"{prefix}_indices.csv", index=False)

        # Edge selection log
        df_sel             = pd.DataFrame(sel_log)
        df_sel["model_name"] = [names[r["fold"]] for r in sel_log]
        df_sel.to_csv(f"{prefix}_edge_selection.csv", index=False)

        # Sensitivity
        pd.DataFrame(sens_rows).to_csv(
            f"{prefix}_sensitivity.csv", index=False
        )

        # PCA log for this dataset
        pd.DataFrame(pca_logs).to_csv(
            f"{prefix}_pca_log.csv", index=False
        )

        print(f"\n  Saved to {prefix}_*.npy / *.csv\n")

        summary_rows.append({
            "dataset"   : ds_name,
            "n_models"  : len(valid_models),
            "max_nodes" : max_nodes,
            "n_edges"   : edge_matrix.shape[1],
            "threshold" : CONFIG["p_threshold"],
            "r_NI"      : stats_NI["r"],
            "p_NI"      : stats_NI["p"],
            "R2_NI"     : stats_NI["R2"],
            "r_pos"     : stats_pos["r"],
            "p_pos"     : stats_pos["p"],
            "r_neg"     : stats_neg["r"],
            "p_neg"     : stats_neg["p"],
            "status"    : "ok",
        })

        # Pause check — create PAUSE.txt to pause after this dataset
        check_pause()

    # ── Global PCA log ─────────────────────────────────────────────────────────
    pd.DataFrame(all_pca_logs).to_csv(
        os.path.join(CONFIG["output_dir"], "pca_log_all.csv"),
        index=False
    )

    # ── Comparison summary ─────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows).sort_values(
        "r_NI", key=abs, ascending=False
    )
    summary_path = os.path.join(CONFIG["output_dir"], "summary_pca.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("PCA CONNECTIVITY — PREDICTIVE VALIDITY COMPARISON")
    print(f"(sorted by |r_NI|, edge selection p<{CONFIG['p_threshold']})")
    print(f"{'='*60}")
    print(f"{'Dataset':<14} {'N':>4} {'Nodes':>7} "
          f"{'Edges':>8} {'r_NI':>8} {'p_NI':>10} {'R2':>7}")
    print("-" * 60)
    for _, row in df_summary.iterrows():
        if row["status"] != "ok":
            print(f"  {row['dataset']:<12}  insufficient data")
            continue
        sig = "*" if row["p_NI"] < 0.05 else \
              "†" if row["p_NI"] < 0.10 else ""
        print(
            f"  {row['dataset']:<12} {int(row['n_models']):>4} "
            f"{int(row['max_nodes']):>7} "
            f"{int(row['n_edges']):>8} "
            f"{row['r_NI']:>+8.4f} "
            f"{row['p_NI']:>10.4e} "
            f"{row['R2_NI']:>7.4f} {sig}"
        )

    # H1/H2 checks
    valid = df_summary[df_summary["status"] == "ok"]
    if len(valid) > 0:
        best = valid.iloc[0]
        print(f"\n  Best connectome: {best['dataset']} "
              f"(r={best['r_NI']:+.4f})")
        if best["dataset"] == "rest":
            print("  ✓ H1 SUPPORTED: resting state best predicts RAPM")
        else:
            print(f"  H1 not supported: {best['dataset']} "
                  f"outperforms resting state")

        visual_ds = {"mathvista", "scienceqa"}
        txt_ds    = {"triviaqa", "gsm8k", "math500", "mmlu"}
        vis_rows  = valid[valid["dataset"].isin(visual_ds)]
        txt_rows  = valid[valid["dataset"].isin(txt_ds)]
        if len(vis_rows) > 0 and len(txt_rows) > 0:
            if abs(vis_rows.iloc[0]["r_NI"]) > abs(txt_rows.iloc[0]["r_NI"]):
                print(f"  ✓ H2 SUPPORTED: visual connectomes outperform text")
            else:
                print(f"  H2 not supported: text connectomes outperform visual")

    print(f"\n  Sign convention:")
    print(f"    r_NI > 0 : higher NI → higher RAPM score")
    print(f"    r_NI < 0 : higher NI → lower RAPM score")
    print(f"\nSummary : {summary_path}")
    print(f"PCA log : {os.path.join(CONFIG['output_dir'], 'pca_log_all.csv')}")
    print(f"(k={CONFIG['pca_n_components']} components per layer — "
          f"fixed across all models and datasets)")

if __name__ == "__main__":
    main()