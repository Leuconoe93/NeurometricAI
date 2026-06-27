# scripts/connectivity_vlm_multitask.py
#
# Phase 4 — Multi-task connectivity analysis.
# Standard pipeline: CPM edge selection (Finn et al., 2015).
#
# For each connectome dataset:
#   - Load activations → compute FC matrix per model
#   - LOOCV: select edges by significance (p < threshold)
#   - Build NI from selected positive/negative networks
#   - Correlate NI with RAPM score (behavioral criterion)
#
# Edge selection procedure (per LOOCV fold):
#   1. Correlate each edge with RAPM score across training subjects
#   2. Positive network: edges where r>0 AND p<threshold
#   3. Negative network: edges where r<0 AND p<threshold
#   4. NI = Σ_pos - Σ_neg for left-out subject
#
# Output:
#   results/multitask/connectivity/
#       <dataset>_NI.npy
#       <dataset>_sigma_pos.npy
#       <dataset>_sigma_neg.npy
#       <dataset>_y_rapm.npy
#       <dataset>_fc_matrices.npy
#       <dataset>_indices.csv
#       <dataset>_edge_selection.csv
#       <dataset>_sensitivity.csv
#       summary_all_datasets.csv

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    "results_dir"    : "results/multitask/",
    "output_dir"     : "results/multitask/connectivity/",

    # Primary edge selection threshold
    "p_threshold"    : 0.05,

    # Sensitivity analysis thresholds
    "p_thresholds"   : [0.05, 0.01, 0.10],

    "criterion_ds"   : "rapm",
    "connectome_ds"  : [
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
    """
    Scan results/multitask/ for models with completed RAPM.
    Returns list of dicts with name, model_dir, rapm_score.
    """
    models = []
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    for model_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        rapm_meta = os.path.join(model_dir, "rapm", "metadata.json")
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

# ── Functional connectivity ────────────────────────────────────────────────────

def compute_fc_matrix(acts):
    """(n_items, n_layers, hidden_dim) → (n_layers, n_layers)."""
    layer_signals = acts.mean(axis=2)
    fc            = np.corrcoef(layer_signals.T)
    fc            = np.nan_to_num(fc, nan=0.0)
    np.fill_diagonal(fc, 0.0)
    return fc

def fc_to_edge_vector(fc, max_edges):
    idx  = np.triu_indices(fc.shape[0], k=1)
    edge = fc[idx]
    if len(edge) < max_edges:
        edge = np.pad(edge, (0, max_edges - len(edge)))
    return edge[:max_edges]

def build_edge_matrix(models, dataset_name):
    """
    Build (n_valid_models, n_edges) edge matrix for one dataset.
    Also returns full FC matrices for saving.
    """
    fc_list   = []
    valid_idx = []

    for i, m in enumerate(models):
        acts = load_activations(m["model_dir"], dataset_name)
        if acts is None:
            fc_list.append(None)
            continue
        fc_list.append(compute_fc_matrix(acts))
        valid_idx.append(i)

    if not valid_idx:
        return None, fc_list, valid_idx

    max_edges = max(
        len(np.triu_indices(fc_list[i].shape[0], k=1)[0])
        for i in valid_idx
    )
    edge_matrix = np.array([
        fc_to_edge_vector(fc_list[i], max_edges)
        for i in valid_idx
    ])
    return edge_matrix, fc_list, valid_idx

# ── CPM edge selection ─────────────────────────────────────────────────────────

def correlate_edges_with_pvalues(X, y):
    """
    Pearson r and p for each edge vs behavior.
    Returns r_arr, p_arr both shape (n_edges,).
    """
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
    """Select edges by significance. Returns pos_mask, neg_mask."""
    pos_mask = (r_arr > 0) & (p_arr < threshold)
    neg_mask = (r_arr < 0) & (p_arr < threshold)
    return pos_mask, neg_mask

def compute_cpm_index(edge_vec, pos_mask, neg_mask):
    """NI = Σ_pos - Σ_neg over selected edges."""
    sigma_pos = float(np.sum(edge_vec[pos_mask]))
    sigma_neg = float(np.sum(edge_vec[neg_mask]))
    NI        = sigma_pos - sigma_neg
    return sigma_pos, sigma_neg, NI

# ── LOOCV ──────────────────────────────────────────────────────────────────────

def run_loocv(X, y, threshold):
    """
    LOOCV with CPM edge selection per fold.
    Returns sigma_pos, sigma_neg, NI arrays and edge selection log.
    """
    n                  = X.shape[0]
    sigma_pos          = np.zeros(n)
    sigma_neg          = np.zeros(n)
    NI                 = np.zeros(n)
    edge_selection_log = []

    print(f"  LOOCV ({n} folds, p < {threshold}) ...")

    for i in range(n):
        train_idx    = np.delete(np.arange(n), i)
        r_arr, p_arr = correlate_edges_with_pvalues(
            X[train_idx], y[train_idx]
        )
        pos_mask, neg_mask = select_edges(r_arr, p_arr, threshold)

        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        if n_pos + n_neg == 0:
            print(f"    ⚠ Fold {i}: no edges selected "
                  f"(N={len(train_idx)} training subjects "
                  f"may be too small for p<{threshold})")

        edge_selection_log.append({
            "fold"           : i,
            "n_edges_total"  : X.shape[1],
            "n_pos_selected" : n_pos,
            "n_neg_selected" : n_neg,
            "pct_pos"        : round(100 * n_pos / X.shape[1], 2),
            "pct_neg"        : round(100 * n_neg / X.shape[1], 2),
        })

        sigma_pos[i], sigma_neg[i], NI[i] = compute_cpm_index(
            X[i], pos_mask, neg_mask
        )

    mean_pos = np.mean([s["n_pos_selected"] for s in edge_selection_log])
    mean_neg = np.mean([s["n_neg_selected"] for s in edge_selection_log])
    print(f"  Mean edges selected: "
          f"pos={mean_pos:.1f}  neg={mean_neg:.1f} "
          f"/ {X.shape[1]} total\n")

    return sigma_pos, sigma_neg, NI, edge_selection_log

# ── Linear model ───────────────────────────────────────────────────────────────

def fit_linear(x, y, label="NI"):
    if x.std() < 1e-10:
        print(f"  [{label}] constant — skipping")
        return {"r": 0.0, "p": 1.0, "R2": 0.0,
                "slope": 0.0, "intercept": float(y.mean())}
    r, p = stats.pearsonr(x, y)
    lm   = LinearRegression().fit(x.reshape(-1,1), y)
    R2   = lm.score(x.reshape(-1,1), y)
    sig  = "*" if p < 0.05 else ("†" if p < 0.10 else "")
    print(f"  [{label}] r={r:+.4f}  p={p:.4e}  R²={R2:.4f} {sig}")
    return {
        "r"        : float(r),
        "p"        : float(p),
        "R2"       : float(R2),
        "slope"    : float(lm.coef_[0]),
        "intercept": float(lm.intercept_),
    }

# ── Helper ─────────────────────────────────────────────────────────────────────

def min_r_for_p(n, p_threshold):
    """Minimum |r| needed to reach p < threshold with n subjects."""
    t_crit = stats.t.ppf(1 - p_threshold/2, df=n-2)
    return float(t_crit / np.sqrt(t_crit**2 + n - 2))

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    models = load_model_list(CONFIG["results_dir"])
    if len(models) < 3:
        print("Need at least 3 models for LOOCV. Exiting.")
        return

    y_all = np.array([m["rapm_score"] for m in models], dtype=float)

    print(f"Behavioral criterion : RAPM score")
    print(f"Edge selection       : p < {CONFIG['p_threshold']}")
    print(f"Min |r| for p<{CONFIG['p_threshold']} "
          f"(N-1={len(models)-1} training): "
          f"{min_r_for_p(len(models)-1, CONFIG['p_threshold']):.3f}\n")

    summary_rows = []

    for ds_name in CONFIG["connectome_ds"]:
        print(f"{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        edge_matrix, fc_list, valid_idx = build_edge_matrix(
            models, ds_name
        )

        if edge_matrix is None or len(valid_idx) < 3:
            print(f"  Insufficient data ({len(valid_idx)} models) "
                  f"— skipping.\n")
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

        # Predictive validity
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

        np.save(f"{prefix}_NI.npy",        NI)
        np.save(f"{prefix}_sigma_pos.npy",  sp)
        np.save(f"{prefix}_sigma_neg.npy",  sn)
        np.save(f"{prefix}_y_rapm.npy",     y)

        # FC matrices — padded to (n_models, max_nl, max_nl)
        valid_fcs = [fc_list[i] for i in valid_idx]
        max_nl    = max(fc.shape[0] for fc in valid_fcs)
        fc_arr    = np.zeros(
            (len(valid_fcs), max_nl, max_nl), dtype=np.float32
        )
        for k, fc in enumerate(valid_fcs):
            nl = fc.shape[0]
            fc_arr[k, :nl, :nl] = fc
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
        df_sel = pd.DataFrame(sel_log)
        df_sel["model_name"] = [names[r["fold"]] for r in sel_log]
        df_sel.to_csv(f"{prefix}_edge_selection.csv", index=False)

        # Sensitivity
        pd.DataFrame(sens_rows).to_csv(
            f"{prefix}_sensitivity.csv", index=False
        )

        print(f"\n  Saved to {prefix}_*.npy / *.csv\n")

        summary_rows.append({
            "dataset"   : ds_name,
            "n_models"  : len(valid_models),
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

    # ── Comparison summary ─────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows).sort_values(
        "r_NI", key=abs, ascending=False
    )
    summary_path = os.path.join(
        CONFIG["output_dir"], "summary_all_datasets.csv"
    )
    df_summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("PREDICTIVE VALIDITY COMPARISON")
    print(f"(sorted by |r_NI|, edge selection p<{CONFIG['p_threshold']})")
    print(f"{'='*60}")
    print(f"{'Dataset':<14} {'N':>4} {'r_NI':>8} "
          f"{'p_NI':>10} {'R2_NI':>8}")
    print("-" * 50)
    for _, row in df_summary.iterrows():
        if row["status"] != "ok":
            print(f"  {row['dataset']:<12}  insufficient data")
            continue
        sig = "*" if row["p_NI"] < 0.05 else \
              "†" if row["p_NI"] < 0.10 else ""
        print(
            f"  {row['dataset']:<12} {int(row['n_models']):>4} "
            f"{row['r_NI']:>+8.4f} "
            f"{row['p_NI']:>10.4e} "
            f"{row['R2_NI']:>8.4f} {sig}"
        )

    # H1/H2/H3 checks
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
        text_ds   = {"triviaqa", "gsm8k", "math500", "mmlu"}
        vis_best  = valid[valid["dataset"].isin(visual_ds)]
        txt_best  = valid[valid["dataset"].isin(text_ds)]
        if len(vis_best) > 0 and len(txt_best) > 0:
            r_vis = abs(vis_best.iloc[0]["r_NI"])
            r_txt = abs(txt_best.iloc[0]["r_NI"])
            if r_vis > r_txt:
                print(f"  ✓ H2 SUPPORTED: visual connectomes "
                      f"(best: {vis_best.iloc[0]['dataset']}) "
                      f"outperform text connectomes")
            else:
                print(f"  H2 not supported: text connectomes "
                      f"(best: {txt_best.iloc[0]['dataset']}) "
                      f"outperform visual connectomes")

    print(f"\n  Sign convention:")
    print(f"    r_NI > 0 : higher NI → higher RAPM score")
    print(f"    r_NI < 0 : higher NI → lower RAPM score")
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()