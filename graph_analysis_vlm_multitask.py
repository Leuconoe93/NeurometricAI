# scripts/graph_analysis_vlm_multitask.py
#
# Phase 4 — Graph fingerprint analysis on PCA-enriched connectomes.
# Reads FC matrices from connectivity_pca/ output.
#
# For each dataset × model:
#   - Load FC matrix (n_layers×5, n_layers×5)
#   - Split into positive/negative subgraphs
#   - Compute 20 graph metrics per subgraph
#   - Fit ER/WS/BA models + AIC/BIC
#
# Output:
#   results/multitask/graph_pca/
#       fingerprints_<dataset>.csv
#       model_selection_<dataset>.csv
#       fingerprints_all.csv
#       model_selection_all.csv
#       model_selection_summary.csv

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("Warning: python-louvain not installed. "
          "Modularity metrics will be 0.")

# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = {
    "connectivity_dir" : "results/multitask/connectivity_pca/",
    "output_dir"       : "results/multitask/graph_pca/",
    "binary_threshold" : 0.1,
    "n_random_graphs"  : 100,
    "n_synthetic"      : 500,
    "seed"             : 42,
    "connectome_ds"    : [
        "triviaqa", "gsm8k", "math500", "mmlu",
        "mathvista", "scienceqa", "rest",
    ],
}

# ── Graph construction ─────────────────────────────────────────────────────────

def split_subgraphs(fc, threshold):
    A_pos = np.where(fc >  threshold,  fc, 0.0)
    A_neg = np.where(fc < -threshold, -fc, 0.0)
    return A_pos, A_neg

def to_graph(A, threshold=0.0):
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] > threshold:
                G.add_edge(i, j, weight=float(A[i, j]))
    return G

def unpad_fc(fc_row):
    """Remove zero-padding — detect actual node count."""
    nonzero = np.any(fc_row != 0, axis=1)
    n = int(nonzero.sum())
    if n < 2:
        return fc_row
    return fc_row[:n, :n]

# ── Graph metrics ──────────────────────────────────────────────────────────────

def safe(val, default=0.0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)

def compute_clustering(G):
    return safe(nx.average_clustering(G, weight="weight")
                if G.number_of_edges() > 0 else 0.0)

def compute_transitivity(G):
    return safe(nx.transitivity(G))

def compute_path_length(G):
    if G.number_of_edges() == 0:
        return -1.0
    if nx.is_connected(G):
        try:
            return safe(nx.average_shortest_path_length(G, weight=None))
        except Exception:
            return -1.0
    largest = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest)
    try:
        return safe(nx.average_shortest_path_length(H, weight=None))
    except Exception:
        return -1.0

def compute_efficiency(G):
    return (safe(nx.global_efficiency(G)),
            safe(nx.local_efficiency(G)))

def compute_small_world(G, n_random, seed):
    n = G.number_of_nodes()
    if G.number_of_edges() < 2 or n < 4:
        return 0.0, 0.0
    C = compute_clustering(G)
    L = compute_path_length(G)
    if L <= 0:
        return 0.0, 0.0
    rng     = np.random.default_rng(seed)
    deg_seq = [d for _, d in G.degree()]
    C_rand_list, L_rand_list = [], []
    for _ in range(n_random):
        try:
            R = nx.configuration_model(
                deg_seq, seed=int(rng.integers(1e6))
            )
            R = nx.Graph(R)
            R.remove_edges_from(nx.selfloop_edges(R))
            C_rand_list.append(nx.average_clustering(R))
            if nx.is_connected(R):
                L_rand_list.append(
                    nx.average_shortest_path_length(R)
                )
        except Exception:
            continue
    if not C_rand_list:
        return 0.0, 0.0
    C_rand = np.mean(C_rand_list)
    L_rand = np.mean(L_rand_list) if L_rand_list else L
    sigma  = (C / max(C_rand, 1e-10)) / (L / max(L_rand, 1e-10))
    omega  = (L_rand / max(L, 1e-10)) - (C / 0.75)
    return safe(sigma), safe(omega)

def compute_degree_stats(G):
    degs = [d for _, d in G.degree()]
    if not degs:
        return 0.0, 0.0, 0
    return float(np.mean(degs)), float(np.std(degs)), int(np.max(degs))

def compute_assortativity(G):
    if G.number_of_edges() < 2:
        return 0.0
    try:
        return safe(nx.degree_assortativity_coefficient(G))
    except Exception:
        return 0.0

def compute_modularity(G, seed):
    if not HAS_LOUVAIN or G.number_of_edges() == 0:
        return 0.0, 1
    try:
        part = community_louvain.best_partition(G, random_state=seed)
        Q    = community_louvain.modularity(part, G)
        return safe(Q), len(set(part.values()))
    except Exception:
        return 0.0, 1

def compute_participation(G, seed):
    if not HAS_LOUVAIN or G.number_of_edges() == 0:
        return 0.0
    try:
        part  = community_louvain.best_partition(G, random_state=seed)
        n_c   = max(part.values()) + 1
        pc    = []
        for node in G.nodes():
            k_i = G.degree(node)
            if k_i == 0:
                pc.append(0.0)
                continue
            k_is = np.zeros(n_c)
            for nb in G.neighbors(node):
                k_is[part[nb]] += 1
            pc.append(1.0 - np.sum((k_is / k_i) ** 2))
        return safe(np.mean(pc))
    except Exception:
        return 0.0

def compute_betweenness(G):
    if G.number_of_edges() == 0:
        return 0.0, 0.0
    bc   = nx.betweenness_centrality(G, normalized=True)
    vals = list(bc.values())
    return safe(max(vals)), safe(np.std(vals))

def compute_rich_club(G):
    if G.number_of_edges() < 2:
        return 0.0
    try:
        k  = max(1, int(round(np.mean([d for _, d in G.degree()]))))
        rc = nx.rich_club_coefficient(G, normalized=False)
        return safe(rc.get(k, 0.0))
    except Exception:
        return 0.0

def compute_spectral(G):
    if G.number_of_nodes() < 2:
        return 0.0, 0.0, 0.0
    try:
        A    = nx.to_numpy_array(G)
        eigs = sorted(np.linalg.eigvalsh(A), reverse=True)
        lam1 = float(eigs[0])
        gap  = float(eigs[0] - eigs[1]) if len(eigs) > 1 else 0.0
        L    = nx.laplacian_matrix(G).toarray().astype(float)
        leig = sorted(np.linalg.eigvalsh(L))
        fied = float(leig[1]) if len(leig) > 1 else 0.0
        return lam1, gap, fied
    except Exception:
        return 0.0, 0.0, 0.0

def compute_fingerprint(A, label, config):
    thr  = config["binary_threshold"]
    seed = config["seed"]
    n_r  = config["n_random_graphs"]
    pfx  = f"{label}_"

    G_w = to_graph(A, threshold=0.0)
    G_b = to_graph(A, threshold=thr)

    m = {}
    m[pfx+"clustering"]             = compute_clustering(G_w)
    m[pfx+"transitivity"]           = compute_transitivity(G_b)
    L                                = compute_path_length(G_b)
    m[pfx+"avg_path_length"]        = L
    g_eff, l_eff                     = compute_efficiency(G_b)
    m[pfx+"global_efficiency"]      = g_eff
    m[pfx+"local_efficiency"]       = l_eff
    sigma, omega                     = compute_small_world(G_b, n_r, seed)
    m[pfx+"small_world_sigma"]      = sigma
    m[pfx+"small_world_omega"]      = omega
    d_mean, d_std, d_max             = compute_degree_stats(G_b)
    m[pfx+"degree_mean"]            = d_mean
    m[pfx+"degree_std"]             = d_std
    m[pfx+"degree_max"]             = d_max
    m[pfx+"assortativity"]          = compute_assortativity(G_b)
    Q, n_comm                        = compute_modularity(G_b, seed)
    m[pfx+"modularity_Q"]           = Q
    m[pfx+"n_communities"]          = n_comm
    m[pfx+"participation_coeff"]    = compute_participation(G_b, seed)
    bc_max, bc_std                   = compute_betweenness(G_b)
    m[pfx+"betweenness_max"]        = bc_max
    m[pfx+"betweenness_std"]        = bc_std
    m[pfx+"rich_club"]              = compute_rich_club(G_b)
    lam1, gap, fied                  = compute_spectral(G_w)
    m[pfx+"spectral_radius"]        = lam1
    m[pfx+"spectral_gap"]           = gap
    m[pfx+"algebraic_connectivity"] = fied
    return m

# ── Graph model fitting ────────────────────────────────────────────────────────

def er_ll(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    max_e = n*(n-1)/2
    if max_e == 0:
        return -np.inf, 1
    p = np.clip(m/max_e, 1e-10, 1-1e-10)
    return float(m*np.log(p) + (max_e-m)*np.log(1-p)), 1

def ws_ll(G, n_syn, seed):
    s = {"n": G.number_of_nodes(),
         "C": nx.average_clustering(G) if G.number_of_edges()>0 else 0,
         "L": compute_path_length(G),
         "k": np.mean([d for _,d in G.degree()]) if G.number_of_nodes()>0 else 0}
    if s["n"] < 4:
        return -np.inf, 2
    rng   = np.random.default_rng(seed)
    best  = -np.inf
    for k in [max(2, int(round(s["k"]*f))) for f in [0.75,1.0,1.25]]:
        for beta in [0.05, 0.1, 0.2, 0.5]:
            if k >= s["n"]:
                continue
            Cs, Ls = [], []
            for _ in range(max(10, n_syn//12)):
                try:
                    R = nx.watts_strogatz_graph(
                        s["n"], k, beta, seed=int(rng.integers(1e6))
                    )
                    Cs.append(nx.average_clustering(R))
                    if nx.is_connected(R):
                        Ls.append(nx.average_shortest_path_length(R))
                except Exception:
                    continue
            if not Cs:
                continue
            ll = (stats.norm.logpdf(
                      s["C"], np.mean(Cs), max(np.std(Cs), 0.01)) +
                  stats.norm.logpdf(
                      s["L"], np.mean(Ls) if Ls else s["L"],
                      max(np.std(Ls), 0.1) if Ls else 1.0))
            if ll > best:
                best = ll
    return float(best), 2

def ba_ll(G, n_syn, seed):
    s   = {"n": G.number_of_nodes(),
           "k": np.mean([d for _,d in G.degree()]) if G.number_of_nodes()>0 else 0}
    if s["n"] < 4:
        return -np.inf, 1
    degs_obs = sorted([d for _,d in G.degree()], reverse=True)
    rng      = np.random.default_rng(seed)
    best     = -np.inf
    for m in range(1, max(2, int(s["k"]))+1):
        dlists = []
        for _ in range(max(10, n_syn//max(1,int(s["k"])))):
            try:
                R = nx.barabasi_albert_graph(
                    s["n"], m, seed=int(rng.integers(1e6))
                )
                dlists.append(
                    sorted([d for _,d in R.degree()], reverse=True)
                )
            except Exception:
                continue
        if not dlists:
            continue
        arr  = np.array(dlists, dtype=float)
        mn   = arr.mean(axis=0)
        sd   = np.maximum(arr.std(axis=0), 0.5)
        ml   = min(len(degs_obs), len(mn))
        ll   = float(np.sum(
            stats.norm.logpdf(degs_obs[:ml], mn[:ml], sd[:ml])
        ))
        if ll > best:
            best = ll
    return float(best), 1

def fit_graph_models(G, n_syn, seed):
    m = G.number_of_edges()
    n = G.number_of_nodes()
    if m == 0:
        return {"best_BIC": "empty", "best_AIC": "empty"}

    ll_er, k_er = er_ll(G)
    ll_ws, k_ws = ws_ll(G, n_syn, seed)
    ll_ba, k_ba = ba_ll(G, n_syn, seed)

    def aic(ll, k):    return 2*k - 2*ll
    def bic(ll, k, n): return k*np.log(max(n,2)) - 2*ll

    res = {}
    for name, ll, k in [("ER",ll_er,k_er),
                         ("WS",ll_ws,k_ws),
                         ("BA",ll_ba,k_ba)]:
        res[f"ll_{name}"]  = round(ll, 4)
        res[f"AIC_{name}"] = round(aic(ll, k), 4)
        res[f"BIC_{name}"] = round(bic(ll, k, m), 4)

    res["best_AIC"] = min(["ER","WS","BA"],
                          key=lambda x: res[f"AIC_{x}"])
    res["best_BIC"] = min(["ER","WS","BA"],
                          key=lambda x: res[f"BIC_{x}"])

    lrt_ws = 2*(ll_ws - ll_er)
    res["LRT_WS_vs_ER"] = round(lrt_ws, 4)
    res["p_WS_vs_ER"]   = round(
        1 - stats.chi2.cdf(lrt_ws, df=k_ws-k_er), 4
    )
    return res

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    all_fp_rows = []
    all_ms_rows = []
    ws_wins     = defaultdict(int)
    ws_total    = defaultdict(int)

    for ds_name in CONFIG["connectome_ds"]:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        fc_path  = os.path.join(
            CONFIG["connectivity_dir"], f"{ds_name}_fc_matrices.npy"
        )
        idx_path = os.path.join(
            CONFIG["connectivity_dir"], f"{ds_name}_indices.csv"
        )

        if not os.path.exists(fc_path):
            print(f"  FC matrices not found — skipping.")
            continue

        fc_arr = np.load(fc_path)
        idx_df = pd.read_csv(idx_path)
        names  = idx_df["model_name"].tolist()
        scores = idx_df["rapm_score"].tolist()

        print(f"  Models: {len(names)} | FC shape: {fc_arr.shape}")

        for i, (name, score) in enumerate(zip(names, scores)):
            fc = unpad_fc(fc_arr[i])
            n_nodes = fc.shape[0]
            if n_nodes < 2:
                print(f"  {name}: too few nodes — skipping")
                continue

            A_pos, A_neg = split_subgraphs(fc, CONFIG["binary_threshold"])

            fp_pos = compute_fingerprint(A_pos, "pos", CONFIG)
            fp_neg = compute_fingerprint(A_neg, "neg", CONFIG)

            fp_row = {
                "dataset"    : ds_name,
                "model_name" : name,
                "rapm_score" : score,
                "n_nodes"    : n_nodes,
            }
            fp_row.update(fp_pos)
            fp_row.update(fp_neg)
            all_fp_rows.append(fp_row)

            G_pos = to_graph(A_pos, threshold=0.0)
            G_neg = to_graph(A_neg, threshold=0.0)

            fits_pos = fit_graph_models(
                G_pos, CONFIG["n_synthetic"], CONFIG["seed"]
            )
            fits_neg = fit_graph_models(
                G_neg, CONFIG["n_synthetic"], CONFIG["seed"]
            )

            ms_row = {
                "dataset"    : ds_name,
                "model_name" : name,
                "rapm_score" : score,
            }
            for k, v in fits_pos.items():
                ms_row[f"pos_{k}"] = v
            for k, v in fits_neg.items():
                ms_row[f"neg_{k}"] = v
            all_ms_rows.append(ms_row)

            ws_total[ds_name] += 1
            if fits_pos.get("best_BIC") == "WS":
                ws_wins[ds_name] += 1

            print(
                f"  {name:<25} "
                f"nodes={n_nodes:>4} "
                f"pos:{fits_pos.get('best_BIC','?'):>3} "
                f"neg:{fits_neg.get('best_BIC','?'):>3} | "
                f"σ={fp_pos['pos_small_world_sigma']:.2f} "
                f"Q={fp_pos['pos_modularity_Q']:.2f}"
            )

        # Save per-dataset files
        ds_fp = pd.DataFrame([r for r in all_fp_rows
                              if r["dataset"] == ds_name])
        ds_ms = pd.DataFrame([r for r in all_ms_rows
                              if r["dataset"] == ds_name])

        if len(ds_fp) > 0:
            ds_fp.to_csv(os.path.join(
                CONFIG["output_dir"], f"fingerprints_{ds_name}.csv"
            ), index=False)
        if len(ds_ms) > 0:
            ds_ms.to_csv(os.path.join(
                CONFIG["output_dir"], f"model_selection_{ds_name}.csv"
            ), index=False)

    # Save combined files
    if all_fp_rows:
        pd.DataFrame(all_fp_rows).to_csv(
            os.path.join(CONFIG["output_dir"], "fingerprints_all.csv"),
            index=False
        )
    if all_ms_rows:
        pd.DataFrame(all_ms_rows).to_csv(
            os.path.join(CONFIG["output_dir"], "model_selection_all.csv"),
            index=False
        )

    # WS win rate summary
    print(f"\n{'='*60}")
    print("GRAPH MODEL SELECTION SUMMARY")
    print("(WS win rate by dataset — positive subgraph, PCA connectomes)")
    print(f"{'='*60}")
    print(f"{'Dataset':<14} {'WS wins':>9} {'Total':>7} {'Rate':>7}")
    print("-" * 40)
    for ds in CONFIG["connectome_ds"]:
        if ws_total[ds] == 0:
            continue
        rate = ws_wins[ds] / ws_total[ds]
        print(f"  {ds:<12} {ws_wins[ds]:>9} "
              f"{ws_total[ds]:>7} {rate:>7.1%}")

    print(f"\nOutputs saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()