"""Evaluation helpers: ROC/PR/AIC/BIC and paper-style model comparisons."""
import numpy as np
import pandas as pd

from .jeffreys import prepare_graph_ut, extract_true_and_median_info


def _roc_pr_auc_from_scores(y_true, scores):
    """ROC AUC and PR AUC without sklearn (rank-based)."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    m = y_true.size
    n_pos = int(y_true.sum()); n_neg = m - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan

    order = np.argsort(scores); sorted_s = scores[order]
    ranks = np.zeros(m, dtype=float)
    i = 0
    while i < m:
        j = i + 1
        while j < m and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    sum_ranks_pos = ranks[y_true == 1].sum()
    roc_auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    order_desc = np.argsort(-scores)
    y_sorted = y_true[order_desc]
    tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    pr_auc, recall_prev = 0.0, 0.0
    for r, p in zip(recall, precision):
        pr_auc += (r - recall_prev) * p
        recall_prev = r
    return float(roc_auc), float(pr_auc)


def compute_point_roc_pr_aic_bic(df_raw, adj_matrix, g, gamma,
                                 feature_col="fitness", region_col="region",
                                 k_params=2, aicbic_mode="upper"):
    """ROC AUC, PR AUC, AIC, BIC at a single (g, gamma) point."""
    G_ut, R_ut, ti, tj, n = prepare_graph_ut(df_raw, feature_col, region_col)
    A = np.asarray(adj_matrix, float)
    if A.shape != (n, n):
        raise ValueError(f"adj_matrix shape {A.shape} != expected {(n, n)}")
    Y_ut = A[ti, tj].astype(int)

    Z = g * G_ut * np.exp(gamma * R_ut)
    P = np.clip(Z / (1.0 + Z), 1e-15, 1.0 - 1e-15)
    loglik_ut = float((Y_ut * np.log(P) + (1 - Y_ut) * np.log(1 - P)).sum())
    roc_auc, pr_auc = _roc_pr_auc_from_scores(Y_ut, P)

    m_ut = Y_ut.size
    if aicbic_mode == "upper":
        loglik_ic, n_ic = loglik_ut, m_ut
    elif aicbic_mode == "full_symmetric":
        loglik_ic, n_ic = 2.0 * loglik_ut, 2 * m_ut
    else:
        raise ValueError("aicbic_mode must be 'upper' or 'full_symmetric'")

    return {
        "g": float(g), "gamma": float(gamma),
        "loglik_ut": loglik_ut, "loglik_ic": loglik_ic,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "AIC": 2 * k_params - 2.0 * loglik_ic,
        "BIC": k_params * np.log(n_ic) - 2.0 * loglik_ic,
        "pred_links": float(P.sum()),
        "L_total_obs_ut": float(Y_ut.sum()),
        "n_edges_ut": m_ut, "n_obs_ic": n_ic,
        "k_params": k_params, "aicbic_mode": aicbic_mode,
    }


def compute_metrics_true_and_median(df_raw, adj_matrix,
                                    df_uniform, highlights,
                                    feature_col="fitness", region_col="region",
                                    k_true=2, k_median=1,
                                    aicbic_mode="upper"):
    true_info, median_info = extract_true_and_median_info(highlights)
    if true_info is None:
        raise ValueError("'True Params' missing in highlights")
    if median_info is None:
        raise ValueError("'Median Entropy' missing in highlights")

    common = dict(df_raw=df_raw, adj_matrix=adj_matrix,
                  feature_col=feature_col, region_col=region_col,
                  aicbic_mode=aicbic_mode)
    return {
        "True Params": compute_point_roc_pr_aic_bic(
            g=true_info["g"], gamma=true_info["gamma"],
            k_params=k_true, **common),
        "Median Entropy": compute_point_roc_pr_aic_bic(
            g=median_info["g"], gamma=median_info["gamma"],
            k_params=k_median, **common),
    }


# ---------------------------------------------------------------
# Paper-style model comparison helpers
# ---------------------------------------------------------------

def fit_block_agnostic_fm(df_raw, total_links=None, adj_matrix=None,
                          feature_col="fitness", mode="full",
                          tol=1e-12, max_iter=500):
    """Fit the block-agnostic Fitness Model parameter ``g``.

    The probability is ``p_ij = g x_i x_j / (1 + g x_i x_j)``.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        Node table with a ``feature_col`` column.
    total_links : float, optional
        Target number of links. If omitted, ``adj_matrix`` is required and the
        target is computed from it.
    adj_matrix : array-like, optional
        Observed adjacency matrix used to infer ``total_links`` when needed.
    mode : {"full", "upper"}
        ``"full"`` uses all off-diagonal ordered pairs and reproduces the
        legacy paper notebooks for directed trade matrices. ``"upper"`` uses
        unordered dyads ``i < j``.
    """
    x = df_raw[feature_col].to_numpy(float)
    n = len(x)
    G = np.outer(x, x)
    if mode == "full":
        mask = ~np.eye(n, dtype=bool)
        G_flat = G[mask]
        if total_links is None:
            if adj_matrix is None:
                raise ValueError("Either total_links or adj_matrix must be provided")
            A = np.asarray(adj_matrix, float)
            if A.shape != (n, n):
                raise ValueError(f"adj_matrix shape {A.shape} != expected {(n, n)}")
            total_links = float(A[mask].sum())
    elif mode == "upper":
        ti, tj = np.triu_indices(n, k=1)
        G_flat = G[ti, tj]
        if total_links is None:
            if adj_matrix is None:
                raise ValueError("Either total_links or adj_matrix must be provided")
            A = np.asarray(adj_matrix, float)
            if A.shape != (n, n):
                raise ValueError(f"adj_matrix shape {A.shape} != expected {(n, n)}")
            total_links = float(A[ti, tj].sum())
    else:
        raise ValueError("mode must be 'full' or 'upper'")

    target = float(total_links)
    if target <= 0:
        return 0.0

    def expected_links(g):
        Z = g * G_flat
        return float((Z / (1.0 + Z)).sum())

    lo, hi = 0.0, 1.0
    while expected_links(hi) < target:
        hi *= 2.0
        if hi > 1e300:
            raise RuntimeError("Could not bracket the FM solution for g")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if expected_links(mid) < target:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) <= tol * max(1.0, hi):
            break
    return float(0.5 * (lo + hi))


def probability_matrix_from_g_gamma(df_raw, g, gamma=0.0,
                                    feature_col="fitness",
                                    region_col="region",
                                    id_col="country_iso3"):
    """Return a symmetric probability matrix for ``p_ij(g, gamma)``."""
    x = df_raw[feature_col].to_numpy(float)
    ids = df_raw[id_col].to_numpy() if id_col in df_raw.columns else np.arange(len(x))
    reg = df_raw[region_col].astype(str).to_numpy()
    G = np.outer(x, x)
    R = (reg[:, None] == reg[None, :]).astype(float)
    Z = float(g) * G * np.exp(float(gamma) * R)
    P = Z / (1.0 + Z)
    np.fill_diagonal(P, 0.0)
    return pd.DataFrame(P, index=ids, columns=ids)


def paper_model_comparison_table(df_raw, adj_matrix, *,
                                 g_fm, g_median, gamma_median,
                                 g_true, gamma_true,
                                 feature_col="fitness",
                                 region_col="region",
                                 id_col="country_iso3"):
    """Comparison table matching the original BACI/UN notebooks.

    ROC AUC and PR AUC are computed by flattening the full matrix, as in the
    exploratory notebooks. AIC/BIC are computed on unordered dyads ``i < j``
    by default, matching the paper convention ``N = n * (n - 1) / 2``.
    """
    from .two_param import compare_matrices, compute_aic_bic

    specs = [
        ("Block-Agnostic FM", 1, float(g_fm), 0.0),
        ("Jeffreys Prior & Median Entropy", 1, float(g_median), float(gamma_median)),
        ("FCBM Planted Partition", 2, float(g_true), float(gamma_true)),
    ]

    rows = []
    for name, k, g, gamma in specs:
        P = probability_matrix_from_g_gamma(
            df_raw, g, gamma,
            feature_col=feature_col, region_col=region_col, id_col=id_col,
        )
        metrics = compare_matrices(adj_matrix, P, plot=False)
        aic, bic = compute_aic_bic(adj_matrix, P, k=k, mode="upper")
        rows.append({
            "Model": name,
            "k": k,
            "beta": float(np.log(g)) if g > 0 else -np.inf,
            "g": g,
            "gamma": gamma,
            "ROC AUC": float(metrics["ROC AUC"]),
            "PR AUC": float(metrics["PR AUC"]),
            "AIC": float(aic),
            "BIC": float(bic),
        })
    return pd.DataFrame(rows)

