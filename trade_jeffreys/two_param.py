"""Two-parameter (a, b) <=> (g, alpha, gamma) model + classification metrics."""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)


def fit_ab_bisection(country_df, L_same_obs, L_diff_obs,
                     feature_col="fitness", region_col="region",
                     epsilon=1e-6, max_iter=1000, verbose=True):
    """Fit a (same-region) and b (diff-region) so undirected link sums match."""
    L_same_obs = L_same_obs / 2  # upper triangle only
    L_diff_obs = L_diff_obs / 2

    fitness = country_df[feature_col].to_numpy()
    region = country_df[region_col].to_numpy()
    n = len(fitness)
    pairs = list(itertools.combinations(range(n), 2))

    def expected_links(param, want_same):
        total = 0.0
        for i, j in pairs:
            same = region[i] == region[j]
            if want_same != same:
                continue
            x = param * fitness[i] * fitness[j]
            total += x / (1 + x)
        return total

    def bisect(L_obs, want_same):
        lo, hi = 1e-12, 1e12
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            L_model = expected_links(mid, want_same)
            if abs(L_model - L_obs) < epsilon or (hi - lo) < epsilon:
                return mid
            if L_model > L_obs:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    a = bisect(L_same_obs, True)
    b = bisect(L_diff_obs, False)
    g, alpha = b, a / b
    if verbose:
        print(f"a = {a:.6f}\nb = {b:.6f}\ng = {g:.6f}\nalpha = {alpha:.6f}")
    return a, b, g, alpha


def reconstruct_probability_matrix(country_df, g, alpha,
                                   feature_col="fitness",
                                   region_col="region",
                                   id_col="country_iso3"):
    """Symmetric link-probability matrix from (g, alpha) and fitness."""
    fitness = country_df[feature_col].to_numpy()
    regions = country_df[region_col].to_numpy()
    ids = country_df[id_col].to_numpy()

    same = regions[:, None] == regions[None, :]
    F = np.outer(fitness, fitness)
    X = g * F * np.where(same, alpha, 1.0)
    P = X / (1.0 + X)
    np.fill_diagonal(P, 0.0)
    return pd.DataFrame(P, index=ids, columns=ids)


def compare_matrices(y_true, y_pred, threshold=0.3, plot=True):
    """Compare true binary matrix vs predicted probability matrix."""
    y_true = (y_true.values if hasattr(y_true, "values") else y_true).flatten()
    y_pred = (y_pred.values if hasattr(y_pred, "values") else y_pred).flatten()
    if not np.any((y_pred > 0) & (y_pred < 1)):
        raise ValueError("Predicted values must be probabilities in (0, 1).")

    y_lab = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_lab).ravel()
    metrics = {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": precision_score(y_true, y_lab, zero_division=0),
        "Recall":    recall_score(y_true, y_lab, zero_division=0),
        "F1 Score":  f1_score(y_true, y_lab, zero_division=0),
        "ROC AUC":   roc_auc_score(y_true, y_pred),
        "PR AUC":    average_precision_score(y_true, y_pred),
    }

    if plot:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        precs, recs, _ = precision_recall_curve(y_true, y_pred)
        baseline = float(np.mean(y_true))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {metrics['ROC AUC']:.2f}")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(recs, precs, lw=2, color="green",
                 label=f"PR AUC = {metrics['PR AUC']:.2f}")
        plt.hlines(baseline, 0, 1, color="red", linestyle="--",
                   label=f"Random Precision = {baseline:.2f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curve"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.show()

    return metrics


def compute_aic_bic(A, P, k=1, mode="upper"):
    """Compute AIC/BIC for binary network reconstruction.

    The paper treats the reconstructed networks as undirected, so the default
    uses only unordered dyads ``i < j``. With this convention the number of
    observations in BIC is ``n * (n - 1) / 2``, matching Table 3 in the paper.

    Parameters
    ----------
    A : array-like or pandas.DataFrame
        Observed binary adjacency matrix.
    P : array-like or pandas.DataFrame
        Predicted probability matrix.
    k : int
        Number of fitted/effective parameters.
    mode : {"upper", "full_offdiag"}
        ``"upper"`` uses unordered dyads ``i < j`` and is the paper default.
        ``"full_offdiag"`` uses all ordered off-diagonal entries and is kept
        only for legacy directed-matrix comparisons.
    """
    A = A.values if hasattr(A, "values") else np.asarray(A)
    P = P.values if hasattr(P, "values") else np.asarray(P)
    if A.shape != P.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and P must be square matrices with the same shape")

    eps = 1e-10
    P_clip = np.clip(P, eps, 1 - eps)
    n = A.shape[0]

    if mode == "upper":
        i_idx, j_idx = np.triu_indices(n, k=1)
        y = A[i_idx, j_idx]
        p = P_clip[i_idx, j_idx]
        N = int(y.size)
    elif mode == "full_offdiag":
        mask = ~np.eye(n, dtype=bool)
        y = A[mask]
        p = P_clip[mask]
        N = int(y.size)
    else:
        raise ValueError("mode must be 'upper' or 'full_offdiag'")

    logL = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    return 2 * k - 2 * logL, k * np.log(N) - 2 * logL
