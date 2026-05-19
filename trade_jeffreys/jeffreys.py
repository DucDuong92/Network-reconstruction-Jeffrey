"""Jeffreys-curve pipeline: scan g, solve gamma(g), build curve, resample."""
import numpy as np
import pandas as pd


# =================================================================
# Upper-triangle construction + per-point evaluation
# =================================================================

def prepare_graph_ut(df_raw, feature_col="fitness", region_col="region"):
    """Return (G_ut, R_ut, triu_i, triu_j, n) from a country dataframe."""
    x = df_raw[feature_col].to_numpy(float)
    reg = df_raw[region_col].astype(str).to_numpy()
    n = len(x)
    same = (reg[:, None] == reg[None, :]).astype(int)
    np.fill_diagonal(same, 0)
    G = np.outer(x, x); np.fill_diagonal(G, 0.0)
    ti, tj = np.triu_indices(n, k=1)
    return G[ti, tj], same[ti, tj].astype(float), ti, tj, n


def Smin_Smax_for_g(g, G_ut, R_ut):
    R0, R1 = (R_ut == 0.0), (R_ut == 1.0)
    Z0 = g * G_ut[R0]
    Smin = float((Z0 / (1.0 + Z0)).sum())
    Smax = Smin + int(np.sum(G_ut[R1] > 0.0))
    return Smin, Smax


def F_sum_pred_links_ut(g, gamma, G_ut, R_ut):
    Z = g * G_ut * np.exp(gamma * R_ut)
    return float((Z / (1.0 + Z)).sum())


def solve_gamma_for_g(g, L_target, G_ut, R_ut,
                      gamma_bounds=(-60.0, 60.0),
                      bisection_tol=1e-9, max_bisect_it=80):
    g_lo, g_hi = gamma_bounds
    Smin, Smax = Smin_Smax_for_g(g, G_ut, R_ut)
    if L_target <= Smin + 1e-12:
        return g_lo, False
    if L_target >= Smax - 1e-12:
        return g_hi, False

    S_lo = F_sum_pred_links_ut(g, g_lo, G_ut, R_ut)
    S_hi = F_sum_pred_links_ut(g, g_hi, G_ut, R_ut)
    expand = 0
    while not (min(S_lo, S_hi) <= L_target <= max(S_lo, S_hi)) and expand < 12:
        span = g_hi - g_lo
        g_lo -= span; g_hi += span
        S_lo = F_sum_pred_links_ut(g, g_lo, G_ut, R_ut)
        S_hi = F_sum_pred_links_ut(g, g_hi, G_ut, R_ut)
        expand += 1
    if not (min(S_lo, S_hi) <= L_target <= max(S_lo, S_hi)):
        return (g_lo if abs(S_lo - L_target) <= abs(S_hi - L_target) else g_hi), False

    abs_stop = max(bisection_tol * max(L_target, 1.0), 1e-9)
    lo, hi = g_lo, g_hi
    for _ in range(max_bisect_it):
        mid = 0.5 * (lo + hi)
        err = F_sum_pred_links_ut(g, mid, G_ut, R_ut) - L_target
        if abs(err) <= abs_stop:
            return mid, True
        if err < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


def eval_point(g, gamma, L_target, G_ut, R_ut, fisher_eps=1e-12, Y_ut=None):
    """Predicted links, entropy, log-likelihood, Fisher J_gamma/J_beta."""
    Z = g * G_ut * np.exp(gamma * R_ut)
    P_raw = Z / (1.0 + Z)
    P_ent = np.clip(P_raw, 1e-15, 1.0 - 1e-15)

    pred_links = float(P_raw.sum())
    entropy = float(-(P_ent * np.log(P_ent)
                      + (1.0 - P_ent) * np.log(1.0 - P_ent)).sum())
    loglik = (None if Y_ut is None else
              float((Y_ut * np.log(P_ent)
                     + (1.0 - Y_ut) * np.log(1.0 - P_ent)).sum()))

    w = P_raw * (1.0 - P_raw)
    S0, S1, S2 = float(w.sum()), float((w * R_ut).sum()), float((w * R_ut**2).sum())
    if S0 <= fisher_eps:
        J_gamma = 0.0
    else:
        J_gamma = float(np.sqrt(max(S2 - (S1 * S1) / max(S0, fisher_eps), 0.0)))
    J_beta = 0.0 if abs(S1) <= fisher_eps else float((S0 / abs(S1)) * J_gamma)

    return {
        "pred_links": pred_links, "link_error": abs(pred_links - L_target),
        "entropy": entropy, "loglik": loglik,
        "S0": S0, "S1": S1, "S2": S2,
        "J_gamma": J_gamma, "J_beta": J_beta,
    }


# =================================================================
# Feasibility, scan, geometry, resampling
# =================================================================

def feasibility_window_for_gs(g_array, G_ut, R_ut, L_target):
    g_array = np.asarray(g_array, float)
    R0, R1 = (R_ut == 0.0), (R_ut == 1.0)
    G_R0, G_R1 = G_ut[R0], G_ut[R1]
    num_R1_pos = int(np.sum(G_R1 > 0.0))
    mask = np.zeros_like(g_array, dtype=bool)
    for k, g in enumerate(g_array):
        Z0 = g * G_R0
        Smin = float((Z0 / (1.0 + Z0)).sum())
        mask[k] = Smin <= L_target <= Smin + num_R1_pos
    return mask


def clip_g_range_to_feasible(df_raw, g_range, total_links,
                             feature_col="fitness", region_col="region"):
    G_ut, R_ut, *_ = prepare_graph_ut(df_raw, feature_col, region_col)
    g_range = np.asarray(g_range, float)
    feas = feasibility_window_for_gs(g_range, G_ut, R_ut, float(total_links))
    if not feas.any():
        raise ValueError("No feasible g found.")
    runs, start = [], None
    for i, ok in enumerate(feas):
        if ok and start is None:
            start = i
        if (not ok or i == len(feas) - 1) and start is not None:
            end = i if ok else i - 1
            runs.append((start, end))
            start = None
    s, e = max(runs, key=lambda t: t[1] - t[0])
    return g_range[s:e + 1], feas


def scan_g_curve(df, total_links, g_range,
                 feature_col="fitness", region_col="region",
                 gamma_bounds=(-60.0, 60.0),
                 bisection_tol=1e-9, max_bisect_it=80,
                 fisher_eps=1e-12, adj_matrix=None):
    G_ut, R_ut, ti, tj, n = prepare_graph_ut(df, feature_col, region_col)
    L_target = float(total_links)
    Y_ut = None
    if adj_matrix is not None:
        A = np.asarray(adj_matrix, dtype=float)
        if A.shape != (n, n):
            raise ValueError(f"adj_matrix shape {A.shape} != expected {(n, n)}")
        Y_ut = A[ti, tj].astype(float)

    rows = []
    for g in np.asarray(g_range, float):
        gamma, feasible = solve_gamma_for_g(
            g, L_target, G_ut, R_ut,
            gamma_bounds=gamma_bounds,
            bisection_tol=bisection_tol, max_bisect_it=max_bisect_it,
        )
        m = eval_point(g, gamma, L_target, G_ut, R_ut,
                       fisher_eps=fisher_eps, Y_ut=Y_ut)
        rows.append({"g": g, "beta": float(np.log(g)), "gamma": gamma,
                     "feasible": bool(feasible), **m})
    return pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)


def _add_arclength_columns(d):
    """Add seg_len/run_cum_len/cum_len based on (run_id, beta, J_beta)."""
    d = d.sort_values(["run_id", "beta"]).reset_index(drop=True)
    d["seg_len"] = 0.0
    d["run_cum_len"] = 0.0
    for _, gdf in d.groupby("run_id", sort=False):
        idx = gdf.index
        bet = gdf["beta"].to_numpy(float)
        Jb = gdf["J_beta"].to_numpy(float)
        seg = np.zeros_like(bet)
        if len(bet) >= 2:
            seg[:-1] = 0.5 * (Jb[:-1] + Jb[1:]) * np.maximum(np.diff(bet), 0.0)
        d.loc[idx, "seg_len"] = seg
        d.loc[idx, "run_cum_len"] = np.cumsum(seg)
    cum, total = np.zeros(len(d)), 0.0
    for _, gdf in d.groupby("run_id", sort=False):
        idx = gdf.index
        cum[idx] = total + d.loc[idx, "run_cum_len"].to_numpy(float)
        total += float(d.loc[idx, "seg_len"].sum())
    d["cum_len"] = cum
    return d


def build_jeffreys_curve(df_points, feasible_only=True):
    if feasible_only and "feasible" in df_points.columns:
        d = df_points[df_points["feasible"]].copy()
    else:
        d = df_points.copy()
    d = d.reset_index(drop=True)
    d["k"] = np.arange(len(d))
    if d.empty:
        return d
    k_arr = d["k"].to_numpy()
    breaks = np.where(np.diff(k_arr) > 1)[0] + 1
    rid = np.zeros(len(d), dtype=int)
    if len(breaks):
        rid[breaks] = 1
    d["run_id"] = np.cumsum(rid)
    return _add_arclength_columns(d)


def filter_and_rebuild_geometry(df_curve, max_link_error=1e-6):
    d = df_curve[pd.to_numeric(df_curve["link_error"], errors="coerce")
                 < max_link_error].copy()
    if d.empty:
        raise ValueError("No rows remain after filtering by link_error")
    if "run_id" not in d.columns:
        d["run_id"] = 0
    return _add_arclength_columns(d)


def resample_uniform_in_jeffreys(df_curve, M=200, recompute=True,
                                 df_raw=None, total_links=None,
                                 feature_col="fitness", region_col="region",
                                 gamma_bounds=(-60.0, 60.0),
                                 bisection_tol=1e-9, max_bisect_it=80,
                                 fisher_eps=1e-12, adj_matrix=None):
    if df_curve.empty:
        return df_curve.copy()
    S_tot = float(df_curve["cum_len"].iloc[-1])
    if S_tot <= 0:
        out = df_curve[["g", "beta", "gamma", "J_beta"]].copy()
        out["s"] = 0.0
        return out

    s_grid = np.linspace(0.0, S_tot, M + 1)
    pieces = []
    for rid, gdf in df_curve.groupby("run_id", sort=False):
        gdf = gdf.sort_values("run_cum_len")
        s_local = gdf["run_cum_len"].to_numpy(float)
        s_offset = float(gdf["cum_len"].iloc[0] - s_local[0])
        s_lo, s_hi = s_offset + s_local[0], s_offset + s_local[-1]
        m = (s_grid >= s_lo - 1e-15) & (s_grid <= s_hi + 1e-15)
        if not np.any(m):
            continue
        s_in = s_grid[m] - s_offset
        beta_s = np.interp(s_in, s_local, gdf["beta"].to_numpy(float))
        gamma_s = np.interp(s_in, s_local, gdf["gamma"].to_numpy(float))
        pieces.append(pd.DataFrame({
            "s": s_grid[m], "beta": beta_s, "gamma": gamma_s,
            "g": np.exp(beta_s), "run_id": rid,
        }))

    if not pieces:
        return pd.DataFrame({"s": s_grid, "beta": np.nan,
                             "gamma": np.nan, "g": np.nan})

    out = pd.concat(pieces, ignore_index=True).sort_values(["s", "run_id"]).reset_index(drop=True)
    if not recompute:
        return out
    if df_raw is None or total_links is None:
        raise ValueError("recompute=True requires df_raw and total_links")

    G_ut, R_ut, ti, tj, n = prepare_graph_ut(df_raw, feature_col, region_col)
    Y_ut = None
    if adj_matrix is not None:
        A = np.asarray(adj_matrix, float)
        if A.shape != (n, n):
            raise ValueError(f"adj_matrix shape {A.shape} != expected {(n, n)}")
        Y_ut = A[ti, tj].astype(float)

    metrics, gamma_refined = [], np.empty(len(out))
    L_target = float(total_links)
    for i, gval in enumerate(out["g"].to_numpy(float)):
        gi, _ = solve_gamma_for_g(gval, L_target, G_ut, R_ut,
                                  gamma_bounds=gamma_bounds,
                                  bisection_tol=bisection_tol,
                                  max_bisect_it=max_bisect_it)
        gamma_refined[i] = gi
        metrics.append(eval_point(gval, gi, L_target, G_ut, R_ut,
                                  fisher_eps=fisher_eps, Y_ut=Y_ut))
    out["gamma"] = gamma_refined
    for key in ("pred_links", "link_error", "entropy", "loglik",
                "S0", "S1", "S2", "J_gamma", "J_beta"):
        out[key] = [m[key] for m in metrics]
    return out


# =================================================================
# Highlight points + true-params fit
# =================================================================

def build_highlight_points(df, true_params=None, max_link_error=1e-6):
    """Min/Max/Mean/Median entropy + optional True Params point."""
    d = df[pd.to_numeric(df["link_error"], errors="coerce")
           < max_link_error].copy()
    if d.empty:
        raise ValueError("No rows remain after filtering by link_error")
    for c in ("g", "gamma", "entropy", "loglik"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    pts = []
    ent = d["entropy"].to_numpy(float)
    g_vals = d["g"].to_numpy(float)
    i_min, i_max = d["entropy"].idxmin(), d["entropy"].idxmax()
    g_lo, g_hi = sorted([float(d.loc[i_min, "g"]), float(d.loc[i_max, "g"])])

    def pick_idx(target):
        diffs = np.abs(ent - target)
        pos = int(np.nanargmin(diffs))
        idx = d.index[pos]
        if g_lo <= g_vals[pos] <= g_hi:
            return idx
        mask = (g_vals >= g_lo) & (g_vals <= g_hi)
        if not mask.any():
            return idx
        cand = np.where(mask)[0]
        return d.index[cand[int(np.nanargmin(np.abs(ent[cand] - target)))]]

    i_mean = pick_idx(float(np.nanmean(ent)))
    i_median = pick_idx(float(np.nanmedian(ent)))

    def mk(idx, label, color, marker, size=350):
        row = d.loc[idx]
        return dict(
            g=float(row["g"]), gamma=float(row["gamma"]),
            entropy=float(row["entropy"]),
            loglik=(float(row["loglik"]) if "loglik" in d.columns
                    and pd.notna(row["loglik"]) else np.nan),
            label=label, color=color, marker=marker,
            size=size, edgecolor="black",
        )

    pts.append(mk(i_min,    "Min Entropy",    "navy",     "v"))
    pts.append(mk(i_max,    "Max Entropy",    "darkred",  "^"))
    pts.append(mk(i_mean,   "Mean Entropy",   "orange",   "o"))
    pts.append(mk(i_median, "Median Entropy", "blue",     "s"))

    if isinstance(true_params, dict) and "gamma" in true_params:
        g_true = float(true_params.get("g", true_params.get("rho")))
        gamma_true = float(true_params["gamma"])
        ent_true = true_params.get("entropy", np.nan)
        ll_true = true_params.get("loglik", np.nan)
        if not np.isfinite(ent_true) or not np.isfinite(ll_true):
            near = ((d["g"] - g_true) ** 2 + (d["gamma"] - gamma_true) ** 2).idxmin()
            if not np.isfinite(ent_true):
                ent_true = float(d.loc[near, "entropy"])
            if not np.isfinite(ll_true) and "loglik" in d.columns:
                ll_true = float(d.loc[near, "loglik"])
        pts.append(dict(g=g_true, gamma=gamma_true,
                        entropy=ent_true, loglik=ll_true,
                        label="True Params", color="yellow",
                        marker="*", size=500, edgecolor="black"))
    return pts



def fit_true_params_from_link_counts(df_raw, L_same_obs, L_diff_obs,
                                     feature_col="fitness", region_col="region",
                                     adj_matrix=None,
                                     tol_links=1e-10, tol_param=1e-10,
                                     max_iter=100):
    """Fit the full-information planted-partition reference from link counts.

    This helper is useful when the sufficient statistics are available as
    counts instead of an explicit adjacency matrix. It solves the two separate
    moment-matching equations for same-block and different-block links and
    returns the package parameterization ``g = exp(beta)`` and ``gamma``.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        Node table with at least ``feature_col`` and ``region_col`` columns.
    L_same_obs : float
        Observed number of links between nodes in the same block.
    L_diff_obs : float
        Observed number of links between nodes in different blocks.
    adj_matrix : array-like, optional
        Binary adjacency matrix. If provided, log-likelihood is evaluated
        against it; otherwise ``loglik`` is returned as NaN.
    """
    G_ut, R_ut, ti, tj, n = prepare_graph_ut(df_raw, feature_col, region_col)
    same = (R_ut == 1.0)
    s_same, s_diff = G_ut[same], G_ut[~same]
    L_same_obs = float(L_same_obs)
    L_diff_obs = float(L_diff_obs)
    L_total = L_same_obs + L_diff_obs

    if not s_same.size or not s_diff.size:
        raise ValueError("Not enough same/diff-block pairs to fit the model")
    if not (0.0 <= L_same_obs <= s_same.size):
        raise ValueError("L_same_obs must be between 0 and the number of same-block pairs")
    if not (0.0 <= L_diff_obs <= s_diff.size):
        raise ValueError("L_diff_obs must be between 0 and the number of different-block pairs")

    def newton(L_obs, s):
        p_bar = np.clip(L_obs / max(s.size, 1), 1e-9, 1.0 - 1e-9)
        s_mean = max(float(np.mean(s)), 1e-15)
        theta = float(np.log(max(p_bar / (s_mean * (1.0 - p_bar)), 1e-12)))
        for _ in range(max_iter):
            param = np.exp(theta)
            x = param * s
            p = x / (1.0 + x)
            F = float(p.sum()) - L_obs
            if abs(F) < tol_links:
                break
            dL = float((s / (1.0 + x) ** 2).sum())
            denom = dL * param
            if abs(denom) < 1e-20:
                break
            step = F / denom
            theta_new = theta - step
            if abs(theta_new - theta) < tol_param * (1.0 + abs(theta)):
                theta = theta_new
                break
            theta = theta_new
        return float(np.exp(theta))

    a = newton(L_same_obs, s_same)
    b = newton(L_diff_obs, s_diff)
    g = b
    alpha = a / b
    gamma = float(np.log(alpha))

    Y_ut = None
    if adj_matrix is not None:
        A = np.asarray(adj_matrix, dtype=float)
        if A.shape != (n, n):
            raise ValueError(f"adj_matrix shape {A.shape} != expected {(n, n)}")
        Y_ut = A[ti, tj].astype(float)

    metrics = eval_point(g, gamma, L_total, G_ut, R_ut, Y_ut=Y_ut)
    if Y_ut is None:
        metrics["loglik"] = np.nan

    return {
        "a": a, "b": b, "g": g, "alpha": alpha, "gamma": gamma,
        "beta": float(np.log(g)),
        "L_same_obs": L_same_obs, "L_diff_obs": L_diff_obs,
        "L_total": L_total,
        **metrics,
    }

def fit_true_params_full_constraints(df_raw, adj_matrix,
                                     feature_col="fitness", region_col="region",
                                     tol_links=1e-10, tol_param=1e-10,
                                     max_iter=100):
    """Fit (a, b) -> (g, alpha, gamma) via Newton on (L_same, L_diff)."""
    G_ut, R_ut, ti, tj, n = prepare_graph_ut(df_raw, feature_col, region_col)
    A = np.asarray(adj_matrix, dtype=float)
    if A.shape != (n, n):
        raise ValueError(f"adj_matrix shape {A.shape} != expected {(n, n)}")
    Y_ut = A[ti, tj].astype(float)
    same = (R_ut == 1.0)

    L_same_obs = float(Y_ut[same].sum())
    L_diff_obs = float(Y_ut[~same].sum())
    L_total = L_same_obs + L_diff_obs
    s_same, s_diff = G_ut[same], G_ut[~same]
    if not s_same.size or not s_diff.size:
        raise ValueError("Not enough same/diff-region pairs to fit a, b")

    def newton(L_obs, s):
        p_bar = np.clip(L_obs / max(s.size, 1), 1e-9, 1.0 - 1e-9)
        s_mean = max(float(np.mean(s)), 1e-15)
        theta = float(np.log(max(p_bar / (s_mean * (1.0 - p_bar)), 1e-12)))
        for _ in range(max_iter):
            param = np.exp(theta)
            x = param * s
            p = x / (1.0 + x)
            F = float(p.sum()) - L_obs
            if abs(F) < tol_links:
                break
            dL = float((s / (1.0 + x) ** 2).sum())
            denom = dL * param
            if abs(denom) < 1e-20:
                break
            step = F / denom
            theta_new = theta - step
            if abs(theta_new - theta) < tol_param * (1.0 + abs(theta)):
                theta = theta_new; break
            theta = theta_new
        return float(np.exp(theta))

    a, b = newton(L_same_obs, s_same), newton(L_diff_obs, s_diff)
    g, alpha = b, a / b
    gamma = float(np.log(alpha))
    metrics = eval_point(g, gamma, L_total, G_ut, R_ut, Y_ut=Y_ut)
    return {"a": a, "b": b, "g": g, "alpha": alpha, "gamma": gamma,
            "L_same_obs": L_same_obs, "L_diff_obs": L_diff_obs,
            "L_total": L_total, **metrics}


# =================================================================
# Orchestrator
# =================================================================

def run_jeffreys_pipeline(df_raw, total_links,
                          g_range=None,
                          feature_col="fitness", region_col="region",
                          max_link_error=1e-6,
                          gamma_bounds=(-60.0, 60.0),
                          bisection_tol=1e-9, max_bisect_it=80,
                          fisher_eps=1e-12,
                          true_params=None, resample_points=200,
                          auto_clip_g_to_feasible=True, adj_matrix=None):
    if g_range is None:
        g_range = np.linspace(1e-5, 3.5, 1000)
    if auto_clip_g_to_feasible:
        g_used, feas = clip_g_range_to_feasible(
            df_raw, g_range, total_links, feature_col, region_col)
    else:
        g_used, feas = np.asarray(g_range, float), None

    df_points = scan_g_curve(
        df_raw, total_links, g_used,
        feature_col=feature_col, region_col=region_col,
        gamma_bounds=gamma_bounds, bisection_tol=bisection_tol,
        max_bisect_it=max_bisect_it, fisher_eps=fisher_eps,
        adj_matrix=adj_matrix,
    )
    df_curve = build_jeffreys_curve(df_points, feasible_only=True)
    df_curve_filtered = filter_and_rebuild_geometry(
        df_curve, max_link_error=max_link_error)

    df_uniform = resample_uniform_in_jeffreys(
        df_curve_filtered, M=resample_points, recompute=True,
        df_raw=df_raw, total_links=total_links,
        feature_col=feature_col, region_col=region_col,
        gamma_bounds=gamma_bounds, bisection_tol=bisection_tol,
        max_bisect_it=max_bisect_it, fisher_eps=fisher_eps,
        adj_matrix=adj_matrix,
    )
    highlights = build_highlight_points(
        df_uniform, true_params=true_params, max_link_error=max_link_error)

    return {
        "g_range_used": g_used, "feasible_mask_preclip": feas,
        "df_points": df_points, "df_curve": df_curve,
        "df_curve_filtered": df_curve_filtered,
        "df_s_uniform_exact": df_uniform, "highlights": highlights,
    }


def extract_true_and_median_info(highlights):
    info = {"True Params": None, "Median Entropy": None}
    for p in highlights:
        if p.get("label") in info:
            info[p["label"]] = {k: p.get(k) for k in ("g", "gamma", "entropy", "loglik")}
    return info["True Params"], info["Median Entropy"]


# =================================================================
# Extra metric: metric2 = mean(exp(loglik per edge))
# =================================================================

def compute_metric2(df_raw, adj_matrix, g, gamma,
                    feature_col="fitness", region_col="region"):
    G_ut, R_ut, ti, tj, n = prepare_graph_ut(df_raw, feature_col, region_col)
    A = np.asarray(adj_matrix, float)
    if A.shape != (n, n):
        raise ValueError(f"adj_matrix shape {A.shape} != expected {(n, n)}")
    Y_ut = A[ti, tj].astype(float)
    Z = g * G_ut * np.exp(gamma * R_ut)
    P = np.clip(Z / (1.0 + Z), 1e-15, 1.0 - 1e-15)
    ll_edge = Y_ut * np.log(P) + (1.0 - Y_ut) * np.log(1.0 - P)
    return float(np.mean(np.exp(ll_edge)))


def add_metric2_to_curve(df_uniform, df_raw, adj_matrix,
                         feature_col="fitness", region_col="region"):
    df_uniform = df_uniform.copy()
    df_uniform["metric2"] = [
        compute_metric2(df_raw, adj_matrix, g, gamma,
                        feature_col=feature_col, region_col=region_col)
        for g, gamma in zip(df_uniform["g"].to_numpy(float),
                            df_uniform["gamma"].to_numpy(float))
    ]
    return df_uniform


def add_metric2_to_highlights(highlights, df_raw, adj_matrix,
                              feature_col="fitness", region_col="region"):
    for p in highlights:
        p["metric2"] = compute_metric2(
            df_raw, adj_matrix, float(p["g"]), float(p["gamma"]),
            feature_col=feature_col, region_col=region_col,
        )
    return highlights
