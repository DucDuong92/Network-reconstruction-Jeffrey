"""Generic 2D / 3D plotting on Jeffreys-curve dataframes."""
import numpy as np
import matplotlib.pyplot as plt

_BASE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def _split_highlights(highlights):
    others = [p for p in highlights if p.get("label") != "True Params"]
    trues = [p for p in highlights if p.get("label") == "True Params"]
    return others, trues


def _scatter_sizes(df, size_by):
    if size_by is None or size_by not in df.columns:
        return np.full(len(df), 40.0)
    s = df[size_by].astype(float).to_numpy()
    s_min, s_max = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max - s_min < 1e-15:
        return np.full(len(df), 40.0)
    return np.clip(80.0 * (s - s_min) / (s_max - s_min + 1e-15) + 20.0, 10.0, 150.0)


def plot_curve_3d(df, z_column="entropy", z_label="Entropy",
                  highlight_points=None, size_by="J_beta",
                  connect_by="run_id", sort_within="beta",
                  log_z=False, elev=30, azim=75, figsize=(14, 9),
                  label_fontsize=20, tick_fontsize=13):
    """3D Jeffreys plot (z = entropy / loglik / metric2)."""
    if z_column not in df.columns:
        raise ValueError(f"{z_column} not found in dataframe")
    eps = 1e-12
    z_raw = df[z_column].astype(float).to_numpy()
    z_vals = np.log10(np.maximum(z_raw, eps)) if log_z else z_raw
    sizes = _scatter_sizes(df, size_by)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if connect_by is not None and connect_by in df.columns:
        for gi, (_, gdf) in enumerate(df.groupby(connect_by, sort=False)):
            gdf = gdf.sort_values(sort_within)
            color = _BASE_COLORS[gi % len(_BASE_COLORS)]
            zg = gdf[z_column].to_numpy(float)
            if log_z:
                zg = np.log10(np.maximum(zg, eps))
            ax.scatter(gdf["g"], gdf["gamma"], zg, color=color,
                       s=sizes[gdf.index], alpha=0.6)
            ax.plot(gdf["g"], gdf["gamma"], zg, color=color,
                    linewidth=2, alpha=0.9)
    else:
        ax.scatter(df["g"], df["gamma"], z_vals,
                   color="tab:blue", s=sizes, alpha=0.6)

    if highlight_points:
        others, trues = _split_highlights(highlight_points)

        def draw(p, scale, zorder):
            v = p.get(z_column, np.nan)
            if not np.isfinite(v):
                return
            zh = np.log10(max(v, eps)) if log_z else v
            ax.scatter(float(p["g"]), float(p["gamma"]), zh,
                       color=p.get("color", "black"),
                       s=scale * p.get("size", 300),
                       marker=p.get("marker", "*"),
                       edgecolor=p.get("edgecolor", "k"),
                       linewidth=1.5, depthshade=False, zorder=zorder)
        for p in others:
            draw(p, 0.95, 200)
        for p in trues:
            draw(p, 1.3, 1000)

    ax.set_xlabel(r"exp($\beta$)", fontsize=label_fontsize, labelpad=18)
    ax.set_ylabel(r"$\gamma$", fontsize=label_fontsize, labelpad=18)
    ax.set_zlabel(z_label + (" (log10)" if log_z else ""),
                  fontsize=label_fontsize, labelpad=25)
    ax.tick_params(axis="both", labelsize=tick_fontsize, pad=6)
    ax.tick_params(axis="z", labelsize=tick_fontsize, pad=12)
    zmin, zmax = np.nanmin(z_vals), np.nanmax(z_vals)
    if np.isfinite(zmin) and np.isfinite(zmax):
        pad = 0.15 * (zmax - zmin + 1e-9)
        ax.set_zlim(zmin - pad, zmax + pad)
    ax.view_init(elev=elev, azim=azim)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.show()


def plot_curve_2d(df, x_column="g", y_column="entropy",
                  x_label=r"exp($\beta$)", y_label="Entropy",
                  connect_by="run_id", sort_within=None,
                  highlight_points=None, figsize=(10, 6),
                  label_fontsize=20, tick_fontsize=14,
                  line_width=2.2, highlight_scale=1.2,
                  show_grid=True):
    """Generic 2D plot used for entropy / loglik / metric2 vs g (or vs gamma)."""
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(f"df must contain '{x_column}' and '{y_column}'")
    sort_within = sort_within or x_column
    fig, ax = plt.subplots(figsize=figsize)

    if connect_by is not None and connect_by in df.columns:
        for gi, (_, gdf) in enumerate(df.groupby(connect_by, sort=False)):
            gdf = gdf.sort_values(sort_within)
            ax.plot(gdf[x_column].to_numpy(float),
                    gdf[y_column].to_numpy(float),
                    color=_BASE_COLORS[gi % len(_BASE_COLORS)],
                    linewidth=line_width)
    else:
        dd = df.sort_values(sort_within)
        ax.plot(dd[x_column].to_numpy(float),
                dd[y_column].to_numpy(float),
                color="tab:blue", linewidth=line_width)

    if highlight_points:
        others, trues = _split_highlights(highlight_points)

        def draw(p, scale, zorder):
            v = p.get(y_column, np.nan)
            x0 = p.get(x_column, np.nan)
            if not np.isfinite(v) or not np.isfinite(x0):
                return
            x0 = float(x0)
            ax.axvline(x0, color=p.get("color", "black"),
                       linestyle="--", linewidth=1.5, alpha=0.7)
            ax.scatter(x0, float(v),
                       color=p.get("color", "black"),
                       s=scale * p.get("size", 300),
                       marker=p.get("marker", "*"),
                       edgecolor=p.get("edgecolor", "k"),
                       linewidth=1.5, zorder=zorder)
        for p in others:
            draw(p, 0.95, 20)
        for p in trues:
            draw(p, highlight_scale, 100)

    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    xmin, xmax = df[x_column].min(), df[x_column].max()
    ymin, ymax = df[y_column].min(), df[y_column].max()
    ax.set_xlim(xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
    plt.tight_layout()
    plt.show()
