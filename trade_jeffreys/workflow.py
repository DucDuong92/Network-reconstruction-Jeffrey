"""High-level workflow: from a `ProductConfig` to full Jeffreys analysis.

Example
-------
>>> from trade_jeffreys import ProductConfig, run_product_analysis
>>> cfg = ProductConfig(
...     name="Cocoa (HS180500)",
...     source="baci",
...     trade_path="BACI19.csv",
...     gravity_path="gravity19.csv",
...     code_path="BACIcountry.csv",
...     hs_codes=["180500"],
... )
>>> result = run_product_analysis(cfg, do_plots=False)
>>> result["true_fit"]["g"]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_loading import load_baci_long, load_uncom_long
from .regions import build_country_region_table, REGION_COUNTRY_MAP
from .pipeline import build_country_table_from_long
from .two_param import (
    fit_ab_bisection, reconstruct_probability_matrix,
    compare_matrices, compute_aic_bic,
)
from .jeffreys import (
    fit_true_params_full_constraints, run_jeffreys_pipeline,
    add_metric2_to_curve, add_metric2_to_highlights,
)
from .plotting import plot_curve_2d, plot_curve_3d
from .visualisation import plot_trade_network, plot_gdp_vs_degree
from .validation import compute_metrics_true_and_median


@dataclass
class ProductConfig:
    """Configuration to run the full pipeline for one product."""
    name: str
    source: str                          # "baci" or "uncom"
    trade_path: str                      # BACI*.csv or UNcom*.csv
    gravity_path: str                    # gravity*.csv (CEPII)
    code_path: Optional[str] = None      # BACIcountry.csv (required for baci)
    hs_codes: Optional[Sequence[str]] = None
    hs_match: str = "exact"              # "exact" or "prefix"
    region_country_map: dict = field(default_factory=lambda: REGION_COUNTRY_MAP)
    g_range: Optional[np.ndarray] = None
    resample_points: int = 200
    max_link_error: float = 1e-6
    compute_metric2: bool = True
    aicbic_mode: str = "upper"
    output_prefix: Optional[str] = None  # if set, dump cleaned CSV files


def _load_long_table(cfg: ProductConfig, code_df_for_baci) -> pd.DataFrame:
    if cfg.source == "baci":
        if not cfg.code_path:
            raise ValueError("BACI source requires `code_path` (BACIcountry.csv).")
        return load_baci_long(
            cfg.trade_path, cfg.code_path,
            hs_codes=cfg.hs_codes, match=cfg.hs_match,
        )
    if cfg.source == "uncom":
        return load_uncom_long(cfg.trade_path)
    raise ValueError(f"Unknown source: {cfg.source!r} (expected 'baci' or 'uncom')")


def run_product_analysis(cfg: ProductConfig, do_plots: bool = True,
                         verbose: bool = True):
    """Run the full pipeline for the product described by ``cfg``.

    Returns a dict with all intermediate artifacts:
        df_long, country_df, adj_matrix, true_fit, two_param,
        jeffreys, hp, metrics_2pts.
    """
    code_df = pd.read_csv(cfg.code_path) if cfg.code_path else None
    region_df = build_country_region_table(code_df, cfg.region_country_map)

    df_long = _load_long_table(cfg, code_df)
    if verbose:
        print(f"[{cfg.name}] raw rows: {len(df_long):,}")

    gravity = pd.read_csv(cfg.gravity_path)
    bundle = build_country_table_from_long(df_long, gravity, region_df)
    country_df = bundle["country_df"]
    adj_matrix = bundle["adj_matrix"]
    intra, inter, total = bundle["intra"], bundle["inter"], bundle["total"]
    if verbose:
        print(f"[{cfg.name}] intra={intra}, inter={inter}, total={total}, "
              f"countries={len(country_df)}")

    # ----- Two-parameter (a, b) baseline -----
    a, b, g0, alpha = fit_ab_bisection(
        country_df, L_same_obs=intra, L_diff_obs=inter, verbose=verbose
    )
    prob_matrix = reconstruct_probability_matrix(country_df, g0, alpha)
    metrics_2p = compare_matrices(adj_matrix, prob_matrix, plot=do_plots)
    aic, bic = compute_aic_bic(adj_matrix, prob_matrix, k=2)
    if verbose:
        print(f"[{cfg.name}] 2-param AIC={aic:.4f}, BIC={bic:.4f}")

    # ----- True params (Newton on (L_same, L_diff)) -----
    true_fit = fit_true_params_full_constraints(country_df, adj_matrix)
    true_params = {k: true_fit[k] for k in ("g", "gamma", "entropy", "loglik")}
    total_links = true_fit["L_total"]

    # ----- Jeffreys pipeline -----
    jeffreys = run_jeffreys_pipeline(
        df_raw=country_df, total_links=total_links,
        g_range=cfg.g_range,
        true_params=true_params,
        resample_points=cfg.resample_points,
        max_link_error=cfg.max_link_error,
        adj_matrix=adj_matrix,
    )
    df_uniform = jeffreys["df_s_uniform_exact"]
    hp = jeffreys["highlights"]

    if cfg.compute_metric2:
        df_uniform = add_metric2_to_curve(df_uniform, country_df, adj_matrix)
        hp = add_metric2_to_highlights(hp, country_df, adj_matrix)
        jeffreys["df_s_uniform_exact"] = df_uniform
        jeffreys["highlights"] = hp

    # ----- Evaluation -----
    metrics_2pts = compute_metrics_true_and_median(
        df_raw=country_df, adj_matrix=adj_matrix,
        df_uniform=df_uniform, highlights=hp,
        aicbic_mode=cfg.aicbic_mode,
    )

    # ----- Optional plots -----
    if do_plots:
        plot_trade_network(bundle["df_links"], country_df, min_degree=3)
        plot_gdp_vs_degree(bundle["df_links"], "i", "GDP", direction="out")
        plot_gdp_vs_degree(bundle["df_links"], "j", "GDP_j", direction="in")

        plot_curve_3d(df_uniform, z_column="entropy",
                      z_label="Entropy", highlight_points=hp)
        plot_curve_2d(df_uniform, x_column="g", y_column="entropy",
                      x_label=r"exp($\beta$)", y_label="Entropy",
                      highlight_points=hp)
        plot_curve_2d(df_uniform, x_column="gamma", y_column="entropy",
                      x_label=r"$\gamma$", y_label="Entropy",
                      sort_within="gamma", highlight_points=hp)

        if df_uniform["loglik"].notna().any():
            plot_curve_3d(df_uniform, z_column="loglik",
                          z_label="Log-Likelihood", highlight_points=hp)
            plot_curve_2d(df_uniform, x_column="g", y_column="loglik",
                          x_label=r"exp($\beta$)", y_label="Log-Likelihood",
                          highlight_points=hp)

        if cfg.compute_metric2:
            plot_curve_3d(df_uniform, z_column="metric2",
                          z_label="Metric2 (mean exp loglik)",
                          highlight_points=hp)
            plot_curve_2d(df_uniform, x_column="g", y_column="metric2",
                          x_label=r"exp($\beta$)",
                          y_label="Metric2 (mean exp loglik)",
                          highlight_points=hp)

    # ----- Optional export -----
    if cfg.output_prefix:
        country_df.to_csv(f"{cfg.output_prefix}_country.csv", index=False)
        adj_matrix.to_csv(f"{cfg.output_prefix}_matrix.csv")
        bundle["df_links"].to_csv(f"{cfg.output_prefix}_links.csv", index=False)
        df_uniform.to_csv(f"{cfg.output_prefix}_jeffreys.csv", index=False)

    return {
        "df_long": bundle["df_links"],
        "country_df": country_df,
        "adj_matrix": adj_matrix,
        "intra": intra, "inter": inter, "total": total,
        "two_param": {"a": a, "b": b, "g": g0, "alpha": alpha,
                      "prob_matrix": prob_matrix,
                      "metrics": metrics_2p, "AIC": aic, "BIC": bic},
        "true_fit": true_fit,
        "jeffreys": jeffreys,
        "hp": hp,
        "metrics_2pts": metrics_2pts,
    }
