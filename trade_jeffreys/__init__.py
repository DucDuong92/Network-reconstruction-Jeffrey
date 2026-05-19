"""trade_jeffreys — reusable trade-network + Jeffreys-curve pipeline.

Public surface (re-exported from sub-modules):

    from trade_jeffreys import (
        # data loading
        load_baci_long, load_uncom_long, filter_by_exact_match, filter_by_prefix,
        # region mapping
        REGION_COUNTRY_MAP, REGION_COLORS, build_country_region_table,
        # pipeline
        extract_gdp_by_iso3, add_gdp_columns, add_region_similarity,
        build_country_info_df, create_product_matrix, count_region_links,
        # two-parameter (a, b) model
        fit_ab_bisection, reconstruct_probability_matrix,
        compare_matrices, compute_aic_bic,
        # Jeffreys curve
        prepare_graph_ut, eval_point, scan_g_curve, build_jeffreys_curve,
        filter_and_rebuild_geometry, resample_uniform_in_jeffreys,
        build_highlight_points, fit_true_params_from_link_counts,
        fit_true_params_full_constraints, run_jeffreys_pipeline,
        extract_true_and_median_info,
        # plotting
        plot_curve_2d, plot_curve_3d,
        plot_trade_network, plot_gdp_vs_degree,
        # validation
        compute_point_roc_pr_aic_bic,
        compute_metrics_true_and_median,
        fit_block_agnostic_fm, probability_matrix_from_g_gamma,
        paper_model_comparison_table,
        # high-level workflow
        run_product_analysis, ProductConfig,
    )
"""

from .regions import REGION_COUNTRY_MAP, REGION_COLORS, build_country_region_table
from .data_loading import (
    load_baci_long, load_uncom_long,
    filter_by_exact_match, filter_by_prefix,
)
from .pipeline import (
    extract_gdp_by_iso3, add_gdp_columns, add_region_similarity,
    build_country_info_df, create_product_matrix, count_region_links,
    build_country_table_from_long,
)
from .two_param import (
    fit_ab_bisection, reconstruct_probability_matrix,
    compare_matrices, compute_aic_bic,
)
from .jeffreys import (
    prepare_graph_ut, eval_point, scan_g_curve, build_jeffreys_curve,
    filter_and_rebuild_geometry, resample_uniform_in_jeffreys,
    build_highlight_points, fit_true_params_from_link_counts,
    fit_true_params_full_constraints, run_jeffreys_pipeline,
    extract_true_and_median_info,
    compute_metric2, add_metric2_to_curve, add_metric2_to_highlights,
)
from .plotting import plot_curve_2d, plot_curve_3d
from .visualisation import plot_trade_network, plot_gdp_vs_degree
from .validation import (
    compute_point_roc_pr_aic_bic,
    compute_metrics_true_and_median,
    fit_block_agnostic_fm, probability_matrix_from_g_gamma,
    paper_model_comparison_table,
)
from .workflow import run_product_analysis, ProductConfig

__version__ = "0.1.0"
__author__ = "Minh Duc Duong; Diego Garlaschelli"
__copyright__ = "Copyright (c) 2026 Minh Duc Duong and Diego Garlaschelli. All rights reserved."
__paper_url__ = "https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ps0A1EYAAAAJ&citation_for_view=ps0A1EYAAAAJ:d1gkVwhDpl0C"
