# trade_jeffreys

Reusable Python package for trade-network reconstruction with the Fitness Model,
the Fitness-Corrected Block Model in planted-partition form, and the Jeffreys-prior
feasible-curve method for missing sufficient statistics.

This package is associated with the research article:

> **Network Reconstruction via Jeffreys Prior under Missing Sufficient Statistics**  
> Minh Duc Duong and Diego Garlaschelli, 2026.  
> Google Scholar record: <https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ps0A1EYAAAAJ&citation_for_view=ps0A1EYAAAAJ:d1gkVwhDpl0C>

The paper studies binary reconstruction of international trade networks when only
node fitnesses, block labels, and the total number of links are available. The core
contribution is a Jeffreys-prior averaging procedure over the one-dimensional
feasible curve defined by the total-link constraint.

## Authors and ownership

Package authors and copyright holders:

- Minh Duc Duong
- Diego Garlaschelli

Copyright (c) 2026 Minh Duc Duong and Diego Garlaschelli. All rights reserved
unless a separate written license is granted by the copyright holders. See
`LICENSE` and `AUTHORS.md` for the package ownership notice.

## Model parameterization

The package uses `g = exp(beta)` internally. Therefore the paper's probability

```text
p_ij(beta, gamma) = exp(beta) exp(gamma R_ij) x_i x_j /
                    (1 + exp(beta) exp(gamma R_ij) x_i x_j)
```

is represented in code as

```text
p_ij(g, gamma) = g exp(gamma R_ij) x_i x_j /
                 (1 + g exp(gamma R_ij) x_i x_j)
```

where `R_ij = 1` for nodes in the same block and `R_ij = 0` otherwise.

## Package layout

```text
trade_jeffreys/
├── regions.py          World Bank region map and colors
├── data_loading.py     BACI / UN Comtrade loading utilities and HS filters
├── pipeline.py         GDP / region pipeline to country_df and adj_matrix
├── two_param.py        two-constraint planted-partition fit and metrics
├── jeffreys.py         Jeffreys feasible-curve scan, resampling, highlights
├── plotting.py         generic 2D and 3D curve plotting
├── visualisation.py    network and GDP-vs-degree plotting
├── validation.py       ROC/PR/AIC/BIC and paper-style comparison helpers
├── workflow.py         ProductConfig and run_product_analysis()
├── examples.py         small script-style examples
├── USAGE.md            detailed usage guide
├── AUTHORS.md          package authorship and ownership notice
└── LICENSE             copyright and rights notice
```

Notebook examples are also provided at:

```text
trade_jeffreys/notebooks/jeffreys_synthetic_usage.ipynb
trade_jeffreys/notebooks/BACI20_wood_real_data_usage.ipynb
```

The synthetic notebook demonstrates a network workflow in which the user initializes node
fitnesses, block labels, total links, intra-block links, and inter-block links;
then computes the full-information reference parameters, the Jeffreys median
entropy solution, metrics, and plots matching the paper's figures.

The BACI 2020 wood notebook is a real-data reproduction example. It expects `BACI20.csv`, `gravity20.csv`, and `BACIcountry.csv` to be available locally and reproduces the Wood 2020 metrics reported in the paper.

## Installation / import

For local use, place the `trade_jeffreys/` folder on your Python path and import:

```python
from trade_jeffreys import ProductConfig, run_product_analysis
```

For notebooks stored inside `trade_jeffreys/notebooks/`, the first cell adds the
project root to `sys.path` automatically.

Core dependencies:

```text
numpy
pandas
matplotlib
networkx
scikit-learn
```

## Quick start: product-level workflow

### BACI source

```python
from trade_jeffreys import ProductConfig, run_product_analysis

cfg = ProductConfig(
    name="Cocoa (HS180500)",
    source="baci",
    trade_path="BACI19.csv",
    gravity_path="gravity19.csv",
    code_path="BACIcountry.csv",
    hs_codes=["180500"],
    hs_match="exact",
    output_prefix="BACI19_choco",  # optional CSV export
)

result = run_product_analysis(cfg, do_plots=True, verbose=True)
```

To analyze a product family by HS prefix:

```python
cfg = ProductConfig(
    name="All cocoa products (HS18*)",
    source="baci",
    trade_path="BACI19.csv",
    gravity_path="gravity19.csv",
    code_path="BACIcountry.csv",
    hs_codes=["18"],
    hs_match="prefix",
)
```

### UN Comtrade source

```python
cfg = ProductConfig(
    name="Steel (UN17)",
    source="uncom",
    trade_path="UNcom17_steel.csv",
    gravity_path="gravity17.csv",
)

result = run_product_analysis(cfg, do_plots=False, verbose=True)
```

## Lower-level workflow

`run_product_analysis` is a convenience wrapper. The paper-level algorithm can
also be run manually from network-level inputs:

```python
from trade_jeffreys import (
    fit_true_params_from_link_counts,
    run_jeffreys_pipeline,
    compute_metrics_true_and_median,
    plot_curve_2d,
    plot_curve_3d,
)
```

The low-level inputs are:

- `country_df`: one row per node, with at least `fitness` and `region` columns;
- `L_total`: observed total number of links;
- `L_same`: observed number of intra-block links, used only for the
  full-information reference fit;
- `L_diff`: observed number of inter-block links, used only for the
  full-information reference fit;
- `adj_matrix`: optional binary adjacency matrix, required for ROC/PR and
  log-likelihood metrics.

## Outputs

`run_product_analysis(cfg)` returns a dictionary containing:

| key             | content                                                |
|-----------------|--------------------------------------------------------|
| `df_long`       | cleaned long edge table with GDP and region columns    |
| `country_df`    | one row per country: ISO3, GDP, region, fitness        |
| `adj_matrix`    | binary adjacency DataFrame indexed by ISO3             |
| `two_param`     | two-constraint fit plus metrics/AIC/BIC                |
| `true_fit`      | full-information planted-partition fit                 |
| `jeffreys`      | Jeffreys feasible curve, uniform samples, highlights   |
| `hp`            | highlight points: min/max/mean/median entropy + true   |
| `metrics_2pts`  | ROC/PR/AIC/BIC at True Params and Median Entropy       |

## Citation

If this package is used in academic work, cite the article above and mention that
the implementation is the companion `trade_jeffreys` package by Minh Duc Duong
and Diego Garlaschelli.


## AIC/BIC convention

AIC and BIC are computed on unordered dyads (`i < j`) by default, with `N = n * (n - 1) / 2` observations in the BIC penalty. This matches the convention used in the paper's Table 3 for undirected binary networks.
