# Usage guide for `trade_jeffreys`

`trade_jeffreys` runs the trade-network reconstruction pipeline used in the
paper **Network Reconstruction via Jeffreys Prior under Missing Sufficient
Statistics**. The package supports both product-level workflows using BACI / UN
Comtrade data and lower-level workflows where the user provides network inputs
such as total links, intra-block links, inter-block links, node fitnesses, and
block labels.

Paper link: <https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ps0A1EYAAAAJ&citation_for_view=ps0A1EYAAAAJ:d1gkVwhDpl0C>

## 1. Import

From the workspace root, where the `trade_jeffreys/` package folder is visible:

```python
from trade_jeffreys import ProductConfig, run_product_analysis
```

For lower-level use:

```python
from trade_jeffreys import (
    fit_true_params_from_link_counts,
    fit_true_params_full_constraints,
    run_jeffreys_pipeline,
    compute_metrics_true_and_median,
    plot_curve_2d,
    plot_curve_3d,
)
```

## 2. Fastest product-level workflow

### BACI data

BACI uses numeric country codes, so three files are needed:

- `BACI19.csv`: bilateral trade data;
- `gravity19.csv`: CEPII gravity data containing GDP;
- `BACIcountry.csv`: numeric-country-code to ISO3 mapping.

Example for cocoa, HS code `180500`:

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
)

result = run_product_analysis(cfg, do_plots=True, verbose=True)
```

To analyze a different product, change only `hs_codes`:

```python
cfg = ProductConfig(
    name="Steel (HS720825)",
    source="baci",
    trade_path="BACI19.csv",
    gravity_path="gravity19.csv",
    code_path="BACIcountry.csv",
    hs_codes=["720825"],
)
```

To analyze a product group by HS prefix:

```python
cfg = ProductConfig(
    name="All cocoa and chocolate products (HS18*)",
    source="baci",
    trade_path="BACI19.csv",
    gravity_path="gravity19.csv",
    code_path="BACIcountry.csv",
    hs_codes=["18"],
    hs_match="prefix",
)
```

### UN Comtrade data

UN Comtrade files are assumed to be pre-filtered to one product and to contain
ISO3 reporter/partner columns.

```python
cfg = ProductConfig(
    name="Steel (UN17)",
    source="uncom",
    trade_path="UNcom17_steel.csv",
    gravity_path="gravity17.csv",
)

result = run_product_analysis(cfg, do_plots=True, verbose=True)
```

## 3. Speed controls

For exploratory runs, reduce the number of curve-scan and resampling points:

```python
import numpy as np

cfg = ProductConfig(
    name="Cocoa quick run",
    source="baci",
    trade_path="BACI19.csv",
    gravity_path="gravity19.csv",
    code_path="BACIcountry.csv",
    hs_codes=["180500"],
    g_range=np.linspace(1e-5, 3.5, 100),
    resample_points=50,
)

result = run_product_analysis(cfg, do_plots=False, verbose=True)
```

`do_plots=False` skips plotting and is recommended for batch runs.

## 4. Returned result dictionary

`result = run_product_analysis(cfg)` returns:

| key | description |
|---|---|
| `df_long` | cleaned long edge table with source/destination GDP and region columns |
| `country_df` | one row per country with ISO3, GDP, region, and fitness |
| `adj_matrix` | binary adjacency matrix indexed by ISO3 |
| `intra` | observed number of intra-region links |
| `inter` | observed number of inter-region links |
| `total` | observed total number of links |
| `two_param` | two-constraint planted-partition baseline and metrics |
| `true_fit` | full-information reference parameters using intra/inter link counts |
| `jeffreys` | Jeffreys feasible curve, filtered curve, uniform samples, highlights |
| `hp` | highlight points: min, max, mean, median entropy, and true parameters |
| `metrics_2pts` | ROC AUC, PR AUC, AIC, and BIC for True Params and Median Entropy |

Typical access pattern:

```python
country_df = result["country_df"]
adj_matrix = result["adj_matrix"]
curve = result["jeffreys"]["df_s_uniform_exact"]
highlights = result["hp"]
metrics = result["metrics_2pts"]

print(len(country_df), result["total"])
print(result["true_fit"]["g"], result["true_fit"]["gamma"])
print(metrics)
```

## 5. Exporting outputs

Set `output_prefix` in `ProductConfig`:

```python
cfg = ProductConfig(
    name="Cocoa (HS180500)",
    source="baci",
    trade_path="BACI19.csv",
    gravity_path="gravity19.csv",
    code_path="BACIcountry.csv",
    hs_codes=["180500"],
    output_prefix="BACI19_choco",
)

result = run_product_analysis(cfg)
```

This writes:

- `BACI19_choco_country.csv`
- `BACI19_choco_matrix.csv`
- `BACI19_choco_links.csv`
- `BACI19_choco_jeffreys.csv`

## 6. Replotting from an existing result

```python
from trade_jeffreys import plot_curve_2d, plot_curve_3d

curve = result["jeffreys"]["df_s_uniform_exact"]
highlights = result["hp"]

plot_curve_3d(curve, z_column="entropy", z_label="Entropy",
              highlight_points=highlights)

plot_curve_2d(curve, x_column="g", y_column="entropy",
              x_label=r"exp($\beta$)", y_label="Entropy",
              highlight_points=highlights)

plot_curve_2d(curve, x_column="gamma", y_column="entropy",
              x_label=r"$\gamma$", y_label="Entropy",
              sort_within="gamma", highlight_points=highlights)
```

## 7. Lower-level network-input workflow

This mode matches the paper's mathematical setup most directly. The user starts
from:

- node fitnesses `x_i`;
- block labels for all nodes;
- total number of observed links `L_total`;
- intra-block link count `L_same`, used for the full-information reference;
- inter-block link count `L_diff`, used for the full-information reference;
- optional binary adjacency matrix for likelihood and ROC/PR evaluation.

A complete runnable notebook is included at:

```text
trade_jeffreys/notebooks/jeffreys_synthetic_usage.ipynb
```

Minimal code sketch:

```python
import numpy as np
import pandas as pd

from trade_jeffreys import (
    fit_true_params_from_link_counts,
    run_jeffreys_pipeline,
    plot_curve_2d,
    plot_curve_3d,
)

country_df = pd.DataFrame({
    "ISO3": [f"N{i:02d}" for i in range(12)],
    "region": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"],
    "fitness": np.linspace(0.5, 1.6, 12),
})

L_same = 12
L_diff = 18
L_total = L_same + L_diff

true_fit = fit_true_params_from_link_counts(country_df, L_same, L_diff)
true_params = {k: true_fit[k] for k in ("g", "gamma", "entropy", "loglik")}

jeffreys = run_jeffreys_pipeline(
    df_raw=country_df,
    total_links=L_total,
    true_params=true_params,
    resample_points=200,
)

curve = jeffreys["df_s_uniform_exact"]
highlights = jeffreys["highlights"]

plot_curve_3d(curve, z_column="entropy", z_label="Entropy",
              highlight_points=highlights)
plot_curve_2d(curve, x_column="g", y_column="entropy",
              x_label=r"exp($\beta$)", y_label="Entropy",
              highlight_points=highlights)
```

## 8. Advanced step-by-step workflow for data files

### Step 1: load data

```python
from trade_jeffreys import load_baci_long, load_uncom_long

df_long = load_baci_long(
    "BACI19.csv",
    "BACIcountry.csv",
    hs_codes=["180500"],
    match="exact",
)
```

or:

```python
df_long = load_uncom_long("UNcom17_steel.csv")
```

### Step 2: build region table

```python
import pandas as pd
from trade_jeffreys import REGION_COUNTRY_MAP, build_country_region_table

code_df = pd.read_csv("BACIcountry.csv")
region_df = build_country_region_table(code_df, REGION_COUNTRY_MAP)
```

### Step 3: build country table and adjacency matrix

```python
from trade_jeffreys import build_country_table_from_long

gravity = pd.read_csv("gravity19.csv")
bundle = build_country_table_from_long(df_long, gravity, region_df)

country_df = bundle["country_df"]
adj_matrix = bundle["adj_matrix"]
intra = bundle["intra"]
inter = bundle["inter"]
total = bundle["total"]
```

### Step 4: fit the full-information reference

```python
from trade_jeffreys import fit_true_params_full_constraints

true_fit = fit_true_params_full_constraints(country_df, adj_matrix)
```

### Step 5: run the Jeffreys-prior feasible-curve method

```python
from trade_jeffreys import run_jeffreys_pipeline

true_params = {k: true_fit[k] for k in ("g", "gamma", "entropy", "loglik")}

jeffreys = run_jeffreys_pipeline(
    df_raw=country_df,
    total_links=total,
    true_params=true_params,
    adj_matrix=adj_matrix,
    resample_points=200,
)
```

### Step 6: compute metrics

```python
from trade_jeffreys import compute_metrics_true_and_median

metrics = compute_metrics_true_and_median(
    df_raw=country_df,
    adj_matrix=adj_matrix,
    df_uniform=jeffreys["df_s_uniform_exact"],
    highlights=jeffreys["highlights"],
)
```

## 9. Common issues

### ImportError: `No module named trade_jeffreys`

Run the notebook from the parent folder containing the `trade_jeffreys/` package,
or add that parent folder to `sys.path`.

### The BACI country-code file is not found

Check that `code_path` points to `BACIcountry.csv`. If the file is in a
subfolder, use a full or relative path, for example:

```python
code_path="data/BACIcountry.csv"
```

### The run is slow

Reduce `g_range`, reduce `resample_points`, and set `do_plots=False`.

### Small numerical discrepancies

Small numerical differences can appear because the feasible curve is constructed by
root-finding, interpolation, and floating-point arithmetic.

### Custom block labels

Pass a custom `region_country_map` in `ProductConfig`, or build `country_df`
manually with a custom `region` column for lower-level workflows.


## 10. Real-data reproduction notebook

The package includes a real-data example notebook:

```text
trade_jeffreys/notebooks/BACI20_wood_real_data_usage.ipynb
```

It rewrites the original BACI 2020 wood exploratory notebook using the package API.
The notebook expects the following local input files:

```text
BACI20.csv
gravity20.csv
BACIcountry.csv
```

It then loads HS code `441300`, builds the country table and adjacency matrix,
fits the full-information FCBM reference point, constructs the Jeffreys feasible
curve using only the total link count, plots entropy and log-likelihood, and
prints the final model-performance table for the paper's Wood 2020 case.


## AIC/BIC convention

AIC and BIC are computed on unordered dyads (`i < j`) by default, with `N = n * (n - 1) / 2` observations in the BIC penalty. This matches the convention used in the paper's Table 3 for undirected binary networks.
