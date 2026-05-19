"""GDP / region pipeline: from a long trade table -> per-country table + matrix."""
import pandas as pd
import numpy as np


def extract_gdp_by_iso3(gravity_df, iso3_o="iso3_o", gdp_o="gdp_o",
                        iso3_d="iso3_d", gdp_d="gdp_d"):
    """Return unique ISO3 -> GDP table by combining origin & destination columns
    of a CEPII gravity dataframe."""
    origin = gravity_df[[iso3_o, gdp_o]].rename(columns={iso3_o: "ISO3", gdp_o: "GDP"})
    dest = gravity_df[[iso3_d, gdp_d]].rename(columns={iso3_d: "ISO3", gdp_d: "GDP"})
    combined = pd.concat([origin, dest], ignore_index=True).dropna()
    return combined.groupby("ISO3", as_index=False).first()


def add_gdp_columns(df, gdp_df, col_i="i", col_j="j"):
    """Merge GDP onto a long trade table -> adds `GDP` (origin) and `GDP_j` (dest)."""
    df = df.copy()
    df[col_i] = df[col_i].astype(str).str.upper().str.strip()
    df[col_j] = df[col_j].astype(str).str.upper().str.strip()
    g = gdp_df.copy()
    g["ISO3"] = g["ISO3"].astype(str).str.upper().str.strip()
    df = df.merge(g.rename(columns={"ISO3": col_i, "GDP": "GDP"}), on=col_i, how="left")
    df = df.merge(g.rename(columns={"ISO3": col_j, "GDP": "GDP_j"}), on=col_j, how="left")
    return df


def add_region_similarity(df, iso3_region_df, col_i="i", col_j="j"):
    """Add `region_i`, `region_j`, `region_similar` columns to a long trade table."""
    df = df.copy()
    iso2reg = dict(zip(iso3_region_df["country_iso3"], iso3_region_df["region"]))
    df["region_i"] = df[col_i].map(iso2reg)
    df["region_j"] = df[col_j].map(iso2reg)
    df["region_similar"] = (df["region_i"] == df["region_j"]).astype(int)
    return df


def build_country_info_df(df, gdp_df, region_df, col_i="i", col_j="j"):
    """One row per country: ISO3, GDP, country_name, region, normalised fitness."""
    iso3 = pd.unique(df[[col_i, col_j]].values.ravel())
    out = pd.DataFrame({"country_iso3": iso3})
    out = out.merge(gdp_df.rename(columns={"ISO3": "country_iso3"}),
                    on="country_iso3", how="left")
    out = out.merge(region_df[["country_iso3", "country_name", "region"]],
                    on="country_iso3", how="left")
    out = out.dropna(subset=["GDP", "region"])
    out["fitness"] = out["GDP"] / out["GDP"].mean()
    return out.sort_values("country_iso3").reset_index(drop=True)


def create_product_matrix(df, col_i="i", col_j="j"):
    """Square binary adjacency matrix indexed by sorted ISO3 codes."""
    iso3 = sorted(pd.unique(df[[col_i, col_j]].values.ravel()))
    return (df.assign(link=1)
              .pivot_table(index=col_i, columns=col_j, values="link", fill_value=0)
              .reindex(index=iso3, columns=iso3, fill_value=0))


def count_region_links(df, region_i="region_i", region_j="region_j"):
    """Vectorised count of intra/inter region links (each row = one link)."""
    valid = df[region_i].notna() & df[region_j].notna()
    same = valid & (df[region_i] == df[region_j])
    diff = valid & (df[region_i] != df[region_j])
    intra, inter = int(same.sum()), int(diff.sum())
    return intra, inter, intra + inter


# ---------------------------------------------------------------
# Convenience: full pipeline from long trade table -> products
# ---------------------------------------------------------------

def build_country_table_from_long(df_long, gravity_df, region_df,
                                  col_i="i", col_j="j",
                                  drop_self_loops=True):
    """Apply the full pipeline.

    Returns a dict with:
      - `df_links`     long edge table (cleaned)
      - `country_df`   one row per country
      - `adj_matrix`   square binary adjacency
      - `intra/inter/total` link counts
      - `gdp_df`       ISO3 -> GDP table
    """
    gdp_df = extract_gdp_by_iso3(gravity_df)

    df_long = (df_long.pipe(add_gdp_columns, gdp_df, col_i=col_i, col_j=col_j)
                       .pipe(add_region_similarity, region_df,
                             col_i=col_i, col_j=col_j))
    if drop_self_loops:
        df_long = df_long[df_long[col_i] != df_long[col_j]]

    df_long = df_long.dropna(subset=["GDP", "GDP_j", "region_i", "region_j"])
    country_df = build_country_info_df(df_long, gdp_df, region_df,
                                       col_i=col_i, col_j=col_j)

    # Restrict matrix to countries kept in the country_df
    adj_full = create_product_matrix(df_long, col_i=col_i, col_j=col_j)
    valid = country_df["country_iso3"].values
    adj_matrix = adj_full.loc[adj_full.index.isin(valid),
                              adj_full.columns.isin(valid)]
    # Re-align ordering
    adj_matrix = adj_matrix.reindex(index=country_df["country_iso3"],
                                    columns=country_df["country_iso3"],
                                    fill_value=0)

    intra, inter, total = count_region_links(df_long)
    return {
        "df_links": df_long.reset_index(drop=True),
        "country_df": country_df,
        "adj_matrix": adj_matrix,
        "intra": intra, "inter": inter, "total": total,
        "gdp_df": gdp_df,
    }
