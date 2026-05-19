"""Data loading: BACI (numeric codes -> ISO3 + product filter) and Uncom (direct)."""
from __future__ import annotations
import pandas as pd


# ---------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------

def filter_by_exact_match(df, value_list, column="k"):
    """Keep rows where `column` exactly matches any value in value_list.
    Returns the full dataframe if value_list is empty."""
    if not value_list:
        return df.copy()
    df = df.copy()
    df[column] = df[column].astype(str)
    return df[df[column].isin({str(v) for v in value_list})]


def filter_by_prefix(df, prefix_list, column="k"):
    """Keep rows where `column` starts with any prefix in prefix_list."""
    df = df.copy()
    df[column] = df[column].astype(str)
    return df[df[column].str.startswith(tuple(prefix_list))]


# ---------------------------------------------------------------
# BACI loaders (numeric codes need ISO3 mapping)
# ---------------------------------------------------------------

def load_baci_long(baci_path, code_path, hs_codes=None, *,
                   hs_column="k", origin_col="i", dest_col="j",
                   match="exact"):
    """Load a BACI yearly CSV and return a long-format trade table.

    - Maps numeric origin/destination codes -> ISO3 via BACIcountry.csv.
    - Optionally filters to ``hs_codes`` (list[str]). ``match`` is
      ``"exact"`` (default) or ``"prefix"``.

    Returns the filtered dataframe with `i`, `j` as ISO3 strings.
    """
    baci = pd.read_csv(baci_path)
    code = pd.read_csv(code_path)

    code_map = dict(zip(code["country_code"], code["country_iso3"]))
    baci[origin_col] = baci[origin_col].map(code_map)
    baci[dest_col] = baci[dest_col].map(code_map)

    if hs_codes:
        if match == "exact":
            baci = filter_by_exact_match(baci, hs_codes, column=hs_column)
        elif match == "prefix":
            baci = filter_by_prefix(baci, hs_codes, column=hs_column)
        else:
            raise ValueError("match must be 'exact' or 'prefix'")
    return baci.reset_index(drop=True)


# ---------------------------------------------------------------
# Uncom loaders (already in ISO3 form, just rename columns)
# ---------------------------------------------------------------

def load_uncom_long(uncom_path, *,
                    reporter_col="reporterISO", partner_col="partnerISO"):
    """Load a UN Comtrade CSV and rename ``reporterISO/partnerISO`` -> ``i/j``.

    Already filtered to one product upstream; no HS filter applied here.
    """
    df = pd.read_csv(uncom_path)
    return df.rename(columns={reporter_col: "i", partner_col: "j"}).reset_index(drop=True)
