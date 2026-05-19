"""World Bank region mapping (default: matches Fabric/Steel/Choco notebooks)."""
import pandas as pd

REGION_COUNTRY_MAP = {
    "Sub-Saharan Africa (SSA)": [
        "Angola", "Botswana", "Burundi", "Comoros", "Dem. Rep. of the Congo", "Eritrea",
        "Eswatini", "Ethiopia", "Kenya", "Lesotho", "Madagascar", "Malawi", "Mauritius",
        "Mozambique", "Namibia", "Rwanda", "Sao Tome and Principe", "Seychelles", "Somalia",
        "South Africa", "South Sudan", "Sudan", "United Rep. of Tanzania", "Uganda", "Zambia",
        "Zimbabwe", "Benin", "Burkina Faso", "Cabo Verde", "Cameroon", "Central African Rep.",
        "Chad", "Congo", "Côte d'Ivoire", "Equatorial Guinea", "Gabon", "Gambia", "Ghana", "Guinea",
        "Guinea-Bissau", "Liberia", "Mali", "Mauritania", "Niger", "Nigeria", "Senegal",
        "Sierra Leone", "Togo",
    ],
    "East Asia & Pacific (EAP)": [
        "American Samoa", "Australia", "Brunei Darussalam", "Cambodia", "China", "Fiji",
        "French Polynesia", "Guam", "China, Hong Kong SAR", "Indonesia", "Japan", "Kiribati",
        "Dem. People's Rep. of Korea", "Rep. of Korea", "Lao People's Dem. Rep.", "China, Macao SAR",
        "Malaysia", "Marshall Isds", "FS Micronesia", "Mongolia", "Myanmar", "Nauru",
        "New Caledonia", "New Zealand", "N. Mariana Isds", "Palau", "Papua New Guinea",
        "Philippines", "Samoa", "Singapore", "Solomon Isds", "Thailand", "Timor-Leste", "Tonga",
        "Tuvalu", "Vanuatu", "Viet Nam",
    ],
    "Europe & Central Asia (ECA)": [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
        "Bosnia Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia",
        "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece", "Greenland", "Hungary",
        "Iceland", "Ireland", "Italy", "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania",
        "Luxembourg", "Rep. of Moldova", "Montenegro", "Netherlands", "North Macedonia", "Norway",
        "Poland", "Portugal", "Romania", "Russian Federation", "Serbia", "Slovakia", "Slovenia",
        "Spain", "Sweden", "Switzerland", "Tajikistan", "Turkmenistan", "Türkiye", "Ukraine",
        "United Kingdom", "Uzbekistan",
    ],
    "Latin America & Caribbean (LCR)": [
        "Antigua and Barbuda", "Argentina", "Aruba", "Bahamas", "Barbados", "Belize",
        "Bolivia (Plurinational State of)", "Brazil", "Br. Virgin Isds", "Cayman Isds", "Chile",
        "Colombia", "Costa Rica", "Cuba", "Dominica", "Dominican Rep.", "Ecuador", "El Salvador",
        "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua",
        "Panama", "Paraguay", "Peru", "Saint Maarten", "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago", "Turks and Caicos Isds",
        "Uruguay", "Venezuela",
    ],
    "Middle East & North Africa (MNA)": [
        "Algeria", "Bahrain", "Djibouti", "Egypt", "Iran", "Iraq", "Jordan", "Kuwait", "Lebanon",
        "Libya", "Malta", "Morocco", "Oman", "Qatar", "Saudi Arabia", "Syria", "Tunisia",
        "United Arab Emirates", "State of Palestine", "Yemen",
    ],
    "North America": ["Bermuda", "Canada", "USA"],
    "South Asia (SAR)": [
        "Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Pakistan", "Sri Lanka",
    ],
}

REGION_COLORS = {
    "East Asia & Pacific (EAP)":         "#D97706",
    "Europe & Central Asia (ECA)":       "#BE123C",
    "Latin America & Caribbean (LCR)":   "#22C55E",
    "Middle East & North Africa (MNA)":  "#9333EA",
    "North America":                     "#4B5563",
    "South Asia (SAR)":                  "#2563EB",
    "Sub-Saharan Africa (SSA)":          "#FACC15",
}


def build_country_region_table(code_df=None, region_country_map=None):
    """Return a country-to-region lookup table with ISO3 codes.

    Parameters
    ----------
    code_df : pandas.DataFrame or path-like
        BACI country-code table. It must contain the columns
        ``country_name`` and ``country_iso3``. Passing the path to
        ``BACIcountry.csv`` is also supported.
    region_country_map : dict, optional
        Mapping from region names to country-name lists. Defaults to
        ``REGION_COUNTRY_MAP``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``country_name``, ``region``, ``country_iso3``.

    Notes
    -----
    The package ships the World Bank region-to-country mapping, but it
    still needs ``BACIcountry.csv`` to translate country names into the
    ISO3 codes used by BACI/CEPII files.
    """
    if code_df is None:
        raise TypeError(
            "build_country_region_table() requires the BACI country-code "
            "table. Use, for example: code_df = pd.read_csv(COUNTRY_CODE_PATH); "
            "region_df = build_country_region_table(code_df)."
        )

    if not isinstance(code_df, pd.DataFrame):
        code_df = pd.read_csv(code_df)

    required = {"country_name", "country_iso3"}
    missing = required.difference(code_df.columns)
    if missing:
        raise ValueError(
            "code_df is missing required column(s): " + ", ".join(sorted(missing))
        )

    region_country_map = region_country_map or REGION_COUNTRY_MAP
    records = [(c, r) for r, cs in region_country_map.items() for c in cs]
    region_df = pd.DataFrame(records, columns=["country_name", "region"]).dropna()
    return region_df.merge(
        code_df[["country_name", "country_iso3"]], on="country_name", how="left"
    )
