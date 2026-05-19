"""Network and GDP-vs-degree plots."""
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from .regions import REGION_COLORS


def plot_trade_network(df_links, df_nodes, col_i="i", col_j="j",
                       region_col="region", min_degree=3, figsize=(20, 12),
                       region_colors=None, anchor_spacing=6):
    """Clustered trade-network plot grouped by region."""
    region_colors = region_colors or REGION_COLORS
    df_links = df_links[df_links[col_i] != df_links[col_j]]
    G = nx.from_pandas_edgelist(df_links, source=col_i, target=col_j)
    nx.set_node_attributes(
        G, df_nodes.set_index("country_iso3")[region_col].to_dict(), name="region"
    )

    G = G.subgraph([n for n, d in G.degree() if d >= min_degree]).copy()

    regions = defaultdict(list)
    for n in G.nodes:
        regions[G.nodes[n].get("region", "Other")].append(n)

    anchors = {}
    for idx, region in enumerate(sorted(regions)):
        anchors[region] = np.array([(idx % 4) * anchor_spacing,
                                    -(idx // 4) * anchor_spacing])

    pos = {}
    for region, nodes in regions.items():
        cluster_pos = nx.circular_layout(G.subgraph(nodes), scale=2.0)
        anchor = anchors.get(region, np.zeros(2))
        for n, (x, y) in cluster_pos.items():
            pos[n] = anchor + np.array([x, y])

    node_colors = [region_colors.get(G.nodes[n].get("region", ""), "gray")
                   for n in G.nodes]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, alpha=0.95)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_gdp_vs_degree(df, id_col, gdp_col, direction="out", figsize=(12, 8)):
    """Scatter: country GDP (log-x) vs in/out degree, with country labels."""
    deg_name = "Out_Degree" if direction == "out" else "In_Degree"
    color = "blue" if direction == "out" else "green"

    deg = df.groupby(id_col).size().reset_index(name=deg_name)
    gdp = df[[id_col, gdp_col]].drop_duplicates()
    gdp.columns = [id_col, "GDP"]
    merged = (deg.merge(gdp, on=id_col)
                 .rename(columns={id_col: "Country_Code"}))
    merged = merged[merged["GDP"].apply(lambda x: pd.notnull(x) and x > 0)]

    plt.figure(figsize=figsize)
    plt.scatter(merged["GDP"], merged[deg_name], alpha=0.7, color=color)
    for _, row in merged.iterrows():
        plt.text(row["GDP"], row[deg_name], row["Country_Code"],
                 fontsize=8, ha="right")
    direction_label = "Outgoing" if direction == "out" else "Incoming"
    plt.title(f"Relationship between Country GDP and {direction_label} "
              f"Trade Links (Binary)")
    plt.xlabel("logGDP")
    plt.ylabel(f"{direction_label} Trade Links ({direction_label[:2]} Degree)")
    plt.xscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
