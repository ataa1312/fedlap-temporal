from pathlib import Path

import numpy as np
import pandas as pd
from registries import datasets
from configs.registry import Registry
from src.datasets._temporal import build_full_graph, encode_node_ids, make_snapshots
from torch_geometric.data import Data

__all__ = ["load_bitcoin_otc", "load_bitcoin_alpha"]


# Bitcoin OTC / Alpha: comma-separated `SOURCE,TARGET,RATING,TIME`, with
# non-consecutive node IDs. Edge feature is [RATING, scaled_TIME] (2-D).
_RAW = {
    "bitcoin_otc": "bitcoin-otc-raw/soc-sign-bitcoinotc.csv",
    "bitcoin_alpha": "bitcoin-alpha-raw/soc-sign-bitcoinalpha.csv",
}


def _load_bitcoin(config: Registry, key: str) -> list[Data]:
    base = Path(config["dataset"]["path"])
    path = base / _RAW[key]
    if not path.is_file():
        raise FileNotFoundError(f"Bitcoin raw file not found: {path}")

    df = pd.read_csv(path, header=None, names=["SRC", "DST", "RATING", "TIMESTAMP"])
    # OTC timestamps carry decimals — round to whole seconds (ROLAND parity).
    df["TIMESTAMP"] = df["TIMESTAMP"].astype(np.int64).astype(np.float64)
    num_nodes = encode_node_ids(df, ["SRC", "DST"])

    g_all = build_full_graph(df, extra_edge_cols=["RATING"], num_nodes=num_nodes)

    if not config["dataset"]["snapshot"]:
        return [g_all]

    snapshots = make_snapshots(g_all, config["dataset"]["snapshot_freq"])
    min_edges = (
        2 if config["dataset"]["split_method"] == "chronological_temporal" else 10
    )
    return [g for g in snapshots if g.num_edges >= min_edges]


@datasets.register("bitcoin_otc", eager=False)
def load_bitcoin_otc(config: Registry) -> list[Data]:
    return _load_bitcoin(config, "bitcoin_otc")


@datasets.register("bitcoin_alpha", eager=False)
def load_bitcoin_alpha(config: Registry) -> list[Data]:
    return _load_bitcoin(config, "bitcoin_alpha")
