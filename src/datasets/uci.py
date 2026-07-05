from pathlib import Path

import pandas as pd
from registries import datasets
from config.registry import Registry
from src.datasets._temporal import make_snapshots, build_full_graph
from torch_geometric.data import Data

__all__ = ["load_uci"]


# CollegeMsg / UCI Messages: file is space-separated `SRC DST TIMESTAMP`,
# 1-indexed node IDs. SNAP format.
_RAW_PATH = "college-msg/college-msg.txt"


@datasets.register("uci", eager=False)
@datasets.register("college_msg", eager=False)
def load_uci(config: Registry) -> list[Data]:
    base = Path(config["dataset"]["path"])
    path = base / _RAW_PATH
    if not path.is_file():
        raise FileNotFoundError(f"CollegeMsg raw file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, names=["SRC", "DST", "TIMESTAMP"])
    df["SRC"] -= 1  # SNAP files use 1-based node IDs
    df["DST"] -= 1
    df = df.reset_index(drop=True)

    g_all = build_full_graph(df)

    if not config["dataset"]["snapshot"]:
        return [g_all]

    snapshots = make_snapshots(g_all, config["dataset"]["snapshot_freq"])

    # Filter tiny snapshots (ROLAND convention: 2 for chronological_temporal,
    # 10 for default 80/10/10 split).
    min_edges = (
        2 if config["dataset"]["split_method"] == "chronological_temporal" else 10
    )
    return [g for g in snapshots if g.num_edges >= min_edges]
