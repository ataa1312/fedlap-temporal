import re
from pathlib import Path

import pandas as pd
from registries import datasets
from config.registry import Registry
from src.datasets._temporal import build_full_graph, encode_node_ids, make_snapshots
from torch_geometric.data import Data

__all__ = ["load_as733"]


# SNAP AS-733: 733 daily snapshot files `asYYYYMMDD.txt` (BGP autonomous-system
# graphs), tab-separated `FromNodeId\tToNodeId` with `#` comment lines. Each
# undirected edge is stored twice (both directions) — kept as-is (directed).
# All-ones node features (AS_node_feature='one'); 1-D scaled-timestamp edge
# feature. Same shape as the UCI loader, so build_full_graph handles it.
_RAW_DIR = "as733-raw"
_DATE_RE = re.compile(r"as(\d{8})\.txt$")
_EPOCH = pd.Timestamp("1970-01-01")


def _load_as733(config: Registry) -> list[Data]:
    base = Path(config["dataset"]["path"])
    raw = base / _RAW_DIR
    files = sorted(raw.glob("as*.txt"))
    if not files:
        raise FileNotFoundError(f"AS-733 files not found under {raw}")

    frames = []
    for fp in files:
        m = _DATE_RE.search(fp.name)
        if m is None:
            continue
        ts = (pd.to_datetime(m.group(1), format="%Y%m%d") - _EPOCH) // pd.Timedelta("1s")
        df = pd.read_csv(fp, sep="\t", comment="#", names=["SRC", "DST"])
        df["TIMESTAMP"] = int(ts)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    num_nodes = encode_node_ids(df, ["SRC", "DST"])
    g_all = build_full_graph(df, num_nodes=num_nodes)

    if not config["dataset"]["snapshot"]:
        return [g_all]

    snapshots = make_snapshots(g_all, config["dataset"]["snapshot_freq"])
    min_edges = (
        2 if config["dataset"]["split_method"] == "chronological_temporal" else 10
    )
    return [g for g in snapshots if g.num_edges >= min_edges]


@datasets.register("as733", eager=False)
def load_as733(config: Registry) -> list[Data]:
    return _load_as733(config)
