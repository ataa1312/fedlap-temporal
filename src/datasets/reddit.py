from pathlib import Path

import numpy as np
import pandas as pd
import torch
from registries import datasets
from config.registry import Registry
from src.datasets._temporal import make_snapshots
from torch_geometric.data import Data

__all__ = ["load_reddit_body", "load_reddit_title"]


# SNAP soc-redditHyperlinks (body/title): tab-separated subreddit hyperlinks,
# `SOURCE TARGET POST_ID TIMESTAMP LINK_SENTIMENT PROPERTIES`. Node IDs are
# subreddit names (ordinal-encoded over the sorted union). Node feature is a
# 300-D subreddit embedding (web-redditEmbeddings); missing subreddits get the
# global mean. Edge feature is [scaled_ts, sentiment, *86 properties] (88-D).
_RAW = {
    "reddit_body": "reddit-raw/soc-redditHyperlinks-body.tsv",
    "reddit_title": "reddit-raw/soc-redditHyperlinks-title.tsv",
}
_NODE_EMB = "reddit-raw/web-redditEmbeddings-subreddits.csv"
_EMB_DIM = 300


def _node_features(emb_path: Path, mapping: dict[str, int]) -> torch.Tensor:
    df_emb = pd.read_csv(emb_path, header=None, index_col=0)
    df_emb = df_emb[~df_emb.index.duplicated()]
    x = torch.full((len(mapping), _EMB_DIM), float(df_emb.values.mean()))
    present = [s for s in df_emb.index if s in mapping]
    rows = [mapping[s] for s in present]
    x[rows] = torch.tensor(df_emb.loc[present].values, dtype=torch.float32)
    return x


def _load_reddit(config: Registry, key: str) -> list[Data]:
    base = Path(config["dataset"]["path"])
    path = base / _RAW[key]
    if not path.is_file():
        raise FileNotFoundError(f"Reddit raw file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    # Ordinal-encode subreddit names over the sorted union of src/dst (ROLAND).
    subreddits = np.sort(
        pd.unique(df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].to_numpy().ravel())
    )
    mapping = {name: i for i, name in enumerate(subreddits)}
    src = df["SOURCE_SUBREDDIT"].map(mapping).to_numpy()
    dst = df["TARGET_SUBREDDIT"].map(mapping).to_numpy()
    num_nodes = len(subreddits)

    x = _node_features(base / _NODE_EMB, mapping)

    # Unix-second timestamps for snapshot slicing.
    ts = pd.to_datetime(df["TIMESTAMP"], format="%Y-%m-%d %H:%M:%S")
    ts = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    edge_time = torch.tensor(ts.to_numpy(), dtype=torch.float64)

    # 88-D edge feature: [scaled_ts in [0, 2], sentiment, *86 properties].
    t_min, t_max = edge_time.min(), edge_time.max()
    scaled_ts = 2.0 * (edge_time - t_min) / (t_max - t_min + 1e-9)
    sentiment = torch.tensor(df["LINK_SENTIMENT"].to_numpy(), dtype=torch.float32)
    props = np.array([s.split(",") for s in df["PROPERTIES"].values], dtype=np.float32)
    edge_attr = torch.cat(
        [scaled_ts.float().view(-1, 1), sentiment.view(-1, 1), torch.from_numpy(props)],
        dim=1,
    )

    g_all = Data(
        x=x,
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=edge_attr,
        edge_time=edge_time.float(),
        num_nodes=num_nodes,
    )

    if not config["dataset"]["snapshot"]:
        return [g_all]

    snapshots = make_snapshots(g_all, config["dataset"]["snapshot_freq"])
    min_edges = (
        2 if config["dataset"]["split_method"] == "chronological_temporal" else 10
    )
    return [g for g in snapshots if g.num_edges >= min_edges]


@datasets.register("reddit_body", eager=False)
def load_reddit_body(config: Registry) -> list[Data]:
    return _load_reddit(config, "reddit_body")


@datasets.register("reddit_title", eager=False)
def load_reddit_title(config: Registry) -> list[Data]:
    return _load_reddit(config, "reddit_title")
