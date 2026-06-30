import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

__all__ = [
    "build_full_graph",
    "encode_node_ids",
    "split_by_seconds",
    "split_by_calendar",
    "make_snapshots",
    "chronological_split",
    "node_degree",
    "get_keep_ratio",
]


_CALENDAR_FREQS = {"D": "%j", "W": "%W", "M": "%m"}


def build_full_graph(
    df: pd.DataFrame,
    *,
    extra_edge_cols: list[str] | None = None,
    num_nodes: int | None = None,
) -> Data:
    """Build one PyG ``Data`` from a (SRC, DST, TIMESTAMP) DataFrame.

    Matches ROLAND's ``load_single_dataset``:
    - All-ones node features (degree-based features would leak future info).
    - ``edge_time`` stores the raw unix timestamp for snapshot slicing.
    - Edge attribute = ``[*extra_edge_cols, MinMax-scaled timestamp in [0, 2]]``.
      With ``extra_edge_cols=None`` (UCI) it's the 1-D scaled timestamp; Bitcoin
      passes ``["RATING"]`` for the 2-D ``[rating, scaled_ts]`` feature.

    ``num_nodes`` defaults to ``max(id) + 1`` (valid only for dense 0-indexed
    IDs); pass it explicitly when IDs were remapped via :func:`encode_node_ids`.
    """
    edge_index = torch.from_numpy(df[["SRC", "DST"]].values.T).long()
    edge_time = torch.from_numpy(df["TIMESTAMP"].values.astype(np.float64)).float()
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    t_min, t_max = edge_time.min(), edge_time.max()
    scaled = 2.0 * (edge_time - t_min) / (t_max - t_min + 1e-9)
    if extra_edge_cols:
        extra = torch.from_numpy(df[extra_edge_cols].values.astype(np.float64)).float()
        edge_attr = torch.cat([extra, scaled.view(-1, 1)], dim=1)
    else:
        edge_attr = scaled.view(-1, 1)

    x = torch.ones(num_nodes, 1)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_time=edge_time,
        num_nodes=num_nodes,
    )


def encode_node_ids(df: pd.DataFrame, cols: list[str]) -> int:
    """Remap non-consecutive node IDs in ``cols`` to a dense ``[0, N)`` range.

    Mutates ``df`` in place and returns the node count ``N``. Ports ROLAND's
    ``OrdinalEncoder`` over the sorted union of source/target IDs (Bitcoin IDs
    are sparse; UCI IDs are already dense so it doesn't need this).
    """
    node_ids = np.sort(pd.unique(df[cols].to_numpy().ravel()))
    mapping = {int(nid): i for i, nid in enumerate(node_ids)}
    for c in cols:
        df[c] = df[c].map(mapping)
    return len(node_ids)


def split_by_seconds(g_all: Data, freq_sec: int) -> list[Data]:
    """Slice by ``edge_time // freq_sec`` bucket. EvolveGCN-style."""
    bucket = (g_all.edge_time // freq_sec).long()
    groups = torch.unique(bucket).sort().values
    return [_subset_edges(g_all, bucket == t) for t in groups]


def split_by_calendar(g_all: Data, freq: str) -> list[Data]:
    """Slice by calendar period: 'D' day-of-year, 'W' week-of-year, 'M' month."""
    freq = freq.upper()
    if freq not in _CALENDAR_FREQS:
        raise ValueError(
            f"Calendar freq must be one of {list(_CALENDAR_FREQS)}, got {freq!r}"
        )

    t_np = g_all.edge_time.numpy().astype(np.int64)
    df = pd.DataFrame(
        {"Timestamp": t_np, "Datetime": pd.to_datetime(t_np, unit="s")},
        index=range(len(t_np)),
    )
    df["Year"] = df["Datetime"].dt.strftime("%Y").astype(int)
    df["SubYear"] = df["Datetime"].dt.strftime(_CALENDAR_FREQS[freq]).astype(int)

    groups = df.groupby(["Year", "SubYear"]).indices
    periods = sorted(groups.keys())

    snapshots: list[Data] = []
    for p in periods:
        mask = torch.zeros(g_all.num_edges, dtype=torch.bool)
        mask[groups[p]] = True
        snapshots.append(_subset_edges(g_all, mask))
    return snapshots


def make_snapshots(g_all: Data, snapshot_freq: str) -> list[Data]:
    """Dispatch on freq format: 'D'/'W'/'M' for calendar, 'NNNs' for seconds."""
    if snapshot_freq.upper() in _CALENDAR_FREQS:
        return split_by_calendar(g_all, snapshot_freq)
    if snapshot_freq.endswith("s") and snapshot_freq[:-1].isdigit():
        return split_by_seconds(g_all, int(snapshot_freq[:-1]))
    raise ValueError(
        f"snapshot_freq must be 'D'/'W'/'M' or 'NNNs', got {snapshot_freq!r}"
    )


def chronological_split(
    snapshots: list[Data], ratios: list[float]
) -> list[list[Data]]:
    """Split a chronological snapshot list into chunks by ratio.

    Example: ``chronological_split(snaps, [0.7, 0.1, 0.2])`` yields
    ``[train_snaps, val_snaps, test_snaps]`` preserving order.

    Ratios must sum to 1.0 (within 1e-6). Returns one chunk per ratio.

    NOTE: This corresponds to roland's ``train.mode='standard'`` baseline —
    a static three-way split done in time order. It is *not* roland's
    headline ``live_update`` method, where the model is evaluated on each
    new snapshot before being fine-tuned on it.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    total = len(snapshots)
    chunks: list[list[Data]] = []
    prev = 0
    cum = 0.0
    for r in ratios[:-1]:
        cum += r
        cut = round(total * cum)        # round, not int — guards 0.7+0.1=0.7999...
        chunks.append(snapshots[prev:cut])
        prev = cut
    chunks.append(snapshots[prev:])
    return chunks


def node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Per-node degree counted over the source row ``edge_index[0]``.

    Matches roland's ``node_degree(..., mode='in')`` (which scatters over
    ``edge_index[0]``). Returns a float tensor of shape ``(num_nodes,)``.
    """
    deg = torch.zeros(num_nodes, device=edge_index.device)
    src = edge_index[0]
    deg.scatter_add_(0, src, torch.ones(src.size(0), device=edge_index.device))
    return deg


def get_keep_ratio(
    existing: torch.Tensor, new: torch.Tensor, mode: str = "linear"
) -> torch.Tensor:
    """Per-node keep ratio for the moving-average embedding update.

    ``state[v,t] = state[v,t-1]*keep_ratio + cand[v,t]*(1-keep_ratio)``.

    ``existing`` is the accumulated degree in G[0..t-1], ``new`` the degree in
    G[t]. A first-seen node (existing=0) gets keep_ratio 0 (full update); an
    inactive node (new=0) gets 1 (frozen). Port of roland's ``get_keep_ratio``.
    """
    if mode == "constant":
        ratio = torch.ones_like(existing)
        ratio[torch.logical_and(existing == 0, new > 0)] = 0
        ratio[torch.logical_and(existing > 0, new > 0)] = 0.5
    elif mode == "linear":
        ratio = existing / (existing + new + 1e-6)
    elif mode == "log":
        ratio = torch.log(existing + 1) / (torch.log(existing + 1) + new + 1e-6)
    elif mode == "sqrt":
        ratio = torch.sqrt(existing) / (torch.sqrt(existing) + new + 1e-6)
    else:
        raise ValueError(f"unknown keep_ratio mode {mode!r}")
    return ratio


def _subset_edges(g_all: Data, mask: torch.Tensor) -> Data:
    return Data(
        x=g_all.x,
        edge_index=g_all.edge_index[:, mask],
        edge_attr=g_all.edge_attr[mask],
        edge_time=g_all.edge_time[mask],
        num_nodes=g_all.num_nodes,
    )
