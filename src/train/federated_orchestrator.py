import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from src import config
from registries import optimizers, schedulers
from src.metrics.mrr import compute_mrr_from_z
from src.metrics.classification import binary_classification_metrics


class _forked_rng:
    """Context manager: run the body under RNGs seeded with `seed` (torch
    CPU/MPS, python random, numpy — PyG internals may consume any of them),
    then restore all prior states so the surrounding run stream continues
    exactly as if the body had consumed nothing."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __enter__(self) -> None:
        self.cpu_state = torch.get_rng_state()
        self.mps_state = (
            torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
        )
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed % 2**32)

    def __exit__(self, *exc: object) -> None:
        torch.set_rng_state(self.cpu_state)
        if self.mps_state is not None:
            torch.mps.set_rng_state(self.mps_state)
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)


# --------------------------------------------------------------------- #
# Step helpers — operate on (snap_today, snap_tomorrow) pairs.
# ROLAND's forecast task: message-pass on G_t, predict edges of G_{t+1}.
# --------------------------------------------------------------------- #


def _model_forward(
    model: nn.Module,
    snap: Data,
    hs: list[torch.Tensor] | None,
    is_recurrent: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
    """Unified forward. Returns ``(pred, label, new_hs)``. For non-recurrent
    models, ``new_hs`` passes through unchanged."""
    if is_recurrent:
        return model(snap, hs)
    pred, label = model(snap)
    return pred, label, hs


def _step_train_pair(
    model: nn.Module,
    snap_today: Data,
    snap_tomorrow: Data,
    hs: list[torch.Tensor] | None,
    loss_fn: Callable,
    optimizer: Optimizer,
    device: str,
    is_recurrent: bool,
    pos_split: str = "train",
    prepared_snap: Data | None = None,
) -> tuple[float, list[torch.Tensor] | None]:
    model.train()
    # Default (prepared_snap=None) rebuilds the batch with fresh negatives on
    # every call — matching ROLAND's train_step, which re-derives the task
    # batch (and thus resamples negatives) each inner epoch.
    if prepared_snap is None:
        pos = _pos_for_split(snap_tomorrow, pos_split).to(device)
        prepared_snap = _attach_future_link_pred_labels(
            snap_today.to(device),
            snap_tomorrow.to(device),
            pos,
        )
    pred, label, new_hs = _model_forward(model, prepared_snap, hs, is_recurrent)
    loss = loss_fn(pred, label.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), new_hs


@torch.no_grad()
def _step_eval_with_mrr_pair(
    model: nn.Module,
    snap_today: Data,
    snap_tomorrow: Data,
    hs: list[torch.Tensor] | None,
    loss_fn: Callable,
    device: str,
    is_recurrent: bool,
    mrr_k: int,
    mrr_method: str,
    pos_split: str = "test",
) -> tuple[float, float, dict[str, float]]:
    """Forecast eval for the (today, tomorrow) task pair on the chosen split."""
    model.eval()
    pos = _pos_for_split(snap_tomorrow, pos_split).to(device)
    snap_prep = _attach_future_link_pred_labels(
        snap_today.to(device),
        snap_tomorrow.to(device),
        pos,
    )

    if is_recurrent:
        z, _ = model.encode(snap_prep, hs)
    else:
        z = model.encode(snap_prep)

    pred, label = model.decode(z, snap_prep)
    loss = loss_fn(pred, label.float()).item()
    mrr = compute_mrr_from_z(z, snap_prep, mrr_k, mrr_method, device, model)
    metrics = binary_classification_metrics(pred, label)
    return loss, mrr, metrics


@torch.no_grad()
def _step_eval_loss_pair(
    model: nn.Module,
    snap_today: Data,
    snap_tomorrow: Data,
    hs: list[torch.Tensor] | None,
    loss_fn: Callable,
    device: str,
    is_recurrent: bool,
    pos_split: str = "val",
    prepared_snap: Data | None = None,
) -> float:
    """BCE loss only — no MRR. Used by the inner loop's early-stopping check.

    If ``prepared_snap`` is provided, skip the label-attachment step and use
    it directly. Callers pass this to keep the val batch (positives +
    sampled negatives) frozen across an inner loop — otherwise negatives
    resample on every call and patience-based early stopping fires on the
    noise rather than on real model changes. When ``prepared_snap`` is given,
    ``snap_today`` / ``snap_tomorrow`` / ``pos_split`` are ignored.
    """
    model.eval()
    if prepared_snap is None:
        pos = _pos_for_split(snap_tomorrow, pos_split).to(device)
        prepared_snap = _attach_future_link_pred_labels(
            snap_today.to(device),
            snap_tomorrow.to(device),
            pos,
        )
    pred, label, _ = _model_forward(model, prepared_snap, hs, is_recurrent)
    return loss_fn(pred, label.float()).item()


@torch.no_grad()
def _refresh_hs(
    model: nn.Module,
    snap_today: Data,
    hs: list[torch.Tensor] | None,
    device: str,
    is_recurrent: bool,
) -> list[torch.Tensor] | None:
    """Re-encode snap_today with the current model to derive a hidden state
    that reflects the most-recently-loaded weights. Called after restoring
    `best_model` so the carry-forward hs doesn't reflect the wasted final
    train step. Non-recurrent models have no hs to refresh — returns None.
    """
    if not is_recurrent:
        return None
    # No model.eval() here — ROLAND's update_node_states inherits whatever
    # mode the inner loop ended in (eval after a patience break, train after
    # exhausting max_epoch). Forcing eval made BN normalize with running
    # stats instead of batch stats during the refresh, shifting the carried
    # state on every batchnorm=true config (the 82-92% rows).
    _, new_hs = model.encode(snap_today.to(device), hs)
    return new_hs


# --------------------------------------------------------------------- #
# Meta-learning: blend two state_dicts into a running W_init.
# --------------------------------------------------------------------- #


def _average_state_dict(
    d_old: dict[str, torch.Tensor],
    d_new: dict[str, torch.Tensor],
    w: float,
) -> dict[str, torch.Tensor]:
    """Return ``(1 - w) * d_old + w * d_new`` key-wise.

    Integer-typed buffers (e.g., BatchNorm's ``num_batches_tracked``) are not
    averaged — we copy from ``d_new`` instead, since blending integers with a
    float weight corrupts their dtype.
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"meta average weight w={w} not in [0, 1]")
    out: dict[str, torch.Tensor] = {}
    for key, old in d_old.items():
        new = d_new[key]
        if torch.is_floating_point(old):
            out[key] = (1.0 - w) * old.detach() + w * new.detach()
        else:
            out[key] = new.detach().clone()
    return out


# --------------------------------------------------------------------- #
# Future-link-prediction label attachment used by live_update.
# Message-pass on snap_today, predict snap_tomorrow's edges. Negatives are
# sampled against the union of both edge sets so we never draw a tomorrow-
# positive (or a today-positive — also not a "future" negative) as a "negative".
# --------------------------------------------------------------------- #


def _attach_future_link_pred_labels(
    snap_today: Data,
    snap_tomorrow: Data,
    pos_edge_index: torch.Tensor | None = None,
) -> Data:
    """Build the batch for live_update's forecast task.

    ``pos_edge_index``: explicit positive set for snap_tomorrow (e.g. the
    train/val/test split subset). Defaults to ``snap_tomorrow.edge_index``
    (all positives) — useful when no per-snapshot split is in play.

    Negatives are sampled against ALL of tomorrow's edges (any split), and
    nothing else — deepsnap 0.2.0 semantics: the forbidden set is the split
    graph's `edge_index ∪ edge_label_index`, which under ROLAND's
    edge_train_mode='all' is the whole snapshot. Today's edges are NOT
    excluded: negatives are sampled snapshot-locally within tomorrow, so
    yesterday's non-recurring edges are legitimate (hard) negatives.
    """
    snap = snap_today.clone()
    pos = pos_edge_index if pos_edge_index is not None else snap_tomorrow.edge_index
    n_pos = pos.size(1)
    if n_pos == 0:
        raise ValueError(
            "snap_tomorrow has no positive edges to predict — cannot build labels"
        )

    forbidden = snap_tomorrow.edge_index
    num_nodes = max(snap_today.num_nodes, snap_tomorrow.num_nodes)
    neg = negative_sampling(
        edge_index=forbidden,
        num_nodes=num_nodes,
        num_neg_samples=n_pos,
    )
    snap.edge_label_index = torch.cat([pos, neg], dim=1)
    snap.edge_label = torch.cat(
        [
            torch.ones(n_pos, device=pos.device),
            torch.zeros(neg.size(1), device=pos.device),
        ]
    )
    return snap


def _partition_edges_per_snapshot(
    snapshots: list[Data],
    ratios: list[float],
    seed: int,
) -> None:
    """Partition each snapshot's positive edges into train/val/test subsets.

    Mutates each snap in place, adding ``pos_train``, ``pos_val``, ``pos_test``
    long tensors of shape [2, n_i]. Rounding remainders go to test.

    The partition is deterministic given (seed, snap_index): each snapshot uses
    ``seed + snap_index`` to seed its own permutation, so two runs with the same
    seed produce identical splits.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(
            f"dataset.split must sum to 1.0, got {ratios} (sum={sum(ratios):.4f})"
        )
    r_train, r_val, _ = ratios
    for i, snap in enumerate(snapshots):
        n_edges = snap.edge_index.size(1)
        if n_edges == 0:
            empty = torch.empty(
                2, 0, dtype=snap.edge_index.dtype, device=snap.edge_index.device
            )
            snap.pos_train = empty
            snap.pos_val = empty
            snap.pos_test = empty
            continue
        n_train = int(n_edges * r_train)
        n_val = int(n_edges * r_val)
        g = torch.Generator(device=snap.edge_index.device).manual_seed(seed + i)
        perm = torch.randperm(n_edges, generator=g, device=snap.edge_index.device)
        shuffled = snap.edge_index[:, perm]
        snap.pos_train = shuffled[:, :n_train]
        snap.pos_val = shuffled[:, n_train : n_train + n_val]
        snap.pos_test = shuffled[:, n_train + n_val :]


def _precompute_keep_ratio(snapshots: list[Data], mode: str) -> None:
    """Attach a per-node ``keep_ratio`` tensor (shape ``[num_nodes, 1]``) to
    each snapshot for the moving_average embedding update.

    Port of ROLAND's ``precompute_edge_degree_info``: ``keep_ratio[t]`` is
    derived from the accumulated node degree in G[0..t-1] vs the degree in
    G[t], so first-seen nodes update fully and inactive nodes freeze. Degrees
    are counted over the full snapshot graph (``edge_index``), matching
    ROLAND's ``edge_train_mode='all'``. Mutates each snap in place.
    """
    from src.datasets._temporal import get_keep_ratio, node_degree

    num_nodes = snapshots[0].num_nodes
    existing = torch.zeros(num_nodes, device=snapshots[0].edge_index.device)
    for snap in snapshots:
        new = node_degree(snap.edge_index, num_nodes)
        ratio = get_keep_ratio(existing, new, mode).unsqueeze(-1)
        snap.keep_ratio = ratio
        snap.node_degree_new = new
        existing = existing + new


def _pos_for_split(snap: Data, split: str) -> torch.Tensor:
    attr = f"pos_{split}"
    if not hasattr(snap, attr):
        raise ValueError(
            f"snap missing attribute {attr!r}; call _partition_edges_per_snapshot "
            f"before _step_*_pair (live_update does this automatically)"
        )
    return getattr(snap, attr)


# --------------------------------------------------------------------- #
# Shared federated helpers: optimizer/scheduler builders (fedlap has no
# factories) + nested-state_dict clone + client-embedding stitch. The federated
# loop itself lives in src/dynamic_server.py::DynamicServer, which reuses the
# base Server FedAvg primitives (share_weights / sum_lod) for weight averaging.
# --------------------------------------------------------------------- #


def _make_optimizer(model: nn.Module) -> Optimizer:
    optim_cfg = config["optim"]
    cls = optimizers[optim_cfg["optimizer"]]
    params = filter(lambda p: p.requires_grad, model.parameters())
    kwargs: dict = {"lr": optim_cfg["base_lr"], "weight_decay": optim_cfg["weight_decay"]}
    if optim_cfg["optimizer"] == "sgd":
        kwargs["momentum"] = optim_cfg["momentum"]
    return cls(params, **kwargs)


def _make_scheduler(optimizer: Optimizer):
    optim_cfg = config["optim"]
    name = optim_cfg["scheduler"]
    if name == "none":
        return None
    cls = schedulers[name]
    if name == "steps":
        return cls(optimizer, milestones=optim_cfg["steps"], gamma=optim_cfg["lr_decay"])
    if name == "cos":
        return cls(optimizer, T_max=config["train"]["num_epochs"])
    raise ValueError(f"Unhandled scheduler: {name!r}")


def _clone_state(sd):
    """Deep-clone a nested state_dict. fedlap's federated protocol nests
    (model/head -> ModelBinder blocks -> tensors), so a flat clone won't do."""
    if isinstance(sd, dict):
        return {k: _clone_state(v) for k, v in sd.items()}
    return sd.detach().clone()


def _stitch_global_z(client_zs, client_node_ids, num_nodes, dim, device):
    """Scatter each client's local embeddings into a global [num_nodes, dim] tensor
    by global node id. Clients partition the node set (each node written once); a
    single client with node_ids=arange(N) makes this the identity."""
    global_z = torch.zeros(num_nodes, dim, device=device)
    for z_c, ids in zip(client_zs, client_node_ids):
        if z_c.numel() > 0:
            global_z[ids] = z_c
    return global_z
