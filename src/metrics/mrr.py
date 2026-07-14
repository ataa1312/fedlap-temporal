import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.metrics.classification import _roc_auc, _average_precision

__all__ = ["compute_mrr", "compute_mrr_from_z", "compute_hard_auc_ap_from_z"]


def _sample_degree_negatives(unique_sources, pos_u, pos_v, deg, N, K, device):
    """K HARD negative targets per source, sampled proportional to node degree
    (popular nodes are structurally plausible targets, so the pair (u, v') is
    genuinely hard to reject — unlike a uniform-random v' which is trivially
    unrelated). Excludes the source's true positives, like the random sampler."""
    n_sources = unique_sources.size(0)
    probs = deg.float() + 1.0
    v_neg = torch.multinomial(probs, n_sources * K, replacement=True).view(n_sources, K).to(device)
    src_col = unique_sources.to(torch.long).unsqueeze(1)
    pos_key = pos_u.to(torch.long) * N + pos_v.to(torch.long)
    for _ in range(8):
        collide = torch.isin(src_col * N + v_neg, pos_key)
        n_collide = int(collide.sum())
        if n_collide == 0:
            break
        v_neg[collide] = torch.multinomial(probs, n_collide, replacement=True).to(device)
    return v_neg


@torch.no_grad()
def compute_hard_auc_ap_from_z(z, snap, K, device, model):
    """De-saturated discrimination: AUC/AP of true edges vs K degree-weighted HARD
    negatives per source (scored via the head), instead of the ~1:1 random-negative
    deepsnap set that saturates near 1.0. Returns (roc_auc, ap)."""
    pos_mask = snap.edge_label == 1.0
    if pos_mask.sum().item() == 0:
        return float("nan"), float("nan")
    pos_edges = snap.edge_label_index[:, pos_mask]
    u, v_pos = pos_edges[0], pos_edges[1]
    N = snap.num_nodes
    deg = torch.bincount(snap.edge_index.reshape(-1).to(device), minlength=N).float()
    unique_sources = torch.unique(u)
    v_neg = _sample_degree_negatives(unique_sources, u, v_pos, deg, N, K, device)
    src_rep = unique_sources.unsqueeze(1).expand(-1, K).flatten()
    v_neg_flat = v_neg.flatten()
    temp = snap.clone()
    temp.edge_label_index = torch.stack([torch.cat([u, src_rep]), torch.cat([v_pos, v_neg_flat])], dim=0)
    temp.edge_label = torch.zeros(temp.edge_label_index.size(1), device=device)
    scores, _ = model.decode(z, temp)
    labels = torch.cat([torch.ones(u.size(0), device=device),
                        torch.zeros(v_neg_flat.size(0), device=device)])
    return _roc_auc(scores, labels), _average_precision(scores, labels)


def _sample_filtered_negatives(
    unique_sources: torch.Tensor,
    pos_u: torch.Tensor,
    pos_v: torch.Tensor,
    N: int,
    K: int,
    device: str,
) -> torch.Tensor:
    """Sample K negative targets per source, excluding the source's true
    positives (roland's ``gen_negative_edges`` + ``edge_index_difference``).

    A sampled ``(u, v')`` that is actually a positive edge would score high and
    inflate the rank of the source's best positive, depressing MRR. Resample
    such collisions (bounded rounds; with N >> K this converges immediately).
    """
    n_sources = unique_sources.size(0)
    v_neg = torch.randint(0, N, (n_sources, K), device=device)
    src_col = unique_sources.to(torch.long).unsqueeze(1)
    pos_key = pos_u.to(torch.long) * N + pos_v.to(torch.long)
    for _ in range(8):
        collide = torch.isin(src_col * N + v_neg, pos_key)
        n_collide = int(collide.sum())
        if n_collide == 0:
            break
        v_neg[collide] = torch.randint(0, N, (n_collide,), device=device)
    return v_neg


@torch.no_grad()
def compute_mrr(
    model: nn.Module,
    snap: Data,
    hs: list[torch.Tensor] | None,
    K: int,
    method: str,
    is_recurrent: bool,
    device: str,
) -> float:
    """Per-source MRR following roland's convention.

    For each unique source node ``u`` that has at least one positive edge in
    ``snap.edge_label_index``:

    1. Compute scores of all positives ``(u, v_i)``.
    2. Aggregate via ``method`` in {``'min'``, ``'max'``, ``'mean'``} → ``p_star``.
    3. Sample ``K`` random target nodes ``v'_1, ..., v'_K``.
    4. Score the ``K`` "negative" pairs ``(u, v'_k)``.
    5. Rank ``p_star`` against those K scores: ``rank = #{k : score_k >= p_star} + 1``.
    6. Reciprocal rank ``RR_u = 1 / rank``.

    Final MRR is the mean of ``RR_u`` over unique sources.

    This is the end-to-end wrapper. If you already have node embeddings ``z``,
    call :func:`compute_mrr_from_z` to skip the encode step.
    """
    model.eval()

    pos_mask = snap.edge_label == 1.0
    if pos_mask.sum().item() == 0:
        return float("nan")

    pos_edges = snap.edge_label_index[:, pos_mask]
    u = pos_edges[0]
    v_pos = pos_edges[1]
    N = snap.num_nodes

    unique_sources = torch.unique(u)
    n_sources = unique_sources.size(0)
    v_neg = _sample_filtered_negatives(unique_sources, u, v_pos, N, K, device)

    src_repeated = unique_sources.unsqueeze(1).expand(-1, K).flatten()
    v_neg_flat = v_neg.flatten()

    all_u = torch.cat([u, src_repeated])
    all_v = torch.cat([v_pos, v_neg_flat])

    temp = snap.clone()
    temp.edge_label_index = torch.stack([all_u, all_v], dim=0)
    temp.edge_label = torch.zeros(all_u.size(0), device=device)

    if is_recurrent:
        scores, _, _ = model(temp, hs)
    else:
        scores, _ = model(temp)

    return _rank_and_aggregate(scores, u, unique_sources, n_sources, K, method)


@torch.no_grad()
def compute_mrr_from_z(
    z: torch.Tensor,
    snap: Data,
    K: int,
    method: str,
    device: str,
    model: nn.Module,
) -> float:
    """Same as :func:`compute_mrr` but takes precomputed embeddings ``z``.

    Only the head's ``decode(z, data)`` runs — no GNN forward. Use this from
    orchestrators that have already encoded the snapshot for an eval-loss
    computation.
    """
    pos_mask = snap.edge_label == 1.0
    if pos_mask.sum().item() == 0:
        return float("nan")

    pos_edges = snap.edge_label_index[:, pos_mask]
    u = pos_edges[0]
    v_pos = pos_edges[1]
    N = snap.num_nodes

    unique_sources = torch.unique(u)
    n_sources = unique_sources.size(0)
    v_neg = _sample_filtered_negatives(unique_sources, u, v_pos, N, K, device)

    src_repeated = unique_sources.unsqueeze(1).expand(-1, K).flatten()
    v_neg_flat = v_neg.flatten()

    all_u = torch.cat([u, src_repeated])
    all_v = torch.cat([v_pos, v_neg_flat])

    temp = snap.clone()
    temp.edge_label_index = torch.stack([all_u, all_v], dim=0)
    temp.edge_label = torch.zeros(all_u.size(0), device=device)

    scores, _ = model.decode(z, temp)

    return _rank_and_aggregate(scores, u, unique_sources, n_sources, K, method)


def _rank_and_aggregate(
    scores: torch.Tensor,
    u: torch.Tensor,
    unique_sources: torch.Tensor,
    n_sources: int,
    K: int,
    method: str,
) -> float:
    n_pos = u.size(0)
    pos_scores_flat = scores[:n_pos]
    neg_scores_per_src = scores[n_pos:].view(n_sources, K)

    rrs: list[torch.Tensor] = []
    for i in range(n_sources):
        src = unique_sources[i]
        src_pos = pos_scores_flat[u == src]

        if method == "max":
            p_star = src_pos.max()
        elif method == "min":
            p_star = src_pos.min()
        elif method == "mean":
            p_star = src_pos.mean()
        else:
            raise ValueError(f"unknown mrr_method: {method!r}")

        rank = (neg_scores_per_src[i] >= p_star).sum() + 1
        rrs.append(1.0 / rank.float())

    return torch.stack(rrs).mean().item()
