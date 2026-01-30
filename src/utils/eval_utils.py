import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.utils import is_attr_good, getLOGGER
from src import *

def get_link_features(
    edge_index: torch.Tensor,
    embeddings: torch.Tensor,
    operator: LinkFeatureOperator = "hadamard",
) -> torch.Tensor:
    """
    Compute link features for edges using embeddings.

    Args:
        edge_index: [2, num_edges] tensor of edge pairs
        embeddings: [num_nodes, embed_dim] tensor of node embeddings
        operator: Feature operator

    Returns:
        [num_edges, embed_dim] or [num_edges] tensor of link features (stays on same device)
    """
    match operator:
        case "hadamard":
            # Hadamard product: element-wise multiplication
            src_emb = embeddings[edge_index[0]]  # [num_edges, embed_dim]
            tgt_emb = embeddings[edge_index[1]]  # [num_edges, embed_dim]
            return src_emb * tgt_emb
        case "dot-product":
            # Dot product: inner multiplication
            src_emb = embeddings[edge_index[0]]  # [num_edges, embed_dim]
            tgt_emb = embeddings[edge_index[1]]  # [num_edges, embed_dim]
            return (src_emb * tgt_emb).sum(dim=-1, keepdim=True)  # [num_edges, 1]
        case "concat":
            # Concatenation: returns [num_edges, 2*embed_dim]
            src_emb = embeddings[edge_index[0]]  # [num_edges, embed_dim]
            tgt_emb = embeddings[edge_index[1]]  # [num_edges, embed_dim]
            return torch.cat([src_emb, tgt_emb], dim=-1)
        case _:
            raise NotImplementedError(f"Operator {operator} not implemented")



def sample_negative_edges(
    edge_index: torch.Tensor,
    num_pos_samples: int,
    num_neg_samples_per_pos: int,
    sampling_retries: int,
    num_nodes: int,
):
    row, col = edge_index
    unique_row = torch.unique(row)

    if unique_row.numel() > num_pos_samples:
        perm = torch.randperm(unique_row.numel())
        pos_nodes = unique_row[perm[:num_pos_samples]]
    else:
        pos_nodes = unique_row

    mask = torch.isin(edge_index[0], pos_nodes)
    pos_edge_index = edge_index[:, mask]

    edge_samples = {}

    for node in pos_nodes:
        node_val = node.item()

        node_mask = pos_edge_index[0] == node
        pos_targets = pos_edge_index[1, node_mask]

        if num_neg_samples_per_pos == -1:
            # Sample all nodes except positive targets and the node itself
            all_nodes_mask = torch.ones(num_nodes, device=edge_index.device, dtype=torch.bool)
            all_nodes_mask[pos_targets] = False
            all_nodes_mask[node] = False
            neg_tgt_nodes = torch.arange(num_nodes, device=edge_index.device)[all_nodes_mask]
        else:
            neg_tgt_nodes = torch.randint(
                num_nodes, size=(num_neg_samples_per_pos,), device=edge_index.device
            )

            counter = 0
            while counter < sampling_retries:
                mask = torch.isin(neg_tgt_nodes, pos_targets)
                if not mask.any():
                    break

                num_collisions = int(mask.sum())
                new_samples = torch.randint(
                    num_nodes, size=(num_collisions,), device=edge_index.device
                )
                neg_tgt_nodes[mask] = new_samples
                counter += 1
        edge_samples[node_val] = {"pos": pos_targets, "neg": neg_tgt_nodes}

    return edge_samples


def compute_ranking_metrics(
    edge_samples: dict[int, dict[str, torch.Tensor]],
    classifier: nn.Module,
    embeddings: torch.Tensor,
    operator: LinkFeatureOperator,
) -> dict[str, float]:
    """
    Compute ranking metrics (AP, MRR) for each node in the sample.

    Args:
        edge_samples: Output from sample_negative_edges
        classifier: Trained Link Classifier
        embeddings: Node embeddings
        operator: Feature operator

    Returns:
        Dictionary with "ranking_ap" and "mrr".
    """
    aps = []
    mrrs = []

    device = embeddings.device

    for node, samples in edge_samples.items():
        pos_targets = samples["pos"]  # [num_pos]
        neg_targets = samples["neg"]  # [num_neg]

        num_pos = pos_targets.shape[0]
        num_neg = neg_targets.shape[0]

        if num_pos == 0:
            continue

        src_pos = torch.full((num_pos,), node, dtype=torch.long, device=device)
        src_neg = torch.full((num_neg,), node, dtype=torch.long, device=device)

        pos_edges = torch.stack([src_pos, pos_targets], dim=0)
        neg_edges = torch.stack([src_neg, neg_targets], dim=0)

        pos_feats = get_link_features(pos_edges, embeddings, operator)
        neg_feats = get_link_features(neg_edges, embeddings, operator)

        with torch.no_grad():
            pos_scores = torch.sigmoid(classifier(pos_feats))
            neg_scores = torch.sigmoid(classifier(neg_feats))

        y_true = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
        y_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()

        if num_pos > 0 and num_neg > 0:
            ap = average_precision_score(y_true, y_scores)
            aps.append(ap)

        # 2. Mean Reciprocal Rank (MRR)
        # For each positive, rank it against ALL negatives
        # Rank = 1 + number of negatives with score > positive score
        
        # Expand dims for broadcasting: [num_pos, 1] vs [1, num_neg]
        rank_indices = (pos_scores.unsqueeze(1) < neg_scores.unsqueeze(0)).sum(1) + 1
        reciprocal_ranks = 1.0 / rank_indices.float()
        mrrs.append(reciprocal_ranks.mean().item())

    return {
        "ranking_ap": float(np.mean(aps)) if aps else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }

def get_metrics(labels, preds):
    return {
        "auc": roc_auc_score(labels, preds),
        "ap": average_precision_score(labels, preds),
    }
