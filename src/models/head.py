from typing import Any

import torch
import torch.nn as nn

from src.models.mlp import MLP
from registries import heads

__all__ = ["GNNNodeHead", "GNNEdgeHead", "GNNGraphHead"]


@heads.register("node", eager=False)
class GNNNodeHead(nn.Module):
    """Node-level prediction head: MLP over node embeddings."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        dims_inner: list[int] | None = None,
        batchnorm: bool = False,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.post_mp = MLP(dim_in, dim_out, dims_inner=dims_inner, batchnorm=batchnorm, act=act)

    def forward(self, z: torch.Tensor, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.post_mp(z)
        label = data.y
        return pred, label


@heads.register("link_pred", eager=False)
@heads.register("edge", eager=False)
class GNNEdgeHead(nn.Module):
    """Edge / link-prediction head.

    Three decoding modes:
    - ``dot``: logit = <z_u, z_v>
    - ``cosine_similarity``: logit = cos(z_u, z_v)
    - ``concat``: logit = MLP([z_u || z_v])
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        decoding: str = "dot",
        dims_inner: list[int] | None = None,
        batchnorm: bool = False,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.decoding = decoding

        if decoding == "concat":
            # MLP eats the concatenated pair, no extra refinement on z first.
            self.post_mp = MLP(
                dim_in * 2, dim_out, dims_inner=dims_inner, batchnorm=batchnorm, act=act
            )
        else:
            # Refine z via an MLP before pairwise decode.
            if dim_out > 1:
                raise ValueError(
                    f"decoding={decoding!r} only supports binary link prediction; "
                    f"got dim_out={dim_out}"
                )
            self.post_mp = MLP(
                dim_in, dim_in, dims_inner=dims_inner, batchnorm=batchnorm, act=act
            )
            if decoding == "dot":
                self._decode = lambda u, v: (u * v).sum(dim=-1)
            elif decoding == "cosine_similarity":
                self._decode = nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError(f"Unknown edge decoding: {decoding!r}")

    def forward(self, z: torch.Tensor, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        edge_label_index = data.edge_label_index
        edge_label = data.edge_label

        if self.decoding == "concat":
            u = z[edge_label_index[0]]
            v = z[edge_label_index[1]]
            pred = self.post_mp(torch.cat([u, v], dim=-1)).squeeze(-1)
        else:
            z = self.post_mp(z)
            u = z[edge_label_index[0]]
            v = z[edge_label_index[1]]
            pred = self._decode(u, v)

        return pred, edge_label


@heads.register("graph", eager=False)
class GNNGraphHead(nn.Module):
    """Graph-level prediction head: pool node embeddings, then MLP."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        pooling: str = "add",
        dims_inner: list[int] | None = None,
        batchnorm: bool = False,
        act: str = "relu",
    ) -> None:
        super().__init__()
        from torch_geometric.nn import (
            global_add_pool,
            global_max_pool,
            global_mean_pool,
        )

        _POOL = {
            "add": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
        }
        if pooling not in _POOL:
            raise ValueError(f"Unknown pooling: {pooling!r}")
        self.pool = _POOL[pooling]
        self.post_mp = MLP(dim_in, dim_out, dims_inner=dims_inner, batchnorm=batchnorm, act=act)

    def forward(self, z: torch.Tensor, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        graph_emb = self.pool(z, data.batch)
        pred = self.post_mp(graph_emb)
        label = data.y
        return pred, label
