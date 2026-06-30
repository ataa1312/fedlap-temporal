import torch
import torch.nn as nn
import torch.nn.functional as F
from registries import layers, activations

__all__ = ["GeneralLayer", "Linear"]


class GeneralLayer(nn.Module):
    def __init__(
        self,
        layer_type: str,
        dim_in: int,
        dim_out: int,
        *,
        has_act: bool = True,
        batchnorm: bool = True,
        dropout: float = 0.0,
        act: str = "relu",
    ) -> None:
        super().__init__()
        layer_cls = layers[layer_type]
        # #15: drop conv bias when BN follows (BN's affine bias absorbs it) — matches
        # ROLAND's `bias=not has_bn`.
        self.layer = layer_cls(dim_in, dim_out, bias=not batchnorm)
        self.bn = nn.BatchNorm1d(dim_out) if batchnorm else None
        self.act_fn = activations[act]() if has_act else None
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_weight is None:
            x = self.layer(x, edge_index)
        else:
            x = self.layer(x, edge_index, edge_weight)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Linear(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        has_act: bool = True,
        batchnorm: bool = False,
        dropout: float = 0.0,
        act: str = "relu",
    ) -> None:
        super().__init__()
        # #15: drop bias when BN follows (BN's affine bias absorbs it). The bare
        # final MLP layer (batchnorm=False) keeps bias — matches ROLAND.
        self.lin = nn.Linear(dim_in, dim_out, bias=not batchnorm)
        self.bn = nn.BatchNorm1d(dim_out) if batchnorm else None
        self.act_fn = activations[act]() if has_act else None
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
