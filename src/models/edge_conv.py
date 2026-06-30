from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, degree

from registries import activations, layers

__all__ = ["GeneralEdgeLayer", "ResidualEdgeConvLayer"]


class GeneralEdgeLayer(nn.Module):
    """Wrapper around an edge-aware conv: skip + BN + act + dropout.

    Parallel to :class:`models.layer.GeneralLayer` but for layers that
    consume ``edge_attr`` (those declaring ``EDGE_AWARE = True``). Skip
    handling is integrated here rather than in the conv itself.

    ``skip_connection``:
        - ``"none"``   — no skip.
        - ``"identity"`` — direct add (requires ``dim_in == dim_out``).
        - ``"affine"``   — learned ``nn.Linear(dim_in, dim_out)`` skip
          (ROLAND's "affine" skip from §4.4).

    ``layer_kwargs`` is forwarded to the conv constructor — that's how
    layer-specific options (``edge_dim``, ``msg_direction``, ``normalize``,
    ``agg``, ...) reach the underlying layer.
    """

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
        skip_connection: str = "none",
        layer_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        layer_cls = layers[layer_type]
        if not getattr(layer_cls, "EDGE_AWARE", False):
            raise ValueError(
                f"GeneralEdgeLayer requires an edge-aware layer "
                f"({layer_type!r} does not declare EDGE_AWARE = True); "
                f"use GeneralLayer instead."
            )
        # #15: drop conv bias when BN follows (BN absorbs it) — ROLAND `bias=not has_bn`.
        self.layer = layer_cls(dim_in, dim_out, bias=not batchnorm, **(layer_kwargs or {}))
        self.bn = nn.BatchNorm1d(dim_out) if batchnorm else None
        self.act_fn = activations[act]() if has_act else None
        self.dropout = dropout

        if skip_connection == "affine":
            self.skip = nn.Linear(dim_in, dim_out, bias=True)
        elif skip_connection == "identity":
            if dim_in != dim_out:
                raise ValueError(
                    f"skip_connection='identity' requires dim_in == dim_out; "
                    f"got {dim_in} != {dim_out}"
                )
            self.skip = nn.Identity()
        elif skip_connection == "none":
            self.skip = None
        else:
            raise ValueError(
                f"skip_connection={skip_connection!r}; expected one of "
                f"'none', 'identity', 'affine'"
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_in = x
        x = self.layer(x, edge_index, edge_attr=edge_attr)
        if self.skip is not None:
            x = x + self.skip(x_in)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ResidualEdgeConvLayer(MessagePassing):
    """ROLAND's edge-feature-aware GNN layer (port of
    roland/graphgym/contrib/layer/residual_edge_conv.py).

    Concatenates the edge feature into each message before a learned linear
    projection. Use ``msg_direction="both"`` to also concat the source node's
    own embedding into the message (richer but doubles the input dim).

    The residual skip from ROLAND lives in :class:`GeneralLayer` here, not in
    the conv itself — keeps skip handling uniform across layer types.
    """

    EDGE_AWARE = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        edge_dim: int,
        msg_direction: str = "single",
        normalize: bool = False,
        agg: str = "add",
        bias: bool = True,
    ) -> None:
        super().__init__(aggr=agg)
        if msg_direction not in {"single", "both"}:
            raise ValueError(
                f"msg_direction={msg_direction!r}; expected 'single' or 'both'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.msg_direction = msg_direction
        self.normalize = normalize

        msg_in = in_channels + edge_dim
        if msg_direction == "both":
            msg_in = in_channels * 2 + edge_dim
        self.linear_msg = nn.Linear(msg_in, out_channels, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError(
                "residual_edge_conv requires edge_attr; got None. "
                "Check that dataset loader produces snap.edge_attr."
            )
        if self.normalize:
            edge_index, norm = self._norm(edge_index, x.size(0))
            # Self-loops added by _norm have no edge features — pad with zeros.
            n_added = edge_index.size(1) - edge_attr.size(0)
            if n_added > 0:
                pad = edge_attr.new_zeros((n_added, edge_attr.size(1)))
                edge_attr = torch.cat([edge_attr, pad], dim=0)
        else:
            norm = None
        return self.propagate(edge_index, x=x, norm=norm, edge_feature=edge_attr)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        norm: Optional[torch.Tensor],
        edge_feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.msg_direction == "both":
            m = torch.cat([x_i, x_j, edge_feature], dim=-1)
        else:
            m = torch.cat([x_j, edge_feature], dim=-1)
        m = self.linear_msg(m)
        return norm.view(-1, 1) * m if norm is not None else m

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    @staticmethod
    def _norm(
        edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device, dtype=torch.float
        )
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
        )
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


layers["residual_edge_conv"] = ResidualEdgeConvLayer
