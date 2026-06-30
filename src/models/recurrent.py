from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.registry import Registry
from src.models.edge_conv import GeneralEdgeLayer
from src.models.encoder import EdgeEncoder, NodeEncoder
from src.models.layer import GeneralLayer
from src.models.mlp import MLP
from src.models.weight_init import init_weights
from registries import heads, layers, models, updaters

__all__ = ["GeneralRecurrentLayer", "RecurrentGNN"]


class GeneralRecurrentLayer(nn.Module):
    """Conv block (GeneralLayer or GeneralEdgeLayer) followed by an
    embedding updater. Dispatches based on whether the underlying conv
    is edge-aware. ``skip_connection`` and ``layer_kwargs`` are honored
    only on the edge-aware path (ROLAND's design — skip lives with the
    edge-aware layer)."""

    def __init__(
        self,
        layer_type: str,
        dim_in: int,
        dim_out: int,
        updater_name: str,
        *,
        batchnorm: bool = True,
        dropout: float = 0.0,
        act: str = "relu",
        skip_connection: str = "none",
        layer_kwargs: dict[str, Any] | None = None,
        updater_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.edge_aware = getattr(layers[layer_type], "EDGE_AWARE", False)
        if self.edge_aware:
            self.block = GeneralEdgeLayer(
                layer_type,
                dim_in,
                dim_out,
                has_act=True,
                batchnorm=batchnorm,
                dropout=dropout,
                act=act,
                skip_connection=skip_connection,
                layer_kwargs=layer_kwargs,
            )
        else:
            if skip_connection != "none":
                raise ValueError(
                    f"skip_connection={skip_connection!r} is only supported on "
                    f"edge-aware layers; got plain layer_type={layer_type!r}. "
                    f"Set skip_connection='none' or pick an EDGE_AWARE layer."
                )
            self.block = GeneralLayer(
                layer_type,
                dim_in,
                dim_out,
                has_act=True,
                batchnorm=batchnorm,
                dropout=dropout,
                act=act,
            )
        updater_cls = updaters[updater_name]
        self.updater = updater_cls(dim=dim_out, **(updater_kwargs or {}))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        keep_ratio: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.edge_aware:
            h_cand = self.block(x, edge_index, edge_attr=edge_attr)
        else:
            h_cand = self.block(x, edge_index)
        # keep_ratio is consumed only by MovingAverageUpdater (per-node,
        # degree-based blend); other updaters swallow it via **_.
        return self.updater(h_prev, h_cand, keep_ratio=keep_ratio, active_mask=active_mask)


@models.register("recurrent_gnn", eager=False)
class RecurrentGNN(nn.Module):
    def __init__(self, config: Registry, dim_in: int, dim_out: int) -> None:
        super().__init__()
        gnn_cfg = config["gnn"]
        model_cfg = config["model"]
        task = config["dataset"]["task"]

        # ----- node-feature encoder (ROLAND node encoder; off for UCI) ----- #
        # Symmetric to the edge encoder: Linear (+ optional BN) on raw node
        # features before pre-MP. Off by default (UCI has all-ones node features
        # → nothing to encode); enabled per dataset.node_encoder.
        ds_cfg = config["dataset"]
        d = dim_in
        if ds_cfg["node_encoder"]:
            self.node_encoder = NodeEncoder(
                dim_in,
                ds_cfg["node_encoder_dim"],
                batchnorm=ds_cfg["node_encoder_bn"],
            )
            d = ds_cfg["node_encoder_dim"]
        else:
            self.node_encoder = None

        # ----- pre-MP (same as static GNN) ----- #
        if gnn_cfg["layers_pre_mp"] > 0:
            inner = gnn_cfg["dim_inner"]
            self.pre_mp = MLP(
                d,
                inner,
                dims_inner=[inner] * (gnn_cfg["layers_pre_mp"] - 1),
                batchnorm=gnn_cfg["batchnorm"],
                dropout=gnn_cfg["dropout"],
                act=gnn_cfg["act"],
                final_act=True,         # #10: ROLAND GNNPreMP applies BN+act on the last pre-MP layer
            )
            d = inner
        else:
            self.pre_mp = None

        # ----- recurrent MP layers ----- #
        inner = gnn_cfg["dim_inner"]
        updater_kwargs: dict[str, Any] = {}
        if gnn_cfg["embed_update_method"] == "moving_average":
            # alpha lives in meta.alpha per ROLAND; fall back to 0.9.
            updater_kwargs["alpha"] = config["meta"]["alpha"]

        # Edge-feature encoder (ROLAND LinearEdgeEncoder, divergence #9): maps the
        # raw edge feature to edge_encoder_dim before message passing. When on,
        # the edge-aware conv consumes the encoded dim, not the raw one.
        raw_edge_dim = ds_cfg["edge_dim"]
        if ds_cfg["edge_encoder"] and raw_edge_dim > 0:
            self.edge_encoder = EdgeEncoder(
                raw_edge_dim,
                ds_cfg["edge_encoder_dim"],
                batchnorm=ds_cfg["edge_encoder_bn"],
            )
            effective_edge_dim = ds_cfg["edge_encoder_dim"]
        else:
            self.edge_encoder = None
            effective_edge_dim = raw_edge_dim

        # Per-layer-type constructor kwargs. residual_edge_conv needs to know
        # the edge feature dim and which message direction / aggregation to use.
        layer_type = gnn_cfg["layer_type"]
        layer_kwargs: dict[str, Any] = {}
        if layer_type == "residual_edge_conv":
            layer_kwargs = {
                "edge_dim": effective_edge_dim,
                "msg_direction": gnn_cfg["msg_direction"],
                "normalize": gnn_cfg["normalize_adj"],
                "agg": gnn_cfg["agg"],
            }

        skip_connection = gnn_cfg["skip_connection"]

        layers = []
        for i in range(gnn_cfg["layers_mp"]):
            layers.append(
                GeneralRecurrentLayer(
                    layer_type=layer_type,
                    dim_in=d if i == 0 else inner,
                    dim_out=inner,
                    updater_name=gnn_cfg["embed_update_method"],
                    batchnorm=gnn_cfg["batchnorm"],
                    dropout=gnn_cfg["dropout"],
                    act=gnn_cfg["act"],
                    skip_connection=skip_connection,
                    layer_kwargs=layer_kwargs,
                    updater_kwargs=updater_kwargs,
                )
            )
        self.mp_layers = nn.ModuleList(layers)
        self.l2norm = gnn_cfg["l2norm"]
        d = inner

        # ----- head ----- #
        head_cls = heads[task]
        layers_post_mp = gnn_cfg["layers_post_mp"]
        head_inner = (
            d * 2 if task in ("edge", "link_pred") and model_cfg["edge_decoding"] == "concat"
            else d
        )
        head_kwargs: dict[str, Any] = {
            "dims_inner": [head_inner] * (layers_post_mp - 1),
            # #16: ROLAND's head MLP intermediates are GeneralLayer-wrapped (BN+act).
            "batchnorm": gnn_cfg["batchnorm"],
            "act": gnn_cfg["act"],
        }
        if task in ("edge", "link_pred"):
            head_kwargs["decoding"] = model_cfg["edge_decoding"]
        self.head = head_cls(d, dim_out, **head_kwargs)

        # ROLAND applies init_weights to the whole model (Xavier-uniform on
        # Linear, BN weight=1/bias=0). Matches roland/.../gnn_recurrent.py:193.
        self.apply(init_weights)

    def encode(
        self,
        data: Any,
        hs: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run pre-MP + recurrent MP stack. Returns ``(z, new_hs)``."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        # Encode raw edge features once before message passing (ROLAND #9).
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        # Per-node degree-based blend weight for the moving_average updater;
        # attached by the orchestrator's _precompute_keep_ratio. None for other
        # updaters / datasets without it (updater falls back to scalar alpha).
        keep_ratio = getattr(data, "keep_ratio", None)
        active_mask = getattr(data, "node_degree_new", None)
        if active_mask is not None:
            active_mask = active_mask > 0

        if self.node_encoder is not None:
            x = self.node_encoder(x)
        if self.pre_mp is not None:
            x = self.pre_mp(x)

        new_hs: list[torch.Tensor] = []
        for i, layer in enumerate(self.mp_layers):
            h_prev = hs[i] if hs is not None else None
            x = layer(
                x,
                edge_index,
                h_prev,
                edge_attr=edge_attr,
                keep_ratio=keep_ratio,
                active_mask=active_mask,
            )
            new_hs.append(x)

        # L2-normalize the head input (ROLAND's GNNStackStage l2norm). Rebind x
        # to a new tensor so new_hs[-1] keeps pointing at the UN-normalized
        # hidden state — matching ROLAND, where the carried node_states is
        # un-normalized but the head sees the normalized embedding.
        if self.l2norm:
            x = F.normalize(x, p=2, dim=-1)

        return x, new_hs

    def decode(self, z: torch.Tensor, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the task head to precomputed embeddings ``z``."""
        return self.head(z, data)

    def forward(
        self,
        data: Any,
        hs: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        z, new_hs = self.encode(data, hs)
        pred, label = self.decode(z, data)
        return pred, label, new_hs
