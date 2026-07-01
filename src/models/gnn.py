from typing import Any

import torch
import torch.nn as nn

from configs.registry import Registry
from src.models.mlp import MLP
from src.models.weight_init import init_weights
from registries import heads, models, stages

__all__ = ["GNN"]


@models.register("gnn", eager=False)
class GNN(nn.Module):
    def __init__(self, config: Registry, dim_in: int, dim_out: int) -> None:
        super().__init__()
        gnn_cfg = config["gnn"]
        model_cfg = config["model"]
        task = config["dataset"]["task"]

        # ----- pre-MP linear stack ----- #
        d = dim_in
        dims_pre_mp = gnn_cfg["dims_pre_mp"]
        if dims_pre_mp:
            self.pre_mp = MLP(
                d,
                dims_pre_mp[-1],
                dims_inner=dims_pre_mp[:-1],
                batchnorm=gnn_cfg["batchnorm"],
                dropout=gnn_cfg["dropout"],
                act=gnn_cfg["act"],
                final_act=True,         # #10: ROLAND GNNPreMP applies BN+act on the last pre-MP layer
            )
            d = dims_pre_mp[-1]
        else:
            self.pre_mp = None

        # ----- MP stage (stack / skipsum / skipconcat) ----- #
        dims = gnn_cfg["dims"]
        stage_cls = stages[gnn_cfg["stage_type"]]
        self.stage = stage_cls(
            layer_type=gnn_cfg["layer_type"],
            dim_in=d,
            dim_out=dims[-1],
            dims_inner=dims[:-1],
            mode=gnn_cfg["stage_type"],   # consumed by skip stages, ignored by stack
            skip_every=gnn_cfg["skip_every"],
            batchnorm=gnn_cfg["batchnorm"],
            dropout=gnn_cfg["dropout"],
            act=gnn_cfg["act"],
            l2norm=gnn_cfg["l2norm"],
        )
        d = self.stage.dim_out

        # ----- task head ----- #
        head_cls = heads[task]
        head_kwargs: dict[str, Any] = {
            "dims_inner": gnn_cfg["dims_post_mp"],
            # #16: ROLAND's head MLP intermediates are GeneralLayer-wrapped (BN+act).
            "batchnorm": gnn_cfg["batchnorm"],
            "act": gnn_cfg["act"],
        }
        if task in ("edge", "link_pred"):
            head_kwargs["decoding"] = model_cfg["edge_decoding"]
        self.head = head_cls(d, dim_out, **head_kwargs)

        # ROLAND-style weight init (Xavier-uniform on Linear, BN weight=1/bias=0).
        self.apply(init_weights)

    def encode(self, data: Any) -> torch.Tensor:
        """Run pre-MP + MP stage. Returns node embeddings ``z``."""
        x = data.x
        edge_index = data.edge_index

        if self.pre_mp is not None:
            x = self.pre_mp(x)

        return self.stage(x, edge_index)

    def decode(self, z: torch.Tensor, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the task head to precomputed embeddings ``z``."""
        return self.head(z, data)

    def forward(self, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode(self.encode(data), data)
