import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layer import GeneralLayer
from registries import activations, stages

__all__ = ["GNNStackStage", "GNNSkipStage"]


@stages.register("stack", eager=False)
class GNNStackStage(nn.Module):
    def __init__(
        self,
        layer_type: str,
        dim_in: int,
        dim_out: int,
        *,
        dims_inner: list[int] | None = None,
        batchnorm: bool = True,
        dropout: float = 0.0,
        act: str = "relu",
        l2norm: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        dims = [dim_in, *(dims_inner or []), dim_out]

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            is_last = i == len(dims) - 2
            self.layers.append(
                GeneralLayer(
                    layer_type,
                    dims[i],
                    dims[i + 1],
                    has_act=not is_last,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    act=act,
                )
            )
        self.dim_out = dim_out
        self.l2norm = l2norm

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        if self.l2norm:
            x = F.normalize(x, p=2, dim=-1)
        return x


class _SkipBlock(nn.Module):
    def __init__(
        self,
        layer_type: str,
        dims: list[int],
        mode: str,
        *,
        batchnorm: bool = True,
        dropout: float = 0.0,
        act: str = "relu",
    ) -> None:
        super().__init__()
        if mode == "skipsum" and dims[0] != dims[-1]:
            raise ValueError(
                f"skipsum requires dims[0] == dims[-1] (got {dims[0]} vs {dims[-1]})"
            )
        self.mode = mode
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            is_last = i == len(dims) - 2
            self.layers.append(
                GeneralLayer(
                    layer_type,
                    dims[i],
                    dims[i + 1],
                    has_act=not is_last,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    act=act,
                )
            )
        self.act_fn = activations[act]()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            x = layer(x, edge_index)
        if self.mode == "skipsum":
            x = x + residual
        elif self.mode == "skipconcat":
            x = torch.cat([residual, x], dim=-1)
        else:
            raise ValueError(f"Invalid mode: {self.mode}!")
        return self.act_fn(x)


@stages.register("skipsum", eager=False)
@stages.register("skipconcat", eager=False)
class GNNSkipStage(nn.Module):
    def __init__(
        self,
        layer_type: str,
        dim_in: int,
        dim_out: int,
        *,
        dims_inner: list[int] | None = None,
        mode: str = "skipsum",
        skip_every: int = 1,
        batchnorm: bool = True,
        dropout: float = 0.0,
        act: str = "relu",
        l2norm: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        all_dims = [dim_in, *(dims_inner or []), dim_out]
        num_layers = len(all_dims) - 1
        if num_layers % skip_every != 0:
            raise ValueError(
                f"num_layers ({num_layers}) must be divisible by skip_every ({skip_every})"
            )

        self.blocks = nn.ModuleList()
        d_running = dim_in
        for i in range(num_layers // skip_every):
            block_start = i * skip_every
            block_end = block_start + skip_every
            # First layer of block consumes d_running (carries from previous block,
            # possibly concatenated). Remaining widths follow all_dims.
            block_dims = [d_running, *all_dims[block_start + 1 : block_end + 1]]

            if mode == "skipsum" and block_dims[-1] != d_running:
                raise ValueError(
                    f"skipsum block {i}: output dim ({block_dims[-1]}) "
                    f"must equal residual dim ({d_running})"
                )

            self.blocks.append(
                _SkipBlock(
                    layer_type,
                    block_dims,
                    mode,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    act=act,
                )
            )

            if mode == "skipconcat":
                d_running = d_running + block_dims[-1]
            # skipsum leaves d_running unchanged.

        self.dim_out = d_running
        self.l2norm = l2norm

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, edge_index)
        if self.l2norm:
            x = F.normalize(x, p=2, dim=-1)
        return x
