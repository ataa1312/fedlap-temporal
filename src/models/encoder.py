import torch
import torch.nn as nn

__all__ = ["NodeEncoder", "EdgeEncoder"]


class _LinearEncoder(nn.Module):
    """Linear (+ optional BN) feature encoder run before message passing.

    Port of ROLAND's Linear*Encoder: maps a raw feature ``[N, dim_in]`` to
    ``[N, dim_out]`` via a learned Linear, with optional BatchNorm. For UCI
    there is no sinusoidal/integer encoding — just the linear map.
    """

    def __init__(self, dim_in: int, dim_out: int, *, batchnorm: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out) if batchnorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class NodeEncoder(_LinearEncoder):
    """Encodes raw node features before pre-MP (ROLAND node encoder).

    Off for UCI (all-ones node features); for datasets with real node features.
    """


class EdgeEncoder(_LinearEncoder):
    """Encodes raw edge features before message passing (ROLAND LinearEdgeEncoder)."""
