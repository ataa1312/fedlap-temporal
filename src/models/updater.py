import torch
import torch.nn as nn
from registries import updaters
from src.models.mlp import MLP

__all__ = ["MovingAverageUpdater", "GRUUpdater", "MaskedGRUUpdater", "MLPUpdater"]


@updaters.register("moving_average", eager=False)
class MovingAverageUpdater(nn.Module):
    def __init__(self, dim: int, alpha: float = 0.9, **_: object) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        h_prev: torch.Tensor | None,
        h_cand: torch.Tensor,
        keep_ratio: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if h_prev is None:
            return h_cand
        # ROLAND uses a per-node, degree-derived keep_ratio (shape (N, 1)),
        # weighting the OLD state. The scalar `self.alpha` is only a fallback
        # for callers without precomputed degrees (e.g. toy tests); the real
        # live_update path threads keep_ratio in. Note: alpha is NOT meta.alpha
        # semantically — that's the meta weight-blend rate, kept separate.
        if keep_ratio is not None:
            return keep_ratio * h_prev + (1.0 - keep_ratio) * h_cand
        return self.alpha * h_prev + (1.0 - self.alpha) * h_cand


@updaters.register("gru", eager=False)
class GRUUpdater(nn.Module):
    def __init__(self, dim: int, **_: object) -> None:
        super().__init__()
        self.cell = nn.GRUCell(dim, dim)

    def forward(
        self, h_prev: torch.Tensor | None, h_cand: torch.Tensor, **_: object
    ) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros_like(h_cand)
        return self.cell(h_cand, h_prev)


@updaters.register("masked_gru", eager=False)
class MaskedGRUUpdater(nn.Module):
    def __init__(self, dim: int, **_: object) -> None:
        super().__init__()
        self.cell = nn.GRUCell(dim, dim)

    def forward(
        self,
        h_prev: torch.Tensor | None,
        h_cand: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros_like(h_cand)
        h_new = self.cell(h_cand, h_prev)
        if active_mask is None:
            return h_new
        return torch.where(active_mask.unsqueeze(-1), h_new, h_prev)


@updaters.register("mlp", eager=False)
class MLPUpdater(nn.Module):
    def __init__(
        self,
        dim: int,
        dims_inner: list[int] | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if dims_inner is None:
            dims_inner = [dim]              # default: 2-layer MLP with `dim` hidden
        self.mlp = MLP(dim * 2, dim, dims_inner=dims_inner)

    def forward(
        self, h_prev: torch.Tensor | None, h_cand: torch.Tensor, **_: object
    ) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros_like(h_cand)
        return self.mlp(torch.cat([h_cand, h_prev], dim=-1))
