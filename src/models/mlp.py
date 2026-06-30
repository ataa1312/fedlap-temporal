import torch
import torch.nn as nn

from src.models.layer import Linear

__all__ = ["MLP"]


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        dims_inner: list[int] | None = None,
        batchnorm: bool = False,
        dropout: float = 0.0,
        act: str = "relu",
        final_act: bool = False,
    ) -> None:
        super().__init__()
        dims = [dim_in, *(dims_inner or []), dim_out]

        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            is_last = i == len(dims) - 2
            # By default the final layer is a bare Linear (logits). final_act=True
            # gives the last layer the full BN+act+dropout treatment too — matches
            # ROLAND's GNNPreMP (final_act=True) vs head MLP (bare final layer).
            full = (not is_last) or final_act
            modules.append(
                Linear(
                    dims[i],
                    dims[i + 1],
                    has_act=full,
                    batchnorm=batchnorm if full else False,
                    dropout=dropout if full else 0.0,
                    act=act,
                )
            )
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
