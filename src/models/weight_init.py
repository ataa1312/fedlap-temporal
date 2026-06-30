import torch.nn as nn

__all__ = ["init_weights"]


def init_weights(m: nn.Module) -> None:
    """ROLAND-style init (port of roland/graphgym/init.py).

    Xavier-uniform with relu gain on ``nn.Linear`` (bias zeroed); BatchNorm
    weight set to 1 and bias to 0. Apply recursively via ``model.apply(...)``.
    The relu gain is hardcoded to match ROLAND even when ``gnn.act`` differs
    (e.g. prelu). PyG's modern conv layers use their own ``Linear`` (not
    ``nn.Linear``) and keep their built-in init.
    """
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            m.bias.data.zero_()
