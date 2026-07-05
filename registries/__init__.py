import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from config.registry import Registry

__all__ = [
    "models",
    "layers",
    "heads",
    "stages",
    "updaters",
    "activations",
    "losses",
    "optimizers",
    "schedulers",
    "datasets",
]


models = Registry("models")
layers = Registry("layers")
heads = Registry("heads")          # keyed by dataset.task
stages = Registry("stages")        # keyed by gnn.stage_type
updaters = Registry("updaters")    # keyed by gnn.embed_update_method (ROLAND)
activations = Registry("activations")  # keyed by gnn.act
losses = Registry("losses")
optimizers = Registry("optimizers")
schedulers = Registry("schedulers")
datasets = Registry("datasets")    # keyed by dataset.name


# -------- pre-registered built-ins -------- #

layers["gcnconv"] = GCNConv
layers["gatconv"] = GATConv
layers["sageconv"] = SAGEConv

# Stored as `nn.Module` *constructors*. Callers instantiate at use
# (`activations[name]()`) so stateful activations like PReLU register their
# learned parameter on the parent module.
activations["relu"] = nn.ReLU
activations["elu"] = nn.ELU
activations["leaky_relu"] = nn.LeakyReLU
activations["tanh"] = nn.Tanh
activations["sigmoid"] = nn.Sigmoid
activations["gelu"] = nn.GELU
activations["prelu"] = nn.PReLU

losses["cross_entropy"] = F.cross_entropy
losses["mse"] = F.mse_loss
losses["bce_with_logits"] = F.binary_cross_entropy_with_logits

optimizers["adam"] = optim.Adam
optimizers["adamw"] = optim.AdamW
optimizers["sgd"] = optim.SGD

schedulers["steps"] = sched.MultiStepLR
schedulers["cos"] = sched.CosineAnnealingLR
# 'none' is not stored — factories interpret it as "no scheduler".
