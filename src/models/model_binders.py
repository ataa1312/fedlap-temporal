import torch.nn as nn
from src import *
from src.MLP.MLP_model import MLP
from src.GNN.GNN_models import GNN, DGCN
from src.models.recurrent import GeneralRecurrentLayer
from src.models.encoder import NodeEncoder, EdgeEncoder
from src.models.mlp import MLP as RolandMLP


class ModelSpecs:
    def __init__(
        self,
        type="GNN",
        layer_sizes=[],
        heads=[],
        final_activation_function="linear",  # can be None, "layer", "batch", "instance"
        dropout=config["model"]["dropout"],
        normalization=None,
        gnn_layer_type=config["model"]["gnn_layer_type"],
        num_layers=None,
        residual=False,
        edge_dim=0,
        block_kwargs=None,
    ):
        self.type = type
        self.layer_sizes = layer_sizes
        self.heads = heads
        self.final_activation_function = final_activation_function
        self.dropout = dropout
        self.normalization = normalization
        self.gnn_layer_type = gnn_layer_type
        self.residual = residual
        self.edge_dim = edge_dim
        # ROLAND blocks (recurrent_layer / node_encoder / edge_encoder / roland_mlp):
        # dims come from layer_sizes ([in, ..., out]); block_kwargs holds the
        # block-specific constructor args (updater_name, skip_connection, ...).
        self.block_kwargs = block_kwargs or {}
        if num_layers is None:
            self.num_layers = len(self.layer_sizes) - 1
        else:
            self.num_layers = num_layers


class ModelBinder(nn.Module):
    def __init__(self, models_specs=[]):
        super().__init__()
        self.models_specs = models_specs

        self.models = self.create_models()

    def __getitem__(self, item):
        return self.models[item]

    def create_models(self):
        models = nn.ParameterList()
        model_propertises: ModelSpecs
        for model_propertises in self.models_specs:
            if model_propertises.type == "GNN":
                model = GNN(
                    layer_sizes=model_propertises.layer_sizes,
                    heads=model_propertises.heads,
                    last_layer=model_propertises.final_activation_function,
                    layer_type=model_propertises.gnn_layer_type,
                    dropout=model_propertises.dropout,
                    normalization=model_propertises.normalization,
                )
            elif model_propertises.type == "MLP":
                model = MLP(
                    layer_sizes=model_propertises.layer_sizes,
                    last_layer=model_propertises.final_activation_function,
                    dropout=model_propertises.dropout,
                    normalization=model_propertises.normalization,
                )
            elif model_propertises.type == "DGCN":
                model = DGCN(
                    num_layers=model_propertises.num_layers,
                    last_layer=model_propertises.final_activation_function,
                    aggr="mean",
                    a=0.0,
                )
            elif model_propertises.type == "recurrent_layer":
                sizes = model_propertises.layer_sizes
                model = GeneralRecurrentLayer(
                    dim_in=sizes[0],
                    dim_out=sizes[-1],
                    **model_propertises.block_kwargs,
                )
            elif model_propertises.type == "node_encoder":
                sizes = model_propertises.layer_sizes
                model = NodeEncoder(
                    sizes[0],
                    sizes[-1],
                    **model_propertises.block_kwargs,
                )
            elif model_propertises.type == "edge_encoder":
                sizes = model_propertises.layer_sizes
                model = EdgeEncoder(
                    sizes[0],
                    sizes[-1],
                    **model_propertises.block_kwargs,
                )
            elif model_propertises.type == "roland_mlp":
                sizes = model_propertises.layer_sizes
                model = RolandMLP(
                    sizes[0],
                    sizes[-1],
                    dims_inner=(list(sizes[1:-1]) or None),
                    **model_propertises.block_kwargs,
                )

            models.append(model)

        return models

    def reset_parameters(self) -> None:
        for model in self.models:
            model.reset_parameters()

    def state_dict(self):
        weights = {}
        for id, model in enumerate(self.models):
            weights[f"model{id}"] = model.state_dict()
        return weights

    def load_state_dict(self, weights: dict) -> None:
        for id, model in enumerate(self.models):
            model.load_state_dict(weights[f"model{id}"])

    def get_grads(self):
        model_parameters = list(self.parameters())
        grads = [parameter.grad for parameter in model_parameters]

        return grads

    def set_grads(self, grads):
        model_parameters = list(self.parameters())
        for grad, parameter in zip(grads, model_parameters):
            parameter.grad = grad

    def step(self, model, h, edge_index=None, edge_weight=None, edge_attr=None) -> None:
        if model.type_ == "MLP":
            return model(h)
        else:
            if isinstance(model, GNN) and model.layer_type == "custom-gat":
                return model(h, edge_index, edge_weight, edge_attr)
            return model(h, edge_index, edge_weight)

    def forward(self, x, edge_index=None, edge_weight=None, edge_attr=None):
        h = x
        for model in self.models:
            h = self.step(model, h, edge_index, edge_weight, edge_attr)
        return h

    def encode(self, x, edge_index, hs=None, edge_attr=None, keep_ratio=None,
               active_mask=None, inject=None):
        """State-aware forward for the ROLAND recurrent encoder stack.

        Dispatches on each block's spec ``type``: ``edge_encoder`` transforms
        ``edge_attr``; ``recurrent_layer`` threads the matching prior hidden state
        ``hs[r]`` and collects the new one; any other (stateless) block maps ``x``.
        Returns ``(z, new_hs)`` with ``z`` UN-normalized — the caller applies l2norm
        so the carried ``new_hs`` stays un-normalized, matching ROLAND's encode.
        Spec order must be [edge_encoder?, node_encoder?, pre_mp?, recurrent_layer...].
        """
        new_hs = []
        r = 0
        # `inject` goes to the LAST recurrent layer only. Per-layer injection would
        # need a projection per width and so extra parameters, which would confound a
        # positive result with capacity (the reading §10 already had to give SignNet).
        n_rec = sum(1 for sp in self.models_specs if sp.type == "recurrent_layer")
        if inject is not None and n_rec == 0:
            raise ValueError(
                "inject was supplied but this stack has no recurrent_layer to "
                "receive it; dropping it would run an injection arm that injects "
                "nothing"
            )
        for spec, model in zip(self.models_specs, self.models):
            if spec.type == "edge_encoder":
                if edge_attr is not None:
                    edge_attr = model(edge_attr)
            elif spec.type == "recurrent_layer":
                h_prev = hs[r] if hs is not None else None
                x = model(
                    x,
                    edge_index,
                    h_prev,
                    edge_attr=edge_attr,
                    keep_ratio=keep_ratio,
                    active_mask=active_mask,
                    inject=(inject if r == n_rec - 1 else None),
                )
                new_hs.append(x)
                r += 1
            else:
                x = model(x)
        return x, new_hs
