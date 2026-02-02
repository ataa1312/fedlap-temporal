import torch.nn as nn
from src import *
from src.MLP.MLP_model import MLP
from src.GNN.GNN_models import GNN, DGCN


class ModelSpecs:
    def __init__(
        self,
        type="GNN",
        layer_sizes=[],
        heads=[],
        final_activation_function="linear",  # can be None, "layer", "batch", "instance"
        dropout=config.model.dropout,
        normalization=None,
        gnn_layer_type=config.model.gnn_layer_type,
        num_layers=None,
        residual=False,
        edge_dim=0,
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
