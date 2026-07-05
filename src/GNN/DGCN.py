import torch
from torch_geometric.loader import NeighborLoader

from src import *
from src.GNN.sGNN import SGNNMaster, SClassifier
from src.utils.graph import AGraph
from src.classifier import Classifier
from src.models.model_binders import (
    ModelBinder,
    ModelSpecs,
)


class DGCN(Classifier):
    def __init__(
        self, graph: AGraph, hidden_layer_size=config.feature_model.DGCN_layer_sizes
    ):
        Classifier.__init__(self, graph)
        self.create_smodel(hidden_layer_size)
        # self.create_model(config.feature_model.DGCN_layer_sizes)

    def create_smodel(self, hidden_layer_size=[]):
        layer_sizes = (
            [self.graph.num_features] + hidden_layer_size + [self.graph.num_classes]
        )

        model_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=layer_sizes,
                final_activation_function="linear",
                normalization="layer",
            ),
        ]

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def get_SFV(self):
        return torch.matmul(self.graph.abar, self.graph.x)

    def get_embeddings(self):
        if self.model is not None:
            H = self.model(self.graph.x)
        else:
            H = self.graph.x
        if self.graph.abar.is_sparse and H.device.type == "mps":
            H = self.graph.abar.matmul(H.cpu()).to(device)
        else:
            H = torch.matmul(self.graph.abar, H)
        return H

    def __call__(self):
        return DGCN.get_embeddings(self)


class SpectralDGCN(DGCN):
    def __init__(
        self,
        graph: AGraph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)

    def get_embeddings(self):
        H = self.graph.x
        if self.graph.abar.is_sparse and H.device.type == "mps":
            H = self.graph.abar.matmul(H.cpu()).to(device)
        else:
            H = torch.matmul(self.graph.abar, H)

        if self.model is not None:
            H = self.model(H)
        return H

    def __call__(self):
        return SpectralDGCN.get_embeddings(self)


class SDGCN(DGCN, SClassifier):
    def __init__(
        self,
        graph: AGraph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        DGCN.__init__(self, graph, hidden_layer_size)

    def get_embeddings(self):
        return DGCN.get_embeddings(self)

    def get_SFV(self):
        return DGCN.get_SFV(self)

    # def __call__(self):
    #     return DGCN.__call__(self)


class SDGCNMaster(SGNNMaster):
    def __init__(
        self,
        graph: AGraph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph)
        self.GNN_structure_embedding = None
        self.create_smodel(hidden_layer_size)

    def create_smodel(self, hidden_layer_size=[]):
        layer_sizes = (
            [self.graph.num_features] + hidden_layer_size + [self.graph.num_classes]
        )

        model_specs = [
            ModelSpecs(
                type="MLP",
                layer_sizes=layer_sizes,
                final_activation_function="linear",
                normalization="layer",
            ),
        ]

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def get_embeddings(self, node_ids):
        if self.GNN_structure_embedding is None:
            H = self.model(self.graph.x)
            if self.graph.abar.is_sparse and H.device.type == "mps":
                self.GNN_structure_embedding = self.graph.abar.matmul(H.cpu()).to(
                    device
                )
            else:
                self.GNN_structure_embedding = torch.matmul(self.graph.abar, H)
        return self.GNN_structure_embedding[node_ids]
