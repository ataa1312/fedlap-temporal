from torch_geometric.loader import NeighborLoader

from src import *
from src.utils.graph import Graph
from src.classifier import Classifier
from src.models.model_binders import (
    ModelBinder,
    ModelSpecs,
)


class FGNN(Classifier):
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.create_smodel()

    def create_smodel(self):
        gnn_layer_sizes = [
            self.graph.num_features
        ] + config.feature_model.gnn_layer_sizes
        mlp_layer_sizes = [gnn_layer_sizes[-1]] + [self.graph.num_classes]

        model_specs = [
            ModelSpecs(
                type="GNN",
                layer_sizes=gnn_layer_sizes,
                final_activation_function="linear",
                # final_activation_function="relu",
                normalization="layer",
                # normalization="batch",
            ),
            ModelSpecs(
                type="MLP",
                layer_sizes=mlp_layer_sizes,
                final_activation_function="linear",
                normalization=None,
            ),
        ]

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def get_embeddings(self):
        H = self.model(self.graph.x, self.graph.edge_index)
        return H
