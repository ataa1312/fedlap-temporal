from src import *
from torch import nn
from src.classifier import Classifier, EdgeClassifier
from src.utils.graph import Graph
from src.GNN.custom_gat import CustomGAT
from torch_geometric.loader import NeighborLoader
from src.utils.config_parser import TrainModelConfig
from src.models.model_binders import (
    ModelSpecs,
    ModelBinder,
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


class NewFGNN(FGNN):
    def __init__(self, graph: Graph):
        super().__init__(graph)

    def create_smodel(self):
        gnn_layer_sizes = [
            self.graph.num_features  # pyright:ignore
        ] + config.feature_model.gnn_layer_sizes
        heads = config.feature_model.heads
        dropout = config.feature_model.dropout
        residual = config.feature_model.residual
        edge_dimension = config.feature_model.edge_dimension

        model_specs = [
            ModelSpecs(
                type="GNN",
                layer_sizes=gnn_layer_sizes,
                heads=heads,
                final_activation_function="none",
                dropout=dropout,
                normalization=None,
                gnn_layer_type=config.model.gnn_layer_type,
                residual=residual,
                edge_dim=edge_dimension,
            )
        ]

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

    def get_embeddings(self):
        H = self.model(self.graph.x, self.graph.edge_index, edge_attr=self.graph.edge_attr)  # pyright: ignore
        # H = self.model(self.graph.x, self.graph.edge_index)  # pyright: ignore
        return H


class FEdgeGNN(EdgeClassifier):
    def __init__(
        self, graph: Graph, link_feature_operator: LinkFeatureOperator = "hadamard"
    ):
        super().__init__(graph, link_feature_operator)
        self.create_smodel()

    def create_smodel(self):
        gnn_layer_sizes = [
            self.graph.num_features
        ] + config.feature_model.gnn_layer_sizes

        model_specs = [
            ModelSpecs(
                type="GNN",
                layer_sizes=gnn_layer_sizes,
                final_activation_function="linear",
                # final_activation_function="relu",
                normalization="layer",
                # normalization="batch",
            )
        ]

        self.model: ModelBinder = ModelBinder(model_specs)
        self.model.to(device)

        match self.link_feature_operator:
            case "hadamard" | "concat":
                if self.link_feature_operator == "hadamard":
                    logistic_regression_layer_sizes = [gnn_layer_sizes[-1], 1]
                else:
                    logistic_regression_layer_sizes = [2 * gnn_layer_sizes[-1], 1]

                logistic_regression_model_specs = [
                    ModelSpecs(
                        type="MLP",
                        dropout=0.0,
                        layer_sizes=logistic_regression_layer_sizes,
                        final_activation_function=None,  # pyright: ignore
                        normalization=None,
                    )
                ]
                self.logistic_regression_model = ModelBinder(
                    logistic_regression_model_specs
                )
                self.logistic_regression_model.to(device)
            case "dot-product":
                self.logistic_regression_model = None
            case _:
                raise NotImplementedError(
                    f"Operator {self.link_feature_operator} not implemented!"
                )

    def get_embeddings(self):
        H = self.model(self.graph.x, self.graph.message_passing_edge_index)
        return H
