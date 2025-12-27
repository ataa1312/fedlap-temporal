import torch

from src import *
from src.GNN.sGNN import SClassifier, SEdgeClassifier
from src.utils.graph import Graph

# from src.GNN.MC import estimate


# class SLaplace(SClassifier):
#     def __init__(
#         self,
#         graph: Graph,
#         hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
#     ):
#         super().__init__(graph, hidden_layer_size)
#         self.create_L()
#         self.old_x = deepcopy(self.graph.x.detach())
#         self.initial_x = deepcopy(self.graph.x.detach())

#     def regularizer(self):
#         s = 0
#         x = self.graph.x
#         delta_x = x - self.old_x
#         # delta_x = x - self.initial_x

#         delta_x_i = delta_x[self.directed_edges[0]]
#         delta_x_o = delta_x[self.directed_edges[1]]
#         old_x_i = self.old_x[self.directed_edges[0]]
#         old_x_o = self.old_x[self.directed_edges[1]]
#         initial_x_i = self.initial_x[self.directed_edges[0]]
#         initial_x_o = self.initial_x[self.directed_edges[1]]
#         x_i = x[self.directed_edges[0]]
#         x_o = x[self.directed_edges[1]]
#         s += (
#             torch.sum(
#                 # (old_x_i - old_x_o) ** 2
#                 # + (delta_x_i - delta_x_o) ** 2
#                 +(x_i - x_o)
#                 * (delta_x_i - 0)
#                 # +(initial_x_i - initial_x_o)
#                 # * (delta_x_i - 0)
#                 # -(old_x_i - old_x_o)
#                 # * (delta_x_o - 0)
#             )
#             / x.shape[1]
#         )
#         # s += (
#         #     torch.sum(
#         #         # (old_x_i - old_x_o) ** 2
#         #         # + (delta_x_i - delta_x_o) ** 2
#         #         +(x_i - x_o)
#         #         # +(old_x_i - old_x_o)
#         #         * (delta_x_i - 0)
#         #     )
#         #     / x.shape[1]
#         # )
#         # s += torch.sum((old_x_i + delta_x_i - old_x_o - delta_x_o) ** 2) / x.shape[1]
#         # s += (
#         #     torch.sum(
#         #         (initial_x_i - initial_x_o) ** 2
#         #         + (delta_x_i - delta_x_o) ** 2
#         #         + 2 * (initial_x_i - 0) * (delta_x_i - delta_x_o)
#         #     )
#         #     / x.shape[1]
#         # )
#         # s += (
#         #     torch.sum((initial_x_i + delta_x_i - initial_x_o - delta_x_o) ** 2)
#         #     / x.shape[1]
#         # )

#         # initial_x_i = self.initial_x[self.directed_edges[0]]
#         # initial_x_o = self.initial_x[self.directed_edges[1]]
#         # x_i = x[self.directed_edges[0]]
#         # delta_x_i = delta_x[self.directed_edges[0]]
#         # s += torch.sum((initial_x_i - initial_x_o) * (x_i - delta_x_i)) / x.shape[1]

#         self.old_x = deepcopy(self.graph.x.detach())

#         # r1 = torch.matmul(self.L, x)
#         # r = torch.matmul(x.T, r1)
#         # s = torch.trace(r) / torch.trace(torch.matmul(x.T, x))
#         # s = torch.trace(r) / x.shape[1]

#         return s


class LaplaceMixin:
    """
    Mixin that provides Laplace regularization functionality.
    Works with both node classification and edge prediction.
    """

    def intrinsic_regularizer(self):
        x = self.graph.x
        r1 = torch.matmul(self.graph.L, x)
        r = torch.matmul(x.T, r1)
        s = torch.trace(r) / torch.trace(torch.matmul(x.T, x))
        # s = torch.trace(r) / x.shape[1]

        return s


class SpectralLaplaceMixin:
    """
    Mixin that provides spectral Laplace functionality.
    Works with both node classification and edge prediction.
    """

    def get_SFV(self):
        W = self.get_W()
        return self.Q @ W

    def set_QD(self, Q, D):
        self.Q = Q
        self.D = D

    def get_W(self):
        return self.graph.x

    def intrinsic_regularizer(self):
        W = self.get_W()
        if len(self.D.shape) == 1:
            # r1 = torch.diag(1 / self.D) @ W
            r1 = torch.einsum("i,ij->ij", self.D, W)
            # r1 = torch.einsum(
            #     "i,ij->ij", (self.D) ** (config.structure_model.DGCN_layers), Q
            # )
        else:
            r1 = torch.einsum("ij,jk->ik", self.D, W)
        r = torch.matmul(W.T, r1)
        s = torch.trace(r) / torch.trace(torch.matmul(W.T, W))
        # s = torch.trace(r) / Q.shape[1]

        # return 1 / s
        return s

    def ambient_regularizer(self):
        H = self.graph.x
        return torch.trace(H @ H.T)

    def get_embeddings(self):
        W = self.get_W()
        Q = self.Q
        Z = Q @ W

        # x = self.graph.z
        # f = x.T @ Q
        # c = torch.softmax(f.T @ f, dim=1)
        # Z = Q @ c @ W
        # Z = self.Q @ torch.diag(self.D ** (0.5)) @ W
        torch.nn.functional.relu(Z, inplace=True)
        H = self.model(Z)
        return H


class SLaplace(LaplaceMixin, SClassifier):
    def __init__(
        self,
        graph: Graph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)
        self.graph.create_L()


class SpectralLaplace(SpectralLaplaceMixin, SClassifier):
    def __init__(
        self,
        graph: Graph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)


class LanczosLaplace(SpectralLaplace):
    def __init__(
        self,
        graph: Graph,
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, hidden_layer_size)


# Edge prediction analogues
class SEdgeLaplace(LaplaceMixin, SEdgeClassifier):
    def __init__(
        self,
        graph: Graph,
        link_feature_operator: LinkFeatureOperator = "hadamard",
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, link_feature_operator, hidden_layer_size)
        self.graph.create_L()


class SpectralEdgeLaplace(SpectralLaplaceMixin, SEdgeClassifier):
    def __init__(
        self,
        graph: Graph,
        link_feature_operator: LinkFeatureOperator = "hadamard",
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, link_feature_operator, hidden_layer_size)


class LanczosEdgeLaplace(SpectralEdgeLaplace):
    def __init__(
        self,
        graph: Graph,
        link_feature_operator: LinkFeatureOperator = "hadamard",
        hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
    ):
        super().__init__(graph, link_feature_operator, hidden_layer_size)


# class SLaplace(SClassifier):
#     def __init__(
#         self,
#         graph: Graph,
#         hidden_layer_size=config.structure_model.DGCN_structure_layers_sizes,
#     ):
#         super().__init__(graph, hidden_layer_size)
#         A, E = estimate(self.graph, self.graph.x.shape[0])
#         self.A = A + E
#         D = torch.sum(self.A, dim=1)
#         self.L = torch.diag(D) - self.A
#         # self.create_L()

#     def regularizer(self):
#         x = self.graph.x
#         r1 = torch.matmul(self.L, x)
#         r = torch.matmul(x.T, r1)
#         # s = torch.trace(r) / torch.trace(torch.matmul(x.T, x))
#         s = torch.trace(r) / x.shape[1]

#         return s

#     def get_grads(self, just_SFV=False):
#         grads = {}
#         if self.model is not None:
#             grads["model"] = self.model.get_grads()

#         return grads

#     def set_grads(self, grads):
#         if "model" in grads.keys():
#             self.model.set_grads(grads["model"])
