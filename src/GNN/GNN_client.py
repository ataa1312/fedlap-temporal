from copy import deepcopy
from torch_sparse import SparseTensor

from src import *
from src.GNN.laplace import LanczosLaplace, SLaplace, SpectralLaplace
from src.client import Client
from src.classifier import Classifier
from src.utils.graph import AGraph, Graph
from src.GNN.fGNN import FGNN
from src.GNN.DGCN import DGCN, SDGCN, SDGCNMaster, SpectralDGCN
from src.GNN.sGNN import SClassifier, SGNNMaster, SGNNSlave
from src.GNN.GNN_classifier import (
    FedDGCN,
    FedDGCNMaster,
    FedGNNMaster,
    FedLanczosLaplaceClassifier,
    FedLaplaceClassifier,
    FedMLPClassifier,
    FedSlave,
    FedSpectralDGCN,
    FedSpectralLaplaceClassifier,
)


class GNNClient(Client):
    def __init__(self, graph: Graph, id: int = 0):
        super().__init__(graph=graph, id=id, classifier_type="GNN")
        # LOGGER.info(f"Number of edges: {self.graph.num_edges}")
        self.classifier: Classifier | None = None
        self.SFVs = []
        self.cf_score_list = []

    def create_FDGCN_data(self) -> AGraph:
        A = create_adj(
            self.graph.edge_index,
            normalization="rw",
            self_loop=True,
            num_nodes=self.graph.num_nodes,
            # nodes=self.graph.node_ids,
        )
        abar = calc_abar(A, config.feature_model.DGCN_layers)

        graph = AGraph(
            abar=abar,
            # edge_index=self.graph.edge_index,
            x=self.graph.x,
            y=self.graph.y,
            node_ids=self.graph.node_ids,
            train_mask=self.graph.train_mask,
            val_mask=self.graph.val_mask,
            test_mask=self.graph.test_mask,
            num_classes=self.graph.num_classes,
        )
        return graph

    def create_SGNN_data(self, **kwargs) -> Graph:
        SFV = kwargs.get("SFV", None)
        SFV_ = torch.tensor(
            SFV.detach().cpu().numpy(),
            requires_grad=SFV.requires_grad,
            device=dev,
        )
        graph = Graph(
            x=SFV_,
            y=self.graph.y,
            edge_index=self.graph.get_edges(),
            node_ids=self.graph.node_ids,
            inter_edges=self.graph.inter_edges,
            external_nodes=self.graph.external_nodes,
            train_mask=self.graph.train_mask,
            val_mask=self.graph.val_mask,
            test_mask=self.graph.test_mask,
            num_classes=self.graph.num_classes,
        )
        return graph

    def create_SDGCN_data(self, **kwargs) -> AGraph:
        abar = kwargs.get("abar", None)
        abar_i = split_abar(abar, self.get_nodes())

        SFV = kwargs.get("SFV", None)
        SFV_ = torch.tensor(
            SFV.detach().cpu().numpy(),
            requires_grad=SFV.requires_grad,
            device=dev,
        )
        graph = AGraph(
            abar=abar_i,
            x=SFV_,
            y=self.graph.y,
            node_ids=self.graph.node_ids,
            train_mask=self.graph.train_mask,
            val_mask=self.graph.val_mask,
            test_mask=self.graph.test_mask,
            num_classes=self.graph.num_classes,
        )
        return graph

    def initialize(
        self,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type="feature",
        **kwargs,
    ) -> None:
        self.classifier = None
        if data_type == "feature":
            if fmodel_type == "GNN":
                self.classifier = FGNN(self.graph)
            else:
                graph = self.create_FDGCN_data()
                self.classifier = DGCN(graph)
        elif data_type == "structure":
            if smodel_type == "GNN":
                if self.id == "Server":
                    graph = self.create_SGNN_data(**kwargs)
                    self.classifier = SGNNMaster(graph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = SGNNSlave(self.graph, server_embedding_func)
            elif smodel_type == "DGCN":
                graph = self.create_SDGCN_data(**kwargs)
                self.classifier = SDGCN(graph)
            elif smodel_type in ["SpectralDGCN", "LanczosDGCN"]:
                graph = self.create_SDGCN_data(**kwargs)
                self.classifier = SpectralDGCN(graph)
            elif smodel_type == "CentralDGCN":
                if self.id == "Server":
                    graph = self.create_SDGCN_data(**kwargs)
                    self.classifier = SDGCNMaster(graph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = SGNNSlave(self.graph, server_embedding_func)
            elif smodel_type == "Laplace":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = SLaplace(sgraph)
            elif smodel_type == "SpectralLaplace":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = SpectralLaplace(sgraph)
                if "U" in kwargs.keys():
                    U = kwargs.get("U", None)[self.graph.node_ids]
                    D = kwargs.get("D", None)
                    self.classifier.set_QD(U, D)
            elif smodel_type == "LanczosLaplace":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = LanczosLaplace(sgraph)
                if "U" in kwargs.keys():
                    U = kwargs.get("U", None)[self.graph.node_ids]
                    D = kwargs.get("D", None)
                    self.classifier.set_QD(U, D)
            elif smodel_type == "MLP":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = SClassifier(sgraph)

        elif data_type == "f+s":
            if fmodel_type == "GNN":
                fgraph = self.graph
            else:
                fgraph = self.create_FDGCN_data()

            if smodel_type == "GNN":
                if self.id == "Server":
                    sgraph = self.create_SGNN_data(**kwargs)
                    self.classifier = FedGNNMaster(fgraph, sgraph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = FedSlave(fgraph, server_embedding_func)
            elif smodel_type == "DGCN":
                sgraph = self.create_SDGCN_data(**kwargs)
                self.classifier = FedDGCN(fgraph, sgraph)
            elif smodel_type in ["SpectralDGCN", "LanczosDGCN"]:
                sgraph = self.create_SDGCN_data(**kwargs)
                self.classifier = FedSpectralDGCN(fgraph, sgraph)
            elif smodel_type == "CentralDGCN":
                if self.id == "Server":
                    sgraph = self.create_SDGCN_data(**kwargs)
                    self.classifier = FedDGCNMaster(fgraph, sgraph)
                else:
                    server_embedding_func = kwargs.get("server_embedding_func", None)
                    self.classifier = FedSlave(fgraph, server_embedding_func)
            elif smodel_type == "Laplace":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = FedLaplaceClassifier(fgraph, sgraph)
            elif smodel_type == "SpectralLaplace":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = FedSpectralLaplaceClassifier(fgraph, sgraph)
                if "U" in kwargs.keys():
                    U = kwargs.get("U", None)[self.graph.node_ids]
                    D = kwargs.get("D", None)
                    self.classifier.set_QD(U, D)
            elif smodel_type == "LanczosLaplace":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = FedLanczosLaplaceClassifier(fgraph, sgraph)
                if "U" in kwargs.keys():
                    U = kwargs.get("U", None)[self.graph.node_ids]
                    D = kwargs.get("D", None)
                    self.classifier.set_QD(U, D)
            elif smodel_type == "MLP":
                sgraph = self.create_SGNN_data(**kwargs)
                self.classifier = FedMLPClassifier(fgraph, sgraph)

        self.classifier.create_optimizer()

    def train_local_model(
        self,
        epochs=config.model.iterations,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type="feature",
        structure_type=config.structure_model.structure_type,
        log=True,
        plot=True,
        **kwargs,
    ):
        model_type = f"Server {data_type} {smodel_type}-{fmodel_type}"
        self.initialize(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            structure_type=structure_type,
        )
        return super().train_local_model(
            epochs=epochs,
            log=log,
            plot=plot,
            model_type=model_type,
        )

    def save_SFVs(self):
        SFV = self.classifier.get_SFV().detach().cpu().numpy()
        self.SFVs.append(deepcopy(SFV))

        y_pred = self.classifier.get_prediction()
        y_pred = y_pred.detach().cpu().numpy()
        y = self.graph.y.cpu().numpy()
        cf_score = y_pred[np.arange(y_pred.shape[0]), y]
        self.cf_score_list.append(cf_score)

    def get_SFVs(self):
        return self.SFVs, self.cf_score_list
