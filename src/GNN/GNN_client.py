from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from src import *
from torch import nn
from src.client import Client
from src.GNN.DGCN import DGCN, SDGCN, SDGCNMaster, SpectralDGCN
from src.GNN.fGNN import FGNN, NewFGNN, FEdgeGNN
from src.GNN.sGNN import SGNNSlave, SGNNMaster, SClassifier
from src.classifier import Classifier
from sklearn.metrics import roc_auc_score
from src.GNN.laplace import (
    SLaplace,
    SEdgeLaplace,
    LanczosLaplace,
    SpectralLaplace,
    LanczosEdgeLaplace,
    SpectralEdgeLaplace,
)
from src.utils.graph import Graph, AGraph
from torch_geometric import transforms
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import negative_sampling
from src.GNN.GNN_classifier import (
    FedDGCN,
    FedSlave,
    FedGNNMaster,
    FedDGCNMaster,
    FedSpectralDGCN,
    FedMLPClassifier,
    FedLaplaceClassifier,
    FedLaplaceEdgeClassifier,
    FedDynamicFeatureClassifier,
    FedLanczosLaplaceClassifier,
    FedSpectralLaplaceClassifier,
    FedLanczosLaplaceEdgeClassifier,
    FedDynamicLanczosLaplaceClassifier,
)
from src.utils.config_parser import BaseTxConfig, ClassifierConfig, EvaluationConfig


@dataclass
class SpectralFeatures:
    U: torch.Tensor
    D: torch.Tensor
    V: torch.Tensor | None = None


class GNNClient(Client):
    # Edge prediction attributes that should be copied when creating new graphs
    # Class variable (shared across all instances) - using tuple for immutability
    EDGE_PREDICTION_ATTRS = (
        "message_passing_edge_index",
        "train_edge_label_index",
        "train_edge_label",
        "val_edge_label_index",
        "val_edge_label",
        "test_edge_label_index",
        "test_edge_label",
    )

    def __init__(self, graph: Graph, id: int = 0):
        super().__init__(graph=graph, id=id, classifier_type="GNN")
        # LOGGER.info(f"Number of edges: {self.graph.num_edges}")
        self.classifier: Classifier | None = None
        self.SFVs = []
        self.cf_score_list = []

        self.edge_indices = dict[int, torch.Tensor]()
        self.sembeddings = dict[int, torch.Tensor]()
        self.tembeddings: torch.Tensor | None = None
        self.stored_sfvs = dict[int, torch.Tensor]()
        self.stored_context_pairs = dict[int, dict[int, list[int]]]()
        self.eval_train_snapshot: PyGData | None = None
        self.stored_spectrals = dict[int, SpectralFeatures]()

    def _extract_edge_prediction_attrs(self) -> dict:
        """
        Extract edge prediction attributes from the graph.
        Returns a dictionary with all edge prediction attributes,
        using None as default for missing attributes.
        """
        return {
            attr: getattr(self.graph, attr, None) for attr in self.EDGE_PREDICTION_ATTRS
        }

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
            **self._extract_edge_prediction_attrs(),  # This should later be fixed
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
        downstream_task: DownstreamTask = kwargs.get(
            "downstream_task", "node-classification"
        )
        if data_type == "feature":
            if fmodel_type == "GNN":
                if downstream_task == "node-classification":
                    self.classifier = FGNN(self.graph)
                elif downstream_task == "edge-prediction":
                    self.classifier = FEdgeGNN(self.graph)
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
                if downstream_task == "node-classification":
                    self.classifier = SLaplace(sgraph)
                elif downstream_task == "edge-prediction":
                    self.classifier = SEdgeLaplace(sgraph)
            elif smodel_type == "SpectralLaplace":
                sgraph = self.create_SGNN_data(**kwargs)
                if downstream_task == "node-classification":
                    self.classifier = SpectralLaplace(sgraph)
                elif downstream_task == "edge-prediction":
                    self.classifier = SpectralEdgeLaplace(sgraph)
                if "U" in kwargs.keys():
                    U = kwargs.get("U", None)[self.graph.node_ids]
                    D = kwargs.get("D", None)
                    self.classifier.set_QD(U, D)
            elif smodel_type == "LanczosLaplace":
                sgraph = self.create_SGNN_data(**kwargs)
                if downstream_task == "node-classification":
                    self.classifier = LanczosLaplace(sgraph)
                elif downstream_task == "edge-prediction":
                    self.classifier = LanczosEdgeLaplace(sgraph)
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
                if downstream_task == "node-classification":
                    self.classifier = FedLaplaceClassifier(fgraph, sgraph)
                elif downstream_task == "edge-prediction":
                    self.classifier = FedLaplaceEdgeClassifier(fgraph, sgraph)
            elif smodel_type == "SpectralLaplace":
                sgraph = self.create_SGNN_data(**kwargs)
                if downstream_task == "node-classification":
                    self.classifier = FedSpectralLaplaceClassifier(fgraph, sgraph)
                elif downstream_task == "edge-prediction":
                    self.classifier = FedSpectralLaplaceEdgeClassifier(fgraph, sgraph)
                if "U" in kwargs.keys():
                    U = kwargs.get("U", None)[self.graph.node_ids]
                    D = kwargs.get("D", None)
                    self.classifier.set_QD(U, D)
            elif smodel_type == "LanczosLaplace":
                sgraph = self.create_SGNN_data(**kwargs)
                if downstream_task == "node-classification":
                    self.classifier = FedLanczosLaplaceClassifier(fgraph, sgraph)
                elif downstream_task == "edge-prediction":
                    self.classifier = FedLanczosLaplaceEdgeClassifier(fgraph, sgraph)
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
            **kwargs,
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

    def update_model(self):
        super().update_model()
        if is_attr_good(self.classifier, "stored_sfvs_grad"):
            self.stored_sfvs = {
                ss_idx: sfv.clone().detach()
                for ss_idx, sfv in self.classifier.stored_sfvs_grad.items()  # pyright: ignore
                if sfv.numel() > 0  # Only copy non-empty tensors
            }

    def shallow_initialize_classifier(
        self,
        smodel_type: str,
        fmodel_type: str,
        data_type: str,
        ss_idx: int,
        num_ss: int,
        downstream_task: DownstreamTask,
    ) -> None:
        if self.classifier is not None:
            return

        if data_type == "feature":
            if fmodel_type == "GNN":
                if downstream_task == "edge-prediction":
                    self.classifier = FedDynamicFeatureClassifier(self.graph, num_ss)

        elif data_type == "f+s":
            if fmodel_type == "GNN":
                fgraph = self.graph

            if smodel_type == "LanczosLaplace":
                stored_sfvs = self.get_stored_sfvs(ss_idx)
                sgraph = self.create_SGNN_data(**{"SFV": stored_sfvs})
                if downstream_task == "edge-prediction":
                    self.classifier = FedDynamicLanczosLaplaceClassifier(
                        fgraph, sgraph, num_ss
                    )
            self.classifier.create_optimizer()

    def update_spectral_features(self, share: dict):
        if not is_attr_good(self.classifier, "smodel"):
            return

        if "U" in share and "D" in share:
            graph_device = self.graph.node_ids.device
            D_client = share["D"].to(graph_device)
            U = share["U"].to(graph_device)
            U_client = U[self.graph.node_ids]
            self.classifier.smodel.set_QD(U_client, D_client)  # pyright: ignore

    def initialize_sfvs(self, num_ss: int):
        structural_features_vectors = dict[int, torch.Tensor]()
        for ss_idx in range(num_ss):
            structure_type = config.structure_model.structure_type
            num_structural_features = config.structure_model.num_structural_features
            num_spectral_features = config.spectral.spectral_len
            assert num_spectral_features != 0

            # This stupid function does not return anything and assigns its returnee to
            # self.graph.structural_features
            self.graph.add_structural_features(
                structure_type=structure_type,
                num_structural_features=num_structural_features,
                num_spectral_features=num_spectral_features,
            )
            assert self.graph.structural_features is not None
            structural_features_vectors[ss_idx] = self.graph.structural_features
        self.store_sfvs(structural_features_vectors)

    def store_sfvs(self, sfvs: dict[int, torch.Tensor]) -> None:
        if len(sfvs) == 1:
            self.stored_sfvs.update(sfvs)
        else:
            for ss_idx, sfv in sfvs.items():
                self.stored_sfvs[ss_idx] = sfv

    def get_stored_sfvs(
        self, ss_idx: int | None
    ) -> dict[int, torch.Tensor] | torch.Tensor:
        if ss_idx is None:
            return self.stored_sfvs
        else:
            if ss_idx in self.stored_sfvs.keys():
                return self.stored_sfvs[ss_idx]
            raise KeyError(f"No SFVs stored for snapshot #{ss_idx}!")

    def get_sembeddings(self, detach=False):
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Call initialize() first.")

        embeddings = self.classifier.get_embeddings()

        if detach:
            embeddings = embeddings.detach()

        return embeddings

    def store_sembeddings(self, snapshot_idx: int, detach: bool = False):
        if is_attr_good(self.classifier, "smodel"):
            current_sfv = self.classifier.smodel.graph.x  # pyright: ignore
            if current_sfv.requires_grad:
                self.classifier.stored_sfvs_grad[snapshot_idx] = current_sfv  # pyright: ignore
                if current_sfv.grad is None:
                    current_sfv.retain_grad()

        sembeddings = self.get_sembeddings(detach=detach)
        self.sembeddings[snapshot_idx] = sembeddings

    def get_stored_sembeddings(self, snapshot_idx: int | None = None):
        if snapshot_idx is None:
            return self.sembeddings
        else:
            return self.sembeddings.get(snapshot_idx)

    def clear_stored_sembeddings(self):
        self.sembeddings = dict[int, torch.Tensor]()
        if self.classifier is not None and is_attr_good(
            self.classifier, "stored_sfvs_grad"
        ):
            for sfv in self.classifier.stored_sfvs_grad.values():  # pyright: ignore
                if sfv is not None and sfv.requires_grad:
                    if sfv.grad is not None:
                        sfv.grad.zero_()
                    else:
                        sfv.grad = None

    def store_context_pairs(
        self,
        snapshot_idx: int,
        num_walks: int = 10,
        walk_len: int = 40,
        window_size: int = 10,
        p: float = 1.0,
        q: float = 1.0,
    ):
        if snapshot_idx not in self.stored_context_pairs:
            LOGGER.info(
                f"Generating context pairs for snapshot {snapshot_idx} on clients {self.id}"
            )
            context_pairs = self.graph.generate_context_pairs(
                num_walks=num_walks,
                walk_len=walk_len,
                window_size=window_size,
                p=p,
                q=q,
            )

            self.stored_context_pairs[snapshot_idx] = context_pairs

    def get_stored_context_pairs(
        self, snapshot_idx: int | None = None
    ) -> list[dict[int, list[int]]] | dict[int, list[int]] | None:
        if snapshot_idx is None:
            sorted_indices = sorted(self.stored_context_pairs.keys())
            return [self.stored_context_pairs[idx] for idx in sorted_indices]
        else:
            return self.stored_context_pairs.get(snapshot_idx, None)

    def clear_stored_context_pairs(self):
        self.stored_context_pairs = dict[int, dict[int, list[int]]]()

    def get_tembeddings(self):
        if not is_attr_good(self.classifier, "tmodel"):
            raise ValueError(
                "Classifier does not have tmodel. Initialize temporal encoder first."
            )

        if len(self.sembeddings) == 0:
            raise ValueError("No stored embeddings available. Encode snapshots first.")

        smbeddings_list = [
            self.sembeddings[ss_idx] for ss_idx in sorted(self.sembeddings.keys())
        ]
        max_num_nodes = max(map(lambda x: len(x), smbeddings_list))
        embeddings = []
        for emb in smbeddings_list:
            zero_pad = torch.zeros(
                max_num_nodes - emb.shape[0], emb.shape[1], device=emb.device
            )
            emb = torch.cat([emb, zero_pad], dim=0)
            emb = emb.unsqueeze(0)
            embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=0)  # [T, N, F]
        embeddings = embeddings.transpose(0, 1)  # [N, T, F] for temporal model
        embeddings = self.classifier.tmodel(embeddings)  # pyright: ignore
        embeddings = embeddings.transpose(0, 1)  # [T, N, F]
        return embeddings

    def compute_random_walk_loss(
        self,
        neg_sample_size: int = 10,
        neg_weight: float = 1.0,
        batch_size: int = 512,
        snapshot_indices: list[int] | None = None,
    ):
        if not is_attr_good(self.classifier, "tmodel"):
            raise ValueError(
                "Classifier does not have tmodel. Initialize temporal encoder first."
            )

        if not hasattr(self.classifier, "compute_random_walk_loss"):
            raise ValueError(
                "Classifier does not support random walk loss computation."
            )

        if len(self.sembeddings) == 0:
            raise ValueError(
                "No stored structural embeddings available. Encode snapshots first."
            )

        tembeddings = self.get_tembeddings()  # [T, N, F]
        self.tembeddings = tembeddings

        if snapshot_indices is None:
            snapshot_indices = sorted(self.stored_context_pairs.keys())

        total_loss = torch.tensor(0.0, device=tembeddings.device, requires_grad=True)
        num_snapshots = 0

        for ss_idx in snapshot_indices:
            context_pairs = self.get_stored_context_pairs(ss_idx)
            if context_pairs is None or len(context_pairs) == 0:
                continue

            z = tembeddings[ss_idx]

            # Get edge_index for this snapshot from the graph
            # Use original_edge_index which is the original edge_index before reindexing
            # edge_index = self.graph.original_edge_index
            assert isinstance(self.edge_indices, dict) and len(self.edge_indices) > 0
            edge_index = self.edge_indices[ss_idx]

            snapshot_loss = self.classifier.compute_random_walk_loss(  # pyright: ignore
                temporal_embeddings=z,
                context_pairs=context_pairs,
                snapshot_idx=ss_idx,
                edge_index=edge_index,
                neg_sample_size=neg_sample_size,
                neg_weight=neg_weight,
                batch_size=batch_size,
            )

            total_loss = total_loss + snapshot_loss
            num_snapshots += 1

        if num_snapshots == 0:
            return torch.tensor(0.0, device=tembeddings.device, requires_grad=True)

        return total_loss / num_snapshots

    def get_stored_tembeddings(self) -> torch.Tensor:
        if self.tembeddings is None:
            raise ValueError("Temporal Embeddings are none!")
        return self.tembeddings

    def clear_stored_tembeddings(self) -> None:
        self.tembeddings = None

    def update_graph(self, new_graph: Graph, ss_idx: int, test: bool = False):
        self.graph = new_graph

        if not test:
            if self.classifier is not None:
                self.classifier.graph = new_graph
                if is_attr_good(self.classifier, "fmodel"):
                    self.classifier.fmodel.graph = new_graph  # pyright: ignore

            if self.classifier is not None:
                if is_attr_good(self.classifier, "smodel"):
                    # Get old SFVs for the next snapshot (or current if first time)
                    # Note: stored_sfvs_grad[ss_idx] should already be set in store_embeddings
                    # before update_graph is called for the next snapshot
                    old_sfvs = self.get_stored_sfvs(ss_idx)
                    sgraph = self.create_SGNN_data(**{"SFV": old_sfvs})
                    self.classifier.smodel.graph = sgraph  # pyright: ignore

    @staticmethod
    def get_link_features(
        edge_index: torch.Tensor,
        embeddings: torch.Tensor,
        operator: LinkFeatureOperator = "hadamard",
    ) -> torch.Tensor:
        """
        Compute link features for edges using embeddings.

        Args:
            edge_index: [2, num_edges] tensor of edge pairs
            embeddings: [num_nodes, embed_dim] tensor of node embeddings
            operator: Feature operator

        Returns:
            [num_edges, embed_dim] or [num_edges] tensor of link features (stays on same device)
        """
        match operator:
            case "hadamard":
                # Hadamard product: element-wise multiplication
                src_emb = embeddings[edge_index[0]]  # [num_edges, embed_dim]
                tgt_emb = embeddings[edge_index[1]]  # [num_edges, embed_dim]
                return src_emb * tgt_emb
            case "dot-product":
                # Dot product: inner multiplication
                src_emb = embeddings[edge_index[0]]  # [num_edges, embed_dim]
                tgt_emb = embeddings[edge_index[1]]  # [num_edges, embed_dim]
                return (src_emb * tgt_emb).sum(dim=-1, keepdim=True)  # [num_edges, 1]
            case "concat":
                # Concatenation: returns [num_edges, 2*embed_dim]
                src_emb = embeddings[edge_index[0]]  # [num_edges, embed_dim]
                tgt_emb = embeddings[edge_index[1]]  # [num_edges, embed_dim]
                return torch.cat([src_emb, tgt_emb], dim=-1)
            case _:
                raise NotImplementedError(f"Operator {operator} not implemented")

    @staticmethod
    def extract_feats_and_labels(
        data: Graph,
        embeddings: torch.Tensor,
        edge_index_attr: str,
        label_attr: str,
        operator: LinkFeatureOperator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_label: torch.Tensor | None = getattr(data, label_attr, None)
        edge_label_index: torch.Tensor | None = getattr(data, edge_index_attr, None)
        if edge_label_index is None:
            raise AttributeError(f"None attribute: {edge_label_index!r}")
        if edge_label is None:
            raise AttributeError(f"None attribute: {edge_label!r}")

        pos_mask = edge_label == 1
        neg_mask = edge_label == 0
        pos_edges = edge_label_index[:, pos_mask]
        neg_edges = edge_label_index[:, neg_mask]
        pos_feats = GNNClient.get_link_features(pos_edges, embeddings, operator)
        neg_feats = GNNClient.get_link_features(neg_edges, embeddings, operator)
        device = embeddings.device
        feats = torch.cat([pos_feats, neg_feats], dim=0).to(device)
        labels = torch.cat(
            [
                torch.ones(len(pos_feats), device=device),
                torch.zeros(len(neg_feats), device=device),
            ],
            dim=0,
        )
        return feats, labels

    def evaluate_with_classifier(
        self,
        eval_config: EvaluationConfig,
    ) -> tuple[dict, dict, dict]:
        transform = transforms.Compose(
            [
                transforms.RandomLinkSplit(
                    num_val=eval_config.data.num_val,
                    num_test=0.0,
                    is_undirected=not eval_config.data.is_directed,
                    add_negative_train_samples=eval_config.data.add_negative_train_samples,
                    neg_sampling_ratio=eval_config.data.neg_sampling_ratio,
                ),
            ]
        )

        operator = eval_config.link_feature_operator
        tembs = self.get_stored_tembeddings()
        embeddings = tembs[-1]
        embeddings = embeddings.detach()
        train_data, val_data, _ = transform(self.eval_train_snapshot)
        train_data_feats, train_labels = self.extract_feats_and_labels(
            train_data, embeddings, "edge_label_index", "edge_label", operator
        )

        val_data_feats, val_labels = self.extract_feats_and_labels(
            val_data, embeddings, "edge_label_index", "edge_label", operator
        )

        assert self.graph.edge_index is not None
        test_pos_edges = self.graph.edge_index
        num_test_pos = test_pos_edges.shape[1]
        num_test_neg = int(num_test_pos * eval_config.data.neg_sampling_ratio)
        test_neg_edges = negative_sampling(
            self.graph.edge_index,
            self.graph.num_nodes,
            num_neg_samples=num_test_neg,
            method="sparse",
            force_undirected=not eval_config.data.is_directed,
        )

        val_ratio: float = eval_config.data.num_val

        num_edges = min(num_test_pos, test_neg_edges.shape[1])
        test_pos_edges = test_pos_edges[:, :num_edges]
        test_neg_edges = test_neg_edges[:, :num_edges]

        perm_indices = torch.randperm(num_edges, device=test_pos_edges.device)
        test_pos_edges = test_pos_edges[:, perm_indices]
        test_neg_edges = test_neg_edges[:, perm_indices]

        # val_num = int(val_ratio * perm_indices.numel())
        #
        # val_pos_edges = test_pos_edges[:, :val_num]
        # val_neg_edges = test_neg_edges[:, :val_num]
        #
        # val_pos_feats = self.get_link_features(val_pos_edges, embeddings, operator)
        # val_neg_feats = self.get_link_features(val_neg_edges, embeddings, operator)
        #
        # val_data_feats = torch.cat([val_pos_feats, val_neg_feats], dim=0)
        # val_labels = torch.cat(
        #     [
        #         torch.ones(len(val_pos_feats), device=embeddings.device),
        #         torch.zeros(len(val_neg_feats), device=embeddings.device),
        #     ],
        #     dim=0,
        # )

        # test_pos_edges = test_pos_edges[:, val_num:]
        # test_neg_edges = test_neg_edges[:, val_num:]

        test_pos_feats = self.get_link_features(test_pos_edges, embeddings, operator)
        test_neg_feats = self.get_link_features(test_neg_edges, embeddings, operator)

        test_data_feats = torch.cat([test_pos_feats, test_neg_feats], dim=0)
        test_labels = torch.cat(
            [
                torch.ones(len(test_pos_feats), device=embeddings.device),
                torch.zeros(len(test_neg_feats), device=embeddings.device),
            ],
            dim=0,
        )

        classifier_config = eval_config.classifier
        classifier = train_logistic_regression(
            train_data_feats,
            train_labels,
            classifier_config=classifier_config,
            device=device,
        )

        criterion = getattr(torch.nn, classifier_config.loss_fn)()
        with torch.no_grad():
            train_logits = classifier(train_data_feats)
            val_logits = classifier(val_data_feats)
            test_logits = classifier(test_data_feats)

            train_loss = criterion(train_logits, train_labels).item()
            val_loss = criterion(val_logits, val_labels).item()
            test_loss = criterion(test_logits, test_labels).item()

            train_pred = torch.sigmoid(train_logits).cpu().numpy()
            val_pred = torch.sigmoid(val_logits).cpu().numpy()
            test_pred = torch.sigmoid(test_logits).cpu().numpy()

        train_auc = roc_auc_score(train_labels.cpu().numpy(), train_pred)
        val_auc = roc_auc_score(val_labels.cpu().numpy(), val_pred)
        test_auc = roc_auc_score(test_labels.cpu().numpy(), test_pred)

        train_results = {operator: train_auc, "loss": train_loss}
        val_results = {operator: val_auc, "loss": val_loss}
        test_results = {operator: test_auc, "loss": test_loss}

        return train_results, val_results, test_results


def get_optimizer(
    net: torch.nn.Module, tx_config: BaseTxConfig
) -> torch.optim.Optimizer:
    tx = tx_config.fn
    moment: float = getattr(tx_config, "moment", 0.0)
    weight_decay: float = getattr(tx_config, "weight_decay", 0.0)
    match tx:
        case torch.optim.SGD:
            return torch.optim.SGD(
                net.parameters(),
                lr=tx_config.lr,
                momentum=moment,
                weight_decay=weight_decay,
            )
        case torch.optim.Adam:
            return torch.optim.Adam(
                net.parameters(), lr=tx_config.lr, weight_decay=weight_decay
            )
        case torch.optim.AdamW:
            return torch.optim.AdamW(
                net.parameters(), lr=tx_config.lr, weight_decay=weight_decay
            )
        case torch.optim.Adagrad:
            return torch.optim.Adagrad(
                net.parameters(), lr=tx_config.lr, weight_decay=weight_decay
            )
        case _:
            raise ValueError("Update `get_optimizer` or use available optimizers.")


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim: int, device: torch.device):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_logistic_regression(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    classifier_config: ClassifierConfig,
    device: torch.device | None = None,
) -> LogisticRegressionClassifier:
    if device is None:
        device = train_features.device

    input_dim = train_features.shape[1]
    classifier = LogisticRegressionClassifier(input_dim, device)
    classifier.train()

    criterion = getattr(torch.nn, classifier_config.loss_fn)()
    tx_config = classifier_config.tx

    if tx_config.fn is not None:
        optimizer = get_optimizer(classifier, tx_config)
    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": classifier.linear.weight,
                    "lr": tx_config.lr,
                    "weight_decay": tx_config.weight_decay,
                },
                {
                    "params": classifier.linear.bias,
                    "lr": tx_config.lr,
                    "weight_decay": 0.0,
                },
            ],
            lr=tx_config.lr,  # Default lr (can be overridden per parameter group)
        )

    for epoch in range(classifier_config.num_epoch):
        optimizer.zero_grad()
        logits = classifier(train_features)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

    classifier.eval()
    return classifier
