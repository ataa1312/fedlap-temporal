from copy import deepcopy

import numpy as np
from src import *
from src.client import Client
from src.GNN.DGCN import DGCN, SDGCN, SDGCNMaster, SpectralDGCN
from src.GNN.fGNN import FGNN, FEdgeGNN
from src.GNN.sGNN import SGNNSlave, SGNNMaster, SClassifier
from torch_sparse import SparseTensor
from src.classifier import Classifier
from src.GNN.laplace import (
    SLaplace,
    SEdgeLaplace,
    LanczosLaplace,
    SpectralLaplace,
    LanczosEdgeLaplace,
    SpectralEdgeLaplace,
)
from src.utils.graph import Graph, AGraph
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.GNN.GNN_classifier import (
    FedDGCN,
    FedSlave,
    FedGNNMaster,
    FedDGCNMaster,
    FedSpectralDGCN,
    FedMLPClassifier,
    FedLaplaceClassifier,
    FedLaplaceEdgeClassifier,
    FedLanczosLaplaceClassifier,
    FedSpectralLaplaceClassifier,
    FedLanczosLaplaceEdgeClassifier,
    FedDynamicLanczosLaplaceClassifier,
)


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
        # Storage for embeddings across snapshots
        # Each element is a dict: {"embeddings": tensor, "node_ids": tensor, "snapshot_idx": int}
        self.stored_embeddings = list[dict[str, torch.Tensor]]()
        # Keep references to old SFVs to prevent gradient flow issues
        # When SFV is trainable (requires_grad=True), we need to keep old SFVs
        # so gradients can flow back to them through stored embeddings
        self.stored_sfvs = dict[int, torch.Tensor]()
        # Storage for context pairs from random walks (DySAT-style)
        # Dictionary mapping snapshot_idx -> context_pairs dict
        # context_pairs maps node_id (original) -> list of context node_ids (original)
        self.stored_context_pairs = dict[int, dict[int, list[int]]]()

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
        # After optimizer step, copy updated SFVs from stored_sfvs_grad to stored_sfvs
        # for use in the next epoch
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

        if data_type == "f+s":
            if fmodel_type == "GNN":
                fgraph = self.graph

            if smodel_type == "LanczosLaplace":
                stored_sfvs = self.get_stored_sfvs(ss_idx)
                sgraph = self.create_SGNN_data(**{"SFV": stored_sfvs})
                if downstream_task == "node-classification":
                    self.classifier = FedLanczosLaplaceClassifier(fgraph, sgraph)
                elif downstream_task == "edge-prediction":
                    self.classifier = FedDynamicLanczosLaplaceClassifier(
                        fgraph, sgraph, num_ss
                    )
                if is_attr_good(self.classifier, "register_stored_sfvs"):
                    self.classifier.register_stored_sfvs(self.stored_sfvs)  # pyright:ignore
                self.classifier.create_optimizer()

    def update_spectral_features(self, share: dict):
        kwargs = share.copy()
        if not is_attr_good(self.classifier, "smodel"):
            return

        if "U" in kwargs and "D" in kwargs:
            D_client = kwargs["D"]
            U_client = kwargs["U"][self.graph.node_ids]
            self.classifier.smodel.set_QD(U_client, D_client)  # pyright: ignore

    def initialize_sfvs(self, num_ss: int):
        structural_features_vectors = dict[int, torch.Tensor]()
        for ss_idx in range(num_ss):
            structure_type = config.structure_model.structure_type
            num_structural_features = config.structure_model.num_structural_features
            num_spectral_features = config.spectral.spectral_len
            assert num_spectral_features != 0

            # This stuoid function does not return anything and assigns its returnee to
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

    def get_embeddings(self, detach=False):
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Call initialize() first.")

        embeddings = self.classifier.get_embeddings()

        if detach:
            embeddings = embeddings.detach()

        return embeddings

    def store_embeddings(self, snapshot_idx: int, detach: bool = False):
        current_sfv = self.classifier.smodel.graph.x  # pyright: ignore
        if current_sfv.requires_grad:
            self.classifier.stored_sfvs_grad[snapshot_idx] = current_sfv  # pyright: ignore
            if current_sfv.grad is None:
                current_sfv.retain_grad()

        embeddings = self.get_embeddings(detach=detach)

        stored_embeddings = {
            "embeddings": embeddings,
            "snapshot_idx": snapshot_idx,
        }

        self.stored_embeddings.append(stored_embeddings)

    def get_stored_embeddings(self, snapshot_idx: int | None = None):
        if snapshot_idx is None:
            return self.stored_embeddings
        else:
            for stored in self.stored_embeddings:
                if stored["snapshot_idx"] == snapshot_idx:
                    return stored
            return None

    def clear_stored_embeddings(self):
        self.stored_embeddings = list[dict[str, torch.Tensor]]()
        # Reset stored_sfvs_grad so new SFVs can be stored in the next epoch
        if self.classifier is not None and is_attr_good(
            self.classifier, "stored_sfvs_grad"
        ):
            num_ss = len(self.classifier.stored_sfvs_grad)  # pyright: ignore
            shape = (
                config.spectral.spectral_len,
                config.structure_model.num_structural_features,
            )
            self.classifier.stored_sfvs_grad = {  # pyright: ignore
                ss_idx: torch.zeros(shape) for ss_idx in range(num_ss)
            }

    def generate_context_pairs(
        self,
        num_walks: int = 10,
        walk_len: int = 40,
        window_size: int = 10,
        p: float = 1.0,
        q: float = 1.0,
    ) -> dict[int, list[int]]:
        return self.graph.generate_context_pairs(
            num_walks=num_walks,
            walk_len=walk_len,
            window_size=window_size,
            p=p,
            q=q,
            use_original_node_ids=False,
        )

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
            context_pairs = self.generate_context_pairs(
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

    def get_temporal_embeddings(self):
        if not is_attr_good(self.classifier, "tmodel"):
            raise ValueError(
                "Classifier does not have tmodel. Initialize temporal encoder first."
            )

        if len(self.stored_embeddings) == 0:
            raise ValueError("No stored embeddings available. Encode snapshots first.")

        stored_embeddings = self.stored_embeddings
        max_num_nodes = max(map(lambda x: len(x["embeddings"]), stored_embeddings))
        embeddings = []
        for d in stored_embeddings:
            emb = d["embeddings"]
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

        if len(self.stored_embeddings) == 0:
            raise ValueError("No stored embeddings available. Encode snapshots first.")

        temporal_embeddings = self.get_temporal_embeddings()  # [T, N, F]

        if snapshot_indices is None:
            snapshot_indices = sorted(self.stored_context_pairs.keys())

        total_loss = torch.tensor(
            0.0, device=temporal_embeddings.device, requires_grad=True
        )
        num_snapshots = 0

        for ss_idx in snapshot_indices:
            context_pairs = self.get_stored_context_pairs(ss_idx)
            if context_pairs is None or len(context_pairs) == 0:
                continue

            z = temporal_embeddings[ss_idx]

            snapshot_loss = self.classifier.compute_random_walk_loss(  # pyright: ignore
                temporal_embeddings=z,
                context_pairs=context_pairs,
                snapshot_idx=ss_idx,
                neg_sample_size=neg_sample_size,
                neg_weight=neg_weight,
                batch_size=batch_size,
            )

            total_loss = total_loss + snapshot_loss
            num_snapshots += 1

        if num_snapshots == 0:
            return torch.tensor(
                0.0, device=temporal_embeddings.device, requires_grad=True
            )

        return total_loss / num_snapshots

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

    def evaluate_with_sklearn_classifier(
        self,
        num_ss: int,
        operator: LinkFeatureOperator = "hadamard",
    ) -> tuple[dict, dict]:
        stored_emb_data = self.get_stored_embeddings(num_ss - 1)
        last_embeddings = stored_emb_data["embeddings"]
        last_embeddings = last_embeddings.clone().detach()
        train_pos_mask = self.graph.train_edge_label == 1
        train_neg_mask = self.graph.train_edge_label == 0
        train_pos_edges = self.graph.train_edge_label_index[:, train_pos_mask]  # pyright: ignore
        train_neg_edges = self.graph.train_edge_label_index[:, train_neg_mask]  # pyright: ignore

        val_pos_mask = self.graph.val_edge_label == 1
        val_neg_mask = self.graph.val_edge_label == 0
        val_pos_edges = self.graph.val_edge_label_index[:, val_pos_mask]  # pyright: ignore
        val_neg_edges = self.graph.val_edge_label_index[:, val_neg_mask]  # pyright: ignore

        test_pos_mask = self.graph.test_edge_label == 1
        test_neg_mask = self.graph.test_edge_label == 0
        test_pos_edges = self.graph.test_edge_label_index[:, test_pos_mask]  # pyright: ignore
        test_neg_edges = self.graph.test_edge_label_index[:, test_neg_mask]  # pyright: ignore

        train_pos_feats = self.classifier.get_edge_embeddings(  # pyright: ignore
            train_pos_edges, last_embeddings, operator
        )
        train_neg_feats = self.classifier.get_edge_embeddings(  # pyright: ignore
            train_neg_edges, last_embeddings, operator
        )
        val_pos_feats = self.classifier.get_edge_embeddings(  # pyright: ignore
            val_pos_edges, last_embeddings, operator
        )
        val_neg_feats = self.classifier.get_edge_embeddings(  # pyright: ignore
            val_neg_edges, last_embeddings, operator
        )
        test_pos_feats = self.classifier.get_edge_embeddings(  # pyright: ignore
            test_pos_edges, last_embeddings, operator
        )
        test_neg_feats = self.classifier.get_edge_embeddings(  # pyright: ignore
            test_neg_edges, last_embeddings, operator
        )

        train_data_feats = np.vstack([train_pos_feats, train_neg_feats])
        train_labels = np.concatenate(
            [np.ones(len(train_pos_feats)), np.zeros(len(train_neg_feats))]
        )

        val_data_feats = np.vstack([val_pos_feats, val_neg_feats])
        val_labels = np.concatenate(
            [np.ones(len(val_pos_feats)), np.zeros(len(val_neg_feats))]
        )

        test_data_feats = np.vstack([test_pos_feats, test_neg_feats])
        test_labels = np.concatenate(
            [np.ones(len(test_pos_feats)), np.zeros(len(test_neg_feats))]
        )

        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(train_data_feats, train_labels)

        val_pred = classifier.predict_proba(val_data_feats)[:, 1]
        test_pred = classifier.predict_proba(test_data_feats)[:, 1]

        val_auc = roc_auc_score(val_labels, val_pred)
        test_auc = roc_auc_score(test_labels, test_pred)

        val_results = {operator: val_auc}
        test_results = {operator: test_auc}

        return val_results, test_results
