import operator
from copy import deepcopy

from src import *
from src.server import Server
from src.utils.graph import Graph
from src.GNN.GNN_client import GNNClient, SpectralFeatures
from src.utils.define_graph import graph_to_pygdata
from src.utils.config_parser import EvaluationConfig, RandomWalkConfig
from src.utils.graph_partitioning import create_subgraphs


class GNNServer(Server, GNNClient):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph)

        self.clients = list[GNNClient]()

    def add_client(self, subgraph):
        client = GNNClient(
            graph=subgraph,
            id=self.num_clients,
        )

        self.clients.append(client)
        self.num_clients += 1

    def initialize(
        self,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type="feature",
        spectral_len=0,
        log=True,
        **kwargs,
    ) -> None:
        share = {}
        if data_type in ["structure", "f+s"]:
            num_spectral_features = None
            if smodel_type in ["DGCN", "CentralDGCN", "SpectralDGCN", "LanczosDGCN"]:
                abar = self.graph.calc_abar(method=smodel_type)
                share["abar"] = abar
                num_spectral_features = abar.shape[1]
            elif smodel_type in ["SpectralLaplace", "LanczosLaplace"]:
                D, U = self.graph.calc_eignvalues(
                    estimate=not (smodel_type.startswith("Spectral")),
                    spectral_len=spectral_len,
                    log=log,
                )
                share["D"] = D
                share["U"] = U
                num_spectral_features = D.shape[0]

            structure_type = kwargs.get(
                "structure_type", config.structure_model.structure_type
            )
            num_structural_features = kwargs.get(
                "num_structural_features",
                # self.graph.num_classes,
                config.structure_model.num_structural_features,
            )

            self.graph.add_structural_features(
                structure_type=structure_type,
                num_structural_features=num_structural_features,
                num_spectral_features=num_spectral_features,
            )
            SFV = self.graph.structural_features
            share["SFV"] = SFV

        kwargs.update(share)

        super().initialize(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            **kwargs,
        )

        if smodel_type in ["CentralDGCN", "GNN"]:
            share["server_embedding_func"] = self.classifier.get_embeddings_func()

        return share

    def initialize_FL(
        self,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type="feature",
        spectral_len=0,
        **kwargs,
    ) -> None:
        share = self.initialize(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            spectral_len=spectral_len,
            **kwargs,
        )

        kwargs.update(share)

        client: GNNClient
        for client in self.clients:
            client.initialize(
                smodel_type=smodel_type,
                fmodel_type=fmodel_type,
                data_type=data_type,
                **kwargs,
            )

    def joint_train_g(
        self,
        epochs=config.model.iterations,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        FL=True,
        data_type="feature",
        log=True,
        plot=True,
        model_type="",
        spectral_len=config.spectral.spectral_len,
        **kwargs,
    ):
        self.initialize_FL(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            spectral_len=spectral_len,
            log=log,
            **kwargs,
        )
        if FL:
            model_type = f"FL {data_type} {smodel_type}-{fmodel_type} GA"
        else:
            model_type = f"Local {data_type} {smodel_type}-{fmodel_type} GA"

        return super().joint_train_g(
            epochs=epochs,
            FL=FL,
            log=log,
            plot=plot,
            model_type=model_type,
        )

    def joint_train_w(
        self,
        epochs=config.model.iterations,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        FL=True,
        data_type="feature",
        log=True,
        plot=True,
        model_type="",
        spectral_len=config.spectral.spectral_len,
        **kwargs,
    ):
        self.initialize_FL(
            smodel_type=smodel_type,
            fmodel_type=fmodel_type,
            data_type=data_type,
            spectral_len=spectral_len,
            **kwargs,
        )
        if FL:
            model_type = f"FL {data_type} {smodel_type}-{fmodel_type} WA"
        else:
            model_type = f"Local {data_type} {smodel_type}-{fmodel_type} WA"

        return super().joint_train_w(
            epochs=epochs,
            FL=FL,
            log=log,
            plot=plot,
            model_type=model_type,
        )

    def get_all_clients_embeddings(self, detach=False):
        client: GNNClient
        for client in self.clients:
            client.get_embeddings(detach=detach)

    def store_snapshot_embeddings(self, snapshot_idx: int, detach=False):
        client: GNNClient
        for client in self.clients:
            client.store_embeddings(snapshot_idx, detach=detach)

    def clear_all_stored_embeddings(self):
        client: GNNClient
        for client in self.clients:
            client.clear_stored_embeddings()

    def generate_context_pairs_for_snapshot(
        self,
        snapshot_idx: int,
        num_walks: int,
        walk_len: int,
        window_size: int,
        p: float,
        q: float,
        log: bool,
    ):
        for client in self.clients:
            client.store_context_pairs(
                snapshot_idx=snapshot_idx,
                num_walks=num_walks,
                walk_len=walk_len,
                window_size=window_size,
                p=p,
                q=q,
            )

    def clear_all_stored_context_pairs(self):
        for client in self.clients:
            client.clear_stored_context_pairs()

    def get_previous_UD(self, spectral_update_mode: str, ss_idx: int):
        prev_D, prev_U = None, None
        # if (
        #     spectral_update_mode in ["keep", "update"]
        #     and is_attr_good(self.classifier.smodel, "Q")  # pyright: ignore
        #     and is_attr_good(self.classifier.smodel, "D")  # pyright: ignore
        # ):
        # prev_D, prev_U = self.classifier.smodel.D, self.classifier.smodel.Q  # pyright: ignore

        if spectral_update_mode in ["keep", "update"]:
            if len(self.stored_spectrals) > 0:
                first_spectrals = min(self.stored_spectrals.keys())
                stored_spectral = self.stored_spectrals[first_spectrals]
                prev_D, prev_U = stored_spectral.D, stored_spectral.U
        else:
            if ss_idx in self.stored_spectrals:
                stored_spectral = self.stored_spectrals[ss_idx]
                prev_D, prev_U = stored_spectral.D, stored_spectral.U

        return prev_D, prev_U

    def get_spectral_features(
        self, smodel_type, ss_idx, spectral_len, spectral_update_mode, prev_U, prev_D
    ):
        share = {}
        num_spectral_features = None

        if smodel_type in ["SpectralLaplace", "LanczosLaplace"]:
            if (
                spectral_update_mode == "keep"
                and prev_U is not None
                and prev_D is not None
            ):
                LOGGER.info("Keeping previous eigenvectors U and eigenvalues D...")
                D, U = prev_D, prev_U
            elif (
                spectral_update_mode == "update"
                and prev_U is not None
                and prev_D is not None
            ):
                LOGGER.info("Updating spectral features...")
                raise NotImplementedError()
            else:
                if ss_idx not in self.stored_spectrals:
                    LOGGER.info("Computing spectral features...")
                    D, U = self.graph.calc_eignvalues(
                        estimate=not (smodel_type.startswith("Spectral")),
                        spectral_len=spectral_len,
                        log=False,
                    )
                    self.stored_spectrals[ss_idx] = SpectralFeatures(U=U, D=D)
                else:
                    stored_spectral = self.stored_spectrals[ss_idx]
                    D, U = stored_spectral.D, stored_spectral.U

            share["D"] = D
            share["U"] = U
            num_spectral_features = D.shape[0]

        return share, num_spectral_features

    def shallow_initialize_classifier(
        self,
        smodel_type: str,
        fmodel_type: str,
        data_type: str,
        ss_idx: int,
        num_ss: int,
        downstream_task: DownstreamTask,
    ) -> None:
        super().shallow_initialize_classifier(
            smodel_type, fmodel_type, data_type, ss_idx, num_ss, downstream_task
        )
        for client in self.clients:
            client.shallow_initialize_classifier(
                smodel_type, fmodel_type, data_type, ss_idx, num_ss, downstream_task
            )

    def initialize_sfvs(self, num_ss: int):
        super().initialize_sfvs(num_ss)
        for client in self.clients:
            client.initialize_sfvs(num_ss)

    def load_snapshot(
        self,
        snapshot: Graph,
        ss_idx: int,
        num_ss: int,
        random_walk_config: RandomWalkConfig,
        smodel_type=config.model.smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type="feature",
        downstream_task: DownstreamTask = "edge-prediction",
        spectral_len=0,
        spectral_update_mode="recompute",
        subgraph_node_ids: dict | None = None,
        log=True,
    ):
        self.update_graph(snapshot, ss_idx)

        if data_type in ["structure", "f+s"]:
            prev_D, prev_U = self.get_previous_UD(spectral_update_mode, ss_idx)
            share, _ = self.get_spectral_features(
                smodel_type, ss_idx, spectral_len, spectral_update_mode, prev_U, prev_D
            )

        # INFO: Check if clients and classifier need initialization
        assert isinstance(subgraph_node_ids, dict)
        subgraphs = create_subgraphs(snapshot, subgraph_node_ids)

        needs_initialization = len(self.clients) == 0 or self.classifier is None
        if needs_initialization:
            if len(self.clients) == 0:
                for subgraph in subgraphs:
                    self.add_client(subgraph)
            if data_type in ["structure", "f+s"]:
                self.initialize_sfvs(num_ss=num_ss)
            self.shallow_initialize_classifier(
                smodel_type, fmodel_type, data_type, ss_idx, num_ss, downstream_task
            )
            self.share_weights()

        else:
            for client, subgraph in zip(self.clients, subgraphs):
                client.update_graph(subgraph, ss_idx)

        if data_type in ["structure", "f+s"] and share:
            self.update_spectral_features(share)
            for client in self.clients:
                client.update_spectral_features(share)

        if ss_idx == num_ss - 1 and self.eval_train_snapshot is None:
            self.eval_train_snapshot = graph_to_pygdata(self.graph)
            for client in self.clients:
                client.eval_train_snapshot = graph_to_pygdata(client.graph)

        self.generate_context_pairs_for_snapshot(
            snapshot_idx=ss_idx,
            num_walks=random_walk_config.num,
            walk_len=random_walk_config.num,
            window_size=random_walk_config.window_size,
            p=1.0,
            q=1.0,
            log=log,
        )

    def load_test_snapshot(
        self,
        snapshot: Graph,
        subgraph_node_ids: dict,
        ss_idx: int,
        **split_edges_kwargs,
    ):
        self.update_graph(snapshot, ss_idx, test=True)
        subgraphs = create_subgraphs(snapshot, subgraph_node_ids, **split_edges_kwargs)
        for client, subgraph in zip(self.clients, subgraphs):
            client.update_graph(subgraph, ss_idx, test=True)

    def train_temporal_models(
        self,
        snapshot_indices: list[int],
        neg_sample_size: int,
        neg_weight: float,
        batch_size: int,
        log: bool,
    ):
        total_loss: float = 0.0
        num_clients_with_loss = 0
        client_losses = {}  # Store individual client losses
        for client in self.clients:
            loss = client.compute_random_walk_loss(
                neg_sample_size=neg_sample_size,
                neg_weight=neg_weight,
                batch_size=batch_size,
                snapshot_indices=snapshot_indices,
            )

            loss.backward(retain_graph=True)
            client_loss = loss.item()
            client_losses[client.id] = client_loss
            if log:
                LOGGER.info(
                    f"Client {client.id}: loss = {client_loss:.4f}, "
                    f"optimizer step completed"
                )
            total_loss += client_loss
            num_clients_with_loss += 1

            torch.nn.utils.clip_grad_norm_(client.classifier.parameters(), max_norm=1.0)

        clients_grads = self.get_grads()

        client_node_counts = []
        for client in self.clients:
            total_nodes = 0
            stored_embs = client.get_stored_embeddings()
            assert stored_embs is not None and isinstance(stored_embs, dict)
            for embeddings in stored_embs.values():
                num_nodes = embeddings.shape[0]
                total_nodes += num_nodes
            client_node_counts.append(total_nodes)

        total_nodes_all_clients = sum(client_node_counts)
        coef = [count / total_nodes_all_clients for count in client_node_counts]

        grads = sum_lod(clients_grads, coef)
        self.share_grads(grads)
        self.update_models()

        avg_loss = (
            total_loss / num_clients_with_loss if num_clients_with_loss > 0 else 0.0
        )
        if log:
            if num_clients_with_loss > 0:
                LOGGER.info(
                    f"Temporal training completed: "
                    f"{num_clients_with_loss} clients, "
                    f"average loss = {avg_loss:.4f}"
                )
            else:
                LOGGER.warning("No clients were trained (all had zero loss or errors)")

        return client_losses, avg_loss

    def evaluate_with_classifier(
        self, eval_config: EvaluationConfig
    ) -> tuple[dict, dict, dict, float, float, float]:
        train_aucs, val_aucs, test_aucs = 0.0, 0.0, 0.0
        client_train_results = {}
        client_val_results = {}
        client_test_results = {}

        operator = eval_config.link_feature_operator
        for client in self.clients:
            train_results, val_results, test_results = client.evaluate_with_classifier(
                eval_config=eval_config
            )
            train_aucs += train_results[operator]
            val_aucs += val_results[operator]
            test_aucs += test_results[operator]
            client_train_results[client.id] = train_results
            client_val_results[client.id] = val_results
            client_test_results[client.id] = test_results
            LOGGER.info(
                f"Client {client.id}: train auc = {train_results[operator]:.4f}, "
                f"val auc = {val_results[operator]:.4f}, "
                f"test auc = {test_results[operator]:.4f}"
            )

        num_clients = len(client_train_results)
        avg_train_auc = train_aucs / num_clients if num_clients > 0 else 0.0
        avg_val_auc = val_aucs / num_clients if num_clients > 0 else 0.0
        avg_test_auc = test_aucs / num_clients if num_clients > 0 else 0.0

        LOGGER.info(
            f"Average across clients: train auc = {avg_train_auc:.4f}, "
            f"val auc = {avg_val_auc:.4f}, "
            f"test auc = {avg_test_auc:.4f}"
        )

        return (
            client_train_results,
            client_val_results,
            client_test_results,
            avg_train_auc,
            avg_val_auc,
            avg_test_auc,
        )
