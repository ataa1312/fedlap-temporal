import os
import json
from pathlib import Path

curr_path = Path(__file__).parent.resolve()
os.chdir(curr_path)

from src import *
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.utils.define_graph import define_graph, build_training_snapshots
from src.FedGCN.FedGCN_server import FedGCNServer
from src.FedPub.fedpub_server import FedPubServer
from src.fedsage.fedsage_server import FedSAGEServer
from src.utils.graph_partitioning import (
    random_assign,
    partition_graph,
    create_subgraphs,
    create_mend_graph,
    fedGCN_partitioning,
)


def get_max_num_nodes(dataset_name: str) -> int:
    root = f"./datasets/{dataset_name}"
    root_path = Path(root)
    graph_file_names = sorted((root_path / "graphs").rglob("*.npz"))
    features_file_names = sorted((root_path / "features").rglob("*.npz"))

    if len(graph_file_names) != len(features_file_names):
        raise ValueError(
            f"Mismatch: found {len(graph_file_names)} graph files and "
            f"{len(features_file_names)} feature files"
        )

    max_num_nodes: int = 0
    for gfn, ffn in zip(graph_file_names, features_file_names):
        graph_tensor = torch.from_numpy(np.load(gfn)["graph"])
        feature_tensor = torch.from_numpy(np.load(ffn)["features"])
        assert len(graph_tensor) == len(feature_tensor)
        max_num_nodes = max(len(graph_tensor), max_num_nodes)

    return max_num_nodes


def get_num_snapshots(dataset_name: str) -> int:
    """Get the number of snapshots available in the dataset."""
    root = f"./datasets/{dataset_name}"
    root_path = Path(root)
    graph_file_names = sorted((root_path / "graphs").rglob("*.npz"))
    return len(graph_file_names)


if __name__ == "__main__":
    # Configuration
    downstream_task: DownstreamTask = "edge-prediction"
    num_epochs = 300  # Number of epochs per T
    min_T = 3  # Start with T=3 (train on 0-1, test on 2)
    max_T = 3  # Use up to T=3 or all snapshots, whichever is smaller
    detach_embeddings = False  # Keep gradients for backprop
    log = True

    # Get number of snapshots available
    max_num_snapshots = get_num_snapshots(config.dataset.dataset_name)
    LOGGER.info(f"Dataset has {max_num_snapshots} snapshots available")

    # Set up graph partitioning
    max_num_nodes = get_max_num_nodes(config.dataset.dataset_name)
    subgraph_node_ids = random_assign(
        max_num_nodes,
        config.subgraph.num_subgraphs,
    )
    train_split_edges_kwargs = {
        "split_edges_for_edge_prediction": True,
        "val_ratio": 0.00,
        "test_ratio": 0.00,
        "is_undirected": True,
        "add_negative_train_samples": True,
        "negative_ratio": 1.0,
    }
    test_split_edges_kwargs = {
        "split_edges_for_edge_prediction": True,
        "val_ratio": 0.10,
        "test_ratio": 0.10,
        "is_undirected": True,
        "add_negative_train_samples": True,
        "negative_ratio": 1.0,
    }

    # Main temporal training loop (DySAT-style)
    # For each T: train on snapshots 0 to T-2, test on snapshot T-1
    if max_T is None:
        max_T = max_num_snapshots

    if log:
        LOGGER.info(
            f"Starting DySAT-style temporal training: {num_epochs} epochs per T, "
            f"T from {min_T} to {max_T}"
        )

    # Outer loop: increment T (number of snapshots to use)
    for T in range(min_T, max_T + 1):
        if log:
            LOGGER.info(f"{'=' * 80}")
            LOGGER.info(f"T = {T}: Training on 0 to {T - 2}, Testing on {T - 1}")
            LOGGER.info(f"{'=' * 80}")

        train_snapshots, train_indices, test_index, test_snapshot = (
            build_training_snapshots(
                T=T,
                dataset_name=config.dataset.dataset_name,
            )
        )
        gnn_server = GNNServer(None)  # pyright:ignore

        # FIXME: This needs to be fixed
        # num_nodes = sum([client.num_nodes() for client in gnn_server.clients])
        # coef = [client.num_nodes() / num_nodes for client in gnn_server.clients]

        for epoch in range(num_epochs):
            if log:
                LOGGER.info(f"\n  T={T}, Epoch {epoch + 1}/{num_epochs}")

            if epoch > 0:
                gnn_server.clear_all_stored_embeddings()
                gnn_server.reset_trainings()
                gnn_server.set_train_mode()

            for idx, snapshot_idx in enumerate(range(T)):
                gnn_server.load_snapshot(
                    snapshot=train_snapshots[snapshot_idx],
                    ss_idx=idx,
                    num_ss=T,
                    smodel_type="LanczosLaplace",
                    fmodel_type="GNN",
                    data_type="f+s",
                    downstream_task=downstream_task,
                    spectral_len=config.spectral.spectral_len,
                    spectral_update_mode="recompute",
                    subgraph_node_ids=subgraph_node_ids,
                    log=log,
                    **train_split_edges_kwargs,
                )

                gnn_server.encode_snapshot(
                    snapshot_idx=idx,
                    detach=detach_embeddings,
                    log=log,
                )

            if log:
                LOGGER.info(f"Stored embeddings: {T} snapshots (0 to {T - 1})")

            encoded_indices = list(range(T))
            gnn_server.train_temporal_models(
                snapshot_indices=encoded_indices,
                neg_sample_size=10,
                neg_weight=1.0,
                batch_size=64,
                log=log,
            )

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                gnn_server.load_test_snapshot(
                    test_snapshot,
                    subgraph_node_ids,
                    test_index,
                    **test_split_edges_kwargs,
                )

                gnn_server.evaluate_with_sklearn_classifier(T)

            # clients_grads = gnn_server.get_grads()
            # # FIXME: Get coef right
            # # grads = sum_lod(clients_grads, coef)
            # grads = sum_lod(clients_grads)
            # gnn_server.share_grads(grads)
            #
            # gnn_server.update_models()

            # gnn_server.update_snapshot(
            #     new_graph=test_snapshot,
            #     ss_idx=len(train_indices),  # FIXME: This may be wrong
            #     num_ss=T,
            #     smodel_type="LanczosLaplace",
            #     fmodel_type="GNN",
            #     data_type="f+s",
            #     downstream_task=downstream_task,
            #     spectral_len=config.spectral.spectral_len,
            #     spectral_update_mode="recompute",
            #     subgraph_node_ids=subgraph_node_ids,
            #     log=log,  # Less verbose for inner loops
            #     **test_split_edges_kwargs,
            # )

            if log:
                LOGGER.info(f"Epoch {epoch + 1} completed for T={T}")

        if log:
            LOGGER.info(
                f"Completed T={T}: trained on {len(train_indices)} snapshots, tested on snapshot {test_index}"
            )

    if log:
        LOGGER.info(
            f"\nTraining completed: T from {min_T} to {max_T}, {num_epochs} epochs per T"
        )
