import os
import json
from typing import cast
from pathlib import Path

curr_path = Path(__file__).parent.resolve()
os.chdir(curr_path)

import wandb
from src import *
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.utils.define_graph import (
    define_graph,
    load_dataset,
    build_cleaned_graphs,
    build_training_graphs,
)
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


def main():
    downstream_task: DownstreamTask = "edge-prediction"
    detach_embeddings = False  # Keep gradients for backprop
    log = True

    if log:
        LOGGER.info(
            f"Starting DySAT-style temporal training: {config.dynamic.evaluation.classifier.num_epoch} epochs per T, "
            f"T from {config.dynamic.min_snapshot} to {config.dynamic.max_snapshot}"
        )

    # Initialize wandb
    # wandb.init(
    #     project=config.wandb.project,
    #     name=config.wandb.name or f"fedlap-T{config.dynamic.min_snapshot}-{config.dynamic.max_snapshot}-epochs{config.dynamic.evaluation.classifier.num_epoch}",
    #     group=config.wandb.group,
    #     job_type=config.wandb.job_type,
    #     config={
    #         "min_T": config.dynamic.min_snapshot,
    #         "max_T": config.dynamic.max_snapshot,
    #         "num_epochs": config.dynamic.evaluation.classifier.num_epoch,
    #         "dataset": config.dataset.dataset_name,
    #         "num_clients": config.subgraph.num_subgraphs,
    #         "downstream_task": downstream_task,
    #     },
    #     mode=config.wandb.mode,
    # )

    data_path = curr_path / "datasets" / config.dataset.dataset_name
    graphs_and_features = load_dataset(config.dataset.dataset_name, data_path)
    graphs, features = graphs_and_features
    assert isinstance(graphs, list)
    assert isinstance(features, list)
    cast(list[torch.Tensor], graphs)
    cast(list[torch.Tensor], features)

    cleaned_graphs, max_num_nodes = build_cleaned_graphs(
        graphs,
        features,
        is_directed=config.dataset.is_directed,
        normalize=config.dataset.normalize,
    )
    subgraph_node_ids = random_assign(max_num_nodes, config.subgraph.num_subgraphs)

    for tdx in range(config.dynamic.min_snapshot, config.dynamic.max_snapshot + 1):
        best_val_auc, best_test_auc, best_train_auc = 0.0, 0.0, 0.0
        LOGGER.info(f"Training on the first {tdx} snapshot(s)...")

        train_graphs, test_graph = build_training_graphs(
            cleaned_graphs=cleaned_graphs,
            tdx=tdx,
            dataset_name=config.dataset.dataset_name,
        )

        gnn_server = GNNServer(None)  # pyright:ignore

        # FIXME: This needs to be fixed
        # num_nodes = sum([client.num_nodes() for client in gnn_server.clients])
        # coef = [client.num_nodes() / num_nodes for client in gnn_server.clients]

        for epoch in range(config.model.iterations):
            LOGGER.info( f"Epoch {epoch + 1}/{config.model.iterations}")

            if epoch > 0:
                gnn_server.clear_all_stored_embeddings()
                gnn_server.reset_trainings()
                gnn_server.set_train_mode()

            for idx, ss_idx in enumerate(range(tdx)):
                gnn_server.load_snapshot(
                    snapshot=train_graphs[ss_idx],
                    ss_idx=idx,
                    num_ss=tdx,
                    random_walk_config=config.dynamic.random_walk,
                    smodel_type=config.model.smodel_type,
                    fmodel_type=config.model.fmodel_type,
                    data_type="f+s",
                    downstream_task=downstream_task,
                    spectral_len=config.spectral.spectral_len,
                    spectral_update_mode="recompute",
                    subgraph_node_ids=subgraph_node_ids,
                    log=log,
                    # **train_split_edges_kwargs,
                )

                LOGGER.info(f"Encoding embeddings for snapshot #{ss_idx + 1}...")
                gnn_server.store_snapshot_embeddings(
                    snapshot_idx=idx,
                    detach=detach_embeddings,
                )

            client_losses, avg_loss = gnn_server.train_temporal_models(
                snapshot_indices=list(range(tdx)),
                neg_sample_size=config.dynamic.unsupervised.neg_sampling.size,
                neg_weight=config.dynamic.unsupervised.neg_sampling.weight,
                batch_size=config.dynamic.unsupervised.batch_size,
                log=log,
            )

            log_dict = {
                f"tdx{tdx}/epoch": epoch + 1,
                f"tdx{tdx}/server/train_loss": avg_loss,
            }

            for client_id, loss in client_losses.items():
                log_dict[f"tdx{tdx}/client_{client_id}/train_loss"] = loss

            should_eval = (
                (epoch % config.dynamic.evaluation.eval_freq == 0)
                or (
                    epoch == config.model.iterations - 1
                    and config.dynamic.evaluation.last_epoch
                )
                or (epoch == 0 and config.dynamic.evaluation.first_epoch)
            )

            if should_eval:
                # test_split_edges_kwargs = {
                #     "split_edges_for_edge_prediction": True,
                #     "val_ratio": 0.2,  # Default, could be from config.dynamic.evaluation.data.num_val if available
                #     "test_ratio": 0.6,  # Default, could be from config.dynamic.evaluation.data.num_test if available
                #     "is_undirected": not config.dataset.is_directed,
                #     "add_negative_train_samples": True,  # Default
                #     "negative_ratio": 1.0,  # Default
                # }

                gnn_server.load_test_snapshot(
                    test_graph,
                    subgraph_node_ids,
                    tdx,
                    # **test_split_edges_kwargs,
                )

                (
                    client_train_aucs,
                    client_val_aucs,
                    client_test_aucs,
                    avg_train_auc,
                    avg_val_auc,
                    avg_test_auc,
                ) = gnn_server.evaluate_with_classifier(
                    eval_config=config.dynamic.evaluation
                )

                # # Log server (average) metrics
                # log_dict.update(
                #     {
                #         f"tdx{tdx}/server/train_auc": avg_train_auc,
                #         f"tdx{tdx}/server/val_auc": avg_val_auc,
                #         f"tdx{tdx}/server/test_auc": avg_test_auc,
                #     }
                # )

                # Log individual client metrics
                # for client_id in client_train_aucs.keys():
                #     log_dict[f"tdx{tdx}/client_{client_id}/train_auc"] = (
                #         client_train_aucs[client_id]
                #     )
                #     log_dict[f"tdx{tdx}/client_{client_id}/val_auc"] = client_val_aucs[
                #         client_id
                #     ]
                #     log_dict[f"tdx{tdx}/client_{client_id}/test_auc"] = (
                #         client_test_aucs[client_id]
                #     )

                # Log to wandb
                # wandb.log(log_dict, step=epoch + 1)

                # Only update test AUC when validation AUC improves
                # if avg_val_auc > best_val_auc:
                #     best_val_auc = avg_val_auc
                #     best_test_auc = avg_test_auc
                #     if log:
                #         LOGGER.info(
                #             f"New best validation AUC: {best_val_auc:.4f} "
                #             f"(test AUC: {best_test_auc:.4f})"
                #         )
            else:
                wandb.log(log_dict, step=epoch + 1)

            if log:
                LOGGER.info(
                    f"Best validation AUC: {best_val_auc:.4f}, "
                    f"test AUC: {best_test_auc:.4f}"
                )

        if log:
            train_indices = list(range(tdx))
            LOGGER.info(
                f"Completed tdx={tdx}: trained on {train_indices} snapshots, tested on snapshot {tdx}"
            )


if __name__ == "__main__":
    main()

    # train_split_edges_kwargs = {
    #     "split_edges_for_edge_prediction": True,
    #     "val_ratio": 0.00,
    #     "test_ratio": 0.00,
    #     "is_undirected": True,
    #     "add_negative_train_samples": True,
    #     "negative_ratio": 1.0,
    # }
    # test_split_edges_kwargs = {
    #     "split_edges_for_edge_prediction": True,
    #     "val_ratio": 0.10,
    #     "test_ratio": 0.10,
    #     "is_undirected": True,
    #     "add_negative_train_samples": True,
    #     "negative_ratio": 1.0,
    # }
