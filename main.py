import os
import json
import random
from typing import cast
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import wandb

curr_path = Path(__file__).parent.resolve()
os.chdir(curr_path)

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


def get_group_name(config) -> str:
    """Generate a meaningful group name for grouping multiple runs of the same experiment."""
    name = ""
    name += f"{config.dataset.dataset_name}|"
    name += f"{config.model.smodel_type}-{config.model.fmodel_type}|"
    name += f"{config.spectral.update_mode}|"
    name += f"C{config.subgraph.num_subgraphs}|"
    name += f"T{config.dynamic.min_snapshot}-{config.dynamic.max_snapshot}|"
    name += datetime.now().strftime("%Y%m%d%H%M%S")
    return name


def get_run_name(config, run_idx: int) -> str:
    """Generate a descriptive run name from config parameters."""
    name = ""
    name += f"{config.dataset.dataset_name}|"
    name += f"{config.model.smodel_type}-{config.model.fmodel_type}|"
    name += f"{config.spectral.update_mode}|"
    name += f"C{config.subgraph.num_subgraphs}|"
    name += f"T{config.dynamic.min_snapshot}-{config.dynamic.max_snapshot}|"
    name += f"Run-{run_idx}"
    return name


def main(run: wandb.Run):
    detach_embeddings = False  # Keep gradients for backprop
    log = True

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
            LOGGER.info(f"Epoch {epoch + 1}/{config.model.iterations}")


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
                    data_type=config.model.data_type,
                    downstream_task=config.downstream_task,
                    spectral_len=config.spectral.spectral_len,
                    spectral_update_mode=config.spectral.update_mode,
                    subgraph_node_ids=subgraph_node_ids,
                    log=log,

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

            should_eval = (
                (epoch % config.dynamic.evaluation.eval_freq == 0)
                or (
                    epoch == config.model.iterations - 1
                    and config.dynamic.evaluation.last_epoch
                )
                or (epoch == 0 and config.dynamic.evaluation.first_epoch)
            )

            # Initialize AUC dictionaries to avoid NameError when should_eval is False
            client_train_aucs = {}
            client_val_aucs = {}
            client_test_aucs = {}

            if should_eval:
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

                if avg_val_auc > best_val_auc:
                    best_train_auc = avg_train_auc
                    best_val_auc = avg_val_auc
                    best_test_auc = avg_test_auc

            if client_train_aucs:
                log_dict = dict[str, torch.Tensor]()
                for cid in client_train_aucs.keys():
                    log_dict[f"tdx{tdx:02}/client_{cid:02}/train_loss"] = client_losses[cid]
                    log_dict[f"tdx{tdx:02}/client_{cid:02}/train_auc"] = client_train_aucs[
                        cid
                    ]
                    log_dict[f"tdx{tdx:02}/client_{cid:02}/val_auc"] = client_val_aucs[cid]
                    log_dict[f"tdx{tdx:02}/client_{cid:02}/test_auc"] = client_test_aucs[
                        cid
                    ]
                run.log(log_dict)

        run.log(
            {
                "tdx": tdx,
                "server/loss": avg_loss,
                "server/train_auc": best_train_auc,
                "server/val_auc": best_val_auc,
                "server/test_auc": best_test_auc,
            },
        )

        if log:
            LOGGER.info(
                f"Best train AUC: {best_train_auc:.4f}, "
                f"Best validation AUC: {best_val_auc:.4f}, "
                f"Best test AUC: {best_test_auc:.4f}"
            )


if __name__ == "__main__":
    wandb.login()

    config.wandb.group = get_group_name(config)
    base_seed = config.seed
    for rdx in range(config.num_runs):
        LOGGER.info(f"Starting run #{rdx + 1}/{config.num_runs}")
        config.seed = base_seed + 100 * rdx
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        config.wandb.name = get_run_name(config, rdx)

        wandb_run = wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            config=config.config,
            group=config.wandb.group,
            job_type=config.wandb.job_type,
            mode=config.wandb.mode,
            save_code=True,
        )

        try:
            main(wandb_run)
        finally:
            wandb_run.finish()
