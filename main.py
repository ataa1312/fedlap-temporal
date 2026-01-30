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
    name += f"{config.model.data_type}|"
    name += f"{config.model.smodel_type}-{config.model.fmodel_type}|"
    name += f"{config.spectral.update_mode}|"
    name += f"C{config.subgraph.num_subgraphs}|"
    name += f"T{config.dynamic.min_snapshot}-{config.dynamic.max_snapshot}|"
    name += f"EM-{config.dynamic.evaluation.mode}|"
    name += datetime.now().strftime("%Y%m%d%H%M%S")
    return name


def get_run_name(config, run_idx: int) -> str:
    """Generate a descriptive run name from config parameters."""
    name = ""
    name += f"{config.dataset.dataset_name}|"
    name += f"{config.model.data_type}|"
    name += f"{config.model.smodel_type}-{config.model.fmodel_type}|"
    name += f"{config.spectral.update_mode}|"
    name += f"C{config.subgraph.num_subgraphs}|"
    name += f"T{config.dynamic.min_snapshot}-{config.dynamic.max_snapshot}|"
    name += f"EM-{config.dynamic.evaluation.mode}|"
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
        best_val_auc = 0.0
        best_train_metrics: dict[str, float] = {}
        best_val_metrics: dict[str, float] = {}
        best_test_metrics: dict[str, float] = {}
        LOGGER.info(f"Training on the first {tdx} snapshot(s)...")

        train_graphs, test_graph = build_training_graphs(
            cleaned_graphs=cleaned_graphs,
            tdx=tdx,
            dataset_name=config.dataset.dataset_name,
        )

        gnn_server = GNNServer(None)  # pyright:ignore
        operator = config.dynamic.evaluation.link_feature_operator

        # FIXME: This needs to be fixed
        # num_nodes = sum([client.num_nodes() for client in gnn_server.clients])
        # coef = [client.num_nodes() / num_nodes for client in gnn_server.clients]

        for epoch in range(config.model.iterations):
            LOGGER.info(f"Epoch {epoch + 1}/{config.model.iterations}")

            if epoch > 0:
                gnn_server.clear_all_stored_sembeddings()
                gnn_server.clear_all_stored_tembeddings()
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
                gnn_server.store_snapshot_sembeddings(
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
            train_res = {}
            val_res = {}
            test_res = {}

            if should_eval:
                gnn_server.load_test_snapshot(
                    test_graph,
                    subgraph_node_ids,
                    tdx,
                    # **test_split_edges_kwargs,
                )

                (
                    train_res,
                    val_res,
                    test_res,
                    avg_train_auc,
                    avg_val_auc,
                    avg_test_auc,
                ) = gnn_server.evaluate_with_classifier(
                    eval_config=config.dynamic.evaluation
                )

                if val_res[gnn_server.id][operator]["auc"] > best_val_auc:
                    best_train_metrics = train_res[gnn_server.id][operator]
                    best_val_metrics = val_res[gnn_server.id][operator]
                    best_test_metrics = test_res[gnn_server.id][operator]
                    best_val_auc = val_res[gnn_server.id][operator]["auc"]

                # if avg_val_auc > best_val_auc:
                #     best_train_auc = avg_train_auc
                #     best_val_auc = avg_val_auc
                #     best_test_auc = avg_test_auc

            log_dict = dict[str, torch.Tensor | float]()
            log_dict[f"tdx{tdx:02}/server/avg_train_auc"] = avg_train_auc
            log_dict[f"tdx{tdx:02}/server/avg_val_auc"] = avg_val_auc
            log_dict[f"tdx{tdx:02}/server/avg_test_auc"] = avg_test_auc

            if train_res:
                all_results_keys = set(train_res.keys())
                
                for cid in all_results_keys:
                    if cid == gnn_server.id:
                        prefix = f"tdx{tdx:02}/server"
                    else:
                        prefix = f"tdx{tdx:02}/client_{cid:02}"
                    
                    if cid in client_losses:
                         log_dict[f"{prefix}/train_loss"] = client_losses[cid]

                    log_dict[f"{prefix}/train_eval_loss"] = train_res[cid]["loss"]
                    log_dict[f"{prefix}/train_auc"] = train_res[cid][operator]["auc"]
                    log_dict[f"{prefix}/train_ap"] = train_res[cid][operator]["ap"]
                    log_dict[f"{prefix}/train_acc"] = train_res[cid][operator]["accuracy"]
                    log_dict[f"{prefix}/train_f1"] = train_res[cid][operator]["f1"]

                    log_dict[f"{prefix}/val_eval_loss"] = val_res[cid]["loss"]
                    log_dict[f"{prefix}/val_auc"] = val_res[cid][operator]["auc"]
                    log_dict[f"{prefix}/val_ap"] = val_res[cid][operator]["ap"]
                    log_dict[f"{prefix}/val_acc"] = val_res[cid][operator]["accuracy"]
                    log_dict[f"{prefix}/val_f1"] = val_res[cid][operator]["f1"]
                    
                    log_dict[f"{prefix}/test_eval_loss"] = test_res[cid]["loss"]
                    log_dict[f"{prefix}/test_auc"] = test_res[cid][operator]["auc"]
                    log_dict[f"{prefix}/test_ap"] = test_res[cid][operator]["ap"]
                    log_dict[f"{prefix}/test_acc"] = test_res[cid][operator]["accuracy"]
                    log_dict[f"{prefix}/test_f1"] = test_res[cid][operator]["f1"]
                    log_dict[f"{prefix}/test_mcc"] = test_res[cid][operator]["mcc"]
                    log_dict[f"{prefix}/test_best_threshold"] = test_res[cid][operator]["best_threshold"]

                run.log(log_dict)

        run.log(
            {
                "tdx": tdx,
                "server/loss": avg_loss,
                "server/train_auc": best_train_metrics["auc"],
                "server/val_auc": best_val_metrics["auc"],
                "server/test_auc": best_test_metrics["auc"],
                "server/train_ap": best_train_metrics["ap"],
                "server/val_ap": best_val_metrics["ap"],
                "server/test_ap": best_test_metrics["ap"],
                "server/test_mrr": best_test_metrics["mrr"],
                "server/test_ranking_ap": best_test_metrics["ranking_ap"],
                "server/test_acc": best_test_metrics["accuracy"],
                "server/test_f1": best_test_metrics["f1"],
                "server/test_precision": best_test_metrics["precision"],
                "server/test_recall": best_test_metrics["recall"],
                "server/test_mcc": best_test_metrics["mcc"],
                "server/test_best_threshold": best_test_metrics["best_threshold"],
            },
        )

        if log:
            LOGGER.info(
                f"Best train AUC: {best_train_metrics['auc']:.4f}, ",
                f"Best validation AUC: {best_val_metrics['auc']:.4f}, "
                f"Best test AUC: {best_test_metrics['auc']:.4f}"
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
