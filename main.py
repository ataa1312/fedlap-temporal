import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

os.chdir(Path(__file__).parent.resolve())

from src import *  # config (global singleton), device, LOGGER
from parser import Parser
from config.config import overlay_config
from config.assertions import assert_cfg
from registries import datasets
import src.datasets  # noqa: F401  triggers uci/bitcoin loader registration
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _init_wandb():
    wb = config["wandb"] if "wandb" in config else None
    mode = wb["mode"] if (wb is not None and "mode" in wb) else "disabled"
    if mode == "disabled":
        return None
    import wandb

    return wandb.init(project=wb["project"], mode=mode, reinit=True)


def run_once() -> dict:
    _seed(config["seed"])
    name = config["dataset"]["name"]
    n_clients = config["subgraph"]["num_subgraphs"]
    LOGGER.info(
        f"dataset={name} clients={n_clients} device={device} seed={config['seed']} "
        f"data_type={config['model']['data_type']}"
    )
    global_snaps = datasets[name](config)
    LOGGER.info(f"loaded {len(global_snaps)} global snapshots")
    client_snaps = partition_snapshots(global_snaps, n_clients)

    server = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server.add_client(snaps)
    results = server.joint_train_w()
    mm = results.get("mean_metrics") or {}
    LOGGER.info(
        f"RESULT dataset={name} clients={n_clients} seed={config['seed']} "
        f"mean_mrr={results['mean_mrr']} std={results['std_mrr']} "
        f"auc={mm.get('roc_auc')} ap={mm.get('ap')} f1={mm.get('f1')} mcc={mm.get('mcc')} "
        f"snapshots={len(results['mrr_history'])}"
    )

    wb = _init_wandb()
    if wb is not None:
        wb.summary["mean_mrr"] = results["mean_mrr"]
        wb.summary["std_mrr"] = results["std_mrr"]
        for k, v in mm.items():
            wb.summary[f"mean_{k}"] = v
        wb.finish()
    return results


def main() -> None:
    parser = Parser()
    args = parser.parse_args()
    # Overlay the YAML + --set overrides onto the GLOBAL config singleton, which
    # DynamicServer / DynamicClassifier read (fedlap threads no config explicitly).
    with open(args.config) as f:
        data = yaml.safe_load(f) or {}
    overlay_config(config, data)
    Parser.apply_overrides(config, args.overrides)
    assert_cfg(config)

    base_seed = config["seed"]
    for i in range(args.repeat):
        config["seed"] = base_seed + i * 100
        run_once()


if __name__ == "__main__":
    main()
