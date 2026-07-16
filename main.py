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


def _wandb_meta():
    """Group/name/config/tags so multi-seed runs of one condition group and
    average in the wandb UI. group excludes seed (so seeds collapse together);
    name/config carry the seed. f+s/structure add update_mode + sfv_share."""
    ds = config["dataset"]["name"]
    m = config["model"]
    dt = m["data_type"]
    C = config["subgraph"]["num_subgraphs"]
    temporal = config["gnn"]["embed_update_method"]
    um = config["spectral"]["update_mode"]
    sfv = config["federated"]["sfv_share"]
    proc = config["spectral"]["use_procrustes"]
    sf = config["dataset"]["snapshot_freq"]
    fl = config["federated"]["fl"]
    custom_freq = isinstance(sf, str) and sf.endswith("s") and sf[:-1].isdigit()
    parts = [ds, temporal, str(dt), f"C{C}"]
    if not fl:  # local-only floor: keep it out of the federated groups' averages
        parts.append("local")
    if dt in ("f+s", "structure"):
        parts += [f"um-{um}", f"sfv-{sfv}"]
        if um in ("update", "recompute"):  # procrustes only applies to these modes
            parts.append(f"proc-{'on' if proc else 'off'}")
    if custom_freq:  # coarsened window; keep distinct from the default calendar-freq groups
        parts.append(f"freq-{sf}")
    group = "_".join(parts)
    cfg = {
        "dataset": ds, "temporal": temporal, "data_type": dt, "num_clients": C,
        "update_mode": um, "sfv_share": sfv, "use_procrustes": proc, "seed": config["seed"],
        "fl": fl,
        "iterations": m["iterations"], "local_epochs": m["local_epochs"],
        "base_lr": config["optim"]["base_lr"], "fusion": m["fusion"],
        "smodel_type": m["smodel_type"], "spectral_len": config["spectral"]["spectral_len"],
        "snapshot_freq": sf,
    }
    tags = [ds, str(dt), f"C{C}", temporal]
    if not fl:
        tags.append("local")
    if dt in ("f+s", "structure"):
        tags += [f"um-{um}", f"sfv-{sfv}"]
        if um in ("update", "recompute"):
            tags.append(f"proc-{'on' if proc else 'off'}")
    if custom_freq:
        tags.append(f"freq-{sf}")
        tags += ["coarse-snap", f"coarse-{round(int(sf[:-1]) / 86400)}d"]
    return group, cfg, tags


def _wandb_id(name):
    import hashlib
    # deterministic id (matches DynamicServer._wandb_id) so a resumed run continues
    # the SAME wandb run instead of minting a duplicate.
    return hashlib.sha1(name.encode()).hexdigest()[:20]


def _init_wandb():
    wb = config["wandb"] if "wandb" in config else None
    mode = wb["mode"] if (wb is not None and "mode" in wb) else "disabled"
    if mode == "disabled":
        return None
    import wandb

    group, cfg, tags = _wandb_meta()
    name = f"{group}_s{config['seed']}"
    run = wandb.init(
        project=wb["project"], mode=mode, reinit=True,
        group=group, name=name, id=_wandb_id(name), resume="allow",
        tags=tags, config=cfg,
    )
    # plot per-snapshot metrics against the snapshot index (not the wandb step) so
    # a resumed run appends cleanly without step-monotonicity conflicts.
    wandb.define_metric("snapshot/idx")
    wandb.define_metric("snapshot/*", step_metric="snapshot/idx")
    return run


def _wandb_snapshot_logger(wb):
    if wb is None:
        return None
    import wandb

    def _cb(t, mrr, metrics):
        d = {"snapshot/idx": t, "snapshot/mrr": mrr}
        for k, v in (metrics or {}).items():
            d[f"snapshot/{k}"] = v
        wandb.log(d)

    return _cb


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

    # Init wandb BEFORE training for LIVE per-snapshot logging that survives resume
    # (deterministic id -> a resumed run continues the same wandb run). Skip entirely
    # if this exact run already completed, so a re-launch doesn't duplicate it.
    ckpt_on = config["train"]["auto_resume"] and config["train"]["ckpt_period"] > 0
    already_done = ckpt_on and server._load_done_ckpt() is not None
    wb = None if already_done else _init_wandb()

    results = server.joint_train_w(
        FL=config["federated"]["fl"], log_cb=_wandb_snapshot_logger(wb)
    )
    mm = results.get("mean_metrics") or {}
    LOGGER.info(
        f"RESULT dataset={name} clients={n_clients} fl={config['federated']['fl']} "
        f"seed={config['seed']} "
        f"mean_mrr={results['mean_mrr']} std={results['std_mrr']} "
        f"auc={mm.get('roc_auc')} ap={mm.get('ap')} f1={mm.get('f1')} mcc={mm.get('mcc')} "
        f"snapshots={len(results['mrr_history'])}"
    )

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
