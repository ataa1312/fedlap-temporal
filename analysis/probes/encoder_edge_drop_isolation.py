"""Prove gnn.encoder_edge_drop reaches the encoder and NOTHING else.

Run from codes/fedlap:
  ../../.venv/bin/python analysis/probes/encoder_edge_drop_isolation.py
"""
import os, sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src import *
from config.config import overlay_config
from config.assertions import assert_cfg
from registries import datasets
import src.datasets  # noqa
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer
from src.models import model_binders
from src.train import federated_orchestrator as fo


def build(drop, dt="f+es"):
    cfg = config
    with open("config/uci_gru.yaml") as f:
        overlay_config(cfg, yaml.safe_load(f))
    cfg["model"]["data_type"] = dt
    cfg["spectral"]["update_mode"] = "update"
    cfg["spectral"]["solver"] = "chebyshev"
    cfg["spectral"]["es_features"] = "spec"
    cfg["metric"]["repeat_new_split"] = True
    cfg["subgraph"]["num_subgraphs"] = 1
    cfg["gnn"]["encoder_edge_drop"] = drop
    cfg["wandb"]["mode"] = "disabled"
    cfg["train"]["auto_resume"] = False
    assert_cfg(cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    snaps = datasets[cfg["dataset"]["name"]](cfg)
    clients = partition_snapshots(snaps, cfg["subgraph"]["num_subgraphs"])
    srv = DynamicServer(snaps)
    for s in clients:
        srv.add_client(s)
    return srv


def probe(drop, n_snaps=4):
    srv = build(drop)
    print(f"\n=== encoder_edge_drop = {drop} ===")
    print(f"run_id: {srv._run_id()}")

    seen = []
    orig = model_binders.ModelBinder.encode

    def spy(self, x, edge_index, hs=None, edge_attr=None, keep_ratio=None, active_mask=None):
        seen.append((edge_index.shape[1], None if edge_attr is None else edge_attr.shape[0]))
        return orig(self, x, edge_index, hs, edge_attr, keep_ratio, active_mask)

    model_binders.ModelBinder.encode = spy
    # short run: stop after n_snaps snapshots
    srv.global_snaps = srv.global_snaps[: n_snaps + 1]
    for cl in srv.clients:
        cl.snaps = cl.snaps[: n_snaps + 1]
    try:
        res = srv.joint_train_w(FL=True, log=False)
    finally:
        model_binders.ModelBinder.encode = orig

    cl = srv.clients[0]
    rows = []
    for t in range(n_snaps + 1):
        s = cl.snaps[t]
        full = s.edge_index.shape[1]
        mp = getattr(s, "mp_edge_index", None)
        mp_n = full if mp is None else mp.shape[1]
        g_full = srv.global_snaps[t].edge_index.shape[1]
        g_has_mp = hasattr(srv.global_snaps[t], "mp_edge_index")
        rows.append((t, g_full, full, mp_n, mp_n / full if full else float("nan"), g_has_mp))
    print(" t  |E_glob| |E_client| |E_mp|  kept   global_has_mp")
    for r in rows:
        print(f"{r[0]:2d} {r[1]:8d} {r[2]:10d} {r[3]:7d} {r[4]:6.3f}   {r[5]}")

    ea_ok = all(
        (getattr(cl.snaps[t], "mp_edge_attr", None) is None)
        or (cl.snaps[t].mp_edge_attr.shape[0] == cl.snaps[t].mp_edge_index.shape[1])
        for t in range(n_snaps + 1)
    )
    print(f"mp_edge_attr rows == mp_edge_index cols: {ea_ok}")

    uniq = sorted(set(seen))
    print(f"distinct (edge_index_cols, edge_attr_rows) reaching ModelBinder.encode: {uniq[:12]}")
    mp_sizes = {getattr(cl.snaps[t], 'mp_edge_index', cl.snaps[t].edge_index).shape[1]
                for t in range(n_snaps + 1)}
    full_sizes = {cl.snaps[t].edge_index.shape[1] for t in range(n_snaps + 1)}
    seen_sizes = {a for a, _ in seen}
    print(f"encoder saw sizes subset of mp sizes: {seen_sizes <= mp_sizes}")
    if drop:
        print(f"encoder saw NO full-size snapshot: "
              f"{not (seen_sizes & (full_sizes - mp_sizes))}")

    fp = {
        "cum_edges": None if srv._cum_edges is None else
        (srv._cum_edges.shape[1], int(srv._cum_edges.sum())),
        "pos_test": [tuple(srv.global_snaps[t].pos_test.reshape(-1)[:6].tolist())
                     for t in range(1, n_snaps + 1)],
        "pos_test_n": [srv.global_snaps[t].pos_test.shape[1] for t in range(n_snaps + 1)],
        "glob_edges": [srv.global_snaps[t].edge_index.shape[1] for t in range(n_snaps + 1)],
        "cli_edges": [cl.snaps[t].edge_index.shape[1] for t in range(n_snaps + 1)],
        "cli_pos_train_n": [cl.snaps[t].pos_train.shape[1] for t in range(n_snaps + 1)],
        "repeat_frac": [m.get("repeat_frac") for m in res["metrics_history"]],
    }
    return fp, res


if __name__ == "__main__":
    fp0, r0 = probe(0.0)
    fp5, r5 = probe(0.5)
    fp75, r75 = probe(0.75)
    print("\n=== INVARIANTS across p (must be identical) ===")
    for k in ("cum_edges", "pos_test", "pos_test_n", "glob_edges", "cli_edges",
              "cli_pos_train_n", "repeat_frac"):
        same = (fp0[k] == fp5[k] == fp75[k])
        print(f"{k:18s} identical={same}")
        if not same:
            print(f"   p=0.00 {fp0[k]}")
            print(f"   p=0.50 {fp5[k]}")
            print(f"   p=0.75 {fp75[k]}")
    print("\nmean_mrr p=0.00 %.4f | p=0.50 %.4f | p=0.75 %.4f"
          % (r0["mean_mrr"], r5["mean_mrr"], r75["mean_mrr"]))
