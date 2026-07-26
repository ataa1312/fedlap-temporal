"""Restore-best semantics of the two nested early stops.

Both run on the default path of every experiment: `local_finetune` early-stops
on the client's internal validation and restores its best state, and the FedAvg
round loop early-stops on the aggregated client val loss and restores its best
aggregate. If either kept the LAST state instead of the best, every reported
number would move while nothing else in the suite would notice.
"""

import copy

import pytest
import torch
from torch_geometric.data import Data

import src
import src.dynamic_client as dc_mod
from src.dynamic_client import DynamicClient
from src.dynamic_server import DynamicServer
from src.train.federated_orchestrator import (
    _clone_state,
    _partition_edges_per_snapshot,
)
from src.utils.graph_partitioning import partition_snapshots
from registries import losses


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


def tiny_run_config(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["model"]["loss_fun"] = "bce_with_logits"
    config["gnn"]["dims"] = [8, 8]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = False
    config["metric"]["mrr_method"] = "min"
    config["experimental"]["rank_eval_multiplier"] = 20
    config["optim"]["optimizer"] = "adam"
    config["optim"]["base_lr"] = 0.05  # big steps so states are clearly distinct
    config["optim"]["scheduler"] = "none"
    config["meta"]["is_meta"] = False
    config["train"]["auto_resume"] = False
    config["seed"] = 42


def make_dense_snapshots(num_nodes=16, num_snaps=2, seed=5):
    g = torch.Generator().manual_seed(seed)
    edges = [(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    snaps = []
    for _ in range(num_snaps):
        snap = Data(
            x=torch.ones(num_nodes, 1),
            edge_index=edge_index.clone(),
            edge_attr=torch.randn(edge_index.size(1), 1, generator=g),
            num_nodes=num_nodes,
        )
        snap.node_ids = torch.arange(num_nodes)
        snaps.append(snap)
    return snaps


def states_equal(a, b):
    for key, val in a.items():
        other = b[key]
        if isinstance(val, dict):
            if not states_equal(val, other):
                return False
        elif not torch.equal(val, other):
            return False
    return True


def test_local_finetune_restores_its_best_state(config):
    tiny_run_config(config)
    config["train"]["internal_validation_tolerance"] = 10  # never break early
    snaps = make_dense_snapshots()
    client = DynamicClient(snaps, id=0)
    client.initialize()
    _partition_edges_per_snapshot(client.snaps, [0.8, 0.1, 0.1], seed=42)

    scripted = [0.5, 0.1, 0.9, 0.8, 0.7]  # the best epoch is index 1, not the last
    candidates = []
    real_eval_loss = dc_mod._step_eval_loss_pair

    def scripted_eval_loss(model, *args, **kwargs):
        candidates.append(_clone_state(model.state_dict()))
        return scripted[len(candidates) - 1]

    dc_mod._step_eval_loss_pair = scripted_eval_loss
    try:
        client.local_finetune(0, len(scripted), losses["bce_with_logits"])
    finally:
        dc_mod._step_eval_loss_pair = real_eval_loss

    assert len(candidates) == len(scripted)
    best = candidates[scripted.index(min(scripted))]
    assert states_equal(client.classifier.state_dict(), best)
    assert not states_equal(client.classifier.state_dict(), candidates[-1])


def test_local_finetune_stops_after_patience(config):
    tiny_run_config(config)
    config["train"]["internal_validation_tolerance"] = 2
    snaps = make_dense_snapshots()
    client = DynamicClient(snaps, id=0)
    client.initialize()
    _partition_edges_per_snapshot(client.snaps, [0.8, 0.1, 0.1], seed=42)

    scripted = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # monotonically worse after epoch 0
    calls = []
    real_eval_loss = dc_mod._step_eval_loss_pair

    def scripted_eval_loss(model, *args, **kwargs):
        calls.append(1)
        return scripted[len(calls) - 1]

    dc_mod._step_eval_loss_pair = scripted_eval_loss
    try:
        client.local_finetune(0, len(scripted), losses["bce_with_logits"])
    finally:
        dc_mod._step_eval_loss_pair = real_eval_loss

    # epoch 0 sets best, epochs 1 and 2 are stale -> break. local_epochs is a MAX.
    assert len(calls) == 3


def test_round_loop_restores_the_best_aggregate(config):
    tiny_run_config(config)
    config["train"]["internal_validation_tolerance"] = 10
    config["model"]["iterations"] = 4
    config["model"]["local_epochs"] = 1
    snaps = make_dense_snapshots(num_snaps=2)
    server = DynamicServer(snaps)
    for client_snaps in partition_snapshots(snaps, 2):
        server.add_client(client_snaps)

    scripted = [0.9, 0.2, 0.7, 0.8]  # the best round is index 1, not the last
    round_states = []
    real_val_loss = DynamicClient.val_loss

    def scripted_val_loss(self, t, loss_fn):
        if self is server.clients[0]:
            # every client is holding the just-broadcast aggregate here
            round_states.append(_clone_state(server.state_dict()))
        return scripted[len(round_states) - 1]

    DynamicClient.val_loss = scripted_val_loss
    try:
        server.joint_train_w(FL=True)
    finally:
        DynamicClient.val_loss = real_val_loss

    assert len(round_states) == len(scripted)
    best = round_states[scripted.index(min(scripted))]
    assert states_equal(server.state_dict(), best)
    assert not states_equal(server.state_dict(), round_states[-1])
    # clients end synced to the restored best, not to their own last local step
    for client in server.clients:
        assert states_equal(client.state_dict(), best)


def test_round_loop_stops_after_patience(config):
    tiny_run_config(config)
    config["train"]["internal_validation_tolerance"] = 2
    config["model"]["iterations"] = 6
    config["model"]["local_epochs"] = 1
    snaps = make_dense_snapshots(num_snaps=2)
    server = DynamicServer(snaps)
    for client_snaps in partition_snapshots(snaps, 2):
        server.add_client(client_snaps)

    scripted = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rounds = []
    real_val_loss = DynamicClient.val_loss

    def scripted_val_loss(self, t, loss_fn):
        if self is server.clients[0]:
            rounds.append(1)
        return scripted[len(rounds) - 1]

    DynamicClient.val_loss = scripted_val_loss
    try:
        server.joint_train_w(FL=True)
    finally:
        DynamicClient.val_loss = real_val_loss

    assert len(rounds) == 3  # round 0 sets best, rounds 1-2 stale -> break
