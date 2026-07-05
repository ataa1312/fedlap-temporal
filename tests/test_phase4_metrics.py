import copy
import math
import random
import numpy as np
import torch
import pytest
from torch_geometric.data import Data
import src
from src.metrics.classification import binary_classification_metrics
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer, _weighted_mean_metrics
from src.dynamic_client import DynamicClient
from src.train.federated_orchestrator import _partition_edges_per_snapshot
from registries import losses

@pytest.fixture
def global_config_restore():
    original_registry = copy.deepcopy(src.config._registry)
    try:
        yield src.config
    finally:
        src.config._registry.clear()
        src.config._registry.update(original_registry)

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def make_server(global_snaps, client_snaps):
    server = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server.add_client(snaps)
    return server

def make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42):
    g = torch.Generator().manual_seed(seed)
    snaps = []
    for _ in range(num_snaps):
        edges = set()
        while len(edges) < 16:
            u = torch.randint(0, N, (1,), generator=g).item()
            v = torch.randint(0, N, (1,), generator=g).item()
            if u != v:
                edges.add((u, v))
        edge_index = torch.tensor(list(edges), dtype=torch.long).t()
        edge_attr = torch.randn(edge_index.size(1), W, generator=g)
        x = torch.ones(N, 1)
        snap = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)
        snap.node_ids = torch.arange(N)
        snaps.append(snap)
    return snaps

def test_mcc_perfect():
    logits = torch.tensor([2.0, -1.0, 3.0, -2.0])
    labels = torch.tensor([1, 0, 1, 0])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    assert metrics["mcc"] == 1.0

def test_mcc_anticorrelated():
    logits = torch.tensor([2.0, -2.0, 2.0, -2.0])
    labels = torch.tensor([0, 1, 0, 1])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    assert metrics["mcc"] == -1.0

def test_mcc_known_value():
    logits = torch.tensor([1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    assert metrics["mcc"] == 0.5
    assert metrics["f1"] == 0.75
    assert metrics["accuracy"] == 0.75

def test_mcc_zero_den():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([1, 0, 1, 0])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    assert metrics["mcc"] == 0.0

def test_best_threshold_separable():
    logits = torch.tensor([2.0, 2.0, -2.0, -2.0])
    labels = torch.tensor([1, 1, 0, 0])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    assert metrics["best_threshold"] == 2.0

def test_best_threshold_one_class():
    logits = torch.tensor([1.0, 2.0, 3.0])
    labels = torch.tensor([0, 0, 0])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    assert metrics["best_threshold"] == 0.0

def test_best_threshold_informational_invariant():
    logits = torch.tensor([3.0, 2.0, 1.0, -1.0])
    labels = torch.tensor([1, 0, 0, 0])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    assert metrics["best_threshold"] == 3.0
    assert metrics["f1"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert math.isclose(metrics["mcc"], 1.0 / 3.0, rel_tol=1e-5)

def test_dict_keys():
    logits = torch.tensor([2.0, -1.0, 3.0, -2.0])
    labels = torch.tensor([1, 0, 1, 0])
    metrics = binary_classification_metrics(logits, labels, threshold=0.0)
    expected_new = {"mcc", "best_threshold"}
    expected_original = {"accuracy", "precision", "recall", "f1", "roc_auc", "ap"}
    assert expected_new.issubset(metrics.keys())
    assert expected_original.issubset(metrics.keys())

def test_weighted_mean_metrics():
    metrics_list = [{"a": 1.0}, {"a": 0.0}]
    weights = [3.0, 1.0]
    res = _weighted_mean_metrics(metrics_list, weights)
    assert math.isclose(res["a"], 0.75, rel_tol=1e-5)

    metrics_list2 = [{"a": 2.0, "b": float("nan")}, {"a": 4.0, "b": 6.0}]
    weights2 = [1.0, 1.0]
    res2 = _weighted_mean_metrics(metrics_list2, weights2)
    assert math.isclose(res2["a"], 3.0, rel_tol=1e-5)
    assert math.isclose(res2["b"], 6.0, rel_tol=1e-5)

    metrics_list3 = [{"a": float("nan")}, {"a": float("nan")}]
    weights3 = [2.0, 3.0]
    res3 = _weighted_mean_metrics(metrics_list3, weights3)
    assert math.isnan(res3["a"])

    res4 = _weighted_mean_metrics([], [])
    assert res4 == {}

def test_server_eval_mrr_mcc(global_config_restore):
    global_config_restore["dataset"]["task"] = "link_pred"
    global_config_restore["dataset"]["edge_dim"] = 1
    global_config_restore["dataset"]["node_encoder"] = False
    global_config_restore["dataset"]["edge_encoder"] = False
    global_config_restore["model"]["data_type"] = "feature"
    global_config_restore["model"]["edge_decoding"] = "concat"
    global_config_restore["model"]["loss_fun"] = "bce_with_logits"
    global_config_restore["gnn"]["dims"] = [16, 16]
    global_config_restore["gnn"]["dims_pre_mp"] = []
    global_config_restore["gnn"]["dims_post_mp"] = []
    global_config_restore["gnn"]["embed_update_method"] = "gru"
    global_config_restore["gnn"]["l2norm"] = False
    global_config_restore["model"]["iterations"] = 2
    global_config_restore["model"]["local_epochs"] = 2
    global_config_restore["train"]["internal_validation_tolerance"] = 5
    global_config_restore["metric"]["mrr_method"] = "min"
    global_config_restore["experimental"]["rank_eval_multiplier"] = 50
    global_config_restore["dataset"]["split"] = [0.8, 0.1, 0.1]
    global_config_restore["optim"]["optimizer"] = "adam"
    global_config_restore["optim"]["base_lr"] = 0.005
    global_config_restore["optim"]["scheduler"] = "none"
    global_config_restore["seed"] = 42

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    server = make_server(global_snaps, client_snaps)
    server.initialize_FL()
    
    _partition_edges_per_snapshot(server.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
        
    mrr_k = 50
    mrr_method = "min"
    
    server.classifier.model.train()
    assert server.classifier.model.training
    
    mrr, metrics = server._eval_mrr(0, mrr_k, mrr_method)
    
    assert server.classifier.model.training
    assert isinstance(mrr, float)
    assert 0.0 <= mrr <= 1.0
    assert isinstance(metrics, dict)
    expected_keys = {"roc_auc", "ap", "f1", "mcc", "best_threshold", "accuracy", "precision", "recall"}
    assert expected_keys.issubset(metrics.keys())

    original_pos_test = server.global_snaps[1].pos_test
    server.global_snaps[1].pos_test = torch.empty((2, 0), dtype=torch.long)
    mrr_none, metrics_none = server._eval_mrr(0, mrr_k, mrr_method)
    assert mrr_none is None
    assert metrics_none is None
    server.global_snaps[1].pos_test = original_pos_test

    loss_fn = losses["bce_with_logits"]
    mrr_loc, metrics_loc = server._eval_mrr_local(0, loss_fn, mrr_k, mrr_method)
    assert isinstance(mrr_loc, float)
    assert 0.0 <= mrr_loc <= 1.0
    assert isinstance(metrics_loc, dict)
    assert expected_keys.issubset(metrics_loc.keys())

    original_client_pos_tests = [cl.snaps[1].pos_test for cl in server.clients]
    for cl in server.clients:
        cl.snaps[1].pos_test = torch.empty((2, 0), dtype=torch.long)
    mrr_loc_none, metrics_loc_none = server._eval_mrr_local(0, loss_fn, mrr_k, mrr_method)
    assert mrr_loc_none is None
    assert metrics_loc_none is None
    for cl, opt in zip(server.clients, original_client_pos_tests):
        cl.snaps[1].pos_test = opt

def test_joint_train_w_metrics(global_config_restore):
    global_config_restore["dataset"]["task"] = "link_pred"
    global_config_restore["dataset"]["edge_dim"] = 1
    global_config_restore["dataset"]["node_encoder"] = False
    global_config_restore["dataset"]["edge_encoder"] = False
    global_config_restore["model"]["data_type"] = "feature"
    global_config_restore["model"]["edge_decoding"] = "concat"
    global_config_restore["model"]["loss_fun"] = "bce_with_logits"
    global_config_restore["gnn"]["dims"] = [16, 16]
    global_config_restore["gnn"]["dims_pre_mp"] = []
    global_config_restore["gnn"]["dims_post_mp"] = []
    global_config_restore["gnn"]["embed_update_method"] = "gru"
    global_config_restore["gnn"]["l2norm"] = False
    global_config_restore["model"]["iterations"] = 2
    global_config_restore["model"]["local_epochs"] = 2
    global_config_restore["train"]["internal_validation_tolerance"] = 5
    global_config_restore["metric"]["mrr_method"] = "min"
    global_config_restore["experimental"]["rank_eval_multiplier"] = 50
    global_config_restore["dataset"]["split"] = [0.8, 0.1, 0.1]
    global_config_restore["optim"]["optimizer"] = "adam"
    global_config_restore["optim"]["base_lr"] = 0.005
    global_config_restore["optim"]["scheduler"] = "none"
    global_config_restore["seed"] = 42

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)

    server_fl = make_server(global_snaps, client_snaps)
    res_fl = server_fl.joint_train_w(FL=True, epochs=2)
    
    assert "mean_mrr" in res_fl
    assert "std_mrr" in res_fl
    assert "mrr_history" in res_fl
    assert "mean_metrics" in res_fl
    assert "metrics_history" in res_fl
    
    assert len(res_fl["metrics_history"]) == len(res_fl["mrr_history"])
    expected_keys = {"roc_auc", "ap", "f1", "mcc", "best_threshold", "accuracy", "precision", "recall"}
    assert expected_keys.issubset(res_fl["mean_metrics"].keys())
    for entry in res_fl["metrics_history"]:
        assert expected_keys.issubset(entry.keys())

    server_loc = make_server(global_snaps, client_snaps)
    res_loc = server_loc.joint_train_w(FL=False, epochs=2)
    
    assert "mean_mrr" in res_loc
    assert "std_mrr" in res_loc
    assert "mrr_history" in res_loc
    assert "mean_metrics" in res_loc
    assert "metrics_history" in res_loc
    
    assert len(res_loc["metrics_history"]) == len(res_loc["mrr_history"])
    assert expected_keys.issubset(res_loc["mean_metrics"].keys())
    for entry in res_loc["metrics_history"]:
        assert expected_keys.issubset(entry.keys())
