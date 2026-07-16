import math
import random
import copy
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from torch_geometric.data import Data

import src
from src import device
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer
from src.dynamic_client import DynamicClient
from config.assertions import assert_cfg
from config.config import get_default_config
from main import _wandb_meta

def make_toy_snapshots(N=30, W=1, num_snaps=4, seed=42, num_edges=120):
    g = torch.Generator().manual_seed(seed)
    snaps = []
    for _ in range(num_snaps):
        edges = set()
        while len(edges) < num_edges:
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

def states_are_close(s1, s2):
    if set(s1.keys()) != set(s2.keys()):
        return False
    for k in s1:
        t1 = s1[k]
        t2 = s2[k]
        if torch.is_tensor(t1) and torch.is_tensor(t2):
            if t1.shape != t2.shape:
                return False
            if not torch.allclose(t1, t2, atol=1e-5, rtol=1e-5):
                return False
        else:
            if isinstance(t1, np.ndarray) and isinstance(t2, np.ndarray):
                if not np.array_equal(t1, t2):
                    return False
            else:
                try:
                    if t1 != t2:
                        return False
                except Exception:
                    if str(t1) != str(t2):
                        return False
    return True

# 1. test_fl_default_is_true
def test_fl_default_is_true():
    cfg = get_default_config()
    assert cfg["federated"]["fl"] is True
    assert isinstance(cfg["federated"]["fl"], bool)

# 2. test_assert_cfg_fl_bool_guard
def test_assert_cfg_fl_bool_guard(config):
    # Check valid cases
    config["federated"]["fl"] = True
    assert_cfg(config)
    
    config["federated"]["fl"] = False
    assert_cfg(config)
    
    # Check invalid cases
    for val in ["flase", "true", 1, None]:
        config["federated"]["fl"] = val
        with pytest.raises(ValueError) as exc:
            assert_cfg(config)
        assert "federated.fl" in str(exc.value)

# 3. test_local_path_does_no_aggregation
def test_local_path_does_no_aggregation(config, monkeypatch):
    import src.dynamic_server
    
    # Tiny run config
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["iterations"] = 2
    config["model"]["local_epochs"] = 2
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    
    global_snaps = make_toy_snapshots(N=30, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 3)
    
    sum_lod_calls = 0
    share_weights_calls = 0
    
    orig_sum_lod = src.dynamic_server.sum_lod
    orig_share_weights = DynamicServer.share_weights
    
    def spy_sum_lod(*args, **kwargs):
        nonlocal sum_lod_calls
        sum_lod_calls += 1
        return orig_sum_lod(*args, **kwargs)
        
    def spy_share_weights(self, *args, **kwargs):
        nonlocal share_weights_calls
        share_weights_calls += 1
        return orig_share_weights(self, *args, **kwargs)
        
    monkeypatch.setattr(src.dynamic_server, "sum_lod", spy_sum_lod)
    monkeypatch.setattr(DynamicServer, "share_weights", spy_share_weights)
    
    # FL=False path
    server_local = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server_local.add_client(snaps)
    server_local.joint_train_w(FL=False)
    
    assert sum_lod_calls == 0
    assert share_weights_calls == 1  # From initialize_FL
    
    # Reset
    sum_lod_calls = 0
    share_weights_calls = 0
    
    # FL=True path
    server_fed = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server_fed.add_client(snaps)
    server_fed.joint_train_w(FL=True)
    
    assert sum_lod_calls > 1
    assert share_weights_calls > 1

# 4. test_eval_dispatch_by_fl
def test_eval_dispatch_by_fl(config, monkeypatch):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["iterations"] = 2
    config["model"]["local_epochs"] = 2
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    
    global_snaps = make_toy_snapshots(N=30, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    eval_mrr_calls = 0
    eval_mrr_local_calls = 0
    
    orig_eval_mrr = DynamicServer._eval_mrr
    orig_eval_mrr_local = DynamicServer._eval_mrr_local
    
    def spy_eval_mrr(self, *args, **kwargs):
        nonlocal eval_mrr_calls
        eval_mrr_calls += 1
        return orig_eval_mrr(self, *args, **kwargs)
        
    def spy_eval_mrr_local(self, *args, **kwargs):
        nonlocal eval_mrr_local_calls
        eval_mrr_local_calls += 1
        return orig_eval_mrr_local(self, *args, **kwargs)
        
    monkeypatch.setattr(DynamicServer, "_eval_mrr", spy_eval_mrr)
    monkeypatch.setattr(DynamicServer, "_eval_mrr_local", spy_eval_mrr_local)
    
    # FL=False path
    server_local = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server_local.add_client(snaps)
    server_local.joint_train_w(FL=False)
    
    assert eval_mrr_local_calls > 0
    assert eval_mrr_calls == 0
    
    # Reset
    eval_mrr_calls = 0
    eval_mrr_local_calls = 0
    
    # FL=True path
    server_fed = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server_fed.add_client(snaps)
    server_fed.joint_train_w(FL=True)
    
    assert eval_mrr_calls > 0
    assert eval_mrr_local_calls == 0

# 5. test_local_clients_start_identical_then_diverge
def test_local_clients_start_identical_then_diverge(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["iterations"] = 2
    config["model"]["local_epochs"] = 2
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    config["optim"]["base_lr"] = 0.1  # higher LR to ensure they diverge
    config["seed"] = 42
    
    global_snaps = make_toy_snapshots(N=30, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    # 1. FL=False
    server_local = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server_local.add_client(snaps)
    server_local.initialize_FL()
    
    assert states_are_close(server_local.clients[0].state_dict(), server_local.clients[1].state_dict())
    
    server_local.joint_train_w(FL=False, epochs=2)
    assert not states_are_close(server_local.clients[0].state_dict(), server_local.clients[1].state_dict())
    
    # 2. FL=True
    server_fed = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server_fed.add_client(snaps)
    server_fed.initialize_FL()
    
    assert states_are_close(server_fed.clients[0].state_dict(), server_fed.clients[1].state_dict())
    
    server_fed.joint_train_w(FL=True, epochs=2)
    assert states_are_close(server_fed.clients[0].state_dict(), server_fed.clients[1].state_dict())

# 6. test_meta_is_fl_scoped
def test_meta_is_fl_scoped(config, monkeypatch):
    import src.dynamic_server
    
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["iterations"] = 2
    config["model"]["local_epochs"] = 2
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    
    config["meta"]["is_meta"] = True
    config["meta"]["method"] = "moving_average"
    config["meta"]["alpha"] = 0.5
    
    global_snaps = make_toy_snapshots(N=30, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    sum_lod_calls = 0
    orig_sum_lod = src.dynamic_server.sum_lod
    
    def spy_sum_lod(*args, **kwargs):
        nonlocal sum_lod_calls
        sum_lod_calls += 1
        return orig_sum_lod(*args, **kwargs)
        
    monkeypatch.setattr(src.dynamic_server, "sum_lod", spy_sum_lod)
    
    # FL=False
    server_local = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server_local.add_client(snaps)
    server_local.joint_train_w(FL=False)
    
    assert sum_lod_calls == 0
    
    # Reset
    sum_lod_calls = 0
    
    # FL=True
    server_fed = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server_fed.add_client(snaps)
    server_fed.joint_train_w(FL=True)
    
    assert sum_lod_calls > 0

# 7. test_eval_mrr_local_weighting
class MockClient:
    def __init__(self, num_nodes, has_test_edges=True, mrr=0.5, metrics=None):
        self._num_nodes = num_nodes
        self.snaps = [
            SimpleNamespace(),
            SimpleNamespace()
        ]
        if has_test_edges:
            self.snaps[1].pos_test = torch.ones(2, 5)
        else:
            self.snaps[1].pos_test = torch.zeros(2, 0)
            
        self.classifier = SimpleNamespace()
        self.mrr = mrr
        self.metrics = metrics or {"auc": 0.9}
        
    def _hs_in(self):
        return None
        
    def num_nodes(self):
        return self._num_nodes

def test_eval_mrr_local_weighting(monkeypatch):
    import src.dynamic_server
    
    snap0 = SimpleNamespace(x=torch.ones(1, 1), edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=torch.zeros(0, 1), num_nodes=1)
    server = DynamicServer([snap0, snap0])
    
    client1 = MockClient(num_nodes=10, has_test_edges=True, mrr=0.4, metrics={"auc": 0.8, "ap": 0.6})
    client2 = MockClient(num_nodes=30, has_test_edges=True, mrr=0.8, metrics={"auc": 0.9, "ap": 0.7})
    client3 = MockClient(num_nodes=20, has_test_edges=False)
    
    server.clients = [client1, client2, client3]
    
    def mock_step_eval(classifier, snap_t, snap_t1, hs, loss_fn, device_arg, val_flag, mrr_k, mrr_method):
        for cl in [client1, client2, client3]:
            if cl.snaps[1] is snap_t1:
                return None, cl.mrr, cl.metrics
        return None, float("nan"), {}
        
    monkeypatch.setattr(src.dynamic_server, "_step_eval_with_mrr_pair", mock_step_eval)
    
    mrr, metrics = server._eval_mrr_local(t=0, loss_fn=None, mrr_k=50, mrr_method="min")
    
    assert mrr is not None
    assert math.isclose(mrr, 0.7, abs_tol=1e-5)
    assert math.isclose(metrics["auc"], 0.875, abs_tol=1e-5)
    assert math.isclose(metrics["ap"], 0.675, abs_tol=1e-5)
    
    # returns (None, None) when no client has test edges
    client_empty1 = MockClient(num_nodes=10, has_test_edges=False)
    client_empty2 = MockClient(num_nodes=20, has_test_edges=False)
    server_empty = DynamicServer([snap0, snap0])
    server_empty.clients = [client_empty1, client_empty2]
    
    mrr_empty, metrics_empty = server_empty._eval_mrr_local(t=0, loss_fn=None, mrr_k=50, mrr_method="min")
    assert mrr_empty is None
    assert metrics_empty is None

# 8. test_wandb_meta_local_marker
def test_wandb_meta_local_marker(config):
    config["dataset"]["name"] = "toy_dataset"
    config["subgraph"]["num_subgraphs"] = 3
    config["gnn"]["embed_update_method"] = "gru"
    config["model"]["data_type"] = "feature"
    config["federated"]["sfv_share"] = "local"
    config["spectral"]["update_mode"] = "update"
    config["spectral"]["use_procrustes"] = True
    config["dataset"]["snapshot_freq"] = "W"
    config["seed"] = 42
    
    # 1. FL=False
    config["federated"]["fl"] = False
    group, cfg, tags = _wandb_meta()
    
    assert "local" in group.split("_")
    assert "local" in tags
    assert cfg["fl"] is False
    
    # 2. FL=True
    config["federated"]["fl"] = True
    group, cfg, tags = _wandb_meta()
    
    assert "local" not in group.split("_")
    assert "local" not in tags
    assert cfg["fl"] is True
