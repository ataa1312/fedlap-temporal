import os
import copy
import math
import random
import numpy as np
import torch
import pytest
from torch_geometric.data import Data
import src
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer, _clone_state
from src.dynamic_client import DynamicClient
from src.train.federated_orchestrator import _partition_edges_per_snapshot
from registries import losses


class _FakeSFVSmodel:
    """Minimal smodel implementing the get_SFV/set_SFV protocol.

    Checkpointing goes through that protocol rather than reaching into
    smodel.graph, so the fake has to implement it; .graph is kept so the
    round-trip assertions below can still read the tensor directly."""

    def __init__(self, x):
        import types as _t
        self.graph = _t.SimpleNamespace(x=x)

    def get_SFV(self):
        return self.graph.x

    def set_SFV(self, w):
        with torch.no_grad():
            self.graph.x.copy_(w.to(self.graph.x.device))


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

def setup_tiny_config(global_config_restore, tmp_path):
    global_config_restore["train"]["auto_resume"] = True
    global_config_restore["train"]["ckpt_period"] = 1
    global_config_restore["train"]["ckpt_clean"] = True
    global_config_restore["train"]["ckpt_dir"] = str(tmp_path)
    
    global_config_restore["dataset"]["name"] = "uci"
    global_config_restore["dataset"]["task"] = "link_pred"
    global_config_restore["dataset"]["edge_dim"] = 1
    global_config_restore["dataset"]["node_encoder"] = False
    global_config_restore["dataset"]["edge_encoder"] = False
    global_config_restore["dataset"]["snapshot_freq"] = "W"
    global_config_restore["dataset"]["split"] = [0.8, 0.1, 0.1]
    
    global_config_restore["subgraph"]["num_subgraphs"] = 1
    global_config_restore["subgraph"]["partitioning"] = "random"
    
    global_config_restore["model"]["data_type"] = "feature"
    global_config_restore["model"]["edge_decoding"] = "concat"
    global_config_restore["model"]["loss_fun"] = "bce_with_logits"
    global_config_restore["model"]["iterations"] = 1
    global_config_restore["model"]["local_epochs"] = 1
    global_config_restore["model"]["smodel_type"] = "LanczosLaplace"
    
    global_config_restore["gnn"]["dims"] = [16, 16]
    global_config_restore["gnn"]["dims_pre_mp"] = []
    global_config_restore["gnn"]["dims_post_mp"] = []
    global_config_restore["gnn"]["embed_update_method"] = "gru"
    global_config_restore["gnn"]["l2norm"] = False
    global_config_restore["gnn"]["keep_ratio_mode"] = "linear"
    
    global_config_restore["spectral"]["update_mode"] = "keep"
    global_config_restore["spectral"]["use_procrustes"] = True
    
    global_config_restore["federated"]["sfv_share"] = "local"
    
    global_config_restore["train"]["internal_validation_tolerance"] = 5
    global_config_restore["metric"]["mrr_method"] = "min"
    global_config_restore["experimental"]["rank_eval_multiplier"] = 50
    global_config_restore["optim"]["optimizer"] = "adam"
    global_config_restore["optim"]["base_lr"] = 0.005
    global_config_restore["optim"]["scheduler"] = "none"
    global_config_restore["seed"] = 42

def test_gating_off(global_config_restore, tmp_path):
    setup_tiny_config(global_config_restore, tmp_path)
    global_config_restore["train"]["auto_resume"] = False

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=3, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    
    server = make_server(global_snaps, client_snaps)
    server.initialize_FL()
    _partition_edges_per_snapshot(server.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
        
    res = server.joint_train_w(FL=True)
    assert len(os.listdir(tmp_path)) == 0

def test_save_cadence(global_config_restore, tmp_path, monkeypatch):
    setup_tiny_config(global_config_restore, tmp_path)
    global_config_restore["train"]["ckpt_period"] = 2
    global_config_restore["train"]["ckpt_clean"] = False

    save_calls = []
    orig_save = DynamicServer._save_partial_ckpt
    def spied_save(self, t, w_init, mrr_history, metrics_history):
        save_calls.append(t)
        return orig_save(self, t, w_init, mrr_history, metrics_history)
    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", spied_save)

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    
    server = make_server(global_snaps, client_snaps)
    server.initialize_FL()
    _partition_edges_per_snapshot(server.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
        
    server.joint_train_w(FL=True)
    
    assert save_calls == [1]
    ckpt_path, done_path = server._ckpt_paths()
    assert os.path.exists(ckpt_path)
    assert os.path.exists(done_path)

def test_round_trip_fidelity(global_config_restore, tmp_path):
    setup_tiny_config(global_config_restore, tmp_path)

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    server = make_server(global_snaps, client_snaps)
    server.initialize_FL()
    
    import types
    server.classifier.smodel = _FakeSFVSmodel(torch.randn(3, 3))
    for c, cl in enumerate(server.clients):
        cl.classifier.smodel = _FakeSFVSmodel(torch.randn(3, 3))
        cl.hs = [torch.randn(1, 16)]
        
    server._first_spectral = types.SimpleNamespace(U=torch.randn(5, 5), D=torch.randn(5))
    server._prev_spectral = types.SimpleNamespace(U=torch.randn(5, 5), D=torch.randn(5))
    server._cum_edges = 123
    
    w_init = _clone_state(server.state_dict())
    mrr_history = [0.1, 0.2]
    metrics_history = [{"roc_auc": 0.9}]
    
    server._save_partial_ckpt(1, w_init, mrr_history, metrics_history)
    
    server_fresh = make_server(global_snaps, client_snaps)
    server_fresh.initialize_FL()
    
    server_fresh.classifier.smodel = _FakeSFVSmodel(torch.zeros(3, 3))
    for cl in server_fresh.clients:
        cl.classifier.smodel = _FakeSFVSmodel(torch.zeros(3, 3))
        
    resumed = server_fresh._load_partial_ckpt()
    assert resumed is not None
    t_start, w_init_restored, mrr_restored, metrics_restored = resumed
    
    assert t_start == 2
    assert mrr_restored == mrr_history
    assert metrics_restored == metrics_history
    
    for p1, p2 in zip(server.classifier.parameters(), server_fresh.classifier.parameters()):
        assert torch.allclose(p1, p2)
        
    assert torch.allclose(server_fresh.classifier.smodel.graph.x, server.classifier.smodel.graph.x)
    for c1, c2 in zip(server.clients, server_fresh.clients):
        assert torch.allclose(c1.classifier.smodel.graph.x, c2.classifier.smodel.graph.x)
        for h1, h2 in zip(c1.hs, c2.hs):
            assert torch.allclose(h1, h2)
            
    assert torch.allclose(server_fresh._first_spectral.U, server._first_spectral.U)
    assert torch.allclose(server_fresh._first_spectral.D, server._first_spectral.D)
    assert torch.allclose(server_fresh._prev_spectral.U, server._prev_spectral.U)
    assert torch.allclose(server_fresh._prev_spectral.D, server._prev_spectral.D)
    assert server_fresh._cum_edges == server._cum_edges
    
    def assert_dict_equal(d1, d2):
        for k, v in d1.items():
            if isinstance(v, dict):
                assert_dict_equal(v, d2[k])
            else:
                assert torch.allclose(v, d2[k])
    assert_dict_equal(w_init_restored, w_init)

def test_resume_completeness(global_config_restore, tmp_path, monkeypatch):
    setup_tiny_config(global_config_restore, tmp_path)
    global_config_restore["train"]["ckpt_clean"] = False

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    
    class SimulatedCrash(Exception):
        pass
        
    orig_save = DynamicServer._save_partial_ckpt
    saved_mrr_prefix = None
    
    def mock_save(self, t, w_init, mrr_history, metrics_history):
        nonlocal saved_mrr_prefix
        orig_save(self, t, w_init, mrr_history, metrics_history)
        if t == 1:
            saved_mrr_prefix = list(mrr_history)
            raise SimulatedCrash("CRASH!")
            
    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", mock_save)
    
    server_crash = make_server(global_snaps, client_snaps)
    _partition_edges_per_snapshot(server_crash.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server_crash.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
        
    with pytest.raises(SimulatedCrash):
        server_crash.joint_train_w(FL=True)
        
    assert saved_mrr_prefix is not None
    assert len(saved_mrr_prefix) == 2
    
    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", orig_save)
    
    server_resume = make_server(global_snaps, client_snaps)
    _partition_edges_per_snapshot(server_resume.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server_resume.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
        
    res = server_resume.joint_train_w(FL=True)
    
    assert len(res["mrr_history"]) == 3
    assert res["mrr_history"][:2] == saved_mrr_prefix

def test_skip_on_done(global_config_restore, tmp_path, monkeypatch):
    setup_tiny_config(global_config_restore, tmp_path)
    global_config_restore["subgraph"]["num_subgraphs"] = 1

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=3, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    
    server = make_server(global_snaps, client_snaps)
    _partition_edges_per_snapshot(server.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
        
    res1 = server.joint_train_w(FL=True)
    assert not res1.get("_resumed_complete", False)
    
    ft_calls = 0
    orig_ft = DynamicClient.local_finetune
    def spied_ft(self, t, local_epochs, loss_fn):
        nonlocal ft_calls
        ft_calls += 1
        return orig_ft(self, t, local_epochs, loss_fn)
    monkeypatch.setattr(DynamicClient, "local_finetune", spied_ft)
    
    server2 = make_server(global_snaps, client_snaps)
    _partition_edges_per_snapshot(server2.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server2.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
        
    res2 = server2.joint_train_w(FL=True)
    assert res2.get("_resumed_complete") is True
    assert ft_calls == 0
    
    init_wandb_calls = 0
    def mock_init_wandb():
        nonlocal init_wandb_calls
        init_wandb_calls += 1
        return "mock_wandb_run"
    import main
    monkeypatch.setattr(main, "_init_wandb", mock_init_wandb)
    
    import registries
    original_uci = registries.datasets["uci"]
    registries.datasets["uci"] = lambda cfg: global_snaps
    try:
        main.run_once()
    finally:
        registries.datasets["uci"] = original_uci
        
    assert init_wandb_calls == 0

def test_identity(global_config_restore):
    setup_tiny_config(global_config_restore, "/tmp/dummy_ckpt_dir")
    global_config_restore["subgraph"]["num_subgraphs"] = 2
    
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=2, seed=42)
    server = DynamicServer(global_snaps)
    
    id1 = server._run_id()
    w_id1 = server._wandb_id()
    
    import main
    group, cfg, tags = main._wandb_meta()
    name = f"{group}_s{src.config['seed']}"
    assert main._wandb_id(name) == w_id1
    
    assert id1 == server._run_id()
    assert w_id1 == server._wandb_id()
    
    global_config_restore["seed"] = 100
    id2 = server._run_id()
    w_id2 = server._wandb_id()
    assert id1 != id2
    assert w_id1 != w_id2

def test_sfv_local_guard(global_config_restore, tmp_path):
    setup_tiny_config(global_config_restore, tmp_path)
    global_config_restore["model"]["data_type"] = "f+s"
    global_config_restore["subgraph"]["num_subgraphs"] = 2

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    server = make_server(global_snaps, client_snaps)
    server.initialize_FL(data_type="feature")
    
    import types
    server.classifier.smodel = _FakeSFVSmodel(torch.ones(3, 3))
    for cl in server.clients:
        cl.classifier.smodel = _FakeSFVSmodel(torch.ones(3, 3))
        cl.hs = [torch.randn(1, 16)]
        
    server._save_partial_ckpt(1, None, [], [])
    
    client0 = server.clients[0]
    with torch.no_grad():
        client0.classifier.smodel.graph.x.fill_(999.0)
        
    assert torch.all(client0.classifier.smodel.graph.x == 999.0)
    
    server._load_partial_ckpt()
    assert torch.allclose(client0.classifier.smodel.graph.x, torch.ones(3, 3))

def test_robustness(global_config_restore, tmp_path):
    setup_tiny_config(global_config_restore, tmp_path)
    
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=2, seed=42)
    server = DynamicServer(global_snaps)
    
    assert server._load_partial_ckpt() is None
    
    ckpt_path, _ = server._ckpt_paths()
    with open(ckpt_path, "wb") as f:
        f.write(b"invalid data")
    assert server._load_partial_ckpt() is None
    
    ckpt = {
        "run_id": "different_run_id",
        "t": 5,
        "server_state": {},
        "client_hs": []
    }
    torch.save(ckpt, ckpt_path)
    assert server._load_partial_ckpt() is None

def test_feature_non_spectral(global_config_restore, tmp_path):
    setup_tiny_config(global_config_restore, tmp_path)
    global_config_restore["subgraph"]["num_subgraphs"] = 2

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=3, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    server = make_server(global_snaps, client_snaps)
    server.initialize_FL()
    
    assert server._first_spectral is None
    assert server._prev_spectral is None
    assert server._cum_edges is None
    
    server._save_partial_ckpt(0, None, [], [])
    
    server_fresh = make_server(global_snaps, client_snaps)
    server_fresh.initialize_FL()
    resumed = server_fresh._load_partial_ckpt()
    assert resumed is not None
    
    assert server_fresh._first_spectral is None
    assert server_fresh._prev_spectral is None
    assert server_fresh._cum_edges is None

def test_wandb_offline(global_config_restore, tmp_path):
    setup_tiny_config(global_config_restore, tmp_path)
    global_config_restore["subgraph"]["num_subgraphs"] = 1

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=3, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    
    server = make_server(global_snaps, client_snaps)
    _partition_edges_per_snapshot(server.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
        
    logged_data = []
    def log_cb(t, mrr, metrics):
        logged_data.append((t, mrr, metrics))
        
    res = server.joint_train_w(FL=True, log_cb=log_cb)
    
    assert len(logged_data) == 2
    t0, mrr0, metrics0 = logged_data[0]
    assert t0 == 0
    assert isinstance(mrr0, float)
    assert "roc_auc" in metrics0
    
    w_id = server._wandb_id()
    server._save_partial_ckpt(0, None, [], [])
    ckpt_path, _ = server._ckpt_paths()
    loaded_ckpt = torch.load(ckpt_path, weights_only=False)
    assert loaded_ckpt["wandb_id"] == w_id
