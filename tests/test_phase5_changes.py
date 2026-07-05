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
from src.dynamic_server import DynamicServer
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

def test_procrustes_recovers_known_rotation():
    seed_all(42)
    A = torch.randn(6, 3)
    from_U, _ = torch.linalg.qr(A)
    
    B = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(B)
    
    to_U = from_U @ Q
    
    g = Graph(x=torch.ones(5, 1), edge_index=torch.tensor([[0], [0]], dtype=torch.long), node_ids=torch.arange(5))
    aligned = g.procrustes_project(from_U, to_U)
    
    assert torch.allclose(aligned, to_U, rtol=1e-5, atol=1e-5)

def test_procrustes_properties():
    seed_all(42)
    A = torch.randn(6, 3)
    from_U, _ = torch.linalg.qr(A)
    to_U = torch.randn(6, 3)
    
    g = Graph(x=torch.ones(5, 1), edge_index=torch.tensor([[0], [0]], dtype=torch.long), node_ids=torch.arange(5))
    aligned = g.procrustes_project(from_U, to_U)
    
    norm_from = torch.linalg.norm(from_U, ord="fro")
    norm_aligned = torch.linalg.norm(aligned, ord="fro")
    assert torch.isclose(norm_from, norm_aligned, rtol=1e-5)
    
    I_approx = aligned.t() @ aligned
    I = torch.eye(3)
    assert torch.allclose(I_approx, I, rtol=1e-5, atol=1e-5)

def test_procrustes_legacy_backend_equivalence():
    seed_all(42)
    A = torch.randn(6, 3)
    from_U, _ = torch.linalg.qr(A)
    to_U = torch.randn(6, 3)
    
    M = torch.matmul(from_U.t(), to_U)
    u_legacy, s_legacy, v_legacy = torch.svd(M)
    R_legacy = torch.matmul(u_legacy, v_legacy.t())
    aligned_legacy = torch.matmul(from_U, R_legacy)
    
    g = Graph(x=torch.ones(5, 1), edge_index=torch.tensor([[0], [0]], dtype=torch.long), node_ids=torch.arange(5))
    aligned_new = g.procrustes_project(from_U, to_U)
    
    assert torch.allclose(aligned_new, aligned_legacy, rtol=1e-5, atol=1e-5)

def test_procrustes_linalg_error_fallback(monkeypatch):
    def mock_svd(*args, **kwargs):
        raise torch.linalg.LinAlgError("SVD failed")
    monkeypatch.setattr(torch.linalg, "svd", mock_svd)
    
    from_U = torch.randn(6, 3)
    to_U = torch.randn(6, 3)
    
    g = Graph(x=torch.ones(5, 1), edge_index=torch.tensor([[0], [0]], dtype=torch.long), node_ids=torch.arange(5))
    aligned = g.procrustes_project(from_U, to_U)
    assert torch.equal(aligned, from_U)

def test_make_scheduler_steps(global_config_restore):
    from src.train.federated_orchestrator import _make_scheduler, _make_optimizer
    from src.GNN.dynamic_classifier import DynamicClassifier
    import torch.optim as optim
    
    global_config_restore["dataset"]["task"] = "link_pred"
    global_config_restore["dataset"]["edge_dim"] = 1
    global_config_restore["dataset"]["node_encoder"] = False
    global_config_restore["dataset"]["edge_encoder"] = False
    global_config_restore["model"]["data_type"] = "feature"
    global_config_restore["model"]["edge_decoding"] = "concat"
    global_config_restore["gnn"]["dims"] = [16]
    global_config_restore["gnn"]["dims_pre_mp"] = []
    global_config_restore["gnn"]["dims_post_mp"] = []
    global_config_restore["gnn"]["embed_update_method"] = "gru"
    
    global_config_restore["optim"]["optimizer"] = "adam"
    global_config_restore["optim"]["base_lr"] = 0.005
    global_config_restore["optim"]["weight_decay"] = 0.01
    global_config_restore["optim"]["scheduler"] = "steps"
    global_config_restore["optim"]["steps"] = [5, 10]
    global_config_restore["optim"]["lr_decay"] = 0.1
    global_config_restore["train"]["num_epochs"] = 15
    
    g = Graph(x=torch.ones(5, 10),
              edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
              node_ids=torch.arange(5))
    g.edge_label_index = torch.tensor([[0, 1]], dtype=torch.long)
    g.edge_label = torch.tensor([1.0])
    
    dyn = DynamicClassifier(g)
    opt = _make_optimizer(dyn)
    sched = _make_scheduler(opt)
    
    assert isinstance(sched, optim.lr_scheduler.MultiStepLR)
    assert list(sched.milestones) == [5, 10]
    assert sched.gamma == 0.1

def test_cos_actually_anneals(global_config_restore, monkeypatch):
    import src.dynamic_client
    from src.train.federated_orchestrator import _make_scheduler as orig_make_scheduler
    
    scheduler_instance = None
    step_calls = 0
    
    def mock_make_scheduler(optimizer):
        nonlocal scheduler_instance
        scheduler_instance = orig_make_scheduler(optimizer)
        if scheduler_instance is not None:
            orig_step = scheduler_instance.step
            def spied_step(*args, **kwargs):
                nonlocal step_calls
                step_calls += 1
                return orig_step(*args, **kwargs)
            scheduler_instance.step = spied_step
        return scheduler_instance
        
    monkeypatch.setattr(src.dynamic_client, "_make_scheduler", mock_make_scheduler)
    
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
    
    global_config_restore["optim"]["optimizer"] = "adam"
    global_config_restore["optim"]["base_lr"] = 0.1
    global_config_restore["optim"]["scheduler"] = "cos"
    global_config_restore["train"]["num_epochs"] = 10
    global_config_restore["train"]["internal_validation_tolerance"] = 100
    global_config_restore["dataset"]["split"] = [0.8, 0.1, 0.1]
    
    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    
    cl = DynamicClient(client_snaps[0], id=0)
    cl.initialize()
    _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42)
    
    loss_fn = losses[global_config_restore["model"]["loss_fun"]]
    cl.local_finetune(t=0, local_epochs=3, loss_fn=loss_fn)
    
    assert scheduler_instance is not None
    assert step_calls == 3
    opt_lr = scheduler_instance.optimizer.param_groups[0]["lr"]
    assert opt_lr < 0.1

def test_scheduler_none_neutral(global_config_restore, monkeypatch):
    import src.dynamic_client
    
    make_scheduler_called = False
    def mock_make_scheduler(optimizer):
        nonlocal make_scheduler_called
        make_scheduler_called = True
        return None
        
    monkeypatch.setattr(src.dynamic_client, "_make_scheduler", mock_make_scheduler)
    
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
    
    global_config_restore["optim"]["optimizer"] = "adam"
    global_config_restore["optim"]["base_lr"] = 0.005
    global_config_restore["optim"]["scheduler"] = "none"
    global_config_restore["train"]["internal_validation_tolerance"] = 100
    global_config_restore["dataset"]["split"] = [0.8, 0.1, 0.1]
    
    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    
    cl = DynamicClient(client_snaps[0], id=0)
    cl.initialize()
    _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42)
    
    loss_fn = losses[global_config_restore["model"]["loss_fun"]]
    cl.local_finetune(t=0, local_epochs=3, loss_fn=loss_fn)
    
    assert make_scheduler_called

def test_meta_disabled_identical(global_config_restore):
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
    client_snaps = partition_snapshots(global_snaps, 1)
    server_default = make_server(global_snaps, client_snaps)
    res_default = server_default.joint_train_w(FL=True, epochs=2)

    global_config_restore["meta"]["is_meta"] = False
    seed_all(42)
    global_snaps_explicit = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps_explicit = partition_snapshots(global_snaps_explicit, 1)
    server_explicit = make_server(global_snaps_explicit, client_snaps_explicit)
    res_explicit = server_explicit.joint_train_w(FL=True, epochs=2)
    
    assert res_default["mrr_history"] == res_explicit["mrr_history"]

def test_meta_enabled_changes_trajectory(global_config_restore):
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

    global_config_restore["meta"]["is_meta"] = False
    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=5, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    server_false = make_server(global_snaps, client_snaps)
    res_false = server_false.joint_train_w(FL=True, epochs=2)

    global_config_restore["meta"]["is_meta"] = True
    global_config_restore["meta"]["alpha"] = 0.5
    global_config_restore["meta"]["method"] = "moving_average"
    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=5, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    server_true = make_server(global_snaps, client_snaps)
    res_true = server_true.joint_train_w(FL=True, epochs=2)
    
    assert res_false["mean_mrr"] != res_true["mean_mrr"]
    assert 0.0 <= res_false["mean_mrr"] <= 1.0
    assert 0.0 <= res_true["mean_mrr"] <= 1.0

def test_meta_methods_completion(global_config_restore):
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

    global_config_restore["meta"]["is_meta"] = True
    
    global_config_restore["meta"]["method"] = "moving_average"
    global_config_restore["meta"]["alpha"] = 0.5
    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=3, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    server_ma = make_server(global_snaps, client_snaps)
    res_ma = server_ma.joint_train_w(FL=True, epochs=2)
    assert len(res_ma["mrr_history"]) >= 2

    global_config_restore["meta"]["method"] = "online_mean"
    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=3, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    server_om = make_server(global_snaps, client_snaps)
    res_om = server_om.joint_train_w(FL=True, epochs=2)
    assert len(res_om["mrr_history"]) >= 2

def test_meta_fl_false_guard(global_config_restore):
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

    global_config_restore["meta"]["is_meta"] = True
    global_config_restore["meta"]["method"] = "moving_average"
    global_config_restore["meta"]["alpha"] = 0.5
    
    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=3, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    server = make_server(global_snaps, client_snaps)
    res = server.joint_train_w(FL=False, epochs=2)
    assert len(res["mrr_history"]) >= 2

def test_meta_blend_bn_safe(global_config_restore):
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
    global_config_restore["gnn"]["batchnorm"] = True
    
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

    global_config_restore["meta"]["is_meta"] = True
    global_config_restore["meta"]["method"] = "moving_average"
    global_config_restore["meta"]["alpha"] = 0.5
    
    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=3, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    server = make_server(global_snaps, client_snaps)
    
    res = server.joint_train_w(FL=True, epochs=2)
    assert len(res["mrr_history"]) >= 2
