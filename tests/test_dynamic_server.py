import math
import random
import copy
import numpy as np
import torch
import pytest
from torch_geometric.data import Data
import src
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer
from src.dynamic_client import DynamicClient

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

def test_dynamic_server_live_update(global_config_restore):
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
    
    global_config_restore["model"]["iterations"] = 3
    global_config_restore["model"]["local_epochs"] = 2
    global_config_restore["train"]["internal_validation_tolerance"] = 5
    global_config_restore["metric"]["mrr_method"] = "min"
    global_config_restore["experimental"]["rank_eval_multiplier"] = 50
    global_config_restore["dataset"]["split"] = [0.8, 0.1, 0.1]
    
    global_config_restore["optim"]["optimizer"] = "adam"
    global_config_restore["optim"]["base_lr"] = 0.005
    global_config_restore["optim"]["scheduler"] = "none"
    global_config_restore["seed"] = 42
    
    # 1. C=1: mrr_history nonempty, deterministic
    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 1)
    
    seed_all(100)
    server1 = make_server(global_snaps, client_snaps)
    res1 = server1.joint_train_w()
    
    assert "mean_mrr" in res1
    assert "std_mrr" in res1
    assert "mrr_history" in res1
    assert len(res1["mrr_history"]) > 0
    for m in res1["mrr_history"]:
        assert 0.0 <= m <= 1.0
        
    seed_all(42)
    global_snaps2 = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps2 = partition_snapshots(global_snaps2, 1)
    
    seed_all(100)
    server2 = make_server(global_snaps2, client_snaps2)
    res2 = server2.joint_train_w()
    assert res1["mrr_history"] == res2["mrr_history"]
    
    # 2. C=2: runs, sensible mrr
    seed_all(42)
    global_snaps_c2 = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps_c2 = partition_snapshots(global_snaps_c2, 2)
    
    seed_all(100)
    server_c2 = make_server(global_snaps_c2, client_snaps_c2)
    res_c2 = server_c2.joint_train_w()
    assert len(res_c2["mrr_history"]) > 0
    
    # 3. C=4 over N=40 sparse snapshots (abstention check)
    seed_all(42)
    global_snaps_c4 = make_toy_snapshots(N=40, W=1, num_snaps=4, seed=42)
    client_snaps_c4 = partition_snapshots(global_snaps_c4, 4)
    
    seed_all(100)
    server_c4 = make_server(global_snaps_c4, client_snaps_c4)
    res_c4 = server_c4.joint_train_w()
    assert "mean_mrr" in res_c4
    
    # 4. single-snapshot raises ValueError
    single_snaps = make_toy_snapshots(N=8, W=1, num_snaps=1, seed=42)
    single_client_snaps = partition_snapshots(single_snaps, 1)
    
    with pytest.raises(ValueError) as exc:
        server_err = make_server(single_snaps, single_client_snaps)
        server_err.joint_train_w()
    assert "needs >= 2 snapshots" in str(exc.value)

def test_dynamic_server_coef(global_config_restore):
    global_snaps = make_toy_snapshots(N=10, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    server = make_server(global_snaps, client_snaps)
    coef = server._coef(server.clients)
    
    assert len(coef) == 2
    assert math.isclose(sum(coef), 1.0)
    assert coef[0] == server.clients[0].num_nodes() / (server.clients[0].num_nodes() + server.clients[1].num_nodes())

def test_dynamic_client_can_train(global_config_restore):
    from src.train.federated_orchestrator import _partition_edges_per_snapshot
    global_config_restore["dataset"]["split"] = [0.8, 0.1, 0.1]
    
    g = torch.Generator().manual_seed(42)
    x = torch.ones(5, 1)
    s0 = Graph(x=x, edge_index=torch.tensor([[0], [1]]), edge_attr=torch.randn(1, 1), node_ids=torch.arange(5))
    s1 = Graph(x=x, edge_index=torch.tensor([[0], [1]]), edge_attr=torch.randn(1, 1), node_ids=torch.arange(5))
    
    cl_sparse = DynamicClient([s0, s1], id=0)
    _partition_edges_per_snapshot(cl_sparse.snaps, [0.8, 0.1, 0.1], seed=42)
    assert not cl_sparse.can_train(0)
    
    # dense today AND tomorrow: >=2 nodes and >=2 edges at snap_t (edge-encoder BN needs
    # >1 edge) plus train/val positives at snap_{t+1} -> can_train.
    s0_dense = Graph(x=x, edge_index=torch.randint(0, 5, (2, 20), generator=g), edge_attr=torch.randn(20, 1), node_ids=torch.arange(5))
    s1_dense = Graph(x=x, edge_index=torch.randint(0, 5, (2, 20), generator=g), edge_attr=torch.randn(20, 1), node_ids=torch.arange(5))

    cl_dense = DynamicClient([s0_dense, s1_dense], id=0)
    _partition_edges_per_snapshot(cl_dense.snaps, [0.8, 0.1, 0.1], seed=42)
    assert cl_dense.can_train(0)

def test_dynamic_client_encode(global_config_restore):
    global_config_restore["dataset"]["task"] = "link_pred"
    global_config_restore["model"]["data_type"] = "feature"
    global_config_restore["gnn"]["dims"] = [16]
    global_config_restore["gnn"]["dims_pre_mp"] = []
    global_config_restore["gnn"]["dims_post_mp"] = []
    global_config_restore["gnn"]["embed_update_method"] = "gru"
    
    snaps = make_toy_snapshots(N=8, W=1, num_snaps=2, seed=42)
    client_snaps = partition_snapshots(snaps, 2)
    cl = DynamicClient(client_snaps[0], id=0)
    cl.initialize()

    z, nid = cl.encode(0)
    assert z.shape == (cl.num_nodes(), 16)
    assert torch.equal(nid, cl.snaps[0].node_ids)

def test_dynamic_client_refresh(global_config_restore):
    global_config_restore["dataset"]["task"] = "link_pred"
    global_config_restore["model"]["data_type"] = "feature"
    global_config_restore["gnn"]["dims"] = [16, 16]
    global_config_restore["gnn"]["dims_pre_mp"] = []
    global_config_restore["gnn"]["dims_post_mp"] = []
    global_config_restore["gnn"]["embed_update_method"] = "gru"
    
    snaps = make_toy_snapshots(N=8, W=1, num_snaps=2, seed=42)
    client_snaps = partition_snapshots(snaps, 2)
    cl = DynamicClient(client_snaps[0], id=0)
    cl.initialize()

    assert cl.hs is None
    cl.refresh(0)
    assert isinstance(cl.hs, list)
    assert len(cl.hs) == 2
    for h in cl.hs:
        assert h.shape == (cl.num_nodes(), 16)

def test_dynamic_client_state_dict_roundtrip(global_config_restore):
    global_config_restore["dataset"]["task"] = "link_pred"
    global_config_restore["model"]["data_type"] = "feature"
    global_config_restore["gnn"]["dims"] = [16]
    global_config_restore["gnn"]["dims_pre_mp"] = []
    global_config_restore["gnn"]["dims_post_mp"] = []
    global_config_restore["gnn"]["embed_update_method"] = "gru"
    
    snaps = make_toy_snapshots(N=8, W=1, num_snaps=2, seed=42)
    client_snaps = partition_snapshots(snaps, 2)
    
    cl1 = DynamicClient(client_snaps[0], id=0)
    cl1.initialize()
    cl2 = DynamicClient(client_snaps[0], id=1)
    cl2.initialize()
    
    z1, _ = cl1.classifier.encode(cl1.snaps[0], None)
    z2, _ = cl2.classifier.encode(cl2.snaps[0], None)
    assert not torch.allclose(z1, z2, atol=1e-6)
    
    cl2.load_state_dict(cl1.state_dict())
    
    z2_new, _ = cl2.classifier.encode(cl2.snaps[0], None)
    assert torch.allclose(z1, z2_new, atol=1e-6)
