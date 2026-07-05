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
from src.GNN.dynamic_classifier import DynamicClassifier

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

def test_client_initialize_dispatch(global_config_restore):
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

    snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    cl = DynamicClient(snaps, id=0)
    assert getattr(cl, "classifier", None) is None
    
    cl.initialize()
    assert isinstance(cl.classifier, DynamicClassifier)
    assert cl.classifier.graph is cl.snaps[0]

    cl2 = DynamicClient(snaps, id=1)
    with pytest.raises(ValueError) as exc:
        cl2.initialize(data_type="f+s")
    assert "server-shared SFV" in str(exc.value)

    cl3 = DynamicClient(snaps, id=2)
    with pytest.raises(NotImplementedError) as exc:
        cl3.initialize(data_type="structure")
    assert "smodel-only subclass" in str(exc.value)

def test_server_initialize_and_share(global_config_restore):
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

    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    server = DynamicServer(global_snaps)
    server.add_client(client_snaps[0])
    server.add_client(client_snaps[1])
    
    assert server.num_clients == 2
    assert server.clients[0].id == 0
    assert server.clients[1].id == 1
    
    share = server.initialize()
    assert share == {}
    assert isinstance(server.classifier, DynamicClassifier)
    
    server.initialize_FL()
    
    for cl in server.clients:
        assert isinstance(cl.classifier, DynamicClassifier)
        sd_cl = cl.state_dict()
        sd_srv = server.state_dict()
        assert set(sd_cl.keys()) == {"model", "head"}
        assert set(sd_srv.keys()) == {"model", "head"}
        
        def assert_dict_equal(d1, d2):
            for k, v in d1.items():
                if isinstance(v, dict):
                    assert_dict_equal(v, d2[k])
                else:
                    assert torch.equal(v, d2[k])
        
        assert_dict_equal(sd_cl, sd_srv)

    with pytest.raises(NotImplementedError) as exc:
        server.initialize(data_type="structure")
    assert "smodel-only subclass" in str(exc.value)

def test_joint_train_g_stub_and_train_step_guard(global_config_restore):
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

    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    server = DynamicServer(global_snaps)
    
    with pytest.raises(NotImplementedError) as exc:
        server.joint_train_g(1, 2, a=3)
    assert "gradient averaging is unused" in str(exc.value)

    g = Graph(x=torch.ones(5, 10),
              edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
              edge_attr=torch.randn(2, 1),
              node_ids=torch.arange(5))
    g.edge_label_index = torch.tensor([[0, 1]], dtype=torch.long)
    g.edge_label = torch.tensor([1.0])
    dyn = DynamicClassifier(g)
    
    with pytest.raises(NotImplementedError) as exc:
        dyn.train_step()
    assert "trains through the live-update loop" in str(exc.value)

def assert_state_dicts_differ(sd1, sd2):
    diff = False
    
    def compare(d1, d2):
        nonlocal diff
        for k, v in d1.items():
            if isinstance(v, dict):
                compare(v, d2[k])
            else:
                if not torch.allclose(v, d2[k], atol=1e-5):
                    diff = True
                    return
    
    compare(sd1, sd2)
    assert diff

def test_joint_train_w_local_only(global_config_restore):
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

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    server = DynamicServer(global_snaps)
    server.add_client(client_snaps[0])
    server.add_client(client_snaps[1])
    
    res = server.joint_train_w(FL=False)
    assert len(res["mrr_history"]) > 0
    for val in res["mrr_history"]:
        assert 0.0 <= val <= 1.0
        
    assert_state_dicts_differ(server.clients[0].state_dict(), server.clients[1].state_dict())
    assert_state_dicts_differ(server.clients[0].state_dict(), server.state_dict())
    
    seed_all(42)
    global_snaps_a = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps_a = partition_snapshots(global_snaps_a, 2)
    server_a = DynamicServer(global_snaps_a)
    server_a.add_client(client_snaps_a[0])
    server_a.add_client(client_snaps_a[1])
    seed_all(100)
    res_a = server_a.joint_train_w(FL=False)
    
    seed_all(42)
    global_snaps_b = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps_b = partition_snapshots(global_snaps_b, 2)
    server_b = DynamicServer(global_snaps_b)
    server_b.add_client(client_snaps_b[0])
    server_b.add_client(client_snaps_b[1])
    seed_all(100)
    res_b = server_b.joint_train_w(FL=False)
    
    assert res_a["mrr_history"] == res_b["mrr_history"]

def test_joint_train_w_epochs_override(global_config_restore):
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

    seed_all(42)
    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    server = DynamicServer(global_snaps)
    server.add_client(client_snaps[0])
    server.add_client(client_snaps[1])
    
    res = server.joint_train_w(epochs=1)
    assert len(res["mrr_history"]) > 0
