import torch
import pytest
from src.utils.graph import Graph
from src.GNN.dynamic_classifier import DynamicClassifier
from src.models.recurrent import RecurrentGNN

def make_graph(N=12, W=1, E=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    graph = Graph(x=torch.ones(N, 1),
                  edge_index=torch.randint(0, N, (2, 30), generator=g),
                  edge_attr=torch.randn(30, W, generator=g),
                  node_ids=torch.arange(N))
    graph.edge_label_index = torch.randint(0, N, (2, E), generator=g)
    graph.edge_label = torch.randint(0, 2, (E,), generator=g).float()
    return graph

def test_dynamic_classifier_construction(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    
    graph = make_graph(N=12, W=1, E=20)
    dyn = DynamicClassifier(graph)
    
    assert dyn.model is not None
    assert dyn.head is not None
    assert dyn.hs is None
    
    pred, label, new_hs = dyn.forward()
    assert pred.shape == (20,)
    assert label.shape == (20,)
    assert len(new_hs) == 2

def test_parity_with_oracle(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    
    graph = make_graph(N=12, W=1, E=20)
    
    oracle = RecurrentGNN(config, dim_in=1, dim_out=1).eval()
    dyn = DynamicClassifier(graph)
    dyn.eval()
    
    for i in range(len(dyn.model.models)):
        dyn.model.models[i].load_state_dict(oracle.mp_layers[i].state_dict())
    dyn.head.load_state_dict(oracle.head.state_dict())
    
    z_dyn, hs_dyn = dyn.encode()
    z_oracle, hs_oracle = oracle.encode(graph, None)
    
    assert torch.allclose(z_dyn, z_oracle, atol=1e-6)
    
    pred_dyn, label_dyn = dyn.decode(z_dyn)
    pred_oracle, label_oracle = oracle.decode(z_oracle, graph)
    
    assert torch.allclose(pred_dyn, pred_oracle, atol=1e-6)
    assert torch.allclose(label_dyn, label_oracle, atol=1e-6)

def test_hs_threading(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    
    graph = make_graph(N=12, W=1, E=20)
    dyn = DynamicClassifier(graph)
    
    assert dyn.hs is None
    pred, label, new_hs = dyn.forward()
    assert dyn.last_hs is not None
    
    for h in new_hs:
        assert h.requires_grad
        
    dyn.refresh_hs()
    assert dyn.hs is not None
    for h in dyn.hs:
        assert not h.requires_grad
        
    pred2, label2, new_hs2 = dyn.forward()
    assert pred2.shape == (20,)

def test_l2norm(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = True
    
    graph = make_graph(N=12, W=1, E=20)
    dyn = DynamicClassifier(graph)
    dyn.eval()
    
    z, new_hs = dyn.encode()
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    for h in new_hs:
        h_norms = torch.norm(h, p=2, dim=-1)
        assert not torch.allclose(h_norms, torch.ones_like(h_norms), atol=1e-3)

def test_spectral_hook(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "f+s"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["spectral"]["spectral_len"] = 8
    
    graph = make_graph(N=12, W=1, E=20)
    graph.structural_features = torch.randn(12, 8)
    
    dyn = DynamicClassifier(graph)
    assert dyn.use_spectral
    assert dyn.spectral_len == 8
    
    z, new_hs = dyn.encode()
    assert z.shape == (12, 16 + 8)
    
    pred, label, new_hs = dyn.forward()
    assert pred.shape == (20,)

def test_federated_protocol(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    
    graph = make_graph(N=12, W=1, E=20)
    dyn1 = DynamicClassifier(graph)
    
    sd = dyn1.state_dict()
    assert set(sd.keys()) == {"model", "head"}
    
    dyn2 = DynamicClassifier(graph)
    dyn2.load_state_dict(sd)
    
    dyn1.eval()
    dyn2.eval()
    pred1, _, _ = dyn1.forward()
    pred2, _, _ = dyn2.forward()
    assert torch.allclose(pred1, pred2)
    
    dyn1.train()
    pred1, _, _ = dyn1.forward()
    loss = pred1.sum()
    loss.backward()
    
    grads = dyn1.get_grads()
    assert set(grads.keys()) == {"model", "head"}
    assert grads["model"] is not None
    assert grads["head"] is not None
    for g in grads["model"]:
        assert g is not None
    for g in grads["head"]:
        assert g is not None
        
    dyn2.train()
    dyn2.zero_grad()
    dyn2.set_grads(grads)
    
    for p, g in zip(dyn2.head.parameters(), grads["head"]):
        assert torch.allclose(p.grad, g)
        
    all_params = dyn1.parameters()
    head_params = list(dyn1.head.parameters())
    for hp in head_params:
        assert any(hp is ap for ap in all_params)
