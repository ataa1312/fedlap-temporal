import torch
from src.utils.graph import Graph
from src.GNN.dynamic_classifier import DynamicClassifier

def make_graph(N=12, W=1, E=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    graph = Graph(x=torch.ones(N, 1),
                  edge_index=torch.randint(0, N, (2, 30), generator=g),
                  edge_attr=torch.randn(30, W, generator=g),
                  node_ids=torch.arange(N))
    graph.edge_label_index = torch.randint(0, N, (2, E), generator=g)
    graph.edge_label = torch.randint(0, 2, (E,), generator=g).float()
    return graph

def test_dynamic_classifier_d1(config):
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
    dyn.eval()
    
    z_explicit_none, hs_explicit_none = dyn.encode(graph, None)
    z_omitted, hs_omitted = dyn.encode()
    assert torch.allclose(z_explicit_none, z_omitted, atol=1e-6)
    
    _, hs1 = dyn.encode(graph, None)
    dyn.hs = hs1
    
    z_fallback, hs_fallback = dyn.encode(graph)
    z_explicit_hs1, hs_explicit_hs1 = dyn.encode(graph, hs1)
    assert torch.allclose(z_fallback, z_explicit_hs1, atol=1e-6)
    
    z_overridden, hs_overridden = dyn.encode(graph, None)
    assert not torch.allclose(z_overridden, z_fallback, atol=1e-6)
    
    res_call = dyn(graph, None)
    res_forward = dyn.forward(graph, None)
    assert len(res_call) == 3
    assert torch.allclose(res_call[0], res_forward[0], atol=1e-6)
    assert torch.allclose(res_call[1], res_forward[1], atol=1e-6)
    for h1, h2 in zip(res_call[2], res_forward[2]):
        assert torch.allclose(h1, h2, atol=1e-6)
        
    z, _ = dyn.encode()
    pred_decode, label_decode = dyn.decode(z, graph)
    pred_head, label_head = dyn.head(z, graph)
    assert torch.allclose(pred_decode, pred_head, atol=1e-6)
    assert torch.allclose(label_decode, label_head, atol=1e-6)
    
    z_no_arg, hs_no_arg = dyn.encode()
    pred_no_arg, label_no_arg = dyn.decode(z_no_arg)
    pred_fwd_no_arg, label_fwd_no_arg, hs_fwd_no_arg = dyn.forward()
    assert z_no_arg.shape == (12, 16)
    assert pred_no_arg.shape == (20,)
    assert pred_fwd_no_arg.shape == (20,)
