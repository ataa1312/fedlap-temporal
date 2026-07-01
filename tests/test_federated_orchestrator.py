import torch
from src.utils.graph import Graph
from src.GNN.dynamic_classifier import DynamicClassifier

def make_graph(N=12, W=1, E=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    edges = set()
    while len(edges) < 30:
        u = torch.randint(0, N, (1,), generator=g).item()
        v = torch.randint(0, N, (1,), generator=g).item()
        if u != v:
            edges.add((u, v))
    edge_index = torch.tensor(list(edges), dtype=torch.long).t()

    graph = Graph(x=torch.ones(N, 1),
                  edge_index=edge_index,
                  edge_attr=torch.randn(30, W, generator=g),
                  node_ids=torch.arange(N))
    graph.edge_label_index = torch.randint(0, N, (2, E), generator=g)
    graph.edge_label = torch.randint(0, 2, (E,), generator=g).float()
    return graph

def test_partition_edges_per_snapshot(config):
    from src.train.federated_orchestrator import _partition_edges_per_snapshot
    s0 = make_graph(seed=12)
    s1 = make_graph(seed=34)
    
    _partition_edges_per_snapshot([s0, s1], [0.8, 0.1, 0.1], seed=1234)
    
    for s in [s0, s1]:
        assert hasattr(s, "pos_train")
        assert hasattr(s, "pos_val")
        assert hasattr(s, "pos_test")
        
        n_total = s.edge_index.size(1)
        n_train = s.pos_train.size(1)
        n_val = s.pos_val.size(1)
        n_test = s.pos_test.size(1)
        assert n_train + n_val + n_test == n_total
        
        set_train = set(zip(s.pos_train[0].tolist(), s.pos_train[1].tolist()))
        set_val = set(zip(s.pos_val[0].tolist(), s.pos_val[1].tolist()))
        set_test = set(zip(s.pos_test[0].tolist(), s.pos_test[1].tolist()))
        
        assert not set_train.intersection(set_val)
        assert not set_train.intersection(set_test)
        assert not set_val.intersection(set_test)
        
        set_total = set(zip(s.edge_index[0].tolist(), s.edge_index[1].tolist()))
        assert set_train.union(set_val).union(set_test) == set_total

    s0_dup = make_graph(seed=12)
    s1_dup = make_graph(seed=34)
    _partition_edges_per_snapshot([s0_dup, s1_dup], [0.8, 0.1, 0.1], seed=1234)
    assert torch.equal(s0.pos_train, s0_dup.pos_train)
    
    s0_diff = make_graph(seed=12)
    s1_diff = make_graph(seed=34)
    _partition_edges_per_snapshot([s0_diff, s1_diff], [0.8, 0.1, 0.1], seed=5678)
    assert not torch.equal(s0.pos_train, s0_diff.pos_train)

def test_attach_future_link_pred_labels():
    from src.train.federated_orchestrator import _attach_future_link_pred_labels
    s0 = make_graph(seed=12)
    s1 = make_graph(seed=34)
    
    pos = s1.edge_index[:, :5]
    n_pos = pos.size(1)
    
    snap = _attach_future_link_pred_labels(s0, s1, pos)
    
    assert snap is not s0
    assert snap.edge_label.shape == (n_pos * 2,)
    assert torch.equal(snap.edge_label[:n_pos], torch.ones(n_pos))
    assert torch.equal(snap.edge_label[n_pos:], torch.zeros(n_pos))
    assert snap.edge_label_index.shape == (2, n_pos * 2)
    assert torch.equal(snap.edge_label_index[:, :n_pos], pos)

def test_average_state_dict():
    from src.train.federated_orchestrator import _average_state_dict
    
    d_old = {
        "weight": torch.tensor([1.0, 2.0]),
        "tracked": torch.tensor([1, 2], dtype=torch.long)
    }
    d_new = {
        "weight": torch.tensor([3.0, 4.0]),
        "tracked": torch.tensor([3, 4], dtype=torch.long)
    }
    
    res_1 = _average_state_dict(d_old, d_new, 1.0)
    assert torch.equal(res_1["weight"], d_new["weight"])
    assert res_1["tracked"].dtype == torch.long
    assert torch.equal(res_1["tracked"], d_new["tracked"])
    
    res_0 = _average_state_dict(d_old, d_new, 0.0)
    assert torch.equal(res_0["weight"], d_old["weight"])
    assert res_0["tracked"].dtype == torch.long
    assert torch.equal(res_0["tracked"], d_new["tracked"])
    
    res_half = _average_state_dict(d_old, d_new, 0.5)
    assert torch.equal(res_half["weight"], torch.tensor([2.0, 3.0]))
    assert res_half["tracked"].dtype == torch.long
    assert torch.equal(res_half["tracked"], d_new["tracked"])

def test_orchestrator_step_helpers(config):
    from src.train.federated_orchestrator import (
        _partition_edges_per_snapshot,
        _step_eval_with_mrr_pair,
        _step_train_pair,
        _refresh_hs
    )
    import torch.nn.functional as F
    
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
    config["gnn"]["l2norm"] = False
    
    s0 = make_graph(seed=12)
    s1 = make_graph(seed=34)
    _partition_edges_per_snapshot([s0, s1], [0.8, 0.1, 0.1], seed=1234)
    
    dyn = DynamicClassifier(s0)
    loss_fn = F.binary_cross_entropy_with_logits
    
    loss, mrr, metrics = _step_eval_with_mrr_pair(
        dyn, s0, s1, None, loss_fn, "cpu", True, 100, "min"
    )
    assert isinstance(loss, float)
    assert isinstance(mrr, float)
    assert isinstance(metrics, dict)
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc", "ap"}
    assert expected_keys.issubset(metrics.keys())
    
    opt = torch.optim.Adam(dyn.parameters(), lr=1e-3)
    init_params = [p.clone() for p in dyn.parameters() if p.requires_grad]
    
    train_loss, new_hs = _step_train_pair(
        dyn, s0, s1, None, loss_fn, opt, "cpu", True
    )
    assert isinstance(train_loss, float)
    assert isinstance(new_hs, list)
    assert len(new_hs) == 2
    
    changed = False
    for p_init, p_new in zip(init_params, [p for p in dyn.parameters() if p.requires_grad]):
        if not torch.allclose(p_init, p_new):
            changed = True
            break
    assert changed
    
    refreshed_hs = _refresh_hs(dyn, s0, None, "cpu", True)
    assert isinstance(refreshed_hs, list)
    assert len(refreshed_hs) == 2
    for h in refreshed_hs:
        assert h.shape == (s0.num_nodes, 16)
