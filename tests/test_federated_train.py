import copy
import torch
import pytest
import src
from src.utils.graph import Graph
from src.train.federated_orchestrator import (
    _clone_state,
    _stitch_global_z,
    _make_optimizer,
    _make_scheduler,
)


@pytest.fixture
def global_config_restore():
    original_registry = copy.deepcopy(src.config._registry)
    try:
        yield src.config
    finally:
        src.config._registry.clear()
        src.config._registry.update(original_registry)


def test_clone_state():
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([3.0, 4.0])
    sd = {
        "model": {"m0": t1},
        "head": {"h": t2}
    }

    clone = _clone_state(sd)
    assert set(clone.keys()) == {"model", "head"}
    assert set(clone["model"].keys()) == {"m0"}
    assert set(clone["head"].keys()) == {"h"}
    assert torch.equal(clone["model"]["m0"], t1)
    assert torch.equal(clone["head"]["h"], t2)

    clone["model"]["m0"][0] = 999.0
    assert sd["model"]["m0"][0] == 1.0


def test_stitch_global_z():
    N = 5
    dim = 4
    z = torch.randn(N, dim)
    node_ids = torch.arange(N)

    stitched = _stitch_global_z([z], [node_ids], N, dim, "cpu")
    assert torch.equal(stitched, z)

    z0 = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    ids0 = torch.tensor([1, 3])

    z1 = torch.tensor([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
    ids1 = torch.tensor([0, 2, 4])

    stitched_two = _stitch_global_z([z0, z1], [ids0, ids1], N, 2, "cpu")
    expected = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0]
    ])
    assert torch.equal(stitched_two, expected)


def test_make_optimizer_scheduler(global_config_restore):
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
    global_config_restore["train"]["num_epochs"] = 10

    g = Graph(x=torch.ones(5, 10),
              edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
              edge_attr=torch.randn(2, 1),
              node_ids=torch.arange(5))
    g.edge_label_index = torch.tensor([[0, 1]], dtype=torch.long)
    g.edge_label = torch.tensor([1.0])

    dyn = DynamicClassifier(g)

    optimizer = _make_optimizer(dyn)
    assert isinstance(optimizer, optim.Adam)
    assert optimizer.defaults["lr"] == 0.005
    assert optimizer.defaults["weight_decay"] == 0.01

    global_config_restore["optim"]["scheduler"] = "none"
    sched_none = _make_scheduler(optimizer)
    assert sched_none is None

    global_config_restore["optim"]["scheduler"] = "cos"
    sched_cos = _make_scheduler(optimizer)
    assert isinstance(sched_cos, optim.lr_scheduler.CosineAnnealingLR)
    assert sched_cos.T_max == 10
