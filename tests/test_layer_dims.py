import torch
import pytest
from configs.assertions import assert_cfg
from src.models.recurrent import RecurrentGNN
from src.utils.graph import Graph

def test_assert_cfg_dims(config):
    config["gnn"]["dims"] = []
    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "gnn.dims must list at least one MP layer width" in str(exc.value)
    
    config["gnn"]["dims"] = [64]
    assert_cfg(config)

def test_non_uniform_mp_widths(config):
    config["gnn"]["dims"] = [24, 16, 8]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    
    model = RecurrentGNN(config, dim_in=10, dim_out=1)
    
    assert len(model.mp_layers) == 3
    assert model.mp_layers[0].block.layer.in_channels == 10
    assert model.mp_layers[0].block.layer.out_channels == 24
    assert model.mp_layers[1].block.layer.in_channels == 24
    assert model.mp_layers[1].block.layer.out_channels == 16
    assert model.mp_layers[2].block.layer.in_channels == 16
    assert model.mp_layers[2].block.layer.out_channels == 8
    
    N = 12
    graph = Graph(x=torch.ones(N, 10),
                  edge_index=torch.tensor([[0, 1], [1, 0]]),
                  edge_attr=torch.randn(2, 1),
                  node_ids=torch.arange(N))
    z, hs = model.encode(graph, None)
    
    assert z.shape == (N, 8)
    assert len(hs) == 3
    assert hs[0].shape == (N, 24)
    assert hs[1].shape == (N, 16)
    assert hs[2].shape == (N, 8)

def test_multi_element_dims_pre_mp(config):
    config["gnn"]["dims"] = [64]
    config["gnn"]["dims_pre_mp"] = [32, 16]
    config["dataset"]["node_encoder"] = False
    
    model = RecurrentGNN(config, dim_in=10, dim_out=1)
    
    assert model.pre_mp is not None
    assert len(model.pre_mp.net) == 2
    assert model.mp_layers[0].block.layer.in_channels == 16

def test_multi_element_dims_post_mp(config):
    config["gnn"]["dims"] = [64]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = [24]
    config["dataset"]["edge_dim"] = 0
    config["dataset"]["node_encoder"] = False
    config["model"]["edge_decoding"] = "concat"
    
    model = RecurrentGNN(config, dim_in=10, dim_out=1)
    
    assert model.head is not None
    assert hasattr(model.head, "post_mp")
    assert len(model.head.post_mp.net) == 2
    assert model.head.post_mp.net[0].lin.out_features == 24
    assert model.head.post_mp.net[1].lin.out_features == 1
