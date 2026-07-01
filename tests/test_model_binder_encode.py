import pytest
import torch
from src.models.model_binders import ModelSpecs, ModelBinder
from torch_geometric.data import Data

def test_recurrent_layer_specs_encode(config):
    dim_in = 8
    dim_inner = 16
    block_kwargs = {
        "layer_type": "gcnconv",
        "updater_name": "gru",
        "batchnorm": False,
        "dropout": 0.0,
        "act": "relu",
        "skip_connection": "none",
        "layer_kwargs": {},
        "updater_kwargs": {},
    }
    specs = [
        ModelSpecs(type="recurrent_layer", layer_sizes=[dim_in, dim_inner], block_kwargs=block_kwargs),
        ModelSpecs(type="recurrent_layer", layer_sizes=[dim_inner, dim_inner], block_kwargs=block_kwargs)
    ]
    binder = ModelBinder(specs)
    
    N = 10
    x = torch.randn(N, dim_in)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    z, new_hs = binder.encode(x, edge_index, hs=None)
    assert z.shape == (N, dim_inner)
    assert len(new_hs) == 2
    for h in new_hs:
        assert h.shape == (N, dim_inner)

def test_hs_threading(config):
    dim_in = 8
    dim_inner = 16
    block_kwargs = {
        "layer_type": "gcnconv",
        "updater_name": "gru",
        "batchnorm": False,
        "dropout": 0.0,
        "act": "relu",
        "skip_connection": "none",
        "layer_kwargs": {},
        "updater_kwargs": {},
    }
    specs = [
        ModelSpecs(type="recurrent_layer", layer_sizes=[dim_in, dim_inner], block_kwargs=block_kwargs),
        ModelSpecs(type="recurrent_layer", layer_sizes=[dim_inner, dim_inner], block_kwargs=block_kwargs)
    ]
    binder = ModelBinder(specs)
    
    N = 10
    x = torch.randn(N, dim_in)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    z1, new_hs1 = binder.encode(x, edge_index, hs=None)
    z2, new_hs2 = binder.encode(x, edge_index, hs=new_hs1)
    
    assert z2.shape == (N, dim_inner)
    assert len(new_hs2) == 2
    for h in new_hs2:
        assert h.shape == (N, dim_inner)
    assert not torch.allclose(z1, z2)

def test_roland_mlp_dim_mapping(config):
    block_kwargs = {
        "batchnorm": False,
        "dropout": 0.0,
        "act": "relu",
        "final_act": False,
    }
    spec = ModelSpecs(type="roland_mlp", layer_sizes=[4, 8, 16], block_kwargs=block_kwargs)
    binder = ModelBinder([spec])
    
    N = 10
    x = torch.randn(N, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    z, new_hs = binder.encode(x, edge_index, hs=None)
    assert z.shape == (N, 16)
    assert len(new_hs) == 0

def test_encoders(config):
    node_spec = ModelSpecs(type="node_encoder", layer_sizes=[5, 10], block_kwargs={"batchnorm": False})
    binder_node = ModelBinder([node_spec])
    x = torch.randn(8, 5)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    z, new_hs = binder_node.encode(x, edge_index, hs=None)
    assert z.shape == (8, 10)
    assert len(new_hs) == 0
    
    edge_spec = ModelSpecs(type="edge_encoder", layer_sizes=[3, 6], block_kwargs={"batchnorm": False})
    recurrent_block_kwargs = {
        "layer_type": "residual_edge_conv",
        "updater_name": "gru",
        "batchnorm": False,
        "dropout": 0.0,
        "act": "relu",
        "skip_connection": "none",
        "layer_kwargs": {"edge_dim": 6},
        "updater_kwargs": {},
    }
    rec_spec = ModelSpecs(type="recurrent_layer", layer_sizes=[10, 10], block_kwargs=recurrent_block_kwargs)
    
    binder = ModelBinder([edge_spec, rec_spec])
    x = torch.randn(8, 10)
    edge_attr = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    z, new_hs = binder.encode(x, edge_index, hs=None, edge_attr=edge_attr)
    assert z.shape == (8, 10)
    assert len(new_hs) == 1

def test_encode_no_recurrent(config):
    node_spec = ModelSpecs(type="node_encoder", layer_sizes=[5, 10], block_kwargs={"batchnorm": False})
    binder = ModelBinder([node_spec])
    x = torch.randn(8, 5)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    z, new_hs = binder.encode(x, edge_index, hs=None)
    assert z.shape == (8, 10)
    assert new_hs == []

def test_forward_stateless(config):
    mlp_spec = ModelSpecs(type="MLP", layer_sizes=[8, 4], final_activation_function="linear")
    binder = ModelBinder([mlp_spec])
    x = torch.randn(5, 8)
    out = binder(x)
    assert out.shape == (5, 4)
    
    gnn_spec = ModelSpecs(type="GNN", layer_sizes=[4, 2], heads=[1], gnn_layer_type="sage")
    binder_gnn = ModelBinder([gnn_spec])
    x_gnn = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out_gnn = binder_gnn(x_gnn, edge_index=edge_index)
    assert out_gnn.shape == (5, 2)

@pytest.mark.parametrize("embed_update_method", ["gru", "moving_average"])
def test_parity_with_recurrent_gnn(config, embed_update_method):
    from src.models.recurrent import RecurrentGNN
    from src.utils.graph import Graph
    
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["layer_type"] = "residual_edge_conv"
    config["gnn"]["msg_direction"] = "single"
    config["gnn"]["normalize_adj"] = False
    config["gnn"]["agg"] = "add"
    config["gnn"]["skip_connection"] = "none"
    config["gnn"]["embed_update_method"] = embed_update_method
    config["gnn"]["l2norm"] = False
    config["meta"]["alpha"] = 0.9
    
    N = 10
    W = 1
    x = torch.ones(N, 1)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, W)
    
    graph = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr, node_ids=torch.arange(N))
    graph.keep_ratio = torch.rand(N, 1)
    graph.node_degree_new = torch.tensor([1, 1, 0, 1, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    active_mask = graph.node_degree_new > 0
    
    oracle = RecurrentGNN(config, dim_in=1, dim_out=16).eval()
    
    effective_edge_dim = 1
    layer_kwargs = {
        "edge_dim": effective_edge_dim,
        "msg_direction": config["gnn"]["msg_direction"],
        "normalize": config["gnn"]["normalize_adj"],
        "agg": config["gnn"]["agg"],
    }
    updater_kwargs = {}
    if embed_update_method == "moving_average":
        updater_kwargs["alpha"] = config["meta"]["alpha"]
        
    rec = {
        "layer_type": config["gnn"]["layer_type"],
        "updater_name": embed_update_method,
        "batchnorm": config["gnn"]["batchnorm"],
        "dropout": config["gnn"]["dropout"],
        "act": config["gnn"]["act"],
        "skip_connection": config["gnn"]["skip_connection"],
        "layer_kwargs": layer_kwargs,
        "updater_kwargs": updater_kwargs,
    }
    
    specs = []
    prev = 1
    for w in config["gnn"]["dims"]:
        specs.append(ModelSpecs(
            type="recurrent_layer",
            layer_sizes=[prev, w],
            block_kwargs=dict(rec),
        ))
        prev = w
        
    binder = ModelBinder(specs).eval()
    for i in range(len(binder.models)):
        binder.models[i].load_state_dict(oracle.mp_layers[i].state_dict())
        
    z_oracle1, hs_oracle1 = oracle.encode(graph, None)
    z_binder1, hs_binder1 = binder.encode(
        graph.x,
        graph.edge_index,
        hs=None,
        edge_attr=graph.edge_attr,
        keep_ratio=graph.keep_ratio,
        active_mask=active_mask,
    )
    
    assert torch.allclose(z_oracle1, z_binder1, atol=1e-6)
    assert len(hs_oracle1) == len(hs_binder1)
    for h1, h2 in zip(hs_oracle1, hs_binder1):
        assert torch.allclose(h1, h2, atol=1e-6)
        
    graph2 = Graph(x=x, edge_index=torch.tensor([[0, 2, 1, 3], [2, 1, 3, 0]], dtype=torch.long),
                   edge_attr=torch.randn(4, W), node_ids=torch.arange(N))
    graph2.keep_ratio = torch.rand(N, 1)
    graph2.node_degree_new = torch.tensor([1, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=torch.long)
    active_mask2 = graph2.node_degree_new > 0
    
    z_oracle2, hs_oracle2 = oracle.encode(graph2, hs_oracle1)
    z_binder2, hs_binder2 = binder.encode(
        graph2.x,
        graph2.edge_index,
        hs=hs_binder1,
        edge_attr=graph2.edge_attr,
        keep_ratio=graph2.keep_ratio,
        active_mask=active_mask2,
    )
    
    assert torch.allclose(z_oracle2, z_binder2, atol=1e-6)
    assert len(hs_oracle2) == len(hs_binder2)
    for h1, h2 in zip(hs_oracle2, hs_binder2):
        assert torch.allclose(h1, h2, atol=1e-6)
