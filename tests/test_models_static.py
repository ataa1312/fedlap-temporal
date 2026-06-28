import torch
from src.models.model_binders import ModelBinder, ModelSpecs

def test_mlp_binder_forward(config):
    specs = ModelSpecs(type="MLP", layer_sizes=[8, 4])
    binder = ModelBinder([specs])
    x = torch.randn(5, 8)
    out = binder(x)
    assert out.shape == (5, 4)

def test_gnn_binder_forward(config):
    specs = ModelSpecs(type="GNN", layer_sizes=[8, 4], heads=[1], gnn_layer_type="sage")
    binder = ModelBinder([specs])
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out = binder(x, edge_index=edge_index)
    assert out.shape == (4, 4)

def test_reset_parameters(config):
    specs = ModelSpecs(type="MLP", layer_sizes=[8, 4])
    binder = ModelBinder([specs])
    binder.reset_parameters()

def test_state_dict_roundtrip(config):
    specs1 = ModelSpecs(type="MLP", layer_sizes=[8, 4])
    specs2 = ModelSpecs(type="MLP", layer_sizes=[8, 4])
    binder1 = ModelBinder([specs1])
    binder2 = ModelBinder([specs2])
    
    sd = binder1.state_dict()
    binder2.load_state_dict(sd)
    
    for p1, p2 in zip(binder1.parameters(), binder2.parameters()):
        assert torch.allclose(p1, p2)

def test_get_set_grads(config):
    specs = ModelSpecs(type="MLP", layer_sizes=[8, 4])
    binder = ModelBinder([specs])
    
    x = torch.randn(5, 8)
    out = binder(x)
    loss = out.sum()
    loss.backward()
    
    grads = binder.get_grads()
    assert grads is not None
    assert len(grads) > 0
    for g in grads:
        assert g is not None
        assert not torch.isnan(g).any()
        
    binder.zero_grad()
    binder.set_grads(grads)
    for p, g in zip(binder.parameters(), grads):
        assert p.grad is not None
        assert torch.allclose(p.grad, g)
