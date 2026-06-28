import torch
from src.utils.graph import Graph

def test_calc_eignvalues_default(config):
    num_nodes = 5
    x = torch.randn(num_nodes, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long)
    g = Graph(edge_index=edge_index, x=x)
    
    # Under the default config (matrix='lap', decompose='eigh', estimate=False),
    # this will raise UnboundLocalError due to a migration gap in calc_eignvalues where V is referenced before assignment.
    # The test runs naturally to expose this failure.
    D, U, V = g.calc_eignvalues()
    assert D.shape[0] == num_nodes
    assert U.shape == (num_nodes, num_nodes)

def test_calc_eignvalues_estimate(config):
    num_nodes = 5
    x = torch.randn(num_nodes, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long)
    g = Graph(edge_index=edge_index, x=x)
    
    config["spectral"]["lanczos_iter"] = 2
    D, U, V = g.calc_eignvalues(estimate=True)
    assert D.ndim == 1
    assert U.ndim == 2
    assert V.ndim == 2

def test_update_eigpairs(config):
    num_nodes = 5
    x = torch.randn(num_nodes, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long)
    g = Graph(edge_index=edge_index, x=x)
    
    prev_Q = torch.randn(num_nodes, 3)
    next_D, next_U, returned_prev_Q = g.update_eigpairs(prev_Q)
    assert next_D.shape == (3,)
    assert next_U.shape == (num_nodes, 3)
    assert returned_prev_Q.shape == prev_Q.shape

def test_procrustes_project(config):
    num_nodes = 5
    x = torch.randn(num_nodes, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long)
    g = Graph(edge_index=edge_index, x=x)
    
    U_a = torch.randn(5, 3)
    U_b = torch.randn(5, 3)
    aligned = g.procrustes_project(U_a, U_b)
    
    assert aligned.shape == U_a.shape
    norm_a = torch.norm(U_a, p="fro")
    norm_aligned = torch.norm(aligned, p="fro")
    assert torch.allclose(norm_a, norm_aligned, atol=1e-4)
