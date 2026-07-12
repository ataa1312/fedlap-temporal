import torch
from src.utils.graph import Graph

def test_graph_clone_to():
    N = 10
    x = torch.ones(N, 1)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 1)
    node_ids = torch.arange(N)
    
    graph = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr, node_ids=node_ids)
    
    clone = graph.clone()
    assert clone is not graph
    assert torch.equal(clone.edge_index, graph.edge_index)
    assert clone.num_nodes == graph.num_nodes
    
    clone.x[0, 0] = 999.0
    assert graph.x[0, 0] == 1.0
    
    clone.foo = "bar"
    assert not hasattr(graph, "foo")
    
    # to() is out-of-place: a new instance with tensors on the target device,
    # leaving the original untouched (persistent snapshots must not be mutated
    # onto the compute device — that stranded/leaked GPU memory).
    returned = graph.to("cpu")
    assert returned is not graph
    assert isinstance(returned, Graph)
    assert returned.x.device.type == "cpu"
    assert returned.edge_index.device.type == "cpu"
    assert returned.edge_attr.device.type == "cpu"
    returned.baz = "qux"
    assert not hasattr(graph, "baz")
