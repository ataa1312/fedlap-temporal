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
    
    returned = graph.to("cpu")
    assert returned is graph
    assert graph.x.device.type == "cpu"
    assert graph.edge_index.device.type == "cpu"
    assert graph.edge_attr.device.type == "cpu"
