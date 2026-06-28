import torch
from src.utils.graph import Graph
from src.utils.graph_partitioning import random_assign, create_subgraphs

def test_random_assign(config):
    num_nodes = 50
    k = 5
    subgraph_node_ids = random_assign(num_nodes, k)
    
    assert len(subgraph_node_ids) == k
    assert set(subgraph_node_ids.keys()) == set(range(k))
    
    all_nodes_list = []
    for part, nodes in subgraph_node_ids.items():
        assert isinstance(nodes, torch.Tensor)
        all_nodes_list.extend(nodes.tolist())
        
    assert set(all_nodes_list) == set(range(num_nodes))
    assert len(all_nodes_list) == num_nodes

def test_create_subgraphs(config):
    num_nodes = 10
    x = torch.randn(num_nodes, 8)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 1)
    node_ids = torch.arange(num_nodes)
    
    g = Graph(edge_index=edge_index, x=x, edge_attr=edge_attr, node_ids=node_ids)
    g.num_classes = 2
    
    subgraph_node_ids = {
        0: torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        1: torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)
    }
    
    subgraphs = create_subgraphs(g, subgraph_node_ids)
    assert len(subgraphs) == 2
    for sg in subgraphs:
        assert isinstance(sg, Graph)
        assert sg.edge_index is not None
        assert sg.edge_attr is not None
