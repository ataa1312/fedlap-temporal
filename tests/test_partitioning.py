import torch
import numpy as np
from src.utils.graph import Graph
from src.utils.graph_partitioning import random_assign, create_subgraphs, partition_snapshots
from torch_geometric.data import Data
import pytest

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

def make_synthetic_snapshots(N=12, W=2, num_snapshots=3):
    x = torch.ones((N, 1), dtype=torch.float)
    snapshots = []
    for t in range(num_snapshots):
        if t == 0:
            edge_index = torch.tensor([
                [0, 1, 4, 5, 8, 9, 0, 4],
                [1, 2, 5, 6, 9, 10, 5, 8]
            ], dtype=torch.long)
        elif t == 1:
            edge_index = torch.tensor([
                [0, 1, 4, 5, 1, 4],
                [1, 2, 5, 6, 5, 2]
            ], dtype=torch.long)
        else:
            edge_index = torch.tensor([
                [2, 3, 6, 7, 10, 11, 3, 7],
                [3, 0, 7, 4, 11, 8, 0, 11]
            ], dtype=torch.long)
        edge_attr = torch.randn((edge_index.shape[1], W), dtype=torch.float)
        snap = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)
        snapshots.append(snap)
    return snapshots

def test_partition_snapshots_basic():
    N = 12
    W = 2
    snapshots = make_synthetic_snapshots(N, W, 3)
    num_subgraphs = 3
    out = partition_snapshots(snapshots, num_subgraphs)
    
    assert len(out) == num_subgraphs
    for client_list in out:
        assert len(client_list) == len(snapshots)
        for sg in client_list:
            assert isinstance(sg, Graph)
            
    client_nodes = [set(out[c][0].node_ids.tolist()) for c in range(num_subgraphs)]
    union_nodes = set()
    for c_nodes in client_nodes:
        assert not union_nodes.intersection(c_nodes)
        union_nodes.update(c_nodes)
    assert union_nodes == set(range(N))
    assert sum(len(c_nodes) for c_nodes in client_nodes) == N
    
    for c in range(num_subgraphs):
        first_node_ids = set(out[c][0].node_ids.tolist())
        for t in range(1, len(snapshots)):
            assert set(out[c][t].node_ids.tolist()) == first_node_ids
            
    for c in range(num_subgraphs):
        for t in range(len(snapshots)):
            sg = out[c][t]
            if sg.edge_index.numel() > 0:
                assert sg.edge_index.max().item() < len(sg.node_ids)
                u, v = sg.original_edge_index
                c_nodes = set(sg.node_ids.tolist())
                for node in u.tolist() + v.tolist():
                    assert node in c_nodes
                    
    for c in range(num_subgraphs):
        for t in range(len(snapshots)):
            sg = out[c][t]
            if sg.edge_index.numel() > 0:
                assert sg.edge_attr.shape[1] == W

def test_partition_snapshots_single_client():
    N = 12
    W = 2
    snapshots = make_synthetic_snapshots(N, W, 3)
    out = partition_snapshots(snapshots, 1)
    
    assert len(out) == 1
    assert len(out[0]) == len(snapshots)
    for t in range(len(snapshots)):
        sg = out[0][t]
        assert set(sg.node_ids.tolist()) == set(range(N))
        assert sg.edge_index.shape[1] == snapshots[t].edge_index.shape[1]

def test_partition_snapshots_mismatch_guard():
    snap1 = Data(x=torch.ones((10, 1)), edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=10)
    snap2 = Data(x=torch.ones((12, 1)), edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=12)
    with pytest.raises(AssertionError) as exc_info:
        partition_snapshots([snap1, snap2], 3)
    assert "snapshots disagree on num_nodes" in str(exc_info.value)

def test_partition_snapshots_empty_input():
    out = partition_snapshots([], 3)
    assert len(out) == 3
    for client_list in out:
        assert client_list == []

def test_partition_snapshots_reproducibility():
    N = 12
    W = 2
    snapshots = make_synthetic_snapshots(N, W, 2)
    
    np.random.seed(42)
    out1 = partition_snapshots(snapshots, 3)
    
    np.random.seed(42)
    out2 = partition_snapshots(snapshots, 3)
    
    for c in range(3):
        assert torch.equal(out1[c][0].node_ids, out2[c][0].node_ids)
