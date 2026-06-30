from pathlib import Path
import pytest
import torch
from torch_geometric.data import Data
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots

_ROOT = Path(__file__).resolve().parent.parent
HAS_UCI = (_ROOT / "data/college-msg/college-msg.txt").is_file()
HAS_BTC = (_ROOT / "data/bitcoin-otc-raw/soc-sign-bitcoinotc.csv").is_file()

@pytest.mark.skipif(not HAS_UCI, reason="UCI CollegeMsg raw data not found")
def test_uci_loader(config):
    import src.datasets
    from registries import datasets as REG
    
    config["dataset"]["name"] = "uci"
    config["dataset"]["path"] = "data"
    snaps = REG["uci"](config)
    
    assert isinstance(snaps, list)
    assert len(snaps) > 0
    assert len(snaps) == 28
    
    first_num_nodes = snaps[0].num_nodes
    for s in snaps:
        assert isinstance(s, Data)
        assert s.num_nodes == first_num_nodes
        assert s.x.shape == (first_num_nodes, 1)
        assert s.x.eq(1).all()
        assert s.edge_attr.ndim == 2
        assert s.edge_attr.shape[1] == 1
        assert s.num_edges >= 10

@pytest.mark.skipif(not HAS_BTC, reason="Bitcoin OTC raw data not found")
def test_bitcoin_otc_loader(config):
    import src.datasets
    from registries import datasets as REG
    
    config["dataset"]["name"] = "bitcoin_otc"
    config["dataset"]["path"] = "data"
    snaps = REG["bitcoin_otc"](config)
    
    assert isinstance(snaps, list)
    assert len(snaps) > 0
    assert len(snaps) == 262
    
    first_num_nodes = snaps[0].num_nodes
    for s in snaps:
        assert isinstance(s, Data)
        assert s.num_nodes == first_num_nodes
        assert s.x.shape == (first_num_nodes, 1)
        assert s.x.eq(1).all()
        assert s.edge_attr.ndim == 2
        assert s.edge_attr.shape[1] == 2
        assert s.num_edges >= 10

@pytest.mark.skipif(not HAS_UCI, reason="UCI CollegeMsg raw data not found")
def test_end_to_end_uci_partition(config):
    import src.datasets
    from registries import datasets as REG
    
    config["dataset"]["name"] = "uci"
    config["dataset"]["path"] = "data"
    snaps = REG["uci"](config)
    
    num_subgraphs = 3
    out = partition_snapshots(snaps, num_subgraphs)
    
    assert len(out) == num_subgraphs
    N = snaps[0].num_nodes
    
    client_nodes = [set(out[c][0].node_ids.tolist()) for c in range(num_subgraphs)]
    union_nodes = set()
    for c_nodes in client_nodes:
        assert not union_nodes.intersection(c_nodes)
        union_nodes.update(c_nodes)
    assert union_nodes == set(range(N))
