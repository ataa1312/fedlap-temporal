import os
from pathlib import Path

# isort: skip_file
import torch
from torch.nn import functional as F
from torch_geometric.transforms import FaceToEdge
from torch_geometric.datasets import (
    Planetoid,
    HeterophilousGraphDataset,
    WikipediaNetwork,
    Amazon,
    Actor,
    FAUST,
    JODIEDataset,
    EllipticBitcoinTemporalDataset,
    PPI,
)
from torch_geometric.utils import (
    dense_to_sparse,
    to_undirected,
    remove_self_loops,
    from_scipy_sparse_matrix,
)
import scipy.sparse as sp
from torch_geometric.data import Data as PyGData

from src import *
from src.utils.graph import Graph
from src.utils.create_graph import create_homophilic_graph2, create_heterophilic_graph2


def define_graph(dataset_name=config["dataset"]["name"], **kwargs):
    root = f"./datasets/{dataset_name}"
    os.makedirs(root, exist_ok=True)
    # try:
    if True:
        dataset = None
        if dataset_name in ["Cora", "PubMed", "CiteSeer"]:
            dataset = Planetoid(root=root, name=dataset_name)
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
        elif dataset_name in ["chameleon", "crocodile", "squirrel"]:
            dataset = WikipediaNetwork(
                root=root, geom_gcn_preprocess=True, name=dataset_name
            )
            node_ids = torch.arange(dataset[0].num_nodes)
            edge_index = dataset[0].edge_index
        elif dataset_name in [
            "Roman-empire",
            "Amazon-ratings",
            "Minesweeper",
            "Tolokers",
            "Questions",
        ]:
            dataset = HeterophilousGraphDataset(root=root, name=dataset_name)
        elif dataset_name in ["Actor"]:
            dataset = Actor(root=root)
        elif dataset_name in ["PPI"]:
            split = kwargs.get("split", "train")
            dataset = PPI(root=root, split=split)
        elif config["dataset"]["name"] in ["Computers", "Photo"]:
            dataset = Amazon(root=root, name=dataset_name)
        elif dataset_name == "Heterophilic_example":
            num_patterns = 500
            graph = create_heterophilic_graph2(num_patterns, use_random_features=True)
        elif dataset_name == "Homophilic_example":
            num_patterns = 100
            graph = create_homophilic_graph2(num_patterns, use_random_features=True)
        elif dataset_name == "Faust":
            dataset = FAUST(root=root, transform=FaceToEdge)
        elif dataset_name == "EllipticBitcoin":
            t = kwargs.get("t", 10)
            dataset = EllipticBitcoinTemporalDataset(root=root, t=t)
        elif dataset_name in [
            "Reddit",
            "Wikipedia",
            "MOOC",
            "LastFM",
        ]:
            dataset = JODIEDataset(root=root, name=dataset_name)

    # except:
    #     # LOGGER.info("dataset name does not exist!")
    #     return None, 0

    if dataset is not None:
        data = dataset._data
        node_ids = torch.arange(data.num_nodes)
        edge_index = data.edge_index
        x = data.x
        if x is None:
            x = data.msg

        # edge_index = to_undirected(edge_index)
        # edge_index = remove_self_loops(edge_index)[0]

        graph = Graph(
            x=x.to(device),
            y=data.y.to(device),
            edge_index=edge_index.to(device),
            node_ids=node_ids.to(device),
            keep_sfvs=True,
            dataset_name=dataset_name,
            train_mask=data.get("train_mask", None),
            val_mask=data.get("val_mask", None),
            test_mask=data.get("test_mask", None),
            time=data.get("t", None),
            num_classes=dataset.num_classes,
        )

    return graph


def get_degrees(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    device: torch.device,
    type: Literal["row", "col", "both"] = "both",
) -> torch.Tensor:
    """Compute node degrees from edge index and optional edge weights.

    Args:
        edge_index: [2, E] edge indices
        edge_weight: [E] optional edge weights (if None, assumes weight=1 for all edges)
        num_nodes: Number of nodes in the graph
        device: Device for tensor operations
        type: Type of degree to compute ("row", "col", or "both")

    Returns:
        [num_nodes] tensor of node degrees
    """
    if edge_weight is not None:
        assert edge_index.shape[1] == edge_weight.shape[0]
        edge_weight = edge_weight.to(device=device)
    else:
        edge_weight = torch.ones(edge_index.shape[1], device=device, dtype=torch.float)

    row_degrees = torch.zeros(num_nodes, device=device, dtype=torch.float)
    row_degrees = row_degrees.scatter_add_(0, edge_index[0], edge_weight)

    if type == "row":
        return row_degrees

    col_degrees = torch.zeros(num_nodes, device=device, dtype=torch.float)
    col_degrees = col_degrees.scatter_add_(0, edge_index[1], edge_weight)

    if type == "col":
        return col_degrees
    if type == "both":
        return (row_degrees + col_degrees) / 2.0


def normalize_adjacency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_nodes: int,
    device: torch.device,
):
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, num_nodes=num_nodes
    )

    degrees = get_degrees(edge_index, edge_weight, num_nodes, device, "both")

    degrees_inv_sqrt = degrees.pow(-0.5)
    degrees_inv_sqrt.masked_fill_(degrees_inv_sqrt == float("inf"), 0.0)

    row, col = edge_index
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=device, dtype=torch.float)
    edge_weight = degrees_inv_sqrt[row] * edge_weight * degrees_inv_sqrt[col]

    return edge_index, edge_weight


def pygdata_to_graph(pygdata: PyGData) -> Graph:
    if is_attr_good(pygdata, "x") and isinstance(pygdata.x, torch.Tensor):
        x = pygdata.x.clone()
        node_ids = safe_argmax(pygdata.x, dim=-1)
    else:
        raise ValueError("pygdata must have a valid x tensor")

    if is_attr_good(pygdata, "edge_index") and isinstance(
        pygdata.edge_index, torch.Tensor
    ):
        edge_index = pygdata.edge_index.clone()
    else:
        edge_index = None

    if is_attr_good(pygdata, "edge_weight") and isinstance(
        pygdata.edge_weight, torch.Tensor
    ):
        edge_weight = pygdata.edge_weight.clone()

    return Graph(
        x=x,
        y=getattr(pygdata, "y", None),
        edge_index=edge_index,
        edge_attr=edge_weight,  # Use edge_weight as edge_attr if available
        node_ids=node_ids,
        keep_sfvs=getattr(pygdata, "keep_sfvs", False),
        dataset_name=getattr(pygdata, "dataset_name", None),
        train_mask=getattr(pygdata, "train_mask", None),
        val_mask=getattr(pygdata, "val_mask", None),
        test_mask=getattr(pygdata, "test_mask", None),
        time=getattr(pygdata, "time", None),
        num_classes=getattr(pygdata, "num_classes", None),
    )


def graph_to_pygdata(graph: Graph) -> PyGData:
    if not is_attr_good(graph, "x"):
        raise ValueError("graph must have a valid x tensor")

    x = graph.x.clone() if isinstance(graph.x, torch.Tensor) else graph.x

    if is_attr_good(graph, "edge_index"):
        edge_index = graph.edge_index.clone()
    else:
        edge_index = None

    edge_weight = None
    if is_attr_good(graph, "edge_attr"):
        if isinstance(graph.edge_attr, torch.Tensor):
            edge_weight = graph.edge_attr.clone()
        else:
            edge_weight = graph.edge_attr

    pygdata = PyGData(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    return pygdata
