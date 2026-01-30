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


def define_graph(dataset_name=config.dataset.dataset_name, **kwargs):
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
        elif config.dataset.dataset_name in ["Computers", "Photo"]:
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


def _load_custom_dataset(path: Path) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    graph_file_names = sorted((path / "graphs").rglob("*.npz"))
    features_file_names = sorted((path / "features").rglob("*.npz"))

    if len(graph_file_names) != len(features_file_names):
        raise ValueError(
            f"Mismatch: found {len(graph_file_names)} graph files and "
            f"{len(features_file_names)} feature files"
        )

    graphs: list[torch.Tensor] = list()
    features: list[torch.Tensor] = list()

    for gfn, ffn in zip(graph_file_names, features_file_names):
        LOGGER.info(f"Loading {gfn.name} and {ffn.name}...")
        garray = np.load(gfn, allow_pickle=True)
        farray = np.load(ffn, allow_pickle=True)

        if "edge_index" in garray.files and "edge_attr" in garray.files:
            ei = torch.from_numpy(garray["edge_index"]).long()
            ea = torch.from_numpy(garray["edge_attr"]).float()
            graphs.append((ei, ea)) # type: ignore
        else:
            g_item = garray["graph"]
            if g_item.ndim == 0:
                graphs.append(g_item.item())
            else:
                graphs.append(torch.from_numpy(g_item))

        f_item = farray["features"]
        if f_item.ndim == 0:
            features.append(f_item.item())
        else:
            features.append(torch.from_numpy(f_item))

    LOGGER.info(f"Loaded {len(graphs)} snapshots from {path}")
    return graphs, features


def load_dataset(
    name: DatasetName,
    path: Path,
) -> tuple[list[torch.Tensor], list[torch.Tensor]] | tuple:
    match name:
        case (
            "custom-enron"
            | "custom-uci"
            | "custom-fb"
            | "bitcoin-alpha"
            | "bitcoin-otc"
        ):
            LOGGER.info(f"Loading {name} dataset from {path}")
            return _load_custom_dataset(path)

        case _:
            raise ValueError(f"{name!r} is not a valid dataset name!")


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


def build_cleaned_graphs(
    graphs: list[torch.Tensor],
    features: list[torch.Tensor],
    normalize: bool,
    is_directed: bool,
) -> tuple[list[PyGData], int]:
    cleaned_graphs: list[PyGData] = []
    max_num_nodes: int = 0

    for idx, (g, f) in enumerate(zip(graphs, features)):
        edge_attr = None
        if isinstance(g, tuple):
            edge_index, edge_attr = g
            # Ensure tensors are on correct device if needed later
            edge_weight = None  # Will be defaulted to ones in normalize_adjacency
        elif sp.issparse(g):
            edge_index, edge_weight = from_scipy_sparse_matrix(g)
        else:
            edge_index, edge_weight = dense_to_sparse(g)

        if edge_weight is not None:
            edge_weight = edge_weight.float()

        if not is_directed:
            if edge_attr is not None:
                ei_w, edge_weight = to_undirected(edge_index, edge_weight, reduce="add")
                ei_a, edge_attr = to_undirected(edge_index, edge_attr, reduce="mean")

                if not torch.equal(ei_w, ei_a):
                    LOGGER.warning(
                        "to_undirected produced different indices for weight vs attr!"
                    )

                edge_index = ei_w
            else:
                edge_index, edge_weight = to_undirected(edge_index, edge_weight)

            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            if edge_weight is not None:
                edge_weight = edge_weight[mask]
            if edge_attr is not None:
                edge_attr = edge_attr[mask]

        if isinstance(g, tuple):
            num_nodes = f.shape[0]
        else:
            num_nodes = g.shape[0]

        if normalize:
            old_num_edges = edge_index.shape[1]
            edge_index, edge_weight = normalize_adjacency(
                edge_index, edge_weight, num_nodes, edge_index.device
            )
            if edge_attr is not None:
                new_num_edges = edge_index.shape[1]
                if new_num_edges > old_num_edges:
                    num_added = new_num_edges - old_num_edges
                    pad = torch.zeros(
                        (num_added, edge_attr.shape[1]),
                        device=edge_attr.device,
                        dtype=edge_attr.dtype,
                    )
                    edge_attr = torch.cat([edge_attr, pad], dim=0)

        if sp.issparse(f):
            f = f.tocoo()
            indices = torch.from_numpy(np.vstack((f.row, f.col))).long()
            values = torch.from_numpy(f.data).float()
            shape = torch.Size(f.shape)
            node_features = torch.sparse_coo_tensor(indices, values, shape).float()
        elif isinstance(f, np.ndarray):
            node_features = torch.from_numpy(f).float()
        else:
            node_features = f.float()

        graph = PyGData(
            x=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            time=torch.tensor(idx, dtype=torch.long),
        )
        cleaned_graphs.append(graph)

        # PyGData.num_nodes might be None inferring from edge_index if x is not set or sparse?
        # But we set x.
        # If x is sparse, num_nodes should be inferrable from x.shape[0].
        # Let's enforce it if we know it.
        # graph.num_nodes = num_nodes # PyGData usually infers this.

        # assert isinstance(graph.num_nodes, int)
        max_num_nodes = max(num_nodes, max_num_nodes)

    LOGGER.info(
        f"Built {len(cleaned_graphs)} cleaned graphs with max {max_num_nodes} nodes"
    )
    return cleaned_graphs, max_num_nodes


def build_training_graphs(
    cleaned_graphs: list[PyGData],
    tdx: int,
    dataset_name: str,
) -> tuple[list[Graph], Graph]:
    union_node_ids = set[int]()
    for t in range(tdx + 1):
        data = cleaned_graphs[t]
        assert isinstance(data.x, torch.Tensor)

        if data.x.is_sparse:
            x_coalesced = data.x.coalesce()
            indices = x_coalesced.indices()
            node_ids = indices[1].tolist()
        else:
            node_ids = data.x.argmax(dim=-1).tolist()

        node_ids = set[int](node_ids)
        union_node_ids.update(node_ids)
    union_node_ids = sorted(list[int](union_node_ids))  # Sort to guarantee order
    union_long_node_ids = torch.tensor(union_node_ids, dtype=torch.long)

    train_graphs: list[PyGData] = []
    for cg in cleaned_graphs[: tdx - 1]:
        copy_cg = cg.clone()
        assert isinstance(copy_cg.x, torch.Tensor)
        dim_features = copy_cg.x.shape[1]
        
        num_nodes = len(union_long_node_ids)
        indices = torch.stack([torch.arange(num_nodes), union_long_node_ids])
        values = torch.ones(num_nodes)
        new_x = torch.sparse_coo_tensor(
            indices, values, (num_nodes, dim_features)
        ).float()

        copy_cg.x = new_x

        # Later for conversion from PyGData to Graph
        copy_cg.dataset_name = dataset_name
        copy_cg.num_classes = 1  # Edge prediction

        train_graphs.append(copy_cg)

    test_graph = cleaned_graphs[tdx]
    last_train_graph = cleaned_graphs[tdx - 1]

    modified_graph = create_inductive_graph(test_graph, last_train_graph)
    assert isinstance(modified_graph.x, torch.Tensor)
    dim_features = modified_graph.x.shape[1]
    
    num_nodes = len(union_long_node_ids)
    indices = torch.stack([torch.arange(num_nodes), union_long_node_ids])
    values = torch.ones(num_nodes)
    new_x = torch.sparse_coo_tensor(
        indices, values, (num_nodes, dim_features)
    ).float()
    
    modified_graph.x = new_x

    # Later for conversion from PyGData to Graph
    modified_graph.dataset_name = dataset_name
    modified_graph.num_classes = 1  # Edge prediction

    train_graphs.append(modified_graph)
    converted_train_graphs: list[Graph] = [
        pygdata_to_graph(pyggraph.to(device=device)) for pyggraph in train_graphs
    ]

    test_graph = cleaned_graphs[tdx].clone()
    test_graph.dataset_name = dataset_name
    test_graph.num_classes = 1  # Edge prediction

    converted_test_graph = pygdata_to_graph(test_graph.to(device=device))

    return converted_train_graphs, converted_test_graph


def create_inductive_graph(
    current_snapshot: PyGData,
    previous_snapshot: PyGData,
) -> PyGData:
    previous_edge_weight = (
        previous_snapshot.edge_weight
        if hasattr(previous_snapshot, "edge_weight")
        and previous_snapshot.edge_weight is not None
        else None
    )
    previous_edge_attr = (
        previous_snapshot.edge_attr
        if hasattr(previous_snapshot, "edge_attr")
        and previous_snapshot.edge_attr is not None
        else None
    )
    inductive_graph = PyGData(
        x=current_snapshot.x,
        edge_index=previous_snapshot.edge_index,
        edge_weight=previous_edge_weight,
        edge_attr=previous_edge_attr,
        time=current_snapshot.time,
    )

    assert inductive_graph.num_nodes == current_snapshot.num_nodes, (
        f"Node count mismatch: inductive_graph has {inductive_graph.num_nodes} nodes, "
        f"current_snapshot has {current_snapshot.num_nodes} nodes"
    )
    assert inductive_graph.x is not None and current_snapshot.x is not None
    # assert torch.equal(inductive_graph.x, current_snapshot.x), (
    #     "Node features should match between inductive_graph and current_snapshot"
    # )
    # The above assertion fails if x is sparse? Or just unnecessary since we assigned it.

    return inductive_graph


def filter_edges_to_old_nodes(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_old_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter edges to only include those between nodes that existed in the previous snapshot.

    This matches DySAT's evaluation scheme where only edges between old nodes are evaluated.

    Args:
        edge_index: [2, num_edges] tensor of edge pairs
        num_old_nodes: Number of nodes that existed in the previous snapshot

    Returns:
        Filtered edge_index containing only edges between old nodes
    """
    mask = (edge_index[0] < num_old_nodes) & (edge_index[1] < num_old_nodes)
    return edge_index[:, mask], edge_weight[mask]


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
