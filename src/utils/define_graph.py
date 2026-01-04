import os
from pathlib import Path

# isort: skip_file
import torch
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
from torch_geometric.utils import dense_to_sparse, to_undirected, remove_self_loops
from torch_geometric.data import Data

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

        elif dataset_name == "CustomEnron" or dataset_name == "CustomUci":
            t = kwargs.get("t", None)
            assert t is not None, "`t` should be provided for dynamic datasets!"

            root_path = Path(root)
            graph_file_names = sorted((root_path / "graphs").rglob("*.npz"))
            features_file_names = sorted((root_path / "features").rglob("*.npz"))

            if len(graph_file_names) != len(features_file_names):
                raise ValueError(
                    f"Mismatch: found {len(graph_file_names)} graph files and "
                    f"{len(features_file_names)} feature files"
                )

            gfn = graph_file_names[t]
            ffn = features_file_names[t]

            LOGGER.info(f"Loading {gfn.name} and {ffn.name}...")
            graph_tensor = torch.from_numpy(np.load(gfn)["graph"])
            feature_tensor = torch.from_numpy(np.load(ffn)["features"]).float()
            LOGGER.info(f"Loaded snapshot #{t} from {root_path}")
            edge_index, edge_attr = dense_to_sparse(graph_tensor)
            edge_index, edge_attr = to_undirected(edge_index, edge_attr)
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)


            node_ids = feature_tensor.argmax(dim=1)
            
            graph = Graph(
                x=feature_tensor,
                y=None,
                edge_index=edge_index.to(device),
                edge_attr=edge_attr.to(device),
                node_ids=node_ids.to(device),
                keep_sfvs=True,
                dataset_name=dataset_name,
                train_mask=None,
                val_mask=None,
                test_mask=None,
                time=t,
                num_classes=1,  # FIXME: This might be wrong
            )
            return graph

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


def build_training_snapshots(
    T: int,
    dataset_name: str = config.dataset.dataset_name,
) -> tuple[dict[int, Graph], list[int], int, Graph]:
    # Training snapshots: 0 to T-2
    train_indices = list[int](range(0, T - 1))
    test_index = T - 1
    
    train_snapshots = dict[int, Graph]()
    for tdx in train_indices:
        snapshot = define_graph(dataset_name, t=tdx)
        train_snapshots[tdx] = snapshot
    
    current_snapshot = define_graph(dataset_name, t=test_index)
    previous_snapshot = train_snapshots[T - 2]  # Snapshot T-2
    
    inductive_graph = Graph(
        x=current_snapshot.x,  # Nodes from current snapshot (T-1)
        y=current_snapshot.y,
        edge_index=previous_snapshot.original_edge_index.clone(),  # Original edges from T-2
        edge_attr=previous_snapshot.edge_attr.clone() if previous_snapshot.edge_attr is not None else None,
        node_ids=current_snapshot.node_ids,  # Node IDs from T-1 (Graph will reindex edges accordingly)
        keep_sfvs=current_snapshot.keep_sfvs,
        dataset_name=getattr(current_snapshot, "dataset_name", None),
        train_mask=getattr(current_snapshot, "train_mask", None),
        val_mask=getattr(current_snapshot, "val_mask", None),
        test_mask=getattr(current_snapshot, "test_mask", None),
        time=getattr(current_snapshot, "time", None),
        num_classes=getattr(current_snapshot, "num_classes", None),
    )
    train_snapshots[T - 1] = inductive_graph
    
    test_snapshot = define_graph(dataset_name, t=test_index)
    
    return train_snapshots, train_indices, test_index, test_snapshot
