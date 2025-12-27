import os
import json
from pathlib import Path

from torch_geometric.utils import num_nodes

curr_path = Path(__file__).parent.resolve()
os.chdir(curr_path)

from src import *
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.utils.define_graph import define_graph
from src.FedGCN.FedGCN_server import FedGCNServer
from src.FedPub.fedpub_server import FedPubServer
from src.fedsage.fedsage_server import FedSAGEServer
from src.utils.graph_partitioning import (
    partition_graph,
    create_mend_graph,
    fedGCN_partitioning,
)


def get_max_num_nodes(dataset_name: str) -> int:
    root = f"./datasets/{dataset_name}"
    root_path = Path(root)
    graph_file_names = sorted((root_path / "graphs").rglob("*.npz"))
    features_file_names = sorted((root_path / "features").rglob("*.npz"))

    if len(graph_file_names) != len(features_file_names):
        raise ValueError(
            f"Mismatch: found {len(graph_file_names)} graph files and "
            f"{len(features_file_names)} feature files"
        )

    max_num_nodes: int = 0
    for gfn, ffn in zip(graph_file_names, features_file_names):
        graph_tensor = torch.from_numpy(np.load(gfn)["graph"])
        feature_tensor = torch.from_numpy(np.load(ffn)["features"])
        assert len(graph_tensor) == len(feature_tensor)
        max_num_nodes = max(len(graph_tensor), max_num_nodes)

    return max_num_nodes


def set_up_system():
    LOGGER.info(json.dumps(config.config, indent=4))

    max_num_nodes = get_max_num_nodes(config.dataset.dataset_name)
    graph = define_graph(config.dataset.dataset_name, t=5)

    if config.model.smodel_type in ["DGCN", "CentralDGCN"]:
        graph.calc_abar(
            config.structure_model.DGCN_layers,
            method=config.model.smodel_type,
            pruning=config.subgraph.prune,
        )

    graph.add_masks(
        train_ratio=config.subgraph.train_ratio,
        test_ratio=config.subgraph.test_ratio,
    )
    graph.split_edges(val_ratio=0.10, test_ratio=0.10, is_undirected=True)

    subgraph_node_ids = None
    subgraphs = partition_graph(
        graph,
        config.subgraph.num_subgraphs,
        config.subgraph.partitioning,
        subgraph_node_ids=subgraph_node_ids,
        # num_nodes=max_num_nodes,
        return_node_ids=False,
    )

    results = {}

    # ============================================================ #

    GNN_server = GNNServer(graph)
    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    downstream_task: DownstreamTask = "edge-prediction"
    LOGGER.info("GNN")
    res = GNN_server.train_local_model(
        data_type="feature", fmodel_type="GNN", downstream_task=downstream_task
    )
    results["Server F GNN"] = round(res["Test Acc"], 4)

    res = GNN_server.train_local_model(
        data_type="structure", downstream_task=downstream_task
    )
    results["Server S GNN"] = round(res["Test Acc"], 4)

    res = GNN_server.train_local_model(
        data_type="f+s", fmodel_type="GNN", downstream_task=downstream_task
    )
    results["Server F+S GNN"] = round(res["Test Acc"], 4)

    # res = GNN_server.joint_train_g(data_type="feature", FL=False)
    # results["Local F GNN"] = round(res["Average"]["Test Acc"], 4)
    #
    # res = GNN_server.joint_train_g(data_type="structure", FL=False)
    # results["Local S GNN"] = round(res["Average"]["Test Acc"], 4)
    #
    # res = GNN_server.joint_train_g(data_type="f+s", FL=False)
    # results["Local F+S GNN"] = round(res["Average"]["Test Acc"], 4)
    #
    # res = GNN_server.joint_train_g(data_type="feature", FL=True)
    # results["FL F GNN"] = round(res["Average"]["Test Acc"], 4)
    #
    # res = GNN_server.joint_train_g(data_type="structure", FL=True)
    # results["FL S GNN"] = round(res["Average"]["Test Acc"], 4)
    #
    # res = GNN_server.joint_train_g(data_type="f+s", FL=True)
    # results["FL F+S GNN"] = round(res["Average"]["Test Acc"], 4)
    # results["FL F+S(F) GNN"] = round(res["Average"]["Test Acc F"], 4)
    # results["FL F+S(S) GNN"] = round(res["Average"]["Test Acc S"], 4)

    LOGGER.info(json.dumps(results, indent=4))


if __name__ == "__main__":
    set_up_system()
    # plt.show()
