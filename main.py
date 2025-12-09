import json
import os
from pathlib import Path

curr_path = Path(__file__).parent.resolve()
os.chdir(curr_path)

from src import *
from src.FedGCN.FedGCN_server import FedGCNServer
from src.FedPub.fedpub_server import FedPubServer
from src.fedsage.fedsage_server import FedSAGEServer
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import (
    create_mend_graph,
    fedGCN_partitioning,
    partition_graph,
)


def set_up_system():
    LOGGER.info(json.dumps(config.config, indent=4))

    graph = define_graph(config.dataset.dataset_name)

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

    subgraphs = partition_graph(
        graph, config.subgraph.num_subgraphs, config.subgraph.partitioning
    )

    results = {}

    MLP_server = MLPServer(graph)
    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)

    LOGGER.info("MLP")
    res = MLP_server.train_local_model()
    results["server MLP"] = round(res["Test Acc"], 4)

    res = MLP_server.joint_train_g(FL=False)
    results["local MLP"] = round(res["Average"]["Test Acc"], 4)

    res = MLP_server.joint_train_g(FL=True)
    results["flga MLP"] = round(res["Average"]["Test Acc"], 4)

    # ============================================================ #

    GNN_server = GNNServer(graph)
    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    LOGGER.info("GNN")
    res = GNN_server.train_local_model(data_type="feature", fmodel_type="GNN")
    results["Server F GNN"] = round(res["Test Acc"], 4)

    res = GNN_server.train_local_model(data_type="structure")
    results["Server S GNN"] = round(res["Test Acc"], 4)

    res = GNN_server.train_local_model(data_type="f+s")
    results["Server F+S GNN"] = round(res["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="feature", FL=False)
    results["Local F GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="structure", FL=False)
    results["Local S GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="f+s", FL=False)
    results["Local F+S GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="feature", FL=True)
    results["FL F GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="structure", FL=True)
    results["FL S GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="f+s", FL=True)
    results["FL F+S GNN"] = round(res["Average"]["Test Acc"], 4)
    results["FL F+S(F) GNN"] = round(res["Average"]["Test Acc F"], 4)
    results["FL F+S(S) GNN"] = round(res["Average"]["Test Acc S"], 4)

    LOGGER.info(json.dumps(results, indent=4))

if __name__ == "__main__":
    set_up_system()
    # plt.show()
