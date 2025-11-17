import os
import sys
import json


from src import *
from src.GNN.GNN_server import GNNServer
from src.MLP.MLP_server import MLPServer
from src.FedGCN.FedGCN_server import FedGCNServer
from src.fedsage.fedsage_server import FedSAGEServer
from src.FedPub.fedpub_server import FedPubServer
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

    MLP_server = MLPServer(graph)
    for subgraph in subgraphs:
        MLP_server.add_client(subgraph)

    GNN_server = GNNServer(graph)
    for subgraph in subgraphs:
        GNN_server.add_client(subgraph)

    GNN_server2 = GNNServer(graph)
    for subgraph in subgraphs:
        mend_graph = create_mend_graph(subgraph, graph, 0)
        GNN_server2.add_client(mend_graph)

    GNN_server3 = GNNServer(graph)
    for subgraph in subgraphs:
        mend_graph = create_mend_graph(subgraph, graph, 1)
        GNN_server3.add_client(mend_graph)

    fedsage_server = FedSAGEServer(graph)
    for subgraph in subgraphs:
        fedsage_server.add_client(subgraph)

    fedpub_server = FedPubServer(graph)
    for subgraph in subgraphs:
        fedpub_server.add_client(subgraph)

    fedgcn_server = FedGCNServer(graph)
    fedgcn_subgraphs = fedGCN_partitioning(
        graph,
        config.subgraph.num_subgraphs,
        method=config.subgraph.partitioning,
        num_hops=config.fedgcn.num_hops,
    )
    for subgraph in fedgcn_subgraphs:
        fedgcn_server.add_client(subgraph)

    results = {}

    LOGGER.info("MLP")
    res = MLP_server.train_local_model()
    results[f"server MLP"] = round(res["Test Acc"], 4)

    res = MLP_server.joint_train_g(FL=False)
    results[f"local MLP"] = round(res["Average"]["Test Acc"], 4)

    res = MLP_server.joint_train_g(FL=True)
    results[f"flga MLP"] = round(res["Average"]["Test Acc"], 4)

    LOGGER.info("GNN")
    res = GNN_server.train_local_model(data_type="feature", fmodel_type="GNN")
    results[f"Server F GNN"] = round(res["Test Acc"], 4)

    res = GNN_server.train_local_model(data_type="structure")
    results[f"Server S GNN"] = round(res["Test Acc"], 4)

    res = GNN_server.train_local_model(data_type="f+s")
    results[f"Server F+S GNN"] = round(res["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="feature", FL=False)
    results[f"Local F GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="structure", FL=False)
    results[f"Local S GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="f+s", FL=False)
    results[f"Local F+S GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="feature", FL=True)
    results[f"FL F GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="structure", FL=True)
    results[f"FL S GNN"] = round(res["Average"]["Test Acc"], 4)

    res = GNN_server.joint_train_g(data_type="f+s", FL=True)
    results[f"FL F+S GNN"] = round(res["Average"]["Test Acc"], 4)
    results[f"FL F+S(F) GNN"] = round(res["Average"]["Test Acc F"], 4)
    results[f"FL F+S(S) GNN"] = round(res["Average"]["Test Acc S"], 4)

    # res = fedsage_server.train_fedSage_plus()
    # results[f"fedsage WA"] = round(res["WA"]["Average"]["Test Acc"], 4)
    # results[f"fedsage GA"] = round(res["GA"]["Average"]["Test Acc"], 4)

    # res = fedpub_server.start()
    # results[f"fedpub"] = round(res["Average"]["Test Acc"], 4)
    # res = GNN_server3.joint_train_w(data_type="feature", fmodel_type="GNN", FL=True)
    # results[f"FedSage Ideal"] = round(res["Average"]["Test Acc"], 4)

    # res = fedgcn_server.joint_train_w()
    # results[f"fedgcn"] = round(res["Average"]["Test Acc"], 4)

    LOGGER.info(json.dumps(results, indent=4))


if __name__ == "__main__":
    set_up_system()
    # plt.show()
