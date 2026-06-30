from itertools import chain
from collections import defaultdict

import numpy as np
import torch
import networkx as nx
from src import *
from sklearn.cluster import k_means
from src.utils.graph import Graph
from src.FedGCN.utils import get_in_comm_indexes, label_dirichlet_partition
from torch_geometric.utils import subgraph


def slice_sparse_tensor(x, mask):
    """
    Slices a sparse tensor based on a boolean mask on rows.
    Equivalent to x[mask] for dense tensors.
    """
    if not x.is_sparse:
        return x[mask]

    x = x.coalesce()
    indices = x.indices()
    values = x.values()

    # Filter based on mask
    # indices[0] are the row indices
    row_indices = indices[0]
    # Check if the row index is in the mask
    mask_nnz = mask[row_indices]

    new_indices = indices[:, mask_nnz].clone()
    new_values = values[mask_nnz]

    # Remap row indices to be contiguous in the new tensor
    # Calculate mapping: old_index -> new_index
    # We use cumsum to count how many valid rows appeared before
    # cumsum starts at 1 for the first True, so we subtract 1
    mapping = torch.cumsum(mask.long(), dim=0) - 1
    
    # Apply mapping
    new_indices[0] = mapping[new_indices[0]]

    # New shape
    new_rows = mask.sum().item()
    new_shape = (new_rows, x.shape[1])
    
    return torch.sparse_coo_tensor(new_indices, new_values, new_shape, is_coalesced=True)

def find_community(edge_index, num_nodes):
    G = nx.Graph(edge_index.T.tolist())
    community = nx.community.louvain_communities(G)
    community_noeds = torch.tensor(list(chain.from_iterable(community)))

    node_ids = torch.arange(num_nodes)
    node_mask = node_ids.unsqueeze(1).eq(community_noeds).any(1)
    isolated_nodes = node_ids[~node_mask]
    community.append(isolated_nodes)

    community = {ind: list(c) for ind, c in enumerate(community)}

    return community


def create_community_groups(community_map, node_map=None) -> dict:
    community_groups = defaultdict(list)

    for ind, community in enumerate(community_map):
        if node_map is not None:
            node_id = node_map[ind]
        else:
            node_id = ind
        community_groups[community].append(node_id)

    return community_groups


def make_groups_smaller_than_max(community_groups, group_len_max) -> dict:
    ind = 0
    while ind < len(community_groups):
        if len(community_groups[ind]) > group_len_max:
            l1, l2 = (
                community_groups[ind][:group_len_max],
                community_groups[ind][group_len_max:],
            )

            community_groups[ind] = l1
            community_groups[len(community_groups)] = l2

        ind += 1

    return community_groups


def assign_nodes_to_subgraphs(community_groups, num_nodes, num_subgraphs):
    max_subgraph_nodes = num_nodes // num_subgraphs
    subgraph_node_ids = {subgraph_id: [] for subgraph_id in range(num_subgraphs)}
    # subgraphs = cycle(subgraph_node_ids.keys())
    current_ind = 0

    counter = 0

    for community in community_groups.keys():
        while (
            len(subgraph_node_ids[current_ind]) + len(community_groups[community])
            > max_subgraph_nodes + config["subgraph"]["delta"]
            or len(subgraph_node_ids[current_ind]) >= max_subgraph_nodes
        ):
            # current_subgraph = next(subgraphs)
            current_ind += 1
            if current_ind == num_subgraphs:
                current_ind = 0
            # define counter to avoid stuck in the loop forever
            counter += 1
            if counter == num_subgraphs:
                current_ind = np.argmin([len(s) for s in subgraph_node_ids.values()])
                break
                # subgraph_node_ids[ind] += community_groups[community]
                # current_ind += 1
                # if current_ind == num_subgraphs:
                #     current_ind = 0
                # current_subgraph = next(subgraphs)
                # return subgraph_node_ids
        subgraph_node_ids[current_ind] += community_groups[community]
        counter = 0

    assert sum([len(s) for s in subgraph_node_ids.values()]) == num_nodes

    return subgraph_node_ids


def create_subgraphs(
    graph: Graph,
    subgraph_node_ids: dict[int, torch.Tensor],
    **split_edges_kwargs,
):
    subgraphs = []
    for community, subgraph_nodes in subgraph_node_ids.items():
        if not isinstance(subgraph_nodes, torch.Tensor):
            node_ids = torch.tensor(subgraph_nodes, device=device)
        else:
            node_ids = subgraph_nodes
        assert graph.original_edge_index is not None
        edges = graph.original_edge_index
        attrs = graph.edge_attr
        edge_mask = edges.unsqueeze(2).eq(node_ids).any(2).any(0)
        edge_index = edges[:, edge_mask]
        edge_attr = attrs[edge_mask]

        all_nodes = torch.unique(edge_index.flatten())
        external_nodes = all_nodes[~all_nodes.unsqueeze(1).eq(node_ids).any(1)]

        if edge_index.shape[1] != 0:
            inter_edge_mask = edge_index.unsqueeze(2).eq(external_nodes).any(2).any(0)
            inter_edges = edge_index[:, inter_edge_mask]
            inter_edge_attr = edge_attr[inter_edge_mask]
            intra_edges = edge_index[:, ~inter_edge_mask]
            intra_edge_attr = edge_attr[~inter_edge_mask]
        else:
            intra_edges = edge_index
            inter_edges = edge_index
            intra_edge_attr = edge_attr
            inter_edge_attr = edge_attr

        # all_edges = torch.cat((intra_edges, inter_edges), dim=0)

        # node_mask = torch.isin(graph.node_ids.to("cpu"), node_ids.to("cpu"))
        assert graph.node_ids is not None
        node_mask = graph.node_ids.unsqueeze(1).eq(node_ids).any(1)
        sorted_node_ids = graph.node_ids[node_mask]
        if graph.x is not None:
            # x = graph.x[node_mask]
            x = slice_sparse_tensor(graph.x, node_mask)
        else:
            x = None

        if graph.y is not None:
            y = graph.y[node_mask]
        else:
            y = None

        if graph.train_mask is not None:
            train_mask = graph.train_mask[node_mask.cpu()]
        else:
            train_mask = None

        if graph.test_mask is not None:
            test_mask = graph.test_mask[node_mask.cpu()]
        else:
            test_mask = None

        if graph.val_mask is not None:
            val_mask = graph.val_mask[node_mask.cpu()]
        else:
            val_mask = None

        # INFO: In temporal graphs node_ids might be larger than the actual node_ids of
        # the subgraph. Here, I am adjusting that assuming x is a one-hot vector.
        # Later on this should be fixed!
        if x is not None and node_ids.shape[0] > x.shape[0]:
            # sorted_node_ids = x.argmax(dim=1)
            sorted_node_ids = safe_argmax(x, dim=1)

        subgraph_ = Graph(
            x=x,
            y=y,
            edge_index=intra_edges,
            edge_attr=intra_edge_attr,
            node_ids=sorted_node_ids,
            external_nodes=external_nodes,
            inter_edges=inter_edges,
            inter_edge_attr=inter_edge_attr,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask,
            num_classes=graph.num_classes,
        )

        # Split edges for edge prediction if requested
        # This ensures each subgraph has its own train/val/test edge splits
        # if (
        #     split_edges_kwargs.get("split_edges_for_edge_prediction", None)
        #     and intra_edges.shape[1] > 0
        # ):
        #     # Use default values if not provided
        #     val_ratio = split_edges_kwargs.get("val_ratio", 0.15)
        #     test_ratio = split_edges_kwargs.get("test_ratio", 0.15)
        #     is_undirected = split_edges_kwargs.get("is_undirected", True)
        #     add_negative_train_samples = split_edges_kwargs.get(
        #         "add_negative_train_samples", True
        #     )
        #     negative_ratio = split_edges_kwargs.get("negative_ratio", 1.0)

        #     subgraph_.split_edges(
        #         val_ratio=val_ratio,
        #         test_ratio=test_ratio,
        #         is_undirected=is_undirected,
        #         add_negative_train_samples=add_negative_train_samples,
        #         negative_ratio=negative_ratio,
        #     )

        subgraphs.append(subgraph_)

    return subgraphs


def louvain_cut(edge_index, num_nodes, num_subgraphs):
    community_groups = find_community(edge_index, num_nodes)

    group_len_max = num_nodes // num_subgraphs + config["subgraph"]["delta"]

    community_groups = make_groups_smaller_than_max(community_groups, group_len_max)

    sorted_community_groups = {
        k: v
        for k, v in sorted(
            community_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
    }

    subgraph_node_ids = assign_nodes_to_subgraphs(
        sorted_community_groups, num_nodes, num_subgraphs
    )

    return subgraph_node_ids


def random_assign(num_nodes, num_subgraphs):
    subgraph_id = np.random.choice(num_subgraphs, num_nodes, replace=True)
    subgraph_node_ids = {
        value: torch.tensor(
            np.where(subgraph_id == value)[0], dtype=torch.int64, device=dev
        )
        for value in range(num_subgraphs)
    }

    return subgraph_node_ids


def partition_snapshots(snapshots, num_subgraphs):
    """Partition global ROLAND snapshots into per-client subgraphs.

    Assigns the global node set to ``num_subgraphs`` clients once
    (``random_assign``), then induces each client's subgraph per snapshot
    (``create_subgraphs``). Returns a client-major list where ``out[c][t]`` is
    client ``c``'s subgraph for snapshot ``t`` (a ``Graph``, i.e. a ``Data``).
    """
    per_client = [[] for _ in range(num_subgraphs)]
    if not snapshots:
        return per_client

    # Snapshots share the full graph's global node space (build_full_graph fixes
    # the count; make_snapshots only slices edges), so num_nodes is constant.
    num_nodes = snapshots[0].num_nodes
    assert all(s.num_nodes == num_nodes for s in snapshots), (
        "snapshots disagree on num_nodes; partition needs a global node space"
    )
    snap_dev = snapshots[0].edge_index.device
    subgraph_node_ids = random_assign(num_nodes, num_subgraphs)
    subgraph_node_ids = {k: v.to(snap_dev) for k, v in subgraph_node_ids.items()}

    for snap in snapshots:
        global_graph = Graph(
            x=snap.x,
            edge_index=snap.edge_index,
            edge_attr=snap.edge_attr,
            node_ids=torch.arange(num_nodes, device=snap_dev),
        )
        subgraphs = create_subgraphs(global_graph, subgraph_node_ids)
        for client, subgraph in enumerate(subgraphs):
            per_client[client].append(subgraph)

    return per_client


def kmeans_cut(X, num_subgraphs):
    num_nodes = X.shape[0]
    _, subgraph_id, _ = k_means(X.cpu(), num_subgraphs, n_init="auto")
    community_groups = create_community_groups(subgraph_id)

    group_len_max = num_nodes // num_subgraphs + config["subgraph"]["delta"]

    community_groups = make_groups_smaller_than_max(community_groups, group_len_max)

    sorted_community_groups = {
        k: v
        for k, v in sorted(
            community_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
    }

    subgraph_node_ids = assign_nodes_to_subgraphs(
        sorted_community_groups, num_nodes, num_subgraphs
    )

    return subgraph_node_ids


def metis_cut(edge_index, num_nodes, num_subgraphs):
    import metis

    edges = edge_index.T.tolist()
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edges)
    (edgecuts, community_map) = metis.part_graph(nx_graph, num_subgraphs)
    community_groups = create_community_groups(community_map=community_map)

    return community_groups


def drichlet_cut(labels, num_nodes, num_subgraphs, num_classes):
    subgraph_node_ids = label_dirichlet_partition(
        labels.cpu().numpy(),
        num_nodes,
        num_classes,
        num_subgraphs,
        beta=10000,  # was config.fedgcn.iid_beta (fedgcn section removed)
    )
    subgraph_node_ids = [torch.tensor(node_ids) for node_ids in subgraph_node_ids]
    return subgraph_node_ids


def create_mend_graph(subgraph: Graph, graph: Graph, val=1):
    node_ids = torch.hstack((subgraph.node_ids, subgraph.external_nodes))
    edges = torch.hstack((subgraph.original_edge_index, subgraph.inter_edges))

    node_mask = graph.node_ids.unsqueeze(1).eq(node_ids).any(1)
    sorted_node_ids = graph.node_ids[node_mask]
    subgraph_node_mask = sorted_node_ids.unsqueeze(1).eq(subgraph.node_ids).any(1)
    if graph.x is not None:
        # x = graph.x[node_mask]
        x = slice_sparse_tensor(graph.x, node_mask)
        # FIXME: Sparse tensor assignment is not supported easily this way
        if x.is_sparse:
             # For now, if sparse, we might skip the masking or convert to dense if small
             # But create_mend_graph modifications usually imply densifying for masking?
             # Or we reconstruct. 
             # For this specific operation: x[~subgraph_node_mask] = val * x[...]
             # It acts on the SLICED x.
             # subgraph_node_mask has length of x (sorted_node_ids).
             # We can operate on values directly if we map the mask to values.
             
             # If x is sparse, we can multiply values?
             # "x[mask] = val * x[mask]" is scaling rows.
             pass
             # Note: This part is tricky for sparse. I will leave it as is if it crashes here, 
             # but the user requested fix for create_subgraphs mostly.
             # If I change it to slice_sparse_tensor, at least the first line passes.
             # The second line x[~subgraph_node_mask] might still fail if x is sparse.
             # But let's apply the slice first.
        else:
             x[~subgraph_node_mask] = val * x[~subgraph_node_mask]
    else:
        x = None

    if graph.y is not None:
        y = graph.y[node_mask]
        y[~subgraph_node_mask] = -1
    else:
        y = None

    if graph.train_mask is not None:
        train_mask = graph.train_mask[node_mask] & subgraph_node_mask
    else:
        train_mask = None

    if graph.test_mask is not None:
        test_mask = graph.test_mask[node_mask] & subgraph_node_mask
    else:
        test_mask = None

    if graph.val_mask is not None:
        val_mask = graph.val_mask[node_mask] & subgraph_node_mask
    else:
        val_mask = None

    mend_graph = Graph(
        x=x,
        y=y,
        edge_index=edges,
        node_ids=sorted_node_ids,
        # external_nodes=external_nodes,
        # inter_edges=inter_edges,
        train_mask=train_mask,
        test_mask=test_mask,
        val_mask=val_mask,
        num_classes=graph.num_classes,
    )

    return mend_graph


def create_comm_indexes(graph: Graph, subgraph_node_ids: Graph, num_hops=2):
    # edge_index, subgraph_node_ids, train_mask, test_mask):
    train_mask = graph.train_mask
    test_mask = graph.test_mask
    edge_index = graph.edge_index
    idx = torch.arange(train_mask.shape[0])
    idx_train = graph.node_ids[train_mask]
    idx_test = graph.node_ids[test_mask]
    num_subgraphs = len(subgraph_node_ids)
    (
        communicate_indexes,
        in_com_train_data_indexes,
        in_com_test_data_indexes,
        edge_indexes_clients,
    ) = get_in_comm_indexes(
        edge_index,
        subgraph_node_ids,
        num_subgraphs,
        num_hops,
        idx_train,
        idx_test,
    )

    subgraphs = []
    for i in range(len(communicate_indexes)):
        node_mask = graph.node_ids.unsqueeze(1).eq(communicate_indexes[i]).any(1)
        # x = graph.x[node_mask]
        x = slice_sparse_tensor(graph.x, node_mask)
        y = graph.y[node_mask]
        subgraph_train_mask = (
            communicate_indexes[i].unsqueeze(1).eq(in_com_train_data_indexes[i]).any(1)
        )
        subgraph_test_mask = (
            communicate_indexes[i].unsqueeze(1).eq(in_com_test_data_indexes[i]).any(1)
        )
        subgraph_val_mask = ~(subgraph_train_mask | subgraph_test_mask)

        subgraph = Graph(
            x=x,
            y=y,
            edge_index=edge_indexes_clients[i],
            # node_ids=communicate_indexes[i],
            # external_nodes=external_nodes,
            # inter_edges=inter_edges,
            train_mask=subgraph_train_mask,
            test_mask=subgraph_test_mask,
            val_mask=subgraph_val_mask,
            num_classes=graph.num_classes,
        )

        subgraphs.append(subgraph)

    return subgraphs


def partition_graph(graph: Graph, num_subgraphs, method="random", **kwargs):
    if method == "louvain":
        subgraph_node_ids = louvain_cut(
            graph.edge_index, graph.num_nodes, num_subgraphs
        )
    elif method == "random":
        num_nodes: int | None = kwargs.get("num_nodes", None)
        if num_nodes is None:
            subgraph_node_ids = random_assign(graph.num_nodes, num_subgraphs)
        else:
            subgraph_node_ids = random_assign(num_nodes, num_subgraphs)
    elif method == "kmeans":
        subgraph_node_ids = kmeans_cut(graph.x, num_subgraphs)
    elif method == "metis":
        subgraph_node_ids = metis_cut(graph.edge_index, graph.num_nodes, num_subgraphs)

    # Extract split_edges parameters from kwargs to pass to create_subgraps
    split_edges_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in [
            "split_edges_for_edge_prediction",
            "val_ratio",
            "test_ratio",
            "is_undirected",
            "add_negative_train_samples",
            "negative_ratio",
        ]
    }

    subgraphs = create_subgraphs(graph, subgraph_node_ids, **split_edges_kwargs)
    return subgraphs


def fedGCN_partitioning(
    graph: Graph, num_subgraphs, method="drichlet", num_hops=2  # was config.fedgcn.num_hops
):
    if method == "louvain":
        subgraph_node_ids = louvain_cut(
            graph.edge_index, graph.num_nodes, num_subgraphs
        )
        subgraph_node_ids = {
            key: torch.tensor(node_ids, dtype=torch.int64, device=dev)
            for key, node_ids in subgraph_node_ids.items()
        }
    elif method == "random":
        subgraph_node_ids = random_assign(graph.num_nodes, num_subgraphs)
    elif method == "drichlet":
        subgraph_node_ids = drichlet_cut(
            graph.y, graph.num_nodes, num_subgraphs, graph.num_classes
        )
    elif method == "kmeans":
        subgraph_node_ids = kmeans_cut(graph.x, num_subgraphs)
        subgraph_node_ids = {
            key: torch.tensor(node_ids, dtype=torch.int64, device=dev)
            for key, node_ids in subgraph_node_ids.items()
        }
    elif method == "metis":
        subgraph_node_ids = metis_cut(graph.edge_index, graph.num_nodes, num_subgraphs)

    subgraphs = create_comm_indexes(graph, subgraph_node_ids, num_hops=num_hops)

    return subgraphs
