import os
from copy import deepcopy
from operator import itemgetter
from collections import defaultdict

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from src import *
from scipy import sparse
from torch_sparse import SparseTensor
from src.models.GDV import GDV
from src.utils.data import Data
from src.GNN.Lanczos import estimate_eigh
from src.utils.utils import create_rw, find_neighbors_
from torch_geometric.nn import MessagePassing
from src.models.Node2Vec import find_node2vect_embedings
from torch_geometric.data import Data as PyGData
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import degree, to_networkx
from torch_geometric.transforms import RandomLinkSplit

dataset_name = config.dataset.dataset_name


class AGraph(Data):
    def __init__(
        self,
        abar: torch.Tensor | SparseTensor,
        x: torch.Tensor | SparseTensor | None = None,
        y: torch.Tensor | None = None,
        node_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(x, y, node_ids, **kwargs)
        self.abar = abar


class Graph(Data):
    def __init__(
        self,
        edge_index: torch.Tensor | None = None,
        x: torch.Tensor | SparseTensor | None = None,
        edge_attr: torch.Tensor | SparseTensor | None = None,
        y: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        node_ids=None,
        keep_sfvs=False,
        **kwargs,
    ) -> None:
        if node_ids is None:
            node_ids = torch.arange(len(x))
        super().__init__(
            x=x,
            y=y,
            node_ids=node_ids,
            **kwargs,
        )

        self.original_edge_index = edge_index
        node_map, new_edges = Graph.reindex_nodes(node_ids, edge_index)
        self.edge_index = new_edges
        self.node_map = node_map
        self.edge_attr = edge_attr
        self.pos = pos
        self.inv_map = {v: k for k, v in node_map.items()}
        self.num_edges = edge_index.shape[1]

        self.inter_edges = kwargs.get("inter_edges", None)
        self.external_nodes = kwargs.get("external_nodes", None)
        self.inter_edge_attr = kwargs.get("inter_edge_attr", None)

        self.keep_sfvs = keep_sfvs
        if self.keep_sfvs:
            self.sfvs = {}

        self.DGCN_abar = None
        self.structural_features = None
        self.L = None

        # Edge prediction attrs
        self.message_passing_edge_index = kwargs.get("message_passing_edge_index", None)
        self.train_edge_label_index = kwargs.get("train_edge_label_index", None)
        self.train_edge_label = kwargs.get("train_edge_label", None)
        self.val_edge_label_index = kwargs.get("val_edge_label_index", None)
        self.val_edge_label = kwargs.get("val_edge_label", None)
        self.test_edge_label_index = kwargs.get("test_edge_label_index", None)
        self.test_edge_label = kwargs.get("test_edge_label", None)

    def get_edges(self):
        # return only the original intra edges
        return self.original_edge_index

    def get_all_edges(self):
        # return both intra and inter connections
        if self.inter_edges is not None:
            return torch.concat((self.original_edge_index, self.inter_edges), dim=1)
        return self.get_edges()

    def split_edges(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        is_undirected: bool = True,
        add_negative_train_samples: bool = True,
        negative_ratio: float = 1.0,
        **kwargs,
    ):
        if val_ratio < 0 or test_ratio < 0:
            raise ValueError("val_ratio and test_ratio must be non-negative")
        if val_ratio + test_ratio >= 1.0:
            raise ValueError(
                f"Sum of val_ratio ({val_ratio}) and test_ratio ({test_ratio}) must be < 1.0. "
                f"Training ratio will be {1.0 - val_ratio - test_ratio}"
            )

        transform = RandomLinkSplit(
            val_ratio,
            test_ratio,
            is_undirected,
            add_negative_train_samples=add_negative_train_samples,
            neg_sampling_ratio=negative_ratio,
        )

        graph = PyGData(self.x, self.edge_index)

        train_graph, val_graph, test_graph = transform(graph)

        self.message_passing_edge_index = train_graph.edge_index
        self.train_edge_label_index = train_graph.edge_label_index
        self.train_edge_label = train_graph.edge_label
        self.val_edge_label_index = val_graph.edge_label_index
        self.val_edge_label = val_graph.edge_label
        self.test_edge_label_index = test_graph.edge_label_index
        self.test_edge_label = test_graph.edge_label

    def reindex_nodes(nodes, edges):
        node_map = {node.item(): ind for ind, node in enumerate(nodes)}
        if edges.shape[1] == 0:
            new_edges = torch.empty((2, 0), dtype=torch.int64, device=edges.device)
        else:
            new_edges = edges.cpu().numpy()

            new_edges = np.vstack(
                (
                    itemgetter(*new_edges[0])(node_map),
                    itemgetter(*new_edges[1])(node_map),
                )
            )

            new_edges = torch.tensor(new_edges, dtype=torch.int64, device=edges.device)

        return node_map, new_edges

    def add_structural_features(
        self,
        structure_type="degree",
        num_structural_features=100,
        num_spectral_features=None,
    ):
        structural_features = None
        if self.keep_sfvs:
            if structure_type in self.sfvs.keys():
                structural_features = self.sfvs[structure_type]

        if structural_features is None:
            structural_features = Graph.add_structural_features_(
                self.get_edges(),
                self.num_nodes,
                structure_type=structure_type,
                num_structural_features=num_structural_features,
                num_spectral_features=num_spectral_features,
                save=True,
            )
            if structure_type in ["degree", "GDV", "node2vec"]:
                if self.keep_sfvs:
                    self.sfvs[structure_type] = deepcopy(structural_features)

        self.structural_features = structural_features
        self.num_structural_features = structural_features.shape[1]

    def add_structural_features_(
        edge_index,
        num_nodes=None,
        structure_type="degree",
        num_structural_features=100,
        num_spectral_features=None,
        save=False,
    ) -> torch.Tensor | None:
        if num_nodes is None:
            num_nodes = max(torch.flatten(edge_index)) + 1

        directory = f"models/{dataset_name}/{structure_type}/"
        path = f"{directory}{structure_type}_model.pkl"
        if os.path.exists(path):
            structural_features = torch.load(path)
            return structural_features

        if structure_type == "degree":
            structural_features = Graph.calc_degree_features(
                edge_index, num_nodes, num_structural_features
            )
        elif structure_type == "GDV":
            structural_features = Graph.calc_GDV(edge_index)
        elif structure_type == "node2vec":
            structural_features = find_node2vect_embedings(
                edge_index, embedding_dim=num_structural_features
            )
        elif structure_type == "mp":
            structural_features = Graph.calc_mp(
                edge_index,
                num_nodes,
                num_structural_features,
                iteration=config.structure_model.num_mp_vectors,
            )
        elif structure_type == "hop2vec":
            if num_spectral_features is None:
                num_spectral_features = num_nodes
            structural_features = Graph.initialize_random_features(
                size=(num_spectral_features, num_structural_features)
            )
        elif structure_type == "fedstar":
            structural_features = Graph.calc_fedStar(
                edge_index, num_nodes, num_structural_features
            )
        else:
            structural_features = None

        if save and structure_type in ["GDV", "node2vec", "fedstar"]:
            os.makedirs(directory, exist_ok=True)
            torch.save(structural_features, path)

        return structural_features

    def calc_degree_features(edge_index, num_nodes, size=100):
        node_degree1 = degree(edge_index[0], num_nodes).float()
        node_degree2 = degree(edge_index[1], num_nodes).float()
        node_degree = torch.round((node_degree1 + node_degree2) / 2).long()
        clipped_degree = torch.clip(node_degree, 0, size - 1)
        structural_features = F.one_hot(clipped_degree, size).float()

        return structural_features

    def calc_GDV(edge_index):
        gdv = GDV()
        structural_features = gdv.count5(edges=edge_index)
        sc = StandardScaler()
        structural_features = sc.fit_transform(structural_features)
        structural_features = torch.tensor(structural_features, dtype=torch.float32)

        return structural_features

    def calc_mp(edge_index, num_nodes, size=100, iteration=10):
        degree = Graph.calc_degree_features(edge_index, num_nodes, size)
        message_passing = MessagePassing(aggr="sum")
        sc = StandardScaler()

        x = degree
        mp = [x]
        for _ in range(iteration - 1):
            x = message_passing.propagate(edge_index, x=x)
            y = sc.fit_transform(x.numpy())
            mp.append(torch.tensor(y))

        mp = torch.sum(torch.stack(mp), dim=0)
        return mp

    def calc_fedStar(edge_index, num_nodes, size=100):
        SE_rw = create_rw(edge_index, num_nodes, config.structure_model.rw_len)
        SE_dg = Graph.calc_degree_features(
            edge_index, num_nodes, size - config.structure_model.rw_len
        )
        SE_rw_dg = torch.cat([SE_rw, SE_dg], dim=1)

        return SE_rw_dg

    def initialize_random_features(size):
        return torch.normal(0, 0.05, size=size, requires_grad=True, device=dev)

        # return torch.full(fill_value=0.05, size=size, requires_grad=True, device=dev)

    def reset_parameters(self) -> None:
        if config.structure_model.structure_type == "hop2vec":
            self.structural_features = Graph.initialize_random_features(
                size=self.structural_features.shape
            )

    def find_neighbors(self, node_id, include_node=False, include_external=False):
        if include_external:
            edges = torch.concat((self.get_edges(), self.inter_edges), dim=1)
        else:
            edges = self.get_edges()

        return find_neighbors_(
            node_id=node_id,
            edge_index=edges,
            include_node=include_node,
        )

    def create_L(self, normalization="normal", self_loop=False):
        nodes = self.node_ids

        num_nodes = self.x.shape[0]
        intra_edges = self.original_edge_index
        inter_edges = self.inter_edges
        if inter_edges is not None:
            edges = torch.concat((intra_edges, inter_edges), dim=1)
        else:
            edges = intra_edges

        A = create_adj(
            edges,
            normalization=normalization,
            self_loop=self_loop,
            num_nodes=self.x.shape[0],
            nodes=nodes,
        )

        if normalization == "normal":
            deg = torch.sum(A, dim=1).to_dense()
            D = sparse_eye(num_nodes, deg, edges.device)
            self.L = D - A
        else:
            I = sparse_eye(num_nodes, dev_=edges.device)
            self.L = I - A
        # self.L2 = self.L2.coalesce()

    def calc_abar(
        self,
        num_layers=config.structure_model.DGCN_layers,
        method="DGCN",
        pruning=False,
        spectral_len=0,
        log=True,
    ):
        if method in ["DGCN", "CentralDGCN"]:
            if self.DGCN_abar is not None:
                abar = self.DGCN_abar
            else:
                A = create_adj(
                    self.edge_index,
                    normalization="rw",
                    self_loop=True,
                    num_nodes=self.num_nodes,
                    nodes=self.node_ids,
                )
                # abar = calc_a2(edge_index, num_nodes, num_layers)
                abar = calc_abar(A, num_layers, pruning)
                self.DGCN_abar = abar
        elif method in ["SpectralDGCN", "LanczosDGCN"]:
            D, U = self.calc_eignvalues(
                estimate=not (method.startswith("Spectral")), spectral_len=spectral_len
            )
            Dbar = D**num_layers
            abar = U @ torch.diag(Dbar)
            # abar = U @ torch.diag(Dbar) @ U.T
            abar = abar.float().to_sparse()
            # abar = calc_a2(A, num_layers, method)
        if method == "random_walk":
            abar = estimate_abar(self.edge_index, self.num_nodes, num_layers)

        return abar

    def procrustes_project(
        self,
        from_U: torch.Tensor,
        to_U: torch.Tensor,
    ):
        """
        Solves the Orthogonal Procrustes problem: find best orthogonal matrix R
        s.t. |to_U - from_U @ R|_F is minimized.
        Returns the aligned from_U (i.e., from_U @ R).
        """

        if from_U.device != to_U.device:
            to_U = to_U.to(from_U.device)
        M = torch.matmul(from_U.t(), to_U)
        u, s, v = torch.svd(M)
        R = torch.matmul(u, v.t())
        return torch.matmul(from_U, R)

    def update_eigpairs(
        self,
        prev_V: torch.Tensor,
        prev_D: torch.Tensor,
        self_loop=True,
    ):
        self.create_L(
            normalization=config.spectral.L_type,
            self_loop=self_loop,
        )
        assert self.L is not None
        next_H = prev_V.T @ self.L @ prev_V
        next_D, next_U = torch.linalg.eigh(next_H)
        next_U = prev_V @ next_U

        return next_D, next_U

    def calc_eignvalues(self, estimate=False, self_loop=True, log=True, spectral_len=0):
        if config.spectral.matrix == "lap":
            self.create_L(
                normalization=config.spectral.L_type,
                self_loop=self_loop,
            )
            if estimate:
                D, U, V = estimate_eigh(
                    self.L,
                    config.spectral.lanczos_iter,
                    method=config.spectral.method,
                    log=log,
                )
            else:
                if config.spectral.decompose == "svd":
                    L = sparse.csr_matrix(self.L.to_dense().cpu().numpy())
                    if spectral_len <= 0:
                        k = min(L.shape) - 1
                    else:
                        k = spectral_len
                    U, D, V = sp.sparse.linalg.svds(L, k=k)
                    U = torch.tensor(U.copy(), dtype=torch.float32, device=dev)
                    D = torch.tensor(D.copy(), dtype=torch.float32, device=dev)
                    # U, D, V = torch.svd(self.L.to_dense())
                    # U, D, V = torch.svd_lowrank(self.L, q=spectral_len)
                else:
                    D, U = torch.linalg.eigh(self.L.to_dense())
        elif config.spectral.matrix == "adj":
            A = create_adj(
                self.edge_index,
                normalization=config.spectral.L_type,
                self_loop=self_loop,
                num_nodes=self.num_nodes,
                nodes=self.node_ids,
            )
            if estimate:
                D, U, _ = estimate_eigh(
                    A,
                    # A @ A.T,
                    config.spectral.lanczos_iter,
                    method=config.spectral.method,
                    log=log,
                )
            else:
                if config.spectral.decompose == "svd":
                    L = sparse.csr_matrix(A.to_dense().cpu().numpy())
                    if spectral_len <= 0:
                        k = min(A.shape) - 1
                    else:
                        k = spectral_len
                    U, D, V = sp.sparse.linalg.svds(L, k=k)
                    U = torch.tensor(U.copy(), dtype=torch.float32, device=dev)
                    D = torch.tensor(D.copy(), dtype=torch.float32, device=dev)
                    # U, D, V = torch.svd(A.to_dense())
                    # U, D, V = torch.svd_lowrank(
                    #     A, q=spectral_len, niter=5
                    # )
                else:
                    D, U = torch.linalg.eigh(A.to_dense())
                    # D2, U2 = torch.linalg.eigh(A.T.to_dense())
                    # D = torch.hstack([D1, D2])
                    # U = torch.hstack([U1, U2])
            D = -D
            # DD = degree(self.edge_index[0], self.num_nodes)
            # A1 = torch.diag(DD) @ A
            # # plt.figure()
            # # plot_abar(1 - A1, self.edge_index, name="A")
            # plt.figure()
            # pos = plot_graph(self.edge_index, self.num_nodes, self.num_classes, self.y)
            # plt.axis("off")
            # plt.tight_layout()

        elif config.spectral.matrix == "inc":
            E = create_inc(
                self.edge_index,
                normalization=config.spectral.L_type,
                num_nodes=self.num_nodes,
                nodes=self.node_ids,
            )
            if estimate:
                D, U, _ = estimate_eigh(
                    E @ E.T,
                    config.spectral.lanczos_iter,
                    method=config.spectral.method,
                    log=log,
                )
            else:
                U, D, V = torch.svd(E.to_dense())
                # U, D, V = torch.svd_lowrank(E, q=spectral_len)

        if len(D.shape) == 1:
            shift = 0
            if spectral_len > 0:
                sorted_eignvals = torch.sort(D, descending=False)
                sorted_indices = sorted_eignvals[1]
                sorted_indices = sorted_indices[shift : shift + spectral_len]

                U = U[:, sorted_indices]
                D = D[sorted_indices]
                if V is not None:
                    V = V[:, sorted_indices]
                # self.V_t = self.V_t[:, sorted_indices]
        # elif len(D.shape) == 2:
        #     if spectral_len > 0:
        #         sorted_eignvals = torch.sort(torch.diagonal(D), descending=False)
        #         sorted_indices = sorted_eignvals[1]
        #         sorted_indices = sorted_indices[: spectral_len]

        # U = U[:, sorted_indices]
        # D = D[sorted_indices, sorted_indices]

        # ss = torch.sign(torch.sum(torch.sign(U), dim=0))
        ss = torch.sign(torch.sum(U, dim=0))
        # ii = torch.argmax(torch.abs(U), dim=0)
        # ss = []
        # for ind, uu in enumerate(ii):
        #     ss.append(torch.sign(U[uu, ind]))
        # ss = torch.tensor(ss)
        # ss = torch.sign(U[0, :])
        U = torch.einsum("i,ji->ji", ss, U)

        if V is not None:
            V = torch.einsum("i,ji->ji", ss, V)

        # AA = torch.diag(DD) @ U @ torch.diag(-D) @ U.T
        # AA[AA > 1] = 1
        # AA[AA < 0] = 0
        # # plt.figure()
        # # plot_abar(1 - AA, self.edge_index, name="AA")
        # rows, cols = torch.where(AA > 0.5)
        # edge_index2 = torch.vstack((rows, cols))
        # edge_index2 = remove_self_loops(edge_index2)[0]
        # plt.figure()
        # plot_graph(edge_index2, self.num_nodes, self.num_classes, self.y)
        # plt.axis("off")
        # plt.tight_layout()
        # plt.figure()
        # plot_graph(edge_index2, self.num_nodes, self.num_classes, self.y, pos=pos)
        # plt.axis("off")
        # plt.tight_layout()
        # plt.show()

        return D, U, V

    def _random_walk(
        self, nx_G, start_node: int, walk_len: int, p: float, q: float
    ) -> list[int]:
        """
        Simulate a single random walk starting from start_node.
        For p=q=1.0, this is equivalent to DeepWalk (weighted random walk based on edge weights).
        """
        walk = [start_node]

        while len(walk) < walk_len:
            cur = walk[-1]
            neighbors = list(nx_G.neighbors(cur))

            if len(neighbors) == 0:
                break

            if len(walk) == 1:
                weights = [nx_G[cur][nbr].get("edge_weight", 1.0) for nbr in neighbors]
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]
                next_node = np.random.choice(neighbors, p=probs)
            else:
                prev = walk[-2]
                next_node = self._node2vec_step(nx_G, prev, cur, neighbors, p, q)

            walk.append(next_node)

        return walk

    def _node2vec_step(
        self, nx_G, prev: int, cur: int, neighbors: list[int], p: float, q: float
    ) -> int:
        """
        Choose next node using node2vec transition probabilities.
        With p=q=1.0, this reduces to uniform random walk (DeepWalk).

        Args:
            nx_G: NetworkX graph
            prev: Previous node (local index)
            cur: Current node (local index)
            neighbors: List of neighbor nodes (local indices)
            p: Return parameter (node2vec)
            q: In-out parameter (node2vec)

        Returns:
            Next node index (local)
        """
        if p == 1.0 and q == 1.0:
            weights = [nx_G[cur][nbr].get("edge_weight", 1.0) for nbr in neighbors]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            return np.random.choice(neighbors, p=probs)

        probs = []
        for neighbor in neighbors:
            if neighbor == prev:
                # Return to previous node
                prob = nx_G[cur][neighbor].get("edge_weight", 1.0) / p
            elif nx_G.has_edge(neighbor, prev):
                # Common neighbor (triangle)
                prob = nx_G[cur][neighbor].get("edge_weight", 1.0)
            else:
                # New node
                prob = nx_G[cur][neighbor].get("edge_weight", 1.0) / q
            probs.append(prob)

        # Normalize probabilities
        total = sum(probs)
        probs = [prob / total for prob in probs]

        # Sample based on probabilities
        return np.random.choice(neighbors, p=probs)

    def generate_context_pairs(
        self,
        num_walks: int = 10,
        walk_len: int = 40,
        window_size: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        use_original_node_ids: bool = False,
    ) -> dict[int, list[int]]:
        """
        Generate context pairs using random walk sampling (DySAT-style).

        Args:
            num_walks: Number of walks per node
            walk_len: Length of each walk
            window_size: Context window size for generating pairs
            p: Return parameter (node2vec), 1.0 = DeepWalk
            q: In-out parameter (node2vec), 1.0 = DeepWalk
            use_original_node_ids: If True, return pairs using original node_ids.
                                 If False, return pairs using local indices (default).

        Returns:
            Dictionary mapping node_id -> list of context node_ids.
            If use_original_node_ids=True, uses original node_ids.
            If use_original_node_ids=False, uses local indices.
        """
        edge_weight = getattr(self, "edge_weight", None)
        edge_weight = (
            getattr(self, "edge_attr", None) if edge_weight is None else edge_weight
        )
        pyg_data = PyGData(
            x=self.x, edge_index=self.edge_index, edge_weight=edge_weight
        )

        nx_G = to_networkx(pyg_data, edge_attrs=["edge_weight"], to_undirected=True)

        for edge in nx_G.edges():
            if "edge_weight" not in nx_G[edge[0]][edge[1]]:
                nx_G[edge[0]][edge[1]]["edge_weight"] = 1.0

        walks = []
        nodes = np.array(list(nx_G.nodes()))

        for _ in range(num_walks):
            np.random.shuffle(nodes)
            for start_node in nodes:
                walk = self._random_walk(nx_G, start_node, walk_len, p, q)
                walks.append(walk)

        pairs_local: dict[int, list[int]] = defaultdict(list)

        for walk in walks:
            for word_index, word in enumerate(walk):
                # Context window: nodes within window_size distance
                start_idx = max(word_index - window_size, 0)
                end_idx = min(word_index + window_size, len(walk)) + 1

                for nb_word in walk[start_idx:end_idx]:
                    if nb_word != word:
                        pairs_local[word].append(nb_word)

        # Optionally map from local indices to original node_ids
        if use_original_node_ids and self.node_ids is not None:
            pairs_original: dict[int, list[int]] = defaultdict(list)
            for local_idx, context_list in pairs_local.items():
                original_node_id = int(self.node_ids[local_idx].item())
                original_contexts = [
                    int(self.node_ids[ctx_idx].item()) for ctx_idx in context_list
                ]
                pairs_original[original_node_id] = original_contexts
            return dict(pairs_original)
        else:
            # Return pairs using local indices
            return dict(pairs_local)
