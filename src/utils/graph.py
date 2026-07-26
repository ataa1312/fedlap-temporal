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
try:
    from torch_sparse import SparseTensor
except (ImportError, OSError):
    class SparseTensor:  # torch_sparse ABI-incompatible with torch>=2.9 on Linux; used here only as a type hint
        pass
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

dataset_name = config["dataset"]["name"]


def _cheb_lowpass_coeffs(cutoff, degree, jackson=True):
    """Chebyshev coefficients of the ideal low-pass h(lambda) = 1 for
    lambda <= cutoff, 0 above, over L_sym's eigenvalue range [0, 2].

    Chebyshev polynomials live on [-1, 1], so lambda is shifted by -1. With
    lambda = 1 + cos(theta) the coefficients of a step have a closed form
    (c_0 = 1 - theta_a/pi, c_j = -2 sin(j theta_a) / (pi j)). The Jackson
    kernel damps them, trading a slightly wider transition band for the removal
    of the Gibbs ringing that would otherwise let high-frequency modes leak
    through with negative gain."""
    a = float(np.clip(cutoff - 1.0, -1.0, 1.0))
    theta = np.arccos(a)
    j = np.arange(degree + 1)
    c = np.empty(degree + 1)
    c[0] = 1.0 - theta / np.pi
    c[1:] = -2.0 * np.sin(j[1:] * theta) / (np.pi * j[1:])
    if jackson:
        N = degree + 1
        alpha = np.pi / (N + 1)
        c = c * (
            ((N - j + 1) * np.cos(j * alpha) + np.sin(j * alpha) / np.tan(alpha)) / (N + 1)
        )
    return c


def _cheb_filter(Lsym, X, coef):
    """Evaluate sum_j coef[j] T_j(L_sym - I) applied to the block X via the
    three-term recurrence T_{j+1} = 2 (L-I) T_j - T_{j-1}: one sparse matvec
    per degree, two blocks of memory, and the filter matrix is never formed."""
    T0 = X
    Y = coef[0] * T0
    if len(coef) > 1:
        T1 = Lsym @ X - X
        Y = Y + coef[1] * T1
        for c in coef[2:]:
            T2 = 2.0 * (Lsym @ T1 - T1) - T0
            Y = Y + c * T2
            T0, T1 = T1, T2
    return Y


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
                iteration=config["structure_model"]["num_mp_vectors"],
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
        SE_rw = create_rw(edge_index, num_nodes, config["structure_model"]["rw_len"])
        SE_dg = Graph.calc_degree_features(
            edge_index, num_nodes, size - config["structure_model"]["rw_len"]
        )
        SE_rw_dg = torch.cat([SE_rw, SE_dg], dim=1)

        return SE_rw_dg

    def initialize_random_features(size):
        return torch.normal(0, 0.05, size=size, requires_grad=True, device=dev)

        # return torch.full(fill_value=0.05, size=size, requires_grad=True, device=dev)

    def reset_parameters(self) -> None:
        if config["structure_model"]["structure_type"] == "hop2vec":
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
        num_layers=config["structure_model"]["DGCN_layers"],
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
        try:
            u, s, vh = torch.linalg.svd(M, full_matrices=False)
        except torch.linalg.LinAlgError:
            # ill-conditioned alignment matrix (near-degenerate spectrum): skip
            # the rotation rather than crash — leaves from_U unaligned this step.
            return from_U
        R = torch.matmul(u, vh)
        return torch.matmul(from_U, R)

    def update_eigpairs(
        self,
        prev_Q: torch.Tensor,
        self_loop=True,
    ):
        self.create_L(
            normalization=config["spectral"]["L_type"],
            self_loop=self_loop,
        )
        assert self.L is not None
        prev_Q = prev_Q.to(self.L.device)  # L is CPU-built; keep the projection on one device
        next_H = prev_Q.T @ self.L @ prev_Q
        # Rayleigh-Ritz. torch.linalg.eigh reads ONE triangle and assumes symmetry, so it
        # is only valid when H is. That holds for L_type in {normal, sym} but NOT for the
        # default 'rw': L_rw = I - D^-1 A is non-symmetric (it is merely *similar* to the
        # symmetric L_sym), so H inherits the asymmetry and eigh silently returns the
        # spectrum of (H+H^T)/2 instead. Fall back to the general solver there.
        # CAVEAT: L_rw's own spectrum is real (via the similarity to L_sym), but the
        # PROJECTION Q^T L_rw Q does not inherit that -- it can and does pick up complex
        # eigenvalues (measured max|Im| ~ 2e-2 on a toy graph, with Ritz residuals ~1e-2).
        # Taking the real part is therefore a pragmatic fallback, NOT an exact identity;
        # the warning below fires when the imaginary part is material. Under L_type=sym
        # this branch is not taken at all and the tracking is exact (residual ~1e-7), which
        # is an argument for 'sym' in `update` mode specifically, independent of MRR.
        # Sorting ascending preserves eigh's ordering contract, which keeps the tracked
        # columns in a consistent order from snapshot to snapshot.
        if torch.allclose(next_H, next_H.transpose(-1, -2), atol=1e-6, rtol=0):
            next_D, next_V = torch.linalg.eigh(next_H)
        else:
            D_c, V_c = torch.linalg.eig(next_H)
            if D_c.imag.abs().max() > 1e-4 * max(D_c.real.abs().max().item(), 1.0):
                LOGGER.warning(
                    "update_eigpairs: projected H has complex spectrum "
                    f"(max|Im|={D_c.imag.abs().max().item():.3e}); taking the real part"
                )
            order = torch.argsort(D_c.real)
            next_D, next_V = D_c.real[order], V_c.real[:, order]
        next_U = prev_Q @ next_V

        return next_D, next_U, prev_Q

    def _active_lsym(self):
        """L_sym = I - D^-1/2 A D^-1/2 restricted to the largest connected
        component of the ACTIVE subgraph, with the global node indices it
        covers. Isolated nodes and satellite components contribute only
        exact-zero eigenpairs (component indicators): on early cumulative
        graphs they stall ARPACK, and on many-component graphs (reddit: up to
        16k components mid-run) they crowd out every informative pair. Both
        solvers drop them here and zero-pad the missing rows afterwards.
        Returns (None, None) when fewer than two active nodes remain."""
        n = self.num_nodes
        e = self.edge_index.cpu().numpy()
        A = sparse.coo_matrix(
            (np.ones(e.shape[1]), (e[0], e[1])), shape=(n, n)
        ).tocsr()
        A = ((A + A.T) > 0).astype(np.float64)
        A.setdiag(0)
        A.eliminate_zeros()
        deg = np.asarray(A.sum(axis=1)).ravel()
        act = np.where(deg > 0)[0]
        if act.size < 2:
            return None, None
        Aa = A[act][:, act]
        ncomp, labels = sp.sparse.csgraph.connected_components(Aa, directed=False)
        if ncomp > 1:
            giant = np.where(labels == np.bincount(labels).argmax())[0]
            act = act[giant]
            Aa = Aa[giant][:, giant]
        m = act.size
        dis = 1.0 / np.sqrt(np.asarray(Aa.sum(axis=1)).ravel())
        Dis = sparse.diags(dis)
        return sparse.eye(m) - Dis @ Aa @ Dis, act

    def calc_eigs_exact_sym(self, k, drop_tol=1e-8, dense_max=3000):
        """Exact k lowest NONTRIVIAL eigenpairs of the symmetric normalized
        Laplacian L_sym = I - D^-1/2 A D^-1/2 over this graph's undirected
        edges. The Krylov estimate (calc_eignvalues estimate=True) cannot
        resolve the clustered low end of the spectrum, which is where the
        structural signal lives — this solver exists for the input-PE path
        (model.data_type=f+pe) where that signal is the whole point.
        Near-zero eigenpairs (constant/component indicators, no pairwise
        information) are dropped; short results are zero-padded to k columns.
        Sign is canonicalized by each eigenvector's largest-|entry| element.
        Dense eigh below dense_max nodes, sparse shift-invert eigsh above."""
        n = self.num_nodes
        Lsym, act = self._active_lsym()
        if Lsym is None:
            return torch.zeros(k), torch.zeros(n, k)
        m = act.size

        if m <= dense_max:
            w, V = np.linalg.eigh(Lsym.toarray())
        else:
            # shift-invert about a point just below 0: L+|sigma|I is SPD, so the
            # factorization is safe, and 'LM' in shifted space = smallest eigs.
            req = min(k + 64, m - 2)
            try:
                w, V = sp.sparse.linalg.eigsh(
                    Lsym.tocsc(), k=req, sigma=-0.01, which="LM",
                    ncv=min(m - 1, max(4 * req, 2 * req + 1)),
                )
            except sp.sparse.linalg.ArpackError:
                if m > 12000:
                    raise
                w, V = np.linalg.eigh(Lsym.toarray())
        order = np.argsort(w)
        w, V = w[order], V[:, order]
        keep = w > drop_tol
        w, V = w[keep][:k], V[:, keep][:, :k]
        if V.shape[1] < k:  # early/tiny graphs: fewer informative pairs than k
            pad = k - V.shape[1]
            V = np.hstack([V, np.zeros((m, pad))])
            w = np.concatenate([w, np.zeros(pad)])
        ii = np.abs(V).argmax(axis=0)
        ss = np.sign(V[ii, np.arange(V.shape[1])])
        ss[ss == 0] = 1.0
        V = V * ss
        Vfull = np.zeros((n, k))
        Vfull[act] = V
        return (
            torch.tensor(w, dtype=torch.float32),
            torch.tensor(Vfull, dtype=torch.float32),
        )

    def calc_eigs_chebyshev(
        self, k, cutoff=None, degree=40, oversample=64, n_iter=3,
        X0=None, drop_tol=1e-8, seed=0,
    ):
        """k lowest eigenpairs by CHEBYSHEV-FILTERED SUBSPACE ITERATION.

        Motivation (measured, results.md §10.12): after the active/giant-
        component truncation the low spectrum of these graphs is not degenerate
        but heavily CLUSTERED — median gaps ~1e-3 with ~1% relative spacing on
        every dataset. Krylov methods must separate eigenvalues to converge, so
        the Arnoldi estimate returns blends (its basis reconstructs its own
        graph at AUC 0.53). A polynomial filter never separates anything: it
        multiplies each eigen-coordinate by p(lambda), so applying it to a block
        of vectors suppresses the high-frequency content and leaves a block
        spanning the low-frequency subspace, however crowded that subspace is.

        p is the Jackson-damped Chebyshev expansion of the ideal low-pass
        (1 below `cutoff`, 0 above) on L_sym's range [0, 2]; the three-term
        recurrence evaluates it with one sparse matvec per degree and no dense
        matrix ever formed. `X0` warm-starts the block with the previous
        snapshot's basis, which is what makes this a drop-in for the tracker.
        Returns the same (eigenvalues, zero-padded eigenvectors) contract as
        calc_eigs_exact_sym.

        `cutoff` must sit AT or just BELOW lambda_k (pass the previous
        snapshot's k-th eigenvalue when tracking; 0.9x is a safe factor).
        Raising it is counter-productive on these graphs: the spectrum is dense,
        so a higher cutoff admits far more than k modes and a block of width
        k+oversample can no longer span them — measured subspace overlap on uci
        falls 1.00 -> 0.48 -> 0.07 as the cutoff goes 1.0x -> 1.3x -> 2.0x
        lambda_k. Buy accuracy with `oversample`/`n_iter`, never with cutoff."""
        n = self.num_nodes
        Lsym, act = self._active_lsym()
        if Lsym is None:
            return torch.zeros(k), torch.zeros(n, k)
        m = act.size
        b = int(min(m - 1, k + oversample))
        rng = np.random.default_rng(seed)
        if X0 is not None:
            X = np.asarray(X0, dtype=np.float64)[act]
            if X.shape[1] < b:
                X = np.hstack([X, rng.standard_normal((m, b - X.shape[1]))])
            else:
                X = X[:, :b]
            if not np.isfinite(X).all() or np.linalg.norm(X) == 0:
                X = rng.standard_normal((m, b))
        else:
            X = rng.standard_normal((m, b))

        coef = _cheb_lowpass_coeffs(0.5 if cutoff is None else float(cutoff), degree)
        for _ in range(max(1, n_iter)):
            X = _cheb_filter(Lsym, X, coef)
            X, _ = np.linalg.qr(X)
        # Rayleigh-Ritz: diagonalise L_sym restricted to the filtered subspace
        B = X.T @ (Lsym @ X)
        w, W = np.linalg.eigh(0.5 * (B + B.T))
        V = X @ W
        keep = w > drop_tol
        w, V = w[keep][:k], V[:, keep][:, :k]
        if V.shape[1] < k:
            pad = k - V.shape[1]
            V = np.hstack([V, np.zeros((m, pad))])
            w = np.concatenate([w, np.zeros(pad)])
        ii = np.abs(V).argmax(axis=0)
        ss = np.sign(V[ii, np.arange(V.shape[1])])
        ss[ss == 0] = 1.0
        V = V * ss
        Vfull = np.zeros((n, k))
        Vfull[act] = V
        return (
            torch.tensor(w, dtype=torch.float32),
            torch.tensor(Vfull, dtype=torch.float32),
        )

    def calc_eignvalues(
        self, estimate=False, self_loop=True, log=True, spectral_len=0, canonicalize_sign=True
    ):
        V = None  # not every decomposition path binds V (the eigh path doesn't)
        if config["spectral"]["matrix"] == "lap":
            self.create_L(
                normalization=config["spectral"]["L_type"],
                self_loop=self_loop,
            )
            if estimate:
                D, U, V = estimate_eigh(
                    self.L,
                    config["spectral"]["lanczos_iter"],
                    method=config["spectral"]["method"],
                    log=log,
                )
            else:
                if config["spectral"]["decompose"] == "svd":
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
        elif config["spectral"]["matrix"] == "adj":
            A = create_adj(
                self.edge_index,
                normalization=config["spectral"]["L_type"],
                self_loop=self_loop,
                num_nodes=self.num_nodes,
                nodes=self.node_ids,
            )
            if estimate:
                D, U, _ = estimate_eigh(
                    A,
                    # A @ A.T,
                    config["spectral"]["lanczos_iter"],
                    method=config["spectral"]["method"],
                    log=log,
                )
            else:
                if config["spectral"]["decompose"] == "svd":
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

        elif config["spectral"]["matrix"] == "inc":
            E = create_inc(
                self.edge_index,
                normalization=config["spectral"]["L_type"],
                num_nodes=self.num_nodes,
                nodes=self.node_ids,
            )
            if estimate:
                D, U, _ = estimate_eigh(
                    E @ E.T,
                    config["spectral"]["lanczos_iter"],
                    method=config["spectral"]["method"],
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

        # Per-eigenvector sign canonicalization. SignNet is exactly invariant to
        # this transformation, so it is inert (and conceptually redundant) under it:
        # the caller passes canonicalize_sign=False for SignNet to leave the solver's
        # raw sign untouched. Procrustes (a rotation alignment, a different gauge) is
        # independent and still applies when enabled.
        if canonicalize_sign:
            if config["spectral"]["robust_sign"]:
                # canonicalize each eigenvector's sign by its LARGEST-|component| entry (well away from
                # zero) instead of the near-zero column-sum, so the sign is stable across snapshots.
                ii = torch.argmax(torch.abs(U), dim=0)
                ss = torch.sign(U[ii, torch.arange(U.shape[1], device=U.device)])
            else:
                ss = torch.sign(torch.sum(U, dim=0))
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

