"""data_type=f+pe: input-side exact LapPE (commits e525928, bcc43af).

Covers the exact sym-Laplacian solver (Graph.calc_eigs_exact_sym), the
FedDynamicPEClassifier wiring, the serve-time sqrt(N) scaling in
DynamicServer._spectral_step, the stability-matched '*_fixed' basis controls,
the config gate, and non-regression of the feature / f+s paths.
"""

import copy

import pytest
import torch
from torch_geometric.data import Data

import src
from config.assertions import assert_cfg
from src.GNN.dynamic_classifier import DynamicClassifier
from src.GNN.fed_dynamic_classifier import FedDynamicPEClassifier
from src.dynamic_server import DynamicServer
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots

DROP_TOL = 1e-8


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


def make_graph(edges, num_nodes):
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return Graph(
        x=torch.ones(num_nodes, 1),
        edge_index=edge_index,
        node_ids=torch.arange(num_nodes),
    )


def lsym_dense(edges, num_nodes):
    """Reference I - D^-1/2 A D^-1/2 over the undirected edge set."""
    A = torch.zeros(num_nodes, num_nodes)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    deg = A.sum(dim=1)
    inv_sqrt = torch.where(deg > 0, deg.clamp(min=1e-12).pow(-0.5), torch.zeros_like(deg))
    return torch.eye(num_nodes) - torch.diag(inv_sqrt) @ A @ torch.diag(inv_sqrt)


TWO_TRIANGLES = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]
PATH_7 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


def make_toy_snapshots(N=12, W=1, num_snaps=3, seed=7):
    g = torch.Generator().manual_seed(seed)
    snaps = []
    for _ in range(num_snaps):
        edges = set()
        while len(edges) < 2 * N:
            u = int(torch.randint(0, N, (1,), generator=g))
            v = int(torch.randint(0, N, (1,), generator=g))
            if u != v:
                edges.add((u, v))
        edge_index = torch.tensor(sorted(edges), dtype=torch.long).t()
        snap = Data(
            x=torch.ones(N, 1),
            edge_index=edge_index,
            edge_attr=torch.randn(edge_index.size(1), W, generator=g),
            num_nodes=N,
        )
        snap.node_ids = torch.arange(N)
        snaps.append(snap)
    return snaps


def set_pe_model_config(config, pe_dim=4, dims_pre_mp=None):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "f+pe"
    config["model"]["edge_decoding"] = "concat"
    config["model"]["loss_fun"] = "bce_with_logits"
    config["gnn"]["dims"] = [8]
    config["gnn"]["dims_pre_mp"] = [] if dims_pre_mp is None else dims_pre_mp
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = False
    config["spectral"]["pe_dim"] = pe_dim
    config["spectral"]["update_mode"] = "keep"
    config["spectral"]["basis_source"] = "laplacian"
    config["spectral"]["use_procrustes"] = False
    config["seed"] = 42


# ---- B1-B4: Graph.calc_eigs_exact_sym ---- #


def test_exact_sym_solver_diagonalizes_the_laplacian(config):
    k = 5
    graph = make_graph(TWO_TRIANGLES, 6)

    D, U = graph.calc_eigs_exact_sym(k)

    assert D.shape == (k,)
    assert U.shape == (6, k)
    assert (D >= 0).all()
    assert torch.all(D[1:] >= D[:-1])          # ascending
    assert (D > DROP_TOL).all()                 # the trivial pair was dropped
    assert torch.allclose(U.norm(dim=0), torch.ones(k), atol=1e-5)
    L = lsym_dense(TWO_TRIANGLES, 6)
    assert torch.allclose(U.T @ L @ U, torch.diag(D), atol=1e-5)


def test_isolated_nodes_do_not_perturb_the_spectrum(config):
    # bcc43af: the solve runs on the ACTIVE subgraph, so padding the graph with
    # unseen nodes (every early cumulative snapshot) must be a no-op.
    k = 5
    small = make_graph(TWO_TRIANGLES, 6)
    padded = make_graph(TWO_TRIANGLES, 106)

    D_small, U_small = small.calc_eigs_exact_sym(k)
    D_padded, U_padded = padded.calc_eigs_exact_sym(k)

    assert U_padded.shape == (106, k)
    assert torch.allclose(D_small, D_padded, atol=1e-6)
    assert torch.allclose(U_small.abs(), U_padded[:6].abs(), atol=1e-6)
    assert U_padded[6:].abs().max() == 0.0


def test_short_spectrum_is_zero_padded_to_k(config):
    # one edge -> a single informative pair; the rest must be exact zeros so the
    # PE width stays fixed at k across snapshots.
    k = 5
    graph = make_graph([(0, 1)], 6)

    D, U = graph.calc_eigs_exact_sym(k)

    assert U.shape == (6, k)
    assert D.shape == (k,)
    assert D[0] > DROP_TOL
    assert torch.equal(D[1:], torch.zeros(k - 1))
    assert U[:, 1:].abs().max() == 0.0


@pytest.mark.parametrize("edges,n,k", [(PATH_7, 7, 6), (TWO_TRIANGLES, 6, 5)])
def test_sign_is_canonicalized_by_the_largest_entry(config, edges, n, k):
    # Well-defined only where the largest |entry| of a column is unique. Columns
    # from a degenerate eigenspace (TWO_TRIANGLES has |entry| ties, e.g. the
    # 1.5-eigenvalue pair) have no canonical gauge to fix, so they are skipped.
    _, U = make_graph(edges, n).calc_eigs_exact_sym(k)

    for j in range(k):
        col = U[:, j]
        if col.abs().max() == 0.0:
            continue
        at_max = (col.abs() == col.abs().max()).sum()
        if at_max > 1:
            continue
        assert col[col.abs().argmax()] > 0


# ---- B5-B6: FedDynamicPEClassifier wiring ---- #


def test_pe_classifier_widens_the_input_and_guards_the_pe(config):
    set_pe_model_config(config, pe_dim=4, dims_pre_mp=[16])
    N, num_features = 12, 3
    graph = Graph(
        x=torch.ones(N, num_features),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.randn(3, 1),
        node_ids=torch.arange(N),
    )

    clf = FedDynamicPEClassifier(graph)

    assert clf.input_dim() == num_features + 4
    assert clf.model.models[0].net[0].lin.in_features == num_features + 4

    # encode before the server ever served a PE: assert, not a shape blowup
    with pytest.raises(AssertionError) as exc:
        clf.encode()
    assert "set_QD" in str(exc.value)

    # a wrong-length slice (client/global mix-up) is caught, not silently used
    clf.set_QD(torch.randn(N + 5, 4), torch.randn(4))
    with pytest.raises(AssertionError) as exc:
        clf.encode()
    assert "slice/order mismatch" in str(exc.value)

    clf.set_QD(torch.randn(N, 4), torch.randn(4))
    z, _ = clf.encode()
    assert z.shape == (N, 8)


def flat_keys(sd, prefix=""):
    out = set()
    for key, val in sd.items():
        if isinstance(val, dict):
            out |= flat_keys(val, f"{prefix}{key}/")
        else:
            out.add(f"{prefix}{key}")
    return out


def test_pe_state_dict_has_no_spectral_entries(config):
    # The PE is SERVED, never learned or federated: f+pe's federated payload
    # must be byte-for-byte the plain fmodel protocol.
    set_pe_model_config(config, pe_dim=4)
    N = 12
    graph = Graph(
        x=torch.ones(N, 1),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.randn(3, 1),
        node_ids=torch.arange(N),
    )

    pe_clf = FedDynamicPEClassifier(graph)
    config["model"]["data_type"] = "feature"
    plain_clf = DynamicClassifier(graph)

    pe_keys = flat_keys(pe_clf.state_dict())
    assert pe_keys == flat_keys(plain_clf.state_dict())
    assert not [k for k in pe_keys if "smodel" in k or "SFV" in k or "PE" in k]

    served = torch.randn(N, 4)
    pe_clf.set_QD(served, torch.randn(4))
    pe_clf.load_state_dict(pe_clf.state_dict())
    assert torch.equal(pe_clf.PE, served.to(pe_clf.PE.device))


# ---- B7: stability-matched '*_fixed' controls ---- #


@pytest.mark.parametrize(
    "source,snapshot_independent",
    [
        ("random", False),
        ("shuffled", False),
        ("random_fixed", True),
        ("shuffled_fixed", True),
    ],
)
def test_fixed_basis_sources_are_snapshot_independent(config, source, snapshot_independent):
    # '*_fixed' seeds on the run seed only: matched numerics AND matched temporal
    # drift, so a difference vs the real basis can only be structure.
    config["spectral"]["basis_source"] = source
    config["seed"] = 42
    server = DynamicServer(make_toy_snapshots(num_snaps=2))
    U, _ = torch.linalg.qr(torch.randn(20, 5, generator=torch.Generator().manual_seed(1)))

    at_0, _ = server._substitute_basis(U, None, 0)
    at_5, _ = server._substitute_basis(U, None, 5)

    assert torch.equal(at_0, at_5) is snapshot_independent


# ---- B8: serve-time sqrt(N) scaling ---- #


def test_serve_time_sqrt_n_scaling_never_touches_the_cache(config):
    set_pe_model_config(config, pe_dim=4)
    N = 12
    snaps = make_toy_snapshots(N=N, num_snaps=3)
    server = DynamicServer(snaps)
    for client_snaps in partition_snapshots(snaps, 2):
        server.add_client(client_snaps)
    server.initialize_FL()

    # what the solver sees at t=0: the cumulative undirected union
    edge_index = snaps[0].edge_index.cpu()
    edge_index = torch.unique(
        torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1
    )
    _, U_ref = Graph(
        x=torch.ones(N, 1), edge_index=edge_index, node_ids=torch.arange(N)
    ).calc_eigs_exact_sym(4)
    scaled = U_ref * (N**0.5)

    server._spectral_step(0, "LanczosLaplace")
    pe_at_0 = server.classifier.PE.detach().cpu().clone()

    assert torch.allclose(pe_at_0, scaled, atol=1e-5)
    assert torch.allclose(server._first_spectral.U, U_ref, atol=1e-6)   # UNSCALED
    assert torch.allclose(server._prev_spectral.U, U_ref, atol=1e-6)    # UNSCALED
    for client in server.clients:
        node_ids = client.snaps[0].node_ids.cpu()
        assert torch.allclose(
            client.classifier.PE.detach().cpu(), scaled[node_ids], atol=1e-5
        )

    # serving again under keep must not compound the scale
    server._spectral_step(1, "LanczosLaplace")
    assert torch.allclose(server.classifier.PE.detach().cpu(), pe_at_0, atol=1e-6)
    assert torch.allclose(server._prev_spectral.U, U_ref, atol=1e-6)


# ---- B9: config gate ---- #


def test_assert_cfg_rules_for_fpe(config):
    config["model"]["data_type"] = "f+pe"
    config["spectral"]["update_mode"] = None
    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.update_mode is None" in str(exc.value)

    config["spectral"]["update_mode"] = "keep"
    assert_cfg(config)  # f+pe is an accepted data_type

    config["spectral"]["pe_dim"] = 0
    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.pe_dim must be >0" in str(exc.value)

    # pe_dim is only load-bearing for f+pe
    config["model"]["data_type"] = "f+s"
    assert_cfg(config)

    config["model"]["data_type"] = "f+pe"
    config["spectral"]["pe_dim"] = 50
    for source in ("laplacian", "random", "shuffled", "random_fixed", "shuffled_fixed"):
        config["spectral"]["basis_source"] = source
        assert_cfg(config)


# ---- B10: the feature / f+s paths are untouched ---- #


def test_feature_input_width_is_unchanged(config):
    set_pe_model_config(config, pe_dim=4, dims_pre_mp=[16])
    config["model"]["data_type"] = "feature"
    num_features = 3
    graph = Graph(
        x=torch.ones(12, num_features),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.randn(3, 1),
        node_ids=torch.arange(12),
    )

    clf = DynamicClassifier(graph)

    assert clf.input_dim() == num_features
    assert clf.model.models[0].net[0].lin.in_features == num_features


def test_spectral_width_is_pe_dim_only_for_fpe(config):
    set_pe_model_config(config, pe_dim=4)
    config["spectral"]["spectral_len"] = 17
    snaps = make_toy_snapshots(num_snaps=2)
    server = DynamicServer(snaps)
    server.add_client(partition_snapshots(snaps, 1)[0])

    seen = {}
    real = DynamicServer.get_spectral_features

    def spy(self, graph, smodel_type, ss_idx, spectral_len, mode, prev, first):
        seen["smodel_type"] = smodel_type
        seen["k"] = spectral_len
        return {}, None

    DynamicServer.get_spectral_features = spy
    try:
        config["model"]["data_type"] = "f+pe"
        server._spectral_step(0, "LanczosLaplace")
        assert seen == {"smodel_type": "ExactPE", "k": 4}

        config["model"]["data_type"] = "f+s"
        server._spectral_step(0, "LanczosLaplace")
        assert seen == {"smodel_type": "LanczosLaplace", "k": 17}
    finally:
        DynamicServer.get_spectral_features = real
