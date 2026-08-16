"""spectral.solver plumbing in DynamicServer.

Which eigensolver actually runs decides what every spectral result measures, and
the choice is made in one if/elif chain inside get_spectral_features. These pin
the routing, the chebyshev warm start under update_mode=update, the _cheb_cutoff
helper, the ExactPE override, and -- most importantly -- that the basis_source
placebo still fires on the chebyshev branches, since a control arm that silently
stopped being substituted would invalidate every placebo-controlled result.
"""

import copy

import pytest
import torch
from torch_geometric.data import Data

import src
from src.dynamic_server import DynamicServer, SpectralFeatures, _cheb_cutoff
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots

EMPTY = SpectralFeatures(U=None, D=None, Q=None)
SOLVER_METHOD = {
    "exact": "calc_eigs_exact_sym",
    "chebyshev": "calc_eigs_chebyshev",
    "arnoldi": "calc_eignvalues",
    "track": "update_eigpairs",
}


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


@pytest.fixture
def solver_spy():
    """Record every eigensolver entry point, delegating to the real one."""
    calls = {name: [] for name in SOLVER_METHOD}
    original = {n: getattr(Graph, m) for n, m in SOLVER_METHOD.items()}

    def wrap(name):
        def spy(self, *args, **kwargs):
            calls[name].append((args, kwargs))
            return original[name](self, *args, **kwargs)

        return spy

    for name, method in SOLVER_METHOD.items():
        setattr(Graph, method, wrap(name))
    yield calls
    for name, method in SOLVER_METHOD.items():
        setattr(Graph, method, original[name])


def make_dense_snapshots(num_nodes=16, num_snaps=3, seed=5):
    g = torch.Generator().manual_seed(seed)
    edges = [(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    snaps = []
    for _ in range(num_snaps):
        snap = Data(
            x=torch.ones(num_nodes, 1),
            edge_index=edge_index.clone(),
            edge_attr=torch.randn(edge_index.size(1), 1, generator=g),
            num_nodes=num_nodes,
        )
        snap.node_ids = torch.arange(num_nodes)
        snaps.append(snap)
    return snaps


def make_dense_graph(num_nodes=16):
    edges = [(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v]
    return Graph(
        x=torch.ones(num_nodes, 1),
        edge_index=torch.tensor(edges, dtype=torch.long).t(),
        node_ids=torch.arange(num_nodes),
    )


def spectral_run_config(config, solver, update_mode, data_type="f+es", pe_dim=4):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = data_type
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [8, 8]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["structure_model"]["DGCN_structure_layers_sizes"] = [16]
    config["spectral"]["solver"] = solver
    config["spectral"]["update_mode"] = update_mode
    config["spectral"]["pe_dim"] = pe_dim
    config["spectral"]["use_procrustes"] = False
    config["spectral"]["lanczos_iter"] = 8
    config["spectral"]["recompute_prob"] = 0.0
    config["seed"] = 42


def build_server(config, num_subgraphs=1, num_snaps=3):
    snaps = make_dense_snapshots(num_snaps=num_snaps)
    server = DynamicServer(snaps)
    for client_snaps in partition_snapshots(snaps, num_subgraphs):
        server.add_client(client_snaps)
    return server


# --------------------------------------------------------------------- #
# 1. routing
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("update_mode", ["keep", "update", "recompute"])
@pytest.mark.parametrize("solver", ["exact", "chebyshev", "arnoldi"])
def test_solver_routing_on_the_fresh_solve_branch(config, solver_spy, solver, update_mode):
    # With no cached basis every update_mode lands on the fresh-solve branch, so
    # the routing is exactly the solver switch.
    spectral_run_config(config, solver, update_mode)
    server = DynamicServer(make_dense_snapshots(num_snaps=2))

    server.get_spectral_features(
        make_dense_graph(), "LanczosLaplace", 0, 4, update_mode, EMPTY, EMPTY
    )

    fired = {name for name, seen in solver_spy.items() if seen}
    assert fired == {solver if solver != "arnoldi" else "arnoldi"}
    if solver == "arnoldi":
        assert solver_spy["arnoldi"][0][1]["estimate"] is True


def test_keep_reuses_the_cache_without_calling_any_solver(config, solver_spy):
    spectral_run_config(config, "chebyshev", "keep")
    server = DynamicServer(make_dense_snapshots(num_snaps=2))
    graph = make_dense_graph()

    server.get_spectral_features(graph, "Invariant", 0, 4, "keep", EMPTY, EMPTY)
    after_first = {name: len(seen) for name, seen in solver_spy.items()}
    server.get_spectral_features(
        graph, "Invariant", 1, 4, "keep",
        server.get_previous_UD("keep", 1), server._first_spectral,
    )

    assert after_first["chebyshev"] == 1
    assert {name: len(seen) for name, seen in solver_spy.items()} == after_first


# --------------------------------------------------------------------- #
# 2. warm start under update_mode=update
# --------------------------------------------------------------------- #


def test_chebyshev_update_warm_starts_from_the_previous_basis(config, solver_spy):
    spectral_run_config(config, "chebyshev", "update")
    server = build_server(config)
    server.initialize_FL()

    server._spectral_step(0, "Invariant")
    prev_U = server._prev_spectral.U.clone()
    prev_D = server._prev_spectral.D.clone()
    server._spectral_step(1, "Invariant")

    assert len(solver_spy["chebyshev"]) == 2
    assert not solver_spy["track"], "update_eigpairs must not run on the chebyshev path"
    _, kwargs = solver_spy["chebyshev"][1]
    assert torch.allclose(torch.as_tensor(kwargs["X0"]), prev_U, atol=1e-6)
    assert kwargs["cutoff"] == pytest.approx(_cheb_cutoff(prev_D))
    assert kwargs["cutoff"] == pytest.approx(0.9 * float(prev_D[prev_D > 0].max()))
    # the first snapshot has no previous spectrum to derive a cutoff from
    assert solver_spy["chebyshev"][0][1]["cutoff"] is None


def test_arnoldi_update_still_tracks_with_update_eigpairs(config, solver_spy):
    spectral_run_config(config, "arnoldi", "update", data_type="f+s")
    server = build_server(config)
    server.initialize_FL()

    server._spectral_step(0, "LanczosLaplace")
    server._spectral_step(1, "LanczosLaplace")

    assert len(solver_spy["arnoldi"]) == 1   # fresh solve at t=0 only
    assert len(solver_spy["track"]) == 1     # t=1 tracked
    assert not solver_spy["chebyshev"]


# --------------------------------------------------------------------- #
# 3. _cheb_cutoff
# --------------------------------------------------------------------- #


def test_cheb_cutoff_rules(config):
    assert _cheb_cutoff(None) is None
    assert _cheb_cutoff(torch.zeros(5)) is None
    # short bases are zero-padded; the padding must not become the maximum
    assert _cheb_cutoff(torch.tensor([0.2, 0.5, 0.0, 0.0])) == pytest.approx(0.45)
    assert _cheb_cutoff(torch.tensor([0.2, 0.5, 0.9])) == pytest.approx(0.81)
    assert _cheb_cutoff(torch.tensor([1.0]), safety=0.5) == pytest.approx(0.5)
    # it is a plain float, ready to hand to the solver
    assert isinstance(_cheb_cutoff(torch.tensor([0.4])), float)


# --------------------------------------------------------------------- #
# 4. the basis_source placebo still fires on the chebyshev branches
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("update_mode", ["recompute", "update"])
def test_placebo_substitution_applies_on_the_chebyshev_branch(config, update_mode):
    # If _substitute_basis were skipped here, a 'shuffled_fixed' run would
    # silently serve the REAL basis and the control arm would become a second
    # copy of the treatment -- every placebo-controlled result depends on this.
    spectral_run_config(config, "chebyshev", update_mode)
    config["spectral"]["basis_source"] = "shuffled_fixed"
    server = build_server(config)
    server.initialize_FL()

    swaps = []
    real_substitute = DynamicServer._substitute_basis

    def spy(self, U, Q, ss_idx):
        out = real_substitute(self, U, Q, ss_idx)
        swaps.append((ss_idx, U.detach().clone(), out[0].detach().clone()))
        return out

    DynamicServer._substitute_basis = spy
    try:
        server._spectral_step(0, "Invariant")
        server._spectral_step(1, "Invariant")
    finally:
        DynamicServer._substitute_basis = real_substitute

    # fired on BOTH the fresh solve (t=0) and the warm-started solve (t=1)
    assert [ss for ss, _, _ in swaps] == [0, 1]
    for _, raw, served in swaps:
        assert not torch.allclose(raw, served)
        assert torch.allclose(
            torch.sort(raw.norm(dim=1)).values,
            torch.sort(served.norm(dim=1)).values,
            atol=1e-5,
        )


def test_laplacian_basis_source_leaves_the_chebyshev_output_untouched(config):
    spectral_run_config(config, "chebyshev", "update")
    config["spectral"]["basis_source"] = "laplacian"
    server = build_server(config)
    server.initialize_FL()

    identical = []
    real_substitute = DynamicServer._substitute_basis

    def spy(self, U, Q, ss_idx):
        out = real_substitute(self, U, Q, ss_idx)
        identical.append(out[0] is U)
        return out

    DynamicServer._substitute_basis = spy
    try:
        server._spectral_step(0, "Invariant")
        server._spectral_step(1, "Invariant")
    finally:
        DynamicServer._substitute_basis = real_substitute

    assert identical == [True, True]


# --------------------------------------------------------------------- #
# 5. ExactPE precedence
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("solver", ["chebyshev", "arnoldi", "exact"])
def test_fpe_always_uses_the_exact_solver(config, solver_spy, solver):
    # f+pe forces smodel_type=ExactPE, which wins over spectral.solver: the
    # input-PE path is defined as the exact low-k basis.
    spectral_run_config(config, solver, "recompute", data_type="f+pe")
    server = build_server(config, num_snaps=2)
    server.initialize_FL()

    server._spectral_step(0, "LanczosLaplace")

    assert len(solver_spy["exact"]) == 1
    assert not solver_spy["chebyshev"]
    assert not solver_spy["arnoldi"]
