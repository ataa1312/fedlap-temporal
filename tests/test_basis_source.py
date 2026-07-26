"""spectral.basis_source placebo controls (commit 1690824).

_substitute_basis swaps the Laplacian eigenbasis for a null basis of the same
shape. These pin the swap itself (identity / orthonormality / permutation /
determinism / Q coupling / dtype), the config gate, and the fact that the swap
happens on the fresh-solve path only -- so `keep` freezes the t=0 substitute
instead of redrawing one per snapshot.
"""

import copy

import pytest
import torch
from torch_geometric.data import Data

import src
from config.assertions import assert_cfg
from src.dynamic_server import DynamicServer, SpectralFeatures
from src.utils.graph import Graph


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


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


def make_server(N=12, num_snaps=3):
    return DynamicServer(make_toy_snapshots(N=N, num_snaps=num_snaps))


def ortho(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, k, generator=g))
    return q


def test_spectral_gauge_defaults(config):
    # basis_source defaults to the real basis (1690824); deterministic_start
    # flipped False -> True in 3291fce, so every fresh solve now starts from the
    # same Arnoldi vector. results.md 2-9 predate the flip.
    assert config["spectral"]["basis_source"] == "laplacian"
    assert config["spectral"]["deterministic_start"] is True


def test_laplacian_is_a_strict_no_op(config):
    # identity, not equality: the real path must not copy or re-wrap the tensors
    config["spectral"]["basis_source"] = "laplacian"
    server = make_server()
    U, Q = ortho(20, 5, 1), ortho(20, 5, 2)

    sub_U, sub_Q = server._substitute_basis(U, Q, 0)

    assert sub_U is U
    assert sub_Q is Q


def test_random_is_orthonormal_with_u_shape(config):
    config["spectral"]["basis_source"] = "random"
    config["seed"] = 42
    server = make_server()
    U = ortho(20, 5, 1)

    sub, _ = server._substitute_basis(U, None, 0)

    assert sub.shape == U.shape
    assert torch.allclose(sub.T @ sub, torch.eye(5), atol=1e-5)


def test_shuffled_is_a_row_permutation_of_u(config):
    config["spectral"]["basis_source"] = "shuffled"
    config["seed"] = 42
    server = make_server()
    U = ortho(20, 5, 1)

    sub, _ = server._substitute_basis(U, None, 0)

    # matched value distribution: the multiset of row norms survives
    assert torch.allclose(
        torch.sort(U.norm(dim=1)).values, torch.sort(sub.norm(dim=1)).values, atol=1e-6
    )
    # and every row of sub is exactly some row of U, each used once
    origin = [
        int(((U - sub[i]).abs().max(dim=1).values < 1e-7).nonzero()[0])
        for i in range(U.shape[0])
    ]
    assert sorted(origin) == list(range(U.shape[0]))
    # a non-degenerate U is genuinely moved (structure destroyed)
    assert not torch.allclose(sub, U)


@pytest.mark.parametrize("source", ["random", "shuffled"])
def test_substitution_is_deterministic_in_seed_and_ss_idx(config, source):
    config["spectral"]["basis_source"] = source
    config["seed"] = 42
    server = make_server()
    U = ortho(20, 5, 1)

    a, _ = server._substitute_basis(U, None, 0)
    b, _ = server._substitute_basis(U, None, 0)
    assert torch.equal(a, b)

    other_idx, _ = server._substitute_basis(U, None, 5)
    assert not torch.equal(a, other_idx)

    config["seed"] = 43
    other_seed, _ = server._substitute_basis(U, None, 0)
    assert not torch.equal(a, other_seed)


@pytest.mark.parametrize("source", ["random", "shuffled"])
def test_q_is_substituted_with_u_and_stays_none_when_none(config, source):
    # Q is the tracking basis for `update`; substituting only U would leave
    # `update` evolving the REAL subspace under a null control.
    config["spectral"]["basis_source"] = source
    config["seed"] = 42
    server = make_server()
    U, Q = ortho(20, 5, 1), ortho(20, 5, 2)

    sub_U, sub_Q = server._substitute_basis(U, Q, 0)
    assert sub_Q is sub_U

    _, none_Q = server._substitute_basis(U, None, 0)
    assert none_Q is None


@pytest.mark.parametrize("source", ["random", "shuffled"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_and_device_of_u_are_preserved(config, source, dtype):
    config["spectral"]["basis_source"] = source
    config["seed"] = 42
    server = make_server()
    U = ortho(20, 5, 1).to(dtype)

    sub, _ = server._substitute_basis(U, None, 0)

    assert sub.dtype == U.dtype
    assert sub.device == U.device


def test_assert_cfg_gates_basis_source(config):
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "keep"

    for source in ("laplacian", "random", "shuffled"):
        config["spectral"]["basis_source"] = source
        assert_cfg(config)

    config["spectral"]["basis_source"] = "nonsense"
    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.basis_source" in str(exc.value)

    # feature has no spectral path -> the key is not validated there
    config["model"]["data_type"] = "feature"
    assert_cfg(config)


def test_keep_freezes_the_t0_substitute(config):
    # The swap sits on the fresh-solve branch, so `keep` (which returns the
    # cached pair without re-solving) must serve the SAME substitute forever.
    # A per-snapshot redraw would make the control temporally unstable and
    # confound structure with stability.
    config["spectral"]["basis_source"] = "random"
    config["spectral"]["update_mode"] = "keep"
    config["spectral"]["use_procrustes"] = False
    config["seed"] = 42
    N, k = 12, 4
    server = make_server(N=N)

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 7]]
    )
    edge_index = torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1)
    graph = Graph(x=torch.ones(N, 1), edge_index=edge_index, node_ids=torch.arange(N))

    solves = []
    real_solver = Graph.calc_eigs_exact_sym

    def counting_solver(self, *args, **kwargs):
        solves.append(1)
        return real_solver(self, *args, **kwargs)

    Graph.calc_eigs_exact_sym = counting_solver
    try:
        empty = SpectralFeatures(U=None, D=None, Q=None)
        share_0, _ = server.get_spectral_features(
            graph, "ExactPE", 0, k, "keep", server.get_previous_UD("keep", 0), empty
        )
        share_1, _ = server.get_spectral_features(
            graph,
            "ExactPE",
            1,
            k,
            "keep",
            server.get_previous_UD("keep", 1),
            server._first_spectral,
        )
    finally:
        Graph.calc_eigs_exact_sym = real_solver

    assert len(solves) == 1  # t=1 reused the cache; it never re-solved
    assert torch.equal(share_0["U"].cpu(), share_1["U"].cpu())

    # and a fresh draw at ss_idx=1 really would have differed
    fresh, _ = server._substitute_basis(share_0["U"].cpu(), None, 1)
    assert not torch.equal(fresh, share_1["U"].cpu())
