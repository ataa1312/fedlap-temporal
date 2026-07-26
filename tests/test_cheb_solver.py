"""Chebyshev-filtered subspace iteration (src/utils/graph.py).

Covers the two module-level helpers (_cheb_lowpass_coeffs, _cheb_filter), the
newly extracted _active_lsym truncation shared by both solvers, and
calc_eigs_chebyshev itself -- its contract parity with calc_eigs_exact_sym, its
accuracy against that solver, and the warm-start / edge-case paths.
"""

import copy

import numpy as np
import pytest
import torch
from numpy.polynomial import chebyshev as chebmod
from scipy import sparse

import src
from src.utils.graph import Graph, _cheb_filter, _cheb_lowpass_coeffs

GRID = np.linspace(0.0, 2.0, 801)  # L_sym's eigenvalue range


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


def make_graph(edges, num_nodes):
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Graph(
        x=torch.ones(num_nodes, 1),
        edge_index=edge_index,
        node_ids=torch.arange(num_nodes),
    )


def make_sbm(n_per_block=30, p_in=0.35, p_out=0.02, seed=3):
    """Two-block SBM: a clean low-frequency structure with a well-separated
    Fiedler value and no repeated eigenvalues, so eigenvectors are unique up to
    sign and the subspace comparisons below are meaningful."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_block
    edges = [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < (p_in if (u < n_per_block) == (v < n_per_block) else p_out)
    ]
    return make_graph(edges, n)


# The pin graph: a 5-node giant component (5-cycle + one chord) on odd global
# ids, a 3-node satellite triangle on even ones, and two isolated nodes.
PIN_GIANT = [(1, 3), (3, 5), (5, 7), (7, 9), (9, 1), (1, 5)]
PIN_SATELLITE = [(2, 4), (4, 6), (6, 2)]
PIN_NODES = 12
PIN_ACTIVE = [1, 3, 5, 7, 9]


def dense_lsym(edges, node_subset):
    """Reference I - D^-1/2 A D^-1/2 over `node_subset`, built independently of
    anything in graph.py."""
    index = {g: i for i, g in enumerate(node_subset)}
    m = len(node_subset)
    A = np.zeros((m, m))
    for u, v in edges:
        if u in index and v in index:
            A[index[u], index[v]] = 1.0
            A[index[v], index[u]] = 1.0
    deg = A.sum(axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return np.eye(m) - inv_sqrt @ A @ inv_sqrt


def subspace_overlap(U_ref, U, k):
    """||U_ref^T U||_F^2 / k -- 1.0 iff the two k-dim subspaces coincide."""
    return float((U_ref.T @ U).pow(2).sum()) / k


# --------------------------------------------------------------------- #
# 1. _cheb_lowpass_coeffs
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    # `margin` skips the transition band, where a finite expansion must blur the
    # step; it can shrink as the degree rises (a sharper transition).
    "cutoff,degree,margin",
    [(0.5, 40, 0.35), (1.0, 60, 0.30), (0.2, 80, 0.15), (1.5, 60, 0.30)],
)
def test_lowpass_approximates_the_ideal_step(config, cutoff, degree, margin):
    p = chebmod.chebval(GRID - 1.0, _cheb_lowpass_coeffs(cutoff, degree))

    below = GRID < cutoff - margin
    above = GRID > cutoff + margin
    assert below.any() and above.any()
    assert np.abs(p[below] - 1.0).max() < 0.02
    assert np.abs(p[above]).max() < 0.02


def test_jackson_damping_kills_the_gibbs_overshoot(config):
    cutoff, degree = 0.5, 40
    damped = _cheb_lowpass_coeffs(cutoff, degree, jackson=True)
    undamped = _cheb_lowpass_coeffs(cutoff, degree, jackson=False)

    # g_0 == 1, so the DC coefficient is untouched
    assert damped[0] == undamped[0]
    # every higher coefficient is damped, strictly so wherever it is nonzero
    assert np.all(np.abs(damped[1:]) <= np.abs(undamped[1:]) + 1e-15)
    nonzero = np.abs(undamped[1:]) > 1e-12
    assert nonzero.any()
    assert np.all(np.abs(damped[1:])[nonzero] < np.abs(undamped[1:])[nonzero])

    p_damped = chebmod.chebval(GRID - 1.0, damped)
    p_undamped = chebmod.chebval(GRID - 1.0, undamped)
    # ringing would give the filter NEGATIVE gain on some high-frequency modes
    assert p_damped.min() > -0.05
    assert p_undamped.min() < -0.05
    assert p_damped.max() <= p_undamped.max()


def test_lowpass_edge_cases(config):
    assert _cheb_lowpass_coeffs(0.5, 0).shape == (1,)
    assert np.isfinite(_cheb_lowpass_coeffs(0.5, 0)).all()

    # cutoff is clipped into [0, 2] rather than producing nan from arccos
    for cutoff in (-3.0, -1.0, 3.0, 7.5):
        coef = _cheb_lowpass_coeffs(cutoff, 5)
        assert np.isfinite(coef).all()
    # below the range: pass nothing; above it: pass everything
    assert np.abs(chebmod.chebval(GRID - 1.0, _cheb_lowpass_coeffs(-3.0, 5))).max() < 1e-12
    assert np.abs(chebmod.chebval(GRID - 1.0, _cheb_lowpass_coeffs(3.0, 5)) - 1.0).max() < 1e-12


# --------------------------------------------------------------------- #
# 2. _cheb_filter
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("degree", [1, 2, 12, 40])
def test_filter_matches_the_explicit_chebyshev_expansion(config, degree):
    # Evaluated on a real L_sym: the recurrence is only well-conditioned when the
    # spectrum sits inside [0, 2], which is the operator this is written for.
    L = dense_lsym(PIN_GIANT, PIN_ACTIVE)
    m = L.shape[0]
    X = np.random.default_rng(0).standard_normal((m, 4))
    coef = _cheb_lowpass_coeffs(0.6, degree)

    got = _cheb_filter(L, X, coef)

    shifted = L - np.eye(m)
    terms = [np.eye(m), shifted]
    while len(terms) < len(coef):
        terms.append(2.0 * shifted @ terms[-1] - terms[-2])
    want = sum(c * (T @ X) for c, T in zip(coef, terms))

    assert got.shape == X.shape
    assert np.allclose(got, want, atol=1e-10, rtol=1e-10)


def test_filter_degree_zero_and_one_closed_forms(config):
    L = dense_lsym(PIN_GIANT, PIN_ACTIVE)
    m = L.shape[0]
    X = np.random.default_rng(1).standard_normal((m, 3))

    assert np.allclose(_cheb_filter(L, X, np.array([2.0])), 2.0 * X, atol=1e-12)
    assert np.allclose(
        _cheb_filter(L, X, np.array([1.0, 3.0])),
        X + 3.0 * ((L - np.eye(m)) @ X),
        atol=1e-12,
    )


def test_filter_is_linear_in_x(config):
    L = dense_lsym(PIN_GIANT, PIN_ACTIVE)
    m = L.shape[0]
    rng = np.random.default_rng(2)
    A, B = rng.standard_normal((m, 3)), rng.standard_normal((m, 3))
    coef = _cheb_lowpass_coeffs(0.6, 20)

    combined = _cheb_filter(L, 2.5 * A - 1.5 * B, coef)
    separate = 2.5 * _cheb_filter(L, A, coef) - 1.5 * _cheb_filter(L, B, coef)

    assert np.allclose(combined, separate, atol=1e-12)


def test_filter_accepts_the_sparse_operator_the_solvers_pass(config):
    L = dense_lsym(PIN_GIANT, PIN_ACTIVE)
    X = np.random.default_rng(3).standard_normal((L.shape[0], 3))
    coef = _cheb_lowpass_coeffs(0.6, 15)

    assert np.allclose(
        np.asarray(_cheb_filter(sparse.csr_matrix(L), X, coef)),
        _cheb_filter(L, X, coef),
        atol=1e-12,
    )


# --------------------------------------------------------------------- #
# 3. _active_lsym
# --------------------------------------------------------------------- #


def test_active_lsym_keeps_only_the_giant_component(config):
    graph = make_graph(PIN_GIANT + PIN_SATELLITE, PIN_NODES)

    L, act = graph._active_lsym()

    assert act.tolist() == PIN_ACTIVE  # size-5 component, by GLOBAL index
    assert L.shape == (len(PIN_ACTIVE), len(PIN_ACTIVE))


@pytest.mark.parametrize(
    "edges,num_nodes", [([], 6), ([], 0), ([(0, 0)], 4)]  # no edges / self-loop only
)
def test_active_lsym_returns_none_below_two_active_nodes(config, edges, num_nodes):
    assert make_graph(edges, num_nodes)._active_lsym() == (None, None)


def test_active_lsym_is_a_normalized_laplacian(config):
    L, act = make_sbm()._active_lsym()
    dense = L.toarray()

    assert np.allclose(dense, dense.T, atol=1e-12)
    assert np.allclose(np.diag(dense), 1.0, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(dense)
    assert eigenvalues.min() > -1e-9
    assert eigenvalues.max() < 2.0 + 1e-9
    assert abs(eigenvalues[0]) < 1e-9  # connected -> exactly one zero mode
    assert abs(np.trace(dense) - len(act)) < 1e-9


def test_calc_eigs_exact_sym_pinned_on_a_fixed_graph(config):
    # Regression guard for the _active_lsym extraction: this graph's giant
    # component has 5 nodes and 4 nontrivial eigenvalues that are exact
    # rationals (2/3, 1, 3/2, 11/6), so the pin is BLAS-independent.
    graph = make_graph(PIN_GIANT + PIN_SATELLITE, PIN_NODES)
    k = 4

    w, U = graph.calc_eigs_exact_sym(k)

    assert w.dtype == torch.float32 and U.dtype == torch.float32
    assert w.shape == (k,) and U.shape == (PIN_NODES, k)
    # only the giant component carries a PE; satellite + isolated rows are zero
    assert torch.nonzero(U.abs().sum(dim=1)).ravel().tolist() == PIN_ACTIVE
    assert torch.allclose(
        w, torch.tensor([2 / 3, 1.0, 3 / 2, 11 / 6], dtype=torch.float32), atol=1e-5
    )

    # the returned vectors really diagonalise the reference Laplacian
    L = dense_lsym(PIN_GIANT, PIN_ACTIVE)
    V = U[PIN_ACTIVE].numpy().astype(np.float64)
    assert np.allclose(V.T @ L @ V, np.diag(w.numpy().astype(np.float64)), atol=1e-5)
    assert np.allclose(np.linalg.norm(V, axis=0), 1.0, atol=1e-5)
    for j in range(k):
        # The sign convention is only well defined where the largest |entry| is
        # unique. This graph is vertex-transitive enough that three of its four
        # columns tie in float32 (the sign is fixed on the float64 solve, then
        # cast), so those columns have no canonical gauge to assert -- same
        # caveat as tests/test_fpe_lappe.py::test_sign_is_canonicalized_*.
        magnitudes = np.abs(V[:, j])
        if (magnitudes == magnitudes.max()).sum() > 1:
            continue
        assert V[magnitudes.argmax(), j] > 0


# --------------------------------------------------------------------- #
# 4. calc_eigs_chebyshev
# --------------------------------------------------------------------- #


def test_chebyshev_matches_the_exact_solver_contract(config):
    graph = make_graph(PIN_GIANT + PIN_SATELLITE, PIN_NODES)
    k = 3

    w, U = graph.calc_eigs_chebyshev(k, cutoff=1.0, seed=0)
    w_exact, U_exact = graph.calc_eigs_exact_sym(k)

    assert w.dtype == w_exact.dtype == torch.float32
    assert U.dtype == U_exact.dtype == torch.float32
    assert w.shape == w_exact.shape == (k,)
    assert U.shape == U_exact.shape == (PIN_NODES, k)
    # inactive and satellite-component rows are EXACTLY zero, as in the exact path
    assert torch.nonzero(U.abs().sum(dim=1)).ravel().tolist() == PIN_ACTIVE
    inactive = [i for i in range(PIN_NODES) if i not in PIN_ACTIVE]
    assert U[inactive].abs().max() == 0.0


def test_chebyshev_sign_is_canonicalized(config):
    graph = make_sbm()
    k = 8

    _, U = graph.calc_eigs_chebyshev(k, cutoff=0.7, seed=0)

    for j in range(k):
        column = U[:, j]
        if column.abs().max() == 0.0:
            continue
        assert column[column.abs().argmax()] > 0


def test_chebyshev_zero_pads_when_fewer_pairs_than_k(config):
    # a 5-node component has only 4 nontrivial pairs
    graph = make_graph(PIN_GIANT + PIN_SATELLITE, PIN_NODES)
    k = 9

    w, U = graph.calc_eigs_chebyshev(k, cutoff=1.5, seed=0)

    assert w.shape == (k,) and U.shape == (PIN_NODES, k)
    assert torch.equal(w[4:], torch.zeros(k - 4))
    assert U[:, 4:].abs().max() == 0.0


def test_chebyshev_accuracy_at_the_documented_cutoff(config):
    graph = make_sbm()
    k = 8
    w_exact, U_exact = graph.calc_eigs_exact_sym(k)
    lambda_k = float(w_exact[k - 1])

    w, U = graph.calc_eigs_chebyshev(k, cutoff=lambda_k, seed=0)

    assert subspace_overlap(U_exact, U, k) > 0.99
    assert float((w - w_exact).abs().max()) < 1e-4
    # the trivial (constant) mode is dropped, so w[0] is the FIRST nontrivial
    # eigenvalue -- not a near-zero component indicator shifting everything.
    assert float(w[0]) == pytest.approx(float(w_exact[0]), abs=1e-4)


def test_chebyshev_overlap_degrades_when_cutoff_rises_well_above_lambda_k(config):
    graph = make_sbm()
    k = 8
    w_exact, U_exact = graph.calc_eigs_exact_sym(k)
    lambda_k = float(w_exact[k - 1])

    at_lambda_k = subspace_overlap(
        U_exact, graph.calc_eigs_chebyshev(k, cutoff=lambda_k, seed=0)[1], k
    )
    far_above = subspace_overlap(
        U_exact, graph.calc_eigs_chebyshev(k, cutoff=lambda_k * 2.0, seed=0)[1], k
    )

    # direction only: the docstring's numbers are dataset-specific
    assert far_above < at_lambda_k


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_warm_start_is_at_least_as_good_as_a_random_block(config, seed):
    graph = make_sbm()
    k = 8
    w_exact, U_exact = graph.calc_eigs_exact_sym(k)
    lambda_k = float(w_exact[k - 1])
    kwargs = dict(cutoff=lambda_k, n_iter=1, oversample=4, seed=seed)

    cold = graph.calc_eigs_chebyshev(k, **kwargs)[1]
    warm = graph.calc_eigs_chebyshev(k, X0=U_exact.numpy(), **kwargs)[1]

    assert subspace_overlap(U_exact, warm, k) >= subspace_overlap(U_exact, cold, k) - 1e-6


@pytest.mark.parametrize("bad", ["zeros", "nan", "all_nan"])
def test_degenerate_x0_falls_back_to_a_random_block(config, bad):
    graph = make_sbm()
    k, oversample = 8, 4
    n = graph.num_nodes
    w_exact, U_exact = graph.calc_eigs_exact_sym(k)
    width = k + oversample
    if bad == "zeros":
        X0 = np.zeros((n, width))
    elif bad == "nan":
        X0 = U_exact.numpy().copy()
        X0[0, 0] = np.nan
    else:
        X0 = np.full((n, width), np.nan)

    w, U = graph.calc_eigs_chebyshev(
        k, cutoff=float(w_exact[k - 1]), oversample=oversample, X0=X0, seed=1
    )

    assert torch.isfinite(w).all() and torch.isfinite(U).all()
    assert U.abs().max() > 0.0  # a real basis, not a silently empty one


@pytest.mark.parametrize("n_cols", [3, 8, 48])
def test_x0_columns_are_padded_or_truncated_to_the_block_width(config, n_cols):
    graph = make_sbm()
    k, oversample = 8, 4
    n = graph.num_nodes
    w_exact, U_exact = graph.calc_eigs_exact_sym(k)
    rng = np.random.default_rng(9)
    base = U_exact.numpy()
    X0 = (
        base[:, :n_cols]
        if n_cols <= k
        else np.hstack([base, rng.standard_normal((n, n_cols - k))])
    )

    w, U = graph.calc_eigs_chebyshev(
        k, cutoff=float(w_exact[k - 1]), oversample=oversample, X0=X0, seed=1
    )

    assert w.shape == (k,) and U.shape == (n, k)
    assert torch.isfinite(U).all()
    assert subspace_overlap(U_exact, U, k) > 0.5


def test_chebyshev_accepts_a_torch_x0(config):
    # the tracker holds the previous snapshot's basis as a torch tensor
    graph = make_sbm()
    k = 8
    w_exact, U_exact = graph.calc_eigs_exact_sym(k)

    w, U = graph.calc_eigs_chebyshev(
        k, cutoff=float(w_exact[k - 1]), X0=U_exact, seed=1
    )

    assert torch.isfinite(U).all()
    assert subspace_overlap(U_exact, U, k) > 0.99


def test_chebyshev_is_deterministic_for_a_fixed_seed(config):
    graph = make_sbm()
    k = 8
    kwargs = dict(cutoff=0.6, degree=30, oversample=16, n_iter=2)

    first = graph.calc_eigs_chebyshev(k, seed=7, **kwargs)
    second = graph.calc_eigs_chebyshev(k, seed=7, **kwargs)
    other = graph.calc_eigs_chebyshev(k, seed=8, **kwargs)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[1], other[1])


@pytest.mark.parametrize(
    "edges,num_nodes,k",
    [
        ([(0, 1), (1, 2)], 10, 50),   # k far beyond the active-node count
        ([(0, 1)], 2, 3),             # a 2-node graph
        ([], 6, 4),                   # empty edge_index
        ([(0, 1), (1, 2), (2, 0)], 3, 3),
    ],
)
def test_chebyshev_edge_cases_keep_the_contract(config, edges, num_nodes, k):
    graph = make_graph(edges, num_nodes)

    w, U = graph.calc_eigs_chebyshev(k, cutoff=1.0, seed=0)

    assert w.dtype == torch.float32 and U.dtype == torch.float32
    assert w.shape == (k,)
    assert U.shape == (num_nodes, k)
    assert torch.isfinite(w).all() and torch.isfinite(U).all()
