import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.utils.graph import Graph
from src.dynamic_server import DynamicServer

N = 8

# Cumulative union at t=2 is a connected chain over nodes 0..6 plus the isolated
# node 7, and (0,1)/(1,2) each appear twice, so every kernel has both a repeated
# and a single-appearance edge to weigh.
SNAP_PAIRS = [
    [(0, 1), (1, 2), (2, 3)],
    [(1, 2), (3, 4), (4, 5)],
    [(0, 1), (5, 6), (2, 5)],
]


def _snap(pairs, n=N):
    ei = (
        torch.tensor(pairs, dtype=torch.long).t()
        if pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    d = Data(x=torch.ones(n, 1), edge_index=ei, edge_attr=torch.zeros(ei.size(1), 1), num_nodes=n)
    d.node_ids = torch.arange(n)
    return d


def _server(pairs_per_snap=None, n=N):
    snaps = [_snap(p, n) for p in (pairs_per_snap or SNAP_PAIRS)]
    return DynamicServer(snaps)


def _accumulate(srv, upto):
    for t in range(upto + 1):
        srv._accumulate_cum_edges(t)
    return srv


def _cum_graph(srv, n=N):
    return Graph(x=torch.ones(n, 1), edge_index=srv._cum_edges, node_ids=torch.arange(n))


def _w(srv, weights, u, v):
    ce = srv._cum_edges
    col = ((ce[0] == u) & (ce[1] == v)).nonzero()
    assert col.numel() == 1
    return float(weights[int(col)])


def _set_kernel(config, kind, param=None):
    config["spectral"]["cum_decay"] = kind
    if param is not None:
        config["spectral"]["cum_decay_param"] = param


def test_default_kernel_leaves_the_cumulative_adjacency_unweighted(config):
    assert config["spectral"]["cum_decay"] == "none"
    srv = _server()
    for t in range(3):
        srv._accumulate_cum_edges(t)
        assert srv._cum_edge_weight(t) is None

    # "none" means the binarizing path, and that path is what an all-ones
    # weighting reproduces on a symmetrized edge list.
    g = _cum_graph(srv)
    L0, a0 = g._active_lsym(None)
    L1, a1 = g._active_lsym(np.ones(srv._cum_edges.size(1)))
    assert np.array_equal(a0, a1)
    assert np.abs(L0.toarray() - L1.toarray()).max() == 0.0


def test_harmonic_weight_is_the_exact_sum_over_appearances(config):
    _set_kernel(config, "harmonic")
    srv = _accumulate(_server(), 2)
    w = srv._cum_edge_weight(2)
    # (0,1) appears at s=0 (age 2) and s=2 (age 0): 1/3 + 1
    assert _w(srv, w, 0, 1) == pytest.approx(1.0 / 3.0 + 1.0, abs=1e-12)
    # (1,2) at s=0 and s=1: 1/3 + 1/2
    assert _w(srv, w, 1, 2) == pytest.approx(1.0 / 3.0 + 1.0 / 2.0, abs=1e-12)
    # single appearances
    assert _w(srv, w, 2, 3) == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert _w(srv, w, 3, 4) == pytest.approx(1.0 / 2.0, abs=1e-12)
    assert _w(srv, w, 5, 6) == pytest.approx(1.0, abs=1e-12)
    # more often and more recently outweighs once and long ago
    assert _w(srv, w, 0, 1) > _w(srv, w, 1, 2) > _w(srv, w, 2, 3)


def test_count_exp_and_window_evaluate_their_stated_kernels(config):
    srv = _accumulate(_server(), 2)

    _set_kernel(config, "count")
    w = srv._cum_edge_weight(2)
    assert _w(srv, w, 0, 1) == pytest.approx(2.0, abs=1e-12)
    assert _w(srv, w, 1, 2) == pytest.approx(2.0, abs=1e-12)
    assert _w(srv, w, 2, 3) == pytest.approx(1.0, abs=1e-12)
    assert _w(srv, w, 5, 6) == pytest.approx(1.0, abs=1e-12)

    _set_kernel(config, "exp", 0.5)
    w = srv._cum_edge_weight(2)
    assert _w(srv, w, 0, 1) == pytest.approx(0.5**2 + 0.5**0, abs=1e-12)
    assert _w(srv, w, 1, 2) == pytest.approx(0.5**2 + 0.5**1, abs=1e-12)
    assert _w(srv, w, 2, 3) == pytest.approx(0.5**2, abs=1e-12)

    # gamma=1 removes the recency term, so exp degenerates to count
    _set_kernel(config, "exp", 1.0)
    w_exp1 = srv._cum_edge_weight(2)
    _set_kernel(config, "count")
    assert torch.equal(w_exp1, srv._cum_edge_weight(2))

    # window(W) counts appearances with age < W, strictly
    _set_kernel(config, "window", 2)
    w = srv._cum_edge_weight(2)
    assert _w(srv, w, 0, 1) == pytest.approx(1.0, abs=1e-12)   # s=2 in, s=0 out
    assert _w(srv, w, 1, 2) == pytest.approx(1.0, abs=1e-12)   # s=1 in, s=0 out
    assert _w(srv, w, 2, 3) == pytest.approx(0.0, abs=1e-12)   # s=0 only, age 2

    _set_kernel(config, "window", 3)
    w = srv._cum_edge_weight(2)
    assert _w(srv, w, 0, 1) == pytest.approx(2.0, abs=1e-12)
    assert _w(srv, w, 2, 3) == pytest.approx(1.0, abs=1e-12)


def test_both_directed_columns_of_an_edge_carry_the_same_weight(config):
    _set_kernel(config, "harmonic")
    srv = _accumulate(_server(), 2)
    w = srv._cum_edge_weight(2)
    ce = srv._cum_edges
    assert w.shape == (ce.size(1),)
    assert w.dtype == torch.float64
    for i in range(ce.size(1)):
        u, v = int(ce[0, i]), int(ce[1, i])
        assert _w(srv, w, u, v) == _w(srv, w, v, u)


@pytest.mark.parametrize(
    "kind,param", [("count", None), ("harmonic", None), ("exp", 0.5), ("window", 2)]
)
def test_weighted_operator_keeps_its_defining_properties(config, kind, param):
    _set_kernel(config, kind, param)
    srv = _accumulate(_server(), 2)
    L, act = _cum_graph(srv)._active_lsym(srv._cum_edge_weight(2))
    M = L.toarray()
    # scipy assembles in float64; the symmetrization is exact so 1e-12 is loose
    assert np.abs(M - M.T).max() < 1e-12
    assert np.abs(np.diag(M) - 1.0).max() < 1e-12
    assert np.trace(M) == pytest.approx(float(act.size), abs=1e-12)
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    assert ev.min() > -1e-9
    assert ev.max() < 2.0 + 1e-9


@pytest.mark.parametrize("c", [0.5, 3.7, 1000.0])
def test_laplacian_is_invariant_to_a_global_rescaling_of_the_weights(config, c):
    _set_kernel(config, "harmonic")
    srv = _accumulate(_server(), 2)
    g = _cum_graph(srv)
    w = srv._cum_edge_weight(2)
    L1, a1 = g._active_lsym(w)
    L2, a2 = g._active_lsym(w * c)
    assert np.array_equal(a1, a2)
    # D^-1/2 A D^-1/2 cancels c analytically; what is left is float64 rounding
    assert np.abs(L1.toarray() - L2.toarray()).max() < 1e-12


@pytest.mark.parametrize("kind,param", [("count", None), ("harmonic", None), ("exp", 0.5)])
def test_strictly_positive_kernels_leave_the_active_set_alone(config, kind, param):
    srv = _accumulate(_server(), 2)
    _, act_none = _cum_graph(srv)._active_lsym(None)
    _set_kernel(config, kind, param)
    w = srv._cum_edge_weight(2)
    assert float(w.min()) > 0.0
    _, act_w = _cum_graph(srv)._active_lsym(w)
    assert np.array_equal(act_none, act_w)


def test_window_may_shrink_the_active_set(config):
    srv = _accumulate(_server(), 2)
    _, act_none = _cum_graph(srv)._active_lsym(None)
    _set_kernel(config, "window", 1)
    w = srv._cum_edge_weight(2)
    # every appearance of (2,3) and (3,4) has aged out -> exactly zero
    assert _w(srv, w, 2, 3) == 0.0
    assert _w(srv, w, 3, 4) == 0.0
    _, act_w = _cum_graph(srv)._active_lsym(w)
    assert set(act_w.tolist()) < set(act_none.tolist())
    assert 3 not in set(act_w.tolist())


def test_negative_weight_is_rejected(config):
    _set_kernel(config, "harmonic")
    srv = _accumulate(_server(), 2)
    w = srv._cum_edge_weight(2).clone()
    w[0] = -1e-9
    with pytest.raises(ValueError, match="non-negative"):
        _cum_graph(srv)._active_lsym(w)


def test_length_mismatched_weight_is_rejected(config):
    srv = _accumulate(_server(), 2)
    g = _cum_graph(srv)
    n_cols = srv._cum_edges.size(1)
    with pytest.raises(ValueError, match=r"entries for \d+ edge columns"):
        g._active_lsym(np.ones(n_cols - 1))
    with pytest.raises(ValueError, match=r"entries for \d+ edge columns"):
        g._active_lsym(np.ones(n_cols + 1))


def test_empty_snapshots_contribute_no_appearances(config):
    _set_kernel(config, "harmonic")
    srv = _accumulate(_server([[(0, 1)], [], [(1, 2)]]), 2)
    assert 1 not in srv._cum_events_t.tolist()
    w = srv._cum_edge_weight(2)
    assert _w(srv, w, 0, 1) == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert _w(srv, w, 1, 2) == pytest.approx(1.0, abs=1e-12)


def test_an_entirely_empty_history_has_no_weights_and_no_operator(config):
    _set_kernel(config, "count")
    srv = _accumulate(_server([[], []]), 1)
    assert srv._cum_edges.numel() == 0
    assert srv._cum_events_key is None
    assert srv._cum_edge_weight(1) is None
    assert _cum_graph(srv)._active_lsym(None) == (None, None)


def test_single_edge_history(config):
    _set_kernel(config, "count")
    srv = _accumulate(_server([[(0, 1)], [(0, 1)]]), 0)
    w = srv._cum_edge_weight(0)
    assert w.tolist() == [1.0, 1.0]  # both directed columns
    L, act = _cum_graph(srv)._active_lsym(w)
    assert act.tolist() == [0, 1]
    assert np.allclose(L.toarray(), np.array([[1.0, -1.0], [-1.0, 1.0]]), atol=1e-12)


def test_self_loop_only_history_is_degenerate_under_every_kernel(config):
    _set_kernel(config, "count")
    srv = _accumulate(_server([[(0, 0)], [(0, 0)]]), 1)
    w = srv._cum_edge_weight(1)
    assert float(w.sum()) == pytest.approx(2.0, abs=1e-12)  # two appearances
    g = _cum_graph(srv)
    assert g._active_lsym(w) == (None, None)
    assert g._active_lsym(None) == (None, None)


def test_kernel_does_not_move_the_union_or_the_repeat_split(config):
    pos = torch.tensor([[2, 0], [3, 5]])  # (2,3) is a repeat, (0,5) is new

    srv_none = _accumulate(_server(), 2)
    union_none = srv_none._cum_edges.clone()
    mask_none = srv_none._repeat_mask(pos).clone()

    # window(1) drives (2,3) to weight zero; the split must not notice
    _set_kernel(config, "window", 1)
    srv_w = _accumulate(_server(), 2)
    assert torch.equal(srv_w._cum_edges, union_none)
    assert torch.equal(srv_w._repeat_mask(pos), mask_none)
    assert bool(mask_none[0]) and not bool(mask_none[1])
    assert _w(srv_w, srv_w._cum_edge_weight(2), 2, 3) == 0.0


def test_rebuilt_appearance_record_matches_uninterrupted_accumulation(config):
    _set_kernel(config, "harmonic")
    live = _accumulate(_server(), 2)

    resumed = _server()
    resumed._cum_edges = live._cum_edges.clone()
    resumed._rebuild_cum_events(2)

    assert torch.equal(resumed._cum_events_key, live._cum_events_key)
    assert torch.equal(resumed._cum_events_t, live._cum_events_t)
    assert torch.equal(resumed._cum_edge_weight(2), live._cum_edge_weight(2))


def test_unknown_kernel_raises_rather_than_defaulting(config):
    _set_kernel(config, "bogus")
    srv = _accumulate(_server(), 2)
    with pytest.raises(ValueError, match="cum_decay"):
        srv._cum_edge_weight(2)


def test_run_identity_is_unchanged_at_default_and_separates_the_arms(config):
    config["dataset"]["name"] = "uci"
    config["model"]["data_type"] = "f+s"
    srv = _server()

    rid_default = srv._run_id()
    assert "cum-" not in rid_default
    del config["spectral"]["cum_decay"]
    assert srv._run_id() == rid_default  # byte-identical to a config predating the key

    ids = set()
    for kind, param in [("count", None), ("harmonic", None), ("exp", 0.5),
                        ("exp", 0.9), ("window", 2), ("window", 5)]:
        _set_kernel(config, kind, param)
        rid = srv._run_id()
        assert rid != rid_default
        ids.add(rid)
    assert len(ids) == 6


def test_weighted_operator_is_the_one_the_exact_and_chebyshev_solvers_diagonalize(config):
    _set_kernel(config, "harmonic")
    srv = _accumulate(_server(), 2)
    w = srv._cum_edge_weight(2)
    L, act = _cum_graph(srv)._active_lsym(w)
    Ld = L.toarray()

    for solve in (
        lambda g: g.calc_eigs_exact_sym(3),
        lambda g: g.calc_eigs_chebyshev(3, cutoff=0.9),
    ):
        g_plain = _cum_graph(srv)
        g_weighted = _cum_graph(srv)
        g_weighted.cum_weight = w
        D0, _ = solve(g_plain)
        Dw, Uw = solve(g_weighted)
        assert not torch.allclose(D0, Dw, atol=1e-4)
        # the returned pairs satisfy L(w) u = lambda u on the covered rows
        Ua = Uw.numpy()[act]
        assert np.abs(Ld @ Ua - Ua * Dw.numpy()).max() < 1e-5


def test_configured_kernel_reaches_the_solver_the_run_actually_uses(config):
    # cum_decay reaches only calc_eigs_exact_sym and calc_eigs_chebyshev. The
    # DEFAULT solver ('arnoldi' -> calc_eignvalues) and the update-mode tracking
    # path (update_eigpairs) both rebuild L via create_L from the unweighted
    # original_edge_index, so neither can be treated. This test pins that fact so
    # the assert_cfg guard which refuses those combinations stays justified: if a
    # future change makes create_L weight-aware, this fails and the guard should
    # be relaxed in the same commit.
    assert config["spectral"]["solver"] == "arnoldi"
    _set_kernel(config, "harmonic")
    srv = _accumulate(_server(), 2)
    w = srv._cum_edge_weight(2)

    g_plain, g_weighted = _cum_graph(srv), _cum_graph(srv)
    g_weighted.cum_weight = w
    D0, _, _ = g_plain.calc_eignvalues(estimate=False, spectral_len=3, log=False)
    Dw, _, _ = g_weighted.calc_eignvalues(estimate=False, spectral_len=3, log=False)
    assert torch.equal(D0, Dw), "calc_eignvalues now honours cum_weight -- relax the assert_cfg guard"

    Q = torch.linalg.qr(torch.randn(N, 3))[0]
    g_plain, g_weighted = _cum_graph(srv), _cum_graph(srv)
    g_weighted.cum_weight = w
    Dt0, _, _ = g_plain.update_eigpairs(Q)
    Dtw, _, _ = g_weighted.update_eigpairs(Q)
    assert torch.equal(Dt0, Dtw), "update_eigpairs now honours cum_weight -- relax the assert_cfg guard"
