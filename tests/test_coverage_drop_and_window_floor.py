"""spectral.coverage_drop and spectral.window_floor: the instrument that has to
separate a window's RECENCY change from its COVERAGE change.

coverage_drop is the control, so its one non-negotiable property is that it
changes NOTHING except which served rows are zero. If the mask leaks into the
tracking basis or the Procrustes anchor it changes the spectrum too, which is
precisely the confound the change exists to remove, and the control stops
controlling for anything. window_floor is the mirror: it must restore the active
set a hard window destroys while leaving in-horizon weights alone. Both knobs are
also silent no-ops on the wrong path, and a silent no-op reads exactly like a
real negative result, so the refusals are pinned too.
"""

import pytest
import torch
from torch_geometric.data import Data

import src.dynamic_server as dynamic_server
from config.assertions import assert_cfg
from config.config import get_default_config
from src.dynamic_server import DynamicServer, SpectralFeatures
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots
from src.train.federated_orchestrator import _partition_edges_per_snapshot
from test_checkpoint_wandb import make_server, make_toy_snapshots, seed_all
from test_edge_score_smodel import set_fes_model_config
from test_fl_local_baseline import make_toy_snapshots as make_dense_snapshots
from test_run_identity import explicit_id, set_identity_config


DENSE_NODES = 12
DENSE_SCHEDULE = [
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    [(0, 1), (1, 2), (5, 6), (6, 7), (7, 8)],
    [(0, 1), (2, 3), (8, 9), (9, 10), (10, 11)],
    [(1, 2), (3, 4), (5, 7), (11, 0)],
]

# 8 nodes. (2,3) is seen ONLY at t=0, so at t=3 under window(1) both its endpoints
# lose every edge and leave the active set -- the node the floor has to rescue.
# (0,1) is seen three times, all of them stale at t=3: the multi-appearance case.
# (4,5) is seen at t=0 AND t=3, so it is in-horizon with a stale appearance too.
FLOOR_NODES = 8
FLOOR_SCHEDULE = [
    [(0, 1), (2, 3), (4, 5)],
    [(0, 1), (5, 6)],
    [(0, 1), (6, 7)],
    [(4, 5), (1, 2)],
]


def snapshots(schedule, n_nodes):
    out = []
    for edges in schedule:
        ei = (torch.tensor(edges, dtype=torch.long).t() if edges
              else torch.zeros(2, 0, dtype=torch.long))
        snap = Data(x=torch.ones(n_nodes, 1), edge_index=ei,
                    edge_attr=torch.ones(ei.size(1), 1), num_nodes=n_nodes)
        snap.node_ids = torch.arange(n_nodes)
        out.append(snap)
    return out


def setup_basis_config(config):
    config["dataset"]["name"] = "uci"
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [8, 8]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["structure_model"]["num_structural_features"] = 16
    config["structure_model"]["DGCN_structure_layers_sizes"] = [8]
    config["spectral"]["spectral_len"] = 4
    config["spectral"]["pe_dim"] = 4
    config["subgraph"]["num_subgraphs"] = 1
    config["seed"] = 42


def served_bases(config, data_type, update_mode, solver, drop, procrustes="auto"):
    """Every U handed to the server classifier over a whole run."""
    snaps = snapshots(DENSE_SCHEDULE, DENSE_NODES)
    config["model"]["data_type"] = data_type
    config["spectral"]["update_mode"] = update_mode
    config["spectral"]["solver"] = solver
    config["spectral"]["coverage_drop"] = drop
    config["spectral"]["use_procrustes"] = procrustes
    seed_all(42)
    server = make_server(snaps, partition_snapshots(snaps, 1))
    server.initialize_FL()
    seen = []
    original = server.classifier.set_QD

    def capture(U, D):
        seen.append(U.detach().cpu().clone())
        original(U, D)

    server.classifier.set_QD = capture
    for t in range(len(snaps)):
        server._spectral_step(t, config["model"]["smodel_type"])
    return seen


def zero_rows(U):
    return U.abs().sum(dim=1) == 0


def synthetic_basis(n=16, k=4, already_zero=(1, 2, 5, 9, 13, 15)):
    U = torch.randn(n, k, generator=torch.Generator().manual_seed(3))
    U[list(already_zero)] = 0
    return U, U.clone()


# --------------------------------------------------------------------- #
# 1. coverage_drop, unit: the mask itself
#    Requirement: Basis Coverage Control
# --------------------------------------------------------------------- #


def test_the_default_leaves_the_basis_bit_identical(config):
    server = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))
    U, Q = synthetic_basis()
    config["spectral"]["coverage_drop"] = 0.0

    out_U, out_Q, covered, zeroed = server._apply_coverage_drop(U, Q, 0)

    assert torch.equal(out_U, U) and torch.equal(out_Q, Q)
    assert covered == 10 and zeroed == 6      # the restriction's own six, nothing added


def test_a_config_predating_the_knob_is_inert(config):
    server = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))
    U, Q = synthetic_basis()
    del config["spectral"]["coverage_drop"]

    out_U, _, covered, zeroed = server._apply_coverage_drop(U, Q, 0)

    assert torch.equal(out_U, U) and (covered, zeroed) == (10, 6)


# 0.26 and 0.78 are the rounding probes: 2.6 and 7.8 of the 10 covered rows, so
# truncation would zero one row too few and the arm would under-match the window
# it is being compared against.
@pytest.mark.parametrize(
    "p,expected_extra",
    [(0.05, 0), (0.1, 1), (0.26, 3), (0.5, 5), (0.78, 8), (0.9, 9)],
)
def test_the_fraction_is_of_the_covered_rows_not_of_all_rows(config, p, expected_extra):
    # 10 of 16 rows are covered. Drawing from all 16 would spend part of the
    # budget re-zeroing rows the active-set restriction already zeroed, and the
    # arm would under-match the window it exists to match -- silently, because
    # the run id would still stamp cov<p>.
    server = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))
    U, Q = synthetic_basis()
    before = zero_rows(U)
    config["spectral"]["coverage_drop"] = p
    config["seed"] = 42

    out_U, out_Q, covered, zeroed = server._apply_coverage_drop(U, Q, 0)
    after = zero_rows(out_U)

    assert covered == 10
    assert expected_extra == round(p * covered)
    assert int(after.sum()) == int(before.sum()) + expected_extra == zeroed
    # every newly zeroed row was previously COVERED, and no covered row survived
    # that should not have
    assert not (before & ~after).any()          # nothing un-zeroed
    assert int((after & ~before).sum()) == expected_extra


def test_the_drop_does_not_mutate_the_solvers_tensors(config):
    server = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))
    U, Q = synthetic_basis()
    U_ref, Q_ref = U.clone(), Q.clone()
    config["spectral"]["coverage_drop"] = 0.5

    server._apply_coverage_drop(U, Q, 0)

    assert torch.equal(U, U_ref) and torch.equal(Q, Q_ref)


def test_q_is_masked_on_exactly_the_rows_u_is(config):
    server = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))
    U, Q = synthetic_basis()
    config["spectral"]["coverage_drop"] = 0.5

    out_U, out_Q, _, _ = server._apply_coverage_drop(U, Q, 0)

    assert torch.equal(zero_rows(out_U), zero_rows(out_Q))


def test_a_none_tracking_basis_survives(config):
    # the exact/chebyshev paths bind Q = U, but calc_eignvalues can return None
    server = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))
    U, _ = synthetic_basis()
    config["spectral"]["coverage_drop"] = 0.5

    out_U, out_Q, covered, zeroed = server._apply_coverage_drop(U, None, 0)

    assert out_Q is None and covered == 10 and zeroed == 11


def test_an_entirely_uncovered_basis_is_a_no_op_not_a_crash(config):
    # Lsym is None on a graph with fewer than two active nodes; calc_eigs_*
    # then returns an all-zero (n, k). The control must report 0 covered rather
    # than divide the budget by nothing.
    server = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))
    Z = torch.zeros(16, 4)
    config["spectral"]["coverage_drop"] = 0.5

    out_U, _, covered, zeroed = server._apply_coverage_drop(Z, Z.clone(), 0)

    assert torch.equal(out_U, Z) and covered == 0 and zeroed == 16


def test_a_fraction_near_one_zeroes_every_covered_row(config):
    server = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))
    U, Q = synthetic_basis()
    config["spectral"]["coverage_drop"] = 0.99

    out_U, _, covered, zeroed = server._apply_coverage_drop(U, Q, 0)

    assert covered == 10 and zeroed == 16 and int(zero_rows(out_U).sum()) == 16


def test_the_selection_is_reproducible_and_varies_with_snapshot_and_seed(config):
    server = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))
    U, Q = synthetic_basis()
    config["spectral"]["coverage_drop"] = 0.5

    def picked(seed, ss_idx):
        config["seed"] = seed
        out_U, _, _, _ = server._apply_coverage_drop(U, Q, ss_idx)
        return tuple(torch.nonzero(zero_rows(out_U)).flatten().tolist())

    assert picked(42, 0) == picked(42, 0)                 # reproducible
    assert len({picked(42, s) for s in range(5)}) == 5     # redrawn per snapshot
    assert len({picked(s, 0) for s in range(5)}) == 5      # and per seed


# --------------------------------------------------------------------- #
# 2. coverage_drop, pipeline: the mask must not reach the SOLVE
#    Requirement: Basis Coverage Control (design D1)
# --------------------------------------------------------------------- #


# (data_type, update_mode, solver, use_procrustes). The pre-registered
# experiment is the f+es/update/chebyshev row; the rest cover the tracking
# branch that bypasses _substitute_basis and the Procrustes anchor.
PIPELINE_ARMS = [
    ("f+es", "update", "chebyshev", "auto"),
    ("f+es", "recompute", "chebyshev", "auto"),
    ("f+es", "keep", "chebyshev", "auto"),
    ("f+s", "update", "chebyshev", False),
    ("f+s", "recompute", "chebyshev", True),
    ("f+s", "update", "arnoldi", True),
    ("f+s", "recompute", "exact", True),
    ("f+pe", "update", "arnoldi", True),
]


@pytest.mark.parametrize("data_type,update_mode,solver,proc", PIPELINE_ARMS)
def test_the_rows_the_control_keeps_are_the_untreated_arms_rows(
    config, data_type, update_mode, solver, proc
):
    # THE property that makes coverage_drop a control. Applied after the solve,
    # the only thing separating it from the drop=0 arm is which pairs the
    # edge-score term can still reorder. If the mask reaches the tracking
    # warm-start (X0 / prev_Q) or the Procrustes anchor it changes the SPECTRUM
    # too -- the exact confound the change exists to remove -- and every retained
    # row moves with the treatment.
    setup_basis_config(config)
    plain = served_bases(config, data_type, update_mode, solver, 0.0, proc)
    dropped = served_bases(config, data_type, update_mode, solver, 0.5, proc)

    assert len(plain) == len(dropped) == len(DENSE_SCHEDULE)
    for t, (a, b) in enumerate(zip(plain, dropped)):
        kept = ~zero_rows(b)
        assert kept.any(), f"t={t}: the whole basis was zeroed, nothing to compare"
        assert torch.equal(a[kept], b[kept]), (
            f"t={t}: the control moved rows it did not zero "
            f"(max |delta| = {float((a[kept] - b[kept]).abs().max())})"
        )


@pytest.mark.parametrize("data_type,update_mode,solver,proc", PIPELINE_ARMS)
def test_the_zeroed_fraction_does_not_compound_across_snapshots(
    config, data_type, update_mode, solver, proc
):
    # A mask fed back into the cache is re-applied to an already-masked basis, so
    # coverage decays like (1-p)^t instead of holding at p. The arm would then be
    # a different (and unnamed) treatment at every snapshot, and could not be
    # matched to a window's measured covered fraction at all.
    setup_basis_config(config)
    plain = served_bases(config, data_type, update_mode, solver, 0.0, proc)
    dropped = served_bases(config, data_type, update_mode, solver, 0.5, proc)

    for t, (a, b) in enumerate(zip(plain, dropped)):
        covered = int((~zero_rows(a)).sum())
        extra = int(zero_rows(b).sum()) - int(zero_rows(a).sum())
        assert extra == round(0.5 * covered), (
            f"t={t}: zeroed {extra} extra rows of {covered} covered, expected "
            f"{round(0.5 * covered)}"
        )


def test_the_control_reaches_the_basis_it_bills_itself_for(config):
    # the other direction: an arm stamped cov<p> that served an unmodified basis
    # would read exactly like a real null.
    setup_basis_config(config)
    plain = served_bases(config, "f+es", "update", "chebyshev", 0.0)
    dropped = served_bases(config, "f+es", "update", "chebyshev", 0.5)

    for a, b in zip(plain, dropped):
        assert int(zero_rows(b).sum()) > int(zero_rows(a).sum())


def test_the_cumulative_graph_split_and_persistence_do_not_move(config):
    # the constraint results.md 20 records as the most important one: if a
    # baseline moves with the treatment the primary readout is uninterpretable.
    def run(drop):
        set_fes_model_config(config, pe_dim=4)
        config["spectral"]["update_mode"] = "keep"
        config["spectral"]["coverage_drop"] = drop
        config["subgraph"]["num_subgraphs"] = 1
        seed_all(42)
        snaps = make_dense_snapshots(N=30, num_snaps=3, seed=7)
        server = make_server(snaps, partition_snapshots(snaps, 1))
        server.initialize_FL()
        for t in range(len(snaps)):
            server._spectral_step(t, config["model"]["smodel_type"])
        probe = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        return (
            server._cum_edges.clone(),
            server._repeat_mask(probe).clone(),
            server.classifier.smodel.keys.cpu().clone(),
            server.classifier.smodel._persistence(probe).clone(),
        )

    ref = run(0.0)
    treated = run(0.75)

    assert torch.equal(treated[0], ref[0])          # cumulative union
    assert torch.equal(treated[1], ref[1])          # repeat/new split
    assert ref[1].any() and not ref[1].all()        # a split with both subsets
    assert torch.equal(treated[2], ref[2])          # served persistence graph
    assert torch.equal(treated[3], ref[3])          # the persistence feature


# --------------------------------------------------------------------- #
# 3. window_floor
#    Requirement: Age Kernel Weight Floor
# --------------------------------------------------------------------- #


def floored_server(config, kernel, param, floor):
    config["spectral"]["cum_decay"] = kernel
    config["spectral"]["cum_decay_param"] = param
    config["spectral"]["window_floor"] = floor
    server = DynamicServer(snapshots(FLOOR_SCHEDULE, FLOOR_NODES))
    for t in range(len(FLOOR_SCHEDULE)):
        server._accumulate_cum_edges(t)
    return server


def weight_of(server, t, u, v):
    w = server._cum_edge_weight(t)
    ce = server._cum_edges
    hit = ((ce[0] == u) & (ce[1] == v)) | ((ce[0] == v) & (ce[1] == u))
    assert hit.any(), "pair is not in the cumulative union"
    return w[hit]


def test_a_zero_floor_reproduces_the_hard_window_bit_identically(config):
    # not "equivalently": the banked window arms must keep resolving to the same
    # numbers, and float64 0.0 + 0.0*floor is only bit-identical at floor == 0.
    hard = floored_server(config, "window", 1, 0.0)
    with_key = hard._cum_edge_weight(3)
    del config["spectral"]["window_floor"]
    without_key = hard._cum_edge_weight(3)

    assert torch.equal(with_key, without_key)
    assert with_key.dtype == torch.float64
    # and it really is the 0/1 indicator: (4,5) is in-horizon, (0,1) and (2,3)
    # are not, however many times they were seen
    assert weight_of(hard, 3, 4, 5).tolist() == [1.0, 1.0]
    assert weight_of(hard, 3, 0, 1).tolist() == [0.0, 0.0]
    assert weight_of(hard, 3, 2, 3).tolist() == [0.0, 0.0]


def test_an_in_horizon_appearance_still_carries_weight_one(config):
    # a floor applied to every edge instead of only the stale ones would flatten
    # the kernel into 'count' scaled by a constant, and L_sym is invariant to a
    # global rescale -- so the arm would silently become the 'none' basis.
    floored = floored_server(config, "window", 1, 0.01)
    # (1,2) is seen ONCE, at t=3, and is in-horizon there with no stale history
    assert weight_of(floored, 3, 1, 2).tolist() == [1.0, 1.0]


def test_an_out_of_horizon_appearance_carries_the_floor_once_per_appearance(config):
    # w_t(e) = sum over appearances of f(age) for every kernel here (the hard
    # window is already a COUNT of in-horizon appearances, not an indicator), so
    # 3 stale appearances give 3*floor. Pinned as the intended semantics: the
    # floor is a per-APPEARANCE weight, not a per-pair one.
    floored = floored_server(config, "window", 1, 0.01)

    assert weight_of(floored, 3, 2, 3).tolist() == pytest.approx([0.01, 0.01])  # 1 stale
    assert weight_of(floored, 3, 0, 1).tolist() == pytest.approx([0.03, 0.03])  # 3 stale
    # (4,5) is in-horizon at t=3 AND was seen at t=0, so it carries 1 + floor
    assert weight_of(floored, 3, 4, 5).tolist() == pytest.approx([1.01, 1.01])
    # the hard window's own count semantics, for comparison: horizon 3 at t=3
    # admits the t=1 and t=2 appearances of (0,1)
    wide = floored_server(config, "window", 3, 0.0)
    assert weight_of(wide, 3, 0, 1).tolist() == [2.0, 2.0]


def test_a_floor_below_one_keeps_every_in_horizon_appearance_ahead(config):
    for floor in (0.01, 0.1, 0.5):
        floored = floored_server(config, "window", 1, floor)
        fresh = weight_of(floored, 3, 1, 2)          # 1 in-horizon appearance
        stale = weight_of(floored, 3, 2, 3)          # 1 out-of-horizon appearance
        assert (fresh > stale).all()


def test_a_floor_at_or_above_one_destroys_the_recency_it_exists_to_preserve(config):
    # DOCUMENTING a gap, not endorsing it: assert_cfg only requires floor >= 0.
    # At exactly 1.0 the window kernel IS the count kernel, and above 1.0 a pair
    # seen once outside the horizon outweighs a pair seen once inside it -- the
    # opposite of the arm's stated meaning, accepted without a word.
    ones = floored_server(config, "window", 1, 1.0)
    counted = floored_server(config, "count", 0.9, 0.0)
    assert torch.equal(ones._cum_edge_weight(3), counted._cum_edge_weight(3))

    inverted = floored_server(config, "window", 1, 2.0)
    assert_cfg_ok = True
    assert (weight_of(inverted, 3, 2, 3) > weight_of(inverted, 3, 1, 2)).all()
    assert assert_cfg_ok


@pytest.mark.parametrize("kernel,param", [("count", 0.9), ("harmonic", 0.9), ("exp", 0.5)])
def test_the_floor_is_inert_under_every_other_kernel(config, kernel, param):
    reference = floored_server(config, kernel, param, 0.0)._cum_edge_weight(3)

    for floor in (0.01, 0.5, 5.0):
        assert torch.equal(
            floored_server(config, kernel, param, floor)._cum_edge_weight(3), reference
        )


def covered_and_operator(config, kernel, param, floor, t=3, solver="exact"):
    """Solve on the floored cumulative graph and report what the basis covers."""
    server = floored_server(config, kernel, param, floor)
    graph = Graph(x=torch.ones(FLOOR_NODES, 1), edge_index=server._cum_edges,
                  node_ids=torch.arange(FLOOR_NODES))
    graph.cum_weight = server._cum_edge_weight(t)
    Lsym, act = graph._active_lsym(graph.cum_weight)
    if solver == "exact":
        D, U = graph.calc_eigs_exact_sym(3)
    else:
        D, U = graph.calc_eigs_chebyshev(3)
    return int((U.abs().sum(dim=1) > 0).sum()), Lsym, act


@pytest.mark.parametrize("solver", ["exact", "chebyshev"])
def test_a_positive_floor_rescues_a_node_whose_every_edge_is_stale(config, solver):
    # end to end through the solver, not by reading the kernel: (2,3) is seen only
    # at t=0, so under window(1) at t=3 nodes 2 and 3 have degree 0, leave the
    # active set, and get zero-padded rows -- which is the coverage collapse the
    # whole change is about. Node 2 is rescued by its t=3 edge; node 3 is not.
    unweighted, _, _ = covered_and_operator(config, "none", 0.9, 0.0, solver=solver)
    hard, _, act_hard = covered_and_operator(config, "window", 1, 0.0, solver=solver)

    assert hard < unweighted                       # the premise: coverage collapses
    assert 3 not in act_hard.tolist()

    for floor in (0.01, 0.5):
        restored, _, act = covered_and_operator(config, "window", 1, floor, solver=solver)
        assert restored == unweighted, f"floor={floor} restored {restored}, want {unweighted}"
        assert 3 in act.tolist()


def test_the_floored_operator_is_still_a_normalised_laplacian(config):
    # task 6.2: symmetric, unit diagonal, spectrum in [0, 2]. A floor is a weight
    # like any other, but a negative or nan floor reaching A would break all three
    # and the failure would show up only as a garbage basis.
    import numpy as np

    for floor in (0.0, 0.01, 0.5, 1.0):
        _, Lsym, act = covered_and_operator(config, "window", 1, floor)
        dense = Lsym.toarray()
        assert np.allclose(dense, dense.T)
        assert np.allclose(np.diag(dense), 1.0)
        w = np.linalg.eigvalsh(dense)
        assert w.min() > -1e-8 and w.max() < 2.0 + 1e-8
        assert np.isclose(np.trace(dense), act.size)


# --------------------------------------------------------------------- #
# 4. coverage instrumentation
#    Requirements: Basis Coverage Is Measured Per Snapshot;
#                  Basis Coverage Fields In The Per-Snapshot Metric Record
# --------------------------------------------------------------------- #


COVERAGE_KEYS = ("basis_covered", "basis_zeroed", "basis_total", "basis_zeroed_pair_frac")


def _serve_zeroed_basis(server, U):
    """Stand in for the serving hook: pin the zero-row mask the pair fraction
    reads, so a hand-built coverage pattern can be checked against a hand
    computation."""
    server._basis_zero_rows = zero_rows(U)


def instrumented_run(config, drop, data_type="f+es", num_snaps=3, features="spec",
                     monkeypatch=None):
    """Serve + evaluate every snapshot, returning (metrics, served U) per t.

    With `monkeypatch`, also captures the eval batch _eval_mrr built, so the
    pair fraction can be recomputed from the pairs that were actually scored
    (the negatives are drawn inside _eval_mrr and are not otherwise recoverable).
    """
    set_fes_model_config(config, pe_dim=4)
    config["model"]["data_type"] = data_type
    config["spectral"]["es_features"] = features
    config["spectral"]["update_mode"] = "keep"
    config["spectral"]["coverage_drop"] = drop
    config["subgraph"]["num_subgraphs"] = 1
    config["metric"]["repeat_new_split"] = True
    config["experimental"]["rank_eval_multiplier"] = 10
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    seed_all(42)
    snaps = make_dense_snapshots(N=30, num_snaps=num_snaps, seed=7)
    server = make_server(snaps, partition_snapshots(snaps, 1))
    server.initialize_FL()
    _partition_edges_per_snapshot(server.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
    served, out, batches = [], [], []
    original = server.classifier.set_QD

    def capture(U, D):
        served.append(U.detach().cpu().clone())
        original(U, D)

    server.classifier.set_QD = capture
    if monkeypatch is not None:
        attach = dynamic_server._attach_future_link_pred_labels

        def spy(today, tomorrow, pos=None):
            snap = attach(today, tomorrow, pos)
            batches.append(snap.edge_label_index.detach().cpu().clone())
            return snap

        monkeypatch.setattr(dynamic_server, "_attach_future_link_pred_labels", spy)
    for t in range(num_snaps - 1):
        if data_type != "feature":
            server._spectral_step(t, config["model"]["smodel_type"])
        else:
            server._accumulate_cum_edges(t)
        _, m = server._eval_mrr(t, 10, "min")
        out.append(m)
    return server, out, served, batches


def test_every_snapshot_reports_the_coverage_counts(config):
    _, metrics, served, _ = instrumented_run(config, 0.5)

    for t, (m, U) in enumerate(zip(metrics, served)):
        for key in COVERAGE_KEYS:
            assert key in m, f"t={t} missing {key}"
        assert m["basis_total"] == U.shape[0] == 30
        assert m["basis_zeroed"] == int(zero_rows(U).sum())
        assert m["basis_covered"] == 30            # this toy graph is fully covered


def test_the_counts_agree_with_the_control(config):
    plain = instrumented_run(config, 0.0)[1]
    treated = instrumented_run(config, 0.6)[1]

    for a, b in zip(plain, treated):
        restriction_excluded = a["basis_total"] - a["basis_covered"]
        assert a["basis_zeroed"] == restriction_excluded
        assert b["basis_zeroed"] == restriction_excluded + round(0.6 * b["basis_covered"])
        assert b["basis_covered"] == a["basis_covered"]   # the solve did not move


def test_full_coverage_reports_a_zero_pair_fraction(config):
    _, metrics, served, _ = instrumented_run(config, 0.0)

    for m, U in zip(metrics, served):
        assert not zero_rows(U).any()              # the premise
        assert m["basis_zeroed_pair_frac"] == 0.0


def test_the_pair_fraction_is_the_served_basis_of_THIS_snapshot(config, monkeypatch):
    # the field the whole attribution rests on. Reading the PREVIOUS snapshot's
    # basis would be undetectable in the log and would silently misattribute the
    # effect, so recompute it from the U that was actually served at t against
    # the pairs that were actually scored at t.
    _, metrics, served, batches = instrumented_run(
        config, 0.5, num_snaps=4, monkeypatch=monkeypatch
    )
    assert len(batches) == len(metrics)
    fractions = []
    for t, (m, U, eli) in enumerate(zip(metrics, served, batches)):
        z = zero_rows(U)
        expected = float((z[eli[0]] | z[eli[1]]).float().mean())
        assert m["basis_zeroed_pair_frac"] == pytest.approx(expected), f"t={t}"
        fractions.append(expected)
    # the snapshots must not all agree, or "current" and "previous" would be
    # indistinguishable and this test would pass on a stale read
    assert len(set(fractions)) > 1


def test_the_fraction_counts_pairs_and_not_nodes(config, monkeypatch):
    # zero ONE high-degree node. The node fraction is 1/n; the pair fraction is
    # whatever share of edge_label_index touches it, which is much larger. A
    # fraction computed over nodes would report 1/n and hide a biased effect.
    server, _, _, batches = instrumented_run(
        config, 0.0, num_snaps=3, monkeypatch=monkeypatch
    )
    n = server.global_snaps[0].num_nodes
    hub = int(torch.bincount(batches[0].flatten(), minlength=n).argmax())

    U = torch.randn(n, 4)
    U[hub] = 0
    server._basis_coverage = {"basis_covered": n, "basis_zeroed": 1, "basis_total": n}
    _serve_zeroed_basis(server, U)
    batches.clear()
    _, m = server._eval_mrr(0, 10, "min")

    eli = batches[0]
    z = zero_rows(U)
    expected = float((z[eli[0]] | z[eli[1]]).float().mean())
    assert m["basis_zeroed_pair_frac"] == pytest.approx(expected)
    assert expected > 1.0 / n                      # pairs, not nodes
    assert m["basis_zeroed"] == 1


def test_the_fields_are_absent_for_the_backbone(config):
    _, metrics, _, _ = instrumented_run(config, 0.0, data_type="feature")

    for m in metrics:
        assert not any(k.startswith("basis") for k in m)


def test_the_coverage_fields_survive_metric_aggregation(config):
    from src.dynamic_server import _weighted_mean_metrics

    _, metrics, _, _ = instrumented_run(config, 0.5)
    # a resumed run can mix records that predate the fields with records that
    # carry them; absent keys must contribute nothing rather than raise
    mixed = [{"mrr": 0.1}] + metrics

    mean = _weighted_mean_metrics(mixed, [1.0] * len(mixed))

    for key in COVERAGE_KEYS:
        assert key in mean and mean[key] == mean[key]      # not nan
    expected = sum(m["basis_zeroed"] for m in metrics) / len(metrics)
    assert mean["basis_zeroed"] == pytest.approx(expected)


# --------------------------------------------------------------------- #
# 5. the silent no-op: coverage_drop on an arm that never reads the basis
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("features", ["persist", "cn"])
def test_the_baseline_arms_never_read_the_basis(config, features):
    # persist and cn return from edge_score before touching Q, so a coverage_drop
    # on those arms is bit-identical to no drop at all while _run_id still stamps
    # cov<p>. That is the same trap as the cn arm that degraded to feature-only
    # without an error -- assert_cfg has to refuse it, and the test below pins
    # that it does. This one pins the premise.
    def scores(drop):
        server = instrumented_run(config, drop, features=features, num_snaps=2)[0]
        sm = server.classifier.smodel
        with torch.no_grad():
            for p in sm.model.parameters():
                p.add_(0.3)
        return sm.edge_score(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))

    assert torch.equal(scores(0.0), scores(0.9))


@pytest.mark.parametrize("features", ["persist", "cn"])
def test_assert_cfg_refuses_coverage_drop_on_an_arm_that_cannot_see_it(
    config, features
):
    set_fes_model_config(config)
    config["spectral"]["es_features"] = features
    config["spectral"]["coverage_drop"] = 0.5

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "NO EFFECT" in str(exc.value) and "es_features" in str(exc.value)


@pytest.mark.parametrize("features", ["spec", "both"])
def test_the_arms_that_do_read_the_basis_are_accepted(config, features):
    set_fes_model_config(config)
    config["spectral"]["es_features"] = features
    config["spectral"]["coverage_drop"] = 0.5

    assert_cfg(config)


def test_the_refusal_does_not_fire_at_the_default(config):
    # persist and cn must stay runnable; only the pointless combination is refused
    set_fes_model_config(config)
    for features in ("spec", "persist", "both", "cn"):
        config["spectral"]["es_features"] = features
        config["spectral"]["coverage_drop"] = 0.0
        assert_cfg(config)


# --------------------------------------------------------------------- #
# 6. config validation
# --------------------------------------------------------------------- #


def fes_cfg(config):
    set_fes_model_config(config)
    config["spectral"]["cum_decay"] = "none"
    config["spectral"]["window_floor"] = 0.0
    config["spectral"]["coverage_drop"] = 0.0


@pytest.mark.parametrize("bad", [-0.1, -1, 1.0, 1.5, 2, True, False, "0.5", None, [0.5]])
def test_assert_cfg_rejects_a_coverage_drop_outside_the_unit_interval(config, bad):
    fes_cfg(config)
    config["spectral"]["coverage_drop"] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.coverage_drop" in str(exc.value)


@pytest.mark.parametrize("good", [0.0, 0.25, 0.5, 0.999, 0])
def test_assert_cfg_accepts_a_coverage_drop_inside_the_unit_interval(config, good):
    fes_cfg(config)
    config["spectral"]["coverage_drop"] = good

    assert_cfg(config)


@pytest.mark.parametrize("bad", [-0.01, -1, True, False, "0.5", None, [0.1]])
def test_assert_cfg_rejects_a_negative_or_non_numeric_floor(config, bad):
    fes_cfg(config)
    config["spectral"]["cum_decay"] = "window"
    config["spectral"]["cum_decay_param"] = 1
    config["spectral"]["window_floor"] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.window_floor" in str(exc.value)


@pytest.mark.parametrize("good", [0.0, 0, 0.01, 0.5, 1.0, 3])
def test_assert_cfg_accepts_a_non_negative_floor_under_a_window(config, good):
    fes_cfg(config)
    config["spectral"]["cum_decay"] = "window"
    config["spectral"]["cum_decay_param"] = 1
    config["spectral"]["window_floor"] = good

    assert_cfg(config)


def test_assert_cfg_refuses_coverage_drop_where_no_basis_is_served(config):
    fes_cfg(config)
    config["model"]["data_type"] = "feature"
    config["spectral"]["coverage_drop"] = 0.5

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "NO EFFECT" in str(exc.value) and "data_type" in str(exc.value)


@pytest.mark.parametrize("kernel,param", [("none", 0.9), ("count", 0.9),
                                          ("harmonic", 0.9), ("exp", 0.5)])
def test_assert_cfg_refuses_a_floor_without_a_window(config, kernel, param):
    fes_cfg(config)
    config["spectral"]["cum_decay"] = kernel
    config["spectral"]["cum_decay_param"] = param
    config["spectral"]["window_floor"] = 0.01

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "NO EFFECT" in str(exc.value) and "cum_decay" in str(exc.value)


def test_assert_cfg_tolerates_a_config_predating_both_keys(config):
    fes_cfg(config)
    del config["spectral"]["coverage_drop"]
    del config["spectral"]["window_floor"]

    assert_cfg(config)


def test_the_package_defaults_are_the_inert_ones(config):
    defaults = get_default_config()["spectral"]

    assert defaults["coverage_drop"] == 0.0
    assert defaults["window_floor"] == 0.0


# --------------------------------------------------------------------- #
# 7. run identity
#    Requirement: Coverage And Floor Are Part Of Run Identity
# --------------------------------------------------------------------- #


def identity_server(config, tmp_path, data_type="f+es"):
    set_identity_config(config, tmp_path)
    config["model"]["data_type"] = data_type
    if data_type == "f+es":
        config["spectral"]["solver"] = "chebyshev"
    config["spectral"]["cum_decay"] = "none"
    config["spectral"]["cum_decay_param"] = 0.9
    config["spectral"]["coverage_drop"] = 0.0
    config["spectral"]["window_floor"] = 0.0
    return DynamicServer(make_toy_snapshots(num_snaps=2))


PRE_CHANGE_IDS = {
    "feature": "uci_gru_feature_C1_s1234",
    "f+s": ("uci_gru_f+s_C1_um-keep_sfv-local_basis-laplacian_k300"
            "_sm-LanczosLaplace_s1234"),
    "f+pe": "uci_gru_f+pe_C1_um-keep_pe50_basis-laplacian_s1234",
    "f+es": ("uci_gru_f+es_C1_um-keep_pe50_basis-laplacian_esf-spec_esp-phi+cos+lev"
             "_solver-chebyshev_s1234"),
}


@pytest.mark.parametrize("data_type", sorted(PRE_CHANGE_IDS))
def test_the_default_identity_is_byte_identical_to_the_pre_change_string(
    config, tmp_path, data_type
):
    server = identity_server(config, tmp_path, data_type)
    config["spectral"]["spectral_len"] = 300
    config["model"]["smodel_type"] = "LanczosLaplace"

    assert explicit_id(server) == PRE_CHANGE_IDS[data_type]


@pytest.mark.parametrize("data_type", sorted(PRE_CHANGE_IDS))
def test_a_config_predating_the_knobs_keeps_its_identity(config, tmp_path, data_type):
    server = identity_server(config, tmp_path, data_type)
    config["spectral"]["spectral_len"] = 300
    config["model"]["smodel_type"] = "LanczosLaplace"
    del config["spectral"]["coverage_drop"]
    del config["spectral"]["window_floor"]

    assert explicit_id(server) == PRE_CHANGE_IDS[data_type]


@pytest.mark.parametrize("data_type", ["f+s", "f+pe", "f+es"])
def test_the_identity_separates_every_coverage_fraction(config, tmp_path, data_type):
    server = identity_server(config, tmp_path, data_type)

    ids = {}
    for p in (0.0, 0.25, 0.5, 0.75, 0.85):
        config["spectral"]["coverage_drop"] = p
        ids[p] = server._run_id()

    assert len(set(ids.values())) == 5
    assert all(f"cov{p:g}" in ids[p].split("_") for p in ids if p)
    assert not any(part.startswith("cov") for part in ids[0.0].split("_"))


@pytest.mark.parametrize("data_type", ["f+s", "f+pe", "f+es"])
def test_the_identity_separates_a_floored_window_from_a_hard_one(
    config, tmp_path, data_type
):
    server = identity_server(config, tmp_path, data_type)
    config["spectral"]["cum_decay"] = "window"
    config["spectral"]["cum_decay_param"] = 1

    ids = {}
    for floor in (0.0, 0.01, 0.1, 0.5):
        config["spectral"]["window_floor"] = floor
        ids[floor] = server._run_id()

    assert len(set(ids.values())) == 4
    assert "cum-window1" in ids[0.0].split("_")
    assert not any(part.startswith("wfl") for part in ids[0.0].split("_"))
    assert "wfl0.01" in ids[0.01].split("_")


def test_a_coverage_arm_cannot_adopt_the_untreated_arms_checkpoint(config, tmp_path):
    # under auto_resume a shared identity means one arm silently inherits the
    # other's result -- and the control would then BE the treatment.
    identity_server(config, tmp_path, "f+es")
    snaps = make_toy_snapshots(num_snaps=2)
    server = make_server(snaps, partition_snapshots(snaps, 1))
    server.initialize_FL()
    server._save_partial_ckpt(0, None, [], [])
    server._save_done_ckpt({"mean_mrr": 0.0, "mrr_history": []})

    config["spectral"]["coverage_drop"] = 0.5
    assert server._load_partial_ckpt() is None
    assert server._load_done_ckpt() is None

    config["spectral"]["coverage_drop"] = 0.0
    config["spectral"]["cum_decay"] = "window"
    config["spectral"]["cum_decay_param"] = 1
    config["spectral"]["window_floor"] = 0.01
    assert server._load_partial_ckpt() is None

    # Every knob back to what it was when the checkpoint was written -- including
    # cum_decay_param, which the 'none' kernel ignores. The completeness
    # fingerprint hashes config bytes, so an inert value still has to be restored.
    config["spectral"]["cum_decay"] = "none"
    config["spectral"]["cum_decay_param"] = 0.9
    config["spectral"]["window_floor"] = 0.0
    assert server._load_partial_ckpt() is not None


def test_neither_knob_touches_a_non_spectral_identity(config, tmp_path):
    # data_type=feature never builds a basis, so neither knob can change its
    # numbers; giving it its own id would re-run banked arms for nothing
    server = identity_server(config, tmp_path, "feature")
    config["spectral"]["coverage_drop"] = 0.5
    config["spectral"]["cum_decay"] = "window"
    config["spectral"]["window_floor"] = 0.5

    assert explicit_id(server) == PRE_CHANGE_IDS["feature"]


# --------------------------------------------------------------------- #
# 8. the two properties results.md 21.3 rests on, asserted directly
# --------------------------------------------------------------------- #


def cached_bases(config, data_type, update_mode, solver, drop, procrustes="auto"):
    """The (first, prev) caches after every snapshot -- what the tracker reads."""
    snaps = snapshots(DENSE_SCHEDULE, DENSE_NODES)
    config["model"]["data_type"] = data_type
    config["spectral"]["update_mode"] = update_mode
    config["spectral"]["solver"] = solver
    config["spectral"]["coverage_drop"] = drop
    config["spectral"]["use_procrustes"] = procrustes
    seed_all(42)
    server = make_server(snaps, partition_snapshots(snaps, 1))
    server.initialize_FL()
    out = []
    for t in range(len(snaps)):
        server._spectral_step(t, config["model"]["smodel_type"])
        out.append((server._first_spectral, server._prev_spectral))
    return out


@pytest.mark.parametrize("data_type,update_mode,solver,proc", PIPELINE_ARMS)
def test_the_solved_basis_the_tracker_caches_never_sees_the_mask(
    config, data_type, update_mode, solver, proc
):
    # The property that makes coverage_drop a control rather than a second
    # treatment. _first_spectral anchors Procrustes and freezes the `keep` basis;
    # _prev_spectral warm-starts the next Chebyshev solve (X0) and feeds the next
    # Rayleigh-Ritz step (prev_Q). If the mask reached either, the arm would
    # change the SPECTRUM as well as the coverage -- exactly the confound the
    # change exists to remove -- and it would compound across snapshots.
    setup_basis_config(config)
    plain = cached_bases(config, data_type, update_mode, solver, 0.0, proc)
    dropped = cached_bases(config, data_type, update_mode, solver, 0.75, proc)

    for t, ((fa, pa), (fb, pb)) in enumerate(zip(plain, dropped)):
        for name, a, b in (("first", fa, fb), ("prev", pa, pb)):
            assert (a is None) == (b is None), f"t={t} {name}"
            if a is None:
                continue
            assert torch.equal(a.U, b.U), f"t={t}: _{name}_spectral.U moved with the mask"
            assert torch.equal(a.D, b.D), f"t={t}: _{name}_spectral.D moved with the mask"
            assert (a.Q is None) == (b.Q is None), f"t={t} {name}.Q"
            if a.Q is not None:
                assert torch.equal(a.Q, b.Q), f"t={t}: _{name}_spectral.Q moved with the mask"
        # and the cached basis really is the SOLVED one, not a masked copy that
        # happens to match: at 0.75 the served basis must have lost rows the
        # cache still carries
        assert int(zero_rows(pb.U).sum()) < DENSE_NODES


def test_the_pair_fraction_counts_negatives_as_well_as_positives(
    config, monkeypatch
):
    # edge_label_index is positives THEN sampled negatives. Counting only the
    # positive half would report a different denominator than the metric the
    # attribution is about -- every candidate the ranker had to order.
    server, metrics, served, batches = instrumented_run(
        config, 0.5, num_snaps=3, monkeypatch=monkeypatch
    )
    U, eli, m = served[0], batches[0], metrics[0]
    n_pos = int(server.global_snaps[1].pos_test.size(1))

    assert eli.size(1) > n_pos                       # negatives are in the batch
    z = zero_rows(U)
    touched = z[eli[0]] | z[eli[1]]
    assert m["basis_zeroed_pair_frac"] == pytest.approx(float(touched.float().mean()))
    # the positives-only reading is a different number, so the assertion above
    # genuinely discriminates between them
    pos_only = float(touched[:n_pos].float().mean())
    assert m["basis_zeroed_pair_frac"] != pytest.approx(pos_only)
