import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.assertions import assert_cfg
from src.utils.graph import Graph
from src.GNN.dynamic_classifier import DynamicClassifier
from src.metrics.mrr import (
    TARGET_EDGES_ATTR,
    _extra_forbidden,
    _sample_filtered_negatives,
    compute_mrr,
    compute_mrr_from_z,
)

N = 400
SRCS = (0, 1)
# target-snapshot edges (10..49) split into the evaluated test positives (10..13)
# and the train/val positives the split filter leaves eligible (14..49); today's
# message-passing edges (50..89) are disjoint from both and must stay drawable.
TARGET_V = range(10, 50)
POS_V = range(10, 14)
OTHER_V = range(14, 50)
TODAY_V = range(50, 90)


def _edges(vs):
    return torch.tensor([[s, v] for s in SRCS for v in vs], dtype=torch.long).t()


def _keys(edge_index):
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


def _make_snap(with_target=True):
    pos = _edges(POS_V)
    snap = Graph(x=torch.ones(N, 1), edge_index=_edges(TODAY_V), node_ids=torch.arange(N))
    snap.edge_label_index = pos
    snap.edge_label = torch.ones(pos.size(1))
    if with_target:
        snap.target_edge_index = _edges(TARGET_V)
    return snap


class _RecordingHead(nn.Module):
    """Scores everything zero and keeps the batch the metric built, so the
    sampled negatives can be inspected through the real metric path."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.seen = []

    def decode(self, z, data):
        self.seen.append(data.edge_label_index.clone())
        return torch.zeros(data.edge_label_index.size(1)), data.edge_label

    def forward(self, data):
        return self.decode(None, data)


def _negatives(model, snap, K, mrr_filter, seed=0):
    torch.manual_seed(seed)
    z = torch.zeros(snap.num_nodes, 4)
    compute_mrr_from_z(z, snap, K, "max", "cpu", model, mrr_filter)
    n_pos = int((snap.edge_label == 1.0).sum())
    return model.seen[-1][:, n_pos:]


@pytest.fixture
def mrr_filter(config):
    def _set(value):
        config["metric"]["mrr_filter"] = value

    yield _set
    config["metric"]["mrr_filter"] = "split"


# --- 1. strict forbids every edge of the target snapshot, not just the split --- #

def test_snapshot_filter_excludes_all_target_splits():
    snap = _make_snap()
    model = _RecordingHead()
    K = 600

    strict = _negatives(model, snap, K, "snapshot")
    loose = _negatives(model, snap, K, "split")

    target, other, today = _keys(_edges(TARGET_V)), _keys(_edges(OTHER_V)), _keys(_edges(TODAY_V))
    assert not (_keys(strict) & target)
    # the split filter only knows the evaluated positives, so the target's
    # train/val edges are still drawn -- without this the test above is vacuous
    assert _keys(loose) & other
    # neither filter touches today's edges: they are legitimate hard negatives
    assert _keys(strict) & today
    assert _keys(loose) & today


def test_compute_mrr_threads_the_filter():
    snap = _make_snap()
    model = _RecordingHead()
    n_pos = snap.edge_label_index.size(1)

    torch.manual_seed(0)
    compute_mrr(model, snap, None, 600, "max", False, "cpu", "snapshot")
    strict = model.seen[-1][:, n_pos:]

    assert not (_keys(strict) & _keys(_edges(TARGET_V)))
    assert _keys(strict) & _keys(_edges(TODAY_V))


# --- 2. both filters keep exactly K negatives per source (resample, not drop) --- #

def test_both_filters_keep_exactly_k_per_source():
    snap = _make_snap()
    model = _RecordingHead()
    K = 600
    n_sources = len(SRCS)

    for mode in ("split", "snapshot"):
        neg = _negatives(model, snap, K, mode)
        assert neg.size(1) == n_sources * K
        # the block is read back as (n_sources, K), so the source column must
        # come out as K repeats of each source in order
        srcs = neg[0].view(n_sources, K)
        for i, s in enumerate(SRCS):
            assert torch.equal(srcs[i], torch.full((K,), s))


def test_strict_resamples_rather_than_discards():
    # half of every candidate is forbidden, so the bounded rounds cannot clear
    # the block; a discard-based implementation would return fewer than K
    K = 2000
    forbidden = torch.stack(
        [torch.zeros(N // 2, dtype=torch.long), torch.arange(N // 2)]
    )
    empty = torch.empty(0, dtype=torch.long)
    out = _sample_filtered_negatives(
        torch.tensor([0]), empty, empty, N, K, "cpu", forbidden
    )
    assert out.shape == (1, K)


# --- 3. the bounded 8-round guarantee --- #

def test_bounded_rounds_returns_the_block_as_is(monkeypatch):
    # every candidate is forbidden -> no draw can ever clear
    n, K = 20, 16
    forbidden = torch.stack([torch.zeros(n, dtype=torch.long), torch.arange(n)])
    empty = torch.empty(0, dtype=torch.long)

    calls = []
    real_randint = torch.randint

    def counting(*args, **kwargs):
        calls.append(1)
        return real_randint(*args, **kwargs)

    monkeypatch.setattr(torch, "randint", counting)
    out = _sample_filtered_negatives(
        torch.tensor([0]), empty, empty, n, K, "cpu", forbidden
    )

    assert out.shape == (1, K)
    assert len(calls) == 9  # one initial draw + at most 8 resample rounds
    # returned as-is: still fully contaminated, neither raised nor dropped
    assert bool(torch.isin(out.flatten(), torch.arange(n)).all())


# --- 4. the target edge set is attached, never inferred from the batch --- #

def test_extra_forbidden_picks_the_target_attribute():
    snap = _make_snap()

    assert _extra_forbidden(snap, "split") is None
    for mode in ("snapshot", "both"):
        got = _extra_forbidden(snap, mode)
        assert got is getattr(snap, TARGET_EDGES_ATTR)
        assert not torch.equal(got, snap.edge_index)


def test_missing_target_falls_back_to_split():
    bare = _make_snap(with_target=False)
    assert not hasattr(bare, TARGET_EDGES_ATTR)
    assert _extra_forbidden(bare, "snapshot") is None
    assert _extra_forbidden(bare, "both") is None

    model = _RecordingHead()
    strict = _negatives(model, bare, 600, "snapshot")
    loose = _negatives(model, bare, 600, "split")
    # identical draws from the same seed == the same forbidden set
    assert torch.equal(strict, loose)
    # and the fallback is to `split`, not to filtering against the batch's own
    # (today's) edge_index
    assert _keys(strict) & _keys(_edges(TODAY_V))


def test_attach_carries_the_target_edge_set():
    from src.train.federated_orchestrator import _attach_future_link_pred_labels

    today = Graph(x=torch.ones(N, 1), edge_index=_edges(TODAY_V), node_ids=torch.arange(N))
    tomorrow = Graph(x=torch.ones(N, 1), edge_index=_edges(TARGET_V), node_ids=torch.arange(N))

    snap = _attach_future_link_pred_labels(today, tomorrow, tomorrow.edge_index[:, :4])

    assert torch.equal(snap.target_edge_index, tomorrow.edge_index)
    assert torch.equal(snap.edge_index, today.edge_index)
    assert not torch.equal(snap.edge_index, snap.target_edge_index)


# --- 5. the knob is gated at configuration time --- #

def test_assert_cfg_gates_mrr_filter(config):
    for value in ("split", "snapshot", "both"):
        config["metric"]["mrr_filter"] = value
        assert_cfg(config)

    for bad in ("Split", "snapshots", "all", "", None, 1):
        config["metric"]["mrr_filter"] = bad
        with pytest.raises(ValueError) as exc:
            assert_cfg(config)
        assert "metric.mrr_filter" in str(exc.value)

    # tolerant of configs predating the key
    del config["metric"]["mrr_filter"]
    try:
        assert_cfg(config)
    finally:
        config["metric"]["mrr_filter"] = "split"


# --- 6. the default is the pre-knob behaviour --- #

def test_default_is_split(config):
    from config.config import get_default_config
    from src.train.federated_orchestrator import _mrr_filter_mode

    assert get_default_config()["metric"]["mrr_filter"] == "split"
    assert _mrr_filter_mode() == "split"

    # configs predating the key must still resolve to the default
    del config["metric"]["mrr_filter"]
    try:
        assert _mrr_filter_mode() == "split"
    finally:
        config["metric"]["mrr_filter"] = "split"


def test_default_arg_matches_explicit_split():
    snap = _make_snap()
    model = _RecordingHead()
    K = 600

    torch.manual_seed(11)
    z = torch.zeros(N, 4)
    compute_mrr_from_z(z, snap, K, "max", "cpu", model)
    default = model.seen[-1]

    torch.manual_seed(11)
    compute_mrr_from_z(z, snap, K, "max", "cpu", model, "split")
    assert torch.equal(default, model.seen[-1])

    torch.manual_seed(11)
    compute_mrr_from_z(z, snap, K, "max", "cpu", model, "snapshot")
    assert not torch.equal(default, model.seen[-1])


def test_no_extra_forbidden_is_bit_identical():
    # with the source's positives unreachable by any draw for that source there
    # can be no collision, so the default path must return the raw draw and
    # leave the RNG stream exactly where a single randint would
    n, K = 64, 32
    src = torch.tensor([0])
    pos_u, pos_v = torch.tensor([5]), torch.tensor([7])

    torch.manual_seed(1234)
    ref = torch.randint(0, n, (1, K))
    ref_state = torch.get_rng_state()

    torch.manual_seed(1234)
    out = _sample_filtered_negatives(src, pos_u, pos_v, n, K, "cpu")
    assert torch.equal(out, ref)
    assert torch.equal(torch.get_rng_state(), ref_state)

    torch.manual_seed(1234)
    empty = _sample_filtered_negatives(
        src, pos_u, pos_v, n, K, "cpu", torch.empty(2, 0, dtype=torch.long)
    )
    assert torch.equal(empty, ref)


# --- paired reporting under `both` --- #

def _tiny_pair(seed):
    g = torch.Generator().manual_seed(seed)
    edges = set()
    while len(edges) < 30:
        u = torch.randint(0, 12, (1,), generator=g).item()
        v = torch.randint(0, 12, (1,), generator=g).item()
        if u != v:
            edges.add((u, v))
    edge_index = torch.tensor(list(edges), dtype=torch.long).t()
    return Graph(
        x=torch.ones(12, 1),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), 1, generator=g),
        node_ids=torch.arange(12),
    )


def test_step_eval_reports_both_arms(config, mrr_filter):
    from src.train.federated_orchestrator import (
        _partition_edges_per_snapshot,
        _step_eval_with_mrr_pair,
    )

    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = False

    s0, s1 = _tiny_pair(12), _tiny_pair(34)
    _partition_edges_per_snapshot([s0, s1], [0.8, 0.1, 0.1], seed=1234)
    dyn = DynamicClassifier(s0)

    def run():
        torch.manual_seed(5)
        return _step_eval_with_mrr_pair(
            dyn, s0, s1, None, F.binary_cross_entropy_with_logits,
            "cpu", True, 200, "min",
        )

    mrr_filter("split")
    _, mrr_split, m_split = run()
    mrr_filter("snapshot")
    _, mrr_strict, m_strict = run()
    mrr_filter("both")
    _, mrr_both, m_both = run()

    assert "mrr_snapshot" not in m_split
    assert "mrr_snapshot" not in m_strict
    assert "mrr_snapshot" in m_both
    assert 0.0 <= m_both["mrr_snapshot"] <= 1.0
    assert 0.0 <= mrr_strict <= 1.0
    # the headline under `both` is still the split arm, from the same draw
    assert mrr_both == mrr_split
    # NB: `both`'s strict arm draws its negatives from a different RNG state
    # than a standalone `snapshot` run, so the two are NOT expected to match --
    # that second independent draw is what buys the equal pool size.


def test_step_eval_runs_under_no_grad(config):
    # guards a regression this change introduced and then fixed: _mrr_filter_mode
    # was inserted between @torch.no_grad() and _step_eval_with_mrr_pair, orphaning
    # the decorator onto the config helper so the eval step built an autograd graph.
    from src.train.federated_orchestrator import (
        _partition_edges_per_snapshot,
        _step_eval_with_mrr_pair,
    )

    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = False

    s0, s1 = _tiny_pair(12), _tiny_pair(34)
    _partition_edges_per_snapshot([s0, s1], [0.8, 0.1, 0.1], seed=1234)
    dyn = DynamicClassifier(s0)

    seen = {}
    real_decode = dyn.decode

    def spy(z, data):
        out = real_decode(z, data)
        seen.setdefault("grad_enabled", torch.is_grad_enabled())
        seen.setdefault("built_graph", out[0].grad_fn is not None)
        return out

    dyn.decode = spy
    try:
        _step_eval_with_mrr_pair(
            dyn, s0, s1, None, F.binary_cross_entropy_with_logits,
            "cpu", True, 50, "min",
        )
    finally:
        dyn.decode = real_decode

    assert seen, "decode was never reached"
    assert seen["grad_enabled"] is False
    assert seen["built_graph"] is False


def test_server_global_eval_reports_both(config, mrr_filter):
    # the dynamic_server path (metric.eval_scope -> global) is what the reported
    # runs use, and it carries its own `both` branch in _eval_mrr
    from torch_geometric.data import Data
    from src.dynamic_server import DynamicServer
    from src.utils.graph_partitioning import partition_snapshots

    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["model"]["loss_fun"] = "bce_with_logits"
    config["model"]["iterations"] = 2
    config["model"]["local_epochs"] = 1
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = False
    config["metric"]["mrr_method"] = "min"
    config["metric"]["eval_scope"] = "global"
    config["experimental"]["rank_eval_multiplier"] = 50
    config["optim"]["optimizer"] = "adam"
    config["optim"]["base_lr"] = 0.005
    config["optim"]["scheduler"] = "none"

    def run():
        torch.manual_seed(42)
        snaps = []
        for _ in range(4):
            g = torch.Generator().manual_seed(len(snaps) + 7)
            edges = set()
            while len(edges) < 16:
                u = torch.randint(0, 8, (1,), generator=g).item()
                v = torch.randint(0, 8, (1,), generator=g).item()
                if u != v:
                    edges.add((u, v))
            ei = torch.tensor(list(edges), dtype=torch.long).t()
            snap = Data(
                x=torch.ones(8, 1),
                edge_index=ei,
                edge_attr=torch.randn(ei.size(1), 1, generator=g),
                num_nodes=8,
            )
            snap.node_ids = torch.arange(8)
            snaps.append(snap)
        server = DynamicServer(snaps)
        for cs in partition_snapshots(snaps, 1):
            server.add_client(cs)
        return server.joint_train_w()

    mrr_filter("split")
    plain = run()
    mrr_filter("both")
    paired = run()

    assert plain["metrics_history"]
    assert all("mrr_snapshot" not in m for m in plain["metrics_history"])
    assert "mrr_snapshot" not in plain["mean_metrics"]

    assert len(paired["metrics_history"]) == len(plain["metrics_history"])
    assert all("mrr_snapshot" in m for m in paired["metrics_history"])
    # main.py reads the paired arm off mean_metrics to build the RESULT line
    assert 0.0 <= paired["mean_metrics"]["mrr_snapshot"] <= 1.0
    # The first eval precedes any extra sampling, so the headline there is the
    # untouched split arm. It does NOT stay equal afterwards: the strict arm's
    # resampling draws are taken from the run's global RNG, so from t=1 on a
    # `both` run trains on a shifted stream. The delta stays paired (both arms
    # come from that same run) but a `both` run's headline is not comparable to
    # a `split` run's.
    assert paired["mrr_history"][0] == plain["mrr_history"][0]
