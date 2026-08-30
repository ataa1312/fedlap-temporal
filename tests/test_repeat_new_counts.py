"""Per-snapshot sample counts behind metric.repeat_new_split.

mrr_repeat/mrr_new are means over SOURCES, and a source is SKIPPED (not zeroed)
in a subset where it has no positive, so the denominator is neither the number
of positives nor the number of sources -- and a source holding one positive of
each kind belongs to BOTH denominators. n_* and src_* exist so a run-level
mrr_new can be re-weighted or bootstrapped afterwards; if they are not the
metric's real numerator and denominator they are worse than nothing, because
they look like they are. These pin the accounting against _rank_and_aggregate's
actual skip logic, the empty-subset edges, the mean over snapshots, the log
line, and main's RESULT provenance.
"""

import copy
import math
import re

import pytest
import torch
from torch_geometric.data import Data

import main
import src
import src.dynamic_server as dynamic_server
from src.dynamic_server import DynamicServer, _weighted_mean_metrics
from src.metrics.mrr import _rank_and_aggregate, compute_mrr_splits_from_z
from src.train.federated_orchestrator import _partition_edges_per_snapshot
from src.utils.graph_partitioning import partition_snapshots
from test_checkpoint_wandb import make_server, seed_all
from test_deterministic_flag import instrument, run_main, write_cfg
from test_lifetime_and_drop_knobs import build_server, tiny_run


SPLIT_KEYS = ("mrr_repeat", "mrr_new", "repeat_frac", "n_repeat", "n_new",
              "src_repeat", "src_new")


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


class _Recorder:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(str(msg))


# --------------------------------------------------------------------- #
# 1. the denominator _rank_and_aggregate actually uses
# --------------------------------------------------------------------- #

# source 0 owns one repeat AND one new positive, source 1 only a repeat,
# source 2 only new ones -- so |repeat sources| + |new sources| = 4 > 3 sources.
U = torch.tensor([0, 0, 1, 2, 2])
RMASK = torch.tensor([True, False, True, False, False])
UNIQUE = torch.tensor([0, 1, 2])
K = 4
POS_SCORES = torch.tensor([0.9, 0.1, 0.5, 0.7, 0.3])
NEG_SCORES = torch.tensor([
    [0.2, 0.4, 0.6, 0.8],
    [0.0, 0.2, 0.4, 0.6],
    [0.1, 0.2, 0.3, 0.4],
])
SCORES = torch.cat([POS_SCORES, NEG_SCORES.reshape(-1)])


def aggregate(keep):
    return _rank_and_aggregate(SCORES, U, UNIQUE, UNIQUE.numel(), K, "max", keep=keep)


def per_source(keep):
    """The subset's value computed one source at a time -- the sources that
    survive ARE the denominator, by the function's own skip rule."""
    vals = []
    for s in UNIQUE.tolist():
        v = aggregate(keep & (U == s))
        if v == v:
            vals.append(v)
    return vals


@pytest.mark.parametrize("subset", ["repeat", "new"])
def test_src_counts_are_the_denominator_the_metric_averages_over(subset):
    keep = RMASK if subset == "repeat" else ~RMASK
    server_count = int(torch.unique(U[keep]).numel())

    vals = per_source(keep)

    assert len(vals) == server_count
    assert aggregate(keep) == pytest.approx(sum(vals) / len(vals))


def test_a_source_with_one_of_each_kind_is_in_both_denominators():
    # the semantics compute_mrr_splits_from_z gets from its `keep` mask: source 0
    # is ranked in the repeat subset AND in the new subset, on its OWN positives.
    assert aggregate(RMASK & (U == 0)) == pytest.approx(1.0)     # 0.9 beats every negative
    assert aggregate(~RMASK & (U == 0)) == pytest.approx(0.2)    # 0.1 beats none

    n_src_repeat = int(torch.unique(U[RMASK]).numel())
    n_src_new = int(torch.unique(U[~RMASK]).numel())

    assert (n_src_repeat, n_src_new) == (2, 2)
    assert n_src_repeat + n_src_new > UNIQUE.numel()   # not a partition of the sources
    assert n_src_new != UNIQUE.numel() - n_src_repeat  # the tempting wrong formula


def test_a_source_absent_from_a_subset_is_skipped_not_zeroed():
    only_two = torch.zeros_like(RMASK)
    only_two[3] = True

    assert aggregate(only_two) == pytest.approx(1.0)  # source 2 alone, rank 1
    assert math.isnan(aggregate(torch.zeros_like(RMASK)))


def test_an_all_repeat_mask_reproduces_the_unsplit_number():
    stub = _stub_model()
    snap = _tiny_eval_snap()
    full, rep, new = compute_mrr_splits_from_z(
        torch.zeros(snap.num_nodes, 2), snap, 4, "max", "cpu", stub,
        torch.ones(int((snap.edge_label == 1.0).sum()), dtype=torch.bool),
    )

    assert full == pytest.approx(rep)
    assert math.isnan(new)


class _StubModel:
    def decode(self, z, data):
        # strictly decreasing in the target id, so ranks are deterministic
        s = -data.edge_label_index[1].to(torch.float32)
        return s, data.edge_label


def _stub_model():
    return _StubModel()


def _tiny_eval_snap():
    snap = Data(x=torch.ones(12, 1), edge_index=torch.tensor([[0, 1], [1, 2]]), num_nodes=12)
    snap.edge_label_index = torch.tensor([[0, 0, 1], [1, 2, 3]])
    snap.edge_label = torch.tensor([1.0, 1.0, 1.0])
    return snap


# --------------------------------------------------------------------- #
# 2. the server's counts, against the mask the metric was handed
# --------------------------------------------------------------------- #


N = 10


def snaps_from(schedule, n_nodes=N):
    out = []
    for edges in schedule:
        ei = torch.tensor(edges, dtype=torch.long).t()
        snap = Data(x=torch.ones(n_nodes, 1), edge_index=ei,
                    edge_attr=torch.ones(ei.size(1), 1), num_nodes=n_nodes)
        snap.node_ids = torch.arange(n_nodes)
        out.append(snap)
    return out


def eval_one(config, schedule):
    """_eval_mrr at t=0 with EVERY edge of snapshot 1 in the test split, so the
    positive set is exactly the schedule and the counts are hand-checkable."""
    tiny_run(config, "feature")
    config["metric"]["repeat_new_split"] = True
    config["subgraph"]["num_subgraphs"] = 1
    config["experimental"]["rank_eval_multiplier"] = 5
    seed_all(42)
    snaps = snaps_from(schedule)
    server = make_server(snaps, partition_snapshots(snaps, 1))
    server.initialize_FL()
    _partition_edges_per_snapshot(server.global_snaps, [0.0, 0.0, 1.0], 42)
    for cl in server.clients:
        _partition_edges_per_snapshot(cl.snaps, [0.0, 0.0, 1.0], 43)
    server._accumulate_cum_edges(0)
    mrr, metrics = server._eval_mrr(0, 5, "min")
    return server, mrr, metrics


PAST = [(0, 1), (2, 3), (4, 5)]


def test_a_source_holding_both_kinds_is_counted_twice(config):
    # THE case the whole instrumentation turns on: (0,1) repeats, (0,7) is new,
    # both belong to source 0. src_repeat + src_new must exceed the source count.
    _, _, m = eval_one(config, [PAST, [(0, 1), (0, 7)]])

    assert (m["n_repeat"], m["n_new"]) == (1, 1)
    assert (m["src_repeat"], m["src_new"]) == (1, 1)
    assert m["src_repeat"] + m["src_new"] == 2 > 1  # one distinct source
    assert not math.isnan(m["mrr_repeat"]) and not math.isnan(m["mrr_new"])


def test_the_counts_match_the_mask_the_metric_was_given(config, monkeypatch):
    # recomputed from the ACTUAL (positives, mask) pair _eval_mrr passed to the
    # metric, by an independent per-column loop.
    tiny_run(config, "feature")
    config["metric"]["repeat_new_split"] = True
    seen = []
    original = DynamicServer._repeat_mask

    def spy(self, pos_edges):
        mask = original(self, pos_edges)
        seen.append((pos_edges.detach().cpu().clone(), mask.detach().cpu().clone()))
        return mask

    monkeypatch.setattr(DynamicServer, "_repeat_mask", spy)
    server = build_server()

    results = server.joint_train_w(FL=True)

    assert len(seen) == len(results["metrics_history"]) > 0
    for (pos, mask), m in zip(seen, results["metrics_history"]):
        rep_src = {int(pos[0][i]) for i in range(pos.size(1)) if bool(mask[i])}
        new_src = {int(pos[0][i]) for i in range(pos.size(1)) if not bool(mask[i])}
        assert m["n_repeat"] == sum(1 for i in range(pos.size(1)) if bool(mask[i]))
        assert m["n_new"] == sum(1 for i in range(pos.size(1)) if not bool(mask[i]))
        assert m["n_repeat"] + m["n_new"] == pos.size(1)
        assert m["src_repeat"] == len(rep_src)
        assert m["src_new"] == len(new_src)
        assert m["src_repeat"] <= m["n_repeat"] and m["src_new"] <= m["n_new"]
        assert m["repeat_frac"] == pytest.approx(m["n_repeat"] / pos.size(1))
    # the positives are the test split of the NEXT snapshot, nothing else
    for t, (pos, _) in enumerate(seen):
        assert pos.size(1) == server.global_snaps[t + 1].pos_test.size(1)
    # and the run is not a degenerate all-repeat or all-new one
    assert any(m["n_new"] for m in results["metrics_history"])
    assert any(m["n_repeat"] for m in results["metrics_history"])


@pytest.mark.parametrize(
    "future,repeat_side",
    [([(0, 1), (2, 3)], True), ([(0, 6), (1, 7), (8, 9)], False)],
)
def test_an_empty_subset_reports_zero_and_a_nan_not_a_zero_metric(config, future, repeat_side):
    # a 0 there would be averaged into the run as a real (terrible) MRR
    _, mrr, m = eval_one(config, [PAST, future])

    empty, full = ("new", "repeat") if repeat_side else ("repeat", "new")
    assert m[f"n_{empty}"] == 0 and m[f"src_{empty}"] == 0
    assert math.isnan(m[f"mrr_{empty}"])
    assert m[f"n_{full}"] == len(future) and m[f"src_{full}"] > 0
    assert not math.isnan(m[f"mrr_{full}"])
    assert m["repeat_frac"] == pytest.approx(1.0 if repeat_side else 0.0)
    assert not math.isnan(mrr)


# --------------------------------------------------------------------- #
# 3. the run-level mean
# --------------------------------------------------------------------- #


def run_split(config, on):
    tiny_run(config, "feature")
    config["metric"]["repeat_new_split"] = on
    server = build_server()
    return server.joint_train_w(FL=True)


def test_the_split_keys_survive_the_weighted_mean(config):
    results = run_split(config, True)

    assert len(results["metrics_history"]) > 1
    for snap_metrics in results["metrics_history"]:
        assert all(k in snap_metrics for k in SPLIT_KEYS)
    for k in SPLIT_KEYS:
        assert k in results["mean_metrics"]
    counts = [m["n_repeat"] for m in results["metrics_history"]]
    assert results["mean_metrics"]["n_repeat"] == pytest.approx(sum(counts) / len(counts))


def test_the_split_keys_are_absent_when_the_split_is_off(config):
    results = run_split(config, False)

    assert len(results["metrics_history"]) > 1
    for snap_metrics in results["metrics_history"]:
        assert not any(k in snap_metrics for k in SPLIT_KEYS)
    assert not any(k in results["mean_metrics"] for k in SPLIT_KEYS)
    assert "roc_auc" in results["mean_metrics"]  # the rest of the dict is untouched


def test_a_nan_subset_drops_out_of_the_mean_instead_of_poisoning_it():
    metrics = [
        {"mrr_new": float("nan"), "n_new": 0.0},
        {"mrr_new": 0.4, "n_new": 2.0},
        {"mrr_new": 0.6, "n_new": 4.0},
    ]

    out = _weighted_mean_metrics(metrics, [1.0, 1.0, 1.0])

    assert out["mrr_new"] == pytest.approx(0.5)   # over the two real snapshots
    assert out["n_new"] == pytest.approx(2.0)     # counts average over all three


def test_the_mean_tolerates_a_snapshot_that_lacks_a_key():
    # was a KeyError: _weighted_mean_metrics indexed every dict with
    # metrics_list[0]'s keys. metric.repeat_new_split now separates the run
    # identity, but a checkpoint written before that separation still restores a
    # metrics_history whose dicts disagree on keys, and the crash landed at the
    # very END of the run, after all the compute.
    out = _weighted_mean_metrics(
        [{"mrr": 0.2, "mrr_new": 0.5}, {"mrr": 0.4}], [1.0, 1.0]
    )

    assert out["mrr"] == pytest.approx(0.3)
    assert out["mrr_new"] == pytest.approx(0.5)   # averaged over the dict that has it


def test_the_mean_tolerates_the_key_appearing_only_later():
    # the reverse resume order: split OFF first, then ON. Reading only
    # metrics_list[0] silently DROPPED the key here instead of raising.
    out = _weighted_mean_metrics(
        [{"mrr": 0.2}, {"mrr": 0.4, "mrr_new": 0.5}], [1.0, 1.0]
    )

    assert out["mrr"] == pytest.approx(0.3)
    assert out["mrr_new"] == pytest.approx(0.5)


def test_a_partial_key_averages_over_exactly_the_dicts_that_have_it():
    out = _weighted_mean_metrics(
        [{"z": 1.0}, {"z": 3.0, "p": 2.0}, {"z": 5.0, "p": 6.0}], [1.0, 1.0, 1.0]
    )

    assert out["z"] == pytest.approx(3.0)
    assert out["p"] == pytest.approx(4.0)   # not 8/3: the absent dict is not a 0


def test_an_all_nan_key_stays_nan_and_the_weighting_is_unchanged():
    nan = float("nan")

    assert math.isnan(_weighted_mean_metrics([{"q": nan}, {"q": nan}], [1.0, 1.0])["q"])
    assert math.isnan(_weighted_mean_metrics([{"a": 1.0}], [0.0])["a"])
    assert _weighted_mean_metrics([], []) == {}
    # fully populated: the weights still do the work they always did
    assert _weighted_mean_metrics([{"a": 1.0}, {"a": 3.0}], [3.0, 1.0])["a"] == pytest.approx(1.5)
    assert _weighted_mean_metrics([{"a": nan}, {"a": 3.0}], [3.0, 1.0])["a"] == pytest.approx(3.0)


def test_the_key_order_is_deterministic_across_processes():
    # a set comprehension over the union would order keys by PYTHONHASHSEED, so
    # mean_metrics -- and the done-checkpoint that stores it -- would differ
    # between two identical runs. First-seen order also keeps the fully-populated
    # case identical to metrics_list[0]'s order.
    import subprocess
    import sys

    first = {"roc_auc": 1.0, "ap": 1.0, "mrr_repeat": 1.0, "n_new": 1.0}
    assert list(_weighted_mean_metrics([first, {"ap": 2.0, "zz": 1.0}], [1.0, 1.0])) == [
        "roc_auc", "ap", "mrr_repeat", "n_new", "zz"
    ]
    snippet = (
        "import sys; sys.path.insert(0, '.');"
        "from src.dynamic_server import _weighted_mean_metrics as W;"
        "print(list(W([{'roc_auc':1.0,'ap':1.0,'mrr_repeat':1.0,'n_new':1.0},"
        "{'ap':2.0,'zz':1.0}], [1.0,1.0])))"
    )
    orders = {
        subprocess.run([sys.executable, "-c", snippet], capture_output=True,
                       text=True, check=True).stdout.strip()
        for _ in range(4)
    }
    assert len(orders) == 1


# --------------------------------------------------------------------- #
# 3b. run identity
# --------------------------------------------------------------------- #


def test_the_split_separates_the_run_identity(config, tmp_path):
    # the split draws its own negatives, so a split run is NOT the run a
    # split-off launch produces; sharing a _run_id meant sharing a checkpoint and
    # a deterministic wandb id with a different experiment.
    from test_run_identity import set_identity_config

    set_identity_config(config, tmp_path)
    config["model"]["data_type"] = "feature"
    server = DynamicServer(_toy())

    config["metric"]["repeat_new_split"] = False
    off, off_wandb = server._run_id(), server._wandb_id()
    config["metric"]["repeat_new_split"] = True
    on, on_wandb = server._run_id(), server._wandb_id()

    from test_run_identity import explicit_id_of

    assert explicit_id_of(off) == "uci_gru_feature_C1_s1234"  # the readable arm
    assert on != off and "split" in on
    assert on_wandb != off_wandb


def test_a_split_run_cannot_adopt_a_split_off_checkpoint(config, tmp_path):
    from test_run_identity import set_identity_config

    set_identity_config(config, tmp_path)
    config["model"]["data_type"] = "feature"
    config["metric"]["repeat_new_split"] = False
    server = DynamicServer(_toy())
    server._save_done_ckpt({"mean_mrr": 0.0, "mrr_history": []})

    config["metric"]["repeat_new_split"] = True
    assert server._load_done_ckpt() is None    # the split run must actually run

    config["metric"]["repeat_new_split"] = False
    assert server._load_done_ckpt() is not None  # and the old one still resolves


def _toy():
    from test_deterministic_flag import make_toy_snapshots

    return make_toy_snapshots(num_snaps=2)


# 4. the per-snapshot log line
# --------------------------------------------------------------------- #

BASE_LINE = r"t=\d+ mrr=\S+ auc=\S+ ap=\S+ f1=\S+ mcc=\S+"
SPLIT_TAIL = (r" mrr_repeat=\S+ mrr_new=\S+ n_repeat=\d+ n_new=\d+"
              r" src_repeat=\d+ src_new=\d+")


def snapshot_lines(config, on, monkeypatch):
    tiny_run(config, "feature")
    config["metric"]["repeat_new_split"] = on
    rec = _Recorder()
    monkeypatch.setattr(dynamic_server, "LOGGER", rec)
    server = build_server()
    results = server.joint_train_w(FL=True, log=True)
    return [l for l in rec.lines if l.startswith("t=")], results


def test_the_log_line_carries_the_counts_when_the_split_is_on(config, monkeypatch):
    lines, results = snapshot_lines(config, True, monkeypatch)

    assert len(lines) == len(results["metrics_history"]) > 0
    for line, m in zip(lines, results["metrics_history"]):
        assert re.fullmatch(BASE_LINE + SPLIT_TAIL, line), line
        assert f" n_repeat={m['n_repeat']} n_new={m['n_new']}" in line
        assert f" src_repeat={m['src_repeat']} src_new={m['src_new']}" in line


def test_the_log_line_is_unchanged_when_the_split_is_off(config, monkeypatch):
    lines, _ = snapshot_lines(config, False, monkeypatch)

    assert lines
    for line in lines:
        assert re.fullmatch(BASE_LINE, line), line  # nothing appended


# --------------------------------------------------------------------- #
# 5. main's RESULT provenance
# --------------------------------------------------------------------- #


def result_line(monkeypatch, tmp_path, server_cls=None):
    instrument(monkeypatch)   # fakes the dataset, the partition and the server
    if server_cls is not None:
        monkeypatch.setattr(main, "DynamicServer", server_cls)
    rec = _Recorder()
    monkeypatch.setattr(main, "LOGGER", rec)
    run_main(monkeypatch, write_cfg(tmp_path))
    return next(l for l in rec.lines if l.startswith("RESULT "))


def test_result_reports_a_real_run_id(config, tmp_path, monkeypatch):
    class _IdServer:
        def __init__(self, global_snaps):
            pass

        def add_client(self, snaps):
            pass

        def _load_done_ckpt(self):
            return None

        def _run_id(self):
            return "uci_gru_feature_C1_s1234"

        def joint_train_w(self, FL=True, log_cb=None):
            return {"mean_mrr": 0.0, "std_mrr": 0.0, "mrr_history": [], "mean_metrics": {}}

    line = result_line(monkeypatch, tmp_path, _IdServer)

    assert line.endswith("run=uci_gru_feature_C1_s1234")


def test_result_does_not_die_on_a_server_without_a_run_id(config, tmp_path, monkeypatch):
    # tests/test_deterministic_flag.py drives main() with exactly such a double;
    # provenance must never be the thing that breaks a run.
    line = result_line(monkeypatch, tmp_path)

    assert line.endswith("run=unavailable")
    assert "mean_mrr=" in line


@pytest.mark.xfail(
    strict=True,
    reason="DEFECT (identity half fixed, divergence half open): "
    "compute_mrr_splits_from_z draws its own negatives from the GLOBAL rng, so "
    "switching the instrumentation on shifts every later draw and the run "
    "diverges from t=1 onward -- the diagnostic changes the number it is "
    "diagnosing, and a split run's mean_mrr is not comparable to a split-off "
    "one's at fixed seed. _run_id now separates the two, so at least they no "
    "longer share a checkpoint. Fixing this needs a forked generator for the "
    "split draw, which would move every banked split number.",
)
def test_turning_the_split_on_does_not_move_the_headline_mrr(config):
    off = run_split(config, False)["mrr_history"]
    on = run_split(config, True)["mrr_history"]

    assert off[0] == pytest.approx(on[0])   # t=0 precedes the extra draw
    assert off == on


# --------------------------------------------------------------------- #
# 6. device placement of the repeat mask
# --------------------------------------------------------------------- #


class _DeviceSpy(torch.Tensor):
    """Records every .to() the repeat mask is asked for."""

    calls = []

    def to(self, *a, **k):
        _DeviceSpy.calls.append(a[0] if a else k.get("device"))
        return torch.Tensor.to(self, *a, **k)


def test_the_repeat_mask_is_moved_to_the_index_tensors_device(config, monkeypatch):
    # _repeat_mask builds its mask on CPU while the positives live on `device`.
    # This box is CPU-only so a mismatch cannot be exercised here; what CAN be
    # asserted is that the mask is explicitly moved before it indexes anything --
    # once by the metric and once by the counting block. Drop either move and the
    # count falls.
    original = DynamicServer._repeat_mask

    def spy(self, pos_edges):
        return original(self, pos_edges).as_subclass(_DeviceSpy)

    monkeypatch.setattr(DynamicServer, "_repeat_mask", spy)
    _DeviceSpy.calls = []
    server, _, m = eval_one(config, [PAST, [(0, 1), (0, 7)]])

    assert m["n_repeat"] + m["n_new"] == 2       # the block really ran
    assert len(_DeviceSpy.calls) == 2
    assert all(torch.device(c) == dynamic_server.device for c in _DeviceSpy.calls)
