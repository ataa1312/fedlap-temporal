"""The per-snapshot record and the aggregation that reads it.

The record exists because every headline number means over SNAPSHOTS first and
only then over seeds, which cannot tell a uniform effect from one carried by a
handful of snapshots, and cannot test whether a deficit is a warm-up artifact of
the sparse early cumulative graph. That makes the record load-bearing in a way
the numbers it copies are not: it is the raw material every later claim is
recomputed from, so a dropped snapshot, a filled-in absent key or two arms
sharing one file is worse than no record at all -- it looks exactly like a
measurement. These pin the writer against the values the run actually reported,
the arm key against every identity shape _run_id can emit, and each aggregation
against a hand computation.
"""

import copy
import json
import logging
import math
import os
import statistics as st
import sys

import pytest
import torch
from torch_geometric.data import Data

import main
import registries
import src
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer, _weighted_mean_metrics
from test_checkpoint_wandb import seed_all
from test_checkpoint_wandb import make_toy_snapshots as sparse_snapshots
from test_edge_score_smodel import set_fes_model_config
from test_fl_local_baseline import make_toy_snapshots as dense_snapshots
from test_lifetime_and_drop_knobs import tiny_run

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
import aggregate_snapshots as agg


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #

IDENTITY_KEYS = ("run_id", "arm", "seed", "t")


def record_cfg(cfg, tmp_path, data_type="feature"):
    """A run small enough to execute inside a test, writing its record to tmp."""
    tiny_run(cfg, data_type=data_type)
    cfg["subgraph"]["num_subgraphs"] = 1
    cfg["wandb"]["mode"] = "disabled"
    cfg["metric"]["snapshot_record_dir"] = str(tmp_path)
    return cfg


def run_seeds(cfg, monkeypatch, snaps, seeds=(42,)):
    """Drive main.run_once() once per seed, as main()'s repeat loop does."""
    original = registries.datasets["uci"]
    registries.datasets["uci"] = lambda c: snaps
    try:
        out = []
        for s in seeds:
            cfg["seed"] = s
            out.append(main.run_once())
        return out
    finally:
        registries.datasets["uci"] = original


def record_files(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))


def read_rows(d, name=None):
    files = record_files(d)
    if name is None:
        assert len(files) == 1, f"expected one record file, got {files}"
        name = files[0]
    with open(os.path.join(d, name)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def metric_only(row):
    return {k: v for k, v in row.items() if k not in IDENTITY_KEYS and k != "mrr"}


def rows_to_records(rows):
    """The aggregator's load() shape, built directly so a test needs no file."""
    out = {}
    for r in rows:
        out.setdefault(r["t"], {})[r["seed"]] = r
    return out


def curve_of(vals_by_t):
    """{t: [per-seed values]} -> the aggregator's per-snapshot input."""
    rows = []
    for t, vals in vals_by_t.items():
        for s, v in enumerate(vals):
            rows.append({"arm": "a", "seed": s, "t": t, "m": v})
    return rows_to_records(rows)


# --------------------------------------------------------------------- #
# 1. the arm key: _run_id minus its seed suffix
#    Requirement: Per-Snapshot Per-Seed Metric Record Is Persisted
# --------------------------------------------------------------------- #


def identity_cfg(cfg):
    cfg["dataset"]["name"] = "uci"
    cfg["dataset"]["snapshot_freq"] = "W"
    cfg["gnn"]["embed_update_method"] = "gru"
    cfg["model"]["data_type"] = "feature"
    cfg["subgraph"]["num_subgraphs"] = 1
    cfg["seed"] = 1234
    cfg["experimental"]["deterministic"] = False


# Every shape _run_id can emit that puts an "_s" INSIDE the arm, which is what a
# naive left-to-right split on "_s" would truncate at: the 'structure' data type,
# the sfv/smodel/solver tokens, the sfv-lifetime flags and the split token.
IDENTITY_SHAPES = {
    "backbone": {},
    "structure_dt": {("model", "data_type"): "structure", ("spectral", "update_mode"): "keep"},
    "fs_sfv_and_smodel": {("model", "data_type"): "f+s", ("spectral", "update_mode"): "keep",
                          ("federated", "sfv_share"): "avg"},
    "fes_esf": {("model", "data_type"): "f+es", ("spectral", "update_mode"): "keep"},
    "shuffled_basis": {("model", "data_type"): "f+s", ("spectral", "update_mode"): "keep",
                       ("spectral", "basis_source"): "shuffled_fixed"},
    "sfvreset": {("model", "data_type"): "f+s", ("spectral", "update_mode"): "keep",
                 ("structure_model", "sfv_reset_per_snapshot"): True},
    "sfvfrozen": {("model", "data_type"): "f+s", ("spectral", "update_mode"): "keep",
                  ("structure_model", "freeze_sfv"): True},
    "split_on": {("metric", "repeat_new_split"): True},
    "solver": {("model", "data_type"): "f+es", ("spectral", "update_mode"): "keep",
               ("spectral", "solver"): "chebyshev"},
    "local_scope": {("federated", "fl"): False, ("metric", "eval_scope"): "local"},
    "deterministic": {("experimental", "deterministic"): True},
}


def written_arm(server, cfg, d):
    """The arm the WRITER actually keys the file by, not a re-derivation of it."""
    cfg["metric"]["snapshot_record_dir"] = str(d)
    main._write_snapshot_record(server, {"mrr_history": [0.5], "metrics_history": [{}],
                                        "t_history": [0]})
    files = record_files(d)
    assert len(files) == 1, files
    os.remove(os.path.join(d, files[0]))
    return files[0][: -len(".jsonl")]


@pytest.mark.parametrize("shape", sorted(IDENTITY_SHAPES))
@pytest.mark.parametrize("seed", [1, 42, 1234, 100200])
def test_the_arm_key_strips_exactly_the_seed_suffix(config, tmp_path, shape, seed):
    # The property, not a hand-written expected string: whatever _run_id emits,
    # the arm must be the identity with ONLY "_s<seed>" removed. Anything that
    # eats an arm token collapses two arms into one file.
    identity_cfg(config)
    for (sect, key), val in IDENTITY_SHAPES[shape].items():
        config[sect][key] = val
    config["seed"] = seed
    server = DynamicServer(sparse_snapshots())

    rid = server._run_id()
    arm = written_arm(server, config, tmp_path)

    assert rid == f"{arm}_s{seed}"
    assert arm == rid[: -len(f"_s{seed}")]


@pytest.mark.parametrize("shape", ["structure_dt", "fs_sfv_and_smodel", "sfvreset", "split_on"])
def test_the_arm_key_survives_an_underscore_s_inside_the_arm(config, shape):
    # the specific failure mode: 'structure', '_sfv-avg', '_sfvreset', '_split'
    # all contain "_s", so a left-to-right split truncates the arm and two
    # different arms land in one file.
    identity_cfg(config)
    for (sect, key), val in IDENTITY_SHAPES[shape].items():
        config[sect][key] = val
    server = DynamicServer(sparse_snapshots())

    rid = server._run_id()
    assert rid.count("_s") > 1, f"{shape} does not exercise the case: {rid}"
    assert rid.rsplit("_s", 1)[0] == rid[: -len("_s1234")]


def test_all_seeds_of_one_arm_share_one_file(config, tmp_path, monkeypatch):
    record_cfg(config, tmp_path)
    seed_all(42)
    run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7), seeds=(42, 142, 242))

    assert len(record_files(tmp_path)) == 1
    assert sorted({r["seed"] for r in read_rows(tmp_path)}) == [42, 142, 242]


KNOBS_IN_THE_IDENTITY = [
    (("metric", "repeat_new_split"), True),
    (("metric", "eval_scope"), "global"),
    (("gnn", "encoder_edge_drop"), 0.5),
    (("experimental", "deterministic"), True),
    (("spectral", "cum_decay"), "count"),
]


@pytest.mark.parametrize("path,val", KNOBS_IN_THE_IDENTITY)
def test_two_arms_differing_only_in_a_knob_do_not_collide(config, tmp_path, monkeypatch, path, val):
    record_cfg(config, tmp_path, data_type="f+es")
    set_fes_model_config(config, pe_dim=4)
    config["spectral"]["update_mode"] = "keep"
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    config["wandb"]["mode"] = "disabled"
    config["subgraph"]["num_subgraphs"] = 1
    seed_all(42)
    snaps = dense_snapshots(N=30, num_snaps=4, seed=7)

    run_seeds(config, monkeypatch, snaps)
    config[path[0]][path[1]] = val
    run_seeds(config, monkeypatch, snaps)

    assert len(record_files(tmp_path)) == 2


# --------------------------------------------------------------------- #
# 2. the writer: entry count, identity, and agreement with the run
#    Requirement: Per-Snapshot Per-Seed Metric Record Is Persisted
# --------------------------------------------------------------------- #


def test_one_entry_per_seed_and_snapshot(config, tmp_path, monkeypatch):
    record_cfg(config, tmp_path)
    seed_all(42)
    res = run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7),
                    seeds=(42, 142, 242))
    rows = read_rows(tmp_path)

    S = len(res[0]["mrr_history"])
    assert S == 3
    assert len(rows) == 3 * S
    assert sorted((r["seed"], r["t"]) for r in rows) == sorted(
        (s, t) for s in (42, 142, 242) for t in range(S)
    )


def test_every_entry_carries_its_own_run_id_and_seed(config, tmp_path, monkeypatch):
    record_cfg(config, tmp_path)
    seed_all(42)
    run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7), seeds=(42, 142))

    for r in read_rows(tmp_path):
        assert r["run_id"] == f"{r['arm']}_s{r['seed']}"


@pytest.mark.parametrize("data_type,split", [("feature", False), ("f+es", True)])
def test_the_persisted_values_are_the_values_the_run_reported(
    config, tmp_path, monkeypatch, data_type, split
):
    # The contract that makes the record usable at all. Exact equality, not
    # approx: the record is a copy, and a copy that needs a tolerance is a
    # second measurement.
    record_cfg(config, tmp_path, data_type="feature")
    if data_type == "f+es":
        set_fes_model_config(config, pe_dim=4)
        config["spectral"]["update_mode"] = "keep"
        config["subgraph"]["num_subgraphs"] = 1
    config["metric"]["repeat_new_split"] = split
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    config["wandb"]["mode"] = "disabled"
    seed_all(42)
    res = run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=5, seed=7),
                    seeds=(42, 142))[0]
    rows = [r for r in read_rows(tmp_path) if r["seed"] == 42]

    assert [r["mrr"] for r in rows] == res["mrr_history"]
    assert st.fmean(r["mrr"] for r in rows) == res["mean_mrr"]
    recomputed = _weighted_mean_metrics([metric_only(r) for r in rows], [1.0] * len(rows))
    assert recomputed == res["mean_metrics"]


def test_the_entries_are_in_snapshot_order(config, tmp_path, monkeypatch):
    # aggregation by index is only sound if the index is the writing order; a
    # reordered record recomputes the same mean and a different curve.
    record_cfg(config, tmp_path)
    seed_all(42)
    res = run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=6, seed=7))[0]
    rows = read_rows(tmp_path)

    assert [r["t"] for r in rows] == list(range(len(res["mrr_history"])))
    for r in rows:
        assert r["mrr"] == res["mrr_history"][r["t"]]


def test_no_snapshot_is_dropped(config, tmp_path, monkeypatch):
    record_cfg(config, tmp_path)
    seed_all(42)
    res = run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=6, seed=7))[0]

    assert len(read_rows(tmp_path)) == len(res["mrr_history"]) == 5


# --------------------------------------------------------------------- #
# 3. ragged keys: absent is absent, not filled
# --------------------------------------------------------------------- #


SUBSET_KEYS = ("mrr_repeat", "mrr_new", "repeat_frac", "n_repeat", "n_new",
               "src_repeat", "src_new")
COVERAGE_KEYS = ("basis_covered", "basis_zeroed", "basis_total", "basis_zeroed_pair_frac")


def test_the_backbone_record_has_no_coverage_or_subset_keys(config, tmp_path, monkeypatch):
    record_cfg(config, tmp_path)
    seed_all(42)
    run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7))

    for r in read_rows(tmp_path):
        for k in COVERAGE_KEYS + SUBSET_KEYS:
            assert k not in r, f"{k} present with value {r.get(k)!r} on a backbone run"


def test_a_spectral_split_run_carries_both_families(config, tmp_path, monkeypatch):
    record_cfg(config, tmp_path, data_type="feature")
    set_fes_model_config(config, pe_dim=4)
    config["spectral"]["update_mode"] = "keep"
    config["subgraph"]["num_subgraphs"] = 1
    config["metric"]["repeat_new_split"] = True
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    config["wandb"]["mode"] = "disabled"
    seed_all(42)
    run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7))

    for r in read_rows(tmp_path):
        for k in COVERAGE_KEYS + SUBSET_KEYS:
            assert k in r


def test_the_split_keys_are_absent_when_the_split_is_off(config, tmp_path, monkeypatch):
    record_cfg(config, tmp_path, data_type="feature")
    set_fes_model_config(config, pe_dim=4)
    config["spectral"]["update_mode"] = "keep"
    config["subgraph"]["num_subgraphs"] = 1
    config["metric"]["repeat_new_split"] = False
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    config["wandb"]["mode"] = "disabled"
    seed_all(42)
    run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7))

    rows = read_rows(tmp_path)
    for r in rows:
        for k in SUBSET_KEYS:
            assert k not in r
        for k in COVERAGE_KEYS:
            assert k in r


def test_the_record_key_set_is_exactly_the_snapshots_key_set(config, tmp_path, monkeypatch):
    # no key invented and none lost: the row is identity + mrr + the snapshot's
    # own metrics dict, nothing else.
    record_cfg(config, tmp_path, data_type="feature")
    set_fes_model_config(config, pe_dim=4)
    config["spectral"]["update_mode"] = "keep"
    config["subgraph"]["num_subgraphs"] = 1
    config["metric"]["repeat_new_split"] = True
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    config["wandb"]["mode"] = "disabled"
    seed_all(42)
    res = run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7))[0]

    for r, m in zip(read_rows(tmp_path), res["metrics_history"]):
        assert set(r) == set(IDENTITY_KEYS) | {"mrr"} | set(m)
        for k, v in m.items():
            assert r[k] == v or (v != v and r[k] != r[k])


# --------------------------------------------------------------------- #
# 4. durability, destination, and the tracking-disabled path
# --------------------------------------------------------------------- #


def test_the_record_is_written_with_tracking_disabled(config, tmp_path, monkeypatch):
    # the normal case for these runs: wandb off, and the record is then the ONLY
    # durable form of the snapshot axis.
    record_cfg(config, tmp_path)
    seen = []
    monkeypatch.setattr(main, "_init_wandb", lambda: seen.append(1))
    seed_all(42)
    run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7))

    assert seen == [1]                       # wandb init returned None
    assert len(read_rows(tmp_path)) == 3


def test_each_seed_is_on_disk_before_the_next_one_starts(config, tmp_path, monkeypatch):
    # D3: a run that dies on seed 3 leaves seeds 1-2 behind.
    record_cfg(config, tmp_path)
    seed_all(42)
    snaps = dense_snapshots(N=30, num_snaps=4, seed=7)
    seen = []

    original = registries.datasets["uci"]
    calls = {"n": 0}

    def loader(c):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("seed 3 died")
        seen.append(len(read_rows(tmp_path)) if record_files(tmp_path) else 0)
        return snaps

    registries.datasets["uci"] = loader
    try:
        for i, s in enumerate((42, 142, 242)):
            config["seed"] = s
            if i == 2:
                with pytest.raises(RuntimeError):
                    main.run_once()
            else:
                main.run_once()
    finally:
        registries.datasets["uci"] = original

    assert seen == [0, 3]                    # seed 1 was already on disk for seed 2
    rows = read_rows(tmp_path)
    assert len(rows) == 6
    assert sorted({r["seed"] for r in rows}) == [42, 142]


def test_the_output_directory_knob_is_honoured(config, tmp_path, monkeypatch):
    record_cfg(config, tmp_path)
    elsewhere = tmp_path / "records"
    config["metric"]["snapshot_record_dir"] = str(elsewhere)
    seed_all(42)
    run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7))

    assert record_files(tmp_path) == []
    assert len(read_rows(str(elsewhere))) == 3


@pytest.mark.parametrize("absent", [True, False])
def test_the_default_destination_is_resolved_at_run_time(config, tmp_path, monkeypatch, absent):
    # It must NOT be the import-time save_path: that is built before main()
    # overlays the YAML, so dataset.name is still None there and every run in the
    # program would land in one shared 'default' directory whatever the dataset.
    record_cfg(config, tmp_path)
    if absent:                               # a config predating the key
        del config["metric"]["snapshot_record_dir"]
    else:
        config["metric"]["snapshot_record_dir"] = None
    monkeypatch.chdir(tmp_path)
    seed_all(42)
    run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=4, seed=7))

    assert len(read_rows(str(tmp_path / "results" / "uci" / "snapshots"))) == 3


def test_the_default_destination_tracks_the_dataset(config, tmp_path, monkeypatch):
    # the fixed defect: the import-time save_path names 'default' regardless of
    # the run, so two datasets shared a directory whose name described neither.
    record_cfg(config, tmp_path)
    del config["metric"]["snapshot_record_dir"]
    monkeypatch.chdir(tmp_path)
    seed_all(42)
    snaps = dense_snapshots(N=30, num_snaps=4, seed=7)
    original = registries.datasets["uci"]
    registries.datasets["uci"] = lambda c: snaps
    try:
        config["seed"] = 42
        main.run_once()
        config["dataset"]["name"] = "as733"
        registries.datasets["as733"] = lambda c: snaps
        main.run_once()
    finally:
        registries.datasets["uci"] = original

    assert (tmp_path / "results" / "uci" / "snapshots").is_dir()
    assert (tmp_path / "results" / "as733" / "snapshots").is_dir()
    assert "default" not in os.listdir(tmp_path / "results")


# --------------------------------------------------------------------- #
# 5. rulings on the silent-no-op class
#    Three times now a knob has read as a real negative result while doing
#    nothing (the cn arm degrading to feature-only, coverage_drop inert on
#    persist/cn, the cached-basis mask). The writer is the same shape of risk:
#    it can produce nothing, or a partial row, without the run noticing.
# --------------------------------------------------------------------- #


class _Recorder:
    """LOGGER is non-propagating, so caplog sees nothing it emits; a test that
    asserts on caplog here passes whatever the writer does."""

    def __init__(self):
        self.info_lines, self.warn_lines = [], []

    def info(self, msg):
        self.info_lines.append(msg)

    def warning(self, msg):
        self.warn_lines.append(msg)


def write_with_logger(monkeypatch, server, results, **kw):
    rec = _Recorder()
    monkeypatch.setattr(main, "LOGGER", rec)
    main._write_snapshot_record(server, results, **kw)
    return rec


def test_logger_lines_do_not_reach_caplog(config, tmp_path, caplog, monkeypatch):
    # the premise for every assertion below: LOGGER does not propagate, so the
    # obvious `assert ... in caplog.text` is vacuous and would never fail.
    identity_cfg(config)
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    caplog.set_level(logging.DEBUG)
    server = DynamicServer(sparse_snapshots())

    main._write_snapshot_record(server, {"mrr_history": [0.5], "metrics_history": [{}],
                                         "t_history": [0]})

    assert len(read_rows(tmp_path)) == 1     # it really did write
    assert "snapshot record" not in caplog.text


def test_an_empty_history_writes_no_file_and_says_so(config, tmp_path, monkeypatch):
    # the silent-no-op class, closed: a run whose every snapshot was skipped
    # leaves no record, and the absence is now announced rather than inferred
    # from a log line that was never printed.
    identity_cfg(config)
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    server = DynamicServer(sparse_snapshots())

    rec = write_with_logger(monkeypatch, server,
                            {"mrr_history": [], "metrics_history": [], "t_history": []})

    assert record_files(tmp_path) == []
    assert any("NOT WRITTEN" in m and "no snapshots" in m for m in rec.warn_lines)


def test_a_metrics_history_shorter_than_the_mrr_history_is_announced(config, tmp_path, monkeypatch):
    # the two histories are appended under separate guards in joint_train_w and
    # the writer joins them BY POSITION, so a short metrics_history yields
    # metric-free rows. That is tolerable -- the mrr axis is still sound -- but
    # only because it is now said out loud.
    identity_cfg(config)
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    server = DynamicServer(sparse_snapshots())

    rec = write_with_logger(monkeypatch, server, {
        "mrr_history": [0.1, 0.2, 0.3],
        "metrics_history": [{"roc_auc": 0.9}],
        "t_history": [0, 1, 2],
    })
    rows = read_rows(tmp_path)

    assert len(rows) == 3
    assert "roc_auc" in rows[0]
    assert "roc_auc" not in rows[1] and "roc_auc" not in rows[2]
    assert any("metrics_history 1 != mrr_history 3" in m for m in rec.warn_lines)


class _Bare:
    """A server double lacking _run_id, as tests/test_deterministic_flag.py uses."""


def test_a_server_without_a_run_id_does_not_kill_the_seed_loop(config, tmp_path, monkeypatch):
    # main's RESULT line guards this exact case ("never let it break a run, and
    # keep test doubles that lack it usable"); the writer three lines below now
    # guards it too. Before, a double with ONE snapshot killed the run after
    # training, taking every later seed of a --repeat sweep with it.
    config["metric"]["snapshot_record_dir"] = str(tmp_path)

    rec = write_with_logger(monkeypatch, _Bare(),
                            {"mrr_history": [0.5], "metrics_history": [{}], "t_history": [0]})

    assert record_files(tmp_path) == []
    assert any("no _run_id" in m for m in rec.warn_lines)


def test_the_writer_never_raises_into_the_seed_loop(config, tmp_path, monkeypatch):
    # every failure mode reachable from a completed run, including a broken
    # logger: the record is reporting, and reporting must not destroy a seed.
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    good = {"mrr_history": [0.5], "metrics_history": [{}], "t_history": [0]}

    class _Explodes:
        def info(self, msg):
            raise RuntimeError("logger blew up")

        warning = info

    class _Mute:
        pass

    for logger in (_Explodes(), _Mute()):
        monkeypatch.setattr(main, "LOGGER", logger)
        main._write_snapshot_record(_Bare(), good)
        main._write_snapshot_record(_Bare(), None)


def test_the_error_handler_cannot_raise_out_of_itself(config, tmp_path, monkeypatch):
    # the bug this guard exists for: the handler called LOGGER.warning, so on a
    # logger lacking it the handler died and the exception escaped the try. A
    # getattr DEFAULT is evaluated eagerly, so falling back to LOGGER.info is not
    # enough either -- a logger with neither must still be survivable.
    config["metric"]["snapshot_record_dir"] = str(tmp_path)

    class _InfoOnly:
        def __init__(self):
            self.lines = []

        def info(self, msg):
            self.lines.append(msg)

    rec = _InfoOnly()
    monkeypatch.setattr(main, "LOGGER", rec)
    main._write_snapshot_record(_Bare(), {"mrr_history": [0.5], "metrics_history": [{}],
                                          "t_history": [0]})

    assert any("no _run_id" in m for m in rec.lines)   # it degraded to info


# --------------------------------------------------------------------- #
# 6. known defects, pinned so a fix is noticed
# --------------------------------------------------------------------- #


def blanked_snapshots(n=5, blank=2):
    """Snapshots with one EDGELESS entry, whose pos_test is empty, so the eval
    at the preceding loop index returns None and that snapshot is skipped."""
    snaps = dense_snapshots(N=30, num_snaps=n, seed=7)
    ref = snaps[blank]
    empty = Data(x=ref.x, edge_index=torch.zeros(2, 0, dtype=torch.long),
                 edge_attr=torch.zeros(0, ref.edge_attr.size(1)), num_nodes=ref.num_nodes)
    empty.node_ids = ref.node_ids
    snaps[blank] = empty
    return snaps


def test_the_recorded_t_is_the_snapshot_the_metrics_came_from(config, tmp_path, monkeypatch):
    # WAS a defect: mrr_history is appended only when the eval returns a finite
    # MRR, so position in the list is not the snapshot index. Relabelling
    # survivors 0..n-1 compressed the axis, putting the early/late split on the
    # wrong snapshots and pairing different snapshots across seeds.
    record_cfg(config, tmp_path)
    seed_all(42)
    evaluated = []
    original = DynamicServer._eval_mrr

    def spy(self, t, *a, **kw):
        mrr, metrics = original(self, t, *a, **kw)
        if mrr is not None and not math.isnan(mrr):
            evaluated.append(t)
        return mrr, metrics

    monkeypatch.setattr(DynamicServer, "_eval_mrr", spy)
    run_seeds(config, monkeypatch, blanked_snapshots())

    assert evaluated == [0, 2, 3]                      # the premise: t=1 was skipped
    assert [r["t"] for r in read_rows(tmp_path)] == evaluated


def test_relaunching_a_finished_run_does_not_duplicate_its_rows(config, tmp_path, monkeypatch):
    # WAS a defect: the done-checkpoint short-circuit returns the stored history
    # and the writer appended it again. load() dedupes last-write-wins, so a
    # rerun silently overwrote banked numbers instead of being caught.
    record_cfg(config, tmp_path)
    config["train"]["auto_resume"] = True
    config["train"]["ckpt_period"] = 1
    config["train"]["ckpt_clean"] = True
    config["train"]["ckpt_dir"] = str(tmp_path / "ckpt")
    seed_all(42)
    snaps = dense_snapshots(N=30, num_snaps=4, seed=7)

    res = run_seeds(config, monkeypatch, snaps)[0]
    again = run_seeds(config, monkeypatch, snaps)[0]

    assert again.get("_resumed_complete") is True      # the premise
    assert len(read_rows(tmp_path)) == len(res["mrr_history"])


@pytest.mark.parametrize("path,value", [
    (("metric", "mrr_filter"), "snapshot"),
    (("metric", "hard_neg"), "degree"),
    (("model", "iterations"), 3),
    (("optim", "base_lr"), 0.05),
    (("experimental", "rank_eval_multiplier"), 3),
    (("dataset", "split"), [0.7, 0.2, 0.1]),
])
def test_a_knob_no_token_records_still_gets_its_own_file(
    config, tmp_path, monkeypatch, path, value
):
    # WAS D3: these knobs change the recorded numbers and no explicit token
    # carries them, so two arms wrote to ONE file under identical (arm, seed, t)
    # keys -- not merely mixed, indistinguishable, with load() keeping whichever
    # was written last. The cfg-<hash> backstop in _run_id separates them without
    # anyone having to remember to add a token, which is what produced D3.
    record_cfg(config, tmp_path)
    seed_all(42)
    snaps = dense_snapshots(N=30, num_snaps=4, seed=7)

    run_seeds(config, monkeypatch, snaps)
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    run_seeds(config, monkeypatch, snaps)

    assert len(record_files(tmp_path)) == 2


def test_the_arm_key_keeps_the_fingerprint(config, tmp_path):
    # the strip must remove the SEED suffix and nothing else: dropping cfg- from
    # the arm key would put the D3 collision straight back into the record while
    # leaving the checkpoints correctly separated -- the worst of both.
    identity_cfg(config)
    server = DynamicServer(sparse_snapshots())

    arm = written_arm(server, config, tmp_path)

    assert any(p.startswith("cfg-") for p in arm.split("_"))
    assert server._run_id() == f"{arm}_s{config['seed']}"


# --------------------------------------------------------------------- #
# 7. per_snapshot: the transpose, and what an undefined value does to it
#    Requirement: Cross-Seed Per-Snapshot Aggregation
# --------------------------------------------------------------------- #


NAN = float("nan")


def test_per_snapshot_reports_mean_spread_and_seed_count(config):
    rows = curve_of({0: [0.1, 0.3, 0.5], 1: [1.0, 1.0, 1.0]})

    c = agg.per_snapshot(rows, "m")

    assert c[0] == (0.3, pytest.approx(math.sqrt(0.08 / 3)), 3)
    assert c[1] == (1.0, 0.0, 3)


def test_a_nan_in_one_seed_reduces_the_count_and_not_the_mean(config):
    # the D6/E-shaped error: averaging nan as 0 would give 0.2 here, and a
    # snapshot backed by two seeds would read as one backed by three.
    rows = curve_of({0: [NAN, 0.2, 0.4]})

    (mean, spread, n), = agg.per_snapshot(rows, "m").values()

    assert (mean, n) == (pytest.approx(0.3), 2)
    assert spread == pytest.approx(0.1)


def test_a_missing_key_is_treated_as_undefined_not_as_zero(config):
    rows = rows_to_records([
        {"arm": "a", "seed": 0, "t": 0, "m": 0.4},
        {"arm": "a", "seed": 1, "t": 0},                 # the ragged case
    ])

    assert agg.per_snapshot(rows, "m")[0] == (0.4, 0.0, 1)


def test_a_snapshot_undefined_in_every_seed_leaves_the_curve(config):
    rows = curve_of({0: [0.4, 0.6], 1: [NAN, NAN], 2: [0.5, 0.5]})

    c = agg.per_snapshot(rows, "m")

    assert sorted(c) == [0, 2]


def test_the_curve_is_ordered_by_snapshot_index(config):
    rows = curve_of({5: [0.5], 0: [0.0], 2: [0.2]})

    assert list(agg.per_snapshot(rows, "m")) == [0, 2, 5]


# --------------------------------------------------------------------- #
# 8. distribution: over SNAPSHOTS, never over seeds
#    Requirement: Distributional Summary Over Snapshots
# --------------------------------------------------------------------- #


# Three quantities that are all different here, which is the point:
#   per-snapshot means      [0.5, 0.5, 0.9]  -> median 0.5
#   raw rows pooled         [0, 0, .9, .9, 1, 1] -> median 0.9
#   per-seed cross-snapshot [0.3, 0.9667]    -> median 0.6333
SPREAD_CASE = {0: [0.0, 1.0], 1: [0.0, 1.0], 2: [0.9, 0.9]}


def test_the_summary_is_over_snapshots_not_over_seeds(config):
    d = agg.distribution(agg.per_snapshot(curve_of(SPREAD_CASE), "m"))

    assert d["n_snapshots"] == 3
    assert d["median"] == pytest.approx(0.5)
    assert d["mean"] == pytest.approx((0.5 + 0.5 + 0.9) / 3)
    assert d["min"] == pytest.approx(0.5) and d["max"] == pytest.approx(0.9)


def test_quantiles_accompany_the_mean(config):
    rows = curve_of({t: [float(t)] for t in range(8)})

    d = agg.distribution(agg.per_snapshot(rows, "m"))

    assert set(d) == {"n_snapshots", "mean", "median", "q1", "q3", "min", "max"}
    assert d["min"] == 0.0 and d["max"] == 7.0 and d["median"] == 3.5
    assert d["min"] < d["q1"] < d["median"] < d["q3"] < d["max"]


@pytest.mark.parametrize("n_snapshots,quartiles_are_extremes", [(2, True), (3, True), (4, False)])
def test_the_small_sample_fallback_reports_min_and_max_as_the_quartiles(
    config, n_snapshots, quartiles_are_extremes
):
    # the len(v) > 3 branch is a DIFFERENT code path: at or below three snapshots
    # it labels the extremes q1/q3, so a 3-snapshot IQR is the full range and
    # must not be read as an IQR. Pinned so the boundary cannot drift silently.
    rows = curve_of({t: [float(t)] for t in range(n_snapshots)})

    d = agg.distribution(agg.per_snapshot(rows, "m"))

    assert (d["q1"] == d["min"] and d["q3"] == d["max"]) is quartiles_are_extremes
    assert d["median"] == st.median(range(n_snapshots))


def test_an_empty_curve_summarises_to_nothing(config):
    assert agg.distribution({}) is None


# --------------------------------------------------------------------- #
# 9. weighted: every pair gets a say, and an empty snapshot gets none
#    Requirement: Pair-Weighted Aggregation Of Subset Metrics
# --------------------------------------------------------------------- #


def weight_rows(triples, metric="mrr_new", wkey="n_new"):
    return rows_to_records([
        {"arm": "a", "seed": 0, "t": t, metric: m, wkey: n}
        for t, (m, n) in enumerate(triples)
    ])


def test_the_two_aggregations_agree_under_equal_counts(config):
    rows = weight_rows([(0.1, 10), (0.2, 10), (0.6, 10)])

    unweighted = agg.distribution(agg.per_snapshot(rows, "mrr_new"))["mean"]

    assert agg.weighted(rows, "mrr_new") == pytest.approx(unweighted)
    assert agg.weighted(rows, "mrr_new") == pytest.approx(0.3)


def test_the_two_aggregations_diverge_under_unequal_counts(config):
    # the as733 shape: two snapshots carrying one new positive each and one
    # carrying a hundred. Every pair gets a say -> 2/102; every snapshot gets a
    # say -> 2/3.
    rows = weight_rows([(1.0, 1), (1.0, 1), (0.0, 100)])

    unweighted = agg.distribution(agg.per_snapshot(rows, "mrr_new"))["mean"]

    assert unweighted == pytest.approx(2 / 3)
    assert agg.weighted(rows, "mrr_new") == pytest.approx(2 / 102)


@pytest.mark.parametrize("empty_value", [0.0, NAN, 999.0])
def test_a_snapshot_with_no_positives_cannot_move_the_weighted_mean(config, empty_value):
    # D6: excluded, not counted as a zero. Whatever a snapshot with n=0 carries
    # -- a filler zero, a nan, or garbage -- the aggregate is the one computed
    # from the snapshots that have positives.
    with_empty = weight_rows([(0.5, 10), (empty_value, 0), (0.1, 10)])
    without = weight_rows([(0.5, 10), (0.1, 10)])

    assert agg.weighted(with_empty, "mrr_new") == agg.weighted(without, "mrr_new")
    assert agg.weighted(with_empty, "mrr_new") == pytest.approx(0.3)


def test_a_nan_metric_is_excluded_from_the_weighted_mean(config):
    rows = weight_rows([(0.5, 10), (NAN, 7), (0.1, 10)])

    assert agg.weighted(rows, "mrr_new") == pytest.approx(0.3)


def test_the_weighted_mean_pools_every_seed(config):
    rows = rows_to_records([
        {"arm": "a", "seed": 0, "t": 0, "mrr_new": 1.0, "n_new": 1},
        {"arm": "a", "seed": 1, "t": 0, "mrr_new": 0.0, "n_new": 99},
    ])

    assert agg.weighted(rows, "mrr_new") == pytest.approx(0.01)


def test_only_the_subset_metrics_have_a_weight(config):
    rows = weight_rows([(0.5, 10)])

    assert agg.weighted(rows, "mrr") is None
    assert agg.weighted(rows, "roc_auc") is None
    assert set(agg.WEIGHT_OF) == {"mrr_repeat", "mrr_new"}


def test_an_all_empty_subset_weights_to_nothing_rather_than_zero(config):
    rows = weight_rows([(NAN, 0), (NAN, 0)])

    assert agg.weighted(rows, "mrr_new") is None


# --------------------------------------------------------------------- #
# 10. split: the warm-up diagnostic the change exists for
#     Requirement: Early-Versus-Late Snapshot Split
# --------------------------------------------------------------------- #


def split_inputs(n=8, cov=None):
    vals = {t: [float(t)] for t in range(n)}
    rows = rows_to_records([
        dict({"arm": "a", "seed": 0, "t": t, "m": float(t)},
             **({"basis_zeroed_pair_frac": cov[t]} if cov else {}))
        for t in range(n)
    ])
    return agg.per_snapshot(rows, "m"), rows


def test_a_fractional_boundary_is_a_fraction_of_the_run(config):
    curve, rows = split_inputs(8)

    s = agg.split(curve, rows, 0.25)

    assert s["boundary_snapshots"] == 2 and s["boundary_arg"] == 0.25
    assert (s["early"]["n_snapshots"], s["late"]["n_snapshots"]) == (2, 6)
    assert s["early"]["mean"] == pytest.approx(0.5)     # snapshots 0,1
    assert s["late"]["mean"] == pytest.approx(4.5)      # snapshots 2..7


def test_the_boundary_is_the_first_k_snapshots_with_no_overlap_and_no_gap(config):
    for n, arg in [(8, 0.25), (8, 0.5), (27, 0.2), (733, 0.1)]:
        curve, rows = split_inputs(n)
        s = agg.split(curve, rows, arg)
        k = s["boundary_snapshots"]
        assert s["early"]["n_snapshots"] == k
        assert s["late"]["n_snapshots"] == n - k
        # early is snapshots 0..k-1 exactly: means fix the membership uniquely
        assert s["early"]["mean"] == pytest.approx((k - 1) / 2)
        assert s["late"]["mean"] == pytest.approx((k + n - 1) / 2)


@pytest.mark.parametrize("arg,k", [(1, 1), (3, 3), (5, 5), (7, 7)])
def test_an_integer_boundary_is_a_fixed_snapshot_count(config, arg, k):
    # documented in the module docstring as "an integer >= 1 for a fixed
    # snapshot count"; --early parses as a float, so it arrives as 3.0.
    curve, rows = split_inputs(8)

    s = agg.split(curve, rows, float(arg))

    assert s["boundary_snapshots"] == k
    assert s["early"]["n_snapshots"] == k


def test_the_boundary_is_clamped_so_the_late_group_is_never_empty(config):
    curve, rows = split_inputs(8)

    for arg in (8.0, 100.0, 0.99):
        s = agg.split(curve, rows, arg)
        assert s["boundary_snapshots"] == 7
        assert s["late"]["n_snapshots"] == 1


def test_a_single_snapshot_run_puts_everything_early(config):
    curve, rows = split_inputs(1)

    s = agg.split(curve, rows, 0.2)

    assert s["boundary_snapshots"] == 1
    assert s["early"]["n_snapshots"] == 1
    assert s["late"] == {"n_snapshots": 0, "mean": None, "median": None,
                         "zeroed_pair_frac": None}


def test_changing_the_boundary_changes_the_split_and_the_stated_value(config):
    curve, rows = split_inputs(20)

    a, b = agg.split(curve, rows, 0.1), agg.split(curve, rows, 0.5)

    assert (a["boundary_snapshots"], b["boundary_snapshots"]) == (2, 10)
    assert (a["boundary_arg"], b["boundary_arg"]) == (0.1, 0.5)
    assert a["early"]["mean"] != b["early"]["mean"]


def test_a_warm_up_deficit_shows_in_the_early_group_alone(config):
    # the finding the split exists to make visible: a deficit confined to the
    # first snapshots, invisible in the run-level mean.
    vals = {t: [0.0 if t < 4 else 0.5] for t in range(20)}
    curve = agg.per_snapshot(curve_of(vals), "m")

    s = agg.split(curve, curve_of(vals), 0.2)

    assert s["early"]["mean"] == 0.0
    assert s["late"]["mean"] == 0.5
    assert agg.distribution(curve)["mean"] == pytest.approx(0.4)   # the mean hides it


def test_basis_coverage_is_reported_per_group(config):
    cov = {t: (0.9 if t < 4 else 0.0) for t in range(20)}
    curve, rows = split_inputs(20, cov=cov)

    s = agg.split(curve, rows, 0.2)

    assert s["early"]["zeroed_pair_frac"] == pytest.approx(0.9)
    assert s["late"]["zeroed_pair_frac"] == pytest.approx(0.0)


def test_coverage_is_absent_when_the_run_did_not_measure_it(config):
    curve, rows = split_inputs(8)

    s = agg.split(curve, rows, 0.25)

    assert s["early"]["zeroed_pair_frac"] is None
    assert s["late"]["zeroed_pair_frac"] is None


# --------------------------------------------------------------------- #
# 11. compare: a mean carried by a minority of snapshots
#     Requirement: Distributional Summary Over Snapshots (win fraction)
# --------------------------------------------------------------------- #


def two_curves(a_vals, b_vals):
    return (agg.per_snapshot(curve_of({t: [v] for t, v in enumerate(a_vals)}), "m"),
            agg.per_snapshot(curve_of({t: [v] for t, v in enumerate(b_vals)}), "m"))


def test_a_positive_mean_difference_with_a_minority_win_fraction(config):
    # the case the feature exists to expose: one big snapshot carries the mean
    # while the arm loses on four snapshots out of five.
    a, b = two_curves([1.6, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.1, 0.1, 0.1])

    c = agg.compare(a, b)

    assert c["mean_diff"] > 0 and c["mean_diff"] == pytest.approx(0.24)
    assert c["wins"] == 1 and c["win_frac"] == pytest.approx(0.2)
    assert c["median_diff"] == pytest.approx(-0.1)


def test_ties_are_not_wins(config):
    a, b = two_curves([0.5, 0.5, 0.9, 0.1], [0.5, 0.5, 0.1, 0.9])

    c = agg.compare(a, b)

    assert c["n_shared"] == 4
    assert c["wins"] == 1
    assert c["win_frac"] == pytest.approx(0.25)
    assert c["mean_diff"] == pytest.approx(0.0)


def test_a_consistent_advantage_wins_everywhere(config):
    a, b = two_curves([0.2, 0.3, 0.4], [0.1, 0.2, 0.3])

    c = agg.compare(a, b)

    assert c["wins"] == 3 and c["win_frac"] == 1.0
    assert c["mean_diff"] == pytest.approx(0.1) == c["median_diff"]


def test_only_snapshots_present_in_both_arms_are_compared(config):
    a = agg.per_snapshot(curve_of({0: [0.5], 1: [0.5], 2: [0.5]}), "m")
    b = agg.per_snapshot(curve_of({1: [0.1], 2: [0.1], 9: [0.1]}), "m")

    c = agg.compare(a, b)

    assert c["n_shared"] == 2 and c["wins"] == 2


def test_disjoint_arms_compare_to_nothing(config):
    a = agg.per_snapshot(curve_of({0: [0.5]}), "m")
    b = agg.per_snapshot(curve_of({7: [0.5]}), "m")

    assert agg.compare(a, b) is None


# --------------------------------------------------------------------- #
# 12. aggregation ORDER (D5): the two means, and when they differ
# --------------------------------------------------------------------- #


def order_by_seed(rows, metric):
    """Mean over seeds of the cross-snapshot mean -- the order results.md's
    headline numbers use. The utility does not compute this; it is here so the
    divergence can be shown rather than asserted."""
    per_seed = {}
    for by_seed in rows.values():
        for seed, r in by_seed.items():
            v = r.get(metric)
            if isinstance(v, (int, float)) and v == v:
                per_seed.setdefault(seed, []).append(v)
    return st.fmean(st.fmean(v) for v in per_seed.values())


def order_by_snapshot(rows, metric):
    return agg.distribution(agg.per_snapshot(rows, metric))["mean"]


def test_the_two_orders_agree_under_uniform_seed_coverage(config):
    rows = curve_of({0: [0.1, 0.9], 1: [0.4, 0.2], 2: [0.7, 0.3]})

    assert order_by_snapshot(rows, "m") == pytest.approx(order_by_seed(rows, "m"))


def test_the_two_orders_differ_when_a_seed_is_missing_a_snapshot(config):
    # t=1 is backed by one seed. Over snapshots of the cross-seed mean:
    # (0.5 + 0.0)/2 = 0.25. Over seeds of the cross-snapshot mean:
    # seed0 = (0.0 + 0.0)/2 = 0.0, seed1 = 1.0/1 -> mean 0.5.
    rows = rows_to_records([
        {"arm": "a", "seed": 0, "t": 0, "m": 0.0},
        {"arm": "a", "seed": 1, "t": 0, "m": 1.0},
        {"arm": "a", "seed": 0, "t": 1, "m": 0.0},
    ])

    assert order_by_snapshot(rows, "m") == pytest.approx(0.25)
    assert order_by_seed(rows, "m") == pytest.approx(0.5)


def test_a_nan_in_one_seed_is_enough_to_split_the_two_orders(config):
    # mrr_new is nan on a snapshot with no new positives, so ragged coverage is
    # the routine case, not the exotic one.
    rows = curve_of({0: [0.0, 1.0], 1: [0.0, NAN]})

    assert order_by_snapshot(rows, "m") == pytest.approx(0.25)
    assert order_by_seed(rows, "m") == pytest.approx(0.5)
    assert agg.per_snapshot(rows, "m")[1][2] == 1        # the reduced count is reported


def test_the_contributing_seed_count_travels_with_every_row(config):
    rows = curve_of({0: [0.0, 1.0], 1: [0.0, NAN], 2: [NAN, 1.0]})

    counts = {t: n for t, (_, _, n) in agg.per_snapshot(rows, "m").items()}

    assert counts == {0: 2, 1: 1, 2: 1}


# --------------------------------------------------------------------- #
# 13. the CLI: what the reader is actually shown
# --------------------------------------------------------------------- #


def write_record(path, arm, rows):
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(dict(r, arm=arm, run_id=f"{arm}_s{r['seed']}")) + "\n")


def run_cli(capsys, *argv):
    old = sys.argv
    sys.argv = ["aggregate_snapshots.py", *argv]
    try:
        rc = agg.main()
    finally:
        sys.argv = old
    return rc, capsys.readouterr().out


def two_arm_records(tmp_path, n=8):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_record(a, "uci_gru_f+es_C1_treated",
                 [{"seed": s, "t": t, "mrr_new": 0.5, "n_new": 4}
                  for s in (1, 2) for t in range(n)])
    write_record(b, "uci_gru_feature_C1_base",
                 [{"seed": s, "t": t, "mrr_new": 0.4, "n_new": 4}
                  for s in (1, 2) for t in range(n)])
    return str(a), str(b)


def test_the_cli_states_the_aggregation_order_it_reports(capsys, tmp_path):
    a, _ = two_arm_records(tmp_path)

    rc, out = run_cli(capsys, a, "--metric", "mrr_new")

    assert rc == 0
    assert "over-snapshots of cross-seed mean" in out
    assert "over-seeds of per-seed snapshot mean" in out
    assert "the order results.md uses" in out


def test_the_cli_states_the_boundary_it_used(capsys, tmp_path):
    a, _ = two_arm_records(tmp_path, n=20)

    _, out = run_cli(capsys, a, "--metric", "mrr_new", "--early", "0.25")

    assert "early-boundary-arg=0.25" in out
    assert "early/late @ first 5 snapshots" in out
    assert "early: n=5" in out and "late : n=15" in out


def test_the_cli_labels_the_weighted_mean_as_a_different_question(capsys, tmp_path):
    a, _ = two_arm_records(tmp_path)

    _, out = run_cli(capsys, a, "--metric", "mrr_new")

    assert "pair-weighted (by n_new)" in out
    assert "they answer different questions" in out


def test_the_cli_reports_a_win_fraction_against_a_named_baseline(capsys, tmp_path):
    a, b = two_arm_records(tmp_path)

    rc, out = run_cli(capsys, a, b, "--metric", "mrr_new", "--vs", "feature")

    assert rc == 0
    assert "pairwise vs uci_gru_feature_C1_base" in out
    assert "mean_diff=+0.1000" in out
    assert "wins=8/8 (100%)" in out


def test_an_ambiguous_baseline_is_refused(capsys, tmp_path):
    a, b = two_arm_records(tmp_path)

    rc, out = run_cli(capsys, a, b, "--metric", "mrr_new", "--vs", "uci")

    assert rc == 1
    assert "matched 2 arms; need exactly 1" in out


def test_the_cli_reads_a_record_the_writer_produced(config, tmp_path, monkeypatch, capsys):
    record_cfg(config, tmp_path, data_type="feature")
    set_fes_model_config(config, pe_dim=4)
    config["spectral"]["update_mode"] = "keep"
    config["subgraph"]["num_subgraphs"] = 1
    config["metric"]["repeat_new_split"] = True
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    config["wandb"]["mode"] = "disabled"
    seed_all(42)
    res = run_seeds(config, monkeypatch, dense_snapshots(N=30, num_snaps=5, seed=7),
                    seeds=(42, 142))
    path = os.path.join(tmp_path, record_files(tmp_path)[0])

    rc, out = run_cli(capsys, path, "--metric", "mrr")

    assert rc == 0
    assert "seeds=[42, 142]  snapshots=4" in out
    # the curve is the cross-seed mean at each snapshot, recomputed by hand
    expected = st.fmean(st.fmean(r[t] for r in (res[0]["mrr_history"], res[1]["mrr_history"]))
                        for t in range(4))
    assert f"mean={expected:.4f}" in out


def test_a_nan_metric_survives_the_json_round_trip(tmp_path):
    # json.dumps writes the non-standard NaN token, which json.loads accepts and
    # a strict JSON reader does not. The aggregation depends on it arriving as a
    # float nan rather than a string or a zero.
    p = tmp_path / "n.jsonl"
    write_record(p, "arm", [{"seed": 1, "t": 0, "mrr_new": NAN, "n_new": 0},
                            {"seed": 1, "t": 1, "mrr_new": 0.5, "n_new": 3}])

    assert "NaN" in p.read_text()
    rows = agg.load([str(p)])["arm"]

    assert math.isnan(rows[0][1]["mrr_new"])
    assert sorted(agg.per_snapshot(rows, "mrr_new")) == [1]
    assert agg.weighted(rows, "mrr_new") == pytest.approx(0.5)


def test_load_groups_by_arm_and_keys_by_seed_and_snapshot(tmp_path):
    a, b = two_arm_records(tmp_path, n=3)

    data = agg.load([a, b])

    assert sorted(data) == ["uci_gru_f+es_C1_treated", "uci_gru_feature_C1_base"]
    rows = data["uci_gru_f+es_C1_treated"]
    assert sorted(rows) == [0, 1, 2]
    assert sorted(rows[0]) == [1, 2]


def test_no_arguments_prints_the_usage_and_fails(capsys):
    rc, out = run_cli(capsys)

    assert rc == 1
    assert "usage:" in out


def test_the_cli_survives_a_run_with_nothing_late(capsys, tmp_path):
    # FIXED here: the late group of a one-snapshot curve has mean None, and the
    # report formatted it with .4f.
    p = tmp_path / "one.jsonl"
    write_record(p, "arm", [{"seed": 1, "t": 0, "mrr_new": 0.4, "n_new": 3}])

    rc, out = run_cli(capsys, str(p), "--metric", "mrr_new")

    assert rc == 0
    assert "early: n=1 mean=0.4000" in out
    assert "late : n=0 mean=n/a" in out


def test_the_cli_accepts_the_documented_fixed_count_boundary(capsys, tmp_path):
    # FIXED here: --early parses as a float, so a fixed count reached the slice
    # as 5.0 and raised TypeError on every run longer than the boundary.
    a, _ = two_arm_records(tmp_path, n=20)

    rc, out = run_cli(capsys, a, "--metric", "mrr_new", "--early", "5")

    assert rc == 0
    assert "early/late @ first 5 snapshots" in out
    assert "early: n=5" in out and "late : n=15" in out


# --------------------------------------------------------------------- #
# 14. the snapshot axis across a resume
#     A crash is the normal case on the datasets this change exists for --
#     as733 is 733 snapshots and takes hours -- so the resumed segment is
#     exactly the data most likely to need the axis, and the least able to
#     reconstruct it.
# --------------------------------------------------------------------- #


def ckpt_cfg(cfg, tmp_path):
    cfg["train"]["auto_resume"] = True
    cfg["train"]["ckpt_period"] = 1
    cfg["train"]["ckpt_clean"] = True
    cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpt")


class _Crash(Exception):
    pass


def crash_after(t_crash):
    original = DynamicServer._save_partial_ckpt

    def spy(self, t, w_init, mrr_history, metrics_history, t_history=None):
        original(self, t, w_init, mrr_history, metrics_history, t_history)
        if t == t_crash:
            raise _Crash

    return original, spy


def checkpointable_server(config, tmp_path):
    config["train"]["ckpt_dir"] = str(tmp_path)
    config["train"]["ckpt_clean"] = False
    snaps = sparse_snapshots()
    server = DynamicServer(snaps)
    for part in partition_snapshots(snaps, 1):
        server.add_client(part)
    server.initialize_FL()
    return server


def test_the_checkpoint_carries_the_snapshot_axis(config, tmp_path):
    identity_cfg(config)
    server = checkpointable_server(config, tmp_path)

    server._save_partial_ckpt(3, None, [0.1, 0.2], [{}, {}], [0, 2])
    resumed = server._load_partial_ckpt()

    assert resumed[-1] == [0, 2]


def test_a_checkpoint_predating_the_axis_restores_it_as_unknown(config, tmp_path):
    identity_cfg(config)
    server = checkpointable_server(config, tmp_path)
    server._save_partial_ckpt(3, None, [0.1, 0.2], [{}, {}], [0, 2])

    path, _ = server._ckpt_paths()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    del ckpt["t_history"]
    torch.save(ckpt, path)

    assert server._load_partial_ckpt()[-1] is None


def test_a_resumed_run_records_the_true_axis_across_the_crash(config, tmp_path, monkeypatch):
    # the whole point of carrying it: the pre-crash segment skipped snapshot 1,
    # and only the checkpoint can still say so.
    record_cfg(config, tmp_path)
    ckpt_cfg(config, tmp_path)
    seed_all(42)
    snaps = blanked_snapshots(n=7, blank=2)
    original, spy = crash_after(3)

    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", spy)
    with pytest.raises(_Crash):
        run_seeds(config, monkeypatch, snaps)
    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", original)
    res = run_seeds(config, monkeypatch, snaps)[0]

    assert res["t_history"] == [0, 2, 3, 4, 5]         # 1 was skipped, before the crash
    rows = read_rows(tmp_path)
    assert [r["t"] for r in rows] == res["t_history"]
    assert [r["mrr"] for r in rows] == res["mrr_history"]


def test_an_unknown_axis_is_refused_rather_than_written_as_minus_one(
    config, tmp_path, monkeypatch
):
    # A run resumed from a checkpoint predating t_history has no axis for its
    # pre-crash segment. Writing those rows anyway is worse than writing nothing:
    # see the collapse test below.
    record_cfg(config, tmp_path)
    ckpt_cfg(config, tmp_path)
    seed_all(42)
    snaps = dense_snapshots(N=30, num_snaps=7, seed=7)
    original, spy = crash_after(3)

    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", spy)
    with pytest.raises(_Crash):
        run_seeds(config, monkeypatch, snaps)
    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", original)

    server = DynamicServer(snaps)
    path, _ = server._ckpt_paths()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    del ckpt["t_history"]                              # a checkpoint from before the key
    torch.save(ckpt, path)

    rec = _Recorder()
    monkeypatch.setattr(main, "LOGGER", rec)
    res = run_seeds(config, monkeypatch, snaps)[0]

    assert res["t_history"][:4] == [-1, -1, -1, -1]    # the premise: axis unknown
    assert record_files(tmp_path) == []
    assert any("NOT WRITTEN" in m and "axis unknown" in m for m in rec.warn_lines)


def test_unknown_indices_would_collapse_to_one_row(config, tmp_path):
    # why refusing beats marking: load() keys on (arm, seed, t), so every row
    # sharing t=-1 overwrites the last. Five snapshots become one, and the mean,
    # the weighted aggregate and the early group all move -- a persisted value
    # disagreeing with the reported one, silently.
    p = tmp_path / "collapsed.jsonl"
    write_record(p, "arm", [{"seed": 1, "t": -1, "mrr": 0.1 * i, "mrr_new": 0.1 * i,
                             "n_new": 4} for i in range(5)]
                 + [{"seed": 1, "t": t, "mrr": 0.5, "mrr_new": 0.5, "n_new": 4}
                    for t in range(5, 10)])
    rows = agg.load([str(p)])["arm"]

    assert sum(len(v) for v in rows.values()) == 6     # 10 written, 6 survive
    curve = agg.per_snapshot(rows, "mrr")
    assert agg.distribution(curve)["mean"] == pytest.approx(0.48333333, abs=1e-6)
    assert st.fmean([0.0, 0.1, 0.2, 0.3, 0.4] + [0.5] * 5) == pytest.approx(0.35)
    assert min(curve) == -1                            # and it sorts before every real one


def test_a_partial_resume_does_not_duplicate_the_pre_crash_rows(config, tmp_path, monkeypatch):
    # the writer runs once per seed, at the end; a crashed seed wrote nothing, so
    # the resumed run's full history is the only copy.
    record_cfg(config, tmp_path)
    ckpt_cfg(config, tmp_path)
    seed_all(42)
    snaps = dense_snapshots(N=30, num_snaps=7, seed=7)
    original, spy = crash_after(3)

    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", spy)
    with pytest.raises(_Crash):
        run_seeds(config, monkeypatch, snaps)
    assert record_files(tmp_path) == []
    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", original)
    res = run_seeds(config, monkeypatch, snaps)[0]

    rows = read_rows(tmp_path)
    assert len(rows) == len(res["mrr_history"]) == 6
    assert len({(r["seed"], r["t"]) for r in rows}) == 6


def test_the_resumed_complete_flag_alone_stops_a_re_append(config, tmp_path, monkeypatch):
    # run_once checks the done-checkpoint before training and the server sets the
    # flag when it short-circuits on it. Either must be enough: a .done written by
    # a peer between the two checks would otherwise slip a duplicate seed in.
    identity_cfg(config)
    config["metric"]["snapshot_record_dir"] = str(tmp_path)
    server = DynamicServer(sparse_snapshots())
    payload = {"mrr_history": [0.5], "metrics_history": [{}], "t_history": [0],
               "_resumed_complete": True}

    rec = write_with_logger(monkeypatch, server, payload, already_done=False)

    assert record_files(tmp_path) == []
    assert any("already complete" in m for m in rec.info_lines)


def test_widening_a_finished_sweep_adds_only_the_new_seeds(config, tmp_path, monkeypatch):
    # the REPEAT knob: a 3-seed cell rerun at REPEAT=5 re-executes seeds 1-3,
    # which short-circuit on their done checkpoints. The log gets an _r5 suffix
    # so it cannot overwrite the banked one; the record has no such suffix, so it
    # must not re-append instead.
    record_cfg(config, tmp_path)
    ckpt_cfg(config, tmp_path)
    seed_all(42)
    snaps = dense_snapshots(N=30, num_snaps=4, seed=7)

    run_seeds(config, monkeypatch, snaps, seeds=(42, 142, 242))
    assert len(read_rows(tmp_path)) == 9
    run_seeds(config, monkeypatch, snaps, seeds=(42, 142, 242, 342, 442))

    rows = read_rows(tmp_path)
    assert len(rows) == 15
    assert len({(r["seed"], r["t"]) for r in rows}) == 15
    assert sorted({r["seed"] for r in rows}) == [42, 142, 242, 342, 442]


# --------------------------------------------------------------------- #
# 15. both aggregation orders, reported side by side (D5, task 3.5)
# --------------------------------------------------------------------- #


def test_seed_order_mean_is_the_order_results_md_uses(config):
    rows = curve_of({0: [0.1, 0.9], 1: [0.3, 0.7]})

    mean, n_seeds = agg.seed_order_mean(rows, "m")

    assert mean == pytest.approx(0.5) and n_seeds == 2
    assert mean == pytest.approx(st.fmean([st.fmean([0.1, 0.3]), st.fmean([0.9, 0.7])]))


def test_the_two_orders_are_computed_separately_and_disagree(config):
    rows = rows_to_records([
        {"arm": "a", "seed": 0, "t": 0, "m": 0.0},
        {"arm": "a", "seed": 1, "t": 0, "m": 1.0},
        {"arm": "a", "seed": 0, "t": 1, "m": 0.0},
    ])

    assert order_by_snapshot(rows, "m") == pytest.approx(0.25)
    assert agg.seed_order_mean(rows, "m")[0] == pytest.approx(0.50)


def test_seed_order_skips_a_seed_with_nothing_defined(config):
    rows = curve_of({0: [0.4, NAN], 1: [0.6, NAN]})

    assert agg.seed_order_mean(rows, "m") == (pytest.approx(0.5), 1)


def test_an_empty_arm_has_no_seed_order_mean(config):
    assert agg.seed_order_mean(curve_of({0: [NAN]}), "m") == (None, 0)


def ragged_record(tmp_path):
    p = tmp_path / "ragged.jsonl"
    write_record(p, "uci_gru_f_C1", [{"seed": 1, "t": 0, "mrr": 0.0},
                                     {"seed": 2, "t": 0, "mrr": 1.0},
                                     {"seed": 1, "t": 1, "mrr": 0.0}])
    return str(p)


def test_the_cli_flags_unequal_seed_coverage(capsys, tmp_path):
    _, out = run_cli(capsys, ragged_record(tmp_path), "--metric", "mrr")

    assert "over-snapshots of cross-seed mean : mean=0.2500" in out
    assert "over-seeds of per-seed snapshot mean : mean=0.5000" in out
    assert "[DIFFERS: unequal seed coverage]" in out
    assert "[agrees]" not in out


def test_the_cli_marks_the_orders_as_agreeing_under_full_coverage(capsys, tmp_path):
    p = tmp_path / "full.jsonl"
    write_record(p, "uci_gru_f_C1", [{"seed": s, "t": t, "mrr": 0.1 * t + 0.5 * s}
                                     for s in (1, 2) for t in range(6)])

    _, out = run_cli(capsys, str(p), "--metric", "mrr")

    assert "[agrees]" in out
    assert "DIFFERS" not in out


def test_ties_are_reported_separately_from_losses(config):
    # win_frac puts ties in the denominator, so wins alone cannot distinguish a
    # tie from a loss -- which matters exactly where the win fraction is the
    # evidence, i.e. an advantage carried by a minority of snapshots.
    a, b = two_curves([0.5, 0.5, 0.9, 0.1, 0.1], [0.5, 0.5, 0.1, 0.9, 0.9])

    c = agg.compare(a, b)

    assert (c["wins"], c["losses"], c["ties"]) == (1, 2, 2)
    assert c["wins"] + c["losses"] + c["ties"] == c["n_shared"]


def test_the_cli_prints_losses_and_ties_beside_the_win_fraction(capsys, tmp_path):
    a, b = two_arm_records(tmp_path)

    _, out = run_cli(capsys, a, b, "--metric", "mrr_new", "--vs", "feature")

    assert "wins=8/8 (100%) losses=0 ties=0" in out
