import logging
import sys

import pytest
import torch
import yaml
from torch_geometric.data import Data

import src
import main
from config.assertions import assert_cfg
from config.config import get_default_config
from parser import Parser
from src.dynamic_server import DynamicServer
from src.utils.graph_partitioning import partition_snapshots


def make_toy_snapshots(N=8, W=1, num_snaps=2, seed=42):
    g = torch.Generator().manual_seed(seed)
    snaps = []
    for _ in range(num_snaps):
        edges = set()
        while len(edges) < 16:
            u = torch.randint(0, N, (1,), generator=g).item()
            v = torch.randint(0, N, (1,), generator=g).item()
            if u != v:
                edges.add((u, v))
        edge_index = torch.tensor(list(edges), dtype=torch.long).t()
        edge_attr = torch.randn(edge_index.size(1), W, generator=g)
        snap = Data(x=torch.ones(N, 1), edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)
        snap.node_ids = torch.arange(N)
        snaps.append(snap)
    return snaps


class _FakeServer:
    """Stands in for DynamicServer so main() completes without training; the
    ordering under test is fixed before the server is ever built."""

    def __init__(self, global_snaps):
        self.global_snaps = global_snaps

    def add_client(self, snaps):
        pass

    def _load_done_ckpt(self):
        return None

    def joint_train_w(self, FL=True, log_cb=None):
        return {"mean_mrr": 0.0, "std_mrr": 0.0, "mrr_history": [], "mean_metrics": {}}


def write_cfg(tmp_path, **experimental):
    body = {
        "seed": 1234,
        "dataset": {"name": "uci", "snapshot_freq": "W"},
        "subgraph": {"num_subgraphs": 1},
        "model": {"data_type": "feature"},
        "gnn": {"embed_update_method": "gru"},
        "wandb": {"mode": "disabled"},
    }
    if experimental:
        body["experimental"] = experimental
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def instrument(monkeypatch):
    """Record the order of the deterministic call and of every RNG consumer
    main() reaches (seeding, dataset load, client partition)."""
    events, det_calls = [], []

    def spy_det(*a, **kw):
        events.append("deterministic")
        det_calls.append((a, kw))

    orig_seed = main._seed

    def spy_seed(s):
        events.append("seed")
        orig_seed(s)

    def fake_loader(cfg):
        events.append("dataset")
        return [object(), object()]

    def fake_partition(snaps, n_clients):
        events.append("partition")
        return [snaps]

    monkeypatch.setattr(torch, "use_deterministic_algorithms", spy_det)
    monkeypatch.setattr(main, "_seed", spy_seed)
    monkeypatch.setattr(main, "datasets", {"uci": fake_loader})
    monkeypatch.setattr(main, "partition_snapshots", fake_partition)
    monkeypatch.setattr(main, "DynamicServer", _FakeServer)
    return events, det_calls


def run_main(monkeypatch, cfg_path, extra=()):
    monkeypatch.setattr(sys, "argv", ["main.py", "-c", str(cfg_path), *extra])
    main.main()


def setup_run_id_config(cfg, tmp_path):
    cfg["dataset"]["name"] = "uci"
    cfg["dataset"]["snapshot_freq"] = "W"
    cfg["gnn"]["embed_update_method"] = "gru"
    cfg["model"]["data_type"] = "feature"
    cfg["subgraph"]["num_subgraphs"] = 1
    cfg["seed"] = 1234
    cfg["train"]["ckpt_dir"] = str(tmp_path)
    cfg["train"]["ckpt_clean"] = False


# ---- the flag itself ---- #

def test_default_is_false_bool():
    fresh = get_default_config()
    assert fresh["experimental"]["deterministic"] is False


def test_enabled_call_precedes_every_rng_consumer(config, tmp_path, monkeypatch):
    events, det_calls = instrument(monkeypatch)
    run_main(monkeypatch, write_cfg(tmp_path, deterministic=True))

    # the whole point: the kernel switch is thrown before seeding, before the
    # dataset is built and before the clients are partitioned.
    assert events == ["deterministic", "seed", "dataset", "partition"]
    assert det_calls == [((True,), {})]  # positional True, no warn_only


def test_enabled_once_for_the_whole_process(config, tmp_path, monkeypatch):
    events, det_calls = instrument(monkeypatch)
    run_main(
        monkeypatch,
        write_cfg(tmp_path),
        extra=["-r", "2", "--set", "experimental.deterministic=true"],
    )

    assert len(det_calls) == 1
    assert events == [
        "deterministic",
        "seed", "dataset", "partition",
        "seed", "dataset", "partition",
    ]


def test_disabled_never_switches_kernels(config, tmp_path, monkeypatch):
    events, det_calls = instrument(monkeypatch)
    run_main(monkeypatch, write_cfg(tmp_path, deterministic=False))

    assert det_calls == []
    assert events == ["seed", "dataset", "partition"]
    assert torch.are_deterministic_algorithms_enabled() is False


def test_absent_key_behaves_as_before_it_existed(config, tmp_path, monkeypatch):
    del config["experimental"]["deterministic"]
    events, det_calls = instrument(monkeypatch)
    run_main(monkeypatch, write_cfg(tmp_path))

    assert det_calls == []
    assert events == ["seed", "dataset", "partition"]
    assert "deterministic" not in src.config["experimental"]


@pytest.mark.parametrize("enabled", [True, False])
def test_mode_is_recorded_in_the_run_log(config, tmp_path, monkeypatch, caplog, enabled):
    instrument(monkeypatch)
    monkeypatch.setattr(main.LOGGER, "propagate", True)  # LOGGER is non-propagating
    caplog.set_level(logging.INFO)
    run_main(monkeypatch, write_cfg(tmp_path, deterministic=enabled))

    assert f"deterministic={enabled}" in caplog.text


# ---- run identity ---- #

def test_run_id_default_is_byte_identical_to_pre_change(config, tmp_path):
    setup_run_id_config(config, tmp_path)
    server = DynamicServer(make_toy_snapshots())

    config["experimental"]["deterministic"] = False
    assert server._run_id() == "uci_gru_feature_C1_s1234"

    del config["experimental"]["deterministic"]  # a config predating the key
    assert server._run_id() == "uci_gru_feature_C1_s1234"


def test_run_id_marks_deterministic_runs(config, tmp_path):
    setup_run_id_config(config, tmp_path)
    server = DynamicServer(make_toy_snapshots())

    config["experimental"]["deterministic"] = False
    off_id, off_wandb = server._run_id(), server._wandb_id()
    off_ckpt, off_done = server._ckpt_paths()

    config["experimental"]["deterministic"] = True
    assert server._run_id() == "uci_gru_feature_C1_det_s1234"
    assert server._run_id() != off_id
    assert server._wandb_id() != off_wandb

    on_ckpt, on_done = server._ckpt_paths()
    assert on_ckpt != off_ckpt
    assert on_done != off_done


def test_deterministic_run_cannot_resume_a_non_deterministic_one(config, tmp_path):
    setup_run_id_config(config, tmp_path)
    config["experimental"]["deterministic"] = False

    global_snaps = make_toy_snapshots()
    server = DynamicServer(global_snaps)
    for snaps in partition_snapshots(global_snaps, 1):
        server.add_client(snaps)
    server.initialize_FL()

    server._save_partial_ckpt(0, None, [], [])
    server._save_done_ckpt({"mean_mrr": 0.0, "mrr_history": []})

    # auto_resume must not let the deterministic run adopt or short-circuit the
    # non-deterministic one's results: they converge to different numbers.
    config["experimental"]["deterministic"] = True
    assert server._load_partial_ckpt() is None
    assert server._load_done_ckpt() is None

    config["experimental"]["deterministic"] = False
    assert server._load_partial_ckpt() is not None
    assert server._load_done_ckpt() is not None


# ---- validation ---- #

@pytest.mark.parametrize("bad", ["maybe", 1, 0.5])
def test_assert_cfg_rejects_non_boolean(config, bad):
    config["model"]["data_type"] = "feature"
    config["experimental"]["deterministic"] = bad
    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "experimental.deterministic" in str(exc.value)


def test_assert_cfg_accepts_bools_and_absence(config):
    config["model"]["data_type"] = "feature"
    for good in (True, False):
        config["experimental"]["deterministic"] = good
        assert_cfg(config)

    del config["experimental"]["deterministic"]
    assert_cfg(config)


def test_cli_override_yields_a_real_bool(config):
    config["model"]["data_type"] = "feature"
    # PyYAML resolves both spellings to booleans, so both are valid input.
    for raw in ("true", "yes"):
        Parser.apply_overrides(config, [f"experimental.deterministic={raw}"])
        assert config["experimental"]["deterministic"] is True
        assert_cfg(config)

    Parser.apply_overrides(config, ["experimental.deterministic=maybe"])
    assert config["experimental"]["deterministic"] == "maybe"
    with pytest.raises(ValueError):
        assert_cfg(config)
