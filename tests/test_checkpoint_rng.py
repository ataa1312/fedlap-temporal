"""RNG state in partial checkpoints (commit 814eabd).

Before this, _load_partial_ckpt restored weights, hidden states, spectral caches
and histories but not the position of the random streams, so a resumed run drew
its negative samples, dropout masks and recompute_prob Bernoulli from a freshly
seeded generator and could never be bit-faithful to an uninterrupted one --
experimental.deterministic included, since that pins kernels, not stream
position. These pin the continuity of all three streams, the resume-equals-
uninterrupted property under determinism, and the tolerance for checkpoints
written before the block existed.
"""

import os
import random

import numpy as np
import pytest
import torch

from src.dynamic_server import DynamicServer
from src.utils.graph_partitioning import partition_snapshots
from test_checkpoint_wandb import (
    make_server,
    make_toy_snapshots,
    seed_all,
    setup_tiny_config,
)


STREAMS = {
    "python": lambda: random.random(),
    "numpy": lambda: float(np.random.rand()),
    "torch": lambda: float(torch.rand(1)),
}


def all_streams():
    return tuple(draw() for draw in STREAMS.values())


@pytest.fixture
def deterministic_kernels():
    """What main.py throws when experimental.deterministic is on. Restored after,
    since it is a process-global switch."""
    previous = torch.are_deterministic_algorithms_enabled()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def checkpointed_server(config, ckpt_dir, num_snaps=3):
    setup_tiny_config(config, ckpt_dir)
    config["train"]["ckpt_clean"] = False
    seed_all(42)
    global_snaps = make_toy_snapshots(num_snaps=num_snaps)
    server = make_server(global_snaps, partition_snapshots(global_snaps, 1))
    server.initialize_FL()
    return server


# ---- stream continuity ---- #


@pytest.mark.parametrize("stream", list(STREAMS))
def test_a_resumed_run_continues_the_saved_stream(config, tmp_path, stream):
    draw = STREAMS[stream]
    server = checkpointed_server(config, tmp_path)
    for _ in range(5):
        draw()  # stands for whatever the run consumed before the crash

    server._save_partial_ckpt(0, None, [], [])
    expected = draw()

    draw()  # advance past the saved position
    seed_all(999)  # and land somewhere else entirely
    assert server._load_partial_ckpt() is not None

    assert draw() == expected

    # and that position is mid-stream, not where a re-seed would put it: the
    # failure mode here is resume carrying on from config['seed'].
    seed_all(config["seed"])
    assert draw() != expected


def test_the_payload_carries_every_stream_as_cpu_state(config, tmp_path):
    server = checkpointed_server(config, tmp_path)

    server._save_partial_ckpt(0, None, [], [])

    ckpt_path, _ = server._ckpt_paths()
    rng = torch.load(ckpt_path, map_location="cpu", weights_only=False)["rng"]
    assert set(rng) == {"python", "numpy", "torch", "cuda"}
    # set_rng_state takes a CPU ByteTensor, so the payload must arrive as one
    # whatever device wrote it
    assert rng["torch"].device.type == "cpu"
    assert rng["torch"].dtype == torch.uint8
    assert (rng["cuda"] is None) is not torch.cuda.is_available()


def test_the_payload_is_read_with_the_tensors_mapped_to_cpu(
    config, tmp_path, monkeypatch
):
    # asserted on the call because it is otherwise invisible on a CPU-only host:
    # the contract only bites for a checkpoint written on CUDA, where an
    # unmapped torch state comes back as a CUDA tensor and set_rng_state rejects
    # it.
    server = checkpointed_server(config, tmp_path)
    server._save_partial_ckpt(0, None, [], [])

    seen = {}
    real_load = torch.load

    def spy(path, *args, **kwargs):
        seen[str(path)] = kwargs.get("map_location")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    assert server._load_partial_ckpt() is not None

    ckpt_path, _ = server._ckpt_paths()
    assert seen[ckpt_path] == "cpu"


# ---- the real test: resumed == uninterrupted ---- #


def run_to_completion(config, ckpt_dir, num_snaps=5):
    seed_all(42)
    global_snaps = make_toy_snapshots(num_snaps=num_snaps)
    server = make_server(global_snaps, partition_snapshots(global_snaps, 1))
    config["train"]["ckpt_dir"] = str(ckpt_dir)
    return server.joint_train_w(FL=True)


def run_until_crash(config, ckpt_dir, monkeypatch, crash_at, num_snaps=5):
    class _Crash(Exception):
        pass

    original = DynamicServer._save_partial_ckpt

    def save_then_crash(self, t, w_init, mrr_history, metrics_history, t_history=None):
        original(self, t, w_init, mrr_history, metrics_history, t_history)
        if t == crash_at:
            raise _Crash

    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", save_then_crash)
    with pytest.raises(_Crash):
        run_to_completion(config, ckpt_dir, num_snaps)
    monkeypatch.setattr(DynamicServer, "_save_partial_ckpt", original)


def test_a_resumed_deterministic_run_matches_an_uninterrupted_one(
    config, tmp_path, monkeypatch, deterministic_kernels
):
    # determinism pins which kernels run; the RNG block pins where the streams
    # are. Only both together make the two runs comparable, which is why this is
    # not asserted without the flag.
    setup_tiny_config(config, tmp_path)
    config["train"]["ckpt_clean"] = False
    config["experimental"]["deterministic"] = True

    uninterrupted = run_to_completion(config, tmp_path / "whole")

    run_until_crash(config, tmp_path / "split", monkeypatch, crash_at=1)
    resumed = run_to_completion(config, tmp_path / "split")

    assert len(uninterrupted["mrr_history"]) == 4
    assert resumed["mrr_history"] == uninterrupted["mrr_history"]
    assert resumed["mean_mrr"] == uninterrupted["mean_mrr"]


def test_dropping_the_rng_block_moves_a_deterministic_resumed_run(
    config, tmp_path, monkeypatch, deterministic_kernels
):
    # the mechanism, stated as a difference: strip the block the fix added and
    # the same interrupted run lands somewhere else
    setup_tiny_config(config, tmp_path)
    config["train"]["ckpt_clean"] = False
    config["experimental"]["deterministic"] = True

    uninterrupted = run_to_completion(config, tmp_path / "whole")

    ckpt_dir = tmp_path / "split"
    run_until_crash(config, ckpt_dir, monkeypatch, crash_at=1)
    ckpt_path = next(p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    del ckpt["rng"]
    torch.save(ckpt, ckpt_path)

    stream_blind = run_to_completion(config, ckpt_dir)

    assert stream_blind["mrr_history"][:2] == uninterrupted["mrr_history"][:2]
    assert stream_blind["mrr_history"] != uninterrupted["mrr_history"]


# ---- backwards compatibility ---- #


def test_a_checkpoint_without_an_rng_block_still_resumes(config, tmp_path):
    server = checkpointed_server(config, tmp_path)
    server._save_partial_ckpt(0, None, [0.1], [{"roc_auc": 0.5}])

    ckpt_path, _ = server._ckpt_paths()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    del ckpt["rng"]  # a checkpoint written before the block existed
    torch.save(ckpt, ckpt_path)

    seed_all(999)
    expected = all_streams()
    seed_all(999)
    resumed = server._load_partial_ckpt()

    assert resumed is not None
    t_start, w_init, mrr_history, metrics_history, t_history = resumed
    assert t_start == 1
    assert mrr_history == [0.1]
    assert metrics_history == [{"roc_auc": 0.5}]
    assert t_history is None  # a checkpoint predating the key carries no axis
    assert all_streams() == expected  # streams left exactly where they were
