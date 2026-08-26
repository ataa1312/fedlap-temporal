"""spectral.use_procrustes tri-state, structure_model.{freeze_sfv,
sfv_reset_per_snapshot} and gnn.encoder_edge_drop.

All four are experiment axes added on top of a running protocol, so each one
carries the same two obligations: at its default the run must be unchanged
(same numbers, same _run_id, same RNG stream, so banked checkpoints stay
loadable), and when it is on it must move exactly the one thing it names and
nothing else. These pin both halves -- the effective-value resolution, the
optimizer/FedAvg membership of a frozen W, the per-boundary redraw, and the
containment of the encoder drop away from the splits, the negatives and the
cumulative union.
"""

import random
import types

import numpy as np
import pytest
import torch

import main
import src.dynamic_server as dynamic_server
from config.assertions import assert_cfg
from src.dynamic_client import DynamicClient
from src.dynamic_server import DynamicServer, _get_sfv, _procrustes_on, _sfv_flag
from src.models.model_binders import ModelBinder
from src.train.federated_orchestrator import (
    MP_EDGE_ATTR_ATTR,
    MP_EDGE_INDEX_ATTR,
    _make_optimizer,
    _mp_graph,
    _partition_edges_per_snapshot,
    _precompute_encoder_edge_drop,
)
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots
from test_checkpoint_wandb import make_server, seed_all
from test_fl_local_baseline import make_toy_snapshots


COORDINATE_TYPES = ("f+s", "f+pe", "structure")

# captured unpatched, so a helper called twice in one test does not spy on its
# own spy and double-count
_ORIG_RESET_SFVS = DynamicServer._reset_sfvs


def tiny_run(cfg, data_type="feature"):
    """A 4-snapshot, 2-client run small enough to execute inside a test."""
    cfg["dataset"]["name"] = "uci"
    cfg["dataset"]["task"] = "link_pred"
    cfg["dataset"]["edge_dim"] = 1
    cfg["dataset"]["node_encoder"] = False
    cfg["dataset"]["edge_encoder"] = False
    cfg["dataset"]["snapshot_freq"] = "W"
    cfg["dataset"]["split"] = [0.8, 0.1, 0.1]
    cfg["subgraph"]["num_subgraphs"] = 2
    cfg["model"]["data_type"] = data_type
    cfg["model"]["edge_decoding"] = "concat"
    cfg["model"]["smodel_type"] = "LanczosLaplace"
    cfg["model"]["iterations"] = 1
    cfg["model"]["local_epochs"] = 1
    cfg["gnn"]["dims"] = [8, 8]
    cfg["gnn"]["dims_pre_mp"] = []
    cfg["gnn"]["dims_post_mp"] = []
    cfg["gnn"]["embed_update_method"] = "gru"
    cfg["gnn"]["l2norm"] = False
    cfg["spectral"]["spectral_len"] = 4
    cfg["spectral"]["pe_dim"] = 4
    cfg["spectral"]["update_mode"] = "keep"
    cfg["structure_model"]["num_structural_features"] = 16
    cfg["structure_model"]["DGCN_structure_layers_sizes"] = [8]
    cfg["train"]["auto_resume"] = False
    cfg["seed"] = 42


def build_server(num_snaps=4, num_clients=2, seed=7):
    seed_all(42)
    global_snaps = make_toy_snapshots(N=30, num_snaps=num_snaps, seed=seed)
    return make_server(global_snaps, partition_snapshots(global_snaps, num_clients))


def sfv_of(owner_clf):
    # _get_sfv aliases the live tensor (detach + a cpu no-op), so anything kept
    # across a training step has to be cloned.
    w = _get_sfv(owner_clf)
    return None if w is None else w.clone()


def all_sfvs(server):
    return [sfv_of(server.classifier)] + [sfv_of(cl.classifier) for cl in server.clients]


def _flat(state, prefix=""):
    """The federated protocol nests state_dicts; flatten to tensor leaves."""
    out = {}
    for key, value in state.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, f"{path}."))
        elif torch.is_tensor(value):
            out[path] = value
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if torch.is_tensor(item):
                    out[f"{path}.{i}"] = item
    return out


def identity_cfg(cfg, tmp_path):
    """The knobs _run_id reads, pinned, so a test can vary exactly one."""
    cfg["dataset"]["name"] = "uci"
    cfg["dataset"]["snapshot_freq"] = "W"
    cfg["gnn"]["embed_update_method"] = "gru"
    cfg["subgraph"]["num_subgraphs"] = 1
    cfg["seed"] = 1234
    cfg["experimental"]["deterministic"] = False
    cfg["federated"]["sfv_share"] = "local"
    cfg["spectral"]["update_mode"] = "update"
    cfg["spectral"]["basis_source"] = "laplacian"
    cfg["spectral"]["solver"] = "arnoldi"
    cfg["spectral"]["pe_dim"] = 50
    cfg["spectral"]["es_features"] = "spec"
    cfg["spectral"]["es_spec_parts"] = "phi+cos+lev"
    cfg["train"]["ckpt_dir"] = str(tmp_path)
    cfg["train"]["ckpt_clean"] = False


# --------------------------------------------------------------------- #
# 1. spectral.use_procrustes: 'auto' | True | False
# --------------------------------------------------------------------- #


def test_auto_is_the_shipped_default(config):
    assert config["spectral"]["use_procrustes"] == "auto"


@pytest.mark.parametrize("data_type", COORDINATE_TYPES)
def test_auto_is_on_where_the_readout_reads_coordinates(config, data_type):
    config["model"]["data_type"] = data_type
    assert _procrustes_on() is True
    assert _procrustes_on(data_type) is True


def test_auto_is_off_on_the_rotation_invariant_path(config):
    config["model"]["data_type"] = "f+es"
    assert _procrustes_on() is False
    assert _procrustes_on("f+es") is False


@pytest.mark.parametrize("explicit", [True, False])
def test_an_explicit_bool_wins_on_every_path(config, explicit):
    config["spectral"]["use_procrustes"] = explicit
    for data_type in COORDINATE_TYPES + ("f+es",):
        config["model"]["data_type"] = data_type
        assert _procrustes_on() is explicit
        assert _procrustes_on(data_type) is explicit


@pytest.mark.parametrize(
    "data_type,solver,auto_rotates",
    [("f+s", "arnoldi", True), ("f+pe", "exact", True), ("f+es", "chebyshev", False)],
)
def test_the_effective_value_drives_the_actual_rotation(
    config, monkeypatch, data_type, solver, auto_rotates
):
    # The label is worthless unless the branch at get_spectral_features follows
    # it, so count real procrustes_project calls over two snapshots under
    # update (ss_idx>0 is the only one that can rotate).
    tiny_run(config, data_type)
    config["spectral"]["update_mode"] = "update"
    config["spectral"]["solver"] = solver

    original = Graph.procrustes_project
    calls = []

    def spy(self, U, ref):
        calls.append(1)
        return original(self, U, ref)

    monkeypatch.setattr(Graph, "procrustes_project", spy)

    def count(value):
        calls.clear()
        config["spectral"]["use_procrustes"] = value
        server = build_server(num_snaps=3)
        server.initialize_FL()
        smt = config["model"]["smodel_type"]
        server._spectral_step(0, smt)
        server._spectral_step(1, smt)
        return len(calls)

    assert count("auto") == (1 if auto_rotates else 0)
    assert count(True) == 1
    assert count(False) == 0


def test_run_id_records_the_effective_value_not_the_configured_one(config, tmp_path):
    identity_cfg(config, tmp_path)
    config["model"]["data_type"] = "f+es"
    config["spectral"]["solver"] = "chebyshev"  # f+es forbids arnoldi
    server = DynamicServer(make_toy_snapshots(N=30, num_snaps=2, seed=7))

    config["spectral"]["use_procrustes"] = "auto"
    auto = server._run_id()
    config["spectral"]["use_procrustes"] = True
    explicit_on = server._run_id()
    config["spectral"]["use_procrustes"] = False
    explicit_off = server._run_id()

    assert "proc-off" in auto.split("_")
    assert "proc-on" in explicit_on.split("_")
    assert auto != explicit_on
    # 'auto' and an explicit False rotate identically, so they are the same run
    # and must share one checkpoint rather than burn two sweep slots.
    assert auto == explicit_off


@pytest.mark.parametrize("data_type", ["f+s", "f+pe"])
def test_the_default_identity_is_unchanged_where_auto_resolves_on(
    config, tmp_path, data_type
):
    # Byte-identity against the pre-change use_procrustes=True default: banked
    # f+s / f+pe checkpoints must keep loading.
    identity_cfg(config, tmp_path)
    config["model"]["data_type"] = data_type
    server = DynamicServer(make_toy_snapshots(N=30, num_snaps=2, seed=7))

    config["spectral"]["use_procrustes"] = "auto"
    shipped = server._run_id()
    config["spectral"]["use_procrustes"] = True
    pre_change = server._run_id()

    assert shipped == pre_change


def test_the_fes_default_moved_and_orphans_its_banked_checkpoints(config, tmp_path):
    # DELIBERATE deviation from "defaults unchanged" (results.md 19): f+es was
    # shipped with use_procrustes=true and now resolves to off. Pinned here so
    # the orphaning is visible in the suite rather than discovered mid-sweep.
    identity_cfg(config, tmp_path)
    config["model"]["data_type"] = "f+es"
    config["spectral"]["solver"] = "chebyshev"
    server = DynamicServer(make_toy_snapshots(N=30, num_snaps=2, seed=7))

    config["spectral"]["use_procrustes"] = True  # the pre-change default
    server._save_done_ckpt({"mean_mrr": 0.0, "mrr_history": []})
    assert server._load_done_ckpt() is not None

    config["spectral"]["use_procrustes"] = "auto"  # the shipped default now
    assert server._load_done_ckpt() is None


@pytest.mark.parametrize("bad", ["AUTO", "on", 1, 0, None, "true"])
def test_assert_cfg_rejects_a_non_bool_non_auto_procrustes(config, bad):
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "keep"
    config["spectral"]["use_procrustes"] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.use_procrustes" in str(exc.value)


def test_assert_cfg_accepts_the_whole_tri_state(config):
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "keep"
    for value in ("auto", True, False):
        config["spectral"]["use_procrustes"] = value
        assert_cfg(config)


def test_wandb_identity_separates_the_fes_procrustes_arms(config):
    # _run_id separates proc-on from proc-off, but the wandb run id is
    # sha1(group + seed) and wandb.init resumes on a match, so two f+es arms at
    # one seed would write into the same wandb run.
    tiny_run(config, "f+es")
    config["spectral"]["update_mode"] = "update"
    config["spectral"]["solver"] = "chebyshev"

    ids = set()
    for value in ("auto", True):
        config["spectral"]["use_procrustes"] = value
        group, _, _ = main._wandb_meta()
        ids.add(main._wandb_id(f"{group}_s{config['seed']}"))

    assert len(ids) == 2


# --------------------------------------------------------------------- #
# 2. structure_model.freeze_sfv
# --------------------------------------------------------------------- #


def test_freeze_defaults_off(config):
    assert config["structure_model"]["freeze_sfv"] is False
    assert _sfv_flag("freeze_sfv") is False


def test_the_flags_tolerate_a_config_predating_them(config, tmp_path):
    identity_cfg(config, tmp_path)
    config["model"]["data_type"] = "feature"
    server = DynamicServer(make_toy_snapshots(N=30, num_snaps=2, seed=7))
    baseline = server._run_id()

    del config["structure_model"]["freeze_sfv"]
    del config["structure_model"]["sfv_reset_per_snapshot"]
    del config["gnn"]["encoder_edge_drop"]

    assert _sfv_flag("freeze_sfv") is False
    assert _sfv_flag("sfv_reset_per_snapshot") is False
    assert server._run_id() == baseline


@pytest.mark.parametrize("frozen", [True, False])
def test_a_frozen_sfv_is_absent_from_every_optimizer_param_group(config, frozen):
    tiny_run(config, "f+s")
    config["structure_model"]["freeze_sfv"] = frozen
    server = build_server()
    server.initialize_FL()

    for clf in [server.classifier] + [cl.classifier for cl in server.clients]:
        w = clf.smodel.graph.x
        assert w.requires_grad is (not frozen)
        in_params = any(p is w for p in clf.parameters())
        in_optim = any(
            p is w for group in _make_optimizer(clf).param_groups for p in group["params"]
        )
        assert in_params is (not frozen)
        assert in_optim is (not frozen)


@pytest.mark.parametrize("frozen", [True, False])
def test_a_frozen_sfv_leaves_the_federated_payload(config, frozen):
    tiny_run(config, "f+s")
    config["federated"]["sfv_share"] = "avg"  # the only mode that ships W at all
    config["structure_model"]["freeze_sfv"] = frozen
    server = build_server()
    server.initialize_FL()

    for clf in [server.classifier] + [cl.classifier for cl in server.clients]:
        assert ("SFV" in clf.smodel.state_dict()) is (not frozen)
        assert ("SFV" in clf.state_dict()["smodel"]) is (not frozen)


@pytest.mark.parametrize("frozen", [True, False])
def test_a_frozen_sfv_is_bit_identical_end_to_end(config, monkeypatch, frozen):
    # sfv_share='avg' is the strictest setting: without freeze it moves W for
    # EVERY owner (local training + FedAvg), so an unchanged W under freeze
    # cannot be an artifact of nothing having trained.
    tiny_run(config, "f+s")
    config["federated"]["sfv_share"] = "avg"
    config["structure_model"]["freeze_sfv"] = frozen
    server = build_server()

    started = {}
    original = DynamicServer.initialize_FL

    def spy(self, *args, **kwargs):
        # joint_train_w re-initializes, so the run's true starting W is only
        # observable from inside it.
        out = original(self, *args, **kwargs)
        started["sfvs"] = all_sfvs(self)
        return out

    monkeypatch.setattr(DynamicServer, "initialize_FL", spy)
    server.joint_train_w(FL=True)

    moved = [
        not torch.equal(before, after)
        for before, after in zip(started["sfvs"], all_sfvs(server))
    ]
    assert all(m is not frozen for m in moved)


# --------------------------------------------------------------------- #
# 3. structure_model.sfv_reset_per_snapshot
# --------------------------------------------------------------------- #


def test_reset_defaults_off(config):
    assert config["structure_model"]["sfv_reset_per_snapshot"] is False
    assert _sfv_flag("sfv_reset_per_snapshot") is False


def run_with_reset_trace(config, monkeypatch, reset, num_snaps=4):
    """Run once, recording every _reset_sfvs draw as seen by every owner."""
    config["structure_model"]["sfv_reset_per_snapshot"] = reset
    server = build_server(num_snaps=num_snaps)
    draws = []

    def spy(self):
        _ORIG_RESET_SFVS(self)
        draws.append(all_sfvs(self))

    monkeypatch.setattr(DynamicServer, "_reset_sfvs", spy)
    server.joint_train_w(FL=True)
    return server, draws


def test_reset_redraws_once_per_snapshot_for_every_owner(config, monkeypatch):
    tiny_run(config, "f+s")
    server, draws = run_with_reset_trace(config, monkeypatch, reset=True)

    n_tasks = len(server.global_snaps) - 1
    assert len(draws) == n_tasks
    for owners in draws:
        assert len(owners) == 1 + len(server.clients)
        # one draw shared by all owners: clients must not start a snapshot from
        # different inits, or the arm would change two things at once.
        assert all(torch.equal(owners[0], w) for w in owners[1:])
    for previous, current in zip(draws, draws[1:]):
        assert not torch.equal(previous[0], current[0])


def test_no_reset_happens_at_the_default(config, monkeypatch):
    tiny_run(config, "f+s")
    _, draws = run_with_reset_trace(config, monkeypatch, reset=False)

    assert draws == []


@pytest.mark.parametrize("reset", [True, False])
def test_the_carry_across_the_snapshot_boundary_is_exactly_what_the_flag_controls(
    config, monkeypatch, reset
):
    # sfv_share='local' keeps W out of state_dict, so between the last training
    # step of snapshot t and the first of t+1 the ONLY thing that can touch it
    # is the reset.
    tiny_run(config, "f+s")
    config["federated"]["sfv_share"] = "local"
    config["structure_model"]["sfv_reset_per_snapshot"] = reset
    server = build_server()

    trace = []
    original = DynamicClient.local_finetune

    def spy(self, t, local_epochs, loss_fn):
        before = sfv_of(self.classifier)
        original(self, t, local_epochs, loss_fn)
        trace.append((self.id, t, before, sfv_of(self.classifier)))

    monkeypatch.setattr(DynamicClient, "local_finetune", spy)
    server.joint_train_w(FL=True)

    boundaries = 0
    for client_id in {row[0] for row in trace}:
        rows = [row for row in trace if row[0] == client_id]
        assert rows and all(not torch.equal(r[2], r[3]) for r in rows)  # W really trains
        for previous, current in zip(rows, rows[1:]):
            if previous[1] == current[1]:
                continue
            boundaries += 1
            assert torch.equal(previous[3], current[2]) is (not reset)
    assert boundaries > 0


def test_two_same_seed_runs_see_the_same_reset_sequence(config, monkeypatch):
    tiny_run(config, "f+s")
    _, first = run_with_reset_trace(config, monkeypatch, reset=True)
    _, second = run_with_reset_trace(config, monkeypatch, reset=True)

    assert len(first) == len(second) > 0
    for a, b in zip(first, second):
        assert torch.equal(a[0], b[0])


@pytest.mark.parametrize("data_type", ["feature", "f+pe"])
def test_reset_is_inert_where_the_smodel_owns_no_sfv(config, data_type):
    tiny_run(config, data_type)
    config["structure_model"]["sfv_reset_per_snapshot"] = True
    server = build_server()
    server.initialize_FL()
    before = dynamic_server._clone_state(server.state_dict())

    server._reset_sfvs()  # must return without raising and without writing

    assert _get_sfv(server.classifier) is None
    assert all(sfv_of(cl.classifier) is None for cl in server.clients)
    after = _flat(server.state_dict())
    for key, value in _flat(before).items():
        assert torch.equal(value, after[key])


def test_wandb_identity_separates_the_sfv_lifetime_arms(config):
    # carry / reset / freeze are three arms of one ablation that differ by
    # nothing else, so a shared wandb id makes the later arm resume and
    # overwrite the earlier one's history and summary.
    tiny_run(config, "f+s")
    config["spectral"]["update_mode"] = "update"

    ids = set()
    for reset, freeze in ((False, False), (True, False), (False, True)):
        config["structure_model"]["sfv_reset_per_snapshot"] = reset
        config["structure_model"]["freeze_sfv"] = freeze
        group, _, _ = main._wandb_meta()
        ids.add(main._wandb_id(f"{group}_s{config['seed']}"))

    assert len(ids) == 3


@pytest.mark.parametrize("key", ["sfv_reset_per_snapshot", "freeze_sfv"])
def test_the_sfv_lifetime_arms_get_distinct_run_ids(config, tmp_path, key):
    identity_cfg(config, tmp_path)
    config["model"]["data_type"] = "f+s"
    server = DynamicServer(make_toy_snapshots(N=30, num_snaps=2, seed=7))

    carry = server._run_id()
    config["structure_model"][key] = True
    assert server._run_id() != carry


@pytest.mark.parametrize("key", ["sfv_reset_per_snapshot", "freeze_sfv"])
@pytest.mark.parametrize("bad", ["true", 1, None, "yes"])
def test_assert_cfg_rejects_a_non_bool_sfv_flag(config, key, bad):
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "keep"
    config["structure_model"][key] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert f"structure_model.{key}" in str(exc.value)


# --------------------------------------------------------------------- #
# 4. gnn.encoder_edge_drop
# --------------------------------------------------------------------- #


def make_snap(n_edges):
    """A minimal snapshot stand-in for the precompute helper. Every column is a
    distinct pair, and edge_attr row i carries i, so a misaligned attr slice or
    a duplicated column is visible rather than absorbed."""
    n_nodes = n_edges + 1
    ei = torch.stack([torch.arange(n_edges), torch.arange(n_edges) + 1]) % max(n_nodes, 1)
    snap = types.SimpleNamespace(edge_index=ei)
    snap.edge_attr = torch.arange(n_edges, dtype=torch.float32).unsqueeze(-1)
    return snap


def rng_states():
    return torch.get_rng_state(), random.getstate(), np.random.get_state()


def rng_states_equal(a, b):
    if not torch.equal(a[0], b[0]) or a[1] != b[1]:
        return False
    return all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
        for x, y in zip(a[2], b[2])
    )


def test_drop_defaults_off(config):
    assert config["gnn"]["encoder_edge_drop"] == 0.0


def test_zero_drop_attaches_nothing_and_draws_no_rng(config):
    snaps = [make_snap(40), make_snap(0), make_snap(1)]
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    before = rng_states()

    _precompute_encoder_edge_drop(snaps, 0.0, 123)

    # a single draw here would shift every downstream negative sample and every
    # weight init, silently changing the default arm's numbers.
    assert rng_states_equal(before, rng_states())
    for snap in snaps:
        assert not hasattr(snap, MP_EDGE_INDEX_ATTR)
        assert not hasattr(snap, MP_EDGE_ATTR_ATTR)
        assert _mp_graph(snap)[0] is snap.edge_index


def test_zero_drop_clears_a_previously_attached_subset(config):
    snaps = [make_snap(40)]
    _precompute_encoder_edge_drop(snaps, 0.5, 123)
    assert hasattr(snaps[0], MP_EDGE_INDEX_ATTR)

    _precompute_encoder_edge_drop(snaps, 0.0, 123)

    assert not hasattr(snaps[0], MP_EDGE_INDEX_ATTR)
    assert not hasattr(snaps[0], MP_EDGE_ATTR_ATTR)


@pytest.mark.parametrize("p", [0.25, 0.5, 0.9])
def test_the_kept_subset_is_a_subset_of_the_right_size_with_aligned_attrs(config, p):
    snap = make_snap(200)
    _precompute_encoder_edge_drop([snap], p, 5)

    kept = snap.mp_edge_index
    full = snap.edge_index
    assert kept.size(1) == pytest.approx(full.size(1) * (1.0 - p), abs=1.0)
    columns = {tuple(c) for c in kept.t().tolist()}
    assert columns <= {tuple(c) for c in full.t().tolist()}
    # edge_attr[i] encodes column i, so this pins row-alignment rather than
    # merely equal lengths.
    idx = snap.mp_edge_attr.squeeze(-1).to(torch.long)
    assert torch.equal(full[:, idx], kept)


@pytest.mark.parametrize("n_edges,expected", [(0, None), (1, None), (2, None), (3, 2)])
def test_the_drop_can_never_starve_a_snapshot_to_one_edge(config, n_edges, expected):
    # the edge encoder's BatchNorm raises on a 1-row batch, so p must not be
    # honoured past that floor.
    snap = make_snap(n_edges)
    _precompute_encoder_edge_drop([snap], 0.99, 5)

    if expected is None:
        assert not hasattr(snap, MP_EDGE_INDEX_ATTR)  # full set kept as-is
    else:
        assert snap.mp_edge_index.size(1) == expected


def test_the_mask_ignores_the_global_rng_stream(config):
    a, b = [make_snap(200) for _ in range(2)], [make_snap(200) for _ in range(2)]
    torch.manual_seed(11)
    _precompute_encoder_edge_drop(a, 0.5, 999)
    torch.manual_seed(4242)
    _precompute_encoder_edge_drop(b, 0.5, 999)

    for x, y in zip(a, b):
        assert torch.equal(x.mp_edge_index, y.mp_edge_index)


def test_the_mask_survives_clone_and_to_on_a_real_snapshot(config):
    # the training batch is a clone of a .to(device)'d snapshot; if the
    # attribute did not survive, training would silently see the full graph
    # while eval saw the subset.
    client_snaps = partition_snapshots(make_toy_snapshots(N=30, num_snaps=2, seed=7), 1)
    snap = client_snaps[0][0]
    _precompute_encoder_edge_drop([snap], 0.5, 3)

    for derived in (snap.clone(), snap.to("cpu"), snap.to("cpu").clone()):
        assert torch.equal(_mp_graph(derived)[0], snap.mp_edge_index)


def test_zero_drop_is_bit_identical_to_not_calling_the_helper(config, monkeypatch):
    tiny_run(config, "feature")

    def run(patched):
        server = build_server()
        if patched:
            monkeypatch.setattr(
                dynamic_server, "_precompute_encoder_edge_drop", lambda *a, **k: None
            )
        out = server.joint_train_w(FL=True)
        monkeypatch.undo()
        return out["mrr_history"]

    assert run(patched=False) == run(patched=True)


def test_the_encoder_and_only_the_encoder_sees_the_subset(config, monkeypatch):
    tiny_run(config, "feature")
    config["gnn"]["encoder_edge_drop"] = 0.5
    server = build_server()

    seen = []
    original = ModelBinder.encode

    def spy(self, x, edge_index, hs=None, **kwargs):
        seen.append(edge_index.clone())
        return original(self, x, edge_index, hs=hs, **kwargs)

    monkeypatch.setattr(ModelBinder, "encode", spy)
    server.joint_train_w(FL=True)
    monkeypatch.undo()

    masks = [cl.snaps[t].mp_edge_index for cl in server.clients
             for t in range(len(cl.snaps)) if hasattr(cl.snaps[t], MP_EDGE_INDEX_ATTR)]
    full = [cl.snaps[t].edge_index for cl in server.clients for t in range(len(cl.snaps))]
    assert seen and masks
    # every message-passing graph the encoder ever saw is one of the FIXED
    # per-(owner, snapshot) subsets -- not a per-epoch resample.
    for observed in seen:
        assert any(torch.equal(observed, m) for m in masks)
        assert not any(torch.equal(observed, f) for f in full)
    assert len(seen) > len(masks)  # reused across epochs and rounds


def test_the_drop_reaches_neither_the_targets_nor_the_splits_nor_the_union(config):
    tiny_run(config, "feature")
    config["metric"]["repeat_new_split"] = True  # forces the cumulative union

    def run(p):
        config["gnn"]["encoder_edge_drop"] = p
        server = build_server()
        server.joint_train_w(FL=True)
        return server

    baseline, dropped = run(0.0), run(0.5)

    assert torch.equal(baseline._cum_edges, dropped._cum_edges)
    for a, b in zip(baseline.global_snaps, dropped.global_snaps):
        assert torch.equal(a.edge_index, b.edge_index)
        for split in ("pos_train", "pos_val", "pos_test"):
            assert torch.equal(getattr(a, split), getattr(b, split))
    for ca, cb in zip(baseline.clients, dropped.clients):
        for a, b in zip(ca.snaps, cb.snaps):
            assert torch.equal(a.edge_index, b.edge_index)
            for split in ("pos_train", "pos_val", "pos_test"):
                assert torch.equal(getattr(a, split), getattr(b, split))


def test_the_global_snapshots_never_receive_the_drop(config):
    tiny_run(config, "feature")
    config["gnn"]["encoder_edge_drop"] = 0.5
    server = build_server()
    server.joint_train_w(FL=True)

    assert not any(hasattr(s, MP_EDGE_INDEX_ATTR) for s in server.global_snaps)
    assert all(hasattr(s, MP_EDGE_INDEX_ATTR) for cl in server.clients for s in cl.snaps)


def test_client_abstention_is_unchanged_by_the_drop(config):
    # can_train reads the FULL edge set, so a client that would have trained
    # must still train -- otherwise the drop changes federation, not just
    # message passing.
    tiny_run(config, "feature")
    server = build_server(num_clients=4)
    for c, cl in enumerate(server.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))

    def abstention():
        return [[cl.can_train(t) for t in range(len(cl.snaps) - 1)] for cl in server.clients]

    baseline = abstention()
    for c, cl in enumerate(server.clients):
        _precompute_encoder_edge_drop(cl.snaps, 0.9, 500000 * (c + 1))

    assert any(hasattr(s, MP_EDGE_INDEX_ATTR) for cl in server.clients for s in cl.snaps)
    assert abstention() == baseline


def test_run_id_and_wandb_identity_separate_the_drop_arms(config, tmp_path):
    identity_cfg(config, tmp_path)
    config["model"]["data_type"] = "feature"
    server = DynamicServer(make_toy_snapshots(N=30, num_snaps=2, seed=7))

    config["gnn"]["encoder_edge_drop"] = 0.0
    default_id = server._run_id()
    assert not any(part.startswith("edrop") for part in default_id.split("_"))

    ids, groups = {default_id}, {main._wandb_meta()[0]}
    for p in (0.25, 0.5, 0.75):
        config["gnn"]["encoder_edge_drop"] = p
        ids.add(server._run_id())
        groups.add(main._wandb_meta()[0])

    assert len(ids) == 4
    assert len(groups) == 4


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1, True, "0.5", None])
def test_assert_cfg_rejects_an_out_of_range_drop(config, bad):
    config["model"]["data_type"] = "feature"
    config["gnn"]["encoder_edge_drop"] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "gnn.encoder_edge_drop" in str(exc.value)


@pytest.mark.parametrize("good", [0.0, 0.25, 0.99])
def test_assert_cfg_accepts_the_supported_drop_range_and_absence(config, good):
    config["model"]["data_type"] = "feature"
    config["gnn"]["encoder_edge_drop"] = good
    assert_cfg(config)

    del config["gnn"]["encoder_edge_drop"]  # a config predating the key
    assert_cfg(config)
