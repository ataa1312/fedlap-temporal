"""Run identity coverage (commit bdaa550).

`_run_id()` is the only thing standing between a re-launched run and a foreign
checkpoint: under train.auto_resume two runs that share an identity share a
.ckpt and a .done path, so one silently adopts or short-circuits the other. The
f+s case is the live one -- basis_source is the placebo switch, so before this
change a real arm and its shuffled_fixed control resolved to the same file.
These pin the distinctions, the byte-identity of the default-solver path (old
checkpoints must keep loading), the solver validation, and the get_SFV/set_SFV
protocol that _get_sfv/_set_sfv now go through.
"""

import pytest
import torch

import src
from config.assertions import assert_cfg
from src.dynamic_server import DynamicServer, _get_sfv, _set_sfv
from src.GNN.fed_dynamic_classifier import (
    DynamicSInvariant,
    DynamicSSignNet,
    make_sgraph,
)
from src.utils.graph_partitioning import partition_snapshots
from test_checkpoint_wandb import make_toy_snapshots, make_server, seed_all


SOLVERS = ("arnoldi", "exact", "chebyshev")
PLACEBOS = ("random", "shuffled", "random_fixed", "shuffled_fixed")
SPECTRAL_TYPES = ("f+s", "structure", "f+pe", "f+es")


def set_identity_config(cfg, tmp_path):
    """The knobs _run_id reads, all pinned, so a test can vary exactly one."""
    cfg["dataset"]["name"] = "uci"
    cfg["dataset"]["snapshot_freq"] = "W"
    cfg["gnn"]["embed_update_method"] = "gru"
    cfg["subgraph"]["num_subgraphs"] = 1
    cfg["seed"] = 1234
    cfg["experimental"]["deterministic"] = False
    cfg["federated"]["sfv_share"] = "local"
    cfg["spectral"]["update_mode"] = "keep"
    cfg["spectral"]["basis_source"] = "laplacian"
    cfg["spectral"]["solver"] = "arnoldi"
    cfg["spectral"]["pe_dim"] = 50
    cfg["spectral"]["use_procrustes"] = True
    cfg["spectral"]["es_features"] = "spec"
    cfg["spectral"]["es_spec_parts"] = "phi+cos+lev"
    cfg["train"]["ckpt_dir"] = str(tmp_path)
    cfg["train"]["ckpt_clean"] = False


def identity_server(cfg, tmp_path):
    set_identity_config(cfg, tmp_path)
    return DynamicServer(make_toy_snapshots())


# ---- the live bug: basis_source on f+s / structure ---- #


@pytest.mark.parametrize("data_type", ["f+s", "structure"])
@pytest.mark.parametrize("update_mode", ["keep", "update", "recompute"])
def test_basis_source_separates_an_arm_from_its_placebo(
    config, tmp_path, data_type, update_mode
):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = data_type
    config["spectral"]["update_mode"] = update_mode

    real = server._run_id()
    ids = {real}
    for placebo in PLACEBOS:
        config["spectral"]["basis_source"] = placebo
        ids.add(server._run_id())

    assert len(ids) == 1 + len(PLACEBOS)


def test_a_placebo_cannot_adopt_the_real_arms_checkpoint(config, tmp_path):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = "f+s"

    global_snaps = make_toy_snapshots()
    server = make_server(global_snaps, partition_snapshots(global_snaps, 1))
    server.initialize_FL()
    server._save_partial_ckpt(0, None, [], [])
    server._save_done_ckpt({"mean_mrr": 0.0, "mrr_history": []})

    # the whole point of the control: the placebo must run, not inherit the
    # result it exists to be compared against.
    config["spectral"]["basis_source"] = "shuffled_fixed"
    assert server._load_partial_ckpt() is None
    assert server._load_done_ckpt() is None

    config["spectral"]["basis_source"] = "laplacian"
    assert server._load_partial_ckpt() is not None
    assert server._load_done_ckpt() is not None


# ---- f+es: the branch that carried nothing ---- #


def test_fes_identity_separates_every_edge_score_knob(config, tmp_path):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = "f+es"
    config["spectral"]["solver"] = "chebyshev"  # f+es forbids arnoldi

    base = server._run_id()
    variants = {
        "es_spec_parts": ("phi", "cos+lev", "phi+lev"),
        "es_features": ("persist", "both"),
        "basis_source": PLACEBOS,
        "pe_dim": (16, 32),
        "update_mode": ("update", "recompute"),
    }
    ids = {base}
    for key, values in variants.items():
        original = config["spectral"][key]
        for value in values:
            config["spectral"][key] = value
            ids.add(server._run_id())
        config["spectral"][key] = original

    assert len(ids) == 1 + sum(len(v) for v in variants.values())


def test_fes_identity_separates_procrustes_under_update_and_recompute(config, tmp_path):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = "f+es"
    config["spectral"]["solver"] = "chebyshev"

    for update_mode in ("update", "recompute"):
        config["spectral"]["update_mode"] = update_mode
        config["spectral"]["use_procrustes"] = True
        on = server._run_id()
        config["spectral"]["use_procrustes"] = False
        assert server._run_id() != on


# ---- solver ---- #


@pytest.mark.parametrize("data_type", SPECTRAL_TYPES)
def test_non_default_solver_yields_a_distinct_identity(config, tmp_path, data_type):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = data_type

    ids = {}
    for solver in SOLVERS:
        config["spectral"]["solver"] = solver
        ids[solver] = server._run_id()

    assert len(set(ids.values())) == len(SOLVERS)
    for solver in ("exact", "chebyshev"):
        assert f"solver-{solver}" in ids[solver].split("_")
    assert not any(p.startswith("solver-") for p in ids["arnoldi"].split("_"))


def test_default_solver_identity_is_byte_identical_to_pre_change(config, tmp_path):
    # literals read off `git show HEAD~2:src/dynamic_server.py`; checkpoints
    # already on the cluster resolve to these paths and must keep loading.
    server = identity_server(config, tmp_path)

    config["model"]["data_type"] = "feature"
    assert server._run_id() == "uci_gru_feature_C1_s1234"

    # f+pe under `keep` is unchanged: procrustes only runs under update/recompute,
    # so no proc- token is appended there.
    config["model"]["data_type"] = "f+pe"
    config["spectral"]["update_mode"] = "keep"
    assert server._run_id() == "uci_gru_f+pe_C1_um-keep_pe50_basis-laplacian_s1234"

    # Under update/recompute f+pe DOES gain proc-, deliberately: the procrustes
    # branch in get_spectral_features is data-type agnostic and changes the
    # numbers, so those identities were previously ambiguous and had to change.
    # Byte-identity is therefore preserved only where the pre-change identity was
    # already unambiguous.
    for update_mode in ("update", "recompute"):
        config["spectral"]["update_mode"] = update_mode
        rid = server._run_id()
        assert rid == (f"uci_gru_f+pe_C1_um-{update_mode}_pe50_basis-laplacian"
                       "_proc-on_s1234")


def test_a_default_solver_run_resumes_a_checkpoint_written_before_the_change(
    config, tmp_path
):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = "f+pe"

    global_snaps = make_toy_snapshots()
    server = make_server(global_snaps, partition_snapshots(global_snaps, 1))
    server.initialize_FL()
    server._save_partial_ckpt(0, None, [], [])

    # rewrite the payload under the pre-change identity, byte for byte
    ckpt_path, _ = server._ckpt_paths()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["run_id"] = "uci_gru_f+pe_C1_um-keep_pe50_basis-laplacian_s1234"
    torch.save(ckpt, ckpt_path)

    assert server._load_partial_ckpt() is not None


@pytest.mark.parametrize("bad", ["chebychev", "lanczos", "krylov", "", None])
def test_assert_cfg_rejects_an_unknown_solver(config, bad):
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "keep"
    config["spectral"]["solver"] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.solver" in str(exc.value)


def test_assert_cfg_accepts_every_supported_solver_and_absence(config):
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "keep"

    for solver in SOLVERS:
        config["spectral"]["solver"] = solver
        assert_cfg(config)

    del config["spectral"]["solver"]  # a config predating the key
    assert_cfg(config)


# ---- SFV capture goes through the smodel protocol ---- #


class _ProtocolOnlySmodel:
    """get_SFV/set_SFV disagree with .graph.x on purpose: a capture that reads
    the graph directly (the pre-change contract) reads the wrong tensor."""

    def __init__(self):
        import types

        self.owned = torch.full((3, 3), 7.0)
        self.graph = types.SimpleNamespace(x=torch.zeros(3, 3))
        self.set_calls = []

    def get_SFV(self):
        return self.owned

    def set_SFV(self, w):
        self.set_calls.append(w)
        with torch.no_grad():
            self.owned.copy_(w.to(self.owned.device))


class _NoSFVSmodel:
    """An smodel owning no SFV -- SignNet and the f+es Invariant both look like
    this, and the Invariant has no `graph` attribute at all."""

    def get_SFV(self):
        return None

    def set_SFV(self, w):
        return


class _Clf:
    def __init__(self, smodel):
        self.smodel = smodel


def test_capture_reads_the_protocol_not_the_graph():
    smodel = _ProtocolOnlySmodel()

    captured = _get_sfv(_Clf(smodel))

    assert torch.equal(captured, torch.full((3, 3), 7.0))
    assert not torch.equal(captured, smodel.graph.x)


def test_restore_writes_through_the_protocol():
    smodel = _ProtocolOnlySmodel()

    _set_sfv(_Clf(smodel), torch.full((3, 3), 5.0))

    assert len(smodel.set_calls) == 1
    assert torch.equal(smodel.owned, torch.full((3, 3), 5.0))
    assert torch.equal(smodel.graph.x, torch.zeros(3, 3))


def test_capture_is_detached_and_on_cpu():
    smodel = _ProtocolOnlySmodel()
    smodel.owned = torch.full((3, 3), 7.0, requires_grad=True)

    captured = _get_sfv(_Clf(smodel))

    assert captured.requires_grad is False
    assert captured.device.type == "cpu"


def test_an_smodel_owning_no_sfv_captures_nothing_without_raising():
    clf = _Clf(_NoSFVSmodel())

    assert _get_sfv(clf) is None
    _set_sfv(clf, torch.ones(3, 3))  # no write to perform, no raise


def test_a_classifier_without_an_smodel_captures_nothing():
    class _Bare:
        pass

    assert _get_sfv(_Bare()) is None
    _set_sfv(_Bare(), torch.ones(3, 3))


def test_the_real_edge_score_smodel_has_no_graph_and_captures_nothing(config):
    config["model"]["data_type"] = "f+es"
    config["spectral"]["update_mode"] = "update"
    config["spectral"]["solver"] = "chebyshev"
    config["spectral"]["pe_dim"] = 6
    config["structure_model"]["DGCN_structure_layers_sizes"] = [16]

    smodel = DynamicSInvariant()

    assert not hasattr(smodel, "graph")  # what made the pre-change capture raise
    assert _get_sfv(_Clf(smodel)) is None
    _set_sfv(_Clf(smodel), torch.ones(3, 3))


def test_the_real_signnet_smodel_owns_no_sfv(config):
    # it does have a .graph, and its x is a leaf with requires_grad off -- the
    # pre-change capture stored that unused tensor; the protocol says None.
    config["gnn"]["dims"] = [16, 16]
    config["spectral"]["signnet_phi_dims"] = [16, 16]
    config["spectral"]["signnet_rho_dims"] = [16]
    config["spectral"]["output_bn"] = False

    smodel = DynamicSSignNet(
        make_sgraph(torch.normal(0, 0.05, size=(8, 32), requires_grad=True)),
        out_dim=16,
    )

    assert smodel.graph.x is not None
    assert _get_sfv(_Clf(smodel)) is None
    _set_sfv(_Clf(smodel), torch.ones(8, 32))
    assert smodel.graph.x.requires_grad is False


def test_a_graph_bearing_smodel_round_trips_through_a_checkpoint(config, tmp_path):
    config["dataset"]["name"] = "uci"
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "f+s"
    config["model"]["edge_decoding"] = "concat"
    config["model"]["smodel_type"] = "LanczosLaplace"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["spectral"]["spectral_len"] = 4
    config["spectral"]["update_mode"] = "keep"
    config["federated"]["sfv_share"] = "local"
    config["subgraph"]["num_subgraphs"] = 2
    config["train"]["ckpt_dir"] = str(tmp_path)
    config["train"]["ckpt_clean"] = False
    config["seed"] = 42

    seed_all(42)
    global_snaps = make_toy_snapshots()
    client_snaps = partition_snapshots(global_snaps, 2)
    server = make_server(global_snaps, client_snaps)
    server.initialize_FL()

    saved = [_get_sfv(cl.classifier).clone() for cl in server.clients]
    assert all(w is not None for w in saved)
    server._save_partial_ckpt(0, None, [], [])

    # sfv_share='local' keeps W out of state_dict, so only the checkpoint's own
    # capture can bring it back
    for cl in server.clients:
        with torch.no_grad():
            cl.classifier.smodel.graph.x.fill_(999.0)

    assert server._load_partial_ckpt() is not None
    for cl, w in zip(server.clients, saved):
        assert torch.equal(cl.classifier.smodel.graph.x.detach().cpu(), w)


def test_the_edge_score_run_can_checkpoint_and_resume(config, tmp_path):
    config["dataset"]["name"] = "uci"
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "f+es"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [8, 8]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["structure_model"]["DGCN_structure_layers_sizes"] = [16]
    config["spectral"]["pe_dim"] = 6
    config["spectral"]["solver"] = "chebyshev"
    config["spectral"]["update_mode"] = "update"
    config["train"]["ckpt_dir"] = str(tmp_path)
    config["train"]["ckpt_clean"] = False
    config["seed"] = 42

    seed_all(42)
    global_snaps = make_toy_snapshots()
    server = make_server(global_snaps, partition_snapshots(global_snaps, 2))
    server.initialize_FL()

    server._save_partial_ckpt(0, None, [], [])  # used to raise AttributeError

    ckpt_path, _ = server._ckpt_paths()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["server_sfv"] is None
    assert ckpt["client_sfv"] == [None, None]
    assert server._load_partial_ckpt() is not None
