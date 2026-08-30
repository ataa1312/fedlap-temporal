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

import os

import pytest
import torch

import src
from config.assertions import assert_cfg
from config.config import get_default_config
from src.dynamic_server import (
    _FP_EXCLUDE_PATH,
    _FP_EXCLUDE_TOP,
    DynamicServer,
    _config_fingerprint,
    _get_sfv,
    _set_sfv,
)
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


def explicit_id(server):
    """The identity minus the completeness fingerprint.

    _run_id now ends with cfg-<hash> over every config value that can move a
    number, so NO identity is byte-stable across a config change any more --
    that is the point of the backstop, and the user traded byte-stability away
    for it deliberately. The EXPLICIT tokens still carry the arm's meaning and
    are still what a human reads, and "this knob adds no token at its default"
    is still a real property worth policing, so the assertions that used to pin
    a literal now pin it with the hash removed.
    """
    return explicit_id_of(server._run_id())


def explicit_id_of(run_id):
    """explicit_id for an identity string already in hand."""
    return "_".join(p for p in run_id.split("_") if not p.startswith("cfg-"))


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
    # literals read off `git show HEAD~2:src/dynamic_server.py`. These pin the
    # EXPLICIT tokens only: since the completeness fingerprint landed, no identity
    # is byte-stable across a config change and old checkpoints no longer resolve
    # -- see test_a_pre_fingerprint_checkpoint_is_not_adopted.
    server = identity_server(config, tmp_path)

    config["model"]["data_type"] = "feature"
    assert explicit_id(server) == "uci_gru_feature_C1_s1234"

    # f+pe under `keep` is unchanged: procrustes only runs under update/recompute,
    # so no proc- token is appended there.
    config["model"]["data_type"] = "f+pe"
    config["spectral"]["update_mode"] = "keep"
    assert explicit_id(server) == "uci_gru_f+pe_C1_um-keep_pe50_basis-laplacian_s1234"

    # Under update/recompute f+pe DOES gain proc-, deliberately: the procrustes
    # branch in get_spectral_features is data-type agnostic and changes the
    # numbers, so those identities were previously ambiguous and had to change.
    # Byte-identity is therefore preserved only where the pre-change identity was
    # already unambiguous.
    for update_mode in ("update", "recompute"):
        config["spectral"]["update_mode"] = update_mode
        assert explicit_id(server) == (
            f"uci_gru_f+pe_C1_um-{update_mode}_pe50_basis-laplacian_proc-on_s1234")


def test_a_pre_fingerprint_checkpoint_is_not_adopted(config, tmp_path):
    """Backward compatibility was traded away deliberately, so prove the trade is
    clean: an old checkpoint must be MISSED, never silently adopted.

    This test previously asserted the opposite -- that a payload stamped with the
    pre-change identity still resumed. The user has since ruled that identity
    correctness outranks preserving banked ids, so the contract inverted. The
    dangerous outcome was never "the old checkpoint stops loading" (that costs a
    re-run); it was "the old checkpoint loads into a run whose config it does not
    describe", which is a wrong-arm resume reported as a real result.
    """
    identity_server(config, tmp_path)
    config["model"]["data_type"] = "f+pe"

    global_snaps = make_toy_snapshots()
    server = make_server(global_snaps, partition_snapshots(global_snaps, 1))
    server.initialize_FL()
    server._save_partial_ckpt(0, None, [], [])
    server._save_done_ckpt({"mean_mrr": 0.0, "mrr_history": []})
    current = server._run_id()
    stale = explicit_id_of(current)
    assert stale != current                     # the premise: a fingerprint is there

    # 1. the FILENAME barrier: the old identity names a file that no longer exists
    for ext in (".ckpt", ".done"):
        os.rename(os.path.join(tmp_path, current + ext),
                  os.path.join(tmp_path, stale + ext))
    assert server._load_partial_ckpt() is None
    assert server._load_done_ckpt() is None

    # 2. the CONTENT barrier, independently: force the filename to match and the
    #    stored run_id still refuses. Either alone would do; both must hold, or a
    #    stale payload could ride in under a name that happens to collide.
    for ext in (".ckpt", ".done"):
        os.rename(os.path.join(tmp_path, stale + ext),
                  os.path.join(tmp_path, current + ext))
        payload = torch.load(os.path.join(tmp_path, current + ext),
                             map_location="cpu", weights_only=False)
        payload["run_id"] = stale
        torch.save(payload, os.path.join(tmp_path, current + ext))
    assert server._load_partial_ckpt() is None
    assert server._load_done_ckpt() is None


# ---- the completeness fingerprint ---- #


FP_MUST_SEPARATE = [
    (("metric", "mrr_filter"), "snapshot"), (("metric", "hard_neg"), "degree"),
    (("metric", "mrr_method"), "min"), (("optim", "base_lr"), 0.5),
    (("model", "iterations"), 7), (("experimental", "rank_eval_multiplier"), 5),
    (("spectral", "spectral_len"), 99), (("dataset", "split"), [0.7, 0.2, 0.1]),
    (("gnn", "dropout"), 0.9), (("train", "num_epochs"), 5),
    (("structure_model", "num_structural_features"), 64),
    # structure_type='node2vec' reaches find_node2vect_embedings, which reads
    # these at call time; assert_cfg does not restrict structure_type, so the
    # path is one --set away and the section is NOT dormant.
    (("node2vec", "walk_length"), 5), (("node2vec", "p"), 0.5),
    (("node2vec", "context_size"), 3),
]
FP_MUST_NOT_SEPARATE = [
    (("metric", "snapshot_record_dir"), "/tmp/x"), (("train", "ckpt_dir"), "/tmp/y"),
    (("train", "auto_resume"), True), (("train", "ckpt_period"), 3),
    (("train", "ckpt_clean"), False), (("dataset", "path"), "/other/root"),
    (("wandb", "mode"), "online"), (("wandb", "group_suffix"), "z"),
    (("device",), "cuda"), (("outdir",), "/o"), (("num_threads",), 1),
    (("num_runs",), 9), (("print",), "file"), (("remark",), "hello"),
]


def fingerprint_with(path=None, value=None):
    cfg = get_default_config()
    src.config._registry.clear()
    src.config._registry.update(cfg._registry)
    if path is not None:
        node = src.config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return _config_fingerprint()


@pytest.mark.parametrize("path,value", FP_MUST_SEPARATE)
def test_the_fingerprint_separates_a_knob_that_moves_a_number(config, path, value):
    assert fingerprint_with(path, value) != fingerprint_with()


@pytest.mark.parametrize("path,value", FP_MUST_NOT_SEPARATE)
def test_the_fingerprint_ignores_a_knob_that_cannot_move_a_number(config, path, value):
    # Every exclusion is a hole in the backstop. seed is excluded because it has
    # its own token; the rest are either read nowhere in live code, or (dataset.path,
    # train.*) excluded as a stated trade -- see _FP_EXCLUDE_* in dynamic_server.
    assert fingerprint_with(path, value) == fingerprint_with()


def test_the_seed_is_not_in_the_fingerprint(config):
    # it already has its own token; hashing it too would be harmless but would
    # make the hash change for every seed, hiding real config differences.
    assert fingerprint_with(("seed",), 9999) == fingerprint_with()


def test_every_other_config_leaf_reaches_the_fingerprint(config):
    """The backstop's whole claim: nothing is missed. Perturb EVERY leaf and
    assert the only inert ones are the declared exclusions -- a walk that stopped
    recursing, or an exclusion added by accident, fails here."""
    from config.registry import Registry

    def leaves(node, path=()):
        if isinstance(node, Registry):
            for key in sorted(iter(node)):
                yield from leaves(node[key], path + (key,))
        else:
            yield path, node

    def perturb(v):
        if isinstance(v, bool):
            return not v
        if isinstance(v, (int, float)):
            return v + 1
        if isinstance(v, str):
            return v + "_X"
        if isinstance(v, list):
            return [*v, 0]
        return "SENTINEL"

    base = fingerprint_with()
    inert = {".".join(p) for p, v in leaves(get_default_config())
             if fingerprint_with(p, perturb(v)) == base}
    declared = {k for k in _FP_EXCLUDE_TOP} | {".".join(p) for p in _FP_EXCLUDE_PATH}
    unaccounted = {k for k in inert if k.split(".")[0] not in declared and k not in declared}

    assert unaccounted == set()
    assert "node2vec.walk_length" not in inert     # the exclusion that was wrong


def test_the_fingerprint_is_stable_across_processes(config):
    """A hash that differs between processes means a run cannot find its own
    checkpoint. repr() of a set or a dict is order-dependent and repr() of an
    object carries an address, so leaves are canonically encoded instead."""
    import subprocess
    import sys as _sys

    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "import src\n"
        "from config.config import get_default_config\n"
        "from src.dynamic_server import _config_fingerprint\n"
        "c = get_default_config()\n"
        "src.config._registry.clear(); src.config._registry.update(c._registry)\n"
        "print(_config_fingerprint())\n"
    ) % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = set()
    for seed in ("0", "1", "123456789"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out.add(subprocess.run([_sys.executable, "-c", script], capture_output=True,
                               text=True, env=env).stdout.strip())

    assert len(out) == 1 and out != {""}
    assert out == {fingerprint_with()}


@pytest.mark.parametrize("value,order_matters", [
    ({"b": 2, "a": 1}, False), ({"gamma", "alpha", "beta"}, False),
    ([3, 1, 2], True), ((3, 1, 2), True),
])
def test_container_leaves_are_canonically_encoded(config, value, order_matters):
    # dict/set repr order is insertion- or PYTHONHASHSEED-dependent, so those are
    # sorted; list order is MEANINGFUL (dataset.split [0.8,0.1,0.1] is not
    # [0.1,0.8,0.1]) so it is preserved.
    reordered = (dict(reversed(list(value.items()))) if isinstance(value, dict)
                 else type(value)(reversed(list(value))))
    same = fingerprint_with(("gnn", "dropout"), value) == \
        fingerprint_with(("gnn", "dropout"), reordered)

    assert same is not order_matters


def test_an_unencodable_leaf_fails_loudly(config):
    # a value whose repr carries a memory address would hash differently every
    # process; refusing at startup beats an identity that silently drifts.
    with pytest.raises(TypeError) as exc:
        fingerprint_with(("gnn", "dropout"), object())

    assert "no stable encoding" in str(exc.value)


def test_the_identity_carries_exactly_one_fingerprint_just_before_the_seed(
    config, tmp_path
):
    server = identity_server(config, tmp_path)

    parts = server._run_id().split("_")
    stamps = [i for i, p in enumerate(parts) if p.startswith("cfg-")]

    assert len(stamps) == 1
    assert parts[stamps[0] + 1] == f"s{config['seed']}"
    assert len(parts[stamps[0]]) == len("cfg-") + 8


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
