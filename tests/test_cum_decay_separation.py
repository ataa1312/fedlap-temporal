"""spectral.cum_decay: the age kernel must move the eigenbasis and NOTHING else.

If a kernel leaked into the repeat/new split or into the 'persist' feature, every
arm would measure a different quantity and no two arms could be compared. If it
fails to reach the basis, the arm is a no-op wearing a distinct run id. Both
directions are load-bearing, so both are pinned here, together with the run
identity (defaults must stay byte-identical or banked checkpoints are orphaned),
the config validation and resume parity.
"""

import pytest
import torch
from torch_geometric.data import Data

from config.assertions import assert_cfg
from src.utils.graph_partitioning import partition_snapshots
from test_checkpoint_wandb import (
    make_server,
    make_toy_snapshots,
    seed_all,
    setup_tiny_config,
)
from test_run_identity import explicit_id, explicit_id_of, set_identity_config


# (kernel, cum_decay_param). window=1 keeps only the current snapshot, which is
# the harshest separation probe: it drives ever-seen edges to weight EXACTLY 0.
KERNELS = (("none", 0.9), ("count", 0.9), ("harmonic", 0.9), ("exp", 0.5), ("window", 1))
NON_DEFAULT_KERNELS = KERNELS[1:]

N_NODES = 8
# (0,1) reappears at t=3 after a gap, (1,2) reappears at t=1 and then never again.
SCHEDULE = [
    [(0, 1), (1, 2)],
    [(1, 2), (2, 3)],
    [(3, 4)],
    [(0, 1), (4, 5)],
]

DENSE_NODES = 12
DENSE_SCHEDULE = [
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    [(0, 1), (1, 2), (5, 6), (6, 7), (7, 8)],
    [(0, 1), (2, 3), (8, 9), (9, 10), (10, 11)],
    [(1, 2), (3, 4), (5, 7), (11, 0)],
]


def snapshots(schedule, n_nodes):
    snaps = []
    for edges in schedule:
        ei = (torch.tensor(edges, dtype=torch.long).t() if edges
              else torch.zeros(2, 0, dtype=torch.long))
        snap = Data(x=torch.ones(n_nodes, 1), edge_index=ei,
                    edge_attr=torch.ones(ei.size(1), 1), num_nodes=n_nodes)
        snap.node_ids = torch.arange(n_nodes)
        snaps.append(snap)
    return snaps


def accumulated_server(config, kernel, param, schedule=SCHEDULE, n_nodes=N_NODES):
    """A server walked through `schedule` exactly as joint_train_w walks it."""
    from src.dynamic_server import DynamicServer

    config["spectral"]["cum_decay"] = kernel
    config["spectral"]["cum_decay_param"] = param
    server = DynamicServer(snapshots(schedule, n_nodes))
    for t in range(len(schedule)):
        server._accumulate_cum_edges(t)
    return server


def pair_key(server, u, v):
    n = server.global_snaps[0].num_nodes
    return min(u, v) * n + max(u, v)


def weight_of(server, t, u, v):
    """The kernel weight carried by pair (u, v)'s columns of _cum_edges."""
    w = server._cum_edge_weight(t)
    ce = server._cum_edges
    hit = ((ce[0] == u) & (ce[1] == v)) | ((ce[0] == v) & (ce[1] == u))
    assert hit.any(), "pair is not in the cumulative union"
    return w[hit]


# ---- separation: the kernel must not reach the repeat/new split ---- #


@pytest.mark.parametrize("kernel,param", NON_DEFAULT_KERNELS)
def test_the_cumulative_union_is_identical_under_every_kernel(config, kernel, param):
    reference = accumulated_server(config, "none", 0.9)
    server = accumulated_server(config, kernel, param)

    assert torch.equal(server._cum_edges, reference._cum_edges)


@pytest.mark.parametrize("kernel,param", NON_DEFAULT_KERNELS)
def test_the_repeat_new_split_is_identical_under_every_kernel(config, kernel, param):
    # both orientations of a repeat, plus a pair that was never seen
    probe = torch.tensor([[0, 2, 4, 6, 1], [1, 1, 5, 7, 0]])
    reference = accumulated_server(config, "none", 0.9)
    ref_mask = reference._repeat_mask(probe)
    # the reference must be the mathematically right split, or "identical to the
    # reference" would be satisfied by an all-False mask under every kernel
    assert ref_mask.tolist() == [True, True, True, False, True]

    server = accumulated_server(config, kernel, param)

    assert torch.equal(server._repeat_mask(probe), ref_mask)


def test_an_edge_the_kernel_zeroes_is_still_a_repeat(config):
    # window(1) at t=3 gives (1,2) -- last seen at t=1 -- weight exactly 0.0.
    # The split must not notice: 'repeat' has to mean the same thing in every arm.
    server = accumulated_server(config, "window", 1)

    assert torch.equal(weight_of(server, 3, 1, 2), torch.zeros(2, dtype=torch.float64))
    assert server._repeat_mask(torch.tensor([[1], [2]])).tolist() == [True]
    keys = (torch.minimum(server._cum_edges[0], server._cum_edges[1]) * N_NODES
            + torch.maximum(server._cum_edges[0], server._cum_edges[1]))
    assert pair_key(server, 1, 2) in keys.tolist()


def test_the_persistence_feature_reads_the_unweighted_union(config):
    # f+es serves the cumulative graph to the smodel as a 1-bit "already an edge"
    # feature. window(1) zeroes most of that union for the BASIS; the feature must
    # still see the full union, on the server and on every client.
    def served_keys(kernel, param):
        snaps = snapshots(DENSE_SCHEDULE, DENSE_NODES)
        config["spectral"]["cum_decay"] = kernel
        config["spectral"]["cum_decay_param"] = param
        seed_all(42)
        server = make_server(snaps, partition_snapshots(snaps, 1))
        server.initialize_FL()
        for t in range(len(snaps)):
            server._spectral_step(t, config["model"]["smodel_type"])
        return (
            set(server.classifier.smodel.keys.cpu().tolist()),
            set(server.clients[0].classifier.smodel.keys.cpu().tolist()),
        )

    setup_edge_score_config(config)
    reference = served_keys("none", 0.9)
    expected = {min(u, v) * DENSE_NODES + max(u, v)
                for snap in DENSE_SCHEDULE for u, v in snap}
    assert reference[0] == expected  # C=1: the client owns every node

    assert served_keys("window", 1) == reference
    assert served_keys("count", 0.9) == reference


def setup_edge_score_config(config):
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
    config["spectral"]["pe_dim"] = 4
    config["spectral"]["solver"] = "chebyshev"
    config["spectral"]["update_mode"] = "keep"
    config["subgraph"]["num_subgraphs"] = 1
    config["seed"] = 42


# ---- separation, other direction: the kernel MUST reach the basis ---- #


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


def served_bases(config, data_type, update_mode, solver, kernel, param):
    """Every U handed to the server classifier over a whole run."""
    snaps = snapshots(DENSE_SCHEDULE, DENSE_NODES)
    config["model"]["data_type"] = data_type
    config["spectral"]["update_mode"] = update_mode
    config["spectral"]["solver"] = solver
    config["spectral"]["cum_decay"] = kernel
    config["spectral"]["cum_decay_param"] = param
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


# `keep` is excluded on purpose: at t=0 every kernel evaluates f(0)=1, so the
# weighted operator IS the binary one there and `keep` freezes that basis -- the
# kernel is provably inert under `keep` for mathematical reasons, not plumbing.
@pytest.mark.parametrize(
    "data_type,update_mode,solver",
    [
        ("f+s", "update", "arnoldi"),
        ("f+s", "recompute", "arnoldi"),
        ("f+s", "update", "exact"),
        ("f+s", "update", "chebyshev"),
        ("f+pe", "update", "arnoldi"),
        ("f+pe", "recompute", "arnoldi"),
        ("f+es", "update", "chebyshev"),
    ],
)
def test_an_arm_stamped_with_a_cum_token_actually_gets_the_treatment(
    config, data_type, update_mode, solver
):
    from src.dynamic_server import DynamicServer

    setup_basis_config(config)
    config["model"]["data_type"] = data_type
    config["spectral"]["update_mode"] = update_mode
    config["spectral"]["solver"] = solver
    config["spectral"]["cum_decay"] = "count"

    # A kernel only reaches the operator through calc_eigs_chebyshev and
    # calc_eigs_exact_sym. Everything else rebuilds L via create_adj, which
    # ignores edge weights. Either the arm is genuinely treated, or the config is
    # REFUSED -- what must never happen is a run that silently produces a `none`
    # basis while stamping `cum-count` on its identity, because that reads exactly
    # like a real negative result.
    eff = "exact" if data_type == "f+pe" else solver
    treated = (
        (eff == "chebyshev" and update_mode in ("update", "recompute"))
        or (eff == "exact" and update_mode == "recompute")
    )

    plain = served_bases(config, data_type, update_mode, solver, "none", 0.9)
    other = served_bases(config, data_type, update_mode, solver, "count", 0.9)
    deltas = [float((a - b).abs().max()) for a, b in zip(plain[1:], other[1:])]

    if treated:
        assert_cfg(config)
        identity = DynamicServer(snapshots(DENSE_SCHEDULE, DENSE_NODES))._run_id()
        assert "cum-count" in identity.split("_")  # it bills itself as treated
        # DENSE_SCHEDULE repeats edges from t=1, so `count` is genuinely
        # non-uniform and the basis it induces must differ from the binary one
        assert max(deltas) > 0.0, f"stamped as treated but inert: {deltas}"
    else:
        with pytest.raises(ValueError, match="has NO EFFECT"):
            assert_cfg(config)
        assert max(deltas) == 0.0, f"expected an inert path, got deltas {deltas}"


# ---- kernel semantics that the separation argument rests on ---- #


@pytest.mark.parametrize("kernel,param", NON_DEFAULT_KERNELS)
def test_both_directed_columns_of_a_pair_carry_the_same_weight(config, kernel, param):
    # the weight vector is aligned to _cum_edges' COLUMNS, and _cum_edges holds
    # both (u,v) and (v,u); unequal columns would make A asymmetric and L_sym
    # would stop being a Laplacian at all
    server = accumulated_server(config, kernel, param)
    w = server._cum_edge_weight(3)
    ce = server._cum_edges
    key = torch.minimum(ce[0], ce[1]) * N_NODES + torch.maximum(ce[0], ce[1])

    for k in key.unique().tolist():
        assert w[key == k].unique().numel() == 1


def test_the_kernel_weights_recency_not_antiquity(config):
    # (0,1) last appeared at t=3, (2,3) at t=1: under any decaying kernel the
    # recent pair must outweigh the stale one
    for kernel, param in (("harmonic", 0.9), ("exp", 0.5), ("window", 1)):
        server = accumulated_server(config, kernel, param)
        assert (weight_of(server, 3, 0, 1) > weight_of(server, 3, 2, 3)).all()


def test_count_counts_snapshots_not_directed_occurrences(config):
    # w_t(e) = sum over SNAPSHOTS s<=t containing e; a pair listed twice inside
    # one snapshot is still one appearance
    schedule = [[(0, 1), (1, 0), (1, 2)], [(0, 1)]]
    server = accumulated_server(config, "count", 0.9, schedule, N_NODES)

    assert (weight_of(server, 1, 0, 1) == 2.0).all()
    assert (weight_of(server, 1, 1, 2) == 1.0).all()


# ---- run identity ---- #


def pin_identity(config, tmp_path):
    """Pin every identity axis that predates this change; leave the new knobs at
    their package defaults, which is exactly what the byte-identity test needs."""
    new_knobs = (
        ("spectral", "use_procrustes"),
        ("spectral", "cum_decay"),
        ("spectral", "cum_decay_param"),
        ("structure_model", "sfv_reset_per_snapshot"),
        ("structure_model", "freeze_sfv"),
        ("gnn", "encoder_edge_drop"),
    )
    defaults = {k: config[k[0]][k[1]] for k in new_knobs}
    set_identity_config(config, tmp_path)
    for (section, key), value in defaults.items():
        config[section][key] = value
    config["spectral"]["update_mode"] = "update"
    config["spectral"]["spectral_len"] = 300
    config["spectral"]["L_type"] = "sym"
    config["model"]["smodel_type"] = "LanczosLaplace"


def identity_server(config, tmp_path):
    from src.dynamic_server import DynamicServer

    pin_identity(config, tmp_path)
    return DynamicServer(make_toy_snapshots())


# literals read off the pre-change code (use_procrustes: true everywhere, no
# cum_decay / sfv / edrop tokens). Banked checkpoints resolve to these paths.
PRE_CHANGE_IDS = {
    "feature": "uci_gru_feature_C1_s1234",
    "f+s": ("uci_gru_f+s_C1_um-update_sfv-local_basis-laplacian_k300"
            "_sm-LanczosLaplace_proc-on_s1234"),
    "structure": ("uci_gru_structure_C1_um-update_sfv-local_basis-laplacian_k300"
                  "_sm-LanczosLaplace_proc-on_s1234"),
    "f+pe": "uci_gru_f+pe_C1_um-update_pe50_basis-laplacian_proc-on_s1234",
    # DELIBERATELY CHANGED 2026-08-20: use_procrustes moved true -> 'auto', which
    # resolves OFF on f+es (its features are rotation-invariant, and the rotation
    # broke the phi block). This is the one data type whose identity moved; the
    # other four are byte-identical. Numerically the change is null -- results.md
    # §19 measures every delta inside the +-0.014 floor. Was: ..._proc-on_solver-...
    "f+es": ("uci_gru_f+es_C1_um-update_pe50_basis-laplacian_esf-spec_esp-phi+cos+lev"
             "_proc-off_solver-chebyshev_s1234"),
}


@pytest.mark.parametrize("data_type", sorted(PRE_CHANGE_IDS))
def test_the_default_identity_is_byte_identical_to_the_pre_change_string(
    config, tmp_path, data_type
):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = data_type
    if data_type == "f+es":
        config["spectral"]["solver"] = "chebyshev"  # f+es forbids arnoldi

    assert explicit_id(server) == PRE_CHANGE_IDS[data_type]


def test_no_new_knob_adds_a_token_at_its_default(config, tmp_path):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = "f+s"
    baseline = server._run_id()

    config["spectral"]["cum_decay"] = "none"
    config["spectral"]["cum_decay_param"] = 0.9
    config["structure_model"]["sfv_reset_per_snapshot"] = False
    config["structure_model"]["freeze_sfv"] = False
    config["gnn"]["encoder_edge_drop"] = 0.0

    assert explicit_id(server) == explicit_id_of(baseline) == PRE_CHANGE_IDS["f+s"]


def test_a_config_predating_the_kernel_keeps_its_identity(config, tmp_path):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = "f+s"
    del config["spectral"]["cum_decay"]

    assert explicit_id(server) == PRE_CHANGE_IDS["f+s"]


@pytest.mark.parametrize("data_type", ["f+s", "f+pe", "structure"])
def test_the_identity_separates_every_kernel(config, tmp_path, data_type):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = data_type

    ids = {}
    for kernel, param in KERNELS:
        config["spectral"]["cum_decay"] = kernel
        config["spectral"]["cum_decay_param"] = param
        ids[kernel] = explicit_id(server)

    assert len(set(ids.values())) == len(KERNELS)
    assert ids["none"] == PRE_CHANGE_IDS[data_type]
    for kernel in ("count", "harmonic", "exp", "window"):
        assert any(p.startswith("cum-") for p in ids[kernel].split("_"))


@pytest.mark.parametrize("kernel,params", [("exp", (0.1, 0.5, 0.9)), ("window", (1, 3, 10))])
def test_the_identity_separates_parameters_within_one_kernel(
    config, tmp_path, kernel, params
):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = "f+s"
    config["spectral"]["cum_decay"] = kernel

    ids = set()
    for param in params:
        config["spectral"]["cum_decay_param"] = param
        ids.add(server._run_id())

    assert len(ids) == len(params)


@pytest.mark.parametrize("kernel", ["none", "count", "harmonic"])
def test_the_parameter_is_inert_for_the_kernels_that_ignore_it(config, tmp_path, kernel):
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = "f+s"
    # a path the kernel actually reaches: on the create_adj-based paths assert_cfg
    # refuses a non-default kernel outright
    config["spectral"]["solver"] = "chebyshev"
    config["spectral"]["update_mode"] = "update"
    config["spectral"]["cum_decay"] = kernel

    ids, hashes, weights = set(), set(), []
    for param in (0.1, 0.9, 1):
        config["spectral"]["cum_decay_param"] = param
        assert_cfg(config)  # accepted: these kernels never read it
        ids.add(explicit_id(server))
        hashes.add(server._run_id())
        weights.append(accumulated_server(config, kernel, param)._cum_edge_weight(3))
        config["spectral"]["cum_decay"] = kernel  # accumulated_server rewrote it

    # The readable arm is unchanged: no kernel here reads the parameter, so no
    # token records it. The completeness fingerprint DOES separate the three,
    # because it hashes the config bytes and cannot know the value is inert.
    # That is over-separation, and it is the accepted direction: the cost is a
    # re-run, where under-separation costs a silent wrong-arm resume.
    assert len(ids) == 1
    assert len(hashes) == 3
    for w in weights[1:]:
        assert (w is None and weights[0] is None) or torch.equal(w, weights[0])


def test_the_kernel_stays_out_of_a_non_spectral_identity(config, tmp_path):
    # data_type=feature never builds a basis, so an age kernel cannot change its
    # numbers; giving it its own id would re-run banked arms for nothing
    server = identity_server(config, tmp_path)
    config["model"]["data_type"] = "feature"
    config["spectral"]["cum_decay"] = "exp"
    config["spectral"]["cum_decay_param"] = 0.5

    assert explicit_id(server) == PRE_CHANGE_IDS["feature"]


def test_the_identity_records_the_effective_procrustes_value(config, tmp_path):
    # 'auto' resolves per data type; two runs that differ in whether the basis was
    # rotated must not share a checkpoint, so the id has to carry the RESOLVED bit
    server = identity_server(config, tmp_path)
    config["spectral"]["use_procrustes"] = "auto"

    config["model"]["data_type"] = "f+s"
    assert "proc-on" in server._run_id().split("_")

    config["model"]["data_type"] = "f+es"
    config["spectral"]["solver"] = "chebyshev"
    auto = server._run_id()
    assert "proc-off" in auto.split("_")

    # an explicit bool still wins on every path, so the on/off A/B stays runnable
    config["spectral"]["use_procrustes"] = True
    assert server._run_id() != auto
    assert "proc-on" in server._run_id().split("_")
    # An explicit False and 'auto' resolve to the SAME effective value on f+es, so
    # the readable arm is identical -- but they are different config bytes, so the
    # fingerprint separates them. Under the backstop "the same arm" means the same
    # config, not the same effective behaviour.
    config["spectral"]["use_procrustes"] = False
    assert explicit_id(server) == explicit_id_of(auto)
    assert server._run_id() != auto


# ---- config validation ---- #


@pytest.mark.parametrize("bad", ["linear", "expo", "EXP", "", None, 0.5])
def test_assert_cfg_rejects_an_unknown_kernel(config, bad):
    config["spectral"]["cum_decay"] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.cum_decay" in str(exc.value)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2, "0.5", None])
def test_assert_cfg_rejects_an_exp_factor_outside_the_unit_interval(config, bad):
    config["spectral"]["cum_decay"] = "exp"
    config["spectral"]["cum_decay_param"] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.cum_decay_param" in str(exc.value)


@pytest.mark.parametrize("bad", [0, -1, 2.5, 1.0, "3", True, None])
def test_assert_cfg_rejects_a_non_positive_or_non_integer_window(config, bad):
    config["spectral"]["cum_decay"] = "window"
    config["spectral"]["cum_decay_param"] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.cum_decay_param" in str(exc.value)


def test_assert_cfg_accepts_every_supported_kernel_and_absence(config):
    config["spectral"]["solver"] = "chebyshev"
    config["spectral"]["update_mode"] = "update"
    for kernel, param in KERNELS:
        config["spectral"]["cum_decay"] = kernel
        config["spectral"]["cum_decay_param"] = param
        assert_cfg(config)

    del config["spectral"]["cum_decay"]  # a config predating the key
    assert_cfg(config)


@pytest.mark.parametrize("bad", ["AUTO", "on", None, 1, 0])
def test_assert_cfg_rejects_a_procrustes_value_that_is_neither_bool_nor_auto(
    config, bad
):
    config["spectral"]["use_procrustes"] = bad

    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.use_procrustes" in str(exc.value)


# ---- resume ---- #


def resumable_server(config, tmp_path, snaps):
    server = make_server(snaps, partition_snapshots(snaps, 1))
    server.initialize_FL()
    return server


@pytest.mark.parametrize("kernel,param", NON_DEFAULT_KERNELS)
def test_resume_reproduces_the_weights_of_an_uninterrupted_run(
    config, tmp_path, kernel, param
):
    setup_tiny_config(config, tmp_path)
    config["train"]["ckpt_clean"] = False
    config["spectral"]["cum_decay"] = kernel
    config["spectral"]["cum_decay_param"] = param

    seed_all(42)
    snaps = make_toy_snapshots(N=8, num_snaps=4, seed=42)
    crashed = resumable_server(config, tmp_path, snaps)
    for t in range(3):
        crashed._accumulate_cum_edges(t)
    crashed._save_partial_ckpt(2, None, [], [])

    resumed = resumable_server(config, tmp_path, snaps)
    assert resumed._load_partial_ckpt() is not None

    assert torch.equal(resumed._cum_edges, crashed._cum_edges)
    assert torch.equal(resumed._cum_edge_weight(2), crashed._cum_edge_weight(2))
    # and the two stay in step as the run continues
    crashed._accumulate_cum_edges(3)
    resumed._accumulate_cum_edges(3)
    assert torch.equal(resumed._cum_edge_weight(3), crashed._cum_edge_weight(3))


def test_the_checkpoint_carries_the_snapshot_index_the_rebuild_needs(config, tmp_path):
    # _load_partial_ckpt replays snapshots 0..ckpt['t'] to rebuild the appearance
    # record. Without 't' it would replay nothing and every weight would collapse
    # to the unweighted basis on resume, silently.
    setup_tiny_config(config, tmp_path)
    config["train"]["ckpt_clean"] = False
    config["spectral"]["cum_decay"] = "harmonic"

    seed_all(42)
    snaps = make_toy_snapshots(N=8, num_snaps=4, seed=42)
    server = resumable_server(config, tmp_path, snaps)
    for t in range(2):
        server._accumulate_cum_edges(t)
    server._save_partial_ckpt(1, None, [], [])

    ckpt_path, _ = server._ckpt_paths()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert "t" in ckpt and ckpt["t"] == 1


def test_rebuilding_the_event_record_is_a_pure_function_of_the_snapshots(config):
    server = accumulated_server(config, "harmonic", 0.9)
    incremental = server._cum_edge_weight(3)

    server._rebuild_cum_events(3)

    assert torch.equal(server._cum_edge_weight(3), incremental)


def test_an_empty_event_record_falls_back_to_the_unweighted_basis(config):
    # Documenting the failure mode, not endorsing it: an empty record makes
    # _cum_edge_weight return None, i.e. the run silently reverts to the binary
    # union while still calling itself a treated arm. Unreachable today (the
    # record and _cum_edges have one writer, and resume replays 0..ckpt['t']), but
    # it is a SILENT degradation rather than a loud one -- a guard belongs here.
    server = accumulated_server(config, "harmonic", 0.9)
    assert server._cum_edge_weight(3) is not None

    server._rebuild_cum_events(-1)

    assert server._cum_edge_weight(3) is None
