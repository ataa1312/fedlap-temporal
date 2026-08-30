"""spectral.inject_at: putting the structural embedding where the recurrence sees it.

The whole point of the arm is that S enters the state, exactly once. Two ways it
can be wrong without looking wrong:

1. S enters TWICE -- once at the last MP layer and again at the output -- in
   which case neither arm measures what it claims and the comparison is void.
2. S enters NOWHERE, while _run_id still stamps `inj-last_mp`. A silent no-op
   reads exactly like a real negative result. This codebase has been bitten by
   that class four times (cn degradation, coverage_drop on persist/cn, the cached
   basis mask, the -1 axis), so the no-op paths are pinned here explicitly: the
   knob is refused where no S exists, the encoder refuses an injection it has no
   recurrent layer to give, and any width that PyTorch would broadcast is refused
   rather than silently absorbed.

The one no-op these cannot forbid is structural and deliberate: output_bn
zero-inits the smodel's BatchNorm gamma, so S is EXACTLY zero at init and the two
arms are numerically identical until training moves gamma. That is pinned below,
together with the gradient that lets the arm escape it -- an arm that starts
inert is not the same thing as an arm that stays inert.
"""

import pytest
import torch
import torch.nn.functional as F

import main
from config.assertions import assert_cfg
from src.GNN.dynamic_classifier import DynamicClassifier
from src.models.model_binders import ModelBinder, ModelSpecs
from src.train.federated_orchestrator import _mp_graph, _partition_edges_per_snapshot
from src.utils.graph_partitioning import partition_snapshots
from test_checkpoint_wandb import make_server, make_toy_snapshots, seed_all, setup_tiny_config
from test_run_identity import explicit_id, set_identity_config


REC_KWARGS = {
    "layer_type": "gcnconv",
    "updater_name": "gru",
    "batchnorm": False,
    "dropout": 0.0,
    "act": "relu",
    "skip_connection": "none",
    "layer_kwargs": {},
    "updater_kwargs": {},
}

N_NODES = 7
WIDTH = 6
EDGE_INDEX = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)


def rec_spec(dim_in=WIDTH, dim_out=WIDTH, updater="gru", **over):
    kw = dict(REC_KWARGS)
    kw["updater_name"] = updater
    kw.update(over)
    return ModelSpecs(type="recurrent_layer", layer_sizes=[dim_in, dim_out], block_kwargs=kw)


def build_binder(n_rec=3, lead=(), updater="gru", seed=0):
    """A stack in the documented spec order: [edge_encoder?, node_encoder?, pre_mp?, rec...].

    `lead` selects which non-recurrent blocks precede the recurrent layers, so a
    test can vary the offset between a block's position in models_specs and its
    index among the recurrent layers -- the two counters the hook must not
    conflate.
    """
    specs, d = [], WIDTH
    if "edge_encoder" in lead:
        specs.append(ModelSpecs(type="edge_encoder", layer_sizes=[1, 1],
                                block_kwargs={"batchnorm": False}))
    if "node_encoder" in lead:
        specs.append(ModelSpecs(type="node_encoder", layer_sizes=[d, d],
                                block_kwargs={"batchnorm": False}))
    if "pre_mp" in lead:
        specs.append(ModelSpecs(type="roland_mlp", layer_sizes=[d, d],
                                block_kwargs={"batchnorm": False, "dropout": 0.0,
                                              "act": "relu", "final_act": True}))
    specs += [rec_spec(d, d, updater) for _ in range(n_rec)]
    torch.manual_seed(seed)
    return ModelBinder(specs).eval()


def inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(N_NODES, WIDTH, generator=g),
            torch.randn(N_NODES, WIDTH, generator=g))


def prior_state(n_rec, seed=2):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N_NODES, WIDTH, generator=g) for _ in range(n_rec)]


def injected_layers(binder):
    """Which recurrent layers were handed a non-None injection, by position."""
    seen = []
    rec = [m for sp, m in zip(binder.models_specs, binder.models)
           if sp.type == "recurrent_layer"]
    originals = [m.forward for m in rec]

    def wrap(idx, fn):
        def inner(*a, **kw):
            if kw.get("inject") is not None:
                seen.append(idx)
            return fn(*a, **kw)
        return inner

    for i, (m, fn) in enumerate(zip(rec, originals)):
        m.forward = wrap(i, fn)
    return seen


# --------------------------------------------------------------------- #
# 1. The encoder hook
#    Requirement: Optional Injection Into The Last Recurrent Layer
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("n_rec", [1, 2, 3])
@pytest.mark.parametrize("lead", [(), ("node_encoder",), ("pre_mp",),
                                  ("edge_encoder", "node_encoder", "pre_mp")])
def test_the_absent_injection_is_bit_identical(config, n_rec, lead):
    # The rollback path. Every stack shape, both call forms.
    b = build_binder(n_rec, lead)
    x, _ = inputs()
    hs = prior_state(n_rec)
    ea = torch.ones(EDGE_INDEX.size(1), 1)

    z0, hs0 = b.encode(x, EDGE_INDEX, hs=hs, edge_attr=ea)
    z1, hs1 = b.encode(x, EDGE_INDEX, hs=hs, edge_attr=ea, inject=None)

    assert torch.equal(z0, z1)
    assert [h.tolist() for h in hs0] == [h.tolist() for h in hs1]


def test_the_injection_lands_on_the_candidate_before_the_state_update(config):
    # Reconstruct the layer's own arithmetic: h_cand = block(x), then
    # updater(h_prev, h_cand + inject). A GRU is not linear in its candidate, so
    # this equality dies if the injection is moved after the updater, applied to
    # the layer input, or scaled.
    b = build_binder(n_rec=1)
    x, inj = inputs()
    hs = prior_state(1)
    layer = b.models[0]

    z, new_hs = b.encode(x, EDGE_INDEX, hs=hs, inject=inj)
    expected = layer.updater(hs[0], layer.block(x, EDGE_INDEX) + inj)

    assert torch.allclose(z, expected, atol=0, rtol=0)
    assert torch.allclose(new_hs[0], expected, atol=0, rtol=0)


def test_an_injection_after_the_updater_would_not_match(config):
    # The negative half of the above: if the term were added to the layer's
    # OUTPUT the result would be z_plain + inject, which is what the output-side
    # fusion already does and what this change exists to stop doing.
    b = build_binder(n_rec=1)
    x, inj = inputs()
    hs = prior_state(1)

    z_plain, _ = b.encode(x, EDGE_INDEX, hs=hs)
    z_inj, _ = b.encode(x, EDGE_INDEX, hs=hs, inject=inj)

    assert not torch.allclose(z_inj, z_plain + inj, atol=1e-6)


def test_an_identity_updater_sees_the_sum(config):
    # The spec's own scenario, stated in its own terms: with the updater the
    # identity on its candidate, the layer output IS the message-passing output
    # plus the injection.
    b = build_binder(n_rec=1, updater="moving_average")
    x, inj = inputs()

    z_plain, _ = b.encode(x, EDGE_INDEX, hs=None)   # h_prev None -> MA returns h_cand
    z_inj, _ = b.encode(x, EDGE_INDEX, hs=None, inject=inj)

    assert torch.allclose(z_inj, z_plain + inj, atol=0, rtol=0)


@pytest.mark.parametrize("lead", [(), ("node_encoder",),
                                  ("edge_encoder", "node_encoder", "pre_mp")])
def test_only_the_last_recurrent_layer_is_injected(config, lead):
    # `r` counts recurrent layers; the loop index counts models_specs entries.
    # With 0-3 leading blocks the two counters disagree by 0-3, so a hook keyed
    # on the wrong one either injects the wrong layer or (if it compares against
    # len(models_specs)) injects none at all.
    b = build_binder(n_rec=3, lead=lead)
    x, inj = inputs()
    hs = prior_state(3)
    ea = torch.ones(EDGE_INDEX.size(1), 1)
    seen = injected_layers(b)

    _, hs_plain = b.encode(x, EDGE_INDEX, hs=hs, edge_attr=ea)
    _, hs_inj = b.encode(x, EDGE_INDEX, hs=hs, edge_attr=ea, inject=inj)

    assert seen == [2]                                   # exactly one, and the last
    assert torch.equal(hs_plain[0], hs_inj[0])
    assert torch.equal(hs_plain[1], hs_inj[1])
    assert not torch.equal(hs_plain[2], hs_inj[2])


def test_the_injection_reaches_the_carried_state(config):
    # The mechanism the change is FOR: the modified candidate goes through the
    # updater, and the updater's output is what new_hs carries to the next
    # snapshot. If it only reached z the arm would be the output fusion again.
    b = build_binder(n_rec=2)
    x, inj = inputs()
    hs = prior_state(2)

    z_plain, hs_plain = b.encode(x, EDGE_INDEX, hs=hs)
    z_inj, hs_inj = b.encode(x, EDGE_INDEX, hs=hs, inject=inj)

    assert not torch.equal(hs_plain[-1], hs_inj[-1])
    assert torch.equal(hs_inj[-1], z_inj)                # the state IS the output
    # and it persists: feeding the two states forward keeps them apart
    z_next_plain, _ = b.encode(x, EDGE_INDEX, hs=hs_plain)
    z_next_inj, _ = b.encode(x, EDGE_INDEX, hs=hs_inj)
    assert not torch.allclose(z_next_plain, z_next_inj, atol=1e-6)


@pytest.mark.parametrize(
    "shape", [(N_NODES, 1), (1, WIDTH), (), (WIDTH,), (N_NODES, 1, WIDTH)]
)
def test_a_width_that_would_broadcast_is_refused(config, shape):
    # A column, a row, a scalar, a bare row vector, a spurious axis: PyTorch
    # broadcasts all of them against (N, d) without complaint, so each would
    # inject a DIFFERENT quantity than the arm claims and nothing would say so.
    b = build_binder(n_rec=2)
    x, _ = inputs()
    bad = torch.randn(shape)

    assert (torch.zeros(N_NODES, WIDTH) + bad).numel() >= N_NODES * WIDTH  # it does broadcast

    with pytest.raises(ValueError, match="refusing to broadcast"):
        b.encode(x, EDGE_INDEX, hs=None, inject=bad)


def test_an_outright_incompatible_width_is_refused(config):
    b = build_binder(n_rec=2)
    x, _ = inputs()

    with pytest.raises(ValueError, match="refusing to broadcast"):
        b.encode(x, EDGE_INDEX, hs=None, inject=torch.randn(N_NODES, WIDTH + 1))


def test_an_injection_with_no_recurrent_layer_is_refused(config):
    # The silent no-op the hook can produce on its own: with n_rec == 0 the
    # routing condition `r == n_rec - 1` is `r == -1`, which no layer satisfies,
    # so the injection would be dropped and the caller would never know.
    b = ModelBinder([ModelSpecs(type="node_encoder", layer_sizes=[WIDTH, WIDTH],
                                block_kwargs={"batchnorm": False})]).eval()
    x, inj = inputs()

    assert b.encode(x, EDGE_INDEX, hs=None)[1] == []      # inert without an injection
    with pytest.raises(ValueError, match="no recurrent_layer"):
        b.encode(x, EDGE_INDEX, hs=None, inject=inj)


# --------------------------------------------------------------------- #
# 2. The knob and its refusals
#    Requirement: Configurable Structural Injection Point
# --------------------------------------------------------------------- #


def set_fs_config(cfg):
    cfg["dataset"]["task"] = "link_pred"
    cfg["model"]["data_type"] = "f+s"
    cfg["model"]["fusion"] = "add"
    cfg["spectral"]["update_mode"] = "keep"


def test_the_default_is_output(config):
    assert config["spectral"]["inject_at"] == "output"


def test_the_membership_is_closed(config):
    set_fs_config(config)
    for bad in ("last", "lastmp", "LAST_MP", "input", "", None, True):
        config["spectral"]["inject_at"] = bad
        with pytest.raises(ValueError, match="spectral.inject_at"):
            assert_cfg(config)


def test_concat_is_refused_at_last_mp(config):
    # Reinterpreting concat as add would record a fusion mode the run did not
    # use; widening the encoder mid-stack is not on offer.
    set_fs_config(config)
    config["spectral"]["inject_at"] = "last_mp"
    config["model"]["fusion"] = "concat"

    with pytest.raises(ValueError, match="additive only"):
        assert_cfg(config)

    config["model"]["fusion"] = "add"
    assert_cfg(config)


@pytest.mark.parametrize("data_type", ["feature", "f+pe", "f+es"])
def test_last_mp_is_refused_where_no_embedding_exists(config, data_type):
    set_fs_config(config)
    config["model"]["data_type"] = data_type
    config["spectral"]["pe_dim"] = 50
    config["spectral"]["solver"] = "exact"
    config["spectral"]["inject_at"] = "last_mp"

    with pytest.raises(ValueError, match=r"NO EFFECT with data_type"):
        assert_cfg(config)


def test_no_permitted_data_type_can_stamp_an_arm_without_an_embedding(config):
    """The refusal must not be permissive: every data type that assert_cfg lets
    through at last_mp has to reach a classifier that actually owns an smodel.

    `structure` is the one to watch -- it is in the allowlist, but the dynamic
    path has no structure-only classifier at all, so it cannot produce an S. It
    is caught downstream instead of by the assertion, which is why this test
    checks the END of the path rather than the assertion's wording.
    """
    from config.assertions import _DATA_TYPES

    for dt in sorted(_DATA_TYPES):
        set_fs_config(config)
        config["model"]["data_type"] = dt
        config["spectral"]["pe_dim"] = 50
        config["spectral"]["solver"] = "exact"
        config["spectral"]["inject_at"] = "last_mp"
        try:
            assert_cfg(config)
        except ValueError:
            continue                                   # refused: nothing to check
        # permitted -- so the run must be able to build something with an smodel
        snaps = make_toy_snapshots(N=8, W=1, num_snaps=2, seed=1)
        server = make_server(snaps, partition_snapshots(snaps, 1))
        try:
            server.initialize_FL()
        except NotImplementedError:
            continue                                   # refused downstream, loudly
        assert getattr(server.classifier, "smodel", None) is not None, dt
        server._spectral_step(0, config["model"]["smodel_type"])
        S = server.classifier.smodel.get_embeddings()
        assert S is not None and S.shape[-1] == config["gnn"]["dims"][-1], dt


# --------------------------------------------------------------------- #
# 3. Identity: checkpoints and the wandb record
# --------------------------------------------------------------------- #


def test_the_injection_point_separates_the_run_identity(config, tmp_path):
    from src.dynamic_server import DynamicServer

    set_identity_config(config, tmp_path)
    server = DynamicServer(make_toy_snapshots())
    config["model"]["data_type"] = "f+s"

    config["spectral"]["inject_at"] = "output"
    default = explicit_id(server)
    config["spectral"]["inject_at"] = "last_mp"
    injected = explicit_id(server)

    # NB the token itself contains an underscore, so it is not a whole `_`-part
    assert "_inj-last_mp_" in f"_{injected}_"
    assert injected != default
    assert "inj-" not in default                  # default adds no token at all


def test_a_config_predating_the_knob_keeps_its_identity(config, tmp_path):
    from src.dynamic_server import DynamicServer

    set_identity_config(config, tmp_path)
    server = DynamicServer(make_toy_snapshots())
    config["model"]["data_type"] = "f+s"

    at_default = explicit_id(server)
    del config["spectral"]["inject_at"]

    assert explicit_id(server) == at_default


def test_the_injection_point_separates_the_wandb_group(config):
    # The wandb run id is sha1(group + seed) with resume='allow', so two arms
    # sharing a group share a RUN and overwrite each other's history.
    config["dataset"]["name"] = "uci"
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "keep"

    config["spectral"]["inject_at"] = "output"
    default_group, cfg_default, default_tags = main._wandb_meta()
    config["spectral"]["inject_at"] = "last_mp"
    injected_group, cfg_inj, injected_tags = main._wandb_meta()

    assert injected_group != default_group
    assert "_inj-last_mp" in f"_{injected_group}"
    assert "inj-last_mp" in injected_tags
    assert "inj-last_mp" not in default_tags
    assert "inj-" not in default_group
    assert (cfg_default["inject_at"], cfg_inj["inject_at"]) == ("output", "last_mp")


def test_the_placebo_separates_the_wandb_group_on_f_plus_s(config):
    # The control this change's experiment is read against. f+pe has carried
    # basis_source in its group since 680688a; f+s did not, so the real arm and
    # its shuffled_fixed control collapsed into one group and one run id.
    config["dataset"]["name"] = "uci"
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "keep"
    config["subgraph"]["num_subgraphs"] = 1

    config["spectral"]["basis_source"] = "laplacian"
    real = main._wandb_meta()[0]
    config["spectral"]["basis_source"] = "shuffled_fixed"
    placebo = main._wandb_meta()[0]

    assert real != placebo
    assert real == "uci_gru_f+s_C1_um-keep_sfv-local"    # banked groups unchanged


def test_all_four_pre_registered_arms_get_their_own_wandb_run(config):
    config["dataset"]["name"] = "uci"
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "keep"
    config["seed"] = 1234

    ids = set()
    for inject_at in ("output", "last_mp"):
        for basis in ("laplacian", "shuffled_fixed"):
            config["spectral"]["inject_at"] = inject_at
            config["spectral"]["basis_source"] = basis
            group = main._wandb_meta()[0]
            ids.add(main._wandb_id(f"{group}_s{config['seed']}"))

    assert len(ids) == 4


# --------------------------------------------------------------------- #
# 4. The classifier wiring: injected once, and only once
# --------------------------------------------------------------------- #


def fs_config(cfg):
    cfg["dataset"]["name"] = "uci"
    cfg["dataset"]["task"] = "link_pred"
    cfg["dataset"]["edge_dim"] = 1
    cfg["dataset"]["node_encoder"] = False
    cfg["dataset"]["edge_encoder"] = False
    cfg["model"]["edge_decoding"] = "concat"
    cfg["model"]["data_type"] = "f+s"
    cfg["model"]["fusion"] = "add"
    cfg["model"]["smodel_type"] = "LanczosLaplace"
    cfg["gnn"]["dims"] = [8, 8]
    cfg["gnn"]["dims_pre_mp"] = []
    cfg["gnn"]["dims_post_mp"] = []
    cfg["gnn"]["embed_update_method"] = "gru"
    cfg["gnn"]["l2norm"] = True
    cfg["structure_model"]["num_structural_features"] = 16
    cfg["structure_model"]["DGCN_structure_layers_sizes"] = [8]
    cfg["spectral"]["spectral_len"] = 4
    cfg["spectral"]["update_mode"] = "keep"
    cfg["subgraph"]["num_subgraphs"] = 1
    cfg["seed"] = 42


def fs_client(config, inject_at, gamma=None):
    """A client f+s classifier with its basis served, at the given injection point."""
    fs_config(config)
    config["spectral"]["inject_at"] = inject_at
    seed_all(42)
    snaps = make_toy_snapshots(N=8, W=1, num_snaps=3, seed=42)
    server = make_server(snaps, partition_snapshots(snaps, 1))
    server.initialize_FL()
    server._spectral_step(0, config["model"]["smodel_type"])
    cl = server.clients[0].classifier
    cl.eval()
    if gamma is not None:
        torch.nn.init.constant_(cl.smodel.bn.weight, gamma)
    return cl, server.clients[0].snaps[0]


def encoder_only(cl, snap, inject):
    """What the encoder alone produces -- no output-side fusion of any kind."""
    mp_ei, mp_ea = _mp_graph(snap)
    return cl.model.encode(cl.node_input(snap), mp_ei, None, edge_attr=mp_ea,
                           keep_ratio=getattr(snap, "keep_ratio", None),
                           active_mask=None, inject=inject)


@torch.no_grad()
def test_last_mp_does_not_also_fuse_at_the_output(config):
    """The single property that makes either arm interpretable.

    If both fired, S would enter twice through two different downstream paths and
    neither `output` nor `last_mp` would measure what it claims.
    """
    cl, snap = fs_client(config, "last_mp", gamma=1.0)
    S = cl.smodel.get_embeddings()
    assert S.abs().max() > 0                       # a zero S would prove nothing

    z, _ = cl.encode(snap, None)
    z_enc, _ = encoder_only(cl, snap, inject=S)
    expected = F.normalize(z_enc, p=2, dim=-1)

    assert torch.allclose(z, expected, atol=1e-6)
    assert not torch.allclose(z, expected + S, atol=1e-4)


@torch.no_grad()
def test_the_output_path_still_fuses_and_never_touches_the_state(config):
    cl, snap = fs_client(config, "output", gamma=1.0)
    S = cl.smodel.get_embeddings()

    z, new_hs = cl.encode(snap, None)
    z_enc, hs_enc = encoder_only(cl, snap, inject=None)

    assert torch.allclose(z, F.normalize(z_enc, p=2, dim=-1) + S, atol=1e-6)
    # the carried state is the plain encoder's: S is bolted on afterwards
    for a, b in zip(new_hs, hs_enc):
        assert torch.equal(a, b)


@torch.no_grad()
def test_the_two_arms_carry_different_state(config):
    cl_out, snap = fs_client(config, "output", gamma=1.0)
    cl_inj, _ = fs_client(config, "last_mp", gamma=1.0)

    # both classifiers read the GLOBAL config at encode time, so the arm has to
    # be re-selected around each call -- otherwise the second build's value wins
    # for both and the comparison is between two copies of one arm
    config["spectral"]["inject_at"] = "output"
    _, hs_out = cl_out.encode(snap, None)
    config["spectral"]["inject_at"] = "last_mp"
    _, hs_inj = cl_inj.encode(snap, None)

    assert torch.equal(hs_out[0], hs_inj[0])            # earlier layers agree
    assert not torch.allclose(hs_out[-1], hs_inj[-1], atol=1e-5)


def test_the_arms_are_parameter_matched(config):
    # design D1: S is already sized to gnn.dims[-1], so last_mp adds no
    # projection and no parameters. Without this, a difference could be capacity.
    out, _ = fs_client(config, "output")
    inj, _ = fs_client(config, "last_mp")

    n_out = [p.numel() for p in out.parameters()]
    n_inj = [p.numel() for p in inj.parameters()]

    assert sum(n_out) == sum(n_inj)
    assert n_out == n_inj                               # same tensors, not just the total


@torch.no_grad()
def test_a_caller_injection_reaches_the_last_layer_at_output(config):
    # DynamicClassifier.encode's new `inject` argument is threaded, not ignored.
    cl, snap = fs_client(config, "output")
    inj = torch.randn(snap.num_nodes, config["gnn"]["dims"][-1])

    z_plain, hs_plain = DynamicClassifier.encode(cl, snap, None)
    z_inj, hs_inj = DynamicClassifier.encode(cl, snap, None, inject=inj)

    assert torch.equal(hs_plain[0], hs_inj[0])
    assert not torch.allclose(hs_plain[-1], hs_inj[-1], atol=1e-6)
    assert not torch.allclose(z_plain, z_inj, atol=1e-6)


@torch.no_grad()
def test_last_mp_overrides_a_caller_injection(config):
    """Documented hazard, pinned so it cannot change silently.

    FedDynamicClassifier.encode takes an `inject` argument, forwards it at
    `output`, and DISCARDS it at `last_mp` in favour of the smodel's S. No
    in-repo caller passes one, so the argument is dead either way -- but if one
    ever does, the override must not be a surprise.
    """
    cl, snap = fs_client(config, "last_mp", gamma=1.0)
    S = cl.smodel.get_embeddings()
    decoy = torch.full_like(S, 1e3)

    z_none, _ = cl.encode(snap, None)
    z_decoy, _ = cl.encode(snap, None, inject=decoy)

    assert torch.equal(z_none, z_decoy)


# --------------------------------------------------------------------- #
# 5. The no-op ruling: S starts at exactly zero, and must be able to leave it
# --------------------------------------------------------------------- #


@torch.no_grad()
def test_the_embedding_is_exactly_zero_at_init(config):
    """spectral.output_bn zero-inits the smodel's BatchNorm gamma, so S == 0 at
    step 0 and BOTH arms are numerically identical there.

    This is deliberate (f+s starts exactly at the feature baseline), but it means
    "the arms agree" is NOT evidence the wiring works, and any smoke check that
    compares one untrained forward pass proves nothing.
    """
    cl, snap = fs_client(config, "last_mp")
    assert config["spectral"]["output_bn"] is True
    assert float(cl.smodel.get_embeddings().abs().max()) == 0.0

    cl_out, _ = fs_client(config, "output")
    assert torch.allclose(cl.encode(snap, None)[0], cl_out.encode(snap, None)[0], atol=1e-6)


def test_the_zero_start_is_escapable_under_last_mp(config):
    # The half that makes the zero start harmless: gradient still reaches the
    # smodel through the GRU, so the arm can train away from S == 0. If it could
    # not, `inj-last_mp` would be stamped on a permanently inert run.
    cl, snap = fs_client(config, "last_mp")
    cl.train()
    cl.zero_grad(set_to_none=True)

    z, _ = cl.encode(snap, None)
    z.pow(2).sum().backward()

    assert cl.smodel.bn.weight.grad is not None
    assert float(cl.smodel.bn.weight.grad.abs().max()) > 0


def test_the_injection_is_live_at_the_end_of_a_real_run(config, tmp_path):
    """End-to-end: after a short federated run the trained model's carried state
    genuinely depends on S, so the arm is not plumbing that measures nothing."""
    setup_tiny_config(config, tmp_path)
    config["train"]["auto_resume"] = False
    config["train"]["early_stopping"] = False
    config["model"]["data_type"] = "f+s"
    config["model"]["fusion"] = "add"
    config["model"]["local_epochs"] = 6
    config["gnn"]["dims"] = [8, 8]
    config["structure_model"]["num_structural_features"] = 16
    config["structure_model"]["DGCN_structure_layers_sizes"] = [8]
    config["spectral"]["spectral_len"] = 4
    config["spectral"]["update_mode"] = "keep"
    config["spectral"]["inject_at"] = "last_mp"

    seed_all(42)
    snaps = make_toy_snapshots(N=12, W=1, num_snaps=4, seed=42)
    server = make_server(snaps, partition_snapshots(snaps, 1))
    server.initialize_FL()
    _partition_edges_per_snapshot(server.global_snaps, [0.8, 0.1, 0.1], 42)
    for c, cl in enumerate(server.clients):
        _partition_edges_per_snapshot(cl.snaps, [0.8, 0.1, 0.1], 42 + 1000 * (c + 1))
    server.joint_train_w(FL=True)

    client = server.clients[0]
    cl = client.classifier
    cl.eval()
    assert float(cl.smodel.bn.weight.detach().abs().max()) > 0   # gamma left zero
    with torch.no_grad():
        _, hs_live = cl.encode(client.snaps[0], None)
        torch.nn.init.zeros_(cl.smodel.bn.weight)
        torch.nn.init.zeros_(cl.smodel.bn.bias)
        _, hs_dead = cl.encode(client.snaps[0], None)

    assert torch.equal(hs_live[0], hs_dead[0])
    assert not torch.allclose(hs_live[-1], hs_dead[-1], atol=1e-6)
