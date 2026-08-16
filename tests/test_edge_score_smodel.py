"""Invariant edge-score smodel (data_type=f+es, commit b93c540).

DynamicSInvariant reads the eigenbasis only through projector-style invariants
    phi_j(u, v) = sum_i f_j(lambda_i) U_ui U_vi
so the score cannot depend on the gauge the solver happened to return. That is
the load-bearing property: the low spectrum of these graphs is clustered, the
individual eigenvectors rotate 50-80 degrees between snapshots, and only the
SUBSPACE is identifiable. These pin the invariance, the zero-init contract, the
decode-time fusion, the federated protocol, and the f+es dispatch.
"""

import copy

import pytest
import torch

import src
from config.assertions import assert_cfg
from src.GNN.dynamic_classifier import DynamicClassifier
from src.GNN.fed_dynamic_classifier import (
    DynamicSInvariant,
    FedDynamicEdgeScoreClassifier,
)
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots
from src.dynamic_server import DynamicServer
from torch_geometric.data import Data

N_NODES = 40
N_PAIRS = 25
K = 6


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


def make_basis(n=N_NODES, k=K, seed=3):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(n, k, generator=g)
    pairs = torch.stack(
        [
            torch.randint(0, n, (N_PAIRS,), generator=g),
            torch.randint(0, n, (N_PAIRS,), generator=g),
        ]
    )
    return Q, pairs


def orthogonal_blocks(sizes, seed=1):
    """Block-diagonal orthogonal matrix -- one random rotation per block."""
    g = torch.Generator().manual_seed(seed)
    blocks = []
    for size in sizes:
        Qb, Rb = torch.linalg.qr(torch.randn(size, size, generator=g))
        blocks.append(Qb * torch.sign(torch.diagonal(Rb)))
    return torch.block_diag(*blocks)


def trained_smodel(config, seed=0, n_filters=4):
    """A smodel with non-zero weights. At construction the readout is zeroed, so
    every invariance assertion would pass vacuously on a fresh one."""
    config["structure_model"]["DGCN_structure_layers_sizes"] = [16]
    smodel = DynamicSInvariant(n_filters=n_filters)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in smodel.model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
        smodel.log_tau.copy_(torch.linspace(-1.5, 1.6, n_filters))
    return smodel


def score(smodel, Q, D, pairs):
    smodel.set_QD(Q, D)
    return smodel.edge_score(pairs).detach()


# --------------------------------------------------------------------- #
# 1-2. rotation / sign invariance
# --------------------------------------------------------------------- #


def test_score_is_invariant_to_rotation_within_eigenspaces(config):
    # phi_j = U f_j(Lambda) U^T is unchanged by U -> UR exactly when R commutes
    # with f_j(Lambda), i.e. when R mixes only within an eigenvalue.
    smodel = trained_smodel(config)
    Q, pairs = make_basis()
    D = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.5, 0.9])
    R = orthogonal_blocks([3, 2, 1], seed=1)
    assert torch.allclose(R.T @ R, torch.eye(K), atol=1e-5)

    base = score(smodel, Q, D, pairs)
    rotated = score(smodel, Q @ R, D, pairs)

    assert base.abs().mean() > 1e-3, "readout is dead; the test would be vacuous"
    assert torch.allclose(base, rotated, atol=1e-5)


def test_rotation_across_distinct_eigenvalues_does_change_the_score(config):
    # The negative control for the test above: a rotation that does NOT commute
    # with the filter must move the score, otherwise invariance is trivial.
    # Scale by the score's SPREAD, not its mean -- the readout carries a large
    # constant offset from the MLP bias, which no rotation can affect.
    smodel = trained_smodel(config)
    Q, pairs = make_basis()
    D = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.5, 0.9])
    commuting = orthogonal_blocks([3, 2, 1], seed=1)
    mixing = orthogonal_blocks([K], seed=2)  # mixes across eigenvalues

    base = score(smodel, Q, D, pairs)
    commuting_err = (base - score(smodel, Q @ commuting, D, pairs)).abs().max()
    mixing_err = (base - score(smodel, Q @ mixing, D, pairs)).abs().max()

    assert mixing_err > 0.1 * base.std()
    assert mixing_err > 1000 * commuting_err


@pytest.mark.parametrize("flipped", [[0], [1, 3], [0, 1, 2, 3, 4, 5], [2, 5]])
def test_score_is_invariant_to_column_sign_flips(config, flipped):
    # diag(+-1) commutes with any diagonal filter, so this holds for ANY spectrum
    # -- the cheap special case of the rotation invariance above.
    smodel = trained_smodel(config)
    Q, pairs = make_basis()
    D = torch.tensor([0.1, 0.4, 0.5, 0.7, 0.9, 1.3])  # all distinct
    signs = torch.ones(K)
    signs[flipped] = -1.0

    base = score(smodel, Q, D, pairs)
    flipped_score = score(smodel, Q * signs, D, pairs)

    assert torch.equal(base, flipped_score)


def test_score_is_stable_under_mixing_of_near_degenerate_eigenvalues(config):
    # The stability claim the design rests on: sensitivity to an ARBITRARY
    # rotation inside a block is O(|f(lambda_i) - f(lambda_j)|), so a smooth
    # filter is stable exactly where the basis is ambiguous. Error must grow
    # with the spread and stay small at the ~1e-3 gaps these graphs actually have.
    smodel = trained_smodel(config)
    Q, pairs = make_basis()
    R = torch.eye(K)
    R[:3, :3] = orthogonal_blocks([3], seed=5)  # general mixing of the first block

    errors = {}
    for spread in (0.0, 1e-4, 1e-3, 1e-2):
        D = torch.tensor([0.30, 0.30 + spread, 0.30 + 2 * spread, 0.9, 1.2, 1.5])
        base = score(smodel, Q, D, pairs)
        mixed = score(smodel, Q @ R, D, pairs)
        errors[spread] = float((base - mixed).abs().max() / base.abs().mean())

    assert errors[0.0] < 1e-5                      # exactly degenerate -> exact
    assert errors[1e-3] < 0.01                     # clustered -> still stable
    assert errors[1e-4] < errors[1e-3] < errors[1e-2]   # O(spread), not a cliff


# --------------------------------------------------------------------- #
# 3-4. zero init, decode/forward fusion
# --------------------------------------------------------------------- #


def make_fgraph(n=20, seed=1):
    g = torch.Generator().manual_seed(seed)
    graph = Graph(
        x=torch.ones(n, 1),
        edge_index=torch.randint(0, n, (2, 60), generator=g),
        edge_attr=torch.randn(60, 1, generator=g),
        node_ids=torch.arange(n),
    )
    graph.edge_label_index = torch.randint(0, n, (2, 15), generator=g)
    graph.edge_label = torch.randint(0, 2, (15,), generator=g).float()
    return graph


def set_fes_model_config(config, pe_dim=6):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "f+es"
    config["model"]["edge_decoding"] = "concat"
    config["model"]["loss_fun"] = "bce_with_logits"
    config["gnn"]["dims"] = [8, 8]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = False
    config["structure_model"]["DGCN_structure_layers_sizes"] = [16]
    config["spectral"]["pe_dim"] = pe_dim
    config["spectral"]["solver"] = "chebyshev"
    config["spectral"]["update_mode"] = "update"
    config["spectral"]["use_procrustes"] = False
    config["seed"] = 42


def test_edge_score_starts_at_exactly_zero(config):
    set_fes_model_config(config)
    Q, pairs = make_basis()
    D = torch.tensor([0.1, 0.4, 0.5, 0.7, 0.9, 1.3])

    smodel = DynamicSInvariant()
    smodel.set_QD(Q, D)
    s = smodel.edge_score(pairs)

    # zero-init readout: an f+es run begins at the feature-only baseline
    assert s.shape == (N_PAIRS,)
    assert torch.equal(s.detach(), torch.zeros(N_PAIRS))


def test_classifier_matches_the_plain_decoder_at_init(config):
    set_fes_model_config(config)
    graph = make_fgraph()
    clf = FedDynamicEdgeScoreClassifier(graph)
    clf.set_QD(*make_basis(n=20)[:1], torch.tensor([0.1, 0.4, 0.5, 0.7, 0.9, 1.3]))
    clf.eval()

    with torch.no_grad():
        z, _ = clf.encode(graph)
        pred, label = clf.decode(z, graph)
        parent_pred, parent_label = DynamicClassifier.decode(clf, z, graph)
        fwd, fwd_label, _ = clf.forward(graph)
        parent_fwd, _, _ = DynamicClassifier.forward(clf, graph)

    assert torch.equal(pred, parent_pred)
    assert torch.equal(label, parent_label)
    assert torch.equal(fwd, parent_fwd)
    assert torch.equal(fwd_label, parent_label)


def test_decode_and_forward_add_the_same_edge_term(config):
    set_fes_model_config(config)
    graph = make_fgraph()
    clf = FedDynamicEdgeScoreClassifier(graph)
    Q, _ = make_basis(n=20)
    clf.set_QD(Q, torch.tensor([0.1, 0.4, 0.5, 0.7, 0.9, 1.3]))
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in clf.smodel.model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    clf.eval()

    with torch.no_grad():
        z, _ = clf.encode(graph)
        pred, _ = clf.decode(z, graph)
        parent_pred, _ = DynamicClassifier.decode(clf, z, graph)
        fwd, _, _ = clf.forward(graph)
        edge = clf.smodel.edge_score(graph.edge_label_index)

    assert not torch.allclose(pred, parent_pred, atol=1e-6)
    assert torch.allclose(pred - parent_pred, edge, atol=1e-6)
    assert torch.allclose(fwd, pred, atol=1e-6)


def test_classifier_degrades_to_the_plain_decoder_without_a_served_basis(config):
    # set_QD is never called (e.g. data_type dispatched but the spectral step
    # skipped): edge_score must return None and the classifier must still decode.
    set_fes_model_config(config)
    graph = make_fgraph()
    clf = FedDynamicEdgeScoreClassifier(graph)
    clf.eval()

    assert clf.smodel.edge_score(graph.edge_label_index) is None

    with torch.no_grad():
        z, _ = clf.encode(graph)
        pred, _ = clf.decode(z, graph)
        parent_pred, _ = DynamicClassifier.decode(clf, z, graph)
        fwd, _, _ = clf.forward(graph)

    assert torch.equal(pred, parent_pred)
    assert torch.equal(fwd, parent_pred)


# --------------------------------------------------------------------- #
# 5. federated protocol
# --------------------------------------------------------------------- #


def test_state_dict_round_trip_restores_identical_scores(config):
    set_fes_model_config(config)
    graph = make_fgraph()
    clf = FedDynamicEdgeScoreClassifier(graph)
    Q, _ = make_basis(n=20)
    clf.set_QD(Q, torch.tensor([0.1, 0.4, 0.5, 0.7, 0.9, 1.3]))
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in clf.smodel.model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)

    assert "smodel" in clf.state_dict()
    saved = copy.deepcopy(clf.state_dict())
    before = clf.smodel.edge_score(graph.edge_label_index).detach().clone()

    with torch.no_grad():  # a FedAvg round moves both the MLP and the taus
        for p in clf.smodel.model.parameters():
            p.add_(torch.randn(p.shape, generator=g))
        clf.smodel.log_tau.add_(0.7)
    mutated = clf.smodel.edge_score(graph.edge_label_index).detach().clone()

    clf.load_state_dict(saved)
    restored = clf.smodel.edge_score(graph.edge_label_index).detach().clone()

    assert not torch.allclose(before, mutated, atol=1e-6)
    assert torch.allclose(before, restored, atol=1e-6)


def test_parameters_include_the_mlp_and_log_tau(config):
    set_fes_model_config(config)
    smodel = DynamicSInvariant(n_filters=4)

    params = smodel.parameters()

    assert any(p is smodel.log_tau for p in params)
    for p in smodel.model.parameters():
        assert any(q is p for q in params)
    assert len(params) == len(list(smodel.model.parameters())) + 1
    # the classifier surfaces them too, so the optimizer sees them
    clf = FedDynamicEdgeScoreClassifier(make_fgraph())
    assert any(p is clf.smodel.log_tau for p in clf.parameters())


def test_grads_round_trip(config):
    set_fes_model_config(config)
    Q, pairs = make_basis()
    D = torch.tensor([0.1, 0.4, 0.5, 0.7, 0.9, 1.3])
    source = trained_smodel(config, seed=2)
    source.set_QD(Q, D)
    source.edge_score(pairs).sum().backward()

    grads = source.get_grads()
    assert set(grads) == {"model"}
    assert len(grads["model"]) == len(source.parameters())
    assert source.log_tau.grad is not None

    target = DynamicSInvariant(n_filters=4)
    target.set_grads(grads)

    for p, expected in zip(target.parameters(), grads["model"]):
        assert expected is not None
        assert torch.equal(p.grad, expected)


def test_train_eval_zero_grad_reach_the_inner_module(config):
    set_fes_model_config(config)
    graph = make_fgraph()
    clf = FedDynamicEdgeScoreClassifier(graph)
    Q, pairs = make_basis(n=20)
    clf.set_QD(Q, torch.tensor([0.1, 0.4, 0.5, 0.7, 0.9, 1.3]))

    clf.train()
    assert clf.smodel.model.training is True
    clf.eval()
    assert clf.smodel.model.training is False

    clf.smodel.edge_score(graph.edge_label_index).sum().backward()
    assert clf.smodel.log_tau.grad is not None
    clf.zero_grad()
    assert float(clf.smodel.log_tau.grad.abs().max()) == 0.0
    for p in clf.smodel.model.parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0
    clf.zero_grad(set_to_none=True)
    assert clf.smodel.log_tau.grad is None


# --------------------------------------------------------------------- #
# 6. config gating
# --------------------------------------------------------------------- #


def test_assert_cfg_rules_for_fes(config):
    set_fes_model_config(config)

    for solver in ("chebyshev", "exact"):
        config["spectral"]["solver"] = solver
        assert_cfg(config)

    config["spectral"]["solver"] = "arnoldi"
    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "f+es" in str(exc.value) and "solver" in str(exc.value)

    config["spectral"]["solver"] = "chebyshev"
    config["spectral"]["pe_dim"] = 0
    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.pe_dim must be >0" in str(exc.value)

    config["spectral"]["pe_dim"] = 6
    config["spectral"]["update_mode"] = None
    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral.update_mode is None" in str(exc.value)


def test_arnoldi_gate_is_scoped_to_fes(config):
    # f+pe and f+s must keep working on the historical solver
    set_fes_model_config(config)
    config["spectral"]["solver"] = "arnoldi"
    for data_type in ("f+pe", "f+s", "feature"):
        config["model"]["data_type"] = data_type
        assert_cfg(config)


# --------------------------------------------------------------------- #
# 7. dispatch + end-to-end
# --------------------------------------------------------------------- #


def make_dense_snapshots(num_nodes=16, num_snaps=3, seed=5):
    g = torch.Generator().manual_seed(seed)
    edges = [(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    snaps = []
    for _ in range(num_snaps):
        snap = Data(
            x=torch.ones(num_nodes, 1),
            edge_index=edge_index.clone(),
            edge_attr=torch.randn(edge_index.size(1), 1, generator=g),
            num_nodes=num_nodes,
        )
        snap.node_ids = torch.arange(num_nodes)
        snaps.append(snap)
    return snaps


def build_fes_server(config, num_subgraphs=2, num_snaps=3):
    snaps = make_dense_snapshots(num_snaps=num_snaps)
    server = DynamicServer(snaps)
    for client_snaps in partition_snapshots(snaps, num_subgraphs):
        server.add_client(client_snaps)
    return server


def test_fes_dispatch_builds_the_edge_score_classifier(config):
    set_fes_model_config(config)
    server = build_fes_server(config)

    server.initialize_FL()

    assert isinstance(server.classifier, FedDynamicEdgeScoreClassifier)
    assert isinstance(server.classifier.smodel, DynamicSInvariant)
    for client in server.clients:
        assert isinstance(client.classifier, FedDynamicEdgeScoreClassifier)
        assert isinstance(client.classifier.smodel, DynamicSInvariant)


def test_fes_run_can_checkpoint(config, tmp_path):
    # was a strict xfail: _get_sfv reached into smodel.graph.x, which
    # DynamicSInvariant does not have, so f+es + auto_resume died with
    # AttributeError at the first save. Now routed through get_SFV/set_SFV.
    set_fes_model_config(config)
    config["model"]["iterations"] = 1
    config["model"]["local_epochs"] = 1
    config["meta"]["is_meta"] = False
    config["metric"]["mrr_method"] = "min"
    config["experimental"]["rank_eval_multiplier"] = 10
    config["optim"]["base_lr"] = 0.01
    config["optim"]["scheduler"] = "none"
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    config["dataset"]["name"] = "toy"
    config["train"]["auto_resume"] = True
    config["train"]["ckpt_period"] = 1
    config["train"]["ckpt_dir"] = str(tmp_path)
    server = build_fes_server(config)

    results = server.joint_train_w(FL=True)

    assert len(results["mrr_history"]) == 2


@pytest.mark.parametrize("update_mode", ["update", "recompute"])
def test_fes_end_to_end_smoke(config, update_mode):
    set_fes_model_config(config)
    config["spectral"]["update_mode"] = update_mode
    config["model"]["iterations"] = 1
    config["model"]["local_epochs"] = 1
    config["train"]["auto_resume"] = False
    config["meta"]["is_meta"] = False
    config["metric"]["mrr_method"] = "min"
    config["experimental"]["rank_eval_multiplier"] = 10
    config["optim"]["base_lr"] = 0.01
    config["optim"]["scheduler"] = "none"
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    server = build_fes_server(config)

    results = server.joint_train_w(FL=True)

    assert len(results["mrr_history"]) == 2
    for mrr in results["mrr_history"]:
        assert 0.0 <= mrr <= 1.0
    # slice/order contract: one served row per node of the owning subgraph
    pe_dim = config["spectral"]["pe_dim"]
    assert server.classifier.smodel.Q.shape == (16, pe_dim)
    for client in server.clients:
        assert client.classifier.smodel.Q.shape == (client.num_nodes(), pe_dim)
        assert client.classifier.smodel.D.shape == (pe_dim,)
