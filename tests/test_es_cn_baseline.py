"""spectral.es_features='cn': the structural baseline a spectral new-pair claim
has to clear.

cn is log1p(|N(u) & N(v)|) over the SAME cumulative graph 'persist' reads,
through a head of the SAME shape, so the two baselines differ only in what they
compute from one graph. That is what makes it an attribution control, and it
only works if the count is exactly the count: a self-loop that survives, a
multi-edge that is summed instead of collapsed, or a chunk loop that misses the
tail all turn the arm into a measurement of something nobody named -- and the
tail is only reachable at full eval-batch size, where nobody looks. These pin
the count, the head shape, the per-arm cost, the identity, the config gate and
the inertness of every other arm.
"""

import copy

import pytest
import torch

import src
import src.utils.graph_partitioning as graph_partitioning
from config.assertions import assert_cfg
from config.config import get_default_config
from src.GNN.dynamic_classifier import DynamicClassifier
from src.GNN.fed_dynamic_classifier import (
    DynamicSInvariant,
    FedDynamicEdgeScoreClassifier,
)
from src.dynamic_server import DynamicServer
from src.utils.graph_partitioning import partition_snapshots
from test_edge_score_smodel import make_fgraph, set_fes_model_config
from test_checkpoint_wandb import make_server, seed_all
from test_fl_local_baseline import make_toy_snapshots
from test_run_identity import set_identity_config


N_NODES = 12
# 0 and 1 share {4,5,6}; 2 shares {4,6} with 0; {3,7,8} is a separate triangle;
# 10 and 11 are isolated.
EDGES = [
    (0, 4), (1, 4), (2, 4),
    (0, 5), (1, 5),
    (0, 6), (1, 6), (2, 6),
    (3, 7), (7, 8), (8, 3),
]
# (0,4) is an EDGE, so a surviving self-loop on 4 would push it from 0 to 1;
# (0,0) is the u==v degenerate; (2,3) crosses components; 10/11 are isolated.
PROBE_PAIRS = [(0, 1), (1, 0), (0, 2), (0, 4), (2, 3), (0, 10), (10, 11), (0, 0), (3, 8), (4, 6)]
EXPECTED_CN = [3, 3, 2, 0, 0, 0, 0, 3, 1, 3]
PROBE = torch.tensor(PROBE_PAIRS, dtype=torch.long).t()

# the same graph with every hazard the constructor claims to neutralise
NOISY_EDGES = EDGES + EDGES[:4] + [(0, 4)] + [(4, 4), (0, 0), (6, 6)] + [(4, 1), (1, 4)]


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


def edge_index(edges):
    return torch.tensor(edges, dtype=torch.long).t()


def snaps_from(schedule, n_nodes=N_NODES):
    from torch_geometric.data import Data

    out = []
    for edges in schedule:
        ei = edge_index(edges)
        snap = Data(x=torch.ones(n_nodes, 1), edge_index=ei,
                    edge_attr=torch.ones(ei.size(1), 1), num_nodes=n_nodes)
        snap.node_ids = torch.arange(n_nodes)
        out.append(snap)
    return out


def reference_cn(edges, pairs):
    """|N(u) & N(v)| from python sets -- self-loops are not neighbour relations
    and a repeated edge is one relation."""
    nb = {}
    for u, v in edges:
        if u != v:
            nb.setdefault(u, set()).add(v)
            nb.setdefault(v, set()).add(u)
    return [len(nb.get(u, set()) & nb.get(v, set())) for u, v in pairs]


def smodel(config, features, n_filters=4, hidden=(16,)):
    config["structure_model"]["DGCN_structure_layers_sizes"] = list(hidden)
    config["spectral"]["es_features"] = features
    return DynamicSInvariant(n_filters=n_filters)


def counts(smodel_, pairs=PROBE):
    feat = smodel_._common_neighbours(pairs)
    return torch.expm1(feat.squeeze(-1))


# --------------------------------------------------------------------- #
# 1. the count itself
# --------------------------------------------------------------------- #


def test_cn_counts_common_neighbours_on_a_hand_enumerable_graph(config):
    sm = smodel(config, "cn")
    sm.set_adj(edge_index(EDGES), N_NODES)

    feat = sm._common_neighbours(PROBE)

    assert feat.shape == (PROBE.size(1), 1)
    # literal expectations, so an equally wrong reference cannot rescue the code
    assert reference_cn(EDGES, PROBE_PAIRS) == EXPECTED_CN
    assert counts(sm).tolist() == pytest.approx(EXPECTED_CN)
    # log1p, not the raw count: monotone, so it cannot reorder anything
    assert torch.allclose(feat.squeeze(-1), torch.log1p(torch.tensor(EXPECTED_CN, dtype=torch.float32)))


def test_multi_edges_and_self_loops_cannot_change_the_count(config):
    # data[:]=1 collapses a repeated edge to one neighbour relation and
    # setdiag(0) drops self-loops. Without either, (0,1) and (0,4) inflate.
    clean = smodel(config, "cn")
    clean.set_adj(edge_index(EDGES), N_NODES)
    noisy = smodel(config, "cn")
    noisy.set_adj(edge_index(NOISY_EDGES), N_NODES)

    assert reference_cn(NOISY_EDGES, PROBE_PAIRS) == EXPECTED_CN
    assert torch.equal(counts(noisy), counts(clean))
    assert counts(noisy).tolist() == pytest.approx(EXPECTED_CN)
    assert clean.adj.max() == 1.0                       # a neighbour SET
    assert float(clean.adj.diagonal().sum()) == 0.0     # no node is its own neighbour


def test_the_count_is_symmetric_and_ignores_the_stored_direction(config):
    sm = smodel(config, "cn")
    sm.set_adj(edge_index(EDGES), N_NODES)
    forward = counts(sm)

    flipped_pairs = smodel(config, "cn")
    flipped_pairs.set_adj(edge_index(EDGES), N_NODES)
    reversed_input = smodel(config, "cn")
    reversed_input.set_adj(edge_index([(v, u) for u, v in EDGES]), N_NODES)

    assert torch.equal(counts(flipped_pairs, PROBE.flip(0)), forward)  # CN(u,v)==CN(v,u)
    assert torch.equal(counts(reversed_input), forward)                # direction is irrelevant


def test_an_isolated_or_absent_node_contributes_nothing(config):
    sm = smodel(config, "cn")
    sm.set_adj(edge_index(EDGES), N_NODES)

    isolated = counts(sm, torch.tensor([[10, 11, 0, 9], [11, 10, 10, 10]]))

    assert isolated.tolist() == [0.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------- #
# 2. chunking -- only wrong on batches nobody eyeballs
# --------------------------------------------------------------------- #


def test_the_chunk_loop_covers_every_pair(config):
    # step=65536 inside _common_neighbours. A full eval batch is
    # rank_eval_multiplier negatives per source, so it clears one chunk easily;
    # an off-by-one there corrupts ONLY the large batches.
    n_pairs = 70000
    assert n_pairs > 65536
    sm = smodel(config, "cn")
    sm.set_adj(edge_index(EDGES), N_NODES)
    tiled = PROBE.repeat(1, n_pairs // PROBE.size(1) + 1)[:, :n_pairs]
    expected = torch.tensor(EXPECTED_CN * (n_pairs // len(EXPECTED_CN) + 1))[:n_pairs]

    big = counts(sm, tiled)

    assert big.shape == (n_pairs,)
    assert torch.allclose(big, expected.to(torch.float32))
    # the chunk boundary itself, and the single-chunk answer for the same prefix
    assert big[65535].item() == pytest.approx(expected[65535].item())
    assert big[65536].item() == pytest.approx(expected[65536].item())
    assert torch.equal(big[: PROBE.size(1)], counts(sm))


def test_an_empty_pair_batch_is_shaped_not_crashed(config):
    sm = smodel(config, "cn")
    sm.set_adj(edge_index(EDGES), N_NODES)

    feat = sm._common_neighbours(torch.zeros(2, 0, dtype=torch.long))

    assert feat.shape == (0, 1)


# --------------------------------------------------------------------- #
# 3. head shape and per-arm cost
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "features,width", [("spec", 6), ("persist", 1), ("both", 7), ("cn", 1)]
)
def test_head_input_width_per_arm(config, features, width):
    # cn must be 1 wide, exactly like persist: the two baselines are compared
    # against each other, so a capacity difference would be a confound.
    sm = smodel(config, features)

    assert sm.model[0].in_features == width


def test_cn_and_persist_have_identical_parameter_shapes(config):
    shapes = {}
    for features in ("persist", "cn"):
        sm = smodel(config, features)
        shapes[features] = [tuple(p.shape) for p in sm.parameters()]

    assert shapes["cn"] == shapes["persist"]
    assert shapes["cn"] != [tuple(p.shape) for p in smodel(config, "spec").parameters()]


@pytest.mark.parametrize("features", ["spec", "persist", "both"])
def test_only_the_cn_arm_builds_an_adjacency(config, features):
    # the CSR is O(nnz) memory per snapshot; no other arm reads it
    other = smodel(config, features)
    other.set_adj(edge_index(EDGES), N_NODES)
    cn = smodel(config, "cn")
    cn.set_adj(edge_index(EDGES), N_NODES)

    assert other.adj is None
    assert other._common_neighbours(PROBE) is None
    assert cn.adj is not None
    assert other.keys is not None and torch.equal(other.keys, cn.keys)  # same graph served


def test_a_second_serve_replaces_the_adjacency(config):
    sm = smodel(config, "cn")
    sm.set_adj(edge_index(EDGES), N_NODES)
    first = counts(sm)

    sm.set_adj(edge_index(EDGES[:3]), N_NODES)

    assert not torch.equal(counts(sm), first)


# --------------------------------------------------------------------- #
# 4. the empty-graph hole (D1): an empty CSR, never None
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("served", ["empty", "none"])
def test_an_edgeless_graph_still_feeds_the_cn_head(config, served):
    # Leaving adj=None here made _common_neighbours and then edge_score return
    # None, so the classifier fell back to the plain decoder with no error and no
    # log -- a cn arm that was silently feature-only while _run_id stamped
    # esf-cn. persist never had that hole (empty keys still feed a real 0), so
    # the two baselines only compare if cn behaves the same way.
    edges = torch.zeros(2, 0, dtype=torch.long) if served == "empty" else None
    persist = smodel(config, "persist")
    persist.set_adj(edges, N_NODES)
    cn = smodel(config, "cn")
    cn.set_adj(edges, N_NODES)

    feat = cn._common_neighbours(PROBE)

    assert cn.adj is not None                       # regression guard
    assert cn.adj.shape == (N_NODES, N_NODES) and cn.adj.nnz == 0
    assert feat is not None
    assert torch.equal(feat, torch.zeros(PROBE.size(1), 1))   # 0, not nan
    assert torch.equal(persist._persistence(PROBE), torch.zeros(PROBE.size(1), 1))


def test_the_head_still_fires_on_an_edgeless_graph(config):
    # what makes the arm non-silent: a trained head returns a real score for
    # every pair instead of dropping the edge term out of the logit.
    cn = smodel(config, "cn")
    cn.set_adj(torch.zeros(2, 0, dtype=torch.long), N_NODES)
    with torch.no_grad():
        for p in cn.model.parameters():
            p.add_(0.5)

    score = cn.edge_score(PROBE)

    assert score is not None and score.shape == (PROBE.size(1),)
    assert float(score.std()) == 0.0 and float(score[0]) != 0.0  # constant, not dropped


def test_an_empty_serve_replaces_a_populated_adjacency(config):
    # set_adj is called once per snapshot; a graph that goes empty must not leave
    # the previous snapshot's counts standing
    cn = smodel(config, "cn")
    cn.set_adj(edge_index(EDGES), N_NODES)
    cn.set_adj(torch.zeros(2, 0, dtype=torch.long), N_NODES)

    assert cn.adj is not None and cn.adj.nnz == 0
    assert float(counts(cn).abs().sum()) == 0.0


def test_a_client_with_no_induced_edges_is_served_an_empty_adjacency(config, monkeypatch):
    # the case D1 named as reachable: under sharding a client can own a node set
    # with no cumulative edge inside it. It must get an empty CSR, not None, or
    # that client alone scores feature-only for the whole run.
    monkeypatch.setattr(
        graph_partitioning, "random_assign",
        lambda n, c: {0: torch.arange(0, n // 2), 1: torch.arange(n // 2, n)},
    )
    server = fes_cn_run(config, 2, snaps=snaps_from([[(0, 1), (1, 2), (2, 3), (0, 3)],
                                                     [(0, 2), (1, 3)],
                                                     [(0, 1), (2, 3)]]))
    server._spectral_step(0, config["model"]["smodel_type"])
    empty_side = server.clients[1]

    assert empty_side.snaps[0].edge_index.numel() == 0   # the premise
    assert empty_side.classifier.smodel.adj is not None
    assert empty_side.classifier.smodel.adj.nnz == 0
    pairs = torch.tensor([[0, 1], [1, 2]])
    assert float(counts(empty_side.classifier.smodel, pairs).abs().sum()) == 0.0
    assert empty_side.classifier.smodel.edge_score(pairs) is not None
    # the populated client is unaffected
    assert server.clients[0].classifier.smodel.adj.nnz > 0


def test_the_pipeline_serves_an_adjacency_to_every_owner_at_every_snapshot(config, monkeypatch):
    # the only remaining route to adj=None is set_adj never being called.
    # _spectral_step returns early on a falsy `share` BEFORE the set_adj block,
    # but f+es forces smodel_type='Invariant', which always fills share -- assert
    # that instead of trusting it.
    server = fes_cn_run(config, 2)
    seen = []
    original = DynamicSInvariant.set_adj

    def spy(self, edge_index, num_nodes):
        original(self, edge_index, num_nodes)
        seen.append(self.adj is not None)

    monkeypatch.setattr(DynamicSInvariant, "set_adj", spy)
    server.joint_train_w(FL=True)

    n_owners = 1 + len(server.clients)
    assert len(seen) == n_owners * (len(server.global_snaps) - 1)
    assert all(seen)


def test_an_unserved_classifier_degrades_instead_of_crashing(config):
    # not reachable from the f+es pipeline (above), but the smodel is constructed
    # directly by probes and tests: no adjacency must mean the plain decoder, not
    # an exception mid-eval.
    set_fes_model_config(config)
    config["spectral"]["es_features"] = "cn"
    graph = make_fgraph()
    clf = FedDynamicEdgeScoreClassifier(graph)
    clf.eval()

    assert clf.smodel.adj is None          # set_adj was never called
    assert clf.smodel.edge_score(graph.edge_label_index) is None
    with torch.no_grad():
        z, _ = clf.encode(graph)
        pred, _ = clf.decode(z, graph)
        parent, _ = DynamicClassifier.decode(clf, z, graph)
    assert torch.equal(pred, parent)


# 5. identity and config gate
# --------------------------------------------------------------------- #


def test_run_identity_separates_cn_from_every_other_arm(config, tmp_path):
    set_identity_config(config, tmp_path)
    config["model"]["data_type"] = "f+es"
    config["spectral"]["solver"] = "chebyshev"
    server = DynamicServer(make_toy_snapshots(num_snaps=2))

    ids = {}
    for features in ("spec", "persist", "both", "cn"):
        config["spectral"]["es_features"] = features
        ids[features] = server._run_id()

    assert len(set(ids.values())) == 4
    assert "esf-cn" in ids["cn"]
    for features in ("spec", "persist", "both"):
        assert "esf-cn" not in ids[features]


def test_assert_cfg_gates_es_features(config):
    set_fes_model_config(config)

    for features in ("spec", "persist", "both", "cn"):
        config["spectral"]["es_features"] = features
        assert_cfg(config)

    for bad in ("CN", "cn ", "adamic_adar", "", None):
        config["spectral"]["es_features"] = bad
        with pytest.raises(ValueError) as exc:
            assert_cfg(config)
        assert "spectral.es_features" in str(exc.value)


def test_the_smodel_refuses_an_unknown_arm_too(config):
    # defence in depth: assert_cfg is the launch-time gate, but probes and tests
    # construct the smodel directly
    config["spectral"]["es_features"] = "adamic_adar"
    with pytest.raises(ValueError) as exc:
        DynamicSInvariant(n_filters=4)
    assert "es_features" in str(exc.value) and "cn" in str(exc.value)


def test_the_gate_is_scoped_to_fes(config):
    # es_features is only read by the f+es smodel; the other data types must not
    # start failing on a stale value
    set_fes_model_config(config)
    config["spectral"]["es_features"] = "cn"
    for data_type in ("f+pe", "f+s", "feature"):
        config["model"]["data_type"] = data_type
        assert_cfg(config)


# --------------------------------------------------------------------- #
# 6. default inertness
# --------------------------------------------------------------------- #


def test_the_default_arm_is_spec_and_pays_for_nothing(config):
    assert get_default_config()["spectral"]["es_features"] == "spec"

    config["structure_model"]["DGCN_structure_layers_sizes"] = [16]
    sm = DynamicSInvariant(n_filters=4)
    sm.set_adj(edge_index(EDGES), N_NODES)

    assert sm.features == "spec"
    assert sm.adj is None
    assert sm.n_filters == 4
    assert sm.model[0].in_features == 6
    # the cn branch of edge_score is unreachable on the default arm
    sm._common_neighbours = _boom
    sm.set_QD(torch.randn(N_NODES, 3), torch.tensor([0.1, 0.4, 0.9]))
    assert sm.edge_score(PROBE).shape == (PROBE.size(1),)


def _boom(*_a, **_k):
    raise AssertionError("the cn path must not run on a non-cn arm")


# --------------------------------------------------------------------- #
# 7. served graph: server and clients
# --------------------------------------------------------------------- #


def fes_cn_run(config, num_clients, num_snaps=3, features="cn", snaps=None):
    set_fes_model_config(config, pe_dim=4)
    config["spectral"]["es_features"] = features
    config["spectral"]["update_mode"] = "keep"
    config["subgraph"]["num_subgraphs"] = num_clients
    config["model"]["iterations"] = 1
    config["model"]["local_epochs"] = 1
    config["train"]["auto_resume"] = False
    config["meta"]["is_meta"] = False
    config["metric"]["mrr_method"] = "min"
    config["experimental"]["rank_eval_multiplier"] = 10
    config["optim"]["base_lr"] = 0.01
    config["optim"]["scheduler"] = "none"
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    seed_all(42)
    snaps = make_toy_snapshots(N=30, num_snaps=num_snaps, seed=7) if snaps is None else snaps
    server = make_server(snaps, partition_snapshots(snaps, num_clients))
    server.initialize_FL()
    return server


def cum_neighbours(server):
    e = server._cum_edges.cpu().tolist()
    nb = {}
    for u, v in zip(e[0], e[1]):
        if u != v:
            nb.setdefault(u, set()).add(v)
            nb.setdefault(v, set()).add(u)
    return nb


@pytest.mark.parametrize("num_clients", [1, 2])
def test_every_owner_is_served_its_own_induced_cn_graph(config, num_clients):
    # the client adjacency is remapped to LOCAL ids in node_ids order, the same
    # order as its Q rows. A remap bug is invisible in the loss and shows up
    # only as a wrong count.
    server = fes_cn_run(config, num_clients)
    t = 0
    server._spectral_step(t, config["model"]["smodel_type"])
    nb = cum_neighbours(server)

    assert server.classifier.smodel.adj is not None
    checked = 0
    for cl in server.clients:
        nid = cl.snaps[t].node_ids.cpu().tolist()
        owned = set(nid)
        local_pairs = [(i, j) for i in range(len(nid)) for j in range(i + 1, len(nid))]
        pairs = torch.tensor(local_pairs, dtype=torch.long).t()
        got = counts(cl.classifier.smodel, pairs).tolist()
        expected = [
            len(nb.get(nid[i], set()) & nb.get(nid[j], set()) & owned)
            for i, j in local_pairs
        ]
        assert got == pytest.approx(expected)
        checked += sum(1 for c in expected if c > 0)
    assert checked > 0  # otherwise every count is a trivial zero


def test_the_server_count_is_the_global_one(config):
    server = fes_cn_run(config, 2)
    server._spectral_step(0, config["model"]["smodel_type"])
    nb = cum_neighbours(server)
    pairs = torch.tensor([(u, v) for u in range(6) for v in range(u + 1, 12)]).t()

    got = counts(server.classifier.smodel, pairs).tolist()

    expected = [len(nb.get(int(u), set()) & nb.get(int(v), set()))
                for u, v in pairs.t().tolist()]
    assert got == pytest.approx(expected)
    assert max(expected) > 0


def test_cn_reaches_the_decoder_and_moves_the_score(config):
    server = fes_cn_run(config, 1)
    server._spectral_step(0, config["model"]["smodel_type"])
    smodel_ = server.classifier.smodel
    graph = server.clients[0].snaps[0]
    graph.edge_label_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    with torch.no_grad():  # zero-init readout: untrained, the term is exactly 0
        assert float(smodel_.edge_score(graph.edge_label_index).abs().max()) == 0.0
        for p in smodel_.model.parameters():
            p.add_(0.3)
    with torch.no_grad():
        scored = smodel_.edge_score(graph.edge_label_index)

    assert scored.shape == (3,)
    assert float(scored.std()) > 0.0  # different counts -> different scores


def test_a_cn_run_completes_end_to_end(config):
    server = fes_cn_run(config, 2)

    results = server.joint_train_w(FL=True)

    assert len(results["mrr_history"]) == 2
    for mrr in results["mrr_history"]:
        assert 0.0 <= mrr <= 1.0
    assert server.classifier.smodel.adj is not None
    for cl in server.clients:
        assert cl.classifier.smodel.adj is not None


def test_the_cn_head_trains_the_mlp_and_leaves_log_tau_dead(config):
    # cn has no filters, so log_tau receives no gradient; it still travels in
    # state_dict, which is how persist already behaves.
    sm = smodel(config, "cn")
    sm.set_adj(edge_index(EDGES), N_NODES)
    with torch.no_grad():
        for p in sm.model.parameters():
            p.add_(0.3)

    sm.edge_score(PROBE).sum().backward()

    assert sm.log_tau.grad is None
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in sm.model.parameters())


def test_a_self_pair_reads_as_the_node_degree_and_the_sampler_can_draw_one(config):
    # NOT wrong -- |N(u) & N(u)| IS deg(u) -- but load-bearing for reading the
    # arm. _sample_filtered_negatives excludes a source's true positives and
    # nothing else, so it can draw v'=u (P ~ 0.41 per source at N=1899, K=1000).
    # cn hands that negative the LARGEST count in u's row while persist hands it
    # a 0, so the artifact penalises cn (and spec's cos, which is 1 there) but
    # not persist -- in a comparison whose whole point is the arms.
    from src.metrics.mrr import _sample_filtered_negatives

    cn = smodel(config, "cn")
    cn.set_adj(edge_index(EDGES), N_NODES)
    persist = smodel(config, "persist")
    persist.set_adj(edge_index(EDGES), N_NODES)
    with_zero = torch.tensor([[0] * N_NODES, list(range(N_NODES))])

    self_count = counts(cn, torch.tensor([[0], [0]]))

    assert float(self_count) == 3.0                       # deg(0)
    assert float(counts(cn, with_zero).max()) == 3.0      # the largest in the row
    assert float(persist._persistence(torch.tensor([[0], [0]]))) == 0.0
    # spec is hit too, and harder: its cos feature is exactly 1 at a self pair,
    # the maximum a cosine can take. Measured on uci C1, removing self negatives
    # lifts mrr_repeat by +0.030 for spec, +0.006 for cn and +0.000 for persist.
    spec = smodel(config, "spec")
    g = torch.Generator().manual_seed(0)
    Q = torch.randn(N_NODES, 6, generator=g)
    spec.set_QD(Q, torch.linspace(0.1, 0.6, 6))
    qn = Q / (Q.norm(dim=-1, keepdim=True) + 1e-12)
    cos = (qn[PROBE[0]] * qn[PROBE[1]]).sum(-1)
    assert float((qn[0] * qn[0]).sum()) == pytest.approx(1.0)
    assert float(cos.abs().max()) <= 1.0 + 1e-6

    torch.manual_seed(0)
    sources = torch.tensor([3, 7, 11])
    v_neg = _sample_filtered_negatives(
        sources, sources, sources + 1, 50, 200, "cpu"
    )
    assert int((v_neg == sources.unsqueeze(1)).sum()) > 0
