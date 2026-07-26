"""Load-bearing invariants of the federated live-update loop.

Every MRR in results.md depends on these four properties, and none of them is
pinned anywhere else in the suite. A silent regression in any one of them makes
the reported numbers mean something other than what they claim:

  1. the reported eval at t runs BEFORE any training on t (leakage-freeness);
  2. the spectral basis served at t is built from edges up to t only;
  3. the reported eval scores snapshot t+1's TEST split, not its train/val edges;
  4. FedAvg aggregates the participating clients weighted by node count, and
     abstaining clients are excluded.
"""

import copy

import pytest
import torch
from torch_geometric.data import Data

import src
import src.dynamic_server as ds_mod
from src.dynamic_client import DynamicClient
from src.dynamic_server import DynamicServer
from src.train.federated_orchestrator import _clone_state
from src.utils.graph_partitioning import partition_snapshots

BASE_NODES = 16
SIG_BASE = 16  # signature edges live on reserved nodes 16..23


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = copy.deepcopy(src.config._registry)
    yield
    src.config._registry.clear()
    src.config._registry.update(saved)


def tiny_run_config(config, num_subgraphs=2):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["dataset"]["split"] = [0.8, 0.1, 0.1]
    config["subgraph"]["num_subgraphs"] = num_subgraphs
    config["model"]["data_type"] = "feature"
    config["model"]["edge_decoding"] = "concat"
    config["model"]["loss_fun"] = "bce_with_logits"
    config["model"]["iterations"] = 2
    config["model"]["local_epochs"] = 2
    config["gnn"]["dims"] = [8, 8]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = False
    config["train"]["internal_validation_tolerance"] = 5
    config["train"]["auto_resume"] = False
    config["metric"]["mrr_method"] = "min"
    config["metric"]["eval_scope"] = "auto"
    config["experimental"]["rank_eval_multiplier"] = 20
    config["optim"]["optimizer"] = "adam"
    config["optim"]["base_lr"] = 0.005
    config["optim"]["scheduler"] = "none"
    config["meta"]["is_meta"] = False
    config["seed"] = 42


def signature_edge(t):
    return (SIG_BASE + 2 * t, SIG_BASE + 2 * t + 1)


def make_signature_snapshots(num_snaps=4, seed=11):
    """Snapshots over a shared dense base plus one edge unique to each snapshot,
    so a graph containing snapshot t+1's signature edge is provably reading ahead.

    The base is complete over BASE_NODES so that every client subgraph still has
    enough edges to form a train/val split at any client count we test (an
    abstaining client would make the loop skip the code under test)."""
    N = SIG_BASE + 2 * num_snaps
    g = torch.Generator().manual_seed(seed)
    base = [(u, v) for u in range(BASE_NODES) for v in range(BASE_NODES) if u != v]
    snaps = []
    for t in range(num_snaps):
        edges = base + [signature_edge(t)]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        snap = Data(
            x=torch.ones(N, 1),
            edge_index=edge_index,
            edge_attr=torch.randn(edge_index.size(1), 1, generator=g),
            num_nodes=N,
        )
        snap.node_ids = torch.arange(N)
        snaps.append(snap)
    return snaps


def build_server(snaps, num_subgraphs):
    server = DynamicServer(snaps)
    for client_snaps in partition_snapshots(snaps, num_subgraphs):
        server.add_client(client_snaps)
    return server


def test_eval_at_t_precedes_all_training_on_t(config):
    # ROLAND live-update scores snapshot t+1 with the model as it stands BEFORE
    # snapshot t's fine-tune. Clients train on t+1's edges too (their own train
    # split), so if training moved ahead of the eval the reported MRR would be
    # scored on edges the model had just fitted.
    tiny_run_config(config, num_subgraphs=2)
    snaps = make_signature_snapshots(num_snaps=4)
    server = build_server(snaps, 2)

    events = []
    real_spectral = DynamicServer._spectral_step
    real_eval = DynamicServer._eval_mrr
    real_finetune = DynamicClient.local_finetune
    real_refresh = DynamicClient.refresh

    DynamicServer._spectral_step = lambda self, t, smt: events.append(("spectral", t))
    DynamicServer._eval_mrr = lambda self, t, k, m: (
        events.append(("eval", t)) or real_eval(self, t, k, m)
    )
    DynamicClient.local_finetune = lambda self, t, e, loss: (
        events.append(("train", t)) or real_finetune(self, t, e, loss)
    )
    DynamicClient.refresh = lambda self, t: (
        events.append(("refresh", t)) or real_refresh(self, t)
    )
    try:
        server.joint_train_w(FL=True)
    finally:
        DynamicServer._spectral_step = real_spectral
        DynamicServer._eval_mrr = real_eval
        DynamicClient.local_finetune = real_finetune
        DynamicClient.refresh = real_refresh

    assert [k for k, _ in events].count("eval") == 3  # 4 snapshots -> 3 task pairs
    for t in range(3):
        eval_at = events.index(("eval", t))
        trains = [i for i, e in enumerate(events) if e == ("train", t)]
        assert trains, f"no training happened at t={t}"
        assert eval_at < min(trains), f"t={t}: trained before the reported eval"
        # and nothing from a LATER snapshot leaked in ahead of this eval
        assert all(
            step <= t for kind, step in events[:eval_at] if kind in ("train", "eval")
        )


def test_spectral_basis_never_sees_future_edges(config):
    # The served basis is a function of the graph; if the cumulative union ran
    # one snapshot ahead, the eigenvectors would encode the very edges being
    # predicted. It must also run BEFORE the eval, since encoding snap_t needs Q_t.
    tiny_run_config(config, num_subgraphs=2)
    config["model"]["data_type"] = "f+pe"
    config["spectral"]["update_mode"] = "recompute"
    config["spectral"]["basis_source"] = "laplacian"
    config["spectral"]["use_procrustes"] = False
    config["spectral"]["pe_dim"] = 4
    num_snaps = 4
    snaps = make_signature_snapshots(num_snaps=num_snaps)
    server = build_server(snaps, 2)

    seen_graphs = {}
    order = []
    real_features = DynamicServer.get_spectral_features
    real_eval = DynamicServer._eval_mrr

    def spy_features(self, graph, smodel_type, ss_idx, *args, **kwargs):
        pairs = {
            (int(a), int(b)) for a, b in zip(graph.edge_index[0], graph.edge_index[1])
        }
        seen_graphs[ss_idx] = pairs
        order.append(("spectral", ss_idx))
        return real_features(self, graph, smodel_type, ss_idx, *args, **kwargs)

    def spy_eval(self, t, k, m):
        order.append(("eval", t))
        return real_eval(self, t, k, m)

    DynamicServer.get_spectral_features = spy_features
    DynamicServer._eval_mrr = spy_eval
    try:
        server.joint_train_w(FL=True)
    finally:
        DynamicServer.get_spectral_features = real_features
        DynamicServer._eval_mrr = real_eval

    assert set(seen_graphs) == set(range(num_snaps - 1))
    for t, pairs in seen_graphs.items():
        for past in range(t + 1):
            u, v = signature_edge(past)
            assert (u, v) in pairs and (v, u) in pairs, f"t={t} lost snapshot {past}"
        for future in range(t + 1, num_snaps):
            u, v = signature_edge(future)
            assert (u, v) not in pairs and (v, u) not in pairs, (
                f"t={t} served a basis containing snapshot {future}'s edge"
            )
        assert order.index(("spectral", t)) < order.index(("eval", t))


def test_reported_eval_scores_only_the_test_split(config):
    # The reported metric must come from snapshot t+1's held-out positives. If it
    # ever fell back to snap.edge_index, it would score the clients' own training
    # edges and every MRR in results.md would be inflated.
    tiny_run_config(config, num_subgraphs=2)
    snaps = make_signature_snapshots(num_snaps=3)
    server = build_server(snaps, 2)

    eval_batches = []
    real_attach = ds_mod._attach_future_link_pred_labels

    def spy_attach(snap_today, snap_tomorrow, pos_edge_index=None):
        eval_batches.append(pos_edge_index.detach().cpu().clone())
        return real_attach(snap_today, snap_tomorrow, pos_edge_index)

    ds_mod._attach_future_link_pred_labels = spy_attach
    try:
        server.joint_train_w(FL=True)
    finally:
        ds_mod._attach_future_link_pred_labels = real_attach

    assert len(eval_batches) == 2
    for t, pos in enumerate(eval_batches):
        tomorrow = server.global_snaps[t + 1]
        assert torch.equal(pos, tomorrow.pos_test.detach().cpu())
        scored = {(int(a), int(b)) for a, b in zip(pos[0], pos[1])}
        assert scored
        for split in ("pos_train", "pos_val"):
            held = getattr(tomorrow, split)
            fitted = {(int(a), int(b)) for a, b in zip(held[0], held[1])}
            assert not scored & fitted


def test_fedavg_weights_participants_by_node_count(config):
    # The paper's C>1 numbers are FedAvg means weighted by client node count.
    # A regression to uniform weights (or to including abstainers) would move
    # every federated cell without failing anything else.
    tiny_run_config(config, num_subgraphs=3)
    config["model"]["iterations"] = 1
    snaps = make_signature_snapshots(num_snaps=3)
    server = build_server(snaps, 3)

    aggregations = []
    real_sum_lod = ds_mod.sum_lod

    def spy_sum_lod(lods, coef=None):
        out = real_sum_lod(lods, coef)
        if lods and isinstance(lods[0], dict):
            aggregations.append(
                ([_clone_state(l) for l in lods], list(coef), _clone_state(out))
            )
        return out

    ds_mod.sum_lod = spy_sum_lod
    try:
        server.joint_train_w(FL=True)
    finally:
        ds_mod.sum_lod = real_sum_lod

    # iterations=1 -> exactly one aggregation per task pair, in snapshot order
    n_tasks = len(server.global_snaps) - 1
    assert len(aggregations) == n_tasks

    checked = []

    def check_weighted_mean(node, parts, coef):
        for key, val in node.items():
            others = [p[key] for p in parts]
            if isinstance(val, dict):
                check_weighted_mean(val, others, coef)
            elif torch.is_floating_point(val):
                expected = sum(w * p for w, p in zip(coef, others))
                assert torch.allclose(val, expected, atol=1e-6)
                checked.append(key)

    for t, (states, coef, aggregate) in enumerate(aggregations):
        participants = [cl for cl in server.clients if cl.can_train(t)]
        assert len(states) == len(coef) == len(participants)
        assert abs(sum(coef) - 1.0) < 1e-6
        # node-count weighting, not uniform
        assert coef == pytest.approx(server._coef(participants))
        total = sum(cl.num_nodes() for cl in participants)
        assert coef == pytest.approx([cl.num_nodes() / total for cl in participants])
        check_weighted_mean(aggregate, states, coef)
    assert checked, "no float tensor was compared; the walk found nothing"

    # the clients really do differ in size, so node-count != uniform here
    sizes = {cl.num_nodes() for cl in server.clients}
    assert len(sizes) > 1, "test graph gives equal-sized clients; weighting is untested"


def test_abstaining_clients_are_excluded_from_the_aggregate(config):
    # A client whose subgraph is too small to form a train/val split must not
    # contribute weight; including it would drag the global model toward a model
    # that was never fine-tuned.
    tiny_run_config(config, num_subgraphs=2)
    config["model"]["iterations"] = 1
    snaps = make_signature_snapshots(num_snaps=3)
    server = build_server(snaps, 2)

    aggregations = []
    real_sum_lod = ds_mod.sum_lod
    real_can_train = DynamicClient.can_train

    def spy_sum_lod(lods, coef=None):
        out = real_sum_lod(lods, coef)
        if lods and isinstance(lods[0], dict):
            aggregations.append(list(coef))
        return out

    # force client 0 to abstain everywhere
    def gated_can_train(self, t):
        return self.id != 0 and real_can_train(self, t)

    ds_mod.sum_lod = spy_sum_lod
    DynamicClient.can_train = gated_can_train
    try:
        server.joint_train_w(FL=True)
    finally:
        ds_mod.sum_lod = real_sum_lod
        DynamicClient.can_train = real_can_train

    assert aggregations
    for coef in aggregations:
        assert coef == [1.0]  # only the single participating client
