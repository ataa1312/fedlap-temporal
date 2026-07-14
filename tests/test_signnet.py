import math
import random
import copy
import numpy as np
import torch
import pytest
from torch_geometric.data import Data

from src import device
from src.utils.graph import Graph
from src.utils.graph_partitioning import partition_snapshots
from src.GNN.fed_dynamic_classifier import (
    DynamicSSignNet,
    make_sgraph,
    make_fed_dynamic_classifier,
    FED_DYNAMIC_CLASSIFIERS,
)
from src.utils.utils import sum_lod
from config.config import get_default_config
from config.assertions import assert_cfg
from src.dynamic_server import DynamicServer
from src.dynamic_client import DynamicClient

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def make_fgraph(N=30, seed=42):
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.randint(0, N, (2, 40), generator=g, device=device)
    edge_attr = torch.randn(40, 1, generator=g, device=device)
    graph = Graph(
        x=torch.ones(N, 1, device=device),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_ids=torch.arange(N, device=device)
    )
    graph.edge_label_index = torch.randint(0, N, (2, 20), generator=g, device=device)
    graph.edge_label = torch.randint(0, 2, (20,), generator=g, device=device).float()
    return graph

def make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42):
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
        x = torch.ones(N, 1)
        snap = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=N)
        snap.node_ids = torch.arange(N)
        snaps.append(snap)
    return snaps

def test_sign_invariance_exact(config):
    config["gnn"]["dims"] = [16, 16]
    config["spectral"]["signnet_phi_dims"] = [64, 64]
    config["spectral"]["signnet_rho_dims"] = [64]
    config["spectral"]["output_bn"] = False  # Disable to avoid trivial zero outputs at init
    
    k = 8
    F = 32
    N = 30
    SFV = torch.normal(0, 0.05, size=(k, F), requires_grad=True, device=device)
    graph = make_sgraph(SFV)
    out_dim = config["gnn"]["dims"][-1]
    
    sm = DynamicSSignNet(graph, out_dim=out_dim)
    sm.eval()
    
    Q = torch.randn(N, k, device=device)
    D = torch.randn(k, device=device)
    sm.set_QD(Q, D)
    
    S0 = sm.get_embeddings()
    
    # Flip an arbitrary subset of eigenvector columns
    signs = torch.ones(k, device=device)
    signs[::3] = -1.0
    
    sm.set_QD(Q * signs, D)
    S1 = sm.get_embeddings()
    
    # Assert exact sign invariance in eval mode
    assert torch.allclose(S0, S1, atol=1e-6)
    
    # Negative test: random orthogonal mixing (rotation) of columns generally changes S
    while True:
        rand_mat = torch.randn(k, k, device=device)
        ortho_mat, _ = torch.linalg.qr(rand_mat)
        if not torch.allclose(ortho_mat.abs(), torch.eye(k, device=device), atol=1e-2):
            break
    sm.set_QD(Q @ ortho_mat, D)
    S_rot = sm.get_embeddings()
    assert not torch.allclose(S0, S_rot, atol=1e-3)

def test_output_shape_and_fusion(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "f+s"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["spectral"]["signnet_phi_dims"] = [64, 64]
    config["spectral"]["signnet_rho_dims"] = [64]
    config["spectral"]["output_bn"] = True
    config["spectral"]["spectral_len"] = 8
    
    k = 8
    F = 32
    N = 30
    SFV = torch.normal(0, 0.05, size=(k, F), requires_grad=True, device=device)
    graph = make_sgraph(SFV)
    out_dim = config["gnn"]["dims"][-1]
    
    # Basic smodel embedding output shape
    sm = DynamicSSignNet(graph, out_dim=out_dim)
    Q = torch.randn(N, k, device=device)
    D = torch.randn(k, device=device)
    sm.set_QD(Q, D)
    S = sm.get_embeddings()
    assert S.shape == (N, out_dim)
    
    # fusion = "add"
    config["model"]["fusion"] = "add"
    fgraph = make_fgraph(N=N, seed=42)
    classifier_add = make_fed_dynamic_classifier("SignNet", fgraph, SFV)
    assert classifier_add.fusion == "add"
    classifier_add.set_QD(Q, D)
    z_add, new_hs_add = classifier_add.encode()
    assert z_add.shape == (N, out_dim)
    
    # fusion = "concat"
    config["model"]["fusion"] = "concat"
    classifier_concat = make_fed_dynamic_classifier("SignNet", fgraph, SFV)
    assert classifier_concat.fusion == "concat"
    classifier_concat.set_QD(Q, D)
    z_concat, new_hs_concat = classifier_concat.encode()
    assert z_concat.shape == (N, out_dim * 2)

def test_output_bn_zero_init(config):
    config["gnn"]["dims"] = [16, 16]
    config["spectral"]["signnet_phi_dims"] = [64, 64]
    config["spectral"]["signnet_rho_dims"] = [64]
    
    k = 8
    F = 32
    N = 30
    SFV = torch.normal(0, 0.05, size=(k, F), requires_grad=True, device=device)
    graph = make_sgraph(SFV)
    out_dim = config["gnn"]["dims"][-1]
    
    # 1. With output_bn = True, S is all-zeros at init
    config["spectral"]["output_bn"] = True
    sm_bn = DynamicSSignNet(graph, out_dim=out_dim)
    Q = torch.randn(N, k, device=device)
    D = torch.randn(k, device=device)
    sm_bn.set_QD(Q, D)
    S_bn = sm_bn.get_embeddings()
    assert S_bn.abs().max() == 0.0
    
    # Backprop through BN model with 0-weight initialization gates phi and rho grads.
    # So phi and rho params will have grads = 0 (or None).
    # But BN parameters (weight/bias) will have non-zero gradients.
    sm_bn.train()
    S_bn = sm_bn.get_embeddings()
    loss = S_bn.sum()
    loss.backward()
    
    # verify BN parameters have gradients
    assert sm_bn.bn.weight.grad is not None
    assert not torch.allclose(sm_bn.bn.weight.grad, torch.zeros_like(sm_bn.bn.weight.grad))
    assert sm_bn.bn.bias.grad is not None
    assert not torch.allclose(sm_bn.bn.bias.grad, torch.zeros_like(sm_bn.bn.bias.grad))
    
    # verify phi/rho parameter gradients are 0 (or None)
    for p in sm_bn.phi.parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
    for p in sm_bn.rho.parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
        
    # 2. With output_bn = False, after one backward, phi/rho params have non-zero grads
    config["spectral"]["output_bn"] = False
    sm_nobn = DynamicSSignNet(graph, out_dim=out_dim)
    sm_nobn.set_QD(Q, D)
    sm_nobn.train()
    S_nobn = sm_nobn.get_embeddings()
    # Should generally not be zero
    assert S_nobn.abs().max() > 0.0
    
    loss_nobn = S_nobn.sum()
    loss_nobn.backward()
    
    # verify we have non-zero gradients
    phi_has_grad = any(p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad)) for p in sm_nobn.phi.parameters())
    rho_has_grad = any(p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad)) for p in sm_nobn.rho.parameters())
    assert phi_has_grad
    assert rho_has_grad

def test_fedavg_state_dict_roundtrip(config):
    config["gnn"]["dims"] = [16, 16]
    config["spectral"]["signnet_phi_dims"] = [64, 64]
    config["spectral"]["signnet_rho_dims"] = [64]
    config["spectral"]["output_bn"] = True
    
    k = 8
    F = 32
    SFV = torch.normal(0, 0.05, size=(k, F), requires_grad=True, device=device)
    graph = make_sgraph(SFV)
    out_dim = config["gnn"]["dims"][-1]
    
    sm_a = DynamicSSignNet(graph, out_dim=out_dim)
    sm_b = DynamicSSignNet(graph, out_dim=out_dim)
    sm_c = DynamicSSignNet(graph, out_dim=out_dim)
    
    # Verify state dict keys
    sd_a = sm_a.state_dict()
    assert set(sd_a.keys()) == {"phi", "rho", "bn"}
    
    # Load roundtrip
    sm_b.load_state_dict(sd_a)
    sd_b = sm_b.state_dict()
    
    def assert_dict_close(d1, d2):
        for k in d1.keys():
            if isinstance(d1[k], dict):
                assert_dict_close(d1[k], d2[k])
            else:
                assert torch.allclose(d1[k], d2[k], atol=1e-6)
                
    assert_dict_close(sd_a, sd_b)
    
    # Build two models with different weights by training or resetting parameters
    seed_all(1)
    sm_a.reset_parameters()
    seed_all(2)
    sm_b.reset_parameters()
    
    sd_a = sm_a.state_dict()
    sd_b = sm_b.state_dict()
    
    avg = sum_lod([sd_a, sd_b], [0.5, 0.5])
    sm_c.load_state_dict(avg)
    sd_c = sm_c.state_dict()
    
    def assert_nested_mean(dict_a, dict_b, dict_c):
        for k in dict_a.keys():
            if isinstance(dict_a[k], dict):
                assert_nested_mean(dict_a[k], dict_b[k], dict_c[k])
            else:
                expected = 0.5 * (dict_a[k] + dict_b[k])
                assert torch.allclose(dict_c[k], expected, atol=1e-6)
                
    assert_nested_mean(sd_a, sd_b, sd_c)

def test_parameters_and_grad_protocol(config):
    config["gnn"]["dims"] = [16, 16]
    config["spectral"]["signnet_phi_dims"] = [64, 64]
    config["spectral"]["signnet_rho_dims"] = [64]
    config["spectral"]["output_bn"] = True
    
    k = 8
    F = 32
    SFV = torch.normal(0, 0.05, size=(k, F), requires_grad=True, device=device)
    graph = make_sgraph(SFV)
    out_dim = config["gnn"]["dims"][-1]
    
    sm = DynamicSSignNet(graph, out_dim=out_dim)
    
    # Verify parameters count
    # 12 parameters with default/specified dims:
    # phi has 4 tensors (2 Linear layers: weight+bias each)
    # rho has 6 tensors (LayerNorm: weight+bias + 2 Linear layers: weight+bias each)
    # bn has 2 tensors (BatchNorm: weight+bias)
    # Total = 12
    params = list(sm.parameters())
    assert len(params) == 12
    
    # Assert SFV leaf graph.x is not among parameters
    assert all(p is not graph.x for p in params)
    
    # Verify grad protocols
    assert sm.get_grads(just_SFV=True) == {}
    
    # Setup dummy forward and backward to get grads
    N = 30
    Q = torch.randn(N, k, device=device)
    D = torch.randn(k, device=device)
    
    # Set output_bn=False just to get gradients on all layers easily, or just train BN.
    config["spectral"]["output_bn"] = False
    sm_nobn = DynamicSSignNet(graph, out_dim=out_dim)
    sm_nobn.set_QD(Q, D)
    sm_nobn.train()
    S = sm_nobn.get_embeddings()
    loss = S.sum()
    loss.backward()
    
    grads = sm_nobn.get_grads()
    assert set(grads.keys()) == {"phi", "rho", "bn"}
    assert grads["phi"] is not None
    assert grads["rho"] is not None
    
    # Test set_grads without error
    sm_nobn2 = DynamicSSignNet(graph, out_dim=out_dim)
    sm_nobn2.set_grads(grads)
    
    # Assert gradients matched
    for p, g in zip(sm_nobn2.phi.parameters(), grads["phi"]):
        if g is not None:
            assert torch.allclose(p.grad, g)
    for p, g in zip(sm_nobn2.rho.parameters(), grads["rho"]):
        if g is not None:
            assert torch.allclose(p.grad, g)

def test_dispatch_and_config(config):
    # 1. Dispatch
    k = 8
    F = 32
    N = 30
    SFV = torch.normal(0, 0.05, size=(k, F), requires_grad=True, device=device)
    fgraph = make_fgraph(N=N, seed=42)
    
    assert "SignNet" in FED_DYNAMIC_CLASSIFIERS
    
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "f+s"
    config["model"]["edge_decoding"] = "concat"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["spectral"]["signnet_phi_dims"] = [16, 16]
    config["spectral"]["signnet_rho_dims"] = [16]
    config["spectral"]["output_bn"] = True
    config["spectral"]["update_mode"] = "recompute"
    config["spectral"]["spectral_len"] = k
    config["model"]["smodel_type"] = "SignNet"
    
    classifier = make_fed_dynamic_classifier("SignNet", fgraph, SFV)
    from src.GNN.fed_dynamic_classifier import FedDynamicSignNetClassifier
    assert isinstance(classifier, FedDynamicSignNetClassifier)
    assert isinstance(classifier.smodel, DynamicSSignNet)
    
    # 2. Config exposes
    cfg = get_default_config()
    assert "signnet_phi_dims" in cfg["spectral"]
    assert "signnet_rho_dims" in cfg["spectral"]
    
    # 3. Validation: assert_cfg
    config["model"]["smodel_type"] = "SignNet"
    config["model"]["data_type"] = "f+s"
    config["spectral"]["update_mode"] = "recompute"
    config["spectral"]["spectral_len"] = 8
    
    # This should not raise any ValueError
    assert_cfg(config)
    
    # When spectral.spectral_len <= 0, it should raise ValueError
    config["spectral"]["spectral_len"] = 0
    with pytest.raises(ValueError) as exc:
        assert_cfg(config)
    assert "spectral_len" in str(exc.value)

def test_sfv_leaf_frozen(config):
    k = 8
    F = 32
    SFV = torch.normal(0, 0.05, size=(k, F), requires_grad=True, device=device)
    graph = make_sgraph(SFV)
    assert graph.x.requires_grad is True
    
    out_dim = 16
    sm = DynamicSSignNet(graph, out_dim=out_dim)
    assert graph.x.requires_grad is False

def test_server_spectral_step_signnet(config):
    config["dataset"]["task"] = "link_pred"
    config["dataset"]["edge_dim"] = 1
    config["dataset"]["node_encoder"] = False
    config["dataset"]["edge_encoder"] = False
    config["model"]["data_type"] = "f+s"
    config["model"]["smodel_type"] = "SignNet"
    config["model"]["edge_decoding"] = "concat"
    config["model"]["loss_fun"] = "bce_with_logits"
    config["gnn"]["dims"] = [16, 16]
    config["gnn"]["dims_pre_mp"] = []
    config["gnn"]["dims_post_mp"] = []
    config["gnn"]["embed_update_method"] = "gru"
    config["gnn"]["l2norm"] = False
    config["spectral"]["spectral_len"] = 4
    config["spectral"]["update_mode"] = "recompute"
    config["structure_model"]["structure_type"] = "spectral"
    config["structure_model"]["num_structural_features"] = 4

    global_snaps = make_toy_snapshots(N=8, W=1, num_snaps=4, seed=42)
    client_snaps = partition_snapshots(global_snaps, 2)
    
    server = DynamicServer(global_snaps)
    for snaps in client_snaps:
        server.add_client(snaps)
        
    k = 4
    F = 16
    SFV = torch.normal(0, 0.05, size=(k, F), requires_grad=True, device=device)
    
    server.initialize_FL(SFV=SFV)
    
    # Verify that clients and server have a DynamicSSignNet smodel
    assert isinstance(server.classifier.smodel, DynamicSSignNet)
    for cl in server.clients:
        assert isinstance(cl.classifier.smodel, DynamicSSignNet)
        
    # Run _spectral_step for snapshot t=0
    server._spectral_step(0, "SignNet")
    
    # Assert Q and D are set on server classifier and clients
    assert server.classifier.smodel.Q is not None
    assert server.classifier.smodel.D is not None
    assert server.classifier.smodel.Q.shape == (8, 4)
    
    # Clients should have sliced version of Q corresponding to their node_ids
    for cl in server.clients:
        nid = cl.snaps[0].node_ids
        expected_shape = (len(nid), 4)
        assert cl.classifier.smodel.Q is not None
        assert cl.classifier.smodel.Q.shape == expected_shape
        assert cl.classifier.smodel.D is not None
        
    # Verify that the forward pass works
    cl = server.clients[0]
    z, nid = cl.encode(0)
    assert z.shape == (len(nid), 16)

def test_canonicalize_sign_protocol(config):
    # 1. Verify calc_eignvalues(canonicalize_sign=False) behaves differently from True (or default)
    g = Graph(
        x=torch.ones(10, 1),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]]),
        node_ids=torch.arange(10)
    )
    config["spectral"]["matrix"] = "lap"
    config["spectral"]["L_type"] = "rw"
    config["spectral"]["decompose"] = "eigh"
    
    # We call calc_eignvalues with canonicalize_sign=False and canonicalize_sign=True
    D_raw, U_raw, _ = g.calc_eignvalues(canonicalize_sign=False)
    D_canon, U_canon, _ = g.calc_eignvalues(canonicalize_sign=True)
    
    # Assert eigenvalues are identical
    assert torch.allclose(D_raw, D_canon, atol=1e-6)
    
    # The columns of U_raw should match U_canon up to a sign flip (i.e. absolute values are identical)
    assert torch.allclose(U_raw.abs(), U_canon.abs(), atol=1e-6)
    
    # Verify that get_spectral_features passes False iff smodel_type == "SignNet".
    called_args = []
    original_calc_eignvalues = g.calc_eignvalues
    
    def mock_calc_eignvalues(*args, **kwargs):
        called_args.append((args, kwargs))
        D, U, V = original_calc_eignvalues(*args, **kwargs)
        # return dummy tensor instead of None to satisfy assertion in get_spectral_features
        return D, U, (V if V is not None else torch.zeros_like(U))
        
    g.calc_eignvalues = mock_calc_eignvalues
    
    # Set up DynamicServer
    server = DynamicServer(make_toy_snapshots(N=8, W=1, num_snaps=2, seed=42))
    
    # Test for "SpectralLaplace"
    called_args.clear()
    from src.dynamic_server import _to_cpu_sf
    prev_spectrals = _to_cpu_sf(None, None, None)
    first_spectral = _to_cpu_sf(None, None, None)
    server.get_spectral_features(
        g, "SpectralLaplace", ss_idx=0, spectral_len=4,
        spectral_update_mode="recompute", prev_spectrals=prev_spectrals,
        first_spectral=first_spectral
    )
    assert len(called_args) == 1
    assert called_args[0][1].get("canonicalize_sign") is True
    
    # Test for "SignNet"
    called_args.clear()
    server.get_spectral_features(
        g, "SignNet", ss_idx=0, spectral_len=4,
        spectral_update_mode="recompute", prev_spectrals=prev_spectrals,
        first_spectral=first_spectral
    )
    assert len(called_args) == 1
    assert called_args[0][1].get("canonicalize_sign") is False
